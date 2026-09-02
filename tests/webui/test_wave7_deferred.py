"""Wave 7: the deferred items, each bound to a number.

* A static injection under strategy='steady_state' is folded into the
  solver's external flux. The measured answer moves with it, and agrees with
  the time-marched steady state of the same cell -- two solvers, one number.
* The quasiparticle budget (number, energy, mean energy per quasiparticle)
  comes from two moments of one exact quadrature: transport alone conserves
  both to roundoff; a source raises both. The phonon side is deliberately
  not reported, and the run says why.
* The phonon temperature field is gated by its own fit residual: a thermal
  spectrum gets the bath, a hotter Bose-Einstein gets its temperature, and a
  spectrum with no single temperature gets NaN and a note.
* The mode benchmarks store a field comparison, and its difference is the
  size the rate error predicts.
* Live frames handed out during a run are the frames the run records.
"""

from __future__ import annotations

import pathlib
import threading
import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="webui tests need the qpsim[ui] extra")

from qpsim.webui import benchmarks, verdicts
from qpsim.webui.builders import validate_setup
from qpsim.webui.execute import LiveFrame, execute_setup, run_kinetics
from qpsim.webui.schemas import KineticsSetup, PhononInitialCondition
from qpsim.webui.terms import term_status

_STATIC = pathlib.Path(__file__).resolve().parents[2] / "qpsim" / "webui" / "static"


def _noop(*_a: Any, **_k: Any) -> None:
    return None


def _never() -> bool:
    return False


def _cell(**over: Any) -> KineticsSetup:
    setup = KineticsSetup()
    setup.geometry.rows = setup.geometry.cols = 1
    setup.grid.num_bins = 24
    setup.T_bath = 0.2
    for path, value in over.items():
        node: Any = setup
        parts = path.split("__")
        for part in parts[:-1]:
            node = getattr(node, part)
        setattr(node, parts[-1], value)
    return setup


class TestSteadyStateCarriesAStaticInjection:
    def test_it_is_accepted_and_the_terms_panel_calls_it_on(self) -> None:
        setup = _cell(strategy="steady_state", injection__enabled=True, injection__rate_per_ns=2e-4)
        assert validate_setup(setup).ok
        assert term_status(setup)["src"].state == "on"
        # What a root find still cannot take is still refused.
        setup.initial.kind = "excess"
        setup.initial.amplitude = 1e-3
        setup.initial.energy.kind = "thermal"
        setup.initial.energy.T_eff = 0.5
        assert not validate_setup(setup).ok

    def test_the_injected_steady_state_moves_and_matches_the_time_march(self) -> None:
        """Two solvers, one number: the root find with the flux folded in must
        agree with the time march of the same cell run to its steady state."""
        quiet = execute_setup(_cell(strategy="steady_state"), _noop, _never)
        rooted = execute_setup(
            _cell(strategy="steady_state", injection__enabled=True, injection__rate_per_ns=2e-4),
            _noop, _never,
        )
        assert rooted.summary["x_qp"] > 3.0 * quiet.summary["x_qp"]
        assert rooted.summary["injection_rate_per_ns"] == 2e-4
        marched = execute_setup(
            _cell(
                strategy="time_march", injection__enabled=True, injection__rate_per_ns=2e-4,
                dt=2.0, max_time=4000.0, stop_tol=1e-9,
            ),
            _noop, _never,
        )
        assert marched.summary["converged"], marched.summary
        assert marched.summary["x_qp_mean"] == pytest.approx(rooted.summary["x_qp"], rel=2e-3)

    def test_an_unaccelerated_picard_with_dynamic_phonons_is_refused_up_front(self) -> None:
        setup = _cell(
            strategy="steady_state", injection__enabled=True, injection__rate_per_ns=2e-4,
            phonons__mode="dynamic_escape", solver__anderson_depth=0,
        )
        report = validate_setup(setup)
        assert not report.ok
        assert any("anderson_depth" in e for e in report.errors)


class TestTheQuasiparticleBudget:
    def test_transport_alone_conserves_number_and_energy(self) -> None:
        """The diffusion catalogue case: every bin is conserved bin by bin,
        so both moments are constant to roundoff across every frame."""
        case = next(c for c in verdicts.catalogue_cases() if c.id == "diff-benchmark")
        setup = verdicts.build_case_setup(case)
        assert isinstance(setup, KineticsSetup)
        setup.grid.num_bins = 24
        setup.geometry.rows, setup.geometry.cols = 4, 8
        payload = run_kinetics(setup, _noop, _never)
        number = payload.arrays["obs_x_qp_total"]
        energy = payload.arrays["obs_E_qp_total"]
        assert number.size >= 3
        assert np.ptp(number) / number[0] < 1e-9
        assert np.ptp(energy) / energy[0] < 1e-9
        mean_e = payload.arrays["obs_E_qp_mean"]
        np.testing.assert_allclose(mean_e, energy / number)
        assert payload.summary["E_qp_mean_ueV"] == pytest.approx(float(mean_e[-1]))

    def test_a_source_raises_both_and_the_mean_energy_sits_on_the_line(self) -> None:
        """Inject a Gaussian line at 2 Δ into an otherwise empty cell and the
        mean energy per quasiparticle heads for the line, from above the
        thermal value it starts at."""
        setup = _cell(
            strategy="time_march", injection__enabled=True, injection__rate_per_ns=1e-2,
            collisions__scattering=False, collisions__recombination=False,
            dt=0.05, max_time=1.0, snapshot_interval=0.25, stop_tol=0.0,
        )
        payload = run_kinetics(setup, _noop, _never)
        number = payload.arrays["obs_x_qp_total"]
        energy = payload.arrays["obs_E_qp_total"]
        assert np.all(np.diff(number) > 0) and np.all(np.diff(energy) > 0)
        mean_over_gap = payload.arrays["obs_E_qp_mean"] / setup.material.Delta_0
        assert mean_over_gap[0] < mean_over_gap[-1] < 2.0 * 1.05
        assert mean_over_gap[-1] > 1.5

    def test_the_phonon_side_is_refused_with_its_reason(self) -> None:
        setup = _cell(
            strategy="time_march", phonons__mode="dynamic_escape",
            dt=0.1, max_time=0.2, snapshot_interval=0.1, stop_tol=0.0,
        )
        payload = run_kinetics(setup, _noop, _never)
        assert any("mode density" in n for n in payload.notes)
        assert not any(k.startswith("obs_E_ph") for k in payload.arrays)

    def test_the_time_series_table_carries_the_budget(self) -> None:
        from qpsim.webui.plots import available_plots, render_csv

        setup = _cell(
            strategy="time_march", dt=0.1, max_time=0.2, snapshot_interval=0.1, stop_tol=0.0,
        )
        payload = run_kinetics(setup, _noop, _never)
        header = render_csv("kinetics", "time_series", payload.arrays, payload.summary).splitlines()[0]
        assert {"x_qp_total", "E_qp_total", "E_qp_mean_ueV"} <= set(header.split(","))
        assert "qp_energy_over_time" in available_plots("kinetics", set(payload.arrays))


class TestThePhononTemperatureField:
    @staticmethod
    def _run(seed: PhononInitialCondition | None, max_time: float = 0.2) -> Any:
        step = min(0.1, max_time / 2.0)
        setup = _cell(
            strategy="time_march", phonons__mode="dynamic_closed",
            geometry__cols=3, dt=step, max_time=max_time, snapshot_interval=step,
            stop_tol=0.0,
        )
        if seed is not None:
            setup.phonons.initial = seed
        return run_kinetics(setup, _noop, _never)

    def test_a_bath_spectrum_reads_the_bath(self) -> None:
        payload = self._run(None)
        t_eff = payload.arrays["phonon_T_eff"]
        assert t_eff.shape == (3,)
        np.testing.assert_allclose(t_eff, 0.2, rtol=1e-6)
        assert payload.summary["phonon_T_eff_cells_fitted"] == 3

    def test_a_hotter_bose_einstein_reads_its_temperature(self) -> None:
        """Read almost at t = 0, before pair breaking has reshaped the
        spectrum: a 0.6 K Bose-Einstein must read 0.6 K in every cell."""
        payload = self._run(
            PhononInitialCondition(kind="thermal_at", T_eff=0.6), max_time=0.01,
        )
        t_eff = payload.arrays["phonon_T_eff"]
        assert np.all(np.isfinite(t_eff)), payload.arrays["phonon_T_eff_residual"]
        np.testing.assert_allclose(t_eff, 0.6, rtol=0.05)

    def test_a_hot_seed_stops_being_thermal_once_pairs_break(self) -> None:
        """The same seed 0.2 ns later has been eaten above 2Δ by pair breaking
        against 0.2 K quasiparticles and is no longer one Bose-Einstein: the
        gate must say so rather than report a temperature."""
        payload = self._run(
            PhononInitialCondition(kind="thermal_at", T_eff=0.6), max_time=0.2,
        )
        residual = payload.arrays["phonon_T_eff_residual"]
        assert np.all(residual > 0.05), residual
        assert np.all(np.isnan(payload.arrays["phonon_T_eff"]))

    def test_a_spectrum_with_no_single_temperature_is_nan_and_says_so(self) -> None:
        bump = PhononInitialCondition(
            kind="expression",
            expression="n_bath * (1.0 + 50.0 * np.exp(-((omega - 450.0) / 25.0) ** 2))",
        )
        payload = self._run(bump)
        t_eff = payload.arrays["phonon_T_eff"]
        residual = payload.arrays["phonon_T_eff_residual"]
        assert np.all(np.isnan(t_eff)), (t_eff, residual)
        assert np.all(residual > 0.05)
        assert any("no single temperature" in n for n in payload.notes)


class TestTheFieldComparison:
    def test_the_rectangle_benchmark_stores_a_field_whose_difference_is_the_rate_error(self) -> None:
        case = next(c for c in verdicts.catalogue_cases() if c.id == "bc-rectangle-benchmark")
        setup = verdicts.build_case_setup(case)
        assert isinstance(setup, KineticsSetup)
        setup.grid.num_bins = 16
        payload = run_kinetics(setup, _noop, _never)
        notes = benchmarks.attach("bc-rectangle", setup, payload.arrays, payload.summary)
        assert notes == []
        sim = payload.arrays["bench_field_sim"]
        ana = payload.arrays["bench_field_analytic"]
        assert sim.shape == ana.shape == (payload.arrays["mask"].sum(),)
        # The rate is 2.7e-3 high over ~2.2 e-foldings, so the field sits
        # ~0.6 % below the closed form -- small, negative, and not zero.
        rel = float(np.max(np.abs(sim - ana)) / np.max(np.abs(ana)))
        assert 1e-4 < rel < 2e-2
        assert payload.summary["benchmark"]["field_label"].startswith("f at E =")

    def test_the_figure_and_table_are_offered_and_render(self) -> None:
        from qpsim.webui.plots import available_csvs, available_plots, render_csv, render_plot

        case = next(c for c in verdicts.catalogue_cases() if c.id == "bc-absorbing-benchmark")
        setup = verdicts.build_case_setup(case)
        assert isinstance(setup, KineticsSetup)
        setup.grid.num_bins = 16
        payload = run_kinetics(setup, _noop, _never)
        benchmarks.attach("bc-absorbing", setup, payload.arrays, payload.summary)
        names = set(payload.arrays)
        assert "analytic_field_comparison" in available_plots("kinetics", names)
        assert "analytic_field" in available_csvs("kinetics", names)
        png = render_plot("kinetics", "analytic_field_comparison", payload.arrays, payload.summary)
        assert png[:4] == b"\x89PNG"
        header = render_csv("kinetics", "analytic_field", payload.arrays, payload.summary).splitlines()[0]
        assert header.split(",")[-3:] == ["simulated", "analytic", "difference"]


class TestLiveFrames:
    def test_the_frames_handed_out_are_the_frames_recorded(self) -> None:
        setup = _cell(
            strategy="time_march", geometry__cols=4, injection__enabled=True,
            injection__rate_per_ns=1e-2, dt=0.1, max_time=0.4, snapshot_interval=0.1,
            stop_tol=0.0,
        )
        seen: list[LiveFrame] = []
        payload = execute_setup(setup, _noop, _never, on_frame=seen.append)
        assert len(seen) == payload.arrays["snap_t_ns"].size
        np.testing.assert_array_equal([f.t_ns for f in seen], payload.arrays["snap_t_ns"])
        for frame, recorded in zip(seen, payload.arrays["snap_xqp_profile"], strict=True):
            np.testing.assert_array_equal(frame.xqp_profile, recorded)
            assert frame.mask.shape == payload.arrays["mask"].shape
            assert frame.x_qp_convention == payload.summary["x_qp_convention"]

    def test_the_route_serves_the_worker_s_latest_frame_only_while_running(
        self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from qpsim.webui import runner as runner_module
        from qpsim.webui.server import create_app

        emitted = threading.Event()
        release = threading.Event()
        real = runner_module.execute_setup

        def slow_execute(setup: Any, progress: Any, is_cancelled: Any, *, on_frame: Any = None) -> Any:
            # Hand out one frame, then hold the run until the test has looked.
            payload = real(setup, progress, is_cancelled, on_frame=None)
            if on_frame is not None:
                on_frame(LiveFrame(
                    t_ns=0.5, xqp_profile=np.asarray(payload.arrays["xqp_profile"], dtype=float),
                    mask=np.asarray(payload.arrays["mask"], dtype=bool),
                    mesh_size_um=float(payload.summary["mesh_size_um"]),
                    x_qp_convention=str(payload.summary["x_qp_convention"]),
                ))
            emitted.set()
            release.wait(timeout=30.0)
            return payload

        monkeypatch.setattr(runner_module, "execute_setup", slow_execute)
        with TestClient(create_app(tmp_path)) as client:
            setup = _cell(strategy="time_march", geometry__cols=3, dt=0.1, max_time=0.2)
            run_id = client.post(
                "/api/runs", json={"name": "live", "setup": setup.model_dump(mode="json")},
            ).json()["id"]
            assert emitted.wait(timeout=60.0)
            manifest = client.get(f"/api/runs/{run_id}").json()
            assert manifest["status"] == "running"
            assert manifest["live_frame_t_ns"] == 0.5
            live = client.get(f"/api/runs/{run_id}/live.png")
            assert live.status_code == 200 and live.content[:4] == b"\x89PNG"
            release.set()
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                manifest = client.get(f"/api/runs/{run_id}").json()
                if manifest["status"] in ("done", "failed"):
                    break
                time.sleep(0.1)
            assert manifest["status"] == "done", manifest.get("error")
            assert "live_frame_t_ns" not in manifest
            assert client.get(f"/api/runs/{run_id}/live.png").status_code == 404

    def test_the_page_shows_the_live_frame(self) -> None:
        js = (_STATIC / "app.js").read_text(encoding="utf-8")
        assert "/live.png" in js and "live_frame_t_ns" in js
