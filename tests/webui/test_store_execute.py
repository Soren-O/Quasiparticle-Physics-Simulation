"""Workspace persistence and the mode executors on tiny grids."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from qpsim.webui.execute import RunCancelledError, execute_setup
from qpsim.webui.schemas import (
    M25JunctionSetup,
    SetupEnvelope,
    Spatial1DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)
from qpsim.webui.store import Workspace


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


class TestWorkspace:
    def test_setup_save_load_list_delete(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        envelope = SetupEnvelope(name="My probe sweep", setup=SteadyState0DSetup())
        slug = ws.save_setup(envelope)
        assert slug == "my-probe-sweep"
        assert [s["slug"] for s in ws.list_setups()] == [slug]
        loaded = ws.load_setup(slug)
        assert loaded.name == "My probe sweep"
        assert loaded.setup == envelope.setup
        ws.delete_setup(slug)
        assert ws.list_setups() == []

    def test_load_migrates_legacy_rho_f_units(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        legacy = SteadyState0DSetup().model_dump()
        legacy["material"]["rho_F"] = 1.74e22  # v1: µeV^-1 m^-3
        (setup_dir / "legacy.json").write_text(
            json.dumps({"name": "legacy", "setup": legacy}),
            encoding="utf-8",
        )

        loaded = ws.load_setup("legacy")

        assert loaded.setup.material.rho_F == pytest.approx(1.74e28)

    def test_load_keeps_versionless_ev_rho_f_unmigrated(self, tmp_path: Path) -> None:
        # The shipped webui wrote versionless setups already on the
        # eV^-1 m^-3 contract (Al 1.74e28); loading one must NOT apply
        # the x1e6 µeV migration.
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        shipped = SteadyState0DSetup().model_dump()
        shipped["material"]["rho_F"] = 1.74e28
        (setup_dir / "shipped.json").write_text(
            json.dumps({"name": "shipped", "setup": shipped}),
            encoding="utf-8",
        )

        loaded = ws.load_setup("shipped")

        assert loaded.setup.material.rho_F == pytest.approx(1.74e28)

    def test_saved_setup_stamps_current_schema_version(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        slug = ws.save_setup(
            SetupEnvelope(name="versioned", setup=SteadyState0DSetup())
        )
        stored = json.loads((ws.setups_dir / f"{slug}.json").read_text("utf-8"))
        assert stored["schema_version"] == 2

    @pytest.mark.parametrize("bad_version", [0, 2.5, 999])
    def test_load_rejects_unsupported_schema_version(
        self, tmp_path: Path, bad_version: object
    ) -> None:
        ws = Workspace(tmp_path)
        ws.setups_dir.mkdir(parents=True)
        (ws.setups_dir / "bad.json").write_text(
            json.dumps(
                {
                    "name": "bad",
                    "schema_version": bad_version,
                    "setup": SteadyState0DSetup().model_dump(),
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported setup schema_version"):
            ws.load_setup("bad")

    def test_run_manifest_and_arrays_round_trip(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(
            run_id,
            {
                "id": run_id,
                "name": "round trip",
                "mode": "steady_state_0d",
                "status": "done",
                "created": "2026-07-13T00:00:00",
                "setup": {},
                "summary": {},
                "notes": [],
            },
        )
        ws.write_arrays(run_id, {"a": np.arange(4.0)})
        assert ws.read_manifest(run_id)["status"] == "done"
        np.testing.assert_array_equal(ws.read_arrays(run_id)["a"], np.arange(4.0))
        assert [m["id"] for m in ws.list_runs()] == [run_id]
        ws.delete_run(run_id)
        assert ws.list_runs() == []


def _tiny_steady_state() -> SteadyState0DSetup:
    setup = SteadyState0DSetup()
    setup.grid.num_bins = 48
    return setup


class TestSteadyState0DExecutor:
    def test_undriven_thermal_solve_recovers_fermi_dirac(self) -> None:
        payload = execute_setup(_tiny_steady_state(), _noop_progress, _never)
        f = payload.arrays["f"]
        f_thermal = payload.arrays["f_thermal"]
        np.testing.assert_allclose(f, f_thermal, atol=1e-10)
        summary = payload.summary
        assert summary["x_qp"] == pytest.approx(summary["x_qp_thermal"], rel=1e-6)
        # Probe enabled by default with ω₀ = 22 μeV < Δ.
        assert summary["Q_i"] > 0.0
        assert summary["sigma2_over_sigmaN"] > 0.0
        assert "delta_eq_ueV" in summary

    def test_dynes_skips_mb_observables_with_note(self) -> None:
        setup = _tiny_steady_state()
        setup.material.dynes_gamma = 0.5
        payload = execute_setup(setup, _noop_progress, _never)
        assert "Q_i" not in payload.summary
        assert any("Dynes" in n for n in payload.notes)

    def test_negative_sigma1_is_reported_as_active_gain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qpsim.webui.execute as execute_mod

        monkeypatch.setattr(
            execute_mod,
            "compute_ac_conductivity",
            lambda *_args, **_kwargs: (-1.0, 2.0),
        )
        monkeypatch.setattr(
            execute_mod,
            "compute_quality_factor",
            lambda *_args, **_kwargs: -20.0,
        )

        payload = execute_setup(_tiny_steady_state(), _noop_progress, _never)

        assert payload.summary["Q_i"] == -20.0
        assert any("active microwave gain" in note for note in payload.notes)

    def test_cancel_before_solve(self) -> None:
        with pytest.raises(RunCancelledError):
            execute_setup(_tiny_steady_state(), _noop_progress, lambda: True)


class TestTransient0DExecutor:
    def test_short_relaxation_run(self) -> None:
        setup = Transient0DSetup()
        setup.grid.num_bins = 24
        setup.dt = 1.0
        setup.total_time = 5.0
        setup.snapshot_interval = 1.0
        setup.probe.enabled = False
        fractions: list[float] = []
        payload = execute_setup(
            setup, lambda fr, _m: fractions.append(fr), _never
        )
        assert payload.arrays["f_snapshots"].shape[0] == payload.arrays["t_ns"].size
        assert payload.summary["n_steps"] == 5
        assert fractions and fractions[-1] == 1.0

    def test_cancel_mid_run(self) -> None:
        setup = Transient0DSetup()
        setup.grid.num_bins = 24
        setup.dt = 1.0
        setup.total_time = 50.0
        setup.probe.enabled = False
        count = [0]

        def cancelled() -> bool:
            count[0] += 1
            return count[0] > 3

        with pytest.raises(RunCancelledError):
            execute_setup(setup, _noop_progress, cancelled)


class TestSpatial1DExecutor:
    def test_short_injection_run(self) -> None:
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 7
        setup.dt = 1.0
        setup.max_time = 5.0
        setup.stop_tol = 0.0
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.arrays["f_final"].shape == (12, 7)
        assert payload.arrays["xqp_profile"].shape == (7,)
        # Left-end injection: the source end carries more QPs.
        profile = payload.arrays["xqp_profile"]
        assert profile[0] > profile[-1]
        assert payload.summary["n_steps"] == 5

    def test_backend_warnings_are_exposed_in_run_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend

        original = T3Spatial1DBackend.run_until_steady_state

        def warning_wrapper(self: T3Spatial1DBackend, *args: object, **kwargs: object):
            warnings.warn(
                "spatial conservation diagnostic", RuntimeWarning, stacklevel=2
            )
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            T3Spatial1DBackend, "run_until_steady_state", warning_wrapper
        )
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 7
        setup.dt = 1.0
        setup.max_time = 1.0
        setup.stop_tol = 0.0

        payload = execute_setup(setup, _noop_progress, _never)

        assert "spatial conservation diagnostic" in payload.notes


class TestM25Executor:
    def test_two_point_sweep(self) -> None:
        setup = M25JunctionSetup()
        setup.T_start_mK = 20.0
        setup.T_stop_mK = 30.0
        setup.T_points = 2
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.arrays["T_mK"].size == 2
        assert payload.summary["points_total"] == 2
        assert payload.summary["points_converged"] >= 1
        finite = np.isfinite(payload.arrays["x_L"])
        assert np.all(payload.arrays["x_L"][finite] > 0.0)
        mu = payload.arrays["mu_L_over_Delta_L"][finite]
        assert np.all((mu > 0.0) & (mu <= 1.05))

    def test_chemical_potentials_paper_exact_inversion(self) -> None:
        # Default M25 config (ω_LR = 5 GHz — the paper's Fig 3b-like
        # large-asymmetry case) at 20/25/30 mK. Guards the services-
        # layer μ inversion (M25 SI Eqs. S2/S4/S5): the naive
        # μ = Δ + T·ln(x) inversion that used to live here dropped
        # the √(Δ/2πT) prefactor and the erf/erfc sub-band partition
        # and got the μ_R> vs μ_R< ordering backwards.
        setup = M25JunctionSetup()
        setup.T_start_mK = 20.0
        setup.T_stop_mK = 30.0
        setup.T_points = 3
        payload = execute_setup(setup, _noop_progress, _never)
        T_mK = payload.arrays["T_mK"]
        mu_L = payload.arrays["mu_L_over_Delta_L"]
        mu_Rgt = payload.arrays["mu_Rgt_over_Delta_L"]
        mu_Rlt = payload.arrays["mu_Rlt_over_Delta_L"]
        i20 = int(np.argmin(np.abs(T_mK - 20.0)))
        assert np.isfinite(mu_L[i20]), "20 mK point did not converge"
        # (a) Sub-band ordering: the erf/erfc partition puts μ_R>
        # above μ_R< (the naive inversion flipped this).
        finite = np.isfinite(mu_Rgt) & np.isfinite(mu_Rlt)
        assert finite[i20]
        assert np.all(mu_Rgt[finite] > mu_Rlt[finite])
        # (b) Published Fig 3 anchor: μ_L/Δ_L ≈ 0.87 at 20 mK (both
        # panels of the paper's Fig 3 sit at this value; the pinned
        # validation baselines give 0.872 (a) / 0.869 (b)).
        assert mu_L[i20] == pytest.approx(0.872, abs=0.02)
