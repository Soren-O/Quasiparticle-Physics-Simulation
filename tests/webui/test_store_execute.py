"""Workspace persistence and the mode executors on tiny grids."""

from __future__ import annotations

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

    def test_run_manifest_and_arrays_round_trip(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(run_id, {"id": run_id, "status": "done"})
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
        setup.probe.enabled = False
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.arrays["f_final"].shape == (12, 7)
        assert payload.arrays["xqp_profile"].shape == (7,)
        # Left-end injection: the source end carries more QPs.
        profile = payload.arrays["xqp_profile"]
        assert profile[0] > profile[-1]
        assert payload.summary["n_steps"] == 5


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
