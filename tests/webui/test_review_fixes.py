"""Regressions for the 2026-07-03 code-review fixes.

Each test pins one reviewed defect: non-finite observables poisoning
the manifest JSON, orphaned "running" manifests after a crash/restart,
run-creation warnings vanishing on view switch, unreadable manifests
disappearing from the listing, secondary diagnostics sinking a
successful solve, and the spatial mode carrying a dead probe config.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from qpsim.webui.execute import execute_setup
from qpsim.webui.schemas import SetupEnvelope, Spatial1DSetup, SteadyState0DSetup
from qpsim.webui.store import Workspace, json_sanitize


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


class TestNonFiniteSanitization:
    def test_json_sanitize_replaces_inf_and_nan(self) -> None:
        data = {
            "summary": {"Q_i": math.inf, "x_qp": 1e-9, "bad": math.nan},
            "notes": [{"nested": -math.inf}],
        }
        clean = json_sanitize(data)
        assert clean["summary"]["Q_i"] is None
        assert clean["summary"]["bad"] is None
        assert clean["summary"]["x_qp"] == 1e-9
        assert clean["notes"][0]["nested"] is None

    def test_manifest_with_inf_round_trips_strict_json(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(run_id, {"id": run_id, "summary": {"Q_i": math.inf}})
        manifest = ws.read_manifest(run_id)
        # Strict serializers (FastAPI's JSONResponse uses allow_nan=False)
        # must be able to re-serialize what the store persists.
        import json

        json.dumps(manifest, allow_nan=False)
        assert manifest["summary"]["Q_i"] is None


class TestStoreRobustness:
    def test_atomic_arrays_leave_no_tmp(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_arrays(run_id, {"a": np.arange(3.0)})
        files = {p.name for p in ws.run_dir(run_id).iterdir()}
        assert files == {"result.npz"}
        assert "a" in ws.array_names(run_id)

    def test_corrupt_manifest_listed_as_unreadable(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_dir = ws.runs_dir / "20990101-000000-abcdef"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text("{ truncated", encoding="utf-8")
        runs = ws.list_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "unreadable"
        assert runs[0]["id"] == "20990101-000000-abcdef"
        ws.delete_run(runs[0]["id"])
        assert ws.list_runs() == []


class TestRunnerRecovery:
    def test_orphaned_running_manifest_reports_interrupted(self, tmp_path: Path) -> None:
        from qpsim.webui.runner import JobRunner

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        # A manifest stuck on "running" with no live job — the shape a
        # crash or server restart leaves behind.
        overlaid = runner.overlay({"id": "ghost", "status": "running"})
        assert overlaid["status"] == "interrupted"
        assert "not recorded" in overlaid["error"]
        # Terminal manifests pass through untouched.
        assert runner.overlay({"id": "ghost", "status": "done"})["status"] == "done"
        runner.shutdown()


class TestDiagnosticsNeverSinkARun:
    def test_failing_gap_suppression_becomes_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import qpsim.webui.execute as execute_mod

        def boom(*_args: object, **_kwargs: object) -> object:
            raise ValueError("synthetic diagnostic failure")

        monkeypatch.setattr(execute_mod, "compute_gap_suppression", boom)
        setup = SteadyState0DSetup()
        setup.grid.num_bins = 48
        payload = execute_setup(setup, _noop_progress, _never)
        # The solve itself survived: arrays and primary observables present.
        assert "f" in payload.arrays
        assert payload.summary["x_qp"] > 0.0
        assert "delta_eq_ueV" not in payload.summary
        assert any("synthetic diagnostic failure" in n for n in payload.notes)


class TestSpatialProbeRemoved:
    def test_spatial_setup_has_no_probe(self) -> None:
        assert "probe" not in Spatial1DSetup.model_fields

    def test_old_setup_with_probe_is_rejected_loudly(self) -> None:
        with pytest.raises(ValueError, match="probe"):
            SetupEnvelope.model_validate(
                {
                    "name": "t",
                    "setup": {"mode": "spatial_1d", "probe": {"enabled": True}},
                }
            )


class TestMaterialDefaultsFromDatabase:
    def test_defaults_match_al_yaml(self) -> None:
        from qpsim.materials import load_material
        from qpsim.webui.schemas import MaterialParams

        al = load_material("Al")
        params = MaterialParams()
        assert params.Delta_0 == al.Delta_0
        assert params.T_c == al.T_c
        assert params.tau_0 == al.tau_0
        assert params.rho_F == al.rho_F  # was hand-copied as 0.0 pre-review
        assert params.rho_F > 0.0

    def test_default_steady_state_reports_density(self) -> None:
        setup = SteadyState0DSetup()
        setup.grid.num_bins = 48
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.summary["n_qp_per_m3"] > 0.0


class TestSharedConstants:
    def test_h_over_kb_matches_repo_literal(self) -> None:
        from qpsim.constants import H_OVER_KB_K_PER_HZ

        # The literal carried by the M25 validation modules and tests.
        assert pytest.approx(4.799243e-11, rel=1e-6) == H_OVER_KB_K_PER_HZ

    def test_builders_commensurate_tol_is_the_engine_constant(self) -> None:
        from qpsim.collisions.sub_gap_photon import COMMENSURATE_TOL as ENGINE_TOL
        from qpsim.webui import builders

        assert builders.COMMENSURATE_TOL is ENGINE_TOL
