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
from qpsim.webui.schemas import (
    EnergyGrid,
    SetupEnvelope,
    Spatial1DSetup,
    SteadyState0DSetup,
)
from qpsim.webui.store import Workspace, json_sanitize


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


def _manifest(run_id: str, *, status: str = "done") -> dict[str, object]:
    return {
        "id": run_id,
        "name": "test run",
        "mode": "steady_state_0d",
        "status": status,
        "created": "2026-07-13T00:00:00",
        "setup": {},
        "summary": {},
        "notes": [],
        "error": None,
        "elapsed_s": None,
    }


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
        stored = _manifest(run_id)
        stored["summary"] = {"Q_i": math.inf}
        ws.write_manifest(run_id, stored)
        manifest = ws.read_manifest(run_id)
        # Strict serializers (FastAPI's JSONResponse uses allow_nan=False)
        # must be able to re-serialize what the store persists.
        import json

        json.dumps(manifest, allow_nan=False)
        assert manifest["summary"]["Q_i"] is None


class TestSchemaNumerics:
    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_all_setup_floats_reject_non_finite_values(self, value: float) -> None:
        with pytest.raises(ValueError, match="finite number"):
            SteadyState0DSetup.model_validate({"T_bath": value})

    @pytest.mark.parametrize(
        ("minimum", "maximum"), [(2.0, 2.0), (3.0, 2.0)]
    )
    def test_energy_grid_requires_ordered_bounds(
        self, minimum: float, maximum: float
    ) -> None:
        with pytest.raises(ValueError, match="max_factor must be greater"):
            EnergyGrid(min_factor=minimum, max_factor=maximum)

    def test_energy_grid_allows_subgap_support(self) -> None:
        grid = EnergyGrid(min_factor=0.5, max_factor=3.0)
        assert grid.min_factor == 0.5

    def test_zero_interface_conductance_is_valid(self) -> None:
        setup = Spatial1DSetup.model_validate(
            {
                "gap_profile": {
                    "kind": "step",
                    "interface_G_N": 0.0,
                }
            }
        )
        assert setup.gap_profile.interface_G_N == 0.0


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

    def test_structurally_invalid_json_manifest_is_unreadable(
        self, tmp_path: Path,
    ) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(run_id, {})

        with pytest.raises(ValueError, match="manifest field"):
            ws.read_manifest(run_id)
        runs = ws.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == run_id
        assert runs[0]["status"] == "unreadable"

    def test_busy_artifact_does_not_delete_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.store as store_module

        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(run_id, _manifest(run_id))
        ws.write_arrays(run_id, {"a": np.arange(3.0)})
        real_unlink = store_module._unlink_with_retry

        def busy_result(path: Path) -> None:
            if path.name == "result.npz":
                raise PermissionError("synthetic sharing violation")
            real_unlink(path)

        monkeypatch.setattr(store_module, "_unlink_with_retry", busy_result)
        with pytest.raises(PermissionError):
            ws.delete_run(run_id)
        assert (ws.run_dir(run_id) / "manifest.json").is_file()
        assert ws.list_runs()[0]["id"] == run_id


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

    def test_terminal_job_is_retired_after_manifest_is_durable(
        self, tmp_path: Path,
    ) -> None:
        from qpsim.webui.runner import JobRunner, JobState

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        job = JobState(run_id="terminal", status="done")
        runner._jobs[job.run_id] = job
        runner._write_manifest_or_stash(
            job, _manifest(job.run_id),
        )
        assert runner.live_state(job.run_id) is None
        assert ws.read_manifest(job.run_id)["status"] == "done"
        runner.shutdown()

    def test_array_persistence_failure_becomes_terminal_failed_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.runner as runner_module
        from qpsim.webui.execute import RunPayload
        from qpsim.webui.runner import JobRunner, JobState

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        run_id = ws.new_run_id()
        job = JobState(run_id=run_id)
        runner._jobs[run_id] = job
        envelope = SetupEnvelope(name="persistence-failure", setup=SteadyState0DSetup())
        manifest = _manifest(run_id, status="queued")

        monkeypatch.setattr(
            runner_module,
            "execute_setup",
            lambda *_args, **_kwargs: RunPayload(arrays={"f": np.ones(2)}),
        )

        def fail_write_arrays(*_args: object, **_kwargs: object) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(ws, "write_arrays", fail_write_arrays)

        runner._run(job, envelope, manifest)

        persisted = ws.read_manifest(run_id)
        assert persisted["status"] == "failed"
        assert "disk full" in persisted["error"]
        assert runner.live_state(run_id) is None
        runner.shutdown()

    def test_stale_running_retry_cannot_overwrite_done_manifest(
        self, tmp_path: Path,
    ) -> None:
        from qpsim.webui.runner import JobRunner, JobState

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        run_id = ws.new_run_id()
        job = JobState(run_id=run_id, status="done")
        done = _manifest(run_id)
        done["summary"] = {"answer": 42}
        ws.write_manifest(run_id, done)

        # Reproduce an overlay thread that captured the old pending active
        # snapshot just before the worker durably wrote the terminal state.
        stale = _manifest(run_id, status="running")
        job.pending_manifest = dict(stale)
        runner._write_manifest_or_stash(job, stale)

        persisted = ws.read_manifest(run_id)
        assert persisted["status"] == "done"
        assert persisted["summary"] == {"answer": 42}
        assert job.pending_manifest is None
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
        assert "delta_eq_ueV" in payload.summary
        assert "delta_suppression_ueV" not in payload.summary
        assert "rel_gap_suppression" not in payload.summary
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
        assert params.rho_F == pytest.approx(1.74e28)

    def test_existing_ev_unit_setup_needs_no_migration(self) -> None:
        envelope = SetupEnvelope.model_validate(
            {
                "name": "existing eV-unit setup",
                "setup": {
                    "mode": "steady_state_0d",
                    "material": {"rho_F": 1.74e28},
                },
            }
        )
        assert envelope.setup.material.rho_F == pytest.approx(1.74e28)

    @pytest.mark.parametrize("rho_F", [float("nan"), float("inf")])
    def test_material_params_reject_non_finite_rho_F(self, rho_F: float) -> None:
        from qpsim.webui.schemas import MaterialParams

        with pytest.raises(ValueError, match="rho_F"):
            MaterialParams.model_validate({"rho_F": rho_F})

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
