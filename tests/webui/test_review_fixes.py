"""Regressions for the 2026-07-03 code-review fixes.

Each test pins one reviewed defect: non-finite observables poisoning
the manifest JSON, orphaned "running" manifests after a crash/restart,
run-creation warnings vanishing on view switch, unreadable manifests
disappearing from the listing, secondary diagnostics sinking a
successful solve, and the spatial mode carrying a dead probe config.
"""

from __future__ import annotations

import json
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from qpsim.webui.execute import execute_setup
from qpsim.webui.schemas import (
    EnergyGrid,
    SetupEnvelope,
    KineticsSetup,
)
from qpsim.webui.store import Workspace, json_sanitize


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


def test_ultracold_field_frames_still_render() -> None:
    """An all-underflow thermal field must render instead of returning 500.

    The frame figures normalise LINEARLY and guard the degenerate range
    explicitly, so an all-underflow field has well-defined colour limits.
    This pins that: a thermal field far below the gap must still produce a
    figure.
    """
    from qpsim.webui import plots

    tiny = 9.8596765e-305
    arrays = {
        "snap_f": np.full((2, 2, 3), tiny),
        "snap_t_ns": np.array([0.0, 1.0]),
        "E_bins": np.array([180.0, 200.0]),
        "mask": np.ones((1, 3), dtype=np.int8),
        "snap_xqp_profile": np.full((2, 3), tiny),
    }
    summary = {"gap_ueV": 180.0, "rows": 1, "cols": 3, "mesh_size_um": 4.0}

    for name in ("energy_resolved_map", "field_over_time"):
        png = plots.render_plot("kinetics", name, arrays, summary)
        assert png.startswith(b"\x89PNG\r\n\x1a\n"), name


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
            KineticsSetup.model_validate({"T_bath": value})

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
        setup = KineticsSetup.model_validate(
            {
                "gap_regions": {
                    "kind": "column_step",
                    "interface_G_N": 0.0,
                }
            }
        )
        # 0.0 is a BARRIER that blocks completely; None is "no interface at
        # all". Collapsing them would silently turn a perfect barrier into an
        # open face.
        assert setup.gap_regions.interface_G_N == 0.0


class TestStoreRobustness:
    def test_atomic_arrays_leave_no_tmp(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_arrays(run_id, {"a": np.arange(3.0)})
        files = {p.name for p in ws.run_dir(run_id).iterdir()}
        assert files == {"result.npz"}
        assert "a" in ws.array_names(run_id)

    def test_object_arrays_are_rejected_before_persistence(
        self, tmp_path: Path,
    ) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        with pytest.raises(ValueError, match="object dtype"):
            ws.write_arrays(
                run_id,
                {"unsafe": np.array([{"payload": 1}], dtype=object)},
            )
        assert not (ws.run_dir(run_id) / "result.npz").exists()

    def test_automatic_setup_slugs_do_not_overwrite_distinct_names(
        self, tmp_path: Path,
    ) -> None:
        ws = Workspace(tmp_path)
        first = SetupEnvelope(name="A+B", setup=KineticsSetup())
        second = SetupEnvelope(name="A B", setup=KineticsSetup())
        slug_first = ws.save_setup(first)
        slug_second = ws.save_setup(second)

        assert slug_first != slug_second
        assert ws.load_setup(slug_first).name == "A+B"
        assert ws.load_setup(slug_second).name == "A B"

    def test_concurrent_automatic_slug_selection_is_one_transaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Colliding request threads must not both claim the base slug."""
        import qpsim.webui.store as store_module

        ws = Workspace(tmp_path)
        first_selected = threading.Event()
        release_first = threading.Event()
        calls_lock = threading.Lock()
        calls = 0
        real_select = store_module._collision_safe_slug

        def synchronize_old_race(directory: Path, name: str) -> str:
            nonlocal calls
            candidate = real_select(directory, name)
            with calls_lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                first_selected.set()
                # With the transaction lock the second selector cannot enter
                # until this returns. Without it, the second selector releases
                # us after choosing the same still-free base slug.
                release_first.wait(0.25)
            else:
                release_first.set()
            return candidate

        monkeypatch.setattr(
            store_module, "_collision_safe_slug", synchronize_old_race
        )
        envelopes = (
            SetupEnvelope(name="A+B", setup=KineticsSetup()),
            SetupEnvelope(name="A B", setup=KineticsSetup()),
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(ws.save_setup, envelope) for envelope in envelopes]
            assert first_selected.wait(1.0)
            slugs = [future.result(timeout=2.0) for future in futures]

        assert len(set(slugs)) == 2
        assert {ws.load_setup(slug).name for slug in slugs} == {"A+B", "A B"}

    def test_long_automatic_setup_slug_is_bounded(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        name = "x" * 300
        slug = ws.save_setup(
            SetupEnvelope(name=name, setup=KineticsSetup())
        )
        assert len(slug) <= 120
        assert ws.load_setup(slug).name == name

    @pytest.mark.parametrize("setup_value", [None, "bad", []])
    def test_malformed_setup_record_is_isolated(
        self, tmp_path: Path, setup_value: object,
    ) -> None:
        ws = Workspace(tmp_path)
        ws.setups_dir.mkdir(parents=True)
        (ws.setups_dir / "bad.json").write_text(
            f'{{"name": "bad", "setup": {json.dumps(setup_value)}}}',
            encoding="utf-8",
        )
        entries = ws.list_setups()
        assert entries == [
            {"slug": "bad", "name": "bad", "mode": "unreadable"}
        ]
        with pytest.raises(ValueError, match=r"setup.*object"):
            ws.load_setup("bad")

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
        envelope = SetupEnvelope(name="persistence-failure", setup=KineticsSetup())
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

    def test_queued_cancel_is_durable_and_never_executes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.runner as runner_module
        from qpsim.webui.execute import RunPayload
        from qpsim.webui.runner import JobRunner

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        first_started = threading.Event()
        release_first = threading.Event()
        calls = 0

        def controlled_execute(*_args: object, **_kwargs: object) -> RunPayload:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                assert release_first.wait(5.0)
            return RunPayload(arrays={"f": np.ones(2)})

        monkeypatch.setattr(runner_module, "execute_setup", controlled_execute)
        envelope = SetupEnvelope(name="queued", setup=KineticsSetup())
        first_id = runner.submit(envelope)
        assert first_started.wait(5.0)
        first_job = runner.live_state(first_id)
        assert first_job is not None
        queued_id = runner.submit(envelope)

        assert runner.cancel(queued_id)
        assert ws.read_manifest(queued_id)["status"] == "cancelled"
        release_first.set()
        assert first_job.worker_finished.wait(5.0)
        runner.shutdown()
        assert calls == 1

    def test_shutdown_durably_cancels_a_queued_future(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.runner as runner_module
        from qpsim.webui.execute import RunPayload
        from qpsim.webui.runner import JobRunner

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        first_started = threading.Event()
        release_first = threading.Event()
        calls = 0

        def controlled_execute(*_args: object, **_kwargs: object) -> RunPayload:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                assert release_first.wait(5.0)
            return RunPayload(arrays={"f": np.ones(2)})

        monkeypatch.setattr(runner_module, "execute_setup", controlled_execute)
        envelope = SetupEnvelope(name="shutdown", setup=KineticsSetup())
        first_id = runner.submit(envelope)
        assert first_started.wait(5.0)
        first_job = runner.live_state(first_id)
        assert first_job is not None
        queued_id = runner.submit(envelope)

        runner.shutdown()
        assert ws.read_manifest(queued_id)["status"] == "cancelled"
        release_first.set()
        assert first_job.worker_finished.wait(5.0)
        assert ws.read_manifest(first_id)["status"] == "cancelled"
        assert calls == 1

    def test_cancel_during_array_persistence_cannot_publish_done(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.runner as runner_module
        from qpsim.webui.execute import RunPayload
        from qpsim.webui.runner import JobRunner

        ws = Workspace(tmp_path)
        runner = JobRunner(ws)
        write_started = threading.Event()
        release_write = threading.Event()
        real_write = ws.write_arrays

        monkeypatch.setattr(
            runner_module,
            "execute_setup",
            lambda *_args, **_kwargs: RunPayload(arrays={"f": np.ones(2)}),
        )

        def blocking_write(
            run_id: str, arrays: dict[str, np.ndarray],
        ) -> None:
            write_started.set()
            assert release_write.wait(5.0)
            real_write(run_id, arrays)

        monkeypatch.setattr(ws, "write_arrays", blocking_write)
        run_id = runner.submit(
            SetupEnvelope(name="cancel-persistence", setup=KineticsSetup())
        )
        assert write_started.wait(5.0)
        job = runner.live_state(run_id)
        assert job is not None
        assert runner.cancel(run_id)
        release_write.set()
        assert job.worker_finished.wait(5.0)

        assert ws.read_manifest(run_id)["status"] == "cancelled"
        assert not (ws.run_dir(run_id) / "result.npz").exists()
        runner.shutdown()


class TestDiagnosticsNeverSinkARun:
    def test_failing_gap_suppression_becomes_a_note(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import qpsim.webui.execute as execute_mod

        def boom(*_args: object, **_kwargs: object) -> object:
            raise ValueError("synthetic diagnostic failure")

        monkeypatch.setattr(execute_mod, "compute_gap_suppression", boom)
        setup = KineticsSetup()
        setup.strategy = "steady_state"
        setup.geometry.rows = setup.geometry.cols = 1
        setup.grid.num_bins = 48
        payload = execute_setup(setup, _noop_progress, _never)
        # The solve itself survived: arrays and primary observables present.
        assert "f" in payload.arrays
        assert payload.summary["x_qp"] > 0.0
        assert "delta_eq_ueV" in payload.summary
        assert "delta_suppression_ueV" not in payload.summary
        assert "rel_gap_suppression" not in payload.summary
        assert any("synthetic diagnostic failure" in n for n in payload.notes)

    def test_effective_temperature_clamp_warning_becomes_a_note(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.webui.execute as execute_mod
        from qpsim.backends.diffusion import DiffusionBackend

        def return_input_state(
            _backend: DiffusionBackend,
            state: object,
            **_kwargs: object,
        ) -> object:
            return state

        def clamped_fit(*_args: object, **_kwargs: object) -> float:
            warnings.warn(
                "effective temperature is a clamp, not a fit",
                RuntimeWarning,
                stacklevel=2,
            )
            return 1.0

        monkeypatch.setattr(
            DiffusionBackend, "steady_state", return_input_state
        )
        monkeypatch.setattr(
            execute_mod, "effective_phonon_temperature", clamped_fit
        )
        setup = KineticsSetup()
        setup.strategy = "steady_state"
        setup.geometry.rows = setup.geometry.cols = 1
        setup.grid.num_bins = 48
        setup.phonons.mode = "dynamic_escape"

        payload = execute_setup(setup, _noop_progress, _never)

        assert payload.summary["T_phonon_eff_K"] == 1.0
        assert any("clamp, not a fit" in note for note in payload.notes)


class TestSpatialProbeIsRefusedRatherThanAbsent:
    """A spatial device is TOLD why its probe yields no conductivity.

    `probe` is a field on every kinetics setup, because a one-cell mask is 0-D
    and its probe is perfectly well defined. On a mask of more than one cell
    the probe is refused at the point of use rather than silently dropped:
    sigma(f) is nonlinear and cells can carry different local gaps, so there
    is no single sigma for a spatially varying f. Refusing where the value
    would be computed is what lets the run say WHY.
    """

    def test_a_spatial_device_gets_no_conductivity_from_the_probe(self) -> None:
        from qpsim.webui.execute import execute_setup

        setup = KineticsSetup()
        setup.grid.num_bins = 24
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 5.0, 0.0
        setup.geometry.rows, setup.geometry.cols = 2, 3
        setup.probe.enabled = True

        payload = execute_setup(setup, _noop_progress, _never)

        assert not [k for k in payload.summary if "sigma" in k or k.startswith("Q_")]
        assert any("Mattis-Bardeen probe skipped" in n for n in payload.notes), (
            "a probe that is silently dropped leaves the caller believing "
            "it ran"
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
        setup = KineticsSetup()
        setup.strategy = "steady_state"
        setup.geometry.rows = setup.geometry.cols = 1
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


class TestPhononSourceSwitchesActOnEveryPath:
    """Four collision switches are offered; all four must do something.

    Two of them were bit-for-bit inert in `transient_0d`. The transient path
    passed the QUASIPARTICLE-side flags into the phonon-side source, so
    `phonon_scattering_source` and `phonon_recombination_source` had nothing
    to do, while `scattering=False` silently switched off the phonon source
    as well -- a different model from the same switches in `steady_state_0d`,
    where the two sides have always been independent.
    """

    @staticmethod
    def _driven(mode: str, flag: str, value: bool) -> float:
        import qpsim.webui.execute as execute
        from qpsim.webui.schemas import KineticsSetup

        setup = KineticsSetup()
        setup.strategy = "time_march" if mode == "transient_0d" else "steady_state"
        setup.geometry.rows = setup.geometry.cols = 1
        setup.T_bath = 0.2
        setup.grid.num_bins = 405
        setup.phonons.mode = "dynamic_escape"
        setup.pb_drive.enabled = True
        setup.pb_drive.omega_PB = 400.0
        setup.pb_drive.n_bar_PB = 1000.0
        setup.pb_drive.c_phot_PB = 1e-9
        if mode == "transient_0d":
            setup.max_time, setup.dt, setup.stop_tol = 4.0, 0.1, 0.0
        setattr(setup.collisions, flag, value)
        payload = execute.execute_setup(setup, lambda *a, **k: None, lambda: False)
        return float(
            payload.summary["x_qp_mean" if mode == "transient_0d" else "x_qp"]
        )

    @pytest.mark.parametrize(
        "flag", ["phonon_scattering_source", "phonon_recombination_source"]
    )
    @pytest.mark.parametrize("mode", ["transient_0d", "steady_state_0d"])
    def test_the_switch_changes_the_answer(self, mode: str, flag: str) -> None:
        on = self._driven(mode, flag, True)
        off = self._driven(mode, flag, False)
        assert on != off, (
            f"{flag} is inert in {mode}: the interface offers a switch that "
            f"does nothing (both runs gave x_qp = {on!r})."
        )
