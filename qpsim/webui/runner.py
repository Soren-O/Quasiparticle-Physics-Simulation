"""Background run manager: one worker thread, progress, cooperative cancel.

Runs execute on a single-worker thread pool — engine solves are
CPU-bound (numpy releases the GIL), and serializing them keeps an
interactive session responsive instead of thrashing; queued runs
start automatically as the worker frees up.

Live state (progress fraction, message) is in-memory only; the
persistent record is the run's manifest, written when the run is
submitted and rewritten when it finishes. The server overlays the
live state onto manifests when listing runs. The overlay is also the
recovery path for two failure shapes the disk alone can't express: a
terminal manifest write that lost the race to a reader (retried
lazily from the stashed copy), and a "running" manifest orphaned by a
crash or restart (reported as ``interrupted`` so it can be deleted).
"""

from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from qpsim.webui.execute import RunCancelledError, execute_setup
from qpsim.webui.schemas import SetupEnvelope
from qpsim.webui.store import Workspace

_ACTIVE = ("queued", "running")
_TERMINAL = ("done", "failed", "cancelled")
_STATUS_RANK = {
    "queued": 0,
    "running": 1,
    "done": 2,
    "failed": 2,
    "cancelled": 2,
}


@dataclass
class JobState:
    """In-memory view of one submitted run."""

    run_id: str
    status: str = "queued"  # queued | running | done | failed | cancelled
    progress: float = 0.0
    message: str = ""
    cancel_event: threading.Event = field(default_factory=threading.Event)
    started_monotonic: float | None = None
    # Terminal manifest whose disk write failed; overlay retries it.
    pending_manifest: dict[str, Any] | None = None
    # Set when the worker has completed its final persistence attempt. A
    # terminal job is not safe to delete before this point because the worker
    # could otherwise recreate its directory after deletion.
    worker_finished: threading.Event = field(
        default_factory=threading.Event,
        repr=False,
    )
    # Once a terminal manifest is durable or the user explicitly deletes a
    # failed-persistence run, stale overlay retries must become no-ops.
    manifest_writes_closed: bool = False
    # Manifest retries happen in request threads while the worker may finish.
    # Serialize them so an older active snapshot can never overwrite a newer
    # terminal manifest.
    manifest_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


class JobRunner:
    """Submit, track, and cancel simulation runs against a workspace."""

    def __init__(self, workspace: Workspace, *, max_workers: int = 1) -> None:
        self.workspace = workspace
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="qpsim-run"
        )
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()

    def submit(self, envelope: SetupEnvelope, *, warnings: list[str] | None = None) -> str:
        """Queue a run; validation warnings persist as the run's first notes."""
        run_id = self.workspace.new_run_id()
        manifest: dict[str, Any] = {
            "id": run_id,
            "name": envelope.name,
            "mode": envelope.setup.mode,
            "status": "queued",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "setup": envelope.setup.model_dump(),
            "summary": {},
            "notes": list(warnings or []),
            "error": None,
            "elapsed_s": None,
        }
        self.workspace.write_manifest(run_id, manifest)
        job = JobState(run_id=run_id)
        with self._lock:
            self._jobs[run_id] = job
        self._executor.submit(self._run, job, envelope, manifest)
        return run_id

    def _run(self, job: JobState, envelope: SetupEnvelope, manifest: dict[str, Any]) -> None:
        with job.manifest_lock:
            job.status = "running"
            job.started_monotonic = time.monotonic()
            manifest["status"] = "running"

        def progress(fraction: float, message: str) -> None:
            job.progress = max(0.0, min(1.0, fraction))
            job.message = message

        try:
            self._write_manifest_or_stash(job, manifest)
            payload = execute_setup(
                envelope.setup, progress, job.cancel_event.is_set
            )
            # Result persistence is part of the worker transaction. Keeping it
            # inside this try ensures an I/O failure becomes a durable failed
            # run instead of escaping through the discarded Future and
            # stranding both the live JobState and manifest at "running".
            self.workspace.write_arrays(job.run_id, payload.arrays)
        except RunCancelledError:
            with job.manifest_lock:
                job.status = "cancelled"
                manifest["status"] = "cancelled"
        except Exception as exc:  # a failed run must not kill the worker
            with job.manifest_lock:
                job.status = "failed"
                manifest["status"] = "failed"
                manifest["error"] = f"{type(exc).__name__}: {exc}"
                manifest["traceback"] = traceback.format_exc()
        else:
            with job.manifest_lock:
                job.status = "done"
                job.progress = 1.0
                manifest["status"] = "done"
                manifest["summary"] = payload.summary
                manifest["notes"] = list(manifest.get("notes", [])) + payload.notes
        finally:
            with job.manifest_lock:
                manifest["elapsed_s"] = round(
                    time.monotonic() - (job.started_monotonic or 0.0), 3
                )
            try:
                self._write_manifest_or_stash(job, manifest)
            finally:
                job.worker_finished.set()

    def _write_manifest_or_stash(self, job: JobState, manifest: dict[str, Any]) -> None:
        """Persist the manifest; on failure stash it so overlay can retry.

        A raise here must never escape: in the worker's ``finally`` it
        would vanish into the discarded Future and strand the run's
        disk record at "running" with no recovery path.
        """
        with job.manifest_lock:
            if job.manifest_writes_closed:
                return
            manifest_status = str(manifest.get("status", ""))
            current_status = job.status
            stale = (
                _STATUS_RANK.get(manifest_status, -1)
                < _STATUS_RANK.get(current_status, -1)
                or (
                    current_status in _TERMINAL
                    and manifest_status != current_status
                )
            )
            if stale:
                if (
                    job.pending_manifest is not None
                    and job.pending_manifest.get("status") == manifest_status
                ):
                    job.pending_manifest = None
                return

            try:
                self.workspace.write_manifest(job.run_id, manifest)
                job.pending_manifest = None
                if (
                    manifest_status in _TERMINAL
                    and manifest_status == job.status
                ):
                    # Retire only after this exact terminal state is durable.
                    job.manifest_writes_closed = True
                    with self._lock:
                        if self._jobs.get(job.run_id) is job:
                            self._jobs.pop(job.run_id, None)
            except OSError:
                job.pending_manifest = dict(manifest)

    def live_state(self, run_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(run_id)

    def release_terminal_for_delete(self, run_id: str) -> bool:
        """Stop retries for a finished terminal job so it can be deleted.

        A terminal job remains registered only when its final manifest could
        not be persisted. Once the worker's final attempt has returned, an
        explicit delete may discard that pending snapshot. Closing writes
        under the per-job lock also makes already-captured overlay retries
        harmless, so none can recreate the run directory after deletion.

        Returns ``True`` when no writer can remain, or ``False`` while the job
        is active/finalizing and deletion must still be refused.
        """
        job = self.live_state(run_id)
        if job is None:
            return True
        with job.manifest_lock:
            if job.status not in _TERMINAL or not job.worker_finished.is_set():
                return False
            job.manifest_writes_closed = True
            job.pending_manifest = None
            with self._lock:
                if self._jobs.get(run_id) is job:
                    self._jobs.pop(run_id, None)
            return True

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None or job.status not in _ACTIVE:
            return False
        job.cancel_event.set()
        return True

    def overlay(self, manifest: dict[str, Any]) -> dict[str, Any]:
        """Merge live state into a disk manifest for API responses.

        Three cases: a live active job contributes progress; a live
        terminal job whose final write failed gets that write retried
        (serving the stashed copy meanwhile); a manifest stuck on an
        active status with no live job — crash or restart — reports as
        ``interrupted`` so it is inspectable and deletable.
        """
        job = self.live_state(str(manifest.get("id", "")))
        if job is None:
            if manifest.get("status") in _ACTIVE:
                manifest = dict(manifest)
                manifest["status"] = "interrupted"
                manifest["error"] = (
                    "The server stopped (or the worker died) while this run "
                    "was active; its result was not recorded."
                )
            return manifest
        with job.manifest_lock:
            stashed = (
                None
                if job.pending_manifest is None
                else dict(job.pending_manifest)
            )
            current_status = job.status
        if stashed is not None:
            self._write_manifest_or_stash(job, stashed)
            if stashed.get("status") == current_status:
                manifest = stashed
        if current_status in _ACTIVE:
            manifest = dict(manifest)
            manifest["status"] = current_status
            manifest["progress"] = job.progress
            manifest["progress_message"] = job.message
        # A live terminal job with a still-active disk manifest means
        # the final write is in flight (or stashed and just retried);
        # serve the disk state as-is — never promote the status without
        # its summary/notes.
        return manifest

    def shutdown(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel_event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)
