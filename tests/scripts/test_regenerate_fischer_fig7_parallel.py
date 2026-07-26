"""Fast failure/restart tests for the guarded Fig. 7 campaign driver."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023 import fig7_paper, fig7_solve

from scripts import regenerate_fischer_fig7_parallel as driver


def _producer() -> fig7_paper.Fig7ProducerIdentity:
    return fig7_paper.capture_producer_identity(Path(driver.__file__).resolve())


def _write_valid_point(
    path: Path,
    *,
    pi: int,
    ti: int,
    power: float,
    temperature: float,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> None:
    shape = (1, 1)
    raw: dict[str, np.ndarray] = {
        "f_solved": np.full((1, 1, fig7_solve.NUM_BINS), 1.0e-8),
        "temperatures": np.asarray([temperature]),
        "powers_dbm": np.asarray([power]),
        "n_bar": np.asarray([fig7_solve._nbar_from_table_iii(power)]),
        "num_bins": np.asarray([fig7_solve.NUM_BINS]),
    }
    raw.update({
        field: np.zeros(shape)
        for field in fig7_solve.NUMBER_CERTIFICATE_FIELDS
    })
    with path.open("wb") as stream:
        np.savez(
            stream,
            **raw,
            point_schema=np.asarray([driver.POINT_SCHEMA]),
            point_power_index=np.asarray([pi]),
            point_temperature_index=np.asarray([ti]),
            run_identity=np.asarray([run_identity]),
            solve_contract_digest=np.asarray(
                [producer.solve_contract_digest]
            ),
            observable_contract_digest=np.asarray(
                [producer.observable_contract_digest]
            ),
            runner_sha256=np.asarray([producer.runner_sha256]),
            producer_runtime_json=np.asarray(
                [driver._canonical_json(producer.runtime)]
            ),
            elapsed_s=np.asarray([1.25]),
        )


def test_thread_controls_are_enforced_before_driver_numpy_import() -> None:
    assert {
        name: os.environ.get(name)
        for name in driver.SINGLE_THREAD_ENVIRONMENT
    } == driver.SINGLE_THREAD_ENVIRONMENT
    driver._assert_single_thread_environment()


def test_fresh_campaign_process_sets_threads_before_numpy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import regenerate_fischer_fig7_parallel as d; "
                "assert not d._NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT; "
                "d._assert_single_thread_environment()"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_run_directory_is_deterministic_and_digest_keyed(tmp_path: Path) -> None:
    producer = _producer()
    first = driver._resolve_run_dir(tmp_path, producer)
    second = driver._resolve_run_dir(tmp_path, producer)
    assert first == second
    identity, path = first
    assert path == tmp_path.resolve() / f"fig7-{identity}"
    assert len(identity) == 64


def test_campaign_lock_rejects_a_second_live_owner(tmp_path: Path) -> None:
    producer = _producer()
    identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    run_dir.mkdir()
    handle = driver._acquire_campaign_lock(
        run_dir,
        run_identity=identity,
        producer=producer,
    )
    try:
        with pytest.raises(RuntimeError, match="already active"):
            driver._acquire_campaign_lock(
                run_dir,
                run_identity=identity,
                producer=producer,
            )
    finally:
        driver._release_campaign_lock(handle)
    assert not handle["lock_dir"].exists()


def test_campaign_lock_recovers_a_stale_process_creation_identity(
    tmp_path: Path,
) -> None:
    producer = _producer()
    identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    run_dir.mkdir()
    lock_dir = run_dir / ".campaign.lock"
    lock_dir.mkdir()
    stale_owner = driver._lock_owner_payload(
        token="stale-token",
        run_identity=identity,
        producer=producer,
        process_identity="previous-boot-or-pid-incarnation",
    )
    driver._atomic_json(lock_dir / "owner.json", stale_owner)

    handle = driver._acquire_campaign_lock(
        run_dir,
        run_identity=identity,
        producer=producer,
    )
    try:
        owner = driver._read_lock_owner(handle["owner_path"])
        assert owner["token"] == handle["token"]
        assert owner["process_identity"] == driver._process_identity(os.getpid())
    finally:
        driver._release_campaign_lock(handle)
    assert not lock_dir.exists()


def test_campaign_lock_fallback_is_stable_and_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingProcPath:
        """Model a POSIX host without Linux's process-creation API."""

        def __init__(self, _raw: object) -> None:
            pass

        def is_file(self) -> bool:
            return False

    monkeypatch.setattr(driver.sys, "platform", "darwin")
    monkeypatch.setattr(driver, "Path", MissingProcPath)
    monkeypatch.setattr(driver.os, "kill", lambda _pid, _signal: None)
    first = driver._process_identity(os.getpid())
    second = driver._process_identity(os.getpid())
    assert first == second == f"live-unverifiable:{os.getpid()}"


def test_campaign_failure_releases_its_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT", False)

    def fail_scan(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated resume scan failure")

    monkeypatch.setattr(driver, "_scan_resume_points", fail_scan)
    producer = _producer()
    _identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    with pytest.raises(RuntimeError, match="resume scan"):
        driver.run_campaign(run_root=tmp_path, max_workers=1)
    assert not (run_dir / ".campaign.lock").exists()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["state"] == "failed"


def test_resume_scans_valid_points_and_returns_only_missing(tmp_path: Path) -> None:
    producer = _producer()
    identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    run_dir.mkdir()
    points = driver._all_points(run_dir, identity, producer)
    first = points[0]
    path = driver._point_path(run_dir, first[0], first[1])
    _write_valid_point(
        path,
        pi=first[0],
        ti=first[1],
        power=first[2],
        temperature=first[3],
        run_identity=identity,
        producer=producer,
    )

    records = driver._scan_resume_points(
        run_dir,
        points,
        identity,
        producer,
    )
    assert set(records) == {(first[0], first[1])}
    missing = driver._missing_points(points, records)
    assert len(missing) == len(points) - 1
    assert first not in missing


@pytest.mark.parametrize("corruption", ["bytes", "runner"])
def test_resume_rejects_corrupt_or_foreign_point(
    tmp_path: Path,
    corruption: str,
) -> None:
    producer = _producer()
    identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    run_dir.mkdir()
    points = driver._all_points(run_dir, identity, producer)
    first = points[0]
    path = driver._point_path(run_dir, first[0], first[1])
    _write_valid_point(
        path,
        pi=first[0],
        ti=first[1],
        power=first[2],
        temperature=first[3],
        run_identity=identity,
        producer=producer,
    )
    if corruption == "bytes":
        path.write_bytes(b"not-an-npz")
    else:
        with np.load(path, allow_pickle=False) as payload:
            values = {name: np.asarray(payload[name]) for name in payload.files}
        values["runner_sha256"] = np.asarray(["0" * 64])
        with path.open("wb") as stream:
            np.savez(stream, **values)

    with pytest.raises(RuntimeError, match="point payload"):
        driver._scan_resume_points(run_dir, points, identity, producer)


def test_resume_rejects_unexpected_point_filename(tmp_path: Path) -> None:
    producer = _producer()
    identity, run_dir = driver._resolve_run_dir(tmp_path, producer)
    run_dir.mkdir()
    (run_dir / "point-p99-t99.npz").write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="unexpected point payload"):
        driver._scan_resume_points(
            run_dir,
            driver._all_points(run_dir, identity, producer),
            identity,
            producer,
        )


def test_status_is_atomic_and_strictly_source_bound(tmp_path: Path) -> None:
    producer = _producer()
    identity = driver._run_identity(producer)
    path = tmp_path / "status.json"
    status = driver._status_payload(
        state="failed",
        run_identity=identity,
        producer=producer,
        started_utc="2026-07-25T00:00:00+00:00",
        attempt=2,
        completed=7,
        error=RuntimeError("sentinel"),
    )
    driver._atomic_json(path, status)
    assert not list(tmp_path.glob(".status.json.*.tmp"))
    restored = driver._read_prior_status(
        path,
        run_identity=identity,
        producer=producer,
    )
    assert restored is not None
    assert restored["state"] == "failed"
    assert restored["completed_points"] == 7
    assert json.loads(path.read_text(encoding="utf-8")) == status

    forged = dict(status)
    forged["runner_sha256"] = "0" * 64
    driver._atomic_json(path, forged)
    with pytest.raises(RuntimeError, match="runner_sha256"):
        driver._read_prior_status(
            path,
            run_identity=identity,
            producer=producer,
        )


def test_first_error_cancellation_terminates_workers() -> None:
    class FakeFuture:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    class FakeExecutor:
        def __init__(self) -> None:
            self.terminated = False

        def terminate_workers(self) -> None:
            self.terminated = True

    futures = [FakeFuture(), FakeFuture()]
    executor = FakeExecutor()
    driver._cancel_executor(executor, futures)  # type: ignore[arg-type]
    assert executor.terminated
    assert all(future.cancelled for future in futures)


def test_second_submit_failure_terminates_first_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFuture:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    class FakeExecutor:
        instance: FakeExecutor | None = None

        def __init__(self, **_kwargs: object) -> None:
            self.submit_calls = 0
            self.first_future = FakeFuture()
            self.terminated = False
            FakeExecutor.instance = self

        def submit(self, *_args: object) -> FakeFuture:
            self.submit_calls += 1
            if self.submit_calls == 2:
                raise RuntimeError("simulated second-submit failure")
            return self.first_future

        def terminate_workers(self) -> None:
            self.terminated = True

    monkeypatch.setattr(driver, "ProcessPoolExecutor", FakeExecutor)
    missing = [
        (0, 0, -100.0, 0.06, str(tmp_path), "r", "s", "o", "h"),
        (0, 1, -100.0, 0.10, str(tmp_path), "r", "s", "o", "h"),
    ]
    with pytest.raises(RuntimeError, match="second-submit"):
        driver._run_missing_points(
            missing,
            max_workers=2,
            records={},
            status_path=tmp_path / "status.json",
            status_base={},
        )

    executor = FakeExecutor.instance
    assert executor is not None
    assert executor.terminated
    assert executor.first_future.cancelled
