# ruff: noqa: E402, I001
"""Guarded, resumable parallel regeneration of Fischer 2023 Fig. 7.

Run from the repository root with::

    python -m scripts.regenerate_fischer_fig7_parallel

The run directory is deterministic and content-addressed by the solve,
observable, runner, configuration, and runtime contracts. Each physical point
is an independent atomic payload. Restarting validates every existing payload
and submits only missing points. Canonical publication goes through the shared
PDF-first/CSV-last publisher in :mod:`validation.fischer_2023.fig7_paper`.
"""

from __future__ import annotations

import os
import sys

# These controls must be established before NumPy/SciPy/BLAS are imported.
_NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT = "numpy" in sys.modules
SINGLE_THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_DYNAMIC": "FALSE",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _name, _value in SINGLE_THREAD_ENVIRONMENT.items():
    os.environ[_name] = _value

import argparse
import hashlib
import json
import multiprocessing
import secrets
import tempfile
import time
import zipfile
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from validation.fischer_2023 import fig7_paper, fig7_solve
from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_FIELDS,
)

STATUS_SCHEMA = "qpsim-fischer-2023-fig7-run-v1"
POINT_SCHEMA = "qpsim-fischer-2023-fig7-point-v1"
LOCK_SCHEMA = "qpsim-fischer-2023-fig7-lock-v1"
DEFAULT_MAX_WORKERS = 6
_INCOMPLETE_LOCK_GRACE_S = 30.0


class PointRecord(TypedDict):
    """Typed identity and timing record for one validated worker payload."""

    pi: int
    ti: int
    power_dbm: float
    temperature_K: float
    elapsed_s: float
    path: str
    sha256: str


class CampaignLockHandle(TypedDict):
    """The identity required to release exactly the lock we acquired."""

    lock_dir: Path
    owner_path: Path
    token: str


def _assert_single_thread_environment() -> None:
    actual = {
        name: os.environ.get(name)
        for name in SINGLE_THREAD_ENVIRONMENT
    }
    if actual != SINGLE_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "Fig. 7 campaign lost its required single-thread environment: "
            f"{actual!r}."
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                payload,
                stream,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_identity_payload(
    producer: fig7_paper.Fig7ProducerIdentity,
) -> dict[str, object]:
    return {
        "schema": STATUS_SCHEMA,
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
        "producer_runtime": producer.runtime,
        "num_bins": fig7_solve.NUM_BINS,
        "powers_dbm": list(fig7_solve.P_READ_DBM),
        "temperatures_K": list(fig7_solve.T_BATH_VALUES),
        "single_thread_environment": dict(SINGLE_THREAD_ENVIRONMENT),
    }


def _run_identity(producer: fig7_paper.Fig7ProducerIdentity) -> str:
    return hashlib.sha256(
        _canonical_json(_run_identity_payload(producer)).encode("utf-8")
    ).hexdigest()


def _resolve_run_dir(
    run_root: Path,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> tuple[str, Path]:
    identity = _run_identity(producer)
    return identity, (run_root.expanduser().resolve() / f"fig7-{identity}")


def _process_identity(pid: int) -> str | None:
    """Return an OS process-creation identity, or ``None`` for a dead PID."""
    if pid < 1:
        return None
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class FILETIME(ctypes.Structure):
            _fields_ = (
                ("dwLowDateTime", wintypes.DWORD),
                ("dwHighDateTime", wintypes.DWORD),
            )

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
            ctypes.POINTER(FILETIME),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied means "possibly live", never "safe to steal".
            if ctypes.get_last_error() == 5:
                return f"live-unverifiable:{pid}"
            return None
        creation = FILETIME()
        exit_time = FILETIME()
        kernel_time = FILETIME()
        user_time = FILETIME()
        try:
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_time),
                ctypes.byref(kernel_time),
                ctypes.byref(user_time),
            ):
                return None
        finally:
            kernel32.CloseHandle(handle)
        ticks = (int(creation.dwHighDateTime) << 32) | int(
            creation.dwLowDateTime
        )
        return f"windows-filetime:{ticks}"

    proc_stat = Path(f"/proc/{pid}/stat")
    boot_id_path = Path("/proc/sys/kernel/random/boot_id")
    if proc_stat.is_file() and boot_id_path.is_file():
        try:
            raw = proc_stat.read_text(encoding="utf-8")
            closing = raw.rfind(")")
            fields_after_comm = raw[closing + 2 :].split()
            # Field 22 (starttime); fields_after_comm begins at field 3.
            start_ticks = fields_after_comm[19]
            boot_id = boot_id_path.read_text(encoding="utf-8").strip()
        except (OSError, IndexError):
            return f"live-unverifiable:{pid}"
        return f"linux:{boot_id}:{start_ticks}"

    # Conservative fallback for platforms without a creation-time API.
    # The marker is deliberately stable across calls, but such a live lock is
    # never stolen because PID reuse cannot be distinguished safely there.
    try:
        os.kill(pid, 0)
    except (OSError, PermissionError):
        return None
    return f"live-unverifiable:{pid}"


def _lock_owner_payload(
    *,
    token: str,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
    process_identity: str,
) -> dict[str, object]:
    return {
        "schema": LOCK_SCHEMA,
        "token": token,
        "run_identity": run_identity,
        "pid": os.getpid(),
        "process_identity": process_identity,
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
        "producer_runtime": producer.runtime,
        "acquired_utc": datetime.now(UTC).isoformat(),
    }


_LOCK_OWNER_FIELDS = {
    "schema",
    "token",
    "run_identity",
    "pid",
    "process_identity",
    "solve_contract_digest",
    "observable_contract_digest",
    "runner_sha256",
    "producer_runtime",
    "acquired_utc",
}


def _read_lock_owner(owner_path: Path) -> dict[str, object]:
    try:
        payload = json.loads(owner_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fig. 7 campaign lock owner is corrupt: {owner_path}.") from exc
    if not isinstance(payload, dict) or set(payload) != _LOCK_OWNER_FIELDS:
        raise RuntimeError(f"Fig. 7 campaign lock owner is malformed: {owner_path}.")
    if payload.get("schema") != LOCK_SCHEMA:
        raise RuntimeError(f"Fig. 7 campaign lock owner has stale schema: {owner_path}.")
    pid = payload.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise RuntimeError(f"Fig. 7 campaign lock owner has invalid PID: {owner_path}.")
    for field in (
        "token",
        "run_identity",
        "process_identity",
        "solve_contract_digest",
        "observable_contract_digest",
        "runner_sha256",
        "acquired_utc",
    ):
        if not isinstance(payload.get(field), str) or not payload[field]:
            raise RuntimeError(
                f"Fig. 7 campaign lock owner has invalid {field}: {owner_path}."
            )
    if not isinstance(payload.get("producer_runtime"), dict):
        raise RuntimeError(
            f"Fig. 7 campaign lock owner has invalid runtime: {owner_path}."
        )
    return cast(dict[str, object], payload)


def _remove_stale_lock_directory(lock_dir: Path, owner_path: Path) -> None:
    """Remove only the known owner file and then the now-empty lock directory."""
    children = list(lock_dir.iterdir())
    known_temporary = [
        child
        for child in children
        if child.is_file() and child.name.startswith(".owner.json.") and child.suffix == ".tmp"
    ]
    unexpected = [
        child
        for child in children
        if child.resolve() != owner_path.resolve() and child not in known_temporary
    ]
    if unexpected:
        raise RuntimeError(
            "Refusing to remove Fig. 7 stale lock with unexpected contents: "
            f"{unexpected}."
        )
    for temporary in known_temporary:
        temporary.unlink(missing_ok=True)
    owner_path.unlink(missing_ok=True)
    try:
        lock_dir.rmdir()
    except OSError as exc:
        raise RuntimeError(f"Cannot remove stale Fig. 7 lock {lock_dir}.") from exc


def _acquire_campaign_lock(
    run_dir: Path,
    *,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> CampaignLockHandle:
    """Acquire the exclusive lock for one deterministic campaign directory."""
    lock_dir = run_dir / ".campaign.lock"
    owner_path = lock_dir / "owner.json"
    process_identity = _process_identity(os.getpid())
    if process_identity is None:
        raise RuntimeError("Cannot determine the current Fig. 7 runner identity.")
    for _attempt in range(10):
        token = secrets.token_hex(32)
        try:
            lock_dir.mkdir()
        except FileExistsError:
            if owner_path.exists():
                owner = _read_lock_owner(owner_path)
            else:
                try:
                    age_s = max(0.0, time.time() - lock_dir.stat().st_mtime)
                except OSError:
                    continue
                if age_s < _INCOMPLETE_LOCK_GRACE_S:
                    raise RuntimeError(
                        "Fig. 7 campaign lock is being acquired by another process; "
                        f"refusing to race {lock_dir}."
                    ) from None
                owner = None
            if owner is not None:
                owner_pid = cast(int, owner["pid"])
                actual_identity = _process_identity(owner_pid)
                if actual_identity == owner["process_identity"]:
                    raise RuntimeError(
                        "Fig. 7 campaign is already active: "
                        f"pid={owner_pid}, run_identity={owner['run_identity']}, "
                        f"lock={lock_dir}."
                    ) from None
                if actual_identity is not None and actual_identity.startswith(
                    "live-unverifiable:"
                ):
                    raise RuntimeError(
                        "Fig. 7 campaign lock belongs to a live process whose "
                        f"creation identity cannot be verified: pid={owner_pid}."
                    ) from None
            stale_dir = lock_dir.with_name(
                f"{lock_dir.name}.stale-{os.getpid()}-{secrets.token_hex(8)}"
            )
            try:
                os.replace(lock_dir, stale_dir)
            except OSError:
                continue
            _remove_stale_lock_directory(stale_dir, stale_dir / "owner.json")
            continue
        owner = _lock_owner_payload(
            token=token,
            run_identity=run_identity,
            producer=producer,
            process_identity=process_identity,
        )
        try:
            _atomic_json(owner_path, owner)
        except BaseException:
            _remove_stale_lock_directory(lock_dir, owner_path)
            raise
        return CampaignLockHandle(
            lock_dir=lock_dir,
            owner_path=owner_path,
            token=token,
        )
    raise RuntimeError(f"Could not acquire Fig. 7 campaign lock {lock_dir}.")


def _release_campaign_lock(handle: CampaignLockHandle) -> None:
    owner = _read_lock_owner(handle["owner_path"])
    if owner.get("token") != handle["token"] or owner.get("pid") != os.getpid():
        raise RuntimeError(
            f"Refusing to release a Fig. 7 campaign lock we do not own: "
            f"{handle['lock_dir']}."
        )
    _remove_stale_lock_directory(handle["lock_dir"], handle["owner_path"])


def _point_path(run_dir: Path, pi: int, ti: int) -> Path:
    return run_dir / f"point-p{pi:02d}-t{ti:02d}.npz"


def _point_args(
    *,
    pi: int,
    ti: int,
    power: float,
    temperature: float,
    run_dir: Path,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> tuple[int, int, float, float, str, str, str, str, str]:
    return (
        pi,
        ti,
        power,
        temperature,
        str(run_dir),
        run_identity,
        producer.solve_contract_digest,
        producer.observable_contract_digest,
        producer.runner_sha256,
    )


def _solve_one(
    args: tuple[int, int, float, float, str, str, str, str, str],
) -> PointRecord:
    (
        pi,
        ti,
        power,
        temperature,
        run_dir_raw,
        run_identity,
        solve_digest,
        observable_digest,
        runner_sha256,
    ) = args
    _assert_single_thread_environment()
    runner_path = Path(__file__).resolve()
    producer = fig7_paper.capture_producer_identity(runner_path)
    if (
        producer.solve_contract_digest != solve_digest
        or producer.observable_contract_digest != observable_digest
        or producer.runner_sha256 != runner_sha256
    ):
        raise RuntimeError(
            f"Fig. 7 source/runner identity changed before point ({pi}, {ti})."
        )
    runtime_json = _canonical_json(producer.runtime)
    started = time.perf_counter()
    raw = fig7_solve.solve(
        temperatures=(temperature,),
        powers_dbm=(power,),
        num_bins=fig7_solve.NUM_BINS,
    )
    elapsed = time.perf_counter() - started
    fig7_paper._validated_raw_payload(raw)
    fig7_paper.assert_producer_identity_current(producer, runner_path)

    final_path = _point_path(Path(run_dir_raw), pi, ti)
    temporary = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez(
                stream,
                f_solved=raw["f_solved"],
                temperatures=raw["temperatures"],
                powers_dbm=raw["powers_dbm"],
                n_bar=raw["n_bar"],
                num_bins=raw["num_bins"],
                qp_residual_inf=raw["qp_residual_inf"],
                qp_backward_error=raw["qp_backward_error"],
                phonon_residual_inf=raw["phonon_residual_inf"],
                phonon_raw_backward_error=raw["phonon_raw_backward_error"],
                phonon_backward_error=raw["phonon_backward_error"],
                qp_number_backward_error=raw["qp_number_backward_error"],
                point_schema=np.asarray([POINT_SCHEMA]),
                point_power_index=np.asarray([pi]),
                point_temperature_index=np.asarray([ti]),
                run_identity=np.asarray([run_identity]),
                solve_contract_digest=np.asarray([solve_digest]),
                observable_contract_digest=np.asarray([observable_digest]),
                runner_sha256=np.asarray([runner_sha256]),
                producer_runtime_json=np.asarray([runtime_json]),
                elapsed_s=np.asarray([elapsed]),
            )
            stream.flush()
            os.fsync(stream.fileno())
        # Recheck the runner immediately before the atomic payload commit.
        fig7_paper.assert_producer_identity_current(producer, runner_path)
        os.replace(temporary, final_path)
    finally:
        temporary.unlink(missing_ok=True)
    return _point_record(
        final_path,
        pi=pi,
        ti=ti,
        power=power,
        temperature=temperature,
        run_identity=run_identity,
        producer=producer,
    )


def _point_record(
    path: Path,
    *,
    pi: int,
    ti: int,
    power: float,
    temperature: float,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> PointRecord:
    """Strictly validate one complete worker payload."""
    expected_fields = {
        "f_solved",
        "temperatures",
        "powers_dbm",
        "n_bar",
        "num_bins",
        *NUMBER_CERTIFICATE_FIELDS,
        "point_schema",
        "point_power_index",
        "point_temperature_index",
        "run_identity",
        "solve_contract_digest",
        "observable_contract_digest",
        "runner_sha256",
        "producer_runtime_json",
        "elapsed_s",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.namelist()
            expected_members = {f"{field}.npy" for field in expected_fields}
            if len(members) != len(set(members)) or set(members) != expected_members:
                raise RuntimeError(
                    f"Fig. 7 point payload {path} has wrong/duplicate NPZ members."
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(
                    f"Fig. 7 point payload {path} has corrupt member {bad_member}."
                )
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_fields:
                raise RuntimeError(f"Fig. 7 point payload {path} has wrong fields.")

            def scalar_string(field: str) -> str:
                values = np.asarray(payload[field])
                if values.shape != (1,) or values.dtype.kind not in {"U", "S"}:
                    raise RuntimeError(
                        f"Fig. 7 point payload {path} has malformed {field}."
                    )
                return str(values[0])

            def scalar_int(field: str) -> int:
                values = np.asarray(payload[field])
                if values.shape != (1,) or np.iscomplexobj(values):
                    raise RuntimeError(
                        f"Fig. 7 point payload {path} has malformed {field}."
                    )
                value = values[0]
                if (
                    isinstance(value, (bool, np.bool_))
                    or not np.isfinite(value)
                    or float(value) != int(value)
                ):
                    raise RuntimeError(
                        f"Fig. 7 point payload {path} has malformed {field}."
                    )
                return int(value)

            checks = {
                "point_schema": POINT_SCHEMA,
                "run_identity": run_identity,
                "solve_contract_digest": producer.solve_contract_digest,
                "observable_contract_digest": (
                    producer.observable_contract_digest
                ),
                "runner_sha256": producer.runner_sha256,
                "producer_runtime_json": _canonical_json(producer.runtime),
            }
            for field, expected in checks.items():
                if scalar_string(field) != expected:
                    raise RuntimeError(
                        f"Fig. 7 point payload {path} has wrong {field}."
                    )
            if scalar_int("point_power_index") != pi:
                raise RuntimeError(f"Fig. 7 point payload {path} has wrong power index.")
            if scalar_int("point_temperature_index") != ti:
                raise RuntimeError(
                    f"Fig. 7 point payload {path} has wrong temperature index."
                )
            elapsed_raw = np.asarray(payload["elapsed_s"], dtype=float)
            if (
                elapsed_raw.shape != (1,)
                or not np.isfinite(elapsed_raw[0])
                or elapsed_raw[0] < 0.0
            ):
                raise RuntimeError(
                    f"Fig. 7 point payload {path} has invalid elapsed time."
                )
            raw = {
                field: np.asarray(payload[field])
                for field in {
                    "f_solved",
                    "temperatures",
                    "powers_dbm",
                    "n_bar",
                    "num_bins",
                    *NUMBER_CERTIFICATE_FIELDS,
                }
            }
            fig7_paper._validated_raw_payload(raw)
            np.testing.assert_array_equal(raw["powers_dbm"], [power])
            np.testing.assert_array_equal(raw["temperatures"], [temperature])
            np.testing.assert_array_equal(raw["num_bins"], [fig7_solve.NUM_BINS])
            elapsed = float(elapsed_raw[0])
    except (OSError, OverflowError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Cannot validate Fig. 7 point payload {path}.") from exc
    return {
        "pi": pi,
        "ti": ti,
        "power_dbm": power,
        "temperature_K": temperature,
        "elapsed_s": elapsed,
        "path": str(path),
        "sha256": fig7_paper.file_sha256(path),
    }


def _all_points(
    run_dir: Path,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> list[tuple[int, int, float, float, str, str, str, str, str]]:
    priority = {
        -68.0: 0,
        -64.0: 1,
        -72.0: 2,
        -80.0: 3,
        -90.0: 4,
        -100.0: 5,
    }
    points = [
        _point_args(
            pi=pi,
            ti=ti,
            power=float(power),
            temperature=float(temperature),
            run_dir=run_dir,
            run_identity=run_identity,
            producer=producer,
        )
        for pi, power in enumerate(fig7_solve.P_READ_DBM)
        for ti, temperature in enumerate(fig7_solve.T_BATH_VALUES)
    ]
    points.sort(key=lambda item: (item[3], priority.get(item[2], 99)))
    return points


def _scan_resume_points(
    run_dir: Path,
    points: list[tuple[int, int, float, float, str, str, str, str, str]],
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> dict[tuple[int, int], PointRecord]:
    """Validate every existing point and reject foreign/corrupt payloads."""
    expected_paths = {
        _point_path(run_dir, point[0], point[1]).resolve()
        for point in points
    }
    for temporary in run_dir.glob(".point-*.npz.*.tmp"):
        temporary.unlink(missing_ok=True)
    actual_paths = {path.resolve() for path in run_dir.glob("*.npz")}
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise RuntimeError(
            f"Fig. 7 run directory contains unexpected point payloads: {unexpected}."
        )
    records: dict[tuple[int, int], PointRecord] = {}
    for point in points:
        pi, ti, power, temperature = point[:4]
        path = _point_path(run_dir, pi, ti)
        if not path.exists():
            continue
        records[(pi, ti)] = _point_record(
            path,
            pi=pi,
            ti=ti,
            power=power,
            temperature=temperature,
            run_identity=run_identity,
            producer=producer,
        )
    return records


def _missing_points(
    points: list[tuple[int, int, float, float, str, str, str, str, str]],
    records: Mapping[tuple[int, int], PointRecord],
) -> list[tuple[int, int, float, float, str, str, str, str, str]]:
    """Return only points not represented by a strictly validated payload."""
    return [
        point
        for point in points
        if (point[0], point[1]) not in records
    ]


def _read_prior_status(
    path: Path,
    *,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fig. 7 run status at {path} is corrupt.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Fig. 7 run status at {path} is malformed.")
    state = payload.get("state")
    common_fields = {
        "schema",
        "state",
        "run_identity",
        "solve_contract_digest",
        "observable_contract_digest",
        "runner_sha256",
        "producer_runtime",
        "single_thread_environment",
        "started_utc",
        "updated_utc",
        "attempt",
        "completed_points",
        "total_points",
    }
    allowed_fields = set(common_fields)
    if state == "failed":
        allowed_fields.add("error")
    elif state == "complete":
        allowed_fields.add("completion")
    if set(payload) != allowed_fields:
        raise RuntimeError(
            f"Fig. 7 run status at {path} has missing or unexpected fields."
        )
    expected = {
        "schema": STATUS_SCHEMA,
        "run_identity": run_identity,
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
        "producer_runtime": producer.runtime,
        "single_thread_environment": dict(SINGLE_THREAD_ENVIRONMENT),
        "total_points": len(fig7_solve.P_READ_DBM) * len(fig7_solve.T_BATH_VALUES),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Fig. 7 run status at {path} has wrong {field}; "
                "use its digest-keyed run directory only with matching source."
            )
    if state not in {"running", "failed", "complete"}:
        raise RuntimeError(f"Fig. 7 run status at {path} has invalid state.")
    for field in ("attempt", "completed_points"):
        value = payload.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < (1 if field == "attempt" else 0)
        ):
            raise RuntimeError(f"Fig. 7 run status at {path} has invalid {field}.")
    if int(payload["completed_points"]) > int(payload["total_points"]):
        raise RuntimeError(
            f"Fig. 7 run status at {path} has too many completed points."
        )
    if state == "complete" and payload["completed_points"] != payload["total_points"]:
        raise RuntimeError(
            f"Fig. 7 complete status at {path} has an incomplete point count."
        )
    return cast(dict[str, object], payload)


def _status_payload(
    *,
    state: str,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
    started_utc: str,
    attempt: int,
    completed: int,
    error: BaseException | None = None,
    completion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": STATUS_SCHEMA,
        "state": state,
        "run_identity": run_identity,
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
        "producer_runtime": producer.runtime,
        "single_thread_environment": dict(SINGLE_THREAD_ENVIRONMENT),
        "started_utc": started_utc,
        "updated_utc": datetime.now(UTC).isoformat(),
        "attempt": attempt,
        "completed_points": completed,
        "total_points": len(fig7_solve.P_READ_DBM) * len(fig7_solve.T_BATH_VALUES),
    }
    if error is not None:
        payload["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    if completion is not None:
        payload["completion"] = dict(completion)
    return payload


def _cancel_executor(
    executor: ProcessPoolExecutor,
    futures: list[Future[PointRecord]],
) -> None:
    for future in futures:
        future.cancel()
    terminate = getattr(executor, "terminate_workers", None)
    if callable(terminate):
        terminate()
    else:  # Python 3.13 fallback: cancel queued jobs and do not wait.
        processes = getattr(executor, "_processes", {})
        if isinstance(processes, dict):
            for process in processes.values():
                process.terminate()
        executor.shutdown(wait=False, cancel_futures=True)


def _run_missing_points(
    missing: list[tuple[int, int, float, float, str, str, str, str, str]],
    *,
    max_workers: int,
    records: dict[tuple[int, int], PointRecord],
    status_path: Path,
    status_base: dict[str, object],
) -> None:
    if not missing:
        return
    context = multiprocessing.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    futures: list[Future[PointRecord]] = []
    try:
        executor = ProcessPoolExecutor(
            max_workers=min(max_workers, len(missing)),
            mp_context=context,
        )
        for point in missing:
            # Track each future immediately: if a later submit fails, every
            # already-started worker is owned by the protected cleanup path.
            futures.append(executor.submit(_solve_one, point))
        for future in as_completed(futures):
            record = future.result()
            key = (record["pi"], record["ti"])
            if key in records:
                raise RuntimeError(f"Fig. 7 worker returned duplicate point {key}.")
            records[key] = record
            status = dict(status_base)
            status["completed_points"] = len(records)
            status["updated_utc"] = datetime.now(UTC).isoformat()
            _atomic_json(status_path, status)
            print(
                f"DONE {len(records):02d}/48 "
                f"P={record['power_dbm']:g} "
                f"T={record['temperature_K']:g} "
                f"elapsed={record['elapsed_s']:.1f}s",
                flush=True,
            )
    except BaseException:
        if executor is not None:
            _cancel_executor(executor, futures)
        raise
    else:
        assert executor is not None
        executor.shutdown(wait=True, cancel_futures=False)


def _assemble_raw(
    records: dict[tuple[int, int], PointRecord],
    *,
    run_identity: str,
    producer: fig7_paper.Fig7ProducerIdentity,
) -> tuple[dict[str, np.ndarray], dict[tuple[int, int], str]]:
    powers = tuple(float(value) for value in fig7_solve.P_READ_DBM)
    temperatures = tuple(float(value) for value in fig7_solve.T_BATH_VALUES)
    expected = {
        (pi, ti)
        for pi in range(len(powers))
        for ti in range(len(temperatures))
    }
    if set(records) != expected:
        raise RuntimeError("Fig. 7 point set is incomplete before assembly.")
    f_solved = np.empty(
        (len(powers), len(temperatures), fig7_solve.NUM_BINS),
        dtype=float,
    )
    certificates = {
        field: np.empty((len(powers), len(temperatures)), dtype=float)
        for field in NUMBER_CERTIFICATE_FIELDS
    }
    n_bar = np.full(len(powers), np.nan, dtype=float)
    payload_hashes: dict[tuple[int, int], str] = {}
    for (pi, ti), record in sorted(records.items()):
        path = Path(str(record["path"]))
        fresh = _point_record(
            path,
            pi=pi,
            ti=ti,
            power=powers[pi],
            temperature=temperatures[ti],
            run_identity=run_identity,
            producer=producer,
        )
        if fresh["sha256"] != record["sha256"]:
            raise RuntimeError(f"Fig. 7 point payload changed during assembly: {path}.")
        payload_hashes[(pi, ti)] = str(fresh["sha256"])
        with np.load(path, allow_pickle=False) as payload:
            f_solved[pi, ti] = payload["f_solved"][0, 0]
            current_n_bar = float(payload["n_bar"][0])
            if np.isnan(n_bar[pi]):
                n_bar[pi] = current_n_bar
            elif current_n_bar != n_bar[pi]:
                raise RuntimeError(f"Inconsistent Fig. 7 n_bar in {path}.")
            certificate = {
                field: float(payload[field][0, 0])
                for field in NUMBER_CERTIFICATE_FIELDS
            }
            fig7_solve._require_certified_point(
                certificate,
                context=f"resumed payload ({pi}, {ti})",
            )
            for field, value in certificate.items():
                certificates[field][pi, ti] = value
    raw = {
        "f_solved": f_solved,
        "temperatures": np.asarray(temperatures),
        "powers_dbm": np.asarray(powers),
        "n_bar": n_bar,
        "num_bins": np.asarray([fig7_solve.NUM_BINS]),
        **certificates,
    }
    fig7_paper._validated_raw_payload(raw)
    return raw, payload_hashes


def run_campaign(
    *,
    run_root: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, object]:
    if _NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT:
        raise RuntimeError(
            "NumPy was imported before the Fig. 7 runner established its "
            "single-thread environment. Launch this runner in a fresh process."
        )
    _assert_single_thread_environment()
    if isinstance(max_workers, bool) or not isinstance(max_workers, int):
        raise ValueError("max_workers must be a positive integer.")
    if max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    started = time.perf_counter()
    runner_path = Path(__file__).resolve()
    producer = fig7_paper.capture_producer_identity(runner_path)
    run_identity, run_dir = _resolve_run_dir(run_root, producer)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    prior_status = _read_prior_status(
        status_path,
        run_identity=run_identity,
        producer=producer,
    )
    started_utc = (
        str(prior_status["started_utc"])
        if prior_status is not None
        else datetime.now(UTC).isoformat()
    )
    if prior_status is None:
        attempt = 1
    else:
        prior_attempt = prior_status["attempt"]
        if isinstance(prior_attempt, bool) or not isinstance(prior_attempt, int):
            # _read_prior_status already enforces this; retain a local typed
            # guard so callers cannot bypass the validated status contract.
            raise RuntimeError("Fig. 7 prior status has invalid attempt.")
        attempt = prior_attempt + 1
    points = _all_points(run_dir, run_identity, producer)
    records: dict[tuple[int, int], PointRecord] = {}
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"RUN_IDENTITY={run_identity}", flush=True)
    print(f"SOLVE_DIGEST={producer.solve_contract_digest}", flush=True)
    print(f"OBSERVABLE_DIGEST={producer.observable_contract_digest}", flush=True)
    print(f"RUNNER_SHA256={producer.runner_sha256}", flush=True)
    campaign_lock = _acquire_campaign_lock(
        run_dir,
        run_identity=run_identity,
        producer=producer,
    )
    try:
        records = _scan_resume_points(
            run_dir,
            points,
            run_identity,
            producer,
        )
        running_status = _status_payload(
            state="running",
            run_identity=run_identity,
            producer=producer,
            started_utc=started_utc,
            attempt=attempt,
            completed=len(records),
        )
        _atomic_json(status_path, running_status)
        missing = _missing_points(points, records)
        print(
            f"RESUME_VALID={len(records)} MISSING={len(missing)}",
            flush=True,
        )
        _run_missing_points(
            missing,
            max_workers=max_workers,
            records=records,
            status_path=status_path,
            status_base=running_status,
        )
        fig7_paper.assert_producer_identity_current(producer, runner_path)
        raw, payload_hashes = _assemble_raw(
            records,
            run_identity=run_identity,
            producer=producer,
        )
        result = fig7_paper.observables(raw)
        wall_s = time.perf_counter() - started
        aggregate_worker_s = sum(
            record["elapsed_s"] for record in records.values()
        )
        csv_path, pdf_path, attestation_path = fig7_paper.publish_baseline_pair(
            result,
            producer=producer,
            runner_path=runner_path,
            worker_payload_hashes=payload_hashes,
            campaign={
                "mode": "parallel-independent-points",
                "run_identity": run_identity,
                "wall_s": wall_s,
                "aggregate_worker_s": aggregate_worker_s,
                "resumed_points": len(records) - len(missing),
                "new_points": len(missing),
            },
        )
        fig7_paper.assert_producer_identity_current(producer, runner_path)
        completion = {
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "attestation": str(attestation_path),
            "csv_sha256": fig7_paper.file_sha256(csv_path),
            "pdf_sha256": fig7_paper.file_sha256(pdf_path),
            "attestation_sha256": fig7_paper.file_sha256(attestation_path),
            "wall_s": wall_s,
            "aggregate_worker_s": aggregate_worker_s,
            "certificate_maxima": {
                field: float(np.max(np.asarray(raw[field])))
                for field in NUMBER_CERTIFICATE_FIELDS
            },
            "point_payload_sha256": {
                f"p{pi:02d}-t{ti:02d}": digest
                for (pi, ti), digest in sorted(payload_hashes.items())
            },
        }
        final_status = _status_payload(
            state="complete",
            run_identity=run_identity,
            producer=producer,
            started_utc=started_utc,
            attempt=attempt,
            completed=len(records),
            completion=completion,
        )
        _atomic_json(status_path, final_status)
        # Independently re-hash every worker payload, the status record, and
        # the promoted artifacts. This deliberately does not trust the hashes
        # copied into the attestation by the same publishing call.
        fig7_paper.validate_promotion_attestation(
            attestation_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            expected_producer=producer,
            run_dir=run_dir,
            status_path=status_path,
        )
        print(json.dumps(final_status, indent=2, sort_keys=True), flush=True)
        return final_status
    except BaseException as exc:
        failed_status = _status_payload(
            state="failed",
            run_identity=run_identity,
            producer=producer,
            started_utc=started_utc,
            attempt=attempt,
            completed=len(records),
            error=exc,
        )
        _atomic_json(status_path, failed_status)
        raise
    finally:
        active_error = sys.exc_info()[1]
        try:
            _release_campaign_lock(campaign_lock)
        except BaseException as release_error:
            if active_error is None:
                raise
            active_error.add_note(
                f"Additionally failed to release Fig. 7 campaign lock: "
                f"{release_error}"
            )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "qpsim-fig7-runs",
        help=(
            "Parent directory for the deterministic fig7-<identity> run "
            "directory (default: system temp/qpsim-fig7-runs)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Independent point workers (default: {DEFAULT_MAX_WORKERS}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    run_campaign(
        run_root=cast(Path, args.run_root),
        max_workers=cast(int, args.max_workers),
    )


if __name__ == "__main__":
    main()
