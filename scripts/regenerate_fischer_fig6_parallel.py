# ruff: noqa: E402, I001
"""Guarded, resumable parallel regeneration of Fischer 2023 Fig. 6.

Each worker owns one complete bath-temperature row and sweeps the canonical
``n_bar`` axis low-to-high in the same continuation order as the serial
producer.  The three rows are physically independent; individual points are
not.  Restarting strictly validates atomic row payloads and runs only missing
rows before deterministically merging and recertifying the full raw state.

Run from the repository root::

    python -m scripts.regenerate_fischer_fig6_parallel \
        --run-root C:\\tmp\\qpsim-fig6-runs --max-workers 3
"""

from __future__ import annotations

import os
import sys

# Establish numerical-thread controls before importing NumPy/SciPy/qpsim.
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
import tempfile
import time
import zipfile
from collections.abc import Iterator, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast

import numpy as np

from validation.fischer_2023 import fig6_paper, fig6_solve
from validation.source_provenance import source_sha256

STATUS_SCHEMA = "qpsim-fischer-2023-fig6-run-v1"
ROW_SCHEMA = "qpsim-fischer-2023-fig6-row-v1"
DEFAULT_MAX_WORKERS = 3

_RAW_ARRAY_KEYS = (
    "tau_0_pb_ns",
    "tau_l_ns",
    "T_bath",
    "n_bar",
    "T_star_over_delta",
    "delta_eq",
    "delta_driven",
    "paper_observable_num",
    "paper_observable_eq53",
    "x_qp_num",
    "x_qp_eq47",
    "state_f",
    "state_n_ph",
    *fig6_solve.FIG6_CERTIFICATE_FIELDS,
)


class RowRecord(TypedDict):
    """Identity and timing for one independently revalidated row payload."""

    row_index: int
    temperature_K: float
    elapsed_s: float
    path: str
    sha256: str
    size_bytes: int


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                dict(payload),
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


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _runtime_identity() -> dict[str, object]:
    return fig6_paper.generation_runtime_identity()


def _runner_record(runner_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    resolved = runner_path.resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": source_sha256(resolved),
    }


def _producer_base(runner_path: Path) -> dict[str, object]:
    runner = _runner_record(runner_path)
    runtime = _runtime_identity()
    fingerprint = fig6_paper.artifact_fingerprint()
    run_identity = fig6_paper.generation_run_identity(
        artifact_fingerprint=fingerprint,
        mode="parallel-temperature-rows",
        runner=runner,
        runtime=runtime,
        single_thread_environment=SINGLE_THREAD_ENVIRONMENT,
    )
    return {
        "artifact_fingerprint": fingerprint,
        "run_identity": run_identity,
        "run_identity_schema": fig6_paper.GENERATION_EVIDENCE_SCHEMA,
        "runner": runner,
        "runtime": runtime,
        "single_thread_environment": dict(SINGLE_THREAD_ENVIRONMENT),
    }


def _producer_sha256(producer: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(producer)).encode("utf-8")).hexdigest()


def _assert_single_thread_environment() -> None:
    actual = {
        name: os.environ.get(name) for name in SINGLE_THREAD_ENVIRONMENT
    }
    if actual != SINGLE_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "Fig. 6 campaign lost its required single-thread environment: "
            f"{actual!r}."
        )


def _assert_producer_current(
    producer: Mapping[str, object],
    runner_path: Path,
) -> None:
    if _producer_base(runner_path) != dict(producer):
        raise RuntimeError(
            "Fischer Fig. 6 source, runner, configuration, or runtime changed "
            "during generation; discard the candidate and restart."
        )


def _run_dir(run_root: Path, producer: Mapping[str, object]) -> Path:
    identity = cast(str, producer["run_identity"])
    return run_root.expanduser().resolve() / f"fig6-{identity}"


def _row_path(run_dir: Path, row_index: int) -> Path:
    return run_dir / f"row-t{row_index:02d}.npz"


def _row_metadata(
    *,
    row_index: int,
    temperature: float,
    producer: Mapping[str, object],
) -> dict[str, object]:
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "n_bar_values": [float(value) for value in fig6_solve.N_BAR_VALUES],
        "num_bins": int(fig6_solve.NUM_BINS),
        "producer_sha256": _producer_sha256(producer),
        "row_index": row_index,
        "run_identity": producer["run_identity"],
        "schema": ROW_SCHEMA,
        "temperature_K": temperature,
    }


_ROW_METADATA_FIELDS = {
    "created_utc",
    "n_bar_values",
    "num_bins",
    "producer_sha256",
    "row_index",
    "run_identity",
    "schema",
    "temperature_K",
}


def _load_row_arrays(
    path: Path,
    *,
    row_index: int,
    temperature: float,
    producer: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], float]:
    expected_names = {*_RAW_ARRAY_KEYS, "elapsed_s", "metadata_json"}
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_names:
                raise RuntimeError(
                    f"Fig. 6 row payload {path} has missing or unexpected arrays."
                )
            metadata_array = np.asarray(payload["metadata_json"])
            elapsed_array = np.asarray(payload["elapsed_s"], dtype=float)
            if metadata_array.shape != () or elapsed_array.shape != (1,):
                raise RuntimeError(f"Fig. 6 row payload {path} has malformed metadata.")
            metadata = json.loads(str(metadata_array.item()))
            raw = {
                name: np.asarray(payload[name]).copy()
                for name in _RAW_ARRAY_KEYS
            }
            elapsed = float(elapsed_array[0])
    except RuntimeError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as exc:
        raise RuntimeError(f"Cannot validate Fig. 6 row payload {path}.") from exc
    if not isinstance(metadata, dict) or set(metadata) != _ROW_METADATA_FIELDS:
        raise RuntimeError(f"Fig. 6 row payload {path} has invalid metadata.")
    expected_metadata = {
        "n_bar_values": [float(value) for value in fig6_solve.N_BAR_VALUES],
        "num_bins": int(fig6_solve.NUM_BINS),
        "producer_sha256": _producer_sha256(producer),
        "row_index": row_index,
        "run_identity": producer["run_identity"],
        "schema": ROW_SCHEMA,
        "temperature_K": temperature,
    }
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"Fig. 6 row payload {path} has wrong {field!r}; "
                "resume refuses foreign or stale work."
            )
    if not isinstance(metadata["created_utc"], str) or not metadata["created_utc"]:
        raise RuntimeError(f"Fig. 6 row payload {path} has invalid creation time.")
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError(f"Fig. 6 row payload {path} has invalid elapsed time.")
    fig6_paper.validate_temperature_row_payload(raw, T_bath=temperature)
    return raw, elapsed


def _row_record(
    path: Path,
    *,
    row_index: int,
    temperature: float,
    producer: Mapping[str, object],
) -> RowRecord:
    _, elapsed = _load_row_arrays(
        path,
        row_index=row_index,
        temperature=temperature,
        producer=producer,
    )
    return RowRecord(
        row_index=row_index,
        temperature_K=temperature,
        elapsed_s=elapsed,
        path=str(path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _solve_one_row(
    args: tuple[int, float, str, str, dict[str, object]],
) -> RowRecord:
    row_index, temperature, run_dir_text, runner_text, producer = args
    _assert_single_thread_environment()
    runner_path = Path(runner_text)
    _assert_producer_current(producer, runner_path)
    started = time.perf_counter()
    raw = fig6_solve.solve_temperature_row(temperature)
    fig6_paper.validate_temperature_row_payload(raw, T_bath=temperature)
    _assert_producer_current(producer, runner_path)
    elapsed = time.perf_counter() - started
    metadata = _row_metadata(
        row_index=row_index,
        temperature=temperature,
        producer=producer,
    )
    arrays = {
        name: np.asarray(raw[name])
        for name in _RAW_ARRAY_KEYS
    }
    arrays["elapsed_s"] = np.asarray([elapsed], dtype=float)
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata))
    path = _row_path(Path(run_dir_text), row_index)
    _atomic_npz(path, arrays)
    return _row_record(
        path,
        row_index=row_index,
        temperature=temperature,
        producer=producer,
    )


@contextmanager
def _campaign_lock(run_dir: Path) -> Iterator[None]:
    """Hold one crash-safe OS lock for the deterministic run directory."""
    lock_path = run_dir / ".campaign.lock"
    stream = lock_path.open("a+b")
    try:
        stream.seek(0)
        if stream.read(1) == b"":
            stream.seek(0)
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
            )
    except OSError as exc:
        stream.close()
        raise RuntimeError(
            f"Another Fig. 6 campaign holds {lock_path}; refusing concurrent work."
        ) from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(  # type: ignore[attr-defined]
                stream.fileno(),
                fcntl.LOCK_UN,  # type: ignore[attr-defined]
            )
        stream.close()


def _scan_rows(
    run_dir: Path,
    producer: Mapping[str, object],
) -> dict[int, RowRecord]:
    temperatures = tuple(float(value) for value in fig6_solve.T_BATH_VALUES)
    expected_paths = {
        _row_path(run_dir, index).resolve()
        for index in range(len(temperatures))
    }
    for temporary in run_dir.glob(".row-t*.npz.*.tmp"):
        temporary.unlink(missing_ok=True)
    actual_paths = {path.resolve() for path in run_dir.glob("*.npz")}
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise RuntimeError(
            f"Fig. 6 run directory contains unexpected row payloads: {unexpected}."
        )
    records: dict[int, RowRecord] = {}
    for index, temperature in enumerate(temperatures):
        path = _row_path(run_dir, index)
        if path.exists():
            records[index] = _row_record(
                path,
                row_index=index,
                temperature=temperature,
                producer=producer,
            )
    return records


def _cancel_executor(
    executor: ProcessPoolExecutor,
    futures: list[Future[RowRecord]],
) -> None:
    for future in futures:
        future.cancel()
    processes = getattr(executor, "_processes", {})
    if isinstance(processes, dict):
        for process in processes.values():
            process.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def _run_missing_rows(
    missing: list[int],
    *,
    max_workers: int,
    run_dir: Path,
    runner_path: Path,
    producer: dict[str, object],
    records: dict[int, RowRecord],
    status_path: Path,
    status_base: dict[str, object],
) -> None:
    if not missing:
        return
    temperatures = tuple(float(value) for value in fig6_solve.T_BATH_VALUES)
    context = multiprocessing.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    futures: list[Future[RowRecord]] = []
    try:
        executor = ProcessPoolExecutor(
            max_workers=min(max_workers, len(missing)),
            mp_context=context,
        )
        for index in missing:
            args = (
                index,
                temperatures[index],
                str(run_dir),
                str(runner_path),
                producer,
            )
            futures.append(executor.submit(_solve_one_row, args))
        for future in as_completed(futures):
            record = future.result()
            index = record["row_index"]
            if index in records:
                raise RuntimeError(f"Fig. 6 worker returned duplicate row {index}.")
            records[index] = record
            status = dict(status_base)
            status["completed_rows"] = len(records)
            status["updated_utc"] = datetime.now(UTC).isoformat()
            _atomic_json(status_path, status)
            print(
                f"DONE {len(records):02d}/{len(temperatures):02d} "
                f"T={record['temperature_K']:g} K "
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
    records: Mapping[int, RowRecord],
    *,
    run_dir: Path,
    producer: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    temperatures = tuple(float(value) for value in fig6_solve.T_BATH_VALUES)
    if set(records) != set(range(len(temperatures))):
        raise RuntimeError("Fig. 6 row set is incomplete before assembly.")
    rows: list[dict[str, np.ndarray]] = []
    evidence: dict[str, dict[str, object]] = {}
    for index, temperature in enumerate(temperatures):
        record = records[index]
        path = Path(record["path"])
        expected_path = _row_path(run_dir, index).resolve()
        if path.resolve() != expected_path:
            raise RuntimeError(
                f"Fig. 6 row {index} points outside its deterministic run path."
            )
        raw, _ = _load_row_arrays(
            path,
            row_index=index,
            temperature=temperature,
            producer=producer,
        )
        current_hash = _sha256_file(path)
        if current_hash != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Fig. 6 row payload changed during assembly: {path}.")
        rows.append(raw)
        evidence[f"t{index:02d}"] = {
            "semantic_sha256": fig6_paper.result_row_sha256(
                fig6_paper.observables(raw),
                0,
            ),
            "temperature_K": temperature,
        }

    scalar_keys = ("tau_0_pb_ns", "tau_l_ns", "n_bar")
    for key in scalar_keys:
        for row in rows[1:]:
            if not np.array_equal(row[key], rows[0][key]):
                raise RuntimeError(f"Fig. 6 rows disagree on {key!r}.")
    raw = {
        "tau_0_pb_ns": rows[0]["tau_0_pb_ns"].copy(),
        "tau_l_ns": rows[0]["tau_l_ns"].copy(),
        "T_bath": np.asarray(temperatures, dtype=float),
        "n_bar": rows[0]["n_bar"].copy(),
    }
    vector_keys = ("delta_eq",)
    matrix_keys = tuple(
        key
        for key in _RAW_ARRAY_KEYS
        if key not in {*scalar_keys, "T_bath", *vector_keys}
    )
    for key in vector_keys:
        raw[key] = np.concatenate([row[key] for row in rows], axis=0)
    for key in matrix_keys:
        raw[key] = np.concatenate([row[key] for row in rows], axis=0)
    result = fig6_paper.observables(raw)
    fig6_paper.validate_artifact_result(result)
    return raw, evidence


def _read_prior_status(
    path: Path,
    *,
    producer: Mapping[str, object],
) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fig. 6 run status at {path} is corrupt.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Fig. 6 run status at {path} is malformed.")
    required = {
        "attempt",
        "completed_rows",
        "producer_sha256",
        "run_identity",
        "schema",
        "started_utc",
        "state",
        "total_rows",
        "updated_utc",
    }
    allowed = set(required)
    if payload.get("state") == "failed":
        allowed.add("error")
    elif payload.get("state") == "complete":
        allowed.add("completion")
    if set(payload) != allowed:
        raise RuntimeError(f"Fig. 6 run status at {path} has invalid fields.")
    expected = {
        "producer_sha256": _producer_sha256(producer),
        "run_identity": producer["run_identity"],
        "schema": STATUS_SCHEMA,
        "total_rows": len(fig6_solve.T_BATH_VALUES),
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise RuntimeError(
                f"Fig. 6 run status at {path} has wrong {field!r}."
            )
    for field in ("attempt", "completed_rows"):
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Fig. 6 run status at {path} has invalid {field!r}.")
    if payload["attempt"] < 1 or payload["completed_rows"] > payload["total_rows"]:
        raise RuntimeError(f"Fig. 6 run status at {path} has invalid counts.")
    if payload["state"] not in {"running", "failed", "complete"}:
        raise RuntimeError(f"Fig. 6 run status at {path} has invalid state.")
    return cast(dict[str, object], payload)


def _status(
    *,
    state: str,
    producer: Mapping[str, object],
    started_utc: str,
    attempt: int,
    completed_rows: int,
    error: BaseException | None = None,
    completion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt": attempt,
        "completed_rows": completed_rows,
        "producer_sha256": _producer_sha256(producer),
        "run_identity": producer["run_identity"],
        "schema": STATUS_SCHEMA,
        "started_utc": started_utc,
        "state": state,
        "total_rows": len(fig6_solve.T_BATH_VALUES),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        payload["error"] = {"message": str(error), "type": type(error).__name__}
    if completion is not None:
        payload["completion"] = dict(completion)
    return payload


def run_campaign(
    *,
    run_root: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, object]:
    """Resume/solve three rows, merge canonically, recertify, and publish."""
    if _NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT:
        raise RuntimeError(
            "NumPy was imported before the Fig. 6 runner established its "
            "single-thread environment. Launch this module in a fresh process."
        )
    _assert_single_thread_environment()
    row_count = len(fig6_solve.T_BATH_VALUES)
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= row_count
    ):
        raise ValueError(f"max_workers must be an integer in [1, {row_count}].")

    started_clock = time.perf_counter()
    runner_path = Path(__file__).resolve()
    producer = _producer_base(runner_path)
    run_dir = _run_dir(run_root, producer)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    records: dict[int, RowRecord] = {}
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"RUN_IDENTITY={producer['run_identity']}", flush=True)
    print(f"RUNNER_SHA256={cast(dict[str, object], producer['runner'])['sha256']}", flush=True)

    with _campaign_lock(run_dir):
        prior = _read_prior_status(status_path, producer=producer)
        started_utc = (
            str(prior["started_utc"])
            if prior is not None
            else datetime.now(UTC).isoformat()
        )
        attempt = 1 if prior is None else cast(int, prior["attempt"]) + 1
        try:
            records = _scan_rows(run_dir, producer)
            missing = [
                index for index in range(row_count) if index not in records
            ]
            running = _status(
                state="running",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed_rows=len(records),
            )
            _atomic_json(status_path, running)
            print(
                f"RESUME_VALID={len(records)} MISSING={len(missing)}",
                flush=True,
            )
            _run_missing_rows(
                missing,
                max_workers=max_workers,
                run_dir=run_dir,
                runner_path=runner_path,
                producer=producer,
                records=records,
                status_path=status_path,
                status_base=running,
            )
            _assert_producer_current(producer, runner_path)
            raw, worker_evidence = _assemble_raw(
                records,
                run_dir=run_dir,
                producer=producer,
            )
            result = fig6_paper.observables(raw)
            wall_s = time.perf_counter() - started_clock
            aggregate_worker_s = sum(
                record["elapsed_s"] for record in records.values()
            )
            generation = {
                "artifact_fingerprint": producer["artifact_fingerprint"],
                "campaign": {
                    "aggregate_worker_s": aggregate_worker_s,
                    "new_rows": len(missing),
                    "resumed_rows": row_count - len(missing),
                    "wall_s": wall_s,
                },
                "mode": "parallel-temperature-rows",
                "run_identity": producer["run_identity"],
                "run_identity_schema": producer["run_identity_schema"],
                "runner": producer["runner"],
                "runtime": producer["runtime"],
                "schema": fig6_paper.GENERATION_EVIDENCE_SCHEMA,
                "single_thread_environment": producer[
                    "single_thread_environment"
                ],
                "worker_payloads": worker_evidence,
            }
            fig6_paper.validate_generation_evidence(generation)
            csv_path, pdf_path, record_path = fig6_paper.publish_artifacts(
                result,
                generation_evidence=generation,
            )
            _assert_producer_current(producer, runner_path)
            fig6_paper.read_promotion_record(
                record_path,
                csv_path=csv_path,
                pdf_path=pdf_path,
            )
            completion = {
                "aggregate_worker_s": aggregate_worker_s,
                "certificate_maxima": {
                    field: float(np.max(np.asarray(raw[field], dtype=float)))
                    for field in fig6_solve.FIG6_CERTIFICATE_FIELDS
                },
                "csv": str(csv_path),
                "csv_sha256": _sha256_file(csv_path),
                "pdf": str(pdf_path),
                "pdf_sha256": _sha256_file(pdf_path),
                "promotion_record": str(record_path),
                "promotion_record_sha256": _sha256_file(record_path),
                "row_archives": {
                    f"t{index:02d}": {
                        "sha256": records[index]["sha256"],
                        "size_bytes": records[index]["size_bytes"],
                        "temperature_K": records[index]["temperature_K"],
                    }
                    for index in sorted(records)
                },
                "wall_s": wall_s,
                "worker_payloads": worker_evidence,
            }
            final = _status(
                state="complete",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed_rows=len(records),
                completion=completion,
            )
            _atomic_json(status_path, final)
            print(json.dumps(final, indent=2, sort_keys=True), flush=True)
            return final
        except BaseException as exc:
            failed = _status(
                state="failed",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed_rows=len(records),
                error=exc,
            )
            _atomic_json(status_path, failed)
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "qpsim-fig6-runs",
        help=(
            "Parent of the deterministic fig6-<identity> run directory "
            "(default: system temp/qpsim-fig6-runs)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Independent temperature-row workers (default: {DEFAULT_MAX_WORKERS}).",
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
