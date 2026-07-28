# ruff: noqa: E402, I001
"""Guarded, resumable parallel regeneration of Fischer 2023 Fig. 5.

The six continuation rows are independent:

* three upper-panel bath temperatures, each sweeping the complete canonical
  ``n_bar`` axis low-to-high; and
* three lower-panel fixed drives, each sweeping the complete canonical bath
  temperature axis low-to-high.

Continuation remains serial inside each row. Workers persist atomic,
identity-bound row payloads; the parent strictly revalidates every row,
deterministically assembles the canonical raw payload, and hands the result to
the existing transactional Fig. 5 publisher.

Run from the repository root::

    python -m scripts.regenerate_fischer_fig5_parallel \
        --run-root C:\\tmp\\qpsim-fig5-runs --max-workers 6
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
from qpsim.collisions.phonon import build_phonon_frequency_map

from validation.fischer_2023 import fig5_paper, fig5_solve
from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_FIELDS,
)
from validation.source_provenance import source_sha256

STATUS_SCHEMA = "qpsim-fischer-2023-fig5-run-v1"
ROW_SCHEMA = "qpsim-fischer-2023-fig5-row-v1"
DEFAULT_MAX_WORKERS = 6

_COMMON_KEYS = ("tau_0_pb_ns", "tau_l_ns", "num_bins")
_UPPER_KEYS = (
    "upper_f",
    "upper_n_ph",
    "upper_T_bath",
    "upper_nbar",
    *(f"upper_{field}" for field in NUMBER_CERTIFICATE_FIELDS),
)
_LOWER_KEYS = (
    "lower_f",
    "lower_n_ph",
    "lower_nbar",
    "lower_T_bath",
    *(f"lower_{field}" for field in NUMBER_CERTIFICATE_FIELDS),
)
RAW_ARRAY_KEYS = (*_COMMON_KEYS, *_UPPER_KEYS, *_LOWER_KEYS)


@dataclass(frozen=True)
class RowSpec:
    """Deterministic identity for one continuation row."""

    ordinal: int
    panel: str
    row_index: int
    axis_value: float

    @property
    def label(self) -> str:
        prefix = "upper-t" if self.panel == "upper" else "lower-n"
        return f"{prefix}{self.row_index:02d}"


class RowRecord(TypedDict):
    """Identity and timing for one independently revalidated row payload."""

    ordinal: int
    label: str
    panel: str
    row_index: int
    axis_value: float
    elapsed_s: float
    path: str
    sha256: str
    size_bytes: int


def _row_specs() -> tuple[RowSpec, ...]:
    upper = tuple(
        RowSpec(index, "upper", index, float(value))
        for index, value in enumerate(fig5_solve.UPPER_T_BATH_K)
    )
    offset = len(upper)
    lower = tuple(
        RowSpec(offset + index, "lower", index, float(value))
        for index, value in enumerate(fig5_solve.LOWER_NBAR)
    )
    return (*upper, *lower)


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


def _runner_record(runner_path: Path) -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    resolved = runner_path.resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "sha256": source_sha256(resolved),
    }


def _producer_base(runner_path: Path) -> dict[str, object]:
    return {
        "artifact_producer": fig5_paper.capture_producer_identity(),
        "mode": "parallel-independent-continuation-rows",
        "runner": _runner_record(runner_path),
        "single_thread_environment": dict(SINGLE_THREAD_ENVIRONMENT),
    }


def _producer_sha256(producer: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(producer)).encode("utf-8")).hexdigest()


def _run_identity(producer: Mapping[str, object]) -> str:
    payload = {
        "producer": dict(producer),
        "row_schema": ROW_SCHEMA,
        "status_schema": STATUS_SCHEMA,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _assert_single_thread_environment() -> None:
    actual = {
        name: os.environ.get(name) for name in SINGLE_THREAD_ENVIRONMENT
    }
    if actual != SINGLE_THREAD_ENVIRONMENT:
        raise RuntimeError(
            "Fig. 5 campaign lost its required single-thread environment: "
            f"{actual!r}."
        )


def _assert_producer_current(
    producer: Mapping[str, object],
    runner_path: Path,
) -> None:
    if _producer_base(runner_path) != dict(producer):
        raise RuntimeError(
            "Fischer Fig. 5 source, runner, configuration, or runtime changed "
            "during generation; discard the candidate and restart."
        )


def _run_dir(run_root: Path, producer: Mapping[str, object]) -> Path:
    return run_root.expanduser().resolve() / f"fig5-{_run_identity(producer)}"


def _row_path(run_dir: Path, spec: RowSpec) -> Path:
    return run_dir / f"row-{spec.label}.npz"


def _campaign_record_path() -> Path:
    """Durable evidence binding the parallel rows to the promoted bundle."""
    promotion = fig5_paper.promotion_record_path()
    suffix = ".promotion.json"
    if not promotion.name.endswith(suffix):
        raise RuntimeError(
            "Fig. 5 promotion record has an unexpected name; refusing to "
            "derive the campaign-evidence path."
        )
    return promotion.with_name(
        f"{promotion.name.removesuffix(suffix)}.campaign.json"
    )


def _row_metadata(
    *,
    spec: RowSpec,
    producer: Mapping[str, object],
) -> dict[str, object]:
    continuation_axis = (
        [float(value) for value in fig5_solve.UPPER_NBAR_VALUES]
        if spec.panel == "upper"
        else [float(value) for value in fig5_solve.LOWER_T_BATH_K]
    )
    return {
        "axis_value": spec.axis_value,
        "continuation_axis": continuation_axis,
        "created_utc": datetime.now(UTC).isoformat(),
        "label": spec.label,
        "num_bins": int(fig5_solve.NUM_BINS),
        "ordinal": spec.ordinal,
        "panel": spec.panel,
        "producer_sha256": _producer_sha256(producer),
        "row_index": spec.row_index,
        "run_identity": _run_identity(producer),
        "schema": ROW_SCHEMA,
    }


_ROW_METADATA_FIELDS = {
    "axis_value",
    "continuation_axis",
    "created_utc",
    "label",
    "num_bins",
    "ordinal",
    "panel",
    "producer_sha256",
    "row_index",
    "run_identity",
    "schema",
}


def _validate_row_payload(
    raw: Mapping[str, np.ndarray],
    *,
    spec: RowSpec,
) -> fig5_paper.Fig5PaperResult:
    """Independently re-certify one row without changing Fig. 5 sources."""
    raw_num_bins = np.asarray(raw["num_bins"])
    if (
        raw_num_bins.shape != (1,)
        or np.issubdtype(raw_num_bins.dtype, np.bool_)
        or int(raw_num_bins[0]) != fig5_solve.NUM_BINS
        or float(raw_num_bins[0]) != float(fig5_solve.NUM_BINS)
    ):
        raise RuntimeError("Fig. 5 continuation row has the wrong grid size.")
    result = fig5_paper.observables(raw)

    expected_tau = fig5_paper.config_metadata().tau_0_pb_ns
    tau_l_raw = np.asarray(raw["tau_l_ns"], dtype=float)
    if (
        tau_l_raw.shape != (1,)
        or not np.isfinite(result.tau_0_pb_ns)
        or result.tau_0_pb_ns <= 0.0
        or not np.isclose(
            result.tau_0_pb_ns,
            expected_tau,
            rtol=1.0e-14,
            atol=0.0,
        )
        or not np.array_equal(tau_l_raw, np.asarray([result.tau_0_pb_ns]))
    ):
        raise RuntimeError("Fig. 5 continuation row has stale lifetime normalization.")

    canonical_upper_T = np.asarray(fig5_solve.UPPER_T_BATH_K, dtype=float)
    canonical_upper_nbar = np.asarray(fig5_solve.UPPER_NBAR_VALUES, dtype=float)
    canonical_lower_nbar = np.asarray(fig5_solve.LOWER_NBAR, dtype=float)
    canonical_lower_T = np.asarray(fig5_solve.LOWER_T_BATH_K, dtype=float)
    if spec.panel == "upper":
        expected_upper_T = canonical_upper_T[spec.row_index : spec.row_index + 1]
        expected_upper_nbar = canonical_upper_nbar
        expected_lower_nbar = np.asarray([], dtype=float)
        expected_lower_T = np.asarray([], dtype=float)
        point_temperatures = np.repeat(expected_upper_T, expected_upper_nbar.size)
        point_drives = expected_upper_nbar
    else:
        expected_upper_T = np.asarray([], dtype=float)
        expected_upper_nbar = np.asarray([], dtype=float)
        expected_lower_nbar = canonical_lower_nbar[
            spec.row_index : spec.row_index + 1
        ]
        expected_lower_T = canonical_lower_T
        point_temperatures = expected_lower_T
        point_drives = np.repeat(expected_lower_nbar, expected_lower_T.size)
    axes = (
        ("upper T_bath", result.upper_T_bath, expected_upper_T),
        ("upper n_bar", result.upper_nbar, expected_upper_nbar),
        ("lower n_bar", result.lower_nbar, expected_lower_nbar),
        ("lower T_bath", result.lower_T_bath, expected_lower_T),
    )
    for name, actual, expected in axes:
        if not np.array_equal(np.asarray(actual, dtype=float), expected):
            raise RuntimeError(f"Fig. 5 continuation row has the wrong {name} axis.")

    _, _, spectral = fig5_solve._build_grid_and_spectral(fig5_solve.NUM_BINS)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    upper_shape = (expected_upper_T.size, expected_upper_nbar.size)
    lower_shape = (expected_lower_nbar.size, expected_lower_T.size)
    upper_f = fig5_paper._required_state_array(
        result,
        "upper_f",
        (*upper_shape, fig5_solve.NUM_BINS),
        upper_bound=1.0,
    )
    lower_f = fig5_paper._required_state_array(
        result,
        "lower_f",
        (*lower_shape, fig5_solve.NUM_BINS),
        upper_bound=1.0,
    )
    upper_n_ph = fig5_paper._required_state_array(
        result,
        "upper_n_ph",
        (*upper_shape, omega.size),
    )
    lower_n_ph = fig5_paper._required_state_array(
        result,
        "lower_n_ph",
        (*lower_shape, omega.size),
    )
    certificates = fig5_paper._certificate_arrays(result, spec.panel)
    expected_shape = upper_shape if spec.panel == "upper" else lower_shape
    for field, values in certificates.items():
        if (
            values.shape != expected_shape
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise RuntimeError(
                f"Fig. 5 continuation certificate {field!r} is malformed."
            )
        if (
            field in fig5_paper._CERTIFIED_BACKWARD_ERROR_FIELDS
            and values.size
            and float(np.max(values)) > fig5_solve.TARGET_BACKWARD_ERROR_LIMIT
        ):
            raise RuntimeError(
                f"Fig. 5 continuation certificate {field!r} exceeds its limit."
            )

    f_rows = upper_f[0] if spec.panel == "upper" else lower_f[0]
    n_ph_rows = upper_n_ph[0] if spec.panel == "upper" else lower_n_ph[0]
    x_num_rows = (
        np.asarray(result.upper_x_qp_num)
        if spec.panel == "upper"
        else np.asarray(result.lower_x_qp_num)
    )[0]
    x_analytic_rows = (
        np.asarray(result.upper_x_qp_analytic)
        if spec.panel == "upper"
        else np.asarray(result.lower_x_qp_analytic)
    )[0]
    for point_index, (T_bath, n_bar) in enumerate(
        zip(point_temperatures, point_drives, strict=True)
    ):
        x_num, x_analytic, certificate = fig5_paper._recomputed_point(
            f=f_rows[point_index],
            n_ph=n_ph_rows[point_index],
            T_bath=float(T_bath),
            n_bar=float(n_bar),
            tau_l=result.tau_0_pb_ns,
            spectral=spectral,
        )
        fig5_paper._require_close(
            x_num_rows[point_index],
            x_num,
            name=f"{spec.label} x_qp_num",
        )
        fig5_paper._require_close(
            x_analytic_rows[point_index],
            x_analytic,
            name=f"{spec.label} x_qp_analytic",
        )
        for field in NUMBER_CERTIFICATE_FIELDS:
            fig5_paper._require_close(
                certificates[field][0, point_index],
                certificate[field],
                name=f"{spec.label} {field}",
            )
    return result


def _load_row_arrays(
    path: Path,
    *,
    spec: RowSpec,
    producer: Mapping[str, object],
) -> tuple[dict[str, np.ndarray], float]:
    expected_names = {*RAW_ARRAY_KEYS, "elapsed_s", "metadata_json"}
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != expected_names:
                raise RuntimeError(
                    f"Fig. 5 row payload {path} has missing or unexpected arrays."
                )
            metadata_array = np.asarray(payload["metadata_json"])
            elapsed_array = np.asarray(payload["elapsed_s"], dtype=float)
            if metadata_array.shape != () or elapsed_array.shape != (1,):
                raise RuntimeError(f"Fig. 5 row payload {path} has malformed metadata.")
            metadata = json.loads(str(metadata_array.item()))
            raw = {
                name: np.asarray(payload[name]).copy()
                for name in RAW_ARRAY_KEYS
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
        raise RuntimeError(f"Cannot validate Fig. 5 row payload {path}.") from exc

    if not isinstance(metadata, dict) or set(metadata) != _ROW_METADATA_FIELDS:
        raise RuntimeError(f"Fig. 5 row payload {path} has invalid metadata.")
    expected_metadata = _row_metadata(spec=spec, producer=producer)
    expected_metadata.pop("created_utc")
    for field, expected in expected_metadata.items():
        if metadata.get(field) != expected:
            raise RuntimeError(
                f"Fig. 5 row payload {path} has wrong {field!r}; "
                "resume refuses foreign or stale work."
            )
    if not isinstance(metadata["created_utc"], str) or not metadata["created_utc"]:
        raise RuntimeError(f"Fig. 5 row payload {path} has invalid creation time.")
    if not np.isfinite(elapsed) or elapsed < 0.0:
        raise RuntimeError(f"Fig. 5 row payload {path} has invalid elapsed time.")
    _validate_row_payload(raw, spec=spec)
    return raw, elapsed


def _row_record(
    path: Path,
    *,
    spec: RowSpec,
    producer: Mapping[str, object],
) -> RowRecord:
    _, elapsed = _load_row_arrays(
        path,
        spec=spec,
        producer=producer,
    )
    return RowRecord(
        ordinal=spec.ordinal,
        label=spec.label,
        panel=spec.panel,
        row_index=spec.row_index,
        axis_value=spec.axis_value,
        elapsed_s=elapsed,
        path=str(path),
        sha256=_sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _solve_one_row(
    args: tuple[RowSpec, str, str, dict[str, object]],
) -> RowRecord:
    spec, run_dir_text, runner_text, producer = args
    _assert_single_thread_environment()
    runner_path = Path(runner_text)
    _assert_producer_current(producer, runner_path)
    started = time.perf_counter()
    if spec.panel == "upper":
        raw = fig5_solve.solve(
            upper_T_bath=(spec.axis_value,),
            upper_nbar=fig5_solve.UPPER_NBAR_VALUES,
            lower_nbar=(),
            lower_T_bath=(),
        )
    else:
        raw = fig5_solve.solve(
            upper_T_bath=(),
            upper_nbar=(),
            lower_nbar=(spec.axis_value,),
            lower_T_bath=fig5_solve.LOWER_T_BATH_K,
        )
    _validate_row_payload(raw, spec=spec)
    _assert_producer_current(producer, runner_path)
    elapsed = time.perf_counter() - started
    metadata = _row_metadata(spec=spec, producer=producer)
    arrays = {
        name: np.asarray(raw[name])
        for name in RAW_ARRAY_KEYS
    }
    arrays["elapsed_s"] = np.asarray([elapsed], dtype=float)
    arrays["metadata_json"] = np.asarray(_canonical_json(metadata))
    path = _row_path(Path(run_dir_text), spec)
    _atomic_npz(path, arrays)
    return _row_record(path, spec=spec, producer=producer)


@contextmanager
def _campaign_lock(run_dir: Path) -> Iterator[None]:
    """Hold one crash-safe OS lock for the deterministic run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
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
            f"Another Fig. 5 campaign holds {lock_path}; refusing concurrent work."
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
    specs = _row_specs()
    expected_paths = {
        _row_path(run_dir, spec).resolve()
        for spec in specs
    }
    for temporary in run_dir.glob(".row-*.npz.*.tmp"):
        temporary.unlink(missing_ok=True)
    actual_paths = {path.resolve() for path in run_dir.glob("*.npz")}
    unexpected = sorted(actual_paths - expected_paths)
    if unexpected:
        raise RuntimeError(
            f"Fig. 5 run directory contains unexpected row payloads: {unexpected}."
        )
    records: dict[int, RowRecord] = {}
    for spec in specs:
        path = _row_path(run_dir, spec)
        if path.exists():
            records[spec.ordinal] = _row_record(
                path,
                spec=spec,
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
    missing: list[RowSpec],
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
    context = multiprocessing.get_context("spawn")
    executor: ProcessPoolExecutor | None = None
    futures: list[Future[RowRecord]] = []
    try:
        executor = ProcessPoolExecutor(
            max_workers=min(max_workers, len(missing)),
            mp_context=context,
        )
        for spec in missing:
            args = (
                spec,
                str(run_dir),
                str(runner_path),
                producer,
            )
            futures.append(executor.submit(_solve_one_row, args))
        for future in as_completed(futures):
            record = future.result()
            ordinal = record["ordinal"]
            if ordinal in records:
                raise RuntimeError(f"Fig. 5 worker returned duplicate row {ordinal}.")
            records[ordinal] = record
            status = dict(status_base)
            status["completed_rows"] = len(records)
            status["updated_utc"] = datetime.now(UTC).isoformat()
            _atomic_json(status_path, status)
            print(
                f"DONE {len(records):02d}/{len(_row_specs()):02d} "
                f"{record['label']} axis={record['axis_value']:.9g} "
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
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    specs = _row_specs()
    if set(records) != {spec.ordinal for spec in specs}:
        raise RuntimeError("Fig. 5 row set is incomplete before assembly.")
    rows: dict[int, dict[str, np.ndarray]] = {}
    evidence: dict[str, str] = {}
    for spec in specs:
        record = records[spec.ordinal]
        path = Path(record["path"])
        if path.resolve() != _row_path(run_dir, spec).resolve():
            raise RuntimeError(f"Fig. 5 row {spec.label} points outside its run path.")
        row_raw, _ = _load_row_arrays(
            path,
            spec=spec,
            producer=producer,
        )
        current_hash = _sha256_file(path)
        if current_hash != record["sha256"] or path.stat().st_size != record["size_bytes"]:
            raise RuntimeError(f"Fig. 5 row payload changed during assembly: {path}.")
        rows[spec.ordinal] = row_raw
        evidence[spec.label] = current_hash

    first = rows[specs[0].ordinal]
    for spec in specs[1:]:
        row = rows[spec.ordinal]
        for key in _COMMON_KEYS:
            if not np.array_equal(row[key], first[key]):
                raise RuntimeError(f"Fig. 5 rows disagree on common field {key!r}.")

    upper_specs = tuple(spec for spec in specs if spec.panel == "upper")
    lower_specs = tuple(spec for spec in specs if spec.panel == "lower")
    upper_rows = [rows[spec.ordinal] for spec in upper_specs]
    lower_rows = [rows[spec.ordinal] for spec in lower_specs]
    raw: dict[str, np.ndarray] = {
        key: first[key].copy()
        for key in _COMMON_KEYS
    }
    raw["upper_T_bath"] = np.concatenate(
        [row["upper_T_bath"] for row in upper_rows],
        axis=0,
    )
    raw["upper_nbar"] = upper_rows[0]["upper_nbar"].copy()
    raw["upper_f"] = np.concatenate([row["upper_f"] for row in upper_rows], axis=0)
    raw["upper_n_ph"] = np.concatenate(
        [row["upper_n_ph"] for row in upper_rows],
        axis=0,
    )
    raw["lower_nbar"] = np.concatenate(
        [row["lower_nbar"] for row in lower_rows],
        axis=0,
    )
    raw["lower_T_bath"] = lower_rows[0]["lower_T_bath"].copy()
    raw["lower_f"] = np.concatenate([row["lower_f"] for row in lower_rows], axis=0)
    raw["lower_n_ph"] = np.concatenate(
        [row["lower_n_ph"] for row in lower_rows],
        axis=0,
    )
    for field in NUMBER_CERTIFICATE_FIELDS:
        raw[f"upper_{field}"] = np.concatenate(
            [row[f"upper_{field}"] for row in upper_rows],
            axis=0,
        )
        raw[f"lower_{field}"] = np.concatenate(
            [row[f"lower_{field}"] for row in lower_rows],
            axis=0,
        )

    result = fig5_paper.observables(raw)
    fig5_paper._validate_result_for_artifact(result)
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
        raise RuntimeError(f"Fig. 5 run status at {path} is corrupt.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Fig. 5 run status at {path} is malformed.")
    if (
        payload.get("schema") != STATUS_SCHEMA
        or payload.get("run_identity") != _run_identity(producer)
        or payload.get("producer_sha256") != _producer_sha256(producer)
        or payload.get("total_rows") != len(_row_specs())
    ):
        raise RuntimeError(
            f"Fig. 5 run status at {path} belongs to foreign or stale work."
        )
    state = payload.get("state")
    if state not in {"running", "failed", "complete"}:
        raise RuntimeError(f"Fig. 5 run status at {path} has invalid state.")
    attempt = payload.get("attempt")
    completed = payload.get("completed_rows")
    if (
        isinstance(attempt, bool)
        or not isinstance(attempt, int)
        or attempt < 1
        or isinstance(completed, bool)
        or not isinstance(completed, int)
        or not 0 <= completed <= len(_row_specs())
    ):
        raise RuntimeError(f"Fig. 5 run status at {path} has invalid counts.")
    return cast(dict[str, object], payload)


def _status_payload(
    *,
    state: str,
    producer: Mapping[str, object],
    started_utc: str,
    attempt: int,
    completed: int,
    error: BaseException | None = None,
    completion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "attempt": attempt,
        "completed_rows": completed,
        "producer_sha256": _producer_sha256(producer),
        "run_identity": _run_identity(producer),
        "schema": STATUS_SCHEMA,
        "started_utc": started_utc,
        "state": state,
        "total_rows": len(_row_specs()),
        "updated_utc": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        payload["error"] = {
            "message": str(error),
            "type": type(error).__name__,
        }
    if completion is not None:
        payload["completion"] = dict(completion)
    return payload


def run_campaign(
    *,
    run_root: Path,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, object]:
    """Run or resume all independent rows, then publish one canonical bundle."""
    if _NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT:
        raise RuntimeError(
            "NumPy was imported before the Fig. 5 runner established its "
            "single-thread environment. Launch this runner in a fresh process."
        )
    _assert_single_thread_environment()
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers must be a positive integer.")

    started = time.perf_counter()
    runner_path = Path(__file__).resolve()
    producer = _producer_base(runner_path)
    run_dir = _run_dir(run_root, producer)
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    prior_status = _read_prior_status(status_path, producer=producer)
    started_utc = (
        str(prior_status["started_utc"])
        if prior_status is not None
        else datetime.now(UTC).isoformat()
    )
    attempt = (
        1
        if prior_status is None
        else cast(int, prior_status["attempt"]) + 1
    )
    records: dict[int, RowRecord] = {}
    print(f"RUN_DIR={run_dir}", flush=True)
    print(f"RUN_IDENTITY={_run_identity(producer)}", flush=True)
    print(f"PRODUCER_SHA256={_producer_sha256(producer)}", flush=True)
    with _campaign_lock(run_dir):
        try:
            records = _scan_rows(run_dir, producer)
            running_status = _status_payload(
                state="running",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed=len(records),
            )
            _atomic_json(status_path, running_status)
            specs = _row_specs()
            missing = [
                spec for spec in specs if spec.ordinal not in records
            ]
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
                status_base=running_status,
            )
            _assert_producer_current(producer, runner_path)
            raw, row_hashes = _assemble_raw(
                records,
                run_dir=run_dir,
                producer=producer,
            )
            result = fig5_paper.observables(raw)
            certificate_maxima = fig5_paper._validate_result_for_artifact(
                result
            )
            artifact_producer = cast(
                Mapping[str, Any],
                producer["artifact_producer"],
            )
            csv_path, pdf_a_path, pdf_b_path, record_path = (
                fig5_paper.publish_artifacts(
                    result,
                    producer=artifact_producer,
                )
            )
            _assert_producer_current(producer, runner_path)
            wall_s = time.perf_counter() - started
            completion = {
                "aggregate_worker_s": sum(
                    float(record["elapsed_s"]) for record in records.values()
                ),
                "campaign": {
                    "mode": producer["mode"],
                    "row_schema": ROW_SCHEMA,
                    "rows": [
                        {
                            "axis_value": records[ordinal]["axis_value"],
                            "label": records[ordinal]["label"],
                            "ordinal": records[ordinal]["ordinal"],
                            "panel": records[ordinal]["panel"],
                            "row_index": records[ordinal]["row_index"],
                            "sha256": records[ordinal]["sha256"],
                            "size_bytes": records[ordinal]["size_bytes"],
                        }
                        for ordinal in sorted(records)
                    ],
                    "run_identity": _run_identity(producer),
                    "runner": producer["runner"],
                    "single_thread_environment": producer[
                        "single_thread_environment"
                    ],
                    "status_schema": STATUS_SCHEMA,
                },
                "certificate_maxima": certificate_maxima,
                "artifacts": {
                    "csv": {"path": str(csv_path), "sha256": _sha256_file(csv_path)},
                    "pdf_a": {"path": str(pdf_a_path), "sha256": _sha256_file(pdf_a_path)},
                    "pdf_b": {"path": str(pdf_b_path), "sha256": _sha256_file(pdf_b_path)},
                    "record": {
                        "path": str(record_path),
                        "sha256": _sha256_file(record_path),
                    },
                },
                "new_rows": len(missing),
                "resumed_rows": len(records) - len(missing),
                "row_payload_sha256": row_hashes,
                "wall_s": wall_s,
            }
            final_status = _status_payload(
                state="complete",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed=len(records),
                completion=completion,
            )
            # The promotion record binds the artifact files to the producer;
            # this companion record binds that promoted set to the exact
            # runner, six authenticated continuation rows, thread controls,
            # and certificate maxima. Keep the deterministic run-directory
            # status byte-identical so either copy can authenticate the other.
            campaign_record_path = _campaign_record_path()
            _atomic_json(campaign_record_path, final_status)
            _atomic_json(status_path, final_status)
            if campaign_record_path.read_bytes() != status_path.read_bytes():
                raise RuntimeError(
                    "Fig. 5 campaign evidence diverged from the durable run "
                    "status after publication."
                )
            print(json.dumps(final_status, indent=2, sort_keys=True), flush=True)
            return final_status
        except BaseException as exc:
            failed_status = _status_payload(
                state="failed",
                producer=producer,
                started_utc=started_utc,
                attempt=attempt,
                completed=len(records),
                error=exc,
            )
            _atomic_json(status_path, failed_status)
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path(tempfile.gettempdir()) / "qpsim-fig5-runs",
        help=(
            "Parent directory for the deterministic fig5-<identity> run "
            "directory (default: system temp/qpsim-fig5-runs)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=f"Independent continuation-row workers (default: {DEFAULT_MAX_WORKERS}).",
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
