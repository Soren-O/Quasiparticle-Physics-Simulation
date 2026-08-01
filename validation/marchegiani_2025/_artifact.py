"""Authenticated, transactional artifact bundles for M25 validation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO, cast

import matplotlib
import numpy as np
import scipy

from validation.marchegiani_2025._pdf import (
    validate_single_nonempty_matplotlib_pdf,
)
from validation.source_provenance import source_manifest

BUNDLE_SCHEMA = "qpsim-m25-artifact-bundle-v1"
TABLE_SCHEMA = "qpsim-m25-artifact-table-v1"

_THREAD_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_CBWR",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "OPENBLAS_NUM_THREADS",
)


class ArtifactValidationError(RuntimeError):
    """Raised when an M25 artifact is malformed, stale, or inconsistent."""


@dataclass(frozen=True)
class ProducerIdentity:
    """Source/configuration and numerical-runtime identity frozen pre-solve."""

    fingerprint: dict[str, Any]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class TablePayload:
    """Strictly decoded table metadata and string rows."""

    metadata: dict[str, Any]
    rows: list[list[str]]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def source_fingerprint(
    *,
    bundle: str,
    config: Mapping[str, Any],
    producer_module: Path,
    extra_validation_modules: Sequence[Path] = (),
) -> dict[str, Any]:
    """Return exact portable source/configuration identity for one bundle."""
    return {
        "bundle": bundle,
        "config": _json_copy(dict(config)),
        "schema": BUNDLE_SCHEMA,
        "sources": source_manifest(
            producer_module,
            extra_validation_modules=(
                Path(__file__),
                Path(__file__).with_name("_pdf.py"),
                *extra_validation_modules,
            ),
        ),
    }


def _numeric_build_provenance(module: Any) -> dict[str, Any]:
    """Return portable BLAS/LAPACK/compiler facts from NumPy-style config."""
    config = getattr(getattr(module, "__config__", None), "CONFIG", {})
    if not isinstance(config, Mapping):
        return {
            "build_dependencies": {},
            "compilers": {},
            "simd_extensions": {},
        }
    dependencies = config.get("Build Dependencies", {})
    dependency_result: dict[str, Any] = {}
    if isinstance(dependencies, Mapping):
        for name in ("blas", "lapack"):
            raw = dependencies.get(name, {})
            if not isinstance(raw, Mapping):
                continue
            dependency_result[name] = {
                key: raw[key]
                for key in (
                    "detection method",
                    "found",
                    "name",
                    "openblas configuration",
                    "version",
                )
                if key in raw
            }
    compilers = config.get("Compilers", {})
    compiler_result: dict[str, Any] = {}
    if isinstance(compilers, Mapping):
        for name, raw in compilers.items():
            if isinstance(raw, Mapping):
                compiler_result[str(name)] = {
                    key: raw[key]
                    for key in ("name", "version")
                    if key in raw
                }
    simd = config.get("SIMD Extensions", {})
    simd_result = dict(simd) if isinstance(simd, Mapping) else {}
    return {
        "build_dependencies": dependency_result,
        "compilers": compiler_result,
        "simd_extensions": simd_result,
    }


def producer_runtime_provenance() -> dict[str, Any]:
    """Capture the producer numerical runtime without checkout-local paths."""
    multiarray = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    raw_cpu_features = getattr(multiarray, "__cpu_features__", {})
    cpu_features = (
        {
            str(name): bool(enabled)
            for name, enabled in sorted(raw_cpu_features.items())
        }
        if isinstance(raw_cpu_features, Mapping)
        else {}
    )
    return {
        "matplotlib": {
            "backend": str(matplotlib.get_backend()),
            "freetype": getattr(
                getattr(matplotlib, "ft2font", None),
                "__freetype_version__",
                None,
            ),
            "version": matplotlib.__version__,
        },
        "numpy": {
            "build": _numeric_build_provenance(np),
            "version": np.__version__,
        },
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "byteorder": sys.byteorder,
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "runtime_cpu_features": cpu_features,
        "scipy": {
            "build": _numeric_build_provenance(scipy),
            "version": scipy.__version__,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in _THREAD_ENVIRONMENT_KEYS
        },
    }


def capture_producer_identity(
    fingerprint: Mapping[str, Any],
) -> ProducerIdentity:
    """Freeze source/config and runtime before a solve begins."""
    matplotlib.use("Agg")
    return ProducerIdentity(
        fingerprint=cast(dict[str, Any], _json_copy(dict(fingerprint))),
        runtime=cast(dict[str, Any], _json_copy(producer_runtime_provenance())),
    )


def _require_runtime_shape(runtime: Any, *, context: str) -> dict[str, Any]:
    expected = {
        "matplotlib",
        "numpy",
        "platform",
        "python",
        "runtime_cpu_features",
        "scipy",
        "thread_environment",
    }
    if not isinstance(runtime, dict) or set(runtime) != expected:
        raise ArtifactValidationError(f"{context} has malformed runtime provenance.")
    if (
        not isinstance(runtime["thread_environment"], dict)
        or set(runtime["thread_environment"]) != set(_THREAD_ENVIRONMENT_KEYS)
    ):
        raise ArtifactValidationError(
            f"{context} has malformed thread-environment provenance."
        )
    matplotlib_record = runtime["matplotlib"]
    if (
        not isinstance(matplotlib_record, dict)
        or set(matplotlib_record) != {"backend", "freetype", "version"}
        or not isinstance(matplotlib_record["backend"], str)
        or not isinstance(matplotlib_record["version"], str)
        or (
            matplotlib_record["freetype"] is not None
            and not isinstance(matplotlib_record["freetype"], str)
        )
    ):
        raise ArtifactValidationError(
            f"{context} has malformed matplotlib provenance."
        )
    for library in ("numpy", "scipy"):
        record = runtime[library]
        if (
            not isinstance(record, dict)
            or set(record) != {"build", "version"}
            or not isinstance(record["version"], str)
            or not isinstance(record["build"], dict)
            or set(record["build"])
            != {"build_dependencies", "compilers", "simd_extensions"}
            or not all(
                isinstance(record["build"][field], dict)
                for field in (
                    "build_dependencies",
                    "compilers",
                    "simd_extensions",
                )
            )
        ):
            raise ArtifactValidationError(
                f"{context} has malformed {library} provenance."
            )
    cpu_features = runtime["runtime_cpu_features"]
    if (
        not isinstance(cpu_features, dict)
        or not all(
            isinstance(name, str) and isinstance(enabled, bool)
            for name, enabled in cpu_features.items()
        )
    ):
        raise ArtifactValidationError(
            f"{context} has malformed runtime CPU-feature provenance."
        )
    return runtime


def assert_producer_identity_current(
    producer: ProducerIdentity,
    current_fingerprint: Mapping[str, Any],
) -> None:
    """Reject publication if source/configuration or runtime changed mid-run."""
    current = capture_producer_identity(current_fingerprint)
    if producer.fingerprint != current.fingerprint:
        raise ArtifactValidationError(
            "M25 producer source/configuration changed during generation; "
            "discard the result and rerun."
        )
    if producer.runtime != current.runtime:
        raise ArtifactValidationError(
            "M25 numerical runtime changed during generation; discard and rerun."
        )


def manifest_path_for(anchor: Path) -> Path:
    return anchor.with_suffix(".artifact.json")


def require_staging_path(
    path: Path,
    canonical_path: Path,
    *,
    kind: str,
) -> Path:
    """Forbid bypassing the manifest-last canonical publisher."""
    if path.resolve() == canonical_path.resolve():
        raise ArtifactValidationError(
            f"Direct writes to canonical M25 {kind} {canonical_path.name} are "
            "forbidden; use generate_baseline()."
        )
    return path


@contextmanager
def artifact_bundle_lock(
    manifest_path: Path,
    *,
    operation: str,
) -> Iterator[None]:
    """Hold one nonblocking OS lock shared by bundle readers and publishers."""
    lock_path = manifest_path.with_suffix(manifest_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
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
            raise ArtifactValidationError(
                f"M25 artifact bundle {manifest_path} is locked by another "
                f"reader or publisher; refusing concurrent {operation}."
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


def _format_cell(value: str | float | int) -> str:
    if isinstance(value, bool):
        raise ArtifactValidationError("Boolean values are not valid table cells.")
    if isinstance(value, (float, int)):
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ArtifactValidationError("Artifact tables require finite numbers.")
        return f"{numeric:.17e}"
    if not isinstance(value, str) or "\n" in value or "\r" in value:
        raise ArtifactValidationError("Artifact string cells must be one line.")
    return value


def _table_payload_sha256(columns: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    return _sha256(
        _canonical_json(
            {"columns": list(columns), "rows": [list(row) for row in rows]}
        ).encode("utf-8")
    )


def write_table(
    path: Path,
    *,
    bundle: str,
    role: str,
    config: Mapping[str, Any],
    columns: Sequence[str],
    rows: Sequence[Sequence[str | float | int]],
    certificate: Mapping[str, Any],
) -> Path:
    """Atomically write one strict noncanonical CSV table."""
    formatted_rows = [[_format_cell(value) for value in row] for row in rows]
    if (
        not columns
        or len(set(columns)) != len(columns)
        or not all(isinstance(column, str) and column for column in columns)
        or any(len(row) != len(columns) for row in formatted_rows)
    ):
        raise ArtifactValidationError("M25 artifact table shape/columns are invalid.")
    metadata = {
        "bundle": bundle,
        "certificate": _json_copy(dict(certificate)),
        "columns": list(columns),
        "config": _json_copy(dict(config)),
        "payload_sha256": _table_payload_sha256(columns, formatted_rows),
        "producer_platform": sys.platform,
        "role": role,
        "row_count": len(formatted_rows),
        "schema": TABLE_SCHEMA,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fp:
            temporary_path = Path(fp.name)
            writer = csv.writer(cast(TextIO, fp), lineterminator="\n")
            writer.writerow([f"# qpsim_artifact_schema={TABLE_SCHEMA}"])
            writer.writerow([f"# qpsim_metadata={_canonical_json(metadata)}"])
            writer.writerow(columns)
            writer.writerows(formatted_rows)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path


def read_table(
    path: Path,
    *,
    bundle: str,
    role: str,
    config: Mapping[str, Any],
    columns: Sequence[str],
    certificate: Mapping[str, Any],
) -> TablePayload:
    """Strictly decode one manifest-authenticated M25 CSV."""
    try:
        with path.open(encoding="utf-8", newline="") as fp:
            csv_rows = list(csv.reader(fp))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ArtifactValidationError(f"Cannot decode M25 table {path}.") from exc
    if len(csv_rows) < 3 or csv_rows[0] != [
        f"# qpsim_artifact_schema={TABLE_SCHEMA}"
    ]:
        raise ArtifactValidationError(f"M25 table {path} has stale/missing schema.")
    if len(csv_rows[1]) != 1 or not csv_rows[1][0].startswith(
        "# qpsim_metadata="
    ):
        raise ArtifactValidationError(f"M25 table {path} lacks strict metadata.")
    try:
        metadata = json.loads(csv_rows[1][0].split("=", 1)[1])
    except (ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"M25 table {path} metadata is malformed.") from exc
    expected_keys = {
        "bundle",
        "certificate",
        "columns",
        "config",
        "payload_sha256",
        "producer_platform",
        "role",
        "row_count",
        "schema",
    }
    if not isinstance(metadata, dict) or set(metadata) != expected_keys:
        raise ArtifactValidationError(f"M25 table {path} metadata fields drifted.")
    expected_metadata = {
        "bundle": bundle,
        "certificate": _json_copy(dict(certificate)),
        "columns": list(columns),
        "config": _json_copy(dict(config)),
        "role": role,
        "schema": TABLE_SCHEMA,
    }
    for key, expected in expected_metadata.items():
        if metadata[key] != expected:
            raise ArtifactValidationError(
                f"M25 table {path} has stale or false {key} metadata."
            )
    if not isinstance(metadata["producer_platform"], str) or not metadata[
        "producer_platform"
    ]:
        raise ArtifactValidationError(
            f"M25 table {path} has malformed producer-platform metadata."
        )
    if csv_rows[2] != list(columns):
        raise ArtifactValidationError(f"M25 table {path} column header drifted.")
    rows = csv_rows[3:]
    if (
        not isinstance(metadata["row_count"], int)
        or isinstance(metadata["row_count"], bool)
        or metadata["row_count"] != len(rows)
        or any(len(row) != len(columns) for row in rows)
        or metadata["payload_sha256"] != _table_payload_sha256(columns, rows)
    ):
        raise ArtifactValidationError(f"M25 table {path} payload is inconsistent.")
    return TablePayload(metadata=metadata, rows=rows)


def _file_record(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(f"Cannot read staged M25 member {path}.") from exc
    if not payload:
        raise ArtifactValidationError(f"M25 member {path} is empty.")
    record: dict[str, Any] = {
        "kind": kind,
        "sha256": _sha256(payload),
        "size_bytes": len(payload),
    }
    if kind == "pdf":
        try:
            validate_single_nonempty_matplotlib_pdf(payload, path=path)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"M25 companion {path} is not a complete nonempty one-page PDF."
            ) from exc
        record["page_count"] = 1
    elif kind != "csv":
        raise ArtifactValidationError(f"Unsupported M25 member kind {kind!r}.")
    return record


def _validate_manifest_payload(
    payload: Any,
    *,
    manifest_path: Path,
    bundle: str,
    fingerprint: Mapping[str, Any],
    expected_members: Mapping[str, str],
    member_paths: Mapping[str, Path],
) -> dict[str, Any]:
    expected_keys = {
        "bundle",
        "files",
        "fingerprint",
        "producer_runtime",
        "schema",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ArtifactValidationError(
            f"M25 bundle manifest {manifest_path} fields drifted."
        )
    if payload["schema"] != BUNDLE_SCHEMA or payload["bundle"] != bundle:
        raise ArtifactValidationError(
            f"M25 bundle manifest {manifest_path} has stale identity."
        )
    if payload["fingerprint"] != _json_copy(dict(fingerprint)):
        raise ArtifactValidationError(
            f"M25 bundle manifest {manifest_path} is stale for current source/config."
        )
    _require_runtime_shape(
        payload["producer_runtime"],
        context=f"M25 bundle manifest {manifest_path}",
    )
    files = payload["files"]
    if not isinstance(files, dict) or set(files) != set(expected_members):
        raise ArtifactValidationError(
            f"M25 bundle manifest {manifest_path} member set drifted."
        )
    for name, kind in expected_members.items():
        record = files[name]
        required = {"kind", "sha256", "size_bytes"}
        if kind == "pdf":
            required.add("page_count")
        if (
            not isinstance(record, dict)
            or set(record) != required
            or record.get("kind") != kind
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
            or not isinstance(record.get("size_bytes"), int)
            or isinstance(record["size_bytes"], bool)
            or record["size_bytes"] <= 0
            or (kind == "pdf" and record.get("page_count") != 1)
        ):
            raise ArtifactValidationError(
                f"M25 bundle manifest {manifest_path} has malformed record {name}."
            )
        actual = _file_record(member_paths[name], kind=kind)
        if actual != record:
            raise ArtifactValidationError(
                f"M25 bundle member {member_paths[name]} does not match manifest."
            )
    return payload


@contextmanager
def verified_bundle(
    *,
    manifest_path: Path,
    bundle: str,
    fingerprint: Mapping[str, Any],
    expected_members: Mapping[str, str],
    member_paths: Mapping[str, Path],
) -> Iterator[dict[str, Any]]:
    """Authenticate all canonical members while holding the bundle lock."""
    with artifact_bundle_lock(manifest_path, operation="read"):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactValidationError(
                f"Cannot decode M25 bundle manifest {manifest_path}."
            ) from exc
        yield _validate_manifest_payload(
            payload,
            manifest_path=manifest_path,
            bundle=bundle,
            fingerprint=fingerprint,
            expected_members=expected_members,
            member_paths=member_paths,
        )


def publish_bundle(
    *,
    manifest_path: Path,
    bundle: str,
    producer: ProducerIdentity,
    current_fingerprint: Callable[[], Mapping[str, Any]],
    members: Mapping[Path, tuple[str, Callable[[Path], Path]]],
    validate_staged: Callable[[Mapping[Path, Path]], None],
) -> tuple[Path, ...]:
    """Stage, validate, and promote a complete bundle; manifest is last."""
    with artifact_bundle_lock(manifest_path, operation="publication"):
        assert_producer_identity_current(producer, current_fingerprint())
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        stages: dict[Path, Path] = {}
        manifest_stage: Path | None = None
        try:
            for canonical, (kind, writer) in members.items():
                canonical.parent.mkdir(parents=True, exist_ok=True)
                with NamedTemporaryFile(
                    mode="wb",
                    dir=canonical.parent,
                    prefix=f".{canonical.name}.",
                    suffix=canonical.suffix,
                    delete=False,
                ) as temporary:
                    stage = Path(temporary.name)
                stages[canonical] = stage
                written = writer(stage)
                if written.resolve() != stage.resolve():
                    raise ArtifactValidationError(
                        f"M25 writer for {canonical.name} ignored its staged path."
                    )
                _file_record(stage, kind=kind)

            validate_staged(stages)
            files = {
                canonical.name: _file_record(stage, kind=members[canonical][0])
                for canonical, stage in stages.items()
            }
            payload = {
                "bundle": bundle,
                "files": files,
                "fingerprint": producer.fingerprint,
                "producer_runtime": producer.runtime,
                "schema": BUNDLE_SCHEMA,
            }
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=manifest_path.parent,
                prefix=f".{manifest_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as fp:
                manifest_stage = Path(fp.name)
                fp.write(json.dumps(payload, indent=2, sort_keys=True))
                fp.write("\n")
                fp.flush()
                os.fsync(fp.fileno())
            staged_by_name = {
                canonical.name: stage for canonical, stage in stages.items()
            }
            _validate_manifest_payload(
                payload,
                manifest_path=manifest_stage,
                bundle=bundle,
                fingerprint=producer.fingerprint,
                expected_members={
                    canonical.name: kind
                    for canonical, (kind, _writer) in members.items()
                },
                member_paths=staged_by_name,
            )
            assert_producer_identity_current(producer, current_fingerprint())

            for canonical, stage in stages.items():
                os.replace(stage, canonical)
            stages.clear()
            os.replace(manifest_stage, manifest_path)
            manifest_stage = None
            return (*members.keys(), manifest_path)
        finally:
            for stage in stages.values():
                stage.unlink(missing_ok=True)
            if manifest_stage is not None:
                manifest_stage.unlink(missing_ok=True)
