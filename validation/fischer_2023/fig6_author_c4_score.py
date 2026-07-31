"""Independently verify the formal Figure 6 C4 photon-operator evidence.

The C4 producer intentionally lives in another module.  This verifier does
not import that producer or the public qpsim photon operator.  It strictly
loads the externally retained raw bundle, replays the accepted C3 parent from
the selected C3 and C2 raw bundles, and independently transcribes the public
photon loop in source order.

C4 is one frozen-state operator differential.  It does not run Newton, change
the frozen occupation, or claim a nonlinear C4 root, stopping result, plotted
ordinate, curve, observable change, or paper parity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as DEFAULT_C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as DEFAULT_C2_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as DEFAULT_C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_SCORE as DEFAULT_C3_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import (
    RAW_SCHEMA as C3_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    RECEIPT_SCHEMA as C3_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    SCHEMA as C3_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    build_c3_score,
    load_c3_raw_bundle,
    load_c3_receipt,
    load_c3_score,
)
from validation.fischer_2023.fig6_author_c3_score import (
    canonical_score_bytes as canonical_c3_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_SCHEMA = "qpsim.fischer2023.fig6-author-c4-photon-bundle.v1"
SCHEMA = "qpsim.fischer2023.fig6-author-c4-photon-score.v1"
RECEIPT_SCHEMA = "qpsim.fischer2023.fig6-author-c4-raw-manifest-receipt.v1"
DEFAULT_SCORE = (
    REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6" / "c4-photon-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c4-raw-manifest-receipt.json"
)

STAGE_ID = "C4"
PARENT_STAGE_ID = "C3"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "photon_operator"
SECONDS_PER_NS = 1.0e-9
_EXPECTED_TERMINAL_INDICES = (1619, 1639)
_ARRAY_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ROUNDING_MULTIPLIER = 64.0
_CONSERVATION_LIMIT = 1.0e-12
_COMPONENT_LOCALITY_STATEMENT = (
    "Only QP photon gain, physical loss, net, and the derived QP "
    "residual change; all non-photon channels and the phonon "
    "residual remain the accepted C3c arrays. Public source-order "
    "arithmetic differs from C3c by roundoff on common rows; only "
    "the semantic endpoint extension is confined to the listed "
    "terminal child indices."
)
_EXPECTED_OPERATOR_COMPARISON = {
    "gain": {
        "l1_absolute_s_inv": 2.9390985920687567e-13,
        "linf_absolute_s_inv": 2.842170943040401e-14,
        "symmetric_relative_l1": 2.987877778643742e-17,
    },
    "loss": {
        "l1_absolute_s_inv": 6.358292238341556e-13,
        "linf_absolute_s_inv": 2.2737367544323206e-13,
        "symmetric_relative_l1": 6.564511414882551e-17,
    },
    "net": {
        "l1_absolute_s_inv": 6.529623265709012e-13,
        "linf_absolute_s_inv": 2.942091015256665e-13,
        "symmetric_relative_l1": 2.017243083178416e-15,
    },
}

_RAW_SOURCE_RELATIVES = frozenset(
    {
        "qpsim/collisions/_uniform_grid.py",
        "qpsim/collisions/_validation.py",
        "qpsim/collisions/sub_gap_photon.py",
        "qpsim/physics/bcs_quadrature.py",
        "qpsim/physics/spectral.py",
        "validation/fischer_2023/fig6_author_c3_score.py",
        "validation/fischer_2023/fig6_author_c4_bundle.py",
        "validation/source_provenance.py",
    }
)
_SOURCE_PATHS = (
    Path(__file__).resolve(),
    *(REPOSITORY_ROOT / relative for relative in sorted(_RAW_SOURCE_RELATIVES)),
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}

_EXPECTED_ARRAY_NAMES = frozenset(
    {
        "arithmetic_delta_gain_s_inv",
        "arithmetic_delta_loss_s_inv",
        "arithmetic_delta_net_s_inv",
        "operator_delta_gain_s_inv",
        "operator_delta_loss_s_inv",
        "operator_delta_net_s_inv",
        "hybrid_phonon_residual_s_inv",
        "hybrid_qp_residual_s_inv",
        "parent_active_mask",
        "parent_cell_weights_ueV",
        "parent_f",
        "parent_qp_photon_gain_s_inv",
        "parent_qp_photon_loss_s_inv",
        "parent_qp_photon_net_s_inv",
        "parent_phonon_residual_s_inv",
        "parent_qp_residual_s_inv",
        "qpsim_author_endpoint_gain_s_inv",
        "qpsim_author_endpoint_loss_s_inv",
        "qpsim_author_endpoint_net_s_inv",
        "qpsim_gain_ns_inv",
        "qpsim_gain_s_inv",
        "qpsim_loss_ns_inv",
        "qpsim_loss_rate_ns_inv",
        "qpsim_loss_s_inv",
        "qpsim_net_ns_inv",
        "qpsim_net_s_inv",
        "terminal_extension_gain_s_inv",
        "terminal_extension_loss_s_inv",
        "terminal_extension_net_s_inv",
        "terminal_extension_support_mask",
    }
)
_RAW_METADATA_KEYS = {
    "array_descriptors",
    "comparison_contract",
    "component_locality",
    "coordinate_contract",
    "endpoint_contract",
    "frozen_inputs",
    "limitations",
    "operator_inputs",
    "parent_bindings",
    "runtime",
    "schema",
    "source_binding",
    "sources",
    "stage",
    "units",
}
_SCORE_KEYS = {
    "acceptance",
    "array_descriptors",
    "component_locality",
    "conservation",
    "endpoint_comparison",
    "limitations",
    "operator_comparison",
    "operator_inputs",
    "parent_bindings",
    "raw_bundle",
    "schema",
    "source_binding",
    "sources",
    "stage",
    "units",
}


class C4ScoreError(ValueError):
    """The C4 raw evidence, parent chain, score, or receipt is malformed."""


@dataclass(frozen=True)
class _DirectoryState:
    root_identity: tuple[int, int, int, int, int]
    entries: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass
class _ParentContext:
    c3_metadata: dict[str, Any]
    c3_arrays: dict[str, np.ndarray]
    c3_manifest_sha256: str
    c3_score: dict[str, Any]
    c3_score_path: Path
    c3_score_bytes: bytes
    c3_receipt: dict[str, Any]
    c3_receipt_path: Path
    c3_receipt_bytes: bytes
    c2_score_path: Path
    c2_score_bytes: bytes
    c2_receipt_path: Path
    c2_receipt_bytes: bytes
    c3_bundle_dir: Path
    c3_directory_state: _DirectoryState
    c2_bundle_dir: Path
    c2_directory_state: _DirectoryState


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C4ScoreError(f"C4 score source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C4ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C4ScoreError(f"Non-finite JSON constant {token!r} is forbidden.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C4ScoreError(f"{label} is not valid UTF-8 JSON.") from exc
    if not isinstance(parsed, dict):
        raise C4ScoreError(f"{label} must contain a JSON object.")
    return parsed


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C4ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise C4ScoreError(f"{label} key closure is invalid; missing={missing}, extra={extra}.")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C4ScoreError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise C4ScoreError(f"{label} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise C4ScoreError(f"{label} must be >= {minimum}.")
    return result


def _finite_scalar(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise C4ScoreError(f"{label} must be a real scalar.")
    result = float(value)
    if not np.isfinite(result):
        raise C4ScoreError(f"{label} must be finite.")
    if positive and result <= 0.0:
        raise C4ScoreError(f"{label} must be positive.")
    if nonnegative and result < 0.0:
        raise C4ScoreError(f"{label} must be non-negative.")
    return result


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(_npy_bytes(array)).hexdigest(),
        "shape": list(array.shape),
    }


def _array_bit_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return _npy_bytes(np.asarray(reference)) == _npy_bytes(np.asarray(candidate))


def _json_value_bit_exact(reference: object, candidate: object) -> bool:
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, float):
        return (
            np.asarray(reference, dtype=np.float64).tobytes()
            == np.asarray(
                candidate,
                dtype=np.float64,
            ).tobytes()
        )
    if isinstance(reference, dict):
        assert isinstance(candidate, dict)
        return set(reference) == set(candidate) and all(
            _json_value_bit_exact(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list):
        assert isinstance(candidate, list)
        return len(reference) == len(candidate) and all(
            _json_value_bit_exact(left, right)
            for left, right in zip(reference, candidate, strict=True)
        )
    return reference == candidate


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    if not np.isfinite(result):
        raise C4ScoreError("C4 scalar record must be finite.")
    return {"hex": result.hex(), "value": result}


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
    )


def _read_regular_file_once(path: Path, label: str) -> bytes:
    """Read one stable non-symlink file and bind the opened handle identity."""

    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise C4ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C4ScoreError(f"{label} must be a regular non-symlink file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise C4ScoreError(f"{label} could not be opened safely.") from exc
    try:
        opened_before = os.fstat(descriptor)
        if _stat_identity(opened_before) != _stat_identity(before):
            raise C4ScoreError(f"{label} changed before its stable read.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
        if _stat_identity(opened_after) != _stat_identity(opened_before):
            raise C4ScoreError(f"{label} changed during its stable read.")
    finally:
        os.close(descriptor)
    try:
        after = candidate.lstat()
    except OSError as exc:
        raise C4ScoreError(f"{label} disappeared after its stable read.") from exc
    if _stat_identity(after) != _stat_identity(before):
        raise C4ScoreError(f"{label} path identity changed during its stable read.")
    return b"".join(chunks)


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    candidate = Path(path)
    if candidate.is_symlink():
        raise C4ScoreError(f"{label} is unsafe or a symlink.")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(REPOSITORY_ROOT)
    except (OSError, ValueError) as exc:
        raise C4ScoreError(f"{label} must be a regular repository-contained file.") from exc
    return resolved, _read_regular_file_once(resolved, label)


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    if _read_regular_file_once(path, label) != expected:
        raise C4ScoreError(f"{label} changed during C4 verification.")


def _directory_state(path: Path, label: str) -> tuple[Path, _DirectoryState]:
    candidate = Path(path)
    try:
        root_before = candidate.lstat()
    except OSError as exc:
        raise C4ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(root_before.st_mode) or not stat.S_ISDIR(root_before.st_mode):
        raise C4ScoreError(f"{label} must be a non-symlink directory.")
    root = candidate.resolve()
    entries: list[tuple[str, tuple[int, int, int, int, int]]] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise C4ScoreError(f"{label} cannot be enumerated safely.") from exc
    for child in children:
        try:
            child_stat = child.lstat()
        except OSError as exc:
            raise C4ScoreError(f"{label} changed during directory enumeration.") from exc
        if stat.S_ISLNK(child_stat.st_mode):
            raise C4ScoreError(f"{label} contains a symlink.")
        entries.append((child.name, _stat_identity(child_stat)))
    try:
        root_after = root.lstat()
    except OSError as exc:
        raise C4ScoreError(f"{label} disappeared during enumeration.") from exc
    if _stat_identity(root_after) != _stat_identity(root_before):
        raise C4ScoreError(f"{label} changed during directory enumeration.")
    return root, _DirectoryState(_stat_identity(root_before), tuple(entries))


def _assert_directory_state(path: Path, expected: _DirectoryState, label: str) -> None:
    _root, current = _directory_state(path, label)
    if current != expected:
        raise C4ScoreError(f"{label} changed during C4 verification.")


def load_c4_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one canonical external C4 raw bundle."""

    root, before = _directory_state(bundle_dir, "C4 raw bundle")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C4 raw manifest")
    manifest = _parse_json(manifest_raw, "C4 raw manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C4 raw manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C4ScoreError("C4 raw manifest schema is unsupported.")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise C4ScoreError("C4 raw manifest is not canonical JSON.")

    files = _mapping(manifest.get("files"), "C4 raw manifest files")
    metadata = _mapping(manifest.get("metadata"), "C4 raw metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C4 raw metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C4ScoreError("C4 raw metadata schema is unsupported.")
    expected_filenames = {f"{name}.npy" for name in _EXPECTED_ARRAY_NAMES}
    if set(files) != expected_filenames or len(files) != 30:
        raise C4ScoreError("C4 raw file closure is invalid.")
    if {name for name, _identity in before.entries} != expected_filenames | {"manifest.json"}:
        raise C4ScoreError("C4 raw directory closure is invalid.")
    if any(not stat.S_ISREG(identity[2]) for _name, identity in before.entries):
        raise C4ScoreError("C4 raw bundle contains a non-file entry.")

    arrays: dict[str, np.ndarray] = {}
    for filename in sorted(expected_filenames):
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise C4ScoreError(f"Unsafe C4 raw filename {filename!r}.")
        name = filename[:-4]
        if _ARRAY_NAME_RE.fullmatch(name) is None:
            raise C4ScoreError(f"Unsafe C4 raw array name {name!r}.")
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        expected_sha = _sha256(record.get("sha256"), f"files.{filename}.sha256")
        expected_size = _strict_int(
            record.get("size_bytes"),
            f"files.{filename}.size_bytes",
            minimum=1,
        )
        content = _read_regular_file_once(root / filename, f"C4 raw {filename}")
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha:
            raise C4ScoreError(f"C4 raw file {filename!r} failed its manifest binding.")
        if len(content) < 8 or content[:6] != b"\x93NUMPY" or content[6:8] != b"\x03\x00":
            raise C4ScoreError(f"C4 raw file {filename!r} is not canonical NPY v3.")
        try:
            stream = io.BytesIO(content)
            loaded = np.lib.format.read_array(stream, allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise C4ScoreError(f"Cannot load C4 raw array {filename!r}.") from exc
        if stream.tell() != len(content):
            raise C4ScoreError(f"C4 raw file {filename!r} contains trailing bytes.")
        array = np.asarray(loaded)
        if array.dtype.kind not in {"b", "i", "u", "f"} or np.iscomplexobj(array):
            raise C4ScoreError(
                f"C4 raw array {filename!r} has forbidden dtype {array.dtype.str!r}."
            )
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise C4ScoreError(f"C4 raw array {filename!r} contains non-finite values.")
        if _npy_bytes(array) != content:
            raise C4ScoreError(
                f"C4 raw file {filename!r} is not a canonical byte-exact NPY v3 encoding."
            )
        arrays[name] = array

    descriptors = _mapping(metadata.get("array_descriptors"), "C4 raw descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C4ScoreError("C4 raw array descriptors are incomplete, forged, or stale.")
    _assert_directory_state(root, before, "C4 raw bundle")
    return metadata, arrays, hashlib.sha256(manifest_raw).hexdigest()


def _accept_parent(
    c3_bundle_dir: Path,
    *,
    c2_bundle_dir: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
) -> _ParentContext:
    """Accept C3 only after replaying its score from selected C3/C2 raw."""

    _assert_source_snapshots()
    c3_score_path, c3_score_bytes = _repository_file_snapshot(
        c3_score_path,
        "C3 score",
    )
    c3_receipt_path, c3_receipt_bytes = _repository_file_snapshot(
        c3_receipt_path,
        "C3 receipt",
    )
    c2_score_path, c2_score_bytes = _repository_file_snapshot(
        c2_score_path,
        "C2 score",
    )
    c2_receipt_path, c2_receipt_bytes = _repository_file_snapshot(
        c2_receipt_path,
        "C2 receipt",
    )
    c3_root, c3_state = _directory_state(c3_bundle_dir, "C3 raw bundle")
    c2_root, c2_state = _directory_state(c2_bundle_dir, "C2 raw bundle")

    accepted_c3 = load_c3_score(c3_score_path, receipt_path=c3_receipt_path)
    accepted_receipt = load_c3_receipt(c3_receipt_path)
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_root)
    raw_binding = _mapping(accepted_c3.get("raw_bundle"), "accepted C3 raw binding")
    if raw_binding != {
        "manifest_sha256": c3_manifest_sha,
        "schema": C3_RAW_SCHEMA,
    }:
        raise C4ScoreError("Selected C3 raw bundle is not the accepted C3 parent.")

    rebuilt_c3 = build_c3_score(
        c3_root,
        c2_bundle_dir=c2_root,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c3_score_bytes(rebuilt_c3) != c3_score_bytes:
        raise C4ScoreError(
            "C4 refuses a C3 parent whose checked score does not independently "
            "reproduce from the selected C3 and C2 raw bundles."
        )

    _assert_file_snapshot(c3_score_path, c3_score_bytes, "C3 score")
    _assert_file_snapshot(c3_receipt_path, c3_receipt_bytes, "C3 receipt")
    _assert_file_snapshot(c2_score_path, c2_score_bytes, "C2 score")
    _assert_file_snapshot(c2_receipt_path, c2_receipt_bytes, "C2 receipt")
    _assert_directory_state(c3_root, c3_state, "C3 raw bundle")
    _assert_directory_state(c2_root, c2_state, "C2 raw bundle")
    return _ParentContext(
        c3_metadata=c3_metadata,
        c3_arrays=c3_arrays,
        c3_manifest_sha256=c3_manifest_sha,
        c3_score=accepted_c3,
        c3_score_path=c3_score_path,
        c3_score_bytes=c3_score_bytes,
        c3_receipt=accepted_receipt,
        c3_receipt_path=c3_receipt_path,
        c3_receipt_bytes=c3_receipt_bytes,
        c2_score_path=c2_score_path,
        c2_score_bytes=c2_score_bytes,
        c2_receipt_path=c2_receipt_path,
        c2_receipt_bytes=c2_receipt_bytes,
        c3_bundle_dir=c3_root,
        c3_directory_state=c3_state,
        c2_bundle_dir=c2_root,
        c2_directory_state=c2_state,
    )


def _recheck_parent(parent: _ParentContext) -> None:
    _assert_file_snapshot(parent.c3_score_path, parent.c3_score_bytes, "C3 score")
    _assert_file_snapshot(parent.c3_receipt_path, parent.c3_receipt_bytes, "C3 receipt")
    _assert_file_snapshot(parent.c2_score_path, parent.c2_score_bytes, "C2 score")
    _assert_file_snapshot(parent.c2_receipt_path, parent.c2_receipt_bytes, "C2 receipt")
    _assert_directory_state(
        parent.c3_bundle_dir,
        parent.c3_directory_state,
        "C3 raw bundle",
    )
    _assert_directory_state(
        parent.c2_bundle_dir,
        parent.c2_directory_state,
        "C2 raw bundle",
    )


def _frozen_parent_names() -> tuple[str, ...]:
    channels = (
        "qp_photon",
        "qp_scattering",
        "qp_pair",
        "phonon_scattering",
        "phonon_pair",
        "phonon_escape",
    )
    fields = ("gain", "loss", "net")
    return (
        "projected_f",
        "native_E_centers_ueV",
        "native_dE_ueV",
        "native_active_mask",
        "native_cell_density_full",
        "native_cell_weights_full",
        "native_K_plus_full",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__{channel}__{field}_s_inv"
            for channel in channels
            for field in fields
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__qp_residual_s_inv",
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )


def _operator_inputs(
    c3_metadata: dict[str, Any],
    c3_arrays: dict[str, np.ndarray],
) -> tuple[dict[str, object], int, float, float, float]:
    parameter_record = _mapping(c3_metadata.get("parameters"), "C3 parameters")
    values = _mapping(parameter_record.get("values"), "C3 parameter values")
    native = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "C3 native grid parameters",
    )
    photon_step = _strict_int(values.get("photon_bin"), "C3 photon_bin", minimum=1)
    h_eV = _finite_scalar(values.get("h_eV"), "C3 h_eV", positive=True)
    n_bar = _finite_scalar(values.get("n_bar"), "C3 n_bar", nonnegative=True)
    c_photon_s_inv = _finite_scalar(
        values.get("c_photon_s_inv"),
        "C3 c_photon_s_inv",
        nonnegative=True,
    )
    gap_ueV = _finite_scalar(native.get("gap_ueV"), "C3 native gap", positive=True)
    dE = np.asarray(c3_arrays["native_dE_ueV"])
    if dE.shape != (1640,) or np.any(~np.isfinite(dE)):
        raise C4ScoreError("C3 native dE is not the accepted 1640-cell grid.")
    dE_ueV = float(dE[0])
    if dE_ueV <= 0.0 or not np.all(dE == dE_ueV):
        raise C4ScoreError("C3 native dE is not exactly uniform.")
    omega_0_ueV = photon_step * h_eV * 1.0e6
    snapped_step = round(omega_0_ueV / dE_ueV)
    if (
        photon_step != snapped_step
        or omega_0_ueV != photon_step * dE_ueV
        or omega_0_ueV >= 2.0 * gap_ueV
    ):
        raise C4ScoreError("C4 photon energy is not the exact inherited sub-gap bin.")
    c_photon_ns_inv = c_photon_s_inv * SECONDS_PER_NS
    record: dict[str, object] = {
        "c_photon_ns_inv": _float_record(c_photon_ns_inv),
        "c_photon_s_inv": _float_record(c_photon_s_inv),
        "dE_ueV": _float_record(dE_ueV),
        "gap_ueV": _float_record(gap_ueV),
        "n_bar": _float_record(n_bar),
        "omega_0_ueV": _float_record(omega_0_ueV),
        "photon_step_bins": photon_step,
        "seconds_per_ns": _float_record(SECONDS_PER_NS),
        "snap_fraction_of_bin": _float_record(abs(omega_0_ueV - photon_step * dE_ueV) / dE_ueV),
    }
    return record, photon_step, n_bar, c_photon_s_inv, c_photon_ns_inv


def _independent_photon_loop(
    f: np.ndarray,
    rho: np.ndarray,
    active: np.ndarray,
    K_plus: np.ndarray,
    *,
    photon_step: int,
    n_bar: float,
    c_photon_ns_inv: float,
    omit_terminal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Transcribe the public source-order loop without importing it."""

    occupation = np.asarray(f, dtype=float)
    density = np.asarray(rho, dtype=float)
    supported = np.asarray(active, dtype=bool)
    coherence = np.asarray(K_plus, dtype=float)
    size = occupation.size
    if (
        occupation.shape != (1640,)
        or density.shape != (1640,)
        or supported.shape != (1640,)
        or coherence.shape != (1640, 1640)
    ):
        raise C4ScoreError("C4 frozen photon inputs have invalid shapes.")
    if (
        np.any(~np.isfinite(occupation))
        or np.any((occupation < 0.0) | (occupation > 1.0))
        or np.any(~np.isfinite(density))
        or np.any(density < 0.0)
        or np.any(~np.isfinite(coherence))
    ):
        raise C4ScoreError("C4 frozen photon inputs contain invalid values.")

    gain = np.zeros(size)
    loss_rate = np.zeros(size)
    one_minus_f = np.maximum(1.0 - occupation, 0.0)
    for i in range(size):
        if not supported[i]:
            continue
        j_up = i + photon_step
        upper_limit = size - 1 if omit_terminal else size
        if j_up < upper_limit:
            coefficient = density[j_up] * coherence[i, j_up]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_up] * (n_bar + 1.0)
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_up] * n_bar
        j_down = i - photon_step
        if j_down >= 0 and supported[j_down] and (not omit_terminal or i < size - 1):
            coefficient = density[j_down] * coherence[i, j_down]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_down] * n_bar
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_down] * (n_bar + 1.0)
    gain_with_pauli = gain * one_minus_f
    physical_loss = loss_rate * occupation
    return gain_with_pauli, loss_rate, physical_loss, gain_with_pauli - physical_loss


def _expected_arrays(
    c3_arrays: dict[str, np.ndarray],
    *,
    photon_step: int,
    n_bar: float,
    c_photon_ns_inv: float,
) -> dict[str, np.ndarray]:
    f = np.asarray(c3_arrays["projected_f"]).copy()
    rho = np.asarray(c3_arrays["native_cell_density_full"])
    active = np.asarray(c3_arrays["native_active_mask"])
    K_plus = np.asarray(c3_arrays["native_K_plus_full"])
    public_gain_ns, public_loss_rate_ns, public_loss_ns, public_net_ns = _independent_photon_loop(
        f,
        rho,
        active,
        K_plus,
        photon_step=photon_step,
        n_bar=n_bar,
        c_photon_ns_inv=c_photon_ns_inv,
        omit_terminal=False,
    )
    endpoint_gain_ns, _endpoint_rate_ns, endpoint_loss_ns, endpoint_net_ns = (
        _independent_photon_loop(
            f,
            rho,
            active,
            K_plus,
            photon_step=photon_step,
            n_bar=n_bar,
            c_photon_ns_inv=c_photon_ns_inv,
            omit_terminal=True,
        )
    )
    public_gain_s = public_gain_ns / SECONDS_PER_NS
    public_loss_s = public_loss_ns / SECONDS_PER_NS
    public_net_s = public_net_ns / SECONDS_PER_NS
    endpoint_gain_s = endpoint_gain_ns / SECONDS_PER_NS
    endpoint_loss_s = endpoint_loss_ns / SECONDS_PER_NS
    endpoint_net_s = endpoint_net_ns / SECONDS_PER_NS
    parent_gain = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__gain_s_inv"]).copy()
    parent_loss = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__loss_s_inv"]).copy()
    parent_net = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__net_s_inv"]).copy()
    terminal_gain = public_gain_s - endpoint_gain_s
    terminal_loss = public_loss_s - endpoint_loss_s
    terminal_net = public_net_s - endpoint_net_s
    terminal_support = (terminal_gain != 0.0) | (terminal_loss != 0.0) | (terminal_net != 0.0)
    parent_qp_residual = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_residual_s_inv"]
    ).copy()
    parent_phonon_residual = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv"]
    ).copy()
    return {
        "arithmetic_delta_gain_s_inv": endpoint_gain_s - parent_gain,
        "arithmetic_delta_loss_s_inv": endpoint_loss_s - parent_loss,
        "arithmetic_delta_net_s_inv": endpoint_net_s - parent_net,
        "operator_delta_gain_s_inv": public_gain_s - parent_gain,
        "operator_delta_loss_s_inv": public_loss_s - parent_loss,
        "operator_delta_net_s_inv": public_net_s - parent_net,
        "hybrid_phonon_residual_s_inv": parent_phonon_residual.copy(),
        "hybrid_qp_residual_s_inv": parent_qp_residual + (public_net_s - parent_net),
        "parent_active_mask": active.copy(),
        "parent_cell_weights_ueV": np.asarray(c3_arrays["native_cell_weights_full"]).copy(),
        "parent_f": f,
        "parent_qp_photon_gain_s_inv": parent_gain,
        "parent_qp_photon_loss_s_inv": parent_loss,
        "parent_qp_photon_net_s_inv": parent_net,
        "parent_phonon_residual_s_inv": parent_phonon_residual,
        "parent_qp_residual_s_inv": parent_qp_residual,
        "qpsim_author_endpoint_gain_s_inv": endpoint_gain_s,
        "qpsim_author_endpoint_loss_s_inv": endpoint_loss_s,
        "qpsim_author_endpoint_net_s_inv": endpoint_net_s,
        "qpsim_gain_ns_inv": public_gain_ns,
        "qpsim_gain_s_inv": public_gain_s,
        "qpsim_loss_ns_inv": public_loss_ns,
        "qpsim_loss_rate_ns_inv": public_loss_rate_ns,
        "qpsim_loss_s_inv": public_loss_s,
        "qpsim_net_ns_inv": public_net_ns,
        "qpsim_net_s_inv": public_net_s,
        "terminal_extension_gain_s_inv": terminal_gain,
        "terminal_extension_loss_s_inv": terminal_loss,
        "terminal_extension_net_s_inv": terminal_net,
        "terminal_extension_support_mask": terminal_support,
    }


def _runtime_record_valid(value: object) -> bool:
    runtime = _mapping(value, "C4 raw runtime")
    _exact_keys(
        runtime,
        {
            "byteorder",
            "implementation",
            "machine",
            "numpy_version",
            "platform",
            "python_version",
        },
        "C4 raw runtime",
    )
    if runtime.get("byteorder") not in {"little", "big"}:
        return False
    return all(
        isinstance(runtime.get(key), str) and bool(runtime.get(key))
        for key in (
            "implementation",
            "machine",
            "numpy_version",
            "platform",
            "python_version",
        )
    )


def _expected_parent_bindings(parent: _ParentContext) -> dict[str, object]:
    c3_parent = _mapping(parent.c3_score.get("parent_bindings"), "C3 parent bindings")
    return {
        "c2_raw_manifest_sha256": c3_parent.get("c2_raw_manifest_sha256"),
        "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
        "c3_raw_manifest_sha256": parent.c3_manifest_sha256,
        "c3_raw_schema": C3_RAW_SCHEMA,
        "c3_receipt_path": parent.c3_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c3_receipt_schema": C3_RECEIPT_SCHEMA,
        "c3_receipt_sha256": hashlib.sha256(parent.c3_receipt_bytes).hexdigest(),
        "c3_score_path": parent.c3_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c3_score_schema": C3_SCORE_SCHEMA,
        "c3_score_sha256": hashlib.sha256(parent.c3_score_bytes).hexdigest(),
        "c3_stage_id": PARENT_STAGE_ID,
    }


def _check_raw_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    parent: _ParentContext,
    operator_inputs: dict[str, object],
) -> None:
    if metadata.get("schema") != RAW_SCHEMA:
        raise C4ScoreError("C4 raw metadata schema is invalid.")
    expected_raw_sources = {
        relative: source_sha256(REPOSITORY_ROOT / relative)
        for relative in sorted(_RAW_SOURCE_RELATIVES)
    }
    if metadata.get("sources") != expected_raw_sources:
        raise C4ScoreError("C4 raw source closure is forged, incomplete, or stale.")
    if metadata.get("source_binding") != {
        "hash_kind": "canonical_sha256_import_time_disk_snapshot",
        "scope": (
            "C4 producer, public sub-gap photon operator and validators, "
            "SpectralContext/quadrature, accepted C3 loader, and provenance source"
        ),
    }:
        raise C4ScoreError("C4 raw source-binding policy is invalid.")
    if not _runtime_record_valid(metadata.get("runtime")):
        raise C4ScoreError("C4 raw runtime record is invalid.")
    if metadata.get("stage") != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
    }:
        raise C4ScoreError("C4 raw stage identity is invalid.")
    if metadata.get("comparison_contract") != {
        "arithmetic_control": (
            "qpsim source-order arithmetic and per-nanosecond units, "
            "with the author terminal omission restored"
        ),
        "candidate": "public qpsim sub-gap photon gain and loss_rate",
        "loss_comparison": (
            "physical loss = returned loss_rate * frozen f; the raw "
            "loss_rate coefficient is never compared directly to C3c loss"
        ),
        "parent": "accepted C3c author-form QP photon gain/loss/net",
        "semantic_delta": (
            "candidate minus arithmetic control, isolated from candidate "
            "minus C3c floating-point reordering"
        ),
    }:
        raise C4ScoreError("C4 raw comparison contract is invalid.")
    if metadata.get("coordinate_contract") != {
        "active_child_indices": "[20, 1640)",
        "coherence": "accepted C3c native SpectralContext K_plus",
        "density": "accepted C3c native partner cell_density",
        "guard_child_indices": "[0, 20), canonical positive zero",
        "native_cell_count": 1640,
        "photon_mapping": "child i <-> child i+20; no interpolation",
    }:
        raise C4ScoreError("C4 raw coordinate contract is invalid.")
    if metadata.get("endpoint_contract") != {
        "author_parent": (
            "C3c author residual omits both directions of every photon "
            "transition touching the final active QP cell"
        ),
        "diagnostic_control": (
            "public qpsim arithmetic and units with the author terminal omission restored"
        ),
        "qpsim_candidate": (
            "all representable supported i+/-m partners, including the final active QP cell"
        ),
        "terminal_child_indices": list(_EXPECTED_TERMINAL_INDICES),
    }:
        raise C4ScoreError("C4 raw endpoint contract is invalid.")
    if metadata.get("component_locality") != {
        "changed_arrays": "QP photon gain, physical loss, net, and the resulting QP residual",
        "inherited_arrays": (
            "QP scattering, QP pair, all three phonon channels, and "
            "the phonon residual remain the accepted C3c arrays"
        ),
        "phonon_residual_bit_exact": True,
        "qp_residual_update": (
            "hybrid_qp_residual = parent_qp_residual + (qpsim_photon_net - parent_photon_net)"
        ),
    }:
        raise C4ScoreError("C4 raw component-locality contract is invalid.")
    expected_frozen_descriptors = {
        name: _array_descriptor(parent.c3_arrays[name]) for name in _frozen_parent_names()
    }
    if metadata.get("frozen_inputs") != {
        "descriptors": expected_frozen_descriptors,
        "mutation_check_after_operator": True,
        "policy": (
            "accepted C3c state, grid, active mask, cell weights, "
            "cell_density, K_plus, and author-form photon arrays are immutable"
        ),
    }:
        raise C4ScoreError("C4 raw frozen-input closure is invalid.")
    if not _json_value_bit_exact(metadata.get("operator_inputs"), operator_inputs):
        raise C4ScoreError("C4 raw operator-input closure is invalid.")
    if metadata.get("parent_bindings") != _expected_parent_bindings(parent):
        raise C4ScoreError("C4 raw parent binding is forged, incomplete, or stale.")
    if metadata.get("units") != {
        "comparison_arrays": "per second",
        "public_native_arrays": "per nanosecond",
        "public_return_contract": (
            "gain includes target Pauli factor; loss_rate multiplies f to form actual loss"
        ),
    }:
        raise C4ScoreError("C4 raw unit contract is invalid.")
    if metadata.get("limitations") != {
        "scope": "one authenticated C3c frozen point only",
        "statement": (
            "No C4 nonlinear root, Newton history, stopping result, "
            "plotted ordinate, 300-point curve, observable change, or "
            "paper-parity claim is made. Non-photon channels are inherited "
            "from C3 and are not re-evaluated."
        ),
    }:
        raise C4ScoreError("C4 raw limitation statement is invalid.")
    descriptors = {name: _array_descriptor(value) for name, value in sorted(arrays.items())}
    if not _json_value_bit_exact(metadata.get("array_descriptors"), descriptors):
        raise C4ScoreError("C4 raw descriptor closure is invalid.")


def _positive_zero(value: np.ndarray) -> bool:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        return bool(np.all(array == 0))
    return bool(np.all(array == 0.0) and not np.any(np.signbit(array)))


def _roundoff_metric(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    scale: np.ndarray | None = None,
) -> tuple[float, float]:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(candidate, dtype=float)
    difference = np.abs(right - left)
    comparison_scale = (
        np.maximum(np.abs(left), np.abs(right)) if scale is None else np.asarray(scale, dtype=float)
    )
    if comparison_scale.shape != difference.shape or np.any(comparison_scale < 0.0):
        raise C4ScoreError("C4 roundoff comparison scale is invalid.")
    permitted = _ROUNDING_MULTIPLIER * np.finfo(float).eps * comparison_scale
    if np.any((comparison_scale == 0.0) & (difference != 0.0)) or np.any(
        (comparison_scale != 0.0) & (difference > permitted)
    ):
        raise C4ScoreError(
            "C4 arithmetic control does not match the accepted C3c photon "
            "array within the declared 64-epsilon source-order bound."
        )
    normalized = np.zeros_like(difference)
    np.divide(
        difference,
        comparison_scale,
        out=normalized,
        where=comparison_scale != 0.0,
    )
    return (
        float(np.max(difference, initial=0.0)),
        float(np.max(normalized, initial=0.0)),
    )


def _conservation_record(
    weights: np.ndarray,
    gain: np.ndarray,
    loss: np.ndarray,
    *,
    label: str,
) -> dict[str, object]:
    w = np.asarray(weights, dtype=float)
    gain_arr = np.asarray(gain, dtype=float)
    loss_arr = np.asarray(loss, dtype=float)
    number_residual = float(np.sum(w * (gain_arr - loss_arr)))
    turnover = float(np.sum(w * (np.abs(gain_arr) + np.abs(loss_arr))))
    if not np.isfinite(number_residual) or not np.isfinite(turnover) or turnover <= 0.0:
        raise C4ScoreError(f"{label} number diagnostic is non-finite or vacuous.")
    relative = abs(number_residual) / turnover
    if relative > _CONSERVATION_LIMIT:
        raise C4ScoreError(f"{label} violates weighted QP-number conservation: {relative:.3e}.")
    return {
        "absolute_number_residual_ueV_s_inv": _float_record(abs(number_residual)),
        "limit_relative_to_turnover": _float_record(_CONSERVATION_LIMIT),
        "relative_to_turnover": _float_record(relative),
        "turnover_ueV_s_inv": _float_record(turnover),
    }


def _operator_comparison(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for field in ("gain", "loss", "net"):
        parent = np.asarray(arrays[f"parent_qp_photon_{field}_s_inv"])
        candidate = np.asarray(arrays[f"qpsim_{field}_s_inv"])
        absolute = np.abs(candidate - parent)
        denominator = float(np.sum(np.abs(parent))) + float(np.sum(np.abs(candidate)))
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise C4ScoreError(f"C4 {field} operator comparison is vacuous.")
        result[field] = {
            "l1_absolute_s_inv": _float_record(float(np.sum(absolute))),
            "linf_absolute_s_inv": _float_record(float(np.max(absolute, initial=0.0))),
            "symmetric_relative_l1": _float_record(float(np.sum(absolute)) / denominator),
        }
    expected = {
        field: {name: _float_record(value) for name, value in metrics.items()}
        for field, metrics in _EXPECTED_OPERATOR_COMPARISON.items()
    }
    if not _json_value_bit_exact(result, expected):
        raise C4ScoreError(
            "C4 public-operator comparison does not reproduce the reviewed "
            "one-point gain/loss/net metrics."
        )
    return result


def _inherited_descriptors(c3_arrays: dict[str, np.ndarray]) -> dict[str, object]:
    names = [
        f"{PARENT_OPERATOR_STAGE_ID}__{channel}__{field}_s_inv"
        for channel in (
            "qp_scattering",
            "qp_pair",
            "phonon_scattering",
            "phonon_pair",
            "phonon_escape",
        )
        for field in ("gain", "loss", "net")
    ]
    names.append(f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv")
    return {name: _array_descriptor(c3_arrays[name]) for name in names}


def build_c4_score(
    c4_bundle_dir: Path,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Recompute all formal frozen C4 arrays and return a checked score."""

    _assert_source_snapshots()
    parent = _accept_parent(
        c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    accepted_stage = _mapping(parent.c3_score.get("stage"), "accepted C3 stage")
    if accepted_stage != {
        "changed_component": "grid_sampling",
        "comparison_stage_id": "C2",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C2",
        "stage_id": "C3",
        "status": "completed",
    }:
        raise C4ScoreError("Accepted parent is not the completed formal C3 stage.")
    raw_metadata, raw_arrays, raw_manifest_sha = load_c4_raw_bundle(c4_bundle_dir)
    operator_inputs, photon_step, n_bar, _c_s, c_ns = _operator_inputs(
        parent.c3_metadata,
        parent.c3_arrays,
    )
    _check_raw_metadata(
        raw_metadata,
        raw_arrays,
        parent=parent,
        operator_inputs=operator_inputs,
    )
    expected_arrays = _expected_arrays(
        parent.c3_arrays,
        photon_step=photon_step,
        n_bar=n_bar,
        c_photon_ns_inv=c_ns,
    )
    if set(expected_arrays) != _EXPECTED_ARRAY_NAMES:
        raise C4ScoreError("Internal C4 expected-array closure is incomplete.")
    for name, expected in expected_arrays.items():
        if not _array_bit_exact(expected, raw_arrays[name]):
            raise C4ScoreError(f"C4 raw array {name!r} does not match independent recomputation.")

    active = np.asarray(parent.c3_arrays["native_active_mask"])
    f = np.asarray(expected_arrays["parent_f"])
    weights = np.asarray(expected_arrays["parent_cell_weights_ueV"])
    density = np.asarray(parent.c3_arrays["native_cell_density_full"])
    if (
        active.dtype.kind != "b"
        or active.shape != (1640,)
        or np.any(active[:20])
        or not np.all(active[20:])
    ):
        raise C4ScoreError("C4 active/guard geometry is invalid.")
    if not (
        _positive_zero(f[:20]) and _positive_zero(weights[:20]) and _positive_zero(density[:20])
    ):
        raise C4ScoreError("C4 frozen guard inputs are not canonical positive zero.")
    for name in (
        "qpsim_gain_ns_inv",
        "qpsim_loss_rate_ns_inv",
        "qpsim_loss_ns_inv",
        "qpsim_net_ns_inv",
        "qpsim_gain_s_inv",
        "qpsim_loss_s_inv",
        "qpsim_net_s_inv",
    ):
        if not _positive_zero(expected_arrays[name][:20]):
            raise C4ScoreError(f"C4 public guard output {name!r} is not positive zero.")

    public_gain_ns = expected_arrays["qpsim_gain_ns_inv"]
    public_loss_rate_ns = expected_arrays["qpsim_loss_rate_ns_inv"]
    public_loss_ns = expected_arrays["qpsim_loss_ns_inv"]
    if (
        np.any(public_gain_ns < 0.0)
        or np.any(public_loss_rate_ns < 0.0)
        or np.any(public_loss_ns < 0.0)
        or np.count_nonzero(public_gain_ns) == 0
        or np.count_nonzero(public_loss_rate_ns) == 0
    ):
        raise C4ScoreError("C4 public photon gain/loss evidence is invalid or vacuous.")
    if not _array_bit_exact(public_loss_ns, public_loss_rate_ns * f):
        raise C4ScoreError("C4 physical loss does not equal returned loss_rate * frozen f.")
    for native_name, per_second_name in (
        ("qpsim_gain_ns_inv", "qpsim_gain_s_inv"),
        ("qpsim_loss_ns_inv", "qpsim_loss_s_inv"),
        ("qpsim_net_ns_inv", "qpsim_net_s_inv"),
    ):
        if not _array_bit_exact(
            expected_arrays[per_second_name],
            expected_arrays[native_name] / SECONDS_PER_NS,
        ):
            raise C4ScoreError(f"C4 unit conversion for {native_name!r} is invalid.")

    semantic_support = np.flatnonzero(expected_arrays["terminal_extension_support_mask"])
    if not np.array_equal(
        semantic_support,
        np.asarray(_EXPECTED_TERMINAL_INDICES, dtype=np.int64),
    ):
        raise C4ScoreError("C4 terminal semantic differential has unexpected support.")
    for name in (
        "terminal_extension_gain_s_inv",
        "terminal_extension_loss_s_inv",
        "terminal_extension_net_s_inv",
    ):
        if not np.any(expected_arrays[name][semantic_support] != 0.0):
            raise C4ScoreError(f"C4 terminal differential {name!r} is vacuous.")

    endpoint_metrics: dict[str, dict[str, object]] = {}
    overall_relative = 0.0
    for field in ("gain", "loss", "net"):
        comparison_scale = None
        if field == "net":
            comparison_scale = np.maximum(
                np.abs(expected_arrays["parent_qp_photon_gain_s_inv"])
                + np.abs(expected_arrays["parent_qp_photon_loss_s_inv"]),
                np.abs(expected_arrays["qpsim_author_endpoint_gain_s_inv"])
                + np.abs(expected_arrays["qpsim_author_endpoint_loss_s_inv"]),
            )
        maximum_absolute, maximum_relative = _roundoff_metric(
            expected_arrays[f"parent_qp_photon_{field}_s_inv"],
            expected_arrays[f"qpsim_author_endpoint_{field}_s_inv"],
            scale=comparison_scale,
        )
        overall_relative = max(overall_relative, maximum_relative)
        endpoint_metrics[field] = {
            "maximum_absolute_s_inv": _float_record(maximum_absolute),
            "maximum_relative": _float_record(maximum_relative),
        }

    if not _array_bit_exact(
        expected_arrays["hybrid_phonon_residual_s_inv"],
        expected_arrays["parent_phonon_residual_s_inv"],
    ):
        raise C4ScoreError("C4 changed the inherited phonon residual.")
    expected_hybrid_qp = (
        expected_arrays["parent_qp_residual_s_inv"] + expected_arrays["operator_delta_net_s_inv"]
    )
    if not _array_bit_exact(
        expected_arrays["hybrid_qp_residual_s_inv"],
        expected_hybrid_qp,
    ):
        raise C4ScoreError("C4 hybrid QP residual update is invalid.")

    public_conservation = _conservation_record(
        weights,
        expected_arrays["qpsim_gain_s_inv"],
        expected_arrays["qpsim_loss_s_inv"],
        label="C4 public photon operator",
    )
    terminal_conservation = _conservation_record(
        weights,
        expected_arrays["terminal_extension_gain_s_inv"],
        expected_arrays["terminal_extension_loss_s_inv"],
        label="C4 terminal extension",
    )
    c3_parent = _mapping(parent.c3_score.get("parent_bindings"), "C3 parent bindings")
    score_parent_bindings = {
        **_expected_parent_bindings(parent),
        "c2_raw_manifest_sha256": _sha256(
            c3_parent.get("c2_raw_manifest_sha256"),
            "C3 parent C2 raw manifest",
        ),
    }
    score = {
        "acceptance": {
            "all_raw_arrays_recomputed_bit_exact": True,
            "c3_parent_replayed_from_c3_and_c2_raw": True,
            "common_rows_within_64eps": True,
            "conservation_within_limit": True,
            "guard_cells_canonical_positive_zero": True,
            "loss_rate_semantics_verified": True,
            "non_photon_locality_verified": True,
            "snap_exact": True,
            "status": "pass",
            "terminal_support_exact": True,
        },
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(expected_arrays.items())
        },
        "component_locality": {
            "semantic_terminal_child_indices": list(_EXPECTED_TERMINAL_INDICES),
            "hybrid_phonon_residual": _array_descriptor(
                expected_arrays["hybrid_phonon_residual_s_inv"]
            ),
            "hybrid_qp_residual": _array_descriptor(expected_arrays["hybrid_qp_residual_s_inv"]),
            "inherited_c3c_arrays": _inherited_descriptors(parent.c3_arrays),
            "statement": _COMPONENT_LOCALITY_STATEMENT,
        },
        "conservation": {
            "public_photon": public_conservation,
            "terminal_extension": terminal_conservation,
        },
        "endpoint_comparison": {
            "arithmetic_control_vs_c3c": endpoint_metrics,
            "maximum_relative": _float_record(overall_relative),
            "roundoff_limit_relative": _float_record(_ROUNDING_MULTIPLIER * np.finfo(float).eps),
            "roundoff_multiplier": _float_record(_ROUNDING_MULTIPLIER),
            "semantic_terminal_child_indices": list(_EXPECTED_TERMINAL_INDICES),
        },
        "limitations": {
            "scope": "one authenticated C3c frozen point only",
            "statement": (
                "No C4 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, observable change, or "
                "paper-parity claim is made. Non-photon channels are inherited "
                "from C3 and are not re-evaluated."
            ),
        },
        "operator_comparison": _operator_comparison(expected_arrays),
        "operator_inputs": operator_inputs,
        "parent_bindings": score_parent_bindings,
        "raw_bundle": {
            "manifest_sha256": raw_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "independent C4 verifier, exact C3/C2 parent replay, and all "
                "raw/public photon source dependencies"
            ),
        },
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
        "units": {
            "comparison_arrays": "per second",
            "public_native_arrays": "per nanosecond",
            "public_return_contract": (
                "gain includes target Pauli factor; loss_rate multiplies f to form actual loss"
            ),
        },
    }
    _recheck_parent(parent)
    _assert_source_snapshots()
    return score


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Return the complete canonical bytes bound by the C4 receipt."""

    return _canonical_json_bytes(score)


def _validate_descriptor(
    value: object,
    label: str,
    *,
    shape: list[int] | None = None,
    bool_dtype: bool = False,
) -> None:
    descriptor = _mapping(value, label)
    _exact_keys(descriptor, {"dtype", "npy_sha256", "shape"}, label)
    dtype_raw = descriptor.get("dtype")
    if not isinstance(dtype_raw, str):
        raise C4ScoreError(f"{label}.dtype must be a string.")
    try:
        dtype = np.dtype(dtype_raw)
    except TypeError as exc:
        raise C4ScoreError(f"{label}.dtype is invalid.") from exc
    if bool_dtype:
        if dtype.kind != "b":
            raise C4ScoreError(f"{label} must describe a boolean array.")
    elif dtype.kind != "f" or dtype.itemsize != 8:
        raise C4ScoreError(f"{label} must describe a binary64 array.")
    raw_shape = descriptor.get("shape")
    if not isinstance(raw_shape, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in raw_shape
    ):
        raise C4ScoreError(f"{label}.shape is invalid.")
    if shape is not None and raw_shape != shape:
        raise C4ScoreError(f"{label}.shape is not {shape}.")
    _sha256(descriptor.get("npy_sha256"), f"{label}.npy_sha256")


def _validate_float_record(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    record = _mapping(value, label)
    _exact_keys(record, {"hex", "value"}, label)
    result = _finite_scalar(
        record.get("value"),
        f"{label}.value",
        positive=positive,
        nonnegative=nonnegative,
    )
    if record.get("hex") != result.hex():
        raise C4ScoreError(f"{label} hexadecimal closure is invalid.")
    return result


def _validate_conservation_record(value: object, label: str) -> None:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "absolute_number_residual_ueV_s_inv",
            "limit_relative_to_turnover",
            "relative_to_turnover",
            "turnover_ueV_s_inv",
        },
        label,
    )
    _validate_float_record(
        record.get("absolute_number_residual_ueV_s_inv"),
        f"{label}.absolute_number_residual_ueV_s_inv",
        nonnegative=True,
    )
    limit = _validate_float_record(
        record.get("limit_relative_to_turnover"),
        f"{label}.limit_relative_to_turnover",
        positive=True,
    )
    relative = _validate_float_record(
        record.get("relative_to_turnover"),
        f"{label}.relative_to_turnover",
        nonnegative=True,
    )
    _validate_float_record(
        record.get("turnover_ueV_s_inv"),
        f"{label}.turnover_ueV_s_inv",
        positive=True,
    )
    if limit != _CONSERVATION_LIMIT or relative > limit:
        raise C4ScoreError(f"{label} acceptance bound is invalid.")


def _validate_score_structure(score: dict[str, Any]) -> None:
    _exact_keys(score, _SCORE_KEYS, "C4 score")
    if score.get("schema") != SCHEMA:
        raise C4ScoreError("Checked C4 score schema is unsupported.")
    if score.get("stage") != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C4ScoreError("Checked C4 stage identity is invalid.")
    if score.get("acceptance") != {
        "all_raw_arrays_recomputed_bit_exact": True,
        "c3_parent_replayed_from_c3_and_c2_raw": True,
        "common_rows_within_64eps": True,
        "conservation_within_limit": True,
        "guard_cells_canonical_positive_zero": True,
        "loss_rate_semantics_verified": True,
        "non_photon_locality_verified": True,
        "snap_exact": True,
        "status": "pass",
        "terminal_support_exact": True,
    }:
        raise C4ScoreError("Checked C4 acceptance closure is invalid.")
    if score.get("source_binding") != {
        "hash_kind": "canonical_sha256_import_time_disk_snapshot",
        "scope": (
            "independent C4 verifier, exact C3/C2 parent replay, and all "
            "raw/public photon source dependencies"
        ),
    }:
        raise C4ScoreError("Checked C4 source-binding policy is invalid.")
    sources = _mapping(score.get("sources"), "C4 score sources")
    if set(sources) != set(_SOURCE_HASHES_AT_IMPORT):
        raise C4ScoreError("Checked C4 source closure is incomplete.")
    for relative, digest in sources.items():
        if source_sha256(REPOSITORY_ROOT / relative) != _sha256(
            digest,
            f"C4 sources.{relative}",
        ):
            raise C4ScoreError("Checked C4 source binding is stale.")

    raw = _mapping(score.get("raw_bundle"), "C4 score raw binding")
    _exact_keys(raw, {"manifest_sha256", "schema"}, "C4 score raw binding")
    if raw.get("schema") != RAW_SCHEMA:
        raise C4ScoreError("Checked C4 raw schema is invalid.")
    _sha256(raw.get("manifest_sha256"), "C4 raw manifest SHA-256")

    descriptors = _mapping(score.get("array_descriptors"), "C4 score descriptors")
    if set(descriptors) != _EXPECTED_ARRAY_NAMES:
        raise C4ScoreError("Checked C4 descriptor closure is invalid.")
    for name, descriptor in descriptors.items():
        expected_shape = (
            [1619]
            if name in {"hybrid_phonon_residual_s_inv", "parent_phonon_residual_s_inv"}
            else [1640]
        )
        _validate_descriptor(
            descriptor,
            f"C4 array_descriptors.{name}",
            shape=expected_shape,
            bool_dtype=name in {"parent_active_mask", "terminal_extension_support_mask"},
        )

    locality = _mapping(score.get("component_locality"), "C4 component locality")
    _exact_keys(
        locality,
        {
            "semantic_terminal_child_indices",
            "hybrid_phonon_residual",
            "hybrid_qp_residual",
            "inherited_c3c_arrays",
            "statement",
        },
        "C4 component locality",
    )
    if locality.get("semantic_terminal_child_indices") != list(_EXPECTED_TERMINAL_INDICES):
        raise C4ScoreError("Checked C4 semantic terminal-index locality is invalid.")
    _validate_descriptor(
        locality.get("hybrid_qp_residual"),
        "C4 component_locality.hybrid_qp_residual",
        shape=[1640],
    )
    _validate_descriptor(
        locality.get("hybrid_phonon_residual"),
        "C4 component_locality.hybrid_phonon_residual",
    )
    inherited = _mapping(
        locality.get("inherited_c3c_arrays"),
        "C4 inherited C3c descriptors",
    )
    expected_inherited_names = {
        f"{PARENT_OPERATOR_STAGE_ID}__{channel}__{field}_s_inv"
        for channel in (
            "qp_scattering",
            "qp_pair",
            "phonon_scattering",
            "phonon_pair",
            "phonon_escape",
        )
        for field in ("gain", "loss", "net")
    }
    expected_inherited_names.add(f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv")
    if set(inherited) != expected_inherited_names:
        raise C4ScoreError("Checked C4 inherited-channel closure is invalid.")
    for name, descriptor in inherited.items():
        _validate_descriptor(
            descriptor,
            f"C4 inherited_c3c_arrays.{name}",
            shape=[1640] if "__qp_" in name else [1619],
        )
    if locality.get("statement") != _COMPONENT_LOCALITY_STATEMENT:
        raise C4ScoreError("Checked C4 locality statement is invalid.")

    conservation = _mapping(score.get("conservation"), "C4 conservation")
    _exact_keys(
        conservation,
        {"public_photon", "terminal_extension"},
        "C4 conservation",
    )
    _validate_conservation_record(
        conservation.get("public_photon"),
        "C4 conservation.public_photon",
    )
    _validate_conservation_record(
        conservation.get("terminal_extension"),
        "C4 conservation.terminal_extension",
    )

    endpoint = _mapping(score.get("endpoint_comparison"), "C4 endpoint comparison")
    _exact_keys(
        endpoint,
        {
            "arithmetic_control_vs_c3c",
            "maximum_relative",
            "roundoff_limit_relative",
            "roundoff_multiplier",
            "semantic_terminal_child_indices",
        },
        "C4 endpoint comparison",
    )
    if endpoint.get("semantic_terminal_child_indices") != list(_EXPECTED_TERMINAL_INDICES):
        raise C4ScoreError("Checked C4 endpoint support is invalid.")
    metrics = _mapping(
        endpoint.get("arithmetic_control_vs_c3c"),
        "C4 endpoint arithmetic metrics",
    )
    _exact_keys(metrics, {"gain", "loss", "net"}, "C4 endpoint arithmetic metrics")
    for field, raw_metric in metrics.items():
        metric = _mapping(raw_metric, f"C4 endpoint metric {field}")
        _exact_keys(
            metric,
            {"maximum_absolute_s_inv", "maximum_relative"},
            f"C4 endpoint metric {field}",
        )
        _validate_float_record(
            metric.get("maximum_absolute_s_inv"),
            f"C4 endpoint metric {field}.maximum_absolute_s_inv",
            nonnegative=True,
        )
        relative_error = _validate_float_record(
            metric.get("maximum_relative"),
            f"C4 endpoint metric {field}.maximum_relative",
            nonnegative=True,
        )
        if relative_error > _ROUNDING_MULTIPLIER * np.finfo(float).eps:
            raise C4ScoreError("Checked C4 endpoint roundoff metric exceeds its bound.")
    overall = _validate_float_record(
        endpoint.get("maximum_relative"),
        "C4 endpoint maximum_relative",
        nonnegative=True,
    )
    limit = _validate_float_record(
        endpoint.get("roundoff_limit_relative"),
        "C4 endpoint roundoff_limit_relative",
        positive=True,
    )
    multiplier = _validate_float_record(
        endpoint.get("roundoff_multiplier"),
        "C4 endpoint roundoff_multiplier",
        positive=True,
    )
    if (
        multiplier != _ROUNDING_MULTIPLIER
        or limit != _ROUNDING_MULTIPLIER * np.finfo(float).eps
        or overall > limit
    ):
        raise C4ScoreError("Checked C4 endpoint acceptance limit is invalid.")

    operator_comparison = _mapping(
        score.get("operator_comparison"),
        "C4 operator comparison",
    )
    _exact_keys(
        operator_comparison,
        {"gain", "loss", "net"},
        "C4 operator comparison",
    )
    for field, raw_metrics in operator_comparison.items():
        metrics = _mapping(raw_metrics, f"C4 operator comparison {field}")
        _exact_keys(
            metrics,
            {
                "l1_absolute_s_inv",
                "linf_absolute_s_inv",
                "symmetric_relative_l1",
            },
            f"C4 operator comparison {field}",
        )
        for name in (
            "l1_absolute_s_inv",
            "linf_absolute_s_inv",
            "symmetric_relative_l1",
        ):
            _validate_float_record(
                metrics.get(name),
                f"C4 operator comparison {field}.{name}",
                nonnegative=True,
            )
    expected_operator_comparison = {
        field: {name: _float_record(value) for name, value in metrics.items()}
        for field, metrics in _EXPECTED_OPERATOR_COMPARISON.items()
    }
    if not _json_value_bit_exact(
        operator_comparison,
        expected_operator_comparison,
    ):
        raise C4ScoreError(
            "Checked C4 operator comparison does not match the independently "
            "recomputed reviewed values."
        )

    operator = _mapping(score.get("operator_inputs"), "C4 operator inputs")
    _exact_keys(
        operator,
        {
            "c_photon_ns_inv",
            "c_photon_s_inv",
            "dE_ueV",
            "gap_ueV",
            "n_bar",
            "omega_0_ueV",
            "photon_step_bins",
            "seconds_per_ns",
            "snap_fraction_of_bin",
        },
        "C4 operator inputs",
    )
    c_ns = _validate_float_record(
        operator.get("c_photon_ns_inv"),
        "C4 c_photon_ns_inv",
        nonnegative=True,
    )
    c_s = _validate_float_record(
        operator.get("c_photon_s_inv"),
        "C4 c_photon_s_inv",
        nonnegative=True,
    )
    dE = _validate_float_record(
        operator.get("dE_ueV"),
        "C4 dE_ueV",
        positive=True,
    )
    gap = _validate_float_record(
        operator.get("gap_ueV"),
        "C4 gap_ueV",
        positive=True,
    )
    _validate_float_record(operator.get("n_bar"), "C4 n_bar", nonnegative=True)
    omega = _validate_float_record(
        operator.get("omega_0_ueV"),
        "C4 omega_0_ueV",
        nonnegative=True,
    )
    step = _strict_int(
        operator.get("photon_step_bins"),
        "C4 photon_step_bins",
        minimum=1,
    )
    seconds = _validate_float_record(
        operator.get("seconds_per_ns"),
        "C4 seconds_per_ns",
        positive=True,
    )
    snap = _validate_float_record(
        operator.get("snap_fraction_of_bin"),
        "C4 snap_fraction_of_bin",
        nonnegative=True,
    )
    if (
        seconds != SECONDS_PER_NS
        or c_ns != c_s * seconds
        or omega != step * dE
        or omega >= 2.0 * gap
        or snap != 0.0
        or step != 20
    ):
        raise C4ScoreError("Checked C4 operator-input relationship is invalid.")

    parent = _mapping(score.get("parent_bindings"), "C4 parent bindings")
    _exact_keys(
        parent,
        {
            "c2_raw_manifest_sha256",
            "c3_operator_stage_id",
            "c3_raw_manifest_sha256",
            "c3_raw_schema",
            "c3_receipt_path",
            "c3_receipt_schema",
            "c3_receipt_sha256",
            "c3_score_path",
            "c3_score_schema",
            "c3_score_sha256",
            "c3_stage_id",
        },
        "C4 parent bindings",
    )
    for field in (
        "c2_raw_manifest_sha256",
        "c3_raw_manifest_sha256",
        "c3_receipt_sha256",
        "c3_score_sha256",
    ):
        _sha256(parent.get(field), f"C4 parent {field}")
    if (
        parent.get("c3_operator_stage_id") != PARENT_OPERATOR_STAGE_ID
        or parent.get("c3_raw_schema") != C3_RAW_SCHEMA
        or parent.get("c3_receipt_schema") != C3_RECEIPT_SCHEMA
        or parent.get("c3_score_schema") != C3_SCORE_SCHEMA
        or parent.get("c3_stage_id") != PARENT_STAGE_ID
        or not isinstance(parent.get("c3_receipt_path"), str)
        or not isinstance(parent.get("c3_score_path"), str)
    ):
        raise C4ScoreError("Checked C4 parent identity is invalid.")

    if score.get("units") != {
        "comparison_arrays": "per second",
        "public_native_arrays": "per nanosecond",
        "public_return_contract": (
            "gain includes target Pauli factor; loss_rate multiplies f to form actual loss"
        ),
    }:
        raise C4ScoreError("Checked C4 units are invalid.")
    if score.get("limitations") != {
        "scope": "one authenticated C3c frozen point only",
        "statement": (
            "No C4 nonlinear root, Newton history, stopping result, "
            "plotted ordinate, 300-point curve, observable change, or "
            "paper-parity claim is made. Non-photon channels are inherited "
            "from C3 and are not re-evaluated."
        ),
    }:
        raise C4ScoreError("Checked C4 limitation statement is invalid.")


def _load_c4_score_unbound(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file_once(path, "checked C4 score")
    score = _parse_json(raw, "checked C4 score")
    if raw != canonical_score_bytes(score):
        raise C4ScoreError("Checked C4 score is not canonical JSON.")
    _validate_score_structure(score)
    return score, raw


def load_c4_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Strictly load the repository C4 score/raw/parent trust anchor."""

    raw = _read_regular_file_once(path, "C4 raw-manifest receipt")
    receipt = _parse_json(raw, "C4 raw-manifest receipt")
    if raw != _canonical_json_bytes(receipt):
        raise C4ScoreError("C4 raw-manifest receipt is not canonical JSON.")
    _exact_keys(
        receipt,
        {"checked_score", "parent_c3", "qualification", "raw_bundle", "schema"},
        "C4 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C4ScoreError("C4 raw-manifest receipt schema is unsupported.")
    if receipt.get("qualification") != (
        "Repository trust anchor for the externally retained C4 raw manifest, "
        "the complete canonical checked-score bytes, and the independently "
        "replayed C3 parent; it does not contain or replace the raw arrays."
    ):
        raise C4ScoreError("C4 raw-manifest receipt qualification is invalid.")
    checked = _mapping(receipt.get("checked_score"), "C4 receipt checked_score")
    _exact_keys(checked, {"file_sha256", "schema"}, "C4 receipt checked_score")
    if checked.get("schema") != SCHEMA:
        raise C4ScoreError("C4 receipt checked-score schema is invalid.")
    _sha256(checked.get("file_sha256"), "C4 receipt score SHA-256")
    raw_bundle = _mapping(receipt.get("raw_bundle"), "C4 receipt raw_bundle")
    _exact_keys(
        raw_bundle,
        {"manifest_sha256", "schema"},
        "C4 receipt raw_bundle",
    )
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C4ScoreError("C4 receipt raw schema is invalid.")
    _sha256(raw_bundle.get("manifest_sha256"), "C4 receipt raw manifest SHA-256")
    parent = _mapping(receipt.get("parent_c3"), "C4 receipt parent_c3")
    _exact_keys(
        parent,
        {
            "raw_manifest_sha256",
            "raw_schema",
            "receipt_file_sha256",
            "receipt_schema",
            "score_file_sha256",
            "score_schema",
        },
        "C4 receipt parent_c3",
    )
    if (
        parent.get("raw_schema") != C3_RAW_SCHEMA
        or parent.get("receipt_schema") != C3_RECEIPT_SCHEMA
        or parent.get("score_schema") != C3_SCORE_SCHEMA
    ):
        raise C4ScoreError("C4 receipt C3 schemas are invalid.")
    for field in (
        "raw_manifest_sha256",
        "receipt_file_sha256",
        "score_file_sha256",
    ):
        _sha256(parent.get(field), f"C4 receipt parent_c3.{field}")
    return receipt


def _receipt_parent_from_score(score: dict[str, Any]) -> dict[str, object]:
    parent = _mapping(score.get("parent_bindings"), "C4 parent bindings")
    return {
        "raw_manifest_sha256": parent.get("c3_raw_manifest_sha256"),
        "raw_schema": parent.get("c3_raw_schema"),
        "receipt_file_sha256": parent.get("c3_receipt_sha256"),
        "receipt_schema": parent.get("c3_receipt_schema"),
        "score_file_sha256": parent.get("c3_score_sha256"),
        "score_schema": parent.get("c3_score_schema"),
    }


def load_c4_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Load a checked C4 score and bind it to current canonical C3 anchors."""

    score, score_raw = _load_c4_score_unbound(path)
    receipt = load_c4_receipt(receipt_path)
    checked = _mapping(receipt.get("checked_score"), "C4 receipt checked_score")
    if hashlib.sha256(score_raw).hexdigest() != checked.get("file_sha256"):
        raise C4ScoreError("Checked C4 score bytes do not match the selected receipt.")
    if score.get("raw_bundle") != receipt.get("raw_bundle"):
        raise C4ScoreError("Checked C4 raw binding does not match the selected receipt.")
    if _receipt_parent_from_score(score) != receipt.get("parent_c3"):
        raise C4ScoreError("Checked C4 C3-parent binding does not match the receipt.")

    parent = _mapping(score.get("parent_bindings"), "C4 parent bindings")
    expected_c3_score_path = DEFAULT_C3_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
    expected_c3_receipt_path = DEFAULT_C3_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
    if (
        parent.get("c3_score_path") != expected_c3_score_path
        or parent.get("c3_receipt_path") != expected_c3_receipt_path
    ):
        raise C4ScoreError("Checked C4 does not bind the canonical C3 parent paths.")
    c3_score_path, c3_score_bytes = _repository_file_snapshot(
        DEFAULT_C3_SCORE,
        "canonical C3 score",
    )
    c3_receipt_path, c3_receipt_bytes = _repository_file_snapshot(
        DEFAULT_C3_RECEIPT,
        "canonical C3 receipt",
    )
    accepted_c3 = load_c3_score(c3_score_path, receipt_path=c3_receipt_path)
    accepted_receipt = load_c3_receipt(c3_receipt_path)
    accepted_raw = _mapping(accepted_c3.get("raw_bundle"), "accepted C3 raw binding")
    if (
        hashlib.sha256(c3_score_bytes).hexdigest() != parent.get("c3_score_sha256")
        or hashlib.sha256(c3_receipt_bytes).hexdigest() != parent.get("c3_receipt_sha256")
        or accepted_c3.get("schema") != parent.get("c3_score_schema")
        or accepted_receipt.get("schema") != parent.get("c3_receipt_schema")
        or accepted_raw.get("schema") != parent.get("c3_raw_schema")
        or accepted_raw.get("manifest_sha256") != parent.get("c3_raw_manifest_sha256")
    ):
        raise C4ScoreError("Checked C4 canonical C3 binding is stale.")
    c3_parent = _mapping(accepted_c3.get("parent_bindings"), "accepted C3 parent")
    if c3_parent.get("c2_raw_manifest_sha256") != parent.get("c2_raw_manifest_sha256"):
        raise C4ScoreError("Checked C4 inherited C2 binding is stale.")
    _assert_file_snapshot(c3_score_path, c3_score_bytes, "canonical C3 score")
    _assert_file_snapshot(c3_receipt_path, c3_receipt_bytes, "canonical C3 receipt")
    return score


def build_c4_receipt(
    score_path: Path = DEFAULT_SCORE,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Build a receipt only after independently reproducing the C4 score."""

    score, score_raw = _load_c4_score_unbound(score_path)
    rebuilt = build_c4_score(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_score_bytes(rebuilt) != score_raw:
        raise C4ScoreError(
            "C4 receipt refuses score bytes that do not independently reproduce "
            "from the selected C4, C3, and C2 raw evidence."
        )
    raw_bundle = _mapping(score.get("raw_bundle"), "C4 score raw binding")
    return {
        "checked_score": {
            "file_sha256": hashlib.sha256(score_raw).hexdigest(),
            "schema": SCHEMA,
        },
        "parent_c3": _receipt_parent_from_score(score),
        "qualification": (
            "Repository trust anchor for the externally retained C4 raw manifest, "
            "the complete canonical checked-score bytes, and the independently "
            "replayed C3 parent; it does not contain or replace the raw arrays."
        ),
        "raw_bundle": dict(raw_bundle),
        "schema": RECEIPT_SCHEMA,
    }


def _fsync_parent_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_exclusive_write(path: Path, content: bytes) -> Path:
    """Publish complete bytes atomically without replacing an existing path."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"C4 output already exists: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        _fsync_parent_directory(target.parent)
    except BaseException:
        if target.exists() and target.stat().st_ino == temporary.stat().st_ino:
            target.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_c4_score(
    output_path: Path,
    c4_bundle_dir: Path,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    score = build_c4_score(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, canonical_score_bytes(score))


def write_c4_receipt(
    output_path: Path,
    *,
    score_path: Path = DEFAULT_SCORE,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    receipt = build_c4_receipt(
        score_path,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, _canonical_json_bytes(receipt))


def _add_parent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c3-score", type=Path, default=DEFAULT_C3_SCORE)
    parser.add_argument("--c3-receipt", type=Path, default=DEFAULT_C3_RECEIPT)
    parser.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    parser.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="build the checked C4 score")
    _add_parent_arguments(score)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    receipt = subparsers.add_parser("receipt", help="build the C4 receipt")
    _add_parent_arguments(receipt)
    receipt.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    receipt.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    common = {
        "c3_bundle_dir": args.c3_bundle,
        "c2_bundle_dir": args.c2_bundle,
        "c3_score_path": args.c3_score,
        "c3_receipt_path": args.c3_receipt,
        "c2_score_path": args.c2_score,
        "c2_receipt_path": args.c2_receipt,
    }
    if args.command == "receipt":
        result = write_c4_receipt(
            args.output,
            score_path=args.score,
            c4_bundle_dir=args.c4_bundle,
            **common,
        )
    else:
        result = write_c4_score(
            args.output,
            args.c4_bundle,
            **common,
        )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
