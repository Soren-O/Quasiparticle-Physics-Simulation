"""Independently verify the formal Figure 6 C5 QP-phonon substitution.

The C5 producer intentionally lives in another module.  This verifier does
not import that producer or the changed public qpsim phonon collision APIs for
its scientific replay.  It strictly loads the externally retained raw bundle,
replays the accepted C4/C3/C2 chain, and independently transcribes:

* the frozen center-grid phonon-frequency map and occupations;
* the QP-side scattering (K-) and pair/recombination (K+) kernels;
* public gain/loss-rate/physical-loss/net semantics;
* the author's scattering Pauli-term bookkeeping rebucketing.

Public qpsim evaluates the final contractions with matrix-vector products.
Their last bits may depend on the BLAS reduction order, so retained public
arrays are byte-bound to their raw manifest while comparison to a clean-room
``math.fsum`` reduction uses a predeclared IEEE-754 gamma bound.  Derived raw
relationships remain byte-exact.

C5 is one frozen-state operator differential.  It does not run Newton, alter
the frozen state, replace the phonon equation, or claim a C5 nonlinear root,
stopping history, observable, plotted ordinate, curve, or paper parity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
    load_c3_raw_bundle,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as DEFAULT_C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_SCORE as DEFAULT_C4_SCORE,
)
from validation.fischer_2023.fig6_author_c4_score import (
    RAW_SCHEMA as C4_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    RECEIPT_SCHEMA as C4_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    SCHEMA as C4_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    build_c4_score,
    load_c4_raw_bundle,
    load_c4_score,
)
from validation.fischer_2023.fig6_author_c4_score import (
    canonical_score_bytes as canonical_c4_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_SCHEMA = "qpsim.fischer2023.fig6-author-c5-qp-phonon-bundle.v1"
SCHEMA = "qpsim.fischer2023.fig6-author-c5-qp-phonon-score.v1"
RECEIPT_SCHEMA = (
    "qpsim.fischer2023.fig6-author-c5-qp-phonon-raw-manifest-receipt.v1"
)
DEFAULT_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c5-qp-phonon-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c5-raw-manifest-receipt.json"
)

STAGE_ID = "C5"
PARENT_STAGE_ID = "C4"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "qp_phonon_operator"
SECONDS_PER_NS = 1.0e-9
N_QP = 1640
N_ACTIVE = 1620
N_OMEGA = 3600
ACTIVE_START = 20
AUTHOR_OMEGA_STOP = 1619

_ARRAY_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FLOAT_EPS = float(np.finfo(np.float64).eps)
# A non-negative dot product forms at most a handful of rounded products per
# term plus one reduction.  8*N+64 is deliberately conservative while still
# more than three orders tighter than the semantic Pauli rebucketing at this
# point.  This is fixed before observing any C5 raw output.
_REDUCTION_OPERATION_BUDGET = 8 * N_QP + 64
_REDUCTION_GAMMA = (
    _REDUCTION_OPERATION_BUDGET
    * _FLOAT_EPS
    / (1.0 - _REDUCTION_OPERATION_BUDGET * _FLOAT_EPS)
)
_SCATTERING_CONSERVATION_LIMIT = 1.0e-12
_NET_PARITY_LIMIT = 1.0e-12
_BUCKET_PARITY_LIMIT = 1.0e-12
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_EXPECTED_RAW_MANIFEST_SHA256 = (
    "167074651996701020e555c3f0b1ee8ca6148e5f899477edf902fa1e08340a61"
)
_EXPECTED_EVIDENCE_DIGEST = (
    "02610733ee5b2c09b2bf1118bcc08411bfa7119ac874ef59e2957422801ab014"
)
_EXPECTED_CANONICAL_METRICS = {
    "combined_net_symmetric_relative_l1": float.fromhex(
        "0x1.474ccb7c7917ep-51"
    ),
    "pair_net_symmetric_relative_l1": float.fromhex("0x1.0236958d9f70ep-51"),
    "pair_weighted_net_s_inv_ueV": float.fromhex("-0x1.63fb0ac2527aap-3"),
    "scattering_net_symmetric_relative_l1": float.fromhex(
        "0x1.4795a25cf323ep-51"
    ),
    "scattering_weighted_number_relative": float.fromhex(
        "0x1.ae63aaeb5112fp-54"
    ),
}
_EXPECTED_RESIDUAL_NPY_SHA256 = {
    "c5p_qp_residual": "ea7be5cc6c5386836b2903f58f8c7c1c4a12316b21785c15bded3e1f3b6b2ab0",
    "c5s_qp_residual": "442fb79a8039ae7025d72db0a597bd83901a784389f659a946e9ec9fa0a9d913",
    "c5sp_qp_residual": "9f8e4f18352c0dec82c1ce8db389cab9b87ad7d336541df885f5d64f290365a7",
    "frozen_phonon_residual": (
        "8902664899769d6d3b66980141f493acdc196d4a00520be7628c010035e91457"
    ),
}
# These four values are producer diagnostics retained verbatim inside the
# accepted raw manifest.  The producer used NumPy reductions for them, so an
# independent verifier must not recompute their last bits with the local
# NumPy build.  They are authenticated by the pinned manifest; the verifier's
# actual conservation results are recomputed separately with ``math.fsum``.
_EXPECTED_RAW_BOOKKEEPING_METRICS = {
    "pair_weighted_net_s_inv_ueV": float.fromhex("-0x1.63fb0ac2527aap-3"),
    "pair_weighted_turnover_s_inv_ueV": float.fromhex("0x1.5aba4035e51cep+2"),
    "scattering_absolute_drift_s_inv_ueV": float.fromhex(
        "0x1.9c581fbad54dep-45"
    ),
    "scattering_relative_drift": float.fromhex("0x1.cc66b6a31ab5ep-54"),
    "scattering_turnover_s_inv_ueV": float.fromhex("0x1.ca8e9dc498303p+8"),
}
_VERIFIER_RUNTIME_CONTRACT = {
    "arithmetic": (
        "fixed-order Python math.fsum reductions plus elementwise IEEE-754 "
        "gamma bounds"
    ),
    "host_runtime_binding": "not pinned",
    "qualification": (
        "The retained public arrays are authenticated to the producer runtime. "
        "Verifier acceptance is host-independent: it does not demand identical "
        "BLAS reduction last bits or a matching OS/Python/NumPy build."
    ),
}
_EXPECTED_SOURCE_BINDING = {
    "hash_kind": "canonical_sha256_import_time_disk_snapshot",
    "scope": (
        "complete qpsim Python/material source tree, C5 producer, C2/C3/C4 "
        "replay verifiers, and provenance helper"
    ),
}

_FLOAT_1640 = ("<f8", (N_QP,))
_BOOL_1640 = ("|b1", (N_QP,))
_FLOAT_3600 = ("<f8", (N_OMEGA,))
_BOOL_3600 = ("|b1", (N_OMEGA,))
_FLOAT_MATRIX = ("<f8", (N_QP, N_QP))
_INT64_MATRIX = ("<i8", (N_QP, N_QP))
_INT8_MATRIX = ("|i1", (N_QP, N_QP))

_ARRAY_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "parent_E_centers_ueV": _FLOAT_1640,
    "parent_dE_ueV": _FLOAT_1640,
    "parent_f": _FLOAT_1640,
    "parent_active_mask": _BOOL_1640,
    "parent_cell_weights_ueV": _FLOAT_1640,
    "parent_projected_n_phonon": _FLOAT_3600,
    "parent_legacy_phonon_support_mask": _BOOL_3600,
    "qpsim_omega_ueV": _FLOAT_3600,
    "qpsim_omega_idx_diff": _INT64_MATRIX,
    "qpsim_omega_idx_sum": _INT64_MATRIX,
    "qpsim_diff_sign": _INT8_MATRIX,
    "qpsim_N_p": _FLOAT_MATRIX,
    "qpsim_N_emit": _FLOAT_MATRIX,
    "qpsim_N_abs": _FLOAT_MATRIX,
    "qpsim_qp_scattering_kernel_ns_inv_ueV_inv": _FLOAT_MATRIX,
    "qpsim_qp_pair_kernel_ns_inv_ueV_inv": _FLOAT_MATRIX,
    "parent_qp_scattering_gain_s_inv": _FLOAT_1640,
    "parent_qp_scattering_loss_s_inv": _FLOAT_1640,
    "parent_qp_scattering_net_s_inv": _FLOAT_1640,
    "parent_qp_pair_gain_s_inv": _FLOAT_1640,
    "parent_qp_pair_loss_s_inv": _FLOAT_1640,
    "parent_qp_pair_net_s_inv": _FLOAT_1640,
    "parent_public_qp_photon_gain_s_inv": _FLOAT_1640,
    "parent_public_qp_photon_loss_s_inv": _FLOAT_1640,
    "parent_public_qp_photon_net_s_inv": _FLOAT_1640,
    "parent_qp_residual_s_inv": _FLOAT_1640,
    "parent_phonon_residual_s_inv": ("<f8", (AUTHOR_OMEGA_STOP,)),
    "qp_scattering_delta_gain_s_inv": _FLOAT_1640,
    "qp_scattering_delta_loss_s_inv": _FLOAT_1640,
    "qp_scattering_delta_net_s_inv": _FLOAT_1640,
    "qp_pair_delta_gain_s_inv": _FLOAT_1640,
    "qp_pair_delta_loss_s_inv": _FLOAT_1640,
    "qp_pair_delta_net_s_inv": _FLOAT_1640,
    "scattering_pauli_cross_term_s_inv": _FLOAT_1640,
    "parent_qp_scattering_rebucketed_gain_s_inv": _FLOAT_1640,
    "parent_qp_scattering_rebucketed_loss_s_inv": _FLOAT_1640,
    "qp_scattering_rebucketed_delta_gain_s_inv": _FLOAT_1640,
    "qp_scattering_rebucketed_delta_loss_s_inv": _FLOAT_1640,
    "c5s_qp_residual_s_inv": _FLOAT_1640,
    "c5p_qp_residual_s_inv": _FLOAT_1640,
    "c5sp_qp_residual_s_inv": _FLOAT_1640,
    "c5sp_phonon_residual_s_inv": ("<f8", (AUTHOR_OMEGA_STOP,)),
}
for _channel in ("scattering", "pair"):
    for _field in ("gain", "loss_rate", "loss", "net"):
        _ARRAY_SPECS[f"qpsim_qp_{_channel}_{_field}_ns_inv"] = _FLOAT_1640
        _ARRAY_SPECS[f"qpsim_qp_{_channel}_{_field}_s_inv"] = _FLOAT_1640

_EXPECTED_ARRAY_NAMES = frozenset(_ARRAY_SPECS)
if len(_EXPECTED_ARRAY_NAMES) != 58:  # pragma: no cover - import-time invariant
    raise RuntimeError("Internal C5 raw array closure must contain exactly 58 arrays.")

_RAW_METADATA_KEYS = {
    "array_descriptors",
    "bookkeeping_contract",
    "comparison_contract",
    "component_locality",
    "coordinate_contract",
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
    "bookkeeping",
    "channel_comparison",
    "component_locality",
    "conservation",
    "contracts",
    "limitations",
    "map_identity",
    "operator_inputs",
    "parent_bindings",
    "raw_bundle",
    "rounding_contract",
    "runtime",
    "schema",
    "source_binding",
    "sources",
    "stage",
    "units",
}

_C5_PRODUCER_SOURCE = (
    REPOSITORY_ROOT / "validation/fischer_2023/fig6_author_c5_bundle.py"
)
_TRANSITIVE_VALIDATION_SOURCES = tuple(
    REPOSITORY_ROOT / relative
    for relative in (
        "validation/__init__.py",
        "validation/author_source.py",
        "validation/reproduction_ladder.py",
        "validation/fischer_2023/__init__.py",
        "validation/fischer_2023/fig6_author_adapter.py",
        "validation/fischer_2023/fig6_author_frozen_state.py",
        "validation/fischer_2023/fig6_author_c0_bundle.py",
        "validation/fischer_2023/fig6_author_c0_summary.py",
        "validation/fischer_2023/fig6_author_c1_score.py",
        "validation/fischer_2023/fig6_author_c2_bundle.py",
        "validation/fischer_2023/fig6_author_c2_parameters.py",
        "validation/fischer_2023/fig6_author_c2_score.py",
        "validation/fischer_2023/fig6_author_c3_bundle.py",
        "validation/fischer_2023/fig6_author_c3_score.py",
        "validation/fischer_2023/fig6_author_c4_bundle.py",
        "validation/fischer_2023/fig6_author_c4_score.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/reference_models/__init__.py",
        "validation/reference_models/fischer_2023/__init__.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    )
)
_RAW_SOURCE_HASHES_AT_IMPORT = source_manifest(
    _C5_PRODUCER_SOURCE,
    extra_validation_modules=_TRANSITIVE_VALIDATION_SOURCES,
)
_SOURCE_HASHES_AT_IMPORT = source_manifest(
    Path(__file__).resolve(),
    extra_validation_modules=(
        *_TRANSITIVE_VALIDATION_SOURCES,
        _C5_PRODUCER_SOURCE,
    ),
)
_SOURCE_BYTES_AT_IMPORT = {
    relative: canonical_source_bytes(REPOSITORY_ROOT / relative)
    for relative in _SOURCE_HASHES_AT_IMPORT
}
if any(
    hashlib.sha256(_SOURCE_BYTES_AT_IMPORT[relative]).hexdigest() != digest
    for relative, digest in _SOURCE_HASHES_AT_IMPORT.items()
):  # pragma: no cover - import-time provenance invariant
    raise RuntimeError("C5 source hashes changed during import.")
_VERIFIER_RELATIVE = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
_RAW_SOURCES_FROM_VERIFIER_CLOSURE = {
    relative: digest
    for relative, digest in _SOURCE_HASHES_AT_IMPORT.items()
    if relative != _VERIFIER_RELATIVE
}
if (
    _RAW_SOURCES_FROM_VERIFIER_CLOSURE != _RAW_SOURCE_HASHES_AT_IMPORT
):  # pragma: no cover - import-time provenance invariant
    raise RuntimeError("C5 verifier and producer source closures are inconsistent.")


class C5ScoreError(ValueError):
    """The C5 raw evidence, parent chain, score, or receipt is malformed."""


@dataclass(frozen=True)
class _DirectoryState:
    root_identity: tuple[int, int, int, int, int]
    entries: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass
class _ParentContext:
    c4_metadata: dict[str, Any]
    c4_arrays: dict[str, np.ndarray]
    c4_manifest_sha256: str
    c4_score: dict[str, Any]
    c4_score_path: Path
    c4_score_bytes: bytes
    c4_receipt_path: Path
    c4_receipt_bytes: bytes
    c3_score_path: Path
    c3_score_bytes: bytes
    c3_receipt_path: Path
    c3_receipt_bytes: bytes
    c2_score_path: Path
    c2_score_bytes: bytes
    c2_receipt_path: Path
    c2_receipt_bytes: bytes
    c3_metadata: dict[str, Any]
    c3_arrays: dict[str, np.ndarray]
    c3_manifest_sha256: str
    c4_bundle_dir: Path
    c4_directory_state: _DirectoryState
    c3_bundle_dir: Path
    c3_directory_state: _DirectoryState
    c2_bundle_dir: Path
    c2_directory_state: _DirectoryState


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C5ScoreError(f"C5 numerical source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C5ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C5ScoreError(f"Forbidden non-finite JSON constant {token!r}.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C5ScoreError(f"{label} is not strict UTF-8 JSON.") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C5ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise C5ScoreError(f"{label} keys are invalid; missing={missing}, extra={extra}.")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C5ScoreError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise C5ScoreError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise C5ScoreError(f"{label} must be >= {minimum}.")
    return value


def _finite_scalar(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C5ScoreError(f"{label} must be a finite scalar.")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise C5ScoreError(f"{label} must be a finite {qualifier}scalar.")
    return result


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


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


def _positive_zero_copy(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    if result.dtype.kind == "f":
        result[result == 0.0] = 0.0
    return result


def _json_value_bit_exact(reference: object, candidate: object) -> bool:
    if isinstance(reference, dict):
        if not isinstance(candidate, dict) or set(reference) != set(candidate):
            return False
        return all(
            _json_value_bit_exact(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list):
        return isinstance(candidate, list) and len(reference) == len(candidate) and all(
            _json_value_bit_exact(left, right)
            for left, right in zip(reference, candidate, strict=True)
        )
    if isinstance(reference, float):
        return (
            isinstance(candidate, float)
            and np.float64(reference).view(np.uint64)
            == np.float64(candidate).view(np.uint64)
        )
    # JSON considers ``true == 1`` in Python.  Evidence does not: primitive
    # types are part of the canonical contract, so Boolean/integer
    # substitutions must fail even when their Python values compare equal.
    return type(reference) is type(candidate) and reference == candidate


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    if not np.isfinite(result):
        raise C5ScoreError("C5 scalar records must be finite.")
    return {"hex": result.hex(), "value": result}


def _fixed_sum(value: np.ndarray) -> float:
    """Return a platform-stable row-major float64 sum."""

    array = np.asarray(value, dtype=np.float64)
    return float(math.fsum(float(item) for item in array.ravel(order="C")))


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
    )


def _read_regular_file_once(path: Path, label: str) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise C5ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C5ScoreError(f"{label} must be a regular non-symlink file.")
    try:
        with candidate.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = candidate.lstat()
    except OSError as exc:
        raise C5ScoreError(f"{label} changed or became unreadable.") from exc
    if not (
        _stat_identity(before)
        == _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(after)
    ):
        raise C5ScoreError(f"{label} changed while it was being read.")
    return content


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C5ScoreError(f"{label} must stay inside the repository.") from exc
    if resolved != Path(path).absolute() or resolved.is_symlink():
        raise C5ScoreError(f"{label} is unsafe or a symlink.")
    return resolved, _read_regular_file_once(resolved, label)


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    if _read_regular_file_once(path, label) != expected:
        raise C5ScoreError(f"{label} changed during C5 verification.")


def _directory_state(path: Path, label: str) -> tuple[Path, _DirectoryState]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise C5ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise C5ScoreError(f"{label} must be a non-symlink directory.")
    root = candidate.resolve()
    if root != candidate.absolute() or root.is_symlink():
        raise C5ScoreError(f"{label} is unsafe or a symlink.")
    entries: list[tuple[str, tuple[int, int, int, int, int]]] = []
    try:
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise C5ScoreError(f"{label} contains a symlink.")
            entries.append((child.name, _stat_identity(child_stat)))
        after = root.lstat()
    except OSError as exc:
        raise C5ScoreError(f"{label} changed while being enumerated.") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise C5ScoreError(f"{label} changed while being enumerated.")
    return root, _DirectoryState(_stat_identity(after), tuple(entries))


def _assert_directory_state(path: Path, expected: _DirectoryState, label: str) -> None:
    root, actual = _directory_state(path, label)
    del root
    if actual != expected:
        raise C5ScoreError(f"{label} changed during C5 verification.")


def load_c5_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one closed canonical NPY-v3 C5 raw bundle."""

    root, before = _directory_state(bundle_dir, "C5 raw bundle")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C5 raw manifest")
    manifest = _parse_json(manifest_raw, "C5 raw manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C5 raw manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C5ScoreError("C5 raw manifest schema is unsupported.")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise C5ScoreError("C5 raw manifest is not canonical JSON.")
    files = _mapping(manifest.get("files"), "C5 raw manifest files")
    metadata = _mapping(manifest.get("metadata"), "C5 raw metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C5 raw metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C5ScoreError("C5 raw metadata schema is unsupported.")
    expected_filenames = {f"{name}.npy" for name in _EXPECTED_ARRAY_NAMES}
    if set(files) != expected_filenames or len(files) != 58:
        raise C5ScoreError("C5 raw file closure is invalid.")
    if {name for name, _identity in before.entries} != expected_filenames | {
        "manifest.json"
    }:
        raise C5ScoreError("C5 raw directory closure is invalid.")
    if any(not stat.S_ISREG(identity[2]) for _name, identity in before.entries):
        raise C5ScoreError("C5 raw bundle contains a non-file entry.")

    arrays: dict[str, np.ndarray] = {}
    for filename in sorted(expected_filenames):
        name = filename[:-4]
        if (
            Path(filename).name != filename
            or _ARRAY_NAME_RE.fullmatch(name) is None
            or not filename.endswith(".npy")
        ):
            raise C5ScoreError(f"Unsafe C5 raw filename {filename!r}.")
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        expected_sha = _sha256(record.get("sha256"), f"files.{filename}.sha256")
        expected_size = _strict_int(
            record.get("size_bytes"),
            f"files.{filename}.size_bytes",
            minimum=1,
        )
        content = _read_regular_file_once(root / filename, f"C5 raw {filename}")
        if (
            len(content) != expected_size
            or hashlib.sha256(content).hexdigest() != expected_sha
        ):
            raise C5ScoreError(f"C5 raw file {filename!r} failed manifest binding.")
        if len(content) < 8 or content[:8] != b"\x93NUMPY\x03\x00":
            raise C5ScoreError(f"C5 raw file {filename!r} is not canonical NPY v3.")
        try:
            stream = io.BytesIO(content)
            loaded = np.lib.format.read_array(stream, allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise C5ScoreError(f"Cannot load C5 raw array {filename!r}.") from exc
        if stream.tell() != len(content):
            raise C5ScoreError(f"C5 raw file {filename!r} contains trailing bytes.")
        array = np.asarray(loaded)
        expected_dtype, expected_shape = _ARRAY_SPECS[name]
        if array.dtype.str != expected_dtype or array.shape != expected_shape:
            raise C5ScoreError(
                f"C5 raw {name!r} expected dtype/shape "
                f"{expected_dtype}/{expected_shape}, got "
                f"{array.dtype.str}/{array.shape}."
            )
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise C5ScoreError(f"C5 raw array {filename!r} contains non-finite values.")
        if array.dtype.kind == "f" and np.any((array == 0.0) & np.signbit(array)):
            raise C5ScoreError(
                f"C5 raw array {filename!r} contains non-canonical signed zero."
            )
        if _npy_bytes(array) != content:
            raise C5ScoreError(
                f"C5 raw file {filename!r} is not byte-canonical NPY v3."
            )
        arrays[name] = array
    descriptors = _mapping(metadata.get("array_descriptors"), "C5 raw descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C5ScoreError("C5 raw array descriptors are incomplete, forged, or stale.")
    _assert_directory_state(root, before, "C5 raw bundle")
    return metadata, arrays, hashlib.sha256(manifest_raw).hexdigest()


def _accept_parent(
    c4_bundle_dir: Path,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path,
    c4_receipt_path: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
) -> _ParentContext:
    """Replay the selected C4/C3/C2 chain and bind canonical C4 bytes."""

    c4_root, c4_state = _directory_state(c4_bundle_dir, "selected C4 raw bundle")
    c3_root, c3_state = _directory_state(c3_bundle_dir, "selected C3 raw bundle")
    c2_root, c2_state = _directory_state(c2_bundle_dir, "selected C2 raw bundle")
    checked_c4_path, checked_c4_bytes = _repository_file_snapshot(
        c4_score_path,
        "checked C4 score",
    )
    checked_c4_receipt_path, checked_c4_receipt_bytes = _repository_file_snapshot(
        c4_receipt_path,
        "checked C4 receipt",
    )
    checked_c3_path, checked_c3_bytes = _repository_file_snapshot(
        c3_score_path,
        "checked C3 score",
    )
    checked_c3_receipt_path, checked_c3_receipt_bytes = _repository_file_snapshot(
        c3_receipt_path,
        "checked C3 receipt",
    )
    checked_c2_path, checked_c2_bytes = _repository_file_snapshot(
        c2_score_path,
        "checked C2 score",
    )
    checked_c2_receipt_path, checked_c2_receipt_bytes = _repository_file_snapshot(
        c2_receipt_path,
        "checked C2 receipt",
    )
    accepted_c4 = load_c4_score(
        checked_c4_path,
        receipt_path=checked_c4_receipt_path,
    )
    rebuilt_c4 = build_c4_score(
        c4_root,
        c3_bundle_dir=c3_root,
        c2_bundle_dir=c2_root,
        c3_score_path=checked_c3_path,
        c3_receipt_path=checked_c3_receipt_path,
        c2_score_path=checked_c2_path,
        c2_receipt_path=checked_c2_receipt_path,
    )
    if canonical_c4_score_bytes(rebuilt_c4) != checked_c4_bytes:
        raise C5ScoreError(
            "Selected C4/C3/C2 raw evidence does not reproduce the complete "
            "checked C4 score bytes."
        )
    if not _json_value_bit_exact(accepted_c4, rebuilt_c4):
        raise C5ScoreError("Accepted and independently replayed C4 scores disagree.")
    c4_metadata, c4_arrays, c4_manifest_sha = load_c4_raw_bundle(c4_root)
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_root)
    raw_binding = _mapping(accepted_c4.get("raw_bundle"), "accepted C4 raw binding")
    if (
        raw_binding.get("schema") != C4_RAW_SCHEMA
        or raw_binding.get("manifest_sha256") != c4_manifest_sha
    ):
        raise C5ScoreError("Selected C4 raw bundle is not the accepted C4 evidence.")
    parent_binding = _mapping(
        accepted_c4.get("parent_bindings"),
        "accepted C4 parent binding",
    )
    if (
        parent_binding.get("c3_raw_schema") != C3_RAW_SCHEMA
        or parent_binding.get("c3_raw_manifest_sha256") != c3_manifest_sha
    ):
        raise C5ScoreError("Selected C3 raw bundle is not C4's accepted parent.")
    stage = _mapping(accepted_c4.get("stage"), "accepted C4 stage")
    if stage != {
        "changed_component": "photon_operator",
        "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C3",
        "stage_id": "C4",
        "status": "completed",
    }:
        raise C5ScoreError("Accepted parent is not the completed formal C4 stage.")
    _assert_file_snapshot(checked_c4_path, checked_c4_bytes, "checked C4 score")
    _assert_file_snapshot(
        checked_c4_receipt_path,
        checked_c4_receipt_bytes,
        "checked C4 receipt",
    )
    _assert_file_snapshot(checked_c3_path, checked_c3_bytes, "checked C3 score")
    _assert_file_snapshot(
        checked_c3_receipt_path,
        checked_c3_receipt_bytes,
        "checked C3 receipt",
    )
    _assert_file_snapshot(checked_c2_path, checked_c2_bytes, "checked C2 score")
    _assert_file_snapshot(
        checked_c2_receipt_path,
        checked_c2_receipt_bytes,
        "checked C2 receipt",
    )
    _assert_directory_state(c4_root, c4_state, "selected C4 raw bundle")
    _assert_directory_state(c3_root, c3_state, "selected C3 raw bundle")
    _assert_directory_state(c2_root, c2_state, "selected C2 raw bundle")
    return _ParentContext(
        c4_metadata=c4_metadata,
        c4_arrays=c4_arrays,
        c4_manifest_sha256=c4_manifest_sha,
        c4_score=accepted_c4,
        c4_score_path=checked_c4_path,
        c4_score_bytes=checked_c4_bytes,
        c4_receipt_path=checked_c4_receipt_path,
        c4_receipt_bytes=checked_c4_receipt_bytes,
        c3_score_path=checked_c3_path,
        c3_score_bytes=checked_c3_bytes,
        c3_receipt_path=checked_c3_receipt_path,
        c3_receipt_bytes=checked_c3_receipt_bytes,
        c2_score_path=checked_c2_path,
        c2_score_bytes=checked_c2_bytes,
        c2_receipt_path=checked_c2_receipt_path,
        c2_receipt_bytes=checked_c2_receipt_bytes,
        c3_metadata=c3_metadata,
        c3_arrays=c3_arrays,
        c3_manifest_sha256=c3_manifest_sha,
        c4_bundle_dir=c4_root,
        c4_directory_state=c4_state,
        c3_bundle_dir=c3_root,
        c3_directory_state=c3_state,
        c2_bundle_dir=c2_root,
        c2_directory_state=c2_state,
    )


def _recheck_parent(parent: _ParentContext) -> None:
    _assert_file_snapshot(parent.c4_score_path, parent.c4_score_bytes, "checked C4 score")
    _assert_file_snapshot(
        parent.c4_receipt_path,
        parent.c4_receipt_bytes,
        "checked C4 receipt",
    )
    _assert_file_snapshot(
        parent.c3_score_path,
        parent.c3_score_bytes,
        "checked C3 score",
    )
    _assert_file_snapshot(
        parent.c3_receipt_path,
        parent.c3_receipt_bytes,
        "checked C3 receipt",
    )
    _assert_file_snapshot(
        parent.c2_score_path,
        parent.c2_score_bytes,
        "checked C2 score",
    )
    _assert_file_snapshot(
        parent.c2_receipt_path,
        parent.c2_receipt_bytes,
        "checked C2 receipt",
    )
    _assert_directory_state(
        parent.c4_bundle_dir,
        parent.c4_directory_state,
        "selected C4 raw bundle",
    )
    _assert_directory_state(
        parent.c3_bundle_dir,
        parent.c3_directory_state,
        "selected C3 raw bundle",
    )
    _assert_directory_state(
        parent.c2_bundle_dir,
        parent.c2_directory_state,
        "selected C2 raw bundle",
    )


def _parameter_values(c3_metadata: dict[str, Any]) -> dict[str, Any]:
    parameters = _mapping(c3_metadata.get("parameters"), "accepted C3 parameters")
    _exact_keys(parameters, {"hex", "values"}, "accepted C3 parameters")
    values = _mapping(parameters.get("values"), "accepted C3 parameter values")
    required = {
        "T_c_K",
        "boltzmann_constant_J_per_K",
        "electron_charge_C",
        "gap_eV",
        "tau_0_s",
        "temperature_K",
    }
    if not required <= set(values):
        raise C5ScoreError("Accepted C3 parameters lack C5 operator inputs.")
    return values


def _operator_inputs(c3_metadata: dict[str, Any]) -> dict[str, object]:
    values = _parameter_values(c3_metadata)
    boltzmann = _finite_scalar(
        values["boltzmann_constant_J_per_K"],
        "boltzmann constant",
        positive=True,
    )
    electron_charge = _finite_scalar(
        values["electron_charge_C"],
        "electron charge",
        positive=True,
    )
    t_c = _finite_scalar(values["T_c_K"], "T_c", positive=True)
    tau_0_parent_s = _finite_scalar(
        values["tau_0_s"],
        "tau_0_s",
        positive=True,
    )
    tau_0_ns = tau_0_parent_s / SECONDS_PER_NS
    native_parameters = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "accepted C3 native grid parameters",
    )
    gap_uev = _finite_scalar(
        native_parameters.get("gap_ueV"),
        "native gap_ueV",
        positive=True,
    )
    gap_parent_ev = _finite_scalar(
        values["gap_eV"],
        "parent gap_eV",
        positive=True,
    )
    k_b_uev_per_k = boltzmann / electron_charge * 1.0e6
    temperature = _finite_scalar(values["temperature_K"], "temperature_K")
    return {
        "T_bath_K": _float_record(temperature),
        "T_c_K": _float_record(t_c),
        "boltzmann_constant_J_per_K": _float_record(boltzmann),
        "electron_charge_C": _float_record(electron_charge),
        "gap_parent_eV": _float_record(gap_parent_ev),
        "gap_ueV": _float_record(gap_uev),
        "kB_T_c_ueV": _float_record(k_b_uev_per_k * t_c),
        "kB_ueV_per_K": _float_record(k_b_uev_per_k),
        "seconds_per_ns": _float_record(SECONDS_PER_NS),
        "tau_0_ns": _float_record(tau_0_ns),
        "tau_0_parent_s": _float_record(tau_0_parent_s),
    }


def _float_value(record: object, label: str) -> float:
    mapped = _mapping(record, label)
    _exact_keys(mapped, {"hex", "value"}, label)
    value = _finite_scalar(mapped.get("value"), f"{label}.value")
    if mapped.get("hex") != value.hex():
        raise C5ScoreError(f"{label} decimal and hexadecimal forms disagree.")
    return value


def _expected_frozen_arrays(
    parent: _ParentContext,
) -> dict[str, np.ndarray]:
    c3 = parent.c3_arrays
    c4 = parent.c4_arrays
    return {
        "parent_E_centers_ueV": np.asarray(c3["native_E_centers_ueV"]),
        "parent_dE_ueV": np.asarray(c3["native_dE_ueV"]),
        "parent_f": np.asarray(c3["projected_f"]),
        "parent_active_mask": np.asarray(c3["native_active_mask"]),
        "parent_cell_weights_ueV": np.asarray(c3["native_cell_weights_full"]),
        "parent_projected_n_phonon": np.asarray(c3["projected_n_phonon"]),
        "parent_legacy_phonon_support_mask": np.asarray(
            c3["legacy_phonon_support_mask"]
        ),
        "parent_qp_scattering_gain_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_scattering__gain_s_inv"]
        ),
        "parent_qp_scattering_loss_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_scattering__loss_s_inv"]
        ),
        "parent_qp_scattering_net_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_scattering__net_s_inv"]
        ),
        "parent_qp_pair_gain_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_pair__gain_s_inv"]
        ),
        "parent_qp_pair_loss_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_pair__loss_s_inv"]
        ),
        "parent_qp_pair_net_s_inv": np.asarray(
            c3[f"{PARENT_OPERATOR_STAGE_ID}__qp_pair__net_s_inv"]
        ),
        "parent_public_qp_photon_gain_s_inv": np.asarray(
            c4["qpsim_gain_s_inv"]
        ),
        "parent_public_qp_photon_loss_s_inv": np.asarray(
            c4["qpsim_loss_s_inv"]
        ),
        "parent_public_qp_photon_net_s_inv": np.asarray(c4["qpsim_net_s_inv"]),
        "parent_qp_residual_s_inv": np.asarray(c4["hybrid_qp_residual_s_inv"]),
        "parent_phonon_residual_s_inv": np.asarray(
            c4["hybrid_phonon_residual_s_inv"]
        ),
    }


def _expected_map_and_occupations(
    n_phonon: np.ndarray,
) -> dict[str, np.ndarray]:
    levels = np.arange(N_QP, dtype=np.int64)
    idx_diff = np.abs(levels[:, None] - levels[None, :])
    # E_i = 160.5+i micro-eV, so E_i+E_j = 321+i+j micro-eV.
    idx_sum = 321 + levels[:, None] + levels[None, :]
    diff_sign = np.sign(levels[:, None] - levels[None, :]).astype(np.int8)
    omega = np.arange(N_OMEGA, dtype=np.float64)
    n_diff = n_phonon[idx_diff]
    n_sum = n_phonon[idx_sum]
    n_p = np.where(diff_sign > 0, 1.0 + n_diff, n_diff)
    np.fill_diagonal(n_p, 0.0)
    return {
        "qpsim_omega_ueV": omega,
        "qpsim_omega_idx_diff": idx_diff,
        "qpsim_omega_idx_sum": idx_sum,
        "qpsim_diff_sign": diff_sign,
        "qpsim_N_p": n_p,
        "qpsim_N_emit": 1.0 + n_sum,
        "qpsim_N_abs": n_sum.copy(),
    }


def _expected_kernels(
    parent: _ParentContext,
    inputs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    e = np.asarray(parent.c3_arrays["native_E_centers_ueV"], dtype=np.float64)
    k_minus = np.asarray(parent.c3_arrays["native_K_minus_full"], dtype=np.float64)
    k_plus = np.asarray(parent.c3_arrays["native_K_plus_full"], dtype=np.float64)
    tau_0_ns = _float_value(inputs["tau_0_ns"], "operator_inputs.tau_0_ns")
    kbt_c = _float_value(inputs["kB_T_c_ueV"], "operator_inputs.kB_T_c_ueV")
    diff = e[:, None] - e[None, :]
    energy_sum = e[:, None] + e[None, :]
    scattering = (1.0 / tau_0_ns) * diff**2 / kbt_c**3 * k_minus
    np.fill_diagonal(scattering, 0.0)
    pair = (
        (1.0 / tau_0_ns)
        * (energy_sum / kbt_c) ** 2
        / kbt_c
        * k_plus
    )
    return scattering, pair


def _kernel_rounding_check(
    candidate: np.ndarray,
    reference: np.ndarray,
    label: str,
) -> dict[str, object]:
    delta = np.abs(np.asarray(candidate) - np.asarray(reference))
    scale = np.maximum(
        np.maximum(np.abs(candidate), np.abs(reference)),
        np.finfo(np.float64).tiny,
    )
    # No reduction occurs here; 32 eps covers the fixed scalar expression
    # order while remaining far below any material kernel change.
    bound = 32.0 * _FLOAT_EPS * scale
    ratios = delta / bound
    maximum = float(np.max(ratios, initial=0.0))
    if maximum > 1.0:
        raise C5ScoreError(f"{label} differs from its independent kernel formula.")
    return {
        "l1_absolute": _float_record(_fixed_sum(delta)),
        "linf_absolute": _float_record(float(np.max(delta, initial=0.0))),
        "maximum_rounding_bound_fraction": _float_record(maximum),
    }


def _clean_channel_reductions(
    f: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
    scattering_kernel: np.ndarray,
    pair_kernel: np.ndarray,
    n_p: np.ndarray,
    n_emit: np.ndarray,
    n_abs: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    """Fixed-order, BLAS-independent channel reductions in ns^-1."""

    one_minus = 1.0 - f
    result: dict[str, dict[str, np.ndarray]] = {}
    for channel in ("scattering", "pair"):
        gain = np.zeros(N_QP, dtype=np.float64)
        loss_rate = np.zeros(N_QP, dtype=np.float64)
        for i in range(N_QP):
            if not bool(active[i]):
                continue
            if channel == "scattering":
                gain_terms = (
                    float(scattering_kernel[j, i])
                    * float(n_p[j, i])
                    * float(weights[j])
                    * float(f[j])
                    for j in range(N_QP)
                )
                loss_terms = (
                    float(scattering_kernel[i, j])
                    * float(n_p[i, j])
                    * float(weights[j])
                    * float(one_minus[j])
                    for j in range(N_QP)
                )
            else:
                gain_terms = (
                    float(pair_kernel[i, j])
                    * float(n_abs[i, j])
                    * float(weights[j])
                    * float(one_minus[j])
                    for j in range(N_QP)
                )
                loss_terms = (
                    float(pair_kernel[i, j])
                    * float(n_emit[i, j])
                    * float(weights[j])
                    * float(f[j])
                    for j in range(N_QP)
                )
            gain[i] = float(one_minus[i]) * math.fsum(gain_terms)
            loss_rate[i] = math.fsum(loss_terms)
        loss = loss_rate * f
        result[channel] = {
            "gain": gain,
            "loss_rate": loss_rate,
            "loss": loss,
            "net": gain - loss,
        }
    return result


def _reduction_rounding_check(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    absolute_bound: np.ndarray | None = None,
) -> dict[str, object]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    absolute = np.abs(left - right)
    if absolute_bound is None:
        scale = np.maximum(
            np.abs(left) + np.abs(right),
            np.finfo(np.float64).tiny,
        )
        bound = _REDUCTION_GAMMA * scale
    else:
        bound = np.maximum(
            np.asarray(absolute_bound, dtype=np.float64),
            np.finfo(np.float64).tiny,
        )
    fraction = absolute / bound
    maximum = float(np.max(fraction, initial=0.0))
    absolute_sum = _fixed_sum(absolute)
    denominator = max(
        _fixed_sum(np.abs(left)) + _fixed_sum(np.abs(right)),
        np.finfo(np.float64).tiny,
    )
    return {
        "l1_absolute": _float_record(absolute_sum),
        "linf_absolute": _float_record(float(np.max(absolute, initial=0.0))),
        "maximum_rounding_bound_fraction": _float_record(maximum),
        "symmetric_relative_l1": _float_record(absolute_sum / denominator),
        "within_rounding_bound": maximum <= 1.0,
    }


def _operator_comparison(
    candidate: np.ndarray,
    parent: np.ndarray,
) -> dict[str, object]:
    absolute = np.abs(np.asarray(candidate) - np.asarray(parent))
    absolute_sum = _fixed_sum(absolute)
    denominator = max(
        _fixed_sum(np.abs(candidate)) + _fixed_sum(np.abs(parent)),
        np.finfo(np.float64).tiny,
    )
    return {
        "l1_absolute_s_inv": _float_record(absolute_sum),
        "linf_absolute_s_inv": _float_record(
            float(np.max(absolute, initial=0.0))
        ),
        "symmetric_relative_l1": _float_record(absolute_sum / denominator),
    }


def _weighted_conservation(
    gain: np.ndarray,
    loss: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    net = np.asarray(gain) - np.asarray(loss)
    weighted = float(math.fsum((weights * net).tolist()))
    turnover = float(
        math.fsum((weights * (np.abs(gain) + np.abs(loss))).tolist())
    )
    relative = abs(weighted) / max(turnover, np.finfo(np.float64).tiny)
    return {
        "symmetric_turnover_relative": _float_record(relative),
        "weighted_net_s_inv_ueV": _float_record(weighted),
        "weighted_turnover_s_inv_ueV": _float_record(turnover),
    }


def _expected_parent_bindings(parent: _ParentContext) -> dict[str, object]:
    inherited = _mapping(
        parent.c4_score.get("parent_bindings"),
        "accepted C4 parent bindings",
    )
    return {
        "c2_raw_manifest_sha256": inherited.get("c2_raw_manifest_sha256"),
        "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
        "c3_raw_manifest_sha256": parent.c3_manifest_sha256,
        "c3_raw_schema": C3_RAW_SCHEMA,
        "c4_raw_manifest_sha256": parent.c4_manifest_sha256,
        "c4_raw_schema": C4_RAW_SCHEMA,
        "c4_receipt_path": parent.c4_receipt_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c4_receipt_schema": C4_RECEIPT_SCHEMA,
        "c4_receipt_sha256": hashlib.sha256(parent.c4_receipt_bytes).hexdigest(),
        "c4_score_path": parent.c4_score_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c4_score_schema": C4_SCORE_SCHEMA,
        "c4_score_sha256": hashlib.sha256(parent.c4_score_bytes).hexdigest(),
        "c4_stage_id": PARENT_STAGE_ID,
    }


def _c3_frozen_names() -> tuple[str, ...]:
    return (
        "projected_f",
        "projected_n_phonon",
        "native_E_centers_ueV",
        "native_dE_ueV",
        "native_active_mask",
        "native_cell_density_full",
        "native_cell_weights_full",
        "native_K_minus_full",
        "native_K_plus_full",
        "native_omega_ueV",
        "legacy_phonon_support_mask",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__qp_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair")
            for field in ("gain", "loss", "net")
        ),
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair", "escape")
            for field in ("gain", "loss", "net")
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )


def _expected_raw_contracts(
    parent: _ParentContext,
) -> dict[str, dict[str, object]]:
    phonon_names = [
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair", "escape")
            for field in ("gain", "loss", "net")
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    ]
    raw_numbers = _EXPECTED_RAW_BOOKKEEPING_METRICS
    return {
        "bookkeeping_contract": {
            "pair_buckets": (
                "parent and public pair gain/loss are physical generation and "
                "physical loss buckets"
            ),
            "pair_weighted_number": {
                "net_s_inv_ueV": _float_record(
                    raw_numbers["pair_weighted_net_s_inv_ueV"]
                ),
                "statement": (
                    "pair generation/recombination changes QP number by "
                    "construction; no zero-drift conservation gate is applied"
                ),
                "turnover_s_inv_ueV": _float_record(
                    raw_numbers["pair_weighted_turnover_s_inv_ueV"]
                ),
            },
            "scattering_cross_term": (
                "author source-order scattering gain and loss each contain the "
                "same Pauli cross-term; it cancels exactly from the source-order net"
            ),
            "scattering_pauli_cross_term_formula": (
                "f * ((K_s0 * n_ph[omega_idx_diff]).T @ "
                "(cell_weights * f))"
            ),
            "scattering_rebucketed_controls": (
                "parent rebucketed gain/loss = parent source-order bucket "
                "- scattering_pauli_cross_term"
            ),
            "scattering_weighted_number_conservation": {
                "absolute_drift_s_inv_ueV": _float_record(
                    raw_numbers["scattering_absolute_drift_s_inv_ueV"]
                ),
                "limit_relative": _float_record(
                    _SCATTERING_CONSERVATION_LIMIT
                ),
                "relative_drift": _float_record(
                    raw_numbers["scattering_relative_drift"]
                ),
                "turnover_s_inv_ueV": _float_record(
                    raw_numbers["scattering_turnover_s_inv_ueV"]
                ),
            },
            "warning": (
                "raw public-minus-parent scattering gain/loss are bookkeeping "
                "differentials; rebucketed gain/loss and the physical net are "
                "the like-for-like comparisons"
            ),
        },
        "comparison_contract": {
            "candidate": (
                "public qpsim QP-equation base scattering/recombination kernels "
                "and phonon_collision_rates evaluated separately"
            ),
            "loss_comparison": (
                "physical loss = returned loss_rate * frozen f; loss_rate is "
                "also retained separately and is never compared to parent loss"
            ),
            "parent": (
                "accepted C3c author-form QP scattering/pair channels carried "
                "through the completed C4 public-photon parent"
            ),
            "parent_photon": (
                "accepted C4 public photon gain/loss/net, copied bit-exact and "
                "not re-evaluated"
            ),
            "public_arithmetic": (
                "matrix products may use BLAS; an independent verifier must "
                "apply declared floating-point tolerances rather than demand "
                "bit equality to a source-order transcription"
            ),
        },
        "component_locality": {
            "c5p": "replace only the QP pair net in the C4 parent residual",
            "c5s": "replace only the QP scattering net in the C4 parent residual",
            "c5sp": (
                "replace both QP scattering and QP pair nets; this is the "
                "formal C5 hybrid QP residual"
            ),
            "changed_arrays": (
                "QP scattering/pair kernels, gain, physical loss, net, and "
                "the derived C5s/C5p/C5sp QP residuals"
            ),
            "inherited_arrays": (
                "C4 public photon channel plus all C3c phonon scattering, "
                "phonon pair, phonon escape, and phonon residual arrays"
            ),
            "phonon_residual_bit_exact": True,
            "qp_residual_updates": {
                "c5p": "parent_qp_residual + qp_pair_delta_net",
                "c5s": "parent_qp_residual + qp_scattering_delta_net",
                "c5sp": (
                    "parent_qp_residual + qp_scattering_delta_net "
                    "+ qp_pair_delta_net"
                ),
            },
        },
        "coordinate_contract": {
            "active_child_indices": "[20, 1640)",
            "frequency_map": (
                "public build_phonon_frequency_map on the accepted C3 "
                "1640-cell center grid"
            ),
            "guard_child_indices": "[0, 20), canonical positive zero",
            "legacy_phonon_support": (
                "omega indices [1, 1620); every other projected n_ph entry "
                "is canonical zero"
            ),
            "native_cell_count": N_QP,
            "native_omega_count": N_OMEGA,
            "phonon_projection": (
                "public phonon_occupation_matrices_from_state on the frozen "
                "C3 projected_n_phonon"
            ),
        },
        "frozen_inputs": {
            "c3_descriptors": {
                name: _array_descriptor(parent.c3_arrays[name])
                for name in _c3_frozen_names()
            },
            "c4_descriptors": {
                name: _array_descriptor(value)
                for name, value in sorted(parent.c4_arrays.items())
            },
            "c4_mutation_check_after_operator": True,
            "c3_mutation_check_after_operator": True,
            "phonon_equation_descriptor_names": phonon_names,
            "policy": (
                "accepted C4 f/public photon/residual and accepted C3 grid, "
                "masks, K_minus/K_plus, projected n_ph, parameters, and every "
                "phonon-equation channel are immutable"
            ),
        },
        "limitations": {
            "scope": "one authenticated C4 frozen point only",
            "statement": (
                "No C5 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, observable change, "
                "coupled QP-phonon conservation claim, or paper-parity claim "
                "is made. The phonon balance remains the inherited author "
                "equation and is not evaluated by qpsim in C5."
            ),
        },
        "source_binding": {
            **_EXPECTED_SOURCE_BINDING,
        },
        "units": {
            "comparison_arrays": "per second",
            "kernel_arrays": "per nanosecond per microelectronvolt",
            "public_native_arrays": "per nanosecond",
            "public_return_contract": (
                "gain includes target Pauli factor; loss_rate multiplies f "
                "to form physical loss"
            ),
        },
    }


def _check_raw_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    parent: _ParentContext,
    operator_inputs: dict[str, object],
) -> None:
    if metadata.get("schema") != RAW_SCHEMA:
        raise C5ScoreError("C5 raw metadata schema is invalid.")
    stage = _mapping(metadata.get("stage"), "C5 raw stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
    }:
        raise C5ScoreError("C5 raw stage metadata is invalid.")
    if not _json_value_bit_exact(metadata.get("operator_inputs"), operator_inputs):
        raise C5ScoreError("C5 raw operator inputs are stale or incomplete.")
    if not _json_value_bit_exact(
        metadata.get("parent_bindings"),
        _expected_parent_bindings(parent),
    ):
        raise C5ScoreError("C5 raw parent bindings are stale or incomplete.")
    descriptors = _mapping(metadata.get("array_descriptors"), "C5 descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C5ScoreError("C5 raw descriptor metadata is stale.")
    sources = _mapping(metadata.get("sources"), "C5 raw sources")
    source_binding = _mapping(metadata.get("source_binding"), "C5 source binding")
    if not _json_value_bit_exact(sources, _RAW_SOURCE_HASHES_AT_IMPORT):
        raise C5ScoreError("C5 raw source closure is incomplete, extra, or stale.")
    for relative, digest in sources.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
        ):
            raise C5ScoreError(f"Unsafe C5 source path {relative!r}.")
        _sha256(digest, f"C5 source {relative}")
        source_path = REPOSITORY_ROOT / relative
        if hashlib.sha256(canonical_source_bytes(source_path)).hexdigest() != digest:
            raise C5ScoreError(f"C5 raw source binding is stale: {relative}.")
    contracts = _expected_raw_contracts(parent)
    for key in (
        "bookkeeping_contract",
        "comparison_contract",
        "component_locality",
        "coordinate_contract",
        "frozen_inputs",
        "limitations",
        "units",
    ):
        if not _json_value_bit_exact(metadata.get(key), contracts[key]):
            raise C5ScoreError(f"C5 raw {key} is invalid or stale.")
    if not _json_value_bit_exact(source_binding, contracts["source_binding"]):
        raise C5ScoreError("C5 raw source_binding is invalid.")
    _validate_runtime_record(metadata.get("runtime"), "C5 raw runtime")


def _validate_exact_derived_arrays(
    arrays: dict[str, np.ndarray],
) -> None:
    f = arrays["parent_f"]
    for channel in ("scattering", "pair"):
        gain_ns = arrays[f"qpsim_qp_{channel}_gain_ns_inv"]
        loss_rate_ns = arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"]
        loss_ns = arrays[f"qpsim_qp_{channel}_loss_ns_inv"]
        net_ns = arrays[f"qpsim_qp_{channel}_net_ns_inv"]
        expected = {
            f"qpsim_qp_{channel}_gain_s_inv": gain_ns / SECONDS_PER_NS,
            f"qpsim_qp_{channel}_loss_s_inv": loss_ns / SECONDS_PER_NS,
            f"qpsim_qp_{channel}_loss_rate_s_inv": (
                loss_rate_ns / SECONDS_PER_NS
            ),
            f"qpsim_qp_{channel}_net_s_inv": net_ns / SECONDS_PER_NS,
            f"qpsim_qp_{channel}_loss_ns_inv": loss_rate_ns * f,
            f"qpsim_qp_{channel}_net_ns_inv": gain_ns - loss_ns,
        }
        for name, value in expected.items():
            if not _array_bit_exact(_positive_zero_copy(value), arrays[name]):
                raise C5ScoreError(f"C5 raw derived array {name!r} is inconsistent.")
        for field in ("gain", "loss", "net"):
            delta_name = f"qp_{channel}_delta_{field}_s_inv"
            expected_delta = (
                arrays[f"qpsim_qp_{channel}_{field}_s_inv"]
                - arrays[f"parent_qp_{channel}_{field}_s_inv"]
            )
            if not _array_bit_exact(
                _positive_zero_copy(expected_delta),
                arrays[delta_name],
            ):
                raise C5ScoreError(f"C5 raw delta {delta_name!r} is inconsistent.")

    cross = arrays["scattering_pauli_cross_term_s_inv"]
    expected_rebucketed = {
        "parent_qp_scattering_rebucketed_gain_s_inv": (
            arrays["parent_qp_scattering_gain_s_inv"] - cross
        ),
        "parent_qp_scattering_rebucketed_loss_s_inv": (
            arrays["parent_qp_scattering_loss_s_inv"] - cross
        ),
    }
    for name, value in expected_rebucketed.items():
        if not _array_bit_exact(_positive_zero_copy(value), arrays[name]):
            raise C5ScoreError(f"C5 raw rebucketed array {name!r} is inconsistent.")
    for field in ("gain", "loss"):
        expected_delta = (
            arrays[f"qpsim_qp_scattering_{field}_s_inv"]
            - arrays[f"parent_qp_scattering_rebucketed_{field}_s_inv"]
        )
        name = f"qp_scattering_rebucketed_delta_{field}_s_inv"
        if not _array_bit_exact(_positive_zero_copy(expected_delta), arrays[name]):
            raise C5ScoreError(f"C5 raw rebucketed delta {name!r} is inconsistent.")

    scatter_delta = arrays["qp_scattering_delta_net_s_inv"]
    pair_delta = arrays["qp_pair_delta_net_s_inv"]
    parent_residual = arrays["parent_qp_residual_s_inv"]
    expected_residuals = {
        "c5s_qp_residual_s_inv": parent_residual + scatter_delta,
        "c5p_qp_residual_s_inv": parent_residual + pair_delta,
        "c5sp_qp_residual_s_inv": parent_residual + scatter_delta + pair_delta,
        "c5sp_phonon_residual_s_inv": arrays["parent_phonon_residual_s_inv"],
    }
    for name, value in expected_residuals.items():
        if not _array_bit_exact(_positive_zero_copy(value), arrays[name]):
            raise C5ScoreError(f"C5 raw hybrid array {name!r} is inconsistent.")


def _clean_pauli_cross_term(
    f: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
    scattering_kernel: np.ndarray,
    n_diff: np.ndarray,
) -> np.ndarray:
    result = np.zeros(N_QP, dtype=np.float64)
    for i in range(N_QP):
        if not bool(active[i]):
            continue
        result[i] = float(f[i]) * math.fsum(
            float(scattering_kernel[i, j])
            * float(n_diff[i, j])
            * float(weights[j])
            * float(f[j])
            for j in range(N_QP)
        )
    return result


def _map_identity_record(
    arrays: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
) -> dict[str, object]:
    fields: dict[str, object] = {}
    for name in (
        "qpsim_omega_ueV",
        "qpsim_omega_idx_diff",
        "qpsim_omega_idx_sum",
        "qpsim_diff_sign",
        "qpsim_N_p",
        "qpsim_N_emit",
        "qpsim_N_abs",
    ):
        bit_exact = _array_bit_exact(arrays[name], expected[name])
        if not bit_exact:
            raise C5ScoreError(f"C5 frozen map/occupation {name!r} changed.")
        fields[name] = {
            "bit_exact": True,
            "descriptor": _array_descriptor(arrays[name]),
        }
    fields["contract"] = (
        "The producer may call qpsim's public frequency-map helpers, but C5 "
        "accepts them only after exact identity to the already-frozen C3 "
        "center labels: diff=|i-j| and sum=321+i+j on omega=0..3599."
    )
    return fields


def _frozen_phonon_descriptors(parent: _ParentContext) -> dict[str, object]:
    names = []
    for channel in ("scattering", "pair", "escape"):
        for field in ("gain", "loss", "net"):
            names.append(
                f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            )
    names.append(f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv")
    return {
        name: _array_descriptor(np.asarray(parent.c3_arrays[name]))
        for name in sorted(names)
    }


def build_c5_score(
    c5_bundle_dir: Path,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Independently replay and score one formal frozen-state C5 bundle."""

    _assert_source_snapshots()
    c5_root, c5_directory_state = _directory_state(
        c5_bundle_dir,
        "selected C5 raw bundle",
    )
    parent = _accept_parent(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    raw_metadata, arrays, raw_manifest_sha = load_c5_raw_bundle(c5_root)
    operator_inputs = _operator_inputs(parent.c3_metadata)
    _check_raw_metadata(
        raw_metadata,
        arrays,
        parent=parent,
        operator_inputs=operator_inputs,
    )

    frozen = _expected_frozen_arrays(parent)
    for name, expected in frozen.items():
        if not _array_bit_exact(_positive_zero_copy(expected), arrays[name]):
            raise C5ScoreError(f"C5 frozen parent array {name!r} changed.")
    active = arrays["parent_active_mask"]
    if (
        int(np.count_nonzero(active)) != N_ACTIVE
        or np.any(active[:ACTIVE_START])
        or not np.all(active[ACTIVE_START:])
    ):
        raise C5ScoreError("C5 active support is not the accepted C3 support.")
    if np.any(arrays["parent_f"][:ACTIVE_START] != 0.0):
        raise C5ScoreError("C5 guard-cell occupations are not canonical positive zero.")
    expected_support = np.zeros(N_OMEGA, dtype=bool)
    expected_support[1 : AUTHOR_OMEGA_STOP + 1] = True
    if not _array_bit_exact(
        arrays["parent_legacy_phonon_support_mask"],
        expected_support,
    ):
        raise C5ScoreError("C5 inherited author phonon support changed.")

    map_expected = _expected_map_and_occupations(
        arrays["parent_projected_n_phonon"]
    )
    map_record = _map_identity_record(arrays, map_expected)
    scattering_kernel, pair_kernel = _expected_kernels(parent, operator_inputs)
    kernel_records = {
        "scattering": _kernel_rounding_check(
            arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"],
            scattering_kernel,
            "QP scattering kernel",
        ),
        "pair": _kernel_rounding_check(
            arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"],
            pair_kernel,
            "QP pair kernel",
        ),
    }
    _validate_exact_derived_arrays(arrays)

    clean = _clean_channel_reductions(
        arrays["parent_f"],
        arrays["parent_cell_weights_ueV"],
        active,
        scattering_kernel,
        pair_kernel,
        map_expected["qpsim_N_p"],
        map_expected["qpsim_N_emit"],
        map_expected["qpsim_N_abs"],
    )
    channel_comparison: dict[str, Any] = {}
    reduction_checks: list[bool] = []
    for channel in ("scattering", "pair"):
        reduction: dict[str, object] = {}
        gain_bound = _REDUCTION_GAMMA * (
            np.abs(arrays[f"qpsim_qp_{channel}_gain_ns_inv"])
            + np.abs(clean[channel]["gain"])
        )
        loss_rate_bound = _REDUCTION_GAMMA * (
            np.abs(arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"])
            + np.abs(clean[channel]["loss_rate"])
        )
        loss_bound = loss_rate_bound * np.abs(arrays["parent_f"]) + (
            4.0
            * _FLOAT_EPS
            * (
                np.abs(arrays[f"qpsim_qp_{channel}_loss_ns_inv"])
                + np.abs(clean[channel]["loss"])
            )
        )
        net_bound = gain_bound + loss_bound + (
            4.0
            * _FLOAT_EPS
            * (
                np.abs(arrays[f"qpsim_qp_{channel}_net_ns_inv"])
                + np.abs(clean[channel]["net"])
            )
        )
        bounds = {
            "gain": gain_bound,
            "loss_rate": loss_rate_bound,
            "loss": loss_bound,
            "net": net_bound,
        }
        for field in ("gain", "loss_rate", "loss", "net"):
            check = _reduction_rounding_check(
                arrays[f"qpsim_qp_{channel}_{field}_ns_inv"],
                clean[channel][field],
                absolute_bound=bounds[field],
            )
            reduction[field] = check
            reduction_checks.append(bool(check["within_rounding_bound"]))
        parent_fields = {
            field: _operator_comparison(
                arrays[f"qpsim_qp_{channel}_{field}_s_inv"],
                arrays[f"parent_qp_{channel}_{field}_s_inv"],
            )
            for field in ("gain", "loss", "net")
        }
        channel_comparison[channel] = {
            "kernel_formula": kernel_records[channel],
            "parent_operator": parent_fields,
            "public_vs_clean_reduction": reduction,
        }

    n_diff = arrays["parent_projected_n_phonon"][
        map_expected["qpsim_omega_idx_diff"]
    ]
    clean_cross_ns = _clean_pauli_cross_term(
        arrays["parent_f"],
        arrays["parent_cell_weights_ueV"],
        active,
        scattering_kernel,
        n_diff,
    )
    raw_cross_ns = arrays["scattering_pauli_cross_term_s_inv"] * SECONDS_PER_NS
    cross_check = _reduction_rounding_check(raw_cross_ns, clean_cross_ns)
    reduction_checks.append(bool(cross_check["within_rounding_bound"]))
    rebucketed = {
        field: _operator_comparison(
            arrays[f"qpsim_qp_scattering_{field}_s_inv"],
            arrays[f"parent_qp_scattering_rebucketed_{field}_s_inv"],
        )
        for field in ("gain", "loss")
    }
    channel_comparison["scattering"]["pauli_rebucketing"] = {
        "cross_term_public_vs_clean": cross_check,
        "public_vs_parent_rebucketed": rebucketed,
        "semantic_statement": (
            "The author puts the same n*f_i*f_j Pauli cross-term in both "
            "scattering gain and loss. Public qpsim removes it from both; "
            "therefore raw gain/loss differ while the physical net is unchanged."
        ),
    }

    public_combined = (
        arrays["qpsim_qp_scattering_net_s_inv"]
        + arrays["qpsim_qp_pair_net_s_inv"]
    )
    parent_combined = (
        arrays["parent_qp_scattering_net_s_inv"]
        + arrays["parent_qp_pair_net_s_inv"]
    )
    combined_comparison = _operator_comparison(public_combined, parent_combined)
    channel_comparison["combined_net"] = combined_comparison

    scattering_conservation = _weighted_conservation(
        arrays["qpsim_qp_scattering_gain_s_inv"],
        arrays["qpsim_qp_scattering_loss_s_inv"],
        arrays["parent_cell_weights_ueV"],
    )
    # Pair recombination changes total QP number.  Record its moment, but do
    # not impose the scattering conservation gate on it.
    pair_number_moment = _weighted_conservation(
        arrays["qpsim_qp_pair_gain_s_inv"],
        arrays["qpsim_qp_pair_loss_s_inv"],
        arrays["parent_cell_weights_ueV"],
    )
    scattering_net_relative = _float_value(
        _mapping(
            channel_comparison["scattering"]["parent_operator"],
            "scattering parent operator",
        )["net"]["symmetric_relative_l1"],
        "scattering net parity",
    )
    pair_net_relative = _float_value(
        _mapping(
            channel_comparison["pair"]["parent_operator"],
            "pair parent operator",
        )["net"]["symmetric_relative_l1"],
        "pair net parity",
    )
    pair_gain_relative = _float_value(
        _mapping(
            channel_comparison["pair"]["parent_operator"],
            "pair parent operator",
        )["gain"]["symmetric_relative_l1"],
        "pair gain parity",
    )
    pair_loss_relative = _float_value(
        _mapping(
            channel_comparison["pair"]["parent_operator"],
            "pair parent operator",
        )["loss"]["symmetric_relative_l1"],
        "pair loss parity",
    )
    scattering_rebucketed_gain_relative = _float_value(
        rebucketed["gain"]["symmetric_relative_l1"],
        "scattering rebucketed gain parity",
    )
    scattering_rebucketed_loss_relative = _float_value(
        rebucketed["loss"]["symmetric_relative_l1"],
        "scattering rebucketed loss parity",
    )
    combined_net_relative = _float_value(
        combined_comparison["symmetric_relative_l1"],
        "combined net parity",
    )
    scattering_conservation_relative = _float_value(
        scattering_conservation["symmetric_turnover_relative"],
        "scattering conservation",
    )
    all_checks = {
        "all_public_reductions_within_predeclared_rounding_bound": all(
            reduction_checks
        ),
        "combined_physical_net_matches_parent": combined_net_relative
        <= _NET_PARITY_LIMIT,
        "frozen_phonon_residual_bit_exact": _array_bit_exact(
            arrays["c5sp_phonon_residual_s_inv"],
            arrays["parent_phonon_residual_s_inv"],
        ),
        "pair_physical_gain_matches_parent": pair_gain_relative
        <= _BUCKET_PARITY_LIMIT,
        "pair_physical_loss_matches_parent": pair_loss_relative
        <= _BUCKET_PARITY_LIMIT,
        "pair_physical_net_matches_parent": pair_net_relative <= _NET_PARITY_LIMIT,
        "scattering_rebucketed_gain_matches_parent": (
            scattering_rebucketed_gain_relative <= _BUCKET_PARITY_LIMIT
        ),
        "scattering_rebucketed_loss_matches_parent": (
            scattering_rebucketed_loss_relative <= _BUCKET_PARITY_LIMIT
        ),
        "scattering_physical_net_matches_parent": scattering_net_relative
        <= _NET_PARITY_LIMIT,
        "scattering_weighted_number_conserved": scattering_conservation_relative
        <= _SCATTERING_CONSERVATION_LIMIT,
    }
    if not all(all_checks.values()):
        failed = sorted(name for name, passed in all_checks.items() if not passed)
        raise C5ScoreError(f"C5 acceptance checks failed: {failed}.")

    source_binding = _mapping(raw_metadata["source_binding"], "C5 source binding")
    score: dict[str, Any] = {
        "acceptance": {
            "all_passed": True,
            "checks": all_checks,
            "limits": {
                "combined_and_per_channel_net_symmetric_relative_l1": _float_record(
                    _NET_PARITY_LIMIT
                ),
                "like_for_like_bucket_symmetric_relative_l1": _float_record(
                    _BUCKET_PARITY_LIMIT
                ),
                "scattering_weighted_number_relative": _float_record(
                    _SCATTERING_CONSERVATION_LIMIT
                ),
            },
        },
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "bookkeeping": {
            "contract": dict(
                _mapping(raw_metadata["bookkeeping_contract"], "bookkeeping contract")
            ),
            "scattering_pauli_rebucketing": dict(
                _mapping(
                    channel_comparison["scattering"]["pauli_rebucketing"],
                    "scattering rebucketing",
                )
            ),
        },
        "channel_comparison": channel_comparison,
        "component_locality": {
            "c5p_qp_residual": _array_descriptor(
                arrays["c5p_qp_residual_s_inv"]
            ),
            "c5s_qp_residual": _array_descriptor(
                arrays["c5s_qp_residual_s_inv"]
            ),
            "c5sp_qp_residual": _array_descriptor(
                arrays["c5sp_qp_residual_s_inv"]
            ),
            "frozen_c4_photon_descriptors": {
                field: _array_descriptor(
                    arrays[f"parent_public_qp_photon_{field}_s_inv"]
                )
                for field in ("gain", "loss", "net")
            },
            "frozen_c3_phonon_descriptors": _frozen_phonon_descriptors(parent),
            "frozen_phonon_residual": _array_descriptor(
                arrays["c5sp_phonon_residual_s_inv"]
            ),
            "raw_contract": dict(
                _mapping(raw_metadata["component_locality"], "component locality")
            ),
        },
        "conservation": {
            "pair_number_change_diagnostic_not_a_conservation_gate": pair_number_moment,
            "scattering_weighted_number": scattering_conservation,
        },
        "contracts": {
            "comparison_contract": dict(
                _mapping(raw_metadata["comparison_contract"], "comparison contract")
            ),
            "coordinate_contract": dict(
                _mapping(raw_metadata["coordinate_contract"], "coordinate contract")
            ),
            "frozen_inputs": dict(
                _mapping(raw_metadata["frozen_inputs"], "frozen inputs")
            ),
        },
        "limitations": dict(_mapping(raw_metadata["limitations"], "limitations")),
        "map_identity": map_record,
        "operator_inputs": operator_inputs,
        "parent_bindings": _expected_parent_bindings(parent),
        "raw_bundle": {
            "manifest_sha256": raw_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "rounding_contract": {
            "comparison": (
                "Retained public arrays are manifest-byte-bound. Independent "
                "science uses fixed-order math.fsum and an elementwise gamma "
                "bound; cross-platform BLAS last bits are not required to match."
            ),
            "float64_epsilon": _float_record(_FLOAT_EPS),
            "gamma": _float_record(_REDUCTION_GAMMA),
            "operation_budget_per_dot": _REDUCTION_OPERATION_BUDGET,
        },
        "runtime": {
            "producer_public_array_generation": dict(
                _mapping(raw_metadata["runtime"], "producer runtime")
            ),
            "verifier_contract": dict(_VERIFIER_RUNTIME_CONTRACT),
        },
        "schema": SCHEMA,
        "source_binding": dict(source_binding),
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
        "units": dict(_mapping(raw_metadata["units"], "units")),
    }
    _recheck_parent(parent)
    _assert_directory_state(
        c5_root,
        c5_directory_state,
        "selected C5 raw bundle",
    )
    _assert_source_snapshots()
    return score


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Return canonical checked-score JSON after strict structure validation."""

    _validate_score_structure(score)
    return _canonical_json_bytes(score)


def _evidence_digest(score: dict[str, Any]) -> str:
    # Bind every score field except ``sources``.  That field contains this
    # verifier's own digest and would make a literal in-source pin circular;
    # it is instead checked exactly against the import-time source closure by
    # ``_validate_score_structure``.  The complete checked-score bytes remain
    # independently bound by the receipt.
    keys = tuple(sorted(_SCORE_KEYS - {"sources"}))
    evidence = {key: score[key] for key in keys}
    return hashlib.sha256(_canonical_json_bytes(evidence)).hexdigest()


def _validate_canonical_pins(score: dict[str, Any]) -> None:
    raw = _mapping(score.get("raw_bundle"), "C5 pinned raw bundle")
    if raw.get("manifest_sha256") != _EXPECTED_RAW_MANIFEST_SHA256:
        raise C5ScoreError("C5 score does not bind the accepted canonical raw manifest.")
    channels = _mapping(score.get("channel_comparison"), "C5 pinned channels")
    scattering = _mapping(channels.get("scattering"), "C5 pinned scattering")
    pair = _mapping(channels.get("pair"), "C5 pinned pair")
    scattering_parent = _mapping(
        scattering.get("parent_operator"),
        "C5 pinned scattering parent",
    )
    pair_parent = _mapping(pair.get("parent_operator"), "C5 pinned pair parent")
    combined = _mapping(channels.get("combined_net"), "C5 pinned combined net")
    conservation = _mapping(score.get("conservation"), "C5 pinned conservation")
    scattering_conservation = _mapping(
        conservation.get("scattering_weighted_number"),
        "C5 pinned scattering conservation",
    )
    pair_diagnostic = _mapping(
        conservation.get("pair_number_change_diagnostic_not_a_conservation_gate"),
        "C5 pinned pair diagnostic",
    )
    actual_metrics = {
        "combined_net_symmetric_relative_l1": _float_value(
            combined.get("symmetric_relative_l1"),
            "pinned combined net",
        ),
        "pair_net_symmetric_relative_l1": _float_value(
            _mapping(pair_parent.get("net"), "pinned pair net").get(
                "symmetric_relative_l1"
            ),
            "pinned pair net relative",
        ),
        "pair_weighted_net_s_inv_ueV": _float_value(
            pair_diagnostic.get("weighted_net_s_inv_ueV"),
            "pinned pair weighted net",
        ),
        "scattering_net_symmetric_relative_l1": _float_value(
            _mapping(scattering_parent.get("net"), "pinned scattering net").get(
                "symmetric_relative_l1"
            ),
            "pinned scattering net relative",
        ),
        "scattering_weighted_number_relative": _float_value(
            scattering_conservation.get("symmetric_turnover_relative"),
            "pinned scattering conservation relative",
        ),
    }
    if not _json_value_bit_exact(actual_metrics, _EXPECTED_CANONICAL_METRICS):
        raise C5ScoreError("C5 canonical numerical metric pins do not match.")
    locality = _mapping(score.get("component_locality"), "C5 pinned locality")
    for name, expected_sha in _EXPECTED_RESIDUAL_NPY_SHA256.items():
        descriptor = _mapping(locality.get(name), f"C5 pinned locality.{name}")
        if descriptor.get("npy_sha256") != expected_sha:
            raise C5ScoreError(f"C5 canonical residual pin {name!r} does not match.")
    if _evidence_digest(score) != _EXPECTED_EVIDENCE_DIGEST:
        raise C5ScoreError("C5 canonical numeric/semantic evidence digest does not match.")


def _validate_descriptor(
    value: object,
    label: str,
    *,
    expected_dtype: str | None = None,
    expected_shape: tuple[int, ...] | None = None,
) -> None:
    descriptor = _mapping(value, label)
    _exact_keys(descriptor, {"dtype", "npy_sha256", "shape"}, label)
    dtype = descriptor.get("dtype")
    shape = descriptor.get("shape")
    if not isinstance(dtype, str):
        raise C5ScoreError(f"{label}.dtype must be a string.")
    _sha256(descriptor.get("npy_sha256"), f"{label}.npy_sha256")
    if (
        not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or any(item < 0 for item in shape)
    ):
        raise C5ScoreError(f"{label}.shape is invalid.")
    if expected_dtype is not None and dtype != expected_dtype:
        raise C5ScoreError(f"{label}.dtype is invalid.")
    if expected_shape is not None and shape != list(expected_shape):
        raise C5ScoreError(f"{label}.shape is invalid.")


def _validate_float_record(value: object, label: str) -> float:
    return _float_value(value, label)


def _validate_metric_record(value: object, label: str) -> None:
    metric = _mapping(value, label)
    required = {
        "l1_absolute",
        "linf_absolute",
        "maximum_rounding_bound_fraction",
        "symmetric_relative_l1",
        "within_rounding_bound",
    }
    _exact_keys(metric, required, label)
    for field in required - {"within_rounding_bound"}:
        _validate_float_record(metric[field], f"{label}.{field}")
    if not isinstance(metric["within_rounding_bound"], bool):
        raise C5ScoreError(f"{label}.within_rounding_bound must be Boolean.")


def _validate_runtime_record(value: object, label: str) -> None:
    runtime = _mapping(value, label)
    _exact_keys(
        runtime,
        {
            "byteorder",
            "implementation",
            "machine",
            "numpy_blas",
            "numpy_version",
            "platform",
            "python_version",
            "thread_environment",
        },
        label,
    )
    for field in (
        "byteorder",
        "implementation",
        "machine",
        "numpy_version",
        "platform",
        "python_version",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise C5ScoreError(f"{label}.{field} must be a nonempty string.")
    if runtime["byteorder"] not in {"little", "big"}:
        raise C5ScoreError(f"{label}.byteorder is invalid.")
    threads = _mapping(runtime["thread_environment"], f"{label}.thread_environment")
    if threads != _THREAD_ENVIRONMENT:
        raise C5ScoreError(f"{label}.thread_environment is invalid.")
    blas = _mapping(runtime["numpy_blas"], f"{label}.numpy_blas")
    _exact_keys(
        blas,
        {"found", "name", "openblas_configuration", "version"},
        f"{label}.numpy_blas",
    )
    if blas["found"] is not None and not isinstance(blas["found"], bool):
        raise C5ScoreError(f"{label}.numpy_blas.found must be Boolean or null.")
    for field in ("name", "openblas_configuration", "version"):
        if blas[field] is not None and not isinstance(blas[field], str):
            raise C5ScoreError(
                f"{label}.numpy_blas.{field} must be a string or null."
            )


def _validate_score_structure(score: dict[str, Any]) -> None:
    _exact_keys(score, _SCORE_KEYS, "C5 score")
    if score.get("schema") != SCHEMA:
        raise C5ScoreError("C5 score schema is unsupported.")
    stage = _mapping(score.get("stage"), "C5 score stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C5ScoreError("C5 score stage is invalid.")
    if not _json_value_bit_exact(score.get("sources"), _SOURCE_HASHES_AT_IMPORT):
        raise C5ScoreError("C5 score source closure is invalid.")
    runtime = _mapping(score.get("runtime"), "C5 score runtime")
    _exact_keys(
        runtime,
        {"producer_public_array_generation", "verifier_contract"},
        "C5 score runtime",
    )
    _validate_runtime_record(
        runtime.get("producer_public_array_generation"),
        "C5 score producer runtime",
    )
    if not _json_value_bit_exact(
        runtime.get("verifier_contract"),
        _VERIFIER_RUNTIME_CONTRACT,
    ):
        raise C5ScoreError("C5 score verifier runtime contract is invalid.")
    raw = _mapping(score.get("raw_bundle"), "C5 score raw bundle")
    _exact_keys(raw, {"manifest_sha256", "schema"}, "C5 score raw bundle")
    if raw.get("schema") != RAW_SCHEMA:
        raise C5ScoreError("C5 score raw schema is invalid.")
    _sha256(raw.get("manifest_sha256"), "C5 score raw manifest SHA-256")
    descriptors = _mapping(score.get("array_descriptors"), "C5 score descriptors")
    if set(descriptors) != _EXPECTED_ARRAY_NAMES:
        raise C5ScoreError("C5 score descriptor closure is invalid.")
    for name, (dtype, shape) in _ARRAY_SPECS.items():
        _validate_descriptor(
            descriptors[name],
            f"array_descriptors.{name}",
            expected_dtype=dtype,
            expected_shape=shape,
        )
    parent = _mapping(score.get("parent_bindings"), "C5 score parent bindings")
    expected_parent_keys = {
        "c2_raw_manifest_sha256",
        "c3_operator_stage_id",
        "c3_raw_manifest_sha256",
        "c3_raw_schema",
        "c4_raw_manifest_sha256",
        "c4_raw_schema",
        "c4_receipt_path",
        "c4_receipt_schema",
        "c4_receipt_sha256",
        "c4_score_path",
        "c4_score_schema",
        "c4_score_sha256",
        "c4_stage_id",
    }
    _exact_keys(parent, expected_parent_keys, "C5 score parent bindings")
    if (
        parent.get("c4_raw_schema") != C4_RAW_SCHEMA
        or parent.get("c4_receipt_schema") != C4_RECEIPT_SCHEMA
        or parent.get("c4_score_schema") != C4_SCORE_SCHEMA
        or parent.get("c3_raw_schema") != C3_RAW_SCHEMA
        or parent.get("c3_operator_stage_id") != PARENT_OPERATOR_STAGE_ID
        or parent.get("c4_stage_id") != PARENT_STAGE_ID
    ):
        raise C5ScoreError("C5 score parent schemas are invalid.")
    for field in (
        "c2_raw_manifest_sha256",
        "c3_raw_manifest_sha256",
        "c4_raw_manifest_sha256",
        "c4_receipt_sha256",
        "c4_score_sha256",
    ):
        _sha256(parent.get(field), f"C5 score parent_bindings.{field}")
    for field in ("c4_score_path", "c4_receipt_path"):
        value = parent.get(field)
        if (
            not isinstance(value, str)
            or value.startswith("/")
            or "\\" in value
            or ".." in Path(value).parts
        ):
            raise C5ScoreError(f"C5 score parent path {field} is unsafe.")

    rounding = _mapping(score.get("rounding_contract"), "C5 rounding contract")
    _exact_keys(
        rounding,
        {
            "comparison",
            "float64_epsilon",
            "gamma",
            "operation_budget_per_dot",
        },
        "C5 rounding contract",
    )
    if rounding.get("comparison") != (
        "Retained public arrays are manifest-byte-bound. Independent "
        "science uses fixed-order math.fsum and an elementwise gamma "
        "bound; cross-platform BLAS last bits are not required to match."
    ):
        raise C5ScoreError("C5 rounding comparison statement is invalid.")
    if (
        _validate_float_record(rounding["float64_epsilon"], "rounding epsilon")
        != _FLOAT_EPS
        or _validate_float_record(rounding["gamma"], "rounding gamma")
        != _REDUCTION_GAMMA
        or rounding.get("operation_budget_per_dot") != _REDUCTION_OPERATION_BUDGET
    ):
        raise C5ScoreError("C5 rounding constants are invalid.")

    channels = _mapping(score.get("channel_comparison"), "C5 channel comparison")
    _exact_keys(channels, {"combined_net", "pair", "scattering"}, "channel comparison")
    for channel in ("scattering", "pair"):
        record = _mapping(channels[channel], f"channel_comparison.{channel}")
        expected = {"kernel_formula", "parent_operator", "public_vs_clean_reduction"}
        if channel == "scattering":
            expected.add("pauli_rebucketing")
        _exact_keys(record, expected, f"channel_comparison.{channel}")
        reductions = _mapping(
            record["public_vs_clean_reduction"],
            f"channel_comparison.{channel}.public_vs_clean_reduction",
        )
        _exact_keys(
            reductions,
            {"gain", "loss", "loss_rate", "net"},
            f"channel_comparison.{channel}.public_vs_clean_reduction",
        )
        for field in ("gain", "loss", "loss_rate", "net"):
            _validate_metric_record(
                reductions[field],
                f"channel_comparison.{channel}.public_vs_clean_reduction.{field}",
            )
        parent_operator = _mapping(
            record["parent_operator"],
            f"channel_comparison.{channel}.parent_operator",
        )
        _exact_keys(
            parent_operator,
            {"gain", "loss", "net"},
            f"channel_comparison.{channel}.parent_operator",
        )
        for field in ("gain", "loss", "net"):
            metric = _mapping(
                parent_operator[field],
                f"channel_comparison.{channel}.parent_operator.{field}",
            )
            _exact_keys(
                metric,
                {
                    "l1_absolute_s_inv",
                    "linf_absolute_s_inv",
                    "symmetric_relative_l1",
                },
                f"channel_comparison.{channel}.parent_operator.{field}",
            )
            for metric_field in metric:
                _validate_float_record(
                    metric[metric_field],
                    f"channel_comparison.{channel}.parent_operator."
                    f"{field}.{metric_field}",
                )
    combined = _mapping(channels["combined_net"], "combined net comparison")
    _exact_keys(
        combined,
        {"l1_absolute_s_inv", "linf_absolute_s_inv", "symmetric_relative_l1"},
        "combined net comparison",
    )
    for value in combined.values():
        _validate_float_record(value, "combined net metric")

    acceptance = _mapping(score.get("acceptance"), "C5 acceptance")
    _exact_keys(acceptance, {"all_passed", "checks", "limits"}, "C5 acceptance")
    checks = _mapping(acceptance.get("checks"), "C5 acceptance checks")
    _exact_keys(
        checks,
        {
            "all_public_reductions_within_predeclared_rounding_bound",
            "combined_physical_net_matches_parent",
            "frozen_phonon_residual_bit_exact",
            "pair_physical_gain_matches_parent",
            "pair_physical_loss_matches_parent",
            "pair_physical_net_matches_parent",
            "scattering_rebucketed_gain_matches_parent",
            "scattering_rebucketed_loss_matches_parent",
            "scattering_physical_net_matches_parent",
            "scattering_weighted_number_conserved",
        },
        "C5 acceptance checks",
    )
    if any(value is not True for value in checks.values()):
        raise C5ScoreError("C5 score contains a failed or malformed acceptance check.")
    if acceptance.get("all_passed") is not True:
        raise C5ScoreError("C5 score acceptance is false.")
    limits = _mapping(acceptance.get("limits"), "C5 acceptance limits")
    _exact_keys(
        limits,
        {
            "combined_and_per_channel_net_symmetric_relative_l1",
            "like_for_like_bucket_symmetric_relative_l1",
            "scattering_weighted_number_relative",
        },
        "C5 acceptance limits",
    )
    if (
        _validate_float_record(
            limits["combined_and_per_channel_net_symmetric_relative_l1"],
            "C5 acceptance net limit",
        )
        != _NET_PARITY_LIMIT
        or _validate_float_record(
            limits["like_for_like_bucket_symmetric_relative_l1"],
            "C5 acceptance bucket limit",
        )
        != _BUCKET_PARITY_LIMIT
        or _validate_float_record(
            limits["scattering_weighted_number_relative"],
            "C5 acceptance conservation limit",
        )
        != _SCATTERING_CONSERVATION_LIMIT
    ):
        raise C5ScoreError("C5 score acceptance limits are invalid.")
    for required_object in (
        "bookkeeping",
        "component_locality",
        "conservation",
        "limitations",
        "map_identity",
        "operator_inputs",
        "units",
    ):
        if not _mapping(score.get(required_object), f"C5 score {required_object}"):
            raise C5ScoreError(f"C5 score {required_object} is empty.")
    if not _json_value_bit_exact(
        score.get("source_binding"),
        _EXPECTED_SOURCE_BINDING,
    ):
        raise C5ScoreError("C5 score source_binding is invalid.")
    contracts = _mapping(score.get("contracts"), "C5 score contracts")
    _exact_keys(
        contracts,
        {"comparison_contract", "coordinate_contract", "frozen_inputs"},
        "C5 score contracts",
    )
    for key in contracts:
        if not _mapping(contracts[key], f"C5 score contracts.{key}"):
            raise C5ScoreError(f"C5 score contracts.{key} is empty.")
    _validate_canonical_pins(score)


def _load_c5_score_unbound(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file_once(path, "checked C5 score")
    score = _parse_json(raw, "checked C5 score")
    if raw != canonical_score_bytes(score):
        raise C5ScoreError("Checked C5 score is not canonical JSON.")
    return score, raw


def _receipt_parent_from_score(score: dict[str, Any]) -> dict[str, object]:
    parent = _mapping(score.get("parent_bindings"), "C5 score parent bindings")
    return {
        "raw_manifest_sha256": parent.get("c4_raw_manifest_sha256"),
        "raw_schema": parent.get("c4_raw_schema"),
        "receipt_file_sha256": parent.get("c4_receipt_sha256"),
        "receipt_schema": parent.get("c4_receipt_schema"),
        "score_file_sha256": parent.get("c4_score_sha256"),
        "score_schema": parent.get("c4_score_schema"),
    }


def load_c5_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Strictly load the repository C5 score/raw/C4 trust anchor."""

    raw = _read_regular_file_once(path, "C5 raw-manifest receipt")
    receipt = _parse_json(raw, "C5 raw-manifest receipt")
    if raw != _canonical_json_bytes(receipt):
        raise C5ScoreError("C5 raw-manifest receipt is not canonical JSON.")
    _exact_keys(
        receipt,
        {"checked_score", "parent_c4", "qualification", "raw_bundle", "schema"},
        "C5 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C5ScoreError("C5 raw-manifest receipt schema is unsupported.")
    if receipt.get("qualification") != (
        "Repository trust anchor for the externally retained C5 raw manifest, "
        "the complete canonical checked-score bytes, and the independently "
        "replayed C4/C3/C2 parent chain; it does not contain or replace the "
        "raw arrays."
    ):
        raise C5ScoreError("C5 raw-manifest receipt qualification is invalid.")
    checked = _mapping(receipt.get("checked_score"), "C5 receipt checked_score")
    _exact_keys(checked, {"file_sha256", "schema"}, "C5 receipt checked_score")
    if checked.get("schema") != SCHEMA:
        raise C5ScoreError("C5 receipt score schema is invalid.")
    _sha256(checked.get("file_sha256"), "C5 receipt score SHA-256")
    raw_bundle = _mapping(receipt.get("raw_bundle"), "C5 receipt raw bundle")
    _exact_keys(raw_bundle, {"manifest_sha256", "schema"}, "C5 receipt raw bundle")
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C5ScoreError("C5 receipt raw schema is invalid.")
    _sha256(raw_bundle.get("manifest_sha256"), "C5 receipt raw manifest SHA-256")
    parent = _mapping(receipt.get("parent_c4"), "C5 receipt parent C4")
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
        "C5 receipt parent C4",
    )
    if (
        parent.get("raw_schema") != C4_RAW_SCHEMA
        or parent.get("receipt_schema") != C4_RECEIPT_SCHEMA
        or parent.get("score_schema") != C4_SCORE_SCHEMA
    ):
        raise C5ScoreError("C5 receipt C4 schemas are invalid.")
    for field in (
        "raw_manifest_sha256",
        "receipt_file_sha256",
        "score_file_sha256",
    ):
        _sha256(parent.get(field), f"C5 receipt parent_c4.{field}")
    return receipt


def load_c5_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Load a checked C5 score and bind it to canonical C4 anchors."""

    score, score_raw = _load_c5_score_unbound(path)
    receipt = load_c5_receipt(receipt_path)
    checked = _mapping(receipt.get("checked_score"), "C5 receipt checked score")
    if hashlib.sha256(score_raw).hexdigest() != checked.get("file_sha256"):
        raise C5ScoreError("Checked C5 score bytes do not match its receipt.")
    if score.get("raw_bundle") != receipt.get("raw_bundle"):
        raise C5ScoreError("Checked C5 raw binding does not match its receipt.")
    if _receipt_parent_from_score(score) != receipt.get("parent_c4"):
        raise C5ScoreError("Checked C5 C4 binding does not match its receipt.")
    parent = _mapping(score.get("parent_bindings"), "checked C5 parent bindings")
    expected_score_path = DEFAULT_C4_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
    expected_receipt_path = DEFAULT_C4_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
    if (
        parent.get("c4_score_path") != expected_score_path
        or parent.get("c4_receipt_path") != expected_receipt_path
    ):
        raise C5ScoreError("Checked C5 does not bind canonical C4 parent paths.")
    c4_score_path, c4_score_bytes = _repository_file_snapshot(
        DEFAULT_C4_SCORE,
        "canonical C4 score",
    )
    c4_receipt_path, c4_receipt_bytes = _repository_file_snapshot(
        DEFAULT_C4_RECEIPT,
        "canonical C4 receipt",
    )
    accepted_c4 = load_c4_score(c4_score_path, receipt_path=c4_receipt_path)
    accepted_raw = _mapping(accepted_c4.get("raw_bundle"), "accepted C4 raw")
    accepted_parent = _mapping(
        accepted_c4.get("parent_bindings"),
        "accepted C4 parent bindings",
    )
    if (
        hashlib.sha256(c4_score_bytes).hexdigest() != parent.get("c4_score_sha256")
        or hashlib.sha256(c4_receipt_bytes).hexdigest()
        != parent.get("c4_receipt_sha256")
        or accepted_raw.get("schema") != parent.get("c4_raw_schema")
        or accepted_raw.get("manifest_sha256")
        != parent.get("c4_raw_manifest_sha256")
        or accepted_parent.get("c3_operator_stage_id")
        != parent.get("c3_operator_stage_id")
        or accepted_parent.get("c3_raw_schema") != parent.get("c3_raw_schema")
        or accepted_parent.get("c3_raw_manifest_sha256")
        != parent.get("c3_raw_manifest_sha256")
        or accepted_parent.get("c2_raw_manifest_sha256")
        != parent.get("c2_raw_manifest_sha256")
    ):
        raise C5ScoreError("Checked C5 canonical C4/C3/C2 binding is stale.")
    _assert_file_snapshot(c4_score_path, c4_score_bytes, "canonical C4 score")
    _assert_file_snapshot(c4_receipt_path, c4_receipt_bytes, "canonical C4 receipt")
    return score


def build_c5_receipt(
    score_path: Path = DEFAULT_SCORE,
    *,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Build a receipt only after independently reproducing C5 score bytes."""

    checked_score_path, checked_score_snapshot = _repository_file_snapshot(
        score_path,
        "checked C5 score for receipt",
    )
    score, score_raw = _load_c5_score_unbound(checked_score_path)
    if score_raw != checked_score_snapshot:
        raise C5ScoreError("Checked C5 score changed before receipt replay.")
    rebuilt = build_c5_score(
        c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_score_bytes(rebuilt) != score_raw:
        raise C5ScoreError(
            "C5 receipt refuses score bytes that do not independently reproduce "
            "from the selected C5/C4/C3/C2 raw evidence."
        )
    _assert_file_snapshot(
        checked_score_path,
        checked_score_snapshot,
        "checked C5 score for receipt",
    )
    return {
        "checked_score": {
            "file_sha256": hashlib.sha256(score_raw).hexdigest(),
            "schema": SCHEMA,
        },
        "parent_c4": _receipt_parent_from_score(score),
        "qualification": (
            "Repository trust anchor for the externally retained C5 raw manifest, "
            "the complete canonical checked-score bytes, and the independently "
            "replayed C4/C3/C2 parent chain; it does not contain or replace the "
            "raw arrays."
        ),
        "raw_bundle": dict(
            _mapping(score.get("raw_bundle"), "C5 score raw bundle")
        ),
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
        raise FileExistsError(f"C5 output already exists: {target}")
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
        try:
            if (
                target.exists()
                and temporary.exists()
                and target.stat().st_ino == temporary.stat().st_ino
            ):
                target.unlink()
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_c5_score(
    output_path: Path,
    c5_bundle_dir: Path,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    score = build_c5_score(
        c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, canonical_score_bytes(score))


def write_c5_receipt(
    output_path: Path,
    *,
    score_path: Path = DEFAULT_SCORE,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    receipt = build_c5_receipt(
        score_path,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, _canonical_json_bytes(receipt))


def _add_parent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c5-bundle", type=Path, required=True)
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c4-score", type=Path, default=DEFAULT_C4_SCORE)
    parser.add_argument("--c4-receipt", type=Path, default=DEFAULT_C4_RECEIPT)
    parser.add_argument("--c3-score", type=Path, default=DEFAULT_C3_SCORE)
    parser.add_argument("--c3-receipt", type=Path, default=DEFAULT_C3_RECEIPT)
    parser.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    parser.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="build the checked C5 score")
    _add_parent_arguments(score)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    receipt = subparsers.add_parser("receipt", help="build the C5 receipt")
    _add_parent_arguments(receipt)
    receipt.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    receipt.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    common = {
        "c4_bundle_dir": args.c4_bundle,
        "c3_bundle_dir": args.c3_bundle,
        "c2_bundle_dir": args.c2_bundle,
        "c4_score_path": args.c4_score,
        "c4_receipt_path": args.c4_receipt,
        "c3_score_path": args.c3_score,
        "c3_receipt_path": args.c3_receipt,
        "c2_score_path": args.c2_score,
        "c2_receipt_path": args.c2_receipt,
    }
    if args.command == "receipt":
        result = write_c5_receipt(
            args.output,
            score_path=args.score,
            c5_bundle_dir=args.c5_bundle,
            **common,
        )
    else:
        result = write_c5_score(
            args.output,
            args.c5_bundle,
            **common,
        )
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
