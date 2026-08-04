"""Independently verify the formal Figure 6 C6 phonon-balance substitution.

The C6 producer intentionally lives in another module.  This verifier does
not import that producer or the changed public qpsim phonon-balance APIs for
its scientific replay.  It strictly loads the externally retained raw bundle,
replays the accepted C5/C4/C3/C2 chain, and independently transcribes:

* the frozen center-grid phonon-frequency map;
* the phonon-side scattering (``2 K^-/(pi Delta tau_0^PB)``) and
  pair/recombination (``K^+/(pi Delta tau_0^PB)``) kernels;
* the per-omega source/sink contractions and their affine ``(a, b)``
  decomposition;
* the Kaplan ``S_+`` pair-breaking quadrature correction from its closed
  elliptic form;
* the thermal phonon occupation, bath-escape balance, and the
  detailed-balance thermal control.

Public qpsim evaluates the pair contractions with vectorized bincounts whose
last bits may depend on the accumulation order, so retained public arrays are
byte-bound to their raw manifest while comparison to a clean-room
``math.fsum`` reduction uses a predeclared IEEE-754 gamma bound.  Derived raw
relationships remain byte-exact.

C6 is one frozen-state operator differential.  It does not run Newton, alter
the frozen state, re-evaluate the QP channels, or claim a C6 nonlinear root,
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
from scipy.special import ellipe

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
    load_c4_raw_bundle,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as DEFAULT_C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_SCORE as DEFAULT_C5_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import (
    RAW_SCHEMA as C5_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    RECEIPT_SCHEMA as C5_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    SCHEMA as C5_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    build_c5_score,
    load_c5_raw_bundle,
    load_c5_score,
)
from validation.fischer_2023.fig6_author_c5_score import (
    canonical_score_bytes as canonical_c5_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_SCHEMA = "qpsim.fischer2023.fig6-author-c6-phonon-balance-bundle.v1"
SCHEMA = "qpsim.fischer2023.fig6-author-c6-phonon-balance-score.v1"
RECEIPT_SCHEMA = (
    "qpsim.fischer2023.fig6-author-c6-phonon-balance-raw-manifest-receipt.v1"
)
DEFAULT_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c6-phonon-balance-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c6-raw-manifest-receipt.json"
)

STAGE_ID = "C6"
PARENT_STAGE_ID = "C5"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "phonon_balance"
SECONDS_PER_NS = 1.0e-9
N_QP = 1640
N_ACTIVE = 1620
N_OMEGA = 3600
ACTIVE_START = 20
AUTHOR_OMEGA_STOP = 1619
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
T_BATH_K = 0.2
GAP_UEV = 180.0

_CHANNELS = ("scattering", "pair", "pair_control", "escape")
_FIELDS = ("gain", "loss", "net")

_ARRAY_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FLOAT_EPS = float(np.finfo(np.float64).eps)
# One omega bin collects at most N_QP ordered pairs per direction; each term
# multiplies a handful of rounded factors before one reduction.  8*N+64 is
# deliberately conservative and matches the C5 verifier's fixed budget.  It
# is fixed before observing any C6 raw output.
_REDUCTION_OPERATION_BUDGET = 8 * N_QP + 64
_REDUCTION_GAMMA = (
    _REDUCTION_OPERATION_BUDGET
    * _FLOAT_EPS
    / (1.0 - _REDUCTION_OPERATION_BUDGET * _FLOAT_EPS)
)
# The Kaplan correction chains the pair reduction with one elliptic-integral
# evaluation and one quotient; scipy builds may differ in the last bits of
# ellipe, so the correction bound doubles the reduction budget.
_CORRECTION_OPERATION_BUDGET = 2 * _REDUCTION_OPERATION_BUDGET
_CORRECTION_GAMMA = (
    _CORRECTION_OPERATION_BUDGET
    * _FLOAT_EPS
    / (1.0 - _CORRECTION_OPERATION_BUDGET * _FLOAT_EPS)
)
_NET_PARITY_LIMIT = 1.0e-12
_BUCKET_PARITY_LIMIT = 1.0e-12
_DETAILED_BALANCE_LIMIT = 1.0e-12
# The escape net is a near-thermal cancellation; its parent difference is
# bounded elementwise by rounding of the two thermal-occupation unit paths
# rather than by a bulk relative gate.
_ESCAPE_NET_ELEMENTWISE_BUDGET = 16.0
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_EXPECTED_RAW_MANIFEST_SHA256 = (
    "1533688e414af1d3e5dc8321657118f565a3f118cf455fddb6ef0612db08be08"
)
_EXPECTED_EVIDENCE_DIGEST = (
    "1e22865a123d2d977b5073bde85f9472896f1c14746228d6c1e5715cb6782f6d"
)
_EXPECTED_CANONICAL_METRICS: dict[str, float] = {
    "c6spe0_residual_symmetric_relative_l1": float.fromhex(
        "0x1.de9eacc8d07cbp-33"
    ),
    "c6spe_residual_symmetric_relative_l1": float.fromhex(
        "0x1.13c479484160ap-19"
    ),
    "detailed_balance_pair_relative": float.fromhex("0x1.1a634e30640dap-51"),
    "detailed_balance_scattering_relative": float.fromhex(
        "0x1.662f50338e7a6p-51"
    ),
    "pair_control_net_symmetric_relative_l1": float.fromhex(
        "0x1.0f0f2dce1fb5cp-52"
    ),
    "pair_public_net_symmetric_relative_l1": float.fromhex(
        "0x1.2d9151989bb24p-7"
    ),
    "scattering_net_symmetric_relative_l1": float.fromhex(
        "0x1.70d082acc23b9p-50"
    ),
}
_EXPECTED_RESIDUAL_NPY_SHA256: dict[str, str] = {
    "c6_qp_residual": (
        "9f8e4f18352c0dec82c1ce8db389cab9b87ad7d336541df885f5d64f290365a7"
    ),
    "c6e_phonon_residual": (
        "221a479354db46dabb166a1051bca4937fe14b4f502953f165f1b8d61882be15"
    ),
    "c6p0_phonon_residual": (
        "679973629bdd404bacce8235230f8e4244cc06aae54559db871c8ccf55b18422"
    ),
    "c6p_phonon_residual": (
        "31e67e12b3936a752c5a5571aa8f0fca6ca38aadadc4d7af1946123d88313099"
    ),
    "c6s_phonon_residual": (
        "483f815553073d18d251bd371f45ba1a4a7ca84a978dcb8e1b99a43970c627bf"
    ),
    "c6spe0_phonon_residual": (
        "bbbac456dfafc4b9316728ea15685504d7ca831637cae1577e181eae844eb9f0"
    ),
    "c6spe_phonon_residual": (
        "47fa70f857ddb78f45154300909e810c5d1028ee513fa07637f32195ad469ff1"
    ),
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
        "complete qpsim Python/material source tree, C6 producer, "
        "C2/C3/C4/C5 replay verifiers, C5 producer, and provenance helper"
    ),
}

_FLOAT_1640 = ("<f8", (N_QP,))
_BOOL_1640 = ("|b1", (N_QP,))
_FLOAT_3600 = ("<f8", (N_OMEGA,))
_BOOL_3600 = ("|b1", (N_OMEGA,))
_FLOAT_1619 = ("<f8", (AUTHOR_OMEGA_STOP,))
_INT64_1619 = ("<i8", (AUTHOR_OMEGA_STOP,))
_FLOAT_MATRIX = ("<f8", (N_QP, N_QP))
_INT64_MATRIX = ("<i8", (N_QP, N_QP))
_INT8_MATRIX = ("|i1", (N_QP, N_QP))

_ARRAY_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "parent_E_centers_ueV": _FLOAT_1640,
    "parent_dE_ueV": _FLOAT_1640,
    "parent_f": _FLOAT_1640,
    "parent_active_mask": _BOOL_1640,
    "parent_cell_density": _FLOAT_1640,
    "parent_cell_weights_ueV": _FLOAT_1640,
    "parent_projected_n_phonon": _FLOAT_3600,
    "parent_legacy_phonon_support_mask": _BOOL_3600,
    "parent_phonon_to_native_omega_index": _INT64_1619,
    "parent_qp_residual_s_inv": _FLOAT_1640,
    "parent_phonon_residual_s_inv": _FLOAT_1619,
    "qpsim_omega_ueV": _FLOAT_3600,
    "qpsim_omega_idx_diff": _INT64_MATRIX,
    "qpsim_omega_idx_sum": _INT64_MATRIX,
    "qpsim_diff_sign": _INT8_MATRIX,
    "qpsim_phonon_scattering_kernel_ns_inv_ueV_inv": _FLOAT_MATRIX,
    "qpsim_phonon_pair_kernel_ns_inv_ueV_inv": _FLOAT_MATRIX,
    "qpsim_scattering_a_ns_inv": _FLOAT_3600,
    "qpsim_scattering_b_ns_inv": _FLOAT_3600,
    "qpsim_pair_a_ns_inv": _FLOAT_3600,
    "qpsim_pair_b_ns_inv": _FLOAT_3600,
    "qpsim_pair_control_a_ns_inv": _FLOAT_3600,
    "qpsim_pair_control_b_ns_inv": _FLOAT_3600,
    "qpsim_combined_a_ns_inv": _FLOAT_3600,
    "qpsim_combined_b_ns_inv": _FLOAT_3600,
    "qpsim_thermal_n_ph": _FLOAT_3600,
    "qpsim_balance_residual_ns_inv": _FLOAT_3600,
    "qpsim_db_f": _FLOAT_1640,
    "qpsim_db_scattering_a_ns_inv": _FLOAT_3600,
    "qpsim_db_scattering_b_ns_inv": _FLOAT_3600,
    "qpsim_db_scattering_net_ns_inv": _FLOAT_3600,
    "qpsim_db_pair_a_ns_inv": _FLOAT_3600,
    "qpsim_db_pair_b_ns_inv": _FLOAT_3600,
    "qpsim_db_pair_net_ns_inv": _FLOAT_3600,
    "c6_qp_residual_s_inv": _FLOAT_1640,
    "c6s_phonon_residual_s_inv": _FLOAT_1619,
    "c6p_phonon_residual_s_inv": _FLOAT_1619,
    "c6p0_phonon_residual_s_inv": _FLOAT_1619,
    "c6e_phonon_residual_s_inv": _FLOAT_1619,
    "c6spe_phonon_residual_s_inv": _FLOAT_1619,
    "c6spe0_phonon_residual_s_inv": _FLOAT_1619,
}
for _parent_channel in ("scattering", "pair", "escape"):
    for _field in _FIELDS:
        _ARRAY_SPECS[
            f"parent_phonon_{_parent_channel}_{_field}_s_inv"
        ] = _FLOAT_1619
for _channel in _CHANNELS:
    for _field in _FIELDS:
        _ARRAY_SPECS[f"qpsim_phonon_{_channel}_{_field}_ns_inv"] = _FLOAT_3600
        _ARRAY_SPECS[f"qpsim_phonon_{_channel}_{_field}_s_inv"] = _FLOAT_3600
        _ARRAY_SPECS[f"phonon_{_channel}_delta_{_field}_s_inv"] = _FLOAT_1619

_EXPECTED_ARRAY_NAMES = frozenset(_ARRAY_SPECS)
if len(_EXPECTED_ARRAY_NAMES) != 86:  # pragma: no cover - import-time invariant
    raise RuntimeError("Internal C6 raw array closure must contain exactly 86 arrays.")

_RAW_METADATA_KEYS = {
    "array_descriptors",
    "balance_certification",
    "bookkeeping_contract",
    "comparison_contract",
    "component_locality",
    "coordinate_contract",
    "detailed_balance",
    "extension_policy",
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
    "balance_certification",
    "bookkeeping",
    "channel_comparison",
    "component_locality",
    "contracts",
    "detailed_balance",
    "extension_policy",
    "kaplan_correction",
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

_C6_PRODUCER_SOURCE = (
    REPOSITORY_ROOT / "validation/fischer_2023/fig6_author_c6_bundle.py"
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
        "validation/fischer_2023/fig6_author_c5_bundle.py",
        "validation/fischer_2023/fig6_author_c5_score.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/reference_models/__init__.py",
        "validation/reference_models/fischer_2023/__init__.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    )
)
_RAW_SOURCE_HASHES_AT_IMPORT = source_manifest(
    _C6_PRODUCER_SOURCE,
    extra_validation_modules=_TRANSITIVE_VALIDATION_SOURCES,
)
_SOURCE_HASHES_AT_IMPORT = source_manifest(
    Path(__file__).resolve(),
    extra_validation_modules=(
        *_TRANSITIVE_VALIDATION_SOURCES,
        _C6_PRODUCER_SOURCE,
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
    raise RuntimeError("C6 source hashes changed during import.")
_VERIFIER_RELATIVE = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
_RAW_SOURCES_FROM_VERIFIER_CLOSURE = {
    relative: digest
    for relative, digest in _SOURCE_HASHES_AT_IMPORT.items()
    if relative != _VERIFIER_RELATIVE
}
if (
    _RAW_SOURCES_FROM_VERIFIER_CLOSURE != _RAW_SOURCE_HASHES_AT_IMPORT
):  # pragma: no cover - import-time provenance invariant
    raise RuntimeError("C6 verifier and producer source closures are inconsistent.")


class C6ScoreError(ValueError):
    """The C6 raw evidence, parent chain, score, or receipt is malformed."""


@dataclass(frozen=True)
class _DirectoryState:
    root_identity: tuple[int, int, int, int, int]
    entries: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass
class _ParentContext:
    c5_metadata: dict[str, Any]
    c5_arrays: dict[str, np.ndarray]
    c5_manifest_sha256: str
    c5_score: dict[str, Any]
    c5_score_path: Path
    c5_score_bytes: bytes
    c5_receipt_path: Path
    c5_receipt_bytes: bytes
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
    c4_manifest_sha256: str
    c5_bundle_dir: Path
    c5_directory_state: _DirectoryState
    c4_bundle_dir: Path
    c4_directory_state: _DirectoryState
    c3_bundle_dir: Path
    c3_directory_state: _DirectoryState
    c2_bundle_dir: Path
    c2_directory_state: _DirectoryState


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C6ScoreError(f"C6 numerical source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C6ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C6ScoreError(f"Forbidden non-finite JSON constant {token!r}.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C6ScoreError(f"{label} is not strict UTF-8 JSON.") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C6ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise C6ScoreError(f"{label} keys are invalid; missing={missing}, extra={extra}.")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C6ScoreError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise C6ScoreError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise C6ScoreError(f"{label} must be >= {minimum}.")
    return value


def _finite_scalar(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C6ScoreError(f"{label} must be a finite scalar.")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise C6ScoreError(f"{label} must be a finite {qualifier}scalar.")
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
        raise C6ScoreError("C6 scalar records must be finite.")
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
        raise C6ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C6ScoreError(f"{label} must be a regular non-symlink file.")
    try:
        with candidate.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = candidate.lstat()
    except OSError as exc:
        raise C6ScoreError(f"{label} changed or became unreadable.") from exc
    if not (
        _stat_identity(before)
        == _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(after)
    ):
        raise C6ScoreError(f"{label} changed while it was being read.")
    return content


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C6ScoreError(f"{label} must stay inside the repository.") from exc
    if resolved != Path(path).absolute() or resolved.is_symlink():
        raise C6ScoreError(f"{label} is unsafe or a symlink.")
    return resolved, _read_regular_file_once(resolved, label)


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    if _read_regular_file_once(path, label) != expected:
        raise C6ScoreError(f"{label} changed during C6 verification.")


def _directory_state(path: Path, label: str) -> tuple[Path, _DirectoryState]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise C6ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise C6ScoreError(f"{label} must be a non-symlink directory.")
    root = candidate.resolve()
    if root != candidate.absolute() or root.is_symlink():
        raise C6ScoreError(f"{label} is unsafe or a symlink.")
    entries: list[tuple[str, tuple[int, int, int, int, int]]] = []
    try:
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise C6ScoreError(f"{label} contains a symlink.")
            entries.append((child.name, _stat_identity(child_stat)))
        after = root.lstat()
    except OSError as exc:
        raise C6ScoreError(f"{label} changed while being enumerated.") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise C6ScoreError(f"{label} changed while being enumerated.")
    return root, _DirectoryState(_stat_identity(after), tuple(entries))


def _assert_directory_state(path: Path, expected: _DirectoryState, label: str) -> None:
    root, actual = _directory_state(path, label)
    del root
    if actual != expected:
        raise C6ScoreError(f"{label} changed during C6 verification.")


def load_c6_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one closed canonical NPY-v3 C6 raw bundle."""

    root, before = _directory_state(bundle_dir, "C6 raw bundle")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C6 raw manifest")
    manifest = _parse_json(manifest_raw, "C6 raw manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C6 raw manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C6ScoreError("C6 raw manifest schema is unsupported.")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise C6ScoreError("C6 raw manifest is not canonical JSON.")
    files = _mapping(manifest.get("files"), "C6 raw manifest files")
    metadata = _mapping(manifest.get("metadata"), "C6 raw metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C6 raw metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C6ScoreError("C6 raw metadata schema is unsupported.")
    expected_filenames = {f"{name}.npy" for name in _EXPECTED_ARRAY_NAMES}
    if set(files) != expected_filenames or len(files) != 86:
        raise C6ScoreError("C6 raw file closure is invalid.")
    if {name for name, _identity in before.entries} != expected_filenames | {
        "manifest.json"
    }:
        raise C6ScoreError("C6 raw directory closure is invalid.")
    if any(not stat.S_ISREG(identity[2]) for _name, identity in before.entries):
        raise C6ScoreError("C6 raw bundle contains a non-file entry.")

    arrays: dict[str, np.ndarray] = {}
    for filename in sorted(expected_filenames):
        name = filename[:-4]
        if (
            Path(filename).name != filename
            or _ARRAY_NAME_RE.fullmatch(name) is None
            or not filename.endswith(".npy")
        ):
            raise C6ScoreError(f"Unsafe C6 raw filename {filename!r}.")
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        expected_sha = _sha256(record.get("sha256"), f"files.{filename}.sha256")
        expected_size = _strict_int(
            record.get("size_bytes"),
            f"files.{filename}.size_bytes",
            minimum=1,
        )
        content = _read_regular_file_once(root / filename, f"C6 raw {filename}")
        if (
            len(content) != expected_size
            or hashlib.sha256(content).hexdigest() != expected_sha
        ):
            raise C6ScoreError(f"C6 raw file {filename!r} failed manifest binding.")
        if len(content) < 8 or content[:8] != b"\x93NUMPY\x03\x00":
            raise C6ScoreError(f"C6 raw file {filename!r} is not canonical NPY v3.")
        try:
            stream = io.BytesIO(content)
            loaded = np.lib.format.read_array(stream, allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise C6ScoreError(f"Cannot load C6 raw array {filename!r}.") from exc
        if stream.tell() != len(content):
            raise C6ScoreError(f"C6 raw file {filename!r} contains trailing bytes.")
        array = np.asarray(loaded)
        expected_dtype, expected_shape = _ARRAY_SPECS[name]
        if array.dtype.str != expected_dtype or array.shape != expected_shape:
            raise C6ScoreError(
                f"C6 raw {name!r} expected dtype/shape "
                f"{expected_dtype}/{expected_shape}, got "
                f"{array.dtype.str}/{array.shape}."
            )
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise C6ScoreError(f"C6 raw array {filename!r} contains non-finite values.")
        if array.dtype.kind == "f" and np.any((array == 0.0) & np.signbit(array)):
            raise C6ScoreError(
                f"C6 raw array {filename!r} contains non-canonical signed zero."
            )
        if _npy_bytes(array) != content:
            raise C6ScoreError(
                f"C6 raw file {filename!r} is not byte-canonical NPY v3."
            )
        arrays[name] = array
    descriptors = _mapping(metadata.get("array_descriptors"), "C6 raw descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C6ScoreError("C6 raw array descriptors are incomplete, forged, or stale.")
    _assert_directory_state(root, before, "C6 raw bundle")
    return metadata, arrays, hashlib.sha256(manifest_raw).hexdigest()


def _accept_parent(
    c5_bundle_dir: Path,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path,
    c5_receipt_path: Path,
    c4_score_path: Path,
    c4_receipt_path: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
) -> _ParentContext:
    """Replay the selected C5/C4/C3/C2 chain and bind canonical C5 bytes."""

    c5_root, c5_state = _directory_state(c5_bundle_dir, "selected C5 raw bundle")
    c4_root, c4_state = _directory_state(c4_bundle_dir, "selected C4 raw bundle")
    c3_root, c3_state = _directory_state(c3_bundle_dir, "selected C3 raw bundle")
    c2_root, c2_state = _directory_state(c2_bundle_dir, "selected C2 raw bundle")
    checked_c5_path, checked_c5_bytes = _repository_file_snapshot(
        c5_score_path,
        "checked C5 score",
    )
    checked_c5_receipt_path, checked_c5_receipt_bytes = _repository_file_snapshot(
        c5_receipt_path,
        "checked C5 receipt",
    )
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
    accepted_c5 = load_c5_score(
        checked_c5_path,
        receipt_path=checked_c5_receipt_path,
    )
    rebuilt_c5 = build_c5_score(
        c5_root,
        c4_bundle_dir=c4_root,
        c3_bundle_dir=c3_root,
        c2_bundle_dir=c2_root,
        c4_score_path=checked_c4_path,
        c4_receipt_path=checked_c4_receipt_path,
        c3_score_path=checked_c3_path,
        c3_receipt_path=checked_c3_receipt_path,
        c2_score_path=checked_c2_path,
        c2_receipt_path=checked_c2_receipt_path,
    )
    if canonical_c5_score_bytes(rebuilt_c5) != checked_c5_bytes:
        raise C6ScoreError(
            "Selected C5/C4/C3/C2 raw evidence does not reproduce the complete "
            "checked C5 score bytes."
        )
    if not _json_value_bit_exact(accepted_c5, rebuilt_c5):
        raise C6ScoreError("Accepted and independently replayed C5 scores disagree.")
    c5_metadata, c5_arrays, c5_manifest_sha = load_c5_raw_bundle(c5_root)
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_root)
    _c4_metadata, _c4_arrays, c4_manifest_sha = load_c4_raw_bundle(c4_root)
    raw_binding = _mapping(accepted_c5.get("raw_bundle"), "accepted C5 raw binding")
    if (
        raw_binding.get("schema") != C5_RAW_SCHEMA
        or raw_binding.get("manifest_sha256") != c5_manifest_sha
    ):
        raise C6ScoreError("Selected C5 raw bundle is not the accepted C5 evidence.")
    parent_binding = _mapping(
        accepted_c5.get("parent_bindings"),
        "accepted C5 parent binding",
    )
    if (
        parent_binding.get("c3_raw_schema") != C3_RAW_SCHEMA
        or parent_binding.get("c3_raw_manifest_sha256") != c3_manifest_sha
        or parent_binding.get("c4_raw_schema") != C4_RAW_SCHEMA
        or parent_binding.get("c4_raw_manifest_sha256") != c4_manifest_sha
    ):
        raise C6ScoreError("Selected C3/C4 raw bundles are not C5's accepted ancestors.")
    stage = _mapping(accepted_c5.get("stage"), "accepted C5 stage")
    if stage != {
        "changed_component": "qp_phonon_operator",
        "comparison_stage_id": "C4",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C4",
        "stage_id": "C5",
        "status": "completed",
    }:
        raise C6ScoreError("Accepted parent is not the completed formal C5 stage.")
    _assert_file_snapshot(checked_c5_path, checked_c5_bytes, "checked C5 score")
    _assert_file_snapshot(
        checked_c5_receipt_path,
        checked_c5_receipt_bytes,
        "checked C5 receipt",
    )
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
    _assert_directory_state(c5_root, c5_state, "selected C5 raw bundle")
    _assert_directory_state(c4_root, c4_state, "selected C4 raw bundle")
    _assert_directory_state(c3_root, c3_state, "selected C3 raw bundle")
    _assert_directory_state(c2_root, c2_state, "selected C2 raw bundle")
    return _ParentContext(
        c5_metadata=c5_metadata,
        c5_arrays=c5_arrays,
        c5_manifest_sha256=c5_manifest_sha,
        c5_score=accepted_c5,
        c5_score_path=checked_c5_path,
        c5_score_bytes=checked_c5_bytes,
        c5_receipt_path=checked_c5_receipt_path,
        c5_receipt_bytes=checked_c5_receipt_bytes,
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
        c4_manifest_sha256=c4_manifest_sha,
        c5_bundle_dir=c5_root,
        c5_directory_state=c5_state,
        c4_bundle_dir=c4_root,
        c4_directory_state=c4_state,
        c3_bundle_dir=c3_root,
        c3_directory_state=c3_state,
        c2_bundle_dir=c2_root,
        c2_directory_state=c2_state,
    )


def _recheck_parent(parent: _ParentContext) -> None:
    _assert_file_snapshot(parent.c5_score_path, parent.c5_score_bytes, "checked C5 score")
    _assert_file_snapshot(
        parent.c5_receipt_path,
        parent.c5_receipt_bytes,
        "checked C5 receipt",
    )
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
        parent.c5_bundle_dir,
        parent.c5_directory_state,
        "selected C5 raw bundle",
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
        "tau_0_pb_s",
        "tau_l_s",
        "temperature_K",
    }
    if not required <= set(values):
        raise C6ScoreError("Accepted C3 parameters lack C6 operator inputs.")
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
    tau_0_pb_parent_s = _finite_scalar(
        values["tau_0_pb_s"],
        "tau_0_pb_s",
        positive=True,
    )
    tau_l_parent_s = _finite_scalar(
        values["tau_l_s"],
        "tau_l_s",
        positive=True,
    )
    tau_0_ns = tau_0_parent_s / SECONDS_PER_NS
    tau_0_pb_ns = tau_0_pb_parent_s / SECONDS_PER_NS
    tau_l_ns = tau_l_parent_s / SECONDS_PER_NS
    if tau_0_pb_ns != TAU_PB_NS or tau_l_ns != TAU_L_NS:
        raise C6ScoreError(
            "Accepted C3 pair-breaking/escape times do not match the "
            "declared C6 operator constants."
        )
    native_parameters = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "accepted C3 native grid parameters",
    )
    gap_uev = _finite_scalar(
        native_parameters.get("gap_ueV"),
        "native gap_ueV",
        positive=True,
    )
    if gap_uev != GAP_UEV:
        raise C6ScoreError("Accepted C3 native gap does not match the C6 constant.")
    gap_parent_ev = _finite_scalar(
        values["gap_eV"],
        "parent gap_eV",
        positive=True,
    )
    k_b_uev_per_k = boltzmann / electron_charge * 1.0e6
    temperature = _finite_scalar(values["temperature_K"], "temperature_K")
    if temperature != T_BATH_K:
        raise C6ScoreError("Accepted C3 bath temperature does not match C6.")
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
        "tau_0_pb_ns": _float_record(tau_0_pb_ns),
        "tau_0_pb_parent_s": _float_record(tau_0_pb_parent_s),
        "tau_l_ns": _float_record(tau_l_ns),
        "tau_l_parent_s": _float_record(tau_l_parent_s),
    }


def _float_value(record: object, label: str) -> float:
    mapped = _mapping(record, label)
    _exact_keys(mapped, {"hex", "value"}, label)
    value = _finite_scalar(mapped.get("value"), f"{label}.value")
    if mapped.get("hex") != value.hex():
        raise C6ScoreError(f"{label} decimal and hexadecimal forms disagree.")
    return value


def _expected_frozen_arrays(
    parent: _ParentContext,
) -> dict[str, np.ndarray]:
    c3 = parent.c3_arrays
    c5 = parent.c5_arrays
    expected: dict[str, np.ndarray] = {
        "parent_E_centers_ueV": np.asarray(c3["native_E_centers_ueV"]),
        "parent_dE_ueV": np.asarray(c3["native_dE_ueV"]),
        "parent_f": np.asarray(c3["projected_f"]),
        "parent_active_mask": np.asarray(c3["native_active_mask"]),
        "parent_cell_density": np.asarray(c3["native_cell_density_full"]),
        "parent_cell_weights_ueV": np.asarray(c3["native_cell_weights_full"]),
        "parent_projected_n_phonon": np.asarray(c3["projected_n_phonon"]),
        "parent_legacy_phonon_support_mask": np.asarray(
            c3["legacy_phonon_support_mask"]
        ),
        "parent_phonon_to_native_omega_index": np.asarray(
            c3["parent_phonon_to_native_omega_index"]
        ),
        "parent_qp_residual_s_inv": np.asarray(c5["c5sp_qp_residual_s_inv"]),
        "parent_phonon_residual_s_inv": np.asarray(
            c5["c5sp_phonon_residual_s_inv"]
        ),
        "qpsim_omega_ueV": np.asarray(c5["qpsim_omega_ueV"]),
        "qpsim_omega_idx_diff": np.asarray(c5["qpsim_omega_idx_diff"]),
        "qpsim_omega_idx_sum": np.asarray(c5["qpsim_omega_idx_sum"]),
        "qpsim_diff_sign": np.asarray(c5["qpsim_diff_sign"]),
    }
    for channel in ("scattering", "pair", "escape"):
        for field in _FIELDS:
            expected[f"parent_phonon_{channel}_{field}_s_inv"] = np.asarray(
                c3[
                    f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__"
                    f"{field}_s_inv"
                ]
            )
    return expected


def _expected_kernels(
    parent: _ParentContext,
    inputs: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    """Transcribe the phonon-side kernel formulas from the frozen coherences."""

    k_minus = np.asarray(parent.c3_arrays["native_K_minus_full"], dtype=np.float64)
    k_plus = np.asarray(parent.c3_arrays["native_K_plus_full"], dtype=np.float64)
    gap = _float_value(inputs["gap_ueV"], "operator_inputs.gap_ueV")
    tau_pb = _float_value(inputs["tau_0_pb_ns"], "operator_inputs.tau_0_pb_ns")
    scattering = (2.0 / (np.pi * gap * tau_pb)) * k_minus
    pair = k_plus / (np.pi * gap * tau_pb)
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
        raise C6ScoreError(f"{label} differs from its independent kernel formula.")
    return {
        "l1_absolute": _float_record(_fixed_sum(delta)),
        "linf_absolute": _float_record(float(np.max(delta, initial=0.0))),
        "maximum_rounding_bound_fraction": _float_record(maximum),
    }


def _clean_thermal_bose(omega: np.ndarray, k_b_uev_per_k: float) -> np.ndarray:
    """Fixed-order Bose occupation with the exact omega=0 bookkeeping zero."""

    result = np.zeros(omega.size, dtype=np.float64)
    for index in range(omega.size):
        value = float(omega[index])
        if value <= 0.0:
            continue
        result[index] = 1.0 / math.expm1(value / (k_b_uev_per_k * T_BATH_K))
    return result


def _clean_fermi(energy: np.ndarray, k_b_uev_per_k: float) -> np.ndarray:
    """Fixed-order Fermi occupation on the native center grid."""

    result = np.zeros(energy.size, dtype=np.float64)
    for index in range(energy.size):
        exponent = float(energy[index]) / (k_b_uev_per_k * T_BATH_K)
        result[index] = 1.0 / (math.exp(exponent) + 1.0)
    return result


def _clean_channel_contractions(
    f: np.ndarray,
    rho: np.ndarray,
    dE: np.ndarray,
    scattering_kernel: np.ndarray,
    pair_kernel: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fixed-order per-omega source/sink contractions in ns^-1.

    On the uniform unit grid, omega = |i - j| indexes scattering bins
    directly and omega = 321 + i + j indexes pair bins directly; both maps
    are enforced bit-exactly against the frozen C5 evidence before this
    transcription is compared.
    """

    one_minus = 1.0 - f
    emission = np.zeros(N_OMEGA, dtype=np.float64)
    absorption = np.zeros(N_OMEGA, dtype=np.float64)
    for w in range(1, N_QP):
        emission[w] = math.fsum(
            float(dE[j])
            * float(rho[j + w])
            * float(f[j + w])
            * float(scattering_kernel[j + w, j])
            * float(rho[j])
            * float(one_minus[j])
            for j in range(N_QP - w)
        )
        absorption[w] = math.fsum(
            float(dE[i + w])
            * float(rho[i])
            * float(f[i])
            * float(scattering_kernel[i, i + w])
            * float(rho[i + w])
            * float(one_minus[i + w])
            for i in range(N_QP - w)
        )
    recombination = np.zeros(N_OMEGA, dtype=np.float64)
    pair_breaking = np.zeros(N_OMEGA, dtype=np.float64)
    for w in range(321, N_OMEGA):
        pair_sum = w - 321
        start = max(0, pair_sum - (N_QP - 1))
        stop = min(N_QP - 1, pair_sum)
        recombination[w] = math.fsum(
            float(dE[pair_sum - i])
            * float(rho[i])
            * float(f[i])
            * float(pair_kernel[i, pair_sum - i])
            * float(rho[pair_sum - i])
            * float(f[pair_sum - i])
            for i in range(start, stop + 1)
        )
        pair_breaking[w] = math.fsum(
            float(dE[pair_sum - i])
            * float(rho[i])
            * float(one_minus[i])
            * float(pair_kernel[i, pair_sum - i])
            * float(rho[pair_sum - i])
            * float(one_minus[pair_sum - i])
            for i in range(start, stop + 1)
        )
    return {
        "absorption": absorption,
        "emission": emission,
        "pair_breaking": pair_breaking,
        "recombination": recombination,
    }


def _clean_kaplan_correction(
    rho: np.ndarray,
    dE: np.ndarray,
    pair_kernel: np.ndarray,
    inputs: dict[str, object],
) -> np.ndarray:
    """Transcribe qpsim's Kaplan S_+ pair-breaking quadrature correction.

    The correction rescales each complete-pair-interval omega bin from its
    midpoint quadrature to the analytic Kaplan total
    ``S_+(omega/Delta) / (pi tau_0_pb)`` with
    ``S_+(x) = x E(1 - 4/x^2)``.  Bins whose pair interval extends beyond
    the represented grid keep their truncated quadrature.
    """

    gap = _float_value(inputs["gap_ueV"], "operator_inputs.gap_ueV")
    tau_pb = _float_value(inputs["tau_0_pb_ns"], "operator_inputs.tau_0_pb_ns")
    discrete = np.zeros(N_OMEGA, dtype=np.float64)
    for w in range(321, N_OMEGA):
        pair_sum = w - 321
        start = max(0, pair_sum - (N_QP - 1))
        stop = min(N_QP - 1, pair_sum)
        discrete[w] = math.fsum(
            float(dE[pair_sum - i])
            * float(rho[i])
            * float(pair_kernel[i, pair_sum - i])
            * float(rho[pair_sum - i])
            for i in range(start, stop + 1)
        )
    lower_edge = 160.0
    upper_edge = 1800.0
    support_tol = 64.0 * _FLOAT_EPS * max(1.0, abs(lower_edge), upper_edge, gap)
    correction = np.ones(N_OMEGA, dtype=np.float64)
    if lower_edge > gap + support_tol:
        return correction
    for w in range(321, N_OMEGA):
        omega = float(w)
        if omega > upper_edge + gap + support_tol:
            continue
        if discrete[w] <= 0.0 or omega < 2.0 * gap:
            continue
        exact = (
            omega
            * float(ellipe(1.0 - 4.0 / ((omega / gap) * (omega / gap))))
            / gap
            / (np.pi * tau_pb)
        )
        if exact <= 0.0:
            continue
        correction[w] = exact / discrete[w]
    return correction


def _reduction_rounding_check(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    gamma: float = _REDUCTION_GAMMA,
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
        bound = gamma * scale
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


def _expected_parent_bindings(parent: _ParentContext) -> dict[str, object]:
    inherited = _mapping(
        parent.c5_score.get("parent_bindings"),
        "accepted C5 parent bindings",
    )
    return {
        "c2_raw_manifest_sha256": inherited.get("c2_raw_manifest_sha256"),
        "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
        "c3_raw_manifest_sha256": parent.c3_manifest_sha256,
        "c3_raw_schema": C3_RAW_SCHEMA,
        "c4_raw_manifest_sha256": parent.c4_manifest_sha256,
        "c4_raw_schema": C4_RAW_SCHEMA,
        "c5_raw_manifest_sha256": parent.c5_manifest_sha256,
        "c5_raw_schema": C5_RAW_SCHEMA,
        "c5_receipt_path": parent.c5_receipt_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c5_receipt_schema": C5_RECEIPT_SCHEMA,
        "c5_receipt_sha256": hashlib.sha256(parent.c5_receipt_bytes).hexdigest(),
        "c5_score_path": parent.c5_score_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c5_score_schema": C5_SCORE_SCHEMA,
        "c5_score_sha256": hashlib.sha256(parent.c5_score_bytes).hexdigest(),
        "c5_stage_id": PARENT_STAGE_ID,
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
        "parent_phonon_to_native_omega_index",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair", "escape")
            for field in _FIELDS
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )


def _check_raw_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    parent: _ParentContext,
    operator_inputs: dict[str, object],
) -> None:
    if metadata.get("schema") != RAW_SCHEMA:
        raise C6ScoreError("C6 raw metadata schema is invalid.")
    stage = _mapping(metadata.get("stage"), "C6 raw stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
    }:
        raise C6ScoreError("C6 raw stage metadata is invalid.")
    if not _json_value_bit_exact(metadata.get("operator_inputs"), operator_inputs):
        raise C6ScoreError("C6 raw operator inputs are stale or incomplete.")
    if not _json_value_bit_exact(
        metadata.get("parent_bindings"),
        _expected_parent_bindings(parent),
    ):
        raise C6ScoreError("C6 raw parent bindings are stale or incomplete.")
    descriptors = _mapping(metadata.get("array_descriptors"), "C6 descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C6ScoreError("C6 raw descriptor metadata is stale.")
    sources = _mapping(metadata.get("sources"), "C6 raw sources")
    source_binding = _mapping(metadata.get("source_binding"), "C6 source binding")
    if not _json_value_bit_exact(sources, _RAW_SOURCE_HASHES_AT_IMPORT):
        raise C6ScoreError("C6 raw source closure is incomplete, extra, or stale.")
    for relative, digest in sources.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or "\\" in relative
            or ".." in Path(relative).parts
        ):
            raise C6ScoreError(f"Unsafe C6 source path {relative!r}.")
        _sha256(digest, f"C6 source {relative}")
        source_path = REPOSITORY_ROOT / relative
        if hashlib.sha256(canonical_source_bytes(source_path)).hexdigest() != digest:
            raise C6ScoreError(f"C6 raw source binding is stale: {relative}.")
    if not _json_value_bit_exact(source_binding, _EXPECTED_SOURCE_BINDING):
        raise C6ScoreError("C6 raw source_binding is invalid.")
    frozen = _mapping(metadata.get("frozen_inputs"), "C6 frozen inputs")
    expected_c3_descriptors = {
        name: _array_descriptor(parent.c3_arrays[name])
        for name in _c3_frozen_names()
    }
    expected_c5_descriptors = {
        name: _array_descriptor(value)
        for name, value in sorted(parent.c5_arrays.items())
    }
    if not _json_value_bit_exact(
        frozen.get("c3_descriptors"),
        expected_c3_descriptors,
    ) or not _json_value_bit_exact(
        frozen.get("c5_descriptors"),
        expected_c5_descriptors,
    ):
        raise C6ScoreError("C6 frozen-input descriptors are stale or incomplete.")
    if (
        frozen.get("c3_mutation_check_after_operator") is not True
        or frozen.get("c5_mutation_check_after_operator") is not True
    ):
        raise C6ScoreError("C6 frozen-input mutation checks are missing.")
    _validate_runtime_record(metadata.get("runtime"), "C6 raw runtime")


def _validate_exact_derived_arrays(
    arrays: dict[str, np.ndarray],
) -> None:
    n_ph = arrays["parent_projected_n_phonon"]
    n_th = arrays["qpsim_thermal_n_ph"]
    support_idx = arrays["parent_phonon_to_native_omega_index"]
    affine = {
        "scattering": (
            arrays["qpsim_scattering_a_ns_inv"],
            arrays["qpsim_scattering_b_ns_inv"],
        ),
        "pair": (
            arrays["qpsim_pair_a_ns_inv"],
            arrays["qpsim_pair_b_ns_inv"],
        ),
        "pair_control": (
            arrays["qpsim_pair_control_a_ns_inv"],
            arrays["qpsim_pair_control_b_ns_inv"],
        ),
    }
    for channel in _CHANNELS:
        if channel == "escape":
            expected_ns = {
                "gain": n_th / TAU_L_NS,
                "loss": n_ph / TAU_L_NS,
                "net": (n_th - n_ph) / TAU_L_NS,
            }
        else:
            a, b = affine[channel]
            expected_ns = {
                "gain": a * (1.0 + n_ph),
                "loss": (a - b) * n_ph,
                "net": a + b * n_ph,
            }
        for field in _FIELDS:
            ns_name = f"qpsim_phonon_{channel}_{field}_ns_inv"
            s_name = f"qpsim_phonon_{channel}_{field}_s_inv"
            delta_name = f"phonon_{channel}_delta_{field}_s_inv"
            if not _array_bit_exact(
                _positive_zero_copy(expected_ns[field]),
                arrays[ns_name],
            ):
                raise C6ScoreError(f"C6 raw derived array {ns_name!r} is inconsistent.")
            if not _array_bit_exact(
                _positive_zero_copy(arrays[ns_name] / SECONDS_PER_NS),
                arrays[s_name],
            ):
                raise C6ScoreError(f"C6 raw derived array {s_name!r} is inconsistent.")
            parent_channel = "pair" if channel == "pair_control" else channel
            expected_delta = (
                arrays[s_name][support_idx]
                - arrays[f"parent_phonon_{parent_channel}_{field}_s_inv"]
            )
            if not _array_bit_exact(
                _positive_zero_copy(expected_delta),
                arrays[delta_name],
            ):
                raise C6ScoreError(f"C6 raw delta {delta_name!r} is inconsistent.")
    for name, net in (
        ("qpsim_db_scattering_net_ns_inv", None),
        ("qpsim_db_pair_net_ns_inv", None),
    ):
        del net
        prefix = name.removesuffix("_net_ns_inv")
        expected_db_net = (
            arrays[f"{prefix}_a_ns_inv"]
            + arrays[f"{prefix}_b_ns_inv"] * n_th
        )
        if not _array_bit_exact(_positive_zero_copy(expected_db_net), arrays[name]):
            raise C6ScoreError(f"C6 raw derived array {name!r} is inconsistent.")
    if not np.array_equal(
        arrays["qpsim_combined_a_ns_inv"],
        arrays["qpsim_scattering_a_ns_inv"] + arrays["qpsim_pair_a_ns_inv"],
    ):
        raise C6ScoreError(
            "C6 combined source coefficient is not the bit-exact channel sum."
        )
    combined_split = (
        arrays["qpsim_scattering_b_ns_inv"] + arrays["qpsim_pair_b_ns_inv"]
    ) - arrays["qpsim_combined_b_ns_inv"]
    combined_bound = 4.0 * _FLOAT_EPS * (
        np.abs(arrays["qpsim_combined_b_ns_inv"])
        + np.abs(arrays["qpsim_scattering_b_ns_inv"])
        + np.abs(arrays["qpsim_pair_b_ns_inv"])
    )
    if np.any(np.abs(combined_split) > combined_bound):
        raise C6ScoreError(
            "C6 combined sink coefficient differs from its channel sum beyond "
            "association-order rounding."
        )

    inv_tau = 1.0 / TAU_L_NS
    expected_balance = np.zeros(N_OMEGA, dtype=np.float64)
    a_all = arrays["qpsim_combined_a_ns_inv"]
    b_all = arrays["qpsim_combined_b_ns_inv"]
    for index in range(N_OMEGA):
        qp_terms = math.fma(
            float(b_all[index]),
            float(n_ph[index]),
            float(a_all[index]),
        )
        expected_balance[index] = math.fma(
            inv_tau,
            float(n_th[index]) - float(n_ph[index]),
            qp_terms,
        )
    if not _array_bit_exact(
        _positive_zero_copy(expected_balance),
        arrays["qpsim_balance_residual_ns_inv"],
    ):
        raise C6ScoreError("C6 public balance residual is not the declared FMA form.")

    scatter_delta = arrays["phonon_scattering_delta_net_s_inv"]
    pair_delta = arrays["phonon_pair_delta_net_s_inv"]
    pair_control_delta = arrays["phonon_pair_control_delta_net_s_inv"]
    escape_delta = arrays["phonon_escape_delta_net_s_inv"]
    parent_residual = arrays["parent_phonon_residual_s_inv"]
    expected_residuals = {
        "c6s_phonon_residual_s_inv": parent_residual + scatter_delta,
        "c6p_phonon_residual_s_inv": parent_residual + pair_delta,
        "c6p0_phonon_residual_s_inv": parent_residual + pair_control_delta,
        "c6e_phonon_residual_s_inv": parent_residual + escape_delta,
        "c6spe_phonon_residual_s_inv": (
            parent_residual + scatter_delta + pair_delta + escape_delta
        ),
        "c6spe0_phonon_residual_s_inv": (
            parent_residual + scatter_delta + pair_control_delta + escape_delta
        ),
        "c6_qp_residual_s_inv": arrays["parent_qp_residual_s_inv"],
    }
    for name, value in expected_residuals.items():
        if not _array_bit_exact(_positive_zero_copy(value), arrays[name]):
            raise C6ScoreError(f"C6 raw hybrid array {name!r} is inconsistent.")


def _map_identity_record(
    arrays: dict[str, np.ndarray],
) -> dict[str, object]:
    levels = np.arange(N_QP, dtype=np.int64)
    expected = {
        "qpsim_omega_ueV": np.arange(N_OMEGA, dtype=np.float64),
        "qpsim_omega_idx_diff": np.abs(levels[:, None] - levels[None, :]),
        # E_i = 160.5+i micro-eV, so E_i+E_j = 321+i+j micro-eV.
        "qpsim_omega_idx_sum": 321 + levels[:, None] + levels[None, :],
        "qpsim_diff_sign": np.sign(
            levels[:, None] - levels[None, :]
        ).astype(np.int8),
    }
    fields: dict[str, object] = {}
    for name, value in expected.items():
        if not _array_bit_exact(arrays[name], value):
            raise C6ScoreError(f"C6 frozen map array {name!r} changed.")
        fields[name] = {
            "bit_exact": True,
            "descriptor": _array_descriptor(arrays[name]),
        }
    fields["contract"] = (
        "The producer may call qpsim's public frequency-map helper, but C6 "
        "accepts it only after exact identity to the already-frozen C5 "
        "center labels: diff=|i-j| and sum=321+i+j on omega=0..3599."
    )
    return fields


def _check_raw_contract_sections(metadata: dict[str, Any]) -> None:
    """Structurally validate the C6 raw metadata contract sections.

    The complete metadata bytes are transitively authenticated by the pinned
    raw-manifest digest at canonical-score validation; this check enforces
    section closure and the load-bearing numeric limits.
    """

    _exact_keys(
        _mapping(metadata.get("balance_certification"), "C6 balance certification"),
        {"certified_backward_error", "contract", "raw_backward_error"},
        "C6 balance certification",
    )
    _float_value(
        metadata["balance_certification"]["raw_backward_error"],
        "C6 balance raw backward error",
    )
    _float_value(
        metadata["balance_certification"]["certified_backward_error"],
        "C6 balance certified backward error",
    )
    _exact_keys(
        _mapping(metadata.get("bookkeeping_contract"), "C6 bookkeeping contract"),
        {
            "affine_channel_decomposition",
            "combined_evaluation",
            "escape_form",
            "kaplan_pair_correction",
        },
        "C6 bookkeeping contract",
    )
    _exact_keys(
        _mapping(metadata.get("comparison_contract"), "C6 comparison contract"),
        {
            "candidate",
            "escape_comparison",
            "parent",
            "parent_qp",
            "public_arithmetic",
        },
        "C6 comparison contract",
    )
    locality = _mapping(metadata.get("component_locality"), "C6 component locality")
    _exact_keys(
        locality,
        {
            "c6e",
            "c6p",
            "c6p0",
            "c6s",
            "c6spe",
            "c6spe0",
            "changed_arrays",
            "inherited_arrays",
            "phonon_residual_updates",
            "qp_residual_bit_exact",
        },
        "C6 component locality",
    )
    if locality.get("qp_residual_bit_exact") is not True:
        raise C6ScoreError("C6 raw locality does not declare a bit-exact QP residual.")
    _exact_keys(
        _mapping(metadata.get("coordinate_contract"), "C6 coordinate contract"),
        {
            "active_child_indices",
            "author_support_window",
            "frequency_map",
            "guard_child_indices",
            "native_cell_count",
            "native_omega_count",
            "omega_zero_bin",
        },
        "C6 coordinate contract",
    )
    if (
        metadata["coordinate_contract"].get("native_cell_count") != N_QP
        or metadata["coordinate_contract"].get("native_omega_count") != N_OMEGA
    ):
        raise C6ScoreError("C6 coordinate counts are invalid.")
    detailed = _mapping(metadata.get("detailed_balance"), "C6 detailed balance")
    _exact_keys(
        detailed,
        {"contract", "limit_relative", "pair", "scattering"},
        "C6 detailed balance",
    )
    if _float_value(
        detailed["limit_relative"],
        "C6 detailed-balance limit",
    ) != _DETAILED_BALANCE_LIMIT:
        raise C6ScoreError("C6 detailed-balance limit is invalid.")
    for channel in ("scattering", "pair"):
        record = _mapping(detailed[channel], f"C6 detailed balance {channel}")
        _exact_keys(
            record,
            {
                "imbalance_l1_ns_inv",
                "relative_imbalance",
                "turnover_l1_ns_inv",
            },
            f"C6 detailed balance {channel}",
        )
        if _float_value(
            record["relative_imbalance"],
            f"C6 detailed balance {channel} relative",
        ) > _DETAILED_BALANCE_LIMIT:
            raise C6ScoreError(
                f"C6 raw detailed balance for {channel} exceeds its limit."
            )
    extension = _mapping(metadata.get("extension_policy"), "C6 extension policy")
    _exact_keys(
        extension,
        {"statement", *(channel for channel in _CHANNELS)},
        "C6 extension policy",
    )
    for channel in _CHANNELS:
        record = _mapping(extension[channel], f"C6 extension {channel}")
        _exact_keys(record, set(_FIELDS), f"C6 extension {channel}")
        for field in _FIELDS:
            totals = _mapping(
                record[field],
                f"C6 extension {channel}.{field}",
            )
            _exact_keys(
                totals,
                {"l1_s_inv", "linf_s_inv", "nonzero_bins"},
                f"C6 extension {channel}.{field}",
            )
            _float_value(totals["l1_s_inv"], "C6 extension L1")
            _float_value(totals["linf_s_inv"], "C6 extension Linf")
            _strict_int(totals["nonzero_bins"], "C6 extension count", minimum=0)
    _exact_keys(
        _mapping(metadata.get("limitations"), "C6 limitations"),
        {"scope", "statement"},
        "C6 limitations",
    )
    _exact_keys(
        _mapping(metadata.get("units"), "C6 units"),
        {
            "affine_coefficient_arrays",
            "comparison_arrays",
            "kernel_arrays",
            "public_native_arrays",
            "public_return_contract",
        },
        "C6 units",
    )


def build_c6_score(
    c6_bundle_dir: Path,
    *,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Independently replay and score one formal frozen-state C6 bundle."""

    _assert_source_snapshots()
    c6_root, c6_directory_state = _directory_state(
        c6_bundle_dir,
        "selected C6 raw bundle",
    )
    parent = _accept_parent(
        c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    raw_metadata, arrays, raw_manifest_sha = load_c6_raw_bundle(c6_root)
    operator_inputs = _operator_inputs(parent.c3_metadata)
    _check_raw_metadata(
        raw_metadata,
        arrays,
        parent=parent,
        operator_inputs=operator_inputs,
    )
    _check_raw_contract_sections(raw_metadata)

    frozen = _expected_frozen_arrays(parent)
    for name, expected in frozen.items():
        if not _array_bit_exact(_positive_zero_copy(expected), arrays[name]):
            raise C6ScoreError(f"C6 frozen parent array {name!r} changed.")
    active = arrays["parent_active_mask"]
    if (
        int(np.count_nonzero(active)) != N_ACTIVE
        or np.any(active[:ACTIVE_START])
        or not np.all(active[ACTIVE_START:])
    ):
        raise C6ScoreError("C6 active support is not the accepted C3 support.")
    if np.any(arrays["parent_f"][:ACTIVE_START] != 0.0):
        raise C6ScoreError("C6 guard-cell occupations are not canonical positive zero.")
    expected_support = np.zeros(N_OMEGA, dtype=bool)
    expected_support[1 : AUTHOR_OMEGA_STOP + 1] = True
    if not _array_bit_exact(
        arrays["parent_legacy_phonon_support_mask"],
        expected_support,
    ):
        raise C6ScoreError("C6 inherited author phonon support changed.")
    support_idx = arrays["parent_phonon_to_native_omega_index"]
    if not np.array_equal(
        support_idx,
        np.arange(1, AUTHOR_OMEGA_STOP + 1, dtype=np.int64),
    ):
        raise C6ScoreError("C6 author support index map is not the identity window.")
    if np.any(
        arrays["parent_projected_n_phonon"][~expected_support] != 0.0
    ):
        raise C6ScoreError("C6 projected phonons populate placeholder-only support.")

    map_record = _map_identity_record(arrays)
    scattering_kernel, pair_kernel = _expected_kernels(parent, operator_inputs)
    kernel_records = {
        "scattering": _kernel_rounding_check(
            arrays["qpsim_phonon_scattering_kernel_ns_inv_ueV_inv"],
            scattering_kernel,
            "phonon-side scattering kernel",
        ),
        "pair": _kernel_rounding_check(
            arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"],
            pair_kernel,
            "phonon-side pair kernel",
        ),
    }
    _validate_exact_derived_arrays(arrays)

    k_b_uev_per_k = _float_value(
        operator_inputs["kB_ueV_per_K"],
        "operator_inputs.kB_ueV_per_K",
    )
    reduction_checks: list[bool] = []
    # A few-ulp difference in the exponent argument is amplified by the
    # exponential, so the occupation bounds scale with (1 + x) where
    # x = omega/(k_B T) or E/(k_B T).
    clean_n_th = _clean_thermal_bose(arrays["qpsim_omega_ueV"], k_b_uev_per_k)
    bose_exponent = arrays["qpsim_omega_ueV"] / (k_b_uev_per_k * T_BATH_K)
    thermal_check = _reduction_rounding_check(
        arrays["qpsim_thermal_n_ph"],
        clean_n_th,
        absolute_bound=32.0
        * _FLOAT_EPS
        * (1.0 + bose_exponent)
        * (np.abs(arrays["qpsim_thermal_n_ph"]) + np.abs(clean_n_th)),
    )
    reduction_checks.append(bool(thermal_check["within_rounding_bound"]))
    clean_db_f = _clean_fermi(arrays["parent_E_centers_ueV"], k_b_uev_per_k)
    fermi_exponent = arrays["parent_E_centers_ueV"] / (
        k_b_uev_per_k * T_BATH_K
    )
    db_f_check = _reduction_rounding_check(
        arrays["qpsim_db_f"],
        clean_db_f,
        absolute_bound=32.0
        * _FLOAT_EPS
        * (1.0 + fermi_exponent)
        * (np.abs(arrays["qpsim_db_f"]) + np.abs(clean_db_f)),
    )
    reduction_checks.append(bool(db_f_check["within_rounding_bound"]))

    rho = arrays["parent_cell_density"]
    dE = arrays["parent_dE_ueV"]
    kernel_pair_retained = arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"]
    kernel_scatter_retained = arrays[
        "qpsim_phonon_scattering_kernel_ns_inv_ueV_inv"
    ]
    clean = _clean_channel_contractions(
        arrays["parent_f"],
        rho,
        dE,
        kernel_scatter_retained,
        kernel_pair_retained,
    )
    correction = _clean_kaplan_correction(
        rho,
        dE,
        kernel_pair_retained,
        operator_inputs,
    )
    clean_affine = {
        "scattering": (
            clean["emission"],
            clean["emission"] - clean["absorption"],
        ),
        "pair": (
            clean["recombination"] * correction,
            clean["recombination"] * correction
            - clean["pair_breaking"] * correction,
        ),
        "pair_control": (
            clean["recombination"],
            clean["recombination"] - clean["pair_breaking"],
        ),
    }
    affine_retained = {
        "scattering": (
            arrays["qpsim_scattering_a_ns_inv"],
            arrays["qpsim_scattering_b_ns_inv"],
        ),
        "pair": (
            arrays["qpsim_pair_a_ns_inv"],
            arrays["qpsim_pair_b_ns_inv"],
        ),
        "pair_control": (
            arrays["qpsim_pair_control_a_ns_inv"],
            arrays["qpsim_pair_control_b_ns_inv"],
        ),
    }
    affine_reduction: dict[str, dict[str, object]] = {}
    for channel, (clean_a, clean_b) in clean_affine.items():
        gamma = _CORRECTION_GAMMA if channel == "pair" else _REDUCTION_GAMMA
        retained_a, retained_b = affine_retained[channel]
        check_a = _reduction_rounding_check(retained_a, clean_a, gamma=gamma)
        check_b = _reduction_rounding_check(retained_b, clean_b, gamma=gamma)
        reduction_checks.append(bool(check_a["within_rounding_bound"]))
        reduction_checks.append(bool(check_b["within_rounding_bound"]))
        affine_reduction[channel] = {"a": check_a, "b": check_b}

    clean_db = _clean_channel_contractions(
        arrays["qpsim_db_f"],
        rho,
        dE,
        kernel_scatter_retained,
        kernel_pair_retained,
    )
    db_clean_affine = {
        "scattering": (
            clean_db["emission"],
            clean_db["emission"] - clean_db["absorption"],
        ),
        "pair": (
            clean_db["recombination"] * correction,
            clean_db["recombination"] * correction
            - clean_db["pair_breaking"] * correction,
        ),
    }
    db_reduction: dict[str, dict[str, object]] = {}
    detailed_balance_verifier: dict[str, Any] = {}
    detailed_balance_flags: dict[str, bool] = {}
    n_th = arrays["qpsim_thermal_n_ph"]
    for channel, (clean_a, clean_b) in db_clean_affine.items():
        gamma = _CORRECTION_GAMMA if channel == "pair" else _REDUCTION_GAMMA
        retained_a = arrays[f"qpsim_db_{channel}_a_ns_inv"]
        retained_b = arrays[f"qpsim_db_{channel}_b_ns_inv"]
        check_a = _reduction_rounding_check(retained_a, clean_a, gamma=gamma)
        check_b = _reduction_rounding_check(retained_b, clean_b, gamma=gamma)
        reduction_checks.append(bool(check_a["within_rounding_bound"]))
        reduction_checks.append(bool(check_b["within_rounding_bound"]))
        db_reduction[channel] = {"a": check_a, "b": check_b}
        net = arrays[f"qpsim_db_{channel}_net_ns_inv"]
        gain = retained_a * (1.0 + n_th)
        loss = (retained_a - retained_b) * n_th
        imbalance = _fixed_sum(np.abs(net))
        turnover = max(
            _fixed_sum(np.abs(gain)) + _fixed_sum(np.abs(loss)),
            float(np.finfo(np.float64).tiny),
        )
        relative = imbalance / turnover
        detailed_balance_flags[channel] = relative <= _DETAILED_BALANCE_LIMIT
        detailed_balance_verifier[channel] = {
            "imbalance_l1_ns_inv": _float_record(imbalance),
            "relative_imbalance": _float_record(relative),
            "turnover_l1_ns_inv": _float_record(turnover),
        }
    detailed_balance_verifier["limit_relative"] = _float_record(
        _DETAILED_BALANCE_LIMIT
    )
    detailed_balance_verifier["reduction"] = db_reduction
    detailed_balance_verifier["thermal_control_inputs"] = {
        "db_f_vs_clean_fermi": db_f_check,
        "thermal_n_ph_vs_clean_bose": thermal_check,
    }

    channel_comparison: dict[str, Any] = {}
    parity_relatives: dict[tuple[str, str], float] = {}
    for channel in _CHANNELS:
        parent_channel = "pair" if channel == "pair_control" else channel
        fields: dict[str, object] = {}
        for field in _FIELDS:
            comparison = _operator_comparison(
                arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][support_idx],
                arrays[f"parent_phonon_{parent_channel}_{field}_s_inv"],
            )
            parity_relatives[(channel, field)] = _float_value(
                comparison["symmetric_relative_l1"],
                f"{channel} {field} parity",
            )
            fields[field] = comparison
        record: dict[str, object] = {"parent_operator": fields}
        if channel != "escape":
            record["public_vs_clean_reduction"] = affine_reduction[channel]
        else:
            support_gain = arrays["qpsim_phonon_escape_gain_s_inv"][support_idx]
            support_loss = arrays["qpsim_phonon_escape_loss_s_inv"][support_idx]
            escape_bound = (
                _ESCAPE_NET_ELEMENTWISE_BUDGET
                * _FLOAT_EPS
                * (np.abs(support_gain) + np.abs(support_loss))
            )
            escape_delta = np.abs(
                arrays["phonon_escape_delta_net_s_inv"]
            )
            escape_fraction = escape_delta / np.maximum(
                escape_bound,
                np.finfo(np.float64).tiny,
            )
            escape_maximum = float(np.max(escape_fraction, initial=0.0))
            record["net_elementwise_rounding"] = {
                "budget_eps_of_gain_plus_loss": _float_record(
                    _ESCAPE_NET_ELEMENTWISE_BUDGET
                ),
                "maximum_bound_fraction": _float_record(escape_maximum),
                "within_bound": escape_maximum <= 1.0,
            }
        channel_comparison[channel] = record
    channel_comparison["scattering"]["kernel_formula"] = kernel_records[
        "scattering"
    ]
    channel_comparison["pair"]["kernel_formula"] = kernel_records["pair"]

    formal_residual_comparison = _operator_comparison(
        arrays["c6spe_phonon_residual_s_inv"],
        arrays["parent_phonon_residual_s_inv"],
    )
    control_residual_comparison = _operator_comparison(
        arrays["c6spe0_phonon_residual_s_inv"],
        arrays["parent_phonon_residual_s_inv"],
    )
    channel_comparison["formal_residual"] = {
        "c6spe0_vs_parent": control_residual_comparison,
        "c6spe_vs_parent": formal_residual_comparison,
        "statement": (
            "The c6spe-versus-parent difference is the recorded qpsim "
            "endpoint change; the correction-off control shows every "
            "non-Kaplan substitution is roundoff-equivalent."
        ),
    }

    correction_support = correction[support_idx]
    away_from_unity = int(
        np.count_nonzero(np.abs(correction_support - 1.0) > 1.0e-6)
    )
    kaplan_record = {
        "clean_correction_max": _float_record(
            float(np.max(correction_support, initial=1.0))
        ),
        "clean_correction_min": _float_record(
            float(np.min(correction_support, initial=1.0))
        ),
        "statement": (
            "Public qpsim rescales each complete-pair-interval pair bin to "
            "the analytic Kaplan S_+ total; the author equation keeps plain "
            "midpoint quadrature. The correction-off control isolates this "
            "as the only pair-channel difference beyond roundoff."
        ),
        "support_bins_with_relative_change_above_1e_minus_6": away_from_unity,
    }

    outside = ~expected_support
    extension_verifier: dict[str, Any] = {}
    for channel in _CHANNELS:
        extension_verifier[channel] = {}
        for field in _FIELDS:
            values = arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][outside]
            magnitude = np.abs(values)
            extension_verifier[channel][field] = {
                "l1_s_inv": _float_record(_fixed_sum(magnitude)),
                "linf_s_inv": _float_record(
                    float(np.max(magnitude, initial=0.0))
                ),
                "nonzero_bins": int(np.count_nonzero(magnitude)),
            }
    scattering_confined = all(
        not np.any(
            arrays[f"qpsim_phonon_scattering_{field}_s_inv"][outside] != 0.0
        )
        for field in _FIELDS
    )
    omega_zero_clean = all(
        float(arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][0]) == 0.0
        for channel in _CHANNELS
        for field in _FIELDS
    )

    all_checks = {
        "all_public_reductions_within_predeclared_rounding_bound": all(
            reduction_checks
        ),
        "detailed_balance_pair_within_limit": detailed_balance_flags["pair"],
        "detailed_balance_scattering_within_limit": detailed_balance_flags[
            "scattering"
        ],
        "escape_net_within_elementwise_rounding_bound": bool(
            channel_comparison["escape"]["net_elementwise_rounding"][
                "within_bound"
            ]
        ),
        "escape_physical_gain_matches_parent": (
            parity_relatives[("escape", "gain")] <= _BUCKET_PARITY_LIMIT
        ),
        "escape_physical_loss_matches_parent": (
            parity_relatives[("escape", "loss")] <= _BUCKET_PARITY_LIMIT
        ),
        "frozen_qp_residual_bit_exact": _array_bit_exact(
            arrays["c6_qp_residual_s_inv"],
            arrays["parent_qp_residual_s_inv"],
        ),
        "omega_zero_bin_exactly_zero": omega_zero_clean,
        "pair_control_physical_gain_matches_parent": (
            parity_relatives[("pair_control", "gain")] <= _BUCKET_PARITY_LIMIT
        ),
        "pair_control_physical_loss_matches_parent": (
            parity_relatives[("pair_control", "loss")] <= _BUCKET_PARITY_LIMIT
        ),
        "pair_control_physical_net_matches_parent": (
            parity_relatives[("pair_control", "net")] <= _NET_PARITY_LIMIT
        ),
        "scattering_confined_to_author_support": scattering_confined,
        "scattering_physical_gain_matches_parent": (
            parity_relatives[("scattering", "gain")] <= _BUCKET_PARITY_LIMIT
        ),
        "scattering_physical_loss_matches_parent": (
            parity_relatives[("scattering", "loss")] <= _BUCKET_PARITY_LIMIT
        ),
        "scattering_physical_net_matches_parent": (
            parity_relatives[("scattering", "net")] <= _NET_PARITY_LIMIT
        ),
    }
    if not all(all_checks.values()):
        failed = sorted(name for name, passed in all_checks.items() if not passed)
        raise C6ScoreError(f"C6 acceptance checks failed: {failed}.")

    score: dict[str, Any] = {
        "acceptance": {
            "all_passed": True,
            "checks": all_checks,
            "limits": {
                "detailed_balance_relative": _float_record(
                    _DETAILED_BALANCE_LIMIT
                ),
                "escape_net_elementwise_eps_budget": _float_record(
                    _ESCAPE_NET_ELEMENTWISE_BUDGET
                ),
                "like_for_like_bucket_symmetric_relative_l1": _float_record(
                    _BUCKET_PARITY_LIMIT
                ),
                "unchanged_channel_net_symmetric_relative_l1": _float_record(
                    _NET_PARITY_LIMIT
                ),
            },
            "ungated_recorded_differences": {
                "pair_public_vs_parent": (
                    "the Kaplan-corrected public pair channel is the "
                    "documented endpoint semantic change and is recorded, "
                    "not gated, against the author parent"
                ),
                "support_extension": (
                    "out-of-support bins are recorded totals, not gates; "
                    "only the scattering channel must stay confined"
                ),
            },
        },
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "balance_certification": {
            "producer": dict(
                _mapping(
                    raw_metadata["balance_certification"],
                    "balance certification",
                )
            ),
            "residual_form": (
                "per-bin FMA a + b*n_ph + (n_th - n_ph)/tau_l, revalidated "
                "bit-exactly against the retained residual array"
            ),
        },
        "bookkeeping": {
            "contract": dict(
                _mapping(raw_metadata["bookkeeping_contract"], "bookkeeping contract")
            ),
        },
        "channel_comparison": channel_comparison,
        "component_locality": {
            "c6_qp_residual": _array_descriptor(arrays["c6_qp_residual_s_inv"]),
            "c6e_phonon_residual": _array_descriptor(
                arrays["c6e_phonon_residual_s_inv"]
            ),
            "c6p0_phonon_residual": _array_descriptor(
                arrays["c6p0_phonon_residual_s_inv"]
            ),
            "c6p_phonon_residual": _array_descriptor(
                arrays["c6p_phonon_residual_s_inv"]
            ),
            "c6s_phonon_residual": _array_descriptor(
                arrays["c6s_phonon_residual_s_inv"]
            ),
            "c6spe0_phonon_residual": _array_descriptor(
                arrays["c6spe0_phonon_residual_s_inv"]
            ),
            "c6spe_phonon_residual": _array_descriptor(
                arrays["c6spe_phonon_residual_s_inv"]
            ),
            "frozen_parent_phonon_descriptors": {
                f"{channel}_{field}": _array_descriptor(
                    arrays[f"parent_phonon_{channel}_{field}_s_inv"]
                )
                for channel in ("scattering", "pair", "escape")
                for field in _FIELDS
            },
            "frozen_parent_phonon_residual": _array_descriptor(
                arrays["parent_phonon_residual_s_inv"]
            ),
            "raw_contract": dict(
                _mapping(raw_metadata["component_locality"], "component locality")
            ),
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
        "detailed_balance": {
            "producer": dict(
                _mapping(raw_metadata["detailed_balance"], "detailed balance")
            ),
            "verifier": detailed_balance_verifier,
        },
        "extension_policy": {
            "producer": dict(
                _mapping(raw_metadata["extension_policy"], "extension policy")
            ),
            "verifier_recomputation": extension_verifier,
        },
        "kaplan_correction": kaplan_record,
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
            "correction_gamma": _float_record(_CORRECTION_GAMMA),
            "correction_operation_budget": _CORRECTION_OPERATION_BUDGET,
            "float64_epsilon": _float_record(_FLOAT_EPS),
            "gamma": _float_record(_REDUCTION_GAMMA),
            "operation_budget_per_reduction": _REDUCTION_OPERATION_BUDGET,
        },
        "runtime": {
            "producer_public_array_generation": dict(
                _mapping(raw_metadata["runtime"], "producer runtime")
            ),
            "verifier_contract": dict(_VERIFIER_RUNTIME_CONTRACT),
        },
        "schema": SCHEMA,
        "source_binding": dict(
            _mapping(raw_metadata["source_binding"], "C6 source binding")
        ),
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
        c6_root,
        c6_directory_state,
        "selected C6 raw bundle",
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
    raw = _mapping(score.get("raw_bundle"), "C6 pinned raw bundle")
    if raw.get("manifest_sha256") != _EXPECTED_RAW_MANIFEST_SHA256:
        raise C6ScoreError("C6 score does not bind the accepted canonical raw manifest.")
    channels = _mapping(score.get("channel_comparison"), "C6 pinned channels")

    def parity(channel: str, field: str) -> float:
        record = _mapping(channels.get(channel), f"C6 pinned {channel}")
        parent_operator = _mapping(
            record.get("parent_operator"),
            f"C6 pinned {channel} parent",
        )
        return _float_value(
            _mapping(
                parent_operator.get(field),
                f"C6 pinned {channel} {field}",
            ).get("symmetric_relative_l1"),
            f"C6 pinned {channel} {field} relative",
        )

    formal = _mapping(channels.get("formal_residual"), "C6 pinned formal residual")
    detailed = _mapping(score.get("detailed_balance"), "C6 pinned detailed balance")
    verifier_db = _mapping(detailed.get("verifier"), "C6 pinned verifier balance")
    actual_metrics = {
        "c6spe0_residual_symmetric_relative_l1": _float_value(
            _mapping(
                formal.get("c6spe0_vs_parent"),
                "pinned control residual",
            ).get("symmetric_relative_l1"),
            "pinned control residual relative",
        ),
        "c6spe_residual_symmetric_relative_l1": _float_value(
            _mapping(
                formal.get("c6spe_vs_parent"),
                "pinned formal residual",
            ).get("symmetric_relative_l1"),
            "pinned formal residual relative",
        ),
        "detailed_balance_pair_relative": _float_value(
            _mapping(
                verifier_db.get("pair"),
                "pinned pair detailed balance",
            ).get("relative_imbalance"),
            "pinned pair detailed balance relative",
        ),
        "detailed_balance_scattering_relative": _float_value(
            _mapping(
                verifier_db.get("scattering"),
                "pinned scattering detailed balance",
            ).get("relative_imbalance"),
            "pinned scattering detailed balance relative",
        ),
        "pair_control_net_symmetric_relative_l1": parity("pair_control", "net"),
        "pair_public_net_symmetric_relative_l1": parity("pair", "net"),
        "scattering_net_symmetric_relative_l1": parity("scattering", "net"),
    }
    if not _json_value_bit_exact(actual_metrics, _EXPECTED_CANONICAL_METRICS):
        raise C6ScoreError("C6 canonical numerical metric pins do not match.")
    locality = _mapping(score.get("component_locality"), "C6 pinned locality")
    for name, expected_sha in _EXPECTED_RESIDUAL_NPY_SHA256.items():
        descriptor = _mapping(locality.get(name), f"C6 pinned locality.{name}")
        if descriptor.get("npy_sha256") != expected_sha:
            raise C6ScoreError(f"C6 canonical residual pin {name!r} does not match.")
    if _evidence_digest(score) != _EXPECTED_EVIDENCE_DIGEST:
        raise C6ScoreError("C6 canonical numeric/semantic evidence digest does not match.")


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
        raise C6ScoreError(f"{label}.dtype must be a string.")
    _sha256(descriptor.get("npy_sha256"), f"{label}.npy_sha256")
    if (
        not isinstance(shape, list)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in shape)
        or any(item < 0 for item in shape)
    ):
        raise C6ScoreError(f"{label}.shape is invalid.")
    if expected_dtype is not None and dtype != expected_dtype:
        raise C6ScoreError(f"{label}.dtype is invalid.")
    if expected_shape is not None and shape != list(expected_shape):
        raise C6ScoreError(f"{label}.shape is invalid.")


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
        _float_value(metric[field], f"{label}.{field}")
    if metric["within_rounding_bound"] is not True:
        raise C6ScoreError(f"{label}.within_rounding_bound must be True.")


def _validate_comparison_record(value: object, label: str) -> None:
    metric = _mapping(value, label)
    _exact_keys(
        metric,
        {"l1_absolute_s_inv", "linf_absolute_s_inv", "symmetric_relative_l1"},
        label,
    )
    for field in metric:
        _float_value(metric[field], f"{label}.{field}")


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
            raise C6ScoreError(f"{label}.{field} must be a nonempty string.")
    if runtime["byteorder"] not in {"little", "big"}:
        raise C6ScoreError(f"{label}.byteorder is invalid.")
    threads = _mapping(runtime["thread_environment"], f"{label}.thread_environment")
    if threads != _THREAD_ENVIRONMENT:
        raise C6ScoreError(f"{label}.thread_environment is invalid.")
    blas = _mapping(runtime["numpy_blas"], f"{label}.numpy_blas")
    _exact_keys(
        blas,
        {"found", "name", "openblas_configuration", "version"},
        f"{label}.numpy_blas",
    )
    if blas["found"] is not None and not isinstance(blas["found"], bool):
        raise C6ScoreError(f"{label}.numpy_blas.found must be Boolean or null.")
    for field in ("name", "openblas_configuration", "version"):
        if blas[field] is not None and not isinstance(blas[field], str):
            raise C6ScoreError(
                f"{label}.numpy_blas.{field} must be a string or null."
            )


def _validate_score_structure(score: dict[str, Any]) -> None:
    _exact_keys(score, _SCORE_KEYS, "C6 score")
    if score.get("schema") != SCHEMA:
        raise C6ScoreError("C6 score schema is unsupported.")
    stage = _mapping(score.get("stage"), "C6 score stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C6ScoreError("C6 score stage is invalid.")
    if not _json_value_bit_exact(score.get("sources"), _SOURCE_HASHES_AT_IMPORT):
        raise C6ScoreError("C6 score source closure is invalid.")
    runtime = _mapping(score.get("runtime"), "C6 score runtime")
    _exact_keys(
        runtime,
        {"producer_public_array_generation", "verifier_contract"},
        "C6 score runtime",
    )
    _validate_runtime_record(
        runtime.get("producer_public_array_generation"),
        "C6 score producer runtime",
    )
    if not _json_value_bit_exact(
        runtime.get("verifier_contract"),
        _VERIFIER_RUNTIME_CONTRACT,
    ):
        raise C6ScoreError("C6 score verifier runtime contract is invalid.")
    raw = _mapping(score.get("raw_bundle"), "C6 score raw bundle")
    _exact_keys(raw, {"manifest_sha256", "schema"}, "C6 score raw bundle")
    if raw.get("schema") != RAW_SCHEMA:
        raise C6ScoreError("C6 score raw schema is invalid.")
    _sha256(raw.get("manifest_sha256"), "C6 score raw manifest SHA-256")
    descriptors = _mapping(score.get("array_descriptors"), "C6 score descriptors")
    if set(descriptors) != _EXPECTED_ARRAY_NAMES:
        raise C6ScoreError("C6 score descriptor closure is invalid.")
    for name, (dtype, shape) in _ARRAY_SPECS.items():
        _validate_descriptor(
            descriptors[name],
            f"array_descriptors.{name}",
            expected_dtype=dtype,
            expected_shape=shape,
        )
    parent = _mapping(score.get("parent_bindings"), "C6 score parent bindings")
    expected_parent_keys = {
        "c2_raw_manifest_sha256",
        "c3_operator_stage_id",
        "c3_raw_manifest_sha256",
        "c3_raw_schema",
        "c4_raw_manifest_sha256",
        "c4_raw_schema",
        "c5_raw_manifest_sha256",
        "c5_raw_schema",
        "c5_receipt_path",
        "c5_receipt_schema",
        "c5_receipt_sha256",
        "c5_score_path",
        "c5_score_schema",
        "c5_score_sha256",
        "c5_stage_id",
    }
    _exact_keys(parent, expected_parent_keys, "C6 score parent bindings")
    if (
        parent.get("c5_raw_schema") != C5_RAW_SCHEMA
        or parent.get("c5_receipt_schema") != C5_RECEIPT_SCHEMA
        or parent.get("c5_score_schema") != C5_SCORE_SCHEMA
        or parent.get("c4_raw_schema") != C4_RAW_SCHEMA
        or parent.get("c3_raw_schema") != C3_RAW_SCHEMA
        or parent.get("c3_operator_stage_id") != PARENT_OPERATOR_STAGE_ID
        or parent.get("c5_stage_id") != PARENT_STAGE_ID
    ):
        raise C6ScoreError("C6 score parent schemas are invalid.")
    for field in (
        "c2_raw_manifest_sha256",
        "c3_raw_manifest_sha256",
        "c4_raw_manifest_sha256",
        "c5_raw_manifest_sha256",
        "c5_receipt_sha256",
        "c5_score_sha256",
    ):
        _sha256(parent.get(field), f"C6 score parent_bindings.{field}")
    for field in ("c5_score_path", "c5_receipt_path"):
        value = parent.get(field)
        if (
            not isinstance(value, str)
            or value.startswith("/")
            or "\\" in value
            or ".." in Path(value).parts
        ):
            raise C6ScoreError(f"C6 score parent path {field} is unsafe.")

    rounding = _mapping(score.get("rounding_contract"), "C6 rounding contract")
    _exact_keys(
        rounding,
        {
            "comparison",
            "correction_gamma",
            "correction_operation_budget",
            "float64_epsilon",
            "gamma",
            "operation_budget_per_reduction",
        },
        "C6 rounding contract",
    )
    if rounding.get("comparison") != (
        "Retained public arrays are manifest-byte-bound. Independent "
        "science uses fixed-order math.fsum and an elementwise gamma "
        "bound; cross-platform BLAS last bits are not required to match."
    ):
        raise C6ScoreError("C6 rounding comparison statement is invalid.")
    if (
        _float_value(rounding["float64_epsilon"], "rounding epsilon")
        != _FLOAT_EPS
        or _float_value(rounding["gamma"], "rounding gamma") != _REDUCTION_GAMMA
        or _float_value(
            rounding["correction_gamma"],
            "correction gamma",
        )
        != _CORRECTION_GAMMA
        or rounding.get("operation_budget_per_reduction")
        != _REDUCTION_OPERATION_BUDGET
        or rounding.get("correction_operation_budget")
        != _CORRECTION_OPERATION_BUDGET
    ):
        raise C6ScoreError("C6 rounding constants are invalid.")

    channels = _mapping(score.get("channel_comparison"), "C6 channel comparison")
    _exact_keys(
        channels,
        {"escape", "formal_residual", "pair", "pair_control", "scattering"},
        "channel comparison",
    )
    for channel in _CHANNELS:
        record = _mapping(channels[channel], f"channel_comparison.{channel}")
        expected = {"parent_operator"}
        if channel == "escape":
            expected.add("net_elementwise_rounding")
        else:
            expected.add("public_vs_clean_reduction")
        if channel in {"scattering", "pair"}:
            expected.add("kernel_formula")
        _exact_keys(record, expected, f"channel_comparison.{channel}")
        parent_operator = _mapping(
            record["parent_operator"],
            f"channel_comparison.{channel}.parent_operator",
        )
        _exact_keys(
            parent_operator,
            set(_FIELDS),
            f"channel_comparison.{channel}.parent_operator",
        )
        for field in _FIELDS:
            _validate_comparison_record(
                parent_operator[field],
                f"channel_comparison.{channel}.parent_operator.{field}",
            )
        if channel != "escape":
            reductions = _mapping(
                record["public_vs_clean_reduction"],
                f"channel_comparison.{channel}.public_vs_clean_reduction",
            )
            _exact_keys(
                reductions,
                {"a", "b"},
                f"channel_comparison.{channel}.public_vs_clean_reduction",
            )
            for field in ("a", "b"):
                _validate_metric_record(
                    reductions[field],
                    f"channel_comparison.{channel}."
                    f"public_vs_clean_reduction.{field}",
                )
        else:
            escape_record = _mapping(
                record["net_elementwise_rounding"],
                "escape net elementwise record",
            )
            _exact_keys(
                escape_record,
                {
                    "budget_eps_of_gain_plus_loss",
                    "maximum_bound_fraction",
                    "within_bound",
                },
                "escape net elementwise record",
            )
            if escape_record.get("within_bound") is not True:
                raise C6ScoreError("C6 escape elementwise bound is not satisfied.")
    formal = _mapping(channels["formal_residual"], "formal residual comparison")
    _exact_keys(
        formal,
        {"c6spe0_vs_parent", "c6spe_vs_parent", "statement"},
        "formal residual comparison",
    )
    _validate_comparison_record(
        formal["c6spe_vs_parent"],
        "formal_residual.c6spe_vs_parent",
    )
    _validate_comparison_record(
        formal["c6spe0_vs_parent"],
        "formal_residual.c6spe0_vs_parent",
    )

    acceptance = _mapping(score.get("acceptance"), "C6 acceptance")
    _exact_keys(
        acceptance,
        {"all_passed", "checks", "limits", "ungated_recorded_differences"},
        "C6 acceptance",
    )
    checks = _mapping(acceptance.get("checks"), "C6 acceptance checks")
    _exact_keys(
        checks,
        {
            "all_public_reductions_within_predeclared_rounding_bound",
            "detailed_balance_pair_within_limit",
            "detailed_balance_scattering_within_limit",
            "escape_net_within_elementwise_rounding_bound",
            "escape_physical_gain_matches_parent",
            "escape_physical_loss_matches_parent",
            "frozen_qp_residual_bit_exact",
            "omega_zero_bin_exactly_zero",
            "pair_control_physical_gain_matches_parent",
            "pair_control_physical_loss_matches_parent",
            "pair_control_physical_net_matches_parent",
            "scattering_confined_to_author_support",
            "scattering_physical_gain_matches_parent",
            "scattering_physical_loss_matches_parent",
            "scattering_physical_net_matches_parent",
        },
        "C6 acceptance checks",
    )
    if any(value is not True for value in checks.values()):
        raise C6ScoreError("C6 score contains a failed or malformed acceptance check.")
    if acceptance.get("all_passed") is not True:
        raise C6ScoreError("C6 score acceptance is false.")
    limits = _mapping(acceptance.get("limits"), "C6 acceptance limits")
    _exact_keys(
        limits,
        {
            "detailed_balance_relative",
            "escape_net_elementwise_eps_budget",
            "like_for_like_bucket_symmetric_relative_l1",
            "unchanged_channel_net_symmetric_relative_l1",
        },
        "C6 acceptance limits",
    )
    if (
        _float_value(
            limits["unchanged_channel_net_symmetric_relative_l1"],
            "C6 acceptance net limit",
        )
        != _NET_PARITY_LIMIT
        or _float_value(
            limits["like_for_like_bucket_symmetric_relative_l1"],
            "C6 acceptance bucket limit",
        )
        != _BUCKET_PARITY_LIMIT
        or _float_value(
            limits["detailed_balance_relative"],
            "C6 acceptance detailed-balance limit",
        )
        != _DETAILED_BALANCE_LIMIT
        or _float_value(
            limits["escape_net_elementwise_eps_budget"],
            "C6 acceptance escape budget",
        )
        != _ESCAPE_NET_ELEMENTWISE_BUDGET
    ):
        raise C6ScoreError("C6 score acceptance limits are invalid.")
    for required_object in (
        "balance_certification",
        "bookkeeping",
        "component_locality",
        "detailed_balance",
        "extension_policy",
        "kaplan_correction",
        "limitations",
        "map_identity",
        "operator_inputs",
        "units",
    ):
        if not _mapping(score.get(required_object), f"C6 score {required_object}"):
            raise C6ScoreError(f"C6 score {required_object} is empty.")
    if not _json_value_bit_exact(
        score.get("source_binding"),
        _EXPECTED_SOURCE_BINDING,
    ):
        raise C6ScoreError("C6 score source_binding is invalid.")
    contracts = _mapping(score.get("contracts"), "C6 score contracts")
    _exact_keys(
        contracts,
        {"comparison_contract", "coordinate_contract", "frozen_inputs"},
        "C6 score contracts",
    )
    for key in contracts:
        if not _mapping(contracts[key], f"C6 score contracts.{key}"):
            raise C6ScoreError(f"C6 score contracts.{key} is empty.")
    _validate_canonical_pins(score)


def _load_c6_score_unbound(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file_once(path, "checked C6 score")
    score = _parse_json(raw, "checked C6 score")
    if raw != canonical_score_bytes(score):
        raise C6ScoreError("Checked C6 score is not canonical JSON.")
    return score, raw


def _receipt_parent_from_score(score: dict[str, Any]) -> dict[str, object]:
    parent = _mapping(score.get("parent_bindings"), "C6 score parent bindings")
    return {
        "raw_manifest_sha256": parent.get("c5_raw_manifest_sha256"),
        "raw_schema": parent.get("c5_raw_schema"),
        "receipt_file_sha256": parent.get("c5_receipt_sha256"),
        "receipt_schema": parent.get("c5_receipt_schema"),
        "score_file_sha256": parent.get("c5_score_sha256"),
        "score_schema": parent.get("c5_score_schema"),
    }


def load_c6_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Strictly load the repository C6 score/raw/C5 trust anchor."""

    raw = _read_regular_file_once(path, "C6 raw-manifest receipt")
    receipt = _parse_json(raw, "C6 raw-manifest receipt")
    if raw != _canonical_json_bytes(receipt):
        raise C6ScoreError("C6 raw-manifest receipt is not canonical JSON.")
    _exact_keys(
        receipt,
        {"checked_score", "parent_c5", "qualification", "raw_bundle", "schema"},
        "C6 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C6ScoreError("C6 raw-manifest receipt schema is unsupported.")
    if receipt.get("qualification") != (
        "Repository trust anchor for the externally retained C6 raw manifest, "
        "the complete canonical checked-score bytes, and the independently "
        "replayed C5/C4/C3/C2 parent chain; it does not contain or replace the "
        "raw arrays."
    ):
        raise C6ScoreError("C6 raw-manifest receipt qualification is invalid.")
    checked = _mapping(receipt.get("checked_score"), "C6 receipt checked_score")
    _exact_keys(checked, {"file_sha256", "schema"}, "C6 receipt checked_score")
    if checked.get("schema") != SCHEMA:
        raise C6ScoreError("C6 receipt score schema is invalid.")
    _sha256(checked.get("file_sha256"), "C6 receipt score SHA-256")
    raw_bundle = _mapping(receipt.get("raw_bundle"), "C6 receipt raw bundle")
    _exact_keys(raw_bundle, {"manifest_sha256", "schema"}, "C6 receipt raw bundle")
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C6ScoreError("C6 receipt raw schema is invalid.")
    _sha256(raw_bundle.get("manifest_sha256"), "C6 receipt raw manifest SHA-256")
    parent = _mapping(receipt.get("parent_c5"), "C6 receipt parent C5")
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
        "C6 receipt parent C5",
    )
    if (
        parent.get("raw_schema") != C5_RAW_SCHEMA
        or parent.get("receipt_schema") != C5_RECEIPT_SCHEMA
        or parent.get("score_schema") != C5_SCORE_SCHEMA
    ):
        raise C6ScoreError("C6 receipt C5 schemas are invalid.")
    for field in (
        "raw_manifest_sha256",
        "receipt_file_sha256",
        "score_file_sha256",
    ):
        _sha256(parent.get(field), f"C6 receipt parent_c5.{field}")
    return receipt


def load_c6_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Load a checked C6 score and bind it to canonical C5 anchors."""

    score, score_raw = _load_c6_score_unbound(path)
    receipt = load_c6_receipt(receipt_path)
    checked = _mapping(receipt.get("checked_score"), "C6 receipt checked score")
    if hashlib.sha256(score_raw).hexdigest() != checked.get("file_sha256"):
        raise C6ScoreError("Checked C6 score bytes do not match its receipt.")
    if score.get("raw_bundle") != receipt.get("raw_bundle"):
        raise C6ScoreError("Checked C6 raw binding does not match its receipt.")
    if _receipt_parent_from_score(score) != receipt.get("parent_c5"):
        raise C6ScoreError("Checked C6 C5 binding does not match its receipt.")
    parent = _mapping(score.get("parent_bindings"), "checked C6 parent bindings")
    expected_score_path = DEFAULT_C5_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
    expected_receipt_path = DEFAULT_C5_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
    if (
        parent.get("c5_score_path") != expected_score_path
        or parent.get("c5_receipt_path") != expected_receipt_path
    ):
        raise C6ScoreError("Checked C6 does not bind canonical C5 parent paths.")
    c5_score_path, c5_score_bytes = _repository_file_snapshot(
        DEFAULT_C5_SCORE,
        "canonical C5 score",
    )
    c5_receipt_path, c5_receipt_bytes = _repository_file_snapshot(
        DEFAULT_C5_RECEIPT,
        "canonical C5 receipt",
    )
    accepted_c5 = load_c5_score(c5_score_path, receipt_path=c5_receipt_path)
    accepted_raw = _mapping(accepted_c5.get("raw_bundle"), "accepted C5 raw")
    accepted_parent = _mapping(
        accepted_c5.get("parent_bindings"),
        "accepted C5 parent bindings",
    )
    if (
        hashlib.sha256(c5_score_bytes).hexdigest() != parent.get("c5_score_sha256")
        or hashlib.sha256(c5_receipt_bytes).hexdigest()
        != parent.get("c5_receipt_sha256")
        or accepted_raw.get("schema") != parent.get("c5_raw_schema")
        or accepted_raw.get("manifest_sha256")
        != parent.get("c5_raw_manifest_sha256")
        or accepted_parent.get("c3_operator_stage_id")
        != parent.get("c3_operator_stage_id")
        or accepted_parent.get("c3_raw_schema") != parent.get("c3_raw_schema")
        or accepted_parent.get("c3_raw_manifest_sha256")
        != parent.get("c3_raw_manifest_sha256")
        or accepted_parent.get("c4_raw_manifest_sha256")
        != parent.get("c4_raw_manifest_sha256")
        or accepted_parent.get("c2_raw_manifest_sha256")
        != parent.get("c2_raw_manifest_sha256")
    ):
        raise C6ScoreError("Checked C6 canonical C5/C4/C3/C2 binding is stale.")
    _assert_file_snapshot(c5_score_path, c5_score_bytes, "canonical C5 score")
    _assert_file_snapshot(c5_receipt_path, c5_receipt_bytes, "canonical C5 receipt")
    return score


def build_c6_receipt(
    score_path: Path = DEFAULT_SCORE,
    *,
    c6_bundle_dir: Path,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Build a receipt only after independently reproducing C6 score bytes."""

    checked_score_path, checked_score_snapshot = _repository_file_snapshot(
        score_path,
        "checked C6 score for receipt",
    )
    score, score_raw = _load_c6_score_unbound(checked_score_path)
    if score_raw != checked_score_snapshot:
        raise C6ScoreError("Checked C6 score changed before receipt replay.")
    rebuilt = build_c6_score(
        c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_score_bytes(rebuilt) != score_raw:
        raise C6ScoreError(
            "C6 receipt refuses score bytes that do not independently reproduce "
            "from the selected C6/C5/C4/C3/C2 raw evidence."
        )
    _assert_file_snapshot(
        checked_score_path,
        checked_score_snapshot,
        "checked C6 score for receipt",
    )
    return {
        "checked_score": {
            "file_sha256": hashlib.sha256(score_raw).hexdigest(),
            "schema": SCHEMA,
        },
        "parent_c5": _receipt_parent_from_score(score),
        "qualification": (
            "Repository trust anchor for the externally retained C6 raw manifest, "
            "the complete canonical checked-score bytes, and the independently "
            "replayed C5/C4/C3/C2 parent chain; it does not contain or replace the "
            "raw arrays."
        ),
        "raw_bundle": dict(
            _mapping(score.get("raw_bundle"), "C6 score raw bundle")
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
        raise FileExistsError(f"C6 output already exists: {target}")
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


def write_c6_score(
    output_path: Path,
    c6_bundle_dir: Path,
    *,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    score = build_c6_score(
        c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, canonical_score_bytes(score))


def write_c6_receipt(
    output_path: Path,
    *,
    score_path: Path = DEFAULT_SCORE,
    c6_bundle_dir: Path,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    receipt = build_c6_receipt(
        score_path,
        c6_bundle_dir=c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    return _atomic_exclusive_write(output_path, _canonical_json_bytes(receipt))


def _add_parent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c6-bundle", type=Path, required=True)
    parser.add_argument("--c5-bundle", type=Path, required=True)
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c5-score", type=Path, default=DEFAULT_C5_SCORE)
    parser.add_argument("--c5-receipt", type=Path, default=DEFAULT_C5_RECEIPT)
    parser.add_argument("--c4-score", type=Path, default=DEFAULT_C4_SCORE)
    parser.add_argument("--c4-receipt", type=Path, default=DEFAULT_C4_RECEIPT)
    parser.add_argument("--c3-score", type=Path, default=DEFAULT_C3_SCORE)
    parser.add_argument("--c3-receipt", type=Path, default=DEFAULT_C3_RECEIPT)
    parser.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    parser.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="build the checked C6 score")
    _add_parent_arguments(score)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    receipt = subparsers.add_parser("receipt", help="build the C6 receipt")
    _add_parent_arguments(receipt)
    receipt.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    receipt.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    common = {
        "c5_bundle_dir": args.c5_bundle,
        "c4_bundle_dir": args.c4_bundle,
        "c3_bundle_dir": args.c3_bundle,
        "c2_bundle_dir": args.c2_bundle,
        "c5_score_path": args.c5_score,
        "c5_receipt_path": args.c5_receipt,
        "c4_score_path": args.c4_score,
        "c4_receipt_path": args.c4_receipt,
        "c3_score_path": args.c3_score,
        "c3_receipt_path": args.c3_receipt,
        "c2_score_path": args.c2_score,
        "c2_receipt_path": args.c2_receipt,
    }
    if args.command == "receipt":
        result = write_c6_receipt(
            args.output,
            score_path=args.score,
            c6_bundle_dir=args.c6_bundle,
            **common,
        )
    else:
        result = write_c6_score(
            args.output,
            args.c6_bundle,
            **common,
        )
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
