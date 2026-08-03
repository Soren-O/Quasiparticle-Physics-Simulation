"""Independently verify the formal Figure 6 C7 nonlinear-solver re-solve.

The C7 producer intentionally lives in another module.  This verifier does
not import that producer, ``coupled_newton_solve``, or the public collision
APIs for its scientific replay.  It strictly loads the externally retained
raw bundle, replays the accepted C6/C5/C4/C3/C2 chain, authenticates the
seed against the committed C0 evidence, and independently transcribes the
complete residual assembly -- the C4 photon loop, the C5 QP-side
scattering/pair contractions with the dynamic phonon occupation, the C6
phonon-side balance with its Kaplan ``S_+`` correction, and the bath
escape -- with fixed-order ``math.fsum`` reductions at both the returned
root and the frozen C6 state.

The Newton linear algebra itself is qpsim runtime semantics (LAPACK plus a
backtracking line search); the root is therefore certified
host-independently through residual and balance bounds rather than by
bit-replaying the iteration path.  The observable records are re-derived
through the clean-room author gap-integral transcription.

C7 records one authenticated seed/root pair.  It claims no 300-point curve,
no paper parity, and no author-equivalence of the changed iteration policy.
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
from validation.fischer_2023.fig6_author_c3_score import load_c3_raw_bundle
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as DEFAULT_C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_SCORE as DEFAULT_C4_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as DEFAULT_C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_SCORE as DEFAULT_C5_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import (
    _clean_channel_reductions as _clean_qp_channel_reductions,
)
from validation.fischer_2023.fig6_author_c5_score import load_c5_raw_bundle
from validation.fischer_2023.fig6_author_c6_score import (
    DEFAULT_RECEIPT as DEFAULT_C6_RECEIPT,
)
from validation.fischer_2023.fig6_author_c6_score import (
    DEFAULT_SCORE as DEFAULT_C6_SCORE,
)
from validation.fischer_2023.fig6_author_c6_score import (
    RAW_SCHEMA as C6_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    RECEIPT_SCHEMA as C6_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    SCHEMA as C6_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    _clean_channel_contractions,
    _clean_kaplan_correction,
    _clean_thermal_bose,
    build_c6_score,
    load_c6_raw_bundle,
    load_c6_score,
)
from validation.fischer_2023.fig6_author_c6_score import (
    canonical_score_bytes as canonical_c6_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_SCHEMA = "qpsim.fischer2023.fig6-author-c7-nonlinear-solver-bundle.v1"
SCHEMA = "qpsim.fischer2023.fig6-author-c7-nonlinear-solver-score.v1"
RECEIPT_SCHEMA = (
    "qpsim.fischer2023.fig6-author-c7-nonlinear-solver-raw-manifest-receipt.v1"
)
DEFAULT_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c7-nonlinear-solver-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c7-raw-manifest-receipt.json"
)
DEFAULT_C0_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c0-author-equivalent-score.json"
)

STAGE_ID = "C7"
PARENT_STAGE_ID = "C6"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "nonlinear_solver"
SECONDS_PER_NS = 1.0e-9
N_QP = 1640
N_ACTIVE = 1620
N_OMEGA = 3600
N_AUTHOR = 1620
ACTIVE_START = 20
AUTHOR_OMEGA_STOP = 1619
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
T_BATH_K = 0.2
OMEGA_0_UEV = 20.0
C_PHOT_NS_INV = 1.0e-9
PHOTON_STEP = 20
STEP_RTOL = 1.0e-7
MAX_ITER = 10
AUTHOR_CONTROL_ORDINATE = 0.12090908988993258

_ARRAY_NAME_RE = re.compile(r"[A-Za-z0-9_]+")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FLOAT_EPS = float(np.finfo(np.float64).eps)
_REDUCTION_OPERATION_BUDGET = 8 * N_QP + 64
_REDUCTION_GAMMA = (
    _REDUCTION_OPERATION_BUDGET
    * _FLOAT_EPS
    / (1.0 - _REDUCTION_OPERATION_BUDGET * _FLOAT_EPS)
)
_CORRECTION_OPERATION_BUDGET = 2 * _REDUCTION_OPERATION_BUDGET
_CORRECTION_GAMMA = (
    _CORRECTION_OPERATION_BUDGET
    * _FLOAT_EPS
    / (1.0 - _CORRECTION_OPERATION_BUDGET * _FLOAT_EPS)
)
# The frozen-state residual assembly reorders channel sums relative to the
# accepted C6 hybrid residuals; the deltas are gated as symmetric relative
# L1 against the same limit used for unchanged channels in C5/C6.
_FROZEN_DELTA_LIMIT = 1.0e-12
# The observable comparison chains the qpsim direct-gap integral against the
# clean-room author transcription; both are fixed scalar expressions over
# 1620 samples.
_OBSERVABLE_RELATIVE_LIMIT = 1.0e-12
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
_EXPECTED_RAW_MANIFEST_SHA256 = (
    "c7906801bee1adb716fed1c2425f9d2104c63acef7bd95b40f44470cae74e73a"
)
_EXPECTED_EVIDENCE_DIGEST = (
    "0a4843fe7d20a58c427cc018cdec06662b685bdbf32a04b748991a802f16b105"
)
_EXPECTED_CANONICAL_METRICS: dict[str, float] = {
    "frozen_phonon_assembly_symmetric_relative_l1": float.fromhex(
        "0x1.852ffb292bf7cp-53"
    ),
    "frozen_qp_assembly_symmetric_relative_l1": float.fromhex(
        "0x1.29dbd9ecbf839p-43"
    ),
    "root_ordinate_authors": float.fromhex("0x1.29d3f1263dc67p-3"),
    "root_ordinate_centers": float.fromhex("0x1.3e2aefabad5f4p-4"),
    "root_ph_balance_l1_ratio": float.fromhex("0x1.951455cf0d1bap-38"),
    "root_qp_balance_l1_ratio": float.fromhex("0x1.4e1dc1819f830p-45"),
}
_EXPECTED_CANONICAL_INTS: dict[str, int] = {
    "accepted_max_iter": 4,
}
_EXPECTED_ROOT_NPY_SHA256: dict[str, str] = {
    "qpsim_root_f": (
        "a59a586b12b4cc3defde8845fde964e7b4569cd8a8332bcb6b60ba7332f45232"
    ),
    "qpsim_root_n_ph": (
        "b051bb97f17692e812dc935334559b24ba1201489e2f52e0c34ef1e39ca3a8c8"
    ),
}
_VERIFIER_RUNTIME_CONTRACT = {
    "arithmetic": (
        "fixed-order Python math.fsum reductions plus elementwise IEEE-754 "
        "gamma bounds"
    ),
    "host_runtime_binding": "not pinned",
    "qualification": (
        "The retained root and residual arrays are authenticated to the "
        "producer runtime. Verifier acceptance is host-independent: it does "
        "not re-run the LAPACK Newton path or demand matching last bits; it "
        "certifies the root through residual and balance bounds."
    ),
}
_EXPECTED_SOURCE_BINDING = {
    "hash_kind": "canonical_sha256_import_time_disk_snapshot",
    "scope": (
        "complete qpsim Python/material source tree, C7 producer, "
        "C2/C3/C4/C5/C6 replay verifiers, C5/C6 producers, and provenance "
        "helper"
    ),
}

_FLOAT_1640 = ("<f8", (N_QP,))
_BOOL_1640 = ("|b1", (N_QP,))
_FLOAT_3600 = ("<f8", (N_OMEGA,))
_BOOL_3600 = ("|b1", (N_OMEGA,))
_FLOAT_1619 = ("<f8", (AUTHOR_OMEGA_STOP,))
_INT64_1619 = ("<i8", (AUTHOR_OMEGA_STOP,))

_ARRAY_SPECS: dict[str, tuple[str, tuple[int, ...]]] = {
    "author_seed_state": ("<f8", (2 * N_AUTHOR - 1,)),
    "frozen_qp_residual_delta_s_inv": _FLOAT_1640,
    "frozen_phonon_residual_delta_support_s_inv": _FLOAT_1619,
    "parent_E_centers_ueV": _FLOAT_1640,
    "parent_active_mask": _BOOL_1640,
    "parent_cell_density": _FLOAT_1640,
    "parent_cell_weights_ueV": _FLOAT_1640,
    "parent_dE_ueV": _FLOAT_1640,
    "parent_f": _FLOAT_1640,
    "parent_legacy_phonon_support_mask": _BOOL_3600,
    "parent_phonon_residual_s_inv": _FLOAT_1619,
    "parent_phonon_to_native_omega_index": _INT64_1619,
    "parent_projected_n_phonon": _FLOAT_3600,
    "parent_qp_residual_s_inv": _FLOAT_1640,
    "parent_thermal_f": _FLOAT_1640,
    "qpsim_frozen_R_f_ns_inv": _FLOAT_1640,
    "qpsim_frozen_R_ph_ns_inv": _FLOAT_3600,
    "qpsim_root_R_f_ns_inv": _FLOAT_1640,
    "qpsim_root_R_ph_ns_inv": _FLOAT_3600,
    "qpsim_root_a_ph_ns_inv": _FLOAT_3600,
    "qpsim_root_b_ph_ns_inv": _FLOAT_3600,
    "qpsim_root_f": _FLOAT_1640,
    "qpsim_root_minus_frozen_f": _FLOAT_1640,
    "qpsim_root_minus_frozen_n_ph": _FLOAT_3600,
    "qpsim_root_n_ph": _FLOAT_3600,
    "qpsim_root_pair_gain_s_inv": _FLOAT_1640,
    "qpsim_root_pair_loss_s_inv": _FLOAT_1640,
    "qpsim_root_pair_net_s_inv": _FLOAT_1640,
    "qpsim_root_phonon_escape_net_ns_inv": _FLOAT_3600,
    "qpsim_root_phonon_pair_a_ns_inv": _FLOAT_3600,
    "qpsim_root_phonon_pair_b_ns_inv": _FLOAT_3600,
    "qpsim_root_phonon_scattering_a_ns_inv": _FLOAT_3600,
    "qpsim_root_phonon_scattering_b_ns_inv": _FLOAT_3600,
    "qpsim_root_photon_gain_s_inv": _FLOAT_1640,
    "qpsim_root_photon_loss_s_inv": _FLOAT_1640,
    "qpsim_root_photon_net_s_inv": _FLOAT_1640,
    "qpsim_root_qp_gain_ns_inv": _FLOAT_1640,
    "qpsim_root_qp_loss_rate_ns_inv": _FLOAT_1640,
    "qpsim_root_scattering_gain_s_inv": _FLOAT_1640,
    "qpsim_root_scattering_loss_s_inv": _FLOAT_1640,
    "qpsim_root_scattering_net_s_inv": _FLOAT_1640,
    "qpsim_thermal_n_ph": _FLOAT_3600,
    "seed_f": _FLOAT_1640,
    "seed_n_ph": _FLOAT_3600,
}
_EXPECTED_ARRAY_NAMES = frozenset(_ARRAY_SPECS)
if len(_EXPECTED_ARRAY_NAMES) != 44:  # pragma: no cover - import-time invariant
    raise RuntimeError("Internal C7 raw array closure must contain exactly 44 arrays.")

_RAW_METADATA_KEYS = {
    "array_descriptors",
    "comparison_contract",
    "component_locality",
    "conservation",
    "limitations",
    "observable",
    "parent_bindings",
    "runtime",
    "schema",
    "seed_binding",
    "solver_contract",
    "sources",
    "stage",
    "units",
}
_SCORE_KEYS = {
    "acceptance",
    "array_descriptors",
    "channel_verification",
    "comparison",
    "component_locality",
    "conservation",
    "contracts",
    "limitations",
    "observable",
    "parent_bindings",
    "raw_bundle",
    "rounding_contract",
    "runtime",
    "schema",
    "seed_binding",
    "solver_contract",
    "source_binding",
    "sources",
    "stage",
    "units",
}

_C7_PRODUCER_SOURCE = (
    REPOSITORY_ROOT / "validation/fischer_2023/fig6_author_c7_bundle.py"
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
        "validation/fischer_2023/fig6_author_c6_bundle.py",
        "validation/fischer_2023/fig6_author_c6_score.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/reference_models/__init__.py",
        "validation/reference_models/fischer_2023/__init__.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    )
)
_RAW_SOURCE_HASHES_AT_IMPORT = source_manifest(
    _C7_PRODUCER_SOURCE,
    extra_validation_modules=_TRANSITIVE_VALIDATION_SOURCES,
)
_SOURCE_HASHES_AT_IMPORT = source_manifest(
    Path(__file__).resolve(),
    extra_validation_modules=(
        *_TRANSITIVE_VALIDATION_SOURCES,
        _C7_PRODUCER_SOURCE,
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
    raise RuntimeError("C7 source hashes changed during import.")
_VERIFIER_RELATIVE = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
_RAW_SOURCES_FROM_VERIFIER_CLOSURE = {
    relative: digest
    for relative, digest in _SOURCE_HASHES_AT_IMPORT.items()
    if relative != _VERIFIER_RELATIVE
}
if (
    _RAW_SOURCES_FROM_VERIFIER_CLOSURE != _RAW_SOURCE_HASHES_AT_IMPORT
):  # pragma: no cover - import-time provenance invariant
    raise RuntimeError("C7 verifier and producer source closures are inconsistent.")


class C7ScoreError(ValueError):
    """The C7 raw evidence, parent chain, score, or receipt is malformed."""


@dataclass(frozen=True)
class _DirectoryState:
    root_identity: tuple[int, int, int, int, int]
    entries: tuple[tuple[str, tuple[int, int, int, int, int]], ...]


@dataclass
class _ParentContext:
    c6_arrays: dict[str, np.ndarray]
    c6_manifest_sha256: str
    c6_score: dict[str, Any]
    c6_score_path: Path
    c6_score_bytes: bytes
    c6_receipt_path: Path
    c6_receipt_bytes: bytes
    c0_score: dict[str, Any]
    c0_score_path: Path
    c0_score_bytes: bytes
    c0_manifest_sha256: str
    c0_seed: np.ndarray
    c3_metadata: dict[str, Any]
    c3_arrays: dict[str, np.ndarray]
    c3_manifest_sha256: str
    c5_arrays: dict[str, np.ndarray]
    c5_manifest_sha256: str
    bundle_states: tuple[tuple[Path, _DirectoryState], ...]
    anchor_snapshots: tuple[tuple[Path, bytes], ...]


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C7ScoreError(f"C7 numerical source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C7ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C7ScoreError(f"Forbidden non-finite JSON constant {token!r}.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C7ScoreError(f"{label} is not strict UTF-8 JSON.") from exc
    return _mapping(value, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C7ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise C7ScoreError(f"{label} keys are invalid; missing={missing}, extra={extra}.")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise C7ScoreError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise C7ScoreError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise C7ScoreError(f"{label} must be >= {minimum}.")
    return value


def _finite_scalar(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C7ScoreError(f"{label} must be a finite scalar.")
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise C7ScoreError(f"{label} must be a finite {qualifier}scalar.")
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
    return type(reference) is type(candidate) and reference == candidate


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    if not np.isfinite(result):
        raise C7ScoreError("C7 scalar records must be finite.")
    return {"hex": result.hex(), "value": result}


def _float_value(record: object, label: str) -> float:
    mapped = _mapping(record, label)
    _exact_keys(mapped, {"hex", "value"}, label)
    value = _finite_scalar(mapped.get("value"), f"{label}.value")
    if mapped.get("hex") != value.hex():
        raise C7ScoreError(f"{label} decimal and hexadecimal forms disagree.")
    return value


def _fixed_sum(value: np.ndarray) -> float:
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
        raise C7ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C7ScoreError(f"{label} must be a regular non-symlink file.")
    try:
        with candidate.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = candidate.lstat()
    except OSError as exc:
        raise C7ScoreError(f"{label} changed or became unreadable.") from exc
    if not (
        _stat_identity(before)
        == _stat_identity(opened_before)
        == _stat_identity(opened_after)
        == _stat_identity(after)
    ):
        raise C7ScoreError(f"{label} changed while it was being read.")
    return content


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C7ScoreError(f"{label} must stay inside the repository.") from exc
    if resolved != Path(path).absolute() or resolved.is_symlink():
        raise C7ScoreError(f"{label} is unsafe or a symlink.")
    return resolved, _read_regular_file_once(resolved, label)


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    if _read_regular_file_once(path, label) != expected:
        raise C7ScoreError(f"{label} changed during C7 verification.")


def _directory_state(path: Path, label: str) -> tuple[Path, _DirectoryState]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise C7ScoreError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise C7ScoreError(f"{label} must be a non-symlink directory.")
    root = candidate.resolve()
    if root != candidate.absolute() or root.is_symlink():
        raise C7ScoreError(f"{label} is unsafe or a symlink.")
    entries: list[tuple[str, tuple[int, int, int, int, int]]] = []
    try:
        for child in sorted(root.iterdir(), key=lambda item: item.name):
            child_stat = child.lstat()
            if stat.S_ISLNK(child_stat.st_mode):
                raise C7ScoreError(f"{label} contains a symlink.")
            entries.append((child.name, _stat_identity(child_stat)))
        after = root.lstat()
    except OSError as exc:
        raise C7ScoreError(f"{label} changed while being enumerated.") from exc
    if _stat_identity(before) != _stat_identity(after):
        raise C7ScoreError(f"{label} changed while being enumerated.")
    return root, _DirectoryState(_stat_identity(after), tuple(entries))


def _assert_directory_state(path: Path, expected: _DirectoryState, label: str) -> None:
    root, actual = _directory_state(path, label)
    del root
    if actual != expected:
        raise C7ScoreError(f"{label} changed during C7 verification.")


def load_c7_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one closed canonical NPY-v3 C7 raw bundle."""

    root, before = _directory_state(bundle_dir, "C7 raw bundle")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C7 raw manifest")
    manifest = _parse_json(manifest_raw, "C7 raw manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C7 raw manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C7ScoreError("C7 raw manifest schema is unsupported.")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise C7ScoreError("C7 raw manifest is not canonical JSON.")
    files = _mapping(manifest.get("files"), "C7 raw manifest files")
    metadata = _mapping(manifest.get("metadata"), "C7 raw metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C7 raw metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C7ScoreError("C7 raw metadata schema is unsupported.")
    expected_filenames = {f"{name}.npy" for name in _EXPECTED_ARRAY_NAMES}
    if set(files) != expected_filenames or len(files) != 44:
        raise C7ScoreError("C7 raw file closure is invalid.")
    if {name for name, _identity in before.entries} != expected_filenames | {
        "manifest.json"
    }:
        raise C7ScoreError("C7 raw directory closure is invalid.")
    if any(not stat.S_ISREG(identity[2]) for _name, identity in before.entries):
        raise C7ScoreError("C7 raw bundle contains a non-file entry.")

    arrays: dict[str, np.ndarray] = {}
    for filename in sorted(expected_filenames):
        name = filename[:-4]
        if (
            Path(filename).name != filename
            or _ARRAY_NAME_RE.fullmatch(name) is None
            or not filename.endswith(".npy")
        ):
            raise C7ScoreError(f"Unsafe C7 raw filename {filename!r}.")
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        expected_sha = _sha256(record.get("sha256"), f"files.{filename}.sha256")
        expected_size = _strict_int(
            record.get("size_bytes"),
            f"files.{filename}.size_bytes",
            minimum=1,
        )
        content = _read_regular_file_once(root / filename, f"C7 raw {filename}")
        if (
            len(content) != expected_size
            or hashlib.sha256(content).hexdigest() != expected_sha
        ):
            raise C7ScoreError(f"C7 raw file {filename!r} failed manifest binding.")
        if len(content) < 8 or content[:8] != b"\x93NUMPY\x03\x00":
            raise C7ScoreError(f"C7 raw file {filename!r} is not canonical NPY v3.")
        try:
            stream = io.BytesIO(content)
            loaded = np.lib.format.read_array(stream, allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise C7ScoreError(f"Cannot load C7 raw array {filename!r}.") from exc
        if stream.tell() != len(content):
            raise C7ScoreError(f"C7 raw file {filename!r} contains trailing bytes.")
        array = np.asarray(loaded)
        expected_dtype, expected_shape = _ARRAY_SPECS[name]
        if array.dtype.str != expected_dtype or array.shape != expected_shape:
            raise C7ScoreError(
                f"C7 raw {name!r} expected dtype/shape "
                f"{expected_dtype}/{expected_shape}, got "
                f"{array.dtype.str}/{array.shape}."
            )
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise C7ScoreError(f"C7 raw array {filename!r} contains non-finite values.")
        if array.dtype.kind == "f" and np.any((array == 0.0) & np.signbit(array)):
            raise C7ScoreError(
                f"C7 raw array {filename!r} contains non-canonical signed zero."
            )
        if _npy_bytes(array) != content:
            raise C7ScoreError(
                f"C7 raw file {filename!r} is not byte-canonical NPY v3."
            )
        arrays[name] = array
    descriptors = _mapping(metadata.get("array_descriptors"), "C7 raw descriptors")
    expected_descriptors = {
        name: _array_descriptor(value) for name, value in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C7ScoreError("C7 raw array descriptors are incomplete, forged, or stale.")
    _assert_directory_state(root, before, "C7 raw bundle")
    return metadata, arrays, hashlib.sha256(manifest_raw).hexdigest()


def _accept_parent(
    c6_bundle_dir: Path,
    *,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c0_bundle_dir: Path,
    c6_score_path: Path,
    c6_receipt_path: Path,
    c5_score_path: Path,
    c5_receipt_path: Path,
    c4_score_path: Path,
    c4_receipt_path: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
    c0_score_path: Path,
) -> _ParentContext:
    """Replay the C6/C5/C4/C3/C2 chain and bind the C0-authenticated seed."""

    bundle_states = tuple(
        _directory_state(path, f"selected {label} raw bundle")
        for label, path in (
            ("C6", c6_bundle_dir),
            ("C5", c5_bundle_dir),
            ("C4", c4_bundle_dir),
            ("C3", c3_bundle_dir),
            ("C2", c2_bundle_dir),
            ("C0", c0_bundle_dir),
        )
    )
    anchors = tuple(
        _repository_file_snapshot(path, label)
        for label, path in (
            ("checked C6 score", c6_score_path),
            ("checked C6 receipt", c6_receipt_path),
            ("checked C5 score", c5_score_path),
            ("checked C5 receipt", c5_receipt_path),
            ("checked C4 score", c4_score_path),
            ("checked C4 receipt", c4_receipt_path),
            ("checked C3 score", c3_score_path),
            ("checked C3 receipt", c3_receipt_path),
            ("checked C2 score", c2_score_path),
            ("checked C2 receipt", c2_receipt_path),
            ("checked C0 score", c0_score_path),
        )
    )
    checked_c6_path, checked_c6_bytes = anchors[0]
    checked_c6_receipt_path, checked_c6_receipt_bytes = anchors[1]
    checked_c0_path, checked_c0_bytes = anchors[10]

    accepted_c6 = load_c6_score(
        checked_c6_path,
        receipt_path=checked_c6_receipt_path,
    )
    rebuilt_c6 = build_c6_score(
        bundle_states[0][0],
        c5_bundle_dir=bundle_states[1][0],
        c4_bundle_dir=bundle_states[2][0],
        c3_bundle_dir=bundle_states[3][0],
        c2_bundle_dir=bundle_states[4][0],
        c5_score_path=anchors[2][0],
        c5_receipt_path=anchors[3][0],
        c4_score_path=anchors[4][0],
        c4_receipt_path=anchors[5][0],
        c3_score_path=anchors[6][0],
        c3_receipt_path=anchors[7][0],
        c2_score_path=anchors[8][0],
        c2_receipt_path=anchors[9][0],
    )
    if canonical_c6_score_bytes(rebuilt_c6) != checked_c6_bytes:
        raise C7ScoreError(
            "Selected C6/C5/C4/C3/C2 raw evidence does not reproduce the "
            "complete checked C6 score bytes."
        )
    if not _json_value_bit_exact(accepted_c6, rebuilt_c6):
        raise C7ScoreError("Accepted and independently replayed C6 scores disagree.")
    _c6_metadata, c6_arrays, c6_manifest_sha = load_c6_raw_bundle(bundle_states[0][0])
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(bundle_states[3][0])
    _c5_metadata, c5_arrays, c5_manifest_sha = load_c5_raw_bundle(bundle_states[1][0])
    raw_binding = _mapping(accepted_c6.get("raw_bundle"), "accepted C6 raw binding")
    if (
        raw_binding.get("schema") != C6_RAW_SCHEMA
        or raw_binding.get("manifest_sha256") != c6_manifest_sha
    ):
        raise C7ScoreError("Selected C6 raw bundle is not the accepted C6 evidence.")
    parent_binding = _mapping(
        accepted_c6.get("parent_bindings"),
        "accepted C6 parent binding",
    )
    if (
        parent_binding.get("c3_raw_manifest_sha256") != c3_manifest_sha
        or parent_binding.get("c5_raw_manifest_sha256") != c5_manifest_sha
    ):
        raise C7ScoreError("Selected C3/C5 raw bundles are not C6's accepted ancestors.")
    stage = _mapping(accepted_c6.get("stage"), "accepted C6 stage")
    if stage != {
        "changed_component": "phonon_balance",
        "comparison_stage_id": "C5",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C5",
        "stage_id": "C6",
        "status": "completed",
    }:
        raise C7ScoreError("Accepted parent is not the completed formal C6 stage.")

    c0_score = _parse_json(checked_c0_bytes, "checked C0 score")
    c0_raw_binding = _mapping(c0_score.get("raw_bundle"), "C0 raw binding")
    c0_manifest_raw = _read_regular_file_once(
        bundle_states[5][0] / "manifest.json",
        "C0 raw manifest",
    )
    c0_manifest_sha = hashlib.sha256(c0_manifest_raw).hexdigest()
    if c0_raw_binding.get("manifest_sha256") != c0_manifest_sha:
        raise C7ScoreError(
            "Selected C0 raw bundle is not the accepted A1-equivalent evidence."
        )
    c0_descriptors = _mapping(c0_score.get("array_descriptors"), "C0 descriptors")
    seed_descriptor = _mapping(
        c0_descriptors.get("initial_state"),
        "C0 initial-state descriptor",
    )
    seed_bytes = _read_regular_file_once(
        bundle_states[5][0] / "initial_state.npy",
        "C0 initial state",
    )
    if hashlib.sha256(seed_bytes).hexdigest() != seed_descriptor.get("npy_sha256"):
        raise C7ScoreError("C0 initial-state bytes do not match the accepted descriptor.")
    c0_seed = np.asarray(
        np.lib.format.read_array(io.BytesIO(seed_bytes), allow_pickle=False)
    )
    if c0_seed.shape != (2 * N_AUTHOR - 1,) or c0_seed.dtype.str != "<f8":
        raise C7ScoreError("C0 initial state has an unexpected shape or dtype.")

    for (path, content), label in zip(
        anchors,
        (
            "checked C6 score",
            "checked C6 receipt",
            "checked C5 score",
            "checked C5 receipt",
            "checked C4 score",
            "checked C4 receipt",
            "checked C3 score",
            "checked C3 receipt",
            "checked C2 score",
            "checked C2 receipt",
            "checked C0 score",
        ),
        strict=True,
    ):
        _assert_file_snapshot(path, content, label)
    for (root, state), label in zip(
        bundle_states,
        ("C6", "C5", "C4", "C3", "C2", "C0"),
        strict=True,
    ):
        _assert_directory_state(root, state, f"selected {label} raw bundle")
    return _ParentContext(
        c6_arrays=c6_arrays,
        c6_manifest_sha256=c6_manifest_sha,
        c6_score=accepted_c6,
        c6_score_path=checked_c6_path,
        c6_score_bytes=checked_c6_bytes,
        c6_receipt_path=checked_c6_receipt_path,
        c6_receipt_bytes=checked_c6_receipt_bytes,
        c0_score=c0_score,
        c0_score_path=checked_c0_path,
        c0_score_bytes=checked_c0_bytes,
        c0_manifest_sha256=c0_manifest_sha,
        c0_seed=c0_seed,
        c3_metadata=c3_metadata,
        c3_arrays=c3_arrays,
        c3_manifest_sha256=c3_manifest_sha,
        c5_arrays=c5_arrays,
        c5_manifest_sha256=c5_manifest_sha,
        bundle_states=bundle_states,
        anchor_snapshots=anchors,
    )


def _recheck_parent(parent: _ParentContext) -> None:
    for path, content in parent.anchor_snapshots:
        _assert_file_snapshot(path, content, "C7 parent anchor")
    for root, state in parent.bundle_states:
        _assert_directory_state(root, state, "C7 parent raw bundle")


def _clean_photon_channel(
    f: np.ndarray,
    rho: np.ndarray,
    supported: np.ndarray,
    k_plus: np.ndarray,
    n_bar: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-order transcription of the public sub-gap photon loop (ns^-1)."""

    gain = np.zeros(N_QP, dtype=np.float64)
    loss_rate = np.zeros(N_QP, dtype=np.float64)
    for i in range(N_QP):
        if not bool(supported[i]):
            continue
        gain_terms: list[float] = []
        loss_terms: list[float] = []
        j_up = i + PHOTON_STEP
        if j_up < N_QP:
            u_plus = float(rho[j_up]) * float(k_plus[i, j_up])
            gain_terms.append(
                C_PHOT_NS_INV * u_plus * float(f[j_up]) * (n_bar + 1.0)
            )
            loss_terms.append(
                C_PHOT_NS_INV * u_plus * (1.0 - float(f[j_up])) * n_bar
            )
        j_down = i - PHOTON_STEP
        if j_down >= 0 and bool(supported[j_down]):
            u_plus = float(rho[j_down]) * float(k_plus[i, j_down])
            gain_terms.append(
                C_PHOT_NS_INV * u_plus * float(f[j_down]) * n_bar
            )
            loss_terms.append(
                C_PHOT_NS_INV * u_plus * (1.0 - float(f[j_down])) * (n_bar + 1.0)
            )
        gain[i] = (1.0 - float(f[i])) * math.fsum(gain_terms)
        loss_rate[i] = math.fsum(loss_terms)
    return gain, loss_rate


def _clean_author_gap(
    f_active: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
    delta0_eV: float,
) -> float:
    """Clean-room transcription of the author left-edge direct-gap integral."""

    occupation = np.asarray(f_active, dtype=np.float64)
    indices = np.arange(occupation.size, dtype=np.float64)
    a_lo = np.arcsinh(np.sqrt(indices * h_eV / (2.0 * gap_eV)))
    a_hi = np.arcsinh(np.sqrt((indices + 1.0) * h_eV / (2.0 * gap_eV)))
    constant = math.fsum(
        4.0 * float(occupation[i]) * (float(a_hi[i]) - float(a_lo[i]))
        for i in range(occupation.size)
    )
    left = h_eV * np.arange(occupation.size - 1, dtype=np.float64)
    linear = math.fsum(
        2.0
        * (float(occupation[i + 1]) - float(occupation[i]))
        / h_eV
        * (
            math.sqrt((float(left[i]) + h_eV) * (float(left[i]) + h_eV + 2.0 * gap_eV))
            - math.sqrt(float(left[i]) * (float(left[i]) + 2.0 * gap_eV))
            - 2.0 * (float(left[i]) + gap_eV) * (float(a_hi[i]) - float(a_lo[i]))
        )
        for i in range(occupation.size - 1)
    )
    return delta0_eV * math.exp(-(constant + linear))


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
        parent.c6_score.get("parent_bindings"),
        "accepted C6 parent bindings",
    )
    return {
        "c0_raw_manifest_sha256": parent.c0_manifest_sha256,
        "c0_score_path": parent.c0_score_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c0_score_sha256": hashlib.sha256(parent.c0_score_bytes).hexdigest(),
        "c2_raw_manifest_sha256": inherited.get("c2_raw_manifest_sha256"),
        "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
        "c3_raw_manifest_sha256": parent.c3_manifest_sha256,
        "c4_raw_manifest_sha256": inherited.get("c4_raw_manifest_sha256"),
        "c5_raw_manifest_sha256": parent.c5_manifest_sha256,
        "c6_raw_manifest_sha256": parent.c6_manifest_sha256,
        "c6_raw_schema": C6_RAW_SCHEMA,
        "c6_receipt_path": parent.c6_receipt_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c6_receipt_schema": C6_RECEIPT_SCHEMA,
        "c6_receipt_sha256": hashlib.sha256(parent.c6_receipt_bytes).hexdigest(),
        "c6_score_path": parent.c6_score_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "c6_score_schema": C6_SCORE_SCHEMA,
        "c6_score_sha256": hashlib.sha256(parent.c6_score_bytes).hexdigest(),
        "c6_stage_id": PARENT_STAGE_ID,
    }


def _expected_frozen_arrays(parent: _ParentContext) -> dict[str, np.ndarray]:
    c3 = parent.c3_arrays
    c6 = parent.c6_arrays
    return {
        "author_seed_state": parent.c0_seed,
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
        "parent_thermal_f": np.asarray(c3["projected_thermal_f"]),
        "parent_qp_residual_s_inv": np.asarray(c6["c6_qp_residual_s_inv"]),
        "parent_phonon_residual_s_inv": np.asarray(
            c6["c6spe_phonon_residual_s_inv"]
        ),
        "qpsim_thermal_n_ph": np.asarray(c6["qpsim_thermal_n_ph"]),
    }


def _residual_assembly_verification(
    arrays: dict[str, np.ndarray],
    parent: _ParentContext,
    *,
    f_state: np.ndarray,
    n_state: np.ndarray,
    n_bar: float,
    correction: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[bool]]:
    """Clean fixed-order (R_f, R_ph) at one state, with gamma-bound records."""

    rho = arrays["parent_cell_density"]
    dE = arrays["parent_dE_ueV"]
    weights = arrays["parent_cell_weights_ueV"]
    active = arrays["parent_active_mask"]
    n_th = arrays["qpsim_thermal_n_ph"]
    k_s0 = np.asarray(
        parent.c5_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"]
    )
    k_r0 = np.asarray(parent.c5_arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"])
    k_s_ph = np.asarray(
        parent.c6_arrays["qpsim_phonon_scattering_kernel_ns_inv_ueV_inv"]
    )
    k_r_ph = np.asarray(
        parent.c6_arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"]
    )
    k_plus = np.asarray(parent.c3_arrays["native_K_plus_full"])

    levels = np.arange(N_QP, dtype=np.int64)
    idx_diff = np.abs(levels[:, None] - levels[None, :])
    idx_sum = 321 + levels[:, None] + levels[None, :]
    diff_sign = np.sign(levels[:, None] - levels[None, :]).astype(np.int8)
    n_diff = n_state[idx_diff]
    n_sum = n_state[idx_sum]
    n_p = np.where(diff_sign > 0, 1.0 + n_diff, n_diff)
    np.fill_diagonal(n_p, 0.0)
    n_emit = 1.0 + n_sum
    n_abs = n_sum.copy()

    qp = _clean_qp_channel_reductions(
        f_state,
        weights,
        active,
        k_s0,
        k_r0,
        n_p,
        n_emit,
        n_abs,
    )
    photon_gain, photon_loss_rate = _clean_photon_channel(
        f_state,
        rho,
        active,
        k_plus,
        n_bar,
    )
    clean_gain = qp["scattering"]["gain"] + qp["pair"]["gain"] + photon_gain
    clean_loss_rate = (
        qp["scattering"]["loss_rate"]
        + qp["pair"]["loss_rate"]
        + photon_loss_rate
    )
    clean_R_f = clean_gain - clean_loss_rate * f_state

    contractions = _clean_channel_contractions(
        f_state,
        rho,
        dE,
        k_s_ph,
        k_r_ph,
    )
    clean_a = (
        contractions["emission"]
        + contractions["recombination"] * correction
    )
    clean_b = (
        (contractions["emission"] - contractions["absorption"])
        + (
            contractions["recombination"] * correction
            - contractions["pair_breaking"] * correction
        )
    )
    clean_R_ph = (
        clean_a
        + clean_b * n_state
        + (n_th - n_state) * (1.0 / TAU_L_NS)
    )

    # The residual assemblies cancel strongly (they vanish at a root), so
    # their bounds must scale with the opposing components, not the result.
    qp_component_scale = (
        np.abs(clean_gain)
        + np.abs(clean_loss_rate * f_state)
        + np.abs(arrays[f"qpsim_{label}_R_f_ns_inv"])
    )
    ph_component_scale = (
        np.abs(clean_a)
        + np.abs(clean_b * n_state)
        + np.abs((n_th - n_state) * (1.0 / TAU_L_NS))
        + np.abs(arrays[f"qpsim_{label}_R_ph_ns_inv"])
    )
    r_f_bound = _CORRECTION_GAMMA * qp_component_scale
    r_ph_bound = _CORRECTION_GAMMA * ph_component_scale

    records: dict[str, object] = {}
    flags: list[bool] = []
    if label == "root":
        checks = {
            "qp_gain": (arrays["qpsim_root_qp_gain_ns_inv"], clean_gain, None),
            "qp_loss_rate": (
                arrays["qpsim_root_qp_loss_rate_ns_inv"],
                clean_loss_rate,
                None,
            ),
            "R_f": (arrays["qpsim_root_R_f_ns_inv"], clean_R_f, r_f_bound),
            "R_ph": (arrays["qpsim_root_R_ph_ns_inv"], clean_R_ph, r_ph_bound),
            "a_ph": (arrays["qpsim_root_a_ph_ns_inv"], clean_a, None),
            "b_ph": (arrays["qpsim_root_b_ph_ns_inv"], clean_b, None),
        }
    else:
        checks = {
            "R_f": (arrays["qpsim_frozen_R_f_ns_inv"], clean_R_f, r_f_bound),
            "R_ph": (
                arrays["qpsim_frozen_R_ph_ns_inv"],
                clean_R_ph,
                r_ph_bound,
            ),
        }
    for name, (retained, clean, bound) in checks.items():
        record = _reduction_rounding_check(
            retained,
            clean,
            gamma=_CORRECTION_GAMMA,
            absolute_bound=bound,
        )
        records[name] = record
        flags.append(bool(record["within_rounding_bound"]))
    return clean_R_f, clean_R_ph, records, flags


def build_c7_score(
    c7_bundle_dir: Path,
    *,
    c6_bundle_dir: Path,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c0_bundle_dir: Path,
    c6_score_path: Path = DEFAULT_C6_SCORE,
    c6_receipt_path: Path = DEFAULT_C6_RECEIPT,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
    c0_score_path: Path = DEFAULT_C0_SCORE,
) -> dict[str, Any]:
    """Independently replay and score one formal C7 re-solve bundle."""

    _assert_source_snapshots()
    c7_root, c7_directory_state = _directory_state(
        c7_bundle_dir,
        "selected C7 raw bundle",
    )
    parent = _accept_parent(
        c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c0_bundle_dir=c0_bundle_dir,
        c6_score_path=c6_score_path,
        c6_receipt_path=c6_receipt_path,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
        c0_score_path=c0_score_path,
    )
    raw_metadata, arrays, raw_manifest_sha = load_c7_raw_bundle(c7_root)

    if raw_metadata.get("schema") != RAW_SCHEMA:
        raise C7ScoreError("C7 raw metadata schema is invalid.")
    stage = _mapping(raw_metadata.get("stage"), "C7 raw stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
    }:
        raise C7ScoreError("C7 raw stage metadata is invalid.")
    if not _json_value_bit_exact(
        raw_metadata.get("parent_bindings"),
        _expected_parent_bindings(parent),
    ):
        raise C7ScoreError("C7 raw parent bindings are stale or incomplete.")
    sources = _mapping(raw_metadata.get("sources"), "C7 raw sources")
    if not _json_value_bit_exact(sources, _RAW_SOURCE_HASHES_AT_IMPORT):
        raise C7ScoreError("C7 raw source closure is incomplete, extra, or stale.")
    for relative, digest in sources.items():
        source_path = REPOSITORY_ROOT / relative
        if hashlib.sha256(canonical_source_bytes(source_path)).hexdigest() != digest:
            raise C7ScoreError(f"C7 raw source binding is stale: {relative}.")
    _validate_runtime_record(raw_metadata.get("runtime"), "C7 raw runtime")

    frozen = _expected_frozen_arrays(parent)
    for name, expected in frozen.items():
        if not _array_bit_exact(_positive_zero_copy(expected), arrays[name]):
            raise C7ScoreError(f"C7 frozen parent array {name!r} changed.")
    active = arrays["parent_active_mask"]
    if (
        int(np.count_nonzero(active)) != N_ACTIVE
        or np.any(active[:ACTIVE_START])
        or not np.all(active[ACTIVE_START:])
    ):
        raise C7ScoreError("C7 active support is not the accepted C3 support.")
    seed = arrays["author_seed_state"]
    expected_seed_f = np.zeros(N_QP, dtype=np.float64)
    expected_seed_f[ACTIVE_START:] = seed[:N_AUTHOR]
    expected_seed_n = np.zeros(N_OMEGA, dtype=np.float64)
    expected_seed_n[1 : AUTHOR_OMEGA_STOP + 1] = seed[N_AUTHOR:]
    seed_projection_exact = _array_bit_exact(
        _positive_zero_copy(expected_seed_f), arrays["seed_f"]
    ) and _array_bit_exact(
        _positive_zero_copy(expected_seed_n), arrays["seed_n_ph"]
    )

    root_f = arrays["qpsim_root_f"]
    root_n = arrays["qpsim_root_n_ph"]
    if np.any((root_f < 0.0) | (root_f > 1.0)) or np.any(root_n < 0.0):
        raise C7ScoreError("C7 root state violates physical bounds.")
    n_th = arrays["qpsim_thermal_n_ph"]
    derived_bit_checks = {
        "qpsim_root_R_f_ns_inv": (
            arrays["qpsim_root_qp_gain_ns_inv"]
            - arrays["qpsim_root_qp_loss_rate_ns_inv"] * root_f
        ),
        "qpsim_root_R_ph_ns_inv": (
            arrays["qpsim_root_a_ph_ns_inv"]
            + arrays["qpsim_root_b_ph_ns_inv"] * root_n
            + (n_th - root_n) * (1.0 / TAU_L_NS)
        ),
        "qpsim_root_phonon_escape_net_ns_inv": (n_th - root_n) / TAU_L_NS,
        "qpsim_root_minus_frozen_f": root_f - arrays["parent_f"],
        "qpsim_root_minus_frozen_n_ph": (
            root_n - arrays["parent_projected_n_phonon"]
        ),
        "frozen_qp_residual_delta_s_inv": (
            arrays["qpsim_frozen_R_f_ns_inv"] / SECONDS_PER_NS
            - arrays["parent_qp_residual_s_inv"]
        ),
        "frozen_phonon_residual_delta_support_s_inv": (
            arrays["qpsim_frozen_R_ph_ns_inv"][
                arrays["parent_phonon_to_native_omega_index"]
            ]
            / SECONDS_PER_NS
            - arrays["parent_phonon_residual_s_inv"]
        ),
    }
    for name, expected_value in derived_bit_checks.items():
        if not _array_bit_exact(_positive_zero_copy(expected_value), arrays[name]):
            raise C7ScoreError(f"C7 raw derived array {name!r} is inconsistent.")
    for channel in ("photon", "scattering", "pair"):
        gain = arrays[f"qpsim_root_{channel}_gain_s_inv"]
        loss = arrays[f"qpsim_root_{channel}_loss_s_inv"]
        net = arrays[f"qpsim_root_{channel}_net_s_inv"]
        bound = 4.0 * _FLOAT_EPS * (
            np.abs(net) + np.abs(gain) + np.abs(loss)
        )
        if np.any(
            np.abs(net - (gain - loss))
            > np.maximum(bound, np.finfo(np.float64).tiny)
        ):
            raise C7ScoreError(
                f"C7 root {channel} net is not gain minus loss within "
                "association-order rounding."
            )

    values = _mapping(
        _mapping(parent.c3_metadata.get("parameters"), "C3 parameters").get(
            "values"
        ),
        "C3 parameter values",
    )
    n_bar = _finite_scalar(values.get("n_bar"), "n_bar", positive=True)
    gap_eV = _finite_scalar(values.get("gap_eV"), "gap_eV", positive=True)
    delta0_eV = _finite_scalar(values.get("delta0_eV"), "delta0_eV", positive=True)
    h_eV = _finite_scalar(values.get("h_eV"), "h_eV", positive=True)
    boltzmann = _finite_scalar(
        values.get("boltzmann_constant_J_per_K"),
        "boltzmann constant",
        positive=True,
    )
    electron_charge = _finite_scalar(
        values.get("electron_charge_C"),
        "electron charge",
        positive=True,
    )
    k_b_uev_per_k = boltzmann / electron_charge * 1.0e6
    clean_n_th = _clean_thermal_bose(
        np.arange(N_OMEGA, dtype=np.float64),
        k_b_uev_per_k,
    )
    bose_exponent = np.arange(N_OMEGA, dtype=np.float64) / (
        k_b_uev_per_k * T_BATH_K
    )
    thermal_delta = np.abs(n_th - clean_n_th)
    thermal_bound = (
        32.0
        * _FLOAT_EPS
        * (1.0 + bose_exponent)
        * np.maximum(
            np.abs(n_th) + np.abs(clean_n_th),
            np.finfo(np.float64).tiny,
        )
    )
    if np.any(thermal_delta > thermal_bound):
        raise C7ScoreError("C7 thermal occupation differs from its clean form.")

    correction_inputs: dict[str, Any] = {
        "gap_ueV": _float_record(180.0),
        "tau_0_pb_ns": _float_record(TAU_PB_NS),
    }
    correction = _clean_kaplan_correction(
        arrays["parent_cell_density"],
        arrays["parent_dE_ueV"],
        np.asarray(parent.c6_arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"]),
        correction_inputs,
    )
    reduction_flags: list[bool] = []
    _root_clean_R_f, _root_clean_R_ph, root_records, root_flags = (
        _residual_assembly_verification(
            arrays,
            parent,
            f_state=root_f,
            n_state=root_n,
            n_bar=n_bar,
            correction=correction,
            label="root",
        )
    )
    reduction_flags.extend(root_flags)
    _f_clean_R_f, _f_clean_R_ph, frozen_records, frozen_flags = (
        _residual_assembly_verification(
            arrays,
            parent,
            f_state=arrays["parent_f"],
            n_state=arrays["parent_projected_n_phonon"],
            n_bar=n_bar,
            correction=correction,
            label="frozen",
        )
    )
    reduction_flags.extend(frozen_flags)

    qp_gain = arrays["qpsim_root_qp_gain_ns_inv"]
    qp_loss = arrays["qpsim_root_qp_loss_rate_ns_inv"] * root_f
    qp_balance = _fixed_sum(np.abs(arrays["qpsim_root_R_f_ns_inv"])) / max(
        _fixed_sum(np.abs(qp_gain)) + _fixed_sum(np.abs(qp_loss)),
        np.finfo(np.float64).tiny,
    )
    driven = arrays["qpsim_root_b_ph_ns_inv"] * root_n
    bath = (n_th - root_n) * (1.0 / TAU_L_NS)
    ph_balance = _fixed_sum(np.abs(arrays["qpsim_root_R_ph_ns_inv"])) / max(
        _fixed_sum(np.abs(arrays["qpsim_root_a_ph_ns_inv"]))
        + _fixed_sum(np.abs(driven))
        + _fixed_sum(np.abs(bath)),
        np.finfo(np.float64).tiny,
    )
    frozen_qp_comparison = _operator_comparison(
        arrays["qpsim_frozen_R_f_ns_inv"] / SECONDS_PER_NS,
        arrays["parent_qp_residual_s_inv"],
    )
    frozen_ph_comparison = _operator_comparison(
        arrays["qpsim_frozen_R_ph_ns_inv"][
            arrays["parent_phonon_to_native_omega_index"]
        ]
        / SECONDS_PER_NS,
        arrays["parent_phonon_residual_s_inv"],
    )
    frozen_qp_relative = _float_value(
        frozen_qp_comparison["symmetric_relative_l1"],
        "frozen QP assembly relative",
    )
    frozen_ph_relative = _float_value(
        frozen_ph_comparison["symmetric_relative_l1"],
        "frozen phonon assembly relative",
    )
    root_state_comparison = {
        "f": _operator_comparison(root_f, arrays["parent_f"]),
        "n_ph": _operator_comparison(
            root_n,
            arrays["parent_projected_n_phonon"],
        ),
    }

    observable = _mapping(raw_metadata.get("observable"), "C7 raw observable")
    recorded = {
        name: _float_value(observable.get(name), f"observable.{name}")
        for name in (
            "author_control_ordinate",
            "frozen_root_ordinate_authors",
            "root_driven_gap_eV_authors",
            "root_driven_gap_eV_centers",
            "root_ordinate_authors",
            "root_ordinate_centers",
            "thermal_gap_eV_authors",
            "thermal_gap_eV_centers",
        )
    }
    if recorded["author_control_ordinate"] != AUTHOR_CONTROL_ORDINATE:
        raise C7ScoreError("C7 author-control ordinate record is invalid.")
    active_slice = slice(ACTIVE_START, N_QP)
    clean_driven = _clean_author_gap(
        root_f[active_slice],
        gap_eV=gap_eV,
        h_eV=h_eV,
        delta0_eV=delta0_eV,
    )
    clean_thermal = _clean_author_gap(
        arrays["parent_thermal_f"][active_slice],
        gap_eV=gap_eV,
        h_eV=h_eV,
        delta0_eV=delta0_eV,
    )
    clean_frozen_driven = _clean_author_gap(
        arrays["parent_f"][active_slice],
        gap_eV=gap_eV,
        h_eV=h_eV,
        delta0_eV=delta0_eV,
    )
    observable_checks = {
        "root_driven_gap": (recorded["root_driven_gap_eV_authors"], clean_driven),
        "thermal_gap": (recorded["thermal_gap_eV_authors"], clean_thermal),
    }
    observable_records: dict[str, object] = {}
    observable_ok = True
    for name, (recorded_value, clean_value) in observable_checks.items():
        gap_relative = abs(recorded_value - clean_value) / max(
            abs(recorded_value) + abs(clean_value),
            float(np.finfo(np.float64).tiny),
        )
        observable_records[name] = {
            "clean_transcription_eV": _float_record(clean_value),
            "recorded_eV": _float_record(recorded_value),
            "symmetric_relative": _float_record(gap_relative),
        }
        observable_ok = observable_ok and gap_relative <= _OBSERVABLE_RELATIVE_LIMIT
    clean_frozen_ordinate = (clean_frozen_driven - clean_thermal) / (
        delta0_eV - clean_thermal
    )
    frozen_ordinate_relative = abs(
        clean_frozen_ordinate - AUTHOR_CONTROL_ORDINATE
    ) / AUTHOR_CONTROL_ORDINATE
    observable_records["frozen_root_ordinate"] = {
        "clean_transcription": _float_record(clean_frozen_ordinate),
        "recorded": _float_record(recorded["frozen_root_ordinate_authors"]),
        "relative_to_author_control": _float_record(frozen_ordinate_relative),
    }
    observable_ok = (
        observable_ok
        and frozen_ordinate_relative <= _OBSERVABLE_RELATIVE_LIMIT
        and recorded["frozen_root_ordinate_authors"] == AUTHOR_CONTROL_ORDINATE
    )
    expected_root_ordinate = (
        recorded["root_driven_gap_eV_authors"]
        - recorded["thermal_gap_eV_authors"]
    ) / (delta0_eV - recorded["thermal_gap_eV_authors"])
    if (
        np.float64(expected_root_ordinate).view(np.uint64)
        != np.float64(recorded["root_ordinate_authors"]).view(np.uint64)
    ):
        raise C7ScoreError("C7 root ordinate is not derived from its gap records.")

    solver_contract = _mapping(raw_metadata.get("solver_contract"), "solver contract")
    accepted_cap = _strict_int(
        solver_contract.get("accepted_max_iter"),
        "solver accepted_max_iter",
        minimum=1,
    )
    declared_cap = _strict_int(
        solver_contract.get("declared_max_iter"),
        "solver declared_max_iter",
        minimum=accepted_cap,
    )
    if declared_cap != MAX_ITER or solver_contract.get("analytic_cross") is not True:
        raise C7ScoreError("C7 solver configuration is invalid.")
    if (
        _float_value(solver_contract.get("step_rtol"), "solver step_rtol")
        != STEP_RTOL
        or _float_value(solver_contract.get("tau_l_ns"), "solver tau_l") != TAU_L_NS
        or _float_value(solver_contract.get("omega_0_ueV"), "solver omega_0")
        != OMEGA_0_UEV
        or _float_value(solver_contract.get("c_phot_ns_inv"), "solver c_phot")
        != C_PHOT_NS_INV
        or _float_value(solver_contract.get("n_bar"), "solver n_bar") != n_bar
    ):
        raise C7ScoreError("C7 solver scalar inputs are invalid.")
    probe = solver_contract.get("iteration_probe")
    if not isinstance(probe, list) or len(probe) != accepted_cap:
        raise C7ScoreError("C7 iteration probe is malformed.")
    for index, entry in enumerate(probe):
        record = _mapping(entry, f"iteration probe[{index}]")
        _exact_keys(record, {"converged", "exception", "max_iter"}, "probe entry")
        if record.get("max_iter") != index + 1:
            raise C7ScoreError("C7 iteration probe caps are not consecutive.")
        expected_converged = index == accepted_cap - 1
        if record.get("converged") is not expected_converged:
            raise C7ScoreError("C7 iteration probe convergence flags are invalid.")
        if expected_converged and record.get("exception") is not None:
            raise C7ScoreError("C7 accepted probe entry carries an exception.")
        if not expected_converged and not isinstance(record.get("exception"), str):
            raise C7ScoreError("C7 failed probe entry lacks its exception class.")
    certificates = _mapping(
        solver_contract.get("balance_certificates"),
        "solver balance certificates",
    )
    if _float_value(certificates.get("limit"), "certificate limit") != STEP_RTOL:
        raise C7ScoreError("C7 balance-certificate limit is invalid.")

    all_checks = {
        "all_clean_assemblies_within_predeclared_rounding_bound": all(
            reduction_flags
        ),
        "frozen_phonon_assembly_matches_accepted_residual": (
            frozen_ph_relative <= _FROZEN_DELTA_LIMIT
        ),
        "frozen_qp_assembly_matches_accepted_residual": (
            frozen_qp_relative <= _FROZEN_DELTA_LIMIT
        ),
        "frozen_root_reproduces_author_ordinate": (
            recorded["frozen_root_ordinate_authors"] == AUTHOR_CONTROL_ORDINATE
        ),
        "iteration_probe_consistent": accepted_cap <= MAX_ITER,
        "observable_transcription_matches": observable_ok,
        "root_phonon_balance_within_step_rtol": ph_balance < STEP_RTOL,
        "root_qp_balance_within_step_rtol": qp_balance < STEP_RTOL,
        "seed_projection_bit_exact": seed_projection_exact,
    }
    if not all(all_checks.values()):
        failed = sorted(name for name, passed in all_checks.items() if not passed)
        raise C7ScoreError(f"C7 acceptance checks failed: {failed}.")

    score: dict[str, Any] = {
        "acceptance": {
            "all_passed": True,
            "checks": all_checks,
            "limits": {
                "frozen_assembly_symmetric_relative_l1": _float_record(
                    _FROZEN_DELTA_LIMIT
                ),
                "observable_symmetric_relative": _float_record(
                    _OBSERVABLE_RELATIVE_LIMIT
                ),
                "root_balance_l1_ratio": _float_record(STEP_RTOL),
            },
            "ungated_recorded_differences": {
                "root_vs_frozen_state": (
                    "the re-solved root legitimately moves away from the "
                    "frozen author root; the state and ordinate changes are "
                    "recorded results of the accumulated component "
                    "substitutions, not gated errors"
                ),
            },
        },
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "channel_verification": {
            "frozen_state": frozen_records,
            "root_state": root_records,
        },
        "comparison": {
            "frozen_phonon_assembly_vs_accepted_residual": frozen_ph_comparison,
            "frozen_qp_assembly_vs_accepted_residual": frozen_qp_comparison,
            "root_balance": {
                "ph_l1_ratio": _float_record(ph_balance),
                "qp_l1_ratio": _float_record(qp_balance),
            },
            "root_vs_frozen_state": root_state_comparison,
        },
        "component_locality": {
            "qpsim_root_f": _array_descriptor(root_f),
            "qpsim_root_n_ph": _array_descriptor(root_n),
            "raw_contract": dict(
                _mapping(raw_metadata["component_locality"], "component locality")
            ),
        },
        "conservation": dict(
            _mapping(raw_metadata["conservation"], "conservation")
        ),
        "contracts": {
            "comparison_contract": dict(
                _mapping(raw_metadata["comparison_contract"], "comparison contract")
            ),
        },
        "limitations": dict(_mapping(raw_metadata["limitations"], "limitations")),
        "observable": {
            "producer": dict(observable),
            "verifier": observable_records,
        },
        "parent_bindings": _expected_parent_bindings(parent),
        "raw_bundle": {
            "manifest_sha256": raw_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "rounding_contract": {
            "comparison": (
                "Retained root and residual arrays are manifest-byte-bound. "
                "Independent science uses fixed-order math.fsum and an "
                "elementwise gamma bound; the LAPACK Newton path is not "
                "re-run."
            ),
            "correction_gamma": _float_record(_CORRECTION_GAMMA),
            "float64_epsilon": _float_record(_FLOAT_EPS),
            "gamma": _float_record(_REDUCTION_GAMMA),
        },
        "runtime": {
            "producer_public_array_generation": dict(
                _mapping(raw_metadata["runtime"], "producer runtime")
            ),
            "verifier_contract": dict(_VERIFIER_RUNTIME_CONTRACT),
        },
        "schema": SCHEMA,
        "seed_binding": dict(
            _mapping(raw_metadata["seed_binding"], "seed binding")
        ),
        "solver_contract": dict(solver_contract),
        "source_binding": {
            **_EXPECTED_SOURCE_BINDING,
        },
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
        c7_root,
        c7_directory_state,
        "selected C7 raw bundle",
    )
    _assert_source_snapshots()
    return score


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Return canonical checked-score JSON after strict structure validation."""

    _validate_score_structure(score)
    return _canonical_json_bytes(score)


def _evidence_digest(score: dict[str, Any]) -> str:
    keys = tuple(sorted(_SCORE_KEYS - {"sources"}))
    evidence = {key: score[key] for key in keys}
    return hashlib.sha256(_canonical_json_bytes(evidence)).hexdigest()


def _validate_canonical_pins(score: dict[str, Any]) -> None:
    raw = _mapping(score.get("raw_bundle"), "C7 pinned raw bundle")
    if raw.get("manifest_sha256") != _EXPECTED_RAW_MANIFEST_SHA256:
        raise C7ScoreError("C7 score does not bind the accepted canonical raw manifest.")
    observable = _mapping(score.get("observable"), "C7 pinned observable")
    producer_observable = _mapping(
        observable.get("producer"),
        "C7 pinned producer observable",
    )
    comparison = _mapping(score.get("comparison"), "C7 pinned comparison")
    balance = _mapping(comparison.get("root_balance"), "C7 pinned balance")
    actual_metrics = {
        "frozen_phonon_assembly_symmetric_relative_l1": _float_value(
            _mapping(
                comparison.get("frozen_phonon_assembly_vs_accepted_residual"),
                "pinned frozen phonon",
            ).get("symmetric_relative_l1"),
            "pinned frozen phonon relative",
        ),
        "frozen_qp_assembly_symmetric_relative_l1": _float_value(
            _mapping(
                comparison.get("frozen_qp_assembly_vs_accepted_residual"),
                "pinned frozen QP",
            ).get("symmetric_relative_l1"),
            "pinned frozen QP relative",
        ),
        "root_ordinate_authors": _float_value(
            producer_observable.get("root_ordinate_authors"),
            "pinned root ordinate",
        ),
        "root_ordinate_centers": _float_value(
            producer_observable.get("root_ordinate_centers"),
            "pinned centers ordinate",
        ),
        "root_ph_balance_l1_ratio": _float_value(
            balance.get("ph_l1_ratio"),
            "pinned phonon balance",
        ),
        "root_qp_balance_l1_ratio": _float_value(
            balance.get("qp_l1_ratio"),
            "pinned QP balance",
        ),
    }
    if not _json_value_bit_exact(actual_metrics, _EXPECTED_CANONICAL_METRICS):
        raise C7ScoreError("C7 canonical numerical metric pins do not match.")
    solver = _mapping(score.get("solver_contract"), "C7 pinned solver contract")
    actual_ints = {
        "accepted_max_iter": solver.get("accepted_max_iter"),
    }
    if actual_ints != _EXPECTED_CANONICAL_INTS:
        raise C7ScoreError("C7 canonical integer pins do not match.")
    locality = _mapping(score.get("component_locality"), "C7 pinned locality")
    for name, expected_sha in _EXPECTED_ROOT_NPY_SHA256.items():
        descriptor = _mapping(locality.get(name), f"C7 pinned locality.{name}")
        if descriptor.get("npy_sha256") != expected_sha:
            raise C7ScoreError(f"C7 canonical root pin {name!r} does not match.")
    if _evidence_digest(score) != _EXPECTED_EVIDENCE_DIGEST:
        raise C7ScoreError("C7 canonical numeric/semantic evidence digest does not match.")


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
            raise C7ScoreError(f"{label}.{field} must be a nonempty string.")
    threads = _mapping(runtime["thread_environment"], f"{label}.thread_environment")
    if threads != _THREAD_ENVIRONMENT:
        raise C7ScoreError(f"{label}.thread_environment is invalid.")


def _validate_score_structure(score: dict[str, Any]) -> None:
    _exact_keys(score, _SCORE_KEYS, "C7 score")
    if score.get("schema") != SCHEMA:
        raise C7ScoreError("C7 score schema is unsupported.")
    stage = _mapping(score.get("stage"), "C7 score stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C7ScoreError("C7 score stage is invalid.")
    if not _json_value_bit_exact(score.get("sources"), _SOURCE_HASHES_AT_IMPORT):
        raise C7ScoreError("C7 score source closure is invalid.")
    runtime = _mapping(score.get("runtime"), "C7 score runtime")
    _exact_keys(
        runtime,
        {"producer_public_array_generation", "verifier_contract"},
        "C7 score runtime",
    )
    _validate_runtime_record(
        runtime.get("producer_public_array_generation"),
        "C7 score producer runtime",
    )
    if not _json_value_bit_exact(
        runtime.get("verifier_contract"),
        _VERIFIER_RUNTIME_CONTRACT,
    ):
        raise C7ScoreError("C7 score verifier runtime contract is invalid.")
    raw = _mapping(score.get("raw_bundle"), "C7 score raw bundle")
    _exact_keys(raw, {"manifest_sha256", "schema"}, "C7 score raw bundle")
    if raw.get("schema") != RAW_SCHEMA:
        raise C7ScoreError("C7 score raw schema is invalid.")
    _sha256(raw.get("manifest_sha256"), "C7 score raw manifest SHA-256")
    descriptors = _mapping(score.get("array_descriptors"), "C7 score descriptors")
    if set(descriptors) != _EXPECTED_ARRAY_NAMES:
        raise C7ScoreError("C7 score descriptor closure is invalid.")
    acceptance = _mapping(score.get("acceptance"), "C7 acceptance")
    _exact_keys(
        acceptance,
        {"all_passed", "checks", "limits", "ungated_recorded_differences"},
        "C7 acceptance",
    )
    checks = _mapping(acceptance.get("checks"), "C7 acceptance checks")
    _exact_keys(
        checks,
        {
            "all_clean_assemblies_within_predeclared_rounding_bound",
            "frozen_phonon_assembly_matches_accepted_residual",
            "frozen_qp_assembly_matches_accepted_residual",
            "frozen_root_reproduces_author_ordinate",
            "iteration_probe_consistent",
            "observable_transcription_matches",
            "root_phonon_balance_within_step_rtol",
            "root_qp_balance_within_step_rtol",
            "seed_projection_bit_exact",
        },
        "C7 acceptance checks",
    )
    if any(value is not True for value in checks.values()):
        raise C7ScoreError("C7 score contains a failed or malformed acceptance check.")
    if acceptance.get("all_passed") is not True:
        raise C7ScoreError("C7 score acceptance is false.")
    parent = _mapping(score.get("parent_bindings"), "C7 score parent bindings")
    for field in (
        "c0_raw_manifest_sha256",
        "c0_score_sha256",
        "c2_raw_manifest_sha256",
        "c3_raw_manifest_sha256",
        "c4_raw_manifest_sha256",
        "c5_raw_manifest_sha256",
        "c6_raw_manifest_sha256",
        "c6_receipt_sha256",
        "c6_score_sha256",
    ):
        _sha256(parent.get(field), f"C7 score parent_bindings.{field}")
    for required_object in (
        "channel_verification",
        "comparison",
        "component_locality",
        "conservation",
        "contracts",
        "limitations",
        "observable",
        "rounding_contract",
        "seed_binding",
        "solver_contract",
        "units",
    ):
        if not _mapping(score.get(required_object), f"C7 score {required_object}"):
            raise C7ScoreError(f"C7 score {required_object} is empty.")
    if not _json_value_bit_exact(
        score.get("source_binding"),
        _EXPECTED_SOURCE_BINDING,
    ):
        raise C7ScoreError("C7 score source_binding is invalid.")
    _validate_canonical_pins(score)


def _load_c7_score_unbound(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file_once(path, "checked C7 score")
    score = _parse_json(raw, "checked C7 score")
    if raw != canonical_score_bytes(score):
        raise C7ScoreError("Checked C7 score is not canonical JSON.")
    return score, raw


def _receipt_parent_from_score(score: dict[str, Any]) -> dict[str, object]:
    parent = _mapping(score.get("parent_bindings"), "C7 score parent bindings")
    return {
        "raw_manifest_sha256": parent.get("c6_raw_manifest_sha256"),
        "raw_schema": parent.get("c6_raw_schema"),
        "receipt_file_sha256": parent.get("c6_receipt_sha256"),
        "receipt_schema": parent.get("c6_receipt_schema"),
        "score_file_sha256": parent.get("c6_score_sha256"),
        "score_schema": parent.get("c6_score_schema"),
    }


def load_c7_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Strictly load the repository C7 score/raw/C6 trust anchor."""

    raw = _read_regular_file_once(path, "C7 raw-manifest receipt")
    receipt = _parse_json(raw, "C7 raw-manifest receipt")
    if raw != _canonical_json_bytes(receipt):
        raise C7ScoreError("C7 raw-manifest receipt is not canonical JSON.")
    _exact_keys(
        receipt,
        {"checked_score", "parent_c6", "qualification", "raw_bundle", "schema"},
        "C7 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C7ScoreError("C7 raw-manifest receipt schema is unsupported.")
    if receipt.get("qualification") != (
        "Repository trust anchor for the externally retained C7 raw manifest, "
        "the complete canonical checked-score bytes, and the independently "
        "replayed C6/C5/C4/C3/C2 parent chain; it does not contain or replace "
        "the raw arrays."
    ):
        raise C7ScoreError("C7 raw-manifest receipt qualification is invalid.")
    checked = _mapping(receipt.get("checked_score"), "C7 receipt checked_score")
    _exact_keys(checked, {"file_sha256", "schema"}, "C7 receipt checked_score")
    if checked.get("schema") != SCHEMA:
        raise C7ScoreError("C7 receipt score schema is invalid.")
    _sha256(checked.get("file_sha256"), "C7 receipt score SHA-256")
    raw_bundle = _mapping(receipt.get("raw_bundle"), "C7 receipt raw bundle")
    _exact_keys(raw_bundle, {"manifest_sha256", "schema"}, "C7 receipt raw bundle")
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C7ScoreError("C7 receipt raw schema is invalid.")
    _sha256(raw_bundle.get("manifest_sha256"), "C7 receipt raw manifest SHA-256")
    parent = _mapping(receipt.get("parent_c6"), "C7 receipt parent C6")
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
        "C7 receipt parent C6",
    )
    if (
        parent.get("raw_schema") != C6_RAW_SCHEMA
        or parent.get("receipt_schema") != C6_RECEIPT_SCHEMA
        or parent.get("score_schema") != C6_SCORE_SCHEMA
    ):
        raise C7ScoreError("C7 receipt C6 schemas are invalid.")
    for field in (
        "raw_manifest_sha256",
        "receipt_file_sha256",
        "score_file_sha256",
    ):
        _sha256(parent.get(field), f"C7 receipt parent_c6.{field}")
    return receipt


def load_c7_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Load a checked C7 score and bind it to canonical C6 anchors."""

    score, score_raw = _load_c7_score_unbound(path)
    receipt = load_c7_receipt(receipt_path)
    checked = _mapping(receipt.get("checked_score"), "C7 receipt checked score")
    if hashlib.sha256(score_raw).hexdigest() != checked.get("file_sha256"):
        raise C7ScoreError("Checked C7 score bytes do not match its receipt.")
    if score.get("raw_bundle") != receipt.get("raw_bundle"):
        raise C7ScoreError("Checked C7 raw binding does not match its receipt.")
    if _receipt_parent_from_score(score) != receipt.get("parent_c6"):
        raise C7ScoreError("Checked C7 C6 binding does not match its receipt.")
    parent = _mapping(score.get("parent_bindings"), "checked C7 parent bindings")
    expected_score_path = DEFAULT_C6_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
    expected_receipt_path = DEFAULT_C6_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
    if (
        parent.get("c6_score_path") != expected_score_path
        or parent.get("c6_receipt_path") != expected_receipt_path
    ):
        raise C7ScoreError("Checked C7 does not bind canonical C6 parent paths.")
    c6_score_path, c6_score_bytes = _repository_file_snapshot(
        DEFAULT_C6_SCORE,
        "canonical C6 score",
    )
    c6_receipt_path, c6_receipt_bytes = _repository_file_snapshot(
        DEFAULT_C6_RECEIPT,
        "canonical C6 receipt",
    )
    accepted_c6 = load_c6_score(c6_score_path, receipt_path=c6_receipt_path)
    accepted_raw = _mapping(accepted_c6.get("raw_bundle"), "accepted C6 raw")
    if (
        hashlib.sha256(c6_score_bytes).hexdigest() != parent.get("c6_score_sha256")
        or hashlib.sha256(c6_receipt_bytes).hexdigest()
        != parent.get("c6_receipt_sha256")
        or accepted_raw.get("manifest_sha256")
        != parent.get("c6_raw_manifest_sha256")
    ):
        raise C7ScoreError("Checked C7 canonical C6 binding is stale.")
    _assert_file_snapshot(c6_score_path, c6_score_bytes, "canonical C6 score")
    _assert_file_snapshot(c6_receipt_path, c6_receipt_bytes, "canonical C6 receipt")
    return score


def build_c7_receipt(
    score_path: Path = DEFAULT_SCORE,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a receipt only after independently reproducing C7 score bytes."""

    checked_score_path, checked_score_snapshot = _repository_file_snapshot(
        score_path,
        "checked C7 score for receipt",
    )
    score, score_raw = _load_c7_score_unbound(checked_score_path)
    if score_raw != checked_score_snapshot:
        raise C7ScoreError("Checked C7 score changed before receipt replay.")
    rebuilt = build_c7_score(**kwargs)
    if canonical_score_bytes(rebuilt) != score_raw:
        raise C7ScoreError(
            "C7 receipt refuses score bytes that do not independently reproduce "
            "from the selected C7/C6/C5/C4/C3/C2 raw evidence."
        )
    _assert_file_snapshot(
        checked_score_path,
        checked_score_snapshot,
        "checked C7 score for receipt",
    )
    return {
        "checked_score": {
            "file_sha256": hashlib.sha256(score_raw).hexdigest(),
            "schema": SCHEMA,
        },
        "parent_c6": _receipt_parent_from_score(score),
        "qualification": (
            "Repository trust anchor for the externally retained C7 raw "
            "manifest, the complete canonical checked-score bytes, and the "
            "independently replayed C6/C5/C4/C3/C2 parent chain; it does not "
            "contain or replace the raw arrays."
        ),
        "raw_bundle": dict(
            _mapping(score.get("raw_bundle"), "C7 score raw bundle")
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
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"C7 output already exists: {target}")
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


def write_c7_score(output_path: Path, **kwargs: Any) -> Path:
    score = build_c7_score(**kwargs)
    return _atomic_exclusive_write(output_path, canonical_score_bytes(score))


def write_c7_receipt(
    output_path: Path,
    *,
    score_path: Path = DEFAULT_SCORE,
    **kwargs: Any,
) -> Path:
    receipt = build_c7_receipt(score_path, **kwargs)
    return _atomic_exclusive_write(output_path, _canonical_json_bytes(receipt))


def _add_parent_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--c7-bundle", type=Path, required=True)
    parser.add_argument("--c6-bundle", type=Path, required=True)
    parser.add_argument("--c5-bundle", type=Path, required=True)
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c0-bundle", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="build the checked C7 score")
    _add_parent_arguments(score)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    receipt = subparsers.add_parser("receipt", help="build the C7 receipt")
    _add_parent_arguments(receipt)
    receipt.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    receipt.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    common = {
        "c7_bundle_dir": args.c7_bundle,
        "c6_bundle_dir": args.c6_bundle,
        "c5_bundle_dir": args.c5_bundle,
        "c4_bundle_dir": args.c4_bundle,
        "c3_bundle_dir": args.c3_bundle,
        "c2_bundle_dir": args.c2_bundle,
        "c0_bundle_dir": args.c0_bundle,
    }
    if args.command == "receipt":
        result = write_c7_receipt(
            args.output,
            score_path=args.score,
            **common,
        )
    else:
        result = write_c7_score(args.output, **common)
    print(result)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
