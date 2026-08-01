"""Generate and validate the checked Fischer Figure 6 staged-solve pilot.

The numerical arrays remain external development artifacts.  This module
strictly verifies one staged-resolve bundle and extracts a compact checked JSON
record that binds the exact input anchor, numerical source closure, convergence
certificates, observables, and native final-state hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.observables.gap_suppression import (
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
)

from validation.fischer_2023.fig6_author_staged_resolve import (
    AUTHOR_CONTROL,
    C3A,
    C3B,
    C3C,
    PAIR_LABELS_ONLY,
    PRIMARY_STAGES,
    SOURCE_BINDING,
    StageSpecification,
    build_staged_operator,
    evaluate_staged_system,
)
from validation.fischer_2023.fig6_author_staged_resolve import (
    SCHEMA as RAW_SCHEMA,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    author_newton_delta,
)
from validation.source_provenance import source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PILOT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "staged-resolve-pilot.json"
)
PILOT_SCHEMA = "qpsim.fischer2023.fig6-author-staged-resolve-pilot.v4"
PILOT_STATUS = "supplemental_pilot_evidence_not_formal_ladder_completion"
EXPECTED_ANCHOR_PATH = (
    "validation/paper_data/fischer_2023/fig6/"
    "author-point-T020-sweep049-exact-anchor.json"
)
EXPECTED_ANCHOR_SHA256 = (
    "3acfa5973a5766a29f23a6bd85704655462e4bc587f232e0576db04c17725010"
)
EXPECTED_INPUT_MANIFEST_SHA256 = (
    "13afe789ed624504336dd969bfaf42bacb78ab16ffe7d8b953aac8d8b1f42aab"
)
_EXPECTED_PRODUCER_VALIDATION_SOURCES = (
    REPOSITORY_ROOT / "validation" / "author_source.py",
    REPOSITORY_ROOT / "validation" / "reproduction_ladder.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_adapter.py",
    REPOSITORY_ROOT
    / "validation"
    / "fischer_2023"
    / "fig6_author_frozen_state.py",
    REPOSITORY_ROOT
    / "validation"
    / "fischer_2023"
    / "fig6_author_staged_resolve.py",
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig6_author_c0.py",
)
PRIMARY_STAGE_IDS = {
    "author_control",
    "c3a_finite_volume_coherence",
    "c3b_center_pair_labels",
    "c3c_native_cell_density",
    "pair_labels_only",
}
ALL_RESULT_IDS = {
    *PRIMARY_STAGE_IDS,
    "c3b_continuation_from_c3a",
    "c3c_continuation_from_c3b",
}
EXPECTED_STAGE_SPECIFICATIONS = {
    "author_control": AUTHOR_CONTROL,
    "pair_labels_only": PAIR_LABELS_ONLY,
    "c3a_finite_volume_coherence": C3A,
    "c3b_center_pair_labels": C3B,
    "c3b_continuation_from_c3a": C3B,
    "c3c_native_cell_density": C3C,
    "c3c_continuation_from_c3b": C3C,
}
EXPECTED_SOLVE_PATHS = {
    "author_control": "independent_from_captured_author_initial_state",
    "pair_labels_only": "independent_from_captured_author_initial_state",
    "c3a_finite_volume_coherence": "independent_from_captured_author_initial_state",
    "c3b_center_pair_labels": "independent_from_captured_author_initial_state",
    "c3b_continuation_from_c3a": "continuation_from_c3a_final_state",
    "c3c_native_cell_density": "independent_from_captured_author_initial_state",
    "c3c_continuation_from_c3b": "continuation_from_c3b_final_state",
}
EXPECTED_PARAMETERS: dict[str, int | float] = {
    "T_c_K": 1.184,
    "c_photon_s_inv": 1.0000352684367704,
    "gap_eV": 0.00017999999999999998,
    "h_eV": 1.0e-6,
    "max_newton_steps": 10,
    "n_bar": 954548.4566618347,
    "num_energy_cells": 1620,
    "photon_bin": 20,
    "relative_step_threshold": 1.0e-7,
    "tau_0_pb_s": 2.5533602006137212e-10,
    "tau_0_s": 4.3800000000000003e-7,
    "tau_l_s": 2.55e-10,
    "temperature_K": 0.2,
}
EXPECTED_COORDINATE_CONTRACT = {
    "carrier_occupation_samples": "author left edges E_i=Delta+i*h",
    "density_of_states": "exact author cell average over [E_i,E_i+h]",
    "finite_volume_coherence": (
        "qpsim SpectralContext exact double-cell-average K_plus/K_minus"
    ),
    "native_density_of_states": (
        "C3c uses the same qpsim SpectralContext's native-micro-eV "
        "cell_density without algebraic re-evaluation in eV"
    ),
    "pair_frequency_offset_meaning": (
        "offset changes the n lookup, represented phonon aggregation, "
        "and the pair-frequency-squared QP rate factor together"
    ),
    "phonon_support": "h through (N-1)h; higher occupations fixed to zero",
}
EXPECTED_DIAGNOSTIC_ID = "fischer-catelani-2023-fig6-c3-staged-resolve-v3"
EXPECTED_PILOT_ID = "fischer-catelani-2023-fig6-c3-single-point-pilot-v4"
EXPECTED_LIMITATIONS = {
    "scope": "one anchored Figure 6 point; no 300-point sweep",
    "seed_sensitivity": (
        "all primary stages use one identical captured author initial state; "
        "this does not survey independent random seeds"
    ),
    "solver": (
        "author explicit matrix inverse and no damping/clipping are retained, "
        "including the authenticated photon endpoint, photon off-diagonal, "
        "and phonon-pair DnN residual/Jacobian mismatches"
    ),
}
EXPECTED_SOLVER_CONTRACT = {
    "linear_algebra": "np.linalg.inv(Dx) followed by matrix-vector product",
    "primary_stage_initialization": "same captured author initial state",
    "state_constraints": "none; no clipping or line search",
    "stopping_rule": "relative L2 norm of quasiparticle Newton step",
}
_CHANNEL_ARRAY_SUFFIXES = {
    "phonon_escape_s_inv",
    "phonon_pair_s_inv",
    "phonon_scattering_s_inv",
    "qp_pair_s_inv",
    "qp_photon_s_inv",
    "qp_scattering_s_inv",
    "residual_s_inv",
}
_RESULT_ARRAY_SUFFIXES = {
    *_CHANNEL_ARRAY_SUFFIXES,
    "final_state",
    "linear_solve_backward_errors",
    "newton_deltas",
    "relative_qp_steps",
    "state_history",
}
_RAW_METADATA_KEYS = {
    "array_descriptors",
    "author_control_reproduction",
    "coordinate_contract",
    "diagnostic_id",
    "input_bundle",
    "limitations",
    "parameters",
    "path_dependence",
    "schema",
    "source_binding",
    "solver_contract",
    "sources",
    "stages",
}
_PILOT_KEYS = {
    "author_control_reproduction",
    "final_states",
    "input_bundle",
    "limitations",
    "numerical_producer_sources",
    "parameters",
    "path_dependence",
    "pilot_id",
    "qualification",
    "raw_array_policy",
    "schema",
    "source_binding",
    "staged_bundle",
    "stages",
    "status",
    "summary_generator",
}
_SUMMARY_STAGE_KEYS = {
    "cancellation_certificate",
    "converged",
    "final_bounds",
    "last_relative_qp_step",
    "newton_iterations",
    "observable",
    "solve_path",
    "specification",
    "termination",
}
_RAW_STAGE_KEYS = {*_SUMMARY_STAGE_KEYS, "max_linear_solve_backward_error"}


class StagedResolvePilotError(ValueError):
    """A raw staged bundle or checked pilot record is malformed."""


@dataclass(frozen=True)
class _RawBundleSnapshot:
    """One immutable in-memory view of a verified staged bundle."""

    manifest: dict[str, Any]
    manifest_sha256: str
    arrays: dict[str, np.ndarray]
    anchor: dict[str, Any]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StagedResolvePilotError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise StagedResolvePilotError(f"Non-finite JSON number {value!r} is forbidden.")


def _load_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StagedResolvePilotError(f"Cannot parse {label}: {exc}") from exc
    return _mapping(payload, label)


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StagedResolvePilotError(f"{label} must be an object.")
    return value


def _exact_keys(
    value: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise StagedResolvePilotError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, "
            f"got {sorted(value)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise StagedResolvePilotError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise StagedResolvePilotError(f"{label} must be a nonempty string.")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise StagedResolvePilotError(f"{label} must be finite and real.")
    result = float(value)
    if not math.isfinite(result):
        raise StagedResolvePilotError(f"{label} must be finite and real.")
    return result


def _safe_filename(value: object, label: str) -> str:
    name = _string(value, label)
    candidate = Path(name)
    if candidate.name != name or name in {".", ".."}:
        raise StagedResolvePilotError(f"{label} must be a safe basename.")
    return name


def _descriptor(value: object, label: str) -> dict[str, Any]:
    descriptor = _mapping(value, label)
    _exact_keys(descriptor, {"dtype", "npy_sha256", "shape"}, label)
    _string(descriptor.get("dtype"), f"{label}.dtype")
    _sha256(descriptor.get("npy_sha256"), f"{label}.npy_sha256")
    shape = descriptor.get("shape")
    if (
        not isinstance(shape, list)
        or not shape
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in shape
        )
    ):
        raise StagedResolvePilotError(f"{label}.shape must contain positive integers.")
    return descriptor


def _native_descriptor(array: np.ndarray) -> dict[str, object]:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(array),
        version=(3, 0),
        allow_pickle=False,
    )
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "shape": [int(size) for size in array.shape],
    }


def _specification_record(specification: StageSpecification) -> dict[str, object]:
    return {
        "changed_convention": specification.changed_convention,
        "coherence_convention": specification.coherence_convention,
        "density_convention": specification.density_convention,
        "pair_frequency_offset_bins": specification.pair_frequency_offset_bins,
        "parent_stage_id": specification.parent_stage_id,
        "stage_id": specification.stage_id,
    }


def _assert_same_array(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise StagedResolvePilotError(f"{label} has the wrong shape or dtype.")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=False):
        difference = float(np.max(np.abs(actual - expected), initial=0.0))
        raise StagedResolvePilotError(
            f"{label} does not match the recomputed value "
            f"(max absolute difference {difference:.3e})."
        )


def _symmetric_relative(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = float(np.linalg.norm(candidate - reference))
    scale = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(float).tiny,
    )
    return difference / scale


def _assert_same_tree(actual: object, expected: object, label: str) -> None:
    """Compare metadata with independently derived semantics."""

    if isinstance(expected, dict):
        actual_mapping = _mapping(actual, label)
        _exact_keys(actual_mapping, set(expected), label)
        for key, expected_value in expected.items():
            _assert_same_tree(actual_mapping[key], expected_value, f"{label}.{key}")
        return
    if isinstance(expected, bool):
        if actual is not expected:
            raise StagedResolvePilotError(f"{label} does not match recomputed truth.")
        return
    if isinstance(expected, str):
        if actual != expected:
            raise StagedResolvePilotError(f"{label} does not match the required value.")
        return
    if isinstance(expected, (int, np.integer)) and not isinstance(expected, bool):
        if isinstance(actual, bool) or not isinstance(actual, (int, np.integer)):
            raise StagedResolvePilotError(f"{label} must be an integer.")
        if int(actual) != int(expected):
            raise StagedResolvePilotError(f"{label} does not match recomputed truth.")
        return
    if isinstance(expected, (float, np.floating)):
        actual_float = _finite(actual, label)
        expected_float = float(expected)
        if not math.isclose(
            actual_float,
            expected_float,
            rel_tol=5.0e-13,
            abs_tol=5.0e-300,
        ):
            raise StagedResolvePilotError(
                f"{label} does not match recomputed truth: "
                f"{actual_float!r} != {expected_float!r}."
            )
        return
    if actual != expected:
        raise StagedResolvePilotError(f"{label} does not match recomputed truth.")


def _validate_sources(value: object, label: str) -> dict[str, Any]:
    sources = _mapping(value, label)
    if not sources:
        raise StagedResolvePilotError(f"{label} must not be empty.")
    for path, digest in sources.items():
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise StagedResolvePilotError(f"{label} contains an unsafe path.")
        _sha256(digest, f"{label}[{path!r}]")
    return sources


def _authenticate_live_sources(value: object, label: str) -> dict[str, Any]:
    """Require the exact conservative producer closure, with live digests."""

    sources = _validate_sources(value, label)
    qpsim_root = REPOSITORY_ROOT / "qpsim"
    expected_paths = {
        *_EXPECTED_PRODUCER_VALIDATION_SOURCES,
        *qpsim_root.rglob("*.py"),
        *qpsim_root.rglob("*.yaml"),
        *qpsim_root.rglob("*.yml"),
    }
    expected_relatives = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in expected_paths
    }
    if set(sources) != expected_relatives:
        missing = sorted(expected_relatives - set(sources))
        extra = sorted(set(sources) - expected_relatives)
        raise StagedResolvePilotError(
            f"{label} does not contain the exact producer source closure; "
            f"missing={missing!r}, extra={extra!r}."
        )
    for relative, expected_digest in sources.items():
        candidate = Path(relative)
        if candidate.as_posix() != relative:
            raise StagedResolvePilotError(
                f"{label} path {relative!r} is not canonical POSIX form."
            )
        live_path = REPOSITORY_ROOT / candidate
        try:
            resolved = live_path.resolve(strict=True)
            resolved.relative_to(REPOSITORY_ROOT.resolve())
        except (OSError, ValueError) as exc:
            raise StagedResolvePilotError(
                f"{label} path {relative!r} is missing or escapes the repository."
            ) from exc
        if not resolved.is_file() or resolved.is_symlink():
            raise StagedResolvePilotError(
                f"{label} path {relative!r} is not a regular source file."
            )
        if source_sha256(resolved) != expected_digest:
            raise StagedResolvePilotError(
                f"{label} source {relative!r} no longer matches its digest."
            )
    return sources


def _validate_stage_record(
    value: object,
    label: str,
    *,
    allow_initial_state: bool = False,
    allow_linear_error: bool = False,
) -> dict[str, Any]:
    stage = _mapping(value, label)
    expected_keys = set(_RAW_STAGE_KEYS if allow_linear_error else _SUMMARY_STAGE_KEYS)
    if allow_initial_state:
        expected_keys.add("initial_state")
    _exact_keys(stage, expected_keys, label)
    if allow_initial_state:
        _descriptor(stage.get("initial_state"), f"{label}.initial_state")
    if not isinstance(stage.get("converged"), bool):
        raise StagedResolvePilotError(f"{label}.converged must be boolean.")
    iterations = stage.get("newton_iterations")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise StagedResolvePilotError(
            f"{label}.newton_iterations must be a positive integer."
        )
    _finite(stage.get("last_relative_qp_step"), f"{label}.last_relative_qp_step")
    if allow_linear_error:
        _finite(
            stage.get("max_linear_solve_backward_error"),
            f"{label}.max_linear_solve_backward_error",
        )
    _string(stage.get("solve_path"), f"{label}.solve_path")
    _string(stage.get("termination"), f"{label}.termination")
    specification = _mapping(stage.get("specification"), f"{label}.specification")
    _exact_keys(
        specification,
        {
            "changed_convention",
            "coherence_convention",
            "density_convention",
            "pair_frequency_offset_bins",
            "parent_stage_id",
            "stage_id",
        },
        f"{label}.specification",
    )
    if specification.get("pair_frequency_offset_bins") not in {0, 1}:
        raise StagedResolvePilotError(
            f"{label}.specification.pair_frequency_offset_bins is invalid."
        )
    if specification.get("density_convention") not in {
        "author_cell_average_eV",
        "qpsim_cell_density_ueV",
    }:
        raise StagedResolvePilotError(
            f"{label}.specification.density_convention is invalid."
        )
    certificate = _mapping(
        stage.get("cancellation_certificate"),
        f"{label}.cancellation_certificate",
    )
    _exact_keys(
        certificate,
        {"phonon", "qp"},
        f"{label}.cancellation_certificate",
    )
    certificate_keys = {
        "channel_turnover_l1_s_inv",
        "residual_l1_over_channel_turnover",
        "residual_l1_s_inv",
        "residual_l2_s_inv",
        "residual_linf_s_inv",
    }
    for sector in ("phonon", "qp"):
        record = _mapping(
            certificate.get(sector),
            f"{label}.cancellation_certificate.{sector}",
        )
        _exact_keys(
            record,
            certificate_keys,
            f"{label}.cancellation_certificate.{sector}",
        )
        _validate_finite_tree(
            record,
            f"{label}.cancellation_certificate.{sector}",
        )
    final_bounds = _mapping(stage.get("final_bounds"), f"{label}.final_bounds")
    _exact_keys(
        final_bounds,
        {"f_max", "f_min", "n_phonon_max", "n_phonon_min"},
        f"{label}.final_bounds",
    )
    _validate_finite_tree(final_bounds, f"{label}.final_bounds")
    observable = _mapping(stage.get("observable"), f"{label}.observable")
    _exact_keys(
        observable,
        {
            "author_control_gap_reference_eV",
            "author_control_ordinate_reference",
            "direct_gap_eV",
            "direct_integral",
            "figure6_ordinate",
            "gap_minus_author_control_eV",
            "ordinate_minus_author_control",
            "thermal_gap_eV_held_fixed",
        },
        f"{label}.observable",
    )
    _validate_finite_tree(observable, f"{label}.observable")
    return stage


def _validate_finite_tree(value: object, label: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_tree(item, f"{label}.{key}")
    elif isinstance(value, bool):
        return
    elif isinstance(value, str):
        if not value:
            raise StagedResolvePilotError(f"{label} must not be empty.")
    elif isinstance(value, (int, float, np.integer, np.floating)):
        _finite(value, label)
    elif value is None:
        return
    else:
        raise StagedResolvePilotError(f"{label} has an unsupported JSON value.")


def _read_raw_staged_bundle(
    bundle_dir: Path,
) -> _RawBundleSnapshot:
    root = bundle_dir.resolve()
    manifest_path = root / "manifest.json"
    if not root.is_dir() or not manifest_path.is_file() or manifest_path.is_symlink():
        raise StagedResolvePilotError(f"Raw staged bundle is missing: {root}.")
    manifest_bytes = manifest_path.read_bytes()
    manifest = _load_json_bytes(manifest_bytes, "raw staged bundle manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "raw staged bundle manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise StagedResolvePilotError("Raw staged bundle schema is stale.")
    metadata = _mapping(manifest.get("metadata"), "raw staged bundle metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "raw staged bundle metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise StagedResolvePilotError("Raw staged metadata schema is stale.")
    _assert_same_tree(
        metadata.get("source_binding"),
        SOURCE_BINDING,
        "raw staged source binding",
    )
    _assert_same_tree(
        metadata.get("coordinate_contract"),
        EXPECTED_COORDINATE_CONTRACT,
        "raw staged coordinate contract",
    )
    if metadata.get("diagnostic_id") != EXPECTED_DIAGNOSTIC_ID:
        raise StagedResolvePilotError("Raw staged diagnostic ID is stale.")
    _assert_same_tree(
        metadata.get("limitations"),
        EXPECTED_LIMITATIONS,
        "raw staged limitations",
    )
    _assert_same_tree(
        metadata.get("solver_contract"),
        EXPECTED_SOLVER_CONTRACT,
        "raw staged solver contract",
    )
    author_control_reproduction = _mapping(
        metadata.get("author_control_reproduction"),
        "raw staged author-control reproduction",
    )
    _exact_keys(
        author_control_reproduction,
        {
            "captured_author_final_state",
            "final_state_relative_l2",
            "qp_final_relative_l2",
        },
        "raw staged author-control reproduction",
    )
    _descriptor(
        author_control_reproduction.get("captured_author_final_state"),
        "raw staged author-control final-state descriptor",
    )
    for key in ("final_state_relative_l2", "qp_final_relative_l2"):
        if (
            _finite(
                author_control_reproduction.get(key),
                f"raw staged author-control reproduction.{key}",
            )
            < 0.0
        ):
            raise StagedResolvePilotError(
                f"raw staged author-control reproduction.{key} is negative."
            )

    files = _mapping(manifest.get("files"), "raw staged bundle files")
    descriptors = _mapping(
        metadata.get("array_descriptors"),
        "raw staged array descriptors",
    )
    expected_files = {f"{name}.npy" for name in descriptors}
    semantic_expected_files = {"captured_author_initial_state.npy"} | {
        f"{result_id}__{suffix}.npy"
        for result_id in ALL_RESULT_IDS
        for suffix in _RESULT_ARRAY_SUFFIXES
    }
    if expected_files != semantic_expected_files:
        raise StagedResolvePilotError(
            "Raw staged array descriptors do not match the exact pilot schema."
        )
    if set(files) != expected_files:
        raise StagedResolvePilotError(
            "Raw staged files do not exactly match its array descriptors."
        )
    actual_npy = {
        path.name
        for path in root.glob("*.npy")
        if path.is_file() and not path.is_symlink()
    }
    if actual_npy != expected_files:
        raise StagedResolvePilotError(
            "Raw staged directory has missing or undeclared NPY files."
        )
    retained_bytes: dict[str, bytes] = {}
    for filename, raw_record in files.items():
        safe_name = _safe_filename(filename, "raw staged filename")
        record = _mapping(raw_record, f"raw staged files[{safe_name!r}]")
        _exact_keys(record, {"sha256", "size_bytes"}, f"raw staged files[{safe_name!r}]")
        expected_sha = _sha256(
            record.get("sha256"),
            f"raw staged files[{safe_name!r}].sha256",
        )
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise StagedResolvePilotError(
                f"raw staged files[{safe_name!r}].size_bytes is invalid."
            )
        path = root / safe_name
        if not path.is_file() or path.is_symlink():
            raise StagedResolvePilotError(f"Raw staged array is missing: {safe_name}.")
        content = path.read_bytes()
        retained_bytes[safe_name] = content
        if len(content) != size or hashlib.sha256(content).hexdigest() != expected_sha:
            raise StagedResolvePilotError(
                f"Raw staged array bytes do not match: {safe_name}."
            )
        array_name = safe_name.removesuffix(".npy")
        expected_descriptor = _descriptor(
            descriptors.get(array_name),
            f"raw staged descriptor {array_name!r}",
        )
        with io.BytesIO(content) as stream:
            array = np.lib.format.read_array(stream, allow_pickle=False)
        if _native_descriptor(array) != expected_descriptor:
            raise StagedResolvePilotError(
                f"Raw staged descriptor does not match array {array_name!r}."
            )

    arrays: dict[str, np.ndarray] = {}
    for filename, content in retained_bytes.items():
        with io.BytesIO(content) as stream:
            value = np.lib.format.read_array(stream, allow_pickle=False)
            if stream.read(1):
                raise StagedResolvePilotError(
                    f"Raw staged array has trailing bytes: {filename}."
                )
        value.setflags(write=False)
        arrays[filename.removesuffix(".npy")] = value

    stages = _mapping(metadata.get("stages"), "raw staged stages")
    if set(stages) != ALL_RESULT_IDS:
        raise StagedResolvePilotError("Raw staged result IDs are incomplete.")
    for result_id, stage in stages.items():
        _validate_stage_record(
            stage,
            f"raw staged stages[{result_id!r}]",
            allow_initial_state=True,
            allow_linear_error=True,
        )
    _authenticate_live_sources(
        metadata.get("sources"),
        "raw staged numerical producer sources",
    )
    input_bundle = _mapping(metadata.get("input_bundle"), "raw staged input bundle")
    anchor_binding = _mapping(
        input_bundle.get("anchor"),
        "raw staged anchor binding",
    )
    _exact_keys(
        anchor_binding,
        {
            "adapter_path",
            "anchor_id",
            "anchor_path",
            "anchor_sha256",
            "author_source_manifest_path",
        },
        "raw staged anchor binding",
    )
    anchor_path_raw = _string(
        anchor_binding.get("anchor_path"),
        "raw staged anchor path",
    )
    if anchor_path_raw != EXPECTED_ANCHOR_PATH:
        raise StagedResolvePilotError("Raw staged anchor path is noncanonical.")
    anchor_path = Path(anchor_path_raw)
    if anchor_path.is_absolute() or ".." in anchor_path.parts:
        raise StagedResolvePilotError("Raw staged anchor path is unsafe.")
    expected_anchor_sha = _sha256(
        anchor_binding.get("anchor_sha256"),
        "raw staged anchor SHA",
    )
    if expected_anchor_sha != EXPECTED_ANCHOR_SHA256:
        raise StagedResolvePilotError("Raw staged anchor digest is noncanonical.")
    input_manifest_sha = _sha256(
        input_bundle.get("manifest_sha256"),
        "raw staged input manifest SHA",
    )
    if input_manifest_sha != EXPECTED_INPUT_MANIFEST_SHA256:
        raise StagedResolvePilotError("Raw staged input manifest is noncanonical.")
    live_anchor = REPOSITORY_ROOT / anchor_path
    if not live_anchor.is_file() or live_anchor.is_symlink():
        raise StagedResolvePilotError("Raw staged bundle's checked anchor is missing.")
    anchor_bytes = live_anchor.read_bytes()
    if hashlib.sha256(anchor_bytes).hexdigest() != expected_anchor_sha:
        raise StagedResolvePilotError(
            "Raw staged bundle is not bound to the live checked anchor."
        )
    anchor = _load_json_bytes(anchor_bytes, "checked author-point anchor")
    if anchor.get("bundle_manifest_sha256") != input_manifest_sha:
        raise StagedResolvePilotError(
            "Checked author-point anchor and staged input manifest disagree."
        )
    return _RawBundleSnapshot(
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        arrays=arrays,
        anchor=anchor,
    )


def _expected_parameters(anchor: dict[str, Any]) -> AuthorSolveParameters:
    expected_result = _mapping(anchor.get("expected_result"), "anchor.expected_result")
    expected_parameters = _mapping(
        anchor.get("expected_parameters"),
        "anchor.expected_parameters",
    )
    constants = _mapping(
        expected_parameters.get("author_numerical_constants"),
        "anchor.expected_parameters.author_numerical_constants",
    )
    return AuthorSolveParameters(
        gap_eV=float(EXPECTED_PARAMETERS["gap_eV"]),
        h_eV=float(EXPECTED_PARAMETERS["h_eV"]),
        temperature_K=float(EXPECTED_PARAMETERS["temperature_K"]),
        T_c_K=float(EXPECTED_PARAMETERS["T_c_K"]),
        tau_0_s=float(EXPECTED_PARAMETERS["tau_0_s"]),
        tau_0_pb_s=float(EXPECTED_PARAMETERS["tau_0_pb_s"]),
        tau_l_s=float(EXPECTED_PARAMETERS["tau_l_s"]),
        photon_bin=int(EXPECTED_PARAMETERS["photon_bin"]),
        n_bar=float(EXPECTED_PARAMETERS["n_bar"]),
        c_photon_s_inv=float(EXPECTED_PARAMETERS["c_photon_s_inv"]),
        delta0_eV=float(EXPECTED_PARAMETERS["gap_eV"]),
        thermal_gap_eV=_finite(
            expected_result.get("gap_thermal_eV"),
            "anchor.expected_result.gap_thermal_eV",
        ),
        max_newton_steps=int(EXPECTED_PARAMETERS["max_newton_steps"]),
        relative_step_threshold=float(
            EXPECTED_PARAMETERS["relative_step_threshold"]
        ),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=_finite(
                constants.get("boltzmann_constant_J_per_K"),
                "anchor author Boltzmann constant",
            ),
            electron_charge_C=_finite(
                constants.get("electron_charge_C"),
                "anchor author electron charge",
            ),
        ),
    )


def _certificate_from_evaluation(
    evaluation: Any,
    size: int,
) -> dict[str, dict[str, float]]:
    qp_residual = evaluation.residual_s_inv[:size]
    phonon_residual = evaluation.residual_s_inv[size:]
    qp_turnover = float(
        np.sum(np.abs(evaluation.qp_photon_s_inv))
        + np.sum(np.abs(evaluation.qp_scattering_s_inv))
        + np.sum(np.abs(evaluation.qp_pair_s_inv))
    )
    phonon_turnover = float(
        np.sum(np.abs(evaluation.phonon_scattering_s_inv))
        + np.sum(np.abs(evaluation.phonon_pair_s_inv))
        + np.sum(np.abs(evaluation.phonon_escape_s_inv))
    )

    def record(residual: np.ndarray, turnover: float) -> dict[str, float]:
        residual_l1 = float(np.sum(np.abs(residual)))
        return {
            "channel_turnover_l1_s_inv": turnover,
            "residual_l1_over_channel_turnover": (
                residual_l1 / turnover if turnover > 0.0 else 0.0
            ),
            "residual_l1_s_inv": residual_l1,
            "residual_l2_s_inv": float(np.linalg.norm(residual)),
            "residual_linf_s_inv": float(
                np.max(np.abs(residual), initial=0.0)
            ),
        }

    return {
        "phonon": record(phonon_residual, phonon_turnover),
        "qp": record(qp_residual, qp_turnover),
    }


def _observable_from_state(
    final_state: np.ndarray,
    parameters: AuthorSolveParameters,
    *,
    author_gap_eV: float,
    author_observable: float,
) -> dict[str, float]:
    size = int(EXPECTED_PARAMETERS["num_energy_cells"])
    f = final_state[:size]
    centers = (
        parameters.gap_eV
        + parameters.h_eV * (np.arange(size, dtype=float) + 0.5)
    )
    integral = gap_integral_from_distribution_direct(
        f,
        centers,
        gap=parameters.gap_eV,
        samples="authors",
    )
    gap = gap_from_distribution_direct(
        f,
        centers,
        gap=parameters.gap_eV,
        delta0=parameters.delta0_eV,
        samples="authors",
    )
    denominator = parameters.gap_eV - parameters.thermal_gap_eV
    if denominator == 0.0:
        raise StagedResolvePilotError("The author thermal observable denominator is zero.")
    ordinate = (gap - parameters.thermal_gap_eV) / denominator
    return {
        "author_control_gap_reference_eV": author_gap_eV,
        "author_control_ordinate_reference": author_observable,
        "direct_gap_eV": gap,
        "direct_integral": integral,
        "figure6_ordinate": ordinate,
        "gap_minus_author_control_eV": gap - author_gap_eV,
        "ordinate_minus_author_control": ordinate - author_observable,
        "thermal_gap_eV_held_fixed": parameters.thermal_gap_eV,
    }


def _verify_stage_semantics(
    snapshot: _RawBundleSnapshot,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, object]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    metadata = _mapping(snapshot.manifest["metadata"], "raw staged metadata")
    _assert_same_tree(
        metadata.get("parameters"),
        EXPECTED_PARAMETERS,
        "raw staged parameters",
    )
    parameters = _expected_parameters(snapshot.anchor)
    expected_result = _mapping(
        snapshot.anchor.get("expected_result"),
        "anchor.expected_result",
    )
    author_gap = _finite(
        expected_result.get("gap_driven_eV"),
        "anchor.expected_result.gap_driven_eV",
    )
    author_observable = _finite(
        expected_result.get("normalized_numerical_observable"),
        "anchor.expected_result.normalized_numerical_observable",
    )
    size = int(EXPECTED_PARAMETERS["num_energy_cells"])
    state_size = 2 * size - 1
    E_left = parameters.gap_eV + parameters.h_eV * np.arange(size, dtype=float)
    operators = {
        specification.stage_id: build_staged_operator(
            E_left,
            parameters,
            specification,
        )
        for specification in PRIMARY_STAGES
    }

    captured_initial = snapshot.arrays["captured_author_initial_state"]
    if captured_initial.shape != (state_size,) or captured_initial.dtype != np.dtype("<f8"):
        raise StagedResolvePilotError("Captured author initial state is malformed.")
    if np.any(~np.isfinite(captured_initial)):
        raise StagedResolvePilotError("Captured author initial state is non-finite.")

    raw_stages = _mapping(metadata.get("stages"), "raw staged stages")
    summary_stages: dict[str, dict[str, Any]] = {}
    final_states: dict[str, dict[str, object]] = {}
    computed_observables: dict[str, float] = {}

    for result_id in sorted(ALL_RESULT_IDS):
        raw_stage = _mapping(raw_stages[result_id], f"raw staged stages[{result_id!r}]")
        expected_specification = EXPECTED_STAGE_SPECIFICATIONS[result_id]
        expected_path = EXPECTED_SOLVE_PATHS[result_id]
        _assert_same_tree(
            raw_stage.get("specification"),
            _specification_record(expected_specification),
            f"raw staged stages[{result_id!r}].specification",
        )
        if raw_stage.get("solve_path") != expected_path:
            raise StagedResolvePilotError(
                f"raw staged stages[{result_id!r}].solve_path is invalid."
            )
        raw_bounds = _mapping(
            raw_stage.get("final_bounds"),
            f"raw staged stages[{result_id!r}].final_bounds",
        )
        if any(
            _finite(raw_bounds.get(key), f"raw staged bounds.{key}") < 0.0
            for key in ("f_min", "n_phonon_min")
        ):
            raise StagedResolvePilotError(
                f"raw staged stages[{result_id!r}].final_bounds claims "
                "a negative occupation."
            )
        history = snapshot.arrays[f"{result_id}__state_history"]
        final_state = snapshot.arrays[f"{result_id}__final_state"]
        newton_deltas = snapshot.arrays[f"{result_id}__newton_deltas"]
        relative_steps = snapshot.arrays[f"{result_id}__relative_qp_steps"]
        linear_errors = snapshot.arrays[
            f"{result_id}__linear_solve_backward_errors"
        ]
        if (
            history.ndim != 2
            or history.shape[1] != state_size
            or history.shape[0] < 2
            or history.shape[0] > parameters.max_newton_steps + 1
            or final_state.shape != (state_size,)
            or newton_deltas.shape != (history.shape[0] - 1, state_size)
            or relative_steps.shape != (history.shape[0] - 1,)
            or linear_errors.shape != relative_steps.shape
        ):
            raise StagedResolvePilotError(
                f"Stage {result_id!r} history/step arrays are malformed."
            )
        for label, value in {
            "state history": history,
            "final state": final_state,
            "Newton deltas": newton_deltas,
            "relative steps": relative_steps,
            "linear errors": linear_errors,
        }.items():
            if np.any(~np.isfinite(value)):
                raise StagedResolvePilotError(
                    f"Stage {result_id!r} {label} is non-finite."
                )
        _assert_same_array(
            final_state,
            history[-1],
            f"stage {result_id!r} final-state endpoint",
        )
        if result_id in PRIMARY_STAGE_IDS:
            _assert_same_array(
                history[0],
                captured_initial,
                f"stage {result_id!r} shared primary initial state",
            )
        elif result_id == "c3b_continuation_from_c3a":
            _assert_same_array(
                history[0],
                snapshot.arrays["c3a_finite_volume_coherence__final_state"],
                "C3b continuation initial state",
            )
        else:
            _assert_same_array(
                history[0],
                snapshot.arrays["c3b_center_pair_labels__final_state"],
                "C3c continuation initial state",
            )

        recomputed_steps = np.empty(history.shape[0] - 1)
        recomputed_linear_errors = np.empty(history.shape[0] - 1)
        operator = operators[expected_specification.stage_id]
        for step_index in range(history.shape[0] - 1):
            before = history[step_index]
            after = history[step_index + 1]
            evaluation = evaluate_staged_system(
                operator,
                before[:size],
                before[size:],
                build_update_matrix=True,
            )
            matrix = evaluation.update_matrix_s_inv
            if matrix is None:  # pragma: no cover - guaranteed above.
                raise AssertionError("Staged verifier did not receive a Newton matrix.")
            expected_delta = author_newton_delta(evaluation)
            if _symmetric_relative(
                expected_delta,
                newton_deltas[step_index],
            ) > 1.0e-12:
                raise StagedResolvePilotError(
                    f"Stage {result_id!r} retained Newton transition "
                    f"{step_index} is not the authenticated update."
                )
            _assert_same_array(
                after,
                before + newton_deltas[step_index],
                f"stage {result_id!r} Newton history transition {step_index}",
            )
            denominator = float(np.linalg.norm(matrix, ord=np.inf)) * float(
                np.linalg.norm(newton_deltas[step_index], ord=np.inf)
            ) + float(np.linalg.norm(evaluation.residual_s_inv, ord=np.inf))
            linear_error = float(
                np.linalg.norm(
                    matrix @ newton_deltas[step_index]
                    + evaluation.residual_s_inv,
                    ord=np.inf,
                )
            )
            recomputed_linear_errors[step_index] = (
                linear_error / denominator if denominator > 0.0 else 0.0
            )
            qp_denominator = float(np.linalg.norm(after[:size]))
            recomputed_steps[step_index] = (
                float(np.linalg.norm(after[:size] - before[:size]))
                / qp_denominator
            )
        _assert_same_array(
            relative_steps,
            recomputed_steps,
            f"stage {result_id!r} recorded relative steps",
            rtol=5.0e-13,
            atol=5.0e-300,
        )
        _assert_same_array(
            linear_errors,
            recomputed_linear_errors,
            f"stage {result_id!r} recorded linear backward errors",
            rtol=5.0e-13,
            atol=5.0e-300,
        )
        if np.any(linear_errors < 0.0):
            raise StagedResolvePilotError(
                f"Stage {result_id!r} has a negative linear backward error."
            )
        converged = bool(relative_steps[-1] < parameters.relative_step_threshold)
        if not converged or np.any(
            relative_steps[:-1] < parameters.relative_step_threshold
        ):
            raise StagedResolvePilotError(
                f"Stage {result_id!r} does not satisfy the exact stopping rule."
            )

        final_evaluation = evaluate_staged_system(
            operator,
            final_state[:size],
            final_state[size:],
            build_update_matrix=False,
        )
        recomputed_channels = {
            "residual_s_inv": final_evaluation.residual_s_inv,
            "qp_photon_s_inv": final_evaluation.qp_photon_s_inv,
            "qp_scattering_s_inv": final_evaluation.qp_scattering_s_inv,
            "qp_pair_s_inv": final_evaluation.qp_pair_s_inv,
            "phonon_scattering_s_inv": final_evaluation.phonon_scattering_s_inv,
            "phonon_pair_s_inv": final_evaluation.phonon_pair_s_inv,
            "phonon_escape_s_inv": final_evaluation.phonon_escape_s_inv,
        }
        for suffix, recomputed in recomputed_channels.items():
            _assert_same_array(
                snapshot.arrays[f"{result_id}__{suffix}"],
                recomputed,
                f"stage {result_id!r} final {suffix}",
                rtol=5.0e-13,
                atol=5.0e-300,
            )
        certificate = _certificate_from_evaluation(final_evaluation, size)
        bounds = {
            "f_max": float(np.max(final_state[:size])),
            "f_min": float(np.min(final_state[:size])),
            "n_phonon_max": float(np.max(final_state[size:])),
            "n_phonon_min": float(np.min(final_state[size:])),
        }
        if bounds["f_min"] < 0.0 or bounds["f_max"] > 1.0:
            raise StagedResolvePilotError(
                f"Stage {result_id!r} has an out-of-bounds QP occupation."
            )
        if bounds["n_phonon_min"] < 0.0:
            raise StagedResolvePilotError(
                f"Stage {result_id!r} has a negative phonon occupation."
            )
        observable = _observable_from_state(
            final_state,
            parameters,
            author_gap_eV=author_gap,
            author_observable=author_observable,
        )
        recomputed_stage: dict[str, Any] = {
            "cancellation_certificate": certificate,
            "converged": converged,
            "final_bounds": bounds,
            "last_relative_qp_step": float(relative_steps[-1]),
            "newton_iterations": int(relative_steps.size),
            "observable": observable,
            "solve_path": expected_path,
            "specification": _specification_record(expected_specification),
            "termination": "relative_qp_step_threshold",
        }
        expected_raw_stage = {
            **recomputed_stage,
            "initial_state": _native_descriptor(history[0]),
            "max_linear_solve_backward_error": float(
                np.max(linear_errors, initial=0.0)
            ),
        }
        _assert_same_tree(
            raw_stage,
            expected_raw_stage,
            f"raw staged stages[{result_id!r}]",
        )
        summary_stages[result_id] = recomputed_stage
        computed_observables[result_id] = observable["figure6_ordinate"]
        final_states[result_id] = {
            "array_name": f"{result_id}__final_state",
            "descriptor": _native_descriptor(final_state),
            "external_npy_sha256": _native_descriptor(final_state)["npy_sha256"],
        }

    path_dependence: dict[str, dict[str, float]] = {}
    for continuation_id, independent_id in (
        ("c3b_continuation_from_c3a", "c3b_center_pair_labels"),
        ("c3c_continuation_from_c3b", "c3c_native_cell_density"),
    ):
        independent = snapshot.arrays[f"{independent_id}__final_state"]
        continued = snapshot.arrays[f"{continuation_id}__final_state"]
        scale = float(np.linalg.norm(independent))
        path_dependence[continuation_id] = {
            "final_state_relative_l2": (
                float(np.linalg.norm(continued - independent)) / scale
                if scale > 0.0
                else 0.0
            ),
            "figure6_ordinate_signed_difference": (
                computed_observables[continuation_id]
                - computed_observables[independent_id]
            ),
        }
    _assert_same_tree(
        metadata.get("path_dependence"),
        path_dependence,
        "raw staged path dependence",
    )
    author_control = {
        "direct_gap_eV": summary_stages["author_control"]["observable"][
            "direct_gap_eV"
        ],
        "direct_gap_minus_anchor_eV": (
            summary_stages["author_control"]["observable"]["direct_gap_eV"]
            - author_gap
        ),
        "figure6_ordinate": computed_observables["author_control"],
        "figure6_ordinate_minus_anchor": (
            computed_observables["author_control"] - author_observable
        ),
    }
    return summary_stages, final_states, path_dependence, author_control


def build_staged_resolve_pilot(staged_bundle_dir: Path) -> dict[str, Any]:
    """Extract the checked compact pilot from one fully verified raw bundle."""

    snapshot = _read_raw_staged_bundle(staged_bundle_dir)
    metadata = _mapping(snapshot.manifest["metadata"], "raw staged metadata")
    (
        summary_stages,
        final_states,
        path_dependence,
        author_control,
    ) = _verify_stage_semantics(snapshot)
    input_bundle = _mapping(metadata["input_bundle"], "raw staged input bundle")

    generator_path = Path(__file__).resolve()
    generator_relative = generator_path.relative_to(REPOSITORY_ROOT).as_posix()
    return {
        "author_control_reproduction": author_control,
        "final_states": final_states,
        "input_bundle": input_bundle,
        "limitations": dict(EXPECTED_LIMITATIONS),
        "numerical_producer_sources": metadata["sources"],
        "parameters": dict(EXPECTED_PARAMETERS),
        "path_dependence": path_dependence,
        "pilot_id": EXPECTED_PILOT_ID,
        "qualification": (
            "Supplemental bounded single-point evidence only. The author control, "
            "pair-label-only branch, C3a, C3b, and the isolated C3c native-DOS "
            "substitution were fully re-solved at the separately anchored "
            "Figure 6 point. This record does not mark C3 or any formal "
            "reproduction-ladder stage completed."
        ),
        "raw_array_policy": {
            "availability": "external_non_redistributed_development_artifact",
            "rerun_requirement": (
                "Recreating the input author bundle requires the authenticated "
                "author archive via QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE; the "
                "checked repository does not redistribute that archive or these "
                "staged raw arrays."
            ),
        },
        "schema": PILOT_SCHEMA,
        "source_binding": dict(SOURCE_BINDING),
        "staged_bundle": {
            "manifest_sha256": snapshot.manifest_sha256,
            "schema": RAW_SCHEMA,
        },
        "stages": summary_stages,
        "status": PILOT_STATUS,
        "summary_generator": {
            "path": generator_relative,
            "sha256": source_sha256(generator_path),
        },
    }


def canonical_pilot_bytes(payload: dict[str, Any]) -> bytes:
    """Return the canonical checked-file representation."""

    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def write_staged_resolve_pilot(
    staged_bundle_dir: Path,
    output_path: Path = DEFAULT_PILOT,
) -> Path:
    """Generate the checked pilot from raw evidence without overwriting."""

    payload = build_staged_resolve_pilot(staged_bundle_dir)
    path = output_path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_pilot_bytes(payload))
    load_staged_resolve_pilot(path)
    return path


def load_staged_resolve_pilot(path: Path = DEFAULT_PILOT) -> dict[str, Any]:
    """Strictly validate and return a checked staged-resolve pilot."""

    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise StagedResolvePilotError(f"Checked staged-resolve pilot is missing: {resolved}.")
    pilot = _load_json_bytes(resolved.read_bytes(), "checked staged-resolve pilot")
    _exact_keys(pilot, _PILOT_KEYS, "checked staged-resolve pilot")
    if pilot.get("schema") != PILOT_SCHEMA:
        raise StagedResolvePilotError("Checked staged-resolve pilot schema is stale.")
    if pilot.get("status") != PILOT_STATUS:
        raise StagedResolvePilotError("Checked staged-resolve pilot status is invalid.")
    _assert_same_tree(
        pilot.get("source_binding"),
        SOURCE_BINDING,
        "checked pilot.source_binding",
    )
    if pilot.get("pilot_id") != EXPECTED_PILOT_ID:
        raise StagedResolvePilotError("Checked pilot ID is stale.")
    _string(pilot.get("qualification"), "checked pilot.qualification")

    staged_bundle = _mapping(pilot.get("staged_bundle"), "checked pilot.staged_bundle")
    _exact_keys(staged_bundle, {"manifest_sha256", "schema"}, "checked pilot.staged_bundle")
    if staged_bundle.get("schema") != RAW_SCHEMA:
        raise StagedResolvePilotError("Checked pilot raw staged schema is stale.")
    _sha256(
        staged_bundle.get("manifest_sha256"),
        "checked pilot.staged_bundle.manifest_sha256",
    )
    _authenticate_live_sources(
        pilot.get("numerical_producer_sources"),
        "checked pilot.numerical_producer_sources",
    )

    generator = _mapping(pilot.get("summary_generator"), "checked pilot.summary_generator")
    _exact_keys(generator, {"path", "sha256"}, "checked pilot.summary_generator")
    generator_path = _string(generator.get("path"), "checked pilot.summary_generator.path")
    expected_generator = Path(__file__).resolve().relative_to(REPOSITORY_ROOT).as_posix()
    if generator_path != expected_generator:
        raise StagedResolvePilotError("Checked pilot summary generator path is noncanonical.")
    if _sha256(
        generator.get("sha256"),
        "checked pilot.summary_generator.sha256",
    ) != source_sha256(Path(__file__).resolve()):
        raise StagedResolvePilotError(
            "Checked pilot summary generator no longer matches live source."
        )

    input_bundle = _mapping(pilot.get("input_bundle"), "checked pilot.input_bundle")
    _exact_keys(
        input_bundle,
        {"anchor", "manifest_sha256"},
        "checked pilot.input_bundle",
    )
    anchor = _mapping(input_bundle.get("anchor"), "checked pilot.input_bundle.anchor")
    _exact_keys(
        anchor,
        {
            "adapter_path",
            "anchor_id",
            "anchor_path",
            "anchor_sha256",
            "author_source_manifest_path",
        },
        "checked pilot.input_bundle.anchor",
    )
    anchor_path_raw = _string(
        anchor.get("anchor_path"),
        "checked pilot.input_bundle.anchor.anchor_path",
    )
    anchor_path = Path(anchor_path_raw)
    if anchor_path.is_absolute() or ".." in anchor_path.parts:
        raise StagedResolvePilotError("Checked pilot anchor path is unsafe.")
    anchor_sha = _sha256(
        anchor.get("anchor_sha256"),
        "checked pilot.input_bundle.anchor.anchor_sha256",
    )
    live_anchor = REPOSITORY_ROOT / anchor_path
    if (
        not live_anchor.is_file()
        or live_anchor.is_symlink()
        or hashlib.sha256(live_anchor.read_bytes()).hexdigest() != anchor_sha
    ):
        raise StagedResolvePilotError("Checked pilot anchor binding is stale.")
    _sha256(
        input_bundle.get("manifest_sha256"),
        "checked pilot.input_bundle.manifest_sha256",
    )

    stages = _mapping(pilot.get("stages"), "checked pilot.stages")
    if set(stages) != ALL_RESULT_IDS:
        raise StagedResolvePilotError("Checked pilot stage IDs are incomplete.")
    for result_id, stage in stages.items():
        validated = _validate_stage_record(stage, f"checked pilot.stages[{result_id!r}]")
        if validated.get("converged") is not True:
            raise StagedResolvePilotError(
                f"Checked pilot stage {result_id!r} did not converge."
            )
        specification = _mapping(
            validated["specification"],
            f"checked pilot stage {result_id!r} specification",
        )
        _assert_same_tree(
            specification,
            _specification_record(EXPECTED_STAGE_SPECIFICATIONS[result_id]),
            f"checked pilot stage {result_id!r} specification",
        )
        if validated.get("solve_path") != EXPECTED_SOLVE_PATHS[result_id]:
            raise StagedResolvePilotError(
                f"Checked pilot stage {result_id!r} has the wrong solve path."
            )

    final_states = _mapping(pilot.get("final_states"), "checked pilot.final_states")
    if set(final_states) != ALL_RESULT_IDS:
        raise StagedResolvePilotError("Checked pilot final-state IDs are incomplete.")
    for result_id, raw_record in final_states.items():
        record = _mapping(raw_record, f"checked pilot.final_states[{result_id!r}]")
        _exact_keys(
            record,
            {"array_name", "descriptor", "external_npy_sha256"},
            f"checked pilot.final_states[{result_id!r}]",
        )
        expected_name = f"{result_id}__final_state"
        if record.get("array_name") != expected_name:
            raise StagedResolvePilotError(
                f"Checked pilot final state {result_id!r} has the wrong name."
            )
        descriptor = _descriptor(
            record.get("descriptor"),
            f"checked pilot.final_states[{result_id!r}].descriptor",
        )
        if _sha256(
            record.get("external_npy_sha256"),
            f"checked pilot.final_states[{result_id!r}].external_npy_sha256",
        ) != descriptor["npy_sha256"]:
            raise StagedResolvePilotError(
                f"Checked pilot final state {result_id!r} hashes disagree."
            )

    author_control = _mapping(
        pilot.get("author_control_reproduction"),
        "checked pilot.author_control_reproduction",
    )
    _exact_keys(
        author_control,
        {
            "direct_gap_eV",
            "direct_gap_minus_anchor_eV",
            "figure6_ordinate",
            "figure6_ordinate_minus_anchor",
        },
        "checked pilot.author_control_reproduction",
    )
    _validate_finite_tree(
        author_control,
        "checked pilot.author_control_reproduction",
    )
    limitations = _mapping(pilot.get("limitations"), "checked pilot.limitations")
    _exact_keys(
        limitations,
        {"scope", "seed_sensitivity", "solver"},
        "checked pilot.limitations",
    )
    _validate_finite_tree(limitations, "checked pilot.limitations")
    parameters = _mapping(pilot.get("parameters"), "checked pilot.parameters")
    _exact_keys(
        parameters,
        {
            "T_c_K",
            "c_photon_s_inv",
            "gap_eV",
            "h_eV",
            "max_newton_steps",
            "n_bar",
            "num_energy_cells",
            "photon_bin",
            "relative_step_threshold",
            "tau_0_pb_s",
            "tau_0_s",
            "tau_l_s",
            "temperature_K",
        },
        "checked pilot.parameters",
    )
    _validate_finite_tree(parameters, "checked pilot.parameters")
    _assert_same_tree(
        parameters,
        EXPECTED_PARAMETERS,
        "checked pilot.parameters",
    )
    path_dependence = _mapping(
        pilot.get("path_dependence"),
        "checked pilot.path_dependence",
    )
    _exact_keys(
        path_dependence,
        {"c3b_continuation_from_c3a", "c3c_continuation_from_c3b"},
        "checked pilot.path_dependence",
    )
    for continuation_id, record_value in path_dependence.items():
        record = _mapping(
            record_value,
            f"checked pilot.path_dependence[{continuation_id!r}]",
        )
        _exact_keys(
            record,
            {"figure6_ordinate_signed_difference", "final_state_relative_l2"},
            f"checked pilot.path_dependence[{continuation_id!r}]",
        )
    _validate_finite_tree(path_dependence, "checked pilot.path_dependence")
    raw_policy = _mapping(
        pilot.get("raw_array_policy"),
        "checked pilot.raw_array_policy",
    )
    _exact_keys(
        raw_policy,
        {"availability", "rerun_requirement"},
        "checked pilot.raw_array_policy",
    )
    _validate_finite_tree(raw_policy, "checked pilot.raw_array_policy")
    return pilot


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_PILOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(write_staged_resolve_pilot(args.staged_bundle, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
