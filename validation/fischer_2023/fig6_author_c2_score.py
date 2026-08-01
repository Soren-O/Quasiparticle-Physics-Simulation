"""Independently verify and summarize the Figure 6 C2 parameter evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023 import fig6_author_c0_summary as c0_summary_verifier
from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY as DEFAULT_C0_SUMMARY,
)
from validation.fischer_2023.fig6_author_c0_summary import (
    load_c0_raw_bundle,
    load_c0_summary,
)
from validation.fischer_2023.fig6_author_c1_score import (
    DEFAULT_SCORE as DEFAULT_C1_SCORE,
)
from validation.fischer_2023.fig6_author_c1_score import (
    build_c1_score,
    load_c1_score,
)
from validation.fischer_2023.fig6_author_c2_bundle import (
    BALANCE_FIELDS,
    CHANNEL_NAMES,
    FROZEN_ARRAY_NAMES,
    array_descriptor,
    json_value_bit_exact,
    load_c2_raw_bundle,
)
from validation.fischer_2023.fig6_author_c2_bundle import (
    SCHEMA as RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c2_parameters import (
    NativeFig6Parameters,
    build_c2_parameter_plan,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    SystemEvaluation,
    build_author_operator,
    evaluate_author_system,
)
from validation.source_provenance import canonical_source_bytes, source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c2-parameter-score.v1"
DEFAULT_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c2-parameter-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c2-raw-manifest-receipt.json"
)
RECEIPT_SCHEMA = "qpsim.fischer2023.fig6-author-c2-raw-manifest-receipt.v1"
STAGE_ID = "C2"
PARENT_STAGE_ID = "C1"
CHANGED_COMPONENT = "parameters"
_RAW_SOURCE_RELATIVES = frozenset(
    {
        "qpsim/constants.py",
        "qpsim/physics/gap_equation.py",
        "validation/source_provenance.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/fischer_2023/fig6_author_c0_summary.py",
        "validation/fischer_2023/fig6_author_c1_score.py",
        "validation/fischer_2023/fig6_author_c2_bundle.py",
        "validation/fischer_2023/fig6_author_c2_parameters.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
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
_TOP_LEVEL_KEYS = {
    "acceptance",
    "comparison",
    "frozen_inputs",
    "limitations",
    "observable_control",
    "parameter_axis",
    "parent_bindings",
    "raw_bundle",
    "schema",
    "sources",
    "stage",
    "steps",
    "structural_identity",
}
_ACCEPTANCE_LIMITS = {
    "c2a_net_subtraction_roundoff_factor": 64.0,
    "raw_array_max_absolute_error": 0.0,
    "raw_scalar_max_absolute_error": 0.0,
}
_ACCEPTANCE_CHECK_KEYS = {
    "all_gain_loss_arrays_nonnegative",
    "c0_c1_parent_chain_recomputed",
    "c2a_control_bit_exact",
    "explicit_energy_native_step_effective_noop",
    "finite_cutoff_tc_changes_qp_phonon_channels",
    "finite_cutoff_tc_leaves_other_channels_exact",
    "fixed_nbar_preserved",
    "frozen_inputs_bit_exact",
    "literal_c_changes_photon",
    "literal_c_leaves_nonphoton_channels_exact",
    "modern_kb_changes_phonon_escape_gain_and_net",
    "modern_kb_changes_qp_phonon_channels",
    "modern_kb_changes_thermal_occupation",
    "modern_kb_leaves_kb_independent_channels_exact",
    "modern_kb_preserves_phonon_escape_loss_exact",
    "parameter_axis_shift_is_nonzero",
    "raw_arrays_independently_recomputed",
    "raw_metadata_independently_recomputed",
    "structural_grid_and_dimensionless_kernels_bit_exact",
    "tau_pb_changes_phonon_kernel_channels",
    "tau_pb_leaves_other_channels_exact",
}
_C2A_CONTROL_CHECK_KEYS = (
    {"residual_bit_exact_to_c0"}
    | {
        f"{channel}_{field}_bit_exact_to_c0"
        for channel in CHANNEL_NAMES
        for field in BALANCE_FIELDS
    }
    | {
        f"{channel}_net_arithmetic_consistent_with_c0_gain_loss"
        for channel in CHANNEL_NAMES
    }
)


class C2ScoreError(ValueError):
    """The C2 raw evidence, parents, or checked score is malformed."""


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C2ScoreError(f"C2 score source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C2ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C2ScoreError(f"Non-finite JSON constant {token!r} is forbidden.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2ScoreError(f"Cannot parse {label}: {exc}.") from exc
    if not isinstance(value, dict):
        raise C2ScoreError(f"{label} must be an object.")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C2ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise C2ScoreError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(value)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C2ScoreError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _finite_scalar(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise C2ScoreError(f"{label} must be a finite real scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C2ScoreError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise C2ScoreError(f"{label} must be finite and valid.")
    return result


def _file_sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise C2ScoreError(f"{label} is missing, unsafe, or a symlink.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_c2_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Load the committed non-circular receipt for the external C2 bundle.

    The raw arrays remain outside the repository, so their manifest digest
    cannot be trusted when it appears only inside the checked score that it is
    meant to authenticate.  This separately committed receipt anchors both
    that external-manifest digest and the complete canonical score bytes.  The
    reproduction ladder binds the receipt itself as supplemental evidence.
    """

    if path.is_symlink() or not path.is_file():
        raise C2ScoreError("Committed C2 raw-manifest receipt is missing, unsafe, or a symlink.")
    receipt = _parse_json(path.read_bytes(), "C2 raw-manifest receipt")
    _exact_keys(
        receipt,
        {"checked_score", "qualification", "raw_bundle", "schema"},
        "C2 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C2ScoreError("C2 raw-manifest receipt schema is unsupported.")
    qualification = receipt.get("qualification")
    if qualification != (
        "Repository trust anchor for the externally retained C2 raw manifest "
        "and the complete canonical checked-score bytes; it does not contain "
        "or replace the raw arrays."
    ):
        raise C2ScoreError("C2 raw-manifest receipt qualification is invalid.")
    checked_score = _mapping(receipt.get("checked_score"), "receipt.checked_score")
    _exact_keys(
        checked_score,
        {"file_sha256", "schema"},
        "receipt.checked_score",
    )
    if checked_score.get("schema") != SCHEMA:
        raise C2ScoreError("C2 raw-manifest receipt score schema is invalid.")
    _sha256(
        checked_score.get("file_sha256"),
        "receipt.checked_score.file_sha256",
    )
    raw_bundle = _mapping(receipt.get("raw_bundle"), "receipt.raw_bundle")
    _exact_keys(
        raw_bundle,
        {"manifest_sha256", "schema"},
        "receipt.raw_bundle",
    )
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C2ScoreError("C2 raw-manifest receipt bundle schema is invalid.")
    _sha256(
        raw_bundle.get("manifest_sha256"),
        "receipt.raw_bundle.manifest_sha256",
    )
    return receipt


def _step_slug(step_id: str) -> str:
    slug = step_id.lower().replace("-", "_")
    if not slug.replace("_", "").isalnum():
        raise C2ScoreError(f"Unsafe C2 step id {step_id!r}.")
    return slug


def _author_parameters_from_summary(summary: dict[str, Any]) -> AuthorSolveParameters:
    raw = _mapping(summary.get("parameters"), "C0 parameters")
    observable = _mapping(summary.get("observable"), "C0 observable")

    def finite(key: str) -> float:
        value = raw.get(key)
        if isinstance(value, bool):
            raise C2ScoreError(f"C0 parameter {key!r} is invalid.")
        result = float(value)  # type: ignore[arg-type]
        if not math.isfinite(result) or result <= 0.0:
            raise C2ScoreError(f"C0 parameter {key!r} is invalid.")
        return result

    photon_bin = raw.get("photon_bin")
    max_steps = raw.get("max_newton_steps")
    if (
        isinstance(photon_bin, bool)
        or not isinstance(photon_bin, int)
        or photon_bin < 1
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 1
    ):
        raise C2ScoreError("C0 integer parameter closure is invalid.")
    thermal_gap_raw = observable.get("thermal_gap_eV")
    if isinstance(thermal_gap_raw, bool):
        raise C2ScoreError("C0 thermal gap is invalid.")
    thermal_gap = float(thermal_gap_raw)  # type: ignore[arg-type]
    if not math.isfinite(thermal_gap) or thermal_gap <= 0.0:
        raise C2ScoreError("C0 thermal gap is invalid.")
    return AuthorSolveParameters(
        gap_eV=finite("gap_eV"),
        h_eV=finite("h_eV"),
        temperature_K=finite("temperature_K"),
        T_c_K=finite("T_c_K"),
        tau_0_s=finite("tau_0_s"),
        tau_0_pb_s=finite("tau_0_pb_s"),
        tau_l_s=finite("tau_l_s"),
        photon_bin=photon_bin,
        n_bar=finite("n_bar"),
        c_photon_s_inv=finite("c_photon_s_inv"),
        delta0_eV=finite("delta0_eV"),
        thermal_gap_eV=thermal_gap,
        max_newton_steps=max_steps,
        relative_step_threshold=finite("relative_step_threshold"),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=finite("boltzmann_constant_J_per_K"),
            electron_charge_C=finite("electron_charge_C"),
        ),
    )


def _effective_record(
    parameters: AuthorSolveParameters,
) -> tuple[dict[str, object], dict[str, str]]:
    values: dict[str, object] = {
        "T_c_K": parameters.T_c_K,
        "boltzmann_constant_J_per_K": parameters.constants.boltzmann_constant_J_per_K,
        "c_photon_s_inv": parameters.c_photon_s_inv,
        "delta0_eV": parameters.delta0_eV,
        "electron_charge_C": parameters.constants.electron_charge_C,
        "gap_eV": parameters.gap_eV,
        "h_eV": parameters.h_eV,
        "n_bar": parameters.n_bar,
        "tau_0_pb_s": parameters.tau_0_pb_s,
        "tau_0_s": parameters.tau_0_s,
        "tau_l_s": parameters.tau_l_s,
        "temperature_K": parameters.temperature_K,
    }
    return values, {key: value.hex() for key, value in values.items() if isinstance(value, float)}


def _evaluation_arrays(
    *,
    slug: str,
    evaluation: SystemEvaluation,
    n_thermal: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {f"{slug}__n_thermal": np.asarray(n_thermal)}
    for channel_name in CHANNEL_NAMES:
        balance = getattr(evaluation, channel_name)
        for field in BALANCE_FIELDS:
            result[f"{slug}__{channel_name}__{field}"] = np.asarray(getattr(balance, field))
    result[f"{slug}__residual_s_inv"] = np.asarray(evaluation.residual_s_inv)
    return result


def _array_bit_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    """Compare complete canonical NPY identities, not only numeric values."""

    return array_descriptor(np.asarray(reference)) == array_descriptor(
        np.asarray(candidate)
    )


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    ref = np.asarray(reference)
    got = np.asarray(candidate)
    delta = got - ref
    l1 = float(np.sum(np.abs(delta)))
    denominator = max(
        float(np.sum(np.abs(ref))) + float(np.sum(np.abs(got))),
        np.finfo(float).tiny,
    )
    return {
        "bit_exact": _array_bit_exact(ref, got),
        "candidate_descriptor": array_descriptor(got),
        "l1_absolute": l1,
        "linf_absolute": float(np.max(np.abs(delta), initial=0.0)),
        "symmetric_relative_l1": l1 / denominator,
    }


def _channel_metrics(
    arrays: dict[str, np.ndarray],
    *,
    reference_slug: str,
    candidate_slug: str,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for channel_name in CHANNEL_NAMES:
        result[channel_name] = {
            field: _difference(
                arrays[f"{reference_slug}__{channel_name}__{field}"],
                arrays[f"{candidate_slug}__{channel_name}__{field}"],
            )
            for field in BALANCE_FIELDS
        }
    return result


def _all_balance_nonnegative(
    arrays: dict[str, np.ndarray],
    *,
    slug: str,
) -> bool:
    return all(
        bool(np.all(arrays[f"{slug}__{channel}__{field}"] >= 0.0))
        for channel in CHANNEL_NAMES
        for field in ("gain_s_inv", "loss_s_inv")
    )


def _assert_recomputed_array_identity(
    raw_metadata: dict[str, Any],
    raw_arrays: dict[str, np.ndarray],
    recomputed_arrays: dict[str, np.ndarray],
) -> None:
    """Require exact array closure, descriptors, and canonical NPY identities."""

    if set(raw_arrays) != set(recomputed_arrays):
        raise C2ScoreError("Raw C2 array closure is invalid.")
    recomputed_descriptors = {
        name: array_descriptor(expected)
        for name, expected in sorted(recomputed_arrays.items())
    }
    raw_descriptors = _mapping(
        raw_metadata.get("array_descriptors"),
        "Raw C2 array descriptors",
    )
    if not json_value_bit_exact(raw_descriptors, recomputed_descriptors):
        raise C2ScoreError(
            "Raw C2 array descriptors failed independent recomputation."
        )
    for name, expected in recomputed_arrays.items():
        if not _array_bit_exact(raw_arrays[name], expected):
            raise C2ScoreError(
                f"Raw C2 array {name!r} failed independent recomputation."
            )


def _expected_raw_sources() -> dict[str, str]:
    return {
        relative: source_sha256(REPOSITORY_ROOT / relative)
        for relative in sorted(_RAW_SOURCE_RELATIVES)
    }


def build_c2_score(
    c2_bundle_dir: Path,
    *,
    c0_bundle_dir: Path,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
    c1_score_path: Path = DEFAULT_C1_SCORE,
) -> dict[str, Any]:
    """Recompute every C2 frozen channel and return a checked strict score."""

    _assert_source_snapshots()
    c0_summary_path = c0_summary_path.resolve()
    c1_score_path = c1_score_path.resolve()
    raw_metadata, raw_arrays, c2_manifest_sha = load_c2_raw_bundle(c2_bundle_dir)
    c0_metadata, c0_arrays, c0_manifest_sha = load_c0_raw_bundle(c0_bundle_dir)
    c0_summary = load_c0_summary(c0_summary_path)
    checked_c1 = load_c1_score(c1_score_path)
    recomputed_c1 = build_c1_score(
        c0_bundle_dir,
        c0_summary_path=c0_summary_path,
    )
    if checked_c1 != recomputed_c1:
        raise C2ScoreError("Checked C1 nested calculation is not reproducible from C0.")

    c0_raw_binding = _mapping(c0_summary.get("raw_bundle"), "C0 raw binding")
    c1_c0_binding = _mapping(checked_c1.get("c0_binding"), "C1 C0 binding")
    if (
        c0_raw_binding.get("manifest_sha256") != c0_manifest_sha
        or c1_c0_binding.get("raw_manifest_sha256") != c0_manifest_sha
        or c0_summary.get("array_descriptors") != c0_metadata.get("array_descriptors")
    ):
        raise C2ScoreError("C2 parent C0/C1/raw chain is inconsistent.")

    expected_parent_bindings = {
        "c0_raw_manifest_sha256": c0_manifest_sha,
        "c0_summary_path": c0_summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c0_summary_sha256": _file_sha256(c0_summary_path, "C0 summary"),
        "c1_score_path": c1_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c1_score_sha256": _file_sha256(c1_score_path, "C1 score"),
    }
    if not json_value_bit_exact(
        raw_metadata.get("parent_bindings"),
        expected_parent_bindings,
    ):
        raise C2ScoreError("Raw C2 parent binding is forged or stale.")
    if not json_value_bit_exact(raw_metadata.get("sources"), _expected_raw_sources()):
        raise C2ScoreError("Raw C2 source closure is forged, incomplete, or stale.")
    if not json_value_bit_exact(
        raw_metadata.get("source_binding"),
        {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "C2 producer, parameter resolver, parent loaders, author core, "
                "and qpsim parameter sources"
            ),
        },
    ):
        raise C2ScoreError("Raw C2 source-binding contract is invalid.")
    if not json_value_bit_exact(
        raw_metadata.get("stage"),
        {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
        },
    ):
        raise C2ScoreError("Raw C2 stage identity is invalid.")
    expected_coordinate = {
        "carrier_occupation_samples": "author left edges E_i=Delta+i*h",
        "grid_projection": "none",
        "pair_frequency_offset_bins": 0,
        "state_layout": "[f(E_0..E_N-1), n(h..(N-1)h)]",
    }
    if not json_value_bit_exact(
        raw_metadata.get("coordinate_contract"),
        expected_coordinate,
    ):
        raise C2ScoreError("C2 changed a grid/sampling contract outside its scope.")

    parent = _author_parameters_from_summary(c0_summary)
    parent_record = dict(_mapping(c0_summary.get("parameters"), "C0 parameters"))
    parent_record["thermal_gap_eV"] = parent.thermal_gap_eV
    plan = build_c2_parameter_plan(parent_record)
    if plan.parent_author_parameters != parent:
        raise C2ScoreError("Independent C2 parameter resolution changed its parent.")
    if not json_value_bit_exact(
        raw_metadata.get("parameter_plan"),
        plan.as_record(),
    ):
        raise C2ScoreError("Raw C2 parameter plan is forged or stale.")

    frozen_descriptors = {name: array_descriptor(c0_arrays[name]) for name in FROZEN_ARRAY_NAMES}
    for name in FROZEN_ARRAY_NAMES:
        if not _array_bit_exact(raw_arrays[name], c0_arrays[name]):
            raise C2ScoreError(f"Raw C2 did not inherit exact frozen {name} bytes.")
    if not json_value_bit_exact(
        raw_metadata.get("frozen_inputs"),
        {
            "descriptors": frozen_descriptors,
            "mutation_check_after_all_steps": True,
        },
    ):
        raise C2ScoreError("Raw C2 frozen-input identity is forged.")

    raw_steps = raw_metadata.get("steps")
    if not isinstance(raw_steps, list):
        raise C2ScoreError("Raw C2 steps must be a list.")
    step_inputs: list[
        tuple[str, tuple[str, ...], NativeFig6Parameters, AuthorSolveParameters, str]
    ] = [
        (
            "C2a-author-value-plumbing",
            (),
            plan.c2a,
            plan.c2a_author_effective,
            (
                "Native units are emitted, while the effective author-space "
                "parameters remain exact parent bits."
            ),
        )
    ]
    effective_by_id = dict(plan.c2b_author_effective_steps())
    step_inputs.extend(
        (
            step.step_id,
            step.changed_fields,
            step.parameters,
            effective_by_id[step.step_id],
            step.qualification,
        )
        for step in plan.c2b_steps
    )
    if len(raw_steps) != len(step_inputs):
        raise C2ScoreError("Raw C2 step closure is invalid.")

    recomputed_arrays: dict[str, np.ndarray] = {
        name: np.asarray(c0_arrays[name]) for name in FROZEN_ARRAY_NAMES
    }
    expected_step_records: list[dict[str, object]] = []
    structural_descriptors: dict[str, dict[str, object]] | None = None
    operator_scalars_by_step: dict[str, dict[str, object]] = {}
    for index, (step_id, changed_fields, native, effective, qualification) in enumerate(
        step_inputs
    ):
        operator = build_author_operator(c0_arrays["E_left_eV"], effective)
        evaluation = evaluate_author_system(
            operator,
            c0_arrays["f_final"],
            c0_arrays["n_phonon_final"],
            build_update_matrix=False,
        )
        slug = _step_slug(step_id)
        evaluated = _evaluation_arrays(
            slug=slug,
            evaluation=evaluation,
            n_thermal=operator.n_thermal,
        )
        recomputed_arrays.update(evaluated)
        structure = {
            "K_minus": array_descriptor(operator.K_minus),
            "K_plus": array_descriptor(operator.K_plus),
            "rho": array_descriptor(operator.rho),
        }
        if structural_descriptors is None:
            structural_descriptors = structure
        operator_scalars: dict[str, object] = {
            "a_delta": operator.a_delta,
            "pair_frequency_offset_bins": operator.pair_frequency_offset_bins,
            "phonon_prefactor_per_eV_s": operator.phonon_prefactor_per_eV_s,
            "qp_prefactor_s_inv": operator.qp_prefactor_s_inv,
        }
        operator_scalars_by_step[step_id] = operator_scalars
        effective_values, effective_hex = _effective_record(effective)
        expected_step_records.append(
            {
                "array_names": list(evaluated),
                "changed_fields": list(changed_fields),
                "effective_author_units": effective_values,
                "effective_hex": effective_hex,
                "index": index,
                "native_parameters": native.as_record(),
                "operator_scalars": operator_scalars,
                "qualification": qualification,
                "step_id": step_id,
                "structural_descriptors": structure,
            }
        )
    if not json_value_bit_exact(raw_steps, expected_step_records):
        raise C2ScoreError("Raw C2 scientific step metadata is forged or stale.")
    _assert_recomputed_array_identity(
        raw_metadata,
        raw_arrays,
        recomputed_arrays,
    )

    c2a_slug = _step_slug(step_inputs[0][0])
    # C2a is the bit-preserving control. Re-evaluate it through the C0
    # verifier's separately source-bound aliases instead of trusting only the
    # C2 producer/scorer aliases. The underlying author evaluator
    # implementation is intentionally shared, so also certify each net
    # directly against arithmetic on the persisted C0 gain/loss arrays below.
    # Equal-and-opposite channel corruption can leave the total residual
    # unchanged; the combination closes both the alias and common-source cases.
    c0_control_build_operator = c0_summary_verifier.__dict__["build_author_operator"]
    c0_control_evaluate = c0_summary_verifier.__dict__["evaluate_author_system"]
    c2a_control_operator = c0_control_build_operator(
        c0_arrays["E_left_eV"],
        plan.c2a_author_effective,
    )
    c2a_control_evaluation = c0_control_evaluate(
        c2a_control_operator,
        c0_arrays["f_final"],
        c0_arrays["n_phonon_final"],
        build_update_matrix=False,
    )
    c2a_control_checks: dict[str, bool] = {
        "residual_bit_exact_to_c0": bool(
            _array_bit_exact(
                raw_arrays[f"{c2a_slug}__residual_s_inv"],
                c0_arrays["final_residual_s_inv"],
            )
        )
    }
    for channel_name in CHANNEL_NAMES:
        control_balance = getattr(c2a_control_evaluation, channel_name)
        for field in BALANCE_FIELDS:
            key = f"{channel_name}_{field}_bit_exact_to_c0"
            expected = (
                c0_arrays[f"{channel_name}__{field}"]
                if field in {"gain_s_inv", "loss_s_inv"}
                else np.asarray(getattr(control_balance, field))
            )
            c2a_control_checks[key] = bool(
                _array_bit_exact(
                    raw_arrays[f"{c2a_slug}__{channel_name}__{field}"],
                    expected,
                )
            )
        gain = c0_arrays[f"{channel_name}__gain_s_inv"]
        loss = c0_arrays[f"{channel_name}__loss_s_inv"]
        net = raw_arrays[f"{c2a_slug}__{channel_name}__net_s_inv"]
        arithmetic_net = gain - loss
        scale = np.abs(gain) + np.abs(loss)
        tolerance = (
            _ACCEPTANCE_LIMITS["c2a_net_subtraction_roundoff_factor"]
            * np.finfo(float).eps
            * np.maximum(scale, np.finfo(float).tiny)
        )
        c2a_control_checks[
            f"{channel_name}_net_arithmetic_consistent_with_c0_gain_loss"
        ] = bool(np.all(np.abs(net - arithmetic_net) <= tolerance))
    if set(c2a_control_checks) != _C2A_CONTROL_CHECK_KEYS:
        raise C2ScoreError("C2a control-check closure is invalid.")

    step_summaries: list[dict[str, object]] = []
    for index, (step_id, changed_fields, native, _effective, qualification) in enumerate(
        step_inputs
    ):
        slug = _step_slug(step_id)
        prior_slug = c2a_slug if index == 0 else _step_slug(step_inputs[index - 1][0])
        step_summaries.append(
            {
                "all_gain_loss_nonnegative": _all_balance_nonnegative(
                    raw_arrays,
                    slug=slug,
                ),
                "changed_fields": list(changed_fields),
                "channel_difference_from_author": _channel_metrics(
                    raw_arrays,
                    reference_slug=c2a_slug,
                    candidate_slug=slug,
                ),
                "channel_difference_from_previous": _channel_metrics(
                    raw_arrays,
                    reference_slug=prior_slug,
                    candidate_slug=slug,
                ),
                "eq35_t_star_over_delta": native.eq35_t_star_over_delta,
                "n_thermal_difference_from_author": _difference(
                    raw_arrays[f"{c2a_slug}__n_thermal"],
                    raw_arrays[f"{slug}__n_thermal"],
                ),
                "operator_scalars": operator_scalars_by_step[step_id],
                "qualification": qualification,
                "residual_difference_from_author": _difference(
                    raw_arrays[f"{c2a_slug}__residual_s_inv"],
                    raw_arrays[f"{slug}__residual_s_inv"],
                ),
                "residual_difference_from_previous": _difference(
                    raw_arrays[f"{prior_slug}__residual_s_inv"],
                    raw_arrays[f"{slug}__residual_s_inv"],
                ),
                "step_id": step_id,
            }
        )

    # Expected locality after ordering the explicit native energy carrier
    # before modern k_B. These exact checks prevent a controlled parameter
    # substep from smuggling in another scalar or operator family.
    explicit_slug = _step_slug(plan.c2b_steps[0].step_id)
    kb_slug = _step_slug(plan.c2b_steps[1].step_id)
    c_slug = _step_slug(plan.c2b_steps[2].step_id)
    pb_slug = _step_slug(plan.c2b_steps[3].step_id)
    tc_slug = _step_slug(plan.c2b_steps[4].step_id)

    def channels_exact(left_slug: str, right_slug: str, names: tuple[str, ...]) -> bool:
        return all(
            _array_bit_exact(
                raw_arrays[f"{left_slug}__{channel}__{field}"],
                raw_arrays[f"{right_slug}__{channel}__{field}"],
            )
            for channel in names
            for field in BALANCE_FIELDS
        )

    def every_channel_changes(
        left_slug: str,
        right_slug: str,
        names: tuple[str, ...],
    ) -> bool:
        return all(
            not channels_exact(left_slug, right_slug, (channel,))
            for channel in names
        )

    def fields_exact(
        left_slug: str,
        right_slug: str,
        *,
        channel: str,
        fields: tuple[str, ...],
    ) -> bool:
        return all(
            _array_bit_exact(
                raw_arrays[f"{left_slug}__{channel}__{field}"],
                raw_arrays[f"{right_slug}__{channel}__{field}"],
            )
            for field in fields
        )

    def fields_change(
        left_slug: str,
        right_slug: str,
        *,
        channel: str,
        fields: tuple[str, ...],
    ) -> bool:
        return all(
            not _array_bit_exact(
                raw_arrays[f"{left_slug}__{channel}__{field}"],
                raw_arrays[f"{right_slug}__{channel}__{field}"],
            )
            for field in fields
        )

    all_channels = tuple(CHANNEL_NAMES)
    no_photon = tuple(name for name in CHANNEL_NAMES if name != "qp_photon")
    no_phonon_kernel = tuple(
        name for name in CHANNEL_NAMES if name not in {"phonon_scattering", "phonon_pair"}
    )
    no_qp_phonon = tuple(name for name in CHANNEL_NAMES if name not in {"qp_scattering", "qp_pair"})
    kb_independent_channels = (
        "qp_photon",
        "phonon_scattering",
        "phonon_pair",
    )
    locality_checks = {
        "explicit_energy_native_step_effective_noop": channels_exact(
            c2a_slug,
            explicit_slug,
            all_channels,
        ),
        "modern_kb_changes_thermal_occupation": not _array_bit_exact(
            raw_arrays[f"{explicit_slug}__n_thermal"],
            raw_arrays[f"{kb_slug}__n_thermal"],
        ),
        "modern_kb_changes_qp_phonon_channels": all(
            fields_change(
                explicit_slug,
                kb_slug,
                channel=channel,
                fields=BALANCE_FIELDS,
            )
            for channel in ("qp_scattering", "qp_pair")
        ),
        "modern_kb_changes_phonon_escape_gain_and_net": fields_change(
            explicit_slug,
            kb_slug,
            channel="phonon_escape",
            fields=("gain_s_inv", "net_s_inv"),
        ),
        "modern_kb_preserves_phonon_escape_loss_exact": fields_exact(
            explicit_slug,
            kb_slug,
            channel="phonon_escape",
            fields=("loss_s_inv",),
        ),
        "modern_kb_leaves_kb_independent_channels_exact": channels_exact(
            explicit_slug,
            kb_slug,
            kb_independent_channels,
        ),
        "literal_c_changes_photon": not channels_exact(
            kb_slug,
            c_slug,
            ("qp_photon",),
        ),
        "literal_c_leaves_nonphoton_channels_exact": channels_exact(
            kb_slug,
            c_slug,
            no_photon,
        ),
        "tau_pb_changes_phonon_kernel_channels": every_channel_changes(
            c_slug,
            pb_slug,
            ("phonon_scattering", "phonon_pair"),
        ),
        "tau_pb_leaves_other_channels_exact": channels_exact(
            c_slug,
            pb_slug,
            no_phonon_kernel,
        ),
        "finite_cutoff_tc_changes_qp_phonon_channels": every_channel_changes(
            pb_slug,
            tc_slug,
            ("qp_scattering", "qp_pair"),
        ),
        "finite_cutoff_tc_leaves_other_channels_exact": channels_exact(
            pb_slug,
            tc_slug,
            no_qp_phonon,
        ),
    }

    point_selection = _mapping(c0_metadata.get("point_selection"), "C0 point selection")
    author_coordinate = point_selection.get("actual_author_t_star_over_delta")
    if isinstance(author_coordinate, bool):
        raise C2ScoreError("C0 author Eq. 35 coordinate is invalid.")
    author_coordinate_value = float(author_coordinate)  # type: ignore[arg-type]
    if not math.isfinite(author_coordinate_value) or author_coordinate_value <= 0.0:
        raise C2ScoreError("C0 author Eq. 35 coordinate is invalid.")
    final_coordinate = plan.c2b.eq35_t_star_over_delta
    fixed_coordinate_nbar = plan.c2b.n_bar * (author_coordinate_value / final_coordinate) ** 6

    frozen_observable = _mapping(
        _mapping(recomputed_c1.get("calculation"), "C1 calculation").get("qpsim_c1"),
        "C1 qpsim calculation",
    )
    first_operator_scalars = _mapping(
        expected_step_records[0].get("operator_scalars"),
        "first C2 operator scalars",
    )
    checks = {
        "c0_c1_parent_chain_recomputed": True,
        "c2a_control_bit_exact": all(c2a_control_checks.values()),
        "fixed_nbar_preserved": plan.c2b.n_bar == parent.n_bar,
        "frozen_inputs_bit_exact": True,
        "parameter_axis_shift_is_nonzero": final_coordinate != author_coordinate_value,
        "raw_arrays_independently_recomputed": True,
        "raw_metadata_independently_recomputed": True,
        "structural_grid_and_dimensionless_kernels_bit_exact": all(
            record["structural_descriptors"] == structural_descriptors
            and _mapping(
                record.get("operator_scalars"),
                "C2 operator scalars",
            ).get("a_delta")
            == first_operator_scalars.get("a_delta")
            and _mapping(
                record.get("operator_scalars"),
                "C2 operator scalars",
            ).get("pair_frequency_offset_bins")
            == 0
            for record in expected_step_records
        ),
        "all_gain_loss_arrays_nonnegative": all(
            summary["all_gain_loss_nonnegative"] is True for summary in step_summaries
        ),
        **locality_checks,
    }
    accepted = all(checks.values())
    if not accepted:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise C2ScoreError(f"C2 acceptance checks failed: {failed!r}.")

    comparison = {
        "c2a_control_checks": c2a_control_checks,
        "locality_checks": locality_checks,
        "raw_array_count": len(raw_arrays),
        "step_count": len(step_inputs),
    }
    score = {
        "acceptance": {
            "accepted": accepted,
            "checks": checks,
            "limits": dict(_ACCEPTANCE_LIMITS),
        },
        "comparison": comparison,
        "frozen_inputs": {
            "descriptors": frozen_descriptors,
            "state_policy": (
                "C0 E_left, driven f, phonon n, and thermal f are immutable; "
                "no projection and no nonlinear solve"
            ),
        },
        "limitations": {
            "scope": "one accepted C0 point only",
            "statement": (
                "C2 completes only the frozen-state parameter/operator "
                "differential. Its nonzero candidate residual is expected and "
                "is not a solved endpoint. No C2 root or plotted ordinate is "
                "claimed; the same-seed nonlinear comparison belongs after "
                "the frozen operator ladder."
            ),
        },
        "observable_control": {
            "c1_frozen_result": frozen_observable,
            "policy": (
                "The C1 qpsim observable and inherited driven/thermal arrays "
                "are recomputed unchanged; parameter-consistent thermal-state "
                "replacement is outside C2."
            ),
        },
        "parameter_axis": {
            "author_authenticated_t_star_over_delta": author_coordinate_value,
            "author_native_reconstruction_t_star_over_delta": (plan.c2a.eq35_t_star_over_delta),
            "author_nbar": parent.n_bar,
            "fixed_author_coordinate_qpsim_nbar": fixed_coordinate_nbar,
            "qpsim_fixed_nbar_t_star_over_delta": final_coordinate,
            "qpsim_minus_author_at_fixed_nbar": (final_coordinate - author_coordinate_value),
            "qpsim_relative_shift_at_fixed_nbar": (
                final_coordinate / author_coordinate_value - 1.0
            ),
            "selected_comparison": "fixed authenticated author n_bar",
        },
        "parent_bindings": expected_parent_bindings,
        "raw_bundle": {
            "manifest_sha256": c2_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "schema": SCHEMA,
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
        "steps": step_summaries,
        "structural_identity": {
            "descriptors": structural_descriptors,
            "grid_projection": "none",
            "pair_frequency_offset_bins": 0,
        },
    }
    _assert_source_snapshots()
    return score


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Serialize a checked C2 score deterministically."""

    return (json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_c2_score(
    c2_bundle_dir: Path,
    output_path: Path,
    *,
    c0_bundle_dir: Path,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
    c1_score_path: Path = DEFAULT_C1_SCORE,
) -> Path:
    """Build and exclusively write one checked C2 score."""

    _assert_source_snapshots()
    score = build_c2_score(
        c2_bundle_dir,
        c0_bundle_dir=c0_bundle_dir,
        c0_summary_path=c0_summary_path,
        c1_score_path=c1_score_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_score_bytes(score))
    _assert_source_snapshots()
    return output_path


def load_c2_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Strictly load a checked C2 score and bind it to the selected receipt.

    The default receipt is the repository trust root.  Supplying a custom
    ``receipt_path`` deliberately replaces that root (for tests or an
    off-repository replica), so the caller then owns its authenticity.
    """

    if path.is_symlink() or not path.is_file():
        raise C2ScoreError("Checked C2 score is missing, unsafe, or a symlink.")
    score_raw = path.read_bytes()
    score = _parse_json(score_raw, "checked C2 score")
    _exact_keys(score, _TOP_LEVEL_KEYS, "checked C2 score")
    if score.get("schema") != SCHEMA:
        raise C2ScoreError("Checked C2 score schema is unsupported.")
    stage = _mapping(score.get("stage"), "stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C2ScoreError("Checked C2 stage identity is invalid.")
    acceptance = _mapping(score.get("acceptance"), "acceptance")
    _exact_keys(acceptance, {"accepted", "checks", "limits"}, "acceptance")
    if acceptance.get("accepted") is not True:
        raise C2ScoreError("Checked C2 score is not accepted.")
    if acceptance.get("limits") != _ACCEPTANCE_LIMITS:
        raise C2ScoreError("Checked C2 acceptance limits are invalid.")
    checks = _mapping(acceptance.get("checks"), "acceptance.checks")
    if set(checks) != _ACCEPTANCE_CHECK_KEYS or any(value is not True for value in checks.values()):
        raise C2ScoreError("Checked C2 acceptance checks are invalid.")

    comparison = _mapping(score.get("comparison"), "comparison")
    _exact_keys(
        comparison,
        {
            "c2a_control_checks",
            "locality_checks",
            "raw_array_count",
            "step_count",
        },
        "comparison",
    )
    c2a_checks = _mapping(
        comparison.get("c2a_control_checks"),
        "comparison.c2a_control_checks",
    )
    locality = _mapping(
        comparison.get("locality_checks"),
        "comparison.locality_checks",
    )
    if (
        set(c2a_checks) != _C2A_CONTROL_CHECK_KEYS
        or any(value is not True for value in c2a_checks.values())
        or set(locality)
        != {
            "explicit_energy_native_step_effective_noop",
            "finite_cutoff_tc_changes_qp_phonon_channels",
            "finite_cutoff_tc_leaves_other_channels_exact",
            "literal_c_changes_photon",
            "literal_c_leaves_nonphoton_channels_exact",
            "modern_kb_changes_phonon_escape_gain_and_net",
            "modern_kb_changes_qp_phonon_channels",
            "modern_kb_changes_thermal_occupation",
            "modern_kb_leaves_kb_independent_channels_exact",
            "modern_kb_preserves_phonon_escape_loss_exact",
            "tau_pb_changes_phonon_kernel_channels",
            "tau_pb_leaves_other_channels_exact",
        }
        or any(value is not True for value in locality.values())
        or comparison.get("raw_array_count") != 124
        or comparison.get("step_count") != 6
    ):
        raise C2ScoreError("Checked C2 comparison closure is invalid.")

    frozen_inputs = _mapping(score.get("frozen_inputs"), "frozen_inputs")
    _exact_keys(frozen_inputs, {"descriptors", "state_policy"}, "frozen_inputs")
    if frozen_inputs.get("state_policy") != (
        "C0 E_left, driven f, phonon n, and thermal f are immutable; "
        "no projection and no nonlinear solve"
    ):
        raise C2ScoreError("Checked C2 frozen-state policy is invalid.")
    descriptors = _mapping(
        frozen_inputs.get("descriptors"),
        "frozen_inputs.descriptors",
    )
    if set(descriptors) != set(FROZEN_ARRAY_NAMES):
        raise C2ScoreError("Checked C2 frozen descriptor closure is invalid.")

    limitations = _mapping(score.get("limitations"), "limitations")
    _exact_keys(limitations, {"scope", "statement"}, "limitations")
    if limitations.get("scope") != "one accepted C0 point only" or (
        "No C2 root or plotted ordinate is claimed" not in str(limitations.get("statement"))
    ):
        raise C2ScoreError("Checked C2 limitations are invalid.")

    parameter_axis = _mapping(score.get("parameter_axis"), "parameter_axis")
    _exact_keys(
        parameter_axis,
        {
            "author_authenticated_t_star_over_delta",
            "author_native_reconstruction_t_star_over_delta",
            "author_nbar",
            "fixed_author_coordinate_qpsim_nbar",
            "qpsim_fixed_nbar_t_star_over_delta",
            "qpsim_minus_author_at_fixed_nbar",
            "qpsim_relative_shift_at_fixed_nbar",
            "selected_comparison",
        },
        "parameter_axis",
    )
    if parameter_axis.get("selected_comparison") != "fixed authenticated author n_bar":
        raise C2ScoreError("Checked C2 abscissa convention is invalid.")
    for key in (
        "author_authenticated_t_star_over_delta",
        "author_native_reconstruction_t_star_over_delta",
        "author_nbar",
        "fixed_author_coordinate_qpsim_nbar",
        "qpsim_fixed_nbar_t_star_over_delta",
    ):
        _finite_scalar(parameter_axis.get(key), f"parameter_axis.{key}", positive=True)
    for key in (
        "qpsim_minus_author_at_fixed_nbar",
        "qpsim_relative_shift_at_fixed_nbar",
    ):
        _finite_scalar(parameter_axis.get(key), f"parameter_axis.{key}")

    steps = score.get("steps")
    if not isinstance(steps, list) or len(steps) != 6:
        raise C2ScoreError("Checked C2 step closure is invalid.")
    expected_step_ids = (
        "C2a-author-value-plumbing",
        "C2b1-explicit-energy-literals",
        "C2b2-modern-kB",
        "C2b3-literal-photon-coupling",
        "C2b4-declared-pair-breaking-time",
        "C2b5-finite-cutoff-critical-temperature",
    )
    expected_changed_fields: tuple[list[str], ...] = (
        [],
        ["delta0_ueV", "gap_ueV"],
        ["kB_ueV_per_K"],
        ["c_photon_ns_inv"],
        ["tau_0_pb_ns"],
        ["T_c_K"],
    )
    step_keys = {
        "all_gain_loss_nonnegative",
        "changed_fields",
        "channel_difference_from_author",
        "channel_difference_from_previous",
        "eq35_t_star_over_delta",
        "n_thermal_difference_from_author",
        "operator_scalars",
        "qualification",
        "residual_difference_from_author",
        "residual_difference_from_previous",
        "step_id",
    }
    for index, raw_step in enumerate(steps):
        step = _mapping(raw_step, f"steps[{index}]")
        _exact_keys(step, step_keys, f"steps[{index}]")
        if (
            step.get("step_id") != expected_step_ids[index]
            or step.get("changed_fields") != expected_changed_fields[index]
            or step.get("all_gain_loss_nonnegative") is not True
            or not isinstance(step.get("qualification"), str)
        ):
            raise C2ScoreError(f"Checked C2 step {index} identity is invalid.")
        _finite_scalar(
            step.get("eq35_t_star_over_delta"),
            f"steps[{index}].eq35_t_star_over_delta",
            positive=True,
        )
        for comparison_name in (
            "channel_difference_from_author",
            "channel_difference_from_previous",
        ):
            channels = _mapping(
                step.get(comparison_name),
                f"steps[{index}].{comparison_name}",
            )
            if set(channels) != set(CHANNEL_NAMES):
                raise C2ScoreError(f"Checked C2 step {index} channel closure is invalid.")

    structural = _mapping(score.get("structural_identity"), "structural_identity")
    _exact_keys(
        structural,
        {"descriptors", "grid_projection", "pair_frequency_offset_bins"},
        "structural_identity",
    )
    if (
        structural.get("grid_projection") != "none"
        or structural.get("pair_frequency_offset_bins") != 0
        or set(_mapping(structural.get("descriptors"), "structural descriptors"))
        != {"K_minus", "K_plus", "rho"}
    ):
        raise C2ScoreError("Checked C2 structural identity is invalid.")

    sources = _mapping(score.get("sources"), "sources")
    expected_paths = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in _SOURCE_PATHS}
    if set(sources) != expected_paths:
        raise C2ScoreError("Checked C2 source closure is incomplete.")
    for relative, digest in sources.items():
        if source_sha256(REPOSITORY_ROOT / relative) != _sha256(
            digest,
            f"sources.{relative}",
        ):
            raise C2ScoreError("Checked C2 source binding is stale.")

    parent = _mapping(score.get("parent_bindings"), "parent_bindings")
    _exact_keys(
        parent,
        {
            "c0_raw_manifest_sha256",
            "c0_summary_path",
            "c0_summary_sha256",
            "c1_score_path",
            "c1_score_sha256",
        },
        "parent_bindings",
    )
    expected_c0_path = DEFAULT_C0_SUMMARY.relative_to(REPOSITORY_ROOT).as_posix()
    expected_c1_path = DEFAULT_C1_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
    if (
        parent.get("c0_summary_path") != expected_c0_path
        or parent.get("c1_score_path") != expected_c1_path
    ):
        raise C2ScoreError("Checked C2 must bind the canonical C0/C1 paths.")
    if _file_sha256(DEFAULT_C0_SUMMARY, "C0 summary") != _sha256(
        parent.get("c0_summary_sha256"),
        "parent_bindings.c0_summary_sha256",
    ):
        raise C2ScoreError("Checked C2 C0-summary binding is stale.")
    if _file_sha256(DEFAULT_C1_SCORE, "C1 score") != _sha256(
        parent.get("c1_score_sha256"),
        "parent_bindings.c1_score_sha256",
    ):
        raise C2ScoreError("Checked C2 C1-score binding is stale.")
    c0_summary = load_c0_summary(DEFAULT_C0_SUMMARY)
    c1_score = load_c1_score(DEFAULT_C1_SCORE)
    observable_control = _mapping(score.get("observable_control"), "observable_control")
    _exact_keys(
        observable_control,
        {"c1_frozen_result", "policy"},
        "observable_control",
    )
    expected_c1_result = _mapping(
        _mapping(c1_score.get("calculation"), "bound C1 calculation").get("qpsim_c1"),
        "bound C1 qpsim result",
    )
    if observable_control.get("c1_frozen_result") != expected_c1_result or observable_control.get(
        "policy"
    ) != (
        "The C1 qpsim observable and inherited driven/thermal arrays "
        "are recomputed unchanged; parameter-consistent thermal-state "
        "replacement is outside C2."
    ):
        raise C2ScoreError("Checked C2 observable control is invalid.")
    c0_raw = _mapping(c0_summary.get("raw_bundle"), "bound C0 raw bundle")
    c1_c0 = _mapping(c1_score.get("c0_binding"), "bound C1 C0 binding")
    parent_raw_sha = _sha256(
        parent.get("c0_raw_manifest_sha256"),
        "parent_bindings.c0_raw_manifest_sha256",
    )
    if (
        c0_raw.get("manifest_sha256") != parent_raw_sha
        or c1_c0.get("raw_manifest_sha256") != parent_raw_sha
    ):
        raise C2ScoreError("Checked C2 raw-parent binding is inconsistent.")
    raw_bundle = _mapping(score.get("raw_bundle"), "raw_bundle")
    _exact_keys(raw_bundle, {"manifest_sha256", "schema"}, "raw_bundle")
    _sha256(raw_bundle.get("manifest_sha256"), "raw_bundle.manifest_sha256")
    if raw_bundle.get("schema") != RAW_SCHEMA:
        raise C2ScoreError("Checked C2 raw-bundle schema is invalid.")
    receipt = _load_c2_receipt(receipt_path)
    receipt_score = _mapping(
        receipt.get("checked_score"),
        "receipt.checked_score",
    )
    if hashlib.sha256(score_raw).hexdigest() != receipt_score.get("file_sha256"):
        raise C2ScoreError(
            "Checked C2 score bytes do not match the committed raw-manifest receipt."
        )
    receipt_raw = _mapping(receipt.get("raw_bundle"), "receipt.raw_bundle")
    if raw_bundle != receipt_raw:
        raise C2ScoreError(
            "Checked C2 raw-bundle binding does not match the committed receipt."
        )
    return score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c0-bundle", type=Path, required=True)
    parser.add_argument("--c0-summary", type=Path, default=DEFAULT_C0_SUMMARY)
    parser.add_argument("--c1-score", type=Path, default=DEFAULT_C1_SCORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c2_score(
            args.c2_bundle,
            args.output,
            c0_bundle_dir=args.c0_bundle,
            c0_summary_path=args.c0_summary,
            c1_score_path=args.c1_score,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
