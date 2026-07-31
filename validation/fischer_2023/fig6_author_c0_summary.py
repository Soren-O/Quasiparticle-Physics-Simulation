"""Strictly verify a raw Fischer Figure 6 C0 bundle and emit its checked score."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_c0_bundle import (
    COMPARISON_STAGE_ID,
    PAPER_ORACLE_TARGET_T_STAR_OVER_DELTA,
    STAGE_ID,
    parameters_from_a1_metadata,
)
from validation.fischer_2023.fig6_author_c0_bundle import (
    SCHEMA as RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c0_bundle import (
    SOURCE_BINDING as RAW_SOURCE_BINDING,
)
from validation.fischer_2023.fig6_author_frozen_state import (
    load_author_point_bundle,
    validate_bundle_anchor,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorSolveParameters,
    ChannelBalance,
    author_direct_gap,
    author_direct_gap_integral,
    author_fermi_occupation,
    author_newton_delta,
    build_author_operator,
    evaluate_author_system,
    qp_number_diagnostics,
)
from validation.source_provenance import source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCHEMA = "qpsim.fischer2023.fig6-author-c0-summary.v1"
DEFAULT_SUMMARY = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c0-author-equivalent-score.json"
)

_CHANNEL_PREFIXES = (
    "qp_photon",
    "qp_scattering",
    "qp_pair",
    "phonon_scattering",
    "phonon_pair",
    "phonon_escape",
)
_ARRAY_NAMES = {
    "E_left_eV",
    "omega_phonon_eV",
    "initial_state",
    "a1_final_state",
    "state_history",
    "newton_deltas",
    "relative_qp_steps",
    "linear_solve_backward_errors",
    "f_final",
    "n_phonon_final",
    "thermal_f",
    "final_residual_s_inv",
    *{f"{prefix}__{side}_s_inv" for prefix in _CHANNEL_PREFIXES for side in ("gain", "loss")},
}
_RAW_METADATA_KEYS = {
    "array_descriptors",
    "comparison",
    "coordinate_contract",
    "diagnostics",
    "input_bundle",
    "observable",
    "parameters",
    "point_selection",
    "qualification",
    "runtime",
    "schema",
    "solver",
    "source_binding",
    "sources",
    "stage",
}
_SUMMARY_KEYS = {
    "acceptance",
    "a1_binding",
    "array_descriptors",
    "comparison",
    "limitations",
    "observable",
    "parameters",
    "raw_bundle",
    "schema",
    "solver_verification",
    "stage",
    "summary_sources",
}
_RAW_PRODUCER_SOURCE_RELATIVES = frozenset(
    {
        "validation/author_source.py",
        "validation/reproduction_ladder.py",
        "validation/source_provenance.py",
        "validation/fischer_2023/fig6_author_adapter.py",
        "validation/fischer_2023/fig6_author_c0_bundle.py",
        "validation/fischer_2023/fig6_author_frozen_state.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    }
)
_SUMMARY_SOURCE_PATHS = (
    Path(__file__).resolve(),
    *(REPOSITORY_ROOT / relative for relative in sorted(_RAW_PRODUCER_SOURCE_RELATIVES)),
)
_ACCEPTANCE_LIMITS = {
    "a1_final_state_relative_l2_max": 1.0e-11,
    "a1_ordinate_absolute_error_max": 1.0e-10,
    "cancellation_residual_over_turnover_max": 1.0e-8,
    "newton_delta_symmetric_relative_max": 1.0e-12,
    "linear_solve_backward_error_max": 1.0e-12,
}


class C0SummaryError(ValueError):
    """A C0 raw bundle or checked summary is malformed or unsupported."""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C0SummaryError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _finite_json(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise C0SummaryError(f"Non-finite JSON number {value!r}.")
    return result


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                C0SummaryError(f"Non-finite JSON constant {token!r}.")
            ),
            parse_float=_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C0SummaryError(f"Cannot parse {label}: {exc}.") from exc
    if not isinstance(value, dict):
        raise C0SummaryError(f"{label} must be a JSON object.")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C0SummaryError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise C0SummaryError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(value)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C0SummaryError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise C0SummaryError(f"{label} must be a finite real scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C0SummaryError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise C0SummaryError(f"{label} must be finite.")
    return result


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _read_regular_file_once(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise C0SummaryError(f"{label} is missing, unsafe, or a symlink.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise C0SummaryError(f"Cannot read {label}: {exc}.") from exc


def _load_raw_bundle(
    raw_bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    root = raw_bundle_dir.resolve()
    if raw_bundle_dir.is_symlink() or not root.is_dir():
        raise C0SummaryError("C0 raw bundle is missing, unsafe, or a symlink.")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C0 manifest")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = _parse_json(manifest_raw, "C0 manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C0 manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C0SummaryError("C0 raw schema is unsupported.")
    metadata = _mapping(manifest.get("metadata"), "C0 metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C0 metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C0SummaryError("C0 metadata schema is unsupported.")
    files = _mapping(manifest.get("files"), "C0 files")
    expected_files = {f"{name}.npy" for name in _ARRAY_NAMES}
    if set(files) != expected_files:
        raise C0SummaryError("C0 raw bundle does not have the exact array closure.")
    actual_entries = {
        path.name for path in root.iterdir() if path.is_file() and not path.is_symlink()
    }
    if actual_entries != {"manifest.json", *expected_files}:
        raise C0SummaryError("C0 raw directory contains missing or extra files.")

    arrays: dict[str, np.ndarray] = {}
    for name in sorted(_ARRAY_NAMES):
        filename = f"{name}.npy"
        record = _mapping(files[filename], f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        raw = _read_regular_file_once(root / filename, filename)
        if len(raw) != record.get("size_bytes"):
            raise C0SummaryError(f"{filename} size does not match its manifest.")
        if hashlib.sha256(raw).hexdigest() != _sha256(
            record.get("sha256"),
            f"files.{filename}.sha256",
        ):
            raise C0SummaryError(f"{filename} SHA-256 does not match its manifest.")
        try:
            array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
        except (ValueError, EOFError) as exc:
            raise C0SummaryError(f"{filename} is not a safe NPY array.") from exc
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.floating):
            raise C0SummaryError(f"{filename} must retain a real floating dtype.")
        if np.any(~np.isfinite(array)):
            raise C0SummaryError(f"{filename} contains non-finite values.")
        arrays[name] = array

    descriptors = _mapping(metadata.get("array_descriptors"), "array_descriptors")
    if set(descriptors) != _ARRAY_NAMES:
        raise C0SummaryError("C0 array descriptors do not have the exact closure.")
    for name, array in arrays.items():
        if descriptors[name] != _descriptor(array):
            raise C0SummaryError(f"array_descriptors.{name} is forged or stale.")
    return metadata, arrays, manifest_sha


def load_c0_raw_bundle(
    raw_bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one C0 raw bundle from single-read authenticated bytes."""

    return _load_raw_bundle(raw_bundle_dir)


def _compare_exact_mapping(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise C0SummaryError(f"{label} does not match the independently derived value.")


def _symmetric_relative(reference: np.ndarray, candidate: np.ndarray) -> float:
    difference = float(np.linalg.norm(candidate - reference))
    scale = max(
        float(np.linalg.norm(reference)),
        float(np.linalg.norm(candidate)),
        np.finfo(float).tiny,
    )
    return difference / scale


def _balance_from_arrays(
    arrays: dict[str, np.ndarray],
    prefix: str,
) -> ChannelBalance:
    gain = arrays[f"{prefix}__gain_s_inv"]
    loss = arrays[f"{prefix}__loss_s_inv"]
    return ChannelBalance(
        gain_s_inv=gain,
        loss_s_inv=loss,
        net_s_inv=gain - loss,
    )


def _certificate(
    balances: tuple[ChannelBalance, ...],
) -> dict[str, float]:
    gains = np.sum(np.stack([balance.gain_s_inv for balance in balances]), axis=0)
    losses = np.sum(np.stack([balance.loss_s_inv for balance in balances]), axis=0)
    residual = gains - losses
    turnover = float(np.sum(np.abs(gains)) + np.sum(np.abs(losses)))
    residual_l1 = float(np.sum(np.abs(residual)))
    return {
        "residual_l1_s_inv": residual_l1,
        "residual_over_turnover": residual_l1 / max(turnover, np.finfo(float).tiny),
        "turnover_l1_s_inv": turnover,
    }


def _expected_parameters(
    params: AuthorSolveParameters,
    size: int,
) -> dict[str, object]:
    return {
        "T_c_K": params.T_c_K,
        "boltzmann_constant_J_per_K": (params.constants.boltzmann_constant_J_per_K),
        "c_photon_s_inv": params.c_photon_s_inv,
        "delta0_eV": params.delta0_eV,
        "electron_charge_C": params.constants.electron_charge_C,
        "gap_eV": params.gap_eV,
        "h_eV": params.h_eV,
        "max_newton_steps": params.max_newton_steps,
        "n_bar": params.n_bar,
        "num_energy_cells": size,
        "photon_bin": params.photon_bin,
        "relative_step_threshold": params.relative_step_threshold,
        "tau_0_pb_s": params.tau_0_pb_s,
        "tau_0_s": params.tau_0_s,
        "tau_l_s": params.tau_l_s,
        "temperature_K": params.temperature_K,
    }


def build_c0_summary(
    raw_bundle_dir: Path,
    *,
    a1_bundle_dir: Path,
    anchor_path: Path,
) -> dict[str, Any]:
    """Independently verify the raw C0 replay and return a checked score."""

    metadata, arrays, raw_manifest_sha = _load_raw_bundle(raw_bundle_dir)
    a1 = load_author_point_bundle(a1_bundle_dir)
    anchor_binding = validate_bundle_anchor(a1, anchor_path)
    parameters = parameters_from_a1_metadata(a1.metadata)
    size = np.asarray(a1.arrays["E_qp"]).size

    raw_sources = _mapping(metadata.get("sources"), "sources")
    if set(raw_sources) != _RAW_PRODUCER_SOURCE_RELATIVES:
        raise C0SummaryError(
            "C0 source closure does not match the explicit producer dependency set."
        )
    expected_sources = {
        relative: source_sha256(REPOSITORY_ROOT / relative)
        for relative in sorted(_RAW_PRODUCER_SOURCE_RELATIVES)
    }
    if raw_sources != expected_sources:
        raise C0SummaryError(
            "C0 source closure is incomplete, stale, or does not match live bytes."
        )
    _compare_exact_mapping(
        metadata.get("source_binding"),
        RAW_SOURCE_BINDING,
        "source_binding",
    )
    _compare_exact_mapping(
        metadata.get("stage"),
        {
            "comparison_stage_id": COMPARISON_STAGE_ID,
            "evidence_class": "clean_room_author_equivalent_numerics",
            "parent_stage_id": None,
            "stage_id": STAGE_ID,
        },
        "stage",
    )
    _compare_exact_mapping(
        metadata.get("input_bundle"),
        {
            "anchor": anchor_binding,
            "a1_manifest_sha256": a1.manifest_sha256,
        },
        "input_bundle",
    )
    expected_parameters = _expected_parameters(parameters, size)
    _compare_exact_mapping(metadata.get("parameters"), expected_parameters, "parameters")
    input_metadata = _mapping(a1.metadata.get("input"), "A1 input")
    _compare_exact_mapping(
        metadata.get("point_selection"),
        {
            "actual_author_t_star_over_delta": _finite(
                a1.metadata.get("actual_t_star_over_delta"),
                "actual_t_star_over_delta",
            ),
            "nearest_author_sweep_index": input_metadata.get("sweep_index"),
            "paper_oracle_target_t_star_over_delta": (PAPER_ORACLE_TARGET_T_STAR_OVER_DELTA),
        },
        "point_selection",
    )

    expected_shapes = {
        "E_left_eV": (size,),
        "omega_phonon_eV": (size - 1,),
        "initial_state": (2 * size - 1,),
        "a1_final_state": (2 * size - 1,),
        "f_final": (size,),
        "n_phonon_final": (size - 1,),
        "thermal_f": (size,),
        "final_residual_s_inv": (2 * size - 1,),
        "relative_qp_steps": None,
        "linear_solve_backward_errors": None,
        "state_history": None,
        "newton_deltas": None,
    }
    for prefix in _CHANNEL_PREFIXES[:3]:
        expected_shapes[f"{prefix}__gain_s_inv"] = (size,)
        expected_shapes[f"{prefix}__loss_s_inv"] = (size,)
    for prefix in _CHANNEL_PREFIXES[3:]:
        expected_shapes[f"{prefix}__gain_s_inv"] = (size - 1,)
        expected_shapes[f"{prefix}__loss_s_inv"] = (size - 1,)
    for name, shape in expected_shapes.items():
        if shape is not None and arrays[name].shape != shape:
            raise C0SummaryError(f"{name} has the wrong native shape.")

    E_left = arrays["E_left_eV"]
    initial_state = arrays["initial_state"]
    a1_final = np.concatenate(
        (np.asarray(a1.arrays["f_final"]), np.asarray(a1.arrays["n_phonon_final"]))
    )
    if not np.array_equal(E_left, np.asarray(a1.arrays["E_qp"])):
        raise C0SummaryError("C0 E_left_eV does not exactly inherit A1.")
    if not np.array_equal(
        arrays["omega_phonon_eV"],
        np.asarray(a1.arrays["omega_phonon"]),
    ):
        raise C0SummaryError("C0 omega_phonon_eV does not exactly inherit A1.")
    if not np.array_equal(initial_state, np.asarray(a1.arrays["initial_state"])):
        raise C0SummaryError("C0 initial_state does not exactly inherit A1.")
    if not np.array_equal(arrays["a1_final_state"], a1_final):
        raise C0SummaryError("C0 copied A1 final state is not exact.")

    operator = build_author_operator(E_left, parameters)
    thermal_f = author_fermi_occupation(
        E_left,
        parameters.temperature_K,
        parameters.constants,
    )
    if not np.array_equal(arrays["thermal_f"], thermal_f):
        raise C0SummaryError("C0 thermal_f is not the exact author distribution.")

    history = arrays["state_history"]
    deltas = arrays["newton_deltas"]
    relative_steps = arrays["relative_qp_steps"]
    linear_errors = arrays["linear_solve_backward_errors"]
    if (
        history.ndim != 2
        or deltas.ndim != 2
        or history.shape[1] != 2 * size - 1
        or deltas.shape != (history.shape[0] - 1, 2 * size - 1)
        or relative_steps.shape != (deltas.shape[0],)
        or linear_errors.shape != (deltas.shape[0],)
        or deltas.shape[0] < 1
    ):
        raise C0SummaryError("C0 Newton history arrays have incompatible shapes.")
    if not np.array_equal(history[0], initial_state):
        raise C0SummaryError("C0 Newton history does not start from A1.")
    if not np.array_equal(history[1:], history[:-1] + deltas):
        raise C0SummaryError("C0 Newton history is not exactly before + delta.")
    if not np.array_equal(history[-1, :size], arrays["f_final"]):
        raise C0SummaryError("C0 f_final is not the Newton-history endpoint.")
    if not np.array_equal(history[-1, size:], arrays["n_phonon_final"]):
        raise C0SummaryError("C0 n_phonon_final is not the Newton-history endpoint.")

    maximum_delta_error = 0.0
    maximum_linear_error_difference = 0.0
    recomputed_relative_steps: list[float] = []
    recomputed_linear_errors: list[float] = []
    for index, before in enumerate(history[:-1]):
        evaluation = evaluate_author_system(
            operator,
            before[:size],
            before[size:],
            build_update_matrix=True,
        )
        matrix = evaluation.update_matrix_s_inv
        if matrix is None:  # pragma: no cover - guaranteed by the call.
            raise AssertionError("C0 verifier did not receive a Newton matrix.")
        expected_delta = author_newton_delta(evaluation)
        delta_error = _symmetric_relative(expected_delta, deltas[index])
        maximum_delta_error = max(maximum_delta_error, delta_error)
        denominator = float(np.linalg.norm(matrix, ord=np.inf)) * float(
            np.linalg.norm(deltas[index], ord=np.inf)
        ) + float(np.linalg.norm(evaluation.residual_s_inv, ord=np.inf))
        linear_error = float(
            np.linalg.norm(
                matrix @ deltas[index] + evaluation.residual_s_inv,
                ord=np.inf,
            )
        )
        linear_error = linear_error / denominator if denominator > 0.0 else 0.0
        recomputed_linear_errors.append(linear_error)
        maximum_linear_error_difference = max(
            maximum_linear_error_difference,
            abs(linear_error - float(linear_errors[index])),
        )
        after = history[index + 1]
        recomputed_relative_steps.append(
            float(np.linalg.norm(deltas[index, :size])) / float(np.linalg.norm(after[:size]))
        )
    recomputed_relative = np.asarray(recomputed_relative_steps)
    recomputed_linear = np.asarray(recomputed_linear_errors)
    if not np.allclose(relative_steps, recomputed_relative, rtol=0.0, atol=1e-15):
        raise C0SummaryError("C0 recorded relative steps are not reproducible.")
    if not np.allclose(linear_errors, recomputed_linear, rtol=0.0, atol=1e-15):
        raise C0SummaryError("C0 linear backward errors are not reproducible.")
    if maximum_delta_error > _ACCEPTANCE_LIMITS["newton_delta_symmetric_relative_max"]:
        raise C0SummaryError("C0 retained Newton deltas are not the author updates.")
    threshold = parameters.relative_step_threshold
    if np.any(relative_steps[:-1] < threshold) or not relative_steps[-1] < threshold:
        raise C0SummaryError("C0 stopping history does not implement the author rule.")

    final_evaluation = evaluate_author_system(
        operator,
        arrays["f_final"],
        arrays["n_phonon_final"],
        build_update_matrix=False,
    )
    expected_balances = {
        "qp_photon": final_evaluation.qp_photon,
        "qp_scattering": final_evaluation.qp_scattering,
        "qp_pair": final_evaluation.qp_pair,
        "phonon_scattering": final_evaluation.phonon_scattering,
        "phonon_pair": final_evaluation.phonon_pair,
        "phonon_escape": final_evaluation.phonon_escape,
    }
    for prefix, balance in expected_balances.items():
        for side, expected in (
            ("gain", balance.gain_s_inv),
            ("loss", balance.loss_s_inv),
        ):
            name = f"{prefix}__{side}_s_inv"
            if not np.array_equal(arrays[name], expected):
                raise C0SummaryError(f"C0 {name} is not the recomputed channel.")
            if np.any(arrays[name] < 0.0):
                raise C0SummaryError(f"C0 {name} contains a negative term.")
    if not np.array_equal(
        arrays["final_residual_s_inv"],
        final_evaluation.residual_s_inv,
    ):
        raise C0SummaryError("C0 final residual is not the channel sum.")

    driven_integral = author_direct_gap_integral(
        arrays["f_final"],
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    thermal_integral = author_direct_gap_integral(
        thermal_f,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    driven_gap = author_direct_gap(
        arrays["f_final"],
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
        delta0_eV=parameters.delta0_eV,
    )
    thermal_gap = author_direct_gap(
        thermal_f,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
        delta0_eV=parameters.delta0_eV,
    )
    ordinate = (driven_gap - thermal_gap) / (parameters.gap_eV - thermal_gap)
    observable = {
        "driven_gap_eV": driven_gap,
        "driven_integral": driven_integral,
        "figure6_ordinate": ordinate,
        "thermal_gap_eV": thermal_gap,
        "thermal_integral": thermal_integral,
    }
    _compare_exact_mapping(metadata.get("observable"), observable, "observable")

    qp_certificate = _certificate(
        tuple(
            _balance_from_arrays(arrays, prefix)
            for prefix in ("qp_photon", "qp_scattering", "qp_pair")
        )
    )
    phonon_certificate = _certificate(
        tuple(
            _balance_from_arrays(arrays, prefix)
            for prefix in (
                "phonon_scattering",
                "phonon_pair",
                "phonon_escape",
            )
        )
    )
    bounds = {
        "f_max": float(np.max(arrays["f_final"])),
        "f_min": float(np.min(arrays["f_final"])),
        "n_phonon_max": float(np.max(arrays["n_phonon_final"])),
        "n_phonon_min": float(np.min(arrays["n_phonon_final"])),
    }
    diagnostics = {
        "final_bounds": bounds,
        "phonon_cancellation": phonon_certificate,
        "qp_cancellation": qp_certificate,
        "qp_number": qp_number_diagnostics(final_evaluation, operator),
    }
    _compare_exact_mapping(metadata.get("diagnostics"), diagnostics, "diagnostics")

    state_relative = _symmetric_relative(a1_final, history[-1])
    qp_relative = _symmetric_relative(a1_final[:size], arrays["f_final"])
    phonon_relative = _symmetric_relative(
        a1_final[size:],
        arrays["n_phonon_final"],
    )
    a1_ordinate = _finite(
        a1.metadata.get("normalized_numerical_observable"),
        "A1 normalized_numerical_observable",
    )
    comparison = {
        "a1_final_state_relative_l2": state_relative,
        "a1_figure6_ordinate": a1_ordinate,
        "a1_phonon_final_relative_l2": phonon_relative,
        "a1_qp_final_relative_l2": qp_relative,
        "figure6_ordinate_signed_difference": ordinate - a1_ordinate,
    }
    # The producer used ||A1|| as the comparison denominator, while the
    # verifier's symmetric metric is deliberately stricter against rescaling.
    raw_comparison = _mapping(metadata.get("comparison"), "comparison")
    for key in (
        "a1_figure6_ordinate",
        "figure6_ordinate_signed_difference",
    ):
        if raw_comparison.get(key) != comparison[key]:
            raise C0SummaryError(f"comparison.{key} is forged.")
    if any(
        abs(_finite(raw_comparison.get(key), f"comparison.{key}") - comparison[key]) > 1e-15
        for key in (
            "a1_final_state_relative_l2",
            "a1_phonon_final_relative_l2",
            "a1_qp_final_relative_l2",
        )
    ):
        raise C0SummaryError("C0 versus A1 state comparison is forged.")

    acceptance_checks = {
        "a1_final_state_relative_l2": (
            state_relative <= _ACCEPTANCE_LIMITS["a1_final_state_relative_l2_max"]
        ),
        "a1_ordinate_absolute_error": (
            abs(ordinate - a1_ordinate) <= _ACCEPTANCE_LIMITS["a1_ordinate_absolute_error_max"]
        ),
        "final_occupations_physical": (
            bounds["f_min"] >= 0.0 and bounds["f_max"] <= 1.0 and bounds["n_phonon_min"] >= 0.0
        ),
        "linear_solve_backward_error": (
            float(np.max(linear_errors)) <= _ACCEPTANCE_LIMITS["linear_solve_backward_error_max"]
        ),
        "newton_transitions": (
            maximum_delta_error <= _ACCEPTANCE_LIMITS["newton_delta_symmetric_relative_max"]
        ),
        "phonon_cancellation": (
            phonon_certificate["residual_over_turnover"]
            <= _ACCEPTANCE_LIMITS["cancellation_residual_over_turnover_max"]
        ),
        "qp_cancellation": (
            qp_certificate["residual_over_turnover"]
            <= _ACCEPTANCE_LIMITS["cancellation_residual_over_turnover_max"]
        ),
        "stopping_rule": True,
    }
    accepted = all(acceptance_checks.values())
    if not accepted:
        failed = sorted(key for key, value in acceptance_checks.items() if not value)
        raise C0SummaryError(f"C0 acceptance checks failed: {failed!r}.")

    return {
        "acceptance": {
            "accepted": accepted,
            "checks": acceptance_checks,
            "limits": dict(_ACCEPTANCE_LIMITS),
        },
        "a1_binding": {
            "anchor": anchor_binding,
            "manifest_sha256": a1.manifest_sha256,
        },
        "array_descriptors": {name: _descriptor(value) for name, value in sorted(arrays.items())},
        "comparison": comparison,
        "limitations": {
            "author_initialization": "not validated; captured A1 initial state reused",
            "continuation": "not validated",
            "scope": "one authenticated point only; no full 300-point replay",
        },
        "observable": observable,
        "parameters": expected_parameters,
        "raw_bundle": {
            "manifest_sha256": raw_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "schema": SUMMARY_SCHEMA,
        "solver_verification": {
            "iterations": int(deltas.shape[0]),
            "last_relative_qp_step": float(relative_steps[-1]),
            "maximum_linear_error_difference": maximum_linear_error_difference,
            "maximum_linear_solve_backward_error": float(np.max(linear_errors)),
            "maximum_newton_delta_symmetric_relative": maximum_delta_error,
            "termination": "relative_qp_step_threshold",
            "verified_every_transition": True,
        },
        "stage": {
            "comparison_stage_id": COMPARISON_STAGE_ID,
            "parent_stage_id": None,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
        "summary_sources": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): source_sha256(path)
            for path in _SUMMARY_SOURCE_PATHS
        },
    }


def canonical_summary_bytes(summary: dict[str, Any]) -> bytes:
    """Serialize one checked C0 summary deterministically."""

    return (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_c0_summary(
    raw_bundle_dir: Path,
    output_path: Path,
    *,
    a1_bundle_dir: Path,
    anchor_path: Path,
) -> Path:
    """Build and exclusively write one checked C0 summary."""

    summary = build_c0_summary(
        raw_bundle_dir,
        a1_bundle_dir=a1_bundle_dir,
        anchor_path=anchor_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_summary_bytes(summary))
    return output_path


def load_c0_summary(path: Path = DEFAULT_SUMMARY) -> dict[str, Any]:
    """Strictly load and repository-bind a checked C0 summary."""

    raw = _read_regular_file_once(path, "checked C0 summary")
    summary = _parse_json(raw, "checked C0 summary")
    _exact_keys(summary, _SUMMARY_KEYS, "checked C0 summary")
    if summary.get("schema") != SUMMARY_SCHEMA:
        raise C0SummaryError("Checked C0 summary schema is unsupported.")
    acceptance = _mapping(summary.get("acceptance"), "acceptance")
    if acceptance.get("accepted") is not True:
        raise C0SummaryError("Checked C0 summary is not accepted.")
    sources = _mapping(summary.get("summary_sources"), "summary_sources")
    expected_paths = {
        path.relative_to(REPOSITORY_ROOT).as_posix() for path in _SUMMARY_SOURCE_PATHS
    }
    if set(sources) != expected_paths:
        raise C0SummaryError("Checked C0 summary source closure is incomplete.")
    for relative, digest in sources.items():
        if source_sha256(REPOSITORY_ROOT / relative) != _sha256(
            digest,
            f"summary_sources.{relative}",
        ):
            raise C0SummaryError("Checked C0 summary source binding is stale.")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-bundle", type=Path, required=True)
    parser.add_argument("--a1-bundle", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c0_summary(
            args.c0_bundle,
            args.output,
            a1_bundle_dir=args.a1_bundle,
            anchor_path=args.anchor,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
