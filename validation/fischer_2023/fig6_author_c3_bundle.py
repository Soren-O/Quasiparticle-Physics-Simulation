"""Build immutable frozen-state evidence for the Figure 6 C3 grid stage.

Formal C3 is a child of the completed C2 parameter endpoint.  It changes only
the quasiparticle grid/discretization contract at one authenticated frozen
state.  The live qpsim Figure 6 grid has 1640 cells on faces
``[160, 1800] micro-eV``: twenty zero-capacity guard cells precede the 1620
author cells.  The projection is therefore an ordinal embedding,
``parent i -> child i + 20``.  It is not interpolation.

Four cumulative frozen operators are retained:

``C3p``
    Embed the state in the full domain while retaining exact C2b5 author
    coefficients and pair labels.  Its active rows reproduce C2b5 exactly.
``C3a``
    Replace only left-edge coherence by the live full-grid
    :class:`qpsim.physics.spectral.SpectralContext` finite-volume coherence.
``C3b``
    Additionally replace the author pair label by the one-bin-higher
    center-carrier pair label.
``C3c``
    Additionally replace the retained author-eV DOS arithmetic by the same
    full SpectralContext's native-micro-eV ``cell_density``.

The author phonon support ``h .. (N_author-1)h`` and all C2/C0 occupations
remain immutable.  The full qpsim omega lattice is recorded only to make that
retained support explicit; values outside it are serialization placeholders,
not solved zero occupations.  This module does not run Newton and does not
claim a C3 root, stopping history, plotted ordinate, curve, or paper parity.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.observables.gap_suppression import (
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
    gap_suppression_ratio_from_integrals,
)
from qpsim.physics.bcs_quadrature import cell_edges_from_widths

from validation.fischer_2023 import fig6_solve
from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY as DEFAULT_C0_SUMMARY,
)
from validation.fischer_2023.fig6_author_c0_summary import load_c0_summary
from validation.fischer_2023.fig6_author_c2_bundle import (
    array_descriptor,
    json_value_bit_exact,
    load_c2_raw_bundle,
)
from validation.fischer_2023.fig6_author_c2_parameters import (
    build_c2_parameter_plan,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as DEFAULT_C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as DEFAULT_C2_SCORE,
)
from validation.fischer_2023.fig6_author_c2_score import load_c2_score
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    ChannelBalance,
    SpectralCoefficients,
    SystemEvaluation,
    build_author_coefficients,
    build_author_operator,
    evaluate_author_system,
)
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c3-grid-bundle.v1"
STAGE_ID = "C3"
PARENT_STAGE_ID = "C2"
CHANGED_COMPONENT = "grid_sampling"
FINAL_C2_STEP_ID = "C2b5-finite-cutoff-critical-temperature"
FINAL_C2_SLUG = "c2b5_finite_cutoff_critical_temperature"

STAGE_IDS = (
    "c3p_projected_author_control",
    "c3a_finite_volume_coherence",
    "c3b_center_pair_labels",
    "c3c_native_cell_density",
)

_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "qpsim" / "collisions" / "phonon.py",
    REPOSITORY_ROOT / "qpsim" / "constants.py",
    REPOSITORY_ROOT / "qpsim" / "grid" / "energy_grid.py",
    REPOSITORY_ROOT / "qpsim" / "observables" / "gap_suppression.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "bcs_quadrature.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "gap_equation.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "spectral.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_solve.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c0_summary.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_parameters.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_score.py",
    REPOSITORY_ROOT / "validation" / "reference_models" / "fischer_2023" / "fig6_author_c0.py",
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}


class C3BundleError(ValueError):
    """The C3 parent evidence, projection, or raw transport is invalid."""


@dataclass(frozen=True)
class _StageDefinition:
    stage_id: str
    parent_stage_id: str
    coherence: str
    density: str
    pair_frequency_offset_bins: int
    changed_convention: str


_STAGES = (
    _StageDefinition(
        stage_id=STAGE_IDS[0],
        parent_stage_id=PARENT_STAGE_ID,
        coherence="author_left_edge",
        density="author_cell_average_eV",
        pair_frequency_offset_bins=0,
        changed_convention=("full-domain ordinal embedding only; exact C2b5 active operator"),
    ),
    _StageDefinition(
        stage_id=STAGE_IDS[1],
        parent_stage_id=STAGE_IDS[0],
        coherence="qpsim_finite_volume",
        density="author_cell_average_eV",
        pair_frequency_offset_bins=0,
        changed_convention=(
            "author left-edge K_plus/K_minus -> live full-grid "
            "SpectralContext finite-volume K_plus/K_minus"
        ),
    ),
    _StageDefinition(
        stage_id=STAGE_IDS[2],
        parent_stage_id=STAGE_IDS[1],
        coherence="qpsim_finite_volume",
        density="author_cell_average_eV",
        pair_frequency_offset_bins=1,
        changed_convention=("pair labels 2*Delta+(i+j)h -> 2*Delta+(i+j+1)h"),
    ),
    _StageDefinition(
        stage_id=STAGE_IDS[3],
        parent_stage_id=STAGE_IDS[2],
        coherence="qpsim_finite_volume",
        density="qpsim_cell_density_ueV",
        pair_frequency_offset_bins=1,
        changed_convention=(
            "author-eV cell-average DOS arithmetic -> the same live "
            "SpectralContext native-micro-eV cell_density"
        ),
    ),
)


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C3BundleError(f"C3 numerical source changed during execution: {relative}.")


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    """Capture one safe repository-contained file image before validation."""

    if path.is_symlink() or not path.is_file():
        raise C3BundleError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C3BundleError(f"{label} must stay inside the repository.") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise C3BundleError(f"{label} is missing, unsafe, or a symlink.")
    return resolved, resolved.read_bytes()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C3BundleError(f"{label} must be an object.")
    return value


def _positive_zero_array(shape: tuple[int, ...], *, dtype: np.dtype[Any]) -> np.ndarray:
    result = np.zeros(shape, dtype=dtype)
    if np.any(np.signbit(result)):
        raise AssertionError("np.zeros unexpectedly emitted signed-zero padding.")
    return result


def _author_parameters_from_c0() -> tuple[AuthorSolveParameters, dict[str, Any]]:
    summary = load_c0_summary(DEFAULT_C0_SUMMARY)
    raw = _mapping(summary.get("parameters"), "C0 parameters")
    observable = _mapping(summary.get("observable"), "C0 observable")

    def positive_float(key: str) -> float:
        value = raw.get(key)
        if isinstance(value, bool):
            raise C3BundleError(f"C0 parameter {key!r} is invalid.")
        try:
            result = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as exc:
            raise C3BundleError(f"C0 parameter {key!r} is invalid.") from exc
        if not np.isfinite(result) or result <= 0.0:
            raise C3BundleError(f"C0 parameter {key!r} is invalid.")
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
        raise C3BundleError("C0 integer parameter closure is invalid.")
    thermal_gap_raw = observable.get("thermal_gap_eV")
    if isinstance(thermal_gap_raw, bool):
        raise C3BundleError("C0 thermal gap is invalid.")
    thermal_gap = float(thermal_gap_raw)  # type: ignore[arg-type]
    if not np.isfinite(thermal_gap) or thermal_gap <= 0.0:
        raise C3BundleError("C0 thermal gap is invalid.")
    parameters = AuthorSolveParameters(
        gap_eV=positive_float("gap_eV"),
        h_eV=positive_float("h_eV"),
        temperature_K=positive_float("temperature_K"),
        T_c_K=positive_float("T_c_K"),
        tau_0_s=positive_float("tau_0_s"),
        tau_0_pb_s=positive_float("tau_0_pb_s"),
        tau_l_s=positive_float("tau_l_s"),
        photon_bin=photon_bin,
        n_bar=positive_float("n_bar"),
        c_photon_s_inv=positive_float("c_photon_s_inv"),
        delta0_eV=positive_float("delta0_eV"),
        thermal_gap_eV=thermal_gap,
        max_newton_steps=max_steps,
        relative_step_threshold=positive_float("relative_step_threshold"),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=positive_float("boltzmann_constant_J_per_K"),
            electron_charge_C=positive_float("electron_charge_C"),
        ),
    )
    return parameters, summary


def _parameter_record(parameters: AuthorSolveParameters) -> dict[str, object]:
    values: dict[str, object] = {
        "T_c_K": parameters.T_c_K,
        "boltzmann_constant_J_per_K": (parameters.constants.boltzmann_constant_J_per_K),
        "c_photon_s_inv": parameters.c_photon_s_inv,
        "delta0_eV": parameters.delta0_eV,
        "electron_charge_C": parameters.constants.electron_charge_C,
        "gap_eV": parameters.gap_eV,
        "h_eV": parameters.h_eV,
        "max_newton_steps": parameters.max_newton_steps,
        "n_bar": parameters.n_bar,
        "photon_bin": parameters.photon_bin,
        "relative_step_threshold": parameters.relative_step_threshold,
        "tau_0_pb_s": parameters.tau_0_pb_s,
        "tau_0_s": parameters.tau_0_s,
        "tau_l_s": parameters.tau_l_s,
        "temperature_K": parameters.temperature_K,
        "thermal_gap_eV": parameters.thermal_gap_eV,
    }
    return {
        "hex": {key: value.hex() for key, value in values.items() if isinstance(value, float)},
        "values": values,
    }


def _embedded_qp(value: np.ndarray, active_indices: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(value)
    result = _positive_zero_array((size,), dtype=source.dtype)
    result[active_indices] = source
    return result


def _balances(evaluation: SystemEvaluation) -> dict[str, ChannelBalance]:
    return {
        "qp_photon": evaluation.qp_photon,
        "qp_scattering": evaluation.qp_scattering,
        "qp_pair": evaluation.qp_pair,
        "phonon_scattering": evaluation.phonon_scattering,
        "phonon_pair": evaluation.phonon_pair,
        "phonon_escape": evaluation.phonon_escape,
    }


def _append_stage_arrays(
    arrays: dict[str, np.ndarray],
    *,
    definition: _StageDefinition,
    evaluation: SystemEvaluation,
    active_indices: np.ndarray,
    full_size: int,
) -> dict[str, object]:
    names: dict[str, dict[str, str]] = {}
    for channel, balance in _balances(evaluation).items():
        channel_names: dict[str, str] = {}
        is_qp = channel.startswith("qp_")
        for field, value in (
            ("gain_s_inv", balance.gain_s_inv),
            ("loss_s_inv", balance.loss_s_inv),
            ("net_s_inv", balance.net_s_inv),
        ):
            name = f"{definition.stage_id}__{channel}__{field}"
            arrays[name] = (
                _embedded_qp(value, active_indices, full_size)
                if is_qp
                else np.asarray(value).copy()
            )
            channel_names[field] = name
        names[channel] = channel_names
    active_size = active_indices.size
    qp_residual_name = f"{definition.stage_id}__qp_residual_s_inv"
    phonon_residual_name = f"{definition.stage_id}__phonon_residual_s_inv"
    arrays[qp_residual_name] = _embedded_qp(
        evaluation.residual_s_inv[:active_size],
        active_indices,
        full_size,
    )
    arrays[phonon_residual_name] = np.asarray(evaluation.residual_s_inv[active_size:]).copy()
    return {
        "array_names": {
            "channels": names,
            "phonon_residual_s_inv": phonon_residual_name,
            "qp_residual_s_inv": qp_residual_name,
        },
        "changed_convention": definition.changed_convention,
        "coherence_convention": definition.coherence,
        "density_convention": definition.density,
        "pair_frequency_offset_bins": definition.pair_frequency_offset_bins,
        "parent_stage_id": definition.parent_stage_id,
        "stage_id": definition.stage_id,
    }


def _projection_observable(
    *,
    parent_f: np.ndarray,
    parent_thermal: np.ndarray,
    parent_E_left_eV: np.ndarray,
    projected_f: np.ndarray,
    projected_thermal: np.ndarray,
    native_centers_ueV: np.ndarray,
    native_delta0_ueV: float,
    native_gap_ueV: float,
    parameters: AuthorSolveParameters,
) -> dict[str, object]:
    """Measure both projection identity and the real center-carrier effect.

    The ordinal embedding preserves the author samples exactly only when the
    child vector is deliberately re-read with the parent's left-edge sampling
    semantics.  qpsim instead declares those stored values to live at cell
    centers.  Both diagnostics are retained so the former control cannot hide
    the approximately half-bin carrier shift introduced by C3.
    """

    parent_centers = parent_E_left_eV + 0.5 * parameters.h_eV
    parent_driven_integral = gap_integral_from_distribution_direct(
        parent_f,
        parent_centers,
        gap=parameters.gap_eV,
        samples="authors",
    )
    parent_thermal_integral = gap_integral_from_distribution_direct(
        parent_thermal,
        parent_centers,
        gap=parameters.gap_eV,
        samples="authors",
    )
    reembedded_driven_integral = gap_integral_from_distribution_direct(
        projected_f,
        native_centers_ueV,
        gap=native_gap_ueV,
        samples="authors",
    )
    reembedded_thermal_integral = gap_integral_from_distribution_direct(
        projected_thermal,
        native_centers_ueV,
        gap=native_gap_ueV,
        samples="authors",
    )
    native_driven_integral = gap_integral_from_distribution_direct(
        projected_f,
        native_centers_ueV,
        gap=native_gap_ueV,
        samples="centers",
    )
    native_thermal_integral = gap_integral_from_distribution_direct(
        projected_thermal,
        native_centers_ueV,
        gap=native_gap_ueV,
        samples="centers",
    )
    parent_driven_gap = gap_from_distribution_direct(
        parent_f,
        parent_centers,
        gap=parameters.gap_eV,
        delta0=parameters.delta0_eV,
        samples="authors",
    )
    parent_thermal_gap = gap_from_distribution_direct(
        parent_thermal,
        parent_centers,
        gap=parameters.gap_eV,
        delta0=parameters.delta0_eV,
        samples="authors",
    )
    reembedded_driven_gap_ueV = gap_from_distribution_direct(
        projected_f,
        native_centers_ueV,
        gap=native_gap_ueV,
        delta0=native_delta0_ueV,
        samples="authors",
    )
    reembedded_thermal_gap_ueV = gap_from_distribution_direct(
        projected_thermal,
        native_centers_ueV,
        gap=native_gap_ueV,
        delta0=native_delta0_ueV,
        samples="authors",
    )
    native_driven_gap_ueV = gap_from_distribution_direct(
        projected_f,
        native_centers_ueV,
        gap=native_gap_ueV,
        delta0=native_delta0_ueV,
        samples="centers",
    )
    native_thermal_gap_ueV = gap_from_distribution_direct(
        projected_thermal,
        native_centers_ueV,
        gap=native_gap_ueV,
        delta0=native_delta0_ueV,
        samples="centers",
    )

    def _diagnostic(
        *,
        driven_integral: float,
        thermal_integral: float,
        driven_gap_ueV: float,
        thermal_gap_ueV: float,
        interpretation: str,
    ) -> dict[str, object]:
        return {
            "child_full_grid": {
                "driven_gap_ueV": driven_gap_ueV,
                "driven_integral": driven_integral,
                "frozen_suppression_ratio": gap_suppression_ratio_from_integrals(
                    driven_integral,
                    thermal_integral,
                ),
                "thermal_gap_ueV": thermal_gap_ueV,
                "thermal_integral": thermal_integral,
            },
            "differences_from_parent": {
                "driven_gap_eV_equivalent_signed": (driven_gap_ueV * 1.0e-6 - parent_driven_gap),
                "driven_integral_relative_signed": (driven_integral / parent_driven_integral - 1.0),
                "driven_integral_signed": (driven_integral - parent_driven_integral),
                "thermal_gap_eV_equivalent_signed": (thermal_gap_ueV * 1.0e-6 - parent_thermal_gap),
                "thermal_integral_relative_signed": (
                    thermal_integral / parent_thermal_integral - 1.0
                ),
                "thermal_integral_signed": (thermal_integral - parent_thermal_integral),
            },
            "interpretation": interpretation,
        }

    return {
        "claim": (
            "Two frozen projection diagnostics are reported: exact ordinal "
            "re-embedding under retained author left-edge semantics, and the "
            "actual qpsim center-carrier reinterpretation. Neither is a C3 "
            "root or plotted ordinate."
        ),
        "author_semantics_reembedding": _diagnostic(
            driven_integral=reembedded_driven_integral,
            thermal_integral=reembedded_thermal_integral,
            driven_gap_ueV=reembedded_driven_gap_ueV,
            thermal_gap_ueV=reembedded_thermal_gap_ueV,
            interpretation=(
                "Projected values deliberately re-read as author left-edge "
                "samples; this is the projection-identity control."
            ),
        ),
        "native_center_carrier": _diagnostic(
            driven_integral=native_driven_integral,
            thermal_integral=native_thermal_integral,
            driven_gap_ueV=native_driven_gap_ueV,
            thermal_gap_ueV=native_thermal_gap_ueV,
            interpretation=(
                "Projected values interpreted at their declared qpsim cell "
                "centers; this reports the half-bin carrier effect."
            ),
        ),
        "parent_author_left_edge": {
            "driven_gap_eV": parent_driven_gap,
            "driven_integral": parent_driven_integral,
            "frozen_suppression_ratio": gap_suppression_ratio_from_integrals(
                parent_driven_integral,
                parent_thermal_integral,
            ),
            "thermal_gap_eV": parent_thermal_gap,
            "thermal_integral": parent_thermal_integral,
        },
    }


def build_c3_bundle(
    c2_bundle_dir: Path,
    *,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the formal one-point C2-to-C3 frozen grid differential."""

    _assert_source_snapshots()
    c2_score_path, c2_score_bytes = _repository_file_snapshot(
        c2_score_path,
        "C2 score",
    )
    c2_receipt_path, c2_receipt_bytes = _repository_file_snapshot(
        c2_receipt_path,
        "C2 receipt",
    )
    c2_score = load_c2_score(c2_score_path, receipt_path=c2_receipt_path)
    if (
        c2_score_path.is_symlink()
        or c2_receipt_path.is_symlink()
        or c2_score_path.read_bytes() != c2_score_bytes
        or c2_receipt_path.read_bytes() != c2_receipt_bytes
    ):
        raise C3BundleError("C2 score or receipt changed during C3 validation.")
    c2_metadata, c2_arrays, c2_manifest_sha = load_c2_raw_bundle(c2_bundle_dir)
    c2_raw_binding = _mapping(c2_score.get("raw_bundle"), "C2 raw bundle")
    if c2_raw_binding.get("manifest_sha256") != c2_manifest_sha:
        raise C3BundleError("C3 raw C2 parent does not match the accepted C2 score.")

    parent_parameters, _ = _author_parameters_from_c0()
    parent_record = dict(
        _mapping(load_c0_summary(DEFAULT_C0_SUMMARY).get("parameters"), "C0 parameters")
    )
    parent_record["thermal_gap_eV"] = parent_parameters.thermal_gap_eV
    plan = build_c2_parameter_plan(parent_record)
    effective_steps = dict(plan.c2b_author_effective_steps())
    parameters = effective_steps[FINAL_C2_STEP_ID]
    final_step = _mapping(
        _mapping(c2_metadata, "C2 metadata").get("steps")[-1],  # type: ignore[index]
        "C2 final step",
    )
    if final_step.get("step_id") != FINAL_C2_STEP_ID:
        raise C3BundleError("The accepted C2 raw bundle has no C2b5 endpoint.")
    effective_raw = _mapping(
        final_step.get("effective_author_units"),
        "C2b5 effective parameters",
    )
    expected_effective = _mapping(
        _parameter_record(parameters).get("values"),
        "expected C2b5 parameters",
    )
    for key, expected in expected_effective.items():
        if key in effective_raw and not json_value_bit_exact(
            expected,
            effective_raw[key],
        ):
            raise C3BundleError(f"C2b5 parameter {key!r} is inconsistent.")

    parent_E = np.asarray(c2_arrays["E_left_eV"])
    parent_f = np.asarray(c2_arrays["f_final"])
    parent_thermal = np.asarray(c2_arrays["thermal_f"])
    parent_n = np.asarray(c2_arrays["n_phonon_final"])
    if (
        parent_E.shape != parent_f.shape
        or parent_thermal.shape != parent_f.shape
        or parent_n.shape != (parent_f.size - 1,)
    ):
        raise C3BundleError("C2 frozen-state dimensions are inconsistent.")
    frozen_parent_descriptors = {
        "E_left_eV": array_descriptor(parent_E),
        "f_final": array_descriptor(parent_f),
        "n_phonon_final": array_descriptor(parent_n),
        "thermal_f": array_descriptor(parent_thermal),
    }

    native_E, native_dE, spectral = fig6_solve._build_grid_and_spectral()
    native_E = np.asarray(native_E).copy()
    native_dE = np.asarray(native_dE).copy()
    native_edges = cell_edges_from_widths(native_E, native_dE)
    native_active = np.asarray(spectral.active_mask).copy()
    native_delta0_ueV = float(fig6_solve.DELTA_0)
    native_gap_ueV = float(spectral.gap)
    active_indices = np.flatnonzero(native_active).astype(np.int64)
    expected_native_E = 160.0 + (np.arange(1640, dtype=float) + 0.5)
    if (
        native_E.size != 1640
        or not np.array_equal(native_E, expected_native_E)
        or not np.array_equal(native_dE, np.ones(1640, dtype=float))
        or not np.array_equal(
            native_edges,
            160.0 + np.arange(1641, dtype=float),
        )
        or native_delta0_ueV != 180.0
        or native_gap_ueV != 180.0
        or active_indices.size != parent_E.size
        or not np.array_equal(
            active_indices,
            np.arange(20, native_E.size, dtype=np.int64),
        )
    ):
        raise C3BundleError("The live Figure 6 grid is not the declared 20-guard-cell C3 domain.")
    parent_to_native = active_indices.copy()
    mapped_left_ueV = native_edges[parent_to_native]
    mapped_left_edge_delta_ueV = mapped_left_ueV - parent_E * 1.0e6
    sample_carrier_delta_ueV = native_E[parent_to_native] - parent_E * 1.0e6

    projected_f = _embedded_qp(parent_f, active_indices, native_E.size)
    projected_thermal = _embedded_qp(
        parent_thermal,
        active_indices,
        native_E.size,
    )
    omega_ueV, _, _, _ = build_phonon_frequency_map(native_E)
    omega_ueV = np.asarray(omega_ueV).copy()
    expected_levels = np.arange(1, parent_n.size + 1, dtype=float)
    parent_phonon_to_omega = np.searchsorted(omega_ueV, expected_levels).astype(np.int64)
    if (
        omega_ueV.shape != (3600,)
        or not np.array_equal(omega_ueV, np.arange(3600, dtype=float))
        or np.any(parent_phonon_to_omega >= omega_ueV.size)
        or not np.array_equal(
            omega_ueV[parent_phonon_to_omega],
            expected_levels,
        )
    ):
        raise C3BundleError("The live qpsim omega lattice is not 0..3599 micro-eV.")
    legacy_support = np.zeros(omega_ueV.size, dtype=bool)
    legacy_support[parent_phonon_to_omega] = True
    projected_n = _positive_zero_array(omega_ueV.shape, dtype=parent_n.dtype)
    projected_n[parent_phonon_to_omega] = parent_n

    author_coefficients = build_author_coefficients(parent_E, parameters)
    native_rho_full = np.asarray(spectral.cell_density).copy()
    native_weights_full = np.asarray(spectral.cell_weights).copy()
    native_anomalous_full = np.asarray(spectral.cell_anomalous_density).copy()
    native_K_plus_full = np.asarray(spectral.K_plus).copy()
    native_K_minus_full = np.asarray(spectral.K_minus).copy()
    native_rho_active = native_rho_full[active_indices]
    native_K_plus_active = native_K_plus_full[np.ix_(active_indices, active_indices)]
    native_K_minus_active = native_K_minus_full[np.ix_(active_indices, active_indices)]

    arrays: dict[str, np.ndarray] = {
        "legacy_phonon_support_mask": legacy_support,
        "mapped_left_edge_delta_ueV": mapped_left_edge_delta_ueV,
        "native_K_minus_full": native_K_minus_full,
        "native_K_plus_full": native_K_plus_full,
        "native_active_mask": native_active,
        "native_cell_anomalous_density_full": native_anomalous_full,
        "native_cell_density_full": native_rho_full,
        "native_cell_edges_ueV": native_edges,
        "native_cell_weights_full": native_weights_full,
        "native_dE_ueV": native_dE,
        "native_E_centers_ueV": native_E,
        "native_omega_ueV": omega_ueV,
        "parent_E_left_eV": parent_E.copy(),
        "parent_K_minus_active": np.asarray(author_coefficients.K_minus).copy(),
        "parent_K_plus_active": np.asarray(author_coefficients.K_plus).copy(),
        "parent_f": parent_f.copy(),
        "parent_n_phonon": parent_n.copy(),
        "parent_rho_active": np.asarray(author_coefficients.rho).copy(),
        "parent_thermal_f": parent_thermal.copy(),
        "parent_to_native_index": parent_to_native,
        "parent_phonon_to_native_omega_index": parent_phonon_to_omega,
        "projected_f": projected_f,
        "projected_n_phonon": projected_n,
        "projected_thermal_f": projected_thermal,
        "sample_carrier_delta_ueV": sample_carrier_delta_ueV,
    }

    stage_records: list[dict[str, object]] = []
    for definition in _STAGES:
        if definition.coherence == "author_left_edge":
            K_plus = np.asarray(author_coefficients.K_plus)
            K_minus = np.asarray(author_coefficients.K_minus)
        else:
            K_plus = native_K_plus_active
            K_minus = native_K_minus_active
        rho = (
            np.asarray(author_coefficients.rho)
            if definition.density == "author_cell_average_eV"
            else native_rho_active
        )
        coefficients = SpectralCoefficients(
            rho=rho,
            K_plus=K_plus,
            K_minus=K_minus,
            pair_frequency_offset_bins=definition.pair_frequency_offset_bins,
        )
        operator = build_author_operator(
            parent_E,
            parameters,
            coefficients=coefficients,
        )
        evaluation = evaluate_author_system(
            operator,
            parent_f,
            parent_n,
            build_update_matrix=False,
        )
        stage_record = _append_stage_arrays(
            arrays,
            definition=definition,
            evaluation=evaluation,
            active_indices=active_indices,
            full_size=native_E.size,
        )
        stage_record["operator_scalars"] = {
            "a_delta": operator.a_delta,
            "phonon_prefactor_per_eV_s": operator.phonon_prefactor_per_eV_s,
            "qp_prefactor_s_inv": operator.qp_prefactor_s_inv,
        }
        stage_records.append(stage_record)

    frozen_parent_after = {
        "E_left_eV": array_descriptor(parent_E),
        "f_final": array_descriptor(parent_f),
        "n_phonon_final": array_descriptor(parent_n),
        "thermal_f": array_descriptor(parent_thermal),
    }
    if frozen_parent_after != frozen_parent_descriptors:
        raise C3BundleError("A C3 frozen evaluation mutated its C2 parent arrays.")

    descriptors = {name: array_descriptor(value) for name, value in sorted(arrays.items())}
    projection_observable = _projection_observable(
        parent_f=parent_f,
        parent_thermal=parent_thermal,
        parent_E_left_eV=parent_E,
        projected_f=projected_f,
        projected_thermal=projected_thermal,
        native_centers_ueV=native_E,
        native_delta0_ueV=native_delta0_ueV,
        native_gap_ueV=native_gap_ueV,
        parameters=parameters,
    )
    c2_final_residual_name = f"{FINAL_C2_SLUG}__residual_s_inv"
    if c2_final_residual_name not in c2_arrays:
        raise C3BundleError("C2b5 parent residual is missing.")

    metadata: dict[str, Any] = {
        "array_descriptors": descriptors,
        "coordinate_contract": {
            "active_child_indices": "[20, 1640)",
            "author_phonon_support": "omega/h = 1..1619 only",
            "child_grid": (
                "live qpsim Figure 6 SpectralContext, 1640 centers on faces [160, 1800] micro-eV"
            ),
            "grid_projection": (
                "ordinal identity embedding parent i -> child i+20; no interpolation"
            ),
            "inactive_padding": (
                "canonical positive zero in child QP cells 0..19; these "
                "cells have zero BCS capacity at the fixed 180 micro-eV gap"
            ),
            "native_omega_policy": (
                "full 0..3599 micro-eV lattice recorded; only legacy author "
                "support 1..1619 is evaluated in C3, and other projected "
                "values are non-solved serialization placeholders"
            ),
            "parent_grid": "1620 author left-edge cells [Delta, 10*Delta)",
            "sample_relabeling": (
                "parent cell values are carried unchanged onto qpsim "
                "cell centers; the approximately half-bin sample-carrier "
                "shift is recorded separately from roundoff in the mapped "
                "left cell faces"
            ),
        },
        "frozen_inputs": {
            "descriptors": frozen_parent_descriptors,
            "mutation_check_after_all_stages": True,
            "policy": (
                "C2b5 parent E_left, driven f, thermal f, and author-support "
                "phonon n are immutable; only an explicit ordinal grid "
                "embedding and frozen operator substitutions are evaluated"
            ),
        },
        "limitations": {
            "scope": "one authenticated C2 frozen point only",
            "statement": (
                "No C3 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, or paper-parity claim is "
                "made. C2b5 is deliberately a foreign frozen state. The "
                "existing staged-resolve pilot remains supplemental "
                "author-parameter sensitivity evidence only."
            ),
        },
        "observable_control": projection_observable,
        "parameters": _parameter_record(parameters),
        "native_qpsim_grid_parameters": {
            "delta0_ueV": native_delta0_ueV,
            "delta0_ueV_hex": native_delta0_ueV.hex(),
            "gap_ueV": native_gap_ueV,
            "gap_ueV_hex": native_gap_ueV.hex(),
            "uniform_dE_ueV": float(native_dE[0]),
            "uniform_dE_ueV_hex": float(native_dE[0]).hex(),
        },
        "parent_bindings": {
            "c2_raw_manifest_sha256": c2_manifest_sha,
            "c2_receipt_path": c2_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c2_receipt_sha256": hashlib.sha256(c2_receipt_bytes).hexdigest(),
            "c2_score_path": c2_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c2_score_sha256": hashlib.sha256(c2_score_bytes).hexdigest(),
            "c2b5_parent_residual": array_descriptor(c2_arrays[c2_final_residual_name]),
            "c2b5_step_id": FINAL_C2_STEP_ID,
        },
        "projection": {
            "active_cell_count": int(active_indices.size),
            "guard_cell_count": int(native_E.size - active_indices.size),
            "mapped_left_edge_delta_ueV_max": float(np.max(mapped_left_edge_delta_ueV)),
            "mapped_left_edge_delta_ueV_min": float(np.min(mapped_left_edge_delta_ueV)),
            "mapped_left_edge_nonzero_count": int(np.count_nonzero(mapped_left_edge_delta_ueV)),
            "native_cell_count": int(native_E.size),
            "native_omega_count": int(omega_ueV.size),
            "parent_cell_count": int(parent_E.size),
            "parent_phonon_count": int(parent_n.size),
            "projection_kind": "ordinal_identity_embedding_no_interpolation",
            "sample_carrier_delta_ueV_max": float(np.max(sample_carrier_delta_ueV)),
            "sample_carrier_delta_ueV_min": float(np.min(sample_carrier_delta_ueV)),
            "sample_carrier_nonzero_count": int(np.count_nonzero(sample_carrier_delta_ueV)),
        },
        "runtime": {
            "byteorder": sys.byteorder,
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "C3 producer, accepted C2 loaders/resolver, retained author "
                "operator, live Figure 6 grid/SpectralContext, phonon-map, "
                "and direct-observable sources"
            ),
        },
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
        },
        "stages": stage_records,
    }
    _assert_source_snapshots()
    return metadata, arrays


def write_c3_bundle(
    c2_bundle_dir: Path,
    output_dir: Path,
    *,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    """Write one immutable C3 raw bundle into a new directory."""

    _assert_source_snapshots()
    metadata, arrays = build_c3_bundle(
        c2_bundle_dir,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    root = output_dir.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"C3 output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=root.parent,
            prefix=f".{root.name}.",
            suffix=".tmp",
        )
    )
    try:
        files: dict[str, dict[str, object]] = {}
        for name, value in sorted(arrays.items()):
            content = _npy_bytes(value)
            filename = f"{name}.npy"
            with (temporary_root / filename).open("xb") as handle:
                handle.write(content)
            files[filename] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        manifest = {
            "files": files,
            "metadata": metadata,
            "schema": SCHEMA,
        }
        content = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
        with (temporary_root / "manifest.json").open("xb") as handle:
            handle.write(content)
        _assert_source_snapshots()
        temporary_root.rename(root)
    except BaseException:
        for child in temporary_root.iterdir():
            child.unlink()
        temporary_root.rmdir()
        raise
    return root / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    parser.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c3_bundle(
            args.c2_bundle,
            args.output_dir,
            c2_score_path=args.c2_score,
            c2_receipt_path=args.c2_receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
