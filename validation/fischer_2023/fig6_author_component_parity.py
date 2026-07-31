"""Frozen-state Figure 6 author/qpsim collision-component comparison.

This diagnostic is deliberately upstream of any nonlinear solve.  It takes a
separately anchored author-point bundle, holds the captured author
``(E, f, n_ph)`` arrays and scalar parameters fixed, and evaluates qpsim's
collision operators on the corresponding finite-volume carrier cells.

The author discretization is hybrid: occupations and coherence factors are
labelled at left edges ``E_i = Delta + i*h``, while its density of states is
averaged over ``[E_i, E_i+h]``.  Qpsim's native finite-volume representation
therefore uses carrier centers ``E_i+h/2``.  This module records that mapping
instead of pretending the two discretizations are byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

from validation.fischer_2023.fig6_author_frozen_state import (
    FrozenStateDiagnosticError,
    load_author_point_bundle,
    validate_bundle_anchor,
)
from validation.reference_models.fischer_2023.fig6_author_operators import (
    AuthorNumericalConstants,
    evaluate_frozen_author_operators,
)
from validation.reproduction_ladder import describe_array
from validation.source_provenance import source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_OPERATOR_SOURCES = (
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig6_author_operators.py",
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig6_author_c0.py",
)
SCHEMA = "qpsim.fischer2023.fig6-author-frozen-components.v2"
_AUTHOR_SECONDS_PER_NS = 1.0e-9
_REQUIRED_AUTHOR_COMPONENT_ARRAYS = {
    "phonon_residual_author_s_inv",
    "qp_phonon_residual_author_s_inv",
    "qp_photon_residual_author_s_inv",
}


@dataclass(frozen=True)
class FrozenComponentResult:
    """Comparison metadata and exact native arrays."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _finite_vector(
    arrays: dict[str, np.ndarray],
    name: str,
    size: int,
) -> np.ndarray:
    if name not in arrays:
        raise FrozenStateDiagnosticError(
            f"Author bundle lacks required component array {name!r}."
        )
    value = np.asarray(arrays[name])
    if value.shape != (size,) or not np.issubdtype(value.dtype, np.floating):
        raise FrozenStateDiagnosticError(
            f"Author component {name!r} must be a floating vector of length {size}."
        )
    if np.any(~np.isfinite(value)):
        raise FrozenStateDiagnosticError(
            f"Author component {name!r} contains non-finite values."
        )
    return value


def _finite_positive(mapping: dict[str, Any], key: str) -> float:
    raw = mapping.get(key)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise FrozenStateDiagnosticError(f"{key!r} must be a finite positive scalar.")
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise FrozenStateDiagnosticError(f"{key!r} must be a finite positive scalar.")
    return value


def _thermal_occupation_with_constants(
    omega_uev: np.ndarray,
    temperature_K: float,
    *,
    boltzmann_constant_J_per_K: float,
    electron_charge_C: float,
) -> np.ndarray:
    """Bose occupation using an explicitly selected numerical constant set."""

    omega = np.asarray(omega_uev, dtype=float)
    if omega.ndim != 1 or np.any(~np.isfinite(omega)) or np.any(omega < 0.0):
        raise FrozenStateDiagnosticError(
            "Explicit-constant Bose frequencies must be finite and non-negative."
        )
    if (
        not math.isfinite(temperature_K)
        or temperature_K <= 0.0
        or not math.isfinite(boltzmann_constant_J_per_K)
        or boltzmann_constant_J_per_K <= 0.0
        or not math.isfinite(electron_charge_C)
        or electron_charge_C <= 0.0
    ):
        raise FrozenStateDiagnosticError(
            "Explicit-constant Bose parameters must be finite and positive."
        )
    k_b_uev_per_K = (
        boltzmann_constant_J_per_K / electron_charge_C * 1.0e6
    )
    result = np.zeros_like(omega)
    positive = omega > 0.0
    exponent = omega[positive] / (k_b_uev_per_K * temperature_K)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        result[positive] = 1.0 / np.expm1(exponent)
    return result


def _map_frequency_matrix_to_lattice(
    frequencies_uev: np.ndarray,
    lattice_uev: np.ndarray,
) -> np.ndarray:
    """Map a frequency matrix onto one sorted lattice without silent snaps."""

    target = np.asarray(frequencies_uev, dtype=float)
    lattice = np.asarray(lattice_uev, dtype=float)
    if (
        target.ndim != 2
        or lattice.ndim != 1
        or lattice.size == 0
        or np.any(~np.isfinite(target))
        or np.any(~np.isfinite(lattice))
        or np.any(np.diff(lattice) <= 0.0)
    ):
        raise FrozenStateDiagnosticError("Frequency-map inputs are malformed.")
    right = np.searchsorted(lattice, target, side="left")
    right = np.clip(right, 0, lattice.size - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(target - lattice[left]) <= np.abs(lattice[right] - target)
    mapped = np.where(choose_left, left, right).astype(np.int64)
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(target))),
        float(np.max(np.abs(lattice))),
    )
    if np.any(np.abs(lattice[mapped] - target) > tolerance):
        raise FrozenStateDiagnosticError(
            "A controlled author frequency is absent from qpsim's lattice."
        )
    return mapped


def _phonon_collision_terms(
    f: np.ndarray,
    ctx: SpectralContext,
    *,
    K_scattering: np.ndarray,
    K_pair: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_full: np.ndarray,
    n_thermal: np.ndarray,
    tau_l_ns: float,
    apply_kaplan_correction: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return scattering, pair, escape, and total terms in s^-1."""

    scatter_a, scatter_b = compute_phonon_source_sink(
        f,
        ctx,
        K_scattering,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_full.size,
        enable_recombination=False,
    )
    if apply_kaplan_correction:
        pair_a, pair_b = compute_phonon_source_sink(
            f,
            ctx,
            None,
            K_pair,
            omega_idx_diff,
            omega_idx_sum,
            diff_sign,
            n_full.size,
            enable_scattering=False,
            K_r0_phonon_side=K_pair,
        )
    else:
        pair_a, pair_b = compute_phonon_source_sink(
            f,
            ctx,
            None,
            K_pair,
            omega_idx_diff,
            omega_idx_sum,
            diff_sign,
            n_full.size,
            enable_scattering=False,
        )
    scattering = (scatter_a + scatter_b * n_full) / _AUTHOR_SECONDS_PER_NS
    pair = (pair_a + pair_b * n_full) / _AUTHOR_SECONDS_PER_NS
    escape = ((n_thermal - n_full) / tau_l_ns) / _AUTHOR_SECONDS_PER_NS
    return scattering, pair, escape, scattering + pair + escape


def _embed_author_phonons(
    author_omega_uev: np.ndarray,
    author_n_ph: np.ndarray,
    qpsim_omega_uev: np.ndarray,
    *,
    spacing_uev: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed the author's truncated phonon grid into qpsim's full sum grid."""

    if author_omega_uev.shape != author_n_ph.shape:
        raise FrozenStateDiagnosticError(
            "Author omega_phonon and n_phonon_final shapes differ."
        )
    full = np.zeros_like(qpsim_omega_uev)
    mapped = np.empty(author_omega_uev.size, dtype=np.int64)
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0,
        float(np.max(np.abs(qpsim_omega_uev))),
    )
    for index, omega in enumerate(author_omega_uev):
        candidate = int(np.argmin(np.abs(qpsim_omega_uev - omega)))
        if abs(float(qpsim_omega_uev[candidate] - omega)) > tolerance:
            raise FrozenStateDiagnosticError(
                f"Author phonon frequency {omega!r} is absent from qpsim's "
                "frequency lattice."
            )
        if index and candidate <= mapped[index - 1]:
            raise FrozenStateDiagnosticError(
                "Author phonon frequencies do not map monotonically and uniquely."
            )
        mapped[index] = candidate
        full[candidate] = author_n_ph[index]
    expected = spacing_uev * np.arange(1, author_omega_uev.size + 1)
    if not np.allclose(author_omega_uev, expected, rtol=0.0, atol=tolerance):
        raise FrozenStateDiagnosticError(
            "Author phonon coordinates do not equal h, 2h, ..., (N-1)h."
        )
    return full, mapped


def _comparison_metrics(
    author: np.ndarray,
    qpsim: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    author_array = np.asarray(author, dtype=float)
    qpsim_array = np.asarray(qpsim, dtype=float)
    if author_array.shape != qpsim_array.shape:
        raise FrozenStateDiagnosticError("Compared component shapes differ.")
    difference = qpsim_array - author_array
    scale = float(np.sum(np.abs(author_array)) + np.sum(np.abs(qpsim_array)))
    result = {
        "author_l1": float(np.sum(np.abs(author_array))),
        "author_linf": float(np.max(np.abs(author_array), initial=0.0)),
        "l1_symmetric_relative_error": (
            float(np.sum(np.abs(difference))) / scale if scale > 0.0 else 0.0
        ),
        "max_absolute_error_s_inv": float(
            np.max(np.abs(difference), initial=0.0)
        ),
        "qpsim_l1": float(np.sum(np.abs(qpsim_array))),
        "qpsim_linf": float(np.max(np.abs(qpsim_array), initial=0.0)),
        "rms_error_s_inv": float(np.sqrt(np.mean(difference * difference))),
    }
    if weights is not None:
        weight_array = np.asarray(weights, dtype=float)
        if weight_array.shape != author_array.shape:
            raise FrozenStateDiagnosticError("Comparison weights have wrong shape.")
        result["author_weighted_sum"] = float(np.sum(weight_array * author_array))
        result["qpsim_weighted_sum"] = float(np.sum(weight_array * qpsim_array))
        result["weighted_sum_signed_error"] = (
            result["qpsim_weighted_sum"] - result["author_weighted_sum"]
        )
    return result


def _array_record(value: np.ndarray) -> dict[str, object]:
    descriptor = describe_array(value)
    return {
        "dtype": descriptor.dtype,
        "npy_sha256": descriptor.npy_sha256,
        "shape": list(descriptor.shape),
    }


def build_frozen_component_comparison(
    bundle_dir: Path,
    *,
    anchor_path: Path,
) -> FrozenComponentResult:
    """Evaluate qpsim collision components on one anchored author state."""

    producer_sources = source_manifest(
        Path(__file__),
        extra_validation_modules=AUTHOR_OPERATOR_SOURCES,
    )
    bundle = load_author_point_bundle(bundle_dir)
    anchor = validate_bundle_anchor(bundle, anchor_path)
    arrays = bundle.arrays
    missing = _REQUIRED_AUTHOR_COMPONENT_ARRAYS - set(arrays)
    if missing:
        raise FrozenStateDiagnosticError(
            f"Author bundle lacks component captures {sorted(missing)!r}."
        )

    E_left_eV = np.asarray(arrays["E_qp"])
    f = np.asarray(arrays["f_final"])
    n_author = np.asarray(arrays["n_phonon_final"])
    omega_author_eV = np.asarray(arrays["omega_phonon"])
    if (
        E_left_eV.ndim != 1
        or f.shape != E_left_eV.shape
        or n_author.shape != (E_left_eV.size - 1,)
        or omega_author_eV.shape != n_author.shape
    ):
        raise FrozenStateDiagnosticError("Captured author state shapes are invalid.")
    if any(
        np.any(~np.isfinite(value))
        for value in (E_left_eV, f, n_author, omega_author_eV)
    ):
        raise FrozenStateDiagnosticError("Captured author state is non-finite.")

    metadata = bundle.metadata
    h_eV = _finite_positive(metadata, "h_eV")
    gap_eV = _finite_positive(metadata, "rounded_gap_eV")
    h_uev = h_eV * 1.0e6
    gap_uev = gap_eV * 1.0e6
    E_left_uev = E_left_eV * 1.0e6
    E_centers_uev = E_left_uev + 0.5 * h_uev
    dE_uev = np.full_like(E_centers_uev, h_uev)
    ctx = SpectralContext(
        E_bins=E_centers_uev,
        dE_bins=dE_uev,
        gap=gap_uev,
    )

    material = metadata.get("material_parameters")
    if not isinstance(material, dict):
        raise FrozenStateDiagnosticError("material_parameters must be a mapping.")
    numerical_constants = metadata.get("author_numerical_constants")
    if not isinstance(numerical_constants, dict):
        raise FrozenStateDiagnosticError(
            "author_numerical_constants must be a mapping."
        )
    author_k_B = _finite_positive(
        numerical_constants,
        "boltzmann_constant_J_per_K",
    )
    author_e = _finite_positive(numerical_constants, "electron_charge_C")
    tau_0_s = _finite_positive(material, "tau_0_s")
    tau_0_pb_s = _finite_positive(material, "tau_0_pb_s")
    tau_l_s = _finite_positive(material, "tau_l_s")
    tau_0_ns = tau_0_s / _AUTHOR_SECONDS_PER_NS
    tau_0_pb_ns = tau_0_pb_s / _AUTHOR_SECONDS_PER_NS
    tau_l_ns = tau_l_s / _AUTHOR_SECONDS_PER_NS
    T_c_K = _finite_positive(material, "T_c_K")
    T_bath_K = _finite_positive(metadata["input"], "temperature_K")
    n_bar = _finite_positive(metadata, "computed_n_bar")
    c_phot_ns_inv = (
        _finite_positive(metadata, "c_photon_per_second")
        * _AUTHOR_SECONDS_PER_NS
    )
    omega_drive_uev = _finite_positive(
        metadata,
        "solved_photon_energy_eV",
    ) * 1.0e6
    photon_bin_raw = metadata.get("photon_bin")
    if (
        isinstance(photon_bin_raw, bool)
        or not isinstance(photon_bin_raw, (int, np.integer))
    ):
        raise FrozenStateDiagnosticError("photon_bin must be an integer.")
    photon_bin = int(photon_bin_raw)
    cleanroom_author = evaluate_frozen_author_operators(
        E_left_eV,
        f,
        n_author,
        gap_eV=gap_eV,
        h_eV=h_eV,
        temperature_K=T_bath_K,
        T_c_K=T_c_K,
        tau_0_s=tau_0_s,
        tau_0_pb_s=tau_0_pb_s,
        tau_l_s=tau_l_s,
        photon_bin=photon_bin,
        n_bar=n_bar,
        c_photon_s_inv=_finite_positive(metadata, "c_photon_per_second"),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=author_k_B,
            electron_charge_C=author_e,
        ),
    )

    K_s0 = build_scattering_kernel_base(ctx, tau_0=tau_0_ns, T_c=T_c_K)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=tau_0_ns, T_c=T_c_K)
    qpsim_omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
        ctx.E
    )
    n_full, author_omega_indices = _embed_author_phonons(
        omega_author_eV * 1.0e6,
        n_author,
        qpsim_omega,
        spacing_uev=h_uev,
    )
    author_pair_frequencies = (
        E_centers_uev[:, None] + E_centers_uev[None, :] - h_uev
    )
    idx_sum_author_labels = _map_frequency_matrix_to_lattice(
        author_pair_frequencies,
        qpsim_omega,
    )
    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_full,
        idx_diff,
        idx_sum,
        diff_sign,
    )
    N_p_author_labels, N_emit_author_labels, N_abs_author_labels = (
        phonon_occupation_matrices_from_state(
            n_full,
            idx_diff,
            idx_sum_author_labels,
            diff_sign,
        )
    )
    gain_phonon, loss_phonon = phonon_collision_rates(
        f,
        ctx,
        K_s0,
        K_r0,
        T_bath_K,
        N_p_override=N_p,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    qpsim_qp_phonon_s_inv = (
        gain_phonon - loss_phonon * f
    ) / _AUTHOR_SECONDS_PER_NS
    gain_phonon_author_labels, loss_phonon_author_labels = phonon_collision_rates(
        f,
        ctx,
        K_s0,
        K_r0,
        T_bath_K,
        N_p_override=N_p_author_labels,
        N_emit_override=N_emit_author_labels,
        N_abs_override=N_abs_author_labels,
    )
    qpsim_qp_phonon_author_pair_labels_s_inv = (
        gain_phonon_author_labels - loss_phonon_author_labels * f
    ) / _AUTHOR_SECONDS_PER_NS
    gain_photon, loss_photon = sub_gap_photon_collision_rates(
        f,
        ctx,
        omega_drive_uev,
        n_bar,
        c_phot_ns_inv,
    )
    qpsim_qp_photon_s_inv = (
        gain_photon - loss_photon * f
    ) / _AUTHOR_SECONDS_PER_NS
    qpsim_qp_total_s_inv = qpsim_qp_phonon_s_inv + qpsim_qp_photon_s_inv

    K_s_ph = build_scattering_kernel_phonon_side(
        ctx,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    K_r_ph = build_recombination_kernel_phonon_side(
        ctx,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    coherence_left = gap_uev**2 / (
        E_left_uev[:, None] * E_left_uev[None, :]
    )
    K_s_ph_left = (
        2.0 / (np.pi * gap_uev * tau_0_pb_ns)
    ) * np.maximum(1.0 - coherence_left, 0.0)
    K_r_ph_left = (
        1.0 / (np.pi * gap_uev * tau_0_pb_ns)
    ) * (1.0 + coherence_left)
    n_thermal_qpsim = thermal_phonon_occupation(qpsim_omega, T_bath_K)
    n_thermal_author = _thermal_occupation_with_constants(
        qpsim_omega,
        T_bath_K,
        boltzmann_constant_J_per_K=author_k_B,
        electron_charge_C=author_e,
    )

    (
        phonon_left_scattering,
        phonon_left_pair_author_labels,
        phonon_escape_author_constants,
        phonon_author_equivalent,
    ) = _phonon_collision_terms(
        f,
        ctx,
        K_scattering=K_s_ph_left,
        K_pair=K_r_ph_left,
        omega_idx_diff=idx_diff,
        omega_idx_sum=idx_sum_author_labels,
        diff_sign=diff_sign,
        n_full=n_full,
        n_thermal=n_thermal_author,
        tau_l_ns=tau_l_ns,
        apply_kaplan_correction=False,
    )
    phonon_escape_qpsim_constants = (
        (n_thermal_qpsim - n_full) / tau_l_ns
    ) / _AUTHOR_SECONDS_PER_NS
    (
        phonon_fv_scattering,
        phonon_fv_pair_author_labels,
        _,
        _,
    ) = _phonon_collision_terms(
        f,
        ctx,
        K_scattering=K_s_ph,
        K_pair=K_r_ph,
        omega_idx_diff=idx_diff,
        omega_idx_sum=idx_sum_author_labels,
        diff_sign=diff_sign,
        n_full=n_full,
        n_thermal=n_thermal_author,
        tau_l_ns=tau_l_ns,
        apply_kaplan_correction=False,
    )
    (
        _,
        phonon_left_pair_native_labels,
        _,
        _,
    ) = _phonon_collision_terms(
        f,
        ctx,
        K_scattering=K_s_ph_left,
        K_pair=K_r_ph_left,
        omega_idx_diff=idx_diff,
        omega_idx_sum=idx_sum,
        diff_sign=diff_sign,
        n_full=n_full,
        n_thermal=n_thermal_author,
        tau_l_ns=tau_l_ns,
        apply_kaplan_correction=False,
    )
    (
        _,
        phonon_fv_pair_native_no_kaplan,
        _,
        _,
    ) = _phonon_collision_terms(
        f,
        ctx,
        K_scattering=K_s_ph,
        K_pair=K_r_ph,
        omega_idx_diff=idx_diff,
        omega_idx_sum=idx_sum,
        diff_sign=diff_sign,
        n_full=n_full,
        n_thermal=n_thermal_author,
        tau_l_ns=tau_l_ns,
        apply_kaplan_correction=False,
    )
    (
        _,
        phonon_fv_pair_native_kaplan,
        _,
        _,
    ) = _phonon_collision_terms(
        f,
        ctx,
        K_scattering=K_s_ph,
        K_pair=K_r_ph,
        omega_idx_diff=idx_diff,
        omega_idx_sum=idx_sum,
        diff_sign=diff_sign,
        n_full=n_full,
        n_thermal=n_thermal_author,
        tau_l_ns=tau_l_ns,
        apply_kaplan_correction=True,
    )

    phonon_modern_constants_only = (
        phonon_left_scattering
        + phonon_left_pair_author_labels
        + phonon_escape_qpsim_constants
    )
    phonon_fv_coherence_only = (
        phonon_fv_scattering
        + phonon_fv_pair_author_labels
        + phonon_escape_author_constants
    )
    phonon_center_pair_labels_only = (
        phonon_left_scattering
        + phonon_left_pair_native_labels
        + phonon_escape_author_constants
    )
    phonon_native_grid_author_constants_no_kaplan = (
        phonon_fv_scattering
        + phonon_fv_pair_native_no_kaplan
        + phonon_escape_author_constants
    )
    phonon_native_grid_author_constants_with_kaplan = (
        phonon_fv_scattering
        + phonon_fv_pair_native_kaplan
        + phonon_escape_author_constants
    )
    qpsim_phonon_full_s_inv = (
        phonon_fv_scattering
        + phonon_fv_pair_native_kaplan
        + phonon_escape_qpsim_constants
    )
    qpsim_phonon_author_grid_s_inv = qpsim_phonon_full_s_inv[
        author_omega_indices
    ]

    author_qp_photon = _finite_vector(
        arrays,
        "qp_photon_residual_author_s_inv",
        E_left_eV.size,
    )
    author_qp_phonon = _finite_vector(
        arrays,
        "qp_phonon_residual_author_s_inv",
        E_left_eV.size,
    )
    author_phonon = _finite_vector(
        arrays,
        "phonon_residual_author_s_inv",
        E_left_eV.size - 1,
    )
    author_qp_total = author_qp_photon + author_qp_phonon
    qpsim_qp_total_author_pair_labels_s_inv = (
        qpsim_qp_phonon_author_pair_labels_s_inv + qpsim_qp_photon_s_inv
    )
    author_grid = author_omega_indices

    result_arrays = {
        "author_phonon_residual_s_inv": author_phonon,
        "author_qp_phonon_residual_s_inv": author_qp_phonon,
        "author_qp_photon_residual_s_inv": author_qp_photon,
        "author_qp_total_residual_s_inv": author_qp_total,
        "author_to_qpsim_omega_indices": author_omega_indices,
        "cleanroom_author_phonon_escape_s_inv": (
            cleanroom_author.phonon_escape_s_inv
        ),
        "cleanroom_author_phonon_pair_s_inv": (
            cleanroom_author.phonon_pair_s_inv
        ),
        "cleanroom_author_phonon_scattering_s_inv": (
            cleanroom_author.phonon_scattering_s_inv
        ),
        "cleanroom_author_phonon_total_s_inv": (
            cleanroom_author.phonon_total_s_inv
        ),
        "cleanroom_author_qp_pair_s_inv": cleanroom_author.qp_pair_s_inv,
        "cleanroom_author_qp_phonon_s_inv": cleanroom_author.qp_phonon_s_inv,
        "cleanroom_author_qp_photon_s_inv": cleanroom_author.qp_photon_s_inv,
        "cleanroom_author_qp_scattering_s_inv": (
            cleanroom_author.qp_scattering_s_inv
        ),
        "cleanroom_author_qp_total_s_inv": cleanroom_author.qp_total_s_inv,
        "phonon_author_equivalent_residual_s_inv": (
            phonon_author_equivalent[author_grid]
        ),
        "phonon_center_pair_labels_only_residual_s_inv": (
            phonon_center_pair_labels_only[author_grid]
        ),
        "phonon_escape_author_constants_s_inv": (
            phonon_escape_author_constants[author_grid]
        ),
        "phonon_escape_qpsim_constants_s_inv": (
            phonon_escape_qpsim_constants[author_grid]
        ),
        "phonon_finite_volume_coherence_only_residual_s_inv": (
            phonon_fv_coherence_only[author_grid]
        ),
        "phonon_fv_pair_author_labels_s_inv": (
            phonon_fv_pair_author_labels[author_grid]
        ),
        "phonon_fv_pair_native_labels_kaplan_s_inv": (
            phonon_fv_pair_native_kaplan[author_grid]
        ),
        "phonon_fv_pair_native_labels_no_kaplan_s_inv": (
            phonon_fv_pair_native_no_kaplan[author_grid]
        ),
        "phonon_fv_scattering_s_inv": phonon_fv_scattering[author_grid],
        "phonon_left_pair_author_labels_s_inv": (
            phonon_left_pair_author_labels[author_grid]
        ),
        "phonon_left_pair_native_labels_s_inv": (
            phonon_left_pair_native_labels[author_grid]
        ),
        "phonon_left_scattering_s_inv": phonon_left_scattering[author_grid],
        "phonon_modern_constants_only_residual_s_inv": (
            phonon_modern_constants_only[author_grid]
        ),
        "phonon_native_grid_author_constants_no_kaplan_residual_s_inv": (
            phonon_native_grid_author_constants_no_kaplan[author_grid]
        ),
        "phonon_native_grid_author_constants_with_kaplan_residual_s_inv": (
            phonon_native_grid_author_constants_with_kaplan[author_grid]
        ),
        "qpsim_phonon_residual_on_author_grid_s_inv": (
            qpsim_phonon_author_grid_s_inv
        ),
        "qpsim_qp_phonon_author_pair_labels_s_inv": (
            qpsim_qp_phonon_author_pair_labels_s_inv
        ),
        "qpsim_qp_phonon_residual_s_inv": qpsim_qp_phonon_s_inv,
        "qpsim_qp_photon_residual_s_inv": qpsim_qp_photon_s_inv,
        "qpsim_qp_total_author_pair_labels_s_inv": (
            qpsim_qp_total_author_pair_labels_s_inv
        ),
        "qpsim_qp_total_residual_s_inv": qpsim_qp_total_s_inv,
    }
    comparisons = {
        "phonon_total": _comparison_metrics(
            author_phonon,
            qpsim_phonon_author_grid_s_inv,
        ),
        "qp_phonon": _comparison_metrics(
            author_qp_phonon,
            qpsim_qp_phonon_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_phonon_author_pair_labels": _comparison_metrics(
            author_qp_phonon,
            qpsim_qp_phonon_author_pair_labels_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_photon": _comparison_metrics(
            author_qp_photon,
            qpsim_qp_photon_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_total": _comparison_metrics(
            author_qp_total,
            qpsim_qp_total_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_total_author_pair_labels": _comparison_metrics(
            author_qp_total,
            qpsim_qp_total_author_pair_labels_s_inv,
            weights=ctx.cell_weights,
        ),
    }
    cleanroom_author_comparisons = {
        "phonon_total": _comparison_metrics(
            author_phonon,
            cleanroom_author.phonon_total_s_inv,
        ),
        "qp_phonon": _comparison_metrics(
            author_qp_phonon,
            cleanroom_author.qp_phonon_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_photon": _comparison_metrics(
            author_qp_photon,
            cleanroom_author.qp_photon_s_inv,
            weights=ctx.cell_weights,
        ),
        "qp_total": _comparison_metrics(
            author_qp_total,
            cleanroom_author.qp_total_s_inv,
            weights=ctx.cell_weights,
        ),
    }
    phonon_swap_arrays = {
        "author_equivalent_accumulator": phonon_author_equivalent[author_grid],
        "center_pair_labels_only": phonon_center_pair_labels_only[author_grid],
        "finite_volume_coherence_only": phonon_fv_coherence_only[author_grid],
        "fully_native": qpsim_phonon_author_grid_s_inv,
        "modern_constants_only": phonon_modern_constants_only[author_grid],
        "native_grid_author_constants_no_kaplan": (
            phonon_native_grid_author_constants_no_kaplan[author_grid]
        ),
        "native_grid_author_constants_with_kaplan": (
            phonon_native_grid_author_constants_with_kaplan[author_grid]
        ),
    }
    phonon_channel_turnover_l1 = float(
        np.sum(np.abs(cleanroom_author.phonon_scattering_s_inv))
        + np.sum(np.abs(cleanroom_author.phonon_pair_s_inv))
        + np.sum(np.abs(cleanroom_author.phonon_escape_s_inv))
    )
    phonon_swap_comparisons: dict[str, dict[str, float]] = {}
    for stage_name, stage_residual in phonon_swap_arrays.items():
        record = _comparison_metrics(author_phonon, stage_residual)
        record["candidate_residual_l1_over_cleanroom_channel_turnover"] = (
            record["qpsim_l1"] / phonon_channel_turnover_l1
        )
        record["difference_l1_over_cleanroom_channel_turnover"] = float(
            np.sum(np.abs(stage_residual - author_phonon))
            / phonon_channel_turnover_l1
        )
        phonon_swap_comparisons[stage_name] = record
    qp_channel_turnover_l1 = float(
        np.sum(np.abs(cleanroom_author.qp_photon_s_inv))
        + np.sum(np.abs(cleanroom_author.qp_scattering_s_inv))
        + np.sum(np.abs(cleanroom_author.qp_pair_s_inv))
    )
    cleanroom_cancellation_certificates = {
        "phonon": {
            "captured_residual_l1_s_inv": float(np.sum(np.abs(author_phonon))),
            "channel_turnover_l1_s_inv": phonon_channel_turnover_l1,
            "cleanroom_residual_l1_s_inv": float(
                np.sum(np.abs(cleanroom_author.phonon_total_s_inv))
            ),
            "difference_l1_over_channel_turnover": float(
                np.sum(
                    np.abs(
                        cleanroom_author.phonon_total_s_inv - author_phonon
                    )
                )
                / phonon_channel_turnover_l1
            ),
        },
        "qp": {
            "captured_residual_l1_s_inv": float(np.sum(np.abs(author_qp_total))),
            "channel_turnover_l1_s_inv": qp_channel_turnover_l1,
            "cleanroom_residual_l1_s_inv": float(
                np.sum(np.abs(cleanroom_author.qp_total_s_inv))
            ),
            "difference_l1_over_channel_turnover": float(
                np.sum(
                    np.abs(cleanroom_author.qp_total_s_inv - author_qp_total)
                )
                / qp_channel_turnover_l1
            ),
        },
    }
    ranked = sorted(
        comparisons,
        key=lambda name: comparisons[name]["l1_symmetric_relative_error"],
        reverse=True,
    )
    metadata_result = {
        "array_descriptors": {
            name: _array_record(value)
            for name, value in sorted(result_arrays.items())
        },
        "cleanroom_author_comparisons": cleanroom_author_comparisons,
        "cleanroom_cancellation_certificates": (
            cleanroom_cancellation_certificates
        ),
        "comparisons": comparisons,
        "phonon_controlled_swaps": phonon_swap_comparisons,
        "coordinate_contract": {
            "author_coherence_coordinates": "left edges E_i = Delta + i*h",
            "author_dos_cells": "[E_i, E_i+h]",
            "author_phonon_support": "h through (N-1)h; higher bins implicit zero",
            "qpsim_carrier_centers": "E_i + h/2",
            "qpsim_phonon_embedding": (
                "author n_ph copied at identical omega; all qpsim frequencies "
                "outside the author phonon support set to zero"
            ),
            "qualification": (
                "The fully-native comparison uses qpsim finite-volume "
                "coherence, center-sum frequency labels, current constants, "
                "and the Kaplan endpoint correction. The controlled-swap "
                "records vary those choices separately; no single native "
                "residual is treated as causal evidence."
            ),
        },
        "diagnostic_id": "fischer-catelani-2023-fig6-frozen-components-v2",
        "input_bundle": {
            "anchor": anchor,
            "manifest_sha256": bundle.manifest_sha256,
        },
        "parameters": {
            "T_bath_K": T_bath_K,
            "T_c_K": T_c_K,
            "author_boltzmann_constant_J_per_K": author_k_B,
            "author_electron_charge_C": author_e,
            "author_kB_ueV_per_K": author_k_B / author_e * 1.0e6,
            "c_phot_ns_inv": c_phot_ns_inv,
            "gap_ueV": gap_uev,
            "h_ueV": h_uev,
            "n_bar": n_bar,
            "omega_drive_ueV": omega_drive_uev,
            "tau_0_ns": tau_0_ns,
            "tau_0_pb_ns": tau_0_pb_ns,
            "tau_l_ns": tau_l_ns,
            "qpsim_kB_ueV_per_K": KB_UEV_PER_K,
        },
        "ranked_component_disagreement": ranked,
        "schema": SCHEMA,
        "sources": producer_sources,
    }
    if (
        source_manifest(
            Path(__file__),
            extra_validation_modules=AUTHOR_OPERATOR_SOURCES,
        )
        != producer_sources
    ):
        raise FrozenStateDiagnosticError(
            "Component diagnostic source closure changed during evaluation."
        )
    return FrozenComponentResult(metadata=metadata_result, arrays=result_arrays)


def write_frozen_component_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    anchor_path: Path,
) -> Path:
    """Write exact component arrays and a manifest into a new directory."""

    result = build_frozen_component_comparison(
        bundle_dir,
        anchor_path=anchor_path,
    )
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, object]] = {}
    for name, value in sorted(result.arrays.items()):
        path = root / f"{name}.npy"
        with path.open("xb") as handle:
            np.lib.format.write_array(
                handle,
                np.asarray(value),
                version=(3, 0),
                allow_pickle=False,
            )
        files[path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    manifest = {
        "files": files,
        "metadata": result.metadata,
        "schema": SCHEMA,
    }
    manifest_path = root / "manifest.json"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_frozen_component_bundle(
            args.bundle,
            args.output_dir,
            anchor_path=args.anchor,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
