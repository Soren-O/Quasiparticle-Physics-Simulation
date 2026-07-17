"""Independent balance certificates for Fischer steady-state validations.

This module deliberately rebuilds the collision kernels from a returned state
instead of trusting solver-internal convergence flags.  Its supported scope is
the common Fischer Fig. 3/6/7 contract: an ideal-BCS, homogeneous fixed-gap
state with electron-phonon collisions, one sub-gap photon drive, and either
thermal phonons or a single finite-escape-time Ph0 branch using the Eq. 12
phonon-side kernels.

Keeping the scope narrow is intentional.  A pair-breaking photon drive,
external flux, Dynes broadening, multiple phonon branches, or a frequency-
dependent escape time needs additional balance terms and must not receive a
plausible-looking partial certificate from this helper.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionState
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
from qpsim.phonon_models.ph0_local import phonon_balance_diagnostics
from qpsim.phonon_models.state import PhononModel
from qpsim.physics.kernels import thermal_phonon_occupation

CERTIFICATE_FIELDS: tuple[str, ...] = (
    "qp_residual_inf",
    "qp_backward_error",
    "phonon_residual_inf",
    "phonon_raw_backward_error",
    "phonon_backward_error",
)
"""Stable payload fields returned by :func:`steady_state_certificate`."""

CERTIFICATE_METRIC_VERSION = "qp-l1-v1_ph0-nearest-binary64-v1"
"""Human-readable version for persisted certificate provenance."""

_SUB_GAP_PHOTON_KEYS = frozenset({"omega_0", "n_bar", "c_phot"})


def normwise_backward_error(residual: np.ndarray, scale: np.ndarray) -> float:
    """Return the L1 normwise backward error for one balance block."""
    residual_arr = np.asarray(residual, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    if residual_arr.shape != scale_arr.shape:
        raise ValueError(
            "residual and scale must have the same shape; got "
            f"{residual_arr.shape} and {scale_arr.shape}."
        )
    if np.any(~np.isfinite(residual_arr)) or np.any(~np.isfinite(scale_arr)):
        raise ValueError("residual and scale must contain only finite values.")
    residual_magnitude = np.abs(residual_arr)
    scale_magnitude = np.abs(scale_arr)
    common = float(
        max(
            np.max(residual_magnitude, initial=0.0),
            np.max(scale_magnitude, initial=0.0),
        )
    )
    if common == 0.0:
        return 0.0
    denominator = float(np.sum(scale_magnitude / common))
    numerator = float(np.sum(residual_magnitude / common))
    return numerator / denominator if denominator > 0.0 else float("inf")


def _validated_photon_params(
    photon_params: Mapping[str, float],
    *,
    gap: float,
) -> tuple[float, float, float]:
    keys = frozenset(photon_params)
    if keys != _SUB_GAP_PHOTON_KEYS:
        missing = sorted(_SUB_GAP_PHOTON_KEYS - keys)
        extra = sorted(keys - _SUB_GAP_PHOTON_KEYS)
        raise ValueError(
            "Fischer certificate supports exactly one sub-gap photon channel "
            f"with keys {sorted(_SUB_GAP_PHOTON_KEYS)}; "
            f"missing={missing}, extra={extra}."
        )
    omega_0 = float(photon_params["omega_0"])
    n_bar = float(photon_params["n_bar"])
    c_phot = float(photon_params["c_phot"])
    if not np.all(np.isfinite([omega_0, n_bar, c_phot])):
        raise ValueError("Sub-gap photon parameters must be finite.")
    if omega_0 <= 0.0 or omega_0 >= 2.0 * gap:
        raise ValueError(
            "Fischer certificate requires 0 < omega_0 < 2*gap for the "
            f"sub-gap channel; got omega_0={omega_0}, gap={gap}."
        )
    if n_bar < 0.0 or c_phot < 0.0:
        raise ValueError("n_bar and c_phot must be non-negative.")
    return omega_0, n_bar, c_phot


def steady_state_certificate(
    state: T3DiffusionState,
    *,
    photon_params: Mapping[str, float],
    tau_l: float | None,
) -> dict[str, float]:
    """Reassemble raw residual norms and normwise backward errors.

    Parameters are explicit so no figure-module globals can silently alter the
    certified equation. ``tau_l=None`` certifies the thermal-phonon shortcut,
    where only the quasiparticle balance is solved. A finite positive ``tau_l``
    certifies both the quasiparticle and dynamic-phonon equations and requires
    the returned state's single Ph0 branch to carry that same constant value.

    The state's material supplies ``tau_0``, ``T_c``, and ``tau_0_pb_ns``. The
    last quantity is required only for finite ``tau_l``, where the phonon block
    is independently rebuilt with the Fischer Eq. 12 phonon-side kernels.
    """
    spectral = state.spectral
    material = state.material
    if spectral.dynes_gamma != 0.0:
        raise ValueError(
            "Fischer certificate supports ideal BCS only; dynes_gamma must be 0."
        )
    if state.f.shape != spectral.E.shape:
        raise ValueError(
            f"state.f shape {state.f.shape} does not match grid {spectral.E.shape}."
        )
    if np.any(~np.isfinite(state.f)) or np.any((state.f < 0.0) | (state.f > 1.0)):
        raise ValueError("state.f must be finite and lie in [0, 1].")
    if not np.isclose(state.gap, spectral.gap, rtol=0.0, atol=0.0):
        raise ValueError(
            f"state.gap={state.gap} does not match spectral.gap={spectral.gap}."
        )
    if (
        not np.isfinite(material.tau_0)
        or material.tau_0 <= 0.0
        or not np.isfinite(material.T_c)
        or material.T_c <= 0.0
    ):
        raise ValueError("state.material must have finite positive tau_0 and T_c.")

    omega_0, n_bar, c_phot = _validated_photon_params(
        photon_params,
        gap=spectral.gap,
    )
    K_s0 = build_scattering_kernel_base(
        spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )
    K_r0 = build_recombination_kernel_base(
        spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(spectral.E)

    if tau_l is None:
        n_ph = thermal_phonon_occupation(omega, state.T_bath)
    else:
        tau_l = float(tau_l)
        if not np.isfinite(tau_l) or tau_l <= 0.0:
            raise ValueError(f"tau_l must be finite and positive; got {tau_l}.")
        if state.phonon.model is not PhononModel.PH0_LOCAL:
            raise ValueError("Fischer certificate supports only the Ph0-local model.")
        expected_n_shape = (1, omega.size, 1)
        expected_grid_shape = (1, omega.size)
        if state.phonon.n_ph.shape != expected_n_shape:
            raise ValueError(
                "Fischer certificate requires one homogeneous phonon branch; "
                f"got n_ph shape {state.phonon.n_ph.shape}, expected "
                f"{expected_n_shape}."
            )
        if state.phonon.omega_bins.shape != expected_grid_shape or not np.array_equal(
            state.phonon.omega_bins[0], omega
        ):
            raise ValueError(
                "Fischer certificate requires phonons on the collision frequency grid."
            )
        if state.phonon.tau_l.shape != expected_grid_shape or not np.all(
            state.phonon.tau_l == tau_l
        ):
            raise ValueError(
                "Fischer certificate requires a constant state.phonon.tau_l "
                f"equal to the explicit tau_l={tau_l}."
            )
        n_ph = state.phonon.n_ph[0, :, 0]

    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_ph,
        idx_diff,
        idx_sum,
        diff_sign,
    )
    gain, loss = phonon_collision_rates(
        state.f,
        spectral,
        K_s0,
        K_r0,
        state.T_bath,
        N_p_override=N_p,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    photon_gain, photon_loss = sub_gap_photon_collision_rates(
        state.f,
        spectral,
        omega_0,
        n_bar,
        c_phot,
    )
    gain = gain + photon_gain
    loss = loss + photon_loss
    loss_term = loss * state.f
    residual_qp = gain - loss_term
    active = spectral.active_mask
    qp_scale = np.abs(gain[active]) + np.abs(loss_term[active])
    certificate = {
        "qp_residual_inf": float(np.max(np.abs(residual_qp[active]))),
        "qp_backward_error": normwise_backward_error(
            residual_qp[active],
            qp_scale,
        ),
        "phonon_residual_inf": float("nan"),
        "phonon_raw_backward_error": float("nan"),
        "phonon_backward_error": float("nan"),
    }

    if tau_l is None:
        return certificate

    tau_0_pb_ns = material.tau_0_pb_ns
    if (
        tau_0_pb_ns is None
        or not np.isfinite(tau_0_pb_ns)
        or tau_0_pb_ns <= 0.0
    ):
        raise ValueError(
            "Finite-tau_l Fischer certification requires finite positive "
            "state.material.tau_0_pb_ns for the Eq. 12 phonon-side kernels."
        )
    K_s0_phonon_side = build_scattering_kernel_phonon_side(
        spectral,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    K_r0_phonon_side = build_recombination_kernel_phonon_side(
        spectral,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    a_ph, b_ph = compute_phonon_source_sink(
        state.f,
        spectral,
        K_s0,
        K_r0,
        idx_diff,
        idx_sum,
        diff_sign,
        omega.size,
        K_s0_phonon_side=K_s0_phonon_side,
        K_r0_phonon_side=K_r0_phonon_side,
    )
    phonon_diagnostics = phonon_balance_diagnostics(
        n_ph,
        a_ph,
        b_ph,
        thermal_phonon_occupation(omega, state.T_bath),
        tau_l=tau_l,
    )
    certificate["phonon_residual_inf"] = float(
        np.max(np.abs(phonon_diagnostics.residual))
    )
    certificate["phonon_raw_backward_error"] = (
        phonon_diagnostics.raw_backward_error
    )
    certificate["phonon_backward_error"] = (
        phonon_diagnostics.certified_backward_error
    )
    return certificate
