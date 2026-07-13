"""Pair-breaking (ω_PB > 2Δ) photon collision integral.

Three-term integral per [F24] Eqs. 2–5:

1. **Scattering** (``K⁺``, number-conserving): partners at ``i ± m``,
   same index-shift structure as the sub-gap case.
2. **Generation** (``K⁻``, pair creation): reflection partner at
   ``E_j = ω_PB − E_i``. Creates a QP pair from the photon.
3. **Recombination** (``K⁻``, pair annihilation): same reflection
   partner. Annihilates a QP pair, emitting a photon.

Ported from ``pair_breaking_photon_collision_rates`` in the old
``qpsim/numerics/collision_phonon.py``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.physics.spectral import SpectralContext

_COMMENSURATE_TOL = 0.01


def pair_breaking_photon_collision_rates(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_PB: float,
    n_bar_PB: float,
    c_phot_PB: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair-breaking photon ``(gain, loss_rate)`` at one pixel.

    Combines scattering (``K⁺``, ``i ± m``), pair generation (``K⁻``,
    reflection partner), and pair recombination (``K⁻``, same partner).
    Requires ``ω_PB ≥ 2Δ`` for pair-breaking to contribute; partners
    with ``E_j < Δ`` are skipped silently.

    Returns
    -------
    gain, loss_rate
        Each of shape ``(NE,)``, with ``df/dt = gain − loss_rate · f``.
    """
    E = ctx.E
    NE = E.size
    gap = ctx.gap
    rho = ctx.rho
    K_plus = ctx.K_plus
    K_minus = ctx.K_minus
    dE_scalar = uniform_grid_spacing(E, ctx.dE, "Pair-breaking photon collision")

    if not np.isfinite(omega_PB) or omega_PB < 0.0:
        raise ValueError(
            f"omega_PB must be finite and non-negative; got {omega_PB}."
        )
    if not np.isfinite(n_bar_PB) or n_bar_PB < 0.0:
        raise ValueError(
            f"n_bar_PB must be finite and non-negative; got {n_bar_PB}."
        )
    if not np.isfinite(c_phot_PB) or c_phot_PB < 0.0:
        raise ValueError(
            f"c_phot_PB must be finite and non-negative; got {c_phot_PB}."
        )

    m = round(omega_PB / dE_scalar)
    if omega_PB == 0.0:
        return np.zeros(NE), np.zeros(NE)
    if m <= 0:
        raise ValueError(
            f"omega_PB={omega_PB:.6g} μeV is below half the grid spacing "
            f"dE={dE_scalar:.6g} μeV and cannot be represented; refine the "
            "energy grid or set omega_PB=0 explicitly to disable the channel."
        )

    frac_err = abs(omega_PB - m * dE_scalar) / dE_scalar
    if frac_err > _COMMENSURATE_TOL:
        warnings.warn(
            f"PB photon frequency omega_PB={omega_PB:.6g} μeV is not "
            f"grid-commensurate (dE={dE_scalar:.6g} μeV, nearest m={m}, "
            f"fractional error={frac_err:.4f} > tol={_COMMENSURATE_TOL}). "
            f"Snapped to m·dE={m * dE_scalar:.6g} μeV.",
            stacklevel=2,
        )

    omega_PB_snapped = m * dE_scalar

    # The K⁻ reflection partner j_r = round((ω − E_i − E[0])/dE) is exact
    # only when ω − 2·E[0] is grid-commensurate (the residual is the same
    # for every i on a uniform grid). A misaligned origin silently places
    # every partner up to dE/2 off and breaks detailed balance at the
    # percent level, so warn just like the ω check above.
    partner_steps = (omega_PB_snapped - 2.0 * E[0]) / dE_scalar
    partner_err = abs(partner_steps - round(partner_steps))
    if partner_err > _COMMENSURATE_TOL:
        warnings.warn(
            f"PB reflection partners are not grid-aligned: omega_PB_snapped"
            f" - 2*E[0] = {omega_PB_snapped - 2.0 * E[0]:.6g} μeV is "
            f"{partner_err:.4f} bins away from the lattice "
            f"(tol={_COMMENSURATE_TOL}). Generation/recombination terms "
            f"will violate detailed balance; shift the grid origin so "
            f"(omega_PB - 2*E_min)/dE is an integer.",
            stacklevel=2,
        )

    gain = np.zeros(NE)
    loss_rate = np.zeros(NE)
    one_minus_f = np.maximum(1.0 - f, 0.0)

    for i in range(NE):
        # --- 1. Scattering (K+, number-conserving) ---
        j_up = i + m
        if j_up < NE:
            U_plus = rho[j_up] * K_plus[i, j_up]
            gain[i] += c_phot_PB * U_plus * f[j_up] * (n_bar_PB + 1)
            loss_rate[i] += c_phot_PB * U_plus * one_minus_f[j_up] * n_bar_PB

        j_down = i - m
        if j_down >= 0 and E[j_down] >= gap:
            U_plus = rho[j_down] * K_plus[i, j_down]
            gain[i] += c_phot_PB * U_plus * f[j_down] * n_bar_PB
            loss_rate[i] += c_phot_PB * U_plus * one_minus_f[j_down] * (n_bar_PB + 1)

        # --- 2 & 3. Generation + Recombination (K-, reflection partner) ---
        E_partner = omega_PB_snapped - E[i]
        if E_partner < gap:
            continue
        j_r = round((E_partner - E[0]) / dE_scalar)
        if not (0 <= j_r < NE):
            continue
        U_minus = rho[j_r] * K_minus[i, j_r]
        # Generation: photon creates a QP pair; this site gains at rate
        # c · U⁻ · n_bar · (1 − f_j).
        gain[i] += c_phot_PB * U_minus * n_bar_PB * one_minus_f[j_r]
        # Recombination: QP pair annihilates, emitting a photon; this
        # site loses at rate c · U⁻ · (1 + n_bar) · f_j.
        loss_rate[i] += c_phot_PB * U_minus * (1 + n_bar_PB) * f[j_r]

    gain_with_pauli = gain * one_minus_f
    return gain_with_pauli, loss_rate
