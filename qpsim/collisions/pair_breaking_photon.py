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

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions._validation import validated_occupation
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
    f = validated_occupation(f, E.shape, "Pair-breaking photon collision")
    # Use the cell-average partner density matched to the finite-volume
    # capacity.  On this required uniform grid, w_i*rho_bar_j is symmetric
    # under i<->j, preserving number for the scattering part and counting
    # pair creation/annihilation consistently for reflection partners.
    rho = ctx.cell_density
    supported = ctx.active_mask
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
    if ctx.dynes_gamma > 0.0:
        raise ValueError(
            "Pair-breaking photon collisions do not support dynes_gamma > 0: "
            "the collision coherence factors are ideal BCS."
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
        raise ValueError(
            f"PB photon frequency omega_PB={omega_PB:.6g} μeV is not "
            f"grid-commensurate (dE={dE_scalar:.6g} μeV, nearest m={m}, "
            f"fractional error={frac_err:.4f} > tol={_COMMENSURATE_TOL}). "
            f"Use m·dE={m * dE_scalar:.6g} μeV or refine the energy grid."
        )

    omega_PB_snapped = m * dE_scalar

    # The K⁻ reflection partner j_r = round((ω − E_i − E[0])/dE) is exact
    # only when ω − 2·E[0] is grid-commensurate (the residual is the same
    # for every i on a uniform grid). A misaligned origin silently places
    # every partner up to dE/2 off and breaks detailed balance at the
    # percent level, so reject just like the ω check above.
    partner_steps = (omega_PB_snapped - 2.0 * E[0]) / dE_scalar
    partner_err = abs(partner_steps - round(partner_steps))
    if omega_PB_snapped >= 2.0 * ctx.gap and partner_err > _COMMENSURATE_TOL:
        raise ValueError(
            f"PB reflection partners are not grid-aligned: omega_PB_snapped"
            f" - 2*E[0] = {omega_PB_snapped - 2.0 * E[0]:.6g} μeV is "
            f"{partner_err:.4f} bins away from the lattice "
            f"(tol={_COMMENSURATE_TOL}). Generation/recombination terms "
            f"would violate detailed balance; shift the grid origin so "
            f"(omega_PB - 2*E_min)/dE is an integer."
        )

    # Every supported cell whose K⁻ reflection partner is physical must be
    # able to represent it. Partners landing ABOVE the grid top are a
    # representability hole, not physics: the affected sites are exactly
    # the singular gap-edge cells (E_i < ω − E_top), and silently skipping
    # them lost ~42% of the total pair-generation rate in a realistic
    # short-grid case while the kernel hard-raised on a 2% partner
    # misalignment (2026-07-19 audit). Fail loud like the alignment guard;
    # sub-gap partners remain a physical exclusion and are still skipped.
    if omega_PB_snapped >= 2.0 * ctx.gap:
        top_edge = E[-1] + 0.5 * dE_scalar
        partner_above_top = supported & (omega_PB_snapped - E > top_edge)
        if np.any(partner_above_top):
            n_dropped = int(np.count_nonzero(partner_above_top))
            raise ValueError(
                f"Pair-breaking photon partners are off-grid: {n_dropped} "
                f"supported gap-edge cells (E <= "
                f"{float(E[partner_above_top].max()):.6g} μeV) have their "
                f"reflection partner ω−E above the grid top "
                f"({float(E[-1]):.6g} μeV) for ω_PB={omega_PB_snapped:.6g} μeV. "
                "Dropping them silently loses the dominant singular-DOS "
                "share of the generation/recombination rate. Extend the "
                f"energy grid to E_max >= ω_PB − Δ = "
                f"{omega_PB_snapped - ctx.gap:.6g} μeV."
            )

    gain = np.zeros(NE)
    loss_rate = np.zeros(NE)
    one_minus_f = np.maximum(1.0 - f, 0.0)

    for i in range(NE):
        if not supported[i]:
            continue

        # --- 1. Scattering (K+, number-conserving) ---
        j_up = i + m
        if j_up < NE:
            U_plus = rho[j_up] * K_plus[i, j_up]
            gain[i] += c_phot_PB * U_plus * f[j_up] * (n_bar_PB + 1)
            loss_rate[i] += c_phot_PB * U_plus * one_minus_f[j_up] * n_bar_PB

        j_down = i - m
        if j_down >= 0 and supported[j_down]:
            U_plus = rho[j_down] * K_plus[i, j_down]
            gain[i] += c_phot_PB * U_plus * f[j_down] * n_bar_PB
            loss_rate[i] += c_phot_PB * U_plus * one_minus_f[j_down] * (n_bar_PB + 1)

        # --- 2 & 3. Generation + Recombination (K-, reflection partner) ---
        E_partner = omega_PB_snapped - E[i]
        j_r = round((E_partner - E[0]) / dE_scalar)
        if not (0 <= j_r < NE) or not supported[j_r]:
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
