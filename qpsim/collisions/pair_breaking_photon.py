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
from dataclasses import dataclass

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions._validation import validated_occupation
from qpsim.physics.spectral import SpectralContext

_COMMENSURATE_TOL = 0.01
# The reflection lattice is grid geometry, not a user-chosen frequency: the
# K⁻ block below solves E_i + E_j = 2·E[0] + round(partner_steps)·dE for
# EVERY i, so any offset is a uniform shift away from ω_PB that no
# re-labelling of the photon energy can absorb — the K⁺ block of the same
# call still runs at exactly m·dE. A thermal pair channel then violates
# detailed balance by expm1(offset·dE/kT) (2026-08-03 review: 6.5% at 50 mK
# for a 0.01-bin offset on a dE = 30 μeV grid, with no diagnostic). Aligned
# grids land within ~1e-12 bins, so this is held at roundoff rather than at
# the 1%-of-a-bin frequency-snap tolerance.
_PARTNER_LATTICE_TOL = 1e-9


def pair_channel_open(omega_uev: float, gap_uev: float) -> bool:
    """True iff the pair channel exists at this photon energy.

    STRICTLY above threshold with a roundoff margin: the K⁻ pair rate is
    zero AT ``ω = 2Δ`` exactly (vanishing pair phase space), and treating
    equality as open let a gap-cut grid emit finite generation at exact
    threshold and a snapped-to-threshold case come out ~7.4× too large
    (2026-07-21 round-5 review). Shared by the rates, the
    threshold-crossing guard, and the analytic Jacobian so all three use
    identical semantics.
    """
    threshold = 2.0 * gap_uev
    tol = 64.0 * float(np.finfo(float).eps) * max(threshold, 1.0)
    return bool(omega_uev > threshold + tol)


@dataclass(frozen=True)
class PairBreakingPhotonGridContract:
    """Validated uniform-grid mapping for one PB photon frequency."""

    index_shift: int
    spacing: float
    snapped_omega: float
    fractional_error: float


def validate_pair_breaking_photon_grid(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
    omega_PB: float,
    *,
    supported: np.ndarray | None = None,
) -> PairBreakingPhotonGridContract:
    """Validate PB snapping and reflected-partner representability.

    This lightweight ``O(NE)`` helper is the shared authority for runtime
    collision evaluation and setup preflight.  It deliberately does not
    construct :class:`~qpsim.physics.spectral.SpectralContext`, whose dense
    coherence matrices are ``O(NE**2)``.  Runtime callers should pass the
    context's exact ``active_mask``; preflight callers may omit it, in which
    case ideal-BCS cell overlap is reconstructed from the uniform grid.
    """
    E_input = np.asarray(E)
    dE_input = np.asarray(dE)
    if np.iscomplexobj(E_input) or np.iscomplexobj(dE_input):
        raise ValueError("E and dE must be real-valued arrays.")
    if E_input.ndim != 1 or dE_input.ndim != 1:
        raise ValueError(
            "E and dE must be 1-D arrays; got shapes "
            f"{E_input.shape} and {dE_input.shape}."
        )
    E_arr = np.asarray(E_input, dtype=float)
    dE_arr = np.asarray(dE_input, dtype=float)
    if not np.isfinite(gap) or gap <= 0.0:
        raise ValueError(f"gap must be finite and positive; got {gap!r}.")
    if not np.isfinite(omega_PB) or omega_PB < 0.0:
        raise ValueError(f"omega_PB must be finite and non-negative; got {omega_PB}.")

    dE_scalar = uniform_grid_spacing(E_arr, dE_arr, "Pair-breaking photon collision")
    m = round(omega_PB / dE_scalar)
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
    if pair_channel_open(omega_PB, gap) != pair_channel_open(omega_PB_snapped, gap):
        raise ValueError(
            f"Snapping omega_PB={omega_PB:.6g} μeV to m·dE="
            f"{omega_PB_snapped:.6g} μeV crosses the 2Δ={2.0 * gap:.6g} "
            "μeV pair threshold and would change whether the pair channel "
            "exists. Supply a grid-commensurate frequency on the intended "
            "side of the threshold."
        )

    partner_steps = (omega_PB_snapped - 2.0 * E_arr[0]) / dE_scalar
    partner_err = abs(partner_steps - round(partner_steps))
    if pair_channel_open(omega_PB_snapped, gap) and partner_err > _PARTNER_LATTICE_TOL:
        raise ValueError(
            f"PB reflection partners are not grid-aligned: omega_PB_snapped"
            f" - 2*E[0] = {omega_PB_snapped - 2.0 * E_arr[0]:.6g} μeV is "
            f"{partner_err:.4g} bins away from the lattice "
            f"(tol={_PARTNER_LATTICE_TOL:g}). Every generation/recombination "
            f"pair would then solve E_i + E_j = "
            f"{2.0 * E_arr[0] + round(partner_steps) * dE_scalar:.6g} μeV "
            f"instead of ω_PB and violate detailed balance; shift the grid "
            f"origin so (omega_PB - 2*E_min)/dE is an integer."
        )

    if supported is None:
        supported_arr = E_arr + 0.5 * dE_scalar > gap
    else:
        supported_input = np.asarray(supported)
        if np.iscomplexobj(supported_input):
            raise ValueError("supported must be a real-valued 1-D mask.")
        if supported_input.ndim != 1:
            raise ValueError(
                "supported must be a 1-D mask; got shape "
                f"{supported_input.shape}."
            )
        supported_arr = np.asarray(supported_input, dtype=bool)
        if supported_arr.shape != E_arr.shape:
            raise ValueError(
                "supported must match the energy-grid shape; got "
                f"{supported_arr.shape} and {E_arr.shape}."
            )

    if pair_channel_open(omega_PB_snapped, gap):
        top_edge = E_arr[-1] + 0.5 * dE_scalar
        partner_above_top = supported_arr & (omega_PB_snapped - E_arr > top_edge)
        if np.any(partner_above_top):
            n_dropped = int(np.count_nonzero(partner_above_top))
            raise ValueError(
                f"Pair-breaking photon partners are off-grid: {n_dropped} "
                f"supported gap-edge cells (E <= "
                f"{float(E_arr[partner_above_top].max()):.6g} μeV) have their "
                f"reflection partner ω−E above the grid top "
                f"({float(E_arr[-1]):.6g} μeV) for ω_PB={omega_PB_snapped:.6g} μeV. "
                "Dropping them silently loses the dominant singular-DOS "
                "share of the generation/recombination rate. Extend the "
                f"energy grid to E_max >= ω_PB − Δ = "
                f"{omega_PB_snapped - gap:.6g} μeV."
            )

        bottom_edge = E_arr[0] - 0.5 * dE_scalar
        E_partner_all = omega_PB_snapped - E_arr
        overlap_tol = 1e-9 * dE_scalar
        partner_below_bottom = (
            supported_arr
            & (E_partner_all < bottom_edge)
            & (E_partner_all + 0.5 * dE_scalar > gap + overlap_tol)
        )
        if np.any(partner_below_bottom):
            n_dropped = int(np.count_nonzero(partner_below_bottom))
            raise ValueError(
                f"Pair-breaking photon partners are off-grid: {n_dropped} "
                "supported cells have their reflection partner ω−E in the "
                f"physical-but-unrepresented window (Δ={gap:.6g}, "
                f"first face {bottom_edge:.6g}) μeV for "
                f"ω_PB={omega_PB_snapped:.6g} μeV. Extend the energy grid "
                "down to the gap edge (energy_min_factor <= 1.0) so "
                "generation/recombination partners are representable."
            )

    return PairBreakingPhotonGridContract(
        index_shift=m,
        spacing=dE_scalar,
        snapped_omega=omega_PB_snapped,
        fractional_error=frac_err,
    )


def pair_breaking_photon_collision_components(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_PB: float,
    n_bar_PB: float,
    c_phot_PB: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Channel-resolved pair-breaking photon rates at one pixel.

    Combines scattering (``K⁺``, ``i ± m``), pair generation (``K⁻``,
    reflection partner), and pair recombination (``K⁻``, same partner).
    Requires ``ω_PB > 2Δ`` for pair-breaking to contribute; partners
    with ``E_j < Δ`` are skipped silently.

    Returns
    -------
    gain_scattering, loss_scattering, gain_pair, loss_pair
        Each has shape ``(NE,)``. Both gains include final-state Pauli
        blocking, so summing the two channel pairs gives
        ``df/dt = gain − loss_rate · f`` exactly as in
        :func:`pair_breaking_photon_collision_rates`. Keeping the
        number-conserving scattering and number-changing pair terms
        separate lets global balance certificates normalize against only
        the physically relevant turnover without duplicating this
        function's validation and grid guards.
    """
    E = ctx.E
    NE = E.size
    f = validated_occupation(f, E.shape, "Pair-breaking photon collision")

    if not np.isfinite(omega_PB) or omega_PB < 0.0:
        raise ValueError(f"omega_PB must be finite and non-negative; got {omega_PB}.")
    if not np.isfinite(n_bar_PB) or n_bar_PB < 0.0:
        raise ValueError(f"n_bar_PB must be finite and non-negative; got {n_bar_PB}.")
    if not np.isfinite(c_phot_PB) or c_phot_PB < 0.0:
        raise ValueError(f"c_phot_PB must be finite and non-negative; got {c_phot_PB}.")
    if omega_PB == 0.0 or c_phot_PB == 0.0:
        # A disabled channel is exactly equivalent to omitting it. Validate
        # the occupation and scalar API first, but do not impose uniform-grid,
        # Dynes, commensurability, threshold, or partner-coverage contracts on
        # a term whose coefficient is identically zero.
        return np.zeros(NE), np.zeros(NE), np.zeros(NE), np.zeros(NE)
    if ctx.dynes_gamma > 0.0:
        raise ValueError(
            "Pair-breaking photon collisions do not support dynes_gamma > 0: "
            "the collision coherence factors are ideal BCS."
        )

    # Use the cell-average partner density matched to the finite-volume
    # capacity.  On this required uniform grid, w_i*rho_bar_j is symmetric
    # under i<->j, preserving number for the scattering part and counting
    # pair creation/annihilation consistently for reflection partners.
    rho = ctx.cell_density
    supported = ctx.active_mask
    K_plus = ctx.K_plus
    K_minus = ctx.K_minus
    contract = validate_pair_breaking_photon_grid(
        E,
        ctx.dE,
        ctx.gap,
        omega_PB,
        supported=supported,
    )
    m = contract.index_shift
    dE_scalar = contract.spacing
    omega_PB_snapped = contract.snapped_omega
    if contract.fractional_error > 1e-6:
        # See sub_gap_photon: the accepted snap changes the SOLVED photon
        # energy; occupancies chosen for the nominal omega (thermal Bose
        # factors especially) must be evaluated at m*dE (2026-07-20 review).
        # Threshold-crossing snaps fail above, before this informational
        # warning, so warnings-as-errors cannot mask the actionable error.
        warnings.warn(
            f"pair_breaking_photon: omega_PB={omega_PB:.6g} μeV snapped to "
            f"m·dE={omega_PB_snapped:.6g} μeV "
            f"({contract.fractional_error:.2e} bins). Evaluate "
            "any energy-dependent photon occupancy at the snapped energy, "
            "not the nominal one.",
            RuntimeWarning,
            stacklevel=2,
        )

    gain_scattering = np.zeros(NE)
    loss_scattering = np.zeros(NE)
    gain_pair = np.zeros(NE)
    loss_pair = np.zeros(NE)
    one_minus_f = np.maximum(1.0 - f, 0.0)

    for i in range(NE):
        if not supported[i]:
            continue

        # --- 1. Scattering (K+, number-conserving) ---
        j_up = i + m
        if j_up < NE:
            U_plus = rho[j_up] * K_plus[i, j_up]
            gain_scattering[i] += c_phot_PB * U_plus * f[j_up] * (n_bar_PB + 1)
            loss_scattering[i] += c_phot_PB * U_plus * one_minus_f[j_up] * n_bar_PB

        j_down = i - m
        if j_down >= 0 and supported[j_down]:
            U_plus = rho[j_down] * K_plus[i, j_down]
            gain_scattering[i] += c_phot_PB * U_plus * f[j_down] * n_bar_PB
            loss_scattering[i] += c_phot_PB * U_plus * one_minus_f[j_down] * (n_bar_PB + 1)

        # --- 2 & 3. Generation + Recombination (K-, reflection partner) ---
        # Hard pair threshold: a photon below 2Δ cannot create or absorb a
        # QP pair, whatever cut-cell partners the grid happens to hold
        # (2026-07-20 review: an exactly commensurate ω_PB = 1.6Δ on a
        # gap-cut grid produced finite pair generation). Scattering above
        # keeps operating at any ω.
        if not pair_channel_open(omega_PB_snapped, ctx.gap):
            continue
        E_partner = omega_PB_snapped - E[i]
        j_r = round((E_partner - E[0]) / dE_scalar)
        if not (0 <= j_r < NE) or not supported[j_r]:
            continue
        U_minus = rho[j_r] * K_minus[i, j_r]
        # Generation: photon creates a QP pair; this site gains at rate
        # c · U⁻ · n_bar · (1 − f_j).
        gain_pair[i] += c_phot_PB * U_minus * n_bar_PB * one_minus_f[j_r]
        # Recombination: QP pair annihilates, emitting a photon; this
        # site loses at rate c · U⁻ · (1 + n_bar) · f_j.
        loss_pair[i] += c_phot_PB * U_minus * (1 + n_bar_PB) * f[j_r]

    return (
        gain_scattering * one_minus_f,
        loss_scattering,
        gain_pair * one_minus_f,
        loss_pair,
    )


def pair_breaking_photon_collision_rates(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_PB: float,
    n_bar_PB: float,
    c_phot_PB: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combined pair-breaking photon ``(gain, loss_rate)`` at one pixel.

    The pair generation/recombination channel is strictly open only for
    ``ω_PB > 2Δ``; number-conserving scattering remains available at any
    positive representable photon energy.
    """
    gain_scattering, loss_scattering, gain_pair, loss_pair = (
        pair_breaking_photon_collision_components(f, ctx, omega_PB, n_bar_PB, c_phot_PB)
    )
    return gain_scattering + gain_pair, loss_scattering + loss_pair
