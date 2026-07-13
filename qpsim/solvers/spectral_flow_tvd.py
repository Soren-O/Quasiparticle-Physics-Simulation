"""TVD finite-volume advection for the spectral-flow operator.

Advances a conserved profile ``u(E)`` under
``∂_t u + ∂_E (v u) = 0`` with spectral-flow velocity
``v(E) = (Δ / E) · Δ̇``. Uses monotonized-centered (MC) slope
reconstruction on a possibly non-uniform energy grid, upwind
numerical fluxes at cell interfaces, and an SSPRK(2,2) time step
(from :mod:`qpsim.solvers.ssprk`).

Ported from ``qpsim/numerics/advection.py`` at Gate 2; the SSPRK
stepper was split out as a generic primitive.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import numpy as np

from qpsim.solvers.ssprk import ssprk22_step

_MAX_SPECTRAL_FLOW_CFL = 0.8


def advect_spectral_flow(
    u: np.ndarray,
    E_bins: np.ndarray,
    dE_bins: np.ndarray,
    gap: float,
    gap_dot: float,
    dt: float,
    *,
    active_mask: np.ndarray | None = None,
) -> np.ndarray:
    """One SSPRK(2,2)+TVD advection step for ``∂_t u + ∂_E (v u) = 0``.

    Spectral-flow velocity is ``v(E) = (gap / E) · gap_dot``. Returns
    ``u`` unchanged when ``|gap_dot|`` is at roundoff (nothing to
    advect). Zero-flux boundary conditions at both ends of the grid.

    Two finite-domain caveats (audited 2026-06-10 against the paper's
    eq:full_kinetic_conservative / eq:dos_continuity):

    * The zero-flux ends are exact for the gap edge (the edge moves at
      exactly ``v(Δ) = Δ̇``, so no flux crosses it in the continuum) but
      are an approximation at ``E_max``: a falling gap starves the top
      ~``|v(E_max)|·t/dE`` cells (the analytic solution draws mass in
      from beyond the grid), and a rising gap piles outflow there. Keep
      ``E_max`` where occupations are negligible.
    * A grid built with ``energy_min_factor >= 1`` has no room below the
      initial gap: a falling gap's spectral inflow band ``(Δ_new, E_min)``
      is then unrepresentable. Gap-dynamics runs should leave sub-gap
      grid room.

    Parameters
    ----------
    u
        Conserved profile, shape ``(NE,)`` or ``(2, NE)``. The latter
        supports T1's ``(f_L, f_T)`` two-component layout.
    E_bins
        Energy bin centers (μeV).
    dE_bins
        Bin widths (μeV).
    gap
        Current gap value (μeV).
    gap_dot
        Time derivative of the gap (μeV/ns).
    dt
        Timestep (ns).
    active_mask
        Optional boolean mask; if given, bins outside the mask are
        zeroed after the step.

    Returns
    -------
    u_new
        Advected profile, same shape as ``u``. No bounds clipping — the
        caller recovers ``f`` from ``u = ρ · f`` by dividing by the
        post-step DOS and applies ``f ∈ [0, 1]`` clipping there.
    """
    u_arr = np.asarray(u, dtype=float)
    E = np.asarray(E_bins, dtype=float).ravel()
    dE = np.asarray(dE_bins, dtype=float).ravel()

    if abs(gap_dot) < 1e-30 or gap <= 0:
        return u_arr.copy()

    # The stability parameter is the *gap displacement* |gap_dot| dt, not
    # gap_dot or dt separately.  apply_gap_update obtains gap_dot from a root
    # jump, so decreasing dt alone leaves this raw CFL unchanged.  Estimate
    # it locally (important on non-uniform grids), then split the requested
    # displacement into genuinely smaller, stable advances.  The midpoint
    # gap in each substep follows the linearly moving spectrum.
    gap_displacement = float(gap_dot) * float(dt)
    gap_end = float(gap) + gap_displacement
    max_gap_magnitude = max(abs(float(gap)), abs(gap_end))
    raw_cfl = float(
        np.max(max_gap_magnitude * abs(gap_displacement) / (np.abs(E) * dE))
    )
    n_substeps = max(1, int(np.ceil(raw_cfl / _MAX_SPECTRAL_FLOW_CFL)))
    if raw_cfl > 1.0:
        warnings.warn(
            f"Spectral-flow displacement CFL {raw_cfl:.2f} > 1 for signed "
            f"gap displacement {gap_displacement:+.6g}; internally "
            f"subcycling into {n_substeps} advances (target CFL "
            f"{_MAX_SPECTRAL_FLOW_CFL:g}). Reduce |gap_dot|*dt or increase "
            "energy resolution to avoid this extra work; reducing dt alone "
            "does not help when gap_dot is recomputed from the same gap jump.",
            stacklevel=2,
        )

    if u_arr.ndim not in (1, 2) or (u_arr.ndim == 2 and u_arr.shape[0] != 2):
        raise ValueError(f"u must be shape (NE,) or (2, NE), got {u_arr.shape}")

    dt_sub = float(dt) / n_substeps
    gap_step = gap_displacement / n_substeps
    u_new = u_arr.copy()

    def _rhs_for_velocity(
        velocity: np.ndarray,
    ) -> Callable[[np.ndarray], np.ndarray]:
        def rhs(f: np.ndarray) -> np.ndarray:
            return _advection_rhs(f, E, velocity, dE)

        return rhs

    for substep in range(n_substeps):
        # Preserve the legacy small-step discretization bit-for-bit.  The
        # midpoint gap is needed only when a large requested displacement is
        # actually split into multiple stable advances.
        gap_mid = (
            float(gap)
            if n_substeps == 1
            else float(gap) + (substep + 0.5) * gap_step
        )
        v = (gap_mid / E) * gap_dot
        rhs = _rhs_for_velocity(v)

        if u_new.ndim == 1:
            u_new = ssprk22_step(u_new, rhs, dt_sub)
        else:
            advanced = np.empty_like(u_new)
            advanced[0] = ssprk22_step(u_new[0], rhs, dt_sub)
            advanced[1] = ssprk22_step(u_new[1], rhs, dt_sub)
            u_new = advanced

    if active_mask is not None:
        mask = np.asarray(active_mask, dtype=bool)
        if u_new.ndim == 1:
            u_new[~mask] = 0.0
        else:
            u_new[:, ~mask] = 0.0

    return u_new


def _advection_rhs(
    f: np.ndarray,
    E: np.ndarray,
    v: np.ndarray,
    dE: np.ndarray,
) -> np.ndarray:
    """Semi-discrete RHS: ``−∂_E flux`` with upwind MC-limited fluxes."""
    flux = _interface_fluxes(f, E, v, dE)
    return -(flux[1:] - flux[:-1]) / dE


def _interface_fluxes(
    f: np.ndarray,
    E: np.ndarray,
    v: np.ndarray,
    dE: np.ndarray,
) -> np.ndarray:
    """Upwind fluxes at interior interfaces with MC-limited left/right states."""
    NE = f.size
    flux = np.zeros(NE + 1, dtype=float)
    slopes = _mc_slopes(f, E)

    for i in range(1, NE):
        v_face = 0.5 * (v[i - 1] + v[i])
        left_state = f[i - 1] + 0.5 * dE[i - 1] * slopes[i - 1]
        right_state = f[i] - 0.5 * dE[i] * slopes[i]
        flux[i] = v_face * (left_state if v_face >= 0.0 else right_state)

    return flux


def _mc_slopes(f: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Monotonized-centered slopes on a possibly non-uniform grid."""
    slopes = np.zeros_like(f, dtype=float)
    if f.size < 3:
        return slopes

    for i in range(1, f.size - 1):
        d_left = max(E[i] - E[i - 1], 1e-30)
        d_right = max(E[i + 1] - E[i], 1e-30)
        s_left = (f[i] - f[i - 1]) / d_left
        s_right = (f[i + 1] - f[i]) / d_right
        s_center = (f[i + 1] - f[i - 1]) / max(E[i + 1] - E[i - 1], 1e-30)
        slopes[i] = _minmod(2.0 * s_left, s_center, 2.0 * s_right)

    return slopes


def _minmod(a: float, b: float, c: float) -> float:
    """Three-argument minmod limiter."""
    if a > 0.0 and b > 0.0 and c > 0.0:
        return min(a, b, c)
    if a < 0.0 and b < 0.0 and c < 0.0:
        return max(a, b, c)
    return 0.0
