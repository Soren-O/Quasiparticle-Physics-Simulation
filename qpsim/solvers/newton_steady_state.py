"""Newton solver for the collision-integral fixed point.

Solves ``I_coll[f] = 0`` for the occupation ``f(E)`` given fixed
collision ingredients: the kernel matrices ``K_s0`` / ``K_r0``, phonon
occupation factors (thermal or overridden), and optional photon-channel
parameters. The analytical Jacobian is assembled element-wise from the
gain-minus-loss decomposition of the collision integral.

Used both as a standalone steady-state solver (thermal-phonon limit,
``τ_l → 0``) and as the inner solve inside the Picard outer loop for
the finite-``τ_l`` regime.

Ported from ``_newton_solve_f`` in ``qpsim/numerics/steady_state.py``
with the Jacobian and residual helpers co-located here.
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    phonon_collision_rates,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.devices.external_flux import ExternalFlux
from qpsim.physics.spectral import SpectralContext


def newton_solve_f(
    ctx: SpectralContext,
    f: np.ndarray,
    *,
    K_s0: np.ndarray | None = None,
    K_r0: np.ndarray | None = None,
    T_bath: float = 0.0,
    active: np.ndarray | None = None,
    N_p_override: np.ndarray | None = None,
    N_emit_override: np.ndarray | None = None,
    N_abs_override: np.ndarray | None = None,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    external_flux: ExternalFlux | None = None,
    tol: float = 1e-14,
    max_iter: int = 200,
) -> np.ndarray:
    """Newton-solve ``f(E)`` against the collision residual.

    Parameters
    ----------
    ctx
        SpectralContext with current Δ.
    f
        Initial guess, shape ``(NE,)``. Overwritten/copied internally.
    K_s0, K_r0
        Base e-ph scattering/recombination kernels (shape ``(NE, NE)``)
        or ``None`` to disable the respective channel.
    T_bath
        Thermal phonon bath temperature. Used to compute default phonon
        occupations when no override is passed.
    active
        Bool mask of "active" bins (solve applies only on these).
        Defaults to ``ctx.active_mask``.
    N_p_override, N_emit_override, N_abs_override
        Non-equilibrium phonon occupation matrices. Overrides the
        thermal Bose-Einstein values computed from ``T_bath``. Used by
        the Picard outer loop (see ``services.steady_state``).
    photon_params
        ``{"omega_0", "n_bar", "c_phot"}`` for the sub-gap channel, or
        ``None`` to disable.
    pb_photon_params
        ``{"omega_PB", "n_bar_PB", "c_phot_PB"}`` for the PB channel,
        or ``None`` to disable.
    external_flux
        Optional :class:`qpsim.devices.ExternalFlux` (gain, loss_rate)
        contract added to the residual. Used by Junction-coupled
        regions in the device architecture (see
        ``docs/Device_Architecture.md``). When ``None`` (default),
        the solver path is bit-for-bit identical to pre-Phase-2 behavior.
    tol
        Absolute/relative convergence tolerance on the residual.
    max_iter
        Hard cap on Newton iterations.

    Returns
    -------
    np.ndarray
        Converged occupation clipped to ``[0, 1]``. Shape ``(NE,)``.

    Raises
    ------
    RuntimeError
        If the Jacobian is singular, the line search fails above ``tol``,
        or Newton doesn't converge within ``max_iter``.
    """
    NE = len(f)
    f_cur = np.array(f, dtype=float).ravel()

    if external_flux is not None:
        external_flux._validate_for_NE(NE)

    if active is None:
        active = ctx.active_mask
    n_active = int(np.sum(active))
    if n_active == 0:
        return f_cur

    # Default to thermal occupations when no overrides given and the
    # corresponding kernel is in play.
    N_p = N_p_override
    N_emit = N_emit_override
    N_abs = N_abs_override
    if N_p is None and K_s0 is not None:
        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
    if N_emit is None and K_r0 is not None:
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)

    max_residual = np.inf
    rate_scale = 0.0
    for iteration in range(max_iter):
        gain, loss_rate = _gain_loss_sum(
            f_cur, ctx, K_s0, K_r0, T_bath,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        R = gain - loss_rate * f_cur

        R_abs = np.abs(R[active])
        max_residual = float(np.max(R_abs))

        rate_scale = float(
            np.max(
                np.maximum(
                    np.abs(gain[active]),
                    np.abs(loss_rate[active] * f_cur[active]),
                )
            )
        )

        converged_abs = max_residual < tol
        converged_rel = rate_scale > 0 and max_residual / rate_scale < tol
        if converged_abs and (converged_rel or rate_scale == 0):
            return np.clip(f_cur, 0.0, 1.0)

        J = _jacobian_analytical(
            f_cur, ctx, K_s0, K_r0,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        J_act = J[np.ix_(active, active)]
        R_act = R[active]

        try:
            delta_f_act = np.linalg.solve(J_act, -R_act)
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                f"Singular Jacobian at Newton iteration {iteration}"
            ) from err

        delta_f = np.zeros(NE)
        delta_f[active] = delta_f_act

        # Simple backtracking line search.
        alpha = 1.0
        accepted = False
        for _ in range(20):
            f_trial = np.clip(f_cur + alpha * delta_f, 0.0, 1.0)
            R_trial = _residual(
                f_trial, ctx, K_s0, K_r0, T_bath,
                photon_params, pb_photon_params,
                N_p, N_emit, N_abs,
                external_flux,
            )
            if np.max(np.abs(R_trial[active])) < max_residual:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Line-search failure near the roundoff floor just means the
            # Newton step is smaller than machine precision. If the
            # residual is already at or below tol, accept — anything else
            # is noise chasing (mirrors scipy.optimize's Newton guard).
            if max_residual < tol:
                return np.clip(f_cur, 0.0, 1.0)
            raise RuntimeError(
                f"Newton line search failed at iteration {iteration}. "
                f"max |residual| = {max_residual:.2e}"
            )

        f_cur = np.clip(f_cur + alpha * delta_f, 0.0, 1.0)

    raise RuntimeError(
        f"Newton iteration did not converge in {max_iter} iterations. "
        f"Final max |residual| = {max_residual:.2e}, "
        f"relative = {max_residual / max(rate_scale, 1e-30):.2e}"
    )


def _gain_loss_sum(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    photon_params: dict[str, float] | None,
    pb_photon_params: dict[str, float] | None,
    N_p: np.ndarray | None,
    N_emit: np.ndarray | None,
    N_abs: np.ndarray | None,
    external_flux: ExternalFlux | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Total (gain, loss_rate) from all enabled collision channels."""
    gain, loss_rate = phonon_collision_rates(
        f, ctx, K_s0, K_r0, T_bath,
        N_p_override=N_p,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )

    if photon_params is not None:
        gain_ph, loss_ph = sub_gap_photon_collision_rates(
            f, ctx,
            photon_params["omega_0"],
            photon_params["n_bar"],
            photon_params["c_phot"],
        )
        gain = gain + gain_ph
        loss_rate = loss_rate + loss_ph

    if pb_photon_params is not None:
        gain_pb, loss_pb = pair_breaking_photon_collision_rates(
            f, ctx,
            pb_photon_params["omega_PB"],
            pb_photon_params["n_bar_PB"],
            pb_photon_params["c_phot_PB"],
        )
        gain = gain + gain_pb
        loss_rate = loss_rate + loss_pb

    if external_flux is not None:
        gain = gain + external_flux.gain
        loss_rate = loss_rate + external_flux.loss_rate

    return gain, loss_rate


def _residual(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    photon_params: dict[str, float] | None,
    pb_photon_params: dict[str, float] | None,
    N_p: np.ndarray | None,
    N_emit: np.ndarray | None,
    N_abs: np.ndarray | None,
    external_flux: ExternalFlux | None = None,
) -> np.ndarray:
    """``df/dt = gain − loss_rate · f`` at the supplied ``f``."""
    gain, loss_rate = _gain_loss_sum(
        f, ctx, K_s0, K_r0, T_bath,
        photon_params, pb_photon_params,
        N_p, N_emit, N_abs,
        external_flux,
    )
    return gain - loss_rate * f


def _jacobian_analytical(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    photon_params: dict[str, float] | None,
    pb_photon_params: dict[str, float] | None,
    N_p: np.ndarray | None,
    N_emit: np.ndarray | None,
    N_abs: np.ndarray | None,
    external_flux: ExternalFlux | None = None,
) -> np.ndarray:
    """Analytical Jacobian ``∂R_i/∂f_j`` of the collision residual.

    Assembled term-by-term from the gain/loss decomposition. Each
    channel's contribution is derived by differentiating through the
    ``(1 − f)`` Pauli factors, Bose-Einstein occupations held fixed,
    and the occupation-bilinear kernel matmuls.
    """
    NE = len(f)
    rho = ctx.rho
    dE = ctx.dE
    w = rho * dE
    omf = np.maximum(1.0 - f, 0.0)
    diag_idx = np.arange(NE)

    J = np.zeros((NE, NE))

    # Scattering
    if K_s0 is not None and N_p is not None:
        K_s_eff = K_s0 * N_p
        # Off-diagonal: ∂R_i/∂f_j = (1 − f_i) K_s[j, i] w_j + f_i K_s[i, j] w_j
        J += (omf[:, None] * K_s_eff.T + f[:, None] * K_s_eff) * w[None, :]
        # Diagonal correction: subtract the bulk in/out rates at i.
        A = K_s_eff.T @ (rho * f * dE)
        B = K_s_eff @ (rho * omf * dE)
        J[diag_idx, diag_idx] -= A + B

    # Recombination (Kaplan Eq. (8) per-QP normalization, matching
    # phonon_collision_rates)
    if K_r0 is not None and N_emit is not None and N_abs is not None:
        mixed = omf[:, None] * N_abs + f[:, None] * N_emit
        J -= K_r0 * mixed * w[None, :]
        C = (K_r0 * N_abs) @ (rho * omf * dE)
        D = (K_r0 * N_emit) @ (rho * f * dE)
        J[diag_idx, diag_idx] -= C + D

    # Sub-gap photon (K+, partners at i ± m)
    if photon_params is not None:
        omega_0 = photon_params["omega_0"]
        n_bar = photon_params["n_bar"]
        c_phot = photon_params["c_phot"]
        dE_scalar = uniform_grid_spacing(
            ctx.E, dE, "Sub-gap photon analytical Jacobian"
        )
        m = round(omega_0 / dE_scalar)
        if m > 0:
            K_plus = ctx.K_plus
            gap = ctx.gap
            for i in range(NE):
                j_up = i + m
                if j_up < NE:
                    U = rho[j_up] * K_plus[i, j_up]
                    J[i, i] -= c_phot * U * (f[j_up] + n_bar)
                    J[i, j_up] += c_phot * U * (n_bar + 1.0 - f[i])

                j_dn = i - m
                if j_dn >= 0 and ctx.E[j_dn] >= gap:
                    U = rho[j_dn] * K_plus[i, j_dn]
                    J[i, i] -= c_phot * U * (n_bar + 1.0 - f[j_dn])
                    J[i, j_dn] += c_phot * U * (n_bar + f[i])

    # Pair-breaking photon (K+ scattering + K- gen/rec)
    if pb_photon_params is not None:
        omega_PB = pb_photon_params["omega_PB"]
        n_bar_pb = pb_photon_params["n_bar_PB"]
        c_pb = pb_photon_params["c_phot_PB"]
        dE_scalar = uniform_grid_spacing(
            ctx.E, dE, "Pair-breaking photon analytical Jacobian"
        )
        m_pb = round(omega_PB / dE_scalar)
        omega_PB_snapped = m_pb * dE_scalar
        K_plus = ctx.K_plus
        K_minus = ctx.K_minus
        gap = ctx.gap
        E = ctx.E

        for i in range(NE):
            if m_pb > 0:
                j_up = i + m_pb
                if j_up < NE:
                    U = rho[j_up] * K_plus[i, j_up]
                    J[i, i] -= c_pb * U * (f[j_up] + n_bar_pb)
                    J[i, j_up] += c_pb * U * (n_bar_pb + 1.0 - f[i])

                j_dn = i - m_pb
                if j_dn >= 0 and E[j_dn] >= gap:
                    U = rho[j_dn] * K_plus[i, j_dn]
                    J[i, i] -= c_pb * U * (n_bar_pb + 1.0 - f[j_dn])
                    J[i, j_dn] += c_pb * U * (n_bar_pb + f[i])

            E_partner = omega_PB_snapped - E[i]
            if E_partner < gap:
                continue
            j_r = round((E_partner - E[0]) / dE_scalar)
            if j_r < 0 or j_r >= NE:
                continue
            U_m = rho[j_r] * K_minus[i, j_r]
            # Generation: R_gen_i = c · U⁻ · n_bar · (1 − f_i)(1 − f_j)
            # Recombination: R_rec_i = −c · U⁻ · (1 + n_bar) · f_j · f_i
            # ∂R_i/∂f_i = −c · U⁻ · (n_bar + f_j)
            # ∂R_i/∂f_j = −c · U⁻ · (n_bar + f_i)
            J[i, i] -= c_pb * U_m * (n_bar_pb + f[j_r])
            J[i, j_r] -= c_pb * U_m * (n_bar_pb + f[i])

    # ExternalFlux contributes -loss_rate to the diagonal (gain is f-
    # independent so contributes zero; the linear -loss_rate*f term has
    # diagonal Jacobian = -loss_rate).
    if external_flux is not None:
        J[diag_idx, diag_idx] -= external_flux.loss_rate

    return J
