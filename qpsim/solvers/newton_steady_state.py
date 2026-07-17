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

from typing import TYPE_CHECKING

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    phonon_collision_rates,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.physics.spectral import SpectralContext

if TYPE_CHECKING:
    # Type-annotation-only import. A runtime import of qpsim.devices.external_flux
    # triggers qpsim.devices.__init__ -> m25_junction -> services ->
    # steady_state -> newton_steady_state, a circular import that made this
    # module (and tests/solvers/test_newton_steady_state.py) unimportable in
    # isolation. PEP 563 keeps the annotations as strings, so the class object
    # is never needed at runtime here.
    from qpsim.devices.external_flux import ExternalFlux


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
    backward_error_tol: float = 1e-6,
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
        Absolute convergence tolerance on the dimensional residual.
    backward_error_tol
        Scale-independent L1 gain/loss backward-error limit. Both this physical
        certificate and the dimensional ``tol`` must pass on every return path.
    max_iter
        Hard cap on Newton iterations.

    Returns
    -------
    np.ndarray
        Converged occupation clipped to ``[0, 1]``. Shape ``(NE,)``.

    Raises
    ------
    ValueError
        If the initial occupation is not a finite one-dimensional array on the
        spectral grid or contains values outside ``[0, 1]``.
    RuntimeError
        If the Jacobian is singular, the line search fails above ``tol``,
        or Newton doesn't converge within ``max_iter``.
    """
    f_cur = np.asarray(f, dtype=float)
    if f_cur.ndim != 1 or f_cur.shape != ctx.E.shape:
        raise ValueError(
            f"initial occupation f must have shape {ctx.E.shape}; got {f_cur.shape}"
        )
    f_cur = f_cur.copy()
    if not np.all(np.isfinite(f_cur)):
        raise ValueError("initial occupation f must contain only finite values")
    if np.any((f_cur < 0.0) | (f_cur > 1.0)):
        raise ValueError("initial occupation f must lie in [0, 1]")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive; got {tol}.")
    if not np.isfinite(T_bath) or T_bath < 0.0:
        raise ValueError(f"T_bath must be finite and non-negative; got {T_bath}.")
    if not np.isfinite(backward_error_tol) or backward_error_tol <= 0.0:
        raise ValueError(
            "backward_error_tol must be finite and positive; "
            f"got {backward_error_tol}."
        )
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise ValueError(f"max_iter must be a positive integer; got {max_iter}.")
    NE = f_cur.size

    if external_flux is not None:
        external_flux._validate_for_NE(NE)
        external_flux._validate_gain_support(ctx.active_mask)

    if active is None:
        active = ctx.active_mask
    else:
        active = np.asarray(active)
        if active.dtype != np.bool_ or active.ndim != 1 or active.shape != ctx.E.shape:
            raise ValueError(
                "active must be a one-dimensional bool mask with shape "
                f"{ctx.E.shape}; got dtype={active.dtype}, shape={active.shape}."
            )
        active = active.copy()
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
    backward_error = float("inf")
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

        backward_error = _gain_loss_backward_error(
            gain,
            loss_rate,
            f_cur,
            active,
        )

        converged_abs = max_residual < tol
        converged_balance = backward_error <= backward_error_tol
        if converged_abs and converged_balance:
            # Return exactly the feasible vector whose residual was certified.
            # The initial state is validated and accepted trials are projected
            # before their residual is evaluated, so no post-hoc clip is needed.
            return f_cur.copy()

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
            raise RuntimeError(
                f"Newton line search failed at iteration {iteration}. "
                f"max |residual| = {max_residual:.2e}, "
                f"max-relative = "
                f"{max_residual / max(rate_scale, 1e-300):.2e}, "
                f"gain/loss backward error = {backward_error:.2e} "
                f"(limit {backward_error_tol:.2e})"
            )

        f_cur = np.clip(f_cur + alpha * delta_f, 0.0, 1.0)

    raise RuntimeError(
        f"Newton iteration did not converge in {max_iter} iterations. "
        f"Final max |residual| = {max_residual:.2e}, "
        f"max-relative = {max_residual / max(rate_scale, 1e-300):.2e}, "
        f"gain/loss backward error = {backward_error:.2e} "
        f"(limit {backward_error_tol:.2e})"
    )


def _gain_loss_backward_error(
    gain: np.ndarray,
    loss_rate: np.ndarray,
    f: np.ndarray,
    active: np.ndarray,
) -> float:
    """L1 normwise backward error of ``gain - loss_rate*f = 0``."""
    gain_arr = np.asarray(gain, dtype=float)
    loss_arr = np.asarray(loss_rate, dtype=float)
    f_arr = np.asarray(f, dtype=float)
    active_arr = np.asarray(active)
    if not (gain_arr.shape == loss_arr.shape == f_arr.shape == active_arr.shape):
        raise ValueError("gain, loss_rate, f, and active must have the same shape.")
    if active_arr.dtype != np.bool_:
        raise ValueError("active must be a bool mask.")
    if np.any(
        ~np.isfinite(np.stack((gain_arr, loss_arr, f_arr), axis=0))
    ):
        raise ValueError("gain, loss_rate, and f must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        loss_term = loss_arr[active_arr] * f_arr[active_arr]
    residual = gain_arr[active_arr] - loss_term
    if np.any(~np.isfinite(loss_term)) or np.any(~np.isfinite(residual)):
        raise ValueError("Assembled gain/loss balance terms must be finite.")
    gain_active = gain_arr[active_arr]
    common = float(
        max(
            np.max(np.abs(gain_active), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
            np.max(np.abs(residual), initial=0.0),
        )
    )
    if common == 0.0:
        return 0.0
    denominator = float(
        np.sum(np.abs(gain_active) / common)
        + np.sum(np.abs(loss_term) / common)
    )
    numerator = float(np.sum(np.abs(residual) / common))
    return numerator / denominator if denominator > 0.0 else float("inf")


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

    # Zero-capacity rows are storage placeholders, not dynamical states.
    # Public solver entry points reject positive ExternalFlux gain on these
    # rows; mask every channel here so harmless loss-only terms cannot make a
    # custom active set or the coupled residual depend on placeholder f.
    unsupported = ~ctx.active_mask
    gain[unsupported] = 0.0
    loss_rate[unsupported] = 0.0

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
    dE = ctx.dE
    w = ctx.cell_weights
    rho_bar = ctx.cell_density
    supported = ctx.active_mask
    omf = np.maximum(1.0 - f, 0.0)
    diag_idx = np.arange(NE)

    J = np.zeros((NE, NE))

    # Scattering
    if K_s0 is not None and N_p is not None:
        K_s_eff = K_s0 * N_p
        # Off-diagonal: ∂R_i/∂f_j = (1 − f_i) K_s[j, i] w_j + f_i K_s[i, j] w_j
        J += (omf[:, None] * K_s_eff.T + f[:, None] * K_s_eff) * w[None, :]
        # Diagonal correction: subtract the bulk in/out rates at i.
        A = K_s_eff.T @ (w * f)
        B = K_s_eff @ (w * omf)
        J[diag_idx, diag_idx] -= A + B

    # Recombination (Kaplan Eq. (8) per-QP normalization, matching
    # phonon_collision_rates)
    if K_r0 is not None and N_emit is not None and N_abs is not None:
        mixed = omf[:, None] * N_abs + f[:, None] * N_emit
        J -= K_r0 * mixed * w[None, :]
        C = (K_r0 * N_abs) @ (w * omf)
        D = (K_r0 * N_emit) @ (w * f)
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
            for i in range(NE):
                if not supported[i]:
                    continue
                j_up = i + m
                if j_up < NE and supported[j_up]:
                    U = rho_bar[j_up] * K_plus[i, j_up]
                    J[i, i] -= c_phot * U * (f[j_up] + n_bar)
                    J[i, j_up] += c_phot * U * (n_bar + 1.0 - f[i])

                j_dn = i - m
                if j_dn >= 0 and supported[j_dn]:
                    U = rho_bar[j_dn] * K_plus[i, j_dn]
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
        E = ctx.E

        for i in range(NE):
            if not supported[i]:
                continue
            if m_pb > 0:
                j_up = i + m_pb
                if j_up < NE and supported[j_up]:
                    U = rho_bar[j_up] * K_plus[i, j_up]
                    J[i, i] -= c_pb * U * (f[j_up] + n_bar_pb)
                    J[i, j_up] += c_pb * U * (n_bar_pb + 1.0 - f[i])

                j_dn = i - m_pb
                if j_dn >= 0 and supported[j_dn]:
                    U = rho_bar[j_dn] * K_plus[i, j_dn]
                    J[i, i] -= c_pb * U * (n_bar_pb + 1.0 - f[j_dn])
                    J[i, j_dn] += c_pb * U * (n_bar_pb + f[i])

            E_partner = omega_PB_snapped - E[i]
            j_r = round((E_partner - E[0]) / dE_scalar)
            if j_r < 0 or j_r >= NE or not supported[j_r]:
                continue
            U_m = rho_bar[j_r] * K_minus[i, j_r]
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

    # Match the collision-rate support contract exactly.  This is mostly
    # defensive for callers supplying a custom active set; the default Newton
    # set is the same finite-volume support.
    J[~ctx.active_mask, :] = 0.0
    return J
