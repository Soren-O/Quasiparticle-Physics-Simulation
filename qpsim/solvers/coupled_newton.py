"""Coupled Newton solver for the joint (f, n_ph) steady state.

Solves the 2-block residual

    R_f(f, n_ph)  = I_coll[f, n_ph]                       = 0
    R_ph(f, n_ph) = a_ph[f] + b_ph[f]·n_ph + (n_th − n_ph)/τ_l = 0

as a single monolithic Newton on the stacked vector ``(f, n_ph)``.
The Jacobian has the block structure

    J = [[J_ff, J_fn], [J_nf, J_nn]]

with:

* ``J_ff = ∂R_f/∂f`` — analytical, reuses the per-channel assembly
  from :mod:`qpsim.solvers.newton_steady_state`.
* ``J_nn = ∂R_ph/∂n_ph`` — diagonal in Ph0: ``b_ph − 1/τ_l``.
* ``J_fn = ∂R_f/∂n_ph`` and ``J_nf = ∂R_ph/∂f`` — forward finite
  differences in this v1 implementation. Analytical cross Jacobians
  would speed production-scale (NE ≳ 800) runs; that's a post-Gate-4
  optimization flagged in :issue:`coupled-newton-analytical-cross`.

Unlocks the Fischer 2023 ``τ_l/τ_PB = 10`` strong-bottleneck regime
that the Picard + Anderson path in
:mod:`qpsim.services.steady_state` fails to converge on.
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import (
    compute_phonon_source_sink,
    phonon_occupation_matrices_from_state,
)
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import _gain_loss_sum, _jacobian_analytical


def coupled_newton_solve(
    ctx: SpectralContext,
    f_init: np.ndarray,
    n_ph_init: np.ndarray,
    *,
    omega_bins: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    K_s0: np.ndarray | None = None,
    K_r0: np.ndarray | None = None,
    T_bath: float = 0.0,
    tau_l: float = 0.0,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    tol: float = 1e-10,
    max_iter: int = 50,
    fd_step: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``R_f = 0`` and ``R_ph = 0`` simultaneously for ``(f, n_ph)``.

    Parameters
    ----------
    ctx
        SpectralContext with current Δ.
    f_init
        QP occupation initial guess, shape ``(NE,)``.
    n_ph_init
        Phonon occupation initial guess, shape ``(N_omega,)``.
    omega_bins, omega_idx_diff, omega_idx_sum, diff_sign
        Outputs of :func:`qpsim.collisions.phonon.build_phonon_frequency_map`
        for ``ctx.E``.
    K_s0, K_r0
        Base e-ph scattering / recombination kernels, shape ``(NE, NE)``
        each (or ``None`` to disable the channel).
    T_bath
        Substrate bath temperature (K); sets ``n_th(ω)``.
    tau_l
        Phonon bath-escape time (ns). ``0.0`` means no substrate
        coupling; ``> 0`` means finite escape.
    photon_params, pb_photon_params
        Optional photon channel dicts (same shape as for
        :func:`qpsim.solvers.newton_steady_state.newton_solve_f`).
    tol
        Infinity-norm tolerance on the combined residual
        ``max(|R_f|, |R_ph|)``.
    max_iter
        Cap on Newton iterations.
    fd_step
        Forward-FD step size for the cross-Jacobian blocks.

    Returns
    -------
    (f, n_ph)
        Converged QP and phonon distributions; ``f`` is clipped to
        ``[0, 1]`` and ``n_ph`` to ``≥ 0``.

    Raises
    ------
    RuntimeError
        On non-convergence, line-search failure, or singular Jacobian.
    """
    NE = int(f_init.size)
    N_omega = int(omega_bins.size)
    if n_ph_init.size != N_omega:
        raise ValueError(
            f"n_ph_init length {n_ph_init.size} must match omega_bins size {N_omega}."
        )

    f = np.clip(np.asarray(f_init, dtype=float).ravel(), 0.0, 1.0)
    n_ph = np.maximum(np.asarray(n_ph_init, dtype=float).ravel(), 0.0)

    n_th = thermal_phonon_occupation(omega_bins, T_bath)
    inv_tau_l = 1.0 / tau_l if tau_l > 0 else 0.0

    def residual(
        f_state: np.ndarray, n_ph_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph_state, omega_idx_diff, omega_idx_sum, diff_sign,
        )
        gain, loss = _gain_loss_sum(
            f_state, ctx, K_s0, K_r0, T_bath,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
        )
        R_f = gain - loss * f_state

        a_ph, b_ph = compute_phonon_source_sink(
            f_state, ctx, K_s0, K_r0,
            omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
        )
        if tau_l > 0:
            R_ph = a_ph + b_ph * n_ph_state + (n_th - n_ph_state) * inv_tau_l
        else:
            R_ph = a_ph + b_ph * n_ph_state

        return R_f, R_ph

    last_norm = np.inf
    for iteration in range(max_iter):
        R_f, R_ph = residual(f, n_ph)
        norm = max(float(np.max(np.abs(R_f))), float(np.max(np.abs(R_ph))))
        last_norm = norm
        if norm < tol:
            return f, n_ph

        # Analytical diagonal blocks.
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph, omega_idx_diff, omega_idx_sum, diff_sign,
        )
        J_ff = _jacobian_analytical(
            f, ctx, K_s0, K_r0,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
        )
        _, b_ph = compute_phonon_source_sink(
            f, ctx, K_s0, K_r0,
            omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
        )
        J_nn = np.diag(b_ph - inv_tau_l)

        # Finite-difference cross blocks.
        J_fn = np.zeros((NE, N_omega))
        for k in range(N_omega):
            n_ph_pert = n_ph.copy()
            n_ph_pert[k] += fd_step
            R_f_pert, _ = residual(f, n_ph_pert)
            J_fn[:, k] = (R_f_pert - R_f) / fd_step

        J_nf = np.zeros((N_omega, NE))
        for j in range(NE):
            f_pert = f.copy()
            f_pert[j] += fd_step
            _, R_ph_pert = residual(f_pert, n_ph)
            J_nf[:, j] = (R_ph_pert - R_ph) / fd_step

        J = np.block([[J_ff, J_fn], [J_nf, J_nn]])
        R = np.concatenate([R_f, R_ph])

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                f"Coupled Newton Jacobian singular at iteration {iteration}."
            ) from err

        delta_f = delta[:NE]
        delta_n = delta[NE:]

        # Backtracking line search on the combined residual norm.
        alpha = 1.0
        accepted = False
        for _ in range(20):
            f_trial = np.clip(f + alpha * delta_f, 0.0, 1.0)
            n_trial = np.maximum(n_ph + alpha * delta_n, 0.0)
            R_f_t, R_ph_t = residual(f_trial, n_trial)
            norm_t = max(
                float(np.max(np.abs(R_f_t))), float(np.max(np.abs(R_ph_t)))
            )
            if norm_t < norm:
                f, n_ph = f_trial, n_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            if norm < tol:
                return f, n_ph
            raise RuntimeError(
                f"Coupled Newton line search failed at iteration {iteration}. "
                f"max |residual| = {norm:.2e}"
            )

    raise RuntimeError(
        f"Coupled Newton did not converge in {max_iter} iterations. "
        f"Final max |residual| = {last_norm:.2e}"
    )
