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
* ``J_fn = ∂R_f/∂n_ph`` and ``J_nf = ∂R_ph/∂f`` — analytical when
  ``analytic_cross=True`` (closed form, O(NE²), exact), else forward
  finite differences (default; O(NE³) and unreliable at strong drive
  where ``f, n_ph`` ≪ 1, the regime that drives Fischer 2023 Fig. 6).
  The analytical path resolves :issue:`coupled-newton-analytical-cross`
  and unlocks the Fig. 6 strong-drive tail at production grids
  (NE ≳ 800) in seconds rather than tens of minutes per point.

Unlocks the Fischer 2023 ``τ_l/τ_PB = 10`` strong-bottleneck regime
that the Picard + Anderson path in
:mod:`qpsim.services.steady_state` fails to converge on.
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import (
    compute_phonon_source_sink,
    phonon_collision_jacobian_nph,
    phonon_occupation_matrices_from_state,
    phonon_source_sink_jacobian_f,
)
from qpsim.devices.external_flux import ExternalFlux
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
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
    T_bath: float = 0.0,
    tau_l: float = 0.0,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    external_flux: ExternalFlux | None = None,
    tol: float = 1e-10,
    step_rtol: float = 0.0,
    max_iter: int = 50,
    fd_step: float = 1e-8,
    fd_floor: float = 1e-12,
    analytic_cross: bool = False,
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
    K_s0_phonon_side, K_r0_phonon_side
        **Opt-in** phonon-side scattering and recombination/pair-breaking
        kernels (build via
        :func:`qpsim.collisions.phonon.build_scattering_kernel_phonon_side` and
        :func:`qpsim.collisions.phonon.build_recombination_kernel_phonon_side`).
        Forwarded to :func:`compute_phonon_source_sink` for the
        phonon-equation residual + diagonal-block Jacobian; when
        supplied, the phonon-equation rates use the F&C 2023 Eq. 12
        prefactors instead of the QP-side kernels. The QP-equation
        residual + ``J_ff`` continue to use ``K_r0`` (which carries the
        QP-side ``(E_sum/k_BT_c)²/(τ₀ k_BT_c)`` prefactor needed by the
        QP collision integrals). ``None`` (default) preserves legacy
        behavior bit-for-bit.
    T_bath
        Substrate bath temperature (K); sets ``n_th(ω)``.
    tau_l
        Phonon bath-escape time (ns). ``0.0`` means no substrate
        coupling; ``> 0`` means finite escape.
    photon_params, pb_photon_params
        Optional photon channel dicts (same shape as for
        :func:`qpsim.solvers.newton_steady_state.newton_solve_f`).
    external_flux
        Optional :class:`qpsim.devices.ExternalFlux` boundary
        source/sink contract on the f-equation. Affects only the
        f-block; the phonon-block residual is unchanged. ``None``
        is bit-for-bit identical to pre-Phase-2 behavior.
    tol
        Absolute infinity-norm tolerance on the combined residual
        ``max(|R_f|, |R_ph|)``. Used as the early-exit test when
        ``step_rtol == 0`` (legacy default), and as a safety floor otherwise.
    step_rtol
        Scale-invariant convergence tolerance on the relative Newton step
        ``max(‖Δf‖∞/‖f‖∞, ‖Δn‖∞/‖n‖∞)``. ``0.0`` (default) keeps the legacy
        absolute-residual behavior exactly. Set ``> 0`` when all physical
        amplitudes are tiny (e.g. f, n_ph ~ 1e-10 in the cold-bath
        strong-suppression regime that drives Fischer Fig. 6), where an
        absolute ``tol`` is unreliable: a warm continuation seed can sit just
        below ``tol`` and exit at iteration 0 with a stale, under-converged
        state. The relative-step test forces a refining step and is meaningful
        whether f ~ 1 or f ~ 1e-10.
    max_iter
        Cap on Newton iterations.
    fd_step
        Relative forward-FD factor for the cross-Jacobian blocks. The
        per-component step is ``h_k = max(fd_step·|x_k|, fd_floor)``, so for
        O(1) state entries this matches the historical absolute ``1e-8`` step,
        while small entries (``f``, ``n_ph`` ≪ 1) get a proportionally smaller
        step instead of one that dwarfs their own value.
    fd_floor
        Absolute lower bound on the per-component FD step, keeping ``h_k``
        above the residual roundoff floor for near-zero entries. Default
        ``1e-12`` suits the ``f``/``n_ph`` ~ 1e-10..1e-18 cold-bath regime.
        Ignored when ``analytic_cross=True``.
    analytic_cross
        When ``True``, assemble the cross blocks ``J_fn`` / ``J_nf`` from
        their closed forms (:func:`qpsim.collisions.phonon.phonon_collision_jacobian_nph`
        and :func:`qpsim.collisions.phonon.phonon_source_sink_jacobian_f`)
        instead of finite differences. Exact and O(NE²) rather than O(NE³),
        and free of the scaling pathology that makes the FD secant
        meaningless when ``f``, ``n_ph`` ≪ 1 (the cold-bath strong-drive
        regime of Fischer Fig. 6). ``False`` (default) preserves the legacy
        FD behavior bit-for-bit, so the pinned validation suite is unaffected.

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
    if external_flux is not None:
        external_flux._validate_for_NE(NE)

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
            external_flux,
        )
        R_f = gain - loss * f_state

        a_ph, b_ph = compute_phonon_source_sink(
            f_state, ctx, K_s0, K_r0,
            omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
            K_s0_phonon_side=K_s0_phonon_side,
            K_r0_phonon_side=K_r0_phonon_side,
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
        # Absolute-residual early exit (legacy). Disabled when step_rtol>0:
        # an absolute tol is unreliable when all amplitudes are tiny, since a
        # warm continuation seed can sit just below tol and exit at iteration 0
        # with a stale state. With step_rtol>0 convergence is judged on the
        # relative Newton step after a refining step is taken (below).
        if step_rtol <= 0.0 and norm < tol:
            return f, n_ph

        # Analytical diagonal blocks.
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph, omega_idx_diff, omega_idx_sum, diff_sign,
        )
        J_ff = _jacobian_analytical(
            f, ctx, K_s0, K_r0,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        _, b_ph = compute_phonon_source_sink(
            f, ctx, K_s0, K_r0,
            omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
            K_s0_phonon_side=K_s0_phonon_side,
            K_r0_phonon_side=K_r0_phonon_side,
        )
        J_nn = np.diag(b_ph - inv_tau_l)

        if analytic_cross:
            # Closed-form cross blocks: exact and O(NE²), vs the O(NE³) and
            # scale-fragile FD path. R_f is linear in n_ph, so J_fn is
            # n_ph-independent and uses the QP-side kernels K_s0/K_r0 (the
            # photon / external-flux channels are n_ph-independent ⇒ zero).
            # R_ph's f-dependence is differentiated through
            # compute_phonon_source_sink, so J_nf carries the same phonon-side
            # kernels + quadrature correction, assembled as
            # J_nf = ∂a_ph/∂f + n_ph · ∂b_ph/∂f.
            J_fn = phonon_collision_jacobian_nph(
                f, ctx, K_s0, K_r0,
                omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
            )
            da_df, db_df = phonon_source_sink_jacobian_f(
                f, ctx, K_s0, K_r0,
                omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
                K_s0_phonon_side=K_s0_phonon_side,
                K_r0_phonon_side=K_r0_phonon_side,
            )
            J_nf = da_df + n_ph[:, None] * db_df
        else:
            # Finite-difference cross blocks with scale-aware per-component steps.
            # A single absolute step is wrong when the state lives far below 1
            # (e.g. f ~ 1e-10, n_ph ~ 1e-18 in the cold-bath strong-suppression
            # regime that drives Fischer Fig. 6): an absolute fd_step=1e-8 then
            # perturbs each component by orders of magnitude more than its own
            # value, so the secant probes deep nonlinearity and the cross-Jacobian
            # is meaningless — Newton converges to a spurious fixed point.
            # h_k = max(fd_step·|x_k|, fd_floor) reduces to the old absolute step
            # for O(1) entries. The relative term keeps the step proportional to the
            # value where that matters most — the f-block, where the residual is
            # nonlinear in f, so an oversized step probes spurious curvature. fd_floor
            # is an empirical lower bound that lands in a usable secant window above
            # the residual roundoff floor; note it can exceed the value of near-zero
            # entries (e.g. n_ph ~ 1e-18), which is acceptable because R_f is ~linear
            # in n_ph and a large relative step there still returns a good derivative.
            h_n = np.maximum(fd_step * np.abs(n_ph), fd_floor)
            J_fn = np.zeros((NE, N_omega))
            for k in range(N_omega):
                n_ph_pert = n_ph.copy()
                n_ph_pert[k] += h_n[k]
                R_f_pert, _ = residual(f, n_ph_pert)
                J_fn[:, k] = (R_f_pert - R_f) / h_n[k]

            h_f = np.maximum(fd_step * np.abs(f), fd_floor)
            J_nf = np.zeros((N_omega, NE))
            for j in range(NE):
                f_pert = f.copy()
                f_pert[j] += h_f[j]
                _, R_ph_pert = residual(f_pert, n_ph)
                J_nf[:, j] = (R_ph_pert - R_ph) / h_f[j]

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
        rel_step = np.inf
        for _ in range(20):
            f_trial = np.clip(f + alpha * delta_f, 0.0, 1.0)
            n_trial = np.maximum(n_ph + alpha * delta_n, 0.0)
            R_f_t, R_ph_t = residual(f_trial, n_trial)
            norm_t = max(
                float(np.max(np.abs(R_f_t))), float(np.max(np.abs(R_ph_t)))
            )
            if norm_t < norm:
                # Relative Newton-step size for the scale-invariant convergence
                # test; measured against the state magnitude so it is meaningful
                # whether f ~ 1 or f ~ 1e-10. ``or 1.0`` guards a zero state.
                f_scale = float(np.max(np.abs(f))) or 1.0
                n_scale = float(np.max(np.abs(n_ph))) or 1.0
                rel_step = max(
                    float(np.max(np.abs(f_trial - f))) / f_scale,
                    float(np.max(np.abs(n_trial - n_ph))) / n_scale,
                )
                f, n_ph = f_trial, n_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            if norm < tol:
                return f, n_ph
            # Scale-invariant fallback: when the line search cannot reduce the
            # residual further but Newton's own step is negligible relative to
            # the state, we are at the fixed point. In the tiny-amplitude
            # strong-drive regime (f, n_ph ~ 1e-10) the residual floors at
            # ~1e-10 — above an absolute tol like 1e-12 — so without this the
            # solver spuriously fails points that are in fact converged (the
            # Fischer Fig. 6 transition NaNs). A genuinely stuck / near-singular
            # step is large in the null direction, so newton_rel stays large and
            # we still raise. Mirrors the accepted-step step_rtol exit below;
            # gated on step_rtol>0 so the legacy absolute-tol path (the pinned
            # suite) is bit-for-bit untouched.
            if step_rtol > 0.0:
                f_scale = float(np.max(np.abs(f))) or 1.0
                n_scale = float(np.max(np.abs(n_ph))) or 1.0
                newton_rel = max(
                    float(np.max(np.abs(delta_f))) / f_scale,
                    float(np.max(np.abs(delta_n))) / n_scale,
                )
                if newton_rel < step_rtol:
                    return f, n_ph
            raise RuntimeError(
                f"Coupled Newton line search failed at iteration {iteration}. "
                f"max |residual| = {norm:.2e}"
            )

        # Scale-invariant convergence: the refining step barely moved the state.
        if step_rtol > 0.0 and rel_step < step_rtol:
            return f, n_ph

    raise RuntimeError(
        f"Coupled Newton did not converge in {max_iter} iterations. "
        f"Final max |residual| = {last_norm:.2e}"
    )
