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

This remains a local nonlinear solve: strong-bottleneck branches may require
parameter continuation to keep the initial state inside the desired basin.
The solver fails closed when its scale-aware convergence certificate is not
met; it does not itself choose among multiple branches.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from qpsim.collisions.phonon import (
    compute_phonon_source_sink,
    phonon_collision_jacobian_nph,
    phonon_occupation_matrices_from_state,
    phonon_source_sink_jacobian_f,
)
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import _gain_loss_sum, _jacobian_analytical

if TYPE_CHECKING:
    # Type-annotation-only import; kept out of runtime to avoid the
    # solvers -> devices -> services -> steady_state -> newton_steady_state
    # import cycle (see newton_steady_state.py). PEP 563 makes annotations
    # strings, so ExternalFlux is never needed at runtime here.
    from qpsim.devices.external_flux import ExternalFlux


class CoupledNewtonLineSearchError(RuntimeError):
    """A coupled-Newton line search could not reduce the residual.

    The structured residual norm lets callers distinguish a roundoff-level
    polish stall from singular-Jacobian, non-finite, and ordinary
    non-convergence failures without parsing an error message.
    """

    def __init__(self, *, iteration: int, residual_norm: float) -> None:
        self.iteration = int(iteration)
        self.residual_norm = float(residual_norm)
        super().__init__(
            f"Coupled Newton line search failed at iteration {self.iteration}. "
            f"max |residual| = {self.residual_norm:.2e}"
        )


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
    tau_l: float,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    external_flux: ExternalFlux | None = None,
    tol: float = 1e-10,
    step_rtol: float = 1e-8,
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
        Finite positive phonon bath-escape time (ns). The closed-phonon
        ``tau_l = 0`` limit has an unconstrained conserved-energy mode, so a
        root in ``(f, n_ph)`` is not unique and coupled Newton rejects it.
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
        ``max(|R_f|, |R_ph|)``. Used as the early-exit test only when
        ``step_rtol == 0`` (legacy opt-out), and for diagnostics otherwise.
    step_rtol
        Scale-invariant convergence tolerance on the relative Newton step
        ``max(‖Δf‖∞/‖f‖∞, ‖Δn‖∞/‖n‖∞)``. The default ``1e-8`` pairs this with a
        normwise gain/loss balance certificate. Set ``0.0`` only to reproduce
        the legacy absolute-residual behavior exactly. A positive value is
        essential when all physical
        amplitudes are tiny (e.g. f, n_ph ~ 1e-10 in the cold-bath
        strong-suppression regime that drives Fischer Fig. 6), where an
        absolute ``tol`` is unreliable: a warm continuation seed can sit just
        below ``tol`` and exit at iteration 0 with a stale, under-converged
        state. The relative-step test forces a refining step and is meaningful
        whether f ~ 1 or f ~ 1e-10. A small step is accepted only when the
        gain/loss terms also balance to this relative tolerance; step size
        alone is not a residual certificate.
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
        Converged, physically bounded QP and phonon distributions.

    Raises
    ------
    ValueError
        If either initial state has the wrong shape, is non-finite, or lies
        outside its physical bounds.
    RuntimeError
        On non-convergence, line-search failure, or singular Jacobian.
    """
    f_arr = np.asarray(f_init, dtype=float)
    if f_arr.ndim != 1 or f_arr.shape != ctx.E.shape:
        raise ValueError(
            f"f_init must have shape {ctx.E.shape}; got {f_arr.shape}."
        )
    if not np.all(np.isfinite(f_arr)) or np.any((f_arr < 0.0) | (f_arr > 1.0)):
        raise ValueError("f_init must be finite and lie in [0, 1].")

    n_arr = np.asarray(n_ph_init, dtype=float)
    if n_arr.ndim != 1 or n_arr.shape != omega_bins.shape:
        raise ValueError(
            f"n_ph_init must have shape {omega_bins.shape}; got {n_arr.shape}."
        )
    if not np.all(np.isfinite(n_arr)) or np.any(n_arr < 0.0):
        raise ValueError("n_ph_init must be finite and non-negative.")
    if not np.isfinite(T_bath) or T_bath < 0.0:
        raise ValueError(f"T_bath must be finite and non-negative; got {T_bath}.")
    if not np.isfinite(tau_l) or tau_l <= 0.0:
        raise ValueError(
            "coupled Newton requires a finite positive tau_l; the tau_l = 0 "
            "closed-phonon residual has an unconstrained conserved-energy mode."
        )
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive; got {tol}.")
    if not np.isfinite(step_rtol) or step_rtol < 0.0:
        raise ValueError(
            f"step_rtol must be finite and non-negative; got {step_rtol}."
        )
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError(f"max_iter must be a positive integer; got {max_iter!r}.")
    if not np.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError(f"fd_step must be finite and positive; got {fd_step}.")
    if not np.isfinite(fd_floor) or fd_floor <= 0.0:
        raise ValueError(f"fd_floor must be finite and positive; got {fd_floor}.")

    NE = int(f_arr.size)
    N_omega = int(omega_bins.size)
    if external_flux is not None:
        external_flux._validate_for_NE(NE)
        external_flux._validate_gain_support(ctx.active_mask)
    active_f = ctx.active_mask
    n_active_f = int(np.count_nonzero(active_f))
    if n_active_f == 0:
        raise ValueError(
            "coupled Newton requires at least one quasiparticle energy row "
            "with non-zero finite-volume spectral capacity. An all-zero-DOS "
            "context has no f unknown to couple; solve its phonon bath with "
            "the phonon-only model instead."
        )

    f = f_arr.copy()
    n_ph = n_arr.copy()

    n_th = thermal_phonon_occupation(omega_bins, T_bath)
    inv_tau_l = 1.0 / tau_l

    def residual(
        f_state: np.ndarray, n_ph_state: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if (
            np.any(~np.isfinite(f_state))
            or np.any((f_state < 0.0) | (f_state > 1.0))
            or np.any(~np.isfinite(n_ph_state))
            or np.any(n_ph_state < 0.0)
        ):
            raise RuntimeError(
                "Coupled Newton produced a non-finite or non-physical trial state."
            )
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph_state, omega_idx_diff, omega_idx_sum, diff_sign,
        )
        gain, loss = _gain_loss_sum(
            f_state, ctx, K_s0, K_r0, T_bath,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        loss_term = loss * f_state
        R_f = gain - loss_term

        a_ph, b_ph = compute_phonon_source_sink(
            f_state, ctx, K_s0, K_r0,
            omega_idx_diff, omega_idx_sum, diff_sign, N_omega,
            K_s0_phonon_side=K_s0_phonon_side,
            K_r0_phonon_side=K_r0_phonon_side,
        )
        driven_phonon = b_ph * n_ph_state
        bath_phonon = (n_th - n_ph_state) * inv_tau_l
        R_ph = a_ph + driven_phonon + bath_phonon

        # Dimensionless backward error for each balance equation. This is the
        # residual certificate paired with ``step_rtol``: a tiny Newton step
        # is meaningful only when the opposing physical terms cancel to the
        # same relative accuracy. Without it, an oversized/ill-conditioned
        # Jacobian can produce a negligible step at an O(1) residual.
        qp_scale = np.abs(gain) + np.abs(loss_term)
        ph_scale = np.abs(a_ph) + np.abs(driven_phonon) + np.abs(bath_phonon)
        if (
            np.any(~np.isfinite(R_f))
            or np.any(~np.isfinite(R_ph))
            or np.any(~np.isfinite(qp_scale))
            or np.any(~np.isfinite(ph_scale))
        ):
            raise RuntimeError("Coupled Newton residual became non-finite.")
        # Normwise backward errors for the two equation blocks.  A pointwise
        # maximum is unusable on cold grids: high-energy tails contain terms
        # hundreds of orders below the resolved signal, where harmless
        # underflow gives a formal local ratio of one.  The L1 block ratio is
        # still scale invariant and remains exactly one for an unbalanced
        # near-vacuum branch (gain present, opposing loss absent), while not
        # allowing numerically empty tail bins to veto a physical root.
        qp_denominator = float(np.sum(qp_scale))
        ph_denominator = float(np.sum(ph_scale))
        qp_ratio = (
            float(np.sum(np.abs(R_f))) / qp_denominator
            if qp_denominator > 0.0
            else (0.0 if not np.any(R_f) else np.inf)
        )
        ph_ratio = (
            float(np.sum(np.abs(R_ph))) / ph_denominator
            if ph_denominator > 0.0
            else (0.0 if not np.any(R_ph) else np.inf)
        )
        balance_ratio = max(qp_ratio, ph_ratio)

        return R_f, R_ph, balance_ratio

    last_norm = np.inf
    for iteration in range(max_iter):
        R_f, R_ph, balance_ratio = residual(f, n_ph)
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
                R_f_pert, _, _ = residual(f, n_ph_pert)
                J_fn[:, k] = (R_f_pert - R_f) / h_n[k]

            h_f = np.maximum(fd_step * np.abs(f), fd_floor)
            J_nf = np.zeros((N_omega, NE))
            for j in np.flatnonzero(active_f):
                f_pert = f.copy()
                # Keep the finite-difference probe inside the public
                # occupation domain.  A forward probe at f[j] == 1 used to
                # step above one and trip residual()'s physical-state guard
                # before Newton could start.  Prefer the requested forward
                # step, switch to a signed backward step near the upper
                # bound, and cap unusually large user-supplied FD steps at
                # the available room.
                signed_step = (
                    h_f[j]
                    if f[j] + h_f[j] <= 1.0
                    else -min(h_f[j], f[j])
                )
                if signed_step == 0.0:
                    signed_step = min(h_f[j], 1.0 - f[j])
                f_pert[j] += signed_step
                _, R_ph_pert, _ = residual(f_pert, n_ph)
                J_nf[:, j] = (R_ph_pert - R_ph) / signed_step

        # Pure-BCS grids may retain rho == 0 rows as storage placeholders.
        # Their residual and Jacobian rows are identically zero, so including
        # them as Newton unknowns makes the monolithic matrix singular. Solve
        # only the represented f-subspace and leave placeholder occupations
        # unchanged, matching newton_solve_f's active-set contract.
        J = np.block(
            [
                [J_ff[np.ix_(active_f, active_f)], J_fn[active_f, :]],
                [J_nf[:, active_f], J_nn],
            ]
        )
        R = np.concatenate([R_f[active_f], R_ph])

        try:
            delta = np.linalg.solve(J, -R)
        except np.linalg.LinAlgError as err:
            raise RuntimeError(
                f"Coupled Newton Jacobian singular at iteration {iteration}."
            ) from err

        delta_f = np.zeros(NE)
        delta_f[active_f] = delta[:n_active_f]
        delta_n = delta[n_active_f:]

        # Backtracking line search on the combined residual norm.
        alpha = 1.0
        accepted = False
        rel_step = np.inf
        balance_ratio_t = np.inf
        for _ in range(20):
            f_trial = np.clip(f + alpha * delta_f, 0.0, 1.0)
            n_trial = np.maximum(n_ph + alpha * delta_n, 0.0)
            R_f_t, R_ph_t, balance_ratio_t = residual(f_trial, n_trial)
            norm_t = max(
                float(np.max(np.abs(R_f_t))), float(np.max(np.abs(R_ph_t)))
            )
            if norm_t < norm:
                # Relative Newton-step size for the scale-invariant convergence
                # test; measured against the state magnitude so it is meaningful
                # whether f ~ 1 or f ~ 1e-10. ``or 1.0`` guards a zero state.
                f_scale = float(np.max(np.abs(f[active_f]))) or 1.0
                n_scale = float(np.max(np.abs(n_ph))) or 1.0
                rel_step = max(
                    float(np.max(np.abs(f_trial[active_f] - f[active_f])))
                    / f_scale,
                    float(np.max(np.abs(n_trial - n_ph))) / n_scale,
                )
                f, n_ph = f_trial, n_trial
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Preserve the legacy absolute-only escape solely when the caller
            # explicitly selected it.  Previously this unconditional branch
            # defeated ``step_rtol``: a line search could walk into a
            # near-vacuum state with a tiny dimensional residual but O(1)
            # gain/loss imbalance, then return it as converged anyway.
            if step_rtol <= 0.0 and norm < tol:
                return f, n_ph
            # Scale-invariant fallback: when the line search cannot reduce the
            # residual further, require both a negligible Newton step and a
            # small dimensionless gain/loss balance error. In the tiny-amplitude
            # strong-drive regime (f, n_ph ~ 1e-10) the residual floors at
            # ~1e-10 — above an absolute tol like 1e-12 — so without this the
            # solver spuriously fails points that are in fact converged (the
            # Fischer Fig. 6 transition NaNs). The balance certificate prevents
            # an ill-conditioned Jacobian from passing merely because it
            # produces a tiny step. Mirrors the accepted-step step_rtol exit
            # below; gated on step_rtol>0 so the legacy absolute-tol path (the
            # pinned suite) is bit-for-bit untouched.
            if step_rtol > 0.0:
                f_scale = float(np.max(np.abs(f[active_f]))) or 1.0
                n_scale = float(np.max(np.abs(n_ph))) or 1.0
                newton_rel = max(
                    float(np.max(np.abs(delta_f[active_f]))) / f_scale,
                    float(np.max(np.abs(delta_n))) / n_scale,
                )
                if newton_rel < step_rtol and balance_ratio < step_rtol:
                    return f, n_ph
            raise CoupledNewtonLineSearchError(
                iteration=iteration,
                residual_norm=norm,
            )

        # Scale-invariant convergence: the refining step barely moved the state.
        if (
            step_rtol > 0.0
            and rel_step < step_rtol
            and balance_ratio_t < step_rtol
        ):
            return f, n_ph

    raise RuntimeError(
        f"Coupled Newton did not converge in {max_iter} iterations. "
        f"Final max |residual| = {last_norm:.2e}"
    )
