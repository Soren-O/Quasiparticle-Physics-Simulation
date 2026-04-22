"""Steady-state orchestrator: thermal-phonon Newton and finite-τ_l Picard paths.

Composes the solver primitives in :mod:`qpsim.solvers` with the
collision primitives in :mod:`qpsim.collisions` and the phonon
frequency mapping to produce a user-facing steady-state solve.

Two regimes:

* ``phonon_escape_time is None`` — thermal phonons (τ_l → 0 limit).
  Phonon occupation is held at the Bose-Einstein value and
  :func:`newton_solve_f` does the whole job.
* ``phonon_escape_time >= 0`` — finite τ_l. Runs a Picard outer loop
  over ``n_ph``, solving the inner Newton for ``f`` at each step and
  recomputing the steady-state ``n_ph`` from the Ph0 phonon-balance
  equation. Anderson acceleration is available via ``anderson_depth``.
  A branch-collapse guard resets the phonon state to the last known
  physical-branch configuration if the Anderson path converges to the
  thermal branch.

Porting note: ``_phonon_steady_state`` below is the Ph0 local-bath
phonon-balance formula. It will move to
``qpsim.phonon_models.ph0_local`` when that module is written in
Gate 2 task 11; until then it lives here next to its only caller.
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    compute_phonon_source_sink,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.anderson import anderson_extrapolate
from qpsim.solvers.newton_steady_state import newton_solve_f


def solve_steady_state(
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    initial_guess: np.ndarray | None = None,
    tol: float = 1e-14,
    max_iter: int = 200,
    phonon_escape_time: float | None = None,
    max_picard_iter: int = 200,
    picard_tol: float = 1e-10,
    picard_mixing: float = 0.3,
    anderson_depth: int = 0,
    phonon_out: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Solve for the steady-state occupation ``f(E)``.

    Finds ``f(E)`` satisfying ``I_coll[f] = 0`` on the energy grid
    using Newton iteration with an analytical Jacobian. When
    ``phonon_escape_time`` is not ``None``, a Picard outer loop on
    ``n_ph`` wraps the Newton inner solve for self-consistent
    ``(f, n_ph)``.

    Parameters
    ----------
    ctx
        SpectralContext with current Δ.
    K_s0, K_r0
        Base e-ph kernel matrices or ``None`` to disable the channel.
    T_bath
        Phonon bath temperature in K.
    photon_params
        ``{"omega_0", "n_bar", "c_phot"}`` for sub-gap channel.
    pb_photon_params
        ``{"omega_PB", "n_bar_PB", "c_phot_PB"}`` for PB channel.
    initial_guess
        Starting ``f(E)``. Defaults to the Fermi-Dirac at ``T_bath``.
    tol, max_iter
        Newton convergence tolerance and iteration cap.
    phonon_escape_time
        ``None`` (default) → thermal phonon shortcut; phonons fixed at
        Bose-Einstein and no Picard loop.
        ``0.0`` → Picard solve with no bath coupling (``n_ph`` balances
        purely against the e-ph source).
        ``> 0`` → finite escape time τ_l in ns.
    max_picard_iter, picard_tol, picard_mixing, anderson_depth
        Picard-loop controls.
    phonon_out
        Optional dict that receives ``"n_ph"`` and ``"omega_bins"`` on
        successful convergence (finite-τ_l path).

    Returns
    -------
    f
        Converged steady-state occupation, shape ``(NE,)``.
    """
    NE = ctx.E.size

    if initial_guess is not None:
        f = np.array(initial_guess, dtype=float).ravel()
        if f.shape != (NE,):
            raise ValueError(
                f"initial_guess shape {f.shape} does not match grid size {NE}"
            )
    else:
        f = _fermi_dirac(ctx.E, T_bath)

    active = ctx.active_mask
    if int(np.sum(active)) == 0:
        return f

    # Thermal-phonon path (τ_l → 0). Just Newton.
    if phonon_escape_time is None:
        return newton_solve_f(
            ctx, f,
            K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, active=active,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            tol=tol, max_iter=max_iter,
        )

    # Finite-τ_l path: Picard over (f, n_ph).
    omega_bins, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(
        ctx.E
    )
    n_ph = thermal_phonon_occupation(omega_bins, T_bath)

    use_anderson = anderson_depth > 0
    X_hist: list[np.ndarray] = []
    G_hist: list[np.ndarray] = []

    # Branch-collapse detection: if Anderson accelerates into the
    # thermal branch (x_qp drops far below the initial value), we reset
    # to the last known physical-branch phonon state and continue.
    rho, dE_ctx = ctx.rho, ctx.dE
    x_qp_ref = float(np.sum(rho * f * dE_ctx)) if use_anderson else 0.0
    n_ph_physical: np.ndarray | None = None

    max_rel_change = float("inf")
    for _ in range(max_picard_iter):
        # Step 1: N_p, N_emit, N_abs from current n_ph.
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph, omega_idx_diff, omega_idx_sum, diff_sign,
        )

        # Step 2: Newton for f at frozen n_ph.
        f = newton_solve_f(
            ctx, f,
            K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, active=active,
            N_p_override=N_p, N_emit_override=N_emit, N_abs_override=N_abs,
            photon_params=photon_params, pb_photon_params=pb_photon_params,
            tol=tol, max_iter=max_iter,
        )

        # Step 3: n_ph steady state from converged f.
        n_ph_new = _phonon_steady_state(
            f, ctx, K_s0, K_r0,
            omega_bins, omega_idx_diff, omega_idx_sum, diff_sign,
            T_bath, phonon_escape_time,
        )

        # Track branch state.
        on_physical = True
        if use_anderson and x_qp_ref > 0:
            x_qp_now = float(np.sum(rho * f * dE_ctx))
            on_physical = x_qp_now >= 0.1 * x_qp_ref
            if on_physical:
                n_ph_physical = n_ph.copy()

        # Convergence check on n_ph.
        fp_change = np.abs(n_ph_new - n_ph)
        fp_scale = np.maximum(np.abs(n_ph), np.abs(n_ph_new)) + picard_tol
        max_rel_change = float(np.max(fp_change / fp_scale))

        if max_rel_change < picard_tol:
            if use_anderson and x_qp_ref > 0 and not on_physical:
                # Collapsed to thermal; reset and retry without Anderson.
                n_ph = (
                    n_ph_physical
                    if n_ph_physical is not None
                    else thermal_phonon_occupation(omega_bins, T_bath)
                )
                X_hist.clear()
                G_hist.clear()
                continue
            if phonon_out is not None:
                phonon_out["n_ph"] = n_ph
                phonon_out["omega_bins"] = omega_bins
            return f

        # Step 4: next Picard iterate (optionally Anderson-accelerated).
        n_ph_mixed = (1.0 - picard_mixing) * n_ph + picard_mixing * n_ph_new
        if use_anderson:
            n_ph_aa = anderson_extrapolate(n_ph, n_ph_mixed, X_hist, G_hist, anderson_depth)
            X_hist.append(n_ph.copy())
            G_hist.append(n_ph_mixed.copy())
            if len(X_hist) > anderson_depth + 1:
                X_hist.pop(0)
                G_hist.pop(0)
            n_ph = n_ph_aa if n_ph_aa is not None else n_ph_mixed
        else:
            n_ph = n_ph_mixed

    raise RuntimeError(
        f"Picard iteration did not converge in {max_picard_iter} iterations. "
        f"Final max |G(n_ph) − n_ph| / n_ph = {max_rel_change:.2e}"
    )


def _phonon_steady_state(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_bins: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    T_bath: float,
    phonon_escape_time: float,
) -> np.ndarray:
    """Solve for ``n_ph`` at phonon steady state given ``f``.

    The Ph0 phonon equation at steady state is

        0 = a_ph[f] + b_ph[f] · n_ph + (n_th − n_ph) / τ_l,

    where ``(a_ph, b_ph)`` are the affine coefficients from the e-ph
    source-sink on the QP distribution and ``τ_l`` is the bath-escape
    time.

    * ``τ_l = 0`` (no substrate coupling) collapses the equation to
      ``n_ph = −a_ph / b_ph``.
    * ``τ_l > 0`` gives ``n_ph = (a_ph + n_th/τ_l) / (1/τ_l − b_ph)``.

    Negative or numerically indeterminate entries are clipped to zero.

    This helper will move to :mod:`qpsim.phonon_models.ph0_local` when
    that module is written in Gate 2 task 11.
    """
    n_omega = len(omega_bins)
    a_ph, b_ph = compute_phonon_source_sink(
        f, ctx, K_s0, K_r0,
        omega_idx_diff, omega_idx_sum, diff_sign, n_omega,
    )

    if phonon_escape_time == 0.0:
        denom = b_ph.copy()
        safe = np.abs(denom) > 1e-30
        n_ph = np.zeros(n_omega)
        n_ph[safe] = -a_ph[safe] / denom[safe]
    else:
        inv_tau_l = 1.0 / phonon_escape_time
        n_th = thermal_phonon_occupation(omega_bins, T_bath)
        denom = inv_tau_l - b_ph
        safe = np.abs(denom) > 1e-30
        n_ph = np.zeros(n_omega)
        n_ph[safe] = (a_ph[safe] + inv_tau_l * n_th[safe]) / denom[safe]

    return np.maximum(n_ph, 0.0)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        return np.where(np.asarray(E) > 0, 0.0, 1.0)
    kT = _KB_UEV_PER_K * T
    exponent = np.minimum(np.asarray(E) / kT, 500.0)
    return 1.0 / (np.exp(exponent) + 1.0)
