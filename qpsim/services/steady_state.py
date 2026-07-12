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

Sentinel trap: ``None`` and ``0.0`` are OPPOSITE limits. ``None`` pins
phonons at the bath (τ_l → 0); the float ``0.0`` enters the Picard path,
where :func:`qpsim.phonon_models.ph0_local.phonon_steady_state` treats it
as the no-substrate-coupling sentinel (τ_l → ∞ limit of the escape term).
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.phonon_models.ph0_local import phonon_steady_state
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.anderson import anderson_extrapolate
from qpsim.solvers.newton_steady_state import newton_solve_f

# Picard convergence floors the per-bin relative-change denominator at this
# fraction of the peak n_ph occupation, so near-zero bins — whose iterate-to-
# iterate change is dominated by the inner Newton's ~tol float noise (~1e-11) —
# cannot pin the convergence metric. See :func:`_picard_max_rel_change`.
_PICARD_DENOM_FLOOR_FRAC = 1e-3


def _picard_max_rel_change(n_ph: np.ndarray, n_ph_new: np.ndarray) -> float:
    """Largest per-bin relative change of n_ph — the finite-τ_l Picard metric.

    The denominator is the per-bin ``max(|n_ph|, |n_ph_new|)`` floored at
    ``_PICARD_DENOM_FLOOR_FRAC`` of the *peak* occupation (not at ``picard_tol``).
    Rationale: the inner Newton converges ``f`` only to ``~newton_tol``, so each
    ``n_ph`` carries a small absolute float-noise floor (~1e-11 for Fischer
    Fig. 7). A near-zero sub-gap bin (``n_ph ~ 1e-8..1e-6``) then shows a
    ``|Δn| ~ 1e-11`` jitter that, divided by its own tiny occupation, looks like
    a ~1e-5 relative change and never falls below ``picard_tol`` — stalling an
    otherwise-converged solve until ``max_picard_iter`` (hit by Fig. 7 at
    P_read=-64 dBm, T_B=0.10 K). Flooring at ``frac * peak`` lets negligible bins
    drop out while bins with meaningful occupation (``scale >> frac * peak``)
    keep their true relative tolerance, so the physical observable stays tightly
    converged.

    Equivalent to an atol+rtol test ``|Δn_i| <= picard_tol * (scale_i + atol)``
    with ``atol = frac * peak``; note this absolute floor scales *with*
    picard_tol, so a caller driving ``picard_tol`` far below the inner-Newton
    noise floor is not protected by this metric (use the coupled-Newton path,
    which has its own residual-norm criterion, for tight tolerances).

    Returns 0.0 when every occupation is zero (trivially converged).
    """
    fp_change = np.abs(n_ph_new - n_ph)
    scale = np.maximum(np.abs(n_ph), np.abs(n_ph_new))
    peak = float(np.max(scale))
    if peak <= 0.0:
        return 0.0
    denom = scale + _PICARD_DENOM_FLOOR_FRAC * peak
    return float(np.max(fp_change / denom))


def solve_steady_state(
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    external_flux: ExternalFlux | None = None,
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
    K_s0_phonon_side, K_r0_phonon_side
        **Opt-in** phonon-side scattering and recombination/pair-breaking
        kernels (build via
        :func:`qpsim.collisions.phonon.build_scattering_kernel_phonon_side` and
        :func:`qpsim.collisions.phonon.build_recombination_kernel_phonon_side`).
        Forwarded to :func:`qpsim.phonon_models.ph0_local.phonon_steady_state`
        on the finite-τ_l Picard path; when supplied, the phonon
        sub-step uses the F&C 2023 Eq. 12 prefactors instead of the
        QP-side kernels. Ignored on the thermal-phonon path
        (``phonon_escape_time is None``). ``None`` (default) preserves
        legacy behavior bit-for-bit.
    T_bath
        Phonon bath temperature in K.
    photon_params
        ``{"omega_0", "n_bar", "c_phot"}`` for sub-gap channel.
    pb_photon_params
        ``{"omega_PB", "n_bar_PB", "c_phot_PB"}`` for PB channel.
    external_flux
        Optional :class:`qpsim.devices.ExternalFlux` boundary
        source/sink. Forwards to the Newton inner solve.
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

    if external_flux is not None:
        external_flux._validate_for_NE(NE)

    # Mirror the backend-level Picard-stability guard: unaccelerated
    # Picard (anderson_depth=0) is brittle when external_flux is
    # non-zero, the f-perturbation feeds through the phonon-emission
    # cycle and Picard oscillates without converging. Direct service
    # callers get the same explicit routing hint as backend callers.
    if (
        external_flux is not None
        and phonon_escape_time is not None
        and anderson_depth == 0
    ):
        ef_max = float(
            np.max(external_flux.gain) + np.max(external_flux.loss_rate)
        )
        if ef_max > 0.0:
            raise ValueError(
                "Unaccelerated Picard (anderson_depth=0) is unstable when "
                "external_flux is non-zero on the finite-τ_l path — the "
                "f-perturbation feeds through the phonon-emission cycle "
                "and Picard oscillates. Pass anderson_depth >= 1 to "
                "stabilize, OR call coupled_newton_solve directly, OR "
                "use phonon_escape_time=None for the τ_l → 0 limit. "
                "Pre-existing Picard stability concern; not a kwarg-"
                "threading bug."
            )

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
            external_flux=external_flux,
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
    f_physical: np.ndarray | None = None

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
            external_flux=external_flux,
            tol=tol, max_iter=max_iter,
        )

        # Step 3: n_ph steady state from converged f.
        n_ph_new = phonon_steady_state(
            f, ctx, K_s0, K_r0,
            omega_bins, omega_idx_diff, omega_idx_sum, diff_sign,
            T_bath, phonon_escape_time,
            K_s0_phonon_side=K_s0_phonon_side,
            K_r0_phonon_side=K_r0_phonon_side,
        )

        # Track branch state.
        on_physical = True
        if use_anderson and x_qp_ref > 0:
            x_qp_now = float(np.sum(rho * f * dE_ctx))
            on_physical = x_qp_now >= 0.1 * x_qp_ref
            if on_physical:
                n_ph_physical = n_ph.copy()
                f_physical = f.copy()

        # Convergence on n_ph. The metric floors the per-bin relative-change
        # denominator at a fraction of the peak occupation so near-zero,
        # noise-dominated bins cannot pin it; see _picard_max_rel_change.
        max_rel_change = _picard_max_rel_change(n_ph, n_ph_new)

        if max_rel_change < picard_tol:
            if use_anderson and x_qp_ref > 0 and not on_physical:
                # Anderson accelerated into the thermal branch. Fall back to
                # the last known physical-branch (f, n_ph) and finish on plain
                # Picard, which cannot jump branches. Disabling Anderson bounds
                # this to a single retry: x_qp_ref is only the *initial guess*
                # x_qp, so a genuinely drained fixed point (true x_qp < 0.1x a
                # hot guess, e.g. under an external-flux drain) would otherwise
                # re-trip this guard on every converged iterate and livelock to
                # max_picard_iter -> spurious "did not converge". Also reset f
                # (not just n_ph) so a real collapse can actually recover.
                n_ph = (
                    n_ph_physical
                    if n_ph_physical is not None
                    else thermal_phonon_occupation(omega_bins, T_bath)
                )
                if f_physical is not None:
                    f = f_physical
                use_anderson = False
                X_hist.clear()
                G_hist.clear()
                continue
            if phonon_out is not None:
                phonon_out["n_ph"] = n_ph
                phonon_out["omega_bins"] = omega_bins
            return f

        # Step 4: next Picard iterate (optionally Anderson-accelerated).
        # Phonon occupations are non-negative — pass clip_non_negative=True
        # so Anderson doesn't wander into unphysical territory.
        n_ph_mixed = (1.0 - picard_mixing) * n_ph + picard_mixing * n_ph_new
        if use_anderson:
            n_ph_aa = anderson_extrapolate(
                n_ph, n_ph_mixed, X_hist, G_hist, anderson_depth,
                clip_non_negative=True,
            )
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


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        return np.where(np.asarray(E) > 0, 0.0, 1.0)
    kT = _KB_UEV_PER_K * T
    exponent = np.minimum(np.asarray(E) / kT, 500.0)
    return 1.0 / (np.exp(exponent) + 1.0)
