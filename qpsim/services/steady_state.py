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
  A branch-collapse guard resets the coupled state to the last known
  physical-branch configuration as soon as an Anderson iterate lands on the
  thermal branch, then finishes with plain Picard.

Sentinel trap: ``None`` and ``0.0`` are OPPOSITE limits. ``None`` pins
phonons at the bath (τ_l → 0); the float ``0.0`` enters the Picard path,
where :func:`qpsim.phonon_models.ph0_local.phonon_steady_state` treats it
as the no-substrate-coupling sentinel (τ_l → ∞ limit of the escape term).
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    phonon_occupation_matrices_from_state,
)
from qpsim.devices.external_flux import ExternalFlux
from qpsim.phonon_models.ph0_local import (
    phonon_balance_diagnostics,
    phonon_steady_state,
)
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.solvers.anderson import AndersonAccelerationError, anderson_extrapolate
from qpsim.solvers.newton_steady_state import newton_solve_f


def _picard_convergence_ratio(
    n_ph: np.ndarray,
    n_ph_new: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> float:
    """Combined local and normwise Picard fixed-point convergence ratio.

    A phonon bin is converged when

    ``|n_new - n| <= atol + rtol * max(|n|, |n_new|)``.

    ``atol`` is an explicit tolerance on dimensionless phonon occupation:
    changes at that absolute noise floor cannot pin the outer Picard loop. It
    is deliberately independent of the inner Newton collision-residual
    tolerance, and it is deliberately not scaled by the peak phonon occupation,
    which would let a large low-energy bin set the floor for the tiny above-gap
    pair-breaking bins (Fischer Fig. 5).

    It is, however, still an *absolute* floor: any bin with
    ``n < atol / rtol`` is tested absolutely, not against its own occupation.
    At the shipped defaults that threshold is 0.1, i.e. essentially every
    phonon bin of a driven Fischer solve, so this test on its own gives the
    ω ≥ 2Δ pair-breaking bins no relative control — measured on a finite-τ_l
    Fig. 5 solve it first certifies at an iterate whose per-bin relative change
    there is still ~1e-1.

    The per-bin test is in particular unsafe when every occupation is below the
    absolute floor: a globally unresolved update can then pass merely because
    each component is smaller than ``atol``.  A second, normwise relative
    fixed-point guard therefore requires

    ``sum(|n_new - n|) / sum(|n| + |n_new|) <= rtol``.

    That guard is insensitive to the overall occupation amplitude, but it is an
    L1 test dominated by the largest bins and does not constrain a sub-floor
    block either.  Per-bin physical control of the above-gap bins comes from
    the Ph0 balance certificate (``picard_balance_tol``), which is what
    actually binds such a solve; callers that want the change test itself to
    resolve that block override the default downward — ``picard_atol=0.0``
    (fig3), 1e-12 (fig5), 1e-13 (fig7), 1e-14 (fig6).  The returned value is
    the larger of the two ratios and is converged at ``<= 1``. Identical
    all-zero iterates return zero.
    """
    if not np.isfinite(rtol) or rtol <= 0.0:
        raise ValueError(f"rtol must be positive; got {rtol}.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError(f"atol must be non-negative; got {atol}.")

    n_ph = np.asarray(n_ph, dtype=float)
    n_ph_new = np.asarray(n_ph_new, dtype=float)
    if n_ph.shape != n_ph_new.shape:
        raise ValueError(
            "n_ph and n_ph_new must have the same shape; "
            f"got {n_ph.shape} and {n_ph_new.shape}."
        )
    if np.any(~np.isfinite(n_ph)) or np.any(~np.isfinite(n_ph_new)):
        raise ValueError("n_ph and n_ph_new must contain only finite values.")

    change = np.abs(n_ph_new - n_ph)
    scale = np.maximum(np.abs(n_ph), np.abs(n_ph_new))
    allowed = atol + rtol * scale
    ratio = np.zeros_like(change, dtype=float)
    np.divide(change, allowed, out=ratio, where=allowed > 0.0)
    if np.any((allowed == 0.0) & (change > 0.0)):
        local_ratio = float("inf")
    else:
        local_ratio = float(np.max(ratio))

    normwise_denominator = float(np.sum(np.abs(n_ph) + np.abs(n_ph_new)))
    if normwise_denominator > 0.0:
        normwise_ratio = (
            float(np.sum(change)) / normwise_denominator / rtol
        )
    else:
        normwise_ratio = 0.0 if not np.any(change) else float("inf")
    return max(local_ratio, normwise_ratio)


def _phonon_balance_backward_error(
    n_ph: np.ndarray,
    a_ph: np.ndarray,
    b_ph: np.ndarray,
    n_th: np.ndarray,
    *,
    tau_l: float,
) -> float:
    """Representability-aware normwise error of the affine Ph0 balance.

    The shared diagnostic retains the raw direct-form error separately while
    this production gate certifies the nearest floating-point occupation.  A
    sub-ULP bath correction therefore cannot pin Picard, but a resolvable
    physical imbalance still fails on the usual L1 physical-term scale.
    """
    return phonon_balance_diagnostics(
        n_ph,
        a_ph,
        b_ph,
        n_th,
        tau_l=tau_l,
    ).certified_backward_error


def _resolve_picard_balance_tol(
    picard_tol: float,
    requested: float | None,
) -> float:
    """Resolve the physical-balance limit without overfitting roundoff.

    The affine phonon terms are independently accumulated O(NE^2) sums. Their
    normwise cancellation commonly floors around 1e-7 even after the fixed-
    point change is smaller, so the generic default keeps one decade of
    headroom at 1e-6. Paper validations can and do pass explicit limits.
    """
    if not np.isfinite(picard_tol) or picard_tol <= 0.0:
        raise ValueError(f"picard_tol must be finite and positive; got {picard_tol}.")
    if requested is None:
        return max(10.0 * picard_tol, 1e-6)
    resolved = float(requested)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            "picard_balance_tol must be finite and positive when provided; "
            f"got {requested}."
        )
    return resolved


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
    external_flux_is_conservative_transfer: bool = False,
    initial_guess: np.ndarray | None = None,
    initial_phonon_guess: np.ndarray | None = None,
    tol: float = 1e-14,
    newton_backward_error_tol: float = 1e-6,
    newton_number_polish_shape_tol: float | None = None,
    max_iter: int = 200,
    phonon_escape_time: float | None = None,
    max_picard_iter: int = 200,
    picard_tol: float = 1e-10,
    picard_atol: float = 1e-11,
    picard_balance_tol: float | None = None,
    picard_mixing: float = 0.3,
    anderson_depth: int = 0,
    phonon_enable_scattering: bool = True,
    phonon_enable_recombination: bool = True,
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
        Pure-BCS SpectralContext with current Δ. Dynes-broadened collision
        kernels are not implemented and are rejected when any collision
        channel is enabled.
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
    external_flux_is_conservative_transfer
        Whether that flux is conservative Device exchange. The flag changes
        only Newton's gain/loss normalization: prescribed sources use the
        default full turnover, while conservative exchange is excluded from
        the normalizer and remains fully present in the residual.
    initial_guess
        Starting ``f(E)``. Defaults to the Fermi-Dirac at ``T_bath``.
    initial_phonon_guess
        Starting ``n_ph(omega)`` for the finite-escape-time Picard loop.
        Defaults to the thermal Bose-Einstein occupation. The array must be
        one-dimensional and live on the QP-derived phonon frequency grid.
        It is rejected on the thermal-phonon shortcut, where ``n_ph`` is
        pinned by definition.
    tol, newton_backward_error_tol, newton_number_polish_shape_tol, max_iter
        Newton dimensional residual tolerance, scale-independent gain/loss
        backward-error limit, optional routing-only shape threshold for the
        scalar number-mode polish, and iteration cap. ``None`` keeps the
        solver's conservative default; it does not alter return certificates.
    phonon_escape_time
        ``None`` (default) → thermal phonon shortcut; phonons fixed at
        Bose-Einstein and no Picard loop.
        ``0.0`` → Picard solve with no bath coupling (``n_ph`` balances
        purely against the e-ph source).
        ``> 0`` → finite escape time τ_l in ns.
    max_picard_iter, picard_tol, picard_atol, picard_balance_tol,
    picard_mixing, anderson_depth
        Picard-loop controls. ``picard_tol`` is the per-bin relative tolerance;
        ``picard_atol`` is an absolute tolerance on the dimensionless phonon
        occupation and is independent of the inner Newton residual tolerance.
        A candidate fixed point must also satisfy a normwise physical phonon-
        balance certificate. ``picard_balance_tol=None`` derives its limit as
        ``max(10*picard_tol, 1e-6)``; pass a finite positive value to override.
    phonon_out
        Optional dict that receives ``"n_ph"`` and ``"omega_bins"`` on
        successful convergence (finite-τ_l path).

    Returns
    -------
    f
        Converged steady-state occupation, shape ``(NE,)``.
    """
    NE = ctx.E.size

    if not np.isfinite(T_bath) or T_bath < 0.0:
        raise ValueError(f"T_bath must be finite and non-negative; got {T_bath}.")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive; got {tol}.")
    if (
        not np.isfinite(newton_backward_error_tol)
        or newton_backward_error_tol <= 0.0
    ):
        raise ValueError(
            "newton_backward_error_tol must be finite and positive; "
            f"got {newton_backward_error_tol}."
        )
    if newton_number_polish_shape_tol is not None and (
        not np.isfinite(newton_number_polish_shape_tol)
        or newton_number_polish_shape_tol <= 0.0
    ):
        raise ValueError(
            "newton_number_polish_shape_tol must be finite and positive "
            f"when provided; got {newton_number_polish_shape_tol}."
        )
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter <= 0
    ):
        raise ValueError(f"max_iter must be a positive integer; got {max_iter}.")
    if phonon_escape_time is not None and (
        not np.isfinite(phonon_escape_time) or phonon_escape_time < 0.0
    ):
        raise ValueError(
            "phonon_escape_time must be finite and non-negative when provided; "
            f"got {phonon_escape_time}."
        )
    if initial_phonon_guess is not None and phonon_escape_time is None:
        raise ValueError(
            "initial_phonon_guess is only valid when phonon_escape_time is "
            "provided; thermal-phonon solves pin n_ph at Bose-Einstein."
        )
    if (
        isinstance(max_picard_iter, (bool, np.bool_))
        or not isinstance(max_picard_iter, (int, np.integer))
        or max_picard_iter <= 0
    ):
        raise ValueError(
            "max_picard_iter must be a positive integer; "
            f"got {max_picard_iter}."
        )
    if not np.isfinite(picard_tol) or picard_tol <= 0.0:
        raise ValueError(
            f"picard_tol must be finite and positive; got {picard_tol}."
        )
    if not np.isfinite(picard_atol) or picard_atol < 0.0:
        raise ValueError(
            f"picard_atol must be finite and non-negative; got {picard_atol}."
        )
    if picard_balance_tol is not None and (
        not np.isfinite(picard_balance_tol) or picard_balance_tol <= 0.0
    ):
        raise ValueError(
            "picard_balance_tol must be finite and positive when provided; "
            f"got {picard_balance_tol}."
        )
    if not np.isfinite(picard_mixing) or not 0.0 < picard_mixing <= 1.0:
        raise ValueError(
            "picard_mixing must be finite and lie in (0, 1]; "
            f"got {picard_mixing}."
        )
    if (
        isinstance(anderson_depth, (bool, np.bool_))
        or not isinstance(anderson_depth, (int, np.integer))
        or anderson_depth < 0
    ):
        raise ValueError(
            "anderson_depth must be a non-negative integer; "
            f"got {anderson_depth}."
        )

    collision_enabled = any(
        kernel is not None
        for kernel in (
            K_s0,
            K_r0,
            K_s0_phonon_side,
            K_r0_phonon_side,
        )
    ) or photon_params is not None or pb_photon_params is not None
    if ctx.dynes_gamma > 0.0 and collision_enabled:
        raise ValueError(
            "Dynes-broadened collision solves are not supported: the current "
            "electron-phonon and photon coherence kernels are ideal-BCS while "
            "the density of states is broadened. Set dynes_gamma=0 until "
            "consistent broadened kernels are implemented."
        )

    if external_flux is not None:
        external_flux._validate_for_NE(NE)
        external_flux._validate_gain_support(ctx.active_mask)
    if not isinstance(external_flux_is_conservative_transfer, (bool, np.bool_)):
        raise ValueError(
            "external_flux_is_conservative_transfer must be boolean; got "
            f"{external_flux_is_conservative_transfer!r}."
        )
    external_flux_is_conservative_transfer = bool(
        external_flux_is_conservative_transfer
    )
    if external_flux_is_conservative_transfer and external_flux is None:
        raise ValueError(
            "external_flux_is_conservative_transfer=True requires an "
            "ExternalFlux."
        )

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
        initial_raw = np.asarray(initial_guess)
        if np.iscomplexobj(initial_raw):
            raise ValueError("initial_guess must be real-valued.")
        initial = np.asarray(initial_raw, dtype=float)
        if initial.ndim != 1 or initial.shape != (NE,):
            raise ValueError(
                f"initial_guess shape {initial.shape} must be one-dimensional "
                f"and match grid shape ({NE},)"
            )
        f = initial.copy()
    else:
        f = _fermi_dirac(ctx.E, T_bath)

    initial_phonon_arr: np.ndarray | None = None
    if initial_phonon_guess is not None:
        candidate_raw = np.asarray(initial_phonon_guess)
        if np.iscomplexobj(candidate_raw):
            raise ValueError("initial_phonon_guess must be real-valued.")
        candidate = np.asarray(candidate_raw, dtype=float)
        if candidate.ndim != 1:
            raise ValueError(
                "initial_phonon_guess must be one-dimensional; got "
                f"shape {candidate.shape}."
            )
        if np.any(~np.isfinite(candidate)) or np.any(candidate < 0.0):
            raise ValueError(
                "initial_phonon_guess must contain only finite non-negative "
                "occupations."
            )
        initial_phonon_arr = candidate.copy()

    active = ctx.active_mask
    if int(np.sum(active)) == 0:
        # Degenerate grid: every above-gap bin lies within the active margin of
        # the gap edge, so there is nothing to solve. Don't silently hand a
        # non-finite / out-of-range initial guess back as a "solution", and
        # honor the phonon_out contract so a finite-τ_l caller does not KeyError
        # on a missing "n_ph".
        if initial_phonon_arr is not None:
            expected_omega = build_phonon_frequency_map(ctx.E)[0]
            if initial_phonon_arr.shape != expected_omega.shape:
                raise ValueError(
                    "initial_phonon_guess must match the derived phonon grid "
                    f"shape {expected_omega.shape}; got "
                    f"{initial_phonon_arr.shape}."
                )
        if np.any(~np.isfinite(f)) or np.any(f < 0.0) or np.any(f > 1.0):
            raise ValueError(
                "solve_steady_state: no active energy bins (every above-gap bin is "
                "within the active margin of the gap edge) and the initial occupation "
                "is non-finite or outside [0, 1]; refusing to return it unchanged. "
                "Check the energy-grid extent and the initial guess."
            )
        if phonon_out is not None:
            omega_bins = build_phonon_frequency_map(ctx.E)[0]
            phonon_out["n_ph"] = thermal_phonon_occupation(omega_bins, T_bath)
            phonon_out["omega_bins"] = omega_bins
        return f

    # Thermal-phonon path (τ_l → 0). Just Newton.
    if phonon_escape_time is None:
        return newton_solve_f(
            ctx, f,
            K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, active=active,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            external_flux=external_flux,
            external_flux_is_conservative_transfer=(
                external_flux_is_conservative_transfer
            ),
            tol=tol,
            backward_error_tol=newton_backward_error_tol,
            number_polish_shape_tol=newton_number_polish_shape_tol,
            max_iter=max_iter,
        )

    # Finite-τ_l path: Picard over (f, n_ph).
    omega_bins, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(
        ctx.E
    )
    n_th = thermal_phonon_occupation(omega_bins, T_bath)
    if initial_phonon_arr is None:
        n_ph = n_th.copy()
    else:
        if initial_phonon_arr.shape != omega_bins.shape:
            raise ValueError(
                "initial_phonon_guess must be one-dimensional and match the "
                f"derived phonon grid shape {omega_bins.shape}; got "
                f"{initial_phonon_arr.shape}."
            )
        n_ph = initial_phonon_arr
    resolved_balance_tol = _resolve_picard_balance_tol(
        picard_tol,
        picard_balance_tol,
    )

    use_anderson = anderson_depth > 0
    anderson_iterate_in_use = False
    X_hist: list[np.ndarray] = []
    G_hist: list[np.ndarray] = []

    # Branch-collapse detection: if Anderson accelerates into the thermal
    # branch (x_qp drops far below the initial value), reset immediately to
    # the last matched physical (f, n_ph) snapshot and continue with plain
    # Picard. Waiting until the collapsed map also passed the outer gates
    # allowed a 200-iteration Anderson cycle to repeat until max_iter in the
    # Fischer Fig. 7 cold/high-drive corner.
    weights = ctx.cell_weights
    x_qp_ref = float(np.sum(weights * f)) if use_anderson else 0.0
    f_initial = f.copy()
    n_ph_physical: np.ndarray | None = None
    f_physical: np.ndarray | None = None

    convergence_ratio = float("inf")
    balance_error = float("inf")
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
            external_flux_is_conservative_transfer=(
                external_flux_is_conservative_transfer
            ),
            tol=tol,
            backward_error_tol=newton_backward_error_tol,
            number_polish_shape_tol=newton_number_polish_shape_tol,
            max_iter=max_iter,
        )

        # Test the branch before constructing G(n_ph). A collapsed inner root
        # can still leave the phonon occupation-space change tiny while its
        # physical balance is O(1) wrong; the former late-only guard was thus
        # unreachable on exactly the path it was meant to rescue. Disabling
        # Anderson makes this a one-shot fallback: if the genuine fixed point
        # is a drained state, plain Picard may still converge to and return it
        # without being reset again.
        if use_anderson:
            x_qp_now = float(np.sum(weights * f))
            if (
                anderson_iterate_in_use
                and x_qp_ref > 0.0
                and x_qp_now < 0.1 * x_qp_ref
            ):
                n_ph = (
                    n_ph_physical.copy()
                    if n_ph_physical is not None
                    else n_th.copy()
                )
                f = (
                    f_physical.copy()
                    if f_physical is not None
                    else f_initial.copy()
                )
                use_anderson = False
                anderson_iterate_in_use = False
                X_hist.clear()
                G_hist.clear()
                continue
            n_ph_physical = n_ph.copy()
            f_physical = f.copy()
            # Follow the last matched physical branch, not only the initial
            # seed. A driven solve may grow many decades from an exact/tiny
            # vacuum seed before a later Anderson extrapolation collapses.
            x_qp_ref = x_qp_now

        # Step 3: n_ph steady state from converged f.
        coefficients: dict[str, np.ndarray] = {}
        n_ph_new = phonon_steady_state(
            f, ctx, K_s0, K_r0,
            omega_bins, omega_idx_diff, omega_idx_sum, diff_sign,
            T_bath, phonon_escape_time,
            K_s0_phonon_side=K_s0_phonon_side,
            K_r0_phonon_side=K_r0_phonon_side,
            enable_scattering=phonon_enable_scattering,
            enable_recombination=phonon_enable_recombination,
            coefficients_out=coefficients,
        )

        # Convergence on n_ph. The explicit occupation-space absolute
        # allowance is independent of the inner Newton residual tolerance;
        # relative convergence remains local to each bin.
        convergence_ratio = _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=picard_tol, atol=picard_atol,
        )

        if convergence_ratio <= 1.0:
            balance_error = _phonon_balance_backward_error(
                n_ph,
                coefficients["a_ph"],
                coefficients["b_ph"],
                n_th,
                tau_l=phonon_escape_time,
            )
            if balance_error > resolved_balance_tol:
                mapped_balance_error = _phonon_balance_backward_error(
                    n_ph_new,
                    coefficients["a_ph"],
                    coefficients["b_ph"],
                    n_th,
                    tau_l=phonon_escape_time,
                )
                if mapped_balance_error <= resolved_balance_tol:
                    # ``n_ph_new`` is the nearest representable affine root for
                    # this converged f, while under-relaxed/Anderson ``n_ph``
                    # can remain several ULPs away after the ordinary Picard
                    # change metric has passed. Project to that exact map, then
                    # run one more inner Newton iteration so the returned f is
                    # certified against the same occupation. Returning the map
                    # immediately would leave a mismatched (f, n_ph) pair.
                    n_ph = n_ph_new
                    anderson_iterate_in_use = False
                    X_hist.clear()
                    G_hist.clear()
                    balance_error = mapped_balance_error
                    continue
        if convergence_ratio <= 1.0 and balance_error <= resolved_balance_tol:
            if phonon_out is not None:
                phonon_out["n_ph"] = n_ph
                phonon_out["omega_bins"] = omega_bins
            return f

        # Step 4: next Picard iterate (optionally Anderson-accelerated).
        # Phonon occupations are non-negative — pass clip_non_negative=True
        # so Anderson doesn't wander into unphysical territory.
        n_ph_mixed = (1.0 - picard_mixing) * n_ph + picard_mixing * n_ph_new
        if use_anderson:
            try:
                n_ph_aa = anderson_extrapolate(
                    n_ph, n_ph_mixed, X_hist, G_hist, anderson_depth,
                    clip_non_negative=True,
                )
            except AndersonAccelerationError as exc:
                # AndersonAccelerationError subclasses only Exception so a
                # caller can catch it and retry without acceleration -- and no
                # caller under qpsim/ did, so it escaped this function,
                # T3DiffusionBackend.steady_state and solve_device_steady_state
                # and aborted the whole run, losing every region already
                # converged. Acceleration is a convergence aid, not physics:
                # the exact recovery it asks for is the `n_ph_aa is None`
                # branch four lines down, which this route already implements.
                # It matters most on the only path that reaches it, since both
                # this function and the diffusion backend FORCE anderson_depth
                # >= 1 whenever external_flux is non-zero on finite tau_l.
                warnings.warn(
                    f"Anderson acceleration failed ({exc}); continuing with "
                    "plain Picard for the rest of this solve. The answer is "
                    "unaffected; convergence may take more iterations.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                use_anderson = False
                X_hist.clear()
                G_hist.clear()
                n_ph_aa = None
            X_hist.append(n_ph.copy())
            G_hist.append(n_ph_mixed.copy())
            if len(X_hist) > anderson_depth + 1:
                X_hist.pop(0)
                G_hist.pop(0)
            anderson_iterate_in_use = n_ph_aa is not None
            n_ph = n_ph_aa if n_ph_aa is not None else n_ph_mixed
        else:
            n_ph = n_ph_mixed

    raise RuntimeError(
        f"Picard iteration did not converge in {max_picard_iter} iterations. "
        "Final max |G(n_ph) − n_ph| / "
        f"(Picard atol + rtol × scale) = {convergence_ratio:.2e}; "
        "last candidate phonon balance backward error = "
        f"{balance_error:.2e} (limit {resolved_balance_tol:.2e})."
    )


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
        return np.where(np.asarray(E) > 0, 0.0, 1.0)
    return fermi_dirac_occupation(E, T)
