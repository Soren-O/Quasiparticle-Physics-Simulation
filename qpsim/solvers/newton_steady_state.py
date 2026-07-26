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

from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.optimize import brentq

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions.pair_breaking_photon import (
    pair_breaking_photon_collision_components,
    pair_breaking_photon_collision_rates,
)
from qpsim.collisions.pair_breaking_photon import (
    pair_channel_open as _pair_channel_open,
)
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


# Stop the scalar amplitude root once its independently evaluated number
# certificate is already at the binary64 compatibility floor.  The public
# tolerance may be much looser (1e-6 by default); stopping there moved the
# resolved 20 mK thermal tail by about 4.9e-7 in a 2026-07-25 A/B test.
# Sixty-four eps preserves the historical O(1e-14) cold-tail agreement while
# avoiding several further Brent evaluations whose only effect is to polish
# the log-amplitude root itself to machine epsilon.
_NUMBER_POLISH_CERTIFICATE_STOP_LIMIT = 64.0 * np.finfo(float).eps


def _real_scalar_control(name: str, value: Any) -> float:
    """Normalize one real scalar control without accepting coercion traps."""
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or np.asarray(value).ndim != 0
    ):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {value!r}."
        ) from exc
    return normalized


def _resolve_number_polish_shape_tol(
    backward_error_tol: float,
    requested: object | None,
) -> float:
    """Resolve the routing-only shape gate for amplitude polishing.

    The conservative default preserves the historical high-dynamic-range
    thermal-tail behavior.  A validation with an independently demonstrated
    shape floor may opt into an explicit wider routing gate; this changes only
    when the scalar polish is attempted.  Its candidate still re-enters every
    ordinary residual and backward-error certificate.
    """
    if requested is None:
        return min(backward_error_tol, 1e-13)
    resolved = _real_scalar_control("number_polish_shape_tol", requested)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(
            "number_polish_shape_tol must be finite and positive when "
            f"provided; got {requested!r}."
        )
    return resolved


def thermal_collision_gain_loss(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Internal (collision/photon) ``(gain, loss_rate)`` at thermal phonons.

    Public wrapper for certification callers (the device outer loop's
    global slow-mode certificate) that need the internal turnover with
    NO external flux, matching exactly the assembly ``newton_solve_f``
    certifies against.
    """
    N_p = (
        _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        if K_s0 is not None
        else None
    )
    N_emit = N_abs = None
    if K_r0 is not None:
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
    return _gain_loss_sum(
        f, ctx, K_s0, K_r0, T_bath,
        photon_params, pb_photon_params,
        N_p, N_emit, N_abs,
        None,
    )


def number_changing_gain_loss(
    f: np.ndarray,
    ctx: SpectralContext,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    pb_photon_params: dict[str, float] | None = None,
    N_emit: np.ndarray | None = None,
    N_abs: np.ndarray | None = None,
    external_flux: ExternalFlux | None = None,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return only channels that change a region's QP number.

    The decomposition deliberately excludes e-ph scattering and sub-gap
    photon scattering.  It includes phonon recombination/pair breaking,
    the K-minus pair component of an above-gap photon channel, and a local
    external source/sink.  The latter changes the number of one region even
    when a Device-level weighted sum later proves that a junction only
    transfers QPs between regions.

    The boolean reports whether a number-changing channel was configured.
    It distinguishes a physically absent number mode from a configured
    channel whose entire float64 turnover underflowed to zero.
    """
    gain = np.zeros_like(f, dtype=float)
    loss_rate = np.zeros_like(f, dtype=float)
    configured = False

    if K_r0 is not None:
        gain_pair, loss_pair = phonon_collision_rates(
            f,
            ctx,
            None,
            K_r0,
            T_bath,
            N_emit_override=N_emit,
            N_abs_override=N_abs,
        )
        gain += gain_pair
        loss_rate += loss_pair
        # A present-but-zero kernel is physically the same as a disabled
        # channel.  Only supported target/partner pairs can contribute;
        # placeholder entries outside ``active_mask`` must not turn an
        # inert kernel into an "underflowed configured channel" failure.
        supported = ctx.active_mask
        supported_pairs = supported[:, None] & supported[None, :]
        configured = bool(
            np.any(np.asarray(K_r0, dtype=float)[supported_pairs] != 0.0)
        )

    if pb_photon_params is not None:
        (
            _gain_scattering,
            _loss_scattering,
            gain_pair,
            loss_pair,
        ) = pair_breaking_photon_collision_components(
            f,
            ctx,
            pb_photon_params["omega_PB"],
            pb_photon_params["n_bar_PB"],
            pb_photon_params["c_phot_PB"],
        )
        gain += gain_pair
        loss_rate += loss_pair
        if (
            pb_photon_params["c_phot_PB"] > 0.0
            and pb_photon_params["omega_PB"] > 0.0
        ):
            dE_scalar = uniform_grid_spacing(
                ctx.E,
                ctx.dE,
                "Pair-number certificate",
            )
            omega_snapped = round(
                pb_photon_params["omega_PB"] / dE_scalar
            ) * dE_scalar
            configured = configured or _pair_channel_open(
                omega_snapped, ctx.gap
            )

    if external_flux is not None:
        gain += external_flux.gain
        loss_rate += external_flux.loss_rate
        supported = ctx.active_mask
        configured = configured or bool(
            np.any(external_flux.gain[supported] != 0.0)
            or np.any(external_flux.loss_rate[supported] != 0.0)
        )

    unsupported = ~ctx.active_mask
    gain[unsupported] = 0.0
    loss_rate[unsupported] = 0.0
    return gain, loss_rate, configured


def _number_conserving_gain_loss(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    T_bath: float,
    *,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    N_p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble only channels that redistribute QPs across energy bins.

    This is the complement of :func:`number_changing_gain_loss` used to
    decide whether a cold distribution's *shape* is settled enough for a
    scalar total-number polish.  Assemble it directly rather than subtracting
    pair rates from the full collision sum: at ultralow temperature the two
    aggregates can span many decades, and ``full - pair`` would recover a
    small physical balance through avoidable floating-point cancellation.

    The aggregate intentionally combines all conserving channels before its
    backward error is measured.  A driven steady state may balance sub-gap or
    above-gap photon heating against e-ph cooling; requiring every channel to
    vanish separately would reject that valid cross-channel balance.
    """
    gain = np.zeros_like(f, dtype=float)
    loss_rate = np.zeros_like(f, dtype=float)

    if K_s0 is not None:
        gain_scattering, loss_scattering = phonon_collision_rates(
            f,
            ctx,
            K_s0,
            None,
            T_bath,
            N_p_override=N_p,
        )
        gain += gain_scattering
        loss_rate += loss_scattering

    if photon_params is not None:
        gain_photon, loss_photon = sub_gap_photon_collision_rates(
            f,
            ctx,
            photon_params["omega_0"],
            photon_params["n_bar"],
            photon_params["c_phot"],
        )
        gain += gain_photon
        loss_rate += loss_photon

    if pb_photon_params is not None:
        gain_pb_scattering, loss_pb_scattering, _gain_pair, _loss_pair = (
            pair_breaking_photon_collision_components(
                f,
                ctx,
                pb_photon_params["omega_PB"],
                pb_photon_params["n_bar_PB"],
                pb_photon_params["c_phot_PB"],
            )
        )
        gain += gain_pb_scattering
        loss_rate += loss_pb_scattering

    unsupported = ~ctx.active_mask
    gain[unsupported] = 0.0
    loss_rate[unsupported] = 0.0
    return gain, loss_rate


def _weighted_number_backward_error(
    gain: np.ndarray,
    loss_rate: np.ndarray,
    f: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
) -> float | None:
    """Scale-stable weighted number-balance backward error.

    Returns ``None`` when every configured turnover term is numerically
    zero.  Rates are normalized *before* multiplication and summation so a
    cold but representable pair channel is not lost merely because its
    dimensional scale is tiny.
    """
    gain_active = np.asarray(gain, dtype=float)[active]
    loss_term = np.asarray(loss_rate, dtype=float)[active] * np.asarray(
        f, dtype=float
    )[active]
    weights_active = np.asarray(weights, dtype=float)[active]
    rate_scale = float(
        max(
            np.max(np.abs(gain_active), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
        )
    )
    if rate_scale == 0.0:
        return None
    gain_scaled = weights_active * (gain_active / rate_scale)
    loss_scaled = weights_active * (loss_term / rate_scale)
    denominator = float(np.sum(np.abs(gain_scaled)) + np.sum(np.abs(loss_scaled)))
    if denominator == 0.0:
        return None
    numerator = abs(float(np.sum(gain_scaled - loss_scaled)))
    return numerator / denominator


def _signed_weighted_number_residual(
    gain: np.ndarray,
    loss_rate: np.ndarray,
    f: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
) -> float | None:
    """Signed, scale-stable total-number residual for root bracketing."""
    gain_active = np.asarray(gain, dtype=float)[active]
    loss_term = np.asarray(loss_rate, dtype=float)[active] * np.asarray(
        f, dtype=float
    )[active]
    weights_active = np.asarray(weights, dtype=float)[active]
    rate_scale = float(
        max(
            np.max(np.abs(gain_active), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
        )
    )
    if rate_scale == 0.0:
        return None
    return float(
        np.sum(
            weights_active
            * (gain_active / rate_scale - loss_term / rate_scale)
        )
    )


def _polish_number_mode_amplitude(
    f: np.ndarray,
    ctx: SpectralContext,
    K_r0: np.ndarray | None,
    T_bath: float,
    active: np.ndarray,
    gain_number: np.ndarray,
    loss_number: np.ndarray,
    *,
    backward_error_tol: float,
    pb_photon_params: dict[str, float] | None,
    N_emit: np.ndarray | None,
    N_abs: np.ndarray | None,
    N_abs_override: np.ndarray | None,
    external_flux: ExternalFlux | None,
) -> np.ndarray | None:
    """Polish the slow total-number mode by scaling the current shape.

    Once the full Newton residual is already quiet, its remaining error can
    lie almost entirely in the exponentially weaker pair-number mode.  The
    ordinary max-residual line search then stalls at its floating-point floor
    even though a tiny amplitude change balances that mode.  Along
    ``f_active -> c*f_active`` the phonon/PB pair residual is monotone (pair
    generation decreases and recombination increases), as is an affine local
    ``ExternalFlux`` balance.  Bracket that one-dimensional root without
    relaxing the caller's advertised backward-error tolerance.

    ``None`` means that the current shape cannot bracket a representable
    number root.  The caller then keeps the ordinary Newton path, which either
    finds another shape or fails loudly.
    """
    f_base = np.asarray(f, dtype=float).copy()
    certificate_stop_tol = min(
        backward_error_tol,
        _NUMBER_POLISH_CERTIFICATE_STOP_LIMIT,
    )

    def certificate_passes(error: float | None) -> bool:
        return error is not None and error <= certificate_stop_tol

    current_signed = _signed_weighted_number_residual(
        gain_number,
        loss_number,
        f_base,
        ctx.cell_weights,
        active,
    )
    if current_signed is None or current_signed == 0.0:
        return None

    max_f = float(np.max(f_base[active], initial=0.0))
    if max_f <= 0.0:
        return None
    log_max_scale = min(
        -float(np.log(max_f)),
        float(np.log(np.finfo(float).max)),
    )

    def candidate(scale: float) -> np.ndarray:
        out = f_base.copy()
        # ``scale`` is bracketed so the exact-real candidate is feasible,
        # but the exp/log round trip used by the upper bracket can round
        # ``scale * max(f)`` one ulp above 1.  Keep that representational
        # overshoot inside the solver's advertised occupation domain; every
        # physical residual/certificate is still recomputed on this bounded
        # candidate before it can be returned.
        out[active] = np.clip(scale * f_base[active], 0.0, 1.0)
        return out

    def evaluate(
        scale: float,
    ) -> tuple[float | None, float | None, np.ndarray]:
        trial = candidate(scale)
        gain_trial, loss_trial, _configured = number_changing_gain_loss(
            trial,
            ctx,
            K_r0,
            T_bath,
            pb_photon_params=pb_photon_params,
            N_emit=N_emit,
            N_abs=N_abs,
            external_flux=external_flux,
        )
        signed = _signed_weighted_number_residual(
            gain_trial,
            loss_trial,
            trial,
            ctx.cell_weights,
            active,
        )
        error = _weighted_number_backward_error(
            gain_trial,
            loss_trial,
            trial,
            ctx.cell_weights,
            active,
        )
        return signed, error, trial

    signed_lo: float | None
    signed_hi: float | None
    if current_signed < 0.0:
        zero_scale = 0.0
        signed_zero, error_zero, zero_trial = evaluate(zero_scale)
        if certificate_passes(error_zero):
            return zero_trial
        if signed_zero is None:
            zero_gain, _zero_loss, zero_configured = number_changing_gain_loss(
                zero_trial,
                ctx,
                K_r0,
                T_bath,
                pb_photon_params=pb_photon_params,
                N_emit=N_emit,
                N_abs=N_abs,
                external_flux=external_flux,
            )
            if zero_configured and _is_exact_absorbing_vacuum(
                zero_trial,
                zero_gain,
                active,
                T_bath=T_bath,
                K_r0=K_r0,
                N_abs=N_abs,
                N_abs_override=N_abs_override,
                pb_photon_params=pb_photon_params,
                ctx=ctx,
            ):
                # Zero turnover is normally uncertifiable, but a proven
                # loss-only vacuum is an exact endpoint root.  Return it to
                # the caller, which re-enters the full residual certificate.
                return zero_trial
        if signed_zero is None or signed_zero <= 0.0:
            return None

        # The number root may be tens or hundreds of decades below the
        # current amplitude.  Calling Brent directly on [0, 1] with the
        # smallest subnormal ``xtol`` then needs roughly one iteration per
        # binary exponent and can exhaust its iteration budget despite a
        # perfectly regular root (reproduced at 20 mK and in the Fig. 7
        # sweep).  Locate a *positive* lower endpoint geometrically, so the
        # final scalar solve can operate in log-amplitude space.
        hi = 1.0
        signed_hi = current_signed
        min_log_scale = float(np.log(np.nextafter(0.0, 1.0)))
        exponent = 1.0
        while True:
            log_probe = max(-exponent, min_log_scale)
            lo = float(np.exp(log_probe))
            signed_lo, error_lo, trial_lo = evaluate(lo)
            if certificate_passes(error_lo):
                return trial_lo
            if signed_lo is None:
                # The configured turnover vanished before a positive
                # binary64 bracket could be formed.  There is no
                # representable amplitude root for this polish to certify.
                return None
            if signed_lo == 0.0:
                if error_lo is not None and error_lo <= backward_error_tol:
                    return trial_lo
                return None
            if signed_lo > 0.0:
                break
            hi = lo
            signed_hi = signed_lo
            if log_probe == min_log_scale:
                return None
            exponent *= 2.0
    else:
        lo = 1.0
        signed_lo = current_signed
        # The positive root can likewise sit tens or hundreds of decades
        # above a near-vacuum seed.  Doubling the *linear* scale only 32 times
        # stranded the Fig. 7 cold recovery.  Double its logarithm instead,
        # reaching the full representable interval in O(log log(max_scale)).
        if log_max_scale <= 0.0:
            return None
        log_hi = min(float(np.log(2.0)), log_max_scale)
        while log_hi > 0.0:
            hi = float(np.exp(log_hi))
            signed_hi, error_hi, trial_hi = evaluate(hi)
            if certificate_passes(error_hi):
                return trial_hi
            if signed_hi is None:
                return None
            if signed_hi == 0.0:
                if error_hi is not None and error_hi <= backward_error_tol:
                    return trial_hi
                return None
            if signed_hi < 0.0:
                break
            lo = hi
            signed_lo = signed_hi
            if log_hi >= log_max_scale:
                return None
            log_hi = min(2.0 * log_hi, log_max_scale)
        if signed_hi is None or signed_hi > 0.0:
            return None

    if signed_lo is None or signed_hi is None or signed_lo * signed_hi > 0.0:
        return None

    class _CertifiedNumberTrialError(Exception):
        """Internal control flow carrying an already-certified trial."""

        def __init__(self, trial: np.ndarray) -> None:
            super().__init__("number certificate reached")
            self.trial = trial

    def signed_at_log_scale(log_scale: float) -> float:
        scale = float(np.exp(log_scale))
        signed, error, trial = evaluate(scale)
        if certificate_passes(error):
            raise _CertifiedNumberTrialError(trial)
        if signed is None:
            raise ValueError("number-changing turnover vanished inside bracket")
        return signed

    try:
        log_root = float(
            brentq(
                signed_at_log_scale,
                float(np.log(lo)),
                float(np.log(hi)),
                xtol=4.0 * np.finfo(float).eps,
                rtol=4.0 * np.finfo(float).eps,
                maxiter=128,
            )
        )
        root = float(np.exp(log_root))
    except _CertifiedNumberTrialError as certified:
        return certified.trial
    except (RuntimeError, ValueError):
        return None

    # Brent's root can lie between adjacent binary64 scales.  Check both
    # neighbours and keep the representable candidate with the best actual
    # number certificate.
    scales = (
        root,
        float(np.nextafter(root, lo)),
        float(np.nextafter(root, hi)),
    )
    best_error = float("inf")
    best_trial: np.ndarray | None = None
    for scale in scales:
        if not lo <= scale <= hi:
            continue
        _, error, trial = evaluate(scale)
        if error is not None and error < best_error:
            best_error = error
            best_trial = trial
    if best_trial is None or best_error > backward_error_tol:
        return None
    return best_trial


def _is_exact_absorbing_vacuum(
    f: np.ndarray,
    gain_number: np.ndarray,
    active: np.ndarray,
    *,
    T_bath: float,
    K_r0: np.ndarray | None,
    N_abs: np.ndarray | None,
    N_abs_override: np.ndarray | None,
    pb_photon_params: dict[str, float] | None,
    ctx: SpectralContext,
) -> bool:
    """Whether zero turnover is an exact absorbing vacuum root.

    A configured pair channel with no representable turnover is normally
    uncertifiable: at low but nonzero temperature its rates may merely have
    underflowed.  Vacuum is safe only when every configured channel is known
    to be loss-only: an implicit thermal pair bath requires exactly ``T=0``;
    an explicit zero-absorption override, zero-occupancy PB bath, and local
    loss-only external flux are absorbing at any temperature.
    """
    if np.any(np.asarray(f)[active] != 0.0):
        return False
    if np.any(np.asarray(gain_number)[active] != 0.0):
        return False
    if K_r0 is not None:
        supported = ctx.active_mask
        supported_pairs = supported[:, None] & supported[None, :]
        phonon_pair_configured = bool(
            np.any(np.asarray(K_r0, dtype=float)[supported_pairs] != 0.0)
        )
    else:
        phonon_pair_configured = False
    if phonon_pair_configured:
        if N_abs_override is None:
            # A finite-T implicit thermal absorption matrix may have merely
            # underflowed to zero; only T=0 proves the bath is loss-only.
            if T_bath != 0.0:
                return False
        elif N_abs is None or np.any(np.asarray(N_abs) != 0.0):
            return False
    if pb_photon_params is not None and (
        pb_photon_params["c_phot_PB"] > 0.0
        and pb_photon_params["omega_PB"] > 0.0
    ):
        dE_scalar = uniform_grid_spacing(
            ctx.E,
            ctx.dE,
            "Pair-number vacuum certificate",
        )
        omega_snapped = round(
            pb_photon_params["omega_PB"] / dE_scalar
        ) * dE_scalar
        if (
            pb_photon_params["c_phot_PB"] > 0.0
            and pb_photon_params["n_bar_PB"] > 0.0
            and _pair_channel_open(omega_snapped, ctx.gap)
        ):
            return False
    return True


def _row_scaled_newton_system(
    jacobian: np.ndarray,
    residual: np.ndarray,
    gain: np.ndarray,
    loss_term: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale each Newton equation by its own physical turnover.

    Cold kinetic systems routinely span more than one hundred decades from
    the gap edge to the high-energy tail.  Solving the raw dimensional system
    can therefore lose the weak rows even though multiplying an equation by a
    nonzero scalar cannot change the mathematical Newton step.  Normalize
    every nonzero row by ``max(|gain_i|, |loss_i f_i|)``; an exactly zero-
    turnover row is left unchanged because it has no local physical scale.

    Deliberately do not impose a global relative floor.  Such a floor restores
    the very cross-row scale disparity this transformation removes (the
    Fischer Fig. 7 cold point has legitimate row scales spanning ~1e142).
    """
    J = np.asarray(jacobian, dtype=float)
    R = np.asarray(residual, dtype=float)
    gain_arr = np.asarray(gain, dtype=float)
    loss_arr = np.asarray(loss_term, dtype=float)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError("jacobian must be a square matrix")
    if not (R.shape == gain_arr.shape == loss_arr.shape == (J.shape[0],)):
        raise ValueError("residual and turnover arrays must match jacobian rows")

    scale = np.maximum(np.abs(gain_arr), np.abs(loss_arr))
    scaled_J = J.copy()
    scaled_R = R.copy()
    nonzero = scale > 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_J[nonzero] /= scale[nonzero, None]
        scaled_R[nonzero] /= scale[nonzero]
    if np.any(~np.isfinite(scaled_J)) or np.any(~np.isfinite(scaled_R)):
        raise RuntimeError(
            "Per-row Newton turnover scaling exceeded the float64 range."
        )
    return scaled_J, scaled_R


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
    external_flux_is_conservative_transfer: bool = False,
    tol: float = 1e-14,
    backward_error_tol: float = 1e-6,
    number_polish_shape_tol: float | None = None,
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
        Defaults to ``ctx.active_mask``. A nonempty mask must equal the
        complete represented-domain mask: proper subsets omit transition
        fluxes across their artificial boundary and cannot currently carry
        a valid total-number certificate. An all-false mask remains an
        explicit no-op.
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
    external_flux_is_conservative_transfer
        Set only when ``external_flux`` is a frozen conservative exchange
        term from a Device component. Such exchange remains in the residual
        but is excluded from the gain/loss certificate's normalization so a
        large balanced transfer cannot hide unequilibrated internal channels.
        The default ``False`` is required for prescribed sources and sinks,
        whose turnover legitimately sets the scale of their steady state.
    tol
        Absolute convergence tolerance on the dimensional residual.
    backward_error_tol
        Scale-independent L1 gain/loss backward-error limit. Both this physical
        certificate and the dimensional ``tol`` must pass on every return path.
    number_polish_shape_tol
        Optional routing threshold for the independently assembled
        number-conserving shape error before trying the scalar total-number
        amplitude polish. ``None`` uses the conservative
        ``min(backward_error_tol, 1e-13)`` default. A wider explicit value is
        suitable only when a validation has independently bounded its shape
        error; it cannot weaken any return certificate.
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
    # Every state/kernel/occupation array in this solver is real-domain.
    # Check before casting: ``np.asarray(complex_array, dtype=float)`` drops
    # the imaginary part with only a warning, and can thereby erase an
    # imaginary NaN while leaving an apparently finite real input.
    real_array_inputs = (
        ("initial occupation f", f),
        ("K_s0", K_s0),
        ("K_r0", K_r0),
        ("N_p_override", N_p_override),
        ("N_emit_override", N_emit_override),
        ("N_abs_override", N_abs_override),
    )
    for name, value in real_array_inputs:
        if value is not None and np.iscomplexobj(value):
            raise ValueError(f"{name} must be real-valued")

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
    tol = _real_scalar_control("tol", tol)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError(f"tol must be finite and positive; got {tol}.")
    T_bath = _real_scalar_control("T_bath", T_bath)
    if not np.isfinite(T_bath) or T_bath < 0.0:
        raise ValueError(f"T_bath must be finite and non-negative; got {T_bath}.")
    backward_error_tol = _real_scalar_control(
        "backward_error_tol", backward_error_tol
    )
    if not np.isfinite(backward_error_tol) or backward_error_tol <= 0.0:
        raise ValueError(
            "backward_error_tol must be finite and positive; "
            f"got {backward_error_tol}."
        )
    resolved_number_polish_shape_tol = _resolve_number_polish_shape_tol(
        backward_error_tol,
        number_polish_shape_tol,
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
        unsupported = active & ~ctx.active_mask
        if np.any(unsupported):
            indices = np.flatnonzero(unsupported)
            preview = ", ".join(str(int(i)) for i in indices[:5])
            suffix = "" if indices.size <= 5 else ", ..."
            raise ValueError(
                "active may only select bins with represented spectral "
                "capacity; it enables "
                f"{indices.size} unsupported bin(s) (indices "
                f"{preview}{suffix})."
            )
    n_active = int(np.sum(active))
    if n_active == 0:
        return f_cur
    # The pair-number certificate is a conservation statement for the full
    # represented QP domain.  An arbitrary caller-supplied proper subset is
    # open: scattering/photon transitions across its boundary are number
    # sources and sinks for that subset.  Silently skipping the certificate
    # made cold partial solves amplitude-blind (a 0.5*f_FD seed could return
    # unchanged).  Until those boundary fluxes are explicitly decomposed,
    # refuse the mathematically incomplete solve.
    if not np.array_equal(active, ctx.active_mask):
        omitted = ctx.active_mask & ~active
        indices = np.flatnonzero(omitted)
        preview = ", ".join(str(int(i)) for i in indices[:5])
        suffix = "" if indices.size <= 5 else ", ..."
        raise ValueError(
            "a nonempty active mask must equal ctx.active_mask; proper "
            "subsets omit collision/photon flux across the artificial "
            f"boundary ({indices.size} represented bin(s) omitted: "
            f"{preview}{suffix}) and cannot be number-certified"
        )
    certify_total_number = True

    # Default to thermal occupations when no overrides given and the
    # corresponding kernel is in play.
    N_p = N_p_override
    N_emit = N_emit_override
    N_abs = N_abs_override
    if N_p is None and K_s0 is not None:
        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
    if N_emit is None and K_r0 is not None:
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)

    if certify_total_number:
        # Loss-only pair systems have an exact absorbing endpoint at f=0.
        # Newton's multiplicative pair loss approaches that boundary only
        # asymptotically, and a Jacobian can become singular before it lands
        # on the representable endpoint.  Prove the vacuum semantics first,
        # then require the complete pointwise residual to be exactly zero;
        # this does not excuse finite-T absorption that merely underflowed.
        vacuum = f_cur.copy()
        vacuum[active] = 0.0
        gain_vacuum, _loss_vacuum, vacuum_channel_configured = (
            number_changing_gain_loss(
                vacuum,
                ctx,
                K_r0,
                T_bath,
                pb_photon_params=pb_photon_params,
                N_emit=N_emit,
                N_abs=N_abs,
                external_flux=external_flux,
            )
        )
        if vacuum_channel_configured and _is_exact_absorbing_vacuum(
            vacuum,
            gain_vacuum,
            active,
            T_bath=T_bath,
            K_r0=K_r0,
            N_abs=N_abs,
            N_abs_override=N_abs_override,
            pb_photon_params=pb_photon_params,
            ctx=ctx,
        ):
            vacuum_residual = _residual(
                vacuum,
                ctx,
                K_s0,
                K_r0,
                T_bath,
                photon_params,
                pb_photon_params,
                N_p,
                N_emit,
                N_abs,
                external_flux,
            )
            if np.all(vacuum_residual[active] == 0.0):
                return vacuum

    max_residual = np.inf
    rate_scale = 0.0
    backward_error = float("inf")
    number_error: float | None = None
    for iteration in range(max_iter):
        gain, loss_rate = _gain_loss_sum(
            f_cur, ctx, K_s0, K_r0, T_bath,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        # Internal (collision/photon-channel) turnover for CERTIFICATION:
        # a large but exactly balanced external-flux exchange must not
        # inflate the acceptance denominator, or a state the internal
        # channels have not equilibrated self-certifies (2026-07-20
        # review: two identical device regions at c*f_FD were accepted
        # unchanged for any c). The residual keeps the full physics.
        if external_flux is not None and external_flux_is_conservative_transfer:
            gain_int, loss_int = _gain_loss_sum(
                f_cur, ctx, K_s0, K_r0, T_bath,
                photon_params, pb_photon_params,
                N_p, N_emit, N_abs,
                None,
            )
            if not (
                np.any(gain_int[active] != 0.0)
                or np.any(loss_int[active] != 0.0)
            ):
                # Flux-only system (no internal channels enabled): the
                # external flux IS the entire equation, so it is the
                # correct normalization — there is nothing internal left
                # to equilibrate.
                gain_int, loss_int = gain, loss_rate
        else:
            gain_int, loss_int = gain, loss_rate
        R = gain - loss_rate * f_cur

        R_abs = np.abs(R[active])
        max_residual = float(np.max(R_abs))

        rate_scale = float(
            np.max(
                np.maximum(
                    np.abs(gain_int[active]),
                    np.abs(loss_int[active] * f_cur[active]),
                )
            )
        )

        backward_error = _gain_loss_backward_error(
            gain,
            loss_rate,
            f_cur,
            active,
            scale_gain=gain_int,
            scale_loss_rate=loss_int,
        )

        converged_abs = max_residual < tol
        converged_balance = backward_error <= backward_error_tol
        if converged_abs:
            if not certify_total_number:
                if converged_balance:
                    return f_cur.copy()
            else:
                # The dimensional residual can reach its absolute gate while
                # the aggregate balance error is still carried almost wholly
                # by the exponentially weak total-number mode.  Waiting for
                # ``converged_balance`` before decomposing the physics makes
                # the amplitude polish unreachable in exactly that regime
                # (the Fig. 7, 60 mK recovery stalled at aggregate backward
                # error 2.8e-3 with max |R| ~2e-32).  Diagnose and, when
                # possible, polish the number mode as soon as the unchanged
                # absolute gate is quiet.  A polished state always re-enters
                # the loop and must pass *all* ordinary return certificates;
                # this broadens no acceptance criterion.
                gain_number, loss_number, number_channel_configured = (
                    number_changing_gain_loss(
                        f_cur,
                        ctx,
                        K_r0,
                        T_bath,
                        pb_photon_params=pb_photon_params,
                        N_emit=N_emit,
                        N_abs=N_abs,
                        external_flux=external_flux,
                    )
                )
                number_error = _weighted_number_backward_error(
                    gain_number,
                    loss_number,
                    f_cur,
                    ctx.cell_weights,
                    active,
                )
                # Scaling the whole distribution is appropriate only after
                # the number-conserving channels have fixed its shape.  An
                # early absolute gate can also occur for an arbitrary cold
                # seed simply because every dimensional rate is tiny; in
                # that case amplitude polishing would preserve the wrong
                # shape.  Assemble the conserving physics directly and
                # require its cross-channel balance to be quiet before taking
                # the scalar shortcut.
                gain_shape, loss_shape = _number_conserving_gain_loss(
                    f_cur,
                    ctx,
                    K_s0,
                    T_bath,
                    photon_params=photon_params,
                    pb_photon_params=pb_photon_params,
                    N_p=N_p,
                )
                if (
                    np.any(gain_shape[active] != 0.0)
                    or np.any(loss_shape[active] != 0.0)
                ):
                    shape_backward_error = _gain_loss_backward_error(
                        gain_shape,
                        loss_shape,
                        f_cur,
                        active,
                    )
                else:
                    shape_backward_error = 0.0
                if number_channel_configured and number_error is None:
                    if _is_exact_absorbing_vacuum(
                        f_cur,
                        gain_number,
                        active,
                        T_bath=T_bath,
                        K_r0=K_r0,
                        N_abs=N_abs,
                        N_abs_override=N_abs_override,
                        pb_photon_params=pb_photon_params,
                        ctx=ctx,
                    ):
                        if converged_balance:
                            return f_cur.copy()
                    elif converged_balance:
                        raise RuntimeError(
                            "Newton aggregate metrics converged, but every "
                            "configured number-changing channel has zero "
                            "representable float64 turnover. The total-QP-number "
                            "mode cannot be certified at this temperature/grid."
                        )
                elif number_error is None or number_error <= backward_error_tol:
                    if converged_balance:
                        # Return exactly the feasible vector whose residual was
                        # certified. The initial state is validated and accepted
                        # trials are projected before their residual is evaluated,
                        # so no post-hoc clip is needed.
                        return f_cur.copy()
                elif shape_backward_error <= resolved_number_polish_shape_tol:
                    polished = _polish_number_mode_amplitude(
                        f_cur,
                        ctx,
                        K_r0,
                        T_bath,
                        active,
                        gain_number,
                        loss_number,
                        backward_error_tol=backward_error_tol,
                        pb_photon_params=pb_photon_params,
                        N_emit=N_emit,
                        N_abs=N_abs,
                        N_abs_override=N_abs_override,
                        external_flux=external_flux,
                    )
                    if polished is not None:
                        # Re-enter through the top so the full pointwise
                        # residual, aggregate balance, and number certificate
                        # are all checked at the polished state.
                        f_cur = polished
                        continue

                # The aggregate checks can become quiet before the
                # exponentially smaller number-changing mode. Do not reject a
                # correctable state merely because this is the first
                # aggregate-quiet iterate: keep taking Newton steps and require
                # the advertised number tolerance on the eventual return.
                # Numerically invisible cold modes still fail loudly in the
                # line search or at the iteration cap.

        J = _jacobian_analytical(
            f_cur, ctx, K_s0, K_r0,
            photon_params, pb_photon_params,
            N_p, N_emit, N_abs,
            external_flux,
        )
        J_act = J[np.ix_(active, active)]
        R_act = R[active]

        # The physical rate scale is local to each balance equation.  On cold
        # grids those scales can span O(1e100) or more; solving the raw rows
        # lets LAPACK discard the weak equations and can even flip the sign of
        # the vacuum-recovery direction.  Row scaling is algebraically exact
        # and changes conditioning only, never the root or any return gate.
        def try_direction(
            delta_f_act: np.ndarray,
            base_f: np.ndarray,
            base_max_residual: float,
            base_backward_error: float,
            base_number_error: float | None,
        ) -> tuple[bool, float]:
            """Backtrack one numerically equivalent Newton direction."""
            delta_f = np.zeros(NE)
            delta_f[active] = delta_f_act
            alpha = 1.0
            best_number_error = base_number_error
            best_number_alpha: float | None = None
            for _ in range(20):
                f_trial = np.clip(base_f + alpha * delta_f, 0.0, 1.0)
                gain_trial, loss_trial = _gain_loss_sum(
                    f_trial, ctx, K_s0, K_r0, T_bath,
                    photon_params, pb_photon_params,
                    N_p, N_emit, N_abs,
                    external_flux,
                )
                R_trial = gain_trial - loss_trial * f_trial
                trial_max_residual = float(np.max(np.abs(R_trial[active])))
                both_absolute_quiet = (
                    base_max_residual < tol and trial_max_residual < tol
                )
                residual_improved = (
                    not both_absolute_quiet
                    and trial_max_residual < base_max_residual
                )

                # Once the dimensional residual is already below its
                # advertised gate, demanding that an even tinier absolute
                # number decrease is not scale stable. Permit progress on the
                # independent physical balance merit while keeping both
                # absolute residuals below the same unchanged tolerance. The
                # next Newton iteration re-runs every return certificate.
                balance_improved = False
                if both_absolute_quiet:
                    if (
                        external_flux is not None
                        and external_flux_is_conservative_transfer
                    ):
                        gain_trial_scale, loss_trial_scale = _gain_loss_sum(
                            f_trial,
                            ctx,
                            K_s0,
                            K_r0,
                            T_bath,
                            photon_params,
                            pb_photon_params,
                            N_p,
                            N_emit,
                            N_abs,
                            None,
                        )
                        if not (
                            np.any(gain_trial_scale[active] != 0.0)
                            or np.any(loss_trial_scale[active] != 0.0)
                        ):
                            gain_trial_scale = gain_trial
                            loss_trial_scale = loss_trial
                    else:
                        gain_trial_scale = gain_trial
                        loss_trial_scale = loss_trial
                    trial_backward_error = _gain_loss_backward_error(
                        gain_trial,
                        loss_trial,
                        f_trial,
                        active,
                        scale_gain=gain_trial_scale,
                        scale_loss_rate=loss_trial_scale,
                    )
                    number_is_only_failed_gate = (
                        base_backward_error <= backward_error_tol
                        and base_number_error is not None
                        and base_number_error > backward_error_tol
                    )
                    if number_is_only_failed_gate:
                        assert base_number_error is not None
                        # Do not spend iterations improving an aggregate
                        # certificate that already passes while the independent
                        # total-number certificate remains stuck.  This was
                        # reachable in the refined Fig. 3 continuation: the
                        # aggregate error fell below 3e-18, yet Newton consumed
                        # all 500 iterations with pair-number error 1.09e-10
                        # against a 1e-10 contract.  Keep the passing aggregate
                        # gate and make strict progress on the certificate that
                        # actually blocks return.  If no Newton trial can do so,
                        # the scalar number-mode polish below gets its intended
                        # chance instead of being starved by irrelevant steps.
                        gain_trial_number, loss_trial_number, _configured = (
                            number_changing_gain_loss(
                                f_trial,
                                ctx,
                                K_r0,
                                T_bath,
                                pb_photon_params=pb_photon_params,
                                N_emit=N_emit,
                                N_abs=N_abs,
                                external_flux=external_flux,
                            )
                        )
                        trial_number_error = _weighted_number_backward_error(
                            gain_trial_number,
                            loss_trial_number,
                            f_trial,
                            ctx.cell_weights,
                            active,
                        )
                        balance_improved = (
                            trial_backward_error <= backward_error_tol
                            and trial_number_error is not None
                            and trial_number_error < base_number_error
                        )
                        if balance_improved:
                            assert trial_number_error is not None
                            # Every advertised return certificate is already
                            # satisfied: both absolute residuals are quiet,
                            # the aggregate backward error passes, and this
                            # trial clears the independent number-mode gate.
                            # Do not spend the remaining backtracking ladder
                            # searching for a smaller value of a certificate
                            # that has already passed.  Cold coupled solves
                            # can make each extra trial very expensive.
                            if trial_number_error <= backward_error_tol:
                                return True, alpha
                            # Search the complete backtracking ladder rather
                            # than accepting the first one-ulp improvement.
                            # The cold number mode is precisely where such tiny
                            # changes can otherwise consume the whole outer
                            # iteration budget.  Retain the feasible trial with
                            # the best actual blocking-certificate value.
                            if (
                                best_number_error is None
                                or trial_number_error < best_number_error
                            ):
                                best_number_error = trial_number_error
                                best_number_alpha = alpha
                            alpha *= 0.5
                            continue
                    else:
                        balance_improved = (
                            trial_backward_error < base_backward_error
                        )

                if residual_improved or balance_improved:
                    return True, alpha
                alpha *= 0.5
            if best_number_alpha is not None:
                return True, best_number_alpha
            return False, alpha

        # Row scaling and the raw system are algebraically identical but can
        # produce different floating-point directions near a singular
        # conserved mode. Prefer the scale-stable system; if its direction
        # cannot make physical progress, try the raw equilibration as a
        # numerical fallback rather than rejecting a recoverable state.
        accepted = False
        alpha = 0.0
        accepted_delta = np.empty(0)
        solve_error: np.linalg.LinAlgError | None = None
        had_direction = False
        try:
            J_solve, R_solve = _row_scaled_newton_system(
                J_act,
                R_act,
                gain[active],
                loss_rate[active] * f_cur[active],
            )
            scaled_delta = np.linalg.solve(J_solve, -R_solve)
            had_direction = True
            accepted, alpha = try_direction(
                scaled_delta,
                f_cur,
                max_residual,
                backward_error,
                number_error,
            )
            if accepted:
                accepted_delta = scaled_delta
        except np.linalg.LinAlgError as err:
            solve_error = err

        if not accepted:
            try:
                raw_delta = np.linalg.solve(J_act, -R_act)
                had_direction = True
                accepted, alpha = try_direction(
                    raw_delta,
                    f_cur,
                    max_residual,
                    backward_error,
                    number_error,
                )
                if accepted:
                    accepted_delta = raw_delta
            except np.linalg.LinAlgError as err:
                solve_error = err

        if not accepted and not had_direction and solve_error is not None:
            raise RuntimeError(
                f"Singular Jacobian at Newton iteration {iteration}"
            ) from solve_error

        if (
            not accepted
            and converged_abs
            and certify_total_number
            and number_error is not None
            and number_error > backward_error_tol
        ):
            # A nearly singular cold Jacobian can have no descent direction
            # for either dimensional residual or normalized balance even
            # though its current nonzero spectral shape can bracket the slow
            # scalar number root.  Use amplitude polishing as a globalization
            # fallback only after both algebraically equivalent Newton
            # directions and their line searches have failed.  Unlike an
            # eager polish this cannot perturb an ordinarily converging shape
            # trajectory (important for the resolved 20 mK tail), and the
            # candidate still re-enters every unchanged return certificate.
            polished = _polish_number_mode_amplitude(
                f_cur,
                ctx,
                K_r0,
                T_bath,
                active,
                gain_number,
                loss_number,
                backward_error_tol=backward_error_tol,
                pb_photon_params=pb_photon_params,
                N_emit=N_emit,
                N_abs=N_abs,
                N_abs_override=N_abs_override,
                external_flux=external_flux,
            )
            if polished is not None:
                f_cur = polished
                continue

        if not accepted:
            number_note = (
                ""
                if number_error is None
                else f", number-changing backward error = {number_error:.2e}"
            )
            raise RuntimeError(
                f"Newton line search failed at iteration {iteration}. "
                f"max |residual| = {max_residual:.2e}, "
                f"max-relative = "
                f"{max_residual / max(rate_scale, 1e-300):.2e}, "
                f"gain/loss backward error = {backward_error:.2e}"
                f"{number_note} "
                f"(limit {backward_error_tol:.2e})"
            )

        delta_f = np.zeros(NE)
        delta_f[active] = accepted_delta
        f_cur = np.clip(f_cur + alpha * delta_f, 0.0, 1.0)

    number_note = (
        ""
        if number_error is None
        else f", number-changing backward error = {number_error:.2e}"
    )
    raise RuntimeError(
        f"Newton iteration did not converge in {max_iter} iterations. "
        f"Final max |residual| = {max_residual:.2e}, "
        f"max-relative = {max_residual / max(rate_scale, 1e-300):.2e}, "
        f"gain/loss backward error = {backward_error:.2e}"
        f"{number_note} "
        f"(limit {backward_error_tol:.2e})"
    )


def _gain_loss_backward_error(
    gain: np.ndarray,
    loss_rate: np.ndarray,
    f: np.ndarray,
    active: np.ndarray,
    *,
    scale_gain: np.ndarray | None = None,
    scale_loss_rate: np.ndarray | None = None,
) -> float:
    """L1 normwise backward error of ``gain - loss_rate*f = 0``.

    ``scale_gain``/``scale_loss_rate`` (when given) supply the turnover
    used for NORMALIZATION while the residual keeps ``gain``/``loss_rate``.
    Callers with an ``external_flux`` pass the internal (collision/photon)
    turnover here: a large but exactly balanced exchange term must not
    inflate the denominator and self-certify a state the internal
    channels have not equilibrated (2026-07-20 review: two device
    regions at ``c * f_FD`` certified for any ``c`` because the balanced
    junction exchange dominated this denominator).
    """
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
    sg_arr = gain_arr if scale_gain is None else np.asarray(scale_gain, dtype=float)
    sl_arr = loss_arr if scale_loss_rate is None else np.asarray(scale_loss_rate, dtype=float)
    if not (sg_arr.shape == sl_arr.shape == f_arr.shape):
        raise ValueError("scale arrays must match the state shape.")
    if np.any(~np.isfinite(np.stack((sg_arr, sl_arr), axis=0))):
        raise ValueError("scale arrays must contain only finite values.")
    with np.errstate(over="ignore", invalid="ignore"):
        loss_term = loss_arr[active_arr] * f_arr[active_arr]
        scale_loss_term = sl_arr[active_arr] * f_arr[active_arr]
    residual = gain_arr[active_arr] - loss_term
    if np.any(~np.isfinite(loss_term)) or np.any(~np.isfinite(residual)):
        raise ValueError("Assembled gain/loss balance terms must be finite.")
    if np.any(~np.isfinite(scale_loss_term)):
        raise ValueError("Assembled scale terms must be finite.")
    scale_gain_active = sg_arr[active_arr]
    common = float(
        max(
            np.max(np.abs(scale_gain_active), initial=0.0),
            np.max(np.abs(scale_loss_term), initial=0.0),
            np.max(np.abs(residual), initial=0.0),
        )
    )
    if common == 0.0:
        return 0.0
    denominator = float(
        np.sum(np.abs(scale_gain_active) / common)
        + np.sum(np.abs(scale_loss_term) / common)
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
    if photon_params is not None and (
        photon_params["c_phot"] > 0.0 and photon_params["omega_0"] > 0.0
    ):
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
    if pb_photon_params is not None and (
        pb_photon_params["c_phot_PB"] > 0.0
        and pb_photon_params["omega_PB"] > 0.0
    ):
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

            # Hard 2Δ pair threshold — the SAME gate the rate function
            # applies (2026-07-20 round-4 review: leaving these
            # derivatives in while the rates zero them gave a 0.74%
            # Jacobian-vs-FD mismatch on gap-cut grids, feeding coupled
            # Newton a wrong Jacobian).
            if not _pair_channel_open(omega_PB_snapped, ctx.gap):
                continue
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
