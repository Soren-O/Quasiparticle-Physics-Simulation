r"""Effective phonon temperature via a weighted Bose-Einstein shape fit.

This is a qpsim diagnostic, not a Fischer 2023 construct, and it is **not**
the ``T_*`` of that paper. Fischer 2023 fixes ``T_*`` in closed form as the
quasiparticle-distribution width scale ``k_B T_* = (A n̄)^{1/6}`` (the Fig.
5/6 abscissa, implemented separately as ``_kBTstar_eq35`` in
``validation/fischer_2023/fig5_paper.py``) and never assigns
the driven phonon distribution a temperature at all — it decomposes it into
the bath Bose-Einstein plus explicit nonequilibrium terms. The quantity
returned here is instead the temperature whose Bose-Einstein *shape* best
matches ``n_ph(ω)`` across the pair-breaking band (``ω ≥ 2Δ``), so it must
not be normalized as ``k_B T / Δ`` and compared against the Fig. 5/6 axis.

In general the driven steady-state phonon distribution is not exactly
proportional to a single Bose-Einstein — low-``ω`` modes stay near the
bath while recombination-heated modes above ``2Δ`` float up — so the
extracted temperature is a best-fit characterization, not an exact
inversion.

The fit minimizes the weighted variance of the log-ratio
``log(n_ph / n_BE)`` over ``ω ≥ 2Δ``, with weights proportional to
``n_ph`` itself so the fit tracks the most-occupied pair-breaking
modes rather than the thermally-suppressed high-``ω`` tail. Both the
weights and the objective are invariant under multiplying the whole
spectrum by a positive constant, so this measures shape only: a
pair-breaking band occupied at ``10 ×`` the bath Bose-Einstein level is
reported as unheated (``T_bath``), not as a hotter phonon system.
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.constants import KB_UEV_PER_K


def _finite_real_scalar(name: str, value: float) -> float:
    """Return one finite real scalar without silent coercion."""
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result!r}.")
    return result


def _finite_real_array(name: str, values: np.ndarray) -> np.ndarray:
    """Return a finite real array, rejecting complex input before casting."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def effective_phonon_temperature(
    n_ph: np.ndarray,
    omega_bins: np.ndarray,
    gap: float,
    *,
    T_bath: float,
    T_max: float | None = None,
    return_residual: bool = False,
) -> float | tuple[float, float]:
    r"""Fit the ``n_ph(ω ≥ 2Δ)`` shape to a Bose-Einstein profile.

    Not the Fischer 2023 ``T_*`` — see the module docstring.

    Parameters
    ----------
    n_ph
        Phonon occupation on ``omega_bins`` (shape ``(N_ω,)``).
    omega_bins
        Phonon-frequency bin centers in μeV (shape ``(N_ω,)``).
    gap
        Superconducting gap Δ in μeV (sets the ``ω ≥ 2Δ`` pair-
        breaking threshold).
    T_bath
        Bath temperature in K. Used as the lower bound on the fitted
        temperature (the fit returns ``T_bath`` when no pair-breaking
        modes have any occupation).
    T_max
        Optional upper bound on the fit range in K. Defaults to
        ``100 × T_bath`` — wide enough to admit driven steady states
        but narrow enough for :func:`scipy.optimize.minimize_scalar`
        to converge in a handful of iterations.

    return_residual
        Also return the fit's residual: the weighted standard deviation of
        ``log(n_ph / n_BE(T))`` over the fitted band at the optimum, i.e. how
        far the spectrum's SHAPE is from any single Bose-Einstein. A caller
        assigning a temperature to a non-thermal spectrum needs this number
        to know whether the temperature means anything; ``0`` for an exact
        Bose-Einstein, and ``0`` too when no pair-breaking mode is occupied
        (the fit then returns ``T_bath`` with nothing to fit).

    Returns
    -------
    float, or (float, float)
        Effective phonon temperature in K, and with ``return_residual`` the
        residual described above.
    """
    n_ph = _finite_real_array("n_ph", n_ph)
    omega_bins = _finite_real_array("omega_bins", omega_bins)
    if n_ph.shape != omega_bins.shape:
        raise ValueError(
            f"n_ph shape {n_ph.shape} must match omega_bins shape {omega_bins.shape}."
        )
    if n_ph.ndim != 1:
        raise ValueError("n_ph and omega_bins must be one-dimensional.")
    if np.any(n_ph < 0.0):
        raise ValueError("n_ph must contain finite, non-negative occupations.")
    if np.any(omega_bins < 0.0):
        raise ValueError("omega_bins must contain finite, non-negative frequencies.")
    gap = _finite_real_scalar("gap", gap)
    T_bath = _finite_real_scalar("T_bath", T_bath)
    if gap <= 0:
        raise ValueError("gap must be positive.")
    if T_bath <= 0:
        raise ValueError("T_bath must be positive.")
    if T_max is not None:
        T_max = _finite_real_scalar("T_max", T_max)
        if T_max <= T_bath:
            raise ValueError("T_max, when supplied, must exceed T_bath.")

    mask = (omega_bins >= 2.0 * gap) & (n_ph > 0.0)
    if not np.any(mask):
        return (float(T_bath), 0.0) if return_residual else float(T_bath)

    omega_fit = omega_bins[mask]
    n_fit = n_ph[mask]
    # Normalize through the peak so a finite common amplitude cannot overflow
    # the sum.  Both the weights and log-shape are invariant under multiplying
    # the entire spectrum by a positive constant.
    peak = float(np.max(n_fit))
    scaled = n_fit / peak
    represented = scaled > 0.0
    if np.count_nonzero(represented) < 2:
        raise ValueError(
            "effective phonon temperature is numerically underdetermined: "
            "fewer than two occupied modes remain representable after "
            "amplitude normalization."
        )
    omega_fit = omega_fit[represented]
    scaled = scaled[represented]
    weights = scaled / float(np.sum(scaled))
    contributing = weights > 0.0
    if np.count_nonzero(contributing) < 2:
        raise ValueError(
            "effective phonon temperature is numerically underdetermined: "
            "fewer than two occupied modes have representable normalized "
            "weights."
        )
    omega_fit = omega_fit[contributing]
    scaled = scaled[contributing]
    weights = weights[contributing]
    weights = weights / float(np.sum(weights))
    if np.unique(omega_fit).size < 2:
        # The objective fits the *shape* of n_ph/n_BE while allowing an
        # arbitrary constant amplitude. With only one distinct frequency its
        # weighted variance is independent of T, even when that frequency was
        # duplicated across multiple bins.
        raise ValueError(
            "effective phonon temperature is underdetermined with fewer than "
            "two distinct occupied pair-breaking frequencies."
        )
    omega_scale = float(np.max(np.abs(omega_fit)))
    minimum_relative_span = np.finfo(float).eps ** (2.0 / 3.0)
    if float(np.ptp(omega_fit)) <= minimum_relative_span * omega_scale:
        raise ValueError(
            "effective phonon temperature is numerically underdetermined: "
            "the occupied frequency span is too narrow for a conditioned "
            "binary64 shape fit."
        )
    # Take the log after normalization. Subtracting two O(700) logarithms for
    # a large common amplitude loses the very shape digits this fit measures.
    log_n_shape = np.log(scaled)

    def _weighted_variance_of_log_ratio(log_T: float) -> float:
        T = float(np.exp(log_T))
        if not np.isfinite(T) or T <= 0.0:
            return 1e30
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            exponent = (omega_fit / KB_UEV_PER_K) / T
            # log(n_BE) = -x - log(1 - exp(-x)).  The expm1 form is
            # accurate for x << 1 and remains finite for every finite x,
            # including occupations that are subnormal in ordinary space.
            log_n_BE = -exponent - np.log(-np.expm1(-exponent))
        if np.any(~np.isfinite(log_n_BE)):
            return 1e30
        log_ratio = log_n_shape - log_n_BE
        mean_lr = float(np.sum(weights * log_ratio))
        return float(np.sum(weights * (log_ratio - mean_lr) ** 2))

    if T_max is None:
        if T_bath > np.finfo(float).max / 100.0:
            raise ValueError(
                "The default T_max=100*T_bath is not representable; "
                "provide an explicit finite T_max."
            )
        upper = 100.0 * T_bath
    else:
        upper = T_max
    # Lower bound is T_bath itself: the bath always populates the
    # pair-breaking modes at at least the BE(T_bath) level, so any
    # fitted temperature below ``T_bath`` would be a fitting artifact, not
    # physics.
    from scipy.optimize import minimize_scalar

    log_lower = float(np.log(T_bath))
    log_upper = float(np.log(upper))
    result = minimize_scalar(
        _weighted_variance_of_log_ratio,
        bounds=(log_lower, log_upper),
        method="bounded",
        # An absolute tolerance in log(T) is a relative tolerance in T, unlike
        # scipy's default absolute-kelvin xatol.
        options={"xatol": 1e-12},
    )
    candidates = (
        (float(result.fun), float(result.x)),
        (_weighted_variance_of_log_ratio(log_lower), log_lower),
        (_weighted_variance_of_log_ratio(log_upper), log_upper),
    )
    best_fun, best_log_T = min(candidates, key=lambda item: item[0])
    # exp(log(bound)) can round one ulp outside the original interval.
    # Preserve the documented hard bounds in the returned representation.
    T_eff = min(max(float(np.exp(best_log_T)), T_bath), upper)
    # A bounded optimizer parks the solution on the upper bound when the true
    # shape temperature exceeds the fit window; the returned value is then a
    # clamp, not a fit, and must not be read as a measured temperature. (The
    # lower bound at T_bath is deliberate physics — n_ph ≥ BE(T_bath) — so it
    # is not flagged.)
    if T_eff >= upper * (1.0 - 1e-3):
        warnings.warn(
            f"effective_phonon_temperature pinned to the upper fit bound "
            f"{upper:g} K; the true effective temperature likely exceeds the fit "
            f"window. Pass a larger T_max — this value is a clamp, not a fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    if return_residual:
        # The objective is the weighted VARIANCE of the log ratio; its square
        # root is the shape's departure from a Bose-Einstein in log units,
        # i.e. a relative deviation for small values.
        return T_eff, float(np.sqrt(max(float(best_fun), 0.0)))
    return T_eff
