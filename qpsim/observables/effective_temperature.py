r"""Effective phonon temperature ``T_*`` via a weighted Bose-Einstein fit.

Fischer 2023 Eq. 36 defines ``T_*`` as the temperature at which the
ratio ``n_ph(ω) / n_BE(ω, T)`` is approximately constant across the
pair-breaking band (``ω ≥ 2Δ``). In general the driven steady-state
phonon distribution is not exactly proportional to a single
Bose-Einstein — low-``ω`` modes stay near the bath while
recombination-heated modes above ``2Δ`` float up — so the extracted
``T_*`` is a best-fit characterization, not an exact inversion.

The fit minimizes the weighted variance of the log-ratio
``log(n_ph / n_BE)`` over ``ω ≥ 2Δ``, with weights proportional to
``n_ph`` itself so the fit tracks the most-occupied pair-breaking
modes rather than the thermally-suppressed high-``ω`` tail.

Port of the legacy ``extract_T_star_phonon`` from the old
``reproduce_fischer_fig5.py`` script.
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.constants import KB_UEV_PER_K


def effective_phonon_temperature(
    n_ph: np.ndarray,
    omega_bins: np.ndarray,
    gap: float,
    *,
    T_bath: float,
    T_max: float | None = None,
) -> float:
    r"""Fit ``n_ph(ω ≥ 2Δ)`` to a Bose-Einstein profile and return ``T_*``.

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
        Bath temperature in K. Used as the lower bound on ``T_*``
        (the fit returns ``T_bath`` when no pair-breaking modes have
        any occupation).
    T_max
        Optional upper bound on the fit range in K. Defaults to
        ``100 × T_bath`` — wide enough to admit driven steady states
        but narrow enough for :func:`scipy.optimize.minimize_scalar`
        to converge in a handful of iterations.

    Returns
    -------
    float
        Effective phonon temperature in K.
    """
    n_ph = np.asarray(n_ph, dtype=float)
    omega_bins = np.asarray(omega_bins, dtype=float)
    if n_ph.shape != omega_bins.shape:
        raise ValueError(
            f"n_ph shape {n_ph.shape} must match omega_bins shape {omega_bins.shape}."
        )
    if n_ph.ndim != 1:
        raise ValueError("n_ph and omega_bins must be one-dimensional.")
    if np.any(~np.isfinite(n_ph)) or np.any(n_ph < 0.0):
        raise ValueError("n_ph must contain finite, non-negative occupations.")
    if np.any(~np.isfinite(omega_bins)) or np.any(omega_bins < 0.0):
        raise ValueError("omega_bins must contain finite, non-negative frequencies.")
    if not np.isfinite(gap) or gap <= 0:
        raise ValueError("gap must be positive.")
    if not np.isfinite(T_bath) or T_bath <= 0:
        raise ValueError("T_bath must be positive.")
    if T_max is not None and (not np.isfinite(T_max) or T_max <= T_bath):
        raise ValueError("T_max, when supplied, must be finite and exceed T_bath.")

    mask = (omega_bins >= 2.0 * gap) & (n_ph > 1e-300)
    if not np.any(mask):
        return float(T_bath)

    omega_fit = omega_bins[mask]
    n_fit = n_ph[mask]
    if omega_fit.size == 1:
        # The objective fits the *shape* of n_ph/n_BE while allowing an
        # arbitrary constant amplitude.  With one mode its weighted variance
        # is identically zero for every T, so returning the optimizer's upper
        # bound is an arbitrary artifact rather than a temperature estimate.
        raise ValueError(
            "effective phonon temperature is underdetermined with only one "
            "occupied pair-breaking mode; provide at least two frequencies."
        )
    weights = n_fit / float(np.sum(n_fit))

    def _weighted_variance_of_log_ratio(T: float) -> float:
        if T <= 0:
            return 1e30
        exponent = np.minimum(omega_fit / (KB_UEV_PER_K * T), 500.0)
        n_BE = 1.0 / np.expm1(exponent)
        n_BE = np.maximum(n_BE, 1e-300)
        log_ratio = np.log(n_fit / n_BE)
        mean_lr = float(np.sum(weights * log_ratio))
        return float(np.sum(weights * (log_ratio - mean_lr) ** 2))

    upper = float(T_max) if T_max is not None else 100.0 * T_bath
    # Lower bound is T_bath itself: the bath always populates the
    # pair-breaking modes at at least the BE(T_bath) level, so any
    # ``T_* < T_bath`` would be a fitting artifact, not physics.
    from scipy.optimize import minimize_scalar

    result = minimize_scalar(
        _weighted_variance_of_log_ratio,
        bounds=(T_bath, upper),
        method="bounded",
    )
    T_star = float(result.x)
    # A bounded optimizer parks the solution on the upper bound when the true
    # T_* exceeds the fit window; the returned value is then a clamp, not a
    # fit, and must not be read as a measured temperature. (The lower bound at
    # T_bath is deliberate physics — n_ph ≥ BE(T_bath) — so it is not flagged.)
    if T_star >= upper * (1.0 - 1e-3):
        warnings.warn(
            f"effective_phonon_temperature pinned to the upper fit bound "
            f"{upper:g} K; the true effective temperature likely exceeds the fit "
            f"window. Pass a larger T_max — this value is a clamp, not a fit.",
            RuntimeWarning,
            stacklevel=2,
        )
    return T_star
