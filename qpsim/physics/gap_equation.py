"""BCS gap self-consistency: equilibrium calibration + reference-subtracted runtime solve.

Calibration (once per T_c, T_bath) computes Δ_eq and caches 1/λ and ω_D.
The runtime solver then takes an occupation ``f`` and returns the
current Δ via Brent's method on the reference-subtracted residual.

Note on naming: the old repo called the runtime occupation parameter
``f_L`` even though the function body computes ``1 − 2 f`` from it
(i.e. it actually expects the Fermi-Dirac occupation ``f``, not the
longitudinal combination ``f_L = 1 − 2f``). Renamed to ``f`` here.

Ported from the old ``qpsim/numerics/gap_equation.py`` at Gate 2.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K

# BCS universal ratio Δ(0) / (kB T_c)
_BCS_DELTA0_RATIO = 1.764


@dataclass
class GapCalibration:
    """Frozen equilibrium constants for the reference-subtracted gap solver.

    Produced once by :func:`calibrate_gap`; consumed by :func:`solve_gap`.
    Leading-underscore fields are private implementation details;
    ``delta_eq`` / ``T_c`` / ``T_bath`` are the user-facing values.
    """

    delta_eq: float       # Equilibrium gap at T_bath (μeV)
    T_c: float            # Critical temperature (K)
    T_bath: float         # Bath temperature (K)
    _ref_integral: float  # Equilibrium integral value = 1/λ
    _omega_D: float       # Debye cutoff (μeV)
    _inv_lambda: float    # 1/λ


def _fermi_dirac(E: np.ndarray | float, T: float) -> np.ndarray | float:
    """f_FD(E, T) = 1 / (exp(E/kT) + 1). Zero-T limit is a step at E=0."""
    if T <= 0:
        return np.where(np.asarray(E) > 0, 0.0, 1.0)
    kT = _KB_UEV_PER_K * T
    exponent = np.minimum(np.asarray(E) / kT, 500.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def _gap_integral_cosh(delta: float, T: float, omega_D: float, n_quad: int) -> float:
    """∫₀^{u_max} [1 − 2 f_FD(Δ cosh u, T)] du.

    Uses E = Δ cosh u to remove the 1/√(E² − Δ²) singularity at the gap edge.
    """
    if delta <= 0:
        return 0.0
    u_max = np.arccosh(omega_D / delta) if omega_D > delta else 0.0
    if u_max <= 0:
        return 0.0
    u = np.linspace(0, u_max, n_quad + 1)
    E_u = delta * np.cosh(u)
    f_u = _fermi_dirac(E_u, T)
    integrand = 1.0 - 2.0 * f_u
    return float(np.trapezoid(integrand, u))


def _gap_integral_f(
    delta: float, f: np.ndarray, E_bins: np.ndarray, omega_D: float,
) -> float:
    """∫_Δ^{ω_D} (1 − 2 f(E)) / √(E² − Δ²) dE via cosh substitution + interpolation.

    ``f`` is the Fermi-Dirac occupation on ``E_bins``; the integrand uses
    the longitudinal combination ``1 − 2f`` as required by the gap equation.
    """
    if delta <= 0:
        return 0.0
    if omega_D <= delta:
        return 0.0
    u_max = np.arccosh(omega_D / delta)
    n_quad = max(len(E_bins) * 2, 256)
    u = np.linspace(0, u_max, n_quad + 1)
    E_u = delta * np.cosh(u)
    f_interp = np.interp(E_u, E_bins, f, left=float(f[0]), right=0.0)
    integrand = 1.0 - 2.0 * f_interp
    return float(np.trapezoid(integrand, u))


def calibrate_gap(
    T_c: float,
    T_bath: float,
    *,
    omega_D_over_Tc: float = 100.0,
    n_quadrature: int = 512,
    xtol: float | None = None,
) -> GapCalibration:
    """Compute Δ_eq(T_bath) and cache 1/λ and ω_D for the runtime solver.

    Uses the BCS gap equation with the E = Δ cosh u substitution. Only
    T_c is user-facing — λ and ω_D are derived from ``omega_D_over_Tc``.

    ``xtol`` (μeV) overrides the default brentq tolerance of ``1e-6 * kBTc``.
    Tighten when the caller needs ``δΔ_T = Δ_0 − Δ_eq`` resolved well below
    the default ~1e-4 μeV — e.g. the Fischer 2023 Fig 6 observable, which
    divides by an exponentially small δΔ_T at T_B ≪ T_c.
    """
    if T_c <= 0:
        raise ValueError("T_c must be positive.")
    if T_bath < 0:
        raise ValueError("T_bath must be non-negative.")

    kBTc = _KB_UEV_PER_K * T_c
    omega_D = omega_D_over_Tc * kBTc
    delta_0 = _BCS_DELTA0_RATIO * kBTc  # BCS zero-T gap

    # 1/λ from T=0 gap equation (integrand = 1 for all E > 0).
    inv_lambda = _gap_integral_cosh(delta_0, 0.0, omega_D, n_quadrature)
    if inv_lambda <= 0:
        raise RuntimeError("Failed to compute 1/λ from T=0 gap equation.")

    if T_bath >= T_c:
        delta_eq = 0.0
    elif T_bath <= 0:
        delta_eq = delta_0
    else:
        from scipy.optimize import brentq

        def residual(delta: float) -> float:
            return _gap_integral_cosh(delta, T_bath, omega_D, n_quadrature) - inv_lambda

        eps = 0.01 * kBTc
        xtol_brentq = 1e-6 * kBTc if xtol is None else xtol
        delta_eq = brentq(residual, eps, delta_0, xtol=xtol_brentq)

    return GapCalibration(
        delta_eq=float(delta_eq),
        T_c=T_c,
        T_bath=T_bath,
        _ref_integral=inv_lambda,
        _omega_D=omega_D,
        _inv_lambda=inv_lambda,
    )


def solve_gap(
    calibration: GapCalibration,
    f: np.ndarray,
    E_bins: np.ndarray,
    *,
    bracket_factor: float = 0.5,
    xtol: float | None = None,
) -> float:
    """Runtime gap solve from the Fermi-Dirac occupation ``f(E)``.

    Solves ``∫_Δ^{ω_D} (1 − 2f(E))/√(E² − Δ²) dE − 1/λ = 0`` using
    Brent's method on a bracket around ``calibration.delta_eq``.
    Returns 0 if no superconducting solution is found (normal state).

    Parameters
    ----------
    calibration
        From :func:`calibrate_gap`.
    f
        Fermi-Dirac occupation on ``E_bins``, shape ``(NE,)``.
    E_bins
        Energy bin centers, shape ``(NE,)``.
    bracket_factor
        Initial half-width of the search bracket as a fraction of Δ_eq.
    xtol
        brentq absolute tolerance in μeV. Default ``1e-6 * delta_eq`` matches
        legacy behavior; tighten when the caller's observable depends on
        sub-default-precision shifts in Δ (e.g. fig6_paper at low T_B).
    """
    delta_eq = calibration.delta_eq
    if delta_eq <= 0:
        return 0.0

    ref_integral = calibration._ref_integral
    E = np.asarray(E_bins, dtype=float).ravel()
    f_arr = np.asarray(f, dtype=float).ravel()
    omega_D = calibration._omega_D

    def residual(delta: float) -> float:
        return _gap_integral_f(delta, f_arr, E, omega_D) - ref_integral

    from scipy.optimize import brentq

    lo = max(delta_eq * (1.0 - bracket_factor), 1e-3)
    hi = delta_eq * (1.0 + bracket_factor)

    r_lo, r_hi = residual(lo), residual(hi)
    for _ in range(5):
        if r_lo * r_hi < 0:
            break
        lo *= 0.5
        hi *= 1.5
        lo = max(lo, 1e-3)
        r_lo, r_hi = residual(lo), residual(hi)
    else:
        # Bracket widening failed. Negative residual at lo ~1e-3 means
        # no superconducting solution (gap collapses to normal state).
        if r_lo < 0:
            return 0.0
        # Positive residual without sign change means the true root sits
        # above the widened bracket (population colder than thermal near
        # T_c). Returning Δ_eq is then an UNDERestimate — keep the legacy
        # fallback for continuity, but say so instead of staying silent.
        warnings.warn(
            "solve_gap: bracket widening found no sign change with a "
            "positive residual at both ends; the self-consistent gap "
            "exceeds the search bracket and Δ_eq is returned as a "
            "fallback (an underestimate).",
            stacklevel=2,
        )
        return delta_eq

    xtol_brentq = 1e-6 * delta_eq if xtol is None else xtol
    return float(brentq(residual, lo, hi, xtol=xtol_brentq))
