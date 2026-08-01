"""Minimal clean-room analytic reference for Fischer--Catelani 2023 Fig. 6.

This module is deliberately independent of :mod:`qpsim`.  It transcribes the
paper's scalar Eq. 35 -> Eq. 47 -> Eq. 53 chain and evaluates the thermal
denominator with a continuum BCS integral.  It is not a reconstruction of the
authors' unpublished numerical solver and must not be described as author
source.

Units follow the paper calculation: energies are in microelectronvolts, times
are in nanoseconds, and ``k_B = hbar = 1`` inside the rate equations after
temperature is converted to energy.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.integrate import quad

# Literal Fig. 6 / Table-I inputs.
KB_UEV_PER_K = 86.17333262145178
DELTA_0_UEV = 180.0
DELTA_0_OVER_KBT_C = 1.764
KBT_C_UEV = DELTA_0_UEV / DELTA_0_OVER_KBT_C
T_C_K = KBT_C_UEV / KB_UEV_PER_K
OMEGA_0_UEV = DELTA_0_UEV / 9.0
TAU_0_NS = 438.0
TAU_0_PB_NS = 0.255
TAU_L_NS = 0.255
C_PHOTON_PER_NS = 1.0e-9  # 1 Hz

# Eq. 35: k_B T_* = (A nbar)^(1/6).
EQ35_A_UEV6 = (
    (105.0 / 64.0)
    * KBT_C_UEV**3
    * C_PHOTON_PER_NS
    * TAU_0_NS
    * OMEGA_0_UEV**2
    * DELTA_0_UEV
)

# Eq. 51 and Appendix-E coefficients used by Eq. 47.
EQ51_GAMMA = 0.84
APPENDIX_E_A_MINUS_HALF = 2.1
APPENDIX_E_A_PLUS_HALF = 0.88
APPENDIX_E_A_PLUS_THREE_HALVES = 0.77

# Eq. 53 gap-suppression bracket.
EQ53_LINEAR = 0.42
EQ53_QUADRATIC = 0.22


def _finite_nonnegative(name: str, raw: float) -> float:
    if isinstance(raw, (bool, np.bool_)) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite non-negative real number.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite non-negative real number.") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number.")
    return value


def eq35_t_star_over_delta(n_bar: float) -> float:
    """Return ``T_*/Delta_0`` from the mean photon number (paper Eq. 35)."""

    occupation = _finite_nonnegative("n_bar", n_bar)
    if occupation == 0.0:
        return 0.0
    kBT_star = (EQ35_A_UEV6 * occupation) ** (1.0 / 6.0)
    return float(kBT_star / DELTA_0_UEV)


def eq35_n_bar(t_star_over_delta: float) -> float:
    """Invert paper Eq. 35 for a requested Fig. 6 abscissa."""

    ratio = _finite_nonnegative("t_star_over_delta", t_star_over_delta)
    return float((ratio * DELTA_0_UEV) ** 6 / EQ35_A_UEV6)


def eq47_x_qp_paper(T_bath_K: float, t_star_over_delta: float) -> float:
    """Return ``N_qp/(2 rho_F Delta_0)`` from the analytic Eq. 47 balance.

    The quadratic balance is

    ``R_bar N^2 - G_drive N - G_thermal = 0``.

    Besides Eqs. 47--51, ``R_bar`` includes the paper's Appendix-E
    ``T_*/Delta_0`` expansion.  Its linear coefficient carries the
    finite-escape factor from the paper's leading Eq. 112 correction,
    ``(1 + tau_l/(2 tau_0^PB)) / (1 + tau_l/tau_0^PB)``.  This factor is
    part of the published dashed Fig. 6 curve: dropping it moves all three
    clean-room curves outside the digitized trace uncertainty.

    ``rho_F`` is set to one while solving for ``N`` and cancels in the
    returned dimensionless fraction.
    """

    temperature = _finite_nonnegative("T_bath_K", T_bath_K)
    ratio = _finite_nonnegative("t_star_over_delta", t_star_over_delta)

    T_bath_uev = KB_UEV_PER_K * temperature
    escape_ratio = TAU_L_NS / TAU_0_PB_NS
    tau_bar = TAU_0_NS * (1.0 + escape_ratio)
    delta_over_kBT_c_cubed = (DELTA_0_UEV / KBT_C_UEV) ** 3

    if T_bath_uev == 0.0:
        G_thermal = 0.0
    else:
        G_thermal = (
            (16.0 * math.pi / tau_bar)
            * delta_over_kBT_c_cubed
            * T_bath_uev
            * math.exp(-2.0 * DELTA_0_UEV / T_bath_uev)
        )

    if ratio == 0.0:
        G_drive = 0.0
    else:
        G_drive = (
            (EQ51_GAMMA / tau_bar)
            * escape_ratio
            * delta_over_kBT_c_cubed
            * ratio**4.5
            * math.exp(-math.sqrt(14.0 / 5.0) * ratio**-3.0)
        )

    c1 = APPENDIX_E_A_PLUS_HALF / APPENDIX_E_A_MINUS_HALF
    c2 = (
        1.25 * APPENDIX_E_A_PLUS_THREE_HALVES / APPENDIX_E_A_MINUS_HALF
        - 0.75 * c1**2
    )
    trapping = (1.0 + 0.5 * escape_ratio) / (1.0 + escape_ratio)
    R_0 = 2.0 * DELTA_0_UEV**2 / (tau_bar * KBT_C_UEV**3)
    R_bar = R_0 * (1.0 + trapping * c1 * ratio + c2 * ratio**2)

    discriminant = G_drive**2 + 4.0 * R_bar * G_thermal
    N_qp = (G_drive + math.sqrt(discriminant)) / (2.0 * R_bar)
    return float(N_qp / (2.0 * DELTA_0_UEV))


def eq53_driven_gap_suppression(T_bath_K: float, t_star_over_delta: float) -> float:
    """Return ``delta Delta / Delta_0`` from the paper's analytic Eq. 53."""

    ratio = _finite_nonnegative("t_star_over_delta", t_star_over_delta)
    x_qp = eq47_x_qp_paper(T_bath_K, ratio)
    bracket = 1.0 - EQ53_LINEAR * ratio + EQ53_QUADRATIC * ratio**2
    return float(x_qp * bracket)


def thermal_gap_integral(T_bath_K: float) -> float:
    r"""Return the continuum thermal direct-gap integral.

    For ideal BCS density of states,

    .. math::
        I_T = 2\int_{\Delta_0}^{\infty}
              \frac{f_{\rm FD}(E,T_B)}{\sqrt{E^2-\Delta_0^2}}\,dE.

    The substitution ``E = Delta_0 cosh(u)`` removes the square-root
    singularity exactly:

    .. math:: I_T = 2\int_0^\infty f_{\rm FD}(\Delta_0\cosh u,T_B)\,du.

    The finite integration endpoint is where the Fermi exponent reaches
    the largest useful float64 exponential argument; the omitted positive
    tail is below float64 resolution.
    """

    temperature = _finite_nonnegative("T_bath_K", T_bath_K)
    if temperature == 0.0:
        return 0.0

    beta_delta = DELTA_0_UEV / (KB_UEV_PER_K * temperature)
    exponent_cutoff = math.log(np.finfo(float).max)
    if beta_delta >= exponent_cutoff:
        return 0.0
    u_max = math.acosh(exponent_cutoff / beta_delta)

    def transformed_integrand(u: float) -> float:
        boltzmann = math.exp(-beta_delta * math.cosh(u))
        return 2.0 * boltzmann / (1.0 + boltzmann)

    value, _error = quad(
        transformed_integrand,
        0.0,
        u_max,
        epsabs=np.finfo(float).tiny,
        epsrel=5.0e-13,
        limit=100,
    )
    return float(value)


def thermal_gap_suppression(T_bath_K: float) -> float:
    """Return ``(Delta_0 - Delta_T) / Delta_0`` from the continuum integral."""

    return float(-math.expm1(-thermal_gap_integral(T_bath_K)))


def fig6_observable(T_bath_K: float, t_star_over_delta: float) -> float:
    """Return Fig. 6 ``(delta Delta_T - delta Delta) / delta Delta_T``."""

    thermal = thermal_gap_suppression(T_bath_K)
    if thermal <= 0.0:
        raise ValueError("T_bath_K is too small to resolve the thermal denominator.")
    driven = eq53_driven_gap_suppression(T_bath_K, t_star_over_delta)
    return float((thermal - driven) / thermal)


def fig6_curve(
    T_bath_K: float,
    t_star_over_delta: Iterable[float] | np.ndarray,
) -> np.ndarray:
    """Evaluate the clean-room analytic Fig. 6 curve on caller-supplied nodes."""

    raw = np.asarray(t_star_over_delta)
    if raw.ndim != 1 or raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("t_star_over_delta must be a one-dimensional real array.")
    ratios = np.asarray(raw, dtype=float)
    if ratios.size < 2 or np.any(~np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError(
            "t_star_over_delta must contain at least two finite non-negative values."
        )
    if np.any(np.diff(ratios) <= 0.0):
        raise ValueError("t_star_over_delta must be strictly increasing.")
    return np.array(
        [fig6_observable(T_bath_K, float(ratio)) for ratio in ratios],
        dtype=float,
    )
