"""Analytical envelope helpers for Fischer 2023 figure overlays.

Plotting-side port of the formulas in
``paper reproductions/fischer2023-repro/src/fischer2023/{kinetic,solver}.py``:
Eqs. (24), (25a), (25b), (28), (30/31), (35), (36/37), (38), (41), (47), (48),
(49), (51), (E2). Used only to overlay analytical envelopes/asymptotes on
the dashed-curve side of paper-style validation plots — no qpsim backend
dependency, no kinetic solve.

References paper-repro implementations 1:1; see those files for the
equation numbers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import airy, kv

KB_UEV_PER_K = 86.17333262145178  # μeV per K


@dataclass(frozen=True)
class EnvelopeParams:
    Delta0: float          # μeV
    Tc_uev: float          # μeV (Tc * kB)
    tau0: float            # ns
    tau0_PB: float         # ns
    tau_l: float           # ns
    TB_uev: float          # μeV (TB * kB)
    nbar: float
    omega0: float          # μeV
    cphot_QP: float        # 1/ns


# --- Energy scales ---------------------------------------------------------

def Tstar(p: EnvelopeParams) -> float:
    """Eq. 24 / Eq. 35: kB T* in μeV."""
    rhs = (105.0 / 64.0) * p.Tc_uev**3 * p.Delta0 \
        * (p.cphot_QP * p.tau0) * p.nbar * p.omega0**2
    return rhs ** (1.0 / 6.0)


def Tstar_tilde(p: EnvelopeParams) -> float:
    """Eq. 28."""
    Ts = Tstar(p)
    return (64.0 / 35.0 * Ts / p.Delta0) ** (1.0 / 5.0) * Ts


def TB_star_uev(p: EnvelopeParams) -> float:
    """Eq. 35: TB^* in μeV."""
    Ts = Tstar(p)
    return Ts**3 / p.Delta0**2


# --- Envelope pieces (Eqs. 25a, 25b, 31, 38) -------------------------------

def f_low(x: np.ndarray, b0: float) -> np.ndarray:
    return b0 * (1.0 - 0.564 * x**2.5 + 0.119 * x**3.5)


def f_intermediate(x: np.ndarray, b0: float) -> np.ndarray:
    # F&C 2023 Eq. (26) [arXiv:2212.08155]: f = 3 b0 Ai(x^2 / 4^{1/3}) — the
    # cube root applies to the 4 only, NOT to x^2/4. Verified against the
    # published text 2026-07-03 (review item 4, REVIEW-2026-07-02).
    Ai, _, _, _ = airy(x**2 / (4.0 ** (1.0 / 3.0)))
    return 3.0 * b0 * Ai


def f_high(E: np.ndarray, p: EnvelopeParams, b0: float) -> np.ndarray:
    Ts = Tstar(p)
    Tt = Tstar_tilde(p)
    x_tilde = (E - p.Delta0) / Tt
    bt0 = b0 * 3 ** 1.5 * 2 ** (2.0 / 3.0) / (5.0 * np.pi) \
        * (64.0 * Ts / 35.0 / p.Delta0) ** (-2.0 / 5.0)
    arg = 0.4 * x_tilde ** 2.5
    out = np.zeros_like(x_tilde, dtype=float)
    mask = x_tilde > 0
    out[mask] = bt0 * np.sqrt(x_tilde[mask]) * kv(0.2, arg[mask])
    return out


def E_star_estimate(p: EnvelopeParams) -> float:
    if p.TB_uev <= 0:
        return np.inf
    Ts = Tstar(p)
    Tt = Tstar_tilde(p)
    TBs = TB_star_uev(p)
    if p.TB_uev < TBs:
        return p.Delta0 + Tt * (Tt / p.TB_uev) ** (2.0 / 3.0)
    return p.Delta0 + Ts * (Ts / p.TB_uev) ** 0.5


def f_thermal_tail(E: np.ndarray, p: EnvelopeParams, b0: float) -> np.ndarray:
    if p.TB_uev <= 0:
        return np.zeros_like(E)
    E_star = E_star_estimate(p)
    f_at_estar = envelope_no_thermal(np.array([E_star]), p, b0)[0]
    bT = f_at_estar * np.exp(E_star / p.TB_uev)
    return bT * np.exp(-E / p.TB_uev)


def envelope_no_thermal(E: np.ndarray, p: EnvelopeParams, b0: float) -> np.ndarray:
    Ts = Tstar(p)
    x = (E - p.Delta0) / Ts
    out = np.zeros_like(E, dtype=float)
    hi = (E - p.Delta0) > 0.5 * p.Delta0
    if np.any(hi):
        out[hi] = f_high(E[hi], p, b0)
    mid = (~hi) & (x > 0.8)
    if np.any(mid):
        out[mid] = f_intermediate(x[mid], b0)
    lo = ~(hi | mid)
    if np.any(lo):
        out[lo] = f_low(x[lo], b0)
    return np.maximum(out, 0.0)


def envelope_with_thermal(E: np.ndarray, p: EnvelopeParams, b0: float) -> np.ndarray:
    f_neq = envelope_no_thermal(E, p, b0)
    if p.TB_uev <= 0:
        return f_neq
    f_th = f_thermal_tail(E, p, b0)
    E_star = E_star_estimate(p)
    return np.maximum(np.where(E_star < E, f_th, f_neq), 0.0)


# --- Generalized Rothwarf-Taylor steady-state b0 ---------------------------

def _tau_bar(p: EnvelopeParams) -> float:
    return p.tau0 * (1.0 + p.tau_l / p.tau0_PB)


def G_thermal(p: EnvelopeParams, rhoF: float = 1.0) -> float:
    """Eq. 48."""
    if p.TB_uev <= 0:
        return 0.0
    return (16.0 * np.pi * rhoF / _tau_bar(p)) \
        * (p.Delta0 / p.Tc_uev) ** 3 * p.TB_uev * np.exp(-2.0 * p.Delta0 / p.TB_uev)


def G_drive(p: EnvelopeParams, x: float) -> float:
    """Eq. 51."""
    if p.tau_l <= 0:
        return 0.0
    gamma = 0.84
    coef = (gamma / _tau_bar(p)) * (p.tau_l / p.tau0_PB) \
        * (p.Delta0 / p.Tc_uev) ** 3
    return coef * x ** 4.5 * np.exp(-np.sqrt(14.0 / 5.0) * x ** (-3.0))


def R_bar(p: EnvelopeParams, rhoF: float = 1.0) -> float:
    """Eq. E2."""
    a_m12, a_p12, a_p32 = 2.1, 0.88, 0.77
    eps = Tstar(p) / p.Delta0
    c1 = a_p12 / a_m12
    c2 = 1.25 * (a_p32 / a_m12) - 0.75 * (a_p12 / a_m12) ** 2
    R = 2.0 * p.Delta0 ** 2 / (rhoF * _tau_bar(p) * p.Tc_uev ** 3)
    return R * (1.0 + c1 * eps + c2 * eps ** 2)


def nqp_steady(p: EnvelopeParams, rhoF: float = 1.0) -> float:
    """Eq. 47."""
    Ts = Tstar(p)
    x = Ts / p.Delta0
    G = G_drive(p, x)
    GT = G_thermal(p, rhoF=rhoF)
    R = R_bar(p, rhoF=rhoF)
    return (G + np.sqrt(G * G + 4.0 * R * GT)) / (2.0 * R)


def solve_b0(p: EnvelopeParams, rhoF: float = 1.0) -> float:
    """Eq. 41 inverted on the steady-state Eq. 47 density."""
    Ts = Tstar(p)
    NQP = nqp_steady(p, rhoF=rhoF)
    return NQP / (4.2 * rhoF * np.sqrt(2.0 * p.Delta0 * Ts))
