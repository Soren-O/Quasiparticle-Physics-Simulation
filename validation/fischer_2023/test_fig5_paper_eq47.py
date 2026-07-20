"""Unit tests for the Fischer Eq. 47 + Appendix-E analytic ``x_qp``.

Pins :func:`fig5_paper._xqp_analytic_eq47` against:

* The closed-form thermal limit  x_qp = sqrt(π T_B / (2 Δ)) e^{-Δ/T_B},
  recovered when n̄ = 0 and τ_l = 0 (Eq. 47 reduces to R N² = G_T).
* The standalone reproduction at
  ``paper reproductions/fischer2023-repro/src/fischer2023/solver.py``,
  values pre-computed and pinned here so the test does not depend on
  that external source tree.

These are fast (no PDE solves), unmarked — they run in the default suite.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig5_paper import (
    DELTA_0,
    KB_UEV_PER_K,
    NQP_PER_X_QP_PAPER,
    NQP_PER_X_QP_QPSIM,
    RHOF_AL_uev,
    _nqp_to_x_qp_paper,
    _x_qp_qpsim_to_nqp,
    _xqp_analytic_eq47,
)

# Pinned τ_0^PB matching qpsim's phonon-side extractor and Fischer Table I.
TAU_0_PB_NS = 0.255


def test_thermal_closed_form_TB_010() -> None:
    """At n̄=0, τ_l=0 the balance is R N² = G_T → analytic thermal x_qp."""
    T_bath = 0.10
    TB_uev = KB_UEV_PER_K * T_bath
    expected = np.sqrt(np.pi * TB_uev / (2.0 * DELTA_0)) * np.exp(-DELTA_0 / TB_uev)
    got = _xqp_analytic_eq47(T_bath, 0.0, tau_l=0.0, tau_0_pb=TAU_0_PB_NS)
    assert got == pytest.approx(expected, rel=1e-12)


def test_thermal_closed_form_TB_020() -> None:
    T_bath = 0.20
    TB_uev = KB_UEV_PER_K * T_bath
    expected = np.sqrt(np.pi * TB_uev / (2.0 * DELTA_0)) * np.exp(-DELTA_0 / TB_uev)
    got = _xqp_analytic_eq47(T_bath, 0.0, tau_l=0.0, tau_0_pb=TAU_0_PB_NS)
    assert got == pytest.approx(expected, rel=1e-12)


# (T_B [K], n̄, τ_l [ns]) → x_qp reference values. Originally pinned from
# the standalone repro's nqp_steady (ρ_F=1, tau0_PB=0.255 ns); the τ_l>0
# entries were UPDATED 2026-07-20 to include the finite-τ_l trapping
# correction on R̄'s linear term (paper Eq. 112 leading order) that the
# standalone repro also omitted — the old pins were self-referential to
# the same untrapped helper (external review). τ_l = 0 entries are
# trap-free and unchanged.
_STANDALONE_REFERENCES = [
    ((0.10, 1.0e6, 0.255), 2.172289e-10),
    ((0.10, 1.0e7, 0.255), 7.379332e-09),
    ((0.15, 1.0e7, 0.255), 2.738610e-07),
    ((0.20, 1.0e8, 0.255), 2.959329e-04),
    ((0.10, 1.0e8, 0.000), 1.906886e-10),
    ((0.10, 0.0,   0.000), 2.325461e-10),
]


@pytest.mark.parametrize("inputs,expected", _STANDALONE_REFERENCES)
def test_matches_standalone_reproduction(
    inputs: tuple[float, float, float], expected: float,
) -> None:
    """Eq. 47 algebra ports 1:1 from standalone solver.nqp_steady."""
    T_bath, n_bar, tau_l = inputs
    got = _xqp_analytic_eq47(
        T_bath, n_bar, tau_l=tau_l, tau_0_pb=TAU_0_PB_NS,
    )
    assert got == pytest.approx(expected, rel=2e-6)


def test_drive_dominated_grows_with_nbar() -> None:
    """In the drive-dominated branch (G ≫ G_T), x_qp grows with n̄.

    Note: Eq. 47 + Appendix-E is non-monotonic across the full sweep —
    at very small n̄ the (1 + c₁ε + c₂ε²) recombination correction
    enhances R̄ before G(x) lifts the numerator, so x_qp dips slightly
    below thermal before rising. Standalone reproduction reproduces the
    same shape; this test only pins the post-knee behavior.
    """
    nbars = [1e7, 1e8, 1e9]
    xs = [
        _xqp_analytic_eq47(0.10, n, tau_l=0.255, tau_0_pb=TAU_0_PB_NS)
        for n in nbars
    ]
    assert all(xs[i + 1] > xs[i] for i in range(len(xs) - 1))


def test_higher_TB_increases_thermal_x_qp() -> None:
    """At n̄=0, τ_l=0, x_qp must increase with T_B (thermal QP density grows)."""
    TBs = [0.06, 0.10, 0.15, 0.20]
    xs = [
        _xqp_analytic_eq47(T, 0.0, tau_l=0.0, tau_0_pb=TAU_0_PB_NS) for T in TBs
    ]
    assert all(xs[i + 1] > xs[i] for i in range(len(xs) - 1))


def test_plot_axis_conversions_match_paper_convention() -> None:
    """Fig. 5 plots paper N and N/(2 rho_F Delta), not qpsim's x_qp."""
    x_qpsim = 3.0e-7
    n_qp = _x_qp_qpsim_to_nqp(x_qpsim)

    assert pytest.approx(4.0 * RHOF_AL_uev * DELTA_0) == NQP_PER_X_QP_QPSIM
    assert pytest.approx(2.0 * RHOF_AL_uev * DELTA_0) == NQP_PER_X_QP_PAPER
    assert n_qp == pytest.approx(4.0 * RHOF_AL_uev * DELTA_0 * x_qpsim)
    assert _nqp_to_x_qp_paper(n_qp) == pytest.approx(2.0 * x_qpsim)
