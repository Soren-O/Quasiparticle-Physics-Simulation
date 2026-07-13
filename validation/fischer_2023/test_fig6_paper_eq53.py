"""Fast tests for the Fischer Eq. 53 dashed-overlay chain in :mod:`fig6_paper`.

Pins:
* :func:`fig6_paper._paper_eq53_analytic_drive` against the verified
  closed-form bracketed factor :math:`1 - 0.42\\,r + 0.22\\,r^2`.
* :func:`fig6_paper._paper_eq53_analytic_thermal` against the Sommerfeld
  counterpart :math:`1 - \\tfrac{1}{2}\\,r + \\tfrac{3}{8}\\,r^2`.
* The full chain ``_xqp_analytic_eq47`` → ``_paper_eq53_analytic_drive``
  at four (T_B, n_bar) points spanning the paper's plotted T_*/Δ range.
  These pins guard against the dashed overlay silently regressing toward
  the numerical x_qp again.

The slow regression test in :mod:`test_fig6_paper` still pins the full
sweep CSV; this test just gives the closed-form chain coverage that does
not require running the ~1-hour Picard + self-consistent BCS gap solve.
"""

from __future__ import annotations

import pytest
from qpsim.constants import KB_UEV_PER_K

from validation.fischer_2023.fig5_paper import _kBTstar_eq35, _xqp_analytic_eq47
from validation.fischer_2023.fig6_solve import (
    DELTA_0,
    _build_grid_and_spectral,
    _paper_eq53_analytic_drive,
    _paper_eq53_analytic_thermal,
    _x_qp_thermal,
)


def test_drive_bracket_at_unit_xqp() -> None:
    """At x_qp = 1, drive overlay returns the bracketed factor."""
    for r in (0.0, 0.25, 0.5, 0.75):
        expected = 1.0 - 0.42 * r + 0.22 * r * r
        assert _paper_eq53_analytic_drive(1.0, r) == pytest.approx(expected, rel=1e-15)


def test_drive_overlay_zero_at_zero_xqp() -> None:
    """No quasiparticles → no gap suppression, regardless of T_*/Δ."""
    for r in (0.0, 0.3, 0.7):
        assert _paper_eq53_analytic_drive(0.0, r) == 0.0


def test_thermal_bracket_at_unit_xqp() -> None:
    """At x_qp_th = 1, thermal counterpart returns Sommerfeld bracket."""
    for r in (0.0, 0.05, 0.10):
        expected = 1.0 - 0.5 * r + (3.0 / 8.0) * r * r
        assert _paper_eq53_analytic_thermal(1.0, r) == pytest.approx(expected, rel=1e-15)


# Pinned (T_bath, n_bar, tau_l, tau_0_pb, expected_x_qp_eq47, expected_T_star_over_Delta,
#  expected_delta_drive_over_Delta_0). Computed at full float64 precision once and
# pinned here. If these drift, either Eq. 47 (fig5) or Eq. 53 coefficients changed.
_DRIVE_CHAIN_PIN = (
    (0.10, 1e6, 0.368, 0.255,
     2.1390538027163498e-10, 0.34257088191155877, 1.886513447865939e-10),
    (0.10, 1e8, 0.368, 0.255,
     0.00040445020425378006, 0.7380465917850781, 0.0003275469266247145),
    (0.20, 1e7, 0.368, 0.255,
     9.935729897452216e-06, 0.5028252895784324, 8.390094948630524e-06),
    (0.15, 5e7, 0.368, 0.255,
     4.502280913067455e-05, 0.6575247625491724, 3.687161587210294e-05),
)


@pytest.mark.parametrize(
    "T_bath, n_bar, tau_l, tau_0_pb, expected_x, expected_r, expected_dr",
    _DRIVE_CHAIN_PIN,
)
def test_drive_chain_eq47_then_eq53(
    T_bath: float, n_bar: float, tau_l: float, tau_0_pb: float,
    expected_x: float, expected_r: float, expected_dr: float,
) -> None:
    """Eq. 47 → Eq. 53 closed-form chain reproduces pinned float64 values."""
    x = _xqp_analytic_eq47(T_bath, n_bar, tau_l=tau_l, tau_0_pb=tau_0_pb)
    r = _kBTstar_eq35(n_bar) / DELTA_0
    dr = _paper_eq53_analytic_drive(x, r)
    assert x == pytest.approx(expected_x, rel=1e-12)
    assert r == pytest.approx(expected_r, rel=1e-12)
    assert dr == pytest.approx(expected_dr, rel=1e-12)


def test_thermal_chain_xqp_th_then_eq53() -> None:
    """Thermal x_qp → Eq. 53 thermal counterpart reproduces pinned values.

    Pins three T_B values spanning the paper's plotted T_B set
    {0.10, 0.15, 0.20} K (uses 0.05 / 0.10 / 0.20 to bracket the range
    and exercise the deeply-cold tail).
    """
    _, _, spectral = _build_grid_and_spectral()
    pins = (
        (0.05, 1.3744248789767633e-19, 0.02393703683929167, 1.3582703696270156e-19),
        (0.10, 2.346672204655683e-10, 0.04787407367858334, 2.2925167261017608e-10),
        (0.20, 1.1652298123216673e-05, 0.09574814735716668, 1.1134514468148642e-05),
    )
    for T_bath, expected_xth, expected_rb, expected_dt in pins:
        xth = _x_qp_thermal(spectral, T_bath)
        rb = KB_UEV_PER_K * T_bath / DELTA_0
        dt = _paper_eq53_analytic_thermal(xth, rb)
        # x_qp_thermal uses the qp_fraction integral over the 1620-bin grid;
        # algebra-form drift at ~1e-12 relative is the float64 floor.
        assert xth == pytest.approx(expected_xth, rel=1e-12, abs=0.0)
        assert rb == pytest.approx(expected_rb, rel=1e-15)
        assert dt == pytest.approx(expected_dt, rel=1e-12, abs=0.0)
