"""Tests for qpsim.physics.gap_equation — BCS calibration and runtime solve."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.gap_equation import calibrate_gap, solve_gap


class TestCalibrateGap:
    def test_bcs_zero_T_ratio(self) -> None:
        # Δ(0) / (kB T_c) ≈ 1.764 for BCS.
        cal = calibrate_gap(T_c=1.2, T_bath=0.0)
        assert cal.delta_eq / (KB_UEV_PER_K * 1.2) == pytest.approx(1.764, rel=1e-4)

    def test_normal_state_above_Tc(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=2.0)
        assert cal.delta_eq == 0.0

    def test_gap_monotonically_decreases_with_T(self) -> None:
        cals = [calibrate_gap(T_c=1.2, T_bath=T) for T in (0.0, 0.3, 0.6, 0.9, 1.1)]
        deltas = [c.delta_eq for c in cals]
        for a, b in pairwise(deltas):
            assert a >= b - 1e-12

    def test_rejects_non_positive_Tc(self) -> None:
        with pytest.raises(ValueError, match="T_c must be positive"):
            calibrate_gap(T_c=0.0, T_bath=0.5)

    def test_rejects_negative_Tbath(self) -> None:
        with pytest.raises(ValueError, match="T_bath must be non-negative"):
            calibrate_gap(T_c=1.2, T_bath=-0.1)


class TestSolveGap:
    def test_equilibrium_roundtrip(self) -> None:
        # Feeding in the thermal Fermi-Dirac occupation f_FD(E, T_bath)
        # must reproduce Δ_eq(T_bath) to high accuracy.
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        # Grid spans past ω_D = 100·kB·T_c so the integrand is captured.
        omega_D = 100.0 * KB_UEV_PER_K * T_c
        E = np.linspace(cal.delta_eq * 1.001, omega_D * 1.01, 3000)
        kT = KB_UEV_PER_K * T_bath
        f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        delta = solve_gap(cal, f, E)
        assert delta == pytest.approx(cal.delta_eq, rel=1e-3)

    def test_normal_state_returns_zero(self) -> None:
        cal = calibrate_gap(T_c=1.2, T_bath=2.0)  # Δ_eq = 0
        E = np.linspace(0.1, 5.0, 100)
        f = np.zeros_like(E)
        assert solve_gap(cal, f, E) == 0.0

    def test_extreme_nonequilibrium_gives_normal_state(self) -> None:
        # f ≈ 1 everywhere above the gap pushes (1 − 2f) < 0, so the
        # gap integral cannot equal the positive reference integral.
        # Expect the solver to detect this and return 0.
        T_c, T_bath = 1.2, 0.3
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        E = np.linspace(cal.delta_eq * 1.001, cal.delta_eq * 60.0, 2000)
        f = np.ones_like(E)
        assert solve_gap(cal, f, E) == 0.0
