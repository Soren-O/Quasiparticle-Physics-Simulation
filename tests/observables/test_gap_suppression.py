"""Tests for qpsim.observables.gap_suppression."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.gap_suppression import (
    compute_gap_suppression,
    gap_suppression_from_deltas,
)
from qpsim.physics.gap_equation import calibrate_gap


def _thermal_f(
    T_bath: float = 0.3,
    T_c: float = 1.2,
    num: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    delta_0 = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=delta_0,
        energy_min_factor=1.001,
        energy_max_factor=10.0,
        num_energy_bins=num,
    )
    dE = integration_widths_from_centers(E)
    del dE  # grid spacing not needed; helper mirrors observables tests style.
    kT = KB_UEV_PER_K * T_bath
    f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return E, f


class TestGapSuppressionFromDeltas:
    def test_positive_suppression(self) -> None:
        result = gap_suppression_from_deltas(1.0, 0.9)
        assert result.delta_suppression == pytest.approx(0.1)
        assert result.rel_suppression == pytest.approx(0.1)

    def test_negative_suppression_is_allowed(self) -> None:
        result = gap_suppression_from_deltas(1.0, 1.1)
        assert result.delta_suppression == pytest.approx(-0.1)
        assert result.rel_suppression == pytest.approx(-0.1)

    def test_rejects_negative_inputs(self) -> None:
        with pytest.raises(ValueError, match="delta_eq"):
            gap_suppression_from_deltas(-1.0, 0.5)
        with pytest.raises(ValueError, match="delta_final"):
            gap_suppression_from_deltas(1.0, -0.5)


class TestComputeGapSuppression:
    def test_thermal_roundtrip_gives_near_zero_suppression(self) -> None:
        T_c, T_bath = 1.2, 0.3
        E, f = _thermal_f(T_bath=T_bath, T_c=T_c, num=300)
        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)

        result = compute_gap_suppression(f, E, T_c=T_c, T_bath=T_bath)

        assert result.delta_eq == pytest.approx(cal.delta_eq, rel=1e-12)
        assert result.delta_final == pytest.approx(cal.delta_eq, rel=3e-3)
        assert result.rel_suppression == pytest.approx(0.0, abs=3e-3)

    def test_hot_nonequilibrium_suppresses_gap(self) -> None:
        E, _ = _thermal_f(T_bath=0.3, T_c=1.2, num=300)
        # Over-occupy the spectrum relative to thermal equilibrium.
        f_hot = np.full_like(E, 0.35)

        result = compute_gap_suppression(f_hot, E, T_c=1.2, T_bath=0.3)

        assert result.delta_final <= result.delta_eq
        assert result.delta_suppression >= 0.0
