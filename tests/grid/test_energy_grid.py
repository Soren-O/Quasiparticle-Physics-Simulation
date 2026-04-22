"""Tests for qpsim.grid.energy_grid."""

from __future__ import annotations

import numpy as np
import pytest

from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)


class TestBuildEnergyGrid:
    def test_basic_shape_and_spacing(self) -> None:
        E_bins, dE = build_energy_grid(
            gap=1.0, energy_min_factor=1.0, energy_max_factor=5.0, num_energy_bins=4
        )
        assert E_bins.shape == (4,)
        assert dE == pytest.approx(1.0)
        np.testing.assert_allclose(E_bins, [1.5, 2.5, 3.5, 4.5])

    def test_single_bin_returns_center_and_sentinel_width(self) -> None:
        E_bins, dE = build_energy_grid(
            gap=1.0, energy_min_factor=1.0, energy_max_factor=3.0, num_energy_bins=1
        )
        assert E_bins.shape == (1,)
        assert E_bins[0] == pytest.approx(2.0)
        assert dE == 1.0

    def test_scales_with_gap(self) -> None:
        E1, _ = build_energy_grid(
            gap=1.0, energy_min_factor=1.0, energy_max_factor=2.0, num_energy_bins=10
        )
        E2, _ = build_energy_grid(
            gap=2.0, energy_min_factor=1.0, energy_max_factor=2.0, num_energy_bins=10
        )
        np.testing.assert_allclose(E2, 2.0 * E1)

    def test_rejects_non_positive_gap(self) -> None:
        with pytest.raises(ValueError, match="gap must be positive"):
            build_energy_grid(
                gap=0.0, energy_min_factor=1.0, energy_max_factor=2.0, num_energy_bins=4
            )

    def test_rejects_non_positive_bin_count(self) -> None:
        with pytest.raises(ValueError, match="num_energy_bins"):
            build_energy_grid(
                gap=1.0, energy_min_factor=1.0, energy_max_factor=2.0, num_energy_bins=0
            )

    def test_rejects_inverted_range(self) -> None:
        with pytest.raises(ValueError, match="energy_max_factor"):
            build_energy_grid(
                gap=1.0, energy_min_factor=2.0, energy_max_factor=1.0, num_energy_bins=4
            )


class TestIntegrationWidths:
    def test_uniform_grid(self) -> None:
        centers = np.array([1.0, 2.0, 3.0, 4.0])
        widths = integration_widths_from_centers(centers)
        np.testing.assert_allclose(widths, 1.0)

    def test_single_center_returns_fallback(self) -> None:
        widths = integration_widths_from_centers(np.array([5.0]))
        assert widths.shape == (1,)
        assert widths[0] == 1.0

    def test_single_center_accepts_custom_fallback(self) -> None:
        widths = integration_widths_from_centers(np.array([5.0]), fallback_width=2.5)
        assert widths[0] == 2.5

    def test_non_uniform(self) -> None:
        # centers: 0, 1, 3, 6
        # edges: -0.5, 0.5, 2.0, 4.5, 7.5
        # widths: 1.0, 1.5, 2.5, 3.0
        widths = integration_widths_from_centers(np.array([0.0, 1.0, 3.0, 6.0]))
        np.testing.assert_allclose(widths, [1.0, 1.5, 2.5, 3.0])

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            integration_widths_from_centers(np.array([]))

    def test_rejects_non_monotonic(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            integration_widths_from_centers(np.array([1.0, 2.0, 1.5]))
