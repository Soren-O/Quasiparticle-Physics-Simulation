"""Tests for analytic pure-BCS gap-edge cell weights."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)


def test_weights_sum_to_exact_bcs_primitive_on_nonuniform_grid() -> None:
    gap = 1.0
    E = np.array([1.25, 1.8, 2.9, 4.5])
    dE = integration_widths_from_centers(E)
    edges = cell_edges_from_widths(E, dE)

    weights = bcs_dos_cell_weights(E, dE, gap)
    exact = np.sqrt(edges[-1] ** 2 - gap**2) - np.sqrt(
        max(edges[0] ** 2 - gap**2, 0.0)
    )
    assert np.sum(weights) == pytest.approx(exact, rel=1e-14)


def test_cell_crossing_gap_integrates_only_above_gap() -> None:
    E = np.array([0.9, 1.1])
    dE = np.array([0.2, 0.2])
    weights = bcs_dos_cell_weights(E, dE, gap=1.0)

    assert weights[0] == 0.0
    assert weights[1] == pytest.approx(np.sqrt(1.2**2 - 1.0), rel=1e-14)


def test_zero_gap_reduces_to_ordinary_cell_widths() -> None:
    E = np.array([1.0, 2.0, 3.0])
    dE = np.ones(3)
    np.testing.assert_allclose(bcs_dos_cell_weights(E, dE, gap=0.0), dE)


@pytest.mark.parametrize("gap", [float("nan"), float("inf"), -1.0])
def test_rejects_invalid_gap(gap: float) -> None:
    with pytest.raises(ValueError, match="gap"):
        bcs_dos_cell_weights(np.array([1.0]), np.array([1.0]), gap)
