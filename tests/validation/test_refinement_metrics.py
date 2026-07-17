"""Fast tests for measure-aware continuum-refinement diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
from validation.refinement_metrics import (
    bcs_capacity_primitive,
    compare_piecewise_refinement,
)


def _threshold_layer(width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit-mass layer that is uniform on ``[0, width]``."""
    return np.array([0.0, width, 1.0]), np.array([1.0 / width, 0.0])


def _identity_capacity(coordinate: np.ndarray) -> np.ndarray:
    return coordinate


def test_threshold_layer_is_strongly_distinct_but_weakly_convergent() -> None:
    """A narrowing integrable layer need not converge in capacity-L1.

    Each dyadic layer has unit mass.  The strong discrepancy stays exactly one,
    while its normalized W1 shape distance halves with the layer width.  All
    strong error is correctly identified as threshold-localized.
    """
    coarse_edges, coarse_values = _threshold_layer(0.5)
    middle_edges, middle_values = _threshold_layer(0.25)
    fine_edges, fine_values = _threshold_layer(0.125)
    first = compare_piecewise_refinement(
        coarse_edges,
        coarse_values,
        middle_edges,
        middle_values,
        capacity_primitive=_identity_capacity,
        localization_centers=np.array([0.0]),
        localization_half_width=0.5,
    )
    second = compare_piecewise_refinement(
        middle_edges,
        middle_values,
        fine_edges,
        fine_values,
        capacity_primitive=_identity_capacity,
        localization_centers=np.array([0.0]),
        localization_half_width=0.25,
    )

    assert first.coarse_mass == pytest.approx(1.0)
    assert first.fine_mass == pytest.approx(1.0)
    assert first.relative_mass_change == pytest.approx(0.0)
    assert first.relative_capacity_l1 == pytest.approx(1.0)
    assert second.relative_capacity_l1 == pytest.approx(1.0)
    assert first.normalized_wasserstein1 == pytest.approx(0.125)
    assert second.normalized_wasserstein1 == pytest.approx(0.0625)
    assert second.normalized_wasserstein1 == pytest.approx(
        0.5 * first.normalized_wasserstein1
    )
    assert first.localized_l1_fraction == pytest.approx(1.0)
    assert second.localized_l1_fraction == pytest.approx(1.0)


def test_bcs_capacity_coordinate_preserves_exact_layer_metrics() -> None:
    """The BCS energy grid gives the same answer in its natural xi measure."""
    gap = 180.0
    coarse_xi_edges, coarse_values = _threshold_layer(0.5)
    fine_xi_edges, fine_values = _threshold_layer(0.25)
    coarse_energy_edges = np.sqrt(gap * gap + coarse_xi_edges * coarse_xi_edges)
    fine_energy_edges = np.sqrt(gap * gap + fine_xi_edges * fine_xi_edges)
    half_width = float(coarse_energy_edges[1] - gap)

    metrics = compare_piecewise_refinement(
        coarse_energy_edges,
        coarse_values,
        fine_energy_edges,
        fine_values,
        capacity_primitive=lambda energy: bcs_capacity_primitive(energy, gap=gap),
        localization_centers=np.array([gap]),
        localization_half_width=half_width,
    )

    # Mapping an exactly chosen xi edge back through float64 energy and then
    # reconstructing xi loses a few 1e-11 near a 180-ueV gap.
    assert metrics.relative_mass_change < 5e-11
    assert metrics.relative_capacity_l1 == pytest.approx(1.0, abs=5e-11)
    assert metrics.normalized_wasserstein1 == pytest.approx(0.125, abs=5e-11)
    assert metrics.localized_l1_fraction == pytest.approx(1.0)


def test_refinement_comparison_rejects_nonphysical_measure_data() -> None:
    with pytest.raises(ValueError, match="energy must be non-negative"):
        bcs_capacity_primitive(np.array([-181.0]), gap=180.0)

    with pytest.raises(ValueError, match="non-negative"):
        compare_piecewise_refinement(
            np.array([0.0, 1.0]),
            np.array([-1.0]),
            np.array([0.0, 1.0]),
            np.array([1.0]),
            capacity_primitive=lambda coordinate: coordinate,
        )

    with pytest.raises(ValueError, match="same support"):
        compare_piecewise_refinement(
            np.array([0.0, 1.0]),
            np.array([1.0]),
            np.array([0.0, 2.0]),
            np.array([1.0]),
            capacity_primitive=lambda coordinate: coordinate,
        )
