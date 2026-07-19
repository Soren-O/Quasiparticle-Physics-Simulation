"""Measure-aware diagnostics for refinement of cell-constant spectra.

Pointwise interpolation is a poor convergence diagnostic when a represented
measure develops an integrable boundary layer.  This module compares two
cell-constant densities against a caller-supplied cumulative capacity
coordinate.  It reports strong capacity-L1 change, total-mass change, and a
mass-normalized Wasserstein-1 shape distance separately.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RefinementMetrics:
    """Dimensionless diagnostics for two cell-constant represented measures.

    ``relative_capacity_l1`` is the strong discrepancy
    ``integral |f_coarse - f_fine| dC / integral f_fine dC``.
    ``normalized_wasserstein1`` first normalizes each nonnegative measure to
    unit mass, then divides its Wasserstein-1 distance by the full capacity
    span.  ``localized_l1_fraction`` is the fraction of the strong numerator
    inside the requested coordinate windows, or ``None`` when no windows were
    requested.
    """

    coarse_mass: float
    fine_mass: float
    relative_mass_change: float
    relative_capacity_l1: float
    normalized_wasserstein1: float
    localized_l1_fraction: float | None


def bcs_capacity_primitive(energy: np.ndarray, *, gap: float) -> np.ndarray:
    r"""Return the clean-BCS capacity coordinate ``sqrt(E**2 - gap**2)``.

    Energies below the gap map to zero, matching the represented clean-BCS
    density-of-states measure.  The factored radicand avoids cancellation near
    the gap edge.
    """
    E = np.asarray(energy, dtype=float)
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap must be finite and non-negative.")
    if np.any(~np.isfinite(E)):
        raise ValueError("energy must contain only finite values.")
    if np.any(E < 0.0):
        raise ValueError("energy must be non-negative.")
    return np.sqrt(np.maximum((E - gap) * (E + gap), 0.0))


def _validated_piecewise_data(
    edges: np.ndarray,
    values: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    edge_array = np.asarray(edges, dtype=float).reshape(-1)
    value_array = np.asarray(values, dtype=float).reshape(-1)
    if edge_array.size < 2:
        raise ValueError(f"{name}_edges must contain at least two entries.")
    if value_array.size != edge_array.size - 1:
        raise ValueError(
            f"{name}_values must have one entry per cell; got "
            f"{value_array.size} values and {edge_array.size} edges."
        )
    if np.any(~np.isfinite(edge_array)) or np.any(np.diff(edge_array) <= 0.0):
        raise ValueError(f"{name}_edges must be finite and strictly increasing.")
    if np.any(~np.isfinite(value_array)) or np.any(value_array < 0.0):
        raise ValueError(f"{name}_values must be finite and non-negative.")
    return edge_array.copy(), value_array


def _integral_abs_linear(y0: float, y1: float, width: float) -> float:
    """Integrate the absolute value of a linear segment from ``y0`` to ``y1``."""
    if y0 * y1 >= 0.0 or y0 == y1:
        return 0.5 * (abs(y0) + abs(y1)) * width
    slope_magnitude = abs(y1 - y0) / width
    return (y0 * y0 + y1 * y1) / (2.0 * slope_magnitude)


def compare_piecewise_refinement(
    coarse_edges: np.ndarray,
    coarse_values: np.ndarray,
    fine_edges: np.ndarray,
    fine_values: np.ndarray,
    *,
    capacity_primitive: Callable[[np.ndarray], np.ndarray],
    localization_centers: np.ndarray | None = None,
    localization_half_width: float | None = None,
) -> RefinementMetrics:
    """Compare two nonnegative cell-constant spectra on the same support.

    ``capacity_primitive(x)`` supplies the cumulative represented measure
    ``C(x)``.  For clean BCS spectra, pass a closure around
    :func:`bcs_capacity_primitive`.  Localization windows are expressed in the
    original edge coordinate, not in capacity space; this lets photon
    thresholds be specified directly in energy.
    """
    coarse_edge_array, coarse_value_array = _validated_piecewise_data(
        coarse_edges, coarse_values, name="coarse"
    )
    fine_edge_array, fine_value_array = _validated_piecewise_data(
        fine_edges, fine_values, name="fine"
    )

    support_scale = max(
        1.0,
        abs(float(coarse_edge_array[0])),
        abs(float(coarse_edge_array[-1])),
        abs(float(fine_edge_array[0])),
        abs(float(fine_edge_array[-1])),
    )
    support_tolerance = 128.0 * np.finfo(float).eps * support_scale
    if (
        abs(float(coarse_edge_array[0] - fine_edge_array[0])) > support_tolerance
        or abs(float(coarse_edge_array[-1] - fine_edge_array[-1])) > support_tolerance
    ):
        raise ValueError("coarse and fine partitions must cover the same support.")
    coarse_edge_array[[0, -1]] = fine_edge_array[[0, -1]]

    window_centers: np.ndarray | None = None
    window_half_width: float | None = None
    extra_edges = np.empty(0, dtype=float)
    if localization_centers is not None:
        window_centers = np.asarray(localization_centers, dtype=float).reshape(-1)
        if window_centers.size == 0 or np.any(~np.isfinite(window_centers)):
            raise ValueError("localization_centers must be non-empty and finite.")
        if localization_half_width is None:
            raise ValueError(
                "localization_half_width is required with localization_centers."
            )
        window_half_width = float(localization_half_width)
        if not np.isfinite(window_half_width) or window_half_width < 0.0:
            raise ValueError("localization_half_width must be finite and non-negative.")
        boundaries = np.concatenate(
            (window_centers - window_half_width, window_centers + window_half_width)
        )
        extra_edges = boundaries[
            (boundaries > fine_edge_array[0]) & (boundaries < fine_edge_array[-1])
        ]
    elif localization_half_width is not None:
        raise ValueError(
            "localization_centers is required with localization_half_width."
        )

    partition = np.unique(
        np.concatenate((coarse_edge_array, fine_edge_array, extra_edges))
    )
    midpoints = 0.5 * (partition[:-1] + partition[1:])
    coarse_indices = np.searchsorted(coarse_edge_array, midpoints, side="right") - 1
    fine_indices = np.searchsorted(fine_edge_array, midpoints, side="right") - 1
    coarse_on_union = coarse_value_array[coarse_indices]
    fine_on_union = fine_value_array[fine_indices]

    capacity = np.asarray(capacity_primitive(partition), dtype=float)
    if capacity.shape != partition.shape or np.any(~np.isfinite(capacity)):
        raise ValueError(
            "capacity_primitive must return one finite value per input edge."
        )
    capacity_widths = np.diff(capacity)
    capacity_scale = max(1.0, float(np.max(np.abs(capacity))))
    capacity_tolerance = 128.0 * np.finfo(float).eps * capacity_scale
    if np.any(capacity_widths < -capacity_tolerance):
        raise ValueError("capacity_primitive must be non-decreasing on the support.")
    capacity_widths = np.maximum(capacity_widths, 0.0)
    capacity_span = float(capacity[-1] - capacity[0])
    if capacity_span <= 0.0:
        raise ValueError("capacity_primitive must have positive span on the support.")

    coarse_mass = float(np.dot(capacity_widths, coarse_on_union))
    fine_mass = float(np.dot(capacity_widths, fine_on_union))
    if coarse_mass <= 0.0 or fine_mass <= 0.0:
        raise ValueError("both represented measures must have positive mass.")

    absolute_difference = np.abs(coarse_on_union - fine_on_union)
    capacity_l1 = float(np.dot(capacity_widths, absolute_difference))
    relative_capacity_l1 = capacity_l1 / fine_mass

    coarse_probability_density = coarse_on_union / coarse_mass
    fine_probability_density = fine_on_union / fine_mass
    cdf_difference = 0.0
    wasserstein1 = 0.0
    for width, coarse_density, fine_density in zip(
        capacity_widths,
        coarse_probability_density,
        fine_probability_density,
        strict=True,
    ):
        next_cdf_difference = cdf_difference + float(
            width * (coarse_density - fine_density)
        )
        if width > 0.0:
            wasserstein1 += _integral_abs_linear(
                cdf_difference, next_cdf_difference, float(width)
            )
        cdf_difference = next_cdf_difference

    localized_l1_fraction: float | None = None
    if window_centers is not None and window_half_width is not None:
        localized = np.any(
            np.abs(midpoints[:, None] - window_centers[None, :])
            <= window_half_width,
            axis=1,
        )
        localized_l1 = float(
            np.dot(capacity_widths[localized], absolute_difference[localized])
        )
        localized_l1_fraction = localized_l1 / capacity_l1 if capacity_l1 > 0.0 else 0.0

    return RefinementMetrics(
        coarse_mass=coarse_mass,
        fine_mass=fine_mass,
        relative_mass_change=abs(coarse_mass - fine_mass) / fine_mass,
        relative_capacity_l1=relative_capacity_l1,
        normalized_wasserstein1=wasserstein1 / capacity_span,
        localized_l1_fraction=localized_l1_fraction,
    )
