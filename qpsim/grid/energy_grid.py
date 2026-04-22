"""Cell-centered energy grid for quasiparticle occupations.

Provides ``build_energy_grid`` (uniform bins between scaled-gap limits)
and ``integration_widths_from_centers`` (midpoint widths for a
strictly increasing centers array). Used across the kinetic,
gap-equation, and observable pipelines.

Ported from the old ``qpsim/numerics/grid.py`` at Gate 2.
"""

from __future__ import annotations

import numpy as np


def build_energy_grid(
    gap: float,
    energy_min_factor: float,
    energy_max_factor: float,
    num_energy_bins: int,
) -> tuple[np.ndarray, float]:
    """Build a cell-centered energy grid and its uniform bin width.

    Parameters
    ----------
    gap
        Superconducting gap Δ (μeV). Sets the energy scale.
    energy_min_factor, energy_max_factor
        Grid spans ``E ∈ [energy_min_factor·Δ, energy_max_factor·Δ]``.
    num_energy_bins
        Number of cell centers (NE).

    Returns
    -------
    E_bins
        Shape ``(NE,)``. Cell centers.
    dE
        Uniform bin width. For ``num_energy_bins == 1`` returns the
        sentinel ``1.0`` (no well-defined spacing from a single bin).
    """
    if gap <= 0:
        raise ValueError("gap must be positive.")
    if num_energy_bins <= 0:
        raise ValueError("num_energy_bins must be >= 1.")

    E_min = energy_min_factor * gap
    E_max = energy_max_factor * gap
    if num_energy_bins == 1:
        center = 0.5 * (E_min + E_max)
        return np.array([center], dtype=float), 1.0
    if E_max <= E_min:
        raise ValueError(
            "energy_max_factor must be > energy_min_factor for num_energy_bins > 1."
        )

    dE = (E_max - E_min) / float(num_energy_bins)
    E_bins = E_min + (np.arange(num_energy_bins, dtype=float) + 0.5) * dE
    return E_bins, dE


def integration_widths_from_centers(
    centers: np.ndarray,
    *,
    fallback_width: float = 1.0,
) -> np.ndarray:
    """Return integration widths for a strictly-increasing centers array.

    Uses midpoint edges: cell ``i`` extends from ``(c_{i−1}+c_i)/2`` to
    ``(c_i+c_{i+1})/2``; the two boundary cells mirror the nearest
    interior spacing. A length-1 input returns ``[fallback_width]``.
    """
    bins = np.asarray(centers, dtype=float).reshape(-1)
    if bins.size == 0:
        raise ValueError("centers must be non-empty.")
    if bins.size == 1:
        return np.array([float(fallback_width)], dtype=float)
    if np.any(~np.isfinite(bins)):
        raise ValueError("centers must contain finite values.")
    if np.any(np.diff(bins) <= 0):
        raise ValueError("centers must be strictly increasing.")
    edges = np.empty(bins.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (bins[:-1] + bins[1:])
    edges[0] = bins[0] - 0.5 * (bins[1] - bins[0])
    edges[-1] = bins[-1] + 0.5 * (bins[-1] - bins[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("Derived non-positive integration width from centers.")
    return widths
