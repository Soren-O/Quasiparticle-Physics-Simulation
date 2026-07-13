r"""Analytic cell weights for integrable pure-BCS gap-edge singularities.

The cell-centered kinetic grid must not sample the divergent BCS density of
states at ``E = Δ``.  Multiplying the DOS evaluated at a cell center by the
cell width consequently under-resolves the finite spectral weight in the first
few cells.  For a cell-constant regular factor, the singular measure is
available exactly:

.. math::

    \int_a^b \frac{E\,dE}{\sqrt{E^2-\Delta^2}}
    = \sqrt{b^2-\Delta^2} - \sqrt{a^2-\Delta^2}.

This module supplies those weights.  It is intentionally limited to pure BCS
spectra; a Dynes DOS has a different, nonsingular measure and should retain a
standard numerical quadrature.
"""

from __future__ import annotations

import numpy as np


def cell_edges_from_widths(
    E_bins: np.ndarray,
    dE_bins: np.ndarray,
) -> np.ndarray:
    """Reconstruct contiguous cell edges from centers and integration widths.

    ``integration_widths_from_centers`` produces exactly this contiguous cell
    partition.  The explicit width input also preserves a caller's chosen
    one-bin fallback width.
    """
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    dE = np.asarray(dE_bins, dtype=float).reshape(-1)
    if E.size == 0:
        raise ValueError("E_bins must be non-empty.")
    if E.shape != dE.shape:
        raise ValueError(
            f"E_bins and dE_bins must have the same shape; got {E.shape} and {dE.shape}."
        )
    if np.any(~np.isfinite(E)) or np.any(~np.isfinite(dE)):
        raise ValueError("E_bins and dE_bins must contain finite values.")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("E_bins must be strictly increasing.")
    if np.any(dE <= 0.0):
        raise ValueError("dE_bins must be positive.")

    edges = np.empty(E.size + 1, dtype=float)
    edges[0] = E[0] - 0.5 * dE[0]
    edges[1:] = edges[0] + np.cumsum(dE)
    return edges


def bcs_dos_cell_weights(
    E_bins: np.ndarray,
    dE_bins: np.ndarray,
    gap: float,
) -> np.ndarray:
    r"""Return exact per-cell weights for the BCS DOS measure ``ρ(E) dE``.

    Each returned weight is

    ``∫_cell E / sqrt(E² - Δ²) dE``

    over the part of that cell above the gap.  Multiplication by a regular
    cell-centered factor gives the finite-volume, cell-constant quadrature.
    The routine does not invent support below the grid's first edge: grids used
    for gap-edge observables should start at ``Δ``.
    """
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap must be finite and non-negative.")
    edges = cell_edges_from_widths(E_bins, dE_bins)
    if gap == 0.0:
        return np.diff(edges)

    lo = np.maximum(edges[:-1], gap)
    hi = np.maximum(edges[1:], gap)
    # Factored form avoids cancellation in E²-Δ² for the first cell.
    xi_lo = np.sqrt(np.maximum((lo - gap) * (lo + gap), 0.0))
    xi_hi = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0))
    return xi_hi - xi_lo
