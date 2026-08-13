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
    # Widths alone define a contiguous partition only if every nominal
    # center actually belongs to the corresponding reconstructed cell.
    # Without this check, an unrelated/inconsistent width vector can shift
    # the cells away from their samples and silently assign exact BCS
    # spectral weight to the wrong occupation value.  Midpoint-derived
    # finite-volume grids need not place a center at the geometric midpoint
    # on a nonuniform mesh, but they do always place it inside its cell.
    scale = np.maximum.reduce([
        np.ones(E.size),
        np.abs(E),
        np.abs(edges[:-1]),
        np.abs(edges[1:]),
    ])
    tol = 128.0 * np.finfo(float).eps * scale
    outside = (edges[:-1] - tol > E) | (edges[1:] + tol < E)
    if np.any(outside):
        first = int(np.flatnonzero(outside)[0])
        raise ValueError(
            "E_bins and dE_bins describe inconsistent cell geometry: "
            f"center E_bins[{first}]={E[first]:g} is outside reconstructed "
            f"cell [{edges[first]:g}, {edges[first + 1]:g}]."
        )
    return edges


def bcs_dos_cell_weights(
    E_bins: np.ndarray,
    dE_bins: np.ndarray,
    gap: float,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> np.ndarray:
    r"""Return exact per-cell weights for a bounded BCS DOS measure.

    Each returned weight is

    ``∫_(cell ∩ [lower_bound, upper_bound]) E / sqrt(E² - Δ²) dE``

    over the part of that cell above the gap.  Multiplication by a regular
    cell-centered factor gives the finite-volume, cell-constant quadrature.

    The default lower bound is the physical gap and the default upper bound is
    the last grid edge.  A requested lower bound must be covered by the grid's
    first cell edge.  In particular, a grid whose lower edge is above ``Δ``
    is rejected rather than silently dropping the singular interval or
    inventing an occupation there.  This is the pure-BCS grid contract used by
    gap-edge observables and moving-gap finite volumes.

    Explicit bounds also allow a cell crossed by a physical band boundary to
    be split exactly, which is needed for the M25 ``R<``/``R>`` moments.
    """
    if not np.isfinite(gap) or gap < 0.0:
        raise ValueError("gap must be finite and non-negative.")
    edges = cell_edges_from_widths(E_bins, dE_bins)
    if gap == 0.0 and lower_bound is None and upper_bound is None:
        return np.diff(edges)

    requested_lo = gap if lower_bound is None else float(lower_bound)
    requested_hi = edges[-1] if upper_bound is None else float(upper_bound)
    if not np.isfinite(requested_lo):
        raise ValueError("lower_bound must be finite.")
    if np.isnan(requested_hi):
        raise ValueError("upper_bound must not be NaN.")

    band_lo = max(gap, requested_lo)
    band_hi = min(float(edges[-1]), requested_hi)
    if upper_bound is not None and requested_hi <= band_lo:
        raise ValueError("upper_bound must be greater than the BCS band lower bound.")

    coverage_tol = 128.0 * np.finfo(float).eps * max(
        1.0, abs(float(edges[0])), abs(band_lo),
    )
    if float(edges[0]) > band_lo + coverage_tol:
        raise ValueError(
            "The energy grid does not cover the requested BCS lower bound: "
            f"first cell edge {float(edges[0]):g} > {band_lo:g}. Start the "
            "grid at or below the gap/band edge; missing singular support "
            "cannot be reconstructed from above-edge samples."
        )

    if band_hi <= band_lo:
        return np.zeros(np.asarray(E_bins).size, dtype=float)

    lo = np.maximum(edges[:-1], band_lo)
    hi = np.minimum(edges[1:], band_hi)
    hi = np.maximum(hi, lo)
    if gap == 0.0:
        return hi - lo

    # Factored form avoids cancellation in E²-Δ² for the first cell.
    xi_lo = np.sqrt(np.maximum((lo - gap) * (lo + gap), 0.0))
    xi_hi = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0))
    return xi_hi - xi_lo


def represented_bcs_weights(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
) -> np.ndarray:
    """Exact BCS capacity on only the represented energy domain.

    Copied out of the 1-D spatial backend, where it was private, so the
    dimension-agnostic core does not import from a module scheduled for
    deletion. The 1-D copy stays put until that deletion, deliberately: it is
    the reference the new core is checked against, so it must not move.
    """
    edges = cell_edges_from_widths(E, dE)
    return bcs_dos_cell_weights(
        E, dE, gap, lower_bound=max(float(gap), float(edges[0])),
    )


def bcs_support_fraction(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
) -> np.ndarray:
    """Fraction of each energy cell lying in the ideal-BCS continuum.

    The dirty-limit flux coefficient is an above-gap indicator, and its exact
    finite-volume average over a cell cut by the gap edge is this fraction --
    not 1 merely because the cell has some capacity.
    """
    edges = cell_edges_from_widths(E, dE)
    overlap = np.maximum(edges[1:] - np.maximum(edges[:-1], float(gap)), 0.0)
    return overlap / dE
