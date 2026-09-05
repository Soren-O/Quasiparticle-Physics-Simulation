"""Kupriyanov-Lukichev interfaces on an arbitrary geometry.

Where the gap steps between neighbouring cells, the face carries a finite
interface conductance instead of the bulk harmonic-mean weight. The energy
channel is

    F = g_N (N_1^L N_1^R - N_2^L N_2^R) (f_L - f_R),

whose coherence-factor (Maki-Griffin) weight equals
``(E^2 - D_L D_R) / (Omega_L Omega_R)``: positive above both gaps, regular at
matched gaps, and automatically closed where one side is sub-gap.

Two properties follow from the geometry being a mask.

*An interface is a set of faces, not a face index.* On a 2-D mask a gap step
is a curve, so the faces are found by comparing each pair of adjacent cells;
on a one-cell-wide mask that curve is the single face between cells ``k`` and
``k+1``.

*The weight is cached per GAP PAIR, not per face.* It depends only on
``(E, dE, gap_L, gap_R)``, and every face along one boundary between two
regions shares that. Recomputing the quadrature per face would be the dominant
cost along a curve of hundreds of faces, for no benefit.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from qpsim.physics.bcs_quadrature import cell_edges_from_widths

__all__ = [
    "interface_face_conductances",
    "interface_face_pairs",
    "kl_energy_weight",
]

_DIRECTIONS: tuple[tuple[int, int], ...] = ((1, 0), (0, 1))  # down, right


def kl_energy_weight(
    E: np.ndarray,
    dE: np.ndarray,
    gap_left: float,
    gap_right: float,
) -> np.ndarray:
    r"""Cell-average KL energy weight at a two-gap interface.

    Integrates ``N1_L*N1_R - N2_L*N2_R`` over the represented part of each
    energy cell. A xi substitution for the larger gap removes the integrable
    endpoint singularity. Matched gaps are handled analytically, preserving the
    exact cancellation to the above-gap support fraction.
    """
    edges = cell_edges_from_widths(E, dE)
    high = max(float(gap_left), float(gap_right))
    low = min(float(gap_left), float(gap_right))
    lo = np.maximum(edges[:-1], high)
    hi = np.maximum(edges[1:], lo)
    result = np.zeros_like(E)

    scale = max(1.0, high, low)
    if abs(high - low) <= 64.0 * np.finfo(float).eps * scale:
        return (hi - lo) / dE
    if high == 0.0:
        return (hi - lo) / dE

    delta_sq = high * high - low * low

    def transformed(xi: float) -> float:
        energy = np.sqrt(high * high + xi * xi)
        return (energy * energy - high * low) / (
            energy * np.sqrt(xi * xi + delta_sq)
        )

    for i in np.flatnonzero(hi > lo):
        xi_lo = np.sqrt(max((lo[i] - high) * (lo[i] + high), 0.0))
        xi_hi = np.sqrt(max((hi[i] - high) * (hi[i] + high), 0.0))
        integral, _error = quad(
            transformed, float(xi_lo), float(xi_hi),
            epsabs=1e-13 * max(1.0, float(dE[i])), epsrel=1e-12, limit=100,
        )
        result[i] = integral / dE[i]
    return result


def interface_face_pairs(
    submask: np.ndarray,
    gap_grid: np.ndarray,
) -> list[tuple[int, int, float, float]]:
    """Adjacent active cells whose gaps differ, as solve-index pairs.

    ``gap_grid`` is the local gap on the full mask grid, same shape as
    ``submask``. Returns ``(p, q, gap_p, gap_q)`` with ``p < q`` in solve
    order, one entry per interface face. Each direction is visited once, so no
    face is reported twice.
    """
    submask = np.asarray(submask, dtype=bool)
    nrow, ncol = submask.shape
    index_map = -np.ones(submask.shape, dtype=int)
    coords = np.argwhere(submask)
    for idx, (row, col) in enumerate(coords):
        index_map[row, col] = idx

    pairs: list[tuple[int, int, float, float]] = []
    for row, col in coords:
        row, col = int(row), int(col)
        for dr, dc in _DIRECTIONS:
            r, c = row + dr, col + dc
            if not (0 <= r < nrow and 0 <= c < ncol) or not submask[r, c]:
                continue
            gap_p = float(gap_grid[row, col])
            gap_q = float(gap_grid[r, c])
            if gap_p != gap_q:
                p, q = int(index_map[row, col]), int(index_map[r, c])
                pairs.append((min(p, q), max(p, q), gap_p, gap_q))
    return pairs


def interface_face_conductances(
    submask: np.ndarray,
    gap_grid: np.ndarray,
    E: np.ndarray,
    dE: np.ndarray,
    conductance: float,
    dx: float,
    energy_index: int,
) -> dict[tuple[int, int], float]:
    """Face conductance override for one energy bin.

    ``g_N * weight(E) / dx`` per interface face -- ``1/dx`` rather than
    ``1/dx^2`` because the KL flux is itself dx-independent and only the
    flux-divergence factor remains.

    The quadrature runs once per distinct gap pair however many faces share
    it, which is the whole reason this is keyed on the pair.
    """
    pairs = interface_face_pairs(submask, gap_grid)
    if not pairs:
        return {}

    weights: dict[tuple[float, float], np.ndarray] = {}
    overrides: dict[tuple[int, int], float] = {}
    for p, q, gap_p, gap_q in pairs:
        key = (min(gap_p, gap_q), max(gap_p, gap_q))
        weight = weights.get(key)
        if weight is None:
            weight = kl_energy_weight(E, dE, key[0], key[1])
            weights[key] = weight
        overrides[(p, q)] = float(conductance) * float(weight[energy_index]) / dx
    return overrides
