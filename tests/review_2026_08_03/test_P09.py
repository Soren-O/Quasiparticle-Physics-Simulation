# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review 2026-08-03, packet P09: frozen-xi gap-edge escape classification."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import _remap_bcs_frozen_xi_cell_mass
from qpsim.physics.bcs_quadrature import cell_edges_from_widths


def _grid(n_cells: int = 60, first_edge: float = 180.0, width: float = 9.0):
    dE = np.full(n_cells, width)
    E = first_edge + 0.5 * width + width * np.arange(n_cells)
    return E, dE


def _occupation(E: np.ndarray) -> np.ndarray:
    return 1e-4 * np.exp(-(E - E[0]) / 40.0)


def test_gap_below_the_first_cell_face_is_rejected_not_deposited_at_e_max() -> None:
    """A new gap under the first face strands gap-edge mass; refuse it.

    ``bcs_dos_cell_weights`` still accepts a gap up to 128 ulp below the first
    cell face, and inside that window the frozen-xi sweep cannot visit the old
    mass in ``xi`` in ``[0, xi_new(edges[0])]``.  That mass leaves at the gap
    edge, so it must not be reported as (or relocated to) an ``E_max`` tail.
    """
    E, dE = _grid()
    edges = cell_edges_from_widths(E, dE)
    old_gap = float(edges[0])
    new_gap = old_gap * (1.0 - 30.0 * np.finfo(float).eps)
    assert new_gap < old_gap

    with pytest.raises(ValueError, match="does not cover the updated BCS gap edge"):
        _remap_bcs_frozen_xi_cell_mass(_occupation(E), E, dE, old_gap, new_gap)


def _xi_edges(edges: np.ndarray, gap: float) -> np.ndarray:
    xi = np.zeros_like(edges)
    above = edges > gap
    xi[above] = np.sqrt((edges[above] - gap) * (edges[above] + gap))
    return xi


def test_covered_falling_gap_loses_no_mass() -> None:
    """The guard is inert whenever the grid covers the new gap edge."""
    E, dE = _grid(first_edge=100.0)
    edges = cell_edges_from_widths(E, dE)
    f_old = _occupation(E)
    old_gap = 180.0
    new_gap = 150.0

    new_mass, escaped_mass = _remap_bcs_frozen_xi_cell_mass(f_old, E, dE, old_gap, new_gap)

    old_mass = float(np.diff(_xi_edges(edges, old_gap)) @ f_old)
    assert escaped_mass == 0.0
    assert float(np.sum(new_mass)) == pytest.approx(old_mass, rel=1e-14, abs=0.0)


def test_rising_gap_still_reports_a_finite_e_max_escape() -> None:
    """A rising gap keeps the genuine high-end residual the caller deposits."""
    E, dE = _grid(first_edge=100.0)
    edges = cell_edges_from_widths(E, dE)
    f_old = _occupation(E)
    old_gap = 150.0
    new_gap = 180.0

    new_mass, escaped_mass = _remap_bcs_frozen_xi_cell_mass(f_old, E, dE, old_gap, new_gap)

    old_xi = _xi_edges(edges, old_gap)
    new_xi = _xi_edges(edges, new_gap)
    # Everything above the new xi_max leaves through the finite E_max face.
    tail = np.clip(old_xi, float(new_xi[-1]), None)
    expected_escape = float(np.diff(tail) @ f_old)
    assert escaped_mass > 0.0
    assert escaped_mass == pytest.approx(expected_escape, rel=1e-8, abs=0.0)
    assert float(np.sum(new_mass)) + escaped_mass == pytest.approx(
        float(np.diff(old_xi) @ f_old), rel=1e-14, abs=0.0
    )
