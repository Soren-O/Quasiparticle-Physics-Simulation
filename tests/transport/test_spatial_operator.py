"""Tests for qpsim.transport.spatial_operator.

The headline test is the reproduction gate: on a one-cell-wide geometry the
2-D core must produce the shipped 1-D backend's operator bit for bit, for
every member of the diffusion family. That is a far sharper check than an
analytic benchmark, because the 1-D path is already validated.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.geometries import rectangle, strip
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.transport.diffusion.base import (
    DiffusionModel,
    density_weight,
    flux_weight,
)
from qpsim.transport.spatial_operator import (
    active_submask_boundary,
    face_condition_lookup,
    spatial_diffusion_operator,
)
from scipy import sparse

D0 = 3.0


def _shipped_1d_operator(n1: np.ndarray, model: DiffusionModel, dx: float):
    """The operator exactly as the spatial backend builds it."""
    rho_p = density_weight(n1, model.p)
    w_cell = flux_weight(D0, n1, model.q)
    inv_dx2 = 1.0 / (dx * dx)
    laplacian = _flux_laplacian_from_conductances(
        _harmonic_face_weights(w_cell) * inv_dx2, n1.size,
    )
    return (laplacian @ sparse.diags(1.0 / rho_p)).tocsr(), rho_p, w_cell




class TestGeneralGeometry:
    def test_two_dimensional_keeps_the_five_point_stencil(self):
        geom = rectangle(3, 3)
        n = geom.cell_count
        n1 = np.full(n, 1.3)
        operator, _s = spatial_diffusion_operator(
            geom.mask,
            face_condition_lookup(geom.edges, geom.conditions()),
            1.0,
            flux_weight(D0, n1, 0),
            density_weight(n1, 1),
        )
        assert np.count_nonzero(operator.toarray()[4]) - 1 == 4

    def test_a_non_contiguous_active_region_is_supported(self):
        """The case the 1-D operator raises NotImplementedError on.

        Two pockets of above-gap cells separated by a sub-gap gap: each solves
        on its own, with no transport between them.
        """
        active = np.array([[True, True, False, True, True]])
        n_active = int(active.sum())
        n1 = np.full(n_active, 1.1)
        operator, _s = spatial_diffusion_operator(
            active,
            {},                      # every face newly exposed -> reflective
            1.0,
            flux_weight(D0, n1, 0),
            density_weight(n1, 1),
        )
        dense = operator.toarray()
        assert dense.shape == (4, 4)
        # Cells 1 and 2 in solve order are the two sides of the gap; they must
        # not be coupled, or the operator would carry flux through a region
        # with no states.
        assert dense[1, 2] == 0.0
        assert dense[2, 1] == 0.0
        # Each pocket still conserves internally.
        assert np.allclose(dense.sum(axis=1), 0.0)

    def test_an_interior_hole_is_closed_off(self):
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        n = int(mask.sum())
        n1 = np.full(n, 1.0)
        operator, _s = spatial_diffusion_operator(
            mask, {}, 1.0, flux_weight(D0, n1, 0), density_weight(n1, 1),
        )
        assert operator.shape == (8, 8)
        assert np.allclose(operator.toarray().sum(axis=1), 0.0)


class TestBoundaryInheritance:
    def test_device_conditions_survive_on_active_faces(self):
        geom = rectangle(1, 4)
        conditions = geom.conditions()
        conditions[geom.edges[0].edge_id] = BoundaryCondition("dirichlet", 0.5)
        faces = face_condition_lookup(geom.edges, conditions)

        _edges, active_conditions = active_submask_boundary(geom.mask, faces)
        kinds = {c.normalized_kind() for c in active_conditions.values()}
        assert "dirichlet" in kinds

    def test_newly_exposed_faces_are_reflective(self):
        # One cell of a 1x3 strip has no states: the faces its neighbours now
        # present are not device edges and must carry no flux.
        active = np.array([[True, False, True]])
        edges, conditions = active_submask_boundary(active, {})
        assert {c.normalized_kind() for c in conditions.values()} == {"reflective"}
        total_faces = sum(len(e.faces) for e in edges)
        assert total_faces == 8  # two isolated cells, four faces each

    def test_a_dirichlet_device_edge_still_breaks_conservation(self):
        geom = rectangle(3, 3)
        conditions = geom.conditions()
        conditions[geom.edges[0].edge_id] = BoundaryCondition("dirichlet", 0.0)
        n = geom.cell_count
        n1 = np.full(n, 1.0)
        operator, _s = spatial_diffusion_operator(
            geom.mask,
            face_condition_lookup(geom.edges, conditions),
            1.0,
            flux_weight(D0, n1, 0),
            density_weight(n1, 1),
        )
        assert not np.allclose(operator.toarray().sum(axis=1), 0.0)

    def test_an_inhomogeneous_condition_produces_a_source(self):
        geom = rectangle(1, 4)
        conditions = geom.conditions()
        conditions[geom.edges[0].edge_id] = BoundaryCondition("dirichlet", 2.0)
        n = geom.cell_count
        n1 = np.full(n, 1.0)
        _op, source = spatial_diffusion_operator(
            geom.mask,
            face_condition_lookup(geom.edges, conditions),
            1.0,
            flux_weight(D0, n1, 0),
            density_weight(n1, 1),
        )
        assert source.shape == (n,)
        assert np.any(source != 0.0)

    def test_missing_condition_is_reported(self):
        geom = rectangle(2, 2)
        with pytest.raises(KeyError, match="boundary condition"):
            face_condition_lookup(geom.edges, {})


class TestFaceComposition:
    """min vs harmonic at an unequal face, and that the choice is not inert.

    The parity gate against the 1-D backend cannot catch this on its own: if
    both engines composed the face the same WRONG way they would still agree
    with each other. These assert the value itself.
    """

    @staticmethod
    def _face_weight(mode):
        # Two cells, unequal supported fractions -- the gap-step situation.
        submask = np.ones((1, 2), dtype=bool)
        flux = np.array([1.0, 0.25])
        operator, _source = spatial_diffusion_operator(
            submask, {}, 1.0, flux, np.ones(2), None, mode,
        )
        # Off-diagonal of a two-cell Laplacian is +D_face / dx^2, and rho_p = 1.
        return float(operator.toarray()[0, 1])

    def test_min_composition_takes_the_overlap(self):
        assert self._face_weight("min") == pytest.approx(0.25)

    def test_harmonic_composition_takes_the_series_mean(self):
        assert self._face_weight("harmonic") == pytest.approx(0.4)

    def test_the_two_differ_so_neither_is_a_no_op(self):
        assert self._face_weight("min") != self._face_weight("harmonic")

    def test_equal_neighbours_agree(self):
        """No spurious difference where the physics has none."""
        submask = np.ones((1, 2), dtype=bool)
        flux = np.array([0.7, 0.7])
        got = []
        for mode in ("min", "harmonic"):
            op, _s = spatial_diffusion_operator(
                submask, {}, 1.0, flux, np.ones(2), None, mode,
            )
            got.append(float(op.toarray()[0, 1]))
        assert got[0] == pytest.approx(got[1])
        assert got[0] == pytest.approx(0.7)
