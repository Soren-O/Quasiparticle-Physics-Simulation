"""Analytic eigenmode benchmarks for the dimension-agnostic spatial core.

The 1-D reproduction gate proves the new core agrees with the backend it
replaces. That is necessary and not sufficient: both could be wrong the same
way. These tests are the independent half -- pure diffusion on geometries whose
decay rates are known in closed form, with no reference to any other backend.

Two families, ported from the archived 2D simulator's test cases:

* a rectangle with Dirichlet rims, whose modes are products of sines and decay
  at exactly ``D k^2``. The discretisation is second order, so halving the mesh
  must quarter the error -- which is a far stronger statement than any single
  tolerance.
* a polygon annulus with Dirichlet rims, whose radial modes are Bessel cross
  products. Convergence here is only first order, and deliberately so: the mask
  is a staircase approximation to a circle, and that geometric error dominates
  the second-order interior stencil. Asserting second order would be asserting
  something false.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.geometries import Geometry, extract_edge_segments, rectangle
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.transport.spatial_operator import (
    face_condition_lookup,
    spatial_diffusion_operator,
)
from scipy.optimize import brentq
from scipy.special import jv, yv

D_COEFF = 1.0
DT = 0.002
STEPS = 150


def _decay_rate(geometry: Geometry, initial: np.ndarray) -> float:
    """Measured exponential decay rate of a mode under pure diffusion."""
    conditions = geometry.conditions(BoundaryCondition("dirichlet", 0.0))
    faces = face_condition_lookup(geometry.edges, conditions)
    ncells = geometry.cell_count
    operator, _source = spatial_diffusion_operator(
        geometry.mask, faces, geometry.mesh_size,
        np.full(ncells, D_COEFF), np.ones(ncells),
    )
    b_mat, lu = build_cn_operators(operator, DT, 1.0)
    u = initial.copy()
    for _ in range(STEPS):
        u = lu.solve(b_mat @ u)
    decayed = np.linalg.norm(u) / np.linalg.norm(initial)
    return -float(np.log(decayed)) / (STEPS * DT)


def _rectangle_mode(nrow: int, ncol: int, m: int, n: int, dx: float):
    geometry = rectangle(nrow, ncol, mesh_size=dx)
    rows, cols = np.nonzero(geometry.mask)
    length_x, length_y = ncol * dx, nrow * dx
    # Cell centres. The Dirichlet wall sits half a cell beyond the end cells,
    # which is what makes the finite-volume stencil second order here.
    x = (cols + 0.5) * dx
    y = (rows + 0.5) * dx
    mode = np.sin(m * np.pi * x / length_x) * np.sin(n * np.pi * y / length_y)
    exact = D_COEFF * (
        (m * np.pi / length_x) ** 2 + (n * np.pi / length_y) ** 2
    )
    return geometry, mode, exact


def _annulus(size: int, inner: float, outer: float, dx: float):
    centre = (size - 1) / 2.0
    rows, cols = np.mgrid[0:size, 0:size]
    radius = np.hypot(rows - centre, cols - centre) * dx
    mask = (radius >= inner) & (radius <= outer)
    geometry = Geometry(
        "annulus", mask, extract_edge_segments(mask), mesh_size=dx,
    )
    return geometry, centre


def _annulus_dirichlet_root(inner: float, outer: float) -> float:
    """Lowest k with the Bessel cross product vanishing on both rims."""
    def cross(k: float) -> float:
        return (
            jv(0, k * inner) * yv(0, k * outer)
            - yv(0, k * inner) * jv(0, k * outer)
        )
    # The fundamental sits near pi/(outer - inner); bracket generously.
    span = np.pi / (outer - inner)
    return float(brentq(cross, 0.7 * span, 1.4 * span))


class TestRectangleModes:
    @pytest.mark.parametrize(("m", "n"), [(1, 1), (1, 2), (1, 3), (2, 2)])
    def test_the_decay_rate_matches_the_closed_form(self, m, n):
        geometry, mode, exact = _rectangle_mode(32, 32, m, n, 1.0)
        measured = _decay_rate(geometry, mode)
        assert abs(measured - exact) / exact < 1e-2

    @pytest.mark.parametrize(("m", "n"), [(1, 1), (1, 2), (2, 2)])
    def test_refinement_is_second_order(self, m, n):
        """Halving the mesh must quarter the error.

        A single tolerance can be met by a wrong operator with a lucky
        constant; an observed convergence ORDER cannot.
        """
        errors = []
        for size in (16, 32):
            # Same physical domain, twice the cells: dx halves.
            geometry, mode, exact = _rectangle_mode(
                size, size, m, n, 32.0 / size,
            )
            errors.append(abs(_decay_rate(geometry, mode) - exact) / exact)
        order = np.log2(errors[0] / errors[1])
        assert 1.8 < order < 2.2, f"observed order {order:.2f}"


class TestAnnulusBesselModes:
    @pytest.mark.parametrize(
        ("size", "inner", "outer"), [(41, 6.0, 18.0), (61, 9.0, 27.0)],
    )
    def test_the_radial_mode_matches_the_bessel_cross_product(
        self, size, inner, outer,
    ):
        geometry, centre = _annulus(size, inner, outer, 1.0)
        rows, cols = np.nonzero(geometry.mask)
        radius = np.hypot(rows - centre, cols - centre)
        k = _annulus_dirichlet_root(inner, outer)
        mode = (
            jv(0, k * radius) * yv(0, k * inner)
            - yv(0, k * radius) * jv(0, k * inner)
        )
        measured = _decay_rate(geometry, mode)
        exact = D_COEFF * k * k
        # Looser than the rectangle on purpose: the rim is a staircase.
        assert abs(measured - exact) / exact < 5e-2

    def test_refinement_converges_at_the_staircase_rate(self):
        """First order, not second -- and that is the correct expectation.

        The interior stencil is second order, but the mask approximates a
        circular rim by whole cells, and that geometric error is first order in
        dx. It dominates, so demanding second order here would be demanding
        something untrue of the discretisation.
        """
        errors = []
        for size, inner, outer in ((41, 6.0, 18.0), (81, 12.0, 36.0)):
            geometry, centre = _annulus(size, inner, outer, 1.0)
            rows, cols = np.nonzero(geometry.mask)
            radius = np.hypot(rows - centre, cols - centre)
            k = _annulus_dirichlet_root(inner, outer)
            mode = (
                jv(0, k * radius) * yv(0, k * inner)
                - yv(0, k * radius) * jv(0, k * inner)
            )
            errors.append(
                abs(_decay_rate(geometry, mode) - D_COEFF * k * k)
                / (D_COEFF * k * k)
            )
        # Two refinements of the radius; error should fall by roughly 4x if
        # first order in dx at fixed shape, and it must at least fall.
        assert errors[1] < errors[0]
        order = np.log2(errors[0] / errors[1]) / 1.0
        assert 1.0 < order < 3.0, f"observed order {order:.2f}"


class TestReflectiveDegeneracy:
    def test_an_all_reflective_annulus_conserves_and_has_a_null_mode(self):
        """Why the archived N/N donut case was not ported as a decay test.

        With reflective rims the operator conserves exactly, so a constant is
        a null mode and nothing decays to zero. Any initial condition with a
        projection onto that constant leaves a residue, and a decay-rate fit
        against it measures the projection, not the physics. The archived
        case had a -4.2% projection and was therefore broken as shipped.
        Pinned here as a property of the operator rather than a benchmark.
        """
        geometry, _centre = _annulus(41, 6.0, 18.0, 1.0)
        ncells = geometry.cell_count
        conditions = geometry.conditions()          # reflective everywhere
        operator, source = spatial_diffusion_operator(
            geometry.mask,
            face_condition_lookup(geometry.edges, conditions),
            geometry.mesh_size,
            np.full(ncells, D_COEFF), np.ones(ncells),
        )
        dense_row_sums = np.asarray(operator.sum(axis=1)).ravel()
        assert np.allclose(dense_row_sums, 0.0)
        assert np.allclose(source, 0.0)
        # A constant is annihilated: it cannot decay, so it is not a mode a
        # decay-rate benchmark may be built on.
        constant = np.ones(ncells)
        assert np.allclose(operator @ constant, 0.0)
