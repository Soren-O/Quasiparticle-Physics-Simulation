"""Tests for qpsim.transport.spatial_transport.

The gate: a full transport step on a one-cell-wide geometry must reproduce
``T3Spatial1DBackend.apply_transport`` bit for bit. That backend already
carries validated harmonic face weights and monotonicity subcycling, so
matching it is a stronger statement than any single analytic check.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.geometries import rectangle, strip
from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import (
    DiffusionModel,
    density_weight,
    flux_weight,
)
from qpsim.transport.spatial_operator import face_condition_lookup
from qpsim.transport.spatial_transport import (
    _MAX_OPERATOR_SETS,
    SpatialTransport,
)

D0 = 6.0
DT = 0.5


def _problem(ne: int = 28, nx: int = 11, seed: int = 7):
    material = load_material("Al")
    energies, _ = build_energy_grid(
        gap=material.Delta_0, energy_min_factor=1.01,
        energy_max_factor=5.0, num_energy_bins=ne,
    )
    spectral = SpectralContext(
        E_bins=energies,
        dE_bins=integration_widths_from_centers(energies),
        gap=material.Delta_0,
        diffusion_coefficient=D0,
    )
    x = np.linspace(0.0, 100.0, nx)
    thermal = 1.0 / (np.exp(np.minimum(energies / (8.617333262e1 * 0.1), 500.0)) + 1.0)
    rng = np.random.default_rng(seed)
    f0 = np.clip(
        np.repeat(thermal[:, None], nx, axis=1) + rng.uniform(0, 3e-4, (ne, nx)),
        0.0, 1.0,
    )
    return material, spectral, x, f0


class TestReproducesTheOneDimensionalBackend:
    @pytest.mark.parametrize("model", list(DiffusionModel))
    def test_a_full_transport_step_matches_bit_for_bit(self, model):
        material, spectral, x, f0 = _problem()
        dx = float(x[1] - x[0])
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=material.Delta_0, spectral=spectral,
            material=material, T_bath=0.1, diffusion_model=model,
        )
        backend = T3Spatial1DBackend()
        expected = backend.apply_transport(state, DT)

        n1 = backend._n1_per_cell(state)
        support = backend._support_fraction_per_cell(state)
        # q == 0 is the dirty-limit indicator: its exact finite-volume average
        # is the supported fraction of a cut cell, not merely "has capacity".
        weights = D0 * support if model.q == 0 else flux_weight(D0, n1, model.q)
        density = density_weight(n1, model.p)

        geom = strip(f0.shape[1], mesh_size=dx)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), dx,
        )
        got, _diagnostics = transport.apply(
            f0.copy(), transport.build(weights, density, DT),
        )
        assert np.array_equal(got, expected.f)

    def test_a_single_cell_is_a_no_op(self):
        geom = rectangle(1, 1)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        f = np.array([[0.3], [0.4]])
        weights = np.full((2, 1), D0)
        density = np.ones((2, 1))
        got, _d = transport.apply(f.copy(), transport.build(weights, density, DT))
        assert np.array_equal(got, f)


class TestOperatorCache:
    def test_repeated_builds_reuse_the_same_operators(self):
        geom = strip(6)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        weights = np.full((3, 6), D0)
        density = np.ones((3, 6))
        first = transport.build(weights, density, DT)
        assert transport.build(weights, density, DT) is first

    def test_the_cache_is_bounded(self):
        geom = strip(6)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        density = np.ones((3, 6))
        for k in range(_MAX_OPERATOR_SETS + 3):
            transport.build(np.full((3, 6), D0 + k), density, DT)
        # Each entry owns one factorization per energy bin; unbounded growth
        # is what makes a 2-D sweep run out of memory.
        assert len(transport._cache) <= _MAX_OPERATOR_SETS

    def test_a_different_active_region_is_not_served_from_cache(self):
        """The mask must be part of the key.

        An operator built for one active region carries flux across cells that
        another region has no states in, and nothing downstream would notice.
        """
        geom = strip(6)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        density = np.ones((1, 6))
        full = np.full((1, 6), D0)
        holed = full.copy()
        holed[0, 3] = 0.0            # one cell has no states at this energy
        ops_full = transport.build(full, density, DT)
        ops_holed = transport.build(holed, density, DT)
        assert ops_full is not ops_holed
        assert ops_full[0][2].size == 6
        assert ops_holed[0][2].size == 5


class TestGeneralGeometry:
    def test_a_non_contiguous_active_region_steps(self):
        geom = strip(5)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        weights = np.array([[D0, D0, 0.0, D0, D0]])
        density = np.ones((1, 5))
        f = np.array([[0.5, 0.1, 0.0, 0.1, 0.5]])
        got, _d = transport.apply(f.copy(), transport.build(weights, density, DT))
        # The dead cell keeps its value: no states, no transport into it.
        assert got[0, 2] == f[0, 2]
        # Each pocket relaxes internally, so its two cells move toward each other.
        assert got[0, 0] < f[0, 0]
        assert got[0, 1] > f[0, 1]

    def test_two_dimensional_transport_spreads_in_both_directions(self):
        geom = rectangle(5, 5)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        n = geom.cell_count
        weights = np.full((1, n), D0)
        density = np.ones((1, n))
        f = np.zeros((1, n))
        f[0, 12] = 1.0                          # a spike at the centre cell
        got, _d = transport.apply(f.copy(), transport.build(weights, density, DT))
        grid = got[0].reshape(5, 5)
        # All four neighbours of the centre receive density, which a 1-D chain
        # laid out row-major could not do.
        assert grid[1, 2] > 0.0 and grid[3, 2] > 0.0
        assert grid[2, 1] > 0.0 and grid[2, 3] > 0.0
        assert np.isclose(grid[1, 2], grid[2, 1])

    def test_conservation_holds_under_reflective_edges(self):
        geom = rectangle(4, 4)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        n = geom.cell_count
        rng = np.random.default_rng(3)
        f = rng.uniform(0.05, 0.4, (1, n))
        weights, density = np.full((1, n), D0), np.ones((1, n))
        got, _d = transport.apply(f.copy(), transport.build(weights, density, DT))
        assert np.isclose(got.sum(), f.sum(), rtol=0, atol=1e-12)


class TestInhomogeneousBoundary:
    def test_a_dirichlet_edge_drives_the_solution_toward_its_value(self):
        """The capability the 1-D backend has no path for at all."""
        geom = strip(6)
        conditions = geom.conditions()
        # The left end of the strip is held at a raised occupation.
        left = next(e for e in geom.edges if e.faces[0].direction == "left")
        conditions[left.edge_id] = BoundaryCondition("dirichlet", 0.8)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, conditions), 1.0,
        )
        weights, density = np.full((1, 6), D0), np.ones((1, 6))
        f = np.zeros((1, 6))
        got, _d = transport.apply(f.copy(), transport.build(weights, density, DT))
        assert got[0, 0] > 0.0, "the held edge injected nothing"
        assert got[0, 0] > got[0, -1], "density did not fall away from the edge"

    def test_reflective_edges_produce_no_forcing(self):
        geom = strip(6)
        transport = SpatialTransport(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()), 1.0,
        )
        ops = transport.build(np.full((1, 6), D0), np.ones((1, 6)), DT)
        assert np.array_equal(ops[0][5], np.zeros(6))
