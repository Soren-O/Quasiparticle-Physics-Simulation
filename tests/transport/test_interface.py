"""Tests for qpsim.transport.interface (Kupriyanov-Lukichev interfaces)."""

from __future__ import annotations

import numpy as np
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.geometries import rectangle, strip
from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.transport.diffusion.base import density_weight
from qpsim.transport.interface import (
    interface_face_conductances,
    interface_face_pairs,
    kl_energy_weight,
)
from qpsim.transport.spatial_operator import (
    apply_face_conductances,
    face_condition_lookup,
    spatial_diffusion_operator,
)

D0 = 6.0
DT = 0.5


class TestInterfaceFaceDiscovery:
    def test_a_one_dimensional_step_is_a_single_face(self):
        gap = np.array([[180.0, 180.0, 240.0, 240.0]])
        pairs = interface_face_pairs(np.ones((1, 4), dtype=bool), gap)
        assert len(pairs) == 1
        p, q, gap_p, gap_q = pairs[0]
        assert (p, q) == (1, 2)
        assert (gap_p, gap_q) == (180.0, 240.0)

    def test_a_two_dimensional_step_is_a_curve_of_faces(self):
        """The generalisation: in 2-D an interface is not one index."""
        mask = np.ones((4, 4), dtype=bool)
        gap = np.where(np.arange(4)[None, :] < 2, 180.0, 240.0)
        gap = np.repeat(gap, 4, axis=0).astype(float)
        pairs = interface_face_pairs(mask, gap)
        # One vertical boundary crossing all four rows.
        assert len(pairs) == 4

    def test_matched_gaps_declare_no_interface(self):
        gap = np.full((3, 3), 180.0)
        assert interface_face_pairs(np.ones((3, 3), dtype=bool), gap) == []

    def test_an_inactive_neighbour_carries_no_interface(self):
        submask = np.array([[True, False, True]])
        gap = np.array([[180.0, 200.0, 240.0]])
        # The cells differ in gap but are not adjacent in the active region.
        assert interface_face_pairs(submask, gap) == []


class TestEnergyWeight:
    def test_matched_gaps_reduce_to_the_support_fraction(self):
        energies, _ = build_energy_grid(
            gap=180.0, energy_min_factor=1.0, energy_max_factor=4.0,
            num_energy_bins=16,
        )
        widths = integration_widths_from_centers(energies)
        weight = kl_energy_weight(energies, widths, 180.0, 180.0)
        assert np.all(weight >= 0.0)
        assert np.all(weight <= 1.0 + 1e-12)

    def test_the_weight_closes_below_the_larger_gap(self):
        energies, _ = build_energy_grid(
            gap=180.0, energy_min_factor=1.0, energy_max_factor=4.0,
            num_energy_bins=24,
        )
        widths = integration_widths_from_centers(energies)
        weight = kl_energy_weight(energies, widths, 180.0, 300.0)
        # Sub-gap on one side means no states there, so the interface is shut.
        assert np.all(weight[energies < 300.0 - widths.max()] == 0.0)

    def test_one_quadrature_per_gap_pair_however_many_faces(self):
        """Faces along one boundary share a gap pair; the weight is not redone.

        A 20-row interface has 20 faces but a single distinct pair.
        """
        rows = 20
        mask = np.ones((rows, 4), dtype=bool)
        gap = np.where(np.arange(4)[None, :] < 2, 180.0, 240.0)
        gap = np.repeat(gap, rows, axis=0).astype(float)
        pairs = interface_face_pairs(mask, gap)
        assert len(pairs) == rows
        assert len({(min(a, b), max(a, b)) for _p, _q, a, b in pairs}) == 1


class TestConservation:
    def test_overriding_a_face_keeps_the_operator_conservative(self):
        geom = rectangle(3, 3)
        n = geom.cell_count
        weights = np.full(n, D0)
        density = np.ones(n)
        faces = face_condition_lookup(geom.edges, geom.conditions())
        plain, _s = spatial_diffusion_operator(
            geom.mask, faces, 1.0, weights, density,
        )
        overridden, _s2 = spatial_diffusion_operator(
            geom.mask, faces, 1.0, weights, density, {(0, 1): 0.123},
        )
        assert not np.allclose(plain.toarray(), overridden.toarray())
        # Reflective edges everywhere: both must still conserve exactly.
        assert np.allclose(overridden.toarray().sum(axis=1), 0.0)

    def test_an_override_equal_to_the_bulk_value_changes_nothing(self):
        geom = strip(4)
        n = geom.cell_count
        weights, density = np.full(n, D0), np.ones(n)
        faces = face_condition_lookup(geom.edges, geom.conditions())
        plain, _s = spatial_diffusion_operator(geom.mask, faces, 1.0, weights, density)
        bulk = float(plain.toarray()[0, 1])
        same, _s2 = spatial_diffusion_operator(
            geom.mask, faces, 1.0, weights, density, {(0, 1): bulk},
        )
        assert np.allclose(plain.toarray(), same.toarray())

    def test_apply_face_conductances_is_a_no_op_when_empty(self):
        geom = strip(4)
        n = geom.cell_count
        laplacian, _s = spatial_diffusion_operator(
            geom.mask, face_condition_lookup(geom.edges, geom.conditions()),
            1.0, np.full(n, D0), np.ones(n),
        )
        assert apply_face_conductances(laplacian, {}) is laplacian


class TestReproducesTheOneDimensionalBackend:
    def test_a_gap_step_interface_matches_bit_for_bit(self):
        material = load_material("Al")
        ne, nx, g_n = 24, 8, 0.7
        gap_left, gap_right = material.Delta_0, 240.0
        energies, _ = build_energy_grid(
            gap=gap_left, energy_min_factor=1.0, energy_max_factor=5.0,
            num_energy_bins=ne,
        )
        widths = integration_widths_from_centers(energies)
        spectral = SpectralContext(
            E_bins=energies, dE_bins=widths, gap=gap_left,
            diffusion_coefficient=D0,
        )
        x = np.linspace(0.0, 70.0, nx)
        dx = float(x[1] - x[0])
        gap_profile = np.where(
            np.arange(nx) < nx // 2, gap_left, gap_right,
        ).astype(float)
        f0 = np.repeat(
            (0.1 * np.exp(-((energies - 2 * gap_left) / 40.0) ** 2))[:, None],
            nx, axis=1,
        )
        state = T3Spatial1DState(
            f=f0, x=x, gap=gap_left, spectral=spectral, material=material,
            T_bath=0.1, gap_profile=gap_profile, interface_conductance=g_n,
        )
        backend = T3Spatial1DBackend()
        expected_ops = backend._build_transport_operators(state, DT)
        n1 = backend._n1_per_cell(state)
        support = backend._support_fraction_per_cell(state)

        geom = strip(nx, mesh_size=dx)
        faces = face_condition_lookup(geom.edges, geom.conditions())
        gap_grid = gap_profile.reshape(1, nx)

        compared = 0
        for i, op in enumerate(expected_ops):
            if op is None:
                continue
            b_expected, _lu, idx, _rho, n_substeps = op
            submask = np.zeros((1, nx), dtype=bool)
            submask[0, idx] = True
            overrides = interface_face_conductances(
                submask, gap_grid, energies, widths, g_n, dx, i,
            )
            operator, _source = spatial_diffusion_operator(
                submask, faces, dx,
                D0 * support[i, idx],          # A1: q == 0
                density_weight(n1[i, idx], 1),
                overrides,
            )
            b_got, _lu2 = build_cn_operators(operator, DT / n_substeps, 1.0)
            assert np.array_equal(b_got.toarray(), b_expected.toarray())
            compared += 1
        assert compared > 0, "no energy bin carried an operator"
