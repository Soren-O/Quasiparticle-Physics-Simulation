"""Tests for qpsim.backends.t3_spatial, the dimension-agnostic backend.

The gate: on a one-cell-wide geometry every composed step must reproduce
``T3Spatial1DBackend`` bit for bit, including under a gap step with a
Kupriyanov-Lukichev interface and over many accumulated steps.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.geometries import rectangle, strip
from qpsim.grid.energy_grid import (
    build_energy_grid,
    integration_widths_from_centers,
)
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel

D0 = 6.0
DT = 0.05
T_BATH = 0.15


def _setup(ne: int = 24):
    material = load_material("Al")
    energies, _ = build_energy_grid(
        gap=material.Delta_0, energy_min_factor=1.0,
        energy_max_factor=5.0, num_energy_bins=ne,
    )
    spectral = SpectralContext(
        E_bins=energies,
        dE_bins=integration_widths_from_centers(energies),
        gap=material.Delta_0,
        diffusion_coefficient=D0,
    )
    return material, spectral


def _occupations(spectral, material, ncells, seed=5):
    rng = np.random.default_rng(seed)
    profile = 0.05 * np.exp(
        -((spectral.E - 2 * material.Delta_0) / 50.0) ** 2
    )
    return np.clip(
        profile[:, None] * np.ones((1, ncells))
        + rng.uniform(0, 1e-3, (spectral.E.size, ncells)),
        0.0, 1.0,
    )


def _run_both(ncells, gap_profile, conductance, model, steps):
    material, spectral = _setup()
    x = np.linspace(0.0, 60.0, ncells)
    dx = float(x[1] - x[0])
    f0 = _occupations(spectral, material, ncells)

    legacy = T3Spatial1DState(
        f=f0.copy(), x=x, gap=material.Delta_0, spectral=spectral,
        material=material, T_bath=T_BATH, gap_profile=gap_profile,
        interface_conductance=conductance, diffusion_model=model,
    )
    unified = T3SpatialState(
        f=f0.copy(), geometry=strip(ncells, mesh_size=dx), spectral=spectral,
        material=material, T_bath=T_BATH, gap_per_cell=gap_profile,
        interface_conductance=conductance, diffusion_model=model,
    )
    legacy_backend, unified_backend = T3Spatial1DBackend(), T3SpatialBackend()
    for _ in range(steps):
        legacy = legacy_backend.step(legacy, DT)
        unified = unified_backend.step(unified, DT)
    return legacy.f, unified.f


class TestReproducesTheOneDimensionalBackend:
    @pytest.mark.parametrize("model", list(DiffusionModel))
    def test_a_composed_step_matches_bit_for_bit(self, model):
        expected, got = _run_both(9, None, None, model, steps=1)
        assert np.array_equal(got, expected)

    def test_a_gap_step_with_an_interface_matches(self):
        material, _spectral = _setup()
        gap_profile = np.where(
            np.arange(8) < 4, material.Delta_0, 235.0,
        ).astype(float)
        expected, got = _run_both(
            8, gap_profile, 0.7, DiffusionModel.A1, steps=1,
        )
        assert np.array_equal(got, expected)

    def test_many_steps_do_not_drift(self):
        """Accumulated error would show here even if one step matched."""
        expected, got = _run_both(9, None, None, DiffusionModel.A1, steps=10)
        assert np.array_equal(got, expected)


class TestDimensionality:
    def test_zero_dimensional_has_no_transport_but_still_collides(self):
        material, spectral = _setup(ne=16)
        f0 = _occupations(spectral, material, 1)
        state = T3SpatialState(
            f=f0.copy(), geometry=rectangle(1, 1), spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        backend = T3SpatialBackend()
        transported = backend.apply_transport(state, DT)
        assert np.array_equal(transported.f, f0)      # nothing to transport
        stepped = backend.step(state, DT)
        assert not np.array_equal(stepped.f, f0)      # collisions still act

    def test_two_dimensional_spreads_across_both_axes(self):
        material, spectral = _setup(ne=16)
        geom = rectangle(5, 5)
        f0 = np.zeros((spectral.E.size, geom.cell_count))
        peak = int(np.argmax(spectral.cell_density > 0))
        f0[peak, 12] = 0.5                            # a spike at the centre
        state = T3SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        got = T3SpatialBackend().apply_transport(state, DT).f
        grid = got[peak].reshape(5, 5)
        assert grid[1, 2] > 0.0 and grid[3, 2] > 0.0
        assert grid[2, 1] > 0.0 and grid[2, 3] > 0.0
        assert np.isclose(grid[1, 2], grid[2, 1])

    def test_an_interior_hole_is_solved_around(self):
        material, spectral = _setup(ne=16)
        mask = np.ones((4, 4), dtype=bool)
        mask[1, 1] = False
        from qpsim.geometries import Geometry, extract_edge_segments
        geom = Geometry("holed", mask, extract_edge_segments(mask))
        f0 = _occupations(spectral, material, geom.cell_count)
        state = T3SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        got = T3SpatialBackend().step(state, DT).f
        assert got.shape == f0.shape
        assert np.all(got >= 0.0) and np.all(got <= 1.0)


class TestTermSwitches:
    def test_collision_channels_can_be_switched_off(self):
        material, spectral = _setup(ne=16)
        geom = strip(5)
        f0 = _occupations(spectral, material, geom.cell_count)
        state = T3SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        both = T3SpatialBackend().step(state, DT).f
        neither = T3SpatialBackend(
            enable_scattering=False, enable_recombination=False,
        ).step(state, DT).f
        assert not np.array_equal(both, neither)


class TestBoundaryConditions:
    def test_a_dirichlet_edge_injects(self):
        """Capability the 1-D backend has no path for at all."""
        material, spectral = _setup(ne=16)
        geom = strip(6)
        conditions = geom.conditions()
        left = next(e for e in geom.edges if e.faces[0].direction == "left")
        conditions[left.edge_id] = BoundaryCondition("dirichlet", 0.4)
        f0 = np.zeros((spectral.E.size, geom.cell_count))
        state = T3SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH, conditions=conditions,
        )
        got = T3SpatialBackend().apply_transport(state, DT).f
        supported = spectral.cell_density > 0
        assert np.any(got[supported, 0] > 0.0)


class TestResidualAndRun:
    def test_thermal_is_a_fixed_point(self):
        from qpsim.observables.gap_suppression import fermi_dirac_distribution
        material, spectral = _setup(ne=20)
        geom = strip(9, mesh_size=10.0)
        f0 = np.repeat(
            fermi_dirac_distribution(spectral.E, T_BATH)[:, None],
            geom.cell_count, axis=1,
        )
        state = T3SpatialState(
            f=f0, geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        residual = float(np.max(np.abs(T3SpatialBackend().rates(state))))
        assert residual < 1e-18

    def test_the_residual_is_not_a_finite_difference(self):
        """It must see a nonzero operator even where f has stopped moving.

        A step whose occupation is pinned at a clip bound stops changing while
        its governing operator is still nonzero. A finite-difference proxy
        would call that converged; the endpoint residual does not.
        """
        material, spectral = _setup(ne=16)
        geom = strip(5, mesh_size=10.0)
        f0 = np.zeros((spectral.E.size, geom.cell_count))
        supported = spectral.cell_density > 0
        f0[supported, 0] = 1.0            # saturated at the upper clip bound
        state = T3SpatialState(
            f=f0, geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        assert float(np.max(np.abs(T3SpatialBackend().rates(state)))) > 0.0

    def test_a_hot_spot_relaxes_and_the_residual_falls(self):
        from qpsim.observables.gap_suppression import fermi_dirac_distribution
        material, spectral = _setup(ne=20)
        geom = rectangle(8, 8, mesh_size=10.0)
        f0 = np.repeat(
            fermi_dirac_distribution(spectral.E, T_BATH)[:, None],
            geom.cell_count, axis=1,
        )
        peak = int(np.argmax(spectral.cell_density > 0))
        f0[peak, 27] += 1e-3
        backend = T3SpatialBackend()
        state = T3SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        before = float(np.max(np.abs(backend.rates(state))))
        out, n_steps, _converged, after = backend.run(
            state, dt=1.0, max_time=200.0, stop_tol=1e-14,
        )
        assert n_steps > 0
        assert after < before
        grid = out.f[peak].reshape(8, 8)
        assert grid[3, 3] < f0[peak, 27]          # the spike drained
        assert grid[3, 2] > f0[peak, 26]          # a neighbour filled
        assert np.isclose(grid[3, 2], grid[2, 3])  # symmetric in both axes

    def test_a_non_positive_step_is_refused(self):
        material, spectral = _setup(ne=16)
        geom = strip(4)
        state = T3SpatialState(
            f=_occupations(spectral, material, geom.cell_count),
            geometry=geom, spectral=spectral, material=material, T_bath=T_BATH,
        )
        with pytest.raises(ValueError, match="dt must be positive"):
            T3SpatialBackend().run(state, dt=0.0, max_time=1.0)


class TestFeatureParityWithTheStrip:
    """Gap regions and injection, the two things spatial_1d had and 2D lacked."""

    def _driven_step(self, conductance, max_time=300.0):
        from qpsim.webui import execute
        from qpsim.webui.schemas import Spatial2DSetup
        setup = Spatial2DSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 32
        setup.dt = 2.0
        setup.max_time = max_time
        setup.geometry.rows = 6
        setup.geometry.cols = 12
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_right = 240.0
        setup.gap_regions.interface_G_N = conductance
        setup.injection.enabled = True     # a steady flux ACROSS the step
        payload = execute.execute_setup(setup, lambda *a, **k: None, lambda: False)
        profile = payload.arrays["xqp_profile"].reshape(6, 12)
        return float(profile[:, :6].mean()) / float(profile[:, 6:].mean())

    def test_a_closing_barrier_impedes_transport_monotonically(self):
        """The barrier only shows when something drives a flux across it.

        At equilibrium the two sides sit at their own local thermal values and
        nothing crosses, so every conductance gives the same answer. Comparing
        there measures nothing -- the test has to drive the interface.
        """
        ratios = [self._driven_step(g) for g in (10.0, 1.0, 0.1, 0.01)]
        assert ratios == sorted(ratios), f"not monotonic: {ratios}"
        assert ratios[-1] > 5.0 * ratios[0], (
            f"a near-opaque barrier barely changed the profile: {ratios}"
        )

    def test_injection_raises_the_population(self):
        from qpsim.webui import execute
        from qpsim.webui.schemas import Spatial2DSetup
        results = []
        for enabled in (False, True):
            setup = Spatial2DSetup()
            setup.T_bath = 0.2
            setup.grid.num_bins = 32
            setup.dt = 2.0
            setup.max_time = 120.0
            setup.geometry.rows = 8
            setup.geometry.cols = 8
            setup.injection.enabled = enabled
            payload = execute.execute_setup(
                setup, lambda *a, **k: None, lambda: False,
            )
            results.append(payload.summary["x_qp_mean"])
        assert results[1] > 5.0 * results[0]

    def test_a_column_step_gives_two_distinct_gaps(self):
        from qpsim.webui.builders import build_state_2d
        from qpsim.webui.schemas import Spatial2DSetup
        setup = Spatial2DSetup()
        setup.geometry.rows = 4
        setup.geometry.cols = 8
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_right = 240.0
        state = build_state_2d(setup)
        assert sorted(np.unique(state.gaps())) == [180.0, 240.0]


class TestPhononPopulationSurvivesAGapChange:
    """The evolved phonon population is state, not a cached kernel.

    ``SpatialCollisions`` is cache-keyed on the gap profile *and* owns
    ``n_ph``. Rebuilding it for a new gap therefore used to re-seed the
    phonons at the bath, so a self-consistent solve -- which moves the gap on
    every step -- silently discarded phonon trapping, the entire reason the
    dynamic sector exists. A gap change of one part in 10^6 was enough.
    """

    @staticmethod
    def _backend_and_state(rows: int = 4, cols: int = 4):
        from qpsim.webui.builders import build_state_2d
        from qpsim.webui.schemas import Spatial2DSetup
        setup = Spatial2DSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 48
        setup.geometry.rows, setup.geometry.cols = rows, cols
        setup.phonons.mode = "dynamic_escape"
        state = build_state_2d(setup)
        backend = T3SpatialBackend(phonon_escape_time=setup.phonons.tau_l_ns)
        return backend, state

    def test_a_nudged_gap_does_not_reset_the_phonons(self):
        from dataclasses import replace
        backend, state = self._backend_and_state()
        collisions = backend._collisions_for(state)
        assert collisions.n_ph is not None, "phonons frozen — the test is inert"

        marked = collisions.n_ph.copy()
        marked += 1.0            # unmistakably away from any thermal seed
        collisions.n_ph = marked

        gaps = state.gaps().copy()
        gaps[0] *= 1.000001      # what one self-consistent iteration does
        moved = replace(state, gap_per_cell=gaps)

        rebuilt = backend._collisions_for(moved)
        assert rebuilt is not collisions, "the kernel cache did not rebuild"
        np.testing.assert_array_equal(rebuilt.n_ph, marked)

    def test_relax_gap_does_not_reset_the_phonons(self):
        backend, state = self._backend_and_state()
        collisions = backend._collisions_for(state)
        marked = collisions.n_ph + 1.0
        collisions.n_ph = marked

        relaxed, _, change = backend.relax_gap(state, max_iter=1)
        assert change > 0.0, "the gap did not move — the test is inert"
        np.testing.assert_array_equal(backend._collisions_for(relaxed).n_ph, marked)

    def test_a_changed_energy_grid_reseeds_instead(self):
        """No correspondence between the two omega grids, so carrying is wrong."""
        from dataclasses import replace
        backend, state = self._backend_and_state()
        collisions = backend._collisions_for(state)
        collisions.n_ph = collisions.n_ph + 1.0

        _, coarser = _setup(ne=32)
        rebuilt = backend._collisions_for(replace(state, spectral=coarser))
        assert rebuilt.omega_bins.shape != collisions.omega_bins.shape
        assert not np.array_equal(rebuilt.n_ph, collisions.n_ph)
