"""Preparing a run somewhere other than equilibrium.

The measurements this unblocks are the relaxation ones: seed an excess and
watch it decay. So the tests here check both that the state is prepared as
stated, and that the resulting dynamics actually go somewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.fields.drive import cell_coordinates
from qpsim.fields.initial import (
    InitialConditionError,
    energy_profile,
    seed_occupation,
    separable_excess,
    spatial_profile,
)
from qpsim.fields.safe_eval import compile_expression
from qpsim.geometries import rectangle, strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.gap_suppression import fermi_dirac_distribution
from qpsim.physics.spectral import SpectralContext

T_BATH = 0.2


def _spectral(ne: int = 24, min_factor: float = 1.0):
    material = load_material("Al")
    energies, _ = build_energy_grid(
        gap=material.Delta_0, energy_min_factor=min_factor,
        energy_max_factor=4.0, num_energy_bins=ne,
    )
    return material, SpectralContext(
        E_bins=energies,
        dE_bins=integration_widths_from_centers(energies),
        gap=material.Delta_0,
        diffusion_coefficient=6.0,
    )


class TestEnergyProfiles:
    def test_every_shape_is_peak_normalised(self):
        """So `amplitude` means the same experiment whichever shape is picked."""
        _m, spectral = _spectral()
        for kind, kw in [
            ("flat", {}),
            ("thermal", {"T_eff": 1.0}),
            ("monoenergetic", {"E_0": 2.2 * 180.0}),
            ("gap_edge", {}),
        ]:
            shape = energy_profile(kind, spectral, **kw)
            assert np.max(shape) == pytest.approx(1.0), kind

    def test_narrowing_a_line_does_not_change_its_height(self):
        """Unit-integral normalisation would; that is why it is not used."""
        _m, spectral = _spectral()
        wide = energy_profile("monoenergetic", spectral, E_0=400.0, width=40.0)
        narrow = energy_profile("monoenergetic", spectral, E_0=400.0, width=10.0)
        assert np.max(wide) == pytest.approx(np.max(narrow))
        assert np.sum(narrow) < np.sum(wide)

    def test_sub_gap_weight_is_removed(self):
        """Occupation below the gap is not a state the solver can carry."""
        _m, spectral = _spectral(min_factor=0.8)
        below = np.asarray(spectral.cell_density) == 0
        assert below.any(), "grid has no sub-gap bins -- the test is inert"
        shape = energy_profile("flat", spectral)
        assert np.all(shape[below] == 0.0)

    def test_a_line_placed_off_the_grid_is_refused_not_silently_empty(self):
        _m, spectral = _spectral()
        with pytest.raises(InitialConditionError, match="zero in every supported"):
            energy_profile("monoenergetic", spectral, E_0=1e6, width=1.0)

    def test_an_expression_profile_sees_the_energy_and_the_gap(self):
        _m, spectral = _spectral()
        fn = compile_expression("np.exp(-(E - 2 * gap) ** 2 / 1e4)",
                                variables=("E", "gap"))
        shape = energy_profile("expression", spectral, expression=fn)
        assert np.max(shape) == pytest.approx(1.0)
        assert int(np.argmax(shape)) > 0


class TestSpatialProfiles:
    def test_a_point_lands_on_exactly_one_cell(self):
        geom = strip(9)
        _x, _y, xn, yn = cell_coordinates(geom)
        shape = spatial_profile("point", xn, yn, x_0=0.0, y_0=0.5)
        assert np.count_nonzero(shape) == 1
        assert int(np.argmax(shape)) == 0

    def test_a_gaussian_is_centred_where_it_was_asked_for(self):
        geom = rectangle(9, 9)
        _x, _y, xn, yn = cell_coordinates(geom)
        shape = spatial_profile("gaussian", xn, yn, x_0=0.5, y_0=0.5, sigma=0.1)
        peak = int(np.argmax(shape))
        assert xn[peak] == pytest.approx(0.5, abs=0.06)
        assert yn[peak] == pytest.approx(0.5, abs=0.06)

    def test_a_point_outside_the_material_snaps_to_the_nearest_cell(self):
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        from qpsim.geometries import Geometry, extract_edge_segments
        geom = Geometry("holed", mask, extract_edge_segments(mask))
        _x, _y, xn, yn = cell_coordinates(geom)
        shape = spatial_profile("point", xn, yn, x_0=0.5, y_0=0.5)
        assert np.count_nonzero(shape) == 1

    def test_an_empty_profile_is_refused(self):
        geom = strip(5)
        _x, _y, xn, yn = cell_coordinates(geom)
        fn = compile_expression("0.0 * x", variables=("x", "y"))
        with pytest.raises(InitialConditionError, match="zero in every cell"):
            spatial_profile("expression", xn, yn, expression=fn)


class TestAssembly:
    def test_no_excess_reproduces_the_thermal_seed_exactly(self):
        """The default path must not move: every existing run starts here."""
        _m, spectral = _spectral()
        seeded = seed_occupation(spectral, 4, T_BATH)
        expected = np.repeat(
            fermi_dirac_distribution(spectral.E, T_BATH)[:, None], 4, axis=1,
        )
        np.testing.assert_array_equal(seeded.f, expected)
        assert seeded.notes == ()

    def test_an_excess_sits_on_top_of_thermal(self):
        _m, spectral = _spectral()
        geom = strip(5)
        _x, _y, xn, yn = cell_coordinates(geom)
        excess = separable_excess(
            energy_profile("flat", spectral),
            spatial_profile("point", xn, yn, x_0=0.0),
            amplitude=1e-3,
        )
        seeded = seed_occupation(spectral, 5, T_BATH, excess=excess)
        thermal = fermi_dirac_distribution(spectral.E, T_BATH)
        supported = np.asarray(spectral.cell_density) > 0
        assert np.max(seeded.f[supported, 0] - thermal[supported]) == pytest.approx(1e-3)
        np.testing.assert_allclose(seeded.f[:, 4], thermal)

    def test_clipping_is_reported_not_absorbed(self):
        """A flattened initial state is not the one that was asked for."""
        _m, spectral = _spectral()
        excess = separable_excess(
            energy_profile("flat", spectral), np.ones(3), amplitude=5.0,
        )
        seeded = seed_occupation(spectral, 3, T_BATH, excess=excess)
        assert seeded.clipped
        assert np.max(seeded.f) <= 1.0
        assert "clipped to [0, 1]" in seeded.notes[0]
        assert "5" in seeded.notes[0]

    def test_excess_and_absolute_together_are_refused(self):
        _m, spectral = _spectral()
        with pytest.raises(InitialConditionError, match="not both"):
            seed_occupation(
                spectral, 2, T_BATH,
                excess=np.zeros((spectral.E.size, 2)),
                absolute=np.zeros((spectral.E.size, 2)),
            )

    def test_a_wrong_shape_is_named(self):
        _m, spectral = _spectral()
        with pytest.raises(InitialConditionError, match="expected"):
            seed_occupation(spectral, 4, T_BATH, excess=np.zeros((3, 3)))

    def test_a_negative_amplitude_is_refused_with_a_reason(self):
        _m, spectral = _spectral()
        with pytest.raises(InitialConditionError, match="below the bath"):
            separable_excess(energy_profile("flat", spectral), np.ones(2), -1.0)


class TestItActuallyRelaxes:
    """The point of the whole module: something to measure a rate from."""

    def _run(self, excess, cells=7, steps=60):
        material, spectral = _spectral(ne=20)
        geom = strip(cells, mesh_size=5.0)
        seeded = seed_occupation(spectral, cells, T_BATH, excess=excess)
        state = T3SpatialState(
            f=seeded.f, geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        return spectral, state, T3SpatialBackend().run(
            state, dt=1.0, max_time=float(steps), stop_tol=0.0,
            snapshot_interval=float(steps) / 6.0,
        )

    def test_a_uniform_excess_decays_toward_thermal(self):
        _material, spectral = _spectral(ne=20)
        excess = separable_excess(
            energy_profile("flat", spectral), np.ones(7), amplitude=2e-3,
        )
        _s, _state, result = self._run(excess)
        totals = np.array([float(np.sum(s.f)) for s in result.snapshots])
        assert totals[-1] < totals[0], "the prepared excess never decayed"
        assert np.all(np.diff(totals) <= 0.0), "the decay was not monotone"

    def test_a_localised_hot_spot_spreads(self):
        _material, spectral = _spectral(ne=20)
        geom = strip(7, mesh_size=5.0)
        _x, _y, xn, yn = cell_coordinates(geom)
        excess = separable_excess(
            energy_profile("flat", spectral),
            spatial_profile("point", xn, yn, x_0=0.5, y_0=0.5),
            amplitude=2e-3,
        )
        _s, _state, result = self._run(excess)
        start = np.sum(result.snapshots[0].f, axis=0)
        end = np.sum(result.snapshots[-1].f, axis=0)
        centre = int(np.argmax(start))
        assert end[centre] < start[centre], "the spot did not drain"
        assert end[0] > start[0], "nothing reached the far end"
