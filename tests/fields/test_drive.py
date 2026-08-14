"""Drives that vary in energy, space and time, and the run loop that samples them.

The experiment this exists for: pulse the sample, switch the drive off, watch
the decay. It was inexpressible while a drive was a fixed array held for the
whole run, and no number of extra presets would have made it expressible.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.fields.drive import (
    DriveEvaluationError,
    ExpressionDrive,
    SeparableDrive,
    StaticDrive,
    SumDrive,
    cell_coordinates,
)
from qpsim.fields.safe_eval import compile_expression
from qpsim.geometries import rectangle, strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.gap_suppression import fermi_dirac_distribution
from qpsim.physics.spectral import SpectralContext

T_BATH = 0.2


def _state(cells: int = 6, ne: int = 20, min_factor: float = 1.0):
    material = load_material("Al")
    energies, _ = build_energy_grid(
        gap=material.Delta_0, energy_min_factor=min_factor,
        energy_max_factor=4.0, num_energy_bins=ne,
    )
    spectral = SpectralContext(
        E_bins=energies,
        dE_bins=integration_widths_from_centers(energies),
        gap=material.Delta_0,
        diffusion_coefficient=6.0,
    )
    geom = strip(cells, mesh_size=5.0)
    f0 = np.repeat(
        fermi_dirac_distribution(spectral.E, T_BATH)[:, None], cells, axis=1,
    )
    return T3SpatialState(
        f=f0, geometry=geom, spectral=spectral,
        material=material, T_bath=T_BATH,
    )


def _pattern(state, *, cell: int | None = None, amplitude: float = 2e-4):
    """A source in the above-gap bins, everywhere or in one cell."""
    _ne, ncells = state.f.shape
    energy = (state.spectral.cell_density > 0).astype(float)
    space = np.ones(ncells) if cell is None else np.eye(ncells)[cell]
    return amplitude * np.outer(energy, space)


class TestTheCanonicalExperiment:
    def test_a_pulse_drives_the_population_up_then_lets_it_decay(self):
        state = _state()
        pattern = _pattern(state)

        def gate(t: float) -> float:
            return 1.0 if 10.0 <= t < 30.0 else 0.0

        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=80.0,
            drive=SeparableDrive(pattern, gate),
            snapshot_interval=5.0,
        )
        times = np.array([s.t for s in result.snapshots])
        total = np.array([float(np.sum(s.f)) for s in result.snapshots])

        before = total[times <= 10.0]
        during = total[(times > 10.0) & (times <= 30.0)]
        after = total[times > 30.0]

        assert np.allclose(before, before[0]), "the drive acted before it was on"
        assert during[-1] > during[0], "the pulse did not raise the population"
        assert after[-1] < during[-1], "nothing decayed once the drive stopped"

    def test_a_time_dependent_drive_does_not_stop_early_on_the_residual(self):
        """Between two pulses the residual passes through zero.

        Early-stopping on it would end the run in the gap and call it a
        converged steady state, which is not a state this system has.
        """
        state = _state()
        pattern = _pattern(state)
        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=40.0, stop_tol=1e30,
            drive=SeparableDrive(pattern, lambda t: 1.0 if t < 5.0 else 0.0),
        )
        assert not result.converged
        assert result.elapsed == pytest.approx(40.0)

    def test_a_static_drive_still_converges(self):
        state = _state()
        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=40.0, stop_tol=1e30,
            drive=StaticDrive(_pattern(state)),
        )
        assert result.converged


class TestSpatialAndSpectralShape:
    def test_a_drive_confined_to_one_cell_starts_there(self):
        state = _state(cells=7)
        result = T3SpatialBackend().run(
            state, dt=0.5, max_time=2.0, stop_tol=0.0,
            drive=StaticDrive(_pattern(state, cell=0)),
        )
        excess = np.sum(result.state.f - state.f, axis=0)
        assert excess[0] == max(excess)
        assert excess[0] > excess[-1]

    def test_a_drive_below_the_gap_cannot_create_anything(self):
        """The measurement must be able to fail: this is the null case."""
        state = _state(min_factor=0.8)
        _ne, ncells = state.f.shape
        below = (state.spectral.cell_density == 0).astype(float)
        assert below.sum() > 0, "this grid has no sub-gap bins -- the test is inert"
        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=10.0, stop_tol=0.0,
            drive=StaticDrive(2e-4 * np.outer(below, np.ones(ncells))),
        )
        assert np.sum(result.state.f) == pytest.approx(np.sum(state.f), rel=1e-9)


class TestGenerality:
    def test_an_expression_drive_can_move_a_spot_across_the_device(self):
        """Non-separable in (x, t): a spot whose position depends on time."""
        state = _state(cells=9)
        x_um, y_um, x_norm, y_norm = cell_coordinates(state.geometry)
        fn = compile_expression(
            "np.where(E > params.get('gap', 180.0), "
            "  2e-3 * np.exp(-((x - t / 40.0) / 0.12) ** 2), 0.0)",
            variables=("E", "x", "y", "x_um", "y_um", "t"),
        )
        drive = ExpressionDrive(
            fn=fn, energies=state.spectral.E,
            x_um=x_um, y_um=y_um, x_norm=x_norm, y_norm=y_norm,
            params={"gap": float(state.spectral.gap)},
        )
        early, _ = drive.sample(0.0)
        late, _ = drive.sample(40.0)
        assert int(np.argmax(early.sum(axis=0))) < int(np.argmax(late.sum(axis=0)))

        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=40.0, drive=drive, snapshot_interval=10.0,
        )
        assert len(result.snapshots) >= 4
        assert np.sum(result.state.f) > np.sum(state.f)

    def test_a_non_finite_expression_is_reported_with_its_time(self):
        state = _state(cells=3)
        x_um, y_um, x_norm, y_norm = cell_coordinates(state.geometry)
        drive = ExpressionDrive(
            fn=compile_expression("1.0 / (t - 5.0)", variables=("E", "x", "y", "t")),
            energies=state.spectral.E,
            x_um=x_um, y_um=y_um, x_norm=x_norm, y_norm=y_norm,
        )
        with pytest.raises(DriveEvaluationError, match="failed at t=5"):
            drive.sample(5.0)

    def test_drives_compose(self):
        state = _state()
        steady = StaticDrive(_pattern(state, amplitude=1e-4))
        pulse = SeparableDrive(
            _pattern(state, amplitude=1e-4), lambda t: 1.0 if t < 5.0 else 0.0,
        )
        both = SumDrive((steady, pulse))
        on, _ = both.sample(0.0)
        off, _ = both.sample(10.0)
        np.testing.assert_allclose(on, 2.0 * off)
        assert not both.is_static

    def test_a_loss_channel_drains_in_proportion_to_what_is_there(self):
        state = _state()
        ne, ncells = state.f.shape
        result = T3SpatialBackend().run(
            state, dt=1.0, max_time=20.0, stop_tol=0.0,
            drive=StaticDrive(loss=1e-3 * np.ones((ne, ncells))),
        )
        assert np.sum(result.state.f) < np.sum(state.f)


class TestWiring:
    def test_passing_both_a_drive_and_arrays_is_refused(self):
        state = _state()
        with pytest.raises(ValueError, match="not both"):
            T3SpatialBackend().run(
                state, dt=1.0, max_time=1.0,
                external_gain=_pattern(state), drive=StaticDrive(),
            )

    def test_arrays_and_the_equivalent_static_drive_agree_exactly(self):
        """The old call shape must remain exactly what it was."""
        state = _state()
        pattern = _pattern(state)
        by_array = T3SpatialBackend().run(
            state, dt=1.0, max_time=10.0, stop_tol=0.0, external_gain=pattern,
        )
        by_drive = T3SpatialBackend().run(
            state, dt=1.0, max_time=10.0, stop_tol=0.0,
            drive=StaticDrive(pattern),
        )
        np.testing.assert_array_equal(by_array.state.f, by_drive.state.f)


class TestCoordinates:
    def test_a_single_cell_sits_at_the_centre_not_at_an_end(self):
        """The archived code used two conventions and presets landed off by half."""
        x_um, y_um, x_norm, y_norm = cell_coordinates(rectangle(1, 1, mesh_size=2.0))
        assert x_norm[0] == 0.5 and y_norm[0] == 0.5
        assert x_um[0] == 1.0 and y_um[0] == 1.0

    def test_coordinates_follow_mask_order(self):
        geom = rectangle(2, 3, mesh_size=1.0)
        x_um, y_um, _, _ = cell_coordinates(geom)
        assert x_um.size == geom.cell_count
        # Row-major over mask.nonzero(): the first row's three cells, then the
        # second row's -- the same order the solver stores columns in.
        np.testing.assert_allclose(x_um, [0.5, 1.5, 2.5, 0.5, 1.5, 2.5])
        np.testing.assert_allclose(y_um, [0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
