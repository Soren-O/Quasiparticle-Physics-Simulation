"""Tests for qpsim.backends.spatial, the dimension-agnostic backend.

The 1-D reduction is asserted directly in ``tests/backends/strip/``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.spatial import SpatialBackend, SpatialState
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






class TestDimensionality:
    def test_zero_dimensional_has_no_transport_but_still_collides(self):
        material, spectral = _setup(ne=16)
        f0 = _occupations(spectral, material, 1)
        state = SpatialState(
            f=f0.copy(), geometry=rectangle(1, 1), spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        backend = SpatialBackend()
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
        state = SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        got = SpatialBackend().apply_transport(state, DT).f
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
        state = SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        got = SpatialBackend().step(state, DT).f
        assert got.shape == f0.shape
        assert np.all(got >= 0.0) and np.all(got <= 1.0)


class TestTermSwitches:
    def test_collision_channels_can_be_switched_off(self):
        material, spectral = _setup(ne=16)
        geom = strip(5)
        f0 = _occupations(spectral, material, geom.cell_count)
        state = SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )
        both = SpatialBackend().step(state, DT).f
        neither = SpatialBackend(
            enable_scattering=False, enable_recombination=False,
        ).step(state, DT).f
        assert not np.array_equal(both, neither)


class TestBoundaryConditions:
    def test_a_dirichlet_edge_injects(self):
        """A Dirichlet edge holds an occupation and injects into the strip."""
        material, spectral = _setup(ne=16)
        geom = strip(6)
        conditions = geom.conditions()
        left = next(e for e in geom.edges if e.faces[0].direction == "left")
        conditions[left.edge_id] = BoundaryCondition("dirichlet", 0.4)
        f0 = np.zeros((spectral.E.size, geom.cell_count))
        state = SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH, conditions=conditions,
        )
        got = SpatialBackend().apply_transport(state, DT).f
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
        state = SpatialState(
            f=f0, geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        residual = float(np.max(np.abs(SpatialBackend().rates(state))))
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
        state = SpatialState(
            f=f0, geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        assert float(np.max(np.abs(SpatialBackend().rates(state)))) > 0.0

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
        backend = SpatialBackend()
        state = SpatialState(
            f=f0.copy(), geometry=geom, spectral=spectral, material=material,
            T_bath=T_BATH,
        )
        before = float(np.max(np.abs(backend.rates(state))))
        result = backend.run(state, dt=1.0, max_time=200.0, stop_tol=1e-14)
        assert result.n_steps > 0
        assert result.last_max_rate < before
        grid = result.state.f[peak].reshape(8, 8)
        assert grid[3, 3] < f0[peak, 27]          # the spike drained
        assert grid[3, 2] > f0[peak, 26]          # a neighbour filled
        assert np.isclose(grid[3, 2], grid[2, 3])  # symmetric in both axes

    def test_a_non_positive_step_is_refused(self):
        material, spectral = _setup(ne=16)
        geom = strip(4)
        state = SpatialState(
            f=_occupations(spectral, material, geom.cell_count),
            geometry=geom, spectral=spectral, material=material, T_bath=T_BATH,
        )
        with pytest.raises(ValueError, match="dt must be positive"):
            SpatialBackend().run(state, dt=0.0, max_time=1.0)


class TestFeatureParityWithTheStrip:
    """Gap regions and injection, the two things spatial_1d had and 2D lacked."""

    def _driven_step(self, conductance, max_time=300.0):
        from qpsim.webui import execute
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
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
        from qpsim.webui.schemas import KineticsSetup
        results = []
        for enabled in (False, True):
            setup = KineticsSetup()
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
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.geometry.rows = 4
        setup.geometry.cols = 8
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_right = 240.0
        state = build_state_2d(setup)
        assert sorted(np.unique(state.gaps())) == [180.0, 240.0]


class TestPhononPopulationSurvivesAGapChange:
    """The evolved phonon population is state, not a cached kernel.

    ``SpatialCollisions`` is cache-keyed on the gap profile *and* owns
    ``n_ph``. Rebuilding it for a new gap must not re-seed the phonons at the
    bath: a self-consistent solve -- which moves the gap on every step --
    would silently discard phonon trapping, the entire reason the dynamic
    sector exists. A gap change of one part in 10^6 is enough to expose it.
    """

    @staticmethod
    def _backend_and_state(rows: int = 4, cols: int = 4):
        from qpsim.webui.builders import build_state_2d
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 48
        setup.geometry.rows, setup.geometry.cols = rows, cols
        setup.phonons.mode = "dynamic_escape"
        state = build_state_2d(setup)
        backend = SpatialBackend(phonon_escape_time=setup.phonons.tau_l_ns)
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


class TestThePhononAdvanceIsSymmetric:
    """The composed step must be second order WITH the phonon sector live.

    The transport wrap was already symmetric -- T(dt/2), collisions, T(dt/2) --
    but the phonon advance inside it ran once, after the collision sub-step, at
    the new ``f``. That is a Lie composition, and a step is only as accurate as
    its weakest factor, so the outer symmetry bought nothing the moment the
    phonons were solved rather than held at the bath.

    Both sub-flows are already exact for their own frozen-coefficient problems
    (``advance_phonons`` integrates an affine ODE in closed form, ETD2 advances
    ``f``), so the entire error lives in the ORDER they are composed in, and
    the fix is a reordering rather than an added stage.
    """

    @staticmethod
    def _backend_and_state(bins: int = 24):
        from qpsim.webui.builders import build_state_2d
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = bins
        setup.geometry.rows, setup.geometry.cols = 2, 3
        setup.phonons.mode = "dynamic_escape"
        backend = SpatialBackend(phonon_escape_time=setup.phonons.tau_l_ns)
        return backend, build_state_2d(setup)

    def test_one_step_advances_the_phonons_in_two_halves(self):
        """Pins the composition directly, so a regression names itself.

        Structural rather than numerical: it fails on the reordering itself,
        not on a tolerance, and it cannot be satisfied by an advance that runs
        once for the whole step no matter how small the resulting error is.
        """
        from qpsim.collisions.spatial import SpatialCollisions
        backend, state = self._backend_and_state()
        assert backend._collisions_for(state).n_ph is not None, (
            "phonons frozen -- the test is inert"
        )

        seen: list[float] = []
        original = SpatialCollisions.advance_phonons

        def spy(self, f, dt):
            seen.append(float(dt))
            return original(self, f, dt)

        SpatialCollisions.advance_phonons = spy
        try:
            backend.step(state, 0.1)
        finally:
            SpatialCollisions.advance_phonons = original

        assert seen == [0.05, 0.05], (
            f"expected two half advances, got {seen} -- an unsymmetric phonon "
            "composition drops the whole step to first order"
        )

    def test_halving_the_step_quarters_the_error(self):
        """Second order, measured against the same integrator at dt/64.

        The reference is this backend at a far smaller step, so what is
        measured is the SPLITTING error alone -- not the accuracy of either
        sub-flow, both of which are exact for their frozen sub-problem.
        """
        T = 0.2

        def evolve(dt: float):
            backend, state = self._backend_and_state()
            collisions = backend._collisions_for(state)
            # Seed away from the bath: at the phonon flow's own fixed point
            # every composition of it agrees, and the test would pass on a
            # sector that never moved.
            collisions.n_ph = collisions.n_ph + 2.0e-3
            seed = collisions.n_ph.copy()
            n = int(round(T / dt))
            s = state
            for _ in range(n):
                s = backend.step(s, dt)
            final = backend._collisions_for(s).n_ph.copy()
            assert np.max(np.abs(final - seed)) > 1e-9, (
                "the phonon population never moved -- the measurement is inert"
            )
            return s.f.copy(), final

        ref_f, ref_n = evolve(T / 64.0)
        errors = []
        for steps in (4, 8):
            f, n_ph = evolve(T / steps)
            errors.append((
                float(np.max(np.abs(f - ref_f))),
                float(np.max(np.abs(n_ph - ref_n))),
            ))
        (coarse_f, coarse_n), (fine_f, fine_n) = errors

        # Second order gives 4; the Lie arrangement this replaced measures 2.1
        # on the same problem, so the two are separated by a wide margin and
        # the bound does not need to be tight.
        assert coarse_f / fine_f > 3.4, (
            f"f converged at ratio {coarse_f / fine_f:.2f}, not ~4"
        )
        assert coarse_n / fine_n > 3.4, (
            f"n_ph converged at ratio {coarse_n / fine_n:.2f}, not ~4"
        )


class TestRecordedFrames:
    """A spatial run has to be observable on the way, not only at the end.

    Without frames the single observable is the endpoint, so "has it settled"
    is indistinguishable from "is still drifting", and every dynamical
    question -- how fast a front moves, when a transient decays -- has nowhere
    to read an answer from.
    """

    @staticmethod
    def _driven_state(cells: int = 6):
        material, spectral = _setup(ne=16)
        geom = strip(cells)
        f0 = _occupations(spectral, material, geom.cell_count)
        return SpatialState(
            f=f0, geometry=geom, spectral=spectral,
            material=material, T_bath=T_BATH,
        )

    def test_no_interval_records_nothing(self):
        """The default path must be untouched: frames cost memory."""
        result = SpatialBackend().run(
            self._driven_state(), dt=1.0, max_time=10.0, stop_tol=0.0,
        )
        assert result.snapshots == []
        assert result.n_steps == 10

    def test_frames_span_the_run_including_both_ends(self):
        result = SpatialBackend().run(
            self._driven_state(), dt=0.5, max_time=10.0,
            stop_tol=0.0, snapshot_interval=2.0,
        )
        times = [s.t for s in result.snapshots]
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(result.elapsed)
        assert times == sorted(times)
        # 0, 2, 4, 6, 8, 10 -- the endpoint coincides with a boundary here, so
        # it must not be recorded twice.
        assert len(times) == len(set(times)) == 6

    def test_the_initial_residual_is_measured_not_infinite(self):
        result = SpatialBackend().run(
            self._driven_state(), dt=1.0, max_time=4.0,
            stop_tol=0.0, snapshot_interval=2.0,
        )
        assert np.isfinite(result.snapshots[0].max_rate)
        assert result.snapshots[0].max_rate > 0.0

    def test_a_frame_is_a_copy_not_a_view(self):
        """A later step must not be able to rewrite recorded history."""
        result = SpatialBackend().run(
            self._driven_state(), dt=1.0, max_time=6.0,
            stop_tol=0.0, snapshot_interval=2.0,
        )
        first, last = result.snapshots[0], result.snapshots[-1]
        assert not np.shares_memory(first.f, result.state.f)
        assert not np.array_equal(first.f, last.f), "the field never moved"

    def test_a_cadence_finer_than_the_cap_is_refused(self):
        with pytest.raises(ValueError, match=r"past the .* cap"):
            SpatialBackend().run(
                self._driven_state(), dt=1.0, max_time=1e6,
                snapshot_interval=1e-3,
            )

    def test_the_phonon_history_starts_at_the_first_frame(self):
        """t=0 is taken before any step, so the layer must be built for it.

        Reading the cached attribute gave the first frame ``None`` and every
        consumer then treated the run as having no phonon history at all.
        """
        from qpsim.webui.builders import build_state_2d
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 24
        setup.geometry.rows, setup.geometry.cols = 2, 4
        setup.phonons.mode = "dynamic_escape"
        backend = SpatialBackend(phonon_escape_time=setup.phonons.tau_l_ns)
        result = backend.run(
            build_state_2d(setup), dt=1.0, max_time=4.0,
            stop_tol=0.0, snapshot_interval=2.0,
        )
        assert all(s.n_ph is not None for s in result.snapshots)
        assert result.snapshots[0].n_ph.shape == result.snapshots[-1].n_ph.shape


class TestPerEdgeBoundaryConditions:
    """Different edges mean different things, and the schema has to say so.

    A rim-wide condition cannot express the standard quasiparticle-trapping
    experiment -- an absorbing trap on one end, reflective elsewhere -- and on
    a one-cell-wide strip it is actively wrong: "absorbing" then also absorbs
    through the long sides, which a 1-D strip does not have.
    """

    @staticmethod
    def _strip_mean(per_edge, kind="reflective"):
        from qpsim.webui import execute
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 24
        setup.geometry.rows, setup.geometry.cols = 1, 20
        setup.dt = 2.0
        setup.max_time = 200.0
        setup.stop_tol = 0.0
        setup.boundary.kind = kind
        setup.boundary.per_edge = per_edge
        payload = execute.execute_setup(setup, lambda *a, **k: None, lambda: False)
        return float(payload.summary["x_qp_mean"])

    def test_absorbing_ends_lose_far_less_than_an_absorbing_rim(self):
        """The quantitative statement of the trap, on the 1-D reduction."""
        from qpsim.webui.schemas import EdgeCondition
        absorbing = EdgeCondition(kind="absorbing")
        ends = self._strip_mean({"left": absorbing, "right": absorbing})
        whole_rim = self._strip_mean({}, kind="absorbing")
        assert whole_rim < ends / 1e4, (
            "a rim-wide absorbing condition should leak through the long "
            "sides that a 1-D strip does not have"
        )

    def test_one_absorbing_end_sits_between_the_two_limits(self):
        from qpsim.webui.schemas import EdgeCondition
        reflective = self._strip_mean({})
        one_end = self._strip_mean({"left": EdgeCondition(kind="absorbing")})
        both = self._strip_mean({
            "left": EdgeCondition(kind="absorbing"),
            "right": EdgeCondition(kind="absorbing"),
        })
        assert both < one_end < reflective

    def test_robin_interpolates_between_reflective_and_absorbing(self):
        """The continuum where every real contact actually sits.

        Measured on this mesh: reflective 7.54e-6, robin(beta=0.05) 1.98e-6,
        absorbing 6.00e-7. Note beta is NOT a dimensionless "transparency" --
        robin(beta=0.5) reproduces the absorbing case to every digit here and
        larger beta drains further still, so absorbing is one particular beta
        for this spacing rather than the beta -> infinity limit. beta=0.05 is
        chosen because it is unambiguously between the two.
        """
        from qpsim.webui.schemas import EdgeCondition
        reflective = self._strip_mean({})
        absorbing = self._strip_mean({"left": EdgeCondition(kind="absorbing")})
        robin = self._strip_mean({
            "left": EdgeCondition(kind="robin", value=0.05, aux_value=0.0),
        })
        assert absorbing < robin < reflective

    def test_robin_without_its_second_coefficient_is_refused(self):
        from pydantic import ValidationError
        from qpsim.webui.schemas import EdgeConditions
        with pytest.raises(ValidationError, match="needs aux_value"):
            EdgeConditions(kind="robin", value=0.5)

    def test_an_unknown_edge_names_what_is_available(self):
        """A condition that silently never applies is worse than an error."""
        from qpsim.webui.schemas import EdgeCondition
        with pytest.raises(ValueError, match="no edge 'north'"):
            self._strip_mean({"north": EdgeCondition(kind="absorbing")})
