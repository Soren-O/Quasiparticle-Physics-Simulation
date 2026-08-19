"""Local collisions on the 1-D strip, solved by the unified spatial backend.

Translated from ``TestT3Spatial1DCollisions`` in
``tests/backends/test_t3_spatial_1d.py``, which tested the retired
``T3Spatial1DBackend``. Every state here is a ``T3SpatialState`` on a
``strip(N, mesh_size=dx)`` geometry -- a ``(1, N)`` mask -- so the unified
backend solves the same strip as a degenerate case of the 2-D core.

The retired backend owned its collision half directly (``_collision_cache``,
``_local_spectral_context``, ``_collision_rate``). The unified backend
delegates to :class:`qpsim.collisions.spatial.SpatialCollisions`, reachable as
``backend._collisions_for(state)``, which carries the same exact-gap grouping,
the same two-entry LRU over dense kernels and the same streamed path past two
distinct gaps. The properties below are asserted against that layer.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.collisions.phonon import (
    apply_phonon_collision,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
    NX: int = 11,
) -> T3SpatialState:
    """The 1-D fixture, on a one-cell-wide geometry.

    The retired fixture used ``x = np.linspace(0.0, 100.0, NX)``, so the mesh
    spacing is ``100/(NX-1)``, not ``100/NX``; it is taken from the coordinate
    array itself to keep that exact. A single cell has no spacing to infer, so
    it keeps the retired state's unit-width convention.
    """
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=5.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    x = np.linspace(0.0, 100.0, NX)
    dx = float(x[1] - x[0]) if NX > 1 else 1.0
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=strip(NX, mesh_size=dx),
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


def _collision_rate(
    backend: T3SpatialBackend,
    state: T3SpatialState,
    *,
    external_gain: np.ndarray | None = None,
    external_loss: np.ndarray | None = None,
) -> np.ndarray:
    """The collision half of ``T3SpatialBackend.rates``, on its own.

    The retired backend exposed this as ``_collision_rate``. The unified
    backend has no such entry point -- ``rates`` returns transport and
    collisions summed -- so the collision residual is harvested here through
    exactly the production objects ``rates`` itself walks (``_collisions_for``,
    ``_groups``, ``local_operator``, ``combined_group_rates``). Nothing is
    reimplemented; the grouping and streaming under test are all inside those.
    """
    collisions = backend._collisions_for(state)
    total = np.zeros_like(state.f)
    for gap, columns in collisions._groups():
        operator = collisions.local_operator(gap)
        gain, loss = collisions.combined_group_rates(
            state.f[:, columns], columns, operator, gap,
            None if external_gain is None else external_gain[:, columns],
            None if external_loss is None else external_loss[:, columns],
        )
        total[:, columns] = gain - loss * state.f[:, columns]
    return total


class TestStripCollisions:
    """The 1-D reduction of the unified backend's collision half.

    Translated from ``TestT3Spatial1DCollisions``
    (``tests/backends/test_t3_spatial_1d.py``), which tested the retired
    ``T3Spatial1DBackend`` / ``T3Spatial1DState`` / ``T3SpatialFlux1D``.
    """

    def test_thermal_equilibrium_stays_stationary_without_flux(self) -> None:
        state = _build_state(T_bath=0.1)
        out = T3SpatialBackend().apply_collisions(state, dt=1.0)
        np.testing.assert_allclose(out.f, state.f, atol=1e-9)

    def test_one_end_flux_changes_source_cell_first(self) -> None:
        state = _build_state(T_bath=0.0)
        gain = np.zeros_like(state.f)
        target = int(np.argmin(np.abs(state.spectral.E - 2.0 * state.gap)))
        gain[target, 0] = 1e-4

        out = T3SpatialBackend().apply_collisions(
            state,
            dt=1.0,
            external_gain=gain,
            external_loss=None,
        )

        assert out.f[target, 0] > out.f[target, -1]
        assert out.f[target, 0] > state.f[target, 0]

    def test_gap_profile_uses_local_collision_contexts(self) -> None:
        material = load_material("Al")
        gap_left, gap_right = material.Delta_0, 200.0
        E, _ = build_energy_grid(gap_left, 1.0, 4.0, 96)
        dE = integration_widths_from_centers(E)
        spectral_left = SpectralContext(E, dE, gap_left)
        spectral_right = SpectralContext(E, dE, gap_right)
        f_column = 0.1 * np.exp(-((E - 2.0 * gap_left) / 20.0) ** 2)
        f = np.repeat(f_column[:, None], 2, axis=1)
        f[gap_right >= E, 1] = 0.0
        state = T3SpatialState(
            f=f,
            geometry=strip(2, mesh_size=1.0),
            spectral=spectral_left,
            material=material,
            T_bath=0.1,
            gap_per_cell=np.array([gap_left, gap_right]),
        )

        dt = 0.1
        out = T3SpatialBackend().apply_collisions(state, dt)

        expected = []
        for column, ctx in enumerate((spectral_left, spectral_right)):
            K_s0 = build_scattering_kernel_base(
                ctx, tau_0=material.tau_0, T_c=material.T_c,
            )
            K_r0 = build_recombination_kernel_base(
                ctx, tau_0=material.tau_0, T_c=material.T_c,
            )
            expected.append(
                apply_phonon_collision(
                    state.f[:, column], ctx, K_s0, K_r0, state.T_bath, dt,
                )
            )

        np.testing.assert_allclose(out.f[:, 0], expected[0], rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(out.f[:, 1], expected[1], rtol=1e-13, atol=1e-15)
        assert np.max(np.abs(out.f[:, 0] - out.f[:, 1])) > 1e-8
        np.testing.assert_array_equal(out.f[~spectral_right.active_mask, 1], 0.0)
        cut = (gap_right >= E) & spectral_right.active_mask
        assert np.any(cut)
        assert np.any(out.f[cut, 1] > 0.0)

    def test_repeated_two_gap_call_hits_cache_before_context_build(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=14, NX=2)
        state.gap_per_cell = np.array([state.gap, 1.2 * state.gap])
        backend = T3SpatialBackend()

        first = backend.apply_collisions(state, 0.05)
        collisions = backend._collisions_for(state)
        assert len(collisions._cache) == 2

        def unexpected_context(*_args: object, **_kwargs: object) -> SpectralContext:
            raise AssertionError("cache hit eagerly rebuilt a SpectralContext")

        monkeypatch.setattr(collisions, "_local_spectral", unexpected_context)
        second = backend.apply_collisions(state, 0.05)

        np.testing.assert_allclose(second.f, first.f, rtol=0.0, atol=0.0)
        assert len(collisions._cache) == 2

    def test_collision_cache_is_lru_bounded_across_many_gaps(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The retired backend held ONE cross-state LRU; the unified backend's
        # LRU lives on the SpatialCollisions layer, so the sequence of local
        # gaps is driven through that layer rather than through five separate
        # single-cell states (each of which would build its own layer and make
        # the eviction order untestable). The property is unchanged: at most
        # two dense operator sets resident, evicted in TRUE LRU order, and
        # evicted BEFORE the replacement context/matrices are built.
        material = load_material("Al")
        E = np.linspace(150.0, 400.0, 10)
        dE = integration_widths_from_centers(E)
        state = T3SpatialState(
            f=np.zeros((E.size, 1)),
            geometry=strip(1),
            spectral=SpectralContext(E, dE, 120.0),
            material=material,
            T_bath=0.0,
        )
        collisions = T3SpatialBackend()._collisions_for(state)

        for gap in (120.0, 125.0, 130.0, 135.0, 140.0):
            collisions.local_operator(gap)
            assert len(collisions._cache) <= 2

        assert len(collisions._cache) == 2
        assert [key[0] for key in collisions._cache] == [135.0, 140.0]

        # Refresh 135, then miss on 145. True LRU order evicts 140, and the
        # eviction happens before the local context/new matrices are built.
        collisions.local_operator(135.0)
        cache_sizes_at_build: list[int] = []
        original = collisions._local_spectral

        def capture_prebuild_size(local_gap: float) -> SpectralContext:
            cache_sizes_at_build.append(len(collisions._cache))
            return original(local_gap)

        monkeypatch.setattr(collisions, "_local_spectral", capture_prebuild_size)
        collisions.local_operator(145.0)

        assert cache_sizes_at_build == [1]
        assert [key[0] for key in collisions._cache] == [135.0, 145.0]

    def test_collision_cache_invalidates_physics_inputs(self) -> None:
        # The retired backend counted misses on its own operator cache. The
        # unified backend's kernel cache is the SpatialCollisions layer itself,
        # which _collisions_for rebuilds on a value signature, so "a miss" is
        # "a new layer object". Each of these inputs feeds the dense kernels
        # and must invalidate them. The fourth -- the spectral cell widths --
        # is split out below because it does NOT.
        state = _build_state(T_bath=0.0, NE=10, NX=1)
        backend = T3SpatialBackend()

        backend.apply_collisions(state, 0.01)
        first = backend._collisions
        backend.apply_collisions(state, 0.01)
        assert backend._collisions is first, "an unchanged state rebuilt kernels"

        layers = [first]

        def rebuilt(what: str) -> None:
            backend.apply_collisions(state, 0.01)
            assert backend._collisions is not layers[-1], (
                f"changing {what} reused the cached collision kernels"
            )
            layers.append(backend._collisions)

        state.T_bath = 0.2
        rebuilt("T_bath")
        state.material.tau_0 *= 1.1
        rebuilt("tau_0")
        state.material.T_c *= 1.01
        rebuilt("T_c")

        assert len(layers) == 4

    def test_collision_cache_invalidates_changed_cell_widths(self) -> None:
        state = _build_state(T_bath=0.0, NE=10, NX=1)
        backend = T3SpatialBackend()
        backend.apply_collisions(state, 0.01)
        before = backend._collisions

        state.spectral = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=1.01 * state.spectral.dE,
            gap=state.gap,
            diffusion_coefficient=state.spectral.diffusion_coefficient,
        )
        backend.apply_collisions(state, 0.01)

        assert backend._collisions is not before, (
            "changing dE reused the cached collision kernels"
        )

    def test_many_distinct_local_gaps_stream_exactly_with_bounded_cache(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.18, NE=14, NX=3)
        gaps = state.gap * np.array([1.0, 1.05, 1.1])
        state.gap_per_cell = gaps
        # Give each column a distinct, represented non-equilibrium profile.
        for column, gap in enumerate(gaps):
            local = SpectralContext(state.spectral.E, state.spectral.dE, gap)
            state.f[:, column] = np.where(
                local.active_mask,
                (column + 1.0) * 1e-3 * np.exp(-state.spectral.E / (4.0 * gap)),
                0.0,
            )

        backend = T3SpatialBackend()
        collisions = backend._collisions_for(state)

        def unexpected_batched_path(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("three-gap profile entered batched assembly")

        monkeypatch.setattr(collisions, "_apply_batched", unexpected_batched_path)
        dt = 0.03
        out = backend.apply_collisions(state, dt)

        expected = []
        for column, gap in enumerate(gaps):
            local = SpectralContext(state.spectral.E, state.spectral.dE, gap)
            K_s0 = build_scattering_kernel_base(
                local,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            K_r0 = build_recombination_kernel_base(
                local,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            expected.append(
                apply_phonon_collision(
                    state.f[:, column],
                    local,
                    K_s0,
                    K_r0,
                    state.T_bath,
                    dt,
                )
            )

        np.testing.assert_allclose(
            out.f,
            np.column_stack(expected),
            rtol=2e-13,
            atol=1e-15,
        )
        assert len(collisions._cache) == 2
        assert [key[0] for key in collisions._cache] == pytest.approx(gaps[-2:])

        # A repeat must reproduce the step bit for bit and must not grow the
        # cache. How many kernel builds it costs is the separate question
        # pinned by test_repeated_streamed_evaluation_reuses_resident_lru_entries
        # below.
        repeated = backend.apply_collisions(state, dt)
        np.testing.assert_array_equal(repeated.f, out.f)
        assert len(collisions._cache) == 2

    def test_streamed_collision_rate_matches_independent_gap_columns(self) -> None:
        state = _build_state(T_bath=0.12, NE=12, NX=4)
        gaps = state.gap * np.array([1.0, 1.03, 1.07, 1.11])
        state.gap_per_cell = gaps
        backend = T3SpatialBackend()

        rate = _collision_rate(backend, state)
        expected = np.empty_like(rate)
        for column, gap in enumerate(gaps):
            one = _build_state(T_bath=state.T_bath, NE=12, NX=1)
            one.gap_per_cell = np.array([gap])
            one.f[:, 0] = state.f[:, column]
            expected[:, column] = _collision_rate(T3SpatialBackend(), one)[:, 0]

        np.testing.assert_allclose(rate, expected, rtol=2e-13, atol=1e-15)
        assert len(backend._collisions_for(state)._cache) == 2

    def test_repeated_streamed_collision_rate_is_reproducible_and_bounded(
        self,
    ) -> None:
        state = _build_state(T_bath=0.12, NE=12, NX=4)
        gaps = state.gap * np.array([1.0, 1.03, 1.07, 1.11])
        state.gap_per_cell = gaps
        backend = T3SpatialBackend()
        collisions = backend._collisions_for(state)

        first = _collision_rate(backend, state)
        assert [key[0] for key in collisions._cache] == pytest.approx(gaps[-2:])

        second = _collision_rate(backend, state)
        np.testing.assert_array_equal(second, first)
        assert len(collisions._cache) == 2

    @pytest.mark.xfail(
        reason=(
            "OPEN REGRESSION (performance, not correctness). The retired "
            "backend's _streaming_group_order visited the gaps already "
            "resident in its two-entry LRU first, so a repeated streamed "
            "evaluation over four gaps rebuilt only the two missing kernels. "
            "SpatialCollisions._groups() always traverses in sorted gap order, "
            "so with a working set larger than the cache the first miss evicts "
            "a gap needed later in the same traversal and every gap misses on "
            "every call. The rebuilt kernels are identical -- the answer is "
            "unaffected -- but a smooth gap profile now rebuilds every dense "
            "(NE, NE) kernel on every step. Un-xfail when the traversal is "
            "ordered against the resident set."
        ),
    )
    def test_repeated_streamed_evaluation_reuses_resident_lru_entries(
        self,
    ) -> None:
        state = _build_state(T_bath=0.12, NE=12, NX=4)
        gaps = state.gap * np.array([1.0, 1.03, 1.07, 1.11])
        state.gap_per_cell = gaps
        backend = T3SpatialBackend()
        collisions = backend._collisions_for(state)

        _collision_rate(backend, state)
        assert [key[0] for key in collisions._cache] == pytest.approx(gaps[-2:])

        context_builds: list[float] = []
        original_context = collisions._local_spectral

        def record_context_build(local_gap: float) -> SpectralContext:
            context_builds.append(float(local_gap))
            return original_context(local_gap)

        collisions._local_spectral = record_context_build  # type: ignore[method-assign]
        try:
            _collision_rate(backend, state)
        finally:
            del collisions._local_spectral

        # The two resident operators are consumed before either can be
        # evicted. Only the two missing gaps should be rebuilt; sorted-order
        # traversal instead causes four misses on every diagnostic call.
        assert context_builds == pytest.approx(gaps[:2])
        assert len(collisions._cache) == 2

    def test_streamed_profile_rejects_gain_on_late_zero_capacity_row(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=3)
        gaps = state.gap * np.array([1.0, 1.2, 1.5])
        state.gap_per_cell = gaps
        high = SpectralContext(state.spectral.E, state.spectral.dE, gaps[-1])
        target = int(np.flatnonzero(~high.active_mask)[0])
        gain = np.zeros_like(state.f)
        gain[target, -1] = 1e-4

        # The unified message names the offending ENERGY BINS and the local
        # gap rather than the (energy, cell) pair the retired backend printed;
        # the cell is identified by its gap group instead.
        with pytest.raises(
            ValueError,
            match=rf"zero-spectral-capacity states \(energy bins {target},",
        ):
            T3SpatialBackend().apply_collisions(
                state, 0.01, external_gain=gain, external_loss=None,
            )

    def test_gain_on_local_zero_capacity_row_is_rejected(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=2)
        high_gap = 1.5 * state.gap
        state.gap_per_cell = np.array([state.gap, high_gap])
        high_ctx = SpectralContext(
            state.spectral.E, state.spectral.dE, high_gap,
        )
        target = int(np.flatnonzero(~high_ctx.active_mask)[0])
        gain = np.zeros_like(state.f)
        gain[target, 1] = 1e-4

        with pytest.raises(ValueError, match="zero-spectral-capacity"):
            T3SpatialBackend().apply_collisions(
                state, 0.01, external_gain=gain, external_loss=None,
            )

        state.f[target, 1] = 0.7
        loss_rate = np.zeros_like(state.f)
        loss_rate[target, 1] = 3.0
        updated = T3SpatialBackend().apply_collisions(
            state, 0.01, external_gain=None, external_loss=loss_rate,
        )
        assert updated.f[target, 1] == state.f[target, 1]

    def test_gain_on_gap_cut_cell_is_evolved(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=2)
        high_gap = 1.2 * state.gap
        state.gap_per_cell = np.array([state.gap, high_gap])
        high_ctx = SpectralContext(
            state.spectral.E, state.spectral.dE, high_gap,
        )
        cut = (high_gap >= state.spectral.E) & high_ctx.active_mask
        target = int(np.flatnonzero(cut)[0])
        assert high_ctx.rho[target] == 0.0
        assert high_ctx.cell_weights[target] > 0.0

        state.f[target, 1] = 0.0
        gain = np.zeros_like(state.f)
        gain[target, 1] = 1e-4
        updated = T3SpatialBackend().apply_collisions(
            state, 0.01, external_gain=gain, external_loss=None,
        )

        assert updated.f[target, 1] > 0.0
