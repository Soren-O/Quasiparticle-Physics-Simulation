"""Tests for the 1D spatial T3 diffusion preview backend."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial_1d import (
    T3Spatial1DBackend,
    T3Spatial1DState,
    T3SpatialFlux1D,
)
from qpsim.collisions.phonon import (
    apply_phonon_collision,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.bcs_quadrature import cell_edges_from_widths
from qpsim.physics.spectral import SpectralContext, bcs_density_of_states
from qpsim.transport.diffusion.base import DiffusionModel
from scipy.integrate import quad


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
) -> T3Spatial1DState:
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
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], NX, axis=1)
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


class TestT3Spatial1DTransport:
    def test_large_finite_diffusivity_remains_finite(self) -> None:
        """The public transport path must not overflow its face mean."""
        state = _build_state(D0=1e200, T_bath=0.0, NE=2, NX=2)
        state.x = np.array([0.0, 1.0])
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2

        out = T3Spatial1DBackend().apply_transport(state, dt=1e-200)

        assert np.all(np.isfinite(out.f))
        assert out.f[-1, 1] > 0.0
        assert float(np.sum(out.f[-1])) == pytest.approx(0.2)

    def test_reflective_transport_preserves_uniform_field(self) -> None:
        state = _build_state()
        out = T3Spatial1DBackend().apply_transport(state, dt=2.0)
        np.testing.assert_allclose(out.f, state.f, atol=1e-13)
    @pytest.mark.parametrize("field", ["gain", "loss_rate"])
    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_before_float_cast(
        self,
        field: str,
        imaginary: float,
    ) -> None:
        arrays = {
            "gain": np.zeros((2, 2)),
            "loss_rate": np.zeros((2, 2)),
        }
        arrays[field] = arrays[field].astype(complex)
        arrays[field][0, 0] = complex(0.0, imaginary)

        with pytest.raises(ValueError, match=rf"{field} must be real-valued"):
            T3SpatialFlux1D(**arrays)

    def test_reflective_transport_spreads_and_conserves_pulse(self) -> None:
        state = _build_state(T_bath=0.0)
        f = np.zeros_like(state.f)
        energy_idx = -1
        f[energy_idx, 0] = 0.2
        state.f = f

        out = T3Spatial1DBackend().apply_transport(state, dt=5.0)

        assert out.f[energy_idx, 0] < state.f[energy_idx, 0]
        assert out.f[energy_idx, 1] > 0.0
        np.testing.assert_allclose(
            np.sum(out.f[energy_idx]),
            np.sum(state.f[energy_idx]),
            atol=1e-13,
        )

    def test_transport_conserves_declared_cell_center_measure(self) -> None:
        length_um = 100.0
        state = _build_state(T_bath=0.0, NX=21)
        dx_um = length_um / state.x.size
        state.x = (np.arange(state.x.size, dtype=float) + 0.5) * dx_um
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2
        before = float(np.sum(state.cell_widths * state.f[-1]))

        backend = T3Spatial1DBackend()
        out = state
        for _ in range(20):
            out = backend.apply_transport(out, dt=0.5)
        after = float(np.sum(out.cell_widths * out.f[-1]))

        assert state.cell_widths.sum() == pytest.approx(length_um)
        assert after == pytest.approx(before, abs=1e-12)

    def test_clip_warning_uses_absolute_mass_not_cancelable_signed_net(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=2, NX=2)
        state.f[:] = 0.5
        backend = T3Spatial1DBackend()

        class _FixedSolve:
            def __init__(self, result: np.ndarray) -> None:
                self.result = result

            def solve(self, _rhs: np.ndarray) -> np.ndarray:
                return self.result

        # Opposite-sign, individually small clips cancel in the signed sum.
        # The absolute diagnostic must still fire.
        idx = np.array([0, 1])
        rho_p = np.ones(2)
        ops = [
            (np.eye(2), _FixedSolve(np.array([1.000001, 0.5])), idx, rho_p, 1),
            (np.eye(2), _FixedSolve(np.array([-0.000001, 0.5])), idx, rho_p, 1),
        ]
        monkeypatch.setattr(
            backend, "_build_transport_operators", lambda _state, _dt: ops,
        )

        with pytest.warns(UserWarning, match="absolute conserved density"):
            out = backend.apply_transport(state, dt=1.0)
        np.testing.assert_array_equal(out.f[0], np.array([1.0, 0.5]))
        np.testing.assert_array_equal(out.f[1], np.array([0.0, 0.5]))

    def test_large_clip_corruption_fails_instead_of_converging(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=1, NX=2)
        state.f[:] = 0.5
        backend = T3Spatial1DBackend()

        class _FixedSolve:
            @staticmethod
            def solve(_rhs: np.ndarray) -> np.ndarray:
                return np.array([1.1, 0.5])

        monkeypatch.setattr(
            backend,
            "_build_transport_operators",
            lambda _state, _dt: [
                (np.eye(2), _FixedSolve(), np.array([0, 1]), np.ones(2), 1),
            ],
        )

        with pytest.raises(RuntimeError, match="absolute conserved density"):
            backend.apply_transport(state, dt=1.0)

    def test_huge_dt_subcycles_before_cn_can_swap_two_cell_pulse(self) -> None:
        state = _build_state(T_bath=0.0, NE=2, NX=2)
        state.x = np.array([0.0, 1.0])
        state.f[:] = 0.0
        state.f[-1, 0] = 0.2
        backend = T3Spatial1DBackend()

        ops = backend._build_transport_operators(state, dt=100.0)
        assert ops[-1] is not None
        assert ops[-1][-1] > 1

        out = backend.apply_transport(state, dt=100.0)
        pulse = out.f[-1]
        # Diffusion damps the antisymmetric mode without changing its sign:
        # the initially occupied cell cannot become less occupied than its
        # neighbour (the phase-flipped, nearly swapped full-step CN result).
        assert pulse[0] >= pulse[1] - 1.0e-14
        assert abs(pulse[0] - pulse[1]) < 1.0e-12
        np.testing.assert_allclose(np.sum(pulse), 0.2, atol=2.0e-13)

    def test_cn_subcycle_work_guard_fails_loudly(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.t3_spatial_1d as spatial_module

        state = _build_state(T_bath=0.0, NE=1, NX=2)
        state.x = np.array([0.0, 1.0])
        monkeypatch.setattr(spatial_module, "_MAX_CN_SUBSTEPS", 2)

        with pytest.raises(RuntimeError, match="monotonicity would require"):
            T3Spatial1DBackend().apply_transport(state, dt=100.0)

    def test_dynes_context_rejected_by_transport_and_collisions(self) -> None:
        # The transport dressings are clean-BCS traces (D_L indicator,
        # KL weight from real N_1/N_2, identity N_1^2 - N_2^2 = 1); a
        # Dynes-broadened context invalidates them (paper's Dynes
        # footnote) and must fail loudly. The collision kernels likewise use
        # ideal-BCS coherence functions, so swapping only the DOS is not a
        # self-consistent broadened model.
        state = _build_state()
        dynes = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            dynes_gamma=0.05,
            diffusion_coefficient=state.spectral.diffusion_coefficient,
        )
        state.spectral = dynes

        backend = T3Spatial1DBackend()
        with pytest.raises(ValueError, match="pure-BCS"):
            backend.apply_transport(state, dt=1.0)
        with pytest.raises(ValueError, match="pure-BCS"):
            backend.apply_collisions(state, dt=1.0)


class TestT3Spatial1DCollisions:
    def test_thermal_equilibrium_stays_stationary_without_flux(self) -> None:
        state = _build_state(T_bath=0.1)
        out = T3Spatial1DBackend().apply_collisions(state, dt=1.0)
        np.testing.assert_allclose(out.f, state.f, atol=1e-9)

    def test_one_end_flux_changes_source_cell_first(self) -> None:
        state = _build_state(T_bath=0.0)
        gain = np.zeros_like(state.f)
        target = int(np.argmin(np.abs(state.spectral.E - 2.0 * state.gap)))
        gain[target, 0] = 1e-4
        flux = T3SpatialFlux1D(gain=gain, loss_rate=np.zeros_like(gain))

        out = T3Spatial1DBackend().apply_collisions(
            state,
            dt=1.0,
            external_flux=flux,
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
        state = T3Spatial1DState(
            f=f,
            x=np.array([0.0, 1.0]),
            gap=gap_left,
            spectral=spectral_left,
            material=material,
            T_bath=0.1,
            gap_profile=np.array([gap_left, gap_right]),
        )

        dt = 0.1
        out = T3Spatial1DBackend().apply_collisions(state, dt)

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
        state.gap_profile = np.array([state.gap, 1.2 * state.gap])
        backend = T3Spatial1DBackend()

        first = backend.apply_collisions(state, 0.05)
        assert len(backend._collision_cache) == 2

        def unexpected_context(*_args: object, **_kwargs: object) -> SpectralContext:
            raise AssertionError("cache hit eagerly rebuilt a SpectralContext")

        monkeypatch.setattr(backend, "_local_spectral_context", unexpected_context)
        second = backend.apply_collisions(state, 0.05)

        np.testing.assert_allclose(second.f, first.f, rtol=0.0, atol=0.0)
        assert len(backend._collision_cache) == 2

    def test_collision_cache_is_lru_bounded_across_many_gaps(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        material = load_material("Al")
        E = np.linspace(150.0, 400.0, 10)
        dE = integration_widths_from_centers(E)
        backend = T3Spatial1DBackend()

        def state_at_gap(gap: float) -> T3Spatial1DState:
            spectral = SpectralContext(E, dE, gap)
            return T3Spatial1DState(
                f=np.zeros((E.size, 1)),
                x=np.array([0.0]),
                gap=gap,
                spectral=spectral,
                material=material,
                T_bath=0.0,
            )

        for gap in (120.0, 125.0, 130.0, 135.0, 140.0):
            backend.apply_collisions(state_at_gap(gap), 0.01)
            assert len(backend._collision_cache) <= 2

        assert len(backend._collision_cache) == 2
        assert [key[0][2] for key in backend._collision_cache] == [135.0, 140.0]

        # Refresh 135, then miss on 145. True LRU order evicts 140, and the
        # eviction happens before the local context/new matrices are built.
        backend.apply_collisions(state_at_gap(135.0), 0.01)
        cache_sizes_at_build: list[int] = []
        original = backend._local_spectral_context

        def capture_prebuild_size(
            base: SpectralContext, local_gap: float,
        ) -> SpectralContext:
            cache_sizes_at_build.append(len(backend._collision_cache))
            return original(base, local_gap)

        monkeypatch.setattr(backend, "_local_spectral_context", capture_prebuild_size)
        backend.apply_collisions(state_at_gap(145.0), 0.01)

        assert cache_sizes_at_build == [1]
        assert [key[0][2] for key in backend._collision_cache] == [135.0, 145.0]

    def test_collision_cache_invalidates_physics_inputs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=10, NX=1)
        backend = T3Spatial1DBackend()
        original = backend._local_spectral_context
        misses: list[tuple[float, float]] = []

        def record_miss(
            base: SpectralContext, local_gap: float,
        ) -> SpectralContext:
            misses.append((float(base.dE[0]), float(state.T_bath)))
            return original(base, local_gap)

        monkeypatch.setattr(backend, "_local_spectral_context", record_miss)
        backend.apply_collisions(state, 0.01)
        backend.apply_collisions(state, 0.01)
        assert len(misses) == 1

        state.T_bath = 0.2
        backend.apply_collisions(state, 0.01)
        state.material.tau_0 *= 1.1
        backend.apply_collisions(state, 0.01)
        state.material.T_c *= 1.01
        backend.apply_collisions(state, 0.01)
        state.spectral = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=1.01 * state.spectral.dE,
            gap=state.gap,
            diffusion_coefficient=state.spectral.diffusion_coefficient,
        )
        backend.apply_collisions(state, 0.01)

        assert len(misses) == 5

    def test_many_distinct_local_gaps_stream_exactly_with_bounded_cache(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.18, NE=14, NX=3)
        gaps = state.gap * np.array([1.0, 1.05, 1.1])
        state.gap_profile = gaps
        # Give each column a distinct, represented non-equilibrium profile.
        for column, gap in enumerate(gaps):
            local = SpectralContext(state.spectral.E, state.spectral.dE, gap)
            state.f[:, column] = np.where(
                local.active_mask,
                (column + 1.0) * 1e-3 * np.exp(-state.spectral.E / (4.0 * gap)),
                0.0,
            )

        backend = T3Spatial1DBackend()

        def unexpected_batched_path(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("three-gap profile entered batched assembly")

        monkeypatch.setattr(backend, "_collision_system", unexpected_batched_path)
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
        assert len(backend._collision_cache) == 2
        assert [key[0][2] for key in backend._collision_cache] == pytest.approx(
            gaps[-2:]
        )

        context_builds: list[float] = []
        original_context = backend._local_spectral_context

        def record_context_build(
            base: SpectralContext,
            local_gap: float,
        ) -> SpectralContext:
            context_builds.append(local_gap)
            return original_context(base, local_gap)

        monkeypatch.setattr(backend, "_local_spectral_context", record_context_build)
        repeated = backend.apply_collisions(state, dt)
        np.testing.assert_array_equal(repeated.f, out.f)
        assert context_builds == pytest.approx([gaps[0]])
        assert len(backend._collision_cache) == 2

    def test_streamed_collision_rate_matches_independent_gap_columns(self) -> None:
        state = _build_state(T_bath=0.12, NE=12, NX=4)
        gaps = state.gap * np.array([1.0, 1.03, 1.07, 1.11])
        state.gap_profile = gaps
        backend = T3Spatial1DBackend()

        rate = backend._collision_rate(state, None)
        expected = np.empty_like(rate)
        for column, gap in enumerate(gaps):
            one = _build_state(T_bath=state.T_bath, NE=12, NX=1)
            one.gap_profile = np.array([gap])
            one.f[:, 0] = state.f[:, column]
            expected[:, column] = T3Spatial1DBackend()._collision_rate(one, None)[:, 0]

        np.testing.assert_allclose(rate, expected, rtol=2e-13, atol=1e-15)
        assert len(backend._collision_cache) == 2

    def test_repeated_streamed_collision_rate_uses_resident_lru_entries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.12, NE=12, NX=4)
        gaps = state.gap * np.array([1.0, 1.03, 1.07, 1.11])
        state.gap_profile = gaps
        backend = T3Spatial1DBackend()

        first = backend._collision_rate(state, None)
        assert [key[0][2] for key in backend._collision_cache] == pytest.approx(
            gaps[-2:]
        )

        context_builds: list[float] = []
        original_context = backend._local_spectral_context

        def record_context_build(
            base: SpectralContext,
            local_gap: float,
        ) -> SpectralContext:
            context_builds.append(local_gap)
            return original_context(base, local_gap)

        monkeypatch.setattr(backend, "_local_spectral_context", record_context_build)
        second = backend._collision_rate(state, None)

        np.testing.assert_array_equal(second, first)
        # The two resident operators are consumed before either can be
        # evicted. Only the two missing gaps should be rebuilt; sorted-order
        # traversal instead caused four misses on every diagnostic call.
        assert context_builds == pytest.approx(gaps[:2])
        assert len(backend._collision_cache) == 2

    def test_streamed_profile_rejects_gain_on_late_zero_capacity_row(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=3)
        gaps = state.gap * np.array([1.0, 1.2, 1.5])
        state.gap_profile = gaps
        high = SpectralContext(state.spectral.E, state.spectral.dE, gaps[-1])
        target = int(np.flatnonzero(~high.active_mask)[0])
        gain = np.zeros_like(state.f)
        gain[target, -1] = 1e-4
        flux = T3SpatialFlux1D(gain=gain, loss_rate=np.zeros_like(gain))

        with pytest.raises(
            ValueError,
            match=rf"zero-spectral-capacity.*\({target}, 2\)",
        ):
            T3Spatial1DBackend().apply_collisions(state, 0.01, external_flux=flux)

    def test_gain_on_local_zero_capacity_row_is_rejected(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=2)
        high_gap = 1.5 * state.gap
        state.gap_profile = np.array([state.gap, high_gap])
        high_ctx = SpectralContext(
            state.spectral.E, state.spectral.dE, high_gap,
        )
        target = int(np.flatnonzero(~high_ctx.active_mask)[0])
        gain = np.zeros_like(state.f)
        gain[target, 1] = 1e-4
        flux = T3SpatialFlux1D(gain=gain, loss_rate=np.zeros_like(gain))

        with pytest.raises(ValueError, match="zero-spectral-capacity"):
            T3Spatial1DBackend().apply_collisions(
                state, 0.01, external_flux=flux,
            )

        state.f[target, 1] = 0.7
        loss_rate = np.zeros_like(state.f)
        loss_rate[target, 1] = 3.0
        loss_only = T3SpatialFlux1D(
            gain=np.zeros_like(state.f),
            loss_rate=loss_rate,
        )
        updated = T3Spatial1DBackend().apply_collisions(
            state, 0.01, external_flux=loss_only,
        )
        assert updated.f[target, 1] == state.f[target, 1]

    def test_gain_on_gap_cut_cell_is_evolved(self) -> None:
        state = _build_state(T_bath=0.0, NE=16, NX=2)
        high_gap = 1.2 * state.gap
        state.gap_profile = np.array([state.gap, high_gap])
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
        flux = T3SpatialFlux1D(gain=gain, loss_rate=np.zeros_like(gain))
        updated = T3Spatial1DBackend().apply_collisions(
            state, 0.01, external_flux=flux,
        )

        assert updated.f[target, 1] > 0.0


def _model_state(base: T3Spatial1DState, f: np.ndarray, model: DiffusionModel) -> T3Spatial1DState:
    return T3Spatial1DState(
        f=f,
        x=base.x,
        gap=base.gap,
        spectral=base.spectral,
        material=base.material,
        T_bath=0.0,
        diffusion_model=model,
    )


class TestT3Spatial1DDiffusionModels:
    def test_default_model_is_a1(self) -> None:
        assert _build_state().diffusion_model is DiffusionModel.A1

    def test_each_closure_conserves_weighted_density(self) -> None:
        backend = T3Spatial1DBackend()
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        packet = np.tile(0.3 * np.exp(-((base.x - 50.0) / 15.0) ** 2), (NE, 1))
        for model in DiffusionModel:
            state = _model_state(base, packet.copy(), model)
            weight = backend._n1_per_cell(state) ** model.p
            before = float(np.sum(weight * state.f))
            evolving = state
            for _ in range(30):
                evolving = backend.apply_transport(evolving, 1.0)
            after = float(np.sum(weight * evolving.f))
            assert abs(after - before) / abs(before) < 1e-12, model

    def test_c_path_matches_finite_volume_modal_step(self) -> None:
        base = _build_state(T_bath=0.0)
        NE, NX = base.f.shape
        f0 = np.tile(0.3 * np.exp(-((base.x - 50.0) / 15.0) ** 2), (NE, 1))
        state = _model_state(base, f0.copy(), DiffusionModel.C)
        new = T3Spatial1DBackend().apply_transport(state, dt=2.0).f

        # Modal Crank-Nicolson step with the mass-lumped finite-volume
        # D_E = D0/N1_bar closure.
        dx = state.dx
        main = -2.0 * np.ones(NX)
        main[0] = -1.0
        main[-1] = -1.0
        lap = (
            np.diag(main)
            + np.diag(np.ones(NX - 1), 1)
            + np.diag(np.ones(NX - 1), -1)
        ) / dx**2
        w, V = np.linalg.eigh(lap)
        D_E_fv = (
            state.spectral.diffusion_coefficient / state.spectral.cell_density
        )
        alpha = 0.5 * 2.0 * D_E_fv[:, None] * w[None, :]
        old = np.clip(((f0 @ V) * (1.0 + alpha) / (1.0 - alpha)) @ V.T, 0.0, 1.0)
        np.testing.assert_allclose(new, old, atol=1e-12)

    def test_a1_and_c_coincide_at_uniform_gap(self) -> None:
        # A1 (conserves N1 f, undressed flux) and C (bare f, 1/N1-dressed
        # flux) share the uniform-gap rate D0/N1, so with N1 x-independent
        # their f-dynamics are identical -- the dirty-Usadel and clean/BRT
        # reductions agree at a uniform gap.
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        f0 = np.tile(0.3 * np.exp(-((base.x - 50.0) / 15.0) ** 2), (NE, 1))
        out = {
            model: T3Spatial1DBackend().apply_transport(
                _model_state(base, f0.copy(), model), 2.0
            ).f
            for model in (DiffusionModel.A1, DiffusionModel.C)
        }
        np.testing.assert_allclose(
            out[DiffusionModel.A1], out[DiffusionModel.C], atol=1e-12
        )

    def test_a1_and_a1p_dynamics_differ(self) -> None:
        # The diagnostic A1P carries the transverse N1^2 dressing and
        # separates from A1 already at a uniform gap.
        base = _build_state(T_bath=0.0)
        NE, _NX = base.f.shape
        f0 = np.tile(0.3 * np.exp(-((base.x - 50.0) / 15.0) ** 2), (NE, 1))
        out = {
            model: T3Spatial1DBackend().apply_transport(
                _model_state(base, f0.copy(), model), 2.0
            ).f
            for model in (DiffusionModel.A1, DiffusionModel.A1P)
        }
        assert np.max(np.abs(out[DiffusionModel.A1] - out[DiffusionModel.A1P])) > 1e-3


def _varying_gap_setup(*, NE: int = 16, NX: int = 21, interface: bool = False):
    material = load_material("Al")
    base_gap = material.Delta_0
    gap_max = 1.6 * base_gap
    E, _ = build_energy_grid(
        gap=gap_max, energy_min_factor=1.05, energy_max_factor=4.0, num_energy_bins=NE
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap_max,
        diffusion_coefficient=6.0,
    )
    x = np.linspace(0.0, 100.0, NX)
    if interface:
        profile = np.where(np.arange(NX) < NX // 2, gap_max, base_gap).astype(float)
    else:
        profile = np.linspace(base_gap, gap_max, NX)
    return material, spectral, x, gap_max, profile


class TestT3Spatial1DVaryingGap:
    def test_gap_profile_shape_validation(self) -> None:
        material, spectral, x, gap_max, _ = _varying_gap_setup()
        NE = spectral.E.size
        bad = T3Spatial1DState(
            f=np.zeros((NE, x.size)),
            x=x,
            gap=gap_max,
            spectral=spectral,
            material=material,
            T_bath=0.1,
            gap_profile=np.ones(x.size + 1),
        )
        with pytest.raises(ValueError, match="gap_profile"):
            T3Spatial1DBackend().apply_transport(bad, 1.0)

    def test_degenerate_and_descending_mesh_rejected(self) -> None:
        # A constant mesh (dx=0) and a descending mesh (dx<0) both pass an
        # equal-diffs test yet corrupt every dx-divided flux downstream.
        state = _build_state()
        backend = T3Spatial1DBackend()
        for bad_x in (np.zeros(state.x.size), state.x[::-1].copy()):
            state.x = bad_x
            with pytest.raises(ValueError, match="strictly increasing"):
                backend.apply_transport(state, 1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_nonfinite_mesh_or_bath_is_rejected(self, bad: float) -> None:
        state = _build_state()
        state.x[0] = bad
        with pytest.raises(ValueError, match=r"state\.x"):
            T3Spatial1DBackend().apply_transport(state, 1.0)

        state = _build_state()
        state.T_bath = bad
        with pytest.raises(ValueError, match="T_bath"):
            T3Spatial1DBackend().apply_collisions(state, 1.0)

    def test_state_and_spectral_gap_must_match(self) -> None:
        state = _build_state()
        state.gap *= 0.9
        with pytest.raises(ValueError, match="must match"):
            T3Spatial1DBackend().apply_transport(state, 1.0)

    def test_negative_interface_conductance_rejected(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        for bad in (-1.0, -np.inf, np.inf, np.nan):
            state = T3Spatial1DState(
                f=np.zeros((NE, NX)), x=x, gap=gap_max, spectral=spectral,
                material=material, T_bath=0.1, gap_profile=profile,
                interface_conductance=bad,
            )
            with pytest.raises(ValueError, match="interface_conductance"):
                T3Spatial1DBackend().apply_transport(state, 0.5)

    def test_interface_conductance_rejects_smooth_gap_profile(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup()
        state = T3Spatial1DState(
            f=np.zeros((spectral.E.size, x.size)),
            x=x,
            gap=gap_max,
            spectral=spectral,
            material=material,
            T_bath=0.1,
            gap_profile=profile,
            interface_conductance=2.0,
        )

        with pytest.raises(ValueError, match="exactly one step"):
            T3Spatial1DBackend().apply_transport(state, 0.5)

    def test_interface_conductance_without_profile_is_rejected(self) -> None:
        state = _build_state()
        state.interface_conductance = 2.0

        with pytest.raises(ValueError, match="requires a gap_profile"):
            T3Spatial1DBackend().apply_transport(state, 0.5)

    def test_zero_interface_conductance_is_an_opaque_interface(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        f0 = np.zeros((NE, NX))
        face = NX // 2 - 1
        f0[:, : face + 1] = 0.4
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral,
            material=material, T_bath=0.1, diffusion_model=DiffusionModel.A1,
            gap_profile=profile, interface_conductance=0.0,
        )

        out = T3Spatial1DBackend().apply_transport(state, 0.5)

        # G_N = 0 is the physical opaque-interface limit.  It is distinct
        # from None, which leaves the step face on the bulk-diffusion path.
        np.testing.assert_array_equal(out.f[:, face + 1 :], 0.0)
        np.testing.assert_allclose(out.f[:, : face + 1], f0[:, : face + 1])

    def test_varying_gap_conserves_weighted_density(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup()
        NE = spectral.E.size
        f0 = np.tile(0.2 * np.exp(-((x - 30.0) / 18.0) ** 2), (NE, 1))
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
        )
        N1 = T3Spatial1DBackend()._n1_per_cell(state)
        before = float(np.sum(N1 * state.f))  # p = 1 for A1
        backend = T3Spatial1DBackend()
        evolving = state
        for _ in range(30):
            evolving = backend.apply_transport(evolving, 1.0)
        after = float(np.sum(N1 * evolving.f))
        assert abs(after - before) / abs(before) < 1e-12

    def test_a1_and_c_differ_in_gap_ramp(self) -> None:
        # A1 and C coincide at a uniform gap but separate once the gap
        # varies: A1 has no DOS-gradient drift (the spectral factor sits
        # outside the divergence), C dresses the flux inside it.
        material, spectral, x, gap_max, profile = _varying_gap_setup()
        NE = spectral.E.size
        f0 = np.tile(0.2 * np.exp(-((x - 30.0) / 18.0) ** 2), (NE, 1))

        def step(model: DiffusionModel) -> np.ndarray:
            state = T3Spatial1DState(
                f=f0.copy(), x=x, gap=gap_max, spectral=spectral,
                material=material, T_bath=0.1, diffusion_model=model,
                gap_profile=profile,
            )
            out = state
            backend = T3Spatial1DBackend()
            for _ in range(10):
                out = backend.apply_transport(out, 1.0)
            return out.f

        assert np.max(np.abs(step(DiffusionModel.A1) - step(DiffusionModel.C))) > 1e-4

    def test_interface_conserves_and_jumps(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
            interface_conductance=2.0,
        )
        N1 = T3Spatial1DBackend()._n1_per_cell(state)
        before = float(np.sum(N1 * state.f))
        out = T3Spatial1DBackend().apply_transport(state, 0.5)
        after = float(np.sum(N1 * out.f))
        assert abs(after - before) / abs(before) < 1e-12  # current continuity
        k, e = NX // 2 - 1, NE - 1
        assert out.f[e, k] > out.f[e, k + 1] + 1e-6  # f-discontinuity at the interface

    def test_interface_differs_from_bulk(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4

        def step(conductance: float | None) -> np.ndarray:
            state = T3Spatial1DState(
                f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
                T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
                interface_conductance=conductance,
            )
            return T3Spatial1DBackend().apply_transport(state, 0.5).f

        assert np.max(np.abs(step(0.1) - step(None))) > 1e-3

    def test_a1_a2_distinct_under_interface_relaxation(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4
        backend = T3Spatial1DBackend()

        def relax(model: DiffusionModel) -> np.ndarray:
            state = T3Spatial1DState(
                f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
                T_bath=0.1, diffusion_model=model, gap_profile=profile,
                interface_conductance=2.0,
            )
            for _ in range(1500):
                state = backend.apply_transport(state, 2.0)
            return state.f

        diff = np.max(np.abs(relax(DiffusionModel.A1) - relax(DiffusionModel.A2)))
        assert diff > 1e-3


def _kl_weight(E: np.ndarray, gap_L: float, gap_R: float) -> np.ndarray:
    """Analytic KL energy weight 𝒲_L = N₁N₁′ − N₂N₂′ (eq:scalar_BC_energy)."""
    from qpsim.physics.spectral import bcs_anomalous_weight

    return (
        bcs_density_of_states(E, gap_L) * bcs_density_of_states(E, gap_R)
        - bcs_anomalous_weight(E, gap_L) * bcs_anomalous_weight(E, gap_R)
    )


def _kl_cell_average_reference(
    E: np.ndarray,
    dE: np.ndarray,
    index: int,
    gap_L: float,
    gap_R: float,
) -> float:
    """Independent direct-energy quadrature of one KL finite volume."""
    edges = cell_edges_from_widths(E, dE)
    lo = max(float(edges[index]), gap_L, gap_R)
    hi = float(edges[index + 1])
    if hi <= lo:
        return 0.0
    if gap_L == gap_R:
        return (hi - lo) / dE[index]

    def integrand(energy: float) -> float:
        return (energy * energy - gap_L * gap_R) / np.sqrt(
            (energy * energy - gap_L * gap_L)
            * (energy * energy - gap_R * gap_R)
        )

    value, _error = quad(
        integrand,
        lo,
        hi,
        points=[lo],
        epsabs=1e-11,
        epsrel=1e-11,
    )
    return value / dE[index]


class TestKupriyanovLukichevWeightFixtures:
    """Paper fixtures for the KL energy-channel weight (eq:scalar_BC_energy).

    𝒲_L = N₁N₁′ − N₂N₂′ must be exactly 1 at matched gaps (the
    coherence-factor cancellation that removes the SIS matched-gap
    singularity carried by the charge-channel product N₁N₁′) and reduce
    to N₁ against a normal contact.
    """

    def test_matched_gap_weight_is_one(self) -> None:
        gap = 180.0
        E = np.linspace(gap * 1.0001, gap * 8.0, 4000)
        W = _kl_weight(E, gap, gap)
        np.testing.assert_allclose(W, 1.0, rtol=1e-9)

    def test_normal_contact_weight_is_N1(self) -> None:
        gap = 180.0
        E = np.linspace(gap * 1.0001, gap * 8.0, 4000)
        W = _kl_weight(E, gap, 0.0)
        np.testing.assert_allclose(W, bcs_density_of_states(E, gap), rtol=1e-12)

    def test_sub_gap_side_closes_interface(self) -> None:
        gap_L, gap_R = 288.0, 180.0
        E = np.linspace(gap_R * 1.001, gap_L * 0.999, 200)  # above R, below L
        np.testing.assert_array_equal(_kl_weight(E, gap_L, gap_R), 0.0)

    def test_backend_face_carries_exact_weight(self) -> None:
        # Extract the interface face conductance from the assembled CN
        # operator: B = I + (dt/2)·L·diag(1/ρ_p), so the (m, m+1) entry
        # is (dt/2)·g_face[m]/ρ_p[m+1] with g_face = G_N·𝒲_L/dx at the
        # gap step.
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        G_N, dt = 2.0, 0.5
        state = T3Spatial1DState(
            f=np.zeros((NE, NX)), x=x, gap=gap_max, spectral=spectral,
            material=material, T_bath=0.1, diffusion_model=DiffusionModel.A1,
            gap_profile=profile, interface_conductance=G_N,
        )
        ops = T3Spatial1DBackend()._build_transport_operators(state, dt)
        dx = state.dx
        face = NX // 2 - 1  # the gap steps between cells face, face+1
        gap_L, gap_R = float(profile[face]), float(profile[face + 1])
        checked = 0
        for i, op in enumerate(ops):
            if op is None:
                continue
            b_mat, _, idx, rho_p, n_substeps = op
            if idx[0] != 0 or idx[-1] != NX - 1:
                continue
            m = face
            W_expected = _kl_cell_average_reference(
                spectral.E, spectral.dE, i, gap_L, gap_R,
            )
            B = b_mat.toarray()
            sub_dt = dt / n_substeps
            W_measured = (
                B[m, m + 1]
                * rho_p[m + 1]
                * dx
                / (0.5 * sub_dt * G_N)
            )
            np.testing.assert_allclose(W_measured, W_expected, rtol=1e-10)
            checked += 1
        assert checked > 0


class TestGapEdgePacketFixture:
    """Paper fixture: a packet pushed against a spatial gap ramp must
    conserve ∫N₁f with zero leakage past the local gap edge (the
    weak-form zero-flux face of paper §V — diffusive Andreev
    retroreflection for the energy mode)."""

    def test_packet_conserves_with_zero_subedge_leakage(self) -> None:
        # Custom grid: the energies must span the gap band
        # (base_gap, gap_max) so mid-band energies have their local edge
        # inside the strip (the shared helper's grid starts above
        # gap_max and would never see an edge).
        material = load_material("Al")
        base_gap = material.Delta_0
        gap_max = 1.6 * base_gap
        NE, NX = 24, 41
        E, _ = build_energy_grid(
            gap=base_gap, energy_min_factor=1.02,
            energy_max_factor=4.8, num_energy_bins=NE,
        )
        spectral = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E),
            gap=gap_max, diffusion_coefficient=6.0,
        )
        x = np.linspace(0.0, 100.0, NX)
        profile = np.linspace(base_gap, gap_max, NX)
        # Packet near the low-gap end; diffusion pushes it up the ramp
        # into each energy's local edge.
        f0 = np.tile(0.3 * np.exp(-(((x - 15.0) / 8.0) ** 2)), (NE, 1))
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
        )
        backend = T3Spatial1DBackend()
        N1 = backend._n1_per_cell(state)
        state.f[N1 == 0.0] = 0.0  # no occupation below the local edge
        before = (N1 * state.f).sum(axis=1)  # per-energy conserved density

        evolving = state
        for _ in range(200):
            evolving = backend.apply_transport(evolving, 1.0)

        after = (N1 * evolving.f).sum(axis=1)
        sub_edge = N1 == 0.0

        # mid-band energies (edge inside the grid) must have hit the edge
        mid_band = (profile.min() < spectral.E) & (profile.max() > spectral.E)
        assert mid_band.any()
        hit = 0
        for i in np.flatnonzero(mid_band):
            active = np.flatnonzero(N1[i] > 0.0)
            if active.size and evolving.f[i, active[-1]] > 1e-6:
                hit += 1
        assert hit > 0

        # exact conservation of the per-energy ∫N₁f and zero leakage
        nz = before > 0
        np.testing.assert_allclose(after[nz], before[nz], rtol=1e-11)
        assert float(np.abs(evolving.f[sub_edge]).max(initial=0.0)) == 0.0


class TestTransportCacheKeying:
    """Regression: the operator/kernel caches must key on the spectral
    CONTENT, not on grid shape or object identity — a backend reused
    across states with different energy grids previously crashed
    (smaller NE) or silently froze the extra rows (larger NE)."""

    def test_backend_reused_across_different_energy_grids(self) -> None:
        from dataclasses import replace

        backend = T3Spatial1DBackend()
        for NE in (28, 20, 36):
            state = _build_state(T_bath=0.1, NE=NE)
            f = state.f.copy()
            f[:, state.x.size // 2] = np.clip(
                f[:, state.x.size // 2] + 0.1, 0.0, 1.0
            )
            state = replace(state, f=f)
            out = backend.apply_transport(state, dt=0.05)
            moved = np.abs(out.f - state.f).max(axis=1)
            # Every energy row is above the gap in _build_state, so every
            # row must diffuse — a stale cached operator for a previous
            # grid leaves the tail rows exactly untouched.
            assert moved.min() > 0.0, (
                f"NE={NE}: some energy rows did not diffuse "
                f"(stale transport-operator cache reused across grids)"
            )


class TestRunUntilSteadyStateResidual:
    def test_transport_and_source_rates_cancel_before_norm(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _build_state(T_bath=0.0, NE=4, NX=3)
        backend = T3Spatial1DBackend()
        monkeypatch.setattr(
            backend, "step", lambda state_arg, _dt, **_kwargs: state_arg,
        )
        monkeypatch.setattr(
            backend,
            "_transport_rate",
            lambda state_arg, _dt: np.ones_like(state_arg.f),
        )
        monkeypatch.setattr(
            backend,
            "_collision_rate",
            lambda state_arg, _flux: -np.ones_like(state_arg.f),
        )

        result = backend.run_until_steady_state(
            state,
            dt=1.0,
            max_time=2.0,
            stop_tol=1.0e-12,
            snapshot_interval=2.0,
        )

        assert result.converged
        assert result.n_steps == 1
        assert result.snapshots[-1].max_rate == 0.0

    def test_constant_thermal_state_stops_on_raw_operator_residual(self) -> None:
        state = _build_state(T_bath=0.1, NE=12, NX=5)
        result = T3Spatial1DBackend().run_until_steady_state(
            state,
            dt=1.0,
            max_time=10.0,
            stop_tol=1.0e-6,
            snapshot_interval=10.0,
        )

        assert result.converged
        assert result.n_steps == 1
        assert result.snapshots[-1].max_rate < 1.0e-6

    def test_saturated_positive_source_does_not_claim_convergence(self) -> None:
        state = _build_state(T_bath=0.0, NE=8, NX=3)
        state.f[:] = 1.0
        source = T3SpatialFlux1D(
            gain=np.full_like(state.f, 1.0e6),
            loss_rate=np.zeros_like(state.f),
        )

        result = T3Spatial1DBackend().run_until_steady_state(
            state,
            dt=0.1,
            max_time=0.3,
            stop_tol=1.0,
            snapshot_interval=0.3,
            external_flux=source,
        )

        assert not result.converged
        assert result.n_steps == 3
        np.testing.assert_allclose(result.state.f, 1.0, rtol=0.0, atol=1.0e-15)
        assert result.snapshots[-1].max_rate > 1.0


class TestRunUntilSteadyStateProgressHook:
    """The driver hook is physics-neutral: reporting only, plus a clean
    cooperative early stop when it returns False."""

    @staticmethod
    def _perturbed_state() -> T3Spatial1DState:
        state = _build_state(NE=12, NX=7)
        f = state.f.copy()
        f[:, 0] = np.clip(f[:, 0] + 0.05, 0.0, 1.0)
        state.f = f
        return state

    def test_none_hook_bit_for_bit_unchanged(self) -> None:
        backend = T3Spatial1DBackend()
        baseline = backend.run_until_steady_state(
            self._perturbed_state(), dt=1.0, max_time=6.0, stop_tol=0.0,
        )
        hooked = backend.run_until_steady_state(
            self._perturbed_state(), dt=1.0, max_time=6.0, stop_tol=0.0,
            progress_hook=lambda _t, _total: True,
        )
        np.testing.assert_array_equal(hooked.state.f, baseline.state.f)
        assert hooked.n_steps == baseline.n_steps
        assert hooked.total_time == baseline.total_time

    def test_hook_called_once_per_step_with_times(self) -> None:
        calls: list[tuple[float, float]] = []

        def hook(t: float, total: float) -> bool:
            calls.append((t, total))
            return True

        result = T3Spatial1DBackend().run_until_steady_state(
            self._perturbed_state(), dt=1.0, max_time=5.0, stop_tol=0.0,
            progress_hook=hook,
        )
        assert len(calls) == result.n_steps == 5
        assert all(total == pytest.approx(5.0) for _, total in calls)

    def test_false_return_stops_early(self) -> None:
        result = T3Spatial1DBackend().run_until_steady_state(
            self._perturbed_state(), dt=1.0, max_time=100.0, stop_tol=0.0,
            snapshot_interval=50.0,
            progress_hook=lambda t, _total: t < 3.0,
        )
        assert result.n_steps == 3
        assert result.total_time == pytest.approx(3.0)
        assert not result.converged
        assert result.snapshots[-1].t == pytest.approx(3.0)

    def test_dt_larger_than_interval_emits_every_snapshot(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = T3Spatial1DBackend()
        monkeypatch.setattr(
            backend,
            "step",
            lambda state, _dt, **_kwargs: state,
        )

        result = backend.run_until_steady_state(
            self._perturbed_state(),
            dt=5.0,
            max_time=10.0,
            stop_tol=0.0,
            snapshot_interval=2.0,
        )

        np.testing.assert_allclose(
            [snapshot.t for snapshot in result.snapshots],
            [0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        )

    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            ("dt", {"dt": float("nan")}),
            ("max_time", {"max_time": float("inf")}),
            ("stop_tol", {"stop_tol": float("nan")}),
            ("snapshot_interval", {"snapshot_interval": float("inf")}),
        ],
    )
    def test_nonfinite_driver_controls_are_rejected(
        self, field: str, kwargs: dict[str, float],
    ) -> None:
        values = {"dt": 1.0, "max_time": 2.0, "stop_tol": 0.0}
        values.update(kwargs)
        with pytest.raises(ValueError, match=field):
            T3Spatial1DBackend().run_until_steady_state(
                self._perturbed_state(),
                **values,
            )
