"""Tests for the 1D spatial T3 diffusion preview backend."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_spatial_1d import (
    T3Spatial1DBackend,
    T3Spatial1DState,
    T3SpatialFlux1D,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext, bcs_density_of_states
from qpsim.transport.diffusion.base import DiffusionModel


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
    def test_reflective_transport_preserves_uniform_field(self) -> None:
        state = _build_state()
        out = T3Spatial1DBackend().apply_transport(state, dt=2.0)
        np.testing.assert_allclose(out.f, state.f, atol=1e-13)

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
            weight = base.spectral.rho[:, None] ** model.p
            before = float(np.sum(weight * state.f))
            evolving = state
            for _ in range(30):
                evolving = backend.apply_transport(evolving, 1.0)
            after = float(np.sum(weight * evolving.f))
            assert abs(after - before) / abs(before) < 1e-12, model

    def test_c_path_matches_legacy_modal_step(self) -> None:
        base = _build_state(T_bath=0.0)
        NE, NX = base.f.shape
        f0 = np.tile(0.3 * np.exp(-((base.x - 50.0) / 15.0) ** 2), (NE, 1))
        state = _model_state(base, f0.copy(), DiffusionModel.C)
        new = T3Spatial1DBackend().apply_transport(state, dt=2.0).f

        # Legacy modal Crank-Nicolson step with the D_E = D0/N1 closure.
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
        alpha = 0.5 * 2.0 * state.spectral.D_E[:, None] * w[None, :]
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

    def test_varying_gap_conserves_weighted_density(self) -> None:
        material, spectral, x, gap_max, profile = _varying_gap_setup()
        NE = spectral.E.size
        N1 = np.column_stack([bcs_density_of_states(spectral.E, float(g)) for g in profile])
        f0 = np.tile(0.2 * np.exp(-((x - 30.0) / 18.0) ** 2), (NE, 1))
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
        )
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
        NE, NX = spectral.E.size, x.size
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
        N1 = np.column_stack([bcs_density_of_states(spectral.E, float(g)) for g in profile])
        f0 = np.zeros((NE, NX))
        f0[:, : NX // 2] = 0.4
        state = T3Spatial1DState(
            f=f0.copy(), x=x, gap=gap_max, spectral=spectral, material=material,
            T_bath=0.1, diffusion_model=DiffusionModel.A1, gap_profile=profile,
            interface_conductance=2.0,
        )
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
