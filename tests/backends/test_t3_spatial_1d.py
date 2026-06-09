"""Tests for the 1D spatial T3 diffusion preview backend."""

from __future__ import annotations

import numpy as np
from qpsim.backends.t3_spatial_1d import (
    T3Spatial1DBackend,
    T3Spatial1DState,
    T3SpatialFlux1D,
)
from qpsim.constants import KB_UEV_PER_K
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
