"""End-to-end smoke test for the T3 diffusion backend.

Doubles as the Gate 2 task 13 integration test: build every piece
from scratch (Material, grid, spectral context, phonon state, T3
state) and exercise the full steady-state pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.base import Tier
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.spectral import SpectralContext


def _build_state(T_bath: float = 0.3, num_energy: int = 30) -> T3DiffusionState:
    """Build a homogeneous Al-like T3 state at thermal equilibrium."""
    material = load_material("Al")
    # Use the BCS Δ(0) as the gap for this test (T_bath ≪ T_c).
    gap = 1.764 * KB_UEV_PER_K * material.T_c

    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)

    # Phonon grid and state: single-branch, spatially homogeneous.
    omega_grid = np.linspace(0.1 * gap, 5.0 * gap, 40)  # arbitrary ω range
    phonon = PhononState(
        n_ph=np.zeros((1, omega_grid.size, 1)),
        omega_bins=omega_grid.reshape(1, -1),
        tau_l=np.full((1, omega_grid.size), 0.25),  # 0.25 ns
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )

    # Fermi-Dirac initial f.
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    return T3DiffusionState(
        f=f_init,
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


class TestT3DiffusionBackendSteadyState:
    def test_thermal_equilibrium_is_fixed_point(self) -> None:
        # Start at Fermi-Dirac at T_bath; the steady-state solve should
        # barely move it, since f_FD(T_bath) is the fixed point of the
        # e-ph collision integral with thermal phonons.
        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-6)

    def test_perturbed_initial_converges_back_to_thermal(self) -> None:
        # Perturb the initial f smoothly; steady state should return to
        # the thermal distribution (within solver tolerance).
        state = _build_state(T_bath=0.3)
        kT = KB_UEV_PER_K * state.T_bath
        f_FD = 1.0 / (np.exp(np.minimum(state.spectral.E / kT, 500.0)) + 1.0)
        perturbed_f = np.clip(
            f_FD * (1.0 + 0.3 * np.sin(state.spectral.E / state.gap)),
            0.0, 1.0,
        )
        state.f = perturbed_f  # type: ignore[misc]
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        np.testing.assert_allclose(new_state.f, f_FD, atol=1e-4)

    def test_returns_new_state_with_updated_f(self) -> None:
        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        # All other fields are identical to the input state.
        assert new_state.tier == Tier.T3_DIFFUSION
        assert new_state.gap == state.gap
        assert new_state.T_bath == state.T_bath
        assert new_state.material is state.material
        assert new_state.spectral is state.spectral
        assert new_state.phonon is state.phonon
        # f is a fresh array.
        assert new_state.f is not state.f

    def test_f_preserved_shape(self) -> None:
        state = _build_state(T_bath=0.3, num_energy=25)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        assert new_state.f.shape == state.f.shape


class TestT3DiffusionState:
    def test_default_tier(self) -> None:
        state = _build_state()
        assert state.tier == Tier.T3_DIFFUSION

    def test_carries_material(self) -> None:
        state = _build_state()
        assert state.material.name == "Al"
        # τ_0 from the Al YAML.
        assert state.material.tau_0 == pytest.approx(438.0)
