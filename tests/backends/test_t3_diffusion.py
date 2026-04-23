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

    def test_returns_new_state_with_updated_f_and_phonon(self) -> None:
        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        # Scalar / reference-identity fields unchanged.
        assert new_state.tier == Tier.T3_DIFFUSION
        assert new_state.gap == state.gap
        assert new_state.T_bath == state.T_bath
        assert new_state.material is state.material
        assert new_state.spectral is state.spectral
        # f and phonon are freshly built to reflect the solver output.
        assert new_state.f is not state.f
        assert new_state.phonon is not state.phonon

    def test_returned_phonon_has_converged_n_ph(self) -> None:
        # The input state seeds n_ph = 0, but a thermal solve at
        # T_bath > 0 must converge to a non-zero n_ph close to the
        # Bose-Einstein distribution.
        from qpsim.physics.kernels import thermal_phonon_occupation

        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)

        # Non-zero; not the stale input.
        assert not np.allclose(new_state.phonon.n_ph, 0.0)

        # At thermal equilibrium, n_ph should match n_BE on the physics grid.
        omega = new_state.phonon.omega_bins[0]
        n_th = thermal_phonon_occupation(omega, state.T_bath)
        np.testing.assert_allclose(new_state.phonon.n_ph[0, :, 0], n_th, atol=1e-10)

    def test_returned_phonon_is_on_physics_omega_grid(self) -> None:
        # Verify the input ω grid was ignored and the output reflects
        # the physics grid (pair-sum / pair-difference of E).
        from qpsim.collisions.phonon import build_phonon_frequency_map

        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)

        expected_omega, _, _, _ = build_phonon_frequency_map(state.spectral.E)
        np.testing.assert_allclose(new_state.phonon.omega_bins[0], expected_omega)

    def test_f_preserved_shape(self) -> None:
        state = _build_state(T_bath=0.3, num_energy=25)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(state)
        assert new_state.f.shape == state.f.shape


class TestT3DiffusionBackendScopeValidation:
    def test_rejects_multi_branch(self) -> None:
        state = _build_state()
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=np.zeros((2, state.phonon.n_omega, 1)),
            omega_bins=np.tile(state.phonon.omega_bins[0], (2, 1)),
            tau_l=np.full((2, state.phonon.n_omega), 0.25),
            model=state.phonon.model,
            branches=[
                PhononBranchSpec(name="longitudinal"),
                PhononBranchSpec(name="transverse"),
            ],
        )
        with pytest.raises(ValueError, match="single-branch"):
            T3DiffusionBackend().steady_state(state)

    def test_rejects_multi_spatial(self) -> None:
        state = _build_state()
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=np.zeros((1, state.phonon.n_omega, 3)),
            omega_bins=state.phonon.omega_bins,
            tau_l=state.phonon.tau_l,
            model=state.phonon.model,
            branches=state.phonon.branches,
        )
        with pytest.raises(ValueError, match="spatially-homogeneous"):
            T3DiffusionBackend().steady_state(state)

    def test_rejects_non_constant_tau_l(self) -> None:
        state = _build_state()
        # Non-uniform τ_l(ω).
        varied = np.linspace(0.1, 0.5, state.phonon.n_omega).reshape(1, -1)
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=state.phonon.n_ph,
            omega_bins=state.phonon.omega_bins,
            tau_l=varied,
            model=state.phonon.model,
            branches=state.phonon.branches,
        )
        with pytest.raises(ValueError, match="constant-τ_l"):
            T3DiffusionBackend().steady_state(state)


class TestT3DiffusionState:
    def test_default_tier(self) -> None:
        state = _build_state()
        assert state.tier == Tier.T3_DIFFUSION

    def test_carries_material(self) -> None:
        state = _build_state()
        assert state.material.name == "Al"
        # τ_0 from the Al YAML.
        assert state.material.tau_0 == pytest.approx(438.0)
