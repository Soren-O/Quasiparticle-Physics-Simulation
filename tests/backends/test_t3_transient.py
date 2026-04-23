"""Tests for the T3 backend transient methods (Strang step + apply_*)."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _state_on_physics_grid(
    T_bath: float = 0.3, num_energy: int = 20,
) -> T3DiffusionState:
    """Build a state whose PhononState sits on the physics ω grid."""
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c

    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)

    # Physics ω grid derived from E (matches what apply_collisions
    # expects and what steady_state would produce).
    omega, _, _, _ = build_phonon_frequency_map(E)
    n_ph_thermal = thermal_phonon_occupation(omega, T_bath)

    phonon = PhononState(
        n_ph=n_ph_thermal.reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), 0.25),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )

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


class TestApplyCollisions:
    def test_thermal_is_near_fixed_point(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.apply_collisions(state, dt=0.01)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-8)

    def test_preserves_bounds(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = T3DiffusionBackend()
        # Perturb f to stress the stepper.
        state.f = np.clip(  # type: ignore[misc]
            state.f * (1.0 + 0.3 * np.cos(state.spectral.E / state.gap)),
            0.0, 1.0,
        )
        new_state = backend.apply_collisions(state, dt=0.05)
        assert np.all(new_state.f >= 0.0)
        assert np.all(new_state.f <= 1.0)

    def test_phonon_unchanged(self) -> None:
        # apply_collisions freezes n_ph; transient phonon dynamics are
        # out of Gate 2 scope.
        state = _state_on_physics_grid()
        backend = T3DiffusionBackend()
        new_state = backend.apply_collisions(state, dt=0.01)
        assert new_state.phonon is state.phonon

    def test_rejects_off_grid_phonon(self) -> None:
        # PhononState with a non-physics ω grid must be rejected.
        state = _state_on_physics_grid()
        off_grid_omega = np.linspace(0.1, 5.0, 40)  # wrong length + values
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=np.zeros((1, off_grid_omega.size, 1)),
            omega_bins=off_grid_omega.reshape(1, -1),
            tau_l=np.full((1, off_grid_omega.size), 0.25),
            model=state.phonon.model,
            branches=state.phonon.branches,
        )
        with pytest.raises(ValueError, match="physics grid"):
            T3DiffusionBackend().apply_collisions(state, dt=0.01)


class TestApplyTransport:
    def test_is_no_op_for_homogeneous(self) -> None:
        state = _state_on_physics_grid()
        backend = T3DiffusionBackend()
        new_state = backend.apply_transport(state, dt=0.1)
        assert new_state is state  # identical object — nothing moved

    def test_rejects_non_1d_f(self) -> None:
        state = _state_on_physics_grid()
        state.f = state.f[:, None]  # type: ignore[misc] — fake spatial axis
        with pytest.raises(NotImplementedError, match="spatial transport"):
            T3DiffusionBackend().apply_transport(state, dt=0.1)


class TestApplyGapUpdate:
    def test_near_thermal_minimal_change(self) -> None:
        # At near-thermal f with a gap set to the BCS Δ(0), the solve
        # should return a very similar gap (calibration at T_bath uses
        # Δ_eq(T_bath), which is close to Δ(0) for low T_bath / T_c).
        state = _state_on_physics_grid(T_bath=0.01)  # very low T
        gap_before = state.gap
        backend = T3DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.1)
        rel_change = abs(new_state.gap - gap_before) / gap_before
        assert rel_change < 0.01  # within 1%

    def test_zero_dt_is_no_op(self) -> None:
        state = _state_on_physics_grid()
        backend = T3DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.0)
        assert new_state is state

    def test_preserves_f_bounds(self) -> None:
        state = _state_on_physics_grid()
        backend = T3DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.01)
        assert np.all(new_state.f >= 0.0)
        assert np.all(new_state.f <= 1.0)


class TestStep:
    def test_thermal_is_near_fixed_point(self) -> None:
        # A full Strang step on the thermal distribution should barely
        # move it (f_FD is the fixed point of the collision integral,
        # and Δ_eq(T_bath) is the fixed point of the gap equation).
        state = _state_on_physics_grid(T_bath=0.01)
        backend = T3DiffusionBackend()
        new_state = backend.step(state, dt=0.01)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-4)

    def test_multiple_steps_remain_bounded(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = T3DiffusionBackend()
        for _ in range(5):
            state = backend.step(state, dt=0.01)
            assert np.all(state.f >= 0.0)
            assert np.all(state.f <= 1.0)
            assert state.gap > 0
