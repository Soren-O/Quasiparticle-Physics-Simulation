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
    gap = material.Delta_0

    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=0.75, energy_max_factor=6.0,
        num_energy_bins=num_energy,
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


def _state_on_custom_grid(
    E: np.ndarray,
    *,
    gap: float = 1.0,
    f: np.ndarray | None = None,
    dynes_gamma: float = 0.0,
) -> T3DiffusionState:
    """Reuse the lightweight phonon fixture with a prescribed QP grid."""
    state = _state_on_physics_grid(T_bath=0.1, num_energy=E.size)
    dE = integration_widths_from_centers(E)
    state.gap = gap
    state.spectral = SpectralContext(
        E_bins=E,
        dE_bins=dE,
        gap=gap,
        dynes_gamma=dynes_gamma,
    )
    state.f = (
        np.asarray(f, dtype=float)
        if f is not None
        else np.where(gap < E, 0.1, 0.0)
    )
    return state


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


    def test_nonuniform_grid_conserves_dE_weighted_mass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.array([0.8, 0.9, 1.01, 1.04, 1.3, 1.6, 2.0])
        state = _state_on_custom_grid(E)
        dE = state.spectral.dE
        mass_before = float(np.sum(state.spectral.rho * state.f * dE))
        monkeypatch.setattr(
            "qpsim.backends.t3_diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.02,
        )

        with pytest.warns(UserWarning, match="conservatively remapped"):
            out = T3DiffusionBackend().apply_gap_update(state, dt=1.0)

        mass_after = float(np.sum(out.spectral.rho * out.f * dE))
        assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-15)

    def test_edge_remap_spreads_mass_without_saturating_one_bin(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(0.8, 2.0, 81)
        f = np.zeros_like(E)
        f[(E > 1.0) & (E < 1.08)] = 0.9
        state = _state_on_custom_grid(E, f=f)
        mass_before = float(
            np.sum(state.spectral.rho * state.f * state.spectral.dE)
        )
        monkeypatch.setattr(
            "qpsim.backends.t3_diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.1,
        )

        with pytest.warns(UserWarning, match="conservatively remapped"):
            out = T3DiffusionBackend().apply_gap_update(state, dt=1.0)

        mass_after = float(
            np.sum(out.spectral.rho * out.f * out.spectral.dE)
        )
        assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-15)
        assert np.count_nonzero(out.f > 1e-10) >= 2
        assert float(np.max(out.f)) < 1.0

    @pytest.mark.parametrize("collapsed_gap", [0.0, -1.0, float("nan")])
    def test_rejects_collapsed_or_nonfinite_gap_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
        collapsed_gap: float,
    ) -> None:
        state = _state_on_physics_grid()
        monkeypatch.setattr(
            "qpsim.backends.t3_diffusion.solve_gap",
            lambda *_args, **_kwargs: collapsed_gap,
        )
        with pytest.raises(RuntimeError, match="collapsed or non-finite"):
            T3DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_dynes_gap_motion(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(0.8, 2.0, 81)
        state = _state_on_custom_grid(E, dynes_gamma=0.01)
        monkeypatch.setattr(
            "qpsim.backends.t3_diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.01,
        )
        with pytest.raises(ValueError, match="dynes_gamma == 0"):
            T3DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_falling_gap_without_lower_grid_room(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(1.01, 2.0, 81)
        state = _state_on_custom_grid(E)
        monkeypatch.setattr(
            "qpsim.backends.t3_diffusion.solve_gap",
            lambda *_args, **_kwargs: 0.9,
        )
        with pytest.raises(ValueError, match="sub-gap grid room"):
            T3DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_state_spectral_gap_mismatch(self) -> None:
        state = _state_on_physics_grid()
        state.gap *= 0.99
        with pytest.raises(ValueError, match="must match"):
            T3DiffusionBackend().apply_gap_update(state, dt=0.0)


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
