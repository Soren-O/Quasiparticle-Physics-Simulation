"""Tests for the coupled-Newton path through T3DiffusionBackend.steady_state."""

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


def _build_state(T_bath: float = 0.3, num_energy: int = 18) -> T3DiffusionState:
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), 0.25),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_init, gap=gap, spectral=spectral, phonon=phonon,
        material=material, T_bath=T_bath,
    )


class TestCoupledNewtonPath:
    def test_method_coupled_newton_thermal_fixed_point(self) -> None:
        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        new_state = backend.steady_state(
            state, method="coupled_newton",
            coupled_newton_tol=1e-8,
        )
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-6)

    def test_method_coupled_newton_matches_picard(self) -> None:
        # At a parameter set where Picard converges, both methods should
        # land on the same (f, n_ph).
        state = _build_state(T_bath=0.3)
        backend = T3DiffusionBackend()
        s_picard = backend.steady_state(state, method="picard", picard_tol=1e-10)
        s_newton = backend.steady_state(state, method="coupled_newton", coupled_newton_tol=1e-8)
        np.testing.assert_allclose(s_newton.f, s_picard.f, atol=1e-5)
        np.testing.assert_allclose(
            s_newton.phonon.n_ph[0, :, 0],
            s_picard.phonon.n_ph[0, :, 0],
            atol=1e-5,
        )

    def test_rejects_unknown_method(self) -> None:
        state = _build_state()
        with pytest.raises(ValueError, match="Unknown method"):
            T3DiffusionBackend().steady_state(state, method="what-is-this")

    def test_coupled_newton_with_off_grid_phonon_seeds_thermal(self) -> None:
        # If state.phonon.omega_bins doesn't match the physics grid,
        # coupled Newton falls back to a thermal initial n_ph rather
        # than erroring. That still produces a sensible converged state
        # (unlike apply_collisions, which requires the physics grid).
        state = _build_state()
        off = np.linspace(0.1, 5.0, 40)
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=np.zeros((1, off.size, 1)),
            omega_bins=off.reshape(1, -1),
            tau_l=np.full((1, off.size), 0.25),
            model=state.phonon.model,
            branches=state.phonon.branches,
        )
        new_state = T3DiffusionBackend().steady_state(
            state, method="coupled_newton", coupled_newton_tol=1e-6,
        )
        # Returned phonon is rebuilt on the physics grid regardless.
        expected_omega, _, _, _ = build_phonon_frequency_map(state.spectral.E)
        np.testing.assert_allclose(new_state.phonon.omega_bins[0], expected_omega)


class TestApplyGapUpdatePreservesSpectralConfig:
    def test_preserves_custom_diffusion_coefficient(self) -> None:
        state = _build_state(T_bath=0.01, num_energy=20)
        # Override the spectral with a non-default D₀.
        state.spectral = SpectralContext(  # type: ignore[misc]
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            diffusion_coefficient=7.0,  # deliberately ≠ material.D_0
        )
        new_state = T3DiffusionBackend().apply_gap_update(state, dt=0.1)
        assert new_state.spectral.diffusion_coefficient == pytest.approx(7.0)

    def test_preserves_custom_dynes_gamma(self) -> None:
        state = _build_state(T_bath=0.01, num_energy=20)
        state.spectral = SpectralContext(  # type: ignore[misc]
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            dynes_gamma=0.05,
        )
        new_state = T3DiffusionBackend().apply_gap_update(state, dt=0.1)
        assert new_state.spectral.dynes_gamma == pytest.approx(0.05)

    def test_preserves_custom_rebuild_tolerance_and_margin(self) -> None:
        state = _build_state(T_bath=0.01, num_energy=20)
        state.spectral = SpectralContext(  # type: ignore[misc]
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            rebuild_tolerance=1e-8,
            active_margin_factor=0.5,
        )
        new_state = T3DiffusionBackend().apply_gap_update(state, dt=0.1)
        assert new_state.spectral.rebuild_tolerance == pytest.approx(1e-8)
        assert new_state.spectral.active_margin_factor == pytest.approx(0.5)
