"""Tests for the use_thermal_phonons routing in DiffusionBackend.steady_state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.diffusion import DiffusionBackend, DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _build_state(T_bath: float = 0.3, num_energy: int = 20) -> DiffusionState:
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
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return DiffusionState(
        f=f_init, gap=gap, spectral=spectral, phonon=phonon,
        material=material, T_bath=T_bath,
    )


class TestThermalPhononPath:
    def test_thermal_initial_is_near_fixed_point(self) -> None:
        state = _build_state(T_bath=0.3)
        backend = DiffusionBackend()
        new_state = backend.steady_state(state, use_thermal_phonons=True)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-8)

    def test_returned_phonon_pinned_at_bose_einstein(self) -> None:
        state = _build_state(T_bath=0.3)
        backend = DiffusionBackend()
        new_state = backend.steady_state(state, use_thermal_phonons=True)

        # n_ph must be the thermal Bose-Einstein distribution on the
        # physics ω grid, independent of what the input state had.
        expected_omega, _, _, _ = build_phonon_frequency_map(state.spectral.E)
        expected_n_th = thermal_phonon_occupation(expected_omega, state.T_bath)
        np.testing.assert_allclose(
            new_state.phonon.n_ph[0, :, 0], expected_n_th, atol=1e-14,
        )
        np.testing.assert_allclose(
            new_state.phonon.omega_bins[0], expected_omega, atol=1e-14,
        )

    def test_matches_picard_at_small_tau_l(self) -> None:
        # As τ_l → 0, the Picard path converges to the thermal-phonon
        # limit. At a moderately small τ_l both paths should produce
        # nearly the same f.
        state_therm = _build_state(T_bath=0.3)
        backend = DiffusionBackend()
        s_therm = backend.steady_state(state_therm, use_thermal_phonons=True)

        state_pic = _build_state(T_bath=0.3)
        # Make τ_l very small so the Picard pinned n_ph ≈ n_BE.
        state_pic.phonon = PhononState(  # type: ignore[misc]
            n_ph=state_pic.phonon.n_ph,
            omega_bins=state_pic.phonon.omega_bins,
            tau_l=np.full_like(state_pic.phonon.tau_l, 1e-4),
            branches=state_pic.phonon.branches,
        )
        s_pic = backend.steady_state(state_pic, method="picard", picard_tol=1e-10)

        np.testing.assert_allclose(s_pic.f, s_therm.f, atol=1e-5)

    def test_conflict_with_coupled_newton_is_rejected(self) -> None:
        state = _build_state()
        with pytest.raises(ValueError, match="coupled_newton"):
            DiffusionBackend().steady_state(
                state, use_thermal_phonons=True, method="coupled_newton",
            )

    def test_picard_default_untouched(self) -> None:
        # Existing default behavior (method="picard", use_thermal_phonons=False)
        # is preserved — regression check.
        state = _build_state(T_bath=0.3)
        backend = DiffusionBackend()
        new_state = backend.steady_state(state)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-6)

    def test_threads_picard_atol_independently(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _build_state(T_bath=0.3, num_energy=12)
        original = backend_module.solve_steady_state
        observed: list[float] = []

        def capture_picard_atol(*args, **kwargs):
            observed.append(float(kwargs["picard_atol"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            backend_module, "solve_steady_state", capture_picard_atol,
        )
        DiffusionBackend().steady_state(
            state,
            method="picard",
            newton_tol=1e-9,
            picard_atol=3e-13,
        )

        assert observed == [3e-13]

    def test_threads_picard_balance_tol(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _build_state(T_bath=0.3, num_energy=12)
        original = backend_module.solve_steady_state
        observed: list[float] = []

        def capture_picard_balance_tol(*args, **kwargs):
            observed.append(float(kwargs["picard_balance_tol"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            backend_module, "solve_steady_state", capture_picard_balance_tol,
        )
        DiffusionBackend().steady_state(
            state,
            method="picard",
            picard_balance_tol=4e-7,
        )

        assert observed == [4e-7]

    @pytest.mark.parametrize("use_thermal_phonons", [False, True])
    def test_threads_newton_backward_error_tol(
        self,
        monkeypatch: pytest.MonkeyPatch,
        use_thermal_phonons: bool,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _build_state(T_bath=0.3, num_energy=12)
        original = backend_module.solve_steady_state
        observed: list[float] = []

        def capture_newton_balance(*args, **kwargs):
            observed.append(float(kwargs["newton_backward_error_tol"]))
            return original(*args, **kwargs)

        monkeypatch.setattr(
            backend_module,
            "solve_steady_state",
            capture_newton_balance,
        )
        DiffusionBackend().steady_state(
            state,
            method="picard",
            use_thermal_phonons=use_thermal_phonons,
            newton_backward_error_tol=3e-7,
        )

        assert observed == [3e-7]
