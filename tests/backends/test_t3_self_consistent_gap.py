"""Tests for the self-consistent-gap path in T3DiffusionBackend.steady_state."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics import acoustic_escape_tau_l, calibrate_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _build_state(
    *,
    T_bath: float = 0.3,
    num_energy: int = 40,
    tau_l_mode: str = "constant",
) -> T3DiffusionState:
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega, _, _, _ = build_phonon_frequency_map(E)
    omega_2d = omega.reshape(1, -1)
    if tau_l_mode == "constant":
        tau_l = np.full((1, omega.size), 0.25)
    elif tau_l_mode == "acoustic":
        material = replace(material, film_thickness=63.0, substrate_transmission_eta=0.2)
        tau_l = acoustic_escape_tau_l(omega_2d, material)
    else:
        raise ValueError(f"Unknown tau_l_mode {tau_l_mode!r}.")

    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega_2d,
        tau_l=tau_l,
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


class TestSelfConsistentGapPath:
    def test_thermal_fixed_point_matches_equilibrium_gap(self) -> None:
        state = _build_state(T_bath=0.3, num_energy=80)
        calibration = calibrate_gap(T_c=state.material.T_c, T_bath=state.T_bath)

        new_state = T3DiffusionBackend().steady_state(
            state,
            use_thermal_phonons=True,
            self_consistent_gap=True,
            gap_tol=1e-6,
        )

        assert new_state.gap == pytest.approx(calibration.delta_eq, rel=3e-3)
        assert new_state.spectral.gap == pytest.approx(new_state.gap, rel=1e-12)

    def test_preserves_spectral_configuration(self) -> None:
        state = _build_state(T_bath=0.25, num_energy=60)
        state.spectral = SpectralContext(
            E_bins=state.spectral.E,
            dE_bins=state.spectral.dE,
            gap=state.gap,
            diffusion_coefficient=7.0,
            rebuild_tolerance=1e-8,
            active_margin_factor=0.5,
        )

        new_state = T3DiffusionBackend().steady_state(
            state,
            use_thermal_phonons=True,
            self_consistent_gap=True,
            gap_tol=1e-6,
        )

        assert new_state.spectral.diffusion_coefficient == pytest.approx(7.0)
        assert new_state.spectral.rebuild_tolerance == pytest.approx(1e-8)
        assert new_state.spectral.active_margin_factor == pytest.approx(0.5)

    def test_acoustic_escape_picard_path_runs(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=30, tau_l_mode="acoustic")
        dE = float(state.spectral.dE[0])
        photon_params = {
            "omega_0": 2.0 * dE,
            "n_bar": 1.0e6,
            "c_phot": 1.0e-9,
        }

        new_state = T3DiffusionBackend().steady_state(
            state,
            method="picard",
            photon_params=photon_params,
            self_consistent_gap=True,
            gap_tol=1e-6,
            gap_max_iter=8,
            picard_tol=1e-8,
            picard_max_iter=500,
            picard_mixing=0.3,
        )

        assert np.isfinite(new_state.gap)
        assert new_state.gap > 0.0
        assert new_state.spectral.gap == pytest.approx(new_state.gap, rel=1e-12)
        assert new_state.phonon.n_ph.shape[1] == new_state.phonon.omega_bins.shape[1]

    def test_rejects_bath_above_tc(self) -> None:
        state = _build_state(T_bath=1.3, num_energy=30)
        with pytest.raises(ValueError, match="T_bath < T_c"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
            )

    def test_gap_collapse_raises_instead_of_drifting(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Force solve_gap to report collapse. Without the explicit
        # delta_raw <= 0 guard, under-relaxation would drift toward a
        # spurious tiny Δ > 0 instead of raising.
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=40)
        monkeypatch.setattr(t3_mod, "solve_gap", lambda *args, **kwargs: 0.0)
        with pytest.raises(RuntimeError, match="gap collapsed"):
            T3DiffusionBackend().steady_state(
                state,
                use_thermal_phonons=True,
                self_consistent_gap=True,
                gap_tol=1e-6,
                gap_max_iter=4,
            )

    def test_gap_under_relaxation_does_not_scale_convergence_residual(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.backends import t3_diffusion as t3_mod

        state = _build_state(T_bath=0.3, num_energy=12)
        backend = T3DiffusionBackend()
        solved_gaps: list[float] = []

        def fixed_gap_identity(
            state_arg: T3DiffusionState, **kwargs: object,
        ) -> T3DiffusionState:
            solved_gaps.append(state_arg.gap)
            return state_arg

        monkeypatch.setattr(backend, "_steady_state_fixed_gap", fixed_gap_identity)
        monkeypatch.setattr(
            t3_mod,
            "solve_gap",
            lambda *args, **kwargs: 2.0 * solved_gaps[-1],
        )

        # The raw gap-map residual is 1.0. A 1e-6 relaxed update is smaller
        # than gap_tol, but must not be reported as a converged gap equation.
        with pytest.raises(RuntimeError, match=r"Final \|Δ_raw - Δ\| / Δ = 1.00e\+00"):
            backend.steady_state(
                state,
                self_consistent_gap=True,
                use_thermal_phonons=True,
                gap_under_relaxation=1e-6,
                gap_tol=1e-4,
                gap_max_iter=1,
            )
