"""Tests for qpsim.services.steady_state."""

from __future__ import annotations

import numpy as np
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state


def _setup(T_bath: float = 0.3, T_c: float = 1.2, num: int = 30):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    return ctx, K_s0, K_r0, T_bath


class TestThermalPhononPath:
    def test_recovers_fermi_dirac(self) -> None:
        # At thermal phonons + bath temperature T_bath, the steady state
        # should be Fermi-Dirac at T_bath.
        ctx, K_s0, K_r0, T_bath = _setup()
        f = solve_steady_state(ctx, K_s0, K_r0, T_bath, tol=1e-12)
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        np.testing.assert_allclose(f, f_FD, atol=1e-8)

    def test_rejects_bad_initial_guess_shape(self) -> None:
        import pytest

        ctx, K_s0, K_r0, T_bath = _setup()
        with pytest.raises(ValueError, match="initial_guess shape"):
            solve_steady_state(
                ctx, K_s0, K_r0, T_bath, initial_guess=np.zeros(3),
            )


class TestFiniteTauLPath:
    def test_tau_l_zero_matches_thermal_at_low_rates(self) -> None:
        # With phonon_escape_time=0 (no substrate coupling) and weak
        # coupling, the Picard solve should still approximately recover
        # f_FD — the only way the QP distribution stays at equilibrium
        # is if the e-ph integral vanishes, which requires thermal f.
        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        phonon_out: dict = {}
        f = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            phonon_escape_time=0.0,
            tol=1e-10, max_iter=200,
            picard_tol=1e-8, max_picard_iter=100,
            phonon_out=phonon_out,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        # Detailed balance is exact only for thermal phonons; with
        # self-consistent n_ph there can be small drift, so we allow
        # a looser bound than the pure-thermal test above.
        np.testing.assert_allclose(f, f_FD, atol=1e-4)
        assert "n_ph" in phonon_out
        assert "omega_bins" in phonon_out

    def test_finite_tau_l_converges(self) -> None:
        # Just exercise the finite-tau_l path end-to-end. With a small
        # tau_l the phonon bath dominates and f should be near f_FD.
        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        f = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            phonon_escape_time=1e-3,  # 1 ps — very fast bath coupling
            tol=1e-10, max_iter=200,
            picard_tol=1e-8, max_picard_iter=200,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        np.testing.assert_allclose(f, f_FD, atol=1e-4)
