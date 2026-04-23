"""Tests for qpsim.solvers.coupled_newton."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state
from qpsim.solvers.coupled_newton import coupled_newton_solve


def _thermal_setup(T_bath: float = 0.3, T_c: float = 1.2, num: int = 18):
    """Modest-size thermal setup (keeps FD Jacobian cost manageable)."""
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
    return ctx, K_s0, K_r0, omega_bins, idx_diff, idx_sum, diff_sign, T_bath


class TestCoupledNewtonSolve:
    def test_thermal_equilibrium_is_fixed_point(self) -> None:
        # At thermal equilibrium, (f_FD, n_BE) is the joint fixed point
        # of the coupled residual; one Newton pass shouldn't move it.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        f_out, n_out = coupled_newton_solve(
            ctx, f_FD, n_BE,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.25,
            tol=1e-8,
        )
        np.testing.assert_allclose(f_out, f_FD, atol=1e-8)
        np.testing.assert_allclose(n_out, n_BE, atol=1e-8)

    def test_converges_from_perturbed_initial(self) -> None:
        # Perturb f smoothly; coupled Newton should converge back to thermal.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)
        f0 = np.clip(f_FD * (1.0 + 0.2 * np.cos(ctx.E / ctx.gap)), 0.0, 1.0)

        f_out, n_out = coupled_newton_solve(
            ctx, f0, n_BE,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.25,
            tol=1e-8,
        )
        np.testing.assert_allclose(f_out, f_FD, atol=1e-6)
        np.testing.assert_allclose(n_out, n_BE, atol=1e-6)

    def test_matches_picard_on_shared_case(self) -> None:
        # Where Picard converges (thermal case), coupled Newton should
        # land on the same (f, n_ph) within tolerance.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=15)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)

        # Picard via the service orchestrator.
        phonon_out: dict[str, np.ndarray] = {}
        f_picard = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            initial_guess=f_init,
            phonon_escape_time=0.5,
            tol=1e-12, picard_tol=1e-10,
            phonon_out=phonon_out,
        )
        n_picard = phonon_out["n_ph"]

        # Coupled Newton from the same initial.
        f_newton, n_newton = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.5,
            tol=1e-10,
        )

        np.testing.assert_allclose(f_newton, f_picard, atol=1e-6)
        np.testing.assert_allclose(n_newton, n_picard, atol=1e-6)

    def test_output_shapes(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=12)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        f_out, n_out = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.1,
            tol=1e-8,
        )
        assert f_out.shape == (ctx.E.size,)
        assert n_out.shape == omega.shape

    def test_rejects_mismatched_n_ph_init(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=10)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        with pytest.raises(ValueError, match="n_ph_init length"):
            coupled_newton_solve(
                ctx, f_init, np.zeros(omega.size + 5),  # wrong length
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.1,
            )

    def test_zero_tau_l_branch(self) -> None:
        # τ_l = 0: no substrate coupling. Residual R_ph reduces to
        # a_ph + b_ph · n_ph. The solver should still converge.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=12)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        f_out, n_out = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.0,
            tol=1e-8,
        )
        # Just check everything is finite and bounded.
        assert np.all(np.isfinite(f_out))
        assert np.all(np.isfinite(n_out))
        assert np.all(f_out >= 0.0)
        assert np.all(f_out <= 1.0)
        assert np.all(n_out >= 0.0)
