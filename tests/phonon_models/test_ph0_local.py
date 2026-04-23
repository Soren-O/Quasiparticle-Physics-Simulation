"""Tests for qpsim.phonon_models.ph0_local."""

from __future__ import annotations

import numpy as np
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.phonon_models.ph0_local import phonon_steady_state
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _setup(T_bath: float = 0.3, T_c: float = 1.2, num: int = 30):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
    kT = KB_UEV_PER_K * T_bath
    f_th = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
    return ctx, K_s0, K_r0, omega_bins, idx_diff, idx_sum, diff_sign, f_th, T_bath


class TestPhononSteadyState:
    def test_thermal_f_gives_thermal_n_ph_finite_tau_l(self) -> None:
        # At thermal equilibrium with finite τ_l, detailed balance
        # forces n_ph → n_BE(ω, T_bath).
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()
        n_ph = phonon_steady_state(
            f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_bath, tau_l=0.25,
        )
        n_th = thermal_phonon_occupation(omega, T_bath)
        # The detailed-balance identity: at thermal equilibrium, both
        # a_ph/(−b_ph) and n_th coincide, so the steady-state formula
        # pins n_ph to n_th independent of τ_l.
        np.testing.assert_allclose(n_ph, n_th, atol=1e-10)

    def test_output_shape(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()
        n_ph = phonon_steady_state(
            f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_bath, tau_l=0.1,
        )
        assert n_ph.shape == omega.shape

    def test_non_negative(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()
        n_ph = phonon_steady_state(
            f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_bath, tau_l=0.1,
        )
        assert np.all(n_ph >= 0.0)

    def test_zero_tau_l_uses_ratio_branch(self) -> None:
        # τ_l = 0 ⇒ n_ph = −a_ph / b_ph (no substrate bath). Exercise
        # the code path and verify finite output.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()
        n_ph = phonon_steady_state(
            f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
            T_bath=T_bath, tau_l=0.0,
        )
        assert np.all(np.isfinite(n_ph))
        assert np.all(n_ph >= 0.0)
