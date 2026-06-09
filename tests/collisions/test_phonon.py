"""Tests for qpsim.collisions.phonon."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    CoherenceAssignment,
    apply_phonon_collision,
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext


def _thermal_setup(T_bath: float = 0.3, T_c: float = 1.2, tau_0: float = 1.0):
    """Small thermal-equilibrium setup used across the detailed-balance tests."""
    gap_init = 1.764 * KB_UEV_PER_K * T_c  # BCS Δ(0); close enough for T_bath = 0.3 K
    E, _ = build_energy_grid(
        gap=gap_init, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=40
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap_init)
    kT = KB_UEV_PER_K * T_bath
    f_thermal = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    K_s0 = build_scattering_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
    return ctx, f_thermal, K_s0, K_r0, T_bath


class TestDetailedBalance:
    """At thermal equilibrium, df/dt = gain − loss_rate · f should vanish."""

    def test_scattering_only(self) -> None:
        ctx, f, K_s0, _, T_bath = _thermal_setup()
        gain, loss_rate = phonon_collision_rates(
            f, ctx, K_s0, None, T_bath,
            enable_scattering=True, enable_recombination=False,
        )
        residual = gain - loss_rate * f
        scale = gain + loss_rate * f + 1e-30
        np.testing.assert_allclose(residual / scale, 0.0, atol=1e-10)

    def test_recombination_only(self) -> None:
        ctx, f, _, K_r0, T_bath = _thermal_setup()
        gain, loss_rate = phonon_collision_rates(
            f, ctx, None, K_r0, T_bath,
            enable_scattering=False, enable_recombination=True,
        )
        residual = gain - loss_rate * f
        scale = gain + loss_rate * f + 1e-30
        np.testing.assert_allclose(residual / scale, 0.0, atol=1e-10)

    def test_both_channels(self) -> None:
        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        gain, loss_rate = phonon_collision_rates(f, ctx, K_s0, K_r0, T_bath)
        residual = gain - loss_rate * f
        scale = gain + loss_rate * f + 1e-30
        np.testing.assert_allclose(residual / scale, 0.0, atol=1e-10)


class TestKernelBuilders:
    def test_phonon_vs_photon_swaps_coherence(self) -> None:
        ctx, _, _, _, _ = _thermal_setup()
        K_s_phonon = build_scattering_kernel_base(
            ctx, tau_0=1.0, T_c=1.2, coherence=CoherenceAssignment.PHONON
        )
        K_s_photon = build_scattering_kernel_base(
            ctx, tau_0=1.0, T_c=1.2, coherence=CoherenceAssignment.PHOTON
        )
        # Different coherence factors ⇒ different kernels overall.
        assert not np.allclose(K_s_phonon, K_s_photon)

        K_r_phonon = build_recombination_kernel_base(
            ctx, tau_0=1.0, T_c=1.2, coherence=CoherenceAssignment.PHONON
        )
        K_r_photon = build_recombination_kernel_base(
            ctx, tau_0=1.0, T_c=1.2, coherence=CoherenceAssignment.PHOTON
        )
        assert not np.allclose(K_r_phonon, K_r_photon)


class TestPhononSidePairBreaking:
    def test_scattering_kernel_uses_eq12_prefactor(self) -> None:
        gap = 180.0
        tau_0_pb = 0.255
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=1620
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        K_ph = build_scattering_kernel_phonon_side(ctx, tau_0_pb_ns=tau_0_pb)

        expected = (2.0 / (np.pi * gap * tau_0_pb)) * ctx.K_minus
        np.testing.assert_allclose(K_ph, expected)

    def test_source_sink_uses_phonon_side_scattering_override(self) -> None:
        gap = 180.0
        tau_0_pb = 0.255
        tau_0 = 438.0
        T_c = gap / (1.764 * KB_UEV_PER_K)
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=400
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(E)
        K_qp = build_scattering_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
        K_ph = build_scattering_kernel_phonon_side(ctx, tau_0_pb_ns=tau_0_pb)

        f = np.zeros(ctx.E.size)
        source_idx = 250
        target_idx = 70
        f[source_idx] = 1e-6
        omega_idx = idx_diff[source_idx, target_idx]

        a_legacy, _ = compute_phonon_source_sink(
            f, ctx, K_qp, None,
            idx_diff, idx_sum, diff_sign, omega.size,
            enable_recombination=False,
        )
        a_phonon_side, _ = compute_phonon_source_sink(
            f, ctx, K_qp, None,
            idx_diff, idx_sum, diff_sign, omega.size,
            enable_recombination=False,
            K_s0_phonon_side=K_ph,
        )

        common = (
            ctx.dE[target_idx]
            * ctx.rho[source_idx]
            * f[source_idx]
            * ctx.rho[target_idx]
        )
        np.testing.assert_allclose(
            a_legacy[omega_idx],
            common * K_qp[source_idx, target_idx],
        )
        np.testing.assert_allclose(
            a_phonon_side[omega_idx],
            common * K_ph[source_idx, target_idx],
        )
        assert a_phonon_side[omega_idx] > 5.0 * a_legacy[omega_idx]

    def test_first_pair_breaking_bin_matches_tau0pb(self) -> None:
        gap = 180.0
        tau_0_pb = 0.255
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=1620
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(E)
        K_ph = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=tau_0_pb)

        _, b_ph = compute_phonon_source_sink(
            np.zeros(ctx.E.size), ctx, None, None,
            idx_diff, idx_sum, diff_sign, omega.size,
            enable_scattering=False,
            enable_recombination=True,
            K_r0_phonon_side=K_ph,
        )

        mask = (omega > 2.0 * gap) & (b_ph < 0.0)
        first_idx = int(np.argmax(mask))
        assert 1.0 / -b_ph[first_idx] == pytest.approx(tau_0_pb, rel=2e-3)


class TestCollisionRates:
    def test_zero_f_gives_zero_gain(self) -> None:
        # With no quasiparticles, in-scattering is zero. The loss-rate
        # coefficient (rate at which a QP would scatter out if placed
        # here) is non-zero in general, but loss · f is trivially zero.
        ctx, _, K_s0, K_r0, _ = _thermal_setup()
        f_zero = np.zeros_like(ctx.E)
        gain, loss_rate = phonon_collision_rates(f_zero, ctx, K_s0, K_r0, T_bath=0.0)
        np.testing.assert_allclose(gain, 0.0)
        np.testing.assert_allclose(loss_rate * f_zero, 0.0)

    def test_output_shapes(self) -> None:
        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        gain, loss_rate = phonon_collision_rates(f, ctx, K_s0, K_r0, T_bath)
        assert gain.shape == ctx.E.shape
        assert loss_rate.shape == ctx.E.shape

    def test_override_matches_thermal(self) -> None:
        # Passing the thermal N_p/N_emit/N_abs as overrides must give
        # the same result as letting the function compute them internally.
        from qpsim.collisions.phonon import (
            _thermal_phonon_recombination_occupations,
            _thermal_phonon_scattering_occupation,
        )

        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        gain_ref, loss_ref = phonon_collision_rates(f, ctx, K_s0, K_r0, T_bath)

        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        gain_ov, loss_ov = phonon_collision_rates(
            f, ctx, K_s0, K_r0, T_bath,
            N_p_override=N_p, N_emit_override=N_emit, N_abs_override=N_abs,
        )
        np.testing.assert_allclose(gain_ov, gain_ref)
        np.testing.assert_allclose(loss_ov, loss_ref)


class TestFrequencyMap:
    def test_shapes_and_values(self) -> None:
        E = np.array([1.0, 2.0, 3.0])
        omega_bins, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(E)

        assert omega_idx_diff.shape == (3, 3)
        assert omega_idx_sum.shape == (3, 3)
        assert diff_sign.shape == (3, 3)
        assert diff_sign.dtype == np.int8

        # Diagonal |E_i - E_j| = 0 ⇒ all diagonal indices point to the 0 bin.
        assert omega_bins[omega_idx_diff[0, 0]] == 0.0
        # Max frequency is 2*E_max = 6.0
        assert omega_bins.max() == pytest.approx(6.0)

    def test_rejects_non_1d(self) -> None:
        with pytest.raises(ValueError, match="1D array"):
            build_phonon_frequency_map(np.array([[1.0, 2.0], [3.0, 4.0]]))


class TestOccupationMatricesFromState:
    def test_thermal_match(self) -> None:
        # Feed a Bose-Einstein n_ph into the projector; result should match
        # the thermal helpers.
        from qpsim.collisions.phonon import (
            _thermal_phonon_recombination_occupations,
            _thermal_phonon_scattering_occupation,
        )

        E = np.array([1.5, 2.5, 3.5])
        T_bath = 0.3
        omega_bins, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(E)

        kT = KB_UEV_PER_K * T_bath
        n_ph = np.zeros_like(omega_bins)
        pos = omega_bins > 0
        n_ph[pos] = 1.0 / (np.exp(np.minimum(omega_bins[pos] / kT, 500.0)) - 1.0)

        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph, omega_idx_diff, omega_idx_sum, diff_sign
        )

        N_p_thermal = _thermal_phonon_scattering_occupation(E, T_bath)
        N_emit_thermal, N_abs_thermal = _thermal_phonon_recombination_occupations(E, T_bath)

        np.testing.assert_allclose(N_p, N_p_thermal, atol=1e-10)
        np.testing.assert_allclose(N_emit, N_emit_thermal, atol=1e-10)
        np.testing.assert_allclose(N_abs, N_abs_thermal, atol=1e-10)


class TestApplyPhononCollision:
    def test_step_preserves_bounds(self) -> None:
        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        f_new = apply_phonon_collision(f, ctx, K_s0, K_r0, T_bath, dt=0.1)
        assert np.all(f_new >= 0.0)
        assert np.all(f_new <= 1.0)

    def test_thermal_is_near_fixed_point(self) -> None:
        # One ETD1 step on the thermal distribution should barely move it.
        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        f_new = apply_phonon_collision(f, ctx, K_s0, K_r0, T_bath, dt=0.01)
        np.testing.assert_allclose(f_new, f, atol=1e-8)
