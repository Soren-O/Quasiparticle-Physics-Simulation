"""Tests for qpsim.collisions.phonon."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    CoherenceAssignment,
    _pair_breaking_quadrature_correction,
    apply_phonon_collision,
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
    phonon_source_sink_jacobian_f,
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
    def test_public_kernel_builders_reject_dynes_context(self) -> None:
        ctx, _, _, _, _ = _thermal_setup()
        dynes = SpectralContext(
            E_bins=ctx.E,
            dE_bins=ctx.dE,
            gap=ctx.gap,
            dynes_gamma=0.1,
        )

        builders = (
            lambda: build_scattering_kernel_base(dynes, tau_0=438.0, T_c=1.18),
            lambda: build_recombination_kernel_base(
                dynes, tau_0=438.0, T_c=1.18,
            ),
            lambda: build_scattering_kernel_phonon_side(
                dynes, tau_0_pb_ns=0.255,
            ),
            lambda: build_recombination_kernel_phonon_side(
                dynes, tau_0_pb_ns=0.255,
            ),
        )
        for build in builders:
            with pytest.raises(ValueError, match="dynes_gamma"):
                build()

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
            * ctx.cell_density[source_idx]
            * f[source_idx]
            * ctx.cell_density[target_idx]
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

    def test_kaplan_correction_does_not_override_k_minus(self) -> None:
        ctx, _, _, _, _ = _thermal_setup()
        omega, _, idx_sum, _ = build_phonon_frequency_map(ctx.E)
        correction = _pair_breaking_quadrature_correction(
            ctx, 7.0 * ctx.K_minus, idx_sum, omega.size
        )

        np.testing.assert_array_equal(correction, 1.0)

    def test_kaplan_correction_does_not_override_dynes_kernel(self) -> None:
        ctx, _, _, _, _ = _thermal_setup()
        dynes = SpectralContext(
            E_bins=ctx.E,
            dE_bins=ctx.dE,
            gap=ctx.gap,
            dynes_gamma=0.5,
        )
        omega, _, idx_sum, _ = build_phonon_frequency_map(dynes.E)
        correction = _pair_breaking_quadrature_correction(
            dynes, 3.0 * dynes.K_plus, idx_sum, omega.size
        )

        np.testing.assert_array_equal(correction, 1.0)

    def test_kaplan_correction_does_not_invent_off_grid_pair_states(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=400,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        omega, _, idx_sum, _ = build_phonon_frequency_map(E)
        K_ph = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=0.255)

        correction = _pair_breaking_quadrature_correction(
            ctx, K_ph, idx_sum, omega.size
        )
        upper_edge = ctx.E[-1] + 0.5 * ctx.dE[-1]
        truncated = omega > upper_edge + gap

        assert np.any(truncated)
        np.testing.assert_array_equal(correction[truncated], 1.0)
        # The complete near-threshold interval still receives the intended
        # analytic endpoint correction.
        assert np.any(correction[(omega > 2.0 * gap) & ~truncated] != 1.0)

    def test_kaplan_correction_requires_gap_edge_grid_coverage(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.01,
            energy_max_factor=10.0,
            num_energy_bins=400,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        omega, _, idx_sum, _ = build_phonon_frequency_map(E)
        K_ph = build_recombination_kernel_phonon_side(ctx, tau_0_pb_ns=0.255)

        correction = _pair_breaking_quadrature_correction(
            ctx, K_ph, idx_sum, omega.size
        )

        np.testing.assert_array_equal(correction, 1.0)


class TestCollisionRates:
    @pytest.mark.parametrize(
        "bad_f",
        [np.zeros(3), np.full(40, np.nan), np.full(40, -0.1), np.full(40, 1.1)],
    )
    def test_rejects_invalid_occupation(self, bad_f: np.ndarray) -> None:
        ctx, _, K_s0, K_r0, T_bath = _thermal_setup()
        with pytest.raises(ValueError, match=r"finite occupations|shape"):
            phonon_collision_rates(bad_f, ctx, K_s0, K_r0, T_bath)

    @pytest.mark.parametrize("bad_temperature", [-0.1, np.nan, np.inf])
    def test_rejects_invalid_bath_temperature(self, bad_temperature: float) -> None:
        ctx, f, K_s0, K_r0, _ = _thermal_setup()
        with pytest.raises(ValueError, match="T_bath"):
            phonon_collision_rates(f, ctx, K_s0, K_r0, bad_temperature)

    def test_rejects_malformed_kernel_and_override_contracts(self) -> None:
        ctx, f, K_s0, K_r0, T_bath = _thermal_setup()
        with pytest.raises(ValueError, match="K_s0 must have shape"):
            phonon_collision_rates(f, ctx, K_s0[:-1], K_r0, T_bath)
        with pytest.raises(ValueError, match="supplied together"):
            phonon_collision_rates(
                f,
                ctx,
                K_s0,
                K_r0,
                T_bath,
                N_emit_override=np.ones_like(K_r0),
            )
        with pytest.raises(ValueError, match="N_p_override must have shape"):
            phonon_collision_rates(
                f,
                ctx,
                K_s0,
                K_r0,
                T_bath,
                N_p_override=np.ones(ctx.E.size),
            )

    def test_cut_cell_qp_and_phonon_event_measures_match(self) -> None:
        # Cell 0 is cut by the gap and has rho(E_center)=0 but finite exact
        # capacity.  For a zero-temperature phonon state, QP scattering energy
        # loss must equal phonon energy creation under the matched line measure
        # dE*rho_bar_i*rho_bar_j = w_i*w_j/dE.
        E = np.arange(0.9, 3.0, 0.4)
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        f = np.zeros(E.size)
        f[3] = 0.2
        K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=1.2)
        omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(E)
        N_p, _N_emit, _N_abs = phonon_occupation_matrices_from_state(
            np.zeros(omega.size), idx_diff, idx_sum, sign,
        )

        gain, loss = phonon_collision_rates(
            f,
            ctx,
            K_s0,
            None,
            T_bath=0.0,
            enable_recombination=False,
            N_p_override=N_p,
        )
        a_ph, _b_ph = compute_phonon_source_sink(
            f,
            ctx,
            K_s0,
            None,
            idx_diff,
            idx_sum,
            sign,
            omega.size,
            enable_recombination=False,
        )
        qp_energy_rate = float(ctx.cell_weights @ (E * (gain - loss * f)))
        phonon_energy_rate = float(dE[0] * (omega @ a_ph))

        assert float(ctx.cell_weights @ (gain - loss * f)) == pytest.approx(
            0.0, abs=1e-20,
        )
        assert qp_energy_rate + phonon_energy_rate == pytest.approx(
            0.0, abs=1e-20,
        )

    def test_cut_cell_source_jacobian_matches_finite_difference(self) -> None:
        E = np.arange(0.9, 3.0, 0.4)
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        f = np.array([0.08, 0.12, 0.03, 0.20, 0.01, 0.15])
        omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(E)
        K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=1.2)
        K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=1.2)
        da_df, db_df = phonon_source_sink_jacobian_f(
            f, ctx, K_s0, K_r0, idx_diff, idx_sum, sign, omega.size,
        )

        h = 1e-7
        da_fd = np.empty_like(da_df)
        db_fd = np.empty_like(db_df)
        for j in range(f.size):
            up = f.copy()
            down = f.copy()
            up[j] += h
            down[j] -= h
            a_up, b_up = compute_phonon_source_sink(
                up, ctx, K_s0, K_r0, idx_diff, idx_sum, sign, omega.size,
            )
            a_down, b_down = compute_phonon_source_sink(
                down, ctx, K_s0, K_r0, idx_diff, idx_sum, sign, omega.size,
            )
            da_fd[:, j] = (a_up - a_down) / (2.0 * h)
            db_fd[:, j] = (b_up - b_down) / (2.0 * h)

        np.testing.assert_allclose(da_df, da_fd, rtol=2e-7, atol=2e-13)
        np.testing.assert_allclose(db_df, db_fd, rtol=2e-7, atol=2e-13)

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

    def test_thermal_scattering_handles_sub_epsilon_spacing(self) -> None:
        from qpsim.collisions.phonon import _thermal_phonon_scattering_occupation

        E = np.array([2.0, np.nextafter(2.0, np.inf)])
        occupation = _thermal_phonon_scattering_occupation(E, 1.0)
        assert np.all(np.isfinite(occupation))
        assert occupation[1, 0] > 1.0
        assert occupation[0, 1] > 0.0
        assert occupation[0, 0] == 0.0
        assert occupation[1, 1] == 0.0


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

    def test_non_binary_uniform_spacing_has_no_twin_frequency_bins(self) -> None:
        E, _ = build_energy_grid(
            gap=180.0,
            energy_min_factor=1.0,
            energy_max_factor=10.0,
            num_energy_bins=401,
        )
        omega, idx_diff, idx_sum, _ = build_phonon_frequency_map(E)

        # The former fixed-decimal deduplication produced 321 adjacent bins
        # separated by about 1e-12 for this otherwise ordinary uniform grid.
        assert float(np.min(np.diff(omega))) > 1e-9
        np.testing.assert_allclose(
            omega[idx_diff], np.abs(E[:, None] - E[None, :]), atol=1e-10, rtol=0.0
        )
        np.testing.assert_allclose(
            omega[idx_sum], E[:, None] + E[None, :], atol=1e-10, rtol=0.0
        )

    @pytest.mark.parametrize(
        "bad",
        [
            np.array([]),
            np.array([1.0, np.nan]),
            np.array([-2.0, -1.0]),
            np.array([2.0, 1.0]),
        ],
    )
    def test_rejects_invalid_energy_bins(self, bad: np.ndarray) -> None:
        with pytest.raises(ValueError):
            build_phonon_frequency_map(bad)


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

    def test_pure_bcs_subgap_storage_is_never_populated(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 0.75, 4.0, 80)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E, dE, gap)
        K_s0 = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.18)
        f = np.zeros(E.size)
        f[~ctx.active_mask] = 0.37
        f[ctx.active_mask] = 0.01 * np.exp(
            -((E[ctx.active_mask] / gap - 2.0) / 0.25) ** 2
        )

        gain, loss = phonon_collision_rates(
            f,
            ctx,
            K_s0,
            None,
            T_bath=0.1,
            enable_recombination=False,
        )
        np.testing.assert_array_equal(gain[~ctx.active_mask], 0.0)
        np.testing.assert_array_equal(loss[~ctx.active_mask], 0.0)

        f_new = apply_phonon_collision(
            f,
            ctx,
            K_s0,
            None,
            T_bath=0.1,
            dt=0.1,
            enable_recombination=False,
        )
        np.testing.assert_array_equal(f_new[~ctx.active_mask], 0.37)

    def test_stiff_scattering_step_preserves_finite_volume_mass(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 1.0, 20.0, 80)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E, dE, gap)
        K_s0 = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.18)
        f = np.full(E.size, 1.0e-3)
        weights = ctx.cell_weights
        initial_mass = float(weights @ f)

        f_new = apply_phonon_collision(
            f,
            ctx,
            K_s0,
            None,
            T_bath=0.1,
            dt=0.1,
            enable_recombination=False,
        )

        assert float(weights @ f_new) == pytest.approx(initial_mass, rel=2e-13)
        assert np.all((f_new >= 0.0) & (f_new <= 1.0))


class TestKaplanRecombinationNormalization:
    """Pin the absolute recombination normalization to Kaplan Eq. (8).

    The per-QP recombination loss is 1/τ_r(E) with NO pair factor: each
    event removes one QP at this energy and one in the partner's bin, so
    the density loses two per event without any explicit 2 in the
    occupation equation (paper eq:J1_occ_bridge: ∂_t f = I_occ with
    Kaplan-form kernels; F&C 2023 Eqs. 47/E2 carry the same
    normalization through R̄·n_th = 1/τ_r). Detailed balance and
    thermal-dominated steady states are blind to a symmetric doubling —
    these tests are the absolute-rate guard (audit 2026-06-10, which
    removed a legacy 2x "pair convention").
    """

    @staticmethod
    def _grid_setup(T_bath: float):
        tau_0, T_c = 438.0, 1.18
        gap = 1.764 * KB_UEV_PER_K * T_c
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.0005,
            energy_max_factor=12.0, num_energy_bins=2000,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        kT = KB_UEV_PER_K * T_bath
        f_th = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        return ctx, f_th, gap, tau_0, T_c, kT

    def test_loss_rate_is_kaplan_eq8_quadrature(self) -> None:
        # Same-grid raw-formula quadrature of Kaplan Eq. (8), built from
        # first principles (not the kernel builders): a factor-2
        # regression in either the rates or the builder prefactors
        # shows up as a ratio of 2.
        T_bath = 0.15
        ctx, f_th, _gap, tau_0, T_c, kT = self._grid_setup(T_bath)
        E = ctx.E
        K_r0 = build_recombination_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
        _, loss = phonon_collision_rates(
            f_th, ctx, None, K_r0, T_bath,
            enable_scattering=False, enable_recombination=True,
        )
        E_sum = E[:, None] + E[None, :]
        N_emit = 1.0 + 1.0 / (np.exp(np.minimum(E_sum / kT, 500.0)) - 1.0)
        kBTc = KB_UEV_PER_K * T_c
        ratio = np.divide(
            ctx.cell_anomalous_density,
            ctx.cell_density,
            out=np.zeros_like(E),
            where=ctx.cell_density > 0.0,
        )
        K_plus = 1.0 + ratio[:, None] * ratio[None, :]
        kaplan = (E_sum / kBTc) ** 2 / kBTc / tau_0 * K_plus * N_emit
        inv_tau_r = kaplan @ (ctx.cell_weights * f_th)
        np.testing.assert_allclose(loss, inv_tau_r, rtol=1e-12)

    def test_gain_is_pair_breaking_mirror(self) -> None:
        T_bath = 0.15
        ctx, f_th, _gap, tau_0, T_c, kT = self._grid_setup(T_bath)
        E = ctx.E
        K_r0 = build_recombination_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
        gain, _ = phonon_collision_rates(
            f_th, ctx, None, K_r0, T_bath,
            enable_scattering=False, enable_recombination=True,
        )
        E_sum = E[:, None] + E[None, :]
        n_BE = 1.0 / (np.exp(np.minimum(E_sum / kT, 500.0)) - 1.0)
        kBTc = KB_UEV_PER_K * T_c
        ratio = np.divide(
            ctx.cell_anomalous_density,
            ctx.cell_density,
            out=np.zeros_like(E),
            where=ctx.cell_density > 0.0,
        )
        K_plus = 1.0 + ratio[:, None] * ratio[None, :]
        kaplan = (E_sum / kBTc) ** 2 / kBTc / tau_0 * K_plus * n_BE
        expected = (1.0 - f_th) * (
            kaplan @ (ctx.cell_weights * (1.0 - f_th))
        )
        np.testing.assert_allclose(gain, expected, rtol=1e-12)

    def test_edge_rate_matches_continuum_kaplan(self) -> None:
        # Continuum Kaplan Eq. (8) at the gap edge via the ξ-substitution
        # (dE' ρ(E') = dξ, which removes the DOS singularity exactly),
        # compared with the grid loss at the lowest bin. A factor-2
        # regression gives ratio 2 — far outside the quadrature-error
        # tolerance.
        from scipy.integrate import quad

        T_bath = 0.15
        ctx, f_th, gap, tau_0, T_c, kT = self._grid_setup(T_bath)
        kBTc = KB_UEV_PER_K * T_c
        E0 = float(ctx.E[0])

        def integrand(xi: float) -> float:
            Ep = np.sqrt(xi * xi + gap * gap)
            coh = 1.0 + gap**2 / (E0 * Ep)
            n_emit = 1.0 + 1.0 / np.expm1(min((E0 + Ep) / kT, 500.0))
            f_p = 1.0 / (np.exp(min(Ep / kT, 500.0)) + 1.0)
            return ((E0 + Ep) / kBTc) ** 2 / kBTc / tau_0 * coh * n_emit * f_p

        xi_max = float(np.sqrt(ctx.E[-1] ** 2 - gap**2))
        inv_tau_r_exact, _ = quad(integrand, 0.0, xi_max, limit=200)

        K_r0 = build_recombination_kernel_base(ctx, tau_0=tau_0, T_c=T_c)
        _, loss = phonon_collision_rates(
            f_th, ctx, None, K_r0, T_bath,
            enable_scattering=False, enable_recombination=True,
        )
        ratio = float(loss[0]) / inv_tau_r_exact
        # Capacity and coherence are integrated exactly over the singular
        # cells.  The remaining ~9% error on this deliberately broad grid is
        # the ordinary mass-lumped quadrature of the smooth energy/frequency
        # prefactor at cell centers (plus the finite upper-energy truncation).
        # Keep a bounded continuum comparison in addition to the exact
        # discrete-form tests above.
        assert 0.85 < ratio < 1.15


class TestGapCutCellPairLabeling:
    """2026-07-20 review adjudication: supported gap-cut cells recombine
    through pairs whose CENTER-sum can label the emitted phonon below 2Δ
    although the capacity-supported pair energy is >= 2Δ. This is a
    DOCUMENTED labeling approximation (bounded by one dE, vanishing on
    covered grids), deliberately NOT masked: zeroing those pairs removed
    physical rate and shifted Fig. 6's derived tau_0^PB by ~21%. These
    tests pin the adjudicated semantics: the pair rate stays live, and
    emission/absorption share the bin (detailed balance exact)."""

    def _cut_ctx(self):
        # gap INSIDE the [0.90, 1.00] cell: its center 0.95 sits below the
        # gap while the [0.97, 1.00] sliver gives it finite capacity — a
        # genuine supported gap-cut cell (a face-aligned gap has none).
        gap = 0.97
        E = np.linspace(0.85, 3.05, 23)  # dE=0.1
        dE = integration_widths_from_centers(E)
        return SpectralContext(E_bins=E, dE_bins=dE, gap=gap)

    def test_supported_cut_cell_pairs_keep_finite_rate(self) -> None:
        ctx = self._cut_ctx()
        K_r0 = build_recombination_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        supported = ctx.cell_weights > 0.0
        pair_sum = ctx.E[:, None] + ctx.E[None, :]
        sub = pair_sum < 2.0 * ctx.gap - 1e-12
        cut_pairs = sub & supported[:, None] & supported[None, :]
        assert cut_pairs.any()  # the labeling case genuinely exists here
        assert float(np.max(K_r0[cut_pairs])) > 0.0  # not masked

    def test_emission_and_absorption_share_the_sub_label_bin(self) -> None:
        # Detailed balance at the discrete level: the thermal fixed point of
        # dn/dt = a + b n at every omega bin (including sub-2Delta-labeled
        # ones) is the Bose-Einstein occupation, because both directions use
        # the same kernel entry.
        ctx = self._cut_ctx()
        K_r0 = build_recombination_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
        T = 0.2
        kT = KB_UEV_PER_K * T
        f_th = np.where(
            ctx.cell_weights > 0.0,
            1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0),
            0.0,
        )
        a_ph, b_ph = compute_phonon_source_sink(
            f_th, ctx, None, K_r0, idx_diff, idx_sum, diff_sign, omega.size,
            enable_scattering=False,
        )
        from qpsim.physics.kernels import thermal_phonon_occupation

        n_th = thermal_phonon_occupation(omega, T)
        act = (-b_ph) > 0
        fixed_point = a_ph[act] / (-b_ph[act])
        np.testing.assert_allclose(fixed_point, n_th[act], rtol=1e-10, atol=1e-30)
