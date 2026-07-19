"""Tests for qpsim.phonon_models.ph0_local."""

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
from qpsim.phonon_models.ph0_local import (
    phonon_balance_diagnostics,
    phonon_steady_state,
)
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

    def test_zero_tau_l_ratio_is_invariant_below_old_absolute_floor(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Uniformly tiny regular balances must not be classified as singular."""
        from qpsim.phonon_models import ph0_local as ph0_mod

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()

        def fake_source_sink(*args, **kwargs):
            n_omega = len(omega)
            return np.full(n_omega, 1e-40), np.full(n_omega, -1e-40)

        monkeypatch.setattr(ph0_mod, "compute_phonon_source_sink", fake_source_sink)
        n_ph = ph0_mod.phonon_steady_state(
            f_th,
            ctx,
            K_s0,
            K_r0,
            omega,
            idx_d,
            idx_s,
            sgn,
            T_bath=T_bath,
            tau_l=0.0,
        )

        np.testing.assert_array_equal(n_ph, np.ones_like(omega))

    def test_zero_tau_l_rejects_negative_fixed_point(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.phonon_models import ph0_local as ph0_mod

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()

        def fake_source_sink(*args, **kwargs):
            n_omega = len(omega)
            return np.ones(n_omega), np.ones(n_omega)

        monkeypatch.setattr(ph0_mod, "compute_phonon_source_sink", fake_source_sink)
        with pytest.raises(RuntimeError, match="unphysical"):
            ph0_mod.phonon_steady_state(
                f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
                T_bath=T_bath, tau_l=0.0,
            )

    def test_zero_tau_l_rejects_tiny_negative_fixed_point(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An absolute clipping tolerance must not turn a negative root into 0."""
        from qpsim.phonon_models import ph0_local as ph0_mod

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()

        def fake_source_sink(*args, **kwargs):
            n_omega = len(omega)
            return np.full(n_omega, 1e-13), np.ones(n_omega)

        monkeypatch.setattr(ph0_mod, "compute_phonon_source_sink", fake_source_sink)
        with pytest.raises(RuntimeError, match="unphysical"):
            ph0_mod.phonon_steady_state(
                f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
                T_bath=T_bath, tau_l=0.0,
            )

    def test_zero_tau_l_rejects_negative_root_that_underflows_to_zero(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.phonon_models import ph0_local as ph0_mod

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()

        def fake_source_sink(*args, **kwargs):
            n_omega = len(omega)
            return (
                np.full(n_omega, np.nextafter(0.0, 1.0)),
                np.full(n_omega, np.finfo(float).max),
            )

        monkeypatch.setattr(ph0_mod, "compute_phonon_source_sink", fake_source_sink)
        with pytest.raises(RuntimeError, match="unphysical"):
            ph0_mod.phonon_steady_state(
                f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
                T_bath=T_bath, tau_l=0.0,
            )

    def test_finite_tau_l_rejects_runaway_denominator(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.phonon_models import ph0_local as ph0_mod

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, f_th, T_bath = _setup()

        def fake_source_sink(*args, **kwargs):
            n_omega = len(omega)
            return np.ones(n_omega), np.full(n_omega, 2.0)

        monkeypatch.setattr(ph0_mod, "compute_phonon_source_sink", fake_source_sink)
        with pytest.raises(RuntimeError, match="phonon runaway"):
            ph0_mod.phonon_steady_state(
                f_th, ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn,
                T_bath=T_bath, tau_l=1.0,
            )


class TestPhononBalanceDiagnostics:
    def test_nearest_representable_bath_root_is_certified(self) -> None:
        n_th = np.array([0.35])
        n_ph = n_th.copy()
        a_ph = np.array([5e-17])
        b_ph = np.array([-5e-17])

        diagnostics = phonon_balance_diagnostics(
            n_ph,
            a_ph,
            b_ph,
            n_th,
            tau_l=0.17,
        )

        assert diagnostics.raw_backward_error > 0.1
        assert diagnostics.certified_backward_error == 0.0

    def test_two_ulp_perturbation_is_not_hidden(self) -> None:
        n_th = np.array([0.35])
        one_ulp = np.nextafter(n_th, np.inf)
        two_ulp = np.nextafter(one_ulp, np.inf)
        diagnostics = phonon_balance_diagnostics(
            two_ulp,
            np.array([5e-17]),
            np.array([-5e-17]),
            n_th,
            tau_l=0.17,
        )

        assert diagnostics.certified_backward_error > 1e-2

    def test_certified_error_is_rate_scale_invariant(self) -> None:
        n_th = np.array([0.35])
        n_ph = np.nextafter(np.nextafter(n_th, np.inf), np.inf)
        a_ph = np.array([5e-17])
        b_ph = np.array([-5e-17])
        reference = phonon_balance_diagnostics(
            n_ph,
            a_ph,
            b_ph,
            n_th,
            tau_l=0.17,
        )
        # A power-of-two rescaling changes only the rate exponent, not the
        # binary significands or their operation-level rounding pattern.
        scale = 2.0**-64
        scaled = phonon_balance_diagnostics(
            n_ph,
            scale * a_ph,
            scale * b_ph,
            n_th,
            tau_l=0.17 / scale,
        )

        assert scaled.raw_backward_error == pytest.approx(
            reference.raw_backward_error,
        )
        assert scaled.certified_backward_error == pytest.approx(
            reference.certified_backward_error,
        )

    def test_negative_root_gets_no_representability_allowance(self) -> None:
        diagnostics = phonon_balance_diagnostics(
            np.array([0.0]),
            np.array([1e-13]),
            np.array([1.0]),
            np.array([0.0]),
            tau_l=0.0,
        )

        assert diagnostics.representability_allowance[0] == 0.0
        assert diagnostics.certified_backward_error == pytest.approx(1.0)

    def test_negative_root_underflowing_to_signed_zero_gets_no_allowance(
        self,
    ) -> None:
        diagnostics = phonon_balance_diagnostics(
            np.array([0.0]),
            np.array([np.nextafter(0.0, 1.0)]),
            np.array([np.finfo(float).max]),
            np.array([0.0]),
            tau_l=0.0,
        )

        assert diagnostics.representability_allowance[0] == 0.0
        assert diagnostics.certified_backward_error == pytest.approx(1.0)

    def test_unrepresentable_inverse_escape_rate_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="1/tau_l"):
            phonon_balance_diagnostics(
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                np.array([0.0]),
                tau_l=1e-320,
            )

    def test_nonrepresentable_affine_root_cannot_receive_infinite_allowance(
        self,
    ) -> None:
        maximum = np.finfo(float).max
        with pytest.raises(ValueError, match="physical scale"):
            phonon_balance_diagnostics(
                np.array([maximum]),
                np.array([maximum]),
                np.array([-0.5]),
                np.array([0.0]),
                tau_l=0.0,
            )

        diagnostics = phonon_balance_diagnostics(
            np.array([0.0]),
            np.array([-maximum]),
            np.array([0.5]),
            np.array([0.0]),
            tau_l=0.0,
        )
        assert diagnostics.representability_allowance[0] == 0.0
        assert diagnostics.certified_backward_error == pytest.approx(1.0)
