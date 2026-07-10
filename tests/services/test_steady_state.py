"""Tests for qpsim.services.steady_state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import (
    _picard_convergence_ratio,
    solve_steady_state,
)


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

    def test_picard_atol_is_independent_of_newton_tol(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.services.steady_state as steady_state_module

        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        observed: list[tuple[float, float]] = []

        def capture_tolerances(
            n_ph: np.ndarray,
            n_ph_new: np.ndarray,
            *,
            rtol: float,
            atol: float,
        ) -> float:
            observed.append((rtol, atol))
            return 0.0

        monkeypatch.setattr(
            steady_state_module,
            "_picard_convergence_ratio",
            capture_tolerances,
        )
        solve_steady_state(
            ctx,
            K_s0,
            K_r0,
            T_bath,
            phonon_escape_time=1e-3,
            tol=1e-10,
            picard_tol=2e-8,
            picard_atol=3e-13,
        )

        assert observed == [(2e-8, 3e-13)]


class TestPicardConvergenceMetric:
    """Unit tests for the finite-τ_l Picard convergence metric.

    Regression coverage for the near-zero-bin stall fixed in f041a85
    (Fischer Fig. 7 at P_read=-64 dBm, T_B=0.10 K): a fully-settled solve whose
    only "unconverged" bin was a sub-gap occupation oscillating at the inner
    Newton's ~1e-11 float-noise floor.
    """

    def test_near_zero_bin_noise_does_not_pin_metric(self) -> None:
        # One dominant bin settled to its true relative tolerance, plus a
        # near-zero sub-gap bin carrying only ~1e-11 numerical noise on a
        # ~1e-6 occupation (|Δ|/n ~ 1e-5). The explicit occupation-space
        # allowance must prevent that jitter from pinning Picard.
        picard_tol = 1e-7  # Fischer Fig. 7's picard_tol
        picard_atol = 1e-11
        n_ph = np.array([8.0, 1e-6])
        n_ph_new = np.array([8.0 * (1.0 + 1e-9), 1e-6 + 1e-11])

        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=picard_tol, atol=picard_atol,
        ) <= 1.0

    def test_small_physical_bin_is_not_hidden_by_unrelated_peak(self) -> None:
        # Regression for Fischer Fig. 5: a large low-energy occupation must not
        # set a global floor that hides evolution in a tiny but dynamically
        # decisive above-gap pair-breaking bin.
        rtol = 1e-7
        atol = 1e-11
        n_ph = np.array([8.0, 1e-12])
        n_ph_new = np.array([8.0 * (1.0 + 1e-9), 5e-10])

        ratio = _picard_convergence_ratio(n_ph, n_ph_new, rtol=rtol, atol=atol)
        assert ratio > 1.0

        # The removed peak-scaled floor (1e-3 * peak) falsely accepted this.
        peak_floor_metric = float(
            np.max(np.abs(n_ph_new - n_ph) / (np.maximum(n_ph, n_ph_new) + 8e-3))
        )
        assert peak_floor_metric < rtol

    def test_meaningful_peak_change_is_not_converged(self) -> None:
        n_ph = np.array([8.0, 4.0])
        n_ph_new = np.array([8.0 * (1.0 + 1e-3), 4.0])
        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=1e-4, atol=1e-11,
        ) > 1.0

    def test_all_zero_is_trivially_converged(self) -> None:
        z = np.zeros(5)
        assert _picard_convergence_ratio(z, z, rtol=1e-7, atol=1e-11) == 0.0

    def test_identical_iterates_are_converged(self) -> None:
        n_ph = np.array([8.0, 1e-6, 0.0])
        assert _picard_convergence_ratio(
            n_ph, n_ph.copy(), rtol=1e-7, atol=1e-11,
        ) == 0.0

    @pytest.mark.parametrize("rtol,atol", [(0.0, 1e-11), (-1.0, 1e-11), (1e-7, -1.0)])
    def test_rejects_invalid_tolerances(self, rtol: float, atol: float) -> None:
        with pytest.raises(ValueError):
            _picard_convergence_ratio(
                np.ones(2), np.ones(2), rtol=rtol, atol=atol,
            )
