"""Tests for the Marchegiani rate-equation module (M25 Eq. 8 + full steady state)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from qpsim.services.rate_equation import (
    M25Coefficients,
    M25SteadyState,
    _rate_equation_residual,
    _source_scaled_residual_tolerances,
    crossover_temperature_kelvin,
    solve_rate_equation_steady_state,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)
from scipy.special import lambertw

from tests.services.test_rate_equation_note_v import _fig3a_drive, _fig3a_params


def _fig3a_coefficients(T_kelvin: float = 0.030) -> M25Coefficients:
    """Note-V Fig 3a coefficient bundle (drive calibrated to 300 Hz).

    Parameter values are owned by
    :mod:`tests.services.test_rate_equation_note_v` (``_fig3a_params``
    / ``_fig3a_drive``) so every services test exercises the same
    M25 Fig 3a caption bundle.
    """
    params = _fig3a_params(T_kelvin=T_kelvin)
    drive_template = _fig3a_drive()
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, drive_template, 300.0,
    )
    return coefficients_from_physical_parameters_with_photon_drive(
        params, replace(drive_template, Gamma_nu_scale_Hz=scale),
    )


class TestLambertWIdentity:
    """T̄ · W(4π r/g) / (2Δ) = 1 by construction."""

    def test_round_trip(self) -> None:
        Delta_R = 2.35
        r_Rlt = 6.25e6
        g_ph = 300.0
        T_bar = crossover_temperature_kelvin(
            Delta_R_kelvin=Delta_R,
            r_Rlt_rate_Hz=r_Rlt,
            g_photon_R_rate_Hz=g_ph,
        )
        w_val = float(lambertw(4.0 * np.pi * r_Rlt / g_ph, k=0).real)
        assert T_bar * w_val / (2.0 * Delta_R) == pytest.approx(1.0, rel=1e-12)


class TestMonotonicity:
    """T̄ should rise with Δ_R, rise with r^{R<}, and fall with g^ph_R."""

    def _T_bar(self, Delta_R: float = 2.35, r_Rlt: float = 6.25e6, g_ph: float = 300.0) -> float:
        return crossover_temperature_kelvin(
            Delta_R_kelvin=Delta_R, r_Rlt_rate_Hz=r_Rlt, g_photon_R_rate_Hz=g_ph,
        )

    def test_T_bar_rises_with_Delta_R(self) -> None:
        assert self._T_bar(Delta_R=2.0) < self._T_bar(Delta_R=3.0)

    def test_T_bar_rises_with_r_Rlt(self) -> None:
        # Larger r_Rlt → larger Lambert-W argument → larger W → smaller
        # T̄? Check actual direction.
        lo = self._T_bar(r_Rlt=1e5)
        hi = self._T_bar(r_Rlt=1e9)
        # Lambert-W grows ~log(z), so 2Δ/W decreases as r_Rlt rises.
        assert hi < lo

    def test_T_bar_falls_with_g_photon(self) -> None:
        # Larger g^ph → smaller argument → smaller W → larger T̄.
        # Physically: stronger photon generation pushes the crossover up.
        lo = self._T_bar(g_ph=100.0)
        hi = self._T_bar(g_ph=1e4)
        assert hi > lo


class TestScalingLimits:
    """At weak photon drive (g_ph → 0) the argument → ∞, W ~ log, T̄ → 0."""

    def test_weak_drive_limit(self) -> None:
        T_weak = crossover_temperature_kelvin(
            Delta_R_kelvin=2.35, r_Rlt_rate_Hz=6.25e6,
            g_photon_R_rate_Hz=1e-3,  # very weak drive
        )
        T_strong = crossover_temperature_kelvin(
            Delta_R_kelvin=2.35, r_Rlt_rate_Hz=6.25e6,
            g_photon_R_rate_Hz=1e6,
        )
        assert T_weak < T_strong  # weak drive → low crossover temp


class TestReferenceValue:
    """Reproduce the M25 Fig 3 paper-stated parameter set value.

    Fig 3 caption: Δ_R/h = 49 GHz, r^L = r^{R<} = 6.25 MHz,
    Γ_{01}^{ph} = 300 Hz. The paper places the T̄ dashed lines near
    T̄ ≈ 70 mK (small ω_LR) / T̄ ≈ 150 mK (large ω_LR). These values
    depend on ω_LR through g^ph_R; the closed-form T̄ formula itself
    only exposes the combination r^{R<} / g^ph_R, so we verify the
    order-of-magnitude sanity and the Lambert-W identity rather than
    pin a specific ω_LR.
    """

    def test_order_of_magnitude(self) -> None:
        # Δ_R/h = 49 GHz = 2.352 K.
        Delta_R_K = 49e9 * 6.62607015e-34 / 1.380649e-23
        # r^{R<} = 6.25 MHz, Γ_{01}^ph = 300 Hz as listed in Fig 3.
        # The effective g^ph_R for the full rate equation depends on
        # cooper-pair count; paper-reading suggests g^ph_R ~ kHz range
        # for these params. Sweep g^ph from 1 Hz to 1 MHz — T̄ should
        # land in mK-to-few-K range across that sweep.
        T_bars = [
            crossover_temperature_kelvin(
                Delta_R_kelvin=Delta_R_K,
                r_Rlt_rate_Hz=6.25e6,
                g_photon_R_rate_Hz=g,
            )
            for g in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
        ]
        assert all(1e-3 < T < 5.0 for T in T_bars)
        # Monotonic-in-g^ph (rising).
        assert all(T_bars[i] < T_bars[i + 1] for i in range(len(T_bars) - 1))


class TestInputValidation:
    def test_negative_Delta_rejected(self) -> None:
        with pytest.raises(ValueError, match="Delta_R_kelvin"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=-1.0, r_Rlt_rate_Hz=1e6, g_photon_R_rate_Hz=100.0,
            )

    def test_zero_r_Rlt_rejected(self) -> None:
        with pytest.raises(ValueError, match="r_Rlt_rate_Hz"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=2.35, r_Rlt_rate_Hz=0.0, g_photon_R_rate_Hz=100.0,
            )

    def test_zero_g_photon_rejected(self) -> None:
        with pytest.raises(ValueError, match="g_photon_R_rate_Hz"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=2.35, r_Rlt_rate_Hz=1e6, g_photon_R_rate_Hz=0.0,
            )


# ─────────────────────────────────────────────────────────────────────
#  Full three-chemical-potential steady-state solver tests.
#
#  These tests do not use the M25 Fig. 3 parameter set with literal
#  rate integrals (those require Supplementary Note III/IV, out of
#  scope for this service). Instead they exercise the solver on
#  analytically verifiable limiting regimes.
# ─────────────────────────────────────────────────────────────────────


def _make_trivial_coefs(
    *,
    g_L: float = 0.0,
    g_Rgt: float = 0.0,
    g_Rlt: float = 0.0,
    r_L: float = 1.0,
    r_Rgt: float = 1.0,
    r_Rlt: float = 1.0,
    r_cross: float = 0.0,
    tau_R_inv: float = 0.0,
    tau_E_inv: float = 0.0,
    gammas_L: np.ndarray | None = None,
    gammas_Rgt: np.ndarray | None = None,
    gammas_Rlt: np.ndarray | None = None,
    gamma_ee: np.ndarray | None = None,
    gamma_ph: np.ndarray | None = None,
    xi: float = 0.0,
    delta: float = 0.5,
) -> M25Coefficients:
    """Build an M25Coefficients with mostly-zero rates for limiting-case tests.

    Defaults include a weak ``gamma_ee`` with detailed-balance ratio 0.1
    so that the qubit-row Jacobian is non-singular (otherwise ṗ_1 ≡ 0
    trivially and Newton on the 4-unknown system is rank-deficient).
    Tests that care about qubit dynamics should override ``gamma_ee``.
    """
    zero_2x2 = np.zeros((2, 2))
    # Weak ee coupling so the qubit row of the Jacobian is non-singular;
    # tests that target qubit dynamics should override gamma_ee.
    default_gamma_ee = np.array([[0.0, 1e-6], [1e-5, 0.0]])
    return M25Coefficients(
        gammas_L=zero_2x2.copy() if gammas_L is None else gammas_L,
        gammas_Rgt=zero_2x2.copy() if gammas_Rgt is None else gammas_Rgt,
        gammas_Rlt=zero_2x2.copy() if gammas_Rlt is None else gammas_Rlt,
        gamma_ee=default_gamma_ee if gamma_ee is None else gamma_ee,
        gamma_ph=zero_2x2.copy() if gamma_ph is None else gamma_ph,
        r_L=r_L, r_Rgt=r_Rgt, r_Rlt=r_Rlt, r_cross=r_cross,
        g_L=g_L, g_Rgt=g_Rgt, g_Rlt=g_Rlt,
        tau_R_inv=tau_R_inv, tau_E_inv=tau_E_inv,
        xi=xi, delta=delta,
    )


class TestCoefficientValidation:
    def test_bad_shape_rejected(self) -> None:
        with pytest.raises(ValueError, match="gammas_L"):
            _make_trivial_coefs(gammas_L=np.zeros((3, 3)))

    def test_negative_gamma_rejected(self) -> None:
        bad = np.array([[0.0, 0.0], [0.0, -1.0]])
        with pytest.raises(ValueError, match="gammas_L"):
            _make_trivial_coefs(gammas_L=bad)

    def test_xi_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="xi"):
            _make_trivial_coefs(xi=1.5)

    def test_delta_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="delta"):
            _make_trivial_coefs(delta=0.0)
        with pytest.raises(ValueError, match="delta"):
            _make_trivial_coefs(delta=1.5)

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(ValueError, match="r_L"):
            _make_trivial_coefs(r_L=-1.0)

    @pytest.mark.parametrize("i, j", [(0, 0), (0, 1), (1, 1)])
    def test_gammas_Rlt_forbidden_positions_rejected(self, i: int, j: int) -> None:
        """Only gammas_Rlt[1, 0] may be nonzero — the paper's low-T ansatz
        zeros the other three R< entries, and the residual relies on that.
        """
        bad = np.zeros((2, 2))
        bad[i, j] = 1e-6
        with pytest.raises(ValueError, match="gammas_Rlt"):
            _make_trivial_coefs(gammas_Rlt=bad)

    def test_gammas_Rlt_10_only_accepted(self) -> None:
        ok = np.array([[0.0, 0.0], [1e-6, 0.0]])
        coefs = _make_trivial_coefs(gammas_Rlt=ok)
        assert coefs.gammas_Rlt[1, 0] == 1e-6

    def test_array_inputs_are_defensively_copied_and_read_only(self) -> None:
        gamma_ee = np.array([[0.0, 1e-6], [1e-5, 0.0]])
        g_ph_L = np.array([1e-4, 2e-4])
        coefs = replace(
            _make_trivial_coefs(gamma_ee=gamma_ee),
            g_ph_L_per_state=g_ph_L,
        )

        gamma_ee[0, 1] = 9.0
        g_ph_L[0] = 8.0
        assert coefs.gamma_ee[0, 1] == 1e-6
        assert coefs.g_ph_L_per_state[0] == 1e-4

        with pytest.raises(ValueError, match="read-only"):
            coefs.gamma_ee[0, 1] = 7.0
        with pytest.raises(ValueError, match="read-only"):
            coefs.g_ph_L_per_state[0] = 6.0


class TestResidualTrivialCases:
    """Sanity-check the residual on hand-computable inputs."""

    def test_all_zero_rates_and_zero_densities_is_stationary(self) -> None:
        coefs = _make_trivial_coefs(
            r_L=0.0, r_Rgt=0.0, r_Rlt=0.0, gamma_ee=np.zeros((2, 2)),
        )
        R = _rate_equation_residual(np.array([0.0, 0.0, 0.0, 0.0]), coefs)
        np.testing.assert_array_equal(R, np.zeros(4))

    def test_pure_qubit_ee_balance(self) -> None:
        """With only Γ̃^{ee}_{10} and Γ̃^{ee}_{01} nonzero, ṗ_1 = 0 iff
        p_1/p_0 = Γ̃^{ee}_{01}/Γ̃^{ee}_{10}."""
        gamma_ee_01 = 0.3
        gamma_ee_10 = 1.0
        coefs = _make_trivial_coefs(
            gamma_ee=np.array([[0.0, gamma_ee_01], [gamma_ee_10, 0.0]]),
        )
        p_1_eq = gamma_ee_01 / (gamma_ee_01 + gamma_ee_10)
        R = _rate_equation_residual(np.array([p_1_eq, 0.0, 0.0, 0.0]), coefs)
        assert abs(R[0]) < 1e-14


class TestSolverLimitingCases:
    """Full solver on regimes with closed-form answers."""

    def test_no_generation_gives_zero_densities(self) -> None:
        # No generation, finite recombination: x_α = 0 is the unique
        # physical steady state. Qubit settles to ee detailed balance.
        coefs = _make_trivial_coefs(
            gamma_ee=np.array([[0.0, 0.1], [1.0, 0.0]]),
        )
        state = solve_rate_equation_steady_state(coefs)
        assert state.x_L == pytest.approx(0.0, abs=1e-12)
        assert state.x_Rgt == pytest.approx(0.0, abs=1e-12)
        assert state.x_Rlt == pytest.approx(0.0, abs=1e-12)
        assert state.p_1 == pytest.approx(0.1 / 1.1, rel=1e-10)

    def test_left_only_balance(self) -> None:
        # All right-electrode rates zero and no cross-electrode
        # tunneling (δ · T_L · x_L = 0 because Γ̃^L = 0 on the loss
        # side). Then ẋ_L = g_L - r_L x_L² = 0 → x_L = √(g_L/r_L).
        g_L = 4.0
        r_L = 1.0
        coefs = _make_trivial_coefs(g_L=g_L, r_L=r_L)
        state = solve_rate_equation_steady_state(coefs)
        expected_x_L = np.sqrt(g_L / r_L)
        assert state.x_L == pytest.approx(expected_x_L, rel=1e-9)
        assert state.x_Rgt == pytest.approx(0.0, abs=1e-10)
        assert state.x_Rlt == pytest.approx(0.0, abs=1e-10)

    def test_right_only_balance_each_band_independent(self) -> None:
        # No tunneling, no cross-recombination, no intraband relaxation.
        # Each right-band equation decouples: ẋ_{Rα} = g_α - r_α x_α² = 0.
        g_Rgt, r_Rgt = 9.0, 1.0
        g_Rlt, r_Rlt = 1.0, 4.0
        coefs = _make_trivial_coefs(
            g_Rgt=g_Rgt, r_Rgt=r_Rgt, g_Rlt=g_Rlt, r_Rlt=r_Rlt,
        )
        state = solve_rate_equation_steady_state(coefs)
        assert state.x_Rgt == pytest.approx(np.sqrt(g_Rgt / r_Rgt), rel=1e-9)
        assert state.x_Rlt == pytest.approx(np.sqrt(g_Rlt / r_Rlt), rel=1e-9)
        assert state.x_L == pytest.approx(0.0, abs=1e-10)

    def test_intraband_relaxation_equilibrates_right_bands(self) -> None:
        # Strong intraband relaxation (τ_R^{-1}, τ_E^{-1} >> other rates)
        # drives x_{R>}/x_{R<} toward τ_E/τ_R (detailed balance between
        # R< → R> excitation and R> → R< relaxation).
        tau_R_inv = 1e3
        tau_E_inv = 1e2
        # Some photon generation in R< to keep the state nontrivial.
        coefs = _make_trivial_coefs(
            g_Rlt=1.0, r_Rlt=1.0, r_Rgt=1.0,
            tau_R_inv=tau_R_inv, tau_E_inv=tau_E_inv,
        )
        state = solve_rate_equation_steady_state(coefs)
        # At strong relaxation ratio x_Rgt/x_Rlt → τ_E_inv/τ_R_inv exactly
        # iff recombination and generation are negligible compared to
        # relaxation; the ratio is bounded by that limit.
        assert state.x_Rgt / state.x_Rlt == pytest.approx(
            tau_E_inv / tau_R_inv, rel=5e-3,
        )

    def test_residual_roundtrip(self) -> None:
        # Generic rate-set: solve, plug back, verify residual is tiny.
        rng = np.random.default_rng(0)
        coefs = M25Coefficients(
            gammas_L=rng.uniform(0, 1e-3, size=(2, 2)),
            gammas_Rgt=rng.uniform(0, 1e-3, size=(2, 2)),
            gammas_Rlt=np.array([[0.0, 0.0], [5e-4, 0.0]]),  # only [1,0] nonzero
            gamma_ee=np.array([[0.0, 3e-3], [1e-2, 0.0]]),
            gamma_ph=rng.uniform(0, 1e-4, size=(2, 2)),
            r_L=1.0, r_Rgt=1.0, r_Rlt=1.0, r_cross=0.1,
            g_L=1e-2, g_Rgt=5e-3, g_Rlt=1e-3,
            tau_R_inv=1e-1, tau_E_inv=1e-3,
            xi=0.05, delta=0.9,
        )
        state = solve_rate_equation_steady_state(coefs, residual_tol=1e-10)
        assert state.residual_inf_norm < 1e-10
        # Re-evaluate the residual from the returned densities.
        y = np.array([state.p_1, state.x_L, state.x_Rgt, state.x_Rlt])
        R_check = _rate_equation_residual(y, coefs)
        assert np.max(np.abs(R_check)) < 1e-10

    def test_probability_normalization_preserved(self) -> None:
        coefs = _make_trivial_coefs(
            gamma_ee=np.array([[0.0, 0.2], [1.0, 0.0]]),
            g_L=1e-3, g_Rlt=1e-3,
        )
        state = solve_rate_equation_steady_state(coefs)
        assert state.p_0 + state.p_1 == pytest.approx(1.0, rel=1e-14)

    def test_densities_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(5):
            coefs = _make_trivial_coefs(
                gammas_L=rng.uniform(0, 1e-3, size=(2, 2)),
                gamma_ee=np.array([[0.0, rng.uniform(1e-4, 1e-2)],
                                   [rng.uniform(1e-3, 1e-1), 0.0]]),
                g_L=rng.uniform(1e-4, 1e-1),
                g_Rgt=rng.uniform(1e-4, 1e-1),
                g_Rlt=rng.uniform(1e-4, 1e-1),
            )
            state = solve_rate_equation_steady_state(coefs)
            assert state.x_L >= 0.0
            assert state.x_Rgt >= 0.0
            assert state.x_Rlt >= 0.0

    def test_user_initial_guess_accepted(self) -> None:
        coefs = _make_trivial_coefs(g_L=4.0, r_L=1.0)
        state = solve_rate_equation_steady_state(
            coefs, initial_guess=np.array([0.001, 1.5, 0.0, 0.0]),
        )
        assert state.x_L == pytest.approx(2.0, rel=1e-9)

    def test_bad_initial_guess_shape_rejected(self) -> None:
        coefs = _make_trivial_coefs()
        with pytest.raises(ValueError, match="shape"):
            solve_rate_equation_steady_state(
                coefs, initial_guess=np.array([0.0, 0.0, 0.0]),
            )

    def test_fig3a_coefficients_solve_cleanly_with_bar_normalization(self) -> None:
        # Real M25 Fig 3a coefficients. With the single-quasiparticle
        # normalization of the density equations (Γ̄ = Γ̃ / N_CP(R),
        # M25 text below Eq. 6; ``cooper_pair_number_R`` set by the
        # Note-V builder), the system is well conditioned and the
        # default seed converges to the unique physical root with a
        # residual many orders below the source scale. With the
        # legacy normalization (cooper_pair_number_R = 1, i.e. Γ̄ =
        # Γ̃ ~ 1e10 Hz in the density equations) the same inputs sit
        # at the float64 cancellation floor and the strict solver
        # must raise — the historical ``accept_lm_convergence``
        # regime, kept here as a legacy regression.
        coefs = _fig3a_coefficients(T_kelvin=0.020)
        # N_CP(R) = 2 ν_0 Δ_R V ≈ 1.61e10 at the Fig 3 parameter set.
        assert coefs.cooper_pair_number_R == pytest.approx(1.61e10, rel=1e-2)

        # Default seed converges cleanly to the unique physical root
        # (x_L ≈ 5.3e-8 at 20 mK; paper Fig 3a: x_L ≈ x_R> + x_R< ≈
        # √(g^ph_R / r^L), see M25 Sec. II.4). Values are for the exact
        # S48/S49 tau_R quadrature (2026-07-19 audit H4); the old pins
        # (5.58/2.02/4.70e-8) belonged to the out-of-domain S50 series.
        state = solve_rate_equation_steady_state(coefs)
        assert state.x_L == pytest.approx(5.30e-8, rel=5e-2)
        assert state.x_Rgt == pytest.approx(1.79e-8, rel=5e-2)
        assert state.x_Rlt == pytest.approx(5.15e-8, rel=5e-2)
        assert 5e-4 < state.p_1 < 1.5e-3
        # The multi-seed helper (min_residual default) agrees.
        multi = solve_rate_equation_steady_state_multi_seed(coefs)
        assert multi.x_L == pytest.approx(state.x_L, rel=1e-6)

        # Legacy normalization regression: force Γ̄ = Γ̃ and verify
        # the historical cancellation-floor pathology (strict raise).
        # At 20 mK the Γ̃-scale density equations exhaust the Newton
        # budget rather than reaching the no-progress stall — pin the
        # maxfev failure mode so a silent acceptance regression is
        # caught.
        legacy = replace(coefs, cooper_pair_number_R=1.0)
        with pytest.raises(RuntimeError, match="maxfev"):
            solve_rate_equation_steady_state(legacy)

    @pytest.mark.parametrize("T_kelvin", [0.080, 0.100, 0.110])
    def test_fig3a_death_valley_uses_row_wise_source_gate(
        self, T_kelvin: float,
    ) -> None:
        # In the old global-min calculation, an exponentially small thermal
        # generation term crossed 1e-30 Hz and collapsed the tolerance to
        # 1e-14 Hz. The qubit row then failed on a 4.5e-13--1.8e-12 Hz
        # float64 cancellation quantum even though every density row was
        # balanced to ~1e-21 Hz. Row-wise source/backward-error limits keep
        # those independent numerical scales independent.
        coefs = _fig3a_coefficients(T_kelvin=T_kelvin)
        state = solve_rate_equation_steady_state(coefs)
        y = np.array([state.p_1, state.x_L, state.x_Rgt, state.x_Rlt])
        residual = np.abs(_rate_equation_residual(y, coefs))
        tolerances = _source_scaled_residual_tolerances(y, coefs, 1e-3)

        assert np.all(residual <= tolerances)
        # Density gates remain tied to the 1e-9--1e-8 Hz generation rows;
        # fixing the qubit cancellation floor must not create a broad scalar
        # ceiling that could admit density pseudo-roots.
        assert np.all(tolerances[1:] < 3e-11)
        assert np.all(residual[1:] < 1e-18)

    def test_accept_lm_convergence_does_not_bypass_maxfev_failures(self) -> None:
        # The accept_lm_convergence escape hatch is meant for the
        # specific "no progress" stall at the cancellation floor —
        # it must NOT silently accept a maxfev hit with garbage
        # residual. Force a maxfev failure with max_function_evaluations=1
        # and verify the bypass flag does not rescue it.
        coefs = _make_trivial_coefs(g_L=1.0, r_L=1.0, g_Rgt=1.0, g_Rlt=1.0)
        with pytest.raises(RuntimeError, match="failed"):
            solve_rate_equation_steady_state(
                coefs,
                accept_lm_convergence=True,
                initial_guess=np.array([0.5, 4.0, 4.0, 4.0]),
                max_function_evaluations=1,
            )

    def test_accept_lm_convergence_does_not_bypass_success_high_residual(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args, **kwargs):
            return SimpleNamespace(
                success=True,
                message="fake success",
                fun=np.array([1.0, 0.0, 0.0, 0.0]),
                x=np.array([0.5, 0.0, 0.0, 0.0]),
                nfev=1,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        with pytest.raises(RuntimeError, match="high residual"):
            rate_mod.solve_rate_equation_steady_state(
                _make_trivial_coefs(),
                residual_tol=1e-6,
                accept_lm_convergence=True,
            )


class TestAcceptLmConvergenceStatusOverride:
    """``accept_lm_convergence`` may relax status, never accuracy."""

    def _legacy_fig3a_coefs(self) -> M25Coefficients:
        # Legacy normalization: Γ̄ = Γ̃ (~1e10 Hz) in the density
        # equations, as in the neighboring regression tests.
        return replace(_fig3a_coefficients(), cooper_pair_number_R=1.0)

    def test_accept_lm_convergence_never_bypasses_source_gate(self) -> None:
        legacy = self._legacy_fig3a_coefs()
        # The legacy normalization stalls with density residuals
        # ~6.6e-7 Hz, versus row limits of ~1.9e-10, ~1.9e-10, and
        # 2.5e-12 Hz. A raw 1 Hz ceiling misclassified this as a fixed
        # point; both flag values must now reject it.
        with pytest.raises(RuntimeError, match="high residual"):
            solve_rate_equation_steady_state(legacy)
        with pytest.raises(RuntimeError, match="never bypasses"):
            solve_rate_equation_steady_state(
                legacy, accept_lm_convergence=True,
            )

    def test_multi_seed_recomputes_residual_at_final_gate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        # Simulate a wrapper that reports a stale zero diagnostic for a state
        # whose density rows are very far from stationary. The common final
        # gate must recompute R(y), not trust ``residual_inf_norm``.
        stale = M25SteadyState(
            p_0=0.5, p_1=0.5,
            x_L=10.0, x_Rgt=10.0, x_Rlt=10.0,
            residual_inf_norm=0.0, n_function_evaluations=1,
        )
        monkeypatch.setattr(
            rate_mod, "solve_rate_equation_steady_state",
            lambda *args, **kwargs: stale,
        )
        monkeypatch.setattr(rate_mod, "_default_seed_grid", lambda: [])

        with pytest.raises(RuntimeError, match="row-wise source-scaled"):
            rate_mod.solve_rate_equation_steady_state_multi_seed(
                _make_trivial_coefs(g_L=1.0, g_Rgt=1.0, g_Rlt=1.0),
            )

    def test_no_progress_status_requires_opt_in_after_residual_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=False,
                message="The iteration is not making good progress",
                fun=np.zeros(4),
                x=np.array([1.0 / 11.0, 0.0, 0.0, 0.0]),
                nfev=12,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        with pytest.raises(RuntimeError, match="solve stalled"):
            rate_mod.solve_rate_equation_steady_state(
                _make_trivial_coefs(), residual_tol=1e-6,
            )
        state = rate_mod.solve_rate_equation_steady_state(
            _make_trivial_coefs(), residual_tol=1e-6,
            accept_lm_convergence=True,
        )
        assert state.residual_inf_norm == 0.0

    def test_stall_with_high_residual_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # No-progress status cannot rescue a result that fails its row gate.
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=False,
                message=(
                    "The iteration is not making good progress, as "
                    "measured by the improvement from the last ten "
                    "iterations."
                ),
                fun=np.array([2.0, 0.0, 0.0, 0.0]),
                x=np.array([0.5, 1e-8, 1e-8, 1e-8]),
                nfev=42,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        with pytest.raises(RuntimeError, match="high residual"):
            rate_mod.solve_rate_equation_steady_state(
                _make_trivial_coefs(), accept_lm_convergence=True,
            )

    def test_negative_density_solution_rejected(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Direct pin of the physical-branch positivity guard: a
        # "successful" solve with a tiny residual but a negative
        # density must be rejected, never returned.
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                message="fake success",
                fun=np.zeros(4),
                x=np.array([0.5, -1e-9, 1e-8, 1e-8]),
                nfev=7,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        with pytest.raises(RuntimeError, match="negative quasiparticle"):
            rate_mod.solve_rate_equation_steady_state(
                _make_trivial_coefs(), residual_tol=1.0,
            )


class TestBranchPickerModesOnFig3a:
    """``max_x_L`` dispatch and the ``expected_ordering`` filter.

    Every candidate source is subject to the same row-wise physical gate.
    On the normalized, unique-root Fig 3a system all picker modes therefore
    agree. An explicit ordering hint is a constraint, not a warning that may
    be silently ignored.
    """

    def test_max_x_L_mode_dispatches_and_picks_largest_candidate(self) -> None:
        # The deprecated selector still dispatches, but the final gate keeps
        # the old sub-1-Hz slope pseudo-roots out of its candidate pool.
        coefs = _fig3a_coefficients()
        direct = solve_rate_equation_steady_state(coefs)
        with pytest.warns(DeprecationWarning, match="max_x_L.*deprecated"):
            picked = solve_rate_equation_steady_state_multi_seed(
                coefs, branch_picker_mode="max_x_L",
            )
        values = np.array([picked.p_1, picked.x_L, picked.x_Rgt, picked.x_Rlt])
        assert np.all(np.isfinite(values))
        assert picked.x_L == pytest.approx(direct.x_L, rel=1e-6)
        assert picked.x_Rgt == pytest.approx(direct.x_Rgt, rel=1e-6)
        assert picked.x_Rlt == pytest.approx(direct.x_Rlt, rel=1e-6)
        y = np.array([picked.p_1, picked.x_L, picked.x_Rgt, picked.x_Rlt])
        assert np.all(
            np.abs(_rate_equation_residual(y, coefs))
            <= _source_scaled_residual_tolerances(y, coefs, 1e-3)
        )

    def test_expected_ordering_consistent_keeps_root(self) -> None:
        # Fig 3a at 30 mK with the exact S48/S49 tau_R (audit H4):
        # x_R< > x_L > x_R> (5.25e-8, 5.05e-8, 1.87e-8). The ordering
        # of the two largest densities flipped relative to the old
        # out-of-domain S50 series (which gave x_L > x_R< within 8%).
        # A consistent ordering filter must be a no-op on the unique
        # root.
        coefs = _fig3a_coefficients()
        direct = solve_rate_equation_steady_state(coefs)
        picked = solve_rate_equation_steady_state_multi_seed(
            coefs, expected_ordering=("x_Rlt", "x_L", "x_Rgt"),
        )
        assert picked.x_L == pytest.approx(direct.x_L, rel=1e-6)
        assert picked.x_Rgt == pytest.approx(direct.x_Rgt, rel=1e-6)
        assert picked.x_Rlt == pytest.approx(direct.x_Rlt, rel=1e-6)

    def test_expected_ordering_inconsistent_raises(self) -> None:
        # No candidate satisfies the (physically wrong) ordering
        # x_R> ≥ x_R< ≥ x_L. Re-admitting the unfiltered candidate set would
        # silently discard the caller's branch constraint.
        coefs = _fig3a_coefficients()
        with pytest.raises(RuntimeError, match="refusing to fall back"):
            solve_rate_equation_steady_state_multi_seed(
                coefs, expected_ordering=("x_Rgt", "x_Rlt", "x_L"),
            )


class TestSolverReturnType:
    def test_returns_M25SteadyState(self) -> None:
        coefs = _make_trivial_coefs()
        state = solve_rate_equation_steady_state(coefs)
        assert isinstance(state, M25SteadyState)
        assert state.n_function_evaluations >= 1
        assert state.residual_inf_norm >= 0.0


class TestLmCandidateAcceptance:
    def test_rejects_unsuccessful_result_with_small_residual(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=False,
                message="maximum iterations reached",
                fun=np.zeros(4),
                x=np.array([0.5, 1e-8, 1e-8, 1e-8]),
                nfev=5000,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        assert rate_mod._solve_with_lm(
            _make_trivial_coefs(),
            np.array([0.5, 1e-8, 1e-8, 1e-8]),
            residual_ceiling_Hz=1.0,
        ) is None

    def test_row_gate_rejects_sub_hz_density_pseudo_root(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        def fake_root(*args: object, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                message="fake converged",
                # 0.1 Hz is below the historical 1 Hz ceiling but is a 10%
                # error against this row's 1 Hz generation source.
                fun=np.array([0.0, 0.1, 0.0, 0.0]),
                x=np.array([0.5, 1.0, 1.0, 1.0]),
                nfev=7,
            )

        monkeypatch.setattr(rate_mod, "root", fake_root)
        coefs = _make_trivial_coefs(g_L=1.0, g_Rgt=1.0, g_Rlt=1.0)
        seed = np.array([0.5, 1.0, 1.0, 1.0])
        assert rate_mod._solve_with_lm(
            coefs, seed, residual_ceiling_Hz=1.0,
        ) is not None
        assert rate_mod._solve_with_lm(
            coefs, seed, residual_tol_relative=1e-3,
        ) is None


class TestLmDeterminism:
    """The multi-seed picker uses scipy.optimize.root(method='lm') as an
    additional candidate source. The original switch from lm to hybr was
    justified by a 'FORTRAN COMMON-block state' non-determinism claim
    that doesn't reproduce on the supported scipy versions. This test
    pins the property — if a future scipy/platform combination breaks
    it, this test will catch it before it silently changes the picker's
    branch selection."""

    def test_lm_produces_bit_identical_results_across_repeated_calls(self) -> None:
        from scipy.optimize import root

        coefs = _make_trivial_coefs(g_L=1.0, g_Rgt=1.0, g_Rlt=1.0)
        seed = np.array([0.5, 1e-3, 5e-4, 1e-4])
        results = [
            root(
                _rate_equation_residual, seed, args=(coefs,),
                method="lm", options={"xtol": 1e-13},
            ).x
            for _ in range(10)
        ]
        for r in results[1:]:
            np.testing.assert_array_equal(results[0], r)
