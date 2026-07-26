"""Tests for M25 SI Notes III + IV coefficient integrals."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest
from qpsim.services.rate_equation import (
    solve_rate_equation_steady_state,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    M25PhysicalParameters,
    _branching_fraction,
    _c_ii_squared,
    _gamma_L_01,
    _gamma_L_10,
    _gamma_L_ii,
    _gamma_Rlt_10,
    _K_incomplete,
    _log_K_incomplete,
    _s_10_squared,
    _tau_E_inverse,
    _tau_R_inverse,
    coefficients_from_physical_parameters,
)
from scipy.special import erf, erfc, k0, k1

_H_OVER_KB = 4.799243e-11  # K / Hz


def _fig3a_params(**overrides: float) -> M25PhysicalParameters:
    """M25 Fig 3a (small-gap-asymmetry) parameter set."""
    base = {
        "Delta_L_kelvin": 49.5e9 * _H_OVER_KB,
        "Delta_R_kelvin": 49.0e9 * _H_OVER_KB,
        "omega_10_kelvin": 5.5e9 * _H_OVER_KB,
        "T_kelvin": 0.030,
        "E_J_kelvin": 14.5e9 * _H_OVER_KB,
        "E_C_kelvin": 290e6 * _H_OVER_KB,
        # AB relation: R_T ≃ 8 (E_J/h) × (Δ̄/Δ_L) in Hz
        "R_T_Hz": 8.0 * 14.5e9 * (49.25 / 49.5),
        "r_L_Hz": 6.25e6,
        "r_Rlt_Hz": 6.25e6,
        "g_ph_L_Hz": 1.0,
        "g_ph_Rlt_Hz": 100.0,
        "g_ph_Rgt_Hz": 0.1,
        "Gamma_ph_00_Hz": 300.0,
        "Gamma_ph_11_Hz": 300.0,
        "Gamma_ee_10_Hz": 100e3,
    }
    base.update(overrides)
    return M25PhysicalParameters(**base)


# ═══════════════════════════════════════════════════════════════════════
#  Input validation
# ═══════════════════════════════════════════════════════════════════════


class TestInputValidation:
    def test_rejects_inverted_gaps(self) -> None:
        with pytest.raises(ValueError, match="Delta_L_kelvin > Delta_R_kelvin"):
            _fig3a_params(Delta_L_kelvin=2.0, Delta_R_kelvin=2.5)

    def test_rejects_equal_gaps(self) -> None:
        # Strict inequality: the SI formulas are singular at ω_LR = 0
        # (Bessel K_1(y) → ∞, erf(0) = 0).
        with pytest.raises(ValueError, match="Delta_L_kelvin > Delta_R_kelvin"):
            _fig3a_params(Delta_L_kelvin=2.352, Delta_R_kelvin=2.352)

    def test_rejects_zero_gap(self) -> None:
        with pytest.raises(ValueError, match="Delta_L_kelvin > Delta_R_kelvin"):
            _fig3a_params(Delta_L_kelvin=0.0)

    def test_rejects_nonpositive_T(self) -> None:
        with pytest.raises(ValueError, match="T_kelvin must be positive"):
            _fig3a_params(T_kelvin=0.0)

    def test_rejects_wrong_transmon_regime(self) -> None:
        with pytest.raises(ValueError, match="E_J_kelvin > E_C_kelvin"):
            _fig3a_params(E_C_kelvin=1.0)

    def test_rejects_negative_rates(self) -> None:
        with pytest.raises(ValueError, match="must be nonnegative"):
            _fig3a_params(r_L_Hz=-1.0)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_nonfinite_primitive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            _fig3a_params(Gamma_ph_00_Hz=bad)

    def test_derived_properties(self) -> None:
        params = _fig3a_params()
        assert params.omega_LR_kelvin == pytest.approx(0.5e9 * _H_OVER_KB, rel=1e-12)
        assert params.Delta_bar_kelvin == pytest.approx(49.25e9 * _H_OVER_KB, rel=1e-12)
        assert params.delta == pytest.approx(49.0 / 49.5, rel=1e-12)
        assert 0.0 < params.delta <= 1.0


# ═══════════════════════════════════════════════════════════════════════
#  Matrix elements (SI S25–S28)
# ═══════════════════════════════════════════════════════════════════════


class TestTransmonMatrixElements:
    def test_s_10_squared_formula(self) -> None:
        # s²_{10} = sqrt(E_C/(8E_J))
        E_J, E_C = 14.5 * _H_OVER_KB * 1e9, 0.29 * _H_OVER_KB * 1e9
        expected = np.sqrt(E_C / (8.0 * E_J))
        assert _s_10_squared(E_J, E_C) == pytest.approx(expected, rel=1e-14)

    def test_c_ii_leading_order_is_unity(self) -> None:
        # At leading order in E_J/E_C → ∞, c²_ii → 1. At E_J/E_C = 10¹²,
        # the residual correction is O(sqrt(E_C/(8 E_J))) ≈ 3.5e-7.
        c0_sq = _c_ii_squared(0, E_J_kelvin=1e6, E_C_kelvin=1e-6)
        c1_sq = _c_ii_squared(1, E_J_kelvin=1e6, E_C_kelvin=1e-6)
        assert c0_sq == pytest.approx(1.0, abs=1e-5)
        assert c1_sq == pytest.approx(1.0, abs=1e-5)

    def test_c_ii_transmon_correction_lowers_c(self) -> None:
        # For finite E_J/E_C, c²_ii < 1.
        c0_sq = _c_ii_squared(0, E_J_kelvin=10.0, E_C_kelvin=1.0)
        assert c0_sq < 1.0
        # c_11 has a larger correction than c_00 (coefficient (i+1/2) grows).
        c1_sq = _c_ii_squared(1, E_J_kelvin=10.0, E_C_kelvin=1.0)
        assert c1_sq < c0_sq


# ═══════════════════════════════════════════════════════════════════════
#  Incomplete Bessel function K_n(z, w)
# ═══════════════════════════════════════════════════════════════════════


class TestIncompleteBessel:
    def test_w_zero_recovers_full_Bessel_K0(self) -> None:
        # K_0(z, 0) = K_0(z) by definition of the incomplete form.
        for z in (0.1, 1.0, 5.0, 20.0):
            assert _K_incomplete(0, z, 0.0) == pytest.approx(k0(z), rel=1e-8)

    def test_w_zero_recovers_full_Bessel_K1(self) -> None:
        for z in (0.1, 1.0, 5.0, 20.0):
            assert _K_incomplete(1, z, 0.0) == pytest.approx(k1(z), rel=1e-8)

    def test_large_w_vanishes(self) -> None:
        # For w ≫ 1 and z = 1, integral is exponentially small.
        assert _K_incomplete(0, 1.0, 20.0) < 1e-8

    def test_monotonic_in_w(self) -> None:
        # ∂/∂w K_n(z, w) ≤ 0 (removing interval at the left shrinks integral).
        z = 2.0
        vals = [_K_incomplete(0, z, w) for w in (0.0, 0.5, 1.0, 2.0, 4.0)]
        for a, b in pairwise(vals):
            assert b < a

    def test_large_z_representable_tail_matches_high_precision_reference(self) -> None:
        # 80-digit mpmath reference for integral_0.5^inf exp(-500 cosh(t)) dt.
        # The old direct quad used its default absolute tolerance and returned
        # exactly zero even though this tail is representable in float64.
        value = _K_incomplete(0, 500.0, 0.5)
        assert value > 0.0
        assert np.log(value) == pytest.approx(
            -569.383921829082105084346828401,
            abs=2e-12,
        )

    def test_log_form_survives_true_float64_underflow(self) -> None:
        # The integral itself is below the minimum float64 subnormal, but its
        # logarithm remains useful to rate ratios and branching fractions.
        assert _K_incomplete(0, 12_000.0, 0.5) == 0.0
        assert _log_K_incomplete(0, 12_000.0, 0.5) == pytest.approx(
            -13_540.2527678644983501654427036,
            abs=2e-11,
        )

    def test_log_form_matches_high_precision_at_normal_scale(self) -> None:
        assert _log_K_incomplete(0, 2.0, 0.5) == pytest.approx(
            -2.96567680798589949012047945651,
            abs=2e-13,
        )

    @pytest.mark.parametrize("n", [-1, 2, 0.5, True])
    def test_rejects_unsupported_order(self, n: object) -> None:
        with pytest.raises(ValueError, match=r"integer order|orders n=0 and n=1"):
            _K_incomplete(n, 1.0, 0.5)  # type: ignore[arg-type]

    @pytest.mark.parametrize("z", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_z(self, z: float) -> None:
        with pytest.raises(ValueError, match="z must be finite and positive"):
            _K_incomplete(0, z, 0.5)

    @pytest.mark.parametrize("w", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_invalid_w(self, w: float) -> None:
        with pytest.raises(ValueError, match="w must be finite and nonnegative"):
            _K_incomplete(0, 1.0, w)


# ═══════════════════════════════════════════════════════════════════════
#  Tunneling rates — sign conventions and regime transitions
# ═══════════════════════════════════════════════════════════════════════


class TestTunnelingRates:
    def test_all_rates_positive(self) -> None:
        """Every tunneling rate in the coefficient bundle must be ≥ 0."""
        params = _fig3a_params()
        coefs = coefficients_from_physical_parameters(params)
        for name in ("gammas_L", "gammas_Rgt", "gammas_Rlt"):
            arr = getattr(coefs, name)
            assert np.all(arr >= 0.0), f"{name} has negative entries: {arr}"

    def test_Rlt_only_10_entry_nonzero(self) -> None:
        """M25 low-T ansatz: only Γ̃^{R<}_{10} survives; others are 0."""
        coefs = coefficients_from_physical_parameters(_fig3a_params())
        rlt = coefs.gammas_Rlt
        assert rlt[1, 0] > 0.0
        assert rlt[0, 0] == 0.0
        assert rlt[0, 1] == 0.0
        assert rlt[1, 1] == 0.0

    def test_gamma_L_ii_grows_with_gap_asymmetry(self) -> None:
        """Γ̃^L_{ii} ∝ sqrt((Δ_L-Δ_R)/(2Δ_L)) × e^y K_1(y) — rises with ω_LR."""
        params_small = _fig3a_params()
        params_large = replace(
            params_small,
            Delta_L_kelvin=54e9 * _H_OVER_KB,   # 5 GHz asymmetry
        )
        assert _gamma_L_ii(params_large, 0) > _gamma_L_ii(params_small, 0)

    def test_gamma_L_10_vs_01_case_I(self) -> None:
        """For ω_10 > ω_LR (case I), Γ̃^L_{01} is exp-suppressed rel. to Γ̃^L_{10}."""
        params = _fig3a_params()  # ω_10 = 264 mK, ω_LR = 24 mK
        assert params.omega_10_kelvin > params.omega_LR_kelvin
        g10 = _gamma_L_10(params)
        g01 = _gamma_L_01(params)
        # The ratio should be roughly exp((ω_10 - ω_LR)/T × some factor).
        # At T = 30 mK, (ω_10 - ω_LR)/T ≈ 8 → exp(8) ≈ 3000.
        assert g10 / g01 > 100.0

    def test_gamma_Rlt_10_case_I_vs_case_II(self) -> None:
        """Γ̃^{R<}_{10} finite for ω_10 > ω_LR (I); suppressed for ω_10 < ω_LR (II)."""
        # Case I: ω_10 = 5.5 GHz > ω_LR = 0.5 GHz.
        params_I = _fig3a_params()
        rate_I = _gamma_Rlt_10(params_I)
        # Case II: ω_10 = 0.5 GHz < ω_LR = 5 GHz (flip via large gap asymmetry).
        params_II = replace(
            params_I,
            Delta_L_kelvin=54e9 * _H_OVER_KB,
            omega_10_kelvin=0.5e9 * _H_OVER_KB,
        )
        rate_II = _gamma_Rlt_10(params_II)
        # Case-II rate is ∝ e^{-(ω_LR - ω_10)/T} — at T = 30 mK, ω_LR - ω_10 ≈ 216 mK,
        # so suppression factor ∝ e^{-7.2} ~ 7×10⁻⁴.
        # Both rates nonzero, but case II much smaller ratio-wise.
        assert rate_II > 0.0
        assert rate_II / rate_I < 0.1

    def test_gamma_Rgt_01_exp_suppressed(self) -> None:
        """Γ̃^{R>}_{01} ∝ e^{-ω_10/T} — near-zero at low T."""
        params = _fig3a_params(T_kelvin=0.020)
        coefs = coefficients_from_physical_parameters(params)
        # Exp(-ω_10/T) = exp(-13) ≈ 2e-6; rate should be much smaller than relaxation
        gamma01 = coefs.gammas_Rgt[0, 1]
        gamma10 = coefs.gammas_Rgt[1, 0]
        assert gamma01 / gamma10 < 1e-3


# ═══════════════════════════════════════════════════════════════════════
#  Detailed balance: ee channel, τ_E/τ_R ratio, thermal generation
# ═══════════════════════════════════════════════════════════════════════


class TestDetailedBalance:
    def test_gamma_ee_detailed_balance(self) -> None:
        params = _fig3a_params()
        coefs = coefficients_from_physical_parameters(params)
        ratio = coefs.gamma_ee[0, 1] / coefs.gamma_ee[1, 0]
        expected = np.exp(-params.omega_10_kelvin / params.T_kelvin)
        assert ratio == pytest.approx(expected, rel=1e-10)

    def test_tau_E_suppressed_relative_to_tau_R(self) -> None:
        # τ_E⁻¹ ∝ e^{-ω_LR/T} × τ_R⁻¹ × (DoS factor).
        # At T ≪ ω_LR, τ_E⁻¹ ≪ τ_R⁻¹.
        params = _fig3a_params(T_kelvin=0.005)  # well below ω_LR = 24 mK
        tR = _tau_R_inverse(params)
        tE = _tau_E_inverse(params)
        assert tE < tR

    def test_tau_E_to_tau_R_ratio_has_correct_sign(self) -> None:
        # τ_E/τ_R = erfc/erf ∈ (0, 1) (erf, erfc both positive; erfc < erf
        # at large ω_LR/T). The exp(-ω_LR/T) suppression is inside erfc.
        params = _fig3a_params(T_kelvin=0.030)
        tR = _tau_R_inverse(params)
        tE = _tau_E_inverse(params)
        assert 0.0 < tE / tR < 1.0

    def test_tau_R_exact_matches_s50_series_in_domain(self) -> None:
        # The exact S48/S49 quadrature must reduce to the S50 series in
        # its validity domain T ≪ ω_LR. At T/ω_LR ≈ 0.04 the residual
        # difference (series truncation + the O(ω_LR/Δ_R) correction the
        # series drops) is ~1%.
        from qpsim.services.rate_equation_coefficients import (
            _tau_R_inverse_series_s50,
        )

        params = _fig3a_params(T_kelvin=0.001)  # T/ω_LR ≈ 0.042
        exact = _tau_R_inverse(params)
        series = _tau_R_inverse_series_s50(params)
        assert series / exact == pytest.approx(1.0, abs=0.02)

    def test_tau_R_absolute_normalization_pinned(self) -> None:
        # The series/exact ratio tests cancel the shared prefactor and
        # cannot detect a normalization change (2026-07-20 review). Pin the
        # ABSOLUTE rate at the Fig 3a parameter set, including the
        # (Delta_R/Delta_bar)^3 = 0.98486 conversion factor required by
        # the paper's r^{R<} ~= 8*pi*b_R*Delta_bar^3 definition (M25 v2
        # App. D.3, verified against the arXiv math source; a first-pass
        # adjudication wrongly refuted the factor from a truncated
        # extract). Dropping the factor inflates these by 1.54%.
        assert _tau_R_inverse(_fig3a_params(T_kelvin=0.030)) == pytest.approx(
            4.059143818533248, rel=1e-6
        )
        assert _tau_R_inverse(_fig3a_params(T_kelvin=0.150)) == pytest.approx(
            207.50905780791763, rel=1e-6
        )

    def test_tau_R_exact_departs_from_series_out_of_domain(self) -> None:
        # 2026-07-19 audit H4 regression guard: on the shipped Fig 3a
        # sweep the series is evaluated far outside T ≪ ω_LR; at
        # T/ω_LR ≈ 6.25 (150 mK, ω_LR = 24 mK) it under-predicts the
        # exact S48/S49 rate by ~5x. If this assert fires because the
        # ratio moved toward 1, _tau_R_inverse has silently regressed
        # to the series.
        from qpsim.services.rate_equation_coefficients import (
            _tau_R_inverse_series_s50,
        )

        params = _fig3a_params(T_kelvin=0.150)
        exact = _tau_R_inverse(params)
        series = _tau_R_inverse_series_s50(params)
        assert series / exact == pytest.approx(0.203, abs=0.02)

    def test_thermal_generation_scales_with_boltzmann(self) -> None:
        # g^{pn}_L ∝ T × e^{-2Δ_L/T} — rises rapidly with T.
        coefs_cold = coefficients_from_physical_parameters(_fig3a_params(T_kelvin=0.020))
        coefs_hot = coefficients_from_physical_parameters(_fig3a_params(T_kelvin=0.100))
        g_L_cold_pn = coefs_cold.g_L - 1.0   # subtract photon contribution
        g_L_hot_pn = coefs_hot.g_L - 1.0
        # At T = 20 mK with Δ_L = 2.38 K: e^{-2Δ/T} = e^{-238} → underflow → ≈ 0.
        # At T = 100 mK: e^{-47.6} ≈ 2e-21. Still tiny but nonzero.
        assert g_L_cold_pn < g_L_hot_pn

    def test_generation_equals_recombination_at_thermal_eq_L(self) -> None:
        # At full thermal equilibrium (µ = 0), g^{pn}_L = r^L × (x_L^{eq})².
        params = _fig3a_params(T_kelvin=0.060)
        coefs = coefficients_from_physical_parameters(params)
        T, Delta_L = params.T_kelvin, params.Delta_L_kelvin
        x_L_eq_sq = (2.0 * np.pi * T / Delta_L) * np.exp(-2.0 * Delta_L / T)
        g_pn_L = coefs.g_L - params.g_ph_L_Hz
        expected = coefs.r_L * x_L_eq_sq
        assert g_pn_L == pytest.approx(expected, rel=1e-12)

    def test_R_generation_branching_is_linear_in_erf_erfc(self) -> None:
        # Pair-breaking creates two independent QPs; each partitions
        # between R< and R> with probability erf, erfc. Counting both
        # QPs per event gives g_R< ∝ erf, g_R> ∝ erfc (first power).
        # Squared forms would violate g_R< + g_R> = r × x_R² (the
        # branching-agnostic total that feeds the Eq. 8 Lambert-W).
        params = _fig3a_params(T_kelvin=0.060)
        coefs = coefficients_from_physical_parameters(params)
        T, Delta_R = params.T_kelvin, params.Delta_R_kelvin
        omega_LR = params.omega_LR_kelvin
        prefactor = coefs.r_Rlt * (2.0 * np.pi * T / Delta_R) * np.exp(-2.0 * Delta_R / T)
        erf_z = erf(np.sqrt(omega_LR / T))
        erfc_z = erfc(np.sqrt(omega_LR / T))
        g_pn_Rlt = coefs.g_Rlt - params.g_ph_Rlt_Hz
        g_pn_Rgt = coefs.g_Rgt - params.g_ph_Rgt_Hz
        assert g_pn_Rlt == pytest.approx(prefactor * erf_z, rel=1e-12)
        assert g_pn_Rgt == pytest.approx(prefactor * erfc_z, rel=1e-12)

    def test_R_generation_sum_matches_Eq8_prefactor(self) -> None:
        # Branching-invariant check: g_R< + g_R> must equal
        # r × x_R² = r × (2πT/Δ_R) e^{-2Δ_R/T}, independent of the
        # erf/erfc partitioning. This is the quantity that feeds the
        # Lambert-W crossover formula (M25 Eq. 8).
        params = _fig3a_params(T_kelvin=0.080)
        coefs = coefficients_from_physical_parameters(params)
        T, Delta_R = params.T_kelvin, params.Delta_R_kelvin
        g_pn_total = (
            (coefs.g_Rlt - params.g_ph_Rlt_Hz)
            + (coefs.g_Rgt - params.g_ph_Rgt_Hz)
        )
        expected_total = coefs.r_Rlt * (2.0 * np.pi * T / Delta_R) * np.exp(-2.0 * Delta_R / T)
        assert g_pn_total == pytest.approx(expected_total, rel=1e-12)

    def test_tau_E_over_tau_R_equals_erfc_over_erf(self) -> None:
        # Detailed balance at µ=0 requires τ_E⁻¹ x_R<^eq = τ_R⁻¹ x_R>^eq,
        # i.e. τ_E⁻¹/τ_R⁻¹ = x_R>^eq / x_R<^eq = erfc/erf. The e^{-ω_LR/T}
        # Boltzmann factor is already inside erfc asymptotically —
        # multiplying by it again (as the first draft did) would
        # double-count the suppression.
        params = _fig3a_params(T_kelvin=0.030)
        tR = _tau_R_inverse(params)
        tE = _tau_E_inverse(params)
        z = np.sqrt(params.omega_LR_kelvin / params.T_kelvin)
        assert tE / tR == pytest.approx(erfc(z) / erf(z), rel=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  Branching fraction
# ═══════════════════════════════════════════════════════════════════════


class TestBranchingFraction:
    def test_xi_in_unit_interval(self) -> None:
        params = _fig3a_params()
        xi = _branching_fraction(params)
        assert 0.0 <= xi <= 1.0

    def test_xi_exp_suppressed_at_low_T(self) -> None:
        # ξ ∝ e^{-min(ω_10, ω_LR)/T} at low T — falls rapidly as T → 0.
        params_warm = _fig3a_params(T_kelvin=0.050)
        params_cold = _fig3a_params(T_kelvin=0.015)
        xi_warm = _branching_fraction(params_warm)
        xi_cold = _branching_fraction(params_cold)
        assert xi_cold < xi_warm

    def test_ultracold_xi_matches_high_precision_reference(self) -> None:
        # 80-digit mpmath evaluation of K_0(z,w)/K_0(z) at the Fig. 3a
        # parameters. Both unscaled integrals used to underflow to zero.
        xi = _branching_fraction(_fig3a_params(T_kelvin=1e-4))
        assert xi == pytest.approx(2.11583916322907e-106, rel=2e-12)

    def test_ultracold_xi_underflows_only_when_mathematically_required(self) -> None:
        # At 10 microkelvin the true value is about 7.91e-1045, so exact
        # float64 zero is correct rather than evidence of a 0/0 failure.
        xi = _branching_fraction(_fig3a_params(T_kelvin=1e-5))
        assert np.isfinite(xi)
        assert xi == 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Integration: build coefficients → solve steady state
# ═══════════════════════════════════════════════════════════════════════


class TestIntegrationWithSolver:
    def test_ultracold_coefficient_bundle_is_finite(self) -> None:
        # Before the scaled/log-domain implementation, erfc and both Bessel
        # factors underflowed independently and coefficient construction
        # failed with 0/0 or inf*0 at this temperature.
        coefs = coefficients_from_physical_parameters(
            _fig3a_params(T_kelvin=1e-5),
        )
        for value in vars(coefs).values():
            assert np.all(np.isfinite(value))

    def test_fig3a_low_T_steady_state_trapped_band_dominant(self) -> None:
        """At T ≪ ω_LR, Fig 3a should have x_{R<} > x_L > x_{R>}.

        Uses the multi-seed production path: after the (Δ_R/Δ̄)³ tau_R
        normalization fix the default single seed stalls at this point
        with an in-tolerance residual (scipy no-progress status), and
        multi-seed reseeding — what the shipped sweeps run — resolves it
        without loosening the strict acceptance gate.
        """
        params = _fig3a_params(T_kelvin=0.020)
        coefs = coefficients_from_physical_parameters(params)
        ss = solve_rate_equation_steady_state_multi_seed(coefs)
        assert ss.x_Rlt > ss.x_L > ss.x_Rgt
        assert 0.0 < ss.p_1 < 1e-3  # negligible excitation at 20 mK

    def test_fig3a_warmer_densities_equilibrate(self) -> None:
        """As T crosses ω_LR, x_R< and x_R> equilibrate (τ_E⁻¹ → τ_R⁻¹).

        This is the nonequilibrium → local-quasiequilibrium transition:
        at low T, x_R< ≫ x_R> (trapped band dominant); as T rises past
        ω_LR ≈ 24 mK, intraband excitation R< → R> catches up with
        relaxation R> → R<, and the two sub-bands equilibrate. x_L
        continues to rise monotonically from thermal-phonon generation.
        """
        cold = solve_rate_equation_steady_state(
            coefficients_from_physical_parameters(_fig3a_params(T_kelvin=0.050)),
        )
        hot = solve_rate_equation_steady_state(
            coefficients_from_physical_parameters(_fig3a_params(T_kelvin=0.120)),
        )
        assert hot.x_L > cold.x_L
        cold_ratio = cold.x_Rlt / cold.x_Rgt
        hot_ratio = hot.x_Rlt / hot.x_Rgt
        assert cold_ratio > hot_ratio
        assert hot_ratio == pytest.approx(1.0, rel=0.3)

    def test_residual_zero_at_returned_state(self) -> None:
        """Sanity: the returned state satisfies R(y) ≈ 0 to the configured tol."""
        params = _fig3a_params(T_kelvin=0.040)
        coefs = coefficients_from_physical_parameters(params)
        ss = solve_rate_equation_steady_state(coefs)
        assert ss.residual_inf_norm < 1e-3
