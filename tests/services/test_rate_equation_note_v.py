"""Tests for M25 SI Note V (photon-assisted tunneling) helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.services.rate_equation import M25Coefficients, _rate_equation_residual
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    _c_ii_squared,
    _s_10_squared,
    _S_ph_Rgt,
    _S_ph_Rlt,
    _S_ph_total,
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters,
    coefficients_from_physical_parameters_with_photon_drive,
)
from scipy.integrate import quad

_H_OVER_KB = 4.799243e-11  # K/Hz
_K_B_J_PER_K = 1.380649e-23


def _fig3a_params(**overrides: float) -> M25PhysicalParameters:
    base = {
        "Delta_L_kelvin": 49.5e9 * _H_OVER_KB,
        "Delta_R_kelvin": 49.0e9 * _H_OVER_KB,
        "omega_10_kelvin": 5.5e9 * _H_OVER_KB,
        "T_kelvin": 0.030,
        "E_J_kelvin": 14.5e9 * _H_OVER_KB,
        "E_C_kelvin": 290e6 * _H_OVER_KB,
        "R_T_Hz": 8.0 * 14.5e9 * (49.25 / 49.5),
        "r_L_Hz": 6.25e6,
        "r_Rlt_Hz": 6.25e6,
        "Gamma_ee_10_Hz": 100e3,
    }
    base.update(overrides)
    return M25PhysicalParameters(**base)


def _fig3a_drive(scale_Hz: float = 1.0) -> M25PhotonDrive:
    return M25PhotonDrive(
        omega_nu_kelvin=119e9 * _H_OVER_KB,
        Gamma_nu_scale_Hz=scale_Hz,
        nu_0_per_J_per_m3=0.73e47,
        volume_m3=506e-6 * 240e-6 * 0.028e-6,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Spectral-density thresholds and sum identity
# ═══════════════════════════════════════════════════════════════════════


class TestSpectralDensityThresholds:
    @pytest.mark.parametrize("sign", [+1, -1])
    def test_S_ph_total_vanishes_below_1_plus_z(self, sign: int) -> None:
        z = 0.5
        for x in (0.0, 0.5, 1.0, 1.3, 1.5):
            assert _S_ph_total(x, z, sign) == 0.0

    @pytest.mark.parametrize("sign", [+1, -1])
    def test_S_ph_Rgt_vanishes_below_x_eq_2(self, sign: int) -> None:
        # Even above the 1+z threshold, R> is blocked until x > 2 (both
        # QPs need to clear Δ_L).
        z = 0.5
        for x in (0.0, 1.0, 1.6, 1.99, 2.0):
            assert _S_ph_Rgt(x, z, sign) == 0.0

    def test_S_ph_total_nonzero_above_threshold(self) -> None:
        z = 0.5
        assert _S_ph_total(1.6, z, +1) > 0.0
        assert _S_ph_total(1.6, z, -1) >= 0.0

    def test_rejects_z_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="Δ_R/Δ_L must lie in"):
            _S_ph_total(3.0, 0.0, +1)
        with pytest.raises(ValueError, match="Δ_R/Δ_L must lie in"):
            _S_ph_total(3.0, 1.0, +1)

    def test_rejects_invalid_sign(self) -> None:
        with pytest.raises(ValueError, match="sign must be"):
            _S_ph_total(3.0, 0.5, 0)


class TestSpectralDensitySumIdentity:
    """``S_Rgt + S_Rlt = S_total`` by construction, but nonobvious
    numerically: both branches use different elliptic integrals, and
    the identity only holds if the common prefactor conventions match."""

    @pytest.mark.parametrize("z", [0.1, 0.3, 0.5, 0.7, 0.99])
    @pytest.mark.parametrize("sign", [+1, -1])
    def test_sum_identity_across_z_and_sign(self, z: float, sign: int) -> None:
        for x in (2.1, 2.5, 3.0, 5.0):
            total = _S_ph_total(x, z, sign)
            rgt = _S_ph_Rgt(x, z, sign)
            rlt = _S_ph_Rlt(x, z, sign)
            assert rgt + rlt == pytest.approx(total, rel=1e-12, abs=1e-12)

    def test_Rlt_alone_below_2_gap(self) -> None:
        # For 1+z < x < 2, R> is kinematically excluded but R< is
        # allowed: S_total = S_Rlt, S_Rgt = 0.
        z = 0.5
        x = 1.7  # between 1+z=1.5 and 2
        assert _S_ph_Rgt(x, z, +1) == 0.0
        assert _S_ph_Rlt(x, z, +1) == pytest.approx(_S_ph_total(x, z, +1), rel=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  Closed form vs direct numerical integration (SI S56 definition)
# ═══════════════════════════════════════════════════════════════════════


class TestSpectralDensityAgainstNumericalIntegral:
    """SI S56 as direct integral, compared against S57 closed form."""

    def _numerical_S_ph(self, x: float, z: float, sign: int) -> float:
        # y' = x - y via delta function; integrate y from 1 to x-z
        def integrand(y: float) -> float:
            yp = x - y
            if y <= 1.0 or yp <= z:
                return 0.0
            return (y * yp + sign * z) / (
                np.sqrt(y * y - 1.0) * np.sqrt(yp * yp - z * z)
            )
        val, _ = quad(integrand, 1.0, x - z, limit=500)
        return float(val)

    @pytest.mark.parametrize(
        ("x", "z", "sign"),
        [
            (2.5, 0.5, +1),
            (2.5, 0.5, -1),
            (3.0, 0.3, +1),
            (3.0, 0.3, -1),
            (2.4, 0.99, +1),
            (2.4, 0.99, -1),
            (5.0, 0.1, +1),
            (5.0, 0.1, -1),
        ],
    )
    def test_closed_form_matches_direct_integral(
        self, x: float, z: float, sign: int
    ) -> None:
        numerical = self._numerical_S_ph(x, z, sign)
        closed = _S_ph_total(x, z, sign)
        assert closed == pytest.approx(numerical, rel=1e-10)


# ═══════════════════════════════════════════════════════════════════════
#  M25PhotonDrive validation
# ═══════════════════════════════════════════════════════════════════════


class TestPhotonDriveValidation:
    def test_rejects_nonpositive_omega_nu(self) -> None:
        with pytest.raises(ValueError, match="omega_nu_kelvin must be positive"):
            M25PhotonDrive(
                omega_nu_kelvin=0.0,
                Gamma_nu_scale_Hz=1.0,
                nu_0_per_J_per_m3=1e47,
                volume_m3=1e-18,
            )

    def test_rejects_nonpositive_volume(self) -> None:
        with pytest.raises(ValueError, match="volume_m3 must be positive"):
            M25PhotonDrive(
                omega_nu_kelvin=5.0,
                Gamma_nu_scale_Hz=1.0,
                nu_0_per_J_per_m3=1e47,
                volume_m3=0.0,
            )

    def test_rejects_negative_scale(self) -> None:
        with pytest.raises(ValueError, match="Gamma_nu_scale_Hz must be positive"):
            M25PhotonDrive(
                omega_nu_kelvin=5.0,
                Gamma_nu_scale_Hz=-1.0,
                nu_0_per_J_per_m3=1e47,
                volume_m3=1e-18,
            )


# ═══════════════════════════════════════════════════════════════════════
#  Γ_ν calibration: back-solved scale yields target Γ^{ph}_{00}
# ═══════════════════════════════════════════════════════════════════════


class TestCalibration:
    def test_back_solved_scale_exactly_reproduces_target(self) -> None:
        params = _fig3a_params()
        drive_template = _fig3a_drive(scale_Hz=1.0)  # placeholder
        target_Hz = 300.0
        scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
            params, drive_template, target_Hz
        )
        drive = replace(drive_template, Gamma_nu_scale_Hz=scale)
        coefs = coefficients_from_physical_parameters_with_photon_drive(params, drive)
        assert coefs.gamma_ph[0, 0] == pytest.approx(target_Hz, rel=1e-12)

    def test_calibration_scales_linearly(self) -> None:
        # Linear in Gamma_nu_scale_Hz, so doubling target doubles scale.
        params = _fig3a_params()
        drive_template = _fig3a_drive()
        scale1 = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(params, drive_template, 300.0)
        scale2 = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(params, drive_template, 600.0)
        assert scale2 == pytest.approx(2.0 * scale1, rel=1e-12)

    def test_rejects_photon_below_threshold(self) -> None:
        # Fake a sub-threshold photon (Δ_L + Δ_R ≈ 4.73 K, so 2 K too small).
        params = _fig3a_params()
        bad_drive = replace(_fig3a_drive(), omega_nu_kelvin=2.0)
        with pytest.raises(ValueError, match="below the pair-breaking threshold"):
            calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(params, bad_drive, 300.0)


# ═══════════════════════════════════════════════════════════════════════
#  Builder: matrix-element structure and Γ^{ph}_{ij} ratios
# ═══════════════════════════════════════════════════════════════════════


class TestBuilderMatrixElementStructure:
    def _build(self) -> M25Coefficients:
        params = _fig3a_params()
        drive_template = _fig3a_drive()
        scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
            params, drive_template, 300.0
        )
        return coefficients_from_physical_parameters_with_photon_drive(
            params, replace(drive_template, Gamma_nu_scale_Hz=scale)
        )

    def test_Gamma_ph_11_over_00_is_c11_over_c00_ratio(self) -> None:
        # Γ^{ph}_{ii} = A × c²_{ii} × S^-(ω_ν/Δ_L; z). Ratio 11/00
        # is c²_{11}/c²_{00}.
        coefs = self._build()
        params = _fig3a_params()
        c00 = _c_ii_squared(0, params.E_J_kelvin, params.E_C_kelvin)
        c11 = _c_ii_squared(1, params.E_J_kelvin, params.E_C_kelvin)
        assert coefs.gamma_ph[1, 1] / coefs.gamma_ph[0, 0] == pytest.approx(
            c11 / c00, rel=1e-12
        )

    def test_Gamma_ph_10_over_00_is_matrix_element_and_spectrum_ratio(self) -> None:
        # Γ^{ph}_{10} = A × s²_{10} × S^+((ω_ν+ω_10)/Δ_L; z)
        # Γ^{ph}_{00} = A × c²_{00} × S^-(ω_ν/Δ_L; z)
        coefs = self._build()
        params = _fig3a_params()
        s10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
        c00_sq = _c_ii_squared(0, params.E_J_kelvin, params.E_C_kelvin)
        z = params.delta
        x_00 = 119e9 * _H_OVER_KB / params.Delta_L_kelvin
        x_10 = (119e9 + 5.5e9) * _H_OVER_KB / params.Delta_L_kelvin
        expected_ratio = (s10_sq * _S_ph_total(x_10, z, +1)) / (
            c00_sq * _S_ph_total(x_00, z, -1)
        )
        assert coefs.gamma_ph[1, 0] / coefs.gamma_ph[0, 0] == pytest.approx(
            expected_ratio, rel=1e-12
        )

    def test_Gamma_ph_01_versus_10_uses_different_energy_argument(self) -> None:
        # Γ^{ph}_{10} uses S^+((ω_ν + ω_10)/Δ_L) — "photon + qubit
        # relaxation"; Γ^{ph}_{01} uses S^+((ω_ν − ω_10)/Δ_L) —
        # "photon − qubit excitation". Same matrix element s²_{10},
        # different spectral arg, so ratio = S^+(x_10)/S^+(x_01).
        coefs = self._build()
        params = _fig3a_params()
        z = params.delta
        x_10 = (119e9 + 5.5e9) * _H_OVER_KB / params.Delta_L_kelvin
        x_01 = (119e9 - 5.5e9) * _H_OVER_KB / params.Delta_L_kelvin
        expected = _S_ph_total(x_10, z, +1) / _S_ph_total(x_01, z, +1)
        assert coefs.gamma_ph[1, 0] / coefs.gamma_ph[0, 1] == pytest.approx(
            expected, rel=1e-12
        )


# ═══════════════════════════════════════════════════════════════════════
#  Builder: per-state g^{ph} and /N_CP(R) invariant
# ═══════════════════════════════════════════════════════════════════════


class TestBuilderPerStateGeneration:
    def _build_and_params(self) -> tuple[M25Coefficients, M25PhysicalParameters, M25PhotonDrive]:
        params = _fig3a_params()
        drive_template = _fig3a_drive()
        scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
            params, drive_template, 300.0
        )
        drive = replace(drive_template, Gamma_nu_scale_Hz=scale)
        return (
            coefficients_from_physical_parameters_with_photon_drive(params, drive),
            params,
            drive,
        )

    def test_g_ph_R_state_0_equals_Gamma_ph_sum_over_NCP(self) -> None:
        # Main-text identity: g^{ph}_R state 0 = (Γ^{ph}_{00} + Γ^{ph}_{01}) / N_CP(R)
        coefs, params, drive = self._build_and_params()
        N_CP_R = (
            2.0 * drive.nu_0_per_J_per_m3 * _K_B_J_PER_K
            * params.Delta_R_kelvin * drive.volume_m3
        )
        g_ph_R_state_0 = (
            coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0]
        )
        expected = (coefs.gamma_ph[0, 0] + coefs.gamma_ph[0, 1]) / N_CP_R
        assert g_ph_R_state_0 == pytest.approx(expected, rel=1e-12)

    def test_g_ph_R_state_1_equals_Gamma_ph_sum_over_NCP(self) -> None:
        coefs, params, drive = self._build_and_params()
        N_CP_R = (
            2.0 * drive.nu_0_per_J_per_m3 * _K_B_J_PER_K
            * params.Delta_R_kelvin * drive.volume_m3
        )
        g_ph_R_state_1 = (
            coefs.g_ph_Rlt_per_state[1] + coefs.g_ph_Rgt_per_state[1]
        )
        expected = (coefs.gamma_ph[1, 1] + coefs.gamma_ph[1, 0]) / N_CP_R
        assert g_ph_R_state_1 == pytest.approx(expected, rel=1e-12)

    def test_g_ph_L_equals_delta_times_g_ph_R(self) -> None:
        # Main text: g^{ph}_L = g^{ph}_R × δ
        coefs, params, _ = self._build_and_params()
        g_ph_R_state_0 = coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0]
        assert coefs.g_ph_L_per_state[0] == pytest.approx(
            params.delta * g_ph_R_state_0, rel=1e-12
        )

    def test_Rlt_Rgt_split_matches_spectral_fraction(self) -> None:
        # Split uses the S^{<,-}/S^- fraction at x = ω_ν/Δ_L.
        coefs, params, drive = self._build_and_params()
        x_00 = drive.omega_nu_kelvin / params.Delta_L_kelvin
        z = params.delta
        frac_Rlt_expected = _S_ph_Rlt(x_00, z, -1) / _S_ph_total(x_00, z, -1)
        g_ph_R_state_0 = (
            coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0]
        )
        assert coefs.g_ph_Rlt_per_state[0] / g_ph_R_state_0 == pytest.approx(
            frac_Rlt_expected, rel=1e-12
        )


# ═══════════════════════════════════════════════════════════════════════
#  Residual integration: g^{ph}(p) is evaluated correctly
# ═══════════════════════════════════════════════════════════════════════


class TestResidualPopulationDependence:
    """The residual should include ``p_0·g_ph[0] + p_1·g_ph[1]`` in g_α."""

    def test_residual_at_p_1_eq_0_uses_state_0_photon_gen(self) -> None:
        # Craft a coefficient bundle with ONLY photon-driven generation
        # (zero thermal g, zero tunneling, zero ee, zero ph_gamma,
        # zero recombination) so the residual on x_L reduces to
        # ẋ_L = g_L(p) - 0 - ... .
        coefs = M25Coefficients(
            gammas_L=np.zeros((2, 2)),
            gammas_Rgt=np.zeros((2, 2)),
            gammas_Rlt=np.zeros((2, 2)),
            gamma_ee=np.zeros((2, 2)),
            gamma_ph=np.zeros((2, 2)),
            r_L=0.0, r_Rgt=0.0, r_Rlt=0.0, r_cross=0.0,
            g_L=0.0, g_Rgt=0.0, g_Rlt=0.0,
            tau_R_inv=0.0, tau_E_inv=0.0,
            xi=0.0, delta=1.0,
            g_ph_L_per_state=np.array([10.0, 20.0]),
            g_ph_Rgt_per_state=np.array([5.0, 15.0]),
            g_ph_Rlt_per_state=np.array([3.0, 7.0]),
        )
        y = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)  # p_1=0, all x=0
        R = _rate_equation_residual(y, coefs)
        # ẋ_L at p_0=1, x=0: just g_L_eff = g_L_thermal + p_0·g_ph[0] + p_1·g_ph[1] = 10
        assert R[1] == pytest.approx(10.0, rel=1e-12)
        assert R[2] == pytest.approx(5.0, rel=1e-12)   # x_Rgt_dot
        assert R[3] == pytest.approx(3.0, rel=1e-12)   # x_Rlt_dot

    def test_residual_at_p_1_eq_1_uses_state_1_photon_gen(self) -> None:
        coefs = M25Coefficients(
            gammas_L=np.zeros((2, 2)),
            gammas_Rgt=np.zeros((2, 2)),
            gammas_Rlt=np.zeros((2, 2)),
            gamma_ee=np.zeros((2, 2)),
            gamma_ph=np.zeros((2, 2)),
            r_L=0.0, r_Rgt=0.0, r_Rlt=0.0, r_cross=0.0,
            g_L=0.0, g_Rgt=0.0, g_Rlt=0.0,
            tau_R_inv=0.0, tau_E_inv=0.0,
            xi=0.0, delta=1.0,
            g_ph_L_per_state=np.array([10.0, 20.0]),
            g_ph_Rgt_per_state=np.array([5.0, 15.0]),
            g_ph_Rlt_per_state=np.array([3.0, 7.0]),
        )
        y = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # p_1=1 → p_0=0
        R = _rate_equation_residual(y, coefs)
        assert R[1] == pytest.approx(20.0, rel=1e-12)
        assert R[2] == pytest.approx(15.0, rel=1e-12)
        assert R[3] == pytest.approx(7.0, rel=1e-12)

    def test_residual_linear_interpolation_at_p_1_eq_half(self) -> None:
        coefs = M25Coefficients(
            gammas_L=np.zeros((2, 2)),
            gammas_Rgt=np.zeros((2, 2)),
            gammas_Rlt=np.zeros((2, 2)),
            gamma_ee=np.zeros((2, 2)),
            gamma_ph=np.zeros((2, 2)),
            r_L=0.0, r_Rgt=0.0, r_Rlt=0.0, r_cross=0.0,
            g_L=1.0, g_Rgt=2.0, g_Rlt=3.0,   # thermal scalars
            tau_R_inv=0.0, tau_E_inv=0.0,
            xi=0.0, delta=1.0,
            g_ph_L_per_state=np.array([10.0, 20.0]),
            g_ph_Rgt_per_state=np.array([5.0, 15.0]),
            g_ph_Rlt_per_state=np.array([3.0, 7.0]),
        )
        y = np.array([0.5, 0.0, 0.0, 0.0], dtype=float)
        R = _rate_equation_residual(y, coefs)
        # g_L_eff = 1 + 0.5·10 + 0.5·20 = 16
        assert R[1] == pytest.approx(16.0, rel=1e-12)
        # g_Rgt_eff = 2 + 0.5·5 + 0.5·15 = 12
        assert R[2] == pytest.approx(12.0, rel=1e-12)
        # g_Rlt_eff = 3 + 0.5·3 + 0.5·7 = 8
        assert R[3] == pytest.approx(8.0, rel=1e-12)


# ═══════════════════════════════════════════════════════════════════════
#  Backward compatibility: unused per-state arrays don't affect behavior
# ═══════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    def test_existing_builder_leaves_per_state_arrays_zero(self) -> None:
        # coefficients_from_physical_parameters (Stage-A coefficients
        # without a photon drive) should leave per-state arrays at zero.
        params = _fig3a_params(g_ph_L_Hz=0.0, g_ph_Rlt_Hz=0.0, g_ph_Rgt_Hz=0.0)
        coefs = coefficients_from_physical_parameters(params)
        assert np.all(coefs.g_ph_L_per_state == 0.0)
        assert np.all(coefs.g_ph_Rgt_per_state == 0.0)
        assert np.all(coefs.g_ph_Rlt_per_state == 0.0)

    def test_M25Coefficients_rejects_wrong_per_state_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            M25Coefficients(
                gammas_L=np.zeros((2, 2)),
                gammas_Rgt=np.zeros((2, 2)),
                gammas_Rlt=np.zeros((2, 2)),
                gamma_ee=np.zeros((2, 2)),
                gamma_ph=np.zeros((2, 2)),
                r_L=0.0, r_Rgt=0.0, r_Rlt=0.0, r_cross=0.0,
                g_L=0.0, g_Rgt=0.0, g_Rlt=0.0,
                tau_R_inv=0.0, tau_E_inv=0.0,
                xi=0.0, delta=1.0,
                g_ph_L_per_state=np.zeros(3),  # wrong shape!
            )
