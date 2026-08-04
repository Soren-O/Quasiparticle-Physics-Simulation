# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review 2026-08-03, packet P08 regressions.

Covers the M25 Note-V builder's thermal-generation fields, the documented
photon-assisted matrix-element structure, the ν_0 / rho_F unit guards, and
the inert ``Material.tau_s``/``tau_r`` fields.  Every change under test is
numerically neutral for the shipped callers, so the builder assertions are
bit-exact against the closed forms.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from qpsim.materials.database import Material, validate_rho_F_eV
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    _c_ii_squared,
    _g_pn_L,
    _g_pn_Rgt,
    _g_pn_Rlt,
    _s_10_squared,
    _S_ph_total,
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)

_H_OVER_KB = 6.62607015e-34 / 1.380649e-23


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


def _calibrated_bundle(params: M25PhysicalParameters):
    template = _fig3a_drive()
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(params, template, 300.0)
    return coefficients_from_physical_parameters_with_photon_drive(
        params, replace(template, Gamma_nu_scale_Hz=scale)
    )


# ═══════════════════════════════════════════════════════════════════════
#  Note-V builder: thermal generation is the closed form, not a residue
# ═══════════════════════════════════════════════════════════════════════


class TestThermalGenerationFields:
    def test_matches_closed_forms_exactly_without_scalar_photon_input(self) -> None:
        params = _fig3a_params()
        coefs = _calibrated_bundle(params)
        # abs=0: these are the identical float expressions, not an
        # approximation of them.
        assert coefs.g_L == _g_pn_L(params)
        assert coefs.g_Rgt == _g_pn_Rgt(params)
        assert coefs.g_Rlt == _g_pn_Rlt(params)

    def test_scalar_photon_input_does_not_perturb_the_thermal_source(self) -> None:
        # `Gamma_ph_ij_Hz` / `g_ph_α_Hz` are documented as ignored by this
        # builder.  Recovering the thermal part by subtracting the scalar
        # back off annihilated it: g^{pn} is 1e-40-1e-9 Hz against an O(1)
        # Hz scalar, so the sum lost every bit of the small term.
        loud = _fig3a_params(g_ph_L_Hz=1.0, g_ph_Rlt_Hz=100.0, g_ph_Rgt_Hz=0.1)
        coefs = _calibrated_bundle(loud)
        assert coefs.g_L == _g_pn_L(loud)
        assert coefs.g_Rgt == _g_pn_Rgt(loud)
        assert coefs.g_Rlt == _g_pn_Rlt(loud)
        assert coefs.g_L > 0.0
        assert coefs.g_Rgt > 0.0
        assert coefs.g_Rlt > 0.0


# ═══════════════════════════════════════════════════════════════════════
#  Γ^{ph}_{ij}: power-law off-diagonal suppression, i-dependent diagonal
# ═══════════════════════════════════════════════════════════════════════


class TestPhotonAssistedRateStructure:
    def test_off_diagonal_entries_are_power_law_suppressed(self) -> None:
        params = _fig3a_params()
        gamma_ph = _calibrated_bundle(params).gamma_ph
        # The 01/10 channels carry s_10² = √(E_C/(8E_J)) (SI Eq. S25), a
        # power law, partly offset by S⁺/S⁻ at x = ω_ν/Δ_L — never the
        # exp-suppressed c_10.  ~0.29-0.30 × Γ^{ph}_{00}, not ~0.
        assert gamma_ph[0, 0] == pytest.approx(300.0, rel=1e-12)
        assert gamma_ph[0, 1] == pytest.approx(85.7086, rel=1e-5)
        assert gamma_ph[1, 0] == pytest.approx(90.3494, rel=1e-5)
        assert _s_10_squared(params.E_J_kelvin, params.E_C_kelvin) == pytest.approx(0.05)
        x_00 = _fig3a_drive().omega_nu_kelvin / params.Delta_L_kelvin
        ratio = _S_ph_total(x_00, params.delta, sign=+1) / _S_ph_total(
            x_00, params.delta, sign=-1
        )
        assert ratio == pytest.approx(5.5674, rel=1e-4)

    def test_diagonal_entries_differ_at_next_to_leading_order(self) -> None:
        # Γ^{ph}_{00} = Γ^{ph}_{11} only at strict leading order: the SI
        # Eq. S27 correction to c_ii is i-dependent.
        params = _fig3a_params()
        gamma_ph = _calibrated_bundle(params).gamma_ph
        c_00_sq = _c_ii_squared(0, params.E_J_kelvin, params.E_C_kelvin)
        c_11_sq = _c_ii_squared(1, params.E_J_kelvin, params.E_C_kelvin)
        assert gamma_ph[1, 1] / gamma_ph[0, 0] == pytest.approx(c_11_sq / c_00_sq, rel=1e-12)
        assert gamma_ph[1, 1] == pytest.approx(267.805, rel=1e-5)


# ═══════════════════════════════════════════════════════════════════════
#  Fermi-level DOS unit guards (eV⁻¹ m⁻³ vs J⁻¹ m⁻³)
# ═══════════════════════════════════════════════════════════════════════


class TestDensityOfStatesUnitGuards:
    @pytest.mark.parametrize("rho_F", [1.74e28, 5.4e28, 3.8e28])
    def test_accepts_shipped_material_values(self, rho_F: float) -> None:
        assert validate_rho_F_eV(rho_F, allow_zero=False) == rho_F

    @pytest.mark.parametrize("rho_F", [1.09e47, 1.0e60])
    def test_rejects_si_per_joule_rho_f(self, rho_F: float) -> None:
        # Al's DOS in J⁻¹ m⁻³ is 1.09e47; rho_F cancels out of every
        # dimensionless observable, so only the reported absolute n_qp
        # would move — by 6.24e18, silently.
        with pytest.raises(ValueError, match="implausibly large"):
            validate_rho_F_eV(rho_F, allow_zero=False)
        with pytest.raises(ValueError, match="implausibly large"):
            Material(name="X", Delta_0=200.0, T_c=1.0, tau_0=1.0, rho_F=rho_F)

    def test_drive_rejects_ev_based_nu_0(self) -> None:
        # The mirror mistake: Material.rho_F pasted into the SI-unit drive
        # field shrinks N_CP and inflates every Γ̄ = Γ̃/N_CP by 6.24e18.
        with pytest.raises(ValueError, match="outside the physical window"):
            M25PhotonDrive(
                omega_nu_kelvin=5.0,
                Gamma_nu_scale_Hz=1.0,
                nu_0_per_J_per_m3=1.74e28,
                volume_m3=1e-18,
            )

    def test_drive_accepts_the_m25_value(self) -> None:
        assert _fig3a_drive().nu_0_per_J_per_m3 == 0.73e47


# ═══════════════════════════════════════════════════════════════════════
#  Material.tau_s / tau_r are accepted but inert
# ═══════════════════════════════════════════════════════════════════════


class TestInertPerChannelTimes:
    def test_warns_when_a_supplied_value_differs_from_tau_0(self) -> None:
        with pytest.warns(UserWarning, match="tau_r=4380.0 is ignored"):
            material = Material(
                name="trapped", Delta_0=180.0, T_c=1.2, tau_0=438.0, tau_r=4380.0
            )
        # The value is still carried, so no existing reader breaks.
        assert material.tau_r == 4380.0

    def test_silent_when_omitted_or_equal_to_tau_0(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            omitted = Material(name="plain", Delta_0=180.0, T_c=1.2, tau_0=438.0)
            equal = Material(
                name="redundant", Delta_0=180.0, T_c=1.2, tau_0=438.0,
                tau_s=438.0, tau_r=438.0,
            )
        assert omitted.tau_s == omitted.tau_r == 438.0
        assert equal.tau_s == equal.tau_r == 438.0
