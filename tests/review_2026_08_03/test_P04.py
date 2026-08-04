# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Regression pins for the 2026-08-03 review fixes in packet P04.

Every fix in P04 is documentation-only; these tests pin the behaviour the
corrected docstrings now describe, so a later "fix" that silently changes a
number instead of the prose fails here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.services.rate_equation import (
    M25Coefficients,
    _default_seed_grid,
    _rate_equation_terms,
    _source_scaled_residual_tolerances,
    analytic_low_T_seed,
    crossover_temperature_kelvin,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)

from tests.services.test_rate_equation_note_v import _fig3a_drive, _fig3a_params

_H_OVER_KB = 4.799243e-11   # K / Hz
_DELTA_R_GHZ = 49.0
_GAMMA_PH_00_HZ = 300.0
_R_RLT_HZ = 6.25e6


def _fig3a_coefficients(T_kelvin: float) -> M25Coefficients:
    """Fig 3a caption bundle, drive recalibrated to Γ^ph_00 = 300 Hz at T."""
    params = _fig3a_params(T_kelvin=T_kelvin)
    drive = _fig3a_drive()
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, drive, _GAMMA_PH_00_HZ,
    )
    return coefficients_from_physical_parameters_with_photon_drive(
        params, replace(drive, Gamma_nu_scale_Hz=scale),
    )


def _opaque_bundle() -> M25Coefficients:
    """Nondegenerate opaque bundle with a recoverable ee Boltzmann ratio."""
    return M25Coefficients(
        gammas_L=np.array([[1.0e3, 2.0e2], [3.0e2, 4.0e3]]),
        gammas_Rgt=np.array([[5.0e3, 1.0e2], [2.0e2, 6.0e3]]),
        gammas_Rlt=np.array([[0.0, 0.0], [7.0e3, 0.0]]),
        gamma_ee=np.array([[0.0, 1.0e-6], [1.0e-5, 0.0]]),
        gamma_ph=np.array([[3.0e-7, 1.0e-7], [1.0e-7, 3.0e-7]]),
        r_L=6.25e6,
        r_Rgt=6.25e6,
        r_Rlt=6.25e6,
        r_cross=1.0e6,
        g_L=1.0e-8,
        g_Rgt=2.0e-8,
        g_Rlt=3.0e-8,
        tau_R_inv=1.0e6,
        tau_E_inv=1.0e2,
        xi=0.1,
        delta=0.9,
        cooper_pair_number_R=1.6e10,
    )


# ─────────────────────────────────────────────────────────────────────
#  crossover_temperature_kelvin — Eq. 8 photon input (docstring line 80)
# ─────────────────────────────────────────────────────────────────────


def test_eq8_photon_input_is_the_population_weighted_total_not_one_element() -> None:
    """``g^ph_R`` is ``Σ_j Γ^ph_{0j} / N_CP(R)``, not ``Γ^ph_{01} / N_CP(R)``.

    Pins the corrected ``crossover_temperature_kelvin`` docstring: the caption
    value 300 Hz is the ``00`` element (the calibration handle), and the Eq. 8
    input is the builder's per-state photon total.
    """
    coefs = _fig3a_coefficients(0.080)
    N_CP = coefs.cooper_pair_number_R

    assert float(coefs.gamma_ph[0, 0]) == pytest.approx(_GAMMA_PH_00_HZ, rel=1e-12)

    g_ph_R = float(coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0])
    assert g_ph_R == pytest.approx(float(coefs.gamma_ph[0, :].sum()) / N_CP, rel=1e-12)

    # The 01 element alone is a different, much smaller number.
    g_from_01_element = float(coefs.gamma_ph[0, 1]) / N_CP
    assert g_ph_R > 4.0 * g_from_01_element

    Delta_R_kelvin = _DELTA_R_GHZ * 1e9 * _H_OVER_KB
    T_bar_correct = crossover_temperature_kelvin(
        Delta_R_kelvin=Delta_R_kelvin,
        r_Rlt_rate_Hz=_R_RLT_HZ,
        g_photon_R_rate_Hz=g_ph_R,
    )
    T_bar_from_01_element = crossover_temperature_kelvin(
        Delta_R_kelvin=Delta_R_kelvin,
        r_Rlt_rate_Hz=_R_RLT_HZ,
        g_photon_R_rate_Hz=g_from_01_element,
    )
    # M25 rates Eq. 8 at "a few percent"; the wrong input costs about that.
    assert abs(T_bar_from_01_element / T_bar_correct - 1.0) > 0.04


# ─────────────────────────────────────────────────────────────────────
#  analytic_low_T_seed — T_kelvin is validated but not used
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["large_asymmetry", "small_asymmetry"])
def test_analytic_seed_thermal_factor_comes_from_the_bundle(case: str) -> None:
    """The Boltzmann factor is the bundle's ee ratio, not ``e^{-ω_10/T}``."""
    coefs = _opaque_bundle()
    seeds = [
        analytic_low_T_seed(coefs, T, case=case)
        for T in (1e-2, 0.5, 1e2, 1e6)
    ]
    for other in seeds[1:]:
        assert np.array_equal(seeds[0], other)

    ee_01 = float(coefs.gamma_ee[0, 1])
    ee_10 = float(coefs.gamma_ee[1, 0])
    boltzmann = ee_01 / ee_10
    expected_p_1 = (
        float(coefs.gamma_ph[0, 1]) / (ee_10 * (1.0 + boltzmann))
        + boltzmann / (1.0 + boltzmann)
    )
    assert seeds[0][0] == pytest.approx(expected_p_1, rel=1e-12)


@pytest.mark.parametrize("T_kelvin", [0.0, -1.0, float("nan"), float("inf")])
def test_analytic_seed_still_validates_T_kelvin(T_kelvin: float) -> None:
    """Unused does not mean unchecked — the guard must survive the doc fix."""
    with pytest.raises(ValueError):
        analytic_low_T_seed(_opaque_bundle(), T_kelvin)


# ─────────────────────────────────────────────────────────────────────
#  _default_seed_grid — docstring corrected, numbers deliberately not
# ─────────────────────────────────────────────────────────────────────


def test_default_seed_grid_ratios_are_unchanged() -> None:
    """The 0.02 anchor is documented as low, not converged — and stays 0.02.

    Re-tuning it to a measured root ratio would change which fixed point the
    multi-seed picker reaches and requires a certified regeneration.
    """
    grid = _default_seed_grid()
    assert len(grid) == 24
    for seed in grid:
        assert seed[2] == pytest.approx(0.4 * seed[1], rel=0.0, abs=0.0)
        assert seed[3] == pytest.approx(0.02 * seed[1], rel=0.0, abs=0.0)


# ─────────────────────────────────────────────────────────────────────
#  _source_scaled_residual_tolerances — measured floor headroom
# ─────────────────────────────────────────────────────────────────────


def test_shipped_fig3a_row_tolerances_clear_the_absolute_floor() -> None:
    """At the shipped operating point the 1e-14 Hz floor is inactive.

    Pins the headroom recorded in the tolerance docstring: the gate is a
    relative certificate at M25 Fig 3 parameters, so removing or keeping the
    floor cannot change an accepted result there.
    """
    coefs = _fig3a_coefficients(0.010)
    state = solve_rate_equation_steady_state_multi_seed(coefs)
    y = np.array([state.p_1, state.x_L, state.x_Rgt, state.x_Rlt])

    tolerances = _source_scaled_residual_tolerances(y, coefs, 1e-3)
    rows = _rate_equation_terms(y, coefs)
    term_sums = np.array(
        [sum(abs(float(term)) for term in row) for row in rows],
    )
    row_sources = np.array([
        max(abs(float(rows[0][0])), abs(float(rows[0][1]))),
        abs(float(rows[1][0])),
        abs(float(rows[2][0])),
        abs(float(rows[3][0])),
    ])
    unfloored = 1e-3 * row_sources + 64.0 * np.finfo(float).eps * term_sums

    assert np.array_equal(tolerances, unfloored)
    assert unfloored[1:].min() > 5e-13
