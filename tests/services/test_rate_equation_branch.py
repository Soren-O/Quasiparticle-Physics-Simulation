"""Tests for the M25 branch-continuation driver (``solve_rate_equation_branch``).

The physics family used throughout is the M25 Fig 3 caption parameter
set (both the small- and large-asymmetry panels), built inline from
the Note-V coefficient builder — the same inputs the validation
figures sweep, on a coarser grid so the tests stay in the default
gate.

Verified contracts (ticket: continuation-based branch tracking):

* smooth, monotone composite curve on a coarse T grid — no pointwise
  branch flapping (bounded ``|Δ log10 x_L|`` between adjacent points);
* agreement with the multi-seed picker at low temperature;
* the composite reaches the thermal branch at/above the M25 Eq. 8
  Lambert-W crossover ``T̄`` and stays photon-dominated well below it;
* determinism (bit-identical output run-to-run);
* fold handling: the reduced 1-D relocation fallback finds the root
  when seeded far off-center and reports "no root" (the fold signal)
  when the window genuinely contains none.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.services.rate_equation import (
    M25BranchSweep,
    M25Coefficients,
    _relocate_root_1d,
    crossover_temperature_kelvin,
    solve_rate_equation_branch,
    solve_rate_equation_steady_state_multi_seed,
    thermal_equilibrium_seed,
)
from qpsim.services.rate_equation_coefficients import (
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)
from scipy.special import erf

from tests.services.test_rate_equation_note_v import _fig3a_drive, _fig3a_params

_H_OVER_KB = 4.799243e-11   # K / Hz
_DELTA_R_GHZ = 49.0
_OMEGA_10_GHZ = 5.5
_GAMMA_PH_00_HZ = 300.0


def _coefficients_at(omega_LR_GHz: float, T_kelvin: float) -> M25Coefficients:
    """M25 Fig 3 caption coefficients with per-T drive recalibration.

    The Fig 3a caption bundle is owned by
    :mod:`tests.services.test_rate_equation_note_v`
    (``_fig3a_params`` / ``_fig3a_drive``); only the panel-dependent
    pieces — ``Δ_L``, the ``R_T`` gap-average factor, and the sweep
    temperature — are overridden here.
    """
    Delta_L_GHz = _DELTA_R_GHZ + omega_LR_GHz
    params = _fig3a_params(
        Delta_L_kelvin=Delta_L_GHz * 1e9 * _H_OVER_KB,
        T_kelvin=T_kelvin,
        R_T_Hz=8.0 * 14.5e9 * ((Delta_L_GHz + _DELTA_R_GHZ) / 2.0 / Delta_L_GHz),
    )
    drive_template = _fig3a_drive()
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, drive_template, _GAMMA_PH_00_HZ,
    )
    return coefficients_from_physical_parameters_with_photon_drive(
        params, replace(drive_template, Gamma_nu_scale_Hz=scale),
    )


def _run_sweep(omega_LR_GHz: float, T_grid: np.ndarray) -> M25BranchSweep:
    Delta_L_K = (_DELTA_R_GHZ + omega_LR_GHz) * 1e9 * _H_OVER_KB
    Delta_R_K = _DELTA_R_GHZ * 1e9 * _H_OVER_KB
    case = "large_asymmetry" if omega_LR_GHz >= 1.0 else "small_asymmetry"
    return solve_rate_equation_branch(
        lambda T: _coefficients_at(omega_LR_GHz, T),
        T_grid,
        photon_seed_case=case,
        thermal_seed=thermal_equilibrium_seed(
            Delta_L_kelvin=Delta_L_K,
            Delta_R_kelvin=Delta_R_K,
            omega_10_kelvin=_OMEGA_10_GHZ * 1e9 * _H_OVER_KB,
            T_kelvin=float(T_grid[-1]),
        ),
    )


def _mu_L_over_Delta_L(sweep: M25BranchSweep, omega_LR_GHz: float) -> np.ndarray:
    """Paper-exact μ_L inversion (arXiv Eq. 11 / SI Eq. S3)."""
    Delta_L_K = (_DELTA_R_GHZ + omega_LR_GHz) * 1e9 * _H_OVER_KB
    T = sweep.T_kelvin
    x_L = np.array([s.x_L for s in sweep.states])
    return (Delta_L_K + T * np.log(x_L * np.sqrt(Delta_L_K / (2.0 * np.pi * T)))) / Delta_L_K


_COARSE_GRID = np.linspace(0.010, 0.150, 8)


class TestCompositeCurve:
    @pytest.mark.parametrize(
        ("keyword", "value"),
        [
            ("merge_rtol", float("nan")),
            ("jump_tol_decades", float("nan")),
            ("T_exchange_hint_kelvin", float("nan")),
            ("max_step_bisections", True),
            ("max_function_evaluations", 0),
        ],
    )
    def test_rejects_invalid_branch_controls(self, keyword: str, value: object) -> None:
        with pytest.raises(ValueError, match=keyword):
            solve_rate_equation_branch(
                lambda T: _coefficients_at(0.5, T),
                np.array([0.02, 0.03]),
                **{keyword: value},
            )

    @pytest.mark.parametrize(
        "grid", [np.array([0.02, np.nan]), np.array([0.0, 0.02])],
    )
    def test_rejects_nonfinite_or_nonpositive_temperature_grid(
        self, grid: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="finite positive"):
            solve_rate_equation_branch(lambda T: _coefficients_at(0.5, T), grid)

    @pytest.mark.parametrize(
        "grid",
        [
            np.array([0.02 + 0.0j, 0.03 + 0.0j]),
            np.array([0.02 + 1e-6j, 0.03 + 0.0j]),
            np.array([complex(0.02, float("nan")), 0.03 + 0.0j]),
        ],
    )
    def test_rejects_complex_temperature_grid_before_float_cast(
        self,
        grid: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            solve_rate_equation_branch(lambda T: _coefficients_at(0.5, T), grid)

    def test_fig3a_no_hint_sweep_survives_old_death_valley(self) -> None:
        # Regression for the coupled F1/F4 failure: on this exact grid the
        # old global-min tolerance killed the high-to-low pass at its first
        # point, while the 1-D fallback let only the opposite pass through.
        # With no exchange hint the driver then (correctly) refused to invent
        # a branch choice. Row-wise gates restore both passes, so every point
        # is independently merged without weakening the no-hint safeguard.
        T_grid = np.linspace(0.020, 0.100, 9)
        sweep = solve_rate_equation_branch(
            lambda T: _coefficients_at(0.5, T),
            T_grid,
            photon_seed_case="small_asymmetry",
            max_step_bisections=2,
        )
        assert sweep.branch_labels == ("merged",) * len(T_grid)
        assert all(state is not None for state in sweep.photon_states)
        assert all(state is not None for state in sweep.thermal_states)
        assert sweep.T_exchange_kelvin is None

    @pytest.mark.parametrize("omega_LR_GHz", [0.5, 5.0])
    def test_smooth_composite_no_branch_flapping(self, omega_LR_GHz: float) -> None:
        sweep = _run_sweep(omega_LR_GHz, _COARSE_GRID)
        # With the Γ̄-normalized density equations the M25 root is
        # unique at every temperature: both directional passes must
        # converge to the same fixed point everywhere.
        assert set(sweep.branch_labels) == {"merged"}
        assert sweep.T_exchange_kelvin is None
        # No pointwise branch flapping: adjacent densities move by
        # well under one decade on this coarse (20 mK step) grid.
        x_L = np.array([s.x_L for s in sweep.states])
        dlog = np.abs(np.diff(np.log10(x_L)))
        assert float(dlog.max()) < 0.5, f"x_L flapping: max Δlog10 = {dlog.max():.3f}"
        # True fixed points only: residuals far below the ~1e-8 Hz
        # photon-generation source scale.
        assert max(s.residual_inf_norm for s in sweep.states) < 1e-10

    @pytest.mark.parametrize("omega_LR_GHz", [0.5, 5.0])
    def test_mu_L_monotone_decreasing(self, omega_LR_GHz: float) -> None:
        sweep = _run_sweep(omega_LR_GHz, _COARSE_GRID)
        mu = _mu_L_over_Delta_L(sweep, omega_LR_GHz)
        assert np.all(np.diff(mu) < 0.0), f"μ_L not monotone: {mu}"

    def test_deterministic_rerun_bit_identical(self) -> None:
        a = _run_sweep(0.5, _COARSE_GRID)
        b = _run_sweep(0.5, _COARSE_GRID)
        for sa, sb in zip(a.states, b.states, strict=True):
            assert sa.p_1 == sb.p_1
            assert sa.x_L == sb.x_L
            assert sa.x_Rgt == sb.x_Rgt
            assert sa.x_Rlt == sb.x_Rlt


class TestLowTemperatureAnchors:
    def test_agrees_with_multi_seed_picker_at_low_T(self) -> None:
        # The multi-seed picker (min_residual) and the continuation
        # driver must land on the same fixed point at low T.
        T_grid = np.array([0.010, 0.020])
        sweep = _run_sweep(0.5, T_grid)
        for i, T in enumerate(T_grid):
            multi = solve_rate_equation_steady_state_multi_seed(
                _coefficients_at(0.5, float(T)),
            )
            s = sweep.states[i]
            assert s.x_L == pytest.approx(multi.x_L, rel=1e-6)
            assert s.x_Rgt == pytest.approx(multi.x_Rgt, rel=1e-6)
            assert s.x_Rlt == pytest.approx(multi.x_Rlt, rel=1e-6)
            assert s.p_1 == pytest.approx(multi.p_1, rel=1e-6)

    def test_paper_figure_values_small_asymmetry(self) -> None:
        # Paper Fig 3a (arXiv 2408.17218): the merged μ_α/Δ_L curve
        # runs ≈ 0.94 at 10 mK, ≈ 0.87 at 20 mK, approximately
        # linearly to zero at T̄ ≈ 0.146 K. (The historical anchor
        # readings 0.95/0.91/0.87/0.81 at 10/20/30/40 mK were pixel
        # misreads consistent only with a pseudo-root of the
        # unnormalized density equations — see STATUS.md.)
        T_grid = np.linspace(0.010, 0.150, 15)
        sweep = _run_sweep(0.5, T_grid)
        mu = _mu_L_over_Delta_L(sweep, 0.5)
        assert mu[0] == pytest.approx(0.938, abs=0.010)     # 10 mK
        assert mu[1] == pytest.approx(0.872, abs=0.010)     # 20 mK
        # Approximately linear from 10 mK down to the crossover: the
        # best-fit line over T ≤ 130 mK deviates < 0.02·Δ_L anywhere.
        mask = T_grid <= 0.131
        coeffs = np.polyfit(T_grid[mask], mu[mask], 1)
        dev = np.abs(np.polyval(coeffs, T_grid[mask]) - mu[mask])
        assert float(dev.max()) < 0.02
        # μ → ~0 at the top of the sweep (T ≈ T̄).
        assert abs(mu[-1]) < 0.03


class TestThermalBranchExchange:
    def test_transition_brackets_T_bar(self) -> None:
        # The a-priori crossover estimate (M25 Eq. 8) computed from
        # the photon generation rate sits at ~146 mK for the Fig 3
        # caption parameters.
        coefs = _coefficients_at(0.5, 0.080)
        g_ph_R = float(coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0])
        T_bar = crossover_temperature_kelvin(
            Delta_R_kelvin=_DELTA_R_GHZ * 1e9 * _H_OVER_KB,
            r_Rlt_rate_Hz=6.25e6,
            g_photon_R_rate_Hz=g_ph_R,
        )
        assert 0.13 < T_bar < 0.16

        T_grid = np.array([0.050, 0.100, 0.140, 0.150])
        sweep = _run_sweep(0.5, T_grid)
        Delta_R_K = _DELTA_R_GHZ * 1e9 * _H_OVER_KB
        omega_LR_K = 0.5e9 * _H_OVER_KB

        def x_Rlt_eq(T: float) -> float:
            return float(
                np.sqrt(2.0 * np.pi * T / Delta_R_K) * np.exp(-Delta_R_K / T)
                * erf(np.sqrt(omega_LR_K / T))
            )

        # Below T̄ the photon drive dominates: x_R< sits orders of
        # magnitude above thermal equilibrium.
        assert sweep.states[0].x_Rlt > 1e3 * x_Rlt_eq(0.050)
        # At/above T̄ (150 mK ≳ T̄ = 146 mK) the composite has reached
        # the thermal branch: x_R< within 50% of thermal equilibrium.
        assert sweep.states[-1].x_Rlt == pytest.approx(x_Rlt_eq(0.150), rel=0.5)
        # And the thermal (downward) pass covered the whole grid,
        # agreeing with the photon pass — the exchange is a smooth
        # crossover, not a discontinuity.
        assert all(s is not None for s in sweep.thermal_states)
        assert set(sweep.branch_labels) == {"merged"}


class TestFoldHandlingMachinery:
    """The reduced 1-D relocation is the driver's fold handler.

    The M25 Fig 3 family has a unique root (no fold to traverse), so
    the fold contract is exercised directly: relocation must find the
    root from a badly off-center window, and must report "no root"
    (the fold signal) for a window that genuinely contains none.
    """

    def test_relocation_finds_root_from_off_center_seed(self) -> None:
        coefs = _coefficients_at(0.5, 0.020)
        # Seed the search two decades below the true root x_L ≈ 5.3e-8
        # (exact S48/S49 tau_R, audit H4) with a deliberately wrong
        # transverse state.
        state = _relocate_root_1d(
            coefs, 5.3e-10, np.array([1e-4, 1e-9, 1e-9]),
            window_decades=6.0, n_samples=37,
            residual_tol_relative=1e-3,
        )
        assert state is not None
        assert state.x_L == pytest.approx(5.3127e-8, rel=1e-3)

    def test_relocation_reports_no_root_outside_window(self) -> None:
        coefs = _coefficients_at(0.5, 0.020)
        # A narrow window far above the unique root contains no sign
        # change: the fold signal is a None return, never a fabricated
        # pseudo-root.
        state = _relocate_root_1d(
            coefs, 1e-3, np.array([1e-3, 1e-4, 1e-4]),
            window_decades=1.0, n_samples=9,
            residual_tol_relative=1e-3,
        )
        assert state is None

    def test_unpolished_reduced_root_must_pass_all_row_gates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        coefs = _toy_coefs()
        monkeypatch.setattr(rate_mod, "_branch_corrector", lambda *args: None)
        # x_L still brackets the exact reduced root at x_L=1, but this fake
        # transverse state leaves the R> and R< rows at -3 Hz. Before the
        # fix, the fallback returned it solely because the x_L row was zero.
        monkeypatch.setattr(
            rate_mod, "_transverse_solve",
            lambda *args: np.array([0.5, 2.0, 2.0]),
        )
        state = rate_mod._relocate_root_1d(
            coefs, 1.0, np.array([0.5, 2.0, 2.0]),
            window_decades=2.0, n_samples=9,
            residual_tol_relative=1e-3,
        )
        assert state is None

    def test_source_valid_unpolished_reduced_root_is_retained(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.services import rate_equation as rate_mod

        coefs = _toy_coefs()
        monkeypatch.setattr(rate_mod, "_branch_corrector", lambda *args: None)
        monkeypatch.setattr(
            rate_mod, "_transverse_solve",
            lambda *args: np.array([1.0 / 11.0, 1.0, 1.0]),
        )
        state = rate_mod._relocate_root_1d(
            coefs, 1.0, np.array([1.0 / 11.0, 1.0, 1.0]),
            window_decades=2.0, n_samples=9,
            residual_tol_relative=1e-3,
        )
        assert state is not None
        assert state.residual_inf_norm < 1e-14


class TestInputValidation:
    def test_decreasing_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            solve_rate_equation_branch(
                lambda T: _coefficients_at(0.5, T),
                np.array([0.020, 0.010]),
            )

    def test_empty_grid_rejected(self) -> None:
        with pytest.raises(ValueError, match="nonempty"):
            solve_rate_equation_branch(
                lambda T: _coefficients_at(0.5, T),
                np.array([]),
            )


def _toy_coefs() -> M25Coefficients:
    """T-independent toy bundle with a unique, cheap fixed point.

    ``x_α = 1`` in every band (g = r = 1, no tunneling), qubit at ee
    detailed balance — both continuation passes converge everywhere,
    so the merge/exchange logic can be driven synthetically by
    monkeypatching ``_states_agree``.
    """
    return M25Coefficients(
        gammas_L=np.zeros((2, 2)),
        gammas_Rgt=np.zeros((2, 2)),
        gammas_Rlt=np.zeros((2, 2)),
        gamma_ee=np.array([[0.0, 1e-6], [1e-5, 0.0]]),
        gamma_ph=np.zeros((2, 2)),
        r_L=1.0, r_Rgt=1.0, r_Rlt=1.0, r_cross=0.0,
        g_L=1.0, g_Rgt=1.0, g_Rlt=1.0,
        tau_R_inv=0.0, tau_E_inv=0.0,
        xi=0.0, delta=0.5,
    )


class TestExchangeSelectionContiguity:
    """Hint-less exchange selection: contiguous-suffix validation.

    Without ``T_exchange_hint_kelvin`` the driver infers the exchange
    point as the lowest merged grid temperature — only meaningful when
    the merged points form a contiguous suffix of the grid. The merge
    pattern is forced by monkeypatching ``_states_agree`` (the only
    consumer inside ``solve_rate_equation_branch``), which is called
    once per grid point in grid order.
    """

    def _run_with_pattern(
        self, monkeypatch: pytest.MonkeyPatch, pattern: list[bool],
    ) -> M25BranchSweep:
        from qpsim.services import rate_equation as rate_mod

        agreement = iter(pattern)
        monkeypatch.setattr(
            rate_mod, "_states_agree",
            lambda a, b, rtol: next(agreement),
        )
        coefs = _toy_coefs()
        T_grid = np.array([0.010, 0.020, 0.030])
        return rate_mod.solve_rate_equation_branch(lambda T: coefs, T_grid)

    def test_non_contiguous_merge_pattern_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        with pytest.raises(RuntimeError, match="contiguous suffix"):
            self._run_with_pattern(monkeypatch, [True, False, True])

    def test_contiguous_suffix_selects_first_merged_temperature(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sweep = self._run_with_pattern(monkeypatch, [False, True, True])
        assert sweep.T_exchange_kelvin == pytest.approx(0.020)
        assert sweep.branch_labels == ("photon", "merged", "merged")


class TestSweepCoefficientsExposure:
    def test_sweep_exposes_grid_ordered_coefficients_from_cache(self) -> None:
        # The driver exposes the per-T coefficient bundles it solved
        # against, in grid order, from its internal cache — the
        # builder runs exactly once per grid temperature even though
        # both passes and the exposure all consume it.
        coefs = _toy_coefs()
        built: list[float] = []

        def coefficients_at(T: float) -> M25Coefficients:
            built.append(T)
            return coefs

        T_grid = np.array([0.010, 0.020])
        sweep = solve_rate_equation_branch(coefficients_at, T_grid)
        assert len(sweep.coefficients) == T_grid.size
        assert all(c is coefs for c in sweep.coefficients)
        assert built == [0.010, 0.020]
