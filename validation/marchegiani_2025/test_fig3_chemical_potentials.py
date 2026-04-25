"""Regression test: M25 Fig 3 chemical-potential temperature sweep
matches the pinned baseline CSVs (panels a and b)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025.fig3_chemical_potentials import (
    baseline_path_a,
    baseline_path_b,
    read_baseline,
    run,
)


def test_matches_pinned_baseline() -> None:
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip(
            "Baseline not found. Generate with: "
            "python -m validation.marchegiani_2025.fig3_chemical_potentials"
        )

    baseline = read_baseline()
    result = run()

    for panel_name, expected, actual in (
        ("panel_a", baseline.panel_a, result.panel_a),
        ("panel_b", baseline.panel_b, result.panel_b),
    ):
        # Temperatures are deterministic (np.linspace).
        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"{panel_name}: T_kelvin drifted",
        )
        # Densities and qubit excited population — moderately tight
        # but tolerant of solver-iteration noise across scipy versions.
        # The branch-selector ("max x_L over all converged seeds")
        # makes the choice of fixed point reproducible; the per-x
        # numerical accuracy is set by the LM xtol/ftol stopping
        # criteria.
        np.testing.assert_allclose(
            actual.x_L, expected.x_L, rtol=1e-3,
            err_msg=f"{panel_name}: x_L drifted",
        )
        np.testing.assert_allclose(
            actual.x_Rgt, expected.x_Rgt, rtol=1e-3,
            err_msg=f"{panel_name}: x_Rgt drifted",
        )
        np.testing.assert_allclose(
            actual.x_Rlt, expected.x_Rlt, rtol=1e-3,
            err_msg=f"{panel_name}: x_Rlt drifted",
        )
        np.testing.assert_allclose(
            actual.p_1, expected.p_1, rtol=1e-3,
            err_msg=f"{panel_name}: p_1 drifted",
        )
        # Chemical potentials are derived μ_α = Δ_α + T·log(x_α);
        # tight rtol because Δ_α is large relative to T·log(x).
        np.testing.assert_allclose(
            actual.mu_L_GHz, expected.mu_L_GHz, atol=0.1, rtol=1e-3,
            err_msg=f"{panel_name}: mu_L_GHz drifted",
        )
        np.testing.assert_allclose(
            actual.mu_Rgt_GHz, expected.mu_Rgt_GHz, atol=0.1, rtol=1e-3,
            err_msg=f"{panel_name}: mu_Rgt_GHz drifted",
        )
        np.testing.assert_allclose(
            actual.mu_Rlt_GHz, expected.mu_Rlt_GHz, atol=0.1, rtol=1e-3,
            err_msg=f"{panel_name}: mu_Rlt_GHz drifted",
        )


def test_panel_a_low_T_chemical_potential_matches_paper() -> None:
    """At T = 20 mK with ω_LR = 0.5 GHz the M25 paper Fig 3a places
    μ_L just below Δ_L (~44 GHz). The driver's max-x_L branch picker
    can land on a slightly different x fixed point than the M25
    junction's default-seed picker (multi-stability in the moment
    system), but both yield μ_L within ~5% of the paper value."""
    if not baseline_path_a().exists():
        pytest.skip("Baseline missing; see test_matches_pinned_baseline.")

    baseline = read_baseline()
    panel_a = baseline.panel_a
    idx = int(np.argmin(np.abs(panel_a.T_kelvin - 0.020)))
    np.testing.assert_allclose(panel_a.T_kelvin[idx], 0.020, atol=1e-9)
    # Paper Fig 3a places μ_L just below Δ_L = 49.5 GHz at T=20 mK.
    # 40-48 GHz is comfortably inside the paper's reading range and
    # rules out the equilibrium branch (μ ≈ 0) or any far-off branch.
    assert 40.0 < panel_a.mu_L_GHz[idx] < 48.0, (
        f"μ_L at T=20mK = {panel_a.mu_L_GHz[idx]:.2f} GHz, "
        "expected 40-48 GHz (paper Fig 3a)"
    )


def test_chemical_potentials_decrease_monotonically() -> None:
    """M25 paper text: μ_α 'decrease monotonically (approximately
    linearly), reaching zero ... around T ≳ 150 mK'. Verify the
    high-T tail of each curve is below the low-T head — this is the
    qualitative architectural pin, not a per-step monotonicity."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline()
    for panel in (baseline.panel_a, baseline.panel_b):
        assert panel.mu_L_GHz[0] > panel.mu_L_GHz[-1], (
            f"ω_LR={panel.omega_LR_GHz}: μ_L did not drop from "
            f"{panel.mu_L_GHz[0]:.2g} (T={panel.T_kelvin[0]:.3f} K) "
            f"to {panel.mu_L_GHz[-1]:.2g} (T={panel.T_kelvin[-1]:.3f} K)"
        )
        assert panel.mu_Rgt_GHz[0] > panel.mu_Rgt_GHz[-1]
        assert panel.mu_Rlt_GHz[0] > panel.mu_Rlt_GHz[-1]
        # Reaches approximately zero at T ≈ 150 mK (the M25 T̄). The
        # 10 GHz tolerance is set by the residual numerical noise
        # near the equilibrium attractor.
        idx_150 = int(np.argmin(np.abs(panel.T_kelvin - 0.150)))
        for mu in (panel.mu_L_GHz, panel.mu_Rgt_GHz, panel.mu_Rlt_GHz):
            assert abs(mu[idx_150]) < 10.0, (
                f"ω_LR={panel.omega_LR_GHz}: |μ| at T≈150 mK "
                f"is {mu[idx_150]:.2g} GHz, expected ≈ 0"
            )
