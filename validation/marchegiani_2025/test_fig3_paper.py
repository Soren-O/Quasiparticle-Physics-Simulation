"""Regression test: M25 Fig 3 paper-target run matches the pinned baseline.

Slow-marked — the two-panel sweep at 29 temperatures with multi-seed
hybr solves takes a few minutes. Opt in with ``pytest -m slow``.

Tolerance is qualitative-trend (rtol=2e-2) because the existing M25
Fig 3 reproduction sits at ~2% paper agreement and the prev-T
continuation seeding does not algorithmically smooth high-T
bifurcation kinks (see :mod:`fig3_paper` module docstring).

First-time generation::

    python -m validation.marchegiani_2025.fig3_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025.fig3_paper import (
    PANEL_A_OMEGA_LR_GHZ,
    PANEL_B_OMEGA_LR_GHZ,
    baseline_path_a,
    baseline_path_b,
    read_baseline,
    run,
)

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip(
            "Baseline not found. Generate with: "
            "python -m validation.marchegiani_2025.fig3_paper"
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
        # Densities — qualitative-trend rtol matching the ~2% paper
        # agreement of the underlying multi-seed solve. Tighter than
        # paper but looser than per-x machine pin so we don't trip on
        # scipy hybr iteration noise.
        np.testing.assert_allclose(
            actual.x_L, expected.x_L, rtol=2e-2,
            err_msg=f"{panel_name}: x_L drifted",
        )
        np.testing.assert_allclose(
            actual.x_Rgt, expected.x_Rgt, rtol=2e-2,
            err_msg=f"{panel_name}: x_Rgt drifted",
        )
        np.testing.assert_allclose(
            actual.x_Rlt, expected.x_Rlt, rtol=2e-2,
            err_msg=f"{panel_name}: x_Rlt drifted",
        )
        np.testing.assert_allclose(
            actual.p_1, expected.p_1, rtol=2e-2,
            err_msg=f"{panel_name}: p_1 drifted",
        )
        # Chemical potentials are derived μ_α = Δ_α + T·log(x_α);
        # absolute tolerance is set by Δ_α (~50 GHz) and the rtol on
        # x_α propagates through T·log(x_α).
        np.testing.assert_allclose(
            actual.mu_L_GHz, expected.mu_L_GHz, atol=0.5, rtol=2e-2,
            err_msg=f"{panel_name}: mu_L_GHz drifted",
        )
        np.testing.assert_allclose(
            actual.mu_Rgt_GHz, expected.mu_Rgt_GHz, atol=0.5, rtol=2e-2,
            err_msg=f"{panel_name}: mu_Rgt_GHz drifted",
        )
        np.testing.assert_allclose(
            actual.mu_Rlt_GHz, expected.mu_Rlt_GHz, atol=0.5, rtol=2e-2,
            err_msg=f"{panel_name}: mu_Rlt_GHz drifted",
        )


def test_panel_omega_LR_values() -> None:
    """Sanity-check the two panels carry the M25 caption ω_LR values."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")
    baseline = read_baseline()
    assert baseline.panel_a.omega_LR_GHz == PANEL_A_OMEGA_LR_GHZ
    assert baseline.panel_b.omega_LR_GHz == PANEL_B_OMEGA_LR_GHZ


def test_chemical_potentials_approach_zero_at_T_bar() -> None:
    """M25 paper text: μ_α → 0 around the Lambert-W crossover T̄."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline()
    for panel in (baseline.panel_a, baseline.panel_b):
        idx_150 = int(np.argmin(np.abs(panel.T_kelvin - 0.150)))
        for mu in (panel.mu_L_GHz, panel.mu_Rgt_GHz, panel.mu_Rlt_GHz):
            assert abs(mu[idx_150]) < 10.0, (
                f"ω_LR={panel.omega_LR_GHz}: |μ| at T≈150 mK "
                f"is {mu[idx_150]:.2g} GHz, expected ≈ 0 (paper M25)"
            )
