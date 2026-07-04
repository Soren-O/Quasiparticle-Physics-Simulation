"""Regression test: M25 Fig 3 chemical-potential temperature sweep
matches the pinned baseline CSVs (panels a and b)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import (
    assert_robust_match,
    skip_unless_pinned_here,
)
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
    skip_unless_pinned_here(baseline_path_a(), baseline_path_b())

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
        # Robust (majority + median) comparison retained for cross-
        # scipy-version headroom; with the branch-continuation driver
        # and the Γ̄-normalized (well-conditioned) density equations
        # the historical branch-selection noise this guarded against
        # is gone — see validation/marchegiani_2025/_robust.py.
        assert_robust_match(actual.x_L, expected.x_L, f"{panel_name}: x_L")
        assert_robust_match(
            actual.x_Rgt, expected.x_Rgt, f"{panel_name}: x_Rgt"
        )
        assert_robust_match(
            actual.x_Rlt, expected.x_Rlt, f"{panel_name}: x_Rlt"
        )
        assert_robust_match(actual.p_1, expected.p_1, f"{panel_name}: p_1")
        # Chemical potentials μ_α = Δ_α + T·log(x_α) are the plotted
        # observables and much stiffer than the moments; the atol
        # floor covers the residual numerical noise near the μ → 0
        # equilibrium attractor at high T.
        assert_robust_match(
            actual.mu_L_GHz, expected.mu_L_GHz,
            f"{panel_name}: mu_L_GHz", atol=2.0,
        )
        assert_robust_match(
            actual.mu_Rgt_GHz, expected.mu_Rgt_GHz,
            f"{panel_name}: mu_Rgt_GHz", atol=2.0,
        )
        assert_robust_match(
            actual.mu_Rlt_GHz, expected.mu_Rlt_GHz,
            f"{panel_name}: mu_Rlt_GHz", atol=2.0,
        )


def test_panel_a_low_T_chemical_potential_matches_paper() -> None:
    """At T = 20 mK with ω_LR = 0.5 GHz the paper Fig 3a (arXiv
    2408.17218) shows the merged μ_α/Δ_L curve at ≈ 0.87 — i.e.
    μ_L ≈ 43 GHz. The unique root of the Γ̄-normalized system gives
    0.872·Δ_L; pin a ±1 GHz band around it."""
    if not baseline_path_a().exists():
        pytest.skip("Baseline missing; see test_matches_pinned_baseline.")

    baseline = read_baseline()
    panel_a = baseline.panel_a
    idx = int(np.argmin(np.abs(panel_a.T_kelvin - 0.020)))
    np.testing.assert_allclose(panel_a.T_kelvin[idx], 0.020, atol=1e-9)
    assert 42.2 < panel_a.mu_L_GHz[idx] < 44.2, (
        f"μ_L at T=20mK = {panel_a.mu_L_GHz[idx]:.2f} GHz, "
        "expected ≈ 43.2 GHz (0.87·Δ_L, paper Fig 3a)"
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
