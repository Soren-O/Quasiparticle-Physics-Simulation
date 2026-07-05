"""Regression test: M25 Fig 3 paper-target run matches the pinned baseline.

Fast since the branch-continuation driver landed: the two-panel sweep
at 29 temperatures is a few seconds of warm-started Newton solves on
the well-conditioned (Γ̄-normalized) moment system, so this runs in
the default gate.

Tolerance is platform-stamped (see ``_robust.assert_pinned_match``):
strict rtol=1e-6 on the generating platform (the driver is
deterministic and the tracked root is unique with residuals
~1e-12 Hz); rtol=1e-3 elsewhere (rounding-level cross-platform
scatter only, post Γ̄-normalization).

First-time generation::

    python -m validation.marchegiani_2025.fig3_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import assert_pinned_match
from validation.marchegiani_2025.fig3_paper import (
    PANEL_A_OMEGA_LR_GHZ,
    PANEL_B_OMEGA_LR_GHZ,
    baseline_path_a,
    baseline_path_b,
    read_baseline,
    run,
)


def test_matches_pinned_baseline() -> None:
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip(
            "Baseline not found. Generate with: "
            "python -m validation.marchegiani_2025.fig3_paper"
        )

    baseline = read_baseline()
    result = run()

    for panel_name, path, expected, actual in (
        ("panel_a", baseline_path_a(), baseline.panel_a, result.panel_a),
        ("panel_b", baseline_path_b(), baseline.panel_b, result.panel_b),
    ):
        # Temperatures are deterministic (np.linspace).
        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"{panel_name}: T_kelvin drifted",
        )
        # Densities — platform-stamped pins (strict on the generating
        # platform, loose-but-running elsewhere).
        assert_pinned_match(
            actual.x_L, expected.x_L, f"{panel_name}: x_L",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.x_Rgt, expected.x_Rgt, f"{panel_name}: x_Rgt",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.x_Rlt, expected.x_Rlt, f"{panel_name}: x_Rlt",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.p_1, expected.p_1, f"{panel_name}: p_1",
            baseline_path=path,
        )
        # Chemical potentials are derived from x_α via the SI Eqs.
        # S2–S5 inversions; the log compresses the density rtol, and
        # the atol floor covers the numerical noise near the μ → 0
        # equilibrium attractor at the top of the sweep.
        assert_pinned_match(
            actual.mu_L_GHz, expected.mu_L_GHz, f"{panel_name}: mu_L_GHz",
            baseline_path=path, atol=1e-3,
        )
        assert_pinned_match(
            actual.mu_Rgt_GHz, expected.mu_Rgt_GHz,
            f"{panel_name}: mu_Rgt_GHz",
            baseline_path=path, atol=1e-3,
        )
        assert_pinned_match(
            actual.mu_Rlt_GHz, expected.mu_Rlt_GHz,
            f"{panel_name}: mu_Rlt_GHz",
            baseline_path=path, atol=1e-3,
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
