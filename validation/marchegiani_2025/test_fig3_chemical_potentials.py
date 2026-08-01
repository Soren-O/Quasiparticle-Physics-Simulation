"""Regression test: M25 Fig 3 chemical-potential temperature sweep
matches the pinned baseline CSVs (panels a and b)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import assert_pinned_match
from validation.marchegiani_2025.fig3_chemical_potentials import (
    DELTA_R_OVER_H_GHZ,
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

    for panel_name, path, expected, actual in (
        ("panel_a", baseline_path_a(), baseline.panel_a, result.panel_a),
        ("panel_b", baseline_path_b(), baseline.panel_b, result.panel_b),
    ):
        # Temperatures are deterministic (np.linspace).
        np.testing.assert_allclose(
            actual.T_kelvin,
            expected.T_kelvin,
            rtol=0.0,
            atol=1e-14,
            err_msg=f"{panel_name}: T_kelvin drifted",
        )
        # Platform-stamped pins (strict rtol=1e-6 on the generating
        # platform, rtol=1e-3 elsewhere): the branch-continuation
        # driver on the Γ̄-normalized density equations tracks a
        # unique root, so the historical branch-selection noise the
        # old majority/median comparison tolerated is gone — see
        # validation/marchegiani_2025/_robust.py.
        assert_pinned_match(
            actual.x_L,
            expected.x_L,
            f"{panel_name}: x_L",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.x_Rgt,
            expected.x_Rgt,
            f"{panel_name}: x_Rgt",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.x_Rlt,
            expected.x_Rlt,
            f"{panel_name}: x_Rlt",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.p_1,
            expected.p_1,
            f"{panel_name}: p_1",
            baseline_path=path,
        )
        # Chemical potentials μ_α are derived from x_α via the SI
        # Eqs. S2–S5 inversions; the log compresses the density rtol,
        # and the atol floor covers the numerical noise near the
        # μ → 0 equilibrium attractor at high T.
        assert_pinned_match(
            actual.mu_L_GHz,
            expected.mu_L_GHz,
            f"{panel_name}: mu_L_GHz",
            baseline_path=path,
            atol=1e-3,
        )
        assert_pinned_match(
            actual.mu_Rgt_GHz,
            expected.mu_Rgt_GHz,
            f"{panel_name}: mu_Rgt_GHz",
            baseline_path=path,
            atol=1e-3,
        )
        assert_pinned_match(
            actual.mu_Rlt_GHz,
            expected.mu_Rlt_GHz,
            f"{panel_name}: mu_Rlt_GHz",
            baseline_path=path,
            atol=1e-3,
        )


def test_panel_a_low_T_chemical_potential_manual_broad_anchor() -> None:
    """Broad manual figure-reading anchor, not digitized parity.

    At T = 20 mK with ω_LR = 0.5 GHz, a manual reading of paper Fig. 3a
    (arXiv:2408.17218) places the merged μ_α/Δ_L curve near 0.87, or
    μ_L ≈ 43 GHz. The unique Γ̄-normalized root gives 0.872·Δ_L; retain a
    deliberately broad ±1 GHz sanity band.
    """
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
    linearly), reaching zero ... around T ≳ 150 mK'. Verify strict
    per-step decrease on the pinned 5 mK temperature grid."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline()
    for panel in (baseline.panel_a, baseline.panel_b):
        for label, mu in (
            ("mu_L", panel.mu_L_GHz),
            ("mu_Rgt", panel.mu_Rgt_GHz),
            ("mu_Rlt", panel.mu_Rlt_GHz),
        ):
            increases = np.flatnonzero(np.diff(mu) >= 0.0)
            assert increases.size == 0, (
                f"ω_LR={panel.omega_LR_GHz}: {label} did not strictly "
                f"decrease at temperature-step indices {increases.tolist()}"
            )
        # The top-of-sweep chemical potentials are small compared with
        # the gap. This is a broad endpoint sanity check, not an
        # independently located crossover.
        idx_150 = int(np.argmin(np.abs(panel.T_kelvin - 0.150)))
        Delta_L_GHz = DELTA_R_OVER_H_GHZ + panel.omega_LR_GHz
        for mu in (panel.mu_L_GHz, panel.mu_Rgt_GHz, panel.mu_Rlt_GHz):
            assert abs(mu[idx_150]) < 0.03 * Delta_L_GHz, (
                f"ω_LR={panel.omega_LR_GHz}: |μ|/Δ_L at 150 mK is "
                f"{abs(mu[idx_150]) / Delta_L_GHz:.3g}; expected < 3%"
            )
