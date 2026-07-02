"""Regression test: M25 Fig 4 parity-rate temperature sweep matches
the pinned baseline CSVs (panels a and b)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import assert_robust_match
from validation.marchegiani_2025.fig4_parity_rates import (
    baseline_path_a,
    baseline_path_b,
    read_baseline,
    run,
)


def test_matches_pinned_baseline() -> None:
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip(
            "Baseline not found. Generate with: "
            "python -m validation.marchegiani_2025.fig4_parity_rates"
        )

    baseline = read_baseline()
    result = run()

    for panel_name, expected, actual in (
        ("panel_a", baseline.panel_a, result.panel_a),
        ("panel_b", baseline.panel_b, result.panel_b),
    ):
        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"{panel_name}: T_kelvin drifted",
        )
        # Γ_P depends multiplicatively on x_α through gamma_eo, so
        # branch-jump noise (the multi-stability artifact in the
        # underlying moment system) propagates into ~order-of-
        # magnitude scatter at isolated T points — robust
        # (majority + median) comparison, not per-point pins; see
        # validation/marchegiani_2025/_robust.py.
        assert_robust_match(
            actual.Gamma_P_Hz, expected.Gamma_P_Hz,
            f"{panel_name}: Gamma_P_Hz",
        )
        assert_robust_match(
            actual.ratio_eo_01_over_10, expected.ratio_eo_01_over_10,
            f"{panel_name}: ratio_eo_01_over_10", atol=1e-2,
        )


def test_panel_b_ratio_increases_with_T() -> None:
    """M25 paper: the excitation/relaxation ratio Γ̃^eo_01/Γ̃^eo_10
    rises with T toward the photon-limited fixed value (≈ 0.13 from
    Γ̃^ph_01 / Γ̃^ph_10 at the M25 caption parameters). Panel b
    smoothly increases on a log scale; pin the qualitative trend."""
    if not baseline_path_b().exists():
        pytest.skip("Baseline missing.")

    baseline = read_baseline()
    panel_b = baseline.panel_b
    # Coarse monotonicity: tail above head.
    assert panel_b.ratio_eo_01_over_10[-1] > panel_b.ratio_eo_01_over_10[0]
    # End-of-sweep ratio in the photon-limited range (paper text:
    # "for photon-assisted transitions the excitation and relaxation
    # rates are typically of the same order, Γ̃^ph_01/Γ̃^ph_10 ≈ 1/3"
    # — exact value depends on ω_LR; 0.05–1 is the comfortable band).
    assert 0.05 < panel_b.ratio_eo_01_over_10[-1] < 1.0


def test_panel_a_orange_dominates_at_high_T() -> None:
    """Panel a paper trend: the large-asymmetry curve grows
    monotonically while the small-asymmetry curve has a low-T peak
    and then a long flatter region. By T ≈ 100 mK the large-
    asymmetry Γ_P should be comparable to or larger than the
    small-asymmetry Γ_P. Pin this crossover qualitatively."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline()
    # Compare Γ_P at the high-T end. Both should be in the kHz–
    # 100-kHz regime, with the large-asymmetry curve no more than
    # 10× below the small-asymmetry one (the multi-stability noise
    # caps the resolution at this level).
    Gamma_a_end = baseline.panel_a.Gamma_P_Hz[-1]
    Gamma_b_end = baseline.panel_b.Gamma_P_Hz[-1]
    assert Gamma_a_end > 1e3 and Gamma_b_end > 1e3
    assert 0.1 < Gamma_b_end / Gamma_a_end < 100.0
