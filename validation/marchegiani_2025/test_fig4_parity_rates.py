"""Regression test: M25 Fig 4 parity-rate temperature sweep matches
the pinned baseline CSVs (panels a and b)."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import assert_pinned_match
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

    baseline = read_baseline(accept_producer_certificate_claims=True)
    result = run()

    for panel_name, path, expected, actual in (
        ("panel_a", baseline_path_a(), baseline.panel_a, result.panel_a),
        ("panel_b", baseline_path_b(), baseline.panel_b, result.panel_b),
    ):
        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"{panel_name}: T_kelvin drifted",
        )
        # Platform-stamped per-point pins: strict rtol=1e-6 on the
        # generating platform and rtol=1e-3 elsewhere; see
        # validation/marchegiani_2025/_robust.py.
        assert_pinned_match(
            actual.Gamma_P_Hz, expected.Gamma_P_Hz,
            f"{panel_name}: Gamma_P_Hz", baseline_path=path,
        )
        assert_pinned_match(
            actual.ratio_eo_01_over_10, expected.ratio_eo_01_over_10,
            f"{panel_name}: ratio_eo_01_over_10", baseline_path=path,
        )


def test_gamma_P_has_no_adjacent_order_of_magnitude_jumps() -> None:
    """Numerical continuity guard on the pinned 5 mK temperature grid."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline(accept_producer_certificate_claims=True)
    for panel in (baseline.panel_a, baseline.panel_b):
        dlog = np.abs(np.diff(np.log10(panel.Gamma_P_Hz)))
        assert float(dlog.max()) < 0.2, (
            f"ω_LR={panel.omega_LR_GHz}: Γ_P jumps by "
            f"10^{dlog.max():.2f} between adjacent T points — "
            "outside the broad continuity bound"
        )


def test_panel_b_ratio_increases_with_T() -> None:
    """M25 paper: the excitation/relaxation ratio Γ̃^eo_01/Γ̃^eo_10
    rises with T toward the photon-limited fixed value (≈ 0.13 from
    Γ̃^ph_01 / Γ̃^ph_10 at the M25 caption parameters). Panel b
    smoothly increases on a log scale; pin the qualitative trend."""
    if not baseline_path_b().exists():
        pytest.skip("Baseline missing.")

    baseline = read_baseline(accept_producer_certificate_claims=True)
    panel_b = baseline.panel_b
    # Coarse monotonicity: tail above head.
    assert panel_b.ratio_eo_01_over_10[-1] > panel_b.ratio_eo_01_over_10[0]
    # End-of-sweep ratio in the photon-limited range (paper text:
    # "for photon-assisted transitions the excitation and relaxation
    # rates are typically of the same order, Γ̃^ph_01/Γ̃^ph_10 ≈ 1/3"
    # — exact value depends on ω_LR; 0.05–1 is the comfortable band).
    assert 0.05 < panel_b.ratio_eo_01_over_10[-1] < 1.0


def test_high_temperature_rates_have_broad_kHz_scale_and_ratio_sanity() -> None:
    """Both computed families end at comparable few-kHz scales."""
    if not (baseline_path_a().exists() and baseline_path_b().exists()):
        pytest.skip("Baseline missing.")

    baseline = read_baseline(accept_producer_certificate_claims=True)
    # This is a broad code-output sanity check, not a paper-digitized
    # dominance claim. Both endpoints are a few kHz, and their ratio
    # should remain within one broad factor-of-four comparability band.
    Gamma_a_end = baseline.panel_a.Gamma_P_Hz[-1]
    Gamma_b_end = baseline.panel_b.Gamma_P_Hz[-1]
    assert 1e3 < Gamma_a_end < 2e4
    assert 1e3 < Gamma_b_end < 2e4
    assert 0.25 < Gamma_b_end / Gamma_a_end < 4.0
