"""Regression test: M25 Fig 4 paper-target run matches the pinned baseline CSV.

Slow-marked --- runs the full Fig. 4 pipeline (six (ω_LR, model)
sweeps, each NUM_T_POINTS rate-equation solves with previous-T
continuation seeds), so total wall-time is several minutes. Opt in
with ``pytest -m slow``.

Tolerance: ``rtol=5e-2`` — qualitative-trend pin per the validation
plan. Panel (a) carries multi-stability scatter from the max-x_L
branch picker (load-bearing gap; see module docstring) and the
reduced-model curves are placeholders (load-bearing gap), so a tighter
tolerance would mainly pin scipy iteration-count drift.

First-time generation::

    python -m validation.marchegiani_2025.fig4_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025.fig4_paper import (
    MODEL_FULL,
    MODEL_GLOBAL,
    MODEL_RENORM,
    baseline_path,
    read_baseline,
    run,
)

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.marchegiani_2025.fig4_paper"
        )

    baseline = read_baseline(path)
    result = run()

    # Same set of (ω_LR, model) panels in both.
    assert set(result.panels.keys()) == set(baseline.panels.keys()), (
        "Panel-key set diverged from baseline."
    )

    for key, expected in baseline.panels.items():
        actual = result.panels[key]
        omega, model = key

        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"(ω_LR={omega} GHz, model={model}): T_kelvin drifted",
        )
        # Multi-stability scatter on panel (a) and placeholder reduced
        # models impose a 5 % rtol per the validation plan; the test
        # therefore pins the baseline against itself only up to scipy
        # iteration-count drift across versions.
        np.testing.assert_allclose(
            actual.Gamma_P_Hz, expected.Gamma_P_Hz, rtol=5e-2,
            err_msg=f"(ω_LR={omega} GHz, model={model}): Gamma_P_Hz drifted",
        )
        np.testing.assert_allclose(
            actual.ratio_eo_01_over_10, expected.ratio_eo_01_over_10,
            rtol=5e-2,
            err_msg=f"(ω_LR={omega} GHz, model={model}): ratio drifted",
        )


def test_full_model_present_for_both_omega() -> None:
    """Sanity: the load-bearing full-model curves are pinned for both
    ω_LR cases. The placeholder curves are not load-bearing for paper
    parity (gap 2), but the full model is."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    for omega_LR_GHz in (0.5, 5.0):
        panel = baseline.get(omega_LR_GHz, MODEL_FULL)
        assert panel.Gamma_P_Hz.size > 0
        assert np.all(np.isfinite(panel.Gamma_P_Hz))
        # Γ_P should sit in the kHz–MHz band at the M25 caption params.
        assert (panel.Gamma_P_Hz > 1e2).all()
        assert (panel.Gamma_P_Hz < 1e8).all()


def test_panel_b_ratio_increases_with_T_full_model() -> None:
    """Paper trend (full model only): the excitation/relaxation ratio
    Γ̃^eo_{01}/Γ̃^eo_{10} rises with T toward the photon-limited band
    by 150 mK. Pin the qualitative monotonic-on-average trend; the
    placeholder reduced models are not pinned against this trend."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    for omega_LR_GHz in (0.5, 5.0):
        panel = baseline.get(omega_LR_GHz, MODEL_FULL)
        assert panel.ratio_eo_01_over_10[-1] > panel.ratio_eo_01_over_10[0]
        # End-of-sweep ratio in the photon-limited band (paper text:
        # "for photon-assisted transitions Γ̃^ph_01/Γ̃^ph_10 ≈ 1/3";
        # exact value depends on ω_LR; 0.05–0.5 is the comfortable band
        # for both panels per the published Fig. 4).
        assert 0.05 < panel.ratio_eo_01_over_10[-1] < 0.5


def test_reduced_model_panels_present() -> None:
    """The placeholder reduced-model curves should still produce finite
    output even though they are not paper-faithful (gap 2). If they
    start emitting NaN or blowing up, that is a regression."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    for omega_LR_GHz in (0.5, 5.0):
        for model in (MODEL_GLOBAL, MODEL_RENORM):
            panel = baseline.get(omega_LR_GHz, model)
            assert np.all(np.isfinite(panel.Gamma_P_Hz)), (
                f"placeholder ({omega_LR_GHz} GHz, {model}): non-finite Γ_P"
            )
