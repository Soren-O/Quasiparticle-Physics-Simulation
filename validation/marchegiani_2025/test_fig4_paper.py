"""Regression test: M25 Fig 4 paper-faithful run matches the pinned baseline CSV.

Fast since the branch-continuation driver landed: the full-model
sweeps are warm-started Newton solves on the well-conditioned
(Γ̄-normalized) moment system and the reduced-model curves are
closed-form fixed-point iterations, so this runs in the default gate.

Tolerance is platform-stamped (see ``_robust.assert_pinned_match``):
strict rtol=1e-6 on the generating platform — all curves are
deterministic; rtol=1e-3 elsewhere (rounding-level cross-platform
scatter only, post Γ̄-normalization).

First-time generation::

    python -m validation.marchegiani_2025.fig4_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025._robust import assert_pinned_match
from validation.marchegiani_2025.fig4_paper import (
    MODEL_FULL,
    MODEL_GLOBAL,
    MODEL_RENORM,
    baseline_path,
    read_baseline,
    run,
)


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.marchegiani_2025.fig4_paper"
        )

    baseline = read_baseline(path)
    result = run()

    # Same set of (ω_LR, model) panels in both: full + global for both
    # ω_LR cases, renorm for the 5 GHz family only (paper Fig. 4).
    expected_keys = {
        (0.5, MODEL_FULL), (0.5, MODEL_GLOBAL),
        (5.0, MODEL_FULL), (5.0, MODEL_GLOBAL), (5.0, MODEL_RENORM),
    }
    assert set(result.panels.keys()) == expected_keys
    assert set(baseline.panels.keys()) == expected_keys

    for key, expected in baseline.panels.items():
        actual = result.panels[key]
        omega, model = key

        np.testing.assert_allclose(
            actual.T_kelvin, expected.T_kelvin,
            rtol=0.0, atol=1e-14,
            err_msg=f"(ω_LR={omega} GHz, model={model}): T_kelvin drifted",
        )
        assert_pinned_match(
            actual.Gamma_P_Hz, expected.Gamma_P_Hz,
            f"(ω_LR={omega} GHz, model={model}): Gamma_P_Hz",
            baseline_path=path,
        )
        assert_pinned_match(
            actual.ratio_eo_01_over_10, expected.ratio_eo_01_over_10,
            f"(ω_LR={omega} GHz, model={model}): ratio",
            baseline_path=path,
        )


def test_full_model_matches_paper_scale_and_shape() -> None:
    """Paper Fig. 4a values: purple (0.5 GHz) full model starts near
    1.9 kHz with a shallow nonmonotonic dip below ~25 mK, rising to
    ~6 kHz at 150 mK; teal (5 GHz) starts near 0.8 kHz and rises
    monotonically to ~4.5 kHz."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)

    a = baseline.get(0.5, MODEL_FULL)
    assert 1.2e3 < a.Gamma_P_Hz[0] < 2.5e3
    assert 5e3 < a.Gamma_P_Hz[-1] < 8e3
    # The low-T nonmonotonic dip (main text: competition between
    # Γ̄^{R>}_{00} tunneling and τ_R^{-1} relaxation).
    assert float(np.min(a.Gamma_P_Hz[:5])) < a.Gamma_P_Hz[0]

    b = baseline.get(5.0, MODEL_FULL)
    assert 5e2 < b.Gamma_P_Hz[0] < 1.5e3
    assert 3e3 < b.Gamma_P_Hz[-1] < 6e3
    assert np.all(np.diff(b.Gamma_P_Hz) > 0.0), "5 GHz Γ_P not monotone"


def test_gamma_P_smooth_no_multistability_scatter() -> None:
    """Ticket acceptance: panel (a) scatter gone. Log-slope between
    adjacent 5 mK points stays far below the order-of-magnitude
    spikes the old max-x_L picker produced (dlog10 ≈ 1); genuine
    physics — the low-T dip and the thermal upturn at the top of the
    sweep — stays below dlog10 ≈ 0.14."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    for key, panel in baseline.panels.items():
        dlog = np.abs(np.diff(np.log10(panel.Gamma_P_Hz)))
        assert float(dlog.max()) < 0.2, (
            f"{key}: Γ_P jumps by 10^{dlog.max():.2f} between adjacent "
            "T points — scatter regression"
        )


def test_panel_b_ratio_increases_with_T_full_model() -> None:
    """Paper trend (full model): the excitation/relaxation ratio rises
    with T, ending near 0.19–0.22 at 150 mK, and sits above the
    detailed-balance reference exp(−ω_10/T) at low T because the
    photon channels have Γ^{ph}_{01}/Γ^{ph}_{10} = O(1)."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    omega_10_K = 5.5e9 * 4.799243e-11
    for omega_LR_GHz in (0.5, 5.0):
        panel = baseline.get(omega_LR_GHz, MODEL_FULL)
        assert panel.ratio_eo_01_over_10[-1] > panel.ratio_eo_01_over_10[0]
        assert 0.1 < panel.ratio_eo_01_over_10[-1] < 0.35
        # Above detailed balance at the lowest temperature.
        db_low = np.exp(-omega_10_K / panel.T_kelvin[0])
        assert panel.ratio_eo_01_over_10[0] > 10.0 * db_low


def test_global_quasiequilibrium_tracks_full_model_small_asymmetry() -> None:
    """Main text Sec. II.5: "For small gap asymmetry, the global
    quasiequilibrium modeling reproduces accurately the full
    nonequilibrium calculation, except at the lowest temperatures."
    Pin: within 15% above 60 mK, and missing the low-T nonmonotonic
    feature (global < full at the lowest T)."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    full = baseline.get(0.5, MODEL_FULL)
    glob = baseline.get(0.5, MODEL_GLOBAL)
    mask = full.T_kelvin > 0.060
    np.testing.assert_allclose(
        glob.Gamma_P_Hz[mask], full.Gamma_P_Hz[mask], rtol=0.15,
    )
    assert glob.Gamma_P_Hz[0] < full.Gamma_P_Hz[0]


def test_renormalized_mimics_full_model_on_Gamma_P_but_not_ratio() -> None:
    """The renormalized parameters were chosen by the paper's authors
    so the global-quasiequilibrium reduction mimics the large-
    asymmetry full model on Γ_P(T) — while deviating visibly on the
    excitation/relaxation ratio at low T (the paper's argument that a
    joint measurement discriminates the regimes)."""
    path = baseline_path()
    if not path.exists():
        pytest.skip("Baseline missing.")
    baseline = read_baseline(path)
    full = baseline.get(5.0, MODEL_FULL)
    renorm = baseline.get(5.0, MODEL_RENORM)
    # Γ_P agreement to ~35% across the sweep (paper: "reasonably well").
    np.testing.assert_allclose(
        renorm.Gamma_P_Hz, full.Gamma_P_Hz, rtol=0.35,
    )
    # Ratio deviates strongly at the lowest temperatures (paper panel
    # b: the dot-dashed curve starts several times above the solid).
    assert renorm.ratio_eo_01_over_10[0] > 3.0 * full.ratio_eo_01_over_10[0]
