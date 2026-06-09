"""Fast invariants for the gap-gradient drift benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators.gap_gradient_drift import run


def test_drift_sign_splits_by_q() -> None:
    # A1/A2 (q = 2) drift up the gap gradient; C/B (q < 0) drift down it.
    result = run(NE=12, NX=31, n_steps=8)
    assert np.all(result.drift_measured["A1"] > 0.0)
    assert np.all(result.drift_measured["A2"] > 0.0)
    assert np.all(result.drift_measured["C"] < 0.0)
    assert np.all(result.drift_measured["B"] < 0.0)


def test_a1_drift_exceeds_a2() -> None:
    # v_A2 / v_A1 = 1/N_1 < 1; sharpest near the gap (index 0, largest N_1).
    result = run(NE=12, NX=31)
    assert result.drift_measured["A1"][0] > result.drift_measured["A2"][0]


def test_drift_matches_analytic_velocity() -> None:
    result = run(NE=12, NX=31)
    for name in ("A1", "A2", "C", "B"):
        measured = result.drift_measured[name]
        analytic = result.drift_analytic[name]
        mask = np.abs(analytic) > 1e-3
        rel = np.abs(measured[mask] - analytic[mask]) / np.abs(analytic[mask])
        assert np.max(rel) < 0.15, (name, float(np.max(rel)))
