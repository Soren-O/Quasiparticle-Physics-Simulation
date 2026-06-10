"""Fast invariants for the gap-gradient drift benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators.gap_gradient_drift import run


def test_drift_sign_splits_by_q() -> None:
    # A1P/A2 (q = 2) drift up the gap gradient; C/B (q < 0) drift down it;
    # the dirty-limit A1 (q = 0) carries no DOS-gradient drift.
    result = run(NE=12, NX=31, n_steps=8)
    assert np.all(result.drift_measured["A1P"] > 0.0)
    assert np.all(result.drift_measured["A2"] > 0.0)
    assert np.all(result.drift_measured["C"] < 0.0)
    assert np.all(result.drift_measured["B"] < 0.0)
    # A1's residual drift is a finite-packet/discretization artifact, far
    # below the dressed drifts.
    a1 = float(np.max(np.abs(result.drift_measured["A1"])))
    a1p = float(np.max(np.abs(result.drift_measured["A1P"])))
    assert a1 < 0.05 * a1p, (a1, a1p)


def test_a1p_drift_exceeds_a2() -> None:
    # v_A2 / v_A1P = 1/N_1 < 1; sharpest near the gap (index 0, largest N_1).
    result = run(NE=12, NX=31)
    assert result.drift_measured["A1P"][0] > result.drift_measured["A2"][0]


def test_drift_matches_analytic_velocity() -> None:
    result = run(NE=12, NX=31)
    # A1 (q = 0): the analytic drift velocity is identically zero.
    assert np.all(result.drift_analytic["A1"] == 0.0)
    for name in ("A1P", "A2", "C", "B"):
        measured = result.drift_measured[name]
        analytic = result.drift_analytic[name]
        mask = np.abs(analytic) > 1e-3
        rel = np.abs(measured[mask] - analytic[mask]) / np.abs(analytic[mask])
        assert np.max(rel) < 0.15, (name, float(np.max(rel)))
