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
    # A1 (q = 0) is driftless absolutely, not merely by comparison. Now that
    # the benchmark reads the state in the measure the backend conserves,
    # support_fraction is 1 everywhere on this grid and the moment telescopes
    # exactly, so what remains is Crank-Nicolson round-off: 1.34e-8 um/ns,
    # measured and NE-independent. The gate is absolute at 1e-7 -- 7.4x that
    # floor, and ~4.5 orders tighter than the relative gate it replaces.
    #
    # Do NOT copy benchmark 4's 1e-8 literal: it fails at this benchmark's
    # own configuration. If cross-host jitter ever breaches 1e-7, relax to
    # 1e-6 (the precedent is the fig7 loss floor) -- that still tightens the
    # old gate by three orders. The relative form must not come back: the
    # weighting artifact it tolerated was negative while the A1P drift is
    # positive, so it admitted a genuine positive leak up to +6.2e-3 um/ns.
    assert float(np.max(np.abs(result.drift_measured["A1"]))) < 1e-7


def test_a1p_drift_exceeds_a2() -> None:
    # v_A2 / v_A1P = 1/N_1 < 1; sharpest near the gap (index 0, largest N_1).
    result = run(NE=12, NX=31)
    assert result.drift_measured["A1P"][0] > result.drift_measured["A2"][0]


def test_drift_matches_analytic_velocity() -> None:
    result = run(NE=12, NX=31)
    # A1 (q = 0): v = D_N q N_1^{q-p-1} d_x N_1 is the literal 0.0 for any
    # backend behaviour, so this pins the analytic formula's q and nothing
    # else; the measured side is gated absolutely in test_drift_sign_splits_by_q.
    assert np.all(result.drift_analytic["A1"] == 0.0)
    for name in ("A1P", "A2", "C", "B"):
        measured = result.drift_measured[name]
        analytic = result.drift_analytic[name]
        mask = np.abs(analytic) > 1e-3
        rel = np.abs(measured[mask] - analytic[mask]) / np.abs(analytic[mask])
        # Tightened 0.15 -> 0.05 once the analytic velocity was rebuilt from
        # the same cell-average N_1 as the measured moment. Mixing the two
        # measures degraded this agreement (A1P 0.025 -> 0.104), so the
        # tighter gate is what pins them to a single measure.
        assert np.max(rel) < 0.05, (name, float(np.max(rel)))
