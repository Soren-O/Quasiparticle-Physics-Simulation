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
    # A1's residual drift is an oracle-weighting artifact, not operator
    # drift: `run` takes the first moment of the POINT DOS N_1^p f while the
    # backend conserves the cell-average capacity, and that ratio varies
    # across the ramp (6.3e-2 in the lowest cell at NE = 12). Re-weighted
    # with the backend's own capacity the A1 drift is 1.3e-8 um/ns, so this
    # relative gate only separates A1 from the dressed drifts; the driftless
    # claim itself is gated by the refinement test below and, absolutely, by
    # test_self_consistent_feedback.py.
    a1 = float(np.max(np.abs(result.drift_measured["A1"])))
    a1p = float(np.max(np.abs(result.drift_measured["A1P"])))
    assert a1 < 0.05 * a1p, (a1, a1p)


def test_a1_drift_collapses_under_energy_refinement() -> None:
    # The A1 (q = 0) residual is the point-vs-cell-average DOS mismatch of
    # the benchmark's own first moment, so it vanishes with the energy-cell
    # width (~NE^-1.6); a genuine q-driven drift velocity would not.
    coarse = float(np.max(np.abs(run(NE=12, NX=31, n_steps=8).drift_measured["A1"])))
    fine = float(np.max(np.abs(run(NE=48, NX=31, n_steps=8).drift_measured["A1"])))
    assert fine < 0.35 * coarse, (coarse, fine)


def test_a1p_drift_exceeds_a2() -> None:
    # v_A2 / v_A1P = 1/N_1 < 1; sharpest near the gap (index 0, largest N_1).
    result = run(NE=12, NX=31)
    assert result.drift_measured["A1P"][0] > result.drift_measured["A2"][0]


def test_drift_matches_analytic_velocity() -> None:
    result = run(NE=12, NX=31)
    # A1 (q = 0): v = D_N q N_1^{q-p-1} d_x N_1 is the literal 0.0 for any
    # backend behaviour, so this pins the analytic formula's q and nothing
    # else; the measured side is gated by the refinement test above.
    assert np.all(result.drift_analytic["A1"] == 0.0)
    for name in ("A1P", "A2", "C", "B"):
        measured = result.drift_measured[name]
        analytic = result.drift_analytic[name]
        mask = np.abs(analytic) > 1e-3
        rel = np.abs(measured[mask] - analytic[mask]) / np.abs(analytic[mask])
        assert np.max(rel) < 0.15, (name, float(np.max(rel)))
