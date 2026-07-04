"""Paired regression test for the transient photon-kick demo.

Closes the gap flagged in ``docs/Validation_Chain.md`` ("baseline committed,
no regression test"): pins :func:`photon_kick_response.run` against the
committed CSV baseline and enforces the demo's physical acceptance criteria
(monotone x_qp rise; late-time agreement with the independently computed
Newton steady state).

One transient solve (1200 ETD2 steps on the 810-bin grid) plus one Newton
steady-state solve, shared across tests via a module fixture. Slow-marked
like the other reproduction pins.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpsim.observables.density import qp_fraction
from validation.transient.photon_kick_response import (
    DELTA_0,
    baseline_path,
    read_baseline,
    run,
)

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def result():
    if not baseline_path().exists():
        pytest.skip(
            f"Baseline not found at {baseline_path()}. "
            "Generate it with: python -m validation.transient.photon_kick_response"
        )
    return run()


def test_matches_pinned_baseline(result) -> None:
    baseline = read_baseline()

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-12)
    # atol accommodates two artifacts of the time axis: dt-accumulation
    # drift in the live run (~2.5e-12 ns by t = 120 ns) and the %g
    # rounding of the baseline header labels. 1e-9 ns is still 8 orders
    # below the snapshot interval.
    np.testing.assert_allclose(
        result.snapshot_times, baseline.snapshot_times, rtol=0.0, atol=1e-9,
    )
    np.testing.assert_allclose(
        result.f_steady_state, baseline.f_steady_state, rtol=1e-6, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.f_snapshots, baseline.f_snapshots, rtol=1e-6, atol=1e-14,
    )
    assert result.x_qp_steady_state == pytest.approx(
        baseline.x_qp_steady_state, rel=1e-6,
    )


def test_x_qp_rises_monotonically(result) -> None:
    # Acceptance criterion from the validation plan: monotone rise of
    # x_qp(t) from the thermal floor under the pair-breaking drive.
    diffs = np.diff(result.x_qp_snapshots)
    assert np.all(diffs > 0.0), (
        f"x_qp(t) is not monotone: min step {diffs.min():.3e}"
    )


def test_late_time_approaches_newton_steady_state(result) -> None:
    # The t → ∞ limit of the transient must agree with the independently
    # computed Newton steady state. At t = 120 ns (≈ 2 τ_0) the residual
    # gap is finite but must already be small and shrinking.
    x_end = result.x_qp_snapshots[-1]
    x_prev = result.x_qp_snapshots[-2]
    x_ss = result.x_qp_steady_state
    gap_end = abs(x_ss - x_end)
    gap_prev = abs(x_ss - x_prev)
    assert gap_end < gap_prev, "transient is not still approaching steady state"
    assert gap_end < 0.10 * x_ss, (
        f"late-time x_qp {x_end:.6e} deviates from Newton steady state "
        f"{x_ss:.6e} by more than 10%"
    )


def test_snapshot_x_qp_consistent_with_f(result) -> None:
    # The recorded x_qp observable must equal qp_fraction of the recorded
    # f snapshot (guards the observable plumbing in run_time_dependent).
    # Rebuild the spectral context via a fresh state is unnecessary — the
    # result carries E only, so recompute using the final snapshot pair.
    from qpsim.grid.energy_grid import integration_widths_from_centers
    from qpsim.physics.spectral import SpectralContext

    spectral = SpectralContext(
        E_bins=result.E,
        dE_bins=integration_widths_from_centers(result.E),
        gap=DELTA_0,
    )
    x_from_f = float(qp_fraction(result.f_snapshots[-1], spectral, delta_0=DELTA_0))
    assert x_from_f == pytest.approx(result.x_qp_snapshots[-1], rel=1e-12)
