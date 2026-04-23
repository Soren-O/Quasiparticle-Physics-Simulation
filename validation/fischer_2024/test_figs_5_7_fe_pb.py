"""Regression test: Fischer 2024 Figs 5-7 f(E) matches pinned baseline to 1e-6.

Iterative-mode tolerance per NFP §6.4.1. Slow-marked (five Newton
solves at 810 bins); opt in with ``pytest -m slow``.

First-time generation::

    python -m validation.fischer_2024.figs_5_7_fe_pb
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2024.figs_5_7_fe_pb import baseline_path, read_baseline, run

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2024.figs_5_7_fe_pb"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.powers == baseline.powers

    np.testing.assert_allclose(
        result.f_thermal, baseline.f_thermal, rtol=1e-6, atol=1e-14,
        err_msg="f_thermal drift",
    )
    for p in result.powers:
        np.testing.assert_allclose(
            result.f_by_power[p], baseline.f_by_power[p],
            rtol=1e-6, atol=1e-14,
            err_msg=f"f drift at power={p:g}",
        )
        assert result.x_qp_by_power[p] == pytest.approx(
            baseline.x_qp_by_power[p], rel=1e-6,
        )
