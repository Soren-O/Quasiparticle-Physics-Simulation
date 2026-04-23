"""Regression test: Fischer 2023 Fig 3 finite-τ_l matches the pinned CSV to 1e-6.

Iterative-mode tolerance per NFP §6.4.1. Slow-marked (four Picard
solves with continuation at 810 bins, ~minute-scale total runtime);
opt in with ``pytest -m slow``.

First-time generation::

    python -m validation.fischer_2023.fig3_finite_tau_l
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig3_finite_tau_l import (
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
            "Generate it with: python -m validation.fischer_2023.fig3_finite_tau_l"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.tau_0_pb == pytest.approx(baseline.tau_0_pb, rel=1e-8)
    assert result.ratios == baseline.ratios

    for ratio in result.ratios:
        np.testing.assert_allclose(
            result.f_by_ratio[ratio],
            baseline.f_by_ratio[ratio],
            rtol=0.0,
            atol=1e-6,
            err_msg=f"Mismatch at τ_l/τ_0^PB = {ratio}",
        )
