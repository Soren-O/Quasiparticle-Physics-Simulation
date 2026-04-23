"""Regression test: Fischer Fig 5 x_qp(T_B) fixed-n̄ matches baseline to 1e-6.

Iterative-mode tolerance per NFP §6.4.1. Slow-marked.

First-time generation::

    python -m validation.fischer_2023.fig5_xqp
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig5_xqp import baseline_path, read_baseline, run

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig5_xqp"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(
        result.x_qp_thermal, baseline.x_qp_thermal, rtol=1e-6, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.x_qp_driven, baseline.x_qp_driven, rtol=1e-6, atol=1e-14,
    )
