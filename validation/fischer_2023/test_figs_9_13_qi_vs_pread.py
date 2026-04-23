"""Regression test: Fischer Figs 9-13 Q_i(P_read) matches baseline to 1e-4.

Iterative-mode tolerance per NFP §6.4.1 (nbar-loop tol × MB sub-gap
quadrature). Slow-marked (21 P_read points × variable nbar-loop
iterations).

First-time generation::

    python -m validation.fischer_2023.figs_9_13_qi_vs_pread
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.figs_9_13_qi_vs_pread import (
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
            "Generate it with: python -m validation.fischer_2023.figs_9_13_qi_vs_pread"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(
        result.P_read_uev_per_ns, baseline.P_read_uev_per_ns,
        rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.P_read_dbm, baseline.P_read_dbm, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.Q_i, baseline.Q_i, rtol=1e-4, atol=1e-14, err_msg="Q_i drift",
    )
    np.testing.assert_allclose(
        result.Q_tot, baseline.Q_tot, rtol=1e-4, atol=1e-14, err_msg="Q_tot drift",
    )
    np.testing.assert_allclose(
        result.n_bar, baseline.n_bar, rtol=1e-4, atol=1e-14, err_msg="n_bar drift",
    )
