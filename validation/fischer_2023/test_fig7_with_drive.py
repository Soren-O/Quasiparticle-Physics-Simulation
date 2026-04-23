"""Regression test: Fischer Fig 7 with drive (n̄ loop) matches baseline to 1e-4.

Iterative-mode tolerance per NFP §6.4.1 is 1e-4 for the combined
n̄-loop + Mattis-Bardeen sub-gap quadrature. Slow-marked (6 dBm
points × 11 T_B points × multiple nbar-loop iterations).

First-time generation::

    python -m validation.fischer_2023.fig7_with_drive
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig7_with_drive import (
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
            "Generate it with: python -m validation.fischer_2023.fig7_with_drive"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    assert result.p_read_dbm == baseline.p_read_dbm

    np.testing.assert_allclose(
        result.Q_thermal, baseline.Q_thermal, rtol=1e-4, atol=1e-14,
        err_msg="Q_thermal drift",
    )
    for p in result.p_read_dbm:
        np.testing.assert_allclose(
            result.Q_driven_by_dbm[p], baseline.Q_driven_by_dbm[p],
            rtol=1e-4, atol=1e-14,
            err_msg=f"Q_driven drift at P_read={p:g} dBm",
        )
        np.testing.assert_allclose(
            result.n_bar_by_dbm[p], baseline.n_bar_by_dbm[p],
            rtol=1e-4, atol=1e-14,
            err_msg=f"n_bar drift at P_read={p:g} dBm",
        )
