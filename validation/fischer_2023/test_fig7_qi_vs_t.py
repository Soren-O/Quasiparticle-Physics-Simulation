"""Regression test: Fischer Fig 7 Q_i(T_B) at fixed n̄ matches baseline to 1e-4.

Tolerance tier per NFP §6.4.1 (Fig 7 with drive = 1e-4 reflecting the
combined nbar-loop + Mattis-Bardeen quadrature drift). Slow-marked;
the sweep is ~11 Newton solves and takes roughly 20-40 s at 360 bins.

First-time generation::

    python -m validation.fischer_2023.fig7_qi_vs_t
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig7_qi_vs_t import baseline_path, read_baseline, run

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig7_qi_vs_t"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)

    # Thermal Q_i can be very large at low T (σ_1 → 0). Compare as a
    # relative error where the reference is large; fall back to atol where
    # the reference is order-unity or infinite.
    finite = np.isfinite(baseline.Q_thermal) & np.isfinite(result.Q_thermal)
    np.testing.assert_allclose(
        result.Q_thermal[finite], baseline.Q_thermal[finite],
        rtol=1e-4, atol=1e-4,
    )
    np.testing.assert_allclose(
        result.Q_driven, baseline.Q_driven, rtol=1e-4, atol=1e-4,
    )
