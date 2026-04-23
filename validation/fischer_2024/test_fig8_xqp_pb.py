"""Regression test: F24 Fig 8 x_qp(T_B, power) matches pinned baseline to 1e-6."""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2024.fig8_xqp_pb import baseline_path, read_baseline, run

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2024.fig8_xqp_pb"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    assert result.powers == baseline.powers
    np.testing.assert_allclose(
        result.x_qp_thermal, baseline.x_qp_thermal, rtol=1e-6, atol=1e-14,
    )
    for power in result.powers:
        np.testing.assert_allclose(
            result.x_qp_by_power[power], baseline.x_qp_by_power[power],
            rtol=1e-6, atol=1e-14,
            err_msg=f"Mismatch at power={power}",
        )
