"""Regression test for the Fischer 2023 Fig. 7 paper-facing validation."""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig7_paper import (
    Q_EXT_BY_DBM,
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
            "Generate it with: python -m validation.fischer_2023.fig7_paper"
        )

    baseline = read_baseline(path)
    result = run(
        temperatures=tuple(float(x) for x in baseline.T_bath),
        powers_dbm=baseline.p_read_dbm,
    )

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    for p in baseline.p_read_dbm:
        assert result.n_bar_by_dbm[p] == pytest.approx(baseline.n_bar_by_dbm[p], rel=1e-12)
        np.testing.assert_allclose(
            result.Q_qp_by_dbm[p], baseline.Q_qp_by_dbm[p],
            rtol=1e-4, atol=1e-14, err_msg=f"Q_qp drift at P_read={p:g} dBm",
        )
        np.testing.assert_allclose(
            result.Q_tot_by_dbm[p], baseline.Q_tot_by_dbm[p],
            rtol=1e-4, atol=1e-14, err_msg=f"Q_tot drift at P_read={p:g} dBm",
        )


def test_low_temperature_plateau_is_extrinsic_limited() -> None:
    result = run(temperatures=(0.06,), powers_dbm=(-64.0,))
    assert result.Q_qp_by_dbm[-64.0][0] > 1e12
    assert result.Q_tot_by_dbm[-64.0][0] == pytest.approx(
        Q_EXT_BY_DBM[-64.0], rel=1e-6,
    )
