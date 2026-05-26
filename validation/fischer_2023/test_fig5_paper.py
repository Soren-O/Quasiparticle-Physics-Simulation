"""Regression test: Fischer 2023 Fig. 5 paper-topology run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- each
panel does dozens of finite-τ_l Picard solves on the 1620-bin paper
grid; total wall-time is on the order of an hour. Opt in with
``pytest -m slow``.

The expensive sweep ranges (``UPPER_NBAR_VALUES`` and
``LOWER_T_BATH_K``) are tunable in :mod:`fig5_paper`; tighten them if
this test starts to dominate the slow suite. The pinned baseline is
self-consistent against whichever ranges are configured at generation
time.

First-time generation::

    python -m validation.fischer_2023.fig5_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig5_paper import (
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
            "Generate it with: python -m validation.fischer_2023.fig5_paper"
        )

    baseline = read_baseline(path)
    result = run()

    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)

    # Upper panel — sweep axes match exactly, then x_qp + T_*/Δ to 1e-6.
    np.testing.assert_allclose(
        result.upper_T_bath, baseline.upper_T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.upper_nbar, baseline.upper_nbar, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        result.upper_T_star, baseline.upper_T_star, rtol=1e-10, atol=0.0,
        err_msg="Upper-panel T_* drift",
    )
    np.testing.assert_allclose(
        result.upper_x_qp_num, baseline.upper_x_qp_num, rtol=0.0, atol=1e-6,
        err_msg="Upper-panel numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.upper_x_qp_analytic, baseline.upper_x_qp_analytic,
        rtol=1e-10, atol=0.0,
        err_msg="Upper-panel analytic x_qp drift",
    )

    # Lower panel.
    np.testing.assert_allclose(
        result.lower_T_bath, baseline.lower_T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.lower_nbar, baseline.lower_nbar, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        result.lower_x_qp_num, baseline.lower_x_qp_num, rtol=0.0, atol=1e-6,
        err_msg="Lower-panel numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.lower_x_qp_analytic, baseline.lower_x_qp_analytic,
        rtol=1e-10, atol=0.0,
        err_msg="Lower-panel analytic x_qp drift",
    )
