"""Regression test: Fischer 2023 Fig. 6 paper-topology run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- this
sweep does $|T_B|\\times|\\bar n|$ joint Picard + self-consistent BCS gap
solves on the 1620-bin paper grid; total wall-time is on the order of an
hour. Opt in with ``pytest -m slow``.

The expensive sweep range (``N_BAR_VALUES``) is tunable in
:mod:`fig6_paper`; tighten it if this test starts to dominate the slow
suite. The pinned baseline is self-consistent against whichever range is
configured at generation time.

First-time generation::

    python -m validation.fischer_2023.fig6_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig6_paper import (
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
            "Generate it with: python -m validation.fischer_2023.fig6_paper"
        )

    baseline = read_baseline(path)
    result = run()

    # τ values --- 1e-8 relative per the pattern in test_fig5_paper.py.
    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)
    assert result.tau_l_ns == pytest.approx(baseline.tau_l_ns, rel=1e-8)

    # Sweep axes match exactly (T_B values are literal floats; n̄ values
    # are np.logspace, so allow a tiny relative slack).
    np.testing.assert_allclose(
        result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.n_bar, baseline.n_bar, rtol=1e-12, atol=0.0,
    )

    # T_*/Δ is closed-form in n̄ — should be reproducible to ~1e-10.
    np.testing.assert_allclose(
        result.T_star_over_delta, baseline.T_star_over_delta,
        rtol=1e-10, atol=0.0,
        err_msg="T_*/Δ axis drift (Eq. 35)",
    )

    # Numerical observables — 1e-6 abs (NFP §6.4.1 iterative-mode tol).
    np.testing.assert_allclose(
        result.delta_eq, baseline.delta_eq, rtol=0.0, atol=1e-6,
        err_msg="Δ_eq(T_B) drift",
    )
    np.testing.assert_allclose(
        result.delta_driven, baseline.delta_driven, rtol=0.0, atol=1e-6,
        err_msg="Δ_driven (self-consistent BCS) drift",
    )
    np.testing.assert_allclose(
        result.x_qp_num, baseline.x_qp_num, rtol=0.0, atol=1e-6,
        err_msg="numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.paper_observable_num, baseline.paper_observable_num,
        rtol=0.0, atol=1e-6,
        err_msg="(δΔ_T - δΔ)/δΔ_T numerical drift",
    )
    # x_qp_eq47 is pure closed-form (Eq. 47 + Appendix-E in T_bath, n_bar,
    # τ_ℓ, τ_0^PB). Pin tightly — drift means a coefficient changed.
    np.testing.assert_allclose(
        result.x_qp_eq47, baseline.x_qp_eq47, rtol=1e-10, atol=0.0,
        err_msg="Eq. 47 analytic x_qp drift",
    )
    # Dashed overlay is Eq. 53 evaluated at (x_qp_eq47, T_*/Δ) and combined
    # with the numerical Δ_eq(T_B). It inherits the Δ_eq tolerance (1e-6 abs
    # in μeV) via composition; the closed-form Eq. 47 + Eq. 53 part itself
    # is float64-exact.
    np.testing.assert_allclose(
        result.paper_observable_eq53, baseline.paper_observable_eq53,
        rtol=0.0, atol=1e-6,
        err_msg="Eq. 53 dashed-overlay drift",
    )
