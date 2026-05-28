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
    LOWER_NBAR,
    LOWER_T_BATH_K,
    UPPER_NBAR_VALUES,
    UPPER_T_BATH_K,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
)


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + both panels' sweep axes.

    Gating :func:`run` (the multi-minute two-panel Picard sweep) behind this
    turns a stale config/baseline pairing — a grid change, a sweep-range edit,
    a τ_0^PB drift — into a seconds-long failure instead of one discovered only
    after the full run. (See ``fig6_paper`` for the same pattern, where
    ``run()`` is ~14 h.)
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)
    axes = read_baseline(path)

    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_c == pytest.approx(meta.t_c, rel=1e-6)  # header stores 6 dp
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)

    np.testing.assert_allclose(
        np.asarray(UPPER_T_BATH_K, dtype=float), axes.upper_T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="upper-panel T_bath axis differs from baseline",
    )
    np.testing.assert_allclose(
        UPPER_NBAR_VALUES, axes.upper_nbar, rtol=1e-12, atol=0.0,
        err_msg="upper-panel n_bar axis (range/count) differs from baseline",
    )
    np.testing.assert_allclose(
        np.asarray(LOWER_NBAR, dtype=float), axes.lower_nbar,
        rtol=1e-12, atol=0.0,
        err_msg="lower-panel n_bar axis differs from baseline",
    )
    np.testing.assert_allclose(
        LOWER_T_BATH_K, axes.lower_T_bath, rtol=0.0, atol=1e-14,
        err_msg="lower-panel T_bath axis (range/count) differs from baseline",
    )


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig5_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the multi-minute run() below, instead of after it.
    _assert_config_matches_baseline(path)

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
