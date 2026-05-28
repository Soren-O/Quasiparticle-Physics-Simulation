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
    N_BAR_VALUES,
    T_BATH_VALUES,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
)


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + sweep axes.

    Gating the ~14 h :func:`run` behind this turns a stale config/baseline
    pairing — a ``TAU_L_MODEL`` swap, a grid change, a sweep-range edit — into
    a seconds-long failure instead of one discovered only after the full
    sweep. Compares the config fingerprint against the baseline header, and
    the configured sweep axes against the baseline's data rows.
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)
    axes = read_baseline(path)

    assert cfg.tau_l_model == meta.tau_l_model, (
        f"TAU_L_MODEL config={cfg.tau_l_model!r} != baseline {meta.tau_l_model!r}; "
        "regenerate the baseline or restore the model before the slow run."
    )
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
    assert cfg.film_thickness_nm == pytest.approx(meta.film_thickness_nm)
    assert cfg.eta == pytest.approx(meta.eta)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)
    assert cfg.tau_l_ns == pytest.approx(meta.tau_l_ns, rel=1e-8)
    np.testing.assert_allclose(
        np.asarray(T_BATH_VALUES, dtype=float), axes.T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="T_bath sweep axis differs from baseline",
    )
    np.testing.assert_allclose(
        N_BAR_VALUES, axes.n_bar, rtol=1e-12, atol=0.0,
        err_msg="n_bar sweep axis (range/count) differs from baseline",
    )


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header.

    This is the standing fast-suite guard that would have caught the τ_ℓ-model
    / baseline mismatch that once wasted 9.5 h. The slow
    ``test_matches_pinned_baseline`` re-runs the same check inline so the 14 h
    sweep is gated even when this fast test is not selected.
    """
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
            "Generate it with: python -m validation.fischer_2023.fig6_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the ~14 h run() below, instead of after it.
    _assert_config_matches_baseline(path)

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
