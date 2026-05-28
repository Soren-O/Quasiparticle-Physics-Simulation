"""Regression test: Fischer 2023 Fig. 3 paper-target run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked ---
this run does the τ_l = 0 thermal-phonon Newton plus the seven-step
continuation through Picard ratios, capped by a coupled-Newton solve
at ratio 10 on the 1620-bin paper grid; total wall-time is several
minutes. Opt in with ``pytest -m slow``.

First-time generation::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig3_paper import (
    PAPER_RATIOS,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
)


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header.

    Gating :func:`run` (the several-minute continuation ladder) behind this
    turns a stale config/baseline pairing — a grid change, a ratio-set edit, a
    τ_0^PB drift — into a seconds-long failure instead of one discovered only
    after the full run. (See ``fig6_paper`` for the same pattern, where
    ``run()`` is ~14 h.)
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)

    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_bath == pytest.approx(meta.t_bath)
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.n_bar == pytest.approx(meta.n_bar)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)
    assert cfg.ratios == meta.ratios, (
        f"paper ratios config={cfg.ratios} != baseline {meta.ratios}; "
        "regenerate the baseline or restore the ratio set before the slow run."
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
            "Generate it with: python -m validation.fischer_2023.fig3_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the several-minute run() below, instead of after it.
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)
    assert result.ratios == baseline.ratios

    np.testing.assert_allclose(
        result.f_FD, baseline.f_FD, rtol=0.0, atol=1e-14,
        err_msg="Fermi-Dirac reference drift",
    )

    for ratio in result.ratios:
        np.testing.assert_allclose(
            result.f_by_ratio[ratio],
            baseline.f_by_ratio[ratio],
            rtol=0.0,
            atol=1e-6,
            err_msg=f"Mismatch at τ_l/τ_0^PB = {ratio}",
        )
