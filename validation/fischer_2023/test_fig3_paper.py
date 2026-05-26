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
            "Generate it with: python -m validation.fischer_2023.fig3_paper"
        )

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
