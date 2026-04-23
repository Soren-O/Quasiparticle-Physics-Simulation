"""Regression test: Fischer Fig. 6 characterization matches pinned baseline."""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig6_gap_suppression import (
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
            "Generate it with: python -m validation.fischer_2023.fig6_gap_suppression"
        )

    baseline = read_baseline(path)
    current = run()

    for field in (
        "T_bath",
        "n_bar",
        "tau_l_ns",
        "T_star_drive_K",
        "kBT_star_over_delta0",
        "delta_eq",
        "delta_final",
        "delta_suppression",
        "rel_suppression",
    ):
        np.testing.assert_allclose(
            getattr(current, field),
            getattr(baseline, field),
            rtol=1e-6,
            atol=1e-10,
        )
