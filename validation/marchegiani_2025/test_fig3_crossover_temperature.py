"""Regression test: M25 Eq. 8 crossover-T̄ sweep matches pinned CSV."""

from __future__ import annotations

import numpy as np
import pytest

from validation.marchegiani_2025.fig3_crossover_temperature import (
    baseline_path,
    read_baseline,
    run,
)


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. Generate it with: "
            "python -m validation.marchegiani_2025.fig3_crossover_temperature"
        )

    baseline = read_baseline(path)
    result = run()

    # The sweep axis is np.logspace output; numpy's SIMD dispatch picks
    # per-CPU code paths for 10**x, so heterogeneous CI runners can
    # differ from the pinning machine by ~1 ULP (observed 2026-07-03:
    # 5.7e-14 at g ≈ 316, exactly one ULP — the old atol=1e-14 sat
    # *below* one ULP at that magnitude). A few-ULP relative tolerance
    # keeps the pin meaningful without demanding cross-microarch bit
    # identity.
    np.testing.assert_allclose(
        result.g_photon_R_Hz, baseline.g_photon_R_Hz, rtol=1e-15, atol=0.0,
    )
    # Eq. 8 is a closed-form Lambert-W evaluation — the whole sweep
    # should reproduce to floating-point precision from run to run.
    np.testing.assert_allclose(
        result.T_bar_kelvin, baseline.T_bar_kelvin, rtol=1e-12, atol=0.0,
    )
