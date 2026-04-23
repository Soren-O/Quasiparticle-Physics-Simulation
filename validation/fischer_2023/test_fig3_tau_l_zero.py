"""Regression test: Fischer 2023 Fig 3 τ_l=0 matches the pinned CSV to 1e-12.

The reproduction is a full Newton solve on the 1620-bin Fischer grid;
runtime is roughly a minute at typical laptop speeds, so the test is
marked slow and skipped by default. Opt in with ``pytest -m slow``.

First-time generation of the baseline is a manual step:

    python -m validation.fischer_2023.fig3_tau_l_zero

After which a PDF lands next to the CSV at
``validation/baselines/ph0_constant/fischer_fig3_tau_l_zero.pdf`` for
visual comparison against the published figure.
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig3_tau_l_zero import baseline_path, read_baseline, run

pytestmark = pytest.mark.slow


def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig3_tau_l_zero"
        )

    baseline = read_baseline(path)
    result = run()

    # E grid is derived deterministically from the script's constants;
    # compare to a tight tolerance to catch accidental grid changes.
    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)

    # Newton steady-state solves: 1e-12 bit-identical is the Gate-3 target.
    np.testing.assert_allclose(
        result.f_thermal, baseline.f_thermal, rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        result.f_photon, baseline.f_photon, rtol=0.0, atol=1e-12,
    )
    # The analytic Fermi-Dirac reference is tightened — it's a closed form.
    np.testing.assert_allclose(
        result.f_FD, baseline.f_FD, rtol=0.0, atol=1e-14,
    )
