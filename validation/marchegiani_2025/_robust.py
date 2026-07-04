"""Branch-jump-tolerant baseline comparison for the M25 pin tests.

Historical rationale: with the density equations run on the ensemble
``Γ̃`` rates (the pre-``cooper_pair_number_R`` normalization) the M25
moment system was pathologically ill-conditioned — a float64-flat
valley of pseudo-roots whose selection was platform-sensitive (BLAS
rounding steered hybr/lm convergence), so isolated temperature points
could land orders of magnitude apart across machines and a per-point
``assert_allclose`` could not be cross-platform stable. The tolerant
criterion here — (a) a majority of sweep points within ``(rtol,
atol)`` and (b) a small median relative deviation — passes a few
branch jumps while still failing on any systematic shift.

Current status: the branch-continuation driver
(``qpsim.services.rate_equation.solve_rate_equation_branch``) plus the
single-quasiparticle Γ̄ normalization removed the noise this module
guards against — the tracked root is unique and solves land at
residuals ~1e-12 Hz, so run-to-run and (expected) cross-platform
scatter is at rounding level. The tolerant comparison is retained as
cheap insurance against scipy-version drift; the ``pinned_on``
platform stamp still gates the strict pins to the generating machine.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PIN_PLATFORM_PREFIX = "# pinned_on: "


def baseline_pin_platform(path: Path) -> str | None:
    """Return the ``sys.platform`` recorded in the baseline CSV header."""
    with path.open() as fp:
        for line in fp:
            if not line.startswith("#"):
                return None
            if line.startswith(PIN_PLATFORM_PREFIX):
                return line.removeprefix(PIN_PLATFORM_PREFIX).strip()
    return None


def skip_unless_pinned_here(*paths: Path) -> None:
    """Skip the calling test unless every baseline was pinned on this
    platform.

    The multi-seed picker's fixed-point selection differs *systematically*
    between platforms (observed: ubuntu CI lands ~70% away from the
    Windows pins at 25/29 sweep points — a different branch family, not
    noise), so the strict pin comparison is a same-platform regression
    gate only. The qualitative paper-anchored tests run everywhere.
    """
    for path in paths:
        pinned_on = baseline_pin_platform(path)
        if pinned_on is None:
            pytest.skip(
                f"{path.name}: baseline carries no platform stamp — "
                "regenerate it on this machine to enable the strict pin"
            )
        if pinned_on != sys.platform:
            pytest.skip(
                f"{path.name}: baseline pinned on {pinned_on!r}, running "
                f"on {sys.platform!r} — fixed-point selection is "
                "platform-dependent; strict pin is same-platform only"
            )


def assert_robust_match(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    *,
    rtol: float = 5e-2,
    atol: float = 0.0,
    min_fraction: float = 0.7,
    median_rtol: float = 1e-2,
) -> None:
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert np.all(np.isfinite(actual)), f"{name}: non-finite values"
    dev = np.abs(actual - expected)
    ok = dev <= atol + rtol * np.abs(expected)
    fraction = float(ok.mean())
    denom = np.maximum(np.abs(expected), atol if atol > 0.0 else 1e-300)
    median_rel = float(np.median(dev / denom))
    assert fraction >= min_fraction and median_rel <= median_rtol, (
        f"{name}: {int((~ok).sum())}/{ok.size} points outside "
        f"rtol={rtol}/atol={atol} (allowed fraction {1 - min_fraction:.2f}), "
        f"median relative deviation {median_rel:.3g} "
        f"(allowed {median_rtol}) — systematic drift, not branch noise"
    )
