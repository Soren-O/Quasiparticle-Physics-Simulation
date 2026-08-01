"""Platform-aware pinned-baseline comparison for the M25 pin tests.

Policy: every baseline CSV records the ``sys.platform`` it was
generated on (authenticated ``producer_platform`` table metadata;
legacy ``# pinned_on:`` headers remain readable). On that platform the pin
comparison is strict (``rtol = 1e-6``): the branch-continuation driver
is deterministic and the tracked root is unique with residuals
~1e-12 Hz, so any drift beyond 1e-6 is a real regression (or a
scipy-version behavior change worth noticing). On every other
platform the same comparison **runs** — at ``rtol = 1e-3`` with the
absolute tolerance floored at 1e-30 — instead of being skipped: only
rounding-level cross-platform scatter (BLAS/libm differences steering
the last Newton steps) is expected post-fix, and 1e-3 still catches
minority-point corruption by orders of magnitude.

Historical note: this module used to (a) *skip* the strict pins
entirely off the generating platform and (b) offer a "robust"
majority-fraction + median comparison (≥70 % of points within 5 %,
median within 1 %). Both were calibrated for the
pre-``cooper_pair_number_R`` regime, in which the density equations
ran on the ensemble ``Γ̃`` rates and the moment system was a
float64-flat valley of pseudo-roots whose selection was
platform-sensitive — ubuntu CI landed ~70 % away from the Windows
pins at 25/29 sweep points, a different branch family rather than
noise. The single-quasiparticle Γ̄ normalization (M25 text below
Eq. 6; commit 3199378) removed the multi-stability and with it the
rationale for skipping cross-platform or tolerating branch jumps, so
both mechanisms are gone.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

PIN_PLATFORM_PREFIX = "# pinned_on: "

#: Same-platform regression tolerance (deterministic driver).
STRICT_RTOL = 1e-6
#: Cross-platform tolerance (rounding-level scatter only, post-fix).
CROSS_PLATFORM_RTOL = 1e-3
#: Cross-platform absolute floor so exact zeros cannot produce
#: spurious relative failures; far below every pinned physical scale.
CROSS_PLATFORM_ATOL_FLOOR = 1e-30


def baseline_pin_platform(path: Path) -> str | None:
    """Return the producer ``sys.platform`` recorded by the baseline."""
    with path.open(encoding="utf-8", newline="") as fp:
        for line in fp:
            if line.startswith("# qpsim_metadata="):
                try:
                    row = next(csv.reader([line]))
                    metadata = json.loads(row[0].split("=", 1)[1])
                    producer_platform = metadata.get("producer_platform")
                except (csv.Error, json.JSONDecodeError, IndexError, ValueError):
                    return None
                return (
                    producer_platform
                    if isinstance(producer_platform, str)
                    else None
                )
            if not line.startswith("#"):
                return None
            if line.startswith(PIN_PLATFORM_PREFIX):
                return line.removeprefix(PIN_PLATFORM_PREFIX).strip()
    return None


def assert_pinned_match(
    actual: np.ndarray,
    expected: np.ndarray,
    name: str,
    *,
    baseline_path: Path,
    atol: float = 0.0,
) -> None:
    """Compare a regenerated curve against its pinned baseline.

    The tolerance is chosen by the baseline's platform stamp: strict
    ``rtol = 1e-6`` on the generating platform; ``rtol = 1e-3``
    everywhere else (including a missing stamp), with ``atol`` floored
    at 1e-30 so pinned exact zeros compare cleanly. ``atol`` lets
    callers widen the absolute floor for observables with a genuine
    numerical noise floor (e.g. ``μ_α`` near the μ → 0 equilibrium
    attractor at the top of the temperature sweep).
    """
    pinned_on = baseline_pin_platform(baseline_path)
    if pinned_on == sys.platform:
        rtol = STRICT_RTOL
    else:
        rtol = CROSS_PLATFORM_RTOL
        atol = max(atol, CROSS_PLATFORM_ATOL_FLOOR)
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=rtol,
        atol=atol,
        err_msg=(
            f"{name}: drifted from pinned baseline {baseline_path.name} "
            f"(pinned_on={pinned_on!r}, running on {sys.platform!r}, "
            f"rtol={rtol:g}, atol={atol:g})"
        ),
    )
