"""Branch-jump-tolerant baseline comparison for the M25 pin tests.

The M25 moment system is multi-stable and the multi-seed picker's
fixed-point selection is platform-sensitive (BLAS rounding steers
hybr/lm convergence): isolated temperature points can land on a
different branch of the flat valley on another machine, with
order-of-magnitude deviations that carry no physics content. A
per-point ``assert_allclose`` therefore cannot be cross-platform
stable at any useful tolerance. Instead require (a) the majority of
sweep points to match the pin within ``(rtol, atol)`` and (b) a small
median relative deviation — a systematic shift (coefficient or solver
bug) moves every point and fails both criteria, while a few branch
jumps fail neither.
"""

from __future__ import annotations

import numpy as np


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
