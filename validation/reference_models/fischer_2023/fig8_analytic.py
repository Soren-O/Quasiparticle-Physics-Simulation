"""Minimal clean-room analytic reference for Fischer--Catelani 2023 Fig. 8.

This module is independent of :mod:`qpsim`. It transcribes only Appendix E,
Eq. E2 (``bar R / R``), using the three numerical moments printed directly
above that equation. It does not reconstruct the solid numerical curve.
"""

from __future__ import annotations

import numpy as np

APPENDIX_E_A_MINUS_HALF = 2.1
APPENDIX_E_A_PLUS_HALF = 0.88
APPENDIX_E_A_PLUS_THREE_HALVES = 0.77

EQ_E2_LINEAR = APPENDIX_E_A_PLUS_HALF / APPENDIX_E_A_MINUS_HALF
EQ_E2_QUADRATIC = (
    1.25 * APPENDIX_E_A_PLUS_THREE_HALVES / APPENDIX_E_A_MINUS_HALF
    - 0.75 * EQ_E2_LINEAR**2
)


def fig8_analytic_curve(t_star_over_delta: np.ndarray) -> np.ndarray:
    """Evaluate paper Eq. E2 on a strictly increasing nonnegative grid."""

    raw = np.asarray(t_star_over_delta)
    if raw.ndim != 1 or raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(
            "t_star_over_delta must be a one-dimensional real array."
        )
    epsilon = np.asarray(raw, dtype=float)
    if (
        epsilon.size < 2
        or np.any(~np.isfinite(epsilon))
        or np.any(epsilon < 0.0)
        or np.any(np.diff(epsilon) <= 0.0)
    ):
        raise ValueError(
            "t_star_over_delta must contain at least two strictly increasing "
            "finite nonnegative values."
        )
    return 1.0 + EQ_E2_LINEAR * epsilon + EQ_E2_QUADRATIC * epsilon**2
