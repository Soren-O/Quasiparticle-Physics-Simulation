"""Tests for qpsim.solvers.anderson."""

from __future__ import annotations

import numpy as np
from qpsim.solvers.anderson import anderson_extrapolate


class TestAndersonExtrapolate:
    def test_returns_none_for_insufficient_history(self) -> None:
        x = np.array([1.0, 2.0])
        gx = np.array([1.1, 2.1])
        assert anderson_extrapolate(x, gx, [], [], depth=5) is None
        assert anderson_extrapolate(x, gx, [x], [gx], depth=5) is None

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5)
        gx = rng.standard_normal(5)
        X_hist = [rng.standard_normal(5) for _ in range(3)]
        G_hist = [rng.standard_normal(5) for _ in range(3)]
        result = anderson_extrapolate(x, gx, X_hist, G_hist, depth=3)
        assert result is not None
        assert result.shape == (5,)

    def test_non_negative_output(self) -> None:
        # Output is clipped to ≥ 0 (correct for non-negative fields).
        rng = np.random.default_rng(1)
        x = rng.standard_normal(4)
        gx = rng.standard_normal(4)
        X_hist = [rng.standard_normal(4) for _ in range(3)]
        G_hist = [rng.standard_normal(4) for _ in range(3)]
        result = anderson_extrapolate(x, gx, X_hist, G_hist, depth=3)
        assert result is not None
        assert np.all(result >= 0.0)

    def test_accelerates_linear_contraction(self) -> None:
        # For a linear contraction, Anderson should produce an iterate
        # significantly closer to the fixed point than the plain Picard
        # step it's derived from.
        A = 0.5 * np.eye(2)
        b = np.array([1.0, 1.0])

        def g(x: np.ndarray) -> np.ndarray:
            return A @ x + b  # Fixed point at [2, 2].

        x0 = np.zeros(2)
        x1 = g(x0)
        x2 = g(x1)
        x3 = g(x2)
        X_hist = [x0, x1]
        G_hist = [g(x0), g(x1)]
        result = anderson_extrapolate(x3, g(x3), X_hist, G_hist, depth=2)
        assert result is not None
        err_plain = np.linalg.norm(g(x3) - np.array([2.0, 2.0]))
        err_aa = np.linalg.norm(result - np.array([2.0, 2.0]))
        assert err_aa < err_plain
