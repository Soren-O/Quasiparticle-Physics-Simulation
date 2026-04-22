"""Tests for qpsim.solvers.ssprk."""

from __future__ import annotations

import numpy as np
from qpsim.solvers.ssprk import ssprk22_step


class TestSsprk22Step:
    def test_exponential_decay(self) -> None:
        # ∂_t x = -k x; exact solution x(t) = x₀ e^{-k t}.
        k = 1.0
        x0 = np.array([1.0])
        dt = 0.05

        def rhs(x: np.ndarray) -> np.ndarray:
            return -k * x

        x = x0.copy()
        t = 0.0
        for _ in range(40):  # integrate to t = 2.0
            x = ssprk22_step(x, rhs, dt)
            t += dt
        expected = x0 * np.exp(-k * t)
        np.testing.assert_allclose(x, expected, rtol=1e-3)

    def test_second_order_accuracy(self) -> None:
        # Halving dt should shrink the error by ≈4× for a 2nd-order method.
        k = 1.0
        x0 = np.array([1.0])
        T = 1.0

        def rhs(x: np.ndarray) -> np.ndarray:
            return -k * x

        def run(dt: float) -> float:
            x = x0.copy()
            steps = round(T / dt)
            for _ in range(steps):
                x = ssprk22_step(x, rhs, dt)
            return float(abs(x[0] - x0[0] * np.exp(-k * T)))

        err_coarse = run(0.1)
        err_fine = run(0.05)
        # Accept anywhere near 4× (dt²-scaling); leave margin for finite-t effects.
        assert err_fine < err_coarse / 3.0

    def test_preserves_zero(self) -> None:
        def rhs(x: np.ndarray) -> np.ndarray:
            return np.zeros_like(x)

        x0 = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(ssprk22_step(x0, rhs, dt=0.5), x0)

    def test_vectorized_application(self) -> None:
        # Works on arbitrary shapes as long as rhs returns matching shape.
        def rhs(x: np.ndarray) -> np.ndarray:
            return -x

        x0 = np.arange(6.0).reshape(2, 3)
        got = ssprk22_step(x0, rhs, dt=0.1)
        # For ∂_t x = -x, exact: x(0.1) = x0 e^{-0.1}.
        np.testing.assert_allclose(got, x0 * (1 - 0.1 + 0.5 * 0.1 ** 2), rtol=1e-12)
