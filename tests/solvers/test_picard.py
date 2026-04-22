"""Tests for qpsim.solvers.picard."""

from __future__ import annotations

import numpy as np
from qpsim.solvers.picard import picard_iterate


class TestPicardIterate:
    def test_converges_on_linear_contraction(self) -> None:
        # G(x) = 0.5·x + 2. Fixed point: x = 4.
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x + 2.0

        x_star, info = picard_iterate(np.array([0.0]), g, mixing=1.0, tol=1e-12, max_iter=100)
        assert info.converged
        np.testing.assert_allclose(x_star, [4.0], atol=1e-10)

    def test_info_fields(self) -> None:
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x + 2.0

        _, info = picard_iterate(np.array([0.0]), g, mixing=1.0, tol=1e-12)
        assert info.n_iter > 0
        assert info.converged is True
        assert info.final_residual < 1e-12

    def test_fails_when_divergent(self) -> None:
        # G(x) = 2x — expansion; no contraction ⇒ doesn't converge.
        def g(x: np.ndarray) -> np.ndarray:
            return 2.0 * x

        x_start = np.array([1.0])
        _, info = picard_iterate(x_start, g, mixing=1.0, tol=1e-12, max_iter=20)
        assert not info.converged

    def test_anderson_accelerates(self) -> None:
        # Same linear map; Anderson should converge at least as fast.
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x + 2.0

        _, info_plain = picard_iterate(
            np.array([0.0]), g, mixing=0.5, anderson_depth=0, tol=1e-12, max_iter=200
        )
        _, info_aa = picard_iterate(
            np.array([0.0]), g, mixing=0.5, anderson_depth=5, tol=1e-12, max_iter=200
        )
        assert info_plain.converged and info_aa.converged
        assert info_aa.n_iter <= info_plain.n_iter

    def test_multidim_fixed_point(self) -> None:
        # Note on sign: anderson_extrapolate clips its output to ≥ 0
        # (it's specialized for phonon occupations). Using a non-
        # negative fixed point keeps the generic test honest.
        A = 0.3 * np.array([[1.0, 0.1], [0.2, 1.0]])
        b = np.array([1.0, 1.0])

        def g(x: np.ndarray) -> np.ndarray:
            return A @ x + b

        x_star, info = picard_iterate(
            np.zeros(2), g, mixing=1.0, anderson_depth=5, tol=1e-12, max_iter=200
        )
        assert info.converged
        expected = np.linalg.solve(np.eye(2) - A, b)
        np.testing.assert_allclose(x_star, expected, atol=1e-10)
