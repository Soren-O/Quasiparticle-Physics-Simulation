"""Tests for qpsim.solvers.picard."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.solvers.picard import picard_iterate


class TestPicardIterate:
    @pytest.mark.parametrize(
        "x0",
        [
            np.array([np.nan]),
            np.array([np.inf]),
            np.array([1.0 + 0.0j]),
            np.array([1.0 + 1.0j]),
        ],
    )
    def test_rejects_nonfinite_or_complex_initial_iterate(
        self,
        x0: np.ndarray,
    ) -> None:
        called = False

        def g(x: np.ndarray) -> np.ndarray:
            nonlocal called
            called = True
            return x

        with pytest.raises(ValueError, match=r"x0 must"):
            picard_iterate(x0, g)
        assert not called

    def test_rejects_empty_initial_iterate(self) -> None:
        with pytest.raises(ValueError, match="x0 must be non-empty"):
            picard_iterate(np.array([]), lambda x: x)

    @pytest.mark.parametrize(
        "mapped",
        [
            np.array([np.nan, 0.0]),
            np.array([np.inf, 0.0]),
            np.array([0.0 + 0.0j, 0.0 + 0.0j]),
            np.array([0.0 + 1.0j, 0.0 + 0.0j]),
        ],
    )
    def test_rejects_nonfinite_or_complex_map_output(
        self,
        mapped: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match=r"g\(x\) output must"):
            picard_iterate(np.zeros(2), lambda _x: mapped)

    @pytest.mark.parametrize(
        "mapped",
        [
            np.array(0.0),
            np.array([0.0]),
            np.zeros((2, 1)),
        ],
    )
    def test_rejects_broadcastable_map_output_shape(
        self,
        mapped: np.ndarray,
    ) -> None:
        # Each of these broadcasts against shape (2,), and the all-zero
        # values previously made the malformed map certify immediately.
        with pytest.raises(
            ValueError,
            match=r"g\(x\) output must have shape \(2,\)",
        ):
            picard_iterate(np.zeros(2), lambda _x: mapped)

    def test_validates_map_output_on_every_iteration(self) -> None:
        calls = 0

        def g(x: np.ndarray) -> np.ndarray:
            nonlocal calls
            calls += 1
            if calls == 1:
                return x + 1.0
            return np.full_like(x, np.nan)

        with pytest.raises(ValueError, match=r"g\(x\) output must contain"):
            picard_iterate(np.zeros(2), g, mixing=1.0, max_iter=3)
        assert calls == 2

    def test_returns_the_snapshot_whose_defect_was_certified(self) -> None:
        """A quiet defect at x must not certify the relaxed successor."""

        def nonlinear_map(x: np.ndarray) -> np.ndarray:
            return x + 5e-5 + 1e6 * (x - 1.0)

        result, info = picard_iterate(
            np.array([1.0]),
            nonlinear_map,
            mixing=0.3,
            tol=1e-4,
        )

        assert info.converged
        np.testing.assert_array_equal(result, np.array([1.0]))
        mapped = nonlinear_map(result)
        actual = float(
            np.max(
                np.abs(mapped - result)
                / (np.maximum(np.abs(result), np.abs(mapped)) + 1e-4)
            )
        )
        assert actual == pytest.approx(info.final_residual)

    def test_in_place_map_cannot_erase_its_fixed_point_defect(self) -> None:
        def in_place_map(x: np.ndarray) -> np.ndarray:
            x += 1.0
            return x

        result, info = picard_iterate(
            np.array([0.0]),
            in_place_map,
            tol=1e-10,
            max_iter=3,
        )

        assert not info.converged
        assert info.final_residual > 0.0
        np.testing.assert_array_equal(result, np.array([0.6]))
        np.testing.assert_array_equal(in_place_map(result.copy()), np.array([1.6]))

    def test_exhausted_iteration_returns_the_measured_snapshot(self) -> None:
        """The final unevaluated relaxed step must not escape with old diagnostics."""

        def nonlinear_map(x: np.ndarray) -> np.ndarray:
            return x + 5e-5 + 1e6 * (x - 1.0)

        result, info = picard_iterate(
            np.array([1.0]),
            nonlinear_map,
            mixing=0.3,
            tol=1e-6,
            max_iter=1,
        )

        assert not info.converged
        np.testing.assert_array_equal(result, np.array([1.0]))
        mapped = nonlinear_map(result)
        actual = float(
            np.max(
                np.abs(mapped - result)
                / (np.maximum(np.abs(result), np.abs(mapped)) + 1e-6)
            )
        )
        assert actual == pytest.approx(info.final_residual)

    def test_mixing_zero_rejected(self) -> None:
        # mixing=0 makes x_mixed == x, so the change-based residual is 0 on
        # the first pass and a non-fixed-point map would falsely "converge".
        def g(x: np.ndarray) -> np.ndarray:
            return x + 100.0  # no fixed point exists

        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="mixing must lie"):
                picard_iterate(np.array([1.0]), g, mixing=bad)

    @pytest.mark.parametrize("tol", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_invalid_tolerance(self, tol: float) -> None:
        # ``tol=inf`` previously made every finite map report success after
        # one iteration because both its normalization and acceptance gate
        # became infinite.
        with pytest.raises(ValueError, match="tol must be finite and positive"):
            picard_iterate(np.array([1.0]), lambda x: x + 100.0, tol=tol)

    @pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
    def test_rejects_invalid_max_iter(self, max_iter: object) -> None:
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            picard_iterate(
                np.array([1.0]),
                lambda x: x,
                max_iter=max_iter,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("depth", [-1, 1.5, True])
    def test_rejects_invalid_anderson_depth(self, depth: object) -> None:
        with pytest.raises(
            ValueError,
            match="anderson_depth must be a non-negative integer",
        ):
            picard_iterate(
                np.array([1.0]),
                lambda x: x,
                anderson_depth=depth,  # type: ignore[arg-type]
            )

    def test_under_relaxation_does_not_scale_convergence_residual(self) -> None:
        # G(x)=2 is far from fixed at x=1. A 1e-6 relaxed step moves only
        # 1e-6, but that small step must not satisfy a 1e-4 fixed-point
        # tolerance after only one iteration.
        _, info = picard_iterate(
            np.array([1.0]),
            lambda x: np.array([2.0]),
            mixing=1e-6,
            tol=1e-4,
            max_iter=1,
        )

        assert not info.converged
        assert info.final_residual > 0.49

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
        A = 0.3 * np.array([[1.0, 0.1], [0.2, 1.0]])
        b = np.array([1.0, -1.0])

        def g(x: np.ndarray) -> np.ndarray:
            return A @ x + b

        x_star, info = picard_iterate(
            np.zeros(2), g, mixing=1.0, anderson_depth=5, tol=1e-12, max_iter=200
        )
        assert info.converged
        expected = np.linalg.solve(np.eye(2) - A, b)
        np.testing.assert_allclose(x_star, expected, atol=1e-10)

    def test_negative_fixed_point_with_anderson(self) -> None:
        # Regression: with the default clip_non_negative=False, Anderson
        # must handle fixed points with negative components. G(x) = 0.5x − 1
        # has fixed point x = −2. A previous version clipped Anderson
        # iterates to ≥ 0 unconditionally and stalled here at 0.
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x - 1.0

        x_star, info = picard_iterate(
            np.array([0.0]), g, mixing=1.0, anderson_depth=5, tol=1e-12, max_iter=100,
        )
        assert info.converged
        np.testing.assert_allclose(x_star, [-2.0], atol=1e-10)
