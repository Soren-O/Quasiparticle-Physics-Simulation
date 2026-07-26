"""Tests for qpsim.solvers.anderson."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.solvers.anderson import (
    AndersonAccelerationError,
    anderson_extrapolate,
)


class TestAndersonExtrapolate:
    def test_depth_zero_disables_acceleration(self) -> None:
        assert (
            anderson_extrapolate(
                np.array([1.0]),
                np.array([2.0]),
                [np.array([0.0])],
                [np.array([1.0])],
                depth=0,
            )
            is None
        )

    @pytest.mark.parametrize("depth", [-1, 1.5, True])
    def test_rejects_invalid_depth(self, depth: object) -> None:
        with pytest.raises(
            ValueError,
            match="depth must be a non-negative integer",
        ):
            anderson_extrapolate(
                np.array([1.0]),
                np.array([2.0]),
                [np.array([0.0])],
                [np.array([1.0])],
                depth=depth,  # type: ignore[arg-type]
            )

    def test_returns_none_for_insufficient_history(self) -> None:
        x = np.array([1.0, 2.0])
        gx = np.array([1.1, 2.1])
        assert anderson_extrapolate(x, gx, [], [], depth=5) is None

    @pytest.mark.parametrize("location", ["x", "gx", "X_hist", "G_hist"])
    @pytest.mark.parametrize(
        "bad_value",
        [1.0 + 2.0j, complex(1.0, float("nan"))],
        ids=["complex", "imaginary-nan"],
    )
    def test_rejects_complex_inputs_before_float_cast(
        self,
        location: str,
        bad_value: complex,
    ) -> None:
        """No current/history complex component may disappear in a cast."""
        x = np.array([1.0])
        gx = np.array([1.5])
        X_hist = [np.array([0.0])]
        G_hist = [np.array([1.0])]
        if location == "x":
            x = np.array([bad_value])
        elif location == "gx":
            gx = np.array([bad_value])
        elif location == "X_hist":
            X_hist = [np.array([bad_value])]
        else:
            G_hist = [np.array([bad_value])]

        with pytest.raises(ValueError, match="real-valued"):
            anderson_extrapolate(
                x,
                gx,
                X_hist,
                G_hist,
                depth=1,
            )

    def test_rejects_nonfinite_map_input_as_value_error(self) -> None:
        """A bad fixed-point map is not an acceleration retry signal."""
        with pytest.raises(ValueError, match="only finite"):
            anderson_extrapolate(
                np.array([1.0]),
                np.array([np.nan]),
                [np.array([0.0])],
                [np.array([1.0])],
                depth=1,
            )

    def test_wraps_lstsq_failure_as_acceleration_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Finite inputs plus a failed AA solve can trigger safe plain retry."""

        def fail_lstsq(*args, **kwargs):
            del args, kwargs
            raise np.linalg.LinAlgError("synthetic SVD failure")

        monkeypatch.setattr(np.linalg, "lstsq", fail_lstsq)
        with pytest.raises(
            AndersonAccelerationError,
            match="least-squares solve failed",
        ):
            anderson_extrapolate(
                np.array([1.0]),
                np.array([1.5]),
                [np.array([0.0])],
                [np.array([1.0])],
                depth=1,
            )

    def test_wraps_finite_input_overflow_as_acceleration_error(self) -> None:
        """Overflow in AA algebra is retryable; its finite inputs are valid."""
        largest = np.finfo(float).max
        with pytest.raises(
            AndersonAccelerationError,
            match="residual-difference arithmetic",
        ):
            anderson_extrapolate(
                np.array([largest]),
                np.array([-largest]),
                [np.array([-largest])],
                [np.array([largest])],
                depth=1,
            )

    def test_depth_one_is_a_working_secant_update(self) -> None:
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x + 1.0  # fixed point at x = 2

        x0 = np.zeros(1)
        x1 = g(x0)
        result = anderson_extrapolate(
            x1, g(x1), [x0], [g(x0)], depth=1,
        )

        assert result is not None
        np.testing.assert_allclose(result, np.array([2.0]), atol=1e-14)

    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(5)
        gx = rng.standard_normal(5)
        X_hist = [rng.standard_normal(5) for _ in range(3)]
        G_hist = [rng.standard_normal(5) for _ in range(3)]
        result = anderson_extrapolate(x, gx, X_hist, G_hist, depth=3)
        assert result is not None
        assert result.shape == (5,)

    def test_clip_non_negative_projects_output(self) -> None:
        # With clip_non_negative=True, the output is projected to ≥ 0.
        rng = np.random.default_rng(1)
        x = rng.standard_normal(4)
        gx = rng.standard_normal(4)
        X_hist = [rng.standard_normal(4) for _ in range(3)]
        G_hist = [rng.standard_normal(4) for _ in range(3)]
        result = anderson_extrapolate(
            x, gx, X_hist, G_hist, depth=3, clip_non_negative=True,
        )
        assert result is not None
        assert np.all(result >= 0.0)

    def test_default_does_not_clip(self) -> None:
        # Default is clip_non_negative=False. For a problem whose exact
        # extrapolated value is negative, the output must preserve the
        # sign rather than snap to 0.
        def g(x: np.ndarray) -> np.ndarray:
            return 0.5 * x - 1.0  # fixed point at x = -2

        x0 = np.zeros(1)
        x1 = g(x0)
        x2 = g(x1)
        x3 = g(x2)
        result = anderson_extrapolate(
            x3, g(x3), [x0, x1], [g(x0), g(x1)], depth=2,
        )
        assert result is not None
        assert result[0] < 0.0  # reached a negative iterate

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
