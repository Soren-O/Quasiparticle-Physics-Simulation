"""Tests for qpsim.solvers.etd."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.solvers.etd import etd1_step, etd2_step


class TestEtd1Step:
    def test_pure_decay(self) -> None:
        # No gain, constant loss rate μ: f(t+dt) = f·e^{-μ dt}.
        f = np.array([0.5, 0.2, 0.8])
        mu = np.array([1.0, 2.0, 0.5])
        dt = 0.1
        got = etd1_step(f, gain=np.zeros_like(f), loss_rate=mu, dt=dt)
        expected = f * np.exp(-mu * dt)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_no_loss_only_gain(self) -> None:
        # Loss rate 0 means coefficient becomes dt; f(t+dt) = f + gain·dt.
        f = np.array([0.1, 0.2, 0.3])
        gain = np.array([0.05, 0.10, 0.15])
        dt = 0.5
        got = etd1_step(f, gain=gain, loss_rate=np.zeros_like(f), dt=dt)
        expected = np.clip(f + gain * dt, 0.0, 1.0)
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_preserves_bounds(self) -> None:
        # Adversarial inputs should still clip to [0, 1].
        f = np.array([0.9, 0.01, 0.5])
        gain = np.array([5.0, 0.0, 0.5])
        loss_rate = np.array([10.0, 1e-20, 1.0])
        got = etd1_step(f, gain, loss_rate, dt=0.3)
        assert np.all(got >= 0.0)
        assert np.all(got <= 1.0)

    def test_equilibrium_fixed_point(self) -> None:
        # If gain = μ·f initially, any dt should leave f unchanged.
        f = np.array([0.3, 0.7])
        mu = np.array([2.0, 1.5])
        gain = mu * f
        got = etd1_step(f, gain, mu, dt=0.25)
        np.testing.assert_allclose(got, f, rtol=1e-12)

    @pytest.mark.parametrize("bad_dt", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_dt(self, bad_dt: float) -> None:
        f = np.array([0.2, 0.4])
        with pytest.raises(ValueError, match="dt must be finite and non-negative"):
            etd1_step(f, np.zeros_like(f), np.ones_like(f), bad_dt)

    @pytest.mark.parametrize("mismatched", ["gain", "loss_rate"])
    def test_rejects_rate_shape_broadcasting(self, mismatched: str) -> None:
        f = np.array([0.2, 0.4])
        gain = np.zeros_like(f)
        loss_rate = np.ones_like(f)
        if mismatched == "gain":
            gain = gain[:1]
        else:
            loss_rate = loss_rate[:1]

        with pytest.raises(ValueError, match="must have the same shape"):
            etd1_step(f, gain, loss_rate, dt=0.1)

    @pytest.mark.parametrize("field", ["f", "gain", "loss_rate"])
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
    def test_rejects_nonfinite_arrays(self, field: str, bad_value: float) -> None:
        f = np.array([0.2, 0.4])
        gain = np.zeros_like(f)
        loss_rate = np.ones_like(f)
        arrays = {"f": f, "gain": gain, "loss_rate": loss_rate}
        arrays[field][0] = bad_value

        with pytest.raises(ValueError, match=field):
            etd1_step(f, gain, loss_rate, dt=0.1)

    def test_finite_adversarial_inputs_still_clip(self) -> None:
        f = np.array([-2.0, 2.0, 0.5])
        gain = np.array([-100.0, 1e6, -0.2])
        loss_rate = np.array([-100.0, 0.0, 1e6])

        got = etd1_step(f, gain, loss_rate, dt=0.3)

        assert np.all(np.isfinite(got))
        assert np.all((got >= 0.0) & (got <= 1.0))


class TestEtd2Step:
    def test_rate_limit_retries_when_predictor_reveals_stiff_loss(self) -> None:
        # x' = 1 - 10^6 x^2, x(0)=0 has
        # x(t)=tanh(1000 t)/1000. Looking only at loss_n sees zero loss on
        # the first attempt; its predictor reaches x=1 and reveals the stiff
        # nonlinear loss that must force a rejected, shorter trial.
        calls = 0

        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            nonlocal calls
            calls += 1
            return np.ones_like(x), 1.0e6 * x

        diagnostics: dict[str, int] = {}
        got = etd2_step(
            np.array([0.0]),
            rhs,
            dt=1.0,
            max_loss_step=0.25,
            _diagnostics=diagnostics,
        )

        exact = np.tanh(1000.0) / 1000.0
        assert got[0] == pytest.approx(exact, rel=2.0e-3)
        assert diagnostics["accepted_substeps"] > 1
        assert diagnostics["rejected_attempts"] > 0
        # Each accepted ETD2 substep has two stages. Rejected predictor
        # trials add RHS evaluations but are not reported as completed work.
        assert calls == (
            2 * diagnostics["accepted_substeps"]
            + diagnostics["rejected_attempts"]
        )

    def test_optional_balance_projection_preserves_stiff_exchange(self) -> None:
        rate = 20.0
        weights = np.ones(2)

        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return (
                rate * np.array([x[1] * (1.0 - x[0]), x[0] * (1.0 - x[1])]),
                rate * np.array([1.0 - x[1], 1.0 - x[0]]),
            )

        f = np.array([0.001, 0.8])
        got = etd2_step(
            f,
            rhs,
            dt=0.5,
            balance_weights=weights,
            max_loss_step=0.25,
        )

        assert float(weights @ got) == pytest.approx(float(weights @ f), abs=1e-14)
        assert np.all((got >= 0.0) & (got <= 1.0))

    def test_balance_axis_prevents_cross_column_compensation(self) -> None:
        rates = np.array([20.0, 5.0])
        weights = np.ones((2, 2))

        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain = np.empty_like(x)
            loss = np.empty_like(x)
            gain[0] = rates * x[1] * (1.0 - x[0])
            gain[1] = rates * x[0] * (1.0 - x[1])
            loss[0] = rates * (1.0 - x[1])
            loss[1] = rates * (1.0 - x[0])
            return gain, loss

        f = np.array([[0.001, 0.75], [0.8, 0.02]])
        got = etd2_step(
            f,
            rhs,
            dt=0.5,
            balance_weights=weights,
            balance_axis=0,
            max_loss_step=0.25,
        )

        np.testing.assert_allclose(
            np.sum(weights * got, axis=0),
            np.sum(weights * f, axis=0),
            atol=2e-14,
            rtol=0.0,
        )

    def test_balance_projection_is_slice_local_across_mass_scales(self) -> None:
        """A huge column must not hide an O(1) defect in another column."""
        weights = np.array([[1.0e16, 1.0], [1.0e16, 1.0]])
        candidate = np.array([[0.5, 0.5], [0.5, 0.5]])

        # Exercise the public step with a source whose Heun balance requests
        # no mass change in the large column and complete removal in the
        # small one.  The unprojected ETD candidate retains mass in column 1.
        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain = np.zeros_like(x)
            loss = np.zeros_like(x)
            loss[:, 1] = 2.0
            return gain, loss

        got = etd2_step(
            candidate,
            rhs,
            dt=1.0,
            balance_weights=weights,
            balance_axis=0,
        )

        residual_n = -np.array([[0.0, 2.0], [0.0, 2.0]]) * candidate
        pred = etd1_step(
            candidate,
            np.zeros_like(candidate),
            np.array([[0.0, 2.0], [0.0, 2.0]]),
            1.0,
        )
        residual_p = -np.array([[0.0, 2.0], [0.0, 2.0]]) * pred
        target = np.sum(weights * candidate, axis=0) + 0.5 * np.sum(
            weights * (residual_n + residual_p), axis=0,
        )
        target = np.clip(target, 0.0, np.sum(weights, axis=0))

        np.testing.assert_allclose(
            np.sum(weights * got, axis=0), target, rtol=2e-15, atol=0.0,
        )

    def test_reduces_to_etd1_for_linear_problem(self) -> None:
        # When rhs is exactly linear (constant gain, constant loss), ETD2
        # and ETD1 agree to machine precision — the averaging step is
        # a no-op because the rates don't depend on f.
        f = np.array([0.3, 0.6])
        mu = np.array([2.0, 1.0])
        gain = np.array([0.1, 0.2])

        def rhs(_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return gain.copy(), mu.copy()

        got_etd2 = etd2_step(f, rhs, dt=0.2)
        got_etd1 = etd1_step(f, gain, mu, dt=0.2)
        np.testing.assert_allclose(got_etd2, got_etd1, rtol=1e-14)

    def test_equilibrium_fixed_point(self) -> None:
        # A true fixed point (gain = μ·f with gain and μ depending on f)
        # should be preserved to high accuracy.
        f = np.array([0.3, 0.7])

        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mu = np.array([2.0, 1.5])
            return mu * x, mu

        got = etd2_step(f, rhs, dt=0.25)
        np.testing.assert_allclose(got, f, rtol=1e-12)

    def test_preserves_bounds(self) -> None:
        f = np.array([0.9, 0.01, 0.5])

        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return 5.0 * (1.0 - x), 2.0 * x + 1.0

        got = etd2_step(f, rhs, dt=0.3)
        assert np.all(got >= 0.0)
        assert np.all(got <= 1.0)

    def test_second_order_accuracy_on_nonlinear_rhs(self) -> None:
        # Nonlinear benchmark: ∂_t x = -x² (exact: x(t) = x0 / (1 + x0 t)).
        # Represented as gain = 0, loss_rate = x. ETD1 is first-order on
        # this problem (frozen-loss approximation); ETD2 is second-order,
        # so halving dt should roughly quarter the global error.
        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(x), x

        T = 1.0
        x0 = np.array([0.9])
        exact = x0 / (1.0 + x0 * T)

        def global_error(dt: float) -> float:
            x = x0.copy()
            steps = round(T / dt)
            for _ in range(steps):
                x = etd2_step(x, rhs, dt)
            return float(abs(x[0] - exact[0]))

        err_coarse = global_error(0.1)
        err_fine = global_error(0.05)
        # Second-order: err(dt/2) ≈ err(dt) / 4. Require at least 3× reduction.
        assert err_fine < err_coarse / 3.0

    def test_rate_limited_subcycling_retains_second_order_accuracy(self) -> None:
        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(x), x

        x0 = np.array([0.9])
        exact = x0 / (1.0 + x0)

        def error(max_loss_step: float) -> float:
            got = etd2_step(
                x0,
                rhs,
                dt=1.0,
                max_loss_step=max_loss_step,
            )
            return float(abs(got[0] - exact[0]))

        err_coarse = error(0.05)
        err_fine = error(0.025)
        assert err_fine < err_coarse / 3.5

    def test_faster_than_etd1_on_nonlinear_rhs(self) -> None:
        # On the same benchmark, ETD2 should beat ETD1 at the same dt.
        def rhs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(x), x

        T = 1.0
        x0 = np.array([0.9])
        exact = x0 / (1.0 + x0 * T)
        dt = 0.1
        steps = round(T / dt)

        x_etd1 = x0.copy()
        x_etd2 = x0.copy()
        for _ in range(steps):
            g, L = rhs(x_etd1)
            x_etd1 = etd1_step(x_etd1, g, L, dt)
            x_etd2 = etd2_step(x_etd2, rhs, dt)

        err_etd1 = float(abs(x_etd1[0] - exact[0]))
        err_etd2 = float(abs(x_etd2[0] - exact[0]))
        assert err_etd2 < err_etd1
