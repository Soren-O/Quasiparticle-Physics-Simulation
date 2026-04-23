"""Tests for qpsim.solvers.etd."""

from __future__ import annotations

import numpy as np
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


class TestEtd2Step:
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
