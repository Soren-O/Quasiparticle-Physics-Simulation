"""Tests for qpsim.solvers.etd."""

from __future__ import annotations

import numpy as np
from qpsim.solvers.etd import etd1_step


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
