"""Exponential-integrator time steppers for the collision operator.

Currently provides only ETD1 (exponential Euler, first-order). The
ETD2 upgrade (second-order exponential midpoint) is a committed Gate 2
port-time change — see the Build Handoff.
"""

from __future__ import annotations

import numpy as np


def etd1_step(
    f: np.ndarray,
    gain: np.ndarray,
    loss_rate: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Exponential-Euler (ETD1) step for ``df/dt = gain − loss_rate · f``.

    The exact solution at constant ``(gain, loss_rate)`` is
    ``f(t + dt) = f(t) · e^{−μ dt} + (gain/μ) · (1 − e^{−μ dt})`` where
    ``μ = max(loss_rate, 0)``. ETD1 is this exact formula applied with
    the values frozen at the start of the step.

    The ``p_term = max(gain + (μ − loss_rate) f, 0)`` adjustment absorbs
    any slight negativity from ``loss_rate < 0`` (possible when the
    photon spontaneous-emission term is combined with a Pauli factor
    that has flipped sign numerically near ``f ≈ 1``); it preserves
    ``0 ≤ f ≤ 1`` under realistic inputs.
    """
    mu = np.maximum(loss_rate, 0.0)
    p_term = np.maximum(gain + (mu - loss_rate) * f, 0.0)

    decay = np.exp(-mu * dt)
    coeff = np.empty_like(mu)
    small = mu < 1e-14
    coeff[~small] = (1.0 - decay[~small]) / mu[~small]
    coeff[small] = dt

    updated = decay * f + coeff * p_term
    return np.clip(updated, 0.0, 1.0)
