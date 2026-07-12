"""Exponential-integrator time steppers for the collision operator.

Provides ``etd1_step`` (first-order exponential Euler) and
``etd2_step`` (second-order Heun-type ETD). Both are specialized to
``∂_t x = gain(x) − loss_rate(x) · x`` with ``loss_rate ≥ 0``; the
linear relaxation in ``loss_rate`` is handled exactly, and the
nonlinear ``gain`` is discretized to first- or second-order.

ETD2 is the committed port-time upgrade from the Build Handoff —
``apply_phonon_collision`` in :mod:`qpsim.collisions.phonon` uses it.
"""

from __future__ import annotations

from collections.abc import Callable

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
    # coeff = (1 - e^{-mu dt}) / mu, the exact ETD1 weight on the source
    # term. Compute it with expm1 so it stays accurate when mu*dt is tiny:
    # forming 1 - exp(-mu dt) directly cancels to 0 once exp(-mu dt) rounds
    # to 1.0 (mu*dt below ~1e-16), which silently zeroed the entire gain
    # contribution for small-but-finite mu. In the mu -> 0 limit coeff -> dt;
    # the inner np.where keeps the division well-defined at mu == 0.
    positive = mu > 0.0
    coeff = np.where(positive, -np.expm1(-mu * dt) / np.where(positive, mu, 1.0), dt)

    updated = decay * f + coeff * p_term
    return np.clip(updated, 0.0, 1.0)


def etd2_step(
    f: np.ndarray,
    rhs: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    dt: float,
) -> np.ndarray:
    """Heun-type ETD2 step for ``df/dt = gain(f) − loss_rate(f) · f``.

    Two-stage predictor-corrector with the linear ``e^{−μ dt}``
    relaxation handled exactly at each stage:

    1. Predictor — run one ``etd1_step`` with ``(gain_n, loss_n) = rhs(f)``.
    2. Corrector — evaluate ``(gain_p, loss_p) = rhs(f_pred)``, then
       run a second ``etd1_step`` on the original ``f`` with the
       averaged rates ``½(gain_n + gain_p)`` and ``½(loss_n + loss_p)``.

    Second-order accurate in ``dt`` for general ``gain(f)`` /
    ``loss_rate(f)``; reduces to ``etd1_step`` exactly when the rates
    are frozen (linear problem).

    Parameters
    ----------
    f
        Initial occupation, shape ``(NE,)``.
    rhs
        Callable ``rhs(f) → (gain, loss_rate)``, each shape ``(NE,)``.
        Invoked twice per step.
    dt
        Time step (ns).

    Returns
    -------
    np.ndarray
        Updated occupation, clipped to ``[0, 1]``.
    """
    gain_n, loss_n = rhs(f)
    f_pred = etd1_step(f, gain_n, loss_n, dt)
    gain_p, loss_p = rhs(f_pred)
    gain_avg = 0.5 * (gain_n + gain_p)
    loss_avg = 0.5 * (loss_n + loss_p)
    return etd1_step(f, gain_avg, loss_avg, dt)
