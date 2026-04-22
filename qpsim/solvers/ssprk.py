"""Second-order strong-stability-preserving Runge-Kutta (Shu–Osher SSPRK(2,2)).

Two-stage, second-order TVD time stepper. Used to integrate the TVD
finite-volume spatial discretization of the spectral-flow operator
(see :mod:`qpsim.solvers.spectral_flow_tvd`).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def ssprk22_step(
    x0: np.ndarray,
    rhs: Callable[[np.ndarray], np.ndarray],
    dt: float,
) -> np.ndarray:
    """One SSPRK(2,2) step for ``∂_t x = rhs(x)``.

    ``x₁ = x_n + dt · rhs(x_n)``
    ``x_{n+1} = ½ x_n + ½ (x₁ + dt · rhs(x₁))``

    Second-order accurate and strong-stability-preserving under the
    forward-Euler stability limit (Gottlieb, Shu & Tadmor 2001). The
    caller is responsible for any bounds clipping (e.g. ``f ∈ [0, 1]``)
    after the step.
    """
    x1 = x0 + dt * rhs(x0)
    return 0.5 * x0 + 0.5 * (x1 + dt * rhs(x1))
