"""Generic Picard fixed-point iteration with optional Anderson acceleration.

Iterates ``x_{k+1} = (1 − α) x_k + α G(x_k)`` until the relative
change falls below ``tol``. When ``anderson_depth > 0``, uses Anderson
extrapolation (see ``qpsim.solvers.anderson``) on the mixed iterate.

The specialized steady-state solver in ``qpsim.services.steady_state``
does *not* call this function directly — it embeds a more elaborate
version inline that also tracks branch-collapse of the ``(f, n_ph)``
coupled system. ``picard_iterate`` here is the plain primitive for
use elsewhere (future coupled solves, rate-equation iterations, etc.).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qpsim.solvers.anderson import anderson_extrapolate


@dataclass
class PicardInfo:
    """Diagnostic info returned alongside the converged iterate.

    Attributes
    ----------
    n_iter
        Number of outer iterations performed (``1``-indexed; 0 means
        the initial guess was already within tolerance).
    converged
        True iff convergence was reached within ``max_iter``.
    final_residual
        The ``max_{i} |x_{k+1} − x_k| / (|x_k| + tol)`` at termination.
    """

    n_iter: int
    converged: bool
    final_residual: float


def picard_iterate(
    x0: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray],
    *,
    mixing: float = 0.3,
    anderson_depth: int = 0,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> tuple[np.ndarray, PicardInfo]:
    """Run Picard iteration on the fixed-point map ``G``.

    Parameters
    ----------
    x0
        Initial iterate.
    g
        Fixed-point map ``x → G(x)``.
    mixing
        Under-relaxation factor in ``(0, 1]``. Smaller values improve
        stability at the cost of more iterations; 0.3 is the default
        used in the steady-state solver.
    anderson_depth
        History window for Anderson acceleration. 0 (default) disables
        acceleration and runs plain mixed Picard. Typical values 5-10.
    tol
        Relative-L∞ tolerance on the iterate change.
    max_iter
        Hard cap on iterations.

    Returns
    -------
    x_star
        The converged iterate (or the last iterate if convergence was
        not reached).
    info
        :class:`PicardInfo` with ``n_iter``, ``converged``, and
        ``final_residual``.
    """
    x = np.array(x0, dtype=float).ravel()
    use_anderson = anderson_depth > 0
    X_hist: list[np.ndarray] = []
    G_hist: list[np.ndarray] = []
    final_residual = float("inf")

    for it in range(1, max_iter + 1):
        gx = g(x)
        x_mixed = (1.0 - mixing) * x + mixing * gx

        change = np.abs(x_mixed - x)
        scale = np.maximum(np.abs(x), np.abs(x_mixed)) + tol
        final_residual = float(np.max(change / scale))

        if final_residual < tol:
            return x_mixed, PicardInfo(n_iter=it, converged=True, final_residual=final_residual)

        if use_anderson:
            x_aa = anderson_extrapolate(x, x_mixed, X_hist, G_hist, anderson_depth)
            X_hist.append(x.copy())
            G_hist.append(x_mixed.copy())
            if len(X_hist) > anderson_depth + 1:
                X_hist.pop(0)
                G_hist.pop(0)
            x = x_aa if x_aa is not None else x_mixed
        else:
            x = x_mixed

    return x, PicardInfo(n_iter=max_iter, converged=False, final_residual=final_residual)
