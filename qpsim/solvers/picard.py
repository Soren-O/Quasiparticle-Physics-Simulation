"""Generic Picard fixed-point iteration with optional Anderson acceleration.

Iterates ``x_{k+1} = (1 − α) x_k + α G(x_k)`` until the normalized
fixed-point residual falls below ``tol`` — normalized, not relative: see the
``tol`` parameter for the exact criterion and the magnitude range over which
it is genuinely relative. When ``anderson_depth > 0``, uses Anderson
extrapolation (see ``qpsim.solvers.anderson``) on the mixed iterate.

The specialized steady-state solver in ``qpsim.services.steady_state``
does *not* call this function directly — it embeds a more elaborate
version inline that also tracks branch-collapse of the ``(f, n_ph)``
coupled system. ``picard_iterate`` here is the plain primitive for
use elsewhere (coupled solves, rate-equation iterations, etc.).
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
        The normalized fixed-point residual
        ``max_i |G(x_k) - x_k| / (max(|x_k|, |G(x_k)|) + tol)`` at
        termination. It is evaluated before mixing, so under-relaxation
        cannot make an unconverged map appear converged.
    """

    n_iter: int
    converged: bool
    final_residual: float


def _validated_real_finite_array(
    values: object,
    *,
    context: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Return a float array after validating a Picard API boundary."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be a finite real numeric array.") from exc
    if expected_shape is not None and raw.shape != expected_shape:
        raise ValueError(
            f"{context} must have shape {expected_shape}; got {raw.shape}."
        )
    if np.iscomplexobj(raw) or not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{context} must be a finite real numeric array.")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{context} must be a finite real numeric array.") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{context} must contain only finite values.")
    return result


def picard_iterate(
    x0: np.ndarray,
    g: Callable[[np.ndarray], np.ndarray],
    *,
    mixing: float = 0.3,
    anderson_depth: int = 0,
    tol: float = 1e-10,
    max_iter: int = 200,
    clip_non_negative: bool = False,
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
        L∞ tolerance on the normalized fixed-point residual (see
        :class:`PicardInfo`), evaluated before under-relaxation. The
        ``+ tol`` regularizer in the denominator — which is what keeps the
        ``x = G(x) = 0`` case from dividing by zero — makes the test mixed
        rather than purely relative: it is exactly equivalent to
        ``|G(x) − x| < tol·max(|x|, |G(x)|) + tol²``, so the effective
        relative tolerance is ``tol·max(1, tol/|x|)``. It is relative with
        tolerance ``tol`` only for components of magnitude ≳ ``tol``, and
        degrades to an absolute floor of ``tol²`` well below that, so the
        verdict is not invariant under a rescaling of the unknown. For
        states whose components sit orders of magnitude below ``tol``, use
        the explicit two-tolerance (``atol``/``rtol``) test plus normwise
        guard in ``qpsim.services.steady_state`` instead.
    max_iter
        Hard cap on iterations.
    clip_non_negative
        If ``True``, Anderson-accelerated iterates are projected onto
        ``≥ 0`` elementwise. Only enable this for problems whose fixed
        point is known to be non-negative (e.g. phonon occupations).
        Default ``False``; see :func:`anderson_extrapolate`.

    Returns
    -------
    x_star
        The converged iterate (or the last iterate if convergence was
        not reached).
    info
        :class:`PicardInfo` with ``n_iter``, ``converged``, and
        ``final_residual``.

    Notes
    -----
    ``x0`` is flattened once before iteration. Every value returned by ``g``
    must be a finite real numeric array with exactly the same one-dimensional
    shape passed to ``g``; broadcastable or otherwise malformed outputs are
    rejected before residual or update arithmetic.
    """
    if not (np.isfinite(tol) and tol > 0.0):
        raise ValueError(f"tol must be finite and positive; got {tol!r}.")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or max_iter < 1
    ):
        raise ValueError(
            f"max_iter must be a positive integer; got {max_iter!r}."
        )
    if (
        isinstance(anderson_depth, (bool, np.bool_))
        or not isinstance(anderson_depth, (int, np.integer))
        or anderson_depth < 0
    ):
        raise ValueError(
            "anderson_depth must be a non-negative integer; "
            f"got {anderson_depth!r}."
        )

    # mixing must be a genuine under-relaxation factor. At mixing == 0 the
    # update is x_mixed = x exactly, so the change-based residual is 0 on the
    # first pass and the solver reports `converged` without ever consulting
    # g(x) — a false positive. Reject the whole invalid range up front.
    if not (np.isfinite(mixing) and 0.0 < mixing <= 1.0):
        raise ValueError(f"mixing must lie in (0, 1]; got {mixing}.")

    x_initial = _validated_real_finite_array(x0, context="x0")
    if x_initial.size == 0:
        raise ValueError("x0 must be non-empty.")
    x = x_initial.ravel()
    use_anderson = anderson_depth > 0
    X_hist: list[np.ndarray] = []
    G_hist: list[np.ndarray] = []
    final_residual = float("inf")

    for it in range(1, max_iter + 1):
        gx = _validated_real_finite_array(
            # The map is an API boundary, not an owner of our live iterate.
            # Pass a copy so an in-place implementation cannot alias ``gx``
            # and ``x`` and erase its own fixed-point defect.
            g(x.copy()),
            context="g(x) output",
            expected_shape=x.shape,
        )
        x_mixed = (1.0 - mixing) * x + mixing * gx

        # Convergence belongs to the original fixed-point equation G(x)=x,
        # not to the relaxed step. Measuring x_mixed-x would multiply the
        # residual by ``mixing`` and could falsely accept the first iteration
        # for a sufficiently small (but valid) relaxation factor.
        change = np.abs(gx - x)
        scale = np.maximum(np.abs(x), np.abs(gx)) + tol
        final_residual = float(np.max(change / scale))

        if final_residual < tol:
            # ``final_residual`` certifies the pre-update iterate ``x``.
            # Returning ``x_mixed`` would expose a different, unevaluated
            # state whose fixed-point defect can be arbitrarily larger for a
            # nonlinear map.  Keep the state and diagnostic bound to the same
            # snapshot; a caller that wants one more relaxed step can take it
            # explicitly and re-certify it.
            return x.copy(), PicardInfo(
                n_iter=it,
                converged=True,
                final_residual=final_residual,
            )

        if it == max_iter:
            # The diagnostic above belongs to ``x``.  Do not take and
            # return one final mixed/Anderson step that has never had its
            # fixed-point defect evaluated: for a nonlinear map that
            # successor can be arbitrarily worse than this certified
            # snapshot even though ``converged`` is false.
            return x.copy(), PicardInfo(
                n_iter=it,
                converged=False,
                final_residual=final_residual,
            )

        if use_anderson:
            x_aa = anderson_extrapolate(
                x, x_mixed, X_hist, G_hist, anderson_depth,
                clip_non_negative=clip_non_negative,
            )
            X_hist.append(x.copy())
            G_hist.append(x_mixed.copy())
            if len(X_hist) > anderson_depth + 1:
                X_hist.pop(0)
                G_hist.pop(0)
            x = x_aa if x_aa is not None else x_mixed
        else:
            x = x_mixed

    return x, PicardInfo(n_iter=max_iter, converged=False, final_residual=final_residual)
