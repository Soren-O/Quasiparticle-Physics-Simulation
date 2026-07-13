"""Anderson (Type-II) acceleration for fixed-point iterations.

Given a history of iterates ``x_k`` and their fixed-point-map images
``G(x_k)``, computes an extrapolated iterate by solving a small
least-squares problem in the residual differences. The
least-squares-on-residual-differences form (with the update
``x = G(x_k) − ΔG·θ``) is the Type-II / "bad Broyden" Anderson update
(Walker & Ni 2011); it is a valid acceleration regardless of the label.

Ported from ``_anderson_extrapolate`` in
``qpsim/numerics/steady_state.py``.
"""

from __future__ import annotations

import numpy as np


def anderson_extrapolate(
    x: np.ndarray,
    gx: np.ndarray,
    X_hist: list[np.ndarray],
    G_hist: list[np.ndarray],
    depth: int,
    *,
    clip_non_negative: bool = False,
) -> np.ndarray | None:
    """Type-II Anderson extrapolation on the raw fixed-point map.

    Given the current iterate ``x``, its image ``gx = G(x)``, and the
    history of prior ``(x_k, G(x_k))`` pairs, returns an extrapolated
    iterate that tends to converge faster than plain Picard. Requires
    at least ``m = min(depth, len(X_hist)) ≥ 1`` history entry;
    returns ``None`` when history is insufficient or the least-squares
    solve is degenerate.

    Parameters
    ----------
    x, gx
        Current iterate and its fixed-point-map image.
    X_hist, G_hist
        Histories of previous iterates and their images.
    depth
        Maximum history window size to use.
    clip_non_negative
        If ``True``, the returned iterate is projected onto ``≥ 0``
        elementwise. Set this only when the fixed point is known to be
        non-negative (e.g. phonon occupations). The default ``False``
        is the mathematically-correct choice for arbitrary fixed-point
        problems — clipping breaks convergence to any fixed point with
        a negative component.
    """
    m = min(depth, len(X_hist))
    if m < 1:
        return None

    # Residuals r_k = G(x_k) − x_k.
    r_curr = gx - x
    R_prev = [G_hist[-i] - X_hist[-i] for i in range(m, 0, -1)]

    # Difference matrix ΔR[:, k] = r_{k+1} − r_k (consecutive residuals).
    cols = [R_prev[k + 1] - R_prev[k] for k in range(m - 1)]
    cols.append(r_curr - R_prev[-1])
    dR = np.column_stack(cols) if len(cols) > 1 else cols[0][:, None]

    try:
        theta, _, _, _ = np.linalg.lstsq(dR, r_curr, rcond=1e-10)
    except np.linalg.LinAlgError:
        return None

    # ΔG: differences of fixed-point outputs, same column layout as dR.
    dG = [G_hist[-m + k + 1] - G_hist[-m + k] for k in range(m - 1)]
    dG.append(gx - G_hist[-1])
    dG_mat = np.column_stack(dG) if len(dG) > 1 else dG[0][:, None]

    x_aa = gx - dG_mat @ theta
    if clip_non_negative:
        return np.maximum(x_aa, 0.0)
    return x_aa
