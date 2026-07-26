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


class AndersonAccelerationError(Exception):
    """Anderson-only numerical failure on otherwise finite iterate data.

    This deliberately does not subclass :class:`ValueError` or
    :class:`RuntimeError`. A safeguarded caller can catch this exact type and
    retry without acceleration while preserving ordinary map/configuration
    ``ValueError`` and solver-convergence ``RuntimeError`` meanings.
    """


def _as_real_float_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Convert one Anderson input only after rejecting complex values."""
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(
            f"Anderson {name} must be real-valued; complex arrays are not "
            "supported."
        )
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Anderson {name} must be a real-valued numeric array."
        ) from error


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
    returns ``None`` when history is insufficient.

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
    Raises
    ------
    ValueError
        If current/history arrays are inconsistent or non-finite. These are
        invalid fixed-point-map inputs, not acceleration failures.
    AndersonAccelerationError
        If finite inputs overflow during Anderson arithmetic, the
        least-squares solve fails, or it produces a non-finite extrapolate.
    """
    if (
        isinstance(depth, (bool, np.bool_))
        or not isinstance(depth, (int, np.integer))
        or depth < 0
    ):
        raise ValueError(
            "Anderson depth must be a non-negative integer; "
            f"got {depth!r}."
        )
    if depth == 0:
        return None
    if len(X_hist) != len(G_hist):
        raise ValueError(
            "Anderson X_hist and G_hist must have the same length; got "
            f"{len(X_hist)} and {len(G_hist)}."
        )
    x_arr = _as_real_float_array(x, name="x")
    gx_arr = _as_real_float_array(gx, name="gx")
    if x_arr.ndim != 1 or gx_arr.shape != x_arr.shape:
        raise ValueError(
            "Anderson x and gx must be one-dimensional arrays with matching "
            f"shapes; got {x_arr.shape} and {gx_arr.shape}."
        )
    all_history_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (x_prev, gx_prev) in enumerate(
        zip(X_hist, G_hist, strict=True)
    ):
        x_prev_arr = _as_real_float_array(
            x_prev,
            name=f"X_hist[{index}]",
        )
        gx_prev_arr = _as_real_float_array(
            gx_prev,
            name=f"G_hist[{index}]",
        )
        if x_prev_arr.shape != x_arr.shape or gx_prev_arr.shape != x_arr.shape:
            raise ValueError(
                "Anderson history arrays must match the current iterate "
                f"shape {x_arr.shape}; history pair {index} has shapes "
                f"{x_prev_arr.shape} and {gx_prev_arr.shape}."
            )
        all_history_pairs.append((x_prev_arr, gx_prev_arr))
    all_arrays = (
        x_arr,
        gx_arr,
        *(array for pair in all_history_pairs for array in pair),
    )
    if any(np.any(~np.isfinite(array)) for array in all_arrays):
        raise ValueError(
            "Anderson current and history arrays must contain only finite "
            "values."
        )
    m = min(depth, len(all_history_pairs))
    if m < 1:
        return None
    history_pairs = all_history_pairs[-m:]

    # Residuals r_k = G(x_k) − x_k.
    with np.errstate(over="ignore", invalid="ignore"):
        r_curr = gx_arr - x_arr
        R_prev = [gx_prev - x_prev for x_prev, gx_prev in history_pairs]

    # Difference matrix ΔR[:, k] = r_{k+1} − r_k (consecutive residuals).
    with np.errstate(over="ignore", invalid="ignore"):
        cols = [R_prev[k + 1] - R_prev[k] for k in range(m - 1)]
        cols.append(r_curr - R_prev[-1])
    dR = np.column_stack(cols) if len(cols) > 1 else cols[0][:, None]
    if np.any(~np.isfinite(dR)):
        raise AndersonAccelerationError(
            "Anderson residual-difference arithmetic produced non-finite "
            "values from finite inputs."
        )

    try:
        theta, _, _, _ = np.linalg.lstsq(dR, r_curr, rcond=1e-10)
    except np.linalg.LinAlgError as error:
        raise AndersonAccelerationError(
            "Anderson least-squares solve failed numerically."
        ) from error
    if np.any(~np.isfinite(theta)):
        raise AndersonAccelerationError(
            "Anderson least-squares solve produced non-finite coefficients."
        )

    # ΔG: differences of fixed-point outputs, same column layout as dR.
    g_history = [gx_prev for _x_prev, gx_prev in history_pairs]
    with np.errstate(over="ignore", invalid="ignore"):
        dG = [g_history[k + 1] - g_history[k] for k in range(m - 1)]
        dG.append(gx_arr - g_history[-1])
    dG_mat = np.column_stack(dG) if len(dG) > 1 else dG[0][:, None]
    if np.any(~np.isfinite(dG_mat)):
        raise AndersonAccelerationError(
            "Anderson map-difference arithmetic produced non-finite values "
            "from finite inputs."
        )

    with np.errstate(over="ignore", invalid="ignore"):
        x_aa = gx_arr - dG_mat @ theta
    if np.any(~np.isfinite(x_aa)):
        raise AndersonAccelerationError(
            "Anderson extrapolation produced a non-finite iterate."
        )
    if clip_non_negative:
        return np.maximum(x_aa, 0.0)
    return x_aa
