"""Exponential-integrator time steppers for the collision operator.

Provides ``etd1_step`` (first-order exponential Euler) and
``etd2_step`` (second-order Heun-type ETD). Both are specialized to
``∂_t x = gain(x) − loss_rate(x) · x`` with ``loss_rate ≥ 0``; the
linear relaxation in ``loss_rate`` is handled exactly, and the
nonlinear ``gain`` is discretized to first- or second-order.

``apply_phonon_collision`` in :mod:`qpsim.collisions.phonon` steps
with ETD2.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_LOSS_STEP_SAFETY = 0.95
_MAX_ETD_ACCEPTED_SUBSTEPS = 1_000_000
_MAX_ETD_REJECTED_ATTEMPTS = 1_000_000


def _sum_balance(values: np.ndarray, axis: int | None) -> np.ndarray | float:
    """Sum a balance slice while preserving a requested axis dimension."""
    if axis is None:
        return float(np.sum(values))
    return np.sum(values, axis=axis, keepdims=True)


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
    f_arr = np.asarray(f, dtype=float)
    gain_arr = np.asarray(gain, dtype=float)
    loss_arr = np.asarray(loss_rate, dtype=float)
    if f_arr.shape != gain_arr.shape or f_arr.shape != loss_arr.shape:
        raise ValueError(
            "f, gain, and loss_rate must have the same shape; got "
            f"{f_arr.shape}, {gain_arr.shape}, and {loss_arr.shape}."
        )
    if np.any(~np.isfinite(f_arr)):
        raise ValueError("f must contain only finite values.")
    if np.any(~np.isfinite(gain_arr)):
        raise ValueError("gain must contain only finite values.")
    if np.any(~np.isfinite(loss_arr)):
        raise ValueError("loss_rate must contain only finite values.")
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError(f"dt must be finite and non-negative; got {dt}.")

    mu = np.maximum(loss_arr, 0.0)
    p_term = np.maximum(gain_arr + (mu - loss_arr) * f_arr, 0.0)

    decay = np.exp(-mu * dt)
    # coeff = (1 - e^{-mu dt}) / mu, the exact ETD1 weight on the source
    # term. Compute it with expm1 so it stays accurate when mu*dt is tiny:
    # forming 1 - exp(-mu dt) directly cancels to 0 once exp(-mu dt) rounds
    # to 1.0 (mu*dt below ~1e-16), which silently zeroed the entire gain
    # contribution for small-but-finite mu. In the mu -> 0 limit coeff -> dt;
    # the inner np.where keeps the division well-defined at mu == 0.
    positive = mu > 0.0
    coeff = np.where(positive, -np.expm1(-mu * dt) / np.where(positive, mu, 1.0), dt)

    updated = decay * f_arr + coeff * p_term
    return np.clip(updated, 0.0, 1.0)


def etd2_step(
    f: np.ndarray,
    rhs: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    dt: float,
    *,
    balance_weights: np.ndarray | None = None,
    balance_axis: int | None = None,
    max_loss_step: float | None = None,
    _diagnostics: dict[str, int] | None = None,
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
        Invoked twice per accepted substep, with additional predictor calls
        when a stage-aware rate-limit trial is rejected.
    dt
        Time step (ns).
    balance_weights
        Optional non-negative weights for a discrete balance law.  When
        supplied, the ETD candidate is projected onto the second-order
        balance

        ``w·f_new = w·f + dt/2 * w·(R(f) + R(f_pred))``.

        Thus a semidiscrete invariant (``w·R == 0``) is preserved to
        roundoff without changing the method's second-order accuracy.  The
        projection rescales occupations or holes and therefore preserves
        ``0 <= f <= 1``.  If a non-conservative source requests a balance
        outside that feasible box, the nearest bound is used (the same box
        constraint already enforced by ETD1).  The default ``None`` retains
        the historical ETD2 behavior exactly.
    balance_axis
        Axis along which to enforce independent weighted balances.  For a
        spatial occupation shaped ``(NE, Nx)``, ``balance_axis=0`` prevents
        mass-error compensation between spatial columns.  ``None`` (default)
        treats the whole array as one balance, preserving the one-dimensional
        behavior.
    max_loss_step
        Optional stage-aware subcycling bound on
        ``max(loss_n, loss_predictor) * sub_dt``. A value such as ``0.25``
        prevents a nominally large ETD step from accumulating a large balance
        projection. A predictor that reveals a stiffer nonlinear loss rejects
        the trial and retries from the unchanged state at a shorter step.
        ``None`` disables subcycling.
    _diagnostics
        Internal mutable output. When provided, receives the exact counts
        ``accepted_substeps`` and ``rejected_attempts``. Rejected predictor
        trials are never reported as completed integration work.

    Returns
    -------
    np.ndarray
        Updated occupation, clipped to ``[0, 1]``.
    """
    if not np.isfinite(dt) or dt < 0.0:
        raise ValueError(f"dt must be finite and non-negative; got {dt}.")
    if max_loss_step is not None and (
        not np.isfinite(max_loss_step) or max_loss_step <= 0.0
    ):
        raise ValueError(
            "max_loss_step must be finite and positive when provided."
        )

    weights: np.ndarray | None = None
    axis: int | None = None
    if balance_weights is not None:
        weights = np.asarray(balance_weights, dtype=float)
        if weights.shape != np.asarray(f).shape:
            raise ValueError(
                "balance_weights must have the same shape as f; got "
                f"{weights.shape} and {np.asarray(f).shape}."
            )
        if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("balance_weights must be finite and non-negative.")
        if not np.any(weights > 0.0):
            raise ValueError("balance_weights must contain a positive entry.")
        if balance_axis is not None:
            if not isinstance(balance_axis, int) or isinstance(balance_axis, bool):
                raise ValueError("balance_axis must be an integer or None.")
            if not -weights.ndim <= balance_axis < weights.ndim:
                raise ValueError(
                    f"balance_axis={balance_axis} is out of bounds for "
                    f"an array with {weights.ndim} dimensions."
                )
            axis = balance_axis % weights.ndim
            supported_per_slice = np.sum(weights > 0.0, axis=axis)
            if np.any(supported_per_slice == 0):
                raise ValueError(
                    "Every balance slice must contain a positive weight."
                )
    elif balance_axis is not None:
        raise ValueError("balance_axis requires balance_weights.")

    if dt == 0.0:
        if _diagnostics is not None:
            _diagnostics["accepted_substeps"] = 0
            _diagnostics["rejected_attempts"] = 0
        return np.clip(np.asarray(f, dtype=float), 0.0, 1.0).copy()

    current = np.asarray(f, dtype=float).copy()
    remaining = float(dt)
    accepted_substeps = 0
    rejected_attempts = 0
    while remaining > 0.0:
        gain_n, loss_n = rhs(current)
        gain_n = np.asarray(gain_n, dtype=float)
        loss_n = np.asarray(loss_n, dtype=float)
        step_dt = remaining
        if max_loss_step is not None:
            max_loss = float(np.max(np.maximum(loss_n, 0.0)))
            if not np.isfinite(max_loss):
                raise ValueError("rhs returned a non-finite loss rate.")
            if max_loss > 0.0:
                step_dt = min(
                    step_dt,
                    _LOSS_STEP_SAFETY * max_loss_step / max_loss,
                )
            if not np.isfinite(step_dt) or step_dt <= 0.0:
                raise RuntimeError(
                    "ETD2 rate-based subcycling could not form a positive "
                    "finite substep; reduce dt or inspect the collision rates."
                )

        rejected_this_substep = 0
        while True:
            f_pred = etd1_step(current, gain_n, loss_n, step_dt)
            gain_p, loss_p = rhs(f_pred)
            gain_p = np.asarray(gain_p, dtype=float)
            loss_p = np.asarray(loss_p, dtype=float)

            if max_loss_step is None:
                break
            max_predictor_loss = float(np.max(np.maximum(loss_p, 0.0)))
            if not np.isfinite(max_predictor_loss):
                raise ValueError("rhs returned a non-finite predictor loss rate.")
            max_stage_loss = max(max_loss, max_predictor_loss)
            allowed_loss = max_loss_step / step_dt
            if max_stage_loss <= allowed_loss:
                break

            rejected_attempts += 1
            rejected_this_substep += 1
            if rejected_attempts > _MAX_ETD_REJECTED_ATTEMPTS:
                raise RuntimeError(
                    "ETD2 rate-based subcycling exceeded 1,000,000 rejected "
                    "predictor trials; reduce dt or inspect the collision rates."
                )

            ratio = allowed_loss / max_stage_loss
            # A predictor-created rate commonly grows linearly with trial dt,
            # so the first retry uses the square-root solution of
            # loss_p(h)*h <= bound. If that model is insufficient, a direct
            # rate-ratio shrink on later retries is conservative for a locally
            # constant rate. Every retry is re-evaluated, so neither heuristic
            # can bypass the hard two-stage acceptance test above.
            factor = (
                _LOSS_STEP_SAFETY * np.sqrt(ratio)
                if rejected_this_substep == 1
                else _LOSS_STEP_SAFETY * ratio
            )
            new_step_dt = step_dt * float(factor)
            if (
                not np.isfinite(new_step_dt)
                or new_step_dt <= 0.0
                or new_step_dt >= step_dt
            ):
                raise RuntimeError(
                    "ETD2 rate-based retry could not form a smaller positive "
                    "substep; reduce dt or inspect the collision rates."
                )
            step_dt = new_step_dt

        gain_avg = 0.5 * (gain_n + gain_p)
        loss_avg = 0.5 * (loss_n + loss_p)
        updated = etd1_step(current, gain_avg, loss_avg, step_dt)

        if weights is not None:
            residual_n = gain_n - loss_n * current
            residual_p = gain_p - loss_p * f_pred
            target = (
                _sum_balance(weights * current, axis)
                + 0.5
                * step_dt
                * _sum_balance(weights * (residual_n + residual_p), axis)
            )
            updated = _project_weighted_balance(
                updated,
                weights,
                target,
                axis=axis,
            )

        current = updated
        new_remaining = remaining - step_dt
        # Eliminate a final ulp-sized iteration after repeated substeps.
        remaining = (
            0.0
            if new_remaining <= 16.0 * np.finfo(float).eps * max(1.0, dt)
            else new_remaining
        )
        accepted_substeps += 1
        if accepted_substeps > _MAX_ETD_ACCEPTED_SUBSTEPS:
            raise RuntimeError(
                "ETD2 rate-based subcycling exceeded 1,000,000 substeps; "
                "reduce dt or inspect the collision rates."
            )

    if _diagnostics is not None:
        _diagnostics["accepted_substeps"] = accepted_substeps
        _diagnostics["rejected_attempts"] = rejected_attempts
    return current


def _project_weighted_balance(
    candidate: np.ndarray,
    weights: np.ndarray,
    target: np.ndarray | float,
    *,
    axis: int | None = None,
) -> np.ndarray:
    """Project a bounded occupation onto ``weights @ f == target``.

    Excess mass is removed by uniformly scaling occupations; missing mass is
    added by uniformly scaling holes.  Both branches are positivity preserving
    and avoid selecting an arbitrary energy bin for a correction.
    """
    total_weight = _sum_balance(weights, axis)
    target_array = np.asarray(target, dtype=float)
    if np.any(~np.isfinite(target_array)):
        raise ValueError("The requested weighted balance must be finite.")
    # ETD1 already evolves on the box 0 <= f <= 1.  A sufficiently strong
    # non-conservative source can ask the Heun balance for more mass than the
    # box can hold (or less than zero); project that request to the nearest
    # feasible face.  Raw-RHS convergence checks still see the uncompensated
    # outward source and therefore cannot mistake saturation for equilibrium.
    target_array = np.clip(target_array, 0.0, total_weight)

    projected = np.clip(np.asarray(candidate, dtype=float), 0.0, 1.0).copy()
    supported = weights > 0.0
    current = _sum_balance(weights * projected, axis)
    # Keep the roundoff allowance local to each independent balance slice.
    # A single scalar based on the largest column made the projection depend
    # on unrelated columns: a 1e16-mass slice could hide an O(1) defect in a
    # neighbouring slice.  Do not impose an absolute floor of one either;
    # doing so made the decision change when all quadrature weights were
    # rescaled.  ``current`` and ``target`` are non-negative weighted sums, so
    # their local magnitude is the appropriate floating-point scale.
    balance_tolerance = 32.0 * np.finfo(float).eps * np.maximum(
        np.abs(current), np.abs(target_array),
    )
    difference = current - target_array
    if np.all(np.abs(difference) <= balance_tolerance):
        return projected

    remove = difference > balance_tolerance
    add = difference < -balance_tolerance

    remove_scale = np.ones_like(current, dtype=float)
    np.divide(
        target_array,
        current,
        out=remove_scale,
        where=remove & (current > 0.0),
    )
    empty = total_weight - current
    hole_scale = np.ones_like(current, dtype=float)
    np.divide(
        total_weight - target_array,
        empty,
        out=hole_scale,
        where=add & (empty > 0.0),
    )

    # Restrict the correction to represented states.  Zero-weight storage
    # cells (e.g. pure-BCS cells below a moving gap) retain their candidate
    # value exactly and can never be changed by a balance projection.
    projected = np.where(remove & supported, projected * remove_scale, projected)
    projected = np.where(
        add & supported,
        1.0 - (1.0 - projected) * hole_scale,
        projected,
    )

    return np.clip(projected, 0.0, 1.0)
