from __future__ import annotations

import numpy as np

from .models import InitialConditionSpec
from .safe_eval import compile_safe_expression


def default_initial_condition() -> InitialConditionSpec:
    return InitialConditionSpec(
        kind="gaussian",
        params={"amplitude": 1.0, "x0": 0.5, "y0": 0.5, "sigma": 0.12},
        custom_body="return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)",
        custom_params={},
    )


def _compile_custom_expression(body: str):
    evaluator = compile_safe_expression(body, variable_names=("x", "y", "params"))

    def user_expression(x, y, params):
        return evaluator(x=x, y=y, params=params)

    return user_expression


def _evaluate_custom_expression_vectorized(
    fn,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    mask: np.ndarray,
    custom_params: dict,
) -> np.ndarray | None:
    masked_x = x_norm[mask]
    masked_y = y_norm[mask]
    if masked_x.size == 0:
        return np.empty((0,), dtype=float)

    try:
        vector_value = fn(masked_x, masked_y, custom_params)
        arr = np.asarray(vector_value, dtype=float)
    except Exception:
        return None

    if arr.ndim == 0:
        return np.full(masked_x.shape[0], float(arr), dtype=float)
    if arr.size == masked_x.size:
        return arr.reshape(masked_x.size)
    if arr.shape == mask.shape:
        return np.asarray(arr[mask], dtype=float)
    return None


def evaluate_gap_expression(
    expression: str,
    mask: np.ndarray,
    energy_gap_default: float,
) -> np.ndarray:
    """Evaluate a spatially varying gap expression over interior pixels.

    Parameters
    ----------
    expression : Python expression body with variables x, y (normalized 0..1).
                 Empty string returns uniform array of *energy_gap_default*.
    mask : 2D boolean array
    energy_gap_default : default gap value Δ in μeV

    Returns
    -------
    1D array of gap values, one per True pixel in *mask* (row-major order).
    """
    n_interior = int(np.sum(mask))
    def _validate_gap_values(values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != n_interior:
            raise ValueError(
                f"Gap expression returned {arr.size} values; expected {n_interior} interior pixels."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Gap expression produced non-finite values.")
        if np.any(arr <= 0.0):
            raise ValueError("Gap expression must produce strictly positive values.")
        return arr

    if not expression.strip():
        return _validate_gap_values(np.full(n_interior, energy_gap_default, dtype=float))

    fn = _compile_custom_expression(expression)
    ny, nx = mask.shape
    y_idx, x_idx = np.indices(mask.shape)
    x_norm = (x_idx + 0.5) / max(1, nx)
    y_norm = (y_idx + 0.5) / max(1, ny)

    vector_result = _evaluate_custom_expression_vectorized(
        fn=fn,
        x_norm=x_norm,
        y_norm=y_norm,
        mask=mask,
        custom_params={},
    )
    if vector_result is not None:
        return _validate_gap_values(vector_result)

    # Scalar fallback
    coords = np.argwhere(mask)
    result = np.empty(n_interior, dtype=float)
    for idx, (row, col) in enumerate(coords):
        result[idx] = float(fn(float(x_norm[row, col]), float(y_norm[row, col]), {}))
    return _validate_gap_values(result)


def build_initial_field(mask: np.ndarray, spec: InitialConditionSpec) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Geometry mask must be 2D.")

    ny, nx = mask.shape
    y_idx, x_idx = np.indices(mask.shape)
    x_norm = (x_idx + 0.5) / max(1, nx)
    y_norm = (y_idx + 0.5) / max(1, ny)

    field = np.zeros(mask.shape, dtype=float)
    kind = spec.kind.strip().lower()
    params = dict(spec.params or {})

    if kind == "gaussian":
        amplitude = float(params.get("amplitude", 1.0))
        x0 = float(params.get("x0", 0.5))
        y0 = float(params.get("y0", 0.5))
        sigma = max(1e-6, float(params.get("sigma", 0.12)))
        rr = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
        field = amplitude * np.exp(-rr / (2.0 * sigma * sigma))
    elif kind == "uniform":
        value = float(params.get("value", 1.0))
        field.fill(value)
    elif kind == "point":
        value = float(params.get("value", 1.0))
        x0 = float(params.get("x0", 0.5))
        y0 = float(params.get("y0", 0.5))
        col = int(np.clip(round(x0 * (nx - 1)), 0, nx - 1))
        row = int(np.clip(round(y0 * (ny - 1)), 0, ny - 1))
        if mask[row, col]:
            field[row, col] = value
        else:
            inside = np.argwhere(mask)
            if inside.size:
                d2 = (inside[:, 0] - row) ** 2 + (inside[:, 1] - col) ** 2
                nearest = inside[int(np.argmin(d2))]
                field[int(nearest[0]), int(nearest[1])] = value
    elif kind == "fermi_dirac":
        # Spatially uniform field; energy distribution is handled by the solver
        # via thermal_qp_weights.  The amplitude here sets the total spatial
        # density before energy redistribution.
        amplitude = float(params.get("amplitude", 1.0))
        field.fill(amplitude)
    elif kind == "custom":
        fn = _compile_custom_expression(spec.custom_body)
        custom_params = dict(spec.custom_params or {})
        vector_values = _evaluate_custom_expression_vectorized(
            fn=fn,
            x_norm=x_norm,
            y_norm=y_norm,
            mask=mask,
            custom_params=custom_params,
        )
        if vector_values is not None:
            field[mask] = vector_values
        else:
            # Fallback keeps compatibility with scalar-only custom expressions.
            for row, col in np.argwhere(mask):
                value = fn(float(x_norm[row, col]), float(y_norm[row, col]), custom_params)
                field[row, col] = float(value)
    else:
        raise ValueError(f"Unsupported initial condition kind: {spec.kind}")

    field[~mask] = 0.0
    return field
