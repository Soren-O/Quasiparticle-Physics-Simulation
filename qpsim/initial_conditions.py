from __future__ import annotations

import math
from textwrap import indent

import numpy as np

from .models import InitialConditionSpec


def default_initial_condition() -> InitialConditionSpec:
    return InitialConditionSpec(
        kind="gaussian",
        params={"amplitude": 1.0, "x0": 0.5, "y0": 0.5, "sigma": 0.12},
        custom_body="return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)",
        custom_params={},
    )


def _compile_custom_expression(body: str):
    source = "def user_expression(x, y, params):\n" + indent(body.strip() or "return 0.0", "    ") + "\n"
    safe_globals = {
        "__builtins__": {},
        "np": np,
        "math": math,
        "abs": abs,
        "min": min,
        "max": max,
        "pow": pow,
    }
    local_ns: dict[str, object] = {}
    exec(source, safe_globals, local_ns)
    fn = local_ns.get("user_expression")
    if fn is None:
        raise ValueError("Custom initial condition did not define user_expression.")
    return fn


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
