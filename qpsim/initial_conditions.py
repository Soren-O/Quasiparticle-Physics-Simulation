from __future__ import annotations

from typing import Any

import numpy as np

from .models import InitialConditionSpec
from .safe_eval import compile_safe_expression

_DEFAULT_SPATIAL_CUSTOM_BODY = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)"
_DEFAULT_ENERGY_CUSTOM_BODY = "return np.ones_like(E)"


def default_initial_condition() -> InitialConditionSpec:
    return InitialConditionSpec(
        kind="gaussian",
        params={"amplitude": 1.0, "x0": 0.5, "y0": 0.5, "sigma": 0.12},
        custom_body=_DEFAULT_SPATIAL_CUSTOM_BODY,
        custom_params={},
        spatial_kind="gaussian",
        spatial_params={"amplitude": 1.0, "x0": 0.5, "y0": 0.5, "sigma": 0.12},
        spatial_custom_body=_DEFAULT_SPATIAL_CUSTOM_BODY,
        spatial_custom_params={},
        energy_kind="dos",
        energy_params={},
        energy_custom_body=_DEFAULT_ENERGY_CUSTOM_BODY,
        energy_custom_params={},
    )


def _resolve_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return normalized spatial IC tuple (kind, params, custom_body, custom_params)."""
    spatial_kind = str(getattr(spec, "spatial_kind", "") or "").strip().lower()
    spatial_params = dict(getattr(spec, "spatial_params", {}) or {})
    spatial_custom_body = str(
        getattr(spec, "spatial_custom_body", "") or _DEFAULT_SPATIAL_CUSTOM_BODY
    )
    spatial_custom_params = dict(getattr(spec, "spatial_custom_params", {}) or {})

    if spatial_kind == "fermi_dirac":
        amplitude = float(spatial_params.get("amplitude", spatial_params.get("value", 1.0)))
        spatial_kind = "uniform"
        spatial_params = {"value": amplitude}

    if spatial_kind:
        return spatial_kind, spatial_params, spatial_custom_body, spatial_custom_params

    # Legacy mapping
    legacy_kind = str(getattr(spec, "kind", "gaussian") or "gaussian").strip().lower()
    legacy_params = dict(getattr(spec, "params", {}) or {})
    legacy_custom_body = str(
        getattr(spec, "custom_body", "") or _DEFAULT_SPATIAL_CUSTOM_BODY
    )
    legacy_custom_params = dict(getattr(spec, "custom_params", {}) or {})
    if legacy_kind == "fermi_dirac":
        # Legacy fermi_dirac used a uniform spatial profile + thermal energy profile.
        amplitude = float(legacy_params.get("amplitude", 1.0))
        return "uniform", {"value": amplitude}, legacy_custom_body, legacy_custom_params
    return legacy_kind, legacy_params, legacy_custom_body, legacy_custom_params


def _resolve_energy_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return normalized energy IC tuple (kind, params, custom_body, custom_params)."""
    energy_kind = str(getattr(spec, "energy_kind", "") or "").strip().lower()
    energy_params = dict(getattr(spec, "energy_params", {}) or {})
    energy_custom_body = str(getattr(spec, "energy_custom_body", "") or _DEFAULT_ENERGY_CUSTOM_BODY)
    energy_custom_params = dict(getattr(spec, "energy_custom_params", {}) or {})
    if energy_kind:
        return energy_kind, energy_params, energy_custom_body, energy_custom_params

    # Legacy mapping
    legacy_kind = str(getattr(spec, "kind", "gaussian") or "gaussian").strip().lower()
    legacy_params = dict(getattr(spec, "params", {}) or {})
    if legacy_kind == "fermi_dirac":
        return "fermi_dirac", {"temperature": float(legacy_params.get("temperature", 0.1))}, _DEFAULT_ENERGY_CUSTOM_BODY, {}
    return "dos", {}, _DEFAULT_ENERGY_CUSTOM_BODY, {}


def resolve_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical spatial initial-condition fields."""
    return _resolve_spatial_spec(spec)


def resolve_energy_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical energy initial-condition fields."""
    return _resolve_energy_spec(spec)


def canonicalize_initial_condition(spec: InitialConditionSpec) -> InitialConditionSpec:
    """Normalize initial-condition data into split spatial/energy fields.

    Legacy fields remain populated for compatibility, but they mirror the
    spatial portion of the canonical representation.
    """
    spatial_kind, spatial_params, spatial_custom_body, spatial_custom_params = _resolve_spatial_spec(spec)
    energy_kind, energy_params, energy_custom_body, energy_custom_params = _resolve_energy_spec(spec)
    return InitialConditionSpec(
        kind=spatial_kind,
        params=dict(spatial_params),
        custom_body=spatial_custom_body,
        custom_params=dict(spatial_custom_params),
        spatial_kind=spatial_kind,
        spatial_params=dict(spatial_params),
        spatial_custom_body=spatial_custom_body,
        spatial_custom_params=dict(spatial_custom_params),
        energy_kind=energy_kind,
        energy_params=dict(energy_params),
        energy_custom_body=energy_custom_body,
        energy_custom_params=dict(energy_custom_params),
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
    kind, params, custom_body, custom_params = _resolve_spatial_spec(spec)

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
        fn = _compile_custom_expression(custom_body)
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
        raise ValueError(f"Unsupported spatial initial-condition kind: '{kind}'.")

    field[~mask] = 0.0
    return field


def build_initial_energy_weights(
    E_bins: np.ndarray,
    gap: float,
    dynes_gamma: float,
    spec: InitialConditionSpec,
    bath_temperature: float,
) -> np.ndarray | None:
    """Build initial energy weights from IC spec.

    Returns
    -------
    None:
        Use solver default DOS weighting.
    1D ndarray:
        Explicit unnormalized weights over energy bins.
    """
    energy_kind, energy_params, energy_custom_body, energy_custom_params = _resolve_energy_spec(spec)
    energy_kind = energy_kind.strip().lower()
    if energy_kind in {"", "dos", "default", "bcs_dos"}:
        return None

    if energy_kind == "fermi_dirac":
        from .solver import thermal_qp_weights  # local import avoids circular dependency

        temp = float(energy_params.get("temperature", bath_temperature))
        return thermal_qp_weights(E_bins, gap, temp, dynes_gamma)

    if energy_kind == "uniform":
        value = float(energy_params.get("value", 1.0))
        if value < 0:
            raise ValueError("Uniform energy profile value must be non-negative.")
        return np.full_like(E_bins, value, dtype=float)

    if energy_kind == "custom":
        fn = compile_safe_expression(
            energy_custom_body.strip() or _DEFAULT_ENERGY_CUSTOM_BODY,
            variable_names=("E", "gap", "params"),
        )
        params = dict(energy_custom_params or {})
        try:
            val = fn(E=np.asarray(E_bins, dtype=float), gap=float(gap), params=params)
            arr = np.asarray(val, dtype=float)
        except Exception:
            arr = np.asarray(
                [float(fn(E=float(E), gap=float(gap), params=params)) for E in np.asarray(E_bins, dtype=float)],
                dtype=float,
            )
        arr = np.asarray(arr, dtype=float).reshape(-1)
        if arr.size == 1:
            arr = np.full_like(E_bins, float(arr[0]), dtype=float)
        if arr.size != E_bins.size:
            raise ValueError(
                f"Custom energy profile must return {E_bins.size} values or a scalar; got {arr.size}."
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("Custom energy profile produced non-finite values.")
        if np.any(arr < 0):
            raise ValueError("Custom energy profile must be non-negative.")
        return arr

    raise ValueError(
        f"Unsupported energy initial-condition kind '{energy_kind}'. "
        "Supported: dos, fermi_dirac, uniform, custom."
    )
