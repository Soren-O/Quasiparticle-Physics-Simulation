from __future__ import annotations

from typing import Any

import numpy as np

from .models import InitialConditionSpec
from .safe_eval import compile_safe_expression

_DEFAULT_SPATIAL_CUSTOM_BODY = "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02)"
_DEFAULT_ENERGY_CUSTOM_BODY = "return np.ones_like(E)"
_DEFAULT_QP_FULL_CUSTOM_BODY = (
    "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) * np.exp(-E / 500.0)"
)
_DEFAULT_PHONON_SPATIAL_CUSTOM_BODY = "return 1.0"
_DEFAULT_PHONON_ENERGY_CUSTOM_BODY = "return np.ones_like(E)"
_DEFAULT_PHONON_FULL_CUSTOM_BODY = (
    "return np.exp(-((x-0.5)**2 + (y-0.5)**2) / 0.02) * np.exp(-E / 500.0)"
)
_K_B_UEV_PER_K = 86.173303


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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
        qp_full_custom_enabled=False,
        qp_full_custom_body=_DEFAULT_QP_FULL_CUSTOM_BODY,
        qp_full_custom_params={},
        phonon_spatial_kind="uniform",
        phonon_spatial_params={"value": 1.0},
        phonon_spatial_custom_body=_DEFAULT_PHONON_SPATIAL_CUSTOM_BODY,
        phonon_spatial_custom_params={},
        phonon_energy_kind="bose_einstein",
        phonon_energy_params={},
        phonon_energy_custom_body=_DEFAULT_PHONON_ENERGY_CUSTOM_BODY,
        phonon_energy_custom_params={},
        phonon_full_custom_enabled=False,
        phonon_full_custom_body=_DEFAULT_PHONON_FULL_CUSTOM_BODY,
        phonon_full_custom_params={},
    )


def _resolve_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return normalized QP spatial IC tuple (kind, params, custom_body, custom_params)."""
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
    """Return normalized QP energy IC tuple (kind, params, custom_body, custom_params)."""
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
        return (
            "fermi_dirac",
            {"temperature": float(legacy_params.get("temperature", 0.1))},
            _DEFAULT_ENERGY_CUSTOM_BODY,
            {},
        )
    return "dos", {}, _DEFAULT_ENERGY_CUSTOM_BODY, {}


def _resolve_phonon_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    kind = str(getattr(spec, "phonon_spatial_kind", "") or "").strip().lower()
    params = dict(getattr(spec, "phonon_spatial_params", {}) or {})
    custom_body = str(
        getattr(spec, "phonon_spatial_custom_body", "") or _DEFAULT_PHONON_SPATIAL_CUSTOM_BODY
    )
    custom_params = dict(getattr(spec, "phonon_spatial_custom_params", {}) or {})
    if kind:
        return kind, params, custom_body, custom_params
    return "uniform", {"value": 1.0}, _DEFAULT_PHONON_SPATIAL_CUSTOM_BODY, {}


def _resolve_phonon_energy_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    kind = str(getattr(spec, "phonon_energy_kind", "") or "").strip().lower()
    params = dict(getattr(spec, "phonon_energy_params", {}) or {})
    custom_body = str(
        getattr(spec, "phonon_energy_custom_body", "") or _DEFAULT_PHONON_ENERGY_CUSTOM_BODY
    )
    custom_params = dict(getattr(spec, "phonon_energy_custom_params", {}) or {})
    if kind:
        return kind, params, custom_body, custom_params
    return "bose_einstein", {}, _DEFAULT_PHONON_ENERGY_CUSTOM_BODY, {}


def resolve_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical QP spatial initial-condition fields."""
    return _resolve_spatial_spec(spec)


def resolve_energy_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical QP energy initial-condition fields."""
    return _resolve_energy_spec(spec)


def resolve_phonon_spatial_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical phonon spatial initial-condition fields."""
    return _resolve_phonon_spatial_spec(spec)


def resolve_phonon_energy_spec(spec: InitialConditionSpec) -> tuple[str, dict[str, Any], str, dict[str, Any]]:
    """Return canonical phonon energy initial-condition fields."""
    return _resolve_phonon_energy_spec(spec)


def resolve_qp_full_custom_spec(spec: InitialConditionSpec) -> tuple[bool, str, dict[str, Any]]:
    return (
        _as_bool(getattr(spec, "qp_full_custom_enabled", False)),
        str(getattr(spec, "qp_full_custom_body", "") or _DEFAULT_QP_FULL_CUSTOM_BODY),
        dict(getattr(spec, "qp_full_custom_params", {}) or {}),
    )


def resolve_phonon_full_custom_spec(spec: InitialConditionSpec) -> tuple[bool, str, dict[str, Any]]:
    return (
        _as_bool(getattr(spec, "phonon_full_custom_enabled", False)),
        str(getattr(spec, "phonon_full_custom_body", "") or _DEFAULT_PHONON_FULL_CUSTOM_BODY),
        dict(getattr(spec, "phonon_full_custom_params", {}) or {}),
    )


def canonicalize_initial_condition(spec: InitialConditionSpec) -> InitialConditionSpec:
    """Normalize initial-condition data into canonical split fields."""
    spatial_kind, spatial_params, spatial_custom_body, spatial_custom_params = _resolve_spatial_spec(spec)
    energy_kind, energy_params, energy_custom_body, energy_custom_params = _resolve_energy_spec(spec)
    ph_spatial_kind, ph_spatial_params, ph_spatial_custom_body, ph_spatial_custom_params = (
        _resolve_phonon_spatial_spec(spec)
    )
    ph_energy_kind, ph_energy_params, ph_energy_custom_body, ph_energy_custom_params = (
        _resolve_phonon_energy_spec(spec)
    )
    qp_full_enabled, qp_full_body, qp_full_params = resolve_qp_full_custom_spec(spec)
    ph_full_enabled, ph_full_body, ph_full_params = resolve_phonon_full_custom_spec(spec)

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
        qp_full_custom_enabled=bool(qp_full_enabled),
        qp_full_custom_body=qp_full_body,
        qp_full_custom_params=dict(qp_full_params),
        phonon_spatial_kind=ph_spatial_kind,
        phonon_spatial_params=dict(ph_spatial_params),
        phonon_spatial_custom_body=ph_spatial_custom_body,
        phonon_spatial_custom_params=dict(ph_spatial_custom_params),
        phonon_energy_kind=ph_energy_kind,
        phonon_energy_params=dict(ph_energy_params),
        phonon_energy_custom_body=ph_energy_custom_body,
        phonon_energy_custom_params=dict(ph_energy_custom_params),
        phonon_full_custom_enabled=bool(ph_full_enabled),
        phonon_full_custom_body=ph_full_body,
        phonon_full_custom_params=dict(ph_full_params),
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


def _build_spatial_field(
    mask: np.ndarray,
    kind: str,
    params: dict[str, Any],
    custom_body: str,
    custom_params: dict[str, Any],
    *,
    default_uniform: float = 1.0,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Geometry mask must be 2D.")

    ny, nx = mask.shape
    y_idx, x_idx = np.indices(mask.shape)
    x_norm = (x_idx + 0.5) / max(1, nx)
    y_norm = (y_idx + 0.5) / max(1, ny)

    field = np.zeros(mask.shape, dtype=float)
    mode = str(kind or "").strip().lower()
    if mode == "gaussian":
        amplitude = float(params.get("amplitude", 1.0))
        x0 = float(params.get("x0", 0.5))
        y0 = float(params.get("y0", 0.5))
        sigma = max(1e-6, float(params.get("sigma", 0.12)))
        rr = (x_norm - x0) ** 2 + (y_norm - y0) ** 2
        field = amplitude * np.exp(-rr / (2.0 * sigma * sigma))
    elif mode == "uniform":
        value = float(params.get("value", default_uniform))
        field.fill(value)
    elif mode == "point":
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
    elif mode == "custom":
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
            for row, col in np.argwhere(mask):
                value = fn(float(x_norm[row, col]), float(y_norm[row, col]), custom_params)
                field[row, col] = float(value)
    else:
        raise ValueError(f"Unsupported spatial initial-condition kind: '{kind}'.")

    field[~mask] = 0.0
    if not np.all(np.isfinite(field[mask])):
        raise ValueError("Spatial initial-condition profile produced non-finite values.")
    return field


def evaluate_gap_expression(
    expression: str,
    mask: np.ndarray,
    energy_gap_default: float,
) -> np.ndarray:
    """Evaluate a spatially varying gap expression over interior pixels."""
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

    coords = np.argwhere(mask)
    result = np.empty(n_interior, dtype=float)
    for idx, (row, col) in enumerate(coords):
        result[idx] = float(fn(float(x_norm[row, col]), float(y_norm[row, col]), {}))
    return _validate_gap_values(result)


def build_initial_field(mask: np.ndarray, spec: InitialConditionSpec) -> np.ndarray:
    kind, params, custom_body, custom_params = _resolve_spatial_spec(spec)
    return _build_spatial_field(
        mask,
        kind,
        params,
        custom_body,
        custom_params,
        default_uniform=1.0,
    )


def build_initial_phonon_spatial_field(mask: np.ndarray, spec: InitialConditionSpec) -> np.ndarray:
    kind, params, custom_body, custom_params = _resolve_phonon_spatial_spec(spec)
    return _build_spatial_field(
        mask,
        kind,
        params,
        custom_body,
        custom_params,
        default_uniform=1.0,
    )


def build_initial_energy_weights(
    E_bins: np.ndarray,
    gap: float,
    dynes_gamma: float,
    spec: InitialConditionSpec,
    bath_temperature: float,
) -> np.ndarray | None:
    """Build QP energy weights from IC spec."""
    qp_full_enabled, _, _ = resolve_qp_full_custom_spec(spec)
    if qp_full_enabled:
        return None

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


def _coerce_energy_spatial_array(
    arr: np.ndarray,
    energy_bins: np.ndarray,
    mask: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    nE = int(np.asarray(energy_bins).size)
    if mask.ndim != 2:
        raise ValueError("Geometry mask must be 2D.")
    ny, nx = mask.shape
    n_spatial = int(np.sum(mask))

    if arr.ndim == 0:
        return np.full((nE, n_spatial), float(arr), dtype=float)
    if arr.shape == (nE, n_spatial):
        return np.asarray(arr, dtype=float)
    if arr.shape == (n_spatial, nE):
        return np.asarray(arr, dtype=float).T
    if arr.shape == (nE, ny, nx):
        return np.asarray(arr, dtype=float)[:, mask]
    if arr.shape == (ny, nx, nE):
        return np.moveaxis(np.asarray(arr, dtype=float), 2, 0)[:, mask]
    if arr.shape == (ny, nx):
        spatial = np.asarray(arr, dtype=float)[mask]
        return np.repeat(spatial[None, :], nE, axis=0)
    if arr.shape == (nE,):
        energy = np.asarray(arr, dtype=float).reshape(nE, 1)
        return np.repeat(energy, n_spatial, axis=1)
    if arr.shape == (n_spatial,):
        spatial = np.asarray(arr, dtype=float).reshape(1, n_spatial)
        return np.repeat(spatial, nE, axis=0)
    if arr.size == nE * n_spatial:
        return np.asarray(arr, dtype=float).reshape(nE, n_spatial)

    raise ValueError(
        f"{label} expression returned shape {arr.shape}; expected scalar, "
        f"(N_E,), (N_x*N_y,), (N_E, N_x*N_y), or full-grid shapes tied to mask {mask.shape}."
    )


def _evaluate_full_custom_state(
    mask: np.ndarray,
    energy_bins: np.ndarray,
    body: str,
    params: dict[str, Any],
    *,
    label: str,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Geometry mask must be 2D.")
    nE = int(np.asarray(energy_bins).size)
    if nE <= 0:
        raise ValueError("Energy bins must be non-empty for full custom profile evaluation.")

    evaluator = compile_safe_expression(
        body.strip(),
        variable_names=("x", "y", "E", "params"),
    )
    ny, nx = mask.shape
    coords = np.argwhere(mask)
    n_spatial = coords.shape[0]
    x_vals = (coords[:, 1].astype(float) + 0.5) / max(1, nx)
    y_vals = (coords[:, 0].astype(float) + 0.5) / max(1, ny)
    e_vals = np.asarray(energy_bins, dtype=float)

    try:
        raw = evaluator(
            x=x_vals[None, :],
            y=y_vals[None, :],
            E=e_vals[:, None],
            params=params,
        )
        arr = np.asarray(raw, dtype=float)
    except Exception:
        arr = np.empty((nE, n_spatial), dtype=float)
        for ie, energy in enumerate(e_vals):
            for px in range(n_spatial):
                arr[ie, px] = float(
                    evaluator(
                        x=float(x_vals[px]),
                        y=float(y_vals[px]),
                        E=float(energy),
                        params=params,
                    )
                )

    state = _coerce_energy_spatial_array(arr, e_vals, mask, label=label)
    if not np.all(np.isfinite(state)):
        raise ValueError(f"{label} expression produced non-finite values.")
    if np.any(state < 0):
        raise ValueError(f"{label} expression must be non-negative.")
    return state


def build_initial_qp_energy_state(
    mask: np.ndarray,
    E_bins: np.ndarray,
    spec: InitialConditionSpec,
) -> np.ndarray | None:
    """Build optional non-separable QP state [N_E, N_spatial]."""
    enabled, body, params = resolve_qp_full_custom_spec(spec)
    if not enabled:
        return None
    return _evaluate_full_custom_state(
        mask=mask,
        energy_bins=np.asarray(E_bins, dtype=float),
        body=body or _DEFAULT_QP_FULL_CUSTOM_BODY,
        params=dict(params or {}),
        label="Full quasiparticle profile",
    )


def _bose_einstein_occupation(energies_uev: np.ndarray, temperature_k: float) -> np.ndarray:
    energies = np.maximum(0.0, np.asarray(energies_uev, dtype=float))
    temp = float(temperature_k)
    if temp <= 0.0:
        return np.zeros_like(energies, dtype=float)
    x = energies / (_K_B_UEV_PER_K * temp)
    x = np.clip(x, 0.0, 700.0)
    den = np.expm1(x)
    return np.divide(
        1.0,
        den,
        out=np.zeros_like(energies, dtype=float),
        where=den > 0.0,
    )


def build_initial_phonon_energy_weights(
    omega_bins: np.ndarray,
    spec: InitialConditionSpec,
    bath_temperature: float,
) -> np.ndarray:
    """Build phonon energy occupancy weights over omega bins."""
    kind, params, custom_body, custom_params = _resolve_phonon_energy_spec(spec)
    mode = kind.strip().lower()
    omega = np.asarray(omega_bins, dtype=float).reshape(-1)
    if omega.size == 0:
        raise ValueError("omega_bins must be non-empty.")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega_bins must contain finite values.")
    if np.any(omega < 0):
        raise ValueError("omega_bins must be non-negative.")

    if mode in {"", "bose_einstein", "be", "thermal"}:
        temp = float(params.get("temperature", bath_temperature))
        values = _bose_einstein_occupation(omega, temp)
    elif mode == "uniform":
        value = float(params.get("value", 1.0))
        if value < 0:
            raise ValueError("Uniform phonon energy profile value must be non-negative.")
        values = np.full_like(omega, value, dtype=float)
    elif mode == "custom":
        evaluator = compile_safe_expression(
            custom_body.strip() or _DEFAULT_PHONON_ENERGY_CUSTOM_BODY,
            variable_names=("E", "params"),
        )
        params_map = dict(custom_params or {})
        try:
            raw = evaluator(E=omega, params=params_map)
            values = np.asarray(raw, dtype=float)
        except Exception:
            values = np.asarray(
                [float(evaluator(E=float(e), params=params_map)) for e in omega],
                dtype=float,
            )
        values = np.asarray(values, dtype=float).reshape(-1)
        if values.size == 1:
            values = np.full_like(omega, float(values[0]), dtype=float)
        if values.size != omega.size:
            raise ValueError(
                f"Custom phonon energy profile must return {omega.size} values or a scalar; got {values.size}."
            )
    else:
        raise ValueError(
            f"Unsupported phonon energy initial-condition kind '{mode}'. "
            "Supported: bose_einstein, uniform, custom."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("Phonon energy profile produced non-finite values.")
    if np.any(values < 0):
        raise ValueError("Phonon energy profile must be non-negative.")
    return values


def build_initial_phonon_energy_state(
    mask: np.ndarray,
    omega_bins: np.ndarray,
    spec: InitialConditionSpec,
    bath_temperature: float,
) -> np.ndarray:
    """Build phonon state [N_omega, N_spatial] for coupled qp-phonon solver."""
    enabled, body, params = resolve_phonon_full_custom_spec(spec)
    omega = np.asarray(omega_bins, dtype=float)
    if enabled:
        return _evaluate_full_custom_state(
            mask=mask,
            energy_bins=omega,
            body=body or _DEFAULT_PHONON_FULL_CUSTOM_BODY,
            params=dict(params or {}),
            label="Full phonon profile",
        )

    spatial = build_initial_phonon_spatial_field(mask, spec)
    spatial_values = np.asarray(spatial[mask], dtype=float).reshape(1, -1)
    energy_values = build_initial_phonon_energy_weights(
        omega_bins=omega,
        spec=spec,
        bath_temperature=bath_temperature,
    ).reshape(-1, 1)
    state = energy_values * spatial_values
    if not np.all(np.isfinite(state)):
        raise ValueError("Phonon initial state produced non-finite values.")
    if np.any(state < 0):
        raise ValueError("Phonon initial state must be non-negative.")
    return state
