from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .initial_conditions import evaluate_gap_expression
from .models import BoundaryCondition, EdgeSegment, SimulationParameters
from .solver import (
    _dynes_density_of_states,
    build_energy_grid,
    recombination_kernel,
    scattering_kernel,
    thermal_qp_weights,
)


def _mask_hash(mask: np.ndarray) -> float:
    """Stable numeric hash for geometry mask shape + topology."""
    import hashlib

    mask_bool = np.asarray(mask, dtype=bool)
    packed = np.packbits(mask_bool.astype(np.uint8, copy=False))
    hasher = hashlib.sha256()
    hasher.update(np.asarray(mask_bool.shape, dtype=np.int64).tobytes())
    hasher.update(packed.tobytes())
    return float(int.from_bytes(hasher.digest()[:8], "big") % (2**53))


def _gap_expression_hash(gap_expression: str) -> float:
    import hashlib

    return float(int(hashlib.sha256(gap_expression.encode()).hexdigest()[:16], 16) % (2**53))


def _make_fingerprint(params: SimulationParameters, mask: np.ndarray) -> np.ndarray:
    """Create a numeric fingerprint of parameters relevant to precomputation.

    Stored as a 1D float array so it can be saved in .npz and compared on load.
    """
    n_spatial = int(np.sum(mask))
    gap_hash = _gap_expression_hash(params.gap_expression)
    values = [
        params.energy_gap,
        params.energy_min_factor,
        params.energy_max_factor,
        float(params.num_energy_bins),
        params.dynes_gamma,
        params.diffusion_coefficient,
        float(params.tau_s if params.tau_s is not None else params.tau_0),
        float(params.tau_r if params.tau_r is not None else params.tau_0),
        params.T_c,
        params.bath_temperature,
        float(n_spatial),
        _mask_hash(mask),
        float(gap_hash),
    ]
    return np.array(values, dtype=float)


def validate_precomputed(
    precomputed: dict[str, Any],
    params: SimulationParameters,
    mask: np.ndarray,
) -> str | None:
    """Validate precomputed arrays against current parameters.

    Returns None if compatible, or a description string of the mismatch.
    """
    stored = precomputed.get("fingerprint")
    if stored is None:
        return "Precomputed file has no fingerprint (created before compatibility checks)."
    current = _make_fingerprint(params, mask)
    labels = [
        "energy_gap", "energy_min_factor", "energy_max_factor",
        "num_energy_bins", "dynes_gamma", "diffusion_coefficient",
        "tau_s", "tau_r", "T_c", "bath_temperature", "n_spatial", "mask_hash", "gap_expression",
    ]
    if stored.shape != current.shape:
        return f"Fingerprint size mismatch: stored {stored.shape} vs current {current.shape}."
    if not np.allclose(stored, current, rtol=1e-12, atol=1e-12):
        diffs = []
        for i, (s, c) in enumerate(zip(stored, current)):
            if abs(s - c) > 1e-12 * max(abs(s), abs(c), 1.0):
                label = labels[i] if i < len(labels) else f"param[{i}]"
                diffs.append(f"{label}: stored={s}, current={c}")
        return "Parameter mismatch: " + "; ".join(diffs)
    return None


def estimate_precompute_memory(
    n_spatial: int,
    n_energy: int,
    is_uniform: bool,
    include_collision_kernels: bool = False,
) -> int:
    """Estimate memory in bytes for precomputed arrays."""
    float_bytes = 8
    # Base payload: D(E, x), E bins, and gap map.
    base = float_bytes * (n_energy * n_spatial + n_energy + n_spatial)
    if not include_collision_kernels:
        return base
    if is_uniform:
        # K_r, K_s: NE x NE; rho_bins, G_therm: NE
        return base + float_bytes * (2 * n_energy ** 2 + 2 * n_energy)
    # Per-spatial: K_r_all, K_s_all: N_spatial x NE x NE; rho_all, G_therm_all: N_spatial x NE
    return base + float_bytes * (
        2 * n_spatial * n_energy ** 2
        + 2 * n_spatial * n_energy
    )


def precompute_arrays(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    params: SimulationParameters,
    progress_callback: Callable[[str], None] | None = None,
    *,
    include_collision_kernels: bool = False,
) -> dict[str, Any]:
    """Precompute diffusion arrays and optionally collision kernels.

    Returns a dict of numpy arrays suitable for .npz storage and solver consumption.
    """
    if params.energy_gap <= 0:
        raise ValueError("precompute_arrays requires energy_gap > 0.")
    gap_default = params.energy_gap
    n_spatial = int(np.sum(mask))
    NE = params.num_energy_bins

    E_bins, dE = build_energy_grid(
        gap_default,
        params.energy_min_factor,
        params.energy_max_factor,
        NE,
    )

    if progress_callback:
        progress_callback("Evaluating gap expression...")

    gap_values = evaluate_gap_expression(params.gap_expression, mask, gap_default)
    unique_gaps = np.unique(gap_values)
    is_uniform = len(unique_gaps) == 1

    if progress_callback:
        progress_callback(f"{'Uniform' if is_uniform else f'{len(unique_gaps)} unique'} gap values")

    gamma = params.dynes_gamma

    # D(E, x) = D0 * sqrt(1 - (Delta(x)/E)^2)
    D_array = np.empty((NE, n_spatial), dtype=float)
    for i in range(NE):
        ratio = np.minimum(gap_values / E_bins[i], 1.0)
        D_array[i] = params.diffusion_coefficient * np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))

    result: dict[str, Any] = {
        "fingerprint": _make_fingerprint(params, mask),
        "E_bins": E_bins,
        "gap_values": gap_values,
        "is_uniform": np.array(is_uniform),
        "D_array": D_array,
    }

    if include_collision_kernels and is_uniform:
        if progress_callback:
            progress_callback("Computing uniform kernels...")
        gap = float(unique_gaps[0])
        tau_r = float(params.tau_r if params.tau_r is not None else params.tau_0)
        tau_s = float(params.tau_s if params.tau_s is not None else params.tau_0)
        K_r = recombination_kernel(E_bins, gap, tau_r, params.T_c, params.bath_temperature)
        K_s = scattering_kernel(E_bins, gap, tau_s, params.T_c, params.bath_temperature)
        rho_bins = _dynes_density_of_states(E_bins, gap, gamma)
        n_eq = thermal_qp_weights(E_bins, gap, params.bath_temperature, gamma)
        G_therm = 2.0 * n_eq * dE * (K_r @ n_eq)

        result["K_r"] = K_r
        result["K_s"] = K_s
        result["rho_bins"] = rho_bins
        result["G_therm"] = G_therm
    elif include_collision_kernels:
        if progress_callback:
            progress_callback("Computing per-pixel kernels (caching by unique gap)...")

        K_r_all = np.empty((n_spatial, NE, NE), dtype=float)
        K_s_all = np.empty((n_spatial, NE, NE), dtype=float)
        rho_all = np.empty((n_spatial, NE), dtype=float)
        G_therm_all = np.empty((n_spatial, NE), dtype=float)

        cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for gap_val in unique_gaps:
            gap_f = float(gap_val)
            tau_r = float(params.tau_r if params.tau_r is not None else params.tau_0)
            tau_s = float(params.tau_s if params.tau_s is not None else params.tau_0)
            kr = recombination_kernel(E_bins, gap_f, tau_r, params.T_c, params.bath_temperature)
            ks = scattering_kernel(E_bins, gap_f, tau_s, params.T_c, params.bath_temperature)
            rho = _dynes_density_of_states(E_bins, gap_f, gamma)
            n_eq = thermal_qp_weights(E_bins, gap_f, params.bath_temperature, gamma)
            g_therm = 2.0 * n_eq * dE * (kr @ n_eq)
            cache[gap_f] = (kr, ks, rho, g_therm)

        for px in range(n_spatial):
            gap_f = float(gap_values[px])
            kr, ks, rho, g_therm = cache[gap_f]
            K_r_all[px] = kr
            K_s_all[px] = ks
            rho_all[px] = rho
            G_therm_all[px] = g_therm

        result["K_r_all"] = K_r_all
        result["K_s_all"] = K_s_all
        result["rho_all"] = rho_all
        result["G_therm_all"] = G_therm_all

    if progress_callback:
        progress_callback(
            "Precomputation complete."
            if include_collision_kernels
            else "Precomputation complete (diffusion/gap arrays only)."
        )

    return result
