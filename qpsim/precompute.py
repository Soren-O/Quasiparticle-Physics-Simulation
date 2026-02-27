from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .initial_conditions import evaluate_gap_expression
from .models import BoundaryCondition, EdgeSegment, SimulationParameters
from .solver import (
    _KB_UEV_PER_K,
    _dynes_density_of_states,
    recombination_kernel,
    scattering_kernel,
    thermal_qp_weights,
)


def _make_fingerprint(params: SimulationParameters, n_spatial: int) -> np.ndarray:
    """Create a numeric fingerprint of parameters relevant to precomputation.

    Stored as a 1D float array so it can be saved in .npz and compared on load.
    """
    import hashlib
    # Hash the gap expression string to a float
    gap_hash = int(hashlib.sha256(params.gap_expression.encode()).hexdigest()[:16], 16) % (2**53)
    values = [
        params.energy_gap,
        params.energy_min_factor,
        params.energy_max_factor,
        float(params.num_energy_bins),
        params.dynes_gamma,
        params.diffusion_coefficient,
        params.tau_0,
        params.T_c,
        params.bath_temperature,
        float(n_spatial),
        float(gap_hash),
    ]
    return np.array(values, dtype=float)


def validate_precomputed(
    precomputed: dict[str, Any],
    params: SimulationParameters,
    n_spatial: int,
) -> str | None:
    """Validate precomputed arrays against current parameters.

    Returns None if compatible, or a description string of the mismatch.
    """
    stored = precomputed.get("fingerprint")
    if stored is None:
        return "Precomputed file has no fingerprint (created before compatibility checks)."
    current = _make_fingerprint(params, n_spatial)
    if stored.shape != current.shape:
        return f"Fingerprint size mismatch: stored {stored.shape} vs current {current.shape}."
    if not np.allclose(stored, current, rtol=1e-12, atol=1e-12):
        labels = [
            "energy_gap", "energy_min_factor", "energy_max_factor",
            "num_energy_bins", "dynes_gamma", "diffusion_coefficient",
            "tau_0", "T_c", "bath_temperature", "n_spatial", "gap_expression",
        ]
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
) -> int:
    """Estimate memory in bytes for precomputed arrays."""
    float_bytes = 8
    if is_uniform:
        # K_r, K_s: NE×NE; rho_bins, G_therm: NE; D_array: NE×N_spatial
        return float_bytes * (2 * n_energy ** 2 + 2 * n_energy + n_energy * n_spatial)
    else:
        # Per-spatial: K_r_all, K_s_all: N_spatial×NE×NE; rho_all, G_therm_all: N_spatial×NE
        return float_bytes * (
            2 * n_spatial * n_energy ** 2
            + 2 * n_spatial * n_energy
            + n_energy * n_spatial
        )


def precompute_arrays(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    params: SimulationParameters,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Precompute collision kernels and diffusion arrays.

    Returns a dict of numpy arrays suitable for .npz storage and solver consumption.
    """
    if params.energy_gap <= 0:
        raise ValueError("precompute_arrays requires energy_gap > 0.")
    gap_default = params.energy_gap
    n_spatial = int(np.sum(mask))
    NE = params.num_energy_bins

    E_min = params.energy_min_factor * gap_default
    E_max = params.energy_max_factor * gap_default
    E_bins = np.linspace(E_min, E_max, NE)
    dE = E_bins[1] - E_bins[0] if NE > 1 else 1.0

    if progress_callback:
        progress_callback("Evaluating gap expression...")

    gap_values = evaluate_gap_expression(params.gap_expression, mask, gap_default)
    unique_gaps = np.unique(gap_values)
    is_uniform = len(unique_gaps) == 1

    if progress_callback:
        progress_callback(f"{'Uniform' if is_uniform else f'{len(unique_gaps)} unique'} gap values")

    gamma = params.dynes_gamma

    # D(E, x) = D₀ × √(1 − (Δ(x)/E)²)
    D_array = np.empty((NE, n_spatial), dtype=float)
    for i in range(NE):
        ratio = np.minimum(gap_values / E_bins[i], 1.0)
        D_array[i] = params.diffusion_coefficient * np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))

    result: dict[str, Any] = {
        "fingerprint": _make_fingerprint(params, n_spatial),
        "E_bins": E_bins,
        "gap_values": gap_values,
        "is_uniform": np.array(is_uniform),
        "D_array": D_array,
    }

    if is_uniform:
        if progress_callback:
            progress_callback("Computing uniform kernels...")
        gap = float(unique_gaps[0])
        K_r = recombination_kernel(E_bins, gap, params.tau_0, params.T_c, params.bath_temperature)
        K_s = scattering_kernel(E_bins, gap, params.tau_0, params.T_c, params.bath_temperature)
        rho_bins = _dynes_density_of_states(E_bins, gap, gamma)
        n_eq = thermal_qp_weights(E_bins, gap, params.bath_temperature, gamma)
        G_therm = 2.0 * n_eq * dE * (K_r @ n_eq)

        result["K_r"] = K_r
        result["K_s"] = K_s
        result["rho_bins"] = rho_bins
        result["G_therm"] = G_therm
    else:
        if progress_callback:
            progress_callback("Computing per-pixel kernels (caching by unique gap)...")

        K_r_all = np.empty((n_spatial, NE, NE), dtype=float)
        K_s_all = np.empty((n_spatial, NE, NE), dtype=float)
        rho_all = np.empty((n_spatial, NE), dtype=float)
        G_therm_all = np.empty((n_spatial, NE), dtype=float)

        # Cache by unique gap value
        cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for gap_val in unique_gaps:
            gap_f = float(gap_val)
            kr = recombination_kernel(E_bins, gap_f, params.tau_0, params.T_c, params.bath_temperature)
            ks = scattering_kernel(E_bins, gap_f, params.tau_0, params.T_c, params.bath_temperature)
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
        progress_callback("Precomputation complete.")

    return result
