from __future__ import annotations

import numpy as np

from qpsim.geometry import extract_edge_segments
from qpsim.models import BoundaryCondition, ExternalGenerationSpec
from qpsim.solver import (
    _dynes_density_of_states,
    build_energy_grid,
    recombination_kernel,
    run_2d_crank_nicolson,
    scattering_kernel,
    thermal_qp_weights,
)


def _mkid_like_reference_1d(
    *,
    nx: int,
    ne: int,
    dt: float,
    steps: int,
    dE: float,
    D_bins: np.ndarray,
    K_r: np.ndarray,
    K_s: np.ndarray,
    rho: np.ndarray,
    n_thermal: np.ndarray,
    weights: np.ndarray,
    initial_spatial: np.ndarray,
    generation_rate: float,
) -> np.ndarray:
    """Run a MKID-style 1D reference update.

    This uses:
    1. Generation before diffusion
    2. 1D CN diffusion (tridiagonal Thomas solve)
    3. Simultaneous scattering/recombination collision update
    """
    # MKID-style CN coefficient: alpha = 2*dx^2/dt, and here dx=1.
    alpha = 2.0 / dt
    D_if = np.repeat(D_bins[:, None], nx - 1, axis=1)  # [ne, nx-1]

    c_prime = np.zeros((ne, nx - 1), dtype=float)
    for j in range(ne):
        D_j = D_if[j]
        c_prime[j, 0] = -D_j[0] / (alpha + D_j[0])
        for i in range(1, nx - 1):
            denom = alpha + D_j[i] + D_j[i - 1] * (1.0 + c_prime[j, i - 1])
            c_prime[j, i] = -D_j[i] / denom

    state = np.empty((ne, nx), dtype=float)
    for j in range(ne):
        state[j] = initial_spatial * weights[j]

    states = [state.copy()]
    for _ in range(steps):
        # Generation first
        state += dt * generation_rate

        # Diffusion second
        for j in range(ne):
            n = state[j]
            D_j = D_if[j]
            cp_j = c_prime[j]

            d = np.zeros(nx, dtype=float)
            d[0] = (alpha - D_j[0]) * n[0] + D_j[0] * n[1]
            d[-1] = D_j[-1] * n[-2] + (alpha - D_j[-1]) * n[-1]
            d[1:-1] = (
                D_j[:-1] * n[:-2]
                + (alpha - D_j[:-1] - D_j[1:]) * n[1:-1]
                + D_j[1:] * n[2:]
            )

            d_prime = np.zeros(nx, dtype=float)
            d_prime[0] = d[0] / (alpha + D_j[0])
            for i in range(1, nx - 1):
                denom = alpha + D_j[i] + D_j[i - 1] * (1.0 + cp_j[i - 1])
                d_prime[i] = (d[i] + D_j[i - 1] * d_prime[i - 1]) / denom
            d_prime[-1] = (d[-1] + D_j[-1] * d_prime[-2]) / (
                alpha + D_j[-1] * (1.0 + cp_j[-1])
            )

            x = np.zeros(nx, dtype=float)
            x[-1] = d_prime[-1]
            for i in range(nx - 2, -1, -1):
                x[i] = d_prime[i] - cp_j[i] * x[i + 1]
            state[j] = x

        # Collision third (simultaneous terms)
        for ix in range(nx):
            n = state[:, ix].copy()
            f = n / np.maximum(rho, 1e-30)
            pauli = np.maximum(1.0 - f, 0.0)
            scatter_in = dE * rho * pauli * (K_s.T @ n)
            scatter_out = n * dE * ((K_s * rho[None, :]) @ pauli)
            recomb = 2.0 * n * dE * (K_r @ n)
            thermal = 2.0 * n_thermal * dE * (K_r @ n_thermal)
            n_new = n + dt * (scatter_in - scatter_out - recomb + thermal)
            state[:, ix] = np.maximum(n_new, 0.0)

        states.append(state.copy())

    return np.array(states)


def test_1d_in_2d_reflective_generation_matches_mkid_style_reference() -> None:
    """Cross-check 1D dynamics in 2D mask against MKID-style reference updates."""
    nx = 48
    ne = 12
    dt = 0.1
    steps = 12
    total_time = dt * steps
    dx = 1.0

    mask = np.ones((1, nx), dtype=bool)
    edges = extract_edge_segments(mask)
    boundary = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}

    gap = 180.0
    energy_min_factor = 1.0
    energy_max_factor = 3.0
    D0 = 6.0
    gamma = 0.18
    tau = 400.0
    T_c = 1.2
    T_bath = 0.1
    generation_rate = 2e-8

    E_bins, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, ne)
    initial_spatial = 1e-4 + 2e-4 * np.exp(
        -(((np.arange(nx) + 0.5) / nx - 0.3) ** 2) / (2.0 * 0.06**2)
    )
    initial_field = initial_spatial.reshape(1, nx)

    weights = thermal_qp_weights(E_bins, gap, T_bath, gamma)
    weights = weights / (np.sum(weights) * dE)

    external_generation = ExternalGenerationSpec(mode="constant", rate=generation_rate)

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=boundary,
        initial_field=initial_field,
        diffusion_coefficient=D0,
        dt=dt,
        total_time=total_time,
        dx=dx,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=ne,
        energy_weights=weights,
        enable_diffusion=True,
        enable_recombination=True,
        enable_scattering=True,
        dynes_gamma=gamma,
        tau_0=tau,
        T_c=T_c,
        bath_temperature=T_bath,
        external_generation=external_generation,
    )

    assert energy_frames is not None
    state_qpsim = np.array(
        [np.array([energy_frame[0, :] for energy_frame in t_slice], dtype=float) for t_slice in energy_frames]
    )  # [time, ne, nx]

    K_r = recombination_kernel(E_bins, gap, tau, T_c, T_bath)
    K_s = scattering_kernel(E_bins, gap, tau, T_c, T_bath)
    rho = _dynes_density_of_states(E_bins, gap, gamma)
    n_thermal = thermal_qp_weights(E_bins, gap, T_bath, gamma)
    D_bins = D0 * np.sqrt(np.maximum(0.0, 1.0 - (gap / E_bins) ** 2))

    state_ref = _mkid_like_reference_1d(
        nx=nx,
        ne=ne,
        dt=dt,
        steps=steps,
        dE=dE,
        D_bins=D_bins,
        K_r=K_r,
        K_s=K_s,
        rho=rho,
        n_thermal=n_thermal,
        weights=weights,
        initial_spatial=initial_spatial,
        generation_rate=generation_rate,
    )

    # Full spectral-array comparison
    max_abs = float(np.max(np.abs(state_qpsim - state_ref)))
    max_ref = float(np.max(np.abs(state_ref)))
    rel = max_abs / max(1e-20, max_ref)

    # Energy-integrated comparison
    integrated_qpsim = np.sum(state_qpsim, axis=1) * dE
    integrated_ref = np.sum(state_ref, axis=1) * dE
    max_abs_int = float(np.max(np.abs(integrated_qpsim - integrated_ref)))
    max_ref_int = float(np.max(np.abs(integrated_ref)))
    rel_int = max_abs_int / max(1e-20, max_ref_int)

    assert rel < 1e-6
    assert rel_int < 1e-6
