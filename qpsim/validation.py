from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import extract_edge_segments
from .models import BoundaryCondition, SimulationParameters
from .solver import (
    build_energy_grid,
    run_2d_crank_nicolson,
    scattering_kernel,
    thermal_qp_weights,
)


_KB_UEV_PER_K = 86.17


def _reflective_line_geometry(nx: int) -> tuple[np.ndarray, list, dict[str, BoundaryCondition]]:
    mask = np.ones((1, nx), dtype=bool)
    edges = extract_edge_segments(mask)
    bcs = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
    return mask, edges, bcs


@dataclass
class ValidationReport:
    detailed_balance: dict[str, Any]
    thermal_stability: dict[str, Any]
    pure_diffusion: dict[str, Any]
    pure_scattering: dict[str, Any]
    pure_recombination: dict[str, Any]

    @property
    def overall_passed(self) -> bool:
        return all(
            bool(section.get("passed", False))
            for section in (
                self.detailed_balance,
                self.thermal_stability,
                self.pure_diffusion,
                self.pure_scattering,
                self.pure_recombination,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "detailed_balance": self.detailed_balance,
            "thermal_stability": self.thermal_stability,
            "pure_diffusion": self.pure_diffusion,
            "pure_scattering": self.pure_scattering,
            "pure_recombination": self.pure_recombination,
            "overall_passed": self.overall_passed,
        }
        return payload


def validate_detailed_balance(
    *,
    gap: float,
    energy_min_factor: float,
    energy_max_factor: float,
    num_energy_bins: int,
    tau_s: float,
    T_c: float,
    bath_temperature: float,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    if bath_temperature <= 0:
        return {"passed": True, "max_relative_error": 0.0, "message": "Skipped (T_bath <= 0)."}

    E_bins, _ = build_energy_grid(gap, energy_min_factor, energy_max_factor, num_energy_bins)
    K_s = scattering_kernel(E_bins, gap, tau_s, T_c, bath_temperature)
    kT = _KB_UEV_PER_K * bath_temperature
    E_diff = E_bins[:, None] - E_bins[None, :]
    rhs = K_s.T * np.exp(np.clip(E_diff / kT, -200.0, 200.0))

    denom = max(1e-30, float(np.max(np.abs(K_s))))
    max_rel = float(np.max(np.abs(K_s - rhs)) / denom)
    return {"passed": max_rel <= tolerance, "max_relative_error": max_rel, "tolerance": tolerance}


def validate_thermal_stability(
    *,
    nx: int,
    dt: float,
    steps: int,
    diffusion_coefficient: float,
    gap: float,
    energy_min_factor: float,
    energy_max_factor: float,
    num_energy_bins: int,
    dynes_gamma: float,
    tau_s: float,
    tau_r: float,
    T_c: float,
    bath_temperature: float,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    mask, edges, bcs = _reflective_line_geometry(nx)
    E_bins, dE = build_energy_grid(gap, energy_min_factor, energy_max_factor, num_energy_bins)
    n_eq = thermal_qp_weights(E_bins, gap, bath_temperature, dynes_gamma)
    # run_2d_crank_nicolson normalizes energy weights so ∫w dE = 1.
    # Set initial integrated density so that spectral state starts at n_eq.
    initial_amplitude = float(np.sum(n_eq) * dE)
    initial_field = np.full((1, nx), initial_amplitude, dtype=float)

    _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=diffusion_coefficient,
        dt=dt,
        total_time=steps * dt,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=num_energy_bins,
        energy_weights=n_eq,
        enable_diffusion=True,
        enable_recombination=True,
        enable_scattering=True,
        dynes_gamma=dynes_gamma,
        collision_solver="bdf",
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=bath_temperature,
    )
    if energy_frames is None:
        return {"passed": False, "max_relative_drift": float("inf"), "tolerance": tolerance}

    state_0 = np.array([frame[0, :] for frame in energy_frames[0]], dtype=float)
    state_f = np.array([frame[0, :] for frame in energy_frames[-1]], dtype=float)
    denom = max(1e-20, float(np.max(np.abs(state_0))))
    max_rel_drift = float(np.max(np.abs(state_f - state_0)) / denom)
    return {
        "passed": max_rel_drift <= tolerance,
        "max_relative_drift": max_rel_drift,
        "tolerance": tolerance,
    }


def validate_pure_diffusion(
    *,
    nx: int,
    dt: float,
    total_time: float,
    diffusion_coefficient: float,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    mask, edges, bcs = _reflective_line_geometry(nx)
    x = (np.arange(nx, dtype=float) + 0.5) / nx
    initial_field = (1.0 + 0.4 * np.cos(2.0 * np.pi * x))[None, :]

    _, _, mass, _, _, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=diffusion_coefficient,
        dt=dt,
        total_time=total_time,
        dx=1.0,
        store_every=1,
        energy_gap=0.0,
        enable_diffusion=True,
    )
    drift = float(abs(mass[-1] - mass[0]) / max(1e-20, abs(mass[0])))
    return {"passed": drift <= tolerance, "mass_relative_drift": drift, "tolerance": tolerance}


def validate_pure_scattering(
    *,
    nx: int,
    dt: float,
    steps: int,
    gap: float,
    energy_min_factor: float,
    energy_max_factor: float,
    num_energy_bins: int,
    dynes_gamma: float,
    tau_s: float,
    T_c: float,
    bath_temperature: float,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    mask, edges, bcs = _reflective_line_geometry(nx)
    E_bins, _ = build_energy_grid(gap, energy_min_factor, energy_max_factor, num_energy_bins)
    weights = np.exp(-((E_bins - 2.6 * gap) / (0.6 * gap)) ** 2)
    initial_field = np.full((1, nx), 2e-4, dtype=float)

    _, _, mass, _, _, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=6.0,
        dt=dt,
        total_time=steps * dt,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=energy_min_factor,
        energy_max_factor=energy_max_factor,
        num_energy_bins=num_energy_bins,
        energy_weights=weights,
        enable_diffusion=False,
        enable_recombination=False,
        enable_scattering=True,
        dynes_gamma=dynes_gamma,
        collision_solver="bdf",
        tau_s=tau_s,
        T_c=T_c,
        bath_temperature=bath_temperature,
    )
    drift = float(abs(mass[-1] - mass[0]) / max(1e-20, abs(mass[0])))
    return {"passed": drift <= tolerance, "mass_relative_drift": drift, "tolerance": tolerance}


def validate_pure_recombination(
    *,
    dt: float,
    steps: int,
    gap: float,
    tau_r: float,
    T_c: float,
    tolerance_nonincreasing: float = 1e-15,
) -> dict[str, Any]:
    mask, edges, bcs = _reflective_line_geometry(1)
    initial_field = np.array([[1e-3]], dtype=float)

    _, _, mass, _, _, _ = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial_field,
        diffusion_coefficient=6.0,
        dt=dt,
        total_time=steps * dt,
        dx=1.0,
        store_every=1,
        energy_gap=gap,
        energy_min_factor=1.5,
        energy_max_factor=1.5,
        num_energy_bins=1,
        enable_diffusion=False,
        enable_recombination=True,
        enable_scattering=False,
        dynes_gamma=0.0,
        collision_solver="forward_euler",
        tau_r=tau_r,
        T_c=T_c,
        bath_temperature=0.0,
    )
    nonincreasing = all(
        mass[i + 1] <= mass[i] + tolerance_nonincreasing for i in range(len(mass) - 1)
    )
    return {"passed": bool(nonincreasing), "mass_start": mass[0], "mass_end": mass[-1]}


def run_fast_validation_suite(params: SimulationParameters | None = None) -> ValidationReport:
    p = params or SimulationParameters(
        diffusion_coefficient=6.0,
        dt=0.1,
        total_time=1.0,
        mesh_size=1.0,
        energy_gap=180.0,
        energy_min_factor=1.0,
        energy_max_factor=4.0,
        num_energy_bins=24,
        dynes_gamma=0.18,
        enable_diffusion=True,
        enable_recombination=True,
        enable_scattering=True,
        tau_s=440.0,
        tau_r=440.0,
        T_c=1.2,
        bath_temperature=0.1,
    )

    tau_s = float(p.tau_s if p.tau_s is not None else p.tau_0)
    tau_r = float(p.tau_r if p.tau_r is not None else p.tau_0)
    detail = validate_detailed_balance(
        gap=p.energy_gap,
        energy_min_factor=p.energy_min_factor,
        energy_max_factor=p.energy_max_factor,
        num_energy_bins=p.num_energy_bins,
        tau_s=tau_s,
        T_c=p.T_c,
        bath_temperature=p.bath_temperature,
    )
    thermal = validate_thermal_stability(
        nx=16,
        dt=min(0.1, p.dt),
        steps=5,
        diffusion_coefficient=p.diffusion_coefficient,
        gap=p.energy_gap,
        energy_min_factor=p.energy_min_factor,
        energy_max_factor=p.energy_max_factor,
        num_energy_bins=p.num_energy_bins,
        dynes_gamma=p.dynes_gamma,
        tau_s=tau_s,
        tau_r=tau_r,
        T_c=p.T_c,
        bath_temperature=p.bath_temperature,
    )
    pure_diff = validate_pure_diffusion(
        nx=64,
        dt=min(0.2, p.dt),
        total_time=2.0,
        diffusion_coefficient=p.diffusion_coefficient,
    )
    pure_scat = validate_pure_scattering(
        nx=8,
        dt=min(0.05, p.dt),
        steps=10,
        gap=p.energy_gap,
        energy_min_factor=p.energy_min_factor,
        energy_max_factor=p.energy_max_factor,
        num_energy_bins=max(12, p.num_energy_bins),
        dynes_gamma=p.dynes_gamma,
        tau_s=tau_s,
        T_c=p.T_c,
        bath_temperature=p.bath_temperature,
    )
    pure_recomb = validate_pure_recombination(
        dt=min(0.1, p.dt),
        steps=20,
        gap=p.energy_gap,
        tau_r=tau_r,
        T_c=p.T_c,
    )

    return ValidationReport(
        detailed_balance=detail,
        thermal_stability=thermal,
        pure_diffusion=pure_diff,
        pure_scattering=pure_scat,
        pure_recombination=pure_recomb,
    )
