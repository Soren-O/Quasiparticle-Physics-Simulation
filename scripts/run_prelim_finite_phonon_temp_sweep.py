"""Small finite-phonon bath-temperature sweep for the prelim strip."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_spatial_1d import T3Spatial1DState
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from scripts.run_prelim_spatial_finite_phonon_one import (
    CONFIG,
    TAU_L_NS,
    FinitePhononSpatialRunner,
)
from scripts.run_prelim_spatial_overnight import (
    ENERGY_MAX_FACTOR,
    LENGTH_UM,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_spatial_finite_phonon_temp_sweep"
T_BATH_VALUES_K = (0.02, 0.05, 0.10, 0.20, 0.30)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state_at_temperature(T_bath_K: float) -> T3Spatial1DState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=CONFIG.NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=CONFIG.D0_values[0],
    )
    x = np.linspace(0.0, LENGTH_UM, CONFIG.NX)
    f0 = np.repeat(_fermi_dirac(E, T_bath_K)[:, None], CONFIG.NX, axis=1)
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=T_bath_K,
    )


def _run_one_temperature(T_bath_K: float) -> dict[str, float | bool]:
    state = _build_state_at_temperature(T_bath_K)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=CONFIG.source_rates_per_ns[0],
        center_delta=CONFIG.source_centers_delta[0],
        sigma_delta=CONFIG.source_sigmas_delta[0],
    )
    runner = FinitePhononSpatialRunner(state, tau_l_ns=TAU_L_NS)
    start = time.monotonic()
    t_ns = 0.0
    n_steps = 0
    converged = False
    max_dfdt = float("inf")
    max_dnphdt = float("inf")

    while t_ns < CONFIG.max_time_ns:
        state, max_dfdt, max_dnphdt = runner.step(state, CONFIG.dt_ns, source)
        t_ns += CONFIG.dt_ns
        n_steps += 1
        if max_dfdt < CONFIG.stop_tol:
            converged = True
            break

    wall_s = time.monotonic() - start
    xqp = _xqp_profile(state)
    shift_rows = _resonator_shifts(state, f_ref)
    shifts = [row["delta_fr_hz_current_weighted"] for row in shift_rows]
    qis = [row["Qi_current_weighted"] for row in shift_rows]

    shift_path = OUT_DIR / f"resonator_shifts_T_{int(T_bath_K * 1000):03d}mK.csv"
    with shift_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(shift_rows[0].keys()))
        writer.writeheader()
        writer.writerows(shift_rows)

    return {
        "T_bath_K": T_bath_K,
        "tau_l_ns": TAU_L_NS,
        "D0_um2_per_ns": CONFIG.D0_values[0],
        "source_rate_per_ns": CONFIG.source_rates_per_ns[0],
        **_source_calibration(CONFIG, CONFIG.source_rates_per_ns[0]),
        "total_time_ns": t_ns,
        "n_steps": n_steps,
        "wall_seconds": wall_s,
        "converged": converged,
        "final_max_dfdt_per_ns": max_dfdt,
        "final_max_dnphdt_per_ns": max_dnphdt,
        "xqp_mean": float(np.mean(xqp)),
        "xqp_source": float(xqp[0]),
        "xqp_open_end": float(xqp[-1]),
        "nph_mean": float(np.mean(runner.n_ph)),
        "nph_max": float(np.max(runner.n_ph)),
        "delta_fr_hz_min": float(min(shifts)),
        "delta_fr_hz_max": float(max(shifts)),
        "abs_delta_fr_hz_median": float(np.median(np.abs(shifts))),
        "Qi_min": float(min(qis)),
        "Qi_max": float(max(qis)),
        "shift_csv": shift_path.name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": __doc__,
        "T_bath_values_K": T_BATH_VALUES_K,
        "config": CONFIG.__dict__,
        "tau_l_ns": TAU_L_NS,
    }
    with (OUT_DIR / "metadata.json").open("w") as fp:
        json.dump(metadata, fp, indent=2)

    rows = []
    for T in T_BATH_VALUES_K:
        print(f"Running finite-phonon T={T:.3f} K ...", flush=True)
        row = _run_one_temperature(T)
        rows.append(row)
        print(
            f"T={T:.3f} K: xqp_mean={row['xqp_mean']:.3e}, "
            f"shift=[{row['delta_fr_hz_min']:.3e}, {row['delta_fr_hz_max']:.3e}] Hz, "
            f"wall={row['wall_seconds']:.2f}s",
            flush=True,
        )

    with (OUT_DIR / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
