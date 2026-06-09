"""Fixed-nbar readout-heating probe for the preliminary 100 um Al strip.

This is a small comparison run, not the final readout-power sweep. It keeps the
same finite-escape local-phonon dynamics used in the 7 mK sweep, then adds a
single sub-gap microwave photon scattering channel weighted by the ideal
quarter-wave ``I^2`` profile of one resonator mode.

The probe deliberately uses a finer energy grid than the main sweep so the
5.14 GHz readout photon is nearly commensurate with the energy spacing.
"""

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
from qpsim.experiments.prelim_resonators import PRELIM_RESONATORS
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from scripts.run_prelim_spatial_finite_phonon_one import (
    FinitePhononSpatialRunner,
    readout_drive_from_resonator,
)
from scripts.run_prelim_spatial_overnight import (
    ENERGY_MAX_FACTOR,
    LENGTH_UM,
    SweepConfig,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_readout_heating_probe"
T_BATH_K = 0.007
TAU_L_NS = 1.0
READOUT_RESONATOR_INDEX = 1
C_PHOT_NS_INV = 1e-9
N_BAR_VALUES = (0.0, 1e5, 1e6)

CONFIG = SweepConfig(
    name="readout_heating_probe",
    NX=11,
    NE=101,
    dt_ns=1.0,
    max_time_ns=1_500.0,
    stop_tol=2e-9,
    snapshot_interval_ns=500.0,
    D0_values=(20.0,),
    source_rates_per_ns=(5e-4,),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(D0: float) -> T3Spatial1DState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=CONFIG.NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    x = np.linspace(0.0, LENGTH_UM, CONFIG.NX)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], CONFIG.NX, axis=1)
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=T_BATH_K,
    )


def _run_case(n_bar: float) -> tuple[dict[str, float | bool | str], list[dict[str, float]]]:
    D0 = CONFIG.D0_values[0]
    source_rate = CONFIG.source_rates_per_ns[0]
    state = _build_state(D0)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=CONFIG.source_centers_delta[0],
        sigma_delta=CONFIG.source_sigmas_delta[0],
    )
    resonator = PRELIM_RESONATORS[READOUT_RESONATOR_INDEX - 1]
    readout_drive = (
        None
        if n_bar == 0.0
        else readout_drive_from_resonator(
            state,
            resonator,
            n_bar=n_bar,
            c_phot=C_PHOT_NS_INV,
        )
    )
    runner = FinitePhononSpatialRunner(state, tau_l_ns=TAU_L_NS)

    start = time.monotonic()
    t_ns = 0.0
    n_steps = 0
    converged = False
    max_dfdt = float("inf")
    max_dnphdt = float("inf")

    while t_ns < CONFIG.max_time_ns:
        state, max_dfdt, max_dnphdt = runner.step(
            state,
            CONFIG.dt_ns,
            source,
            readout_drive=readout_drive,
        )
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
    run_id = f"nbar_{n_bar:.0e}".replace("+", "")

    profile_path = OUT_DIR / f"profile_{run_id}.csv"
    with profile_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp_value in zip(state.x, xqp, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp_value)})

    row = {
        "run_id": run_id,
        "T_bath_K": T_BATH_K,
        "D0_um2_per_ns": D0,
        "tau_l_ns": TAU_L_NS,
        "source_rate_per_ns": source_rate,
        **_source_calibration(CONFIG, source_rate),
        "readout_resonator_index": float(READOUT_RESONATOR_INDEX),
        "readout_frequency_ghz": resonator.frequency_ghz,
        "readout_omega_uev": resonator.probe_energy_uev,
        "readout_n_bar_peak": n_bar,
        "readout_c_phot_ns_inv": C_PHOT_NS_INV,
        "dt_ns": CONFIG.dt_ns,
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
        "profile_csv": profile_path.name,
    }
    enriched_shift_rows = [
        {
            "run_id": run_id,
            "T_bath_K": T_BATH_K,
            "readout_n_bar_peak": n_bar,
            "readout_c_phot_ns_inv": C_PHOT_NS_INV,
            **shift_row,
        }
        for shift_row in shift_rows
    ]
    return row, enriched_shift_rows


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "metadata.json").open("w") as fp:
        json.dump(
            {
                "description": __doc__,
                "config": CONFIG.__dict__,
                "tau_l_ns": TAU_L_NS,
                "readout_resonator_index": READOUT_RESONATOR_INDEX,
                "readout_c_phot_ns_inv": C_PHOT_NS_INV,
                "n_bar_values": N_BAR_VALUES,
                "model_note": (
                    "Fixed peak nbar, local sub-gap photon scattering weighted "
                    "by the quarter-wave I^2 profile; no self-consistent "
                    "nbar(P_read) loop yet."
                ),
            },
            fp,
            indent=2,
        )

    summary_rows: list[dict[str, object]] = []
    shift_rows_all: list[dict[str, object]] = []
    for n_bar in N_BAR_VALUES:
        print(f"readout nbar={n_bar:.3g}", flush=True)
        row, shift_rows = _run_case(n_bar)
        summary_rows.append(row)
        shift_rows_all.extend(shift_rows)
        print(
            f"  t={row['total_time_ns']:.0f} ns, converged={row['converged']}, "
            f"xqp_mean={row['xqp_mean']:.3e}, "
            f"delta_fr=[{row['delta_fr_hz_min']:.3e}, {row['delta_fr_hz_max']:.3e}] Hz",
            flush=True,
        )

    _write_rows(OUT_DIR / "summary.csv", summary_rows)
    _write_rows(OUT_DIR / "resonator_shifts.csv", shift_rows_all)
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
