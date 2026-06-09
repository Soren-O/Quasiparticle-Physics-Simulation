"""Focused grid/time-step convergence checks for the prelim 100 um strip."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_spatial_1d import T3Spatial1DState
from qpsim.constants import KB_UEV_PER_K
from qpsim.experiments.prelim_resonators import (
    AL_STRIP_LENGTH_UM,
    EXPERIMENT_BATH_TEMPERATURE_K,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from scripts.run_prelim_spatial_finite_phonon_one import FinitePhononSpatialRunner
from scripts.run_prelim_spatial_overnight import (
    ENERGY_MAX_FACTOR,
    SweepConfig,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_convergence_checks"

TARGET_SOURCE_QP_PER_S = 313_200_000_000.0
D0_UM2_PER_NS = 20.0
TAU_L_NS = 1.0
SOURCE_CENTER_DELTA = 2.0
SOURCE_SIGMA_DELTA = 0.08
STOP_TOL = 2e-9
MAX_TIME_NS = 12_000.0


@dataclass(frozen=True)
class ConvergenceCase:
    name: str
    NX: int
    NE: int
    dt_ns: float


CASES = (
    ConvergenceCase("coarse", NX=15, NE=20, dt_ns=2.0),
    ConvergenceCase("baseline", NX=21, NE=28, dt_ns=1.0),
    ConvergenceCase("dt_half", NX=21, NE=28, dt_ns=0.5),
    ConvergenceCase("energy_40", NX=21, NE=40, dt_ns=1.0),
    ConvergenceCase("space_41", NX=41, NE=28, dt_ns=1.0),
    ConvergenceCase("space_41_energy_40", NX=41, NE=40, dt_ns=1.0),
)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _case_config(case: ConvergenceCase, source_rate_per_ns: float) -> SweepConfig:
    return SweepConfig(
        name=case.name,
        NX=case.NX,
        NE=case.NE,
        dt_ns=case.dt_ns,
        max_time_ns=MAX_TIME_NS,
        stop_tol=STOP_TOL,
        snapshot_interval_ns=1_000.0,
        D0_values=(D0_UM2_PER_NS,),
        source_rates_per_ns=(source_rate_per_ns,),
        source_centers_delta=(SOURCE_CENTER_DELTA,),
        source_sigmas_delta=(SOURCE_SIGMA_DELTA,),
    )


def _source_rate_for_target_qps(case: ConvergenceCase) -> tuple[float, dict[str, float]]:
    unit_rate_config = _case_config(case, source_rate_per_ns=1.0)
    unit_calibration = _source_calibration(unit_rate_config, 1.0)
    qps_per_unit_rate = unit_calibration["estimated_source_qp_per_s"]
    source_rate = TARGET_SOURCE_QP_PER_S / qps_per_unit_rate
    config = _case_config(case, source_rate)
    return source_rate, _source_calibration(config, source_rate)


def _build_state(case: ConvergenceCase) -> T3Spatial1DState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=case.NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0_UM2_PER_NS,
    )
    x = np.linspace(0.0, AL_STRIP_LENGTH_UM, case.NX)
    f0 = np.repeat(
        _fermi_dirac(E, EXPERIMENT_BATH_TEMPERATURE_K)[:, None],
        case.NX,
        axis=1,
    )
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=EXPERIMENT_BATH_TEMPERATURE_K,
    )


def _relative_delta(value: float, reference: float) -> float:
    if reference == 0.0:
        return 0.0 if value == 0.0 else float("inf")
    return abs((value - reference) / reference)


def _write_profile(path: Path, state: T3Spatial1DState) -> None:
    xqp = _xqp_profile(state)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp_value in zip(state.x, xqp, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp_value)})


def _run_case(
    case: ConvergenceCase,
) -> tuple[dict[str, float | bool | str], list[dict[str, float | str]]]:
    source_rate, source_calibration = _source_rate_for_target_qps(case)
    config = _case_config(case, source_rate)
    state = _build_state(case)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=SOURCE_CENTER_DELTA,
        sigma_delta=SOURCE_SIGMA_DELTA,
    )
    runner = FinitePhononSpatialRunner(state, tau_l_ns=TAU_L_NS)

    start = time.monotonic()
    t_ns = 0.0
    n_steps = 0
    converged = False
    max_dfdt = float("inf")
    max_dnphdt = float("inf")
    while t_ns < config.max_time_ns:
        state, max_dfdt, max_dnphdt = runner.step(state, config.dt_ns, source)
        t_ns += config.dt_ns
        n_steps += 1
        if max_dfdt < config.stop_tol:
            converged = True
            break

    wall_s = time.monotonic() - start
    xqp = _xqp_profile(state)
    shift_rows = _resonator_shifts(state, f_ref)
    res1 = shift_rows[0]
    res6 = shift_rows[-1]
    profile_name = f"profile_{case.name}.csv"
    _write_profile(OUT_DIR / profile_name, state)

    row = {
        "case": case.name,
        "NX": case.NX,
        "NE": case.NE,
        "dt_ns": config.dt_ns,
        "dx_um": AL_STRIP_LENGTH_UM / (case.NX - 1),
        "D0_um2_per_ns": D0_UM2_PER_NS,
        "tau_l_ns": TAU_L_NS,
        "T_bath_K": EXPERIMENT_BATH_TEMPERATURE_K,
        "source_rate_per_ns": source_rate,
        **source_calibration,
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
        "res1_delta_fr_hz": res1["delta_fr_hz_current_weighted"],
        "res6_delta_fr_hz": res6["delta_fr_hz_current_weighted"],
        "res1_Qi": res1["Qi_current_weighted"],
        "res6_Qi": res6["Qi_current_weighted"],
        "res1_sigma1_norm": res1["sigma1_current_weighted_norm"],
        "res1_sigma2_norm": res1["sigma2_current_weighted_norm"],
        "profile_csv": profile_name,
    }
    enriched_shift_rows = [
        {
            "case": case.name,
            "NX": case.NX,
            "NE": case.NE,
            "dt_ns": config.dt_ns,
            **shift_row,
        }
        for shift_row in shift_rows
    ]
    return row, enriched_shift_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
                "target_source_qp_per_s": TARGET_SOURCE_QP_PER_S,
                "D0_um2_per_ns": D0_UM2_PER_NS,
                "tau_l_ns": TAU_L_NS,
                "T_bath_K": EXPERIMENT_BATH_TEMPERATURE_K,
                "source_center_delta": SOURCE_CENTER_DELTA,
                "source_sigma_delta": SOURCE_SIGMA_DELTA,
                "cases": [asdict(case) for case in CASES],
                "model_note": (
                    "Dynamic local Ph0 phonons with finite escape to bath; "
                    "total injected QP/s held fixed as dx changes."
                ),
            },
            fp,
            indent=2,
        )

    summary_rows: list[dict[str, object]] = []
    shift_rows_all: list[dict[str, object]] = []
    start_all = time.monotonic()
    for index, case in enumerate(CASES, start=1):
        print(
            f"[{index}/{len(CASES)}] {case.name}: "
            f"NX={case.NX}, NE={case.NE}, dt={case.dt_ns:g} ns",
            flush=True,
        )
        row, shift_rows = _run_case(case)
        summary_rows.append(row)
        shift_rows_all.extend(shift_rows)
        print(
            f"    shift1={row['res1_delta_fr_hz']:.3e} Hz, "
            f"xqp={row['xqp_mean']:.3e}, wall={row['wall_seconds']:.1f}s",
            flush=True,
        )

    baseline = next(row for row in summary_rows if row["case"] == "baseline")
    for row in summary_rows:
        row["rel_delta_xqp_mean_vs_baseline"] = _relative_delta(
            float(row["xqp_mean"]),
            float(baseline["xqp_mean"]),
        )
        row["rel_delta_res1_shift_vs_baseline"] = _relative_delta(
            float(row["res1_delta_fr_hz"]),
            float(baseline["res1_delta_fr_hz"]),
        )
        row["rel_delta_res6_shift_vs_baseline"] = _relative_delta(
            float(row["res6_delta_fr_hz"]),
            float(baseline["res6_delta_fr_hz"]),
        )
        row["rel_delta_res1_Qi_vs_baseline"] = _relative_delta(
            float(row["res1_Qi"]),
            float(baseline["res1_Qi"]),
        )

    _write_csv(OUT_DIR / "summary.csv", summary_rows)
    _write_csv(OUT_DIR / "resonator_shifts.csv", shift_rows_all)
    print(f"Completed {len(CASES)} convergence runs in {time.monotonic() - start_all:.1f}s")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
