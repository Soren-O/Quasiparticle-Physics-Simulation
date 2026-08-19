"""Finite-escape phonon bottleneck sweep at the experimental 7 mK bath."""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.geometries import strip
from qpsim.backends.t3_spatial import T3SpatialState
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from scripts.run_prelim_spatial_finite_phonon_one import FinitePhononSpatialRunner
from scripts.run_prelim_spatial_overnight import (
    strip_coordinates,
    ENERGY_MAX_FACTOR,
    SweepConfig,
    _cell_centered_strip_grid,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_finite_phonon_sweep_7mk"
T_BATH_K = 0.007

CONFIG = SweepConfig(
    name="finite_phonon_sweep_7mk",
    NX=21,
    NE=28,
    dt_ns=1.0,
    max_time_ns=12_000.0,
    stop_tol=2e-9,
    snapshot_interval_ns=1_000.0,
    D0_values=(6.0, 20.0, 60.0),
    source_rates_per_ns=(1e-4, 5e-4, 1e-3),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)

TAU_L_VALUES_NS = (0.1, 0.3, 1.0, 3.0, 10.0)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E)
    return fermi_dirac_occupation(E, T)


def _build_state(D0: float) -> T3SpatialState:
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
        diffusion_coefficient=D0,
    )
    x, dx_um = _cell_centered_strip_grid(CONFIG.NX)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], CONFIG.NX, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=strip(
            int(np.asarray(x).size),
            mesh_size=float(dx_um),
        ),
        spectral=spectral,
        material=material,
        T_bath=T_BATH_K,
    )


def _run_case(D0: float, source_rate: float, tau_l_ns: float) -> tuple[
    dict[str, float | bool | str],
    list[dict[str, float]],
]:
    state = _build_state(D0)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=CONFIG.source_centers_delta[0],
        sigma_delta=CONFIG.source_sigmas_delta[0],
    )
    runner = FinitePhononSpatialRunner(state, tau_l_ns=tau_l_ns)
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
        # Converged only when BOTH residuals are quiet: the phonon field's
        # max|dn_ph/dt| lags max|df/dt| by up to ~8.7x on real
        # trajectories (2026-07-20 review); gating on f alone declared
        # convergence with the coupled phonons still moving.
        if max(max_dfdt, max_dnphdt) < CONFIG.stop_tol:
            converged = True
            break

    wall_s = time.monotonic() - start
    xqp = _xqp_profile(state)
    shift_rows = _resonator_shifts(state, f_ref)
    shifts = [row["delta_fr_hz_current_weighted"] for row in shift_rows]
    qis = [row["Qi_current_weighted"] for row in shift_rows]
    run_id = f"D0_{D0:g}_rate_{source_rate:.0e}_taul_{tau_l_ns:g}".replace(
        ".",
        "p",
    ).replace("+", "")

    profile_path = OUT_DIR / f"profile_{run_id}.csv"
    with profile_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp_value in zip(strip_coordinates(state), xqp, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp_value)})

    row = {
        "run_id": run_id,
        "T_bath_K": T_BATH_K,
        "D0_um2_per_ns": D0,
        "tau_l_ns": tau_l_ns,
        "source_rate_per_ns": source_rate,
        **_source_calibration(CONFIG, source_rate),
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
            "D0_um2_per_ns": D0,
            "tau_l_ns": tau_l_ns,
            "source_rate_per_ns": source_rate,
            **shift_row,
        }
        for shift_row in shift_rows
    ]
    return row, enriched_shift_rows


def _append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _write_metadata_atomically(path: Path, payload: dict[str, object]) -> None:
    """Replace campaign metadata without exposing a partial JSON document."""
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = OUT_DIR / "metadata.json"
    metadata: dict[str, object] = {
        "description": __doc__,
        "config": CONFIG.__dict__,
        "tau_l_values_ns": TAU_L_VALUES_NS,
        "model_note": (
            "Dynamic local Ph0 phonons with finite escape to bath; "
            "no lateral phonon transport."
        ),
    }
    summary_path = OUT_DIR / "summary.csv"
    shifts_path = OUT_DIR / "resonator_shifts.csv"
    total = len(CONFIG.D0_values) * len(CONFIG.source_rates_per_ns) * len(TAU_L_VALUES_NS)
    count = 0
    reset_aggregates = True
    summary_fields: list[str] | None = None
    shift_fields: list[str] | None = None
    start_all = time.monotonic()

    for D0 in CONFIG.D0_values:
        for source_rate in CONFIG.source_rates_per_ns:
            for tau_l_ns in TAU_L_VALUES_NS:
                count += 1
                print(
                    f"[{count}/{total}] D0={D0:g}, rate={source_rate:.0e}/ns, "
                    f"tau_l={tau_l_ns:g} ns",
                    flush=True,
                )
                row, shift_rows = _run_case(D0, source_rate, tau_l_ns)
                # Preserve the previous completed aggregates until one new
                # case has survived the integration and observable stages.
                # Keep its metadata too: replacing metadata before the first
                # successful case would falsely label preserved old CSVs as
                # results from the new configuration.
                if reset_aggregates:
                    _write_metadata_atomically(metadata_path, metadata)
                    summary_path.unlink(missing_ok=True)
                    shifts_path.unlink(missing_ok=True)
                    reset_aggregates = False
                if summary_fields is None:
                    summary_fields = list(row.keys())
                if shift_fields is None:
                    shift_fields = list(shift_rows[0].keys())
                _append_rows(summary_path, [row], summary_fields)
                _append_rows(shifts_path, shift_rows, shift_fields)
                print(
                    f"    xqp={row['xqp_mean']:.3e}, "
                    f"shift=[{row['delta_fr_hz_min']:.3e}, "
                    f"{row['delta_fr_hz_max']:.3e}] Hz, "
                    f"wall={row['wall_seconds']:.1f}s",
                    flush=True,
                )

    print(f"Completed {total} runs in {time.monotonic() - start_all:.1f}s")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
