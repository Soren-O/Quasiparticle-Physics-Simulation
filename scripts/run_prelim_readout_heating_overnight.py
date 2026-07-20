"""Overnight finite-phonon/readout-heating sweep for the prelim 100 um strip.

This batch is intentionally deeper than the quick readout probe:

* 7 mK bath
* 1D spatial diffusion along the 100 um Al strip
* QP scattering/recombination with dynamic local phonons
* finite phonon escape to the bath
* injected QP source near 2 Delta_Al
* fixed-nbar sub-gap readout photon scattering weighted by I^2

Resume is safe only WITHIN one physics revision: run ids carry a
``_PHYSICS_REV`` token, so rows produced by an older model revision are
never silently accepted as complete — they are re-run (bump the token
whenever runner physics or the summary schema changes). ``--no-resume``
truncates the aggregate CSVs for a genuinely fresh start. The script can
stop cleanly after a wall-clock limit.  It does not yet run the
Fischer-style self-consistent nbar(P_read, Q_i, Q_c) loop; the overnight
sweep uses fixed peak nbar values.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from itertools import product
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
    snap_omega_to_grid,
)
from scripts.run_prelim_spatial_overnight import (
    ENERGY_MAX_FACTOR,
    LENGTH_UM,
    _resonator_shifts,
    _source_calibration,
    _source_flux,
    _xqp_profile,
)


OUT_DIR = ROOT / "outputs" / "prelim_readout_heating_overnight"
T_BATH_K = 0.007
C_PHOT_NS_INV = 1e-9


@dataclass(frozen=True)
class ReadoutOvernightConfig:
    name: str
    NX: int
    NE: int
    dt_ns: float
    max_time_ns: float
    stop_tol: float
    snapshot_interval_ns: float
    D0_values: tuple[float, ...]
    source_rates_per_ns: tuple[float, ...]
    source_centers_delta: tuple[float, ...]
    source_sigmas_delta: tuple[float, ...]
    tau_l_values_ns: tuple[float, ...]
    n_bar_values: tuple[float, ...]
    readout_resonator_indices: tuple[int, ...]


SMOKE_CONFIG = ReadoutOvernightConfig(
    name="smoke",
    NX=5,
    NE=101,
    dt_ns=1.0,
    max_time_ns=3.0,
    stop_tol=1e-6,
    snapshot_interval_ns=1.0,
    D0_values=(20.0,),
    source_rates_per_ns=(5e-4,),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
    tau_l_values_ns=(1.0,),
    n_bar_values=(0.0, 1e5),
    readout_resonator_indices=(1,),
)

OVERNIGHT_CONFIG = ReadoutOvernightConfig(
    name="overnight",
    NX=21,
    NE=101,
    dt_ns=0.5,
    max_time_ns=30_000.0,
    stop_tol=5e-10,
    snapshot_interval_ns=2_500.0,
    # Ordered so the first block gives the nominal comparison first.
    D0_values=(20.0, 6.0, 60.0),
    source_rates_per_ns=(5e-4, 1e-4, 1e-3),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
    tau_l_values_ns=(1.0, 0.3, 3.0),
    n_bar_values=(0.0, 1e5, 1e6, 1e7),
    readout_resonator_indices=(1,),
)


SUMMARY_FIELDS = [
    "run_id",
    "status",
    "T_bath_K",
    "D0_um2_per_ns",
    "tau_l_ns",
    "source_rate_per_ns",
    "source_cell_volume_um3",
    "qps_per_xqp_source_cell",
    "estimated_source_qp_per_s",
    "source_center_delta",
    "source_sigma_delta",
    "readout_resonator_index",
    "readout_frequency_ghz",
    "readout_omega_uev",
    "readout_omega_used_uev",
    "readout_omega_grid_harmonic",
    "readout_omega_snap_rel_shift",
    "readout_n_bar_peak",
    "readout_c_phot_ns_inv",
    "dt_ns",
    "max_time_ns",
    "total_time_ns",
    "n_steps",
    "wall_seconds",
    "converged",
    "final_max_dfdt_per_ns",
    "final_max_dnphdt_per_ns",
    "xqp_mean",
    "xqp_source",
    "xqp_open_end",
    "nph_mean",
    "nph_max",
    "delta_fr_hz_min",
    "delta_fr_hz_max",
    "abs_delta_fr_hz_median",
    "Qi_min",
    "Qi_max",
    "trace_csv",
    "profile_csv",
    "error",
]

TRACE_FIELDS = [
    "t_ns",
    "max_dfdt_per_ns",
    "max_dnphdt_per_ns",
    "xqp_mean",
    "xqp_source",
    "xqp_open_end",
    "nph_mean",
    "nph_max",
]


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(config: ReadoutOvernightConfig, D0: float) -> T3Spatial1DState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=config.NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    x = np.linspace(0.0, LENGTH_UM, config.NX)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], config.NX, axis=1)
    return T3Spatial1DState(
        f=f0,
        x=x,
        gap=gap,
        spectral=spectral,
        material=material,
        T_bath=T_BATH_K,
    )


# Physics/schema revision folded into every run id. BUMP THIS whenever the
# runner's physics model or output schema changes, so resume cannot silently
# accept rows computed with invalidated physics (2026-07-20 review: rev-less
# ids let pre-fix rows — legacy QP-side phonon kernels, silently snapped
# readout omega — satisfy the resume gate for the corrected model).
#   rev2 (2026-07-20): phonon-side kernels in the phonon equation (audit H1)
#     + explicit readout-omega grid snap with recorded shift (audit H2)
#     + readout_omega_* summary columns.
_PHYSICS_REV = "rev2"


def _run_id(
    D0: float,
    source_rate: float,
    tau_l_ns: float,
    n_bar: float,
    readout_index: int,
    config: ReadoutOvernightConfig,
) -> str:
    # Fold the resolution-defining parameters into the id: otherwise the smoke
    # and overnight presets (which share the default OUT_DIR) produce identical
    # ids for the same physics point, and a resumed smoke result would silently
    # substitute for an overnight case.
    return (
        f"{_PHYSICS_REV}_D0_{D0:g}_rate_{source_rate:.0e}_taul_{tau_l_ns:g}_"
        f"nbar_{n_bar:.0e}_mode_{readout_index}_"
        f"nx{config.NX}_ne{config.NE}_dt{config.dt_ns:g}_"
        f"tmax{config.max_time_ns:g}_tol{config.stop_tol:.0e}"
    ).replace("+", "").replace(".", "p")


def _append_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
        fp.flush()


def _completed_run_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open() as fp:
        return {
            row["run_id"]
            for row in csv.DictReader(fp)
            if row.get("status") == "completed"
        }


def _trace_row(
    state: T3Spatial1DState,
    runner: FinitePhononSpatialRunner,
    *,
    t_ns: float,
    max_dfdt: float,
    max_dnphdt: float,
) -> dict[str, float]:
    xqp = _xqp_profile(state)
    return {
        "t_ns": t_ns,
        "max_dfdt_per_ns": max_dfdt,
        "max_dnphdt_per_ns": max_dnphdt,
        "xqp_mean": float(np.mean(xqp)),
        "xqp_source": float(xqp[0]),
        "xqp_open_end": float(xqp[-1]),
        "nph_mean": float(np.mean(runner.n_ph)),
        "nph_max": float(np.max(runner.n_ph)),
    }


def _write_profile(path: Path, state: T3Spatial1DState) -> None:
    xqp = _xqp_profile(state)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp_value in zip(state.x, xqp, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp_value)})


def _run_case(
    config: ReadoutOvernightConfig,
    out_dir: Path,
    *,
    D0: float,
    source_rate: float,
    tau_l_ns: float,
    n_bar: float,
    readout_index: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    state = _build_state(config, D0)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=config.source_centers_delta[0],
        sigma_delta=config.source_sigmas_delta[0],
    )
    resonator = PRELIM_RESONATORS[readout_index - 1]
    # Record the grid snap for every case (including the undriven baseline)
    # so nominal-vs-used photon energies are always in the output rows.
    omega_used, omega_harmonic, omega_shift = snap_omega_to_grid(
        float(resonator.probe_energy_uev), float(state.spectral.dE[0])
    )
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
    runner = FinitePhononSpatialRunner(state, tau_l_ns=tau_l_ns)
    run_id = _run_id(D0, source_rate, tau_l_ns, n_bar, readout_index, config)
    trace_path = out_dir / f"trace_{run_id}.csv"
    profile_path = out_dir / f"profile_{run_id}.csv"
    trace_path.unlink(missing_ok=True)
    profile_path.unlink(missing_ok=True)

    start = time.monotonic()
    t_ns = 0.0
    n_steps = 0
    converged = False
    max_dfdt = float("inf")
    max_dnphdt = float("inf")
    next_snapshot_ns = 0.0

    _append_rows(
        trace_path,
        [
            _trace_row(
                state,
                runner,
                t_ns=t_ns,
                max_dfdt=0.0,
                max_dnphdt=0.0,
            )
        ],
        TRACE_FIELDS,
    )

    while t_ns < config.max_time_ns:
        state, max_dfdt, max_dnphdt = runner.step(
            state,
            config.dt_ns,
            source,
            readout_drive=readout_drive,
        )
        t_ns += config.dt_ns
        n_steps += 1
        if t_ns >= next_snapshot_ns:
            _append_rows(
                trace_path,
                [
                    _trace_row(
                        state,
                        runner,
                        t_ns=t_ns,
                        max_dfdt=max_dfdt,
                        max_dnphdt=max_dnphdt,
                    )
                ],
                TRACE_FIELDS,
            )
            next_snapshot_ns += config.snapshot_interval_ns
        if max_dfdt < config.stop_tol and max_dnphdt < config.stop_tol:
            converged = True
            break

    wall_s = time.monotonic() - start
    _write_profile(profile_path, state)
    shift_rows = _resonator_shifts(state, f_ref)
    shifts = [float(row["delta_fr_hz_current_weighted"]) for row in shift_rows]
    qis = [float(row["Qi_current_weighted"]) for row in shift_rows]
    xqp = _xqp_profile(state)

    base_row: dict[str, object] = {
        "run_id": run_id,
        # Don't overload "completed": a run that hit max_time without meeting
        # stop_tol is not converged, and resume gates on status == "completed"
        # (so it would otherwise skip re-running a nonconverged case).
        "status": "completed" if converged else "max_time_reached",
        "T_bath_K": T_BATH_K,
        "D0_um2_per_ns": D0,
        "tau_l_ns": tau_l_ns,
        "source_rate_per_ns": source_rate,
        **_source_calibration(config, source_rate),
        "source_center_delta": config.source_centers_delta[0],
        "source_sigma_delta": config.source_sigmas_delta[0],
        "readout_resonator_index": readout_index,
        "readout_frequency_ghz": resonator.frequency_ghz,
        "readout_omega_uev": resonator.probe_energy_uev,
        "readout_omega_used_uev": omega_used,
        "readout_omega_grid_harmonic": float(omega_harmonic),
        "readout_omega_snap_rel_shift": omega_shift,
        "readout_n_bar_peak": n_bar,
        "readout_c_phot_ns_inv": C_PHOT_NS_INV,
        "dt_ns": config.dt_ns,
        "max_time_ns": config.max_time_ns,
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
        "delta_fr_hz_min": min(shifts),
        "delta_fr_hz_max": max(shifts),
        "abs_delta_fr_hz_median": float(np.median(np.abs(shifts))),
        "Qi_min": min(qis),
        "Qi_max": max(qis),
        "trace_csv": trace_path.name,
        "profile_csv": profile_path.name,
        "error": "",
    }
    enriched_shift_rows = [
        {
            "run_id": run_id,
            "T_bath_K": T_BATH_K,
            "D0_um2_per_ns": D0,
            "tau_l_ns": tau_l_ns,
            "source_rate_per_ns": source_rate,
            "readout_resonator_index": readout_index,
            "readout_n_bar_peak": n_bar,
            "readout_c_phot_ns_inv": C_PHOT_NS_INV,
            **shift_row,
        }
        for shift_row in shift_rows
    ]
    return base_row, enriched_shift_rows


def _combinations(config: ReadoutOvernightConfig) -> list[tuple[float, float, float, float, int]]:
    return list(
        product(
            config.D0_values,
            config.source_rates_per_ns,
            config.tau_l_values_ns,
            config.n_bar_values,
            config.readout_resonator_indices,
        )
    )


def _write_metadata(out_dir: Path, config: ReadoutOvernightConfig) -> None:
    with (out_dir / "metadata.json").open("w") as fp:
        json.dump(
            {
                "description": __doc__,
                "config": config.__dict__,
                "T_bath_K": T_BATH_K,
                "readout_c_phot_ns_inv": C_PHOT_NS_INV,
                "energy_grid_note": (
                    "The 5.142857 GHz readout mode is NOT grid-commensurate "
                    "at NE=101 (|omega - m*dE|/dE ~ 1.64%, above the kernel's "
                    "1% fail-loud tolerance). readout_drive_from_resonator "
                    "snaps the drive to the nearest grid harmonic m*dE and "
                    "each run row records readout_omega_uev (nominal) vs "
                    "readout_omega_used_uev (snapped) plus the relative "
                    "shift. An earlier note falsely claimed the mode was "
                    "within the 1% tolerance."
                ),
                "model_note": (
                    "Fixed peak nbar, local sub-gap photon scattering weighted "
                    "by quarter-wave I^2; no self-consistent nbar(P_read) loop."
                ),
            },
            fp,
            indent=2,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "overnight"), default="overnight")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = SMOKE_CONFIG if args.preset == "smoke" else OVERNIGHT_CONFIG
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(out_dir, config)

    summary_path = out_dir / "summary.csv"
    shifts_path = out_dir / "resonator_shifts.csv"
    if args.no_resume:
        # A fresh start must not append duplicate/wider rows beneath an old
        # header (2026-07-20 review): truncate the aggregate CSVs instead of
        # merely ignoring the resume gate. Per-run trace/profile files are
        # already unlinked per run id before rewriting.
        summary_path.unlink(missing_ok=True)
        shifts_path.unlink(missing_ok=True)
        completed: set[str] = set()
    else:
        completed = _completed_run_ids(summary_path)
    combinations = _combinations(config)
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]
    wall_limit_s = None if args.wall_hours is None else args.wall_hours * 3600.0
    start_all = time.monotonic()

    print(
        f"Preset {config.name}: {len(combinations)} queued runs, "
        f"dt={config.dt_ns:g} ns, NX={config.NX}, NE={config.NE}, "
        f"tmax={config.max_time_ns:g} ns.",
        flush=True,
    )

    shift_fields: list[str] | None = None
    if shifts_path.exists():
        with shifts_path.open() as fp:
            reader = csv.reader(fp)
            shift_fields = next(reader, None)

    for run_number, (D0, source_rate, tau_l_ns, n_bar, readout_index) in enumerate(
        combinations,
        start=1,
    ):
        if wall_limit_s is not None and time.monotonic() - start_all >= wall_limit_s:
            print("Wall-time limit reached; stopping cleanly.", flush=True)
            break

        run_id = _run_id(D0, source_rate, tau_l_ns, n_bar, readout_index, config)
        if run_id in completed:
            print(f"[{run_number}/{len(combinations)}] skip completed {run_id}", flush=True)
            continue

        print(f"[{run_number}/{len(combinations)}] start {run_id}", flush=True)
        try:
            summary_row, shift_rows = _run_case(
                config,
                out_dir,
                D0=D0,
                source_rate=source_rate,
                tau_l_ns=tau_l_ns,
                n_bar=n_bar,
                readout_index=readout_index,
            )
            if shift_fields is None:
                shift_fields = list(shift_rows[0].keys())
            _append_rows(summary_path, [summary_row], SUMMARY_FIELDS)
            _append_rows(shifts_path, shift_rows, shift_fields)
            completed.add(run_id)
            print(
                f"[{run_number}/{len(combinations)}] done {run_id}: "
                f"xqp_mean={summary_row['xqp_mean']:.3e}, "
                f"delta_fr=[{summary_row['delta_fr_hz_min']:.3e}, "
                f"{summary_row['delta_fr_hz_max']:.3e}] Hz, "
                f"Qi=[{summary_row['Qi_min']:.3e}, {summary_row['Qi_max']:.3e}], "
                f"converged={summary_row['converged']}, "
                f"wall={summary_row['wall_seconds']:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failure_row = {
                "run_id": run_id,
                "status": "failed",
                "T_bath_K": T_BATH_K,
                "D0_um2_per_ns": D0,
                "tau_l_ns": tau_l_ns,
                "source_rate_per_ns": source_rate,
                **_source_calibration(config, source_rate),
                "source_center_delta": config.source_centers_delta[0],
                "source_sigma_delta": config.source_sigmas_delta[0],
                "readout_resonator_index": readout_index,
                "readout_n_bar_peak": n_bar,
                "readout_c_phot_ns_inv": C_PHOT_NS_INV,
                "dt_ns": config.dt_ns,
                "max_time_ns": config.max_time_ns,
                "error": repr(exc),
            }
            _append_rows(summary_path, [failure_row], SUMMARY_FIELDS)
            print(f"[{run_number}/{len(combinations)}] FAILED {run_id}: {exc!r}", flush=True)

    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
