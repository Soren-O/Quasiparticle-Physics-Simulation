"""Overnight 1D spatial sweep for the prelim validation experiment.

This is the slower companion to ``run_prelim_spatial_100um.py``.  It
uses a finer time step and mesh, sweeps diffusion/source assumptions, and
reports absolute resonant-frequency shifts for the six target modes from
the TeX presentation:

    f_n = 5 + n / 7 GHz, n = 1, ..., 6.

The source is still normalized as a local ``x_qp`` generation rate.  Once
the SIS I(V) / junction-resistance calibration is ready, replace that
normalization with a spectral current-to-gain conversion.
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

from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState, T3SpatialFlux1D
from qpsim.constants import KB_UEV_PER_K
from qpsim.experiments.prelim_resonators import (
    AL_STRIP_LENGTH_UM,
    AL_STRIP_THICKNESS_UM,
    AL_STRIP_WIDTH_UM,
    COUPLING_LENGTH_UM,
    FIXED_RESONATOR_LENGTH_UM,
    KINETIC_INDUCTANCE_FRACTION,
    PRELIM_RESONATORS,
    TARGET_RESONATOR_FREQUENCIES_GHZ,
    TARGET_RESONATOR_LENGTHS_UM,
    VARIABLE_PIECE_LENGTHS_UM,
    current_squared_profile,
    full_resonator_current_integral_um,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.observables.spatial_ac_response import compute_current_weighted_ac_response
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext


LENGTH_UM = AL_STRIP_LENGTH_UM
T_BATH_K = 0.1
ENERGY_MAX_FACTOR = 5.0
# Max transport diffusion number D0*dt/dx^2 per run; each config's dt_ns is a
# cap that is reduced per D0 below. The spatial Crank-Nicolson step clips [0,1]
# over/undershoot and the backend raises once a step alters >0.1% of the
# conserved density (~diffusion number 11); 4 keeps a clip-free margin.
CFL_TARGET = 4.0
N_SUBGAP_QUAD = 120


@dataclass(frozen=True)
class SweepConfig:
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


SMOKE_CONFIG = SweepConfig(
    name="smoke",
    NX=21,
    NE=24,
    dt_ns=2.0,
    max_time_ns=20_000.0,
    stop_tol=2e-10,
    snapshot_interval_ns=2_000.0,
    D0_values=(6.0, 60.0),
    source_rates_per_ns=(2e-5,),
    source_centers_delta=(2.0,),
    source_sigmas_delta=(0.08,),
)

OVERNIGHT_CONFIG = SweepConfig(
    name="overnight",
    NX=41,
    NE=40,
    dt_ns=1.0,
    max_time_ns=60_000.0,
    stop_tol=5e-11,
    snapshot_interval_ns=5_000.0,
    D0_values=(3.0, 6.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0),
    source_rates_per_ns=(1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7),
    source_centers_delta=(1.95, 2.0, 2.1),
    source_sigmas_delta=(0.05, 0.08),
)

CALIBRATED_CONFIG = SweepConfig(
    name="calibrated",
    NX=41,
    NE=40,
    dt_ns=1.0,
    max_time_ns=60_000.0,
    stop_tol=5e-11,
    snapshot_interval_ns=5_000.0,
    D0_values=(3.0, 6.0, 10.0, 20.0, 40.0, 60.0, 100.0, 150.0),
    # For the current 2.5 um source cell, these span roughly
    # 3e10--3e12 QP/s, covering the SIS threshold estimate.
    source_rates_per_ns=(1e-2, 3e-3, 1e-3, 3e-4, 1e-4),
    source_centers_delta=(1.95, 2.0, 2.1),
    source_sigmas_delta=(0.05, 0.08),
)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(config: SweepConfig, D0: float) -> T3Spatial1DState:
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


def _source_flux(
    state: T3Spatial1DState,
    *,
    local_xqp_generation_rate_per_ns: float,
    center_delta: float,
    sigma_delta: float,
) -> T3SpatialFlux1D:
    center = center_delta * state.gap
    sigma = sigma_delta * state.gap
    profile = np.exp(-0.5 * ((state.spectral.E - center) / sigma) ** 2)
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    xqp_norm = float(np.sum(spectral_weights * profile) / state.gap)
    if xqp_norm <= 0.0:
        raise RuntimeError("Could not normalize source spectrum.")
    gain_spectrum = local_xqp_generation_rate_per_ns * profile / xqp_norm

    gain = np.zeros_like(state.f)
    gain[:, 0] = gain_spectrum
    return T3SpatialFlux1D(
        gain=gain,
        loss_rate=np.zeros_like(gain),
        diagnostics={
            "local_xqp_generation_rate_per_ns": local_xqp_generation_rate_per_ns,
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
            "source_center_uev": center,
            "source_sigma_uev": sigma,
        },
    )


def _source_calibration(
    config: SweepConfig,
    source_rate_per_ns: float,
) -> dict[str, float]:
    """Estimate QP/s represented by a local source-cell x_qp rate.

    The Al database value ``rho_F = 1.74e28`` is the customary Al DOS in
    eV^-1 m^-3, so pair it with ``Delta`` in eV here.
    """
    material = load_material("Al")
    dx_um = LENGTH_UM / (config.NX - 1)
    source_cell_volume_um3 = dx_um * AL_STRIP_WIDTH_UM * AL_STRIP_THICKNESS_UM
    source_cell_volume_m3 = source_cell_volume_um3 * 1e-18
    delta_eV = material.Delta_0 * 1e-6
    qps_per_xqp_source_cell = (
        4.0 * float(material.rho_F) * delta_eV * source_cell_volume_m3
    )
    return {
        "source_cell_volume_um3": source_cell_volume_um3,
        "qps_per_xqp_source_cell": qps_per_xqp_source_cell,
        "estimated_source_qp_per_s": (
            source_rate_per_ns * 1e9 * qps_per_xqp_source_cell
        ),
    }


def _xqp_profile(state: T3Spatial1DState) -> np.ndarray:
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    return np.sum(spectral_weights[:, None] * state.f, axis=0) / state.gap


def _mean_f(state: T3Spatial1DState) -> np.ndarray:
    return np.mean(state.f, axis=1)


def _current_weights(state: T3Spatial1DState, resonator_length_um: float) -> np.ndarray:
    """Current-squared profile over the strip at the shorted end.

    Gao writes the modal weighting as ``sin^2(pi x / 2l)`` with ``x`` measured
    from the open end, so a strip coordinate ``s`` measured away from the short
    maps to ``cos^2(pi s / 2l)``.
    """
    return current_squared_profile(state.x, resonator_length_um)


def _current_weighted_f(
    state: T3Spatial1DState,
    resonator_length_um: float,
) -> tuple[np.ndarray, float]:
    weights = _current_weights(state, resonator_length_um)
    norm_um = float(np.trapezoid(weights, state.x))
    if norm_um <= 0.0:
        raise RuntimeError("Current weighting normalization vanished.")
    weighted_f = np.trapezoid(state.f * weights[None, :], state.x, axis=1) / norm_um

    # Integral of cos^2(pi s / 2l) over the full quarter-wave resonator is l/2.
    current_participation = norm_um / (0.5 * resonator_length_um)
    return weighted_f, current_participation


def _current_weighted_xqp(
    state: T3Spatial1DState,
    resonator_length_um: float,
) -> float:
    weights = _current_weights(state, resonator_length_um)
    norm_um = float(np.trapezoid(weights, state.x))
    profile = _xqp_profile(state)
    return float(np.trapezoid(profile * weights, state.x) / norm_um)


def _trace_observables() -> dict[str, object]:
    def xqp_mean(state: T3Spatial1DState) -> float:
        return float(np.mean(_xqp_profile(state)))

    def xqp_source(state: T3Spatial1DState) -> float:
        return float(_xqp_profile(state)[0])

    def xqp_open_end(state: T3Spatial1DState) -> float:
        return float(_xqp_profile(state)[-1])

    return {
        "xqp_mean": xqp_mean,
        "xqp_source": xqp_source,
        "xqp_open_end": xqp_open_end,
    }


def _resonator_shifts(
    state: T3Spatial1DState,
    f_ref: np.ndarray,
) -> list[dict[str, float]]:
    f_uniform = _mean_f(state)
    rows = []
    for resonator in PRELIM_RESONATORS:
        f0_ghz = resonator.frequency_ghz
        length_um = resonator.total_length_um
        probe_energy = resonator.probe_energy_uev
        response = compute_current_weighted_ac_response(
            state.f,
            f_ref,
            state.x,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            current_weights=_current_weights(state, length_um),
            full_current_integral_um=full_resonator_current_integral_um(length_um),
            n_subgap=N_SUBGAP_QUAD,
        )
        delta_hz_current_weighted = response.frac_freq_shift * f0_ghz * 1e9

        f_current_weighted, _ = _current_weighted_f(state, resonator_length_um=length_um)
        f_ref_current_weighted, _ = _current_weighted_f(
            T3Spatial1DState(
                f=np.repeat(f_ref[:, None], state.x.size, axis=1),
                x=state.x,
                gap=state.gap,
                spectral=state.spectral,
                material=state.material,
                T_bath=state.T_bath,
            ),
            resonator_length_um=length_um,
        )
        qi_current_weighted = compute_quality_factor(
            f_current_weighted,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_strip_only = compute_frequency_shift(
            f_current_weighted,
            f_ref_current_weighted,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_weighted_f_legacy = (
            frac_shift_strip_only * response.current_participation
        )

        qi_uniform = compute_quality_factor(
            f_uniform,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        frac_shift_uniform = compute_frequency_shift(
            f_uniform,
            f_ref,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=N_SUBGAP_QUAD,
        )
        delta_hz_uniform = frac_shift_uniform * f0_ghz * 1e9
        rows.append(
            {
                "resonator_index": float(resonator.index),
                "resonator_label": resonator.label,
                "f0_ghz": f0_ghz,
                "resonator_length_um": length_um,
                "variable_piece_length_um": resonator.variable_piece_length_um,
                "current_participation": response.current_participation,
                "current_integral_strip_um": response.strip_current_integral_um,
                "current_integral_full_um": response.full_current_integral_um,
                "xqp_current_weighted": _current_weighted_xqp(state, length_um),
                "sigma1_current_weighted_norm": response.sigma1_current_weighted_norm,
                "sigma2_current_weighted_norm": response.sigma2_current_weighted_norm,
                "sigma1_ref_current_weighted_norm": (
                    response.sigma1_ref_current_weighted_norm
                ),
                "sigma2_ref_current_weighted_norm": (
                    response.sigma2_ref_current_weighted_norm
                ),
                "relative_sigma2_change_current_weighted": (
                    response.relative_sigma2_change_current_weighted
                ),
                "inverse_Qi_current_weighted": response.inverse_qi_qp,
                "Qi_current_weighted": response.qi_qp,
                "Qi_weighted_f_legacy": qi_current_weighted,
                "frac_freq_shift_current_weighted": response.frac_freq_shift,
                "delta_fr_hz_current_weighted": delta_hz_current_weighted,
                "shifted_fr_ghz_current_weighted": (
                    f0_ghz + delta_hz_current_weighted / 1e9
                ),
                "frac_freq_shift_weighted_f_legacy": frac_shift_weighted_f_legacy,
                "delta_fr_hz_weighted_f_legacy": (
                    frac_shift_weighted_f_legacy * f0_ghz * 1e9
                ),
                "Qi_uniform_strip_average": qi_uniform,
                "frac_freq_shift_uniform_strip_average": frac_shift_uniform,
                "delta_fr_hz_uniform_strip_average": delta_hz_uniform,
            }
        )
    return rows


def require_matching_header(path: Path, fieldnames: list[str]) -> None:
    """Refuse to append rows beneath a stale aggregate-CSV header.

    Appending a wider/reordered row set under an old header silently
    mislabels every later column (2026-07-20 review: 40-column rev2 rows
    under a 37-column legacy header shifted values across fields, with
    overflow landing in csv.DictReader's None key). A schema change
    requires ``--no-resume`` (which truncates) or a fresh output
    directory. Shared by the spatial and readout overnight runners.
    """
    if not path.exists():
        return
    with path.open("r", newline="") as fp:
        first = fp.readline().strip()
    if not first:
        return
    existing = first.split(",")
    if existing != list(fieldnames):
        raise SystemExit(
            f"{path} has a {len(existing)}-column header that does not match "
            f"the current {len(fieldnames)}-column schema. Appending would "
            "silently mislabel columns. Rerun with --no-resume (truncates "
            "the aggregate CSVs) or use a fresh --out-dir."
        )


def purge_run_id_rows(path: Path, run_id: str) -> None:
    """Atomically remove every row for ``run_id`` before a re-run.

    A retried case (failed or max_time_reached) previously APPENDED a
    second row set for the same run id, leaving duplicates and orphan
    shift rows that no reader disambiguates (2026-07-20 round-4 review).
    Purging first makes re-runs idempotent: the files contain exactly one
    attempt per run id. Shared by the spatial and readout runners.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="") as fp:
        rows = list(csv.reader(fp))
    if not rows:
        return
    header, data = rows[0], rows[1:]
    try:
        idx = header.index("run_id")
    except ValueError:
        return
    kept = [r for r in data if len(r) <= idx or r[idx] != run_id]
    if len(kept) == len(data):
        return
    tmp = path.with_suffix(path.suffix + ".purge.tmp")
    with tmp.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerows(kept)
    tmp.replace(path)


def _append_csv(path: Path, row: dict[str, object], fieldnames: list[str]) -> None:
    # Header when the file is missing OR zero-byte: a truncated/empty file
    # must not silently accumulate headerless rows (2026-07-20 review).
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
        fp.flush()


def _write_trace(path: Path, snapshots: list[object]) -> None:
    fieldnames = ["t_ns", "max_dfdt_per_ns", "xqp_mean", "xqp_source", "xqp_open_end"]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for snap in snapshots:
            writer.writerow(
                {
                    "t_ns": snap.t,
                    "max_dfdt_per_ns": snap.max_rate,
                    **snap.observables,
                }
            )


def _write_profile(path: Path, state: T3Spatial1DState) -> None:
    profile = _xqp_profile(state)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp in zip(state.x, profile, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp)})


# Physics/schema revision folded into every run id — BUMP whenever the
# runner's physics or summary schema changes so resume cannot silently
# accept invalidated rows (2026-07-20 review).
#   rev2 (2026-07-20): non-converged runs record status=max_time_reached
#     (previously mislabeled "completed" and permanently resume-excluded).
_PHYSICS_REV = "rev2"


def _run_id(
    D0: float,
    rate: float,
    center_delta: float,
    sigma_delta: float,
) -> str:
    return (
        f"{_PHYSICS_REV}_D0_{D0:g}_rate_{rate:.0e}_center_{center_delta:g}"
        f"_sigma_{sigma_delta:g}"
        .replace("+", "")
        .replace(".", "p")
    )


def _completed_run_ids(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open() as fp:
        return {
            row["run_id"]
            for row in csv.DictReader(fp)
            if row.get("status") == "completed"
        }


def _write_metadata(out_dir: Path, config: SweepConfig) -> None:
    metadata = {
        "preset": config.name,
        "target_resonator_frequencies_ghz": TARGET_RESONATOR_FREQUENCIES_GHZ,
        "target_resonator_lengths_um": TARGET_RESONATOR_LENGTHS_UM,
        "fixed_resonator_length_um": FIXED_RESONATOR_LENGTH_UM,
        "coupling_length_um": COUPLING_LENGTH_UM,
        "variable_piece_lengths_um": VARIABLE_PIECE_LENGTHS_UM,
        "kinetic_inductance_fraction": KINETIC_INDUCTANCE_FRACTION,
        "length_um": LENGTH_UM,
        "T_bath_K": T_BATH_K,
        "energy_max_factor": ENERGY_MAX_FACTOR,
        "n_subgap_quad": N_SUBGAP_QUAD,
        "config": config.__dict__,
        "source_note": (
            "Source is normalized as a local source-cell x_qp generation rate. "
            "CSV fields estimate QP/s using the source-cell volume and the "
            "Al rho_F database value as eV^-1 m^-3."
        ),
    }
    with (out_dir / "metadata.json").open("w") as fp:
        json.dump(metadata, fp, indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "overnight", "calibrated"),
        default="overnight",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "outputs" / "prelim_spatial_overnight",
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_by_name = {
        "smoke": SMOKE_CONFIG,
        "overnight": OVERNIGHT_CONFIG,
        "calibrated": CALIBRATED_CONFIG,
    }
    config = config_by_name[args.preset]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(out_dir, config)

    summary_path = out_dir / "summary.csv"
    shifts_path = out_dir / "resonator_shifts.csv"
    if args.no_resume:
        # Fresh start: truncate the aggregate CSVs so a rerun cannot append
        # duplicate or schema-shifted rows beneath an old header
        # (2026-07-20 review).
        summary_path.unlink(missing_ok=True)
        shifts_path.unlink(missing_ok=True)
        completed: set[str] = set()
    else:
        completed = _completed_run_ids(summary_path)

    summary_fields = [
        "run_id",
        "status",
        "D0_um2_per_ns",
        "source_rate_per_ns",
        "estimated_source_qp_per_s",
        "source_cell_volume_um3",
        "qps_per_xqp_source_cell",
        "source_center_delta",
        "source_sigma_delta",
        "dt_ns",
        "max_time_ns",
        "total_time_ns",
        "n_steps",
        "converged",
        "final_max_dfdt_per_ns",
        "xqp_mean",
        "xqp_source",
        "xqp_open_end",
        "delta_fr_hz_min",
        "delta_fr_hz_max",
        "error",
        "trace_csv",
        "profile_csv",
    ]
    shift_fields = [
        "run_id",
        "D0_um2_per_ns",
        "source_rate_per_ns",
        "estimated_source_qp_per_s",
        "source_cell_volume_um3",
        "qps_per_xqp_source_cell",
        "source_center_delta",
        "source_sigma_delta",
        "resonator_index",
        "resonator_label",
        "f0_ghz",
        "resonator_length_um",
        "variable_piece_length_um",
        "current_participation",
        "current_integral_strip_um",
        "current_integral_full_um",
        "xqp_current_weighted",
        "sigma1_current_weighted_norm",
        "sigma2_current_weighted_norm",
        "sigma1_ref_current_weighted_norm",
        "sigma2_ref_current_weighted_norm",
        "relative_sigma2_change_current_weighted",
        "inverse_Qi_current_weighted",
        "Qi_current_weighted",
        "Qi_weighted_f_legacy",
        "frac_freq_shift_current_weighted",
        "delta_fr_hz_current_weighted",
        "shifted_fr_ghz_current_weighted",
        "frac_freq_shift_weighted_f_legacy",
        "delta_fr_hz_weighted_f_legacy",
        "Qi_uniform_strip_average",
        "frac_freq_shift_uniform_strip_average",
        "delta_fr_hz_uniform_strip_average",
    ]

    if not args.no_resume:
        require_matching_header(summary_path, summary_fields)
        require_matching_header(shifts_path, shift_fields)

    combinations = list(
        product(
            config.D0_values,
            config.source_rates_per_ns,
            config.source_centers_delta,
            config.source_sigmas_delta,
        )
    )
    if args.max_runs is not None:
        combinations = combinations[: args.max_runs]

    start_wall = time.monotonic()
    wall_limit_s = None if args.wall_hours is None else args.wall_hours * 3600.0
    print(
        f"Preset {config.name}: {len(combinations)} queued runs, "
        f"dt={config.dt_ns:g} ns, NX={config.NX}, NE={config.NE}.",
        flush=True,
    )

    for run_number, (D0, rate, center_delta, sigma_delta) in enumerate(combinations, start=1):
        if wall_limit_s is not None and time.monotonic() - start_wall >= wall_limit_s:
            print("Wall-time limit reached; stopping cleanly.", flush=True)
            break

        run_id = _run_id(D0, rate, center_delta, sigma_delta)
        if run_id in completed:
            print(f"[{run_number}/{len(combinations)}] skip completed {run_id}", flush=True)
            continue

        print(f"[{run_number}/{len(combinations)}] start {run_id}", flush=True)
        # Idempotent re-run: drop any stale rows from a prior failed or
        # non-converged attempt of this run id (2026-07-20 round-4 review).
        purge_run_id_rows(summary_path, run_id)
        purge_run_id_rows(shifts_path, run_id)
        trace_path = out_dir / f"trace_{run_id}.csv"
        profile_path = out_dir / f"profile_{run_id}.csv"
        source_calibration = _source_calibration(config, rate)

        # Bound the transport diffusion number per D0 (see CFL_TARGET): the
        # spatial Crank-Nicolson step clip-raises on an under-resolved dt.
        dx_um = LENGTH_UM / (config.NX - 1)
        dt_run = min(config.dt_ns, CFL_TARGET * dx_um * dx_um / D0)

        base_row: dict[str, object] = {
            "run_id": run_id,
            "D0_um2_per_ns": D0,
            "source_rate_per_ns": rate,
            **source_calibration,
            "source_center_delta": center_delta,
            "source_sigma_delta": sigma_delta,
            "dt_ns": dt_run,
            "max_time_ns": config.max_time_ns,
            "trace_csv": trace_path.name,
            "profile_csv": profile_path.name,
        }

        try:
            state = _build_state(config, D0)
            f_ref = _mean_f(state)
            flux = _source_flux(
                state,
                local_xqp_generation_rate_per_ns=rate,
                center_delta=center_delta,
                sigma_delta=sigma_delta,
            )
            backend = T3Spatial1DBackend()
            result = backend.run_until_steady_state(
                state,
                dt=dt_run,
                max_time=config.max_time_ns,
                external_flux=flux,
                stop_tol=config.stop_tol,
                snapshot_interval=config.snapshot_interval_ns,
                observables=_trace_observables(),
            )
            _write_trace(trace_path, result.snapshots)
            _write_profile(profile_path, result.state)
            shift_rows = _resonator_shifts(result.state, f_ref)
            for shift_row in shift_rows:
                _append_csv(
                    shifts_path,
                    {
                        **base_row,
                        **shift_row,
                    },
                    shift_fields,
                )

            final_obs = result.snapshots[-1].observables
            delta_values = [row["delta_fr_hz_current_weighted"] for row in shift_rows]
            # Don't overload "completed": a run that hit max_time without
            # meeting stop_tol is not converged, and resume gates on
            # status == "completed" — marking it completed would both
            # misrepresent the row and permanently exclude the case from
            # re-running on resume (2026-07-19 audit).
            summary_row = {
                **base_row,
                "status": "completed" if result.converged else "max_time_reached",
                "total_time_ns": result.total_time,
                "n_steps": result.n_steps,
                "converged": result.converged,
                "final_max_dfdt_per_ns": result.snapshots[-1].max_rate,
                **final_obs,
                "delta_fr_hz_min": min(delta_values),
                "delta_fr_hz_max": max(delta_values),
                "error": "",
            }
            _append_csv(summary_path, summary_row, summary_fields)
            print(
                f"[{run_number}/{len(combinations)}] done {run_id}: "
                f"xqp_mean={final_obs['xqp_mean']:.3e}, "
                f"delta_fr=[{min(delta_values):.3e}, {max(delta_values):.3e}] Hz, "
                f"converged={result.converged}",
                flush=True,
            )
        except Exception as exc:
            _append_csv(
                summary_path,
                {
                    **base_row,
                    "status": "failed",
                    "error": repr(exc),
                },
                summary_fields,
            )
            print(f"[{run_number}/{len(combinations)}] FAILED {run_id}: {exc!r}", flush=True)


if __name__ == "__main__":
    main()
