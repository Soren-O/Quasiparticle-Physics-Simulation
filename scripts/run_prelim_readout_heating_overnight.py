"""Overnight finite-phonon/readout-heating sweep for the prelim 100 um strip.

This batch is intentionally deeper than the quick readout probe:

* 7 mK bath
* 1D spatial diffusion along the 100 um Al strip
* QP scattering/recombination with dynamic local phonons
* finite phonon escape to the bath
* injected QP source near 2 Delta_Al
* fixed-nbar sub-gap readout photon scattering weighted by I^2

Resume identity is source-sensitive as well as revision-sensitive: run ids carry a
``_PHYSICS_REV`` token plus exact point/config, physics-constant, and
source/material-code fingerprints, so stale rows are
never silently accepted as complete — they are re-run (bump the token
whenever runner physics or the summary schema changes). ``--no-resume``
restarts only the selected exact run ids while preserving other campaigns in
the shared output directory. The script can
stop cleanly after a wall-clock limit.  It does not yet run the
Fischer-style self-consistent nbar(P_read, Q_i, Q_c) loop; the overnight
sweep uses fixed peak nbar values.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.geometries import strip
from qpsim.backends.t3_spatial import T3SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.experiments.prelim_resonators import PRELIM_RESONATORS
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from scripts.run_prelim_spatial_finite_phonon_one import (
    FinitePhononSpatialRunner,
    readout_drive_from_resonator,
    snap_omega_to_grid,
)
from scripts.run_prelim_spatial_overnight import (
    strip_coordinates,
    ENERGY_MAX_FACTOR,
    LENGTH_UM,
    PROFILE_FIELDS,
    RESONATOR_SHIFT_FIELDS,
    SPATIAL_GRID_CONVENTION,
    _cell_centered_strip_grid,
    _atomic_write_csv,
    acquire_campaign_lock,
    purge_run_artifacts,
    purge_run_id_rows,
    purge_run_ids_rows,
    release_campaign_lock,
    require_matching_header,
    validate_campaign_aggregate_rows,
    validate_campaign_limits,
    _verify_completed_run_ids,
    _resonator_shifts,
    _runtime_environment,
    _source_calibration,
    _source_flux,
    _source_material_code_digest,
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


# NE must keep the two phonon lattices commensurate. The omega grid is the
# union of the difference lattice k*dE (scattering) and the sum lattice
# 2*E_face + m*dE (pair breaking), and they share bins only when 2*E_face/dE
# is an integer. On [Delta, 5*Delta] that means NE even: NE=101 gives 50.5 and
# leaves 49 difference bins above the pair threshold with no sum-lattice
# partner, so scattering-emitted phonons above 2*Delta cannot break pairs and
# recombination phonons cannot be reabsorbed by scattering. The two channels
# then evolve on disjoint sublattices -- an inconsistent discretisation rather
# than a coarse one, which converges to a different limit per grid parity.
# NE=100 measures 50.0 with zero orphans.
SMOKE_CONFIG = ReadoutOvernightConfig(
    name="smoke",
    NX=5,
    NE=100,
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
    NE=100,  # commensurate; see the note on SMOKE_CONFIG
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
    "NX",
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

SHIFT_FIELDS = [
    "run_id",
    "T_bath_K",
    "D0_um2_per_ns",
    "tau_l_ns",
    "source_rate_per_ns",
    "source_center_delta",
    "source_sigma_delta",
    "readout_resonator_index",
    "readout_n_bar_peak",
    "readout_c_phot_ns_inv",
    *RESONATOR_SHIFT_FIELDS,
]


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    return fermi_dirac_occupation(E, T)


def _build_state(config: ReadoutOvernightConfig, D0: float) -> T3SpatialState:
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
    x, dx_um = _cell_centered_strip_grid(config.NX)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], config.NX, axis=1)
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


# Physics/schema revision folded into every run id. BUMP THIS whenever the
# runner's physics model or output schema changes, so resume cannot silently
# accept rows computed with invalidated physics (2026-07-20 review: rev-less
# ids let pre-fix rows — legacy QP-side phonon kernels, silently snapped
# readout omega — satisfy the resume gate for the corrected model).
#   rev2 (2026-07-20): phonon-side kernels in the phonon equation (audit H1)
#     + explicit readout-omega grid snap with recorded shift (audit H2)
#     + readout_omega_* summary columns.
#   rev3 (2026-07-21): summary records NX; resume verifies the profile
#     cardinality and the trace endpoint against the committed summary.
#   rev4 (2026-07-21): source-shape and complete campaign configuration are
#     folded into the run id; provenance is config-addressed and atomic.
#   rev5 (2026-07-21): source shape is a true sweep dimension and exact
#     point/config/source fingerprints invalidate stale campaign artifacts.
#   rev6 (2026-07-21): shared finite-volume cell-center geometry makes the
#     readout campaign's diffusion, source volume, and observables use one
#     physical measure. Endpoint-grid artifacts are intentionally stale.
_PHYSICS_REV = "rev6"


def _physics_constants() -> dict[str, object]:
    """Return imported/module constants that affect campaign physics."""
    return {
        "C_PHOT_NS_INV": C_PHOT_NS_INV,
        "ENERGY_MAX_FACTOR": ENERGY_MAX_FACTOR,
        "KB_UEV_PER_K": KB_UEV_PER_K,
        "LENGTH_UM": LENGTH_UM,
        "SPATIAL_GRID_CONVENTION": SPATIAL_GRID_CONVENTION,
        "T_BATH_K": T_BATH_K,
    }


def _config_digest(config: ReadoutOvernightConfig) -> str:
    encoded = json.dumps(
        {
            "config": config.__dict__,
            "physics_constants": _physics_constants(),
            "physics_revision": _PHYSICS_REV,
            "runtime_environment": _runtime_environment(),
            "source_material_code_sha256": _source_material_code_digest(),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _point_digest(
    D0: float,
    source_rate: float,
    source_center_delta: float,
    source_sigma_delta: float,
    tau_l_ns: float,
    n_bar: float,
    readout_index: int,
) -> str:
    """Hash exact, round-trip point values instead of lossy id labels."""
    encoded = json.dumps(
        {
            "D0_um2_per_ns": float(D0),
            "readout_n_bar_peak": float(n_bar),
            "readout_resonator_index": int(readout_index),
            "source_center_delta": float(source_center_delta),
            "source_rate_per_ns": float(source_rate),
            "source_sigma_delta": float(source_sigma_delta),
            "tau_l_ns": float(tau_l_ns),
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_id(
    D0: float,
    source_rate: float,
    source_center_delta: float,
    source_sigma_delta: float,
    tau_l_ns: float,
    n_bar: float,
    readout_index: int,
    config: ReadoutOvernightConfig,
) -> str:
    # Fold the resolution-defining parameters into the id: otherwise the smoke
    # and overnight presets (which share the default OUT_DIR) produce identical
    # ids for the same physics point, and a resumed smoke result would silently
    # substitute for an overnight case.
    point_sha256 = _point_digest(
        D0,
        source_rate,
        source_center_delta,
        source_sigma_delta,
        tau_l_ns,
        n_bar,
        readout_index,
    )
    readable = (
        f"{_PHYSICS_REV}_D0_{D0:g}_rate_{source_rate:.0e}_taul_{tau_l_ns:g}_"
        f"nbar_{n_bar:.0e}_mode_{readout_index}_"
        f"center{source_center_delta:g}_sigma{source_sigma_delta:g}_"
        f"nx{config.NX}_ne{config.NE}_dt{config.dt_ns:g}_"
        f"tmax{config.max_time_ns:g}_tol{config.stop_tol:.0e}_"
        f"snap{config.snapshot_interval_ns:g}_"
        f"point{point_sha256[:20]}_"
        f"cfg{_config_digest(config)[:12]}"
    )
    return readable.replace("+", "").replace(".", "p")


def _append_rows(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: list[str],
) -> None:
    # Header when the file is missing OR zero-byte: a truncated/empty file
    # must not silently accumulate headerless rows (2026-07-20 review).
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerows(rows)
        fp.flush()


def _completed_run_ids(
    summary_path: Path,
    shifts_path: Path | None = None,
    out_dir: Path | None = None,
    *,
    expected_run_ids: set[str] | None = None,
    expected_rows: Mapping[str, Mapping[str, float | int]] | None = None,
    stop_tol: float | None = None,
) -> set[str]:
    """Run ids whose latest attempt has all six verified output rows/files."""
    return _verify_completed_run_ids(
        summary_path,
        shifts_path,
        out_dir,
        summary_fields=SUMMARY_FIELDS,
        shift_fields=SHIFT_FIELDS,
        trace_fields=TRACE_FIELDS,
        profile_fields=PROFILE_FIELDS,
        expected_run_ids=expected_run_ids,
        expected_rows=expected_rows,
        stop_tol=stop_tol,
    )


def _trace_row(
    state: T3SpatialState,
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


def _write_profile(path: Path, state: T3SpatialState) -> None:
    xqp = _xqp_profile(state)
    rows = [
        {"x_um": float(x_um), "xqp": float(xqp_value)}
        for x_um, xqp_value in zip(strip_coordinates(state), xqp, strict=True)
    ]
    _atomic_write_csv(path, PROFILE_FIELDS, rows)


def _promote_completed_trace(work_path: Path, final_path: Path) -> None:
    """Flush and atomically promote one fully assembled trace.

    Campaign output directories are single-writer by contract. If flushing
    or promotion fails, retain any previous complete final artifact and
    remove the inert work file so the next retry starts cleanly.
    """
    try:
        # Windows requires a write-capable descriptor for ``os.fsync``.
        with work_path.open("r+b") as trace_fp:
            os.fsync(trace_fp.fileno())
        os.replace(work_path, final_path)
    except BaseException:
        work_path.unlink(missing_ok=True)
        raise


def _expected_readout_snap(
    config: ReadoutOvernightConfig,
    readout_index: int,
) -> tuple[float, int, float]:
    """Reconstruct the exact photon-grid snap used by ``_run_case``."""
    material = load_material("Al")
    E, _ = build_energy_grid(
        gap=material.Delta_0,
        energy_min_factor=1.0,
        energy_max_factor=ENERGY_MAX_FACTOR,
        num_energy_bins=config.NE,
    )
    dE = integration_widths_from_centers(E)
    resonator = PRELIM_RESONATORS[readout_index - 1]
    return snap_omega_to_grid(float(resonator.probe_energy_uev), float(dE[0]))


def _run_case(
    config: ReadoutOvernightConfig,
    out_dir: Path,
    *,
    D0: float,
    source_rate: float,
    source_center_delta: float,
    source_sigma_delta: float,
    tau_l_ns: float,
    n_bar: float,
    readout_index: int,
) -> tuple[dict[str, object], list[dict[str, float | str]]]:
    # Refuse invalid integration controls before constructing state or
    # creating output files. A zero step can otherwise report a false
    # completion at t=0 for a quiet state, or loop forever for a driven one.
    for name, value, allow_zero in (
        ("dt_ns", config.dt_ns, False),
        ("max_time_ns", config.max_time_ns, True),
    ):
        if isinstance(value, (bool, np.bool_, str, bytes)) or np.iscomplexobj(
            value
        ):
            raise ValueError(f"{name} must be a finite real number.")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a finite real number.") from exc
        if not np.isfinite(numeric) or numeric < 0.0 or (
            not allow_zero and numeric == 0.0
        ):
            qualifier = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be finite and {qualifier}; got {value!r}.")

    state = _build_state(config, D0)
    f_ref = np.mean(state.f, axis=1)
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=source_center_delta,
        sigma_delta=source_sigma_delta,
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
    run_id = _run_id(
        D0,
        source_rate,
        source_center_delta,
        source_sigma_delta,
        tau_l_ns,
        n_bar,
        readout_index,
        config,
    )
    trace_path = out_dir / f"trace_{run_id}.csv"
    profile_path = out_dir / f"profile_{run_id}.csv"
    trace_work_path = trace_path.with_suffix(trace_path.suffix + ".tmp")
    # Preserve any prior complete artifacts until their replacements are
    # ready. The aggregate summary row was already purged by the caller, so
    # these old files cannot satisfy resume while this attempt is running.
    trace_work_path.unlink(missing_ok=True)

    start = time.monotonic()
    t_ns = 0.0
    n_steps = 0
    converged = False
    max_dfdt = float("inf")
    max_dnphdt = float("inf")
    next_snapshot_ns = 0.0
    last_trace_time_ns = t_ns

    _append_rows(
        trace_work_path,
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
        remaining_ns = config.max_time_ns - t_ns
        dt_step_ns = min(config.dt_ns, remaining_ns)
        state, max_dfdt, max_dnphdt = runner.step(
            state,
            dt_step_ns,
            source,
            readout_drive=readout_drive,
        )
        # Snap an intentionally shortened final step to the exact requested
        # horizon. This mirrors the backend/campaign resume invariant and
        # avoids a rounded sum landing just above max_time_ns.
        t_ns = (
            config.max_time_ns
            if dt_step_ns == remaining_ns
            else t_ns + dt_step_ns
        )
        n_steps += 1
        if t_ns >= next_snapshot_ns:
            _append_rows(
                trace_work_path,
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
            last_trace_time_ns = t_ns
            next_snapshot_ns += config.snapshot_interval_ns
        if max_dfdt < config.stop_tol and max_dnphdt < config.stop_tol:
            converged = True
            break

    # A run can converge between trace cadences. The summary is the commit
    # marker, so its total time must also exist as the final trace sample.
    if last_trace_time_ns != t_ns:
        _append_rows(
            trace_work_path,
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

    wall_s = time.monotonic() - start
    _write_profile(profile_path, state)
    # `_append_rows` closes each incremental write, but a close alone does
    # not guarantee that the completed temporary trace reached stable
    # storage before promotion. (`_atomic_write_csv` applies the same rule
    # to the profile.)
    _promote_completed_trace(trace_work_path, trace_path)
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
        "NX": config.NX,
        "T_bath_K": T_BATH_K,
        "D0_um2_per_ns": D0,
        "tau_l_ns": tau_l_ns,
        "source_rate_per_ns": source_rate,
        **_source_calibration(config, source_rate),
        "source_center_delta": source_center_delta,
        "source_sigma_delta": source_sigma_delta,
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
            "source_center_delta": source_center_delta,
            "source_sigma_delta": source_sigma_delta,
            "readout_resonator_index": readout_index,
            "readout_n_bar_peak": n_bar,
            "readout_c_phot_ns_inv": C_PHOT_NS_INV,
            **shift_row,
        }
        for shift_row in shift_rows
    ]
    return base_row, enriched_shift_rows


def _combinations(
    config: ReadoutOvernightConfig,
) -> list[tuple[float, float, float, float, float, float, int]]:
    return list(
        product(
            config.D0_values,
            config.source_rates_per_ns,
            config.source_centers_delta,
            config.source_sigmas_delta,
            config.tau_l_values_ns,
            config.n_bar_values,
            config.readout_resonator_indices,
        )
    )


def _write_metadata(out_dir: Path, config: ReadoutOvernightConfig) -> Path:
    config_sha256 = _config_digest(config)
    source_material_code_sha256 = _source_material_code_digest()
    metadata = {
        "physics_revision": _PHYSICS_REV,
        "config_sha256": config_sha256,
        "physics_constants": _physics_constants(),
        "runtime_environment": _runtime_environment(),
        "source_material_code_sha256": source_material_code_sha256,
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
        "spatial_grid_convention": SPATIAL_GRID_CONVENTION,
        "spatial_grid_note": (
            "Uniform finite-volume cell centers x_i=(i+1/2)L/NX with "
            "dx=L/NX, shared with the spatial overnight campaign."
        ),
        "model_note": (
            "Fixed peak nbar, local sub-gap photon scattering weighted "
            "by quarter-wave I^2; no self-consistent nbar(P_read) loop."
        ),
    }
    path = out_dir / f"metadata_{_PHYSICS_REV}_{config_sha256[:12]}.json"
    fd, temporary_name = tempfile.mkstemp(
        dir=out_dir,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True, allow_nan=False)
            fp.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "overnight"), default="overnight")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--wall-hours", type=float, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def _run_campaign(
    args: argparse.Namespace,
    config: ReadoutOvernightConfig,
    out_dir: Path,
) -> None:
    summary_path = out_dir / "summary.csv"
    shifts_path = out_dir / "resonator_shifts.csv"
    all_combinations = _combinations(config)
    combinations = (
        all_combinations
        if args.max_runs is None
        else all_combinations[: args.max_runs]
    )
    expected_run_ids = {
        _run_id(
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
            config,
        )
        for (
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
        ) in all_combinations
    }
    selected_run_ids = {
        _run_id(
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
            config,
        )
        for (
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
        ) in combinations
    }
    expected_rows: dict[str, dict[str, float | int]] = {}
    expected_snaps = {
        readout_index: _expected_readout_snap(config, readout_index)
        for readout_index in config.readout_resonator_indices
    }
    for (
        D0,
        source_rate,
        source_center_delta,
        source_sigma_delta,
        tau_l_ns,
        n_bar,
        readout_index,
    ) in all_combinations:
        run_id = _run_id(
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
            config,
        )
        resonator = PRELIM_RESONATORS[readout_index - 1]
        omega_used, omega_harmonic, omega_shift = expected_snaps[readout_index]
        expected_rows[run_id] = {
            "NX": config.NX,
            "T_bath_K": T_BATH_K,
            "D0_um2_per_ns": D0,
            "tau_l_ns": tau_l_ns,
            "source_rate_per_ns": source_rate,
            **_source_calibration(config, source_rate),
            "source_center_delta": source_center_delta,
            "source_sigma_delta": source_sigma_delta,
            "readout_resonator_index": readout_index,
            "readout_frequency_ghz": resonator.frequency_ghz,
            "readout_omega_uev": resonator.probe_energy_uev,
            "readout_omega_used_uev": omega_used,
            "readout_omega_grid_harmonic": omega_harmonic,
            "readout_omega_snap_rel_shift": omega_shift,
            "readout_n_bar_peak": n_bar,
            "readout_c_phot_ns_inv": C_PHOT_NS_INV,
            "dt_ns": config.dt_ns,
            "max_time_ns": config.max_time_ns,
        }
    if len(expected_run_ids) != len(all_combinations):
        raise RuntimeError(
            "Readout campaign generated duplicate run ids; every exact sweep "
            "point must have a unique persistent identity."
        )
    require_matching_header(summary_path, SUMMARY_FIELDS)
    require_matching_header(shifts_path, SHIFT_FIELDS)
    if args.no_resume:
        # Validate both shared aggregates before deleting any selected rows
        # or artifacts. Unattributable/non-canonical ids cannot be repaired
        # by a selected-id fresh start.
        validate_campaign_aggregate_rows(summary_path, SUMMARY_FIELDS)
        validate_campaign_aggregate_rows(shifts_path, SHIFT_FIELDS)
        # Restart only the selected exact run ids. The aggregate files are
        # intentionally shared by config-addressed presets, so wholesale
        # truncation would destroy other campaigns' resume state.
        purge_run_ids_rows(summary_path, selected_run_ids)
        purge_run_ids_rows(shifts_path, selected_run_ids)
        purge_run_artifacts(out_dir, selected_run_ids)
        completed: set[str] = set()
    else:
        completed = _completed_run_ids(
            summary_path,
            shifts_path,
            out_dir,
            expected_run_ids=expected_run_ids,
            expected_rows=expected_rows,
            stop_tol=config.stop_tol,
        )
    # Do not mutate provenance until resume/header validation succeeds.
    # Config-addressed files preserve every campaign represented in this
    # shared output directory instead of overwriting one ambiguous file.
    _write_metadata(out_dir, config)
    wall_limit_s = None if args.wall_hours is None else args.wall_hours * 3600.0
    start_all = time.monotonic()

    print(
        f"Preset {config.name}: {len(combinations)} queued runs, "
        f"dt={config.dt_ns:g} ns, NX={config.NX}, NE={config.NE}, "
        f"tmax={config.max_time_ns:g} ns.",
        flush=True,
    )

    for run_number, combination in enumerate(combinations, start=1):
        (
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
        ) = combination
        if wall_limit_s is not None and time.monotonic() - start_all >= wall_limit_s:
            print("Wall-time limit reached; stopping cleanly.", flush=True)
            break

        run_id = _run_id(
            D0,
            source_rate,
            source_center_delta,
            source_sigma_delta,
            tau_l_ns,
            n_bar,
            readout_index,
            config,
        )
        if run_id in completed:
            print(f"[{run_number}/{len(combinations)}] skip completed {run_id}", flush=True)
            continue

        print(f"[{run_number}/{len(combinations)}] start {run_id}", flush=True)
        # Idempotent re-run: drop stale rows from a prior failed or
        # non-converged attempt of this run id (2026-07-20 round-4 review).
        purge_run_id_rows(summary_path, run_id)
        purge_run_id_rows(shifts_path, run_id)
        try:
            summary_row, shift_rows = _run_case(
                config,
                out_dir,
                D0=D0,
                source_rate=source_rate,
                source_center_delta=source_center_delta,
                source_sigma_delta=source_sigma_delta,
                tau_l_ns=tau_l_ns,
                n_bar=n_bar,
                readout_index=readout_index,
            )
            # Shift rows FIRST, the summary row LAST: the summary row is the
            # commit marker resume gates on, so a crash between the two
            # writes leaves an orphan shift attempt (harmless, superseded on
            # re-run) rather than a "completed" summary with no shifts
            # (2026-07-20 review). Readers take the LAST row per run_id.
            _append_rows(shifts_path, shift_rows, SHIFT_FIELDS)
            _append_rows(summary_path, [summary_row], SUMMARY_FIELDS)
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
                "NX": config.NX,
                "T_bath_K": T_BATH_K,
                "D0_um2_per_ns": D0,
                "tau_l_ns": tau_l_ns,
                "source_rate_per_ns": source_rate,
                **_source_calibration(config, source_rate),
                "source_center_delta": source_center_delta,
                "source_sigma_delta": source_sigma_delta,
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


def main() -> None:
    args = _parse_args()
    args.max_runs, args.wall_hours = validate_campaign_limits(
        args.max_runs,
        args.wall_hours,
    )
    config = SMOKE_CONFIG if args.preset == "smoke" else OVERNIGHT_CONFIG
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_campaign_lock(
        out_dir,
        owner_label=f"readout:{config.name}",
        config_digest=_config_digest(config),
    )
    try:
        _run_campaign(args, config, out_dir)
    finally:
        release_campaign_lock(lock)


if __name__ == "__main__":
    main()
