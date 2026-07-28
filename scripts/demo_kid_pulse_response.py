"""Paper-2 capability demo: KID pulse response via the transient service.

A pair-breaking photon drive (Fischer-2024 Al parameters, omega_PB =
2.8 Delta) switches ON at t = 0 for a 30 ns pulse, then OFF; f(E, t)
evolves through rise and recombination-limited decay. Each snapshot is
mapped through the Mattis-Bardeen observable chain to the detector
response: fractional frequency shift df_r/f_r and internal quality
factor Q_i at a 5.5 GHz readout (alpha_KI = 0.08, the prelim resonator
values) — i.e. the pulse a KID would actually see.

Two chained `run_time_dependent` calls (drive on / drive off); the
driven Newton steady state is computed independently as the
"pulse-stays-on" reference. The decay is nonlinear (recombination gives
an instantaneous lifetime growing as x_qp drops), so the script records
the instantaneous tau = -x/xdot along the tail rather than one fitted
constant.

Usage::

    python scripts/demo_kid_pulse_response.py
"""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.density import qp_fraction
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.services.transient import run_time_dependent

# ── Fischer 2024 Al parameters (as in the photon-kick demo) ─────────
DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1
OMEGA_PB = 2.8 * DELTA_0
N_BAR_PB = 1e6
C_PHOT_PB = 1e-9              # c*nbar = 1e-3 ns^-1 during the pulse

E_MAX_FACTOR = 10.0
NUM_BINS = 810                # omega_PB/dE = 252 commensurate

# Readout (prelim resonator values).
F_READOUT_GHZ = 5.5
# h = 2*pi*hbar = 4.135667696 ueV*ns exactly, so h*f[GHz] is in ueV:
# 5.5 GHz -> 22.75 ueV. (A previous revision had 4.1357e-3 here — an
# effective 5.5 MHz readout; caught by the 2026-07-04 deep review.)
OMEGA_READOUT_UEV = 4.135667696 * F_READOUT_GHZ
ALPHA_KI = 0.08

# Pulse schedule.
T_PULSE_NS = 30.0
DT_ON = 0.1
SNAP_ON = 2.0
T_DECAY_NS = 1000.0
DT_OFF = 0.2
SNAP_OFF = 20.0

OUT_DIR = ROOT / "outputs" / "demo_kid_pulse_response"


def _material() -> Material:
    return Material(name="Al_Fischer2024", Delta_0=DELTA_0, T_c=T_C, tau_0=TAU_0)


def _fermi_dirac(E: np.ndarray) -> np.ndarray:
    return fermi_dirac_occupation(E, T_BATH)


def _build_state(f_init: np.ndarray | None = None) -> T3DiffusionState:
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=1.0,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    spectral = SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=DELTA_0,
    )
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_BATH).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=_fermi_dirac(E) if f_init is None else f_init,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=_material(),
        T_bath=T_BATH,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    backend = T3DiffusionBackend()
    state = _build_state()
    f_ref = state.f.copy()  # thermal reference for df_r/f_r

    observables = {
        "x_qp": lambda s: qp_fraction(s.f, s.spectral, delta_0=DELTA_0),
        "df_over_f": lambda s: compute_frequency_shift(
            s.f, f_ref, s.spectral, OMEGA_READOUT_UEV, ALPHA_KI,
        ),
        "Q_i": lambda s: compute_quality_factor(
            s.f, s.spectral, OMEGA_READOUT_UEV, ALPHA_KI,
        ),
    }
    pb = {"omega_PB": OMEGA_PB, "n_bar_PB": N_BAR_PB, "c_phot_PB": C_PHOT_PB}

    print(f"Stage 1: pulse ON for {T_PULSE_NS:g} ns ...", flush=True)
    on = run_time_dependent(
        state,
        dt=DT_ON,
        total_time=T_PULSE_NS,
        snapshot_interval=SNAP_ON,
        pb_photon_params=pb,
        observables=observables,
        backend=backend,
    )
    f_end_pulse = on.snapshots[-1].f

    print(f"Stage 2: pulse OFF, decay for {T_DECAY_NS:g} ns ...", flush=True)
    off = run_time_dependent(
        _build_state(f_init=f_end_pulse),
        dt=DT_OFF,
        total_time=T_DECAY_NS,
        snapshot_interval=SNAP_OFF,
        observables=observables,
        backend=backend,
    )

    print("Reference: driven Newton steady state (pulse stays on) ...", flush=True)
    driven_ss = backend.steady_state(
        _build_state(),
        use_thermal_phonons=True,
        pb_photon_params=pb,
        newton_tol=1e-12,
        newton_max_iter=500,
    )
    x_qp_ss = float(qp_fraction(driven_ss.f, driven_ss.spectral, delta_0=DELTA_0))

    rows: list[dict[str, float]] = []
    for snap in on.snapshots:
        rows.append({"t_ns": float(snap.t), "drive_on": 1.0,
                     **{k: float(v) for k, v in snap.observables.items()}})
    for snap in off.snapshots:
        if abs(snap.t) < 0.5 * DT_OFF:
            continue  # duplicate of the pulse-end point
        rows.append({"t_ns": T_PULSE_NS + float(snap.t), "drive_on": 0.0,
                     **{k: float(v) for k, v in snap.observables.items()}})

    # Instantaneous decay time tau = -x / xdot along the tail.
    decay = [r for r in rows if r["drive_on"] == 0.0]
    for a, b in pairwise(decay):
        dx_dt = (b["x_qp"] - a["x_qp"]) / (b["t_ns"] - a["t_ns"])
        mid_x = 0.5 * (a["x_qp"] + b["x_qp"])
        b["tau_inst_ns"] = float(-mid_x / dx_dt) if dx_dt < 0 else float("nan")

    csv_path = OUT_DIR / "pulse_response.csv"
    fieldnames = sorted({k for r in rows for k in r})
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    peak = max(rows, key=lambda r: r["x_qp"])
    tail = decay[-1]
    with (OUT_DIR / "metadata.json").open("w") as fp:
        json.dump(
            {
                "description": __doc__,
                "pulse_ns": T_PULSE_NS,
                "decay_window_ns": T_DECAY_NS,
                "c_nbar_per_ns": C_PHOT_PB * N_BAR_PB,
                "omega_readout_uev": OMEGA_READOUT_UEV,
                "alpha_KI": ALPHA_KI,
                "x_qp_peak": peak["x_qp"],
                "x_qp_end": tail["x_qp"],
                "tau_inst_end_ns": tail.get("tau_inst_ns"),
                "x_qp_driven_steady_state": x_qp_ss,
            },
            fp,
            indent=2,
        )

    print(f"  peak x_qp = {peak['x_qp']:.3e} at t = {peak['t_ns']:g} ns "
          f"(driven ss would be {x_qp_ss:.3e})")
    print(f"  end-of-window x_qp = {tail['x_qp']:.3e}, "
          f"instantaneous tau = {tail.get('tau_inst_ns', float('nan')):.0f} ns")
    print(f"  df/f swing: {min(r['df_over_f'] for r in rows):.3e} .. 0")
    print(f"  Q_i swing: {min(r['Q_i'] for r in rows):.3e} .. "
          f"{max(r['Q_i'] for r in rows):.3e}")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
