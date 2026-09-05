"""Run a first-pass spatial simulation for the prelim validation strip.

Model:
* 100 um reflective Al strip, uniform 1D mesh.
* Local electron-phonon collisions with thermal phonons at 100 mK.
* Energy-dependent legacy diffusion ``D(E) = D0 sqrt(1 - (Delta/E)^2)``.
* Continuous one-end external source centered near ``2 Delta``.

The source is intentionally normalized as a dimensionless local
``x_qp`` generation rate because the measured SIS I(V) is not wired into
the spatial backend yet.  That makes this a useful scale-and-shape run,
not a calibrated prediction from junction resistance.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.geometries import strip
from qpsim.backends.spatial import SpatialBackend, SpatialState
from qpsim.constants import HBAR_UEV_NS
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from scripts.run_prelim_spatial_overnight import _cell_centered_strip_grid, strip_coordinates


OUT_DIR = ROOT / "outputs" / "prelim_spatial_100um"

LENGTH_UM = 100.0
NX = 31
NE = 32
T_BATH_K = 0.1
DT_NS = 5.0  # collision-timescale cap; reduced per D0 below to bound the transport step
# Max diffusion number D0*dt/dx^2 per run. The spatial Crank-Nicolson step
# clips [0,1] over/undershoot and the backend raises once a step alters >0.1%
# of the conserved density (~diffusion number 11); 4 keeps a clip-free margin.
CFL_TARGET = 4.0
MAX_TIME_NS = 20_000.0
SNAPSHOT_INTERVAL_NS = 500.0
STOP_TOL = 2e-10

D0_SWEEP_UM2_PER_NS = (6.0, 20.0, 60.0)
LOCAL_XQP_GENERATION_RATE_PER_NS = 2e-5
SOURCE_CENTER_FACTOR = 2.0
SOURCE_SIGMA_FACTOR = 0.08

RESONATOR_FREQUENCY_GHZ = 5.5
KINETIC_INDUCTANCE_FRACTION = 0.08


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    return fermi_dirac_occupation(E, T)


def _build_state(D0: float) -> SpatialState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=5.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    x, dx_um = _cell_centered_strip_grid(NX, length_um=LENGTH_UM)
    f0 = np.repeat(_fermi_dirac(E, T_BATH_K)[:, None], NX, axis=1)
    return SpatialState(
        f=f0,
        geometry=strip(
            int(np.asarray(x).size),
            mesh_size=float(dx_um),
        ),
        spectral=spectral,
        material=material,
        T_bath=T_BATH_K,
    )


def _source_flux(state: SpatialState) -> StripFlux:
    center = SOURCE_CENTER_FACTOR * state.gap
    sigma = SOURCE_SIGMA_FACTOR * state.gap
    profile = np.exp(-0.5 * ((state.spectral.E - center) / sigma) ** 2)
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    xqp_norm = float(np.sum(spectral_weights * profile) / state.gap)
    if xqp_norm <= 0.0:
        raise RuntimeError("Could not normalize source spectrum.")
    gain_spectrum = LOCAL_XQP_GENERATION_RATE_PER_NS * profile / xqp_norm

    gain = np.zeros_like(state.f)
    gain[:, 0] = gain_spectrum
    return StripFlux(
        gain=gain,
        loss_rate=np.zeros_like(gain),
        diagnostics={
            "local_xqp_generation_rate_per_ns": LOCAL_XQP_GENERATION_RATE_PER_NS,
            "source_center_uev": center,
            "source_sigma_uev": sigma,
        },
    )


def _xqp_profile(state: SpatialState) -> np.ndarray:
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E, state.spectral.dE, state.gap,
    )
    return np.sum(spectral_weights[:, None] * state.f, axis=0) / state.gap


def _mean_f(state: SpatialState) -> np.ndarray:
    return np.mean(state.f, axis=1)


def _observables(f_ref: np.ndarray) -> dict[str, object]:
    probe_energy = HBAR_UEV_NS * 2.0 * np.pi * RESONATOR_FREQUENCY_GHZ

    def xqp_mean(state: SpatialState) -> float:
        return float(np.mean(_xqp_profile(state)))

    def xqp_source(state: SpatialState) -> float:
        return float(_xqp_profile(state)[0])

    def xqp_open_end(state: SpatialState) -> float:
        return float(_xqp_profile(state)[-1])

    def qi_uniform_weight(state: SpatialState) -> float:
        return compute_quality_factor(
            _mean_f(state),
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=200,
        )

    def frac_freq_shift_uniform_weight(state: SpatialState) -> float:
        return compute_frequency_shift(
            _mean_f(state),
            f_ref,
            state.spectral,
            omega_0=probe_energy,
            alpha=KINETIC_INDUCTANCE_FRACTION,
            n_subgap=200,
        )

    return {
        "xqp_mean": xqp_mean,
        "xqp_source": xqp_source,
        "xqp_open_end": xqp_open_end,
        "Qi_uniform_weight": qi_uniform_weight,
        "frac_freq_shift_uniform_weight": frac_freq_shift_uniform_weight,
    }


def _write_trace(path: Path, snapshots: list[object]) -> None:
    fieldnames = [
        "t_ns",
        "max_dfdt_per_ns",
        "xqp_mean",
        "xqp_source",
        "xqp_open_end",
        "Qi_uniform_weight",
        "frac_freq_shift_uniform_weight",
    ]
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for snap in snapshots:
            row = {
                "t_ns": snap.t,
                "max_dfdt_per_ns": snap.max_rate,
                **snap.observables,
            }
            writer.writerow(row)


def _write_profile(path: Path, state: SpatialState) -> None:
    profile = _xqp_profile(state)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["x_um", "xqp"])
        writer.writeheader()
        for x_um, xqp in zip(strip_coordinates(state), profile, strict=True):
            writer.writerow({"x_um": float(x_um), "xqp": float(xqp)})


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for D0 in D0_SWEEP_UM2_PER_NS:
        print(f"Running D0={D0:g} um^2/ns ...", flush=True)
        state = _build_state(D0)
        f_ref = _mean_f(state)
        flux = _source_flux(state)
        backend = SpatialBackend()
        dx_um = state.geometry.mesh_size
        dt = min(DT_NS, CFL_TARGET * dx_um * dx_um / D0)
        print(f"  dt={dt:g} ns (diffusion number {D0 * dt / dx_um**2:.1f})", flush=True)
        result = backend.run_until_steady_state(
            state,
            dt=dt,
            max_time=MAX_TIME_NS,
            external_flux=flux,
            stop_tol=STOP_TOL,
            snapshot_interval=SNAPSHOT_INTERVAL_NS,
            observables=_observables(f_ref),
        )

        label = f"D0_{D0:g}".replace(".", "p")
        trace_path = OUT_DIR / f"time_trace_{label}.csv"
        profile_path = OUT_DIR / f"final_profile_{label}.csv"
        _write_trace(trace_path, result.snapshots)
        _write_profile(profile_path, result.state)

        final_obs = result.snapshots[-1].observables
        summary_rows.append(
            {
                "D0_um2_per_ns": D0,
                "converged": result.converged,
                "total_time_ns": result.total_time,
                "n_steps": result.n_steps,
                "final_max_dfdt_per_ns": result.snapshots[-1].max_rate,
                **final_obs,
                "trace_csv": trace_path.name,
                "profile_csv": profile_path.name,
            }
        )
        print(
            f"D0={D0:g} um^2/ns: converged={result.converged}, "
            f"t={result.total_time:g} ns, xqp_mean={final_obs['xqp_mean']:.3e}, "
            f"xqp_source={final_obs['xqp_source']:.3e}, "
            f"xqp_open={final_obs['xqp_open_end']:.3e}, "
            f"dfdt={result.snapshots[-1].max_rate:.3e}/ns"
        )

    with (OUT_DIR / "summary.csv").open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
