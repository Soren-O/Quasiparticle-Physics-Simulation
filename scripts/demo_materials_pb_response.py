"""Paper-2 capability demo: one drive, three materials from the YAML database.

Solves the driven steady state x_qp(T_B) for Al, Nb, and TiN under an
identical pair-breaking photon drive (omega_PB = 2.8 Delta per material,
fixed drive product c*nbar), using nothing but `load_material` and the
0-D T3 backend. The point of the figure: material parameters (tau_0,
Delta_0, T_c) come from the database, everything else is shared code —
the drive-set low-T plateau and the thermal takeover at higher T_B/T_c
fall out per material with no per-material code.

Grid is scale-invariant in Delta (810 bins to 10 Delta, so
omega_PB/dE = 252 is integer-commensurate for every material).

Usage::

    python scripts/demo_materials_pb_response.py
"""

# ruff: noqa: E402

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

MATERIALS = ("Al", "Nb", "TiN")
T_OVER_TC = (0.02, 0.05, 0.08, 0.11, 0.15, 0.20, 0.25, 0.30)

OMEGA_PB_FACTOR = 2.8      # omega_PB / Delta_0, pair-breaking for every material
N_BAR_PB = 1e5
C_PHOT_PB = 1e-9           # c*nbar = 1e-4 ns^-1, shared drive product

E_MAX_FACTOR = 10.0
NUM_BINS = 810             # omega_PB/dE = 252, integer for all materials

OUT_DIR = ROOT / "outputs" / "demo_materials_pb_response"


def _fermi_dirac(E: np.ndarray, T_K: float) -> np.ndarray:
    if T_K <= 0.0:
        return np.zeros_like(E)
    kT = KB_UEV_PER_K * T_K
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(material_name: str, T_bath_K: float,
                 f_init: np.ndarray | None = None) -> T3DiffusionState:
    material = load_material(material_name)
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.0,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    spectral = SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap,
    )
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath_K).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    f0 = _fermi_dirac(E, T_bath_K) if f_init is None else f_init
    return T3DiffusionState(
        f=f0,
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath_K,
    )


def run_material(material_name: str) -> list[dict[str, float | str | bool]]:
    material = load_material(material_name)
    backend = T3DiffusionBackend()
    pb_params = {
        "omega_PB": OMEGA_PB_FACTOR * material.Delta_0,
        "n_bar_PB": N_BAR_PB,
        "c_phot_PB": C_PHOT_PB,
    }
    rows: list[dict[str, float | str | bool]] = []
    f_warm: np.ndarray | None = None
    for t_frac in T_OVER_TC:
        T_bath = t_frac * material.T_c
        start = time.monotonic()
        state = _build_state(material_name, T_bath, f_init=f_warm)
        try:
            solved = backend.steady_state(
                state,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                newton_tol=1e-12,
                newton_max_iter=500,
            )
        except RuntimeError:
            # Cold restart from the thermal floor if the warm start
            # stranded Newton outside its basin. (RuntimeError only —
            # Newton non-convergence / singular Jacobian / line-search
            # failure; genuine API or data bugs must propagate.)
            state = _build_state(material_name, T_bath, f_init=None)
            solved = backend.steady_state(
                state,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                newton_tol=1e-12,
                newton_max_iter=500,
            )
        f_warm = solved.f
        wall = time.monotonic() - start
        x_driven = float(qp_fraction(solved.f, solved.spectral,
                                     delta_0=material.Delta_0))
        thermal = _build_state(material_name, T_bath, f_init=None)
        x_thermal = float(qp_fraction(thermal.f, thermal.spectral,
                                      delta_0=material.Delta_0))
        rows.append({
            "material": material_name,
            "Delta_0_uev": material.Delta_0,
            "T_c_K": material.T_c,
            "tau_0_ns": material.tau_0,
            "T_over_Tc": t_frac,
            "T_bath_K": T_bath,
            "omega_PB_uev": pb_params["omega_PB"],
            "c_nbar_per_ns": C_PHOT_PB * N_BAR_PB,
            "x_qp_driven": x_driven,
            "x_qp_thermal": x_thermal,
            "wall_seconds": wall,
        })
        print(
            f"  {material_name} T/Tc={t_frac:.2f} (T={T_bath * 1e3:.1f} mK): "
            f"x_qp={x_driven:.3e} (thermal {x_thermal:.3e}), wall={wall:.1f}s",
            flush=True,
        )
    return rows


def write_plot(rows: list[dict[str, float | str | bool]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Al": "tab:blue", "Nb": "tab:orange", "TiN": "tab:green"}
    for name in MATERIALS:
        sub = [r for r in rows if r["material"] == name]
        t = [r["T_over_Tc"] for r in sub]
        ax.semilogy(t, [r["x_qp_driven"] for r in sub], "o-",
                    color=colors[name], label=f"{name} driven")
        ax.semilogy(t, [r["x_qp_thermal"] for r in sub], ":",
                    color=colors[name], alpha=0.7, label=f"{name} thermal")
    ax.set_xlabel(r"$T_B / T_c$", fontsize=12)
    ax.set_ylabel(r"$x_\mathrm{qp}$", fontsize=12)
    ax.set_title(
        "One drive, three materials from the YAML database\n"
        rf"$\omega_\mathrm{{PB}} = {OMEGA_PB_FACTOR}\,\Delta_0$, "
        rf"$c\,\bar n = {C_PHOT_PB * N_BAR_PB:g}$ ns$^{{-1}}$",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, float | str | bool]] = []
    for name in MATERIALS:
        print(f"Material {name} ...", flush=True)
        all_rows.extend(run_material(name))
    csv_path = OUT_DIR / "xqp_vs_T.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with (OUT_DIR / "metadata.json").open("w") as fp:
        json.dump(
            {
                "description": __doc__,
                "materials": MATERIALS,
                "T_over_Tc": T_OVER_TC,
                "omega_PB_factor": OMEGA_PB_FACTOR,
                "c_nbar_per_ns": C_PHOT_PB * N_BAR_PB,
                "num_bins": NUM_BINS,
            },
            fp,
            indent=2,
        )
    write_plot(all_rows, OUT_DIR / "xqp_vs_T.pdf")
    print(f"Wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
