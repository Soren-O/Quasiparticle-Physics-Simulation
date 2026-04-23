"""Fischer 2023 Fig 5 reproduction — x_qp(T_B) with sub-gap photon drive.

Sweeps ``T_bath`` at a fixed photon occupation, solves the τ_l=0
(thermal-phonon) steady state at each ``T``, and computes the
dimensionless QP fraction ``x_qp = ∫ρ(E)f(E)dE / Δ₀`` (Fischer
2023 Eq. 4 convention) on both the Fermi-Dirac reference and the
driven steady state.

Same parameter set as :mod:`fig7_qi_vs_t` (Fischer Table I,
α = 0.5 only relevant for Q; not used here). Grid: 405 bins so
``ω₀/dE = 5`` is integer-commensurate with the sub-gap photon.

Usage::

    python -m validation.fischer_2023.fig5_xqp

Scope caveat: the driven curve is at a single fixed ``n̄``, not
along Fischer's Fig-5 curves parameterized by applied power
through the nbar self-consistency loop. Regression target only.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 parameters ──────────────────────────────────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0
C_PHOT = 1e-9
N_BAR_FIXED = 1e6

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 405              # ω₀/dE = 5, integer commensurate

T_BATH_VALUES: tuple[float, ...] = tuple(np.linspace(0.08, 0.28, 11).tolist())


@dataclass(frozen=True)
class Fig5Result:
    T_bath: np.ndarray
    x_qp_thermal: np.ndarray   # x_qp at f = f_FD(T), no drive
    x_qp_driven: np.ndarray    # x_qp at driven steady-state f (fixed n̄)


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state(material: Material, T_bath: float) -> T3DiffusionState:
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def run() -> Fig5Result:
    material = _fischer_material()
    backend = T3DiffusionBackend()
    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR_FIXED, "c_phot": C_PHOT}

    T_values = np.array(T_BATH_VALUES)
    x_thermal = np.zeros_like(T_values)
    x_driven = np.zeros_like(T_values)

    for i, T in enumerate(T_values):
        state = _build_state(material, T)
        x_thermal[i] = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)

        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            photon_params=photon_params,
            newton_tol=1e-14,
            newton_max_iter=500,
        )
        x_driven[i] = qp_fraction(driven.f, driven.spectral, delta_0=DELTA_0)

    return Fig5Result(T_bath=T_values, x_qp_thermal=x_thermal, x_qp_driven=x_driven)


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig5_xqp.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig5Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig 5 — x_qp(T_B) with fixed-n̄ drive; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"c_phot={C_PHOT} n_bar_fixed={N_BAR_FIXED}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow(["T_bath_K", "x_qp_thermal", "x_qp_driven"])
        for T, xt, xd in zip(
            result.T_bath, result.x_qp_thermal, result.x_qp_driven, strict=True,
        ):
            writer.writerow([f"{T:.17e}", f"{xt:.17e}", f"{xd:.17e}"])
    return path


def read_baseline(path: Path | None = None) -> Fig5Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line or line[0].startswith("#"):
                continue
            if line[0] == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig5Result(
        T_bath=data[:, 0],
        x_qp_thermal=data[:, 1],
        x_qp_driven=data[:, 2],
    )


def write_plot(result: Fig5Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(result.T_bath, result.x_qp_thermal, "ko-", lw=1.5,
                label=r"$x_{qp}$ thermal ($f = f_\mathrm{FD}$)")
    ax.semilogy(result.T_bath, result.x_qp_driven, "bs-", lw=1.5,
                label=rf"$x_{{qp}}$ driven ($\bar n = {N_BAR_FIXED:.0e}$)")
    ax.set_xlabel(r"$T_B$ [K]", fontsize=14)
    ax.set_ylabel(r"$x_{qp}$", fontsize=14)
    ax.set_title(
        rf"Fischer 2023 Fig 5 — $x_{{qp}}(T_B)$ fixed-$\bar n$ drive"
        f"\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, $c_\mathrm{{phot}}=1$ Hz",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig 5 — x_qp(T_B) with fixed-n̄ drive ...")
    print(
        f"  Δ₀={DELTA_0} μeV, ω₀={OMEGA_0:.2f} μeV, "
        f"n̄={N_BAR_FIXED:.0e}, c_phot={C_PHOT:.0e} ns⁻¹"
    )
    print(f"  Grid: NE={NUM_BINS}")
    print(
        f"  T_bath sweep: {len(T_BATH_VALUES)} points, "
        f"{T_BATH_VALUES[0]:.3f} → {T_BATH_VALUES[-1]:.3f} K"
    )
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
