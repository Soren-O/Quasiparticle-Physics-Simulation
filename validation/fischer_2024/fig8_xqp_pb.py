"""Fischer & Catelani 2024 Fig 8 — x_qp(T_B) under pair-breaking photon drive.

Exercises the pair-breaking photon collision channel
(:mod:`qpsim.collisions.pair_breaking_photon`) at the Fischer-2024
parameter set, sweeps ``T_bath`` for each of five power levels, and
produces the 5-curve x_qp comparison from F24 Fig 8.

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV:

    Δ₀             = 189 μeV
    τ₀             = 63 ns     (note: faster than F23 — different Al film)
    ω_PB           = 2.8 · Δ₀  =  529.2 μeV   (above 2Δ, pair-breaking active)
    n̄_PB           = 1e6
    c_phot_PB × n̄_PB ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} ns⁻¹   (5 power levels)

Grid: 810 bins so ω_PB/dE = 252 is integer commensurate (the old
851-bin choice snapped ω_PB by ~0.3 %, below the 1% tolerance but
sacrificing bit-reproducibility).

Usage::

    python -m validation.fischer_2024.fig8_xqp_pb
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

# ── F24 Sec. IV parameters ───────────────────────────────────────────

DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_PB = 2.8 * DELTA_0  # 529.2 μeV
N_BAR_PB = 1e6

POWER_LEVELS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
"""c_phot_PB · n̄_PB products in ns⁻¹ (F24 Sec. IV)."""

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810  # ω_PB/dE = 252 exactly at this grid

T_BATH_VALUES: tuple[float, ...] = tuple(np.linspace(0.05, 0.22, 8).tolist())


@dataclass(frozen=True)
class Fig8Result:
    T_bath: np.ndarray
    powers: tuple[float, ...]
    x_qp_thermal: np.ndarray                 # shape (NT,)
    x_qp_by_power: dict[float, np.ndarray]   # power → shape (NT,)


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
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


def run() -> Fig8Result:
    material = _material()
    backend = T3DiffusionBackend()
    T_values = np.array(T_BATH_VALUES)
    x_thermal = np.zeros_like(T_values)
    x_by_power: dict[float, np.ndarray] = {p: np.zeros_like(T_values) for p in POWER_LEVELS}

    # Verify commensurability once (all T_bath use the same grid).
    probe_state = _build_state(material, float(T_values[0]))
    dE_scalar = float(probe_state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"ω_PB={OMEGA_PB} is not integer-commensurate with dE={dE_scalar:.4f}"
        )

    for i, T in enumerate(T_values):
        state = _build_state(material, T)
        # Thermal reference (no drive).
        x_thermal[i] = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)

        for power in POWER_LEVELS:
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                "c_phot_PB": power / N_BAR_PB,
            }
            driven = backend.steady_state(
                state,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                newton_tol=1e-14,
                newton_max_iter=500,
            )
            x_by_power[power][i] = qp_fraction(
                driven.f, driven.spectral, delta_0=DELTA_0,
            )

    return Fig8Result(
        T_bath=T_values,
        powers=POWER_LEVELS,
        x_qp_thermal=x_thermal,
        x_qp_by_power=x_by_power,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "f24_fig8_xqp_pb.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig8Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer & Catelani 2024 Fig 8 — x_qp(T_B) with PB-photon drive; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_PB={OMEGA_PB} "
            f"n_bar_PB={N_BAR_PB}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        powers_csv = ",".join(f"{p:g}" for p in result.powers)
        writer.writerow([f"# powers_ns_inv={powers_csv}"])
        header = ["T_bath_K", "x_qp_thermal"] + [f"x_qp_power_{p:g}" for p in result.powers]
        writer.writerow(header)
        for i in range(result.T_bath.size):
            row = [f"{result.T_bath[i]:.17e}", f"{result.x_qp_thermal[i]:.17e}"]
            row.extend(f"{result.x_qp_by_power[p][i]:.17e}" for p in result.powers)
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig8Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    powers: tuple[float, ...] = ()
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# powers_ns_inv"):
                powers = tuple(float(x) for x in first.split("=", 1)[1].split(","))
                continue
            if first.startswith("#") or first == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    if not powers:
        raise RuntimeError(f"Baseline at {path} missing '# powers_ns_inv=' metadata.")
    data = np.array(rows, dtype=float)
    return Fig8Result(
        T_bath=data[:, 0],
        powers=powers,
        x_qp_thermal=data[:, 1],
        x_qp_by_power={p: data[:, i + 2] for i, p in enumerate(powers)},
    )


def write_plot(result: Fig8Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(result.T_bath, result.x_qp_thermal, "k--", lw=1.5,
              label=r"thermal (no PB drive)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result.powers)))
    for power, color in zip(result.powers, colors, strict=True):
        ax.loglog(result.T_bath, result.x_qp_by_power[power], lw=2.0, color=color,
                  label=rf"$c \cdot \bar n = {power:g}$ ns$^{{-1}}$")
    ax.set_xlabel(r"$T_B$ [K]", fontsize=14)
    ax.set_ylabel(r"$x_{qp}$", fontsize=14)
    ax.set_title(
        "Fischer & Catelani 2024 Fig 8 — PB-photon drive\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\tau_0={TAU_0:.0f}$ ns, "
        rf"$\omega_{{\mathrm{{PB}}}}=2.8\,\Delta_0$",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Fig 8 — x_qp(T_B) with PB-photon drive ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_PB={OMEGA_PB:.2f} μeV, "
        f"n̄_PB={N_BAR_PB:.0e}"
    )
    print(f"  Powers (c·n̄, ns⁻¹): {list(POWER_LEVELS)}")
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
