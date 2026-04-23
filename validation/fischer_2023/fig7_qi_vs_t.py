"""Fischer 2023 Fig 7 reproduction — Q_i(T_B) with sub-gap photon drive.

Sweeps bath temperature at a fixed photon occupation, solves the
τ_l=0 (thermal-phonon) steady state with the sub-gap drive on at
each temperature, and computes the internal quality factor via the
Mattis-Bardeen observables. Produces two curves for comparison:

* ``Q_thermal`` — ``Q_i`` evaluated on the Fermi-Dirac distribution
  at each ``T_B`` (no drive).
* ``Q_driven`` — ``Q_i`` evaluated on the driven steady-state ``f``
  (same drive parameters, but ``n̄`` held fixed rather than following
  the nbar self-consistency loop from Table III; that loop arrives
  with the services layer post-Gate-4).

Fischer, Catelani — Phys. Rev. Applied 19, 054087 (2023):

    Δ₀      = 180 μeV       (Table I)
    τ_0     = 438 ns
    ω_0     = Δ₀/9 = 20 μeV (sub-gap probe / drive)
    α       = 0.5           (kinetic-inductance fraction)
    c_phot  = 1 Hz = 1e-9 ns⁻¹
    n̄_FIXED = 1e6           (representative Table III operating point)

Grid: 405 bins, dE = 4 μeV. Chosen so ``ω₀/dE = 5`` is an integer
(required for the sub-gap photon partners to land on grid points).
Tolerance tier per NFP §6.4.1 is 1e-4, so paper-grid resolution
(1620 bins) is overkill here.

Caveat: the Q_driven / Q_thermal ordering you see is for a *single
fixed* ``n̄`` and won't match the published figure's trend lines,
which are parameterized by applied microwave power (dBm) through the
nbar self-consistency loop. This is a regression-only baseline; a
Fischer-parity reproduction needs the nbar loop in
``qpsim.services`` (post-Gate-4).

Usage::

    python -m validation.fischer_2023.fig7_qi_vs_t
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
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 parameters ──────────────────────────────────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV — drive and probe
ALPHA_KI = 0.5
C_PHOT = 1e-9
N_BAR_FIXED = 1e6          # representative Fischer Table III value

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 405             # Q_i tolerance tier is 1e-4; 405 gives dE=4 μeV, ω₀/dE=5 (int)

# T_bath sweep — evenly spaced across the regime where Q_i is experimentally
# resolvable for an Al film (≪ T_c = 1.18 K).
T_BATH_VALUES: tuple[float, ...] = tuple(np.linspace(0.08, 0.28, 11).tolist())


@dataclass(frozen=True)
class Fig7Result:
    T_bath: np.ndarray
    Q_thermal: np.ndarray  # Q_i at f = f_FD(T), no drive
    Q_driven: np.ndarray   # Q_i at driven steady-state f (fixed n̄)


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


def run() -> Fig7Result:
    material = _fischer_material()
    backend = T3DiffusionBackend()
    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR_FIXED, "c_phot": C_PHOT}

    T_values = np.array(T_BATH_VALUES)
    Q_thermal = np.zeros_like(T_values)
    Q_driven = np.zeros_like(T_values)

    for i, T in enumerate(T_values):
        state = _build_state(material, T)

        # Thermal Q: from the Fermi-Dirac seeded into state.f (no drive).
        Q_thermal[i] = compute_quality_factor(
            state.f, state.spectral, OMEGA_0, ALPHA_KI,
        )

        # Driven Q: solve the τ_l=0 steady state with the sub-gap drive.
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            photon_params=photon_params,
            newton_tol=1e-14,
            newton_max_iter=500,
        )
        Q_driven[i] = compute_quality_factor(
            driven.f, driven.spectral, OMEGA_0, ALPHA_KI,
        )

    return Fig7Result(T_bath=T_values, Q_thermal=Q_thermal, Q_driven=Q_driven)


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig7_qi_vs_t.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig7Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig 7 — Q_i(T_B) with fixed-n̄ drive; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} n_bar_fixed={N_BAR_FIXED}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow(["T_bath_K", "Q_thermal", "Q_driven"])
        for T, qt, qd in zip(
            result.T_bath, result.Q_thermal, result.Q_driven, strict=True,
        ):
            writer.writerow([f"{T:.17e}", f"{qt:.17e}", f"{qd:.17e}"])
    return path


def read_baseline(path: Path | None = None) -> Fig7Result:
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
    return Fig7Result(
        T_bath=data[:, 0],
        Q_thermal=data[:, 1],
        Q_driven=data[:, 2],
    )


def write_plot(result: Fig7Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(result.T_bath, result.Q_thermal, "ko-", lw=1.5, label=r"$Q_i$ thermal ($f = f_{\mathrm{FD}}$)")
    ax.semilogy(result.T_bath, result.Q_driven, "bs-", lw=1.5,
                label=rf"$Q_i$ driven ($\bar n = {N_BAR_FIXED:.0e}$)")
    ax.set_xlabel(r"$T_B$ [K]", fontsize=14)
    ax.set_ylabel(r"$Q_i$", fontsize=14)
    ax.set_title(
        rf"Fischer 2023 Fig 7 — Q_i(T_B) fixed-$\bar n$ drive"
        f"\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\alpha={ALPHA_KI}$, $c_\mathrm{{phot}}=1$ Hz",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=11, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig 7 — Q_i(T_B) with fixed-n̄ drive ...")
    print(
        f"  Δ₀={DELTA_0} μeV, α={ALPHA_KI}, ω₀={OMEGA_0:.2f} μeV, "
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
