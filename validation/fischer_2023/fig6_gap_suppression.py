"""Fischer 2023 Fig. 6 characterization — self-consistent gap suppression.

This is the missing Gate 4.5 / "Ph0-Kaplan + sc-gap" capability in the
new framework: sweep sub-gap photon drive at several bath temperatures,
solve the coupled ``(f, n_ph, Δ)`` steady state, and record the gap
suppression observables

    δΔ = Δ_eq(T_B) - Δ_final
    δΔ / Δ_eq

against the analytical drive-equivalent effective temperature from
Fischer 2023 Eq. 35,

    k_B T_* = (A \bar n)^{1/6}.

The "Ph0-Kaplan" label here follows the project-wide notation note:
the QP kernels already use the Kaplan-convention electron-phonon
timescales, while the phonon bath relaxation uses the acoustic-escape
``τ_l`` builder sourced from material geometry. In the current v1
implementation that builder is frequency-independent, but it is still
derived from the acoustic-escape formula rather than supplied as a
manual scalar.

This baseline is self-pinned under ``validation/baselines/ph0_kaplan/``.
It is a characterization test, not a parity test against the old repo.

Usage::

    python -m validation.fischer_2023.fig6_gap_suppression
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material, load_material
from qpsim.observables.gap_suppression import gap_suppression_from_deltas
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics import acoustic_escape_tau_l, calibrate_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 Table I + acoustic-escape phonon model ──────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0
C_PHOT = 1e-9

FILM_THICKNESS_NM = 63.0
SUBSTRATE_ETA = 0.2

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 405  # dE = 4 μeV so ω0/dE = 5 is integer-commensurate

T_BATH_VALUES: tuple[float, ...] = (0.10, 0.15, 0.20)
# Drive ladder spans seven decades of microwave occupation plus the
# zero-drive reference, so the suppression onset (somewhere around
# n̄ ≈ 1e8 at T_B = 0.1 K) is resolved inside the sweep.
N_BAR_VALUES: tuple[float, ...] = (0.0, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10)


@dataclass(frozen=True)
class Fig6Result:
    T_bath: np.ndarray
    n_bar: np.ndarray
    tau_l_ns: np.ndarray
    T_star_drive_K: np.ndarray
    kBT_star_over_delta0: np.ndarray
    delta_eq: np.ndarray
    delta_final: np.ndarray
    delta_suppression: np.ndarray
    rel_suppression: np.ndarray


def _material() -> Material:
    """Al film with Fischer 2023 superconducting parameters and 63 nm thickness."""
    return replace(
        load_material("Al"),
        name="Al_Fischer2023_Kaplan",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        film_thickness=FILM_THICKNESS_NM,
        substrate_transmission_eta=SUBSTRATE_ETA,
    )


def _kBTstar_analytic(
    n_bar: float,
    *,
    c_phot: float,
    tau_0: float,
    T_c: float,
    omega_0: float,
    delta: float,
) -> float:
    """Analytical ``k_B T_*`` from Fischer 2023 Eq. 35."""
    if n_bar <= 0:
        return 0.0
    A = (
        (105.0 / 64.0)
        * (KB_UEV_PER_K * T_c) ** 3
        * c_phot
        * tau_0
        * omega_0 ** 2
        * delta
    )
    return float((A * n_bar) ** (1.0 / 6.0))


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
    omega_2d = omega.reshape(1, -1)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega_2d,
        tau_l=acoustic_escape_tau_l(omega_2d, material),
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


def run() -> Fig6Result:
    material = _material()
    backend = T3DiffusionBackend()

    T_rows: list[float] = []
    nbar_rows: list[float] = []
    tau_rows: list[float] = []
    Tstar_rows: list[float] = []
    ratio_rows: list[float] = []
    delta_eq_rows: list[float] = []
    delta_final_rows: list[float] = []
    delta_supp_rows: list[float] = []
    rel_supp_rows: list[float] = []

    for T_bath in T_BATH_VALUES:
        state = _build_state(material, T_bath)
        calibration = calibrate_gap(T_c=material.T_c, T_bath=T_bath)

        for n_bar in N_BAR_VALUES:
            photon_params = None
            if n_bar > 0:
                photon_params = {
                    "omega_0": OMEGA_0,
                    "n_bar": n_bar,
                    "c_phot": C_PHOT,
                }

            state = backend.steady_state(
                state,
                method="picard",
                photon_params=photon_params,
                self_consistent_gap=True,
                gap_tol=1e-6,
                gap_max_iter=20,
                gap_under_relaxation=0.5,
                picard_tol=1e-8,
                picard_max_iter=500,
                picard_mixing=0.3,
                newton_tol=1e-14,
                newton_max_iter=500,
            )

            gap_obs = gap_suppression_from_deltas(calibration.delta_eq, state.gap)
            kBT_star = _kBTstar_analytic(
                n_bar,
                c_phot=C_PHOT,
                tau_0=TAU_0,
                T_c=T_C,
                omega_0=OMEGA_0,
                delta=DELTA_0,
            )

            T_rows.append(T_bath)
            nbar_rows.append(n_bar)
            tau_rows.append(float(state.phonon.tau_l[0, 0]))
            Tstar_rows.append(kBT_star / KB_UEV_PER_K)
            ratio_rows.append(kBT_star / DELTA_0)
            delta_eq_rows.append(gap_obs.delta_eq)
            delta_final_rows.append(gap_obs.delta_final)
            delta_supp_rows.append(gap_obs.delta_suppression)
            rel_supp_rows.append(gap_obs.rel_suppression)

    return Fig6Result(
        T_bath=np.array(T_rows, dtype=float),
        n_bar=np.array(nbar_rows, dtype=float),
        tau_l_ns=np.array(tau_rows, dtype=float),
        T_star_drive_K=np.array(Tstar_rows, dtype=float),
        kBT_star_over_delta0=np.array(ratio_rows, dtype=float),
        delta_eq=np.array(delta_eq_rows, dtype=float),
        delta_final=np.array(delta_final_rows, dtype=float),
        delta_suppression=np.array(delta_supp_rows, dtype=float),
        rel_suppression=np.array(rel_supp_rows, dtype=float),
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "validation"
        / "baselines"
        / "ph0_kaplan"
        / "fischer_fig6_gap_suppression.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig6Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "# Fischer 2023 Fig 6 characterization — Ph0-Kaplan + self-consistent gap; "
                "self-pinned by qpsim"
            ]
        )
        writer.writerow(
            [
                f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
                f"c_phot={C_PHOT} film_thickness_nm={FILM_THICKNESS_NM} eta={SUBSTRATE_ETA}"
            ]
        )
        writer.writerow(
            [f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"]
        )
        writer.writerow(
            [
                "T_bath_K",
                "n_bar",
                "tau_l_ns",
                "T_star_drive_K",
                "kBT_star_over_delta0",
                "delta_eq_ueV",
                "delta_final_ueV",
                "delta_suppression_ueV",
                "rel_suppression",
            ]
        )
        for i in range(result.T_bath.size):
            writer.writerow(
                [
                    f"{result.T_bath[i]:.17e}",
                    f"{result.n_bar[i]:.17e}",
                    f"{result.tau_l_ns[i]:.17e}",
                    f"{result.T_star_drive_K[i]:.17e}",
                    f"{result.kBT_star_over_delta0[i]:.17e}",
                    f"{result.delta_eq[i]:.17e}",
                    f"{result.delta_final[i]:.17e}",
                    f"{result.delta_suppression[i]:.17e}",
                    f"{result.rel_suppression[i]:.17e}",
                ]
            )
    return path


def read_baseline(path: Path | None = None) -> Fig6Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line or line[0].startswith("#") or line[0] == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig6Result(
        T_bath=data[:, 0],
        n_bar=data[:, 1],
        tau_l_ns=data[:, 2],
        T_star_drive_K=data[:, 3],
        kBT_star_over_delta0=data[:, 4],
        delta_eq=data[:, 5],
        delta_final=data[:, 6],
        delta_suppression=data[:, 7],
        rel_suppression=data[:, 8],
    )


def write_plot(result: Fig6Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.colormaps["viridis"]
    colors = cmap(np.linspace(0.15, 0.85, len(T_BATH_VALUES)))
    for T_bath, color in zip(T_BATH_VALUES, colors, strict=True):
        mask = (result.T_bath == T_bath) & (result.n_bar > 0)
        ax.semilogx(
            result.kBT_star_over_delta0[mask],
            result.rel_suppression[mask],
            "o-",
            lw=1.8,
            ms=5,
            color=color,
            label=rf"$T_B = {T_bath:.2f}$ K",
        )

    ax.axhline(0.0, color="k", lw=1.0, ls=":", alpha=0.5)
    ax.set_xlabel(r"$k_B T_* / \Delta_0$ (Eq. 35 drive map)", fontsize=12)
    ax.set_ylabel(r"$\delta \Delta / \Delta_\mathrm{eq}$", fontsize=12)
    ax.set_title(
        "Fischer 2023 Fig. 6 characterization\n"
        "Ph0-Kaplan (acoustic-escape $\\tau_\\ell$) + self-consistent gap",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig 6 characterization — gap suppression with Ph0-Kaplan ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω₀={OMEGA_0:.2f} μeV, "
        f"c_phot={C_PHOT:.0e} ns⁻¹"
    )
    print(
        f"  Acoustic escape: d={FILM_THICKNESS_NM:.0f} nm, "
        f"η={SUBSTRATE_ETA:.2f}"
    )
    print(f"  Grid: NE={NUM_BINS}")
    print(f"  T_bath values: {list(T_BATH_VALUES)}")
    print(f"  n_bar values:  {list(N_BAR_VALUES)}")
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
