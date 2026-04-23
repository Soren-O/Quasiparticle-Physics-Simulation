"""Fischer 2023 Fig 3 τ_l=0 reproduction — photon-driven f(E) with thermal phonons.

The ``τ_l = 0`` limit pins ``n_ph`` at the substrate Bose–Einstein
distribution (thermal-phonon shortcut), reducing the steady-state solve
to Newton-on-f alone — no Picard, no coupled Newton, no iteration-order
drift. That's why this is the one Fischer figure where a strict
bit-identical regression target is meaningful.

Fischer, Catelani — Phys. Rev. Applied 19, 054087 (2023), Table I:

    Δ₀      = 180 μeV
    τ₀      = 438 ns   (Kaplan 1976 for Al)
    T_bath  = 0.1 K
    ω₀      = Δ₀ / 9  =  20 μeV  (sub-gap photon, integer commensurate with dE)
    n̄       = 1 × 10⁷
    c_phot  = 1 Hz  =  1 × 10⁻⁹ ns⁻¹

Energy grid: ``NE = 1620``, ``dE = 1 μeV``, ``E ∈ [Δ₀, 10 Δ₀]``.

Usage — generate baseline + PDF::

    python -m validation.fischer_2023.fig3_tau_l_zero

The regression test in ``test_fig3_tau_l_zero.py`` reruns :func:`run`
and asserts 1e-12 match against the pinned baseline CSV.
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
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 Table I parameters ──────────────────────────────────

DELTA_0 = 180.0            # μeV
TAU_0 = 438.0              # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)  # BCS ratio → T_c
T_BATH = 0.1               # K
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV
N_BAR = 1e7
C_PHOT = 1e-9              # ns⁻¹  =  1 Hz

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1620            # dE = (10−1) · 180 / 1620 = 1 μeV; ω₀/dE = 20 int


@dataclass(frozen=True)
class Fig3Result:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    f_thermal: np.ndarray  # steady-state f with no photon drive
    f_photon: np.ndarray   # steady-state f with photon drive, τ_l = 0
    f_FD: np.ndarray       # reference Fermi–Dirac at T_bath


def _fischer_material() -> Material:
    """Al with Fischer 2023 Table I τ₀ and gap (other fields left at defaults)."""
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state() -> T3DiffusionState:
    """Construct the T3 state seeded with the Fermi–Dirac initial guess."""
    material = _fischer_material()

    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    frac_err = abs(OMEGA_0 - m * dE_scalar) / OMEGA_0
    if frac_err > 1e-10:
        raise RuntimeError(
            f"Fischer Fig 3 grid not commensurate: "
            f"ω₀ = {OMEGA_0}, m·dE = {m * dE_scalar}, frac_err = {frac_err}."
        )

    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)

    # Phonon state: thermal-phonon shortcut. n_ph is pinned at Bose–Einstein
    # (unused by the solver in this path, but kept on the physics grid for
    # shape consistency); τ_l = 0 is the Fischer convention.
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_BATH).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_BATH,
    )


def run() -> Fig3Result:
    """Run the full reproduction: thermal + photon-driven steady states."""
    state = _build_state()
    backend = T3DiffusionBackend()

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(state.spectral.E / kT, 500.0)) + 1.0)

    state_thermal = backend.steady_state(
        state,
        use_thermal_phonons=True,
        newton_tol=1e-14,
        newton_max_iter=200,
    )

    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR, "c_phot": C_PHOT}
    state_photon = backend.steady_state(
        state,
        use_thermal_phonons=True,
        photon_params=photon_params,
        newton_tol=1e-14,
        newton_max_iter=500,
    )

    return Fig3Result(
        E=state.spectral.E,
        f_thermal=state_thermal.f,
        f_photon=state_photon.f,
        f_FD=f_FD,
    )


def baseline_path() -> Path:
    """Path to the pinned regression CSV."""
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig3_tau_l_zero.csv"


def plot_path() -> Path:
    """Path to the companion PDF for visual verification against Fischer's figure."""
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig3Result, path: Path | None = None) -> Path:
    """Write the reproduction arrays to a CSV suitable for regression comparison."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig 3 τ_l=0 — pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} "
            f"omega_0={OMEGA_0} n_bar={N_BAR} c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow(["E_uev", "f_thermal", "f_photon", "f_FD"])
        for e, ft, fp_, fd in zip(
            result.E, result.f_thermal, result.f_photon, result.f_FD, strict=True,
        ):
            writer.writerow([
                f"{e:.17e}", f"{ft:.17e}", f"{fp_:.17e}", f"{fd:.17e}",
            ])
    return path


def read_baseline(path: Path | None = None) -> Fig3Result:
    """Read a pinned baseline CSV back into a :class:`Fig3Result`."""
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line or line[0].lstrip().startswith("#"):
                continue
            if line[0] == "E_uev":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig3Result(
        E=data[:, 0],
        f_thermal=data[:, 1],
        f_photon=data[:, 2],
        f_FD=data[:, 3],
    )


def write_plot(result: Fig3Result, path: Path | None = None) -> Path:
    """Produce the Fig-3-style log-scale plot for visual verification."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), height_ratios=[3, 1], sharex=True,
    )
    x = result.E / DELTA_0

    ax1.semilogy(x, result.f_FD, "k--", lw=1.5, label=rf"Fermi-Dirac, $T_B={T_BATH}$ K")
    ax1.semilogy(x, result.f_photon, "b-", lw=2.0, label=r"$\tau_\ell=0$ (photon-driven)")
    # Mark photon-step energies.
    for n_step in range(1, 10):
        E_step = DELTA_0 + n_step * OMEGA_0
        if E_step / DELTA_0 < E_MAX_FACTOR:
            ax1.axvline(E_step / DELTA_0, color="gray", ls=":", lw=0.5, alpha=0.4)
    ax1.set_ylabel(r"$f(E)$", fontsize=14)
    ax1.set_title(
        "Fischer 2023 Fig 3 — thermal-phonon limit "
        rf"($\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\bar n=10^7$, $c_\mathrm{{phot}}=1$ Hz, $T_B={T_BATH}$ K)",
        fontsize=10,
    )
    ax1.set_xlim(1.0, E_MAX_FACTOR)
    positive = result.f_photon[result.f_photon > 0]
    if positive.size > 0:
        ax1.set_ylim(max(float(positive.min()), 1e-80) / 10, 1e-8)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.grid(True, which="both", ls=":", alpha=0.3)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(result.f_FD > 0, result.f_photon / result.f_FD, 1.0)
    ax2.semilogy(x, ratio, "b-", lw=2.0)
    ax2.axhline(1.0, color="k", ls="--", lw=0.8)
    for n_step in range(1, 10):
        E_step = DELTA_0 + n_step * OMEGA_0
        if E_step / DELTA_0 < E_MAX_FACTOR:
            ax2.axvline(E_step / DELTA_0, color="gray", ls=":", lw=0.5, alpha=0.4)
    ax2.set_xlabel(r"$E / \Delta_0$", fontsize=14)
    ax2.set_ylabel(r"$f_\mathrm{photon} / f_\mathrm{FD}$", fontsize=12)
    ax2.grid(True, which="both", ls=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    """Run the reproduction and write both the CSV baseline and the PDF plot."""
    print("Fischer 2023 Fig 3 τ_l=0 reproduction ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, T_bath={T_BATH} K, "
        f"ω₀={OMEGA_0:.2f} μeV, n̄={N_BAR:.0e}, c_phot={C_PHOT:.0e} ns⁻¹"
    )
    print(f"  Grid: {NUM_BINS} bins, dE = {(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
