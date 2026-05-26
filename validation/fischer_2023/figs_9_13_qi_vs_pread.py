"""Fischer 2023 Figs 9-13 — Q_i(P_read) via the self-consistent n̄ loop.

Sweeps drive power ``P_read`` log-spaced from -100 dBm to -60 dBm at
fixed ``T_B = 0.1 K`` and records ``Q_i``, ``Q_tot``, and the
converged ``n̄`` per point. The sweep warm-starts each successive
drive power from the previous converged ``n̄`` to keep iteration
counts low — crucial for the high-power end where the map stiffens
as ``Q_i`` drops and ``Q_tot → Q_i``.

This is a logarithmic-in-P_read characterization sweep at fixed
``T_B``; it consumes the same :func:`qpsim.services.nbar_loop.solve_nbar_loop`
service used by the paper-track :mod:`fig7_paper` (which sweeps
``T_B`` at Tables II/III drive powers).

Parameters (Fischer 2023 Table I / default ``qi_vs_pread``):

    Δ₀       = 180 μeV
    τ_0      = 438 ns
    ω_0      = Δ₀/9 = 20 μeV
    α_KI     = 0.5
    c_phot   = 1e-9 ns⁻¹
    Q_c      = 1e5
    T_B      = 0.1 K
    P_read   ∈ [-100, -60] dBm, 21 log-spaced points

Grid: 405 bins (ω_0/dE = 5 integer). Tolerance tier per NFP §6.4.1
is 1e-4 (nbar-loop tol × MB sub-gap quadrature).

Usage::

    python -m validation.fischer_2023.figs_9_13_qi_vs_pread
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
from qpsim.services.nbar_loop import dbm_to_uev_per_ns, solve_nbar_loop

# ── Fischer 2023 Table I parameters ──────────────────────────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV — drive and probe
ALPHA_KI = 0.5
C_PHOT = 1e-9

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 405             # Q_i tolerance tier 1e-4; dE=4 μeV, ω₀/dE=5 (int)


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


# ── Sweep parameters ─────────────────────────────────────────────────

T_BATH = 0.1  # K
Q_C = 1.0e5
NUM_POINTS = 21           # 21 log-spaced points across -100…-60 dBm
P_READ_DBM_MIN = -100.0
P_READ_DBM_MAX = -60.0


@dataclass(frozen=True)
class Figs913Result:
    P_read_uev_per_ns: np.ndarray   # shape (N,) drive power in code units
    P_read_dbm: np.ndarray          # shape (N,) drive power in dBm
    Q_i: np.ndarray                 # shape (N,) converged Q_i
    Q_tot: np.ndarray               # shape (N,) loaded Q = (Q_i⁻¹ + Q_c⁻¹)⁻¹
    n_bar: np.ndarray               # shape (N,) converged photon occupancy
    iterations: np.ndarray          # shape (N,) nbar-loop iteration count


def run() -> Figs913Result:
    material = _fischer_material()
    backend = T3DiffusionBackend()
    state = _build_state(material, T_BATH)

    dbm_values = np.linspace(P_READ_DBM_MIN, P_READ_DBM_MAX, NUM_POINTS)
    P_read_values = np.array([dbm_to_uev_per_ns(float(d)) for d in dbm_values])

    Q_i = np.zeros(NUM_POINTS)
    Q_tot = np.zeros(NUM_POINTS)
    n_bar = np.zeros(NUM_POINTS)
    iterations = np.zeros(NUM_POINTS, dtype=np.int64)

    def solve_f(n_bar_val: float) -> np.ndarray:
        photon_params = {"omega_0": OMEGA_0, "n_bar": n_bar_val, "c_phot": C_PHOT}
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            photon_params=photon_params,
            newton_tol=1e-14,
            newton_max_iter=500,
        )
        return driven.f

    def compute_Q_i(f: np.ndarray) -> float:
        return compute_quality_factor(f, state.spectral, OMEGA_0, ALPHA_KI)

    n_bar_warm: float | None = None
    for idx, P_read in enumerate(P_read_values):
        loop = solve_nbar_loop(
            P_read_uev_per_ns=float(P_read),
            Q_c=Q_C,
            omega_0=OMEGA_0,
            solve_f=solve_f,
            compute_Q_i=compute_Q_i,
            n_bar_initial=n_bar_warm,
            tol=1e-4,
            max_iter=50,
            under_relaxation=1.0,
        )
        if not loop.converged:
            raise RuntimeError(
                f"nbar loop failed at P_read={dbm_values[idx]:.2f} dBm "
                f"(n̄={loop.n_bar:.3e}, iterations={loop.iterations})"
            )
        Q_i[idx] = loop.Q_i
        Q_tot[idx] = loop.Q_tot
        n_bar[idx] = loop.n_bar
        iterations[idx] = loop.iterations
        n_bar_warm = loop.n_bar

    return Figs913Result(
        P_read_uev_per_ns=P_read_values,
        P_read_dbm=dbm_values,
        Q_i=Q_i,
        Q_tot=Q_tot,
        n_bar=n_bar,
        iterations=iterations,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_figs_9_13_qi_vs_pread.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Figs913Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Figs 9-13 — Q_i(P_read) via n̄ loop; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} Q_c={Q_C:g} T_bath={T_BATH}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow([
            f"# P_read_dbm: {P_READ_DBM_MIN} .. {P_READ_DBM_MAX}, "
            f"{NUM_POINTS} points (linear in dBm)"
        ])
        writer.writerow([
            "P_read_uev_per_ns", "P_read_dbm",
            "Q_i", "Q_tot", "n_bar", "iterations",
        ])
        for i in range(result.P_read_uev_per_ns.size):
            writer.writerow([
                f"{result.P_read_uev_per_ns[i]:.17e}",
                f"{result.P_read_dbm[i]:.17e}",
                f"{result.Q_i[i]:.17e}",
                f"{result.Q_tot[i]:.17e}",
                f"{result.n_bar[i]:.17e}",
                str(int(result.iterations[i])),
            ])
    return path


def read_baseline(path: Path | None = None) -> Figs913Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    iter_col: list[int] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("#") or first == "P_read_uev_per_ns":
                continue
            rows.append([float(x) for x in line[:-1]])
            iter_col.append(int(line[-1]))
    data = np.array(rows, dtype=float)
    return Figs913Result(
        P_read_uev_per_ns=data[:, 0],
        P_read_dbm=data[:, 1],
        Q_i=data[:, 2],
        Q_tot=data[:, 3],
        n_bar=data[:, 4],
        iterations=np.array(iter_col, dtype=np.int64),
    )


def write_plot(result: Figs913Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_q, ax_n) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    # Use dBm on the x-axis directly (linear in dBm ≡ log in P_read).
    ax_q.semilogy(result.P_read_dbm, result.Q_i, "o-", lw=1.5, ms=5, label=r"$Q_i$")
    ax_q.semilogy(result.P_read_dbm, result.Q_tot, "s--", lw=1.0, ms=4,
                  color="tab:orange", label=r"$Q_\mathrm{tot}$")
    ax_q.axhline(Q_C, color="k", ls=":", lw=1.0, alpha=0.5, label=rf"$Q_c={Q_C:g}$")
    ax_q.set_ylabel(r"Quality factor", fontsize=12)
    ax_q.grid(True, which="both", ls=":", alpha=0.3)
    ax_q.legend(loc="best", fontsize=10)
    ax_q.set_title(
        "Fischer 2023 Figs 9-13 — $Q_i(P_\\mathrm{read})$ via n̄ loop\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\alpha={ALPHA_KI}$, $Q_c={Q_C:g}$, $T_B={T_BATH}$ K",
        fontsize=10,
    )

    ax_n.semilogy(result.P_read_dbm, result.n_bar, "v-", lw=1.5, ms=5,
                  color="tab:green")
    ax_n.set_xlabel(r"$P_\mathrm{read}$ [dBm]", fontsize=12)
    ax_n.set_ylabel(r"$\bar n$ (converged)", fontsize=12)
    ax_n.grid(True, which="both", ls=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Figs 9-13 — Q_i(P_read) via n̄ loop ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_0={OMEGA_0:.2f} μeV, "
        f"α={ALPHA_KI}, Q_c={Q_C:g}, T_B={T_BATH} K"
    )
    print(f"  P_read: {NUM_POINTS} pts, {P_READ_DBM_MIN} → {P_READ_DBM_MAX} dBm (linear in dBm)")
    print(f"  Grid: NE={NUM_BINS}")
    result = run()
    print(f"  Q_i spans {result.Q_i.min():.2e} → {result.Q_i.max():.2e}")
    print(
        f"  n̄ spans  {result.n_bar.min():.2e} → {result.n_bar.max():.2e}, "
        f"iterations mean = {result.iterations.mean():.1f}"
    )
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
