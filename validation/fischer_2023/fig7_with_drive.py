"""Fischer 2023 Fig 7 (with drive) — Q_i(T_B) at several read-out powers.

Sweeps bath temperature across six Fischer Table III drive powers
(-100 / -90 / -80 / -72 / -68 / -64 dBm). At every ``(T_B, P_read)``
point the service
:func:`qpsim.services.nbar_loop.solve_nbar_loop` closes the
self-consistent ``n̄ ↔ Q_i`` relation [F23 Eqs. 59-60], then we
evaluate ``Q_i`` on the converged ``f`` via the Mattis-Bardeen
observables.

This is the "drive on" companion to
:mod:`validation.fischer_2023.fig7_qi_vs_t` (which holds ``n̄``
fixed at a single representative value and is not a Fischer-parity
baseline).

Parameters (Fischer Table I / III):

    Δ₀       = 180 μeV
    τ_0      = 438 ns
    ω_0      = Δ₀/9 = 20 μeV
    α_KI     = 0.5
    c_phot   = 1e-9 ns⁻¹  (= 1 Hz)
    Q_c      = 1e5        (Fischer default)
    P_read   ∈ {-100, -90, -80, -72, -68, -64} dBm

Grid: 405 bins, dE = 4 μeV  (ω_0/dE = 5 is integer commensurate).
Tolerance tier per NFP §6.4.1 is ``1e-4`` (combined nbar-loop
tol=1e-4 × MB sub-gap midpoint rule).

Usage::

    python -m validation.fischer_2023.fig7_with_drive
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.services.nbar_loop import dbm_to_uev_per_ns, solve_nbar_loop

from validation.fischer_2023.fig7_qi_vs_t import (
    ALPHA_KI,
    C_PHOT,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    NUM_BINS,
    OMEGA_0,
    T_C,
    TAU_0,
    _build_state,
    _fischer_material,
)

# Fischer Table III drive powers.
P_READ_DBM: tuple[float, ...] = (-100.0, -90.0, -80.0, -72.0, -68.0, -64.0)

# Fischer default coupling quality factor.
Q_C = 1.0e5

# Bath-temperature sweep (same endpoints as fig7_qi_vs_t).
T_BATH_VALUES: tuple[float, ...] = tuple(np.linspace(0.08, 0.28, 11).tolist())


@dataclass(frozen=True)
class Fig7DriveResult:
    T_bath: np.ndarray
    p_read_dbm: tuple[float, ...]
    Q_thermal: np.ndarray                     # (NT,) Q_i at f_FD(T)
    Q_driven_by_dbm: dict[float, np.ndarray]  # dBm → (NT,) Q_i at converged f
    n_bar_by_dbm: dict[float, np.ndarray]     # dBm → (NT,) converged n̄


def _Q_i_from_f(f: np.ndarray, state) -> float:  # type: ignore[no-untyped-def]
    return compute_quality_factor(f, state.spectral, OMEGA_0, ALPHA_KI)


def run() -> Fig7DriveResult:
    material = _fischer_material()
    backend = T3DiffusionBackend()

    T_values = np.array(T_BATH_VALUES)
    Q_thermal = np.zeros_like(T_values)
    Q_driven_by_dbm: dict[float, np.ndarray] = {
        p: np.zeros_like(T_values) for p in P_READ_DBM
    }
    n_bar_by_dbm: dict[float, np.ndarray] = {
        p: np.zeros_like(T_values) for p in P_READ_DBM
    }

    for i, T in enumerate(T_values):
        state = _build_state(material, float(T))
        Q_thermal[i] = _Q_i_from_f(state.f, state)

        for p_dbm in P_READ_DBM:
            P_read = dbm_to_uev_per_ns(p_dbm)

            def solve_f(n_bar: float, state=state) -> np.ndarray:
                photon_params = {
                    "omega_0": OMEGA_0, "n_bar": n_bar, "c_phot": C_PHOT,
                }
                driven = backend.steady_state(
                    state,
                    use_thermal_phonons=True,
                    photon_params=photon_params,
                    newton_tol=1e-14,
                    newton_max_iter=500,
                )
                return driven.f

            def compute_Q_i(f: np.ndarray, state=state) -> float:
                return compute_quality_factor(
                    f, state.spectral, OMEGA_0, ALPHA_KI,
                )

            loop = solve_nbar_loop(
                P_read_uev_per_ns=P_read,
                Q_c=Q_C,
                omega_0=OMEGA_0,
                solve_f=solve_f,
                compute_Q_i=compute_Q_i,
                tol=1e-4,
                max_iter=50,
                under_relaxation=1.0,
            )
            if not loop.converged:
                raise RuntimeError(
                    f"nbar loop failed at T={T:.3f} K, P_read={p_dbm} dBm: "
                    f"after {loop.iterations} iterations, n̄={loop.n_bar:.3e}"
                )
            Q_driven_by_dbm[p_dbm][i] = loop.Q_i
            n_bar_by_dbm[p_dbm][i] = loop.n_bar

    return Fig7DriveResult(
        T_bath=T_values,
        p_read_dbm=P_READ_DBM,
        Q_thermal=Q_thermal,
        Q_driven_by_dbm=Q_driven_by_dbm,
        n_bar_by_dbm=n_bar_by_dbm,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig7_with_drive.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig7DriveResult, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig 7 — Q_i(T_B) with n̄ loop at six P_read; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} Q_c={Q_C:g}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        dbm_csv = ",".join(f"{p:g}" for p in result.p_read_dbm)
        writer.writerow([f"# p_read_dbm={dbm_csv}"])
        header = ["T_bath_K", "Q_thermal"]
        for p in result.p_read_dbm:
            header.append(f"Q_driven_dbm_{p:g}")
        for p in result.p_read_dbm:
            header.append(f"n_bar_dbm_{p:g}")
        writer.writerow(header)
        for i in range(result.T_bath.size):
            row = [f"{result.T_bath[i]:.17e}", f"{result.Q_thermal[i]:.17e}"]
            row.extend(
                f"{result.Q_driven_by_dbm[p][i]:.17e}" for p in result.p_read_dbm
            )
            row.extend(
                f"{result.n_bar_by_dbm[p][i]:.17e}" for p in result.p_read_dbm
            )
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig7DriveResult:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    dbm: tuple[float, ...] = ()
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# p_read_dbm"):
                dbm = tuple(float(x) for x in first.split("=", 1)[1].split(","))
                continue
            if first.startswith("#") or first == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    if not dbm:
        raise RuntimeError(f"Baseline at {path} missing '# p_read_dbm=' metadata.")
    data = np.array(rows, dtype=float)
    n_dbm = len(dbm)
    # Layout: T_bath, Q_thermal, [Q_driven × n_dbm], [n_bar × n_dbm]
    q_driven_start = 2
    n_bar_start = 2 + n_dbm
    return Fig7DriveResult(
        T_bath=data[:, 0],
        p_read_dbm=dbm,
        Q_thermal=data[:, 1],
        Q_driven_by_dbm={
            p: data[:, q_driven_start + i] for i, p in enumerate(dbm)
        },
        n_bar_by_dbm={
            p: data[:, n_bar_start + i] for i, p in enumerate(dbm)
        },
    )


def write_plot(result: Fig7DriveResult, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(result.T_bath, result.Q_thermal, "k--", lw=1.5,
                label=r"thermal (no drive)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result.p_read_dbm)))
    for p, color in zip(result.p_read_dbm, colors, strict=True):
        ax.semilogy(
            result.T_bath, result.Q_driven_by_dbm[p],
            lw=2.0, color=color,
            label=rf"$P_\mathrm{{read}}={p:g}$ dBm",
        )
    ax.set_xlabel(r"$T_B$ [K]", fontsize=14)
    ax.set_ylabel(r"$Q_i$", fontsize=14)
    ax.set_title(
        "Fischer 2023 Fig 7 — $Q_i(T_B)$ with n̄ loop\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\alpha={ALPHA_KI}$, $Q_c={Q_C:g}$",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig 7 with drive — Q_i(T_B) via n̄ loop ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_0={OMEGA_0:.2f} μeV, "
        f"α_KI={ALPHA_KI}, Q_c={Q_C:g}"
    )
    print(f"  P_read (dBm): {list(P_READ_DBM)}")
    print(f"  Grid: NE={NUM_BINS}")
    print(f"  T_bath sweep: {len(T_BATH_VALUES)} pts, "
          f"{T_BATH_VALUES[0]:.3f} → {T_BATH_VALUES[-1]:.3f} K")
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
