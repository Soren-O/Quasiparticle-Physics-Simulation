r"""Marchegiani 2025 Fig 4 — parity rates Γ_P and Γ^eo_01/Γ^eo_10 vs T.

Both panels use the M25 Fig 3 caption parameter set (Δ_R/h = 49 GHz,
ω_10/(2π) = 5.5 GHz, Γ̃^ph_00 = 300 Hz) for the small-asymmetry
(ω_LR/(2π) = 0.5 GHz) and large-asymmetry (ω_LR/(2π) = 5 GHz) cases.
Sweeps T ∈ [10, 150] mK and at each point reports:

* ``Gamma_P`` — total parity-switching rate
  ``Γ_P = p_0 (Γ̃^eo_01 + Γ̃^eo_00) + p_1 (Γ̃^eo_10 + Γ̃^eo_11)``
* ``ratio_eo_01_over_10`` — excitation/relaxation ratio of the
  parity-flipping channels, ``Γ̃^eo_01 / Γ̃^eo_10``

Reuses the branch-continuation sweep from
:mod:`fig3_chemical_potentials`
(:func:`~validation.marchegiani_2025.fig3_chemical_potentials.solve_panel_branch_sweep`)
so both figures track the same steady-state root. With the
Γ̄-normalized density equations the root is unique, so the historical
panel-(a) multi-stability scatter is gone.

Usage::

    python -m validation.marchegiani_2025.fig4_parity_rates
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import M25SteadyState

from validation.marchegiani_2025.fig3_chemical_potentials import (
    DELTA_R_OVER_H_GHZ,
    GAMMA_PH_00_HZ,
    NUM_T_POINTS,
    OMEGA_10_OVER_H_GHZ,
    T_MAX_K,
    T_MIN_K,
    solve_panel_branch_sweep,
)


@dataclass(frozen=True)
class Fig4PanelResult:
    omega_LR_GHz: float
    T_kelvin: np.ndarray
    Gamma_P_Hz: np.ndarray
    ratio_eo_01_over_10: np.ndarray


@dataclass(frozen=True)
class Fig4Result:
    panel_a: Fig4PanelResult   # ω_LR = 0.5 GHz
    panel_b: Fig4PanelResult   # ω_LR = 5.0 GHz


def _gamma_eo(coefs, sol: M25SteadyState) -> np.ndarray:
    """Effective parity-flipping rate matrix at the M25 fixed point."""
    return (
        coefs.gamma_ph
        + coefs.gammas_L * sol.x_L
        + coefs.gammas_Rgt * sol.x_Rgt
        + coefs.gammas_Rlt * sol.x_Rlt
    )


def _run_panel(omega_LR_GHz: float) -> Fig4PanelResult:
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    sweep = solve_panel_branch_sweep(omega_LR_GHz, T_sweep)
    # Consume the driver's own per-T coefficient bundles (see the
    # matching note in fig4_paper._full_curve).
    for i, (coefs, sol) in enumerate(
        zip(sweep.coefficients, sweep.states, strict=True)
    ):
        gamma_eo = _gamma_eo(coefs, sol)
        Gamma_P[i] = (
            sol.p_0 * (gamma_eo[0, 1] + gamma_eo[0, 0])
            + sol.p_1 * (gamma_eo[1, 0] + gamma_eo[1, 1])
        )
        ratio[i] = gamma_eo[0, 1] / gamma_eo[1, 0] if gamma_eo[1, 0] > 0 else np.nan

    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        T_kelvin=T_sweep,
        Gamma_P_Hz=Gamma_P,
        ratio_eo_01_over_10=ratio,
    )


def run() -> Fig4Result:
    return Fig4Result(panel_a=_run_panel(0.5), panel_b=_run_panel(5.0))


# ── baseline I/O ─────────────────────────────────────────────────────


def _baseline_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "marchegiani_2025"


def baseline_path_a() -> Path:
    return _baseline_dir() / "m25_fig4a_parity_rates.csv"


def baseline_path_b() -> Path:
    return _baseline_dir() / "m25_fig4b_parity_rates.csv"


def plot_path() -> Path:
    return _baseline_dir() / "m25_fig4_parity_rates.pdf"


def _write_panel_csv(panel: Fig4PanelResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        # See note in fig3_chemical_potentials._write_panel_csv.
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 4 — Γ_P and "
            "Γ^eo_01/Γ^eo_10 vs T; pinned by qpsim"
        ])
        writer.writerow([
            f"# omega_LR_GHz={panel.omega_LR_GHz:g}  "
            f"Delta_R_GHz={DELTA_R_OVER_H_GHZ:g}  "
            f"omega_10_GHz={OMEGA_10_OVER_H_GHZ:g}  "
            f"Gamma_ph_00_Hz={GAMMA_PH_00_HZ:g}"
        ])
        # The platform stamp selects the pin-test tolerance: strict
        # rtol=1e-6 on the generating platform, rtol=1e-3 elsewhere
        # (see validation/marchegiani_2025/_robust.py).
        writer.writerow([f"# pinned_on: {sys.platform}"])
        writer.writerow(["T_kelvin", "Gamma_P_Hz", "ratio_eo_01_over_10"])
        for i in range(panel.T_kelvin.size):
            writer.writerow([
                f"{panel.T_kelvin[i]:.17e}",
                f"{panel.Gamma_P_Hz[i]:.17e}",
                f"{panel.ratio_eo_01_over_10[i]:.17e}",
            ])
    return path


def _read_panel_csv(path: Path, omega_LR_GHz: float) -> Fig4PanelResult:
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line or line[0].startswith("#") or line[0] == "T_kelvin":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        T_kelvin=data[:, 0],
        Gamma_P_Hz=data[:, 1],
        ratio_eo_01_over_10=data[:, 2],
    )


def write_baseline(result: Fig4Result) -> tuple[Path, Path]:
    return (
        _write_panel_csv(result.panel_a, baseline_path_a()),
        _write_panel_csv(result.panel_b, baseline_path_b()),
    )


def read_baseline() -> Fig4Result:
    return Fig4Result(
        panel_a=_read_panel_csv(baseline_path_a(), 0.5),
        panel_b=_read_panel_csv(baseline_path_b(), 5.0),
    )


def write_plot(result: Fig4Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel a: Γ_P vs T (semilog y)
    ax = axes[0]
    for panel, color, label in (
        (result.panel_a, "tab:blue", r"$\omega_{LR}/(2\pi) = 0.5$ GHz"),
        (result.panel_b, "tab:orange", r"$\omega_{LR}/(2\pi) = 5$ GHz"),
    ):
        ax.semilogy(panel.T_kelvin * 1e3, panel.Gamma_P_Hz, "o-",
                    lw=1.5, ms=4, color=color, label=label)
    ax.set_xlabel(r"$T$ [mK]", fontsize=12)
    ax.set_ylabel(r"$\Gamma_P$ [Hz]", fontsize=12)
    ax.set_title("(a) Total parity-switching rate", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    # Panel b: Γ^eo_01/Γ^eo_10 vs T (semilog y)
    ax = axes[1]
    for panel, color, label in (
        (result.panel_a, "tab:blue", r"$\omega_{LR}/(2\pi) = 0.5$ GHz"),
        (result.panel_b, "tab:orange", r"$\omega_{LR}/(2\pi) = 5$ GHz"),
    ):
        ax.semilogy(panel.T_kelvin * 1e3, panel.ratio_eo_01_over_10, "s-",
                    lw=1.5, ms=4, color=color, label=label)
    ax.set_xlabel(r"$T$ [mK]", fontsize=12)
    ax.set_ylabel(r"$\widetilde\Gamma^\mathrm{eo}_{01} / "
                  r"\widetilde\Gamma^\mathrm{eo}_{10}$", fontsize=12)
    ax.set_title("(b) Excitation / relaxation ratio", fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=10, loc="best")

    fig.suptitle(
        "Marchegiani 2025 Fig 4 — parity-switching observables vs temperature\n"
        rf"$\Delta_R/h = {DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi) = {OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}} = {GAMMA_PH_00_HZ:g}$ Hz",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path, Path]:
    print("M25 Fig 4 — Γ_P and Γ^eo_01/Γ^eo_10 vs T")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz, T ∈ [{T_MIN_K*1e3:.0f}, {T_MAX_K*1e3:.0f}] mK")
    print("  Panel a (ω_LR = 0.5 GHz) ...")
    panel_a = _run_panel(0.5)
    print("  Panel b (ω_LR = 5.0 GHz) ...")
    panel_b = _run_panel(5.0)
    result = Fig4Result(panel_a=panel_a, panel_b=panel_b)
    csv_a, csv_b = write_baseline(result)
    pdf = write_plot(result)
    print(f"  Baselines: {csv_a.name}, {csv_b.name}")
    print(f"  PDF plot:  {pdf.name}")
    return csv_a, csv_b, pdf


if __name__ == "__main__":
    generate_baseline()
