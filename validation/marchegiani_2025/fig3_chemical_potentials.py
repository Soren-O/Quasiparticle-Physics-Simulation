r"""Marchegiani 2025 Fig 3 — chemical potentials μ_α(T) full reproduction.

Sweeps the bath temperature for the two M25 Fig 3 panels:

* Panel a — small gap asymmetry, ``ω_LR/(2π) = 0.5 GHz``
  (Δ_L/h = 49.5 GHz, Δ_R/h = 49.0 GHz)
* Panel b — large gap asymmetry, ``ω_LR/(2π) = 5 GHz``
  (Δ_L/h = 54.0 GHz, Δ_R/h = 49.0 GHz)

At each temperature recalibrate the photon drive to maintain
``Γ̃^ph_00 = 300 Hz`` (the M25 Fig 3 caption value), solve the
4-unknown moment system via the deterministic multi-seed helper
:func:`qpsim.services.rate_equation.solve_rate_equation_steady_state_multi_seed`
(``scipy.optimize.root(method='hybr')`` under the hood), and
extract the chemical potentials

    μ_α = Δ_α + T · log(x_α)

(M25 main text, "Chemical potentials vs temperature" subsection).

Branch selection: the M25 4-unknown system is multi-stable. The
sweep delegates to
:func:`qpsim.services.rate_equation.solve_rate_equation_steady_state_multi_seed`,
which tries the default ``√(g_eff/r)`` seed plus a hand-tuned x
grid (and the previous T's solution as a continuation seed) and
returns the converged positive-density candidate with the largest
``x_L`` (the photon-driven nonequilibrium branch). The previous-T
solution is preferred over max-x_L when both lie within 5× of
each other, providing branch continuity across small T changes;
larger jumps are real bifurcations of the underlying M25 system
and show up as visible kinks in the plot.

Usage::

    python -m validation.marchegiani_2025.fig3_chemical_potentials
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import (
    M25SteadyState,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    M25Coefficients,
    M25PhotonDrive,
    M25PhysicalParameters,
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)

_H_OVER_KB = 4.799243e-11   # K / Hz

# ── M25 Fig 3 caption parameters ─────────────────────────────────────
DELTA_R_OVER_H_GHZ = 49.0
OMEGA_10_OVER_H_GHZ = 5.5
E_J_OVER_H_GHZ = 14.5
E_C_OVER_H_GHZ = 0.290
R_RECOMB_HZ = 6.25e6        # r^L = r^{R<}
GAMMA_EE_10_HZ = 100e3      # Γ̃^ee_10
GAMMA_PH_00_HZ = 300.0      # Γ̃^ph_00 — fixed across the T sweep

# Photon drive (only Gamma_nu_scale_Hz is recalibrated at each T).
DRIVE_TEMPLATE = M25PhotonDrive(
    omega_nu_kelvin=119e9 * _H_OVER_KB,
    Gamma_nu_scale_Hz=1.0,
    nu_0_per_J_per_m3=0.73e47,
    volume_m3=506e-6 * 240e-6 * 0.028e-6,
)

# T sweep range — bounded above by the M25 T̄ ≈ 150 mK crossover
# (beyond that the system is at thermal equilibrium and μ_α ≈ 0 with
# numerical noise from competing near-equilibrium fixed points) and
# below by where the LM solver still finds the nonequilibrium branch
# from any seed (the very-low-T regime exp(-Δ/T) → 0 stresses scipy's
# polynomial Newton even with continuation).
T_MIN_K = 0.010
T_MAX_K = 0.150
NUM_T_POINTS = 29   # 5 mK spacing


@dataclass(frozen=True)
class Fig3PanelResult:
    """Per-panel temperature sweep output."""

    omega_LR_GHz: float
    T_kelvin: np.ndarray
    x_L: np.ndarray
    x_Rgt: np.ndarray
    x_Rlt: np.ndarray
    p_1: np.ndarray
    mu_L_GHz: np.ndarray
    mu_Rgt_GHz: np.ndarray
    mu_Rlt_GHz: np.ndarray


@dataclass(frozen=True)
class Fig3Result:
    panel_a: Fig3PanelResult   # ω_LR = 0.5 GHz
    panel_b: Fig3PanelResult   # ω_LR = 5.0 GHz


def _make_params(omega_LR_GHz: float, T_kelvin: float) -> M25PhysicalParameters:
    Delta_R_GHz = DELTA_R_OVER_H_GHZ
    Delta_L_GHz = Delta_R_GHz + omega_LR_GHz
    return M25PhysicalParameters(
        Delta_L_kelvin=Delta_L_GHz * 1e9 * _H_OVER_KB,
        Delta_R_kelvin=Delta_R_GHz * 1e9 * _H_OVER_KB,
        omega_10_kelvin=OMEGA_10_OVER_H_GHZ * 1e9 * _H_OVER_KB,
        T_kelvin=T_kelvin,
        E_J_kelvin=E_J_OVER_H_GHZ * 1e9 * _H_OVER_KB,
        E_C_kelvin=E_C_OVER_H_GHZ * 1e9 * _H_OVER_KB,
        R_T_Hz=8.0 * E_J_OVER_H_GHZ * 1e9
        * ((Delta_L_GHz + Delta_R_GHz) / 2.0 / Delta_L_GHz),
        r_L_Hz=R_RECOMB_HZ,
        r_Rlt_Hz=R_RECOMB_HZ,
        Gamma_ee_10_Hz=GAMMA_EE_10_HZ,
    )


def _coefficients_at(omega_LR_GHz: float, T_kelvin: float) -> M25Coefficients:
    params = _make_params(omega_LR_GHz, T_kelvin)
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, DRIVE_TEMPLATE, GAMMA_PH_00_HZ,
    )
    drive = replace(DRIVE_TEMPLATE, Gamma_nu_scale_Hz=scale)
    return coefficients_from_physical_parameters_with_photon_drive(params, drive)


def _try_solve(
    coefs: M25Coefficients,
    *,
    previous: np.ndarray | None = None,
) -> M25SteadyState | None:
    """Pick the M25 photon-driven branch via the shared multi-seed helper.

    When ``previous`` is supplied (the previous T point's solution
    array ``[p_1, x_L, x_{R>}, x_{R<}]``), it's used as the
    ``preferred_seed`` so the sweep tracks the same branch across
    small T changes — the helper falls back to max-x_L only when
    the previous branch has bifurcated away (drops by 5× or more).
    """
    try:
        return solve_rate_equation_steady_state_multi_seed(
            coefs, preferred_seed=previous,
        )
    except RuntimeError:
        return None


def _run_panel(omega_LR_GHz: float) -> Fig3PanelResult:
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    x_L = np.full(n, np.nan)
    x_Rgt = np.full(n, np.nan)
    x_Rlt = np.full(n, np.nan)
    p_1 = np.full(n, np.nan)

    last_y: np.ndarray | None = None
    for i, T_K in enumerate(T_sweep):
        coefs = _coefficients_at(omega_LR_GHz, float(T_K))
        sol = _try_solve(coefs, previous=last_y)
        if sol is None:
            raise RuntimeError(
                f"M25 Fig 3 panel ω_LR={omega_LR_GHz} GHz: no seed yielded "
                f"a positive-density solution at T = {T_K:.4f} K."
            )
        x_L[i] = sol.x_L
        x_Rgt[i] = sol.x_Rgt
        x_Rlt[i] = sol.x_Rlt
        p_1[i] = sol.p_1
        last_y = np.array([sol.p_1, sol.x_L, sol.x_Rgt, sol.x_Rlt])

    # Chemical potentials. μ_α = Δ_α + T · log(x_α). Convert to GHz/(2π) by
    # dividing by h_over_kB = 1/(h/k_B). At full equilibrium x_α →
    # x_α^thermal so μ_α → 0; at low T with photon drive μ_α > 0.
    Delta_L_K = (DELTA_R_OVER_H_GHZ + omega_LR_GHz) * 1e9 * _H_OVER_KB
    Delta_R_K = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    mu_L_GHz = (Delta_L_K + T_sweep * np.log(x_L)) / _H_OVER_KB / 1e9
    mu_Rgt_GHz = (Delta_R_K + T_sweep * np.log(x_Rgt)) / _H_OVER_KB / 1e9
    mu_Rlt_GHz = (Delta_R_K + T_sweep * np.log(x_Rlt)) / _H_OVER_KB / 1e9

    return Fig3PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        T_kelvin=T_sweep,
        x_L=x_L, x_Rgt=x_Rgt, x_Rlt=x_Rlt, p_1=p_1,
        mu_L_GHz=mu_L_GHz, mu_Rgt_GHz=mu_Rgt_GHz, mu_Rlt_GHz=mu_Rlt_GHz,
    )


def run() -> Fig3Result:
    return Fig3Result(
        panel_a=_run_panel(0.5),
        panel_b=_run_panel(5.0),
    )


# ── baseline I/O ─────────────────────────────────────────────────────


def _baseline_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "marchegiani_2025"


def baseline_path_a() -> Path:
    return _baseline_dir() / "m25_fig3a_chemical_potentials.csv"


def baseline_path_b() -> Path:
    return _baseline_dir() / "m25_fig3b_chemical_potentials.csv"


def plot_path() -> Path:
    return _baseline_dir() / "m25_fig3_chemical_potentials.pdf"


def _write_panel_csv(panel: Fig3PanelResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        # ``lineterminator='\n'`` overrides csv.writer's default CRLF
        # so the baseline CSVs don't trigger ``git diff --check``
        # trailing-whitespace warnings.
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 3 — μ_α(T) full rate-eq; "
            "pinned by qpsim"
        ])
        writer.writerow([
            f"# omega_LR_GHz={panel.omega_LR_GHz:g}  "
            f"Delta_R_GHz={DELTA_R_OVER_H_GHZ:g}  "
            f"omega_10_GHz={OMEGA_10_OVER_H_GHZ:g}  "
            f"Gamma_ph_00_Hz={GAMMA_PH_00_HZ:g}"
        ])
        writer.writerow([
            "T_kelvin", "x_L", "x_Rgt", "x_Rlt", "p_1",
            "mu_L_GHz", "mu_Rgt_GHz", "mu_Rlt_GHz",
        ])
        for i in range(panel.T_kelvin.size):
            writer.writerow([
                f"{panel.T_kelvin[i]:.17e}",
                f"{panel.x_L[i]:.17e}",
                f"{panel.x_Rgt[i]:.17e}",
                f"{panel.x_Rlt[i]:.17e}",
                f"{panel.p_1[i]:.17e}",
                f"{panel.mu_L_GHz[i]:.17e}",
                f"{panel.mu_Rgt_GHz[i]:.17e}",
                f"{panel.mu_Rlt_GHz[i]:.17e}",
            ])
    return path


def _read_panel_csv(path: Path, omega_LR_GHz: float) -> Fig3PanelResult:
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line or line[0].startswith("#") or line[0] == "T_kelvin":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig3PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        T_kelvin=data[:, 0],
        x_L=data[:, 1], x_Rgt=data[:, 2], x_Rlt=data[:, 3], p_1=data[:, 4],
        mu_L_GHz=data[:, 5], mu_Rgt_GHz=data[:, 6], mu_Rlt_GHz=data[:, 7],
    )


def write_baseline(result: Fig3Result) -> tuple[Path, Path]:
    return (
        _write_panel_csv(result.panel_a, baseline_path_a()),
        _write_panel_csv(result.panel_b, baseline_path_b()),
    )


def read_baseline() -> Fig3Result:
    return Fig3Result(
        panel_a=_read_panel_csv(baseline_path_a(), 0.5),
        panel_b=_read_panel_csv(baseline_path_b(), 5.0),
    )


def write_plot(result: Fig3Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, panel, label in (
        (axes[0], result.panel_a, "a"),
        (axes[1], result.panel_b, "b"),
    ):
        T_mK = panel.T_kelvin * 1e3
        ax.plot(T_mK, panel.mu_L_GHz, "o-", lw=1.5, ms=4,
                color="tab:red", label=r"$\mu_L$")
        ax.plot(T_mK, panel.mu_Rgt_GHz, "s-", lw=1.5, ms=4,
                color="tab:blue", label=r"$\mu_{R>}$")
        ax.plot(T_mK, panel.mu_Rlt_GHz, "^-", lw=1.5, ms=4,
                color="tab:green", label=r"$\mu_{R<}$")
        ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel(r"$T$ [mK]", fontsize=12)
        if label == "a":
            ax.set_ylabel(r"$\mu_\alpha / (h/2\pi)$ [GHz]", fontsize=12)
        ax.set_title(
            rf"({label}) $\omega_{{LR}}/(2\pi) = {panel.omega_LR_GHz}$ GHz",
            fontsize=11,
        )
        ax.grid(True, which="both", ls=":", alpha=0.3)
        ax.legend(fontsize=10, loc="upper right")

    fig.suptitle(
        "Marchegiani 2025 Fig 3 — chemical potentials vs temperature\n"
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
    print("M25 Fig 3 — chemical potentials μ_α(T) full rate-equation reproduction")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz (recalibrated at each T)")
    print(
        f"  T sweep: {NUM_T_POINTS} pts, "
        f"{T_MIN_K * 1e3:.0f} → {T_MAX_K * 1e3:.0f} mK"
    )
    print("  Panel a (ω_LR = 0.5 GHz) ...")
    panel_a = _run_panel(0.5)
    print("  Panel b (ω_LR = 5.0 GHz) ...")
    panel_b = _run_panel(5.0)
    result = Fig3Result(panel_a=panel_a, panel_b=panel_b)
    csv_a, csv_b = write_baseline(result)
    pdf = write_plot(result)
    print(f"  Baselines: {csv_a.name}, {csv_b.name}")
    print(f"  PDF plot:  {pdf.name}")
    return csv_a, csv_b, pdf


if __name__ == "__main__":
    generate_baseline()
