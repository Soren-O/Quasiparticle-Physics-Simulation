r"""Marchegiani 2025 Fig 3 — paper-target reproduction of μ_α(T).

This script targets the published Marchegiani & Catelani 2025 Fig 3:
three sub-band chemical potentials $\mu_L(T)$, $\mu_{R>}(T)$,
$\mu_{R<}(T)$ in a gap-asymmetric junction at fixed
``Γ̃^ph_00 = 300 Hz``, plotted as $\mu_\alpha/\Delta_L$ versus $T$ on
two panels:

* Panel (a) — small gap asymmetry, ``ω_LR/(2π) = 0.5 GHz``
  (Δ_L/h = 49.5 GHz, Δ_R/h = 49.0 GHz).
* Panel (b) — large gap asymmetry, ``ω_LR/(2π) = 5 GHz``
  (Δ_L/h = 54.0 GHz, Δ_R/h = 49.0 GHz).

Each panel carries an inset with $\mu_L - \mu_{R<}$ (solid) and
$\mu_{R>} - \mu_{R<}$ (dashed) on a $\Delta\mu/\Delta_L$ /
$\Delta\mu/\omega_{LR}$ axis pair, and the main panels overlay the
dot-dashed constant-density reference line (the locus of $\mu_\alpha =
\Delta_\alpha + T \log x_\alpha^{\rm low\,T}$ at the low-T plateau
density). Regime shading (low-T / crossover / high-T) and the dotted
low-T marker / dashed $\bar T$ marker follow the paper figure.

Method
------
For each $T$ in the sweep, the photon drive scale is recalibrated so
that $\widetilde\Gamma^{\rm ph}_{00} = 300$ Hz (M25 caption value),
the M25 coefficient bundle is built from
:func:`coefficients_from_physical_parameters_with_photon_drive`
(single-quasiparticle Γ̄ normalization of the density equations
included via ``cooper_pair_number_R``), and the steady state is
tracked across the sweep by the deterministic branch-continuation
driver :func:`qpsim.services.rate_equation.solve_rate_equation_branch`
(photon branch continued up from the SI low-T analytic seed, thermal
branch continued down from the full-equilibrium seed, composite per
the driver's documented exchange rule). Chemical potentials are
recovered with the paper-exact inversions of arXiv Eqs. (10)–(13)
(published SI Eqs. S2–S5), including the ``√(Δ_α/2πT)`` and
erf/erfc partition factors.

Branch tracking — status
------------------------
The historical "multi-stability" of the 4-unknown system was an
artifact of running the density equations on the ensemble ``Γ̃``
rates; with the Γ̄ normalization the Fig 3 parameter set has a unique
physical root at every temperature, both continuation passes agree
everywhere ("merged"), and the μ_α(T) curves are smooth through the
crossover with μ → 0 near T̄ ≈ 146 mK — matching the published
figure without any per-point branch picking. The bidirectional
tracker remains the guard against genuine folds at other parameter
sets.

Usage::

    python -m validation.marchegiani_2025.fig3_paper
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from validation.marchegiani_2025.fig3_chemical_potentials import (
    DELTA_R_OVER_H_GHZ,
    GAMMA_PH_00_HZ,
    NUM_T_POINTS,
    OMEGA_10_OVER_H_GHZ,
    T_MAX_K,
    T_MIN_K,
    _chemical_potentials_GHz,
    _T_bar_estimate,
    solve_panel_branch_sweep,
)

_H_OVER_KB = 4.799243e-11   # K / Hz

# Panel-omega_LR pairs.
PANEL_A_OMEGA_LR_GHZ = 0.5
PANEL_B_OMEGA_LR_GHZ = 5.0


@dataclass(frozen=True)
class Fig3PanelResult:
    """Per-panel temperature sweep output."""

    omega_LR_GHz: float
    Delta_L_GHz: float
    T_kelvin: np.ndarray
    x_L: np.ndarray
    x_Rgt: np.ndarray
    x_Rlt: np.ndarray
    p_1: np.ndarray
    mu_L_GHz: np.ndarray
    mu_Rgt_GHz: np.ndarray
    mu_Rlt_GHz: np.ndarray
    T_bar_kelvin: float


@dataclass(frozen=True)
class Fig3Result:
    panel_a: Fig3PanelResult   # ω_LR = 0.5 GHz
    panel_b: Fig3PanelResult   # ω_LR = 5.0 GHz


def _run_panel(omega_LR_GHz: float) -> Fig3PanelResult:
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    # Branch-tracked sweep (photon-up + thermal-down composite); the
    # Fig 3 parameter set has a unique root so the passes merge at
    # every point — see the driver and module docstrings.
    sweep = solve_panel_branch_sweep(omega_LR_GHz, T_sweep)
    x_L = np.array([s.x_L for s in sweep.states])
    x_Rgt = np.array([s.x_Rgt for s in sweep.states])
    x_Rlt = np.array([s.x_Rlt for s in sweep.states])
    p_1 = np.array([s.p_1 for s in sweep.states])

    mu_L_GHz, mu_Rgt_GHz, mu_Rlt_GHz = _chemical_potentials_GHz(
        omega_LR_GHz, T_sweep, x_L, x_Rgt, x_Rlt,
    )

    return Fig3PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        Delta_L_GHz=DELTA_R_OVER_H_GHZ + omega_LR_GHz,
        T_kelvin=T_sweep,
        x_L=x_L, x_Rgt=x_Rgt, x_Rlt=x_Rlt, p_1=p_1,
        mu_L_GHz=mu_L_GHz, mu_Rgt_GHz=mu_Rgt_GHz, mu_Rlt_GHz=mu_Rlt_GHz,
        T_bar_kelvin=_T_bar_estimate(omega_LR_GHz),
    )


def run() -> Fig3Result:
    return Fig3Result(
        panel_a=_run_panel(PANEL_A_OMEGA_LR_GHZ),
        panel_b=_run_panel(PANEL_B_OMEGA_LR_GHZ),
    )


# ── baseline I/O ─────────────────────────────────────────────────────


def _baseline_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "marchegiani_2025"


def baseline_path_a() -> Path:
    """Panel-a CSV path (paper-faithful reproduction)."""
    return _baseline_dir() / "m25_fig3a_paper.csv"


def baseline_path_b() -> Path:
    """Panel-b CSV path (paper-faithful reproduction)."""
    return _baseline_dir() / "m25_fig3b_paper.csv"


def plot_path() -> Path:
    """Combined two-panel PDF (paper-faithful reproduction)."""
    return _baseline_dir() / "m25_fig3_paper.pdf"


def _write_panel_csv(panel: Fig3PanelResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        # ``lineterminator='\n'`` overrides csv.writer's default CRLF
        # so the baseline CSVs don't trigger ``git diff --check``
        # trailing-whitespace warnings.
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 3 — paper-target μ_α(T); "
            "branch-continuation driver on the Γ̄-normalized moment system; "
            "paper-exact μ inversion (SI Eqs. S2–S5)"
        ])
        writer.writerow([
            f"# omega_LR_GHz={panel.omega_LR_GHz:g}  "
            f"Delta_L_GHz={panel.Delta_L_GHz:g}  "
            f"Delta_R_GHz={DELTA_R_OVER_H_GHZ:g}  "
            f"omega_10_GHz={OMEGA_10_OVER_H_GHZ:g}  "
            f"Gamma_ph_00_Hz={GAMMA_PH_00_HZ:g}  "
            f"T_bar_kelvin={panel.T_bar_kelvin:.6e}"
        ])
        # Fixed-point selection is platform-dependent; the strict pin
        # test only runs where the baseline was generated.
        writer.writerow([f"# pinned_on: {sys.platform}"])
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
    T_bar_kelvin: float | None = None
    Delta_L_GHz = DELTA_R_OVER_H_GHZ + omega_LR_GHz
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("#"):
                # Parse T_bar from the metadata line if present.
                if "T_bar_kelvin=" in first:
                    for token in first.split():
                        if token.startswith("T_bar_kelvin="):
                            T_bar_kelvin = float(token.split("=", 1)[1])
                continue
            if first == "T_kelvin":
                continue
            rows.append([float(x) for x in line])
    if T_bar_kelvin is None:
        # Fallback for older CSVs without T̄ in the header.
        T_bar_kelvin = _T_bar_estimate(omega_LR_GHz)
    data = np.array(rows, dtype=float)
    return Fig3PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        Delta_L_GHz=Delta_L_GHz,
        T_kelvin=data[:, 0],
        x_L=data[:, 1], x_Rgt=data[:, 2], x_Rlt=data[:, 3], p_1=data[:, 4],
        mu_L_GHz=data[:, 5], mu_Rgt_GHz=data[:, 6], mu_Rlt_GHz=data[:, 7],
        T_bar_kelvin=T_bar_kelvin,
    )


def write_baseline(result: Fig3Result) -> tuple[Path, Path]:
    return (
        _write_panel_csv(result.panel_a, baseline_path_a()),
        _write_panel_csv(result.panel_b, baseline_path_b()),
    )


def read_baseline() -> Fig3Result:
    return Fig3Result(
        panel_a=_read_panel_csv(baseline_path_a(), PANEL_A_OMEGA_LR_GHZ),
        panel_b=_read_panel_csv(baseline_path_b(), PANEL_B_OMEGA_LR_GHZ),
    )


# ── plotting ─────────────────────────────────────────────────────────

# Paper-figure colors (eyeballed from m25_fig3_paper.png):
#   μ_L      = deep purple
#   μ_R>     = teal
#   μ_R<     = yellow
_COLOR_MU_L = "#3B0F70"
_COLOR_MU_RGT = "#2C7E7E"
_COLOR_MU_RLT = "#F0E442"
_COLOR_DOT_DASH = "#101010"

# Regime-shading colors per panel (from the inset color bands):
#   panel a: blue (low T) → orange (mid) → yellow (high T)
#   panel b: blue (low T) → green (mid)  → yellow (high T)
_SHADE_ALPHA = 0.22
_SHADE_PANEL_A = ("#9FC2E6", "#F2B070", "#F4DC74")
_SHADE_PANEL_B = ("#9FC2E6", "#B7D497", "#F4DC74")


def _plot_panel(
    ax,
    inset_ax,
    panel: Fig3PanelResult,
    *,
    shading: tuple[str, str, str],
    panel_label: str,
    asym_label: str,
    show_legend: bool,
) -> None:
    """Plot one paper-style panel (μ_α/Δ_L vs T) with inset."""

    T_K = panel.T_kelvin
    Delta_L_GHz = panel.Delta_L_GHz
    Delta_R_GHz = DELTA_R_OVER_H_GHZ

    # Regime shading: low-T / crossover / high-T bands. Boundary
    # placement follows the paper-figure inset color split — at ~T̄/3
    # (where the photon-drive plateau peels off the constant-density
    # asymptote) and at T̄ (Lambert-W crossover).
    T_bar = panel.T_bar_kelvin
    T_lo_edge = T_bar / 3.0
    T_hi_edge = T_bar
    T_full_lo = float(T_K.min())
    T_full_hi = float(T_K.max())
    ax.axvspan(T_full_lo, T_lo_edge, color=shading[0], alpha=_SHADE_ALPHA, zorder=0)
    ax.axvspan(T_lo_edge, T_hi_edge, color=shading[1], alpha=_SHADE_ALPHA, zorder=0)
    ax.axvspan(T_hi_edge, T_full_hi, color=shading[2], alpha=_SHADE_ALPHA, zorder=0)

    # Vertical reference markers from the paper:
    #   dotted = low-T sample T (≈ 20 mK on the paper figure)
    #   dashed = T̄ crossover (Lambert-W)
    ax.axvline(0.020, color="black", ls=":", lw=0.7, alpha=0.6, zorder=1)
    ax.axvline(T_bar, color="black", ls="--", lw=0.7, alpha=0.6, zorder=1)

    # Dot-dashed constant-density reference. The reference holds x_α
    # fixed at its low-T plateau value (taken at the lowest T in the
    # sweep), so the line traces μ_α = Δ_α + T·log(x_α^{plateau}). One
    # line per sub-band, but in the paper figure they overlay onto a
    # single black dot-dashed locus near μ_R<; we draw the μ_R< line
    # to match the paper's visual.
    x_Rlt_plateau = float(panel.x_Rlt[0])
    Delta_R_K = Delta_R_GHz * 1e9 * _H_OVER_KB
    mu_Rlt_const = (
        Delta_R_K + T_K * np.log(max(x_Rlt_plateau, 1e-300))
    ) / _H_OVER_KB / 1e9
    ax.plot(
        T_K, mu_Rlt_const / Delta_L_GHz,
        color=_COLOR_DOT_DASH, ls=(0, (3, 1, 1, 1)), lw=1.2,
        zorder=2, label=r"const-$x_{R<}$",
    )

    # Main μ_α curves.
    ax.plot(
        T_K, panel.mu_L_GHz / Delta_L_GHz,
        color=_COLOR_MU_L, lw=2.0, zorder=3, label=r"$\mu_L$",
    )
    ax.plot(
        T_K, panel.mu_Rgt_GHz / Delta_L_GHz,
        color=_COLOR_MU_RGT, lw=2.0, zorder=3, label=r"$\mu_{R>}$",
    )
    ax.plot(
        T_K, panel.mu_Rlt_GHz / Delta_L_GHz,
        color=_COLOR_MU_RLT, lw=2.0, zorder=3, label=r"$\mu_{R<}$",
    )

    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4, zorder=1)

    ax.set_ylabel(r"$\mu_\alpha / \Delta_L$", fontsize=12)
    ax.set_xlim(T_full_lo, T_full_hi)
    ax.set_ylim(-0.05, 1.05)
    ax.text(
        0.97, 0.05, f"({panel_label})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=12,
    )
    ax.text(
        1.01, 0.5, asym_label,
        transform=ax.transAxes, ha="left", va="center",
        rotation=90, fontsize=10,
    )

    if show_legend:
        ax.legend(fontsize=9, loc="lower left", frameon=False)

    # ── inset: chemical-potential differences ─────────────────────
    # Inset shading: same three-band color scheme as the main panel
    # (blue / orange|green / yellow). Inset x-axis: T in Kelvin
    # extended to 0.25 (paper figure inset axis).
    T_inset_lo = 0.0
    T_inset_hi = 0.25
    inset_ax.axvspan(T_inset_lo, T_lo_edge, color=shading[0], alpha=_SHADE_ALPHA, zorder=0)
    inset_ax.axvspan(T_lo_edge, T_hi_edge, color=shading[1], alpha=_SHADE_ALPHA, zorder=0)
    inset_ax.axvspan(T_hi_edge, T_inset_hi, color=shading[2], alpha=_SHADE_ALPHA, zorder=0)
    inset_ax.axhline(0.0, color="black", lw=0.4, alpha=0.4, zorder=1)

    dmu_L_minus_Rlt = (panel.mu_L_GHz - panel.mu_Rlt_GHz) / Delta_L_GHz
    dmu_Rgt_minus_Rlt = (panel.mu_Rgt_GHz - panel.mu_Rlt_GHz) / Delta_L_GHz

    inset_ax.plot(
        T_K, dmu_L_minus_Rlt,
        color="black", lw=1.4, zorder=3, label=r"$\mu_L - \mu_{R<}$",
    )
    inset_ax.plot(
        T_K, dmu_Rgt_minus_Rlt,
        color="black", ls="--", lw=1.4, zorder=3, label=r"$\mu_{R>} - \mu_{R<}$",
    )

    # Diagonal gray line in the paper-figure inset traces ω_LR/Δ_L on
    # the right axis — i.e. a unit-slope reference in
    # (Δμ/ω_LR) units. We render it as a faint guide.
    inset_ax.plot(
        [T_inset_lo, T_inset_hi],
        [
            T_inset_lo * (panel.omega_LR_GHz / Delta_L_GHz)
            * (1.0 / max(panel.omega_LR_GHz / Delta_L_GHz, 1e-30)),
            T_inset_hi * (panel.omega_LR_GHz / Delta_L_GHz)
            * (1.0 / max(panel.omega_LR_GHz / Delta_L_GHz, 1e-30)),
        ],
        color="gray", lw=2.0, alpha=0.6, zorder=2,
    )

    inset_ax.set_xlim(T_inset_lo, T_inset_hi)
    # y-limits: pick a comfortable window above zero.
    ymax = max(
        float(np.nanmax(dmu_L_minus_Rlt)) * 1.2,
        float(panel.omega_LR_GHz / Delta_L_GHz) * 1.2,
    )
    inset_ax.set_ylim(-0.005, max(ymax, 0.01))
    inset_ax.set_xlabel(r"$T$ [K]", fontsize=8, labelpad=1)
    inset_ax.set_ylabel(r"$\Delta\mu/\Delta_L$", fontsize=8, labelpad=1)
    inset_ax.tick_params(axis="both", labelsize=7, pad=1)
    inset_ax.legend(fontsize=7, loc="upper right", frameon=False)

    # Right-side secondary y-axis: Δμ/ω_LR.
    sec = inset_ax.secondary_yaxis(
        "right",
        functions=(
            lambda y: y * Delta_L_GHz / panel.omega_LR_GHz,
            lambda y: y * panel.omega_LR_GHz / Delta_L_GHz,
        ),
    )
    sec.set_ylabel(r"$\Delta\mu/\omega_{LR}$", fontsize=8, labelpad=1)
    sec.tick_params(axis="y", labelsize=7, pad=1)


def write_plot(result: Fig3Result, path: Path | None = None) -> Path:
    """Two-panel paper-style plot with insets and regime shading."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        2, 1, figsize=(7.0, 8.5), sharex=True,
    )

    # Top axis (T in units of Δ_R/k_B) — placed on panel a only.
    Delta_R_K = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    sec_top = ax_a.secondary_xaxis(
        "top",
        functions=(lambda T: T / Delta_R_K, lambda u: u * Delta_R_K),
    )
    sec_top.set_xlabel(r"Temperature [$\Delta_R / k_B$]", fontsize=11)

    # Insets (occupy upper-right of each panel).
    inset_a = inset_axes(
        ax_a, width="48%", height="50%",
        loc="upper right", borderpad=1.0,
    )
    inset_b = inset_axes(
        ax_b, width="48%", height="50%",
        loc="upper right", borderpad=1.0,
    )

    _plot_panel(
        ax_a, inset_a, result.panel_a,
        shading=_SHADE_PANEL_A,
        panel_label="a",
        asym_label="Small gap asymmetry",
        show_legend=True,
    )
    _plot_panel(
        ax_b, inset_b, result.panel_b,
        shading=_SHADE_PANEL_B,
        panel_label="b",
        asym_label="Large gap asymmetry",
        show_legend=False,
    )

    ax_b.set_xlabel(r"$T$ [K]", fontsize=12)

    fig.suptitle(
        "Marchegiani 2025 Fig 3 — μ_α(T), branch-continuation tracked\n"
        rf"$\Delta_R/h = {DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi) = {OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}} = {GAMMA_PH_00_HZ:g}$ Hz",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path, Path]:
    print("M25 Fig 3 paper-target reproduction")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, "
          f"ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz (recalibrated at each T)")
    print(
        f"  T sweep: {NUM_T_POINTS} pts, "
        f"{T_MIN_K * 1e3:.0f} → {T_MAX_K * 1e3:.0f} mK"
    )
    print(
        "  Branch tracking: bidirectional continuation driver "
        "(solve_rate_equation_branch); unique root — passes merge."
    )
    print(f"  Panel a (ω_LR = {PANEL_A_OMEGA_LR_GHZ} GHz) ...")
    panel_a = _run_panel(PANEL_A_OMEGA_LR_GHZ)
    print(f"  Panel b (ω_LR = {PANEL_B_OMEGA_LR_GHZ} GHz) ...")
    panel_b = _run_panel(PANEL_B_OMEGA_LR_GHZ)
    result = Fig3Result(panel_a=panel_a, panel_b=panel_b)
    csv_a, csv_b = write_baseline(result)
    pdf = write_plot(result)
    print(f"  Baselines: {csv_a.name}, {csv_b.name}")
    print(f"  PDF plot:  {pdf.name}")
    return csv_a, csv_b, pdf


if __name__ == "__main__":
    generate_baseline()
