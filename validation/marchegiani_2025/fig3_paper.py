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
:func:`coefficients_from_physical_parameters_with_photon_drive`, and
the 4-unknown rate-equation steady state is solved via
:func:`qpsim.services.rate_equation.solve_rate_equation_steady_state_multi_seed`
(scipy ``root(method='hybr')`` under a deterministic seed grid). The
chemical potentials are recovered as $\mu_\alpha = \Delta_\alpha +
T \log x_\alpha$.

Branch tracking — gap (load-bearing)
------------------------------------
The M25 4-unknown system is multi-stable. The paper-target
reproduction calls for a continuation/bifurcation tracker so the
high-$T$ transition is smooth; that is **not** implemented here.
Instead, the simpler **prev-$T$ continuation seeding** is used:
each $T$-step warm-starts the multi-seed solver with the previous
$T$'s converged state as ``preferred_seed``, so the solver tracks
the same branch across small $T$ changes. The fallback to max-$x_L$
remains active for cases where the prev-$T$ seed leaves the basin
of attraction of the photon-driven branch (typically near $\bar T$,
where the system genuinely bifurcates). This delivers the paper's
qualitative shape and the ~2% paper-agreement quoted in the existing
M25 Fig 3 reproduction, but kinks at bifurcations are **not**
algorithmically suppressed; visible kinks near $\bar T$ are real
artifacts of the heuristic, not physics. A full continuation tracker
would land here next.

Naming honesty
--------------
Because the bifurcation tracker is a stub (prev-$T$ seeding rather
than a true continuation algorithm) the output artifacts carry the
``_qpsim_native`` suffix, not ``_paper``: the curves are paper-style
formatted but the high-$T$ tail is **not** algorithmically smooth.
The plot title, run banner and CSV header all flag this. Once a
continuation tracker lands (and the kinks vanish at the level of
the paper figure), rename artifacts to ``m25_fig3_paper.{csv,pdf}``
and drop the gap text from the docstring + plot title.

Usage::

    python -m validation.marchegiani_2025.fig3_paper
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import (
    M25SteadyState,
    analytic_low_T_seed,
    crossover_temperature_kelvin,
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
# and below by where the multi-seed helper still finds the
# nonequilibrium branch.
T_MIN_K = 0.010
T_MAX_K = 0.150
NUM_T_POINTS = 29   # 5 mK spacing

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
    extra_seeds: list[np.ndarray] | None = None,
    expected_ordering: tuple[str, ...] | None = None,
) -> M25SteadyState | None:
    """Solve via the multi-seed helper, optionally warm-started.

    Passing the previous-$T$ converged state as ``preferred_seed`` is
    the prev-$T$ continuation seeding step that turns the otherwise
    independent per-$T$ solves into a tracked sweep. We use
    ``branch_picker_mode="lock_to_preferred"``: if the prev-$T$ seed
    converges to a positive-density branch the picker keeps it
    regardless of $x_L$ magnitude (fixing the load-bearing M25 Fig 3a
    inversion where max-$x_L$ selected an inverted-ordering branch).
    First-$T$ point with no preferred seed falls back to min-residual,
    optionally constrained by ``expected_ordering`` so the analytic
    T → 0 seed (M25 SI Eqs. S69-S71) anchors to the photon-driven
    physical branch instead of the inverted-ordering high-x_L root.
    """
    try:
        return solve_rate_equation_steady_state_multi_seed(
            coefs, preferred_seed=previous,
            extra_seeds=extra_seeds,
            expected_ordering=expected_ordering,
            branch_picker_mode="lock_to_preferred",
        )
    except RuntimeError:
        return None


def _T_bar_for_panel(omega_LR_GHz: float) -> float:
    """Estimate T̄ at the panel's mid-sweep coefficients.

    Used only for plot annotation (vertical dashed line + regime
    shading boundaries). The exact T̄ varies weakly with the
    coefficient bundle; the M25 Fig 3 caption parameters give ~150 mK.
    """
    coefs = _coefficients_at(omega_LR_GHz, 0.080)
    Delta_R_kelvin = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    return crossover_temperature_kelvin(
        Delta_R_kelvin=Delta_R_kelvin,
        r_Rlt_rate_Hz=R_RECOMB_HZ,
        g_photon_R_rate_Hz=max(coefs.g_Rlt, 1e-30),
    )


def _run_panel(omega_LR_GHz: float) -> Fig3PanelResult:
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    x_L = np.full(n, np.nan)
    x_Rgt = np.full(n, np.nan)
    x_Rlt = np.full(n, np.nan)
    p_1 = np.full(n, np.nan)

    # Asymmetry-dependent ordering and analytic-seed case selection
    # (paper Figs. S1, S2):
    #   ω_LR/2π ≥ 1 GHz → large-asymmetry: x_R< ≫ x_L ≫ x_R> (SI Eqs. S69-S71)
    #   ω_LR/2π <  1 GHz → small-asymmetry: ~all equal (SI Eqs. S64-S66, no
    #                                             ordering filter)
    if omega_LR_GHz >= 1.0:
        analytic_case = "large_asymmetry"
        expected_ordering: tuple[str, ...] | None = ("x_Rlt", "x_L", "x_Rgt")
    else:
        analytic_case = "small_asymmetry"
        expected_ordering = None

    last_y: np.ndarray | None = None
    prev_prev_y: np.ndarray | None = None
    for i, T_K in enumerate(T_sweep):
        coefs = _coefficients_at(omega_LR_GHz, float(T_K))
        # Analytic SI low-T seed only at the first (lowest-T) point —
        # the closure assumptions of S64-S66 / S69-S71 are valid for
        # T ≪ ω_LR (≪ T̄), so the seed is most reliable at the start of
        # the sweep. Subsequent T points warm-start from the previous
        # converged state, preserving branch identity.
        extra_seeds: list[np.ndarray] | None = None
        if i == 0:
            seed = analytic_low_T_seed(coefs, float(T_K), case=analytic_case)
            extra_seeds = [seed]
        elif i >= 2 and prev_prev_y is not None and last_y is not None:
            # Predictor step: linear extrapolation of the last two
            # converged states to T[i]. The corrector is the multi-seed
            # solve itself — passing y_pred as an extra_seeds entry lets
            # the picker compete y_pred against the prev-T seed and the
            # default seed grid. ``lock_to_preferred`` still keeps
            # last_y when last_y converges, so the predictor only
            # matters when continuation from last_y would lose the
            # branch (the M25 Fig 3/4 ~70 mK spike case).
            dT = T_sweep[i] - T_sweep[i - 1]
            dT_prev = T_sweep[i - 1] - T_sweep[i - 2]
            if dT_prev > 0.0:
                y_pred = last_y + (last_y - prev_prev_y) * (dT / dT_prev)
                # Keep the predictor on the physical side of the
                # boundary (Newton's basin shrinks rapidly once a
                # density crosses zero). Floor matches
                # analytic_low_T_seed's clamp.
                y_pred = np.array([
                    float(np.clip(y_pred[0], 1e-12, 1.0 - 1e-12)),
                    max(float(y_pred[1]), 1e-30),
                    max(float(y_pred[2]), 1e-30),
                    max(float(y_pred[3]), 1e-30),
                ])
                extra_seeds = [y_pred]
        sol = _try_solve(
            coefs, previous=last_y,
            extra_seeds=extra_seeds,
            expected_ordering=expected_ordering,
        )
        if sol is None:
            raise RuntimeError(
                f"M25 Fig 3 panel ω_LR={omega_LR_GHz} GHz: no seed yielded "
                f"a positive-density solution at T = {T_K:.4f} K."
            )
        x_L[i] = sol.x_L
        x_Rgt[i] = sol.x_Rgt
        x_Rlt[i] = sol.x_Rlt
        p_1[i] = sol.p_1
        prev_prev_y = last_y
        last_y = np.array([sol.p_1, sol.x_L, sol.x_Rgt, sol.x_Rlt])

    Delta_L_GHz = DELTA_R_OVER_H_GHZ + omega_LR_GHz
    Delta_L_K = Delta_L_GHz * 1e9 * _H_OVER_KB
    Delta_R_K = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    mu_L_GHz = (Delta_L_K + T_sweep * np.log(x_L)) / _H_OVER_KB / 1e9
    mu_Rgt_GHz = (Delta_R_K + T_sweep * np.log(x_Rgt)) / _H_OVER_KB / 1e9
    mu_Rlt_GHz = (Delta_R_K + T_sweep * np.log(x_Rlt)) / _H_OVER_KB / 1e9

    return Fig3PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        Delta_L_GHz=Delta_L_GHz,
        T_kelvin=T_sweep,
        x_L=x_L, x_Rgt=x_Rgt, x_Rlt=x_Rlt, p_1=p_1,
        mu_L_GHz=mu_L_GHz, mu_Rgt_GHz=mu_Rgt_GHz, mu_Rlt_GHz=mu_Rlt_GHz,
        T_bar_kelvin=_T_bar_for_panel(omega_LR_GHz),
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
    """Panel-a CSV path (qpsim-native suffix; see module docstring)."""
    return _baseline_dir() / "m25_fig3a_qpsim_native.csv"


def baseline_path_b() -> Path:
    """Panel-b CSV path (qpsim-native suffix; see module docstring)."""
    return _baseline_dir() / "m25_fig3b_qpsim_native.csv"


def plot_path() -> Path:
    """Combined two-panel PDF (qpsim-native suffix; see module docstring)."""
    return _baseline_dir() / "m25_fig3_qpsim_native.pdf"


def _write_panel_csv(panel: Fig3PanelResult, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        # ``lineterminator='\n'`` overrides csv.writer's default CRLF
        # so the baseline CSVs don't trigger ``git diff --check``
        # trailing-whitespace warnings.
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 3 — qpsim-native μ_α(T) "
            "with prev-T continuation seeding"
        ])
        writer.writerow([
            f"# omega_LR_GHz={panel.omega_LR_GHz:g}  "
            f"Delta_L_GHz={panel.Delta_L_GHz:g}  "
            f"Delta_R_GHz={DELTA_R_OVER_H_GHZ:g}  "
            f"omega_10_GHz={OMEGA_10_OVER_H_GHZ:g}  "
            f"Gamma_ph_00_Hz={GAMMA_PH_00_HZ:g}  "
            f"T_bar_kelvin={panel.T_bar_kelvin:.6e}"
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
        T_bar_kelvin = _T_bar_for_panel(omega_LR_GHz)
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
        "Marchegiani 2025 Fig 3 — μ_α(T) with prev-T continuation seeding\n"
        rf"$\Delta_R/h = {DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi) = {OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}} = {GAMMA_PH_00_HZ:g}$ Hz"
        "\n"
        "qpsim-native: bifurcation tracking is prev-T continuation only "
        "(not full continuation); kinks near $\\bar T$ are heuristic, not "
        "physics.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path, Path]:
    print("M25 Fig 3 paper-target reproduction (qpsim-native artifact)")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, "
          f"ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz (recalibrated at each T)")
    print(
        f"  T sweep: {NUM_T_POINTS} pts, "
        f"{T_MIN_K * 1e3:.0f} → {T_MAX_K * 1e3:.0f} mK"
    )
    print(
        "  Branch tracking: prev-T continuation seeding only (full "
        "continuation/bifurcation tracker not yet implemented)."
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
