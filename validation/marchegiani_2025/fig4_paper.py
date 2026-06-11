r"""Marchegiani 2025 Fig 4 — qpsim-native parity-rate reproduction at paper topology.

This is the **structural** Fig. 4 reproduction: two stacked panels in
the paper's sweep topology, both ω_LR cases overlaid, on the M25 Fig 3
caption parameter set. But until the two paper-parity gaps below close,
the artifact this script writes is **not** a paper-faithful reproduction;
it is a qpsim-native characterization at the paper's topology. The CSV /
PDF filenames and plot titles make that explicit so a downstream consumer
of the artifact is not misled.

The published Marchegiani & Catelani 2025 Fig. 4 has two panels:

* **Panel (a).** $\Gamma_P(T)$ on linear axes, $T \in [0, 0.15]$~K, with
  three curves per ω_LR case: the full nonequilibrium model (solid),
  the global quasiequilibrium reduction (dashed), and the renormalized
  reduction (dot-dashed). Both ω_LR/(2π) ∈ {0.5, 5} GHz cases overlaid.
* **Panel (b).** $\widetilde\Gamma^{\rm eo}_{01} / \widetilde\Gamma^{\rm eo}_{10}$
  on linear axes, same T range, same three-curve × two-ω_LR overlay.
  Bottom panel also has a black dotted reference (paper text: photon-only
  "$\widetilde\Gamma^{\rm ph}_{01}/\widetilde\Gamma^{\rm ph}_{10}$" baseline
  at T=0; we emit it from the photon coefficients directly).

The existing :mod:`fig4_parity_rates` script computes only the full-model
solid curves at the same parameter set. This script is the paper-target
upgrade: same full-model logic, plus the two reduced-model overlays and
paper-style dashed/dot-dashed line styling.

Outstanding paper-parity gaps (load-bearing)
--------------------------------------------

1. **Panel (a) multi-stability noise.** The M25 four-unknown moment
   system is multi-stable. The current branch picker
   (:func:`solve_rate_equation_steady_state_multi_seed`, max-x_L with
   previous-T continuation fall-back) jumps between competing
   nonequilibrium fixed points when their basins of attraction shift
   with T, producing visible scatter at some T points on Γ_P(T). Paper-
   grade smoothness needs proper bifurcation tracking (predictor-
   corrector continuation across each fixed point, not just the
   previous-T seed). Flagged in :mod:`STATUS.md` and in the M25 Fig 4
   page of the validation plan.

2. **Reduced-model overlays — placeholder.** The paper's "global
   quasiequilibrium" and "renormalized" reductions have not been
   derived from text on hand here. Until they are pinned analytically,
   :func:`_global_quasiequilibrium_curve` and :func:`_renormalized_curve`
   return clearly-labeled placeholders (the global-quasiequilibrium one
   reuses the full-model thermal moments at T_bath; the renormalized
   one collapses x_{R>} = x_{R<} after the full-model solve and re-
   evaluates Γ̃^eo against the collapsed pair). Both are qualitatively
   in the right ballpark but are **not** paper formulas. Same convention
   as :func:`_xqp_analytic_eq47` in :mod:`fig5_paper`: the function bodies
   are TODO-marked and the plot title carries the warning.

Once both gaps land, this script becomes the paper-faithful Fig 4
reproduction. At that point: rename the artifacts to
``m25_fig4_paper.{csv,pdf}``, drop the "qpsim-native" wording from the
plot title, and remove the warning text from the run banner.

M25 Fig 3/4 caption parameters (shared with :mod:`fig3_chemical_potentials`):

    Δ_R/h         = 49.0 GHz
    ω_10/(2π)     = 5.5  GHz
    E_J/h         = 14.5 GHz
    E_C/h         = 0.290 GHz
    r^L = r^{R<}  = 6.25e6 Hz
    Γ̃^ee_10      = 100 kHz
    Γ̃^ph_00      = 300 Hz  (recalibrated at each T)

Usage::

    python -m validation.marchegiani_2025.fig4_paper
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import (
    M25Coefficients,
    M25SteadyState,
    analytic_low_T_seed,
)

from validation.marchegiani_2025.fig3_chemical_potentials import (
    DELTA_R_OVER_H_GHZ,
    GAMMA_PH_00_HZ,
    NUM_T_POINTS,
    OMEGA_10_OVER_H_GHZ,
    T_MAX_K,
    T_MIN_K,
    _coefficients_at,
)

# Use the smarter ``_try_solve`` from ``fig3_paper``: it carries
# ``extra_seeds`` (for the analytic SI low-T seed and the predictor-
# corrector step) and ``expected_ordering`` (large-asymmetry density
# ordering filter) on top of the basic prev-T continuation seeding
# in ``fig3_chemical_potentials._try_solve``.
from validation.marchegiani_2025.fig3_paper import _try_solve

# ── reduced-model labels (used on the plot legend) ───────────────────
MODEL_FULL = "full"
MODEL_GLOBAL = "global"
MODEL_RENORM = "renorm"
MODEL_LABELS: dict[str, str] = {
    MODEL_FULL: "Full model",
    MODEL_GLOBAL: "Global (placeholder)",
    MODEL_RENORM: "Renormalized (placeholder)",
}


@dataclass(frozen=True)
class Fig4PanelResult:
    """Per-(ω_LR, model) temperature sweep output."""

    omega_LR_GHz: float
    model: str                          # one of MODEL_FULL/GLOBAL/RENORM
    T_kelvin: np.ndarray
    Gamma_P_Hz: np.ndarray
    ratio_eo_01_over_10: np.ndarray


@dataclass(frozen=True)
class Fig4Result:
    """Full Fig 4 reproduction: 2 ω_LR cases × 3 models = 6 panels.

    The full-model panels are the load-bearing reproduction; the global
    and renormalized panels are placeholders until the paper formulas
    are pinned (see module docstring, gap 2).
    """

    panels: dict[tuple[float, str], Fig4PanelResult]

    def get(self, omega_LR_GHz: float, model: str) -> Fig4PanelResult:
        return self.panels[(omega_LR_GHz, model)]


# ── core observables on a single fixed-point solution ────────────────


def _gamma_eo(coefs: M25Coefficients, sol: M25SteadyState) -> np.ndarray:
    """Effective parity-flipping rate matrix Γ̃^eo at the fixed point.

    Same expression as :mod:`fig4_parity_rates._gamma_eo`:

        Γ̃^eo_{ij} = Γ̃^ph_{ij} + Γ̃^L_{ij} x_L
                                  + Γ̃^{R>}_{ij} x_{R>}
                                  + Γ̃^{R<}_{ij} x_{R<}
    """
    return (
        coefs.gamma_ph
        + coefs.gammas_L * sol.x_L
        + coefs.gammas_Rgt * sol.x_Rgt
        + coefs.gammas_Rlt * sol.x_Rlt
    )


def _gamma_P(sol: M25SteadyState, gamma_eo: np.ndarray) -> float:
    """Total parity-switching rate Γ_P from the qubit master equation.

    Γ_P = p_0 (Γ̃^eo_{01} + Γ̃^eo_{00}) + p_1 (Γ̃^eo_{10} + Γ̃^eo_{11}).
    """
    return float(
        sol.p_0 * (gamma_eo[0, 1] + gamma_eo[0, 0])
        + sol.p_1 * (gamma_eo[1, 0] + gamma_eo[1, 1])
    )


def _ratio_eo(gamma_eo: np.ndarray) -> float:
    """Excitation/relaxation ratio Γ̃^eo_{01}/Γ̃^eo_{10}.

    Returns NaN if the relaxation channel vanishes (e.g. zero photon
    drive at T = 0 limit), which on the plot leaves a gap rather than
    inserting a spurious infinity.
    """
    if gamma_eo[1, 0] > 0:
        return float(gamma_eo[0, 1] / gamma_eo[1, 0])
    return float("nan")


# ── full-model curve (shared with fig4_parity_rates) ─────────────────


def _solve_branch_tracked_sweep(
    omega_LR_GHz: float,
    T_sweep: np.ndarray,
    *,
    label: str,
) -> tuple[
    list[M25Coefficients], list[M25SteadyState],
]:
    """Solve the M25 fixed point at each T with branch-tracked seeding.

    Mirrors :func:`fig3_paper._run_panel`'s seeding strategy: analytic
    SI low-T seed at the first point, prev-T continuation seed plus a
    linear-extrapolation predictor for ``i ≥ 2``, ``expected_ordering``
    filter for the large-asymmetry case. Returns the per-T coefficient
    bundles and the corresponding converged states; downstream curves
    re-evaluate Γ_P / ratio observables (and any model collapse) on
    the shared solver output.
    """
    if omega_LR_GHz >= 1.0:
        analytic_case = "large_asymmetry"
        expected_ordering: tuple[str, ...] | None = ("x_Rlt", "x_L", "x_Rgt")
    else:
        analytic_case = "small_asymmetry"
        expected_ordering = None

    coefs_list: list[M25Coefficients] = []
    sols: list[M25SteadyState] = []
    last_y: np.ndarray | None = None
    prev_prev_y: np.ndarray | None = None
    for i, T_K in enumerate(T_sweep):
        coefs = _coefficients_at(omega_LR_GHz, float(T_K))
        extra_seeds: list[np.ndarray] | None = None
        if i == 0:
            seed = analytic_low_T_seed(coefs, float(T_K), case=analytic_case)
            extra_seeds = [seed]
        elif i >= 2 and prev_prev_y is not None and last_y is not None:
            # Predictor: linear extrapolation of the last two converged
            # states. The corrector is the multi-seed solve itself —
            # the picker competes y_pred against the prev-T seed and
            # default seed grid; ``lock_to_preferred`` still keeps
            # last_y unless its solve fails. See fig3_paper._run_panel.
            dT = T_sweep[i] - T_sweep[i - 1]
            dT_prev = T_sweep[i - 1] - T_sweep[i - 2]
            if dT_prev > 0.0:
                y_pred = last_y + (last_y - prev_prev_y) * (dT / dT_prev)
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
                f"M25 Fig 4 ({label}) panel ω_LR={omega_LR_GHz} GHz: no seed "
                f"yielded a positive-density solution at T = {T_K:.4f} K."
            )
        coefs_list.append(coefs)
        sols.append(sol)
        prev_prev_y = last_y
        last_y = np.array([sol.p_1, sol.x_L, sol.x_Rgt, sol.x_Rlt])
    return coefs_list, sols


def _full_curve(omega_LR_GHz: float) -> Fig4PanelResult:
    """Full-model Γ_P(T) and Γ̃^eo_{01}/Γ̃^eo_{10} sweep.

    Identical to :func:`fig4_parity_rates._run_panel`: re-implemented
    here rather than imported so the three model curves share one
    panel-result dataclass and the CSV layout is uniform.
    """
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    coefs_list, sols = _solve_branch_tracked_sweep(
        omega_LR_GHz, T_sweep, label="full",
    )
    for i, (coefs, sol) in enumerate(zip(coefs_list, sols, strict=True)):
        gamma_eo = _gamma_eo(coefs, sol)
        Gamma_P[i] = _gamma_P(sol, gamma_eo)
        ratio[i] = _ratio_eo(gamma_eo)

    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        model=MODEL_FULL,
        T_kelvin=T_sweep,
        Gamma_P_Hz=Gamma_P,
        ratio_eo_01_over_10=ratio,
    )


# ── reduced-model placeholders (load-bearing TODO; see docstring) ────


def _global_quasiequilibrium_curve(omega_LR_GHz: float) -> Fig4PanelResult:
    """Placeholder for the M25 global-quasiequilibrium reduction.

    TODO(paper-parity): replace with the paper's reduction in which all
    three electrodes share one chemical potential μ = μ_L = μ_{R>} =
    μ_{R<}. The collapsed system is a single moment equation in μ at
    each T whose Γ_P / ratio outputs reproduce the paper's dashed
    curves. The proper derivation needs the paper-text definition of
    the collapsed photon-driven generation rate, which has not been
    transcribed here yet.

    Current placeholder behaviour: solve the full system as in
    :func:`_full_curve`, then enforce x_L = x_{R>} = x_{R<} = (full
    geometric mean) and re-evaluate Γ̃^eo / ratio against that
    collapsed density. This is qualitatively in the right ballpark
    (dashed curves trend below the full curves at low T, converge to
    them at high T) but is **not** the paper's formula.
    """
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    coefs_list, sols = _solve_branch_tracked_sweep(
        omega_LR_GHz, T_sweep, label="global",
    )
    for i, (coefs, sol) in enumerate(zip(coefs_list, sols, strict=True)):
        # Collapse: single μ → single x for all electrodes. Geometric
        # mean preserves the rough multiplicative scale across the three
        # bands (placeholder; paper formula goes here).
        x_common = float(np.cbrt(max(sol.x_L * sol.x_Rgt * sol.x_Rlt, 0.0)))
        collapsed = M25SteadyState(
            p_0=sol.p_0, p_1=sol.p_1,
            x_L=x_common, x_Rgt=x_common, x_Rlt=x_common,
            residual_inf_norm=sol.residual_inf_norm,
            n_function_evaluations=sol.n_function_evaluations,
        )
        gamma_eo = _gamma_eo(coefs, collapsed)
        Gamma_P[i] = _gamma_P(collapsed, gamma_eo)
        ratio[i] = _ratio_eo(gamma_eo)

    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        model=MODEL_GLOBAL,
        T_kelvin=T_sweep,
        Gamma_P_Hz=Gamma_P,
        ratio_eo_01_over_10=ratio,
    )


def _renormalized_curve(omega_LR_GHz: float) -> Fig4PanelResult:
    """Placeholder for the M25 renormalized reduction.

    TODO(paper-parity): replace with the paper's reduction in which
    μ_L ≠ μ_R but μ_{R>} = μ_{R<} are collapsed via fast intraband
    relaxation (τ_R, τ_E → ∞). The proper derivation reduces the
    four-unknown system to a three-unknown one (p_1, x_L, x_R); needs
    paper-text confirmation of the collapsed Γ̃^{R}_{ij} mapping.

    Current placeholder behaviour: solve the full system, then collapse
    x_{R>} and x_{R<} to their (population-weighted) average and re-
    evaluate Γ̃^eo / ratio. Qualitatively in the right ballpark but
    **not** the paper's formula.
    """
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    coefs_list, sols = _solve_branch_tracked_sweep(
        omega_LR_GHz, T_sweep, label="renorm",
    )
    for i, (coefs, sol) in enumerate(zip(coefs_list, sols, strict=True)):
        # Collapse R> and R< via geometric mean (placeholder; paper
        # formula goes here). Keep μ_L distinct.
        x_R_common = float(np.sqrt(max(sol.x_Rgt * sol.x_Rlt, 0.0)))
        collapsed = M25SteadyState(
            p_0=sol.p_0, p_1=sol.p_1,
            x_L=sol.x_L, x_Rgt=x_R_common, x_Rlt=x_R_common,
            residual_inf_norm=sol.residual_inf_norm,
            n_function_evaluations=sol.n_function_evaluations,
        )
        gamma_eo = _gamma_eo(coefs, collapsed)
        Gamma_P[i] = _gamma_P(collapsed, gamma_eo)
        ratio[i] = _ratio_eo(gamma_eo)

    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        model=MODEL_RENORM,
        T_kelvin=T_sweep,
        Gamma_P_Hz=Gamma_P,
        ratio_eo_01_over_10=ratio,
    )


# ── orchestration ─────────────────────────────────────────────────────


def run() -> Fig4Result:
    """Compute all six (ω_LR, model) panels."""
    panels: dict[tuple[float, str], Fig4PanelResult] = {}
    for omega_LR_GHz in (0.5, 5.0):
        panels[(omega_LR_GHz, MODEL_FULL)] = _full_curve(omega_LR_GHz)
        panels[(omega_LR_GHz, MODEL_GLOBAL)] = _global_quasiequilibrium_curve(omega_LR_GHz)
        panels[(omega_LR_GHz, MODEL_RENORM)] = _renormalized_curve(omega_LR_GHz)
    return Fig4Result(panels=panels)


# ── baseline I/O ─────────────────────────────────────────────────────


def _baseline_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "marchegiani_2025"


def baseline_path() -> Path:
    """Output CSV path.

    Named ``m25_fig4_qpsim_native.csv`` to flag both the multi-stability
    scatter on panel (a) and the placeholder reduced-model curves
    (global / renormalized). Rename to ``m25_fig4_paper.csv`` once both
    gaps in the module docstring close.
    """
    return _baseline_dir() / "m25_fig4_qpsim_native.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig4Result, path: Path | None = None) -> Path:
    """Write all six (ω_LR, model) curves to a single CSV.

    Row layout: one row per (ω_LR, model, T) point with columns
    ``omega_LR_GHz, model, T_kelvin, Gamma_P_Hz, ratio_eo_01_over_10``.
    The flat layout keeps the reader simple and makes it easy to append
    more model reductions later without breaking format compatibility.
    """
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        # ``lineterminator='\n'`` avoids CRLF line endings (matches
        # fig3_chemical_potentials._write_panel_csv convention).
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 4 — qpsim-native characterization "
            "at paper topology"
        ])
        writer.writerow([
            f"# Delta_R_GHz={DELTA_R_OVER_H_GHZ:g}  "
            f"omega_10_GHz={OMEGA_10_OVER_H_GHZ:g}  "
            f"Gamma_ph_00_Hz={GAMMA_PH_00_HZ:g}"
        ])
        writer.writerow([
            "# omega_LR cases: 0.5, 5.0 GHz; "
            "models: full, global (placeholder), renorm (placeholder)"
        ])
        writer.writerow([
            "omega_LR_GHz", "model", "T_kelvin",
            "Gamma_P_Hz", "ratio_eo_01_over_10",
        ])
        for (omega_LR_GHz, model), panel in result.panels.items():
            for i in range(panel.T_kelvin.size):
                writer.writerow([
                    f"{omega_LR_GHz:g}",
                    model,
                    f"{panel.T_kelvin[i]:.17e}",
                    f"{panel.Gamma_P_Hz[i]:.17e}",
                    f"{panel.ratio_eo_01_over_10[i]:.17e}",
                ])
    return path


def read_baseline(path: Path | None = None) -> Fig4Result:
    """Read a pinned baseline CSV back into a :class:`Fig4Result`.

    Reconstructs the six (ω_LR, model) panels from the flat row-per-
    point layout written by :func:`write_baseline`.
    """
    if path is None:
        path = baseline_path()
    rows: list[tuple[float, str, float, float, float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("#") or first == "omega_LR_GHz":
                continue
            omega = float(line[0])
            model = line[1]
            T = float(line[2])
            G = float(line[3])
            r = float(line[4])
            rows.append((omega, model, T, G, r))

    by_key: dict[tuple[float, str], list[tuple[float, float, float]]] = {}
    for omega, model, T, G, r in rows:
        by_key.setdefault((omega, model), []).append((T, G, r))

    panels: dict[tuple[float, str], Fig4PanelResult] = {}
    for key, triples in by_key.items():
        triples.sort(key=lambda x: x[0])
        T_arr = np.array([t[0] for t in triples], dtype=float)
        G_arr = np.array([t[1] for t in triples], dtype=float)
        r_arr = np.array([t[2] for t in triples], dtype=float)
        panels[key] = Fig4PanelResult(
            omega_LR_GHz=key[0],
            model=key[1],
            T_kelvin=T_arr,
            Gamma_P_Hz=G_arr,
            ratio_eo_01_over_10=r_arr,
        )
    return Fig4Result(panels=panels)


def write_plot(result: Fig4Result, path: Path | None = None) -> Path:
    """Two-stack plot styled to match the published M25 Fig. 4.

    Paper styling (see ``m25_fig4_paper.png``):

    * Solid:     full model.
    * Dashed:    global quasiequilibrium reduction.
    * Dot-dashed: renormalized reduction.
    * Color:     dark purple for ω_LR/(2π) = 0.5 GHz,
                 teal for ω_LR/(2π) = 5 GHz.
    * Panel (a): Γ_P in kHz, linear axes, T in K.
    * Panel (b): ratio, linear axes, T in K, with a black dotted
                 baseline at zero (the paper's photon-only T → 0
                 reference; it sits near zero because Γ̃^ph_01 is
                 exponentially suppressed at the M25 caption ω_10).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # M25 Fig 4 colors (matches the paper figure, sampled by eye).
    color_for_omega = {0.5: "#3f1f5e", 5.0: "#3aa7a0"}
    # Paper line styles per reduced model.
    linestyle_for_model = {
        MODEL_FULL: "-",
        MODEL_GLOBAL: "--",
        MODEL_RENORM: "-.",
    }

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

    # Panel (a): Γ_P in kHz vs T. Linear axes, paper convention.
    for omega_LR_GHz in (0.5, 5.0):
        for model in (MODEL_FULL, MODEL_GLOBAL, MODEL_RENORM):
            panel = result.get(omega_LR_GHz, model)
            ax_a.plot(
                panel.T_kelvin,
                panel.Gamma_P_Hz / 1e3,           # Hz → kHz
                color=color_for_omega[omega_LR_GHz],
                ls=linestyle_for_model[model],
                lw=1.8,
            )
    ax_a.set_ylabel(r"$\Gamma_P$ [kHz]", fontsize=13)
    ax_a.set_xlim(0.0, 0.15)
    ax_a.set_ylim(bottom=0.0)
    ax_a.text(0.04, 0.92, "(a)", transform=ax_a.transAxes,
              fontsize=14, fontweight="bold")

    # Per-panel legends. Panel (a) carries the model-style legend.
    style_handles = [
        plt.Line2D([0], [0], color="black", ls=linestyle_for_model[m], lw=1.8,
                   label=MODEL_LABELS[m])
        for m in (MODEL_FULL, MODEL_GLOBAL, MODEL_RENORM)
    ]
    ax_a.legend(handles=style_handles, fontsize=10, loc="upper left",
                frameon=False, bbox_to_anchor=(0.18, 1.0))

    # Panel (b): excitation/relaxation ratio vs T. Linear axes.
    for omega_LR_GHz in (0.5, 5.0):
        for model in (MODEL_FULL, MODEL_GLOBAL, MODEL_RENORM):
            panel = result.get(omega_LR_GHz, model)
            ax_b.plot(
                panel.T_kelvin,
                panel.ratio_eo_01_over_10,
                color=color_for_omega[omega_LR_GHz],
                ls=linestyle_for_model[model],
                lw=1.8,
            )

    # Paper black dotted T → 0 photon-only baseline. Use the photon
    # coefficient ratio at the highest-T point of the small-asymmetry
    # case (where the photon drive is set deterministically to keep
    # Γ̃^ph_00 = 300 Hz; the ratio at low T is the relevant reference).
    T_dot = result.get(0.5, MODEL_FULL).T_kelvin
    coefs_low = _coefficients_at(0.5, float(T_dot[0]))
    if coefs_low.gamma_ph[1, 0] > 0:
        ratio_phot_only = coefs_low.gamma_ph[0, 1] / coefs_low.gamma_ph[1, 0]
        ax_b.plot(
            T_dot,
            np.full_like(T_dot, ratio_phot_only),
            color="black", ls=":", lw=1.2,
        )

    ax_b.set_xlabel(r"Temperature [K]", fontsize=13)
    ax_b.set_ylabel(r"$\Gamma^{\mathrm{eo}}_{01}/\Gamma^{\mathrm{eo}}_{10}$",
                    fontsize=13)
    ax_b.set_xlim(0.0, 0.15)
    ax_b.set_ylim(bottom=0.0)
    ax_b.text(0.04, 0.92, "(b)", transform=ax_b.transAxes,
              fontsize=14, fontweight="bold")

    # Panel (b) carries the ω_LR-color legend.
    color_handles = [
        plt.Line2D([0], [0], color=color_for_omega[w], lw=1.8,
                   label=f"{w:g}")
        for w in (0.5, 5.0)
    ]
    leg_b = ax_b.legend(
        handles=color_handles, fontsize=10, loc="upper left",
        title=r"$\omega_{LR}/2\pi$ [GHz]", frameon=False,
        bbox_to_anchor=(0.20, 1.0),
    )
    leg_b.get_title().set_fontsize(10)

    fig.suptitle(
        "Marchegiani 2025 Fig 4 — qpsim-native characterization at paper topology\n"
        rf"$\Delta_R/h={DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi)={OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}}={GAMMA_PH_00_HZ:g}$ Hz"
        "\n"
        "panel (a) carries multi-stability scatter; "
        "global / renormalized curves are placeholders",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("M25 Fig 4 — paper-target reproduction (qpsim-native; see docstring) ...")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz, T ∈ [{T_MIN_K*1e3:.0f}, {T_MAX_K*1e3:.0f}] mK")
    print("  ω_LR cases: 0.5, 5.0 GHz × models: full, global, renorm")
    print(
        "  ⚠ panel (a) carries multi-stability scatter from the max-x_L "
        "branch picker;\n"
        "    paper-grade smoothness needs proper bifurcation tracking.\n"
        "  ⚠ global / renormalized reduced-model curves are placeholders; "
        "see module docstring."
    )
    result = run()
    csv_p = write_baseline(result)
    pdf_p = write_plot(result)
    print(f"  Baseline: {csv_p}")
    print(f"  PDF plot: {pdf_p}")
    return csv_p, pdf_p


if __name__ == "__main__":
    generate_baseline()
