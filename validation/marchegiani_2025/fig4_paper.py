r"""M25 Fig. 4 paper-topology qpsim parity-rate regression.

Recreates the two-panel topology of the published Fig. 4 using qpsim
outputs. Broad trends are checked against manual paper readings; no
digitized paper points are used.
(Marchegiani & Catelani, *Commun. Phys.* **8**, 120 (2025); arXiv
2408.17218):

* **Panel (a).** Total parity-switching rate
  $\Gamma_P = p_0(\Gamma^{eo}_{01}+\Gamma^{eo}_{00})
  + p_1(\Gamma^{eo}_{10}+\Gamma^{eo}_{11})$ vs $T$ on linear axes.
* **Panel (b).** Excitation/relaxation ratio
  $\Gamma^{eo}_{01}/\Gamma^{eo}_{10}$ vs $T$, with the detailed-
  balance reference $e^{-\omega_{10}/T}$ as a black dotted line
  (main text, Sec. II.5).

Three model curves per the paper:

* **Full model** (solid) — the 4-unknown nonequilibrium steady state,
  tracked across the sweep by the branch-continuation driver
  (:func:`qpsim.services.rate_equation.solve_rate_equation_branch`)
  via :func:`fig3_chemical_potentials.solve_panel_branch_sweep`.
  Both ω_LR/(2π) ∈ {0.5, 5} GHz cases.

* **Global quasiequilibrium** (dashed) — the paper's reduction in
  which all quasiparticles share a single chemical potential and the
  bath temperature. Main text Sec. II.5: "the ratio between the
  quasiparticle densities is fixed by the gap asymmetry and the
  temperature (see last paragraph in Appendix F) and the overall
  density by the total generation rate, cf. Eq. (7)". The density
  ratios (arXiv Appendix A.1 last paragraph — published SI Note 1,
  context of Eqs. S6–S7):

      x_{R>}/x_{R<} = erfc[√(ω_LR/T)] / erf[√(ω_LR/T)]
      x_L / x_R     = √(Δ_R/Δ_L) · exp[−ω_LR/T],  x_R = x_{R<}+x_{R>}

  and the closure is qubit Eq. (3) plus the total-density equation
  obtained by summing Eqs. (4)–(6) (in which all tunneling and
  intraband terms cancel identically, leaving generation-vs-
  recombination balance). See :func:`_global_quasiequilibrium_state`.
  Both ω_LR cases.

* **Renormalized** (dot-dashed) — the same global-quasiequilibrium
  calculation with rescaled parameters, per the Fig. 4 caption: "For
  the dot-dashed curves, Γ^{ph}_{00} = 600 Hz and ω_LR/(2π) = 6 GHz"
  (photon rate ×2, gap asymmetry ×1.2 relative to the large-asymmetry
  case — main text Sec. II.5: "by repeating the calculation with
  different photon rate and the gap asymmetry … the global
  quasiequilibrium approach can describe reasonably well the results
  of the full model"). Plotted for the large-asymmetry family only,
  as in the paper.

M25 Fig 3/4 caption parameters (shared with
:mod:`fig3_chemical_potentials`):

    Δ_R/h         = 49.0 GHz
    ω_10/(2π)     = 5.5  GHz
    E_J/h         = 14.5 GHz
    E_C/h         = 0.290 GHz
    r^L = r^{R<}  = 6.25e6 Hz
    Γ̃^ee_10      = 100 kHz
    Γ̃^ph_00      = 300 Hz  (recalibrated at each T; 600 Hz for the
                             renormalized curves)

Usage::

    python -m validation.marchegiani_2025.fig4_paper
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import (
    M25Coefficients,
    M25SteadyState,
)
from scipy.special import erf, erfc

from validation.marchegiani_2025 import fig3_chemical_potentials as _chem
from validation.marchegiani_2025._artifact import (
    ArtifactValidationError,
    ProducerIdentity,
    capture_producer_identity,
    manifest_path_for,
    publish_bundle,
    read_table,
    require_staging_path,
    source_fingerprint,
    verified_bundle,
    write_table,
)
from validation.marchegiani_2025.fig3_chemical_potentials import (
    DELTA_R_OVER_H_GHZ,
    GAMMA_PH_00_HZ,
    NUM_T_POINTS,
    OMEGA_10_OVER_H_GHZ,
    T_MAX_K,
    T_MIN_K,
    _coefficients_at,
    solve_panel_branch_sweep,
)

_H_OVER_KB = 4.799243e-11   # K / Hz

# ── model labels ─────────────────────────────────────────────────────
MODEL_FULL = "full"
MODEL_GLOBAL = "global"
MODEL_RENORM = "renorm"
MODEL_LABELS: dict[str, str] = {
    MODEL_FULL: "Full model",
    MODEL_GLOBAL: "Global quasiequilibrium",
    MODEL_RENORM: "Renormalized",
}
LIVE_CERTIFICATE_SCOPE = "independently-reassembled-live-state"
SUMMARY_CERTIFICATE_SCOPE = (
    "producer-asserted-summary-only; full-state-and-closure-iterates-omitted"
)

# Fig. 4 caption parameters for the renormalized (dot-dashed) curves.
RENORM_GAMMA_PH_00_HZ = 600.0
RENORM_OMEGA_LR_GHZ = 6.0
_BUNDLE = "m25-fig4-paper"
_CERTIFICATE = {
    "assertion": (
        "summary observables only; full states and reduced closure iterates "
        "not persisted; no reader-side residual reconstruction"
    ),
    "kind": "producer_assertion",
    "scope": SUMMARY_CERTIFICATE_SCOPE,
}
_COLUMNS = (
    "omega_LR_GHz",
    "model",
    "T_kelvin",
    "Gamma_P_Hz",
    "ratio_eo_01_over_10",
)
_EXPECTED_KEYS = {
    (0.5, MODEL_FULL),
    (0.5, MODEL_GLOBAL),
    (5.0, MODEL_FULL),
    (5.0, MODEL_GLOBAL),
    (5.0, MODEL_RENORM),
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
    """Full paper-topology qpsim regression payload.

    Five panels: full + global quasiequilibrium for both ω_LR cases,
    plus the renormalized global-quasiequilibrium curve for the
    large-asymmetry family (the paper's dot-dashed curves).
    """

    panels: dict[tuple[float, str], Fig4PanelResult]
    certificate_scope: str = LIVE_CERTIFICATE_SCOPE

    def get(self, omega_LR_GHz: float, model: str) -> Fig4PanelResult:
        return self.panels[(omega_LR_GHz, model)]


# ── core observables on a single fixed-point solution ────────────────


def _gamma_eo(coefs: M25Coefficients, sol: M25SteadyState) -> np.ndarray:
    """Effective parity-flipping rate matrix Γ̃^eo at the fixed point.

    M25 main text below Eq. (3):

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
    """Total parity-switching rate Γ_P (M25 main text, Sec. II.5).

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


# ── full-model curve ─────────────────────────────────────────────────


def _full_curve(omega_LR_GHz: float) -> Fig4PanelResult:
    """Full-model Γ_P(T) and Γ̃^eo_{01}/Γ̃^eo_{10} sweep.

    Same branch-tracked steady states as the Fig 3 reproduction
    (:func:`fig3_chemical_potentials.solve_panel_branch_sweep`).
    """
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    sweep = solve_panel_branch_sweep(omega_LR_GHz, T_sweep)
    # Consume the driver's own per-T coefficient bundles — the exact
    # objects the states were converged on — rather than re-calling
    # the builder per T (structurally identical, but this makes the
    # observable ↔ steady-state pairing impossible to desynchronize).
    for i, (coefs, sol) in enumerate(
        zip(sweep.coefficients, sweep.states, strict=True)
    ):
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


# ── global-quasiequilibrium reduction (paper's dashed curves) ────────


def _global_quasiequilibrium_state(
    coefs: M25Coefficients,
    T_kelvin: float,
    omega_LR_kelvin: float,
) -> M25SteadyState:
    r"""Solve the M25 global-quasiequilibrium reduction at one T.

    Implements the closure prescribed in the M25 main text (Sec. II.5)
    and the last paragraph of arXiv Appendix A.1 (published SI Note 1,
    context of Eqs. S6–S7):

    * density ratios fixed by gap asymmetry and temperature —
      ``x_{R>}/x_{R<} = erfc[√(ω_LR/T)]/erf[√(ω_LR/T)]`` (local
      quasiequilibrium ``μ_{R>} = μ_{R<}``) and
      ``x_L/x_R = √(Δ_R/Δ_L)·e^{−ω_LR/T}`` (global quasiequilibrium
      ``μ_L = μ_R``), with ``x_R = x_{R<} + x_{R>}``;
    * overall density from the total generation rate (M25 Eq. 7,
      ``g^α = g^ph_α + g^pn_α``) balanced against recombination in
      the summed density equation (sum of Eqs. (4)–(6), in which all
      tunneling and intraband-relaxation terms cancel identically):

      .. math::

         g^L/\delta + g^{R>} + g^{R<}
           = (r^L/\delta)\,x_L^2 + r^{R>} x_{R>}^2 + r^{R<} x_{R<}^2
             + 2 r^{<>} x_{R<} x_{R>};

    * qubit steady state from Eq. (3),
      ``p_1 = (Γ^{eo}_{01}+Γ^{ee}_{01}) / (Γ^{eo}_{01}+Γ^{ee}_{01}
      + Γ^{eo}_{10}+Γ^{ee}_{10})`` with
      ``Γ^{eo}_{ij} = Γ^{ph}_{ij} + Σ_α Γ̃^α_{ij} x_α`` evaluated at
      the constrained densities.

    Because the generation rates are population-dependent (per-state
    ``g^{ph}_α`` arrays) and ``Γ^{eo}`` depends on the densities, the
    two scalar equations are closed by deterministic fixed-point
    iteration on ``p_1`` (converges in a handful of steps; the
    coupling is weak since ``Γ^{ph}_{00} ≈ Γ^{ph}_{11}``).
    """
    T = float(T_kelvin)
    sqrt_ratio = float(np.sqrt(omega_LR_kelvin / T))
    # x_{R>} = ρ x_{R<}; x_L = c_L x_{R<}.
    rho = float(erfc(sqrt_ratio) / erf(sqrt_ratio))
    c_L = float(
        np.sqrt(coefs.delta) * np.exp(-omega_LR_kelvin / T) * (1.0 + rho)
    )
    # Quadratic recombination coefficient of the summed density
    # equation, in units of x_{R<}²:
    A = (
        (coefs.r_L / coefs.delta) * c_L * c_L
        + coefs.r_Rgt * rho * rho
        + coefs.r_Rlt
        + 2.0 * coefs.r_cross * rho
    )

    p_1 = 0.0
    s = 0.0
    converged = False
    p_1_step = float("inf")
    for _ in range(200):
        p_0 = 1.0 - p_1
        # Total generation (M25 Eq. 7; per-state photon pieces at p).
        g_L = (
            coefs.g_L
            + p_0 * coefs.g_ph_L_per_state[0] + p_1 * coefs.g_ph_L_per_state[1]
        )
        g_Rgt = (
            coefs.g_Rgt
            + p_0 * coefs.g_ph_Rgt_per_state[0] + p_1 * coefs.g_ph_Rgt_per_state[1]
        )
        g_Rlt = (
            coefs.g_Rlt
            + p_0 * coefs.g_ph_Rlt_per_state[0] + p_1 * coefs.g_ph_Rlt_per_state[1]
        )
        g_total = g_L / coefs.delta + g_Rgt + g_Rlt
        s = float(np.sqrt(g_total / A))

        gamma_eo = (
            coefs.gamma_ph
            + coefs.gammas_L * (c_L * s)
            + coefs.gammas_Rgt * (rho * s)
            + coefs.gammas_Rlt * s
        )
        up = float(gamma_eo[0, 1] + coefs.gamma_ee[0, 1])
        down = float(gamma_eo[1, 0] + coefs.gamma_ee[1, 0])
        p_1_new = up / (up + down) if up + down > 0.0 else 0.0
        p_1_step = abs(p_1_new - p_1)
        if p_1_step <= 1e-15 * max(p_1_new, 1e-30):
            p_1 = p_1_new
            converged = True
            break
        p_1 = p_1_new

    if not converged:
        raise RuntimeError(
            "Global-quasiequilibrium fixed-point iteration on p_1 did not "
            f"converge in 200 iterations (last |Δp_1| = {p_1_step:.3e}). "
            "The p_1 ↔ density coupling is not weak for these parameters; "
            "the reduction's closed-form assumptions need re-examination."
        )

    return M25SteadyState(
        p_0=1.0 - p_1,
        p_1=p_1,
        x_L=c_L * s,
        x_Rgt=rho * s,
        x_Rlt=s,
        # This state solves the REDUCED global-QE closure, not the full
        # 4-unknown residual, so a full-model residual norm is not
        # applicable — NaN, not a fake 0.0 (deep-review finding 3).
        residual_inf_norm=float("nan"),
        n_function_evaluations=0,
    )


def _global_curve(
    omega_LR_GHz: float,
    *,
    model: str = MODEL_GLOBAL,
    coefficients_omega_LR_GHz: float | None = None,
    gamma_ph_00_Hz: float = GAMMA_PH_00_HZ,
) -> Fig4PanelResult:
    """Global-quasiequilibrium Γ_P / ratio sweep.

    ``omega_LR_GHz`` labels the output family (plot color / CSV key);
    ``coefficients_omega_LR_GHz`` and ``gamma_ph_00_Hz`` are the
    parameters actually used to build the coefficient bundle — they
    differ from the label only for the renormalized curves (Fig. 4
    caption: Γ^{ph}_{00} = 600 Hz, ω_LR/(2π) = 6 GHz).
    """
    omega_used = (
        omega_LR_GHz if coefficients_omega_LR_GHz is None
        else coefficients_omega_LR_GHz
    )
    omega_LR_K = omega_used * 1e9 * _H_OVER_KB
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    n = T_sweep.size
    Gamma_P = np.full(n, np.nan)
    ratio = np.full(n, np.nan)
    for i, T_K in enumerate(T_sweep):
        coefs = _coefficients_at(
            omega_used, float(T_K), gamma_ph_00_Hz=gamma_ph_00_Hz,
        )
        sol = _global_quasiequilibrium_state(coefs, float(T_K), omega_LR_K)
        gamma_eo = _gamma_eo(coefs, sol)
        Gamma_P[i] = _gamma_P(sol, gamma_eo)
        ratio[i] = _ratio_eo(gamma_eo)

    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        model=model,
        T_kelvin=T_sweep,
        Gamma_P_Hz=Gamma_P,
        ratio_eo_01_over_10=ratio,
    )


def _renormalized_curve(omega_LR_GHz: float) -> Fig4PanelResult:
    """Renormalized global-quasiequilibrium curve (paper dot-dashed).

    Fig. 4 caption: "For the dot-dashed curves, Γ^{ph}_{00} = 600 Hz
    and ω_LR/(2π) = 6 GHz" — the global-quasiequilibrium calculation
    repeated with the photon rate doubled and the gap asymmetry
    rescaled ×1.2, chosen by the authors so the reduction mimics the
    large-asymmetry full model on Γ_P(T).
    """
    return _global_curve(
        omega_LR_GHz,
        model=MODEL_RENORM,
        coefficients_omega_LR_GHz=RENORM_OMEGA_LR_GHZ,
        gamma_ph_00_Hz=RENORM_GAMMA_PH_00_HZ,
    )


# ── orchestration ─────────────────────────────────────────────────────


def run() -> Fig4Result:
    """Compute the five (ω_LR, model) panels of the published Fig. 4."""
    panels: dict[tuple[float, str], Fig4PanelResult] = {}
    for omega_LR_GHz in (0.5, 5.0):
        panels[(omega_LR_GHz, MODEL_FULL)] = _full_curve(omega_LR_GHz)
        panels[(omega_LR_GHz, MODEL_GLOBAL)] = _global_curve(omega_LR_GHz)
    # Dot-dashed renormalized curves exist for the large-asymmetry
    # family only (paper Fig. 4 caption).
    panels[(5.0, MODEL_RENORM)] = _renormalized_curve(5.0)
    return Fig4Result(panels=panels)


# ── baseline I/O ─────────────────────────────────────────────────────


def _baseline_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "marchegiani_2025"


def baseline_path() -> Path:
    """Output CSV path (paper-topology qpsim regression)."""
    return _baseline_dir() / "m25_fig4_paper.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def manifest_path() -> Path:
    return manifest_path_for(baseline_path())


def _artifact_config() -> dict[str, object]:
    return {
        "models": [
            [0.5, MODEL_FULL],
            [0.5, MODEL_GLOBAL],
            [5.0, MODEL_FULL],
            [5.0, MODEL_GLOBAL],
            [5.0, MODEL_RENORM],
        ],
        "renormalized": {
            "Gamma_ph_00_Hz": RENORM_GAMMA_PH_00_HZ,
            "omega_LR_over_h_GHz": RENORM_OMEGA_LR_GHZ,
        },
        "source_branch_config": _chem._artifact_config(),
    }


def artifact_fingerprint() -> dict[str, object]:
    return source_fingerprint(
        bundle=_BUNDLE,
        config=_artifact_config(),
        producer_module=Path(__file__),
        extra_validation_modules=(Path(_chem.__file__),),
    )


def _member_paths() -> dict[str, Path]:
    return {
        baseline_path().name: baseline_path(),
        plot_path().name: plot_path(),
    }


def _expected_members() -> dict[str, str]:
    return {baseline_path().name: "csv", plot_path().name: "pdf"}


def write_baseline(result: Fig4Result, path: Path | None = None) -> Path:
    """Write all (ω_LR, model) curves to a single CSV.

    Row layout: one row per (ω_LR, model, T) point with columns
    ``omega_LR_GHz, model, T_kelvin, Gamma_P_Hz, ratio_eo_01_over_10``.
    """
    if path is None:
        path = baseline_path()
    require_staging_path(path, baseline_path(), kind="CSV")
    if result.certificate_scope != LIVE_CERTIFICATE_SCOPE:
        raise ArtifactValidationError(
            "M25 Fig. 4 publication requires live model results; a "
            "summary readback carries producer assertions only."
        )
    if set(result.panels) != _EXPECTED_KEYS:
        raise ArtifactValidationError(
            "M25 Fig. 4 result does not contain the exact model/panel set."
        )
    rows: list[list[str | float | int]] = []
    for omega_LR_GHz, model in sorted(_EXPECTED_KEYS):
        panel = result.panels[(omega_LR_GHz, model)]
        rows.extend(
            [
                [
                    omega_LR_GHz,
                    model,
                    panel.T_kelvin[index],
                    panel.Gamma_P_Hz[index],
                    panel.ratio_eo_01_over_10[index],
                ]
                for index in range(panel.T_kelvin.size)
            ]
        )
    return write_table(
        path,
        bundle=_BUNDLE,
        role="all_models_summary",
        config=_artifact_config(),
        columns=_COLUMNS,
        rows=rows,
        certificate=_CERTIFICATE,
    )


def read_baseline(
    path: Path | None = None,
    *,
    accept_producer_certificate_claims: bool = False,
) -> Fig4Result:
    """Read summary data only after explicit producer-claim acceptance."""
    if not accept_producer_certificate_claims:
        raise ArtifactValidationError(
            "M25 Fig. 4 summary omits full states and reduced closure "
            "iterates, so its certificate is a producer assertion only. "
            "Pass accept_producer_certificate_claims=True to read the "
            "summary, or regenerate for independently reassembled evidence."
        )
    if path is None:
        path = baseline_path()
    canonical = path.resolve() == baseline_path().resolve()

    def _decode() -> Fig4Result:
        payload = read_table(
            path,
            bundle=_BUNDLE,
            role="all_models_summary",
            config=_artifact_config(),
            columns=_COLUMNS,
            certificate=_CERTIFICATE,
        )
        by_key: dict[tuple[float, str], list[tuple[float, float, float]]] = {}
        for row in payload.rows:
            try:
                omega = float(row[0])
                model = row[1]
                triple = (float(row[2]), float(row[3]), float(row[4]))
            except (IndexError, ValueError) as exc:
                raise ArtifactValidationError(
                    f"M25 Fig. 4 table {path} contains malformed rows."
                ) from exc
            key = (omega, model)
            if key not in _EXPECTED_KEYS:
                raise ArtifactValidationError(
                    f"M25 Fig. 4 table {path} contains an unknown model key."
                )
            by_key.setdefault(key, []).append(triple)
        if set(by_key) != _EXPECTED_KEYS:
            raise ArtifactValidationError(
                f"M25 Fig. 4 table {path} omits a required model curve."
            )
        expected_T = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
        panels: dict[tuple[float, str], Fig4PanelResult] = {}
        for key, triples in by_key.items():
            if len(triples) != NUM_T_POINTS:
                raise ArtifactValidationError(
                    f"M25 Fig. 4 table {path} has duplicate/missing points."
                )
            data = np.array(triples, dtype=float)
            if (
                not np.all(np.isfinite(data))
                or not np.array_equal(data[:, 0], expected_T)
                or np.any(data[:, 1:] < 0.0)
            ):
                raise ArtifactValidationError(
                    f"M25 Fig. 4 table {path} has invalid grid/observables."
                )
            panels[key] = Fig4PanelResult(
                omega_LR_GHz=key[0],
                model=key[1],
                T_kelvin=data[:, 0],
                Gamma_P_Hz=data[:, 1],
                ratio_eo_01_over_10=data[:, 2],
            )
        return Fig4Result(
            panels=panels,
            certificate_scope=SUMMARY_CERTIFICATE_SCOPE,
        )

    if not canonical:
        return _decode()
    with verified_bundle(
        manifest_path=manifest_path(),
        bundle=_BUNDLE,
        fingerprint=artifact_fingerprint(),
        expected_members=_expected_members(),
        member_paths=_member_paths(),
    ):
        return _decode()


def write_plot(result: Fig4Result, path: Path | None = None) -> Path:
    """Two-stack plot styled to match the published M25 Fig. 4.

    * Solid: full model. Dashed: global quasiequilibrium.
      Dot-dashed: renormalized (teal / large-asymmetry family only).
    * Color: dark purple for ω_LR/(2π) = 0.5 GHz, teal for 5 GHz.
    * Panel (a): Γ_P in kHz, linear axes, T in K.
    * Panel (b): ratio, linear axes, with the black dotted
      detailed-balance reference ``exp(−ω_10/T)`` (main text: "the
      excitation/relaxation ratio deviates from the detailed balance
      expectation Γ^{eo}_{01}/Γ^{eo}_{10} = exp(−ω_10/T) (dotted
      line)").
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    if path is None:
        path = plot_path()
    require_staging_path(path, plot_path(), kind="PDF")
    path.parent.mkdir(parents=True, exist_ok=True)

    # M25 Fig 4 colors (matches the paper figure, sampled by eye).
    color_for_omega = {0.5: "#3f1f5e", 5.0: "#3aa7a0"}
    linestyle_for_model = {
        MODEL_FULL: "-",
        MODEL_GLOBAL: "--",
        MODEL_RENORM: "-.",
    }

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(7, 9), sharex=True)

    # Panel (a): Γ_P in kHz vs T. Linear axes, paper convention.
    for (omega_LR_GHz, model), panel in result.panels.items():
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

    # Panel (a) carries the model-style legend.
    style_handles = [
        Line2D([0], [0], color="black", ls=linestyle_for_model[m], lw=1.8,
               label=MODEL_LABELS[m])
        for m in (MODEL_FULL, MODEL_GLOBAL, MODEL_RENORM)
    ]
    ax_a.legend(handles=style_handles, fontsize=10, loc="upper left",
                frameon=False, bbox_to_anchor=(0.18, 1.0))

    # Panel (b): excitation/relaxation ratio vs T. Linear axes.
    for (omega_LR_GHz, model), panel in result.panels.items():
        ax_b.plot(
            panel.T_kelvin,
            panel.ratio_eo_01_over_10,
            color=color_for_omega[omega_LR_GHz],
            ls=linestyle_for_model[model],
            lw=1.8,
        )

    # Black dotted detailed-balance reference exp(−ω_10/T).
    T_dot = np.linspace(T_MIN_K, T_MAX_K, 200)
    omega_10_K = OMEGA_10_OVER_H_GHZ * 1e9 * _H_OVER_KB
    ax_b.plot(
        T_dot, np.exp(-omega_10_K / T_dot),
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
        Line2D([0], [0], color=color_for_omega[w], lw=1.8,
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
        "M25 Fig. 4 topology — qpsim parity-rate regression\n"
        "(manual broad paper anchors; no digitized paper points)\n"
        rf"$\Delta_R/h={DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi)={OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}}={GAMMA_PH_00_HZ:g}$ Hz "
        rf"(renorm: {RENORM_GAMMA_PH_00_HZ:g} Hz, "
        rf"$\omega_{{LR}}/2\pi={RENORM_OMEGA_LR_GHZ:g}$ GHz)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path)
    plt.close(fig)
    return path


def _validate_staged_bundle(stages: Mapping[Path, Path]) -> None:
    read_baseline(
        stages[baseline_path()],
        accept_producer_certificate_claims=True,
    )


def generate_baseline() -> tuple[Path, Path]:
    fingerprint = artifact_fingerprint()
    producer: ProducerIdentity = capture_producer_identity(fingerprint)
    print("M25 Fig 4 — paper-topology qpsim regression ...")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz, T ∈ [{T_MIN_K*1e3:.0f}, {T_MAX_K*1e3:.0f}] mK")
    print("  models: full + global QE (0.5, 5.0 GHz); renorm "
          f"(5.0 GHz family; Γ̃^ph_00 = {RENORM_GAMMA_PH_00_HZ:g} Hz, "
          f"ω_LR/2π = {RENORM_OMEGA_LR_GHZ:g} GHz)")
    result = run()
    csv_p, pdf_p, _manifest = publish_bundle(
        manifest_path=manifest_path(),
        bundle=_BUNDLE,
        producer=producer,
        current_fingerprint=artifact_fingerprint,
        members={
            baseline_path(): (
                "csv",
                lambda path: write_baseline(result, path),
            ),
            plot_path(): ("pdf", lambda path: write_plot(result, path)),
        },
        validate_staged=_validate_staged_bundle,
    )
    print(f"  Baseline: {csv_p}")
    print(f"  PDF plot: {pdf_p}")
    return csv_p, pdf_p


if __name__ == "__main__":
    generate_baseline()
