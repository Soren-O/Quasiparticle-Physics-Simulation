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
so both figures consume the same steady-state branch and remain
cross-consistent.

Usage::

    python -m validation.marchegiani_2025.fig4_parity_rates
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import M25Coefficients, M25SteadyState

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
    solve_panel_branch_sweep,
)

_BUNDLE = "m25-fig4-parity-rates"
LIVE_CERTIFICATE_SCOPE = "independently-reassembled-live-state"
SUMMARY_CERTIFICATE_SCOPE = (
    "producer-asserted-summary-only; full-state-omitted"
)
_CERTIFICATE = {
    "assertion": (
        "summary observables derived from source-gated M25 branch states; "
        "raw states not persisted"
    ),
    "kind": "producer_assertion",
    "scope": SUMMARY_CERTIFICATE_SCOPE,
}
_COLUMNS = ("T_kelvin", "Gamma_P_Hz", "ratio_eo_01_over_10")


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
    certificate_scope: str = LIVE_CERTIFICATE_SCOPE


def _gamma_eo(coefs: M25Coefficients, sol: M25SteadyState) -> np.ndarray:
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


def manifest_path() -> Path:
    return manifest_path_for(plot_path())


def _artifact_config() -> dict[str, object]:
    return {
        "source_branch_config": _chem._artifact_config(),
        "summary": ["Gamma_P_Hz", "ratio_eo_01_over_10"],
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
        baseline_path_a().name: baseline_path_a(),
        baseline_path_b().name: baseline_path_b(),
        plot_path().name: plot_path(),
    }


def _expected_members() -> dict[str, str]:
    return {
        baseline_path_a().name: "csv",
        baseline_path_b().name: "csv",
        plot_path().name: "pdf",
    }


def _panel_role(omega_LR_GHz: float) -> str:
    if omega_LR_GHz == 0.5:
        return "panel_a"
    if omega_LR_GHz == 5.0:
        return "panel_b"
    raise ArtifactValidationError(
        f"Unexpected M25 Fig. 4 parity panel frequency {omega_LR_GHz!r}."
    )


def _write_panel_csv(panel: Fig4PanelResult, path: Path) -> Path:
    return write_table(
        path,
        bundle=_BUNDLE,
        role=_panel_role(panel.omega_LR_GHz),
        config=_artifact_config(),
        columns=_COLUMNS,
        rows=list(
            zip(
                panel.T_kelvin,
                panel.Gamma_P_Hz,
                panel.ratio_eo_01_over_10,
                strict=True,
            )
        ),
        certificate=_CERTIFICATE,
    )


def _read_panel_csv(path: Path, omega_LR_GHz: float) -> Fig4PanelResult:
    payload = read_table(
        path,
        bundle=_BUNDLE,
        role=_panel_role(omega_LR_GHz),
        config=_artifact_config(),
        columns=_COLUMNS,
        certificate=_CERTIFICATE,
    )
    try:
        data = np.array(payload.rows, dtype=float)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"M25 parity table {path} contains nonnumeric cells."
        ) from exc
    if data.shape != (NUM_T_POINTS, len(_COLUMNS)) or not np.all(
        np.isfinite(data)
    ):
        raise ArtifactValidationError(
            f"M25 parity table {path} has invalid shape/nonfinite data."
        )
    if not np.array_equal(
        data[:, 0], np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
    ):
        raise ArtifactValidationError(
            f"M25 parity table {path} has the wrong temperature grid."
        )
    if np.any(data[:, 1:] < 0.0):
        raise ArtifactValidationError(
            f"M25 parity table {path} contains negative rates/ratios."
        )
    return Fig4PanelResult(
        omega_LR_GHz=omega_LR_GHz,
        T_kelvin=data[:, 0],
        Gamma_P_Hz=data[:, 1],
        ratio_eo_01_over_10=data[:, 2],
    )


def write_baseline(
    result: Fig4Result,
    paths: tuple[Path, Path] | None = None,
) -> tuple[Path, Path]:
    if paths is None:
        paths = (baseline_path_a(), baseline_path_b())
    if result.certificate_scope != LIVE_CERTIFICATE_SCOPE:
        raise ArtifactValidationError(
            "M25 Fig. 4 parity publication requires live model results; "
            "a summary readback carries producer assertions only."
        )
    for path, canonical in zip(
        paths,
        (baseline_path_a(), baseline_path_b()),
        strict=True,
    ):
        require_staging_path(path, canonical, kind="CSV")
    return (
        _write_panel_csv(result.panel_a, paths[0]),
        _write_panel_csv(result.panel_b, paths[1]),
    )


def read_baseline(
    *,
    accept_producer_certificate_claims: bool = False,
) -> Fig4Result:
    """Read summaries only after explicit producer-claim acceptance."""
    if not accept_producer_certificate_claims:
        raise ArtifactValidationError(
            "M25 Fig. 4 parity summaries omit the branch states, so their "
            "certificate is a producer assertion only. Pass "
            "accept_producer_certificate_claims=True to read the summaries, "
            "or regenerate for independently reassembled evidence."
        )
    with verified_bundle(
        manifest_path=manifest_path(),
        bundle=_BUNDLE,
        fingerprint=artifact_fingerprint(),
        expected_members=_expected_members(),
        member_paths=_member_paths(),
    ):
        return Fig4Result(
            panel_a=_read_panel_csv(baseline_path_a(), 0.5),
            panel_b=_read_panel_csv(baseline_path_b(), 5.0),
            certificate_scope=SUMMARY_CERTIFICATE_SCOPE,
        )


def write_plot(result: Fig4Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    require_staging_path(path, plot_path(), kind="PDF")
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
        "M25 Fig. 4 topology — qpsim parity-observable regression\n"
        "(manual broad paper anchors; no digitized paper points)\n"
        rf"$\Delta_R/h = {DELTA_R_OVER_H_GHZ:g}$ GHz, "
        rf"$\omega_{{10}}/(2\pi) = {OMEGA_10_OVER_H_GHZ:g}$ GHz, "
        rf"$\widetilde\Gamma^\mathrm{{ph}}_{{00}} = {GAMMA_PH_00_HZ:g}$ Hz",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90))
    fig.savefig(path)
    plt.close(fig)
    return path


def _validate_staged_bundle(stages: Mapping[Path, Path]) -> None:
    _read_panel_csv(stages[baseline_path_a()], 0.5)
    _read_panel_csv(stages[baseline_path_b()], 5.0)


def generate_baseline() -> tuple[Path, Path, Path]:
    fingerprint = artifact_fingerprint()
    producer: ProducerIdentity = capture_producer_identity(fingerprint)
    print("M25 Fig 4 — Γ_P and Γ^eo_01/Γ^eo_10 vs T")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_GHZ} GHz, ω_10/(2π) = {OMEGA_10_OVER_H_GHZ} GHz")
    print(f"  Γ̃^ph_00 = {GAMMA_PH_00_HZ} Hz, T ∈ [{T_MIN_K*1e3:.0f}, {T_MAX_K*1e3:.0f}] mK")
    print("  Panel a (ω_LR = 0.5 GHz) ...")
    panel_a = _run_panel(0.5)
    print("  Panel b (ω_LR = 5.0 GHz) ...")
    panel_b = _run_panel(5.0)
    result = Fig4Result(panel_a=panel_a, panel_b=panel_b)
    csv_a, csv_b, pdf, _manifest = publish_bundle(
        manifest_path=manifest_path(),
        bundle=_BUNDLE,
        producer=producer,
        current_fingerprint=artifact_fingerprint,
        members={
            baseline_path_a(): (
                "csv",
                lambda path: _write_panel_csv(result.panel_a, path),
            ),
            baseline_path_b(): (
                "csv",
                lambda path: _write_panel_csv(result.panel_b, path),
            ),
            plot_path(): ("pdf", lambda path: write_plot(result, path)),
        },
        validate_staged=_validate_staged_bundle,
    )
    print(f"  Baselines: {csv_a.name}, {csv_b.name}")
    print(f"  PDF plot:  {pdf.name}")
    return csv_a, csv_b, pdf


if __name__ == "__main__":
    generate_baseline()
