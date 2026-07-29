r"""Marchegiani 2025 Fig 3 — analytic crossover temperature T̄(g^ph_R).

Direct formula transcription and self-regression of M25 Eq. 8 — the
Lambert-W crossover that separates the low-``T`` nonequilibrium regime
from the local-quasiequilibrium regime in a gap-asymmetric transmon
junction. The reader recomputes the same closed form from the persisted
axis; this module does not independently solve the full rate equations
or validate Eq. 8 against them.

The *paper* states that Eq. 8 agrees "within a few percent" with its
numerical results. That paper-reported comparison is context, not an
independent result produced by this module.

Reproduces the Fig 3 caption parameter set (Δ_R/h = 49 GHz,
r^L = r^{R<} = 6.25 MHz) and sweeps the photon-generation rate
``g^\mathrm{ph}_R`` from ``1e-25`` Hz to ``1e6`` Hz. The output is the
monotonic T̄(g^ph_R) curve, annotated with a broad manual reading of the
published dashed line near 150 mK. No digitized paper points are used.

Usage::

    python -m validation.marchegiani_2025.fig3_crossover_temperature
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import crossover_temperature_kelvin

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

# ── M25 Fig 3 caption parameters ─────────────────────────────────────
_H_J_S = 6.62607015e-34
_KB_J_PER_K = 1.380649e-23

DELTA_R_OVER_H_HZ = 49.0e9           # Δ_R/h = 49 GHz
DELTA_R_KELVIN = DELTA_R_OVER_H_HZ * _H_J_S / _KB_J_PER_K  # ≈ 2.352 K
R_RLT_RATE_HZ = 6.25e6                # r^{R<} = 6.25 MHz

# Sweep span: the only T̄ value broadly readable from the published M25
# Fig. 3 is the ≈150 mK dashed-line position. Inverting the same Eq. 8
# transcription at 150 mK with the caption parameters gives
# g^{ph}_R ~ 5e−8 Hz; this is not an independent numerical crossing.
# The sweep extends far below that point to expose the Lambert-W tail
# and up to the Δ_R asymptote at high g^{ph}. An earlier revision quoted
# a "paper ≈70 mK" anchor; that value is not in the paper and was removed.
G_PHOTON_MIN_HZ = 1.0e-25
G_PHOTON_MAX_HZ = 1.0e6
NUM_POINTS = 63  # ~2 points per decade
_GRID_REASSEMBLY_RTOL = 1e-15
_BUNDLE = "m25-fig3-crossover-temperature"
_CERTIFICATE = {
    "kind": "formula_transcription_self_regression",
    "metric_version": "m25-eq8-lambert-w-v1",
}
_COLUMNS = ("g_photon_R_Hz", "T_bar_kelvin")


@dataclass(frozen=True)
class Fig3Result:
    g_photon_R_Hz: np.ndarray
    T_bar_kelvin: np.ndarray


def run() -> Fig3Result:
    g_sweep = np.logspace(
        np.log10(G_PHOTON_MIN_HZ),
        np.log10(G_PHOTON_MAX_HZ),
        NUM_POINTS,
    )
    T_bar = np.array(
        [
            crossover_temperature_kelvin(
                Delta_R_kelvin=DELTA_R_KELVIN,
                r_Rlt_rate_Hz=R_RLT_RATE_HZ,
                g_photon_R_rate_Hz=float(g),
            )
            for g in g_sweep
        ],
        dtype=float,
    )
    return Fig3Result(g_photon_R_Hz=g_sweep, T_bar_kelvin=T_bar)


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "marchegiani_2025"
        / "m25_fig3_crossover_temperature.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def manifest_path() -> Path:
    return manifest_path_for(baseline_path())


def _artifact_config() -> dict[str, object]:
    return {
        "Delta_R_kelvin": DELTA_R_KELVIN,
        "Delta_R_over_h_Hz": DELTA_R_OVER_H_HZ,
        # Fingerprint the exact semantic inputs, not libm-produced samples:
        # identical logspace requests can differ in their last bits across
        # hosts even though the numerical campaign is unchanged.
        "g_photon_R_grid": {
            "minimum_Hz": G_PHOTON_MIN_HZ,
            "maximum_Hz": G_PHOTON_MAX_HZ,
            "num_points": NUM_POINTS,
            "spacing": "log10",
        },
        "r_Rlt_rate_Hz": R_RLT_RATE_HZ,
    }


def artifact_fingerprint() -> dict[str, object]:
    return source_fingerprint(
        bundle=_BUNDLE,
        config=_artifact_config(),
        producer_module=Path(__file__),
    )


def _member_paths() -> dict[str, Path]:
    return {
        baseline_path().name: baseline_path(),
        plot_path().name: plot_path(),
    }


def _expected_members() -> dict[str, str]:
    return {baseline_path().name: "csv", plot_path().name: "pdf"}


def write_baseline(result: Fig3Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    require_staging_path(path, baseline_path(), kind="CSV")
    return write_table(
        path,
        bundle=_BUNDLE,
        role="closed_form_sweep",
        config=_artifact_config(),
        columns=_COLUMNS,
        rows=list(
            zip(
                result.g_photon_R_Hz,
                result.T_bar_kelvin,
                strict=True,
            )
        ),
        certificate=_CERTIFICATE,
    )


def read_baseline(path: Path | None = None) -> Fig3Result:
    if path is None:
        path = baseline_path()
    canonical = path.resolve() == baseline_path().resolve()

    def _decode() -> Fig3Result:
        payload = read_table(
            path,
            bundle=_BUNDLE,
            role="closed_form_sweep",
            config=_artifact_config(),
            columns=_COLUMNS,
            certificate=_CERTIFICATE,
        )
        try:
            data = np.array(payload.rows, dtype=float)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"M25 crossover table {path} contains nonnumeric cells."
            ) from exc
        if data.shape != (NUM_POINTS, len(_COLUMNS)) or not np.all(
            np.isfinite(data)
        ):
            raise ArtifactValidationError(
                f"M25 crossover table {path} has invalid shape/nonfinite data."
            )
        expected_g = np.logspace(
            np.log10(G_PHOTON_MIN_HZ),
            np.log10(G_PHOTON_MAX_HZ),
            NUM_POINTS,
        )
        if not np.allclose(
            data[:, 0],
            expected_g,
            rtol=_GRID_REASSEMBLY_RTOL,
            atol=0.0,
        ):
            raise ArtifactValidationError(
                f"M25 crossover table {path} has the wrong generation grid."
            )
        # Reassemble Eq. 8 from the authenticated persisted axis.  Reusing
        # the host-generated `expected_g` here would reintroduce the same
        # one-ULP cross-microarchitecture mismatch the axis check permits.
        expected_T = np.array(
            [
                crossover_temperature_kelvin(
                    Delta_R_kelvin=DELTA_R_KELVIN,
                    r_Rlt_rate_Hz=R_RLT_RATE_HZ,
                    g_photon_R_rate_Hz=float(g_photon),
                )
                for g_photon in data[:, 0]
            ]
        )
        if not np.allclose(data[:, 1], expected_T, rtol=2e-13, atol=0.0):
            raise ArtifactValidationError(
                f"M25 crossover table {path} fails closed-form reassembly."
            )
        return Fig3Result(g_photon_R_Hz=data[:, 0], T_bar_kelvin=data[:, 1])

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


def write_plot(result: Fig3Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    require_staging_path(path, plot_path(), kind="PDF")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogx(
        result.g_photon_R_Hz, result.T_bar_kelvin * 1e3,
        "o-", lw=1.5, ms=4, color="tab:purple",
        label=r"$\bar T = 2\Delta_R / W(4\pi r^{R<}/g^\mathrm{ph}_R)$",
    )

    # Annotate the one broad T̄ value manually readable from the published
    # figure: the ≈150 mK dashed-line position. The curve itself is the same
    # Eq. 8 transcription, not an independently located numerical crossing.
    ax.axhline(150.0, color="tab:olive", ls=":", lw=1.0, alpha=0.7,
               label=r"M25 Fig 3 dashed lines: $\bar T \approx 150$ mK "
                     r"(broad manual reading)")

    # Δ_R upper bound (T̄ cannot exceed Δ_R since W > 0 for positive args).
    ax.axhline(DELTA_R_KELVIN * 1e3, color="red", ls="--", lw=0.8, alpha=0.4,
               label=rf"$\Delta_R = {DELTA_R_KELVIN * 1e3:.0f}$ mK")

    ax.set_xlabel(r"$g^\mathrm{ph}_R$ [Hz] (photon-assisted PB rate on R)",
                  fontsize=12)
    ax.set_ylabel(r"$\bar T$ [mK]", fontsize=12)
    ax.set_title(
        "M25 Eq. 8 transcription: Lambert-W self-regression\n"
        "(one manual broad paper anchor; no digitized paper points)\n"
        rf"Fig 3 params: $\Delta_R/h={DELTA_R_OVER_H_HZ/1e9:.0f}$ GHz, "
        rf"$r^{{R<}}={R_RLT_RATE_HZ/1e6:.2f}$ MHz",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _validate_staged_bundle(stages: Mapping[Path, Path]) -> None:
    read_baseline(stages[baseline_path()])


def generate_baseline() -> tuple[Path, Path]:
    fingerprint = artifact_fingerprint()
    producer: ProducerIdentity = capture_producer_identity(fingerprint)
    print("M25 Fig 3 — Eq. 8 Lambert-W transcription/self-regression")
    print(f"  Δ_R/h = {DELTA_R_OVER_H_HZ/1e9:.1f} GHz → Δ_R = {DELTA_R_KELVIN:.4f} K")
    print(f"  r^{{R<}} = {R_RLT_RATE_HZ/1e6:.2f} MHz")
    print(
        f"  g^ph_R: {NUM_POINTS} pts, {G_PHOTON_MIN_HZ:.0e} → "
        f"{G_PHOTON_MAX_HZ:.0e} Hz (logspaced)"
    )
    result = run()
    print(
        f"  T̄ spans {result.T_bar_kelvin.min() * 1e3:.2f} → "
        f"{result.T_bar_kelvin.max() * 1e3:.2f} mK"
    )
    csv_path, pdf_path, _manifest = publish_bundle(
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
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
