r"""Marchegiani 2025 Fig 3 — analytic crossover temperature T̄(g^ph_R).

Closed-form reproduction of M25 Eq. 8 — the Lambert-W crossover
that separates the low-``T`` nonequilibrium regime from the local-
quasiequilibrium regime in a gap-asymmetric transmon junction.

The paper validates Eq. 8 "within a few percent of the numerical
results" from the full three-chemical-potential rate-equation
integration (which is deferred per NFP §7.10 Gate 8 strategy A).

Reproduces the Fig 3 caption parameter set (Δ_R/h = 49 GHz,
r^L = r^{R<} = 6.25 MHz) and sweeps the photon-generation rate
``g^\mathrm{ph}_R`` across the physically interesting range
(1 Hz → 1 MHz). The output is the monotonic T̄(g^ph_R) curve —
projecting onto the paper's figure the crossover T̄ ≈ 70 mK (small
ω_LR) and T̄ ≈ 150 mK (large ω_LR) dashed lines in panels (a)/(b).

Usage::

    python -m validation.marchegiani_2025.fig3_crossover_temperature
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import crossover_temperature_kelvin

# ── M25 Fig 3 caption parameters ─────────────────────────────────────
_H_J_S = 6.62607015e-34
_KB_J_PER_K = 1.380649e-23

DELTA_R_OVER_H_HZ = 49.0e9           # Δ_R/h = 49 GHz
DELTA_R_KELVIN = DELTA_R_OVER_H_HZ * _H_J_S / _KB_J_PER_K  # ≈ 2.352 K
R_RLT_RATE_HZ = 6.25e6                # r^{R<} = 6.25 MHz

# Sweep span chosen to bracket the M25 Fig 3 paper-reading T̄ values
# (≈ 70 mK and ≈ 150 mK). Inverting Eq. 8 at those T̄ with the caption
# parameters gives g^{ph}_R ~ 1e−23 Hz and ~ 5e−8 Hz respectively —
# i.e. the paper's Fig 3(a) operates at vanishingly small photon drive.
# The sweep extends down 26 decades to include both reference points
# plus the Δ_R asymptote at the high-g^{ph} end.
G_PHOTON_MIN_HZ = 1.0e-25
G_PHOTON_MAX_HZ = 1.0e6
NUM_POINTS = 63  # ~2 points per decade


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


def write_baseline(result: Fig3Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "# Marchegiani & Catelani 2025 Fig 3 — Eq. 8 Lambert-W T̄; "
            "pinned by qpsim"
        ])
        writer.writerow([
            f"# Delta_R_over_h_Hz={DELTA_R_OVER_H_HZ:g}  "
            f"Delta_R_kelvin={DELTA_R_KELVIN:.6f}  "
            f"r_Rlt_rate_Hz={R_RLT_RATE_HZ:g}"
        ])
        writer.writerow(["g_photon_R_Hz", "T_bar_kelvin"])
        for g, T in zip(result.g_photon_R_Hz, result.T_bar_kelvin, strict=True):
            writer.writerow([f"{g:.17e}", f"{T:.17e}"])
    return path


def read_baseline(path: Path | None = None) -> Fig3Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("#") or first == "g_photon_R_Hz":
                continue
            rows.append([float(x) for x in line])
    data = np.array(rows, dtype=float)
    return Fig3Result(g_photon_R_Hz=data[:, 0], T_bar_kelvin=data[:, 1])


def write_plot(result: Fig3Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogx(
        result.g_photon_R_Hz, result.T_bar_kelvin * 1e3,
        "o-", lw=1.5, ms=4, color="tab:purple",
        label=r"$\bar T = 2\Delta_R / W(4\pi r^{R<}/g^\mathrm{ph}_R)$",
    )

    # Annotate the Fig 3 paper-reading points (approximate, from the
    # published figure's dashed-line positions). The full rate-equation
    # integration behind Fig 3 fixes g^ph_R via ω_LR and the photon
    # drive settings; the closed form here lets us tabulate T̄ over a
    # range and overlay the paper's two panels.
    ax.axhline(70.0, color="tab:cyan", ls=":", lw=1.0, alpha=0.7,
               label=r"M25 Fig 3(a) numerical: $\bar T \approx 70$ mK  "
                     r"(small $\omega_{LR}$)")
    ax.axhline(150.0, color="tab:olive", ls=":", lw=1.0, alpha=0.7,
               label=r"M25 Fig 3(b) numerical: $\bar T \approx 150$ mK  "
                     r"(large $\omega_{LR}$)")

    # Δ_R upper bound (T̄ cannot exceed Δ_R since W > 0 for positive args).
    ax.axhline(DELTA_R_KELVIN * 1e3, color="red", ls="--", lw=0.8, alpha=0.4,
               label=rf"$\Delta_R = {DELTA_R_KELVIN * 1e3:.0f}$ mK")

    ax.set_xlabel(r"$g^\mathrm{ph}_R$ [Hz] (photon-assisted PB rate on R)",
                  fontsize=12)
    ax.set_ylabel(r"$\bar T$ [mK]", fontsize=12)
    ax.set_title(
        "Marchegiani 2025 Eq. 8: Lambert-W crossover temperature\n"
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


def generate_baseline() -> tuple[Path, Path]:
    print("M25 Fig 3 — Lambert-W crossover temperature T̄(g^ph_R)")
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
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
