"""Fischer & Catelani 2024 Figs 5-7 — f(E) under pair-breaking photon drive.

Sweeps the five power levels from F24 Sec. IV at fixed bath
temperature ``T_B = 0.1 K`` and records the converged ``f(E)`` per
power, producing the overlay figure that Fischer 2024 presents as
three panels (Figs 5, 6, 7). We emit a single combined plot and a
single baseline CSV; the tests assert bit-identity column-by-column.

Parameters are shared with :mod:`validation.fischer_2024.fig8_xqp_pb`
— same ω_PB / n̄_PB / grid / powers — so the two modules exercise the
same PB-photon collision kernel from orthogonal axes (scan power at
fixed T_B here; scan T_B at each power in Fig 8).

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV.

Usage::

    python -m validation.fischer_2024.figs_5_7_fe_pb
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend
from qpsim.observables.density import qp_fraction

from validation.fischer_2024.fig8_xqp_pb import (
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    N_BAR_PB,
    NUM_BINS,
    OMEGA_PB,
    POWER_LEVELS,
    T_C,
    TAU_0,
    _build_state,
    _material,
)

T_BATH_FE = 0.1  # F24 Figs 5-7 fix T_B at 0.1 K


@dataclass(frozen=True)
class Figs57Result:
    E: np.ndarray
    powers: tuple[float, ...]
    f_thermal: np.ndarray               # shape (NE,); f at T_bath, no PB drive
    f_by_power: dict[float, np.ndarray]  # power → (NE,) driven f
    x_qp_by_power: dict[float, float]    # power → scalar x_qp


def run() -> Figs57Result:
    material = _material()
    backend = T3DiffusionBackend()

    state = _build_state(material, T_BATH_FE)

    # Commensurability probe once — inherits the Fig-8 grid, so this is a
    # belt-and-braces check.
    dE_scalar = float(state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"ω_PB={OMEGA_PB} is not integer-commensurate with dE={dE_scalar:.4f}"
        )

    # Thermal reference (no PB drive): state.f already populated with f_FD.
    f_thermal = state.f.copy()

    f_by_power: dict[float, np.ndarray] = {}
    x_qp_by_power: dict[float, float] = {}

    for power in POWER_LEVELS:
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": power / N_BAR_PB,
        }
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            pb_photon_params=pb_params,
            newton_tol=1e-14,
            newton_max_iter=500,
        )
        f_by_power[power] = driven.f.copy()
        x_qp_by_power[power] = float(
            qp_fraction(driven.f, driven.spectral, delta_0=DELTA_0),
        )

    return Figs57Result(
        E=state.spectral.E,
        powers=POWER_LEVELS,
        f_thermal=f_thermal,
        f_by_power=f_by_power,
        x_qp_by_power=x_qp_by_power,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "f24_figs_5_7_fe_pb.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Figs57Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer & Catelani 2024 Figs 5-7 — f(E) with PB-photon drive; pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_PB={OMEGA_PB} "
            f"n_bar_PB={N_BAR_PB} T_bath={T_BATH_FE}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        powers_csv = ",".join(f"{p:g}" for p in result.powers)
        writer.writerow([f"# powers_ns_inv={powers_csv}"])
        x_qp_csv = ",".join(f"{result.x_qp_by_power[p]:.17e}" for p in result.powers)
        writer.writerow([f"# x_qp_by_power={x_qp_csv}"])
        header = ["E_uev", "f_thermal"] + [f"f_power_{p:g}" for p in result.powers]
        writer.writerow(header)
        for i in range(result.E.size):
            row = [f"{result.E[i]:.17e}", f"{result.f_thermal[i]:.17e}"]
            row.extend(f"{result.f_by_power[p][i]:.17e}" for p in result.powers)
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Figs57Result:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    powers: tuple[float, ...] = ()
    x_qp_list: list[float] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# powers_ns_inv"):
                powers = tuple(float(x) for x in first.split("=", 1)[1].split(","))
                continue
            if first.startswith("# x_qp_by_power"):
                x_qp_list = [float(x) for x in first.split("=", 1)[1].split(",")]
                continue
            if first.startswith("#") or first == "E_uev":
                continue
            rows.append([float(x) for x in line])
    if not powers:
        raise RuntimeError(f"Baseline at {path} missing '# powers_ns_inv=' metadata.")
    if len(x_qp_list) != len(powers):
        raise RuntimeError(
            f"Baseline at {path} missing '# x_qp_by_power=' metadata."
        )
    data = np.array(rows, dtype=float)
    return Figs57Result(
        E=data[:, 0],
        powers=powers,
        f_thermal=data[:, 1],
        f_by_power={p: data[:, i + 2] for i, p in enumerate(powers)},
        x_qp_by_power=dict(zip(powers, x_qp_list, strict=True)),
    )


def write_plot(result: Figs57Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = result.E / DELTA_0
    ax.semilogy(x, result.f_thermal, "k-", lw=1.5, alpha=0.8,
                label=rf"thermal ($T_B={T_BATH_FE}$ K, no PB)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result.powers)))
    for power, color in zip(result.powers, colors, strict=True):
        ax.semilogy(x, result.f_by_power[power], lw=2.0, color=color,
                    label=rf"$c \cdot \bar n = {power:g}$ ns$^{{-1}}$")
    ax.axvline(OMEGA_PB / (2 * DELTA_0), color="red", ls="--", lw=0.8, alpha=0.5,
               label=rf"$\omega_{{PB}}/2\Delta = {OMEGA_PB / (2 * DELTA_0):.2f}$")
    ax.set_xlabel(r"$E / \Delta$", fontsize=14)
    ax.set_ylabel(r"$f(E)$", fontsize=14)
    ax.set_title(
        "Fischer & Catelani 2024 Figs 5-7 — PB-photon drive at fixed $T_B$\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\tau_0={TAU_0:.0f}$ ns, "
        rf"$\omega_{{\mathrm{{PB}}}}=2.8\,\Delta_0$, $T_B={T_BATH_FE}$ K",
        fontsize=10,
    )
    ax.set_xlim(1.0, min(E_MAX_FACTOR, 5.0))
    f_stack = np.concatenate([result.f_by_power[p] for p in result.powers])
    f_pos = f_stack[f_stack > 0]
    if f_pos.size > 0:
        ax.set_ylim(max(float(f_pos.min()), 1e-80) / 10, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Figs 5-7 — f(E) with PB-photon drive ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_PB={OMEGA_PB:.2f} μeV, "
        f"T_B={T_BATH_FE} K"
    )
    print(f"  Powers (c·n̄, ns⁻¹): {list(POWER_LEVELS)}")
    print(f"  Grid: NE={NUM_BINS}")
    result = run()
    for p in result.powers:
        print(f"    c·n̄ = {p:g}:  x_qp = {result.x_qp_by_power[p]:.4e}")
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
