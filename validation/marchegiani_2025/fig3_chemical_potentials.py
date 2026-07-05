r"""Marchegiani 2025 Fig 3 — chemical potentials μ_α(T) full reproduction.

Sweeps the bath temperature for the two M25 Fig 3 panels:

* Panel a — small gap asymmetry, ``ω_LR/(2π) = 0.5 GHz``
  (Δ_L/h = 49.5 GHz, Δ_R/h = 49.0 GHz)
* Panel b — large gap asymmetry, ``ω_LR/(2π) = 5 GHz``
  (Δ_L/h = 54.0 GHz, Δ_R/h = 49.0 GHz)

At each temperature recalibrate the photon drive to maintain
``Γ̃^ph_00 = 300 Hz`` (the M25 Fig 3 caption value), track the
4-unknown moment-system steady state with the deterministic
branch-continuation driver
:func:`qpsim.services.rate_equation.solve_rate_equation_branch`
(photon branch continued upward from the SI low-T analytic seed,
thermal branch continued downward from the full-equilibrium seed,
composite per the driver's documented exchange rule), and extract
the chemical potentials with the paper-exact density inversions
(arXiv 2408.17218 Eqs. 10–13 / published SI Eqs. S2–S5):

    μ_L    = Δ_L + T · log( x_L    √(Δ_L/2πT) )
    μ_{R>} = Δ_R + T · log( x_{R>} √(Δ_R/2πT) / erfc√(ω_LR/T) )
    μ_{R<} = Δ_R + T · log( x_{R<} √(Δ_R/2πT) / erf √(ω_LR/T) )

Branch selection: with the single-quasiparticle normalization of the
density equations (``M25Coefficients.cooper_pair_number_R``; M25 text
below Eq. 6) the system has a **unique physical root** at every
temperature of the Fig 3 sweep — the photon-up and thermal-down
passes converge to the same fixed point everywhere ("merged" labels)
and the historical multi-stability scatter is gone. The bidirectional
driver is kept because it is the honest guard: if a fold or genuine
branch pair ever appears (other parameter sets), the composite curve
switches branches at the documented exchange rule instead of silently
jumping.

Usage::

    python -m validation.marchegiani_2025.fig3_chemical_potentials
"""

from __future__ import annotations

import csv
import functools
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from qpsim.services.rate_equation import (
    M25BranchSweep,
    chemical_potentials_kelvin,
    crossover_temperature_kelvin,
    solve_rate_equation_branch,
    thermal_equilibrium_seed,
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

# T sweep range — matches the M25 Fig 3 plotted range up to the
# T̄ ≈ 146 mK crossover (beyond that the system is at full thermal
# equilibrium, μ_α ≈ 0).
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


def _coefficients_at(
    omega_LR_GHz: float,
    T_kelvin: float,
    *,
    gamma_ph_00_Hz: float = GAMMA_PH_00_HZ,
) -> M25Coefficients:
    """Coefficient bundle at (ω_LR, T) with per-T drive recalibration.

    ``gamma_ph_00_Hz`` defaults to the Fig 3 caption value; the Fig 4
    renormalized (dot-dashed) curves override it per the Fig. 4
    caption (600 Hz).
    """
    params = _make_params(omega_LR_GHz, T_kelvin)
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, DRIVE_TEMPLATE, gamma_ph_00_Hz,
    )
    drive = replace(DRIVE_TEMPLATE, Gamma_nu_scale_Hz=scale)
    return coefficients_from_physical_parameters_with_photon_drive(params, drive)


def _T_bar_estimate(omega_LR_GHz: float) -> float:
    """M25 Eq. 8 Lambert-W crossover ``T̄`` from the photon generation.

    ``g^ph_R`` is the photon-assisted generation rate in the low-gap
    electrode, evaluated at ground-state qubit weighting (``p_0 ≈ 1``
    below the crossover) with the drive calibration at a mid-sweep
    temperature — the Fig 3 recalibration keeps ``Γ̃^ph_00`` fixed, so
    the T dependence of ``g^ph_R`` is weak. For the Fig 3 caption
    parameters this lands at T̄ ≈ 146 mK (both panels).
    """
    coefs = _coefficients_at(omega_LR_GHz, 0.080)
    g_ph_R = float(coefs.g_ph_Rlt_per_state[0] + coefs.g_ph_Rgt_per_state[0])
    return crossover_temperature_kelvin(
        Delta_R_kelvin=DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB,
        r_Rlt_rate_Hz=R_RECOMB_HZ,
        g_photon_R_rate_Hz=max(g_ph_R, 1e-300),
    )


def solve_panel_branch_sweep(
    omega_LR_GHz: float,
    T_sweep: np.ndarray,
    *,
    gamma_ph_00_Hz: float = GAMMA_PH_00_HZ,
) -> M25BranchSweep:
    """Branch-tracked steady-state sweep for one Fig 3/4 panel.

    Wires the panel's coefficient builder into
    :func:`qpsim.services.rate_equation.solve_rate_equation_branch`:
    photon branch seeded by the SI low-T analytics (case selected by
    the panel's gap asymmetry, SI Note VI A/B), thermal branch seeded
    by the full-equilibrium densities at the top of the sweep, and
    the M25 Eq. 8 ``T̄`` estimate supplied as the exchange hint (only
    consulted if the branches ever disagree — they do not for the
    Fig 3 caption parameters, where the root is unique).

    ``gamma_ph_00_Hz`` is forwarded into the per-T drive
    recalibration (:func:`_coefficients_at`); it defaults to the
    Fig 3 caption value, and e.g. the Fig 4 caption's renormalized
    600 Hz drive can be swept without bypassing this entry point.

    Results are memoized per ``(ω_LR, grid, Γ̃^ph_00)`` for the
    lifetime of the process: the four validation modules (fig3/fig4 ×
    paper/observables) sweep the same two panels, so one pytest run
    computes each sweep once. Safe because the driver is
    deterministic (bit-identical rerun is pinned in
    ``tests/services/test_rate_equation_branch.py``) and the returned
    sweep is treated as immutable by all consumers.
    """
    return _solve_panel_branch_sweep_cached(
        float(omega_LR_GHz),
        tuple(float(T) for T in np.asarray(T_sweep, dtype=float)),
        float(gamma_ph_00_Hz),
    )


@functools.cache
def _solve_panel_branch_sweep_cached(
    omega_LR_GHz: float,
    T_sweep: tuple[float, ...],
    gamma_ph_00_Hz: float,
) -> M25BranchSweep:
    """Tuple-keyed worker behind :func:`solve_panel_branch_sweep`."""
    T_grid = np.array(T_sweep, dtype=float)
    Delta_L_K = (DELTA_R_OVER_H_GHZ + omega_LR_GHz) * 1e9 * _H_OVER_KB
    Delta_R_K = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    case = "large_asymmetry" if omega_LR_GHz >= 1.0 else "small_asymmetry"
    return solve_rate_equation_branch(
        lambda T: _coefficients_at(
            omega_LR_GHz, T, gamma_ph_00_Hz=gamma_ph_00_Hz,
        ),
        T_grid,
        photon_seed_case=case,
        thermal_seed=thermal_equilibrium_seed(
            Delta_L_kelvin=Delta_L_K,
            Delta_R_kelvin=Delta_R_K,
            omega_10_kelvin=OMEGA_10_OVER_H_GHZ * 1e9 * _H_OVER_KB,
            T_kelvin=float(T_grid[-1]),
        ),
        T_exchange_hint_kelvin=_T_bar_estimate(omega_LR_GHz),
    )


def _chemical_potentials_GHz(
    omega_LR_GHz: float,
    T_sweep: np.ndarray,
    x_L: np.ndarray,
    x_Rgt: np.ndarray,
    x_Rlt: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Paper-exact density → chemical-potential inversions, in GHz.

    Thin GHz-unit alias for
    :func:`qpsim.services.rate_equation.chemical_potentials_kelvin`
    (arXiv 2408.17218 Eqs. 10, 12, 13 — published SI Eqs. S2, S4,
    S5), which owns the physics: the ``√(Δ/2πT)`` prefactor and the
    erf/erfc partition factors whose omission flips the μ_{R>} vs
    μ_{R<} ordering relative to the published figure.
    """
    Delta_L_K = (DELTA_R_OVER_H_GHZ + omega_LR_GHz) * 1e9 * _H_OVER_KB
    Delta_R_K = DELTA_R_OVER_H_GHZ * 1e9 * _H_OVER_KB
    mu_L, mu_Rgt, mu_Rlt = chemical_potentials_kelvin(
        Delta_L_kelvin=Delta_L_K,
        Delta_R_kelvin=Delta_R_K,
        T_kelvin=T_sweep,
        x_L=x_L,
        x_Rgt=x_Rgt,
        x_Rlt=x_Rlt,
    )
    to_GHz = 1.0 / _H_OVER_KB / 1e9
    return mu_L * to_GHz, mu_Rgt * to_GHz, mu_Rlt * to_GHz


def _run_panel(omega_LR_GHz: float) -> Fig3PanelResult:
    T_sweep = np.linspace(T_MIN_K, T_MAX_K, NUM_T_POINTS)
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
            "# Marchegiani & Catelani 2025 Fig 3 — μ_α(T) full rate-eq "
            "(branch-continuation driver; Γ̄-normalized density eqs; "
            "paper-exact μ inversion); pinned by qpsim"
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
