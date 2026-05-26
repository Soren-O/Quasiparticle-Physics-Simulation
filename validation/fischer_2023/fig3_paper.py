"""Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction.

This is the Fig. 3 reproduction path: four curves at
$\\tau_\\ell / \\tau_0^{PB} \\in \\{0, 0.1, 1, 10\\}$ on the paper's energy
grid, with $\\tau_0^{PB}$ extracted through the phonon-side Kaplan
pair-breaking rate.

The two existing scripts in this directory cover only sub-sets:

* :mod:`fig3_tau_l_zero` — the $\\tau_\\ell = 0$ panel, bit-identical pinned.
* :mod:`fig3_finite_tau_l` — finite ratios at the *characterization* set
  $\\{0.5, 1, 2, 5, 10\\}$ on a coarser 810-bin grid.

Neither matches the paper's published legend. This script does:

* Paper grid: ``NE = 1620``, ``dE = 1 μeV``, integer-commensurate with
  $\\omega_0 = \\Delta_0/9 = 20\\,\\mu$eV.
* Paper legend ratios: $\\{0, 0.1, 1, 10\\}$.
* Continuation through intermediate ratios for stability of the strong-
  bottleneck branch.
* Paper-style axis: $f(E)$ vs $E/\\Delta_0 - 1$ on $[0, 4]$, with photon-
  step markers at $n\\,\\omega_0/\\Delta_0$.

The older qpsim-native extraction reused the QP-side recombination kernel
inside the phonon equation and produced an apparent ~38x tau_0^PB mismatch.
This module uses the F&C/Kaplan phonon-side pair-breaking kernel and the
analytic near-threshold S_+ quadrature correction, giving tau_0^PB ~= 255 ps
for the Table I parameters.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:

    Δ_0     = 180 μeV
    τ_0     = 438 ns
    T_bath  = 0.1 K
    ω_0     = Δ_0 / 9 = 20 μeV
    n̄       = 1 × 10^7
    c_phot  = 1 Hz = 1 × 10^-9 ns^-1

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 Table I parameters ──────────────────────────────────

DELTA_0 = 180.0            # μeV
TAU_0 = 438.0              # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1               # K
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV (sub-gap, integer-commensurate with dE)
N_BAR = 1e7
C_PHOT = 1e-9              # ns^-1 (1 Hz)

# Paper grid: 1620 bins, dE = 1 μeV. ω_0 / dE = 20 (integer).
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1620

# Paper-target legend ratios (Fischer 2023 Fig. 3).
PAPER_RATIOS: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)

# Continuation ladder: smooth ramp through intermediate ratios so the
# Picard fixed point stays in the basin of attraction. Targets pulled
# out and stored; non-target ratios are discarded after the continuation
# step. Picard struggles above ratio ~5 (the map is non-contractile near
# the strong-bottleneck branch), so the final 5 → 10 step switches to
# coupled Newton on the joint (f, n_ph) state.
CONTINUATION_RATIOS: tuple[float, ...] = (
    0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
)

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the numerical tau_0^PB diverges from the paper-quoted 255 ps."""


@dataclass(frozen=True)
class Fig3PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    tau_0_pb_ns: float
    ratios: tuple[float, ...]   # paper ratios {0, 0.1, 1, 10}
    f_by_ratio: dict[float, np.ndarray]
    f_FD: np.ndarray            # thermal reference at T_bath


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,  # F&C 2023 Table I: τ_0^PB = 255 ps
    )


def _build_grid_and_spectral() -> tuple[np.ndarray, np.ndarray, SpectralContext]:
    """Build the paper-grid energy axis + dE widths + spectral context."""
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    frac_err = abs(OMEGA_0 - m * dE_scalar) / OMEGA_0
    if frac_err > 1e-10:
        raise RuntimeError(
            f"Fischer Fig. 3 paper grid not commensurate: "
            f"ω_0 = {OMEGA_0}, m·dE = {m * dE_scalar}, frac_err = {frac_err}."
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _compute_tau_0_pb(spectral: SpectralContext) -> float:
    """Numerical τ_0^PB from the simulator's phonon-side kernel at f=0, ω≈2Δ.

    Uses the F&C 2023 Eq. 12 phonon-side kernel ``K⁺/(π Δ τ_0^PB)``
    (built via :func:`build_recombination_kernel_phonon_side`) — same
    convention as :func:`fig3_finite_tau_l._compute_tau_0_pb` post the
    F23 phonon-side-kernel wiring.
    """
    K_r0_phonon_side = build_recombination_kernel_phonon_side(
        spectral, tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,
    )
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
        spectral.E,
    )
    f_zero = np.zeros(spectral.E.size)
    _, b_ph = compute_phonon_source_sink(
        f_zero, spectral, None, None,
        idx_diff, idx_sum, diff_sign,
        omega_bins.size,
        enable_scattering=False, enable_recombination=True,
        K_r0_phonon_side=K_r0_phonon_side,
    )
    threshold = 2.0 * spectral.gap
    above = (omega_bins >= threshold) & (b_ph < -1e-30)
    if not np.any(above):
        raise RuntimeError(
            "Could not find a phonon bin above 2Δ with a pair-breaking rate."
        )
    first_idx = int(np.argmax(above))
    return float(1.0 / -b_ph[first_idx])


def _build_state(
    material: Material,
    spectral: SpectralContext,
    f_seed: np.ndarray,
    tau_l_scalar: float,
    *,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    """Build a T3 state with the given τ_l scalar + (f, n_ph) seeds."""
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    if n_ph_seed is None:
        n_ph_seed = thermal_phonon_occupation(omega, T_BATH)
    phonon = PhononState(
        n_ph=n_ph_seed.reshape(1, -1, 1).copy(),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_scalar),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=f_seed.copy(),
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_BATH,
    )


def _solve_tau_l_zero(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
) -> T3DiffusionState:
    """τ_l = 0: thermal-phonon shortcut. Newton-on-f only."""
    return backend.steady_state(
        state,
        use_thermal_phonons=True,
        photon_params=photon_params,
        newton_tol=1e-12,
        newton_max_iter=500,
    )


def _solve_picard(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    mixing: float,
) -> T3DiffusionState:
    """Picard + Anderson on (f, n_ph). Mixing under-relaxed at high ratios."""
    return backend.steady_state(
        state,
        method="picard",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        picard_tol=1e-8,
        picard_max_iter=10000,
        picard_mixing=mixing,
        anderson_depth=0,
        newton_tol=1e-12,
        newton_max_iter=500,
    )


def _solve_coupled_newton(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
) -> T3DiffusionState:
    """Coupled Newton on the joint (f, n_ph) vector (strong-bottleneck branch)."""
    return backend.steady_state(
        state,
        method="coupled_newton",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        coupled_newton_tol=1e-10,
        coupled_newton_max_iter=50,
        coupled_newton_fd_step=1e-8,
    )


def _picard_mixing_for_ratio(ratio: float) -> float:
    return 0.15 if ratio > 2.0 else 0.30


def run() -> Fig3PaperResult:
    """Solve Fischer Fig. 3 at all paper ratios via continuation."""
    material = _fischer_material()
    E, _, spectral = _build_grid_and_spectral()

    tau_0_pb = _compute_tau_0_pb(spectral)
    tau_0_pb_ps = tau_0_pb * 1000.0  # ns → ps
    ratio_paper = tau_0_pb_ps / PAPER_TAU_0_PB_PS
    print(f"  τ_0^PB (phonon-side extracted)       = {tau_0_pb:.4f} ns "
          f"({tau_0_pb_ps:.1f} ps)")
    print(f"  Paper-quoted τ_0^PB                   ≈ {PAPER_TAU_0_PB_PS:.0f} ps")
    if ratio_paper > TAU_0_PB_WARN_FACTOR or ratio_paper < 1.0 / TAU_0_PB_WARN_FACTOR:
        print(
            f"  ⚠ τ_0^PB normalization mismatch: extracted/paper = {ratio_paper:.2f}×.",
            flush=True,
        )

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    backend = T3DiffusionBackend()
    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR, "c_phot": C_PHOT}

    f_by_ratio: dict[float, np.ndarray] = {}

    # ── ratio 0: thermal-phonon shortcut (Newton-only, paper τ_l=0 curve) ──
    state0 = _build_state(material, spectral, f_FD, tau_l_scalar=0.0)
    print("  τ_l/τ_0^PB = 0     → target  thermal-phonon shortcut", flush=True)
    converged0 = _solve_tau_l_zero(backend, state0, photon_params)
    f_by_ratio[0.0] = converged0.f.copy()

    # ── continuation ladder for finite ratios ──
    f_seed = converged0.f.copy()
    n_ph_seed: np.ndarray | None = None  # bath thermal at first finite step
    for ratio in CONTINUATION_RATIOS:
        tau_l = ratio * tau_0_pb
        state = _build_state(
            material, spectral, f_seed, tau_l, n_ph_seed=n_ph_seed,
        )
        is_target = ratio in PAPER_RATIOS
        tag = "→ target  " if is_target else "(continuation)"

        if ratio > 5.0:
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} coupled_newton "
                f"(seeded from prior-ratio (f, n_ph))",
                flush=True,
            )
            converged = _solve_coupled_newton(backend, state, photon_params)
        else:
            mixing = _picard_mixing_for_ratio(ratio)
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} picard "
                f"(mixing={mixing}, AA=0)",
                flush=True,
            )
            converged = _solve_picard(backend, state, photon_params, mixing=mixing)

        if is_target:
            f_by_ratio[ratio] = converged.f.copy()
        f_seed = converged.f.copy()
        n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()

    # Sanity-check all paper ratios captured.
    missing = [r for r in PAPER_RATIOS if r not in f_by_ratio]
    if missing:
        raise RuntimeError(f"Continuation did not capture paper ratios: {missing}")

    return Fig3PaperResult(
        E=E,
        tau_0_pb_ns=tau_0_pb,
        ratios=PAPER_RATIOS,
        f_by_ratio=f_by_ratio,
        f_FD=f_FD,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer_fig3_paper.csv`` because the tau_0^PB normalization is
    now pinned to the paper/Kaplan phonon-side pair-breaking rate.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer_fig3_paper.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


_TAU_0_PB_RE = re.compile(r"tau_0_pb_ns=([\deE.+-]+)")
_RATIOS_RE = re.compile(r"ratios=\[([^\]]+)\]")


def write_baseline(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Write the four paper-ratio f(E) arrays + thermal reference to CSV."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["E_uev", "f_FD"] + [f"f_ratio_{r:g}" for r in result.ratios]
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "# Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction"
        ])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} "
            f"omega_0={OMEGA_0} n_bar={N_BAR} c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow([f"# tau_0_pb_ns={result.tau_0_pb_ns} ratios={list(result.ratios)}"])
        writer.writerow(header_cols)
        n = result.E.size
        for i in range(n):
            row = [f"{result.E[i]:.17e}", f"{result.f_FD[i]:.17e}"]
            row.extend(f"{result.f_by_ratio[r][i]:.17e}" for r in result.ratios)
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig3PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig3PaperResult`."""
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    tau_0_pb: float | None = None
    ratios: tuple[float, ...] = ()
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# tau_0_pb_ns"):
                m_tau = _TAU_0_PB_RE.search(first)
                m_ratios = _RATIOS_RE.search(first)
                if m_tau:
                    tau_0_pb = float(m_tau.group(1))
                if m_ratios:
                    ratios = tuple(
                        float(x.strip()) for x in m_ratios.group(1).split(",") if x.strip()
                    )
                continue
            if first.startswith("#") or first == "E_uev":
                continue
            rows.append([float(x) for x in line])
    if tau_0_pb is None or not ratios:
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb_ns / ratios metadata.")
    data = np.array(rows, dtype=float)
    # Column layout: E_uev, f_FD, f_ratio_<r0>, f_ratio_<r1>, ...
    return Fig3PaperResult(
        E=data[:, 0],
        tau_0_pb_ns=tau_0_pb,
        ratios=ratios,
        f_by_ratio={r: data[:, 2 + i] for i, r in enumerate(ratios)},
        f_FD=data[:, 1],
    )


def write_plot(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Paper-style plot: log-scale f(E) with all four ratios + thermal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    from validation.fischer_2023._paper_envelope import (
        EnvelopeParams,
        envelope_no_thermal,
        envelope_with_thermal,
        solve_b0,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # Paper Fig. 3 axis: f(E) vs E/Δ_0 - 1 on [0, 4].
    x = result.E / DELTA_0 - 1.0

    # Paper Fig. 3 palette (named matplotlib colors, matching the standalone
    # paper reproduction): solid grayscale numerical, dashed analytical.
    solid_colors = ["k", "dimgray", "gray", "lightgray"]
    dash_colors = ["red", "green", "blue", "blue"]
    Eplot = np.linspace(DELTA_0 + 1e-3, 5.0 * DELTA_0, 4000)
    TB_uev = T_BATH * KB_UEV_PER_K

    for r, sc, dc in zip(result.ratios, solid_colors, dash_colors, strict=True):
        ax.semilogy(
            x, np.maximum(result.f_by_ratio[r], 1e-40),
            color=sc, lw=1.0,
            label=rf"$\tau_\ell/\tau_0^{{\rm PB}}={r:g}$ (num)",
        )
        ep = EnvelopeParams(
            Delta0=DELTA_0,
            Tc_uev=T_C * KB_UEV_PER_K,
            tau0=TAU_0,
            tau0_PB=result.tau_0_pb_ns,
            tau_l=r * result.tau_0_pb_ns,
            TB_uev=TB_uev,
            nbar=N_BAR,
            omega0=OMEGA_0,
            cphot_QP=C_PHOT,
        )
        b0 = solve_b0(ep)
        f_env = (envelope_with_thermal(Eplot, ep, b0) if r == 0.0
                 else envelope_no_thermal(Eplot, ep, b0))
        ax.semilogy(Eplot / DELTA_0 - 1.0, np.maximum(f_env, 1e-40),
                    color=dc, ls="--", lw=0.8)

    ax.set_xlabel(r"$E/\Delta_0 - 1$")
    ax.set_ylabel(r"$f(E)$")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(1e-35, 3e-7)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, T_bath={T_BATH} K, "
        f"ω_0={OMEGA_0:.2f} μeV"
    )
    print(f"  Grid: NE={NUM_BINS}, dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    print(f"  Paper ratios: {list(PAPER_RATIOS)}")
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
