"""Fischer 2023 Fig 3 finite-τ_l reproduction — Ph0-constant phonon model.

Exercises ``T3DiffusionBackend.steady_state`` at five non-trivial τ_l
values:

    τ_l / τ_0^PB ∈ {0.5, 1, 2, 5, 10}

where ``τ_0^PB`` is Kaplan's characteristic phonon pair-breaking time,
computed numerically from the low-temperature limit of
``compute_phonon_source_sink``. The ``τ_l/τ_0^PB = 0`` case is in
:mod:`validation.fischer_2023.fig3_tau_l_zero` (Newton-only). Ratios
0.5 through 5 use the Picard orchestrator; the ``τ_l/τ_0^PB = 10``
strong-bottleneck case cannot converge through Picard + Anderson
(Roadmap Phase 2.1: the Picard map is non-contractile and orbits
the physical fixed point at ~1e-7 residual). That case uses the
coupled-Newton solver on the joint ``(f, n_ph)`` vector, seeded
from the ratio-5 converged state.

Continuation strategy mirrors the legacy script: each ratio seeds
from the previous ratio's converged ``(f, n_ph)``, which is essential
for Picard to stabilize at ratios ≳ 2 and for coupled Newton to pick
up the physical branch at ratio 10 (the trivial ``f = 0`` branch is
a second fixed point in the strong-bottleneck regime).

Fischer, Catelani — Phys. Rev. Applied 19, 054087 (2023), Table I:
same parameters as :mod:`fig3_tau_l_zero`. Grid: 810 bins (Chapter 3
thesis convention per NFP §6.4.1), not the 1620-bin paper grid —
the iterative mode's ``1e-6`` tolerance tier doesn't demand paper
resolution.

Usage — generate baseline + PDF::

    python -m validation.fischer_2023.fig3_finite_tau_l
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
    build_recombination_kernel_base,
    compute_phonon_source_sink,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 Table I parameters (shared with fig3_tau_l_zero) ────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1
OMEGA_0 = DELTA_0 / 9.0
N_BAR = 1e7
C_PHOT = 1e-9

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810  # Ch. 3 thesis convention per NFP §6.4.1

TAU_L_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
"""Target ratios pinned in the baseline CSV.

The ``0`` case is handled in :mod:`fig3_tau_l_zero` (Newton-only).
Ratio 10 uses coupled Newton (see ``_solver_method`` below); ratios
0.5-5 use Picard.
"""

_CONTINUATION_RATIOS: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0)
"""Continuation schedule including intermediate ratios 3 and 4.

Picard at ratio 5 cannot converge by seeding from ratio 2 directly —
the jump is too large and the iteration orbits the physical branch.
The extra 3 and 4 solves are cheap (they converge quickly from the
prior ratio's seed) and reliably carry the continuation to 5. The
final step (5 → 10) switches solvers (coupled Newton) because
Picard cannot traverse this bifurcation.
"""


@dataclass(frozen=True)
class Fig3FiniteResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    tau_0_pb: float
    ratios: tuple[float, ...]
    f_by_ratio: dict[float, np.ndarray]


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _compute_tau_0_pb(
    ctx: SpectralContext,
    K_r0: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    omega_bins: np.ndarray,
) -> float:
    """Extract τ_0^PB from the low-T pair-breaking rate at ω ≳ 2Δ.

    At ``f = 0`` the e-ph source-sink reduces to a pure pair-breaking
    absorption: ``dn_ph/dt = b_ph · n_ph`` with ``b_ph < 0`` above
    ``2Δ``. The first non-zero entry just above the 2Δ threshold sets
    the characteristic rate, whose inverse is ``τ_0^PB`` (ns).
    """
    f_zero = np.zeros(ctx.E.size)
    _, b_ph = compute_phonon_source_sink(
        f_zero, ctx, None, K_r0,
        omega_idx_diff, omega_idx_sum, diff_sign,
        omega_bins.size,
        enable_scattering=False, enable_recombination=True,
    )
    threshold = 2.0 * ctx.gap
    above = (omega_bins >= threshold) & (b_ph < -1e-30)
    if not np.any(above):
        raise RuntimeError(
            "Could not find a phonon bin above 2Δ with a pair-breaking "
            "rate — the grid may be too coarse."
        )
    first_idx = int(np.argmax(above))
    return float(1.0 / -b_ph[first_idx])


def _initial_state(
    material: Material,
    f_seed: np.ndarray,
    tau_l_scalar: float,
    *,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    """Build a T3 state with the given τ_l scalar, ``f`` seed, and ``n_ph`` seed.

    ``n_ph_seed`` defaults to the bath thermal occupation (equilibrium
    start). Pass a converged phonon array from the previous continuation
    step to hand both ``(f, n_ph)`` forward — required for coupled
    Newton at ratio 10 to pick up the physical branch.
    """
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
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


def _solver_method(ratio: float) -> str:
    """Ratio 10 uses coupled Newton; everything else uses Picard."""
    return "coupled_newton" if ratio > 5.0 else "picard"


def _solver_knobs(ratio: float) -> dict[str, float | int]:
    """Picard / Anderson parameters that converge at each Picard ratio.

    Mixing is under-relaxed to 0.15 above ratio 2. Ratio 10 uses the
    coupled-Newton path instead; its knobs live in
    :func:`_coupled_newton_knobs`.
    """
    if ratio > 2.0:
        return {"picard_mixing": 0.15, "anderson_depth": 0}
    return {"picard_mixing": 0.3, "anderson_depth": 0}


def run() -> Fig3FiniteResult:
    """Solve Fischer Fig 3 at each finite-τ_l ratio via continuation."""
    material = _fischer_material()

    # Grid + kernels + phonon frequency map once — τ_0^PB and the
    # continuation all share them.
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    if abs(OMEGA_0 - m * dE_scalar) / OMEGA_0 > 1e-10:
        raise RuntimeError(
            f"Fig 3 finite-τ_l grid not commensurate: ω₀={OMEGA_0}, m·dE={m * dE_scalar}"
        )

    # The backend rebuilds K_s0 and K_r0 internally on every solve, so
    # here we only need K_r0 for the τ_0^PB probe (pair-breaking-only).
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    K_r0 = build_recombination_kernel_base(spectral, tau_0=TAU_0, T_c=T_C)
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(E)

    tau_0_pb = _compute_tau_0_pb(
        spectral, K_r0, idx_diff, idx_sum, diff_sign, omega_bins,
    )

    # Fermi-Dirac seed for the first ratio; n_ph seed is bath thermal.
    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    backend = T3DiffusionBackend()
    f_seed = f_FD
    n_ph_seed: np.ndarray | None = None
    f_by_ratio: dict[float, np.ndarray] = {}

    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR, "c_phot": C_PHOT}

    for ratio in _CONTINUATION_RATIOS:
        tau_l = ratio * tau_0_pb
        state = _initial_state(material, f_seed, tau_l, n_ph_seed=n_ph_seed)
        method = _solver_method(ratio)
        is_target = ratio in TAU_L_RATIOS
        tag = "→ target" if is_target else "(continuation)"

        if method == "picard":
            knobs = _solver_knobs(ratio)
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g} {tag}  picard "
                f"(mixing={knobs['picard_mixing']}, AA={knobs['anderson_depth']})",
                flush=True,
            )
            converged = backend.steady_state(
                state,
                method="picard",
                photon_params=photon_params,
                picard_tol=1e-8,
                picard_max_iter=10000,
                picard_mixing=float(knobs["picard_mixing"]),
                anderson_depth=int(knobs["anderson_depth"]),
                newton_tol=1e-12,
                newton_max_iter=500,
            )
        else:
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g} {tag}  coupled_newton "
                f"(seeded from prior-ratio (f, n_ph))",
                flush=True,
            )
            converged = backend.steady_state(
                state,
                method="coupled_newton",
                photon_params=photon_params,
                coupled_newton_tol=1e-10,
                coupled_newton_max_iter=50,
                coupled_newton_fd_step=1e-8,
            )

        if is_target:
            f_by_ratio[ratio] = converged.f.copy()
        # Continuation: hand both f and n_ph forward.
        f_seed = converged.f
        n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()

    return Fig3FiniteResult(
        E=E,
        tau_0_pb=tau_0_pb,
        ratios=TAU_L_RATIOS,
        f_by_ratio=f_by_ratio,
    )


def baseline_path() -> Path:
    """Path to the pinned regression CSV."""
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer_fig3_finite_tau_l_constant.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig3FiniteResult, path: Path | None = None) -> Path:
    """Write the continuation arrays to a CSV suitable for regression."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["E_uev"] + [f"f_ratio_{r:g}" for r in result.ratios]
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig 3 finite-τ_l (Ph0-constant) — pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} omega_0={OMEGA_0} "
            f"n_bar={N_BAR} c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow([f"# tau_0_pb={result.tau_0_pb} ratios={list(result.ratios)}"])
        writer.writerow(header_cols)
        n = result.E.size
        for i in range(n):
            row = [f"{result.E[i]:.17e}"]
            row.extend(f"{result.f_by_ratio[r][i]:.17e}" for r in result.ratios)
            writer.writerow(row)
    return path


_TAU_0_PB_RE = re.compile(r"tau_0_pb=([\deE.+-]+)")
_RATIOS_RE = re.compile(r"ratios=\[([^\]]+)\]")


def read_baseline(path: Path | None = None) -> Fig3FiniteResult:
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
            if first.startswith("# tau_0_pb"):
                # "# tau_0_pb=<float> ratios=[<r>, <r>, ...]"
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
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb / ratios metadata.")

    data = np.array(rows, dtype=float)
    return Fig3FiniteResult(
        E=data[:, 0],
        tau_0_pb=tau_0_pb,
        ratios=ratios,
        f_by_ratio={r: data[:, i + 1] for i, r in enumerate(ratios)},
    )


def write_plot(result: Fig3FiniteResult, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(result.E / kT, 500.0)) + 1.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = result.E / DELTA_0
    ax.semilogy(x, f_FD, "k--", lw=1.5, label=r"Fermi-Dirac ($\tau_\ell=0$ limit)")
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(result.ratios)))
    for r, color in zip(result.ratios, colors, strict=True):
        ax.semilogy(x, result.f_by_ratio[r], lw=2.0, color=color,
                    label=rf"$\tau_\ell / \tau_0^{{PB}} = {r:g}$")

    for n_step in range(1, 10):
        E_step = DELTA_0 + n_step * OMEGA_0
        if E_step / DELTA_0 < E_MAX_FACTOR:
            ax.axvline(E_step / DELTA_0, color="gray", ls=":", lw=0.5, alpha=0.4)

    ax.set_xlabel(r"$E / \Delta_0$", fontsize=14)
    ax.set_ylabel(r"$f(E)$", fontsize=14)
    ax.set_title(
        "Fischer 2023 Fig 3 — finite-$\\tau_\\ell$ (Ph0-constant)\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\bar n=10^7$, $T_B={T_BATH}$ K, "
        rf"$\tau_0^{{PB}}={result.tau_0_pb:.3f}$ ns",
        fontsize=10,
    )
    ax.set_xlim(1.0, E_MAX_FACTOR)
    # Lower bound: about an order of magnitude below the smallest non-trivial f.
    all_positive = np.concatenate([result.f_by_ratio[r] for r in result.ratios])
    all_positive = all_positive[all_positive > 0]
    if all_positive.size > 0:
        ax.set_ylim(max(float(all_positive.min()), 1e-80) / 10, 1e-8)
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", ls=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig 3 finite-τ_l (Ph0-constant) reproduction ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, T_bath={T_BATH} K, "
        f"ω₀={OMEGA_0:.2f} μeV"
    )
    print(f"  Grid: NE={NUM_BINS}, dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    print(f"  Ratios: {list(TAU_L_RATIOS)}  (continuation from Fermi-Dirac seed)")

    result = run()
    print(f"  τ_0^PB = {result.tau_0_pb:.4f} ns (computed numerically)")

    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
