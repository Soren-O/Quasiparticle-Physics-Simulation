"""Fischer 2023 Fig. 5 — paper-faithful two-panel reproduction.

Two panels in the paper's sweep topology, on the paper's 1620-bin
energy grid, at the paper's nominal $\\tau_\\ell = \\tau_0^{PB}$ phonon
escape ratio.

* **Upper panel.** Vary photon number $\\bar n$, plot $x_{\\rm qp}$ vs
  $T_*/\\Delta$ at three bath temperatures $T_B \\in \\{0.10, 0.15, 0.20\\}$ K,
  with $T_*$ given by Eq. 35: $k_B T_* = (A \\bar n)^{1/6}$ where
  $A = (105/64)\\,(k_B T_c)^3\\, c_{\\rm phot}\\, \\tau_0\\, \\omega_0^2\\, \\Delta$.
* **Lower panel.** Sweep $T_B$ at three fixed $T_*/\\Delta$ values.

Solid: numerical kinetic-equation solutions (T3 backend, finite-$\\tau_\\ell$
Picard with Anderson). Dashed: analytical density balance from
:func:`_xqp_analytic_eq47` --- generalized Rothwarf-Taylor (Eq. 47) with
Appendix-E recombination corrections (Eq. E2).

Cross-checked against the standalone reproduction at
``paper reproductions/fischer2023-repro/src/fischer2023/solver.py``;
see :mod:`test_fig5_paper_eq47`.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:
parameters identical to :mod:`fig3_paper`.

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig5_paper
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
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── Fischer 2023 Table I parameters (shared with fig3_paper) ─────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0
C_PHOT = 1e-9

# Paper grid: 1620 bins, dE = 1 μeV (same as fig3_paper.py).
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1620

# Eq. 35 prefactor (μeV^6) — A·n̄ has units μeV^6, (A·n̄)^(1/6) is in μeV.
_A_EQ35 = (
    (105.0 / 64.0)
    * (KB_UEV_PER_K * T_C) ** 3
    * C_PHOT
    * TAU_0
    * OMEGA_0 ** 2
    * DELTA_0
)

# Upper panel — three bath temperatures, swept over n̄ chosen so the
# T_*/Δ axis (Eq. 35) covers the paper's plotted range [0.30, 0.95].
UPPER_T_BATH_K: tuple[float, ...] = (0.10, 0.15, 0.20)
UPPER_T_STAR_OVER_DELTA: np.ndarray = np.linspace(0.30, 0.95, 14)
UPPER_NBAR_VALUES: np.ndarray = (UPPER_T_STAR_OVER_DELTA * DELTA_0) ** 6 / _A_EQ35

# Lower panel — sweep T_B at three FIXED T_*/Δ values. The paper plots
# fixed T_*/Δ; under Eq. 35 with A independent of T_B, fixed T_*/Δ
# corresponds to a fixed n̄, so the per-T_*/Δ continuation is just a
# T_B sweep at the corresponding n̄.
LOWER_T_STAR_OVER_DELTA: tuple[float, ...] = (0.50, 0.66, 0.74)
LOWER_NBAR: tuple[float, ...] = tuple(
    float((t * DELTA_0) ** 6 / _A_EQ35) for t in LOWER_T_STAR_OVER_DELTA
)
LOWER_T_BATH_K: np.ndarray = np.linspace(0.10, 0.40, 13)

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the numerical tau_0^PB diverges from the paper-quoted 255 ps."""


@dataclass(frozen=True)
class Fig5PaperResult:
    """Arrays returned by :func:`run`."""

    tau_0_pb_ns: float
    # Upper panel: shape (n_T_bath, n_nbar) for each.
    upper_T_bath: np.ndarray
    upper_nbar: np.ndarray
    upper_T_star: np.ndarray         # T_* / Δ (per (T_B, nbar))
    upper_x_qp_num: np.ndarray
    upper_x_qp_analytic: np.ndarray
    # Lower panel: shape (n_nbar, n_T_bath) for each.
    lower_nbar: np.ndarray
    lower_T_bath: np.ndarray
    lower_x_qp_num: np.ndarray
    lower_x_qp_analytic: np.ndarray


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,  # F&C 2023 Table I: τ_0^PB = 255 ps
    )


def _build_grid_and_spectral() -> tuple[np.ndarray, np.ndarray, SpectralContext]:
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
            f"Fischer Fig. 5 paper grid not commensurate: ω_0={OMEGA_0}, m·dE={m*dE_scalar}"
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _compute_tau_0_pb(spectral: SpectralContext) -> float:
    """Same definition as :func:`fig3_paper._compute_tau_0_pb`."""
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


def _kBTstar_eq35(n_bar: float) -> float:
    """Fischer 2023 Eq. 35: $k_B T_* = (A\\bar n)^{1/6}$, in μeV.

    Same expression as :func:`fig6_gap_suppression._kBTstar_analytic`.
    Reuses the module-level :data:`_A_EQ35` prefactor.
    """
    if n_bar <= 0:
        return 0.0
    return float((_A_EQ35 * n_bar) ** (1.0 / 6.0))


def _xqp_analytic_eq47(
    T_bath: float,
    n_bar: float,
    *,
    tau_l: float,
    tau_0_pb: float,
) -> float:
    """Fischer 2023 Eq. 47 — generalized Rothwarf-Taylor balance with
    Appendix-E recombination corrections.

    Solves R̄ N² − G(x) N − G_T = 0  (x ≡ T_*/Δ_0) for the QP density N
    and returns qpsim's x_qp = N / (4 ρ_F Δ_0) at ρ_F = 1.

    Coefficients (units: time in ns, energies in μeV, k_B = ℏ = 1):

      τ̄    = τ_0 (1 + τ_l/τ_0^PB)                                 (Eq. 47)
      G_T  = (16π / τ̄) (Δ/T_c)³ T_B exp(−2Δ/T_B)                  (Eq. 48)
      R    = 2 Δ² / (τ̄ T_c³)                                      (Eq. 49)
      G(x) = (γ/τ̄)(τ_l/τ_0^PB)(Δ/T_c)³ x^{9/2}
                     exp(−√(14/5) x^{−3}),  γ ≈ 0.84               (Eq. 51)
      R̄/R  = 1 + c₁ ε + c₂ ε², ε = T_*/Δ_0,                       (Eq. E2)
             c₁ = a_{1/2}/a_{−1/2},
             c₂ = (5/4)(a_{3/2}/a_{−1/2}) − (3/4)(a_{1/2}/a_{−1/2})²,
             (a_{−1/2}, a_{1/2}, a_{3/2}) = (2.1, 0.88, 0.77).

    Cross-reference: ``paper reproductions/fischer2023-repro/src/
    fischer2023/solver.py`` (``nqp_steady`` / ``R_bar`` / ``G_thermal``
    / ``G_drive``); identical algebra at ρ_F = 1.

    Thermal sanity check: at n̄ = 0 this reduces to
    x_qp = √(π T_B / (2 Δ)) · exp(−Δ/T_B), matching ``qp_fraction``
    applied to a Fermi-Dirac distribution at T_B (Fischer Eq. 4).
    """
    if T_bath <= 0.0 and n_bar <= 0.0:
        return 0.0

    Tc_uev = T_C * KB_UEV_PER_K
    TB_uev = KB_UEV_PER_K * T_bath
    eps_pb = tau_l / tau_0_pb if tau_0_pb > 0.0 else 0.0
    tau_bar = TAU_0 * (1.0 + eps_pb)
    delta_over_Tc_cubed = (DELTA_0 / Tc_uev) ** 3

    # Eq. 48: thermal generation
    G_T = (
        (16.0 * np.pi / tau_bar)
        * delta_over_Tc_cubed
        * TB_uev
        * np.exp(-2.0 * DELTA_0 / TB_uev)
        if TB_uev > 0.0
        else 0.0
    )

    # Eq. 51: photon-driven generation
    x = _kBTstar_eq35(n_bar) / DELTA_0
    if tau_l > 0.0 and x > 0.0:
        G_drive = (
            (0.84 / tau_bar)
            * (tau_l / tau_0_pb)
            * delta_over_Tc_cubed
            * x ** 4.5
            * np.exp(-np.sqrt(14.0 / 5.0) * x ** (-3.0))
        )
    else:
        G_drive = 0.0

    # Eq. 49 + Appendix E2: ε-corrected recombination
    R0 = 2.0 * DELTA_0 ** 2 / (tau_bar * Tc_uev ** 3)
    a_m12, a_p12, a_p32 = 2.1, 0.88, 0.77
    c1 = a_p12 / a_m12
    c2 = 1.25 * (a_p32 / a_m12) - 0.75 * (a_p12 / a_m12) ** 2
    R_bar = R0 * (1.0 + c1 * x + c2 * x * x)

    if R_bar <= 0.0 or (G_drive == 0.0 and G_T == 0.0):
        return 0.0

    disc = G_drive * G_drive + 4.0 * R_bar * G_T
    N = (G_drive + np.sqrt(disc)) / (2.0 * R_bar)
    return float(N / (4.0 * DELTA_0))


def _build_state(
    material: Material,
    spectral: SpectralContext,
    T_bath: float,
    tau_l_scalar: float,
    *,
    f_seed: np.ndarray | None = None,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    if n_ph_seed is None:
        n_ph_seed = thermal_phonon_occupation(omega, T_bath)
    phonon = PhononState(
        n_ph=n_ph_seed.reshape(1, -1, 1).copy(),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_scalar),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    if f_seed is None:
        kT = KB_UEV_PER_K * T_bath
        f_seed = 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_seed.copy(),
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _solve_picard(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    mixing: float = 0.30,
) -> T3DiffusionState:
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


def _check_tau_0_pb(tau_0_pb: float) -> None:
    tau_ps = tau_0_pb * 1000.0
    print(f"  τ_0^PB (phonon-side extracted)       = {tau_0_pb:.4f} ns "
          f"({tau_ps:.1f} ps)")
    print(f"  Paper-quoted τ_0^PB                   ≈ {PAPER_TAU_0_PB_PS:.0f} ps")
    ratio = tau_ps / PAPER_TAU_0_PB_PS
    if ratio > TAU_0_PB_WARN_FACTOR or ratio < 1.0 / TAU_0_PB_WARN_FACTOR:
        print(
            f"  ⚠ τ_0^PB normalization mismatch: extracted/paper = {ratio:.2f}×.",
            flush=True,
        )


def _solve_upper_panel(
    backend: T3DiffusionBackend,
    material: Material,
    spectral: SpectralContext,
    tau_l: float,
    tau_0_pb: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upper panel: x_qp vs T_*/Δ, sweep n̄ at three T_B.

    Returns (T_star_over_delta, x_qp_num, x_qp_analytic), each shape
    (len(UPPER_T_BATH_K), len(UPPER_NBAR_VALUES)).
    """
    n_T = len(UPPER_T_BATH_K)
    n_n = UPPER_NBAR_VALUES.size
    T_star = np.zeros((n_T, n_n))
    x_num = np.zeros((n_T, n_n))
    x_ana = np.zeros((n_T, n_n))

    for i, T_bath in enumerate(UPPER_T_BATH_K):
        f_seed: np.ndarray | None = None
        n_ph_seed: np.ndarray | None = None
        # n̄ continuation: warm-start each step from the previous-n̄
        # converged (f, n_ph). At the lowest n̄ the drive is negligible
        # and the solution is essentially f_FD, which makes the start
        # very cheap.
        for j, n_bar in enumerate(UPPER_NBAR_VALUES):
            state = _build_state(
                material, spectral, T_bath, tau_l,
                f_seed=f_seed, n_ph_seed=n_ph_seed,
            )
            photon_params = {
                "omega_0": OMEGA_0, "n_bar": float(n_bar), "c_phot": C_PHOT,
            }
            converged = _solve_picard(backend, state, photon_params)
            x_num[i, j] = qp_fraction(
                converged.f, converged.spectral, delta_0=DELTA_0,
            )
            x_ana[i, j] = _xqp_analytic_eq47(
                T_bath, float(n_bar), tau_l=tau_l, tau_0_pb=tau_0_pb,
            )
            T_star[i, j] = _kBTstar_eq35(float(n_bar)) / DELTA_0
            f_seed = converged.f.copy()
            n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()
            print(
                f"  upper  T_B={T_bath:.2f} K  n̄={n_bar:.2e}  "
                f"T_*/Δ={T_star[i, j]:.3f}  x_qp(num)={x_num[i, j]:.3e}",
                flush=True,
            )

    return T_star, x_num, x_ana


def _solve_lower_panel(
    backend: T3DiffusionBackend,
    material: Material,
    spectral: SpectralContext,
    tau_l: float,
    tau_0_pb: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Lower panel: x_qp vs T_B, sweep T_B at three n̄ values.

    Returns (x_qp_num, x_qp_analytic), each shape
    (len(LOWER_NBAR), len(LOWER_T_BATH_K)).
    """
    n_n = len(LOWER_NBAR)
    n_T = LOWER_T_BATH_K.size
    x_num = np.zeros((n_n, n_T))
    x_ana = np.zeros((n_n, n_T))

    for i, n_bar in enumerate(LOWER_NBAR):
        f_seed: np.ndarray | None = None
        n_ph_seed: np.ndarray | None = None
        # T_B continuation low → high. At the lowest T_B the drive
        # dominates and convergence is hardest; warm-start from the
        # adjacent T_B's converged state to keep Picard in-basin.
        for j, T_bath in enumerate(LOWER_T_BATH_K):
            state = _build_state(
                material, spectral, float(T_bath), tau_l,
                f_seed=f_seed, n_ph_seed=n_ph_seed,
            )
            photon_params = {
                "omega_0": OMEGA_0, "n_bar": float(n_bar), "c_phot": C_PHOT,
            }
            converged = _solve_picard(backend, state, photon_params)
            x_num[i, j] = qp_fraction(
                converged.f, converged.spectral, delta_0=DELTA_0,
            )
            x_ana[i, j] = _xqp_analytic_eq47(
                float(T_bath), float(n_bar), tau_l=tau_l, tau_0_pb=tau_0_pb,
            )
            f_seed = converged.f.copy()
            n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()
            print(
                f"  lower  n̄={n_bar:.2e}  T_B={T_bath:.3f} K  "
                f"x_qp(num)={x_num[i, j]:.3e}",
                flush=True,
            )

    return x_num, x_ana


def run() -> Fig5PaperResult:
    """Solve Fischer Fig. 5 — both panels, τ_l = τ_0^PB."""
    material = _fischer_material()
    _, _, spectral = _build_grid_and_spectral()

    tau_0_pb = _compute_tau_0_pb(spectral)
    _check_tau_0_pb(tau_0_pb)
    tau_l = 1.0 * tau_0_pb  # paper: τ_ℓ = τ_0^PB throughout Fig. 5

    backend = T3DiffusionBackend()

    print("Upper panel — sweep n̄ at three T_B:")
    upper_T_star, upper_x_num, upper_x_ana = _solve_upper_panel(
        backend, material, spectral, tau_l, tau_0_pb,
    )

    print("Lower panel — sweep T_B at three n̄:")
    lower_x_num, lower_x_ana = _solve_lower_panel(
        backend, material, spectral, tau_l, tau_0_pb,
    )

    return Fig5PaperResult(
        tau_0_pb_ns=tau_0_pb,
        upper_T_bath=np.array(UPPER_T_BATH_K),
        upper_nbar=UPPER_NBAR_VALUES,
        upper_T_star=upper_T_star,
        upper_x_qp_num=upper_x_num,
        upper_x_qp_analytic=upper_x_ana,
        lower_nbar=np.array(LOWER_NBAR),
        lower_T_bath=LOWER_T_BATH_K,
        lower_x_qp_num=lower_x_num,
        lower_x_qp_analytic=lower_x_ana,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer_fig5_paper.csv``; the CSV includes both numerical
    curves and the Eq. 47 + Appendix-E analytical overlay.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer_fig5_paper.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def plot_path_a() -> Path:
    return baseline_path().with_name(baseline_path().stem + "_a.pdf")


def plot_path_b() -> Path:
    return baseline_path().with_name(baseline_path().stem + "_b.pdf")


# Aluminum density of states at the Fermi level used by paper Fig. 5 axis
# conversions. Same value as the standalone paper reproduction
# (figures/fig5a.py): rho_F ~= 1.74e4 / (micro-eV micro-m^3).
RHOF_AL_uev = 1.74e4
NQP_PER_X_QP_QPSIM = 4.0 * RHOF_AL_uev * DELTA_0
NQP_PER_X_QP_PAPER = 2.0 * RHOF_AL_uev * DELTA_0


_TAU_0_PB_RE = re.compile(r"tau_0_pb_ns=([\deE.+-]+)")


def write_baseline(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Write both panels to a single CSV with a `panel` column."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "# Fischer 2023 Fig. 5 — paper-topology reproduction"
        ])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow([f"# tau_0_pb_ns={result.tau_0_pb_ns}  tau_l = 1.0 * tau_0_pb"])
        writer.writerow([
            "panel", "T_bath_K", "n_bar", "T_star_over_delta",
            "x_qp_num", "x_qp_analytic",
        ])
        # Upper panel rows.
        for i, T_bath in enumerate(result.upper_T_bath):
            for j, n_bar in enumerate(result.upper_nbar):
                writer.writerow([
                    "upper",
                    f"{T_bath:.17e}",
                    f"{n_bar:.17e}",
                    f"{result.upper_T_star[i, j]:.17e}",
                    f"{result.upper_x_qp_num[i, j]:.17e}",
                    f"{result.upper_x_qp_analytic[i, j]:.17e}",
                ])
        # Lower panel rows (T_*/Δ undefined for x-axis; store NaN).
        for i, n_bar in enumerate(result.lower_nbar):
            for j, T_bath in enumerate(result.lower_T_bath):
                writer.writerow([
                    "lower",
                    f"{T_bath:.17e}",
                    f"{n_bar:.17e}",
                    "nan",
                    f"{result.lower_x_qp_num[i, j]:.17e}",
                    f"{result.lower_x_qp_analytic[i, j]:.17e}",
                ])
    return path


def read_baseline(path: Path | None = None) -> Fig5PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig5PaperResult`.

    Reconstructs the (n_T, n_n) and (n_n, n_T) panel arrays from the
    flat row-per-point layout written by :func:`write_baseline`.
    """
    if path is None:
        path = baseline_path()
    tau_0_pb: float | None = None
    upper_rows: list[tuple[float, float, float, float, float]] = []
    lower_rows: list[tuple[float, float, float, float]] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# tau_0_pb_ns"):
                m_tau = _TAU_0_PB_RE.search(first)
                if m_tau:
                    tau_0_pb = float(m_tau.group(1))
                continue
            if first.startswith("#") or first == "panel":
                continue
            panel = first
            T_bath = float(line[1])
            n_bar = float(line[2])
            T_star = float(line[3]) if line[3] != "nan" else float("nan")
            x_num = float(line[4])
            x_ana = float(line[5])
            if panel == "upper":
                upper_rows.append((T_bath, n_bar, T_star, x_num, x_ana))
            elif panel == "lower":
                lower_rows.append((T_bath, n_bar, x_num, x_ana))
            else:
                raise RuntimeError(f"Unknown panel tag: {panel!r}")
    if tau_0_pb is None:
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb_ns metadata.")

    # Reconstruct upper-panel (n_T, n_n) arrays.
    upper_T_bath_unique = sorted({r[0] for r in upper_rows})
    upper_nbar_unique = sorted({r[1] for r in upper_rows})
    n_T_up = len(upper_T_bath_unique)
    n_n_up = len(upper_nbar_unique)
    T_idx = {t: i for i, t in enumerate(upper_T_bath_unique)}
    n_idx = {n: i for i, n in enumerate(upper_nbar_unique)}
    upper_T_star = np.full((n_T_up, n_n_up), np.nan)
    upper_x_num = np.full((n_T_up, n_n_up), np.nan)
    upper_x_ana = np.full((n_T_up, n_n_up), np.nan)
    for T_bath, n_bar, T_star, x_num, x_ana in upper_rows:
        i, j = T_idx[T_bath], n_idx[n_bar]
        upper_T_star[i, j] = T_star
        upper_x_num[i, j] = x_num
        upper_x_ana[i, j] = x_ana

    # Reconstruct lower-panel (n_n, n_T) arrays.
    lower_nbar_unique = sorted({r[1] for r in lower_rows})
    lower_T_unique = sorted({r[0] for r in lower_rows})
    n_n_lo = len(lower_nbar_unique)
    n_T_lo = len(lower_T_unique)
    lT_idx = {t: i for i, t in enumerate(lower_T_unique)}
    ln_idx = {n: i for i, n in enumerate(lower_nbar_unique)}
    lower_x_num = np.full((n_n_lo, n_T_lo), np.nan)
    lower_x_ana = np.full((n_n_lo, n_T_lo), np.nan)
    for T_bath, n_bar, x_num, x_ana in lower_rows:
        i, j = ln_idx[n_bar], lT_idx[T_bath]
        lower_x_num[i, j] = x_num
        lower_x_ana[i, j] = x_ana

    return Fig5PaperResult(
        tau_0_pb_ns=tau_0_pb,
        upper_T_bath=np.array(upper_T_bath_unique),
        upper_nbar=np.array(upper_nbar_unique),
        upper_T_star=upper_T_star,
        upper_x_qp_num=upper_x_num,
        upper_x_qp_analytic=upper_x_ana,
        lower_nbar=np.array(lower_nbar_unique),
        lower_T_bath=np.array(lower_T_unique),
        lower_x_qp_num=lower_x_num,
        lower_x_qp_analytic=lower_x_ana,
    )


def _x_qp_qpsim_to_nqp(x_qp: np.ndarray | float) -> np.ndarray | float:
    """Convert qpsim x_qp = N/(4 rho_F Delta) to N [1/micro-m^3]."""
    return NQP_PER_X_QP_QPSIM * x_qp


def _nqp_to_x_qp_paper(n_qp: np.ndarray | float) -> np.ndarray | float:
    """Convert N [1/micro-m^3] to paper x_qp = N/(2 rho_F Delta)."""
    return n_qp / NQP_PER_X_QP_PAPER


def _twin_paper_x_qp_axis(ax) -> None:
    """Mirror left N axis with the paper-convention x_qp on the right."""
    ax2 = ax.twinx()
    ax2.set_yscale("log")
    lo, hi = ax.get_ylim()
    ax2.set_ylim(_nqp_to_x_qp_paper(lo), _nqp_to_x_qp_paper(hi))
    ax2.set_ylabel(r"$N/(2\Delta\rho_F)$")


# Paper Fig. 5 palette (matches standalone reproduction figures/fig5a,b.py):
# matplotlib defaults C2 (green), C0 (blue), C3 (red), low → high parameter.
_FIG5_COLORS = ["C2", "C0", "C3"]


def write_plot_a(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Fig. 5(a): N vs T_*/Delta with paper-convention x_qp on the right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path_a()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for i, T_bath in enumerate(result.upper_T_bath):
        color = _FIG5_COLORS[i]
        ax.semilogy(
            result.upper_T_star[i], _x_qp_qpsim_to_nqp(result.upper_x_qp_num[i]),
            "-", color=color, lw=1.5,
            label=rf"$T_B = {T_bath:g}$ K",
        )
        ax.semilogy(
            result.upper_T_star[i], _x_qp_qpsim_to_nqp(result.upper_x_qp_analytic[i]),
            color=color, ls=(0, (5, 2)), lw=1.3, zorder=4,
        )
    ax.set_xlabel(r"$T_*/\Delta$")
    ax.set_ylabel(r"$N\;(1/\mu m^3)$")
    ax.set_xlim(0.30, 0.95)
    ax.set_ylim(1e-3, 1e5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    _twin_paper_x_qp_axis(ax)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_plot_b(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Fig. 5(b): N vs T_B with paper-convention x_qp on the right."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path_b()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for i, t_star in enumerate(LOWER_T_STAR_OVER_DELTA):
        color = _FIG5_COLORS[i]
        ax.semilogy(
            result.lower_T_bath, _x_qp_qpsim_to_nqp(result.lower_x_qp_num[i]),
            "-", color=color, lw=1.5,
            label=rf"$T_*/\Delta = {t_star:g}$",
        )
        ax.semilogy(
            result.lower_T_bath, _x_qp_qpsim_to_nqp(result.lower_x_qp_analytic[i]),
            color=color, ls=(0, (5, 2)), lw=1.3, zorder=4,
        )
    ax.set_xlabel(r"$T_B$ (K)")
    ax.set_ylabel(r"$N\;(1/\mu m^3)$")
    ax.set_xlim(0.08, 0.40)
    ax.set_ylim(1e-2, 1e5)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=9, loc="lower right")
    _twin_paper_x_qp_axis(ax)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def write_plot(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Write both single-panel PDFs (5a, 5b) from one solver run.

    Returns the (5a) path for backward compatibility with callers that
    expect a single Path; the (5b) path is :func:`plot_path_b`.
    """
    a = write_plot_a(result)
    write_plot_b(result)
    return a


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig. 5 — paper-topology reproduction ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, ω_0={OMEGA_0:.2f} μeV, "
        f"c_phot={C_PHOT:.0e} ns⁻¹"
    )
    print(f"  Grid: NE={NUM_BINS}, dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    print(f"  Upper panel: T_B={list(UPPER_T_BATH_K)} K, n̄ ∈ "
          f"[{UPPER_NBAR_VALUES[0]:.0e}, {UPPER_NBAR_VALUES[-1]:.0e}] "
          f"({UPPER_NBAR_VALUES.size} pts)")
    print(f"  Lower panel: n̄={list(LOWER_NBAR)}, T_B ∈ "
          f"[{LOWER_T_BATH_K[0]:.3f}, {LOWER_T_BATH_K[-1]:.3f}] K "
          f"({LOWER_T_BATH_K.size} pts)")
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
