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
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.density import qp_fraction

from validation import sweep_cache
from validation.fischer_2023 import fig5_solve
from validation.fischer_2023.fig5_solve import (
    _A_EQ35,
    C_PHOT,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    LOWER_NBAR,
    LOWER_T_BATH_K,
    LOWER_T_STAR_OVER_DELTA,
    NUM_BINS,
    OMEGA_0,
    T_C,
    TAU_0,
    UPPER_NBAR_VALUES,
    UPPER_T_BATH_K,
    _build_grid_and_spectral,
    _compute_tau_0_pb,
    solve,
    solver_fingerprint,
)


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


def _kBTstar_eq35(n_bar: float) -> float:
    """Fischer 2023 Eq. 35: $k_B T_* = (A\\bar n)^{1/6}$, in μeV.

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


def observables(raw: Mapping[str, np.ndarray]) -> Fig5PaperResult:
    """Derive x_qp / Eq.-47 overlay / Eq.-35 T_* axis from a raw solve payload.

    The cheap downstream half of Fig. 5: rebuilds the (deterministic) spectral
    grid and evaluates ``qp_fraction`` per converged f, plus the analytic
    overlays. A pure function of ``raw`` — the cache leaves it uncached so editing
    it (or the analytic helpers / plots below) never triggers a re-solve.
    """
    num_bins = int(np.asarray(raw["num_bins"]).reshape(-1)[0])
    tau_0_pb = float(np.asarray(raw["tau_0_pb_ns"]).reshape(-1)[0])
    tau_l = float(np.asarray(raw["tau_l_ns"]).reshape(-1)[0])
    up_T = np.asarray(raw["upper_T_bath"], dtype=float)
    up_N = np.asarray(raw["upper_nbar"], dtype=float)
    lo_N = np.asarray(raw["lower_nbar"], dtype=float)
    lo_T = np.asarray(raw["lower_T_bath"], dtype=float)
    upper_f = np.asarray(raw["upper_f"], dtype=float)
    lower_f = np.asarray(raw["lower_f"], dtype=float)

    _, _, spectral = _build_grid_and_spectral(num_bins)

    upper_T_star = np.zeros((up_T.size, up_N.size))
    upper_x_num = np.zeros_like(upper_T_star)
    upper_x_ana = np.zeros_like(upper_T_star)
    for i in range(up_T.size):
        for j in range(up_N.size):
            upper_x_num[i, j] = qp_fraction(upper_f[i, j], spectral, delta_0=DELTA_0)
            upper_x_ana[i, j] = _xqp_analytic_eq47(
                float(up_T[i]), float(up_N[j]), tau_l=tau_l, tau_0_pb=tau_0_pb,
            )
            upper_T_star[i, j] = _kBTstar_eq35(float(up_N[j])) / DELTA_0

    lower_x_num = np.zeros((lo_N.size, lo_T.size))
    lower_x_ana = np.zeros_like(lower_x_num)
    for i in range(lo_N.size):
        for j in range(lo_T.size):
            lower_x_num[i, j] = qp_fraction(lower_f[i, j], spectral, delta_0=DELTA_0)
            lower_x_ana[i, j] = _xqp_analytic_eq47(
                float(lo_T[j]), float(lo_N[i]), tau_l=tau_l, tau_0_pb=tau_0_pb,
            )

    return Fig5PaperResult(
        tau_0_pb_ns=tau_0_pb,
        upper_T_bath=up_T,
        upper_nbar=up_N,
        upper_T_star=upper_T_star,
        upper_x_qp_num=upper_x_num,
        upper_x_qp_analytic=upper_x_ana,
        lower_nbar=lo_N,
        lower_T_bath=lo_T,
        lower_x_qp_num=lower_x_num,
        lower_x_qp_analytic=lower_x_ana,
    )


def run(
    *,
    num_bins: int = NUM_BINS,
    upper_T_bath: tuple[float, ...] = UPPER_T_BATH_K,
    upper_nbar: np.ndarray | tuple[float, ...] = UPPER_NBAR_VALUES,
    lower_nbar: tuple[float, ...] = LOWER_NBAR,
    lower_T_bath: np.ndarray | tuple[float, ...] = LOWER_T_BATH_K,
) -> Fig5PaperResult:
    """Solve both panels and derive observables — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this (no args) so it always truly recomputes against the pinned
    baseline; the cached dev / regen path is :func:`run_cached`.
    """
    return observables(
        solve(
            num_bins=num_bins,
            upper_T_bath=upper_T_bath,
            upper_nbar=upper_nbar,
            lower_nbar=lower_nbar,
            lower_T_bath=lower_T_bath,
        )
    )


def run_cached(
    *,
    num_bins: int = NUM_BINS,
    upper_T_bath: tuple[float, ...] = UPPER_T_BATH_K,
    upper_nbar: np.ndarray | tuple[float, ...] = UPPER_NBAR_VALUES,
    lower_nbar: tuple[float, ...] = LOWER_NBAR,
    lower_T_bath: np.ndarray | tuple[float, ...] = LOWER_T_BATH_K,
) -> Fig5PaperResult:
    """Like :func:`run`, but the expensive two-panel solve is served from the
    disk cache when nothing solve-relevant has changed (see
    :mod:`validation.sweep_cache`). Used by the regen / ``__main__`` path; editing
    the observables / analytic overlays / plotting here does not invalidate the
    cached solve. Disable with ``QPSIM_SWEEP_CACHE=0``.
    """
    kwargs = {
        "num_bins": int(num_bins),
        "upper_T_bath": [float(x) for x in upper_T_bath],
        "upper_nbar": [float(x) for x in np.asarray(upper_nbar, dtype=float)],
        "lower_nbar": [float(x) for x in lower_nbar],
        "lower_T_bath": [float(x) for x in np.asarray(lower_T_bath, dtype=float)],
    }
    raw = sweep_cache.cached_solve(
        "fischer_2023/fig5",
        lambda: solve(
            num_bins=num_bins,
            upper_T_bath=upper_T_bath,
            upper_nbar=upper_nbar,
            lower_nbar=lower_nbar,
            lower_T_bath=lower_T_bath,
        ),
        fingerprint=solver_fingerprint(num_bins=num_bins),
        kwargs=kwargs,
        extra_source=inspect.getsource(fig5_solve),
    )
    return observables(raw)


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
_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_c": re.compile(r"T_c=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
}


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


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config)
    without touching the data rows or running the two-panel sweep.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds rather than after the multi-minute run
    (see :mod:`fig6_paper`, where the same pattern saves ~14 h). The four sweep
    axes are compared separately against the baseline data rows.
    """

    delta_0: float
    tau_0: float
    t_c: float
    omega_0: float
    c_phot: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    tau_0_pb_ns: float


def read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve). Raises
    ``RuntimeError`` if any stamped field is missing — an old/malformed header
    should fail loudly rather than silently skip the check.
    """
    if path is None:
        path = baseline_path()
    text = path.read_text()

    def _num(rx: re.Pattern[str], field: str) -> float:
        m = rx.search(text)
        if m is None:
            raise RuntimeError(
                f"Baseline header at {path} missing {field} metadata."
            )
        return float(m.group(1))

    ne_m = _GRID_NE_RE.search(text)
    if ne_m is None:
        raise RuntimeError(f"Baseline header at {path} missing NE metadata.")
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_c=_num(_HEADER_PARAM_RE["t_c"], "T_c"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        num_bins=int(ne_m.group(1)),
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        tau_0_pb_ns=_num(_TAU_0_PB_RE, "tau_0_pb_ns"),
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — computed without the (multi-minute) two-panel sweep.

    ``tau_0_pb_ns`` is produced by the exact :func:`_compute_tau_0_pb` call
    :func:`run` makes, so it can never drift from a real run; everything else
    is read straight off the module constants.
    """
    _, _, spectral = _build_grid_and_spectral()
    tau_0_pb = _compute_tau_0_pb(spectral)
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_c=T_C,
        omega_0=OMEGA_0,
        c_phot=C_PHOT,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        tau_0_pb_ns=tau_0_pb,
    )


def _x_qp_qpsim_to_nqp(x_qp: np.ndarray | float) -> np.ndarray | float:
    """Convert qpsim x_qp = N/(4 rho_F Delta) to N [1/micro-m^3]."""
    return NQP_PER_X_QP_QPSIM * x_qp


def _nqp_to_x_qp_paper(n_qp: np.ndarray | float) -> np.ndarray | float:
    """Convert N [1/micro-m^3] to paper x_qp = N/(2 rho_F Delta)."""
    return n_qp / NQP_PER_X_QP_PAPER


def _twin_paper_x_qp_axis(ax: Any) -> None:
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
    result = run_cached()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
