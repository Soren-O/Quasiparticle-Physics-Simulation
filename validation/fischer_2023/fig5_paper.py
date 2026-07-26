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

import base64
import binascii
import csv
import hashlib
import inspect
import json
import os
import re
import tempfile
import zlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.density import qp_fraction

from validation import sweep_cache
from validation.fischer_2023 import fig5_solve
from validation.fischer_2023 import steady_state_certificate as certificate_module
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
    TARGET_BACKWARD_ERROR_LIMIT,
    TAU_0,
    UPPER_NBAR_VALUES,
    UPPER_T_BATH_K,
    _build_grid_and_spectral,
    _compute_tau_0_pb,
    solve,
    solver_fingerprint,
)
from validation.source_provenance import source_sha256


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
    upper_qp_residual_inf: np.ndarray
    upper_qp_backward_error: np.ndarray
    upper_phonon_residual_inf: np.ndarray
    upper_phonon_raw_backward_error: np.ndarray
    upper_phonon_backward_error: np.ndarray
    lower_qp_residual_inf: np.ndarray
    lower_qp_backward_error: np.ndarray
    lower_phonon_residual_inf: np.ndarray
    lower_phonon_raw_backward_error: np.ndarray
    lower_phonon_backward_error: np.ndarray
    # Exact returned solver states.  Artifact certification is bound to these
    # arrays; callers cannot supply certificate scalars without the state that
    # independently reproduces them.
    upper_f: np.ndarray | None = None
    lower_f: np.ndarray | None = None
    upper_n_ph: np.ndarray | None = None
    lower_n_ph: np.ndarray | None = None


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

    # Eq. 49 + Appendix E2: ε-corrected recombination, with the finite-τ_l
    # trapping correction (paper Eq. 112 leading order) on the linear term —
    # the same factor the Fig. 6 dashed derivation applies; omitting it left
    # this overlay 1.5–5.9% low (2026-07-20 review).
    R0 = 2.0 * DELTA_0 ** 2 / (tau_bar * Tc_uev ** 3)
    a_m12, a_p12, a_p32 = 2.1, 0.88, 0.77
    ratio = tau_l / tau_0_pb if tau_0_pb > 0.0 else 0.0
    trap = (1.0 + 0.5 * ratio) / (1.0 + ratio) if ratio > 0.0 else 1.0
    c1 = a_p12 / a_m12
    c2 = 1.25 * (a_p32 / a_m12) - 0.75 * (a_p12 / a_m12) ** 2
    R_bar = R0 * (1.0 + trap * c1 * x + c2 * x * x)

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
    upper_n_ph = np.asarray(raw["upper_n_ph"], dtype=float)
    lower_n_ph = np.asarray(raw["lower_n_ph"], dtype=float)

    upper_certificate_shape = (up_T.size, up_N.size)
    lower_certificate_shape = (lo_N.size, lo_T.size)
    upper_certificates = _certificate_arrays_from_raw(
        raw,
        prefix="upper",
        expected_shape=upper_certificate_shape,
    )
    lower_certificates = _certificate_arrays_from_raw(
        raw,
        prefix="lower",
        expected_shape=lower_certificate_shape,
    )

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
        upper_qp_residual_inf=upper_certificates["qp_residual_inf"],
        upper_qp_backward_error=upper_certificates["qp_backward_error"],
        upper_phonon_residual_inf=upper_certificates["phonon_residual_inf"],
        upper_phonon_raw_backward_error=upper_certificates[
            "phonon_raw_backward_error"
        ],
        upper_phonon_backward_error=upper_certificates["phonon_backward_error"],
        lower_qp_residual_inf=lower_certificates["qp_residual_inf"],
        lower_qp_backward_error=lower_certificates["qp_backward_error"],
        lower_phonon_residual_inf=lower_certificates["phonon_residual_inf"],
        lower_phonon_raw_backward_error=lower_certificates[
            "phonon_raw_backward_error"
        ],
        lower_phonon_backward_error=lower_certificates["phonon_backward_error"],
        upper_f=upper_f,
        lower_f=lower_f,
        upper_n_ph=upper_n_ph,
        lower_n_ph=lower_n_ph,
    )


def _certificate_arrays_from_raw(
    raw: Mapping[str, np.ndarray],
    *,
    prefix: str,
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Load independently computed solve certificates without a fail-open default."""
    arrays: dict[str, np.ndarray] = {}
    for field in certificate_module.CERTIFICATE_FIELDS:
        key = f"{prefix}_{field}"
        if key not in raw:
            raise ValueError(f"Fig. 5 raw solve payload is missing certificate field {key!r}.")
        values = np.asarray(raw[key], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"Fig. 5 raw certificate field {key!r} has shape {values.shape}; "
                f"expected {expected_shape}."
            )
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(
                f"Fig. 5 raw certificate field {key!r} must be finite and non-negative."
            )
        arrays[field] = values
    for field in ("qp_backward_error", "phonon_backward_error"):
        values = arrays[field]
        if values.size and float(np.max(values)) > TARGET_BACKWARD_ERROR_LIMIT:
            raise RuntimeError(
                f"Fig. 5 raw certificate {prefix}_{field} exceeds "
                f"{TARGET_BACKWARD_ERROR_LIMIT:g}."
            )
    return arrays


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
        extra_source=(
            inspect.getsource(fig5_solve)
            + inspect.getsource(certificate_module)
        ),
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


def _legacy_write_baseline(result: Fig5PaperResult, path: Path | None = None) -> Path:
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


def _legacy_read_baseline(path: Path | None = None) -> Fig5PaperResult:
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
        upper_qp_residual_inf=np.full_like(upper_x_num, np.nan),
        upper_qp_backward_error=np.full_like(upper_x_num, np.nan),
        upper_phonon_residual_inf=np.full_like(upper_x_num, np.nan),
        upper_phonon_raw_backward_error=np.full_like(upper_x_num, np.nan),
        upper_phonon_backward_error=np.full_like(upper_x_num, np.nan),
        lower_qp_residual_inf=np.full_like(lower_x_num, np.nan),
        lower_qp_backward_error=np.full_like(lower_x_num, np.nan),
        lower_phonon_residual_inf=np.full_like(lower_x_num, np.nan),
        lower_phonon_raw_backward_error=np.full_like(lower_x_num, np.nan),
        lower_phonon_backward_error=np.full_like(lower_x_num, np.nan),
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


def _legacy_read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
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
    _, _, spectral = _build_grid_and_spectral(NUM_BINS)
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


ARTIFACT_SCHEMA = "qpsim.fischer2023.fig5_paper.v2"
_LEGACY_CANONICAL_LOGICAL_SHA256 = (
    "8a16aee9fcadf05eed62b00998c7e6294a0e08b511cf87af382a15db1acf8a95"
)
_BASELINE_COLUMNS = (
    "panel",
    "T_bath_K",
    "n_bar",
    "T_star_over_delta",
    "x_qp_num",
    "x_qp_analytic",
    *certificate_module.CERTIFICATE_FIELDS,
    "state_f_f64_zlib_base64",
    "state_n_ph_f64_zlib_base64",
    "state_sha256",
)
_CERTIFIED_BACKWARD_ERROR_FIELDS = (
    "qp_backward_error",
    "phonon_backward_error",
)
_METADATA_KEYS = {
    "certificate_fields",
    "certificate_maxima",
    "certificate_metric_version",
    "certificate_target_backward_error",
    "columns",
    "fingerprint",
    "payload_sha256",
    "row_count",
    "schema",
}
_SOURCE_FINGERPRINT_FILES: tuple[str, ...] = (
    "validation/fischer_2023/fig5_paper.py",
    "validation/fischer_2023/fig5_solve.py",
    "validation/fischer_2023/steady_state_certificate.py",
    "validation/sweep_cache.py",
    "validation/source_provenance.py",
    "qpsim/backends/base.py",
    "qpsim/backends/t3_diffusion.py",
    "qpsim/collisions/_uniform_grid.py",
    "qpsim/collisions/_validation.py",
    "qpsim/collisions/pair_breaking_photon.py",
    "qpsim/collisions/phonon.py",
    "qpsim/collisions/sub_gap_photon.py",
    "qpsim/constants.py",
    "qpsim/devices/external_flux.py",
    "qpsim/grid/energy_grid.py",
    "qpsim/materials/database.py",
    "qpsim/observables/density.py",
    "qpsim/phonon_models/ph0_local.py",
    "qpsim/phonon_models/state.py",
    "qpsim/physics/bcs_quadrature.py",
    "qpsim/physics/gap_equation.py",
    "qpsim/physics/kaplan_pair_breaking.py",
    "qpsim/physics/kernels.py",
    "qpsim/physics/spectral.py",
    "qpsim/services/steady_state.py",
    "qpsim/solvers/anderson.py",
    "qpsim/solvers/coupled_newton.py",
    "qpsim/solvers/etd.py",
    "qpsim/solvers/newton_steady_state.py",
)


class LegacyArtifactError(RuntimeError):
    """Raised only for the known pre-schema canonical Fig. 5 artifact."""


class ArtifactValidationError(RuntimeError):
    """Raised when a current-schema Fig. 5 artifact is incomplete or stale."""


def source_hashes() -> dict[str, str]:
    """Hash logical validation and numerical sources defining the artifact."""
    root = Path(__file__).resolve().parents[2]
    return {
        relative: source_sha256(root / relative)
        for relative in _SOURCE_FINGERPRINT_FILES
    }


def artifact_fingerprint() -> dict[str, Any]:
    """Return the exact config, axes, solver, and source identity."""
    return {
        "axes": {
            "lower_nbar": [float(value) for value in LOWER_NBAR],
            "lower_T_bath_K": [float(value) for value in LOWER_T_BATH_K],
            "upper_nbar": [float(value) for value in UPPER_NBAR_VALUES],
            "upper_T_bath_K": [float(value) for value in UPPER_T_BATH_K],
        },
        "config": asdict(config_metadata()),
        "gap_mode": "fixed",
        "solver": solver_fingerprint(num_bins=NUM_BINS),
        "source_sha256": source_hashes(),
        "tau_l_over_tau_0_pb": 1.0,
    }


def _expected_row_count() -> int:
    return (
        len(UPPER_T_BATH_K) * UPPER_NBAR_VALUES.size
        + len(LOWER_NBAR) * LOWER_T_BATH_K.size
    )


def _certificate_arrays(
    result: Fig5PaperResult,
    prefix: str,
) -> dict[str, np.ndarray]:
    return {
        field: np.asarray(getattr(result, f"{prefix}_{field}"), dtype=float)
        for field in certificate_module.CERTIFICATE_FIELDS
    }


def _validate_axis(
    values: np.ndarray | tuple[float, ...],
    expected: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if axis.shape != expected.shape or not np.array_equal(axis, expected):
        raise ArtifactValidationError(
            f"Fig. 5 artifact {name} axis is stale or reordered."
        )
    if np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ArtifactValidationError(
            f"Fig. 5 artifact {name} axis must be finite, unique, and increasing."
        )
    return axis


def _require_close(
    claimed: np.ndarray | float,
    recomputed: np.ndarray | float,
    *,
    name: str,
) -> None:
    """Require near-bitwise agreement with an independently rebuilt value."""
    if not np.allclose(
        np.asarray(claimed, dtype=float),
        np.asarray(recomputed, dtype=float),
        rtol=128.0 * np.finfo(float).eps,
        atol=np.finfo(float).tiny,
    ):
        raise ArtifactValidationError(
            f"Fig. 5 artifact {name} does not match its persisted solver state."
        )


def _required_state_array(
    result: Fig5PaperResult,
    name: str,
    expected_shape: tuple[int, ...],
    *,
    upper_bound: float | None = None,
) -> np.ndarray:
    raw = getattr(result, name)
    if raw is None:
        raise ArtifactValidationError(
            f"Fig. 5 artifact is missing returned-state payload {name!r}."
        )
    raw_array = np.asarray(raw)
    if np.issubdtype(raw_array.dtype, np.bool_):
        raise ArtifactValidationError(
            f"Fig. 5 state payload {name!r} cannot be boolean."
        )
    values = np.asarray(raw_array, dtype=float)
    if values.shape != expected_shape or np.any(~np.isfinite(values)):
        raise ArtifactValidationError(
            f"Fig. 5 state payload {name!r} must have finite shape "
            f"{expected_shape}."
        )
    if np.any(values < 0.0) or (
        upper_bound is not None and np.any(values > upper_bound)
    ):
        domain = f"[0, {upper_bound:g}]" if upper_bound is not None else "[0, inf)"
        raise ArtifactValidationError(
            f"Fig. 5 state payload {name!r} must lie in {domain}."
        )
    return values


def _recomputed_point(
    *,
    f: np.ndarray,
    n_ph: np.ndarray,
    T_bath: float,
    n_bar: float,
    tau_l: float,
    spectral: Any,
) -> tuple[float, float, dict[str, float]]:
    """Rebuild one returned state and independently evaluate all claims."""
    state = fig5_solve._build_state(
        fig5_solve._fischer_material(),
        spectral,
        T_bath,
        tau_l,
        f_seed=f,
        n_ph_seed=n_ph,
    )
    certificate = certificate_module.steady_state_certificate(
        state,
        photon_params={"omega_0": OMEGA_0, "n_bar": n_bar, "c_phot": C_PHOT},
        tau_l=tau_l,
    )
    return (
        qp_fraction(f, spectral, delta_0=DELTA_0),
        _xqp_analytic_eq47(
            T_bath,
            n_bar,
            tau_l=tau_l,
            tau_0_pb=tau_l,
        ),
        certificate,
    )


def _validate_result_for_artifact(result: Fig5PaperResult) -> dict[str, float]:
    """Validate and bind every table/certificate claim to returned raw states."""
    upper_T = _validate_axis(
        result.upper_T_bath,
        np.asarray(UPPER_T_BATH_K, dtype=float),
        name="upper T_bath",
    )
    upper_nbar = _validate_axis(
        result.upper_nbar,
        np.asarray(UPPER_NBAR_VALUES, dtype=float),
        name="upper n_bar",
    )
    lower_nbar = _validate_axis(
        result.lower_nbar,
        np.asarray(LOWER_NBAR, dtype=float),
        name="lower n_bar",
    )
    lower_T = _validate_axis(
        result.lower_T_bath,
        np.asarray(LOWER_T_BATH_K, dtype=float),
        name="lower T_bath",
    )
    expected_tau = config_metadata().tau_0_pb_ns
    if (
        not np.isfinite(result.tau_0_pb_ns)
        or result.tau_0_pb_ns <= 0.0
        or not np.isclose(
            result.tau_0_pb_ns,
            expected_tau,
            rtol=1.0e-14,
            atol=0.0,
        )
    ):
        raise ArtifactValidationError(
            "Fig. 5 artifact tau_0_pb_ns is stale or non-physical."
        )

    upper_shape = (upper_T.size, upper_nbar.size)
    lower_shape = (lower_nbar.size, lower_T.size)
    expected_upper_T_star = np.broadcast_to(
        np.asarray(
            [_kBTstar_eq35(float(value)) / DELTA_0 for value in upper_nbar]
        ),
        upper_shape,
    )
    payloads = (
        ("upper_T_star", result.upper_T_star, upper_shape),
        ("upper_x_qp_num", result.upper_x_qp_num, upper_shape),
        ("upper_x_qp_analytic", result.upper_x_qp_analytic, upper_shape),
        ("lower_x_qp_num", result.lower_x_qp_num, lower_shape),
        ("lower_x_qp_analytic", result.lower_x_qp_analytic, lower_shape),
    )
    for name, raw_values, expected_shape in payloads:
        raw_array = np.asarray(raw_values)
        if np.issubdtype(raw_array.dtype, np.bool_):
            raise ArtifactValidationError(
                f"Fig. 5 artifact {name} cannot be boolean."
            )
        values = np.asarray(raw_array, dtype=float)
        if values.shape != expected_shape or np.any(~np.isfinite(values)):
            raise ArtifactValidationError(
                f"Fig. 5 artifact {name} must have finite shape {expected_shape}."
            )
        if name != "upper_T_star" and np.any(values < 0.0):
            raise ArtifactValidationError(
                f"Fig. 5 artifact {name} must be non-negative."
            )
    if not np.array_equal(np.asarray(result.upper_T_star), expected_upper_T_star):
        raise ArtifactValidationError(
            "Fig. 5 artifact T_star values are inconsistent with Eq. 35."
        )

    _, _, spectral = _build_grid_and_spectral(NUM_BINS)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    upper_f = _required_state_array(
        result,
        "upper_f",
        (*upper_shape, NUM_BINS),
        upper_bound=1.0,
    )
    lower_f = _required_state_array(
        result,
        "lower_f",
        (*lower_shape, NUM_BINS),
        upper_bound=1.0,
    )
    upper_n_ph = _required_state_array(
        result,
        "upper_n_ph",
        (*upper_shape, omega.size),
    )
    lower_n_ph = _required_state_array(
        result,
        "lower_n_ph",
        (*lower_shape, omega.size),
    )

    maxima: dict[str, float] = {}
    upper_certificates = _certificate_arrays(result, "upper")
    lower_certificates = _certificate_arrays(result, "lower")
    for field in certificate_module.CERTIFICATE_FIELDS:
        if np.issubdtype(
            np.asarray(getattr(result, f"upper_{field}")).dtype,
            np.bool_,
        ) or np.issubdtype(
            np.asarray(getattr(result, f"lower_{field}")).dtype,
            np.bool_,
        ):
            raise ArtifactValidationError(
                f"Fig. 5 certificate field {field!r} cannot be boolean."
            )
        upper = upper_certificates[field]
        lower = lower_certificates[field]
        if upper.shape != upper_shape or lower.shape != lower_shape:
            raise ArtifactValidationError(
                f"Fig. 5 certificate field {field!r} has the wrong panel shape."
            )
        if (
            np.any(~np.isfinite(upper))
            or np.any(~np.isfinite(lower))
            or np.any(upper < 0.0)
            or np.any(lower < 0.0)
        ):
            raise ArtifactValidationError(
                f"Fig. 5 certificate field {field!r} must be finite and non-negative."
            )
        maximum = float(max(np.max(upper), np.max(lower)))
        maxima[field] = maximum
        if (
            field in _CERTIFIED_BACKWARD_ERROR_FIELDS
            and maximum > TARGET_BACKWARD_ERROR_LIMIT
        ):
            raise ArtifactValidationError(
                f"Fig. 5 certificate field {field!r} has maximum {maximum:.3e}, "
                f"above {TARGET_BACKWARD_ERROR_LIMIT:.3e}."
            )

    tau_l = result.tau_0_pb_ns
    try:
        for i, T_bath in enumerate(upper_T):
            for j, n_bar in enumerate(upper_nbar):
                x_num, x_analytic, certificate = _recomputed_point(
                    f=upper_f[i, j],
                    n_ph=upper_n_ph[i, j],
                    T_bath=float(T_bath),
                    n_bar=float(n_bar),
                    tau_l=tau_l,
                    spectral=spectral,
                )
                _require_close(
                    result.upper_x_qp_num[i, j],
                    x_num,
                    name=f"upper x_qp_num[{i},{j}]",
                )
                _require_close(
                    result.upper_x_qp_analytic[i, j],
                    x_analytic,
                    name=f"upper x_qp_analytic[{i},{j}]",
                )
                for field in certificate_module.CERTIFICATE_FIELDS:
                    _require_close(
                        upper_certificates[field][i, j],
                        certificate[field],
                        name=f"upper {field}[{i},{j}]",
                    )
        for i, n_bar in enumerate(lower_nbar):
            for j, T_bath in enumerate(lower_T):
                x_num, x_analytic, certificate = _recomputed_point(
                    f=lower_f[i, j],
                    n_ph=lower_n_ph[i, j],
                    T_bath=float(T_bath),
                    n_bar=float(n_bar),
                    tau_l=tau_l,
                    spectral=spectral,
                )
                _require_close(
                    result.lower_x_qp_num[i, j],
                    x_num,
                    name=f"lower x_qp_num[{i},{j}]",
                )
                _require_close(
                    result.lower_x_qp_analytic[i, j],
                    x_analytic,
                    name=f"lower x_qp_analytic[{i},{j}]",
                )
                for field in certificate_module.CERTIFICATE_FIELDS:
                    _require_close(
                        lower_certificates[field][i, j],
                        certificate[field],
                        name=f"lower {field}[{i},{j}]",
                    )
    except ArtifactValidationError:
        raise
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ArtifactValidationError(
            "Fig. 5 returned-state certificate could not be recomputed."
        ) from exc
    return maxima


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Write beside the destination, fsync, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _array_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f8", order="C").tobytes(order="C")


def _encode_state_array(values: np.ndarray) -> str:
    return base64.b64encode(zlib.compress(_array_bytes(values), level=9)).decode(
        "ascii"
    )


def _state_sha256(
    *,
    panel: str,
    T_bath: float,
    n_bar: float,
    f: np.ndarray,
    n_ph: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {"T_bath_K": T_bath, "n_bar": n_bar, "panel": panel}
        ).encode("ascii")
    )
    digest.update(b"\0f\0")
    digest.update(_array_bytes(f))
    digest.update(b"\0n_ph\0")
    digest.update(_array_bytes(n_ph))
    return digest.hexdigest()


def _payload_sha256(rows: list[list[str]]) -> str:
    """Hash exact logical data rows, including table, certs, and raw states."""
    return hashlib.sha256(_canonical_json(rows).encode("ascii")).hexdigest()


def write_baseline(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Atomically write states and claims only after independent recertification."""
    if path is None:
        path = baseline_path()
    maxima = _validate_result_for_artifact(result)
    upper_f = np.asarray(result.upper_f, dtype=float)
    lower_f = np.asarray(result.lower_f, dtype=float)
    upper_n_ph = np.asarray(result.upper_n_ph, dtype=float)
    lower_n_ph = np.asarray(result.lower_n_ph, dtype=float)
    upper_certificates = _certificate_arrays(result, "upper")
    lower_certificates = _certificate_arrays(result, "lower")
    rows: list[list[str]] = []
    for i, T_bath in enumerate(result.upper_T_bath):
        for j, n_bar in enumerate(result.upper_nbar):
            state_f = upper_f[i, j]
            state_n_ph = upper_n_ph[i, j]
            rows.append(
                [
                    "upper",
                    f"{T_bath:.17e}",
                    f"{n_bar:.17e}",
                    f"{result.upper_T_star[i, j]:.17e}",
                    f"{result.upper_x_qp_num[i, j]:.17e}",
                    f"{result.upper_x_qp_analytic[i, j]:.17e}",
                    *[
                        f"{upper_certificates[field][i, j]:.17e}"
                        for field in certificate_module.CERTIFICATE_FIELDS
                    ],
                    _encode_state_array(state_f),
                    _encode_state_array(state_n_ph),
                    _state_sha256(
                        panel="upper",
                        T_bath=float(T_bath),
                        n_bar=float(n_bar),
                        f=state_f,
                        n_ph=state_n_ph,
                    ),
                ]
            )
    for i, n_bar in enumerate(result.lower_nbar):
        T_star = _kBTstar_eq35(float(n_bar)) / DELTA_0
        for j, T_bath in enumerate(result.lower_T_bath):
            state_f = lower_f[i, j]
            state_n_ph = lower_n_ph[i, j]
            rows.append(
                [
                    "lower",
                    f"{T_bath:.17e}",
                    f"{n_bar:.17e}",
                    f"{T_star:.17e}",
                    f"{result.lower_x_qp_num[i, j]:.17e}",
                    f"{result.lower_x_qp_analytic[i, j]:.17e}",
                    *[
                        f"{lower_certificates[field][i, j]:.17e}"
                        for field in certificate_module.CERTIFICATE_FIELDS
                    ],
                    _encode_state_array(state_f),
                    _encode_state_array(state_n_ph),
                    _state_sha256(
                        panel="lower",
                        T_bath=float(T_bath),
                        n_bar=float(n_bar),
                        f=state_f,
                        n_ph=state_n_ph,
                    ),
                ]
            )
    metadata = {
        "certificate_fields": list(certificate_module.CERTIFICATE_FIELDS),
        "certificate_maxima": maxima,
        "certificate_metric_version": certificate_module.CERTIFICATE_METRIC_VERSION,
        "certificate_target_backward_error": TARGET_BACKWARD_ERROR_LIMIT,
        "columns": list(_BASELINE_COLUMNS),
        "fingerprint": artifact_fingerprint(),
        "payload_sha256": _payload_sha256(rows),
        "row_count": _expected_row_count(),
        "schema": ARTIFACT_SCHEMA,
    }
    with _atomic_text_file(path) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"])
        writer.writerow([f"# qpsim_metadata={_canonical_json(metadata)}"])
        writer.writerow(_BASELINE_COLUMNS)
        writer.writerows(rows)
    return path


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArtifactValidationError(
                f"Fig. 5 metadata contains duplicate JSON key {key!r}."
            )
        value[key] = item
    return value


def _read_csv_rows(path: Path) -> list[list[str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ArtifactValidationError(
            f"Could not read Fig. 5 artifact at {path}."
        ) from exc
    if not rows or any(not row for row in rows):
        raise ArtifactValidationError(
            "Fig. 5 artifact must be non-empty and contain no blank rows."
        )
    return rows


def _is_known_legacy_artifact(rows: list[list[str]]) -> bool:
    # This digest is over parsed logical CSV rows, so newline conventions do
    # not matter, but every header and data value must match the known pinned
    # pre-schema canonical.  A merely legacy-looking corrupt file is not xfailed.
    return (
        len(rows) == _expected_row_count() + 5
        and _payload_sha256(rows) == _LEGACY_CANONICAL_LOGICAL_SHA256
    )


def _artifact_metadata(path: Path, rows: list[list[str]]) -> dict[str, Any]:
    expected_marker = [f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"]
    if rows[0] != expected_marker:
        if _is_known_legacy_artifact(rows):
            raise LegacyArtifactError(
                f"The pinned Fig. 5 artifact at {path} predates the certified "
                f"{ARTIFACT_SCHEMA} schema."
            )
        raise ArtifactValidationError(
            "Fig. 5 artifact has a missing or unsupported schema marker."
        )
    if len(rows) < 3 or len(rows[1]) != 1:
        raise ArtifactValidationError("Fig. 5 artifact metadata row is malformed.")
    prefix = "# qpsim_metadata="
    if not rows[1][0].startswith(prefix):
        raise ArtifactValidationError("Fig. 5 artifact metadata marker is missing.")
    try:
        metadata = json.loads(
            rows[1][0][len(prefix):],
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ArtifactValidationError(
                    f"Fig. 5 metadata contains non-finite JSON token {value!r}."
                )
            ),
        )
    except ArtifactValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError("Fig. 5 artifact metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        raise ArtifactValidationError(
            "Fig. 5 artifact metadata fields do not match the exact schema."
        )
    if metadata["schema"] != ARTIFACT_SCHEMA:
        raise ArtifactValidationError("Fig. 5 metadata schema is stale.")
    if metadata["columns"] != list(_BASELINE_COLUMNS):
        raise ArtifactValidationError("Fig. 5 metadata columns are stale or reordered.")
    if metadata["certificate_fields"] != list(certificate_module.CERTIFICATE_FIELDS):
        raise ArtifactValidationError("Fig. 5 certificate field list is stale.")
    if (
        metadata["certificate_metric_version"]
        != certificate_module.CERTIFICATE_METRIC_VERSION
    ):
        raise ArtifactValidationError("Fig. 5 certificate metric version is stale.")
    target = metadata["certificate_target_backward_error"]
    if (
        isinstance(target, bool)
        or not isinstance(target, (int, float))
        or not np.isfinite(target)
        or float(target) != TARGET_BACKWARD_ERROR_LIMIT
    ):
        raise ArtifactValidationError("Fig. 5 certificate target is stale or invalid.")
    row_count = metadata["row_count"]
    if isinstance(row_count, bool) or row_count != _expected_row_count():
        raise ArtifactValidationError("Fig. 5 artifact row count metadata is stale.")
    if len(rows) != int(row_count) + 3:
        raise ArtifactValidationError("Fig. 5 artifact is truncated or has extra rows.")
    if rows[2] != list(_BASELINE_COLUMNS):
        raise ArtifactValidationError("Fig. 5 CSV header is stale or reordered.")
    try:
        live_fingerprint = artifact_fingerprint()
    except (OSError, ValueError, RuntimeError) as exc:
        raise ArtifactValidationError(
            "Could not construct the live Fig. 5 artifact fingerprint."
        ) from exc
    if metadata["fingerprint"] != live_fingerprint:
        raise ArtifactValidationError(
            "Fig. 5 artifact fingerprint is stale (config, axes, solver, or source)."
        )
    payload_hash = metadata["payload_sha256"]
    if (
        not isinstance(payload_hash, str)
        or len(payload_hash) != 64
        or any(character not in "0123456789abcdef" for character in payload_hash)
        or payload_hash != _payload_sha256(rows[3:])
    ):
        raise ArtifactValidationError(
            "Fig. 5 table/certificate/state payload hash does not match."
        )
    return metadata


def _finite_float(text: str, *, name: str) -> float:
    try:
        value = float(text)
    except ValueError as exc:
        raise ArtifactValidationError(f"Fig. 5 row field {name!r} is not numeric.") from exc
    if not np.isfinite(value):
        raise ArtifactValidationError(f"Fig. 5 row field {name!r} is non-finite.")
    return value


def _decode_state_array(text: str, *, size: int, name: str) -> np.ndarray:
    expected_bytes = size * np.dtype("<f8").itemsize
    if len(text) > 2 * expected_bytes + 1024:
        raise ArtifactValidationError(f"Fig. 5 state payload {name!r} is oversized.")
    try:
        compressed = base64.b64decode(text, validate=True)
        decompressor = zlib.decompressobj()
        payload = decompressor.decompress(compressed, expected_bytes + 1)
    except (binascii.Error, zlib.error, ValueError) as exc:
        raise ArtifactValidationError(
            f"Fig. 5 state payload {name!r} is not valid compressed float64 data."
        ) from exc
    if (
        len(payload) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ArtifactValidationError(
            f"Fig. 5 state payload {name!r} has the wrong decoded size."
        )
    return np.frombuffer(payload, dtype="<f8", count=size).astype(float, copy=True)


def _read_artifact(path: Path) -> tuple[Fig5PaperResult, dict[str, Any]]:
    rows = _read_csv_rows(path)
    metadata = _artifact_metadata(path, rows)
    upper_T = np.asarray(UPPER_T_BATH_K, dtype=float)
    upper_nbar = np.asarray(UPPER_NBAR_VALUES, dtype=float)
    lower_nbar = np.asarray(LOWER_NBAR, dtype=float)
    lower_T = np.asarray(LOWER_T_BATH_K, dtype=float)
    upper_shape = (upper_T.size, upper_nbar.size)
    lower_shape = (lower_nbar.size, lower_T.size)
    _, _, spectral = _build_grid_and_spectral(NUM_BINS)
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)

    upper_T_star = np.empty(upper_shape)
    upper_x_num = np.empty(upper_shape)
    upper_x_analytic = np.empty(upper_shape)
    lower_x_num = np.empty(lower_shape)
    lower_x_analytic = np.empty(lower_shape)
    upper_certificates = {
        field: np.empty(upper_shape) for field in certificate_module.CERTIFICATE_FIELDS
    }
    lower_certificates = {
        field: np.empty(lower_shape) for field in certificate_module.CERTIFICATE_FIELDS
    }
    upper_f = np.empty((*upper_shape, NUM_BINS))
    lower_f = np.empty((*lower_shape, NUM_BINS))
    upper_n_ph = np.empty((*upper_shape, omega.size))
    lower_n_ph = np.empty((*lower_shape, omega.size))

    contexts: list[tuple[str, int, int, float, float]] = []
    contexts.extend(
        ("upper", i, j, float(T_bath), float(n_bar))
        for i, T_bath in enumerate(upper_T)
        for j, n_bar in enumerate(upper_nbar)
    )
    contexts.extend(
        ("lower", i, j, float(T_bath), float(n_bar))
        for i, n_bar in enumerate(lower_nbar)
        for j, T_bath in enumerate(lower_T)
    )
    for row, (panel, i, j, expected_T, expected_nbar) in zip(
        rows[3:], contexts, strict=True
    ):
        if len(row) != len(_BASELINE_COLUMNS) or row[0] != panel:
            raise ArtifactValidationError(
                "Fig. 5 data row has the wrong width, panel, or ordering."
            )
        T_bath = _finite_float(row[1], name="T_bath_K")
        n_bar = _finite_float(row[2], name="n_bar")
        T_star = _finite_float(row[3], name="T_star_over_delta")
        x_num = _finite_float(row[4], name="x_qp_num")
        x_analytic = _finite_float(row[5], name="x_qp_analytic")
        expected_T_star = _kBTstar_eq35(expected_nbar) / DELTA_0
        if (
            T_bath != expected_T
            or n_bar != expected_nbar
            or T_star != expected_T_star
            or x_num < 0.0
            or x_analytic < 0.0
        ):
            raise ArtifactValidationError(
                "Fig. 5 data row axes, Eq. 35 value, or physical domain is invalid."
            )
        certificate = {
            field: _finite_float(row[6 + offset], name=field)
            for offset, field in enumerate(certificate_module.CERTIFICATE_FIELDS)
        }
        if any(value < 0.0 for value in certificate.values()):
            raise ArtifactValidationError(
                "Fig. 5 certificate fields must be non-negative."
            )
        for field in _CERTIFIED_BACKWARD_ERROR_FIELDS:
            if certificate[field] > TARGET_BACKWARD_ERROR_LIMIT:
                raise ArtifactValidationError(
                    f"Fig. 5 certificate field {field!r} exceeds its gate."
                )
        state_offset = 6 + len(certificate_module.CERTIFICATE_FIELDS)
        state_f = _decode_state_array(
            row[state_offset], size=NUM_BINS, name=f"{panel}.f[{i},{j}]"
        )
        state_n_ph = _decode_state_array(
            row[state_offset + 1],
            size=omega.size,
            name=f"{panel}.n_ph[{i},{j}]",
        )
        expected_state_hash = _state_sha256(
            panel=panel,
            T_bath=T_bath,
            n_bar=n_bar,
            f=state_f,
            n_ph=state_n_ph,
        )
        if row[state_offset + 2] != expected_state_hash:
            raise ArtifactValidationError("Fig. 5 per-point state hash does not match.")
        if panel == "upper":
            upper_T_star[i, j] = T_star
            upper_x_num[i, j] = x_num
            upper_x_analytic[i, j] = x_analytic
            upper_f[i, j] = state_f
            upper_n_ph[i, j] = state_n_ph
            for field, value in certificate.items():
                upper_certificates[field][i, j] = value
        else:
            lower_x_num[i, j] = x_num
            lower_x_analytic[i, j] = x_analytic
            lower_f[i, j] = state_f
            lower_n_ph[i, j] = state_n_ph
            for field, value in certificate.items():
                lower_certificates[field][i, j] = value

    config = metadata["fingerprint"]["config"]
    result = Fig5PaperResult(
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
        upper_T_bath=upper_T,
        upper_nbar=upper_nbar,
        upper_T_star=upper_T_star,
        upper_x_qp_num=upper_x_num,
        upper_x_qp_analytic=upper_x_analytic,
        lower_nbar=lower_nbar,
        lower_T_bath=lower_T,
        lower_x_qp_num=lower_x_num,
        lower_x_qp_analytic=lower_x_analytic,
        upper_qp_residual_inf=upper_certificates["qp_residual_inf"],
        upper_qp_backward_error=upper_certificates["qp_backward_error"],
        upper_phonon_residual_inf=upper_certificates["phonon_residual_inf"],
        upper_phonon_raw_backward_error=upper_certificates[
            "phonon_raw_backward_error"
        ],
        upper_phonon_backward_error=upper_certificates["phonon_backward_error"],
        lower_qp_residual_inf=lower_certificates["qp_residual_inf"],
        lower_qp_backward_error=lower_certificates["qp_backward_error"],
        lower_phonon_residual_inf=lower_certificates["phonon_residual_inf"],
        lower_phonon_raw_backward_error=lower_certificates[
            "phonon_raw_backward_error"
        ],
        lower_phonon_backward_error=lower_certificates["phonon_backward_error"],
        upper_f=upper_f,
        lower_f=lower_f,
        upper_n_ph=upper_n_ph,
        lower_n_ph=lower_n_ph,
    )
    recomputed_maxima = _validate_result_for_artifact(result)
    stamped_maxima = metadata["certificate_maxima"]
    if not isinstance(stamped_maxima, dict) or set(stamped_maxima) != set(
        certificate_module.CERTIFICATE_FIELDS
    ):
        raise ArtifactValidationError("Fig. 5 certificate maxima metadata is invalid.")
    for field, recomputed in recomputed_maxima.items():
        stamped = stamped_maxima[field]
        if (
            isinstance(stamped, bool)
            or not isinstance(stamped, (int, float))
            or not np.isfinite(stamped)
            or float(stamped) != recomputed
        ):
            raise ArtifactValidationError(
                f"Fig. 5 stamped certificate maximum {field!r} is forged."
            )
    return result, metadata


def read_baseline(path: Path | None = None) -> Fig5PaperResult:
    """Read and independently re-certify an exact current-schema artifact."""
    resolved = baseline_path() if path is None else path
    return _read_artifact(resolved)[0]


def read_baseline_metadata(path: Path | None = None) -> BaselineMetadata:
    """Read metadata only after the complete artifact has passed validation."""
    resolved = baseline_path() if path is None else path
    _, metadata = _read_artifact(resolved)
    config = metadata["fingerprint"]["config"]
    return BaselineMetadata(
        delta_0=float(config["delta_0"]),
        tau_0=float(config["tau_0"]),
        t_c=float(config["t_c"]),
        omega_0=float(config["omega_0"]),
        c_phot=float(config["c_phot"]),
        num_bins=int(config["num_bins"]),
        e_min_factor=float(config["e_min_factor"]),
        e_max_factor=float(config["e_max_factor"]),
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
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
    print("Fischer 2023 Fig. 5 -- paper-topology reproduction ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, "
        f"omega_0={OMEGA_0:.2f} micro-eV, c_phot={C_PHOT:.0e} ns^-1"
    )
    print(
        f"  Grid: NE={NUM_BINS}, "
        f"dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} micro-eV"
    )
    print(f"  Upper panel: T_B={list(UPPER_T_BATH_K)} K, nbar in "
          f"[{UPPER_NBAR_VALUES[0]:.0e}, {UPPER_NBAR_VALUES[-1]:.0e}] "
          f"({UPPER_NBAR_VALUES.size} pts)")
    print(f"  Lower panel: nbar={list(LOWER_NBAR)}, T_B in "
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
