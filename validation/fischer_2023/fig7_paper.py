"""Fischer 2023 Fig. 7 paper-facing Q_i(T_B) validation.

Uses the experimental-comparison parameters from Tables II/III:

* Delta_0 = 189 micro-eV, tau_0 = 63 ns, omega_0 = 22 micro-eV
* c_phot = 0.06 Hz, tau_l = 170 ps, alpha = 0.13
* h = Delta_T / 189, E_max = 10 Delta_T
* nbar set by the Table III Tstar,0/Delta values
* plotted quality factor capped by Eq. (65) with Table III Q_i,ext

The quasiparticle kinetic solve is done by the qpsim T3 backend with the
finite-tau_l phonon field.  The intrinsic quasiparticle Q_i is evaluated
with the same leading Fischer Eq. (57) form used by the standalone paper
reproduction:

    Q_i,qp = pi Delta / (omega_0 alpha sigma_1).

Usage:

    python -m validation.fischer_2023.fig7_paper
"""

from __future__ import annotations

import csv
import inspect
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.ac_conductivity import compute_ac_conductivity

from validation import sweep_cache
from validation.fischer_2023 import fig7_solve
from validation.fischer_2023.fig7_solve import (
    C_PHOT,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    NUM_BINS,
    OMEGA_0,
    P_READ_DBM,
    T_BATH_VALUES,
    T_C,
    TAU_0,
    TAU_0_PB,
    TAU_L,
    TSTAR_OVER_DELTA,
    _build_grid,
    solve,
    solver_fingerprint,
)

_CACHE_ROOT = Path(tempfile.gettempdir()) / "qpsim-cache"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

# Observable-only constants — they do NOT affect the kinetic solve, so they live
# here rather than in fig7_solve: editing them keeps a cached solve warm. ALPHA_KI
# (kinetic-inductance fraction) enters only Q_i,qp = pi Delta / (omega_0 alpha
# sigma_1); Q_EXT only the parallel-loss cap (Eq. 65). Every solve-affecting
# Table II/III constant (Delta_0, omega_0, tau_l, grid, P_read, T*/Delta, T_bath,
# and the solver knobs) lives in fig7_solve and is imported above.
ALPHA_KI = 0.13
Q_EXT_BY_DBM: dict[float, float] = {
    -100.0: 2.5e6,
    -90.0: 2.5e6,
    -80.0: 2.5e6,
    -72.0: 1.3e6,
    -68.0: 0.9e6,
    -64.0: 0.7e6,
}


@dataclass(frozen=True)
class Fig7PaperResult:
    T_bath: np.ndarray
    p_read_dbm: tuple[float, ...]
    n_bar_by_dbm: dict[float, float]
    Q_qp_by_dbm: dict[float, np.ndarray]
    Q_tot_by_dbm: dict[float, np.ndarray]
    sigma1_by_dbm: dict[float, np.ndarray]


def _parallel_quality_factor(Q_qp: float, Q_ext: float) -> float:
    if not np.isfinite(Q_qp):
        return float(Q_ext)
    return float(1.0 / (1.0 / Q_qp + 1.0 / Q_ext))


def observables(raw: Mapping[str, np.ndarray]) -> Fig7PaperResult:
    """Derive sigma_1 / Q_i,qp / Q_i,tot from a raw :func:`fig7_solve.solve` payload.

    The cheap downstream half of Fig. 7: rebuilds the (deterministic) spectral
    grid and evaluates the Mattis-Bardeen sigma_1 and the quality factors per
    (power, T_bath) point. A pure function of ``raw`` — this is what the cache
    leaves uncached, so editing it never triggers a re-solve.
    """
    T_values = np.asarray(raw["temperatures"], dtype=float)
    powers = tuple(float(p) for p in raw["powers_dbm"])
    n_bar = np.asarray(raw["n_bar"], dtype=float)
    num_bins = int(np.asarray(raw["num_bins"]).reshape(-1)[0])
    f_solved = np.asarray(raw["f_solved"], dtype=float)

    spectral, _omega = _build_grid(num_bins)

    n_bar_by_dbm = {p: float(n_bar[pi]) for pi, p in enumerate(powers)}
    Q_qp_by_dbm = {p: np.zeros_like(T_values) for p in powers}
    Q_tot_by_dbm = {p: np.zeros_like(T_values) for p in powers}
    sigma1_by_dbm = {p: np.zeros_like(T_values) for p in powers}

    for pi, p in enumerate(powers):
        for i in range(T_values.size):
            sigma1, _ = compute_ac_conductivity(f_solved[pi, i], spectral, OMEGA_0)
            Q_qp = (
                np.pi * DELTA_0 / (OMEGA_0 * ALPHA_KI * sigma1)
                if sigma1 > 0.0
                else float("inf")
            )
            Q_qp_by_dbm[p][i] = Q_qp
            Q_tot_by_dbm[p][i] = _parallel_quality_factor(Q_qp, Q_EXT_BY_DBM[p])
            sigma1_by_dbm[p][i] = sigma1

    return Fig7PaperResult(
        T_bath=T_values,
        p_read_dbm=powers,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
    )


def run(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> Fig7PaperResult:
    """Solve the sweep and derive observables — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this so it always truly recomputes against the pinned baseline;
    the cached dev / regen path is :func:`run_cached`.
    """
    return observables(
        solve(temperatures=temperatures, powers_dbm=powers_dbm, num_bins=num_bins)
    )


def run_cached(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> Fig7PaperResult:
    """Like :func:`run`, but the expensive solve is served from the disk cache
    when nothing solve-relevant has changed (see :mod:`validation.sweep_cache`).

    Used by the regen / ``__main__`` path. Editing the observables or plotting
    code in this module does not invalidate the cached solve; any edit to
    :mod:`fig7_solve` or to the ``qpsim`` solver subtree does. Disable entirely
    with ``QPSIM_SWEEP_CACHE=0``.
    """
    kwargs = {
        "temperatures": [float(x) for x in temperatures],
        "powers_dbm": [float(x) for x in powers_dbm],
        "num_bins": int(num_bins),
    }
    raw = sweep_cache.cached_solve(
        "fischer_2023/fig7",
        lambda: solve(
            temperatures=temperatures, powers_dbm=powers_dbm, num_bins=num_bins
        ),
        fingerprint=solver_fingerprint(num_bins=num_bins),
        kwargs=kwargs,
        extra_source=inspect.getsource(fig7_solve),
    )
    return observables(raw)


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_fig7_paper.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_c": re.compile(r"T_c=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "alpha": re.compile(r"alpha=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
    "tau_l": re.compile(r"tau_l=([\deE.+-]+)"),
    "tau_0_pb": re.compile(r"tau_0_pb=([\deE.+-]+)"),
}


def write_baseline(result: Fig7PaperResult, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["# Fischer 2023 Fig. 7 paper-facing Q_i(T_B); pinned by qpsim"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_0={OMEGA_0} "
            f"alpha={ALPHA_KI} c_phot={C_PHOT} tau_l={TAU_L} tau_0_pb={TAU_0_PB}"
        ])
        writer.writerow([f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta E_max={E_MAX_FACTOR}*Delta"])
        writer.writerow([f"# p_read_dbm={','.join(f'{p:g}' for p in result.p_read_dbm)}"])
        writer.writerow(["T_bath_K", "P_read_dbm", "n_bar", "Q_qp", "Q_tot", "sigma1"])
        for p in result.p_read_dbm:
            for i, T_bath in enumerate(result.T_bath):
                writer.writerow([
                    f"{T_bath:.17e}",
                    f"{p:.17e}",
                    f"{result.n_bar_by_dbm[p]:.17e}",
                    f"{result.Q_qp_by_dbm[p][i]:.17e}",
                    f"{result.Q_tot_by_dbm[p][i]:.17e}",
                    f"{result.sigma1_by_dbm[p][i]:.17e}",
                ])
    return path


def read_baseline(path: Path | None = None) -> Fig7PaperResult:
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    powers: list[float] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# p_read_dbm"):
                powers = [float(x) for x in first.split("=", 1)[1].split(",")]
                continue
            if first.startswith("#") or first == "T_bath_K":
                continue
            rows.append([float(x) for x in line])
    if not powers:
        raise RuntimeError(f"Baseline at {path} missing '# p_read_dbm=' metadata.")
    data = np.array(rows, dtype=float)
    temps = np.unique(data[:, 0])
    p_tuple = tuple(powers)
    n_bar_by_dbm: dict[float, float] = {}
    Q_qp_by_dbm: dict[float, np.ndarray] = {}
    Q_tot_by_dbm: dict[float, np.ndarray] = {}
    sigma1_by_dbm: dict[float, np.ndarray] = {}
    for p in p_tuple:
        subset = data[data[:, 1] == p]
        if subset.shape[0] != temps.size:
            raise RuntimeError(f"Baseline has {subset.shape[0]} rows for P={p}, expected {temps.size}.")
        order = np.argsort(subset[:, 0])
        subset = subset[order]
        n_bar_by_dbm[p] = float(subset[0, 2])
        Q_qp_by_dbm[p] = subset[:, 3]
        Q_tot_by_dbm[p] = subset[:, 4]
        sigma1_by_dbm[p] = subset[:, 5]
    return Fig7PaperResult(
        T_bath=temps,
        p_read_dbm=p_tuple,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
    )


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config) without
    touching the data rows or running the kinetic sweep.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds (see :mod:`fig6_paper` for the same
    pattern). Note Fig. 7's ``τ_l`` / ``τ_0^PB`` are fixed Table II/III scalars
    (``TAU_L`` / ``TAU_0_PB``), not phonon-side extractions. The ``p_read_dbm``
    powers and per-power ``n_bar`` are compared separately against the baseline
    rows.
    """

    delta_0: float
    tau_0: float
    t_c: float
    omega_0: float
    alpha: float
    c_phot: float
    tau_l: float
    tau_0_pb: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float


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
        alpha=_num(_HEADER_PARAM_RE["alpha"], "alpha"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        tau_l=_num(_HEADER_PARAM_RE["tau_l"], "tau_l"),
        tau_0_pb=_num(_HEADER_PARAM_RE["tau_0_pb"], "tau_0_pb"),
        num_bins=int(ne_m.group(1)),
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — pure constants, so effectively instant (Fig. 7's
    ``τ_l`` / ``τ_0^PB`` are fixed Table II/III scalars, not extracted)."""
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_c=T_C,
        omega_0=OMEGA_0,
        alpha=ALPHA_KI,
        c_phot=C_PHOT,
        tau_l=TAU_L,
        tau_0_pb=TAU_0_PB,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
    )


def write_plot(result: Fig7PaperResult, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paper Fig. 7 palette per readout power, matching standalone reproduction
    # figures/fig7.py PAPER_COLORS.
    PAPER_COLORS = {
        -100.0: "firebrick", -90.0: "dimgray", -80.0: "black",
        -72.0: "red", -68.0: "green", -64.0: "blue",
    }
    fallback_cmap = matplotlib.colormaps["viridis_r"]

    # Analytical Q_i overlay (paper-repro figures/fig7.py): combine Eq. F1
    # thermal-equilibrium Q with Eq. 62 low-T non-equilibrium plateau in
    # parallel with Q_ext (Eq. 65). Closed-form, no kinetic solve needed.
    from scipy.special import k0 as _k0

    GAMMA0 = 19.3

    def _Qth(T_K: float) -> float:
        T_uev = T_K * KB_UEV_PER_K
        if T_uev <= 0:
            return np.inf
        x = OMEGA_0 / (2.0 * T_uev)
        return np.pi / (4.0 * ALPHA_KI * np.sinh(x) * _k0(x)) * np.exp(DELTA_0 / T_uev)

    def _Q_neq_lowT(p_dbm: float) -> float:
        # Eq. 62: depends only on (Δ, ω0, α, τ_l, τ_0^PB, T*).
        kBTs = (TSTAR_OVER_DELTA[float(p_dbm)] * DELTA_0)
        if TAU_L <= 0:
            return np.inf
        arg = float(np.sqrt(14.0 / 5.0)) * (DELTA_0 / kBTs) ** 3
        if arg > 700.0:
            return np.inf
        return (GAMMA0 * DELTA_0 / (ALPHA_KI * OMEGA_0)) * (TAU_0_PB / TAU_L) \
            * (DELTA_0 / kBTs) ** 1.5 * np.exp(arg)

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    T_dense = np.linspace(float(np.min(result.T_bath)),
                          float(np.max(result.T_bath)), 200)
    for i, p in enumerate(result.p_read_dbm):
        color = PAPER_COLORS.get(
            float(p), fallback_cmap(i / max(1, len(result.p_read_dbm) - 1))
        )
        ax.semilogy(
            result.T_bath, result.Q_tot_by_dbm[p],
            "o-", lw=1.4, ms=3.0, color=color, label=f"{p:g} dBm",
        )
        if float(p) in TSTAR_OVER_DELTA:
            Q_neq = _Q_neq_lowT(float(p))
            Q_ext = Q_EXT_BY_DBM[float(p)]
            Qa = np.array([
                1.0 / (1.0 / _Qth(T) + 1.0 / max(Q_neq, 1.0)) for T in T_dense
            ])
            Qa = 1.0 / (1.0 / np.maximum(Qa, 1.0) + 1.0 / Q_ext)
            ax.semilogy(T_dense, Qa, color=color, ls="--", lw=1.0)
    ax.set_xlabel("T (K)")
    ax.set_ylabel(r"$Q_{i,\mathrm{tot}}$")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Fig. 7 paper-facing Q_i(T_B) ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, omega_0={OMEGA_0} micro-eV, "
        f"alpha={ALPHA_KI}, tau_l={TAU_L} ns"
    )
    print(f"  P_read (dBm): {list(P_READ_DBM)}")
    print(f"  T_B: {list(T_BATH_VALUES)}")
    result = run_cached()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  wrote {csv_path}")
    print(f"  wrote {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
