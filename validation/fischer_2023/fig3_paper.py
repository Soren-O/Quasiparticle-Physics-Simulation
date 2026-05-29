"""Fischer 2023 Fig. 3 — paper-target legend-ratio reproduction.

Four curves at $\\tau_\\ell / \\tau_0^{PB} \\in \\{0, 0.1, 1, 10\\}$ on the
paper's energy grid, with $\\tau_0^{PB}$ extracted through the phonon-side
Kaplan pair-breaking rate.

* Paper grid: ``NE = 1620``, ``dE = 1 μeV``, integer-commensurate with
  $\\omega_0 = \\Delta_0/9 = 20\\,\\mu$eV.
* Paper legend ratios: $\\{0, 0.1, 1, 10\\}$.
* Continuation through intermediate ratios for stability of the strong-
  bottleneck branch.
* Paper-style axis: $f(E)$ vs $E/\\Delta_0 - 1$ on $[0, 4]$, with photon-
  step markers at $n\\,\\omega_0/\\Delta_0$.

The module uses the F&C/Kaplan phonon-side pair-breaking kernel and the
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
import inspect
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.constants import KB_UEV_PER_K

from validation import sweep_cache
from validation.fischer_2023 import fig3_solve
from validation.fischer_2023.fig3_solve import (
    C_PHOT,
    CONTINUATION_RATIOS,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    N_BAR,
    NUM_BINS,
    OMEGA_0,
    PAPER_RATIOS,
    T_BATH,
    T_C,
    TAU_0,
    _build_grid_and_spectral,
    _compute_tau_0_pb,
    solve,
    solver_fingerprint,
)


@dataclass(frozen=True)
class Fig3PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    tau_0_pb_ns: float
    ratios: tuple[float, ...]   # paper ratios {0, 0.1, 1, 10}
    f_by_ratio: dict[float, np.ndarray]
    f_FD: np.ndarray            # thermal reference at T_bath


def observables(raw: Mapping[str, np.ndarray]) -> Fig3PaperResult:
    """Repackage a raw :func:`fig3_solve.solve` payload into a Fig3PaperResult.

    Fig. 3's plotted quantity is the converged f(E) per ratio, so there is no
    expensive downstream derivation here — this is a pure unpacking of the cached
    arrays. It exists to keep the ``run() = observables(solve())`` split uniform
    with the other figures and to be the boundary the cache leaves uncached.
    """
    E = np.asarray(raw["E"], dtype=float)
    f_FD = np.asarray(raw["f_FD"], dtype=float)
    ratios = tuple(float(r) for r in raw["ratios"])
    tau_0_pb_ns = float(np.asarray(raw["tau_0_pb_ns"]).reshape(-1)[0])
    f_ratios = np.asarray(raw["f_ratios"], dtype=float)
    f_by_ratio = {r: f_ratios[i] for i, r in enumerate(ratios)}
    return Fig3PaperResult(
        E=E,
        tau_0_pb_ns=tau_0_pb_ns,
        ratios=ratios,
        f_by_ratio=f_by_ratio,
        f_FD=f_FD,
    )


def run(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
) -> Fig3PaperResult:
    """Solve Fischer Fig. 3 and repackage — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this (no args) so it always truly recomputes against the pinned
    baseline; the cached dev / regen path is :func:`run_cached`.
    """
    return observables(
        solve(
            num_bins=num_bins,
            paper_ratios=paper_ratios,
            continuation_ratios=continuation_ratios,
        )
    )


def run_cached(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
) -> Fig3PaperResult:
    """Like :func:`run`, but the expensive continuation solve is served from the
    disk cache when nothing solve-relevant has changed (see
    :mod:`validation.sweep_cache`). Used by the regen / ``__main__`` path; editing
    the plotting / observable code here does not invalidate the cached solve.
    Disable with ``QPSIM_SWEEP_CACHE=0``.
    """
    kwargs = {
        "num_bins": int(num_bins),
        "paper_ratios": [float(r) for r in paper_ratios],
        "continuation_ratios": [float(r) for r in continuation_ratios],
    }
    raw = sweep_cache.cached_solve(
        "fischer_2023/fig3",
        lambda: solve(
            num_bins=num_bins,
            paper_ratios=paper_ratios,
            continuation_ratios=continuation_ratios,
        ),
        fingerprint=solver_fingerprint(num_bins=num_bins),
        kwargs=kwargs,
        extra_source=inspect.getsource(fig3_solve),
    )
    return observables(raw)


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
_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_bath": re.compile(r"T_bath=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "n_bar": re.compile(r"n_bar=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
}


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


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config)
    without touching the data rows or running the continuation ladder.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds instead of after the several-minute
    run (see :mod:`fig6_paper` for the same pattern, where the payoff is ~14 h).
    """

    delta_0: float
    tau_0: float
    t_bath: float
    omega_0: float
    n_bar: float
    c_phot: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    tau_0_pb_ns: float
    ratios: tuple[float, ...]


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
    ratios_m = _RATIOS_RE.search(text)
    if ne_m is None or ratios_m is None:
        raise RuntimeError(
            f"Baseline header at {path} missing NE / ratios metadata."
        )
    ratios = tuple(
        float(x.strip()) for x in ratios_m.group(1).split(",") if x.strip()
    )
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_bath=_num(_HEADER_PARAM_RE["t_bath"], "T_bath"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        n_bar=_num(_HEADER_PARAM_RE["n_bar"], "n_bar"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        num_bins=int(ne_m.group(1)),
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        tau_0_pb_ns=_num(_TAU_0_PB_RE, "tau_0_pb_ns"),
        ratios=ratios,
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — computed without the (several-minute) continuation run.

    ``tau_0_pb_ns`` is produced by the exact :func:`_compute_tau_0_pb` call
    :func:`run` makes, so it can never drift from a real run; everything else
    is read straight off the module constants.
    """
    _, _, spectral = _build_grid_and_spectral()
    tau_0_pb = _compute_tau_0_pb(spectral)
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_bath=T_BATH,
        omega_0=OMEGA_0,
        n_bar=N_BAR,
        c_phot=C_PHOT,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        tau_0_pb_ns=tau_0_pb,
        ratios=PAPER_RATIOS,
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
    result = run_cached()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
