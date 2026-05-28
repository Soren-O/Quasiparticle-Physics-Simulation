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
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

_CACHE_ROOT = Path("/private/tmp/qpsim-cache")
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

# Fischer 2023 Tables II/III.
DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = 22.0
C_PHOT = 0.06e-9
TAU_0_PB = 0.040
TAU_L = 0.170
ALPHA_KI = 0.13

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1701  # h = Delta / 189, E in [Delta, 10 Delta].

P_READ_DBM: tuple[float, ...] = (-100.0, -90.0, -80.0, -72.0, -68.0, -64.0)
TSTAR_OVER_DELTA: dict[float, float] = {
    -100.0: 0.12,
    -90.0: 0.18,
    -80.0: 0.26,
    -72.0: 0.36,
    -68.0: 0.42,
    -64.0: 0.49,
}
Q_EXT_BY_DBM: dict[float, float] = {
    -100.0: 2.5e6,
    -90.0: 2.5e6,
    -80.0: 2.5e6,
    -72.0: 1.3e6,
    -68.0: 0.9e6,
    -64.0: 0.7e6,
}

# A compact curve grid.  The standalone reproduction uses a denser visual grid;
# this is enough to pin the plateau and high-temperature rolloff without making
# the slow validation suite painfully large.
T_BATH_VALUES: tuple[float, ...] = (0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34)


@dataclass(frozen=True)
class Fig7PaperResult:
    T_bath: np.ndarray
    p_read_dbm: tuple[float, ...]
    n_bar_by_dbm: dict[float, float]
    Q_qp_by_dbm: dict[float, np.ndarray]
    Q_tot_by_dbm: dict[float, np.ndarray]
    sigma1_by_dbm: dict[float, np.ndarray]


def _paper_material() -> Material:
    return Material(
        name="Al_Fischer2023_TableII",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=TAU_0_PB,
    )


def _nbar_from_table_iii(power_dbm: float) -> float:
    """Return nbar whose Tstar,0/Delta matches Fischer Table III."""
    target = TSTAR_OVER_DELTA[float(power_dbm)] * DELTA_0
    T_c_uev = T_C * KB_UEV_PER_K
    denom = (
        (105.0 / 64.0)
        * T_c_uev**3
        * DELTA_0
        * (C_PHOT * TAU_0)
        * OMEGA_0**2
    )
    return float(target**6 / denom)


def _parallel_quality_factor(Q_qp: float, Q_ext: float) -> float:
    if not np.isfinite(Q_qp):
        return float(Q_ext)
    return float(1.0 / (1.0 / Q_qp + 1.0 / Q_ext))


def _build_grid(num_bins: int = NUM_BINS) -> tuple[SpectralContext, np.ndarray]:
    E, dE_scalar = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=num_bins,
    )
    shift = OMEGA_0 / dE_scalar
    if abs(shift - round(shift)) > 1e-10:
        raise ValueError(
            f"Fig. 7 paper grid must be commensurate with omega_0. "
            f"Got dE={dE_scalar:.6g}, omega_0/dE={shift:.6g}."
        )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
    return spectral, omega


def _fermi_dirac(E: np.ndarray, T_bath: float) -> np.ndarray:
    kT = KB_UEV_PER_K * T_bath
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(
    material: Material,
    spectral: SpectralContext,
    omega: np.ndarray,
    T_bath: float,
    *,
    f_seed: np.ndarray | None = None,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    f0 = _fermi_dirac(spectral.E, T_bath) if f_seed is None else f_seed.copy()
    n_ph = (
        thermal_phonon_occupation(omega, T_bath)
        if n_ph_seed is None
        else n_ph_seed.copy()
    )
    phonon = PhononState(
        n_ph=n_ph.reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), TAU_L),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=f0,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def run(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> Fig7PaperResult:
    material = _paper_material()
    spectral, omega = _build_grid(num_bins)
    backend = T3DiffusionBackend()

    T_values = np.array(temperatures, dtype=float)
    n_bar_by_dbm = {p: _nbar_from_table_iii(p) for p in powers_dbm}
    Q_qp_by_dbm = {p: np.zeros_like(T_values) for p in powers_dbm}
    Q_tot_by_dbm = {p: np.zeros_like(T_values) for p in powers_dbm}
    sigma1_by_dbm = {p: np.zeros_like(T_values) for p in powers_dbm}

    for p_dbm in powers_dbm:
        nbar = n_bar_by_dbm[p_dbm]
        photon_params = {"omega_0": OMEGA_0, "n_bar": nbar, "c_phot": C_PHOT}

        for i, T_bath in enumerate(T_values):
            state = _build_state(
                material,
                spectral,
                omega,
                float(T_bath),
            )
            solved = backend.steady_state(
                state,
                method="picard",
                use_phonon_side_kernel=True,
                photon_params=photon_params,
                picard_tol=1e-7,
                picard_max_iter=500,
                picard_mixing=0.2,
                anderson_depth=3,
                newton_tol=1e-11,
                newton_max_iter=300,
            )
            sigma1, _ = compute_ac_conductivity(solved.f, solved.spectral, OMEGA_0)
            Q_qp = (
                np.pi * DELTA_0 / (OMEGA_0 * ALPHA_KI * sigma1)
                if sigma1 > 0.0
                else float("inf")
            )
            Q_qp_by_dbm[p_dbm][i] = Q_qp
            Q_tot_by_dbm[p_dbm][i] = _parallel_quality_factor(
                Q_qp, Q_EXT_BY_DBM[p_dbm],
            )
            sigma1_by_dbm[p_dbm][i] = sigma1

    return Fig7PaperResult(
        T_bath=T_values,
        p_read_dbm=powers_dbm,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
    )


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

    Tc_uev = T_C * KB_UEV_PER_K
    A35 = (105.0 / 64.0) * Tc_uev ** 3 * DELTA_0 * (C_PHOT * TAU_0) * OMEGA_0 ** 2
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
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  wrote {csv_path}")
    print(f"  wrote {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
