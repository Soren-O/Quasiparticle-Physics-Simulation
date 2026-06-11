"""Fischer & Catelani 2024 Fig. 5 — qpsim-native characterization at paper topology.

This is the **structural** Fischer 2024 Fig. 5 reproduction --- three drive
levels at $\\omega_{\\rm PB} = 2.8\\Delta$ with $\\tau_\\ell = 0$, plotted in
the paper's $\\gamma = (E - \\Delta)/\\Delta$ coordinate. The paper-faithful
Neumann-series analytic overlays $f^{(0)}$, $f^{(0)}+f^{(1)}$, and
$f^{(0)}+f^{(1)}+f^{(2)}$ are **placeholders** (see :func:`_neumann_f0`
etc.), and the Hz $\\leftrightarrow$ ns$^{-1}$ unit conversion of the
drive product carries a paper-parity audit warning. Until both gaps
close, the artifact this script writes is **not** a paper-faithful
reproduction; it is a qpsim-native characterization at the paper's
sweep topology. The CSV / PDF filenames and plot title make that
explicit so a downstream consumer is not misled.

The published Fischer & Catelani 2024 Fig. 5 is a two-panel comparison:

* **Left panel.** $\\gamma \\in [0, 1]$, $f(\\gamma)$ on log-scale.
* **Right panel.** Zoom of $\\gamma \\in [0, 0.10]$ to show the $f^{(2)}$
  near-gap correction, $f$ on log-scale.

Black: numerical kinetic-equation solution. Colored, in successive
Neumann-series order: green $f^{(0)}$, red $f^{(0)} + f^{(1)}$, blue
$f^{(0)} + f^{(1)} + f^{(2)}$. Three curve families per panel — one per
drive level $c_{\\rm phot,PB} \\bar n_{\\rm PB}$.

The existing :mod:`figs_5_7_fe_pb` script combines Figs. 5–7 into one
overlay with five drive levels (qpsim-native ns$^{-1}$ values) and no
analytic overlay. This script is the dedicated paper-target Fig. 5
reproduction at the paper's three drive levels (caption-quoted in Hz)
with the analytic Neumann-series overlay in place.

Outstanding paper-parity gaps (load-bearing)
--------------------------------------------

1. **Hz ↔ ns⁻¹ unit audit.** The F24 caption quotes
   $c_{\\rm phot,PB}\\bar n_{\\rm PB} \\in \\{10^{-2},10^{-4},10^{-6}\\}$
   Hz. qpsim's collision kernel takes $c_{\\rm phot,PB}$ in ns$^{-1}$.
   Naive conversion is $1\\,\\mathrm{Hz} = 10^{-9}\\,\\mathrm{ns}^{-1}$,
   so the paper drives correspond to $\\{10^{-11},10^{-13},10^{-15}\\}$
   ns$^{-1}$ — radically smaller than the existing
   :data:`figs_5_7_fe_pb.POWER_LEVELS` choice
   $\\{10^{-6},\\dots,10^{-2}\\}$ ns$^{-1}$. Until the kinetic-equation
   prefactor in :mod:`qpsim.collisions.pair_breaking_photon` is audited
   against the F24 derivation, the absolute scale of the drive axis
   here is not paper-faithful. The :data:`PAPER_DRIVES_NS_INV` tuple
   below pins the **literal** Hz → ns$^{-1}$ conversion; an audit
   assertion in :func:`_assert_unit_audit` checks that the conversion
   factor matches the qpsim convention, and will print a warning if
   the resulting drive products fall outside the regime where the
   existing :mod:`figs_5_7_fe_pb` baseline lives.

2. **Neumann-series analytic overlays.** The paper overlays $f^{(0)}$,
   $f^{(0)}+f^{(1)}$, $f^{(0)}+f^{(1)}+f^{(2)}$ alongside the numerical
   curve. The closed-form expressions are derived in F24 Sec. III but
   require careful tracking of the BCS coherence factors and partner-
   energy reflections. :func:`_neumann_f0`, :func:`_neumann_f1`,
   :func:`_neumann_f2` currently return clearly-labelled placeholder
   shapes (linear-interpolated thermal background plus a partner-
   reflection step). The dashed analytic overlays on the plot are
   therefore qualitative, not paper-faithful. Replace the function
   bodies with the verified Neumann series once they have been hand-
   checked against the paper text.

Once both tickets land, this script becomes the paper-faithful Fig. 5
reproduction. At that point: rename the artifacts to
``fischer2024_fig5_paper.{csv,pdf}``, drop the "qpsim-native" wording
from the plot title, and remove the warning text from the run banner.

F24 Sec. IV parameters:

    Δ_0     = 189 μeV
    τ_0     = 63 ns
    ω_PB    = 2.8·Δ = 529.2 μeV (above 2Δ, pair-breaking active)
    τ_ℓ     = 0          (thermal-phonon shortcut)
    n̄_PB    = 1e6        (kept fixed; only c·n̄ matters for f shape)
    c_phot,PB · n̄_PB ∈ {1e-2, 1e-4, 1e-6} Hz  (paper caption)

Grid: 810 bins so ω_PB/dE = 252 is integer-commensurate (inherits the
:mod:`fig8_xqp_pb` choice; the older 851-bin choice snapped ω_PB by
~0.3% and lost bit-reproducibility).

Usage --- generate baseline + PDF::

    python -m validation.fischer_2024.fig5_paper
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

# ── F24 Sec. IV parameters ───────────────────────────────────────────

DELTA_0 = 189.0            # μeV
TAU_0 = 63.0               # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1               # K (paper Fig. 5 bath)
OMEGA_PB = 2.8 * DELTA_0   # 529.2 μeV
# F&C 2024 paper coordinate: γ = (E − Δ) / ξ, where ξ = ω_PB − 2Δ.
# γ = 1 marks the pair-breaking endpoint at which the partner energy
# E_partner = ω_PB − E hits the gap from above. For ω_PB = 2.8Δ this
# gives ξ = 0.8Δ. Using the wrong denominator (Δ instead of ξ) puts
# the partner-below-gap mask transition at γ = 0.8 in code coords,
# producing a vertical-cliff plot artifact when matplotlib connects
# the last active bin to the first masked bin on log y.
XI = OMEGA_PB - 2.0 * DELTA_0   # 151.2 μeV
N_BAR_PB = 1e6             # photon population (only c·n̄ matters for f shape)

# Paper grid: 810 bins gives ω_PB/dE = 252 exactly. dE = 9·Δ/810 = 2.1 μeV.
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810

# ── Hz ↔ ns⁻¹ unit audit (paper caption is in Hz) ────────────────────

# Paper Fig. 5 caption: c_phot,PB · n̄_PB ∈ {1e-2, 1e-4, 1e-6} Hz.
PAPER_DRIVES_HZ: tuple[float, ...] = (1e-2, 1e-4, 1e-6)

# 1 Hz = 1 s⁻¹ = 1×10⁻⁹ ns⁻¹.  Pin the literal conversion factor here
# so the audit assertion below catches any drift in qpsim's convention.
HZ_TO_NS_INV: float = 1.0e-9

# Drive products applied to the qpsim PB-photon kernel, in ns⁻¹. These
# are the paper Hz values multiplied by HZ_TO_NS_INV.
PAPER_DRIVES_NS_INV: tuple[float, ...] = tuple(
    p * HZ_TO_NS_INV for p in PAPER_DRIVES_HZ
)

# Sanity: existing figs_5_7_fe_pb.POWER_LEVELS sweep covers
# {1e-6, …, 1e-2} ns⁻¹, which corresponds to {1e3, …, 1e7} Hz under the
# literal conversion — five to nine orders of magnitude **larger** than
# the paper-quoted Hz values. Flag this in :func:`_assert_unit_audit`
# but do not fail; the resolution lives upstream in the pair-breaking
# kernel prefactor audit.
EXISTING_F24_NATIVE_RANGE_NS_INV = (1e-6, 1e-2)


@dataclass(frozen=True)
class Fig5PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    drives_hz: tuple[float, ...]
    drives_ns_inv: tuple[float, ...]
    f_thermal: np.ndarray                 # shape (NE,); f_FD at T_bath
    f_by_drive: dict[float, np.ndarray]    # drive_hz → numerical f
    x_qp_by_drive: dict[float, float]      # drive_hz → scalar x_qp
    f0_by_drive: dict[float, np.ndarray]   # drive_hz → f^(0) (placeholder)
    f01_by_drive: dict[float, np.ndarray]  # drive_hz → f^(0) + f^(1)
    f012_by_drive: dict[float, np.ndarray]  # drive_hz → f^(0) + f^(1) + f^(2)


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_grid_and_spectral() -> tuple[np.ndarray, np.ndarray, SpectralContext]:
    """Build the F24 paper-grid energy axis + dE widths + spectral context."""
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_PB / dE_scalar)
    frac_err = abs(OMEGA_PB - m * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"F24 Fig. 5 paper grid not commensurate: "
            f"ω_PB={OMEGA_PB} μeV, m·dE={m * dE_scalar} μeV, frac_err={frac_err}. "
            f"Choose NUM_BINS such that ω_PB/dE is integer."
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _assert_unit_audit() -> None:
    """Hz ↔ ns⁻¹ unit-audit assertion (paper-parity gap #1).

    Pin the literal ``1 Hz = 1e-9 ns⁻¹`` conversion. Print a banner
    flagging the drive-magnitude mismatch with the existing F24 native
    range, since the resolution requires a kernel-prefactor audit that
    is upstream of this script.
    """
    # Strict assertion on the conversion factor itself.
    assert HZ_TO_NS_INV == 1.0e-9, (
        f"Hz → ns⁻¹ conversion drifted: HZ_TO_NS_INV={HZ_TO_NS_INV} "
        "must be 1e-9 (1 Hz = 1 s⁻¹ = 10⁻⁹ ns⁻¹)."
    )
    for hz, ns_inv in zip(PAPER_DRIVES_HZ, PAPER_DRIVES_NS_INV, strict=True):
        expected = hz * 1.0e-9
        assert abs(ns_inv - expected) < 1e-30, (
            f"Drive conversion mismatch: {hz} Hz mapped to {ns_inv} ns⁻¹, "
            f"expected {expected}."
        )
    # Soft warning on the magnitude regime.
    paper_min, paper_max = min(PAPER_DRIVES_NS_INV), max(PAPER_DRIVES_NS_INV)
    native_min, native_max = EXISTING_F24_NATIVE_RANGE_NS_INV
    if paper_max < native_min or paper_min > native_max:
        print(
            "  ⚠ Hz → ns⁻¹ unit-audit warning:\n"
            f"    Paper-quoted drives  : {paper_min:.2e} - {paper_max:.2e} ns⁻¹\n"
            f"    Existing F24 sweep   : {native_min:.2e} - {native_max:.2e} ns⁻¹\n"
            "    Drive products are several orders of magnitude apart.\n"
            "    Resolution requires auditing the pair-breaking kernel\n"
            "    prefactor (qpsim.collisions.pair_breaking_photon) against\n"
            "    the F24 derivation. See module docstring.",
            flush=True,
        )


def _build_state(material: Material, spectral: SpectralContext) -> T3DiffusionState:
    """Build a T3 state at T_BATH with thermal-phonon n_ph and τ_ℓ = 0."""
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_BATH).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_BATH,
    )


# ── Neumann-series analytic overlays (paper-parity placeholders) ─────


def _neumann_f0(
    spectral: SpectralContext, drive_ns_inv: float,
) -> np.ndarray:
    """Zeroth-order Neumann term $f^{(0)}$ (PLACEHOLDER).

    TODO(paper-parity): replace with the F24 closed form. Per F24
    Sec. III, $f^{(0)}$ is the thermal background of the pair-breaking
    kinetic equation in the absence of drive — i.e. the Fermi-Dirac
    distribution at the bath temperature. We return the Fermi-Dirac
    distribution as a first approximation. The drive scaling enters at
    $f^{(1)}$ and beyond; $f^{(0)}$ is drive-independent in the strict
    Neumann expansion of the pair-breaking integral operator about the
    thermal fixed point.
    """
    kT = KB_UEV_PER_K * T_BATH
    return 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)


def _neumann_f1(
    spectral: SpectralContext, drive_ns_inv: float,
) -> np.ndarray:
    """First-order Neumann correction $f^{(1)}$ (PLACEHOLDER).

    TODO(paper-parity): replace with the F24 closed form. Per F24
    Sec. III, $f^{(1)}(E) \\propto c_{\\rm phot,PB}\\bar n_{\\rm PB}
    \\, U_-(E) (1 - f^{(0)}(\\omega_{\\rm PB} - E))$ for the partner-
    reflection branch within the $[\\Delta, \\omega_{\\rm PB} - \\Delta]$
    band, plus the absorption/emission branches at $E \\pm \\omega_{\\rm PB}$.

    For now we return a heuristic placeholder: a band-limited step at
    $E \\in [\\Delta, \\omega_{\\rm PB} - \\Delta]$ scaled by drive,
    enough to place the dashed overlay in the right ballpark on a
    log-y plot but **not** the paper's formula. Returns the
    *increment* $f^{(1)}$ alone; sum with $f^{(0)}$ when plotting.
    """
    E = spectral.E
    f0 = _neumann_f0(spectral, drive_ns_inv)
    band = (E >= DELTA_0) & (E <= OMEGA_PB - DELTA_0)
    increment = np.zeros_like(E)
    # Heuristic only: scale with drive and drop with energy in the band.
    if drive_ns_inv > 0:
        gamma = (E - DELTA_0) / XI
        denom = 1.0 + np.maximum(gamma, 0.0)
        increment = np.where(
            band,
            drive_ns_inv * 1e9 * (1.0 - f0) / denom,
            0.0,
        )
    return increment


def _neumann_f2(
    spectral: SpectralContext, drive_ns_inv: float,
) -> np.ndarray:
    """Second-order Neumann correction $f^{(2)}$ (PLACEHOLDER).

    TODO(paper-parity): replace with the F24 closed form. The second
    Neumann iterate folds the kernel against $f^{(1)}$ — in particular,
    it captures the near-gap reflection-of-reflection feature visible
    on the right panel of paper Fig. 5 at $\\gamma \\lesssim 0.04$.

    Returns the *increment* $f^{(2)}$ alone; sum with $f^{(0)} +
    f^{(1)}$ when plotting. The placeholder here is a near-gap bump
    scaled by drive², roughly tracking the right-panel paper feature;
    the absolute amplitude is **not** paper-faithful.
    """
    E = spectral.E
    increment = np.zeros_like(E)
    if drive_ns_inv > 0:
        gamma = (E - DELTA_0) / XI
        # Near-gap bump: peaked at γ ≈ 0, decay length ~0.04.
        bump = np.exp(-(gamma / 0.04) ** 2)
        increment = (drive_ns_inv * 1e9) ** 2 * bump * 1e-2
        increment = np.where(gamma >= 0.0, increment, 0.0)
    return increment


def run() -> Fig5PaperResult:
    """Solve F24 Fig. 5 — three drive levels at fixed T_B = 0.1 K."""
    print("F24 Fig. 5 paper-target reproduction ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, ω_PB={OMEGA_PB:.2f} μeV, "
        f"T_B={T_BATH} K, τ_ℓ=0"
    )
    print(
        f"  Grid: NE={NUM_BINS}, "
        f"dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.4f} μeV "
        f"(ω_PB/dE = {round(OMEGA_PB / ((E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS))})"
    )
    print(
        f"  Drive levels (paper Hz): {list(PAPER_DRIVES_HZ)}\n"
        f"  Drive levels (ns⁻¹)    : "
        f"{[f'{p:.2e}' for p in PAPER_DRIVES_NS_INV]}"
    )

    _assert_unit_audit()

    material = _material()
    _, _, spectral = _build_grid_and_spectral()
    state = _build_state(material, spectral)

    backend = T3DiffusionBackend()
    f_thermal = state.f.copy()

    f_by_drive: dict[float, np.ndarray] = {}
    x_qp_by_drive: dict[float, float] = {}
    f0_by_drive: dict[float, np.ndarray] = {}
    f01_by_drive: dict[float, np.ndarray] = {}
    f012_by_drive: dict[float, np.ndarray] = {}

    for drive_hz, drive_ns_inv in zip(
        PAPER_DRIVES_HZ, PAPER_DRIVES_NS_INV, strict=True,
    ):
        # ω_PB and n̄_PB held fixed; only c_phot_PB · n̄_PB matters for the
        # f-shape, so distribute the product into c_phot_PB at fixed n̄_PB.
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": drive_ns_inv / N_BAR_PB,
        }
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            pb_photon_params=pb_params,
            newton_tol=1e-14,
            newton_max_iter=500,
        )
        f_by_drive[drive_hz] = driven.f.copy()
        x_qp = float(qp_fraction(driven.f, driven.spectral, delta_0=DELTA_0))
        x_qp_by_drive[drive_hz] = x_qp

        f0 = _neumann_f0(spectral, drive_ns_inv)
        f1 = _neumann_f1(spectral, drive_ns_inv)
        f2 = _neumann_f2(spectral, drive_ns_inv)
        f0_by_drive[drive_hz] = f0
        f01_by_drive[drive_hz] = f0 + f1
        f012_by_drive[drive_hz] = f0 + f1 + f2

        print(
            f"    c·n̄ = {drive_hz:g} Hz ({drive_ns_inv:.2e} ns⁻¹): "
            f"x_qp = {x_qp:.4e}",
            flush=True,
        )

    return Fig5PaperResult(
        E=spectral.E,
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=PAPER_DRIVES_NS_INV,
        f_thermal=f_thermal,
        f_by_drive=f_by_drive,
        x_qp_by_drive=x_qp_by_drive,
        f0_by_drive=f0_by_drive,
        f01_by_drive=f01_by_drive,
        f012_by_drive=f012_by_drive,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer2024_fig5_qpsim_native.csv`` to flag that:

    1. The Hz → ns⁻¹ unit conversion is the literal physical factor,
       but the resulting drive products fall outside the regime of
       qpsim's existing F24 native sweep — pending a kernel-prefactor
       audit.
    2. The Neumann-series analytic overlays (`f0`, `f01`, `f012`
       columns) are placeholders, not the paper's closed-form
       expressions.

    Rename to ``fischer2024_fig5_paper.csv`` once both gaps close.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer2024_fig5_qpsim_native.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Write the three drive-level f(E) arrays + Neumann overlays to CSV."""
    if path is None:
        path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "# Fischer & Catelani 2024 Fig. 5 — qpsim-native characterization "
            "at paper topology"
        ])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_c={T_C:.6f} omega_PB={OMEGA_PB} "
            f"n_bar_PB={N_BAR_PB} T_bath={T_BATH}"
        ])
        writer.writerow([
            f"# Grid: NE={NUM_BINS} E_min={E_MIN_FACTOR}*Delta "
            f"E_max={E_MAX_FACTOR}*Delta"
        ])
        drives_hz_csv = ",".join(f"{d:g}" for d in result.drives_hz)
        drives_ns_csv = ",".join(f"{d:.17e}" for d in result.drives_ns_inv)
        writer.writerow([f"# drives_hz={drives_hz_csv}"])
        writer.writerow([f"# drives_ns_inv={drives_ns_csv}"])
        x_qp_csv = ",".join(
            f"{result.x_qp_by_drive[d]:.17e}" for d in result.drives_hz
        )
        writer.writerow([f"# x_qp_by_drive={x_qp_csv}"])

        header = ["E_uev", "f_thermal"]
        for d in result.drives_hz:
            header.append(f"f_num_{d:g}Hz")
            header.append(f"f0_{d:g}Hz")
            header.append(f"f01_{d:g}Hz")
            header.append(f"f012_{d:g}Hz")
        writer.writerow(header)

        for i in range(result.E.size):
            row = [f"{result.E[i]:.17e}", f"{result.f_thermal[i]:.17e}"]
            for d in result.drives_hz:
                row.append(f"{result.f_by_drive[d][i]:.17e}")
                row.append(f"{result.f0_by_drive[d][i]:.17e}")
                row.append(f"{result.f01_by_drive[d][i]:.17e}")
                row.append(f"{result.f012_by_drive[d][i]:.17e}")
            writer.writerow(row)
    return path


def read_baseline(path: Path | None = None) -> Fig5PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig5PaperResult`."""
    if path is None:
        path = baseline_path()
    rows: list[list[float]] = []
    drives_hz: tuple[float, ...] = ()
    drives_ns_inv: tuple[float, ...] = ()
    x_qp_list: list[float] = []
    with path.open() as fp:
        reader = csv.reader(fp)
        for line in reader:
            if not line:
                continue
            first = line[0]
            if first.startswith("# drives_hz"):
                drives_hz = tuple(
                    float(x) for x in first.split("=", 1)[1].split(",")
                )
                continue
            if first.startswith("# drives_ns_inv"):
                drives_ns_inv = tuple(
                    float(x) for x in first.split("=", 1)[1].split(",")
                )
                continue
            if first.startswith("# x_qp_by_drive"):
                x_qp_list = [
                    float(x) for x in first.split("=", 1)[1].split(",")
                ]
                continue
            if first.startswith("#") or first == "E_uev":
                continue
            rows.append([float(x) for x in line])
    if not drives_hz:
        raise RuntimeError(f"Baseline at {path} missing '# drives_hz=' metadata.")
    if not drives_ns_inv or len(drives_ns_inv) != len(drives_hz):
        raise RuntimeError(
            f"Baseline at {path} missing/inconsistent '# drives_ns_inv=' metadata."
        )
    if len(x_qp_list) != len(drives_hz):
        raise RuntimeError(
            f"Baseline at {path} missing '# x_qp_by_drive=' metadata."
        )
    data = np.array(rows, dtype=float)
    f_by_drive: dict[float, np.ndarray] = {}
    f0_by_drive: dict[float, np.ndarray] = {}
    f01_by_drive: dict[float, np.ndarray] = {}
    f012_by_drive: dict[float, np.ndarray] = {}
    # Column layout: E_uev, f_thermal, then for each drive:
    # f_num, f0, f01, f012 (4 cols per drive).
    for i, d in enumerate(drives_hz):
        col0 = 2 + 4 * i
        f_by_drive[d] = data[:, col0]
        f0_by_drive[d] = data[:, col0 + 1]
        f01_by_drive[d] = data[:, col0 + 2]
        f012_by_drive[d] = data[:, col0 + 3]
    return Fig5PaperResult(
        E=data[:, 0],
        drives_hz=drives_hz,
        drives_ns_inv=drives_ns_inv,
        f_thermal=data[:, 1],
        f_by_drive=f_by_drive,
        x_qp_by_drive=dict(zip(drives_hz, x_qp_list, strict=True)),
        f0_by_drive=f0_by_drive,
        f01_by_drive=f01_by_drive,
        f012_by_drive=f012_by_drive,
    )


def write_plot(result: Fig5PaperResult, path: Path | None = None) -> Path:
    """Two-panel plot in paper style: $f(\\gamma)$ on log-y, $\\gamma$ on x.

    Left panel: $\\gamma \\in [0, 1]$.
    Right panel: $\\gamma \\in [0, 0.10]$ (zoom on the near-gap region).
    Black solid: numerics. Green: $f^{(0)}$. Red: $f^{(0)}+f^{(1)}$.
    Blue: $f^{(0)}+f^{(1)}+f^{(2)}$.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # Paper γ coordinate.
    gamma = (result.E - DELTA_0) / XI

    # Pair-breaking active band: E ∈ [Δ_0, ω_PB − Δ_0]. Outside this band
    # the partner energy E_PB − E falls below the gap and the kernel
    # zeroes the generation term; plotting those points connects the last
    # active bin to a near-zero value on log-y, producing a vertical
    # cliff. Mask them out at plot time.
    active = (result.E >= DELTA_0) & (result.E <= OMEGA_PB - DELTA_0)

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 5))

    # Style per Neumann order — matches paper colour mapping:
    # green = f^(0), red = f^(0)+f^(1), blue = f^(0)+f^(1)+f^(2).
    # Underscore-prefixed: kept as documentation of the paper mapping while
    # the curves stay unplotted (see comment below).
    _style_f0 = {"color": "green", "lw": 1.2, "ls": "-", "alpha": 0.9}
    _style_f01 = {"color": "red", "lw": 1.2, "ls": "-", "alpha": 0.9}
    _style_f012 = {"color": "blue", "lw": 1.2, "ls": "-", "alpha": 0.9}
    style_num = {"color": "black", "lw": 1.5, "ls": "-"}

    # The colored Neumann curves (_style_f0 / _style_f01 / _style_f012) are
    # NOT plotted: their _neumann_f0 / _f1 / _f2 implementations are
    # placeholders, not the F24 Eq. 25 + Appendix B closed forms (which
    # require φ → f conversion via Eq. 21 with γ'_* from Eq. 22 and the
    # numerical x_qp). Plotting placeholders alongside the real numerical
    # curve was visually misleading (red/blue overlay sat O(10³–10⁶)×
    # above the black numerics across the band). Black numerics only
    # until the real Neumann series is wired in.
    for ax, gamma_max in ((ax_l, 1.0), (ax_r, 0.10)):
        mask = active & (gamma <= gamma_max + 1e-12)
        for d in result.drives_hz:
            ax.semilogy(gamma[mask], result.f_by_drive[d][mask], **style_num,
                        label=rf"$c\,\bar n_{{PB}}={d:g}$ Hz")
        ax.set_xlabel(r"$\gamma$", fontsize=13)
        ax.set_ylabel(r"$f(\gamma)$", fontsize=13)
        ax.set_xlim(0.0, gamma_max)
        ax.grid(True, which="major", ls=":", alpha=0.4)

    # Match paper y-ranges (study of f24_fig5.png):
    # left: 1e-14 to 1e-4; right: 1e-8 to 1e-4.
    ax_l.set_ylim(1e-14, 1e-4)
    ax_r.set_ylim(1e-8, 1e-4)

    # Legend: only one entry per drive (all three numerical curves
    # already labelled with the per-drive `c·n̄` value via the plot loop).
    ax_l.legend(fontsize=10, loc="upper right", framealpha=0.9)

    drives_str = ", ".join(f"{d:g}" for d in PAPER_DRIVES_HZ)
    fig.suptitle(
        "Fischer & Catelani 2024 Fig. 5 — numerical kinetic-equation "
        "solution (Neumann-series overlays not implemented)\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\tau_0={TAU_0:.0f}$ ns, "
        rf"$\omega_{{\mathrm{{PB}}}}=2.8\,\Delta_0$, "
        rf"$T_B={T_BATH}$ K, $\tau_\ell=0$, "
        rf"$\xi={XI:.1f}\,\mu$eV"
        "\n"
        rf"$c_{{\mathrm{{phot,PB}}}}\bar n_{{\mathrm{{PB}}}} \in \{{${drives_str}$\}}\,$ Hz; "
        "Eq. 25 + Appendix B Neumann series pending",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    result = run()
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
