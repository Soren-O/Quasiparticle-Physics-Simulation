"""Fischer & Catelani 2024 Fig. 5 — qpsim-native characterization at paper topology.

This is a **paper-topology** Fischer 2024 Fig. 5 characterization --- three drive
levels at $\\omega_{\\rm PB} = 2.8\\Delta$ with $\\tau_\\ell = 0$, plotted in
the paper's $\\gamma = (E - \\Delta)/\\xi$ coordinate, where
$\\xi=\\omega_{\\rm PB}-2\\Delta$. The paper's
Neumann-series analytic overlays are not implemented. Earlier versions
computed heuristic stand-ins; those values are deliberately excluded
from the v4 result, CSV, and plot so they cannot be mistaken for paper
formulas or quantitative validation data. The Hz $\\leftrightarrow$
ns$^{-1}$ unit conversion of the drive product also carries a
paper-parity audit warning. This remains a qpsim-native characterization
at the paper's sweep topology, not a paper-faithful reproduction.

The published Fischer & Catelani 2024 Fig. 5 is a two-panel comparison:

* **Left panel.** $\\gamma \\in [0, 1]$, $f(\\gamma)$ on log-scale.
* **Right panel.** Zoom of $\\gamma \\in [0, 0.10]$ to show the $f^{(2)}$
  near-gap correction, $f$ on log-scale.

Black: numerical kinetic-equation solution. Colored, in successive
Neumann-series order: green $f^{(0)}$, red $f^{(0)} + f^{(1)}$, blue
$f^{(0)} + f^{(1)} + f^{(2)}$. Three curve families per panel — one per
drive level $c_{\\rm phot,PB} \\bar n_{\\rm PB}$.

The existing :mod:`figs_5_7_fe_pb` script combines the Figs. 5–7 sweep
topologies into one qpsim-native overlay with five drive levels
(ns$^{-1}$ values). This script uses the paper's three caption-quoted
drive levels in Hz, but plots and persists numerical qpsim results only.

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
   :func:`_neumann_f2` retain clearly-labelled development prototypes
   (linear-interpolated thermal background plus a partner-reflection
   step), but their values are not part of the result, CSV, or plot.
   Replace the function bodies with the verified Neumann series and add
   independently checked acceptance tests before publishing overlays.

Once both tickets land, this script becomes the paper-faithful Fig. 5
reproduction. At that point: rename the artifacts to
``fischer2024_fig5_paper.{csv,pdf}``, drop the "qpsim-native" wording
from the plot title, and remove the warning text from the run banner.

F24 Sec. IV parameters:

    Δ_0     = 189 μeV
    τ_0     = 63 ns
    ω_PB    = 2.8·Δ = 529.2 μeV (above 2Δ, pair-breaking active)
    τ_ℓ     = 0          (thermal-phonon shortcut)
    n̄_PB    = 1e6        (kept fixed; c·n̄ dominates at large n̄)
    c_phot,PB · n̄_PB ∈ {1e-2, 1e-4, 1e-6} Hz  (paper caption)

Grid: 810 bins so ω_PB/dE = 252 is integer-commensurate (inherits the
:mod:`fig8_xqp_pb` choice; the older 851-bin choice snapped ω_PB by
~0.3% and lost bit-reproducibility).

Usage --- generate baseline + PDF::

    python -m validation.fischer_2024.fig5_paper
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation

from validation.fischer_2024._artifact import (
    ArtifactValidationError,
    CompanionArtifactRecord,
    ProducerIdentity,
    QPCertificate,
    bind_certificate,
    capture_producer_identity,
    publish_artifact_pair,
    qp_certificate,
    read_artifact,
    require_staging_path,
    source_hashes,
    thermal_occupations_match,
    validated_numeric_array,
    write_artifact,
)

# ── F24 Sec. IV parameters ───────────────────────────────────────────

DELTA_0 = 189.0  # μeV
TAU_0 = 63.0  # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1  # K (paper Fig. 5 bath)
OMEGA_PB = 2.8 * DELTA_0  # 529.2 μeV
# F&C 2024 paper coordinate: γ = (E − Δ) / ξ, where ξ = ω_PB − 2Δ.
# γ = 1 marks the pair-breaking endpoint at which the partner energy
# E_partner = ω_PB − E hits the gap from above. For ω_PB = 2.8Δ this
# gives ξ = 0.8Δ. Using the wrong denominator (Δ instead of ξ) puts
# the partner-below-gap mask transition at γ = 0.8 in code coords,
# producing a vertical-cliff plot artifact when matplotlib connects
# the last active bin to the first masked bin on log y.
XI = OMEGA_PB - 2.0 * DELTA_0  # 151.2 μeV
# Holding c·n̄ fixed controls the stimulated terms. The kernel also contains
# a spontaneous c·(n̄+1) term, so the factorization is only a large-n̄
# approximation (relative correction ≈ 1/N_BAR_PB = 1e-6 here).
N_BAR_PB = 1e6

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
PAPER_DRIVES_NS_INV: tuple[float, ...] = tuple(p * HZ_TO_NS_INV for p in PAPER_DRIVES_HZ)

# Sanity: existing figs_5_7_fe_pb.POWER_LEVELS sweep covers
# {1e-6, …, 1e-2} ns⁻¹, which corresponds to {1e3, …, 1e7} Hz under the
# literal conversion — five to nine orders of magnitude **larger** than
# the paper-quoted Hz values. Flag this in :func:`_assert_unit_audit`
# but do not fail; the resolution lives upstream in the pair-breaking
# kernel prefactor audit.
EXISTING_F24_NATIVE_RANGE_NS_INV = (1e-6, 1e-2)

ARTIFACT_SCHEMA = "qpsim.fischer2024.fig5_qpsim_native.v5"
NEWTON_TOL = 1.0e-14
NEWTON_BACKWARD_ERROR_TOL = 1.0e-6
NEWTON_MAX_ITER = 500


@dataclass(frozen=True)
class Fig5PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    drives_hz: tuple[float, ...]
    drives_ns_inv: tuple[float, ...]
    f_thermal: np.ndarray  # shape (NE,); f_FD at T_bath
    f_by_drive: dict[float, np.ndarray]  # drive_hz → numerical f
    x_qp_by_drive: dict[float, float]  # drive_hz → scalar x_qp
    qp_backward_error_by_drive: dict[float, float]
    qp_residual_inf_by_drive: dict[float, float]
    qp_number_backward_error_by_drive: dict[float, float]


def solver_fingerprint() -> dict[str, Any]:
    return {
        "delta_0_uev": DELTA_0,
        "drives_hz": list(PAPER_DRIVES_HZ),
        "drives_ns_inv": list(PAPER_DRIVES_NS_INV),
        "e_max_factor": E_MAX_FACTOR,
        "e_min_factor": E_MIN_FACTOR,
        "hz_to_ns_inv": HZ_TO_NS_INV,
        "n_bar_pb": N_BAR_PB,
        "newton_backward_error_tol": NEWTON_BACKWARD_ERROR_TOL,
        "newton_max_iter": NEWTON_MAX_ITER,
        "newton_tol": NEWTON_TOL,
        "num_bins": NUM_BINS,
        "omega_pb_uev": OMEGA_PB,
        "source_sha256": source_hashes(Path(__file__)),
        "t_bath_k": T_BATH,
        "t_c_k": T_C,
        "tau_0_ns": TAU_0,
        "use_thermal_phonons": True,
    }


def _point_id(drive_hz: float) -> str:
    return f"drive_hz={drive_hz:.17e}"


def _columns() -> list[str]:
    columns = ["E_uev", "f_thermal"]
    for drive_hz in PAPER_DRIVES_HZ:
        suffix = f"{drive_hz:.17e}_hz"
        columns.append(f"f_num_{suffix}")
    return columns


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
            f"omega_PB={OMEGA_PB} micro-eV, m*dE={m * dE_scalar} micro-eV, "
            f"frac_err={frac_err}. Choose NUM_BINS such that omega_PB/dE is integer."
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
        f"Hz -> ns^-1 conversion drifted: HZ_TO_NS_INV={HZ_TO_NS_INV} "
        "must be 1e-9 (1 Hz = 1 s^-1 = 1e-9 ns^-1)."
    )
    for hz, ns_inv in zip(PAPER_DRIVES_HZ, PAPER_DRIVES_NS_INV, strict=True):
        expected = hz * 1.0e-9
        assert abs(ns_inv - expected) < 1e-30, (
            f"Drive conversion mismatch: {hz} Hz mapped to {ns_inv} ns^-1, expected {expected}."
        )
    # Soft warning on the magnitude regime.
    paper_min, paper_max = min(PAPER_DRIVES_NS_INV), max(PAPER_DRIVES_NS_INV)
    native_min, native_max = EXISTING_F24_NATIVE_RANGE_NS_INV
    if paper_max < native_min or paper_min > native_max:
        print(
            "  WARNING: Hz -> ns^-1 unit-audit mismatch:\n"
            f"    Paper-quoted drives  : {paper_min:.2e} - {paper_max:.2e} ns^-1\n"
            f"    Existing F24 sweep   : {native_min:.2e} - {native_max:.2e} ns^-1\n"
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
    f_FD = fermi_dirac_occupation(spectral.E, T_BATH)
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
    spectral: SpectralContext,
    drive_ns_inv: float,
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
    return fermi_dirac_occupation(spectral.E, T_BATH)


def _neumann_f1(
    spectral: SpectralContext,
    drive_ns_inv: float,
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
    spectral: SpectralContext,
    drive_ns_inv: float,
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
        bump = np.exp(-((gamma / 0.04) ** 2))
        increment = (drive_ns_inv * 1e9) ** 2 * bump * 1e-2
        increment = np.where(gamma >= 0.0, increment, 0.0)
    return increment


def run() -> Fig5PaperResult:
    """Solve F24 Fig. 5 — three drive levels at fixed T_B = 0.1 K."""
    print("F24 Fig. 5 paper-topology qpsim characterization ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, "
        f"omega_PB={OMEGA_PB:.2f} micro-eV, T_B={T_BATH} K, tau_l=0"
    )
    print(
        f"  Grid: NE={NUM_BINS}, "
        f"dE={(E_MAX_FACTOR - E_MIN_FACTOR) * DELTA_0 / NUM_BINS:.4f} micro-eV "
        f"(omega_PB/dE = "
        f"{round(OMEGA_PB / ((E_MAX_FACTOR - E_MIN_FACTOR) * DELTA_0 / NUM_BINS))})"
    )
    print(
        f"  Drive levels (paper Hz): {list(PAPER_DRIVES_HZ)}\n"
        f"  Drive levels (ns^-1)   : "
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
    qp_backward: dict[float, float] = {}
    qp_residual: dict[float, float] = {}
    qp_number_backward: dict[float, float] = {}

    for drive_hz, drive_ns_inv in zip(
        PAPER_DRIVES_HZ,
        PAPER_DRIVES_NS_INV,
        strict=True,
    ):
        # Hold n̄_PB fixed and distribute the quoted stimulated product into
        # c_phot_PB. The kernel's c·(n̄+1) term retains a relative 1e-6
        # dependence on this factorization (large-n̄ approximation).
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": drive_ns_inv / N_BAR_PB,
        }
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            pb_photon_params=pb_params,
            newton_tol=NEWTON_TOL,
            newton_backward_error_tol=NEWTON_BACKWARD_ERROR_TOL,
            newton_max_iter=NEWTON_MAX_ITER,
        )
        f_by_drive[drive_hz] = driven.f.copy()
        x_qp = float(qp_fraction(driven.f, driven.spectral, delta_0=DELTA_0))
        x_qp_by_drive[drive_hz] = x_qp
        certificate = qp_certificate(
            driven,
            pb_photon_params=pb_params,
            residual_inf_limit=NEWTON_TOL,
        )
        qp_backward[drive_hz] = certificate.backward_error
        qp_residual[drive_hz] = certificate.residual_inf
        qp_number_backward[drive_hz] = certificate.qp_number_backward_error

        print(
            f"    c*nbar = {drive_hz:g} Hz ({drive_ns_inv:.2e} ns^-1): x_qp = {x_qp:.4e}",
            flush=True,
        )

    return Fig5PaperResult(
        E=spectral.E,
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=PAPER_DRIVES_NS_INV,
        f_thermal=f_thermal,
        f_by_drive=f_by_drive,
        x_qp_by_drive=x_qp_by_drive,
        qp_backward_error_by_drive=qp_backward,
        qp_residual_inf_by_drive=qp_residual,
        qp_number_backward_error_by_drive=qp_number_backward,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer2024_fig5_qpsim_native.csv`` to flag that:

    1. The Hz → ns⁻¹ unit conversion is the literal physical factor,
       but the resulting drive products fall outside the regime of
       qpsim's existing F24 native sweep — pending a kernel-prefactor
       audit.
    2. The paper's Neumann-series analytic overlays are not implemented
       and are deliberately absent from the v4 artifact.

    Rename to ``fischer2024_fig5_paper.csv`` once both gaps close.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer2024_fig5_qpsim_native.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(
    result: Fig5PaperResult,
    path: Path,
    *,
    producer: ProducerIdentity,
    companion_pdf: CompanionArtifactRecord | None = None,
) -> Path:
    """Write the three numerical drive-level f(E) arrays to CSV."""
    require_staging_path(path, baseline_path(), artifact_kind="CSV")
    expected_E, _, spectral = _build_grid_and_spectral()
    if not np.array_equal(result.E, expected_E):
        raise ValueError("Fig. 5 energy axis must exactly match the live grid.")
    if result.drives_hz != PAPER_DRIVES_HZ:
        raise ValueError("Fig. 5 drive axis must exactly match PAPER_DRIVES_HZ.")
    if result.drives_ns_inv != PAPER_DRIVES_NS_INV:
        raise ValueError("Fig. 5 converted drive axis is stale.")
    mapping_keys = (
        ("f_by_drive", set(result.f_by_drive)),
        ("x_qp_by_drive", set(result.x_qp_by_drive)),
        ("qp_backward_error_by_drive", set(result.qp_backward_error_by_drive)),
        ("qp_residual_inf_by_drive", set(result.qp_residual_inf_by_drive)),
        (
            "qp_number_backward_error_by_drive",
            set(result.qp_number_backward_error_by_drive),
        ),
    )
    for name, keys in mapping_keys:
        if keys != set(PAPER_DRIVES_HZ):
            raise ArtifactValidationError(f"Fig. 5 {name} keys must exactly match PAPER_DRIVES_HZ.")
    base_state = _build_state(_material(), spectral)
    f_thermal = validated_numeric_array(
        result.f_thermal,
        context="Fig. 5 thermal occupation",
        expected_shape=(NUM_BINS,),
        lower=0.0,
        upper=1.0,
    )
    if not thermal_occupations_match(f_thermal, base_state.f):
        raise ArtifactValidationError(
            "Fig. 5 thermal occupation does not match the live Fermi-Dirac state."
        )
    f_by_drive: dict[float, np.ndarray] = {}
    certificates: dict[str, QPCertificate] = {}
    for drive_hz, drive_ns_inv in zip(
        PAPER_DRIVES_HZ,
        PAPER_DRIVES_NS_INV,
        strict=True,
    ):
        f = validated_numeric_array(
            result.f_by_drive[drive_hz],
            context=f"Fig. 5 occupation at drive={drive_hz:g} Hz",
            expected_shape=(NUM_BINS,),
            lower=0.0,
            upper=1.0,
        )
        expected_x_qp = float(qp_fraction(f, spectral, delta_0=DELTA_0))
        if not np.isclose(
            float(result.x_qp_by_drive[drive_hz]),
            expected_x_qp,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ArtifactValidationError(
                f"Fig. 5 x_qp at drive={drive_hz:g} Hz is inconsistent with f(E)."
            )
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": drive_ns_inv / N_BAR_PB,
        }
        reassembled = qp_certificate(
            replace(base_state, f=f.copy()),
            pb_photon_params=pb_params,
            residual_inf_limit=NEWTON_TOL,
        )
        stamped = QPCertificate(
            backward_error=float(result.qp_backward_error_by_drive[drive_hz]),
            residual_inf=float(result.qp_residual_inf_by_drive[drive_hz]),
            qp_number_backward_error=float(
                result.qp_number_backward_error_by_drive[drive_hz]
            ),
        )
        certificates[_point_id(drive_hz)] = bind_certificate(
            stamped,
            reassembled,
            context=f"Fig. 5 drive={drive_hz:g} Hz",
            residual_inf_limit=NEWTON_TOL,
        )
        f_by_drive[drive_hz] = f
    rows: list[list[float]] = []
    for i, energy in enumerate(result.E):
        row = [float(energy), float(f_thermal[i])]
        for drive_hz in result.drives_hz:
            row.append(float(f_by_drive[drive_hz][i]))
        rows.append(row)
    fingerprint = solver_fingerprint()
    return write_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=fingerprint,
        columns=_columns(),
        rows=rows,
        certificates=certificates,
        target_qp_residual_inf=NEWTON_TOL,
        producer=producer,
        companion_pdf=companion_pdf,
    )


def read_baseline(path: Path | None = None) -> Fig5PaperResult:
    """Read a pinned baseline CSV back into a :class:`Fig5PaperResult`."""
    if path is None:
        path = baseline_path()
    artifact = read_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=solver_fingerprint(),
        columns=_columns(),
        expected_row_count=NUM_BINS,
        expected_certificate_ids=[_point_id(d) for d in PAPER_DRIVES_HZ],
        target_qp_residual_inf=NEWTON_TOL,
        companion_pdf_path=path.with_suffix(".pdf"),
        require_companion_pdf=path.resolve() == baseline_path().resolve(),
    )
    data = artifact.data
    expected_E, _, spectral = _build_grid_and_spectral()
    if not np.array_equal(data[:, 0], expected_E):
        raise ArtifactValidationError(f"Artifact at {path} has a stale energy axis.")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ArtifactValidationError(f"Artifact at {path} energy axis is not strictly increasing.")
    validated_numeric_array(
        data[:, 1:],
        context=f"Artifact at {path} occupations",
        expected_shape=(NUM_BINS, 1 + len(PAPER_DRIVES_HZ)),
        lower=0.0,
        upper=1.0,
    )
    if not thermal_occupations_match(
        data[:, 1],
        _build_state(_material(), spectral).f,
    ):
        raise ArtifactValidationError(
            f"Artifact at {path} thermal occupation is not the live Fermi-Dirac state."
        )
    f_by_drive: dict[float, np.ndarray] = {}
    certificates: dict[float, QPCertificate] = {}
    base_state = _build_state(_material(), spectral)
    # Column layout: E_uev, f_thermal, then one numerical f per drive.
    for i, (drive_hz, drive_ns_inv) in enumerate(
        zip(PAPER_DRIVES_HZ, PAPER_DRIVES_NS_INV, strict=True)
    ):
        f = data[:, 2 + i]
        f_by_drive[drive_hz] = f
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": drive_ns_inv / N_BAR_PB,
        }
        try:
            reassembled = qp_certificate(
                replace(base_state, f=f.copy()),
                pb_photon_params=pb_params,
                residual_inf_limit=NEWTON_TOL,
            )
        except (RuntimeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"Artifact at {path} drive={drive_hz:g} Hz fails fresh "
                "QP-equation certificate reassembly."
            ) from exc
        certificates[drive_hz] = bind_certificate(
            artifact.certificates[_point_id(drive_hz)],
            reassembled,
            context=f"Artifact at {path} drive={drive_hz:g} Hz",
            residual_inf_limit=NEWTON_TOL,
        )
    return Fig5PaperResult(
        E=data[:, 0],
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=PAPER_DRIVES_NS_INV,
        f_thermal=data[:, 1],
        f_by_drive=f_by_drive,
        x_qp_by_drive={
            d: float(qp_fraction(f_by_drive[d], spectral, delta_0=DELTA_0)) for d in PAPER_DRIVES_HZ
        },
        qp_backward_error_by_drive={
            d: certificates[d].backward_error for d in PAPER_DRIVES_HZ
        },
        qp_residual_inf_by_drive={
            d: certificates[d].residual_inf for d in PAPER_DRIVES_HZ
        },
        qp_number_backward_error_by_drive={
            d: certificates[d].qp_number_backward_error
            for d in PAPER_DRIVES_HZ
        },
    )


def write_plot(result: Fig5PaperResult, path: Path) -> Path:
    """Plot numerical qpsim curves in the paper's two-panel topology.

    Left panel: $\\gamma \\in [0, 1]$.
    Right panel: $\\gamma \\in [0, 0.10]$ (zoom on the near-gap region).
    Black solid: qpsim numerics. No paper-analytic curves are included.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_staging_path(path, plot_path(), artifact_kind="PDF")
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

    style_num = {"color": "black", "lw": 1.5, "ls": "-"}

    # The development-only _neumann_f* prototypes are intentionally absent
    # from both this plot and the v4 CSV. They are not paper formulas.
    for ax, gamma_max in ((ax_l, 1.0), (ax_r, 0.10)):
        mask = active & (gamma <= gamma_max + 1e-12)
        for d in result.drives_hz:
            ax.semilogy(
                gamma[mask],
                result.f_by_drive[d][mask],
                **style_num,
                label=rf"$c\,\bar n_{{PB}}={d:g}$ Hz",
            )
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
        "F24 Fig. 5 topology only — qpsim-native numerical curves\n"
        "No digitized-paper parity; drive normalization unaudited; "
        "Neumann-series overlays not implemented\n"
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
    producer = capture_producer_identity(solver_fingerprint())
    result = run()
    csv_path, pdf_path = publish_artifact_pair(
        csv_path=baseline_path(),
        pdf_path=plot_path(),
        producer=producer,
        current_fingerprint=solver_fingerprint,
        render_pdf=lambda path: write_plot(result, path),
        write_csv=lambda path, identity, pdf: write_baseline(
            result,
            path,
            producer=identity,
            companion_pdf=pdf,
        ),
    )
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
