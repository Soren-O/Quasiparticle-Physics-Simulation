"""Fischer & Catelani 2024 Fig. 8 — qpsim-native characterization at paper topology.

This is a **paper-topology** Fig. 8 characterization: $x_{\\rm qp}$ vs $T_B$
in the low-heating regime at three fixed pair-breaking drive products
$c_{\\rm phot,PB}\\bar n_{\\rm PB}\\in\\{10^{-2},10^{-4},10^{-6}\\}$~Hz.
The artifact and plot contain kinetic-equation numerics only. The
paper's analytical steady-state density curve is not implemented.

The companion script :mod:`fig8_xqp_pb` already exists but (a) bakes
the drive products in as ns$^{-1}$ rather than Hz (a 9-decade unit
slip — see audit below), and (b) uses five drive levels rather than
the paper's three. This script enforces the published Hz units and
the paper's three-curve stratification, with naming honesty so a
downstream consumer is not misled until both gaps below close.

Outstanding paper-parity gaps (load-bearing)
--------------------------------------------

1. **Hz ↔ ns$^{-1}$ unit conversion.** The paper quotes drive
   products $c_{\\rm phot,PB}\\bar n_{\\rm PB}$ in Hz; the simulator's
   :mod:`qpsim.collisions.pair_breaking_photon` works in ns$^{-1}$.
   The conversion is pinned at module load via :data:`HZ_TO_NS_INV`
   and asserted at the top of :func:`run`. The companion
   :mod:`fig8_xqp_pb` script silently elides this conversion — its
   curves are at drive products $10^9$× higher than the paper's, so
   the two scripts are **not** comparable.

2. **Analytical density-equation dashed overlay.** The paper plots
   solid numerical curves with dashed analytic steady-state density
   curves. The analytic formula is not implemented here. A former
   heuristic $\\sqrt{c\\bar n\\tau_0 U}$ stand-in was neither the paper
   formula nor an acceptance test; v4 removes it from the result, CSV,
   and plot. Add an independently checked implementation and tests
   before publishing an analytical overlay.

Once both tickets land, this script becomes the paper-faithful Fig. 8
reproduction. At that point: rename the artifacts to
``fischer2024_fig8_paper.{csv,pdf}``, drop the "qpsim-native" wording
from the plot title, and remove the warning text from the run banner.

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV:

    Δ_0     = 189 μeV
    τ_0     = 63 ns      (faster than F23 — different Al film)
    ω_PB    = 2.8 · Δ_0  (above 2Δ, pair-breaking active)
    τ_ℓ     = 0          (thermal-phonon shortcut)
    n̄_PB    = 1 × 10^6   (large-n̄ split; stimulated product dominates)
    c·n̄ ∈ {1e-2, 1e-4, 1e-6} Hz   (paper Fig. 8 stratification)

Usage --- generate baseline + PDF::

    python -m validation.fischer_2024.fig8_paper
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import pairwise
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
    validated_numeric_array,
    write_artifact,
)

# ── Fischer & Catelani 2024, Sec. IV parameters ──────────────────────

DELTA_0 = 189.0  # μeV
TAU_0 = 63.0  # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_PB = 2.8 * DELTA_0  # 529.2 μeV  (pair-breaking active above 2Δ)
# Holding c·n̄ fixed controls the stimulated terms. The kernel also contains
# a spontaneous c·(n̄+1) term, so this split is a large-n̄ approximation
# with relative correction ≈ 1/N_BAR_PB = 1e-6.
N_BAR_PB = 1e6

# Paper grid: 810 bins so ω_PB / dE = 252 is integer-commensurate.
# Same grid as :mod:`fig8_xqp_pb` and :mod:`figs_5_7_fe_pb`.
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810

# ── Hz ↔ ns⁻¹ unit audit ─────────────────────────────────────────────
#
# Paper quotes c_phot_PB · n̄_PB in Hz (cycles per second).
# qpsim's pair_breaking_photon kernel multiplies c_phot_PB into the
# kinetic equation in ns⁻¹.  Conversion is exact:
#
#     1 Hz = 1 cycle / s = 1e-9 cycles / ns = 1e-9 ns⁻¹
#
# Pinned and asserted at the top of run() to prevent the 9-decade
# slip baked into the companion fig8_xqp_pb script.
HZ_TO_NS_INV = 1e-9

# Paper-target drive products (Hz). F24 Fig. 5 caption pins these
# as ``c_phot_PB · n̄_PB ∈ {10⁻², 10⁻⁴, 10⁻⁶} Hz``. The Fig. 8 caption
# only states "fixed numbers of pair-breaking photons (see parameter
# in the inset)" without explicit values, so we use the Fig. 5 set.
PAPER_DRIVES_HZ: tuple[float, ...] = (1e-2, 1e-4, 1e-6)

# T_B sweep: paper Fig. 8 plots T_B/T_c on the x-axis; we sweep T_B in
# kelvin and convert at plot time. Range chosen to cover the paper's
# typical [0.04, 0.20] range of T_B/T_c (≈ 0.05–0.25 K for T_c ≈ 1.244 K).
T_BATH_VALUES: np.ndarray = np.linspace(0.05, 0.30, 12)

# ── ρ_F for absolute-density (left-axis) labelling ───────────────────
#
# Paper Fig. 8 plots N_qp [1/μm³] on the left axis and a dimensionless
# x_qp on the right axis. F24 doesn't redefine ρ_F; we use the F23
# appendix value:
#
#     ρ_F = 1.74 × 10⁴ / (μeV · μm³)  [single-spin DOS at the Fermi level for Al]
#
# qpsim's ``qp_fraction`` returns ``x_qp_qpsim = N_qp / (4 ρ_F Δ)``;
# the paper's ``x_qp`` is ``N_qp / (2 ρ_F Δ)``. So:
#
#     x_qp_paper = 2 · x_qp_qpsim
#     N_qp [1/μm³]   = 4 · ρ_F · Δ_0 · x_qp_qpsim
#
RHO_F_INV_UEV_UM3 = 1.74e4
NQP_PER_X_QP_QPSIM = 4.0 * RHO_F_INV_UEV_UM3 * DELTA_0  # ≈ 1.32e7 [1/μm³]

ARTIFACT_SCHEMA = "qpsim.fischer2024.fig8_qpsim_native.v5"
NEWTON_TOL = 1.0e-10
NEWTON_BACKWARD_ERROR_TOL = 1.0e-6
NEWTON_MAX_ITER = 500
LIVE_CERTIFICATE_SCOPE = "independently-reassembled-live-state"
SUMMARY_CERTIFICATE_SCOPE = "producer-asserted-summary-only; f(E)-state-omitted"


@dataclass(frozen=True)
class Fig8PaperResult:
    """Arrays returned by :func:`run`."""

    T_bath: np.ndarray  # shape (NT,) in K
    drives_hz: tuple[float, ...]  # paper drive products in Hz
    drives_ns_inv: tuple[float, ...]  # converted to ns⁻¹
    x_qp_thermal: np.ndarray  # shape (NT,)
    x_qp_num_by_drive: dict[float, np.ndarray]  # drive (Hz) → (NT,) numerical
    qp_backward_error_by_drive: dict[float, np.ndarray]
    qp_residual_inf_by_drive: dict[float, np.ndarray]
    qp_number_backward_error_by_drive: dict[float, np.ndarray]
    # Live runs retain f(E) so artifact generation can bind the summary and
    # certificate to the actual solved state. Readback is summary-only and
    # cannot independently reassemble those certificates.
    f_by_drive: dict[float, np.ndarray] | None = None
    certificate_scope: str = LIVE_CERTIFICATE_SCOPE


def solver_fingerprint() -> dict[str, Any]:
    return {
        "continuation": "reset-per-temperature_strong-to-weak-full-state",
        "delta_0_uev": DELTA_0,
        "drives_hz": list(PAPER_DRIVES_HZ),
        "drives_ns_inv": [d * HZ_TO_NS_INV for d in PAPER_DRIVES_HZ],
        "e_max_factor": E_MAX_FACTOR,
        "e_min_factor": E_MIN_FACTOR,
        "hz_to_ns_inv": HZ_TO_NS_INV,
        "n_bar_pb": N_BAR_PB,
        "newton_backward_error_tol": NEWTON_BACKWARD_ERROR_TOL,
        "newton_max_iter": NEWTON_MAX_ITER,
        "newton_tol": NEWTON_TOL,
        "num_bins": NUM_BINS,
        "omega_pb_uev": OMEGA_PB,
        "persisted_certificate_scope": SUMMARY_CERTIFICATE_SCOPE,
        "source_sha256": source_hashes(Path(__file__)),
        "t_bath_k": np.asarray(T_BATH_VALUES, dtype=float).tolist(),
        "t_c_k": T_C,
        "tau_0_ns": TAU_0,
        "use_thermal_phonons": True,
    }


def _point_id(T_bath: float, drive_hz: float) -> str:
    return f"T_bath_K={T_bath:.17e}|drive_hz={drive_hz:.17e}"


def _columns() -> list[str]:
    columns = ["T_bath_K", "x_qp_thermal"]
    for drive_hz in PAPER_DRIVES_HZ:
        suffix = f"{drive_hz:.17e}_hz"
        columns.append(f"x_qp_num_drive_{suffix}")
    return columns


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state(material: Material, T_bath: float) -> T3DiffusionState:
    """Build a fresh thermal-seed state at the paper grid + given T_B."""
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),  # τ_ℓ = 0 throughout F24 Fig. 8
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    f_FD = fermi_dirac_occupation(E, T_bath)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _assert_unit_audit() -> None:
    """Sanity-check the Hz → ns⁻¹ conversion before any kernel call.

    Pin the conversion factor (``HZ_TO_NS_INV = 1e-9``) and verify
    that the paper-target drive products land in the expected ns⁻¹
    range when funneled through the conversion. The companion
    :mod:`fig8_xqp_pb` script silently uses ns⁻¹ values that look
    numerically identical to the paper's Hz values, which is a
    9-decade slip; this assertion makes that slip impossible here.
    """
    assert HZ_TO_NS_INV == 1e-9, (
        f"Hz -> ns^-1 conversion mis-pinned: {HZ_TO_NS_INV} != 1e-9. "
        "Refusing to run a paper-target script with a corrupted unit factor."
    )
    # Independently verify via dimensional analysis:
    # 1 Hz = 1 / s = 1e-9 / ns
    derived = 1.0 / 1e9
    assert abs(HZ_TO_NS_INV - derived) < 1e-30, (
        f"HZ_TO_NS_INV = {HZ_TO_NS_INV} disagrees with dimensional derivation 1/1e9 = {derived}."
    )
    # Sanity-check that paper drive products convert into the expected
    # ns⁻¹ window for the PB-photon kernel.
    for d_hz in PAPER_DRIVES_HZ:
        d_ns = d_hz * HZ_TO_NS_INV
        assert 1e-20 < d_ns < 1e-5, (
            f"Drive {d_hz} Hz -> {d_ns} ns^-1 is outside the expected "
            f"PB-photon kernel range; suspect unit slip."
        )
    assert all(stronger > weaker for stronger, weaker in pairwise(PAPER_DRIVES_HZ)), (
        "PAPER_DRIVES_HZ must be unique and strictly strong-to-weak."
    )


def run() -> Fig8PaperResult:
    """Sweep T_B at the three paper-caption drive products."""
    _assert_unit_audit()

    material = _material()
    backend = T3DiffusionBackend()

    # Convert Hz → ns⁻¹ once, up front, so the kernel only ever sees
    # ns⁻¹ values.
    drives_ns_inv: tuple[float, ...] = tuple(d * HZ_TO_NS_INV for d in PAPER_DRIVES_HZ)

    # Commensurability probe (all T_B share the same grid).
    T_values = np.asarray(T_BATH_VALUES, dtype=float)
    probe_state = _build_state(material, float(T_values[0]))
    dE_scalar = float(probe_state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"omega_PB={OMEGA_PB} not integer-commensurate with dE={dE_scalar:.4f} "
            f"(frac_err={frac_err:.2e}). Choose NUM_BINS so omega_PB/dE is integer."
        )

    nT = T_values.size
    x_thermal = np.zeros(nT)
    x_num: dict[float, np.ndarray] = {d: np.zeros(nT) for d in PAPER_DRIVES_HZ}
    qp_backward: dict[float, np.ndarray] = {d: np.zeros(nT) for d in PAPER_DRIVES_HZ}
    qp_residual: dict[float, np.ndarray] = {d: np.zeros(nT) for d in PAPER_DRIVES_HZ}
    qp_number_backward: dict[float, np.ndarray] = {
        d: np.zeros(nT) for d in PAPER_DRIVES_HZ
    }
    f_by_drive: dict[float, np.ndarray] = {d: np.zeros((nT, NUM_BINS)) for d in PAPER_DRIVES_HZ}

    for i, T in enumerate(T_values):
        state = _build_state(material, float(T))
        # Thermal reference (no PB drive); state.f already = f_FD(T_B).
        x_thermal[i] = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)

        seed = state
        for d_hz, d_ns in zip(PAPER_DRIVES_HZ, drives_ns_inv, strict=True):
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                # Fix the stimulated c·n̄ product. The spontaneous c term
                # leaves a relative 1/N_BAR_PB factorization dependence.
                "c_phot_PB": d_ns / N_BAR_PB,
            }
            driven = backend.steady_state(
                seed,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                # The dimensional gate follows this validation's established
                # tier. The independent backward-error gate still rejects an
                # unchanged thermal seed; strong-to-weak continuation above
                # supplies a feasible, well-scaled Newton starting state.
                newton_tol=NEWTON_TOL,
                newton_backward_error_tol=NEWTON_BACKWARD_ERROR_TOL,
                newton_max_iter=NEWTON_MAX_ITER,
            )
            seed = driven
            f_by_drive[d_hz][i] = driven.f
            x_num[d_hz][i] = qp_fraction(
                driven.f,
                driven.spectral,
                delta_0=DELTA_0,
            )
            certificate = qp_certificate(
                driven,
                pb_photon_params=pb_params,
                residual_inf_limit=NEWTON_TOL,
            )
            qp_backward[d_hz][i] = certificate.backward_error
            qp_residual[d_hz][i] = certificate.residual_inf
            qp_number_backward[d_hz][i] = (
                certificate.qp_number_backward_error
            )
            print(
                f"  T_B={T:.3f} K  c*nbar={d_hz:.0e} Hz "
                f"({d_ns:.2e} ns^-1)  x_qp(num)={x_num[d_hz][i]:.3e}",
                flush=True,
            )

    return Fig8PaperResult(
        T_bath=T_values.copy(),
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=drives_ns_inv,
        x_qp_thermal=x_thermal,
        x_qp_num_by_drive=x_num,
        qp_backward_error_by_drive=qp_backward,
        qp_residual_inf_by_drive=qp_residual,
        qp_number_backward_error_by_drive=qp_number_backward,
        f_by_drive=f_by_drive,
    )


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer2024_fig8_qpsim_native.csv`` to flag that this is a
    qpsim-native numerical sweep at paper topology. The paper's analytic
    density formula is not implemented or serialized.
    """
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer2024_fig8_qpsim_native.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(
    result: Fig8PaperResult,
    path: Path,
    *,
    producer: ProducerIdentity,
    companion_pdf: CompanionArtifactRecord | None = None,
) -> Path:
    """Write the T_B grid, thermal curve, and per-drive numerical curves."""
    require_staging_path(path, baseline_path(), artifact_kind="CSV")
    expected_T = np.asarray(T_BATH_VALUES, dtype=float)
    expected_ns = tuple(d * HZ_TO_NS_INV for d in PAPER_DRIVES_HZ)
    if not np.array_equal(result.T_bath, expected_T):
        raise ValueError("Fig. 8 T_bath axis must exactly match T_BATH_VALUES.")
    if result.drives_hz != PAPER_DRIVES_HZ:
        raise ValueError("Fig. 8 drive axis must exactly match PAPER_DRIVES_HZ.")
    if result.drives_ns_inv != expected_ns:
        raise ValueError("Fig. 8 converted drive axis is stale.")
    if result.f_by_drive is None:
        raise ArtifactValidationError(
            "Fig. 8 baseline generation requires full returned f(E) states; "
            "a summary readback cannot be re-certified."
        )
    if result.certificate_scope != LIVE_CERTIFICATE_SCOPE:
        raise ArtifactValidationError(
            "Fig. 8 baseline generation accepts only independently "
            "reassembled live-state certificates."
        )
    mappings = {
        "x_qp_num_by_drive": result.x_qp_num_by_drive,
        "qp_backward_error_by_drive": result.qp_backward_error_by_drive,
        "qp_residual_inf_by_drive": result.qp_residual_inf_by_drive,
        "qp_number_backward_error_by_drive": (
            result.qp_number_backward_error_by_drive
        ),
        "f_by_drive": result.f_by_drive,
    }
    for name, mapping in mappings.items():
        if set(mapping) != set(PAPER_DRIVES_HZ):
            raise ArtifactValidationError(f"Fig. 8 {name} keys must exactly match PAPER_DRIVES_HZ.")
    validated_numeric_array(
        result.x_qp_thermal,
        context="Fig. 8 thermal x_qp",
        expected_shape=(expected_T.size,),
        lower=0.0,
    )
    rows: list[list[float]] = []
    certificates: dict[str, QPCertificate] = {}
    for i, T_bath in enumerate(result.T_bath):
        base_state = _build_state(_material(), float(T_bath))
        expected_thermal = float(qp_fraction(base_state.f, base_state.spectral, delta_0=DELTA_0))
        if not np.isclose(
            float(result.x_qp_thermal[i]),
            expected_thermal,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ArtifactValidationError(
                f"Fig. 8 thermal x_qp at T_bath={T_bath:g} K is inconsistent."
            )
        row = [float(T_bath), float(result.x_qp_thermal[i])]
        for drive_hz, drive_ns_inv in zip(
            result.drives_hz,
            result.drives_ns_inv,
            strict=True,
        ):
            f_states = validated_numeric_array(
                result.f_by_drive[drive_hz],
                context=f"Fig. 8 f(E) states at drive={drive_hz:g} Hz",
                expected_shape=(expected_T.size, NUM_BINS),
                lower=0.0,
                upper=1.0,
            )
            x_num = validated_numeric_array(
                result.x_qp_num_by_drive[drive_hz],
                context=f"Fig. 8 numerical x_qp at drive={drive_hz:g} Hz",
                expected_shape=(expected_T.size,),
                lower=0.0,
            )
            expected_x_qp = float(qp_fraction(f_states[i], base_state.spectral, delta_0=DELTA_0))
            if not np.isclose(
                float(x_num[i]),
                expected_x_qp,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ArtifactValidationError(
                    f"Fig. 8 numerical x_qp at T_bath={T_bath:g} K, "
                    f"drive={drive_hz:g} Hz is inconsistent with f(E)."
                )
            row.append(float(x_num[i]))
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                "c_phot_PB": drive_ns_inv / N_BAR_PB,
            }
            reassembled = qp_certificate(
                replace(base_state, f=f_states[i].copy()),
                pb_photon_params=pb_params,
                residual_inf_limit=NEWTON_TOL,
            )
            stamped = QPCertificate(
                backward_error=float(result.qp_backward_error_by_drive[drive_hz][i]),
                residual_inf=float(result.qp_residual_inf_by_drive[drive_hz][i]),
                qp_number_backward_error=float(
                    result.qp_number_backward_error_by_drive[drive_hz][i]
                ),
            )
            certificates[_point_id(float(T_bath), drive_hz)] = bind_certificate(
                stamped,
                reassembled,
                context=f"Fig. 8 T_bath={T_bath:g} K, drive={drive_hz:g} Hz",
                residual_inf_limit=NEWTON_TOL,
            )
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


def read_baseline(
    path: Path | None = None,
    *,
    accept_producer_certificate_claims: bool = False,
) -> Fig8PaperResult:
    """Read the summary only after explicit acceptance of producer claims.

    The CSV omits every solved ``f(E)`` state. Its certificate stamps are
    authenticated producer records, but a reader cannot independently
    reconstruct the QP equation from the persisted summary. Callers needing
    independently reassembled physics evidence must regenerate the sweep.
    """
    if not accept_producer_certificate_claims:
        raise ArtifactValidationError(
            "Fig. 8 summary omits solved f(E) states, so its QP certificate "
            "stamps are producer assertions only. Pass "
            "accept_producer_certificate_claims=True to read the summary, "
            "or regenerate for independent live-state certification."
        )
    if path is None:
        path = baseline_path()
    T_values = np.asarray(T_BATH_VALUES, dtype=float)
    artifact = read_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=solver_fingerprint(),
        columns=_columns(),
        expected_row_count=T_values.size,
        expected_certificate_ids=[
            _point_id(float(T), drive_hz) for T in T_values for drive_hz in PAPER_DRIVES_HZ
        ],
        target_qp_residual_inf=NEWTON_TOL,
        companion_pdf_path=path.with_suffix(".pdf"),
        require_companion_pdf=path.resolve() == baseline_path().resolve(),
    )
    data = artifact.data
    if not np.array_equal(data[:, 0], T_values):
        raise ArtifactValidationError(f"Artifact at {path} has a stale T_bath axis.")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ArtifactValidationError(f"Artifact at {path} T_bath axis is not strictly increasing.")
    validated_numeric_array(
        data[:, 1:],
        context=f"Artifact at {path} x_qp values",
        expected_shape=(T_values.size, 1 + len(PAPER_DRIVES_HZ)),
        lower=0.0,
    )
    # Column layout: T_bath, x_qp_thermal, then one numerical x_qp per drive.
    x_num: dict[float, np.ndarray] = {}
    for i, d in enumerate(PAPER_DRIVES_HZ):
        x_num[d] = data[:, 2 + i]
    return Fig8PaperResult(
        T_bath=data[:, 0],
        drives_hz=PAPER_DRIVES_HZ,
        drives_ns_inv=tuple(d * HZ_TO_NS_INV for d in PAPER_DRIVES_HZ),
        x_qp_thermal=data[:, 1],
        x_qp_num_by_drive=x_num,
        qp_backward_error_by_drive={
            d: np.asarray(
                [artifact.certificates[_point_id(float(T), d)].backward_error for T in T_values]
            )
            for d in PAPER_DRIVES_HZ
        },
        qp_residual_inf_by_drive={
            d: np.asarray(
                [artifact.certificates[_point_id(float(T), d)].residual_inf for T in T_values]
            )
            for d in PAPER_DRIVES_HZ
        },
        qp_number_backward_error_by_drive={
            d: np.asarray(
                [
                    artifact.certificates[
                        _point_id(float(T), d)
                    ].qp_number_backward_error
                    for T in T_values
                ]
            )
            for d in PAPER_DRIVES_HZ
        },
        f_by_drive=None,
        certificate_scope=SUMMARY_CERTIFICATE_SCOPE,
    )


def write_plot(result: Fig8PaperResult, path: Path) -> Path:
    """Plot numerical x_qp vs T_B with the paper's axes and drive topology."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_staging_path(path, plot_path(), artifact_kind="PDF")
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Paper Fig. 8 axes: x = T_B/T_c, left y = N_qp [1/μm³], right y = x_qp.
    # qpsim's x_qp = N_qp/(4 ρ_F Δ); paper's x_qp = N_qp/(2 ρ_F Δ).
    # Convert at plot time so the right axis matches the paper convention.
    x = result.T_bath / T_C
    nqp_thermal = NQP_PER_X_QP_QPSIM * result.x_qp_thermal

    # Thermal floor (no PB drive).
    ax.loglog(x, nqp_thermal, "k--", lw=1.5, label=r"thermal (no PB drive)")

    # Drive curves: paper uses {green, blue, red} for low → high drive.
    paper_rgb = ["#1f7a3f", "#1f4e9c", "#c83737"]  # low → high in PAPER_DRIVES_HZ ascending
    # PAPER_DRIVES_HZ is ordered high→low (1e-2, 1e-4, 1e-6); reverse so the
    # brightest red sits at the highest drive.
    drives_high_to_low = list(result.drives_hz)
    color_by_drive = {d: paper_rgb[2 - i] for i, d in enumerate(drives_high_to_low)}
    for d_hz in result.drives_hz:
        color = color_by_drive[d_hz]
        nqp_num = NQP_PER_X_QP_QPSIM * result.x_qp_num_by_drive[d_hz]
        ax.loglog(
            x,
            nqp_num,
            "-",
            color=color,
            lw=1.6,
            label=rf"$c\,\bar n_{{PB}} = {d_hz:g}$ Hz",
        )

    ax.set_xlabel(r"$T_B / T_c$", fontsize=13)
    ax.set_ylabel(r"$N_{\mathrm{qp}}$ $[1/\mu\mathrm{m}^3]$", fontsize=13)
    ax.set_title(
        "F24 Fig. 8 topology only — qpsim-native numerical curves\n"
        "Drive levels borrowed from the Fig. 5 caption; Fig. 8 inset "
        "values are unverified",
        fontsize=10,
    )

    # Twin right-axis carries the paper-convention dimensionless x_qp.
    # The conversion is N_qp = 2 ρ_F Δ x_qp_paper, so right_y/left_y =
    # 1/(2 ρ_F Δ).
    ax_r = ax.twinx()
    ax_r.set_yscale("log")
    ymin, ymax = ax.get_ylim()
    nqp_per_xqp_paper = 2.0 * RHO_F_INV_UEV_UM3 * DELTA_0  # ≈ 6.58e6
    ax_r.set_ylim(ymin / nqp_per_xqp_paper, ymax / nqp_per_xqp_paper)
    ax_r.set_ylabel(r"$x_{\mathrm{qp}} = N_{\mathrm{qp}} / (2\rho_F\Delta)$", fontsize=12)

    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.grid(True, which="major", ls="-", lw=0.5, color="0.85")
    ax.grid(True, which="minor", ls=":", lw=0.5, color="0.92")
    ax.legend(fontsize=10, loc="lower right", framealpha=0.95, edgecolor="0.7")

    paper_ratio_note = (
        f"qpsim-native (axes converted to paper convention)\n"
        rf"$\rho_F=1.74\!\times\!10^4 / (\mu eV\!\cdot\!\mu m^3)$, "
        rf"$T_c={T_C:.3f}$ K; paper analytic overlay not implemented"
        "\nCSV omits f(E): persisted certificate stamps are producer claims"
    )
    fig.text(
        0.5,
        0.01,
        paper_ratio_note,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="0.45",
    )

    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.92))
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Fig. 8 paper-topology qpsim characterization ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, "
        f"omega_PB={OMEGA_PB:.2f} micro-eV, nbar_PB={N_BAR_PB:.0e}, tau_l=0"
    )
    print(f"  Grid: NE={NUM_BINS}")
    print(
        f"  Drive products: {list(PAPER_DRIVES_HZ)} Hz "
        f"(x HZ_TO_NS_INV={HZ_TO_NS_INV} -> "
        f"{[f'{d * HZ_TO_NS_INV:.2e}' for d in PAPER_DRIVES_HZ]} ns^-1)"
    )
    print(
        f"  T_B sweep: {T_BATH_VALUES.size} points, "
        f"{T_BATH_VALUES[0]:.3f} -> {T_BATH_VALUES[-1]:.3f} K"
    )
    print("  NOTE: The paper's analytic dashed overlay is not implemented or emitted.")
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
