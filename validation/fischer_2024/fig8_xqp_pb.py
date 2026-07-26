"""Fischer & Catelani 2024 Fig 8 — x_qp(T_B) under pair-breaking photon drive.

Exercises the pair-breaking photon collision channel
(:mod:`qpsim.collisions.pair_breaking_photon`) at the Fischer-2024
parameter set, sweeps ``T_bath`` for each of five power levels, and
produces the 5-curve x_qp comparison from F24 Fig 8.

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV:

    Δ₀             = 189 μeV
    τ₀             = 63 ns     (note: faster than F23 — different Al film)
    ω_PB           = 2.8 · Δ₀  =  529.2 μeV   (above 2Δ, pair-breaking active)
    n̄_PB           = 1e6
    c_phot_PB × n̄_PB ∈ {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} ns⁻¹   (5 power levels)

Grid: 810 bins so ω_PB/dE = 252 is integer commensurate (the old
851-bin choice snapped ω_PB by ~0.3 %, below the 1% tolerance but
sacrificing bit-reproducibility).

Conventions: the stored CSV columns keep qpsim's Fischer-convention
``qp_fraction`` :math:`x_\\mathrm{qp} = N_\\mathrm{qp}/(4\\rho_F\\Delta_0)`
(:mod:`qpsim.observables.density`), preserving the certified artifact.
F24 Eq. 7 defines :math:`x_\\mathrm{qp} = N_\\mathrm{qp}/(2\\rho_F\\Delta_0)`
— exactly twice qpsim's — so :func:`write_plot` applies the ×2 conversion
at the figure layer and labels the axis with the paper's definition
(audit fix 2026-07-19; the pre-fix plot drew qpsim-convention values
under an unqualified ``x_qp`` label, a factor 2 below paper convention).

Usage::

    python -m validation.fischer_2024.fig8_xqp_pb
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
from qpsim.physics.spectral import SpectralContext

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

# ── F24 Sec. IV parameters ───────────────────────────────────────────

DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_PB = 2.8 * DELTA_0  # 529.2 μeV
N_BAR_PB = 1e6

POWER_LEVELS: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
"""c_phot_PB · n̄_PB products in ns⁻¹ (F24 Sec. IV)."""

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 810  # ω_PB/dE = 252 exactly at this grid

T_BATH_VALUES: tuple[float, ...] = tuple(np.linspace(0.05, 0.22, 8).tolist())

ARTIFACT_SCHEMA = "qpsim.fischer2024.fig8_xqp_pb.v3"
NEWTON_TOL = 1.0e-14
NEWTON_BACKWARD_ERROR_TOL = 1.0e-6
NEWTON_MAX_ITER = 500


@dataclass(frozen=True)
class Fig8Result:
    T_bath: np.ndarray
    powers: tuple[float, ...]
    x_qp_thermal: np.ndarray  # shape (NT,)
    x_qp_by_power: dict[float, np.ndarray]  # power → shape (NT,)
    qp_backward_error_by_power: dict[float, np.ndarray]
    qp_residual_inf_by_power: dict[float, np.ndarray]
    # Full returned states are retained in memory so baseline generation can
    # bind scalar x_qp values and certificate stamps to the solved f(E).
    # Summary artifacts intentionally omit them; read_baseline returns None.
    f_by_power: dict[float, np.ndarray] | None = None


def solver_fingerprint() -> dict[str, Any]:
    """Resolved physics, axes, and solver knobs stamped into the CSV."""
    return {
        "delta_0_uev": DELTA_0,
        "e_max_factor": E_MAX_FACTOR,
        "e_min_factor": E_MIN_FACTOR,
        "n_bar_pb": N_BAR_PB,
        "newton_backward_error_tol": NEWTON_BACKWARD_ERROR_TOL,
        "newton_max_iter": NEWTON_MAX_ITER,
        "newton_tol": NEWTON_TOL,
        "num_bins": NUM_BINS,
        "omega_pb_uev": OMEGA_PB,
        "source_sha256": source_hashes(Path(__file__)),
        "powers_ns_inv": list(POWER_LEVELS),
        "t_bath_k": list(T_BATH_VALUES),
        "t_c_k": T_C,
        "tau_0_ns": TAU_0,
        "use_thermal_phonons": True,
    }


def _point_id(T_bath: float, power: float) -> str:
    return f"T_bath_K={T_bath:.17e}|power_ns_inv={power:.17e}"


def _columns() -> list[str]:
    return ["T_bath_K", "x_qp_thermal"] + [f"x_qp_power_{power:.17e}" for power in POWER_LEVELS]


def _material() -> Material:
    return Material(
        name="Al_Fischer2024",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
    )


def _build_state(material: Material, T_bath: float) -> T3DiffusionState:
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
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_FD,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def run() -> Fig8Result:
    material = _material()
    backend = T3DiffusionBackend()
    T_values = np.array(T_BATH_VALUES)
    x_thermal = np.zeros_like(T_values)
    x_by_power: dict[float, np.ndarray] = {p: np.zeros_like(T_values) for p in POWER_LEVELS}
    qp_backward: dict[float, np.ndarray] = {p: np.zeros_like(T_values) for p in POWER_LEVELS}
    qp_residual: dict[float, np.ndarray] = {p: np.zeros_like(T_values) for p in POWER_LEVELS}
    f_by_power: dict[float, np.ndarray] = {
        p: np.zeros((T_values.size, NUM_BINS)) for p in POWER_LEVELS
    }

    # Verify commensurability once (all T_bath use the same grid).
    probe_state = _build_state(material, float(T_values[0]))
    dE_scalar = float(probe_state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(
            f"omega_PB={OMEGA_PB} is not integer-commensurate with dE={dE_scalar:.4f}"
        )

    for i, T in enumerate(T_values):
        state = _build_state(material, T)
        # Thermal reference (no drive).
        x_thermal[i] = qp_fraction(state.f, state.spectral, delta_0=DELTA_0)

        for power in POWER_LEVELS:
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                "c_phot_PB": power / N_BAR_PB,
            }
            driven = backend.steady_state(
                state,
                use_thermal_phonons=True,
                pb_photon_params=pb_params,
                newton_tol=NEWTON_TOL,
                newton_backward_error_tol=NEWTON_BACKWARD_ERROR_TOL,
                newton_max_iter=NEWTON_MAX_ITER,
            )
            x_by_power[power][i] = qp_fraction(
                driven.f,
                driven.spectral,
                delta_0=DELTA_0,
            )
            f_by_power[power][i] = driven.f
            certificate = qp_certificate(
                driven,
                pb_photon_params=pb_params,
                residual_inf_limit=NEWTON_TOL,
            )
            qp_backward[power][i] = certificate.backward_error
            qp_residual[power][i] = certificate.residual_inf

    return Fig8Result(
        T_bath=T_values,
        powers=POWER_LEVELS,
        x_qp_thermal=x_thermal,
        x_qp_by_power=x_by_power,
        qp_backward_error_by_power=qp_backward,
        qp_residual_inf_by_power=qp_residual,
        f_by_power=f_by_power,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "f24_fig8_xqp_pb.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(
    result: Fig8Result,
    path: Path,
    *,
    producer: ProducerIdentity,
    companion_pdf: CompanionArtifactRecord | None = None,
) -> Path:
    require_staging_path(path, baseline_path(), artifact_kind="CSV")
    expected_T = np.asarray(T_BATH_VALUES, dtype=float)
    if not np.array_equal(result.T_bath, expected_T):
        raise ValueError("Fig. 8 T_bath axis must exactly match T_BATH_VALUES.")
    if result.powers != POWER_LEVELS:
        raise ValueError("Fig. 8 power axis must exactly match POWER_LEVELS.")
    if result.f_by_power is None:
        raise ArtifactValidationError(
            "Fig. 8 baseline generation requires full returned f(E) states; "
            "a summary readback cannot be re-certified."
        )
    mappings = {
        "x_qp_by_power": result.x_qp_by_power,
        "qp_backward_error_by_power": result.qp_backward_error_by_power,
        "qp_residual_inf_by_power": result.qp_residual_inf_by_power,
        "f_by_power": result.f_by_power,
    }
    for name, mapping in mappings.items():
        if set(mapping) != set(POWER_LEVELS):
            raise ArtifactValidationError(f"Fig. 8 {name} keys must exactly match POWER_LEVELS.")
    validated_numeric_array(
        result.x_qp_thermal,
        context="Fig. 8 thermal x_qp",
        expected_shape=(len(T_BATH_VALUES),),
        lower=0.0,
    )
    certificates: dict[str, QPCertificate] = {}
    for i, T_bath in enumerate(T_BATH_VALUES):
        base_state = _build_state(_material(), T_bath)
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
        for power in POWER_LEVELS:
            f_states = validated_numeric_array(
                result.f_by_power[power],
                context=f"Fig. 8 f(E) states at power={power:g}",
                expected_shape=(len(T_BATH_VALUES), NUM_BINS),
                lower=0.0,
                upper=1.0,
            )
            x_values = validated_numeric_array(
                result.x_qp_by_power[power],
                context=f"Fig. 8 x_qp at power={power:g}",
                expected_shape=(len(T_BATH_VALUES),),
                lower=0.0,
            )
            expected_x_qp = float(qp_fraction(f_states[i], base_state.spectral, delta_0=DELTA_0))
            if not np.isclose(
                float(x_values[i]),
                expected_x_qp,
                rtol=1.0e-12,
                atol=0.0,
            ):
                raise ArtifactValidationError(
                    f"Fig. 8 x_qp at T_bath={T_bath:g} K, power={power:g} "
                    "is inconsistent with f(E)."
                )
            pb_params = {
                "omega_PB": OMEGA_PB,
                "n_bar_PB": N_BAR_PB,
                "c_phot_PB": power / N_BAR_PB,
            }
            reassembled = qp_certificate(
                replace(base_state, f=f_states[i].copy()),
                pb_photon_params=pb_params,
                residual_inf_limit=NEWTON_TOL,
            )
            stamped = QPCertificate(
                backward_error=float(result.qp_backward_error_by_power[power][i]),
                residual_inf=float(result.qp_residual_inf_by_power[power][i]),
            )
            certificates[_point_id(T_bath, power)] = bind_certificate(
                stamped,
                reassembled,
                context=f"Fig. 8 T_bath={T_bath:g} K, power={power:g}",
                residual_inf_limit=NEWTON_TOL,
            )
    rows: list[list[float]] = []
    for i, T_bath in enumerate(result.T_bath):
        rows.append(
            [float(T_bath), float(result.x_qp_thermal[i])]
            + [float(result.x_qp_by_power[p][i]) for p in result.powers]
        )
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


def read_baseline(path: Path | None = None) -> Fig8Result:
    if path is None:
        path = baseline_path()
    expected_ids = [_point_id(T_bath, power) for T_bath in T_BATH_VALUES for power in POWER_LEVELS]
    artifact = read_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=solver_fingerprint(),
        columns=_columns(),
        expected_row_count=len(T_BATH_VALUES),
        expected_certificate_ids=expected_ids,
        target_qp_residual_inf=NEWTON_TOL,
        companion_pdf_path=path.with_suffix(".pdf"),
        require_companion_pdf=path.resolve() == baseline_path().resolve(),
    )
    data = artifact.data
    expected_T = np.asarray(T_BATH_VALUES, dtype=float)
    if not np.array_equal(data[:, 0], expected_T):
        raise ArtifactValidationError(f"Artifact at {path} has a stale T_bath axis.")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ArtifactValidationError(f"Artifact at {path} T_bath axis is not strictly increasing.")
    validated_numeric_array(
        data[:, 1:],
        context=f"Artifact at {path} x_qp values",
        expected_shape=(len(T_BATH_VALUES), 1 + len(POWER_LEVELS)),
        lower=0.0,
    )
    qp_backward: dict[float, np.ndarray] = {}
    qp_residual: dict[float, np.ndarray] = {}
    for power in POWER_LEVELS:
        qp_backward[power] = np.asarray(
            [artifact.certificates[_point_id(T, power)].backward_error for T in T_BATH_VALUES]
        )
        qp_residual[power] = np.asarray(
            [artifact.certificates[_point_id(T, power)].residual_inf for T in T_BATH_VALUES]
        )
    return Fig8Result(
        T_bath=data[:, 0],
        powers=POWER_LEVELS,
        x_qp_thermal=data[:, 1],
        x_qp_by_power={p: data[:, i + 2] for i, p in enumerate(POWER_LEVELS)},
        qp_backward_error_by_power=qp_backward,
        qp_residual_inf_by_power=qp_residual,
        f_by_power=None,
    )


def write_plot(result: Fig8Result, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    require_staging_path(path, plot_path(), artifact_kind="PDF")
    path.parent.mkdir(parents=True, exist_ok=True)

    # Stored columns are qpsim-convention N_qp/(4 rho_F Delta_0); F24 Eq. 7's
    # x_qp is N_qp/(2 rho_F Delta_0), so the figure layer applies the ×2 here.
    _XQP_QPSIM_TO_PAPER = 2.0
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(
        result.T_bath,
        _XQP_QPSIM_TO_PAPER * result.x_qp_thermal,
        "k--", lw=1.5, label=r"thermal (no PB drive)",
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(result.powers)))
    for power, color in zip(result.powers, colors, strict=True):
        ax.loglog(
            result.T_bath,
            _XQP_QPSIM_TO_PAPER * result.x_qp_by_power[power],
            lw=2.0,
            color=color,
            label=rf"$c \cdot \bar n = {power:g}$ ns$^{{-1}}$",
        )
    ax.set_xlabel(r"$T_B$ [K]", fontsize=14)
    ax.set_ylabel(r"$x_{qp} = N_{qp}/(2\rho_F\Delta_0)$  (F24 Eq. 7)", fontsize=14)
    ax.set_title(
        "Fischer & Catelani 2024 Fig 8 — PB-photon drive\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\tau_0={TAU_0:.0f}$ ns, "
        rf"$\omega_{{\mathrm{{PB}}}}=2.8\,\Delta_0$",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Fig 8 -- x_qp(T_B) with PB-photon drive ...")
    print(
        f"  Delta_0={DELTA_0} micro-eV, tau_0={TAU_0} ns, "
        f"omega_PB={OMEGA_PB:.2f} micro-eV, nbar_PB={N_BAR_PB:.0e}"
    )
    print(f"  Powers (c*nbar, ns^-1): {list(POWER_LEVELS)}")
    print(f"  Grid: NE={NUM_BINS}")
    print(
        f"  T_bath sweep: {len(T_BATH_VALUES)} points, "
        f"{T_BATH_VALUES[0]:.3f} -> {T_BATH_VALUES[-1]:.3f} K"
    )
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
