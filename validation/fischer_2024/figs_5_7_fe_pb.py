"""Fischer & Catelani 2024 Figs 5-7 — f(E) under pair-breaking photon drive.

Sweeps the five power levels from F24 Sec. IV at fixed bath
temperature ``T_B = 0.1 K`` and records the converged ``f(E)`` per
power, producing the overlay figure that Fischer 2024 presents as
three panels (Figs 5, 6, 7). We emit a single combined plot and a
single baseline CSV; the tests assert bit-identity column-by-column.

Parameters are shared with :mod:`validation.fischer_2024.fig8_xqp_pb`
— same ω_PB / n̄_PB / grid / powers — so the two modules exercise the
same PB-photon collision kernel from orthogonal axes (scan power at
fixed T_B here; scan T_B at each power in Fig 8).

Fischer & Catelani — SciPost Phys. 17, 070 (2024), Sec. IV.

Usage::

    python -m validation.fischer_2024.figs_5_7_fe_pb
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend
from qpsim.observables.density import qp_fraction

from validation.fischer_2024._artifact import (
    ArtifactValidationError,
    QPCertificate,
    bind_certificate,
    qp_certificate,
    read_artifact,
    source_hashes,
    thermal_occupations_match,
    validated_numeric_array,
    write_artifact,
)
from validation.fischer_2024.fig8_xqp_pb import (
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    N_BAR_PB,
    NUM_BINS,
    OMEGA_PB,
    POWER_LEVELS,
    T_C,
    TAU_0,
    _build_state,
    _material,
)

T_BATH_FE = 0.1  # F24 Figs 5-7 fix T_B at 0.1 K

ARTIFACT_SCHEMA = "qpsim.fischer2024.figs_5_7_fe_pb.v2"
NEWTON_TOL = 1.0e-14
NEWTON_BACKWARD_ERROR_TOL = 1.0e-6
NEWTON_MAX_ITER = 500


@dataclass(frozen=True)
class Figs57Result:
    E: np.ndarray
    powers: tuple[float, ...]
    f_thermal: np.ndarray  # shape (NE,); f at T_bath, no PB drive
    f_by_power: dict[float, np.ndarray]  # power → (NE,) driven f
    x_qp_by_power: dict[float, float]  # power → scalar x_qp
    qp_backward_error_by_power: dict[float, float]
    qp_residual_inf_by_power: dict[float, float]


def solver_fingerprint() -> dict[str, Any]:
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
        "source_sha256": source_hashes(
            Path(__file__),
            extra_validation_modules=(Path(__file__).with_name("fig8_xqp_pb.py"),),
        ),
        "powers_ns_inv": list(POWER_LEVELS),
        "t_bath_k": T_BATH_FE,
        "t_c_k": T_C,
        "tau_0_ns": TAU_0,
        "use_thermal_phonons": True,
    }


def _point_id(power: float) -> str:
    return f"power_ns_inv={power:.17e}"


def _columns() -> list[str]:
    return ["E_uev", "f_thermal"] + [f"f_power_{power:.17e}" for power in POWER_LEVELS]


def run() -> Figs57Result:
    material = _material()
    backend = T3DiffusionBackend()

    state = _build_state(material, T_BATH_FE)

    # Commensurability probe once — inherits the Fig-8 grid, so this is a
    # belt-and-braces check.
    dE_scalar = float(state.spectral.dE[0])
    frac_err = abs(OMEGA_PB - round(OMEGA_PB / dE_scalar) * dE_scalar) / OMEGA_PB
    if frac_err > 1e-10:
        raise RuntimeError(f"ω_PB={OMEGA_PB} is not integer-commensurate with dE={dE_scalar:.4f}")

    # Thermal reference (no PB drive): state.f already populated with f_FD.
    f_thermal = state.f.copy()

    f_by_power: dict[float, np.ndarray] = {}
    x_qp_by_power: dict[float, float] = {}
    qp_backward: dict[float, float] = {}
    qp_residual: dict[float, float] = {}

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
        f_by_power[power] = driven.f.copy()
        x_qp_by_power[power] = float(
            qp_fraction(driven.f, driven.spectral, delta_0=DELTA_0),
        )
        certificate = qp_certificate(
            driven,
            pb_photon_params=pb_params,
            residual_inf_limit=NEWTON_TOL,
        )
        qp_backward[power] = certificate.backward_error
        qp_residual[power] = certificate.residual_inf

    return Figs57Result(
        E=state.spectral.E,
        powers=POWER_LEVELS,
        f_thermal=f_thermal,
        f_by_power=f_by_power,
        x_qp_by_power=x_qp_by_power,
        qp_backward_error_by_power=qp_backward,
        qp_residual_inf_by_power=qp_residual,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "f24_figs_5_7_fe_pb.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def write_baseline(result: Figs57Result, path: Path | None = None) -> Path:
    if path is None:
        path = baseline_path()
    base_state = _build_state(_material(), T_BATH_FE)
    expected_E = base_state.spectral.E
    if not np.array_equal(result.E, expected_E):
        raise ValueError("Figs. 5-7 energy axis must exactly match the live grid.")
    if result.powers != POWER_LEVELS:
        raise ValueError("Figs. 5-7 power axis must exactly match POWER_LEVELS.")
    mapping_keys = (
        ("f_by_power", set(result.f_by_power)),
        ("x_qp_by_power", set(result.x_qp_by_power)),
        ("qp_backward_error_by_power", set(result.qp_backward_error_by_power)),
        ("qp_residual_inf_by_power", set(result.qp_residual_inf_by_power)),
    )
    for name, keys in mapping_keys:
        if keys != set(POWER_LEVELS):
            raise ArtifactValidationError(f"Figs. 5-7 {name} keys must exactly match POWER_LEVELS.")
    f_thermal = validated_numeric_array(
        result.f_thermal,
        context="Figs. 5-7 thermal occupation",
        expected_shape=(NUM_BINS,),
        lower=0.0,
        upper=1.0,
    )
    if not thermal_occupations_match(f_thermal, base_state.f):
        raise ArtifactValidationError(
            "Figs. 5-7 thermal occupation does not match the live Fermi-Dirac state."
        )
    f_by_power: dict[float, np.ndarray] = {}
    certificates: dict[str, QPCertificate] = {}
    for power in POWER_LEVELS:
        f = validated_numeric_array(
            result.f_by_power[power],
            context=f"Figs. 5-7 occupation at power={power:g}",
            expected_shape=(NUM_BINS,),
            lower=0.0,
            upper=1.0,
        )
        expected_x_qp = float(qp_fraction(f, base_state.spectral, delta_0=DELTA_0))
        if not np.isclose(
            float(result.x_qp_by_power[power]),
            expected_x_qp,
            rtol=1.0e-12,
            atol=0.0,
        ):
            raise ArtifactValidationError(
                f"Figs. 5-7 x_qp at power={power:g} is inconsistent with f(E)."
            )
        pb_params = {
            "omega_PB": OMEGA_PB,
            "n_bar_PB": N_BAR_PB,
            "c_phot_PB": power / N_BAR_PB,
        }
        reassembled = qp_certificate(
            replace(base_state, f=f.copy()),
            pb_photon_params=pb_params,
            residual_inf_limit=NEWTON_TOL,
        )
        stamped = QPCertificate(
            backward_error=float(result.qp_backward_error_by_power[power]),
            residual_inf=float(result.qp_residual_inf_by_power[power]),
        )
        certificates[_point_id(power)] = bind_certificate(
            stamped,
            reassembled,
            context=f"Figs. 5-7 power={power:g}",
            residual_inf_limit=NEWTON_TOL,
        )
        f_by_power[power] = f
    rows = [
        [float(result.E[i]), float(f_thermal[i])] + [float(f_by_power[p][i]) for p in result.powers]
        for i in range(result.E.size)
    ]
    return write_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=solver_fingerprint(),
        columns=_columns(),
        rows=rows,
        certificates=certificates,
        target_qp_residual_inf=NEWTON_TOL,
    )


def read_baseline(path: Path | None = None) -> Figs57Result:
    if path is None:
        path = baseline_path()
    artifact = read_artifact(
        path,
        schema=ARTIFACT_SCHEMA,
        fingerprint=solver_fingerprint(),
        columns=_columns(),
        expected_row_count=NUM_BINS,
        expected_certificate_ids=[_point_id(power) for power in POWER_LEVELS],
        target_qp_residual_inf=NEWTON_TOL,
    )
    data = artifact.data
    expected_E = _build_state(_material(), T_BATH_FE).spectral.E
    if not np.array_equal(data[:, 0], expected_E):
        raise ArtifactValidationError(f"Artifact at {path} has a stale energy axis.")
    if np.any(np.diff(data[:, 0]) <= 0.0):
        raise ArtifactValidationError(f"Artifact at {path} energy axis is not strictly increasing.")
    validated_numeric_array(
        data[:, 1:],
        context=f"Artifact at {path} occupations",
        expected_shape=(NUM_BINS, 1 + len(POWER_LEVELS)),
        lower=0.0,
        upper=1.0,
    )
    if not thermal_occupations_match(
        data[:, 1],
        _build_state(_material(), T_BATH_FE).f,
    ):
        raise ArtifactValidationError(
            f"Artifact at {path} thermal occupation is not the live Fermi-Dirac state."
        )
    x_qp_by_power = {
        power: float(
            qp_fraction(
                data[:, i + 2], _build_state(_material(), T_BATH_FE).spectral, delta_0=DELTA_0
            )
        )
        for i, power in enumerate(POWER_LEVELS)
    }
    return Figs57Result(
        E=data[:, 0],
        powers=POWER_LEVELS,
        f_thermal=data[:, 1],
        f_by_power={p: data[:, i + 2] for i, p in enumerate(POWER_LEVELS)},
        x_qp_by_power=x_qp_by_power,
        qp_backward_error_by_power={
            p: artifact.certificates[_point_id(p)].backward_error for p in POWER_LEVELS
        },
        qp_residual_inf_by_power={
            p: artifact.certificates[_point_id(p)].residual_inf for p in POWER_LEVELS
        },
    )


def write_plot(result: Figs57Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    x = result.E / DELTA_0
    ax.semilogy(
        x, result.f_thermal, "k-", lw=1.5, alpha=0.8, label=rf"thermal ($T_B={T_BATH_FE}$ K, no PB)"
    )
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(result.powers)))
    for power, color in zip(result.powers, colors, strict=True):
        ax.semilogy(
            x,
            result.f_by_power[power],
            lw=2.0,
            color=color,
            label=rf"$c \cdot \bar n = {power:g}$ ns$^{{-1}}$",
        )
    ax.axvline(
        OMEGA_PB / (2 * DELTA_0),
        color="red",
        ls="--",
        lw=0.8,
        alpha=0.5,
        label=rf"$\omega_{{PB}}/2\Delta = {OMEGA_PB / (2 * DELTA_0):.2f}$",
    )
    ax.set_xlabel(r"$E / \Delta$", fontsize=14)
    ax.set_ylabel(r"$f(E)$", fontsize=14)
    ax.set_title(
        "Fischer & Catelani 2024 Figs 5-7 — PB-photon drive at fixed $T_B$\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\tau_0={TAU_0:.0f}$ ns, "
        rf"$\omega_{{\mathrm{{PB}}}}=2.8\,\Delta_0$, $T_B={T_BATH_FE}$ K",
        fontsize=10,
    )
    ax.set_xlim(1.0, min(E_MAX_FACTOR, 5.0))
    f_stack = np.concatenate([result.f_by_power[p] for p in result.powers])
    f_pos = f_stack[f_stack > 0]
    if f_pos.size > 0:
        ax.set_ylim(max(float(f_pos.min()), 1e-80) / 10, 1.0)
    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer & Catelani 2024 Figs 5-7 — f(E) with PB-photon drive ...")
    print(f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_PB={OMEGA_PB:.2f} μeV, T_B={T_BATH_FE} K")
    print(f"  Powers (c·n̄, ns⁻¹): {list(POWER_LEVELS)}")
    print(f"  Grid: NE={NUM_BINS}")
    result = run()
    for p in result.powers:
        print(f"    c·n̄ = {p:g}:  x_qp = {result.x_qp_by_power[p]:.4e}")
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
