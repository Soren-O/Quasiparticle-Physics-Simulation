"""Fischer 2023 Figs 9-13 — Q_i(P_read) via the self-consistent n̄ loop.

Sweeps drive power ``P_read`` log-spaced from -100 dBm to -60 dBm at
fixed ``T_B = 0.1 K`` and records ``Q_i``, ``Q_tot``, and the
converged ``n̄`` per point. The sweep warm-starts each successive
drive power from the previous converged ``n̄`` to keep iteration
counts low — crucial for the high-power end where the map stiffens
as ``Q_i`` drops and ``Q_tot → Q_i``.

This is a logarithmic-in-P_read characterization sweep at fixed
``T_B`` built on :func:`qpsim.services.nbar_loop.solve_nbar_loop`.
(The paper-track :mod:`fig7_paper` does NOT use the n̄ loop — it runs
fixed Table-III ``T*/Δ`` drive values while sweeping ``T_B``.)

Parameters (Fischer 2023 Table I / default ``qi_vs_pread``):

    Δ₀       = 180 μeV
    τ_0      = 438 ns
    ω_0      = Δ₀/9 = 20 μeV
    α_KI     = 0.5
    c_phot   = 1e-9 ns⁻¹
    Q_c      = 1e5
    T_B      = 0.1 K
    P_read   ∈ [-100, -60] dBm, 21 log-spaced points

Grid: 405 bins (ω_0/dE = 5 integer). Tolerance tier per NFP §6.4.1
is 1e-4 (nbar-loop tol × MB sub-gap quadrature).

Artifact policy
---------------

The historical CSV predates versioned solver certificates and is deliberately
quarantined by :func:`read_baseline`. A development artifact may be generated
with the command below, but it must not replace the canonical pin until an
independent energy-grid refinement study supports promotion.

Development generation::

    python -m validation.fischer_2023.figs_9_13_qi_vs_pread
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    phonon_collision_rates,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.constants import HBAR_UEV_NS, KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.nbar_loop import dbm_to_uev_per_ns, solve_nbar_loop

from validation.source_provenance import canonical_source_bytes

# ── Fischer 2023 Table I parameters ──────────────────────────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0  # 20 μeV — drive and probe
ALPHA_KI = 0.5
C_PHOT = 1e-9

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 405  # Q_i tolerance tier 1e-4; dE=4 μeV, ω₀/dE=5 (int)


ARTIFACT_SCHEMA_VERSION = 2
QP_CERTIFICATE_METRIC = "l1_gain_loss_backward_error_v1"
NBAR_CERTIFICATE_METRIC = "relative_fixed_point_residual_v1"
QP_BACKWARD_ERROR_LIMIT = 1.0e-6
QP_MAX_ABS_RESIDUAL_LIMIT = 1.0e-14
NBAR_FIXED_POINT_RESIDUAL_LIMIT = 1.0e-4

_SOURCE_FINGERPRINT_FILES: tuple[str, ...] = (
    "validation/fischer_2023/figs_9_13_qi_vs_pread.py",
    "validation/source_provenance.py",
    "qpsim/backends/t3_diffusion.py",
    "qpsim/collisions/_uniform_grid.py",
    "qpsim/collisions/_validation.py",
    "qpsim/collisions/phonon.py",
    "qpsim/collisions/sub_gap_photon.py",
    "qpsim/constants.py",
    "qpsim/grid/energy_grid.py",
    "qpsim/materials/database.py",
    "qpsim/observables/ac_conductivity.py",
    "qpsim/observables/quality_factor.py",
    "qpsim/phonon_models/state.py",
    "qpsim/physics/bcs_quadrature.py",
    "qpsim/physics/kernels.py",
    "qpsim/physics/spectral.py",
    "qpsim/services/steady_state.py",
    "qpsim/services/nbar_loop.py",
    "qpsim/solvers/newton_steady_state.py",
)

_CSV_COLUMNS: tuple[str, ...] = (
    "P_read_uev_per_ns",
    "P_read_dbm",
    "Q_i",
    "Q_tot",
    "n_bar",
    "iterations",
    "qp_backward_error",
    "qp_max_abs_residual",
    "nbar_fixed_point_residual",
)


class LegacyBaselineError(RuntimeError):
    """A pre-schema artifact that is deliberately quarantined."""


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
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


def source_fingerprint_sha256() -> str:
    """Hash every logical source defining this validation's solve contract."""
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for relative in _SOURCE_FINGERPRINT_FILES:
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(canonical_source_bytes(path))
        digest.update(b"\0")
    return digest.hexdigest()


def _config_json() -> str:
    return json.dumps(
        {
            "Delta_0": DELTA_0,
            "tau_0": TAU_0,
            "T_c": T_C,
            "omega_0": OMEGA_0,
            "alpha_ki": ALPHA_KI,
            "c_phot": C_PHOT,
            "E_min_factor": E_MIN_FACTOR,
            "E_max_factor": E_MAX_FACTOR,
            "num_bins": NUM_BINS,
            "T_bath": T_BATH,
            "Q_c": Q_C,
            "num_points": NUM_POINTS,
            "P_read_dbm_min": P_READ_DBM_MIN,
            "P_read_dbm_max": P_READ_DBM_MAX,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Write a UTF-8 text artifact beside its target, then replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as fp:
            yield fp
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def _qp_balance_certificate(
    f: np.ndarray,
    state: T3DiffusionState,
    K_s0: np.ndarray,
    K_r0: np.ndarray,
    n_bar: float,
) -> tuple[float, float]:
    """Return the versioned L1 gain/loss error and maximum absolute residual."""
    gain, loss_rate = phonon_collision_rates(
        f,
        state.spectral,
        K_s0,
        K_r0,
        state.T_bath,
    )
    photon_gain, photon_loss = sub_gap_photon_collision_rates(
        f,
        state.spectral,
        OMEGA_0,
        n_bar,
        C_PHOT,
    )
    gain = gain + photon_gain
    loss_rate = loss_rate + photon_loss
    active = state.spectral.active_mask
    loss_term = loss_rate[active] * f[active]
    residual = gain[active] - loss_term
    common = float(
        max(
            np.max(np.abs(gain[active]), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
            np.max(np.abs(residual), initial=0.0),
        )
    )
    if common == 0.0:
        backward_error = 0.0
    else:
        denominator = float(
            np.sum(np.abs(gain[active]) / common) + np.sum(np.abs(loss_term) / common)
        )
        numerator = float(np.sum(np.abs(residual) / common))
        backward_error = numerator / denominator if denominator > 0.0 else np.inf
    return backward_error, float(np.max(np.abs(residual), initial=0.0))


def _nbar_fixed_point_residual(
    P_read_uev_per_ns: float,
    n_bar: float,
    Q_tot: float,
) -> float:
    """Return the relative residual of the persisted outer-loop fixed point."""
    prefactor = 2.0 * HBAR_UEV_NS / (OMEGA_0**2 * Q_C)
    n_bar_raw = prefactor * Q_tot**2 * P_read_uev_per_ns
    return abs(n_bar_raw - n_bar) / max(abs(n_bar), 1e-300)


# ── Sweep parameters ─────────────────────────────────────────────────

T_BATH = 0.1  # K
Q_C = 1.0e5
NUM_POINTS = 21  # 21 log-spaced points across -100…-60 dBm
P_READ_DBM_MIN = -100.0
P_READ_DBM_MAX = -60.0


@dataclass(frozen=True)
class Figs913Result:
    P_read_uev_per_ns: np.ndarray  # shape (N,) drive power in code units
    P_read_dbm: np.ndarray  # shape (N,) drive power in dBm
    Q_i: np.ndarray  # shape (N,) converged Q_i
    Q_tot: np.ndarray  # shape (N,) loaded Q = (Q_i⁻¹ + Q_c⁻¹)⁻¹
    n_bar: np.ndarray  # shape (N,) converged photon occupancy
    iterations: np.ndarray  # shape (N,) nbar-loop iteration count
    qp_backward_error: np.ndarray  # shape (N,) L1 gain/loss certificate
    qp_max_abs_residual: np.ndarray  # shape (N,) max dimensional residual
    nbar_fixed_point_residual: np.ndarray  # shape (N,) outer-map certificate


def run() -> Figs913Result:
    material = _fischer_material()
    backend = T3DiffusionBackend()
    state = _build_state(material, T_BATH)
    K_s0 = build_scattering_kernel_base(
        state.spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )
    K_r0 = build_recombination_kernel_base(
        state.spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )

    dbm_values = np.linspace(P_READ_DBM_MIN, P_READ_DBM_MAX, NUM_POINTS)
    P_read_values = np.array([dbm_to_uev_per_ns(float(d)) for d in dbm_values])

    Q_i = np.zeros(NUM_POINTS)
    Q_tot = np.zeros(NUM_POINTS)
    n_bar = np.zeros(NUM_POINTS)
    iterations = np.zeros(NUM_POINTS, dtype=np.int64)
    qp_backward_error = np.zeros(NUM_POINTS)
    qp_max_abs_residual = np.zeros(NUM_POINTS)
    nbar_fixed_point_residual = np.zeros(NUM_POINTS)

    def solve_f(n_bar_val: float) -> np.ndarray:
        photon_params = {"omega_0": OMEGA_0, "n_bar": n_bar_val, "c_phot": C_PHOT}
        driven = backend.steady_state(
            state,
            use_thermal_phonons=True,
            photon_params=photon_params,
            newton_tol=1e-14,
            newton_backward_error_tol=QP_BACKWARD_ERROR_LIMIT,
            newton_max_iter=500,
        )
        return driven.f

    def compute_Q_i(f: np.ndarray) -> float:
        return compute_quality_factor(f, state.spectral, OMEGA_0, ALPHA_KI)

    n_bar_warm: float | None = None
    for idx, P_read in enumerate(P_read_values):
        loop = solve_nbar_loop(
            P_read_uev_per_ns=float(P_read),
            Q_c=Q_C,
            omega_0=OMEGA_0,
            solve_f=solve_f,
            compute_Q_i=compute_Q_i,
            n_bar_initial=n_bar_warm,
            tol=NBAR_FIXED_POINT_RESIDUAL_LIMIT,
            max_iter=50,
            under_relaxation=1.0,
        )
        if not loop.converged:
            raise RuntimeError(
                f"nbar loop failed at P_read={dbm_values[idx]:.2f} dBm "
                f"(n̄={loop.n_bar:.3e}, iterations={loop.iterations})"
            )
        Q_i[idx] = loop.Q_i
        Q_tot[idx] = loop.Q_tot
        n_bar[idx] = loop.n_bar
        iterations[idx] = loop.iterations
        qp_backward_error[idx], qp_max_abs_residual[idx] = _qp_balance_certificate(
            loop.f,
            state,
            K_s0,
            K_r0,
            loop.n_bar,
        )
        nbar_fixed_point_residual[idx] = _nbar_fixed_point_residual(
            float(P_read),
            loop.n_bar,
            loop.Q_tot,
        )
        if (
            not np.isfinite(qp_backward_error[idx])
            or qp_backward_error[idx] > QP_BACKWARD_ERROR_LIMIT
            or not np.isfinite(qp_max_abs_residual[idx])
            or qp_max_abs_residual[idx] >= QP_MAX_ABS_RESIDUAL_LIMIT
            or not np.isfinite(nbar_fixed_point_residual[idx])
            or nbar_fixed_point_residual[idx] >= NBAR_FIXED_POINT_RESIDUAL_LIMIT
        ):
            raise RuntimeError(
                "Figs. 9-13 returned-state certification failed at "
                f"P_read={dbm_values[idx]:.2f} dBm: "
                f"qp_backward_error={qp_backward_error[idx]:.3e}, "
                f"qp_max_abs_residual={qp_max_abs_residual[idx]:.3e}, "
                "nbar_fixed_point_residual="
                f"{nbar_fixed_point_residual[idx]:.3e}."
            )
        n_bar_warm = loop.n_bar

    return Figs913Result(
        P_read_uev_per_ns=P_read_values,
        P_read_dbm=dbm_values,
        Q_i=Q_i,
        Q_tot=Q_tot,
        n_bar=n_bar,
        iterations=iterations,
        qp_backward_error=qp_backward_error,
        qp_max_abs_residual=qp_max_abs_residual,
        nbar_fixed_point_residual=nbar_fixed_point_residual,
    )


def baseline_path() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "validation" / "baselines" / "ph0_constant" / "fischer_figs_9_13_qi_vs_pread.csv"


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "source_fingerprint_sha256",
        "config_json",
        "qp_certificate_metric",
        "nbar_certificate_metric",
        "qp_backward_error_limit",
        "qp_max_abs_residual_limit",
        "nbar_fixed_point_residual_limit",
        "max_qp_backward_error",
        "max_qp_abs_residual",
        "max_nbar_fixed_point_residual",
    }
)


def _validated_result(result: Figs913Result) -> dict[str, float]:
    names = (
        "P_read_uev_per_ns",
        "P_read_dbm",
        "Q_i",
        "Q_tot",
        "n_bar",
        "iterations",
        "qp_backward_error",
        "qp_max_abs_residual",
        "nbar_fixed_point_residual",
    )
    arrays = {name: np.asarray(getattr(result, name)) for name in names}
    expected_shape = (NUM_POINTS,)
    for name, values in arrays.items():
        if values.shape != expected_shape:
            raise RuntimeError(
                f"Figs. 9-13 artifact field {name!r} must have shape "
                f"{expected_shape}; got {values.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise RuntimeError(f"Figs. 9-13 artifact field {name!r} contains non-finite values.")

    expected_dbm = np.linspace(P_READ_DBM_MIN, P_READ_DBM_MAX, NUM_POINTS)
    expected_power = np.array([dbm_to_uev_per_ns(float(value)) for value in expected_dbm])
    if not np.array_equal(arrays["P_read_dbm"], expected_dbm):
        raise RuntimeError("Figs. 9-13 artifact has a stale or reordered dBm axis.")
    if not np.array_equal(arrays["P_read_uev_per_ns"], expected_power):
        raise RuntimeError("Figs. 9-13 artifact has a stale or reordered power axis.")
    if np.any(np.diff(arrays["P_read_dbm"]) <= 0.0) or np.any(
        np.diff(arrays["P_read_uev_per_ns"]) <= 0.0
    ):
        raise RuntimeError("Figs. 9-13 power axes must be unique and increasing.")
    if np.any(arrays["Q_i"] <= 0.0) or np.any(arrays["Q_tot"] <= 0.0):
        raise RuntimeError("Figs. 9-13 quality factors must be positive.")
    if np.any(arrays["n_bar"] < 0.0):
        raise RuntimeError("Figs. 9-13 photon occupations must be non-negative.")
    iteration_values = arrays["iterations"].astype(np.int64)
    if np.any(iteration_values < 1) or not np.array_equal(arrays["iterations"], iteration_values):
        raise RuntimeError("Figs. 9-13 iteration counts must be positive integers.")
    expected_q_tot = 1.0 / (1.0 / arrays["Q_i"] + 1.0 / Q_C)
    if not np.allclose(arrays["Q_tot"], expected_q_tot, rtol=5e-15, atol=0.0):
        raise RuntimeError("Figs. 9-13 Q_tot is inconsistent with Q_i and Q_c.")

    recomputed_nbar_residual = np.asarray(
        [
            _nbar_fixed_point_residual(float(power), float(n_bar), float(q_tot))
            for power, n_bar, q_tot in zip(
                arrays["P_read_uev_per_ns"],
                arrays["n_bar"],
                arrays["Q_tot"],
                strict=True,
            )
        ],
        dtype=float,
    )
    if not np.allclose(
        arrays["nbar_fixed_point_residual"],
        recomputed_nbar_residual,
        rtol=8.0 * np.finfo(float).eps,
        atol=0.0,
    ):
        difference = np.abs(arrays["nbar_fixed_point_residual"] - recomputed_nbar_residual)
        worst = int(np.argmax(difference))
        raise RuntimeError(
            "Figs. 9-13 stored nbar fixed-point certificate does not match "
            "the residual recomputed from its persisted power, n_bar, and Q_tot "
            f"at row {worst}: stored="
            f"{arrays['nbar_fixed_point_residual'][worst]:.17e}, "
            f"recomputed={recomputed_nbar_residual[worst]:.17e}."
        )

    for name in (
        "qp_backward_error",
        "qp_max_abs_residual",
        "nbar_fixed_point_residual",
    ):
        if np.any(arrays[name] < 0.0):
            raise RuntimeError(f"Certificate field {name!r} must be non-negative.")
    maxima = {
        "max_qp_backward_error": float(np.max(arrays["qp_backward_error"])),
        "max_qp_abs_residual": float(np.max(arrays["qp_max_abs_residual"])),
        "max_nbar_fixed_point_residual": float(np.max(arrays["nbar_fixed_point_residual"])),
    }
    if maxima["max_qp_backward_error"] > QP_BACKWARD_ERROR_LIMIT:
        raise RuntimeError("Figs. 9-13 QP backward-error certificate exceeds its gate.")
    if maxima["max_qp_abs_residual"] >= QP_MAX_ABS_RESIDUAL_LIMIT:
        raise RuntimeError("Figs. 9-13 QP absolute-residual certificate exceeds its gate.")
    if maxima["max_nbar_fixed_point_residual"] >= NBAR_FIXED_POINT_RESIDUAL_LIMIT:
        raise RuntimeError("Figs. 9-13 nbar fixed-point certificate exceeds its gate.")
    return maxima


def write_baseline(result: Figs913Result, path: Path | None = None) -> Path:
    """Atomically write a versioned, source-bound, certified artifact."""
    if path is None:
        path = baseline_path()
    maxima = _validated_result(result)
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp)
        metadata: tuple[tuple[str, str], ...] = (
            ("schema_version", str(ARTIFACT_SCHEMA_VERSION)),
            ("source_fingerprint_sha256", source_fingerprint_sha256()),
            ("config_json", _config_json()),
            ("qp_certificate_metric", QP_CERTIFICATE_METRIC),
            ("nbar_certificate_metric", NBAR_CERTIFICATE_METRIC),
            ("qp_backward_error_limit", f"{QP_BACKWARD_ERROR_LIMIT:.17e}"),
            ("qp_max_abs_residual_limit", f"{QP_MAX_ABS_RESIDUAL_LIMIT:.17e}"),
            (
                "nbar_fixed_point_residual_limit",
                f"{NBAR_FIXED_POINT_RESIDUAL_LIMIT:.17e}",
            ),
            ("max_qp_backward_error", f"{maxima['max_qp_backward_error']:.17e}"),
            ("max_qp_abs_residual", f"{maxima['max_qp_abs_residual']:.17e}"),
            (
                "max_nbar_fixed_point_residual",
                f"{maxima['max_nbar_fixed_point_residual']:.17e}",
            ),
        )
        for key, value in metadata:
            writer.writerow([f"# {key}={value}"])
        writer.writerow(_CSV_COLUMNS)
        for i in range(result.P_read_uev_per_ns.size):
            writer.writerow(
                [
                    f"{result.P_read_uev_per_ns[i]:.17e}",
                    f"{result.P_read_dbm[i]:.17e}",
                    f"{result.Q_i[i]:.17e}",
                    f"{result.Q_tot[i]:.17e}",
                    f"{result.n_bar[i]:.17e}",
                    str(int(result.iterations[i])),
                    f"{result.qp_backward_error[i]:.17e}",
                    f"{result.qp_max_abs_residual[i]:.17e}",
                    f"{result.nbar_fixed_point_residual[i]:.17e}",
                ]
            )
    return path


def read_baseline(path: Path | None = None) -> Figs913Result:
    """Read the current schema, rejecting legacy or uncertified artifacts."""
    if path is None:
        path = baseline_path()
    text = path.read_text(encoding="utf-8")
    if "# schema_version=" not in text:
        raise LegacyBaselineError(
            f"Legacy Figs. 9-13 baseline at {path} has no versioned schema or "
            "numerical certificates; it is quarantined pending refinement."
        )

    metadata: dict[str, str] = {}
    rows: list[list[str]] = []
    header_seen = False
    with path.open(encoding="utf-8", newline="") as fp:
        reader = csv.reader(fp)
        for line_number, line in enumerate(reader, start=1):
            if not line:
                raise RuntimeError(f"Blank row in Figs. 9-13 artifact at line {line_number}.")
            first = line[0]
            if first.startswith("#"):
                if len(line) != 1 or "=" not in first:
                    raise RuntimeError(
                        f"Malformed metadata in Figs. 9-13 artifact at line {line_number}."
                    )
                key, value = first[1:].strip().split("=", 1)
                if key in metadata:
                    raise RuntimeError(f"Duplicate metadata key {key!r} in Figs. 9-13 artifact.")
                metadata[key] = value
                continue
            if not header_seen:
                if tuple(line) != _CSV_COLUMNS:
                    raise RuntimeError("Figs. 9-13 artifact has a stale or malformed CSV header.")
                header_seen = True
                continue
            if len(line) != len(_CSV_COLUMNS):
                raise RuntimeError(
                    f"Figs. 9-13 data row {line_number} has {len(line)} columns; "
                    f"expected {len(_CSV_COLUMNS)}."
                )
            rows.append(line)

    if set(metadata) != _METADATA_KEYS:
        missing = sorted(_METADATA_KEYS - set(metadata))
        extra = sorted(set(metadata) - _METADATA_KEYS)
        raise RuntimeError(
            f"Figs. 9-13 metadata contract mismatch; missing={missing}, extra={extra}."
        )
    try:
        schema_version = int(metadata["schema_version"])
    except ValueError as err:
        raise RuntimeError("Figs. 9-13 schema_version must be an integer.") from err
    if schema_version != ARTIFACT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported Figs. 9-13 schema {schema_version}; expected {ARTIFACT_SCHEMA_VERSION}."
        )
    if metadata["source_fingerprint_sha256"] != source_fingerprint_sha256():
        raise RuntimeError("Figs. 9-13 source fingerprint is stale.")
    if metadata["config_json"] != _config_json():
        raise RuntimeError("Figs. 9-13 configuration metadata is stale.")
    if metadata["qp_certificate_metric"] != QP_CERTIFICATE_METRIC:
        raise RuntimeError("Figs. 9-13 QP certificate metric is stale.")
    if metadata["nbar_certificate_metric"] != NBAR_CERTIFICATE_METRIC:
        raise RuntimeError("Figs. 9-13 nbar certificate metric is stale.")

    def metadata_float(name: str) -> float:
        try:
            value = float(metadata[name])
        except ValueError as err:
            raise RuntimeError(f"Metadata {name!r} must be numeric.") from err
        if not np.isfinite(value):
            raise RuntimeError(f"Metadata {name!r} must be finite.")
        return value

    limits = {
        "qp_backward_error_limit": QP_BACKWARD_ERROR_LIMIT,
        "qp_max_abs_residual_limit": QP_MAX_ABS_RESIDUAL_LIMIT,
        "nbar_fixed_point_residual_limit": NBAR_FIXED_POINT_RESIDUAL_LIMIT,
    }
    for name, expected in limits.items():
        if metadata_float(name) != expected:
            raise RuntimeError(f"Figs. 9-13 certificate limit {name!r} is stale.")
    if not header_seen or len(rows) != NUM_POINTS:
        raise RuntimeError(
            f"Figs. 9-13 artifact must contain one header and {NUM_POINTS} rows; "
            f"got header={header_seen}, rows={len(rows)}."
        )

    try:
        first_five = np.array([[float(value) for value in row[:5]] for row in rows], dtype=float)
        iterations = np.array([int(row[5]) for row in rows], dtype=np.int64)
        certificates = np.array([[float(value) for value in row[6:]] for row in rows], dtype=float)
    except ValueError as err:
        raise RuntimeError("Figs. 9-13 artifact contains malformed numeric data.") from err
    result = Figs913Result(
        P_read_uev_per_ns=first_five[:, 0],
        P_read_dbm=first_five[:, 1],
        Q_i=first_five[:, 2],
        Q_tot=first_five[:, 3],
        n_bar=first_five[:, 4],
        iterations=iterations,
        qp_backward_error=certificates[:, 0],
        qp_max_abs_residual=certificates[:, 1],
        nbar_fixed_point_residual=certificates[:, 2],
    )
    maxima = _validated_result(result)
    for name, actual in maxima.items():
        stamped = metadata_float(name)
        if not np.isclose(stamped, actual, rtol=1e-14, atol=0.0):
            raise RuntimeError(f"Figs. 9-13 metadata maximum {name!r} does not match its data.")
    return result


def write_plot(result: Figs913Result, path: Path | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None:
        path = plot_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_q, ax_n) = plt.subplots(
        2,
        1,
        figsize=(8, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    # Use dBm on the x-axis directly (linear in dBm ≡ log in P_read).
    ax_q.semilogy(result.P_read_dbm, result.Q_i, "o-", lw=1.5, ms=5, label=r"$Q_i$")
    ax_q.semilogy(
        result.P_read_dbm,
        result.Q_tot,
        "s--",
        lw=1.0,
        ms=4,
        color="tab:orange",
        label=r"$Q_\mathrm{tot}$",
    )
    ax_q.axhline(Q_C, color="k", ls=":", lw=1.0, alpha=0.5, label=rf"$Q_c={Q_C:g}$")
    ax_q.set_ylabel(r"Quality factor", fontsize=12)
    ax_q.grid(True, which="both", ls=":", alpha=0.3)
    ax_q.legend(loc="best", fontsize=10)
    ax_q.set_title(
        "Fischer 2023 Figs 9-13 — $Q_i(P_\\mathrm{read})$ via n̄ loop\n"
        rf"$\Delta_0={DELTA_0:.0f}$ μeV, $\omega_0=\Delta_0/9$, "
        rf"$\alpha={ALPHA_KI}$, $Q_c={Q_C:g}$, $T_B={T_BATH}$ K",
        fontsize=10,
    )

    ax_n.semilogy(result.P_read_dbm, result.n_bar, "v-", lw=1.5, ms=5, color="tab:green")
    ax_n.set_xlabel(r"$P_\mathrm{read}$ [dBm]", fontsize=12)
    ax_n.set_ylabel(r"$\bar n$ (converged)", fontsize=12)
    ax_n.grid(True, which="both", ls=":", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_baseline() -> tuple[Path, Path]:
    print("Fischer 2023 Figs 9-13 — Q_i(P_read) via n̄ loop ...")
    print(
        f"  Δ₀={DELTA_0} μeV, τ_0={TAU_0} ns, ω_0={OMEGA_0:.2f} μeV, "
        f"α={ALPHA_KI}, Q_c={Q_C:g}, T_B={T_BATH} K"
    )
    print(f"  P_read: {NUM_POINTS} pts, {P_READ_DBM_MIN} → {P_READ_DBM_MAX} dBm (linear in dBm)")
    print(f"  Grid: NE={NUM_BINS}")
    result = run()
    print(f"  Q_i spans {result.Q_i.min():.2e} → {result.Q_i.max():.2e}")
    print(
        f"  n̄ spans  {result.n_bar.min():.2e} → {result.n_bar.max():.2e}, "
        f"iterations mean = {result.iterations.mean():.1f}"
    )
    csv_path = write_baseline(result)
    pdf_path = write_plot(result)
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    return csv_path, pdf_path


if __name__ == "__main__":
    generate_baseline()
