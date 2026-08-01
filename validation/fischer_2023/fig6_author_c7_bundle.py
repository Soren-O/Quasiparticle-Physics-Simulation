"""Build immutable re-solve evidence for the Figure 6 C7 nonlinear solver.

Formal C7 is a child of the completed C6 phonon-balance stage and the first
ladder stage that replaces the root-finding itself.  Every operator is held
at its accepted public configuration -- the C4 photon channel, the C5
QP-side scattering/pair kernels, and the C6 phonon-side balance with the
Kaplan ``S_+`` pair-breaking quadrature correction -- and only the author's
explicit-inverse Newton driver is replaced by qpsim's public
``coupled_newton_solve`` (fixed gap, analytic cross-Jacobian, author-intent
``max_iter=10`` and ``step_rtol=1e-7``).

The solve starts from the identical captured author continuation seed
(the A1 initial state bound by the accepted C0 evidence), ordinally
projected to the native 1640-cell/3600-bin grids exactly as C3 projected
the root.  The bundle retains the seed, the returned root, the solver's
bit-reproduced residual assembly at both the root and the frozen C6 state,
per-channel decompositions, and a minimal-``max_iter`` probe that measures
the accepted iteration count through the public API alone.

The public solver returns only the final state; its internal Newton path is
qpsim semantics, bound by the hash-closed solver source rather than by
per-iteration captures.  C7 records one authenticated point.  It claims no
300-point curve, no paper parity, and no author-equivalence of the changed
iteration policy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.gap_suppression import gap_from_distribution_direct
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.coupled_newton import coupled_newton_solve

from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as DEFAULT_C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as DEFAULT_C2_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as DEFAULT_C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_SCORE as DEFAULT_C3_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import load_c3_raw_bundle
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as DEFAULT_C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_SCORE as DEFAULT_C4_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as DEFAULT_C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_SCORE as DEFAULT_C5_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import load_c5_raw_bundle
from validation.fischer_2023.fig6_author_c6_score import (
    DEFAULT_RECEIPT as DEFAULT_C6_RECEIPT,
)
from validation.fischer_2023.fig6_author_c6_score import (
    DEFAULT_SCORE as DEFAULT_C6_SCORE,
)
from validation.fischer_2023.fig6_author_c6_score import (
    RAW_SCHEMA as C6_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    RECEIPT_SCHEMA as C6_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    SCHEMA as C6_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c6_score import (
    build_c6_score,
    load_c6_raw_bundle,
    load_c6_receipt,
    load_c6_score,
)
from validation.fischer_2023.fig6_author_c6_score import (
    canonical_score_bytes as canonical_c6_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c7-nonlinear-solver-bundle.v1"
STAGE_ID = "C7"
PARENT_STAGE_ID = "C6"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "nonlinear_solver"

SECONDS_PER_NS = 1.0e-9
T_C_K = 1.184309192877208
TAU_0_NS = 438.0
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
T_BATH_K = 0.2
OMEGA_0_UEV = 20.0
C_PHOT_NS_INV = 1.0e-9
STEP_RTOL = 1.0e-7
MAX_ITER = 10
N_QP = 1640
N_OMEGA = 3600
N_AUTHOR = 1620
AUTHOR_OMEGA_STOP = 1619
DEFAULT_C0_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c0-author-equivalent-score.json"
)
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

ARRAY_NAMES = (
    "author_seed_state",
    "frozen_qp_residual_delta_s_inv",
    "frozen_phonon_residual_delta_support_s_inv",
    "parent_E_centers_ueV",
    "parent_active_mask",
    "parent_cell_density",
    "parent_cell_weights_ueV",
    "parent_dE_ueV",
    "parent_f",
    "parent_legacy_phonon_support_mask",
    "parent_phonon_residual_s_inv",
    "parent_phonon_to_native_omega_index",
    "parent_projected_n_phonon",
    "parent_qp_residual_s_inv",
    "parent_thermal_f",
    "qpsim_frozen_R_f_ns_inv",
    "qpsim_frozen_R_ph_ns_inv",
    "qpsim_root_R_f_ns_inv",
    "qpsim_root_R_ph_ns_inv",
    "qpsim_root_a_ph_ns_inv",
    "qpsim_root_b_ph_ns_inv",
    "qpsim_root_f",
    "qpsim_root_minus_frozen_f",
    "qpsim_root_minus_frozen_n_ph",
    "qpsim_root_n_ph",
    "qpsim_root_pair_gain_s_inv",
    "qpsim_root_pair_loss_s_inv",
    "qpsim_root_pair_net_s_inv",
    "qpsim_root_phonon_escape_net_ns_inv",
    "qpsim_root_phonon_pair_a_ns_inv",
    "qpsim_root_phonon_pair_b_ns_inv",
    "qpsim_root_phonon_scattering_a_ns_inv",
    "qpsim_root_phonon_scattering_b_ns_inv",
    "qpsim_root_photon_gain_s_inv",
    "qpsim_root_photon_loss_s_inv",
    "qpsim_root_photon_net_s_inv",
    "qpsim_root_qp_gain_ns_inv",
    "qpsim_root_qp_loss_rate_ns_inv",
    "qpsim_root_scattering_gain_s_inv",
    "qpsim_root_scattering_loss_s_inv",
    "qpsim_root_scattering_net_s_inv",
    "qpsim_thermal_n_ph",
    "seed_f",
    "seed_n_ph",
)

_REPLAY_VALIDATION_MODULES = (
    REPOSITORY_ROOT / "validation" / "__init__.py",
    REPOSITORY_ROOT / "validation" / "author_source.py",
    REPOSITORY_ROOT / "validation" / "reproduction_ladder.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "__init__.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_solve.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_adapter.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_frozen_state.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c0_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c0_summary.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c1_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_parameters.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c3_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c3_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c4_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c4_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c5_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c5_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c6_bundle.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c6_score.py",
    REPOSITORY_ROOT / "validation" / "reference_models" / "__init__.py",
    (
        REPOSITORY_ROOT
        / "validation"
        / "reference_models"
        / "fischer_2023"
        / "__init__.py"
    ),
    (
        REPOSITORY_ROOT
        / "validation"
        / "reference_models"
        / "fischer_2023"
        / "fig6_author_c0.py"
    ),
)
_SOURCE_MANIFEST_AT_IMPORT = source_manifest(
    Path(__file__),
    extra_validation_modules=_REPLAY_VALIDATION_MODULES,
)
_SOURCE_BYTES_AT_IMPORT = {
    relative: canonical_source_bytes(REPOSITORY_ROOT / relative)
    for relative in _SOURCE_MANIFEST_AT_IMPORT
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}
if _SOURCE_HASHES_AT_IMPORT != _SOURCE_MANIFEST_AT_IMPORT:
    raise RuntimeError("C7 source tree changed while its import snapshot was captured.")


class C7BundleError(ValueError):
    """The C7 parent, seed, solver input, or transport is invalid."""


def _assert_source_snapshots() -> None:
    if (
        source_manifest(
            Path(__file__),
            extra_validation_modules=_REPLAY_VALIDATION_MODULES,
        )
        != _SOURCE_HASHES_AT_IMPORT
    ):
        raise C7BundleError("C7 numerical source closure changed during execution.")
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C7BundleError(f"C7 numerical source changed during execution: {relative}.")


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    try:
        before = path.lstat()
    except OSError as exc:
        raise C7BundleError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C7BundleError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C7BundleError(f"{label} must stay inside the repository.") from exc
    if resolved != path.absolute() or resolved.is_symlink() or not resolved.is_file():
        raise C7BundleError(f"{label} is missing, unsafe, or a symlink.")
    try:
        with resolved.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = resolved.lstat()
    except OSError as exc:
        raise C7BundleError(f"{label} changed or became unreadable.") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )

    if not (
        identity(before) == identity(opened_before) == identity(opened_after) == identity(after)
    ):
        raise C7BundleError(f"{label} changed while it was being read.")
    return resolved, content


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    checked_path, checked = _repository_file_snapshot(path, label)
    if checked_path != path or checked != expected:
        raise C7BundleError(f"{label} changed during C7 construction.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C7BundleError(f"{label} must be an object.")
    return value


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _write_new_file(path: Path, content: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(_npy_bytes(array)).hexdigest(),
        "shape": list(array.shape),
    }


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    if not np.isfinite(result):
        raise C7BundleError("C7 scalar metadata must be finite.")
    return {"hex": result.hex(), "value": result}


def _runtime_record() -> dict[str, object]:
    thread_environment = {
        name: os.environ.get(name)
        for name in sorted(_THREAD_ENVIRONMENT)
    }
    if thread_environment != _THREAD_ENVIRONMENT:
        raise C7BundleError(
            "C7 BLAS evidence requires MKL_NUM_THREADS=1, "
            "OMP_NUM_THREADS=1, and OPENBLAS_NUM_THREADS=1."
        )
    config = getattr(np.__config__, "CONFIG", {})
    build_dependencies = (
        config.get("Build Dependencies", {})
        if isinstance(config, dict)
        else {}
    )
    blas = (
        build_dependencies.get("blas", {})
        if isinstance(build_dependencies, dict)
        else {}
    )
    if not isinstance(blas, dict):
        blas = {}
    return {
        "byteorder": sys.byteorder,
        "implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "numpy_blas": {
            "found": blas.get("found"),
            "name": blas.get("name"),
            "openblas_configuration": blas.get("openblas configuration"),
            "version": blas.get("version"),
        },
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "thread_environment": thread_environment,
    }


def _assert_context_matches_parent(
    ctx: SpectralContext,
    c3_arrays: dict[str, np.ndarray],
) -> None:
    comparisons = {
        "native_E_centers_ueV": ctx.E,
        "native_dE_ueV": ctx.dE,
        "native_active_mask": ctx.active_mask,
        "native_cell_density_full": ctx.cell_density,
        "native_cell_weights_full": ctx.cell_weights,
        "native_K_minus_full": ctx.K_minus,
        "native_K_plus_full": ctx.K_plus,
    }
    for name, live in comparisons.items():
        if not np.array_equal(np.asarray(live), np.asarray(c3_arrays[name])):
            raise C7BundleError(
                f"Live C7 SpectralContext does not reproduce accepted C3 array {name!r}."
            )


def _solver_residual(
    f_state: np.ndarray,
    n_state: np.ndarray,
    *,
    ctx: SpectralContext,
    K_s0: np.ndarray,
    K_r0: np.ndarray,
    K_s_ph: np.ndarray,
    K_r_ph: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_th: np.ndarray,
    n_bar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce ``coupled_newton_solve``'s residual assembly bit-for-bit.

    Returns ``(gain, loss_rate, R_f, a_ph, b_ph, R_ph)`` in ns^-1 with the
    exact public-call and accumulation order of the solver's internal
    ``residual`` closure: one combined e-ph ``phonon_collision_rates`` call
    with dynamic occupation overrides, then the photon channel added, the
    inactive rows zeroed, and the three-term phonon balance.
    """

    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_state,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
    )
    gain, loss_rate = phonon_collision_rates(
        f_state,
        ctx,
        K_s0,
        K_r0,
        T_BATH_K,
        N_p_override=N_p,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    gain_ph, loss_ph = sub_gap_photon_collision_rates(
        f_state,
        ctx,
        OMEGA_0_UEV,
        n_bar,
        C_PHOT_NS_INV,
    )
    gain = gain + gain_ph
    loss_rate = loss_rate + loss_ph
    unsupported = ~ctx.active_mask
    gain[unsupported] = 0.0
    loss_rate[unsupported] = 0.0
    R_f = gain - loss_rate * f_state

    a_ph, b_ph = compute_phonon_source_sink(
        f_state,
        ctx,
        K_s0,
        K_r0,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        N_OMEGA,
        K_s0_phonon_side=K_s_ph,
        K_r0_phonon_side=K_r_ph,
    )
    driven_phonon = b_ph * n_state
    bath_phonon = (n_th - n_state) * (1.0 / TAU_L_NS)
    R_ph = a_ph + driven_phonon + bath_phonon
    return gain, loss_rate, R_f, a_ph, b_ph, R_ph


def build_c7_bundle(
    c6_bundle_dir: Path,
    *,
    c5_bundle_dir: Path,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c0_bundle_dir: Path,
    c6_score_path: Path = DEFAULT_C6_SCORE,
    c6_receipt_path: Path = DEFAULT_C6_RECEIPT,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
    c0_score_path: Path = DEFAULT_C0_SCORE,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run and bind the formal one-point C6-to-C7 nonlinear-solver re-solve."""

    _assert_source_snapshots()
    runtime = _runtime_record()
    anchors = {
        label: _repository_file_snapshot(path, label)
        for label, path in {
            "C6 score": c6_score_path,
            "C6 receipt": c6_receipt_path,
            "C5 score": c5_score_path,
            "C5 receipt": c5_receipt_path,
            "C4 score": c4_score_path,
            "C4 receipt": c4_receipt_path,
            "C3 score": c3_score_path,
            "C3 receipt": c3_receipt_path,
            "C2 score": c2_score_path,
            "C2 receipt": c2_receipt_path,
            "C0 score": c0_score_path,
        }.items()
    }
    c6_score_path, c6_score_bytes = anchors["C6 score"]
    c6_receipt_path, c6_receipt_bytes = anchors["C6 receipt"]
    c0_score_path, c0_score_bytes = anchors["C0 score"]

    accepted_c6 = load_c6_score(c6_score_path, receipt_path=c6_receipt_path)
    accepted_c6_receipt = load_c6_receipt(c6_receipt_path)
    rebuilt_c6 = build_c6_score(
        c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c6_score_bytes(rebuilt_c6) != c6_score_bytes:
        raise C7BundleError(
            "C7 refuses a C6 score/receipt pair that does not independently "
            "reproduce from the selected C6/C5/C4/C3/C2 raw evidence."
        )
    _c6_metadata, c6_arrays, c6_manifest_sha = load_c6_raw_bundle(c6_bundle_dir)
    c6_raw_binding = _mapping(accepted_c6.get("raw_bundle"), "accepted C6 raw binding")
    if c6_raw_binding != {
        "manifest_sha256": c6_manifest_sha,
        "schema": C6_RAW_SCHEMA,
    }:
        raise C7BundleError("Selected C6 raw bundle is not the accepted parent.")
    if _mapping(accepted_c6_receipt.get("raw_bundle"), "C6 receipt raw") != c6_raw_binding:
        raise C7BundleError("Accepted C6 receipt does not bind the selected raw bundle.")
    c6_parent_bindings = _mapping(
        accepted_c6.get("parent_bindings"),
        "accepted C6 parent bindings",
    )
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_bundle_dir)
    _c5_metadata, c5_arrays, c5_manifest_sha = load_c5_raw_bundle(c5_bundle_dir)
    if (
        c6_parent_bindings.get("c3_raw_manifest_sha256") != c3_manifest_sha
        or c6_parent_bindings.get("c5_raw_manifest_sha256") != c5_manifest_sha
    ):
        raise C7BundleError("Selected C3/C5 raw bundles are not C6's accepted ancestors.")

    # Seed authentication: the committed C0 score binds the external C0 raw
    # bundle manifest and the A1 initial-state descriptor.
    c0_score = json.loads(c0_score_bytes.decode("utf-8"))
    c0_raw_binding = _mapping(c0_score.get("raw_bundle"), "C0 raw binding")
    c0_manifest_raw = (Path(c0_bundle_dir) / "manifest.json").read_bytes()
    c0_manifest_sha = hashlib.sha256(c0_manifest_raw).hexdigest()
    if c0_raw_binding.get("manifest_sha256") != c0_manifest_sha:
        raise C7BundleError("Selected C0 raw bundle is not the accepted A1-equivalent evidence.")
    c0_descriptors = _mapping(c0_score.get("array_descriptors"), "C0 descriptors")
    seed_descriptor = _mapping(
        c0_descriptors.get("initial_state"),
        "C0 initial-state descriptor",
    )
    seed_bytes = (Path(c0_bundle_dir) / "initial_state.npy").read_bytes()
    if hashlib.sha256(seed_bytes).hexdigest() != seed_descriptor.get("npy_sha256"):
        raise C7BundleError("C0 initial-state bytes do not match the accepted descriptor.")
    author_seed = np.asarray(
        np.lib.format.read_array(io.BytesIO(seed_bytes), allow_pickle=False)
    )
    if author_seed.shape != (2 * N_AUTHOR - 1,) or author_seed.dtype.str != "<f8":
        raise C7BundleError("C0 initial state has an unexpected shape or dtype.")

    parameter_record = _mapping(c3_metadata.get("parameters"), "C3 parameters")
    parameter_values = _mapping(parameter_record.get("values"), "C3 parameter values")
    inherited_n_bar = float(parameter_values.get("n_bar"))  # type: ignore[arg-type]
    inherited_gap_eV = float(parameter_values.get("gap_eV"))  # type: ignore[arg-type]
    inherited_delta0_eV = float(parameter_values.get("delta0_eV"))  # type: ignore[arg-type]
    inherited_h_eV = float(parameter_values.get("h_eV"))  # type: ignore[arg-type]
    inherited_boltzmann = float(
        parameter_values.get("boltzmann_constant_J_per_K")  # type: ignore[arg-type]
    )
    inherited_electron_charge = float(
        parameter_values.get("electron_charge_C")  # type: ignore[arg-type]
    )
    if (
        float(parameter_values.get("T_c_K")) != T_C_K  # type: ignore[arg-type]
        or float(parameter_values.get("tau_0_s")) != TAU_0_NS * SECONDS_PER_NS  # type: ignore[arg-type]
        or float(parameter_values.get("tau_0_pb_s")) != TAU_PB_NS * SECONDS_PER_NS  # type: ignore[arg-type]
        or float(parameter_values.get("tau_l_s")) != TAU_L_NS * SECONDS_PER_NS  # type: ignore[arg-type]
        or float(parameter_values.get("temperature_K")) != T_BATH_K  # type: ignore[arg-type]
        or inherited_boltzmann / inherited_electron_charge * 1.0e6 != KB_UEV_PER_K
    ):
        raise C7BundleError("C7 parent parameters do not match the declared solver inputs.")
    native_parameters = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "C3 native grid parameters",
    )
    gap_ueV = float(native_parameters.get("gap_ueV"))  # type: ignore[arg-type]

    E = np.asarray(c3_arrays["native_E_centers_ueV"]).copy()
    dE = np.asarray(c3_arrays["native_dE_ueV"]).copy()
    frozen_f = np.asarray(c3_arrays["projected_f"]).copy()
    frozen_n = np.asarray(c3_arrays["projected_n_phonon"]).copy()
    active = np.asarray(c3_arrays["native_active_mask"]).copy()
    weights = np.asarray(c3_arrays["native_cell_weights_full"]).copy()
    cell_density = np.asarray(c3_arrays["native_cell_density_full"]).copy()
    legacy_support = np.asarray(c3_arrays["legacy_phonon_support_mask"]).copy()
    omega_index_map = np.asarray(
        c3_arrays["parent_phonon_to_native_omega_index"]
    ).copy()
    thermal_f = np.asarray(c3_arrays["projected_thermal_f"]).copy()
    parent_qp_residual = np.asarray(c6_arrays["c6_qp_residual_s_inv"]).copy()
    parent_phonon_residual = np.asarray(
        c6_arrays["c6spe_phonon_residual_s_inv"]
    ).copy()

    ctx = SpectralContext(E, dE, gap_ueV)
    _assert_context_matches_parent(ctx, c3_arrays)
    omega, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(E)
    for name, live in (
        ("qpsim_omega_ueV", omega),
        ("qpsim_omega_idx_diff", omega_idx_diff),
        ("qpsim_omega_idx_sum", omega_idx_sum),
        ("qpsim_diff_sign", diff_sign),
    ):
        if not np.array_equal(np.asarray(live), np.asarray(c6_arrays[name])):
            raise C7BundleError(
                f"Public C7 frequency map array {name!r} does not reproduce "
                "the accepted C6 frozen map."
            )
    K_s0 = build_scattering_kernel_base(ctx, TAU_0_NS, T_C_K)
    K_r0 = build_recombination_kernel_base(ctx, TAU_0_NS, T_C_K)
    K_s_ph = build_scattering_kernel_phonon_side(ctx, TAU_PB_NS)
    K_r_ph = build_recombination_kernel_phonon_side(ctx, TAU_PB_NS)
    if not np.array_equal(
        K_s0,
        np.asarray(c5_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"]),
    ) or not np.array_equal(
        K_r0,
        np.asarray(c5_arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"]),
    ):
        raise C7BundleError("C7 QP-side kernels do not reproduce the accepted C5 kernels.")
    if not np.array_equal(
        K_s_ph,
        np.asarray(c6_arrays["qpsim_phonon_scattering_kernel_ns_inv_ueV_inv"]),
    ) or not np.array_equal(
        K_r_ph,
        np.asarray(c6_arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"]),
    ):
        raise C7BundleError("C7 phonon-side kernels do not reproduce the accepted C6 kernels.")
    n_th = thermal_phonon_occupation(omega, T_BATH_K)
    if not np.array_equal(n_th, np.asarray(c6_arrays["qpsim_thermal_n_ph"])):
        raise C7BundleError("C7 thermal occupation does not reproduce the accepted C6 array.")

    seed_f = np.zeros(N_QP, dtype=np.float64)
    seed_f[N_QP - N_AUTHOR :] = author_seed[:N_AUTHOR]
    seed_n = np.zeros(N_OMEGA, dtype=np.float64)
    seed_n[1 : AUTHOR_OMEGA_STOP + 1] = author_seed[N_AUTHOR:]

    solver_common: dict[str, Any] = {
        "omega_bins": omega,
        "omega_idx_diff": omega_idx_diff,
        "omega_idx_sum": omega_idx_sum,
        "diff_sign": diff_sign,
        "K_s0": K_s0,
        "K_r0": K_r0,
        "K_s0_phonon_side": K_s_ph,
        "K_r0_phonon_side": K_r_ph,
        "T_bath": T_BATH_K,
        "tau_l": TAU_L_NS,
        "photon_params": {
            "omega_0": OMEGA_0_UEV,
            "n_bar": inherited_n_bar,
            "c_phot": C_PHOT_NS_INV,
        },
        "step_rtol": STEP_RTOL,
        "analytic_cross": True,
    }
    probe: list[dict[str, object]] = []
    accepted_cap: int | None = None
    root_f: np.ndarray | None = None
    root_n: np.ndarray | None = None
    for cap in range(1, MAX_ITER + 1):
        try:
            candidate_f, candidate_n = coupled_newton_solve(
                ctx,
                seed_f,
                seed_n,
                max_iter=cap,
                **solver_common,
            )
        except RuntimeError as exc:
            probe.append(
                {
                    "converged": False,
                    "exception": type(exc).__name__,
                    "max_iter": cap,
                }
            )
            continue
        probe.append({"converged": True, "exception": None, "max_iter": cap})
        accepted_cap = cap
        root_f, root_n = candidate_f, candidate_n
        break
    if accepted_cap is None or root_f is None or root_n is None:
        raise C7BundleError(
            f"qpsim coupled Newton did not converge within {MAX_ITER} iterations."
        )
    full_f, full_n = coupled_newton_solve(
        ctx,
        seed_f,
        seed_n,
        max_iter=MAX_ITER,
        **solver_common,
    )
    if not np.array_equal(full_f, root_f) or not np.array_equal(full_n, root_n):
        raise C7BundleError(
            "The declared-cap solve does not reproduce the minimal-cap root "
            "bit-for-bit; the iteration-count probe is invalid."
        )

    residual_kwargs: dict[str, Any] = {
        "ctx": ctx,
        "K_s0": K_s0,
        "K_r0": K_r0,
        "K_s_ph": K_s_ph,
        "K_r_ph": K_r_ph,
        "omega_idx_diff": omega_idx_diff,
        "omega_idx_sum": omega_idx_sum,
        "diff_sign": diff_sign,
        "n_th": n_th,
        "n_bar": inherited_n_bar,
    }
    (
        root_gain,
        root_loss_rate,
        root_R_f,
        root_a_ph,
        root_b_ph,
        root_R_ph,
    ) = _solver_residual(root_f, root_n, **residual_kwargs)
    (
        _frozen_gain,
        _frozen_loss_rate,
        frozen_R_f,
        _frozen_a_ph,
        _frozen_b_ph,
        frozen_R_ph,
    ) = _solver_residual(frozen_f, frozen_n, **residual_kwargs)
    frozen_qp_delta = frozen_R_f / SECONDS_PER_NS - parent_qp_residual
    frozen_ph_delta = (
        frozen_R_ph[omega_index_map] / SECONDS_PER_NS - parent_phonon_residual
    )

    qp_scale = np.abs(root_gain) + np.abs(root_loss_rate * root_f)
    root_driven = root_b_ph * root_n
    root_bath = (n_th - root_n) * (1.0 / TAU_L_NS)
    ph_scale = np.abs(root_a_ph) + np.abs(root_driven) + np.abs(root_bath)
    qp_balance_ratio = float(np.sum(np.abs(root_R_f))) / float(np.sum(qp_scale))
    ph_balance_ratio = float(np.sum(np.abs(root_R_ph))) / float(np.sum(ph_scale))
    if max(qp_balance_ratio, ph_balance_ratio) >= STEP_RTOL:
        raise C7BundleError(
            "The returned C7 root does not satisfy the declared balance "
            "certificates at re-evaluation."
        )

    # Per-channel decompositions at the root through the same accepted
    # public operators used by C4/C5/C6.
    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        root_n,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
    )
    photon_gain_ns, photon_loss_rate_ns = sub_gap_photon_collision_rates(
        root_f,
        ctx,
        OMEGA_0_UEV,
        inherited_n_bar,
        C_PHOT_NS_INV,
    )
    scattering_gain_ns, scattering_loss_rate_ns = phonon_collision_rates(
        root_f,
        ctx,
        K_s0,
        None,
        T_BATH_K,
        enable_scattering=True,
        enable_recombination=False,
        N_p_override=N_p,
    )
    pair_gain_ns, pair_loss_rate_ns = phonon_collision_rates(
        root_f,
        ctx,
        None,
        K_r0,
        T_BATH_K,
        enable_scattering=False,
        enable_recombination=True,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    photon_gain_s = photon_gain_ns / SECONDS_PER_NS
    photon_loss_s = (photon_loss_rate_ns * root_f) / SECONDS_PER_NS
    photon_net_s = (photon_gain_ns - photon_loss_rate_ns * root_f) / SECONDS_PER_NS
    scattering_gain_s = scattering_gain_ns / SECONDS_PER_NS
    scattering_loss_s = (scattering_loss_rate_ns * root_f) / SECONDS_PER_NS
    scattering_net_s = (
        scattering_gain_ns - scattering_loss_rate_ns * root_f
    ) / SECONDS_PER_NS
    pair_gain_s = pair_gain_ns / SECONDS_PER_NS
    pair_loss_s = (pair_loss_rate_ns * root_f) / SECONDS_PER_NS
    pair_net_s = (pair_gain_ns - pair_loss_rate_ns * root_f) / SECONDS_PER_NS
    phonon_scattering_a, phonon_scattering_b = compute_phonon_source_sink(
        root_f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        N_OMEGA,
        enable_scattering=True,
        enable_recombination=False,
        K_s0_phonon_side=K_s_ph,
    )
    phonon_pair_a, phonon_pair_b = compute_phonon_source_sink(
        root_f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        N_OMEGA,
        enable_scattering=False,
        enable_recombination=True,
        K_r0_phonon_side=K_r_ph,
    )
    escape_net_ns = (n_th - root_n) / TAU_L_NS

    scattering_number = float(np.sum(weights * scattering_net_s))
    scattering_turnover = float(
        np.sum(weights * (np.abs(scattering_gain_s) + np.abs(scattering_loss_s)))
    )
    pair_number = float(np.sum(weights * pair_net_s))
    pair_turnover = float(
        np.sum(weights * (np.abs(pair_gain_s) + np.abs(pair_loss_s)))
    )
    photon_number = float(np.sum(weights * photon_net_s))
    photon_turnover = float(
        np.sum(weights * (np.abs(photon_gain_s) + np.abs(photon_loss_s)))
    )

    E_left_eV = inherited_gap_eV + inherited_h_eV * np.arange(
        N_AUTHOR, dtype=np.float64
    )
    carrier_eV = E_left_eV + 0.5 * inherited_h_eV
    active_slice = slice(N_QP - N_AUTHOR, N_QP)

    def gaps_and_ordinate(f_active: np.ndarray, samples: str) -> tuple[float, float, float]:
        driven = gap_from_distribution_direct(
            f_active,
            carrier_eV,
            gap=inherited_gap_eV,
            delta0=inherited_delta0_eV,
            samples=samples,
        )
        thermal = gap_from_distribution_direct(
            thermal_f[active_slice],
            carrier_eV,
            gap=inherited_gap_eV,
            delta0=inherited_delta0_eV,
            samples=samples,
        )
        return (
            driven,
            thermal,
            (driven - thermal) / (inherited_delta0_eV - thermal),
        )

    driven_gap, thermal_gap, ordinate_authors = gaps_and_ordinate(
        root_f[active_slice], "authors"
    )
    (
        driven_gap_centers,
        thermal_gap_centers,
        ordinate_centers,
    ) = gaps_and_ordinate(root_f[active_slice], "centers")
    (
        _frozen_driven_gap,
        _frozen_thermal_gap,
        frozen_ordinate_authors,
    ) = gaps_and_ordinate(frozen_f[active_slice], "authors")
    author_control_ordinate = 0.12090908988993258
    if frozen_ordinate_authors != author_control_ordinate:
        raise C7BundleError(
            "The frozen author root does not reproduce the author control "
            "ordinate under the accepted observable convention."
        )

    arrays: dict[str, np.ndarray] = {
        "author_seed_state": author_seed,
        "frozen_qp_residual_delta_s_inv": frozen_qp_delta,
        "frozen_phonon_residual_delta_support_s_inv": frozen_ph_delta,
        "parent_E_centers_ueV": E,
        "parent_active_mask": active,
        "parent_cell_density": cell_density,
        "parent_cell_weights_ueV": weights,
        "parent_dE_ueV": dE,
        "parent_f": frozen_f,
        "parent_legacy_phonon_support_mask": legacy_support,
        "parent_phonon_residual_s_inv": parent_phonon_residual,
        "parent_phonon_to_native_omega_index": omega_index_map,
        "parent_projected_n_phonon": frozen_n,
        "parent_qp_residual_s_inv": parent_qp_residual,
        "parent_thermal_f": thermal_f,
        "qpsim_frozen_R_f_ns_inv": frozen_R_f,
        "qpsim_frozen_R_ph_ns_inv": frozen_R_ph,
        "qpsim_root_R_f_ns_inv": root_R_f,
        "qpsim_root_R_ph_ns_inv": root_R_ph,
        "qpsim_root_a_ph_ns_inv": root_a_ph,
        "qpsim_root_b_ph_ns_inv": root_b_ph,
        "qpsim_root_f": root_f,
        "qpsim_root_minus_frozen_f": root_f - frozen_f,
        "qpsim_root_minus_frozen_n_ph": root_n - frozen_n,
        "qpsim_root_n_ph": root_n,
        "qpsim_root_pair_gain_s_inv": pair_gain_s,
        "qpsim_root_pair_loss_s_inv": pair_loss_s,
        "qpsim_root_pair_net_s_inv": pair_net_s,
        "qpsim_root_phonon_escape_net_ns_inv": escape_net_ns,
        "qpsim_root_phonon_pair_a_ns_inv": phonon_pair_a,
        "qpsim_root_phonon_pair_b_ns_inv": phonon_pair_b,
        "qpsim_root_phonon_scattering_a_ns_inv": phonon_scattering_a,
        "qpsim_root_phonon_scattering_b_ns_inv": phonon_scattering_b,
        "qpsim_root_photon_gain_s_inv": photon_gain_s,
        "qpsim_root_photon_loss_s_inv": photon_loss_s,
        "qpsim_root_photon_net_s_inv": photon_net_s,
        "qpsim_root_qp_gain_ns_inv": root_gain,
        "qpsim_root_qp_loss_rate_ns_inv": root_loss_rate,
        "qpsim_root_scattering_gain_s_inv": scattering_gain_s,
        "qpsim_root_scattering_loss_s_inv": scattering_loss_s,
        "qpsim_root_scattering_net_s_inv": scattering_net_s,
        "qpsim_thermal_n_ph": n_th,
        "seed_f": seed_f,
        "seed_n_ph": seed_n,
    }
    if set(arrays) != set(ARRAY_NAMES) or len(arrays) != len(ARRAY_NAMES):
        raise C7BundleError("C7 raw array-name closure is inconsistent.")
    for name, value in tuple(arrays.items()):
        if value.dtype.kind == "f":
            normalized = np.asarray(value).copy()
            normalized[normalized == 0.0] = 0.0
            arrays[name] = normalized

    metadata: dict[str, Any] = {
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "comparison_contract": {
            "candidate": (
                "qpsim's public coupled_newton_solve driven from the identical "
                "captured author continuation seed under the accepted "
                "C4/C5/C6 operator configuration"
            ),
            "frozen_state_binding": (
                "the solver's residual assembly evaluated at the frozen C6 "
                "state is retained together with its deltas against the "
                "accepted C6 hybrid residuals, binding the one-component "
                "claim: only the root-finding changed"
            ),
            "parent": (
                "accepted C6 frozen state, hybrid residuals, kernels, maps, "
                "and thermal occupation, reproduced bit-for-bit before the "
                "solve"
            ),
            "public_arithmetic": (
                "the Newton linear algebra uses LAPACK and a backtracking "
                "line search; the returned root is runtime-authenticated and "
                "certified host-independently through residual and balance "
                "bounds, not by bit-replaying the iteration path"
            ),
        },
        "component_locality": {
            "changed_arrays": (
                "the returned root, its residual assemblies, channel "
                "decompositions, observables, and the root-minus-frozen "
                "deltas"
            ),
            "inherited_arrays": (
                "the frozen C6 state, hybrid residuals, grid, masks, thermal "
                "arrays, and the authenticated author seed, copied bit-exact"
            ),
            "kernel_identity": (
                "QP-side kernels reproduce the accepted C5 retained kernels "
                "bit-for-bit; phonon-side kernels and the thermal occupation "
                "reproduce the accepted C6 arrays bit-for-bit; the frequency "
                "map reproduces the accepted C6 frozen map bit-for-bit"
            ),
        },
        "conservation": {
            "pair_weighted_number": {
                "net_s_inv_ueV": _float_record(pair_number),
                "statement": (
                    "pair generation/recombination changes QP number by "
                    "construction; recorded as a diagnostic"
                ),
                "turnover_s_inv_ueV": _float_record(pair_turnover),
            },
            "photon_weighted_number": {
                "net_s_inv_ueV": _float_record(photon_number),
                "turnover_s_inv_ueV": _float_record(photon_turnover),
            },
            "scattering_weighted_number": {
                "net_s_inv_ueV": _float_record(scattering_number),
                "turnover_s_inv_ueV": _float_record(scattering_turnover),
            },
        },
        "limitations": {
            "scope": "one authenticated seed/root pair only",
            "statement": (
                "No C7 300-point curve, paper-parity, or "
                "author-iteration-equivalence claim is made. The public "
                "solver's internal Newton path (line search, certificates) "
                "is qpsim semantics bound by the hash-closed solver source; "
                "only its returned root, iteration-count probe, and "
                "host-independent residual certificates are scientific "
                "evidence. The ordinate records are one-point diagnostics, "
                "not curve reproduction."
            ),
        },
        "observable": {
            "author_control_ordinate": _float_record(author_control_ordinate),
            "contract": (
                "the accepted C1 direct-gap observable at the author "
                "left-edge sampling convention; the center-carrier "
                "interpretation is recorded as the C3-documented sampling "
                "diagnostic, not a second claim"
            ),
            "frozen_root_ordinate_authors": _float_record(
                frozen_ordinate_authors
            ),
            "root_driven_gap_eV_authors": _float_record(driven_gap),
            "root_driven_gap_eV_centers": _float_record(driven_gap_centers),
            "root_ordinate_authors": _float_record(ordinate_authors),
            "root_ordinate_centers": _float_record(ordinate_centers),
            "thermal_gap_eV_authors": _float_record(thermal_gap),
            "thermal_gap_eV_centers": _float_record(thermal_gap_centers),
        },
        "parent_bindings": {
            "c0_raw_manifest_sha256": c0_manifest_sha,
            "c0_score_path": c0_score_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c0_score_sha256": hashlib.sha256(c0_score_bytes).hexdigest(),
            "c2_raw_manifest_sha256": c6_parent_bindings.get(
                "c2_raw_manifest_sha256"
            ),
            "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
            "c3_raw_manifest_sha256": c3_manifest_sha,
            "c4_raw_manifest_sha256": c6_parent_bindings.get(
                "c4_raw_manifest_sha256"
            ),
            "c5_raw_manifest_sha256": c5_manifest_sha,
            "c6_raw_manifest_sha256": c6_manifest_sha,
            "c6_raw_schema": C6_RAW_SCHEMA,
            "c6_receipt_path": c6_receipt_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c6_receipt_schema": C6_RECEIPT_SCHEMA,
            "c6_receipt_sha256": hashlib.sha256(c6_receipt_bytes).hexdigest(),
            "c6_score_path": c6_score_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c6_score_schema": C6_SCORE_SCHEMA,
            "c6_score_sha256": hashlib.sha256(c6_score_bytes).hexdigest(),
            "c6_stage_id": PARENT_STAGE_ID,
        },
        "runtime": runtime,
        "schema": SCHEMA,
        "seed_binding": {
            "projection": (
                "author f cell i -> native cell i+20; author phonon level k "
                "-> native omega index k; guard and placeholder entries are "
                "canonical positive zero"
            ),
            "source": (
                "A1 initial (continuation) state retained by the accepted C0 "
                "author-equivalent bundle and bound by the committed C0 "
                "score descriptor"
            ),
        },
        "solver_contract": {
            "accepted_max_iter": accepted_cap,
            "analytic_cross": True,
            "balance_certificates": {
                "limit": _float_record(STEP_RTOL),
                "ph_l1_ratio": _float_record(ph_balance_ratio),
                "qp_l1_ratio": _float_record(qp_balance_ratio),
            },
            "c_phot_ns_inv": _float_record(C_PHOT_NS_INV),
            "declared_max_iter": MAX_ITER,
            "iteration_probe": probe,
            "n_bar": _float_record(inherited_n_bar),
            "omega_0_ueV": _float_record(OMEGA_0_UEV),
            "probe_contract": (
                "the public solver returns only the final state; the "
                "smallest max_iter cap that converges measures the accepted "
                "iteration count, and the declared-cap solve must reproduce "
                "that root bit-for-bit"
            ),
            "step_rtol": _float_record(STEP_RTOL),
            "tau_l_ns": _float_record(TAU_L_NS),
        },
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
        },
        "units": {
            "gap_records": "electronvolt",
            "kernel_arrays": "per nanosecond per microelectronvolt",
            "residual_arrays": "per nanosecond",
            "comparison_arrays": "per second",
        },
    }

    for label, (path, content) in anchors.items():
        _assert_file_snapshot(path, content, label)
    final_c6 = build_c6_score(
        c6_bundle_dir,
        c5_bundle_dir=c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c6_score_bytes(final_c6) != c6_score_bytes:
        raise C7BundleError("C6/C5/C4/C3/C2 evidence changed during C7 construction.")
    _assert_source_snapshots()
    return metadata, arrays


def write_c7_bundle(
    c6_bundle_dir: Path,
    output_dir: Path,
    **kwargs: Any,
) -> Path:
    """Write one immutable C7 raw bundle into a new directory."""

    _assert_source_snapshots()
    if output_dir.resolve().exists() or output_dir.resolve().is_symlink():
        raise FileExistsError(f"C7 output already exists: {output_dir.resolve()}")
    metadata, arrays = build_c7_bundle(c6_bundle_dir, **kwargs)
    root = output_dir.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"C7 output already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            dir=root.parent,
            prefix=f".{root.name}.",
            suffix=".tmp",
        )
    )
    try:
        files: dict[str, dict[str, object]] = {}
        for name, value in sorted(arrays.items()):
            content = _npy_bytes(value)
            filename = f"{name}.npy"
            _write_new_file(temporary_root / filename, content)
            files[filename] = {
                "sha256": hashlib.sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
        manifest = {
            "files": files,
            "metadata": metadata,
            "schema": SCHEMA,
        }
        content = (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        _write_new_file(temporary_root / "manifest.json", content)
        _assert_source_snapshots()
        temporary_root.rename(root)
    except BaseException:
        for child in temporary_root.iterdir():
            child.unlink()
        temporary_root.rmdir()
        raise
    return root / "manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c6-bundle", type=Path, required=True)
    parser.add_argument("--c5-bundle", type=Path, required=True)
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c0-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c7_bundle(
            args.c6_bundle,
            args.output_dir,
            c5_bundle_dir=args.c5_bundle,
            c4_bundle_dir=args.c4_bundle,
            c3_bundle_dir=args.c3_bundle,
            c2_bundle_dir=args.c2_bundle,
            c0_bundle_dir=args.c0_bundle,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
