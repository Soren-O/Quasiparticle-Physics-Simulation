"""Build immutable frozen-state evidence for the Figure 6 C5 QP operator.

Formal C5 is a child of the completed C4 photon stage.  It holds the accepted
C4 frozen occupation, grid, public photon channel, every phonon-equation
channel, and every parameter fixed.  It replaces only the QP-equation
electron--phonon scattering and pair channels with the public qpsim
``build_*_kernel_base`` and ``phonon_collision_rates`` path.

The author scattering implementation used source-order bookkeeping buckets:
the same Pauli cross-term appears in both its reported gain and loss.  That
term cancels from the physical net.  C5 therefore records the raw buckets,
the cross-term, rebucketed controls, and the net separately.  Pair gain and
loss are already physical buckets.

This module never evaluates the phonon balance and, in particular, never
calls ``compute_phonon_source_sink``.  It does not run Newton, change the
frozen state, or claim a C5 root, stopping history, plotted ordinate, curve,
observable change, or paper parity.
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
    build_scattering_kernel_base,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.spectral import SpectralContext

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
from validation.fischer_2023.fig6_author_c3_score import (
    RAW_SCHEMA as C3_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import load_c3_raw_bundle
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as DEFAULT_C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_SCORE as DEFAULT_C4_SCORE,
)
from validation.fischer_2023.fig6_author_c4_score import (
    RAW_SCHEMA as C4_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    RECEIPT_SCHEMA as C4_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    SCHEMA as C4_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    build_c4_score,
    load_c4_raw_bundle,
    load_c4_receipt,
    load_c4_score,
)
from validation.fischer_2023.fig6_author_c4_score import (
    canonical_score_bytes as canonical_c4_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c5-qp-phonon-bundle.v1"
STAGE_ID = "C5"
PARENT_STAGE_ID = "C4"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "qp_phonon_operator"

SECONDS_PER_NS = 1.0e-9
T_C_K = 1.184309192877208
TAU_0_NS = 438.0
T_BATH_K = 0.2
_SCATTERING_CONSERVATION_LIMIT = 1.0e-12
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

ARRAY_NAMES = (
    "c5p_qp_residual_s_inv",
    "c5s_qp_residual_s_inv",
    "c5sp_phonon_residual_s_inv",
    "c5sp_qp_residual_s_inv",
    "parent_E_centers_ueV",
    "parent_active_mask",
    "parent_cell_weights_ueV",
    "parent_dE_ueV",
    "parent_f",
    "parent_legacy_phonon_support_mask",
    "parent_phonon_residual_s_inv",
    "parent_projected_n_phonon",
    "parent_public_qp_photon_gain_s_inv",
    "parent_public_qp_photon_loss_s_inv",
    "parent_public_qp_photon_net_s_inv",
    "parent_qp_pair_gain_s_inv",
    "parent_qp_pair_loss_s_inv",
    "parent_qp_pair_net_s_inv",
    "parent_qp_residual_s_inv",
    "parent_qp_scattering_gain_s_inv",
    "parent_qp_scattering_loss_s_inv",
    "parent_qp_scattering_net_s_inv",
    "parent_qp_scattering_rebucketed_gain_s_inv",
    "parent_qp_scattering_rebucketed_loss_s_inv",
    "qp_pair_delta_gain_s_inv",
    "qp_pair_delta_loss_s_inv",
    "qp_pair_delta_net_s_inv",
    "qp_scattering_delta_gain_s_inv",
    "qp_scattering_delta_loss_s_inv",
    "qp_scattering_delta_net_s_inv",
    "qp_scattering_rebucketed_delta_gain_s_inv",
    "qp_scattering_rebucketed_delta_loss_s_inv",
    "qpsim_N_abs",
    "qpsim_N_emit",
    "qpsim_N_p",
    "qpsim_diff_sign",
    "qpsim_omega_idx_diff",
    "qpsim_omega_idx_sum",
    "qpsim_omega_ueV",
    "qpsim_qp_pair_gain_ns_inv",
    "qpsim_qp_pair_gain_s_inv",
    "qpsim_qp_pair_kernel_ns_inv_ueV_inv",
    "qpsim_qp_pair_loss_ns_inv",
    "qpsim_qp_pair_loss_rate_ns_inv",
    "qpsim_qp_pair_loss_rate_s_inv",
    "qpsim_qp_pair_loss_s_inv",
    "qpsim_qp_pair_net_ns_inv",
    "qpsim_qp_pair_net_s_inv",
    "qpsim_qp_scattering_gain_ns_inv",
    "qpsim_qp_scattering_gain_s_inv",
    "qpsim_qp_scattering_kernel_ns_inv_ueV_inv",
    "qpsim_qp_scattering_loss_ns_inv",
    "qpsim_qp_scattering_loss_rate_ns_inv",
    "qpsim_qp_scattering_loss_rate_s_inv",
    "qpsim_qp_scattering_loss_s_inv",
    "qpsim_qp_scattering_net_ns_inv",
    "qpsim_qp_scattering_net_s_inv",
    "scattering_pauli_cross_term_s_inv",
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
    raise RuntimeError("C5 source tree changed while its import snapshot was captured.")


class C5BundleError(ValueError):
    """The C5 parent, frozen operator input, or transport is invalid."""


def _assert_source_snapshots() -> None:
    if (
        source_manifest(
            Path(__file__),
            extra_validation_modules=_REPLAY_VALIDATION_MODULES,
        )
        != _SOURCE_HASHES_AT_IMPORT
    ):
        raise C5BundleError("C5 numerical source closure changed during execution.")
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C5BundleError(f"C5 numerical source changed during execution: {relative}.")


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    """Capture a regular repository-contained file before validation."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise C5BundleError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C5BundleError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C5BundleError(f"{label} must stay inside the repository.") from exc
    if resolved != path.absolute() or resolved.is_symlink() or not resolved.is_file():
        raise C5BundleError(f"{label} is missing, unsafe, or a symlink.")
    try:
        with resolved.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = resolved.lstat()
    except OSError as exc:
        raise C5BundleError(f"{label} changed or became unreadable.") from exc

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
        raise C5BundleError(f"{label} changed while it was being read.")
    return resolved, content


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    checked_path, checked = _repository_file_snapshot(path, label)
    if checked_path != path or checked != expected:
        raise C5BundleError(f"{label} changed during C5 construction.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C5BundleError(f"{label} must be an object.")
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
    """Exclusively publish one complete file inside the temporary bundle."""

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
        raise C5BundleError("C5 scalar metadata must be finite.")
    return {"hex": result.hex(), "value": result}


def _runtime_record() -> dict[str, object]:
    thread_environment = {
        name: os.environ.get(name)
        for name in sorted(_THREAD_ENVIRONMENT)
    }
    if thread_environment != _THREAD_ENVIRONMENT:
        raise C5BundleError(
            "C5 BLAS evidence requires MKL_NUM_THREADS=1, "
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


def _descriptor_subset(
    arrays: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    return {name: _array_descriptor(arrays[name]) for name in names}


def _assert_context_matches_parent(
    ctx: SpectralContext,
    arrays: dict[str, np.ndarray],
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
        if not np.array_equal(np.asarray(live), np.asarray(arrays[name])):
            raise C5BundleError(
                f"Live C5 SpectralContext does not reproduce accepted C3 array {name!r}."
            )


def _c3_frozen_names() -> tuple[str, ...]:
    return (
        "projected_f",
        "projected_n_phonon",
        "native_E_centers_ueV",
        "native_dE_ueV",
        "native_active_mask",
        "native_cell_density_full",
        "native_cell_weights_full",
        "native_K_minus_full",
        "native_K_plus_full",
        "native_omega_ueV",
        "legacy_phonon_support_mask",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__qp_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair")
            for field in ("gain", "loss", "net")
        ),
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair", "escape")
            for field in ("gain", "loss", "net")
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )


def _snapshot_repository_anchors(
    *,
    c4_score_path: Path,
    c4_receipt_path: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
) -> dict[str, tuple[Path, bytes]]:
    paths = {
        "C4 score": c4_score_path,
        "C4 receipt": c4_receipt_path,
        "C3 score": c3_score_path,
        "C3 receipt": c3_receipt_path,
        "C2 score": c2_score_path,
        "C2 receipt": c2_receipt_path,
    }
    return {
        label: _repository_file_snapshot(path, label)
        for label, path in paths.items()
    }


def _recheck_repository_anchors(
    snapshots: dict[str, tuple[Path, bytes]],
) -> None:
    for label, (path, content) in snapshots.items():
        _assert_file_snapshot(path, content, label)


def build_c5_bundle(
    c4_bundle_dir: Path,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the formal one-point C4-to-C5 frozen QP-operator differential."""

    _assert_source_snapshots()
    runtime = _runtime_record()
    anchors = _snapshot_repository_anchors(
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    c4_score_path, c4_score_bytes = anchors["C4 score"]
    c4_receipt_path, c4_receipt_bytes = anchors["C4 receipt"]
    c3_score_path, _c3_score_bytes = anchors["C3 score"]
    c3_receipt_path, _c3_receipt_bytes = anchors["C3 receipt"]
    c2_score_path, _c2_score_bytes = anchors["C2 score"]
    c2_receipt_path, _c2_receipt_bytes = anchors["C2 receipt"]

    accepted_c4 = load_c4_score(c4_score_path, receipt_path=c4_receipt_path)
    accepted_c4_receipt = load_c4_receipt(c4_receipt_path)
    rebuilt_c4 = build_c4_score(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c4_score_bytes(rebuilt_c4) != c4_score_bytes:
        raise C5BundleError(
            "C5 refuses a C4 score/receipt pair that does not independently "
            "reproduce from the selected C4, C3, and C2 raw evidence."
        )

    _c4_metadata, c4_arrays, c4_manifest_sha = load_c4_raw_bundle(c4_bundle_dir)
    c4_raw_binding = _mapping(accepted_c4.get("raw_bundle"), "accepted C4 raw binding")
    if c4_raw_binding != {
        "manifest_sha256": c4_manifest_sha,
        "schema": C4_RAW_SCHEMA,
    }:
        raise C5BundleError("Selected C4 raw bundle is not the accepted parent.")
    c4_stage = _mapping(accepted_c4.get("stage"), "accepted C4 stage")
    if c4_stage != {
        "changed_component": "photon_operator",
        "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C3",
        "stage_id": PARENT_STAGE_ID,
        "status": "completed",
    }:
        raise C5BundleError("Accepted parent is not the completed formal C4 stage.")
    c4_receipt_raw = _mapping(
        accepted_c4_receipt.get("raw_bundle"),
        "accepted C4 receipt raw binding",
    )
    if c4_receipt_raw != c4_raw_binding:
        raise C5BundleError("Accepted C4 receipt does not bind the selected raw bundle.")

    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_bundle_dir)
    c4_parent_bindings = _mapping(
        accepted_c4.get("parent_bindings"),
        "accepted C4 parent bindings",
    )
    if (
        c4_parent_bindings.get("c3_raw_schema") != C3_RAW_SCHEMA
        or c4_parent_bindings.get("c3_raw_manifest_sha256") != c3_manifest_sha
    ):
        raise C5BundleError("Selected C3 raw bundle is not C4's accepted ancestor.")

    c4_frozen_before = {
        name: _array_descriptor(value)
        for name, value in sorted(c4_arrays.items())
    }
    c3_frozen_names = _c3_frozen_names()
    c3_frozen_before = _descriptor_subset(c3_arrays, c3_frozen_names)
    parameters_before = json.dumps(
        c3_metadata.get("parameters"),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )

    parameter_record = _mapping(c3_metadata.get("parameters"), "C3 parameters")
    parameter_values = _mapping(parameter_record.get("values"), "C3 parameter values")
    parameter_hex = _mapping(parameter_record.get("hex"), "C3 parameter hex values")
    native_parameters = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "C3 native grid parameters",
    )
    inherited_T_c = float(parameter_values.get("T_c_K"))  # type: ignore[arg-type]
    inherited_tau_0_s = float(
        parameter_values.get("tau_0_s")  # type: ignore[arg-type]
    )
    inherited_T_bath = float(
        parameter_values.get("temperature_K")  # type: ignore[arg-type]
    )
    inherited_boltzmann = float(
        parameter_values.get("boltzmann_constant_J_per_K")  # type: ignore[arg-type]
    )
    inherited_electron_charge = float(
        parameter_values.get("electron_charge_C")  # type: ignore[arg-type]
    )
    inherited_gap_eV = float(
        parameter_values.get("gap_eV")  # type: ignore[arg-type]
    )
    inherited_kB_ueV_per_K = (
        inherited_boltzmann / inherited_electron_charge * 1.0e6
    )
    if (
        inherited_T_c != T_C_K
        or inherited_tau_0_s != TAU_0_NS * SECONDS_PER_NS
        or inherited_T_bath != T_BATH_K
        or inherited_kB_ueV_per_K != KB_UEV_PER_K
        or parameter_hex.get("tau_0_s") != inherited_tau_0_s.hex()
        or parameter_hex.get("boltzmann_constant_J_per_K")
        != inherited_boltzmann.hex()
        or parameter_hex.get("electron_charge_C")
        != inherited_electron_charge.hex()
        or parameter_hex.get("gap_eV") != inherited_gap_eV.hex()
    ):
        raise C5BundleError(
            "C5 parent parameters do not match the exact declared "
            "T_c/tau_0/T_bath inputs."
        )

    gap_ueV = float(native_parameters.get("gap_ueV"))  # type: ignore[arg-type]
    E = np.asarray(c3_arrays["native_E_centers_ueV"]).copy()
    dE = np.asarray(c3_arrays["native_dE_ueV"]).copy()
    f = np.asarray(c4_arrays["parent_f"]).copy()
    n_ph = np.asarray(c3_arrays["projected_n_phonon"]).copy()
    active = np.asarray(c3_arrays["native_active_mask"]).copy()
    weights = np.asarray(c3_arrays["native_cell_weights_full"]).copy()
    legacy_phonon_support = np.asarray(
        c3_arrays["legacy_phonon_support_mask"]
    ).copy()
    if not np.array_equal(f, np.asarray(c3_arrays["projected_f"])):
        raise C5BundleError("C4 frozen occupation does not equal its accepted C3 input.")

    ctx = SpectralContext(E, dE, gap_ueV)
    _assert_context_matches_parent(ctx, c3_arrays)
    omega, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(E)
    if not np.array_equal(omega, np.asarray(c3_arrays["native_omega_ueV"])):
        raise C5BundleError("Public C5 frequency map does not reproduce C3's native omega grid.")
    if (
        n_ph.shape != omega.shape
        or legacy_phonon_support.shape != omega.shape
        or active.shape != f.shape
        or weights.shape != f.shape
    ):
        raise C5BundleError("C5 frozen grid/state support shapes are inconsistent.")
    if np.any(n_ph[~legacy_phonon_support] != 0.0):
        raise C5BundleError("C5 projected phonons populate placeholder-only support.")

    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_ph,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
    )
    K_s0 = build_scattering_kernel_base(ctx, TAU_0_NS, T_C_K)
    K_r0 = build_recombination_kernel_base(ctx, TAU_0_NS, T_C_K)

    scattering_gain_ns, scattering_loss_rate_ns = phonon_collision_rates(
        f,
        ctx,
        K_s0,
        None,
        T_BATH_K,
        enable_scattering=True,
        enable_recombination=False,
        N_p_override=N_p,
    )
    pair_gain_ns, pair_loss_rate_ns = phonon_collision_rates(
        f,
        ctx,
        None,
        K_r0,
        T_BATH_K,
        enable_scattering=False,
        enable_recombination=True,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    scattering_loss_ns = scattering_loss_rate_ns * f
    scattering_net_ns = scattering_gain_ns - scattering_loss_ns
    pair_loss_ns = pair_loss_rate_ns * f
    pair_net_ns = pair_gain_ns - pair_loss_ns

    scattering_gain_s = scattering_gain_ns / SECONDS_PER_NS
    scattering_loss_rate_s = scattering_loss_rate_ns / SECONDS_PER_NS
    scattering_loss_s = scattering_loss_ns / SECONDS_PER_NS
    scattering_net_s = scattering_net_ns / SECONDS_PER_NS
    pair_gain_s = pair_gain_ns / SECONDS_PER_NS
    pair_loss_rate_s = pair_loss_rate_ns / SECONDS_PER_NS
    pair_loss_s = pair_loss_ns / SECONDS_PER_NS
    pair_net_s = pair_net_ns / SECONDS_PER_NS
    scattering_weighted_number_drift = float(
        np.sum(weights * scattering_net_s)
    )
    scattering_weighted_number_turnover = float(
        np.sum(weights * (np.abs(scattering_gain_s) + np.abs(scattering_loss_s)))
    )
    if scattering_weighted_number_turnover <= 0.0:
        raise C5BundleError("The public C5 scattering comparison is vacuous.")
    scattering_weighted_number_relative_drift = (
        abs(scattering_weighted_number_drift)
        / scattering_weighted_number_turnover
    )
    if (
        scattering_weighted_number_relative_drift
        > _SCATTERING_CONSERVATION_LIMIT
    ):
        raise C5BundleError(
            "The public C5 scattering operator violates weighted QP-number "
            "conservation."
        )
    pair_weighted_number_net = float(np.sum(weights * pair_net_s))
    pair_weighted_number_turnover = float(
        np.sum(weights * (np.abs(pair_gain_s) + np.abs(pair_loss_s)))
    )

    c3_prefix = PARENT_OPERATOR_STAGE_ID
    parent_scattering_gain = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_scattering__gain_s_inv"]
    ).copy()
    parent_scattering_loss = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_scattering__loss_s_inv"]
    ).copy()
    parent_scattering_net = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_scattering__net_s_inv"]
    ).copy()
    parent_pair_gain = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_pair__gain_s_inv"]
    ).copy()
    parent_pair_loss = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_pair__loss_s_inv"]
    ).copy()
    parent_pair_net = np.asarray(
        c3_arrays[f"{c3_prefix}__qp_pair__net_s_inv"]
    ).copy()
    parent_photon_gain = np.asarray(c4_arrays["qpsim_gain_s_inv"]).copy()
    parent_photon_loss = np.asarray(c4_arrays["qpsim_loss_s_inv"]).copy()
    parent_photon_net = np.asarray(c4_arrays["qpsim_net_s_inv"]).copy()
    parent_qp_residual = np.asarray(c4_arrays["hybrid_qp_residual_s_inv"]).copy()
    parent_phonon_residual = np.asarray(
        c4_arrays["hybrid_phonon_residual_s_inv"]
    ).copy()

    n_diff = n_ph[omega_idx_diff]
    pauli_cross_ns = f * ((K_s0 * n_diff).T @ (weights * f))
    pauli_cross_s = pauli_cross_ns / SECONDS_PER_NS
    parent_scattering_rebucketed_gain = parent_scattering_gain - pauli_cross_s
    parent_scattering_rebucketed_loss = parent_scattering_loss - pauli_cross_s

    scattering_delta_gain = scattering_gain_s - parent_scattering_gain
    scattering_delta_loss = scattering_loss_s - parent_scattering_loss
    scattering_delta_net = scattering_net_s - parent_scattering_net
    pair_delta_gain = pair_gain_s - parent_pair_gain
    pair_delta_loss = pair_loss_s - parent_pair_loss
    pair_delta_net = pair_net_s - parent_pair_net
    scattering_rebucketed_delta_gain = (
        scattering_gain_s - parent_scattering_rebucketed_gain
    )
    scattering_rebucketed_delta_loss = (
        scattering_loss_s - parent_scattering_rebucketed_loss
    )

    c5s_qp_residual = parent_qp_residual + scattering_delta_net
    c5p_qp_residual = parent_qp_residual + pair_delta_net
    c5sp_qp_residual = (
        parent_qp_residual + scattering_delta_net + pair_delta_net
    )
    c5sp_phonon_residual = parent_phonon_residual.copy()

    arrays: dict[str, np.ndarray] = {
        "c5p_qp_residual_s_inv": c5p_qp_residual,
        "c5s_qp_residual_s_inv": c5s_qp_residual,
        "c5sp_phonon_residual_s_inv": c5sp_phonon_residual,
        "c5sp_qp_residual_s_inv": c5sp_qp_residual,
        "parent_E_centers_ueV": E,
        "parent_active_mask": active,
        "parent_cell_weights_ueV": weights,
        "parent_dE_ueV": dE,
        "parent_f": f,
        "parent_legacy_phonon_support_mask": legacy_phonon_support,
        "parent_phonon_residual_s_inv": parent_phonon_residual,
        "parent_projected_n_phonon": n_ph,
        "parent_public_qp_photon_gain_s_inv": parent_photon_gain,
        "parent_public_qp_photon_loss_s_inv": parent_photon_loss,
        "parent_public_qp_photon_net_s_inv": parent_photon_net,
        "parent_qp_pair_gain_s_inv": parent_pair_gain,
        "parent_qp_pair_loss_s_inv": parent_pair_loss,
        "parent_qp_pair_net_s_inv": parent_pair_net,
        "parent_qp_residual_s_inv": parent_qp_residual,
        "parent_qp_scattering_gain_s_inv": parent_scattering_gain,
        "parent_qp_scattering_loss_s_inv": parent_scattering_loss,
        "parent_qp_scattering_net_s_inv": parent_scattering_net,
        "parent_qp_scattering_rebucketed_gain_s_inv": (
            parent_scattering_rebucketed_gain
        ),
        "parent_qp_scattering_rebucketed_loss_s_inv": (
            parent_scattering_rebucketed_loss
        ),
        "qp_pair_delta_gain_s_inv": pair_delta_gain,
        "qp_pair_delta_loss_s_inv": pair_delta_loss,
        "qp_pair_delta_net_s_inv": pair_delta_net,
        "qp_scattering_delta_gain_s_inv": scattering_delta_gain,
        "qp_scattering_delta_loss_s_inv": scattering_delta_loss,
        "qp_scattering_delta_net_s_inv": scattering_delta_net,
        "qp_scattering_rebucketed_delta_gain_s_inv": (
            scattering_rebucketed_delta_gain
        ),
        "qp_scattering_rebucketed_delta_loss_s_inv": (
            scattering_rebucketed_delta_loss
        ),
        "qpsim_N_abs": N_abs,
        "qpsim_N_emit": N_emit,
        "qpsim_N_p": N_p,
        "qpsim_diff_sign": diff_sign,
        "qpsim_omega_idx_diff": omega_idx_diff,
        "qpsim_omega_idx_sum": omega_idx_sum,
        "qpsim_omega_ueV": omega,
        "qpsim_qp_pair_gain_ns_inv": pair_gain_ns,
        "qpsim_qp_pair_gain_s_inv": pair_gain_s,
        "qpsim_qp_pair_kernel_ns_inv_ueV_inv": K_r0,
        "qpsim_qp_pair_loss_ns_inv": pair_loss_ns,
        "qpsim_qp_pair_loss_rate_ns_inv": pair_loss_rate_ns,
        "qpsim_qp_pair_loss_rate_s_inv": pair_loss_rate_s,
        "qpsim_qp_pair_loss_s_inv": pair_loss_s,
        "qpsim_qp_pair_net_ns_inv": pair_net_ns,
        "qpsim_qp_pair_net_s_inv": pair_net_s,
        "qpsim_qp_scattering_gain_ns_inv": scattering_gain_ns,
        "qpsim_qp_scattering_gain_s_inv": scattering_gain_s,
        "qpsim_qp_scattering_kernel_ns_inv_ueV_inv": K_s0,
        "qpsim_qp_scattering_loss_ns_inv": scattering_loss_ns,
        "qpsim_qp_scattering_loss_rate_ns_inv": scattering_loss_rate_ns,
        "qpsim_qp_scattering_loss_rate_s_inv": scattering_loss_rate_s,
        "qpsim_qp_scattering_loss_s_inv": scattering_loss_s,
        "qpsim_qp_scattering_net_ns_inv": scattering_net_ns,
        "qpsim_qp_scattering_net_s_inv": scattering_net_s,
        "scattering_pauli_cross_term_s_inv": pauli_cross_s,
    }
    if set(arrays) != set(ARRAY_NAMES) or len(arrays) != len(ARRAY_NAMES):
        raise C5BundleError("C5 raw array-name closure is inconsistent.")
    for name, value in tuple(arrays.items()):
        if value.dtype.kind == "f":
            normalized = np.asarray(value).copy()
            normalized[normalized == 0.0] = 0.0
            arrays[name] = normalized

    for channel in ("scattering", "pair"):
        if np.any(arrays[f"qpsim_qp_{channel}_gain_ns_inv"] < 0.0) or np.any(
            arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"] < 0.0
        ):
            raise C5BundleError(f"The public C5 {channel} operator emitted a negative rate.")
        if np.any(arrays[f"qpsim_qp_{channel}_gain_ns_inv"][~active] != 0.0) or np.any(
            arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"][~active] != 0.0
        ):
            raise C5BundleError(f"The public C5 {channel} operator populated unsupported cells.")

    c4_frozen_after = {
        name: _array_descriptor(value)
        for name, value in sorted(c4_arrays.items())
    }
    c3_frozen_after = _descriptor_subset(c3_arrays, c3_frozen_names)
    parameters_after = json.dumps(
        c3_metadata.get("parameters"),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if c4_frozen_after != c4_frozen_before:
        raise C5BundleError("A C5 frozen evaluation mutated its C4 parent arrays.")
    if c3_frozen_after != c3_frozen_before or parameters_after != parameters_before:
        raise C5BundleError("A C5 frozen evaluation mutated its C3 ancestor inputs.")

    metadata: dict[str, Any] = {
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "bookkeeping_contract": {
            "pair_buckets": (
                "parent and public pair gain/loss are physical generation and "
                "physical loss buckets"
            ),
            "pair_weighted_number": {
                "net_s_inv_ueV": _float_record(pair_weighted_number_net),
                "statement": (
                    "pair generation/recombination changes QP number by "
                    "construction; no zero-drift conservation gate is applied"
                ),
                "turnover_s_inv_ueV": _float_record(
                    pair_weighted_number_turnover
                ),
            },
            "scattering_cross_term": (
                "author source-order scattering gain and loss each contain the "
                "same Pauli cross-term; it cancels exactly from the source-order net"
            ),
            "scattering_pauli_cross_term_formula": (
                "f * ((K_s0 * n_ph[omega_idx_diff]).T @ "
                "(cell_weights * f))"
            ),
            "scattering_rebucketed_controls": (
                "parent rebucketed gain/loss = parent source-order bucket "
                "- scattering_pauli_cross_term"
            ),
            "scattering_weighted_number_conservation": {
                "absolute_drift_s_inv_ueV": _float_record(
                    abs(scattering_weighted_number_drift)
                ),
                "limit_relative": _float_record(
                    _SCATTERING_CONSERVATION_LIMIT
                ),
                "relative_drift": _float_record(
                    scattering_weighted_number_relative_drift
                ),
                "turnover_s_inv_ueV": _float_record(
                    scattering_weighted_number_turnover
                ),
            },
            "warning": (
                "raw public-minus-parent scattering gain/loss are bookkeeping "
                "differentials; rebucketed gain/loss and the physical net are "
                "the like-for-like comparisons"
            ),
        },
        "comparison_contract": {
            "candidate": (
                "public qpsim QP-equation base scattering/recombination kernels "
                "and phonon_collision_rates evaluated separately"
            ),
            "loss_comparison": (
                "physical loss = returned loss_rate * frozen f; loss_rate is "
                "also retained separately and is never compared to parent loss"
            ),
            "parent": (
                "accepted C3c author-form QP scattering/pair channels carried "
                "through the completed C4 public-photon parent"
            ),
            "parent_photon": (
                "accepted C4 public photon gain/loss/net, copied bit-exact and "
                "not re-evaluated"
            ),
            "public_arithmetic": (
                "matrix products may use BLAS; an independent verifier must "
                "apply declared floating-point tolerances rather than demand "
                "bit equality to a source-order transcription"
            ),
        },
        "component_locality": {
            "c5p": "replace only the QP pair net in the C4 parent residual",
            "c5s": "replace only the QP scattering net in the C4 parent residual",
            "c5sp": (
                "replace both QP scattering and QP pair nets; this is the "
                "formal C5 hybrid QP residual"
            ),
            "changed_arrays": (
                "QP scattering/pair kernels, gain, physical loss, net, and "
                "the derived C5s/C5p/C5sp QP residuals"
            ),
            "inherited_arrays": (
                "C4 public photon channel plus all C3c phonon scattering, "
                "phonon pair, phonon escape, and phonon residual arrays"
            ),
            "phonon_residual_bit_exact": bool(
                np.array_equal(c5sp_phonon_residual, parent_phonon_residual)
            ),
            "qp_residual_updates": {
                "c5p": "parent_qp_residual + qp_pair_delta_net",
                "c5s": "parent_qp_residual + qp_scattering_delta_net",
                "c5sp": (
                    "parent_qp_residual + qp_scattering_delta_net "
                    "+ qp_pair_delta_net"
                ),
            },
        },
        "coordinate_contract": {
            "active_child_indices": "[20, 1640)",
            "frequency_map": (
                "public build_phonon_frequency_map on the accepted C3 "
                "1640-cell center grid"
            ),
            "guard_child_indices": "[0, 20), canonical positive zero",
            "legacy_phonon_support": (
                "omega indices [1, 1620); every other projected n_ph entry "
                "is canonical zero"
            ),
            "native_cell_count": int(f.size),
            "native_omega_count": int(omega.size),
            "phonon_projection": (
                "public phonon_occupation_matrices_from_state on the frozen "
                "C3 projected_n_phonon"
            ),
        },
        "frozen_inputs": {
            "c3_descriptors": c3_frozen_before,
            "c4_descriptors": c4_frozen_before,
            "c4_mutation_check_after_operator": True,
            "c3_mutation_check_after_operator": True,
            "phonon_equation_descriptor_names": [
                *(
                    f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
                    for channel in ("scattering", "pair", "escape")
                    for field in ("gain", "loss", "net")
                ),
                f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
            ],
            "policy": (
                "accepted C4 f/public photon/residual and accepted C3 grid, "
                "masks, K_minus/K_plus, projected n_ph, parameters, and every "
                "phonon-equation channel are immutable"
            ),
        },
        "limitations": {
            "scope": "one authenticated C4 frozen point only",
            "statement": (
                "No C5 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, observable change, "
                "coupled QP-phonon conservation claim, or paper-parity claim "
                "is made. The phonon balance remains the inherited author "
                "equation and is not evaluated by qpsim in C5."
            ),
        },
        "operator_inputs": {
            "T_bath_K": _float_record(T_BATH_K),
            "T_c_K": _float_record(T_C_K),
            "boltzmann_constant_J_per_K": _float_record(
                inherited_boltzmann
            ),
            "electron_charge_C": _float_record(inherited_electron_charge),
            "gap_parent_eV": _float_record(inherited_gap_eV),
            "gap_ueV": _float_record(gap_ueV),
            "kB_T_c_ueV": _float_record(KB_UEV_PER_K * T_C_K),
            "kB_ueV_per_K": _float_record(KB_UEV_PER_K),
            "seconds_per_ns": _float_record(SECONDS_PER_NS),
            "tau_0_ns": _float_record(TAU_0_NS),
            "tau_0_parent_s": _float_record(inherited_tau_0_s),
        },
        "parent_bindings": {
            "c2_raw_manifest_sha256": c4_parent_bindings.get(
                "c2_raw_manifest_sha256"
            ),
            "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
            "c3_raw_manifest_sha256": c3_manifest_sha,
            "c3_raw_schema": C3_RAW_SCHEMA,
            "c4_raw_manifest_sha256": c4_manifest_sha,
            "c4_raw_schema": C4_RAW_SCHEMA,
            "c4_receipt_path": c4_receipt_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c4_receipt_schema": C4_RECEIPT_SCHEMA,
            "c4_receipt_sha256": hashlib.sha256(c4_receipt_bytes).hexdigest(),
            "c4_score_path": c4_score_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c4_score_schema": C4_SCORE_SCHEMA,
            "c4_score_sha256": hashlib.sha256(c4_score_bytes).hexdigest(),
            "c4_stage_id": PARENT_STAGE_ID,
        },
        "runtime": runtime,
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "complete qpsim Python/material source tree, C5 producer, "
                "C2/C3/C4 replay verifiers, and provenance helper"
            ),
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
            "comparison_arrays": "per second",
            "kernel_arrays": "per nanosecond per microelectronvolt",
            "public_native_arrays": "per nanosecond",
            "public_return_contract": (
                "gain includes target Pauli factor; loss_rate multiplies f "
                "to form physical loss"
            ),
        },
    }

    _recheck_repository_anchors(anchors)
    final_c4 = build_c4_score(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c4_score_bytes(final_c4) != c4_score_bytes:
        raise C5BundleError("C4/C3/C2 evidence changed during C5 construction.")
    _recheck_repository_anchors(anchors)
    _assert_source_snapshots()
    return metadata, arrays


def write_c5_bundle(
    c4_bundle_dir: Path,
    output_dir: Path,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    """Write one immutable C5 raw bundle into a new directory."""

    _assert_source_snapshots()
    metadata, arrays = build_c5_bundle(
        c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    root = output_dir.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"C5 output already exists: {root}")
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
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c4-score", type=Path, default=DEFAULT_C4_SCORE)
    parser.add_argument("--c4-receipt", type=Path, default=DEFAULT_C4_RECEIPT)
    parser.add_argument("--c3-score", type=Path, default=DEFAULT_C3_SCORE)
    parser.add_argument("--c3-receipt", type=Path, default=DEFAULT_C3_RECEIPT)
    parser.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    parser.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c5_bundle(
            args.c4_bundle,
            args.output_dir,
            c3_bundle_dir=args.c3_bundle,
            c2_bundle_dir=args.c2_bundle,
            c4_score_path=args.c4_score,
            c4_receipt_path=args.c4_receipt,
            c3_score_path=args.c3_score,
            c3_receipt_path=args.c3_receipt,
            c2_score_path=args.c2_score,
            c2_receipt_path=args.c2_receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
