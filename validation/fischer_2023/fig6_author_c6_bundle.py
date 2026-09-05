"""Build immutable frozen-state evidence for the Figure 6 C6 phonon balance.

Formal C6 is a child of the completed C5 QP-phonon stage.  It holds the
accepted C5 frozen occupation, projected phonon occupation, grid, public QP
channels, and every parameter fixed.  It replaces only the phonon-side
balance: the author phonon scattering, pair, and escape channels become
qpsim's public phonon-side kernels (``build_scattering_kernel_phonon_side``
and ``build_recombination_kernel_phonon_side``), the public frequency map,
the ``compute_phonon_source_sink`` source/sink contraction, and the
``local`` bath-escape form ``(n_th - n_ph) / tau_l`` with the public
``thermal_phonon_occupation``.

Two deliberate endpoint-policy differences from the inherited author
equation are retained rather than hidden:

* the public pair path applies qpsim's Kaplan ``S_+`` pair-breaking
  quadrature correction near the ``2 Delta`` threshold; a same-kernel
  correction-off control isolates it bin-for-bin;
* the public balance is evaluated on the full native 3600-bin omega
  lattice, while the author equation exists only on support
  ``1..1619 micro-eV``; every out-of-support contribution is recorded.

This module never re-evaluates the QP-side channels; the C5 hybrid QP
residual is copied bit-exact.  It does not run Newton, change the frozen
state, or claim a C6 root, stopping history, plotted ordinate, curve,
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
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.phonon_models.local import phonon_balance_diagnostics
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation

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
from validation.fischer_2023.fig6_author_c4_score import load_c4_raw_bundle
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as DEFAULT_C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_SCORE as DEFAULT_C5_SCORE,
)
from validation.fischer_2023.fig6_author_c5_score import (
    RAW_SCHEMA as C5_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    RECEIPT_SCHEMA as C5_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    SCHEMA as C5_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    build_c5_score,
    load_c5_raw_bundle,
    load_c5_receipt,
    load_c5_score,
)
from validation.fischer_2023.fig6_author_c5_score import (
    canonical_score_bytes as canonical_c5_score_bytes,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c6-phonon-balance-bundle.v1"
STAGE_ID = "C6"
PARENT_STAGE_ID = "C5"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "phonon_balance"

SECONDS_PER_NS = 1.0e-9
T_C_K = 1.184309192877208
TAU_0_NS = 438.0
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
T_BATH_K = 0.2
N_QP = 1640
N_OMEGA = 3600
AUTHOR_OMEGA_STOP = 1619
_DETAILED_BALANCE_LIMIT = 1.0e-12
_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}

_CHANNELS = ("scattering", "pair", "pair_control", "escape")
_FIELDS = ("gain", "loss", "net")

ARRAY_NAMES = (
    "c6_qp_residual_s_inv",
    "c6e_phonon_residual_s_inv",
    "c6p0_phonon_residual_s_inv",
    "c6p_phonon_residual_s_inv",
    "c6s_phonon_residual_s_inv",
    "c6spe0_phonon_residual_s_inv",
    "c6spe_phonon_residual_s_inv",
    "parent_E_centers_ueV",
    "parent_active_mask",
    "parent_cell_density",
    "parent_cell_weights_ueV",
    "parent_dE_ueV",
    "parent_f",
    "parent_legacy_phonon_support_mask",
    "parent_phonon_escape_gain_s_inv",
    "parent_phonon_escape_loss_s_inv",
    "parent_phonon_escape_net_s_inv",
    "parent_phonon_pair_gain_s_inv",
    "parent_phonon_pair_loss_s_inv",
    "parent_phonon_pair_net_s_inv",
    "parent_phonon_residual_s_inv",
    "parent_phonon_scattering_gain_s_inv",
    "parent_phonon_scattering_loss_s_inv",
    "parent_phonon_scattering_net_s_inv",
    "parent_phonon_to_native_omega_index",
    "parent_projected_n_phonon",
    "parent_qp_residual_s_inv",
    "qpsim_balance_residual_ns_inv",
    "qpsim_combined_a_ns_inv",
    "qpsim_combined_b_ns_inv",
    "qpsim_db_f",
    "qpsim_db_pair_a_ns_inv",
    "qpsim_db_pair_b_ns_inv",
    "qpsim_db_pair_net_ns_inv",
    "qpsim_db_scattering_a_ns_inv",
    "qpsim_db_scattering_b_ns_inv",
    "qpsim_db_scattering_net_ns_inv",
    "qpsim_diff_sign",
    "qpsim_omega_idx_diff",
    "qpsim_omega_idx_sum",
    "qpsim_omega_ueV",
    "qpsim_pair_a_ns_inv",
    "qpsim_pair_b_ns_inv",
    "qpsim_pair_control_a_ns_inv",
    "qpsim_pair_control_b_ns_inv",
    "qpsim_phonon_pair_kernel_ns_inv_ueV_inv",
    "qpsim_phonon_scattering_kernel_ns_inv_ueV_inv",
    "qpsim_scattering_a_ns_inv",
    "qpsim_scattering_b_ns_inv",
    "qpsim_thermal_n_ph",
    *(
        f"qpsim_phonon_{channel}_{field}_{unit}"
        for channel in _CHANNELS
        for field in _FIELDS
        for unit in ("ns_inv", "s_inv")
    ),
    *(
        f"phonon_{channel}_delta_{field}_s_inv"
        for channel in _CHANNELS
        for field in _FIELDS
    ),
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
    raise RuntimeError("C6 source tree changed while its import snapshot was captured.")


class C6BundleError(ValueError):
    """The C6 parent, frozen operator input, or transport is invalid."""


def _assert_source_snapshots() -> None:
    if (
        source_manifest(
            Path(__file__),
            extra_validation_modules=_REPLAY_VALIDATION_MODULES,
        )
        != _SOURCE_HASHES_AT_IMPORT
    ):
        raise C6BundleError("C6 numerical source closure changed during execution.")
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C6BundleError(f"C6 numerical source changed during execution: {relative}.")


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    """Capture a regular repository-contained file before validation."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise C6BundleError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C6BundleError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C6BundleError(f"{label} must stay inside the repository.") from exc
    if resolved != path.absolute() or resolved.is_symlink() or not resolved.is_file():
        raise C6BundleError(f"{label} is missing, unsafe, or a symlink.")
    try:
        with resolved.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = resolved.lstat()
    except OSError as exc:
        raise C6BundleError(f"{label} changed or became unreadable.") from exc

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
        raise C6BundleError(f"{label} changed while it was being read.")
    return resolved, content


def _assert_file_snapshot(path: Path, expected: bytes, label: str) -> None:
    checked_path, checked = _repository_file_snapshot(path, label)
    if checked_path != path or checked != expected:
        raise C6BundleError(f"{label} changed during C6 construction.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C6BundleError(f"{label} must be an object.")
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
        raise C6BundleError("C6 scalar metadata must be finite.")
    return {"hex": result.hex(), "value": result}


def _runtime_record() -> dict[str, object]:
    thread_environment = {
        name: os.environ.get(name)
        for name in sorted(_THREAD_ENVIRONMENT)
    }
    if thread_environment != _THREAD_ENVIRONMENT:
        raise C6BundleError(
            "C6 BLAS evidence requires MKL_NUM_THREADS=1, "
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
            raise C6BundleError(
                f"Live C6 SpectralContext does not reproduce accepted C3 array {name!r}."
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
        "parent_phonon_to_native_omega_index",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__phonon_{channel}__{field}_s_inv"
            for channel in ("scattering", "pair", "escape")
            for field in ("gain", "loss", "net")
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )


def _channel_totals(values_s_inv: np.ndarray) -> dict[str, object]:
    magnitude = np.abs(np.asarray(values_s_inv, dtype=np.float64))
    return {
        "l1_s_inv": _float_record(float(np.sum(magnitude))),
        "linf_s_inv": _float_record(float(np.max(magnitude, initial=0.0))),
        "nonzero_bins": int(np.count_nonzero(magnitude)),
    }


def _snapshot_repository_anchors(
    *,
    c5_score_path: Path,
    c5_receipt_path: Path,
    c4_score_path: Path,
    c4_receipt_path: Path,
    c3_score_path: Path,
    c3_receipt_path: Path,
    c2_score_path: Path,
    c2_receipt_path: Path,
) -> dict[str, tuple[Path, bytes]]:
    paths = {
        "C5 score": c5_score_path,
        "C5 receipt": c5_receipt_path,
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


def build_c6_bundle(
    c5_bundle_dir: Path,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the formal one-point C5-to-C6 frozen phonon-balance differential."""

    _assert_source_snapshots()
    runtime = _runtime_record()
    anchors = _snapshot_repository_anchors(
        c5_score_path=c5_score_path,
        c5_receipt_path=c5_receipt_path,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    c5_score_path, c5_score_bytes = anchors["C5 score"]
    c5_receipt_path, c5_receipt_bytes = anchors["C5 receipt"]
    c4_score_path, _c4_score_bytes = anchors["C4 score"]
    c4_receipt_path, _c4_receipt_bytes = anchors["C4 receipt"]
    c3_score_path, _c3_score_bytes = anchors["C3 score"]
    c3_receipt_path, _c3_receipt_bytes = anchors["C3 receipt"]
    c2_score_path, _c2_score_bytes = anchors["C2 score"]
    c2_receipt_path, _c2_receipt_bytes = anchors["C2 receipt"]

    accepted_c5 = load_c5_score(c5_score_path, receipt_path=c5_receipt_path)
    accepted_c5_receipt = load_c5_receipt(c5_receipt_path)
    rebuilt_c5 = build_c5_score(
        c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c5_score_bytes(rebuilt_c5) != c5_score_bytes:
        raise C6BundleError(
            "C6 refuses a C5 score/receipt pair that does not independently "
            "reproduce from the selected C5, C4, C3, and C2 raw evidence."
        )

    _c5_metadata, c5_arrays, c5_manifest_sha = load_c5_raw_bundle(c5_bundle_dir)
    c5_raw_binding = _mapping(accepted_c5.get("raw_bundle"), "accepted C5 raw binding")
    if c5_raw_binding != {
        "manifest_sha256": c5_manifest_sha,
        "schema": C5_RAW_SCHEMA,
    }:
        raise C6BundleError("Selected C5 raw bundle is not the accepted parent.")
    c5_stage = _mapping(accepted_c5.get("stage"), "accepted C5 stage")
    if c5_stage != {
        "changed_component": "qp_phonon_operator",
        "comparison_stage_id": "C4",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C4",
        "stage_id": PARENT_STAGE_ID,
        "status": "completed",
    }:
        raise C6BundleError("Accepted parent is not the completed formal C5 stage.")
    c5_receipt_raw = _mapping(
        accepted_c5_receipt.get("raw_bundle"),
        "accepted C5 receipt raw binding",
    )
    if c5_receipt_raw != c5_raw_binding:
        raise C6BundleError("Accepted C5 receipt does not bind the selected raw bundle.")

    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_bundle_dir)
    _c4_metadata, _c4_arrays, c4_manifest_sha = load_c4_raw_bundle(c4_bundle_dir)
    c5_parent_bindings = _mapping(
        accepted_c5.get("parent_bindings"),
        "accepted C5 parent bindings",
    )
    if (
        c5_parent_bindings.get("c3_raw_schema") != C3_RAW_SCHEMA
        or c5_parent_bindings.get("c3_raw_manifest_sha256") != c3_manifest_sha
        or c5_parent_bindings.get("c4_raw_schema") != C4_RAW_SCHEMA
        or c5_parent_bindings.get("c4_raw_manifest_sha256") != c4_manifest_sha
    ):
        raise C6BundleError("Selected C3/C4 raw bundles are not C5's accepted ancestors.")

    c5_frozen_before = {
        name: _array_descriptor(value)
        for name, value in sorted(c5_arrays.items())
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
    inherited_tau_0_pb_s = float(
        parameter_values.get("tau_0_pb_s")  # type: ignore[arg-type]
    )
    inherited_tau_l_s = float(
        parameter_values.get("tau_l_s")  # type: ignore[arg-type]
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
        or inherited_tau_0_pb_s != TAU_PB_NS * SECONDS_PER_NS
        or inherited_tau_l_s != TAU_L_NS * SECONDS_PER_NS
        or inherited_T_bath != T_BATH_K
        or inherited_kB_ueV_per_K != KB_UEV_PER_K
        or parameter_hex.get("tau_0_s") != inherited_tau_0_s.hex()
        or parameter_hex.get("tau_0_pb_s") != inherited_tau_0_pb_s.hex()
        or parameter_hex.get("tau_l_s") != inherited_tau_l_s.hex()
        or parameter_hex.get("boltzmann_constant_J_per_K")
        != inherited_boltzmann.hex()
        or parameter_hex.get("electron_charge_C")
        != inherited_electron_charge.hex()
        or parameter_hex.get("gap_eV") != inherited_gap_eV.hex()
    ):
        raise C6BundleError(
            "C6 parent parameters do not match the exact declared "
            "T_c/tau_0/tau_0_pb/tau_l/T_bath inputs."
        )

    gap_ueV = float(native_parameters.get("gap_ueV"))  # type: ignore[arg-type]
    E = np.asarray(c3_arrays["native_E_centers_ueV"]).copy()
    dE = np.asarray(c3_arrays["native_dE_ueV"]).copy()
    f = np.asarray(c5_arrays["parent_f"]).copy()
    n_ph = np.asarray(c3_arrays["projected_n_phonon"]).copy()
    active = np.asarray(c3_arrays["native_active_mask"]).copy()
    weights = np.asarray(c3_arrays["native_cell_weights_full"]).copy()
    cell_density = np.asarray(c3_arrays["native_cell_density_full"]).copy()
    legacy_phonon_support = np.asarray(
        c3_arrays["legacy_phonon_support_mask"]
    ).copy()
    omega_index_map = np.asarray(
        c3_arrays["parent_phonon_to_native_omega_index"]
    ).copy()
    if not np.array_equal(f, np.asarray(c3_arrays["projected_f"])):
        raise C6BundleError("C5 frozen occupation does not equal its accepted C3 input.")
    if not np.array_equal(
        n_ph,
        np.asarray(c5_arrays["parent_projected_n_phonon"]),
    ):
        raise C6BundleError(
            "C5 frozen phonon occupation does not equal its accepted C3 input."
        )
    if not np.array_equal(
        omega_index_map,
        np.arange(1, AUTHOR_OMEGA_STOP + 1, dtype=np.int64),
    ):
        raise C6BundleError(
            "The inherited author phonon support is not the contiguous native "
            "omega window [1, 1620)."
        )

    parent_scattering_gain = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_scattering__gain_s_inv"]
    ).copy()
    parent_scattering_loss = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_scattering__loss_s_inv"]
    ).copy()
    parent_scattering_net = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_scattering__net_s_inv"]
    ).copy()
    parent_pair_gain = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_pair__gain_s_inv"]
    ).copy()
    parent_pair_loss = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_pair__loss_s_inv"]
    ).copy()
    parent_pair_net = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_pair__net_s_inv"]
    ).copy()
    parent_escape_gain = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_escape__gain_s_inv"]
    ).copy()
    parent_escape_loss = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_escape__loss_s_inv"]
    ).copy()
    parent_escape_net = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_escape__net_s_inv"]
    ).copy()
    parent_phonon_residual = np.asarray(
        c5_arrays["c5sp_phonon_residual_s_inv"]
    ).copy()
    parent_qp_residual = np.asarray(c5_arrays["c5sp_qp_residual_s_inv"]).copy()
    if not np.array_equal(
        parent_phonon_residual,
        np.asarray(
            c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv"]
        ),
    ):
        raise C6BundleError(
            "The C5 hybrid phonon residual is not the bit-exact inherited "
            "C3c author phonon residual."
        )
    parent_channels = {
        ("scattering", "gain"): parent_scattering_gain,
        ("scattering", "loss"): parent_scattering_loss,
        ("scattering", "net"): parent_scattering_net,
        ("pair", "gain"): parent_pair_gain,
        ("pair", "loss"): parent_pair_loss,
        ("pair", "net"): parent_pair_net,
        ("pair_control", "gain"): parent_pair_gain,
        ("pair_control", "loss"): parent_pair_loss,
        ("pair_control", "net"): parent_pair_net,
        ("escape", "gain"): parent_escape_gain,
        ("escape", "loss"): parent_escape_loss,
        ("escape", "net"): parent_escape_net,
    }

    ctx = SpectralContext(E, dE, gap_ueV)
    _assert_context_matches_parent(ctx, c3_arrays)
    omega, omega_idx_diff, omega_idx_sum, diff_sign = build_phonon_frequency_map(E)
    if not np.array_equal(omega, np.asarray(c3_arrays["native_omega_ueV"])):
        raise C6BundleError("Public C6 frequency map does not reproduce C3's native omega grid.")
    for name, live in (
        ("qpsim_omega_ueV", omega),
        ("qpsim_omega_idx_diff", omega_idx_diff),
        ("qpsim_omega_idx_sum", omega_idx_sum),
        ("qpsim_diff_sign", diff_sign),
    ):
        if not np.array_equal(np.asarray(live), np.asarray(c5_arrays[name])):
            raise C6BundleError(
                f"Public C6 frequency map array {name!r} does not reproduce "
                "the accepted C5 frozen map."
            )
    if (
        n_ph.shape != omega.shape
        or legacy_phonon_support.shape != omega.shape
        or active.shape != f.shape
        or weights.shape != f.shape
    ):
        raise C6BundleError("C6 frozen grid/state support shapes are inconsistent.")
    if np.any(n_ph[~legacy_phonon_support] != 0.0):
        raise C6BundleError("C6 projected phonons populate placeholder-only support.")

    K_s_ph = build_scattering_kernel_phonon_side(ctx, TAU_PB_NS)
    K_r_ph = build_recombination_kernel_phonon_side(ctx, TAU_PB_NS)
    n_omega = int(omega.size)

    scattering_a, scattering_b = compute_phonon_source_sink(
        f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=True,
        enable_recombination=False,
        K_s0_phonon_side=K_s_ph,
    )
    pair_a, pair_b = compute_phonon_source_sink(
        f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=False,
        enable_recombination=True,
        K_r0_phonon_side=K_r_ph,
    )
    # Same-kernel correction-off control: the identical phonon-side pair
    # kernel through the QP-side argument slot bypasses only the Kaplan
    # S_+ quadrature correction.
    pair_control_a, pair_control_b = compute_phonon_source_sink(
        f,
        ctx,
        None,
        K_r_ph,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=False,
        enable_recombination=True,
    )
    combined_a, combined_b = compute_phonon_source_sink(
        f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=True,
        enable_recombination=True,
        K_s0_phonon_side=K_s_ph,
        K_r0_phonon_side=K_r_ph,
    )
    if not np.array_equal(combined_a, scattering_a + pair_a):
        raise C6BundleError(
            "The combined public source coefficient is not the bit-exact sum "
            "of its channel evaluations."
        )
    combined_b_split = (scattering_b + pair_b) - combined_b
    combined_b_bound = 4.0 * np.finfo(np.float64).eps * (
        np.abs(combined_b) + np.abs(scattering_b) + np.abs(pair_b)
    )
    if np.any(np.abs(combined_b_split) > combined_b_bound):
        raise C6BundleError(
            "The combined public sink coefficient differs from its channel "
            "sum beyond association-order rounding."
        )

    n_th = thermal_phonon_occupation(omega, T_BATH_K)

    channel_arrays_ns: dict[str, dict[str, np.ndarray]] = {}
    for channel, (a, b) in (
        ("scattering", (scattering_a, scattering_b)),
        ("pair", (pair_a, pair_b)),
        ("pair_control", (pair_control_a, pair_control_b)),
    ):
        emission = a
        absorption = a - b
        gain = emission * (1.0 + n_ph)
        loss = absorption * n_ph
        net = a + b * n_ph
        if np.any(emission < 0.0) or np.any(absorption < 0.0):
            raise C6BundleError(
                f"The public C6 {channel} channel emitted a negative rate."
            )
        channel_arrays_ns[channel] = {"gain": gain, "loss": loss, "net": net}
    channel_arrays_ns["escape"] = {
        "gain": n_th / TAU_L_NS,
        "loss": n_ph / TAU_L_NS,
        "net": (n_th - n_ph) / TAU_L_NS,
    }
    if np.any(channel_arrays_ns["escape"]["gain"] < 0.0) or np.any(
        channel_arrays_ns["escape"]["loss"] < 0.0
    ):
        raise C6BundleError("The public C6 escape channel emitted a negative rate.")

    outside = ~legacy_phonon_support
    if np.any(channel_arrays_ns["scattering"]["net"][outside] != 0.0) or np.any(
        channel_arrays_ns["scattering"]["gain"][outside] != 0.0
    ) or np.any(channel_arrays_ns["scattering"]["loss"][outside] != 0.0):
        raise C6BundleError(
            "The public C6 scattering channel populated bins outside the "
            "author support; scattering frequencies must stay below 1620 ueV."
        )
    zero_bin_values = [
        float(channel_arrays_ns[channel][field][0])
        for channel in _CHANNELS
        for field in _FIELDS
    ]
    if any(value != 0.0 for value in zero_bin_values):
        raise C6BundleError("The omega=0 bookkeeping bin acquired a nonzero rate.")

    balance = phonon_balance_diagnostics(
        n_ph,
        combined_a,
        combined_b,
        n_th,
        tau_l=TAU_L_NS,
    )

    db_f = fermi_dirac_occupation(E, T_BATH_K)
    db_scattering_a, db_scattering_b = compute_phonon_source_sink(
        db_f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=True,
        enable_recombination=False,
        K_s0_phonon_side=K_s_ph,
    )
    db_pair_a, db_pair_b = compute_phonon_source_sink(
        db_f,
        ctx,
        None,
        None,
        omega_idx_diff,
        omega_idx_sum,
        diff_sign,
        n_omega,
        enable_scattering=False,
        enable_recombination=True,
        K_r0_phonon_side=K_r_ph,
    )
    detailed_balance: dict[str, object] = {}
    db_nets_ns: dict[str, np.ndarray] = {}
    for channel, (a, b) in (
        ("scattering", (db_scattering_a, db_scattering_b)),
        ("pair", (db_pair_a, db_pair_b)),
    ):
        net = a + b * n_th
        gain = a * (1.0 + n_th)
        loss = (a - b) * n_th
        turnover = float(np.sum(np.abs(gain) + np.abs(loss)))
        imbalance = float(np.sum(np.abs(net)))
        if turnover <= 0.0:
            raise C6BundleError("The C6 detailed-balance control is vacuous.")
        relative = imbalance / turnover
        if relative > _DETAILED_BALANCE_LIMIT:
            raise C6BundleError(
                f"The public C6 {channel} channel violates detailed balance "
                "at the thermal control state."
            )
        db_nets_ns[channel] = net
        detailed_balance[channel] = {
            "imbalance_l1_ns_inv": _float_record(imbalance),
            "relative_imbalance": _float_record(relative),
            "turnover_l1_ns_inv": _float_record(turnover),
        }
    detailed_balance["contract"] = (
        "The thermal control evaluates the public channels at the native "
        "center-grid Fermi occupation and the public thermal phonon "
        "occupation at T_bath; each channel's |net| L1 over its "
        "gain-plus-loss L1 must stay within the declared limit. The escape "
        "channel vanishes identically at n_ph = n_th and is not re-recorded."
    )
    detailed_balance["limit_relative"] = _float_record(_DETAILED_BALANCE_LIMIT)

    support_idx = omega_index_map
    deltas_s: dict[tuple[str, str], np.ndarray] = {}
    channel_s: dict[tuple[str, str], np.ndarray] = {}
    for channel in _CHANNELS:
        for field in _FIELDS:
            full_s = channel_arrays_ns[channel][field] / SECONDS_PER_NS
            channel_s[(channel, field)] = full_s
            deltas_s[(channel, field)] = (
                full_s[support_idx] - parent_channels[(channel, field)]
            )

    extension_policy: dict[str, object] = {
        "statement": (
            "The public balance is evaluated on the full native 3600-bin "
            "omega lattice. The author equation exists only on support "
            "[1, 1620) micro-eV; the recorded totals are the complete "
            "out-of-support content per channel. The scattering channel is "
            "structurally confined to the support, and the omega=0 "
            "bookkeeping bin is exactly zero."
        ),
    }
    for channel in _CHANNELS:
        extension_policy[channel] = {
            field: _channel_totals(
                channel_s[(channel, field)][outside]
            )
            for field in _FIELDS
        }

    c6s_phonon_residual = parent_phonon_residual + deltas_s[("scattering", "net")]
    c6p_phonon_residual = parent_phonon_residual + deltas_s[("pair", "net")]
    c6p0_phonon_residual = (
        parent_phonon_residual + deltas_s[("pair_control", "net")]
    )
    c6e_phonon_residual = parent_phonon_residual + deltas_s[("escape", "net")]
    c6spe_phonon_residual = (
        parent_phonon_residual
        + deltas_s[("scattering", "net")]
        + deltas_s[("pair", "net")]
        + deltas_s[("escape", "net")]
    )
    c6spe0_phonon_residual = (
        parent_phonon_residual
        + deltas_s[("scattering", "net")]
        + deltas_s[("pair_control", "net")]
        + deltas_s[("escape", "net")]
    )
    c6_qp_residual = parent_qp_residual.copy()

    arrays: dict[str, np.ndarray] = {
        "c6_qp_residual_s_inv": c6_qp_residual,
        "c6e_phonon_residual_s_inv": c6e_phonon_residual,
        "c6p0_phonon_residual_s_inv": c6p0_phonon_residual,
        "c6p_phonon_residual_s_inv": c6p_phonon_residual,
        "c6s_phonon_residual_s_inv": c6s_phonon_residual,
        "c6spe0_phonon_residual_s_inv": c6spe0_phonon_residual,
        "c6spe_phonon_residual_s_inv": c6spe_phonon_residual,
        "parent_E_centers_ueV": E,
        "parent_active_mask": active,
        "parent_cell_density": cell_density,
        "parent_cell_weights_ueV": weights,
        "parent_dE_ueV": dE,
        "parent_f": f,
        "parent_legacy_phonon_support_mask": legacy_phonon_support,
        "parent_phonon_escape_gain_s_inv": parent_escape_gain,
        "parent_phonon_escape_loss_s_inv": parent_escape_loss,
        "parent_phonon_escape_net_s_inv": parent_escape_net,
        "parent_phonon_pair_gain_s_inv": parent_pair_gain,
        "parent_phonon_pair_loss_s_inv": parent_pair_loss,
        "parent_phonon_pair_net_s_inv": parent_pair_net,
        "parent_phonon_residual_s_inv": parent_phonon_residual,
        "parent_phonon_scattering_gain_s_inv": parent_scattering_gain,
        "parent_phonon_scattering_loss_s_inv": parent_scattering_loss,
        "parent_phonon_scattering_net_s_inv": parent_scattering_net,
        "parent_phonon_to_native_omega_index": omega_index_map,
        "parent_projected_n_phonon": n_ph,
        "parent_qp_residual_s_inv": parent_qp_residual,
        "qpsim_balance_residual_ns_inv": balance.residual,
        "qpsim_combined_a_ns_inv": combined_a,
        "qpsim_combined_b_ns_inv": combined_b,
        "qpsim_db_f": db_f,
        "qpsim_db_pair_a_ns_inv": db_pair_a,
        "qpsim_db_pair_b_ns_inv": db_pair_b,
        "qpsim_db_pair_net_ns_inv": db_nets_ns["pair"],
        "qpsim_db_scattering_a_ns_inv": db_scattering_a,
        "qpsim_db_scattering_b_ns_inv": db_scattering_b,
        "qpsim_db_scattering_net_ns_inv": db_nets_ns["scattering"],
        "qpsim_diff_sign": diff_sign,
        "qpsim_omega_idx_diff": omega_idx_diff,
        "qpsim_omega_idx_sum": omega_idx_sum,
        "qpsim_omega_ueV": omega,
        "qpsim_pair_a_ns_inv": pair_a,
        "qpsim_pair_b_ns_inv": pair_b,
        "qpsim_pair_control_a_ns_inv": pair_control_a,
        "qpsim_pair_control_b_ns_inv": pair_control_b,
        "qpsim_phonon_pair_kernel_ns_inv_ueV_inv": K_r_ph,
        "qpsim_phonon_scattering_kernel_ns_inv_ueV_inv": K_s_ph,
        "qpsim_scattering_a_ns_inv": scattering_a,
        "qpsim_scattering_b_ns_inv": scattering_b,
        "qpsim_thermal_n_ph": n_th,
    }
    for channel in _CHANNELS:
        for field in _FIELDS:
            arrays[f"qpsim_phonon_{channel}_{field}_ns_inv"] = (
                channel_arrays_ns[channel][field]
            )
            arrays[f"qpsim_phonon_{channel}_{field}_s_inv"] = (
                channel_s[(channel, field)]
            )
            arrays[f"phonon_{channel}_delta_{field}_s_inv"] = (
                deltas_s[(channel, field)]
            )
    if set(arrays) != set(ARRAY_NAMES) or len(arrays) != len(ARRAY_NAMES):
        raise C6BundleError("C6 raw array-name closure is inconsistent.")
    for name, value in tuple(arrays.items()):
        if value.dtype.kind == "f":
            normalized = np.asarray(value).copy()
            normalized[normalized == 0.0] = 0.0
            arrays[name] = normalized

    c5_frozen_after = {
        name: _array_descriptor(value)
        for name, value in sorted(c5_arrays.items())
    }
    c3_frozen_after = _descriptor_subset(c3_arrays, c3_frozen_names)
    parameters_after = json.dumps(
        c3_metadata.get("parameters"),
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    if c5_frozen_after != c5_frozen_before:
        raise C6BundleError("A C6 frozen evaluation mutated its C5 parent arrays.")
    if c3_frozen_after != c3_frozen_before or parameters_after != parameters_before:
        raise C6BundleError("A C6 frozen evaluation mutated its C3 ancestor inputs.")

    metadata: dict[str, Any] = {
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "balance_certification": {
            "certified_backward_error": _float_record(
                balance.certified_backward_error
            ),
            "contract": (
                "qpsim's public phonon_balance_diagnostics evaluates the full "
                "three-term balance a + b*n_ph + (n_th - n_ph)/tau_l at the "
                "frozen state with per-bin FMA arithmetic. The frozen state "
                "is an author-model root, not a qpsim balance root, so these "
                "backward errors are frozen-state diagnostics, not "
                "convergence gates."
            ),
            "raw_backward_error": _float_record(balance.raw_backward_error),
        },
        "bookkeeping_contract": {
            "affine_channel_decomposition": (
                "each e-ph channel is retained as public affine coefficients "
                "(a, b) with dn_ph/dt = a + b*n_ph; gain = a*(1+n_ph), loss = "
                "(a-b)*n_ph, and net = a + b*n_ph are declared derived "
                "identities of that decomposition"
            ),
            "combined_evaluation": (
                "the single public call with both channels enabled is "
                "retained separately; its source coefficient equals the "
                "channel sum bit-for-bit while its sink coefficient differs "
                "only by association order within 4-eps elementwise bounds"
            ),
            "escape_form": (
                "escape gain = n_th/tau_l, loss = n_ph/tau_l, net = "
                "(n_th - n_ph)/tau_l with the public thermal phonon "
                "occupation, following the local balance form"
            ),
            "kaplan_pair_correction": (
                "the public pair path scales each omega bin by qpsim's "
                "Kaplan S_+ quadrature correction; the same-kernel "
                "correction-off control isolates that policy bin-for-bin, "
                "and the public-minus-control difference is the recorded "
                "endpoint semantic change"
            ),
        },
        "comparison_contract": {
            "candidate": (
                "public qpsim phonon-side scattering/recombination kernels, "
                "frequency map, source/sink contraction, and bath-escape "
                "balance evaluated per channel"
            ),
            "escape_comparison": (
                "escape nets difference against the parent is bounded by "
                "elementwise rounding of the thermal-occupation unit paths; "
                "a bulk relative gate would misread benign near-thermal "
                "cancellation"
            ),
            "parent": (
                "accepted C3c author-form phonon scattering/pair/escape "
                "channels carried bit-exact through the completed C4 and C5 "
                "parents"
            ),
            "parent_qp": (
                "accepted C5 hybrid QP residual, copied bit-exact and not "
                "re-evaluated"
            ),
            "public_arithmetic": (
                "matrix contractions may use BLAS; an independent verifier "
                "must apply declared floating-point tolerances rather than "
                "demand bit equality to a source-order transcription"
            ),
        },
        "component_locality": {
            "c6e": "replace only the phonon escape net in the parent residual",
            "c6p": (
                "replace only the phonon pair net using the public "
                "Kaplan-corrected path"
            ),
            "c6p0": (
                "replace only the phonon pair net using the same-kernel "
                "correction-off control"
            ),
            "c6s": "replace only the phonon scattering net in the parent residual",
            "c6spe": (
                "replace scattering, public pair, and escape nets; this is "
                "the formal C6 hybrid phonon residual"
            ),
            "c6spe0": (
                "replace scattering, correction-off pair, and escape nets; "
                "this isolates the Kaplan correction inside the formal "
                "residual"
            ),
            "changed_arrays": (
                "phonon-side kernels, affine coefficients, per-channel "
                "gain/loss/net, support-restricted deltas, and the derived "
                "C6 phonon residuals"
            ),
            "inherited_arrays": (
                "C5 hybrid QP residual plus every C3c phonon-channel array, "
                "copied bit-exact"
            ),
            "phonon_residual_updates": {
                "c6e": "parent_phonon_residual + phonon_escape_delta_net",
                "c6p": "parent_phonon_residual + phonon_pair_delta_net",
                "c6p0": (
                    "parent_phonon_residual + phonon_pair_control_delta_net"
                ),
                "c6s": (
                    "parent_phonon_residual + phonon_scattering_delta_net"
                ),
                "c6spe": (
                    "parent_phonon_residual + phonon_scattering_delta_net "
                    "+ phonon_pair_delta_net + phonon_escape_delta_net"
                ),
                "c6spe0": (
                    "parent_phonon_residual + phonon_scattering_delta_net "
                    "+ phonon_pair_control_delta_net "
                    "+ phonon_escape_delta_net"
                ),
            },
            "qp_residual_bit_exact": bool(
                np.array_equal(c6_qp_residual, parent_qp_residual)
            ),
        },
        "coordinate_contract": {
            "active_child_indices": "[20, 1640)",
            "author_support_window": (
                "native omega indices [1, 1620), mapped by the retained "
                "identity index array"
            ),
            "frequency_map": (
                "public build_phonon_frequency_map on the accepted C3 "
                "1640-cell center grid, bit-identical to the accepted C5 "
                "frozen map"
            ),
            "guard_child_indices": "[0, 20), canonical positive zero",
            "native_cell_count": int(f.size),
            "native_omega_count": int(omega.size),
            "omega_zero_bin": (
                "index 0 is the zero-transfer bookkeeping mode and stays "
                "exactly zero in every channel"
            ),
        },
        "detailed_balance": detailed_balance,
        "extension_policy": extension_policy,
        "frozen_inputs": {
            "c3_descriptors": c3_frozen_before,
            "c5_descriptors": c5_frozen_before,
            "c3_mutation_check_after_operator": True,
            "c5_mutation_check_after_operator": True,
            "policy": (
                "accepted C5 f/QP residual/frequency map and accepted C3 "
                "grid, masks, cell density, K_minus/K_plus, projected n_ph, "
                "parameters, and every phonon-channel array are immutable"
            ),
            "qp_equation_descriptor_names": [
                "c5sp_qp_residual_s_inv",
            ],
        },
        "limitations": {
            "scope": "one authenticated C5 frozen point only",
            "statement": (
                "No C6 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, observable change, "
                "coupled QP-phonon conservation claim, or paper-parity claim "
                "is made. The QP-side channels remain the accepted C5 "
                "evidence and are not re-evaluated in C6."
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
            "tau_0_pb_ns": _float_record(TAU_PB_NS),
            "tau_0_pb_parent_s": _float_record(inherited_tau_0_pb_s),
            "tau_l_ns": _float_record(TAU_L_NS),
            "tau_l_parent_s": _float_record(inherited_tau_l_s),
        },
        "parent_bindings": {
            "c2_raw_manifest_sha256": c5_parent_bindings.get(
                "c2_raw_manifest_sha256"
            ),
            "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
            "c3_raw_manifest_sha256": c3_manifest_sha,
            "c3_raw_schema": C3_RAW_SCHEMA,
            "c4_raw_manifest_sha256": c4_manifest_sha,
            "c4_raw_schema": C4_RAW_SCHEMA,
            "c5_raw_manifest_sha256": c5_manifest_sha,
            "c5_raw_schema": C5_RAW_SCHEMA,
            "c5_receipt_path": c5_receipt_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c5_receipt_schema": C5_RECEIPT_SCHEMA,
            "c5_receipt_sha256": hashlib.sha256(c5_receipt_bytes).hexdigest(),
            "c5_score_path": c5_score_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "c5_score_schema": C5_SCORE_SCHEMA,
            "c5_score_sha256": hashlib.sha256(c5_score_bytes).hexdigest(),
            "c5_stage_id": PARENT_STAGE_ID,
        },
        "runtime": runtime,
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "complete qpsim Python/material source tree, C6 producer, "
                "C2/C3/C4/C5 replay verifiers, C5 producer, and provenance "
                "helper"
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
            "affine_coefficient_arrays": (
                "a per nanosecond; b per nanosecond per occupation"
            ),
            "comparison_arrays": "per second",
            "kernel_arrays": "per nanosecond per microelectronvolt",
            "public_native_arrays": "per nanosecond",
            "public_return_contract": (
                "compute_phonon_source_sink returns affine (a, b) with "
                "dn_ph/dt = a + b*n_ph; occupations are dimensionless"
            ),
        },
    }

    _recheck_repository_anchors(anchors)
    final_c5 = build_c5_score(
        c5_bundle_dir,
        c4_bundle_dir=c4_bundle_dir,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c4_score_path=c4_score_path,
        c4_receipt_path=c4_receipt_path,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_c5_score_bytes(final_c5) != c5_score_bytes:
        raise C6BundleError("C5/C4/C3/C2 evidence changed during C6 construction.")
    _recheck_repository_anchors(anchors)
    _assert_source_snapshots()
    return metadata, arrays


def write_c6_bundle(
    c5_bundle_dir: Path,
    output_dir: Path,
    *,
    c4_bundle_dir: Path,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c5_score_path: Path = DEFAULT_C5_SCORE,
    c5_receipt_path: Path = DEFAULT_C5_RECEIPT,
    c4_score_path: Path = DEFAULT_C4_SCORE,
    c4_receipt_path: Path = DEFAULT_C4_RECEIPT,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    """Write one immutable C6 raw bundle into a new directory."""

    _assert_source_snapshots()
    if output_dir.resolve().exists() or output_dir.resolve().is_symlink():
        raise FileExistsError(f"C6 output already exists: {output_dir.resolve()}")
    metadata, arrays = build_c6_bundle(
        c5_bundle_dir,
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
    root = output_dir.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"C6 output already exists: {root}")
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
    parser.add_argument("--c5-bundle", type=Path, required=True)
    parser.add_argument("--c4-bundle", type=Path, required=True)
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c5-score", type=Path, default=DEFAULT_C5_SCORE)
    parser.add_argument("--c5-receipt", type=Path, default=DEFAULT_C5_RECEIPT)
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
        write_c6_bundle(
            args.c5_bundle,
            args.output_dir,
            c4_bundle_dir=args.c4_bundle,
            c3_bundle_dir=args.c3_bundle,
            c2_bundle_dir=args.c2_bundle,
            c5_score_path=args.c5_score,
            c5_receipt_path=args.c5_receipt,
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
