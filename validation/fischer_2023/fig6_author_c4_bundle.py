"""Build immutable frozen-state evidence for the Figure 6 C4 photon stage.

Formal C4 is a child of the completed C3 grid stage.  It holds the accepted
``C3c`` state, grid, finite-volume coherence, and native cell density fixed,
then replaces only the clean-room author photon residual with
``qpsim.collisions.sub_gap_photon.sub_gap_photon_collision_rates``.

The authenticated author residual omits transitions touching its final QP
cell.  The public qpsim operator includes every representable supported
partner, including the final pair.  This bundle records three distinct
quantities so that ordinary floating-point reordering cannot be confused with
that semantic endpoint change:

* the accepted C3c author-form gain/loss/net arrays;
* the public qpsim gain/loss/net arrays;
* a qpsim-arithmetic control with the author terminal-cell omission restored.

This is one frozen operator comparison.  It does not run Newton, change the
state, or claim a C4 root, stopping history, plotted ordinate, curve, or paper
parity.
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
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.physics.spectral import SpectralContext

from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as DEFAULT_C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_SCORE as DEFAULT_C3_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import (
    RAW_SCHEMA as C3_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    build_c3_score,
    load_c3_raw_bundle,
    load_c3_receipt,
    load_c3_score,
)
from validation.fischer_2023.fig6_author_c3_score import (
    canonical_score_bytes as canonical_c3_score_bytes,
)
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c4-photon-bundle.v1"
STAGE_ID = "C4"
PARENT_STAGE_ID = "C3"
PARENT_OPERATOR_STAGE_ID = "c3c_native_cell_density"
CHANGED_COMPONENT = "photon_operator"
SECONDS_PER_NS = 1.0e-9

_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "qpsim" / "collisions" / "_uniform_grid.py",
    REPOSITORY_ROOT / "qpsim" / "collisions" / "_validation.py",
    REPOSITORY_ROOT / "qpsim" / "collisions" / "sub_gap_photon.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "bcs_quadrature.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "spectral.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c3_score.py",
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}


class C4BundleError(ValueError):
    """The C4 parent, frozen operator input, or transport is invalid."""


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C4BundleError(f"C4 numerical source changed during execution: {relative}.")


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    """Capture a regular repository-contained file before validation."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise C4BundleError(f"{label} is missing or unreadable.") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise C4BundleError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C4BundleError(f"{label} must stay inside the repository.") from exc
    if resolved != path.absolute() or resolved.is_symlink() or not resolved.is_file():
        raise C4BundleError(f"{label} is missing, unsafe, or a symlink.")
    try:
        with resolved.open("rb") as handle:
            opened_before = os.fstat(handle.fileno())
            content = handle.read()
            opened_after = os.fstat(handle.fileno())
        after = resolved.lstat()
    except OSError as exc:
        raise C4BundleError(f"{label} changed or became unreadable.") from exc

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
        raise C4BundleError(f"{label} changed while it was being read.")
    return resolved, content


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C4BundleError(f"{label} must be an object.")
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
        raise C4BundleError("C4 scalar metadata must be finite.")
    return {"hex": result.hex(), "value": result}


def _author_endpoint_public_arithmetic(
    f: np.ndarray,
    ctx: SpectralContext,
    *,
    photon_step: int,
    n_bar: float,
    c_photon_ns_inv: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate public arithmetic while restoring the author endpoint policy."""

    occupation = np.asarray(f, dtype=float)
    size = occupation.size
    rho = ctx.cell_density
    supported = ctx.active_mask
    K_plus = ctx.K_plus
    gain = np.zeros(size)
    loss_rate = np.zeros(size)
    one_minus_f = np.maximum(1.0 - occupation, 0.0)

    for i in range(size):
        if not supported[i]:
            continue
        j_up = i + photon_step
        if j_up < size - 1:
            coefficient = rho[j_up] * K_plus[i, j_up]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_up] * (n_bar + 1.0)
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_up] * n_bar
        j_down = i - photon_step
        if j_down >= 0 and supported[j_down] and i < size - 1:
            coefficient = rho[j_down] * K_plus[i, j_down]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_down] * n_bar
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_down] * (n_bar + 1.0)

    gain *= one_minus_f
    loss = loss_rate * occupation
    return gain, loss, gain - loss


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
        "native_K_plus_full": ctx.K_plus,
    }
    for name, live in comparisons.items():
        if not np.array_equal(np.asarray(live), np.asarray(arrays[name])):
            raise C4BundleError(
                f"Live C4 SpectralContext does not reproduce accepted C3 array {name!r}."
            )


def build_c4_bundle(
    c3_bundle_dir: Path,
    *,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build the formal one-point C3c-to-C4 frozen photon differential."""

    _assert_source_snapshots()
    c3_score_path, c3_score_bytes = _repository_file_snapshot(
        c3_score_path,
        "C3 score",
    )
    c3_receipt_path, c3_receipt_bytes = _repository_file_snapshot(
        c3_receipt_path,
        "C3 receipt",
    )
    accepted_c3 = load_c3_score(c3_score_path, receipt_path=c3_receipt_path)
    accepted_c3_receipt = load_c3_receipt(c3_receipt_path)
    checked_score_path, checked_score_bytes = _repository_file_snapshot(
        c3_score_path,
        "C3 score",
    )
    checked_receipt_path, checked_receipt_bytes = _repository_file_snapshot(
        c3_receipt_path,
        "C3 receipt",
    )
    if (
        checked_score_path != c3_score_path
        or checked_receipt_path != c3_receipt_path
        or checked_score_bytes != c3_score_bytes
        or checked_receipt_bytes != c3_receipt_bytes
    ):
        raise C4BundleError("C3 score or receipt changed during C4 validation.")
    rebuilt_c3 = build_c3_score(
        c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
    )
    if canonical_c3_score_bytes(rebuilt_c3) != c3_score_bytes:
        raise C4BundleError(
            "C4 refuses a C3 score/receipt pair that does not independently "
            "reproduce from the selected C3 and C2 raw evidence."
        )
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(c3_bundle_dir)
    raw_binding = _mapping(accepted_c3.get("raw_bundle"), "accepted C3 raw binding")
    if (
        raw_binding.get("schema") != C3_RAW_SCHEMA
        or raw_binding.get("manifest_sha256") != c3_manifest_sha
    ):
        raise C4BundleError("Selected C3 raw bundle is not the accepted parent.")
    stage = _mapping(accepted_c3.get("stage"), "accepted C3 stage")
    if stage != {
        "changed_component": "grid_sampling",
        "comparison_stage_id": "C2",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C2",
        "stage_id": "C3",
        "status": "completed",
    }:
        raise C4BundleError("Accepted parent is not the completed formal C3 stage.")

    parameter_record = _mapping(c3_metadata.get("parameters"), "C3 parameters")
    parameter_values = _mapping(parameter_record.get("values"), "C3 parameter values")
    native_parameters = _mapping(
        c3_metadata.get("native_qpsim_grid_parameters"),
        "C3 native grid parameters",
    )
    projected_f = np.asarray(c3_arrays["projected_f"]).copy()
    frozen_names = (
        "projected_f",
        "native_E_centers_ueV",
        "native_dE_ueV",
        "native_active_mask",
        "native_cell_density_full",
        "native_cell_weights_full",
        "native_K_plus_full",
        *(
            f"{PARENT_OPERATOR_STAGE_ID}__{channel}__{field}_s_inv"
            for channel in (
                "qp_photon",
                "qp_scattering",
                "qp_pair",
                "phonon_scattering",
                "phonon_pair",
                "phonon_escape",
            )
            for field in ("gain", "loss", "net")
        ),
        f"{PARENT_OPERATOR_STAGE_ID}__qp_residual_s_inv",
        f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
    )
    frozen_before = {name: _array_descriptor(c3_arrays[name]) for name in frozen_names}

    gap_ueV = float(native_parameters.get("gap_ueV"))  # type: ignore[arg-type]
    ctx = SpectralContext(
        np.asarray(c3_arrays["native_E_centers_ueV"]),
        np.asarray(c3_arrays["native_dE_ueV"]),
        gap_ueV,
    )
    _assert_context_matches_parent(ctx, c3_arrays)

    photon_bin_raw = parameter_values.get("photon_bin")
    if isinstance(photon_bin_raw, bool) or not isinstance(
        photon_bin_raw,
        (int, np.integer),
    ):
        raise C4BundleError("C3 photon_bin is not an integer.")
    photon_step = int(photon_bin_raw)
    h_eV = float(parameter_values.get("h_eV"))  # type: ignore[arg-type]
    n_bar = float(parameter_values.get("n_bar"))  # type: ignore[arg-type]
    c_photon_s_inv = float(
        parameter_values.get("c_photon_s_inv")  # type: ignore[arg-type]
    )
    if (
        photon_step <= 0
        or not np.isfinite(h_eV)
        or h_eV <= 0.0
        or not np.isfinite(n_bar)
        or n_bar < 0.0
        or not np.isfinite(c_photon_s_inv)
        or c_photon_s_inv < 0.0
    ):
        raise C4BundleError("C4 photon inputs inherited from C3 are invalid.")
    omega_0_ueV = photon_step * h_eV * 1.0e6
    dE_ueV = float(np.asarray(c3_arrays["native_dE_ueV"])[0])
    snapped_step = round(omega_0_ueV / dE_ueV)
    if (
        photon_step != snapped_step
        or omega_0_ueV != photon_step * dE_ueV
        or omega_0_ueV >= 2.0 * gap_ueV
    ):
        raise C4BundleError("C4 photon energy is not the exact inherited sub-gap bin.")
    c_photon_ns_inv = c_photon_s_inv * SECONDS_PER_NS

    public_gain_ns, public_loss_rate_ns = sub_gap_photon_collision_rates(
        projected_f,
        ctx,
        omega_0_ueV,
        n_bar,
        c_photon_ns_inv,
    )
    public_loss_ns = public_loss_rate_ns * projected_f
    public_net_ns = public_gain_ns - public_loss_ns
    public_gain_s = public_gain_ns / SECONDS_PER_NS
    public_loss_s = public_loss_ns / SECONDS_PER_NS
    public_net_s = public_net_ns / SECONDS_PER_NS

    endpoint_gain_ns, endpoint_loss_ns, endpoint_net_ns = _author_endpoint_public_arithmetic(
        projected_f,
        ctx,
        photon_step=photon_step,
        n_bar=n_bar,
        c_photon_ns_inv=c_photon_ns_inv,
    )
    endpoint_gain_s = endpoint_gain_ns / SECONDS_PER_NS
    endpoint_loss_s = endpoint_loss_ns / SECONDS_PER_NS
    endpoint_net_s = endpoint_net_ns / SECONDS_PER_NS

    parent_gain = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__gain_s_inv"]).copy()
    parent_loss = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__loss_s_inv"]).copy()
    parent_net = np.asarray(c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_photon__net_s_inv"]).copy()
    terminal_gain = public_gain_s - endpoint_gain_s
    terminal_loss = public_loss_s - endpoint_loss_s
    terminal_net = public_net_s - endpoint_net_s
    terminal_support = (terminal_gain != 0.0) | (terminal_loss != 0.0) | (terminal_net != 0.0)
    expected_terminal_indices = np.array(
        [projected_f.size - 1 - photon_step, projected_f.size - 1],
        dtype=np.int64,
    )
    if not np.array_equal(np.flatnonzero(terminal_support), expected_terminal_indices):
        raise C4BundleError("C4 terminal-policy differential has unexpected support.")
    parent_qp_residual = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__qp_residual_s_inv"]
    ).copy()
    parent_phonon_residual = np.asarray(
        c3_arrays[f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv"]
    ).copy()
    hybrid_qp_residual = parent_qp_residual + (public_net_s - parent_net)
    hybrid_phonon_residual = parent_phonon_residual.copy()

    arrays: dict[str, np.ndarray] = {
        "arithmetic_delta_gain_s_inv": endpoint_gain_s - parent_gain,
        "arithmetic_delta_loss_s_inv": endpoint_loss_s - parent_loss,
        "arithmetic_delta_net_s_inv": endpoint_net_s - parent_net,
        "operator_delta_gain_s_inv": public_gain_s - parent_gain,
        "operator_delta_loss_s_inv": public_loss_s - parent_loss,
        "operator_delta_net_s_inv": public_net_s - parent_net,
        "hybrid_phonon_residual_s_inv": hybrid_phonon_residual,
        "hybrid_qp_residual_s_inv": hybrid_qp_residual,
        "parent_active_mask": np.asarray(c3_arrays["native_active_mask"]).copy(),
        "parent_cell_weights_ueV": np.asarray(c3_arrays["native_cell_weights_full"]).copy(),
        "parent_f": projected_f.copy(),
        "parent_qp_photon_gain_s_inv": parent_gain,
        "parent_qp_photon_loss_s_inv": parent_loss,
        "parent_qp_photon_net_s_inv": parent_net,
        "parent_phonon_residual_s_inv": parent_phonon_residual,
        "parent_qp_residual_s_inv": parent_qp_residual,
        "qpsim_author_endpoint_gain_s_inv": endpoint_gain_s,
        "qpsim_author_endpoint_loss_s_inv": endpoint_loss_s,
        "qpsim_author_endpoint_net_s_inv": endpoint_net_s,
        "qpsim_gain_ns_inv": public_gain_ns,
        "qpsim_gain_s_inv": public_gain_s,
        "qpsim_loss_ns_inv": public_loss_ns,
        "qpsim_loss_rate_ns_inv": public_loss_rate_ns,
        "qpsim_loss_s_inv": public_loss_s,
        "qpsim_net_ns_inv": public_net_ns,
        "qpsim_net_s_inv": public_net_s,
        "terminal_extension_gain_s_inv": terminal_gain,
        "terminal_extension_loss_s_inv": terminal_loss,
        "terminal_extension_net_s_inv": terminal_net,
        "terminal_extension_support_mask": terminal_support,
    }
    if np.any(arrays["qpsim_gain_ns_inv"] < 0.0) or np.any(arrays["qpsim_loss_rate_ns_inv"] < 0.0):
        raise C4BundleError("The public C4 operator emitted a negative gain/loss rate.")
    if np.any(arrays["qpsim_gain_ns_inv"][:20] != 0.0) or np.any(
        arrays["qpsim_loss_rate_ns_inv"][:20] != 0.0
    ):
        raise C4BundleError("The public C4 operator populated zero-capacity guards.")

    frozen_after = {name: _array_descriptor(c3_arrays[name]) for name in frozen_names}
    if frozen_after != frozen_before:
        raise C4BundleError("A C4 frozen evaluation mutated its C3 parent arrays.")

    metadata: dict[str, Any] = {
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "comparison_contract": {
            "arithmetic_control": (
                "qpsim source-order arithmetic and per-nanosecond units, "
                "with the author terminal omission restored"
            ),
            "candidate": "public qpsim sub-gap photon gain and loss_rate",
            "loss_comparison": (
                "physical loss = returned loss_rate * frozen f; the raw "
                "loss_rate coefficient is never compared directly to C3c loss"
            ),
            "parent": "accepted C3c author-form QP photon gain/loss/net",
            "semantic_delta": (
                "candidate minus arithmetic control, isolated from candidate "
                "minus C3c floating-point reordering"
            ),
        },
        "coordinate_contract": {
            "active_child_indices": "[20, 1640)",
            "coherence": "accepted C3c native SpectralContext K_plus",
            "density": "accepted C3c native partner cell_density",
            "guard_child_indices": "[0, 20), canonical positive zero",
            "native_cell_count": int(projected_f.size),
            "photon_mapping": "child i <-> child i+20; no interpolation",
        },
        "endpoint_contract": {
            "author_parent": (
                "C3c author residual omits both directions of every photon "
                "transition touching the final active QP cell"
            ),
            "diagnostic_control": (
                "public qpsim arithmetic and units with the author terminal omission restored"
            ),
            "qpsim_candidate": (
                "all representable supported i+/-m partners, including the final active QP cell"
            ),
            "terminal_child_indices": expected_terminal_indices.tolist(),
        },
        "component_locality": {
            "changed_arrays": ("QP photon gain, physical loss, net, and the resulting QP residual"),
            "inherited_arrays": (
                "QP scattering, QP pair, all three phonon channels, and "
                "the phonon residual remain the accepted C3c arrays"
            ),
            "phonon_residual_bit_exact": bool(
                np.array_equal(hybrid_phonon_residual, parent_phonon_residual)
            ),
            "qp_residual_update": (
                "hybrid_qp_residual = parent_qp_residual + (qpsim_photon_net - parent_photon_net)"
            ),
        },
        "frozen_inputs": {
            "descriptors": frozen_before,
            "mutation_check_after_operator": True,
            "policy": (
                "accepted C3c state, grid, active mask, cell weights, "
                "cell_density, K_plus, and author-form photon arrays are immutable"
            ),
        },
        "limitations": {
            "scope": "one authenticated C3c frozen point only",
            "statement": (
                "No C4 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, observable change, or "
                "paper-parity claim is made. Non-photon channels are inherited "
                "from C3 and are not re-evaluated."
            ),
        },
        "operator_inputs": {
            "c_photon_ns_inv": _float_record(c_photon_ns_inv),
            "c_photon_s_inv": _float_record(c_photon_s_inv),
            "dE_ueV": _float_record(dE_ueV),
            "gap_ueV": _float_record(gap_ueV),
            "n_bar": _float_record(n_bar),
            "omega_0_ueV": _float_record(omega_0_ueV),
            "photon_step_bins": photon_step,
            "seconds_per_ns": _float_record(SECONDS_PER_NS),
            "snap_fraction_of_bin": _float_record(abs(omega_0_ueV - photon_step * dE_ueV) / dE_ueV),
        },
        "parent_bindings": {
            "c2_raw_manifest_sha256": _mapping(
                accepted_c3.get("parent_bindings"),
                "accepted C3 parent bindings",
            ).get("c2_raw_manifest_sha256"),
            "c3_operator_stage_id": PARENT_OPERATOR_STAGE_ID,
            "c3_raw_manifest_sha256": c3_manifest_sha,
            "c3_raw_schema": C3_RAW_SCHEMA,
            "c3_receipt_path": c3_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c3_receipt_schema": accepted_c3_receipt.get("schema"),
            "c3_receipt_sha256": hashlib.sha256(c3_receipt_bytes).hexdigest(),
            "c3_score_path": c3_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c3_score_schema": accepted_c3.get("schema"),
            "c3_score_sha256": hashlib.sha256(c3_score_bytes).hexdigest(),
            "c3_stage_id": PARENT_STAGE_ID,
        },
        "runtime": {
            "byteorder": sys.byteorder,
            "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_version": platform.python_version(),
        },
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": (
                "C4 producer, public sub-gap photon operator and validators, "
                "SpectralContext/quadrature, accepted C3 loader, and provenance source"
            ),
        },
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_OPERATOR_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
        },
        "units": {
            "comparison_arrays": "per second",
            "public_native_arrays": "per nanosecond",
            "public_return_contract": (
                "gain includes target Pauli factor; loss_rate multiplies f to form actual loss"
            ),
        },
    }
    final_score_path, final_score_bytes = _repository_file_snapshot(
        c3_score_path,
        "C3 score",
    )
    final_receipt_path, final_receipt_bytes = _repository_file_snapshot(
        c3_receipt_path,
        "C3 receipt",
    )
    if (
        final_score_path != c3_score_path
        or final_receipt_path != c3_receipt_path
        or final_score_bytes != c3_score_bytes
        or final_receipt_bytes != c3_receipt_bytes
    ):
        raise C4BundleError("C3 score or receipt changed during C4 construction.")
    _assert_source_snapshots()
    return metadata, arrays


def write_c4_bundle(
    c3_bundle_dir: Path,
    output_dir: Path,
    *,
    c2_bundle_dir: Path,
    c3_score_path: Path = DEFAULT_C3_SCORE,
    c3_receipt_path: Path = DEFAULT_C3_RECEIPT,
) -> Path:
    """Write one immutable C4 raw bundle into a new directory."""

    _assert_source_snapshots()
    metadata, arrays = build_c4_bundle(
        c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c3_score_path=c3_score_path,
        c3_receipt_path=c3_receipt_path,
    )
    root = output_dir.resolve()
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"C4 output already exists: {root}")
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
        content = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
            "utf-8"
        )
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
    parser.add_argument("--c3-bundle", type=Path, required=True)
    parser.add_argument("--c2-bundle", type=Path, required=True)
    parser.add_argument("--c3-score", type=Path, default=DEFAULT_C3_SCORE)
    parser.add_argument("--c3-receipt", type=Path, default=DEFAULT_C3_RECEIPT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c4_bundle(
            args.c3_bundle,
            args.output_dir,
            c2_bundle_dir=args.c2_bundle,
            c3_score_path=args.c3_score,
            c3_receipt_path=args.c3_receipt,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
