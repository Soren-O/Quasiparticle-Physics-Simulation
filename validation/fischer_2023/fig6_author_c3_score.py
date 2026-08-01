"""Independently verify and summarize the formal Figure 6 C3 grid evidence.

The C3 producer deliberately lives in a different module.  This verifier does
not import it, its staged operator builder, or the clean-room author evaluator.
It reloads the external raw transport strictly, derives the 1640-cell qpsim
finite-volume geometry locally, and transcribes the six frozen author channel
balances in source order.  C3 is only a one-point frozen-state differential:
no nonlinear C3 root, Newton history, plotted ordinate, curve, or paper-parity
claim is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_c2_bundle import (
    SCHEMA as C2_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c2_bundle import (
    load_c2_raw_bundle,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as DEFAULT_C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as DEFAULT_C2_SCORE,
)
from validation.fischer_2023.fig6_author_c2_score import (
    load_c2_score,
)
from validation.source_provenance import canonical_source_bytes, source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_SCHEMA = "qpsim.fischer2023.fig6-author-c3-grid-bundle.v1"
SCHEMA = "qpsim.fischer2023.fig6-author-c3-grid-score.v1"
RECEIPT_SCHEMA = "qpsim.fischer2023.fig6-author-c3-raw-manifest-receipt.v1"
DEFAULT_SCORE = (
    REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6" / "c3-grid-score.json"
)
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c3-raw-manifest-receipt.json"
)

STAGE_ID = "C3"
PARENT_STAGE_ID = "C2"
CHANGED_COMPONENT = "grid_sampling"
FINAL_C2_STEP_ID = "C2b5-finite-cutoff-critical-temperature"
FINAL_C2_SLUG = "c2b5_finite_cutoff_critical_temperature"
STAGE_IDS = (
    "c3p_projected_author_control",
    "c3a_finite_volume_coherence",
    "c3b_center_pair_labels",
    "c3c_native_cell_density",
)
CHANNEL_NAMES = (
    "qp_photon",
    "qp_scattering",
    "qp_pair",
    "phonon_scattering",
    "phonon_pair",
    "phonon_escape",
)
BALANCE_FIELDS = ("gain_s_inv", "loss_s_inv", "net_s_inv")

_RAW_SOURCE_RELATIVES = frozenset(
    {
        "qpsim/collisions/phonon.py",
        "qpsim/constants.py",
        "qpsim/grid/energy_grid.py",
        "qpsim/observables/gap_suppression.py",
        "qpsim/physics/bcs_quadrature.py",
        "qpsim/physics/gap_equation.py",
        "qpsim/physics/spectral.py",
        "validation/source_provenance.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/fischer_2023/fig6_author_c0_summary.py",
        "validation/fischer_2023/fig6_author_c2_bundle.py",
        "validation/fischer_2023/fig6_author_c2_parameters.py",
        "validation/fischer_2023/fig6_author_c2_score.py",
        "validation/fischer_2023/fig6_author_c3_bundle.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    }
)
_SOURCE_PATHS = (
    Path(__file__).resolve(),
    *(REPOSITORY_ROOT / relative for relative in sorted(_RAW_SOURCE_RELATIVES)),
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}

_RAW_METADATA_KEYS = {
    "array_descriptors",
    "coordinate_contract",
    "frozen_inputs",
    "limitations",
    "native_qpsim_grid_parameters",
    "observable_control",
    "parameters",
    "parent_bindings",
    "projection",
    "runtime",
    "schema",
    "source_binding",
    "sources",
    "stage",
    "stages",
}
_SCORE_KEYS = {
    "acceptance",
    "comparison",
    "limitations",
    "observable_control",
    "parent_bindings",
    "projection",
    "raw_bundle",
    "schema",
    "sources",
    "stage",
    "stages",
    "structural_identity",
}
_ACCEPTANCE_LIMITS = {
    "density_max_symmetric_relative": 2.0e-7,
    "net_subtraction_roundoff_factor": 64.0,
    "number_conserving_channel_max_symmetric_relative": 1.0e-12,
    "observable_integral_max_absolute_error": 4.0e-18,
    "raw_array_max_absolute_error": 0.0,
}
_ACCEPTANCE_CHECK_KEYS = {
    "all_gain_loss_arrays_nonnegative",
    "all_stage_guard_padding_is_positive_zero",
    "c2_parent_chain_bound",
    "c3p_active_outputs_bit_exact_to_c2b5",
    "density_agrees_with_author_within_limit",
    "frozen_parent_inputs_bit_exact",
    "grid_geometry_independently_derived",
    "locality_checks_pass",
    "native_center_carrier_effect_nonzero",
    "net_subtraction_within_roundoff",
    "number_conserving_channels_close",
    "number_diagnostics_finite",
    "observable_projection_within_limit",
    "phonon_support_projection_exact",
    "projection_is_copy_without_interpolation",
    "raw_arrays_independently_recomputed",
    "raw_metadata_independently_checked",
    "residual_closure_bit_exact",
    "sample_carrier_shift_explicit",
    "source_closure_bound",
}
_ARRAY_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")


class C3ScoreError(ValueError):
    """The C3 raw evidence, parent chain, score, or receipt is malformed."""


@dataclass(frozen=True)
class _Parameters:
    gap_eV: float
    h_eV: float
    temperature_K: float
    T_c_K: float
    tau_0_s: float
    tau_0_pb_s: float
    tau_l_s: float
    photon_bin: int
    n_bar: float
    c_photon_s_inv: float
    delta0_eV: float
    thermal_gap_eV: float
    max_newton_steps: int
    relative_step_threshold: float
    boltzmann_constant_J_per_K: float
    electron_charge_C: float


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C3ScoreError(f"C3 score source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C3ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C3ScoreError(f"Non-finite JSON constant {token!r} is forbidden.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C3ScoreError(f"Cannot parse {label}: {exc}.") from exc
    if not isinstance(value, dict):
        raise C3ScoreError(f"{label} must be an object.")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C3ScoreError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise C3ScoreError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(value)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C3ScoreError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, int):
        raise C3ScoreError(f"{label} must be an integer.")
    if minimum is not None and value < minimum:
        raise C3ScoreError(f"{label} must be at least {minimum}.")
    return value


def _finite_scalar(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise C3ScoreError(f"{label} must be a finite real scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C3ScoreError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise C3ScoreError(f"{label} is outside its finite scalar contract.")
    return result


def _file_sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise C3ScoreError(f"{label} is missing, unsafe, or a symlink.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repository_file_snapshot(path: Path, label: str) -> tuple[Path, bytes]:
    """Capture one safe repository-contained trust anchor before validation."""

    if path.is_symlink() or not path.is_file():
        raise C3ScoreError(f"{label} is missing, unsafe, or a symlink.")
    resolved = path.resolve()
    try:
        resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as exc:
        raise C3ScoreError(f"{label} must stay inside the repository.") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise C3ScoreError(f"{label} is missing, unsafe, or a symlink.")
    return resolved, resolved.read_bytes()


def _read_regular_file_once(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise C3ScoreError(f"{label} is missing, unsafe, or a symlink.")
    return path.read_bytes()


def _canonical_json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _json_value_bit_exact(reference: object, candidate: object) -> bool:
    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, float):
        return (
            np.asarray(reference, dtype=np.float64).tobytes()
            == np.asarray(
                candidate,
                dtype=np.float64,
            ).tobytes()
        )
    if isinstance(reference, dict):
        candidate_dict = candidate
        assert isinstance(candidate_dict, dict)
        return set(reference) == set(candidate_dict) and all(
            _json_value_bit_exact(reference[key], candidate_dict[key]) for key in reference
        )
    if isinstance(reference, list):
        candidate_list = candidate
        assert isinstance(candidate_list, list)
        return len(reference) == len(candidate_list) and all(
            _json_value_bit_exact(left, right)
            for left, right in zip(reference, candidate_list, strict=True)
        )
    return reference == candidate


def _array_bit_exact(reference: np.ndarray, candidate: np.ndarray) -> bool:
    return _npy_bytes(np.asarray(reference)) == _npy_bytes(np.asarray(candidate))


def _positive_zero(value: np.ndarray) -> bool:
    array = np.asarray(value)
    if array.dtype.kind != "f":
        return bool(np.all(array == 0))
    return bool(np.all(array == 0.0) and not np.any(np.signbit(array)))


def _expected_array_names() -> set[str]:
    base = {
        "legacy_phonon_support_mask",
        "mapped_left_edge_delta_ueV",
        "native_K_minus_full",
        "native_K_plus_full",
        "native_active_mask",
        "native_cell_anomalous_density_full",
        "native_cell_density_full",
        "native_cell_edges_ueV",
        "native_cell_weights_full",
        "native_dE_ueV",
        "native_E_centers_ueV",
        "native_omega_ueV",
        "parent_E_left_eV",
        "parent_K_minus_active",
        "parent_K_plus_active",
        "parent_f",
        "parent_n_phonon",
        "parent_rho_active",
        "parent_thermal_f",
        "parent_to_native_index",
        "parent_phonon_to_native_omega_index",
        "projected_f",
        "projected_n_phonon",
        "projected_thermal_f",
        "sample_carrier_delta_ueV",
    }
    for stage_id in STAGE_IDS:
        for channel in CHANNEL_NAMES:
            for field in BALANCE_FIELDS:
                base.add(f"{stage_id}__{channel}__{field}")
        base.add(f"{stage_id}__qp_residual_s_inv")
        base.add(f"{stage_id}__phonon_residual_s_inv")
    return base


def load_c3_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load one canonical external C3 raw bundle.

    The loader rejects duplicate/non-canonical JSON, unlisted files,
    subdirectories, symlinks, unsafe names, non-v3 NPY encodings, trailing
    bytes, object/complex/string arrays, descriptor drift, byte-order drift,
    and signed-zero rewriting.  Re-serializing every array is intentionally
    part of the transport check.
    """

    root = bundle_dir.resolve()
    if bundle_dir.is_symlink() or not root.is_dir():
        raise C3ScoreError("C3 raw bundle root is missing, unsafe, or a symlink.")
    manifest_path = root / "manifest.json"
    manifest_raw = _read_regular_file_once(manifest_path, "C3 raw manifest")
    manifest = _parse_json(manifest_raw, "C3 raw manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C3 raw manifest")
    if manifest.get("schema") != RAW_SCHEMA:
        raise C3ScoreError("C3 raw manifest schema is unsupported.")
    if manifest_raw != _canonical_json_bytes(manifest):
        raise C3ScoreError("C3 raw manifest is not canonical JSON.")

    files = _mapping(manifest.get("files"), "C3 raw manifest files")
    metadata = _mapping(manifest.get("metadata"), "C3 raw metadata")
    _exact_keys(metadata, _RAW_METADATA_KEYS, "C3 raw metadata")
    if metadata.get("schema") != RAW_SCHEMA:
        raise C3ScoreError("C3 raw metadata schema is unsupported.")
    expected_names = _expected_array_names()
    expected_filenames = {f"{name}.npy" for name in expected_names}
    if set(files) != expected_filenames or len(files) != 105:
        raise C3ScoreError("C3 raw file closure is invalid.")

    entries = list(root.iterdir())
    if any(entry.is_symlink() for entry in entries):
        raise C3ScoreError("C3 raw bundle contains a symlink.")
    actual_names = {entry.name for entry in entries}
    if actual_names != expected_filenames | {"manifest.json"}:
        raise C3ScoreError("C3 raw directory closure is invalid.")
    if any(not entry.is_file() for entry in entries):
        raise C3ScoreError("C3 raw bundle contains a non-file entry.")

    arrays: dict[str, np.ndarray] = {}
    for filename in sorted(expected_filenames):
        if Path(filename).name != filename or not filename.endswith(".npy"):
            raise C3ScoreError(f"Unsafe C3 raw filename {filename!r}.")
        name = filename[:-4]
        if _ARRAY_NAME_RE.fullmatch(name) is None:
            raise C3ScoreError(f"Unsafe C3 raw array name {name!r}.")
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        expected_sha = _sha256(record.get("sha256"), f"files.{filename}.sha256")
        expected_size = _strict_int(
            record.get("size_bytes"),
            f"files.{filename}.size_bytes",
            minimum=1,
        )
        content = _read_regular_file_once(root / filename, f"C3 raw {filename}")
        if len(content) != expected_size or hashlib.sha256(content).hexdigest() != expected_sha:
            raise C3ScoreError(f"C3 raw file {filename!r} failed its manifest binding.")
        if len(content) < 8 or content[:6] != b"\x93NUMPY" or content[6:8] != b"\x03\x00":
            raise C3ScoreError(f"C3 raw file {filename!r} is not canonical NPY v3.")
        try:
            stream = io.BytesIO(content)
            loaded = np.lib.format.read_array(stream, allow_pickle=False)
        except (ValueError, TypeError, EOFError) as exc:
            raise C3ScoreError(f"Cannot load C3 raw array {filename!r}.") from exc
        if stream.tell() != len(content):
            raise C3ScoreError(f"C3 raw file {filename!r} contains trailing bytes.")
        array = np.asarray(loaded)
        if array.dtype.kind not in {"b", "i", "u", "f"} or np.iscomplexobj(array):
            raise C3ScoreError(
                f"C3 raw array {filename!r} has a forbidden dtype {array.dtype.str!r}."
            )
        if array.dtype.kind == "f" and np.any(~np.isfinite(array)):
            raise C3ScoreError(f"C3 raw array {filename!r} contains non-finite values.")
        if _npy_bytes(array) != content:
            raise C3ScoreError(
                f"C3 raw file {filename!r} is not a canonical byte-exact NPY v3 encoding."
            )
        arrays[name] = array

    descriptors = _mapping(
        metadata.get("array_descriptors"),
        "C3 raw array descriptors",
    )
    expected_descriptors = {
        name: _array_descriptor(array) for name, array in sorted(arrays.items())
    }
    if not _json_value_bit_exact(descriptors, expected_descriptors):
        raise C3ScoreError("C3 raw array descriptors are incomplete, forged, or stale.")
    return metadata, arrays, hashlib.sha256(manifest_raw).hexdigest()


def _parameters_from_metadata(metadata: dict[str, Any]) -> _Parameters:
    raw = _mapping(metadata.get("parameters"), "C3 parameters")
    _exact_keys(raw, {"hex", "values"}, "C3 parameters")
    values = _mapping(raw.get("values"), "C3 parameter values")
    expected = {
        "T_c_K",
        "boltzmann_constant_J_per_K",
        "c_photon_s_inv",
        "delta0_eV",
        "electron_charge_C",
        "gap_eV",
        "h_eV",
        "max_newton_steps",
        "n_bar",
        "photon_bin",
        "relative_step_threshold",
        "tau_0_pb_s",
        "tau_0_s",
        "tau_l_s",
        "temperature_K",
        "thermal_gap_eV",
    }
    _exact_keys(values, expected, "C3 parameter values")
    floats: dict[str, float] = {}
    for key in expected - {"max_newton_steps", "photon_bin"}:
        floats[key] = _finite_scalar(
            values.get(key),
            f"C3 parameter {key}",
            positive=key not in {"c_photon_s_inv", "n_bar"},
            nonnegative=key in {"c_photon_s_inv", "n_bar"},
        )
    photon_bin = _strict_int(values.get("photon_bin"), "C3 photon_bin", minimum=1)
    max_steps = _strict_int(
        values.get("max_newton_steps"),
        "C3 max_newton_steps",
        minimum=1,
    )
    hex_record = _mapping(raw.get("hex"), "C3 parameter hex record")
    expected_hex = {key: value.hex() for key, value in floats.items()}
    if hex_record != expected_hex:
        raise C3ScoreError("C3 parameter hexadecimal closure is invalid.")
    return _Parameters(
        gap_eV=floats["gap_eV"],
        h_eV=floats["h_eV"],
        temperature_K=floats["temperature_K"],
        T_c_K=floats["T_c_K"],
        tau_0_s=floats["tau_0_s"],
        tau_0_pb_s=floats["tau_0_pb_s"],
        tau_l_s=floats["tau_l_s"],
        photon_bin=photon_bin,
        n_bar=floats["n_bar"],
        c_photon_s_inv=floats["c_photon_s_inv"],
        delta0_eV=floats["delta0_eV"],
        thermal_gap_eV=floats["thermal_gap_eV"],
        max_newton_steps=max_steps,
        relative_step_threshold=floats["relative_step_threshold"],
        boltzmann_constant_J_per_K=floats["boltzmann_constant_J_per_K"],
        electron_charge_C=floats["electron_charge_C"],
    )


def _parameter_values(parameters: _Parameters) -> dict[str, object]:
    return {
        "T_c_K": parameters.T_c_K,
        "boltzmann_constant_J_per_K": parameters.boltzmann_constant_J_per_K,
        "c_photon_s_inv": parameters.c_photon_s_inv,
        "delta0_eV": parameters.delta0_eV,
        "electron_charge_C": parameters.electron_charge_C,
        "gap_eV": parameters.gap_eV,
        "h_eV": parameters.h_eV,
        "max_newton_steps": parameters.max_newton_steps,
        "n_bar": parameters.n_bar,
        "photon_bin": parameters.photon_bin,
        "relative_step_threshold": parameters.relative_step_threshold,
        "tau_0_pb_s": parameters.tau_0_pb_s,
        "tau_0_s": parameters.tau_0_s,
        "tau_l_s": parameters.tau_l_s,
        "temperature_K": parameters.temperature_K,
        "thermal_gap_eV": parameters.thermal_gap_eV,
    }


def _derive_grid(
    parameters: _Parameters,
) -> dict[str, np.ndarray]:
    """Derive the live C3 geometry without calling qpsim grid/spectral code."""

    # The live Figure 6 grid is built from the module's declared native
    # ``DELTA_0 = 180.0`` micro-eV literal, not by round-tripping the
    # effective author-unit eV carrier through ``*1e6``.
    gap_ueV = 180.0
    delta0_ueV = 180.0
    omega0_ueV = delta0_ueV / 9.0
    energy_min_factor = (delta0_ueV - omega0_ueV) / delta0_ueV
    energy_max_factor = 10.0
    lower = energy_min_factor * delta0_ueV
    upper = energy_max_factor * delta0_ueV
    count = 1640
    dE_scalar = (upper - lower) / float(count)
    E = lower + (np.arange(count, dtype=float) + 0.5) * dE_scalar
    dE = np.empty(count, dtype=float)
    edges_from_centers = np.empty(count + 1, dtype=float)
    edges_from_centers[1:-1] = 0.5 * (E[:-1] + E[1:])
    edges_from_centers[0] = E[0] - 0.5 * (E[1] - E[0])
    edges_from_centers[-1] = E[-1] + 0.5 * (E[-1] - E[-2])
    dE[:] = np.diff(edges_from_centers)

    edges = np.empty(count + 1, dtype=float)
    edges[0] = E[0] - 0.5 * dE[0]
    edges[1:] = edges[0] + np.cumsum(dE)
    lo = np.maximum(edges[:-1], gap_ueV)
    hi = np.minimum(edges[1:], edges[-1])
    hi = np.maximum(hi, lo)
    xi_lo = np.sqrt(np.maximum((lo - gap_ueV) * (lo + gap_ueV), 0.0))
    xi_hi = np.sqrt(np.maximum((hi - gap_ueV) * (hi + gap_ueV), 0.0))
    weights = xi_hi - xi_lo
    density = weights / dE

    anomalous_weight = gap_ueV * (
        np.arccosh(np.maximum(hi / gap_ueV, 1.0)) - np.arccosh(np.maximum(lo / gap_ueV, 1.0))
    )
    anomalous_density = anomalous_weight / dE
    ratio = np.zeros_like(E)
    active = weights > 0.0
    ratio[active] = anomalous_weight[active] / weights[active]
    ratio = np.clip(ratio, 0.0, 1.0)
    product = ratio[:, None] * ratio[None, :]
    K_plus = 1.0 + product
    K_minus = np.maximum(1.0 - product, 0.0)
    omega = np.arange(3600, dtype=float)
    return {
        "native_E_centers_ueV": E,
        "native_dE_ueV": dE,
        "native_cell_edges_ueV": edges,
        "native_cell_weights_full": weights,
        "native_cell_density_full": density,
        "native_cell_anomalous_density_full": anomalous_density,
        "native_active_mask": active,
        "native_K_plus_full": K_plus,
        "native_K_minus_full": K_minus,
        "native_omega_ueV": omega,
    }


def _author_coefficients(
    E_left_eV: np.ndarray,
    parameters: _Parameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E = np.asarray(E_left_eV, dtype=float)
    gap = parameters.gap_eV
    h = parameters.h_eV
    lower = np.sqrt(np.maximum(E**2 - gap**2, 0.0))
    upper = np.sqrt((E + h) ** 2 - gap**2)
    rho = (upper - lower) / h
    product = gap**2 / (E[:, None] * E[None, :])
    return rho, 1.0 + product, np.maximum(1.0 - product, 0.0)


def _author_thermal_phonons(
    size: int,
    parameters: _Parameters,
) -> np.ndarray:
    omega = parameters.h_eV * np.arange(1, size, dtype=float)
    exponent = (
        omega
        * parameters.electron_charge_C
        / (parameters.boltzmann_constant_J_per_K * parameters.temperature_K)
    )
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        return 1.0 / np.expm1(exponent)


def _evaluate_frozen(
    *,
    f: np.ndarray,
    n_phonon: np.ndarray,
    rho: np.ndarray,
    K_plus: np.ndarray,
    K_minus: np.ndarray,
    pair_frequency_offset_bins: int,
    parameters: _Parameters,
) -> dict[str, dict[str, np.ndarray] | np.ndarray]:
    """Independently transcribe the six author balances in source order."""

    f_value = np.asarray(f, dtype=float)
    n_value = np.asarray(n_phonon, dtype=float)
    size = f_value.size
    phonon_size = size - 1
    photon_step = parameters.photon_bin
    n_bar = parameters.n_bar
    c_photon = parameters.c_photon_s_inv
    a_delta = round(parameters.gap_eV / parameters.h_eV)
    qp_prefactor = (
        parameters.h_eV**3
        * parameters.electron_charge_C**3
        / (parameters.tau_0_s * (parameters.boltzmann_constant_J_per_K * parameters.T_c_K) ** 3)
    )
    phonon_prefactor = 1.0 / (np.pi * parameters.gap_eV * parameters.tau_0_pb_s)

    qp_photon_gain = np.zeros(size)
    qp_photon_loss = np.zeros(size)
    qp_photon_net = np.zeros(size)
    upper_i = np.arange(0, size - 1 - photon_step)
    upper_j = upper_i + photon_step
    U_upper = K_plus[upper_i, upper_j] * rho[upper_j]
    qp_photon_gain[upper_i] += U_upper * f_value[upper_j] * (1.0 - f_value[upper_i]) * (1.0 + n_bar)
    qp_photon_loss[upper_i] += U_upper * f_value[upper_i] * (1.0 - f_value[upper_j]) * n_bar
    qp_photon_net[upper_i] += U_upper * (
        f_value[upper_j] * (1.0 - f_value[upper_i]) * (1.0 + n_bar)
        - f_value[upper_i] * (1.0 - f_value[upper_j]) * n_bar
    )
    lower_i = np.arange(photon_step, size - 1)
    lower_j = lower_i - photon_step
    U_lower = K_plus[lower_i, lower_j] * rho[lower_j]
    qp_photon_gain[lower_i] += U_lower * f_value[lower_j] * (1.0 - f_value[lower_i]) * n_bar
    qp_photon_loss[lower_i] += U_lower * f_value[lower_i] * (1.0 - f_value[lower_j]) * (1.0 + n_bar)
    qp_photon_net[lower_i] += U_lower * (
        f_value[lower_j] * (1.0 - f_value[lower_i]) * n_bar
        - f_value[lower_i] * (1.0 - f_value[lower_j]) * (1.0 + n_bar)
    )
    qp_photon_gain *= c_photon
    qp_photon_loss *= c_photon
    qp_photon_net *= c_photon

    qp_scattering_gain = np.zeros(size)
    qp_scattering_loss = np.zeros(size)
    qp_scattering_net = np.zeros(size)
    qp_pair_gain = np.zeros(size)
    qp_pair_loss = np.zeros(size)
    qp_pair_net = np.zeros(size)
    phonon_scattering_gain = np.zeros(phonon_size)
    phonon_scattering_loss = np.zeros(phonon_size)
    phonon_scattering_net = np.zeros(phonon_size)
    phonon_pair_gain = np.zeros(phonon_size)
    phonon_pair_loss = np.zeros(phonon_size)
    phonon_pair_net = np.zeros(phonon_size)
    n_thermal = _author_thermal_phonons(size, parameters)
    phonon_escape_gain = n_thermal / parameters.tau_l_s
    phonon_escape_loss = n_value / parameters.tau_l_s
    phonon_escape_net = (n_thermal - n_value) / parameters.tau_l_s

    for transfer in range(1, size):
        low = np.arange(size - transfer)
        high = low + transfer
        n_transfer = n_value[transfer - 1]
        K_diag = K_minus[low, high]
        transfer_squared = float(transfer * transfer)
        qp_weight_low = qp_prefactor * transfer_squared * K_diag * rho[high]
        qp_weight_high = qp_prefactor * transfer_squared * K_diag * rho[low]
        qp_scattering_gain[low] += qp_weight_low * f_value[high] * (1.0 + n_transfer - f_value[low])
        qp_scattering_loss[low] += qp_weight_low * n_transfer * f_value[low]
        qp_scattering_gain[high] += qp_weight_high * f_value[low] * n_transfer
        qp_scattering_loss[high] += (
            qp_weight_high * (1.0 - f_value[low] + n_transfer) * f_value[high]
        )
        qp_scattering_net[low] += qp_weight_low * (
            -n_transfer * f_value[low] + f_value[high] * (1.0 + n_transfer - f_value[low])
        )
        qp_scattering_net[high] += qp_weight_high * (
            f_value[low] * n_transfer - (1.0 - f_value[low] + n_transfer) * f_value[high]
        )

        phonon_weight = 2.0 * phonon_prefactor * parameters.h_eV * rho[low] * rho[high] * K_diag
        phonon_scattering_gain[transfer - 1] = np.sum(
            phonon_weight * f_value[high] * (1.0 - f_value[low]) * (1.0 + n_transfer)
        )
        phonon_scattering_loss[transfer - 1] = np.sum(
            phonon_weight * f_value[low] * (1.0 - f_value[high]) * n_transfer
        )
        phonon_scattering_net[transfer - 1] = np.sum(
            phonon_weight
            * (
                f_value[high] * (1.0 - f_value[low]) * (1.0 + n_transfer)
                - f_value[low] * (1.0 - f_value[high]) * n_transfer
            )
        )

    partner = np.arange(size)
    for index in range(size):
        levels = index + partner + 2 * a_delta + pair_frequency_offset_bins
        represented = levels < size
        n_pair = np.zeros(size)
        n_pair[represented] = n_value[levels[represented] - 1]
        weight = qp_prefactor * levels.astype(float) ** 2 * K_plus[index] * rho
        qp_pair_gain[index] = np.sum(weight * (1.0 - f_value[index]) * (1.0 - f_value) * n_pair)
        qp_pair_loss[index] = np.sum(weight * f_value[index] * f_value * (1.0 + n_pair))
        qp_pair_net[index] = np.sum(
            weight * ((1.0 - f_value[index] - f_value) * n_pair - f_value[index] * f_value)
        )

    for level in range(1, size):
        pair_sum = level - 2 * a_delta - pair_frequency_offset_bins
        if pair_sum < 0:
            continue
        first = np.arange(pair_sum + 1)
        second = pair_sum - first
        n_level = n_value[level - 1]
        pair_weight = (
            phonon_prefactor * parameters.h_eV * rho[first] * rho[second] * K_plus[first, second]
        )
        phonon_pair_gain[level - 1] = np.sum(
            pair_weight * (1.0 + n_level) * f_value[first] * f_value[second]
        )
        phonon_pair_loss[level - 1] = np.sum(
            pair_weight * (1.0 - f_value[first]) * (1.0 - f_value[second]) * n_level
        )
        phonon_pair_net[level - 1] = np.sum(
            pair_weight
            * (
                (1.0 + n_level) * f_value[first] * f_value[second]
                - (1.0 - f_value[first]) * (1.0 - f_value[second]) * n_level
            )
        )

    channels: dict[str, dict[str, np.ndarray] | np.ndarray] = {
        "qp_photon": {
            "gain_s_inv": qp_photon_gain,
            "loss_s_inv": qp_photon_loss,
            "net_s_inv": qp_photon_net,
        },
        "qp_scattering": {
            "gain_s_inv": qp_scattering_gain,
            "loss_s_inv": qp_scattering_loss,
            "net_s_inv": qp_scattering_net,
        },
        "qp_pair": {
            "gain_s_inv": qp_pair_gain,
            "loss_s_inv": qp_pair_loss,
            "net_s_inv": qp_pair_net,
        },
        "phonon_scattering": {
            "gain_s_inv": phonon_scattering_gain,
            "loss_s_inv": phonon_scattering_loss,
            "net_s_inv": phonon_scattering_net,
        },
        "phonon_pair": {
            "gain_s_inv": phonon_pair_gain,
            "loss_s_inv": phonon_pair_loss,
            "net_s_inv": phonon_pair_net,
        },
        "phonon_escape": {
            "gain_s_inv": phonon_escape_gain,
            "loss_s_inv": phonon_escape_loss,
            "net_s_inv": phonon_escape_net,
        },
    }
    qp_residual = qp_photon_net + qp_scattering_net + qp_pair_net
    phonon_residual = phonon_scattering_net + phonon_pair_net + phonon_escape_net
    channels["qp_residual_s_inv"] = qp_residual
    channels["phonon_residual_s_inv"] = phonon_residual
    return channels


def _embed_active(value: np.ndarray, active_indices: np.ndarray) -> np.ndarray:
    source = np.asarray(value)
    result = np.zeros(1640, dtype=source.dtype)
    result[active_indices] = source
    return result


def _expected_stage_arrays(
    *,
    stage_id: str,
    evaluation: dict[str, dict[str, np.ndarray] | np.ndarray],
    active_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for channel in CHANNEL_NAMES:
        balance = evaluation[channel]
        assert isinstance(balance, dict)
        for field in BALANCE_FIELDS:
            value = np.asarray(balance[field])
            result[f"{stage_id}__{channel}__{field}"] = (
                _embed_active(value, active_indices) if channel.startswith("qp_") else value.copy()
            )
    result[f"{stage_id}__qp_residual_s_inv"] = _embed_active(
        np.asarray(evaluation["qp_residual_s_inv"]),
        active_indices,
    )
    result[f"{stage_id}__phonon_residual_s_inv"] = np.asarray(
        evaluation["phonon_residual_s_inv"]
    ).copy()
    return result


def _direct_gap_integral(
    f: np.ndarray,
    centers: np.ndarray,
    *,
    gap: float,
    samples: str,
) -> float:
    """Local transcription of the direct author/center gap integral."""

    occupations = np.asarray(f, dtype=float)
    E = np.asarray(centers, dtype=float)
    edge_h = float(np.mean(np.diff(E)))
    h = float(E[1] - E[0])
    edges = E - 0.5 * edge_h
    tolerance = 64.0 * np.finfo(float).eps * max(gap, h, 1.0)
    grid_lo = float(edges[0])
    if grid_lo > gap and grid_lo <= gap + tolerance:
        edges = edges - (grid_lo - gap)
    if samples == "authors":
        values = np.maximum(occupations, 0.0)
    elif samples == "centers":
        active = gap < edges + h
        if np.count_nonzero(active) < 2:
            raise C3ScoreError("C3 native-center observable has insufficient positive capacity.")
        E_active = E[active]
        f_active = occupations[active]
        values = np.interp(edges, E_active, f_active)
        slope_left = (f_active[1] - f_active[0]) / (E_active[1] - E_active[0])
        below_first_active = edges < E_active[0]
        values[below_first_active] = f_active[0] + slope_left * (
            edges[below_first_active] - E_active[0]
        )
        values = np.maximum(values, 0.0)
    else:  # pragma: no cover - internal enum.
        raise AssertionError(f"Unsupported C3 sample mode {samples!r}.")
    cell_x_lo = edges - gap
    x_lo = np.maximum(cell_x_lo, 0.0)
    x_hi = np.maximum(cell_x_lo + h, 0.0)
    a_hi = np.arcsinh(np.sqrt(x_hi / (2.0 * gap)))
    a_lo = np.arcsinh(np.sqrt(x_lo / (2.0 * gap)))
    constant = float(np.sum(4.0 * values * (a_hi - a_lo)))
    x0 = x_lo[:-1]
    x1 = x_hi[:-1]
    da = a_hi[:-1] - a_lo[:-1]
    linear_weight = (
        np.sqrt(x1 * (x1 + 2.0 * gap))
        - np.sqrt(x0 * (x0 + 2.0 * gap))
        - 2.0 * (cell_x_lo[:-1] + gap) * da
    )
    linear = float(np.sum(2.0 * (values[1:] - values[:-1]) / h * linear_weight))
    return constant + linear


def _observable_control(
    arrays: dict[str, np.ndarray],
    parameters: _Parameters,
) -> dict[str, object]:
    parent_centers = arrays["parent_E_left_eV"] + 0.5 * parameters.h_eV
    child_centers = arrays["native_E_centers_ueV"]
    native_gap_ueV = 180.0
    native_delta0_ueV = 180.0
    parent_driven = _direct_gap_integral(
        arrays["parent_f"],
        parent_centers,
        gap=parameters.gap_eV,
        samples="authors",
    )
    parent_thermal = _direct_gap_integral(
        arrays["parent_thermal_f"],
        parent_centers,
        gap=parameters.gap_eV,
        samples="authors",
    )
    reembedded_driven = _direct_gap_integral(
        arrays["projected_f"],
        child_centers,
        gap=native_gap_ueV,
        samples="authors",
    )
    reembedded_thermal = _direct_gap_integral(
        arrays["projected_thermal_f"],
        child_centers,
        gap=native_gap_ueV,
        samples="authors",
    )
    native_driven = _direct_gap_integral(
        arrays["projected_f"],
        child_centers,
        gap=native_gap_ueV,
        samples="centers",
    )
    native_thermal = _direct_gap_integral(
        arrays["projected_thermal_f"],
        child_centers,
        gap=native_gap_ueV,
        samples="centers",
    )
    parent_driven_gap = parameters.delta0_eV * float(np.exp(-parent_driven))
    parent_thermal_gap = parameters.delta0_eV * float(np.exp(-parent_thermal))
    reembedded_driven_gap = native_delta0_ueV * float(np.exp(-reembedded_driven))
    reembedded_thermal_gap = native_delta0_ueV * float(np.exp(-reembedded_thermal))
    native_driven_gap = native_delta0_ueV * float(np.exp(-native_driven))
    native_thermal_gap = native_delta0_ueV * float(np.exp(-native_thermal))

    def suppression_ratio(driven: float, thermal: float) -> float:
        denominator = -np.expm1(-thermal)
        numerator = np.exp(-thermal) * np.expm1(thermal - driven)
        return float(numerator / denominator)

    def diagnostic(
        *,
        driven_integral: float,
        thermal_integral: float,
        driven_gap_ueV: float,
        thermal_gap_ueV: float,
        interpretation: str,
    ) -> dict[str, object]:
        return {
            "child_full_grid": {
                "driven_gap_ueV": driven_gap_ueV,
                "driven_integral": driven_integral,
                "frozen_suppression_ratio": suppression_ratio(
                    driven_integral,
                    thermal_integral,
                ),
                "thermal_gap_ueV": thermal_gap_ueV,
                "thermal_integral": thermal_integral,
            },
            "differences_from_parent": {
                "driven_gap_eV_equivalent_signed": (driven_gap_ueV * 1.0e-6 - parent_driven_gap),
                "driven_integral_relative_signed": (driven_integral / parent_driven - 1.0),
                "driven_integral_signed": (driven_integral - parent_driven),
                "thermal_gap_eV_equivalent_signed": (thermal_gap_ueV * 1.0e-6 - parent_thermal_gap),
                "thermal_integral_relative_signed": (thermal_integral / parent_thermal - 1.0),
                "thermal_integral_signed": (thermal_integral - parent_thermal),
            },
            "interpretation": interpretation,
        }

    return {
        "claim": (
            "Two frozen projection diagnostics are reported: exact ordinal "
            "re-embedding under retained author left-edge semantics, and the "
            "actual qpsim center-carrier reinterpretation. Neither is a C3 "
            "root or plotted ordinate."
        ),
        "author_semantics_reembedding": diagnostic(
            driven_integral=reembedded_driven,
            thermal_integral=reembedded_thermal,
            driven_gap_ueV=reembedded_driven_gap,
            thermal_gap_ueV=reembedded_thermal_gap,
            interpretation=(
                "Projected values deliberately re-read as author left-edge "
                "samples; this is the projection-identity control."
            ),
        ),
        "native_center_carrier": diagnostic(
            driven_integral=native_driven,
            thermal_integral=native_thermal,
            driven_gap_ueV=native_driven_gap,
            thermal_gap_ueV=native_thermal_gap,
            interpretation=(
                "Projected values interpreted at their declared qpsim cell "
                "centers; this reports the half-bin carrier effect."
            ),
        ),
        "parent_author_left_edge": {
            "driven_gap_eV": parent_driven_gap,
            "driven_integral": parent_driven,
            "frozen_suppression_ratio": suppression_ratio(
                parent_driven,
                parent_thermal,
            ),
            "thermal_gap_eV": parent_thermal_gap,
            "thermal_integral": parent_thermal,
        },
    }


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    left = np.asarray(reference)
    right = np.asarray(candidate)
    delta = right - left
    absolute = np.abs(delta)
    denominator = max(
        float(np.sum(np.abs(left))) + float(np.sum(np.abs(right))),
        np.finfo(float).tiny,
    )
    return {
        "bit_exact": _array_bit_exact(left, right),
        "candidate_descriptor": _array_descriptor(right),
        "l1_absolute": float(np.sum(absolute)),
        "linf_absolute": float(np.max(absolute, initial=0.0)),
        "symmetric_relative_l1": float(np.sum(absolute)) / denominator,
    }


def _channel_differences(
    arrays: dict[str, np.ndarray],
    *,
    reference_stage: str,
    candidate_stage: str,
) -> dict[str, object]:
    return {
        channel: {
            field: _difference(
                arrays[f"{reference_stage}__{channel}__{field}"],
                arrays[f"{candidate_stage}__{channel}__{field}"],
            )
            for field in BALANCE_FIELDS
        }
        for channel in CHANNEL_NAMES
    }


def _channel_changed(
    arrays: dict[str, np.ndarray],
    reference_stage: str,
    candidate_stage: str,
    channel: str,
) -> bool:
    return any(
        not _array_bit_exact(
            arrays[f"{reference_stage}__{channel}__{field}"],
            arrays[f"{candidate_stage}__{channel}__{field}"],
        )
        for field in BALANCE_FIELDS
    )


def _net_roundoff_ok(
    arrays: dict[str, np.ndarray],
) -> tuple[bool, float]:
    factor = _ACCEPTANCE_LIMITS["net_subtraction_roundoff_factor"]
    worst_ratio = 0.0
    for stage_id in STAGE_IDS:
        for channel in CHANNEL_NAMES:
            gain = arrays[f"{stage_id}__{channel}__gain_s_inv"]
            loss = arrays[f"{stage_id}__{channel}__loss_s_inv"]
            net = arrays[f"{stage_id}__{channel}__net_s_inv"]
            error = np.abs(net - (gain - loss))
            scale = factor * np.finfo(float).eps * (np.abs(gain) + np.abs(loss))
            if np.any(error > scale):
                return False, float("inf")
            nonzero = scale > 0.0
            if np.any(nonzero):
                worst_ratio = max(
                    worst_ratio,
                    float(np.max(error[nonzero] / scale[nonzero])),
                )
    return True, worst_ratio


def _number_diagnostics(
    arrays: dict[str, np.ndarray],
    *,
    stage_id: str,
    rho_active: np.ndarray,
    parameters: _Parameters,
) -> dict[str, object]:
    active = arrays["parent_to_native_index"]
    weights = parameters.h_eV * np.asarray(rho_active)
    channels: dict[str, object] = {}
    for channel in ("qp_photon", "qp_scattering", "qp_pair"):
        gain = arrays[f"{stage_id}__{channel}__gain_s_inv"][active]
        loss = arrays[f"{stage_id}__{channel}__loss_s_inv"][active]
        net = arrays[f"{stage_id}__{channel}__net_s_inv"][active]
        weighted = float(weights @ net)
        turnover = float(np.sum(weights * (np.abs(gain) + np.abs(loss))))
        channels[channel] = {
            "symmetric_turnover_relative": abs(weighted)
            / max(
                turnover,
                np.finfo(float).tiny,
            ),
            "weighted_number_rate_eV_s_inv": weighted,
            "weighted_turnover_eV_s_inv": turnover,
        }
    residual = arrays[f"{stage_id}__qp_residual_s_inv"][active]
    return {
        "channels": channels,
        "total_weighted_number_rate_eV_s_inv": float(weights @ residual),
    }


def _runtime_record_valid(value: object) -> bool:
    runtime = _mapping(value, "C3 raw runtime")
    expected = {
        "byteorder",
        "implementation",
        "machine",
        "numpy_version",
        "platform",
        "python_version",
    }
    _exact_keys(runtime, expected, "C3 raw runtime")
    return all(isinstance(runtime[key], str) and runtime[key] for key in expected)


def _stage_array_names(stage_id: str) -> dict[str, object]:
    return {
        "channels": {
            channel: {field: f"{stage_id}__{channel}__{field}" for field in BALANCE_FIELDS}
            for channel in CHANNEL_NAMES
        },
        "phonon_residual_s_inv": f"{stage_id}__phonon_residual_s_inv",
        "qp_residual_s_inv": f"{stage_id}__qp_residual_s_inv",
    }


def _expected_stage_metadata(
    parameters: _Parameters,
) -> list[dict[str, object]]:
    qp_prefactor = (
        parameters.h_eV**3
        * parameters.electron_charge_C**3
        / (parameters.tau_0_s * (parameters.boltzmann_constant_J_per_K * parameters.T_c_K) ** 3)
    )
    phonon_prefactor = 1.0 / (np.pi * parameters.gap_eV * parameters.tau_0_pb_s)
    definitions = (
        (
            STAGE_IDS[0],
            PARENT_STAGE_ID,
            "author_left_edge",
            "author_cell_average_eV",
            0,
            "full-domain ordinal embedding only; exact C2b5 active operator",
        ),
        (
            STAGE_IDS[1],
            STAGE_IDS[0],
            "qpsim_finite_volume",
            "author_cell_average_eV",
            0,
            (
                "author left-edge K_plus/K_minus -> live full-grid "
                "SpectralContext finite-volume K_plus/K_minus"
            ),
        ),
        (
            STAGE_IDS[2],
            STAGE_IDS[1],
            "qpsim_finite_volume",
            "author_cell_average_eV",
            1,
            "pair labels 2*Delta+(i+j)h -> 2*Delta+(i+j+1)h",
        ),
        (
            STAGE_IDS[3],
            STAGE_IDS[2],
            "qpsim_finite_volume",
            "qpsim_cell_density_ueV",
            1,
            (
                "author-eV cell-average DOS arithmetic -> the same live "
                "SpectralContext native-micro-eV cell_density"
            ),
        ),
    )
    return [
        {
            "array_names": _stage_array_names(stage_id),
            "changed_convention": changed,
            "coherence_convention": coherence,
            "density_convention": density,
            "operator_scalars": {
                "a_delta": round(parameters.gap_eV / parameters.h_eV),
                "phonon_prefactor_per_eV_s": phonon_prefactor,
                "qp_prefactor_s_inv": qp_prefactor,
            },
            "pair_frequency_offset_bins": offset,
            "parent_stage_id": parent,
            "stage_id": stage_id,
        }
        for stage_id, parent, coherence, density, offset, changed in definitions
    ]


def _check_raw_metadata(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    parameters: _Parameters,
    c2_metadata: dict[str, Any],
    c2_arrays: dict[str, np.ndarray],
    c2_manifest_sha: str,
    c2_score_path: Path,
    c2_receipt_path: Path,
    c2_score_sha256: str,
    c2_receipt_sha256: str,
) -> dict[str, bool]:
    if metadata.get("schema") != RAW_SCHEMA:
        raise C3ScoreError("C3 raw metadata schema is unsupported.")
    expected_sources = {
        relative: source_sha256(REPOSITORY_ROOT / relative)
        for relative in sorted(_RAW_SOURCE_RELATIVES)
    }
    source_ok = _json_value_bit_exact(metadata.get("sources"), expected_sources)
    if not source_ok:
        raise C3ScoreError("C3 raw source closure is forged, incomplete, or stale.")

    expected_parent = {
        "c2_raw_manifest_sha256": c2_manifest_sha,
        "c2_receipt_path": c2_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c2_receipt_sha256": c2_receipt_sha256,
        "c2_score_path": c2_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c2_score_sha256": c2_score_sha256,
        "c2b5_parent_residual": _array_descriptor(c2_arrays[f"{FINAL_C2_SLUG}__residual_s_inv"]),
        "c2b5_step_id": FINAL_C2_STEP_ID,
    }
    if not _json_value_bit_exact(metadata.get("parent_bindings"), expected_parent):
        raise C3ScoreError("C3 raw C2 parent binding is forged or stale.")

    final_steps = c2_metadata.get("steps")
    if not isinstance(final_steps, list) or not final_steps:
        raise C3ScoreError("Accepted C2 raw step closure is invalid.")
    final = _mapping(final_steps[-1], "C2b5 raw step")
    if final.get("step_id") != FINAL_C2_STEP_ID:
        raise C3ScoreError("Accepted C2 raw endpoint is not C2b5.")
    parameter_plan = _mapping(
        c2_metadata.get("parameter_plan"),
        "accepted C2 parameter plan",
    )
    c2b = _mapping(parameter_plan.get("c2b"), "accepted C2b parameter plan")
    carriers = _mapping(
        c2b.get("author_operator_carriers"),
        "accepted C2b author-operator carriers",
    )
    final_carrier = _mapping(
        carriers.get(FINAL_C2_STEP_ID),
        "accepted C2b5 author-operator carrier",
    )
    if not _json_value_bit_exact(metadata.get("parameters"), final_carrier):
        raise C3ScoreError(
            "C3 parameters are not the complete accepted C2b5 author-operator carrier."
        )
    if not _json_value_bit_exact(
        _mapping(final_carrier.get("values"), "C2b5 carrier values"),
        _parameter_values(parameters),
    ):
        raise C3ScoreError("C3 parsed parameters differ from the accepted C2b5 carrier.")

    frozen = _mapping(metadata.get("frozen_inputs"), "C3 raw frozen inputs")
    _exact_keys(
        frozen,
        {"descriptors", "mutation_check_after_all_stages", "policy"},
        "C3 raw frozen inputs",
    )
    frozen_descriptors = {
        "E_left_eV": _array_descriptor(c2_arrays["E_left_eV"]),
        "f_final": _array_descriptor(c2_arrays["f_final"]),
        "n_phonon_final": _array_descriptor(c2_arrays["n_phonon_final"]),
        "thermal_f": _array_descriptor(c2_arrays["thermal_f"]),
    }
    if (
        not _json_value_bit_exact(frozen.get("descriptors"), frozen_descriptors)
        or frozen.get("mutation_check_after_all_stages") is not True
        or frozen.get("policy")
        != (
            "C2b5 parent E_left, driven f, thermal f, and author-support "
            "phonon n are immutable; only an explicit ordinal grid embedding "
            "and frozen operator substitutions are evaluated"
        )
    ):
        raise C3ScoreError("C3 raw frozen-input metadata is invalid.")

    if metadata.get("stage") != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
    }:
        raise C3ScoreError("C3 raw stage identity is invalid.")
    if not _json_value_bit_exact(
        metadata.get("native_qpsim_grid_parameters"),
        {
            "delta0_ueV": 180.0,
            "delta0_ueV_hex": (180.0).hex(),
            "gap_ueV": 180.0,
            "gap_ueV_hex": (180.0).hex(),
            "uniform_dE_ueV": 1.0,
            "uniform_dE_ueV_hex": (1.0).hex(),
        },
    ):
        raise C3ScoreError("C3 raw native-grid parameter binding is invalid.")
    if metadata.get("coordinate_contract") != {
        "active_child_indices": "[20, 1640)",
        "author_phonon_support": "omega/h = 1..1619 only",
        "child_grid": (
            "live qpsim Figure 6 SpectralContext, 1640 centers on faces [160, 1800] micro-eV"
        ),
        "grid_projection": ("ordinal identity embedding parent i -> child i+20; no interpolation"),
        "inactive_padding": (
            "canonical positive zero in child QP cells 0..19; these cells "
            "have zero BCS capacity at the fixed 180 micro-eV gap"
        ),
        "native_omega_policy": (
            "full 0..3599 micro-eV lattice recorded; only legacy author "
            "support 1..1619 is evaluated in C3, and other projected "
            "values are non-solved serialization placeholders"
        ),
        "parent_grid": "1620 author left-edge cells [Delta, 10*Delta)",
        "sample_relabeling": (
            "parent cell values are carried unchanged onto qpsim cell "
            "centers; the approximately half-bin sample-carrier shift is "
            "recorded separately from roundoff in the mapped left cell faces"
        ),
    }:
        raise C3ScoreError("C3 raw coordinate contract is invalid.")
    if not _json_value_bit_exact(
        metadata.get("stages"),
        _expected_stage_metadata(parameters),
    ):
        raise C3ScoreError("C3 raw stage metadata failed independent closure.")
    _runtime_record_valid(metadata.get("runtime"))
    if metadata.get("source_binding") != {
        "hash_kind": "canonical_sha256_import_time_disk_snapshot",
        "scope": (
            "C3 producer, accepted C2 loaders/resolver, retained author "
            "operator, live Figure 6 grid/SpectralContext, phonon-map, "
            "and direct-observable sources"
        ),
    }:
        raise C3ScoreError("C3 raw source-binding policy is invalid.")
    limitations = _mapping(metadata.get("limitations"), "C3 raw limitations")
    _exact_keys(limitations, {"scope", "statement"}, "C3 raw limitations")
    if limitations.get(
        "scope"
    ) != "one authenticated C2 frozen point only" or "No C3 nonlinear root" not in str(
        limitations.get("statement")
    ):
        raise C3ScoreError("C3 raw limitation is invalid.")
    return {"parent": True, "sources": source_ok, "frozen": True, "stages": True}


def build_c3_score(
    c3_bundle_dir: Path,
    *,
    c2_bundle_dir: Path,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Recompute all formal frozen C3 arrays and return a checked score."""

    _assert_source_snapshots()
    c2_score_path, c2_score_bytes = _repository_file_snapshot(
        c2_score_path,
        "C2 score",
    )
    c2_receipt_path, c2_receipt_bytes = _repository_file_snapshot(
        c2_receipt_path,
        "C2 receipt",
    )
    accepted_c2 = load_c2_score(c2_score_path, receipt_path=c2_receipt_path)
    if (
        c2_score_path.is_symlink()
        or c2_receipt_path.is_symlink()
        or c2_score_path.read_bytes() != c2_score_bytes
        or c2_receipt_path.read_bytes() != c2_receipt_bytes
    ):
        raise C3ScoreError("C2 score or receipt changed during C3 verification.")
    c2_score_sha256 = hashlib.sha256(c2_score_bytes).hexdigest()
    c2_receipt_sha256 = hashlib.sha256(c2_receipt_bytes).hexdigest()
    c2_metadata, c2_arrays, c2_manifest_sha = load_c2_raw_bundle(c2_bundle_dir)
    c2_raw_binding = _mapping(accepted_c2.get("raw_bundle"), "accepted C2 raw binding")
    if (
        c2_raw_binding.get("schema") != C2_RAW_SCHEMA
        or c2_raw_binding.get("manifest_sha256") != c2_manifest_sha
    ):
        raise C3ScoreError("Selected external C2 raw bundle is not the accepted parent.")

    raw_metadata, raw_arrays, raw_manifest_sha = load_c3_raw_bundle(c3_bundle_dir)
    parameters = _parameters_from_metadata(raw_metadata)
    raw_checks = _check_raw_metadata(
        raw_metadata,
        raw_arrays,
        parameters=parameters,
        c2_metadata=c2_metadata,
        c2_arrays=c2_arrays,
        c2_manifest_sha=c2_manifest_sha,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
        c2_score_sha256=c2_score_sha256,
        c2_receipt_sha256=c2_receipt_sha256,
    )

    parent_bindings = _mapping(raw_metadata.get("parent_bindings"), "C3 raw parent")
    if parent_bindings.get("c2_raw_manifest_sha256") != c2_manifest_sha:
        raise C3ScoreError("C3 raw bundle is not parented by the selected C2 bundle.")
    for raw_name, parent_name in (
        ("parent_E_left_eV", "E_left_eV"),
        ("parent_f", "f_final"),
        ("parent_n_phonon", "n_phonon_final"),
        ("parent_thermal_f", "thermal_f"),
    ):
        if not _array_bit_exact(raw_arrays[raw_name], c2_arrays[parent_name]):
            raise C3ScoreError(f"C3 frozen parent array {raw_name!r} differs from C2.")

    expected_arrays: dict[str, np.ndarray] = {}
    grid = _derive_grid(parameters)
    expected_arrays.update(grid)
    parent_E = np.asarray(c2_arrays["E_left_eV"])
    parent_f = np.asarray(c2_arrays["f_final"])
    parent_n = np.asarray(c2_arrays["n_phonon_final"])
    parent_thermal = np.asarray(c2_arrays["thermal_f"])
    parent_rho, parent_K_plus, parent_K_minus = _author_coefficients(
        parent_E,
        parameters,
    )
    active_indices = np.arange(20, 1640, dtype=np.int64)
    parent_to_native = active_indices.copy()
    mapped_left_edge_delta = grid["native_cell_edges_ueV"][active_indices] - parent_E * 1.0e6
    sample_carrier_delta = grid["native_E_centers_ueV"][active_indices] - parent_E * 1.0e6
    parent_phonon_to_omega = np.arange(1, 1620, dtype=np.int64)
    legacy_support = np.zeros(3600, dtype=bool)
    legacy_support[parent_phonon_to_omega] = True
    projected_f = np.zeros(1640, dtype=parent_f.dtype)
    projected_f[active_indices] = parent_f
    projected_thermal = np.zeros(1640, dtype=parent_thermal.dtype)
    projected_thermal[active_indices] = parent_thermal
    projected_n = np.zeros(3600, dtype=parent_n.dtype)
    projected_n[parent_phonon_to_omega] = parent_n
    expected_arrays.update(
        {
            "legacy_phonon_support_mask": legacy_support,
            "mapped_left_edge_delta_ueV": mapped_left_edge_delta,
            "parent_E_left_eV": parent_E.copy(),
            "parent_K_minus_active": parent_K_minus,
            "parent_K_plus_active": parent_K_plus,
            "parent_f": parent_f.copy(),
            "parent_n_phonon": parent_n.copy(),
            "parent_rho_active": parent_rho,
            "parent_thermal_f": parent_thermal.copy(),
            "parent_to_native_index": parent_to_native,
            "parent_phonon_to_native_omega_index": parent_phonon_to_omega,
            "projected_f": projected_f,
            "projected_n_phonon": projected_n,
            "projected_thermal_f": projected_thermal,
            "sample_carrier_delta_ueV": sample_carrier_delta,
        }
    )

    native_K_plus = grid["native_K_plus_full"][np.ix_(active_indices, active_indices)]
    native_K_minus = grid["native_K_minus_full"][np.ix_(active_indices, active_indices)]
    native_rho = grid["native_cell_density_full"][active_indices]
    stage_specs = (
        (STAGE_IDS[0], parent_rho, parent_K_plus, parent_K_minus, 0),
        (STAGE_IDS[1], parent_rho, native_K_plus, native_K_minus, 0),
        (STAGE_IDS[2], parent_rho, native_K_plus, native_K_minus, 1),
        (STAGE_IDS[3], native_rho, native_K_plus, native_K_minus, 1),
    )
    for stage_id, rho, K_plus, K_minus, offset in stage_specs:
        evaluated = _evaluate_frozen(
            f=parent_f,
            n_phonon=parent_n,
            rho=rho,
            K_plus=K_plus,
            K_minus=K_minus,
            pair_frequency_offset_bins=offset,
            parameters=parameters,
        )
        expected_arrays.update(
            _expected_stage_arrays(
                stage_id=stage_id,
                evaluation=evaluated,
                active_indices=active_indices,
            )
        )

    if set(expected_arrays) != set(raw_arrays):
        raise C3ScoreError("C3 independently derived array closure is invalid.")
    for name, expected in expected_arrays.items():
        if not _array_bit_exact(raw_arrays[name], expected):
            raise C3ScoreError(f"C3 raw array {name!r} failed independent bit-exact recomputation.")

    c3p = STAGE_IDS[0]
    for channel in CHANNEL_NAMES:
        for field in BALANCE_FIELDS:
            raw = raw_arrays[f"{c3p}__{channel}__{field}"]
            parent = c2_arrays[f"{FINAL_C2_SLUG}__{channel}__{field}"]
            selected = raw[active_indices] if channel.startswith("qp_") else raw
            if not _array_bit_exact(selected, parent):
                raise C3ScoreError(f"C3p {channel}.{field} is not bit-exact to C2b5.")
    c2_residual = c2_arrays[f"{FINAL_C2_SLUG}__residual_s_inv"]
    if not _array_bit_exact(
        raw_arrays[f"{c3p}__qp_residual_s_inv"][active_indices],
        c2_residual[:1620],
    ) or not _array_bit_exact(
        raw_arrays[f"{c3p}__phonon_residual_s_inv"],
        c2_residual[1620:],
    ):
        raise C3ScoreError("C3p residual is not bit-exact to C2b5.")

    guard_arrays = [
        raw_arrays["projected_f"][:20],
        raw_arrays["projected_thermal_f"][:20],
        *[
            raw_arrays[f"{stage_id}__{channel}__{field}"][:20]
            for stage_id in STAGE_IDS
            for channel in ("qp_photon", "qp_scattering", "qp_pair")
            for field in BALANCE_FIELDS
        ],
        *[raw_arrays[f"{stage_id}__qp_residual_s_inv"][:20] for stage_id in STAGE_IDS],
    ]
    guard_positive_zero = all(_positive_zero(value) for value in guard_arrays)
    if not guard_positive_zero:
        raise C3ScoreError("C3p inactive guard padding is not canonical positive zero.")
    outside_support = ~legacy_support
    phonon_padding_positive_zero = _positive_zero(projected_n[outside_support])
    if not phonon_padding_positive_zero:
        raise C3ScoreError("C3 projected phonon placeholders are not positive zero.")

    net_ok, worst_net_ratio = _net_roundoff_ok(raw_arrays)
    if not net_ok:
        raise C3ScoreError("C3 net arrays exceed subtraction-roundoff closure.")
    residual_closure = True
    for stage_id in STAGE_IDS:
        qp_sum = (
            raw_arrays[f"{stage_id}__qp_photon__net_s_inv"]
            + raw_arrays[f"{stage_id}__qp_scattering__net_s_inv"]
            + raw_arrays[f"{stage_id}__qp_pair__net_s_inv"]
        )
        phonon_sum = (
            raw_arrays[f"{stage_id}__phonon_scattering__net_s_inv"]
            + raw_arrays[f"{stage_id}__phonon_pair__net_s_inv"]
            + raw_arrays[f"{stage_id}__phonon_escape__net_s_inv"]
        )
        residual_closure &= _array_bit_exact(
            qp_sum,
            raw_arrays[f"{stage_id}__qp_residual_s_inv"],
        )
        residual_closure &= _array_bit_exact(
            phonon_sum,
            raw_arrays[f"{stage_id}__phonon_residual_s_inv"],
        )
    if not residual_closure:
        raise C3ScoreError("C3 raw residual arrays fail source-order closure.")

    all_nonnegative = all(
        bool(np.all(raw_arrays[f"{stage}__{channel}__{field}"] >= 0.0))
        for stage in STAGE_IDS
        for channel in CHANNEL_NAMES
        for field in ("gain_s_inv", "loss_s_inv")
    )
    if not all_nonnegative:
        raise C3ScoreError("C3 raw gain/loss balance contains a negative value.")

    locality = {
        "c3a_changes_all_non_escape_channels": all(
            _channel_changed(raw_arrays, STAGE_IDS[0], STAGE_IDS[1], channel)
            for channel in CHANNEL_NAMES
            if channel != "phonon_escape"
        ),
        "c3a_leaves_escape_exact": not _channel_changed(
            raw_arrays,
            STAGE_IDS[0],
            STAGE_IDS[1],
            "phonon_escape",
        ),
        "c3b_changes_only_pair_channels": (
            all(
                _channel_changed(raw_arrays, STAGE_IDS[1], STAGE_IDS[2], channel)
                for channel in ("qp_pair", "phonon_pair")
            )
            and all(
                not _channel_changed(raw_arrays, STAGE_IDS[1], STAGE_IDS[2], channel)
                for channel in CHANNEL_NAMES
                if channel not in {"qp_pair", "phonon_pair"}
            )
        ),
        "c3c_changes_all_density_dependent_channels": all(
            _channel_changed(raw_arrays, STAGE_IDS[2], STAGE_IDS[3], channel)
            for channel in CHANNEL_NAMES
            if channel != "phonon_escape"
        ),
        "c3c_leaves_escape_exact": not _channel_changed(
            raw_arrays,
            STAGE_IDS[2],
            STAGE_IDS[3],
            "phonon_escape",
        ),
    }
    if any(value is not True for value in locality.values()):
        raise C3ScoreError("C3 component-substitution locality is invalid.")

    density_denominator = np.abs(parent_rho) + np.abs(native_rho)
    density_symmetric = np.zeros_like(parent_rho)
    nonzero_density = density_denominator > 0.0
    density_symmetric[nonzero_density] = (
        2.0
        * np.abs(native_rho[nonzero_density] - parent_rho[nonzero_density])
        / density_denominator[nonzero_density]
    )
    density_max = float(np.max(density_symmetric))
    density_ok = density_max <= _ACCEPTANCE_LIMITS["density_max_symmetric_relative"]
    if not density_ok:
        raise C3ScoreError("C3 native density differs beyond the declared limit.")

    observable = _observable_control(raw_arrays, parameters)
    if not _json_value_bit_exact(
        raw_metadata.get("observable_control"),
        observable,
    ):
        raise C3ScoreError("C3 raw observable control failed independent recomputation.")
    reembedding = _mapping(
        observable.get("author_semantics_reembedding"),
        "C3 author-semantics reembedding",
    )
    observable_differences = _mapping(
        reembedding.get("differences_from_parent"),
        "C3 reembedding differences",
    )
    observable_error = max(
        abs(float(observable_differences["driven_integral_signed"])),
        abs(float(observable_differences["thermal_integral_signed"])),
    )
    observable_ok = observable_error <= _ACCEPTANCE_LIMITS["observable_integral_max_absolute_error"]
    if not observable_ok:
        raise C3ScoreError("C3 projection changes a direct integral beyond its limit.")
    native_carrier = _mapping(
        observable.get("native_center_carrier"),
        "C3 native-center carrier",
    )
    native_differences = _mapping(
        native_carrier.get("differences_from_parent"),
        "C3 native-center differences",
    )
    native_driven_relative = _finite_scalar(
        native_differences.get("driven_integral_relative_signed"),
        "C3 native driven-integral relative shift",
    )
    native_thermal_relative = _finite_scalar(
        native_differences.get("thermal_integral_relative_signed"),
        "C3 native thermal-integral relative shift",
    )
    native_carrier_nonzero = bool(
        0.01 < abs(native_driven_relative) < 0.2 and 0.01 < abs(native_thermal_relative) < 0.2
    )
    if not native_carrier_nonzero:
        raise C3ScoreError("C3 native center-carrier observable effect is missing or implausible.")

    number_by_stage: dict[str, object] = {}
    for stage_id, rho, _, _, _ in stage_specs:
        number_by_stage[stage_id] = _number_diagnostics(
            raw_arrays,
            stage_id=stage_id,
            rho_active=rho,
            parameters=parameters,
        )
    number_values: list[float] = []
    for stage_record in number_by_stage.values():
        stage_mapping = _mapping(stage_record, "C3 number diagnostics")
        number_values.append(float(stage_mapping["total_weighted_number_rate_eV_s_inv"]))
        channels = _mapping(stage_mapping["channels"], "C3 number channels")
        for channel_record in channels.values():
            channel_mapping = _mapping(channel_record, "C3 number channel")
            number_values.extend(float(value) for value in channel_mapping.values())
    number_finite = all(math.isfinite(value) for value in number_values)
    if not number_finite:
        raise C3ScoreError("C3 number diagnostics contain a non-finite value.")
    number_conservation_ok = True
    for stage_record in number_by_stage.values():
        stage_mapping = _mapping(stage_record, "C3 number diagnostics")
        channels = _mapping(stage_mapping["channels"], "C3 number channels")
        for channel in ("qp_photon", "qp_scattering"):
            channel_mapping = _mapping(
                channels[channel],
                f"C3 {channel} number diagnostic",
            )
            number_conservation_ok &= (
                float(channel_mapping["symmetric_turnover_relative"])
                <= _ACCEPTANCE_LIMITS["number_conserving_channel_max_symmetric_relative"]
            )
    if not number_conservation_ok:
        raise C3ScoreError("C3 number-conserving photon/scattering channel closure failed.")

    expected_projection = {
        "active_cell_count": 1620,
        "guard_cell_count": 20,
        "mapped_left_edge_delta_ueV_max": float(np.max(mapped_left_edge_delta)),
        "mapped_left_edge_delta_ueV_min": float(np.min(mapped_left_edge_delta)),
        "mapped_left_edge_nonzero_count": int(np.count_nonzero(mapped_left_edge_delta)),
        "native_cell_count": 1640,
        "native_omega_count": 3600,
        "parent_cell_count": 1620,
        "parent_phonon_count": 1619,
        "projection_kind": "ordinal_identity_embedding_no_interpolation",
        "sample_carrier_delta_ueV_max": float(np.max(sample_carrier_delta)),
        "sample_carrier_delta_ueV_min": float(np.min(sample_carrier_delta)),
        "sample_carrier_nonzero_count": int(np.count_nonzero(sample_carrier_delta)),
    }
    if not _json_value_bit_exact(raw_metadata.get("projection"), expected_projection):
        raise C3ScoreError("C3 raw projection metadata is forged or stale.")

    projection_copy = (
        np.array_equal(raw_arrays["parent_to_native_index"], active_indices)
        and _array_bit_exact(projected_f[active_indices], parent_f)
        and _array_bit_exact(projected_thermal[active_indices], parent_thermal)
        and _array_bit_exact(
            projected_n[parent_phonon_to_omega],
            parent_n,
        )
    )
    if not projection_copy:
        raise C3ScoreError("C3 projection is not an exact ordinal copy.")
    sample_carrier_explicit = bool(
        np.all(sample_carrier_delta > 0.0)
        and float(np.min(sample_carrier_delta)) > 0.49
        and float(np.max(sample_carrier_delta)) < 0.51
    )
    if not sample_carrier_explicit:
        raise C3ScoreError("C3 author-left-sample to qpsim-center carrier shift is not explicit.")

    stage_summaries: list[dict[str, object]] = []
    previous = PARENT_STAGE_ID
    for stage_id, _rho, _, _, offset in stage_specs:
        stage_summaries.append(
            {
                "channel_difference_from_c3p": _channel_differences(
                    raw_arrays,
                    reference_stage=STAGE_IDS[0],
                    candidate_stage=stage_id,
                ),
                "channel_difference_from_previous": (
                    {
                        channel: {
                            field: {
                                "bit_exact": True,
                                "candidate_descriptor": _array_descriptor(
                                    raw_arrays[f"{stage_id}__{channel}__{field}"]
                                ),
                                "l1_absolute": 0.0,
                                "linf_absolute": 0.0,
                                "symmetric_relative_l1": 0.0,
                            }
                            for field in BALANCE_FIELDS
                        }
                        for channel in CHANNEL_NAMES
                    }
                    if previous == PARENT_STAGE_ID
                    else _channel_differences(
                        raw_arrays,
                        reference_stage=previous,
                        candidate_stage=stage_id,
                    )
                ),
                "number_diagnostics": number_by_stage[stage_id],
                "pair_frequency_offset_bins": offset,
                "residual_descriptors": {
                    "phonon": _array_descriptor(raw_arrays[f"{stage_id}__phonon_residual_s_inv"]),
                    "qp": _array_descriptor(raw_arrays[f"{stage_id}__qp_residual_s_inv"]),
                },
                "stage_id": stage_id,
            }
        )
        previous = stage_id

    acceptance_checks = {
        "all_gain_loss_arrays_nonnegative": all_nonnegative,
        "all_stage_guard_padding_is_positive_zero": guard_positive_zero,
        "c2_parent_chain_bound": raw_checks["parent"],
        "c3p_active_outputs_bit_exact_to_c2b5": True,
        "density_agrees_with_author_within_limit": density_ok,
        "frozen_parent_inputs_bit_exact": raw_checks["frozen"],
        "grid_geometry_independently_derived": True,
        "locality_checks_pass": all(locality.values()),
        "native_center_carrier_effect_nonzero": native_carrier_nonzero,
        "net_subtraction_within_roundoff": net_ok,
        "number_conserving_channels_close": number_conservation_ok,
        "number_diagnostics_finite": number_finite,
        "observable_projection_within_limit": observable_ok,
        "phonon_support_projection_exact": phonon_padding_positive_zero,
        "projection_is_copy_without_interpolation": projection_copy,
        "raw_arrays_independently_recomputed": True,
        "raw_metadata_independently_checked": raw_checks["stages"],
        "residual_closure_bit_exact": residual_closure,
        "sample_carrier_shift_explicit": sample_carrier_explicit,
        "source_closure_bound": raw_checks["sources"],
    }
    if set(acceptance_checks) != _ACCEPTANCE_CHECK_KEYS or any(
        value is not True for value in acceptance_checks.values()
    ):
        raise C3ScoreError("C3 acceptance closure is incomplete.")

    score: dict[str, Any] = {
        "acceptance": {
            "accepted": True,
            "checks": acceptance_checks,
            "limits": dict(_ACCEPTANCE_LIMITS),
        },
        "comparison": {
            "c3p_control_check_count": (len(CHANNEL_NAMES) * len(BALANCE_FIELDS) + 2),
            "locality_checks": locality,
            "net_subtraction_worst_fraction_of_limit": worst_net_ratio,
            "raw_array_count": len(raw_arrays),
            "stage_count": len(STAGE_IDS),
        },
        "limitations": {
            "scope": "one authenticated C2 frozen point only",
            "statement": (
                "No C3 nonlinear root, Newton history, stopping result, "
                "plotted ordinate, 300-point curve, or paper-parity claim "
                "is made. The full native omega lattice outside inherited "
                "author support contains serialization placeholders only."
            ),
        },
        "observable_control": {
            "independently_recomputed": observable,
            "author_reembedding_maximum_integral_absolute_difference": (observable_error),
            "native_center_carrier_relative_shifts": {
                "driven_integral": native_driven_relative,
                "thermal_integral": native_thermal_relative,
            },
            "policy": (
                "Native arrays remain authoritative; the near-invariant "
                "author-semantics control and the material native-center "
                "carrier effect are both reported separately and are never "
                "hidden in a solver tolerance."
            ),
        },
        "parent_bindings": {
            "c2_raw_manifest_sha256": c2_manifest_sha,
            "c2_receipt_path": c2_receipt_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c2_receipt_sha256": c2_receipt_sha256,
            "c2_score_path": c2_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "c2_score_sha256": c2_score_sha256,
            "c2b5_step_id": FINAL_C2_STEP_ID,
        },
        "projection": {
            "mapped_left_edge_delta_ueV": {
                "descriptor": _array_descriptor(mapped_left_edge_delta),
                "maximum": float(np.max(mapped_left_edge_delta)),
                "minimum": float(np.min(mapped_left_edge_delta)),
                "nonzero_count": int(np.count_nonzero(mapped_left_edge_delta)),
            },
            "density_comparison": {
                "maximum_symmetric_relative": density_max,
                "native_descriptor": _array_descriptor(native_rho),
                "parent_descriptor": _array_descriptor(parent_rho),
            },
            "mapping_descriptor": _array_descriptor(active_indices),
            "policy": "parent i -> child i+20; no interpolation",
            "sample_carrier_delta_ueV": {
                "descriptor": _array_descriptor(sample_carrier_delta),
                "maximum": float(np.max(sample_carrier_delta)),
                "minimum": float(np.min(sample_carrier_delta)),
                "nonzero_count": int(np.count_nonzero(sample_carrier_delta)),
            },
        },
        "raw_bundle": {
            "manifest_sha256": raw_manifest_sha,
            "schema": RAW_SCHEMA,
        },
        "schema": SCHEMA,
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
        "stages": stage_summaries,
        "structural_identity": {
            "active_child_indices": "[20, 1640)",
            "author_phonon_support": "omega/h = 1..1619 only",
            "grid": {
                "active_cell_count": 1620,
                "guard_cell_count": 20,
                "native_cell_count": 1640,
                "native_omega_count": 3600,
            },
            "projection_kind": "ordinal_identity_embedding_no_interpolation",
        },
    }
    _assert_source_snapshots()
    return score


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Serialize a checked C3 score deterministically."""

    return _canonical_json_bytes(score)


def write_c3_score(
    c3_bundle_dir: Path,
    output_path: Path,
    *,
    c2_bundle_dir: Path,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    """Build and exclusively write one checked C3 score."""

    _assert_source_snapshots()
    score = build_c3_score(
        c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_score_bytes(score))
    _assert_source_snapshots()
    return output_path


def _validate_descriptor(
    value: object,
    label: str,
    *,
    dtype: str,
    shape: list[int],
) -> None:
    descriptor = _mapping(value, label)
    _exact_keys(descriptor, {"dtype", "npy_sha256", "shape"}, label)
    if descriptor.get("dtype") != dtype or descriptor.get("shape") != shape:
        raise C3ScoreError(f"{label} has an invalid dtype or shape.")
    _sha256(descriptor.get("npy_sha256"), f"{label}.npy_sha256")


def _validate_difference_record(
    value: object,
    label: str,
    *,
    shape: list[int],
) -> bool:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {
            "bit_exact",
            "candidate_descriptor",
            "l1_absolute",
            "linf_absolute",
            "symmetric_relative_l1",
        },
        label,
    )
    bit_exact = record.get("bit_exact")
    if not isinstance(bit_exact, bool):
        raise C3ScoreError(f"{label}.bit_exact must be boolean.")
    _validate_descriptor(
        record.get("candidate_descriptor"),
        f"{label}.candidate_descriptor",
        dtype="<f8",
        shape=shape,
    )
    errors = [
        _finite_scalar(record.get(key), f"{label}.{key}", nonnegative=True)
        for key in ("l1_absolute", "linf_absolute", "symmetric_relative_l1")
    ]
    if bit_exact != all(error == 0.0 for error in errors):
        raise C3ScoreError(f"{label} bit identity and error metrics disagree.")
    return bit_exact


def _validate_channel_differences(
    value: object,
    label: str,
) -> dict[str, bool]:
    channels = _mapping(value, label)
    _exact_keys(channels, set(CHANNEL_NAMES), label)
    identities: dict[str, bool] = {}
    for channel in CHANNEL_NAMES:
        fields = _mapping(channels.get(channel), f"{label}.{channel}")
        _exact_keys(fields, set(BALANCE_FIELDS), f"{label}.{channel}")
        shape = [1640] if channel.startswith("qp_") else [1619]
        identities[channel] = all(
            _validate_difference_record(
                fields.get(field),
                f"{label}.{channel}.{field}",
                shape=shape,
            )
            for field in BALANCE_FIELDS
        )
    return identities


def _validate_number_diagnostics(value: object, label: str) -> None:
    record = _mapping(value, label)
    _exact_keys(
        record,
        {"channels", "total_weighted_number_rate_eV_s_inv"},
        label,
    )
    _finite_scalar(
        record.get("total_weighted_number_rate_eV_s_inv"),
        f"{label}.total_weighted_number_rate_eV_s_inv",
    )
    channels = _mapping(record.get("channels"), f"{label}.channels")
    _exact_keys(
        channels,
        {"qp_pair", "qp_photon", "qp_scattering"},
        f"{label}.channels",
    )
    for channel, value_by_channel in channels.items():
        channel_record = _mapping(value_by_channel, f"{label}.channels.{channel}")
        _exact_keys(
            channel_record,
            {
                "symmetric_turnover_relative",
                "weighted_number_rate_eV_s_inv",
                "weighted_turnover_eV_s_inv",
            },
            f"{label}.channels.{channel}",
        )
        _finite_scalar(
            channel_record.get("symmetric_turnover_relative"),
            f"{label}.channels.{channel}.symmetric_turnover_relative",
            nonnegative=True,
        )
        _finite_scalar(
            channel_record.get("weighted_number_rate_eV_s_inv"),
            f"{label}.channels.{channel}.weighted_number_rate_eV_s_inv",
        )
        _finite_scalar(
            channel_record.get("weighted_turnover_eV_s_inv"),
            f"{label}.channels.{channel}.weighted_turnover_eV_s_inv",
            nonnegative=True,
        )


def _validate_observable_record(value: object) -> None:
    record = _mapping(value, "C3 observable control")
    _exact_keys(
        record,
        {
            "author_reembedding_maximum_integral_absolute_difference",
            "independently_recomputed",
            "native_center_carrier_relative_shifts",
            "policy",
        },
        "C3 observable control",
    )
    expected_policy = (
        "Native arrays remain authoritative; the near-invariant "
        "author-semantics control and the material native-center carrier "
        "effect are both reported separately and are never hidden in a "
        "solver tolerance."
    )
    if record.get("policy") != expected_policy:
        raise C3ScoreError("Checked C3 observable policy is invalid.")
    independent = _mapping(
        record.get("independently_recomputed"),
        "C3 independently recomputed observable",
    )
    _exact_keys(
        independent,
        {
            "author_semantics_reembedding",
            "claim",
            "native_center_carrier",
            "parent_author_left_edge",
        },
        "C3 independently recomputed observable",
    )
    if independent.get("claim") != (
        "Two frozen projection diagnostics are reported: exact ordinal "
        "re-embedding under retained author left-edge semantics, and the "
        "actual qpsim center-carrier reinterpretation. Neither is a C3 root "
        "or plotted ordinate."
    ):
        raise C3ScoreError("Checked C3 observable claim is invalid.")

    parent = _mapping(
        independent.get("parent_author_left_edge"),
        "C3 parent observable",
    )
    parent_keys = {
        "driven_gap_eV",
        "driven_integral",
        "frozen_suppression_ratio",
        "thermal_gap_eV",
        "thermal_integral",
    }
    _exact_keys(parent, parent_keys, "C3 parent observable")
    for key in parent_keys:
        _finite_scalar(parent.get(key), f"C3 parent observable.{key}")

    interpretations = {
        "author_semantics_reembedding": (
            "Projected values deliberately re-read as author left-edge "
            "samples; this is the projection-identity control."
        ),
        "native_center_carrier": (
            "Projected values interpreted at their declared qpsim cell "
            "centers; this reports the half-bin carrier effect."
        ),
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for key, interpretation in interpretations.items():
        diagnostic = _mapping(independent.get(key), f"C3 observable {key}")
        _exact_keys(
            diagnostic,
            {"child_full_grid", "differences_from_parent", "interpretation"},
            f"C3 observable {key}",
        )
        if diagnostic.get("interpretation") != interpretation:
            raise C3ScoreError(f"Checked C3 observable {key} interpretation is invalid.")
        child = _mapping(
            diagnostic.get("child_full_grid"),
            f"C3 observable {key}.child_full_grid",
        )
        _exact_keys(
            child,
            {
                "driven_gap_ueV",
                "driven_integral",
                "frozen_suppression_ratio",
                "thermal_gap_ueV",
                "thermal_integral",
            },
            f"C3 observable {key}.child_full_grid",
        )
        for child_key, child_value in child.items():
            _finite_scalar(
                child_value,
                f"C3 observable {key}.child_full_grid.{child_key}",
            )
        differences = _mapping(
            diagnostic.get("differences_from_parent"),
            f"C3 observable {key}.differences_from_parent",
        )
        _exact_keys(
            differences,
            {
                "driven_gap_eV_equivalent_signed",
                "driven_integral_relative_signed",
                "driven_integral_signed",
                "thermal_gap_eV_equivalent_signed",
                "thermal_integral_relative_signed",
                "thermal_integral_signed",
            },
            f"C3 observable {key}.differences_from_parent",
        )
        for difference_key, difference_value in differences.items():
            _finite_scalar(
                difference_value,
                f"C3 observable {key}.differences_from_parent.{difference_key}",
            )
        diagnostics[key] = diagnostic

    author_differences = _mapping(
        diagnostics["author_semantics_reembedding"]["differences_from_parent"],
        "C3 author reembedding differences",
    )
    author_error = max(
        abs(float(author_differences["driven_integral_signed"])),
        abs(float(author_differences["thermal_integral_signed"])),
    )
    recorded_author_error = _finite_scalar(
        record.get("author_reembedding_maximum_integral_absolute_difference"),
        "C3 author reembedding maximum error",
        nonnegative=True,
    )
    if (
        recorded_author_error != author_error
        or recorded_author_error > _ACCEPTANCE_LIMITS["observable_integral_max_absolute_error"]
    ):
        raise C3ScoreError("Checked C3 author-reembedding error is invalid.")

    native_differences = _mapping(
        diagnostics["native_center_carrier"]["differences_from_parent"],
        "C3 native-center differences",
    )
    shifts = _mapping(
        record.get("native_center_carrier_relative_shifts"),
        "C3 native-center relative shifts",
    )
    _exact_keys(
        shifts,
        {"driven_integral", "thermal_integral"},
        "C3 native-center relative shifts",
    )
    expected_shifts = {
        "driven_integral": native_differences["driven_integral_relative_signed"],
        "thermal_integral": native_differences["thermal_integral_relative_signed"],
    }
    if not _json_value_bit_exact(shifts, expected_shifts):
        raise C3ScoreError("Checked C3 native-center shift summary is inconsistent.")
    if not (
        0.01 < abs(float(shifts["driven_integral"])) < 0.2
        and 0.01 < abs(float(shifts["thermal_integral"])) < 0.2
    ):
        raise C3ScoreError("Checked C3 native-center shifts are implausible.")


def _validate_score_structure(score: dict[str, Any]) -> None:
    _exact_keys(score, _SCORE_KEYS, "checked C3 score")
    if score.get("schema") != SCHEMA:
        raise C3ScoreError("Checked C3 score schema is unsupported.")
    if score.get("stage") != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C3ScoreError("Checked C3 stage identity is invalid.")
    acceptance = _mapping(score.get("acceptance"), "C3 acceptance")
    _exact_keys(acceptance, {"accepted", "checks", "limits"}, "C3 acceptance")
    if acceptance.get("accepted") is not True or acceptance.get("limits") != _ACCEPTANCE_LIMITS:
        raise C3ScoreError("Checked C3 acceptance declaration is invalid.")
    checks = _mapping(acceptance.get("checks"), "C3 acceptance checks")
    if set(checks) != _ACCEPTANCE_CHECK_KEYS or any(value is not True for value in checks.values()):
        raise C3ScoreError("Checked C3 acceptance checks are invalid.")
    raw = _mapping(score.get("raw_bundle"), "C3 raw binding")
    _exact_keys(raw, {"manifest_sha256", "schema"}, "C3 raw binding")
    if raw.get("schema") != RAW_SCHEMA:
        raise C3ScoreError("Checked C3 raw schema is invalid.")
    _sha256(raw.get("manifest_sha256"), "C3 raw manifest SHA-256")
    parent = _mapping(score.get("parent_bindings"), "C3 parent bindings")
    _exact_keys(
        parent,
        {
            "c2_raw_manifest_sha256",
            "c2_receipt_path",
            "c2_receipt_sha256",
            "c2_score_path",
            "c2_score_sha256",
            "c2b5_step_id",
        },
        "C3 parent bindings",
    )
    if (
        parent.get("c2_score_path") != DEFAULT_C2_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
        or parent.get("c2_receipt_path")
        != DEFAULT_C2_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
        or parent.get("c2b5_step_id") != FINAL_C2_STEP_ID
    ):
        raise C3ScoreError("Checked C3 parent identity is invalid.")
    _sha256(parent.get("c2_raw_manifest_sha256"), "C3 parent raw manifest SHA-256")
    if _file_sha256(DEFAULT_C2_SCORE, "C2 score") != _sha256(
        parent.get("c2_score_sha256"),
        "C3 parent C2 score SHA-256",
    ) or _file_sha256(DEFAULT_C2_RECEIPT, "C2 receipt") != _sha256(
        parent.get("c2_receipt_sha256"),
        "C3 parent C2 receipt SHA-256",
    ):
        raise C3ScoreError("Checked C3 parent score/receipt binding is stale.")
    stages = score.get("stages")
    if (
        not isinstance(stages, list)
        or len(stages) != 4
        or [
            _mapping(stage, f"C3 stages[{index}]").get("stage_id")
            for index, stage in enumerate(stages)
        ]
        != list(STAGE_IDS)
    ):
        raise C3ScoreError("Checked C3 stage closure is invalid.")
    comparison = _mapping(score.get("comparison"), "C3 comparison")
    _exact_keys(
        comparison,
        {
            "c3p_control_check_count",
            "locality_checks",
            "net_subtraction_worst_fraction_of_limit",
            "raw_array_count",
            "stage_count",
        },
        "C3 comparison",
    )
    if (
        comparison.get("c3p_control_check_count") != 20
        or comparison.get("raw_array_count") != 105
        or comparison.get("stage_count") != 4
    ):
        raise C3ScoreError("Checked C3 comparison counts are invalid.")
    locality = _mapping(comparison.get("locality_checks"), "C3 locality checks")
    expected_locality = {
        "c3a_changes_all_non_escape_channels",
        "c3a_leaves_escape_exact",
        "c3b_changes_only_pair_channels",
        "c3c_changes_all_density_dependent_channels",
        "c3c_leaves_escape_exact",
    }
    if set(locality) != expected_locality or any(value is not True for value in locality.values()):
        raise C3ScoreError("Checked C3 locality checks are invalid.")
    worst = _finite_scalar(
        comparison.get("net_subtraction_worst_fraction_of_limit"),
        "C3 net subtraction fraction",
        nonnegative=True,
    )
    if worst > 1.0:
        raise C3ScoreError("Checked C3 net subtraction fraction exceeds its limit.")

    projection = _mapping(score.get("projection"), "C3 projection")
    _exact_keys(
        projection,
        {
            "density_comparison",
            "mapped_left_edge_delta_ueV",
            "mapping_descriptor",
            "policy",
            "sample_carrier_delta_ueV",
        },
        "C3 projection",
    )
    if projection.get("policy") != "parent i -> child i+20; no interpolation":
        raise C3ScoreError("Checked C3 projection policy is invalid.")
    _validate_descriptor(
        projection.get("mapping_descriptor"),
        "C3 projection mapping descriptor",
        dtype="<i8",
        shape=[1620],
    )
    mapped = _mapping(
        projection.get("mapped_left_edge_delta_ueV"),
        "C3 mapped-left-edge delta",
    )
    _exact_keys(
        mapped,
        {"descriptor", "maximum", "minimum", "nonzero_count"},
        "C3 mapped-left-edge delta",
    )
    _validate_descriptor(
        mapped.get("descriptor"),
        "C3 mapped-left-edge descriptor",
        dtype="<f8",
        shape=[1620],
    )
    mapped_min = _finite_scalar(
        mapped.get("minimum"),
        "C3 mapped-left-edge minimum",
    )
    mapped_max = _finite_scalar(
        mapped.get("maximum"),
        "C3 mapped-left-edge maximum",
    )
    if _strict_int(
        mapped.get("nonzero_count"),
        "C3 mapped-left-edge nonzero count",
        minimum=1,
    ) != 449 or not (-3e-13 < mapped_min < 0.0 < mapped_max < 3e-13):
        raise C3ScoreError("Checked C3 mapped-left-edge summary is invalid.")
    carrier = _mapping(
        projection.get("sample_carrier_delta_ueV"),
        "C3 sample-carrier delta",
    )
    _exact_keys(
        carrier,
        {"descriptor", "maximum", "minimum", "nonzero_count"},
        "C3 sample-carrier delta",
    )
    _validate_descriptor(
        carrier.get("descriptor"),
        "C3 sample-carrier descriptor",
        dtype="<f8",
        shape=[1620],
    )
    carrier_min = _finite_scalar(
        carrier.get("minimum"),
        "C3 sample-carrier minimum",
    )
    carrier_max = _finite_scalar(
        carrier.get("maximum"),
        "C3 sample-carrier maximum",
    )
    if _strict_int(
        carrier.get("nonzero_count"),
        "C3 sample-carrier nonzero count",
        minimum=1,
    ) != 1620 or not (0.49 < carrier_min <= carrier_max < 0.51):
        raise C3ScoreError("Checked C3 sample-carrier summary is invalid.")
    density = _mapping(
        projection.get("density_comparison"),
        "C3 density comparison",
    )
    _exact_keys(
        density,
        {
            "maximum_symmetric_relative",
            "native_descriptor",
            "parent_descriptor",
        },
        "C3 density comparison",
    )
    density_max = _finite_scalar(
        density.get("maximum_symmetric_relative"),
        "C3 density maximum symmetric relative",
        nonnegative=True,
    )
    if density_max > _ACCEPTANCE_LIMITS["density_max_symmetric_relative"]:
        raise C3ScoreError("Checked C3 density comparison exceeds its limit.")
    for key in ("native_descriptor", "parent_descriptor"):
        _validate_descriptor(
            density.get(key),
            f"C3 density {key}",
            dtype="<f8",
            shape=[1620],
        )

    _validate_observable_record(score.get("observable_control"))

    expected_offsets = (0, 0, 1, 1)
    expected_previous_identity = (
        set(CHANNEL_NAMES),
        {"phonon_escape"},
        set(CHANNEL_NAMES) - {"qp_pair", "phonon_pair"},
        {"phonon_escape"},
    )
    expected_c3p_identity = (
        set(CHANNEL_NAMES),
        {"phonon_escape"},
        {"phonon_escape"},
        {"phonon_escape"},
    )
    for index, stage_value in enumerate(stages):
        stage = _mapping(stage_value, f"C3 stages[{index}]")
        _exact_keys(
            stage,
            {
                "channel_difference_from_c3p",
                "channel_difference_from_previous",
                "number_diagnostics",
                "pair_frequency_offset_bins",
                "residual_descriptors",
                "stage_id",
            },
            f"C3 stages[{index}]",
        )
        if (
            stage.get("stage_id") != STAGE_IDS[index]
            or stage.get("pair_frequency_offset_bins") != expected_offsets[index]
        ):
            raise C3ScoreError(f"Checked C3 stage {index} identity is invalid.")
        from_c3p = _validate_channel_differences(
            stage.get("channel_difference_from_c3p"),
            f"C3 stages[{index}].channel_difference_from_c3p",
        )
        from_previous = _validate_channel_differences(
            stage.get("channel_difference_from_previous"),
            f"C3 stages[{index}].channel_difference_from_previous",
        )
        if {channel for channel, exact in from_c3p.items() if exact} != expected_c3p_identity[
            index
        ] or {
            channel for channel, exact in from_previous.items() if exact
        } != expected_previous_identity[index]:
            raise C3ScoreError(f"Checked C3 stage {index} locality detail is invalid.")
        _validate_number_diagnostics(
            stage.get("number_diagnostics"),
            f"C3 stages[{index}].number_diagnostics",
        )
        residuals = _mapping(
            stage.get("residual_descriptors"),
            f"C3 stages[{index}].residual_descriptors",
        )
        _exact_keys(
            residuals,
            {"phonon", "qp"},
            f"C3 stages[{index}].residual_descriptors",
        )
        _validate_descriptor(
            residuals.get("phonon"),
            f"C3 stages[{index}].phonon residual",
            dtype="<f8",
            shape=[1619],
        )
        _validate_descriptor(
            residuals.get("qp"),
            f"C3 stages[{index}].qp residual",
            dtype="<f8",
            shape=[1640],
        )

    structural = _mapping(score.get("structural_identity"), "C3 structural identity")
    if structural != {
        "active_child_indices": "[20, 1640)",
        "author_phonon_support": "omega/h = 1..1619 only",
        "grid": {
            "active_cell_count": 1620,
            "guard_cell_count": 20,
            "native_cell_count": 1640,
            "native_omega_count": 3600,
        },
        "projection_kind": "ordinal_identity_embedding_no_interpolation",
    }:
        raise C3ScoreError("Checked C3 structural identity is invalid.")

    limitations = _mapping(score.get("limitations"), "C3 limitations")
    _exact_keys(limitations, {"scope", "statement"}, "C3 limitations")
    if limitations != {
        "scope": "one authenticated C2 frozen point only",
        "statement": (
            "No C3 nonlinear root, Newton history, stopping result, plotted "
            "ordinate, 300-point curve, or paper-parity claim is made. The "
            "full native omega lattice outside inherited author support "
            "contains serialization placeholders only."
        ),
    }:
        raise C3ScoreError("Checked C3 limitation is invalid.")
    sources = _mapping(score.get("sources"), "C3 score sources")
    expected_paths = {path.relative_to(REPOSITORY_ROOT).as_posix() for path in _SOURCE_PATHS}
    if set(sources) != expected_paths:
        raise C3ScoreError("Checked C3 source closure is incomplete.")
    for relative, digest in sources.items():
        if source_sha256(REPOSITORY_ROOT / relative) != _sha256(
            digest,
            f"C3 sources.{relative}",
        ):
            raise C3ScoreError("Checked C3 source binding is stale.")


def load_c3_receipt(path: Path = DEFAULT_RECEIPT) -> dict[str, Any]:
    """Strictly load the repository trust anchor for C3 raw/score bytes."""

    raw = _read_regular_file_once(path, "C3 raw-manifest receipt")
    receipt = _parse_json(raw, "C3 raw-manifest receipt")
    if raw != _canonical_json_bytes(receipt):
        raise C3ScoreError("C3 raw-manifest receipt is not canonical JSON.")
    _exact_keys(
        receipt,
        {"checked_score", "qualification", "raw_bundle", "schema"},
        "C3 raw-manifest receipt",
    )
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise C3ScoreError("C3 raw-manifest receipt schema is unsupported.")
    if receipt.get("qualification") != (
        "Repository trust anchor for the externally retained C3 raw manifest "
        "and the complete canonical checked-score bytes; it does not contain "
        "or replace the raw arrays."
    ):
        raise C3ScoreError("C3 raw-manifest receipt qualification is invalid.")
    checked = _mapping(receipt.get("checked_score"), "receipt.checked_score")
    _exact_keys(checked, {"file_sha256", "schema"}, "receipt.checked_score")
    if checked.get("schema") != SCHEMA:
        raise C3ScoreError("C3 receipt checked-score schema is invalid.")
    _sha256(checked.get("file_sha256"), "receipt.checked_score.file_sha256")
    bundle = _mapping(receipt.get("raw_bundle"), "receipt.raw_bundle")
    _exact_keys(bundle, {"manifest_sha256", "schema"}, "receipt.raw_bundle")
    if bundle.get("schema") != RAW_SCHEMA:
        raise C3ScoreError("C3 receipt raw-bundle schema is invalid.")
    _sha256(bundle.get("manifest_sha256"), "receipt.raw_bundle.manifest_sha256")
    return receipt


def _load_c3_score_unbound(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_file_once(path, "checked C3 score")
    score = _parse_json(raw, "checked C3 score")
    if raw != canonical_score_bytes(score):
        raise C3ScoreError("Checked C3 score is not canonical JSON.")
    _validate_score_structure(score)
    return score, raw


def load_c3_score(
    path: Path = DEFAULT_SCORE,
    *,
    receipt_path: Path = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    """Strictly load a checked C3 score and bind its complete bytes to a receipt."""

    score, score_raw = _load_c3_score_unbound(path)
    receipt = load_c3_receipt(receipt_path)
    checked = _mapping(receipt.get("checked_score"), "receipt.checked_score")
    if hashlib.sha256(score_raw).hexdigest() != checked.get("file_sha256"):
        raise C3ScoreError("Checked C3 score bytes do not match the selected receipt.")
    if score.get("raw_bundle") != receipt.get("raw_bundle"):
        raise C3ScoreError("Checked C3 raw binding does not match the selected receipt.")
    parent = _mapping(score.get("parent_bindings"), "C3 parent bindings")
    _exact_keys(
        parent,
        {
            "c2_raw_manifest_sha256",
            "c2_receipt_path",
            "c2_receipt_sha256",
            "c2_score_path",
            "c2_score_sha256",
            "c2b5_step_id",
        },
        "C3 parent bindings",
    )
    if (
        parent.get("c2_score_path") != DEFAULT_C2_SCORE.relative_to(REPOSITORY_ROOT).as_posix()
        or parent.get("c2_receipt_path")
        != DEFAULT_C2_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix()
        or parent.get("c2b5_step_id") != FINAL_C2_STEP_ID
    ):
        raise C3ScoreError("Checked C3 does not bind the canonical C2 parent paths.")
    if _file_sha256(DEFAULT_C2_SCORE, "C2 score") != _sha256(
        parent.get("c2_score_sha256"),
        "C3 parent C2 score SHA-256",
    ):
        raise C3ScoreError("Checked C3 C2-score binding is stale.")
    if _file_sha256(DEFAULT_C2_RECEIPT, "C2 receipt") != _sha256(
        parent.get("c2_receipt_sha256"),
        "C3 parent C2 receipt SHA-256",
    ):
        raise C3ScoreError("Checked C3 C2-receipt binding is stale.")
    accepted_c2 = load_c2_score(DEFAULT_C2_SCORE, receipt_path=DEFAULT_C2_RECEIPT)
    c2_raw = _mapping(accepted_c2.get("raw_bundle"), "accepted C2 raw binding")
    if parent.get("c2_raw_manifest_sha256") != c2_raw.get("manifest_sha256"):
        raise C3ScoreError("Checked C3 accepted-C2 raw binding is stale.")
    return score


def build_c3_receipt(
    score_path: Path = DEFAULT_SCORE,
    *,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> dict[str, Any]:
    """Build a receipt only after independently reproducing the score bytes."""

    score, score_raw = _load_c3_score_unbound(score_path)
    rebuilt = build_c3_score(
        c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    if canonical_score_bytes(rebuilt) != score_raw:
        raise C3ScoreError(
            "C3 receipt refuses to anchor score bytes that do not reproduce "
            "from the selected raw C3/C2 evidence."
        )
    raw_bundle = _mapping(score.get("raw_bundle"), "C3 raw binding")
    return {
        "checked_score": {
            "file_sha256": hashlib.sha256(score_raw).hexdigest(),
            "schema": SCHEMA,
        },
        "qualification": (
            "Repository trust anchor for the externally retained C3 raw "
            "manifest and the complete canonical checked-score bytes; it "
            "does not contain or replace the raw arrays."
        ),
        "raw_bundle": dict(raw_bundle),
        "schema": RECEIPT_SCHEMA,
    }


def write_c3_receipt(
    output_path: Path,
    *,
    score_path: Path = DEFAULT_SCORE,
    c3_bundle_dir: Path,
    c2_bundle_dir: Path,
    c2_score_path: Path = DEFAULT_C2_SCORE,
    c2_receipt_path: Path = DEFAULT_C2_RECEIPT,
) -> Path:
    """Build and exclusively write one C3 score/raw receipt."""

    receipt = build_c3_receipt(
        score_path,
        c3_bundle_dir=c3_bundle_dir,
        c2_bundle_dir=c2_bundle_dir,
        c2_score_path=c2_score_path,
        c2_receipt_path=c2_receipt_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(_canonical_json_bytes(receipt))
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    score = subparsers.add_parser("score", help="build a checked C3 score")
    score.add_argument("--c3-bundle", type=Path, required=True)
    score.add_argument("--c2-bundle", type=Path, required=True)
    score.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    score.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)
    score.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    receipt = subparsers.add_parser("receipt", help="build the C3 receipt")
    receipt.add_argument("--score", type=Path, default=DEFAULT_SCORE)
    receipt.add_argument("--c3-bundle", type=Path, required=True)
    receipt.add_argument("--c2-bundle", type=Path, required=True)
    receipt.add_argument("--c2-score", type=Path, default=DEFAULT_C2_SCORE)
    receipt.add_argument("--c2-receipt", type=Path, default=DEFAULT_C2_RECEIPT)
    receipt.add_argument("--output", type=Path, default=DEFAULT_RECEIPT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "receipt":
        print(
            write_c3_receipt(
                args.output,
                score_path=args.score,
                c3_bundle_dir=args.c3_bundle,
                c2_bundle_dir=args.c2_bundle,
                c2_score_path=args.c2_score,
                c2_receipt_path=args.c2_receipt,
            )
        )
    else:
        print(
            write_c3_score(
                args.c3_bundle,
                args.output,
                c2_bundle_dir=args.c2_bundle,
                c2_score_path=args.c2_score,
                c2_receipt_path=args.c2_receipt,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
