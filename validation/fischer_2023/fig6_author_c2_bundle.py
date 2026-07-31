"""Build immutable frozen-state evidence for the Figure 6 C2 parameter stage.

C2 changes only scalar parameters and numerical constants.  The author
left-edge grid, frozen C0 quasiparticle/phonon state, spectral formulae,
channel equations, and nonlinear policy remain untouched.  Consequently this
bundle records gain, loss, net, and total-residual arrays at the same frozen
state for a controlled cumulative parameter path.  It does not run a
nonlinear solve or claim a new plotted ordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY as DEFAULT_C0_SUMMARY,
)
from validation.fischer_2023.fig6_author_c0_summary import (
    load_c0_raw_bundle,
    load_c0_summary,
)
from validation.fischer_2023.fig6_author_c1_score import (
    DEFAULT_SCORE as DEFAULT_C1_SCORE,
)
from validation.fischer_2023.fig6_author_c1_score import load_c1_score
from validation.fischer_2023.fig6_author_c2_parameters import (
    NativeFig6Parameters,
    build_c2_parameter_plan,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorOperator,
    AuthorSolveParameters,
    SystemEvaluation,
    build_author_operator,
    evaluate_author_system,
)
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c2-frozen-bundle.v1"
STAGE_ID = "C2"
PARENT_STAGE_ID = "C1"
CHANGED_COMPONENT = "parameters"
CHANNEL_NAMES = (
    "qp_photon",
    "qp_scattering",
    "qp_pair",
    "phonon_scattering",
    "phonon_pair",
    "phonon_escape",
)
BALANCE_FIELDS = ("gain_s_inv", "loss_s_inv", "net_s_inv")
FROZEN_ARRAY_NAMES = (
    "E_left_eV",
    "f_final",
    "n_phonon_final",
    "thermal_f",
)

_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "qpsim" / "constants.py",
    REPOSITORY_ROOT / "qpsim" / "physics" / "gap_equation.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_solve.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c0_summary.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c1_score.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_parameters.py",
    REPOSITORY_ROOT / "validation" / "reference_models" / "fischer_2023" / "fig6_author_c0.py",
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_HASHES_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}


class C2BundleError(ValueError):
    """The C2 parent evidence, raw transport, or parameter path is invalid."""


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        if canonical_source_bytes(REPOSITORY_ROOT / relative) != expected:
            raise C2BundleError(f"C2 numerical source changed during execution: {relative}.")


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C2BundleError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise C2BundleError(f"Non-finite JSON constant {token!r} is forbidden.")


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C2BundleError(f"Cannot parse {label}: {exc}.") from exc
    if not isinstance(value, dict):
        raise C2BundleError(f"{label} must be an object.")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C2BundleError(f"{label} must be an object.")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise C2BundleError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(value)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C2BundleError(f"{label} must be a lowercase SHA-256 digest.")
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


def json_value_bit_exact(reference: object, candidate: object) -> bool:
    """Compare values in the JSON data model without Python's coercive equality.

    Python considers ``False == 0`` and ``0.0 == -0.0``.  Neither is an
    acceptable identity rule for provenance metadata: the former changes the
    JSON scalar type and the latter changes the IEEE-754 bit pattern.  C2 uses
    this helper at independently recomputed metadata boundaries.
    """

    if type(reference) is not type(candidate):
        return False
    if isinstance(reference, dict):
        assert isinstance(candidate, dict)
        return set(reference) == set(candidate) and all(
            json_value_bit_exact(reference[key], candidate[key]) for key in reference
        )
    if isinstance(reference, list):
        assert isinstance(candidate, list)
        return len(reference) == len(candidate) and all(
            json_value_bit_exact(left, right)
            for left, right in zip(reference, candidate, strict=True)
        )
    if isinstance(reference, float):
        assert isinstance(candidate, float)
        return (
            math.isfinite(reference)
            and math.isfinite(candidate)
            and struct.pack(">d", reference) == struct.pack(">d", candidate)
        )
    if reference is None or isinstance(reference, (bool, int, str)):
        return reference == candidate
    return False


def array_descriptor(value: np.ndarray) -> dict[str, object]:
    """Return the portable descriptor used by both C2 evidence layers."""

    array = np.asarray(value)
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _file_sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise C2BundleError(f"{label} is missing, unsafe, or a symlink.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_regular_file_once(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise C2BundleError(f"{label} is missing, unsafe, or a symlink.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise C2BundleError(f"Cannot read {label}: {exc}.") from exc


def _author_parameters_from_summary(summary: dict[str, Any]) -> AuthorSolveParameters:
    raw = _mapping(summary.get("parameters"), "C0 parameters")
    observable = _mapping(summary.get("observable"), "C0 observable")
    required = {
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
    }
    if set(raw) != {*required, "num_energy_cells"}:
        raise C2BundleError("C0 parameter closure is invalid for C2.")

    def finite(key: str, *, positive: bool = True) -> float:
        value = raw[key]
        if isinstance(value, bool):
            raise C2BundleError(f"C0 parameter {key!r} is not a real scalar.")
        result = float(value)
        if not math.isfinite(result) or (positive and result <= 0.0):
            raise C2BundleError(f"C0 parameter {key!r} is invalid.")
        return result

    photon_bin = raw["photon_bin"]
    max_steps = raw["max_newton_steps"]
    if (
        isinstance(photon_bin, bool)
        or not isinstance(photon_bin, int)
        or photon_bin < 1
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps < 1
    ):
        raise C2BundleError("C0 integer parameters are invalid.")
    thermal_gap = observable.get("thermal_gap_eV")
    if isinstance(thermal_gap, bool):
        raise C2BundleError("C0 thermal gap is invalid.")
    thermal_gap_value = float(thermal_gap)  # type: ignore[arg-type]
    if not math.isfinite(thermal_gap_value) or thermal_gap_value <= 0.0:
        raise C2BundleError("C0 thermal gap is invalid.")
    return AuthorSolveParameters(
        gap_eV=finite("gap_eV"),
        h_eV=finite("h_eV"),
        temperature_K=finite("temperature_K"),
        T_c_K=finite("T_c_K"),
        tau_0_s=finite("tau_0_s"),
        tau_0_pb_s=finite("tau_0_pb_s"),
        tau_l_s=finite("tau_l_s"),
        photon_bin=photon_bin,
        n_bar=finite("n_bar"),
        c_photon_s_inv=finite("c_photon_s_inv"),
        delta0_eV=finite("delta0_eV"),
        thermal_gap_eV=thermal_gap_value,
        max_newton_steps=max_steps,
        relative_step_threshold=finite("relative_step_threshold"),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=finite("boltzmann_constant_J_per_K"),
            electron_charge_C=finite("electron_charge_C"),
        ),
    )


def _step_slug(step_id: str) -> str:
    slug = step_id.lower().replace("-", "_")
    if not slug.replace("_", "").isalnum():
        raise C2BundleError(f"Unsafe C2 step id {step_id!r}.")
    return slug


def _append_evaluation_arrays(
    output: dict[str, np.ndarray],
    *,
    slug: str,
    operator: AuthorOperator,
    evaluation: SystemEvaluation,
) -> dict[str, object]:
    names: list[str] = []
    n_name = f"{slug}__n_thermal"
    output[n_name] = np.asarray(operator.n_thermal)
    names.append(n_name)
    for channel_name in CHANNEL_NAMES:
        balance = getattr(evaluation, channel_name)
        for field in BALANCE_FIELDS:
            name = f"{slug}__{channel_name}__{field}"
            output[name] = np.asarray(getattr(balance, field))
            names.append(name)
    residual_name = f"{slug}__residual_s_inv"
    output[residual_name] = np.asarray(evaluation.residual_s_inv)
    names.append(residual_name)
    return {
        "array_names": names,
        "operator_scalars": {
            "a_delta": operator.a_delta,
            "pair_frequency_offset_bins": operator.pair_frequency_offset_bins,
            "phonon_prefactor_per_eV_s": operator.phonon_prefactor_per_eV_s,
            "qp_prefactor_s_inv": operator.qp_prefactor_s_inv,
        },
        "structural_descriptors": {
            "K_minus": array_descriptor(operator.K_minus),
            "K_plus": array_descriptor(operator.K_plus),
            "rho": array_descriptor(operator.rho),
        },
    }


def _parent_bindings(
    *,
    c0_summary_path: Path,
    c1_score_path: Path,
    raw_manifest_sha: str,
) -> dict[str, str]:
    return {
        "c0_raw_manifest_sha256": raw_manifest_sha,
        "c0_summary_path": c0_summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c0_summary_sha256": _file_sha256(c0_summary_path, "C0 summary"),
        "c1_score_path": c1_score_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "c1_score_sha256": _file_sha256(c1_score_path, "C1 score"),
    }


def build_c2_bundle(
    c0_bundle_dir: Path,
    *,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
    c1_score_path: Path = DEFAULT_C1_SCORE,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate the controlled C2 parameter path on exact frozen C0 arrays."""

    _assert_source_snapshots()
    c0_summary_path = c0_summary_path.resolve()
    c1_score_path = c1_score_path.resolve()
    raw_metadata, c0_arrays, raw_manifest_sha = load_c0_raw_bundle(c0_bundle_dir)
    c0_summary = load_c0_summary(c0_summary_path)
    c1_score = load_c1_score(c1_score_path)
    c0_raw = _mapping(c0_summary.get("raw_bundle"), "C0 raw binding")
    c1_c0 = _mapping(c1_score.get("c0_binding"), "C1 C0 binding")
    if (
        c0_raw.get("manifest_sha256") != raw_manifest_sha
        or c1_c0.get("raw_manifest_sha256") != raw_manifest_sha
        or c0_summary.get("array_descriptors") != raw_metadata.get("array_descriptors")
    ):
        raise C2BundleError("C2 parent C0/C1/raw bindings disagree.")

    parent = _author_parameters_from_summary(c0_summary)
    parent_record = dict(_mapping(c0_summary.get("parameters"), "C0 parameters"))
    parent_record["thermal_gap_eV"] = parent.thermal_gap_eV
    plan = build_c2_parameter_plan(parent_record)
    if plan.parent_author_parameters != parent:
        raise C2BundleError("C2 parameter resolver did not preserve the accepted parent.")
    arrays = {name: np.asarray(c0_arrays[name]).copy() for name in FROZEN_ARRAY_NAMES}
    for value in arrays.values():
        value.setflags(write=False)
    frozen_before = {name: array_descriptor(value) for name, value in arrays.items()}
    f_final = arrays["f_final"]
    n_final = arrays["n_phonon_final"]

    step_records: list[dict[str, object]] = []
    step_inputs: list[tuple[str, tuple[str, ...], NativeFig6Parameters, str]] = [
        (
            "C2a-author-value-plumbing",
            (),
            plan.c2a,
            (
                "Native units are emitted, while the effective author-space "
                "parameters remain exact parent bits."
            ),
        ),
        *[
            (
                step.step_id,
                step.changed_fields,
                step.parameters,
                step.qualification,
            )
            for step in plan.c2b_steps
        ],
    ]
    c2b_effective = dict(plan.c2b_author_effective_steps())
    for index, (step_id, changed_fields, native, qualification) in enumerate(step_inputs):
        # C2a is a plumbing identity: do not feed a lossy native energy
        # round-trip into the still-author operator.
        effective = plan.c2a_author_effective if index == 0 else c2b_effective[step_id]
        operator = build_author_operator(arrays["E_left_eV"], effective)
        evaluation = evaluate_author_system(
            operator,
            f_final,
            n_final,
            build_update_matrix=False,
        )
        slug = _step_slug(step_id)
        record = _append_evaluation_arrays(
            arrays,
            slug=slug,
            operator=operator,
            evaluation=evaluation,
        )
        record.update(
            {
                "changed_fields": list(changed_fields),
                "effective_author_units": {
                    "T_c_K": effective.T_c_K,
                    "boltzmann_constant_J_per_K": (effective.constants.boltzmann_constant_J_per_K),
                    "c_photon_s_inv": effective.c_photon_s_inv,
                    "delta0_eV": effective.delta0_eV,
                    "electron_charge_C": effective.constants.electron_charge_C,
                    "gap_eV": effective.gap_eV,
                    "h_eV": effective.h_eV,
                    "n_bar": effective.n_bar,
                    "tau_0_pb_s": effective.tau_0_pb_s,
                    "tau_0_s": effective.tau_0_s,
                    "tau_l_s": effective.tau_l_s,
                    "temperature_K": effective.temperature_K,
                },
                "effective_hex": {
                    "T_c_K": effective.T_c_K.hex(),
                    "boltzmann_constant_J_per_K": (
                        effective.constants.boltzmann_constant_J_per_K.hex()
                    ),
                    "c_photon_s_inv": effective.c_photon_s_inv.hex(),
                    "delta0_eV": effective.delta0_eV.hex(),
                    "electron_charge_C": effective.constants.electron_charge_C.hex(),
                    "gap_eV": effective.gap_eV.hex(),
                    "h_eV": effective.h_eV.hex(),
                    "n_bar": effective.n_bar.hex(),
                    "tau_0_pb_s": effective.tau_0_pb_s.hex(),
                    "tau_0_s": effective.tau_0_s.hex(),
                    "tau_l_s": effective.tau_l_s.hex(),
                    "temperature_K": effective.temperature_K.hex(),
                },
                "index": index,
                "native_parameters": native.as_record(),
                "qualification": qualification,
                "step_id": step_id,
            }
        )
        step_records.append(record)
    frozen_after = {name: array_descriptor(arrays[name]) for name in FROZEN_ARRAY_NAMES}
    if frozen_after != frozen_before:
        raise C2BundleError("A C2 evaluation mutated its frozen parent arrays.")
    array_descriptors = {name: array_descriptor(value) for name, value in sorted(arrays.items())}
    metadata: dict[str, Any] = {
        "array_descriptors": array_descriptors,
        "coordinate_contract": {
            "carrier_occupation_samples": "author left edges E_i=Delta+i*h",
            "grid_projection": "none",
            "pair_frequency_offset_bins": 0,
            "state_layout": "[f(E_0..E_N-1), n(h..(N-1)h)]",
        },
        "frozen_inputs": {
            "descriptors": frozen_before,
            "mutation_check_after_all_steps": True,
        },
        "parameter_plan": plan.as_record(),
        "parent_bindings": _parent_bindings(
            c0_summary_path=c0_summary_path,
            c1_score_path=c1_score_path,
            raw_manifest_sha=raw_manifest_sha,
        ),
        "schema": SCHEMA,
        "source_binding": {
            "hash_kind": "canonical_sha256_import_time_disk_snapshot",
            "scope": "C2 producer, parameter resolver, parent loaders, author core, and qpsim parameter sources",
        },
        "sources": dict(_SOURCE_HASHES_AT_IMPORT),
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "evidence_class": "hybrid_component_substitution",
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
        },
        "steps": step_records,
    }
    _assert_source_snapshots()
    return metadata, arrays


def write_c2_bundle(
    c0_bundle_dir: Path,
    output_dir: Path,
    *,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
    c1_score_path: Path = DEFAULT_C1_SCORE,
) -> Path:
    """Write one immutable C2 raw bundle into a new directory."""

    _assert_source_snapshots()
    metadata, arrays = build_c2_bundle(
        c0_bundle_dir,
        c0_summary_path=c0_summary_path,
        c1_score_path=c1_score_path,
    )
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, object]] = {}
    for name, value in sorted(arrays.items()):
        content = _npy_bytes(value)
        filename = f"{name}.npy"
        with (root / filename).open("xb") as handle:
            handle.write(content)
        files[filename] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {
        "files": files,
        "metadata": metadata,
        "schema": SCHEMA,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_path = root / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _assert_source_snapshots()
    return manifest_path


def load_c2_raw_bundle(
    bundle_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
    """Strictly load a C2 raw bundle from single-read authenticated bytes."""

    root = bundle_dir.resolve()
    if bundle_dir.is_symlink() or not root.is_dir():
        raise C2BundleError("C2 raw bundle is missing, unsafe, or a symlink.")
    manifest_raw = _read_regular_file_once(root / "manifest.json", "C2 manifest")
    manifest_sha = hashlib.sha256(manifest_raw).hexdigest()
    manifest = _parse_json(manifest_raw, "C2 manifest")
    _exact_keys(manifest, {"files", "metadata", "schema"}, "C2 manifest")
    if manifest.get("schema") != SCHEMA:
        raise C2BundleError("C2 raw schema is unsupported.")
    metadata = _mapping(manifest.get("metadata"), "C2 metadata")
    _exact_keys(
        metadata,
        {
            "array_descriptors",
            "coordinate_contract",
            "frozen_inputs",
            "parameter_plan",
            "parent_bindings",
            "schema",
            "source_binding",
            "sources",
            "stage",
            "steps",
        },
        "C2 metadata",
    )
    if metadata.get("schema") != SCHEMA:
        raise C2BundleError("C2 metadata schema is unsupported.")
    descriptors = _mapping(metadata.get("array_descriptors"), "array_descriptors")
    files = _mapping(manifest.get("files"), "C2 files")
    expected_files = {f"{name}.npy" for name in descriptors}
    if set(files) != expected_files:
        raise C2BundleError("C2 raw file closure does not match its descriptors.")
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != {"manifest.json", *expected_files}:
        raise C2BundleError("C2 raw directory contains missing or extra entries.")

    arrays: dict[str, np.ndarray] = {}
    for name in sorted(descriptors):
        filename = f"{name}.npy"
        record = _mapping(files.get(filename), f"files.{filename}")
        _exact_keys(record, {"sha256", "size_bytes"}, f"files.{filename}")
        raw = _read_regular_file_once(root / filename, filename)
        size_bytes = record.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int):
            raise C2BundleError(f"files.{filename}.size_bytes is invalid.")
        if len(raw) != size_bytes:
            raise C2BundleError(f"{filename} size does not match its manifest.")
        if hashlib.sha256(raw).hexdigest() != _sha256(
            record.get("sha256"),
            f"files.{filename}.sha256",
        ):
            raise C2BundleError(f"{filename} SHA-256 does not match its manifest.")
        try:
            array = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
        except (ValueError, EOFError) as exc:
            raise C2BundleError(f"{filename} is not a safe NPY array.") from exc
        if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.floating):
            raise C2BundleError(f"{filename} must retain a real floating dtype.")
        if np.any(~np.isfinite(array)):
            raise C2BundleError(f"{filename} contains non-finite values.")
        canonical_raw = _npy_bytes(array)
        if raw != canonical_raw:
            raise C2BundleError(
                f"{filename} is not the canonical NPY encoding declared by this schema."
            )
        if not json_value_bit_exact(descriptors[name], array_descriptor(array)):
            raise C2BundleError(f"array_descriptors.{name} is forged or stale.")
        arrays[name] = array
    return metadata, arrays, manifest_sha


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-bundle", type=Path, required=True)
    parser.add_argument("--c0-summary", type=Path, default=DEFAULT_C0_SUMMARY)
    parser.add_argument("--c1-score", type=Path, default=DEFAULT_C1_SCORE)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c2_bundle(
            args.c0_bundle,
            args.output_dir,
            c0_summary_path=args.c0_summary,
            c1_score_path=args.c1_score,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
