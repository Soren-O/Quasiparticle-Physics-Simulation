"""Frozen-state Fig. 6 author-observable comparison.

This diagnostic deliberately does **not** run or replace a nonlinear solver.
It authenticates a completed :mod:`fig6_author_adapter` bundle, holds the
captured author ``(E, f)`` state fixed, and evaluates only qpsim's direct,
linearly interpolated gap observable on that state.

The author array stores occupations at left-edge nodes
``E_i = Delta + i h``.  Qpsim's ``samples="authors"`` API accepts those same
occupation samples while its coordinate argument carries cell centers.
Consequently the only coordinate view constructed here is
``E_centers = E_qp + h/2``; the occupation array is never interpolated,
projected, or coerced.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_adapter import (
    ADAPTER_DEPENDENCIES,
    ADAPTER_SCHEMA,
    ADAPTER_SOURCE_EXECUTION_BINDING,
    ADAPTER_SOURCE_HASH_KIND,
    SOURCE_ROLE_FILENAMES,
    SOURCE_ROLES,
    SUBPROCESS_WRAPPER_SHA256,
)
from validation.source_provenance import (
    canonical_source_bytes,
    canonical_source_text,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
OBSERVABLE_SOURCE = (
    REPOSITORY_ROOT / "qpsim" / "observables" / "gap_suppression.py"
)
AUTHOR_SOURCE_MANIFEST = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "author-source.json"
)
DIAGNOSTIC_SCHEMA = "qpsim.fischer2023.fig6-author-frozen-gap.v2"
ANCHOR_SCHEMA = "qpsim.fischer2023.fig6-author-point-anchor.v1"
# Preserve the exact binary64 expression used by the attachment's material
# table.  ``180.0e-6`` rounds one ULP higher and is therefore not an
# interchangeable spelling in a one-component observable substitution.
AUTHOR_DELTA0_EV = 180 * 10**-6
_REQUIRED_ARRAYS = {"E_qp", "f_final"}
_ADAPTER_ARRAYS = {
    "E_qp",
    "f_final",
    "initial_state",
    "n_phonon_final",
    "omega_phonon",
    "phonon_residual_author_s_inv",
    "qp_phonon_residual_author_s_inv",
    "qp_photon_residual_author_s_inv",
    "residual",
}
_ADAPTER_INPUT_KEYS = {
    "max_newton_steps",
    "mode",
    "num_energy_cells",
    "relative_step_threshold",
    "rng_seed",
    "sweep_index",
    "target_t_star_over_delta",
    "temperature_K",
}
_ANCHOR_PARAMETER_KEYS = {
    "actual_t_star_over_delta",
    "author_numerical_constants",
    "computed_n_bar",
    "h_eV",
    "rounded_gap_eV",
    "solved_photon_energy_eV",
}
_ANCHOR_RESULT_SCALAR_KEYS = {
    "gap_driven_eV",
    "gap_thermal_eV",
    "last_relative_qp_step",
    "max_power_mismatch",
    "newton_iterations",
    "normalized_numerical_observable",
}
_ADAPTER_METADATA_KEYS = {
    "actual_t_star_over_delta",
    "adapter",
    "all_finite",
    "array_origins",
    "author_driver_contract",
    "author_numerical_constants",
    "author_source",
    "c_photon_per_second",
    "computed_n_bar",
    "constructor_seconds",
    "executed_source",
    "execution_security",
    "f_max",
    "f_min",
    "gap_driven_eV",
    "gap_thermal_eV",
    "h_eV",
    "input",
    "last_relative_qp_step",
    "material_parameters",
    "max_power_mismatch",
    "n_ph_max",
    "n_ph_min",
    "newton_iterations",
    "newton_seconds",
    "normalized_numerical_observable",
    "photon_bin",
    "qualification",
    "rounded_gap_eV",
    "runtime",
    "schema",
    "scientific_status",
    "solved_photon_energy_eV",
    "subprocess_isolation",
}
_EXPECTED_ADAPTER_PATH = "validation/fischer_2023/fig6_author_adapter.py"
_EXPECTED_ADAPTER_DEPENDENCIES = {
    path.relative_to(REPOSITORY_ROOT).as_posix() for path in ADAPTER_DEPENDENCIES
}
_DIAGNOSTIC_SOURCE_BYTES_AT_IMPORT = canonical_source_bytes(Path(__file__))
_OBSERVABLE_SOURCE_BYTES_AT_IMPORT = canonical_source_bytes(OBSERVABLE_SOURCE)
_DIAGNOSTIC_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    _DIAGNOSTIC_SOURCE_BYTES_AT_IMPORT
).hexdigest()
_OBSERVABLE_SOURCE_SHA256_AT_IMPORT = hashlib.sha256(
    _OBSERVABLE_SOURCE_BYTES_AT_IMPORT
).hexdigest()


class FrozenStateDiagnosticError(ValueError):
    """A completed author bundle is malformed or no longer authenticated."""


@dataclass(frozen=True)
class LoadedAuthorPointBundle:
    """Authenticated native arrays and metadata from one adapter bundle."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]
    manifest_sha256: str
    files: dict[str, dict[str, object]]
    array_descriptors: dict[str, dict[str, object]]


@dataclass(frozen=True)
class CheckedRepositoryFile:
    """One repository file verified and retained as a single byte snapshot."""

    path: Path
    content: bytes
    canonical_sha256: str


def _read_regular_file_snapshot(path: Path, label: str) -> bytes:
    """Retain the exact bytes that are subsequently hashed and consumed."""

    if not path.is_file() or path.is_symlink():
        raise FrozenStateDiagnosticError(f"{label} is missing or is a symlink: {path}.")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FrozenStateDiagnosticError(f"Cannot read {label} {path}: {exc}") from exc


def _native_array_descriptor(value: np.ndarray) -> dict[str, object]:
    """Describe exact native bytes without depending on the ladder machinery."""

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FrozenStateDiagnosticError("Object arrays are forbidden.")
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        array,
        version=(3, 0),
        allow_pickle=False,
    )
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "shape": [int(size) for size in array.shape],
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrozenStateDiagnosticError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise FrozenStateDiagnosticError(
        f"Non-finite JSON number {value!r} is forbidden."
    )


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenStateDiagnosticError(f"{label} must be a JSON object.")
    return value


def _exact_keys(
    mapping: dict[str, Any],
    expected: set[str],
    label: str,
) -> None:
    if set(mapping) != expected:
        raise FrozenStateDiagnosticError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, "
            f"got {sorted(mapping)!r}."
        )


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FrozenStateDiagnosticError(
            f"{label} must be a lowercase SHA-256 digest."
        )
    return value


def _finite_scalar(mapping: dict[str, Any], key: str) -> float:
    raw: object = mapping.get(key)
    if isinstance(raw, bool) or not isinstance(
        raw,
        (int, float, np.integer, np.floating),
    ):
        raise FrozenStateDiagnosticError(f"metadata.{key} must be finite and real.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise FrozenStateDiagnosticError(
            f"metadata.{key} must be finite and real."
        ) from exc
    if not math.isfinite(value):
        raise FrozenStateDiagnosticError(f"metadata.{key} must be finite and real.")
    return value


def _safe_bundle_filename(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise FrozenStateDiagnosticError(f"{label} must be a nonempty filename.")
    candidate = Path(raw)
    if candidate.name != raw or raw in {".", ".."}:
        raise FrozenStateDiagnosticError(f"{label} must be a safe basename.")
    return raw


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = _read_regular_file_snapshot(path, "JSON manifest")
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenStateDiagnosticError(
            f"Cannot parse author bundle manifest {path}: {exc}"
        ) from exc
    return _mapping(payload, "author bundle manifest"), hashlib.sha256(raw).hexdigest()


def _parse_json_snapshot(raw: bytes, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FrozenStateDiagnosticError(f"Cannot parse {label}: {exc}") from exc
    return _mapping(payload, label)


def load_author_point_bundle(bundle_dir: Path) -> LoadedAuthorPointBundle:
    """Verify internal integrity of every declared file and array in a bundle.

    This is not an external authenticity check: the mutable manifest and its
    files could be rewritten together. Scientific comparisons must additionally
    call :func:`validate_bundle_anchor` with a separately reviewed anchor.
    """

    root = bundle_dir.resolve()
    if not root.is_dir():
        raise FrozenStateDiagnosticError(f"Author bundle is not a directory: {root}.")
    manifest, manifest_sha256 = _load_json(root / "manifest.json")
    _exact_keys(
        manifest,
        {
            "arrays",
            "files",
            "metadata",
            "stderr_sha256",
            "stdout_sha256",
        },
        "author bundle manifest",
    )
    metadata = _mapping(manifest["metadata"], "author bundle metadata")
    _exact_keys(metadata, _ADAPTER_METADATA_KEYS, "author bundle metadata")
    if metadata.get("schema") != ADAPTER_SCHEMA:
        raise FrozenStateDiagnosticError("Author bundle adapter schema is stale.")
    adapter = _mapping(metadata["adapter"], "metadata.adapter")
    _exact_keys(
        adapter,
        {
            "dependency_sha256s",
            "execution_binding",
            "hash_kind",
            "path",
            "sha256",
            "subprocess_wrapper_sha256",
        },
        "metadata.adapter",
    )
    if not isinstance(adapter.get("path"), str) or not adapter["path"]:
        raise FrozenStateDiagnosticError("metadata.adapter.path is invalid.")
    if adapter["path"] != _EXPECTED_ADAPTER_PATH:
        raise FrozenStateDiagnosticError(
            "metadata.adapter.path is not the checked Figure 6 adapter."
        )
    if adapter.get("hash_kind") != ADAPTER_SOURCE_HASH_KIND:
        raise FrozenStateDiagnosticError("metadata.adapter.hash_kind is invalid.")
    if adapter.get("execution_binding") != ADAPTER_SOURCE_EXECUTION_BINDING:
        raise FrozenStateDiagnosticError(
            "metadata.adapter.execution_binding is invalid."
        )
    _sha256(adapter.get("sha256"), "metadata.adapter.sha256")
    if (
        _sha256(
            adapter.get("subprocess_wrapper_sha256"),
            "metadata.adapter.subprocess_wrapper_sha256",
        )
        != SUBPROCESS_WRAPPER_SHA256
    ):
        raise FrozenStateDiagnosticError(
            "metadata.adapter subprocess wrapper is not the checked implementation."
        )
    dependencies = _mapping(
        adapter["dependency_sha256s"],
        "metadata.adapter.dependency_sha256s",
    )
    if set(dependencies) != _EXPECTED_ADAPTER_DEPENDENCIES:
        raise FrozenStateDiagnosticError(
            "metadata.adapter.dependency_sha256s does not identify the exact "
            "checked adapter dependency closure."
        )
    for dependency_path, dependency_sha in dependencies.items():
        if not isinstance(dependency_path, str) or not dependency_path:
            raise FrozenStateDiagnosticError(
                "metadata.adapter.dependency_sha256s has an invalid path."
            )
        _sha256(
            dependency_sha,
            f"metadata.adapter.dependency_sha256s[{dependency_path!r}]",
        )
    author_source = _mapping(metadata["author_source"], "metadata.author_source")
    _exact_keys(
        author_source,
        {"archive_sha256", "member_sha256s", "source_id"},
        "metadata.author_source",
    )
    _sha256(
        author_source.get("archive_sha256"),
        "metadata.author_source.archive_sha256",
    )
    if (
        not isinstance(author_source.get("source_id"), str)
        or not author_source["source_id"]
    ):
        raise FrozenStateDiagnosticError("metadata.author_source.source_id is invalid.")
    member_sha256s = _mapping(
        author_source["member_sha256s"],
        "metadata.author_source.member_sha256s",
    )
    if len(member_sha256s) != len(SOURCE_ROLES) + 1:
        raise FrozenStateDiagnosticError(
            "metadata.author_source.member_sha256s does not contain the exact "
            "entry-point plus solver-module closure."
        )
    for member_path, member_sha in member_sha256s.items():
        if not isinstance(member_path, str) or not member_path:
            raise FrozenStateDiagnosticError(
                "metadata.author_source.member_sha256s has an invalid path."
            )
        _sha256(
            member_sha,
            f"metadata.author_source.member_sha256s[{member_path!r}]",
        )
    expected_source_filenames = {
        "Figure_6.py",
        *SOURCE_ROLE_FILENAMES.values(),
    }
    if {Path(path).name for path in member_sha256s} != expected_source_filenames:
        raise FrozenStateDiagnosticError(
            "metadata.author_source.member_sha256s does not identify the "
            "canonical entry-point plus solver filenames."
        )
    if not isinstance(metadata["qualification"], str) or not metadata["qualification"]:
        raise FrozenStateDiagnosticError("metadata.qualification is invalid.")
    input_metadata = _mapping(metadata["input"], "metadata.input")
    _exact_keys(input_metadata, _ADAPTER_INPUT_KEYS, "metadata.input")
    array_origins = _mapping(metadata["array_origins"], "metadata.array_origins")
    _exact_keys(array_origins, _ADAPTER_ARRAYS, "metadata.array_origins")
    _mapping(metadata["author_driver_contract"], "metadata.author_driver_contract")
    executed_source = _mapping(
        metadata["executed_source"],
        "metadata.executed_source",
    )
    _exact_keys(
        executed_source,
        {"loading", "member_sha256s", "module_sha256s", "transformations"},
        "metadata.executed_source",
    )
    if executed_source.get("loading") != "immutable_child_byte_snapshots_v1":
        raise FrozenStateDiagnosticError(
            "metadata.executed_source.loading is not the checked snapshot loader."
        )
    executed_members = _mapping(
        executed_source["member_sha256s"],
        "metadata.executed_source.member_sha256s",
    )
    executed_modules = _mapping(
        executed_source["module_sha256s"],
        "metadata.executed_source.module_sha256s",
    )
    if len(executed_members) != len(SOURCE_ROLES):
        raise FrozenStateDiagnosticError(
            "metadata.executed_source.member_sha256s does not contain the "
            "exact solver-module closure."
        )
    for member_path, member_sha in executed_members.items():
        if not isinstance(member_path, str) or not member_path:
            raise FrozenStateDiagnosticError(
                "metadata.executed_source.member_sha256s has an invalid path."
            )
        _sha256(
            member_sha,
            f"metadata.executed_source.member_sha256s[{member_path!r}]",
        )
    if len(executed_modules) != len(SOURCE_ROLES):
        raise FrozenStateDiagnosticError(
            "metadata.executed_source.module_sha256s does not contain the "
            "exact solver-module closure."
        )
    for module_name, module_sha in executed_modules.items():
        if not isinstance(module_name, str) or not module_name.isidentifier():
            raise FrozenStateDiagnosticError(
                "metadata.executed_source.module_sha256s has an invalid "
                "module name."
            )
        _sha256(
            module_sha,
            f"metadata.executed_source.module_sha256s[{module_name!r}]",
        )
    expected_modules = {
        Path(member_path).stem: digest
        for member_path, digest in executed_members.items()
    }
    if executed_modules != expected_modules:
        raise FrozenStateDiagnosticError(
            "metadata.executed_source.module_sha256s does not match the "
            "executed solver-member closure."
        )
    transformations = executed_source["transformations"]
    if not isinstance(transformations, list) or any(
        not isinstance(item, dict) for item in transformations
    ):
        raise FrozenStateDiagnosticError(
            "metadata.executed_source.transformations must be a list of records."
        )
    if input_metadata.get("mode") == "author_equivalent":
        if transformations != []:
            raise FrozenStateDiagnosticError(
                "author_equivalent evidence must execute untransformed author source."
            )
        expected_executed_members = {
            path: digest
            for path, digest in member_sha256s.items()
            if Path(path).name in set(SOURCE_ROLE_FILENAMES.values())
        }
        if executed_members != expected_executed_members:
            raise FrozenStateDiagnosticError(
                "author_equivalent executed source digests do not equal the "
                "authenticated solver-member closure."
            )
    numerical_constants = _mapping(
        metadata["author_numerical_constants"],
        "metadata.author_numerical_constants",
    )
    _exact_keys(
        numerical_constants,
        {
            "boltzmann_constant_J_per_K",
            "electron_charge_C",
            "reduced_planck_constant_J_s",
        },
        "metadata.author_numerical_constants",
    )
    for constant_name in numerical_constants:
        if _finite_scalar(numerical_constants, constant_name) <= 0.0:
            raise FrozenStateDiagnosticError(
                f"metadata.author_numerical_constants.{constant_name} "
                "must be positive."
            )
    material_parameters = _mapping(
        metadata["material_parameters"],
        "metadata.material_parameters",
    )
    _exact_keys(
        material_parameters,
        {
            "T_c_K",
            "debye_energy_eV",
            "fermi_density_of_states_per_eV_m3",
            "ion_density_per_m3",
            "tau_0_s",
            "tau_0_pb_s",
            "tau_l_s",
            "zero_temperature_gap_eV",
        },
        "metadata.material_parameters",
    )
    runtime = _mapping(metadata["runtime"], "metadata.runtime")
    _exact_keys(
        runtime,
        {"matplotlib", "numba", "numpy", "platform", "python", "scipy"},
        "metadata.runtime",
    )
    isolation = _mapping(
        metadata["subprocess_isolation"],
        "metadata.subprocess_isolation",
    )
    _exact_keys(
        isolation,
        {"flags", "isolated_python", "os_sandbox", "user_site_enabled"},
        "metadata.subprocess_isolation",
    )
    if isolation != {
        "flags": ["-I", "-B"],
        "isolated_python": True,
        "os_sandbox": False,
        "user_site_enabled": False,
    }:
        raise FrozenStateDiagnosticError(
            "metadata.subprocess_isolation is not the checked isolated-Python "
            "execution contract."
        )

    raw_files = _mapping(manifest["files"], "author bundle files")
    files: dict[str, dict[str, object]] = {}
    file_snapshots: dict[str, bytes] = {}
    for raw_name, raw_descriptor in raw_files.items():
        name = _safe_bundle_filename(raw_name, f"files[{raw_name!r}]")
        descriptor = _mapping(raw_descriptor, f"files[{name!r}]")
        _exact_keys(descriptor, {"role", "sha256", "size_bytes"}, f"files[{name!r}]")
        role = descriptor.get("role")
        size = descriptor.get("size_bytes")
        if not isinstance(role, str) or not role:
            raise FrozenStateDiagnosticError(f"files[{name!r}].role is invalid.")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise FrozenStateDiagnosticError(f"files[{name!r}].size_bytes is invalid.")
        expected_sha = _sha256(descriptor.get("sha256"), f"files[{name!r}].sha256")
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise FrozenStateDiagnosticError(
                f"Declared author-bundle file is missing or a symlink: {name}."
            )
        content = _read_regular_file_snapshot(
            path,
            f"declared author-bundle file {name!r}",
        )
        if len(content) != size:
            raise FrozenStateDiagnosticError(
                f"Declared size does not match author-bundle file {name!r}."
            )
        if hashlib.sha256(content).hexdigest() != expected_sha:
            raise FrozenStateDiagnosticError(
                f"SHA-256 does not match author-bundle file {name!r}."
            )
        file_snapshots[name] = content
        files[name] = {
            "role": role,
            "sha256": expected_sha,
            "size_bytes": size,
        }

    actual_files = {path.name for path in root.iterdir()}
    expected_files = {"manifest.json", *files}
    if actual_files != expected_files:
        raise FrozenStateDiagnosticError(
            "Author bundle contains missing or unbound files: "
            f"expected {sorted(expected_files)!r}, got {sorted(actual_files)!r}."
        )

    for log_name in ("stdout", "stderr"):
        filename = f"{log_name}.txt"
        if filename not in files:
            raise FrozenStateDiagnosticError(f"Author bundle lacks {filename}.")
        log_sha = _sha256(
            manifest.get(f"{log_name}_sha256"),
            f"{log_name}_sha256",
        )
        if hashlib.sha256(file_snapshots[filename]).hexdigest() != log_sha:
            raise FrozenStateDiagnosticError(
                f"{log_name}_sha256 does not match {filename}."
            )

    raw_arrays = _mapping(manifest["arrays"], "author bundle arrays")
    _exact_keys(raw_arrays, _ADAPTER_ARRAYS, "author bundle arrays")
    arrays: dict[str, np.ndarray] = {}
    array_descriptors: dict[str, dict[str, object]] = {}
    for name, raw_descriptor in sorted(raw_arrays.items()):
        _safe_bundle_filename(f"{name}.npy", f"array {name!r}")
        descriptor = _mapping(raw_descriptor, f"arrays[{name!r}]")
        _exact_keys(
            descriptor,
            {"dtype", "npy_sha256", "shape"},
            f"arrays[{name!r}]",
        )
        dtype = descriptor.get("dtype")
        shape = descriptor.get("shape")
        if not isinstance(dtype, str) or not dtype:
            raise FrozenStateDiagnosticError(f"arrays[{name!r}].dtype is invalid.")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in shape
            )
        ):
            raise FrozenStateDiagnosticError(f"arrays[{name!r}].shape is invalid.")
        expected_npy_sha = _sha256(
            descriptor.get("npy_sha256"),
            f"arrays[{name!r}].npy_sha256",
        )
        filename = f"{name}.npy"
        file_descriptor = files.get(filename)
        role_prefix = f"array:{name}:"
        if (
            file_descriptor is None
            or not str(file_descriptor["role"]).startswith(role_prefix)
        ):
            raise FrozenStateDiagnosticError(
                f"Author bundle does not bind array {name!r} to {filename!r}."
            )
        try:
            value = np.load(io.BytesIO(file_snapshots[filename]), allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise FrozenStateDiagnosticError(
                f"Cannot load native author array {filename!r}: {exc}"
            ) from exc
        native = np.asarray(value)
        actual = _native_array_descriptor(native)
        if (
            actual["dtype"] != dtype
            or actual["shape"] != shape
            or actual["npy_sha256"] != expected_npy_sha
        ):
            raise FrozenStateDiagnosticError(
                f"Native array descriptor does not match {filename!r}."
            )
        arrays[name] = native
        array_descriptors[name] = actual

    return LoadedAuthorPointBundle(
        arrays=arrays,
        metadata=metadata,
        manifest_sha256=manifest_sha256,
        files=files,
        array_descriptors=array_descriptors,
    )


def _checked_repository_file(
    raw: object,
    sha: object,
    label: str,
) -> CheckedRepositoryFile:
    if not isinstance(raw, str) or not raw:
        raise FrozenStateDiagnosticError(f"{label}.path must be a nonempty string.")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise FrozenStateDiagnosticError(f"{label}.path must be repository-relative.")
    unresolved = REPOSITORY_ROOT / relative
    current = REPOSITORY_ROOT
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise FrozenStateDiagnosticError(
                f"{label}.path is missing, unsafe, or a symlink."
            )
    path = unresolved.resolve()
    if not path.is_relative_to(REPOSITORY_ROOT) or not path.is_file():
        raise FrozenStateDiagnosticError(f"{label}.path is missing, unsafe, or a symlink.")
    expected_sha = _sha256(sha, f"{label}.sha256")
    content = path.read_bytes()
    canonical = canonical_source_text(content.decode("utf-8")).encode("utf-8")
    actual_sha = hashlib.sha256(canonical).hexdigest()
    if actual_sha != expected_sha:
        raise FrozenStateDiagnosticError(
            f"{label}.sha256 does not match canonical live source."
        )
    return CheckedRepositoryFile(
        path=path,
        content=content,
        canonical_sha256=actual_sha,
    )


def validate_bundle_anchor(
    bundle: LoadedAuthorPointBundle,
    anchor_path: Path,
) -> dict[str, object]:
    """Bind a self-consistent bundle to a separately reviewed trust record."""

    anchor_input_path = anchor_path.absolute()
    anchor, anchor_sha256 = _load_json(anchor_input_path)
    anchor_resolved_path = anchor_input_path.resolve()
    _exact_keys(
        anchor,
        {
            "adapter",
            "anchor_id",
            "author_source_manifest",
            "bundle_manifest_sha256",
            "expected_input",
            "expected_parameters",
            "expected_result",
            "expected_scientific_status",
            "schema",
        },
        "author point anchor",
    )
    if anchor.get("schema") != ANCHOR_SCHEMA:
        raise FrozenStateDiagnosticError("Author point anchor schema is stale.")
    anchor_id = anchor.get("anchor_id")
    if not isinstance(anchor_id, str) or not anchor_id:
        raise FrozenStateDiagnosticError("author point anchor.anchor_id is invalid.")
    if (
        _sha256(
            anchor.get("bundle_manifest_sha256"),
            "author point anchor.bundle_manifest_sha256",
        )
        != bundle.manifest_sha256
    ):
        raise FrozenStateDiagnosticError(
            "Author bundle manifest is not the one bound by the reviewed anchor."
        )

    adapter = _mapping(anchor["adapter"], "author point anchor.adapter")
    _exact_keys(
        adapter,
        {
            "dependency_sha256s",
            "execution_binding",
            "hash_kind",
            "path",
            "sha256",
            "subprocess_wrapper_sha256",
        },
        "author point anchor.adapter",
    )
    if adapter.get("path") != _EXPECTED_ADAPTER_PATH:
        raise FrozenStateDiagnosticError(
            "author point anchor.adapter.path is not the canonical checked adapter."
        )
    if adapter.get("hash_kind") != ADAPTER_SOURCE_HASH_KIND:
        raise FrozenStateDiagnosticError(
            "author point anchor.adapter.hash_kind is invalid."
        )
    if adapter.get("execution_binding") != ADAPTER_SOURCE_EXECUTION_BINDING:
        raise FrozenStateDiagnosticError(
            "author point anchor.adapter.execution_binding is invalid."
        )
    adapter_snapshot = _checked_repository_file(
        adapter.get("path"),
        adapter.get("sha256"),
        "author point anchor.adapter",
    )
    if bundle.metadata.get("adapter") != adapter:
        raise FrozenStateDiagnosticError(
            "Author bundle adapter identity does not match the reviewed anchor."
    )
    adapter_dependencies = _mapping(
        adapter["dependency_sha256s"],
        "author point anchor.adapter.dependency_sha256s",
    )
    if set(adapter_dependencies) != _EXPECTED_ADAPTER_DEPENDENCIES:
        raise FrozenStateDiagnosticError(
            "author point anchor.adapter.dependency_sha256s does not bind the "
            "exact checked adapter dependency closure."
        )
    for dependency_path, dependency_sha in adapter_dependencies.items():
        _checked_repository_file(
            dependency_path,
            dependency_sha,
            f"author point anchor adapter dependency {dependency_path!r}",
        )

    source_binding = _mapping(
        anchor["author_source_manifest"],
        "author point anchor.author_source_manifest",
    )
    _exact_keys(
        source_binding,
        {"path", "sha256"},
        "author point anchor.author_source_manifest",
    )
    expected_source_manifest_path = AUTHOR_SOURCE_MANIFEST.relative_to(
        REPOSITORY_ROOT
    ).as_posix()
    if source_binding.get("path") != expected_source_manifest_path:
        raise FrozenStateDiagnosticError(
            "author point anchor.author_source_manifest.path is not canonical."
        )
    source_snapshot = _checked_repository_file(
        source_binding.get("path"),
        source_binding.get("sha256"),
        "author point anchor.author_source_manifest",
    )
    source_manifest = _parse_json_snapshot(
        source_snapshot.content,
        "author point anchor author-source manifest",
    )
    recorded_source = _mapping(
        bundle.metadata["author_source"],
        "metadata.author_source",
    )
    if recorded_source["archive_sha256"] != source_manifest["archive"]["sha256"]:
        raise FrozenStateDiagnosticError(
            "Author bundle archive identity does not match the checked source manifest."
        )
    if recorded_source["source_id"] != source_manifest["source_id"]:
        raise FrozenStateDiagnosticError(
            "Author bundle source_id does not match the checked source manifest."
        )
    members = source_manifest.get("members")
    if not isinstance(members, list):
        raise FrozenStateDiagnosticError(
            "Checked author-source manifest members are malformed."
        )
    required_roles = {"entry_point", *SOURCE_ROLES}
    declared_by_role: dict[str, tuple[str, str]] = {}
    for raw_member in members:
        member = _mapping(raw_member, "author-source manifest member")
        role = member.get("role")
        path = member.get("path")
        member_sha = member.get("sha256")
        if (
            role in required_roles
            and isinstance(path, str)
            and isinstance(member_sha, str)
        ):
            if role in declared_by_role:
                raise FrozenStateDiagnosticError(
                    f"Checked author-source manifest duplicates role {role!r}."
                )
            declared_by_role[role] = (path, member_sha)
    if set(declared_by_role) != required_roles:
        raise FrozenStateDiagnosticError(
            "Checked author-source manifest lacks the exact entry-point plus "
            "solver-role closure."
        )
    declared_members = dict(declared_by_role.values())
    if recorded_source["member_sha256s"] != declared_members:
        raise FrozenStateDiagnosticError(
            "Author bundle member identities do not equal the checked source "
            "role closure."
        )

    expected_input = _mapping(
        anchor["expected_input"],
        "author point anchor.expected_input",
    )
    _exact_keys(
        expected_input,
        _ADAPTER_INPUT_KEYS,
        "author point anchor.expected_input",
    )
    actual_input = _mapping(bundle.metadata["input"], "metadata.input")
    if actual_input != expected_input:
        raise FrozenStateDiagnosticError(
            "Author bundle input controls do not match the complete reviewed "
            "anchor contract."
        )
    expected_parameters = _mapping(
        anchor["expected_parameters"],
        "author point anchor.expected_parameters",
    )
    _exact_keys(
        expected_parameters,
        _ANCHOR_PARAMETER_KEYS,
        "author point anchor.expected_parameters",
    )
    actual_parameters = {
        key: bundle.metadata.get(key) for key in _ANCHOR_PARAMETER_KEYS
    }
    if actual_parameters != expected_parameters:
        raise FrozenStateDiagnosticError(
            "Author bundle grid/drive parameters do not match the reviewed anchor."
        )
    expected_result = _mapping(
        anchor["expected_result"],
        "author point anchor.expected_result",
    )
    _exact_keys(
        expected_result,
        {"arrays", *_ANCHOR_RESULT_SCALAR_KEYS},
        "author point anchor.expected_result",
    )
    expected_iterations = expected_result.get("newton_iterations")
    if (
        not isinstance(expected_iterations, int)
        or isinstance(expected_iterations, bool)
        or expected_iterations < 1
    ):
        raise FrozenStateDiagnosticError(
            "author point anchor.expected_result.newton_iterations must be "
            "a positive integer."
        )
    for key in _ANCHOR_RESULT_SCALAR_KEYS - {"newton_iterations"}:
        _finite_scalar(expected_result, key)
    actual_result = {
        key: bundle.metadata.get(key) for key in _ANCHOR_RESULT_SCALAR_KEYS
    }
    if actual_result != {
        key: expected_result[key] for key in _ANCHOR_RESULT_SCALAR_KEYS
    }:
        raise FrozenStateDiagnosticError(
            "Author bundle scalar result does not match the reviewed anchor."
        )
    expected_arrays = _mapping(
        expected_result["arrays"],
        "author point anchor.expected_result.arrays",
    )
    _exact_keys(
        expected_arrays,
        _ADAPTER_ARRAYS,
        "author point anchor.expected_result.arrays",
    )
    for array_name, raw_descriptor in expected_arrays.items():
        descriptor = _mapping(
            raw_descriptor,
            f"author point anchor.expected_result.arrays[{array_name!r}]",
        )
        _exact_keys(
            descriptor,
            {"dtype", "npy_sha256", "shape"},
            f"author point anchor.expected_result.arrays[{array_name!r}]",
        )
        if not isinstance(descriptor.get("dtype"), str) or not descriptor["dtype"]:
            raise FrozenStateDiagnosticError(
                f"author point anchor expected array {array_name!r} has an "
                "invalid dtype."
            )
        shape = descriptor.get("shape")
        if (
            not isinstance(shape, list)
            or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0
                for size in shape
            )
        ):
            raise FrozenStateDiagnosticError(
                f"author point anchor expected array {array_name!r} has an "
                "invalid shape."
            )
        _sha256(
            descriptor.get("npy_sha256"),
            (
                "author point anchor.expected_result."
                f"arrays[{array_name!r}].npy_sha256"
            ),
        )
    if expected_arrays != bundle.array_descriptors:
        raise FrozenStateDiagnosticError(
            "Author bundle array result does not match the reviewed anchor."
        )
    expected_status = anchor.get("expected_scientific_status")
    if expected_status not in {"finite_state", "nonfinite_author_state"}:
        raise FrozenStateDiagnosticError(
            "author point anchor.expected_scientific_status is invalid."
        )
    if bundle.metadata.get("scientific_status") != expected_status:
        raise FrozenStateDiagnosticError(
            "Author bundle scientific status does not match the reviewed anchor."
        )
    return {
        "anchor_id": anchor_id,
        "anchor_path": anchor_resolved_path.relative_to(REPOSITORY_ROOT).as_posix()
        if anchor_resolved_path.is_relative_to(REPOSITORY_ROOT)
        else str(anchor_resolved_path),
        "anchor_sha256": anchor_sha256,
        "adapter_path": adapter_snapshot.path.relative_to(REPOSITORY_ROOT).as_posix(),
        "author_source_manifest_path": source_snapshot.path.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
    }


def _descriptor(value: np.ndarray) -> dict[str, object]:
    return _native_array_descriptor(value)


def _freeze_loaded_source_identities() -> dict[str, str]:
    current = {
        "diagnostic": canonical_source_bytes(Path(__file__)),
        "qpsim_direct_gap_observable": canonical_source_bytes(OBSERVABLE_SOURCE),
    }
    imported = {
        "diagnostic": _DIAGNOSTIC_SOURCE_BYTES_AT_IMPORT,
        "qpsim_direct_gap_observable": _OBSERVABLE_SOURCE_BYTES_AT_IMPORT,
    }
    if current != imported:
        raise FrozenStateDiagnosticError(
            "Loaded diagnostic/observable code no longer matches its on-disk "
            "source; launch a fresh Python process."
        )
    return {
        name: hashlib.sha256(content).hexdigest()
        for name, content in current.items()
    }


def _assert_sources_unchanged(identities: dict[str, str]) -> None:
    current = {
        "diagnostic": hashlib.sha256(
            canonical_source_bytes(Path(__file__))
        ).hexdigest(),
        "qpsim_direct_gap_observable": hashlib.sha256(
            canonical_source_bytes(OBSERVABLE_SOURCE)
        ).hexdigest(),
    }
    if current != identities:
        raise FrozenStateDiagnosticError(
            "Diagnostic/observable source changed during the frozen-state "
            "calculation; no evidence was emitted."
        )


def build_frozen_gap_comparison(
    bundle_dir: Path,
    *,
    anchor_path: Path,
) -> dict[str, Any]:
    """Evaluate only qpsim's direct gap observable on author-defined states.

    Both the captured driven distribution and the deterministic author thermal
    distribution are evaluated.  Every grid and scalar parameter, including
    the exact binary64 ``Delta_0`` value, is inherited from the authenticated
    parent bundle.
    """

    # Keep the bundle/anchor loader importable by the qpsim-free C0 path.
    # Qpsim enters only when this observable-substitution function is called.
    from qpsim.observables.gap_suppression import (
        gap_from_distribution_direct,
        gap_integral_from_distribution_direct,
    )

    source_identities = _freeze_loaded_source_identities()
    bundle = load_author_point_bundle(bundle_dir)
    anchor_binding = validate_bundle_anchor(bundle, anchor_path)
    E_qp = np.asarray(bundle.arrays["E_qp"])
    f_final = np.asarray(bundle.arrays["f_final"])
    if E_qp.ndim != 1 or f_final.ndim != 1 or E_qp.shape != f_final.shape:
        raise FrozenStateDiagnosticError(
            "E_qp and f_final must be equal-length one-dimensional arrays."
        )
    if E_qp.size < 2:
        raise FrozenStateDiagnosticError("Frozen-state diagnostic needs at least two nodes.")
    if not (
        np.issubdtype(E_qp.dtype, np.floating)
        and np.issubdtype(f_final.dtype, np.floating)
    ):
        raise FrozenStateDiagnosticError(
            "E_qp and f_final must retain floating-point author storage."
        )
    if np.any(~np.isfinite(E_qp)) or np.any(~np.isfinite(f_final)):
        raise FrozenStateDiagnosticError("E_qp and f_final must contain finite values.")
    if np.any((f_final < 0.0) | (f_final > 1.0)):
        raise FrozenStateDiagnosticError("f_final contains occupations outside [0, 1].")

    h_eV = _finite_scalar(bundle.metadata, "h_eV")
    gap_eV = _finite_scalar(bundle.metadata, "rounded_gap_eV")
    if h_eV <= 0.0 or gap_eV <= 0.0:
        raise FrozenStateDiagnosticError("h_eV and rounded_gap_eV must be positive.")
    spacing = np.diff(E_qp)
    coordinate_tol = (
        64.0
        * np.finfo(float).eps
        * max(float(np.max(np.abs(E_qp))), abs(h_eV), np.finfo(float).tiny)
    )
    if np.any(spacing <= 0.0) or not np.allclose(
        spacing,
        h_eV,
        rtol=64.0 * np.finfo(float).eps,
        atol=coordinate_tol,
    ):
        raise FrozenStateDiagnosticError(
            "Captured E_qp is not uniform with the recorded h_eV."
        )
    if not math.isclose(
        float(E_qp[0]),
        gap_eV,
        rel_tol=64.0 * np.finfo(float).eps,
        abs_tol=coordinate_tol,
    ):
        raise FrozenStateDiagnosticError(
            "Captured E_qp does not start at the recorded author gap."
        )

    # Qpsim represents the author's nodes as the left edges of these carrier
    # cells.  No occupation interpolation or projection occurs.
    qpsim_coordinate_carrier = E_qp + 0.5 * h_eV
    material_parameters = _mapping(
        bundle.metadata["material_parameters"],
        "metadata.material_parameters",
    )
    numerical_constants = _mapping(
        bundle.metadata["author_numerical_constants"],
        "metadata.author_numerical_constants",
    )
    input_metadata = _mapping(bundle.metadata["input"], "metadata.input")
    delta0_eV = _finite_scalar(
        material_parameters,
        "zero_temperature_gap_eV",
    )
    temperature_K = _finite_scalar(input_metadata, "temperature_K")
    boltzmann_constant = _finite_scalar(
        numerical_constants,
        "boltzmann_constant_J_per_K",
    )
    electron_charge = _finite_scalar(
        numerical_constants,
        "electron_charge_C",
    )
    if (
        delta0_eV <= 0.0
        or temperature_K <= 0.0
        or boltzmann_constant <= 0.0
        or electron_charge <= 0.0
    ):
        raise FrozenStateDiagnosticError(
            "Inherited gap, temperature, and numerical constants must be positive."
        )

    exponent = E_qp * electron_charge / (boltzmann_constant * temperature_K)
    with np.errstate(over="ignore", under="ignore", invalid="raise"):
        thermal_f = 1.0 / (np.exp(exponent) + 1.0)

    qpsim_integral = gap_integral_from_distribution_direct(
        f_final,
        qpsim_coordinate_carrier,
        gap=gap_eV,
        samples="authors",
    )
    qpsim_gap = gap_from_distribution_direct(
        f_final,
        qpsim_coordinate_carrier,
        gap=gap_eV,
        delta0=delta0_eV,
        samples="authors",
    )
    qpsim_thermal_integral = gap_integral_from_distribution_direct(
        thermal_f,
        qpsim_coordinate_carrier,
        gap=gap_eV,
        samples="authors",
    )
    qpsim_thermal_gap = gap_from_distribution_direct(
        thermal_f,
        qpsim_coordinate_carrier,
        gap=gap_eV,
        delta0=delta0_eV,
        samples="authors",
    )

    author_gap = _finite_scalar(bundle.metadata, "gap_driven_eV")
    author_thermal_gap = _finite_scalar(bundle.metadata, "gap_thermal_eV")
    author_normalized = _finite_scalar(
        bundle.metadata,
        "normalized_numerical_observable",
    )
    if author_gap <= 0.0 or author_thermal_gap <= 0.0:
        raise FrozenStateDiagnosticError("Recorded author gaps must be positive.")
    if author_gap > delta0_eV or author_thermal_gap > delta0_eV:
        raise FrozenStateDiagnosticError(
            "Recorded author gap exceeds the inherited zero-temperature gap."
        )
    denominator = gap_eV - qpsim_thermal_gap
    if denominator == 0.0:
        raise FrozenStateDiagnosticError(
            "Recomputed thermal reference makes the normalized observable singular."
        )
    author_inferred_integral = -math.log(author_gap / delta0_eV)
    author_inferred_thermal_integral = -math.log(author_thermal_gap / delta0_eV)
    qpsim_normalized = (qpsim_gap - qpsim_thermal_gap) / denominator
    gap_signed = qpsim_gap - author_gap
    thermal_gap_signed = qpsim_thermal_gap - author_thermal_gap
    integral_signed = qpsim_integral - author_inferred_integral
    thermal_integral_signed = (
        qpsim_thermal_integral - author_inferred_thermal_integral
    )
    normalized_signed = qpsim_normalized - author_normalized

    payload = {
        "calculation": {
            "author_reported": {
                "direct_gap_eV": author_gap,
                "inferred_direct_integral": author_inferred_integral,
                "inferred_thermal_integral": author_inferred_thermal_integral,
                "normalized_numerical_observable": author_normalized,
                "thermal_gap_eV": author_thermal_gap,
            },
            "deltas": {
                "direct_gap_absolute_eV": abs(gap_signed),
                "direct_gap_relative": gap_signed / author_gap,
                "direct_gap_signed_eV": gap_signed,
                "direct_gap_signed_ulps": gap_signed / math.ulp(author_gap),
                "direct_integral_absolute": abs(integral_signed),
                "direct_integral_signed": integral_signed,
                "normalized_observable_absolute": abs(normalized_signed),
                "normalized_observable_signed": normalized_signed,
                "thermal_gap_absolute_eV": abs(thermal_gap_signed),
                "thermal_gap_signed_eV": thermal_gap_signed,
                "thermal_gap_signed_ulps": (
                    thermal_gap_signed / math.ulp(author_thermal_gap)
                ),
                "thermal_integral_absolute": abs(thermal_integral_signed),
                "thermal_integral_signed": thermal_integral_signed,
            },
            "parameters": {
                "boltzmann_constant_J_per_K": boltzmann_constant,
                "delta0_eV": delta0_eV,
                "electron_charge_C": electron_charge,
                "gap_eV": gap_eV,
                "samples": "authors",
                "temperature_K": temperature_K,
            },
            "qpsim_recomputed": {
                "direct_gap_eV": qpsim_gap,
                "direct_integral": qpsim_integral,
                "normalized_numerical_observable": qpsim_normalized,
                "thermal_gap_eV": qpsim_thermal_gap,
                "thermal_integral": qpsim_thermal_integral,
            },
        },
        "coordinate_contract": {
            "captured_E_semantics": "author left-edge nodes E_i = Delta + i*h",
            "captured_E_qp": _descriptor(E_qp),
            "captured_f_final": _descriptor(f_final),
            "coordinate_transform": "E_bins_for_qpsim = E_qp + 0.5*h_eV",
            "derived_qpsim_coordinate_carrier": _descriptor(
                qpsim_coordinate_carrier
            ),
            "derived_thermal_f": _descriptor(thermal_f),
            "h_eV": h_eV,
            "occupation_transform": "none",
            "qpsim_api_semantics": (
                "samples='authors': f values are left-edge samples; E_bins "
                "carries cell centers whose reconstructed left edges are E_qp"
            ),
            "thermal_distribution": (
                "author Fermi function evaluated on the unchanged left-edge "
                "grid with inherited k_B, e, T, and kinetic gap"
            ),
        },
        "diagnostic_id": "fischer-catelani-2023-fig6-author-frozen-gap-v1",
        "evidence_class": "frozen_state_observable_substitution_diagnostic",
        "input_bundle": {
            "anchor": anchor_binding,
            "adapter_schema": bundle.metadata["schema"],
            "array_descriptors": bundle.array_descriptors,
            "author_source": bundle.metadata["author_source"],
            "files": bundle.files,
            "manifest_sha256": bundle.manifest_sha256,
            "qualification": bundle.metadata["qualification"],
        },
        "qualification": (
            "Frozen-state observable diagnostic only: the separately anchored "
            "author E/f bundle is unchanged and no kinetic operator or nonlinear "
            "solver is run or replaced. Qpsim's direct observable is applied to "
            "both the driven state and the deterministic author thermal state "
            "using the exact inherited binary64 parameters. This record remains "
            "separate from the nonlinear component-replacement ladder."
        ),
        "runtime": {
            "numpy_version": np.__version__,
            "platform": platform.platform(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "python_cache_tag": sys.implementation.cache_tag,
        },
        "schema": DIAGNOSTIC_SCHEMA,
        "sources": {
            "diagnostic": {
                "execution_binding": (
                    "import-time on-disk source snapshot, required unchanged "
                    "before and after the calculation; not an independent "
                    "interpreter-bytecode attestation"
                ),
                "hash_kind": "canonical_sha256_import_time_disk_snapshot",
                "path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": source_identities["diagnostic"],
            },
            "qpsim_direct_gap_observable": {
                "api": (
                    "qpsim.observables.gap_suppression."
                    "gap_from_distribution_direct"
                ),
                "execution_binding": (
                    "import-time on-disk source snapshot, required unchanged "
                    "before and after the calculation; not an independent "
                    "interpreter-bytecode attestation"
                ),
                "hash_kind": "canonical_sha256_import_time_disk_snapshot",
                "path": OBSERVABLE_SOURCE.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": source_identities["qpsim_direct_gap_observable"],
            },
        },
    }
    _assert_sources_unchanged(source_identities)
    return payload


def write_frozen_gap_comparison(
    bundle_dir: Path,
    output_path: Path,
    *,
    anchor_path: Path,
) -> Path:
    """Write one deterministic, provenance-bound comparison manifest."""

    payload = build_frozen_gap_comparison(bundle_dir, anchor_path=anchor_path)
    resolved = output_path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return resolved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = _parse_args()
    print(
        write_frozen_gap_comparison(
            args.bundle,
            args.output,
            anchor_path=args.anchor,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
