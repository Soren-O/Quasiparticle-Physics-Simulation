"""Contracts for one-component-at-a-time paper-reproduction ladders."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from validation.source_provenance import source_sha256

LADDER_SCHEMA = "qpsim.reproduction-ladder.v1"
COMPONENTS = (
    "physical_model",
    "parameters",
    "grid_sampling",
    "photon_operator",
    "qp_phonon_operator",
    "phonon_balance",
    "nonlinear_solver",
    "observable",
    "io_capture",
)
EVIDENCE_CLASSES = {
    "author_supplied_reproduction_source",
    "clean_room_paper_equations",
    "clean_room_author_equivalent_numerics",
    "hybrid_component_substitution",
    "qpsim_author_semantics",
    "qpsim_model_extension",
}
STAGE_STATUSES = {
    "source_authenticated_not_replayed",
    "runnable",
    "planned",
    "blocked_missing_source",
    "completed",
}
_SATISFIED_PREREQUISITE_STATUSES = {
    "source_authenticated_not_replayed",
    "completed",
}
_ACTIVE_STAGE_STATUSES = {"runnable", "completed"}
_INTERNAL_COMPARISON_EVIDENCE_CLASSES = {
    "clean_room_author_equivalent_numerics",
    "hybrid_component_substitution",
    "qpsim_author_semantics",
    "qpsim_model_extension",
}
_COMPARISON_STAGE_ID = re.compile(r"^[A-Z][A-Z0-9_-]*\d[A-Z0-9_-]*$")
_IMPLEMENTATION_ROLE_MARKERS = {
    "adapter",
    "implementation",
    "model",
    "operator",
    "script",
    "solver",
}
_RESULT_ROLE_MARKERS = {
    "certificate",
    "comparison",
    "report",
    "result",
    "score",
    "snapshot",
}
_RESULT_SUFFIXES = {".csv", ".json", ".npy", ".npz", ".png", ".tsv", ".txt"}
_IMPLEMENTATION_SUFFIXES = {".json", ".py"}


class ReproductionLadderError(ValueError):
    """A reproduction ladder is malformed or scientifically ambiguous."""


@dataclass(frozen=True)
class StageDefinition:
    """One declared stage with a complete scientific component map."""

    stage_id: str
    branch: str
    parent_stage_id: str | None
    changed_component: str | None
    evidence_class: str
    status: str
    comparison_target: str
    comparison_stage_ids: tuple[str, ...]
    qualification: str
    components: dict[str, str]
    evidence: tuple[EvidenceBinding, ...]


@dataclass(frozen=True)
class EvidenceBinding:
    """One checked implementation or result artifact for a stage."""

    role: str
    path: str
    sha256: str
    hash_kind: str
    role_kind: str


@dataclass(frozen=True)
class ArrayDescriptor:
    """Exact native-array identity without dtype or shape coercion."""

    dtype: str
    shape: tuple[int, ...]
    npy_sha256: str


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReproductionLadderError(f"{label} must be an object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise ReproductionLadderError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, "
            f"got {sorted(mapping)!r}."
        )


def _string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ReproductionLadderError(f"{label}.{key} must be a nonempty string.")
    return value


def _optional_string(mapping: dict[str, Any], key: str, label: str) -> str | None:
    value = mapping.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise ReproductionLadderError(f"{label}.{key} must be null or a nonempty string.")
    return value


def _digest(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ReproductionLadderError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _safe_relative_path(value: str, label: str) -> Path:
    relative = Path(value)
    if (
        relative.is_absolute()
        or bool(relative.drive)
        or ".." in relative.parts
        or not relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise ReproductionLadderError(
            f"{label} must be a safe repository-relative path."
        )
    return relative


def _evidence_role_kind(role: str, path: str, label: str) -> str:
    """Classify legacy free-text roles without weakening status evidence gates."""

    words = set(re.findall(r"[a-z0-9]+", role.lower()))
    suffix = Path(path).suffix.lower()
    if {"authenticated", "source", "manifest"} <= words:
        if suffix != ".json":
            raise ReproductionLadderError(
                f"{label}.role declares an authenticated source manifest, "
                "but its path is not JSON."
            )
        return "authenticated_source_manifest"
    if words & _RESULT_ROLE_MARKERS:
        if suffix not in _RESULT_SUFFIXES:
            raise ReproductionLadderError(
                f"{label}.role declares a result artifact, but its path has "
                f"unsupported suffix {suffix!r}."
            )
        return "result"
    if words & _IMPLEMENTATION_ROLE_MARKERS:
        if suffix not in _IMPLEMENTATION_SUFFIXES:
            raise ReproductionLadderError(
                f"{label}.role declares implementation evidence, but its path "
                f"has unsupported suffix {suffix!r}."
            )
        return "implementation"
    return "supplemental"


def _evidence_bindings(raw: object, label: str) -> tuple[EvidenceBinding, ...]:
    if not isinstance(raw, list):
        raise ReproductionLadderError(f"{label} must be a list.")
    bindings: list[EvidenceBinding] = []
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    for index, item in enumerate(raw):
        item_label = f"{label}[{index}]"
        binding = _mapping(item, item_label)
        _exact_keys(
            binding,
            {"hash_kind", "path", "role", "sha256"},
            item_label,
        )
        path = _string(binding, "path", item_label)
        relative = _safe_relative_path(path, f"{item_label}.path")
        role = _string(binding, "role", item_label)
        path_key = relative.as_posix().casefold()
        role_key = role.casefold()
        if path_key in seen_paths or role_key in seen_roles:
            raise ReproductionLadderError(
                f"{item_label} must have a unique path and role."
            )
        seen_paths.add(path_key)
        seen_roles.add(role_key)
        bindings.append(
            EvidenceBinding(
                hash_kind=_string(binding, "hash_kind", item_label),
                role=role,
                path=path,
                sha256=_digest(binding.get("sha256"), f"{item_label}.sha256"),
                role_kind=_evidence_role_kind(role, path, item_label),
            )
        )
        if bindings[-1].hash_kind not in {"file_bytes", "source_canonical"}:
            raise ReproductionLadderError(
                f"{item_label}.hash_kind must be 'file_bytes' or "
                "'source_canonical'."
            )
    return tuple(bindings)


def _comparison_stage_ids(comparison_target: str) -> tuple[str, ...]:
    """Extract machine references from the leading token of legacy target text.

    Existing catalogs use ``"A0 ..."`` and ``"C7 and ..."``.  A leading
    stage-like identifier is therefore a dependency, while prose beginning
    with a lowercase word denotes an external oracle.
    """

    head = comparison_target.split(maxsplit=1)[0].rstrip(":")
    candidates = tuple(part for part in re.split(r"[,+]", head) if part)
    if candidates and all(_COMPARISON_STAGE_ID.fullmatch(part) for part in candidates):
        return candidates
    return ()


def _validate_evidence_for_status(
    *,
    stage_id: str,
    status: str,
    evidence_class: str,
    evidence: tuple[EvidenceBinding, ...],
    parent_stage_id: str | None,
) -> None:
    kinds = {binding.role_kind for binding in evidence}
    if status == "source_authenticated_not_replayed":
        if evidence_class != "author_supplied_reproduction_source":
            raise ReproductionLadderError(
                f"Stage {stage_id!r} can only use source-authenticated status "
                "for author-supplied reproduction source."
            )
        if parent_stage_id is not None:
            raise ReproductionLadderError(
                f"Source-authenticated stage {stage_id!r} must be a root stage."
            )
        if "authenticated_source_manifest" not in kinds:
            raise ReproductionLadderError(
                f"Stage {stage_id!r} needs authenticated-source-manifest evidence."
            )
    elif status == "runnable":
        if "implementation" not in kinds:
            raise ReproductionLadderError(
                f"Runnable stage {stage_id!r} needs implementation evidence."
            )
    elif status == "completed":
        missing = {"implementation", "result"} - kinds
        if missing:
            raise ReproductionLadderError(
                f"Completed stage {stage_id!r} lacks required evidence roles: "
                f"{sorted(missing)!r}."
            )


def _parse_stage(raw: object, index: int) -> StageDefinition:
    label = f"stages[{index}]"
    stage = _mapping(raw, label)
    _exact_keys(
        stage,
        {
            "branch",
            "changed_component",
            "comparison_target",
            "components",
            "evidence",
            "evidence_class",
            "parent_stage_id",
            "qualification",
            "stage_id",
            "status",
        },
        label,
    )
    evidence_class = _string(stage, "evidence_class", label)
    if evidence_class not in EVIDENCE_CLASSES:
        raise ReproductionLadderError(f"{label}.evidence_class is unsupported.")
    status = _string(stage, "status", label)
    if status not in STAGE_STATUSES:
        raise ReproductionLadderError(f"{label}.status is unsupported.")
    components = _mapping(stage["components"], f"{label}.components")
    if set(components) != set(COMPONENTS):
        raise ReproductionLadderError(
            f"{label}.components must declare every component exactly once."
        )
    if any(not isinstance(value, str) or not value for value in components.values()):
        raise ReproductionLadderError(f"{label}.components values must be nonempty strings.")
    changed_component = _optional_string(stage, "changed_component", label)
    if changed_component is not None and changed_component not in COMPONENTS:
        raise ReproductionLadderError(f"{label}.changed_component is unknown.")
    evidence = _evidence_bindings(stage["evidence"], f"{label}.evidence")
    if status in {"completed", "runnable", "source_authenticated_not_replayed"} and not evidence:
        raise ReproductionLadderError(
            f"{label}.evidence cannot be empty when status is {status!r}."
        )
    stage_id = _string(stage, "stage_id", label)
    parent_stage_id = _optional_string(stage, "parent_stage_id", label)
    comparison_target = _string(stage, "comparison_target", label)
    _validate_evidence_for_status(
        stage_id=stage_id,
        status=status,
        evidence_class=evidence_class,
        evidence=evidence,
        parent_stage_id=parent_stage_id,
    )
    comparison_stage_ids = _comparison_stage_ids(comparison_target)
    if (
        evidence_class in _INTERNAL_COMPARISON_EVIDENCE_CLASSES
        and not comparison_stage_ids
    ):
        raise ReproductionLadderError(
            f"Stage {stage_id!r} must begin comparison_target with a "
            "machine-readable stage ID."
        )
    return StageDefinition(
        stage_id=stage_id,
        branch=_string(stage, "branch", label),
        parent_stage_id=parent_stage_id,
        changed_component=changed_component,
        evidence_class=evidence_class,
        status=status,
        comparison_target=comparison_target,
        comparison_stage_ids=comparison_stage_ids,
        qualification=_string(stage, "qualification", label),
        components={key: str(components[key]) for key in COMPONENTS},
        evidence=evidence,
    )


def _verify_repository_binding(
    repository_root: Path,
    *,
    path: str,
    sha256: str,
    hash_kind: str,
    label: str,
) -> None:
    relative = _safe_relative_path(path, label)
    lexical_root = repository_root.absolute()
    lexical_candidate = lexical_root / relative

    # Check each unresolved component.  Calling ``resolve`` first erases the
    # fact that a file or an intermediate directory was reached via symlink.
    cursor = lexical_root
    if cursor.is_symlink():
        raise ReproductionLadderError(f"{label} is missing, unsafe, or a symlink.")
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ReproductionLadderError(f"{label} is missing, unsafe, or a symlink.")

    root = lexical_root.resolve()
    try:
        candidate = lexical_candidate.resolve(strict=True)
    except (FileNotFoundError, OSError):
        raise ReproductionLadderError(
            f"{label} is missing, unsafe, or a symlink."
        ) from None
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ReproductionLadderError(f"{label} is missing, unsafe, or a symlink.")
    actual = (
        hashlib.sha256(candidate.read_bytes()).hexdigest()
        if hash_kind == "file_bytes"
        else source_sha256(candidate)
    )
    if actual != sha256:
        raise ReproductionLadderError(f"{label} SHA-256 does not match live bytes.")


def _reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ReproductionLadderError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ReproductionLadderError(f"Non-finite JSON number {value!r}.")
    return parsed


def _reject_json_constant(value: str) -> None:
    raise ReproductionLadderError(f"Non-finite JSON constant {value!r}.")


def load_stage_catalog(
    path: Path,
    *,
    repository_root: Path | None = None,
) -> tuple[StageDefinition, ...]:
    """Strictly load and validate a reproduction-ladder JSON file.

    Duplicate object keys, JavaScript-style non-finite constants, overflowing
    finite-number literals, and symlinked catalog paths are rejected before
    scientific validation.
    """

    catalog_path = Path(path)
    if catalog_path.is_symlink():
        raise ReproductionLadderError("The reproduction-ladder path is a symlink.")
    try:
        raw = catalog_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReproductionLadderError(
            f"Cannot read reproduction ladder {catalog_path}: {exc}"
        ) from exc
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
        )
    except json.JSONDecodeError as exc:
        raise ReproductionLadderError(
            f"Invalid reproduction-ladder JSON: {exc}"
        ) from exc
    return validate_stage_catalog(payload, repository_root=repository_root)


def validate_stage_catalog(
    payload: object,
    *,
    repository_root: Path | None = None,
) -> tuple[StageDefinition, ...]:
    """Validate declarations and, when requested, every checked file binding."""

    catalog = _mapping(payload, "ladder")
    _exact_keys(catalog, {"claim", "ladder_id", "oracle", "schema", "stages"}, "ladder")
    if catalog.get("schema") != LADDER_SCHEMA:
        raise ReproductionLadderError("The reproduction-ladder schema is stale.")
    _string(catalog, "ladder_id", "ladder")
    _string(catalog, "claim", "ladder")
    oracle = _mapping(catalog["oracle"], "ladder.oracle")
    _exact_keys(
        oracle,
        {"hash_kind", "manifest_path", "manifest_sha256"},
        "ladder.oracle",
    )
    oracle_path = _string(oracle, "manifest_path", "ladder.oracle")
    _safe_relative_path(oracle_path, "ladder.oracle.manifest_path")
    oracle_sha = _digest(
        oracle.get("manifest_sha256"),
        "ladder.oracle.manifest_sha256",
    )
    oracle_hash_kind = _string(oracle, "hash_kind", "ladder.oracle")
    if oracle_hash_kind not in {"file_bytes", "source_canonical"}:
        raise ReproductionLadderError("ladder.oracle.hash_kind is unsupported.")
    if repository_root is not None:
        _verify_repository_binding(
            repository_root,
            path=oracle_path,
            sha256=oracle_sha,
            hash_kind=oracle_hash_kind,
            label="ladder.oracle",
        )

    raw_stages = catalog.get("stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        raise ReproductionLadderError("ladder.stages must be a nonempty list.")
    stages = tuple(_parse_stage(raw, index) for index, raw in enumerate(raw_stages))
    by_id: dict[str, StageDefinition] = {}
    for stage in stages:
        if stage.stage_id in by_id:
            raise ReproductionLadderError(f"Duplicate stage_id {stage.stage_id!r}.")
        if stage.parent_stage_id is None:
            if stage.changed_component is not None:
                raise ReproductionLadderError(
                    f"Root stage {stage.stage_id!r} cannot declare changed_component."
                )
        else:
            if stage.parent_stage_id not in by_id:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} must follow its declared parent."
                )
            parent = by_id[stage.parent_stage_id]
            if stage.branch != parent.branch:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} cannot cross branches through parentage."
                )
            differences = [
                component
                for component in COMPONENTS
                if stage.components[component] != parent.components[component]
            ]
            if differences != [stage.changed_component]:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} declares {stage.changed_component!r} "
                    f"but changed components are {differences!r}."
                )
            if stage.parent_stage_id not in stage.comparison_stage_ids:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} must identify parent "
                    f"{stage.parent_stage_id!r} in comparison_target."
                )

        for comparison_stage_id in stage.comparison_stage_ids:
            comparison_stage = by_id.get(comparison_stage_id)
            if comparison_stage is None:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} comparison_target references "
                    f"unknown or later stage {comparison_stage_id!r}."
                )
            if (
                stage.status in _ACTIVE_STAGE_STATUSES
                and comparison_stage.status
                not in _SATISFIED_PREREQUISITE_STATUSES
            ):
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} is {stage.status!r}, but comparison "
                    f"stage {comparison_stage_id!r} is "
                    f"{comparison_stage.status!r}."
                )

        if stage.parent_stage_id is not None and stage.status in _ACTIVE_STAGE_STATUSES:
            parent = by_id[stage.parent_stage_id]
            if parent.status not in _SATISFIED_PREREQUISITE_STATUSES:
                raise ReproductionLadderError(
                    f"Stage {stage.stage_id!r} is {stage.status!r}, but prerequisite "
                    f"parent {parent.stage_id!r} is {parent.status!r}."
                )
        by_id[stage.stage_id] = stage
        if repository_root is not None:
            for binding in stage.evidence:
                _verify_repository_binding(
                    repository_root,
                    path=binding.path,
                    sha256=binding.sha256,
                    hash_kind=binding.hash_kind,
                    label=f"stage {stage.stage_id!r} evidence {binding.role!r}",
                )
    return stages


def describe_array(value: np.ndarray) -> ArrayDescriptor:
    """Hash one native NumPy array in portable ``.npy`` form.

    Object arrays are forbidden because their pickle representation is neither
    a safe nor a language-independent scientific record.  No dtype, byte order,
    dimensionality, or complexness conversion is performed.
    """

    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ReproductionLadderError("Object arrays cannot enter a stage snapshot.")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(3, 0), allow_pickle=False)
    return ArrayDescriptor(
        dtype=array.dtype.str,
        shape=tuple(int(size) for size in array.shape),
        npy_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),
    )


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    """Compare equal-shape stage arrays without hiding representation changes."""

    ref = np.asarray(reference)
    got = np.asarray(candidate)
    ref_descriptor = describe_array(ref)
    got_descriptor = describe_array(got)
    if ref.shape != got.shape:
        raise ReproductionLadderError(
            f"Cannot compare shapes {ref.shape!r} and {got.shape!r}."
        )
    if ref.dtype.kind not in {"b", "i", "u", "f", "c"} or got.dtype.kind not in {
        "b",
        "i",
        "u",
        "f",
        "c",
    }:
        raise ReproductionLadderError("Stage comparisons require numeric or boolean arrays.")
    integer_comparison = ref.dtype.kind in {"b", "i", "u"} and got.dtype.kind in {
        "b",
        "i",
        "u",
    }
    if integer_comparison:
        differences: list[int] = []
        relative_values: list[float] = []
        for ref_item, got_item in zip(ref.flat, got.flat, strict=True):
            ref_integer = int(ref_item)
            difference = abs(int(got_item) - ref_integer)
            differences.append(difference)
            relative_values.append(
                0.0
                if difference == 0
                else (
                    float("inf")
                    if ref_integer == 0
                    else float(difference) / abs(ref_integer)
                )
            )
        max_absolute_error = float(max(differences, default=0))
        max_relative_error = max(relative_values, default=0.0)
        if differences:
            max_difference = max(differences)
            rms_error = (
                0.0
                if max_difference == 0
                else float(max_difference)
                * math.sqrt(
                    math.fsum(
                        (difference / max_difference) ** 2
                        for difference in differences
                    )
                    / len(differences)
                )
            )
        else:
            rms_error = 0.0
    else:
        comparison_dtype = (
            np.complex128 if np.iscomplexobj(ref) or np.iscomplexobj(got) else np.float64
        )
        ref_numeric = np.asarray(ref, dtype=comparison_dtype)
        got_numeric = np.asarray(got, dtype=comparison_dtype)
        if np.any(~np.isfinite(ref_numeric)) or np.any(~np.isfinite(got_numeric)):
            raise ReproductionLadderError("Stage comparison arrays must be finite.")
        absolute = np.abs(got_numeric - ref_numeric)
        scale = np.maximum(np.abs(ref_numeric), np.finfo(float).tiny)
        relative = absolute / scale
        max_absolute_error = float(np.max(absolute, initial=0.0))
        max_relative_error = float(np.max(relative, initial=0.0))
        rms_error = (
            float(np.sqrt(np.mean(np.square(absolute)))) if absolute.size else 0.0
        )
    return {
        "candidate": {
            "dtype": got_descriptor.dtype,
            "npy_sha256": got_descriptor.npy_sha256,
            "shape": list(got_descriptor.shape),
        },
        "exact_array_identity": ref_descriptor == got_descriptor,
        "max_absolute_error": max_absolute_error,
        "max_relative_error": max_relative_error,
        "reference": {
            "dtype": ref_descriptor.dtype,
            "npy_sha256": ref_descriptor.npy_sha256,
            "shape": list(ref_descriptor.shape),
        },
        "rms_error": rms_error,
    }
