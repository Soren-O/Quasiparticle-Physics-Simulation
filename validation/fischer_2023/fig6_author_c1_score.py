"""Score the isolated qpsim observable substitution on accepted C0 states."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.observables.gap_suppression import (
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
)

from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY as DEFAULT_C0_SUMMARY,
)
from validation.fischer_2023.fig6_author_c0_summary import (
    load_c0_raw_bundle,
    load_c0_summary,
)
from validation.source_provenance import source_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c1-observable-score.v1"
DEFAULT_SCORE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "c1-observable-score.json"
)
STAGE_ID = "C1"
PARENT_STAGE_ID = "C0"
CHANGED_COMPONENT = "observable"
_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "qpsim" / "observables" / "gap_suppression.py",
)
_SCORE_KEYS = {
    "acceptance",
    "calculation",
    "c0_binding",
    "input_identity",
    "limitations",
    "schema",
    "sources",
    "stage",
}
_ACCEPTANCE_LIMITS = {
    "integral_absolute_error_max": 1.0e-15,
}


class C1ScoreError(ValueError):
    """The C1 parent evidence or checked score is malformed."""


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise C1ScoreError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _finite_json(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise C1ScoreError(f"Non-finite JSON number {value!r}.")
    return result


def _parse_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                C1ScoreError(f"Non-finite JSON constant {token!r}.")
            ),
            parse_float=_finite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C1ScoreError(f"Cannot parse {label}: {exc}.") from exc
    if not isinstance(value, dict):
        raise C1ScoreError(f"{label} must be a JSON object.")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C1ScoreError(f"{label} must be an object.")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise C1ScoreError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise C1ScoreError(f"{label} must be a finite real scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C1ScoreError(f"{label} must be a finite real scalar.") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise C1ScoreError(f"{label} must be finite and valid.")
    return result


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _file_sha256(path: Path, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise C1ScoreError(f"{label} is missing, unsafe, or a symlink.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_c1_score(
    c0_bundle_dir: Path,
    *,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
) -> dict[str, Any]:
    """Apply only qpsim's direct observable to exact accepted C0 arrays."""

    c0_summary_path = c0_summary_path.resolve()
    raw_metadata, arrays, raw_manifest_sha = load_c0_raw_bundle(c0_bundle_dir)
    c0_summary = load_c0_summary(c0_summary_path)
    c0_raw_binding = _mapping(c0_summary.get("raw_bundle"), "C0 raw_bundle")
    if c0_raw_binding.get("manifest_sha256") != raw_manifest_sha:
        raise C1ScoreError("C1 raw C0 bundle does not match the accepted C0 score.")
    if c0_summary.get("array_descriptors") != raw_metadata.get("array_descriptors"):
        raise C1ScoreError("C1 C0 arrays do not match the accepted C0 descriptors.")
    c0_stage = _mapping(c0_summary.get("stage"), "C0 stage")
    if c0_stage.get("stage_id") != PARENT_STAGE_ID:
        raise C1ScoreError("C1 parent summary is not stage C0.")
    acceptance = _mapping(c0_summary.get("acceptance"), "C0 acceptance")
    if acceptance.get("accepted") is not True:
        raise C1ScoreError("C1 parent C0 evidence is not accepted.")

    parameters = _mapping(c0_summary.get("parameters"), "C0 parameters")
    gap = _finite(parameters.get("gap_eV"), "gap_eV", positive=True)
    h = _finite(parameters.get("h_eV"), "h_eV", positive=True)
    delta0 = _finite(parameters.get("delta0_eV"), "delta0_eV", positive=True)
    E_left = arrays["E_left_eV"]
    driven_f = arrays["f_final"]
    thermal_f = arrays["thermal_f"]
    inherited_descriptors = {
        "E_left_eV": _descriptor(E_left),
        "f_final": _descriptor(driven_f),
        "thermal_f": _descriptor(thermal_f),
    }
    expected_descriptors = _mapping(
        c0_summary.get("array_descriptors"),
        "C0 array_descriptors",
    )
    for name, descriptor in inherited_descriptors.items():
        if expected_descriptors.get(name) != descriptor:
            raise C1ScoreError(f"C1 did not inherit exact C0 {name} bytes.")

    # This carrier is the only coordinate adaptation: qpsim reconstructs its
    # left edges as E_bins-h/2, exactly recovering E_left.
    coordinate_carrier = E_left + 0.5 * h
    driven_integral = gap_integral_from_distribution_direct(
        driven_f,
        coordinate_carrier,
        gap=gap,
        samples="authors",
    )
    driven_gap = gap_from_distribution_direct(
        driven_f,
        coordinate_carrier,
        gap=gap,
        delta0=delta0,
        samples="authors",
    )
    thermal_integral = gap_integral_from_distribution_direct(
        thermal_f,
        coordinate_carrier,
        gap=gap,
        samples="authors",
    )
    thermal_gap = gap_from_distribution_direct(
        thermal_f,
        coordinate_carrier,
        gap=gap,
        delta0=delta0,
        samples="authors",
    )
    denominator = gap - thermal_gap
    if denominator == 0.0:
        raise C1ScoreError("C1 normalized observable has a zero denominator.")
    ordinate = (driven_gap - thermal_gap) / denominator

    # Recheck byte identity after the calls. A mutating observable would make
    # the comparison scientifically invalid even if its scalar happened to pass.
    for name, array in (
        ("E_left_eV", E_left),
        ("f_final", driven_f),
        ("thermal_f", thermal_f),
    ):
        if _descriptor(array) != inherited_descriptors[name]:
            raise C1ScoreError(f"qpsim observable mutated inherited C0 {name}.")

    author = _mapping(c0_summary.get("observable"), "C0 observable")
    author_driven_integral = _finite(
        author.get("driven_integral"),
        "C0 driven_integral",
    )
    author_thermal_integral = _finite(
        author.get("thermal_integral"),
        "C0 thermal_integral",
    )
    author_driven_gap = _finite(
        author.get("driven_gap_eV"),
        "C0 driven_gap_eV",
    )
    author_thermal_gap = _finite(
        author.get("thermal_gap_eV"),
        "C0 thermal_gap_eV",
    )
    author_ordinate = _finite(
        author.get("figure6_ordinate"),
        "C0 figure6_ordinate",
    )
    differences = {
        "driven_gap_signed_eV": driven_gap - author_driven_gap,
        "driven_gap_signed_ulps": ((driven_gap - author_driven_gap) / math.ulp(author_driven_gap)),
        "driven_integral_signed": driven_integral - author_driven_integral,
        "figure6_ordinate_signed": ordinate - author_ordinate,
        "thermal_gap_signed_eV": thermal_gap - author_thermal_gap,
        "thermal_gap_signed_ulps": (
            (thermal_gap - author_thermal_gap) / math.ulp(author_thermal_gap)
        ),
        "thermal_integral_signed": thermal_integral - author_thermal_integral,
    }
    checks = {
        "driven_gap_bit_exact": driven_gap == author_driven_gap,
        "driven_integral_within_limit": (
            abs(differences["driven_integral_signed"])
            <= _ACCEPTANCE_LIMITS["integral_absolute_error_max"]
        ),
        "figure6_ordinate_bit_exact": ordinate == author_ordinate,
        "input_arrays_unchanged": True,
        "thermal_gap_bit_exact": thermal_gap == author_thermal_gap,
        "thermal_integral_within_limit": (
            abs(differences["thermal_integral_signed"])
            <= _ACCEPTANCE_LIMITS["integral_absolute_error_max"]
        ),
    }
    accepted = all(checks.values())
    if not accepted:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise C1ScoreError(f"C1 observable acceptance checks failed: {failed!r}.")

    return {
        "acceptance": {
            "accepted": accepted,
            "checks": checks,
            "limits": dict(_ACCEPTANCE_LIMITS),
        },
        "calculation": {
            "author_c0": {
                "driven_gap_eV": author_driven_gap,
                "driven_integral": author_driven_integral,
                "figure6_ordinate": author_ordinate,
                "thermal_gap_eV": author_thermal_gap,
                "thermal_integral": author_thermal_integral,
            },
            "differences": differences,
            "qpsim_c1": {
                "driven_gap_eV": driven_gap,
                "driven_integral": driven_integral,
                "figure6_ordinate": ordinate,
                "thermal_gap_eV": thermal_gap,
                "thermal_integral": thermal_integral,
            },
        },
        "c0_binding": {
            "raw_manifest_sha256": raw_manifest_sha,
            "summary_path": c0_summary_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "summary_sha256": _file_sha256(c0_summary_path, "C0 summary"),
        },
        "input_identity": {
            "coordinate_carrier": _descriptor(coordinate_carrier),
            "inherited_arrays": inherited_descriptors,
            "parameter_hex": {
                "delta0_eV": delta0.hex(),
                "gap_eV": gap.hex(),
                "h_eV": h.hex(),
            },
            "parameters": {
                "delta0_eV": delta0,
                "gap_eV": gap,
                "h_eV": h,
                "samples": "authors",
            },
            "transform": "coordinate_carrier = E_left_eV + h_eV/2; no f projection",
        },
        "limitations": {
            "scope": "one accepted C0 point only",
            "statement": (
                "C1 validates only the direct observable substitution on fixed "
                "C0 driven and thermal states; it does not validate qpsim kinetics."
            ),
        },
        "schema": SCHEMA,
        "sources": {
            path.relative_to(REPOSITORY_ROOT).as_posix(): source_sha256(path)
            for path in _SOURCE_PATHS
        },
        "stage": {
            "changed_component": CHANGED_COMPONENT,
            "comparison_stage_id": PARENT_STAGE_ID,
            "parent_stage_id": PARENT_STAGE_ID,
            "stage_id": STAGE_ID,
            "status": "completed",
        },
    }


def canonical_score_bytes(score: dict[str, Any]) -> bytes:
    """Serialize a checked C1 score deterministically."""

    return (json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def write_c1_score(
    c0_bundle_dir: Path,
    output_path: Path,
    *,
    c0_summary_path: Path = DEFAULT_C0_SUMMARY,
) -> Path:
    """Build and exclusively write one checked C1 score."""

    score = build_c1_score(
        c0_bundle_dir,
        c0_summary_path=c0_summary_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        handle.write(canonical_score_bytes(score))
    return output_path


def load_c1_score(path: Path = DEFAULT_SCORE) -> dict[str, Any]:
    """Strictly load and repository-bind a checked C1 score."""

    if path.is_symlink() or not path.is_file():
        raise C1ScoreError("Checked C1 score is missing, unsafe, or a symlink.")
    score = _parse_json(path.read_bytes(), "checked C1 score")
    if set(score) != _SCORE_KEYS:
        raise C1ScoreError("Checked C1 score fields are invalid.")
    if score.get("schema") != SCHEMA:
        raise C1ScoreError("Checked C1 score schema is unsupported.")
    acceptance = _mapping(score.get("acceptance"), "acceptance")
    if acceptance.get("accepted") is not True:
        raise C1ScoreError("Checked C1 score is not accepted.")
    stage = _mapping(score.get("stage"), "stage")
    if stage != {
        "changed_component": CHANGED_COMPONENT,
        "comparison_stage_id": PARENT_STAGE_ID,
        "parent_stage_id": PARENT_STAGE_ID,
        "stage_id": STAGE_ID,
        "status": "completed",
    }:
        raise C1ScoreError("Checked C1 stage identity is invalid.")
    sources = _mapping(score.get("sources"), "sources")
    expected_paths = {source.relative_to(REPOSITORY_ROOT).as_posix() for source in _SOURCE_PATHS}
    if set(sources) != expected_paths:
        raise C1ScoreError("Checked C1 source closure is incomplete.")
    for relative, digest in sources.items():
        if source_sha256(REPOSITORY_ROOT / relative) != _sha256(
            digest,
            f"sources.{relative}",
        ):
            raise C1ScoreError("Checked C1 source binding is stale.")
    c0_binding = _mapping(score.get("c0_binding"), "c0_binding")
    if set(c0_binding) != {
        "raw_manifest_sha256",
        "summary_path",
        "summary_sha256",
    }:
        raise C1ScoreError("Checked C1 C0 binding fields are invalid.")
    expected_summary_relative = DEFAULT_C0_SUMMARY.relative_to(REPOSITORY_ROOT).as_posix()
    if c0_binding.get("summary_path") != expected_summary_relative:
        raise C1ScoreError("Checked C1 must bind the canonical C0 summary path.")
    summary_path = DEFAULT_C0_SUMMARY
    if _file_sha256(summary_path, "bound C0 summary") != _sha256(
        c0_binding.get("summary_sha256"),
        "c0_binding.summary_sha256",
    ):
        raise C1ScoreError("Checked C1 C0-summary binding is stale.")
    c0_summary = load_c0_summary(summary_path)
    raw_bundle = _mapping(c0_summary.get("raw_bundle"), "bound C0 raw_bundle")
    if _sha256(
        c0_binding.get("raw_manifest_sha256"),
        "c0_binding.raw_manifest_sha256",
    ) != _sha256(
        raw_bundle.get("manifest_sha256"),
        "bound C0 raw_bundle.manifest_sha256",
    ):
        raise C1ScoreError("Checked C1 raw-manifest binding does not match C0.")
    return score


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c0-bundle", type=Path, required=True)
    parser.add_argument("--c0-summary", type=Path, default=DEFAULT_C0_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCORE)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c1_score(
            args.c0_bundle,
            args.output,
            c0_summary_path=args.c0_summary,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
