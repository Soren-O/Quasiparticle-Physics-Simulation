"""Build the provenance-bound clean-room analytic comparison for Fig. 6.

This scorer is intentionally separate from both the author attachment and
qpsim's numerical artifact.  It answers one narrow question: does an
independent transcription of the paper's Eq. 47/53 analytic chain reproduce
the dashed traces in the authenticated paper raster?
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from validation.paper_parity import (
    CurveScore,
    DigitizedPoint,
    lf_canonical_sha256,
    load_digitized_points,
    parse_strict_json_bytes,
    resolve_contained_path,
    score_curve,
    select_curve,
)
from validation.reference_models.fischer_2023 import fig6_analytic
from validation.source_provenance import source_sha256

SCORE_SCHEMA = "qpsim.fischer2023.fig6-cleanroom-analytic-score.v1"
PAPER_POINT_ANCHOR_SCHEMA = "qpsim.paper-point-anchor.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6"
ORACLE_PATH = ORACLE_DIRECTORY / "oracle.json"
AUTHOR_SOURCE_MANIFEST = ORACLE_DIRECTORY / "author-source.json"
DEFAULT_SCORE = ORACLE_DIRECTORY / "cleanroom-analytic-score.json"
MODEL_SOURCE = (
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig6_analytic.py"
)
SCORER_SOURCE = REPOSITORY_ROOT / "validation" / "paper_parity.py"
SOURCE_PROVENANCE_MODULE = REPOSITORY_ROOT / "validation" / "source_provenance.py"
PRODUCER_SOURCE_PATHS = (
    Path(__file__).resolve(),
    MODEL_SOURCE,
    SCORER_SOURCE,
    SOURCE_PROVENANCE_MODULE,
)


def _source_snapshot() -> dict[str, str]:
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): source_sha256(path)
        for path in PRODUCER_SOURCE_PATHS
    }


_IMPORTED_SOURCE_SNAPSHOT = _source_snapshot()


def _begin_source_snapshot() -> dict[str, str]:
    """Bind scoring to the source identities loaded with this module."""

    current = _source_snapshot()
    if current != _IMPORTED_SOURCE_SNAPSHOT:
        raise RuntimeError(
            "Figure 6 clean-room producer source no longer matches the "
            "source snapshot taken when the scorer was imported; reload the "
            "module before scoring."
        )
    return current


def _assert_source_snapshot(snapshot: dict[str, str]) -> None:
    current = _source_snapshot()
    if current != snapshot:
        changed = sorted(
            path
            for path in set(snapshot) | set(current)
            if snapshot.get(path) != current.get(path)
        )
        raise RuntimeError(
            "Figure 6 clean-room producer source changed during scoring: "
            + ", ".join(changed)
        )


def _evidence_snapshot() -> tuple[dict[Path, bytes], Path]:
    """Freeze every recorded evidence file before evaluating the model."""

    try:
        manifest_bytes = ORACLE_PATH.read_bytes()
        author_manifest_bytes = AUTHOR_SOURCE_MANIFEST.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot snapshot Figure 6 evidence: {exc}") from exc
    manifest = parse_strict_json_bytes(manifest_bytes, "Figure 6 paper-data manifest")
    data = manifest.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("path"), str):
        raise ValueError("Figure 6 paper-data manifest has no data.path.")
    points_path = resolve_contained_path(
        ORACLE_PATH.parent,
        data["path"],
        "Figure 6 paper-data manifest.data.path",
    )
    try:
        points_bytes = points_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot snapshot Figure 6 paper points: {exc}") from exc
    return {
        ORACLE_PATH: manifest_bytes,
        points_path: points_bytes,
        AUTHOR_SOURCE_MANIFEST: author_manifest_bytes,
    }, points_path


def _assert_file_snapshot(snapshot: dict[Path, bytes], label: str) -> None:
    changed: list[str] = []
    for path, expected in snapshot.items():
        try:
            current = path.read_bytes()
        except OSError:
            changed.append(path.as_posix())
            continue
        if current != expected:
            changed.append(path.as_posix())
    if changed:
        raise RuntimeError(f"{label} changed during scoring: {', '.join(changed)}")


def _runtime_provenance() -> dict[str, Any]:
    return {
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _paper_point_anchor(point: DigitizedPoint) -> dict[str, Any]:
    values: dict[str, Any] = {
        "curve_id": point.curve_id,
        "curve_kind": point.curve_kind,
        "trace_max_y_pixel": point.trace_max_y_pixel,
        "trace_min_y_pixel": point.trace_min_y_pixel,
        "x_pixel": point.x_pixel,
        "x_uncertainty": point.x_uncertainty,
        "x_value": point.x_value,
        "y_pixel": point.y_pixel,
        "y_uncertainty": point.y_uncertainty,
        "y_value": point.y_value,
    }
    canonical = json.dumps(
        values,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return {
        "schema": PAPER_POINT_ANCHOR_SCHEMA,
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "values": values,
    }


def _stable_float(value: float) -> float:
    """Quantize diagnostics far below the raster uncertainty scale."""

    return float(f"{value:.12g}")


def _reference_score(
    score: CurveScore,
    temperature: float,
    paper_points: list[DigitizedPoint],
) -> dict[str, Any]:
    serialized = score.as_dict()
    raw_points = serialized.pop("points")
    if not isinstance(raw_points, list):
        raise TypeError("CurveScore serialization must contain a points list.")
    points: list[dict[str, object]] = []
    for raw_point, paper_point in zip(raw_points, paper_points, strict=True):
        if not isinstance(raw_point, dict):
            raise TypeError("CurveScore points must serialize as objects.")
        point = dict(raw_point)
        point["reference_y"] = point.pop("qpsim_y")
        point["paper_point"] = _paper_point_anchor(paper_point)
        points.append(
            {
                key: _stable_float(value) if isinstance(value, float) else value
                for key, value in point.items()
            }
        )
    result: dict[str, Any] = {
        key: _stable_float(value) if isinstance(value, float) else value
        for key, value in serialized.items()
    }
    result["reference_model"] = "clean_room_paper_equations"
    result["temperature_K"] = temperature
    result["native_reference_domain"] = [0.24, 0.42]
    result["native_reference_node_count"] = 1801
    result["points"] = points
    return result


def build_score() -> dict[str, Any]:
    """Return the deterministic checked score payload."""

    source_snapshot = _begin_source_snapshot()
    evidence_snapshot, points_path = _evidence_snapshot()
    oracle, points = load_digitized_points(ORACLE_PATH)
    x = np.linspace(0.24, 0.42, 1801)
    curve_scores: list[dict[str, Any]] = []
    for temperature in (0.10, 0.15, 0.20):
        reference_y = fig6_analytic.fig6_curve(temperature, x)
        selected_points = select_curve(
            points,
            f"T_bath_{temperature:.2f}K",
            "paper_analytic",
        )
        score = score_curve(
            selected_points,
            x,
            reference_y,
            uncertainty_normalized_limit=1.0,
        )
        curve_scores.append(_reference_score(score, temperature, selected_points))

    parameters = {
        "c_photon_per_ns": fig6_analytic.C_PHOTON_PER_NS,
        "delta_0_over_kBT_c": fig6_analytic.DELTA_0_OVER_KBT_C,
        "delta_0_ueV": fig6_analytic.DELTA_0_UEV,
        "omega_0_ueV": fig6_analytic.OMEGA_0_UEV,
        "tau_0_ns": fig6_analytic.TAU_0_NS,
        "tau_0_pb_ns": fig6_analytic.TAU_0_PB_NS,
        "tau_l_ns": fig6_analytic.TAU_L_NS,
    }
    accepted = all(bool(score["accepted"]) for score in curve_scores)
    if accepted:
        status = "accepted_clean_room_analytic"
        claim = (
            "The independent clean-room Eq. 47/53 implementation agrees with "
            "the digitized dashed Fig. 6 traces within the bounded raster "
            "digitization uncertainty."
        )
    else:
        status = "diagnostic_mismatch"
        claim = (
            "The independent clean-room Eq. 47/53 implementation does not "
            "agree with every digitized dashed Fig. 6 trace within the "
            "bounded raster digitization uncertainty; this payload is "
            "diagnostic and is not acceptance evidence."
        )
    payload = {
        "accepted": accepted,
        "author_source": {
            "manifest_path": AUTHOR_SOURCE_MANIFEST.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "manifest_sha256": hashlib.sha256(
                evidence_snapshot[AUTHOR_SOURCE_MANIFEST]
            ).hexdigest(),
            "used_by_this_comparison": False,
        },
        "claim": claim,
        "curve_scores": curve_scores,
        "evidence_class": "clean_room_paper_equations",
        "implementation": {
            "forbidden_dependency": "qpsim",
            "parameters": parameters,
            "path": MODEL_SOURCE.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": source_snapshot[
                MODEL_SOURCE.relative_to(REPOSITORY_ROOT).as_posix()
            ],
            "thermal_denominator": (
                "continuum ideal-BCS direct-gap integral after E=Delta*cosh(u)"
            ),
        },
        "paper_oracle": {
            "archive_sha256": oracle["source"]["archive_sha256"],
            "manifest_path": ORACLE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "manifest_sha256": hashlib.sha256(
                evidence_snapshot[ORACLE_PATH]
            ).hexdigest(),
            "points_path": points_path.relative_to(REPOSITORY_ROOT).as_posix(),
            # Content-defined, so this records the same identity the oracle
            # manifest pins in `data.sha256` no matter which newline policy
            # the producing checkout used. `.gitattributes` holds the oracle
            # JSON at eol=lf but not this CSV, so hashing it raw would write
            # a Windows-only digest into a certified artifact.
            "points_sha256": lf_canonical_sha256(evidence_snapshot[points_path]),
        },
        "producer": {
            "runtime": _runtime_provenance(),
            "sources": {
                path: {
                    "hash_kind": "source_canonical_sha256",
                    "sha256": digest,
                }
                for path, digest in source_snapshot.items()
            },
        },
        "qualification": {
            "does_not_claim": [
                "identity with the author-supplied numerical source",
                "agreement with the paper's solid numerical curves",
                "validation of qpsim's collision kernels or nonlinear solver",
            ],
            "metric_scope": "raster_digitization_uncertainty_only",
            "parameter_uncertainty": "not included",
            "quadrature": "adaptive continuum integral; source and tolerances bound above",
        },
        "schema": SCORE_SCHEMA,
        "scorer": {
            "metric": "raster_digitization_normalized_mismatch",
            "path": SCORER_SOURCE.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": source_snapshot[
                SCORER_SOURCE.relative_to(REPOSITORY_ROOT).as_posix()
            ],
        },
        "status": status,
    }
    _assert_file_snapshot(evidence_snapshot, "Figure 6 clean-room evidence")
    _assert_source_snapshot(source_snapshot)
    return payload


def write_score(path: Path = DEFAULT_SCORE) -> Path:
    """Write the canonical score JSON."""

    payload = build_score()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


if __name__ == "__main__":
    print(write_score())
