"""Score the independent Fig. 8 Eq. E2 model against the paper raster.

This layer deliberately scores only the black dashed analytic curve. The
blue solid numerical trace is independently digitized but remains an oracle
for the later author/minimal/qpsim replacement ladder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from validation.paper_parity import (
    CurveScore,
    PaperParityError,
    file_sha256,
    load_digitized_points,
    load_strict_json,
    resolve_contained_path,
    score_curve,
    select_curve,
)
from validation.reference_models.fischer_2023 import fig8_analytic

SPEC_SCHEMA = "qpsim.fischer2023.fig8-comparison.v1"
SCORE_SCHEMA = "qpsim.fischer2023.fig8-cleanroom-analytic-score.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = (
    REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig8"
)
DEFAULT_SPEC = ORACLE_DIRECTORY / "comparison-spec.json"
DEFAULT_SCORE = ORACLE_DIRECTORY / "cleanroom-analytic-score.json"
RUNNER_SOURCE = Path(__file__).resolve()
ENGINE_SOURCE = REPOSITORY_ROOT / "validation" / "paper_parity.py"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperParityError(f"{label} must be a JSON object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise PaperParityError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, "
            f"got {sorted(mapping)!r}."
        )


def _string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PaperParityError(f"{label}.{key} must be a nonempty string.")
    return value


def _sha(mapping: dict[str, Any], key: str, label: str) -> str:
    value = _string(mapping, key, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PaperParityError(f"{label}.{key} must be a lowercase SHA-256 digest.")
    return value


def _finite_number(
    mapping: dict[str, Any],
    key: str,
    label: str,
    *,
    positive: bool = False,
) -> float:
    value = mapping.get(key)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive finite" if positive else "finite"
        raise PaperParityError(f"{label}.{key} must be a {qualifier} number.")
    return float(value)


def _load_spec(path: Path) -> tuple[dict[str, Any], Path]:
    spec = load_strict_json(path, "Fig. 8 comparison specification")
    _exact_keys(
        spec,
        {
            "comparison_id",
            "error_budget",
            "interpolation",
            "mapping",
            "metric",
            "numerical_curve_status",
            "observable_identity",
            "oracle",
            "schema",
        },
        "comparison specification",
    )
    if spec.get("schema") != SPEC_SCHEMA:
        raise PaperParityError("Fig. 8 comparison specification schema is stale.")
    _string(spec, "comparison_id", "comparison specification")

    oracle = _mapping(spec["oracle"], "comparison specification.oracle")
    _exact_keys(oracle, {"path", "sha256"}, "comparison specification.oracle")
    oracle_path = resolve_contained_path(
        REPOSITORY_ROOT,
        _string(oracle, "path", "comparison specification.oracle"),
        "comparison specification.oracle.path",
    )
    if not oracle_path.is_file() or file_sha256(oracle_path) != _sha(
        oracle,
        "sha256",
        "comparison specification.oracle",
    ):
        raise PaperParityError("The Fig. 8 comparison specification's oracle is stale.")

    mapping = _mapping(spec["mapping"], "comparison specification.mapping")
    _exact_keys(
        mapping,
        {
            "curve_id",
            "curve_kind",
            "reference_function",
            "reference_path",
            "reference_sha256",
        },
        "comparison specification.mapping",
    )
    if (
        _string(mapping, "curve_id", "comparison specification.mapping")
        != "T_bath_0K"
        or _string(mapping, "curve_kind", "comparison specification.mapping")
        != "paper_analytic"
        or _string(
            mapping,
            "reference_function",
            "comparison specification.mapping",
        )
        != "fig8_analytic_curve"
    ):
        raise PaperParityError("The Fig. 8 analytic mapping is stale.")
    model_path = resolve_contained_path(
        REPOSITORY_ROOT,
        _string(mapping, "reference_path", "comparison specification.mapping"),
        "comparison specification.mapping.reference_path",
    )
    if (
        model_path != Path(fig8_analytic.__file__).resolve()
        or not model_path.is_file()
        or file_sha256(model_path)
        != _sha(mapping, "reference_sha256", "comparison specification.mapping")
    ):
        raise PaperParityError("The Fig. 8 clean-room reference source is stale.")

    metric = _mapping(spec["metric"], "comparison specification.metric")
    _exact_keys(
        metric,
        {"acceptance_rule", "name", "uncertainty_normalized_limit"},
        "comparison specification.metric",
    )
    if (
        metric.get("name") != "raster_digitization_normalized_mismatch"
        or metric.get("acceptance_rule")
        != "max_raster_digitization_normalized_error <= limit"
    ):
        raise PaperParityError("The Fig. 8 comparison metric is stale.")
    _finite_number(
        metric,
        "uncertainty_normalized_limit",
        "comparison specification.metric",
        positive=True,
    )

    interpolation = _mapping(
        spec["interpolation"],
        "comparison specification.interpolation",
    )
    _exact_keys(
        interpolation,
        {"extrapolation", "method", "native_domain", "native_node_count"},
        "comparison specification.interpolation",
    )
    if (
        interpolation.get("extrapolation")
        != "forbidden, including every paper x-uncertainty interval"
        or interpolation.get("method") != "piecewise_linear_on_dense_cleanroom_nodes"
        or interpolation.get("native_domain") != [0.0, 0.92]
        or interpolation.get("native_node_count") != 921
    ):
        raise PaperParityError("The Fig. 8 interpolation contract is stale.")

    numerical = _mapping(
        spec["numerical_curve_status"],
        "comparison specification.numerical_curve_status",
    )
    _exact_keys(
        numerical,
        {"curve_id", "curve_kind", "note", "score_eligible", "status"},
        "comparison specification.numerical_curve_status",
    )
    if (
        numerical.get("curve_id") != "T_bath_0K"
        or numerical.get("curve_kind") != "paper_numerical"
        or numerical.get("score_eligible") is not False
        or numerical.get("status") != "digitized_reference_only"
    ):
        raise PaperParityError("The Fig. 8 numerical-curve status is stale.")
    _string(numerical, "note", "comparison specification.numerical_curve_status")

    error_budget = _mapping(
        spec["error_budget"],
        "comparison specification.error_budget",
    )
    _exact_keys(
        error_budget,
        {"components", "gate_eligible", "gate_ineligibility_reasons"},
        "comparison specification.error_budget",
    )
    if error_budget.get("gate_eligible") is not False:
        raise PaperParityError("The Fig. 8 comparison must remain gate-ineligible.")
    reasons = error_budget.get("gate_ineligibility_reasons")
    if not isinstance(reasons, list) or len(reasons) < 2 or any(
        not isinstance(reason, str) or not reason for reason in reasons
    ):
        raise PaperParityError("The Fig. 8 gate-ineligibility reasons are malformed.")
    components = error_budget.get("components")
    if not isinstance(components, list) or len(components) != 2:
        raise PaperParityError("The Fig. 8 error budget must declare two components.")

    observable = _mapping(
        spec["observable_identity"],
        "comparison specification.observable_identity",
    )
    _exact_keys(
        observable,
        {"schema", "x", "y"},
        "comparison specification.observable_identity",
    )
    if observable.get("schema") != "qpsim.fischer2023.fig8.observable-identity.v1":
        raise PaperParityError("The Fig. 8 observable identity is stale.")
    return spec, oracle_path


def _stable_float(value: float) -> float:
    return float(f"{value:.12g}")


def _reference_score(score: CurveScore) -> dict[str, Any]:
    serialized = score.as_dict()
    points: list[dict[str, object]] = []
    raw_points = serialized.pop("points")
    if not isinstance(raw_points, list):
        raise PaperParityError("CurveScore points must serialize as a list.")
    for raw_point in raw_points:
        if not isinstance(raw_point, dict):
            raise PaperParityError("CurveScore point must serialize as an object.")
        point = dict(raw_point)
        point["reference_y"] = point.pop("qpsim_y")
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
    result["reference_model"] = "clean_room_paper_equation_E2"
    result["temperature_K"] = 0.0
    result["native_reference_domain"] = [0.0, 0.92]
    result["native_reference_node_count"] = 921
    result["points"] = points
    return result


def build_score(spec_path: Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Return the deterministic provenance-bound score payload."""

    spec, oracle_path = _load_spec(spec_path)
    oracle, points = load_digitized_points(oracle_path)
    x = np.linspace(0.0, 0.92, 921)
    reference_y = fig8_analytic.fig8_analytic_curve(x)
    limit = float(spec["metric"]["uncertainty_normalized_limit"])
    curve_score = score_curve(
        select_curve(points, "T_bath_0K", "paper_analytic"),
        x,
        reference_y,
        uncertainty_normalized_limit=limit,
    )
    points_path = oracle_path.parent / oracle["data"]["path"]
    return {
        "accepted": curve_score.accepted,
        "claim": (
            "The independent clean-room transcription of paper Eq. E2 agrees "
            "with the digitized black dashed Fig. 8 trace within the bounded "
            "raster digitization uncertainty."
        ),
        "comparison_spec": {
            "path": spec_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": file_sha256(spec_path),
        },
        "curve_score": _reference_score(curve_score),
        "evidence_class": "clean_room_paper_equation",
        "implementation": {
            "forbidden_dependency": "qpsim",
            "parameters": {
                "a_minus_half": fig8_analytic.APPENDIX_E_A_MINUS_HALF,
                "a_plus_half": fig8_analytic.APPENDIX_E_A_PLUS_HALF,
                "a_plus_three_halves": (
                    fig8_analytic.APPENDIX_E_A_PLUS_THREE_HALVES
                ),
                "eq_E2_linear": fig8_analytic.EQ_E2_LINEAR,
                "eq_E2_quadratic": fig8_analytic.EQ_E2_QUADRATIC,
            },
            "path": spec["mapping"]["reference_path"],
            "sha256": spec["mapping"]["reference_sha256"],
        },
        "paper_oracle": {
            "archive_sha256": oracle["source"]["archive_sha256"],
            "manifest_path": oracle_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "manifest_sha256": file_sha256(oracle_path),
            "points_path": points_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "points_sha256": file_sha256(points_path),
        },
        "qualification": {
            "does_not_claim": [
                "reproduction of the blue solid numerical curve",
                "identity with author-supplied numerical source",
                "validation of qpsim physics or numerics",
            ],
            "gate_eligible": False,
            "metric_scope": "raster_digitization_uncertainty_only",
            "printed_moment_precision": "unbounded",
        },
        "schema": SCORE_SCHEMA,
        "scorer": {
            "engine_path": ENGINE_SOURCE.relative_to(REPOSITORY_ROOT).as_posix(),
            "engine_sha256": file_sha256(ENGINE_SOURCE),
            "metric": "raster_digitization_normalized_mismatch",
            "runner_path": RUNNER_SOURCE.relative_to(REPOSITORY_ROOT).as_posix(),
            "runner_sha256": file_sha256(RUNNER_SOURCE),
        },
        "status": "accepted_clean_room_analytic",
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def verify_checked_score(
    score_path: Path = DEFAULT_SCORE,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, Any]:
    """Require the checked score bytes to equal a fresh deterministic build."""

    expected = _canonical_json_bytes(build_score(spec_path))
    try:
        actual = score_path.read_bytes()
    except OSError as exc:
        raise PaperParityError(f"Cannot read Fig. 8 checked score: {exc}") from exc
    if actual != expected:
        raise PaperParityError("The checked Fig. 8 clean-room score is stale.")
    return load_strict_json(score_path, "Fig. 8 checked score")


def write_score(
    path: Path = DEFAULT_SCORE,
    spec_path: Path = DEFAULT_SPEC,
) -> Path:
    """Write the canonical score JSON."""

    path.write_bytes(_canonical_json_bytes(build_score(spec_path)))
    return path


if __name__ == "__main__":
    print(write_score())
