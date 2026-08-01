"""Score promoted qpsim Fig. 6 observables against digitized paper curves.

This is deliberately downstream of the expensive Fig. 6 producer.  It reads
one exact promotion-bound CSV snapshot, never imports the numerical solve, and
keeps the independent paper oracle separate from the qpsim-to-paper mapping.
The result is diagnostic while parameter and discretization uncertainties
remain unbounded.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from validation.paper_parity import (
    CurveScore,
    PaperParityError,
    file_sha256,
    load_digitized_points,
    load_strict_json,
    parse_strict_json_bytes,
    resolve_contained_path,
    score_curve,
    select_curve,
)

COMPARISON_SCHEMA = "qpsim.paper-comparison.v1"
SCORE_SCHEMA = "qpsim.fischer2023.fig6-paper-parity-score.v1"
ARTIFACT_SCHEMA = "qpsim.fischer2023.fig6_paper.v2"
PROMOTION_SCHEMA = "qpsim.fischer2023.fig6_promotion.v2"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6"
DEFAULT_SPEC = ORACLE_DIRECTORY / "comparison-spec.json"
DEFAULT_SCORE = ORACLE_DIRECTORY / "score.json"

ARTIFACT_COLUMNS = (
    "T_bath_K",
    "n_bar",
    "T_star_over_delta",
    "delta_eq_T_bath_ueV",
    "delta_driven_ueV",
    "x_qp_num",
    "x_qp_eq47",
    "paper_observable_num",
    "paper_observable_eq53",
    "qp_residual_inf",
    "qp_backward_error",
    "qp_number_backward_error",
    "phonon_residual_inf",
    "phonon_raw_backward_error",
    "phonon_backward_error",
    "gap_fixed_point_abs_error_uev",
    "state_f_f64_zlib_base64",
    "state_n_ph_f64_zlib_base64",
    "state_sha256",
)
METADATA_KEYS = {
    "certificate_fields",
    "certificate_maxima",
    "certificate_metric_version",
    "certificate_target_backward_error",
    "columns",
    "fingerprint",
    "payload_sha256",
    "row_count",
    "schema",
}


@dataclass(frozen=True)
class CanonicalSnapshot:
    """Scalar columns from one promotion-bound immutable CSV byte snapshot."""

    csv_path: Path
    promotion_path: Path
    csv_bytes: bytes
    promotion_bytes: bytes
    columns: dict[str, np.ndarray]
    metadata: dict[str, Any]


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperParityError(f"{label} must be a JSON object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise PaperParityError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(mapping)!r}."
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
    mapping: dict[str, Any], key: str, label: str, *, positive: bool = False
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


def _read_spec(path: Path) -> tuple[dict[str, Any], Path]:
    spec = load_strict_json(path, "Fig. 6 comparison specification")
    _exact_keys(
        spec,
        {
            "comparison_id",
            "error_budget",
            "interpolation",
            "mappings",
            "metric",
            "observable_identity",
            "oracle",
            "qpsim_artifact",
            "schema",
        },
        "comparison specification",
    )
    if spec.get("schema") != COMPARISON_SCHEMA:
        raise PaperParityError("Fig. 6 comparison specification schema is stale.")
    _string(spec, "comparison_id", "comparison specification")

    oracle = _mapping(spec["oracle"], "comparison specification.oracle")
    _exact_keys(oracle, {"path", "sha256"}, "comparison specification.oracle")
    oracle_path = resolve_contained_path(
        REPOSITORY_ROOT,
        _string(oracle, "path", "comparison specification.oracle"),
        "comparison specification.oracle.path",
    )
    if not oracle_path.is_file() or file_sha256(oracle_path) != _sha(
        oracle, "sha256", "comparison specification.oracle"
    ):
        raise PaperParityError("The comparison specification's paper oracle is stale.")

    observable_identity = _mapping(
        spec["observable_identity"],
        "comparison specification.observable_identity",
    )
    _exact_keys(
        observable_identity,
        {"schema", "x", "y"},
        "comparison specification.observable_identity",
    )
    if observable_identity.get("schema") != "qpsim.fischer2023.fig6.observable-identity.v1":
        raise PaperParityError("The Fig. 6 observable-identity schema is stale.")
    x_identity = _mapping(
        observable_identity["x"],
        "comparison specification.observable_identity.x",
    )
    y_identity = _mapping(
        observable_identity["y"],
        "comparison specification.observable_identity.y",
    )
    expected_x_identity = {
        "conversion": "identity_ratio_not_energy",
        "paper_quantity": "T_star_over_Delta",
        "paper_unit": "dimensionless",
        "qpsim_column": "T_star_over_delta",
        "qpsim_unit": "dimensionless",
    }
    expected_y_identity = {
        "conversion": "identity_fraction_not_percent",
        "definition": (
            "delta_Delta_T is thermal-equilibrium gap suppression at T_bath; "
            "delta_Delta is driven nonequilibrium gap suppression"
        ),
        "denominator": "delta_Delta_T",
        "paper_quantity": "(delta_Delta_T - delta_Delta) / delta_Delta_T",
        "paper_unit": "dimensionless",
        "qpsim_analytic_column": "paper_observable_eq53",
        "qpsim_numerical_column": "paper_observable_num",
        "qpsim_unit": "dimensionless",
        "sign_convention": (
            "positive_when_driven_gap_suppression_is_smaller_than_thermal_equilibrium"
        ),
    }
    if x_identity != expected_x_identity or y_identity != expected_y_identity:
        raise PaperParityError(
            "The Fig. 6 unit/normalization identity mapping is stale or incomplete."
        )

    interpolation = _mapping(spec["interpolation"], "comparison specification.interpolation")
    _exact_keys(
        interpolation,
        {"extrapolation", "method"},
        "comparison specification.interpolation",
    )
    if (
        interpolation.get("method") != "piecewise_linear_on_native_qpsim_nodes"
        or interpolation.get("extrapolation")
        != "forbidden, including every paper x-uncertainty interval"
    ):
        raise PaperParityError("Unsupported Fig. 6 interpolation contract.")

    metric = _mapping(spec["metric"], "comparison specification.metric")
    _exact_keys(
        metric,
        {"acceptance_rule", "name", "uncertainty_normalized_limit"},
        "comparison specification.metric",
    )
    if (
        metric.get("name") != "raster_digitization_normalized_mismatch"
        or metric.get("acceptance_rule") != "max_raster_digitization_normalized_error <= limit"
    ):
        raise PaperParityError("Unsupported Fig. 6 comparison metric.")
    _finite_number(
        metric,
        "uncertainty_normalized_limit",
        "comparison specification.metric",
        positive=True,
    )

    error_budget = _mapping(spec["error_budget"], "comparison specification.error_budget")
    _exact_keys(
        error_budget,
        {"components", "gate_eligible", "gate_ineligibility_reasons"},
        "comparison specification.error_budget",
    )
    if error_budget.get("gate_eligible") is not False:
        raise PaperParityError(
            "Fig. 6 paper parity is not gate-eligible until every error-budget "
            "component is bounded."
        )
    reasons = error_budget.get("gate_ineligibility_reasons")
    if (
        not isinstance(reasons, list)
        or not reasons
        or any(not isinstance(reason, str) or not reason for reason in reasons)
    ):
        raise PaperParityError("Gate-ineligibility reasons must be explicit.")
    components = error_budget.get("components")
    if not isinstance(components, list) or not components:
        raise PaperParityError("The comparison error budget must have components.")
    component_contract: dict[str, tuple[bool, str]] = {}
    for index, raw_component in enumerate(components):
        component = _mapping(
            raw_component, f"comparison specification.error_budget.components[{index}]"
        )
        _exact_keys(
            component,
            {"included_in_metric", "name", "note", "status"},
            f"comparison specification.error_budget.components[{index}]",
        )
        name = _string(
            component,
            "name",
            f"comparison specification.error_budget.components[{index}]",
        )
        if name in component_contract:
            raise PaperParityError(f"Duplicate error-budget component {name!r}.")
        included = component.get("included_in_metric")
        if not isinstance(included, bool):
            raise PaperParityError(
                f"Error-budget component {name!r} needs a boolean inclusion flag."
            )
        component_contract[name] = (
            included,
            _string(
                component,
                "status",
                f"comparison specification.error_budget.components[{index}]",
            ),
        )
        _string(
            component,
            "note",
            f"comparison specification.error_budget.components[{index}]",
        )
    expected_component_contract = {
        "native_curve_interpolation": (False, "declared_method"),
        "parameter_uncertainty": (False, "unbounded"),
        "qpsim_discretization": (False, "unbounded"),
        "raster_digitization": (True, "bounded"),
        "solver_certification": (False, "promotion_attested_not_replayed"),
    }
    expected_reasons = [
        "parameter_uncertainty is unbounded",
        "qpsim_discretization is unbounded",
    ]
    if component_contract != expected_component_contract or reasons != expected_reasons:
        raise PaperParityError(
            "The Fig. 6 metric inclusion, solver-replay, and unbounded "
            "error-budget contracts must remain exact and explicit."
        )
    return spec, oracle_path


def _validate_paper_identity(spec: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Cross-check the Fig. 6 page/panel and coordinate identities."""

    paper = _mapping(manifest["paper"], "manifest.paper")
    expected_paper = {
        "arxiv_id": "2212.08155",
        "citation": "P. B. Fischer and G. Catelani, Phys. Rev. Applied 19, 054087 (2023)",
        "doi": "10.1103/PhysRevApplied.19.054087",
        "figure": "Figure 6",
        "panel": "full",
        "pdf_page_1_based": 12,
        "version": "arXiv:2212.08155v2",
    }
    if paper != expected_paper:
        raise PaperParityError("The Fig. 6 paper/version/page/panel identity is stale.")
    asset = _mapping(manifest["asset"], "manifest.asset")
    if asset.get("member") != "gap_test.png":
        raise PaperParityError("The Fig. 6 source-raster member identity is stale.")
    expected_curves = {
        ("T_bath_0.10K", "paper_analytic"): (7, "dashed", 0.10),
        ("T_bath_0.10K", "paper_numerical"): (7, "solid", 0.10),
        ("T_bath_0.15K", "paper_analytic"): (7, "dashed", 0.15),
        ("T_bath_0.15K", "paper_numerical"): (7, "solid", 0.15),
        ("T_bath_0.20K", "paper_analytic"): (7, "dashed", 0.20),
        ("T_bath_0.20K", "paper_numerical"): (7, "solid", 0.20),
    }
    actual_curves: dict[tuple[str, str], tuple[int, str, float]] = {}
    raw_curves = manifest.get("curves")
    if not isinstance(raw_curves, list):
        raise PaperParityError("The Fig. 6 curve declarations are malformed.")
    for index, raw_curve in enumerate(raw_curves):
        curve = _mapping(raw_curve, f"manifest.curves[{index}]")
        key = (str(curve.get("curve_id")), str(curve.get("curve_kind")))
        actual_curves[key] = (
            int(curve.get("point_count", -1)),
            str(curve.get("source_style")),
            float(curve.get("temperature_K", math.nan)),
        )
    if actual_curves != expected_curves:
        raise PaperParityError("The Fig. 6 curve/temperature/style identity is stale.")

    identity = _mapping(
        spec["observable_identity"],
        "comparison specification.observable_identity",
    )
    x_identity = _mapping(
        identity["x"],
        "comparison specification.observable_identity.x",
    )
    y_identity = _mapping(
        identity["y"],
        "comparison specification.observable_identity.y",
    )
    panel = _mapping(manifest["panel"], "manifest.panel")
    x_axis = _mapping(panel["x_axis"], "manifest.panel.x_axis")
    y_axis = _mapping(panel["y_axis"], "manifest.panel.y_axis")
    if (
        x_axis.get("quantity") != x_identity.get("paper_quantity")
        or x_axis.get("unit") != x_identity.get("paper_unit")
        or x_axis.get("scale") != "linear"
        or y_axis.get("quantity") != y_identity.get("paper_quantity")
        or y_axis.get("unit") != y_identity.get("paper_unit")
        or y_axis.get("scale") != "linear"
    ):
        raise PaperParityError(
            "The paper axes disagree with the comparison's unit/normalization identity."
        )


def _validate_artifact_contract(spec: dict[str, Any]) -> dict[str, Any]:
    artifact = _mapping(spec["qpsim_artifact"], "comparison specification.qpsim_artifact")
    _exact_keys(
        artifact,
        {
            "artifact_schema",
            "csv_path",
            "csv_sha256",
            "filter_column",
            "native_row_count",
            "promotion_path",
            "promotion_schema",
            "promotion_sha256",
            "required_fingerprint_mode",
            "x_column",
        },
        "comparison specification.qpsim_artifact",
    )
    if (
        artifact.get("artifact_schema") != ARTIFACT_SCHEMA
        or artifact.get("promotion_schema") != PROMOTION_SCHEMA
        or artifact.get("filter_column") != "T_bath_K"
        or artifact.get("x_column") != "T_star_over_delta"
    ):
        raise PaperParityError("The Fig. 6 qpsim artifact contract is stale.")
    row_count = artifact.get("native_row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count != 66:
        raise PaperParityError("The Fig. 6 native row count must be exactly 66.")
    _sha(artifact, "csv_sha256", "comparison specification.qpsim_artifact")
    _sha(artifact, "promotion_sha256", "comparison specification.qpsim_artifact")
    mode = _mapping(
        artifact["required_fingerprint_mode"],
        "comparison specification.qpsim_artifact.required_fingerprint_mode",
    )
    _exact_keys(
        mode,
        {"direct_gap_observable", "fixed_gap_kinetics"},
        "comparison specification.qpsim_artifact.required_fingerprint_mode",
    )
    if mode != {
        "direct_gap_observable": False,
        "fixed_gap_kinetics": False,
    }:
        raise PaperParityError("The Fig. 6 observable mode is not canonical.")
    return artifact


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _payload_sha256(rows: list[list[str]]) -> str:
    canonical = json.dumps(
        rows,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _parse_csv_snapshot(
    payload: bytes,
    artifact: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PaperParityError("Fig. 6 canonical CSV is not UTF-8.") from exc
    rows = list(csv.reader(io.StringIO(text, newline="")))
    expected_row_count = int(artifact["native_row_count"])
    if len(rows) != expected_row_count + 3:
        raise PaperParityError("Fig. 6 canonical CSV row count is stale.")
    if rows[0] != [f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"]:
        raise PaperParityError("Fig. 6 canonical CSV schema marker is stale.")
    metadata_prefix = "# qpsim_metadata="
    if len(rows[1]) != 1 or not rows[1][0].startswith(metadata_prefix):
        raise PaperParityError("Fig. 6 canonical CSV metadata marker is malformed.")
    metadata = parse_strict_json_bytes(
        rows[1][0][len(metadata_prefix) :].encode("utf-8"),
        "Fig. 6 artifact metadata",
    )
    _exact_keys(metadata, METADATA_KEYS, "Fig. 6 artifact metadata")
    if (
        metadata.get("schema") != ARTIFACT_SCHEMA
        or metadata.get("columns") != list(ARTIFACT_COLUMNS)
        or metadata.get("row_count") != expected_row_count
        or metadata.get("payload_sha256") != _payload_sha256(rows[3:])
    ):
        raise PaperParityError("Fig. 6 canonical CSV metadata is stale or corrupt.")
    fingerprint = _mapping(metadata["fingerprint"], "Fig. 6 artifact fingerprint")
    mode = _mapping(fingerprint.get("mode"), "Fig. 6 artifact fingerprint.mode")
    if mode != artifact["required_fingerprint_mode"]:
        raise PaperParityError("Fig. 6 canonical CSV observable mode is stale.")
    if rows[2] != list(ARTIFACT_COLUMNS):
        raise PaperParityError("Fig. 6 canonical CSV columns are stale or reordered.")
    numeric_count = 16
    numeric_rows: list[list[float]] = []
    for row_number, row in enumerate(rows[3:], start=4):
        if len(row) != len(ARTIFACT_COLUMNS):
            raise PaperParityError(f"Malformed Fig. 6 CSV row {row_number}.")
        try:
            numeric = [float(value) for value in row[:numeric_count]]
        except ValueError as exc:
            raise PaperParityError(f"Non-numeric Fig. 6 scalar at row {row_number}.") from exc
        if not all(math.isfinite(value) for value in numeric):
            raise PaperParityError(f"Nonfinite Fig. 6 scalar at row {row_number}.")
        if (
            not row[16]
            or not row[17]
            or len(row[18]) != 64
            or any(character not in "0123456789abcdef" for character in row[18])
        ):
            raise PaperParityError(f"Malformed persisted-state fields at Fig. 6 row {row_number}.")
        numeric_rows.append(numeric)
    matrix = np.asarray(numeric_rows, dtype=float)
    columns = {
        name: matrix[:, index] for index, name in enumerate(ARTIFACT_COLUMNS[:numeric_count])
    }
    return columns, metadata


def _load_canonical_snapshot(artifact: dict[str, Any]) -> CanonicalSnapshot:
    csv_path = resolve_contained_path(
        REPOSITORY_ROOT,
        _string(artifact, "csv_path", "comparison specification.qpsim_artifact"),
        "comparison specification.qpsim_artifact.csv_path",
    )
    promotion_path = resolve_contained_path(
        REPOSITORY_ROOT,
        _string(
            artifact,
            "promotion_path",
            "comparison specification.qpsim_artifact",
        ),
        "comparison specification.qpsim_artifact.promotion_path",
    )
    try:
        promotion_before = promotion_path.read_bytes()
        csv_bytes = csv_path.read_bytes()
        promotion_after = promotion_path.read_bytes()
    except OSError as exc:
        raise PaperParityError(f"Cannot read the promoted Fig. 6 snapshot: {exc}") from exc
    if promotion_before != promotion_after:
        raise PaperParityError("The Fig. 6 promotion changed during snapshotting.")
    if hashlib.sha256(csv_bytes).hexdigest() != artifact["csv_sha256"]:
        raise PaperParityError("The canonical Fig. 6 CSV differs from the comparison spec.")
    if hashlib.sha256(promotion_before).hexdigest() != artifact["promotion_sha256"]:
        raise PaperParityError("The canonical Fig. 6 promotion differs from the comparison spec.")
    promotion = parse_strict_json_bytes(promotion_before, "Fig. 6 promotion record")
    _exact_keys(
        promotion,
        {"artifact_schema", "artifacts", "generation", "schema"},
        "Fig. 6 promotion record",
    )
    if (
        promotion.get("schema") != PROMOTION_SCHEMA
        or promotion.get("artifact_schema") != ARTIFACT_SCHEMA
    ):
        raise PaperParityError("The Fig. 6 promotion record schema is stale.")
    artifacts = _mapping(promotion["artifacts"], "Fig. 6 promotion artifacts")
    _exact_keys(artifacts, {"csv", "pdf"}, "Fig. 6 promotion artifacts")
    csv_identity = _mapping(artifacts["csv"], "Fig. 6 promotion CSV identity")
    _exact_keys(csv_identity, {"sha256", "size_bytes"}, "Fig. 6 promotion CSV identity")
    if csv_identity != {
        "sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "size_bytes": len(csv_bytes),
    }:
        raise PaperParityError("The Fig. 6 promotion does not bind the CSV snapshot.")
    columns, metadata = _parse_csv_snapshot(csv_bytes, artifact)
    if promotion_path.read_bytes() != promotion_before or csv_path.read_bytes() != csv_bytes:
        raise PaperParityError("The promoted Fig. 6 tuple changed during parsing.")
    return CanonicalSnapshot(
        csv_path=csv_path,
        promotion_path=promotion_path,
        csv_bytes=csv_bytes,
        promotion_bytes=promotion_before,
        columns=columns,
        metadata=metadata,
    )


def _validated_mappings(
    spec: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    raw_mappings = spec.get("mappings")
    if not isinstance(raw_mappings, list) or len(raw_mappings) != 6:
        raise PaperParityError("Fig. 6 comparison must declare exactly six mappings.")
    declared_curves = {
        (
            str(curve["curve_id"]),
            str(curve["curve_kind"]),
        ): float(curve["temperature_K"])
        for curve in manifest["curves"]
    }
    mappings: list[dict[str, Any]] = []
    keys: set[tuple[str, str]] = set()
    for index, raw_mapping in enumerate(raw_mappings):
        mapping = _mapping(raw_mapping, f"comparison specification.mappings[{index}]")
        _exact_keys(
            mapping,
            {"curve_id", "curve_kind", "filter_value", "qpsim_y_column"},
            f"comparison specification.mappings[{index}]",
        )
        curve_id = _string(mapping, "curve_id", f"comparison specification.mappings[{index}]")
        curve_kind = _string(mapping, "curve_kind", f"comparison specification.mappings[{index}]")
        key = (curve_id, curve_kind)
        if key in keys or key not in declared_curves:
            raise PaperParityError(f"Invalid or duplicate paper-curve mapping {key!r}.")
        keys.add(key)
        filter_value = _finite_number(
            mapping,
            "filter_value",
            f"comparison specification.mappings[{index}]",
            positive=True,
        )
        if filter_value != declared_curves[key]:
            raise PaperParityError(f"Temperature mapping is stale for {key!r}.")
        expected_column = (
            "paper_observable_eq53" if curve_kind == "paper_analytic" else "paper_observable_num"
        )
        if mapping.get("qpsim_y_column") != expected_column:
            raise PaperParityError(f"Solid/dashed target mapping is inverted for {key!r}.")
        mappings.append(mapping)
    if keys != set(declared_curves):
        raise PaperParityError("The comparison mappings do not cover every paper curve.")
    return mappings


def build_score(spec_path: Path = DEFAULT_SPEC) -> dict[str, object]:
    """Build a deterministic diagnostic score from strictly bound inputs."""

    spec, oracle_path = _read_spec(spec_path)
    artifact = _validate_artifact_contract(spec)
    manifest, points = load_digitized_points(oracle_path)
    _validate_paper_identity(spec, manifest)
    mappings = _validated_mappings(spec, manifest)
    snapshot = _load_canonical_snapshot(artifact)
    metric = _mapping(spec["metric"], "comparison specification.metric")
    limit = float(metric["uncertainty_normalized_limit"])
    filter_values = snapshot.columns[str(artifact["filter_column"])]
    curve_records: list[dict[str, object]] = []
    scores: list[CurveScore] = []
    for mapping in mappings:
        temperature = float(mapping["filter_value"])
        selected_rows = np.flatnonzero(filter_values == temperature)
        if selected_rows.size == 0:
            raise PaperParityError(f"No native qpsim rows exist for T_bath={temperature:g} K.")
        x = snapshot.columns[str(artifact["x_column"])][selected_rows]
        y_column = str(mapping["qpsim_y_column"])
        y = snapshot.columns[y_column][selected_rows]
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        paper_curve = select_curve(
            points,
            str(mapping["curve_id"]),
            str(mapping["curve_kind"]),
        )
        score = score_curve(
            paper_curve,
            x,
            y,
            uncertainty_normalized_limit=limit,
        )
        scores.append(score)
        record = score.as_dict()
        record.update(
            {
                "filter_value_K": temperature,
                "native_qpsim_domain": [float(x[0]), float(x[-1])],
                "native_qpsim_node_count": int(x.size),
                "qpsim_y_column": y_column,
            }
        )
        curve_records.append(record)

    analytic = [score for score in scores if score.curve_kind == "paper_analytic"]
    numerical = [score for score in scores if score.curve_kind == "paper_numerical"]
    analytic_accepted = len(analytic) == 3 and all(score.accepted for score in analytic)
    numerical_accepted = len(numerical) == 3 and all(score.accepted for score in numerical)
    status = (
        "oracle_control_failure"
        if not analytic_accepted
        else "diagnostic_agreement"
        if numerical_accepted
        else "diagnostic_mismatch"
    )
    scorer_sources = {
        "validation/fischer_2023/fig6_paper_parity.py": file_sha256(Path(__file__)),
        "validation/paper_parity.py": file_sha256(
            REPOSITORY_ROOT / "validation" / "paper_parity.py"
        ),
    }
    data = _mapping(manifest["data"], "manifest.data")
    return {
        "comparison_id": spec["comparison_id"],
        "curve_scores": curve_records,
        "error_budget": spec["error_budget"],
        "observable_identity": spec["observable_identity"],
        "inputs": {
            "comparison_spec": {
                "path": spec_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": file_sha256(spec_path),
            },
            "paper_oracle": {
                "archive_sha256": manifest["source"]["archive_sha256"],
                "manifest_path": oracle_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
                "manifest_sha256": file_sha256(oracle_path),
                "points_path": resolve_contained_path(
                    oracle_path.parent,
                    str(data["path"]),
                    "manifest.data.path",
                )
                .resolve()
                .relative_to(REPOSITORY_ROOT)
                .as_posix(),
                "points_sha256": data["sha256"],
                "source_raster_reproduction": {
                    "command": (
                        "python -m validation.fischer_2023."
                        "extract_fig6_paper_data <exact-arxiv-v2-source.tar>"
                    ),
                    "status": "not_replayed_by_scorer",
                },
            },
            "qpsim_artifact": {
                "authentication": {
                    "authoritative_replay": (
                        "separate slow gate under the recorded single-thread environment"
                    ),
                    "status": "promotion_attested_not_replayed",
                },
                "csv_path": snapshot.csv_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
                "csv_sha256": hashlib.sha256(snapshot.csv_bytes).hexdigest(),
                "promotion_path": snapshot.promotion_path.resolve()
                .relative_to(REPOSITORY_ROOT)
                .as_posix(),
                "promotion_sha256": hashlib.sha256(snapshot.promotion_bytes).hexdigest(),
            },
            "scorer_sources": scorer_sources,
        },
        "result": {
            "analytic_control_accepted": analytic_accepted,
            "gate_eligible": False,
            "numerical_parity_accepted": numerical_accepted,
            "status": status,
        },
        "schema": SCORE_SCHEMA,
    }


def score_bytes(spec_path: Path = DEFAULT_SPEC) -> bytes:
    """Return canonical deterministic JSON for the current score."""

    return _canonical_json_bytes(build_score(spec_path))


def verify_checked_score(
    score_path: Path = DEFAULT_SCORE,
    spec_path: Path = DEFAULT_SPEC,
) -> dict[str, object]:
    """Require the checked score to equal a fresh deterministic rebuild."""

    rebuilt = build_score(spec_path)
    expected = _canonical_json_bytes(rebuilt)
    try:
        actual = score_path.read_bytes()
    except OSError as exc:
        raise PaperParityError(f"Cannot read checked Fig. 6 score: {exc}") from exc
    if actual != expected:
        raise PaperParityError("The checked Fig. 6 paper-parity score is stale or noncanonical.")
    return rebuilt
