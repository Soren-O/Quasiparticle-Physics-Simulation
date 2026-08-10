"""Fail-closed primitives for quantitative paper-curve comparisons.

Paper-derived coordinates are independent reference data.  They must never be
stored in, or inferred from, qpsim's generated baseline CSVs.  This module
authenticates a tidy point dataset and scores one qpsim curve at the paper
coordinates without extrapolation.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from validation.source_provenance import source_sha256

PAPER_DATA_SCHEMA = "qpsim.paper-data.v1"
POINT_COLUMNS = (
    "curve_id",
    "curve_kind",
    "x_pixel",
    "y_pixel",
    "trace_min_y_pixel",
    "trace_max_y_pixel",
    "x_value",
    "y_value",
    "x_uncertainty",
    "y_uncertainty",
)

_TRACE_IDENTITY_ORDERS = {
    "caption-bound analytic-above-numerical; no crossings": (
        "paper_analytic",
        "paper_numerical",
    ),
    "caption-bound numerical-above-analytic; no crossings": (
        "paper_numerical",
        "paper_analytic",
    ),
}


class PaperParityError(ValueError):
    """A paper-data or comparison contract is malformed or stale."""


@dataclass(frozen=True)
class DigitizedPoint:
    """One calibrated point from a declared paper curve."""

    curve_id: str
    curve_kind: str
    x_pixel: int
    y_pixel: float
    trace_min_y_pixel: int
    trace_max_y_pixel: int
    x_value: float
    y_value: float
    x_uncertainty: float
    y_uncertainty: float


@dataclass(frozen=True)
class PointScore:
    """One paper point and the qpsim value evaluated at its abscissa."""

    x_value: float
    paper_y: float
    qpsim_y: float
    absolute_error: float
    relative_error: float
    effective_uncertainty: float
    uncertainty_normalized_error: float

    def as_dict(self) -> dict[str, float]:
        return {
            "absolute_error": self.absolute_error,
            "effective_uncertainty": self.effective_uncertainty,
            "paper_y": self.paper_y,
            "qpsim_y": self.qpsim_y,
            "relative_error": self.relative_error,
            "uncertainty_normalized_error": self.uncertainty_normalized_error,
            "x_value": self.x_value,
        }


@dataclass(frozen=True)
class CurveScore:
    """Uncertainty-aware comparison of one qpsim and paper curve."""

    curve_id: str
    curve_kind: str
    point_count: int
    max_absolute_error: float
    mean_absolute_error: float
    rms_error: float
    max_relative_error: float
    max_uncertainty_normalized_error: float
    uncertainty_normalized_limit: float
    accepted: bool
    points: tuple[PointScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "curve_id": self.curve_id,
            "curve_kind": self.curve_kind,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "max_uncertainty_normalized_error": (self.max_uncertainty_normalized_error),
            "mean_absolute_error": self.mean_absolute_error,
            "point_count": self.point_count,
            "points": [point.as_dict() for point in self.points],
            "rms_error": self.rms_error,
            "uncertainty_normalized_limit": self.uncertainty_normalized_limit,
        }


def file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path`` exactly as stored on disk.

    Raw-byte identity, for artifacts whose checkout bytes are themselves
    pinned: the author archive, and the JSON manifests held at ``eol=lf`` by
    ``.gitattributes``.  Text whose digest was produced from logical content
    must use :func:`lf_canonical_sha256` or
    :func:`~validation.source_provenance.source_sha256` instead.
    """

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def lf_canonical_sha256(payload: bytes) -> str:
    """Return the SHA-256 of ``payload`` with every newline represented as LF.

    ``manifest.data.sha256`` is written by the extractor over the CSV it holds
    in memory, which is LF, before that text is ever written to disk.  Hashing
    the file back raw therefore compares against a digest of different bytes
    the moment the working tree is CRLF, and authentication fails on a pin that
    is correct.  That asymmetry is invisible on the Linux CI that produced the
    pins and fires on every Windows checkout, so it reads as data corruption
    when nothing is wrong.  Newline policy is a property of the checkout;
    it must not decide provenance.
    """

    return hashlib.sha256(payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperParityError(f"{label} must be a JSON object.")
    return value


def _require_exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(mapping)
    if actual != expected:
        raise PaperParityError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, got {sorted(actual)!r}."
        )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise PaperParityError(f"JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def parse_strict_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    """Parse one finite UTF-8 JSON object while rejecting duplicate keys."""

    try:
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PaperParityError(f"{label} contains non-finite JSON token {value!r}.")
            ),
        )
    except PaperParityError:
        raise
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PaperParityError(f"Cannot parse {label}: {exc}") from exc
    return _require_mapping(raw, label)


def load_strict_json(path: Path, label: str) -> dict[str, Any]:
    """Read a finite JSON object while rejecting duplicate keys."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PaperParityError(f"Cannot read {label} {path}: {exc}") from exc
    return parse_strict_json_bytes(payload, label)


def _require_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PaperParityError(f"{label}.{key} must be a nonempty string.")
    return value


def _require_finite_number(
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


def _require_sha256(mapping: dict[str, Any], key: str, label: str) -> str:
    value = _require_string(mapping, key, label)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PaperParityError(f"{label}.{key} must be a lowercase SHA-256 digest.")
    return value


def resolve_contained_path(base: Path, raw: str, label: str) -> Path:
    pure = PurePosixPath(raw)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise PaperParityError(f"{label} must be a contained relative POSIX path.")
    resolved_base = base.resolve()
    resolved = (base / Path(*pure.parts)).resolve()
    if resolved != resolved_base and resolved_base not in resolved.parents:
        raise PaperParityError(f"{label} escapes its manifest directory.")
    return resolved


def _axis_affine(axis: dict[str, Any], label: str) -> tuple[float, float, float]:
    """Validate a two-point linear calibration and its held-out controls."""

    calibration = _require_mapping(axis.get("calibration"), f"{label}.calibration")
    _require_exact_keys(
        calibration,
        {
            "fit_control_points",
            "held_out_control_points",
            "held_out_residual_limit_px",
            "slope_data_per_pixel",
            "intercept_data",
            "held_out_max_residual_px",
        },
        f"{label}.calibration",
    )
    fit = calibration["fit_control_points"]
    held_out = calibration["held_out_control_points"]
    if not isinstance(fit, list) or len(fit) != 2:
        raise PaperParityError(f"{label} needs exactly two fit control points.")
    if not isinstance(held_out, list) or not held_out:
        raise PaperParityError(f"{label} needs at least one held-out control point.")

    def parse_control(raw: object, control_label: str) -> tuple[float, float]:
        control = _require_mapping(raw, control_label)
        _require_exact_keys(control, {"pixel", "value"}, control_label)
        return (
            _require_finite_number(control, "pixel", control_label),
            _require_finite_number(control, "value", control_label),
        )

    (p0, v0), (p1, v1) = [
        parse_control(item, f"{label}.fit_control_points[{index}]")
        for index, item in enumerate(fit)
    ]
    if p0 == p1:
        raise PaperParityError(f"{label} calibration pixels are degenerate.")
    slope = (v1 - v0) / (p1 - p0)
    intercept = v0 - slope * p0
    if slope == 0.0 or not math.isfinite(slope) or not math.isfinite(intercept):
        raise PaperParityError(f"{label} calibration is degenerate.")
    declared_slope = _require_finite_number(
        calibration, "slope_data_per_pixel", f"{label}.calibration"
    )
    declared_intercept = _require_finite_number(
        calibration, "intercept_data", f"{label}.calibration"
    )
    if not math.isclose(declared_slope, slope, rel_tol=0.0, abs_tol=1e-17):
        raise PaperParityError(f"{label} declared calibration slope is stale.")
    if not math.isclose(declared_intercept, intercept, rel_tol=0.0, abs_tol=1e-15):
        raise PaperParityError(f"{label} declared calibration intercept is stale.")
    residuals = []
    for index, item in enumerate(held_out):
        pixel, value = parse_control(item, f"{label}.held_out_control_points[{index}]")
        residuals.append(abs(value - (slope * pixel + intercept)) / abs(slope))
    declared_residual = _require_finite_number(
        calibration,
        "held_out_max_residual_px",
        f"{label}.calibration",
    )
    if not math.isclose(declared_residual, max(residuals), rel_tol=0.0, abs_tol=1e-12):
        raise PaperParityError(f"{label} declared held-out residual is stale.")
    residual_limit = _require_finite_number(
        calibration,
        "held_out_residual_limit_px",
        f"{label}.calibration",
        positive=True,
    )
    if declared_residual > residual_limit:
        raise PaperParityError(
            f"{label} held-out residual {declared_residual:.6g} px exceeds "
            f"its {residual_limit:.6g} px acceptance limit."
        )
    return slope, intercept, declared_residual


def load_paper_manifest(path: Path) -> dict[str, Any]:
    """Load and authenticate a qpsim paper-data manifest and points file."""

    manifest = load_strict_json(path, "paper-data manifest")
    _require_exact_keys(
        manifest,
        {
            "asset",
            "curves",
            "data",
            "dataset_id",
            "extraction",
            "panel",
            "paper",
            "schema",
            "source",
        },
        "manifest",
    )
    if manifest.get("schema") != PAPER_DATA_SCHEMA:
        raise PaperParityError(
            f"Unsupported paper-data schema {manifest.get('schema')!r}; "
            f"expected {PAPER_DATA_SCHEMA!r}."
        )
    _require_string(manifest, "dataset_id", "manifest")
    paper = _require_mapping(manifest.get("paper"), "manifest.paper")
    _require_exact_keys(
        paper,
        {
            "arxiv_id",
            "citation",
            "doi",
            "figure",
            "panel",
            "pdf_page_1_based",
            "version",
        },
        "manifest.paper",
    )
    for key in ("arxiv_id", "citation", "doi", "figure", "panel", "version"):
        _require_string(paper, key, "manifest.paper")
    pdf_page = paper.get("pdf_page_1_based")
    if not isinstance(pdf_page, int) or isinstance(pdf_page, bool) or pdf_page < 1:
        raise PaperParityError("manifest.paper.pdf_page_1_based must be a positive integer.")
    source = _require_mapping(manifest.get("source"), "manifest.source")
    _require_exact_keys(
        source,
        {"archive_sha256", "redistribution_note", "retrieved_utc", "url"},
        "manifest.source",
    )
    for key in ("url", "retrieved_utc", "redistribution_note"):
        _require_string(source, key, "manifest.source")
    _require_sha256(source, "archive_sha256", "manifest.source")
    asset = _require_mapping(manifest.get("asset"), "manifest.asset")
    _require_exact_keys(
        asset,
        {
            "dpi",
            "member",
            "panel_crop_pixels",
            "pixel_dimensions",
            "sha256",
            "software",
        },
        "manifest.asset",
    )
    _require_string(asset, "member", "manifest.asset")
    _require_sha256(asset, "sha256", "manifest.asset")
    _require_string(asset, "software", "manifest.asset")
    member = PurePosixPath(_require_string(asset, "member", "manifest.asset"))
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise PaperParityError("manifest.asset.member must be a safe POSIX path.")
    dimensions = asset.get("pixel_dimensions")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 2
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in dimensions
        )
    ):
        raise PaperParityError(
            "manifest.asset.pixel_dimensions must contain two positive integers."
        )
    dpi = asset.get("dpi")
    if (
        not isinstance(dpi, list)
        or len(dpi) != 2
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) <= 0.0
            for item in dpi
        )
    ):
        raise PaperParityError("manifest.asset.dpi must contain two positive finite numbers.")
    crop = asset.get("panel_crop_pixels")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(not isinstance(item, int) or isinstance(item, bool) for item in crop)
    ):
        raise PaperParityError("manifest.asset.panel_crop_pixels must contain four integers.")
    x0, y0, x1, y1 = crop
    width, height = dimensions
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise PaperParityError("manifest.asset.panel_crop_pixels is out of bounds.")
    panel = _require_mapping(manifest.get("panel"), "manifest.panel")
    _require_exact_keys(panel, {"x_axis", "y_axis"}, "manifest.panel")
    for axis_name in ("x_axis", "y_axis"):
        axis = _require_mapping(panel.get(axis_name), f"manifest.panel.{axis_name}")
        _require_exact_keys(
            axis,
            {"calibration", "quantity", "scale", "unit"},
            f"manifest.panel.{axis_name}",
        )
        if axis.get("scale") != "linear":
            raise PaperParityError(
                f"Only linear axes are supported in v1; got "
                f"{axis_name}.scale={axis.get('scale')!r}."
            )
        _require_string(axis, "quantity", f"manifest.panel.{axis_name}")
        _require_string(axis, "unit", f"manifest.panel.{axis_name}")
        _axis_affine(axis, f"manifest.panel.{axis_name}")
        calibration = _require_mapping(
            axis["calibration"], f"manifest.panel.{axis_name}.calibration"
        )
        pixel_low, pixel_high = (crop[0], crop[2]) if axis_name == "x_axis" else (crop[1], crop[3])
        controls = [
            *calibration["fit_control_points"],
            *calibration["held_out_control_points"],
        ]
        for index, raw_control in enumerate(controls):
            control = _require_mapping(
                raw_control,
                f"manifest.panel.{axis_name}.control_points[{index}]",
            )
            pixel = _require_finite_number(
                control,
                "pixel",
                f"manifest.panel.{axis_name}.control_points[{index}]",
            )
            if not pixel_low <= pixel < pixel_high:
                raise PaperParityError(
                    f"manifest.panel.{axis_name} control pixel {pixel:g} is outside the panel crop."
                )
    extraction = _require_mapping(manifest.get("extraction"), "manifest.extraction")
    _require_exact_keys(
        extraction,
        {
            "accepted_y_value_window",
            "color_curve_order",
            "color_masks",
            "curve_order_top_to_bottom",
            "horizontal_half_window_px",
            "minimum_trace_separation_px",
            "method",
            "plot_rows_inclusive",
            "sample_x_values",
            "script",
            "script_sha256",
            "trace_identity_policy",
            "uncertainty",
            "version",
        },
        "manifest.extraction",
    )
    _require_string(extraction, "method", "manifest.extraction")
    script_raw = _require_string(extraction, "script", "manifest.extraction")
    expected_script_sha = _require_sha256(extraction, "script_sha256", "manifest.extraction")
    trace_identity_policy = _require_string(
        extraction, "trace_identity_policy", "manifest.extraction"
    )
    if trace_identity_policy not in _TRACE_IDENTITY_ORDERS:
        raise PaperParityError("manifest.extraction.trace_identity_policy is unsupported.")
    _require_string(extraction, "version", "manifest.extraction")
    _require_finite_number(
        extraction,
        "minimum_trace_separation_px",
        "manifest.extraction",
        positive=True,
    )
    repository_root = Path(__file__).resolve().parents[1]
    script_path = resolve_contained_path(repository_root, script_raw, "manifest.extraction.script")
    if not script_path.is_file():
        raise PaperParityError(f"Digitizer source is missing: {script_path}")
    # Verified with the same helper that writes the pin: the producer records
    # `source_sha256(...)` (see fig6_author_output_parity._source_snapshot),
    # and `tests/review_2026_08_03/test_P16.py` pins that agreement. Reading
    # the file raw here compared two different digests of the same source and
    # failed every Windows checkout.
    if source_sha256(script_path) != expected_script_sha:
        raise PaperParityError("Digitizer source does not match manifest.extraction.script_sha256.")
    sample_values = extraction.get("sample_x_values")
    if (
        not isinstance(sample_values, list)
        or len(sample_values) < 3
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in sample_values
        )
        or any(right <= left for left, right in pairwise(sample_values))
    ):
        raise PaperParityError(
            "manifest.extraction.sample_x_values must be strictly increasing finite numbers."
        )
    half_window = extraction.get("horizontal_half_window_px")
    if (
        not isinstance(half_window, int)
        or isinstance(half_window, bool)
        or not 0 <= half_window <= 5
    ):
        raise PaperParityError("manifest.extraction.horizontal_half_window_px must be in [0, 5].")
    plot_rows = extraction.get("plot_rows_inclusive")
    if (
        not isinstance(plot_rows, list)
        or len(plot_rows) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in plot_rows)
        or not crop[1] <= plot_rows[0] < plot_rows[1] < crop[3]
        or plot_rows != [crop[1], crop[3] - 1]
    ):
        raise PaperParityError(
            "manifest.extraction.plot_rows_inclusive must cover the full panel crop."
        )
    y_window = extraction.get("accepted_y_value_window")
    if (
        not isinstance(y_window, list)
        or len(y_window) != 2
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in y_window
        )
        or not float(y_window[0]) < float(y_window[1])
    ):
        raise PaperParityError("manifest.extraction.accepted_y_value_window is invalid.")
    curve_order = extraction.get("curve_order_top_to_bottom")
    if curve_order != list(_TRACE_IDENTITY_ORDERS[trace_identity_policy]):
        raise PaperParityError("manifest.extraction.curve_order_top_to_bottom is invalid.")
    uncertainty = _require_mapping(extraction.get("uncertainty"), "manifest.extraction.uncertainty")
    _require_exact_keys(
        uncertainty,
        {
            "minimum_y_uncertainty",
            "model",
            "sampling_half_width_px",
            "serialization_absolute_tolerance",
        },
        "manifest.extraction.uncertainty",
    )
    _require_string(uncertainty, "model", "manifest.extraction.uncertainty")
    for key in (
        "minimum_y_uncertainty",
        "sampling_half_width_px",
        "serialization_absolute_tolerance",
    ):
        _require_finite_number(
            uncertainty,
            key,
            "manifest.extraction.uncertainty",
            positive=True,
        )
    colour_masks = _require_mapping(
        extraction.get("color_masks"), "manifest.extraction.color_masks"
    )
    if not colour_masks:
        raise PaperParityError("manifest.extraction.color_masks must not be empty.")
    kind_specific_masks: bool | None = None
    for curve_id, raw_mask in colour_masks.items():
        if not isinstance(curve_id, str) or not curve_id:
            raise PaperParityError("Colour-mask curve IDs must be nonempty strings.")
        mask = _require_mapping(raw_mask, f"manifest.extraction.color_masks.{curve_id}")
        is_kind_specific = set(mask) == {"paper_analytic", "paper_numerical"}
        if kind_specific_masks is None:
            kind_specific_masks = is_kind_specific
        elif kind_specific_masks != is_kind_specific:
            raise PaperParityError(
                "manifest.extraction.color_masks must use one mask schema consistently."
            )
        masks_by_kind = (
            {
                kind: _require_mapping(
                    mask[kind],
                    f"manifest.extraction.color_masks.{curve_id}.{kind}",
                )
                for kind in ("paper_analytic", "paper_numerical")
            }
            if is_kind_specific
            else {"shared": mask}
        )
        for kind, curve_mask in masks_by_kind.items():
            mask_label = f"manifest.extraction.color_masks.{curve_id}.{kind}"
            mode = _require_string(curve_mask, "mode", mask_label) if is_kind_specific else ""
            if not is_kind_specific or mode == "channel_dominance":
                expected = (
                    {"channel", "dominance_ratio", "minimum", "mode"}
                    if is_kind_specific
                    else {"channel", "dominance_ratio", "minimum"}
                )
                _require_exact_keys(curve_mask, expected, mask_label)
                if _require_string(curve_mask, "channel", mask_label) not in {
                    "blue",
                    "green",
                    "red",
                }:
                    raise PaperParityError(f"Unsupported colour channel for {curve_id!r}.")
                _require_finite_number(
                    curve_mask,
                    "minimum",
                    mask_label,
                    positive=True,
                )
                dominance = _require_finite_number(
                    curve_mask,
                    "dominance_ratio",
                    mask_label,
                    positive=True,
                )
                if dominance <= 1.0:
                    raise PaperParityError(
                        f"Colour dominance ratio must exceed one for {curve_id!r}."
                    )
            elif mode == "grayscale":
                _require_exact_keys(
                    curve_mask,
                    {"channel_spread_maximum", "maximum", "mode"},
                    mask_label,
                )
                maximum = _require_finite_number(curve_mask, "maximum", mask_label)
                spread = _require_finite_number(
                    curve_mask,
                    "channel_spread_maximum",
                    mask_label,
                )
                if not 0.0 <= maximum < 255.0 or not 0.0 <= spread < 255.0:
                    raise PaperParityError(
                        f"Grayscale mask bounds are invalid for {curve_id!r}."
                    )
            else:
                raise PaperParityError(
                    f"Unsupported kind-specific colour-mask mode {mode!r}."
                )
    colour_curve_order = extraction.get("color_curve_order")
    if (
        not isinstance(colour_curve_order, list)
        or any(not isinstance(curve_id, str) or not curve_id for curve_id in colour_curve_order)
        or len(colour_curve_order) != len(set(colour_curve_order))
        or set(colour_curve_order) != set(colour_masks)
    ):
        raise PaperParityError(
            "manifest.extraction.color_curve_order must list every colour-mask curve exactly once."
        )
    curves = manifest.get("curves")
    if not isinstance(curves, list) or not curves:
        raise PaperParityError("manifest.curves must be a nonempty array.")
    curve_contracts: dict[tuple[str, str], float] = {}
    for index, raw_curve in enumerate(curves):
        curve = _require_mapping(raw_curve, f"manifest.curves[{index}]")
        _require_exact_keys(
            curve,
            {
                "curve_id",
                "curve_kind",
                "point_count",
                "source_style",
                "temperature_K",
            },
            f"manifest.curves[{index}]",
        )
        kind = _require_string(curve, "curve_kind", f"manifest.curves[{index}]")
        if kind not in {"paper_analytic", "paper_numerical"}:
            raise PaperParityError(f"manifest.curves[{index}].curve_kind is unsupported.")
        curve_id = _require_string(curve, "curve_id", f"manifest.curves[{index}]")
        _require_string(curve, "source_style", f"manifest.curves[{index}]")
        temperature = _require_finite_number(
            curve,
            "temperature_K",
            f"manifest.curves[{index}]",
        )
        if temperature < 0.0:
            raise PaperParityError(
                f"manifest.curves[{index}].temperature_K must be nonnegative."
            )
        point_count = curve.get("point_count")
        if (
            not isinstance(point_count, int)
            or isinstance(point_count, bool)
            or point_count != len(sample_values)
        ):
            raise PaperParityError(
                f"manifest.curves[{index}].point_count must equal the declared "
                "sample_x_values count."
            )
        expected_style = "dashed" if kind == "paper_analytic" else "solid"
        if curve["source_style"] != expected_style:
            raise PaperParityError(
                f"manifest.curves[{index}].source_style must be {expected_style!r} for {kind!r}."
            )
        curve_key = (curve_id, kind)
        if curve_key in curve_contracts:
            raise PaperParityError(f"Duplicate declared curve {curve_key!r}.")
        curve_contracts[curve_key] = temperature
    family_temperatures: dict[str, float] = {}
    for curve_id in {key[0] for key in curve_contracts}:
        family = {
            kind: temperature
            for (declared_id, kind), temperature in curve_contracts.items()
            if declared_id == curve_id
        }
        if set(family) != {"paper_analytic", "paper_numerical"}:
            raise PaperParityError(
                f"Curve family {curve_id!r} must contain analytic and numerical traces."
            )
        if family["paper_analytic"] != family["paper_numerical"]:
            raise PaperParityError(
                f"Curve family {curve_id!r} assigns different temperatures "
                "to its analytic and numerical traces."
            )
        family_temperatures[curve_id] = family["paper_analytic"]
    if len(set(family_temperatures.values())) != len(family_temperatures):
        raise PaperParityError("Paper curve families must have unique temperatures.")
    declared_ids = {
        _require_string(
            _require_mapping(curve, f"manifest.curves[{index}]"),
            "curve_id",
            f"manifest.curves[{index}]",
        )
        for index, curve in enumerate(curves)
    }
    if set(colour_masks) != declared_ids:
        raise PaperParityError("Colour-mask IDs and declared curve IDs do not match exactly.")
    data = _require_mapping(manifest.get("data"), "manifest.data")
    _require_exact_keys(data, {"columns", "path", "sha256"}, "manifest.data")
    relative_data_path = _require_string(data, "path", "manifest.data")
    expected_data_sha = _require_sha256(data, "sha256", "manifest.data")
    columns = data.get("columns")
    if columns != list(POINT_COLUMNS):
        raise PaperParityError(f"manifest.data.columns must equal {list(POINT_COLUMNS)!r}.")
    data_path = resolve_contained_path(path.parent, relative_data_path, "manifest.data.path")
    if not data_path.is_file():
        raise PaperParityError(f"Paper-data points file is missing: {data_path}")
    try:
        data_bytes = data_path.read_bytes()
    except OSError as exc:
        raise PaperParityError(f"Cannot read paper-data points file {data_path}: {exc}") from exc
    actual_data_sha = lf_canonical_sha256(data_bytes)
    if actual_data_sha != expected_data_sha:
        raise PaperParityError(
            f"Paper-data points SHA-256 mismatch: expected {expected_data_sha}, "
            f"got {actual_data_sha}."
        )
    return manifest


def load_digitized_points(manifest_path: Path) -> tuple[dict[str, Any], list[DigitizedPoint]]:
    """Authenticate ``manifest_path`` and return its strictly validated points."""

    manifest = load_paper_manifest(manifest_path)
    data = _require_mapping(manifest["data"], "manifest.data")
    data_path = resolve_contained_path(
        manifest_path.parent,
        _require_string(data, "path", "manifest.data"),
        "manifest.data.path",
    )
    points: list[DigitizedPoint] = []
    asset = _require_mapping(manifest["asset"], "manifest.asset")
    dimensions = asset["pixel_dimensions"]
    crop = asset["panel_crop_pixels"]
    panel = _require_mapping(manifest["panel"], "manifest.panel")
    x_slope, x_intercept, x_calibration_residual = _axis_affine(
        _require_mapping(panel["x_axis"], "manifest.panel.x_axis"),
        "manifest.panel.x_axis",
    )
    y_slope, y_intercept, y_calibration_residual = _axis_affine(
        _require_mapping(panel["y_axis"], "manifest.panel.y_axis"),
        "manifest.panel.y_axis",
    )
    extraction = _require_mapping(manifest["extraction"], "manifest.extraction")
    sample_values = [float(value) for value in extraction["sample_x_values"]]
    expected_x_pixels = [
        round((sample_value - x_intercept) / x_slope) for sample_value in sample_values
    ]
    if any(right <= left for left, right in pairwise(expected_x_pixels)):
        raise PaperParityError(
            "Declared sample_x_values do not map to strictly increasing raster columns."
        )
    plot_row_min, plot_row_max = extraction["plot_rows_inclusive"]
    accepted_y_min, accepted_y_max = map(float, extraction["accepted_y_value_window"])
    minimum_trace_separation = float(extraction["minimum_trace_separation_px"])
    uncertainty = _require_mapping(
        extraction["uncertainty"],
        "manifest.extraction.uncertainty",
    )
    serialization_tolerance = _require_finite_number(
        uncertainty,
        "serialization_absolute_tolerance",
        "manifest.extraction.uncertainty",
        positive=True,
    )
    sampling_half_width = _require_finite_number(
        uncertainty,
        "sampling_half_width_px",
        "manifest.extraction.uncertainty",
        positive=True,
    )
    minimum_y_uncertainty = _require_finite_number(
        uncertainty,
        "minimum_y_uncertainty",
        "manifest.extraction.uncertainty",
        positive=True,
    )
    horizontal_half_window = extraction["horizontal_half_window_px"]
    crop_x0, _, crop_x1, _ = crop
    if any(
        x_pixel - horizontal_half_window < crop_x0 or x_pixel + horizontal_half_window >= crop_x1
        for x_pixel in expected_x_pixels
    ):
        raise PaperParityError(
            "A declared sample_x_values extraction window leaves the panel crop."
        )
    try:
        payload = data_path.read_bytes()
        expected_sha = _require_sha256(data, "sha256", "manifest.data")
        # Same digest convention as the authentication above, so this
        # re-read cannot disagree with it about an unchanged file.
        actual_sha = lf_canonical_sha256(payload)
        if actual_sha != expected_sha:
            raise PaperParityError(
                f"Paper-data points changed after manifest authentication: "
                f"expected {expected_sha}, got {actual_sha}."
            )
        text = payload.decode("utf-8")
        with io.StringIO(text, newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(POINT_COLUMNS):
                raise PaperParityError(
                    f"Paper-data CSV header must equal {list(POINT_COLUMNS)!r}; "
                    f"got {reader.fieldnames!r}."
                )
            for row_number, row in enumerate(reader, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise PaperParityError(f"Malformed paper-data CSV row {row_number}.")
                curve_id = row["curve_id"]
                curve_kind = row["curve_kind"]
                if not curve_id:
                    raise PaperParityError(f"curve_id is empty at paper-data row {row_number}.")
                if curve_kind not in {"paper_analytic", "paper_numerical"}:
                    raise PaperParityError(
                        f"Unsupported curve_kind {curve_kind!r} at row {row_number}."
                    )
                try:
                    x_pixel = int(row["x_pixel"])
                    values = [
                        float(row[key])
                        for key in (
                            "y_pixel",
                            "x_value",
                            "y_value",
                            "x_uncertainty",
                            "y_uncertainty",
                        )
                    ]
                    trace_min_y_pixel = int(row["trace_min_y_pixel"])
                    trace_max_y_pixel = int(row["trace_max_y_pixel"])
                except ValueError as exc:
                    raise PaperParityError(
                        f"Non-numeric paper-data value at row {row_number}."
                    ) from exc
                if str(x_pixel) != row["x_pixel"]:
                    raise PaperParityError(
                        f"x_pixel must be a canonical integer at row {row_number}."
                    )
                if (
                    str(trace_min_y_pixel) != row["trace_min_y_pixel"]
                    or str(trace_max_y_pixel) != row["trace_max_y_pixel"]
                ):
                    raise PaperParityError(
                        f"Trace bounds must be canonical integers at row {row_number}."
                    )
                if not all(math.isfinite(value) for value in values):
                    raise PaperParityError(
                        f"Paper-data row {row_number} contains a nonfinite value."
                    )
                y_pixel, x_value, y_value, x_uncertainty, y_uncertainty = values
                if (
                    x_pixel < 0
                    or y_pixel < 0.0
                    or trace_min_y_pixel < 0
                    or trace_max_y_pixel < trace_min_y_pixel
                    or not trace_min_y_pixel <= y_pixel <= trace_max_y_pixel
                ):
                    raise PaperParityError(
                        f"Paper-data pixel geometry is invalid at row {row_number}."
                    )
                if x_uncertainty <= 0.0 or y_uncertainty <= 0.0:
                    raise PaperParityError(
                        f"Paper-data uncertainties must be positive at row {row_number}."
                    )
                width, height = dimensions
                if x_pixel >= width or y_pixel >= height or trace_max_y_pixel >= height:
                    raise PaperParityError(
                        f"Paper-data pixel lies outside the source raster at row {row_number}."
                    )
                x0, y0, x1, y1 = crop
                if not (x0 <= x_pixel < x1 and y0 <= y_pixel < y1):
                    raise PaperParityError(
                        f"Paper-data pixel lies outside the declared panel crop "
                        f"at row {row_number}."
                    )
                if not (
                    y0 <= trace_min_y_pixel <= trace_max_y_pixel < y1
                    and plot_row_min <= trace_min_y_pixel <= trace_max_y_pixel <= plot_row_max
                ):
                    raise PaperParityError(
                        f"Paper-data trace bounds lie outside the declared "
                        f"panel/plot rows at row {row_number}."
                    )
                expected_x = x_slope * x_pixel + x_intercept
                expected_y = y_slope * y_pixel + y_intercept
                if not math.isclose(x_value, expected_x, rel_tol=0.0, abs_tol=1e-11):
                    raise PaperParityError(
                        f"x_value is inconsistent with calibration at row {row_number}."
                    )
                if not math.isclose(y_value, expected_y, rel_tol=0.0, abs_tol=1e-11):
                    raise PaperParityError(
                        f"y_value is inconsistent with calibration at row {row_number}."
                    )
                if not accepted_y_min <= y_value <= accepted_y_max:
                    raise PaperParityError(
                        f"y_value lies outside the declared accepted window at row {row_number}."
                    )
                expected_x_uncertainty = abs(x_slope) * (
                    horizontal_half_window + sampling_half_width + x_calibration_residual
                )
                expected_y_uncertainty = max(
                    minimum_y_uncertainty,
                    abs(y_slope)
                    * (
                        0.5 * (trace_max_y_pixel - trace_min_y_pixel)
                        + sampling_half_width
                        + y_calibration_residual
                    ),
                )
                if not math.isclose(
                    x_uncertainty,
                    expected_x_uncertainty,
                    rel_tol=0.0,
                    abs_tol=serialization_tolerance,
                ):
                    raise PaperParityError(
                        f"x_uncertainty is inconsistent with the declared "
                        f"model at row {row_number}."
                    )
                if not math.isclose(
                    y_uncertainty,
                    expected_y_uncertainty,
                    rel_tol=0.0,
                    abs_tol=serialization_tolerance,
                ):
                    raise PaperParityError(
                        f"y_uncertainty is inconsistent with the declared "
                        f"model at row {row_number}."
                    )
                points.append(
                    DigitizedPoint(
                        curve_id=curve_id,
                        curve_kind=curve_kind,
                        x_pixel=x_pixel,
                        y_pixel=y_pixel,
                        trace_min_y_pixel=trace_min_y_pixel,
                        trace_max_y_pixel=trace_max_y_pixel,
                        x_value=x_value,
                        y_value=y_value,
                        x_uncertainty=x_uncertainty,
                        y_uncertainty=y_uncertainty,
                    )
                )
    except OSError as exc:
        raise PaperParityError(f"Cannot read paper-data points {data_path}: {exc}") from exc

    if not points:
        raise PaperParityError("Paper-data CSV contains no points.")
    keys = [(point.curve_id, point.curve_kind, point.x_pixel) for point in points]
    if len(keys) != len(set(keys)):
        raise PaperParityError("Paper-data points contain duplicate curve/pixel keys.")

    declared_curves = manifest.get("curves")
    if not isinstance(declared_curves, list) or not declared_curves:
        raise PaperParityError("manifest.curves must be a nonempty array.")
    declared: dict[tuple[str, str], int] = {}
    for index, item in enumerate(declared_curves):
        curve = _require_mapping(item, f"manifest.curves[{index}]")
        curve_id = _require_string(curve, "curve_id", f"manifest.curves[{index}]")
        curve_kind = _require_string(curve, "curve_kind", f"manifest.curves[{index}]")
        if curve_kind not in {"paper_analytic", "paper_numerical"}:
            raise PaperParityError(f"manifest.curves[{index}].curve_kind is unsupported.")
        count = curve.get("point_count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 3:
            raise PaperParityError(f"manifest.curves[{index}].point_count must be an integer >= 3.")
        key = (curve_id, curve_kind)
        if key in declared:
            raise PaperParityError(f"Duplicate declared curve {key!r}.")
        declared[key] = count

    actual: dict[tuple[str, str], list[DigitizedPoint]] = {}
    for point in points:
        actual.setdefault((point.curve_id, point.curve_kind), []).append(point)
    if set(actual) != set(declared):
        raise PaperParityError(
            f"Declared and actual paper curves differ: declared={sorted(declared)!r}, "
            f"actual={sorted(actual)!r}."
        )
    for key, curve_points in actual.items():
        if len(curve_points) != declared[key]:
            raise PaperParityError(
                f"Curve {key!r} has {len(curve_points)} points; manifest declares {declared[key]}."
            )
        x_values = [point.x_value for point in curve_points]
        if any(right <= left for left, right in pairwise(x_values)):
            raise PaperParityError(f"Curve {key!r} x coordinates must be strictly increasing.")
        x_pixels = [point.x_pixel for point in curve_points]
        if x_pixels != expected_x_pixels:
            raise PaperParityError(
                f"Curve {key!r} raster columns do not match the declared sample_x_values geometry."
            )

    extraction = _require_mapping(manifest["extraction"], "manifest.extraction")
    trace_identity_policy = _require_string(
        extraction,
        "trace_identity_policy",
        "manifest.extraction",
    )
    top_kind, bottom_kind = _TRACE_IDENTITY_ORDERS[trace_identity_policy]
    colour_masks = _require_mapping(
        extraction["color_masks"],
        "manifest.extraction.color_masks",
    )
    masks_are_kind_specific = all(
        set(_require_mapping(mask, "manifest.extraction.color_masks entry"))
        == {"paper_analytic", "paper_numerical"}
        for mask in colour_masks.values()
    )
    paired: dict[tuple[str, int], dict[str, DigitizedPoint]] = {}
    for point in points:
        paired.setdefault((point.curve_id, point.x_pixel), {})[point.curve_kind] = point
    for pair_key, kinds in paired.items():
        if set(kinds) != {"paper_analytic", "paper_numerical"}:
            raise PaperParityError(
                f"Paper trace pair {pair_key!r} must contain analytic and numerical curves."
            )
        top = kinds[top_kind]
        bottom = kinds[bottom_kind]
        separation = bottom.y_pixel - top.y_pixel
        if separation < minimum_trace_separation:
            raise PaperParityError(
                f"Paper trace pair {pair_key!r} violates the declared "
                f"{trace_identity_policy.split(';', maxsplit=1)[0].removeprefix('caption-bound ')} "
                "separation."
            )
        if (
            not masks_are_kind_specific
            and top.trace_max_y_pixel >= bottom.trace_min_y_pixel
        ):
            raise PaperParityError(
                f"Paper trace pair {pair_key!r} has overlapping analytic/numerical bands."
            )
    return manifest, points


def select_curve(
    points: list[DigitizedPoint], curve_id: str, curve_kind: str
) -> list[DigitizedPoint]:
    """Return one declared curve, preserving its stored x order."""

    selected = [
        point for point in points if point.curve_id == curve_id and point.curve_kind == curve_kind
    ]
    if not selected:
        raise PaperParityError(f"Paper-data curve ({curve_id!r}, {curve_kind!r}) is absent.")
    return selected


def _validated_real_array(value: object, label: str) -> np.ndarray:
    """Reject lossy coercions before returning one float64 array."""

    array = np.asarray(value)
    if array.dtype.kind not in {"f", "i", "u"}:
        raise PaperParityError(
            f"{label} must have a real floating-point or integer dtype; got {array.dtype}."
        )
    result = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(result)):
        raise PaperParityError(f"{label} contains nonfinite values.")
    return result


def score_curve(
    paper_points: list[DigitizedPoint],
    qpsim_x: np.ndarray,
    qpsim_y: np.ndarray,
    *,
    uncertainty_normalized_limit: float,
) -> CurveScore:
    """Interpolate qpsim linearly and score it against one paper curve.

    The uncertainty envelope is the stored vertical digitization uncertainty
    plus the local qpsim slope times the stored horizontal uncertainty.  The
    two quantities are added conservatively because the manifest records
    bounds, not independent Gaussian standard deviations.
    """

    if not paper_points:
        raise PaperParityError("Cannot score an empty paper curve.")
    if (
        not isinstance(uncertainty_normalized_limit, (int, float, np.integer, np.floating))
        or isinstance(uncertainty_normalized_limit, (bool, np.bool_))
        or not math.isfinite(float(uncertainty_normalized_limit))
        or float(uncertainty_normalized_limit) <= 0.0
    ):
        raise PaperParityError("uncertainty_normalized_limit must be finite and positive.")
    if any(not isinstance(point, DigitizedPoint) for point in paper_points):
        raise PaperParityError(
            "paper_points must contain only authenticated DigitizedPoint values."
        )
    if any(
        not point.curve_id
        or point.curve_kind not in {"paper_analytic", "paper_numerical"}
        or not isinstance(point.x_pixel, int)
        or isinstance(point.x_pixel, bool)
        or not isinstance(point.trace_min_y_pixel, int)
        or isinstance(point.trace_min_y_pixel, bool)
        or not isinstance(point.trace_max_y_pixel, int)
        or isinstance(point.trace_max_y_pixel, bool)
        or not point.trace_min_y_pixel <= point.y_pixel <= point.trace_max_y_pixel
        for point in paper_points
    ):
        raise PaperParityError("paper_points contain invalid curve identity or pixel geometry.")
    paper_values = np.array(
        [
            [
                point.x_pixel,
                point.y_pixel,
                point.trace_min_y_pixel,
                point.trace_max_y_pixel,
                point.x_value,
                point.y_value,
                point.x_uncertainty,
                point.y_uncertainty,
            ]
            for point in paper_points
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(paper_values)):
        raise PaperParityError("paper_points contain nonfinite values.")
    if np.any(paper_values[:, 6:] <= 0.0):
        raise PaperParityError("Paper-point uncertainties must be positive.")
    if np.any(paper_values[:, :4] < 0.0):
        raise PaperParityError("Paper-point pixel coordinates must be nonnegative.")
    if np.any(paper_values[:, 3] < paper_values[:, 2]):
        raise PaperParityError("Paper-point trace pixel bounds are inverted.")
    curve_keys = {(point.curve_id, point.curve_kind) for point in paper_points}
    if len(curve_keys) != 1:
        raise PaperParityError("paper_points must all belong to one curve.")
    qx = _validated_real_array(qpsim_x, "qpsim_x")
    qy = _validated_real_array(qpsim_y, "qpsim_y")
    if qx.ndim != 1 or qy.ndim != 1 or qx.shape != qy.shape or qx.size < 2:
        raise PaperParityError(
            "qpsim_x and qpsim_y must be equal-length 1-D arrays with >= 2 points."
        )
    if np.any(np.diff(qx) <= 0.0):
        raise PaperParityError("qpsim_x must be strictly increasing.")

    px = np.array([point.x_value for point in paper_points], dtype=float)
    py = np.array([point.y_value for point in paper_points], dtype=float)
    px_unc = np.array([point.x_uncertainty for point in paper_points], dtype=float)
    py_unc = np.array([point.y_uncertainty for point in paper_points], dtype=float)
    if np.any(np.diff(px) <= 0.0):
        raise PaperParityError("Paper x coordinates must be strictly increasing.")
    lower_bounds = px - px_unc
    upper_bounds = px + px_unc
    if np.any(lower_bounds < qx[0]) or np.any(upper_bounds > qx[-1]):
        raise PaperParityError(
            "A paper x-uncertainty interval exceeds the qpsim domain "
            f"[{qx[0]:.12g}, {qx[-1]:.12g}]; extrapolation is forbidden."
        )

    q_at_paper = np.interp(px, qx, qy)
    horizontal_effects: list[float] = []
    for x_value, lower, upper, center_y in zip(
        px, lower_bounds, upper_bounds, q_at_paper, strict=True
    ):
        knots = qx[(qx > lower) & (qx < upper)]
        candidates = np.concatenate(([lower, x_value, upper], knots))
        candidate_y = np.interp(candidates, qx, qy)
        horizontal_effects.append(float(np.max(np.abs(candidate_y - center_y))))
    uncertainty_envelope = py_unc + np.asarray(horizontal_effects)
    if np.any(~np.isfinite(uncertainty_envelope)) or np.any(uncertainty_envelope <= 0.0):
        raise PaperParityError("Effective digitization uncertainty is invalid.")

    absolute_error = np.abs(q_at_paper - py)
    relative_error = absolute_error / np.maximum(np.abs(py), uncertainty_envelope)
    normalized_error = absolute_error / uncertainty_envelope
    point_scores = tuple(
        PointScore(
            x_value=float(x_value),
            paper_y=float(paper_y),
            qpsim_y=float(qpsim_y),
            absolute_error=float(error),
            relative_error=float(relative),
            effective_uncertainty=float(uncertainty),
            uncertainty_normalized_error=float(normalized),
        )
        for (
            x_value,
            paper_y,
            qpsim_y,
            error,
            relative,
            uncertainty,
            normalized,
        ) in zip(
            px,
            py,
            q_at_paper,
            absolute_error,
            relative_error,
            uncertainty_envelope,
            normalized_error,
            strict=True,
        )
    )
    curve_id, curve_kind = next(iter(curve_keys))
    max_normalized = float(np.max(normalized_error))
    return CurveScore(
        curve_id=curve_id,
        curve_kind=curve_kind,
        point_count=len(paper_points),
        max_absolute_error=float(np.max(absolute_error)),
        mean_absolute_error=float(np.mean(absolute_error)),
        rms_error=float(np.sqrt(np.mean(np.square(absolute_error)))),
        max_relative_error=float(np.max(relative_error)),
        max_uncertainty_normalized_error=max_normalized,
        uncertainty_normalized_limit=float(uncertainty_normalized_limit),
        accepted=max_normalized <= uncertainty_normalized_limit,
        points=point_scores,
    )
