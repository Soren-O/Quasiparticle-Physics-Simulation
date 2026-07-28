"""Reproduce the Fischer-2023 Fig. 6 digitized oracle from arXiv source.

The source raster is not redistributed.  Supply the exact arXiv-v2 source
archive named in the checked manifest; this extractor verifies the archive
and member hashes before reading pixels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import sys
import tarfile
from collections.abc import Callable
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from validation.paper_parity import (
    POINT_COLUMNS,
    PaperParityError,
    file_sha256,
    load_paper_manifest,
    resolve_contained_path,
)

EXTRACTOR_VERSION = "qpsim.fischer2023.fig6-digitizer.v1"
DEFAULT_ORACLE_DIR = Path(__file__).resolve().parents[1] / "paper_data" / "fischer_2023" / "fig6"
DEFAULT_MANIFEST = DEFAULT_ORACLE_DIR / "oracle.json"


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PaperParityError(f"{label} must be a JSON object.")
    return value


def _linear_axis(axis: dict[str, Any], label: str) -> tuple[float, float, float]:
    calibration = _mapping(axis.get("calibration"), f"{label}.calibration")
    fit = calibration.get("fit_control_points")
    held_out = calibration.get("held_out_control_points")
    if not isinstance(fit, list) or len(fit) != 2:
        raise PaperParityError(f"{label} needs exactly two fit control points.")
    if not isinstance(held_out, list) or not held_out:
        raise PaperParityError(f"{label} needs held-out control points.")

    parsed: list[tuple[float, float]] = []
    for index, item in enumerate(fit):
        point = _mapping(item, f"{label}.fit_control_points[{index}]")
        pixel = point.get("pixel")
        value = point.get("value")
        if (
            not isinstance(pixel, (int, float))
            or isinstance(pixel, bool)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(pixel))
            or not math.isfinite(float(value))
        ):
            raise PaperParityError(f"{label} control points must be finite numbers.")
        parsed.append((float(pixel), float(value)))
    (p0, v0), (p1, v1) = parsed
    if p0 == p1:
        raise PaperParityError(f"{label} calibration pixels are degenerate.")
    slope = (v1 - v0) / (p1 - p0)
    intercept = v0 - slope * p0
    if slope == 0.0 or not math.isfinite(slope) or not math.isfinite(intercept):
        raise PaperParityError(f"{label} calibration is degenerate.")

    residual_pixels: list[float] = []
    for index, item in enumerate(held_out):
        point = _mapping(item, f"{label}.held_out_control_points[{index}]")
        pixel = point.get("pixel")
        value = point.get("value")
        if (
            not isinstance(pixel, (int, float))
            or isinstance(pixel, bool)
            or not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            raise PaperParityError(
                f"{label} held-out points must contain numeric pixel/value pairs."
            )
        pixel_f = float(pixel)
        value_f = float(value)
        if not math.isfinite(pixel_f) or not math.isfinite(value_f):
            raise PaperParityError(f"{label} held-out points must be finite.")
        residual_pixels.append(abs(value_f - (slope * pixel_f + intercept)) / abs(slope))
    maximum = max(residual_pixels)
    declared = calibration.get("held_out_max_residual_px")
    if (
        not isinstance(declared, (int, float))
        or isinstance(declared, bool)
        or not math.isfinite(float(declared))
        or abs(float(declared) - maximum) > 1e-12
    ):
        raise PaperParityError(
            f"{label} held-out residual is stale: declared {declared!r}, recomputed {maximum:.17g}."
        )
    residual_limit = calibration.get("held_out_residual_limit_px")
    if (
        not isinstance(residual_limit, (int, float))
        or isinstance(residual_limit, bool)
        or not math.isfinite(float(residual_limit))
        or float(residual_limit) <= 0.0
        or maximum > float(residual_limit)
    ):
        raise PaperParityError(
            f"{label} held-out residual {maximum:.6g} px exceeds its "
            f"declared acceptance limit {residual_limit!r}."
        )
    return slope, intercept, maximum


def _colour_mask(
    rgb: np.ndarray, channel: str, minimum: float, dominance_ratio: float
) -> np.ndarray:
    channels = {"red": 0, "green": 1, "blue": 2}
    if channel not in channels:
        raise PaperParityError(f"Unsupported colour channel {channel!r}.")
    selected = channels[channel]
    others = [index for index in range(3) if index != selected]
    return (
        (rgb[:, selected] > minimum)
        & (rgb[:, selected] > rgb[:, others[0]] * dominance_ratio)
        & (rgb[:, selected] > rgb[:, others[1]] * dominance_ratio)
    )


def _clusters(rows: list[int]) -> list[list[int]]:
    clusters: list[list[int]] = []
    for row in sorted(set(rows)):
        if not clusters or row > clusters[-1][-1] + 1:
            clusters.append([row])
        else:
            clusters[-1].append(row)
    return clusters


def _read_verified_asset(archive_path: Path, manifest: dict[str, Any]) -> np.ndarray:
    source = _mapping(manifest.get("source"), "manifest.source")
    expected_archive = source.get("archive_sha256")
    actual_archive = file_sha256(archive_path)
    if actual_archive != expected_archive:
        raise PaperParityError(
            f"arXiv source archive SHA-256 mismatch: expected {expected_archive}, "
            f"got {actual_archive}."
        )
    asset = _mapping(manifest.get("asset"), "manifest.asset")
    member_name = asset.get("member")
    if not isinstance(member_name, str) or not member_name:
        raise PaperParityError("manifest.asset.member must be a nonempty string.")
    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            member = archive.getmember(member_name)
            if not member.isfile() or member.size <= 0 or member.size > 100 * 1024 * 1024:
                raise PaperParityError(
                    f"Source member {member_name!r} is not a bounded nonempty file."
                )
            stream = archive.extractfile(member)
            if stream is None:
                raise PaperParityError(f"Cannot read source member {member_name!r}.")
            payload = stream.read()
    except (OSError, tarfile.TarError, KeyError) as exc:
        raise PaperParityError(f"Cannot read {member_name!r} from source archive: {exc}") from exc
    actual_member = hashlib.sha256(payload).hexdigest()
    if actual_member != asset.get("sha256"):
        raise PaperParityError(
            f"Source member SHA-256 mismatch: expected {asset.get('sha256')}, got {actual_member}."
        )

    try:
        from PIL import Image

        with Image.open(io.BytesIO(payload)) as source_image:
            if source_image.format != "PNG":
                raise PaperParityError(
                    f"Source member must decode as PNG; got {source_image.format!r}."
                )
            expected_dpi = asset.get("dpi")
            actual_dpi = source_image.info.get("dpi")
            if (
                not isinstance(expected_dpi, list)
                or not isinstance(actual_dpi, tuple)
                or len(expected_dpi) != 2
                or len(actual_dpi) != 2
                or any(
                    not math.isclose(
                        float(actual),
                        float(expected),
                        rel_tol=0.0,
                        abs_tol=1e-6,
                    )
                    for actual, expected in zip(actual_dpi, expected_dpi, strict=True)
                )
            ):
                raise PaperParityError(
                    f"Source PNG DPI changed: expected {expected_dpi!r}, got {actual_dpi!r}."
                )
            expected_software = asset.get("software")
            actual_software = source_image.info.get("Software")
            if actual_software != expected_software:
                raise PaperParityError(
                    f"Source PNG software metadata changed: expected "
                    f"{expected_software!r}, got {actual_software!r}."
                )
            source_image.load()
            image = np.asarray(source_image)
    except Exception as exc:
        if isinstance(exc, PaperParityError):
            raise
        raise PaperParityError(f"Cannot decode source PNG: {exc}") from exc
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise PaperParityError(f"Source PNG has unexpected shape {image.shape!r}.")
    if image.shape[2] == 4:
        alpha = np.asarray(image[:, :, 3], dtype=float)
        opaque = 1.0 if float(np.max(alpha)) <= 1.0 else 255.0
        if not np.all(np.isfinite(alpha)) or not np.all(alpha == opaque):
            raise PaperParityError(
                "Source PNG must be fully opaque; transparent colour pixels "
                "would make curve masks background-dependent."
            )
    expected_dimensions = asset.get("pixel_dimensions")
    if expected_dimensions != [int(image.shape[1]), int(image.shape[0])]:
        raise PaperParityError(
            f"Source PNG dimensions changed: expected {expected_dimensions!r}, "
            f"got {[int(image.shape[1]), int(image.shape[0])]!r}."
        )
    rgb = np.asarray(image[:, :, :3], dtype=float)
    if float(np.max(rgb)) <= 1.0:
        rgb *= 255.0
    if not np.all(np.isfinite(rgb)) or np.min(rgb) < 0.0 or np.max(rgb) > 255.0:
        raise PaperParityError("Decoded source PNG contains invalid RGB values.")
    return rgb


def extract_points_bytes(archive_path: Path, manifest_path: Path = DEFAULT_MANIFEST) -> bytes:
    """Return the deterministic tidy CSV extracted from the verified source."""

    manifest = load_paper_manifest(manifest_path)
    extraction = _mapping(manifest.get("extraction"), "manifest.extraction")
    if extraction.get("version") != EXTRACTOR_VERSION:
        raise PaperParityError(
            f"Extractor version mismatch: manifest has {extraction.get('version')!r}, "
            f"code has {EXTRACTOR_VERSION!r}."
        )
    image = _read_verified_asset(archive_path, manifest)
    panel = _mapping(manifest.get("panel"), "manifest.panel")
    x_axis = _mapping(panel.get("x_axis"), "manifest.panel.x_axis")
    y_axis = _mapping(panel.get("y_axis"), "manifest.panel.y_axis")
    x_slope, x_intercept, x_calibration_residual = _linear_axis(x_axis, "manifest.panel.x_axis")
    y_slope, y_intercept, y_calibration_residual = _linear_axis(y_axis, "manifest.panel.y_axis")
    if x_slope <= 0.0 or y_slope >= 0.0:
        raise PaperParityError(
            "Fig. 6 expects rightward-positive x and downward-negative y calibration."
        )

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
    ):
        raise PaperParityError("sample_x_values must contain at least three numbers.")
    if any(right <= left for left, right in pairwise(sample_values)):
        raise PaperParityError("sample_x_values must be strictly increasing.")
    half_window = extraction.get("horizontal_half_window_px")
    if (
        not isinstance(half_window, int)
        or isinstance(half_window, bool)
        or not 0 <= half_window <= 5
    ):
        raise PaperParityError("horizontal_half_window_px must be an integer in [0, 5].")
    plot_rows = extraction.get("plot_rows_inclusive")
    if (
        not isinstance(plot_rows, list)
        or len(plot_rows) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) for value in plot_rows)
    ):
        raise PaperParityError("plot_rows_inclusive must contain two integers.")
    row_min, row_max = plot_rows
    if not 0 <= row_min < row_max < image.shape[0]:
        raise PaperParityError("plot_rows_inclusive is outside the source raster.")

    colour_masks = _mapping(extraction.get("color_masks"), "manifest.extraction.color_masks")
    colour_curve_order = extraction.get("color_curve_order")
    if (
        not isinstance(colour_curve_order, list)
        or set(colour_curve_order) != set(colour_masks)
        or len(colour_curve_order) != len(colour_masks)
    ):
        raise PaperParityError("color_curve_order must list every colour mask exactly once.")
    curve_order = extraction.get("curve_order_top_to_bottom")
    if curve_order != ["paper_analytic", "paper_numerical"]:
        raise PaperParityError(
            "curve_order_top_to_bottom must preserve dashed analytic then solid numerical."
        )
    y_window = extraction.get("accepted_y_value_window")
    if (
        not isinstance(y_window, list)
        or len(y_window) != 2
        or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in y_window)
    ):
        raise PaperParityError("accepted_y_value_window must contain two numbers.")
    y_min, y_max = map(float, y_window)
    if not math.isfinite(y_min) or not math.isfinite(y_max) or not y_min < y_max:
        raise PaperParityError("accepted_y_value_window is invalid.")
    identity_policy = extraction.get("trace_identity_policy")
    if identity_policy != "caption-bound analytic-above-numerical; no crossings":
        raise PaperParityError("Unsupported Fig. 6 trace-identity policy.")
    minimum_trace_separation = extraction.get("minimum_trace_separation_px")
    if (
        not isinstance(minimum_trace_separation, (int, float))
        or isinstance(minimum_trace_separation, bool)
        or not math.isfinite(float(minimum_trace_separation))
        or float(minimum_trace_separation) <= 0.0
    ):
        raise PaperParityError("minimum_trace_separation_px must be positive.")
    uncertainty = _mapping(extraction.get("uncertainty"), "manifest.extraction.uncertainty")
    half_pixel_floor = float(uncertainty.get("sampling_half_width_px", math.nan))
    minimum_y = float(uncertainty.get("minimum_y_uncertainty", math.nan))
    if (
        not math.isfinite(half_pixel_floor)
        or half_pixel_floor <= 0.0
        or not math.isfinite(minimum_y)
        or minimum_y <= 0.0
    ):
        raise PaperParityError("Extraction uncertainty controls must be positive.")

    mask_functions: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
    for curve_id in colour_curve_order:
        raw_spec = colour_masks[curve_id]
        if not isinstance(curve_id, str) or not curve_id:
            raise PaperParityError("Colour-mask curve IDs must be nonempty strings.")
        spec = _mapping(raw_spec, f"color_masks.{curve_id}")
        channel = spec.get("channel")
        minimum = spec.get("minimum")
        dominance = spec.get("dominance_ratio")
        if (
            not isinstance(channel, str)
            or not isinstance(minimum, (int, float))
            or isinstance(minimum, bool)
            or not isinstance(dominance, (int, float))
            or isinstance(dominance, bool)
            or not math.isfinite(float(minimum))
            or not math.isfinite(float(dominance))
            or float(minimum) <= 0.0
            or float(dominance) <= 1.0
        ):
            raise PaperParityError(f"Invalid colour mask for {curve_id!r}.")
        mask_functions[curve_id] = partial(
            _colour_mask,
            channel=channel,
            minimum=float(minimum),
            dominance_ratio=float(dominance),
        )

    rows: list[dict[str, object]] = []
    for target_x in map(float, sample_values):
        x_pixel = round((target_x - x_intercept) / x_slope)
        if x_pixel - half_window < 0 or x_pixel + half_window >= image.shape[1]:
            raise PaperParityError(f"Sample x={target_x} maps outside the source raster.")
        x_value = x_slope * x_pixel + x_intercept
        for curve_id, mask_function in mask_functions.items():
            coloured_rows: list[int] = []
            for column in range(x_pixel - half_window, x_pixel + half_window + 1):
                rgb_column = image[row_min : row_max + 1, column, :]
                coloured_rows.extend((np.flatnonzero(mask_function(rgb_column)) + row_min).tolist())
            candidates: list[tuple[float, float, int, int]] = []
            for cluster in _clusters(coloured_rows):
                center = float(np.mean(cluster))
                value = y_slope * center + y_intercept
                if y_min <= value <= y_max:
                    candidates.append((value, center, cluster[0], cluster[-1]))
            if len(candidates) != 2:
                raise PaperParityError(
                    f"Expected exactly two curve traces for {curve_id} at "
                    f"x={target_x}; found {candidates!r}."
                )
            candidates.sort(reverse=True)
            trace_separation = abs(candidates[0][1] - candidates[1][1])
            if trace_separation < float(minimum_trace_separation):
                raise PaperParityError(
                    f"Trace identity is ambiguous for {curve_id} at x={target_x}: "
                    f"separation {trace_separation:.3g} px is below the "
                    f"{float(minimum_trace_separation):.3g} px non-crossing limit."
                )
            for curve_kind, (y_value, y_pixel, cluster_min, cluster_max) in zip(
                curve_order, candidates, strict=True
            ):
                x_uncertainty = abs(x_slope) * (
                    half_window + half_pixel_floor + x_calibration_residual
                )
                y_uncertainty = max(
                    minimum_y,
                    abs(y_slope)
                    * (
                        0.5 * (cluster_max - cluster_min)
                        + half_pixel_floor
                        + y_calibration_residual
                    ),
                )
                rows.append(
                    {
                        "curve_id": curve_id,
                        "curve_kind": curve_kind,
                        "x_pixel": x_pixel,
                        "y_pixel": f"{y_pixel:.6f}",
                        "trace_min_y_pixel": cluster_min,
                        "trace_max_y_pixel": cluster_max,
                        "x_value": f"{x_value:.12g}",
                        "y_value": f"{y_value:.12g}",
                        "x_uncertainty": f"{x_uncertainty:.12g}",
                        "y_uncertainty": f"{y_uncertainty:.12g}",
                    }
                )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(POINT_COLUMNS), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def verify_external_source(archive_path: Path, manifest_path: Path = DEFAULT_MANIFEST) -> bytes:
    """Re-extract points and require exact equality with the checked dataset."""

    generated = extract_points_bytes(archive_path, manifest_path)
    manifest = load_paper_manifest(manifest_path)
    data = _mapping(manifest.get("data"), "manifest.data")
    expected_hash = data.get("sha256")
    actual_hash = hashlib.sha256(generated).hexdigest()
    if actual_hash != expected_hash:
        raise PaperParityError(
            f"Re-extracted points SHA-256 mismatch: expected {expected_hash}, got {actual_hash}."
        )
    raw_checked = data.get("path")
    if not isinstance(raw_checked, str):
        raise PaperParityError("manifest.data.path must be a string.")
    checked = resolve_contained_path(manifest_path.parent, raw_checked, "manifest.data.path")
    if generated != checked.read_bytes():
        raise PaperParityError(
            "Re-extracted points differ byte-for-byte despite their declared hash."
        )
    return generated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="exact arXiv-v2 source tar")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        help="optional output CSV; existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    generated = extract_points_bytes(args.archive, args.manifest)
    manifest = load_paper_manifest(args.manifest)
    expected = _mapping(manifest.get("data"), "manifest.data").get("sha256")
    actual = hashlib.sha256(generated).hexdigest()
    if actual != expected:
        raise PaperParityError(
            f"Generated points hash {actual} does not match manifest {expected}."
        )
    if args.output is None:
        # Write the canonical bytes directly.  Text-mode stdout translates
        # LF to CRLF on Windows, which makes a redirected replay differ from
        # the manifest-bound points.csv despite identical extracted values.
        sys.stdout.buffer.write(generated)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("xb") as stream:
                stream.write(generated)
        except FileExistsError as exc:
            raise PaperParityError(f"Refusing to overwrite existing {args.output}.") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
