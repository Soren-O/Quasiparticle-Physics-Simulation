"""Compare the bundled author-output PNG with the published Fig. 6 raster."""

from __future__ import annotations

import hashlib
import io
import json
import math
import platform
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image

from validation.author_source import (
    AuthenticatedMember,
    authenticate_author_source,
)
from validation.paper_parity import (
    DigitizedPoint,
    lf_canonical_sha256,
    load_digitized_points,
    parse_strict_json_bytes,
    resolve_contained_path,
    score_curve,
    select_curve,
)
from validation.source_provenance import source_sha256

SCORE_SCHEMA = "qpsim.fischer2023.fig6-author-output-score.v1"
PAPER_POINT_ANCHOR_SCHEMA = "qpsim.paper-point-anchor.v1"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6"
ORACLE_PATH = ORACLE_DIRECTORY / "oracle.json"
AUTHOR_MANIFEST = ORACLE_DIRECTORY / "author-source.json"
DEFAULT_SCORE = ORACLE_DIRECTORY / "author-output-score.json"
AUTHOR_SOURCE_MODULE = REPOSITORY_ROOT / "validation" / "author_source.py"
PAPER_PARITY_MODULE = REPOSITORY_ROOT / "validation" / "paper_parity.py"
SOURCE_PROVENANCE_MODULE = REPOSITORY_ROOT / "validation" / "source_provenance.py"
PRODUCER_SOURCE_PATHS = (
    Path(__file__).resolve(),
    AUTHOR_SOURCE_MODULE,
    PAPER_PARITY_MODULE,
    SOURCE_PROVENANCE_MODULE,
)

X_CONTROL_PIXELS = np.array([2716.0, 3700.0, 4684.0, 5668.0])
X_CONTROL_VALUES = np.array([0.30, 0.40, 0.50, 0.60])
Y_CONTROL_PIXELS = np.array([981.0, 1723.0, 2464.0, 3206.0])
Y_CONTROL_VALUES = np.array([0.20, 0.15, 0.10, 0.05])
SAMPLE_X = np.linspace(0.245, 0.415, 171)
HORIZONTAL_HALF_WINDOW_PIXELS = 12
GROUP_GAP_PIXELS = 20
MINIMUM_GROUP_ROWS = 5


@dataclass(frozen=True)
class ExtractedCurve:
    """One trace extracted from the authenticated high-resolution PNG."""

    x: np.ndarray
    y: np.ndarray
    y_uncertainty: np.ndarray


@dataclass(frozen=True)
class PaperEvidenceSnapshot:
    """Paper manifest, points, and digitizer retained as one byte snapshot."""

    files: dict[Path, bytes]
    manifest: dict[str, Any]
    points_path: Path
    digitizer_path: Path


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
            "Figure 6 author-output producer source no longer matches the "
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
            "Figure 6 author-output producer source changed during scoring: "
            + ", ".join(changed)
        )


def _paper_evidence_snapshot() -> PaperEvidenceSnapshot:
    """Freeze every paper-evidence file before validation and scoring."""

    try:
        manifest_bytes = ORACLE_PATH.read_bytes()
    except OSError as exc:
        raise ValueError(f"Cannot snapshot paper oracle {ORACLE_PATH}: {exc}") from exc
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
        raise ValueError(f"Cannot snapshot paper points {points_path}: {exc}") from exc
    extraction = manifest.get("extraction")
    if (
        not isinstance(extraction, dict)
        or not isinstance(extraction.get("script"), str)
    ):
        raise ValueError("Figure 6 paper-data manifest has no extraction.script.")
    digitizer_path = resolve_contained_path(
        REPOSITORY_ROOT,
        extraction["script"],
        "Figure 6 paper-data manifest.extraction.script",
    )
    try:
        digitizer_bytes = digitizer_path.read_bytes()
    except OSError as exc:
        raise ValueError(
            f"Cannot snapshot paper digitizer {digitizer_path}: {exc}"
        ) from exc
    return PaperEvidenceSnapshot(
        files={
            ORACLE_PATH: manifest_bytes,
            points_path: points_bytes,
            digitizer_path: digitizer_bytes,
        },
        manifest=manifest,
        points_path=points_path,
        digitizer_path=digitizer_path,
    )


def _load_digitized_points_from_snapshot(
    snapshot: PaperEvidenceSnapshot,
) -> list[DigitizedPoint]:
    """Validate/parse only retained bytes, never the mutable originals."""

    # paper_parity's public loader is path-based. Materialize a private,
    # repository-contained mirror so its digitizer-containment rule still
    # applies, while every consumed byte comes from ``snapshot``.
    with tempfile.TemporaryDirectory(
        prefix=".qpsim-fig6-paper-snapshot-",
        dir=REPOSITORY_ROOT,
    ) as temporary:
        root = Path(temporary)
        points_path = root / "points.csv"
        digitizer_path = root / "digitizer.py"
        oracle_path = root / "oracle.json"
        points_path.write_bytes(snapshot.files[snapshot.points_path])
        digitizer_path.write_bytes(snapshot.files[snapshot.digitizer_path])
        mirror = json.loads(
            json.dumps(snapshot.manifest, allow_nan=False)
        )
        mirror["data"]["path"] = points_path.name
        mirror["extraction"]["script"] = digitizer_path.relative_to(
            REPOSITORY_ROOT
        ).as_posix()
        oracle_path.write_text(
            json.dumps(mirror, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _mirror_manifest, points = load_digitized_points(oracle_path)
    return points


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
            "Pillow": PIL.__version__,
            "numpy": np.__version__,
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


def _member_by_role(
    members: tuple[AuthenticatedMember, ...],
    role: str,
) -> AuthenticatedMember:
    matches = [member for member in members if member.role == role]
    if len(matches) != 1:
        raise ValueError(
            f"Authenticated author source must bind one {role!r} member; "
            f"found {len(matches)}."
        )
    return matches[0]


def _line_fit(
    pixel: np.ndarray,
    value: np.ndarray,
) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(pixel, value, 1)
    residual = float(np.max(np.abs(value - (slope * pixel + intercept))))
    return float(slope), float(intercept), residual


def _color_masks(rgb: np.ndarray) -> dict[str, np.ndarray]:
    red = rgb[:, :, 0].astype(float)
    green = rgb[:, :, 1].astype(float)
    blue = rgb[:, :, 2].astype(float)
    return {
        "T_bath_0.10K": (green > 60.0) & (green > 1.5 * red) & (green > 1.5 * blue),
        "T_bath_0.15K": (blue > 100.0) & (blue > 1.5 * red) & (blue > 1.5 * green),
        "T_bath_0.20K": (red > 120.0) & (red > 1.5 * green) & (red > 1.5 * blue),
    }


def _groups_at_x(mask: np.ndarray, x_pixel: int) -> list[np.ndarray]:
    lower = x_pixel - HORIZONTAL_HALF_WINDOW_PIXELS
    upper = x_pixel + HORIZONTAL_HALF_WINDOW_PIXELS + 1
    if lower < 0 or upper > mask.shape[1]:
        raise ValueError("Author-output extraction window leaves the raster.")
    rows = np.unique(np.where(mask[:, lower:upper])[0])
    if rows.size == 0:
        return []
    starts = np.r_[0, np.where(np.diff(rows) > GROUP_GAP_PIXELS)[0] + 1]
    ends = np.r_[starts[1:], rows.size]
    return [
        rows[start:end]
        for start, end in zip(starts, ends, strict=True)
        if end - start >= MINIMUM_GROUP_ROWS
    ]


def extract_author_output(png_bytes: bytes) -> dict[tuple[str, str], ExtractedCurve]:
    """Extract solid and dashed curves from the exact bundled author PNG."""

    with Image.open(io.BytesIO(png_bytes)) as image:
        if image.size != (6400, 4800):
            raise ValueError(f"Unexpected author-output dimensions {image.size!r}.")
        dpi = image.info.get("dpi")
        if (
            not isinstance(dpi, tuple)
            or len(dpi) != 2
            or any(abs(float(value) - 1000.0) > 0.01 for value in dpi)
        ):
            raise ValueError(f"Unexpected author-output DPI metadata {dpi!r}.")
        rgb = np.asarray(image.convert("RGB"))

    x_slope, x_intercept, x_residual = _line_fit(X_CONTROL_PIXELS, X_CONTROL_VALUES)
    y_slope, y_intercept, y_residual = _line_fit(Y_CONTROL_PIXELS, Y_CONTROL_VALUES)
    if x_residual > 1e-12 or y_residual > 3e-5:
        raise ValueError("Author-output axis calibration exceeds its declared residual bound.")

    curves: dict[tuple[str, str], ExtractedCurve] = {}
    for curve_id, mask in _color_masks(rgb).items():
        medians: dict[str, list[float]] = {
            "paper_analytic": [],
            "paper_numerical": [],
        }
        uncertainties: dict[str, list[float]] = {
            "paper_analytic": [],
            "paper_numerical": [],
        }
        for x_value in SAMPLE_X:
            x_pixel = round((float(x_value) - x_intercept) / x_slope)
            groups = _groups_at_x(mask, x_pixel)
            if len(groups) < 2:
                raise ValueError(
                    f"Could not isolate two traces for {curve_id} at x={x_value:.6g}."
                )
            # Over the authenticated rising-branch window, the caption-bound
            # dashed analytic trace is above the solid numerical trace.
            for curve_kind, group in zip(
                ("paper_analytic", "paper_numerical"),
                groups[:2],
                strict=True,
            ):
                median_pixel = float(np.median(group))
                medians[curve_kind].append(y_slope * median_pixel + y_intercept)
                trace_half_height = 0.5 * float(group[-1] - group[0])
                uncertainties[curve_kind].append(
                    abs(y_slope) * (trace_half_height + 1.5) + y_residual
                )
        for curve_kind in medians:
            curves[(curve_id, curve_kind)] = ExtractedCurve(
                x=SAMPLE_X.copy(),
                y=np.asarray(medians[curve_kind], dtype=float),
                y_uncertainty=np.asarray(uncertainties[curve_kind], dtype=float),
            )
    return curves


def _stable_float(value: float) -> float:
    return float(f"{value:.12g}")


def _score_curve_pair(
    paper_points: list[Any],
    author_curve: ExtractedCurve,
) -> dict[str, Any]:
    base = score_curve(
        paper_points,
        author_curve.x,
        author_curve.y,
        uncertainty_normalized_limit=1.0,
    )
    point_rows: list[dict[str, float]] = []
    normalized_errors: list[float] = []
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    for point, base_point in zip(paper_points, base.points, strict=True):
        author_uncertainty = float(
            np.interp(point.x_value, author_curve.x, author_curve.y_uncertainty)
        )
        effective = base_point.effective_uncertainty + author_uncertainty
        normalized = base_point.absolute_error / effective
        point_row: dict[str, Any] = {
            "absolute_error": _stable_float(base_point.absolute_error),
            "author_output_y": _stable_float(base_point.qpsim_y),
            "author_raster_uncertainty": _stable_float(author_uncertainty),
            "effective_uncertainty": _stable_float(effective),
            "paper_point": _paper_point_anchor(point),
            "paper_y": _stable_float(base_point.paper_y),
            "relative_error": _stable_float(base_point.relative_error),
            "uncertainty_normalized_error": _stable_float(normalized),
            "x_value": _stable_float(base_point.x_value),
        }
        point_rows.append(point_row)
        normalized_errors.append(normalized)
        absolute_errors.append(base_point.absolute_error)
        relative_errors.append(base_point.relative_error)
    return {
        "accepted": max(normalized_errors) <= 1.0,
        "curve_id": base.curve_id,
        "curve_kind": base.curve_kind,
        "max_absolute_error": _stable_float(max(absolute_errors)),
        "max_relative_error": _stable_float(max(relative_errors)),
        "max_uncertainty_normalized_error": _stable_float(max(normalized_errors)),
        "mean_absolute_error": _stable_float(float(np.mean(absolute_errors))),
        "point_count": len(point_rows),
        "points": point_rows,
        "rms_error": _stable_float(
            math.sqrt(float(np.mean(np.square(absolute_errors))))
        ),
        "uncertainty_normalized_limit": 1.0,
    }


def build_score(archive_path: Path | None = None) -> dict[str, Any]:
    """Authenticate, extract, and compare the bundled author output."""

    source_snapshot = _begin_source_snapshot()
    paper_snapshot = _paper_evidence_snapshot()
    authenticated = authenticate_author_source(
        AUTHOR_MANIFEST,
        archive_path,
    )
    bundled_output = _member_by_role(
        authenticated.members,
        "bundled_output_oracle",
    )
    png_bytes = bundled_output.content
    curves = extract_author_output(png_bytes)
    bundled_output_sha256 = hashlib.sha256(png_bytes).hexdigest()
    if bundled_output_sha256 != bundled_output.sha256:
        raise ValueError("Authenticated bundled-output bytes disagree with their manifest.")
    oracle = paper_snapshot.manifest
    paper_points = _load_digitized_points_from_snapshot(paper_snapshot)
    scores: list[dict[str, Any]] = []
    for curve_id in ("T_bath_0.10K", "T_bath_0.15K", "T_bath_0.20K"):
        for curve_kind in ("paper_analytic", "paper_numerical"):
            scores.append(
                _score_curve_pair(
                    select_curve(paper_points, curve_id, curve_kind),
                    curves[(curve_id, curve_kind)],
                )
            )
    accepted = all(bool(score["accepted"]) for score in scores)
    if accepted:
        status = "accepted_author_output_identity"
        claim = (
            "The bundled output in the authenticated author attachment agrees "
            "with the published arXiv-v2 Figure 6 raster within the combined "
            "high-resolution and publication-raster digitization bounds."
        )
    else:
        status = "diagnostic_mismatch"
        claim = (
            "The bundled output in the authenticated author attachment does "
            "not agree with every published arXiv-v2 Figure 6 trace within "
            "the combined high-resolution and publication-raster digitization "
            "bounds; this payload is diagnostic and is not acceptance evidence."
        )
    payload = {
        "accepted": accepted,
        "author_source": {
            "archive_sha256": authenticated.archive_sha256,
            "bundled_output_sha256": bundled_output_sha256,
            "manifest_path": AUTHOR_MANIFEST.relative_to(REPOSITORY_ROOT).as_posix(),
            "manifest_sha256": authenticated.manifest_sha256,
            "source_id": authenticated.source_id,
        },
        "claim": claim,
        "curve_scores": scores,
        "does_not_claim": [
            "that the current runtime regenerated the bundled PNG",
            "that the attachment is the historical publication checkout",
            "that the author equations are physically correct",
        ],
        "evidence_class": (
            "authenticated_author_output_identity"
            if accepted
            else "authenticated_author_output_mismatch_diagnostic"
        ),
        "extraction": {
            "horizontal_half_window_pixels": HORIZONTAL_HALF_WINDOW_PIXELS,
            "image_dimensions": [6400, 4800],
            "image_dpi": 1000.0,
            "sample_count_per_curve": int(SAMPLE_X.size),
            "script_path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(),
            "script_sha256": source_snapshot[
                Path(__file__).relative_to(REPOSITORY_ROOT).as_posix()
            ],
            "trace_identity": "caption-bound dashed-above-solid on rising branch",
            "x_control_pixels": X_CONTROL_PIXELS.astype(int).tolist(),
            "x_control_values": X_CONTROL_VALUES.tolist(),
            "y_control_pixels": Y_CONTROL_PIXELS.astype(int).tolist(),
            "y_control_values": Y_CONTROL_VALUES.tolist(),
        },
        "paper_oracle": {
            "archive_sha256": oracle["source"]["archive_sha256"],
            "manifest_path": ORACLE_PATH.relative_to(REPOSITORY_ROOT).as_posix(),
            "manifest_sha256": hashlib.sha256(
                paper_snapshot.files[ORACLE_PATH]
            ).hexdigest(),
            "points_path": paper_snapshot.points_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            # Content-defined: tests/validation/test_fig6_author_output_parity.py
            # asserts this equals the oracle manifest's `data.sha256`, which is
            # itself the LF digest, so hashing the CSV raw made that identity
            # hold only on an LF checkout.
            "points_sha256": lf_canonical_sha256(
                paper_snapshot.files[paper_snapshot.points_path]
            ),
        },
        "producer": {
            "runtime": _runtime_provenance(),
            "sources": {
                path: {
                    "execution_binding": (
                        "import-time on-disk source snapshot, required "
                        "unchanged before and after scoring; not an independent "
                        "interpreter-bytecode attestation"
                    ),
                    "hash_kind": "canonical_sha256_import_time_disk_snapshot",
                    "sha256": digest,
                }
                for path, digest in source_snapshot.items()
            },
        },
        "schema": SCORE_SCHEMA,
        "status": status,
    }
    _assert_file_snapshot(paper_snapshot.files, "Figure 6 paper evidence")
    _assert_source_snapshot(source_snapshot)
    return payload


def write_score(
    path: Path = DEFAULT_SCORE,
    *,
    archive_path: Path | None = None,
) -> Path:
    """Write the canonical author-output identity score."""

    payload = build_score(archive_path)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


if __name__ == "__main__":
    print(write_score())
