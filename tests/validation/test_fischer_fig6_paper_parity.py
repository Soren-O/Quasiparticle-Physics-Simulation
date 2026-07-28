"""Fail-closed tests for the Fischer-2023 Fig. 6 paper-data comparison."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023 import fig6_paper_parity as fig6_parity
from validation.fischer_2023.extract_fig6_paper_data import (
    verify_external_source,
)
from validation.fischer_2023.fig6_paper_parity import (
    DEFAULT_SCORE,
    DEFAULT_SPEC,
    build_score,
    verify_checked_score,
)
from validation.paper_parity import (
    POINT_COLUMNS,
    DigitizedPoint,
    PaperParityError,
    load_digitized_points,
    load_paper_manifest,
    score_curve,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIRECTORY = REPOSITORY_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6"
ORACLE_PATH = ORACLE_DIRECTORY / "oracle.json"
POINTS_PATH = ORACLE_DIRECTORY / "points.csv"


def _manifest() -> dict[str, object]:
    value = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_oracle_bundle(
    directory: Path,
    manifest: dict[str, object],
    points_bytes: bytes = POINTS_PATH.read_bytes(),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "points.csv").write_bytes(points_bytes)
    path = directory / "oracle.json"
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _csv_bytes(rows: list[list[str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _bind_points(manifest: dict[str, object], payload: bytes) -> None:
    data = manifest["data"]
    assert isinstance(data, dict)
    data["sha256"] = hashlib.sha256(payload).hexdigest()


def _point(
    x_value: float,
    *,
    x_uncertainty: float = 0.01,
    curve_kind: str = "paper_numerical",
) -> DigitizedPoint:
    return DigitizedPoint(
        curve_id="synthetic",
        curve_kind=curve_kind,
        x_pixel=round(100.0 * x_value),
        y_pixel=50.0,
        trace_min_y_pixel=49,
        trace_max_y_pixel=51,
        x_value=x_value,
        y_value=x_value,
        x_uncertainty=x_uncertainty,
        y_uncertainty=0.01,
    )


def test_manifest_rejects_an_unknown_field(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["unexpected"] = "must fail closed"
    path = _write_oracle_bundle(tmp_path, manifest)

    with pytest.raises(PaperParityError, match="manifest fields are invalid"):
        load_paper_manifest(path)


def test_points_byte_tamper_rejects_stale_manifest_hash(tmp_path: Path) -> None:
    path = _write_oracle_bundle(
        tmp_path,
        _manifest(),
        POINTS_PATH.read_bytes() + b"\n",
    )

    with pytest.raises(PaperParityError, match="SHA-256 mismatch"):
        load_digitized_points(path)


def test_rehashed_coordinate_tamper_still_fails_calibration(
    tmp_path: Path,
) -> None:
    rows = list(csv.reader(io.StringIO(POINTS_PATH.read_text(encoding="utf-8"), newline="")))
    assert rows[0] == list(POINT_COLUMNS)
    x_value_index = rows[0].index("x_value")
    rows[1][x_value_index] = str(float(rows[1][x_value_index]) + 0.01)

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    tampered = output.getvalue().encode("utf-8")

    manifest = _manifest()
    data = manifest["data"]
    assert isinstance(data, dict)
    data["sha256"] = hashlib.sha256(tampered).hexdigest()
    path = _write_oracle_bundle(tmp_path, manifest, tampered)

    with pytest.raises(PaperParityError, match="inconsistent with calibration"):
        load_digitized_points(path)


def test_rehashed_shifted_sample_grid_is_rejected(tmp_path: Path) -> None:
    rows = list(csv.reader(io.StringIO(POINTS_PATH.read_text(encoding="utf-8"), newline="")))
    header = rows[0]
    x_pixel_index = header.index("x_pixel")
    x_value_index = header.index("x_value")
    for row in rows[1:]:
        if row[x_pixel_index] == "221":
            row[x_pixel_index] = "222"
            row[x_value_index] = f"{0.0010273972602739725 * 222 + 0.022602739726027388:.12g}"
    tampered = _csv_bytes(rows)
    manifest = _manifest()
    _bind_points(manifest, tampered)
    path = _write_oracle_bundle(tmp_path, manifest, tampered)

    with pytest.raises(PaperParityError, match="sample_x_values geometry"):
        load_digitized_points(path)


def test_rehashed_trace_bounds_outside_plot_are_rejected(tmp_path: Path) -> None:
    rows = list(csv.reader(io.StringIO(POINTS_PATH.read_text(encoding="utf-8"), newline="")))
    trace_min_index = rows[0].index("trace_min_y_pixel")
    rows[1][trace_min_index] = "23"
    tampered = _csv_bytes(rows)
    manifest = _manifest()
    _bind_points(manifest, tampered)
    path = _write_oracle_bundle(tmp_path, manifest, tampered)

    with pytest.raises(PaperParityError, match="panel/plot rows"):
        load_digitized_points(path)


def test_rehashed_swapped_trace_identity_is_rejected(tmp_path: Path) -> None:
    rows = list(csv.reader(io.StringIO(POINTS_PATH.read_text(encoding="utf-8"), newline="")))
    header = rows[0]
    trace_fields = [
        header.index(name)
        for name in (
            "y_pixel",
            "trace_min_y_pixel",
            "trace_max_y_pixel",
            "y_value",
            "y_uncertainty",
        )
    ]
    for index in trace_fields:
        rows[1][index], rows[2][index] = rows[2][index], rows[1][index]
    tampered = _csv_bytes(rows)
    manifest = _manifest()
    _bind_points(manifest, tampered)
    path = _write_oracle_bundle(tmp_path, manifest, tampered)

    with pytest.raises(PaperParityError, match="analytic-above-numerical"):
        load_digitized_points(path)


def test_curve_family_cannot_split_across_temperatures(tmp_path: Path) -> None:
    manifest = _manifest()
    curves = manifest["curves"]
    assert isinstance(curves, list)
    numerical = next(
        curve
        for curve in curves
        if curve["curve_id"] == "T_bath_0.10K" and curve["curve_kind"] == "paper_numerical"
    )
    numerical["temperature_K"] = 0.15
    path = _write_oracle_bundle(tmp_path, manifest)

    with pytest.raises(PaperParityError, match="different temperatures"):
        load_digitized_points(path)


@pytest.mark.parametrize(
    ("qpsim_x", "qpsim_y"),
    [
        (
            np.asarray([0.0 + 0.0j, 0.5 + 0.0j, 1.0 + 0.0j]),
            np.asarray([0.0, 0.5, 1.0]),
        ),
        (
            np.asarray([False, True]),
            np.asarray([0.0, 1.0]),
        ),
    ],
    ids=["complex", "boolean"],
)
def test_score_curve_rejects_lossy_qpsim_dtypes(
    qpsim_x: np.ndarray,
    qpsim_y: np.ndarray,
) -> None:
    points = [_point(0.25), _point(0.50), _point(0.75)]

    with pytest.raises(PaperParityError, match="real floating-point or integer"):
        score_curve(
            points,
            qpsim_x,
            qpsim_y,
            uncertainty_normalized_limit=1.0,
        )


def test_score_curve_rejects_unsorted_paper_coordinates() -> None:
    points = [_point(0.25), _point(0.75), _point(0.50)]

    with pytest.raises(PaperParityError, match="strictly increasing"):
        score_curve(
            points,
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, 0.5, 1.0]),
            uncertainty_normalized_limit=1.0,
        )


def test_score_curve_rejects_out_of_domain_uncertainty_interval() -> None:
    points = [
        _point(0.05, x_uncertainty=0.10),
        _point(0.50),
        _point(0.90),
    ]

    with pytest.raises(PaperParityError, match="x-uncertainty interval"):
        score_curve(
            points,
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, 0.5, 1.0]),
            uncertainty_normalized_limit=1.0,
        )


def test_score_curve_rejects_boolean_acceptance_limit() -> None:
    points = [_point(0.25), _point(0.50), _point(0.75)]

    with pytest.raises(PaperParityError, match="finite and positive"):
        score_curve(
            points,
            np.asarray([0.0, 0.5, 1.0]),
            np.asarray([0.0, 0.5, 1.0]),
            uncertainty_normalized_limit=True,
        )


def test_current_oracle_control_passes_while_numerical_curves_mismatch() -> None:
    score = build_score(DEFAULT_SPEC)
    result = score["result"]
    assert isinstance(result, dict)
    assert result == {
        "analytic_control_accepted": True,
        "gate_eligible": False,
        "numerical_parity_accepted": False,
        "status": "diagnostic_mismatch",
    }

    curve_scores = score["curve_scores"]
    assert isinstance(curve_scores, list)
    analytic = [curve for curve in curve_scores if curve["curve_kind"] == "paper_analytic"]
    numerical = [curve for curve in curve_scores if curve["curve_kind"] == "paper_numerical"]
    assert len(analytic) == len(numerical) == 3
    assert all(curve["accepted"] is True for curve in analytic)
    assert all(curve["accepted"] is False for curve in numerical)


def test_inverted_solid_dashed_mapping_is_rejected(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    assert isinstance(spec, dict)
    mappings = spec["mappings"]
    assert isinstance(mappings, list)
    first = mappings[0]
    assert isinstance(first, dict)
    assert first["curve_kind"] == "paper_analytic"
    first["qpsim_y_column"] = "paper_observable_num"

    path = tmp_path / "comparison-spec.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperParityError, match="mapping is inverted"):
        build_score(path)


def test_solver_replay_qualification_cannot_be_overclaimed(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    assert isinstance(spec, dict)
    error_budget = spec["error_budget"]
    assert isinstance(error_budget, dict)
    components = error_budget["components"]
    assert isinstance(components, list)
    solver = next(
        component for component in components if component["name"] == "solver_certification"
    )
    solver["status"] = "bounded"

    path = tmp_path / "comparison-spec.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperParityError, match="solver-replay"):
        build_score(path)


def test_error_budget_semantics_cannot_be_disabled(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    error_budget = spec["error_budget"]
    components = error_budget["components"]
    raster = next(
        component for component in components if component["name"] == "raster_digitization"
    )
    raster["included_in_metric"] = False
    raster["status"] = "unbounded"
    error_budget["gate_ineligibility_reasons"] = ["banana"]

    path = tmp_path / "comparison-spec.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperParityError, match="error-budget contracts"):
        build_score(path)


def test_observable_normalization_cannot_change_to_percent(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text(encoding="utf-8"))
    identity = spec["observable_identity"]
    identity["y"]["conversion"] = "multiply_by_100_percent"

    path = tmp_path / "comparison-spec.json"
    path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperParityError, match="unit/normalization identity"):
        build_score(path)


def test_score_verification_rebuilds_only_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_build_score(_spec_path: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"snapshot": calls}

    monkeypatch.setattr(fig6_parity, "build_score", fake_build_score)
    expected = (
        json.dumps(
            {"snapshot": 1},
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    score_path = tmp_path / "score.json"
    score_path.write_bytes(expected)

    assert verify_checked_score(score_path, DEFAULT_SPEC) == {"snapshot": 1}
    assert calls == 1


def test_checked_score_is_current() -> None:
    assert DEFAULT_SCORE.is_file(), "The promoted Fig. 6 score is missing."

    score = verify_checked_score(DEFAULT_SCORE, DEFAULT_SPEC)
    result = score["result"]
    assert isinstance(result, dict)
    assert result["status"] == "diagnostic_mismatch"
    inputs = score["inputs"]
    assert isinstance(inputs, dict)
    qpsim_artifact = inputs["qpsim_artifact"]
    assert isinstance(qpsim_artifact, dict)
    assert qpsim_artifact["authentication"] == {
        "authoritative_replay": ("separate slow gate under the recorded single-thread environment"),
        "status": "promotion_attested_not_replayed",
    }


def test_exact_external_source_replays_checked_points() -> None:
    raw_archive = os.environ.get("QPSIM_FISCHER2023_ARXIV_SOURCE")
    if not raw_archive:
        pytest.skip(
            "Set QPSIM_FISCHER2023_ARXIV_SOURCE to the exact arXiv-v2 "
            "source archive to replay the paper extraction."
        )

    archive = Path(raw_archive)
    assert archive.is_file(), f"Configured arXiv source archive is missing: {archive}"
    generated = verify_external_source(archive, ORACLE_PATH)
    assert generated == POINTS_PATH.read_bytes()

    # The CLI must emit those exact bytes too. Text-mode stdout translated
    # LF to CRLF on Windows before this regression guard was added.
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.fischer_2023.extract_fig6_paper_data",
            str(archive),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
    )
    assert replay.stdout == generated
