"""Provenance and clean-room parity tests for Fischer-2023 Fig. 8."""

from __future__ import annotations

import ast
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
from validation.fischer_2023.extract_fig8_paper_data import verify_external_source
from validation.fischer_2023.fig8_cleanroom_parity import (
    DEFAULT_SCORE,
    DEFAULT_SPEC,
    build_score,
    verify_checked_score,
)
from validation.paper_parity import (
    POINT_COLUMNS,
    PaperParityError,
    load_digitized_points,
)
from validation.reference_models.fischer_2023 import fig8_analytic

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIG6_ORACLE = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "oracle.json"
)
ORACLE_DIRECTORY = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig8"
)
ORACLE_PATH = ORACLE_DIRECTORY / "oracle.json"
POINTS_PATH = ORACLE_DIRECTORY / "points.csv"
MODEL_SOURCE = (
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig8_analytic.py"
)


def _manifest() -> dict[str, object]:
    value = json.loads(ORACLE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_oracle_bundle(
    directory: Path,
    manifest: dict[str, object],
    points_bytes: bytes,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "points.csv").write_bytes(points_bytes)
    data = manifest["data"]
    assert isinstance(data, dict)
    data["sha256"] = hashlib.sha256(points_bytes).hexdigest()
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


def test_legacy_fig6_order_contract_still_loads() -> None:
    manifest, points = load_digitized_points(FIG6_ORACLE)
    assert manifest["extraction"]["trace_identity_policy"] == (
        "caption-bound analytic-above-numerical; no crossings"
    )
    assert len(points) == 42


def test_fig8_reverse_order_and_kind_specific_masks_load() -> None:
    manifest, points = load_digitized_points(ORACLE_PATH)
    assert manifest["asset"]["sha256"] == (
        "4325a6c8e4e44d72c252848256285cd25306f8b230160e6d9e1a4f9cf47b6cda"
    )
    assert manifest["extraction"]["trace_identity_policy"] == (
        "caption-bound numerical-above-analytic; no crossings"
    )
    assert len(points) == 20

    paired: dict[int, dict[str, object]] = {}
    for point in points:
        paired.setdefault(point.x_pixel, {})[point.curve_kind] = point
    assert len(paired) == 10
    for kinds in paired.values():
        numerical = kinds["paper_numerical"]
        analytic = kinds["paper_analytic"]
        assert numerical.y_pixel < analytic.y_pixel

    # At low x the antialiased trace bands overlap, but separate colour/style
    # masks preserve identity without using an invented geometric gap.
    assert any(
        kinds["paper_numerical"].trace_max_y_pixel
        >= kinds["paper_analytic"].trace_min_y_pixel
        for kinds in paired.values()
    )


def test_rehashed_swapped_reverse_trace_identity_is_rejected(
    tmp_path: Path,
) -> None:
    rows = list(
        csv.reader(io.StringIO(POINTS_PATH.read_text(encoding="utf-8"), newline=""))
    )
    assert rows[0] == list(POINT_COLUMNS)
    fields = [
        rows[0].index(name)
        for name in (
            "y_pixel",
            "trace_min_y_pixel",
            "trace_max_y_pixel",
            "y_value",
            "y_uncertainty",
        )
    ]
    for index in fields:
        rows[1][index], rows[2][index] = rows[2][index], rows[1][index]
    path = _write_oracle_bundle(tmp_path, _manifest(), _csv_bytes(rows))

    with pytest.raises(PaperParityError, match="numerical-above-analytic"):
        load_digitized_points(path)


def test_cleanroom_eq_e2_matches_digitized_dashed_curve() -> None:
    score = build_score()
    assert score["accepted"] is True
    curve = score["curve_score"]
    assert curve["curve_kind"] == "paper_analytic"
    assert curve["max_uncertainty_normalized_error"] < 0.30
    assert score["qualification"]["does_not_claim"][0] == (
        "reproduction of the blue solid numerical curve"
    )


def test_checked_cleanroom_score_is_current() -> None:
    score = verify_checked_score(DEFAULT_SCORE, DEFAULT_SPEC)
    assert score["status"] == "accepted_clean_room_analytic"
    assert score["qualification"]["gate_eligible"] is False


def test_cleanroom_model_has_no_qpsim_import() -> None:
    tree = ast.parse(MODEL_SOURCE.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "qpsim" or name.startswith("qpsim.") for name in imported)


@pytest.mark.parametrize(
    "bad",
    [
        np.asarray([0.0, 0.0]),
        np.asarray([0.0, np.nan]),
        np.asarray([-0.1, 0.1]),
        np.asarray([False, True]),
        np.asarray([0.0 + 0.0j, 0.1 + 0.0j]),
    ],
)
def test_cleanroom_model_rejects_invalid_grids(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fig8_analytic.fig8_analytic_curve(bad)


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
    replay = subprocess.run(
        [
            sys.executable,
            "-m",
            "validation.fischer_2023.extract_fig8_paper_data",
            str(archive),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    assert replay.stdout == generated
