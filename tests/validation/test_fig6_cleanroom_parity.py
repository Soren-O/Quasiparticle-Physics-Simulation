from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy
from validation.fischer_2023 import fig6_cleanroom_parity as parity
from validation.fischer_2023.fig6_cleanroom_parity import (
    DEFAULT_SCORE,
    ORACLE_PATH,
    PAPER_POINT_ANCHOR_SCHEMA,
    SCORE_SCHEMA,
    build_score,
)
from validation.paper_parity import (
    load_digitized_points,
    load_strict_json,
)
from validation.paper_parity import (
    score_curve as paper_score_curve,
)
from validation.source_provenance import source_sha256


def _without_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload, allow_nan=False))
    result["producer"].pop("runtime")
    return result


def _expected_anchor(point: Any) -> dict[str, Any]:
    values = {
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


def test_checked_cleanroom_score_is_current() -> None:
    checked = load_strict_json(DEFAULT_SCORE, "checked clean-room score")
    current = build_score()
    assert _without_runtime(checked) == _without_runtime(current)
    assert current["producer"]["runtime"] == {
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
    for path, identity in current["producer"]["sources"].items():
        assert identity["hash_kind"] == "source_canonical_sha256"
        assert identity["sha256"] == source_sha256(
            Path(__file__).resolve().parents[2] / path
        )


def test_cleanroom_score_cannot_overclaim_numerical_validation() -> None:
    score = build_score()
    assert score["schema"] == SCORE_SCHEMA
    assert score["accepted"] is True
    assert score["evidence_class"] == "clean_room_paper_equations"
    assert score["author_source"]["used_by_this_comparison"] is False
    qualifications = score["qualification"]["does_not_claim"]
    assert "agreement with the paper's solid numerical curves" in qualifications
    assert "validation of qpsim's collision kernels or nonlinear solver" in qualifications
    assert all(curve["curve_kind"] == "paper_analytic" for curve in score["curve_scores"])


def test_cleanroom_score_rows_bind_complete_paper_points() -> None:
    score = build_score()
    _oracle, digitized = load_digitized_points(ORACLE_PATH)
    expected = {
        (point.curve_id, point.curve_kind, point.x_value): _expected_anchor(point)
        for point in digitized
        if point.curve_kind == "paper_analytic"
    }
    for curve in score["curve_scores"]:
        for point in curve["points"]:
            key = (curve["curve_id"], curve["curve_kind"], point["x_value"])
            assert point["paper_point"] == expected[key]
            assert point["paper_y"] == point["paper_point"]["values"]["y_value"]


def test_cleanroom_mismatch_cannot_emit_an_acceptance_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def rejected_score(*args: Any, **kwargs: Any) -> Any:
        score = paper_score_curve(*args, **kwargs)
        return replace(
            score,
            accepted=False,
            max_uncertainty_normalized_error=2.0,
        )

    monkeypatch.setattr(parity, "score_curve", rejected_score)
    score = build_score()
    assert score["accepted"] is False
    assert score["status"] == "diagnostic_mismatch"
    assert "does not agree" in score["claim"]
    assert "not acceptance evidence" in score["claim"]


def test_cleanroom_source_change_during_scoring_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = parity._source_snapshot()
    calls = 0

    def changing_snapshot() -> dict[str, str]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        changed = dict(initial)
        first = next(iter(changed))
        changed[first] = "0" * 64
        return changed

    monkeypatch.setattr(parity, "_source_snapshot", changing_snapshot)
    with pytest.raises(RuntimeError, match="source changed during scoring"):
        build_score()


def test_cleanroom_stale_imported_source_fails_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = parity._source_snapshot()
    first = next(iter(changed))
    changed[first] = "0" * 64
    monkeypatch.setattr(parity, "_source_snapshot", lambda: changed)
    with pytest.raises(RuntimeError, match="no longer matches"):
        build_score()


def test_checked_score_lives_beside_the_authenticated_oracle() -> None:
    expected_parent = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "paper_data"
        / "fischer_2023"
        / "fig6"
    )
    assert DEFAULT_SCORE.parent == expected_parent
