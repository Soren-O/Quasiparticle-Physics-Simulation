from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import PIL
import pytest
from validation.author_source import (
    AuthenticatedAuthorSource,
    AuthenticatedMember,
    load_author_source_manifest,
)
from validation.fischer_2023 import fig6_author_output_parity as parity
from validation.fischer_2023.fig6_author_output_parity import (
    AUTHOR_MANIFEST,
    DEFAULT_SCORE,
    ORACLE_PATH,
    PAPER_POINT_ANCHOR_SCHEMA,
    SCORE_SCHEMA,
    ExtractedCurve,
    build_score,
)
from validation.paper_parity import (
    file_sha256,
    load_digitized_points,
    load_strict_json,
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


def test_checked_author_output_score_is_narrow_and_provenance_bound() -> None:
    score = load_strict_json(DEFAULT_SCORE, "checked author-output score")
    author_manifest = load_author_source_manifest(AUTHOR_MANIFEST)
    oracle = load_strict_json(ORACLE_PATH, "Figure 6 paper oracle")
    assert score["schema"] == SCORE_SCHEMA
    assert score["accepted"] is True
    assert score["status"] == "accepted_author_output_identity"
    assert score["author_source"]["manifest_sha256"] == file_sha256(AUTHOR_MANIFEST)
    assert score["author_source"]["archive_sha256"] == author_manifest["archive"]["sha256"]
    output_members = [
        member
        for member in author_manifest["members"]
        if member["role"] == "bundled_output_oracle"
    ]
    assert len(output_members) == 1
    assert score["author_source"]["bundled_output_sha256"] == output_members[0]["sha256"]
    assert score["paper_oracle"]["manifest_sha256"] == file_sha256(ORACLE_PATH)
    assert score["paper_oracle"]["points_sha256"] == oracle["data"]["sha256"]
    assert score["extraction"]["script_sha256"] == source_sha256(
        Path(__file__).resolve().parents[2] / score["extraction"]["script_path"]
    )
    # This is the historical runtime that extracted the authenticated PNG,
    # not a claim that an arbitrary reader environment replayed unavailable
    # external bytes. The ladder binds the complete checked score; when the
    # archive is supplied, the exact-replay test below compares all scientific
    # content under the current runtime.
    runtime = score["producer"]["runtime"]
    assert set(runtime) == {"packages", "python"}
    assert set(runtime["packages"]) == {"Pillow", "numpy"}
    assert set(runtime["python"]) == {
        "cache_tag",
        "implementation",
        "version",
    }
    assert all(
        isinstance(value, str) and value
        for section in runtime.values()
        for value in section.values()
    )
    for path, identity in score["producer"]["sources"].items():
        assert identity["hash_kind"] == (
            "canonical_sha256_import_time_disk_snapshot"
        )
        assert "not an independent interpreter-bytecode attestation" in identity[
            "execution_binding"
        ]
        assert identity["sha256"] == source_sha256(
            Path(__file__).resolve().parents[2] / path
        )
    assert len(score["curve_scores"]) == 6
    assert {
        (curve["curve_id"], curve["curve_kind"])
        for curve in score["curve_scores"]
    } == {
        (curve_id, curve_kind)
        for curve_id in ("T_bath_0.10K", "T_bath_0.15K", "T_bath_0.20K")
        for curve_kind in ("paper_analytic", "paper_numerical")
    }
    for curve in score["curve_scores"]:
        points = curve["points"]
        absolute = [point["absolute_error"] for point in points]
        relative = [point["relative_error"] for point in points]
        normalized = [point["uncertainty_normalized_error"] for point in points]
        assert curve["point_count"] == len(points) == 7
        assert curve["max_absolute_error"] == max(absolute)
        assert curve["max_relative_error"] == max(relative)
        assert curve["max_uncertainty_normalized_error"] == max(normalized)
        assert curve["accepted"] is (
            max(normalized) <= curve["uncertainty_normalized_limit"]
        )
        assert curve["mean_absolute_error"] == pytest.approx(
            sum(absolute) / len(absolute),
            rel=2e-11,
        )
        assert curve["rms_error"] == pytest.approx(
            math.sqrt(sum(value * value for value in absolute) / len(absolute)),
            rel=2e-11,
        )
    assert score["accepted"] is all(
        curve["accepted"] for curve in score["curve_scores"]
    )
    assert "that the current runtime regenerated the bundled PNG" in score["does_not_claim"]


def test_author_output_score_rows_bind_complete_paper_points() -> None:
    configured = os.environ.get("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE")
    if not configured:
        pytest.skip("Set QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE for exact replay.")
    score = build_score(Path(configured))
    _oracle, digitized = load_digitized_points(ORACLE_PATH)
    expected = {
        (point.curve_id, point.curve_kind, point.x_value): _expected_anchor(point)
        for point in digitized
    }
    for curve in score["curve_scores"]:
        for point in curve["points"]:
            key = (curve["curve_id"], curve["curve_kind"], point["x_value"])
            assert point["paper_point"] == expected[key]
            assert point["paper_y"] == point["paper_point"]["values"]["y_value"]


def test_exact_author_output_score_replays_when_archive_is_supplied() -> None:
    configured = os.environ.get("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE")
    if not configured:
        pytest.skip("Set QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE for exact replay.")
    checked = load_strict_json(DEFAULT_SCORE, "checked author-output score")
    replayed = build_score(Path(configured))
    assert _without_runtime(replayed) == _without_runtime(checked)
    assert replayed["producer"]["runtime"]["packages"] == {
        "Pillow": PIL.__version__,
        "numpy": np.__version__,
    }


def _mock_author_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    accepted: bool,
) -> None:
    content = b"authenticated synthetic PNG snapshot"
    member = AuthenticatedMember(
        path="root/output.png",
        role="bundled_output_oracle",
        size_bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
        content=content,
    )
    authenticated = AuthenticatedAuthorSource(
        source_id="synthetic-source",
        archive_path=Path("synthetic.zip"),
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        members=(member,),
    )
    monkeypatch.setattr(
        parity,
        "authenticate_author_source",
        lambda *_args, **_kwargs: authenticated,
    )
    curve = ExtractedCurve(
        x=np.array([0.2, 0.5]),
        y=np.array([0.1, 0.2]),
        y_uncertainty=np.array([0.01, 0.01]),
    )
    monkeypatch.setattr(
        parity,
        "extract_author_output",
        lambda _content: {
            (curve_id, curve_kind): curve
            for curve_id in ("T_bath_0.10K", "T_bath_0.15K", "T_bath_0.20K")
            for curve_kind in ("paper_analytic", "paper_numerical")
        },
    )

    def fake_score(
        paper_points: list[Any],
        _author_curve: ExtractedCurve,
    ) -> dict[str, Any]:
        point = paper_points[0]
        normalized = 0.5 if accepted else 2.0
        return {
            "accepted": accepted,
            "curve_id": point.curve_id,
            "curve_kind": point.curve_kind,
            "max_absolute_error": 0.01,
            "max_relative_error": 0.1,
            "max_uncertainty_normalized_error": normalized,
            "mean_absolute_error": 0.01,
            "point_count": 1,
            "points": [],
            "rms_error": 0.01,
            "uncertainty_normalized_limit": 1.0,
        }

    monkeypatch.setattr(parity, "_score_curve_pair", fake_score)


def test_author_output_mismatch_cannot_emit_an_acceptance_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_author_build(monkeypatch, accepted=False)
    score = build_score(Path("ignored.zip"))
    assert score["accepted"] is False
    assert score["status"] == "diagnostic_mismatch"
    assert score["evidence_class"] == (
        "authenticated_author_output_mismatch_diagnostic"
    )
    assert "does not agree" in score["claim"]
    assert "not acceptance evidence" in score["claim"]


def test_author_output_source_change_during_scoring_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_author_build(monkeypatch, accepted=True)
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
        build_score(Path("ignored.zip"))


def test_author_output_stale_imported_source_fails_before_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = parity._source_snapshot()
    first = next(iter(changed))
    changed[first] = "0" * 64
    monkeypatch.setattr(parity, "_source_snapshot", lambda: changed)
    with pytest.raises(RuntimeError, match="no longer matches"):
        build_score(Path("ignored.zip"))


def test_paper_points_are_parsed_only_from_the_retained_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = parity._paper_evidence_snapshot()
    original_read_bytes = Path.read_bytes
    forbidden = set(snapshot.files)

    def guarded_read_bytes(path: Path) -> bytes:
        if path in forbidden:
            raise AssertionError(f"mutable original was re-read: {path}")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    points = parity._load_digitized_points_from_snapshot(snapshot)
    assert len(points) == 42
