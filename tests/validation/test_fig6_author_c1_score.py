from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from validation.fischer_2023.fig6_author_c1_score import (
    DEFAULT_SCORE,
    SCHEMA,
    C1ScoreError,
    build_c1_score,
    canonical_score_bytes,
    load_c1_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_checked_c1_score_is_exact_and_component_isolated() -> None:
    score = load_c1_score()
    assert score["schema"] == SCHEMA
    assert score["acceptance"]["accepted"] is True
    assert score["stage"] == {
        "changed_component": "observable",
        "comparison_stage_id": "C0",
        "parent_stage_id": "C0",
        "stage_id": "C1",
        "status": "completed",
    }
    differences = score["calculation"]["differences"]
    assert differences["driven_gap_signed_eV"] == 0.0
    assert differences["thermal_gap_signed_eV"] == 0.0
    assert differences["figure6_ordinate_signed"] == 0.0
    assert score["input_identity"]["parameter_hex"]["delta0_eV"] == (180 * 10**-6).hex()
    assert (180 * 10**-6).hex() != (180.0e-6).hex()
    assert "thermal states" in score["limitations"]["statement"]


def test_c1_score_regenerates_byte_exactly_when_c0_raw_is_available() -> None:
    raw = os.environ.get("QPSIM_FISCHER2023_FIG6_C0_BUNDLE")
    if raw is None:
        pytest.skip("Set QPSIM_FISCHER2023_FIG6_C0_BUNDLE for C1 regeneration.")
    regenerated = build_c1_score(Path(raw))
    assert canonical_score_bytes(regenerated) == DEFAULT_SCORE.read_bytes()


def test_checked_c1_score_rejects_extra_claims(tmp_path: Path) -> None:
    score = load_c1_score()
    score["unreviewed_claim"] = True
    path = tmp_path / "c1.json"
    path.write_text(
        json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1ScoreError, match="fields are invalid"):
        load_c1_score(path)


def test_checked_c1_score_binds_the_live_c0_summary(tmp_path: Path) -> None:
    score = load_c1_score()
    score["c0_binding"]["summary_sha256"] = "0" * 64
    path = tmp_path / "c1.json"
    path.write_text(
        json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1ScoreError, match="C0-summary binding is stale"):
        load_c1_score(path)


def test_checked_c1_score_binds_the_c0_raw_manifest(tmp_path: Path) -> None:
    score = load_c1_score()
    score["c0_binding"]["raw_manifest_sha256"] = "0" * 64
    path = tmp_path / "c1.json"
    path.write_text(
        json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1ScoreError, match="raw-manifest binding"):
        load_c1_score(path)


def test_checked_c1_score_requires_canonical_c0_summary_path(tmp_path: Path) -> None:
    score = load_c1_score()
    score["c0_binding"]["summary_path"] = (
        "validation/paper_data/fischer_2023/fig6/../fig6/"
        "c0-author-equivalent-score.json"
    )
    path = tmp_path / "c1.json"
    path.write_text(
        json.dumps(score, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(C1ScoreError, match="canonical C0 summary path"):
        load_c1_score(path)
