"""Adversarial evidence tests for the checked Figure 6 C7 score and receipt."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from validation.fischer_2023 import fig6_author_c7_score as c7_score

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNS = REPOSITORY_ROOT / "tmp" / "author-runs"
C0_BUNDLE = _RUNS / "fig6-T020-sweep049-C0-author-equivalent-v1"
C2_BUNDLE = _RUNS / "fig6-T020-sweep049-C2-parameters-v1"
C3_BUNDLE = _RUNS / "fig6-T020-sweep049-C3-grid-regen-v1"
C4_BUNDLE = _RUNS / "fig6-T020-sweep049-C4-photon-regen-v1"
C5_BUNDLE = _RUNS / "fig6-T020-sweep049-C5-qp-phonon-regen-v1"
C6_BUNDLE = _RUNS / "fig6-T020-sweep049-C6-phonon-balance-v1"
C7_BUNDLE = _RUNS / "fig6-T020-sweep049-C7-solver-v1"

_BUNDLE_KWARGS = {
    "c6_bundle_dir": C6_BUNDLE,
    "c5_bundle_dir": C5_BUNDLE,
    "c4_bundle_dir": C4_BUNDLE,
    "c3_bundle_dir": C3_BUNDLE,
    "c2_bundle_dir": C2_BUNDLE,
    "c0_bundle_dir": C0_BUNDLE,
}


def _require_checked() -> None:
    required = (
        *(path / "manifest.json" for path in _BUNDLE_KWARGS.values()),
        C7_BUNDLE / "manifest.json",
        c7_score.DEFAULT_SCORE,
        c7_score.DEFAULT_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Checked C7 evidence or its external raw parents are unavailable.")


@pytest.fixture(scope="module")
def checked_score() -> dict[str, Any]:
    _require_checked()
    return c7_score.load_c7_score()


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def test_checked_score_and_receipt_load_strictly(
    checked_score: dict[str, Any],
) -> None:
    assert checked_score["schema"] == c7_score.SCHEMA
    assert checked_score["stage"]["stage_id"] == "C7"
    assert checked_score["stage"]["changed_component"] == "nonlinear_solver"
    assert checked_score["acceptance"]["all_passed"] is True
    receipt = c7_score.load_c7_receipt()
    assert receipt["raw_bundle"] == checked_score["raw_bundle"]
    assert (
        receipt["checked_score"]["file_sha256"]
        == hashlib.sha256(c7_score.DEFAULT_SCORE.read_bytes()).hexdigest()
    )


def test_external_raw_rebuilds_complete_checked_score() -> None:
    _require_checked()
    rebuilt = c7_score.build_c7_score(C7_BUNDLE, **_BUNDLE_KWARGS)
    committed = c7_score.DEFAULT_SCORE.read_bytes()
    assert c7_score.canonical_score_bytes(rebuilt) == committed


def test_receipt_rebuild_binds_complete_score_raw_and_parent() -> None:
    _require_checked()
    rebuilt = c7_score.build_c7_receipt(
        c7_bundle_dir=C7_BUNDLE,
        **_BUNDLE_KWARGS,
    )
    committed = c7_score.DEFAULT_RECEIPT.read_bytes()
    assert _canonical_json(rebuilt) == committed


def test_raw_source_closure_is_complete_and_acyclic(
    checked_score: dict[str, Any],
) -> None:
    metadata, _arrays, _sha = c7_score.load_c7_raw_bundle(C7_BUNDLE)
    raw_sources = set(metadata["sources"])
    assert "validation/fischer_2023/fig6_author_c7_bundle.py" in raw_sources
    assert "validation/fischer_2023/fig6_author_c7_score.py" not in raw_sources
    score_sources = set(checked_score["sources"])
    assert score_sources == raw_sources | {
        "validation/fischer_2023/fig6_author_c7_score.py"
    }


def test_scope_excludes_curve_parity_and_iteration_equivalence(
    checked_score: dict[str, Any],
) -> None:
    statement = checked_score["limitations"]["statement"]
    for phrase in (
        "300-point curve",
        "paper-parity",
        "author-iteration-equivalence",
    ):
        assert phrase in statement
    ungated = checked_score["acceptance"]["ungated_recorded_differences"]
    assert "root_vs_frozen_state" in ungated


def test_score_metric_forgery_breaks_the_pins(tmp_path: Path) -> None:
    _require_checked()
    score = json.loads(c7_score.DEFAULT_SCORE.read_text(encoding="utf-8"))
    record = score["observable"]["producer"]["root_ordinate_authors"]
    forged = 0.5 * record["value"]
    record["value"] = forged
    record["hex"] = float(forged).hex()
    mutated = tmp_path / "forged-score.json"
    mutated.write_bytes(_canonical_json(score))
    with pytest.raises(c7_score.C7ScoreError):
        c7_score.load_c7_score(mutated)


def test_receipt_mutations_cannot_rebind_raw_or_parent(tmp_path: Path) -> None:
    _require_checked()
    receipt = json.loads(c7_score.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    receipt["raw_bundle"]["manifest_sha256"] = "0" * 64
    mutated = tmp_path / "mutated-receipt.json"
    mutated.write_bytes(_canonical_json(receipt))
    with pytest.raises(c7_score.C7ScoreError):
        c7_score.load_c7_score(receipt_path=mutated)


def test_raw_loader_rejects_missing_or_extra_files(tmp_path: Path) -> None:
    _require_checked()
    clone = tmp_path / "c7-clone"
    shutil.copytree(C7_BUNDLE, clone)
    (clone / "qpsim_root_f.npy").unlink()
    with pytest.raises(c7_score.C7ScoreError):
        c7_score.load_c7_raw_bundle(clone)
    clone2 = tmp_path / "c7-clone2"
    shutil.copytree(C7_BUNDLE, clone2)
    (clone2 / "extra.npy").write_bytes(b"\x93NUMPY\x03\x00")
    with pytest.raises(c7_score.C7ScoreError):
        c7_score.load_c7_raw_bundle(clone2)


def test_c7_refuses_wrong_raw_parent_chain() -> None:
    _require_checked()
    with pytest.raises(ValueError):
        c7_score.build_c7_score(
            C7_BUNDLE,
            c6_bundle_dir=C5_BUNDLE,
            c5_bundle_dir=C6_BUNDLE,
            c4_bundle_dir=C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
            c0_bundle_dir=C0_BUNDLE,
        )


def test_canonical_pins_are_bound(checked_score: dict[str, Any]) -> None:
    assert (
        checked_score["raw_bundle"]["manifest_sha256"]
        == c7_score._EXPECTED_RAW_MANIFEST_SHA256
    )
    assert (
        c7_score._evidence_digest(checked_score)
        == c7_score._EXPECTED_EVIDENCE_DIGEST
    )
    locality = checked_score["component_locality"]
    for name, expected in c7_score._EXPECTED_ROOT_NPY_SHA256.items():
        assert locality[name]["npy_sha256"] == expected
    assert (
        checked_score["solver_contract"]["accepted_max_iter"]
        == c7_score._EXPECTED_CANONICAL_INTS["accepted_max_iter"]
    )
