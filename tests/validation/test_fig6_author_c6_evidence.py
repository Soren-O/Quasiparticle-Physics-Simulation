"""Adversarial evidence tests for the checked Figure 6 C6 score and receipt."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from validation.fischer_2023 import fig6_author_c6_score as c6_score
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import DEFAULT_SCORE as C2_SCORE
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import DEFAULT_SCORE as C3_SCORE
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import DEFAULT_SCORE as C4_SCORE
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import DEFAULT_SCORE as C5_SCORE

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNS = REPOSITORY_ROOT / "tmp" / "author-runs"
C2_BUNDLE = _RUNS / "fig6-T020-sweep049-C2-parameters-v1"
C3_BUNDLE = _RUNS / "fig6-T020-sweep049-C3-grid-regen-v1"
C4_BUNDLE = _RUNS / "fig6-T020-sweep049-C4-photon-regen-v1"
C5_BUNDLE = _RUNS / "fig6-T020-sweep049-C5-qp-phonon-regen-v1"
C6_BUNDLE = _RUNS / "fig6-T020-sweep049-C6-phonon-balance-v1"


def _require_checked() -> None:
    required = (
        C2_BUNDLE / "manifest.json",
        C3_BUNDLE / "manifest.json",
        C4_BUNDLE / "manifest.json",
        C5_BUNDLE / "manifest.json",
        C6_BUNDLE / "manifest.json",
        C2_SCORE,
        C2_RECEIPT,
        C3_SCORE,
        C3_RECEIPT,
        C4_SCORE,
        C4_RECEIPT,
        C5_SCORE,
        C5_RECEIPT,
        c6_score.DEFAULT_SCORE,
        c6_score.DEFAULT_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Checked C6 evidence or its external raw parents are unavailable.")


@pytest.fixture(scope="module")
def checked_score() -> dict[str, Any]:
    _require_checked()
    return c6_score.load_c6_score()


@pytest.fixture(scope="module")
def rebuilt_score() -> dict[str, Any]:
    _require_checked()
    return c6_score.build_c6_score(
        C6_BUNDLE,
        c5_bundle_dir=C5_BUNDLE,
        c4_bundle_dir=C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _clone_raw_bundle(tmp_path: Path) -> Path:
    target = tmp_path / "c6-clone"
    shutil.copytree(C6_BUNDLE, target)
    return target


def _load_manifest(target: Path) -> dict[str, Any]:
    return json.loads((target / "manifest.json").read_text(encoding="utf-8"))


def _write_manifest(target: Path, manifest: dict[str, Any]) -> None:
    (target / "manifest.json").write_bytes(_canonical_json(manifest))


def _replace_array(
    target: Path,
    name: str,
    value: np.ndarray,
    *,
    rebind_manifest: bool,
) -> None:
    content = _npy_bytes(value)
    (target / f"{name}.npy").write_bytes(content)
    if rebind_manifest:
        manifest = _load_manifest(target)
        manifest["files"][f"{name}.npy"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        descriptor = {
            "dtype": np.asarray(value).dtype.str,
            "npy_sha256": hashlib.sha256(content).hexdigest(),
            "shape": list(np.asarray(value).shape),
        }
        manifest["metadata"]["array_descriptors"][name] = descriptor
        _write_manifest(target, manifest)


def test_checked_score_and_receipt_load_strictly(
    checked_score: dict[str, Any],
) -> None:
    assert checked_score["schema"] == c6_score.SCHEMA
    assert checked_score["stage"]["stage_id"] == "C6"
    assert checked_score["stage"]["changed_component"] == "phonon_balance"
    assert checked_score["acceptance"]["all_passed"] is True
    receipt = c6_score.load_c6_receipt()
    assert receipt["raw_bundle"] == checked_score["raw_bundle"]
    assert (
        receipt["checked_score"]["file_sha256"]
        == hashlib.sha256(c6_score.DEFAULT_SCORE.read_bytes()).hexdigest()
    )


def test_external_raw_rebuilds_complete_checked_score(
    rebuilt_score: dict[str, Any],
) -> None:
    committed = c6_score.DEFAULT_SCORE.read_bytes()
    assert c6_score.canonical_score_bytes(rebuilt_score) == committed


def test_receipt_rebuild_binds_complete_score_raw_and_parent() -> None:
    _require_checked()
    rebuilt = c6_score.build_c6_receipt(
        c6_bundle_dir=C6_BUNDLE,
        c5_bundle_dir=C5_BUNDLE,
        c4_bundle_dir=C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    committed = c6_score.DEFAULT_RECEIPT.read_bytes()
    assert _canonical_json(rebuilt) == committed


def test_raw_source_closure_is_complete_and_acyclic(
    checked_score: dict[str, Any],
) -> None:
    metadata, _arrays, _sha = c6_score.load_c6_raw_bundle(C6_BUNDLE)
    raw_sources = set(metadata["sources"])
    assert "validation/fischer_2023/fig6_author_c6_bundle.py" in raw_sources
    assert "validation/fischer_2023/fig6_author_c6_score.py" not in raw_sources
    score_sources = set(checked_score["sources"])
    assert "validation/fischer_2023/fig6_author_c6_score.py" in score_sources
    assert score_sources == raw_sources | {
        "validation/fischer_2023/fig6_author_c6_score.py"
    }


def test_scope_explicitly_excludes_solver_curve_and_observable_claims(
    checked_score: dict[str, Any],
) -> None:
    statement = checked_score["limitations"]["statement"]
    for phrase in (
        "No C6 nonlinear root",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "observable change",
        "paper-parity",
    ):
        assert phrase in statement
    ungated = checked_score["acceptance"]["ungated_recorded_differences"]
    assert "pair_public_vs_parent" in ungated
    assert "support_extension" in ungated


def test_score_mutation_is_rejected(tmp_path: Path) -> None:
    _require_checked()
    score = json.loads(c6_score.DEFAULT_SCORE.read_text(encoding="utf-8"))
    score["acceptance"]["checks"]["frozen_qp_residual_bit_exact"] = False
    mutated = tmp_path / "mutated-score.json"
    mutated.write_bytes(_canonical_json(score))
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_score(mutated)


def test_score_metric_forgery_breaks_the_evidence_digest(
    tmp_path: Path,
) -> None:
    _require_checked()
    score = json.loads(c6_score.DEFAULT_SCORE.read_text(encoding="utf-8"))
    record = score["channel_comparison"]["pair"]["parent_operator"]["net"][
        "symmetric_relative_l1"
    ]
    forged = 0.5 * record["value"] if record["value"] != 0.0 else 1.0e-30
    record["value"] = forged
    record["hex"] = float(forged).hex()
    mutated = tmp_path / "forged-score.json"
    mutated.write_bytes(_canonical_json(score))
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_score(mutated)


def test_receipt_mutations_cannot_rebind_raw_or_parent(tmp_path: Path) -> None:
    _require_checked()
    receipt = json.loads(c6_score.DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    receipt["raw_bundle"]["manifest_sha256"] = "0" * 64
    mutated = tmp_path / "mutated-receipt.json"
    mutated.write_bytes(_canonical_json(receipt))
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_score(receipt_path=mutated)


def test_raw_loader_rejects_an_extra_file(tmp_path: Path) -> None:
    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    (clone / "extra.npy").write_bytes(b"\x93NUMPY\x03\x00")
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)


def test_raw_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    (clone / "qpsim_pair_a_ns_inv.npy").unlink()
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)


def test_raw_loader_rejects_unrebound_array_change(tmp_path: Path) -> None:
    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    _metadata, arrays, _sha = c6_score.load_c6_raw_bundle(C6_BUNDLE)
    tampered = arrays["qpsim_pair_a_ns_inv"].copy()
    tampered[400] *= 1.0 + 1.0e-12
    _replace_array(clone, "qpsim_pair_a_ns_inv", tampered, rebind_manifest=False)
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)


def test_raw_loader_rejects_non_canonical_signed_zero(tmp_path: Path) -> None:
    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    _metadata, arrays, _sha = c6_score.load_c6_raw_bundle(C6_BUNDLE)
    tampered = arrays["qpsim_scattering_a_ns_inv"].copy()
    tampered[0] = -0.0
    _replace_array(clone, "qpsim_scattering_a_ns_inv", tampered, rebind_manifest=True)
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)


def test_raw_loader_rejects_npy_v2_and_trailing_bytes(tmp_path: Path) -> None:
    _require_checked()
    _metadata, arrays, _sha = c6_score.load_c6_raw_bundle(C6_BUNDLE)
    value = arrays["qpsim_thermal_n_ph"]

    clone = _clone_raw_bundle(tmp_path)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(2, 0), allow_pickle=False)
    content = stream.getvalue()
    (clone / "qpsim_thermal_n_ph.npy").write_bytes(content)
    manifest = _load_manifest(clone)
    manifest["files"]["qpsim_thermal_n_ph.npy"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    _write_manifest(clone, manifest)
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)

    trailing = tmp_path / "c6-trailing"
    shutil.copytree(C6_BUNDLE, trailing)
    original = (trailing / "qpsim_thermal_n_ph.npy").read_bytes() + b"\x00"
    (trailing / "qpsim_thermal_n_ph.npy").write_bytes(original)
    manifest = _load_manifest(trailing)
    manifest["files"]["qpsim_thermal_n_ph.npy"] = {
        "sha256": hashlib.sha256(original).hexdigest(),
        "size_bytes": len(original),
    }
    _write_manifest(trailing, manifest)
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(trailing)


def test_raw_loader_rejects_duplicate_or_noncanonical_json(
    tmp_path: Path,
) -> None:
    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    raw = (clone / "manifest.json").read_bytes()
    (clone / "manifest.json").write_bytes(raw + b"\n")
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.load_c6_raw_bundle(clone)


def test_fully_rebound_derived_array_forgery_is_rejected(
    tmp_path: Path,
) -> None:
    """A self-consistent manifest cannot hide a broken derived identity."""

    _require_checked()
    clone = _clone_raw_bundle(tmp_path)
    _metadata, arrays, _sha = c6_score.load_c6_raw_bundle(C6_BUNDLE)
    tampered = arrays["c6spe_phonon_residual_s_inv"].copy()
    tampered[100] += 1.0e-9
    _replace_array(
        clone,
        "c6spe_phonon_residual_s_inv",
        tampered,
        rebind_manifest=True,
    )
    with pytest.raises(c6_score.C6ScoreError):
        c6_score.build_c6_score(
            clone,
            c5_bundle_dir=C5_BUNDLE,
            c4_bundle_dir=C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_c6_refuses_wrong_raw_parent_chain() -> None:
    _require_checked()
    with pytest.raises(ValueError):
        c6_score.build_c6_score(
            C6_BUNDLE,
            c5_bundle_dir=C4_BUNDLE,
            c4_bundle_dir=C5_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_score_builder_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = dict(c6_score._SOURCE_BYTES_AT_IMPORT)
    key = next(iter(drifted))
    drifted[key] = drifted[key] + b"# drift\n"
    monkeypatch.setattr(c6_score, "_SOURCE_BYTES_AT_IMPORT", drifted)
    with pytest.raises(c6_score.C6ScoreError):
        c6_score._assert_source_snapshots()


def test_canonical_pins_are_bound(checked_score: dict[str, Any]) -> None:
    assert (
        checked_score["raw_bundle"]["manifest_sha256"]
        == c6_score._EXPECTED_RAW_MANIFEST_SHA256
    )
    assert (
        c6_score._evidence_digest(checked_score)
        == c6_score._EXPECTED_EVIDENCE_DIGEST
    )
    locality = checked_score["component_locality"]
    for name, expected in c6_score._EXPECTED_RESIDUAL_NPY_SHA256.items():
        assert locality[name]["npy_sha256"] == expected
