"""Adversarial evidence tests for the formal Figure 6 C3 score and receipt."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from validation.fischer_2023 import fig6_author_c3_score as c3_score
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT,
    DEFAULT_SCORE,
    RAW_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA,
    C3ScoreError,
    build_c3_receipt,
    build_c3_score,
    canonical_score_bytes,
    load_c3_raw_bundle,
    load_c3_receipt,
    load_c3_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _first_listable_bundle(*candidates: Path) -> Path:
    """Prefer the original bundle directory; fall back to a byte-identical
    regeneration when the original is present but unreadable on this host."""

    for candidate in candidates:
        try:
            if (candidate / "manifest.json").is_file():
                next(iter(candidate.iterdir()), None)
                return candidate
        except OSError:
            continue
    return candidates[0]

C2_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C2-parameters-v1"
C3_BUNDLE = _first_listable_bundle(
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C3-grid-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C3-grid-regen-v1",
)

RawEvidence = tuple[dict[str, Any], dict[str, np.ndarray], str]


def _require_external_c3() -> None:
    if not (C3_BUNDLE / "manifest.json").is_file():
        pytest.skip("Canonical external C3 raw bundle is unavailable.")


@pytest.fixture(scope="module")
def checked_score() -> dict[str, Any]:
    # Deliberately no skip: once checked artifacts are committed, their
    # absence or rejection is a hard repository failure.
    return load_c3_score(DEFAULT_SCORE, receipt_path=DEFAULT_RECEIPT)


@pytest.fixture(scope="module")
def raw_evidence() -> RawEvidence:
    _require_external_c3()
    return load_c3_raw_bundle(C3_BUNDLE)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(
    value: np.ndarray,
    *,
    version: tuple[int, int] = (3, 0),
    allow_pickle: bool = False,
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=version,
        allow_pickle=allow_pickle,
    )
    return stream.getvalue()


def _descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    raw = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
    }


def _clone_raw_bundle(tmp_path: Path) -> Path:
    _require_external_c3()
    target = tmp_path / "c3"

    def link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    shutil.copytree(C3_BUNDLE, target, copy_function=link_or_copy)
    return target


def _replace_bytes(path: Path, content: bytes) -> None:
    # Copies normally use hard links to avoid multiplying the two large
    # coherence matrices. Detach before mutation so canonical evidence is
    # immutable even if the test aborts.
    path.unlink()
    path.write_bytes(content)


def _load_manifest(target: Path) -> dict[str, Any]:
    value = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(target: Path, manifest: dict[str, Any]) -> None:
    _replace_bytes(target / "manifest.json", _canonical_json(manifest))


def _replace_array(
    target: Path,
    name: str,
    value: np.ndarray,
    *,
    version: tuple[int, int] = (3, 0),
    allow_pickle: bool = False,
    rebind_descriptor: bool = False,
) -> None:
    raw = _npy_bytes(
        value,
        version=version,
        allow_pickle=allow_pickle,
    )
    _replace_bytes(target / f"{name}.npy", raw)
    manifest = _load_manifest(target)
    manifest["files"][f"{name}.npy"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    if rebind_descriptor:
        manifest["metadata"]["array_descriptors"][name] = _descriptor(value)
    _write_manifest(target, manifest)


def test_checked_score_and_receipt_load_strictly(
    checked_score: dict[str, Any],
) -> None:
    receipt = load_c3_receipt(DEFAULT_RECEIPT)
    assert checked_score["schema"] == SCHEMA
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert checked_score["stage"] == {
        "changed_component": "grid_sampling",
        "comparison_stage_id": "C2",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C2",
        "stage_id": "C3",
        "status": "completed",
    }
    assert checked_score["raw_bundle"]["schema"] == RAW_SCHEMA


def test_receipt_binds_complete_checked_score_and_raw_manifest(
    checked_score: dict[str, Any],
) -> None:
    _require_external_c3()
    receipt = load_c3_receipt(DEFAULT_RECEIPT)
    assert receipt == build_c3_receipt(
        DEFAULT_SCORE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert receipt["checked_score"] == {
        "file_sha256": hashlib.sha256(DEFAULT_SCORE.read_bytes()).hexdigest(),
        "schema": SCHEMA,
    }
    assert receipt["raw_bundle"] == checked_score["raw_bundle"]


def test_receipt_rejects_a_structurally_valid_checked_score_byte_change(
    tmp_path: Path,
) -> None:
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    comparison = score["comparison"]
    original = float(comparison["net_subtraction_worst_fraction_of_limit"])
    comparison["net_subtraction_worst_fraction_of_limit"] = float(np.nextafter(original, 1.0))
    assert comparison["net_subtraction_worst_fraction_of_limit"] <= 1.0
    path = tmp_path / "score.json"
    path.write_bytes(_canonical_json(score))

    with pytest.raises(
        C3ScoreError,
        match="Checked C3 score bytes do not match the selected receipt",
    ):
        load_c3_score(path, receipt_path=DEFAULT_RECEIPT)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        pytest.param(
            lambda score: score["stages"][2].__setitem__(
                "pair_frequency_offset_bins",
                0,
            ),
            "Checked C3 stage 2 identity is invalid",
            id="pair-offset-policy",
        ),
        pytest.param(
            lambda score: score["comparison"].__setitem__(
                "net_subtraction_worst_fraction_of_limit",
                float(
                    np.nextafter(
                        score["comparison"]["net_subtraction_worst_fraction_of_limit"],
                        1.0,
                    )
                ),
            ),
            "receipt refuses to anchor score bytes that do not reproduce",
            id="structurally-valid-numeric-forgery",
        ),
    ),
)
def test_receipt_rebuild_refuses_a_canonical_forged_score(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    _require_external_c3()
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    mutation(score)  # type: ignore[operator]
    forged = tmp_path / "forged-score.json"
    forged.write_bytes(_canonical_json(score))

    with pytest.raises(C3ScoreError, match=match):
        build_c3_receipt(
            forged,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_checked_score_rejects_an_expanded_or_reversed_no_root_claim(
    tmp_path: Path,
) -> None:
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    score["limitations"] = {
        "root_verified": True,
        "scope": "one authenticated C2 frozen point only",
        "statement": (
            "No C3 nonlinear root limitation remains: a C3 root, "
            "300-point curve, and paper parity are fully verified."
        ),
    }
    forged = tmp_path / "forged-limitations.json"
    forged.write_bytes(_canonical_json(score))
    with pytest.raises(C3ScoreError, match="C3 limitations fields are invalid"):
        load_c3_score(forged, receipt_path=DEFAULT_RECEIPT)


def test_external_raw_rebuilds_the_checked_score_canonically(
    raw_evidence: RawEvidence,
) -> None:
    _metadata, arrays, manifest_sha = raw_evidence
    assert len(arrays) == 105
    rebuilt = build_c3_score(C3_BUNDLE, c2_bundle_dir=C2_BUNDLE)
    assert canonical_score_bytes(rebuilt) == DEFAULT_SCORE.read_bytes()
    assert rebuilt["raw_bundle"] == {
        "manifest_sha256": manifest_sha,
        "schema": RAW_SCHEMA,
    }


def test_acceptance_is_explicit_and_scope_excludes_root_and_ordinate(
    checked_score: dict[str, Any],
) -> None:
    acceptance = checked_score["acceptance"]
    assert acceptance["accepted"] is True
    assert len(acceptance["checks"]) == 20
    assert all(value is True for value in acceptance["checks"].values())
    assert acceptance["limits"]["raw_array_max_absolute_error"] == 0.0

    limitations = checked_score["limitations"]
    assert limitations["scope"] == "one authenticated C2 frozen point only"
    for excluded in (
        "No C3 nonlinear root",
        "Newton history",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "paper-parity claim",
    ):
        assert excluded in limitations["statement"]
    assert checked_score["observable_control"]["independently_recomputed"]["claim"].endswith(
        "Neither is a C3 root or plotted ordinate."
    )


def test_native_gap_is_exact_180_while_author_carrier_is_one_ulp_lower(
    raw_evidence: RawEvidence,
) -> None:
    metadata, arrays, _manifest_sha = raw_evidence
    native = metadata["native_qpsim_grid_parameters"]
    assert native["gap_ueV"] == native["delta0_ueV"] == 180.0
    assert native["gap_ueV_hex"] == native["delta0_ueV_hex"] == (180.0).hex()

    author_gap_ueV = metadata["parameters"]["values"]["gap_eV"] * 1.0e6
    assert author_gap_ueV == np.nextafter(180.0, 0.0)
    assert author_gap_ueV.hex() == "0x1.67fffffffffffp+7"
    assert arrays["native_cell_edges_ueV"][20] == 180.0
    assert arrays["parent_E_left_eV"][0] * 1.0e6 == author_gap_ueV


def test_face_roundoff_is_separate_from_half_bin_carrier_and_both_observables_shift(
    checked_score: dict[str, Any],
    raw_evidence: RawEvidence,
) -> None:
    _metadata, arrays, _manifest_sha = raw_evidence
    projection = checked_score["projection"]
    face = projection["mapped_left_edge_delta_ueV"]
    carrier = projection["sample_carrier_delta_ueV"]

    assert 0 < face["nonzero_count"] < 1620
    assert -1.0e-12 < face["minimum"] < 0.0
    assert 0.0 < face["maximum"] < 1.0e-12
    assert carrier["nonzero_count"] == 1620
    assert 0.49 < carrier["minimum"] <= carrier["maximum"] < 0.51
    assert np.array_equal(
        arrays["sample_carrier_delta_ueV"],
        arrays["mapped_left_edge_delta_ueV"] + 0.5,
    )

    observable = checked_score["observable_control"]
    assert (
        observable["author_reembedding_maximum_integral_absolute_difference"]
        <= checked_score["acceptance"]["limits"]["observable_integral_max_absolute_error"]
    )
    shifts = observable["native_center_carrier_relative_shifts"]
    assert 0.01 < abs(shifts["driven_integral"]) < 0.2
    assert 0.01 < abs(shifts["thermal_integral"]) < 0.2


def test_raw_loader_rejects_an_extra_file(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    (target / "undeclared.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(C3ScoreError, match="directory closure"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_a_missing_json_key(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    manifest = _load_manifest(target)
    del manifest["schema"]
    _write_manifest(target, manifest)
    with pytest.raises(C3ScoreError, match="manifest fields are invalid"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    manifest = (target / "manifest.json").read_text(encoding="utf-8")
    duplicate = manifest.replace(
        "{\n",
        '{\n  "schema": "qpsim.fischer2023.fig6-author-c3-grid-bundle.v1",\n',
        1,
    )
    _replace_bytes(target / "manifest.json", duplicate.encode("utf-8"))
    with pytest.raises(C3ScoreError, match="Duplicate JSON key 'schema'"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_a_symlinked_bundle_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c3()
    original = Path.is_symlink

    def selected_root_is_a_symlink(path: Path) -> bool:
        if path == C3_BUNDLE:
            return True
        return original(path)

    monkeypatch.setattr(Path, "is_symlink", selected_root_is_a_symlink)
    with pytest.raises(C3ScoreError, match="missing, unsafe, or a symlink"):
        load_c3_raw_bundle(C3_BUNDLE)


def test_raw_loader_rejects_noncanonical_npy_version_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "parent_to_native_index"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    _replace_array(target, name, array, version=(2, 0))
    with pytest.raises(C3ScoreError, match="not canonical NPY v3"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_pickle_payload_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "parent_to_native_index"
    payload = np.array([{"not": "numeric evidence"}], dtype=object)
    _replace_array(target, name, payload, allow_pickle=True)
    with pytest.raises(C3ScoreError, match="Cannot load C3 raw array"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_wrong_dtype_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "parent_f"
    array = np.load(target / f"{name}.npy", allow_pickle=False).astype(np.float32)
    _replace_array(target, name, array)
    with pytest.raises(C3ScoreError, match="array descriptors are incomplete, forged, or stale"):
        load_c3_raw_bundle(target)


def test_raw_loader_rejects_signed_zero_padding_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "projected_f"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    array = np.asarray(array).copy()
    assert array[0] == 0.0 and not np.signbit(array[0])
    array[0] = -0.0
    _replace_array(target, name, array)
    with pytest.raises(C3ScoreError, match="array descriptors are incomplete, forged, or stale"):
        load_c3_raw_bundle(target)


def test_independent_score_rejects_scientific_array_mutation_after_full_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "c3c_native_cell_density__qp_pair__gain_s_inv"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    array = np.asarray(array).copy()
    index = int(np.argmax(np.abs(array)))
    array[index] = np.nextafter(array[index], np.inf)
    _replace_array(
        target,
        name,
        array,
        rebind_descriptor=True,
    )

    # The canonical transport is internally consistent after the adversary
    # rebinds both file metadata and the scientific descriptor. The
    # independent source-order recomputation must still reject it.
    load_c3_raw_bundle(target)
    with pytest.raises(
        C3ScoreError,
        match="failed independent bit-exact recomputation",
    ):
        build_c3_score(target, c2_bundle_dir=C2_BUNDLE)


def test_score_builder_binds_the_complete_c2b5_parameter_carrier(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    manifest = _load_manifest(target)
    parameters = manifest["metadata"]["parameters"]
    parameters["values"]["max_newton_steps"] = 999
    parameters["values"]["relative_step_threshold"] = 0.123
    parameters["values"]["thermal_gap_eV"] = 0.0001
    parameters["hex"]["relative_step_threshold"] = (0.123).hex()
    parameters["hex"]["thermal_gap_eV"] = (0.0001).hex()
    _write_manifest(target, manifest)

    load_c3_raw_bundle(target)
    with pytest.raises(
        C3ScoreError,
        match="complete accepted C2b5 author-operator carrier",
    ):
        build_c3_score(target, c2_bundle_dir=C2_BUNDLE)


def test_score_builder_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c3()
    target_relative = "qpsim/physics/spectral.py"
    target = REPOSITORY_ROOT / target_relative
    original = c3_score.canonical_source_bytes

    def drifted(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == target.resolve():
            return content + b"\n# simulated verifier-source drift\n"
        return content

    assert target_relative in c3_score._SOURCE_BYTES_AT_IMPORT
    monkeypatch.setattr(c3_score, "canonical_source_bytes", drifted)
    with pytest.raises(
        C3ScoreError,
        match=r"C3 score source changed during execution: qpsim/physics/spectral\.py",
    ):
        build_c3_score(C3_BUNDLE, c2_bundle_dir=C2_BUNDLE)
