"""Regression and adversarial checks for formal Figure 6 C2 evidence."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023 import fig6_author_c2_bundle as c2_bundle
from validation.fischer_2023 import fig6_author_c2_score as c2_score
from validation.fischer_2023.fig6_author_c2_bundle import (
    C2BundleError,
    array_descriptor,
    load_c2_raw_bundle,
    write_c2_bundle,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as C2_SCORE,
)
from validation.fischer_2023.fig6_author_c2_score import (
    C2ScoreError,
    build_c2_score,
    canonical_score_bytes,
    load_c2_score,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
C0_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C0-author-equivalent-v1"
C2_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C2-parameters-v1"


def _require_external_bundles() -> None:
    if not (C0_BUNDLE / "manifest.json").is_file():
        pytest.skip("Canonical external C0 raw bundle is unavailable.")
    if not (C2_BUNDLE / "manifest.json").is_file():
        pytest.skip("Canonical external C2 raw bundle is unavailable.")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def test_checked_c2_score_is_accepted_and_narrow() -> None:
    score = load_c2_score()
    assert score["acceptance"]["accepted"] is True
    assert score["stage"] == {
        "changed_component": "parameters",
        "comparison_stage_id": "C1",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C1",
        "stage_id": "C2",
        "status": "completed",
    }
    assert score["parameter_axis"]["selected_comparison"] == ("fixed authenticated author n_bar")
    assert score["parameter_axis"]["qpsim_fixed_nbar_t_star_over_delta"] == (
        pytest.approx(0.3399503360830364, rel=0.0, abs=0.0)
    )
    assert score["parameter_axis"]["qpsim_relative_shift_at_fixed_nbar"] == (
        pytest.approx(1.2485355715696755e-4, rel=1e-13)
    )
    assert "No C2 root or plotted ordinate is claimed" in (score["limitations"]["statement"])
    assert len(score["steps"]) == 6


def test_checked_score_regenerates_byte_exactly() -> None:
    _require_external_bundles()
    score = build_c2_score(
        C2_BUNDLE,
        c0_bundle_dir=C0_BUNDLE,
    )
    assert canonical_score_bytes(score) == C2_SCORE.read_bytes()


def test_committed_receipt_anchors_complete_score_and_external_raw_manifest() -> None:
    score = json.loads(C2_SCORE.read_text(encoding="utf-8"))
    receipt = json.loads(C2_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["checked_score"] == {
        "file_sha256": hashlib.sha256(C2_SCORE.read_bytes()).hexdigest(),
        "schema": score["schema"],
    }
    assert receipt["raw_bundle"] == score["raw_bundle"]


def test_raw_bundle_loads_with_exact_file_closure() -> None:
    _require_external_bundles()
    metadata, arrays, manifest_sha = load_c2_raw_bundle(C2_BUNDLE)
    assert len(manifest_sha) == 64
    assert metadata["schema"].endswith("frozen-bundle.v1")
    assert set(arrays) == set(metadata["array_descriptors"])
    assert len(metadata["steps"]) == 6
    assert metadata["steps"][0]["step_id"] == "C2a-author-value-plumbing"
    assert metadata["steps"][-1]["step_id"] == ("C2b5-finite-cutoff-critical-temperature")


def test_bundle_regeneration_preserves_all_raw_arrays(tmp_path: Path) -> None:
    _require_external_bundles()
    regenerated = tmp_path / "c2"
    write_c2_bundle(C0_BUNDLE, regenerated)
    expected_metadata, expected_arrays, _ = load_c2_raw_bundle(C2_BUNDLE)
    got_metadata, got_arrays, _ = load_c2_raw_bundle(regenerated)
    assert got_metadata == expected_metadata
    assert set(got_arrays) == set(expected_arrays)
    for name in expected_arrays:
        np.testing.assert_array_equal(got_arrays[name], expected_arrays[name])


def test_raw_bundle_rejects_extra_file(tmp_path: Path) -> None:
    _require_external_bundles()
    target = tmp_path / "c2"
    shutil.copytree(C2_BUNDLE, target)
    (target / "extra.txt").write_text("not part of the evidence", encoding="utf-8")
    with pytest.raises(C2BundleError, match="missing or extra"):
        load_c2_raw_bundle(target)


def test_score_rejects_scientific_array_tamper_even_when_transport_rebound(
    tmp_path: Path,
) -> None:
    _require_external_bundles()
    target = tmp_path / "c2"
    shutil.copytree(C2_BUNDLE, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = "c2b5_finite_cutoff_critical_temperature__qp_pair__gain_s_inv"
    path = target / f"{name}.npy"
    array = np.load(path, allow_pickle=False)
    array = np.asarray(array).copy()
    array[0] = np.nextafter(array[0], np.inf)
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(3, 0), allow_pickle=False)
    raw = stream.getvalue()
    path.write_bytes(raw)
    manifest["files"][path.name] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    manifest["metadata"]["array_descriptors"][name] = array_descriptor(array)
    _write_json(manifest_path, manifest)
    with pytest.raises(C2ScoreError, match="failed independent recomputation"):
        build_c2_score(target, c0_bundle_dir=C0_BUNDLE)


def test_score_rejects_representation_tamper_even_when_transport_rebound(
    tmp_path: Path,
) -> None:
    _require_external_bundles()
    target = tmp_path / "c2"
    shutil.copytree(C2_BUNDLE, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = "c2b5_finite_cutoff_critical_temperature__qp_pair__gain_s_inv"
    path = target / f"{name}.npy"
    original = np.load(path, allow_pickle=False)
    represented = np.asarray(original).astype(original.dtype.newbyteorder("S"))
    assert np.array_equal(original, represented)
    assert array_descriptor(original) != array_descriptor(represented)
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        represented,
        version=(3, 0),
        allow_pickle=False,
    )
    raw = stream.getvalue()
    path.write_bytes(raw)
    manifest["files"][path.name] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    manifest["metadata"]["array_descriptors"][name] = array_descriptor(represented)
    _write_json(manifest_path, manifest)

    with pytest.raises(C2ScoreError, match="array descriptors failed"):
        build_c2_score(target, c0_bundle_dir=C0_BUNDLE)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        pytest.param(
            lambda metadata: metadata["coordinate_contract"].__setitem__(
                "pair_frequency_offset_bins",
                -0.0,
            ),
            "grid/sampling contract",
            id="signed-zero-coordinate",
        ),
        pytest.param(
            lambda metadata: metadata["steps"][0].__setitem__("index", False),
            "scientific step metadata",
            id="bool-for-step-index",
        ),
        pytest.param(
            lambda metadata: metadata["frozen_inputs"].__setitem__(
                "mutation_check_after_all_steps",
                1,
            ),
            "frozen-input identity",
            id="int-for-bool",
        ),
    ],
)
def test_score_rejects_python_equal_but_json_distinct_metadata(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    _require_external_bundles()
    target = tmp_path / "c2"
    shutil.copytree(C2_BUNDLE, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest["metadata"])  # type: ignore[operator]
    _write_json(manifest_path, manifest)

    with pytest.raises(C2ScoreError, match=match):
        build_c2_score(target, c0_bundle_dir=C0_BUNDLE)


def test_c2a_control_rejects_equal_opposite_per_channel_net_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_bundles()
    original_evaluator = c2_score.evaluate_author_system

    def corrupt_channel_nets(*args: object, **kwargs: object) -> object:
        evaluation = original_evaluator(*args, **kwargs)
        offset = 0.125
        return replace(
            evaluation,
            qp_photon=replace(
                evaluation.qp_photon,
                net_s_inv=evaluation.qp_photon.net_s_inv + offset,
            ),
            qp_scattering=replace(
                evaluation.qp_scattering,
                net_s_inv=evaluation.qp_scattering.net_s_inv - offset,
            ),
        )

    # Simulate a common evaluator defect across producer, scorer, and the C0
    # verifier alias. Persisted C0 gain/loss plus the independent subtraction
    # roundoff gate must still expose the per-channel attribution change.
    monkeypatch.setattr(c2_bundle, "evaluate_author_system", corrupt_channel_nets)
    monkeypatch.setattr(c2_score, "evaluate_author_system", corrupt_channel_nets)
    monkeypatch.setattr(
        c2_score.c0_summary_verifier,
        "evaluate_author_system",
        corrupt_channel_nets,
    )
    target = tmp_path / "c2"
    write_c2_bundle(C0_BUNDLE, target)

    with pytest.raises(C2ScoreError, match="c2a_control_bit_exact"):
        build_c2_score(target, c0_bundle_dir=C0_BUNDLE)


@pytest.mark.parametrize(
    "candidate",
    [
        pytest.param(
            np.array([0.0, 1.0], dtype="<f4"),
            id="dtype",
        ),
        pytest.param(
            np.array([0.0, 1.0], dtype=">f8"),
            id="endianness",
        ),
        pytest.param(
            np.array([-0.0, 1.0], dtype="<f8"),
            id="signed-zero",
        ),
    ],
)
def test_bit_exact_metric_rejects_numeric_only_identity(
    candidate: np.ndarray,
) -> None:
    reference = np.array([0.0, 1.0], dtype="<f8")
    assert np.array_equal(reference, candidate)
    assert array_descriptor(reference) != array_descriptor(candidate)
    assert c2_score._difference(reference, candidate)["bit_exact"] is False
    with pytest.raises(C2ScoreError, match="array descriptors failed"):
        c2_score._assert_recomputed_array_identity(
            {"array_descriptors": {"sample": array_descriptor(candidate)}},
            {"sample": candidate},
            {"sample": reference},
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda score: score["stage"].__setitem__("stage_id", "C3"),
            "stage identity",
        ),
        (
            lambda score: score["acceptance"]["limits"].__setitem__(
                "raw_array_max_absolute_error",
                1.0,
            ),
            "acceptance limits",
        ),
        (
            lambda score: score["sources"].__setitem__(
                "qpsim/constants.py",
                "0" * 64,
            ),
            "source binding",
        ),
        (
            lambda score: score["parent_bindings"].__setitem__(
                "c1_score_sha256",
                "0" * 64,
            ),
            "C1-score binding",
        ),
    ],
)
def test_checked_loader_rejects_bound_field_tampering(
    tmp_path: Path,
    mutation: object,
    match: str,
) -> None:
    score = json.loads(C2_SCORE.read_text(encoding="utf-8"))
    mutation(score)  # type: ignore[operator]
    path = tmp_path / "score.json"
    _write_json(path, score)
    with pytest.raises(C2ScoreError, match=match):
        load_c2_score(path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda score: score["raw_bundle"].__setitem__(
            "manifest_sha256",
            "0" * 64,
        ),
        lambda score: score["steps"][-1].__setitem__(
            "eq35_t_star_over_delta",
            123.0,
        ),
        lambda score: score["steps"][-1]["channel_difference_from_author"].__setitem__(
            "qp_photon",
            {},
        ),
        lambda score: score["structural_identity"]["descriptors"].__setitem__(
            "rho",
            {},
        ),
        lambda score: score["frozen_inputs"]["descriptors"].__setitem__(
            "f_final",
            {},
        ),
        lambda score: score["comparison"].__setitem__(
            "c2a_control_checks",
            {"invented": True},
        ),
        lambda score: score["parameter_axis"].__setitem__(
            "qpsim_fixed_nbar_t_star_over_delta",
            99.0,
        ),
        lambda score: score["steps"][-1].__setitem__(
            "qualification",
            "This forged text claims a solved C2 root.",
        ),
    ],
)
def test_checked_loader_rejects_scientific_or_raw_digest_tampering_without_raw_bundle(
    tmp_path: Path,
    mutation: object,
) -> None:
    score = json.loads(C2_SCORE.read_text(encoding="utf-8"))
    mutation(score)  # type: ignore[operator]
    path = tmp_path / "score.json"
    _write_json(path, score)
    with pytest.raises(
        C2ScoreError,
        match=r"committed raw-manifest receipt|comparison closure",
    ):
        load_c2_score(path)


def test_checked_loader_rejects_tampered_receipt(tmp_path: Path) -> None:
    receipt = json.loads(C2_RECEIPT.read_text(encoding="utf-8"))
    receipt["raw_bundle"]["manifest_sha256"] = "0" * 64
    path = tmp_path / "receipt.json"
    _write_json(path, receipt)
    with pytest.raises(C2ScoreError, match="committed receipt"):
        load_c2_score(receipt_path=path)


def test_plural_locality_claims_require_each_named_channel_to_change() -> None:
    score = load_c2_score()
    by_id = {step["step_id"]: step for step in score["steps"]}
    cases = {
        "C2b4-declared-pair-breaking-time": (
            "phonon_scattering",
            "phonon_pair",
        ),
        "C2b5-finite-cutoff-critical-temperature": (
            "qp_scattering",
            "qp_pair",
        ),
    }
    for step_id, channels in cases.items():
        differences = by_id[step_id]["channel_difference_from_previous"]
        for channel in channels:
            assert any(
                not comparison["bit_exact"]
                for comparison in differences[channel].values()
            )


def test_modern_kb_locality_covers_every_changed_and_preserved_channel() -> None:
    score = load_c2_score()
    modern_kb = next(
        step
        for step in score["steps"]
        if step["step_id"] == "C2b2-modern-kB"
    )
    differences = modern_kb["channel_difference_from_previous"]

    for channel in ("qp_scattering", "qp_pair"):
        assert all(
            comparison["bit_exact"] is False
            for comparison in differences[channel].values()
        )
    assert differences["phonon_escape"]["gain_s_inv"]["bit_exact"] is False
    assert differences["phonon_escape"]["net_s_inv"]["bit_exact"] is False
    assert differences["phonon_escape"]["loss_s_inv"]["bit_exact"] is True
    for channel in ("qp_photon", "phonon_scattering", "phonon_pair"):
        assert all(
            comparison["bit_exact"] is True
            for comparison in differences[channel].values()
        )

    locality = score["comparison"]["locality_checks"]
    assert {
        "modern_kb_changes_phonon_escape_gain_and_net",
        "modern_kb_changes_qp_phonon_channels",
        "modern_kb_changes_thermal_occupation",
        "modern_kb_leaves_kb_independent_channels_exact",
        "modern_kb_preserves_phonon_escape_loss_exact",
    } <= {name for name, passed in locality.items() if passed is True}


def test_c2_sources_do_not_run_a_nonlinear_solve() -> None:
    for relative in (
        "validation/fischer_2023/fig6_author_c2_bundle.py",
        "validation/fischer_2023/fig6_author_c2_score.py",
    ):
        source = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "solve_author_system(" not in source
        assert "newton_deltas" not in source
        assert "state_history" not in source


def test_c2_parameter_layer_does_not_load_implicit_material_defaults() -> None:
    source = (
        REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_c2_parameters.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not any(name.startswith("qpsim.materials") for name in imported_modules)
    assert not any(
        name is not None and name.startswith("qpsim.materials")
        for name in imported_from
    )
    assert "load_material" not in calls
