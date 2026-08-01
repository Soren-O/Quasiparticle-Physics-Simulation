from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from validation.reproduction_ladder import (
    COMPONENTS,
    ReproductionLadderError,
    compare_arrays,
    describe_array,
    load_stage_catalog,
    validate_stage_catalog,
)


def _catalog() -> dict[str, object]:
    author = {component: f"author:{component}" for component in COMPONENTS}
    adapter = dict(author)
    adapter["io_capture"] = "qpsim:A1-capture"
    return {
        "claim": "A synthetic one-change ladder.",
        "ladder_id": "test-ladder",
        "oracle": {
            "hash_kind": "file_bytes",
            "manifest_path": "validation/oracle.json",
            "manifest_sha256": "a" * 64,
        },
        "schema": "qpsim.reproduction-ladder.v1",
        "stages": [
            {
                "branch": "author",
                "changed_component": None,
                "comparison_target": "paper oracle",
                "components": author,
                "evidence": [
                    {
                        "hash_kind": "file_bytes",
                        "path": "validation/oracle.json",
                        "role": "authenticated synthetic-source manifest",
                        "sha256": "b" * 64,
                    }
                ],
                "evidence_class": "author_supplied_reproduction_source",
                "parent_stage_id": None,
                "qualification": "Synthetic root.",
                "stage_id": "A0",
                "status": "source_authenticated_not_replayed",
            },
            {
                "branch": "author",
                "changed_component": "io_capture",
                "comparison_target": "A0",
                "components": adapter,
                "evidence": [],
                "evidence_class": "hybrid_component_substitution",
                "parent_stage_id": "A0",
                "qualification": "Only captures arrays.",
                "stage_id": "A1",
                "status": "planned",
            },
        ],
    }


def test_one_component_transition_is_accepted() -> None:
    stages = validate_stage_catalog(_catalog())
    assert [stage.stage_id for stage in stages] == ["A0", "A1"]


def test_two_component_transition_is_rejected() -> None:
    catalog = _catalog()
    stages = catalog["stages"]
    assert isinstance(stages, list)
    child = stages[1]
    assert isinstance(child, dict)
    components = child["components"]
    assert isinstance(components, dict)
    components["parameters"] = "qpsim:parameters"
    with pytest.raises(ReproductionLadderError, match="changed components"):
        validate_stage_catalog(catalog)


def test_declared_but_unchanged_component_is_rejected() -> None:
    catalog = _catalog()
    changed = copy.deepcopy(catalog)
    stages = changed["stages"]
    assert isinstance(stages, list)
    child = stages[1]
    assert isinstance(child, dict)
    child["changed_component"] = "parameters"
    with pytest.raises(ReproductionLadderError, match="changed components"):
        validate_stage_catalog(changed)


def test_array_identity_preserves_dtype_shape_and_complexness() -> None:
    real32 = np.array([[1.0, 2.0]], dtype="<f4")
    real64 = real32.astype("<f8")
    complex64 = real32.astype("<c8")
    assert describe_array(real32) != describe_array(real64)
    assert describe_array(real32) != describe_array(complex64)
    comparison = compare_arrays(real32, real64)
    assert comparison["max_absolute_error"] == 0.0
    assert comparison["exact_array_identity"] is False


def test_uint64_neighbors_have_nonzero_exact_error() -> None:
    reference = np.array([2**63], dtype=np.uint64)
    candidate = np.array([2**63 + 1], dtype=np.uint64)
    comparison = compare_arrays(reference, candidate)
    assert comparison["exact_array_identity"] is False
    assert comparison["max_absolute_error"] == 1.0
    assert comparison["max_relative_error"] > 0.0
    assert comparison["rms_error"] == 1.0


def test_object_arrays_are_refused() -> None:
    with pytest.raises(ReproductionLadderError, match="Object arrays"):
        describe_array(np.array([{"not": "scientific"}], dtype=object))


def test_checked_fig6_ladder_obeys_one_component_rule() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "validation" / "paper_data" / "fischer_2023" / "fig6" / "reproduction-ladder.json"
    stages = load_stage_catalog(path, repository_root=root)
    assert [stage.stage_id for stage in stages] == [
        "A0",
        "A1",
        "D0",
        "C0",
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "Q0",
        "Q1",
    ]
    assert stages[0].status == "source_authenticated_not_replayed"
    assert stages[1].status == "completed"
    assert stages[2].status == "completed"
    by_id = {stage.stage_id: stage for stage in stages}
    assert by_id["C0"].status == "completed"
    assert by_id["C1"].status == "completed"
    assert by_id["C2"].status == "completed"
    assert by_id["C3"].status == "completed"
    assert by_id["C4"].status == "completed"
    assert by_id["C5"].status == "completed"
    assert by_id["C6"].status == "completed"
    assert [item.role_kind for item in by_id["C6"].evidence] == [
        "implementation",
        "implementation",
        "supplemental",
        "result",
    ]
    assert by_id["C6"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c6-phonon-balance-score.json"
    )
    assert by_id["C7"].status == "completed"
    assert [item.role_kind for item in by_id["C7"].evidence] == [
        "implementation",
        "implementation",
        "supplemental",
        "result",
    ]
    assert by_id["C7"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c7-nonlinear-solver-score.json"
    )
    assert [item.role_kind for item in by_id["C2"].evidence] == [
        "implementation",
        "implementation",
        "implementation",
        "supplemental",
        "result",
    ]
    assert by_id["C2"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c2-raw-manifest-receipt.json"
    )
    assert [item.role_kind for item in by_id["C3"].evidence] == [
        "implementation",
        "implementation",
        "supplemental",
        "result",
        "supplemental",
    ]
    assert by_id["C3"].evidence[2].path == (
        "validation/paper_data/fischer_2023/fig6/c3-raw-manifest-receipt.json"
    )
    assert by_id["C3"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c3-grid-score.json"
    )
    assert [item.role_kind for item in by_id["C4"].evidence] == [
        "implementation",
        "implementation",
        "supplemental",
        "result",
    ]
    assert by_id["C4"].evidence[2].path == (
        "validation/paper_data/fischer_2023/fig6/c4-raw-manifest-receipt.json"
    )
    assert by_id["C4"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c4-photon-score.json"
    )
    assert [item.role_kind for item in by_id["C5"].evidence] == [
        "implementation",
        "implementation",
        "supplemental",
        "result",
    ]
    assert by_id["C5"].evidence[2].path == (
        "validation/paper_data/fischer_2023/fig6/c5-raw-manifest-receipt.json"
    )
    assert by_id["C5"].evidence[3].path == (
        "validation/paper_data/fischer_2023/fig6/c5-qp-phonon-score.json"
    )


def test_checked_catalog_rejects_a_stale_oracle_binding() -> None:
    root = Path(__file__).resolve().parents[2]
    path = root / "validation" / "paper_data" / "fischer_2023" / "fig6" / "reproduction-ladder.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["oracle"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ReproductionLadderError, match=r"oracle.*SHA-256"):
        validate_stage_catalog(payload, repository_root=root)


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "ladder.json"
    path.write_text('{"schema": "first", "schema": "second"}', encoding="utf-8")
    with pytest.raises(ReproductionLadderError, match="Duplicate JSON key 'schema'"):
        load_stage_catalog(path)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_strict_loader_rejects_nonfinite_numbers(
    tmp_path: Path,
    literal: str,
) -> None:
    path = tmp_path / "ladder.json"
    path.write_text(f'{{"number": {literal}}}', encoding="utf-8")
    with pytest.raises(ReproductionLadderError, match="Non-finite JSON"):
        load_stage_catalog(path)


def test_repository_binding_rejects_intermediate_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    oracle = real / "oracle.json"
    oracle.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    catalog = _catalog()
    oracle_mapping = catalog["oracle"]
    assert isinstance(oracle_mapping, dict)
    oracle_mapping["manifest_path"] = "linked/oracle.json"
    oracle_mapping["manifest_sha256"] = hashlib.sha256(oracle.read_bytes()).hexdigest()
    with pytest.raises(ReproductionLadderError, match="symlink"):
        validate_stage_catalog(catalog, repository_root=tmp_path)


def test_completed_stage_rejects_unrelated_project_metadata_as_proof() -> None:
    catalog = _catalog()
    stages = catalog["stages"]
    assert isinstance(stages, list)
    child = stages[1]
    assert isinstance(child, dict)
    child["status"] = "completed"
    child["evidence"] = [
        {
            "hash_kind": "file_bytes",
            "path": "pyproject.toml",
            "role": "project configuration",
            "sha256": "c" * 64,
        }
    ]
    with pytest.raises(ReproductionLadderError, match="required evidence roles"):
        validate_stage_catalog(catalog)


def _completed_evidence() -> list[dict[str, str]]:
    return [
        {
            "hash_kind": "source_canonical",
            "path": "validation/reproduction_ladder.py",
            "role": "hybrid stage implementation",
            "sha256": "c" * 64,
        },
        {
            "hash_kind": "file_bytes",
            "path": "validation/paper_data/result.json",
            "role": "provenance-bound comparison score",
            "sha256": "d" * 64,
        },
    ]


def _checked_catalog_payload() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    path = root / "validation" / "paper_data" / "fischer_2023" / "fig6" / "reproduction-ladder.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_completed_q1_rejects_planned_parent() -> None:
    payload = _checked_catalog_payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    q1 = next(stage for stage in stages if stage["stage_id"] == "Q1")
    q1["status"] = "completed"
    q1["evidence"] = _completed_evidence()
    with pytest.raises(
        ReproductionLadderError,
        match=r"(?:comparison stage|prerequisite parent) 'Q0'.*(?:planned|runnable)",
    ):
        validate_stage_catalog(payload)


def test_completed_c0_rejects_downgraded_runnable_a1_comparison() -> None:
    payload = _checked_catalog_payload()
    stages = payload["stages"]
    assert isinstance(stages, list)
    a1 = next(stage for stage in stages if stage["stage_id"] == "A1")
    a1["status"] = "runnable"
    c0 = next(stage for stage in stages if stage["stage_id"] == "C0")
    c0["status"] = "completed"
    c0["evidence"] = _completed_evidence()
    with pytest.raises(
        ReproductionLadderError,
        match=r"comparison stage 'A1'.*runnable",
    ):
        validate_stage_catalog(payload)


def test_child_must_machine_reference_parent_in_comparison_target() -> None:
    catalog = _catalog()
    stages = catalog["stages"]
    assert isinstance(stages, list)
    child = stages[1]
    assert isinstance(child, dict)
    child["comparison_target"] = "paper oracle"
    with pytest.raises(
        ReproductionLadderError,
        match=r"machine-readable stage ID|identify parent 'A0'",
    ):
        validate_stage_catalog(catalog)
