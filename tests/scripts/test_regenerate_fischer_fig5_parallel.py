"""Cheap contract tests for the guarded Fischer Fig. 5 campaign driver.

Every payload in this module is synthetic.  No test calls the numerical
Fig. 5 solver.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path, PureWindowsPath
from typing import Any, cast

import numpy as np
import pytest
from validation.fischer_2023 import fig5_paper, fig5_solve
from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_FIELDS,
)

from scripts import regenerate_fischer_fig5_parallel as driver


@pytest.fixture(scope="module")
def producer() -> dict[str, object]:
    return driver._producer_base(Path(driver.__file__).resolve())


def _reject_nonfinite_json(token: str) -> object:
    raise ValueError(f"non-finite JSON token {token!r}")


def _load_strict_canonical_json(path: Path) -> dict[str, Any]:
    """Reject duplicate keys/non-finite tokens and require canonical bytes."""
    text = path.read_text(encoding="utf-8")
    payload = json.loads(
        text,
        object_pairs_hook=fig5_paper._strict_json_object,
        parse_constant=_reject_nonfinite_json,
    )
    assert isinstance(payload, dict)
    assert text == json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return cast(dict[str, Any], payload)


def _assert_sha256(value: object) -> None:
    assert isinstance(value, str)
    assert len(value) == 64
    assert all(character in "0123456789abcdef" for character in value)


def _minimal_row_arrays() -> dict[str, np.ndarray]:
    """Return the exact archive member set; semantic checks are mocked."""
    return {
        name: np.asarray([index], dtype=float)
        for index, name in enumerate(driver.RAW_ARRAY_KEYS)
    }


def _write_authenticated_row(
    path: Path,
    *,
    spec: driver.RowSpec,
    producer: dict[str, object],
) -> None:
    arrays = _minimal_row_arrays()
    arrays["elapsed_s"] = np.asarray([1.25], dtype=float)
    arrays["metadata_json"] = np.asarray(
        driver._canonical_json(
            driver._row_metadata(spec=spec, producer=producer)
        )
    )
    driver._atomic_npz(path, arrays)


def _synthetic_row(spec: driver.RowSpec) -> dict[str, np.ndarray]:
    """Small, shape-compatible row used only to test deterministic assembly."""
    upper_nbar = np.array(
        tuple(float(value) for value in fig5_solve.UPPER_NBAR_VALUES),
        dtype=float,
    )
    lower_temperatures = np.array(
        tuple(float(value) for value in fig5_solve.LOWER_T_BATH_K),
        dtype=float,
    )
    common: dict[str, np.ndarray] = {
        "tau_0_pb_ns": np.asarray([0.255]),
        "tau_l_ns": np.asarray([0.255]),
        "num_bins": np.asarray([1]),
    }
    if spec.panel == "upper":
        upper_shape = (1, upper_nbar.size)
        lower_shape = (0, 0)
        upper_value = float(spec.ordinal + 1)
        lower_value = 0.0
        axes: dict[str, np.ndarray] = {
            "upper_T_bath": np.asarray([spec.axis_value]),
            "upper_nbar": upper_nbar.copy(),
            "lower_nbar": np.asarray([], dtype=float),
            "lower_T_bath": np.asarray([], dtype=float),
        }
    else:
        upper_shape = (0, 0)
        lower_shape = (1, lower_temperatures.size)
        upper_value = 0.0
        lower_value = float(spec.ordinal + 1)
        axes = {
            "upper_T_bath": np.asarray([], dtype=float),
            "upper_nbar": np.asarray([], dtype=float),
            "lower_nbar": np.asarray([spec.axis_value]),
            "lower_T_bath": lower_temperatures.copy(),
        }
    arrays: dict[str, np.ndarray] = {
        **common,
        **axes,
        "upper_f": np.full((*upper_shape, 1), upper_value),
        "upper_n_ph": np.full((*upper_shape, 1), upper_value),
        "lower_f": np.full((*lower_shape, 1), lower_value),
        "lower_n_ph": np.full((*lower_shape, 1), lower_value),
    }
    for field in NUMBER_CERTIFICATE_FIELDS:
        arrays[f"upper_{field}"] = np.full(upper_shape, upper_value)
        arrays[f"lower_{field}"] = np.full(lower_shape, lower_value)
    return arrays


def _record_for(
    path: Path,
    spec: driver.RowSpec,
) -> driver.RowRecord:
    return driver.RowRecord(
        ordinal=spec.ordinal,
        label=spec.label,
        panel=spec.panel,
        row_index=spec.row_index,
        axis_value=spec.axis_value,
        elapsed_s=1.0 + spec.ordinal,
        path=str(path),
        sha256=driver._sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def test_exact_six_row_decomposition_covers_all_81_points() -> None:
    specs = driver._row_specs()
    assert [
        (spec.ordinal, spec.label, spec.panel, spec.row_index)
        for spec in specs
    ] == [
        (0, "upper-t00", "upper", 0),
        (1, "upper-t01", "upper", 1),
        (2, "upper-t02", "upper", 2),
        (3, "lower-n00", "lower", 0),
        (4, "lower-n01", "lower", 1),
        (5, "lower-n02", "lower", 2),
    ]
    np.testing.assert_array_equal(
        [spec.axis_value for spec in specs[:3]],
        fig5_solve.UPPER_T_BATH_K,
    )
    np.testing.assert_array_equal(
        [spec.axis_value for spec in specs[3:]],
        fig5_solve.LOWER_NBAR,
    )
    upper_points = (
        len(specs[:3]) * fig5_solve.UPPER_NBAR_VALUES.size
    )
    lower_points = (
        len(specs[3:]) * fig5_solve.LOWER_T_BATH_K.size
    )
    assert (upper_points, lower_points, upper_points + lower_points) == (
        42,
        39,
        81,
    )


def test_campaign_record_is_a_sibling_of_the_promotion_record() -> None:
    promotion = fig5_paper.promotion_record_path()
    campaign = driver._campaign_record_path()

    assert campaign.parent == promotion.parent
    assert campaign.name == "fischer_fig5_paper.campaign.json"


def test_canonical_campaign_record_authenticates_promoted_bundle() -> None:
    """Bind the promoted bundle to six claimed row identities.

    The committed companion is portable and self-consistent; it does not
    contain the external NPZ row bytes. The opt-in closeout test below
    independently reauthenticates those bytes when ``QPSIM_FIG5_RUN_ROOT`` is
    available.
    """
    campaign_path = driver._campaign_record_path()
    assert campaign_path.is_file()
    campaign_bytes = campaign_path.read_bytes()
    status = _load_strict_canonical_json(campaign_path)
    promotion = fig5_paper.read_promotion_record()

    assert set(status) == {
        "attempt",
        "completed_rows",
        "completion",
        "producer_sha256",
        "run_identity",
        "schema",
        "started_utc",
        "state",
        "total_rows",
        "updated_utc",
    }
    assert isinstance(status["schema"], str) and status["schema"]
    assert status["state"] == "complete"
    assert status["completed_rows"] == status["total_rows"] == 6

    completion = cast(dict[str, Any], status["completion"])
    assert set(completion) == {
        "aggregate_worker_s",
        "artifacts",
        "campaign",
        "certificate_maxima",
        "new_rows",
        "resumed_rows",
        "row_payload_sha256",
        "wall_s",
    }
    assert (
        isinstance(completion["new_rows"], int)
        and not isinstance(completion["new_rows"], bool)
        and isinstance(completion["resumed_rows"], int)
        and not isinstance(completion["resumed_rows"], bool)
        and completion["new_rows"] + completion["resumed_rows"] == 6
    )
    assert float(completion["aggregate_worker_s"]) > 0.0
    assert float(completion["wall_s"]) > 0.0

    campaign = cast(dict[str, Any], completion["campaign"])
    producer = fig5_paper._campaign_producer(
        cast(dict[str, object], promotion["producer"]),
        campaign,
    )
    producer_sha256 = fig5_paper._campaign_producer_sha256(producer)
    run_identity = fig5_paper._campaign_run_identity(
        producer,
        row_schema=campaign["row_schema"],
        status_schema=campaign["status_schema"],
    )
    # Authenticate the historical campaign from its own frozen runner and
    # environment evidence. A provenance-only reader rebind must not require
    # the current runner source hash to equal the historical one.
    assert status["producer_sha256"] == producer_sha256
    assert status["run_identity"] == run_identity
    assert set(campaign) == {
        "mode",
        "row_schema",
        "rows",
        "run_identity",
        "runner",
        "single_thread_environment",
        "status_schema",
    }
    assert campaign["mode"] == producer["mode"]
    assert isinstance(campaign["row_schema"], str) and campaign["row_schema"]
    assert (
        isinstance(campaign["status_schema"], str)
        and campaign["status_schema"]
    )
    assert status["schema"] == campaign["status_schema"]
    assert campaign["run_identity"] == run_identity
    assert campaign["runner"] == producer["runner"]
    assert (
        campaign["single_thread_environment"]
        == producer["single_thread_environment"]
    )

    row_entries = cast(list[dict[str, Any]], campaign["rows"])
    specs = driver._row_specs()
    assert len(row_entries) == len(specs) == 6
    expected_hashes: dict[str, str] = {}
    for entry, spec in zip(row_entries, specs, strict=True):
        assert set(entry) == {
            "axis_value",
            "label",
            "ordinal",
            "panel",
            "row_index",
            "sha256",
            "size_bytes",
        }
        assert (
            entry["ordinal"],
            entry["label"],
            entry["panel"],
            entry["row_index"],
            entry["axis_value"],
        ) == (
            spec.ordinal,
            spec.label,
            spec.panel,
            spec.row_index,
            spec.axis_value,
        )
        _assert_sha256(entry["sha256"])
        assert (
            isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] > 0
        )
        expected_hashes[spec.label] = cast(str, entry["sha256"])
    assert completion["row_payload_sha256"] == expected_hashes

    artifact_paths = {
        "csv": fig5_paper.baseline_path(),
        "pdf_a": fig5_paper.plot_path_a(),
        "pdf_b": fig5_paper.plot_path_b(),
        "record": fig5_paper.promotion_record_path(),
    }
    publication = cast(dict[str, Any], promotion["publication"])
    if publication["kind"] == fig5_paper._PROVENANCE_REBIND:
        rebound_from = cast(
            dict[str, Any],
            publication["rebound_from"],
        )
        generation_artifacts = dict(
            cast(dict[str, dict[str, object]], rebound_from["artifacts"])
        )
        generation_artifacts["record"] = cast(
            dict[str, object],
            rebound_from["promotion_record"],
        )
    else:
        generation_artifacts = {
            **cast(
                dict[str, dict[str, object]],
                promotion["artifacts"],
            ),
            "record": fig5_paper._file_identity(
                fig5_paper.promotion_record_path()
            ),
        }
    artifact_entries = cast(dict[str, dict[str, object]], completion["artifacts"])
    assert set(artifact_entries) == set(artifact_paths)
    for name, path in artifact_paths.items():
        evidence = artifact_entries[name]
        assert set(evidence) == {"path", "sha256"}
        # Absolute Windows paths are host annotations.  Filename + digest are
        # the portable identity carried by the committed companion.
        # The authenticated campaign was produced on Windows, so its
        # historical path uses backslashes even when a Linux CI reader
        # verifies it.
        assert PureWindowsPath(cast(str, evidence["path"])).name == path.name
        assert evidence["sha256"] == generation_artifacts[name]["sha256"]

    rows = fig5_paper._read_csv_rows(fig5_paper.baseline_path())
    metadata = fig5_paper._artifact_metadata(fig5_paper.baseline_path(), rows)
    assert completion["certificate_maxima"] == metadata["certificate_maxima"]
    assert campaign_path.read_bytes() == campaign_bytes


def test_external_completed_campaign_rows_are_reauthenticated() -> None:
    """Opt-in closeout: validate durable rows before their dtype is coerced."""
    root_text = os.environ.get("QPSIM_FIG5_RUN_ROOT")
    if not root_text:
        pytest.skip("set QPSIM_FIG5_RUN_ROOT to revalidate external Fig. 5 rows")

    promotion = fig5_paper.read_promotion_record()
    canonical_path = driver._campaign_record_path()
    canonical = _load_strict_canonical_json(canonical_path)
    campaign = cast(dict[str, Any], canonical["completion"]["campaign"])
    producer = fig5_paper._campaign_producer(
        cast(dict[str, object], promotion["producer"]),
        campaign,
    )
    run_identity = fig5_paper._campaign_run_identity(
        producer,
        row_schema=campaign["row_schema"],
        status_schema=campaign["status_schema"],
    )
    run_dir = (
        Path(root_text).expanduser().resolve()
        / f"fig5-{run_identity}"
    )
    status_path = run_dir / "status.json"
    assert status_path.read_bytes() == canonical_path.read_bytes()
    status = _load_strict_canonical_json(status_path)
    assert status == canonical
    assert status["state"] == "complete"
    assert status["run_identity"] == run_identity

    records = driver._scan_rows(run_dir, producer)
    assert set(records) == {spec.ordinal for spec in driver._row_specs()}
    evidence_by_label = {
        entry["label"]: entry for entry in cast(list[dict[str, Any]], campaign["rows"])
    }
    expected_names = {*driver.RAW_ARRAY_KEYS, "elapsed_s", "metadata_json"}
    for spec in driver._row_specs():
        record = records[spec.ordinal]
        evidence = evidence_by_label[spec.label]
        path = driver._row_path(run_dir, spec)
        assert evidence["sha256"] == record["sha256"] == driver._sha256_file(path)
        assert evidence["size_bytes"] == record["size_bytes"] == path.stat().st_size
        with np.load(path, allow_pickle=False) as payload:
            assert set(payload.files) == expected_names
            assert payload["num_bins"].dtype == np.dtype("int64")
            assert payload["elapsed_s"].dtype == np.dtype("float64")
            assert payload["metadata_json"].dtype.kind == "U"
            for name in driver.RAW_ARRAY_KEYS:
                if name != "num_bins":
                    assert payload[name].dtype == np.dtype("float64"), (
                        path,
                        name,
                        payload[name].dtype,
                    )


def test_run_identity_is_deterministic_and_sensitive_to_every_provenance_lens(
    tmp_path: Path,
    producer: dict[str, object],
) -> None:
    identity = driver._run_identity(producer)
    assert identity == driver._run_identity(deepcopy(producer))
    assert len(identity) == 64
    assert driver._run_dir(tmp_path, producer) == (
        tmp_path.resolve() / f"fig5-{identity}"
    )

    changed_runner = deepcopy(producer)
    cast(dict[str, Any], changed_runner["runner"])["sha256"] = "0" * 64

    changed_source = deepcopy(producer)
    artifact = cast(dict[str, Any], changed_source["artifact_producer"])
    fingerprint = cast(dict[str, Any], artifact["fingerprint"])
    fingerprint["source_sha256"] = {"synthetic.py": "1" * 64}

    changed_runtime = deepcopy(producer)
    artifact = cast(dict[str, Any], changed_runtime["artifact_producer"])
    runtime = cast(dict[str, Any], artifact["runtime"])
    runtime["synthetic-runtime-marker"] = "changed"

    identities = {
        identity,
        driver._run_identity(changed_runner),
        driver._run_identity(changed_source),
        driver._run_identity(changed_runtime),
    }
    assert len(identities) == 4


def test_runner_record_hashes_logical_source_across_newline_policies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LF/CRLF checkout policy must not change the campaign identity."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    runner_path = scripts_dir / "runner.py"
    monkeypatch.setattr(
        driver,
        "__file__",
        str(scripts_dir / "regenerate_fischer_fig5_parallel.py"),
    )

    records: list[dict[str, object]] = []
    for payload in (b"alpha\nbeta\n", b"alpha\r\nbeta\r\n", b"alpha\rbeta\r"):
        runner_path.write_bytes(payload)
        records.append(driver._runner_record(runner_path))

    expected = {
        "path": "scripts/runner.py",
        "sha256": hashlib.sha256(b"alpha\nbeta\n").hexdigest(),
    }
    assert records == [expected, expected, expected]


def test_thread_controls_are_enforced_before_driver_numpy_import() -> None:
    assert {
        name: os.environ.get(name)
        for name in driver.SINGLE_THREAD_ENVIRONMENT
    } == driver.SINGLE_THREAD_ENVIRONMENT
    driver._assert_single_thread_environment()


def test_fresh_campaign_process_sets_threads_before_numpy() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import regenerate_fischer_fig5_parallel as d; "
                "assert not d._NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT; "
                "d._assert_single_thread_environment()"
            ),
        ],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_atomic_row_and_status_helpers_replace_without_temp_debris(
    tmp_path: Path,
) -> None:
    row_path = tmp_path / "row-upper-t00.npz"
    driver._atomic_npz(
        row_path,
        {
            "a": np.asarray([1.0, 2.0]),
            "b": np.asarray("first"),
        },
    )
    driver._atomic_npz(
        row_path,
        {
            "a": np.asarray([3.0]),
            "b": np.asarray("replacement"),
        },
    )
    with np.load(row_path, allow_pickle=False) as payload:
        assert set(payload.files) == {"a", "b"}
        np.testing.assert_array_equal(payload["a"], [3.0])
        assert str(np.asarray(payload["b"]).item()) == "replacement"
    assert not list(tmp_path.glob(".row-upper-t00.npz.*.tmp"))

    status_path = tmp_path / "status.json"
    status: dict[str, object] = {"state": "running", "completed_rows": 2}
    driver._atomic_json(status_path, status)
    assert json.loads(status_path.read_text(encoding="utf-8")) == status
    with pytest.raises(ValueError, match="Out of range float values"):
        driver._atomic_json(status_path, {"bad": float("nan")})
    assert json.loads(status_path.read_text(encoding="utf-8")) == status
    assert not list(tmp_path.glob(".status.json.*.tmp"))


def test_resume_accepts_only_an_authenticated_current_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
) -> None:
    monkeypatch.setattr(
        driver,
        "_validate_row_payload",
        lambda *_args, **_kwargs: None,
    )
    run_dir = driver._run_dir(tmp_path, producer)
    run_dir.mkdir(parents=True)
    spec = driver._row_specs()[0]
    path = driver._row_path(run_dir, spec)
    _write_authenticated_row(path, spec=spec, producer=producer)

    records = driver._scan_rows(run_dir, producer)
    assert set(records) == {spec.ordinal}
    record = records[spec.ordinal]
    assert record["label"] == spec.label
    assert record["sha256"] == driver._sha256_file(path)
    assert record["size_bytes"] == path.stat().st_size


@pytest.mark.parametrize("corruption", ["bytes", "producer"])
def test_resume_rejects_corrupt_or_foreign_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
    corruption: str,
) -> None:
    monkeypatch.setattr(
        driver,
        "_validate_row_payload",
        lambda *_args, **_kwargs: None,
    )
    run_dir = driver._run_dir(tmp_path, producer)
    run_dir.mkdir(parents=True)
    spec = driver._row_specs()[0]
    path = driver._row_path(run_dir, spec)
    _write_authenticated_row(path, spec=spec, producer=producer)

    if corruption == "bytes":
        path.write_bytes(b"not-an-npz")
    else:
        with np.load(path, allow_pickle=False) as payload:
            arrays = {
                name: np.asarray(payload[name]).copy()
                for name in payload.files
            }
        metadata = json.loads(str(arrays["metadata_json"].item()))
        metadata["producer_sha256"] = "0" * 64
        arrays["metadata_json"] = np.asarray(
            driver._canonical_json(metadata)
        )
        driver._atomic_npz(path, arrays)

    with pytest.raises(RuntimeError, match=r"row payload|producer_sha256"):
        driver._scan_rows(run_dir, producer)


def test_resume_rejects_an_unexpected_row_file(
    tmp_path: Path,
    producer: dict[str, object],
) -> None:
    run_dir = driver._run_dir(tmp_path, producer)
    run_dir.mkdir(parents=True)
    (run_dir / "row-foreign.npz").write_bytes(b"foreign")
    with pytest.raises(RuntimeError, match="unexpected row payload"):
        driver._scan_rows(run_dir, producer)


def test_assembly_is_spec_ordered_even_when_records_are_reversed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
) -> None:
    run_dir = driver._run_dir(tmp_path, producer)
    run_dir.mkdir(parents=True)
    specs = driver._row_specs()
    rows: dict[int, dict[str, np.ndarray]] = {}
    records: dict[int, driver.RowRecord] = {}
    for spec in reversed(specs):
        path = driver._row_path(run_dir, spec)
        path.write_bytes(f"row-{spec.ordinal}".encode())
        rows[spec.ordinal] = _synthetic_row(spec)
        records[spec.ordinal] = _record_for(path, spec)

    def fake_load(
        _path: Path,
        *,
        spec: driver.RowSpec,
        producer: dict[str, object],
    ) -> tuple[dict[str, np.ndarray], float]:
        del producer
        return rows[spec.ordinal], 1.0

    observed: list[dict[str, np.ndarray]] = []

    def fake_observables(
        raw: dict[str, np.ndarray],
    ) -> object:
        observed.append(raw)
        return object()

    monkeypatch.setattr(driver, "_load_row_arrays", fake_load)
    monkeypatch.setattr(fig5_paper, "observables", fake_observables)
    monkeypatch.setattr(
        fig5_paper,
        "_validate_result_for_artifact",
        lambda _result: {},
    )

    assembled, evidence = driver._assemble_raw(
        records,
        run_dir=run_dir,
        producer=producer,
    )
    assert observed == [assembled]
    np.testing.assert_array_equal(
        assembled["upper_T_bath"],
        fig5_solve.UPPER_T_BATH_K,
    )
    np.testing.assert_array_equal(
        assembled["lower_nbar"],
        fig5_solve.LOWER_NBAR,
    )
    np.testing.assert_array_equal(
        assembled["upper_f"][:, 0, 0],
        [1.0, 2.0, 3.0],
    )
    np.testing.assert_array_equal(
        assembled["lower_f"][:, 0, 0],
        [4.0, 5.0, 6.0],
    )
    assert list(evidence) == [spec.label for spec in specs]


def test_assembly_rejects_an_incomplete_row_set_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
) -> None:
    loaded = False

    def fail_if_loaded(*_args: object, **_kwargs: object) -> object:
        nonlocal loaded
        loaded = True
        raise AssertionError("incomplete assembly must not load rows")

    monkeypatch.setattr(driver, "_load_row_arrays", fail_if_loaded)
    with pytest.raises(RuntimeError, match="incomplete before assembly"):
        driver._assemble_raw({}, run_dir=tmp_path, producer=producer)
    assert not loaded


def test_campaign_never_publishes_an_incomplete_row_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
) -> None:
    monkeypatch.setattr(
        driver,
        "_NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT",
        False,
    )
    monkeypatch.setattr(driver, "_producer_base", lambda _path: producer)
    monkeypatch.setattr(driver, "_scan_rows", lambda *_args: {})
    monkeypatch.setattr(
        driver,
        "_run_missing_rows",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        driver,
        "_assert_producer_current",
        lambda *_args, **_kwargs: None,
    )
    published = False

    def fail_if_published(*_args: object, **_kwargs: object) -> object:
        nonlocal published
        published = True
        raise AssertionError("publisher must not receive incomplete rows")

    monkeypatch.setattr(
        fig5_paper,
        "publish_artifacts",
        fail_if_published,
    )

    with pytest.raises(RuntimeError, match="incomplete before assembly"):
        driver.run_campaign(run_root=tmp_path, max_workers=6)
    assert not published
    run_dir = driver._run_dir(tmp_path, producer)
    status = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["completed_rows"] == 0


def test_complete_status_binds_certificate_maxima_and_sorted_row_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    producer: dict[str, object],
) -> None:
    monkeypatch.setattr(
        driver,
        "_NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT",
        False,
    )
    monkeypatch.setattr(driver, "_producer_base", lambda _path: producer)
    monkeypatch.setattr(
        driver,
        "_assert_producer_current",
        lambda *_args, **_kwargs: None,
    )
    campaign_record = tmp_path / "canonical.campaign.json"
    monkeypatch.setattr(
        driver,
        "_campaign_record_path",
        lambda: campaign_record,
    )

    run_dir = driver._run_dir(tmp_path, producer)
    records: dict[int, driver.RowRecord] = {}
    for spec in reversed(driver._row_specs()):
        path = driver._row_path(run_dir, spec)
        records[spec.ordinal] = driver.RowRecord(
            ordinal=spec.ordinal,
            label=spec.label,
            panel=spec.panel,
            row_index=spec.row_index,
            axis_value=spec.axis_value,
            elapsed_s=float(spec.ordinal + 1),
            path=str(path),
            sha256=f"{spec.ordinal + 1:064x}",
            size_bytes=1000 + spec.ordinal,
        )
    monkeypatch.setattr(
        driver,
        "_scan_rows",
        lambda *_args: dict(records),
    )

    raw = {"synthetic": np.asarray([1.0])}
    row_hashes = {
        spec.label: records[spec.ordinal]["sha256"]
        for spec in driver._row_specs()
    }
    monkeypatch.setattr(
        driver,
        "_assemble_raw",
        lambda *_args, **_kwargs: (raw, row_hashes),
    )
    result = object()
    monkeypatch.setattr(
        fig5_paper,
        "observables",
        lambda actual: result if actual is raw else None,
    )
    maxima = {
        field: float(index) * 1.0e-12
        for index, field in enumerate(NUMBER_CERTIFICATE_FIELDS)
    }
    monkeypatch.setattr(
        fig5_paper,
        "_validate_result_for_artifact",
        lambda actual: maxima if actual is result else {},
    )

    artifacts = tuple(
        tmp_path / name
        for name in (
            "candidate.csv",
            "candidate-a.pdf",
            "candidate-b.pdf",
            "candidate.promotion.json",
        )
    )
    artifact_producer_expected = producer["artifact_producer"]

    def fake_publish(
        actual: object,
        *,
        producer: object,
    ) -> tuple[Path, Path, Path, Path]:
        assert actual is result
        assert producer == artifact_producer_expected
        for index, path in enumerate(artifacts):
            path.write_bytes(f"artifact-{index}".encode())
        return cast(tuple[Path, Path, Path, Path], artifacts)

    # The fake publisher checks the result and materializes hashable outputs;
    # producer identity is already pinned independently in ``campaign`` below.
    monkeypatch.setattr(fig5_paper, "publish_artifacts", fake_publish)

    status = driver.run_campaign(run_root=tmp_path, max_workers=6)
    completion = cast(dict[str, Any], status["completion"])
    campaign = cast(dict[str, Any], completion["campaign"])
    expected_rows = [
        {
            "axis_value": records[spec.ordinal]["axis_value"],
            "label": records[spec.ordinal]["label"],
            "ordinal": records[spec.ordinal]["ordinal"],
            "panel": records[spec.ordinal]["panel"],
            "row_index": records[spec.ordinal]["row_index"],
            "sha256": records[spec.ordinal]["sha256"],
            "size_bytes": records[spec.ordinal]["size_bytes"],
        }
        for spec in driver._row_specs()
    ]
    assert campaign == {
        "mode": "parallel-independent-continuation-rows",
        "row_schema": driver.ROW_SCHEMA,
        "rows": expected_rows,
        "run_identity": driver._run_identity(producer),
        "runner": producer["runner"],
        "single_thread_environment": producer[
            "single_thread_environment"
        ],
        "status_schema": driver.STATUS_SCHEMA,
    }
    assert completion["certificate_maxima"] == maxima
    assert completion["row_payload_sha256"] == row_hashes
    assert completion["aggregate_worker_s"] == 21.0
    assert completion["new_rows"] == 0
    assert completion["resumed_rows"] == 6
    assert status["state"] == "complete"
    assert status["completed_rows"] == 6
    restored = json.loads(
        (run_dir / "status.json").read_text(encoding="utf-8")
    )
    assert restored == status
    assert json.loads(campaign_record.read_text(encoding="utf-8")) == status
    assert (
        campaign_record.read_bytes()
        == (run_dir / "status.json").read_bytes()
    )
