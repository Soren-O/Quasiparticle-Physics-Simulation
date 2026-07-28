"""Focused orchestration tests for the resumable Fig. 6 row campaign."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from validation.fischer_2023 import fig6_paper, fig6_solve

from scripts import regenerate_fischer_fig6_parallel as runner


def _producer() -> dict[str, object]:
    return {
        "run_identity": "a" * 64,
        "runner": {
            "path": "scripts/regenerate_fischer_fig6_parallel.py",
            "sha256": "b" * 64,
        },
        "runtime": {"python": "test"},
        "single_thread_environment": dict(runner.SINGLE_THREAD_ENVIRONMENT),
    }


def _raw_row(
    temperature: float,
    *,
    marker: float = 1.0,
) -> dict[str, np.ndarray]:
    n_bar = np.asarray(fig6_solve.N_BAR_VALUES, dtype=float)
    shape = (1, n_bar.size)
    matrices = {
        name: np.full(shape, marker, dtype=float)
        for name in (
            "T_star_over_delta",
            "delta_driven",
            "paper_observable_num",
            "paper_observable_eq53",
            "x_qp_num",
            "x_qp_eq47",
            *fig6_solve.FIG6_CERTIFICATE_FIELDS,
        )
    }
    matrices["delta_driven"].fill(179.0)
    return {
        "tau_0_pb_ns": np.asarray([0.255]),
        "tau_l_ns": np.asarray([0.255]),
        "T_bath": np.asarray([temperature]),
        "n_bar": n_bar,
        "delta_eq": np.asarray([179.5]),
        "state_f": np.full((*shape, 3), marker * 1.0e-8),
        "state_n_ph": np.full((*shape, 4), marker * 1.0e-10),
        **matrices,
    }


def _write_row(
    run_dir: Path,
    *,
    row_index: int,
    temperature: float,
    producer: dict[str, object],
    marker: float,
) -> Path:
    raw = _raw_row(temperature, marker=marker)
    metadata = runner._row_metadata(
        row_index=row_index,
        temperature=temperature,
        producer=producer,
    )
    arrays = {name: raw[name] for name in runner._RAW_ARRAY_KEYS}
    arrays["elapsed_s"] = np.asarray([marker])
    arrays["metadata_json"] = np.asarray(runner._canonical_json(metadata))
    path = runner._row_path(run_dir, row_index)
    runner._atomic_npz(path, arrays)
    return path


def test_temperature_row_entry_point_keeps_complete_continuation_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.1, 0.2, 0.3))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.asarray([1.0, 2.0, 4.0]))
    monkeypatch.setattr(fig6_solve, "_fischer_material", lambda: object())
    monkeypatch.setattr(
        fig6_solve,
        "_build_grid_and_spectral",
        lambda: (None, None, SimpleNamespace(E=np.arange(5.0))),
    )
    monkeypatch.setattr(fig6_solve, "_compute_tau_0_pb", lambda _spectral: 0.255)
    monkeypatch.setattr(fig6_solve, "T3DiffusionBackend", lambda: object())
    monkeypatch.setattr(fig6_solve, "_check_tau_0_pb", lambda *_args: None)
    captured: dict[str, object] = {}

    def fake_sweep(
        *_args: object,
        **kwargs: object,
    ) -> tuple[object, ...]:
        captured.update(kwargs)
        shape = (1, 3)
        certificates = {
            field: np.zeros(shape) for field in fig6_solve.FIG6_CERTIFICATE_FIELDS
        }
        return (
            np.zeros(shape),
            np.asarray([179.5]),
            np.full(shape, 179.0),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.asarray([0.255]),
            certificates,
            np.zeros((1, 3, 5)),
            np.zeros((1, 3, 7)),
        )

    monkeypatch.setattr(fig6_solve, "_solve_sweep", fake_sweep)
    raw = fig6_solve.solve_temperature_row(0.2)
    assert captured["t_bath_values"] == (0.2,)
    np.testing.assert_array_equal(
        captured["n_bar_values"],
        np.asarray([1.0, 2.0, 4.0]),
    )
    np.testing.assert_array_equal(raw["T_bath"], [0.2])
    np.testing.assert_array_equal(raw["n_bar"], [1.0, 2.0, 4.0])


def test_atomic_row_resume_is_strict_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer()
    monkeypatch.setattr(
        fig6_paper,
        "validate_temperature_row_payload",
        lambda raw, *, T_bath: fig6_paper.observables(raw),
    )
    temperatures = tuple(float(value) for value in fig6_solve.T_BATH_VALUES)
    path = _write_row(
        tmp_path,
        row_index=0,
        temperature=temperatures[0],
        producer=producer,
        marker=1.0,
    )
    record = runner._row_record(
        path,
        row_index=0,
        temperature=temperatures[0],
        producer=producer,
    )
    assert record["sha256"] == runner._sha256_file(path)
    assert runner._scan_rows(tmp_path, producer)[0] == record

    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]).copy() for name in payload.files}
    metadata = runner._row_metadata(
        row_index=0,
        temperature=temperatures[0],
        producer=producer,
    )
    metadata["run_identity"] = "f" * 64
    arrays["metadata_json"] = np.asarray(runner._canonical_json(metadata))
    runner._atomic_npz(path, arrays)
    with pytest.raises(RuntimeError, match="run_identity"):
        runner._scan_rows(tmp_path, producer)


def test_unexpected_or_corrupt_resume_payload_is_not_silently_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer()
    monkeypatch.setattr(
        fig6_paper,
        "validate_temperature_row_payload",
        lambda raw, *, T_bath: fig6_paper.observables(raw),
    )
    (tmp_path / "foreign.npz").write_bytes(b"not a row")
    with pytest.raises(RuntimeError, match="unexpected"):
        runner._scan_rows(tmp_path, producer)

    (tmp_path / "foreign.npz").unlink()
    path = runner._row_path(tmp_path, 0)
    path.write_bytes(b"truncated")
    with pytest.raises(RuntimeError, match="Cannot validate"):
        runner._scan_rows(tmp_path, producer)


def test_merge_is_in_canonical_temperature_order_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer()
    temperatures = tuple(float(value) for value in fig6_solve.T_BATH_VALUES)
    monkeypatch.setattr(
        fig6_paper,
        "validate_temperature_row_payload",
        lambda raw, *, T_bath: fig6_paper.observables(raw),
    )
    monkeypatch.setattr(fig6_paper, "validate_artifact_result", lambda _result: {})
    records: dict[int, runner.RowRecord] = {}
    for index in reversed(range(len(temperatures))):
        path = _write_row(
            tmp_path,
            row_index=index,
            temperature=temperatures[index],
            producer=producer,
            marker=float(index + 1),
        )
        records[index] = runner._row_record(
            path,
            row_index=index,
            temperature=temperatures[index],
            producer=producer,
        )
    raw, evidence = runner._assemble_raw(
        records,
        run_dir=tmp_path,
        producer=producer,
    )
    np.testing.assert_array_equal(raw["T_bath"], temperatures)
    np.testing.assert_array_equal(
        raw["paper_observable_num"][:, 0],
        np.arange(1.0, len(temperatures) + 1.0),
    )
    assert list(evidence) == [f"t{index:02d}" for index in range(len(temperatures))]
    result = fig6_paper.observables(raw)
    for index in range(len(temperatures)):
        assert evidence[f"t{index:02d}"]["semantic_sha256"] == (
            fig6_paper.result_row_sha256(result, index)
        )


def test_worker_writes_only_after_row_recertification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = _producer()
    temperature = float(fig6_solve.T_BATH_VALUES[0])
    raw = _raw_row(temperature)
    checks: list[float] = []
    monkeypatch.setattr(runner, "_assert_producer_current", lambda *_args: None)
    monkeypatch.setattr(
        fig6_solve,
        "solve_temperature_row",
        lambda _temperature: raw,
    )

    def validate(
        candidate: Mapping[str, np.ndarray],
        *,
        T_bath: float,
    ) -> fig6_paper.Fig6PaperResult:
        checks.append(float(T_bath))
        return fig6_paper.observables(candidate)

    monkeypatch.setattr(fig6_paper, "validate_temperature_row_payload", validate)
    record = runner._solve_one_row(
        (0, temperature, str(tmp_path), str(Path(runner.__file__)), producer)
    )
    assert checks == [temperature, temperature]
    assert Path(record["path"]).is_file()


def test_parallel_generation_evidence_binds_current_runner_and_all_rows() -> None:
    runner_path = Path(runner.__file__).resolve()
    producer = runner._producer_base(runner_path)
    payloads = {
        f"t{index:02d}": {
            "semantic_sha256": f"{index + 11:064x}",
            "temperature_K": float(temperature),
        }
        for index, temperature in enumerate(fig6_solve.T_BATH_VALUES)
    }
    evidence = {
        "campaign": {
            "aggregate_worker_s": 3.0,
            "new_rows": 2,
            "resumed_rows": 1,
            "wall_s": 2.0,
        },
        "mode": "parallel-temperature-rows",
        "run_identity": producer["run_identity"],
        "runner": producer["runner"],
        "runtime": producer["runtime"],
        "schema": fig6_paper.GENERATION_EVIDENCE_SCHEMA,
        "single_thread_environment": producer["single_thread_environment"],
        "worker_payloads": payloads,
    }
    assert fig6_paper.validate_generation_evidence(evidence) == evidence
    evidence["runner"] = dict(cast(dict[str, object], evidence["runner"]))
    cast(dict[str, object], evidence["runner"])["sha256"] = "0" * 64
    with pytest.raises(fig6_paper.ArtifactValidationError, match="runner source"):
        fig6_paper.validate_generation_evidence(evidence)


def test_campaign_lock_and_worker_bound_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with (
        runner._campaign_lock(tmp_path),
        pytest.raises(RuntimeError, match=r"Another Fig\. 6 campaign"),
        runner._campaign_lock(tmp_path),
    ):
        pytest.fail("nested campaign unexpectedly acquired the OS lock")
    monkeypatch.setattr(runner, "_NUMPY_IMPORTED_BEFORE_THREAD_ENFORCEMENT", False)
    with pytest.raises(ValueError, match=r"\[1, 3\]"):
        runner.run_campaign(run_root=tmp_path, max_workers=4)
