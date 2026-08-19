"""Resume-integrity tests for the two prelim overnight campaign runners."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scripts.run_prelim_readout_heating_overnight as readout
import scripts.run_prelim_spatial_overnight as spatial

RUNNERS = (spatial, readout)
TEST_NX = 3
TEST_TOTAL_TIME_NS = 10.0


def _write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    runner: object,
    run_id: str,
    *,
    status: str = "completed",
    converged: bool = True,
    nx: int | float | str = TEST_NX,
    total_time_ns: float = TEST_TOTAL_TIME_NS,
) -> dict[str, object]:
    fields = runner.SUMMARY_FIELDS
    row: dict[str, object] = dict.fromkeys(fields, "0")
    row.update(
        {
            "run_id": run_id,
            "status": status,
            "NX": nx,
            "dt_ns": 0.1,
            "max_time_ns": TEST_TOTAL_TIME_NS,
            "total_time_ns": total_time_ns,
            "n_steps": round(total_time_ns / 0.1),
            "converged": converged,
            "error": "" if status == "completed" else "failed attempt",
            "trace_csv": f"trace_{run_id}.csv",
            "profile_csv": f"profile_{run_id}.csv",
            "xqp_mean": 2.0,
            "xqp_source": 1.0,
            "xqp_open_end": 3.0,
            "delta_fr_hz_min": 1.0,
            "delta_fr_hz_max": 1.0,
        }
    )
    if "abs_delta_fr_hz_median" in row:
        row["abs_delta_fr_hz_median"] = 1.0
    if "Qi_min" in row:
        row.update({"Qi_min": 1.0, "Qi_max": 1.0})
    return row


def _shift_rows(
    runner: object,
    run_id: str,
    *,
    summary: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    summary = _summary_row(runner, run_id) if summary is None else summary
    rows: list[dict[str, object]] = []
    for index in sorted(spatial.EXPECTED_RESONATOR_INDICES):
        row: dict[str, object] = dict.fromkeys(runner.SHIFT_FIELDS, "1")
        row.update(
            {
                "run_id": run_id,
                "resonator_index": float(index),
                "resonator_label": spatial.EXPECTED_RESONATOR_LABELS[index],
            }
        )
        for field in set(runner.SUMMARY_FIELDS) & set(runner.SHIFT_FIELDS):
            if field not in {"run_id", "resonator_index"}:
                row[field] = summary[field]
        rows.append(row)
    return rows


def _write_artifacts(
    runner: object,
    out_dir: Path,
    run_id: str,
    *,
    nx: int = TEST_NX,
    total_time_ns: float = TEST_TOTAL_TIME_NS,
    xqp_values: list[float] | None = None,
) -> None:
    if xqp_values is None:
        xqp_values = [float(index + 1) for index in range(nx)]
    if len(xqp_values) != nx:
        raise ValueError("xqp_values must contain exactly nx entries")
    trace_rows = [dict.fromkeys(runner.TRACE_FIELDS, 0.0)]
    if total_time_ns != 0.0:
        final_trace = dict.fromkeys(runner.TRACE_FIELDS, 0.0)
        final_trace.update(
            {
                "t_ns": total_time_ns,
                "xqp_mean": float(np.mean(xqp_values)),
                "xqp_source": xqp_values[0],
                "xqp_open_end": xqp_values[-1],
            }
        )
        trace_rows.append(
            final_trace
        )
    _write_csv(
        out_dir / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        trace_rows,
    )
    _write_csv(
        out_dir / f"profile_{run_id}.csv",
        runner.PROFILE_FIELDS,
        [
            {
                "x_um": (index + 0.5) * spatial.LENGTH_UM / nx,
                "xqp": xqp_values[index],
            }
            for index in range(nx)
        ],
    )


def _write_valid_attempt(runner: object, out_dir: Path, run_id: str) -> None:
    _write_csv(
        out_dir / "summary.csv",
        runner.SUMMARY_FIELDS,
        [_summary_row(runner, run_id)],
    )
    _write_csv(
        out_dir / "resonator_shifts.csv",
        runner.SHIFT_FIELDS,
        _shift_rows(runner, run_id),
    )
    _write_artifacts(runner, out_dir, run_id)


def _campaign_run_ids(runner: object, config: object) -> list[str]:
    if runner is spatial:
        combinations = [
            (D0, rate, center_delta, sigma_delta)
            for D0 in config.D0_values
            for rate in config.source_rates_per_ns
            for center_delta in config.source_centers_delta
            for sigma_delta in config.source_sigmas_delta
        ]
    else:
        combinations = readout._combinations(config)
    return [runner._run_id(*combination, config) for combination in combinations]


def _assert_campaign_lock_available(out_dir: Path) -> None:
    handle = spatial.acquire_campaign_lock(
        out_dir,
        owner_label="test-probe",
        config_digest="test-probe-config",
    )
    spatial.release_campaign_lock(handle)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_requires_six_unique_resonator_rows(runner, tmp_path: Path) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    assert runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    ) == {run_id}

    _write_csv(
        tmp_path / "resonator_shifts.csv",
        runner.SHIFT_FIELDS,
        _shift_rows(runner, run_id)[:1],
    )
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    malformed = _shift_rows(runner, run_id)
    malformed[0]["f0_ghz"] = "garbage"
    _write_csv(tmp_path / "resonator_shifts.csv", runner.SHIFT_FIELDS, malformed)
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    malformed = _shift_rows(runner, run_id)
    malformed[0]["resonator_label"] = "wrong mode"
    _write_csv(tmp_path / "resonator_shifts.csv", runner.SHIFT_FIELDS, malformed)
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    duplicate = _shift_rows(runner, run_id)
    duplicate[-1]["resonator_index"] = duplicate[-2]["resonator_index"]
    _write_csv(tmp_path / "resonator_shifts.csv", runner.SHIFT_FIELDS, duplicate)
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_requires_valid_run_local_artifacts(runner, tmp_path: Path) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)

    trace = tmp_path / f"trace_{run_id}.csv"
    trace.write_text("", encoding="utf-8")
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    _write_csv(trace, ["stale_field"], [{"stale_field": 0.0}])
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    _write_csv(trace, runner.TRACE_FIELDS, [])
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    _write_artifacts(runner, tmp_path, run_id)
    profile = tmp_path / f"profile_{run_id}.csv"
    profile.unlink()
    profile.mkdir()
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    profile.rmdir()
    _write_artifacts(runner, tmp_path, run_id)
    summary = _summary_row(runner, run_id)
    summary["profile_csv"] = ""
    _write_csv(tmp_path / "summary.csv", runner.SUMMARY_FIELDS, [summary])
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir()
    outside = outside_dir / f"trace_{run_id}.csv"
    _write_csv(
        outside,
        runner.TRACE_FIELDS,
        [dict.fromkeys(runner.TRACE_FIELDS, 0.0)],
    )
    summary = _summary_row(runner, run_id)
    summary["trace_csv"] = f"../{outside_dir.name}/{outside.name}"
    _write_csv(tmp_path / "summary.csv", runner.SUMMARY_FIELDS, [summary])
    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_requires_trace_endpoint_matching_summary(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    completed_args = (
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )
    assert runner._completed_run_ids(*completed_args) == {run_id}

    one_row_at_endpoint = [
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS,
        }
    ]
    _write_csv(
        tmp_path / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        one_row_at_endpoint,
    )
    assert not runner._completed_run_ids(*completed_args)

    missing_initial_sample = [
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": 1.0,
        },
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS,
        },
    ]
    _write_csv(
        tmp_path / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        missing_initial_sample,
    )
    assert not runner._completed_run_ids(*completed_args)

    truncated_trace = [
        dict.fromkeys(runner.TRACE_FIELDS, 0.0),
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS - 1.0,
        },
    ]
    _write_csv(
        tmp_path / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        truncated_trace,
    )
    assert not runner._completed_run_ids(*completed_args)

    roundoff_offset = 0.5 * spatial._CSV_TIME_RTOL * TEST_TOTAL_TIME_NS
    endpoint_trace = [
        dict.fromkeys(runner.TRACE_FIELDS, 0.0),
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS + roundoff_offset,
            "xqp_mean": 2.0,
            "xqp_source": 1.0,
            "xqp_open_end": 3.0,
        },
    ]
    _write_csv(
        tmp_path / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        endpoint_trace,
    )
    assert runner._completed_run_ids(*completed_args) == {run_id}

    out_of_order_trace = [
        dict.fromkeys(runner.TRACE_FIELDS, 0.0),
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS + 1.0,
        },
        {
            **dict.fromkeys(runner.TRACE_FIELDS, 0.0),
            "t_ns": TEST_TOTAL_TIME_NS,
        },
    ]
    _write_csv(
        tmp_path / f"trace_{run_id}.csv",
        runner.TRACE_FIELDS,
        out_of_order_trace,
    )
    assert not runner._completed_run_ids(*completed_args)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_requires_exact_profile_nx_cardinality(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    completed_args = (
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )
    assert runner._completed_run_ids(*completed_args) == {run_id}

    profile_path = tmp_path / f"profile_{run_id}.csv"
    for row_count in (TEST_NX - 1, TEST_NX + 1):
        _write_csv(
            profile_path,
            runner.PROFILE_FIELDS,
            [
                {"x_um": float(index), "xqp": 1.0}
                for index in range(row_count)
            ],
        )
        assert not runner._completed_run_ids(*completed_args)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_requires_physical_profile_grid(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    completed_args = (
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )
    profile_path = tmp_path / f"profile_{run_id}.csv"

    _write_csv(
        profile_path,
        runner.PROFILE_FIELDS,
        [{"x_um": 0.0, "xqp": 1.0} for _ in range(TEST_NX)],
    )
    assert not runner._completed_run_ids(*completed_args)

    # rev5 and older stored endpoint samples.  They describe a different
    # control-volume measure and must never satisfy rev6 resume.
    _write_csv(
        profile_path,
        runner.PROFILE_FIELDS,
        [
            {
                "x_um": index * spatial.LENGTH_UM / (TEST_NX - 1),
                "xqp": 1.0,
            }
            for index in range(TEST_NX)
        ],
    )
    assert not runner._completed_run_ids(*completed_args)

    _write_artifacts(runner, tmp_path, run_id)
    rows = list(csv.DictReader(profile_path.open(encoding="utf-8")))
    rows[1]["xqp"] = "-1e-9"
    _write_csv(profile_path, runner.PROFILE_FIELDS, rows)
    assert not runner._completed_run_ids(*completed_args)

    _write_artifacts(runner, tmp_path, run_id)
    for invalid_nx in (0, 3.5, "not-an-integer"):
        _write_csv(
            tmp_path / "summary.csv",
            runner.SUMMARY_FIELDS,
            [_summary_row(runner, run_id, nx=invalid_nx)],
        )
        assert not runner._completed_run_ids(*completed_args)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_rechecks_config_residuals_and_cross_artifact_semantics(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    args = (
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )
    expected_rows = {run_id: {"NX": TEST_NX, "D0_um2_per_ns": 0.0}}
    kwargs = {"expected_rows": expected_rows, "stop_tol": 1e-6}
    assert runner._completed_run_ids(*args, **kwargs) == {run_id}

    summary = _summary_row(runner, run_id)
    summary["D0_um2_per_ns"] = 999.0
    _write_csv(tmp_path / "summary.csv", runner.SUMMARY_FIELDS, [summary])
    assert not runner._completed_run_ids(*args, **kwargs)

    _write_valid_attempt(runner, tmp_path, run_id)
    summary = _summary_row(runner, run_id)
    summary["final_max_dfdt_per_ns"] = 1e-3
    _write_csv(tmp_path / "summary.csv", runner.SUMMARY_FIELDS, [summary])
    trace_path = tmp_path / f"trace_{run_id}.csv"
    trace_rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    trace_rows[-1]["max_dfdt_per_ns"] = "1e-3"
    _write_csv(trace_path, runner.TRACE_FIELDS, trace_rows)
    assert not runner._completed_run_ids(*args, **kwargs)

    _write_valid_attempt(runner, tmp_path, run_id)
    trace_rows = list(csv.DictReader(trace_path.open(encoding="utf-8")))
    trace_rows[-1]["xqp_mean"] = "123"
    _write_csv(trace_path, runner.TRACE_FIELDS, trace_rows)
    assert not runner._completed_run_ids(*args, **kwargs)

    _write_valid_attempt(runner, tmp_path, run_id)
    profile_path = tmp_path / f"profile_{run_id}.csv"
    profile_rows = list(csv.DictReader(profile_path.open(encoding="utf-8")))
    profile_rows[1]["xqp"] = "123"
    _write_csv(profile_path, runner.PROFILE_FIELDS, profile_rows)
    assert not runner._completed_run_ids(*args, **kwargs)

    _write_valid_attempt(runner, tmp_path, run_id)
    shift_path = tmp_path / "resonator_shifts.csv"
    shifts = list(csv.DictReader(shift_path.open(encoding="utf-8")))
    shifts[0]["delta_fr_hz_current_weighted"] = "123"
    _write_csv(shift_path, runner.SHIFT_FIELDS, shifts)
    assert not runner._completed_run_ids(*args, **kwargs)


def test_readout_resume_binds_the_exact_grid_snap(tmp_path: Path) -> None:
    run_id = "run-a"
    _write_valid_attempt(readout, tmp_path, run_id)
    config = readout.SMOKE_CONFIG
    mode = config.readout_resonator_indices[0]
    resonator = readout.PRELIM_RESONATORS[mode - 1]
    used, harmonic, shift = readout._expected_readout_snap(config, mode)
    summary = _summary_row(readout, run_id)
    summary.update(
        {
            "readout_omega_uev": resonator.probe_energy_uev,
            "readout_omega_used_uev": used,
            "readout_omega_grid_harmonic": harmonic,
            "readout_omega_snap_rel_shift": shift,
        }
    )
    summary_path = tmp_path / "summary.csv"
    shifts_path = tmp_path / "resonator_shifts.csv"
    _write_csv(summary_path, readout.SUMMARY_FIELDS, [summary])
    expected = {
        run_id: {
            "readout_omega_uev": resonator.probe_energy_uev,
            "readout_omega_used_uev": used,
            "readout_omega_grid_harmonic": harmonic,
            "readout_omega_snap_rel_shift": shift,
        }
    }
    args = (summary_path, shifts_path, tmp_path)
    kwargs = {"expected_run_ids": {run_id}, "expected_rows": expected}
    assert readout._completed_run_ids(*args, **kwargs) == {run_id}

    # Keep the altered metadata finite and internally coherent: move to the
    # next grid harmonic and recompute its relative shift. The run id still
    # identifies the configured resonator/grid, so resume must reject it.
    dE = used / harmonic
    altered_used = (harmonic + 1) * dE
    summary["readout_omega_used_uev"] = altered_used
    summary["readout_omega_grid_harmonic"] = harmonic + 1
    summary["readout_omega_snap_rel_shift"] = abs(
        altered_used - resonator.probe_energy_uev
    ) / resonator.probe_energy_uev
    _write_csv(summary_path, readout.SUMMARY_FIELDS, [summary])
    assert not readout._completed_run_ids(*args, **kwargs)


def test_atomic_csv_write_preserves_previous_artifact_on_promotion_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "trace_run-a.csv"
    target.write_text("previous-complete-artifact\n", encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated interrupted promotion")

    monkeypatch.setattr(spatial.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted promotion"):
        spatial._atomic_write_csv(target, ["value"], [{"value": 1.0}])

    assert target.read_text(encoding="utf-8") == "previous-complete-artifact\n"
    assert not list(tmp_path.glob("*.tmp"))


def test_readout_trace_promotion_preserves_previous_artifact_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "trace_run-a.csv"
    work = target.with_suffix(target.suffix + ".tmp")
    target.write_text("previous-complete-artifact\n", encoding="utf-8")
    work.write_text("new-complete-artifact\n", encoding="utf-8")

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated interrupted trace promotion")

    monkeypatch.setattr(readout.os, "replace", fail_replace)
    with pytest.raises(OSError, match="interrupted trace promotion"):
        readout._promote_completed_trace(work, target)

    assert target.read_text(encoding="utf-8") == "previous-complete-artifact\n"
    assert not work.exists()


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_latest_summary_attempt_is_authoritative_and_malformed_is_safe(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    summary_path = tmp_path / "summary.csv"

    _write_csv(
        summary_path,
        runner.SUMMARY_FIELDS,
        [
            _summary_row(runner, run_id),
            _summary_row(runner, run_id, status="failed", converged=False),
        ],
    )
    assert not runner._completed_run_ids(
        summary_path,
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )
    _write_csv(
        summary_path,
        runner.SUMMARY_FIELDS,
        [{**_summary_row(runner, run_id), "n_steps": "garbage"}],
    )
    assert not runner._completed_run_ids(
        summary_path,
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    _write_csv(
        summary_path,
        runner.SUMMARY_FIELDS,
        [
            _summary_row(runner, run_id, status="failed", converged=False),
            _summary_row(runner, run_id),
        ],
    )
    assert runner._completed_run_ids(
        summary_path,
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    ) == {run_id}

    with summary_path.open("a", encoding="utf-8") as fp:
        fp.write(f"{run_id},completed\n")
    assert not runner._completed_run_ids(
        summary_path,
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    )

    runner.purge_run_id_rows(summary_path, run_id)
    with summary_path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=runner.SUMMARY_FIELDS)
        writer.writerow(_summary_row(runner, run_id))
    assert runner._completed_run_ids(
        summary_path,
        tmp_path / "resonator_shifts.csv",
        tmp_path,
    ) == {run_id}

    # A malformed unknown id is ambiguous: fail loud instead of either
    # falling back to an older completion or entering an unhealable rerun.
    with summary_path.open("a", encoding="utf-8") as fp:
        fp.write("orphan-corrupt-row\n")
    with pytest.raises(SystemExit, match="unknown run_id"):
        runner._completed_run_ids(
            summary_path,
            tmp_path / "resonator_shifts.csv",
            tmp_path,
        )

    _write_csv(summary_path, runner.SUMMARY_FIELDS, [_summary_row(runner, run_id)])
    with summary_path.open("a", encoding="utf-8") as fp:
        fp.write("../bad\n")
    with pytest.raises(SystemExit, match="invalid run_id"):
        runner._completed_run_ids(
            summary_path,
            tmp_path / "resonator_shifts.csv",
            tmp_path,
        )

    # A malformed row with no run id cannot be repaired by per-id purge, so
    # resume must fail loud instead of looping over every campaign case.
    _write_csv(summary_path, runner.SUMMARY_FIELDS, [_summary_row(runner, run_id)])
    with summary_path.open("a", encoding="utf-8") as fp:
        fp.write(",completed\n")
    with pytest.raises(SystemExit, match="no run_id"):
        runner._completed_run_ids(
            summary_path,
            tmp_path / "resonator_shifts.csv",
            tmp_path,
        )


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_resume_rejects_elapsed_time_step_count_mismatch(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    summary = _summary_row(runner, run_id)
    summary["n_steps"] = 999
    _write_csv(
        tmp_path / "summary.csv",
        runner.SUMMARY_FIELDS,
        [summary],
    )
    _write_csv(
        tmp_path / "resonator_shifts.csv",
        runner.SHIFT_FIELDS,
        _shift_rows(runner, run_id, summary=summary),
    )
    _write_artifacts(runner, tmp_path, run_id)

    assert not runner._completed_run_ids(
        tmp_path / "summary.csv",
        tmp_path / "resonator_shifts.csv",
        tmp_path,
        expected_run_ids={run_id},
        expected_rows={run_id: {"NX": TEST_NX, "dt_ns": 0.1}},
        stop_tol=1.0,
    )


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_noncanonical_run_id_fails_loud_instead_of_poisoning_retry(
    runner,
    tmp_path: Path,
) -> None:
    run_id = "run-a"
    _write_valid_attempt(runner, tmp_path, run_id)
    shifts = _shift_rows(runner, run_id)
    shifts[0]["run_id"] = f" {run_id} "
    _write_csv(tmp_path / "resonator_shifts.csv", runner.SHIFT_FIELDS, shifts)

    with pytest.raises(SystemExit, match="non-canonical run_id"):
        runner._completed_run_ids(
            tmp_path / "summary.csv",
            tmp_path / "resonator_shifts.csv",
            tmp_path,
        )


def test_readout_validates_shift_schema_before_skipping(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy_metadata = tmp_path / "metadata.json"
    legacy_metadata.write_text("do not replace", encoding="utf-8")
    _write_csv(tmp_path / "summary.csv", readout.SUMMARY_FIELDS, [])
    (tmp_path / "resonator_shifts.csv").write_text(
        "run_id,legacy_only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        readout,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=False,
        ),
    )

    with pytest.raises(SystemExit, match="--no-resume"):
        readout.main()
    assert legacy_metadata.read_text(encoding="utf-8") == "do not replace"
    assert not list(tmp_path.glob("metadata_rev*.json"))


def test_spatial_rejected_resume_does_not_mutate_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    legacy_metadata = tmp_path / "metadata.json"
    legacy_metadata.write_text("do not replace", encoding="utf-8")
    (tmp_path / "summary.csv").write_text(
        "run_id,legacy_only\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        spatial,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=False,
        ),
    )

    with pytest.raises(SystemExit, match="--no-resume"):
        spatial.main()
    assert legacy_metadata.read_text(encoding="utf-8") == "do not replace"
    assert not list(tmp_path.glob("metadata_rev*.json"))


def test_readout_partial_attempt_retries_once_then_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = readout.SMOKE_CONFIG
    D0 = config.D0_values[0]
    rate = config.source_rates_per_ns[0]
    center = config.source_centers_delta[0]
    sigma = config.source_sigmas_delta[0]
    tau_l = config.tau_l_values_ns[0]
    n_bar = config.n_bar_values[0]
    mode = config.readout_resonator_indices[0]
    run_id = readout._run_id(D0, rate, center, sigma, tau_l, n_bar, mode, config)
    _write_valid_attempt(readout, tmp_path, run_id)
    _write_csv(
        tmp_path / "resonator_shifts.csv",
        readout.SHIFT_FIELDS,
        _shift_rows(readout, run_id)[:1],
    )

    args = argparse.Namespace(
        preset="smoke",
        out_dir=tmp_path,
        max_runs=1,
        wall_hours=None,
        no_resume=False,
    )
    monkeypatch.setattr(readout, "_parse_args", lambda: args)
    calls = 0

    def fake_run_case(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        total_time_ns = config.max_time_ns
        xqp_values = [float(index + 1) for index in range(config.NX)]
        _write_artifacts(
            readout,
            tmp_path,
            run_id,
            nx=config.NX,
            total_time_ns=total_time_ns,
            xqp_values=xqp_values,
        )
        summary = _summary_row(
            readout,
            run_id,
            nx=config.NX,
            total_time_ns=total_time_ns,
        )
        resonator = readout.PRELIM_RESONATORS[mode - 1]
        omega_used, omega_harmonic, omega_shift = readout._expected_readout_snap(
            config,
            mode,
        )
        summary.update(
            {
                "T_bath_K": readout.T_BATH_K,
                "D0_um2_per_ns": D0,
                "tau_l_ns": tau_l,
                "source_rate_per_ns": rate,
                **readout._source_calibration(config, rate),
                "source_center_delta": center,
                "source_sigma_delta": sigma,
                "readout_resonator_index": mode,
                "readout_frequency_ghz": resonator.frequency_ghz,
                "readout_omega_uev": resonator.probe_energy_uev,
                "readout_omega_used_uev": omega_used,
                "readout_omega_grid_harmonic": omega_harmonic,
                "readout_omega_snap_rel_shift": omega_shift,
                "readout_n_bar_peak": n_bar,
                "readout_c_phot_ns_inv": readout.C_PHOT_NS_INV,
                "dt_ns": config.dt_ns,
                "max_time_ns": config.max_time_ns,
                "n_steps": round(total_time_ns / config.dt_ns),
                "xqp_mean": float(np.mean(xqp_values)),
                "xqp_source": xqp_values[0],
                "xqp_open_end": xqp_values[-1],
                "delta_fr_hz_min": 1.0,
                "delta_fr_hz_max": 1.0,
                "Qi_min": 1.0,
                "Qi_max": 1.0,
                "wall_seconds": 0.0,
            }
        )
        return summary, _shift_rows(readout, run_id, summary=summary)

    monkeypatch.setattr(readout, "_run_case", fake_run_case)
    readout.main()
    assert calls == 1
    assert len(list(csv.DictReader((tmp_path / "summary.csv").open()))) == 1
    assert len(list(csv.DictReader((tmp_path / "resonator_shifts.csv").open()))) == 6

    readout.main()
    assert calls == 1


def test_readout_run_case_commits_true_final_trace_and_nx_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        readout.SMOKE_CONFIG,
        NX=TEST_NX,
        dt_ns=1.0,
        max_time_ns=5.0,
        stop_tol=0.1,
        snapshot_interval_ns=10.0,
        source_centers_delta=(2.0, 2.1),
        source_sigmas_delta=(0.08, 0.09),
    )
    state = SimpleNamespace(
        f=np.zeros((2, config.NX)),
        geometry=SimpleNamespace(mesh_size=1.0),
        spectral=SimpleNamespace(dE=np.array([1.0])),
    )
    expected_run_id = readout._run_id(
        config.D0_values[0],
        config.source_rates_per_ns[0],
        config.source_centers_delta[1],
        config.source_sigmas_delta[1],
        config.tau_l_values_ns[0],
        0.0,
        config.readout_resonator_indices[0],
        config,
    )

    class FakeRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            self.n_ph = np.zeros(2)
            self.n_steps = 0

        def step(self, current, *_args, **_kwargs):
            self.n_steps += 1
            assert not (tmp_path / f"trace_{expected_run_id}.csv").exists()
            assert (tmp_path / f"trace_{expected_run_id}.csv.tmp").exists()
            residual = 1.0 if self.n_steps == 1 else 0.0
            return current, residual, residual

    monkeypatch.setattr(readout, "_build_state", lambda *_args: state)
    source_shapes: list[tuple[float, float]] = []

    def fake_source_flux(*_args, center_delta, sigma_delta, **_kwargs):
        source_shapes.append((center_delta, sigma_delta))
        return object()

    monkeypatch.setattr(readout, "_source_flux", fake_source_flux)
    monkeypatch.setattr(readout, "snap_omega_to_grid", lambda *_args: (1.0, 1, 0.0))
    monkeypatch.setattr(readout, "FinitePhononSpatialRunner", FakeRunner)
    monkeypatch.setattr(readout, "_xqp_profile", lambda _state: np.ones(config.NX))
    monkeypatch.setattr(
        readout,
        "_source_calibration",
        lambda *_args: {
            "source_cell_volume_um3": 1.0,
            "qps_per_xqp_source_cell": 1.0,
            "estimated_source_qp_per_s": 1.0,
        },
    )
    monkeypatch.setattr(
        readout,
        "_resonator_shifts",
        lambda *_args: [
            {
                "delta_fr_hz_current_weighted": float(index),
                "Qi_current_weighted": float(index + 1),
            }
            for index in range(6)
        ],
    )

    summary, _shifts = readout._run_case(
        config,
        tmp_path,
        D0=config.D0_values[0],
        source_rate=config.source_rates_per_ns[0],
        source_center_delta=config.source_centers_delta[1],
        source_sigma_delta=config.source_sigmas_delta[1],
        tau_l_ns=config.tau_l_values_ns[0],
        n_bar=0.0,
        readout_index=config.readout_resonator_indices[0],
    )

    trace_path = tmp_path / str(summary["trace_csv"])
    with trace_path.open(newline="", encoding="utf-8") as fp:
        trace_times = [float(row["t_ns"]) for row in csv.DictReader(fp)]
    assert trace_times == [0.0, 1.0, 2.0]
    assert trace_times[-1] == summary["total_time_ns"]
    assert not (tmp_path / f"trace_{expected_run_id}.csv.tmp").exists()

    profile_path = tmp_path / str(summary["profile_csv"])
    with profile_path.open(newline="", encoding="utf-8") as fp:
        assert len(list(csv.DictReader(fp))) == config.NX == summary["NX"]
    assert source_shapes == [
        (config.source_centers_delta[1], config.source_sigmas_delta[1])
    ]
    assert summary["source_center_delta"] == config.source_centers_delta[1]
    assert summary["source_sigma_delta"] == config.source_sigmas_delta[1]
    assert all(
        row["source_center_delta"] == config.source_centers_delta[1]
        and row["source_sigma_delta"] == config.source_sigmas_delta[1]
        for row in _shifts
    )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("dt_ns", 0.0),
        ("dt_ns", -1.0),
        ("dt_ns", np.nan),
        ("dt_ns", np.inf),
        ("max_time_ns", -1.0),
        ("max_time_ns", np.nan),
        ("max_time_ns", np.inf),
    ],
)
def test_readout_run_case_rejects_invalid_time_controls_before_building_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field_name: str,
    bad_value: float,
) -> None:
    config = replace(readout.SMOKE_CONFIG, **{field_name: bad_value})

    def unexpected_build(*_args, **_kwargs):
        raise AssertionError("invalid controls must fail before state construction")

    monkeypatch.setattr(readout, "_build_state", unexpected_build)
    with pytest.raises(ValueError, match=field_name):
        readout._run_case(
            config,
            tmp_path,
            D0=config.D0_values[0],
            source_rate=config.source_rates_per_ns[0],
            source_center_delta=config.source_centers_delta[0],
            source_sigma_delta=config.source_sigmas_delta[0],
            tau_l_ns=config.tau_l_values_ns[0],
            n_bar=0.0,
            readout_index=config.readout_resonator_indices[0],
        )


def test_readout_run_case_shortens_nondivisible_final_step_and_resumes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The integrator must land exactly on a non-integral time horizon."""
    config = replace(
        readout.SMOKE_CONFIG,
        NX=TEST_NX,
        dt_ns=1.0,
        max_time_ns=2.5,
        stop_tol=0.1,
        snapshot_interval_ns=10.0,
    )
    cell_centers = (
        np.arange(config.NX, dtype=float) + 0.5
    ) * spatial.LENGTH_UM / config.NX
    state = SimpleNamespace(
        f=np.zeros((2, config.NX)),
        geometry=SimpleNamespace(mesh_size=float(cell_centers[1] - cell_centers[0])),
        spectral=SimpleNamespace(dE=np.array([1.0])),
    )
    step_sizes: list[float] = []

    class FakeRunner:
        def __init__(self, *_args, **_kwargs) -> None:
            self.n_ph = np.zeros(2)

        def step(self, current, dt_ns, *_args, **_kwargs):
            step_sizes.append(float(dt_ns))
            # Stay unconverged until the shortened horizon-closing step.
            residual = 0.0 if len(step_sizes) == 3 else 1.0
            return current, residual, residual

    monkeypatch.setattr(readout, "_build_state", lambda *_args: state)
    monkeypatch.setattr(readout, "_source_flux", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(readout, "snap_omega_to_grid", lambda *_args: (1.0, 1, 0.0))
    monkeypatch.setattr(readout, "FinitePhononSpatialRunner", FakeRunner)
    monkeypatch.setattr(readout, "_xqp_profile", lambda _state: np.ones(config.NX))
    monkeypatch.setattr(
        readout,
        "_source_calibration",
        lambda *_args: {
            "source_cell_volume_um3": 1.0,
            "qps_per_xqp_source_cell": 1.0,
            "estimated_source_qp_per_s": 1.0,
        },
    )
    monkeypatch.setattr(
        readout,
        "_resonator_shifts",
        lambda *_args: [
            {
                "delta_fr_hz_current_weighted": 1.0,
                "Qi_current_weighted": 1.0,
            }
            for _index in range(6)
        ],
    )

    summary, _ = readout._run_case(
        config,
        tmp_path,
        D0=config.D0_values[0],
        source_rate=config.source_rates_per_ns[0],
        source_center_delta=config.source_centers_delta[0],
        source_sigma_delta=config.source_sigmas_delta[0],
        tau_l_ns=config.tau_l_values_ns[0],
        n_bar=0.0,
        readout_index=config.readout_resonator_indices[0],
    )

    assert step_sizes == [1.0, 1.0, 0.5]
    assert summary["total_time_ns"] == 2.5
    assert summary["n_steps"] == 3
    assert summary["status"] == "completed"
    run_id = str(summary["run_id"])
    trace_path = tmp_path / str(summary["trace_csv"])
    with trace_path.open(newline="", encoding="utf-8") as fp:
        trace_times = [float(row["t_ns"]) for row in csv.DictReader(fp)]
    assert trace_times == [0.0, 1.0, 2.5]

    summary_path = tmp_path / "summary.csv"
    shifts_path = tmp_path / "resonator_shifts.csv"
    _write_csv(summary_path, readout.SUMMARY_FIELDS, [summary])
    _write_csv(
        shifts_path,
        readout.SHIFT_FIELDS,
        _shift_rows(readout, run_id, summary=summary),
    )
    assert readout._completed_run_ids(
        summary_path,
        shifts_path,
        tmp_path,
        expected_run_ids={run_id},
        expected_rows={
            run_id: {
                "NX": config.NX,
                "dt_ns": config.dt_ns,
                "max_time_ns": config.max_time_ns,
            }
        },
        stop_tol=config.stop_tol,
    ) == {run_id}


def test_spatial_partial_attempt_retries_once_then_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = spatial.SMOKE_CONFIG
    D0 = config.D0_values[0]
    rate = config.source_rates_per_ns[0]
    center = config.source_centers_delta[0]
    sigma = config.source_sigmas_delta[0]
    run_id = spatial._run_id(D0, rate, center, sigma, config)
    _write_valid_attempt(spatial, tmp_path, run_id)
    _write_csv(
        tmp_path / "resonator_shifts.csv",
        spatial.SHIFT_FIELDS,
        _shift_rows(spatial, run_id)[:1],
    )

    args = argparse.Namespace(
        preset="smoke",
        out_dir=tmp_path,
        max_runs=1,
        wall_hours=None,
        no_resume=False,
    )
    monkeypatch.setattr(spatial, "_parse_args", lambda: args)
    builds = 0

    def fake_build_state(*_args, **_kwargs):
        nonlocal builds
        builds += 1
        return object()

    snapshot = SimpleNamespace(
        observables={"xqp_mean": 0.0, "xqp_source": 0.0, "xqp_open_end": 0.0},
        max_rate=0.0,
    )
    result = SimpleNamespace(
        converged=True,
        total_time=2.0,
        n_steps=1,
        state=object(),
        snapshots=[snapshot],
    )

    class FakeBackend:
        def run_until_steady_state(self, *_args, **_kwargs):
            return result

    monkeypatch.setattr(spatial, "_build_state", fake_build_state)
    monkeypatch.setattr(spatial, "_mean_f", lambda _state: object())
    monkeypatch.setattr(spatial, "_source_flux", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(spatial, "T3SpatialBackend", FakeBackend)
    monkeypatch.setattr(
        spatial,
        "_write_trace",
        lambda path, _snapshots: _write_csv(
            path,
            spatial.TRACE_FIELDS,
            [
                dict.fromkeys(spatial.TRACE_FIELDS, 0.0),
                {
                    **dict.fromkeys(spatial.TRACE_FIELDS, 0.0),
                    "t_ns": result.total_time,
                },
            ],
        ),
    )
    monkeypatch.setattr(
        spatial,
        "_write_profile",
        lambda path, _state: _write_csv(
            path,
            spatial.PROFILE_FIELDS,
            [
                {
                    "x_um": (index + 0.5) * spatial.LENGTH_UM / config.NX,
                    "xqp": 0.0,
                }
                for index in range(config.NX)
            ],
        ),
    )

    def fake_shifts(*_args, **_kwargs):
        rows = []
        for index in sorted(spatial.EXPECTED_RESONATOR_INDICES):
            row: dict[str, object] = dict.fromkeys(spatial.RESONATOR_SHIFT_FIELDS, 0.0)
            row["resonator_index"] = float(index)
            row["resonator_label"] = spatial.EXPECTED_RESONATOR_LABELS[index]
            rows.append(row)
        return rows

    monkeypatch.setattr(spatial, "_resonator_shifts", fake_shifts)
    spatial.main()
    assert builds == 1
    assert len(list(csv.DictReader((tmp_path / "summary.csv").open()))) == 1
    assert len(list(csv.DictReader((tmp_path / "resonator_shifts.csv").open()))) == 6

    spatial.main()
    assert builds == 1


def test_spatial_write_trace_normalizes_backend_initial_inf(tmp_path: Path) -> None:
    observables = {
        "xqp_mean": 1.0,
        "xqp_source": 2.0,
        "xqp_open_end": 3.0,
    }
    snapshots = [
        SimpleNamespace(t=0.0, max_rate=math.inf, observables=observables),
        SimpleNamespace(t=1.0, max_rate=4.0, observables=observables),
    ]
    path = tmp_path / "trace.csv"

    spatial._write_trace(path, snapshots)

    with path.open(newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    assert float(rows[0]["max_dfdt_per_ns"]) == 0.0
    assert all(
        math.isfinite(float(row[field]))
        for row in rows
        for field in spatial.TRACE_FIELDS
    )


@pytest.mark.parametrize(
    ("t_ns", "max_rate", "observable_updates"),
    (
        (1.0, math.inf, {}),
        (0.0, -math.inf, {}),
        (math.nan, 0.0, {}),
        (0.0, math.inf, {"xqp_mean": math.nan}),
        (0.0, 0.0, {"xqp_open_end": math.inf}),
    ),
    ids=(
        "noninitial-positive-inf",
        "negative-inf",
        "nan-time",
        "nan-observable",
        "inf-observable",
    ),
)
def test_spatial_write_trace_rejects_other_nonfinite_values(
    tmp_path: Path,
    t_ns: float,
    max_rate: float,
    observable_updates: dict[str, float],
) -> None:
    observables = {
        "xqp_mean": 1.0,
        "xqp_source": 2.0,
        "xqp_open_end": 3.0,
        **observable_updates,
    }
    path = tmp_path / "trace.csv"
    snapshot = SimpleNamespace(t=t_ns, max_rate=max_rate, observables=observables)

    with pytest.raises(ValueError, match="non-finite"):
        spatial._write_trace(path, [snapshot])
    assert not path.exists()


def test_spatial_run_id_separates_numerical_campaign_configs() -> None:
    config = spatial.SMOKE_CONFIG
    args = (
        config.D0_values[0],
        config.source_rates_per_ns[0],
        config.source_centers_delta[0],
        config.source_sigmas_delta[0],
    )
    reference = spatial._run_id(*args, config)

    for changed in (
        replace(config, NX=config.NX + 2),
        replace(config, NE=config.NE + 2),
        replace(config, dt_ns=config.dt_ns / 2.0),
        replace(config, max_time_ns=config.max_time_ns * 2.0),
        replace(config, stop_tol=config.stop_tol / 10.0),
        replace(config, snapshot_interval_ns=config.snapshot_interval_ns / 2.0),
    ):
        assert spatial._run_id(*args, changed) != reference


def test_spatial_run_id_uses_exact_point_digest() -> None:
    config = spatial.SMOKE_CONFIG
    rate = config.source_rates_per_ns[0]
    adjacent_rate = float(np.nextafter(rate, math.inf))
    args = (
        config.D0_values[0],
        rate,
        config.source_centers_delta[0],
        config.source_sigmas_delta[0],
    )
    adjacent_args = (args[0], adjacent_rate, args[2], args[3])

    assert f"{rate:.0e}" == f"{adjacent_rate:.0e}"
    assert spatial._run_id(*args, config) != spatial._run_id(*adjacent_args, config)
    assert spatial._point_digest(*args)[:20] in spatial._run_id(*args, config)


def test_readout_run_id_separates_source_shape_and_config() -> None:
    config = readout.SMOKE_CONFIG
    args = (
        config.D0_values[0],
        config.source_rates_per_ns[0],
        config.source_centers_delta[0],
        config.source_sigmas_delta[0],
        config.tau_l_values_ns[0],
        config.n_bar_values[0],
        config.readout_resonator_indices[0],
    )
    reference = readout._run_id(*args, config)

    for changed in (
        replace(config, source_centers_delta=(2.1,)),
        replace(config, source_sigmas_delta=(0.09,)),
        replace(config, snapshot_interval_ns=config.snapshot_interval_ns / 2.0),
    ):
        assert readout._run_id(*args, changed) != reference


def test_readout_run_id_uses_exact_point_digest() -> None:
    config = readout.SMOKE_CONFIG
    rate = config.source_rates_per_ns[0]
    adjacent_rate = float(np.nextafter(rate, math.inf))
    args = (
        config.D0_values[0],
        rate,
        config.source_centers_delta[0],
        config.source_sigmas_delta[0],
        config.tau_l_values_ns[0],
        config.n_bar_values[0],
        config.readout_resonator_indices[0],
    )
    adjacent_args = (args[0], adjacent_rate, *args[2:])

    assert f"{rate:.0e}" == f"{adjacent_rate:.0e}"
    assert readout._run_id(*args, config) != readout._run_id(*adjacent_args, config)
    assert readout._point_digest(*args)[:20] in readout._run_id(*args, config)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_main_rejects_generated_run_id_collisions(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(runner.SMOKE_CONFIG, D0_values=(1.0, 2.0))
    monkeypatch.setattr(runner, "SMOKE_CONFIG", config)
    monkeypatch.setattr(runner, "_run_id", lambda *_args, **_kwargs: "collision")
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=None,
            wall_hours=None,
            no_resume=True,
        ),
    )

    with pytest.raises(RuntimeError, match="duplicate run ids"):
        runner.main()
    assert not list(tmp_path.glob("metadata_rev*.json"))


def test_readout_source_shape_is_a_sweep_dimension_and_failure_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = replace(
        readout.SMOKE_CONFIG,
        source_centers_delta=(1.9, 2.1),
        source_sigmas_delta=(0.05, 0.09),
        n_bar_values=(0.0,),
    )
    combinations = readout._combinations(config)
    expected_shapes = {(1.9, 0.05), (1.9, 0.09), (2.1, 0.05), (2.1, 0.09)}
    assert {(point[2], point[3]) for point in combinations} == expected_shapes
    assert len(combinations) == 4

    monkeypatch.setattr(readout, "SMOKE_CONFIG", config)
    monkeypatch.setattr(readout, "_source_calibration", lambda *_args: {})
    monkeypatch.setattr(
        readout,
        "_run_case",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        readout,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=None,
            wall_hours=None,
            no_resume=True,
        ),
    )

    readout.main()

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as fp:
        failure_rows = list(csv.DictReader(fp))
    assert len(failure_rows) == 4
    assert {
        (float(row["source_center_delta"]), float(row["source_sigma_delta"]))
        for row in failure_rows
    } == expected_shapes
    assert all(row["status"] == "failed" for row in failure_rows)
    assert len({row["run_id"] for row in failure_rows}) == 4


@pytest.mark.parametrize(
    ("runner", "constant_name"),
    (
        (spatial, "T_BATH_K"),
        (readout, "T_BATH_K"),
        (readout, "C_PHOT_NS_INV"),
    ),
    ids=("spatial-bath", "readout-bath", "readout-photon-coupling"),
)
def test_config_digest_includes_module_physics_constants(
    runner,
    constant_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.SMOKE_CONFIG
    reference = runner._config_digest(config)
    monkeypatch.setattr(runner, constant_name, getattr(runner, constant_name) * 1.5)

    assert runner._config_digest(config) != reference


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_config_digest_includes_source_material_code_digest(
    runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = runner.SMOKE_CONFIG
    reference = runner._config_digest(config)
    monkeypatch.setattr(
        runner,
        "_source_material_code_digest",
        lambda: "changed-source-material-code",
    )

    assert runner._config_digest(config) != reference


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_config_digest_includes_numerical_runtime(
    runner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = runner._config_digest(runner.SMOKE_CONFIG)
    monkeypatch.setattr(np, "__version__", f"{np.__version__}-different-runtime")

    assert runner._config_digest(runner.SMOKE_CONFIG) != reference


@pytest.mark.parametrize(
    "variable",
    (
        "BLIS_NUM_THREADS",
        "MKL_CBWR",
        "MKL_DYNAMIC",
        "OPENBLAS_CORETYPE",
        "OMP_DYNAMIC",
    ),
)
def test_config_digest_includes_extended_numerical_environment(
    variable: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(variable, raising=False)
    reference = spatial._config_digest(spatial.SMOKE_CONFIG)
    monkeypatch.setenv(variable, "audit-different")

    assert spatial._config_digest(spatial.SMOKE_CONFIG) != reference


@pytest.mark.parametrize(
    ("writer", "rows"),
    (
        (spatial._append_csv, {"a": 1, "extra": 2}),
        (readout._append_rows, [{"a": 1, "extra": 2}]),
    ),
    ids=("spatial", "readout"),
)
def test_aggregate_writers_reject_unknown_fields(
    writer,
    rows,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="fields not in fieldnames"):
        if writer is spatial._append_csv:
            writer(tmp_path / "aggregate.csv", rows, ["a"])
        else:
            writer(tmp_path / "aggregate.csv", rows, ["a"])


def test_fresh_start_purges_only_exact_current_run_artifacts(
    tmp_path: Path,
) -> None:
    run_ids = {"run-a", "run-b"}
    owned = [
        tmp_path / f"{prefix}_{run_id}.csv"
        for run_id in run_ids
        for prefix in ("trace", "profile")
    ]
    owned += [
        tmp_path / f"trace_{run_id}.csv.tmp"
        for run_id in run_ids
    ]
    owned += [
        tmp_path / f".profile_{run_id}.csv.interrupted.tmp"
        for run_id in run_ids
    ]
    unrelated = tmp_path / "trace_other-run.csv"
    for path in [*owned, unrelated]:
        path.write_text("stale", encoding="utf-8")

    spatial.purge_run_artifacts(tmp_path, run_ids)

    assert not any(path.exists() for path in owned)
    assert unrelated.read_text(encoding="utf-8") == "stale"


def test_artifact_purge_validates_all_ids_before_any_unlink(
    tmp_path: Path,
) -> None:
    valid_artifacts = [
        tmp_path / "trace_valid.csv",
        tmp_path / "profile_valid.csv",
    ]
    glob_target = tmp_path / ".trace_bad-target.csv.interrupted.tmp"
    for path in [*valid_artifacts, glob_target]:
        path.write_text("must survive", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid campaign run id"):
        spatial.purge_run_artifacts(tmp_path, ["valid", "bad*"])

    for path in [*valid_artifacts, glob_target]:
        assert path.read_text(encoding="utf-8") == "must survive"


def test_source_material_code_digest_is_cached() -> None:
    digest_function = spatial._source_material_code_digest
    digest_function.cache_clear()

    first = digest_function()
    after_first = digest_function.cache_info()
    second = digest_function()
    after_second = digest_function.cache_info()

    assert first == second
    assert len(first) == 64
    assert after_first.misses == 1
    assert after_second.hits == after_first.hits + 1


@pytest.mark.parametrize(
    ("runner", "config"),
    ((spatial, spatial.SMOKE_CONFIG), (readout, readout.SMOKE_CONFIG)),
    ids=("spatial", "readout"),
)
def test_metadata_is_atomic_config_addressed_and_coexists(
    runner,
    config,
    tmp_path: Path,
) -> None:
    first = runner._write_metadata(tmp_path, config)
    changed = replace(config, name=f"{config.name}-variant", NX=config.NX + 2)
    second = runner._write_metadata(tmp_path, changed)

    assert first != second
    assert first.is_file() and second.is_file()
    assert not (tmp_path / "metadata.json").exists()
    for path, expected in ((first, config), (second, changed)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["physics_revision"] == runner._PHYSICS_REV
        assert payload["config_sha256"] == runner._config_digest(expected)
        assert payload["config"]["NX"] == expected.NX
        assert payload["physics_constants"] == runner._physics_constants()
        assert (
            payload["source_material_code_sha256"]
            == runner._source_material_code_digest()
        )


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_no_resume_max_runs_restarts_only_selected_ids(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = runner.SMOKE_CONFIG
    run_ids = _campaign_run_ids(runner, config)
    assert len(run_ids) >= 2
    selected, unselected = run_ids[:2]
    foreign = "foreign-config-run"
    initial_ids = (selected, unselected, foreign)
    _write_csv(
        tmp_path / "summary.csv",
        runner.SUMMARY_FIELDS,
        [_summary_row(runner, run_id) for run_id in initial_ids],
    )
    _write_csv(
        tmp_path / "resonator_shifts.csv",
        runner.SHIFT_FIELDS,
        [
            row
            for run_id in initial_ids
            for row in _shift_rows(runner, run_id)
        ],
    )
    for run_id in initial_ids:
        _write_artifacts(runner, tmp_path, run_id)

    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=True,
        ),
    )
    if runner is spatial:
        monkeypatch.setattr(
            spatial,
            "_build_state",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("selected retry stopped")
            ),
        )
    else:
        monkeypatch.setattr(
            readout,
            "_run_case",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("selected retry stopped")
            ),
        )

    runner.main()

    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as fp:
        summary_rows = list(csv.DictReader(fp))
    by_id = {row["run_id"]: row for row in summary_rows}
    assert set(by_id) == set(initial_ids)
    assert by_id[selected]["status"] == "failed"
    assert by_id[unselected]["status"] == "completed"
    assert by_id[foreign]["status"] == "completed"

    with (tmp_path / "resonator_shifts.csv").open(
        newline="", encoding="utf-8"
    ) as fp:
        shift_ids = {row["run_id"] for row in csv.DictReader(fp)}
    assert shift_ids == {unselected, foreign}

    for prefix in ("trace", "profile"):
        assert not (tmp_path / f"{prefix}_{selected}.csv").exists()
        assert (tmp_path / f"{prefix}_{unselected}.csv").is_file()
        assert (tmp_path / f"{prefix}_{foreign}.csv").is_file()


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_no_resume_rejects_mismatched_header_without_mutation(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = runner.SMOKE_CONFIG
    selected = _campaign_run_ids(runner, config)[0]
    summary_path = tmp_path / "summary.csv"
    shifts_path = tmp_path / "resonator_shifts.csv"
    summary_path.write_bytes(b"run_id,stale_field\r\nsentinel,keep\r\n")
    _write_csv(
        shifts_path,
        runner.SHIFT_FIELDS,
        _shift_rows(runner, "sentinel"),
    )
    _write_artifacts(runner, tmp_path, selected)
    before_summary = summary_path.read_bytes()
    before_shifts = shifts_path.read_bytes()
    before_artifacts = {
        path.name: path.read_bytes()
        for path in tmp_path.glob(f"*_{selected}.csv")
        if path not in {summary_path, shifts_path}
    }
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=True,
        ),
    )

    with pytest.raises(SystemExit, match="header"):
        runner.main()

    assert summary_path.read_bytes() == before_summary
    assert shifts_path.read_bytes() == before_shifts
    assert {
        path.name: path.read_bytes()
        for path in tmp_path.glob(f"*_{selected}.csv")
        if path not in {summary_path, shifts_path}
    } == before_artifacts
    assert not list(tmp_path.glob("metadata_rev*.json"))
    _assert_campaign_lock_available(tmp_path)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_no_resume_rejects_unattributable_rows_before_any_mutation(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = runner.SMOKE_CONFIG
    selected = _campaign_run_ids(runner, config)[0]
    summary_path = tmp_path / "summary.csv"
    shifts_path = tmp_path / "resonator_shifts.csv"
    _write_csv(
        summary_path,
        runner.SUMMARY_FIELDS,
        [_summary_row(runner, selected)],
    )
    corrupt_shifts = _shift_rows(runner, selected)
    corrupt_shifts[-1]["run_id"] = f" {selected} "
    _write_csv(shifts_path, runner.SHIFT_FIELDS, corrupt_shifts)
    _write_artifacts(runner, tmp_path, selected)
    before = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
    }
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=True,
        ),
    )

    with pytest.raises(SystemExit, match="non-canonical run_id"):
        runner.main()

    after = {
        path.name: path.read_bytes()
        for path in tmp_path.iterdir()
        if path.is_file()
        and path.name
        not in {
            spatial._CAMPAIGN_LOCK_FILENAME,
            spatial._CAMPAIGN_LOCK_OWNER_FILENAME,
        }
    }
    assert after == before
    assert not list(tmp_path.glob("metadata_rev*.json"))
    _assert_campaign_lock_available(tmp_path)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
@pytest.mark.parametrize(
    ("max_runs", "wall_hours"),
    (
        (0, None),
        (-1, None),
        (True, None),
        (1.0, None),
        (None, 0.0),
        (None, -1.0),
        (None, math.nan),
        (None, math.inf),
        (None, True),
    ),
    ids=(
        "zero-runs",
        "negative-runs",
        "boolean-runs",
        "float-runs",
        "zero-wall",
        "negative-wall",
        "nan-wall",
        "infinite-wall",
        "boolean-wall",
    ),
)
def test_main_rejects_invalid_limits_before_output_mutation(
    runner,
    max_runs: object,
    wall_hours: object,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=out_dir,
            max_runs=max_runs,
            wall_hours=wall_hours,
            no_resume=True,
        ),
    )

    with pytest.raises(ValueError):
        runner.main()

    assert not out_dir.exists()


def test_bulk_purge_preserves_original_on_atomic_promotion_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    aggregate = tmp_path / "summary.csv"
    _write_csv(
        aggregate,
        ["run_id", "value"],
        [
            {"run_id": "remove-me", "value": 1},
            {"run_id": "keep-me", "value": 2},
        ],
    )
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        spatial.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("promotion failed")
        ),
    )

    with pytest.raises(OSError, match="promotion failed"):
        spatial.purge_run_ids_rows(aggregate, {"remove-me"})

    assert aggregate.read_bytes() == before
    assert not list(tmp_path.glob(".summary.csv.purge.*.tmp"))


def test_campaign_lock_owns_empty_file_before_initializing_byte(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The first write must occur under the Windows mandatory byte lock."""
    observed_sizes: list[int] = []
    original_lock = spatial._lock_campaign_stream

    def observe_then_lock(stream) -> None:
        observed_sizes.append(spatial.os.fstat(stream.fileno()).st_size)
        original_lock(stream)

    monkeypatch.setattr(spatial, "_lock_campaign_stream", observe_then_lock)
    handle = spatial.acquire_campaign_lock(
        tmp_path,
        owner_label="ordering-test",
        config_digest="a" * 64,
    )
    try:
        assert observed_sizes == [0]
        assert spatial.os.fstat(handle.stream.fileno()).st_size == 1
    finally:
        spatial.release_campaign_lock(handle)


def test_campaign_lock_allows_exactly_one_simultaneous_process(
    tmp_path: Path,
) -> None:
    contender_code = r"""
import importlib
import sys
import time
from pathlib import Path

module = importlib.import_module(sys.argv[1])
out_dir = Path(sys.argv[2])
ready = Path(sys.argv[3])
start = Path(sys.argv[4])
release = Path(sys.argv[5])
result = Path(sys.argv[6])
ready.write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 15.0
while not start.exists():
    if time.monotonic() >= deadline:
        raise SystemExit("start barrier timeout")
    time.sleep(0.001)
try:
    handle = module.acquire_campaign_lock(
        out_dir,
        owner_label=module.__name__,
        config_digest=module.__name__,
    )
except RuntimeError:
    result.write_text("REFUSED", encoding="utf-8")
    raise SystemExit(0)
result.write_text("ACQUIRED", encoding="utf-8")
while not release.exists():
    if time.monotonic() >= deadline:
        raise SystemExit("release barrier timeout")
    time.sleep(0.001)
module.release_campaign_lock(handle)
"""
    start = tmp_path / "start"
    release = tmp_path / "release"
    modules = (
        "scripts.run_prelim_spatial_overnight",
        "scripts.run_prelim_readout_heating_overnight",
    )
    processes: list[subprocess.Popen[str]] = []
    ready_paths: list[Path] = []
    result_paths: list[Path] = []
    for index, module_name in enumerate(modules):
        ready = tmp_path / f"ready-{index}"
        result = tmp_path / f"result-{index}"
        ready_paths.append(ready)
        result_paths.append(result)
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    contender_code,
                    module_name,
                    str(tmp_path),
                    str(ready),
                    str(start),
                    str(release),
                    str(result),
                ],
                cwd=spatial.ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        )
    outputs: list[tuple[str, str]] = []
    try:
        deadline = time.monotonic() + 15.0
        while not all(path.exists() for path in ready_paths):
            if time.monotonic() >= deadline:
                pytest.fail("campaign-lock contenders did not reach the barrier")
            time.sleep(0.005)
        start.write_text("go", encoding="utf-8")
        while not all(path.exists() for path in result_paths):
            if time.monotonic() >= deadline:
                pytest.fail("campaign-lock contenders did not report results")
            time.sleep(0.005)
        assert sorted(
            path.read_text(encoding="utf-8") for path in result_paths
        ) == ["ACQUIRED", "REFUSED"]
    finally:
        release.write_text("release", encoding="utf-8")
        for process in processes:
            try:
                outputs.append(process.communicate(timeout=15.0))
            except subprocess.TimeoutExpired:
                process.kill()
                outputs.append(process.communicate())
    assert all(process.returncode == 0 for process in processes), outputs
    assert readout.acquire_campaign_lock is spatial.acquire_campaign_lock
    _assert_campaign_lock_available(tmp_path)


def test_campaign_lock_is_os_released_after_process_crash(
    tmp_path: Path,
) -> None:
    crash_code = r"""
import os
import sys
from pathlib import Path
from scripts.run_prelim_spatial_overnight import acquire_campaign_lock

handle = acquire_campaign_lock(
    Path(sys.argv[1]),
    owner_label="crash-test",
    config_digest="crash-test",
)
print("ACQUIRED", flush=True)
assert handle.stream.closed is False
os._exit(23)
"""
    crashed = subprocess.run(
        [sys.executable, "-c", crash_code, str(tmp_path)],
        cwd=spatial.ROOT,
        text=True,
        capture_output=True,
        timeout=15.0,
        check=False,
    )
    assert crashed.returncode == 23, crashed.stderr
    assert crashed.stdout.strip() == "ACQUIRED"

    # The stale diagnostic sidecar remains after the hard exit but has no
    # authority. The OS-released descriptor lock permits immediate reuse.
    _assert_campaign_lock_available(tmp_path)


def test_campaign_lock_metadata_is_diagnostic_only(tmp_path: Path) -> None:
    handle = spatial.acquire_campaign_lock(
        tmp_path,
        owner_label="first",
        config_digest="first-config",
    )
    handle.owner_path.write_text("{malformed", encoding="utf-8")
    try:
        with pytest.raises(RuntimeError, match="live writer"):
            readout.acquire_campaign_lock(
                tmp_path,
                owner_label="second",
                config_digest="second-config",
            )
    finally:
        spatial.release_campaign_lock(handle)

    handle.owner_path.write_text("{stale-malformed", encoding="utf-8")
    _assert_campaign_lock_available(tmp_path)


def test_campaign_lock_metadata_failure_cannot_wedge_acquisition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    with monkeypatch.context() as context:
        context.setattr(
            spatial,
            "_atomic_write_json",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("metadata promotion failed")
            ),
        )
        with pytest.raises(OSError, match="metadata promotion failed"):
            spatial.acquire_campaign_lock(
                tmp_path,
                owner_label="broken-metadata",
                config_digest="broken-metadata",
            )

    _assert_campaign_lock_available(tmp_path)


def test_campaign_lock_release_preamble_failure_still_unlocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    handle = spatial.acquire_campaign_lock(
        tmp_path,
        owner_label="release-preamble",
        config_digest="release-preamble",
    )
    with monkeypatch.context() as context:
        context.setattr(
            spatial.time,
            "time",
            lambda: (_ for _ in ()).throw(RuntimeError("clock failed")),
        )
        with pytest.warns(RuntimeWarning, match="clock failed"):
            spatial.release_campaign_lock(handle)

    assert handle.stream.closed
    _assert_campaign_lock_available(tmp_path)


@pytest.mark.parametrize("runner", RUNNERS, ids=("spatial", "readout"))
def test_main_releases_campaign_lock_when_campaign_raises(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        runner,
        "_parse_args",
        lambda: argparse.Namespace(
            preset="smoke",
            out_dir=tmp_path,
            max_runs=1,
            wall_hours=None,
            no_resume=False,
        ),
    )
    monkeypatch.setattr(
        runner,
        "_run_campaign",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("campaign crashed")
        ),
    )

    with pytest.raises(RuntimeError, match="campaign crashed"):
        runner.main()

    _assert_campaign_lock_available(tmp_path)
