"""Regression test: Fischer Figs 9-13 Q_i(P_read) matches baseline to 1e-4.

Iterative-mode tolerance per NFP §6.4.1 (nbar-loop tol × MB sub-gap
quadrature). Slow-marked (21 P_read points × variable nbar-loop
iterations).

The legacy canonical artifact is intentionally xfailed because it predates the
current schema and certificates. Development-only generation (not promotion)::

    python -m validation.fischer_2023.figs_9_13_qi_vs_pread
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qpsim.constants import HBAR_UEV_NS
from qpsim.services.nbar_loop import dbm_to_uev_per_ns

from validation.fischer_2023.figs_9_13_qi_vs_pread import (
    NBAR_FIXED_POINT_RESIDUAL_LIMIT,
    NUM_POINTS,
    OMEGA_0,
    P_READ_DBM_MAX,
    P_READ_DBM_MIN,
    Q_C,
    QP_BACKWARD_ERROR_LIMIT,
    Figs913Result,
    LegacyBaselineError,
    _nbar_fixed_point_residual,
    baseline_path,
    read_baseline,
    run,
    write_baseline,
)


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.figs_9_13_qi_vs_pread"
        )

    try:
        baseline = read_baseline(path)
    except LegacyBaselineError as err:
        pytest.xfail(str(err))
    result = run()

    np.testing.assert_allclose(
        result.P_read_uev_per_ns,
        baseline.P_read_uev_per_ns,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.P_read_dbm,
        baseline.P_read_dbm,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.Q_i,
        baseline.Q_i,
        rtol=1e-4,
        atol=1e-14,
        err_msg="Q_i drift",
    )
    np.testing.assert_allclose(
        result.Q_tot,
        baseline.Q_tot,
        rtol=1e-4,
        atol=1e-14,
        err_msg="Q_tot drift",
    )
    np.testing.assert_allclose(
        result.n_bar,
        baseline.n_bar,
        rtol=1e-4,
        atol=1e-14,
        err_msg="n_bar drift",
    )
    np.testing.assert_array_equal(result.iterations, baseline.iterations)
    for field in (
        "qp_backward_error",
        "qp_max_abs_residual",
        "nbar_fixed_point_residual",
    ):
        np.testing.assert_allclose(
            getattr(result, field),
            getattr(baseline, field),
            rtol=1e-6,
            atol=1e-30,
            err_msg=f"{field} certificate drift",
        )


def _valid_result() -> Figs913Result:
    dbm = np.linspace(P_READ_DBM_MIN, P_READ_DBM_MAX, NUM_POINTS)
    power = np.array([dbm_to_uev_per_ns(float(value)) for value in dbm])
    q_i = np.linspace(1.0e8, 2.0e8, NUM_POINTS)
    q_tot = 1.0 / (1.0 / q_i + 1.0 / Q_C)
    n_bar = 2.0 * HBAR_UEV_NS * q_tot**2 * power / (OMEGA_0**2 * Q_C)
    nbar_residual = np.asarray(
        [
            _nbar_fixed_point_residual(float(p), float(n), float(q))
            for p, n, q in zip(power, n_bar, q_tot, strict=True)
        ]
    )
    return Figs913Result(
        P_read_uev_per_ns=power,
        P_read_dbm=dbm,
        Q_i=q_i,
        Q_tot=q_tot,
        n_bar=n_bar,
        iterations=np.full(NUM_POINTS, 2, dtype=np.int64),
        qp_backward_error=np.full(NUM_POINTS, 1.0e-8),
        qp_max_abs_residual=np.full(NUM_POINTS, 1.0e-18),
        nbar_fixed_point_residual=nbar_residual,
    )


def _csv_rows(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as fp:
        return list(csv.reader(fp))


def _write_csv_rows(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp).writerows(rows)


def _current_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "figs913.csv"
    write_baseline(_valid_result(), path)
    return path


def test_current_schema_round_trips_atomically(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    actual = read_baseline(path)
    expected = _valid_result()
    for field in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_legacy_artifact_is_explicitly_quarantined(tmp_path: Path) -> None:
    path = tmp_path / "legacy.csv"
    path.write_text("# legacy\nP_read_uev_per_ns,Q_i\n1,2\n", encoding="utf-8")
    with pytest.raises(LegacyBaselineError, match="quarantined"):
        read_baseline(path)


def test_reader_rejects_duplicate_metadata(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    rows.insert(1, rows[0].copy())
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="Duplicate metadata"):
        read_baseline(path)


def test_reader_rejects_missing_metadata(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = [row for row in _csv_rows(path) if not row[0].startswith("# source_fingerprint_sha256=")]
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="metadata contract mismatch"):
        read_baseline(path)


def test_reader_rejects_stale_source_fingerprint(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    fingerprint_index = next(
        i for i, row in enumerate(rows) if row[0].startswith("# source_fingerprint_sha256=")
    )
    rows[fingerprint_index][0] = f"# source_fingerprint_sha256={'0' * 64}"
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="source fingerprint is stale"):
        read_baseline(path)


def test_reader_rejects_nonfinite_data(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    header_index = next(i for i, row in enumerate(rows) if row[0] == "P_read_uev_per_ns")
    rows[header_index + 1][2] = "nan"
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="non-finite"):
        read_baseline(path)


def test_reader_rejects_duplicate_power_axis(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    header_index = next(i for i, row in enumerate(rows) if row[0] == "P_read_uev_per_ns")
    rows[header_index + 2][0] = rows[header_index + 1][0]
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="power axis"):
        read_baseline(path)


def test_reader_rejects_wrong_row_count(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    rows.pop()
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match=f"{NUM_POINTS} rows"):
        read_baseline(path)


def test_reader_rejects_over_gate_certificate(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    header_index = next(i for i, row in enumerate(rows) if row[0] == "P_read_uev_per_ns")
    over = 2.0 * QP_BACKWARD_ERROR_LIMIT
    rows[header_index + 1][6] = f"{over:.17e}"
    maximum_index = next(
        i for i, row in enumerate(rows) if row[0].startswith("# max_qp_backward_error=")
    )
    rows[maximum_index][0] = f"# max_qp_backward_error={over:.17e}"
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="exceeds its gate"):
        read_baseline(path)


def test_reader_rejects_forged_nbar_certificate(tmp_path: Path) -> None:
    path = _current_artifact(tmp_path)
    rows = _csv_rows(path)
    header_index = next(i for i, row in enumerate(rows) if row[0] == "P_read_uev_per_ns")
    forged = 1.0e-8
    rows[header_index + 1][8] = f"{forged:.17e}"
    maximum_index = next(
        i for i, row in enumerate(rows) if row[0].startswith("# max_nbar_fixed_point_residual=")
    )
    rows[maximum_index][0] = f"# max_nbar_fixed_point_residual={forged:.17e}"
    _write_csv_rows(path, rows)
    with pytest.raises(RuntimeError, match="does not match"):
        read_baseline(path)


def test_writer_rejects_forged_nbar_certificate(tmp_path: Path) -> None:
    result = _valid_result()
    forged = np.full(NUM_POINTS, 1.0e-8)
    with pytest.raises(RuntimeError, match="does not match"):
        write_baseline(
            replace(result, nbar_fixed_point_residual=forged),
            tmp_path / "forged.csv",
        )


def test_writer_rejects_over_gate_certificate(tmp_path: Path) -> None:
    result = _valid_result()
    over = np.full(NUM_POINTS, 2.0 * NBAR_FIXED_POINT_RESIDUAL_LIMIT)
    with pytest.raises(RuntimeError, match="nbar fixed-point"):
        write_baseline(
            replace(result, nbar_fixed_point_residual=over),
            tmp_path / "over.csv",
        )
