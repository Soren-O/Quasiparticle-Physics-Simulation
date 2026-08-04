# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Regression tests for review packet P06 (2026-08-03).

Covers the canonical-path refusal added to the Fischer Sec. V
``Q_i(P_read)`` development driver. That artifact pair is deliberately
quarantined; before this fix the quarantine was a property of the on-disk
bytes alone (``read_baseline`` rejects files with no ``# schema_version=``
marker), so a single ``write_baseline(run())`` at the public default would
have replaced it with a self-certified, fingerprint-matching artifact and
silently lifted the quarantine.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023.figs_9_13_qi_vs_pread import (
    NUM_POINTS,
    Figs913Result,
    LegacyBaselineError,
    baseline_path,
    development_baseline_path,
    development_plot_path,
    plot_path,
    read_baseline,
    write_baseline,
    write_plot,
)


def _sentinel_result() -> Figs913Result:
    """A structurally valid but numerically invalid result.

    The path guard must fire before ``_validated_result`` ever sees this, so
    a refusal proves the destination was rejected rather than the payload.
    """
    zeros = np.zeros(NUM_POINTS)
    return Figs913Result(
        P_read_uev_per_ns=zeros,
        P_read_dbm=zeros,
        Q_i=zeros,
        Q_tot=zeros,
        n_bar=zeros,
        iterations=np.zeros(NUM_POINTS, dtype=np.int64),
        qp_backward_error=zeros,
        qp_max_abs_residual=zeros,
        nbar_fixed_point_residual=zeros,
    )


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_direct_writers_cannot_clobber_quarantined_canonical() -> None:
    csv_path = baseline_path()
    pdf_path = plot_path()
    csv_before = _digest(csv_path)
    pdf_before = _digest(pdf_path)
    result = _sentinel_result()

    # Both the defaulting call and the explicit canonical target must refuse,
    # for the CSV and for its companion PDF.
    with pytest.raises(RuntimeError, match="forbidden"):
        write_baseline(result)
    with pytest.raises(RuntimeError, match="forbidden"):
        write_baseline(result, csv_path)
    with pytest.raises(RuntimeError, match="forbidden"):
        write_plot(result)
    with pytest.raises(RuntimeError, match="forbidden"):
        write_plot(result, pdf_path)

    assert _digest(csv_path) == csv_before
    assert _digest(pdf_path) == pdf_before
    assert not list(csv_path.parent.glob(f".{csv_path.name}.*.tmp"))


def test_quarantine_survives_the_write_attempts() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"No canonical artifact at {path}.")
    with pytest.raises(LegacyBaselineError, match="quarantined"):
        read_baseline(path)


def test_development_paths_are_still_writable_targets(tmp_path: Path) -> None:
    # The guard must key on the canonical paths only; the development targets
    # generate_baseline() uses are not canonical and must remain acceptable.
    for candidate in (development_baseline_path(), development_plot_path()):
        assert candidate.resolve() != baseline_path().resolve()
        assert candidate.resolve() != plot_path().resolve()

    # A non-canonical destination gets past the guard and fails on the payload
    # instead, i.e. the refusal above really was about the destination.
    with pytest.raises(RuntimeError) as excinfo:
        write_baseline(_sentinel_result(), tmp_path / "figs913.development.csv")
    assert "forbidden" not in str(excinfo.value)
