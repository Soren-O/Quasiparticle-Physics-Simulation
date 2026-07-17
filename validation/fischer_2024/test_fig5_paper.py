"""Certified artifact tests for the Fischer 2024 Fig. 5 characterization."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from qpsim.observables.density import qp_fraction

import validation.fischer_2024.fig5_paper as target
from validation.fischer_2024._artifact import (
    TARGET_QP_BACKWARD_ERROR_LIMIT,
    ArtifactValidationError,
    LegacyArtifactError,
    QPCertificate,
)


def _synthetic_result() -> target.Fig5PaperResult:
    E, _, spectral = target._build_grid_and_spectral()
    state = target._build_state(target._material(), spectral)
    f_by_drive = {
        drive: np.clip(state.f + drive * np.exp(-(E - E[0]) / 100.0), 0.0, 1.0)
        for drive in target.PAPER_DRIVES_HZ
    }
    f0 = {
        drive: target._neumann_f0(spectral, drive_ns)
        for drive, drive_ns in zip(
            target.PAPER_DRIVES_HZ,
            target.PAPER_DRIVES_NS_INV,
            strict=True,
        )
    }
    f01 = {
        drive: f0[drive] + target._neumann_f1(spectral, drive_ns)
        for drive, drive_ns in zip(
            target.PAPER_DRIVES_HZ,
            target.PAPER_DRIVES_NS_INV,
            strict=True,
        )
    }
    f012 = {
        drive: f01[drive] + target._neumann_f2(spectral, drive_ns)
        for drive, drive_ns in zip(
            target.PAPER_DRIVES_HZ,
            target.PAPER_DRIVES_NS_INV,
            strict=True,
        )
    }
    return target.Fig5PaperResult(
        E=E,
        drives_hz=target.PAPER_DRIVES_HZ,
        drives_ns_inv=target.PAPER_DRIVES_NS_INV,
        f_thermal=state.f,
        f_by_drive=f_by_drive,
        x_qp_by_drive={
            d: float(qp_fraction(f_by_drive[d], spectral, delta_0=target.DELTA_0))
            for d in target.PAPER_DRIVES_HZ
        },
        f0_by_drive=f0,
        f01_by_drive=f01,
        f012_by_drive=f012,
        qp_backward_error_by_drive=dict.fromkeys(target.PAPER_DRIVES_HZ, 1e-08),
        qp_residual_inf_by_drive=dict.fromkeys(target.PAPER_DRIVES_HZ, 1e-16),
    )


def _rewrite_csv(path: Path, mutate: Callable[[list[list[str]]], None]) -> None:
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)


@pytest.fixture
def _synthetic_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        target,
        "qp_certificate",
        lambda *_args, **_kwargs: QPCertificate(1.0e-8, 1.0e-16),
    )


def test_unit_audit_conversion_pinned() -> None:
    assert target.HZ_TO_NS_INV == 1.0e-9
    for hz, ns_inv in zip(
        target.PAPER_DRIVES_HZ,
        target.PAPER_DRIVES_NS_INV,
        strict=True,
    ):
        assert ns_inv == pytest.approx(hz * 1.0e-9, rel=0.0, abs=1e-30)
    target._assert_unit_audit()


def test_current_artifact_round_trips(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    reference = _synthetic_result()
    path = target.write_baseline(reference, tmp_path / "fig5.csv")
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    metadata = json.loads(rows[1][0].split("=", 1)[1])
    assert metadata["certificate_target_qp_residual_inf"] == target.NEWTON_TOL
    decoded = target.read_baseline(path)
    np.testing.assert_array_equal(decoded.E, reference.E)
    assert decoded.drives_hz == target.PAPER_DRIVES_HZ
    for drive in target.PAPER_DRIVES_HZ:
        assert decoded.qp_backward_error_by_drive[drive] == 1.0e-8


def test_writer_rejects_residual_at_newton_tolerance(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    result = _synthetic_result()
    result.qp_residual_inf_by_drive[target.PAPER_DRIVES_HZ[0]] = target.NEWTON_TOL
    with pytest.raises(RuntimeError, match="residual"):
        target.write_baseline(result, tmp_path / "fig5.csv")


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "duplicate", "missing", "nonfinite", "out_of_range", "over_gate"],
)
def test_reader_rejects_invalid_artifacts(
    tmp_path: Path,
    mutation: str,
    _synthetic_certificate: None,
) -> None:
    path = target.write_baseline(_synthetic_result(), tmp_path / "fig5.csv")

    def mutate(rows: list[list[str]]) -> None:
        if mutation == "malformed":
            rows[2][0] = "wrong_axis"
        elif mutation == "duplicate":
            rows[4][0] = rows[3][0]
        elif mutation == "missing":
            rows.pop()
        elif mutation == "nonfinite":
            rows[3][1] = "-inf"
        elif mutation == "out_of_range":
            rows[3][1] = "1.1"
        else:
            metadata = json.loads(rows[1][0].split("=", 1)[1])
            metadata["certificate_points"][0]["qp_backward_error"] = (
                2.0 * TARGET_QP_BACKWARD_ERROR_LIMIT
            )
            rows[1][0] = "# qpsim_metadata=" + json.dumps(metadata)

    _rewrite_csv(path, mutate)
    with pytest.raises(ArtifactValidationError) as captured:
        target.read_baseline(path)
    assert not isinstance(captured.value, LegacyArtifactError)


def test_writer_reassembles_and_rejects_forged_certificate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"QP (backward error|residual_inf)"):
        target.write_baseline(_synthetic_result(), tmp_path / "forged.csv")


def test_writer_rejects_out_of_domain_occupation(tmp_path: Path) -> None:
    result = _synthetic_result()
    result.f_by_drive[target.PAPER_DRIVES_HZ[0]][0] = 1.1
    with pytest.raises(ArtifactValidationError, match=r"above 1\.0"):
        target.write_baseline(result, tmp_path / "invalid_f.csv")


def test_reduced_live_run_writes_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drive_hz = 1.0e-2
    monkeypatch.setattr(target, "NUM_BINS", 90)
    monkeypatch.setattr(target, "PAPER_DRIVES_HZ", (drive_hz,))
    monkeypatch.setattr(
        target,
        "PAPER_DRIVES_NS_INV",
        (drive_hz * target.HZ_TO_NS_INV,),
    )
    result = target.run()
    decoded = target.read_baseline(target.write_baseline(result, tmp_path / "reduced_fig5.csv"))
    np.testing.assert_allclose(
        decoded.f_by_drive[drive_hz],
        result.f_by_drive[drive_hz],
        rtol=0.0,
        atol=0.0,
    )


def test_archived_legacy_artifact_is_explicitly_rejected() -> None:
    legacy_path = (
        target.baseline_path().parent.parent
        / "legacy"
        / "fischer_2024_pre_strict_v2"
        / target.baseline_path().name
    )
    with pytest.raises(ArtifactValidationError, match=r"legacy|wrong schema"):
        target.read_baseline(legacy_path)


def test_promoted_canonical_is_current_and_certified() -> None:
    path = target.baseline_path()
    assert path.is_file(), f"Promoted strict-v2 canonical is missing at {path}."
    baseline = target.read_baseline(path)
    assert baseline.E.shape == (target.NUM_BINS,)
    assert baseline.drives_hz == target.PAPER_DRIVES_HZ


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = target.baseline_path()
    if not path.exists():
        pytest.xfail(f"Certified baseline not found at {path}.")
    try:
        baseline = target.read_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(f"Legacy uncertified canonical baseline is quarantined: {exc}")

    result = target.run()
    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.drives_hz == baseline.drives_hz
    np.testing.assert_allclose(
        result.drives_ns_inv,
        baseline.drives_ns_inv,
        rtol=0.0,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        result.f_thermal,
        baseline.f_thermal,
        rtol=1e-6,
        atol=1e-14,
    )
    for drive in result.drives_hz:
        np.testing.assert_allclose(
            result.f_by_drive[drive],
            baseline.f_by_drive[drive],
            rtol=1e-6,
            atol=1e-14,
        )
        assert result.x_qp_by_drive[drive] == pytest.approx(
            baseline.x_qp_by_drive[drive],
            rel=1e-6,
        )
