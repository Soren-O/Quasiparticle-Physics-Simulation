"""Certified artifact tests for Fischer 2024 native Figs. 5-7."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qpsim.observables.density import qp_fraction

import validation.fischer_2024.fig8_xqp_pb as shared
import validation.fischer_2024.figs_5_7_fe_pb as target
from validation.fischer_2024._artifact import (
    TARGET_QP_BACKWARD_ERROR_LIMIT,
    ArtifactValidationError,
    LegacyArtifactError,
    QPCertificate,
)
from validation.fischer_2024.fig8_xqp_pb import (
    DELTA_0,
    POWER_LEVELS,
    _build_state,
    _material,
)


def _synthetic_result() -> target.Figs57Result:
    state = _build_state(_material(), target.T_BATH_FE)
    E = state.spectral.E
    f_thermal = state.f.copy()
    f_by_power = {
        power: np.clip(f_thermal + power * np.exp(-(E - E[0]) / 100.0), 0.0, 1.0)
        for power in POWER_LEVELS
    }
    return target.Figs57Result(
        E=E,
        powers=POWER_LEVELS,
        f_thermal=f_thermal,
        f_by_power=f_by_power,
        x_qp_by_power={
            p: float(qp_fraction(f_by_power[p], state.spectral, delta_0=DELTA_0))
            for p in POWER_LEVELS
        },
        qp_backward_error_by_power=dict.fromkeys(POWER_LEVELS, 1e-08),
        qp_residual_inf_by_power=dict.fromkeys(POWER_LEVELS, 1e-16),
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


def test_current_artifact_round_trips(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    reference = _synthetic_result()
    path = target.write_baseline(reference, tmp_path / "figs57.csv")
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    metadata = json.loads(rows[1][0].split("=", 1)[1])
    assert metadata["certificate_target_qp_residual_inf"] == target.NEWTON_TOL
    decoded = target.read_baseline(path)
    np.testing.assert_array_equal(decoded.E, reference.E)
    assert decoded.powers == POWER_LEVELS
    for power in POWER_LEVELS:
        assert decoded.qp_backward_error_by_power[power] == 1.0e-8


def test_thermal_seed_accepts_ulp_roundoff_but_rejects_drift(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    rounded = _synthetic_result()
    np.nextafter(rounded.f_thermal, np.inf, out=rounded.f_thermal)
    decoded = target.read_baseline(
        target.write_baseline(rounded, tmp_path / "rounded-thermal.csv")
    )
    np.testing.assert_array_equal(decoded.f_thermal, rounded.f_thermal)

    drifted = _synthetic_result()
    drifted.f_thermal[0] *= 1.0 + 1.0e-9
    with pytest.raises(ArtifactValidationError, match="thermal occupation"):
        target.write_baseline(drifted, tmp_path / "drifted-thermal.csv")


def test_writer_rejects_residual_at_newton_tolerance(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    result = _synthetic_result()
    result.qp_residual_inf_by_power[POWER_LEVELS[0]] = target.NEWTON_TOL
    with pytest.raises(RuntimeError, match="residual"):
        target.write_baseline(result, tmp_path / "figs57.csv")


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "duplicate", "missing", "nonfinite", "out_of_range", "over_gate"],
)
def test_reader_rejects_invalid_artifacts(
    tmp_path: Path,
    mutation: str,
    _synthetic_certificate: None,
) -> None:
    path = target.write_baseline(_synthetic_result(), tmp_path / "figs57.csv")

    def mutate(rows: list[list[str]]) -> None:
        if mutation == "malformed":
            rows[2][0] = "wrong_axis"
        elif mutation == "duplicate":
            rows[4][0] = rows[3][0]
        elif mutation == "missing":
            rows.pop()
        elif mutation == "nonfinite":
            rows[3][1] = "inf"
        elif mutation == "out_of_range":
            rows[3][1] = "-0.1"
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


def test_reduced_live_run_writes_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target, "NUM_BINS", 90)
    monkeypatch.setattr(target, "POWER_LEVELS", (1.0e-2,))
    monkeypatch.setattr(shared, "NUM_BINS", 90)
    result = target.run()
    decoded = target.read_baseline(target.write_baseline(result, tmp_path / "reduced_figs57.csv"))
    np.testing.assert_allclose(
        decoded.f_by_power[1.0e-2],
        result.f_by_power[1.0e-2],
        rtol=0.0,
        atol=0.0,
    )


def test_writer_preserves_existing_file_on_mid_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    _synthetic_certificate: None,
) -> None:
    path = tmp_path / "figs57.csv"
    path.write_text("sentinel\n", encoding="utf-8")
    real_writer = csv.writer

    class FailingWriter:
        def __init__(self, fp: Any, **kwargs: Any) -> None:
            self._writer = real_writer(fp, **kwargs)
            self._calls = 0

        def writerow(self, row: Any) -> Any:
            self._calls += 1
            if self._calls == 4:
                raise OSError("injected write failure")
            return self._writer.writerow(row)

    monkeypatch.setattr(csv, "writer", FailingWriter)
    with pytest.raises(OSError, match="injected"):
        target.write_baseline(_synthetic_result(), path)
    assert path.read_text(encoding="utf-8") == "sentinel\n"


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
    assert baseline.E.shape == (shared.NUM_BINS,)
    assert baseline.powers == POWER_LEVELS


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
    assert result.powers == baseline.powers
    np.testing.assert_allclose(
        result.f_thermal,
        baseline.f_thermal,
        rtol=1e-6,
        atol=1e-14,
    )
    for power in result.powers:
        np.testing.assert_allclose(
            result.f_by_power[power],
            baseline.f_by_power[power],
            rtol=1e-6,
            atol=1e-14,
            err_msg=f"f drift at power={power:g}",
        )
        assert result.x_qp_by_power[power] == pytest.approx(
            baseline.x_qp_by_power[power],
            rel=1e-6,
        )
