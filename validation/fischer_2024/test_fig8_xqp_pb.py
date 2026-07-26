"""Certified artifact tests for the Fischer 2024 native Fig. 8 sweep."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from qpsim.observables.density import qp_fraction

import validation.fischer_2024.fig8_xqp_pb as target
from validation.fischer_2024._artifact import (
    TARGET_QP_BACKWARD_ERROR_LIMIT,
    ArtifactValidationError,
    LegacyArtifactError,
    QPCertificate,
    capture_producer_identity,
)
from validation.fischer_2024.fig8_xqp_pb import (
    NEWTON_TOL,
    POWER_LEVELS,
    T_BATH_VALUES,
    Fig8Result,
    baseline_path,
    read_baseline,
    run,
)


def _synthetic_result() -> Fig8Result:
    T_bath = np.asarray(T_BATH_VALUES, dtype=float)
    states = [target._build_state(target._material(), float(T)) for T in T_bath]
    f_by_power = {
        power: np.asarray(
            [
                np.clip(
                    state.f + power * np.exp(-(state.spectral.E - state.spectral.E[0]) / 100.0),
                    0.0,
                    1.0,
                )
                for state in states
            ]
        )
        for power in POWER_LEVELS
    }
    return Fig8Result(
        T_bath=T_bath,
        powers=POWER_LEVELS,
        x_qp_thermal=np.asarray(
            [qp_fraction(state.f, state.spectral, delta_0=target.DELTA_0) for state in states]
        ),
        x_qp_by_power={
            power: np.asarray(
                [
                    qp_fraction(f, state.spectral, delta_0=target.DELTA_0)
                    for f, state in zip(f_by_power[power], states, strict=True)
                ]
            )
            for power in POWER_LEVELS
        },
        qp_backward_error_by_power={p: np.full(T_bath.size, 1.0e-8) for p in POWER_LEVELS},
        qp_residual_inf_by_power={p: np.full(T_bath.size, 1.0e-16) for p in POWER_LEVELS},
        f_by_power=f_by_power,
    )


def _rewrite_csv(path: Path, mutate: Callable[[list[list[str]]], None]) -> None:
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)


def _write_baseline(result: Fig8Result, path: Path) -> Path:
    producer = capture_producer_identity(target.solver_fingerprint())
    return target.write_baseline(result, path, producer=producer)


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
    path = _write_baseline(_synthetic_result(), tmp_path / "fig8.csv")
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    metadata = json.loads(rows[1][0].split("=", 1)[1])
    assert metadata["certificate_target_qp_residual_inf"] == NEWTON_TOL
    decoded = read_baseline(path)
    assert decoded.powers == POWER_LEVELS
    np.testing.assert_array_equal(decoded.T_bath, T_BATH_VALUES)
    for power in POWER_LEVELS:
        np.testing.assert_array_equal(
            decoded.qp_backward_error_by_power[power],
            np.full(len(T_BATH_VALUES), 1.0e-8),
        )


def test_writer_rejects_residual_at_newton_tolerance(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    result = _synthetic_result()
    result.qp_residual_inf_by_power[POWER_LEVELS[0]][0] = NEWTON_TOL
    with pytest.raises(RuntimeError, match="residual"):
        _write_baseline(result, tmp_path / "fig8.csv")


@pytest.mark.parametrize(
    "mutation",
    [
        "malformed",
        "duplicate",
        "missing",
        "nonfinite",
        "negative",
        "payload_drift",
        "over_gate",
    ],
)
def test_reader_rejects_invalid_artifacts(
    tmp_path: Path,
    mutation: str,
    _synthetic_certificate: None,
) -> None:
    path = _write_baseline(_synthetic_result(), tmp_path / "fig8.csv")

    def mutate(rows: list[list[str]]) -> None:
        if mutation == "malformed":
            rows[2][0] = "wrong_axis"
        elif mutation == "duplicate":
            rows[4][0] = rows[3][0]
        elif mutation == "missing":
            rows.pop()
        elif mutation == "nonfinite":
            rows[3][1] = "nan"
        elif mutation == "negative":
            rows[3][1] = "-1.0"
        elif mutation == "payload_drift":
            rows[3][1] = f"{float(rows[3][1]) * 1.001:.17e}"
        else:
            metadata = json.loads(rows[1][0].split("=", 1)[1])
            metadata["certificate_points"][0]["qp_backward_error"] = (
                2.0 * TARGET_QP_BACKWARD_ERROR_LIMIT
            )
            rows[1][0] = "# qpsim_metadata=" + json.dumps(metadata)

    _rewrite_csv(path, mutate)
    with pytest.raises(ArtifactValidationError) as captured:
        read_baseline(path)
    assert not isinstance(captured.value, LegacyArtifactError)
    if mutation == "payload_drift":
        assert "payload hash" in str(captured.value)


def test_writer_reassembles_and_rejects_forged_certificate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"QP (backward error|residual_inf)"):
        _write_baseline(_synthetic_result(), tmp_path / "forged.csv")


def test_writer_rejects_negative_x_qp(tmp_path: Path) -> None:
    result = _synthetic_result()
    result.x_qp_thermal[0] = -1.0
    with pytest.raises(ArtifactValidationError, match=r"below 0\.0"):
        _write_baseline(result, tmp_path / "negative.csv")


def test_reduced_live_run_writes_bound_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target, "NUM_BINS", 90)
    monkeypatch.setattr(target, "T_BATH_VALUES", (0.10,))
    monkeypatch.setattr(target, "POWER_LEVELS", (1.0e-2,))
    result = target.run()
    decoded = target.read_baseline(
        _write_baseline(result, tmp_path / "reduced_fig8.csv")
    )
    np.testing.assert_allclose(
        decoded.x_qp_by_power[1.0e-2],
        result.x_qp_by_power[1.0e-2],
        rtol=0.0,
        atol=0.0,
    )


def test_archived_legacy_artifact_is_explicitly_rejected() -> None:
    legacy_path = (
        baseline_path().parent.parent
        / "legacy"
        / "fischer_2024_pre_strict_v2"
        / baseline_path().name
    )
    with pytest.raises(ArtifactValidationError, match=r"legacy|wrong schema"):
        read_baseline(legacy_path)


def test_promoted_canonical_is_current_and_certified() -> None:
    path = baseline_path()
    assert path.is_file(), f"Promoted strict-v2 canonical is missing at {path}."
    baseline = read_baseline(path)
    np.testing.assert_array_equal(baseline.T_bath, target.T_BATH_VALUES)
    assert baseline.powers == target.POWER_LEVELS


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.xfail(f"Certified baseline not found at {path}.")
    try:
        baseline = read_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(f"Legacy uncertified canonical baseline is quarantined: {exc}")

    result = run()
    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    assert result.powers == baseline.powers
    np.testing.assert_allclose(
        result.x_qp_thermal,
        baseline.x_qp_thermal,
        rtol=1e-6,
        atol=1e-14,
    )
    for power in result.powers:
        np.testing.assert_allclose(
            result.x_qp_by_power[power],
            baseline.x_qp_by_power[power],
            rtol=1e-6,
            atol=1e-14,
            err_msg=f"Mismatch at power={power}",
        )
        # Certificates are acceptance bounds, not physical observables.
        # Their tiny values depend on the last floating-point iterate and may
        # vary by orders of magnitude around 1e-12 across platforms while the
        # solved x_qp curves agree. Require the advertised contracts instead
        # of bit-pinning diagnostic roundoff to the baseline producer.
        backward = result.qp_backward_error_by_power[power]
        residual = result.qp_residual_inf_by_power[power]
        assert np.all(np.isfinite(backward))
        assert np.all(np.isfinite(residual))
        assert np.all(
            (backward >= 0.0) & (backward <= TARGET_QP_BACKWARD_ERROR_LIMIT)
        )
        assert np.all((residual >= 0.0) & (residual < NEWTON_TOL))
