"""Certified artifact and continuation tests for Fischer 2024 Fig. 8."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.observables.density import qp_fraction

import validation.fischer_2024.fig8_paper as target
from validation.fischer_2024._artifact import (
    TARGET_QP_BACKWARD_ERROR_LIMIT,
    TARGET_QP_RESIDUAL_INF_LIMIT,
    ArtifactValidationError,
    LegacyArtifactError,
    QPCertificate,
    _atomic_text_file,
    capture_producer_identity,
)


def _synthetic_result() -> target.Fig8PaperResult:
    T_bath = np.asarray(target.T_BATH_VALUES, dtype=float)
    drives_ns = tuple(d * target.HZ_TO_NS_INV for d in target.PAPER_DRIVES_HZ)
    states = [target._build_state(target._material(), float(T)) for T in T_bath]
    f_by_drive = {
        drive: np.asarray(
            [
                np.clip(
                    state.f + drive * np.exp(-(state.spectral.E - state.spectral.E[0]) / 100.0),
                    0.0,
                    1.0,
                )
                for state in states
            ]
        )
        for drive in target.PAPER_DRIVES_HZ
    }
    return target.Fig8PaperResult(
        T_bath=T_bath,
        drives_hz=target.PAPER_DRIVES_HZ,
        drives_ns_inv=drives_ns,
        x_qp_thermal=np.asarray(
            [qp_fraction(state.f, state.spectral, delta_0=target.DELTA_0) for state in states]
        ),
        x_qp_num_by_drive={
            drive: np.asarray(
                [
                    qp_fraction(f, state.spectral, delta_0=target.DELTA_0)
                    for f, state in zip(f_by_drive[drive], states, strict=True)
                ]
            )
            for drive in target.PAPER_DRIVES_HZ
        },
        qp_backward_error_by_drive={
            d: np.full(T_bath.size, 1.0e-8) for d in target.PAPER_DRIVES_HZ
        },
        qp_residual_inf_by_drive={d: np.full(T_bath.size, 1.0e-12) for d in target.PAPER_DRIVES_HZ},
        qp_number_backward_error_by_drive={
            d: np.full(T_bath.size, 1.0e-9) for d in target.PAPER_DRIVES_HZ
        },
        f_by_drive=f_by_drive,
    )


def _rewrite_csv(path: Path, mutate: Callable[[list[list[str]]], None]) -> None:
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    mutate(rows)
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)


def _write_baseline(result: target.Fig8PaperResult, path: Path) -> Path:
    producer = capture_producer_identity(target.solver_fingerprint())
    return target.write_baseline(result, path, producer=producer)


def _read_summary(path: Path | None = None) -> target.Fig8PaperResult:
    return target.read_baseline(
        path,
        accept_producer_certificate_claims=True,
    )


@pytest.fixture
def _synthetic_certificate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        target,
        "qp_certificate",
        lambda *_args, **_kwargs: QPCertificate(1.0e-8, 1.0e-12, 1.0e-9),
    )


def test_unit_audit_pinned() -> None:
    assert target.HZ_TO_NS_INV == 1e-9
    assert target.PAPER_DRIVES_HZ == (1e-2, 1e-4, 1e-6)
    target._assert_unit_audit()


def test_v5_schema_contains_numerical_curves_only() -> None:
    assert target.ARTIFACT_SCHEMA.endswith(".v5")
    columns = target._columns()
    assert len(columns) == 2 + len(target.PAPER_DRIVES_HZ)
    assert all("x_qp_num_" in column for column in columns[2:])
    assert not any(
        token in column
        for column in columns
        for token in ("ana", "analytic", "heuristic", "placeholder")
    )


def test_plot_emits_only_thermal_and_numerical_curves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.axes import Axes

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    titles: list[str] = []
    original_loglog = Axes.loglog
    original_set_title = Axes.set_title

    def recording_loglog(
        self: Axes,
        *args: object,
        **kwargs: object,
    ) -> list[object]:
        calls.append((args, kwargs))
        return original_loglog(self, *args, **kwargs)

    def recording_set_title(
        self: Axes,
        label: str,
        *args: object,
        **kwargs: object,
    ) -> object:
        titles.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "loglog", recording_loglog)
    monkeypatch.setattr(Axes, "set_title", recording_set_title)
    target.write_plot(_synthetic_result(), tmp_path / "fig8.pdf")

    assert len(calls) == 1 + len(target.PAPER_DRIVES_HZ)
    assert calls[0][0][2] == "k--"  # thermal reference only
    assert all(call[0][2] == "-" for call in calls[1:])
    assert any("borrowed from the Fig. 5 caption" in title for title in titles)
    assert any("topology only" in title for title in titles)


def test_current_artifact_round_trips(
    tmp_path: Path,
    _synthetic_certificate: None,
) -> None:
    reference = _synthetic_result()
    path = _write_baseline(reference, tmp_path / "fig8_paper.csv")
    with pytest.raises(ArtifactValidationError, match="producer assertions only"):
        target.read_baseline(path)
    decoded = _read_summary(path)
    np.testing.assert_array_equal(decoded.T_bath, reference.T_bath)
    assert decoded.drives_hz == target.PAPER_DRIVES_HZ
    assert decoded.f_by_drive is None
    assert decoded.certificate_scope == target.SUMMARY_CERTIFICATE_SCOPE
    for drive in target.PAPER_DRIVES_HZ:
        np.testing.assert_array_equal(
            decoded.qp_backward_error_by_drive[drive],
            np.full(reference.T_bath.size, 1.0e-8),
        )
        np.testing.assert_array_equal(
            decoded.qp_number_backward_error_by_drive[drive],
            np.full(reference.T_bath.size, 1.0e-9),
        )
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


def test_atomic_writer_preserves_target_on_exception(tmp_path: Path) -> None:
    path = tmp_path / "existing.csv"
    path.write_text("old", encoding="utf-8")
    with (
        pytest.raises(RuntimeError, match="abort"),
        _atomic_text_file(path) as fp,
    ):
        fp.write("new")
        raise RuntimeError("abort")
    assert path.read_text(encoding="utf-8") == "old"
    assert not list(tmp_path.glob(f".{path.name}.*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    ["malformed", "duplicate", "missing", "nonfinite", "negative", "over_gate"],
)
def test_reader_rejects_invalid_artifacts(
    tmp_path: Path,
    mutation: str,
    _synthetic_certificate: None,
) -> None:
    path = _write_baseline(_synthetic_result(), tmp_path / "fig8_paper.csv")

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
        else:
            metadata = json.loads(rows[1][0].split("=", 1)[1])
            metadata["certificate_points"][0]["qp_backward_error"] = (
                2.0 * TARGET_QP_BACKWARD_ERROR_LIMIT
            )
            rows[1][0] = "# qpsim_metadata=" + json.dumps(metadata)

    _rewrite_csv(path, mutate)
    with pytest.raises(ArtifactValidationError) as captured:
        _read_summary(path)
    assert not isinstance(captured.value, LegacyArtifactError)


def test_writer_reassembles_and_rejects_forged_certificate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match=r"QP (backward error|residual_inf)"):
        _write_baseline(_synthetic_result(), tmp_path / "forged.csv")


def test_temperature_reset_and_full_state_strong_to_weak_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target, "NUM_BINS", 90)
    monkeypatch.setattr(target, "T_BATH_VALUES", np.asarray([0.10, 0.20]))
    calls: list[tuple[T3DiffusionState, T3DiffusionState, dict[str, float]]] = []

    class FakeBackend:
        def steady_state(
            self,
            state: T3DiffusionState,
            **kwargs: object,
        ) -> T3DiffusionState:
            returned = replace(
                state,
                f=np.full_like(state.f, 1.0e-4 * (len(calls) + 1)),
            )
            params = kwargs["pb_photon_params"]
            assert isinstance(params, dict)
            calls.append((state, returned, params))
            return returned

    monkeypatch.setattr(target, "T3DiffusionBackend", FakeBackend)
    monkeypatch.setattr(
        target,
        "qp_certificate",
        lambda *_args, **_kwargs: QPCertificate(1.0e-8, 1.0e-12, 1.0e-9),
    )

    target.run()
    assert len(calls) == 2 * len(target.PAPER_DRIVES_HZ)
    per_temperature = len(target.PAPER_DRIVES_HZ)
    for offset in (0, per_temperature):
        for index in range(1, per_temperature):
            assert calls[offset + index][0] is calls[offset + index - 1][1]
    assert calls[per_temperature][0] is not calls[per_temperature - 1][1]
    assert calls[0][0].T_bath == 0.10
    assert calls[per_temperature][0].T_bath == 0.20
    products = [call[2]["c_phot_PB"] * call[2]["n_bar_PB"] for call in calls[:per_temperature]]
    np.testing.assert_allclose(
        products,
        [d * target.HZ_TO_NS_INV for d in target.PAPER_DRIVES_HZ],
        rtol=0.0,
        atol=0.0,
    )


def test_reduced_sweep_returns_independently_certified_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(target, "NUM_BINS", 90)
    monkeypatch.setattr(target, "T_BATH_VALUES", np.asarray([0.10]))
    monkeypatch.setattr(target, "PAPER_DRIVES_HZ", (1.0e-2,))
    result = target.run()
    backward = result.qp_backward_error_by_drive[1.0e-2][0]
    residual = result.qp_residual_inf_by_drive[1.0e-2][0]
    assert np.isfinite(backward) and backward <= TARGET_QP_BACKWARD_ERROR_LIMIT
    assert np.isfinite(residual) and residual <= TARGET_QP_RESIDUAL_INF_LIMIT
    decoded = _read_summary(
        _write_baseline(result, tmp_path / "reduced_fig8.csv")
    )
    np.testing.assert_allclose(
        decoded.x_qp_num_by_drive[1.0e-2],
        result.x_qp_num_by_drive[1.0e-2],
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
        _read_summary(legacy_path)


def test_promoted_canonical_is_current_summary_with_producer_claims() -> None:
    path = target.baseline_path()
    assert path.is_file(), f"Promoted current-schema canonical is missing at {path}."
    baseline = _read_summary(path)
    np.testing.assert_array_equal(baseline.T_bath, target.T_BATH_VALUES)
    assert baseline.drives_hz == target.PAPER_DRIVES_HZ


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = target.baseline_path()
    if not path.exists():
        pytest.xfail(f"Certified baseline not found at {path}.")
    try:
        baseline = _read_summary(path)
    except LegacyArtifactError as exc:
        pytest.xfail(f"Legacy uncertified canonical baseline is quarantined: {exc}")

    result = target.run()
    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    assert result.drives_hz == baseline.drives_hz
    np.testing.assert_allclose(
        result.drives_ns_inv,
        baseline.drives_ns_inv,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.x_qp_thermal,
        baseline.x_qp_thermal,
        rtol=1e-6,
        atol=0.0,
    )
    for drive in result.drives_hz:
        np.testing.assert_allclose(
            result.x_qp_num_by_drive[drive],
            baseline.x_qp_num_by_drive[drive],
            rtol=1e-6,
            atol=1e-14,
        )
