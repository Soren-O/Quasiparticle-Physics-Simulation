"""Synthetic tests for the noncanonical Fischer Fig. 6 signed renderer."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
from validation.fischer_2023 import fig6_paper

from scripts import render_fischer_fig6_signed_diagnostic as diagnostic


def _result() -> Any:
    x = np.asarray(
        [
            [0.15, 0.30, 0.50, 0.70],
            [0.15, 0.30, 0.50, 0.70],
        ]
    )
    return SimpleNamespace(
        T_bath=np.asarray([0.10, 0.20]),
        n_bar=np.asarray([1.0, 2.0, 3.0, 4.0]),
        T_star_over_delta=x,
        paper_observable_num=np.asarray(
            [
                [0.10, 0.20, -2.0, -10.0],
                [0.05, 0.10, 0.20, 0.30],
            ]
        ),
        paper_observable_eq53=np.asarray(
            [
                [0.12, 0.22, -3.0, -12.0],
                [0.06, 0.11, 0.21, 0.31],
            ]
        ),
    )


def test_direct_script_invocation_bootstraps_this_checkout(tmp_path: Path) -> None:
    """A foreign cwd must not resolve a stale installed validation package."""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(diagnostic.__file__).resolve()),
            "--help",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Noncanonical output PDF path" in completed.stdout


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolate_synthetic_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep synthetic renderer tests independent of the real canonical tuple."""

    monkeypatch.setattr(
        diagnostic,
        "_require_canonical_current_unlocked",
        lambda _authenticated: None,
    )
    monkeypatch.setattr(fig6_paper, "_publication_lock", nullcontext)


def _authenticated_snapshot(
    tmp_path: Path,
    *,
    run_identity_character: str,
) -> diagnostic.AuthenticatedCanonical:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    return diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {
                "run_identity": run_identity_character * 64,
            },
        },
    )


def test_runtime_capture_selects_agg_before_recording_nondefault_backend() -> None:
    """An ambient GUI/vector default must not be stamped as the used backend."""

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "svg"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from scripts import "
                "render_fischer_fig6_signed_diagnostic as diagnostic; "
                "print(diagnostic._renderer_runtime_identity()"
                "['matplotlib']['backend'])"
            ),
        ],
        cwd=Path(diagnostic.__file__).resolve().parents[1],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().lower() == "agg"


def test_stats_report_clipping_and_extrema() -> None:
    stats = diagnostic.compute_plot_stats(_result())

    numerical = cast(dict[str, Any], stats["numerical"])
    analytic = cast(dict[str, Any], stats["analytic_eq53"])
    assert numerical["finite_samples"] == 8
    assert numerical["visible_in_paper_window"] == 3
    assert numerical["clipped_from_paper_window"] == 5
    assert numerical["minimum"] == -10.0
    assert numerical["maximum"] == 0.30
    assert analytic["finite_samples"] == 8
    assert analytic["visible_in_paper_window"] == 3
    assert analytic["clipped_from_paper_window"] == 5
    assert analytic["minimum"] == -12.0
    assert analytic["maximum"] == 0.31


def test_figure_contains_all_markers_and_visible_disclosure() -> None:
    stats = diagnostic.compute_plot_stats(_result())
    figure, render = diagnostic.build_figure(_result(), stats)
    try:
        axes = figure.axes[0]
        numerical_markers = sum(
            len(line.get_xdata())
            for line in axes.lines
            if line.get_marker() == "o" and line.get_linestyle() == "None"
        )
        analytic_markers = sum(
            len(line.get_xdata())
            for line in axes.lines
            if line.get_marker() == "x" and line.get_linestyle() == "None"
        )
        visible_text = "\n".join(text.get_text() for text in figure.texts)
        assert numerical_markers == 8
        assert analytic_markers == 8
        assert render["raw_numerical_markers"] == 8
        assert render["raw_analytic_markers"] == 8
        assert "qpsim self-regression, no digitized-data parity" in visible_text
        assert "hides 5/8 numerical and 5/8 Eq. 53 samples" in visible_text
        assert "PCHIP lines are visual interpolation only" in visible_text
        assert "[-10, 0.3]" in visible_text
    finally:
        plt.close(figure)


def test_nonfinite_samples_are_counted_and_disclosed_but_not_plotted() -> None:
    result = _result()
    result.paper_observable_num = result.paper_observable_num.copy()
    result.paper_observable_eq53 = result.paper_observable_eq53.copy()
    result.paper_observable_num[0, 0] = np.nan
    result.paper_observable_eq53[1, 1] = np.nan
    stats = diagnostic.compute_plot_stats(result)
    figure, render = diagnostic.build_figure(result, stats)
    try:
        axes = figure.axes[0]
        numerical_markers = sum(
            len(line.get_xdata())
            for line in axes.lines
            if line.get_marker() == "o" and line.get_linestyle() == "None"
        )
        analytic_markers = sum(
            len(line.get_xdata())
            for line in axes.lines
            if line.get_marker() == "x" and line.get_linestyle() == "None"
        )
        visible_text = "\n".join(text.get_text() for text in figure.texts)
        assert numerical_markers == 7
        assert analytic_markers == 7
        assert render["raw_numerical_markers"] == 7
        assert render["raw_analytic_markers"] == 7
        assert stats["numerical"]["nonfinite_samples"] == 1  # type: ignore[index]
        assert stats["analytic_eq53"]["nonfinite_samples"] == 1  # type: ignore[index]
        assert "1 numerical; 1 Eq. 53" in visible_text
        assert "recorded, not plottable" in visible_text
    finally:
        plt.close(figure)


def test_authenticated_load_binds_reader_record_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "canonical.csv"
    pdf_path = tmp_path / "canonical.pdf"
    record_path = tmp_path / "canonical.promotion.json"
    csv_path.write_bytes(b"certified-state-csv")
    pdf_path.write_bytes(b"certified-plot-pdf")
    record_path.write_bytes(b'{"authenticated":true}\n')
    record = {
        "schema": "promotion-test-v1",
        "artifacts": {
            "csv": {
                "sha256": _sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            }
        },
        "generation": {"run_identity": "a" * 64},
    }
    calls: list[str] = []

    def read_record(**_kwargs: object) -> dict[str, Any]:
        calls.append("record")
        return record

    def read_result(_path: Path) -> tuple[Any, dict[str, object]]:
        calls.append("reader")
        return _result(), {}

    monkeypatch.setattr(
        diagnostic,
        "_read_bound_promotion_record_unlocked",
        read_record,
    )
    monkeypatch.setattr(
        fig6_paper,
        "_read_artifact",
        read_result,
    )
    monkeypatch.setattr(
        fig6_paper,
        "validate_generation_evidence",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(fig6_paper, "_publication_lock", nullcontext)
    monkeypatch.setattr(fig6_paper, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fig6_paper, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(
        fig6_paper,
        "promotion_record_path",
        lambda: record_path,
    )

    authenticated = diagnostic.load_authenticated_canonical()

    assert calls == ["record", "reader"]
    assert authenticated.csv_identity["sha256"] == _sha256(csv_path)
    assert authenticated.promotion_identity["sha256"] == _sha256(record_path)
    assert authenticated.promotion_record == record


def test_bound_record_rejects_mixed_canonical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "canonical.csv"
    pdf_path = tmp_path / "canonical.pdf"
    record_path = tmp_path / "canonical.promotion.json"
    csv_path.write_bytes(b"certified-state-csv")
    pdf_path.write_bytes(b"certified-plot-pdf")
    record = {
        "artifact_schema": fig6_paper.ARTIFACT_SCHEMA,
        "artifacts": {
            "csv": fig6_paper._file_identity(csv_path),
            "pdf": fig6_paper._file_identity(pdf_path),
        },
        "generation": {"synthetic": True},
        "schema": fig6_paper.PROMOTION_RECORD_SCHEMA,
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(fig6_paper, "_require_valid_pdf", lambda _path: None)
    monkeypatch.setattr(
        fig6_paper,
        "validate_generation_evidence",
        lambda *_args, **_kwargs: {},
    )

    assert diagnostic._read_bound_promotion_record_unlocked(
        result=None,
        csv_path=csv_path,
        pdf_path=pdf_path,
        promotion_path=record_path,
    ) == record

    csv_path.write_bytes(b"mixed-producer-state-csv")
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="CSV does not match",
    ):
        diagnostic._read_bound_promotion_record_unlocked(
            result=None,
            csv_path=csv_path,
            pdf_path=pdf_path,
            promotion_path=record_path,
        )


def test_cached_canonical_currentness_recheck_refuses_drift_without_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "canonical.csv"
    pdf_path = tmp_path / "canonical.pdf"
    promotion_path = tmp_path / "canonical.promotion.json"
    csv_path.write_bytes(b"certified-state-csv")
    pdf_path.write_bytes(b"certified-plot-pdf")
    promotion_path.write_bytes(b'{"promotion":"old"}\n')
    record = {
        "schema": "promotion-test-v1",
        "artifacts": {
            "csv": {
                "sha256": _sha256(csv_path),
                "size_bytes": csv_path.stat().st_size,
            }
        },
        "generation": {"run_identity": "1" * 64},
    }
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=csv_path,
        promotion_path=promotion_path,
        csv_identity=diagnostic._file_identity(csv_path),
        promotion_identity=diagnostic._file_identity(promotion_path),
        promotion_record=record,
    )
    validated_results: list[Any] = []

    def read_bound(*, result: Any, **_kwargs: object) -> dict[str, Any]:
        validated_results.append(result)
        return record

    monkeypatch.setattr(fig6_paper, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fig6_paper, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(
        fig6_paper,
        "promotion_record_path",
        lambda: promotion_path,
    )
    monkeypatch.setattr(
        diagnostic,
        "_read_bound_promotion_record_unlocked",
        read_bound,
    )
    monkeypatch.setattr(
        fig6_paper,
        "_read_artifact",
        lambda _path: pytest.fail("canonical currentness recheck replayed states"),
    )

    diagnostic._require_canonical_current_unlocked(authenticated)
    assert validated_results == [authenticated.result]

    csv_path.write_bytes(b"new-canonical-state-csv")
    with pytest.raises(
        diagnostic.DiagnosticAuthenticationError,
        match="changed while rendering",
    ):
        diagnostic._require_canonical_current_unlocked(authenticated)
    assert validated_results == [authenticated.result]


def test_overlapping_output_pairs_contend_on_each_shared_resource(
    tmp_path: Path,
) -> None:
    shared_pdf = tmp_path / "shared.pdf"
    sidecar_a = tmp_path / "a.json"
    sidecar_b = tmp_path / "b.json"
    other_pdf = tmp_path / "other.pdf"

    with diagnostic._diagnostic_publication_lock(shared_pdf, sidecar_a):
        with (
            pytest.raises(
                diagnostic.DiagnosticPublicationError,
                match="overlapping output lock",
            ),
            diagnostic._diagnostic_publication_lock(shared_pdf, sidecar_b),
        ):
            pytest.fail("a shared PDF unexpectedly acquired a second lock")
        with (
            pytest.raises(
                diagnostic.DiagnosticPublicationError,
                match="overlapping output lock",
            ),
            diagnostic._diagnostic_publication_lock(other_pdf, sidecar_a),
        ):
            pytest.fail("a shared sidecar unexpectedly acquired a second lock")


def test_generate_locks_outputs_before_any_stage_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent same-process calls must not share or unlink stage files."""

    authenticated = _authenticated_snapshot(
        tmp_path,
        run_identity_character="9",
    )
    authenticated_reads: list[bool] = []

    def load_authenticated() -> diagnostic.AuthenticatedCanonical:
        authenticated_reads.append(True)
        return authenticated

    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        load_authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    real_create_stage = diagnostic._create_unique_stage_path
    first_stage_entered = threading.Event()
    release_first_stage = threading.Event()
    stage_calls: list[Path] = []

    def blocking_create_stage(target: Path, *, suffix: str) -> Path:
        stage_calls.append(target)
        if len(stage_calls) == 1:
            first_stage_entered.set()
            if not release_first_stage.wait(timeout=10):
                raise TimeoutError("test did not release the first staged publisher")
        return real_create_stage(target, suffix=suffix)

    monkeypatch.setattr(
        diagnostic,
        "_create_unique_stage_path",
        blocking_create_stage,
    )
    pdf_path = tmp_path / "shared.pdf"
    json_path = tmp_path / "shared.json"
    worker_errors: list[BaseException] = []

    def first_publisher() -> None:
        try:
            diagnostic.generate_diagnostic(
                pdf_path=pdf_path,
                json_path=json_path,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            worker_errors.append(exc)

    worker = threading.Thread(target=first_publisher)
    worker.start()
    try:
        assert first_stage_entered.wait(timeout=10)
        with pytest.raises(
            diagnostic.DiagnosticPublicationError,
            match="overlapping output lock",
        ):
            diagnostic.generate_diagnostic(
                pdf_path=pdf_path,
                json_path=json_path,
            )
    finally:
        release_first_stage.set()
        worker.join(timeout=30)

    assert not worker.is_alive()
    assert worker_errors == []
    assert authenticated_reads == [True]
    assert stage_calls == [pdf_path, json_path]


def test_stage_guard_rejects_alias_with_final_or_lock_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    other_stage = tmp_path / ".other.unique.stage.json"

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="alias a final output",
    ):
        diagnostic._guard_stage_paths(
            pdf_path,
            json_path,
            pdf_stage=json_path,
            json_stage=other_stage,
        )

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="alias a final output",
    ):
        diagnostic._guard_stage_paths(
            pdf_path,
            json_path,
            pdf_stage=diagnostic._diagnostic_resource_lock_path(pdf_path),
            json_stage=other_stage,
        )


def test_authentication_failure_is_loud_and_never_recertifies_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads: list[bool] = []

    def stale_record(**_kwargs: object) -> dict[str, object]:
        raise fig6_paper.ArtifactValidationError("stale/tampered")

    monkeypatch.setattr(
        diagnostic,
        "_read_bound_promotion_record_unlocked",
        stale_record,
    )
    monkeypatch.setattr(
        fig6_paper,
        "_read_artifact",
        lambda _path: reads.append(True),
    )
    monkeypatch.setattr(fig6_paper, "_publication_lock", nullcontext)
    with pytest.raises(
        diagnostic.DiagnosticAuthenticationError,
        match="authentication/currentness",
    ):
        diagnostic.load_authenticated_canonical()
    assert reads == []


def test_generate_diagnostic_writes_bound_sidecar_without_solving(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "b" * 64},
        },
    )
    authenticated_reads: list[bool] = []

    def read_authenticated() -> diagnostic.AuthenticatedCanonical:
        authenticated_reads.append(True)
        return authenticated

    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        read_authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    monkeypatch.setattr(
        fig6_paper,
        "run",
        lambda *_args, **_kwargs: pytest.fail("renderer called expensive run()"),
    )
    monkeypatch.setattr(
        fig6_paper,
        "run_cached",
        lambda *_args, **_kwargs: pytest.fail("renderer called run_cached()"),
    )
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"

    produced_pdf, produced_json = diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    assert produced_pdf == pdf_path
    assert produced_json == json_path
    assert len(authenticated_reads) == 1
    assert pdf_path.read_bytes().startswith(b"%PDF")
    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    assert evidence["schema"] == diagnostic.SCHEMA
    assert evidence["mode"] == "noncanonical-signed-diagnostic"
    assert evidence["claim_scope"] == (
        "qpsim self-regression, no digitized-data parity"
    )
    assert evidence["canonical"]["csv"]["sha256"] == _sha256(source_csv)
    assert evidence["canonical"]["promotion_record"]["sha256"] == _sha256(
        source_record
    )
    assert evidence["renderer"]["sha256"] == _sha256(Path(diagnostic.__file__))
    assert evidence["runtime"] == diagnostic._renderer_runtime_identity()
    assert evidence["output_pdf"]["sha256"] == _sha256(pdf_path)
    assert evidence["stats"]["numerical"]["clipped_from_paper_window"] == 5
    assert evidence["render"]["raw_numerical_markers"] == 8
    assert (
        diagnostic._render_commitment(
            evidence["render"],
            evidence["runtime"],
            evidence["created_utc"],
        ).encode("ascii")
        in pdf_path.read_bytes()
    )
    assert diagnostic.read_diagnostic_record(
        pdf_path=pdf_path,
        json_path=json_path,
    ) == evidence
    assert len(authenticated_reads) == 2


def test_generate_rechecks_canonical_while_publication_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "7" * 64},
        },
    )
    lock_held = False
    rechecks: list[bool] = []

    @contextmanager
    def fake_canonical_lock() -> Iterator[None]:
        nonlocal lock_held
        assert not lock_held
        lock_held = True
        try:
            yield
        finally:
            lock_held = False

    def require_current(_authenticated: diagnostic.AuthenticatedCanonical) -> None:
        assert lock_held
        rechecks.append(True)

    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    monkeypatch.setattr(fig6_paper, "_publication_lock", fake_canonical_lock)
    monkeypatch.setattr(
        diagnostic,
        "_require_canonical_current_unlocked",
        require_current,
    )

    diagnostic.generate_diagnostic(
        pdf_path=tmp_path / "signed.pdf",
        json_path=tmp_path / "signed.json",
    )
    assert rechecks == [True]
    assert not lock_held


def test_generate_refuses_renderer_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "8" * 64},
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)

    renderer_path = Path(diagnostic.__file__).resolve()
    real_identity = diagnostic._file_identity
    renderer_reads = 0

    def drifting_identity(path: Path) -> dict[str, object]:
        nonlocal renderer_reads
        identity = real_identity(path)
        if path.resolve() == renderer_path:
            renderer_reads += 1
            if renderer_reads > 1:
                identity["sha256"] = "0" * 64
        return identity

    monkeypatch.setattr(diagnostic, "_file_identity", drifting_identity)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="renderer source changed",
    ):
        diagnostic.generate_diagnostic(
            pdf_path=pdf_path,
            json_path=json_path,
        )
    assert renderer_reads == 2
    assert not pdf_path.exists()
    assert not json_path.exists()


def test_reader_binds_pchip_claim_to_pdf_render_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "e" * 64},
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
        interpolate=True,
    )

    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    # This is internally plausible for a marker-only render, but it does not
    # describe the PDF that was actually generated with PCHIP enabled.
    evidence["render"]["pchip_interpolation_requested"] = False
    evidence["render"]["pchip_curves_rendered"] = 0
    json_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="render commitment",
    ):
        diagnostic.read_diagnostic_record(
            pdf_path=pdf_path,
            json_path=json_path,
        )


def test_reader_binds_runtime_claim_to_pdf_render_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "f" * 64},
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    evidence["runtime"]["scipy"] = "forged-runtime"
    json_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="render commitment",
    ):
        diagnostic.read_diagnostic_record(
            pdf_path=pdf_path,
            json_path=json_path,
        )


def test_reader_binds_creation_time_to_pdf_render_commitment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authenticated = _authenticated_snapshot(
        tmp_path,
        run_identity_character="1",
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    evidence["created_utc"] = "2001-02-03T04:05:06+00:00"
    json_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="render commitment",
    ):
        diagnostic.read_diagnostic_record(
            pdf_path=pdf_path,
            json_path=json_path,
        )


@pytest.mark.parametrize(
    "invalid_created_utc",
    (
        "2026-07-28T12:00:00",
        "2026-07-28T13:00:00+01:00",
    ),
)
def test_reader_requires_timezone_aware_utc_creation_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_created_utc: str,
) -> None:
    authenticated = _authenticated_snapshot(
        tmp_path,
        run_identity_character="2",
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    evidence = json.loads(json_path.read_text(encoding="utf-8"))
    evidence["created_utc"] = invalid_created_utc
    json_path.write_text(
        json.dumps(evidence, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="timezone-aware UTC",
    ):
        diagnostic.read_diagnostic_record(
            pdf_path=pdf_path,
            json_path=json_path,
        )


def test_json_is_last_commit_marker_and_reader_rejects_mixed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "c" * 64},
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    real_replace = os.replace
    promotions: list[Path] = []

    def tracked_replace(source: str | Path, destination: str | Path) -> None:
        target = Path(destination)
        if target in {pdf_path, json_path}:
            promotions.append(target)
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracked_replace)
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    assert promotions[-2:] == [pdf_path, json_path]
    diagnostic.read_diagnostic_record(
        pdf_path=pdf_path,
        json_path=json_path,
    )

    # This is the observable state an abrupt interruption after the first
    # promotion can leave: a new PDF beneath an old JSON commit marker.
    # It must be loud, never an accepted mixed pair.
    pdf_path.write_bytes(pdf_path.read_bytes() + b"\n% interrupted replacement\n")
    with pytest.raises(
        diagnostic.DiagnosticPublicationError,
        match="does not match",
    ):
        diagnostic.read_diagnostic_record(
            pdf_path=pdf_path,
            json_path=json_path,
        )


def test_failed_commit_marker_promotion_restores_previous_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_csv = tmp_path / "source.csv"
    source_record = tmp_path / "source.promotion.json"
    source_csv.write_bytes(b"source-csv")
    source_record.write_bytes(b"source-record")
    authenticated = diagnostic.AuthenticatedCanonical(
        result=_result(),
        csv_path=source_csv,
        promotion_path=source_record,
        csv_identity=diagnostic._file_identity(source_csv),
        promotion_identity=diagnostic._file_identity(source_record),
        promotion_record={
            "schema": "promotion-test-v1",
            "generation": {"run_identity": "d" * 64},
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "load_authenticated_canonical",
        lambda: authenticated,
    )
    _isolate_synthetic_publication(monkeypatch)
    pdf_path = tmp_path / "signed.pdf"
    json_path = tmp_path / "signed.json"
    diagnostic.generate_diagnostic(
        pdf_path=pdf_path,
        json_path=json_path,
    )
    old_pdf = pdf_path.read_bytes()
    old_json = json_path.read_bytes()
    real_replace = os.replace

    def fail_commit_marker(
        source: str | Path,
        destination: str | Path,
    ) -> None:
        source_path = Path(source)
        if (
            Path(destination) == json_path
            and source_path.name.endswith(".stage.json")
        ):
            raise OSError("simulated commit-marker interruption")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_commit_marker)
    with pytest.raises(OSError, match="commit-marker interruption"):
        diagnostic.generate_diagnostic(
            pdf_path=pdf_path,
            json_path=json_path,
            interpolate=False,
        )

    assert pdf_path.read_bytes() == old_pdf
    assert json_path.read_bytes() == old_json
    diagnostic.read_diagnostic_record(
        pdf_path=pdf_path,
        json_path=json_path,
    )
    assert list(tmp_path.glob(".*.stage.*")) == []


def test_default_output_is_noncanonical_tmp_and_canonical_path_is_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    default_pdf, default_json = diagnostic.output_paths()
    assert default_pdf.parent.name == "tmp"
    assert default_json == default_pdf.with_suffix(".json")

    canonical_pdf = tmp_path / "canonical.pdf"
    monkeypatch.setattr(
        fig6_paper,
        "baseline_path",
        lambda: tmp_path / "canonical.csv",
    )
    monkeypatch.setattr(fig6_paper, "plot_path", lambda: canonical_pdf)
    monkeypatch.setattr(
        fig6_paper,
        "promotion_record_path",
        lambda: tmp_path / "canonical.promotion.json",
    )
    with pytest.raises(ValueError, match="canonical"):
        diagnostic.output_paths(canonical_pdf, tmp_path / "diagnostic.json")
