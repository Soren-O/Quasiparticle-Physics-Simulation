"""Artifact-pair promotion tests for the Fischer 2023 Fig. 7 pipeline."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023 import fig7_paper, fig7_solve


def _full_result() -> fig7_paper.Fig7PaperResult:
    temperatures = np.asarray(fig7_solve.T_BATH_VALUES, dtype=float)
    powers = tuple(float(power) for power in fig7_solve.P_READ_DBM)
    shape = (len(powers), temperatures.size)
    sigma1_by_dbm: dict[float, np.ndarray] = {}
    Q_qp_by_dbm: dict[float, np.ndarray] = {}
    Q_tot_by_dbm: dict[float, np.ndarray] = {}
    n_bar_by_dbm: dict[float, float] = {}
    for pi, power in enumerate(powers):
        sigma1 = np.full(temperatures.shape, 1.0e-7 * (pi + 1))
        Q_qp = (
            np.pi
            * fig7_solve.DELTA_0
            / (fig7_solve.OMEGA_0 * fig7_paper.ALPHA_KI * sigma1)
        )
        sigma1_by_dbm[power] = sigma1
        Q_qp_by_dbm[power] = Q_qp
        Q_tot_by_dbm[power] = 1.0 / (
            1.0 / Q_qp + 1.0 / fig7_paper.Q_EXT_BY_DBM[power]
        )
        n_bar_by_dbm[power] = fig7_solve._nbar_from_table_iii(power)
    certificates = {
        field: np.full(shape, (index + 1) * 1.0e-12)
        for index, field in enumerate(fig7_solve.NUMBER_CERTIFICATE_FIELDS)
    }
    return fig7_paper.Fig7PaperResult(
        T_bath=temperatures,
        p_read_dbm=powers,
        n_bar_by_dbm=n_bar_by_dbm,
        Q_qp_by_dbm=Q_qp_by_dbm,
        Q_tot_by_dbm=Q_tot_by_dbm,
        sigma1_by_dbm=sigma1_by_dbm,
        qp_residual_inf=certificates["qp_residual_inf"],
        qp_backward_error=certificates["qp_backward_error"],
        phonon_residual_inf=certificates["phonon_residual_inf"],
        phonon_raw_backward_error=certificates["phonon_raw_backward_error"],
        phonon_backward_error=certificates["phonon_backward_error"],
        qp_number_backward_error=certificates[
            fig7_solve.QP_NUMBER_CERTIFICATE_FIELD
        ],
    )


def _producer() -> tuple[fig7_paper.Fig7ProducerIdentity, Path]:
    runner = Path(fig7_paper.__file__).resolve()
    return fig7_paper.capture_producer_identity(runner), runner


def test_complete_pdf_record_rejects_token_shaped_blank_pdf(
    tmp_path: Path,
) -> None:
    path = tmp_path / "token-shaped-blank.pdf"
    prefix = (
        b"%PDF-1.4\n"
        b"/Type /Catalog\n"
        b"/Type /Pages\n"
        b"/Type /Page\n"
    )
    path.write_bytes(
        prefix
        + b"xref\n"
        + b"startxref\n"
        + str(len(prefix)).encode("ascii")
        + b"\n%%EOF"
    )

    with pytest.raises(RuntimeError, match="one-page Matplotlib PDF"):
        fig7_paper.complete_pdf_record(path)


def test_observable_contract_fingerprints_shared_pdf_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visited: list[Path] = []
    original = fig7_paper.source_provenance.canonical_source_bytes

    def tracking_source(path: Path) -> bytes:
        visited.append(path.resolve())
        return original(path)

    monkeypatch.setattr(
        fig7_paper.source_provenance,
        "canonical_source_bytes",
        tracking_source,
    )
    assert len(fig7_paper.observable_contract_digest()) == 64
    assert Path(fig7_paper._pdf_validation.__file__).resolve() in visited


def test_shared_publisher_binds_pair_and_writes_durable_attestation(
    tmp_path: Path,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    worker_hashes = {
        (pi, ti): hashlib_value
        for pi in range(len(fig7_solve.P_READ_DBM))
        for ti in range(len(fig7_solve.T_BATH_VALUES))
        for hashlib_value in [
            f"{pi:02x}{ti:02x}" + "a" * 60
        ]
    }

    promoted = fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
        worker_payload_hashes=worker_hashes,
        campaign={"mode": "unit-test"},
    )
    assert promoted == (csv_path, pdf_path, attestation_path)
    restored = fig7_paper.read_baseline(
        csv_path,
        companion_pdf_path=pdf_path,
        require_companion_pdf=True,
        accept_producer_certificate_claims=True,
    )
    fig7_paper._assert_same_result(result, restored)
    attestation = fig7_paper.validate_promotion_attestation(
        attestation_path,
        csv_path=csv_path,
        pdf_path=pdf_path,
        expected_producer=producer,
    )
    assert attestation["campaign"] == {"mode": "unit-test"}
    assert (
        attestation["certificate_scope"]
        == fig7_paper.SUMMARY_CERTIFICATE_SCOPE
    )
    assert len(attestation["point_hashes"]) == 48


def test_publisher_rolls_back_all_canonical_bytes_on_csv_replace_failure(
    tmp_path: Path,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    previous = {
        csv_path: b"old-csv",
        pdf_path: b"old-pdf",
        attestation_path: b"old-attestation",
    }
    for path, payload in previous.items():
        path.write_bytes(payload)
    calls = 0

    def fail_second_replace(source: Path, destination: Path) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated CSV promotion failure")
        return os.replace(source, destination)

    with pytest.raises(OSError, match="CSV promotion"):
        fig7_paper.publish_baseline_pair(
            result,
            producer=producer,
            runner_path=runner,
            csv_path=csv_path,
            pdf_path=pdf_path,
            attestation_path=attestation_path,
            replace_file=fail_second_replace,
        )
    assert calls == 2
    for path, payload in previous.items():
        assert path.read_bytes() == payload
    assert not list(tmp_path.glob(".*.rollback.*"))


def test_pdf_and_attestation_corruption_are_rejected(tmp_path: Path) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )

    original_pdf = pdf_path.read_bytes()
    pdf_path.write_bytes(original_pdf + b"trailing-corruption")
    with pytest.raises(RuntimeError, match="one-page Matplotlib PDF"):
        fig7_paper.read_baseline(
            csv_path,
            companion_pdf_path=pdf_path,
            require_companion_pdf=True,
            accept_producer_certificate_claims=True,
        )
    pdf_path.write_bytes(original_pdf)

    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["artifacts"]["csv"]["sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="authenticate csv"):
        fig7_paper.validate_promotion_attestation(
            attestation_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            expected_producer=producer,
        )


def test_attestation_rejects_wrong_producer_certificate_scope(
    tmp_path: Path,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["certificate_scope"] = "independently-reassembled"
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="producer-certificate scope"):
        fig7_paper.validate_promotion_attestation(
            attestation_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            expected_producer=producer,
        )


def _make_paths_canonical(
    monkeypatch: pytest.MonkeyPatch,
    *,
    csv_path: Path,
    pdf_path: Path,
    attestation_path: Path,
) -> None:
    monkeypatch.setattr(fig7_paper, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fig7_paper, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(
        fig7_paper,
        "promotion_attestation_path",
        lambda _csv_path=None: attestation_path,
    )


def test_canonical_read_requires_an_intact_promotion_attestation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    _make_paths_canonical(
        monkeypatch,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    with pytest.raises(RuntimeError, match="producer assertions only"):
        fig7_paper.read_baseline()
    fig7_paper.read_baseline(accept_producer_certificate_claims=True)

    original = attestation_path.read_bytes()
    attestation_path.unlink()
    with pytest.raises(fig7_paper.LegacyArtifactError, match="no current"):
        fig7_paper.read_baseline(accept_producer_certificate_claims=True)
    attestation_path.write_bytes(b"{not-json")
    with pytest.raises(RuntimeError, match="Cannot read"):
        fig7_paper.read_baseline(accept_producer_certificate_claims=True)
    attestation_path.write_bytes(original)
    fig7_paper.read_baseline(accept_producer_certificate_claims=True)


def test_canonical_read_is_portable_across_consumer_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    payload = json.loads(attestation_path.read_text(encoding="utf-8"))
    payload["producer_runtime"] = {
        "python": "3.13.99",
        "platform": "foreign-hosted-runner",
        "numpy": "foreign",
        "scipy": "foreign",
    }
    attestation_path.write_text(json.dumps(payload), encoding="utf-8")
    _make_paths_canonical(
        monkeypatch,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )

    restored = fig7_paper.read_baseline(
        accept_producer_certificate_claims=True,
    )
    fig7_paper._assert_same_result(result, restored)
    assert restored.certificate_scope == fig7_paper.SUMMARY_CERTIFICATE_SCOPE


def test_canonical_readers_refuse_snapshot_while_publication_lock_is_held(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    _make_paths_canonical(
        monkeypatch,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )

    with fig7_paper._publication_lock(attestation_path):
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.read_baseline(
                accept_producer_certificate_claims=True,
            )
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.read_baseline_metadata()
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.validate_promotion_attestation(
                attestation_path,
                csv_path=csv_path,
                pdf_path=pdf_path,
            )
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.publish_baseline_pair(
                result,
                producer=producer,
                runner_path=runner,
                csv_path=csv_path,
                pdf_path=pdf_path,
                attestation_path=attestation_path,
            )


def test_publisher_holds_lock_across_every_canonical_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    _make_paths_canonical(
        monkeypatch,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    replacements = 0

    def guarded_replace(source: Path, destination: Path) -> object:
        nonlocal replacements
        replacements += 1
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.read_baseline(
                accept_producer_certificate_claims=True,
            )
        return os.replace(source, destination)

    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
        replace_file=guarded_replace,
    )
    assert replacements == 3


def test_cross_process_lock_excludes_canonical_reader_and_publisher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    _make_paths_canonical(
        monkeypatch,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
    )
    script = """
import sys
import time
from pathlib import Path
from validation.fischer_2023.fig7_paper import _publication_lock

attestation, ready, release = map(Path, sys.argv[1:4])
with _publication_lock(attestation):
    ready.write_text("ready", encoding="utf-8")
    while not release.exists():
        time.sleep(0.01)
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            str(attestation_path),
            str(ready),
            str(release),
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        deadline = time.monotonic() + 10.0
        while (
            not ready.exists()
            and process.poll() is None
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        assert ready.exists(), "child failed to acquire the Fig. 7 artifact lock"
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.read_baseline_metadata()
        with pytest.raises(RuntimeError, match="publisher or reader"):
            fig7_paper.publish_baseline_pair(
                result,
                producer=producer,
                runner_path=runner,
                csv_path=csv_path,
                pdf_path=pdf_path,
                attestation_path=attestation_path,
            )
    finally:
        release.write_text("release", encoding="utf-8")
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
    assert process.returncode == 0


def test_post_run_validation_independently_authenticates_worker_payloads(
    tmp_path: Path,
) -> None:
    result = _full_result()
    producer, runner = _producer()
    run_identity = "b" * 64
    run_dir = tmp_path / f"fig7-{run_identity}"
    run_dir.mkdir()
    worker_hashes: dict[tuple[int, int], str] = {}
    status_hashes: dict[str, str] = {}
    for pi in range(len(fig7_solve.P_READ_DBM)):
        for ti in range(len(fig7_solve.T_BATH_VALUES)):
            point_path = run_dir / f"point-p{pi:02d}-t{ti:02d}.npz"
            point_path.write_bytes(f"worker:{pi}:{ti}".encode())
            digest = fig7_paper.file_sha256(point_path)
            worker_hashes[(pi, ti)] = digest
            status_hashes[f"p{pi:02d}-t{ti:02d}"] = digest

    csv_path = tmp_path / "fig7.csv"
    pdf_path = tmp_path / "fig7.pdf"
    attestation_path = tmp_path / "fig7.promotion.json"
    campaign = {
        "mode": "parallel-independent-points",
        "run_identity": run_identity,
        "wall_s": 120.0,
        "aggregate_worker_s": 600.0,
        "resumed_points": 0,
        "new_points": 48,
    }
    fig7_paper.publish_baseline_pair(
        result,
        producer=producer,
        runner_path=runner,
        csv_path=csv_path,
        pdf_path=pdf_path,
        attestation_path=attestation_path,
        worker_payload_hashes=worker_hashes,
        campaign=campaign,
    )
    maxima = {
        field: float(np.max(np.asarray(getattr(result, field))))
        for field in fig7_solve.NUMBER_CERTIFICATE_FIELDS
    }
    status = {
        "schema": "qpsim-fischer-2023-fig7-run-v1",
        "state": "complete",
        "run_identity": run_identity,
        "solve_contract_digest": producer.solve_contract_digest,
        "observable_contract_digest": producer.observable_contract_digest,
        "runner_sha256": producer.runner_sha256,
        "producer_runtime": producer.runtime,
        "single_thread_environment": {
            "BLIS_NUM_THREADS": "1",
            "MKL_DYNAMIC": "FALSE",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
        "started_utc": "2026-07-25T00:00:00+00:00",
        "updated_utc": "2026-07-25T01:00:00+00:00",
        "attempt": 1,
        "completed_points": 48,
        "total_points": 48,
        "completion": {
            "csv": str(csv_path),
            "pdf": str(pdf_path),
            "attestation": str(attestation_path),
            "csv_sha256": fig7_paper.file_sha256(csv_path),
            "pdf_sha256": fig7_paper.file_sha256(pdf_path),
            "attestation_sha256": fig7_paper.file_sha256(attestation_path),
            "wall_s": campaign["wall_s"],
            "aggregate_worker_s": campaign["aggregate_worker_s"],
            "certificate_maxima": maxima,
            "point_payload_sha256": status_hashes,
        },
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    fig7_paper.validate_promotion_attestation(
        attestation_path,
        csv_path=csv_path,
        pdf_path=pdf_path,
        expected_producer=producer,
        run_dir=run_dir,
        status_path=status_path,
    )

    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["point_hashes"][0]["worker_payload_sha256"] = "0" * 64
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="worker payloads"):
        fig7_paper.validate_promotion_attestation(
            attestation_path,
            csv_path=csv_path,
            pdf_path=pdf_path,
            expected_producer=producer,
            run_dir=run_dir,
            status_path=status_path,
        )


def test_generic_generate_baseline_routes_through_shared_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _full_result()
    calls: dict[str, object] = {}
    fake_paths = (Path("out.csv"), Path("out.pdf"), Path("out.json"))

    monkeypatch.setattr(fig7_paper, "run_cached", lambda: result)

    def fake_publish(
        received: fig7_paper.Fig7PaperResult,
        **kwargs: object,
    ) -> tuple[Path, Path, Path]:
        calls["result"] = received
        calls["kwargs"] = kwargs
        return fake_paths

    monkeypatch.setattr(fig7_paper, "publish_baseline_pair", fake_publish)
    assert fig7_paper.generate_baseline() == fake_paths[:2]
    assert calls["result"] is result
    assert calls["kwargs"]["campaign"] == {
        "mode": "generic-cached-generate-baseline"
    }
