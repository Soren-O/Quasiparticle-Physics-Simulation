"""Portable byte-contract tests for the transient photon-kick artifact."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid
from qpsim.physics.spectral import fermi_dirac_occupation
from validation.transient import photon_kick_response as target
from validation.transient.photon_kick_response import (
    ARTIFACT_SCHEMA,
    NUM_BINS,
    SNAPSHOT_INTERVAL,
    TOTAL_TIME,
    PhotonKickArtifactError,
    PhotonKickResult,
    baseline_path,
    companion_artifact_record,
    plot_path,
    read_baseline,
    write_baseline,
)

_TEST_CERTIFICATE = {
    "metric": target.STEADY_STATE_CERTIFICATE_METRIC,
    "qp_backward_error": 1.0e-12,
    "qp_residual_inf": 1.0e-15,
    "qp_number_backward_error": 1.0e-12,
}


def _valid_result() -> PhotonKickResult:
    times = np.arange(
        0.0,
        TOTAL_TIME + 0.5 * SNAPSHOT_INTERVAL,
        SNAPSHOT_INTERVAL,
    )
    E, _ = build_energy_grid(
        gap=target.DELTA_0,
        energy_min_factor=target.E_MIN_FACTOR,
        energy_max_factor=target.E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    f_thermal = fermi_dirac_occupation(E, target.T_BATH)
    f_steady_state = np.maximum(f_thermal, 1.0e-2)
    progress = np.linspace(0.0, 0.995, times.size)
    f_snapshots = f_thermal + progress[:, None] * (f_steady_state - f_thermal)
    x_qp_snapshots, x_qp_steady_state = target._x_qp_from_occupations(
        E,
        f_snapshots,
        f_steady_state,
    )
    return PhotonKickResult(
        E=E,
        snapshot_times=times,
        f_snapshots=f_snapshots,
        x_qp_snapshots=x_qp_snapshots,
        f_steady_state=f_steady_state,
        x_qp_steady_state=x_qp_steady_state,
    )


def _write_real_pdf(result: PhotonKickResult, path: Path) -> None:
    target.write_plot(result, path)
    assert path.is_file()


def _rewrite_numeric_payload(
    path: Path,
    transform: Callable[[np.ndarray, np.ndarray], None],
) -> None:
    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    marker = "# qpsim_metadata="
    metadata = json.loads(rows[1][0][len(marker) :])
    data = np.asarray(
        [[float(value) for value in row] for row in rows[3:]],
        dtype=float,
    )
    times = np.asarray(metadata["snapshot_times_ns"], dtype=float)
    transform(data, times)
    f_snapshots = np.ascontiguousarray(data[:, 2:].T)
    x_qp, x_qp_ss = target._x_qp_from_occupations(
        data[:, 0],
        f_snapshots,
        data[:, 1],
    )
    metadata["x_qp_snapshots"] = [float(value) for value in x_qp]
    metadata["x_qp_steady_state"] = float(x_qp_ss)
    metadata["payload_sha256"] = target._payload_sha256(
        data,
        times,
        x_qp,
        float(x_qp_ss),
    )
    rows[1] = [f"{marker}{target._canonical_json(metadata)}"]
    rows[3:] = [[f"{float(value):.17e}" for value in row] for row in data]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerows(rows)


def test_canonical_photon_kick_baseline_is_utf8_lf() -> None:
    payload = baseline_path().read_bytes()

    text = payload.decode("utf-8")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert text.startswith(
        f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}\n"
    ), "Canonical transient artifact is legacy or has an unsupported schema."
    baseline = read_baseline()
    assert baseline.fingerprint
    assert baseline.trajectory_evidence_scope == target.TRAJECTORY_EVIDENCE_SCOPE
    assert plot_path().is_file()


def test_photon_kick_writer_is_utf8_lf_and_round_trips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    pdf_path = tmp_path / "photon_kick.pdf"
    _write_real_pdf(result, pdf_path)
    path = write_baseline(
        result,
        tmp_path / "photon_kick.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )

    payload = path.read_bytes()
    text = payload.decode("utf-8")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert text.startswith(f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}\n")

    baseline = read_baseline(
        path,
        companion_pdf_path=pdf_path,
        require_current=False,
    )
    np.testing.assert_array_equal(baseline.E, result.E)
    np.testing.assert_array_equal(baseline.snapshot_times, result.snapshot_times)
    np.testing.assert_array_equal(baseline.f_snapshots, result.f_snapshots)
    np.testing.assert_array_equal(baseline.x_qp_snapshots, result.x_qp_snapshots)
    np.testing.assert_array_equal(baseline.f_steady_state, result.f_steady_state)
    assert baseline.x_qp_steady_state == result.x_qp_steady_state
    assert baseline.trajectory_evidence_scope == target.TRAJECTORY_EVIDENCE_SCOPE

    payload_pdf = pdf_path.read_bytes()
    tampered_pdf = payload_pdf.replace(b"/Title", b"/Tytle", 1)
    assert tampered_pdf != payload_pdf
    pdf_path.write_bytes(tampered_pdf)
    with pytest.raises(PhotonKickArtifactError, match="does not authenticate"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_pair_publication_authenticates_pdf_and_promotes_csv_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "canonical.csv"
    pdf_path = tmp_path / "canonical.pdf"
    result = _valid_result()
    E = result.E
    monkeypatch.setattr(target, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(target, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    original_write_plot = target.write_plot

    def fake_write_plot(_result: PhotonKickResult, path: Path | None = None) -> Path:
        assert path is not None
        return original_write_plot(_result, path)

    monkeypatch.setattr(target, "write_plot", fake_write_plot)
    promoted: list[Path] = []
    original_replace = os.replace

    def tracking_replace(source: str | Path, destination: str | Path) -> None:
        destination_path = Path(destination)
        if destination_path in {csv_path, pdf_path}:
            promoted.append(destination_path)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracking_replace)
    assert target.publish_baseline_pair(
        result,
        fingerprint=target.solver_fingerprint(),
        producer_runtime=target.producer_runtime_provenance(),
    ) == (csv_path, pdf_path)
    assert promoted == [pdf_path, csv_path]
    restored = target.read_baseline()
    np.testing.assert_array_equal(restored.E, E)

    perturbed_E = E.copy()
    perturbed_E[1] = np.nextafter(perturbed_E[1], np.inf)
    bad_x_qp, bad_x_qp_ss = target._x_qp_from_occupations(
        perturbed_E,
        result.f_snapshots,
        result.f_steady_state,
    )
    bad_result = PhotonKickResult(
        E=perturbed_E,
        snapshot_times=result.snapshot_times,
        f_snapshots=result.f_snapshots,
        x_qp_snapshots=bad_x_qp,
        f_steady_state=result.f_steady_state,
        x_qp_steady_state=bad_x_qp_ss,
    )
    with pytest.raises(PhotonKickArtifactError, match="thermal Fermi-Dirac"):
        target.publish_baseline_pair(
            bad_result,
            fingerprint=target.solver_fingerprint(),
            producer_runtime=target.producer_runtime_provenance(),
        )
    np.testing.assert_array_equal(target.read_baseline().E, E)


def test_reader_requires_honest_snapshot_only_trajectory_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stored curve is not an independent proof of its ETD2 path."""
    result = _valid_result()
    pdf_path = tmp_path / "scope.pdf"
    _write_real_pdf(result, pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    path = write_baseline(
        result,
        tmp_path / "scope.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )
    restored = read_baseline(
        path,
        companion_pdf_path=pdf_path,
        require_current=False,
    )
    assert restored.trajectory_evidence_scope == target.TRAJECTORY_EVIDENCE_SCOPE
    assert "slow-live-recomputation" in restored.trajectory_evidence_scope

    with path.open(encoding="utf-8", newline="") as fp:
        rows = list(csv.reader(fp))
    marker = "# qpsim_metadata="
    metadata = json.loads(rows[1][0][len(marker) :])
    metadata["trajectory_evidence_scope"] = "full-dynamics-reader-certified"
    rows[1] = [f"{marker}{target._canonical_json(metadata)}"]
    with path.open("w", encoding="utf-8", newline="") as fp:
        csv.writer(fp, lineterminator="\n").writerows(rows)
    with pytest.raises(PhotonKickArtifactError, match="evidence scope"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_companion_pdf_requires_a_parsed_single_nonempty_page(
    tmp_path: Path,
) -> None:
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"%PDF-1.4\nunit-test\n%%EOF\n")
    with pytest.raises(PhotonKickArtifactError, match=r"xref|PDF"):
        companion_artifact_record(fake)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    two_pages = tmp_path / "two-pages.pdf"
    with PdfPages(two_pages) as pdf:
        for value in (1.0, 2.0):
            fig, ax = plt.subplots()
            ax.plot([0.0, 1.0], [0.0, value])
            pdf.savefig(fig)
            plt.close(fig)
    with pytest.raises(PhotonKickArtifactError, match="exactly one"):
        companion_artifact_record(two_pages)

    blank = tmp_path / "blank.pdf"
    fig = plt.figure(figsize=(2.0, 1.0))
    fig.savefig(blank)
    plt.close(fig)
    with pytest.raises(PhotonKickArtifactError, match=r"plotted|background"):
        companion_artifact_record(blank)

    one_page = tmp_path / "one-page.pdf"
    _write_real_pdf(_valid_result(), one_page)
    assert companion_artifact_record(one_page).size_bytes == one_page.stat().st_size


def test_reader_accepts_last_bit_thermal_seed_drift_but_rejects_material_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    perturbed = result.f_snapshots.copy()
    positive = np.flatnonzero(perturbed[0] > 0.0)
    assert positive.size
    index = int(positive[0])
    perturbed[0, index] = np.nextafter(perturbed[0, index], np.inf)
    x_qp, x_qp_ss = target._x_qp_from_occupations(
        result.E,
        perturbed,
        result.f_steady_state,
    )
    rounded = PhotonKickResult(
        E=result.E,
        snapshot_times=result.snapshot_times,
        f_snapshots=perturbed,
        x_qp_snapshots=x_qp,
        f_steady_state=result.f_steady_state,
        x_qp_steady_state=x_qp_ss,
    )
    pdf_path = tmp_path / "rounded.pdf"
    _write_real_pdf(rounded, pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    path = write_baseline(
        rounded,
        tmp_path / "rounded.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )
    read_baseline(path, companion_pdf_path=pdf_path, require_current=False)

    def materially_perturb_seed(data: np.ndarray, _times: np.ndarray) -> None:
        data[index, 2] *= 1.0001

    _rewrite_numeric_payload(path, materially_perturb_seed)
    with pytest.raises(PhotonKickArtifactError, match="thermal Fermi-Dirac"):
        read_baseline(path, companion_pdf_path=pdf_path, require_current=False)


def test_reader_rejects_self_hashed_decreasing_transient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    pdf_path = tmp_path / "decreasing.pdf"
    _write_real_pdf(result, pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    path = write_baseline(
        result,
        tmp_path / "decreasing.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )

    def make_decreasing(data: np.ndarray, _times: np.ndarray) -> None:
        data[:, 2 + 10] = data[:, 2 + 9] * 0.5

    _rewrite_numeric_payload(path, make_decreasing)
    with pytest.raises(PhotonKickArtifactError, match="decreasing x_qp"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_reader_rejects_self_hashed_nonthermal_initial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    pdf_path = tmp_path / "nonthermal.pdf"
    _write_real_pdf(result, pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    path = write_baseline(
        result,
        tmp_path / "nonthermal.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )

    def perturb_initial_state(data: np.ndarray, _times: np.ndarray) -> None:
        data[:, 2] *= 1.0001

    _rewrite_numeric_payload(path, perturb_initial_state)
    with pytest.raises(PhotonKickArtifactError, match="thermal Fermi-Dirac"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_reader_rejects_self_hashed_trajectory_far_from_steady_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    pdf_path = tmp_path / "far.pdf"
    _write_real_pdf(result, pdf_path)
    monkeypatch.setattr(
        target,
        "_steady_state_certificate",
        lambda *_args: dict(_TEST_CERTIFICATE),
    )
    path = write_baseline(
        result,
        tmp_path / "far.csv",
        companion_pdf=companion_artifact_record(pdf_path),
    )

    def stop_halfway(data: np.ndarray, _times: np.ndarray) -> None:
        thermal = data[:, 2].copy()
        steady = data[:, 1]
        progress = np.linspace(0.0, 0.5, data.shape[1] - 2)
        data[:, 2:] = thermal[:, None] + progress[None, :] * (steady - thermal)[:, None]

    _rewrite_numeric_payload(path, stop_halfway)
    with pytest.raises(PhotonKickArtifactError, match="final x_qp"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_reader_reassembles_and_rejects_forged_steady_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _valid_result()
    pdf_path = tmp_path / "certificate.pdf"
    _write_real_pdf(result, pdf_path)
    with monkeypatch.context() as context:
        context.setattr(
            target,
            "_steady_state_certificate",
            lambda *_args: dict(_TEST_CERTIFICATE),
        )
        path = write_baseline(
            result,
            tmp_path / "certificate.csv",
            companion_pdf=companion_artifact_record(pdf_path),
        )
    with pytest.raises(PhotonKickArtifactError, match=r"certificate|reassembled"):
        read_baseline(
            path,
            companion_pdf_path=pdf_path,
            require_current=False,
        )


def test_certificate_binding_allows_blas_roundoff_but_rejects_material_forgery() -> None:
    stamped: dict[str, float | str] = {
        "metric": target.STEADY_STATE_CERTIFICATE_METRIC,
        "qp_backward_error": 1.719302487851625e-13,
        "qp_residual_inf": 9.367506770274758e-16,
        "qp_number_backward_error": 2.7293117085953792e-14,
    }
    # Measured by reassembling the same persisted 810-bin state with the
    # default 32-thread OpenBLAS runtime instead of its one-thread producer.
    cross_thread: dict[str, float | str] = {
        "metric": target.STEADY_STATE_CERTIFICATE_METRIC,
        "qp_backward_error": 1.7193796059297093e-13,
        "qp_residual_inf": 9.367506770274758e-16,
        "qp_number_backward_error": 2.728740767557039e-14,
    }
    bound = target._bind_steady_state_certificate(
        stamped,
        cross_thread,
        context="cross-thread regression",
    )
    assert bound == cross_thread

    # Hosted Linux NumPy 2.5.1 (on both Python 3.13 and 3.14) changes the
    # cancellation-limited raw residual by 1.28e-17 while leaving it eight
    # million times below its independent physics gate.
    hosted_linux: dict[str, float | str] = {
        "metric": target.STEADY_STATE_CERTIFICATE_METRIC,
        "qp_backward_error": 1.7194037161209342e-13,
        "qp_residual_inf": 9.495442626628048e-16,
        "qp_number_backward_error": 2.7314170710979955e-14,
    }
    bound = target._bind_steady_state_certificate(
        stamped,
        hosted_linux,
        context="hosted-Linux regression",
    )
    assert bound == hosted_linux

    materially_forged: dict[str, float | str] = dict(stamped)
    materially_forged["qp_backward_error"] = float(materially_forged["qp_backward_error"]) * 1.01
    with pytest.raises(PhotonKickArtifactError, match="does not match fresh"):
        target._bind_steady_state_certificate(
            materially_forged,
            stamped,
            context="material-forgery regression",
        )

    forged_residual: dict[str, float | str] = dict(stamped)
    forged_residual["qp_residual_inf"] = (
        float(forged_residual["qp_residual_inf"])
        + 2.0 * target.STEADY_STATE_RESIDUAL_BINDING_ATOL
    )
    with pytest.raises(PhotonKickArtifactError, match="does not match fresh"):
        target._bind_steady_state_certificate(
            forged_residual,
            stamped,
            context="raw-residual-forgery regression",
        )


def test_os_lock_excludes_concurrent_reader_and_publisher(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "locked.csv"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = """
import sys
import time
from pathlib import Path
from validation.transient.photon_kick_response import _artifact_lock

artifact, ready, release = map(Path, sys.argv[1:4])
with _artifact_lock(artifact, exclusive=True):
    ready.write_text("ready", encoding="utf-8")
    while not release.exists():
        time.sleep(0.01)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(artifact), str(ready), str(release)],
        cwd=Path(__file__).resolve().parents[2],
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), "child failed to acquire the transient artifact lock"
        with pytest.raises(PhotonKickArtifactError, match="concurrent reader"):
            read_baseline(artifact, require_current=False)
        with (
            pytest.raises(PhotonKickArtifactError, match="concurrent publisher"),
            target._artifact_lock(artifact, exclusive=True),
        ):
            pass
    finally:
        release.write_text("release", encoding="utf-8")
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10.0)
    assert process.returncode == 0
