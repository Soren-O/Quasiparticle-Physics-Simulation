"""Regression test: Fischer 2023 Fig. 6 qpsim run matches its pinned qpsim CSV.

Iterative-mode tolerance: 1e-6. The exact regeneration is
both ``slow`` and ``manual_slow``: its sweep performs
$|T_B|\\times|\\bar n|$ joint Picard + self-consistent BCS gap solves on the
1640-bin paper-parameter grid (including sub-gap guard cells). This is not a
numerical comparison against digitized paper-curve data. The promoted
campaign measured 12.075 hours of aggregate worker time and 4.229 hours wall
time with three concurrent single-thread rows.
Opt in explicitly with ``pytest -m "slow and manual_slow"``.

The expensive sweep range (``N_BAR_VALUES``) is tunable in
:mod:`fig6_paper`; tighten it if this test starts to dominate the slow
suite. The pinned baseline is self-consistent against whichever range is
configured at generation time.

First-time generation::

    python -m validation.fischer_2023.fig6_paper
"""

from __future__ import annotations

import ast
import csv
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qpsim.backends.diffusion import DiffusionBackend
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics import calibrate_gap
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import (
    _weighted_number_backward_error,
    number_changing_gain_loss,
)

import validation.fischer_2023.fig5_paper as fig5_paper
import validation.fischer_2023.fig5_solve as fig5_solve
import validation.fischer_2023.fig6_paper as fig6_paper
import validation.fischer_2023.fig6_solve as fig6_solve
from validation.fischer_2023.fig6_paper import (
    FIG6_BASELINE_COLUMNS,
    T_BATH_VALUES,
    LegacyArtifactError,
    baseline_path,
    config_metadata,
    read_baseline,
    run,
)
from validation.fischer_2023.fig6_solve import (
    DELTA_0,
    DIRECT_GAP_BACKWARD_ERROR_LIMIT,
    FIG6_CERTIFICATE_FIELDS,
    FINITE_CUTOFF_DELTA0_OVER_KBTC,
    GAP_FIXED_POINT_ABS_TOL_UEV,
    N_BAR_VALUES,
    OMEGA_0,
    T_C,
    TARGET_BACKWARD_ERROR_LIMIT,
    _build_grid_and_spectral,
    _require_target_certificate,
)
from validation.fischer_2023.steady_state_certificate import (
    QP_NUMBER_CERTIFICATE_FIELD,
)


@pytest.fixture
def schema_result(monkeypatch) -> fig6_paper.Fig6PaperResult:
    """Cheap current-schema fixture whose claims are state-derived."""
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_paper, "T_BATH_VALUES", (0.20,))

    def fake_certificate(state, *, photon_params, tau_l):
        del photon_params, tau_l
        seed = (
            1.0e-12
            + 1.0e-4 * float(state.f[0])
            + 1.0e-6 * float(state.phonon.n_ph[0, 0, 0])
        )
        return {
            "qp_residual_inf": seed,
            "qp_backward_error": 0.5 * seed,
            "qp_number_backward_error": 0.6 * seed,
            "phonon_residual_inf": 0.7 * seed,
            "phonon_raw_backward_error": 0.8 * seed,
            "phonon_backward_error": 0.9 * seed,
        }

    monkeypatch.setattr(
        fig6_paper.certificate_module,
        "steady_state_certificate",
        fake_certificate,
    )
    delta_eq = 179.9
    monkeypatch.setattr(
        fig6_paper,
        "calibrate_gap",
        lambda **_kwargs: SimpleNamespace(delta_eq=delta_eq),
    )
    monkeypatch.setattr(
        fig6_paper,
        "solve_gap",
        lambda *_args, reference_gap, **_kwargs: reference_gap,
    )

    _, _, spectral = fig6_solve._build_grid_and_spectral()
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    f = np.full(spectral.E.size, 1.0e-8)
    n_ph = np.full(omega.size, 1.0e-10)
    gap = 179.8
    state = fig6_paper._rebuild_state(
        f=f,
        n_ph=n_ph,
        T_bath=0.20,
        gap=gap,
        base_spectral=spectral,
    )
    tau_l = float(state.phonon.tau_l[0, 0])
    tau_0_pb = fig6_paper.config_metadata().tau_0_pb_ns
    certificate = fake_certificate(state, photon_params={}, tau_l=tau_l)
    T_star = fig6_solve._kBTstar_eq35(1.0e4) / DELTA_0
    x_num = fig6_paper.qp_fraction(f, state.spectral, delta_0=DELTA_0)
    x_eq47 = fig5_paper._xqp_analytic_eq47(
        0.20,
        1.0e4,
        tau_l=tau_l,
        tau_0_pb=tau_0_pb,
    )
    delta_T = DELTA_0 - delta_eq
    obs_eq53 = (
        delta_T
        - DELTA_0 * fig6_solve._paper_eq53_analytic_drive(x_eq47, T_star)
    ) / delta_T
    return fig6_paper.Fig6PaperResult(
        tau_0_pb_ns=tau_0_pb,
        tau_l_ns=tau_l,
        T_bath=np.array([0.20]),
        n_bar=np.array([1.0e4]),
        T_star_over_delta=np.array([[T_star]]),
        delta_eq=np.array([delta_eq]),
        delta_driven=np.array([[gap]]),
        delta_thermal_T_bath=np.array([delta_eq]),
        paper_observable_num=np.array([[(gap - delta_eq) / delta_T]]),
        paper_observable_eq53=np.array([[obs_eq53]]),
        x_qp_num=np.array([[x_num]]),
        x_qp_eq47=np.array([[x_eq47]]),
        qp_residual_inf=np.array([[certificate["qp_residual_inf"]]]),
        qp_backward_error=np.array([[certificate["qp_backward_error"]]]),
        qp_number_backward_error=np.array(
            [[certificate["qp_number_backward_error"]]]
        ),
        phonon_residual_inf=np.array([[certificate["phonon_residual_inf"]]]),
        phonon_raw_backward_error=np.array(
            [[certificate["phonon_raw_backward_error"]]]
        ),
        phonon_backward_error=np.array(
            [[certificate["phonon_backward_error"]]]
        ),
        gap_fixed_point_abs_error_uev=np.array([[0.0]]),
        state_f=f.reshape(1, 1, -1),
        state_n_ph=n_ph.reshape(1, 1, -1),
    )


def test_recertification_uses_authenticated_producer_gap_for_derived_ratios(
    monkeypatch: pytest.MonkeyPatch,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    """A reader-host ULP in Delta_eq must not be amplified into a false drift."""
    producer_delta_eq = float(schema_result.delta_eq[0])
    reader_delta_eq = math.nextafter(producer_delta_eq, math.inf)
    monkeypatch.setattr(
        fig6_paper,
        "calibrate_gap",
        lambda **_kwargs: SimpleNamespace(delta_eq=reader_delta_eq),
    )

    # The current calibration still authenticates near-bitwise, while
    # observables are rebound to the exact producer anchor persisted beside
    # the returned state.  Before the repair, this one-ULP reader drift was
    # amplified beyond the 256-epsilon observable identity gate.
    fig6_paper.validate_artifact_result(schema_result)


def test_recertification_accepts_runtime_drift_and_returns_fresh_certificates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    """Reader certificates are fresh scientific checks, not producer bit pins."""
    fresh = {
        # Raw diagnostics deliberately drift by many orders while remaining
        # finite/non-negative; they are not scientific acceptance gates.
        "qp_residual_inf": 0.25,
        "qp_backward_error": 0.25 * TARGET_BACKWARD_ERROR_LIMIT,
        QP_NUMBER_CERTIFICATE_FIELD: 0.50 * TARGET_BACKWARD_ERROR_LIMIT,
        "phonon_residual_inf": 0.50,
        "phonon_raw_backward_error": 0.75,
        "phonon_backward_error": 0.75 * TARGET_BACKWARD_ERROR_LIMIT,
    }
    monkeypatch.setattr(
        fig6_paper.certificate_module,
        "steady_state_certificate",
        lambda *_args, **_kwargs: fresh.copy(),
    )
    path = fig6_paper._write_rebound_baseline(
        schema_result,
        tmp_path / "portable-certificates.csv",
    )
    stamped, _ = fig6_paper._read_artifact(
        path,
        return_stamped_certificates=True,
    )
    np.testing.assert_array_equal(
        stamped.qp_backward_error,
        schema_result.qp_backward_error,
    )
    restored = fig6_paper.read_baseline(path)
    for field, value in fresh.items():
        np.testing.assert_array_equal(
            getattr(restored, field),
            np.array([[value]]),
        )
    np.testing.assert_array_equal(
        restored.gap_fixed_point_abs_error_uev,
        np.array([[0.0]]),
    )


def test_current_writer_rejects_under_limit_forged_producer_stamp(
    tmp_path: Path,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    """Current production must bind its stamps to its same-runtime state."""
    forged = replace(
        schema_result,
        qp_backward_error=np.full_like(
            schema_result.qp_backward_error,
            0.25 * TARGET_BACKWARD_ERROR_LIMIT,
        ),
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match=r"producer certificate qp_backward_error",
    ):
        fig6_paper.write_baseline(
            forged,
            tmp_path / "forged-under-limit.csv",
        )


def test_recertification_rejects_bad_stamped_scientific_certificate(
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    bad = replace(
        schema_result,
        qp_backward_error=np.full_like(
            schema_result.qp_backward_error,
            1.01 * TARGET_BACKWARD_ERROR_LIMIT,
        ),
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match=r"stamped certificate field 'qp_backward_error'.*scientific gate",
    ):
        fig6_paper.validate_artifact_result(bad)


def test_recertification_rejects_bad_fresh_scientific_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    path = fig6_paper.write_baseline(
        schema_result,
        tmp_path / "bad-fresh-scientific.csv",
    )
    fresh = {
        "qp_residual_inf": 0.0,
        "qp_backward_error": 0.0,
        QP_NUMBER_CERTIFICATE_FIELD: 0.0,
        "phonon_residual_inf": 0.0,
        "phonon_raw_backward_error": 0.0,
        "phonon_backward_error": 1.01 * TARGET_BACKWARD_ERROR_LIMIT,
    }
    monkeypatch.setattr(
        fig6_paper.certificate_module,
        "steady_state_certificate",
        lambda *_args, **_kwargs: fresh.copy(),
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match=r"reassembled certificate field 'phonon_backward_error'.*scientific gate",
    ):
        fig6_paper.read_baseline(path)


def test_recertification_rejects_bad_fresh_gap_certificate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    path = fig6_paper.write_baseline(
        schema_result,
        tmp_path / "bad-fresh-gap.csv",
    )
    monkeypatch.setattr(
        fig6_paper,
        "solve_gap",
        lambda *_args, reference_gap, **_kwargs: (
            reference_gap + 1.01 * GAP_FIXED_POINT_ABS_TOL_UEV
        ),
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match=r"reassembled gap fixed-point certificate.*scientific gate",
    ):
        fig6_paper.read_baseline(path)


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ("stamped", r"stamped certificate field 'qp_residual_inf'.*non-negative"),
        ("reassembled", r"reassembled certificate field 'qp_residual_inf'.*non-negative"),
    ),
)
def test_recertification_rejects_negative_raw_diagnostic(
    source: str,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    if source == "stamped":
        candidate = replace(
            schema_result,
            qp_residual_inf=np.full_like(schema_result.qp_residual_inf, -1.0),
        )
        with pytest.raises(fig6_paper.ArtifactValidationError, match=expected):
            fig6_paper.validate_artifact_result(candidate)
        return

    path = fig6_paper.write_baseline(
        schema_result,
        tmp_path / "negative-fresh-diagnostic.csv",
    )
    monkeypatch.setattr(
        fig6_paper.certificate_module,
        "steady_state_certificate",
        lambda *_args, **_kwargs: {
            "qp_residual_inf": -1.0,
            "qp_backward_error": 0.0,
            QP_NUMBER_CERTIFICATE_FIELD: 0.0,
            "phonon_residual_inf": 0.0,
            "phonon_raw_backward_error": 0.0,
            "phonon_backward_error": 0.0,
        },
    )
    with pytest.raises(fig6_paper.ArtifactValidationError, match=expected):
        fig6_paper.read_baseline(path)


def test_canonical_publication_record_rejects_mixed_pdf(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    csv_path = tmp_path / "fig6.csv"
    pdf_path = tmp_path / "fig6.pdf"
    record_path = tmp_path / "fig6.promotion.json"
    monkeypatch.setattr(fig6_paper, "baseline_path", lambda **_kwargs: csv_path)
    monkeypatch.setattr(fig6_paper, "plot_path", lambda **_kwargs: pdf_path)
    monkeypatch.setattr(fig6_paper, "promotion_record_path", lambda: record_path)

    fig6_paper.publish_artifacts(
        schema_result,
        generation_evidence=fig6_paper.serial_generation_evidence(),
    )
    fig6_paper.read_promotion_record()
    pdf_path.write_bytes(pdf_path.read_bytes() + b"\n% mixed producer\n")
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="does not match",
    ):
        fig6_paper.read_promotion_record()


def test_publication_record_binds_parallel_campaign_evidence(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    from scripts import regenerate_fischer_fig6_parallel as parallel

    csv_path = tmp_path / "fig6.csv"
    pdf_path = tmp_path / "fig6.pdf"
    record_path = tmp_path / "fig6.promotion.json"
    monkeypatch.setattr(fig6_paper, "baseline_path", lambda **_kwargs: csv_path)
    monkeypatch.setattr(fig6_paper, "plot_path", lambda **_kwargs: pdf_path)
    monkeypatch.setattr(fig6_paper, "promotion_record_path", lambda: record_path)
    producer = parallel._producer_base(Path(parallel.__file__).resolve())
    worker_payloads = {
        "t00": {
            "semantic_sha256": fig6_paper.result_row_sha256(schema_result, 0),
            "temperature_K": 0.20,
        }
    }
    evidence = {
        "artifact_fingerprint": producer["artifact_fingerprint"],
        "campaign": {
            "aggregate_worker_s": 10.0,
            "new_rows": 1,
            "resumed_rows": 0,
            "wall_s": 10.0,
        },
        "mode": "parallel-temperature-rows",
        "run_identity": producer["run_identity"],
        "run_identity_schema": producer["run_identity_schema"],
        "runner": producer["runner"],
        "runtime": producer["runtime"],
        "schema": fig6_paper.GENERATION_EVIDENCE_SCHEMA,
        "single_thread_environment": producer["single_thread_environment"],
        "worker_payloads": worker_payloads,
    }
    fig6_paper.publish_artifacts(
        schema_result,
        generation_evidence=evidence,
    )
    assert fig6_paper.read_promotion_record()["generation"] == evidence
    forged = json.loads(json.dumps(evidence))
    forged["worker_payloads"]["t00"]["semantic_sha256"] = "0" * 64
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="semantic row digest",
    ):
        fig6_paper.validate_generation_evidence(
            forged,
            result=schema_result,
        )


def test_publication_lock_refuses_concurrent_writer(
    tmp_path,
    monkeypatch,
) -> None:
    record_path = tmp_path / "fig6.promotion.json"
    monkeypatch.setattr(fig6_paper, "promotion_record_path", lambda: record_path)
    with (
        fig6_paper._publication_lock(),
        pytest.raises(RuntimeError, match=r"Another Fig\. 6 publisher"),
        fig6_paper._publication_lock(),
    ):
        pytest.fail("nested publisher unexpectedly acquired the lock")


def test_canonical_readers_hold_lock_across_artifact_and_record(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    csv_path = tmp_path / "fig6.csv"
    pdf_path = tmp_path / "fig6.pdf"
    record_path = tmp_path / "fig6.promotion.json"
    monkeypatch.setattr(fig6_paper, "baseline_path", lambda **_kwargs: csv_path)
    monkeypatch.setattr(fig6_paper, "plot_path", lambda **_kwargs: pdf_path)
    monkeypatch.setattr(fig6_paper, "promotion_record_path", lambda: record_path)
    fig6_paper.publish_artifacts(
        schema_result,
        generation_evidence=fig6_paper.serial_generation_evidence(),
    )

    # A publisher replaces PDF -> CSV -> record while holding this same lock.
    # Both readers must refuse instead of observing an old/new mixed tuple.
    with fig6_paper._publication_lock():
        with pytest.raises(RuntimeError, match=r"Another Fig\. 6 publisher"):
            fig6_paper.read_baseline()
        with pytest.raises(RuntimeError, match=r"Another Fig\. 6 publisher"):
            fig6_paper.read_baseline_metadata()
        with pytest.raises(RuntimeError, match=r"Another Fig\. 6 publisher"):
            fig6_paper.read_promotion_record()


def test_serial_generation_rejects_source_change_during_solve(
    monkeypatch,
    schema_result,
) -> None:
    revision = {"value": "before"}
    published: list[bool] = []
    before = fig6_paper.artifact_fingerprint()
    after = json.loads(json.dumps(before))
    after["config"]["delta_0"] = float(after["config"]["delta_0"]) + 1.0
    monkeypatch.setattr(
        fig6_paper,
        "artifact_fingerprint",
        lambda: before if revision["value"] == "before" else after,
    )

    def fake_run_cached(**_kwargs):
        revision["value"] = "after"
        return schema_result

    monkeypatch.setattr(fig6_paper, "run_cached", fake_run_cached)
    monkeypatch.setattr(
        fig6_paper,
        "publish_artifacts",
        lambda *_args, **_kwargs: published.append(True),
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="source/configuration changed",
    ):
        fig6_paper.generate_baseline()
    assert not published


def test_plot_dashed_curve_is_the_stored_certified_eq53_array(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import matplotlib.axes

    dashed_y: list[np.ndarray] = []
    real_plot = matplotlib.axes.Axes.plot

    def capture_plot(self, *args, **kwargs):
        if kwargs.get("ls") == (0, (5, 2)):
            dashed_y.append(np.asarray(args[1], dtype=float).copy())
        return real_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", capture_plot)
    fig6_paper.write_plot(schema_result, tmp_path / "fig6.pdf")
    assert len(dashed_y) == 1
    np.testing.assert_array_equal(
        dashed_y[0],
        schema_result.paper_observable_eq53[0],
    )


def test_direct_writers_cannot_clobber_canonical_bundle(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    csv_path = tmp_path / "canonical.csv"
    pdf_path = tmp_path / "canonical.pdf"
    record_path = csv_path.with_suffix(".promotion.json")
    csv_path.write_bytes(b"csv-sentinel")
    pdf_path.write_bytes(b"pdf-sentinel")
    record_path.write_bytes(b"record-sentinel")

    def fake_baseline_path(*, direct_gap_observable: bool = False) -> Path:
        if direct_gap_observable:
            return tmp_path / "direct.csv"
        return csv_path

    def fake_plot_path(*, direct_gap_observable: bool = False) -> Path:
        if direct_gap_observable:
            return tmp_path / "direct.pdf"
        return pdf_path

    monkeypatch.setattr(fig6_paper, "baseline_path", fake_baseline_path)
    monkeypatch.setattr(fig6_paper, "plot_path", fake_plot_path)
    for writer in (
        lambda: fig6_paper.write_baseline(schema_result),
        lambda: fig6_paper.write_baseline(schema_result, csv_path),
        lambda: fig6_paper.write_baseline(schema_result, pdf_path),
        lambda: fig6_paper.write_plot(schema_result),
        lambda: fig6_paper.write_plot(schema_result, pdf_path),
        lambda: fig6_paper.write_plot(schema_result, record_path),
    ):
        with pytest.raises(
            fig6_paper.ArtifactValidationError,
            match="canonical",
        ):
            writer()
    assert csv_path.read_bytes() == b"csv-sentinel"
    assert pdf_path.read_bytes() == b"pdf-sentinel"
    assert record_path.read_bytes() == b"record-sentinel"


def test_artifact_fingerprint_includes_sweep_cache_source(
    monkeypatch,
) -> None:
    captured: dict[str, tuple[Path, ...]] = {}

    def fake_source_manifest(
        _primary: Path,
        *,
        extra_validation_modules: tuple[Path, ...],
    ) -> dict[str, str]:
        captured["extra"] = extra_validation_modules
        return {"unit-test": "0" * 64}

    monkeypatch.setattr(fig6_paper, "source_manifest", fake_source_manifest)
    fig6_paper.artifact_fingerprint()
    assert Path(fig6_paper.sweep_cache.__file__) in captured["extra"]


def test_artifact_fingerprint_allows_only_bounded_float_ulp_drift() -> None:
    current = fig6_paper.artifact_fingerprint()
    claimed = json.loads(json.dumps(current))
    key = "finite_cutoff_delta0_over_kbtc"
    for _ in range(fig6_paper._ARTIFACT_FINGERPRINT_MAX_ULPS):
        claimed["config"][key] = math.nextafter(claimed["config"][key], math.inf)
    assert fig6_paper._fingerprint_mismatch(claimed, current) is None

    claimed["config"][key] = math.nextafter(claimed["config"][key], math.inf)
    assert "ULP" in str(fig6_paper._fingerprint_mismatch(claimed, current))


def test_artifact_fingerprint_keeps_shape_types_and_sources_exact() -> None:
    current = fig6_paper.artifact_fingerprint()
    forged = json.loads(json.dumps(current))
    forged["config"]["num_bins"] = float(forged["config"]["num_bins"])
    assert "type" in str(fig6_paper._fingerprint_mismatch(forged, current))

    forged = json.loads(json.dumps(current))
    source_path = next(iter(forged["source_sha256"]))
    forged["source_sha256"][source_path] = "0" * 64
    assert source_path in str(fig6_paper._fingerprint_mismatch(forged, current))

    forged = json.loads(json.dumps(current))
    forged["axes"]["T_bath_K"] = list(reversed(forged["axes"]["T_bath_K"]))
    assert "T_bath_K" in str(fig6_paper._fingerprint_mismatch(forged, current))


def test_generation_identity_uses_captured_not_live_fingerprint(monkeypatch) -> None:
    evidence = fig6_paper.serial_generation_evidence()
    captured = evidence["artifact_fingerprint"]
    identity = evidence["run_identity"]
    monkeypatch.setattr(
        fig6_paper,
        "artifact_fingerprint",
        lambda: {"different": "host"},
    )
    assert (
        fig6_paper.generation_run_identity(
            artifact_fingerprint=captured,
            mode=str(evidence["mode"]),
            runner=evidence["runner"],
            runtime=evidence["runtime"],
            single_thread_environment=evidence["single_thread_environment"],
            generation_schema=str(evidence["run_identity_schema"]),
        )
        == identity
    )


def test_migrated_v1_generation_identity_remains_historical() -> None:
    evidence = fig6_paper.serial_generation_evidence()
    evidence["run_identity_schema"] = (
        fig6_paper._LEGACY_GENERATION_EVIDENCE_SCHEMA
    )
    evidence["run_identity"] = fig6_paper.generation_run_identity(
        artifact_fingerprint=evidence["artifact_fingerprint"],
        mode=str(evidence["mode"]),
        runner=evidence["runner"],
        runtime=evidence["runtime"],
        single_thread_environment=evidence["single_thread_environment"],
        generation_schema=fig6_paper._LEGACY_GENERATION_EVIDENCE_SCHEMA,
    )
    assert fig6_paper.validate_generation_evidence(evidence) == evidence
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="current generation identity schema",
    ):
        fig6_paper.validate_generation_evidence(
            evidence,
            require_current_runtime=True,
        )


def test_pdf_validator_rejects_token_shaped_non_pdf(tmp_path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_bytes(
        b"%PDF-1.4\n"
        + b"/Type /Catalog /Type /Pages /Type /Page\n"
        + b"xref\n0 1\n0000000000 65535 f \n"
        + b"trailer << /Root 1 0 R >>\nstartxref\n0\n%%EOF\n"
        + b" " * 2048
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="complete nonempty Matplotlib PDF",
    ):
        fig6_paper._require_valid_pdf(path)


def test_console_diagnostic_literals_are_ascii() -> None:
    """An expensive Windows CLI run must not fail while printing results."""
    offenders: list[tuple[str, int, str]] = []
    for module in (fig5_solve, fig5_paper, fig6_solve, fig6_paper):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            console_roots: list[ast.AST] = []
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                console_roots.append(node)
            console_roots.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"description", "help"}
            )
            for root in console_roots:
                for child in ast.walk(root):
                    if not (
                        isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                    ):
                        continue
                    try:
                        child.value.encode("ascii")
                    except UnicodeEncodeError:
                        offenders.append(
                            (module.__name__, node.lineno, child.value)
                        )

    assert offenders == []


def test_direct_plot_limits_expose_signed_low_drive_point() -> None:
    x_limits, y_limits = fig6_paper._direct_plot_limits(
        np.array([0.159, 0.30, 0.70]),
        np.array([-0.0168, 0.10, 0.20]),
    )

    assert x_limits[0] < 0.159
    assert x_limits[1] > 0.70
    assert y_limits[0] < -0.0168
    assert y_limits[1] == 0.25


def test_programmatic_direct_generation_cannot_clobber_canonical(
    monkeypatch,
    schema_result,
) -> None:
    reference = schema_result
    written: dict[str, object] = {}

    monkeypatch.setattr(fig6_paper, "run_cached", lambda **_kwargs: reference)

    def record_csv(_result, path=None):
        written["csv"] = path
        return path

    def record_pdf(_result, path=None, **kwargs):
        written["pdf"] = path
        written["direct"] = kwargs["direct_gap_observable"]
        return path

    monkeypatch.setattr(fig6_paper, "_legacy_write_baseline", record_csv)
    monkeypatch.setattr(fig6_paper, "write_plot", record_pdf)

    csv_path, pdf_path = fig6_paper.generate_baseline(
        direct_gap_observable=True,
        fixed_gap_kinetics=True,
    )

    canonical = baseline_path()
    assert csv_path == fig6_paper.baseline_path(direct_gap_observable=True)
    assert pdf_path == fig6_paper.plot_path(direct_gap_observable=True)
    assert csv_path != canonical
    assert pdf_path != canonical.with_suffix(".pdf")
    assert written == {"csv": csv_path, "pdf": pdf_path, "direct": True}


def test_grid_covers_self_consistent_gap_support() -> None:
    """The kinetic grid must not begin at the gap it is allowed to suppress."""
    E, dE, spectral = _build_grid_and_spectral()
    first_edge = float(E[0] - 0.5 * dE[0])
    spacing = float(dE[0])

    assert first_edge <= DELTA_0 - OMEGA_0
    assert OMEGA_0 / spacing == pytest.approx(round(OMEGA_0 / spacing))
    assert not np.any(spectral.active_mask[DELTA_0 > E])


def test_finite_cutoff_calibration_anchors_delta0_and_tc() -> None:
    assert pytest.approx(
        1.7637398024450115,
        rel=0.0,
        abs=1e-15,
    ) == FINITE_CUTOFF_DELTA0_OVER_KBTC
    assert pytest.approx(1.184309192877208, rel=0.0, abs=1e-15) == T_C

    calibration = calibrate_gap(T_c=T_C, T_bath=0.0, xtol=1e-12)
    assert calibration.delta_0_bcs == pytest.approx(
        DELTA_0,
        rel=0.0,
        abs=5e-14,
    )
    assert calibration.delta_0_bcs / (KB_UEV_PER_K * T_C) == pytest.approx(
        FINITE_CUTOFF_DELTA0_OVER_KBTC,
        rel=0.0,
        abs=1e-15,
    )

    fingerprint = fig6_solve.solver_fingerprint()
    assert fingerprint["finite_cutoff_delta0_over_kbtc"] == pytest.approx(
        FINITE_CUTOFF_DELTA0_OVER_KBTC,
        rel=0.0,
        abs=0.0,
    )
    assert fingerprint["t_c"] == pytest.approx(T_C, rel=0.0, abs=0.0)
    assert fingerprint["gap_fixed_point_abs_tol_uev"] == pytest.approx(
        GAP_FIXED_POINT_ABS_TOL_UEV,
        rel=0.0,
        abs=0.0,
    )
    assert fingerprint["certificate_fields"] == list(FIG6_CERTIFICATE_FIELDS)


@pytest.mark.parametrize(
    ("T_bath", "expected_suppression_uev"),
    [
        (0.10, 8.321535460709129e-8),
        (0.15, 1.0737232068436242e-4),
        (0.20, 4.019584551002708e-3),
    ],
)
def test_finite_cutoff_thermal_suppressions_are_resolved(
    T_bath: float,
    expected_suppression_uev: float,
) -> None:
    calibration = calibrate_gap(
        T_c=T_C,
        T_bath=T_bath,
        Delta_0=DELTA_0,
        xtol=1e-12,
    )
    assert DELTA_0 - calibration.delta_eq == pytest.approx(
        expected_suppression_uev,
        rel=1e-6,
        abs=2e-12,
    )


def _assert_certified_baseline_balances(path, axes) -> None:
    """Require the current schema and gate every accepted certificate field."""
    text = path.read_text(encoding="utf-8")
    header = next(
        (line for line in text.splitlines() if line.startswith("T_bath_K,")),
        "",
    )
    assert tuple(header.split(",")) == fig6_paper._ARTIFACT_COLUMNS, (
        "certified Fig. 6 preflight requires the current authenticated "
        "artifact schema"
    )
    accepted = np.isfinite(axes.x_qp_num)
    for field in FIG6_CERTIFICATE_FIELDS:
        values = getattr(axes, field)[accepted]
        assert np.all(np.isfinite(values)), (
            f"certified baseline has non-finite {field} at an accepted point"
        )
        assert np.all(values >= 0.0), (
            f"certified baseline has negative {field} at an accepted point"
        )
    for field in (
        "qp_backward_error",
        QP_NUMBER_CERTIFICATE_FIELD,
        "phonon_backward_error",
    ):
        values = getattr(axes, field)[accepted]
        assert np.all(values <= TARGET_BACKWARD_ERROR_LIMIT), (
            f"certified baseline {field} exceeds "
            f"{TARGET_BACKWARD_ERROR_LIMIT:g}: {values.tolist()}"
        )
    gap_errors = axes.gap_fixed_point_abs_error_uev[accepted]
    assert np.all(gap_errors <= GAP_FIXED_POINT_ABS_TOL_UEV), (
        "certified baseline gap-map error exceeds "
        f"{GAP_FIXED_POINT_ABS_TOL_UEV:g}: {gap_errors.tolist()}"
    )


def _lightweight_canonical_preflight(
    path: Path,
) -> tuple[fig6_paper.BaselineMetadata, SimpleNamespace]:
    """Authenticate bytes and parse scalar claims without decoding 66 states.

    The public canonical readers deliberately decode and independently
    re-certify every stored state.  That is the right publication/closeout
    contract, but it made this advertised one-second configuration preflight
    perform the full recertification four times (twice through each public
    reader).  Here the promotion lock protects one cheap snapshot while we:

    * bind the exact CSV/PDF bytes to the promotion record;
    * validate current generation/source evidence without recomputing row
      semantic digests; and
    * parse only the scalar header, axes, and stored certificate columns.

    Full state-derived recertification remains in ``read_baseline()`` and in
    the signed-diagnostic/closeout path.
    """

    record_path = fig6_paper.promotion_record_path()
    pdf_path = fig6_paper.plot_path()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert isinstance(record, dict)
    assert set(record) == {"artifact_schema", "artifacts", "generation", "schema"}
    assert record["schema"] == fig6_paper.PROMOTION_RECORD_SCHEMA
    assert record["artifact_schema"] == fig6_paper.ARTIFACT_SCHEMA
    assert record["artifacts"] == {
        "csv": fig6_paper._file_identity(path),
        "pdf": fig6_paper._file_identity(pdf_path),
    }
    fig6_paper._require_valid_pdf(pdf_path)
    fig6_paper.validate_generation_evidence(
        record["generation"],
        result=None,
    )

    rows = fig6_paper._read_csv_rows(path)
    artifact_metadata = fig6_paper._artifact_metadata(path, rows)
    config = artifact_metadata["fingerprint"]["config"]
    metadata = fig6_paper.BaselineMetadata(
        delta_0=float(config["delta_0"]),
        finite_cutoff_delta0_over_kbtc=float(
            config["finite_cutoff_delta0_over_kbtc"]
        ),
        tau_0=float(config["tau_0"]),
        t_c=float(config["t_c"]),
        omega_0=float(config["omega_0"]),
        c_phot=float(config["c_phot"]),
        film_thickness_nm=float(config["film_thickness_nm"]),
        eta=float(config["eta"]),
        num_bins=int(config["num_bins"]),
        e_min_factor=float(config["e_min_factor"]),
        e_max_factor=float(config["e_max_factor"]),
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
        tau_l_ns=float(config["tau_l_ns"]),
        tau_l_model=str(config["tau_l_model"]),
        gap_fixed_point_abs_tol_uev=float(
            config["gap_fixed_point_abs_tol_uev"]
        ),
        certificate_metric_version=str(config["certificate_metric_version"]),
    )
    data_rows: list[dict[str, float]] = []
    for row in rows[3:]:
        assert len(row) == len(fig6_paper._ARTIFACT_COLUMNS)
        values = {
            name: float(row[index])
            for index, name in enumerate(fig6_paper._ARTIFACT_COLUMNS[:16])
        }
        assert np.all(np.isfinite(tuple(values.values())))
        data_rows.append(values)

    assert data_rows
    T_values = np.asarray(
        list(dict.fromkeys(row["T_bath_K"] for row in data_rows)),
        dtype=float,
    )
    n_values = np.asarray(
        list(dict.fromkeys(row["n_bar"] for row in data_rows)),
        dtype=float,
    )
    expected_coordinates = [
        (float(T_bath), float(n_bar))
        for T_bath in T_values
        for n_bar in n_values
    ]
    actual_coordinates = [
        (row["T_bath_K"], row["n_bar"]) for row in data_rows
    ]
    assert actual_coordinates == expected_coordinates
    shape = (T_values.size, n_values.size)
    arrays = {
        name: np.asarray([row[name] for row in data_rows], dtype=float).reshape(shape)
        for name in fig6_paper._ARTIFACT_COLUMNS[2:16]
    }
    return metadata, SimpleNamespace(
        T_bath=T_values,
        n_bar=n_values,
        **arrays,
    )


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + sweep axes.

    Gating the 12.075 h aggregate-worker :func:`run` behind this turns a stale
    config/baseline pairing — a ``TAU_L_MODEL`` swap, a grid change, a
    sweep-range edit — into a seconds-long failure instead of one discovered
    only after the full sweep. Compares the config fingerprint against the
    baseline header, and the configured sweep axes against the baseline's
    data rows.
    """
    cfg = config_metadata()
    with fig6_paper._publication_lock():
        meta, axes = _lightweight_canonical_preflight(path)
        # Keep this human-readable header check in the same authenticated
        # snapshot as the promotion-record and row reads.
        _assert_certified_baseline_balances(path, axes)

    assert cfg.tau_l_model == meta.tau_l_model, (
        f"TAU_L_MODEL config={cfg.tau_l_model!r} != baseline {meta.tau_l_model!r}; "
        "regenerate the baseline or restore the model before the slow run."
    )
    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.finite_cutoff_delta0_over_kbtc == pytest.approx(
        meta.finite_cutoff_delta0_over_kbtc,
        rel=0.0,
        abs=1e-15,
    )
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_c == pytest.approx(meta.t_c, rel=0.0, abs=1e-15)
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.film_thickness_nm == pytest.approx(meta.film_thickness_nm)
    assert cfg.eta == pytest.approx(meta.eta)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)
    assert cfg.tau_l_ns == pytest.approx(meta.tau_l_ns, rel=1e-8)
    assert cfg.gap_fixed_point_abs_tol_uev == pytest.approx(
        meta.gap_fixed_point_abs_tol_uev,
        rel=0.0,
        abs=0.0,
    )
    assert cfg.certificate_metric_version == meta.certificate_metric_version
    np.testing.assert_allclose(
        np.asarray(T_BATH_VALUES, dtype=float), axes.T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="T_bath sweep axis differs from baseline",
    )
    np.testing.assert_allclose(
        N_BAR_VALUES, axes.n_bar, rtol=1e-12, atol=0.0,
        err_msg="n_bar sweep axis (range/count) differs from baseline",
    )


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header.

    This is the standing fast-suite guard that would have caught the τ_ℓ-model
    / baseline mismatch that once wasted 9.5 h. The ``manual_slow``
    ``test_matches_pinned_baseline`` re-runs the same check inline so the
    12.075 h aggregate-worker sweep is gated even when this fast test is not
    selected.
    """
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    try:
        _assert_config_matches_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(str(exc))


@pytest.mark.slow
def test_canonical_bundle_authenticates_and_recertifies() -> None:
    """One explicit current-artifact gate performs full state recertification."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    fig6_paper.read_promotion_record()


def test_lightweight_preflight_rejects_bad_number_certificate(
    tmp_path: Path,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    """The fast gate must retain the amplitude-sensitive QP-number check."""
    path = tmp_path / "fig6-current-header.csv"
    path.write_text(
        ",".join(fig6_paper._ARTIFACT_COLUMNS) + "\n",
        encoding="utf-8",
    )
    bad = replace(
        schema_result,
        qp_number_backward_error=np.full_like(
            schema_result.qp_number_backward_error,
            1.01 * TARGET_BACKWARD_ERROR_LIMIT,
        ),
    )

    with pytest.raises(AssertionError, match="qp_number_backward_error exceeds"):
        _assert_certified_baseline_balances(path, bad)


def test_lightweight_preflight_rejects_negative_certificate(
    tmp_path: Path,
    schema_result: fig6_paper.Fig6PaperResult,
) -> None:
    path = tmp_path / "fig6-current-header.csv"
    path.write_text(
        ",".join(fig6_paper._ARTIFACT_COLUMNS) + "\n",
        encoding="utf-8",
    )
    bad = replace(
        schema_result,
        qp_residual_inf=np.full_like(schema_result.qp_residual_inf, -1.0),
    )

    with pytest.raises(AssertionError, match="negative qp_residual_inf"):
        _assert_certified_baseline_balances(path, bad)


def test_converged_target_with_bad_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 2e-5,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": 0.0,
    }
    with pytest.raises(RuntimeError, match="converged target failed"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_converged_target_with_bad_gap_map_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 1e-8,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": 1.01 * GAP_FIXED_POINT_ABS_TOL_UEV,
    }
    with pytest.raises(RuntimeError, match="gap-map certificate"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_fixed_gap_certificate_allows_nan_gap_map_metric() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 1e-8,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": float("nan"),
    }
    _require_target_certificate(
        certificate,
        T_bath=0.1,
        n_bar=1e6,
        require_gap_fixed_point=False,
    )


def test_converged_target_with_bad_number_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 0.0,
        "qp_number_backward_error": 0.6,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 0.0,
        "phonon_backward_error": 0.0,
        "gap_fixed_point_abs_error_uev": 0.0,
    }
    with pytest.raises(RuntimeError, match="qp_number"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_direct_gap_certificate_uses_strict_certified_metrics() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 0.5 * DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        # Raw phonon balance includes irreducible affine-root rounding; direct
        # mode gates the representability-aware certified excess below.
        "phonon_raw_backward_error": 1e-6,
        "phonon_backward_error": 0.0,
        "gap_fixed_point_abs_error_uev": float("nan"),
    }
    _require_target_certificate(
        certificate,
        T_bath=0.1,
        n_bar=1e4,
        require_gap_fixed_point=False,
        backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
    )

    certificate["qp_backward_error"] = 2.0 * DIRECT_GAP_BACKWARD_ERROR_LIMIT
    with pytest.raises(RuntimeError, match="limit=1e-09"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e4,
            require_gap_fixed_point=False,
            backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        )


@pytest.mark.parametrize("obs", (-1.04, -0.0168056837102, 0.0, 1.0))
def test_signed_finite_gap_suppression_ratio_is_acceptable(obs: float) -> None:
    assert fig6_solve._acceptable_ratio(obs)


@pytest.mark.parametrize("obs", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_gap_suppression_ratio_is_rejected(obs: float) -> None:
    assert not fig6_solve._acceptable_ratio(obs)


def test_fixed_gap_solver_falls_back_to_picard(monkeypatch) -> None:
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    state = fig6_solve._build_state(material, spectral, 0.1)
    calls: list[str] = []

    def fail_coupled(*_args, **_kwargs):
        calls.append("coupled")
        raise fig6_solve.CoupledNewtonLineSearchError(
            iteration=1,
            residual_norm=0.5 * fig6_solve.PICARD_TOL,
        )

    def pass_picard(_backend, state_arg, _photon_params):
        calls.append("picard")
        return state_arg

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    monkeypatch.setattr(fig6_solve, "_solve_picard_fixed_gap", pass_picard)

    result = fig6_solve._solve_fixed_gap_kinetics(
        DiffusionBackend(),
        state,
        {"omega_0": OMEGA_0, "n_bar": 1e4, "c_phot": fig6_solve.C_PHOT},
    )

    assert result is state
    assert calls == ["coupled", "picard"]


@pytest.mark.parametrize(
    "error",
    (
        fig6_solve.CoupledNewtonLineSearchError(
            iteration=2,
            residual_norm=2.0 * fig6_solve.PICARD_TOL,
        ),
        RuntimeError("Coupled Newton Jacobian singular at iteration 2."),
    ),
)
def test_fixed_gap_solver_does_not_mask_non_roundoff_failure(
    monkeypatch,
    error: RuntimeError,
) -> None:
    def fail_coupled(*_args, **_kwargs):
        raise error

    def fail_if_picard_runs(*_args, **_kwargs):
        pytest.fail("Picard fallback must be reserved for a roundoff-level stall")

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    monkeypatch.setattr(
        fig6_solve,
        "_solve_picard_fixed_gap",
        fail_if_picard_runs,
    )

    with pytest.raises(type(error), match=str(error).split(" at iteration")[0]):
        fig6_solve._solve_fixed_gap_kinetics(
            DiffusionBackend(),
            object(),  # type: ignore[arg-type]
            None,
        )


def test_direct_point_solve_propagates_fixed_gap_failure(monkeypatch) -> None:
    """The public point path must not relabel fixed-gap errors as folds."""
    expected = RuntimeError("synthetic fixed-gap singular Jacobian")

    def fail_fixed_gap(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(fig6_solve, "_solve_fixed_gap_kinetics", fail_fixed_gap)
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    with pytest.raises(RuntimeError, match="singular Jacobian") as caught:
        fig6_solve._solve_and_measure(
            DiffusionBackend(),
            material,
            spectral,
            0.1,
            1e4,
            None,
            fixed_gap_kinetics=True,
            direct_gap_observable=True,
            thermal_integral=0.0,
            delta_eq=DELTA_0,
            delta_T=1.0,
        )

    assert caught.value is expected


def test_reduced_direct_gap_picard_fallback_is_repeatable_and_certified(
    monkeypatch,
) -> None:
    """Exercise the real strict fallback on a small commensurate grid."""
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "PICARD_TOL", 1e-9)

    def fail_coupled(*_args, **_kwargs):
        raise fig6_solve.CoupledNewtonLineSearchError(
            iteration=1,
            residual_norm=0.5 * fig6_solve.PICARD_TOL,
        )

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    thermal_integral = fig6_solve.thermal_gap_integral_direct(
        spectral.E,
        gap=DELTA_0,
        T_bath=0.1,
        samples="centers",
    )
    delta_eq = DELTA_0 * float(np.exp(-thermal_integral))
    kwargs = {
        "fixed_gap_kinetics": True,
        "direct_gap_observable": True,
        "thermal_integral": thermal_integral,
        "delta_eq": delta_eq,
        "delta_T": DELTA_0 - delta_eq,
    }

    first = fig6_solve._solve_and_measure(
        DiffusionBackend(), material, spectral, 0.1, 1e4, None, **kwargs,
    )
    repeated = fig6_solve._solve_and_measure(
        DiffusionBackend(), material, spectral, 0.1, 1e4, None, **kwargs,
    )

    assert first[0].gap == DELTA_0
    assert first[1] == pytest.approx(repeated[1], rel=0.0, abs=1e-14)
    np.testing.assert_array_equal(first[0].f, repeated[0].f)
    np.testing.assert_array_equal(
        first[0].phonon.n_ph,
        repeated[0].phonon.n_ph,
    )
    for result in (first, repeated):
        certificate = result[4]
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e4,
            require_gap_fixed_point=False,
            backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        )
        assert certificate["qp_backward_error"] < DIRECT_GAP_BACKWARD_ERROR_LIMIT
        # The number-mode amplitude polish can move the returned phonon
        # fixed point to the adjacent representable float. Its balance is
        # then roundoff-small rather than bitwise zero.
        assert certificate["phonon_backward_error"] < 1e-14


def test_legacy_nine_column_baseline_reads_with_nan_certificates(tmp_path) -> None:
    path = tmp_path / "legacy_fig6.csv"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n",
        encoding="utf-8",
    )

    result = fig6_paper._legacy_read_baseline(path)
    for field in (
        "qp_residual_inf",
        "qp_backward_error",
        "phonon_residual_inf",
        "phonon_raw_backward_error",
        "phonon_backward_error",
        "gap_fixed_point_abs_error_uev",
    ):
        assert np.all(np.isnan(getattr(result, field)))


def test_baseline_reader_rejects_duplicate_coordinates(tmp_path) -> None:
    path = tmp_path / "duplicate_fig6.csv"
    row = "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS[:9])
        + "\n"
        + row
        + row,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"duplicate \(T_bath, n_bar\)"):
        fig6_paper._legacy_read_baseline(path)


def test_baseline_reader_rejects_missing_cartesian_coordinate(tmp_path) -> None:
    path = tmp_path / "missing_coordinate_fig6.csv"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS[:9])
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
        "0.1,20000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
        "0.2,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing Cartesian"):
        fig6_paper._legacy_read_baseline(path)


def test_old_thirteen_column_certificate_maps_without_reinterpretation(
    tmp_path,
) -> None:
    path = tmp_path / "old_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53,qp_residual_inf,qp_backward_error,"
        "phonon_residual_inf,phonon_backward_error\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "2e-5,1e-20,1e-8\n",
        encoding="utf-8",
    )
    result = fig6_paper._legacy_read_baseline(path)

    assert result.phonon_backward_error[0, 0] == pytest.approx(1e-8)
    assert np.isnan(result.phonon_raw_backward_error[0, 0])
    assert np.isnan(result.gap_fixed_point_abs_error_uev[0, 0])

    with pytest.raises(
        AssertionError,
        match="current authenticated artifact schema",
    ):
        _assert_certified_baseline_balances(path, result)


def test_current_baseline_preflight_rejects_bad_balance(tmp_path) -> None:
    path = tmp_path / "bad_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS)
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "2e-5,1e-20,1e-8,1e-8,1e-12\n",
        encoding="utf-8",
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="schema marker",
    ):
        read_baseline(path)


def test_current_baseline_preflight_rejects_bad_gap_map_error(tmp_path) -> None:
    path = tmp_path / "bad_gap_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53,qp_residual_inf,qp_backward_error,"
        "phonon_residual_inf,phonon_raw_backward_error,"
        "phonon_backward_error,gap_fixed_point_abs_error_uev\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        f"1e-8,1e-20,1e-8,1e-8,{1.01 * GAP_FIXED_POINT_ABS_TOL_UEV}\n",
        encoding="utf-8",
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="schema marker",
    ):
        read_baseline(path)


def test_current_baseline_preflight_requires_finite_certificate_fields(
    tmp_path,
) -> None:
    path = tmp_path / "nonfinite_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS)
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "1e-8,1e-20,nan,1e-8,1e-12\n",
        encoding="utf-8",
    )
    with pytest.raises(
        fig6_paper.ArtifactValidationError,
        match="schema marker",
    ):
        read_baseline(path)


def test_sweep_carries_full_state_within_row_and_resets_between_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.10, 0.20))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4, 2.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    calls: list[tuple[float, float | None, float | None]] = []
    row_counts = {0.10: 0, 0.20: 0}

    def fake_solve_and_measure(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        continuation_seed,
        **_kwargs,
    ):
        seed_gap = None if continuation_seed is None else continuation_seed.gap
        seed_spectral_gap = (
            None if continuation_seed is None else continuation_seed.spectral.gap
        )
        calls.append((T_bath, seed_gap, seed_spectral_gap))
        state = fig6_solve._build_state(
            material_arg,
            spectral_arg,
            T_bath,
            continuation_seed=continuation_seed,
        )

        row_counts[T_bath] += 1
        target_gap = DELTA_0 - 0.25 * (1 + int(T_bath > 0.15)) - 0.01 * row_counts[
            T_bath
        ]
        converged = replace(
            state,
            gap=target_gap,
            spectral=SpectralContext(
                E_bins=state.spectral.E,
                dE_bins=state.spectral.dE,
                gap=target_gap,
            ),
        )
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        return converged, 0.1, target_gap, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        fake_solve_and_measure,
    )
    result = fig6_solve._solve_sweep(
        DiffusionBackend(),
        material,
        spectral,
        tau_0_pb=0.255,
    )

    assert calls[0] == (0.10, None, None)
    assert calls[1][0] == 0.10
    assert calls[1][1] == calls[1][2] == DELTA_0 - 0.26
    assert calls[2] == (0.20, None, None)
    assert calls[3][0] == 0.20
    assert calls[3][1] == calls[3][2] == DELTA_0 - 0.51

    certificates = result[-3]
    for field in FIG6_CERTIFICATE_FIELDS:
        np.testing.assert_array_equal(certificates[field], np.zeros((2, 2)))


def test_direct_gap_sweep_applies_strict_certificate_limit(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.10,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def fake_solve_and_measure(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        _continuation_seed,
        **_kwargs,
    ):
        state = fig6_solve._build_state(material_arg, spectral_arg, T_bath)
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        certificate["qp_backward_error"] = (
            2.0 * DIRECT_GAP_BACKWARD_ERROR_LIMIT
        )
        certificate["gap_fixed_point_abs_error_uev"] = float("nan")
        return state, 0.1, DELTA_0, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        fake_solve_and_measure,
    )

    with pytest.raises(RuntimeError, match="limit=1e-09"):
        fig6_solve._solve_sweep(
            DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
            direct_gap_observable=True,
            fixed_gap_kinetics=True,
        )


@pytest.mark.parametrize(
    ("direct_gap_observable", "fixed_gap_kinetics"),
    ((True, False), (False, True)),
)
def test_solve_rejects_inconsistent_numerical_modes_before_setup(
    monkeypatch,
    direct_gap_observable: bool,
    fixed_gap_kinetics: bool,
) -> None:
    def fail_if_setup_runs():
        pytest.fail("mode validation must precede the expensive grid setup")

    monkeypatch.setattr(fig6_solve, "_fischer_material", fail_if_setup_runs)

    with pytest.raises(ValueError, match="must be enabled together"):
        fig6_solve.solve(
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=fixed_gap_kinetics,
        )


def test_independent_certificate_runtime_error_propagates(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    monkeypatch.setattr(
        fig6_solve,
        "_solve_picard_sc_gap",
        lambda _backend, state, _photon_params: state,
    )

    def fail_certificate(*_args, **_kwargs):
        raise RuntimeError("independent certificate assembly failed")

    monkeypatch.setattr(
        fig6_solve,
        "steady_state_certificate",
        fail_certificate,
    )

    with pytest.raises(RuntimeError, match="independent certificate assembly failed"):
        fig6_solve._solve_sweep(
            DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
        )


@pytest.mark.parametrize("direct_gap_observable", (False, True))
def test_nonfinite_derived_observable_propagates_through_sweep(
    monkeypatch,
    direct_gap_observable: bool,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def nonfinite_measurement(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        _continuation_seed,
        **_kwargs,
    ):
        state = fig6_solve._build_state(material_arg, spectral_arg, T_bath)
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        if direct_gap_observable:
            certificate["gap_fixed_point_abs_error_uev"] = float("nan")
        return state, float("nan"), DELTA_0, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        nonfinite_measurement,
    )

    with pytest.raises(RuntimeError, match="non-finite derived measurement"):
        fig6_solve._solve_sweep(
            DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=direct_gap_observable,
        )


def test_self_consistent_inner_solver_failure_propagates_through_sweep(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    expected = RuntimeError("synthetic inner Picard exhaustion")

    def fail_inner_solver(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(fig6_solve, "_solve_picard_sc_gap", fail_inner_solver)

    with pytest.raises(RuntimeError, match="inner Picard exhaustion") as caught:
        fig6_solve._solve_sweep(
            DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
        )

    assert caught.value is expected


def test_explicit_self_consistent_collapse_is_recorded_as_nan(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def collapse(*_args, **_kwargs):
        raise fig6_solve.SelfConsistentGapCollapseError(
            iteration=4,
            max_occupation=0.25,
        )

    monkeypatch.setattr(fig6_solve, "_solve_picard_sc_gap", collapse)
    result = fig6_solve._solve_sweep(
        DiffusionBackend(),
        material,
        spectral,
        tau_0_pb=0.255,
    )

    assert np.isnan(result[2][0, 0])
    assert np.isnan(result[3][0, 0])
    assert np.isnan(result[5][0, 0])


@pytest.mark.slow
def test_reduced_full_state_continuation_is_certified_and_repeatable(
    monkeypatch,
) -> None:
    """A real reduced full-state continuation is deterministic and certified."""
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "PICARD_TOL", 1e-9)

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    calibration = calibrate_gap(
        T_c=T_C,
        T_bath=0.20,
        Delta_0=DELTA_0,
        xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
    )
    delta_eq = calibration.delta_eq
    delta_T = DELTA_0 - delta_eq
    solve_kwargs = {
        "fixed_gap_kinetics": False,
        "direct_gap_observable": False,
        "thermal_integral": None,
        "delta_eq": delta_eq,
        "delta_T": delta_T,
    }
    backend = DiffusionBackend()
    first = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e4,
        None,
        **solve_kwargs,
    )
    continued = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e5,
        first[0],
        **solve_kwargs,
    )
    repeated_first = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e4,
        None,
        **solve_kwargs,
    )
    repeated = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e5,
        repeated_first[0],
        **solve_kwargs,
    )

    assert first[0].gap == first[0].spectral.gap
    assert first[0].gap < DELTA_0
    # Round-6 Newton number-mode certification moved this reduced-grid
    # endpoint off the former aggregate-only root.  The old pin had an
    # independently reassembled pair-number backward error of 7.45e-6,
    # above NEWTON_BACKWARD_ERROR_TOL=1e-6; this root is deterministic and
    # improves that error to 4.50e-7.  Keep the original tight pin tolerances.
    assert continued[0].gap == pytest.approx(
        179.9969260485,
        rel=0.0,
        abs=2e-9,
    )
    assert continued[1] == pytest.approx(0.23525642, rel=0.0, abs=1e-6)
    assert continued[0].gap == pytest.approx(
        repeated[0].gap,
        rel=0.0,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        continued[0].f,
        repeated[0].f,
        rtol=0.0,
        atol=1e-14,
    )
    assert continued[1] == pytest.approx(repeated[1], rel=0.0, abs=1e-10)
    for result in (continued, repeated):
        state = result[0]
        certificate = result[4]
        assert certificate["qp_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT
        assert certificate["phonon_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT
        assert (
            certificate["gap_fixed_point_abs_error_uev"]
            <= GAP_FIXED_POINT_ABS_TOL_UEV
        )

        # Independently reassemble the number-changing pair channel from the
        # returned joint (f, n_ph) snapshot.  This is the decisive Round-6
        # gate that invalidated the old absolute pin; aggregate QP turnover
        # alone is dominated by number-conserving scattering.
        K_r0 = build_recombination_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        _, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
            state.spectral.E
        )
        _, N_emit, N_abs = phonon_occupation_matrices_from_state(
            state.phonon.n_ph[0, :, 0], idx_diff, idx_sum, diff_sign
        )
        gain_number, loss_number, configured = number_changing_gain_loss(
            state.f,
            state.spectral,
            K_r0,
            state.T_bath,
            N_emit=N_emit,
            N_abs=N_abs,
        )
        number_error = _weighted_number_backward_error(
            gain_number,
            loss_number,
            state.f,
            state.spectral.cell_weights,
            state.spectral.active_mask,
        )
        assert configured
        assert number_error is not None
        assert number_error <= fig6_solve.NEWTON_BACKWARD_ERROR_TOL


@pytest.mark.slow
@pytest.mark.manual_slow
def test_matches_pinned_baseline() -> None:
    """Run the full 1640-bin paper-parameter sweep (manual validation only).

    The promoted campaign measured 12.075 aggregate worker-hours (4.229 wall
    hours with three concurrent rows), which is not a bounded pull-request
    check. Keep this test executable with
    ``pytest -m 'slow and manual_slow'`` while the author-style fixed-gap,
    direct-Delta[f] path is evaluated as the replacement CI target.
    """
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig6_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the 12.075 h aggregate-worker run() below, instead of after it.
    try:
        _assert_config_matches_baseline(path)
        baseline = read_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(str(exc))
    result = run()

    # τ values --- 1e-8 relative per the pattern in test_fig5_paper.py.
    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)
    assert result.tau_l_ns == pytest.approx(baseline.tau_l_ns, rel=1e-8)

    # Sweep axes match exactly (T_B values are literal floats; n̄ values
    # are np.logspace, so allow a tiny relative slack).
    np.testing.assert_allclose(
        result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.n_bar, baseline.n_bar, rtol=1e-12, atol=0.0,
    )

    # T_*/Δ is closed-form in n̄ — should be reproducible to ~1e-10.
    np.testing.assert_allclose(
        result.T_star_over_delta, baseline.T_star_over_delta,
        rtol=1e-10, atol=0.0,
        err_msg="T_*/Δ axis drift (Eq. 35)",
    )

    # Gap observables — 1e-6 abs (iterative-mode tolerance).
    np.testing.assert_allclose(
        result.delta_eq, baseline.delta_eq, rtol=0.0, atol=1e-6,
        err_msg="Δ_eq(T_B) drift",
    )
    np.testing.assert_allclose(
        result.delta_driven, baseline.delta_driven, rtol=0.0, atol=1e-6,
        err_msg="Δ_driven (self-consistent BCS) drift",
    )
    np.testing.assert_allclose(
        result.x_qp_num, baseline.x_qp_num, rtol=1e-3, atol=1e-14,
        err_msg="numerical x_qp drift",
    )
    # (δΔ_T - δΔ)/δΔ_T divides by δΔ_T = Δ_0 - Δ_eq(T_B), which is only
    # 8.3215e-8 μeV at T_B = 0.10 K (1.0737e-4 at 0.15 K, 4.0196e-3 at 0.20 K).
    # This atol is therefore NOT the gap atol above carried through: 1 μeV of
    # Δ_driven drift lands here amplified by up to 1.2017e7, so the 1e-6
    # headroom granted to Δ_driven is unreachable in practice and this line is
    # the binding constraint of the whole comparison. Against pinned |values|
    # reaching 2.3043e6 an absolute 1e-6 is ~4e-13 relative (≈2 ULP), i.e. it
    # implicitly demands ~8.3e-14 μeV gap reproducibility. Left as-is
    # deliberately: restating it per row against its own suppression scale is
    # blocked on N37 (docs/AUDIT-2026-07-15-numerical-software.md:157), which
    # holds that the 66-point production sweep certifies the solver contract
    # and not this ratio.
    np.testing.assert_allclose(
        result.paper_observable_num, baseline.paper_observable_num,
        rtol=0.0, atol=1e-6,
        err_msg="(δΔ_T - δΔ)/δΔ_T numerical drift",
    )
    # x_qp_eq47 is pure closed-form (Eq. 47 + Appendix-E in T_bath, n_bar,
    # τ_ℓ, τ_0^PB). Pin tightly — drift means a coefficient changed.
    np.testing.assert_allclose(
        result.x_qp_eq47, baseline.x_qp_eq47, rtol=1e-10, atol=0.0,
        err_msg="Eq. 47 analytic x_qp drift",
    )
    # Dashed overlay is Eq. 53 evaluated at (x_qp_eq47, T_*/Δ) and combined
    # with the numerical Δ_eq(T_B). The closed-form Eq. 47 + Eq. 53 part is
    # float64-exact, but this gate does NOT inherit the Δ_eq tolerance via
    # composition: obs_eq53 = 1 - ΔΔ_drive/δΔ_T, so
    # ∂obs_eq53/∂Δ_eq = ΔΔ_drive/δΔ_T² ≈ 3.98e13 per μeV at T_B = 0.10 K
    # (ΔΔ_drive = 0.2759 μeV, δΔ_T = 8.3215e-8 μeV). A 1e-6 μeV Δ_eq drift
    # moves it by ~4e7, not by 1e-6. Same standing constraint as
    # paper_observable_num above (N37).
    np.testing.assert_allclose(
        result.paper_observable_eq53, baseline.paper_observable_eq53,
        rtol=0.0, atol=1e-6,
        err_msg="Eq. 53 dashed-overlay drift",
    )


class TestFig6CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves the otherwise 12.075 h aggregate-worker solve from disk.
    The expensive solve is stubbed so the test is fast; it exercises the real
    cache + observables (pure unpack) wiring. Engine-level key/store properties
    are covered in
    ``tests/validation/test_sweep_cache.py``.
    """

    def _stub_payload(self) -> dict:
        # Synthetic raw payload (fig6's observables is a pure unpack — no grid
        # rebuild — so tiny placeholder arrays suffice). One T_B, two n̄.
        return {
            "tau_0_pb_ns": np.array([0.255]),
            "tau_l_ns": np.array([0.255]),
            "T_bath": np.array([0.20]),
            "n_bar": np.array([1.0e4, 1.0e5]),
            "T_star_over_delta": np.array([[0.10, 0.20]]),
            "delta_eq": np.array([179.9]),
            "delta_driven": np.array([[179.8, 179.5]]),
            "paper_observable_num": np.array([[0.10, 0.20]]),
            "paper_observable_eq53": np.array([[0.11, 0.21]]),
            "x_qp_num": np.array([[1.0e-5, 2.0e-5]]),
            "x_qp_eq47": np.array([[1.1e-5, 2.1e-5]]),
            "qp_residual_inf": np.array([[1.0e-15, 2.0e-15]]),
            "qp_backward_error": np.array([[1.0e-8, 2.0e-8]]),
            "qp_number_backward_error": np.array([[1.1e-8, 2.1e-8]]),
            "phonon_residual_inf": np.array([[3.0e-15, 4.0e-15]]),
            "phonon_raw_backward_error": np.array([[3.1e-8, 4.1e-8]]),
            "phonon_backward_error": np.array([[3.0e-8, 4.0e-8]]),
            "gap_fixed_point_abs_error_uev": np.array([[1.0e-12, 2.0e-12]]),
            "state_f": np.zeros((1, 2, 4)),
            "state_n_ph": np.zeros((1, 2, 7)),
        }

    def test_run_cached_hits_disk_on_second_call(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig6_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "1")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        r1 = fp.run_cached()
        assert calls["n"] == 1  # cache miss -> solve ran once

        r2 = fp.run_cached()
        assert calls["n"] == 1  # cache hit -> solve NOT re-run

        ref = fp.observables(payload)
        for res in (r1, r2):
            for fld in ("paper_observable_num", "paper_observable_eq53",
                        "delta_driven", "x_qp_num", "x_qp_eq47",
                         "qp_residual_inf", "qp_backward_error",
                         "phonon_residual_inf", "phonon_raw_backward_error",
                         "phonon_backward_error",
                         "gap_fixed_point_abs_error_uev"):
                np.testing.assert_array_equal(getattr(res, fld), getattr(ref, fld))

    def test_certified_payload_csv_round_trip(
        self,
        tmp_path,
        schema_result,
    ) -> None:
        import validation.fischer_2023.fig6_paper as fp

        reference = schema_result
        path = fp.write_baseline(reference, tmp_path / "certified_fig6.csv")
        restored = fp.read_baseline(path)

        payload = path.read_bytes()
        assert b"\r\n" not in payload
        assert payload.endswith(b"\n")

        for field in (
            "qp_residual_inf",
            "qp_backward_error",
            "qp_number_backward_error",
            "phonon_residual_inf",
            "phonon_raw_backward_error",
            "phonon_backward_error",
            "gap_fixed_point_abs_error_uev",
        ):
            np.testing.assert_array_equal(
                getattr(restored, field),
                getattr(reference, field),
            )

        metadata = fp.read_baseline_metadata(path)
        assert metadata.finite_cutoff_delta0_over_kbtc == pytest.approx(
            FINITE_CUTOFF_DELTA0_OVER_KBTC,
            rel=0.0,
            abs=1e-15,
        )
        assert metadata.t_c == pytest.approx(T_C, rel=0.0, abs=1e-15)
        assert metadata.gap_fixed_point_abs_tol_uev == pytest.approx(
            GAP_FIXED_POINT_ABS_TOL_UEV,
            rel=0.0,
            abs=0.0,
        )
        assert (
            metadata.certificate_metric_version
            == fp.certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
        )

    def test_reforged_arbitrary_curves_and_certificates_are_rejected(
        self,
        tmp_path,
        schema_result,
    ) -> None:
        """A fresh checksum cannot turn invented curves into validation."""
        import validation.fischer_2023.fig6_paper as fp

        path = fp.write_baseline(schema_result, tmp_path / "source.csv")
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
        header = rows[2]
        for name in (
            "x_qp_num",
            "x_qp_eq47",
            "paper_observable_num",
            "paper_observable_eq53",
        ):
            rows[3][header.index(name)] = "1.23000000000000000e+02"
        for name in fp._ARTIFACT_CERTIFICATE_FIELDS:
            rows[3][header.index(name)] = "0.00000000000000000e+00"
        prefix = "# qpsim_metadata="
        metadata = json.loads(rows[1][0][len(prefix):])
        metadata["payload_sha256"] = fp._payload_sha256(rows[3:])
        rows[1][0] = prefix + fp._canonical_json(metadata)
        forged = tmp_path / "forged.csv"
        with forged.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream, lineterminator="\n").writerows(rows)

        with pytest.raises(
            fp.ArtifactValidationError,
            match="persisted solver state",
        ):
            fp.read_baseline(forged)

    def test_baseline_write_is_atomic_on_failure(
        self,
        tmp_path,
        schema_result,
    ) -> None:
        import validation.fischer_2023.fig6_paper as fp

        path = tmp_path / "certified_fig6.csv"
        path.write_text("previous-good-baseline\n", encoding="utf-8")
        reference = schema_result
        malformed = replace(
            reference,
            T_star_over_delta=np.empty((0, 0)),
        )

        with pytest.raises(fp.ArtifactValidationError):
            fp.write_baseline(malformed, path)

        assert path.read_text(encoding="utf-8") == "previous-good-baseline\n"
        assert not path.with_name(f".{path.name}.{fp.os.getpid()}.tmp").exists()

    def test_run_cached_disabled_always_recomputes(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig6_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached()
        fp.run_cached()
        assert calls["n"] == 2  # disabled -> recompute each call
