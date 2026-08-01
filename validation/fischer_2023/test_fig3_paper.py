"""Regression test: Fischer 2023 Fig. 3 qpsim run matches the pinned CSV.

Slow-marked: this run does the τ_l = 0 thermal-phonon Newton plus the
13-step branch-preserving Picard continuation, with a same-ratio coupled-
Newton polish at ratio 10 on the 1620-bin paper grid. Curve tolerances scale
with the pinned signal rather than using the former vacuous absolute ``1e-6``.
Opt in with ``pytest -m slow``.

First-time generation::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import fermi_dirac_occupation
from qpsim.solvers.anderson import AndersonAccelerationError

from validation.fischer_2023 import fig3_paper as fig3_target
from validation.fischer_2023.fig3_paper import (
    CURVE_REGRESSION_ATOL_OVER_PEAK,
    CURVE_REGRESSION_RTOL,
    STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL,
    VALIDATION_RECORD_SCHEMA,
    Fig3PaperResult,
    baseline_path,
    config_metadata,
    curve_regression_rtol,
    observables,
    read_baseline,
    read_baseline_metadata,
    read_validation_record,
    run,
    validation_record_path,
    write_baseline,
)
from validation.fischer_2023.fig3_solve import (
    CONTINUATION_RATIOS,
    DELTA_0,
    INNER_QP_BACKWARD_ERROR_LIMIT,
    INNER_QP_NUMBER_POLISH_SHAPE_LIMIT,
    PAPER_RATIOS,
    TARGET_BACKWARD_ERROR_LIMIT,
    Fig3StepEvent,
    _solve_coupled_newton,
    _solve_picard,
    _solve_picard_predictor,
    _solve_tau_l_zero,
    _validate_ratio_ladder,
)
from validation.fischer_2023.fig3_solve import solve as solve_raw
from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_FIELDS as CERTIFICATE_FIELDS,
)
from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_METRIC_VERSION as CERTIFICATE_METRIC_VERSION,
)

# Ratios through one retain the strict 1e-4 curve gate.  The residual-polished
# ratio-10 state uses its measured 1.5% fixed-grid envelope only for the
# Windows/Linux OS-family case. The peak-scaled absolute floor is
# O(1e-16..1e-14), not the old vacuous 1e-6.
_TEST_SOLVE_CONTRACT_DIGEST = "0" * 64


def _small_certified_payload() -> dict[str, np.ndarray]:
    ratios = np.asarray([0.0, 0.1, 1.0, 10.0])
    return {
        "E": np.asarray([180.5, 181.5, 182.5]),
        "f_FD": np.asarray([3e-9, 2e-9, 1e-9]),
        "f_ratios": np.full((ratios.size, 3), 1e-8),
        "ratios": ratios,
        "tau_0_pb_ns": np.asarray([0.255]),
        "qp_residual_inf": np.asarray([1e-20, 2e-20, 3e-20, 4e-20]),
        "qp_backward_error": np.asarray([1e-12, 2e-12, 3e-12, 4e-12]),
        "qp_number_backward_error": np.asarray([5e-13, 6e-13, 7e-13, 8e-13]),
        "phonon_residual_inf": np.asarray([np.nan, 2e-18, 3e-18, 4e-18]),
        "phonon_raw_backward_error": np.asarray([np.nan, 2e-9, 3e-9, 4e-9]),
        "phonon_backward_error": np.asarray([np.nan, 2e-8, 3e-8, 4e-8]),
    }


def _small_certified_result() -> Fig3PaperResult:
    return observables(
        _small_certified_payload(),
        producer_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
        validated_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
    )


def _assert_baseline_curves_are_nonvacuous(baseline) -> None:
    """Fast tripwire against empty/collapsed paper-curve baselines."""
    dE = integration_widths_from_centers(baseline.E)
    capacity = bcs_dos_cell_weights(baseline.E, dE, DELTA_0)
    peaks: list[float] = []
    integrals: list[float] = []
    for ratio in baseline.ratios:
        curve = np.asarray(baseline.f_by_ratio[ratio], dtype=float)
        assert np.all(np.isfinite(curve)), f"ratio {ratio:g} baseline is non-finite"
        assert np.all(curve >= 0.0), f"ratio {ratio:g} baseline has negative occupation"
        peak = float(np.max(curve))
        assert peak > 1e-14, (
            f"ratio {ratio:g} baseline has no resolved occupation signal "
            f"(peak={peak:.3e})"
        )
        assert np.count_nonzero(curve) > curve.size // 2, (
            f"ratio {ratio:g} baseline is mostly/all zero"
        )
        peaks.append(peak)
        integrals.append(float(np.sum(capacity * curve)))

    assert np.all(np.diff(peaks) > 0.0), (
        f"bottleneck-curve peaks must increase with tau_l/tau_0^PB: {peaks}"
    )
    assert np.all(np.diff(integrals) > 0.0), (
        "integrated occupations must increase with tau_l/tau_0^PB: "
        f"{integrals}"
    )


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header.

    Gating :func:`run` (the multi-hour full-grid continuation ladder) behind this
    turns a stale config/baseline pairing — a grid change, a ratio-set edit, a
    τ_0^PB drift — into a seconds-long failure instead of one discovered only
    after the full run. (See ``fig6_paper`` for the same pattern, where
    ``run()`` is ~14 h.)
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)

    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_bath == pytest.approx(meta.t_bath)
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.n_bar == pytest.approx(meta.n_bar)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)
    assert cfg.ratios == meta.ratios, (
        f"paper ratios config={cfg.ratios} != baseline {meta.ratios}; "
        "regenerate the baseline or restore the ratio set before the slow run."
    )
    assert cfg.certificate_metric_version == meta.certificate_metric_version
    assert cfg.certificate_metric_version == CERTIFICATE_METRIC_VERSION
    assert cfg.target_backward_error_limit == pytest.approx(
        meta.target_backward_error_limit,
    )
    assert (
        cfg.validated_solve_contract_digest
        == meta.validated_solve_contract_digest
    )
    assert len(meta.producer_solve_contract_digest) == 64
    assert meta.pinned_on
    assert set(meta.certificate_maxima) == set(CERTIFICATE_FIELDS)
    assert np.all(np.isfinite(tuple(meta.certificate_maxima.values())))
    for field in (
        "qp_backward_error",
        "qp_number_backward_error",
        "phonon_backward_error",
    ):
        assert meta.certificate_maxima[field] <= TARGET_BACKWARD_ERROR_LIMIT


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


def test_canonical_validation_record_authenticates_promoted_pair() -> None:
    """The promoted CSV/PDF must be the exact pair accepted by the verifier."""
    csv_path = baseline_path()
    pdf_path = csv_path.with_suffix(".pdf")
    record_path = validation_record_path()

    for path in (csv_path, pdf_path, record_path):
        assert path.is_file(), f"Canonical Fig. 3 artifact is missing: {path}"
        assert path.stat().st_size > 0, f"Canonical Fig. 3 artifact is empty: {path}"

    record = read_validation_record()
    assert record["schema"] == VALIDATION_RECORD_SCHEMA
    assert record["status"] == "pass"
    assert record["verifier"]["source_unchanged"] is True
    assert record["producer"]["current_solve_contract_payload"] is True

    cfg = config_metadata()
    validated_digest = cfg.validated_solve_contract_digest
    assert (
        record["producer"]["source_identity"]["solve_contract_digest"]
        == validated_digest
    )
    artifacts = record["artifacts"]
    assert artifacts["readback"]["strict_read_passed"] is True
    assert (
        artifacts["readback"]["metadata"]["validated_solve_contract_digest"]
        == validated_digest
    )
    assert len(record["fresh_certificate_reassembly"]["rows"]) == len(
        PAPER_RATIOS
    )


def test_triple_publisher_authenticates_and_promotes_record_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    csv_path = tmp_path / "fig3.csv"
    pdf_path = tmp_path / "fig3.pdf"
    record_path = tmp_path / "fig3.validation.json"
    monkeypatch.setattr(fig3_target, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fig3_target, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(fig3_target, "validation_record_path", lambda: record_path)

    source_identity = fig3_target._validation_source_identity()
    runtime = fig3_target._validation_runtime_provenance()
    digest = source_identity["solve_contract_digest"]
    _E, _dE, spectral = fig3_target._build_grid_and_spectral()
    raw = _small_certified_payload()
    raw["E"] = spectral.E
    raw["f_FD"] = fermi_dirac_occupation(spectral.E, fig3_target.T_BATH)
    raw["f_ratios"] = np.full((len(PAPER_RATIOS), spectral.E.size), 1e-8)
    raw["tau_0_pb_ns"] = np.asarray(
        [fig3_target.config_metadata().tau_0_pb_ns]
    )
    result = fig3_target.observables(
        raw,
        producer_solve_contract_digest=digest,
        validated_solve_contract_digest=digest,
    )
    _kwargs, _fingerprint, _extra_source, cache_identity = (
        fig3_target._solve_cache_inputs(
            num_bins=fig3_target.NUM_BINS,
            paper_ratios=PAPER_RATIOS,
            continuation_ratios=fig3_target.CONTINUATION_RATIOS,
        )
    )
    producer_evidence = {
        "cache": {
            "array_payload_sha256": (
                fig3_target.sweep_cache._array_payload_sha256(raw)
            ),
            "cache_enabled": False,
            "execution_mode": "solver_invoked",
        },
        "restart": {
            "canonical_request": _kwargs,
            "checkpoint_existed_before": False,
            "checkpoint_identity": cache_identity,
            "checkpoint_path": None,
            "qualification": "unit-test restart qualification",
        },
    }

    original_write_plot = fig3_target.write_plot

    def fake_plot(_result, path=None):
        assert path is not None
        return original_write_plot(_result, path)

    monkeypatch.setattr(fig3_target, "write_plot", fake_plot)
    def fake_reassembly(_result):
        return {
            "maxima": result.certificate_maxima,
            "qualification": "unit-test qualification",
            "rows": [
                {
                    "ratio": ratio,
                    "persisted_f_sha256": fig3_target._sha256_array(
                        result.f_by_ratio[ratio]
                    ),
                    "certificate": {
                        field: (
                            None
                            if ratio == 0.0 and field.startswith("phonon_")
                            else result.certificate_maxima[field]
                        )
                        for field in CERTIFICATE_FIELDS
                    },
                    "reconstructed_n_ph": {
                        "max": 0.0,
                        "min": 0.0,
                        "sha256": "0" * 64,
                    },
                }
                for ratio in PAPER_RATIOS
            ],
            "scope": "unit-test scope",
        }

    monkeypatch.setattr(
        fig3_target,
        "_reassemble_artifact_certificates",
        fake_reassembly,
    )

    forged_origin = json.loads(json.dumps(producer_evidence))
    forged_origin["cache"]["array_payload_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match=r"raw solve payload"):
        fig3_target.publish_baseline_triple(
            result,
            source_identity=source_identity,
            producer_runtime=runtime,
            producer_evidence=forged_origin,
        )

    noncanonical_origin = json.loads(json.dumps(producer_evidence))
    noncanonical_origin["restart"]["canonical_request"][
        "continuation_ratios"
    ] = [0.1, 1.0, 10.0]
    with pytest.raises(RuntimeError, match=r"invalid restart evidence"):
        fig3_target.publish_baseline_triple(
            result,
            source_identity=source_identity,
            producer_runtime=runtime,
            producer_evidence=noncanonical_origin,
        )

    for artifact_name in ("csv", "pdf"):
        def mutate_stage(_result, *, _artifact_name=artifact_name):
            stage = next(tmp_path.glob(f".*.stage.{_artifact_name}"))
            if _artifact_name == "csv":
                stage.write_bytes(stage.read_bytes() + b"\n")
            else:
                stage.write_bytes(
                    b"%PDF-1.4\nUNRELATED-COMPLETE-PDF"
                    + b"x" * 2048
                    + b"\n%%EOF\n"
                )
            return fake_reassembly(_result)

        monkeypatch.setattr(
            fig3_target,
            "_reassemble_artifact_certificates",
            mutate_stage,
        )
        with pytest.raises(RuntimeError, match=r"changed during semantic"):
            fig3_target.publish_baseline_triple(
                result,
                source_identity=source_identity,
                producer_runtime=runtime,
                producer_evidence=producer_evidence,
            )
        assert not csv_path.exists()
        assert not pdf_path.exists()
        assert not record_path.exists()
    monkeypatch.setattr(
        fig3_target,
        "_reassemble_artifact_certificates",
        fake_reassembly,
    )

    original_write_baseline = fig3_target.write_baseline

    def swap_pdf_while_writing_csv(_result, path=None):
        pdf_stage = next(tmp_path.glob(".*.stage.pdf"))
        pdf_stage.write_bytes(
            b"%PDF-1.4\nPREFREEZE-SUBSTITUTION"
            + b"x" * 2048
            + b"\n%%EOF\n"
        )
        return original_write_baseline(_result, path)

    monkeypatch.setattr(
        fig3_target,
        "write_baseline",
        swap_pdf_while_writing_csv,
    )
    with pytest.raises(RuntimeError, match=r"PDF changed while its CSV"):
        fig3_target.publish_baseline_triple(
            result,
            source_identity=source_identity,
            producer_runtime=runtime,
            producer_evidence=producer_evidence,
        )
    assert not csv_path.exists()
    assert not pdf_path.exists()
    assert not record_path.exists()
    monkeypatch.setattr(
        fig3_target,
        "write_baseline",
        original_write_baseline,
    )

    promoted: list[Path] = []
    original_replace = os.replace

    def tracking_replace(source, destination):
        destination_path = Path(destination)
        if destination_path in {pdf_path, csv_path, record_path}:
            promoted.append(destination_path)
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", tracking_replace)
    assert fig3_target.publish_baseline_triple(
        result,
        source_identity=source_identity,
        producer_runtime=runtime,
        producer_evidence=producer_evidence,
    ) == (csv_path, pdf_path, record_path)
    assert promoted == [pdf_path, csv_path, record_path]
    assert fig3_target.read_validation_record()["status"] == "pass"
    fig3_target.read_baseline()
    fig3_target.read_baseline_metadata()

    original_csv = csv_path.read_bytes()
    original_pdf = pdf_path.read_bytes()
    original_record = record_path.read_bytes()

    pdf_path.write_bytes(
        b"%PDF-1.4\n"
        + b"not a PDF object graph\n"
        + b"x" * 2048
        + b"\n%%EOF\n"
    )
    token_shaped_pdf = json.loads(original_record)
    token_shaped_pdf["artifacts"]["pdf"] = {
        "sha256": fig3_target._sha256_file(pdf_path),
        "size_bytes": pdf_path.stat().st_size,
    }
    record_path.write_text(
        json.dumps(token_shaped_pdf, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="one-page Matplotlib PDF"):
        fig3_target.read_validation_record()
    pdf_path.write_bytes(original_pdf)
    record_path.write_bytes(original_record)

    contradicted_payload = json.loads(original_record)
    contradicted_payload["producer"]["payload_origin"]["cache"][
        "array_payload_sha256"
    ] = "0" * 64
    record_path.write_text(
        json.dumps(contradicted_payload, allow_nan=False),
        encoding="utf-8",
    )
    for reader in (
        fig3_target.read_baseline,
        fig3_target.read_baseline_metadata,
        fig3_target.read_validation_record,
    ):
        with pytest.raises(RuntimeError, match="recorded raw payload"):
            reader()
    record_path.write_bytes(original_record)

    contradicted_runtime = json.loads(original_record)
    contradicted_runtime["verifier"]["runtime_after"] = {"contradiction": True}
    record_path.write_text(
        json.dumps(contradicted_runtime, allow_nan=False),
        encoding="utf-8",
    )
    for reader in (
        fig3_target.read_baseline,
        fig3_target.read_baseline_metadata,
        fig3_target.read_validation_record,
    ):
        with pytest.raises(RuntimeError, match="not bound to current source"):
            reader()
    record_path.write_bytes(original_record)

    empty_row_evidence = json.loads(original_record)
    empty_row_evidence["fresh_certificate_reassembly"]["rows"][0][
        "certificate"
    ] = {}
    record_path.write_text(
        json.dumps(empty_row_evidence, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete certificate"):
        fig3_target.read_validation_record()
    record_path.write_bytes(original_record)

    stale_digest = "0" * 64
    csv_path.write_text(
        original_csv.decode("utf-8").replace(digest, stale_digest, 1),
        encoding="utf-8",
        newline="",
    )
    contradicted_csv = json.loads(original_record)
    contradicted_csv["artifacts"]["csv"] = {
        "sha256": fig3_target._sha256_file(csv_path),
        "size_bytes": csv_path.stat().st_size,
    }
    contradicted_csv["artifacts"]["readback"]["metadata"][
        "producer_solve_contract_digest"
    ] = stale_digest
    record_path.write_text(
        json.dumps(contradicted_csv, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match=r"current Fig. 3 contract"):
        fig3_target.read_validation_record()
    csv_path.write_bytes(original_csv)
    record_path.write_bytes(original_record)

    pdf_path.write_bytes(b"%PDF-1.4\ntampered\n%%EOF\n")
    with pytest.raises(RuntimeError, match="does not authenticate"):
        fig3_target.read_validation_record()


def test_reassembly_preflight_rejects_forged_thermal_reference() -> None:
    result = _small_certified_result()
    expected = replace(
        result,
        f_FD=fermi_dirac_occupation(result.E, fig3_target.T_BATH),
    )
    fig3_target._validate_persisted_grid_and_thermal_reference(
        expected,
        expected_E=result.E.copy(),
    )

    portable = np.asarray(expected.f_FD, dtype=float).copy()
    for _ in range(fig3_target.THERMAL_REFERENCE_BINDING_ULPS):
        portable[0] = np.nextafter(portable[0], np.inf)
    fig3_target._validate_persisted_grid_and_thermal_reference(
        replace(expected, f_FD=portable),
        expected_E=result.E.copy(),
    )

    outside_envelope = portable.copy()
    outside_envelope[0] = np.nextafter(outside_envelope[0], np.inf)
    with pytest.raises(RuntimeError, match="thermal reference"):
        fig3_target._validate_persisted_grid_and_thermal_reference(
            replace(expected, f_FD=outside_envelope),
            expected_E=result.E.copy(),
        )

    forged = replace(expected, f_FD=np.full_like(expected.f_FD, 0.123))
    with pytest.raises(RuntimeError, match="thermal reference"):
        fig3_target._validate_persisted_grid_and_thermal_reference(
            forged,
            expected_E=result.E.copy(),
        )


def test_canonical_publication_lock_rejects_concurrent_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    csv_path = tmp_path / "fig3.csv"
    pdf_path = tmp_path / "fig3.pdf"
    record_path = tmp_path / "fig3.validation.json"
    monkeypatch.setattr(fig3_target, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fig3_target, "plot_path", lambda: pdf_path)
    monkeypatch.setattr(fig3_target, "validation_record_path", lambda: record_path)
    with (  # noqa: SIM117 - the inner acquisition must occur under raises
        fig3_target._publication_lock(record_path),
        pytest.raises(RuntimeError, match=r"Another Fig. 3 publisher"),
    ):
        with fig3_target._publication_lock(record_path):
            pytest.fail("A second publisher acquired the canonical lock.")

    with fig3_target._publication_lock(record_path):
        for reader in (
            fig3_target.read_baseline,
            fig3_target.read_baseline_metadata,
            fig3_target.read_validation_record,
        ):
            with pytest.raises(
                RuntimeError,
                match=r"Another Fig. 3 publisher",
            ):
                reader()


def test_baseline_curves_are_nonvacuous() -> None:
    """Fast data sanity gate: every pinned legend curve carries signal."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    try:
        baseline = read_baseline(path)
    except RuntimeError as exc:
        pytest.xfail(f"Canonical Fig. 3 artifact is stale/quarantined: {exc}")
    _assert_baseline_curves_are_nonvacuous(baseline)


def test_baseline_roundtrip_preserves_certificate_maxima(tmp_path) -> None:
    expected = replace(
        _small_certified_result(),
        producer_solve_contract_digest="1" * 64,
        validated_solve_contract_digest="2" * 64,
    )
    path = write_baseline(expected, tmp_path / "fig3.csv")
    restored = read_baseline(path)
    metadata = read_baseline_metadata(path)

    assert restored.certificate_maxima == expected.certificate_maxima
    assert (
        restored.producer_solve_contract_digest
        == expected.producer_solve_contract_digest
    )
    assert (
        restored.validated_solve_contract_digest
        == expected.validated_solve_contract_digest
    )
    assert metadata.certificate_metric_version == CERTIFICATE_METRIC_VERSION
    assert metadata.target_backward_error_limit == TARGET_BACKWARD_ERROR_LIMIT
    assert metadata.certificate_maxima == expected.certificate_maxima
    assert metadata.pinned_on == sys.platform
    assert (
        metadata.producer_solve_contract_digest
        == "1" * 64
    )
    assert (
        metadata.validated_solve_contract_digest
        == "2" * 64
    )
    header = path.read_text(encoding="utf-8")
    assert f"# pinned_on: {sys.platform}" in header
    assert "certificate_metric_version=" in header
    assert "target_backward_error_limit=1e-05" in header
    assert b"\r\n" not in path.read_bytes()

    legacy_path = tmp_path / "fig3-cp1252.csv"
    legacy_path.write_text(header, encoding="cp1252")
    legacy = read_baseline(legacy_path)
    assert legacy.certificate_maxima == expected.certificate_maxima


def _replace_occupation(
    result: Fig3PaperResult,
    field: str,
    value: float | complex,
) -> Fig3PaperResult:
    dtype = complex if np.iscomplexobj(value) else float
    if field == "f_FD":
        thermal = np.asarray(result.f_FD, dtype=dtype).copy()
        thermal[0] = value
        return replace(result, f_FD=thermal)
    if field == "ratio":
        ratio = result.ratios[0]
        curves = {
            key: np.asarray(curve, dtype=dtype).copy()
            for key, curve in result.f_by_ratio.items()
        }
        curves[ratio][0] = value
        return replace(result, f_by_ratio=curves)
    raise AssertionError(f"unsupported test field {field!r}")


@pytest.mark.parametrize("field", ["f_FD", "ratio"])
@pytest.mark.parametrize(
    "bad_value",
    [np.nextafter(0.0, -np.inf), np.nextafter(1.0, np.inf)],
)
def test_writer_rejects_out_of_domain_occupation_without_replacing_sentinel(
    tmp_path: Path,
    field: str,
    bad_value: float,
) -> None:
    result = _replace_occupation(_small_certified_result(), field, bad_value)
    path = tmp_path / "preserved_fig3.csv"
    path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"inclusive \[0, 1\]"):
        write_baseline(result, path)
    assert path.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize("field", ["f_FD", "ratio"])
def test_writer_rejects_complex_occupation_without_replacing_sentinel(
    tmp_path: Path,
    field: str,
) -> None:
    smallest_imaginary = np.nextafter(0.0, np.inf)
    result = _replace_occupation(
        _small_certified_result(),
        field,
        complex(0.5, smallest_imaginary),
    )
    path = tmp_path / "preserved_complex_fig3.csv"
    path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="real-valued"):
        write_baseline(result, path)
    assert path.read_text(encoding="utf-8") == "sentinel\n"


def test_plot_rejects_out_of_domain_occupation(tmp_path: Path) -> None:
    result = _replace_occupation(
        _small_certified_result(),
        "ratio",
        np.nextafter(1.0, np.inf),
    )
    path = tmp_path / "invalid.pdf"

    with pytest.raises(RuntimeError, match=r"inclusive \[0, 1\]"):
        fig3_target.write_plot(result, path)
    assert not path.exists()


def test_plot_rejects_complex_occupation(tmp_path: Path) -> None:
    result = _replace_occupation(
        _small_certified_result(),
        "ratio",
        complex(0.5, np.nextafter(0.0, np.inf)),
    )
    path = tmp_path / "complex.pdf"

    with pytest.raises(RuntimeError, match="real-valued"):
        fig3_target.write_plot(result, path)
    assert not path.exists()


@pytest.mark.parametrize("column", [1, 2])
@pytest.mark.parametrize(
    "bad_value",
    [np.nextafter(0.0, -np.inf), np.nextafter(1.0, np.inf)],
)
def test_reader_rejects_out_of_domain_occupation(
    tmp_path: Path,
    column: int,
    bad_value: float,
) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "bad-domain.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("E_uev"))
    row = lines[header_index + 1].split(",")
    row[column] = repr(float(bad_value))
    lines[header_index + 1] = ",".join(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"inclusive \[0, 1\]"):
        read_baseline(path)


@pytest.mark.parametrize("column", [1, 2])
def test_reader_rejects_complex_occupation(
    tmp_path: Path,
    column: int,
) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "complex.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("E_uev"))
    row = lines[header_index + 1].split(",")
    row[column] = "5.00000000000000000e-01+4.94065645841246544e-324j"
    lines[header_index + 1] = ",".join(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="non-numeric data row"):
        read_baseline(path)


def test_occupation_domain_is_inclusive_at_exact_zero_and_one(
    tmp_path: Path,
) -> None:
    result = _small_certified_result()
    thermal = np.asarray(result.f_FD, dtype=float).copy()
    thermal[:2] = (0.0, 1.0)
    ratio = result.ratios[0]
    curves = {
        key: np.asarray(curve, dtype=float).copy()
        for key, curve in result.f_by_ratio.items()
    }
    curves[ratio][:2] = (1.0, 0.0)
    expected = replace(result, f_FD=thermal, f_by_ratio=curves)

    restored = read_baseline(
        write_baseline(expected, tmp_path / "inclusive-domain.csv")
    )
    np.testing.assert_array_equal(restored.f_FD[:2], [0.0, 1.0])
    np.testing.assert_array_equal(restored.f_by_ratio[ratio][:2], [1.0, 0.0])


def test_observables_rejects_missing_number_certificate() -> None:
    raw = {
        "E": np.asarray([180.5, 181.5, 182.5]),
        "f_FD": np.full(3, 1e-9),
        "f_ratios": np.full((4, 3), 1e-8),
        "ratios": np.asarray([0.0, 0.1, 1.0, 10.0]),
        "tau_0_pb_ns": np.asarray([0.255]),
    }
    raw.update(
        {
            field: np.zeros(4)
            for field in CERTIFICATE_FIELDS
            if field != "qp_number_backward_error"
        }
    )
    for field in CERTIFICATE_FIELDS:
        if field.startswith("phonon_"):
            raw[field][0] = np.nan

    with pytest.raises(ValueError, match="qp_number_backward_error"):
        observables(
            raw,
            producer_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
            validated_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
        )


@pytest.mark.parametrize(
    ("field", "index", "value"),
    (
        ("qp_number_backward_error", 1, np.nan),
        ("qp_backward_error", 2, -1.0),
        ("phonon_backward_error", 0, 0.0),
        ("phonon_raw_backward_error", 2, np.inf),
    ),
)
def test_observables_rejects_invalid_per_ratio_certificate(
    field: str,
    index: int,
    value: float,
) -> None:
    raw = _small_certified_payload()
    raw[field][index] = value

    with pytest.raises(ValueError, match=field):
        observables(
            raw,
            producer_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
            validated_solve_contract_digest=_TEST_SOLVE_CONTRACT_DIGEST,
        )


def test_curve_regression_policy_is_platform_and_ratio_scoped() -> None:
    assert curve_regression_rtol(
        10.0, pinned_on="win32", running_on="win32"
    ) == pytest.approx(CURVE_REGRESSION_RTOL)
    assert curve_regression_rtol(
        1.0, pinned_on="win32", running_on="linux"
    ) == pytest.approx(CURVE_REGRESSION_RTOL)
    assert curve_regression_rtol(
        10.0, pinned_on="win32", running_on="darwin"
    ) == pytest.approx(CURVE_REGRESSION_RTOL)
    portable = curve_regression_rtol(
        10.0, pinned_on="win32", running_on="linux"
    )
    assert portable == pytest.approx(STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            [1.013], [1.0], rtol=CURVE_REGRESSION_RTOL, atol=0.0
        )
    np.testing.assert_allclose([1.013], [1.0], rtol=portable, atol=0.0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([1.02], [1.0], rtol=portable, atol=0.0)


def test_uncertified_baseline_is_rejected(tmp_path) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "uncertified.csv")
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "certificate_metric_version=" not in line
        and not line.startswith("# certificate_maxima")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="certificate metric"):
        read_baseline(path)


def test_invalid_pin_platform_records_are_rejected(tmp_path) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "unstamped.csv")
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("# pinned_on:")
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="pin-platform"):
        read_baseline(path)

    path = write_baseline(_small_certified_result(), tmp_path / "duplicated.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.insert(2, f"# pinned_on: {sys.platform}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="pin-platform"):
        read_baseline(path)


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("qp_residual_inf", "nan", "finite and non-negative"),
        (
            "qp_backward_error",
            f"{2.0 * TARGET_BACKWARD_ERROR_LIMIT:.17e}",
            "above target",
        ),
        (
            "qp_number_backward_error",
            f"{2.0 * TARGET_BACKWARD_ERROR_LIMIT:.17e}",
            "above target",
        ),
    ],
)
def test_bad_certificate_maximum_is_rejected(
    tmp_path,
    field: str,
    replacement: str,
    match: str,
) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "bad_maximum.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# certificate_maxima"):
            parts = line.split()
            field_index = next(
                i for i, part in enumerate(parts) if part.startswith(f"{field}=")
            )
            parts[field_index] = f"{field}={replacement}"
            lines[index] = " ".join(parts)
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=match):
        read_baseline(path)


def test_missing_certificate_maximum_is_rejected(tmp_path) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "missing_maximum.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if line.startswith("# certificate_maxima"):
            lines[index] = " ".join(
                part
                for part in line.split()
                if not part.startswith("phonon_backward_error=")
            )
            break
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fields are incomplete"):
        read_baseline(path)


def test_wrong_ne_row_count_is_rejected(tmp_path) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "short.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="2 data rows; expected NE=3"):
        read_baseline(path)


def test_wrong_curve_column_schema_is_rejected(tmp_path) -> None:
    path = write_baseline(_small_certified_result(), tmp_path / "columns.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(i for i, line in enumerate(lines) if line.startswith("E_uev"))
    lines[header_index] = lines[header_index].rsplit(",", 1)[0]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="has columns"):
        read_baseline(path)


def test_invalid_result_does_not_replace_existing_artifact(tmp_path) -> None:
    reference = _small_certified_result()
    bad = replace(reference, certificate_maxima={})
    path = tmp_path / "preserved_fig3.csv"
    path.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="fields are incomplete"):
        write_baseline(bad, path)
    assert path.read_text(encoding="utf-8") == "sentinel\n"


@pytest.mark.parametrize(
    ("paper_ratios", "continuation_ratios", "match"),
    [
        ((0.0, -1.0), (0.1,), "finite and non-negative"),
        ((0.0, 1.0, 1.0), (0.1, 1.0), "duplicates"),
        ((0.0, 1.0), (0.5, 0.1, 1.0), "strictly increasing"),
        ((0.0, 1.0, 10.0), (0.1, 1.0), "missing.*10"),
    ],
)
def test_ratio_ladder_rejects_invalid_continuation_before_solve(
    paper_ratios: tuple[float, ...],
    continuation_ratios: tuple[float, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_ratio_ladder(paper_ratios, continuation_ratios)


def test_fig3_threads_inner_qp_resolution_limit_to_nested_newton() -> None:
    class BackendStub:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def steady_state(self, state, **kwargs):
            self.kwargs = kwargs
            return state

    backend = BackendStub()
    state = object()
    photon_params = {"omega_0": 20.0, "n_bar": 1e7, "c_phot": 1e-9}

    assert _solve_tau_l_zero(backend, state, photon_params) is state
    assert (
        backend.kwargs["newton_backward_error_tol"]
        == INNER_QP_BACKWARD_ERROR_LIMIT
    )
    assert (
        backend.kwargs["newton_number_polish_shape_tol"]
        == INNER_QP_NUMBER_POLISH_SHAPE_LIMIT
    )

    assert _solve_coupled_newton(backend, state, photon_params) is state
    assert backend.kwargs["method"] == "coupled_newton"
    assert backend.kwargs["coupled_newton_step_rtol"] == pytest.approx(1e-6)

    assert _solve_picard(
        backend,
        state,
        photon_params,
        mixing=0.3,
    ) is state
    assert (
        backend.kwargs["newton_backward_error_tol"]
        == INNER_QP_BACKWARD_ERROR_LIMIT
    )
    assert (
        backend.kwargs["newton_number_polish_shape_tol"]
        == INNER_QP_NUMBER_POLISH_SHAPE_LIMIT
    )


def test_picard_predictor_retries_only_acceleration_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AA arithmetic failure retries plain Picard from the untouched seed."""
    import validation.fischer_2023.fig3_solve as fs

    spectral = SimpleNamespace(cell_weights=np.ones(2))
    seed = SimpleNamespace(f=np.array([1.0, 2.0]), spectral=spectral)
    fallback = SimpleNamespace(f=np.array([2.0, 3.0]), spectral=spectral)
    depths: list[int] = []

    def fake_solve(
        _backend,
        state,
        _photon_params,
        *,
        mixing,
        anderson_depth,
    ):
        del mixing
        assert state is seed
        depths.append(anderson_depth)
        if anderson_depth:
            raise AndersonAccelerationError("synthetic non-finite AA iterate")
        return fallback

    monkeypatch.setattr(fs, "_solve_picard", fake_solve)
    result = _solve_picard_predictor(
        object(),
        seed,
        {},
        ratio=6.0,
        mixing=0.15,
        fallback_mixing=0.05,
    )

    assert result is fallback
    assert depths == [fs.PICARD_ANDERSON_DEPTH, 0]


def test_picard_predictor_does_not_swallow_configuration_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid user/configuration inputs must not masquerade as AA failure."""
    import validation.fischer_2023.fig3_solve as fs

    calls = 0

    def reject_configuration(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        raise ValueError("invalid configured photon frequency")

    monkeypatch.setattr(fs, "_solve_picard", reject_configuration)
    with pytest.raises(ValueError, match="invalid configured"):
        _solve_picard_predictor(
            object(),
            object(),
            {},
            ratio=6.0,
            mixing=0.15,
            fallback_mixing=0.05,
        )
    assert calls == 1


def test_bad_ratio_zero_number_certificate_aborts_before_picard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An early amplitude failure must not consume the continuation ladder."""
    import validation.fischer_2023.fig3_solve as fs

    calls = {"ratio0": 0, "picard": 0}
    monkeypatch.setattr(fs, "_compute_tau_0_pb", lambda _spectral: 0.255)

    def fake_ratio0(_backend, state, _photon_params):
        calls["ratio0"] += 1
        return replace(state, f=state.f + 1e-10)

    def fake_predictor(
        _backend,
        state,
        _photon_params,
        *,
        ratio,
        mixing,
        fallback_mixing,
    ):
        del mixing, fallback_mixing
        calls["picard"] += 1
        return replace(state, f=state.f + (float(ratio) + 1.0) * 1e-10)

    def fake_certificate(*args, **kwargs):
        del args
        thermal = kwargs.get("tau_l") is None
        phonon_value = float("nan") if thermal else 0.0
        return {
            "qp_residual_inf": 0.0,
            "phonon_residual_inf": phonon_value,
            "phonon_raw_backward_error": phonon_value,
            "qp_backward_error": 0.0,
            "qp_number_backward_error": (
                2.0 * TARGET_BACKWARD_ERROR_LIMIT if thermal else 0.0
            ),
            "phonon_backward_error": phonon_value,
        }

    monkeypatch.setattr(fs, "_solve_tau_l_zero", fake_ratio0)
    monkeypatch.setattr(fs, "_solve_picard_predictor", fake_predictor)
    monkeypatch.setattr(fs, "steady_state_certificate", fake_certificate)

    with pytest.raises(
        RuntimeError,
        match=r"ratio 0: qp_number_backward_error=",
    ):
        solve_raw(
            num_bins=81,
            paper_ratios=(0.0, 1.0),
            continuation_ratios=(1.0,),
        )

    assert calls == {"ratio0": 1, "picard": 0}


def test_target_failure_checkpoints_then_resumes_without_recomputing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target mismatch must stop early and preserve continuation state."""
    import validation.fischer_2023.fig3_solve as fs

    checkpoint = tmp_path / "fig3-restart.npz"
    calls = {"ratio0": 0, "picard": 0}

    monkeypatch.setattr(fs, "_compute_tau_0_pb", lambda _spectral: 0.255)

    def fake_ratio0(_backend, state, _photon_params):
        calls["ratio0"] += 1
        return replace(state, f=state.f + 1e-10)

    def fake_predictor(
        _backend,
        state,
        _photon_params,
        *,
        ratio,
        mixing,
        fallback_mixing,
    ):
        del mixing, fallback_mixing
        calls["picard"] += 1
        return replace(state, f=state.f + (float(ratio) + 1.0) * 1e-10)

    def fake_certificate(*args, **kwargs):
        del args
        phonon_value = (
            float("nan") if kwargs.get("tau_l") is None else 0.0
        )
        return {
            "qp_residual_inf": 0.0,
            "phonon_residual_inf": phonon_value,
            "phonon_raw_backward_error": phonon_value,
            "qp_backward_error": 0.0,
            "qp_number_backward_error": 0.0,
            "phonon_backward_error": phonon_value,
        }

    monkeypatch.setattr(fs, "_solve_tau_l_zero", fake_ratio0)
    monkeypatch.setattr(fs, "_solve_picard_predictor", fake_predictor)
    monkeypatch.setattr(fs, "steady_state_certificate", fake_certificate)

    def reject_ratio_zero(event: Fig3StepEvent) -> None:
        assert event.ratio == 0.0
        assert not event.resumed
        raise AssertionError("synthetic ratio-zero baseline mismatch")

    with pytest.raises(AssertionError, match="ratio-zero baseline mismatch"):
        solve_raw(
            num_bins=81,
            paper_ratios=(0.0, 0.1, 1.0),
            continuation_ratios=(0.1, 1.0),
            checkpoint_path=checkpoint,
            checkpoint_identity="unit-test-identity",
            on_step=reject_ratio_zero,
        )

    assert checkpoint.exists()
    assert calls == {"ratio0": 1, "picard": 0}

    events: list[Fig3StepEvent] = []
    result = solve_raw(
        num_bins=81,
        paper_ratios=(0.0, 0.1, 1.0),
        continuation_ratios=(0.1, 1.0),
        checkpoint_path=checkpoint,
        checkpoint_identity="unit-test-identity",
        on_step=events.append,
    )

    assert calls == {"ratio0": 1, "picard": 2}
    assert [event.ratio for event in events] == [0.0, 0.1, 1.0]
    assert events[0].resumed
    assert not events[1].resumed
    assert np.all(np.diff([event.cumulative_seconds for event in events]) >= 0.0)
    assert result["f_ratios"].shape == (3, 81)
    assert checkpoint.exists()

    # A completed checkpoint is the durable handoff to an outer artifact
    # writer. Re-opening it performs no numerical work and returns the same
    # payload; only that persistence owner may remove it after promotion.
    calls_before_replay = calls.copy()
    replayed = solve_raw(
        num_bins=81,
        paper_ratios=(0.0, 0.1, 1.0),
        continuation_ratios=(0.1, 1.0),
        checkpoint_path=checkpoint,
        checkpoint_identity="unit-test-identity",
    )
    assert calls == calls_before_replay
    np.testing.assert_array_equal(replayed["f_ratios"], result["f_ratios"])


def test_non_target_callback_failure_is_replayed_before_continuation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable non-target step cannot silently skip a failed callback."""
    import validation.fischer_2023.fig3_solve as fs

    checkpoint = tmp_path / "fig3-restart.npz"
    calls = {"ratio0": 0, "picard": 0}

    monkeypatch.setattr(fs, "_compute_tau_0_pb", lambda _spectral: 0.255)

    def fake_ratio0(_backend, state, _photon_params):
        calls["ratio0"] += 1
        return replace(state, f=state.f + 1e-10)

    def fake_predictor(
        _backend,
        state,
        _photon_params,
        *,
        ratio,
        mixing,
        fallback_mixing,
    ):
        del mixing, fallback_mixing
        calls["picard"] += 1
        return replace(state, f=state.f + (float(ratio) + 1.0) * 1e-10)

    def fake_certificate(*args, **kwargs):
        del args
        phonon_value = (
            float("nan") if kwargs.get("tau_l") is None else 0.0
        )
        return {
            "qp_residual_inf": 0.0,
            "phonon_residual_inf": phonon_value,
            "phonon_raw_backward_error": phonon_value,
            "qp_backward_error": 0.0,
            "qp_number_backward_error": 0.0,
            "phonon_backward_error": phonon_value,
        }

    monkeypatch.setattr(fs, "_solve_tau_l_zero", fake_ratio0)
    monkeypatch.setattr(fs, "_solve_picard_predictor", fake_predictor)
    monkeypatch.setattr(fs, "steady_state_certificate", fake_certificate)

    def reject_non_target(event: Fig3StepEvent) -> None:
        if event.ratio == 0.1:
            assert not event.is_target
            assert not event.resumed
            raise AssertionError("synthetic non-target callback failure")

    with pytest.raises(AssertionError, match="non-target callback failure"):
        solve_raw(
            num_bins=81,
            paper_ratios=(0.0, 1.0),
            continuation_ratios=(0.1, 1.0),
            checkpoint_path=checkpoint,
            checkpoint_identity="unit-test-identity",
            on_step=reject_non_target,
        )

    assert checkpoint.exists()
    assert calls == {"ratio0": 1, "picard": 1}

    events: list[Fig3StepEvent] = []
    result = solve_raw(
        num_bins=81,
        paper_ratios=(0.0, 1.0),
        continuation_ratios=(0.1, 1.0),
        checkpoint_path=checkpoint,
        checkpoint_identity="unit-test-identity",
        on_step=events.append,
    )

    assert calls == {"ratio0": 1, "picard": 2}
    assert [event.ratio for event in events] == [0.0, 0.1, 1.0]
    assert [event.resumed for event in events] == [True, True, False]
    assert result["f_ratios"].shape == (2, 81)
    assert checkpoint.exists()


def test_restart_checkpoint_identity_mismatch_is_not_reused(
    tmp_path,
) -> None:
    import validation.fischer_2023.fig3_solve as fs

    path = tmp_path / "restart.npz"
    fs._write_restart_checkpoint(
        path,
        identity="old",
        completed_step_index=0,
        pending_callback_step_index=-1,
        f_seed=np.full(81, 0.1),
        n_ph_seed=None,
        target_complete=np.asarray([True]),
        target_f=np.full((1, 81), 0.1),
        certificate_by_ratio={
            0.0: {
                "qp_residual_inf": 0.0,
                "phonon_residual_inf": 0.0,
                "phonon_raw_backward_error": 0.0,
                "qp_backward_error": 0.0,
                "qp_number_backward_error": 0.0,
                "phonon_backward_error": 0.0,
            }
        },
        predictor_certificate_by_ratio={},
        polish_relative_f_by_ratio={},
        elapsed_by_step=np.asarray([1.0, np.nan]),
    )

    with pytest.warns(RuntimeWarning, match="identity mismatch"):
        loaded = fs._load_restart_checkpoint(
            path,
            identity="new",
            num_bins=81,
            num_phonon_bins=161,
            paper_ratios=(0.0,),
            step_ratios=(0.0, 0.1),
        )
    assert loaded is None


def test_restart_checkpoint_rejects_incomplete_certificate(
    tmp_path,
) -> None:
    import validation.fischer_2023.fig3_solve as fs

    path = tmp_path / "restart.npz"
    fs._write_restart_checkpoint(
        path,
        identity="current",
        completed_step_index=0,
        pending_callback_step_index=-1,
        f_seed=np.full(81, 0.1),
        n_ph_seed=None,
        target_complete=np.asarray([True]),
        target_f=np.full((1, 81), 0.1),
        certificate_by_ratio={
            0.0: {
                "qp_residual_inf": 0.0,
                "phonon_residual_inf": float("nan"),
                "phonon_raw_backward_error": float("nan"),
                "qp_backward_error": 0.0,
                # Deliberately omit qp_number_backward_error.
                "phonon_backward_error": float("nan"),
            }
        },
        predictor_certificate_by_ratio={},
        polish_relative_f_by_ratio={},
        elapsed_by_step=np.asarray([1.0, np.nan]),
    )

    with pytest.warns(RuntimeWarning, match="inconsistent"):
        loaded = fs._load_restart_checkpoint(
            path,
            identity="current",
            num_bins=81,
            num_phonon_bins=161,
            paper_ratios=(0.0,),
            step_ratios=(0.0, 0.1),
        )
    assert loaded is None


@pytest.mark.parametrize("missing", ["predictor", "polish"])
def test_completed_high_ratio_checkpoint_requires_polish_evidence(
    tmp_path,
    missing: str,
) -> None:
    import validation.fischer_2023.fig3_solve as fs

    path = tmp_path / f"restart-missing-{missing}.npz"
    thermal_certificate = {
        "qp_residual_inf": 0.0,
        "phonon_residual_inf": float("nan"),
        "phonon_raw_backward_error": float("nan"),
        "qp_backward_error": 0.0,
        "qp_number_backward_error": 0.0,
        "phonon_backward_error": float("nan"),
    }
    dynamic_certificate = dict.fromkeys(
        (
            "qp_residual_inf",
            "phonon_residual_inf",
            "phonon_raw_backward_error",
            "qp_backward_error",
            "qp_number_backward_error",
            "phonon_backward_error",
        ),
        0.0,
    )
    fs._write_restart_checkpoint(
        path,
        identity="current",
        completed_step_index=1,
        pending_callback_step_index=-1,
        f_seed=np.full(81, 0.1),
        n_ph_seed=np.zeros(161),
        target_complete=np.asarray([True, True]),
        target_f=np.full((2, 81), 0.1),
        certificate_by_ratio={
            0.0: thermal_certificate,
            10.0: dynamic_certificate,
        },
        predictor_certificate_by_ratio=(
            {} if missing == "predictor" else {10.0: dynamic_certificate}
        ),
        polish_relative_f_by_ratio=(
            {} if missing == "polish" else {10.0: 0.0}
        ),
        elapsed_by_step=np.asarray([1.0, 2.0]),
    )

    with pytest.warns(RuntimeWarning, match="inconsistent"):
        loaded = fs._load_restart_checkpoint(
            path,
            identity="current",
            num_bins=81,
            num_phonon_bins=161,
            paper_ratios=(0.0, 10.0),
            step_ratios=(0.0, 10.0),
        )
    assert loaded is None


@pytest.mark.slow
def test_reduced_anderson_matches_historical_plain_picard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AA and plain Picard reach the same certified low-ratio branch.

    This 81-bin A/B reaches ratio 1 in about 10 seconds total. Extending the
    historical 5%-mixed plain policy through ratio 10 costs roughly five
    minutes even at 81 bins, so the separate ladder-refinement test below
    covers strong-bottleneck continuation without imposing that cost on each
    test run.
    """
    import validation.fischer_2023.fig3_solve as fs

    kwargs = {
        "num_bins": 81,
        "paper_ratios": (0.0, 0.1, 1.0),
        "continuation_ratios": (0.1, 0.3, 0.5, 1.0),
    }
    accelerated = solve_raw(**kwargs)
    monkeypatch.setattr(fs, "PICARD_ANDERSON_DEPTH", 0)
    historical_plain = solve_raw(**kwargs)

    for raw in (accelerated, historical_plain):
        assert np.all(raw["qp_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT)
        finite_phonon = np.isfinite(raw["phonon_backward_error"])
        assert np.all(
            raw["phonon_backward_error"][finite_phonon]
            <= TARGET_BACKWARD_ERROR_LIMIT
        )
        peaks = np.max(raw["f_ratios"], axis=1)
        assert np.all(np.diff(peaks) > 0.0)

    for index in range(len(kwargs["paper_ratios"])):
        peak = float(np.max(historical_plain["f_ratios"][index]))
        np.testing.assert_allclose(
            accelerated["f_ratios"][index],
            historical_plain["f_ratios"][index],
            rtol=1e-5,
            atol=5e-6 * peak,
        )


@pytest.mark.slow
def test_reduced_ladder_refinement_preserves_nonzero_branch() -> None:
    """Halving the high-ratio continuation spacing stays on one branch."""
    refined_ladder = (
        0.1,
        0.3,
        0.5,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
        8.5,
        9.0,
        9.5,
        10.0,
    )
    unit = solve_raw(num_bins=81, continuation_ratios=CONTINUATION_RATIOS)
    refined = solve_raw(num_bins=81, continuation_ratios=refined_ladder)

    for raw in (unit, refined):
        assert np.all(raw["qp_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT)
        finite_phonon = np.isfinite(raw["phonon_backward_error"])
        assert np.all(
            raw["phonon_backward_error"][finite_phonon]
            <= TARGET_BACKWARD_ERROR_LIMIT
        )
        assert float(np.max(raw["f_ratios"][-1])) > 1e-8

    # Lower-ratio targets are reached before the ladders diverge and should be
    # identical. At ratio 10 one path Newton-polishes while the more accurate
    # refined predictor can already sit at the residual floor; both certified
    # states agree well inside the continuation-discretization allowance.
    np.testing.assert_array_equal(unit["f_ratios"][:-1], refined["f_ratios"][:-1])
    peak = float(np.max(unit["f_ratios"][-1]))
    np.testing.assert_allclose(
        refined["f_ratios"][-1],
        unit["f_ratios"][-1],
        rtol=5e-4,
        atol=5e-6 * peak,
    )


@pytest.mark.slow
@pytest.mark.manual_slow
def test_matches_pinned_baseline() -> None:
    """Recompute the exact 1620-bin pin; intentionally manual, not PR CI.

    The corrected producer took 10,671.777 s on the audited Windows host.
    Pull-request CI instead runs the reduced-grid branch/refinement solves and
    the fast strict baseline metadata, non-vacuity, digest, and certificate
    gates. Keeping this exact regeneration runnable but ``manual_slow`` avoids
    launching the same approximately three-hour producer on every supported
    Python interpreter for every commit.
    """
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig3_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the multi-hour run() below, instead of after it.
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
    metadata = read_baseline_metadata(path)
    _assert_baseline_curves_are_nonvacuous(baseline)
    def validate_completed_target(event: Fig3StepEvent) -> None:
        if not event.is_target:
            return
        ratio = event.ratio
        peak = float(np.max(np.abs(baseline.f_by_ratio[ratio])))
        rtol = curve_regression_rtol(ratio, pinned_on=metadata.pinned_on)
        np.testing.assert_allclose(
            event.f,
            baseline.f_by_ratio[ratio],
            rtol=rtol,
            atol=CURVE_REGRESSION_ATOL_OVER_PEAK * peak,
            err_msg=(
                f"Mismatch at τ_l/τ_0^PB = {ratio} "
                f"(pinned_on={metadata.pinned_on!r}, "
                f"running_on={sys.platform!r}, rtol={rtol:g}, "
                f"resumed={event.resumed})"
            ),
        )

    # Each target is checked immediately after completion, so a ratio-zero
    # drift stops before the continuation ladder. This pure regression path
    # deliberately has no implicit restart state; long manual runs opt in via
    # restart_checkpoint_path.
    result = run(on_step=validate_completed_target)

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)
    assert result.ratios == baseline.ratios

    np.testing.assert_allclose(
        result.f_FD, baseline.f_FD, rtol=0.0, atol=1e-14,
        err_msg="Fermi-Dirac reference drift",
    )

    for ratio in result.ratios:
        peak = float(np.max(np.abs(baseline.f_by_ratio[ratio])))
        rtol = curve_regression_rtol(ratio, pinned_on=metadata.pinned_on)
        np.testing.assert_allclose(
            result.f_by_ratio[ratio],
            baseline.f_by_ratio[ratio],
            rtol=rtol,
            atol=CURVE_REGRESSION_ATOL_OVER_PEAK * peak,
            err_msg=(
                f"Mismatch at τ_l/τ_0^PB = {ratio} "
                f"(pinned_on={metadata.pinned_on!r}, "
                f"running_on={sys.platform!r}, rtol={rtol:g})"
            ),
        )


class TestFig3CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves an unchanged continuation solve from disk. The expensive
    solve is stubbed so the test is fast; it exercises the real cache +
    observables wiring. Engine-level key/store properties are covered in
    ``tests/validation/test_sweep_cache.py``.
    """

    def _stub_payload(self) -> dict:
        # Synthetic raw payload (fig3's observables is a pure unpack — no grid
        # rebuild — so a tiny placeholder array suffices).
        ne = 8
        payload = {
            "E": np.linspace(180.0, 200.0, ne),
            "f_FD": np.full(ne, 1e-9),
            "f_ratios": np.full((3, ne), 1e-8),
            "ratios": np.array([0.0, 0.1, 1.0]),
            "tau_0_pb_ns": np.array([0.2515]),
        }
        payload.update(
            {field: np.zeros(3) for field in CERTIFICATE_FIELDS}
        )
        for field in CERTIFICATE_FIELDS:
            if field.startswith("phonon_"):
                payload[field][0] = np.nan
        return payload

    def _cfg(self) -> dict:
        return {
            "num_bins": 162,
            "paper_ratios": (0.0, 0.1, 1.0),
            "continuation_ratios": (0.1, 0.3, 0.5, 1.0),
        }

    def test_observables_owns_an_immutable_raw_snapshot(self) -> None:
        import validation.fischer_2023.fig3_paper as fp

        payload = self._stub_payload()
        result = fp.observables(
            payload,
            producer_solve_contract_digest="a" * 64,
            validated_solve_contract_digest="b" * 64,
        )
        original_energy = result.E.copy()
        original_curve = result.f_by_ratio[0.1].copy()
        original_maxima = dict(result.certificate_maxima)

        payload["E"][0] = -1.0
        payload["f_ratios"][1, 0] = 0.5
        payload[CERTIFICATE_FIELDS[0]][0] = 0.5
        np.testing.assert_array_equal(result.E, original_energy)
        np.testing.assert_array_equal(result.f_by_ratio[0.1], original_curve)
        assert dict(result.certificate_maxima) == original_maxima

        with pytest.raises(ValueError):
            result.E[0] = -1.0
        with pytest.raises(ValueError):
            result.f_by_ratio[0.1][0] = 0.5
        with pytest.raises(TypeError):
            result.f_by_ratio[0.1] = np.zeros_like(result.E)
        with pytest.raises(TypeError):
            result.certificate_maxima[CERTIFICATE_FIELDS[0]] = 0.0

    @pytest.mark.parametrize("field", ["f_FD", "f_ratios"])
    def test_observables_rejects_complex_raw_occupations(self, field: str) -> None:
        import validation.fischer_2023.fig3_paper as fp

        payload = self._stub_payload()
        values = np.asarray(payload[field], dtype=complex)
        values.reshape(-1)[0] += complex(
            0.0,
            np.nextafter(0.0, np.inf),
        )
        payload[field] = values

        with pytest.raises(ValueError, match="real-valued"):
            fp.observables(
                payload,
                producer_solve_contract_digest="a" * 64,
                validated_solve_contract_digest="b" * 64,
            )

    def test_run_cached_hits_disk_on_second_call(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig3_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "1")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        r1 = fp.run_cached(**self._cfg())
        assert calls["n"] == 1  # cache miss -> solve ran once

        r2 = fp.run_cached(**self._cfg())
        assert calls["n"] == 1  # cache hit -> solve NOT re-run

        ref = fp.observables(
            payload,
            producer_solve_contract_digest=r1.producer_solve_contract_digest,
            validated_solve_contract_digest=r1.validated_solve_contract_digest,
        )
        for res in (r1, r2):
            assert res.ratios == ref.ratios
            assert res.tau_0_pb_ns == pytest.approx(ref.tau_0_pb_ns)
            for r in ref.ratios:
                np.testing.assert_array_equal(res.f_by_ratio[r], ref.f_by_ratio[r])

    def test_run_cached_disabled_always_recomputes(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig3_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        seen_kwargs: list[dict[str, object]] = []
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            seen_kwargs.append(dict(kwargs))
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached(**self._cfg())
        fp.run_cached(**self._cfg())
        assert calls["n"] == 2  # disabled -> recompute each call
        assert all("checkpoint_path" not in kwargs for kwargs in seen_kwargs)
        assert all("checkpoint_identity" not in kwargs for kwargs in seen_kwargs)

    def test_disabled_cache_allows_explicit_restart_opt_in(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        import validation.fischer_2023.fig3_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        payload = self._stub_payload()
        seen_kwargs: list[dict[str, object]] = []

        def stub_solve(**kwargs):
            seen_kwargs.append(dict(kwargs))
            return {key: value.copy() for key, value in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)
        restart = tmp_path / "explicit-restart.npz"
        fp.run_cached(**self._cfg(), restart_checkpoint_path=restart)

        assert seen_kwargs[0]["checkpoint_path"] == restart
        assert isinstance(seen_kwargs[0]["checkpoint_identity"], str)
        assert seen_kwargs[0]["checkpoint_identity"]

    def test_uncached_run_restart_is_explicit_only(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        import validation.fischer_2023.fig3_paper as fp

        payload = self._stub_payload()
        seen_kwargs: list[dict[str, object]] = []

        def stub_solve(**kwargs):
            seen_kwargs.append(dict(kwargs))
            return {key: value.copy() for key, value in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)
        fp.run(**self._cfg())
        restart = tmp_path / "explicit-restart.npz"
        fp.run(**self._cfg(), restart_checkpoint_path=restart)

        assert "checkpoint_path" not in seen_kwargs[0]
        assert "checkpoint_identity" not in seen_kwargs[0]
        assert seen_kwargs[1]["checkpoint_path"] == restart
        assert isinstance(seen_kwargs[1]["checkpoint_identity"], str)
