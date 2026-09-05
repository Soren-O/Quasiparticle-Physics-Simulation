"""Regression test: Fischer 2023 Fig. 5 paper-topology run matches the pinned CSV.

Iterative-mode tolerance: 1e-6. The complete live
81-point regeneration is both ``slow`` and ``manual_slow``. The promoted
campaign measured 9.853 aggregate worker-hours and 2.606 wall hours with six
concurrent single-thread rows, so it is not a bounded CI check. Opt in
explicitly with
``pytest -m "slow and manual_slow"``. Smaller focused slow tests below
remain suitable for the hosted slow gate.

The expensive sweep ranges (``UPPER_NBAR_VALUES`` and
``LOWER_T_BATH_K``) are tunable in :mod:`fig5_paper`; tighten them if
this test starts to dominate the slow suite. The pinned baseline is
self-consistent against whichever ranges are configured at generation
time.

First-time generation::

    python -m validation.fischer_2023.fig5_paper
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import validation.fischer_2023.fig5_paper as fig5_paper
import validation.fischer_2023.fig5_solve as fig5_solve
from validation.fischer_2023 import steady_state_certificate as certificate_module
from validation.fischer_2023.fig5_paper import (
    ARTIFACT_SCHEMA,
    LOWER_NBAR,
    LOWER_T_BATH_K,
    UPPER_NBAR_VALUES,
    UPPER_T_BATH_K,
    ArtifactValidationError,
    Fig5PaperResult,
    LegacyArtifactError,
    baseline_path,
    config_metadata,
    read_baseline,
    run,
    write_baseline,
)


def test_live_gate_rejects_number_only_certificate_failure() -> None:
    certificate = {
        "qp_backward_error": 0.0,
        "qp_number_backward_error": 0.6,
        "phonon_backward_error": 0.0,
    }
    with pytest.raises(RuntimeError, match="qp_number"):
        fig5_solve._require_certified_point(certificate, context="test point")


def _lightweight_canonical_preflight(
    path: Path,
) -> tuple[fig5_paper.BaselineMetadata, SimpleNamespace]:
    """Authenticate the bundle and parse scalar claims without replaying states.

    The canonical public reader deliberately reassembles and independently
    certifies all 81 persisted states.  That is the right slow closeout gate,
    but composing ``read_baseline_metadata()`` and ``read_baseline()`` made the
    advertised fast configuration preflight perform that work twice (160.88 s
    measured on the 2026-07-28 audit host).

    The caller holds the publication lock.  This helper binds the exact
    CSV/two-PDF set to the promotion record, validates the current artifact
    fingerprint and table hash, then parses only scalar configuration, axes,
    and stored certificate columns.  Full state-derived recertification remains
    the separate ``slow`` test below.
    """
    assert path.resolve() == baseline_path().resolve()
    fig5_paper.read_promotion_record(_publication_lock_held=True)
    rows = fig5_paper._read_csv_rows(path)
    artifact_metadata = fig5_paper._artifact_metadata(path, rows)
    config = artifact_metadata["fingerprint"]["config"]
    metadata = fig5_paper.BaselineMetadata(
        delta_0=float(config["delta_0"]),
        tau_0=float(config["tau_0"]),
        t_c=float(config["t_c"]),
        omega_0=float(config["omega_0"]),
        c_phot=float(config["c_phot"]),
        num_bins=int(config["num_bins"]),
        e_min_factor=float(config["e_min_factor"]),
        e_max_factor=float(config["e_max_factor"]),
        tau_0_pb_ns=float(config["tau_0_pb_ns"]),
    )

    numeric_columns = fig5_paper._BASELINE_COLUMNS[
        1 : 6 + len(certificate_module.NUMBER_CERTIFICATE_FIELDS)
    ]
    data: list[tuple[str, dict[str, float]]] = []
    for row in rows[3:]:
        assert len(row) == len(fig5_paper._BASELINE_COLUMNS)
        values = {
            name: float(row[index])
            for index, name in enumerate(numeric_columns, start=1)
        }
        assert np.all(np.isfinite(tuple(values.values())))
        assert all(
            values[field] >= 0.0
            for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
        )
        for field in fig5_paper._CERTIFIED_BACKWARD_ERROR_FIELDS:
            assert values[field] <= fig5_solve.TARGET_BACKWARD_ERROR_LIMIT
        data.append((row[0], values))

    upper = [values for panel, values in data if panel == "upper"]
    lower = [values for panel, values in data if panel == "lower"]
    expected_upper = [
        (float(T_bath), float(n_bar))
        for T_bath in UPPER_T_BATH_K
        for n_bar in UPPER_NBAR_VALUES
    ]
    expected_lower = [
        (float(n_bar), float(T_bath))
        for n_bar in LOWER_NBAR
        for T_bath in LOWER_T_BATH_K
    ]
    assert [
        (values["T_bath_K"], values["n_bar"]) for values in upper
    ] == expected_upper
    assert [
        (values["n_bar"], values["T_bath_K"]) for values in lower
    ] == expected_lower

    return metadata, SimpleNamespace(
        upper_T_bath=np.asarray(
            list(dict.fromkeys(values["T_bath_K"] for values in upper)),
            dtype=float,
        ),
        upper_nbar=np.asarray(
            list(dict.fromkeys(values["n_bar"] for values in upper)),
            dtype=float,
        ),
        lower_nbar=np.asarray(
            list(dict.fromkeys(values["n_bar"] for values in lower)),
            dtype=float,
        ),
        lower_T_bath=np.asarray(
            list(dict.fromkeys(values["T_bath_K"] for values in lower)),
            dtype=float,
        ),
    )


def _assert_config_matches_baseline(path: Path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + both panels' sweep axes.

    Gating :func:`run` (the 9.853 h aggregate-worker two-panel Picard sweep)
    behind this turns a stale config/baseline pairing — a grid change, a
    sweep-range edit, a τ_0^PB drift — into a seconds-long failure instead of
    one discovered only after the full run. (See ``fig6_paper`` for the same
    pattern, where the promoted rows total 12.075 aggregate worker-hours.)
    """
    cfg = config_metadata()
    with fig5_paper._publication_lock():
        meta, axes = _lightweight_canonical_preflight(path)

    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_c == pytest.approx(meta.t_c, rel=1e-6)  # header stores 6 dp
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)

    np.testing.assert_allclose(
        np.asarray(UPPER_T_BATH_K, dtype=float), axes.upper_T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="upper-panel T_bath axis differs from baseline",
    )
    np.testing.assert_allclose(
        UPPER_NBAR_VALUES, axes.upper_nbar, rtol=1e-12, atol=0.0,
        err_msg="upper-panel n_bar axis (range/count) differs from baseline",
    )
    np.testing.assert_allclose(
        np.asarray(LOWER_NBAR, dtype=float), axes.lower_nbar,
        rtol=1e-12, atol=0.0,
        err_msg="lower-panel n_bar axis differs from baseline",
    )
    np.testing.assert_allclose(
        LOWER_T_BATH_K, axes.lower_T_bath, rtol=0.0, atol=1e-14,
        err_msg="lower-panel T_bath axis (range/count) differs from baseline",
    )


def test_canonical_artifact_matches_current_contract() -> None:
    """Fast preflight for the promoted artifact; legacy stays quarantined."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    try:
        _assert_config_matches_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(str(exc))


@pytest.mark.slow
def test_canonical_bundle_authenticates_and_recertifies() -> None:
    """One explicit current-artifact gate replays all 81 persisted states."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    try:
        read_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(str(exc))


def test_reduced_transition_is_seed_independent_when_tightly_certified() -> None:
    """The former forward/reverse split was a loose-Newton stopping artifact."""
    target = float(UPPER_NBAR_VALUES[6])  # T_*/Delta = 0.60
    paths = (
        tuple(float(UPPER_NBAR_VALUES[index]) for index in (0, 5, 6)),
        tuple(float(UPPER_NBAR_VALUES[index]) for index in (13, 9, 7, 6)),
        (target,),
    )

    results = [
        run(
            num_bins=162,
            upper_T_bath=(0.10,),
            upper_nbar=path,
            lower_nbar=(),
            lower_T_bath=(),
        )
        for path in paths
    ]
    transition_x_qp = np.asarray(
        [result.upper_x_qp_num[0, -1] for result in results],
        dtype=float,
    )

    assert np.ptp(transition_x_qp) / np.max(transition_x_qp) <= 1.0e-3
    for result in results:
        assert result.upper_qp_backward_error[0, -1] <= 1.0e-9
        assert result.upper_phonon_backward_error[0, -1] <= 1.0e-9


@pytest.mark.slow
def test_high_drive_does_not_false_converge_to_thermal_branch() -> None:
    """A tiny above-gap phonon population must still drive the QP branch.

    Uses just the two endpoints of the 0.10 K upper-panel sweep. The former
    peak-scaled Picard denominator floor declared the high-drive point converged
    at the thermal solution because unrelated low-energy phonon bins set a huge
    global scale; the paper grid is required to expose that failure mode.
    """
    result = run(
        upper_T_bath=(0.10,),
        upper_nbar=(float(UPPER_NBAR_VALUES[0]), float(UPPER_NBAR_VALUES[-1])),
        lower_nbar=(),
        lower_T_bath=(),
    )
    low_drive, high_drive = result.upper_x_qp_num[0]

    assert low_drive < 1e-8
    assert high_drive > 1e-3
    assert high_drive > 1e6 * low_drive


@pytest.mark.slow
@pytest.mark.manual_slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig5_paper"
        )

    try:
        baseline = read_baseline(path)
    except LegacyArtifactError as exc:
        pytest.xfail(str(exc))
    result = run()

    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)

    # Upper panel — sweep axes match exactly, then x_qp + T_*/Δ to 1e-6.
    np.testing.assert_allclose(
        result.upper_T_bath, baseline.upper_T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.upper_nbar, baseline.upper_nbar, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        result.upper_T_star, baseline.upper_T_star, rtol=1e-10, atol=0.0,
        err_msg="Upper-panel T_* drift",
    )
    # Scale-aware: the former atol=1e-6 exceeded the complete low-drive
    # signal (down to ~2e-10), so an amplitude-collapsed result passed.
    # Current artifacts also carry and independently recertify raw states.
    np.testing.assert_allclose(
        result.upper_x_qp_num,
        baseline.upper_x_qp_num,
        rtol=5e-3,
        atol=1e-30,
        err_msg="Upper-panel numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.upper_x_qp_analytic, baseline.upper_x_qp_analytic,
        rtol=1e-10, atol=0.0,
        err_msg="Upper-panel analytic x_qp drift",
    )

    # Lower panel.
    np.testing.assert_allclose(
        result.lower_T_bath, baseline.lower_T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.lower_nbar, baseline.lower_nbar, rtol=1e-12, atol=0.0,
    )
    np.testing.assert_allclose(
        result.lower_x_qp_num,
        baseline.lower_x_qp_num,
        rtol=5e-3,
        atol=1e-30,
        err_msg="Lower-panel numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.lower_x_qp_analytic, baseline.lower_x_qp_analytic,
        rtol=1e-10, atol=0.0,
        err_msg="Lower-panel analytic x_qp drift",
    )
    for panel in ("upper", "lower"):
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
            np.testing.assert_allclose(
                getattr(result, f"{panel}_{field}"),
                getattr(baseline, f"{panel}_{field}"),
                rtol=1e-6,
                atol=1e-30,
                err_msg=f"{panel} {field} certificate drift",
            )


_FAKE_CERTIFICATE_FACTORS = {
    "qp_residual_inf": 1.0,
    "qp_backward_error": 0.5,
    "qp_number_backward_error": 0.6,
    "phonon_residual_inf": 2.0,
    "phonon_raw_backward_error": 1.5,
    "phonon_backward_error": 0.75,
}


def _fake_certificate_seed(f0: float, n_ph0: float) -> float:
    return 1.0e-12 + 1.0e-4 * f0 + 1.0e-6 * n_ph0


@pytest.fixture
def schema_result(monkeypatch) -> Fig5PaperResult:
    """Cheap state-bound artifact fixture; production certificates stay real."""
    import validation.fischer_2023.fig5_paper as fp

    def fake_certificate(state, *, photon_params, tau_l):
        del photon_params, tau_l
        seed = _fake_certificate_seed(
            float(state.f[0]),
            float(state.phonon.n_ph[0, 0, 0]),
        )
        return {
            field: factor * seed
            for field, factor in _FAKE_CERTIFICATE_FACTORS.items()
        }

    monkeypatch.setattr(
        fp.certificate_module,
        "steady_state_certificate",
        fake_certificate,
    )
    upper_T = np.asarray(UPPER_T_BATH_K, dtype=float)
    upper_nbar = np.asarray(UPPER_NBAR_VALUES, dtype=float)
    lower_nbar = np.asarray(LOWER_NBAR, dtype=float)
    lower_T = np.asarray(LOWER_T_BATH_K, dtype=float)
    _, _, spectral = fp._build_grid_and_spectral(fp.NUM_BINS)
    omega, _, _, _ = fp.fig5_solve.build_phonon_frequency_map(spectral.E)
    upper_shape = (upper_T.size, upper_nbar.size)
    lower_shape = (lower_nbar.size, lower_T.size)
    upper_levels = 1.0e-8 * (
        np.arange(np.prod(upper_shape), dtype=float).reshape(upper_shape) + 1.0
    )
    lower_levels = 2.0e-8 * (
        np.arange(np.prod(lower_shape), dtype=float).reshape(lower_shape) + 1.0
    )
    upper_f = np.broadcast_to(
        upper_levels[..., None], (*upper_shape, fp.NUM_BINS)
    ).copy()
    lower_f = np.broadcast_to(
        lower_levels[..., None], (*lower_shape, fp.NUM_BINS)
    ).copy()
    upper_n_ph = np.broadcast_to(
        (0.01 * upper_levels)[..., None], (*upper_shape, omega.size)
    ).copy()
    lower_n_ph = np.broadcast_to(
        (0.01 * lower_levels)[..., None], (*lower_shape, omega.size)
    ).copy()
    tau_0_pb = fp.config_metadata().tau_0_pb_ns
    raw: dict[str, np.ndarray] = {
        "upper_f": upper_f,
        "lower_f": lower_f,
        "upper_n_ph": upper_n_ph,
        "lower_n_ph": lower_n_ph,
        "upper_T_bath": upper_T,
        "upper_nbar": upper_nbar,
        "lower_nbar": lower_nbar,
        "lower_T_bath": lower_T,
        "tau_0_pb_ns": np.asarray([tau_0_pb]),
        "tau_l_ns": np.asarray([tau_0_pb]),
        "num_bins": np.asarray([fp.NUM_BINS]),
    }
    for prefix, f_values, n_ph_values in (
        ("upper", upper_f, upper_n_ph),
        ("lower", lower_f, lower_n_ph),
    ):
        seed = (
            1.0e-12
            + 1.0e-4 * f_values[..., 0]
            + 1.0e-6 * n_ph_values[..., 0]
        )
        for field, factor in _FAKE_CERTIFICATE_FACTORS.items():
            raw[f"{prefix}_{field}"] = factor * seed
    return fp.observables(raw)


def _read_rows(path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.reader(stream))


def _write_rows(path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream, lineterminator="\n").writerows(rows)


def _refresh_payload_hash(rows: list[list[str]]) -> None:
    import validation.fischer_2023.fig5_paper as fp

    prefix = "# qpsim_metadata="
    metadata = json.loads(rows[1][0][len(prefix):])
    metadata["payload_sha256"] = fp._payload_sha256(rows[3:])
    rows[1][0] = prefix + fp._canonical_json(metadata)


def test_current_schema_round_trip_is_state_bound(tmp_path, schema_result) -> None:
    path = write_baseline(schema_result, tmp_path / "fig5.csv")
    rows = _read_rows(path)
    assert rows[0] == [f"# qpsim_artifact_schema={ARTIFACT_SCHEMA}"]
    restored = read_baseline(path)
    for field in schema_result.__dataclass_fields__:
        expected = getattr(schema_result, field)
        actual = getattr(restored, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual == expected


def test_reader_returns_fresh_certificate_across_runtime_diagnostic_drift(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    """Producer stamps are claims, not cross-runtime floating-point pins."""
    import validation.fischer_2023.fig5_paper as fp

    path = write_baseline(schema_result, tmp_path / "cross-runtime.csv")
    producer_certificate = fp.certificate_module.steady_state_certificate
    fresh_values = {
        "qp_residual_inf": 1.25e-1,
        "qp_backward_error": 0.75 * fp.TARGET_BACKWARD_ERROR_LIMIT,
        "qp_number_backward_error": 0.80 * fp.TARGET_BACKWARD_ERROR_LIMIT,
        "phonon_residual_inf": 2.50e-1,
        "phonon_raw_backward_error": 5.00e-1,
        "phonon_backward_error": (
            0.75 * fp.PHONON_REASSEMBLY_BACKWARD_ERROR_LIMIT
        ),
    }

    def cross_runtime_certificate(*args, **kwargs):
        certificate = producer_certificate(*args, **kwargs)
        certificate.update(fresh_values)
        return certificate

    monkeypatch.setattr(
        fp.certificate_module,
        "steady_state_certificate",
        cross_runtime_certificate,
    )
    restored = read_baseline(path)

    # The exact file retains the producer's authenticated measurements.
    rows = _read_rows(path)
    for offset, field in enumerate(certificate_module.NUMBER_CERTIFICATE_FIELDS):
        assert float(rows[3][6 + offset]) == getattr(
            schema_result,
            f"upper_{field}",
        )[0, 0]

    # The public reader exposes what this runtime independently reassembled.
    for panel in ("upper", "lower"):
        for field, value in fresh_values.items():
            np.testing.assert_array_equal(
                getattr(restored, f"{panel}_{field}"),
                np.full_like(getattr(restored, f"{panel}_{field}"), value),
            )


def test_direct_writers_cannot_clobber_canonical_bundle(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    csv_path = tmp_path / "canonical.csv"
    pdf_a_path = tmp_path / "canonical-a.pdf"
    pdf_b_path = tmp_path / "canonical-b.pdf"
    record_path = csv_path.with_suffix(".promotion.json")
    sentinels = {
        csv_path: b"csv-sentinel",
        pdf_a_path: b"pdf-a-sentinel",
        pdf_b_path: b"pdf-b-sentinel",
        record_path: b"record-sentinel",
    }
    for path, payload in sentinels.items():
        path.write_bytes(payload)
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)

    for writer in (
        lambda: fp.write_baseline(schema_result),
        lambda: fp.write_baseline(schema_result, csv_path),
        lambda: fp.write_baseline(schema_result, pdf_a_path),
        lambda: fp.write_plot_a(schema_result),
        lambda: fp.write_plot_a(schema_result, pdf_a_path),
        lambda: fp.write_plot_a(schema_result, csv_path),
        lambda: fp.write_plot_b(schema_result),
        lambda: fp.write_plot_b(schema_result, pdf_b_path),
        lambda: fp.write_plot_b(schema_result, record_path),
        lambda: fp.write_plot(schema_result),
    ):
        with pytest.raises(
            ArtifactValidationError,
            match=r"canonical|explicit",
        ):
            writer()
    for path, payload in sentinels.items():
        assert path.read_bytes() == payload


def test_write_plot_honors_explicit_panel_a_path(tmp_path, schema_result) -> None:
    import validation.fischer_2023.fig5_paper as fp

    panel_a = tmp_path / "review.pdf"
    panel_b = tmp_path / "review_b.pdf"
    assert fp.write_plot(schema_result, panel_a) == panel_a
    fp._require_plausible_pdf(panel_a)
    fp._require_plausible_pdf(panel_b)


def test_canonical_publication_record_rejects_mixed_pdf(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    csv_path = tmp_path / "fig5.csv"
    pdf_a_path = tmp_path / "fig5_a.pdf"
    pdf_b_path = tmp_path / "fig5_b.pdf"
    record_path = tmp_path / "fig5.promotion.json"
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)

    fp.publish_artifacts(
        schema_result,
        producer=fp.capture_producer_identity(),
    )
    fp.read_promotion_record()
    current_fingerprint = fp.artifact_fingerprint()
    future_fingerprint = json.loads(json.dumps(current_fingerprint))
    future_fingerprint["source_sha256"][
        "validation/fischer_2023/fig5_paper.py"
    ] = "f" * 64
    monkeypatch.setattr(
        fp,
        "artifact_fingerprint",
        lambda: future_fingerprint,
    )
    with pytest.raises(
        fp.ArtifactValidationError,
        match="current publication fingerprint is stale",
    ):
        fp.read_promotion_record()
    monkeypatch.setattr(
        fp,
        "artifact_fingerprint",
        lambda: current_fingerprint,
    )
    pdf_b_path.write_bytes(pdf_b_path.read_bytes() + b"\n% mixed producer\n")
    with pytest.raises(fp.ArtifactValidationError, match="does not match"):
        fp.read_promotion_record()


def test_publication_record_rejects_token_shaped_non_pdf(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    csv_path = tmp_path / "fig5.csv"
    pdf_a_path = tmp_path / "fig5_a.pdf"
    pdf_b_path = tmp_path / "fig5_b.pdf"
    record_path = tmp_path / "fig5.promotion.json"
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)

    fp.publish_artifacts(
        schema_result,
        producer=fp.capture_producer_identity(),
    )
    pdf_a_path.write_bytes(
        b"%PDF-1.4\n"
        + b"not a PDF object graph\n"
        + b"x" * 2048
        + b"\n%%EOF\n"
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["artifacts"]["pdf_a"] = fp._file_identity(pdf_a_path)
    record_path.write_text(
        json.dumps(record, allow_nan=False, sort_keys=True),
        encoding="utf-8",
    )
    with pytest.raises(
        fp.ArtifactValidationError,
        match="one-page Matplotlib PDF",
    ):
        fp.read_promotion_record()


def test_portable_publication_record_uses_lock_and_commit_point_recheck(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    csv_path = source_dir / "fig5.csv"
    pdf_a_path = source_dir / "fig5_a.pdf"
    pdf_b_path = source_dir / "fig5_b.pdf"
    record_path = source_dir / "fig5.promotion.json"
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)
    fp.publish_artifacts(
        schema_result,
        producer=fp.capture_producer_identity(),
    )

    copied_dir = tmp_path / "portable"
    copied_dir.mkdir()
    copied_csv = copied_dir / csv_path.name
    copied_pdf_a = copied_dir / pdf_a_path.name
    copied_pdf_b = copied_dir / pdf_b_path.name
    copied_record = copied_dir / record_path.name
    for source, destination in (
        (csv_path, copied_csv),
        (pdf_a_path, copied_pdf_a),
        (pdf_b_path, copied_pdf_b),
        (record_path, copied_record),
    ):
        destination.write_bytes(source.read_bytes())

    with (
        fp._publication_lock(copied_record),
        pytest.raises(RuntimeError, match="publisher or reader"),
    ):
        fp.read_promotion_record(
            copied_record,
            csv_path=copied_csv,
            pdf_a_path=copied_pdf_a,
            pdf_b_path=copied_pdf_b,
        )

    real_pdf_validator = fp._require_plausible_pdf
    validated_pdfs = 0

    def mutate_csv_at_commit_point(path: Path) -> None:
        nonlocal validated_pdfs
        real_pdf_validator(path)
        validated_pdfs += 1
        if validated_pdfs == 2:
            copied_csv.write_bytes(b"changed after initial identity check")

    monkeypatch.setattr(
        fp,
        "_require_plausible_pdf",
        mutate_csv_at_commit_point,
    )
    with pytest.raises(
        fp.ArtifactValidationError,
        match="csv changed during publication record validation",
    ):
        fp.read_promotion_record(
            copied_record,
            csv_path=copied_csv,
            pdf_a_path=copied_pdf_a,
            pdf_b_path=copied_pdf_b,
        )


def test_provenance_rebind_preserves_history_rows_and_pdfs(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    """A reader-only migration must never impersonate a new solve."""
    import validation.fischer_2023.fig5_paper as fp

    current_fingerprint = fp.artifact_fingerprint()
    historical_fingerprint = json.loads(
        json.dumps(current_fingerprint)
    )
    historical_fingerprint["source_sha256"][
        "validation/fischer_2023/fig5_paper.py"
    ] = "0" * 64
    selected = {"fingerprint": historical_fingerprint}
    monkeypatch.setattr(
        fp,
        "artifact_fingerprint",
        lambda: json.loads(json.dumps(selected["fingerprint"])),
    )

    csv_path = tmp_path / "fig5.csv"
    pdf_a_path = tmp_path / "fig5_a.pdf"
    pdf_b_path = tmp_path / "fig5_b.pdf"
    record_path = tmp_path / "fig5.promotion.json"
    campaign_path = tmp_path / "fig5.campaign.json"
    fp.write_baseline(schema_result, csv_path)
    fp.write_plot_a(schema_result, pdf_a_path)
    fp.write_plot_b(schema_result, pdf_b_path)

    historical_producer = {
        "fingerprint": historical_fingerprint,
        "runtime": {"test-runtime": "historical"},
    }
    legacy_record = {
        "artifact_schema": fp.ARTIFACT_SCHEMA,
        "artifacts": {
            "csv": fp._file_identity(csv_path),
            "pdf_a": fp._file_identity(pdf_a_path),
            "pdf_b": fp._file_identity(pdf_b_path),
        },
        "producer": historical_producer,
        "schema": fp._LEGACY_PROMOTION_RECORD_SCHEMA,
    }
    record_path.write_text(
        json.dumps(legacy_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    runner = {"path": "scripts/test.py", "sha256": "1" * 64}
    environment = {"OPENBLAS_NUM_THREADS": "1"}
    campaign_rows = [
        *[
            {
                "axis_value": float(axis_value),
                "label": f"upper-t{row_index:02d}",
                "ordinal": row_index,
                "panel": "upper",
                "row_index": row_index,
                "sha256": f"{row_index + 2:064x}",
                "size_bytes": row_index + 1,
            }
            for row_index, axis_value in enumerate(fp.UPPER_T_BATH_K)
        ],
        *[
            {
                "axis_value": float(axis_value),
                "label": f"lower-n{row_index:02d}",
                "ordinal": len(fp.UPPER_T_BATH_K) + row_index,
                "panel": "lower",
                "row_index": row_index,
                "sha256": (
                    f"{len(fp.UPPER_T_BATH_K) + row_index + 2:064x}"
                ),
                "size_bytes": len(fp.UPPER_T_BATH_K) + row_index + 1,
            }
            for row_index, axis_value in enumerate(fp.LOWER_NBAR)
        ],
    ]
    campaign_identity = {
        "mode": fp._HISTORICAL_CAMPAIGN_MODE,
        "row_schema": fp._HISTORICAL_CAMPAIGN_ROW_SCHEMA,
        "rows": campaign_rows,
        "runner": runner,
        "single_thread_environment": environment,
        "status_schema": fp._HISTORICAL_CAMPAIGN_STATUS_SCHEMA,
    }
    campaign_producer = fp._campaign_producer(
        historical_producer,
        campaign_identity,
    )
    producer_sha256 = fp._campaign_producer_sha256(campaign_producer)
    run_identity = fp._campaign_run_identity(
        campaign_producer,
        row_schema=campaign_identity["row_schema"],
        status_schema=campaign_identity["status_schema"],
    )
    campaign_identity["run_identity"] = run_identity
    row_hashes = {
        row["label"]: row["sha256"]
        for row in campaign_identity["rows"]
    }
    campaign_status = {
        "attempt": 1,
        "completed_rows": 6,
        "completion": {
            "aggregate_worker_s": 1.0,
            "artifacts": {
                "csv": {
                    "path": str(csv_path),
                    "sha256": legacy_record["artifacts"]["csv"]["sha256"],
                },
                "pdf_a": {
                    "path": str(pdf_a_path),
                    "sha256": legacy_record["artifacts"]["pdf_a"]["sha256"],
                },
                "pdf_b": {
                    "path": str(pdf_b_path),
                    "sha256": legacy_record["artifacts"]["pdf_b"]["sha256"],
                },
                "record": {
                    "path": str(record_path),
                    "sha256": fp._sha256_file(record_path),
                },
            },
            "campaign": campaign_identity,
            "certificate_maxima": fp._artifact_metadata(
                csv_path,
                fp._read_csv_rows(csv_path),
            )["certificate_maxima"],
            "new_rows": 6,
            "resumed_rows": 0,
            "row_payload_sha256": row_hashes,
            "wall_s": 1.0,
        },
        "producer_sha256": producer_sha256,
        "run_identity": run_identity,
        "schema": campaign_identity["status_schema"],
        "started_utc": "2026-01-01T00:00:00+00:00",
        "state": "complete",
        "total_rows": 6,
        "updated_utc": "2026-01-01T00:00:01+00:00",
    }
    campaign_path.write_text(
        json.dumps(campaign_status, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)
    monkeypatch.setattr(fp, "campaign_record_path", lambda: campaign_path)
    verified_manifests: list[dict[str, object]] = []

    def record_historical_verification(
        _commit: str,
        manifest: Mapping[str, object],
    ) -> None:
        verified_manifests.append(dict(manifest))

    monkeypatch.setattr(
        fp,
        "_verify_historical_source_commit",
        record_historical_verification,
    )

    original_rows = csv_path.read_bytes().splitlines(keepends=True)
    original_pdf_a = pdf_a_path.read_bytes()
    original_pdf_b = pdf_b_path.read_bytes()
    original_campaign = campaign_path.read_bytes()
    expected = {
        "campaign": fp._sha256_file(campaign_path),
        "csv": fp._sha256_file(csv_path),
        "pdf_a": fp._sha256_file(pdf_a_path),
        "pdf_b": fp._sha256_file(pdf_b_path),
        "promotion": fp._sha256_file(record_path),
    }
    selected["fingerprint"] = current_fingerprint
    fp.rebind_authenticated_artifacts(
        schema_result,
        historical_source_commit="a" * 40,
        expected_previous_sha256=expected,
    )
    assert verified_manifests[-1] == {
        runner["path"]: runner["sha256"],
    }

    rebound_rows = csv_path.read_bytes().splitlines(keepends=True)
    assert rebound_rows[0] == original_rows[0]
    assert rebound_rows[2:] == original_rows[2:]
    assert pdf_a_path.read_bytes() == original_pdf_a
    assert pdf_b_path.read_bytes() == original_pdf_b
    assert campaign_path.read_bytes() == original_campaign

    record = fp.read_promotion_record()
    assert record["producer"] == historical_producer
    assert record["publication"]["kind"] == fp._PROVENANCE_REBIND
    assert record["publication"]["fingerprint"] == current_fingerprint
    origin = record["publication"]["rebound_from"]
    assert origin["artifacts"] == legacy_record["artifacts"]
    assert origin["run_identity"] == run_identity
    assert origin["producer_sha256"] == producer_sha256
    assert origin["source_commit"] == "a" * 40

    copied = tmp_path / "portable-copy"
    copied.mkdir()
    copied_csv = copied / "fischer_fig5_paper.csv"
    copied_pdf_a = copied / "fischer_fig5_paper_a.pdf"
    copied_pdf_b = copied / "fischer_fig5_paper_b.pdf"
    copied_record = copied / "fischer_fig5_paper.promotion.json"
    copied_campaign = copied / "fischer_fig5_paper.campaign.json"
    for source, destination in (
        (csv_path, copied_csv),
        (pdf_a_path, copied_pdf_a),
        (pdf_b_path, copied_pdf_b),
        (record_path, copied_record),
        (campaign_path, copied_campaign),
    ):
        destination.write_bytes(source.read_bytes())
    copied_attestation = fp.read_promotion_record(
        copied_record,
        csv_path=copied_csv,
        pdf_a_path=copied_pdf_a,
        pdf_b_path=copied_pdf_b,
        campaign_path=copied_campaign,
    )
    assert copied_attestation == record

    future_fingerprint = json.loads(json.dumps(current_fingerprint))
    future_fingerprint["source_sha256"][
        "validation/fischer_2023/fig5_paper.py"
    ] = "f" * 64
    selected["fingerprint"] = future_fingerprint
    with pytest.raises(
        fp.ArtifactValidationError,
        match="current publication fingerprint is stale",
    ):
        fp.read_promotion_record()
    selected["fingerprint"] = current_fingerprint

    current_hashes = {
        "campaign": fp._sha256_file(campaign_path),
        "csv": fp._sha256_file(csv_path),
        "pdf_a": fp._sha256_file(pdf_a_path),
        "pdf_b": fp._sha256_file(pdf_b_path),
        "promotion": fp._sha256_file(record_path),
    }
    with pytest.raises(
        fp.ArtifactValidationError,
        match="supported v2 source bundle",
    ):
        fp.rebind_authenticated_artifacts(
            schema_result,
            historical_source_commit="a" * 40,
            expected_previous_sha256=current_hashes,
        )


def test_historical_source_commit_verification_is_content_bound() -> None:
    import validation.fischer_2023.fig5_paper as fp

    root = Path(fp.__file__).resolve().parents[2]
    commit = fp.subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    relative = "qpsim/constants.py"
    payload = fp.subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{relative}"],
        check=True,
        capture_output=True,
    ).stdout
    expected = fp.hashlib.sha256(
        payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()

    fp._verify_historical_source_commit(commit, {relative: expected})
    with pytest.raises(
        fp.ArtifactValidationError,
        match="does not match the producer manifest",
    ):
        fp._verify_historical_source_commit(
            commit,
            {relative: "0" * 64},
        )
    with pytest.raises(
        fp.ArtifactValidationError,
        match="cannot be resolved",
    ):
        fp._verify_historical_source_commit("0" * 40, {relative: expected})


def test_historical_campaign_runner_is_bound_to_source_commit() -> None:
    import validation.fischer_2023.fig5_paper as fp

    root = Path(fp.__file__).resolve().parents[2]
    commit = fp.subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    runner_path = "scripts/regenerate_fischer_fig5_parallel.py"
    payload = fp.subprocess.run(
        ["git", "-C", str(root), "show", f"{commit}:{runner_path}"],
        check=True,
        capture_output=True,
    ).stdout
    runner_sha256 = fp.hashlib.sha256(
        payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()
    status = {
        "completion": {
            "campaign": {
                "runner": {
                    "path": runner_path,
                    "sha256": runner_sha256,
                }
            }
        }
    }

    fp._verify_historical_campaign_runner(commit, status)
    status["completion"]["campaign"]["runner"]["sha256"] = "0" * 64
    with pytest.raises(
        fp.ArtifactValidationError,
        match="does not match the producer manifest",
    ):
        fp._verify_historical_campaign_runner(commit, status)


def _prepare_minimal_rebind_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Build a fast orchestration fixture; numerical replay is tested above."""
    import validation.fischer_2023.fig5_paper as fp

    current = fp.artifact_fingerprint()
    historical = json.loads(json.dumps(current))
    historical["source_sha256"][
        "validation/fischer_2023/fig5_paper.py"
    ] = "0" * 64
    selected = {"fingerprint": historical}
    monkeypatch.setattr(
        fp,
        "artifact_fingerprint",
        lambda: json.loads(json.dumps(selected["fingerprint"])),
    )

    csv_path = tmp_path / "fig5.csv"
    pdf_a_path = tmp_path / "fig5_a.pdf"
    pdf_b_path = tmp_path / "fig5_b.pdf"
    record_path = tmp_path / "fig5.promotion.json"
    campaign_path = tmp_path / "fig5.campaign.json"
    data_rows = [["upper", "unchanged-numerical-row"]]
    old_metadata = {
        "fingerprint": historical,
        "payload_sha256": fp._payload_sha256(data_rows),
    }
    old_rows = [
        [f"# qpsim_artifact_schema={fp.ARTIFACT_SCHEMA}"],
        [f"# qpsim_metadata={fp._canonical_json(old_metadata)}"],
        ["panel", "payload"],
        *data_rows,
    ]
    _write_rows(csv_path, old_rows)
    pdf_a_path.write_bytes(b"historical panel a")
    pdf_b_path.write_bytes(b"historical panel b")
    producer = {
        "fingerprint": historical,
        "runtime": {"python": "historical"},
    }
    legacy_record = {
        "artifact_schema": fp.ARTIFACT_SCHEMA,
        "artifacts": {
            "csv": fp._file_identity(csv_path),
            "pdf_a": fp._file_identity(pdf_a_path),
            "pdf_b": fp._file_identity(pdf_b_path),
        },
        "producer": producer,
        "schema": fp._LEGACY_PROMOTION_RECORD_SCHEMA,
    }
    record_path.write_text(
        json.dumps(legacy_record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    campaign_path.write_text(
        json.dumps(
            {
                "producer_sha256": "1" * 64,
                "run_identity": "2" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)
    monkeypatch.setattr(fp, "campaign_record_path", lambda: campaign_path)
    monkeypatch.setattr(
        fp,
        "_verify_historical_source_commit",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        fp,
        "_verify_historical_campaign_runner",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(fp, "_read_artifact", lambda *_args, **_kwargs: None)

    selected["fingerprint"] = current
    current_metadata = {
        "fingerprint": current,
        "payload_sha256": fp._payload_sha256(data_rows),
    }
    current_rows = [
        old_rows[0],
        [f"# qpsim_metadata={fp._canonical_json(current_metadata)}"],
        *old_rows[2:],
    ]
    candidate_path = tmp_path / "candidate.csv"
    _write_rows(candidate_path, current_rows)
    candidate_bytes = candidate_path.read_bytes()

    def write_candidate(_result, path):
        path.write_bytes(candidate_bytes)
        return path

    monkeypatch.setattr(fp, "_write_rebound_baseline", write_candidate)
    expected = {
        "campaign": fp._sha256_file(campaign_path),
        "csv": fp._sha256_file(csv_path),
        "pdf_a": fp._sha256_file(pdf_a_path),
        "pdf_b": fp._sha256_file(pdf_b_path),
        "promotion": fp._sha256_file(record_path),
    }
    return SimpleNamespace(
        campaign_path=campaign_path,
        candidate_bytes=candidate_bytes,
        csv_path=csv_path,
        current=current,
        expected=expected,
        legacy_record=legacy_record,
        old_campaign=campaign_path.read_bytes(),
        old_csv=csv_path.read_bytes(),
        old_pdf_a=pdf_a_path.read_bytes(),
        old_pdf_b=pdf_b_path.read_bytes(),
        old_record=record_path.read_bytes(),
        pdf_a_path=pdf_a_path,
        pdf_b_path=pdf_b_path,
        record_path=record_path,
        selected=selected,
    )


def test_rebind_rejects_wrong_history_drift_and_changed_rows(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    bundle = _prepare_minimal_rebind_bundle(tmp_path, monkeypatch)
    wrong_history = dict(bundle.expected)
    wrong_history["csv"] = "f" * 64
    with pytest.raises(
        fp.ArtifactValidationError,
        match="historical csv bytes",
    ):
        fp.rebind_authenticated_artifacts(
            object(),
            historical_source_commit="a" * 40,
            expected_previous_sha256=wrong_history,
        )

    drifted = json.loads(json.dumps(bundle.current))
    drifted["config"]["delta_0"] += 1.0
    bundle.selected["fingerprint"] = drifted
    with pytest.raises(
        fp.ArtifactValidationError,
        match="solve-relevant configuration changed",
    ):
        fp.rebind_authenticated_artifacts(
            object(),
            historical_source_commit="a" * 40,
            expected_previous_sha256=bundle.expected,
        )
    bundle.selected["fingerprint"] = bundle.current

    changed = bundle.candidate_bytes.replace(
        b"unchanged-numerical-row",
        b"tampered-numerical-row",
    )
    monkeypatch.setattr(
        fp,
        "_write_rebound_baseline",
        lambda _result, path: path.write_bytes(changed) and path,
    )
    with pytest.raises(
        fp.ArtifactValidationError,
        match="changed a schema marker, column header, or numerical row",
    ):
        fp.rebind_authenticated_artifacts(
            object(),
            historical_source_commit="a" * 40,
            expected_previous_sha256=bundle.expected,
        )
    assert bundle.csv_path.read_bytes() == bundle.old_csv
    assert bundle.record_path.read_bytes() == bundle.old_record


def test_rebind_rolls_back_if_commit_marker_promotion_fails(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    bundle = _prepare_minimal_rebind_bundle(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fp,
        "read_promotion_record",
        lambda *_args, **_kwargs: {},
    )
    real_replace = fp.os.replace
    stage_record = bundle.record_path.with_name(
        f".{bundle.record_path.stem}.{fp.os.getpid()}.rebind.stage.json"
    )

    def fail_commit_marker(source, destination):
        if (
            Path(source) == stage_record
            and Path(destination) == bundle.record_path
        ):
            raise OSError("injected commit-marker promotion failure")
        return real_replace(source, destination)

    monkeypatch.setattr(fp.os, "replace", fail_commit_marker)
    with pytest.raises(
        OSError,
        match="injected commit-marker promotion failure",
    ):
        fp.rebind_authenticated_artifacts(
            object(),
            historical_source_commit="a" * 40,
            expected_previous_sha256=bundle.expected,
        )
    assert bundle.csv_path.read_bytes() == bundle.old_csv
    assert bundle.record_path.read_bytes() == bundle.old_record
    assert bundle.pdf_a_path.read_bytes() == bundle.old_pdf_a
    assert bundle.pdf_b_path.read_bytes() == bundle.old_pdf_b
    assert bundle.campaign_path.read_bytes() == bundle.old_campaign
    assert not stage_record.exists()
    assert not bundle.csv_path.with_name(
        f".{bundle.csv_path.stem}.{fp.os.getpid()}.rebind.stage.csv"
    ).exists()


def test_rebind_rolls_back_after_promoted_marker_fails_validation(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    bundle = _prepare_minimal_rebind_bundle(tmp_path, monkeypatch)

    def validate_record(*_args, **kwargs):
        if kwargs.get("_publication_lock_held"):
            raise RuntimeError("injected post-promotion validation failure")
        return {}

    monkeypatch.setattr(fp, "read_promotion_record", validate_record)
    with pytest.raises(
        RuntimeError,
        match="injected post-promotion validation failure",
    ):
        fp.rebind_authenticated_artifacts(
            object(),
            historical_source_commit="a" * 40,
            expected_previous_sha256=bundle.expected,
        )
    assert bundle.csv_path.read_bytes() == bundle.old_csv
    assert bundle.record_path.read_bytes() == bundle.old_record
    assert bundle.pdf_a_path.read_bytes() == bundle.old_pdf_a
    assert bundle.pdf_b_path.read_bytes() == bundle.old_pdf_b
    assert bundle.campaign_path.read_bytes() == bundle.old_campaign


def test_rollback_attempts_every_snapshot_after_one_restore_fails(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    first = tmp_path / "first"
    second = tmp_path / "second"
    attempted: list[Path] = []

    def restore(path, payload):
        del payload
        attempted.append(path)
        if path == first:
            raise OSError("first restore failed")

    monkeypatch.setattr(fp, "_restore_payload", restore)
    with pytest.raises(RuntimeError, match="rollback was incomplete"):
        fp._restore_payloads({first: b"old-first", second: b"old-second"})
    assert attempted == [first, second]


def test_campaign_attestation_rejects_same_run_mixed_artifact_bundle(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    producer = {
        "fingerprint": fp.artifact_fingerprint(),
        "runtime": {"python": "historical"},
    }
    campaign = {
        "mode": fp._HISTORICAL_CAMPAIGN_MODE,
        "row_schema": fp._HISTORICAL_CAMPAIGN_ROW_SCHEMA,
        "rows": [
            *[
                {
                    "axis_value": float(axis_value),
                    "label": f"upper-t{row_index:02d}",
                    "ordinal": row_index,
                    "panel": "upper",
                    "row_index": row_index,
                    "sha256": f"{row_index + 1:064x}",
                    "size_bytes": row_index + 1,
                }
                for row_index, axis_value in enumerate(fp.UPPER_T_BATH_K)
            ],
            *[
                {
                    "axis_value": float(axis_value),
                    "label": f"lower-n{row_index:02d}",
                    "ordinal": len(fp.UPPER_T_BATH_K) + row_index,
                    "panel": "lower",
                    "row_index": row_index,
                    "sha256": (
                        f"{len(fp.UPPER_T_BATH_K) + row_index + 1:064x}"
                    ),
                    "size_bytes": len(fp.UPPER_T_BATH_K) + row_index + 1,
                }
                for row_index, axis_value in enumerate(fp.LOWER_NBAR)
            ],
        ],
        "runner": {"path": "scripts/old.py", "sha256": "7" * 64},
        "single_thread_environment": {"OMP_NUM_THREADS": "1"},
        "status_schema": fp._HISTORICAL_CAMPAIGN_STATUS_SCHEMA,
    }
    campaign_producer = fp._campaign_producer(producer, campaign)
    producer_sha256 = fp._campaign_producer_sha256(campaign_producer)
    run_identity = fp._campaign_run_identity(
        campaign_producer,
        row_schema=campaign["row_schema"],
        status_schema=campaign["status_schema"],
    )
    campaign["run_identity"] = run_identity
    rebound_from = {
        "artifacts": {
            name: {"sha256": digest, "size_bytes": 1}
            for name, digest in {
                "csv": "1" * 64,
                "pdf_a": "2" * 64,
                "pdf_b": "3" * 64,
            }.items()
        },
        "campaign": {},
        "payload_sha256": "4" * 64,
        "producer_sha256": producer_sha256,
        "promotion_record": {"sha256": "5" * 64, "size_bytes": 1},
        "run_identity": run_identity,
        "source_commit": "a" * 40,
    }
    row_hashes = {
        row["label"]: row["sha256"] for row in campaign["rows"]
    }
    artifacts = {
        "csv": {"path": "old.csv", "sha256": "f" * 64},
        "pdf_a": {"path": "old_a.pdf", "sha256": "2" * 64},
        "pdf_b": {"path": "old_b.pdf", "sha256": "3" * 64},
        "record": {"path": "old.json", "sha256": "5" * 64},
    }
    status = {
        "attempt": 1,
        "completed_rows": 6,
        "completion": {
            "aggregate_worker_s": 1.0,
            "artifacts": artifacts,
            "campaign": campaign,
            "certificate_maxima": {"metric": 0.0},
            "new_rows": 6,
            "resumed_rows": 0,
            "row_payload_sha256": row_hashes,
            "wall_s": 1.0,
        },
        "producer_sha256": producer_sha256,
        "run_identity": run_identity,
        "schema": campaign["status_schema"],
        "started_utc": "2026-01-01T00:00:00+00:00",
        "state": "complete",
        "total_rows": 6,
        "updated_utc": "2026-01-01T00:00:01+00:00",
    }
    campaign_path = tmp_path / "fig5.campaign.json"
    campaign_path.write_text(json.dumps(status), encoding="utf-8")
    rebound_from["campaign"] = fp._file_identity(campaign_path)
    monkeypatch.setattr(
        fp,
        "_artifact_metadata",
        lambda *_args, **_kwargs: {
            "certificate_maxima": {"metric": 0.0}
        },
    )
    original_axis = campaign["rows"][0]["axis_value"]
    campaign["rows"][0]["axis_value"] = original_axis + 0.01
    campaign_path.write_text(json.dumps(status), encoding="utf-8")
    rebound_from["campaign"] = fp._file_identity(campaign_path)
    with pytest.raises(
        fp.ArtifactValidationError,
        match=r"exact six-row Fig. 5 campaign topology",
    ):
        fp._read_campaign_attestation(
            producer=producer,
            rebound_from=rebound_from,
            csv_path=tmp_path / "unused.csv",
            campaign_path=campaign_path,
        )

    campaign["rows"][0]["axis_value"] = original_axis
    campaign_path.write_text(json.dumps(status), encoding="utf-8")
    rebound_from["campaign"] = fp._file_identity(campaign_path)
    with pytest.raises(
        fp.ArtifactValidationError,
        match="campaign csv evidence does not match",
    ):
        fp._read_campaign_attestation(
            producer=producer,
            rebound_from=rebound_from,
            csv_path=tmp_path / "unused.csv",
            campaign_path=campaign_path,
        )


def test_canonical_reader_refuses_mixed_snapshot_during_publication(
    tmp_path,
    monkeypatch,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    csv_path = tmp_path / "fig5.csv"
    pdf_a_path = tmp_path / "fig5_a.pdf"
    pdf_b_path = tmp_path / "fig5_b.pdf"
    record_path = tmp_path / "fig5.promotion.json"
    monkeypatch.setattr(fp, "baseline_path", lambda: csv_path)
    monkeypatch.setattr(fp, "plot_path_a", lambda: pdf_a_path)
    monkeypatch.setattr(fp, "plot_path_b", lambda: pdf_b_path)
    monkeypatch.setattr(fp, "promotion_record_path", lambda: record_path)
    for reader in (
        fp.read_baseline,
        fp.read_baseline_metadata,
        fp.read_promotion_record,
    ):
        with (
            fp._publication_lock(),
            pytest.raises(RuntimeError, match="publisher or reader"),
        ):
            reader()


def test_generate_rejects_source_drift_before_publication(
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    fingerprints = iter(({"source": "before"}, {"source": "after"}))
    monkeypatch.setattr(fp, "artifact_fingerprint", lambda: next(fingerprints))
    monkeypatch.setattr(
        fp,
        "producer_runtime_provenance",
        lambda: {"runtime": "fixed"},
    )
    monkeypatch.setattr(fp, "run_cached", lambda: schema_result)

    published = False

    def forbidden_publish(*_args, **_kwargs):
        nonlocal published
        published = True
        raise AssertionError("stale result reached publication")

    monkeypatch.setattr(fp, "publish_artifacts", forbidden_publish)
    with pytest.raises(fp.ArtifactValidationError, match="changed during generation"):
        fp.generate_baseline()
    assert not published


def test_source_manifest_includes_transitive_material_dependencies() -> None:
    import validation.fischer_2023.fig5_paper as fp

    sources = fp.source_hashes()
    assert "qpsim/materials/substrate.py" in sources


def test_writer_rejects_bad_producer_claim_or_missing_state(
    tmp_path,
    schema_result,
) -> None:
    forged_zero = replace(
        schema_result,
        upper_qp_backward_error=np.zeros_like(
            schema_result.upper_qp_backward_error
        ),
    )
    with pytest.raises(
        ArtifactValidationError,
        match="producer certificate",
    ):
        write_baseline(forged_zero, tmp_path / "forged-zero.csv")

    forged_values = schema_result.upper_qp_backward_error.copy()
    forged_values[0, 0] = 1.01 * fig5_solve.TARGET_BACKWARD_ERROR_LIMIT
    forged = replace(schema_result, upper_qp_backward_error=forged_values)
    with pytest.raises(ArtifactValidationError, match="above"):
        write_baseline(forged, tmp_path / "forged.csv")
    forged_observable = schema_result.upper_x_qp_num.copy()
    forged_observable[0, 0] += 1.0e-6
    with pytest.raises(ArtifactValidationError, match="persisted solver state"):
        write_baseline(
            replace(schema_result, upper_x_qp_num=forged_observable),
            tmp_path / "forged-observable.csv",
        )
    with pytest.raises(ArtifactValidationError, match="missing returned-state"):
        write_baseline(
            replace(schema_result, upper_f=None),
            tmp_path / "missing-state.csv",
        )
    with pytest.raises(ArtifactValidationError, match="cannot be boolean"):
        write_baseline(
            replace(
                schema_result,
                upper_qp_residual_inf=np.zeros_like(
                    schema_result.upper_qp_residual_inf,
                    dtype=bool,
                ),
            ),
            tmp_path / "boolean-certificate.csv",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("qp_backward_error", 1.01 * fig5_solve.TARGET_BACKWARD_ERROR_LIMIT),
        (
            certificate_module.QP_NUMBER_CERTIFICATE_FIELD,
            1.01 * fig5_solve.TARGET_BACKWARD_ERROR_LIMIT,
        ),
        (
            "phonon_backward_error",
            1.01 * fig5_paper.PHONON_REASSEMBLY_BACKWARD_ERROR_LIMIT,
        ),
        ("qp_residual_inf", -1.0),
    ],
)
def test_writer_rejects_bad_reassembled_certificate(
    tmp_path,
    monkeypatch,
    schema_result,
    field,
    value,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    valid_certificate = fp.certificate_module.steady_state_certificate

    def bad_certificate(*args, **kwargs):
        certificate = valid_certificate(*args, **kwargs)
        certificate[field] = value
        return certificate

    monkeypatch.setattr(
        fp.certificate_module,
        "steady_state_certificate",
        bad_certificate,
    )
    with pytest.raises(fp.ArtifactValidationError, match="reassembled certificate"):
        fp.write_baseline(schema_result, tmp_path / f"bad-live-{field}.csv")


def test_reader_rejects_corrupt_current_schema_artifacts(
    tmp_path,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    source = write_baseline(schema_result, tmp_path / "source.csv")
    original = _read_rows(source)
    cases: list[tuple[str, list[list[str]]]] = []

    wrong_marker = [row.copy() for row in original]
    wrong_marker[0] = ["# qpsim_artifact_schema=unknown"]
    cases.append(("wrong-marker", wrong_marker))

    stale_source = [row.copy() for row in original]
    prefix = "# qpsim_metadata="
    metadata = json.loads(stale_source[1][0][len(prefix):])
    source_hashes = metadata["fingerprint"]["source_sha256"]
    source_hashes[next(iter(source_hashes))] = "0" * 64
    stale_source[1][0] = prefix + fp._canonical_json(metadata)
    cases.append(("stale-source", stale_source))

    nonfinite = [row.copy() for row in original]
    nonfinite[3][4] = "nan"
    _refresh_payload_hash(nonfinite)
    cases.append(("nonfinite", nonfinite))

    negative = [row.copy() for row in original]
    negative[3][4] = "-1e-9"
    _refresh_payload_hash(negative)
    cases.append(("negative", negative))

    reordered = [row.copy() for row in original]
    reordered[3], reordered[4] = reordered[4], reordered[3]
    _refresh_payload_hash(reordered)
    cases.append(("reordered", reordered))

    truncated = [row.copy() for row in original[:-1]]
    cases.append(("truncated", truncated))

    checksum = [row.copy() for row in original]
    checksum[3][4] = f"{2.0 * float(checksum[3][4]):.17e}"
    cases.append(("checksum", checksum))

    for name, rows in cases:
        path = tmp_path / f"{name}.csv"
        _write_rows(path, rows)
        with pytest.raises(ArtifactValidationError):
            read_baseline(path)


def test_reader_recomputes_claims_after_hashes_are_reforged(
    tmp_path,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    source = write_baseline(schema_result, tmp_path / "source.csv")
    original = _read_rows(source)

    forged_certificate = [row.copy() for row in original]
    certificate_column = forged_certificate[2].index("qp_backward_error")
    forged_certificate[3][certificate_column] = (
        f"{2.0 * fig5_solve.TARGET_BACKWARD_ERROR_LIMIT:.17e}"
    )
    _refresh_payload_hash(forged_certificate)
    certificate_path = tmp_path / "forged-certificate.csv"
    _write_rows(certificate_path, forged_certificate)
    with pytest.raises(ArtifactValidationError, match="exceeds its gate"):
        read_baseline(certificate_path)

    forged_state = [row.copy() for row in original]
    f_column = forged_state[2].index("state_f_f64_zlib_base64")
    n_ph_column = forged_state[2].index("state_n_ph_f64_zlib_base64")
    hash_column = forged_state[2].index("state_sha256")
    f_values = fp._decode_state_array(
        forged_state[3][f_column], size=fp.NUM_BINS, name="test.f"
    )
    _, _, spectral = fp._build_grid_and_spectral(fp.NUM_BINS)
    omega, _, _, _ = fp.fig5_solve.build_phonon_frequency_map(spectral.E)
    n_ph_values = fp._decode_state_array(
        forged_state[3][n_ph_column], size=omega.size, name="test.n_ph"
    )
    f_values[0] += 1.0e-8
    forged_state[3][f_column] = fp._encode_state_array(f_values)
    forged_state[3][hash_column] = fp._state_sha256(
        panel=forged_state[3][0],
        T_bath=float(forged_state[3][1]),
        n_bar=float(forged_state[3][2]),
        f=f_values,
        n_ph=n_ph_values,
    )
    _refresh_payload_hash(forged_state)
    state_path = tmp_path / "forged-state.csv"
    _write_rows(state_path, forged_state)
    with pytest.raises(ArtifactValidationError, match="persisted solver state"):
        read_baseline(state_path)

    forged_maximum = [row.copy() for row in original]
    metadata = json.loads(forged_maximum[1][0][len(prefix := "# qpsim_metadata="):])
    metadata["certificate_maxima"]["qp_backward_error"] += 1.0e-12
    forged_maximum[1][0] = prefix + fp._canonical_json(metadata)
    maximum_path = tmp_path / "forged-maximum.csv"
    _write_rows(maximum_path, forged_maximum)
    with pytest.raises(ArtifactValidationError, match="maximum"):
        read_baseline(maximum_path)


def test_unknown_legacy_looking_input_fails_as_corrupt(tmp_path) -> None:
    path = tmp_path / "legacy-looking.csv"
    _write_rows(
        path,
        [
            ["# Fischer 2023 Fig. 5 â€” paper-topology reproduction"],
            ["# damaged"],
            ["panel", "T_bath_K"],
        ],
    )
    with pytest.raises(ArtifactValidationError):
        read_baseline(path)


def test_atomic_write_preserves_existing_destination(
    tmp_path,
    monkeypatch,
    schema_result,
) -> None:
    import validation.fischer_2023.fig5_paper as fp

    destination = tmp_path / "fig5.csv"
    destination.write_text("sentinel", encoding="utf-8")
    real_writer = csv.writer

    class FailingWriter:
        def __init__(self, stream, **kwargs):
            self.delegate = real_writer(stream, **kwargs)

        def writerow(self, row):
            return self.delegate.writerow(row)

        def writerows(self, rows):
            self.delegate.writerow(rows[0])
            raise OSError("injected write failure")

    monkeypatch.setattr(fp.csv, "writer", FailingWriter)
    with pytest.raises(OSError, match="injected"):
        fp.write_baseline(schema_result, destination)
    assert destination.read_text(encoding="utf-8") == "sentinel"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


class TestFig5CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves an unchanged two-panel solve from disk. The expensive solve
    is stubbed so the test is fast; it exercises the real cache + observables
    wiring (qp_fraction on the rebuilt grid + the analytic overlays). Engine-level
    key/store properties are covered in ``tests/validation/test_sweep_cache.py``.
    """

    _NE = 162  # commensurate reduced grid (omega_0/dE = 2)

    def _stub_payload(self) -> dict:
        import validation.fischer_2023.fig5_paper as fp

        ne = self._NE
        _, _, spectral = fp._build_grid_and_spectral(ne)
        omega, _, _, _ = fp.fig5_solve.build_phonon_frequency_map(spectral.E)
        payload = {
            "upper_f": np.full((1, 1, ne), 1e-6),
            "lower_f": np.full((1, 1, ne), 1e-6),
            "upper_n_ph": np.zeros((1, 1, omega.size)),
            "lower_n_ph": np.zeros((1, 1, omega.size)),
            "upper_T_bath": np.array([0.10]),
            "upper_nbar": np.array([1.0e7]),
            "lower_nbar": np.array([1.0e7]),
            "lower_T_bath": np.array([0.10]),
            "tau_0_pb_ns": np.array([0.255]),
            "tau_l_ns": np.array([0.255]),
            "num_bins": np.array([ne]),
        }
        for panel in ("upper", "lower"):
            for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
                payload[f"{panel}_{field}"] = np.full((1, 1), 1.0e-10)
        return payload

    def _cfg(self) -> dict:
        return {
            "num_bins": self._NE,
            "upper_T_bath": (0.10,),
            "upper_nbar": np.array([1.0e7]),
            "lower_nbar": (1.0e7,),
            "lower_T_bath": np.array([0.10]),
        }

    def test_run_cached_hits_disk_on_second_call(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig5_paper as fp

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

        ref = fp.observables(payload)
        for res in (r1, r2):
            for fld in ("upper_x_qp_num", "upper_x_qp_analytic",
                        "lower_x_qp_num", "lower_x_qp_analytic", "upper_T_star"):
                np.testing.assert_array_equal(getattr(res, fld), getattr(ref, fld))

    def test_run_cached_disabled_always_recomputes(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig5_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached(**self._cfg())
        fp.run_cached(**self._cfg())
        assert calls["n"] == 2  # disabled -> recompute each call
