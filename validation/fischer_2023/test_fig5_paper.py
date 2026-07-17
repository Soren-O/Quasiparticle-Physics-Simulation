"""Regression test: Fischer 2023 Fig. 5 paper-topology run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- each
panel does dozens of finite-τ_l Picard solves on the 1620-bin paper
grid; total wall-time is on the order of an hour. Opt in with
``pytest -m slow``.

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
from dataclasses import replace

import numpy as np
import pytest

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
    read_baseline_metadata,
    run,
    write_baseline,
)


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + both panels' sweep axes.

    Gating :func:`run` (the multi-minute two-panel Picard sweep) behind this
    turns a stale config/baseline pairing — a grid change, a sweep-range edit,
    a τ_0^PB drift — into a seconds-long failure instead of one discovered only
    after the full run. (See ``fig6_paper`` for the same pattern, where
    ``run()`` is ~14 h.)
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)
    axes = read_baseline(path)

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


def test_legacy_canonical_is_explicitly_quarantined() -> None:
    """Only the exact known pre-schema canonical receives legacy treatment."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    with pytest.raises(LegacyArtifactError):
        read_baseline(path)
    with pytest.raises(LegacyArtifactError):
        read_baseline_metadata(path)


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
    np.testing.assert_allclose(
        result.upper_x_qp_num, baseline.upper_x_qp_num, rtol=0.0, atol=1e-6,
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
        result.lower_x_qp_num, baseline.lower_x_qp_num, rtol=0.0, atol=1e-6,
        err_msg="Lower-panel numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.lower_x_qp_analytic, baseline.lower_x_qp_analytic,
        rtol=1e-10, atol=0.0,
        err_msg="Lower-panel analytic x_qp drift",
    )
    for panel in ("upper", "lower"):
        for field in certificate_module.CERTIFICATE_FIELDS:
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


def test_writer_rejects_forged_certificate_or_missing_state(
    tmp_path,
    schema_result,
) -> None:
    forged_values = schema_result.upper_qp_backward_error.copy()
    forged_values[0, 0] += 1.0e-12
    forged = replace(schema_result, upper_qp_backward_error=forged_values)
    with pytest.raises(ArtifactValidationError, match="persisted solver state"):
        write_baseline(forged, tmp_path / "forged.csv")
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
    forged_certificate[3][certificate_column] = f"{2.0e-12:.17e}"
    _refresh_payload_hash(forged_certificate)
    certificate_path = tmp_path / "forged-certificate.csv"
    _write_rows(certificate_path, forged_certificate)
    with pytest.raises(ArtifactValidationError, match="persisted solver state"):
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
            for field in certificate_module.CERTIFICATE_FIELDS:
                payload[f"{panel}_{field}"] = np.full((1, 1), 1.0e-8)
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
