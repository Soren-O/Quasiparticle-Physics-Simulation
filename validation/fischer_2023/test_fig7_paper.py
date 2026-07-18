"""Regression test for the Fischer 2023 Fig. 7 paper-facing validation."""

from __future__ import annotations

import platform
from dataclasses import replace

import numpy as np
import pytest

from validation.fischer_2023.fig7_paper import (
    NUM_BINS,
    P_READ_DBM,
    Q_EXT_BY_DBM,
    Q_TOTAL_CROSS_PLATFORM_RTOL,
    Q_TOTAL_REGRESSION_RTOL,
    QP_LOSS_CROSS_PLATFORM_RTOL,
    QP_LOSS_REGRESSION_ATOL,
    QP_LOSS_REGRESSION_RTOL,
    T_BATH_VALUES,
    LegacyArtifactError,
    baseline_path,
    config_metadata,
    fig7_regression_tolerances,
    observables,
    read_baseline,
    read_baseline_metadata,
    reproduction_provenance,
    run,
    solve_contract_digest,
    write_baseline,
)
from validation.fischer_2023.fig7_solve import (
    SOLVER_KWARGS,
    TARGET_BACKWARD_ERROR_LIMIT,
    _nbar_from_table_iii,
    _validated_sweep_request,
)
from validation.fischer_2023.steady_state_certificate import (
    CERTIFICATE_FIELDS,
    CERTIFICATE_METRIC_VERSION,
)


def _assert_certified_baseline_balances(base) -> None:
    for field in CERTIFICATE_FIELDS:
        values = getattr(base, field)
        assert np.all(np.isfinite(values)), (
            f"certified baseline has non-finite {field}"
        )
    for field in ("qp_backward_error", "phonon_backward_error"):
        values = getattr(base, field)
        assert np.all(values <= TARGET_BACKWARD_ERROR_LIMIT), (
            f"certified baseline {field} exceeds "
            f"{TARGET_BACKWARD_ERROR_LIMIT:g}: {values.tolist()}"
        )


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~instant, no solve): the live module config must match
    the pinned baseline's stamped header, power set, and per-power n_bar.

    Gating :func:`run` behind this flags a stale config/baseline pairing — a
    grid change, a Table II/III parameter edit, a T*/Δ-mapping change — in a
    fraction of a second. (Fig. 7's slow run can separately fail inside the
    Picard solver; this preflight does not address that. See ``fig6_paper`` for
    the same pattern, where ``run()`` is ~14 h.)
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)
    base = read_baseline(path)

    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_c == meta.t_c
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.alpha == pytest.approx(meta.alpha)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_l == pytest.approx(meta.tau_l)
    assert cfg.tau_0_pb == pytest.approx(meta.tau_0_pb)
    assert cfg.certificate_metric_version == meta.certificate_metric_version
    assert cfg.certificate_metric_version == CERTIFICATE_METRIC_VERSION
    assert cfg.target_backward_error_limit == pytest.approx(
        meta.target_backward_error_limit,
    )
    assert cfg.solve_contract_digest == meta.solve_contract_digest
    assert meta.generator_platform
    assert meta.generator_python
    assert meta.generator_numpy
    assert meta.generator_scipy

    assert tuple(P_READ_DBM) == base.p_read_dbm, (
        f"P_read powers config={tuple(P_READ_DBM)} != baseline {base.p_read_dbm}; "
        "regenerate the baseline or restore the power set before the slow run."
    )
    np.testing.assert_allclose(
        np.asarray(T_BATH_VALUES, dtype=float), base.T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="T_bath axis (range/count) differs from baseline",
    )
    for p in base.p_read_dbm:
        assert _nbar_from_table_iii(p) == pytest.approx(
            base.n_bar_by_dbm[p], rel=1e-12,
        ), f"n_bar(P={p:g} dBm) config != baseline — Table III T*/Δ mapping changed?"
    _assert_certified_baseline_balances(base)


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    if "certificate_metric_version=" not in path.read_text(encoding="utf-8"):
        pytest.xfail(
            "legacy Fig. 7 baseline is not independently certified; "
            "the active full regeneration must complete before preflight can pass"
        )
    try:
        _assert_config_matches_baseline(path)
    except LegacyArtifactError as error:
        pytest.xfail(str(error))


def test_reproducible_solver_contract_is_tight() -> None:
    """The former loose inner Newton gate caused percent-scale pin drift."""
    assert SOLVER_KWARGS["newton_tol"] == pytest.approx(1e-13)
    assert SOLVER_KWARGS["newton_backward_error_tol"] == pytest.approx(1e-8)
    assert SOLVER_KWARGS["picard_tol"] == pytest.approx(1e-9)
    assert SOLVER_KWARGS["picard_atol"] == pytest.approx(1e-13)
    assert SOLVER_KWARGS["picard_balance_tol"] == pytest.approx(1e-8)
    assert pytest.approx(2e-8) == TARGET_BACKWARD_ERROR_LIMIT


def test_regression_tolerances_are_platform_scoped() -> None:
    strict_loss, strict_q_tot = fig7_regression_tolerances(
        "Windows-11-10.0.26100-SP0", running_system="Windows"
    )
    assert strict_loss == pytest.approx(QP_LOSS_REGRESSION_RTOL)
    assert strict_q_tot == pytest.approx(Q_TOTAL_REGRESSION_RTOL)
    assert fig7_regression_tolerances(
        "macOS-15.5-arm64-arm-64bit", running_system="Darwin"
    ) == pytest.approx((QP_LOSS_REGRESSION_RTOL, Q_TOTAL_REGRESSION_RTOL))
    assert fig7_regression_tolerances(
        "Windows-11-10.0.26100-SP0", running_system="Darwin"
    ) == pytest.approx((QP_LOSS_REGRESSION_RTOL, Q_TOTAL_REGRESSION_RTOL))

    portable_loss, portable_q_tot = fig7_regression_tolerances(
        "Windows-11-10.0.26100-SP0", running_system="Linux"
    )
    assert portable_loss == pytest.approx(QP_LOSS_CROSS_PLATFORM_RTOL)
    assert portable_q_tot == pytest.approx(Q_TOTAL_CROSS_PLATFORM_RTOL)

    pinned_loss = np.array([4.062049012662113e-8])
    hosted_loss = np.array([4.043230878753479e-8])
    pinned_q_tot = np.array([868257.8466418125])
    hosted_q_tot = np.array([868399.734408639])
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            hosted_loss,
            pinned_loss,
            rtol=strict_loss,
            atol=QP_LOSS_REGRESSION_ATOL,
        )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            hosted_q_tot, pinned_q_tot, rtol=strict_q_tot, atol=0.0
        )
    np.testing.assert_allclose(
        hosted_loss,
        pinned_loss,
        rtol=portable_loss,
        atol=QP_LOSS_REGRESSION_ATOL,
    )
    np.testing.assert_allclose(
        hosted_q_tot, pinned_q_tot, rtol=portable_q_tot, atol=0.0
    )
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([1.009], [1.0], rtol=portable_loss, atol=0.0)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose([1.0003], [1.0], rtol=portable_q_tot, atol=0.0)


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig7_paper"
        )

    # Cheap preflight first: reject a stale config/baseline pairing before the
    # (slow, and currently Picard-fragile) run() below.
    try:
        _assert_config_matches_baseline(path)
    except LegacyArtifactError as error:
        pytest.xfail(str(error))

    metadata = read_baseline_metadata(path)
    loss_rtol, q_tot_rtol = fig7_regression_tolerances(
        metadata.generator_platform
    )
    baseline = read_baseline(path)
    result = run(
        temperatures=tuple(float(x) for x in baseline.T_bath),
        powers_dbm=baseline.p_read_dbm,
    )

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    for p in baseline.p_read_dbm:
        assert result.n_bar_by_dbm[p] == pytest.approx(baseline.n_bar_by_dbm[p], rel=1e-12)
        # Compare qp losses (1/Q_qp), not Q_qp.  The historical atol=1e-10
        # masked every loss below Q=1e10.  The replacement 2e-19 floor is nine
        # orders smaller and covers only the measured T=0.06 K cross-platform
        # tail envelope; every physically visible point is controlled by the
        # same-platform 0.4% relative gate. Q_tot has its own 1e-4 gate;
        # comparisons against a pin generated on another OS use the narrowly
        # measured 0.8% / 2e-4 portability envelopes.
        np.testing.assert_allclose(
            1.0 / np.asarray(result.Q_qp_by_dbm[p]),
            1.0 / np.asarray(baseline.Q_qp_by_dbm[p]),
            rtol=loss_rtol,
            atol=QP_LOSS_REGRESSION_ATOL,
            err_msg=(
                f"Q_qp loss drift at P_read={p:g} dBm "
                f"(pinned_on={metadata.generator_platform!r}, "
                f"running_on={platform.system()!r}, rtol={loss_rtol:g})"
            ),
        )
        np.testing.assert_allclose(
            result.Q_tot_by_dbm[p], baseline.Q_tot_by_dbm[p],
            rtol=q_tot_rtol,
            atol=0.0,
            err_msg=(
                f"Q_tot drift at P_read={p:g} dBm "
                f"(pinned_on={metadata.generator_platform!r}, "
                f"running_on={platform.system()!r}, rtol={q_tot_rtol:g})"
            ),
        )


@pytest.mark.slow
def test_low_temperature_plateau_is_extrinsic_limited() -> None:
    result = run(temperatures=(0.06,), powers_dbm=(-64.0,))
    assert result.Q_qp_by_dbm[-64.0][0] > 1e12
    assert result.Q_tot_by_dbm[-64.0][0] == pytest.approx(
        Q_EXT_BY_DBM[-64.0], rel=1e-6,
    )


class TestFig7CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves an unchanged solve from disk. The expensive solve is stubbed
    so the test is fast; it exercises the real cache + observables wiring (not the
    Picard solver). Engine-level key/store properties are covered separately in
    ``tests/validation/test_sweep_cache.py``.
    """

    def _stub_payload(
        self,
        *,
        temperatures: tuple[float, ...] = (0.10,),
        powers_dbm: tuple[float, ...] = (-64.0,),
    ) -> dict:
        # One (power, T) point on the real NUM_BINS grid; f is a placeholder
        # occupation — observables only needs a valid array on the grid.
        shape = (len(powers_dbm), len(temperatures))
        f_solved = np.full((*shape, NUM_BINS), 1e-6, dtype=float)
        payload = {
            "f_solved": f_solved,
            "temperatures": np.asarray(temperatures, dtype=float),
            "powers_dbm": np.asarray(powers_dbm, dtype=float),
            "n_bar": np.asarray(
                [_nbar_from_table_iii(power) for power in powers_dbm],
                dtype=float,
            ),
            "num_bins": np.array([NUM_BINS]),
        }
        payload.update(
            {
                field: np.full(shape, value, dtype=float)
                for field, value in zip(
                    CERTIFICATE_FIELDS,
                    (1e-20, 1e-8, 2e-20, 0.3, 2e-8),
                    strict=True,
                )
            }
        )
        return payload

    def test_run_cached_hits_disk_on_second_call(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig7_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "1")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        r1 = fp.run_cached(temperatures=(0.10,), powers_dbm=(-64.0,))
        assert calls["n"] == 1  # cache miss -> solve ran once

        r2 = fp.run_cached(temperatures=(0.10,), powers_dbm=(-64.0,))
        assert calls["n"] == 1  # cache hit -> solve NOT re-run

        ref = fp.observables(payload)
        for res in (r1, r2):
            np.testing.assert_allclose(res.Q_qp_by_dbm[-64.0], ref.Q_qp_by_dbm[-64.0])
            np.testing.assert_allclose(res.Q_tot_by_dbm[-64.0], ref.Q_tot_by_dbm[-64.0])
            np.testing.assert_allclose(res.sigma1_by_dbm[-64.0], ref.sigma1_by_dbm[-64.0])
            for field in CERTIFICATE_FIELDS:
                np.testing.assert_array_equal(
                    getattr(res, field),
                    getattr(ref, field),
                )
            assert res.n_bar_by_dbm[-64.0] == pytest.approx(
                _nbar_from_table_iii(-64.0)
            )

    def test_run_cached_disabled_always_recomputes(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig7_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached(temperatures=(0.10,), powers_dbm=(-64.0,))
        fp.run_cached(temperatures=(0.10,), powers_dbm=(-64.0,))
        assert calls["n"] == 2  # disabled -> recompute each call

    def test_legacy_six_column_baseline_is_rejected_as_uncertified(
        self,
        tmp_path,
    ) -> None:
        path = tmp_path / "legacy_fig7.csv"
        path.write_text(
            "# legacy Fig. 7 baseline\n"
            "# p_read_dbm=-64\n"
            "T_bath_K,P_read_dbm,n_bar,Q_qp,Q_tot,sigma1\n"
            "0.1,-64,1e5,1e12,7e5,1e-12\n",
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError, match="certificate metric"):
            read_baseline(path)

    def test_certified_baseline_round_trip(self, tmp_path) -> None:
        reference = observables(self._stub_payload())
        path = write_baseline(reference, tmp_path / "certified_fig7.csv")
        assert b"\r" not in path.read_bytes()
        restored = read_baseline(path)

        for field in CERTIFICATE_FIELDS:
            np.testing.assert_array_equal(
                getattr(restored, field),
                getattr(reference, field),
            )

    @pytest.mark.parametrize(
        ("temperatures", "powers", "match"),
        [
            ((0.10, 0.10), (-64.0,), "temperatures"),
            ((0.10,), (-64.0, -64.0), "powers"),
            ((0.10,), (-63.0,), "Table-III"),
            ((0.0,), (-64.0,), "temperatures"),
        ],
    )
    def test_solve_request_rejects_ambiguous_axes_before_work(
        self,
        temperatures,
        powers,
        match,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _validated_sweep_request(temperatures, powers, NUM_BINS)

    def test_raw_payload_rejects_duplicate_power_keys(self) -> None:
        payload = self._stub_payload(powers_dbm=(-64.0, -64.0))
        with pytest.raises(ValueError, match="powers must be finite and unique"):
            observables(payload)

    @pytest.mark.parametrize(
        ("mutation", "match"),
        [
            (lambda payload: payload.__setitem__("f_solved", payload["f_solved"][..., :-1]), "f_solved has shape"),
            (lambda payload: payload["f_solved"].__setitem__((0, 0, 0), -1.0), "occupations in"),
            (lambda payload: payload["n_bar"].__setitem__(0, 2.0), "Table-III power axis"),
            (lambda payload: payload.__setitem__("num_bins", np.array([NUM_BINS + 0.5])), "finite integer"),
        ],
    )
    def test_raw_payload_rejects_shape_and_domain_corruption(
        self,
        mutation,
        match,
    ) -> None:
        payload = self._stub_payload()
        mutation(payload)
        with pytest.raises(ValueError, match=match):
            observables(payload)

    def test_reproduction_provenance_binds_source_and_environment(self) -> None:
        digest = solve_contract_digest()
        provenance = reproduction_provenance()

        assert len(digest) == 64
        assert provenance["solve_contract_digest"] == digest
        assert provenance["python"]
        assert provenance["numpy"]
        assert provenance["scipy"]
        assert provenance["platform"]
        assert provenance["solver_fingerprint"]["num_bins"] == NUM_BINS

    def test_reader_rejects_observable_not_bound_to_sigma1(self, tmp_path) -> None:
        reference = observables(self._stub_payload())
        path = write_baseline(reference, tmp_path / "forged_observable_fig7.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[-1].split(",")
        fields[3] = f"{1.01 * float(fields[3]):.17e}"
        lines[-1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match=r"Q_qp .* not bound to sigma1"):
            read_baseline(path)

    def test_duplicate_cartesian_key_replacing_missing_key_is_rejected(
        self,
        tmp_path,
    ) -> None:
        reference = observables(
            self._stub_payload(
                temperatures=(0.10, 0.20),
                powers_dbm=(-68.0, -64.0),
            )
        )
        path = write_baseline(reference, tmp_path / "duplicate_fig7.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        first_data = next(i for i, line in enumerate(lines) if line[0].isdigit())
        # Preserve both per-power row counts while replacing (-64, 0.20) by a
        # duplicate (-64, 0.10), the exact corruption the old reader accepted.
        lines[first_data + 3] = lines[first_data + 2]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match=r"duplicate .*P_read, T_bath"):
            read_baseline(path)

    def test_missing_cartesian_key_is_rejected(self, tmp_path) -> None:
        reference = observables(
            self._stub_payload(
                temperatures=(0.10, 0.20),
                powers_dbm=(-68.0, -64.0),
            )
        )
        path = write_baseline(reference, tmp_path / "missing_fig7.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        lines.pop()
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="not the exact Cartesian"):
            read_baseline(path)

    @pytest.mark.parametrize(
        ("replacement", "match"),
        [
            ("nan", "non-finite data or certificates"),
            (f"{2.0 * TARGET_BACKWARD_ERROR_LIMIT:.17e}", "above target"),
        ],
    )
    def test_bad_persisted_certificate_is_rejected(
        self,
        tmp_path,
        replacement: str,
        match: str,
    ) -> None:
        reference = observables(self._stub_payload())
        path = write_baseline(reference, tmp_path / "bad_certificate_fig7.csv")
        lines = path.read_text(encoding="utf-8").splitlines()
        fields = lines[-1].split(",")
        fields[7] = replacement  # qp_backward_error
        lines[-1] = ",".join(fields)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match=match):
            read_baseline(path)

    def test_malformed_current_schema_is_rejected(self, tmp_path) -> None:
        reference = observables(self._stub_payload())
        path = write_baseline(reference, tmp_path / "malformed_fig7.csv")
        text = path.read_text(encoding="utf-8")
        text = text.replace(",phonon_backward_error\n", "\n", 1)
        path.write_text(text, encoding="utf-8")

        with pytest.raises(RuntimeError, match="expected current certified schema"):
            read_baseline(path)

    def test_invalid_result_does_not_replace_existing_artifact(self, tmp_path) -> None:
        reference = observables(self._stub_payload())
        bad = replace(
            reference,
            qp_backward_error=np.array([[2.0 * TARGET_BACKWARD_ERROR_LIMIT]]),
        )
        path = tmp_path / "preserved_fig7.csv"
        path.write_text("sentinel\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match="above target"):
            write_baseline(bad, path)
        assert path.read_text(encoding="utf-8") == "sentinel\n"

    def test_certified_preflight_rejects_bad_balance(self) -> None:
        import validation.fischer_2023.fig7_paper as fp

        reference = fp.observables(self._stub_payload())
        bad = replace(
            reference,
            qp_backward_error=np.array([[2.0 * TARGET_BACKWARD_ERROR_LIMIT]]),
        )

        with pytest.raises(AssertionError, match="qp_backward_error exceeds"):
            _assert_certified_baseline_balances(bad)
