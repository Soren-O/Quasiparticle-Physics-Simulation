"""Regression test for the Fischer 2023 Fig. 7 paper-facing validation."""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2023.fig7_paper import (
    NUM_BINS,
    P_READ_DBM,
    Q_EXT_BY_DBM,
    T_BATH_VALUES,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
)
from validation.fischer_2023.fig7_solve import _nbar_from_table_iii


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
    assert cfg.t_c == pytest.approx(meta.t_c, rel=1e-6)  # header stores 6 dp
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.alpha == pytest.approx(meta.alpha)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.tau_l == pytest.approx(meta.tau_l)
    assert cfg.tau_0_pb == pytest.approx(meta.tau_0_pb)

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


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


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
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
    result = run(
        temperatures=tuple(float(x) for x in baseline.T_bath),
        powers_dbm=baseline.p_read_dbm,
    )

    np.testing.assert_allclose(result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14)
    for p in baseline.p_read_dbm:
        assert result.n_bar_by_dbm[p] == pytest.approx(baseline.n_bar_by_dbm[p], rel=1e-12)
        # Compare qp losses (1/Q_qp), not Q_qp: on the extrinsic plateau
        # Q_qp reaches ~1e17-1e18 where the digits are pure float noise;
        # the 1e-18 loss floor ignores that. rtol is 1e-3, not 1e-4: the
        # baseline is win32-pinned and ubuntu CI reproducibly lands up to
        # ~2e-4 away at ordinary points (solver-chain platform noise, far
        # below the ~% level of agreement with the published figure), so
        # 1e-3 is the tightest cross-platform-safe regression gate.
        np.testing.assert_allclose(
            1.0 / np.asarray(result.Q_qp_by_dbm[p]),
            1.0 / np.asarray(baseline.Q_qp_by_dbm[p]),
            rtol=1e-3, atol=1e-18, err_msg=f"Q_qp loss drift at P_read={p:g} dBm",
        )
        np.testing.assert_allclose(
            result.Q_tot_by_dbm[p], baseline.Q_tot_by_dbm[p],
            rtol=1e-3, atol=1e-14, err_msg=f"Q_tot drift at P_read={p:g} dBm",
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

    def _stub_payload(self) -> dict:
        # One (power, T) point on the real NUM_BINS grid; f is a placeholder
        # occupation — observables only needs a valid array on the grid.
        f_solved = np.full((1, 1, NUM_BINS), 1e-6, dtype=float)
        return {
            "f_solved": f_solved,
            "temperatures": np.array([0.10], dtype=float),
            "powers_dbm": np.array([-64.0], dtype=float),
            "n_bar": np.array([1.234e5], dtype=float),
            "num_bins": np.array([NUM_BINS]),
        }

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
            assert res.n_bar_by_dbm[-64.0] == pytest.approx(1.234e5)

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
