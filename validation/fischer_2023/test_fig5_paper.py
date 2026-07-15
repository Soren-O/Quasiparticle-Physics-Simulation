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

import numpy as np
import pytest

from validation.fischer_2023.fig5_paper import (
    LOWER_NBAR,
    LOWER_T_BATH_K,
    UPPER_NBAR_VALUES,
    UPPER_T_BATH_K,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
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


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


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

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the multi-minute run() below, instead of after it.
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
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


class TestFig5CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves an unchanged two-panel solve from disk. The expensive solve
    is stubbed so the test is fast; it exercises the real cache + observables
    wiring (qp_fraction on the rebuilt grid + the analytic overlays). Engine-level
    key/store properties are covered in ``tests/validation/test_sweep_cache.py``.
    """

    _NE = 162  # commensurate reduced grid (omega_0/dE = 2)

    def _stub_payload(self) -> dict:
        ne = self._NE
        return {
            "upper_f": np.full((1, 1, ne), 1e-6),
            "lower_f": np.full((1, 1, ne), 1e-6),
            "upper_T_bath": np.array([0.10]),
            "upper_nbar": np.array([1.0e7]),
            "lower_nbar": np.array([1.0e7]),
            "lower_T_bath": np.array([0.10]),
            "tau_0_pb_ns": np.array([0.255]),
            "tau_l_ns": np.array([0.255]),
            "num_bins": np.array([ne]),
        }

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
