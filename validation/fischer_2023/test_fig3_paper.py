"""Regression test: Fischer 2023 Fig. 3 paper-target run matches the pinned CSV.

Slow-marked: this run does the τ_l = 0 thermal-phonon Newton plus the
13-step branch-preserving Picard continuation, with a same-ratio coupled-
Newton polish at ratio 10 on the 1620-bin paper grid. Curve tolerances scale
with the pinned signal rather than using the former vacuous absolute ``1e-6``.
Opt in with ``pytest -m slow``.

First-time generation::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np
import pytest
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights

from validation.fischer_2023.fig3_paper import (
    CURVE_REGRESSION_ATOL_OVER_PEAK,
    CURVE_REGRESSION_RTOL,
    STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL,
    baseline_path,
    config_metadata,
    curve_regression_rtol,
    observables,
    read_baseline,
    read_baseline_metadata,
    run,
    write_baseline,
)
from validation.fischer_2023.fig3_solve import (
    CONTINUATION_RATIOS,
    DELTA_0,
    INNER_QP_BACKWARD_ERROR_LIMIT,
    TARGET_BACKWARD_ERROR_LIMIT,
    _solve_coupled_newton,
    _solve_picard,
    _solve_tau_l_zero,
    _validate_ratio_ladder,
)
from validation.fischer_2023.fig3_solve import solve as solve_raw
from validation.fischer_2023.steady_state_certificate import (
    CERTIFICATE_FIELDS,
    CERTIFICATE_METRIC_VERSION,
)

# Ratios through one retain the strict 1e-4 curve gate.  The residual-polished
# ratio-10 state uses its measured 1.5% fixed-grid envelope only for the
# Windows/Linux OS-family case. The peak-scaled absolute floor is
# O(1e-16..1e-14), not the old vacuous 1e-6.


def _small_certified_result():
    ratios = np.asarray([0.0, 0.1, 1.0, 10.0])
    return observables(
        {
            "E": np.asarray([180.5, 181.5, 182.5]),
            "f_FD": np.asarray([3e-9, 2e-9, 1e-9]),
            "f_ratios": np.full((ratios.size, 3), 1e-8),
            "ratios": ratios,
            "tau_0_pb_ns": np.asarray([0.255]),
            "qp_residual_inf": np.asarray([1e-20, 2e-20, 3e-20, 4e-20]),
            "qp_backward_error": np.asarray([1e-12, 2e-12, 3e-12, 4e-12]),
            "phonon_residual_inf": np.asarray([np.nan, 2e-18, 3e-18, 4e-18]),
            "phonon_raw_backward_error": np.asarray(
                [np.nan, 2e-9, 3e-9, 4e-9]
            ),
            "phonon_backward_error": np.asarray([np.nan, 2e-8, 3e-8, 4e-8]),
        }
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

    Gating :func:`run` (the several-minute continuation ladder) behind this
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
    assert meta.pinned_on
    assert set(meta.certificate_maxima) == set(CERTIFICATE_FIELDS)
    assert np.all(np.isfinite(tuple(meta.certificate_maxima.values())))
    for field in ("qp_backward_error", "phonon_backward_error"):
        assert meta.certificate_maxima[field] <= TARGET_BACKWARD_ERROR_LIMIT


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header. Mirrors the inline gate in the slow test below."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


def test_baseline_curves_are_nonvacuous() -> None:
    """Fast data sanity gate: every pinned legend curve carries signal."""
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    baseline = read_baseline(path)
    _assert_baseline_curves_are_nonvacuous(baseline)


def test_baseline_roundtrip_preserves_certificate_maxima(tmp_path) -> None:
    expected = _small_certified_result()
    path = write_baseline(expected, tmp_path / "fig3.csv")
    restored = read_baseline(path)
    metadata = read_baseline_metadata(path)

    assert restored.certificate_maxima == expected.certificate_maxima
    assert metadata.certificate_metric_version == CERTIFICATE_METRIC_VERSION
    assert metadata.target_backward_error_limit == TARGET_BACKWARD_ERROR_LIMIT
    assert metadata.certificate_maxima == expected.certificate_maxima
    assert metadata.pinned_on == sys.platform
    header = path.read_text(encoding="utf-8")
    assert f"# pinned_on: {sys.platform}" in header
    assert "certificate_metric_version=" in header
    assert "target_backward_error_limit=1e-05" in header
    assert b"\r\n" not in path.read_bytes()

    legacy_path = tmp_path / "fig3-cp1252.csv"
    legacy_path.write_text(header, encoding="cp1252")
    legacy = read_baseline(legacy_path)
    assert legacy.certificate_maxima == expected.certificate_maxima


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
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig3_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the several-minute run() below, instead of after it.
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
    metadata = read_baseline_metadata(path)
    _assert_baseline_curves_are_nonvacuous(baseline)
    result = run()

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
        return {
            "E": np.linspace(180.0, 200.0, ne),
            "f_FD": np.full(ne, 1e-9),
            "f_ratios": np.full((3, ne), 1e-8),
            "ratios": np.array([0.0, 0.1, 1.0]),
            "tau_0_pb_ns": np.array([0.2515]),
        }

    def _cfg(self) -> dict:
        return {
            "num_bins": 162,
            "paper_ratios": (0.0, 0.1, 1.0),
            "continuation_ratios": (0.1, 0.3, 0.5, 1.0),
        }

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

        ref = fp.observables(payload)
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
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached(**self._cfg())
        fp.run_cached(**self._cfg())
        assert calls["n"] == 2  # disabled -> recompute each call
