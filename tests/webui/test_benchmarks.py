"""The benchmark framework itself — scoring, gating, and failure modes.

These test the machinery, not the physics. The physics each benchmark
asserts is checked by the benchmark's own registration (and by
``validation/analytic/``); what matters here is that a *wrong* curve is
reported as wrong, that a check which cannot run degrades to a note instead
of destroying a completed run, and that a green verdict cannot be produced by
an absent comparison.

That last one is the point. The failure this framework exists to prevent is a
check that silently does nothing and reads as a pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("pydantic", reason="benchmark tests need the qpsim[ui] extra")

from qpsim.webui import benchmarks
from qpsim.webui.benchmarks import Benchmark, Curve, residuals


class _Setup:
    """Minimal stand-in for a setup: the framework only reads ``mode``."""

    def __init__(self, mode: str = "transient_0d") -> None:
        self.mode = mode


def _curve(sim: np.ndarray, ana: np.ndarray, **kw: Any) -> Curve:
    return Curve(
        x=np.arange(sim.shape[-1], dtype=float),
        y_sim=sim,
        y_analytic=ana,
        x_label="t",
        y_label="y",
        **kw,
    )


def _bench(name: str, build: Any, *, rel_tol: float = 1e-6, **kw: Any) -> Benchmark:
    return Benchmark(
        name=name,
        title=name,
        tier=kw.pop("tier", "T1"),
        formula_latex="y = 1",
        reason="because",
        rel_tol=rel_tol,
        convergence="measured",
        modes=kw.pop("modes", ("transient_0d",)),
        build=build,
        **kw,
    )


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """Each test registers into a pristine registry and leaves none behind."""
    saved = dict(benchmarks._REGISTRY)
    benchmarks._REGISTRY.clear()
    yield
    benchmarks._REGISTRY.clear()
    benchmarks._REGISTRY.update(saved)


# -- scoring ----------------------------------------------------------


def test_identical_curves_score_zero() -> None:
    y = np.array([1.0, 2.0, 3.0])
    score = residuals(_curve(y, y.copy()))
    assert score["max_rel_err"] == 0.0
    assert score["scale_rel_err"] == 0.0
    assert score["n_points"] == 3


def test_pointwise_error_uses_each_point_not_the_peak() -> None:
    """A small absolute miss on a small value is a LARGE relative miss.

    This is the distinction that makes a decay curve worth checking: an
    exponential spans decades, and scoring it against the peak would let the
    entire tail be wrong while still reporting agreement.
    """
    ana = np.array([1.0, 1e-6])
    sim = np.array([1.0, 2e-6])  # the tail is 100% wrong
    score = residuals(_curve(sim, ana))
    assert score["max_rel_err"] == pytest.approx(1.0)
    # Against the peak the same miss looks like nothing at all.
    assert score["scale_rel_err"] == pytest.approx(1e-6)


def test_scale_metric_survives_a_zero_crossing() -> None:
    """A curve through zero has no meaningful pointwise relative error."""
    ana = np.array([1.0, 0.0, -1.0])
    sim = ana + 1e-9
    score = residuals(_curve(sim, ana, metric="scale"))
    assert score["metric"] == "scale"
    assert score["error"] == pytest.approx(1e-9)
    assert np.isfinite(score["error"])


def test_multi_series_scores_across_every_series() -> None:
    ana = np.ones((3, 5))
    sim = ana.copy()
    sim[2, 4] = 1.5
    score = residuals(_curve(sim, ana))
    assert score["n_series"] == 3
    assert score["n_points"] == 15
    assert score["max_rel_err"] == pytest.approx(0.5)


def test_shape_mismatch_is_an_error_not_a_broadcast() -> None:
    """Silent broadcasting would compare a curve against the wrong points."""
    with pytest.raises(ValueError, match="shape mismatch"):
        residuals(_curve(np.ones(4), np.ones(5)))


# -- verdicts ---------------------------------------------------------


def test_verdict_fails_when_the_curve_disagrees() -> None:
    ana = np.exp(-np.arange(20) / 5.0)
    sim = ana * 1.01  # 1% off everywhere
    benchmarks.register(_bench("b", lambda s, a, m: _curve(sim, ana), rel_tol=1e-6))
    _, score = benchmarks.evaluate("b", _Setup(), {}, {})
    assert score["verdict"] == "fail"
    assert score["max_rel_err"] == pytest.approx(0.01, rel=1e-6)


def test_verdict_passes_only_inside_the_declared_tolerance() -> None:
    ana = np.exp(-np.arange(20) / 5.0)
    benchmarks.register(
        _bench("b", lambda s, a, m: _curve(ana * (1 + 1e-9), ana), rel_tol=1e-8)
    )
    _, score = benchmarks.evaluate("b", _Setup(), {}, {})
    assert score["verdict"] == "pass"


# -- gating and degradation -------------------------------------------


def test_unknown_name_notes_and_leaves_the_run_intact() -> None:
    arrays: dict[str, np.ndarray] = {"E_bins": np.ones(3)}
    summary: dict[str, Any] = {"x_qp": 1.0}
    notes = benchmarks.attach("does-not-exist", _Setup(), arrays, summary)
    assert notes and "No analytic benchmark" in notes[0]
    assert "benchmark" not in summary
    assert set(arrays) == {"E_bins"}


def test_wrong_mode_notes_rather_than_scoring_nonsense() -> None:
    benchmarks.register(
        _bench("b", lambda s, a, m: _curve(np.ones(3), np.ones(3)),
               modes=("transient_0d",))
    )
    summary: dict[str, Any] = {}
    notes = benchmarks.attach("b", _Setup("spatial_2d"), {}, summary)
    assert notes and "not to 'spatial_2d'" in notes[0]
    assert "benchmark" not in summary


def test_a_raising_benchmark_does_not_take_the_run_down() -> None:
    def boom(setup: Any, arrays: Any, summary: Any) -> Curve:
        raise RuntimeError("kernel unavailable")

    benchmarks.register(_bench("b", boom))
    summary: dict[str, Any] = {"x_qp": 1.0}
    notes = benchmarks.attach("b", _Setup(), {}, summary)
    assert notes and "kernel unavailable" in notes[0]
    assert summary == {"x_qp": 1.0}


def test_a_failed_check_is_never_reported_as_absent() -> None:
    """The dangerous confusion: "no verdict" must not read as "passed".

    A benchmark that could not run writes NO ``benchmark`` block, and the
    interface shows an error for a case that asked for one. A benchmark that
    ran and disagreed writes a block with verdict "fail". These are different
    states and neither may be mistaken for a pass.
    """
    benchmarks.register(
        _bench("b", lambda s, a, m: _curve(np.ones(3) * 2, np.ones(3)))
    )
    summary: dict[str, Any] = {}
    assert benchmarks.attach("b", _Setup(), {}, summary) == []
    assert summary["benchmark"]["verdict"] == "fail"

    absent: dict[str, Any] = {}
    benchmarks.attach("b", _Setup("spatial_1d"), {}, absent)
    assert "benchmark" not in absent


# -- payload wiring ---------------------------------------------------


def test_attach_stores_the_curve_it_scored() -> None:
    """The figure and the verdict must read the same numbers.

    Drawn from one source and scored from another, a plot and a verdict can
    disagree without either being wrong, which is harder to diagnose than
    having neither.
    """
    ana = np.exp(-np.arange(6) / 2.0)
    sim = ana * 1.000001
    benchmarks.register(_bench("b", lambda s, a, m: _curve(sim, ana), rel_tol=1e-3))
    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {}
    assert benchmarks.attach("b", _Setup(), arrays, summary) == []

    assert arrays["bench_sim"].shape == (1, 6)
    np.testing.assert_array_equal(arrays["bench_sim"][0], sim)
    np.testing.assert_array_equal(arrays["bench_analytic"][0], ana)
    recomputed = np.max(np.abs(arrays["bench_sim"] - arrays["bench_analytic"])
                        / np.abs(arrays["bench_analytic"]))
    assert recomputed == pytest.approx(summary["benchmark"]["max_rel_err"])


def test_the_declared_tier_reaches_the_payload() -> None:
    """A T2 result reuses the kernel it tests; the label must survive."""
    benchmarks.register(
        _bench("b", lambda s, a, m: _curve(np.ones(3), np.ones(3)), tier="T2")
    )
    summary: dict[str, Any] = {}
    benchmarks.attach("b", _Setup(), {}, summary)
    assert summary["benchmark"]["tier"] == "T2"


def test_duplicate_registration_is_rejected() -> None:
    benchmarks.register(_bench("b", lambda s, a, m: _curve(np.ones(3), np.ones(3))))
    with pytest.raises(ValueError, match="Duplicate"):
        benchmarks.register(_bench("b", lambda s, a, m: _curve(np.ones(3), np.ones(3))))


def test_unknown_tier_is_rejected_at_registration() -> None:
    with pytest.raises(ValueError, match="unknown tier"):
        benchmarks.register(
            _bench("b", lambda s, a, m: _curve(np.ones(3), np.ones(3)), tier="T9")
        )
