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


# Tests that read the SHIPPED registry rather than registering their own.
_USES_REAL_REGISTRY = ("TestCatalogueBenchmarkCases",)


@pytest.fixture(autouse=True)
def _clean_registry(request: Any) -> Any:
    """A pristine registry for the framework tests, the real one for the rest.

    Two things have to hold at once. The framework tests register throwaway
    benchmarks and must not see or leave the shipped ten. The catalogue tests
    check the shipped ten against catalogue.json and must see exactly them.
    Emptying the registry for both makes the second group fail for a reason
    that exists only under this fixture -- and, because registration is lazy
    and its already-imported flag is global, clearing it once poisons every
    later test rather than only the one.
    """
    benchmarks._ensure_registered()
    if request.cls is not None and request.cls.__name__ in _USES_REAL_REGISTRY:
        yield
        return
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


# -- prescribed fields through the web schema --------------------------


class TestPrescribedFieldsReachTheRun:
    """The engine can express these; the question is whether the API can.

    The gap that started this work was exactly here: `run_time_dependent`
    accepted any starting state, but nothing in the schema could ask for one,
    so no user could reach it.
    """

    @staticmethod
    def _run(setup):
        import qpsim.webui.execute as execute
        return execute.execute_setup(setup, lambda *a, **k: None, lambda: False)

    @staticmethod
    def _base():
        from qpsim.webui.schemas import Spatial2DSetup
        setup = Spatial2DSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 24
        setup.geometry.rows, setup.geometry.cols = 5, 5
        setup.dt = 1.0
        setup.max_time = 40.0
        setup.snapshot_interval = 10.0
        setup.stop_tol = 0.0
        return setup

    def test_thermal_is_still_the_default(self):
        """Every setup written before this existed must be unaffected."""
        setup = self._base()
        assert setup.initial.kind == "thermal"
        assert setup.drives == []

    def test_a_seeded_excess_decays_from_the_start(self):
        setup = self._base()
        setup.initial.kind = "excess"
        setup.initial.amplitude = 5e-4
        series = self._run(setup).arrays["obs_x_qp_mean"]
        assert series[0] > series[-1], "the seeded excess never relaxed"

    def test_a_pulse_rises_while_on_and_falls_after(self):
        """The canonical measurement, through the schema."""
        from qpsim.webui.schemas import (
            DriveSpec,
            EnergyProfileSpec,
            SpatialProfileSpec,
            TimeProfileSpec,
        )
        setup = self._base()
        setup.max_time = 60.0
        setup.snapshot_interval = 5.0
        setup.drives = [DriveSpec(
            enabled=True, amplitude=3e-5,
            energy=EnergyProfileSpec(kind="monoenergetic", E_0=400.0, width=40.0),
            space=SpatialProfileSpec(kind="uniform"),
            time=TimeProfileSpec(kind="pulse", t_on=10.0, t_off=30.0),
        )]
        payload = self._run(setup)
        t = payload.arrays["snap_t_ns"]
        y = payload.arrays["obs_x_qp_mean"]
        during = y[(t > 10.0) & (t <= 30.0)]
        after = y[t > 30.0]
        assert during[-1] > during[0], "the pulse did not raise the population"
        assert after[-1] < during[-1], "nothing decayed once the drive stopped"

    def test_an_enabled_drive_with_no_amplitude_is_refused(self):
        """The recurring failure here is a default that makes a run inert."""
        from pydantic import ValidationError
        from qpsim.webui.schemas import DriveSpec
        with pytest.raises(ValidationError, match="applies nothing"):
            DriveSpec(enabled=True, amplitude=0.0)

    def test_a_clipped_initial_condition_says_so(self):
        setup = self._base()
        setup.initial.kind = "excess"
        setup.initial.amplitude = 5.0        # far past f = 1
        notes = self._run(setup).notes
        assert any("clipped to [0, 1]" in n for n in notes)

    def test_a_non_separable_expression_drive_runs(self):
        from qpsim.webui.schemas import DriveSpec
        setup = self._base()
        setup.drives = [DriveSpec(
            enabled=True, amplitude=1.0,
            expression=(
                "np.where(E > gap * 1.5, "
                "  params.get('a', 2e-5) * (1.0 + x) * np.exp(-t / 30.0), 0.0)"
            ),
            params={"a": 2e-5},
        )]
        series = self._run(setup).arrays["obs_x_qp_mean"]
        assert series[-1] > series[0], "the expression drive created nothing"

    def test_a_mistyped_expression_is_reported_before_the_solve(self):
        from qpsim.fields.safe_eval import SafeExpressionError
        from qpsim.webui.schemas import DriveSpec
        setup = self._base()
        setup.drives = [DriveSpec(
            enabled=True, amplitude=1.0, expression="np.exp(-Q / 30.0)",
        )]
        with pytest.raises(SafeExpressionError, match="Unknown name 'Q'"):
            self._run(setup)


# -- the catalogue's benchmark cases -----------------------------------


class TestCatalogueBenchmarkCases:
    """Every case that names a benchmark must actually be able to run it.

    These are cheap structural checks, and they are the ones that catch the
    real failure: a case whose overrides do not resolve is reported by the
    interface as "unknown setup fields" and never runs, so the benchmark it
    names is silently never exercised. Found exactly that way -- one case
    flattened a free-form `params` dict into dotted paths whose leaves do not
    exist in the defaults.
    """

    @staticmethod
    def _cases():
        import json
        from pathlib import Path
        path = (
            Path(__file__).resolve().parents[2]
            / "qpsim" / "webui" / "static" / "catalogue.json"
        )
        cat = json.loads(path.read_text(encoding="utf-8"))
        return [
            case
            for category in cat["categories"]
            for item in category["items"]
            for case in item.get("cases", [])
            if case.get("benchmark")
        ]

    def test_every_term_has_a_benchmark_case(self):
        named = {c["benchmark"] for c in self._cases()}
        assert named == set(benchmarks.names()), (
            f"registered but not in the catalogue: "
            f"{set(benchmarks.names()) - named}; "
            f"in the catalogue but not registered: {named - set(benchmarks.names())}"
        )

    def test_each_case_names_a_benchmark_that_applies_to_its_mode(self):
        for case in self._cases():
            bench = benchmarks.get(case["benchmark"])
            assert bench is not None, case["benchmark"]
            assert case["mode"] in bench.modes, (
                f"{case['id']} runs {case['mode']} but "
                f"{case['benchmark']} applies to {bench.modes}"
            )

    def test_every_override_path_resolves_on_that_mode(self):
        """An unresolved path means the case is never run at all."""
        from qpsim.webui.schemas import MODE_CLASSES
        problems = []
        for case in self._cases():
            setup = MODE_CLASSES[case["mode"]]()
            for path in case["overrides"]:
                node = setup
                for key in path.split(".")[:-1]:
                    node = getattr(node, key, None)
                    if node is None:
                        break
                leaf = path.split(".")[-1]
                if node is None or not hasattr(node, leaf):
                    problems.append(f"{case['id']}: {path}")
        assert not problems, f"override paths that do not resolve: {problems}"

    def test_each_case_survives_schema_validation(self):
        """Resolving is not enough -- the values must be legal too."""
        from qpsim.webui.schemas import MODE_CLASSES
        for case in self._cases():
            setup = MODE_CLASSES[case["mode"]]()
            for path, value in case["overrides"].items():
                node = setup
                for key in path.split(".")[:-1]:
                    node = getattr(node, key)
                setattr(node, path.split(".")[-1], value)
            # Re-validating catches a value the field would reject on POST.
            MODE_CLASSES[case["mode"]].model_validate(setup.model_dump())
