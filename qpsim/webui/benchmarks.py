"""Analytic benchmarks: a closed form per term, checked as a whole curve.

Every switchable term in the kinetic equations has a reduction in which it
acts alone, and most of those reductions have an exact solution. This module
states that solution and checks the solver against it *pointwise over a real
abscissa* — time, energy, position, frequency or temperature — rather than at
a single summary number.

The distinction matters. A scalar check ("x_qp ended at 2.2e-10") can be
satisfied by a solution that is wrong everywhere except the one moment it is
read, and it says nothing about the *rate* at which anything happened. A curve
check pins the whole trajectory, so a kernel that is off by a constant factor,
a term wired to the wrong bins, or a time integrator converging to the right
answer along the wrong path all show up as a visible gap.

Tiers
-----
Each benchmark declares how independent its analytic side really is, because
that sets what a green result is worth:

``T1``
    A closed form written from the physics — an exponential relaxation, a
    two-body ``1/t`` decay, a diffusion eigenmode, the BCS gap curve. The
    analytic side does not consult the engine's kernels, so disagreement
    implicates the engine and agreement is evidence about the physics.
``T2``
    The exact solution of the reduced problem, assembled from the engine's
    *own* kernel arrays — a matrix exponential, or the root of an affine ODE.
    Exact in time, and it validates operator assembly and time integration,
    but it reuses the kernel under test and therefore **cannot detect a wrong
    kernel**. Only used where T1 is genuinely out of reach, and always
    labelled, because a T2 that passes is a weaker claim than it looks.
``T3``
    An independently written quadrature of the same physics.

A benchmark reports its tier in the payload and the interface shows it, so a
reader is never invited to mistake "the integrator reproduces the kernel" for
"the kernel is right".

Tolerances
----------
Every ``rel_tol`` here is a measured discretisation floor, not a number chosen
until the check passed. Each entry records the refinement evidence behind it in
``convergence``, and the interface displays that alongside the verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Values below this fraction of a curve's own peak are treated as numerical
# zero for pointwise relative error: dividing by them reports the noise floor
# of the small end as if it were a physics discrepancy.
_REL_FLOOR = 1e-9


@dataclass(frozen=True)
class Curve:
    """One simulated curve and the analytic curve it is checked against.

    ``y_sim`` and ``y_analytic`` are either ``(n,)`` for a single series or
    ``(k, n)`` for ``k`` series sharing the abscissa ``x`` — several energy
    bins relaxing in time, say, where the per-series rate is the prediction.
    """

    x: np.ndarray
    y_sim: np.ndarray
    y_analytic: np.ndarray
    x_label: str
    y_label: str
    series_labels: tuple[str, ...] = ()
    log_x: bool = False
    log_y: bool = False
    # "pointwise" divides by |analytic| at each point, which is the honest
    # measure for a curve of one sign. "scale" divides by the curve's peak,
    # which is what a curve crossing zero needs — a mode amplitude passing
    # through zero would otherwise report an infinite relative error at a
    # point where nothing is wrong.
    metric: str = "pointwise"
    note: str = ""


@dataclass(frozen=True)
class Benchmark:
    """A named closed form, the case it applies to, and how to check it."""

    name: str
    title: str
    tier: str
    formula_latex: str
    reason: str
    rel_tol: float
    convergence: str
    modes: tuple[str, ...]
    build: Callable[[Any, dict[str, np.ndarray], dict[str, Any]], Curve]
    caveat: str = ""
    # What proves the term under test is actually doing something in this
    # case. A benchmark that would pass with the term switched off measures
    # nothing, and that failure mode has been reached more than once here.
    activity: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


def residuals(curve: Curve) -> dict[str, Any]:
    """Pointwise and scale-relative error between the two curves."""
    sim = np.atleast_2d(np.asarray(curve.y_sim, dtype=float))
    ana = np.atleast_2d(np.asarray(curve.y_analytic, dtype=float))
    if sim.shape != ana.shape:
        raise ValueError(
            f"Benchmark curve shape mismatch: sim {sim.shape} vs analytic {ana.shape}."
        )
    diff = np.abs(sim - ana)
    peak = float(np.max(np.abs(ana))) if ana.size else 0.0

    scale_err = float(np.max(diff) / peak) if peak > 0.0 else float(np.max(diff))

    floor = peak * _REL_FLOOR
    usable = np.abs(ana) > floor
    if np.any(usable):
        point = diff[usable] / np.abs(ana[usable])
        max_point = float(np.max(point))
        rms_point = float(np.sqrt(np.mean(point**2)))
    else:
        max_point = rms_point = scale_err

    chosen = scale_err if curve.metric == "scale" else max_point
    return {
        "max_rel_err": max_point,
        "rms_rel_err": rms_point,
        "scale_rel_err": scale_err,
        "metric": curve.metric,
        "error": chosen,
        "n_points": int(sim.size),
        "n_series": int(sim.shape[0]),
        "n_below_floor": int(np.count_nonzero(~usable)),
    }


_REGISTRY: dict[str, Benchmark] = {}
_REGISTERED = False


def _ensure_registered() -> None:
    """Import the benchmark modules on first use.

    The registry is populated by importing :mod:`qpsim.webui.bench` for its
    side effects, and relying on some *other* module to have done that is the
    fragile arrangement this replaces: the runner is what calls
    :func:`attach`, but it worked only because the server happened to import
    the package first, so any path that built a runner without the server got
    an empty registry and every run degraded to a "no such benchmark" note
    instead of failing. Late-imported here rather than at module scope
    because the bench modules import this one.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True  # set first: a failed import must not retry per call
    from qpsim.webui import bench  # noqa: F401  (imported for registration)


def register(bench: Benchmark) -> Benchmark:
    """Add a benchmark to the registry, rejecting duplicate names."""
    if bench.name in _REGISTRY:
        raise ValueError(f"Duplicate benchmark name {bench.name!r}.")
    if bench.tier not in ("T1", "T2", "T3"):
        raise ValueError(f"Benchmark {bench.name!r} has unknown tier {bench.tier!r}.")
    _REGISTRY[bench.name] = bench
    return bench


def get(name: str) -> Benchmark | None:
    _ensure_registered()
    return _REGISTRY.get(name)


def names() -> list[str]:
    _ensure_registered()
    return sorted(_REGISTRY)


def evaluate(
    name: str,
    setup: Any,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> tuple[Curve, dict[str, Any]] | None:
    """Build a benchmark's curve and score it, or None if it does not apply.

    Returning None rather than raising is deliberate: a run is a legitimate
    result whether or not a closed form happens to cover it, and a benchmark
    that cannot be built must not take the run down with it.
    """
    _ensure_registered()
    bench = _REGISTRY.get(name)
    if bench is None:
        return None
    mode = getattr(setup, "mode", None)
    if bench.modes and mode not in bench.modes:
        return None
    curve = bench.build(setup, arrays, summary)
    score = residuals(curve)
    score["verdict"] = "pass" if score["error"] <= bench.rel_tol else "fail"
    score["rel_tol"] = bench.rel_tol
    score["tier"] = bench.tier
    score["name"] = bench.name
    score["title"] = bench.title
    score["formula_latex"] = bench.formula_latex
    score["reason"] = bench.reason
    score["convergence"] = bench.convergence
    score["caveat"] = bench.caveat
    score["activity"] = bench.activity
    return curve, score


def attach(
    name: str,
    setup: Any,
    arrays: dict[str, np.ndarray],
    summary: dict[str, Any],
) -> list[str]:
    """Evaluate a benchmark and write its curve and verdict into a payload.

    Mutates ``arrays`` and ``summary`` in place and returns any notes. The
    arrays are stored so the comparison figure and the CSV export both read
    the same numbers the verdict was computed from — a plot drawn from one
    source and a verdict from another can disagree without either being wrong,
    which is worse than having neither.
    """
    _ensure_registered()
    bench = _REGISTRY.get(name)
    if bench is None:
        return [
            f"No analytic benchmark named {name!r}; the run is unaffected. "
            f"Known benchmarks: {', '.join(names()) or 'none'}."
        ]
    mode = getattr(setup, "mode", None)
    if bench.modes and mode not in bench.modes:
        return [
            f"Analytic benchmark {name!r} applies to "
            f"{', '.join(bench.modes)}, not to {mode!r}, so it was not checked."
        ]
    try:
        result = evaluate(name, setup, arrays, summary)
    except Exception as exc:
        return [f"Analytic benchmark {name!r} could not be evaluated: {exc}"]
    if result is None:
        return []
    curve, score = result

    arrays["bench_x"] = np.asarray(curve.x, dtype=float)
    arrays["bench_sim"] = np.atleast_2d(np.asarray(curve.y_sim, dtype=float))
    arrays["bench_analytic"] = np.atleast_2d(
        np.asarray(curve.y_analytic, dtype=float)
    )
    summary["benchmark"] = {
        **score,
        "x_label": curve.x_label,
        "y_label": curve.y_label,
        "series_labels": list(curve.series_labels),
        "log_x": curve.log_x,
        "log_y": curve.log_y,
        "note": curve.note,
    }
    return []
