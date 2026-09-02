"""Every analytic benchmark in the catalogue, run in CI and pinned.

Before this the ten recorded pass numbers came from a session-local script
nobody's CI ran, and nine of the ten ``bench/*._build`` functions had no test
that called them. Each case here is run exactly as the interface runs it --
the mode's defaults plus the case's overrides, then the benchmark attached --
and its error is required to sit inside the declared tolerance AND at the
recorded value, so a kernel change that moves an error without breaking the
tolerance is still seen. A ``_build`` that stops being called by a case fails
the structural test at the bottom.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pydantic", reason="webui tests need the qpsim[ui] extra")

from qpsim.webui import benchmarks, verdicts

# The pointwise errors measured on 2026-09-01 (matching the plan's record of
# 2026-08-31 to the digit). A change here is a change in the engine or in the
# benchmark's own derivation, and either is worth a look before re-pinning.
RECORDED_ERROR = {
    "diff-benchmark": 2.7275e-03,
    "scat-benchmark": 3.7003e-04,
    "recomb-benchmark": 6.0627e-06,
    "photsg-benchmark": 4.2124e-08,
    "photpb-benchmark": 6.5277e-06,
    "src-benchmark": 2.8528e-15,
    "psc-benchmark": 3.8994e-05,
    "prc-benchmark": 5.6967e-08,
    "pesc-benchmark": 3.4394e-12,
    "gapeq-benchmark": 3.1528e-04,
}

_BENCH_CASES = [c for c in verdicts.catalogue_cases() if c.benchmark]


@pytest.mark.parametrize("case", _BENCH_CASES, ids=[c.id for c in _BENCH_CASES])
def test_the_benchmark_case_passes_at_its_recorded_error(case: verdicts.CaseRef) -> None:
    row = verdicts.score_case(case, verdicts.run_case(case))
    assert row["status"] == "done", row["detail"]
    assert row["verdict"] == "pass", row["detail"]
    assert row["error"] <= row["rel_tol"]
    assert case.id in RECORDED_ERROR, f"record the error for {case.id}"
    # 2 % on a deterministic discretisation error: BLAS-level jitter is
    # 1e-12 relative, a kernel change is orders of magnitude.
    assert row["error"] == pytest.approx(RECORDED_ERROR[case.id], rel=0.02)


def test_every_registered_build_is_exercised_above() -> None:
    """The structural half: no benchmark module can be registered and
    silently never run."""
    exercised = {c.benchmark for c in _BENCH_CASES}
    assert exercised == set(benchmarks.names())
    assert set(RECORDED_ERROR) == {c.id for c in _BENCH_CASES}


def test_the_recorded_errors_sit_inside_their_tolerances() -> None:
    """Guards the pin itself: a recorded value above tolerance would mean the
    table above was written from a failing run."""
    for case in _BENCH_CASES:
        bench = benchmarks.get(case.benchmark or "")
        assert bench is not None
        assert RECORDED_ERROR[case.id] <= bench.rel_tol, case.id
