"""Repository-level guards for deterministic numerical CI."""

from __future__ import annotations

import ast
from pathlib import Path

import yaml


def _ci_document() -> dict:
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(workflow.read_text(encoding="utf-8"))


def test_ci_pins_certified_blas_thread_contract() -> None:
    environment = _ci_document()["jobs"]["test"]["env"]

    assert environment == {
        "BLIS_NUM_THREADS": "1",
        "MKL_DYNAMIC": "FALSE",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_ci_runs_the_slow_numerical_regression_gate() -> None:
    """The slow step is the only hosted lane for selected Fischer pins.

    Default addopts deselect both ``slow`` and ``paper_validation``; the CI
    marker expression explicitly selects the bounded slow set. If it disappears,
    every slow numerical regression silently stops while CI stays green.
    """
    steps = _ci_document()["jobs"]["test"]["steps"]
    slow_runs = [
        step.get("run", "")
        for step in steps
        if "slow" in step.get("run", "")
    ]
    assert any(
        'pytest -m "slow and not manual_slow"' in run for run in slow_runs
    ), "ci.yml no longer runs the slow numerical-regression pytest gate."

    fast_runs = [s.get("run", "") for s in steps if s.get("run", "").strip() == "pytest"]
    assert fast_runs, "ci.yml no longer runs the default pytest gate."


def test_ci_excludes_multi_hour_full_figure_regenerations() -> None:
    """Keep exact full-figure producers runnable without launching them per PR."""
    root = Path(__file__).resolve().parents[2]
    cases = {
        "Fig. 3 exact 1620-bin regeneration (10,671.777 s)": (
            root / "validation" / "fischer_2023" / "test_fig3_paper.py"
        ),
        "Fig. 7 serial 48-target regeneration (13,292.818 worker-s)": (
            root / "validation" / "fischer_2023" / "test_fig7_paper.py"
        ),
    }
    for label, source_path in cases.items():
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_matches_pinned_baseline"
        )
        markers = {ast.unparse(decorator) for decorator in function.decorator_list}
        assert {"pytest.mark.slow", "pytest.mark.manual_slow"} <= markers, (
            f"{label} must remain manual_slow; PR CI covers reduced/live "
            "probes and strict artifact currentness instead."
        )
