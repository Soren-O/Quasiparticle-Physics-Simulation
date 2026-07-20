"""Repository-level guards for deterministic numerical CI."""

from __future__ import annotations

from pathlib import Path

import yaml


def _ci_document() -> dict:
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    return yaml.safe_load(workflow.read_text(encoding="utf-8"))


def test_ci_pins_certified_blas_thread_contract() -> None:
    environment = _ci_document()["jobs"]["test"]["env"]

    assert environment == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_ci_runs_the_slow_paper_validation_gate() -> None:
    """The slow step is the ONLY place any Fischer/Marchegiani paper pin
    executes (default addopts deselect ``slow``). If it disappears from
    ci.yml, every paper regression silently stops running while CI stays
    green — this guard makes that removal loud (2026-07-19 audit)."""
    steps = _ci_document()["jobs"]["test"]["steps"]
    slow_runs = [
        step.get("run", "")
        for step in steps
        if "slow" in step.get("run", "")
    ]
    assert any(
        'pytest -m "slow and not manual_slow"' in run for run in slow_runs
    ), "ci.yml no longer runs the slow paper-validation pytest gate."

    fast_runs = [s.get("run", "") for s in steps if s.get("run", "").strip() == "pytest"]
    assert fast_runs, "ci.yml no longer runs the default pytest gate."
