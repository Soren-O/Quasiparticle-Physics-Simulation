"""Repository-level guards for deterministic numerical CI."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_ci_pins_certified_blas_thread_contract() -> None:
    workflow = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yml"
    document = yaml.safe_load(workflow.read_text(encoding="utf-8"))
    environment = document["jobs"]["test"]["env"]

    assert environment == {
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
