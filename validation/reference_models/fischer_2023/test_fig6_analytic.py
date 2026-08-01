"""Independent tests for the clean-room Fischer 2023 Fig. 6 analytic leg."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from validation.paper_parity import load_digitized_points, score_curve, select_curve
from validation.reference_models.fischer_2023.fig6_analytic import (
    eq35_n_bar,
    eq35_t_star_over_delta,
    fig6_curve,
    thermal_gap_integral,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ORACLE = REPO_ROOT / "validation" / "paper_data" / "fischer_2023" / "fig6" / "oracle.json"


def test_clean_room_module_does_not_import_qpsim() -> None:
    """Importing the reference leg in a fresh interpreter must not load qpsim."""

    code = """
import sys
import validation.reference_models.fischer_2023.fig6_analytic
loaded = sorted(name for name in sys.modules if name == "qpsim" or name.startswith("qpsim."))
if loaded:
    raise SystemExit("clean-room module loaded qpsim: " + ", ".join(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("n_bar", [1.0e4, 1.0e6, 1.0e8])
def test_eq35_inverse_round_trip(n_bar: float) -> None:
    ratio = eq35_t_star_over_delta(n_bar)
    assert eq35_n_bar(ratio) == pytest.approx(n_bar, rel=2.0e-15)


def test_continuum_thermal_integral_pins_paper_temperatures() -> None:
    expected = {
        0.10: 4.6238078830940807e-10,
        0.15: 5.966007429323547e-07,
        0.20: 2.2329296542187093e-05,
    }
    for temperature, target in expected.items():
        assert thermal_gap_integral(temperature) == pytest.approx(target, rel=2.0e-13)


@pytest.mark.parametrize("temperature", [0.10, 0.15, 0.20])
def test_clean_room_curve_matches_digitized_paper_analytic_trace(
    temperature: float,
) -> None:
    """Score the independent equation chain against the authenticated raster."""

    _manifest, points = load_digitized_points(ORACLE)
    x = np.linspace(0.24, 0.42, 1801)
    y = fig6_curve(temperature, x)
    score = score_curve(
        select_curve(points, f"T_bath_{temperature:.2f}K", "paper_analytic"),
        x,
        y,
        uncertainty_normalized_limit=1.0,
    )
    assert score.accepted, score.as_dict()
    assert score.max_uncertainty_normalized_error < 0.4
