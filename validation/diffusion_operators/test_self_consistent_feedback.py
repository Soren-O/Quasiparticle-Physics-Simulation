"""Fast invariants for the self-consistent gap-feedback benchmark."""

from __future__ import annotations

import numpy as np

from validation.diffusion_operators.self_consistent_feedback import (
    _suppressed_gap,
    dig_well,
    run,
)

_SMALL = {"NE": 12, "NX": 101, "n_steps": 12, "fit_steps": 4, "well_depth": 0.08}


def test_well_is_self_consistent_and_calibrated() -> None:
    result = run(**_SMALL)
    well = result.gap_initial["A1"]
    # Depth lands near the direct-closure calibration target.
    depth = 1.0 - float(np.min(well)) / result.gap0
    assert abs(depth - result.well_depth_target) < 0.3 * result.well_depth_target
    # Suppression only: the well never exceeds the bulk gap.
    assert np.all(well <= result.gap0 + 1e-12)
    # All models start in the same well.
    for name, profile in result.gap_initial.items():
        assert np.allclose(profile, well), name


def test_dig_well_reaches_fixed_point() -> None:
    result = run(**_SMALL)
    E, x, gap0 = result.E, result.x, result.gap0
    f = 0.02 * np.exp(-(E[:, None] - gap0) / (0.2 * gap0)) * np.exp(
        -(((x - 40.0) / 10.0) ** 2)
    )[None, :]
    well = dig_well(f, E, gap0)
    once_more = _suppressed_gap(f, E, well, gap0)
    assert float(np.max(np.abs(once_more - well))) < 1e-10 * gap0


def test_probe_drift_signs_split_by_q() -> None:
    # Static self-dug well: A1 (q = 0) does not move at all; C/B (q < 0)
    # fall toward the well; A1P/A2 (q = 2) are expelled from it.
    result = run(**_SMALL)
    ei = result.e_index
    assert abs(result.drift_measured["A1"][ei]) < 1e-8
    assert result.drift_measured["C"][ei] < 0.0
    assert result.drift_measured["B"][ei] < 0.0
    assert result.drift_measured["A1P"][ei] > 0.0
    assert result.drift_measured["A2"][ei] > 0.0


def test_drift_matches_analytic_velocity() -> None:
    result = run(**_SMALL)
    ei = result.e_index
    assert result.drift_analytic["A1"][ei] == 0.0
    for name in ("C", "B"):
        measured = float(result.drift_measured[name][ei])
        analytic = float(result.drift_analytic[name][ei])
        assert abs(measured - analytic) < 0.25 * abs(analytic), (
            name, measured, analytic,
        )


def test_dynamic_mode_quantifies_bookkeeping() -> None:
    result = run(dynamic=True, **_SMALL)
    ei = result.e_index
    # C conserves bare f: untouched by gap updates (round-off only).
    assert result.conservation_drift["C"] < 1e-12
    # A1's N_1 f bookkeeping responds to the moving well: finite, small,
    # and reported rather than hidden.
    assert 0.0 < result.conservation_drift["A1"] < 0.05
    # The self-focusing signature survives full dynamics.
    assert result.drift_measured["C"][ei] < 0.0
