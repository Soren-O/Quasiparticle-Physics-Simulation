"""Fast invariants for the self-consistent gap-feedback benchmark."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.bcs_quadrature import cell_edges_from_widths

import validation.diffusion_operators.self_consistent_feedback as feedback
from validation.diffusion_operators.self_consistent_feedback import (
    GAP_FIXED_POINT_RTOL,
    GAP_FLOOR_FACTOR,
    _calibrate_heavy_amplitude,
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


def test_grid_and_seed_cover_the_closure_floor() -> None:
    gap0 = float(load_material("Al").Delta_0)
    E, _ = build_energy_grid(
        gap=gap0,
        energy_min_factor=GAP_FLOOR_FACTOR,
        energy_max_factor=4.0,
        num_energy_bins=12,
    )
    dE = integration_widths_from_centers(E)
    first_edge = float(cell_edges_from_widths(E, dE)[0])
    seed_weight = np.exp(-np.maximum(E - gap0, 0.0) / (0.2 * gap0))

    assert first_edge <= GAP_FLOOR_FACTOR * gap0
    assert first_edge == pytest.approx(
        GAP_FLOOR_FACTOR * gap0,
        rel=64.0 * np.finfo(float).eps,
        abs=0.0,
    )
    assert np.all(seed_weight[gap0 > E] == 1.0)
    assert np.all((seed_weight > 0.0) & (seed_weight <= 1.0))


def test_depth_calibration_is_a_converged_raw_map_fixed_point() -> None:
    gap0 = float(load_material("Al").Delta_0)
    for NE in (12, 24):
        E, _ = build_energy_grid(
            gap=gap0,
            energy_min_factor=GAP_FLOOR_FACTOR,
            energy_max_factor=4.0,
            num_energy_bins=NE,
        )
        seed_weight = np.exp(-np.maximum(E - gap0, 0.0) / (0.2 * gap0))
        for depth in (0.05, 0.08, 0.20, 0.40):
            amplitude = _calibrate_heavy_amplitude(
                seed_weight,
                E,
                gap0,
                depth,
            )
            f = (amplitude * seed_weight)[:, None]
            target = np.asarray([gap0 * (1.0 - depth)])
            target_map = _suppressed_gap(f, E, target, gap0)
            well = dig_well(f, E, gap0)
            next_map = _suppressed_gap(f, E, well, gap0)

            assert np.max(np.abs(target_map - target)) / gap0 <= 2e-15
            assert np.max(np.abs(next_map - well)) / gap0 <= GAP_FIXED_POINT_RTOL
            assert abs((1.0 - float(well[0]) / gap0) - depth) <= 3e-12


def test_advertised_ten_percent_well_is_certified_on_default_grid() -> None:
    result = run(well_depth=0.10, n_steps=1, fit_steps=1)
    well = result.gap_initial["A1"]
    seed_weight = np.exp(
        -np.maximum(result.E - result.gap0, 0.0) / (0.2 * result.gap0)
    )
    amplitude = _calibrate_heavy_amplitude(
        seed_weight,
        result.E,
        result.gap0,
        0.10,
    )
    heavy_shape = np.exp(-(((result.x - 40.0) / 10.0) ** 2))
    f_heavy = amplitude * seed_weight[:, None] * heavy_shape[None, :]
    mapped = _suppressed_gap(
        f_heavy,
        result.E,
        well,
        result.gap0,
    )

    assert np.max(np.abs(mapped - well)) / result.gap0 <= GAP_FIXED_POINT_RTOL
    assert 1.0 - float(np.min(well)) / result.gap0 == pytest.approx(
        0.10,
        rel=0.0,
        abs=3e-12,
    )
    for drift in result.drift_analytic.values():
        assert np.all(np.isfinite(drift))
        assert np.max(np.abs(drift)) < 1.0e100


def test_dig_well_fails_loudly_at_iteration_limit() -> None:
    gap0 = float(load_material("Al").Delta_0)
    E, _ = build_energy_grid(
        gap=gap0,
        energy_min_factor=GAP_FLOOR_FACTOR,
        energy_max_factor=4.0,
        num_energy_bins=12,
    )
    seed_weight = np.exp(-np.maximum(E - gap0, 0.0) / (0.2 * gap0))
    amplitude = _calibrate_heavy_amplitude(seed_weight, E, gap0, 0.08)

    with pytest.raises(RuntimeError, match="did not converge after 1 iterations"):
        dig_well((amplitude * seed_weight)[:, None], E, gap0, max_iterations=1)


def test_run_rejects_zero_or_invalid_well_depth() -> None:
    for depth in (0.0, -0.01, 0.5, float("nan")):
        with pytest.raises(ValueError, match="well_depth"):
            run(well_depth=depth, NE=12, NX=11, n_steps=1, fit_steps=1)


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


def test_dynamic_mode_quantifies_bookkeeping(monkeypatch) -> None:
    raw_map_residuals: list[float] = []
    real_dig_well = feedback.dig_well

    def checked_dig_well(*args, **kwargs):
        well = real_dig_well(*args, **kwargs)
        f_heavy, E, gap0 = args[:3]
        mapped = _suppressed_gap(f_heavy, E, well, gap0)
        raw_map_residuals.append(
            float(np.max(np.abs(mapped - well))) / float(gap0)
        )
        return well

    monkeypatch.setattr(feedback, "dig_well", checked_dig_well)
    result = run(dynamic=True, **_SMALL)
    ei = result.e_index
    assert len(raw_map_residuals) > 1
    assert max(raw_map_residuals) <= GAP_FIXED_POINT_RTOL
    # C conserves bare f: untouched by gap updates (round-off only).
    assert result.conservation_drift["C"] < 1e-12
    # A1's N_1 f bookkeeping responds to the moving well: finite, small,
    # and reported rather than hidden.
    assert 0.0 < result.conservation_drift["A1"] < 0.05
    # The self-focusing signature survives full dynamics.
    assert result.drift_measured["C"][ei] < 0.0
