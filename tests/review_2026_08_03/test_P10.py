# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Review 2026-08-03, packet P10 — validation-layer contract regressions."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid
from qpsim.materials.database import load_material
from validation.diffusion_operators.self_consistent_feedback import (
    GAP_FIXED_POINT_RTOL,
    GAP_FLOOR_FACTOR,
    _calibrate_heavy_amplitude,
    _raw_suppressed_gap,
    dig_well,
)


def _floor_probe_grid(NE: int = 24) -> tuple[np.ndarray, np.ndarray, float]:
    """The benchmark's own grid and seed shape, at ``NE`` bins."""
    gap0 = float(load_material("Al").Delta_0)
    E, _ = build_energy_grid(
        gap=gap0,
        energy_min_factor=GAP_FLOOR_FACTOR,
        energy_max_factor=4.0,
        num_energy_bins=NE,
    )
    seed_weight = np.exp(-np.maximum(E - gap0, 0.0) / (0.2 * gap0))
    return E, seed_weight, gap0


@pytest.mark.parametrize("NE", [12, 24, 48])
def test_dig_well_rejects_a_gap_below_the_represented_floor(NE: int) -> None:
    # `_suppressed_gap` floors its image at GAP_FLOOR_FACTOR * gap0, and the
    # floor is its own fixed point: measuring the residual on the floored map
    # certified 0.5 * gap0 with residual 0.0 and returned it silently, against
    # a documented 1e-12 *raw*-map gate.  Drive the closure past the floor and
    # require the loud failure the docstring promises.  The amplitude is
    # derived from the calibration ceiling (`run` bounds well_depth < 0.5) so
    # it stays illegal as the grid changes.
    E, seed_weight, gap0 = _floor_probe_grid(NE)
    amplitude = 1.2 * _calibrate_heavy_amplitude(seed_weight, E, gap0, 0.4999)
    f_heavy = (amplitude * seed_weight)[:, None]

    with pytest.raises(RuntimeError, match="left the represented interval"):
        dig_well(f_heavy, E, gap0)


@pytest.mark.parametrize("well_depth", [0.05, 0.20, 0.40, 0.49])
def test_dig_well_returns_an_unfloored_fixed_point(well_depth: float) -> None:
    # The co-located benchmark tests all re-apply the *clamped* map, so they
    # cannot tell a converged well from a floor-pinned one.  Certify against
    # the raw image instead, and pin that the clamp stays inert on the legal
    # envelope.
    E, seed_weight, gap0 = _floor_probe_grid()
    amplitude = _calibrate_heavy_amplitude(seed_weight, E, gap0, well_depth)
    f_heavy = (amplitude * seed_weight)[:, None]

    well = dig_well(f_heavy, E, gap0)
    raw = _raw_suppressed_gap(f_heavy, E, well, gap0)

    assert np.all(well > GAP_FLOOR_FACTOR * gap0)
    assert np.max(np.abs(raw - well)) <= GAP_FIXED_POINT_RTOL * gap0
    assert abs((1.0 - float(well[0]) / gap0) - well_depth) <= 3e-12
