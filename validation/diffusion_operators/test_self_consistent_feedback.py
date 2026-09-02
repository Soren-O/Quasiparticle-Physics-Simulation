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


def test_probe_drift_splits_by_readout_law() -> None:
    # Static self-dug well, probe read on its quasiparticle density N_1 f:
    # A1 does not move at all; the legacy C (+1) and A1P (+2) drift *away*
    # from the well; A2 and B (0) show only the finite-packet residual of
    # the well's curvature.
    result = run(**_SMALL)
    ei = result.e_index
    assert abs(result.drift_measured["A1"][ei]) < 1e-8
    assert result.drift_measured["C"][ei] > 0.0
    assert result.drift_measured["A1P"][ei] > 0.0
    assert result.drift_measured["A1P"][ei] > result.drift_measured["C"][ei]
    a1p = abs(float(result.drift_measured["A1P"][ei]))
    for name in ("A2", "B"):
        residual = abs(float(result.drift_measured[name][ei]))
        assert residual < 0.2 * a1p, (name, residual, a1p)


def test_drift_matches_analytic_velocity() -> None:
    result = run(**_SMALL)
    ei = result.e_index
    a1p_analytic = abs(float(result.drift_analytic["A1P"][ei]))
    # A1: exactly null (undressed flux telescopes) -- round-off only.
    assert abs(float(result.drift_analytic["A1"][ei])) < 1e-12 * a1p_analytic
    # A2, B: null at leading order; only the shape term of the well's curvature.
    for name in ("A2", "B"):
        assert abs(float(result.drift_analytic[name][ei])) < 0.1 * a1p_analytic, name
    # The fitted mean over the 4-step window sits below the exact initial
    # rate by the packet-spreading correction (about 10% for C and 15% for
    # the fast-diffusing A1P at the paper's settings).
    for name, tol in (("C", 0.15), ("A1P", 0.25)):
        measured = float(result.drift_measured[name][ei])
        analytic = float(result.drift_analytic[name][ei])
        assert abs(measured - analytic) < tol * abs(analytic), (name, measured, analytic)


def test_dynamic_mode_quantifies_bookkeeping() -> None:
    result = run(dynamic=True, **_SMALL)
    ei = result.e_index
    # C conserves bare f: untouched by gap updates (round-off only).
    assert result.conservation_drift["C"] < 1e-12
    # A1's N_1 f bookkeeping responds to the moving well: finite, small,
    # and reported rather than hidden.
    assert 0.0 < result.conservation_drift["A1"] < 0.05
    # The legacy placement's outward drift of the quasiparticle density
    # survives full dynamics.
    assert result.drift_measured["C"][ei] > 0.0
