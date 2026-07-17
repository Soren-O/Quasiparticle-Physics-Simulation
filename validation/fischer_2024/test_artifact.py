"""Unit tests for shared Fischer 2024 artifact semantics."""

from __future__ import annotations

import numpy as np

from validation.fischer_2024._artifact import (
    THERMAL_OCCUPATION_RTOL,
    thermal_occupations_match,
)


def test_thermal_occupation_match_is_ulp_scale_and_shape_strict() -> None:
    np.testing.assert_equal(
        THERMAL_OCCUPATION_RTOL,
        8.0 * np.finfo(np.float64).eps,
    )
    expected = np.geomspace(1.0e-96, 1.0e-10, 810)
    assert thermal_occupations_match(expected, expected)
    rounded_up = np.nextafter(np.nextafter(expected, np.inf), np.inf)
    assert thermal_occupations_match(rounded_up, expected)
    assert thermal_occupations_match(np.nextafter(expected, 0.0), expected)

    drifted = expected.copy()
    drifted[0] *= 1.0 + 32.0 * THERMAL_OCCUPATION_RTOL
    assert not thermal_occupations_match(drifted, expected)
    assert not thermal_occupations_match(expected[:-1], expected)
    assert not thermal_occupations_match(np.array([np.inf]), np.array([np.inf]))

    smallest_subnormal = np.nextafter(0.0, 1.0)
    assert not thermal_occupations_match(
        np.array([smallest_subnormal]),
        np.array([0.0]),
    )
    assert not thermal_occupations_match(
        np.array([0.0]),
        np.array([smallest_subnormal]),
    )
