from __future__ import annotations

import numpy as np
import pytest
from qpsim.experiments.prelim_resonators import (
    PRELIM_RESONATORS,
    TARGET_RESONATOR_LENGTHS_UM,
    current_participation,
    current_squared_profile,
    full_resonator_current_integral_um,
)


def test_tex_resonator_lengths_are_encoded() -> None:
    assert TARGET_RESONATOR_LENGTHS_UM == (
        5543.414000000001,
        5393.1990000000005,
        5250.898999999999,
        5116.8189999999995,
        4988.534,
        4866.509,
    )
    assert PRELIM_RESONATORS[0].label == "Top Left"
    assert PRELIM_RESONATORS[-1].frequency_ghz == 5.0 + 6.0 / 7.0


def test_current_profile_is_high_at_shorted_end() -> None:
    length_um = PRELIM_RESONATORS[0].total_length_um
    x_um = np.array([0.0, length_um])
    weights = current_squared_profile(x_um, length_um)
    assert weights[0] == 1.0
    assert weights[1] < 1e-30


def test_short_strip_participation_matches_integral_ratio() -> None:
    """Against a CLOSED FORM, not against the implementation's own pieces.

    This used to restate current_participation's definition with the same two
    functions on both sides, so it held for every possible implementation of
    either: flipping cos^2 to sin^2, or dropping the 1/2 in the full-resonator
    integral, changes both sides identically and the assertion survives. Only
    the magnitude pin below it was doing any work.

    For a quarter-wave resonator with I^2 proportional to cos^2(pi x / 2L),
    the participation of the first `a` microns is the ratio of
    int_0^a cos^2(pi x / 2L) dx = a/2 + (L/(2 pi)) sin(pi a / L)
    to the full L/2, which is an independent expression of the same quantity.
    """
    length_um = PRELIM_RESONATORS[0].total_length_um
    a = 100.0
    x_um = np.linspace(0.0, a, 101)
    participation = current_participation(x_um, length_um)

    partial = 0.5 * a + (length_um / (2.0 * np.pi)) * np.sin(
        np.pi * a / length_um
    )
    expected = partial / (0.5 * length_um)
    assert participation == pytest.approx(expected, rel=1e-4)
    assert 0.035 < participation < 0.037

