from __future__ import annotations

import numpy as np
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
    length_um = PRELIM_RESONATORS[0].total_length_um
    x_um = np.linspace(0.0, 100.0, 101)
    participation = current_participation(x_um, length_um)
    assert participation == np.trapezoid(
        current_squared_profile(x_um, length_um),
        x_um,
    ) / full_resonator_current_integral_um(length_um)
    assert 0.035 < participation < 0.037

