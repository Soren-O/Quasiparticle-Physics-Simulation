"""Geometry and readout constants for the preliminary-exam experiment.

The resonator lengths are transcribed from:

    /Users/sorenormseth/Documents/Graduate/Presentations/
    Simulation for Experiment Presentation/slides_revised.tex

The coordinate convention used here matches the 1D Al strip simulations:
``x = 0`` is the shorted end of the quarter-wave resonator and increasing
``x`` moves along the Al strip toward the open/coupled end.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qpsim.constants import HBAR_UEV_NS

EXPERIMENT_BATH_TEMPERATURE_K = 0.007

AL_STRIP_LENGTH_UM = 100.0
AL_STRIP_WIDTH_UM = 0.1
AL_STRIP_THICKNESS_UM = 0.1

NB_CPW_GAP_LEFT_UM = 6.0
NB_CPW_CENTER_WIDTH_UM = 10.0
NB_CPW_GAP_RIGHT_UM = 6.0
NB_FILM_THICKNESS_NM_NOMINAL = 80.0

KINETIC_INDUCTANCE_FRACTION = 0.08

TARGET_RESONATOR_FREQUENCIES_GHZ = tuple(5.0 + n / 7.0 for n in range(1, 7))
RESONATOR_LOCATION_LABELS = (
    "Top Left",
    "Top Middle",
    "Top Right",
    "Bottom Left",
    "Bottom Middle",
    "Bottom Right",
)

FIXED_RESONATOR_LENGTH_UM = 3016.194
COUPLING_LENGTH_UM = 240.0
VARIABLE_PIECE_LENGTHS_UM = (457.444, 427.401, 398.941, 372.125, 346.468, 322.063)
TARGET_RESONATOR_LENGTHS_UM = tuple(
    FIXED_RESONATOR_LENGTH_UM + COUPLING_LENGTH_UM + 5.0 * length
    for length in VARIABLE_PIECE_LENGTHS_UM
)

# Values quoted in the TeX presentation for the 7 mK, 6-10-6 um Nb CPW.
NB_CPW_Z0_OHM_TEX = 51.343
NB_CPW_EPS_EFF_TEX = 6.8875
NB_CPW_L_PRIME_H_PER_M_TEX = 4.4946e-7
NB_CPW_C_PRIME_F_PER_M_TEX = 1.7050e-10


@dataclass(frozen=True)
class PrelimResonator:
    index: int
    label: str
    frequency_ghz: float
    variable_piece_length_um: float
    total_length_um: float

    @property
    def probe_energy_uev(self) -> float:
        return probe_energy_uev(self.frequency_ghz)


PRELIM_RESONATORS = tuple(
    PrelimResonator(
        index=index,
        label=label,
        frequency_ghz=frequency_ghz,
        variable_piece_length_um=variable_piece_length_um,
        total_length_um=total_length_um,
    )
    for index, (label, frequency_ghz, variable_piece_length_um, total_length_um) in enumerate(
        zip(
            RESONATOR_LOCATION_LABELS,
            TARGET_RESONATOR_FREQUENCIES_GHZ,
            VARIABLE_PIECE_LENGTHS_UM,
            TARGET_RESONATOR_LENGTHS_UM,
            strict=True,
        ),
        start=1,
    )
)


def probe_energy_uev(frequency_ghz: float) -> float:
    """Convert a microwave probe frequency in GHz to hbar omega in micro-eV."""
    return HBAR_UEV_NS * 2.0 * np.pi * frequency_ghz


def current_squared_profile(
    x_um: np.ndarray,
    resonator_length_um: float,
    *,
    strip_offset_from_short_um: float = 0.0,
) -> np.ndarray:
    """Ideal shorted-quarter-wave current-squared profile on the Al strip.

    Gao's convention is often written as ``sin^2(pi x / 2l)`` with ``x``
    measured from the open end. With ``s`` measured from the shorted end,
    this becomes ``cos^2(pi s / 2l)``.
    """
    if resonator_length_um <= 0.0:
        raise ValueError("resonator_length_um must be positive.")
    s_um = np.asarray(x_um, dtype=float) + strip_offset_from_short_um
    return np.cos(0.5 * np.pi * s_um / resonator_length_um) ** 2


def full_resonator_current_integral_um(resonator_length_um: float) -> float:
    """Integral of the ideal ``cos^2`` short-end current profile over 0..l."""
    if resonator_length_um <= 0.0:
        raise ValueError("resonator_length_um must be positive.")
    return 0.5 * resonator_length_um


def strip_current_integral_um(
    x_um: np.ndarray,
    resonator_length_um: float,
    *,
    strip_offset_from_short_um: float = 0.0,
) -> float:
    weights = current_squared_profile(
        x_um,
        resonator_length_um,
        strip_offset_from_short_um=strip_offset_from_short_um,
    )
    return float(np.trapezoid(weights, np.asarray(x_um, dtype=float)))


def current_participation(
    x_um: np.ndarray,
    resonator_length_um: float,
    *,
    strip_offset_from_short_um: float = 0.0,
) -> float:
    """Fraction of ideal resonator ``I^2`` contained in the simulated strip."""
    return strip_current_integral_um(
        x_um,
        resonator_length_um,
        strip_offset_from_short_um=strip_offset_from_short_um,
    ) / full_resonator_current_integral_um(resonator_length_um)
