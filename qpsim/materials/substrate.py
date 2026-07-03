"""Substrate descriptor for thin-film superconducting devices.

A substrate is needed to compute the acoustic-mismatch transmission
``η`` that sets ``τ_l ≈ 4 d / (η s)`` (see
``docs/Phonon_Escape_Time.md``). Gate 2 v1 uses a constant-``τ_l``
phonon bath by default, so the acoustic fields are optional.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Substrate:
    """Substrate descriptor (Si, Al₂O₃, etc.).

    Attributes
    ----------
    name
        Common identifier, e.g. ``"Si"``, ``"Al2O3"``, ``"SiN"``.
    density
        Mass density in kg/m³. Optional; needed for acoustic-impedance
        ``η`` calculations in the AMM escape-time model.
    sound_velocity
        Longitudinal sound velocity in m/s. Optional; same role as
        ``density``.
    """

    name: str
    density: float | None = None
    sound_velocity: float | None = None

    def __post_init__(self) -> None:
        # YAML 1.1 loads unsigned-exponent notation ("2.3e3") as a
        # string; coerce so user-supplied substrate YAMLs behave.
        if self.density is not None:
            self.density = float(self.density)
        if self.sound_velocity is not None:
            self.sound_velocity = float(self.sound_velocity)
