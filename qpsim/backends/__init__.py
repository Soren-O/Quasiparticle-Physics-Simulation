"""Tier-specific backends.

Ported/written (Gate 2):
- base.py — Tier enum
- t3_diffusion.py — homogeneous T3 steady-state and transient collision/gap backend
- t3_spatial_1d.py — T3Spatial1DState + T3Spatial1DBackend (1D strip preview)

Planned (New Framework Plan §5):
- t2_kinetic.py — T2 scalar-kinetic (v2)
- t1_two_component.py — T1 (f_L, f_T) (v3)
- reductions.py — tier-to-tier state conversions (T1→T2, T2→T3, …)
- Backend protocol + BackendState union once T2/T1 arrive.
"""

from qpsim.backends.base import Tier
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.backends.t3_spatial_1d import (
    T3Spatial1DBackend,
    T3Spatial1DState,
    T3SpatialFlux1D,
)

__all__ = [
    "T3DiffusionBackend",
    "T3DiffusionState",
    "T3Spatial1DBackend",
    "T3Spatial1DState",
    "T3SpatialFlux1D",
    "Tier",
]
