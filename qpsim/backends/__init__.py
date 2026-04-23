"""Tier-specific backends.

Ported/written (Gate 2):
- base.py — Tier enum
- t3_diffusion.py — T3DiffusionState + T3DiffusionBackend (steady-state only)

Planned (New Framework Plan §5):
- t2_kinetic.py — T2 scalar-kinetic (v2)
- t1_two_component.py — T1 (f_L, f_T) (v3)
- reductions.py — tier-to-tier state conversions (T1→T2, T2→T3, …)
- Backend protocol + BackendState union once T2/T1 arrive.
"""

from qpsim.backends.base import Tier
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState

__all__ = ["T3DiffusionBackend", "T3DiffusionState", "Tier"]
