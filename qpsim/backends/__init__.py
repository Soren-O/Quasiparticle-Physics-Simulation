"""Kinetic backends.

- diffusion.py — homogeneous steady-state and transient collision/gap solver
- spatial.py — the same kinetics on a geometry of any dimensionality
"""

from qpsim.backends.diffusion import DiffusionBackend, DiffusionState

__all__ = [
    "DiffusionBackend",
    "DiffusionState",
]
