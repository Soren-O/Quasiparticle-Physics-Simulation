"""Kinetic backends.

- diffusion.py — the homogeneous single-cell solver: dirty-limit diffusion in
  ENERGY (the isotropic Keldysh kinetics with the collision and gap
  equations), no spatial transport; steady-state and transient paths.
- spatial.py — the same collision kernels composed with spatial transport on
  a geometry of any dimensionality: a 1x1 mask is 0-D, a 1xN mask a strip,
  anything else 2-D.

The spatial diffusion OPERATORS live in ``qpsim.transport.diffusion``; the
``diffusion`` here names the energy-space approximation, not a spatial step.
"""

from qpsim.backends.diffusion import DiffusionBackend, DiffusionState
from qpsim.backends.spatial import SpatialBackend, SpatialState

__all__ = [
    "DiffusionBackend",
    "DiffusionState",
    "SpatialBackend",
    "SpatialState",
]
