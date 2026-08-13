"""Discretization support.

Ported (Gate 2):
- energy_grid.py — cell-centered energy grid, midpoint-edge widths
- spatial_grid.py — BoundaryCondition/EdgeSegment + Laplacian builders

Planned (New Framework Plan §5):
- angular_grid.py — Gauss–Legendre quadrature for T1/T2
- phonon_grid.py — derived phonon frequency grid
"""

from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.grid.spatial_grid import (
    BOUNDARY_KINDS,
    BoundaryAssignmentError,
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    build_laplacian_with_boundaries,
    build_variable_diffusion_laplacian,
    edges_from_mask,
    mask_to_index,
    reconstruct_field,
)

__all__ = [
    "BOUNDARY_KINDS",
    "BoundaryAssignmentError",
    "BoundaryCondition",
    "BoundaryFace",
    "EdgeSegment",
    "build_energy_grid",
    "build_laplacian_with_boundaries",
    "build_variable_diffusion_laplacian",
    "edges_from_mask",
    "integration_widths_from_centers",
    "mask_to_index",
    "reconstruct_field",
]
