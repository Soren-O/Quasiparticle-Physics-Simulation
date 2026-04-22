"""Discretization support.

Ported (Gate 2):
- energy_grid.py — cell-centered energy grid, midpoint-edge widths

Planned (New Framework Plan §5):
- angular_grid.py — Gauss–Legendre quadrature for T1/T2
- spatial_grid.py — 2D masked grid, geometry import
- phonon_grid.py — derived phonon frequency grid
"""

from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers

__all__ = ["build_energy_grid", "integration_widths_from_centers"]
