"""Device geometries.

A :class:`Geometry` is a 2D boolean mask on a uniform square grid plus the
boundary segments its outward faces form. It is the single spatial description
the engine solves on, and the lower-dimensional cases are configurations of it
rather than separate objects:

* :func:`rectangle` with ``rows=1`` is a 1-D strip -- the 5-point Laplacian
  degenerates to the 3-point chain exactly.
* :func:`rectangle` with ``rows=cols=1`` is 0-D -- the operator is identically
  zero, i.e. collisions with no transport.

Constructors: :func:`rectangle`, :func:`strip`, :func:`from_gds`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qpsim.geometries.gds import (
    discover_gds_layers,
    gds_support_available,
    rasterize_gds_layer,
)
from qpsim.geometries.mask import (
    connected_component_count,
    default_conditions,
    extract_edge_segments,
    rectangle_mask,
    strip_mask,
)
from qpsim.grid.spatial_grid import BoundaryCondition, EdgeSegment

__all__ = [
    "Geometry",
    "connected_component_count",
    "default_conditions",
    "discover_gds_layers",
    "extract_edge_segments",
    "from_gds",
    "gds_support_available",
    "rectangle",
    "rectangle_mask",
    "strip",
    "strip_mask",
]


@dataclass(frozen=True)
class Geometry:
    """A meshed device region.

    ``mask`` is ``(nrow, ncol)`` boolean; ``mesh_size`` is the square cell
    pitch in layout units and is the ``dx`` the spatial operators are built
    with. ``edges`` covers every outward face exactly once, which is what the
    assembler requires.
    """

    name: str
    mask: np.ndarray
    edges: list[EdgeSegment]
    mesh_size: float = 1.0
    bounds: tuple[float, float, float, float] | None = None
    source: str | None = None
    layer: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.mask.shape[0]), int(self.mask.shape[1]))

    @property
    def cell_count(self) -> int:
        """Cells actually solved for -- the size of the state vector."""
        return int(np.count_nonzero(self.mask))

    @property
    def dimensionality(self) -> int:
        """0, 1 or 2, from the mask's extent rather than from a mode flag."""
        rows, cols = self.shape
        if self.cell_count <= 1:
            return 0
        return 1 if (rows == 1 or cols == 1) else 2

    def conditions(
        self, condition: BoundaryCondition | str = "reflective",
    ) -> dict[str, BoundaryCondition]:
        """One condition on every edge, ready to override by ``edge_id``."""
        return default_conditions(self.edges, condition)


def rectangle(rows: int, cols: int, mesh_size: float = 1.0,
              name: str | None = None) -> Geometry:
    """A solid rectangular region. ``rows=1`` gives 1-D, ``1x1`` gives 0-D."""
    mask = rectangle_mask(rows, cols)
    return Geometry(
        name=name or f"rectangle_{rows}x{cols}",
        mask=mask,
        edges=extract_edge_segments(mask),
        mesh_size=float(mesh_size),
        bounds=(0.0, 0.0, float(cols) * mesh_size, float(rows) * mesh_size),
    )


def strip(cells: int, mesh_size: float = 1.0, name: str | None = None) -> Geometry:
    """The 1-D strip: one cell wide, ``cells`` long."""
    return rectangle(1, cells, mesh_size, name or f"strip_{cells}")


def from_gds(
    gds_path: str | Path,
    layer: int,
    mesh_size: float,
    *,
    require_connected: bool = True,
) -> Geometry:
    """Rasterize one layer of a GDSII layout into a solvable geometry.

    ``require_connected`` rejects a layout that rasterizes to more than one
    region. That is usually what you want: a second region is normally a stray
    polygon on the same layer, or a mesh too coarse to keep a narrow neck
    joined -- and a disconnected mask solves happily while silently modelling
    a different device.
    """
    mask, bounds = rasterize_gds_layer(gds_path, layer, mesh_size)
    if require_connected:
        count = connected_component_count(mask)
        if count != 1:
            raise ValueError(
                f"Geometry must be one connected region; the rasterized layer "
                f"has {count}. Either the layer carries more than one shape, or "
                f"mesh_size={mesh_size:g} is too coarse to keep a narrow "
                f"feature joined."
            )
    return Geometry(
        name=f"{Path(gds_path).stem}_L{int(layer)}",
        mask=mask,
        edges=extract_edge_segments(mask),
        mesh_size=float(mesh_size),
        bounds=(bounds[0], bounds[1], bounds[2], bounds[3]),
        source=str(gds_path),
        layer=int(layer),
    )
