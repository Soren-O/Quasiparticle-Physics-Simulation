"""GDSII layout import: polygons on a layer to a geometry mask.

``gdstk`` is an optional dependency. This module imports without it; only the
functions that actually read a file raise, and :func:`gds_support_available`
lets a caller check first rather than catching.

Ported from the archived 2D simulator's ``qpsim/geometry.py``. Copied rather
than imported: this tree does not depend on that one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "discover_gds_layers",
    "gds_support_available",
    "rasterize_gds_layer",
    "rasterize_polygons",
]

try:  # optional
    import gdstk  # type: ignore[import-not-found, unused-ignore]
except ImportError:  # pragma: no cover - exercised by the availability test
    gdstk = None  # type: ignore[assignment, unused-ignore]


def gds_support_available() -> bool:
    """Whether GDS import is usable in this environment."""
    return gdstk is not None


def _require_gdstk() -> None:
    if gdstk is None:
        raise RuntimeError(
            "GDS import needs the optional 'gdstk' package, which is not "
            "installed. Install it with: pip install -e \".[gds]\""
        )


def _iter_top_polygons(gds_path: str | Path) -> list[Any]:
    """Every polygon in the file, with references flattened away."""
    _require_gdstk()
    library: Any = gdstk.read_gds(str(gds_path))
    top_cells = library.top_level() or list(library.cells)
    polygons: list[Any] = []
    for idx, top_cell in enumerate(top_cells):
        # Copy before flattening so the library on disk is never mutated.
        cell = top_cell.copy(f"__flattened__{idx}")
        cell.flatten()
        polygons.extend(cell.polygons)
    return polygons


def _polygon_signed_area(points: np.ndarray) -> float:
    """Shoelace area; the SIGN carries the winding direction."""
    if points.shape[0] < 3:
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def discover_gds_layers(gds_path: str | Path) -> list[int]:
    """Sorted layer numbers that carry polygons in ``gds_path``."""
    layers = sorted({int(poly.layer) for poly in _iter_top_polygons(gds_path)})
    if not layers:
        raise ValueError("No polygons were found in the selected GDS file.")
    return layers


def rasterize_gds_layer(
    gds_path: str | Path,
    layer: int,
    mesh_size: float,
) -> tuple[np.ndarray, list[float]]:
    """Rasterize one layer onto a uniform square grid.

    Returns ``(mask, [min_x, min_y, max_x, max_y])``. ``mesh_size`` is the cell
    pitch in layout units and becomes the ``dx`` the spatial core is built on;
    cells are square, so it sets the resolution in both directions and must be
    fine enough for the narrowest feature that matters.

    Holes are honoured. Contours are accumulated by winding number with the
    sign of their own area, so a counter-oriented contour inside another one
    carves a hole instead of being unioned as solid -- which is how an annulus
    survives import.
    """
    if mesh_size <= 0:
        raise ValueError("Mesh size must be positive.")

    polys = [
        np.asarray(poly.points)
        for poly in _iter_top_polygons(gds_path)
        if int(poly.layer) == int(layer)
    ]
    if not polys:
        raise ValueError(f"No polygons found on layer {layer}.")

    return rasterize_polygons(polys, mesh_size)


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Even-odd ray cast, vectorized over points and looped over edges.

    Hand-rolled rather than taken from matplotlib: matplotlib is only a dev
    extra here, and the geometry layer should need nothing beyond numpy.
    """
    x, y = points[:, 0], points[:, 1]
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    inside = np.zeros(points.shape[0], dtype=bool)
    for ax, ay, bx, by in zip(x1, y1, x2, y2, strict=True):
        if ay == by:            # a horizontal edge crosses no horizontal ray
            continue
        straddles = (ay > y) != (by > y)
        # x of the intersection with the horizontal ray through each point
        crossing = (bx - ax) * (y - ay) / (by - ay) + ax
        inside ^= straddles & (x < crossing)
    return inside


def rasterize_polygons(
    polygons: list[np.ndarray],
    mesh_size: float,
) -> tuple[np.ndarray, list[float]]:
    """Rasterize explicit polygons; the file-free half of the importer.

    Split out so the winding logic is testable without gdstk or a GDS file.
    """
    if mesh_size <= 0:
        raise ValueError("Mesh size must be positive.")
    if not polygons:
        raise ValueError("No polygons to rasterize.")

    all_points = np.vstack(polygons)
    # One cell of padding leaves an explicit ring of exterior cells, so every
    # material cell on the outer rim has a real outward face to attach a
    # boundary condition to.
    pad = mesh_size
    min_x = float(np.min(all_points[:, 0])) - pad
    min_y = float(np.min(all_points[:, 1])) - pad
    max_x = float(np.max(all_points[:, 0])) + pad
    max_y = float(np.max(all_points[:, 1])) + pad

    ncol = max(8, int(np.ceil((max_x - min_x) / mesh_size)))
    nrow = max(8, int(np.ceil((max_y - min_y) / mesh_size)))

    x_centers = min_x + (np.arange(ncol) + 0.5) * mesh_size
    y_centers = min_y + (np.arange(nrow) + 0.5) * mesh_size
    gx, gy = np.meshgrid(x_centers, y_centers)
    query = np.column_stack([gx.ravel(), gy.ravel()])

    areas = np.array([_polygon_signed_area(p) for p in polygons], dtype=float)
    dominant = float(np.sign(areas[int(np.argmax(np.abs(areas)))])) or 1.0

    winding = np.zeros(query.shape[0], dtype=np.int32)
    for poly, area in zip(polygons, areas, strict=True):
        inside = _points_in_polygon(query, poly)
        sign = float(np.sign(area)) or dominant
        winding[inside] += 1 if sign == dominant else -1

    mask = (winding > 0).reshape((nrow, ncol))
    if not mask.any():
        raise ValueError("Layer rasterization produced an empty geometry mask.")
    return mask, [min_x, min_y, max_x, max_y]
