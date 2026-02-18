from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from matplotlib.path import Path as MplPath

from .models import BoundaryFace, EdgeSegment, GeometryData

try:
    from scipy import ndimage as ndi
except Exception:
    ndi = None

try:
    import gdstk  # type: ignore
except Exception:
    gdstk = None


def gds_support_available() -> bool:
    return gdstk is not None


def _iter_top_polygons(gds_path: str | Path) -> Iterable[Any]:
    if gdstk is None:
        raise RuntimeError("gdstk is not installed. Install requirements first.")
    lib = gdstk.read_gds(str(gds_path))
    top_cells = lib.top_level() or list(lib.cells)
    if not top_cells:
        return []
    polygons: list[Any] = []
    for idx, top_cell in enumerate(top_cells):
        cell = top_cell.copy(f"__flattened__{idx}")
        cell.flatten()
        polygons.extend(cell.polygons)
    return polygons


def _polygon_signed_area(points: np.ndarray) -> float:
    if points.shape[0] < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def discover_gds_layers(gds_path: str | Path) -> list[int]:
    layers = sorted({int(poly.layer) for poly in _iter_top_polygons(gds_path)})
    if not layers:
        raise ValueError("No polygons were found in the selected GDS file.")
    return layers


def rasterize_gds_layer(
    gds_path: str | Path,
    layer: int,
    mesh_size: float,
) -> tuple[np.ndarray, list[float]]:
    if mesh_size <= 0:
        raise ValueError("Mesh size must be positive.")

    polys = [np.asarray(poly.points) for poly in _iter_top_polygons(gds_path) if int(poly.layer) == int(layer)]
    if not polys:
        raise ValueError(f"No polygons found on layer {layer}.")

    all_points = np.vstack(polys)
    min_x = float(np.min(all_points[:, 0]))
    max_x = float(np.max(all_points[:, 0]))
    min_y = float(np.min(all_points[:, 1]))
    max_y = float(np.max(all_points[:, 1]))

    # Padding creates an explicit outer boundary layer for edge picking.
    pad = mesh_size
    min_x -= pad
    min_y -= pad
    max_x += pad
    max_y += pad

    nx = max(8, int(np.ceil((max_x - min_x) / mesh_size)))
    ny = max(8, int(np.ceil((max_y - min_y) / mesh_size)))

    x_centers = min_x + (np.arange(nx) + 0.5) * mesh_size
    y_centers = min_y + (np.arange(ny) + 0.5) * mesh_size
    gx, gy = np.meshgrid(x_centers, y_centers)
    query_points = np.column_stack([gx.ravel(), gy.ravel()])

    # Use orientation-aware winding accumulation so opposite-oriented contours
    # can carve cutouts/holes instead of always being unioned as solid fill.
    areas = np.array([_polygon_signed_area(poly) for poly in polys], dtype=float)
    dominant_idx = int(np.argmax(np.abs(areas)))
    dominant_sign = np.sign(areas[dominant_idx]) or 1.0

    winding = np.zeros(query_points.shape[0], dtype=np.int32)
    for poly, area in zip(polys, areas):
        path = MplPath(poly)
        inside = path.contains_points(query_points)
        sign = np.sign(area) or dominant_sign
        weight = 1 if sign == dominant_sign else -1
        winding[inside] += int(weight)

    mask = (winding > 0).reshape((ny, nx))
    if not np.any(mask):
        raise ValueError("Layer rasterization produced an empty geometry mask.")

    return mask, [min_x, min_y, max_x, max_y]


def connected_component_count(mask: np.ndarray) -> int:
    if mask.ndim != 2:
        raise ValueError("Mask must be 2D.")
    if ndi is not None:
        structure = np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=np.int8,
        )
        _, count = ndi.label(mask, structure=structure)
        return int(count)

    # Fallback for environments without scipy.ndimage.
    visited = np.zeros_like(mask, dtype=bool)
    ny, nx = mask.shape
    components = 0

    for row in range(ny):
        for col in range(nx):
            if not mask[row, col] or visited[row, col]:
                continue
            components += 1
            q: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= ny or nc < 0 or nc >= nx:
                        continue
                    if mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
    return components


def extract_edge_segments(mask: np.ndarray) -> list[EdgeSegment]:
    ny, nx = mask.shape
    horizontal_groups: dict[tuple[str, int], list[tuple[int, int, BoundaryFace]]] = defaultdict(list)
    vertical_groups: dict[tuple[str, int], list[tuple[int, int, BoundaryFace]]] = defaultdict(list)

    for row in range(ny):
        for col in range(nx):
            if not mask[row, col]:
                continue
            if row == 0 or not mask[row - 1, col]:
                horizontal_groups[("up", row)].append((col, col + 1, BoundaryFace(row=row, col=col, direction="up")))
            if row == ny - 1 or not mask[row + 1, col]:
                y = row + 1
                horizontal_groups[("down", y)].append((col, col + 1, BoundaryFace(row=row, col=col, direction="down")))
            if col == 0 or not mask[row, col - 1]:
                vertical_groups[("left", col)].append((row, row + 1, BoundaryFace(row=row, col=col, direction="left")))
            if col == nx - 1 or not mask[row, col + 1]:
                x = col + 1
                vertical_groups[("right", x)].append((row, row + 1, BoundaryFace(row=row, col=col, direction="right")))

    segments: list[EdgeSegment] = []
    edge_counter = 0

    def next_edge_id() -> str:
        nonlocal edge_counter
        edge_counter += 1
        return f"edge_{edge_counter:04d}"

    for (normal, y), entries in sorted(horizontal_groups.items(), key=lambda item: (item[0][1], item[0][0])):
        entries.sort(key=lambda item: item[0])
        run_start, run_end, run_faces = entries[0][0], entries[0][1], [entries[0][2]]
        for x0, x1, face in entries[1:]:
            if x0 == run_end:
                run_end = x1
                run_faces.append(face)
            else:
                segments.append(
                    EdgeSegment(
                        edge_id=next_edge_id(),
                        x0=float(run_start),
                        y0=float(y),
                        x1=float(run_end),
                        y1=float(y),
                        normal=normal,
                        faces=run_faces,
                    )
                )
                run_start, run_end, run_faces = x0, x1, [face]
        segments.append(
            EdgeSegment(
                edge_id=next_edge_id(),
                x0=float(run_start),
                y0=float(y),
                x1=float(run_end),
                y1=float(y),
                normal=normal,
                faces=run_faces,
            )
        )

    for (normal, x), entries in sorted(vertical_groups.items(), key=lambda item: (item[0][1], item[0][0])):
        entries.sort(key=lambda item: item[0])
        run_start, run_end, run_faces = entries[0][0], entries[0][1], [entries[0][2]]
        for y0, y1, face in entries[1:]:
            if y0 == run_end:
                run_end = y1
                run_faces.append(face)
            else:
                segments.append(
                    EdgeSegment(
                        edge_id=next_edge_id(),
                        x0=float(x),
                        y0=float(run_start),
                        x1=float(x),
                        y1=float(run_end),
                        normal=normal,
                        faces=run_faces,
                    )
                )
                run_start, run_end, run_faces = y0, y1, [face]
        segments.append(
            EdgeSegment(
                edge_id=next_edge_id(),
                x0=float(x),
                y0=float(run_start),
                x1=float(x),
                y1=float(run_end),
                normal=normal,
                faces=run_faces,
            )
        )

    return segments


def create_intrinsic_geometry(mesh_size: float = 1.0, width: int = 120, height: int = 64) -> GeometryData:
    mask = np.zeros((height, width), dtype=bool)
    pad_y = max(1, min(8, max(1, height // 4)))
    pad_x = max(1, min(8, max(1, width // 4)))
    if height - 2 * pad_y <= 0 or width - 2 * pad_x <= 0:
        mask[:, :] = True
    else:
        mask[pad_y:-pad_y, pad_x:-pad_x] = True
    edges = extract_edge_segments(mask)
    return GeometryData(
        name="IntrinsicRectangle",
        source_path="intrinsic",
        layer=0,
        mesh_size=mesh_size,
        mask=mask.astype(int).tolist(),
        edges=edges,
        bounds=[0.0, 0.0, float(width), float(height)],
    )


def create_geometry_from_gds(gds_path: str | Path, layer: int, mesh_size: float) -> GeometryData:
    mask, bounds = rasterize_gds_layer(gds_path, layer, mesh_size)
    component_count = connected_component_count(mask)
    if component_count != 1:
        raise ValueError(
            f"Geometry must have exactly one connected region. Found {component_count} connected regions."
        )
    edges = extract_edge_segments(mask)
    return GeometryData(
        name=f"{Path(gds_path).stem}_L{layer}",
        source_path=str(gds_path),
        layer=int(layer),
        mesh_size=float(mesh_size),
        mask=mask.astype(int).tolist(),
        edges=edges,
        bounds=bounds,
    )


def point_to_segment_distance(px: float, py: float, edge: EdgeSegment) -> float:
    x1, y1, x2, y2 = edge.x0, edge.y0, edge.x1, edge.y1
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    seg_len_sq = vx * vx + vy * vy
    if seg_len_sq <= 0.0:
        return float(np.hypot(px - x1, py - y1))
    t = (wx * vx + wy * vy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * vx
    proj_y = y1 + t * vy
    return float(np.hypot(px - proj_x, py - proj_y))
