"""Geometry masks and the boundary segments they imply.

A geometry is a 2D boolean mask over a uniform square grid: ``True`` is
material, ``False`` is outside. The spatial core in :mod:`qpsim.grid.spatial_grid`
consumes exactly that, so everything here produces masks and the
:class:`~qpsim.grid.spatial_grid.EdgeSegment` lists that go with them.

The reductions are configurations of the same object rather than special
cases: a one-cell-wide mask is a 1-D strip, and a single cell is 0-D.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from qpsim.grid.spatial_grid import BoundaryCondition, BoundaryFace, EdgeSegment

__all__ = [
    "connected_component_count",
    "default_conditions",
    "extract_edge_segments",
    "rectangle_mask",
    "strip_mask",
]


def rectangle_mask(rows: int, cols: int) -> np.ndarray:
    """A solid ``rows x cols`` mask.

    ``rectangle_mask(1, n)`` is the 1-D strip and ``rectangle_mask(1, 1)`` is
    the 0-D cell; both assemble correctly through the spatial core.
    """
    if rows < 1 or cols < 1:
        raise ValueError(f"rectangle_mask needs positive extents; got {rows}x{cols}.")
    return np.ones((int(rows), int(cols)), dtype=bool)


def strip_mask(cells: int) -> np.ndarray:
    """The 1-D strip: one cell wide, ``cells`` long."""
    return rectangle_mask(1, cells)


def connected_component_count(mask: np.ndarray) -> int:
    """Number of 4-connected components of ``mask``.

    A geometry that is not a single connected region is almost always an
    import mistake -- a stray polygon on the same layer, or a mesh too coarse
    to keep a narrow neck joined -- so callers generally want this to be 1.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("Mask must be 2D.")
    mask_arr = mask_arr.astype(bool, copy=False)

    try:  # SciPy is a hard dependency, but this stays usable without it.
        from scipy import ndimage as ndi
    except ImportError:
        ndi = None

    if ndi is not None:
        # 4-connectivity: a diagonal touch is not a connection, because the
        # transport stencil cannot carry flux across one.
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
        _labels, count = ndi.label(mask_arr, structure=structure)
        return int(count)

    visited = np.zeros_like(mask_arr, dtype=bool)
    nrow, ncol = mask_arr.shape
    components = 0
    for row in range(nrow):
        for col in range(ncol):
            if not mask_arr[row, col] or visited[row, col]:
                continue
            components += 1
            queue: deque[tuple[int, int]] = deque([(row, col)])
            visited[row, col] = True
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    inside = 0 <= nr < nrow and 0 <= nc < ncol
                    if inside and mask_arr[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))
    return components


def extract_edge_segments(mask: np.ndarray) -> list[EdgeSegment]:
    """Group every outward face of ``mask`` into contiguous named runs.

    :func:`qpsim.grid.edges_from_mask` bundles the same faces by direction and
    is the right tool for a uniform boundary. This instead merges each maximal
    straight run of collinear faces into its own segment (``edge_0001``, ...),
    which is what a real device needs: the two ends of a strip, or the inner
    and outer rim of an annulus, become separately addressable so they can
    carry different conditions.

    Coordinates are in cell units, with faces lying on integer grid lines.
    """
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("Mask must be 2D.")
    mask_arr = mask_arr.astype(bool, copy=False)
    nrow, ncol = mask_arr.shape

    horizontal: dict[tuple[str, int], list[tuple[int, int, BoundaryFace]]]
    vertical: dict[tuple[str, int], list[tuple[int, int, BoundaryFace]]]
    horizontal = defaultdict(list)
    vertical = defaultdict(list)

    for row in range(nrow):
        for col in range(ncol):
            if not mask_arr[row, col]:
                continue
            if row == 0 or not mask_arr[row - 1, col]:
                horizontal[("up", row)].append(
                    (col, col + 1, BoundaryFace(row, col, "up"))
                )
            if row == nrow - 1 or not mask_arr[row + 1, col]:
                horizontal[("down", row + 1)].append(
                    (col, col + 1, BoundaryFace(row, col, "down"))
                )
            if col == 0 or not mask_arr[row, col - 1]:
                vertical[("left", col)].append(
                    (row, row + 1, BoundaryFace(row, col, "left"))
                )
            if col == ncol - 1 or not mask_arr[row, col + 1]:
                vertical[("right", col + 1)].append(
                    (row, row + 1, BoundaryFace(row, col, "right"))
                )

    segments: list[EdgeSegment] = []

    def emit(normal: str, fixed: int, start: int, end: int,
             faces: list[BoundaryFace], horizontal_run: bool) -> None:
        edge_id = f"edge_{len(segments) + 1:04d}"
        if horizontal_run:
            coords = (float(start), float(fixed), float(end), float(fixed))
        else:
            coords = (float(fixed), float(start), float(fixed), float(end))
        segments.append(EdgeSegment(edge_id, *coords, normal, faces))

    for groups, horizontal_run in ((horizontal, True), (vertical, False)):
        for (normal, fixed), entries in sorted(
            groups.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            entries.sort(key=lambda item: item[0])
            run_start, run_end = entries[0][0], entries[0][1]
            run_faces = [entries[0][2]]
            for lo, hi, face in entries[1:]:
                if lo == run_end:          # still the same straight run
                    run_end = hi
                    run_faces.append(face)
                else:                      # a gap: close this run, open another
                    emit(normal, fixed, run_start, run_end, run_faces, horizontal_run)
                    run_start, run_end, run_faces = lo, hi, [face]
            emit(normal, fixed, run_start, run_end, run_faces, horizontal_run)

    return segments


def default_conditions(
    edges: list[EdgeSegment],
    condition: BoundaryCondition | str = "reflective",
) -> dict[str, BoundaryCondition]:
    """Assign one condition to every edge, ready to be overridden per id."""
    bc = BoundaryCondition(condition) if isinstance(condition, str) else condition
    bc.validate()
    return {edge.edge_id: bc for edge in edges}
