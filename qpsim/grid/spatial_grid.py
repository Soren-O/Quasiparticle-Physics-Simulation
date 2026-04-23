"""Spatial-grid helpers: boundary-condition dataclasses and Laplacian assembly.

Ported from ``qpsim/numerics/operators.py`` plus the
``BoundaryCondition`` / ``BoundaryFace`` / ``EdgeSegment`` definitions
from the old ``qpsim/models.py``. Those three dataclasses describe the
2D masked-grid topology and its boundary conditions; the Laplacian
builders are co-located here because they are tightly coupled to them.

The Crank-Nicolson stepper in :mod:`qpsim.solvers.crank_nicolson`
takes an assembled Laplacian matrix as an input, not the raw mask.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import sparse

BOUNDARY_KINDS = frozenset({"reflective", "neumann", "dirichlet", "absorbing", "robin"})


class BoundaryAssignmentError(ValueError):
    """Raised when boundary conditions are missing or malformed."""


@dataclass
class BoundaryCondition:
    """Boundary condition for a single face of a spatial cell.

    ``kind`` must be one of :data:`BOUNDARY_KINDS`. Each kind uses
    ``value`` (and optionally ``aux_value``) differently:

    * ``reflective``: no BC contribution to the Laplacian.
    * ``absorbing``: zero-value (absorbing) wall; no ``value`` needed.
    * ``dirichlet``: fixed field value ``value`` at the face.
    * ``neumann``: fixed normal flux ``value`` at the face.
    * ``robin``: mixed condition ``∂ₙ φ + β φ = γ`` with
      ``value = β``, ``aux_value = γ``.
    """

    kind: str
    value: float | None = None
    aux_value: float | None = None

    def normalized_kind(self) -> str:
        return self.kind.strip().lower()

    def validate(self) -> None:
        kind = self.normalized_kind()
        if kind not in BOUNDARY_KINDS:
            raise ValueError(f"Unsupported boundary condition kind: {self.kind}")
        if kind in {"reflective", "absorbing"}:
            return
        if kind in {"neumann", "dirichlet", "robin"} and self.value is None:
            raise ValueError(f"Boundary condition '{kind}' requires a numeric value")


@dataclass
class BoundaryFace:
    """One face of a cell on a boundary edge.

    ``direction`` is one of ``"up"``, ``"down"``, ``"left"``, ``"right"``.
    """

    row: int
    col: int
    direction: str


@dataclass
class EdgeSegment:
    """A named edge with a list of boundary faces.

    The ``edge_id`` is the key in the caller's ``edge_conditions`` dict
    (mapping edge-id → :class:`BoundaryCondition`).
    """

    edge_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    normal: str
    faces: list[BoundaryFace]


_DIR_OFFSETS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def reconstruct_field(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Expand a flat interior-point array back to the full 2D grid.

    Cells in the mask take their value from ``values``; cells outside
    the mask are set to ``NaN``. Inverse of "flatten a 2D field to the
    interior-only vector".
    """
    field = np.full(mask.shape, np.nan, dtype=float)
    field[mask] = values
    return field


def mask_to_index(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return ``(index_map, coords)`` for a 2D boolean mask.

    ``index_map`` has the same shape as ``mask``; interior cells get
    sequential indices ``0..N-1``, exterior cells get ``-1``. ``coords``
    is the corresponding list of ``(row, col)`` tuples in index order.
    """
    index_map = -np.ones(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    for idx, (row, col) in enumerate(coords):
        index_map[row, col] = idx
    return index_map, [(int(rc[0]), int(rc[1])) for rc in coords]


def _normalized_bc(bc: BoundaryCondition) -> BoundaryCondition:
    return replace(bc, kind=bc.normalized_kind())


def _build_face_bc_lookup(
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
) -> dict[tuple[int, int, str], BoundaryCondition]:
    lookup: dict[tuple[int, int, str], BoundaryCondition] = {}
    for edge in edges:
        bc = edge_conditions.get(edge.edge_id)
        if bc is None:
            continue
        checked = _normalized_bc(bc)
        checked.validate()
        for face in edge.faces:
            lookup[(face.row, face.col, face.direction)] = checked
    return lookup


def _apply_boundary_contribution(
    bc: BoundaryCondition,
    row_idx: int,
    inv_dx2: float,
    inv_dx: float,
    rows: list[int],
    cols: list[int],
    data: list[float],
    source: np.ndarray,
) -> None:
    kind = bc.normalized_kind()
    if kind == "reflective":
        return
    if kind == "absorbing":
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-2.0 * inv_dx2)
        return
    if kind == "dirichlet":
        g = float(bc.value or 0.0)
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-2.0 * inv_dx2)
        source[row_idx] += 2.0 * g * inv_dx2
        return
    if kind == "neumann":
        qn = float(bc.value or 0.0)
        source[row_idx] += qn * inv_dx
        return
    if kind == "robin":
        beta = float(bc.value or 0.0)
        gamma = float(bc.aux_value or 0.0)
        rows.append(row_idx)
        cols.append(row_idx)
        data.append(-beta * inv_dx)
        source[row_idx] += gamma * inv_dx
        return
    raise BoundaryAssignmentError(f"Unsupported boundary kind: {bc.kind}")


def build_laplacian_with_boundaries(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    dx: float,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Assemble the 5-point Laplacian on the interior of ``mask`` with BCs.

    Returns ``(laplacian, source, index_map)``:

    * ``laplacian``: sparse ``N × N`` matrix (``N`` = interior count).
    * ``source``: inhomogeneous RHS from Dirichlet/Neumann/Robin BCs.
    * ``index_map``: interior-point index at each ``(row, col)`` of the
      mask shape; ``-1`` outside.

    Uses constant grid spacing ``dx``. For a variable diffusion
    coefficient, see :func:`build_variable_diffusion_laplacian`.
    """
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    index_map, coords = mask_to_index(mask)
    n = len(coords)
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")

    face_bc = _build_face_bc_lookup(edges, edge_conditions)
    missing_edges = [edge.edge_id for edge in edges if edge.edge_id not in edge_conditions]
    if missing_edges:
        raise BoundaryAssignmentError(
            "All edges must be assigned boundary conditions before simulation. "
            f"Missing: {len(missing_edges)}"
        )

    inv_dx = 1.0 / dx
    inv_dx2 = inv_dx * inv_dx
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    source = np.zeros(n, dtype=float)

    ny, nx = mask.shape
    for p, (row, col) in enumerate(coords):
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                q = int(index_map[nr, nc])
                rows.append(p)
                cols.append(p)
                data.append(-inv_dx2)
                rows.append(p)
                cols.append(q)
                data.append(inv_dx2)
            else:
                bc = face_bc.get((row, col, direction))
                if bc is None:
                    raise BoundaryAssignmentError(
                        f"Missing boundary condition for face at cell ({row}, {col}) "
                        f"direction '{direction}'."
                    )
                _apply_boundary_contribution(
                    bc=bc, row_idx=p, inv_dx2=inv_dx2, inv_dx=inv_dx,
                    rows=rows, cols=cols, data=data, source=source,
                )

    laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return laplacian, source, index_map


def build_variable_diffusion_laplacian(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    dx: float,
    D_spatial: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble ``∇ · (D ∇ ·)`` with spatially-varying diffusion coefficient.

    Uses the harmonic mean ``D_face = 2 D_p D_q / (D_p + D_q)`` at each
    interior face to preserve discrete conservation. BC contributions
    are scaled by the boundary-cell's local ``D_p``.

    ``D_spatial`` must be a 1D array of length ``N`` (the interior-point
    count), indexed the same as the output of :func:`mask_to_index`.
    """
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    D_arr = np.asarray(D_spatial, dtype=float)
    if D_arr.ndim != 1:
        raise ValueError("D_spatial must be a 1D array.")
    if np.any(D_arr < 0.0):
        raise ValueError(
            "D_spatial must be non-negative everywhere; negative entries "
            "would construct an anti-diffusive operator."
        )

    index_map, coords = mask_to_index(mask)
    n = len(coords)
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")
    if D_arr.shape[0] != n:
        raise ValueError(
            f"D_spatial length {D_arr.shape[0]} does not match "
            f"interior-point count {n}."
        )

    face_bc = _build_face_bc_lookup(edges, edge_conditions)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dx = 1.0 / dx

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    source = np.zeros(n, dtype=float)
    ny, nx = mask.shape

    for p, (row, col) in enumerate(coords):
        D_p = D_arr[p]
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                q = int(index_map[nr, nc])
                D_q = D_arr[q]
                D_face = 2.0 * D_p * D_q / max(D_p + D_q, 1e-30)
                rows.append(p)
                cols.append(p)
                data.append(-D_face * inv_dx2)
                rows.append(p)
                cols.append(q)
                data.append(D_face * inv_dx2)
            else:
                bc = face_bc.get((row, col, direction))
                if bc is None:
                    raise BoundaryAssignmentError(
                        f"Missing boundary condition for face at cell ({row}, {col}) "
                        f"direction '{direction}'."
                    )
                kind = bc.normalized_kind()
                if kind == "reflective":
                    pass
                elif kind == "absorbing":
                    rows.append(p)
                    cols.append(p)
                    data.append(-2.0 * D_p * inv_dx2)
                elif kind == "dirichlet":
                    g = float(bc.value or 0.0)
                    rows.append(p)
                    cols.append(p)
                    data.append(-2.0 * D_p * inv_dx2)
                    source[p] += 2.0 * D_p * g * inv_dx2
                elif kind == "neumann":
                    qn = float(bc.value or 0.0)
                    source[p] += D_p * qn * inv_dx
                elif kind == "robin":
                    beta = float(bc.value or 0.0)
                    gamma = float(bc.aux_value or 0.0)
                    rows.append(p)
                    cols.append(p)
                    data.append(-D_p * beta * inv_dx)
                    source[p] += D_p * gamma * inv_dx

    L_D = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return L_D, source
