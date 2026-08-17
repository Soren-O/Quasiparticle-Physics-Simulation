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
        try:
            value = float(self.value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Boundary condition '{kind}' value must be finite and numeric"
            ) from exc
        if not np.isfinite(value):
            raise ValueError(
                f"Boundary condition '{kind}' value must be finite and numeric"
            )
        if kind == "robin" and self.aux_value is not None:
            try:
                aux_value = float(self.aux_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Boundary condition 'robin' aux_value must be finite and numeric"
                ) from exc
            if not np.isfinite(aux_value):
                raise ValueError(
                    "Boundary condition 'robin' aux_value must be finite and numeric"
                )


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


def _validated_boolean_mask(mask: np.ndarray) -> np.ndarray:
    """Return a 2D boolean geometry mask without truthiness coercion."""
    mask_arr = np.asarray(mask)
    if mask_arr.ndim != 2:
        raise ValueError("mask must be 2D.")
    if mask_arr.dtype != np.dtype(bool):
        raise ValueError("mask must be a boolean array.")
    return mask_arr


def reconstruct_field(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Expand a flat interior-point array back to the full 2D grid.

    Cells in the mask take their value from ``values``; cells outside
    the mask are set to ``NaN``. Inverse of "flatten a 2D field to the
    interior-only vector".
    """
    mask = _validated_boolean_mask(mask)
    field = np.full(mask.shape, np.nan, dtype=float)
    field[mask] = values
    return field


def mask_to_index(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return ``(index_map, coords)`` for a 2D boolean mask.

    ``index_map`` has the same shape as ``mask``; interior cells get
    sequential indices ``0..N-1``, exterior cells get ``-1``. ``coords``
    is the corresponding list of ``(row, col)`` tuples in index order.
    """
    mask = _validated_boolean_mask(mask)
    index_map = -np.ones(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    for idx, (row, col) in enumerate(coords):
        index_map[row, col] = idx
    return index_map, [(int(rc[0]), int(rc[1])) for rc in coords]


def edges_from_mask(
    mask: np.ndarray,
    *,
    condition: BoundaryCondition | str = "reflective",
    group: str = "direction",
) -> tuple[list[EdgeSegment], dict[str, BoundaryCondition]]:
    """Declare every outward face of ``mask`` as boundary, in one call.

    Assembly requires that *every* face pointing out of the mask is named in
    an :class:`EdgeSegment` and assigned a condition; an undeclared face is an
    error, and there is deliberately no default. That is right for a device
    whose edges mean different things, but it makes the degenerate cases
    absurdly expensive to state by hand: a 1xN strip needs ``2N+2`` faces and
    a single 0-D cell needs 4, purely to say "nothing leaves".

    ``group`` selects how the faces are bundled into edges:

    * ``"direction"`` (default) -- four edges, ``up``/``down``/``left``/
      ``right``, so a caller can reassign one side afterwards.
    * ``"all"`` -- one edge named ``boundary``.

    ``condition`` applies to every edge produced; pass a
    :class:`BoundaryCondition` for anything needing a value. Reassign entries
    of the returned dict to give individual sides different conditions.

    The geometric fields of each :class:`EdgeSegment` are set to the mask's
    bounding box. They are carried metadata only -- the assembler reads just
    ``edge_id`` and ``faces`` -- but they keep the segments plottable.
    """
    mask = _validated_boolean_mask(mask)
    if not mask.any():
        raise BoundaryAssignmentError("Geometry mask has no interior points.")
    bc = BoundaryCondition(condition) if isinstance(condition, str) else condition
    bc.validate()

    nrow, ncol = mask.shape
    by_dir: dict[str, list[BoundaryFace]] = {d: [] for d in _DIR_OFFSETS}
    for row, col in np.argwhere(mask):
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            r, c = int(row) + dr, int(col) + dc
            outward = (
                r < 0 or r >= nrow or c < 0 or c >= ncol or not bool(mask[r, c])
            )
            if outward:
                by_dir[direction].append(
                    BoundaryFace(int(row), int(col), direction)
                )

    box = (0.0, 0.0, float(ncol), float(nrow))
    if group == "all":
        faces = [f for d in _DIR_OFFSETS for f in by_dir[d]]
        edges = [EdgeSegment("boundary", *box, "up", faces)]
    elif group == "direction":
        # Keep empty sides out: an EdgeSegment with no faces is still a real
        # edge that must be assigned, so emitting them would only add noise.
        edges = [
            EdgeSegment(direction, *box, direction, by_dir[direction])
            for direction in _DIR_OFFSETS
            if by_dir[direction]
        ]
    else:
        raise ValueError(f"group must be 'direction' or 'all'; got {group!r}.")

    return edges, {edge.edge_id: bc for edge in edges}


def _normalized_bc(bc: BoundaryCondition) -> BoundaryCondition:
    return replace(bc, kind=bc.normalized_kind())


def _build_face_bc_lookup(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
) -> dict[tuple[int, int, str], BoundaryCondition]:
    """Validate declared edges and return the boundary condition per face.

    Empty edge segments are valid no-ops, but they are still part of the
    public boundary specification and therefore require an assigned
    condition.  Validating every declared face here keeps the constant- and
    variable-coefficient builders on exactly the same contract: a face must
    belong to an interior cell, point outside the domain, use a supported
    direction, and be assigned at most once.
    """
    missing_edges = [
        edge.edge_id for edge in edges if edge.edge_id not in edge_conditions
    ]
    if missing_edges:
        raise BoundaryAssignmentError(
            "All edges must be assigned boundary conditions before simulation. "
            f"Missing: {len(missing_edges)}"
        )

    lookup: dict[tuple[int, int, str], BoundaryCondition] = {}
    ny, nx = mask.shape
    for edge in edges:
        bc = edge_conditions[edge.edge_id]
        checked = _normalized_bc(bc)
        checked.validate()
        for face in edge.faces:
            if not isinstance(face.direction, str) or face.direction not in _DIR_OFFSETS:
                raise BoundaryAssignmentError(
                    f"Edge '{edge.edge_id}' has unsupported face direction "
                    f"{face.direction!r}."
                )
            if (
                isinstance(face.row, (bool, np.bool_))
                or isinstance(face.col, (bool, np.bool_))
                or not isinstance(face.row, (int, np.integer))
                or not isinstance(face.col, (int, np.integer))
            ):
                raise BoundaryAssignmentError(
                    f"Edge '{edge.edge_id}' has a face with non-integer cell "
                    f"coordinates ({face.row!r}, {face.col!r})."
                )

            row = int(face.row)
            col = int(face.col)
            if not (0 <= row < ny and 0 <= col < nx):
                raise BoundaryAssignmentError(
                    f"Edge '{edge.edge_id}' has a face outside the mask at "
                    f"cell ({row}, {col})."
                )
            if not mask[row, col]:
                raise BoundaryAssignmentError(
                    f"Edge '{edge.edge_id}' has a face at non-interior cell "
                    f"({row}, {col})."
                )

            dr, dc = _DIR_OFFSETS[face.direction]
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                raise BoundaryAssignmentError(
                    f"Edge '{edge.edge_id}' declares the interior face at "
                    f"cell ({row}, {col}) direction '{face.direction}' as a "
                    "boundary."
                )

            key = (row, col, face.direction)
            if key in lookup:
                raise BoundaryAssignmentError(
                    f"Boundary face at cell ({row}, {col}) direction "
                    f"'{face.direction}' is assigned more than once."
                )
            lookup[key] = checked
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
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be positive and finite.")
    mask = _validated_boolean_mask(mask)

    index_map, coords = mask_to_index(mask)
    n = len(coords)
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")

    face_bc = _build_face_bc_lookup(mask, edges, edge_conditions)

    dx_value = float(dx)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inv_dx = 1.0 / dx_value
        # 1/(dx*dx), not (1/dx)**2: one rounding instead of two, and it
        # matches the 1-D backend exactly so a one-cell-wide grid
        # reproduces it bit for bit.
        inv_dx2 = 1.0 / (dx * dx)
    if not np.isfinite(inv_dx) or not np.isfinite(inv_dx2):
        raise ValueError("dx is too small to assemble a finite Laplacian.")
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

    if np.any(~np.isfinite(np.asarray(data))) or np.any(~np.isfinite(source)):
        raise ValueError(
            "Boundary values and dx must assemble a finite Laplacian and source."
        )
    laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if np.any(~np.isfinite(laplacian.data)):
        raise ValueError("dx must assemble a finite Laplacian.")
    return laplacian, source, index_map


def build_variable_diffusion_laplacian(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    dx: float,
    D_spatial: np.ndarray,
    face_composition: str = "harmonic",
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble ``∇ · (D ∇ ·)`` with spatially-varying diffusion coefficient.

    ``face_composition`` selects how the two cell values combine at an
    interior face, and the right answer depends on what ``D`` MEANS:

    * ``"harmonic"`` — ``D_face = 2 D_p D_q / (D_p + D_q)``. Correct when ``D``
      is a genuine continuum diffusivity, because the two cells are then media
      in series across the face.
    * ``"min"`` — ``D_face = min(D_p, D_q)``. Correct when ``D`` is an
      above-gap INDICATOR (the ``q == 0`` dirty-limit member), where each
      sub-energy conducts in parallel and only where it is supported on both
      sides, so the exact bin-averaged face coefficient is the overlap
      measure. The harmonic mean over-weights a cut bin by up to 2x there.

    BC contributions are scaled by the boundary cell's local ``D_p`` and are
    unaffected by this choice.

    ``D_spatial`` must be a 1D array of length ``N`` (the interior-point
    count), indexed the same as the output of :func:`mask_to_index`.
    """
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be positive and finite.")
    mask = _validated_boolean_mask(mask)

    D_raw = np.asarray(D_spatial)
    if np.iscomplexobj(D_raw):
        raise ValueError("D_spatial must be real-valued.")
    D_arr = np.asarray(D_raw, dtype=float)
    if D_arr.ndim != 1:
        raise ValueError("D_spatial must be a 1D array.")
    if np.any(~np.isfinite(D_arr)) or np.any(D_arr < 0.0):
        raise ValueError(
            "D_spatial must be finite and non-negative everywhere; invalid "
            "entries would construct a non-physical diffusion operator."
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

    face_bc = _build_face_bc_lookup(mask, edges, edge_conditions)
    dx_value = float(dx)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        inv_dx = 1.0 / dx_value
        # 1/(dx*dx), not (1/dx)**2: one rounding instead of two, and it
        # matches the 1-D backend exactly so a one-cell-wide grid
        # reproduces it bit for bit.
        inv_dx2 = 1.0 / (dx * dx)
    if not np.isfinite(inv_dx) or not np.isfinite(inv_dx2):
        raise ValueError("dx is too small to assemble a finite diffusion operator.")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    source = np.zeros(n, dtype=float)
    ny, nx = mask.shape

    for p, (row, col) in enumerate(coords):
        D_p = float(D_arr[p])
        for direction, (dr, dc) in _DIR_OFFSETS.items():
            nr, nc = row + dr, col + dc
            if 0 <= nr < ny and 0 <= nc < nx and mask[nr, nc]:
                q = int(index_map[nr, nc])
                D_q = float(D_arr[q])
                lo = min(D_p, D_q)
                hi = max(D_p, D_q)
                if face_composition == "min":
                    # The coefficient is an INDICATOR, not a diffusivity: each
                    # sub-energy in this bin either has states on both sides of
                    # the face or it does not. Sub-energies conduct in
                    # parallel, and each conducts only where it is supported on
                    # both sides, so the exact bin-averaged face coefficient is
                    # the measure of the overlap. A harmonic mean answers the
                    # series-resistance question instead and over-weights the
                    # bin cut by the larger gap by up to 2x.
                    D_face = lo
                else:
                    # Algebraically equal to 2*Dp*Dq/(Dp+Dq), but avoids both
                    # product and sum overflow for large finite diffusivities.
                    D_face = 0.0 if lo == 0.0 else lo * (2.0 / (1.0 + lo / hi))
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

    if np.any(~np.isfinite(np.asarray(data))) or np.any(~np.isfinite(source)):
        raise ValueError(
            "D_spatial, boundary values, and dx must assemble a finite "
            "diffusion operator and source."
        )
    L_D = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    if np.any(~np.isfinite(L_D.data)):
        raise ValueError(
            "D_spatial and dx must assemble a finite diffusion operator."
        )
    return L_D, source
