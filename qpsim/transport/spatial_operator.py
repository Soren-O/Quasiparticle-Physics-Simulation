"""Per-energy spatial transport operators on an arbitrary geometry.

The operator family is

    L_{p,q}[f] = N_1^{-p} div( D_0 N_1^q  grad f ),

built here through :mod:`qpsim.grid.spatial_grid` so it holds in any
dimension. The reductions are configurations rather than code paths: on a
one-cell-wide mask the 5-point Laplacian is the 3-point chain, and on a single
cell it is identically zero.

The per-energy active set -- the cells with states above the local gap -- is
itself a mask, which is what makes this general. The 1-D backend had to require
that set be a contiguous interval; here it can be any shape, including the
disconnected pockets a 2-D gap profile produces.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from qpsim.grid.spatial_grid import (
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    build_variable_diffusion_laplacian,
)

__all__ = [
    "active_submask_boundary",
    "apply_face_conductances",
    "face_condition_lookup",
    "spatial_diffusion_operator",
]

_DIRECTIONS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def face_condition_lookup(
    edges: list[EdgeSegment],
    conditions: dict[str, BoundaryCondition],
) -> dict[tuple[int, int, str], BoundaryCondition]:
    """Flatten declared edges into a ``(row, col, direction) -> condition`` map."""
    lookup: dict[tuple[int, int, str], BoundaryCondition] = {}
    for edge in edges:
        condition = conditions.get(edge.edge_id)
        if condition is None:
            raise KeyError(f"Edge {edge.edge_id!r} has no boundary condition.")
        for face in edge.faces:
            lookup[(int(face.row), int(face.col), face.direction)] = condition
    return lookup


def active_submask_boundary(
    submask: np.ndarray,
    device_faces: dict[tuple[int, int, str], BoundaryCondition],
) -> tuple[list[EdgeSegment], dict[str, BoundaryCondition]]:
    """Boundary for one energy's active region.

    A face of the active region is one of two things, and they must not be
    confused:

    * a face of the *device* -- it keeps whatever condition the geometry gave
      it, so a Dirichlet rim stays Dirichlet for every energy that reaches it;
    * a face newly exposed because the neighbouring cell has no states at this
      energy -- reflective, since no states means no flux, which is the same
      statement as the flux coefficient vanishing where ``N_1 == 0``.

    Faces are grouped by the condition they carry, so the returned edge list is
    short however ragged the active region is.
    """
    submask = np.asarray(submask, dtype=bool)
    nrow, ncol = submask.shape
    reflective = BoundaryCondition("reflective")

    grouped: dict[int, tuple[BoundaryCondition, list[BoundaryFace]]] = {}
    for row, col in np.argwhere(submask):
        row, col = int(row), int(col)
        for direction, (dr, dc) in _DIRECTIONS.items():
            r, c = row + dr, col + dc
            if 0 <= r < nrow and 0 <= c < ncol and submask[r, c]:
                continue  # interior to the active region
            condition = device_faces.get((row, col, direction), reflective)
            key = id(condition)
            if key not in grouped:
                grouped[key] = (condition, [])
            grouped[key][1].append(BoundaryFace(row, col, direction))

    edges: list[EdgeSegment] = []
    conditions: dict[str, BoundaryCondition] = {}
    box = (0.0, 0.0, float(ncol), float(nrow))
    for n, (condition, faces) in enumerate(grouped.values()):
        edge_id = f"active_{n:03d}"
        edges.append(EdgeSegment(edge_id, *box, "up", faces))
        conditions[edge_id] = condition
    return edges, conditions


def apply_face_conductances(
    laplacian: sparse.csr_matrix,
    overrides: dict[tuple[int, int], float],
) -> sparse.csr_matrix:
    """Replace the bulk conductance on named faces.

    ``overrides`` maps a solve-index pair ``(p, q)``, ``p < q``, to the
    conductance that face should carry. Used for a Kupriyanov-Lukichev
    interface, where a finite barrier conductance stands in for the bulk
    harmonic mean.

    The existing off-diagonal ``L[p, q]`` IS the bulk face conductance, so the
    correction is applied as a delta in the conservative four-entry pattern --
    both off-diagonals and both diagonals -- which keeps every row summing to
    what it summed to before, i.e. the operator stays conservative.
    """
    if not overrides:
        return laplacian
    lil = laplacian.tolil()
    for (p, q), g_new in overrides.items():
        delta = float(g_new) - float(lil[p, q])
        lil[p, q] += delta
        lil[q, p] += delta
        lil[p, p] -= delta
        lil[q, q] -= delta
    return lil.tocsr()


def spatial_diffusion_operator(
    submask: np.ndarray,
    device_faces: dict[tuple[int, int, str], BoundaryCondition],
    dx: float,
    flux_weight_active: np.ndarray,
    density_weight_active: np.ndarray,
    face_conductances: dict[tuple[int, int], float] | None = None,
    face_composition: str = "harmonic",
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble ``L = div(W grad .) / rho_p`` on one energy's active region.

    ``flux_weight_active`` and ``density_weight_active`` are per-active-cell,
    in the row-major order the grid layer indexes by, i.e. ``mask.nonzero()``
    order. Returns ``(operator, source)``; ``source`` is nonzero only where an
    inhomogeneous boundary condition contributes.

    ``face_composition`` selects how two cell values combine at an interior
    face, and which is right depends on what the weight MEANS. For a genuine
    continuum diffusivity (``q != 0``) the cells are media in series and the
    harmonic mean is correct. For the ``q == 0`` dirty-limit member the weight
    is an above-gap indicator: sub-energies conduct in parallel and only where
    supported on BOTH sides, so the exact bin-averaged face coefficient is the
    overlap measure ``min(s_L, s_R)``. Using the harmonic mean there
    over-weights the bin cut by the larger gap by up to 2x -- which this
    operator did unconditionally until the composition became selectable.
    """
    edges, conditions = active_submask_boundary(submask, device_faces)
    laplacian, source = build_variable_diffusion_laplacian(
        submask, edges, conditions, dx=dx, D_spatial=flux_weight_active,
        face_composition=face_composition,
    )
    if face_conductances:
        laplacian = apply_face_conductances(laplacian, face_conductances)
    # Divide through by the conserved-density dressing: the equation solved is
    # d(N_1^p f)/dt = div(W grad f), and the state carried is f.
    operator = (laplacian @ sparse.diags(1.0 / density_weight_active)).tocsr()
    return operator, source
