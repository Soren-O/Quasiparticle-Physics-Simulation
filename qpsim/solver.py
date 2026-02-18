from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

from .models import BoundaryCondition, EdgeSegment


class BoundaryAssignmentError(ValueError):
    pass


_DIR_OFFSETS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


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


def _mask_to_index(mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    index_map = -np.ones(mask.shape, dtype=int)
    coords = np.argwhere(mask)
    for idx, (row, col) in enumerate(coords):
        index_map[row, col] = idx
    return index_map, [tuple(map(int, rc)) for rc in coords]


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
    if dx <= 0:
        raise ValueError("dx must be positive.")
    if mask.ndim != 2:
        raise ValueError("mask must be 2D.")

    index_map, coords = _mask_to_index(mask)
    n = len(coords)
    if n == 0:
        raise ValueError("Geometry mask has no interior points.")

    face_bc = _build_face_bc_lookup(edges, edge_conditions)
    missing_edges = [edge.edge_id for edge in edges if edge.edge_id not in edge_conditions]
    if missing_edges:
        raise BoundaryAssignmentError(
            f"All edges must be assigned boundary conditions before simulation. Missing: {len(missing_edges)}"
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
                        f"Missing boundary condition for face at cell ({row}, {col}) direction '{direction}'."
                    )
                _apply_boundary_contribution(
                    bc=bc,
                    row_idx=p,
                    inv_dx2=inv_dx2,
                    inv_dx=inv_dx,
                    rows=rows,
                    cols=cols,
                    data=data,
                    source=source,
                )

    laplacian = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    return laplacian, source, index_map


def reconstruct_field(mask: np.ndarray, values: np.ndarray) -> np.ndarray:
    field = np.full(mask.shape, np.nan, dtype=float)
    field[mask] = values
    return field


def run_2d_crank_nicolson(
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
    initial_field: np.ndarray,
    diffusion_coefficient: float,
    dt: float,
    total_time: float,
    dx: float,
    store_every: int = 1,
) -> tuple[list[float], list[np.ndarray], list[float], list[float]]:
    if dt <= 0 or total_time <= 0:
        raise ValueError("dt and total_time must be positive.")
    if diffusion_coefficient <= 0:
        raise ValueError("Diffusion coefficient must be positive.")
    if store_every <= 0:
        store_every = 1
    if initial_field.shape != mask.shape:
        raise ValueError("Initial field shape must match mask shape.")

    laplacian, source, _ = build_laplacian_with_boundaries(mask, edges, edge_conditions, dx)
    interior_values = initial_field[mask].astype(float)
    n = interior_values.shape[0]
    full_steps = int(np.floor(total_time / dt + 1e-12))
    remainder_dt = float(total_time - full_steps * dt)
    if remainder_dt < 1e-12:
        remainder_dt = 0.0
    total_steps = full_steps + (1 if remainder_dt > 0.0 else 0)

    identity = sparse.eye(n, format="csc")
    alpha = 0.5 * dt * diffusion_coefficient
    a_mat = (identity - alpha * laplacian).tocsc()
    b_mat = (identity + alpha * laplacian).tocsr()
    lu = spla.splu(a_mat)
    final_lu = None
    final_b_mat = None
    if remainder_dt > 0.0:
        alpha_final = 0.5 * remainder_dt * diffusion_coefficient
        final_a = (identity - alpha_final * laplacian).tocsc()
        final_b_mat = (identity + alpha_final * laplacian).tocsr()
        final_lu = spla.splu(final_a)

    times: list[float] = [0.0]
    frames: list[np.ndarray] = [reconstruct_field(mask, interior_values)]
    mass: list[float] = [float(np.sum(interior_values) * dx * dx)]

    current = interior_values
    current_time = 0.0
    for step in range(1, total_steps + 1):
        if step <= full_steps:
            dt_step = dt
            b_step = b_mat
            lu_step = lu
        else:
            dt_step = remainder_dt
            if final_b_mat is None or final_lu is None:
                raise RuntimeError("Internal error: final-step solver matrices were not initialized.")
            b_step = final_b_mat
            lu_step = final_lu
        rhs = b_step @ current + dt_step * diffusion_coefficient * source
        current = lu_step.solve(rhs)
        current_time += dt_step
        if step % store_every == 0 or step == total_steps:
            times.append(float(current_time))
            frame = reconstruct_field(mask, current)
            frames.append(frame)
            mass.append(float(np.sum(current) * dx * dx))

    min_val = float(np.nanmin(np.stack(frames)))
    max_val = float(np.nanmax(np.stack(frames)))
    if abs(max_val - min_val) < 1e-12:
        max_val = min_val + 1e-9
    return times, frames, mass, [min_val, max_val]
