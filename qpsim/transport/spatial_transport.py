"""Crank-Nicolson spatial transport on an arbitrary geometry.

Drives the per-energy operators from :mod:`qpsim.transport.spatial_operator`
through a monotonicity-subcycled CN step. Dimension-agnostic: a one-cell-wide
geometry reproduces the 1-D backend, and a single cell has no transport at all.

Two things here differ deliberately from the 1-D backend they generalise.

*The factor cache is bounded.* The 1-D backend keeps one SuperLU factor per
energy bin in an unbounded dict. For NE tridiagonals that is cheap; for NE
sparse LUs of an N-by-N 5-point Laplacian it is not, and it would exhaust
memory long before the physics went wrong.

*The cache key fingerprints the mask.* The active region changes with the gap
profile, and an operator built for one mask is silently wrong for another --
it would carry flux across cells that have no states. A key that omitted the
mask would return that wrong operator without any error.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np

from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.transport.spatial_operator import spatial_diffusion_operator

__all__ = ["EnergyTransportOp", "SpatialTransport"]

# Largest dt * max-exit-rate a single CN step may take. Above this the step
# can overshoot; the driver subcycles instead.
_MAX_CN_EXIT_STEP = 1.0
_MAX_CN_SUBSTEPS = 1_000_000
# Operator sets held at once. Each entry owns NE sparse factorizations, so
# this is the memory knob that keeps 2-D tractable.
_MAX_OPERATOR_SETS = 2

# (B, LU[A], solve-order cell indices, rho_p, n_substeps, per-substep forcing)
EnergyTransportOp = tuple[Any, Any, np.ndarray, np.ndarray, int, np.ndarray]


class SpatialTransport:
    """Per-energy CN transport operators for one geometry, cached."""

    def __init__(
        self,
        mask: np.ndarray,
        device_faces: dict[tuple[int, int, str], BoundaryCondition],
        dx: float,
    ) -> None:
        self.mask = np.asarray(mask, dtype=bool)
        self.device_faces = device_faces
        self.dx = float(dx)
        self._cache: OrderedDict[tuple[Any, ...], list[EnergyTransportOp | None]]
        self._cache = OrderedDict()

    # -- operator construction -------------------------------------------

    def build(
        self,
        flux_weight_grid: np.ndarray,
        density_weight_grid: np.ndarray,
        dt: float,
        *,
        cache_key: tuple[Any, ...] | None = None,
        face_overrides: dict[int, dict[tuple[int, int], float]] | None = None,
    ) -> list[EnergyTransportOp | None]:
        """Operators for every energy bin.

        ``flux_weight_grid`` and ``density_weight_grid`` are ``(NE, Ncells)``
        over the geometry's cells in mask order. A bin whose flux weight is
        zero in all but at most one cell gets ``None``: nothing to diffuse.

        ``cache_key`` should fingerprint everything the arrays derive from.
        Passing ``None`` hashes the arrays themselves, which is correct but
        costs a pass over them -- fine for a few calls, wasteful in a loop.

        ``face_overrides`` maps an energy-bin index to that bin's face
        conductance overrides, which is how a Kupriyanov-Lukichev interface
        enters: the barrier weight is energy dependent, so the affected faces
        differ bin by bin.
        """
        key = self._cache_key(flux_weight_grid, density_weight_grid, dt, cache_key)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        # Evict before building: an entry owns NE factorizations, so holding
        # the old set while assembling the new one doubles peak memory.
        while len(self._cache) >= _MAX_OPERATOR_SETS:
            self._cache.popitem(last=False)

        cell_rows, cell_cols = np.nonzero(self.mask)
        n_energy = int(flux_weight_grid.shape[0])
        ops: list[EnergyTransportOp | None] = []
        for i in range(n_energy):
            ops.append(
                self._build_one(
                    flux_weight_grid[i], density_weight_grid[i],
                    cell_rows, cell_cols, dt,
                    None if face_overrides is None else face_overrides.get(i),
                )
            )
        self._cache[key] = ops
        return ops

    def _build_one(
        self,
        flux_weight: np.ndarray,
        density_weight: np.ndarray,
        cell_rows: np.ndarray,
        cell_cols: np.ndarray,
        dt: float,
        face_overrides: dict[tuple[int, int], float] | None = None,
    ) -> EnergyTransportOp | None:
        active = flux_weight > 0.0
        if int(np.count_nonzero(active)) < 2:
            return None  # nothing to diffuse in this bin

        submask = np.zeros_like(self.mask)
        submask[cell_rows[active], cell_cols[active]] = True

        operator, source = spatial_diffusion_operator(
            submask,
            self.device_faces,
            self.dx,
            flux_weight[active],
            density_weight[active],
            face_overrides,
        )
        exit_rate = float(np.max(-operator.diagonal()))
        stiffness = dt * exit_rate
        if not np.isfinite(stiffness) or stiffness < 0.0:
            raise RuntimeError(
                "Spatial diffusion produced a non-finite or negative stiffness "
                "estimate; inspect the transport coefficients."
            )
        n_substeps = max(1, int(np.ceil(stiffness / _MAX_CN_EXIT_STEP)))
        if n_substeps > _MAX_CN_SUBSTEPS:
            raise RuntimeError(
                "Crank-Nicolson monotonicity would require "
                f"{n_substeps:,} internal substeps for one energy channel "
                f"(dt*max_exit_rate={stiffness:g}). Reduce dt or coarsen the "
                "spatial mesh."
            )
        sub_dt = dt / n_substeps
        b_mat, lu = build_cn_operators(operator, sub_dt, 1.0)
        # Scale the boundary forcing to the substep here, so the stepper does
        # not have to carry dt around and cannot forget the factor.
        return (b_mat, lu, np.flatnonzero(active), density_weight[active],
                n_substeps, sub_dt * source)

    def _cache_key(
        self,
        flux_weight_grid: np.ndarray,
        density_weight_grid: np.ndarray,
        dt: float,
        supplied: tuple[Any, ...] | None,
    ) -> tuple[Any, ...]:
        # The mask is part of the key on purpose: an operator built for one
        # active region carries flux where another has no states, and nothing
        # downstream would notice.
        base: tuple[Any, ...] = (
            self.mask.tobytes(), self.mask.shape, float(self.dx), float(dt),
            len(self.device_faces),
        )
        if supplied is not None:
            return (*base, *supplied)
        return (*base, flux_weight_grid.tobytes(), density_weight_grid.tobytes())

    # -- stepping ---------------------------------------------------------

    def apply(
        self,
        f: np.ndarray,
        ops: list[EnergyTransportOp | None],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """One transport step over ``(NE, Ncells)`` occupations.

        Returns the stepped occupations and a diagnostics dict reporting how
        much conserved density the ``[0, 1]`` clip moved. Exact conservation
        holds only while CN stays inside the representable box; the stiffness
        bound makes a material clip unexpected, so the caller should treat a
        large value as a failure rather than a rounding detail.
        """
        f_new = f.copy()
        signed = 0.0
        absolute = 0.0
        mass_scale = 0.0
        for i, op in enumerate(ops):
            if op is None:
                continue
            b_mat, lu, idx, rho_p, n_substeps, forcing = op
            u = rho_p * f_new[i, idx]
            mass_scale += float(np.sum(np.abs(u)))
            # An inhomogeneous boundary contributes a constant rate over the
            # step and enters the CN right-hand side already scaled to the
            # substep. Reflective and homogeneous-Neumann edges leave it
            # exactly zero, so the branch costs nothing in that case.
            has_forcing = bool(forcing.any())
            for _ in range(n_substeps):
                rhs = b_mat @ u
                if has_forcing:
                    rhs = rhs + forcing
                u_next = lu.solve(rhs)
                f_clipped = np.clip(u_next / rho_p, 0.0, 1.0)
                clip_change = u_next - rho_p * f_clipped
                signed += float(np.sum(clip_change))
                absolute += float(np.sum(np.abs(clip_change)))
                u = rho_p * f_clipped
            f_new[i, idx] = u / rho_p

        diagnostics = {
            "clip_absolute_fraction": (
                absolute / mass_scale if mass_scale > 0.0 else 0.0
            ),
            "clip_signed_fraction": (
                signed / mass_scale if mass_scale > 0.0 else 0.0
            ),
        }
        return f_new, diagnostics
