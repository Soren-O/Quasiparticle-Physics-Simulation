"""Tests for qpsim.solvers.crank_nicolson."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.spatial_grid import (
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    build_laplacian_with_boundaries,
)
from qpsim.solvers.crank_nicolson import build_cn_operators


def _square_laplacian(n: int = 5):
    mask = np.ones((n, n), dtype=bool)
    top = [BoundaryFace(0, j, "up") for j in range(n)]
    bot = [BoundaryFace(n - 1, j, "down") for j in range(n)]
    left = [BoundaryFace(i, 0, "left") for i in range(n)]
    right = [BoundaryFace(i, n - 1, "right") for i in range(n)]
    edges = [
        EdgeSegment("top", 0, 0, n, 0, "up", top),
        EdgeSegment("bot", 0, n, n, n, "down", bot),
        EdgeSegment("left", 0, 0, 0, n, "left", left),
        EdgeSegment("right", n, 0, n, n, "right", right),
    ]
    edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
    L, _, _ = build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=1.0)
    return L, n * n


class TestBuildCnOperators:
    def test_shapes(self) -> None:
        L, N = _square_laplacian(4)
        B, lu = build_cn_operators(L, dt=0.1, diffusion_coefficient=0.5)
        assert B.shape == (N, N)
        # SuperLU factor should be callable.
        u = np.ones(N)
        result = lu.solve(B @ u)
        assert result.shape == (N,)

    def test_constant_is_a_fixed_point_on_reflective_domain(self) -> None:
        # ∂_t u = D ∇²u with reflective BCs: u = const must be a
        # fixed point. One CN step on u = 1 returns u = 1.
        L, N = _square_laplacian(3)
        B, lu = build_cn_operators(L, dt=0.5, diffusion_coefficient=1.0)
        u0 = np.ones(N)
        u1 = lu.solve(B @ u0)
        np.testing.assert_allclose(u1, 1.0, atol=1e-12)

    def test_total_conserved_on_reflective_domain(self) -> None:
        # With reflective BCs, ∫ u dΩ is conserved under ∂_t u = D ∇²u.
        # The discrete version: sum(u) is invariant under the CN step.
        rng = np.random.default_rng(0)
        L, N = _square_laplacian(5)
        B, lu = build_cn_operators(L, dt=0.1, diffusion_coefficient=0.7)
        u0 = rng.random(N)
        u1 = u0.copy()
        for _ in range(5):
            u1 = lu.solve(B @ u1)
        assert float(np.sum(u1)) == pytest.approx(float(np.sum(u0)), rel=1e-10)

    def test_rejects_non_positive_dt(self) -> None:
        L, _ = _square_laplacian(3)
        with pytest.raises(ValueError, match="dt must be positive"):
            build_cn_operators(L, dt=0.0, diffusion_coefficient=1.0)

    def test_rejects_negative_D(self) -> None:
        L, _ = _square_laplacian(3)
        with pytest.raises(ValueError, match="diffusion_coefficient"):
            build_cn_operators(L, dt=0.1, diffusion_coefficient=-0.1)
