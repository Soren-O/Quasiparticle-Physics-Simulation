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
from scipy import sparse


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

    @pytest.mark.parametrize("sparse_format", ["lil", "dok", "coo"])
    def test_accepts_supported_sparse_storage_formats(
        self, sparse_format: str
    ) -> None:
        L, N = _square_laplacian(3)
        converted = L.asformat(sparse_format)
        B, lu = build_cn_operators(
            converted, dt=0.1, diffusion_coefficient=0.5
        )
        np.testing.assert_allclose(lu.solve(B @ np.ones(N)), 1.0, atol=1e-12)

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

    @pytest.mark.parametrize(
        ("parameter", "bad_value"),
        [
            ("dt", np.nan),
            ("dt", np.inf),
            ("dt", complex(0.1, 0.0)),
            ("dt", True),
            ("diffusion_coefficient", np.nan),
            ("diffusion_coefficient", np.inf),
            ("diffusion_coefficient", complex(1.0, 0.0)),
            ("diffusion_coefficient", True),
        ],
    )
    def test_rejects_nonfinite_complex_or_boolean_controls(
        self, parameter: str, bad_value: complex | float
    ) -> None:
        L, _ = _square_laplacian(3)
        values: dict[str, complex | float] = {
            "dt": 0.1,
            "diffusion_coefficient": 1.0,
        }
        values[parameter] = bad_value
        with pytest.raises(ValueError, match=parameter):
            build_cn_operators(L, **values)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("laplacian", "match"),
        [
            (np.eye(2), "sparse"),
            (sparse.csr_matrix(np.ones((2, 3))), "square"),
            (sparse.csr_matrix((0, 0)), "nonempty"),
            (
                sparse.csr_matrix(np.array([[1.0, np.nan], [0.0, 1.0]])),
                "finite",
            ),
            (
                sparse.csr_matrix(np.eye(2, dtype=complex)),
                "real-valued",
            ),
        ],
    )
    def test_rejects_malformed_laplacian(
        self, laplacian: object, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            build_cn_operators(  # type: ignore[arg-type]
                laplacian,
                dt=0.1,
                diffusion_coefficient=1.0,
            )
