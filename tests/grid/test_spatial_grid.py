"""Tests for qpsim.grid.spatial_grid."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.grid.spatial_grid import (
    BOUNDARY_KINDS,
    BoundaryAssignmentError,
    BoundaryCondition,
    BoundaryFace,
    EdgeSegment,
    build_laplacian_with_boundaries,
    build_variable_diffusion_laplacian,
    mask_to_index,
    reconstruct_field,
)


def _full_mask_with_four_edges(n: int = 3) -> tuple[np.ndarray, list[EdgeSegment]]:
    """A full n×n mask with all four outer edges as separate EdgeSegments."""
    mask = np.ones((n, n), dtype=bool)
    top = [BoundaryFace(0, j, "up") for j in range(n)]
    bot = [BoundaryFace(n - 1, j, "down") for j in range(n)]
    left = [BoundaryFace(i, 0, "left") for i in range(n)]
    right = [BoundaryFace(i, n - 1, "right") for i in range(n)]
    edges = [
        EdgeSegment("top", 0.0, 0.0, float(n), 0.0, "up", top),
        EdgeSegment("bot", 0.0, float(n), float(n), float(n), "down", bot),
        EdgeSegment("left", 0.0, 0.0, 0.0, float(n), "left", left),
        EdgeSegment("right", float(n), 0.0, float(n), float(n), "right", right),
    ]
    return mask, edges


class TestBoundaryCondition:
    def test_normalized_kind_lowercases_and_strips(self) -> None:
        assert BoundaryCondition(kind=" Dirichlet ").normalized_kind() == "dirichlet"

    def test_validates_known_kinds(self) -> None:
        for kind in BOUNDARY_KINDS:
            bc = BoundaryCondition(kind=kind, value=0.0)
            bc.validate()  # should not raise

    def test_rejects_unknown_kind(self) -> None:
        with pytest.raises(ValueError, match="Unsupported"):
            BoundaryCondition(kind="bogus").validate()

    def test_requires_value_for_nontrivial_kinds(self) -> None:
        for kind in ("dirichlet", "neumann", "robin"):
            with pytest.raises(ValueError, match="requires a numeric value"):
                BoundaryCondition(kind=kind).validate()

    def test_reflective_absorbing_need_no_value(self) -> None:
        BoundaryCondition(kind="reflective").validate()
        BoundaryCondition(kind="absorbing").validate()


class TestReconstructField:
    def test_values_placed_and_nan_elsewhere(self) -> None:
        mask = np.array([[True, False], [False, True]])
        values = np.array([1.5, -2.5])
        field = reconstruct_field(mask, values)
        assert field.shape == mask.shape
        assert field[0, 0] == 1.5
        assert field[1, 1] == -2.5
        assert np.isnan(field[0, 1])
        assert np.isnan(field[1, 0])


class TestMaskToIndex:
    def test_sequential_indices_and_coords(self) -> None:
        mask = np.array([[True, False], [True, True]])
        index_map, coords = mask_to_index(mask)
        assert coords == [(0, 0), (1, 0), (1, 1)]
        np.testing.assert_array_equal(index_map, [[0, -1], [1, 2]])


class TestBuildLaplacian:
    def test_reflective_nullspace(self) -> None:
        # With all-reflective BCs, the Laplacian has a constant nullspace:
        # L · 1 = 0 to roundoff.
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        L, source, _ = build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=1.0)
        ones = np.ones(9)
        Lv = L @ ones
        np.testing.assert_allclose(Lv, 0.0, atol=1e-12)
        np.testing.assert_allclose(source, 0.0)

    def test_shape_and_sizes(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        L, source, index_map = build_laplacian_with_boundaries(
            mask, edges, edge_conditions, dx=1.0
        )
        assert L.shape == (9, 9)
        assert source.shape == (9,)
        assert index_map.shape == (3, 3)

    def test_rejects_missing_bc(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        # Omit the "top" edge BC.
        edge_conditions = {
            e.edge_id: BoundaryCondition(kind="reflective") for e in edges if e.edge_id != "top"
        }
        with pytest.raises(BoundaryAssignmentError, match="Missing"):
            build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=1.0)

    def test_dirichlet_source_contribution(self) -> None:
        # On a 1×1 mask with Dirichlet value 2.0 on all edges, the source
        # should pick up 2·g·inv_dx² contributions from all 4 faces.
        mask = np.ones((1, 1), dtype=bool)
        top = [BoundaryFace(0, 0, "up")]
        bot = [BoundaryFace(0, 0, "down")]
        left = [BoundaryFace(0, 0, "left")]
        right = [BoundaryFace(0, 0, "right")]
        edges = [
            EdgeSegment("top", 0, 0, 1, 0, "up", top),
            EdgeSegment("bot", 0, 1, 1, 1, "down", bot),
            EdgeSegment("left", 0, 0, 0, 1, "left", left),
            EdgeSegment("right", 1, 0, 1, 1, "right", right),
        ]
        g = 2.0
        edge_conditions = {e.edge_id: BoundaryCondition(kind="dirichlet", value=g) for e in edges}
        _, source, _ = build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=1.0)
        # 4 faces × 2·g·inv_dx² (inv_dx² = 1) = 8·g = 16
        assert source[0] == pytest.approx(16.0)

    def test_rejects_non_positive_dx(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="dx must be positive"):
            build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=0.0)


class TestBuildVariableDiffusionLaplacian:
    def test_uniform_D_matches_constant_laplacian_scaled(self) -> None:
        # Uniform D_spatial should produce a Laplacian equal to D × L.
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        D_val = 2.5
        D_spatial = np.full(9, D_val)
        L_var, _ = build_variable_diffusion_laplacian(
            mask, edges, edge_conditions, dx=1.0, D_spatial=D_spatial
        )
        L_const, _, _ = build_laplacian_with_boundaries(
            mask, edges, edge_conditions, dx=1.0
        )
        np.testing.assert_allclose(L_var.toarray(), (D_val * L_const).toarray(), atol=1e-12)

    def test_rejects_empty_mask(self) -> None:
        mask = np.zeros((3, 3), dtype=bool)
        with pytest.raises(ValueError, match="no interior points"):
            build_variable_diffusion_laplacian(
                mask, edges=[], edge_conditions={}, dx=1.0, D_spatial=np.array([])
            )

    def test_rejects_non_2d_mask(self) -> None:
        with pytest.raises(ValueError, match="mask must be 2D"):
            build_variable_diffusion_laplacian(
                np.array([True, True]), edges=[], edge_conditions={}, dx=1.0,
                D_spatial=np.ones(2),
            )

    def test_rejects_non_1d_D_spatial(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="D_spatial must be a 1D array"):
            build_variable_diffusion_laplacian(
                mask, edges, edge_conditions, dx=1.0,
                D_spatial=np.ones((3, 3)),
            )

    def test_rejects_wrong_length_D_spatial(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="does not match interior-point count"):
            build_variable_diffusion_laplacian(
                mask, edges, edge_conditions, dx=1.0,
                D_spatial=np.ones(8),  # should be 9
            )

    def test_rejects_non_positive_dx(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="dx must be positive"):
            build_variable_diffusion_laplacian(
                mask, edges, edge_conditions, dx=0.0, D_spatial=np.ones(9),
            )
