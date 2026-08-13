"""Tests for qpsim.geometries."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.geometries import (
    Geometry,
    connected_component_count,
    extract_edge_segments,
    from_gds,
    gds_support_available,
    rectangle,
    strip,
)
from qpsim.geometries.gds import rasterize_polygons
from qpsim.grid import build_laplacian_with_boundaries
from qpsim.grid.spatial_grid import BoundaryCondition


def _assemble(geom: Geometry, conditions=None):
    return build_laplacian_with_boundaries(
        geom.mask,
        geom.edges,
        conditions if conditions is not None else geom.conditions(),
        dx=geom.mesh_size,
    )


def _annulus_mask(n: int = 9, inner: float = 1.8, outer: float = 4.2) -> np.ndarray:
    rows, cols = np.mgrid[0:n, 0:n]
    radius = np.hypot(rows - n // 2, cols - n // 2)
    return (radius <= outer) & (radius >= inner)


class TestDimensionalityFallsOut:
    """0-D and 1-D are configurations of the 2-D object, not separate paths."""

    def test_single_cell_reports_zero_dimensional(self):
        geom = rectangle(1, 1)
        assert geom.dimensionality == 0
        assert geom.cell_count == 1

    def test_one_cell_wide_reports_one_dimensional(self):
        geom = strip(7)
        assert geom.dimensionality == 1
        assert geom.cell_count == 7

    def test_a_genuine_rectangle_reports_two_dimensional(self):
        assert rectangle(4, 5).dimensionality == 2

    def test_zero_dimensional_operator_is_identically_zero(self):
        laplacian, _source, _index = _assemble(rectangle(1, 1))
        assert np.allclose(laplacian.toarray(), 0.0)

    def test_one_dimensional_operator_is_the_three_point_chain(self):
        n = 6
        laplacian, _source, _index = _assemble(strip(n))
        expected = np.zeros((n, n))
        for i in range(n):
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    expected[i, j] = 1.0
                    expected[i, i] -= 1.0
        assert np.array_equal(laplacian.toarray(), expected)

    def test_two_dimensional_keeps_the_five_point_stencil(self):
        laplacian, _s, _i = _assemble(rectangle(3, 3))
        assert np.count_nonzero(laplacian.toarray()[4]) - 1 == 4

    def test_mesh_size_scales_the_operator_quadratically(self):
        coarse, _s, _i = _assemble(strip(5))
        fine, _s2, _i2 = _assemble(strip(5, mesh_size=0.5))
        assert np.allclose(fine.toarray(), coarse.toarray() * 4.0)


class TestEdgeSegments:
    """extract_edge_segments must satisfy the assembler's contract exactly."""

    def test_every_outward_face_declared_exactly_once(self):
        for mask in (
            np.ones((1, 1), dtype=bool),
            np.ones((1, 5), dtype=bool),
            np.ones((4, 6), dtype=bool),
            _annulus_mask(),
        ):
            faces = [f for e in extract_edge_segments(mask) for f in e.faces]
            keys = {(f.row, f.col, f.direction) for f in faces}
            assert len(keys) == len(faces), "a face was declared twice"

            expected = set()
            nrow, ncol = mask.shape
            offsets = {"up": (-1, 0), "down": (1, 0),
                       "left": (0, -1), "right": (0, 1)}
            for row, col in np.argwhere(mask):
                for direction, (dr, dc) in offsets.items():
                    r, c = int(row) + dr, int(col) + dc
                    if not (0 <= r < nrow and 0 <= c < ncol) or not mask[r, c]:
                        expected.add((int(row), int(col), direction))
            assert keys == expected

    def test_runs_are_merged_into_maximal_straight_segments(self):
        # A solid rectangle has exactly four straight boundary runs.
        assert len(extract_edge_segments(np.ones((4, 6), dtype=bool))) == 4

    def test_an_annulus_separates_its_two_rims(self):
        mask = _annulus_mask()
        edges = extract_edge_segments(mask)
        # Far more than four: the inner rim is its own set of runs, which is
        # the point -- the two rims can carry different conditions.
        assert len(edges) > 4
        geom = Geometry("annulus", mask, edges)
        laplacian, _s, _i = _assemble(geom)
        assert laplacian.shape == (int(mask.sum()), int(mask.sum()))
        assert np.allclose(laplacian.toarray().sum(axis=1), 0.0)

    def test_a_dirichlet_rim_breaks_the_conserved_null_mode(self):
        geom = rectangle(4, 4)
        conditions = geom.conditions()
        conditions[geom.edges[0].edge_id] = BoundaryCondition("dirichlet", 0.0)
        laplacian, source, _i = _assemble(geom, conditions)
        assert not np.allclose(laplacian.toarray().sum(axis=1), 0.0)
        assert source.shape == (geom.cell_count,)

    def test_rejects_a_non_2d_mask(self):
        with pytest.raises(ValueError, match="2D"):
            extract_edge_segments(np.ones(4, dtype=bool))


class TestConnectivity:
    def test_solid_region_is_one_component(self):
        assert connected_component_count(np.ones((4, 4), dtype=bool)) == 1

    def test_annulus_is_still_one_component(self):
        assert connected_component_count(_annulus_mask()) == 1

    def test_two_islands_are_two_components(self):
        mask = np.zeros((3, 5), dtype=bool)
        mask[1, 0] = True
        mask[1, 4] = True
        assert connected_component_count(mask) == 2

    def test_diagonal_touch_is_not_a_connection(self):
        # The transport stencil carries no flux across a corner, so counting
        # a diagonal as connected would claim a path the operator lacks.
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True
        mask[1, 1] = True
        assert connected_component_count(mask) == 2


class TestRasterization:
    def test_a_counter_wound_contour_carves_a_hole(self):
        outer = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
        inner = np.array([[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]])
        holed, _bounds = rasterize_polygons([outer, inner], mesh_size=1.0)
        solid, _bounds2 = rasterize_polygons([outer], mesh_size=1.0)
        assert solid[solid.shape[0] // 2, solid.shape[1] // 2]
        assert not holed[holed.shape[0] // 2, holed.shape[1] // 2]
        assert int(solid.sum() - holed.sum()) == 16

    def test_padding_leaves_an_exterior_ring(self):
        square = np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]])
        mask, _bounds = rasterize_polygons([square], mesh_size=1.0)
        # Every border row/column of the raster is outside the shape, so each
        # material cell on the rim has a real outward face to condition.
        assert not mask[0].any() and not mask[-1].any()
        assert not mask[:, 0].any() and not mask[:, -1].any()

    def test_rejects_a_non_positive_mesh(self):
        square = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])
        with pytest.raises(ValueError, match="positive"):
            rasterize_polygons([square], mesh_size=0.0)

    def test_rejects_no_polygons(self):
        with pytest.raises(ValueError, match="No polygons"):
            rasterize_polygons([], mesh_size=1.0)


class TestGdsOptionality:
    def test_availability_is_reportable_without_importing(self):
        assert isinstance(gds_support_available(), bool)

    @pytest.mark.skipif(gds_support_available(), reason="gdstk is installed")
    def test_from_gds_explains_the_missing_dependency(self):
        with pytest.raises(RuntimeError, match="gdstk"):
            from_gds("nonexistent.gds", layer=0, mesh_size=1.0)
