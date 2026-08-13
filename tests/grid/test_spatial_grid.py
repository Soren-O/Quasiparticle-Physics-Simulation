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
    edges_from_mask,
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


def _assemble_laplacian(
    *,
    variable: bool,
    mask: np.ndarray,
    edges: list[EdgeSegment],
    edge_conditions: dict[str, BoundaryCondition],
) -> None:
    if variable:
        build_variable_diffusion_laplacian(
            mask,
            edges,
            edge_conditions,
            dx=1.0,
            D_spatial=np.ones(int(np.count_nonzero(mask))),
        )
    else:
        build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=1.0)


@pytest.mark.parametrize("variable", [False, True])
@pytest.mark.parametrize(
    "bad_mask",
    [
        pytest.param(np.ones((1, 1), dtype=int), id="integer"),
        pytest.param(np.array([[np.nan]]), id="nan-float"),
    ],
)
def test_builders_reject_nonboolean_geometry_masks(
    variable: bool,
    bad_mask: np.ndarray,
) -> None:
    _mask, edges = _full_mask_with_four_edges(1)
    edge_conditions = {
        edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
    }
    with pytest.raises(ValueError, match="boolean"):
        _assemble_laplacian(
            variable=variable,
            mask=bad_mask,
            edges=edges,
            edge_conditions=edge_conditions,
        )


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

    @pytest.mark.parametrize("kind", ["dirichlet", "neumann", "robin"])
    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "bad"])
    def test_rejects_nonfinite_or_nonnumeric_active_value(
        self,
        kind: str,
        bad_value: object,
    ) -> None:
        with pytest.raises(ValueError, match="finite and numeric"):
            BoundaryCondition(kind=kind, value=bad_value).validate()  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, "bad"])
    def test_rejects_nonfinite_or_nonnumeric_robin_aux_value(
        self,
        bad_value: object,
    ) -> None:
        with pytest.raises(ValueError, match="aux_value must be finite and numeric"):
            BoundaryCondition(
                kind="robin",
                value=1.0,
                aux_value=bad_value,  # type: ignore[arg-type]
            ).validate()


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


class TestSharedEdgeValidation:
    @pytest.mark.parametrize("variable", [False, True])
    def test_empty_mask_keeps_geometry_error_precedence(self, variable: bool) -> None:
        mask = np.zeros((2, 2), dtype=bool)
        edges = [EdgeSegment("unassigned-empty", 0, 0, 0, 0, "up", [])]

        with pytest.raises(ValueError, match="no interior points"):
            _assemble_laplacian(
                variable=variable,
                mask=mask,
                edges=edges,
                edge_conditions={},
            )

    @pytest.mark.parametrize("variable", [False, True])
    def test_unassigned_empty_segment_is_rejected(self, variable: bool) -> None:
        """An empty segment is a no-op, not an exemption from BC assignment."""
        mask, edges = _full_mask_with_four_edges(2)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }
        edges.append(EdgeSegment("empty", 0, 0, 0, 0, "up", []))

        with pytest.raises(BoundaryAssignmentError, match="Missing: 1"):
            _assemble_laplacian(
                variable=variable,
                mask=mask,
                edges=edges,
                edge_conditions=edge_conditions,
            )

    @pytest.mark.parametrize("variable", [False, True])
    def test_assigned_empty_segment_remains_a_valid_no_op(self, variable: bool) -> None:
        mask, edges = _full_mask_with_four_edges(2)
        empty = EdgeSegment("empty", 0, 0, 0, 0, "up", [])
        edges.append(empty)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }

        _assemble_laplacian(
            variable=variable,
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
        )

    @pytest.mark.parametrize("variable", [False, True])
    @pytest.mark.parametrize(
        ("bad_face", "message"),
        [
            (BoundaryFace(0, 0, "diagonal"), "unsupported face direction"),
            (BoundaryFace(-1, 0, "up"), "outside the mask"),
            (BoundaryFace(0, 0, "right"), "declares the interior face"),
            (BoundaryFace(0, 0, "up"), "assigned more than once"),
        ],
    )
    def test_malformed_declared_face_is_rejected_by_both_builders(
        self,
        variable: bool,
        bad_face: BoundaryFace,
        message: str,
    ) -> None:
        mask, edges = _full_mask_with_four_edges(2)
        edges.append(EdgeSegment("bad", 0, 0, 0, 0, "up", [bad_face]))
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }

        with pytest.raises(BoundaryAssignmentError, match=message):
            _assemble_laplacian(
                variable=variable,
                mask=mask,
                edges=edges,
                edge_conditions=edge_conditions,
            )


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

    @pytest.mark.parametrize("bad_dx", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_nonfinite_dx(self, bad_dx: float) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="dx must be positive"):
            build_laplacian_with_boundaries(mask, edges, edge_conditions, dx=bad_dx)

    def test_rejects_finite_inputs_whose_scaled_source_overflows(self) -> None:
        mask, edges = _full_mask_with_four_edges(1)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(
                kind="dirichlet",
                value=np.finfo(float).max,
            )
            for edge in edges
        }
        with pytest.raises(ValueError, match="finite Laplacian and source"):
            build_laplacian_with_boundaries(
                mask,
                edges,
                edge_conditions,
                dx=1.0,
            )


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

    @pytest.mark.parametrize("bad_dx", [0.0, -1.0, np.nan, np.inf, -np.inf])
    def test_rejects_non_positive_or_nonfinite_dx(self, bad_dx: float) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        with pytest.raises(ValueError, match="dx must be positive"):
            build_variable_diffusion_laplacian(
                mask, edges, edge_conditions, dx=bad_dx, D_spatial=np.ones(9),
            )

    def test_rejects_negative_D_spatial(self) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {e.edge_id: BoundaryCondition(kind="reflective") for e in edges}
        D_bad = np.ones(9)
        D_bad[4] = -1.0  # flip one cell negative
        with pytest.raises(ValueError, match="non-negative"):
            build_variable_diffusion_laplacian(
                mask, edges, edge_conditions, dx=1.0, D_spatial=D_bad,
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_D_spatial(self, bad_value: float) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }
        D_bad = np.ones(9)
        D_bad[4] = bad_value
        with pytest.raises(ValueError, match="finite and non-negative"):
            build_variable_diffusion_laplacian(
                mask,
                edges,
                edge_conditions,
                dx=1.0,
                D_spatial=D_bad,
            )

    @pytest.mark.parametrize("imaginary", [1e-20, np.nan])
    def test_rejects_complex_D_spatial_before_float_cast(
        self,
        imaginary: float,
    ) -> None:
        mask, edges = _full_mask_with_four_edges(3)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }
        D_bad = np.ones(9, dtype=complex)
        D_bad[4] = complex(1.0, imaginary)
        with pytest.raises(ValueError, match="real-valued"):
            build_variable_diffusion_laplacian(
                mask,
                edges,
                edge_conditions,
                dx=1.0,
                D_spatial=D_bad,
            )

    def test_large_finite_equal_D_uses_overflow_safe_harmonic_mean(self) -> None:
        mask, edges = _full_mask_with_four_edges(2)
        edge_conditions = {
            edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges
        }
        D_value = np.finfo(float).max / 4.0
        operator, source = build_variable_diffusion_laplacian(
            mask,
            edges,
            edge_conditions,
            dx=1.0,
            D_spatial=np.full(4, D_value),
        )
        assert np.all(np.isfinite(operator.data))
        assert np.all(np.isfinite(source))


class TestEdgesFromMask:
    """The helper that makes the degenerate configurations usable.

    Assembly demands every outward face be declared and assigned, which is
    right for a device with meaningful edges and prohibitive for the cases the
    engine is supposed to reduce to: a 1xN strip needs 2N+2 faces by hand and a
    single cell needs 4, only to say "nothing leaves".
    """

    def test_single_cell_is_the_zero_operator(self):
        mask = np.ones((1, 1), dtype=bool)
        edges, conditions = edges_from_mask(mask)
        laplacian, _source, _index = build_laplacian_with_boundaries(
            mask, edges, conditions, dx=1.0,
        )
        # 0-D: one cell, no transport at all.
        assert laplacian.shape == (1, 1)
        assert np.allclose(laplacian.toarray(), 0.0)

    def test_one_cell_wide_is_exactly_the_1d_chain(self):
        n = 5
        mask = np.ones((1, n), dtype=bool)
        edges, conditions = edges_from_mask(mask)
        laplacian, _source, _index = build_laplacian_with_boundaries(
            mask, edges, conditions, dx=1.0,
        )
        expected = np.zeros((n, n))
        for i in range(n):
            for j in (i - 1, i + 1):
                if 0 <= j < n:
                    expected[i, j] = 1.0
                    expected[i, i] -= 1.0
        # Not "close to" the 1D operator -- it IS the 1D operator.
        assert np.array_equal(laplacian.toarray(), expected)

    def test_a_column_strip_matches_a_row_strip(self):
        row = np.ones((1, 6), dtype=bool)
        col = np.ones((6, 1), dtype=bool)
        mats = []
        for mask in (row, col):
            edges, conditions = edges_from_mask(mask)
            laplacian, _s, _i = build_laplacian_with_boundaries(
                mask, edges, conditions, dx=1.0,
            )
            mats.append(laplacian.toarray())
        assert np.array_equal(mats[0], mats[1])

    def test_two_dimensional_mask_keeps_the_five_point_stencil(self):
        mask = np.ones((3, 3), dtype=bool)
        edges, conditions = edges_from_mask(mask)
        laplacian, _s, _i = build_laplacian_with_boundaries(
            mask, edges, conditions, dx=1.0,
        )
        dense = laplacian.toarray()
        # The centre of a 3x3 couples to all four neighbours.
        assert np.count_nonzero(dense[4]) - 1 == 4

    def test_interior_hole_gets_its_own_faces(self):
        mask = np.ones((5, 5), dtype=bool)
        mask[2, 2] = False
        edges, conditions = edges_from_mask(mask)
        # The four cells around the hole each face it, so a mask with a hole
        # declares strictly more faces than the same mask without one.
        solid_edges, _ = edges_from_mask(np.ones((5, 5), dtype=bool))
        assert sum(len(e.faces) for e in edges) == (
            sum(len(e.faces) for e in solid_edges) + 4
        )
        laplacian, _s, _i = build_laplacian_with_boundaries(
            mask, edges, conditions, dx=1.0,
        )
        assert laplacian.shape == (24, 24)

    def test_sides_stay_individually_addressable(self):
        mask = np.ones((3, 3), dtype=bool)
        edges, conditions = edges_from_mask(mask)
        assert {e.edge_id for e in edges} == {"up", "down", "left", "right"}
        conditions["left"] = BoundaryCondition("dirichlet", 0.0)
        laplacian, source, _i = build_laplacian_with_boundaries(
            mask, edges, conditions, dx=1.0,
        )
        # A Dirichlet side removes the conserved constant null mode.
        assert not np.allclose(laplacian.toarray().sum(axis=1), 0.0)
        assert source.shape == (9,)

    def test_group_all_produces_one_edge(self):
        mask = np.ones((3, 3), dtype=bool)
        edges, conditions = edges_from_mask(mask, group="all")
        assert [e.edge_id for e in edges] == ["boundary"]
        assert set(conditions) == {"boundary"}

    def test_default_reflective_matches_neumann_zero(self):
        mask = np.ones((4, 4), dtype=bool)
        reflective = edges_from_mask(mask)
        neumann = edges_from_mask(mask, condition=BoundaryCondition("neumann", 0.0))
        mats = []
        for edges, conditions in (reflective, neumann):
            laplacian, _s, _i = build_laplacian_with_boundaries(
                mask, edges, conditions, dx=1.0,
            )
            mats.append(laplacian.toarray())
        assert np.array_equal(mats[0], mats[1])

    def test_rejects_an_empty_mask(self):
        with pytest.raises(BoundaryAssignmentError):
            edges_from_mask(np.zeros((3, 3), dtype=bool))

    def test_rejects_an_unknown_grouping(self):
        with pytest.raises(ValueError, match="group must be"):
            edges_from_mask(np.ones((2, 2), dtype=bool), group="sideways")
