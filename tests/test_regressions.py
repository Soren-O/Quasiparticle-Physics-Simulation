from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from qpsim.geometry import connected_component_count, extract_edge_segments
from qpsim.initial_conditions import build_initial_field
from qpsim.models import BoundaryCondition, InitialConditionSpec, TestCaseResultData, TestSuiteData, utc_now_iso
from qpsim.solver import run_2d_crank_nicolson
from qpsim.storage import (
    TEST_SUITE_FORMAT_VERSION,
    frame_from_jsonable,
    frame_to_jsonable,
    load_test_suite,
    save_test_suite,
)


class RegressionTests(unittest.TestCase):
    def test_frame_json_roundtrip_preserves_nan(self) -> None:
        frame = np.array([[0.5, np.nan], [-2.0, 3.25]], dtype=float)
        payload = frame_to_jsonable(frame)
        self.assertIsNone(payload[0][1])

        recovered = frame_from_jsonable(payload)
        self.assertTrue(np.isnan(recovered[0, 1]))
        self.assertTrue(np.allclose(np.nan_to_num(recovered), np.nan_to_num(frame)))

    def test_custom_ic_vectorized_expression(self) -> None:
        mask = np.ones((32, 40), dtype=bool)
        spec = InitialConditionSpec(kind="custom", custom_body="return x + 2.0 * y")
        field = build_initial_field(mask, spec)

        y_idx, x_idx = np.indices(mask.shape)
        x_norm = (x_idx + 0.5) / mask.shape[1]
        y_norm = (y_idx + 0.5) / mask.shape[0]
        expected = x_norm + 2.0 * y_norm
        self.assertTrue(np.allclose(field, expected))

    def test_custom_ic_scalar_fallback_expression(self) -> None:
        mask = np.ones((24, 24), dtype=bool)
        spec = InitialConditionSpec(
            kind="custom",
            custom_body="return 1.0 if x > params.get('cutoff', 0.5) else 0.0",
            custom_params={"cutoff": 0.5},
        )
        field = build_initial_field(mask, spec)

        x_idx = np.indices(mask.shape)[1]
        x_norm = (x_idx + 0.5) / mask.shape[1]
        expected = (x_norm > 0.5).astype(float)
        self.assertTrue(np.array_equal(field, expected))

    def test_connected_component_count_uses_4_connectivity(self) -> None:
        mask = np.array(
            [
                [1, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
            ],
            dtype=bool,
        )
        self.assertEqual(connected_component_count(mask), 5)

    def test_save_test_suite_clamps_format_version(self) -> None:
        suite = TestSuiteData(
            suite_id="suite123",
            created_at=utc_now_iso(),
            cases=[],
            geometry_groups=[],
            metadata={"format_version": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_test_suite(suite, path=Path(tmpdir) / "suite.json")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["format_version"], TEST_SUITE_FORMAT_VERSION)

    def test_save_test_suite_preserves_higher_format_version(self) -> None:
        future_version = TEST_SUITE_FORMAT_VERSION + 2
        suite = TestSuiteData(
            suite_id="suite456",
            created_at=utc_now_iso(),
            cases=[],
            geometry_groups=[],
            metadata={"format_version": future_version},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_test_suite(suite, path=Path(tmpdir) / "suite.json")
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["metadata"]["format_version"], future_version)

    def test_save_test_suite_preserves_flat_cases_roundtrip(self) -> None:
        suite = TestSuiteData(
            suite_id="suite789",
            created_at=utc_now_iso(),
            cases=[
                TestCaseResultData(
                    case_id="legacy_case",
                    title="Legacy",
                    boundary_label="Reflective",
                    formula_latex="u=1",
                    initial_condition_latex="u=1",
                    description="legacy flat format case",
                    x=[0.5],
                    times=[0.0],
                    simulated=[[1.0]],
                    analytic=[[1.0]],
                    metadata={},
                )
            ],
            geometry_groups=[],
            metadata={"format_version": 1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = save_test_suite(suite, path=Path(tmpdir) / "suite.json")
            loaded = load_test_suite(out_path)
        self.assertEqual(len(loaded.cases), 1)
        self.assertEqual(loaded.cases[0].case_id, "legacy_case")

    def test_load_test_suite_raises_on_missing_group_sidecar(self) -> None:
        payload = {
            "suite_id": "suite_missing_group",
            "created_at": utc_now_iso(),
            "cases": [],
            "geometry_groups": [
                {
                    "geometry_id": "g1",
                    "title": "Group 1",
                    "description": "missing sidecar should raise",
                    "view_mode": "line1d",
                    "preview_mask": [[1, 1, 1]],
                    "cases": [],
                    "case_count": 1,
                    "group_file": "g1.json",
                }
            ],
            "metadata": {"format_version": TEST_SUITE_FORMAT_VERSION},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "suite.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_suite(path, load_group_cases=True)

    def test_load_test_suite_can_skip_missing_group_sidecar(self) -> None:
        payload = {
            "suite_id": "suite_missing_group_skip",
            "created_at": utc_now_iso(),
            "cases": [],
            "geometry_groups": [
                {
                    "geometry_id": "g1",
                    "title": "Group 1",
                    "description": "metadata-only load",
                    "view_mode": "line1d",
                    "preview_mask": [[1, 1, 1]],
                    "cases": [],
                    "case_count": 1,
                    "group_file": "g1.json",
                }
            ],
            "metadata": {"format_version": TEST_SUITE_FORMAT_VERSION},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "suite.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            suite = load_test_suite(path, load_group_cases=False)
        self.assertEqual(len(suite.geometry_groups), 1)
        self.assertEqual(suite.geometry_groups[0].geometry_id, "g1")

    def test_reflective_uniform_field_is_stationary(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 3.0, dtype=float)

        _, frames, mass, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial,
            diffusion_coefficient=1.0,
            dt=0.2,
            total_time=1.0,
            dx=1.0,
            store_every=1,
        )

        for frame in frames:
            self.assertTrue(np.allclose(frame[mask], 3.0, atol=1e-12))
        self.assertTrue(np.allclose(mass, [12.0] * len(mass), atol=1e-12))

    def test_solver_final_time_matches_total_time(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.ones(mask.shape, dtype=float)

        times, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial,
            diffusion_coefficient=1.0,
            dt=0.3,
            total_time=1.0,
            dx=1.0,
            store_every=1,
        )
        self.assertAlmostEqual(times[-1], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
