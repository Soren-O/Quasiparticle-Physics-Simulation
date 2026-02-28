from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

import numpy as np

from qpsim.geometry import connected_component_count, extract_edge_segments
from qpsim.initial_conditions import build_initial_field
from qpsim.models import BoundaryCondition, ExternalGenerationSpec, InitialConditionSpec, SimulationParameters, TestSuiteData, utc_now_iso
from qpsim.precompute import precompute_arrays, validate_precomputed
from qpsim.solver import _bcs_density_of_states, _dynes_density_of_states, run_2d_crank_nicolson
from qpsim.storage import (
    TEST_SUITE_FORMAT_VERSION,
    frame_from_jsonable,
    frame_to_jsonable,
    load_test_geometry_group,
    load_test_suite,
    save_test_suite,
)

_SANDBOX_TMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp_test"
_SANDBOX_TMP_ROOT.mkdir(parents=True, exist_ok=True)


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

    def test_custom_ic_rejects_unsafe_expression(self) -> None:
        mask = np.ones((8, 8), dtype=bool)
        spec = InitialConditionSpec(
            kind="custom",
            custom_body="__import__('os').system('echo unsafe')",
        )
        with self.assertRaises(ValueError):
            build_initial_field(mask, spec)

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

    def test_save_test_suite_requires_geometry_groups(self) -> None:
        suite = TestSuiteData(
            suite_id="suite123",
            created_at=utc_now_iso(),
            cases=[],
            geometry_groups=[],
            metadata={"format_version": TEST_SUITE_FORMAT_VERSION},
        )
        path = _SANDBOX_TMP_ROOT / f"suite_{uuid.uuid4().hex}.json"
        try:
            with self.assertRaises(ValueError):
                save_test_suite(suite, path=path)
        finally:
            path.unlink(missing_ok=True)

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
        path = _SANDBOX_TMP_ROOT / f"suite_{uuid.uuid4().hex}.json"
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_suite(path, load_group_cases=True)
        finally:
            path.unlink(missing_ok=True)

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
        path = _SANDBOX_TMP_ROOT / f"suite_{uuid.uuid4().hex}.json"
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
            suite = load_test_suite(path, load_group_cases=False)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(len(suite.geometry_groups), 1)
        self.assertEqual(suite.geometry_groups[0].geometry_id, "g1")

    def test_load_test_geometry_group_rejects_path_escape(self) -> None:
        uid = uuid.uuid4().hex
        path = _SANDBOX_TMP_ROOT / f"suite_{uid}.json"
        outside_group = _SANDBOX_TMP_ROOT / f"outside_{uid}.json"
        payload = {
            "suite_id": "suite_escape",
            "created_at": utc_now_iso(),
            "cases": [],
            "geometry_groups": [
                {
                    "geometry_id": "g1",
                    "title": "Group 1",
                    "description": "path escape should be rejected",
                    "view_mode": "line1d",
                    "preview_mask": [[1, 1, 1]],
                    "cases": [],
                    "case_count": 1,
                    "group_file": f"..\\{outside_group.name}",
                }
            ],
            "metadata": {"format_version": TEST_SUITE_FORMAT_VERSION},
        }
        try:
            outside_group.write_text(
                json.dumps(
                    {
                        "suite_id": "suite_escape",
                        "group": {
                            "geometry_id": "g1",
                            "title": "Group 1",
                            "description": "",
                            "view_mode": "line1d",
                            "preview_mask": [[1, 1, 1]],
                            "cases": [],
                            "case_count": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_geometry_group(path, "g1")
        finally:
            path.unlink(missing_ok=True)
            outside_group.unlink(missing_ok=True)

    def test_load_test_suite_rejects_legacy_flat_case_format(self) -> None:
        payload = {
            "suite_id": "legacy_suite",
            "created_at": utc_now_iso(),
            "cases": [
                {
                    "case_id": "legacy_case",
                    "title": "Legacy",
                    "boundary_label": "Reflective",
                    "formula_latex": "u=1",
                    "initial_condition_latex": "u=1",
                    "description": "legacy flat format case",
                    "x": [0.5],
                    "times": [0.0],
                    "simulated": [[1.0]],
                    "analytic": [[1.0]],
                    "metadata": {},
                }
            ],
            "metadata": {"format_version": 1},
        }
        path = _SANDBOX_TMP_ROOT / f"suite_{uuid.uuid4().hex}.json"
        try:
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_test_suite(path)
        finally:
            path.unlink(missing_ok=True)

    def test_reflective_uniform_field_is_stationary(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 3.0, dtype=float)

        _, frames, mass, _, _, _ = run_2d_crank_nicolson(
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

        times, _, _, _, _, _ = run_2d_crank_nicolson(
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

    def test_solver_progress_callback_receives_stored_frames(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.ones(mask.shape, dtype=float)
        callback_times: list[float] = []
        callback_frames: list[np.ndarray] = []

        def _progress_callback(time_ns: float, frame: np.ndarray) -> None:
            callback_times.append(float(time_ns))
            callback_frames.append(np.array(frame, copy=True))

        times, frames, _, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial,
            diffusion_coefficient=1.0,
            dt=0.1,
            total_time=0.3,
            dx=1.0,
            store_every=1,
            progress_callback=_progress_callback,
        )
        self.assertEqual(len(callback_times), len(times))
        self.assertAlmostEqual(callback_times[0], 0.0, places=12)
        self.assertAlmostEqual(callback_times[-1], times[-1], places=12)
        self.assertEqual(len(callback_frames), len(frames))
        self.assertTrue(np.allclose(np.nan_to_num(callback_frames[-1]), np.nan_to_num(frames[-1])))


    def test_dynes_dos_gamma_zero_matches_bcs(self) -> None:
        """Dynes DOS with gamma=0 should match BCS DOS exactly."""
        E = np.linspace(180.0, 900.0, 50)
        bcs = _bcs_density_of_states(E, 180.0)
        dynes = _dynes_density_of_states(E, 180.0, 0.0)
        self.assertTrue(np.allclose(bcs, dynes, atol=1e-14))

    def test_dynes_dos_smooths_singularity(self) -> None:
        """Dynes DOS with gamma>0 should be finite everywhere (no singularity at gap edge)."""
        E = np.linspace(179.0, 181.0, 100)
        dos = _dynes_density_of_states(E, 180.0, 5.0)
        self.assertTrue(np.all(np.isfinite(dos)))
        self.assertTrue(np.all(dos >= 0.0))
        # Should have non-zero DOS even slightly below the gap
        self.assertGreater(dos[0], 0.0)

    def test_precompute_uniform_matches_direct(self) -> None:
        """Uniform precomputed arrays should match direct solver results."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 1.0, dtype=float)

        params = SimulationParameters(
            diffusion_coefficient=6.0, dt=1.0, total_time=3.0, mesh_size=1.0,
            store_every=1, energy_gap=180.0, energy_max_factor=5.0,
            num_energy_bins=10, enable_diffusion=True, enable_recombination=True,
            tau_0=440.0, T_c=1.2, bath_temperature=0.1,
        )
        precomp = precompute_arrays(mask, edges, edge_conditions, params)
        self.assertTrue(bool(precomp["is_uniform"]))

        # Run with precomputed
        _, _, mass_pre, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=10, enable_diffusion=True,
            enable_recombination=True, tau_0=440.0, T_c=1.2, bath_temperature=0.1,
            precomputed=precomp,
        )
        # Run without precomputed
        _, _, mass_dir, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=10, enable_diffusion=True,
            enable_recombination=True, tau_0=440.0, T_c=1.2, bath_temperature=0.1,
        )
        for m_pre, m_dir in zip(mass_pre, mass_dir):
            self.assertAlmostEqual(m_pre, m_dir, places=10)

    def test_precompute_nonuniform_gap_runs(self) -> None:
        """Non-uniform gap expression should produce non-uniform precomputed arrays."""
        mask = np.ones((4, 4), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}

        params = SimulationParameters(
            diffusion_coefficient=6.0, dt=1.0, total_time=2.0, mesh_size=1.0,
            store_every=1, energy_gap=180.0, energy_max_factor=5.0,
            num_energy_bins=5, enable_diffusion=True,
            gap_expression="return 180 + 20 * x",
        )
        precomp = precompute_arrays(mask, edges, edge_conditions, params)
        self.assertFalse(bool(precomp["is_uniform"]))
        self.assertIn("K_r_all", precomp)

        # Run with non-uniform precomputed
        initial = np.full(mask.shape, 1.0, dtype=float)
        times, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=2.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5, enable_diffusion=True,
            precomputed=precomp,
        )
        self.assertAlmostEqual(times[-1], 2.0, places=10)
        for m in mass:
            self.assertTrue(np.isfinite(m))

    def test_precompute_rejects_non_finite_gap_expression(self) -> None:
        mask = np.ones((4, 4), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        params = SimulationParameters(
            diffusion_coefficient=6.0,
            dt=0.1,
            total_time=0.1,
            mesh_size=1.0,
            energy_gap=180.0,
            energy_min_factor=1.0,
            energy_max_factor=3.0,
            num_energy_bins=8,
            gap_expression="np.nan",
        )
        with self.assertRaises(ValueError):
            precompute_arrays(mask, edges, edge_conditions, params)

    def test_external_generation_constant_increases_mass(self) -> None:
        """Constant external generation should increase total quasiparticle mass."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 0.1, dtype=float)
        ext_gen = ExternalGenerationSpec(mode="constant", rate=0.01)

        _, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=5.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=8,
            enable_diffusion=True,
            external_generation=ext_gen,
        )
        # Mass should increase over time
        self.assertGreater(mass[-1], mass[0])

    def test_external_generation_pulse_only_during_window(self) -> None:
        """Pulse generation should only be active during the pulse window."""
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 0.0, dtype=float)
        # Pulse from t=0 to t=2, then nothing
        ext_gen = ExternalGenerationSpec(mode="pulse", pulse_rate=1.0, pulse_start=0.0, pulse_duration=2.0)

        times, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=4.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5,
            enable_diffusion=False,
            external_generation=ext_gen,
        )
        # Mass should increase during pulse and stabilize after
        self.assertGreater(mass[2], mass[0])  # During pulse
        # After pulse ends (t=2..4), mass should be constant with no other processes
        self.assertAlmostEqual(mass[3], mass[2], places=10)

    def test_external_generation_none_matches_baseline(self) -> None:
        """mode='none' should produce identical results to no external generation."""
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 1.0, dtype=float)
        ext_gen = ExternalGenerationSpec(mode="none")

        _, _, mass_ext, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5,
            external_generation=ext_gen,
        )
        _, _, mass_none, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5,
        )
        for m_ext, m_none in zip(mass_ext, mass_none):
            self.assertAlmostEqual(m_ext, m_none, places=12)

    def test_external_generation_custom_rejects_unsafe_expression(self) -> None:
        mask = np.ones((1, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        ext_gen = ExternalGenerationSpec(
            mode="custom",
            custom_body="__import__('os').system('echo unsafe')",
        )
        with self.assertRaises(ValueError):
            run_2d_crank_nicolson(
                mask=mask,
                edges=edges,
                edge_conditions=edge_conditions,
                initial_field=np.zeros((1, 2), dtype=float),
                diffusion_coefficient=6.0,
                dt=0.1,
                total_time=0.1,
                dx=1.0,
                energy_gap=180.0,
                energy_min_factor=1.0,
                energy_max_factor=3.0,
                num_energy_bins=8,
                enable_diffusion=False,
                external_generation=ext_gen,
            )

    def test_boltzphlow_collision_solver_runs(self) -> None:
        """BoltzPhlow time-relaxation collision solver should run and produce finite results."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 1.0, dtype=float)

        times, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=2.0,
            total_time=6.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=8,
            enable_diffusion=True, enable_recombination=True, enable_scattering=True,
            collision_solver="boltzphlow_relaxation",
            tau_0=440.0, T_c=1.2, bath_temperature=0.1,
        )
        self.assertAlmostEqual(times[-1], 6.0, places=10)
        for m in mass:
            self.assertTrue(np.isfinite(m))
            self.assertGreaterEqual(m, 0.0)

    def test_scattering_and_recombination_combined(self) -> None:
        """Scattering + recombination together should run without error and conserve mass reasonably."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 1.0, dtype=float)

        times, frames, mass, _, energy_frames, energy_bins = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial,
            diffusion_coefficient=6.0,
            dt=1.0,
            total_time=5.0,
            dx=1.0,
            store_every=1,
            energy_gap=180.0,
            energy_min_factor=1.0,
            energy_max_factor=5.0,
            num_energy_bins=10,
            enable_diffusion=True,
            enable_recombination=True,
            enable_scattering=True,
            tau_0=440.0,
            T_c=1.2,
            bath_temperature=0.1,
        )
        self.assertGreater(len(times), 1)
        self.assertAlmostEqual(times[-1], 5.0, places=10)
        # Energy frames should be present
        self.assertIsNotNone(energy_frames)
        self.assertIsNotNone(energy_bins)
        # Mass should remain finite and non-negative
        for m in mass:
            self.assertTrue(np.isfinite(m))
            self.assertGreaterEqual(m, 0.0)


    # --- Codex review regression tests ---

    def test_nonuniform_dirichlet_bc_produces_nonzero(self) -> None:
        """Non-uniform gap + non-zero Dirichlet BC should drive the field toward the BC value."""
        mask = np.ones((4, 4), dtype=bool)
        edges = extract_edge_segments(mask)
        # Set one edge to Dirichlet with value 5.0, rest reflective
        edge_conditions = {}
        dirichlet_set = False
        for edge in edges:
            if not dirichlet_set:
                edge_conditions[edge.edge_id] = BoundaryCondition(kind="dirichlet", value=5.0)
                dirichlet_set = True
            else:
                edge_conditions[edge.edge_id] = BoundaryCondition(kind="reflective")
        initial = np.zeros(mask.shape, dtype=float)

        params = SimulationParameters(
            diffusion_coefficient=6.0, dt=0.5, total_time=5.0, mesh_size=1.0,
            store_every=1, energy_gap=180.0, energy_max_factor=5.0,
            num_energy_bins=5, enable_diffusion=True,
            gap_expression="return 180 + 10 * x",
        )
        precomp = precompute_arrays(mask, edges, edge_conditions, params)
        self.assertFalse(bool(precomp["is_uniform"]))

        _, frames, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=0.5,
            total_time=5.0, dx=1.0, store_every=5, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5, enable_diffusion=True,
            gap_expression="return 180 + 10 * x",
            precomputed=precomp,
        )
        # Dirichlet BC should push mass above zero
        self.assertGreater(mass[-1], 0.0)

    def test_gap_expression_without_precompute_auto_computes(self) -> None:
        """Setting gap_expression without precompute should still produce non-uniform results."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.full(mask.shape, 1.0, dtype=float)

        # Run with gap_expression but no precomputed sidecar
        times, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
            total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=5, enable_diffusion=True,
            gap_expression="return 180 + 20 * x",
        )
        self.assertAlmostEqual(times[-1], 3.0, places=10)
        for m in mass:
            self.assertTrue(np.isfinite(m))

    def test_precompute_validates_changed_parameters(self) -> None:
        """Changing parameters after precompute should be detected by validation."""
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}

        params1 = SimulationParameters(
            diffusion_coefficient=6.0, dt=1.0, total_time=3.0, mesh_size=1.0,
            energy_gap=180.0, energy_max_factor=5.0, num_energy_bins=5,
        )
        precomp = precompute_arrays(mask, edges, edge_conditions, params1)

        # Same params should validate
        self.assertIsNone(validate_precomputed(precomp, params1, mask))

        # Changed gap should fail validation
        params2 = SimulationParameters(
            diffusion_coefficient=6.0, dt=1.0, total_time=3.0, mesh_size=1.0,
            energy_gap=200.0, energy_max_factor=5.0, num_energy_bins=5,
        )
        mismatch = validate_precomputed(precomp, params2, mask)
        self.assertIsNotNone(mismatch)
        self.assertIn("energy_gap", mismatch)

    def test_precompute_validation_checks_mask_hash(self) -> None:
        """Different masks with the same n_spatial must not pass precompute validation."""
        mask_a = np.array([[1, 1, 1, 1]], dtype=bool)
        mask_b = np.array([[1, 1], [1, 1]], dtype=bool)  # same n_spatial=4, different topology
        edges_a = extract_edge_segments(mask_a)
        edge_conditions_a = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges_a}
        params = SimulationParameters(
            diffusion_coefficient=6.0, dt=1.0, total_time=3.0, mesh_size=1.0,
            energy_gap=180.0, energy_max_factor=5.0, num_energy_bins=5,
        )
        precomp = precompute_arrays(mask_a, edges_a, edge_conditions_a, params)

        mismatch = validate_precomputed(precomp, params, mask_b)
        self.assertIsNotNone(mismatch)
        self.assertIn("mask_hash", mismatch)

    def test_invalid_collision_solver_rejected_by_parameters(self) -> None:
        with self.assertRaises(ValueError):
            SimulationParameters(
                diffusion_coefficient=6.0,
                dt=1.0,
                total_time=3.0,
                mesh_size=1.0,
                collision_solver="not_a_solver",
            )

    def test_invalid_collision_solver_rejected_by_solver(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.ones(mask.shape, dtype=float)
        with self.assertRaises(ValueError):
            run_2d_crank_nicolson(
                mask=mask, edges=edges, edge_conditions=edge_conditions,
                initial_field=initial, diffusion_coefficient=6.0, dt=1.0,
                total_time=3.0, dx=1.0, store_every=1, energy_gap=180.0,
                energy_max_factor=5.0, num_energy_bins=5,
                collision_solver=" definitely-not-valid ",
            )

    def test_diffusion_disabled_does_not_require_boundary_assignment(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        initial = np.full(mask.shape, 1.0, dtype=float)
        _, _, mass, _, _, _ = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions={},  # intentionally empty
            initial_field=initial,
            diffusion_coefficient=6.0,
            dt=1.0,
            total_time=3.0,
            dx=1.0,
            store_every=1,
            energy_gap=180.0,
            energy_max_factor=5.0,
            num_energy_bins=5,
            enable_diffusion=False,
        )
        self.assertAlmostEqual(mass[0], mass[-1], places=12)

    def test_energy_grid_cell_centers_are_above_gap(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.ones(mask.shape, dtype=float)
        _, _, _, _, _, energy_bins = run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=edge_conditions,
            initial_field=initial,
            diffusion_coefficient=6.0,
            dt=1.0,
            total_time=1.0,
            dx=1.0,
            energy_gap=180.0,
            energy_min_factor=1.0,
            energy_max_factor=5.0,
            num_energy_bins=10,
            enable_diffusion=False,
        )
        self.assertIsNotNone(energy_bins)
        self.assertGreater(float(np.min(np.asarray(energy_bins, dtype=float))), 180.0)

    def test_energy_weights_validation_rejects_negative_values(self) -> None:
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        initial = np.ones(mask.shape, dtype=float)
        with self.assertRaises(ValueError):
            run_2d_crank_nicolson(
                mask=mask,
                edges=edges,
                edge_conditions=edge_conditions,
                initial_field=initial,
                diffusion_coefficient=6.0,
                dt=1.0,
                total_time=1.0,
                dx=1.0,
                energy_gap=180.0,
                energy_min_factor=1.0,
                energy_max_factor=5.0,
                num_energy_bins=10,
                energy_weights=np.full(10, -1.0),
                enable_diffusion=False,
            )

    def test_boltzphlow_collision_non_negative(self) -> None:
        """BoltzPhlow collision solver should never produce negative spectral densities."""
        mask = np.ones((2, 2), dtype=bool)
        edges = extract_edge_segments(mask)
        edge_conditions = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
        # Start with very small initial density — prone to going negative without clamp
        initial = np.full(mask.shape, 0.001, dtype=float)

        _, _, _, _, energy_frames, _ = run_2d_crank_nicolson(
            mask=mask, edges=edges, edge_conditions=edge_conditions,
            initial_field=initial, diffusion_coefficient=6.0, dt=5.0,
            total_time=50.0, dx=1.0, store_every=5, energy_gap=180.0,
            energy_max_factor=5.0, num_energy_bins=8,
            enable_diffusion=True, enable_recombination=True, enable_scattering=True,
            collision_solver="boltzphlow_relaxation",
            tau_0=440.0, T_c=1.2, bath_temperature=0.1,
        )
        self.assertIsNotNone(energy_frames)
        for time_slice in energy_frames:
            for eframe in time_slice:
                vals = eframe[~np.isnan(eframe)]
                self.assertTrue(np.all(vals >= 0.0), f"Negative spectral density found: {vals.min()}")

    def test_variable_diffusion_missing_bc_raises(self) -> None:
        """Variable-diffusion builder should raise on missing face BC, same as scalar path."""
        from qpsim.solver import build_variable_diffusion_laplacian
        mask = np.ones((3, 3), dtype=bool)
        edges = extract_edge_segments(mask)
        # Only assign some edges — leave gaps
        edge_conditions = {}
        for i, edge in enumerate(edges):
            if i < len(edges) // 2:
                edge_conditions[edge.edge_id] = BoundaryCondition(kind="reflective")
        D_spatial = np.ones(int(np.sum(mask)), dtype=float)
        with self.assertRaises(Exception):
            build_variable_diffusion_laplacian(mask, edges, edge_conditions, 1.0, D_spatial)


if __name__ == "__main__":
    unittest.main()
