from __future__ import annotations

import pytest
import numpy as np

from qpsim.geometry import extract_edge_segments
from qpsim.models import BoundaryCondition, ExternalGenerationSpec, SimulationParameters
from qpsim.solver import run_2d_crank_nicolson
from qpsim.validation import run_fast_validation_suite


def _line_geometry(nx: int) -> tuple[np.ndarray, list, dict[str, BoundaryCondition]]:
    mask = np.ones((1, nx), dtype=bool)
    edges = extract_edge_segments(mask)
    bcs = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
    return mask, edges, bcs


def test_simulation_parameters_resolve_tau_alias_to_split() -> None:
    p = SimulationParameters(
        diffusion_coefficient=6.0,
        dt=0.1,
        total_time=1.0,
        mesh_size=1.0,
        tau_0=300.0,
    )
    assert p.tau_s == pytest.approx(300.0)
    assert p.tau_r == pytest.approx(300.0)
    assert p.tau_0 == pytest.approx(300.0)


def test_simulation_parameters_keep_independent_tau_s_tau_r() -> None:
    p = SimulationParameters(
        diffusion_coefficient=6.0,
        dt=0.1,
        total_time=1.0,
        mesh_size=1.0,
        tau_s=250.0,
        tau_r=900.0,
    )
    assert p.tau_s == pytest.approx(250.0)
    assert p.tau_r == pytest.approx(900.0)
    assert p.tau_0 == pytest.approx(575.0)


def test_external_generation_rejects_negative_rate() -> None:
    with pytest.raises(ValueError):
        SimulationParameters(
            diffusion_coefficient=6.0,
            dt=0.1,
            total_time=1.0,
            mesh_size=1.0,
            external_generation=ExternalGenerationSpec(mode="constant", rate=-1.0),
        )


def test_solver_pauli_violation_raises_when_enforced() -> None:
    mask, edges, bcs = _line_geometry(1)
    initial_field = np.array([[2.0]], dtype=float)

    with pytest.raises(ValueError, match="Pauli occupation exceeded limit"):
        run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=bcs,
            initial_field=initial_field,
            diffusion_coefficient=6.0,
            dt=0.1,
            total_time=0.2,
            dx=1.0,
            energy_gap=180.0,
            energy_min_factor=1.5,
            energy_max_factor=1.5,
            num_energy_bins=1,
            enable_diffusion=False,
            enable_recombination=False,
            enable_scattering=False,
            enforce_pauli=True,
            pauli_error_threshold=1.0,
        )


def test_solver_pauli_violation_can_warn_without_raise() -> None:
    mask, edges, bcs = _line_geometry(1)
    initial_field = np.array([[2.0]], dtype=float)

    with pytest.warns(UserWarning, match="Pauli occupation exceeded limit"):
        run_2d_crank_nicolson(
            mask=mask,
            edges=edges,
            edge_conditions=bcs,
            initial_field=initial_field,
            diffusion_coefficient=6.0,
            dt=0.1,
            total_time=0.2,
            dx=1.0,
            energy_gap=180.0,
            energy_min_factor=1.5,
            energy_max_factor=1.5,
            num_energy_bins=1,
            enable_diffusion=False,
            enable_recombination=False,
            enable_scattering=False,
            enforce_pauli=False,
            pauli_error_threshold=1.0,
        )


def test_fast_validation_suite_passes_default_configuration() -> None:
    report = run_fast_validation_suite()
    payload = report.as_dict()
    assert payload["detailed_balance"]["passed"] is True
    assert payload["thermal_stability"]["passed"] is True
    assert payload["pure_diffusion"]["passed"] is True
    assert payload["pure_scattering"]["passed"] is True
    assert payload["pure_recombination"]["passed"] is True
    assert payload["overall_passed"] is True
