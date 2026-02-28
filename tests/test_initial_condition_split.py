from __future__ import annotations

import numpy as np

from qpsim.initial_conditions import (
    build_initial_energy_weights,
    build_initial_field,
    build_initial_phonon_energy_state,
    build_initial_qp_energy_state,
    canonicalize_initial_condition,
)
from qpsim.models import InitialConditionSpec
from qpsim.solver import build_energy_grid, run_2d_crank_nicolson
from qpsim.storage import deserialize_setup


def test_default_spatial_profile_builds_finite_field() -> None:
    mask = np.ones((3, 4), dtype=bool)
    spec = canonicalize_initial_condition(InitialConditionSpec())
    field = build_initial_field(mask, spec)
    assert np.all(np.isfinite(field))
    assert np.all(field[mask] >= 0.0)


def test_split_energy_profile_fermi_dirac_returns_weights() -> None:
    E_bins, _ = build_energy_grid(180.0, 1.0, 3.0, 12)
    spec = InitialConditionSpec(
        spatial_kind="gaussian",
        spatial_params={"amplitude": 1.0, "x0": 0.5, "y0": 0.5, "sigma": 0.2},
        energy_kind="fermi_dirac",
        energy_params={"temperature": 0.1},
    )
    weights = build_initial_energy_weights(
        E_bins=E_bins,
        gap=180.0,
        dynes_gamma=0.1,
        spec=spec,
        bath_temperature=0.1,
    )
    assert weights is not None
    assert weights.shape == E_bins.shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)


def test_split_energy_profile_uniform_honors_value() -> None:
    E_bins, _ = build_energy_grid(180.0, 1.0, 2.0, 6)
    spec = InitialConditionSpec(
        spatial_kind="uniform",
        spatial_params={"value": 1.0},
        energy_kind="uniform",
        energy_params={"value": 3.2},
    )
    weights = build_initial_energy_weights(
        E_bins=E_bins,
        gap=180.0,
        dynes_gamma=0.0,
        spec=spec,
        bath_temperature=0.1,
    )
    assert weights is not None
    assert np.allclose(weights, 3.2)


def test_canonicalize_preserves_split_custom_fields() -> None:
    spec = InitialConditionSpec(
        spatial_kind="custom",
        spatial_custom_body="return x + y",
        spatial_custom_params={"alpha": 1.0},
    )
    normalized = canonicalize_initial_condition(spec)
    assert normalized.spatial_kind == "custom"
    assert normalized.spatial_custom_body == "return x + y"
    assert normalized.spatial_custom_params == {"alpha": 1.0}
    assert normalized.energy_kind == "dos"


def test_deserialize_setup_split_ic_maps_to_split_fields() -> None:
    payload = {
        "setup_id": "split123",
        "name": "Split Setup",
        "created_at": "2026-01-01T00:00:00+00:00",
        "geometry": {
            "name": "split-geom",
            "source_path": "",
            "layer": 0,
            "mesh_size": 1.0,
            "mask": [[1]],
            "edges": [],
        },
        "boundary_conditions": {},
        "parameters": {
            "diffusion_coefficient": 1.0,
            "dt": 0.1,
            "total_time": 1.0,
            "mesh_size": 1.0,
        },
        "initial_condition": {
            "spatial_kind": "uniform",
            "spatial_params": {"value": 1.8},
            "energy_kind": "fermi_dirac",
            "energy_params": {"temperature": 0.2},
        },
    }
    setup = deserialize_setup(payload)
    ic = setup.initial_condition
    assert ic.spatial_kind == "uniform"
    assert np.isclose(float(ic.spatial_params.get("value", -1.0)), 1.8)
    assert ic.energy_kind == "fermi_dirac"
    assert np.isclose(float(ic.energy_params.get("temperature", -1.0)), 0.2)


def test_qp_full_custom_profile_builds_nonseparable_state() -> None:
    mask = np.ones((2, 3), dtype=bool)
    E_bins, _ = build_energy_grid(180.0, 1.0, 2.0, 5)
    spec = InitialConditionSpec(
        spatial_kind="custom",
        energy_kind="custom",
        qp_full_custom_enabled=True,
        qp_full_custom_body="return x + 2.0 * y + 0.001 * E",
    )
    state = build_initial_qp_energy_state(mask, E_bins, spec)
    assert state is not None
    assert state.shape == (E_bins.size, int(np.sum(mask)))
    assert np.all(np.isfinite(state))
    assert np.all(state >= 0.0)


def test_phonon_default_profile_builds_finite_state() -> None:
    mask = np.ones((2, 2), dtype=bool)
    E_bins, _ = build_energy_grid(180.0, 1.0, 2.0, 4)
    spec = canonicalize_initial_condition(InitialConditionSpec())
    state = build_initial_phonon_energy_state(
        mask=mask,
        omega_bins=E_bins,
        spec=spec,
        bath_temperature=0.1,
    )
    assert state.shape == (E_bins.size, int(np.sum(mask)))
    assert np.all(np.isfinite(state))
    assert np.all(state >= 0.0)


def test_solver_uses_full_custom_qp_profile_initialization() -> None:
    mask = np.ones((1, 1), dtype=bool)
    E_bins, dE = build_energy_grid(180.0, 1.0, 2.0, 4)
    spec = InitialConditionSpec(
        spatial_kind="custom",
        energy_kind="custom",
        qp_full_custom_enabled=True,
        qp_full_custom_body="return 0.1",
    )
    expected_integrated = float(np.sum(np.full(E_bins.shape, 0.1)) * dE)
    times, frames, *_ = run_2d_crank_nicolson(
        mask=mask,
        edges=[],
        edge_conditions={},
        initial_field=np.zeros(mask.shape, dtype=float),
        diffusion_coefficient=1.0,
        dt=0.1,
        total_time=0.1,
        dx=1.0,
        energy_gap=180.0,
        energy_min_factor=1.0,
        energy_max_factor=2.0,
        num_energy_bins=4,
        enable_diffusion=False,
        enable_recombination=False,
        enable_scattering=False,
        initial_condition_spec=spec,
    )
    assert np.isclose(times[0], 0.0)
    assert np.isclose(frames[0][0, 0], expected_integrated)
