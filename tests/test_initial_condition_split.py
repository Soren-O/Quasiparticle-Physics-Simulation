from __future__ import annotations

import numpy as np

from qpsim.initial_conditions import (
    build_initial_energy_weights,
    build_initial_field,
    canonicalize_initial_condition,
)
from qpsim.models import InitialConditionSpec
from qpsim.solver import build_energy_grid
from qpsim.storage import deserialize_setup


def test_legacy_fermi_dirac_spatial_maps_to_uniform_field() -> None:
    mask = np.ones((3, 4), dtype=bool)
    spec = InitialConditionSpec(
        kind="fermi_dirac",
        params={"amplitude": 2.5, "temperature": 0.12},
    )
    field = build_initial_field(mask, spec)
    assert np.allclose(field[mask], 2.5)


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


def test_canonicalize_legacy_custom_maps_to_split_fields() -> None:
    spec = InitialConditionSpec(
        kind="custom",
        custom_body="return x + y",
        custom_params={"alpha": 1.0},
    )
    normalized = canonicalize_initial_condition(spec)
    assert normalized.spatial_kind == "custom"
    assert normalized.spatial_custom_body == "return x + y"
    assert normalized.spatial_custom_params == {"alpha": 1.0}
    assert normalized.energy_kind == "dos"


def test_deserialize_setup_legacy_fermi_dirac_ic_maps_to_split_fields() -> None:
    payload = {
        "setup_id": "legacy123",
        "name": "Legacy Setup",
        "created_at": "2026-01-01T00:00:00+00:00",
        "geometry": {
            "name": "legacy-geom",
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
            "kind": "fermi_dirac",
            "params": {"amplitude": 1.8, "temperature": 0.2},
        },
    }
    setup = deserialize_setup(payload)
    ic = setup.initial_condition
    assert ic.spatial_kind == "uniform"
    assert np.isclose(float(ic.spatial_params.get("value", -1.0)), 1.8)
    assert ic.energy_kind == "fermi_dirac"
    assert np.isclose(float(ic.energy_params.get("temperature", -1.0)), 0.2)
