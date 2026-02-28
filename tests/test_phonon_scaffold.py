from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from qpsim.geometry import extract_edge_segments
from qpsim.models import BoundaryCondition, SimulationResultData
from qpsim.solver import (
    build_fixed_phonon_history,
    run_2d_crank_nicolson,
    thermal_phonon_occupation,
)
from qpsim.storage import deserialize_simulation, load_simulation, save_simulation


def test_fixed_phonon_history_matches_custom_geometry_mask_and_energy_bins() -> None:
    mask = np.array(
        [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=bool,
    )
    times = [0.0, 0.4, 0.8]
    bath_temperature = 0.125
    omega_bins = np.array([180.0, 240.0, 360.0], dtype=float)

    frames, energy_frames, bins_out, meta = build_fixed_phonon_history(
        mask=mask,
        times=times,
        bath_temperature=bath_temperature,
        phonon_energy_bins=omega_bins,
    )

    assert len(frames) == len(times)
    assert energy_frames is not None
    assert bins_out is not None
    assert len(energy_frames) == len(times)
    assert np.allclose(bins_out, omega_bins)
    assert meta["mode"] == "fixed_temperature"
    assert float(meta["phonon_temperature_K"]) == bath_temperature

    for frame in frames:
        assert frame.shape == mask.shape
        assert np.allclose(frame[mask], bath_temperature)
        assert np.all(np.isnan(frame[~mask]))

    expected_occ = thermal_phonon_occupation(omega_bins, bath_temperature)
    for time_slice in energy_frames:
        assert len(time_slice) == len(omega_bins)
        for omega_idx, omega_frame in enumerate(time_slice):
            assert omega_frame.shape == mask.shape
            assert np.allclose(omega_frame[mask], expected_occ[omega_idx])
            assert np.all(np.isnan(omega_frame[~mask]))


def test_simulation_roundtrip_preserves_phonon_fields() -> None:
    result = SimulationResultData(
        simulation_id="sim123",
        setup_id="setup456",
        setup_name="Roundtrip",
        created_at="2026-02-27T00:00:00Z",
        times=[0.0, 1.0],
        frames=[
            [[1.0, None], [None, 2.0]],
            [[1.5, None], [None, 2.5]],
        ],
        mass_over_time=[3.0, 4.0],
        color_limits=[1.0, 2.5],
        metadata={"energy_qp_total": [1.0, 1.1]},
        phonon_frames=[
            [[0.1, None], [None, 0.1]],
            [[0.1, None], [None, 0.1]],
        ],
        phonon_energy_frames=[
            [
                [[0.01, None], [None, 0.01]],
                [[0.02, None], [None, 0.02]],
            ],
            [
                [[0.01, None], [None, 0.01]],
                [[0.02, None], [None, 0.02]],
            ],
        ],
        phonon_energy_bins=[180.0, 220.0],
        phonon_metadata={"mode": "fixed_temperature", "phonon_temperature_K": 0.1},
    )

    sandbox = Path(__file__).resolve().parents[1] / ".tmp_test"
    sandbox.mkdir(parents=True, exist_ok=True)
    path = sandbox / f"sim_with_phonons_{uuid.uuid4().hex}.json"
    try:
        save_simulation(result, path=path)
        loaded = load_simulation(path)

        assert loaded.phonon_frames == result.phonon_frames
        assert loaded.phonon_energy_frames == result.phonon_energy_frames
        assert loaded.phonon_energy_bins == result.phonon_energy_bins
        assert loaded.phonon_metadata == result.phonon_metadata
    finally:
        path.unlink(missing_ok=True)


def test_deserialize_simulation_without_phonon_fields_remains_compatible() -> None:
    legacy_payload = {
        "simulation_id": "legacy123",
        "setup_id": "setup123",
        "setup_name": "Legacy",
        "created_at": "2026-02-27T00:00:00Z",
        "times": [0.0],
        "frames": [[[1.0]]],
        "mass_over_time": [1.0],
        "color_limits": [1.0, 1.0],
        "metadata": {},
        "energy_frames": None,
        "energy_bins": None,
    }
    loaded = deserialize_simulation(legacy_payload)
    assert loaded.phonon_frames is None
    assert loaded.phonon_energy_frames is None
    assert loaded.phonon_energy_bins is None
    assert loaded.phonon_metadata is None


def test_phonon_scaffold_generation_does_not_modify_qp_outputs() -> None:
    mask = np.ones((2, 3), dtype=bool)
    edges = extract_edge_segments(mask)
    bcs = {edge.edge_id: BoundaryCondition(kind="reflective") for edge in edges}
    initial = np.full(mask.shape, 0.25, dtype=float)

    times, frames, mass, _, energy_frames, energy_bins = run_2d_crank_nicolson(
        mask=mask,
        edges=edges,
        edge_conditions=bcs,
        initial_field=initial,
        diffusion_coefficient=6.0,
        dt=0.2,
        total_time=1.0,
        dx=1.0,
        store_every=1,
        energy_gap=180.0,
        energy_min_factor=1.0,
        energy_max_factor=3.0,
        num_energy_bins=6,
        enable_diffusion=True,
        enable_recombination=True,
        enable_scattering=True,
        tau_0=440.0,
        T_c=1.2,
        bath_temperature=0.1,
    )

    frames_before = [frame.copy() for frame in frames]
    mass_before = np.array(mass, dtype=float)
    assert energy_frames is not None
    energy_before = [[eframe.copy() for eframe in time_slice] for time_slice in energy_frames]

    build_fixed_phonon_history(
        mask=mask,
        times=times,
        bath_temperature=0.1,
        phonon_energy_bins=energy_bins,
    )

    for frame, frame_before in zip(frames, frames_before):
        assert np.allclose(frame, frame_before, equal_nan=True)
    assert np.allclose(np.array(mass, dtype=float), mass_before)
    for time_slice, time_slice_before in zip(energy_frames, energy_before):
        for eframe, eframe_before in zip(time_slice, time_slice_before):
            assert np.allclose(eframe, eframe_before, equal_nan=True)
