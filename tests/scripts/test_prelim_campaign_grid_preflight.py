"""Fast observable preflights for the preliminary campaign builders."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from qpsim.backends.spatial import SpatialState
from qpsim.geometries import strip
from scripts.run_prelim_spatial_overnight import strip_coordinates
from qpsim.experiments.prelim_resonators import PRELIM_RESONATORS
from scripts.run_prelim_spatial_overnight import (
    LENGTH_UM,
    _cell_centered_strip_grid,
    _resonator_shifts,
)

from scripts import run_prelim_convergence_checks as convergence
from scripts import run_prelim_finite_phonon_sweep_7mk as sweep_7mk
from scripts import run_prelim_finite_phonon_sweep_7mk_lowD0 as sweep_low_d0
from scripts import run_prelim_finite_phonon_temp_sweep as temp_sweep
from scripts import run_prelim_readout_heating_probe as readout_probe
from scripts import run_prelim_spatial_100um as spatial_100um


def _assert_observable_ready(state: SpatialState) -> None:
    """Exercise the late campaign observable before an expensive solve."""
    expected_x, expected_dx = _cell_centered_strip_grid(state.f.shape[1])
    np.testing.assert_array_equal(strip_coordinates(state), expected_x)
    assert state.geometry.mesh_size == pytest.approx(expected_dx)
    assert float(np.sum(np.full(state.f.shape[1], state.geometry.mesh_size))) == pytest.approx(LENGTH_UM)

    f_ref = np.mean(state.f, axis=1)
    rows = _resonator_shifts(state, f_ref)
    assert len(rows) == len(PRELIM_RESONATORS)
    assert all(float(row["current_integral_strip_um"]) > 0.0 for row in rows)


@pytest.mark.parametrize(
    "builder",
    [
        pytest.param(
            lambda: sweep_7mk._build_state(sweep_7mk.CONFIG.D0_values[0]),
            id="finite-phonon-7mk",
        ),
        pytest.param(
            lambda: temp_sweep._build_state_at_temperature(
                temp_sweep.T_BATH_VALUES_K[0]
            ),
            id="finite-phonon-temperature",
        ),
        pytest.param(
            lambda: readout_probe._build_state(
                readout_probe.CONFIG.D0_values[0]
            ),
            id="readout-heating-probe",
        ),
        pytest.param(
            lambda: convergence._build_state(convergence.CASES[0]),
            id="convergence-check",
        ),
        pytest.param(
            lambda: spatial_100um._build_state(
                spatial_100um.D0_SWEEP_UM2_PER_NS[0]
            ),
            id="spatial-100um",
        ),
    ],
)
def test_campaign_builder_is_observable_ready(
    builder: Callable[[], SpatialState],
) -> None:
    _assert_observable_ready(builder())


def test_cell_centered_grid_honors_explicit_strip_length() -> None:
    x, dx = _cell_centered_strip_grid(5, length_um=250.0)
    assert dx == 50.0
    np.testing.assert_array_equal(x, np.array([25.0, 75.0, 125.0, 175.0, 225.0]))


def test_low_d0_wrapper_inherits_observable_ready_base_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The low-D0 wrapper must exercise the corrected base builder."""
    captured: list[SpatialState] = []

    # Register globals mutated by the wrapper for restoration at teardown.
    monkeypatch.setattr(sweep_7mk, "CONFIG", sweep_7mk.CONFIG)
    monkeypatch.setattr(sweep_7mk, "OUT_DIR", sweep_7mk.OUT_DIR)
    monkeypatch.setattr(sweep_7mk, "__doc__", sweep_7mk.__doc__)

    def preflight_only() -> None:
        assert sweep_7mk.CONFIG.name == "finite_phonon_sweep_7mk_lowD0"
        captured.append(sweep_7mk._build_state(sweep_7mk.CONFIG.D0_values[0]))

    monkeypatch.setattr(sweep_7mk, "main", preflight_only)
    sweep_low_d0.main()

    assert len(captured) == 1
    _assert_observable_ready(captured[0])


def test_7mk_keeps_completed_aggregates_if_first_replacement_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    """Do not erase the previous campaign before one replacement is ready."""
    summary_path = tmp_path / "summary.csv"
    shifts_path = tmp_path / "resonator_shifts.csv"
    metadata_path = tmp_path / "metadata.json"
    summary_path.write_text("run_id,status\nold,completed\n", encoding="utf-8")
    shifts_path.write_text("run_id,shift\nold,1\n", encoding="utf-8")
    metadata_path.write_text('{"campaign": "old"}\n', encoding="utf-8")

    monkeypatch.setattr(sweep_7mk, "OUT_DIR", tmp_path)

    def fail_first_case(*_args: float) -> None:
        raise RuntimeError("preflight failed")

    monkeypatch.setattr(sweep_7mk, "_run_case", fail_first_case)
    with pytest.raises(RuntimeError, match="preflight failed"):
        sweep_7mk.main()

    assert summary_path.read_text(encoding="utf-8") == (
        "run_id,status\nold,completed\n"
    )
    assert shifts_path.read_text(encoding="utf-8") == "run_id,shift\nold,1\n"
    assert metadata_path.read_text(encoding="utf-8") == '{"campaign": "old"}\n'
