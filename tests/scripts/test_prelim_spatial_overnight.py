"""Unit-contract checks for the prelim spatial source calibration."""

from __future__ import annotations

import numpy as np
import pytest
import scripts.run_prelim_readout_heating_overnight as readout
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from scripts.run_prelim_spatial_overnight import (
    AL_STRIP_THICKNESS_UM,
    AL_STRIP_WIDTH_UM,
    CALIBRATED_CONFIG,
    LENGTH_UM,
    SMOKE_CONFIG,
    _build_state,
    _campaign_cell_widths,
    _cell_centered_strip_grid,
    _source_calibration,
    _source_flux,
)


def test_calibrated_source_rate_uses_ev_dos_contract() -> None:
    calibration = _source_calibration(CALIBRATED_CONFIG, source_rate_per_ns=1.0e-2)

    # rev6 uses NX equal finite-volume cells, not NX endpoint samples.
    assert calibration["qps_per_xqp_source_cell"] == pytest.approx(
        3.132e5 * 40.0 / 41.0
    )
    assert calibration["estimated_source_qp_per_s"] == pytest.approx(
        3.132e12 * 40.0 / 41.0
    )


def test_source_label_equals_physical_cell_integral() -> None:
    config = SMOKE_CONFIG
    source_rate = config.source_rates_per_ns[0]
    state = _build_state(config, D0=config.D0_values[0])
    source = _source_flux(
        state,
        local_xqp_generation_rate_per_ns=source_rate,
        center_delta=config.source_centers_delta[0],
        sigma_delta=config.source_sigmas_delta[0],
    )
    widths_um = _campaign_cell_widths(state)
    spectral_weights = bcs_dos_cell_weights(
        state.spectral.E,
        state.spectral.dE,
        state.gap,
    )
    xqp_rate_per_ns = (
        np.sum(spectral_weights[:, None] * source.gain, axis=0) / state.gap
    )
    qps_per_xqp_um3 = (
        4.0 * float(state.material.rho_F) * state.gap * 1e-6 * 1e-18
    )
    physical_qp_per_s = float(
        np.sum(xqp_rate_per_ns * widths_um)
        * AL_STRIP_WIDTH_UM
        * AL_STRIP_THICKNESS_UM
        * qps_per_xqp_um3
        * 1e9
    )

    calibration = _source_calibration(config, source_rate)
    np.testing.assert_allclose(
        xqp_rate_per_ns,
        np.r_[source_rate, np.zeros(config.NX - 1)],
        rtol=1e-13,
        atol=1e-18,
    )
    assert widths_um.sum() == pytest.approx(LENGTH_UM)
    assert physical_qp_per_s == pytest.approx(
        calibration["estimated_source_qp_per_s"], rel=1e-13
    )


def test_spatial_and_readout_campaigns_share_cell_center_grid() -> None:
    expected_x, expected_dx = _cell_centered_strip_grid(SMOKE_CONFIG.NX)
    spatial_state = _build_state(SMOKE_CONFIG, D0=SMOKE_CONFIG.D0_values[0])
    readout_config = readout.SMOKE_CONFIG
    readout_x, readout_dx = _cell_centered_strip_grid(readout_config.NX)
    readout_state = readout._build_state(
        readout_config,
        D0=readout_config.D0_values[0],
    )

    np.testing.assert_array_equal(spatial_state.x, expected_x)
    np.testing.assert_array_equal(readout_state.x, readout_x)
    assert spatial_state.dx == pytest.approx(expected_dx)
    assert readout_state.dx == pytest.approx(readout_dx)
    assert spatial_state.cell_widths.sum() == pytest.approx(LENGTH_UM)
    assert readout_state.cell_widths.sum() == pytest.approx(LENGTH_UM)


class TestAggregateHeaderGuard:
    """2026-07-20 review: appending rows under a stale aggregate-CSV header
    silently mislabels columns; the runners must refuse instead."""

    def test_stale_header_raises_with_guidance(self, tmp_path) -> None:
        import pytest
        from scripts.run_prelim_spatial_overnight import require_matching_header

        path = tmp_path / "summary.csv"
        path.write_text("run_id,status,old_col\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="--no-resume"):
            require_matching_header(path, ["run_id", "status", "old_col", "new_col"])

    def test_matching_or_missing_header_passes(self, tmp_path) -> None:
        from scripts.run_prelim_spatial_overnight import require_matching_header

        fields = ["run_id", "status", "col"]
        path = tmp_path / "summary.csv"
        require_matching_header(path, fields)  # missing file: fine
        path.write_text("run_id,status,col\n", encoding="utf-8")
        require_matching_header(path, fields)  # exact match: fine
