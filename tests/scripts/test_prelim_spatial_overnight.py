"""Unit-contract checks for the prelim spatial source calibration."""

from __future__ import annotations

import pytest
from scripts.run_prelim_spatial_overnight import CALIBRATED_CONFIG, _source_calibration


def test_calibrated_source_rate_uses_ev_dos_contract() -> None:
    calibration = _source_calibration(CALIBRATED_CONFIG, source_rate_per_ns=1.0e-2)

    assert calibration["qps_per_xqp_source_cell"] == pytest.approx(3.132e5)
    assert calibration["estimated_source_qp_per_s"] == pytest.approx(3.132e12)
