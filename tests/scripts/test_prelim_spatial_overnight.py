"""Unit-contract checks for the prelim spatial source calibration."""

from __future__ import annotations

import pytest
from scripts.run_prelim_spatial_overnight import CALIBRATED_CONFIG, _source_calibration


def test_calibrated_source_rate_uses_ev_dos_contract() -> None:
    calibration = _source_calibration(CALIBRATED_CONFIG, source_rate_per_ns=1.0e-2)

    assert calibration["qps_per_xqp_source_cell"] == pytest.approx(3.132e5)
    assert calibration["estimated_source_qp_per_s"] == pytest.approx(3.132e12)


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
