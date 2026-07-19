"""Portable byte-contract tests for the transient photon-kick artifact."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from validation.transient.photon_kick_response import (
    PhotonKickResult,
    baseline_path,
    read_baseline,
    write_baseline,
)


def test_canonical_photon_kick_baseline_is_utf8_lf() -> None:
    payload = baseline_path().read_bytes()

    text = payload.decode("utf-8")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert text.startswith("# Photon-kick response \u2014 qpsim transient-driver demo\n")


def test_photon_kick_writer_is_utf8_lf_and_round_trips(tmp_path: Path) -> None:
    result = PhotonKickResult(
        E=np.array([190.05, 192.15]),
        snapshot_times=np.array([0.0, 6.0]),
        f_snapshots=np.array([[1.0e-9, 2.0e-9], [3.0e-4, 4.0e-4]]),
        x_qp_snapshots=np.array([1.0e-8, 2.0e-3]),
        f_steady_state=np.array([5.0e-3, 6.0e-3]),
        x_qp_steady_state=7.0e-3,
    )
    path = write_baseline(result, tmp_path / "photon_kick.csv")

    payload = path.read_bytes()
    text = payload.decode("utf-8")
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert b"\r" not in payload
    assert "Photon-kick response \u2014 qpsim" in text

    baseline = read_baseline(path)
    np.testing.assert_array_equal(baseline.E, result.E)
    np.testing.assert_array_equal(baseline.snapshot_times, result.snapshot_times)
    np.testing.assert_array_equal(baseline.f_snapshots, result.f_snapshots)
    np.testing.assert_array_equal(baseline.f_steady_state, result.f_steady_state)
    assert baseline.x_qp_steady_state == result.x_qp_steady_state
