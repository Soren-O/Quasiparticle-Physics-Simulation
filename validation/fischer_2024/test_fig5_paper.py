"""Regression test: Fischer & Catelani 2024 Fig. 5 paper-target run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- three
Newton-on-$f$ steady-state solves at 810 bins; total wall-time is on
the order of a minute. Opt in with ``pytest -m slow``.

Also covers the Hz ↔ ns⁻¹ unit-audit assertion baked into the script,
so a drift in :data:`fig5_paper.HZ_TO_NS_INV` will fail this test
before any numerics regress.

First-time generation::

    python -m validation.fischer_2024.fig5_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2024.fig5_paper import (
    HZ_TO_NS_INV,
    PAPER_DRIVES_HZ,
    PAPER_DRIVES_NS_INV,
    _assert_unit_audit,
    baseline_path,
    read_baseline,
    run,
)


def test_unit_audit_conversion_pinned() -> None:
    """Hz → ns⁻¹ conversion factor must be 1e-9; per-drive product values
    must round-trip exactly under that conversion.

    Cheap (no kinetic solve), so deliberately NOT slow-marked --- this
    sentinel must run on every CI pass to catch the kind of unit slip
    that hid in ``fig8_xqp_pb.py`` for nine decades.
    """
    assert HZ_TO_NS_INV == 1.0e-9
    for hz, ns_inv in zip(PAPER_DRIVES_HZ, PAPER_DRIVES_NS_INV, strict=True):
        assert ns_inv == pytest.approx(hz * 1.0e-9, rel=0.0, abs=1e-30)
    # Module-level audit assertion must not raise.
    _assert_unit_audit()


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2024.fig5_paper"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(result.E, baseline.E, rtol=0.0, atol=1e-14)
    assert result.drives_hz == baseline.drives_hz
    np.testing.assert_allclose(
        result.drives_ns_inv, baseline.drives_ns_inv,
        rtol=0.0, atol=1e-30,
    )

    np.testing.assert_allclose(
        result.f_thermal, baseline.f_thermal, rtol=1e-6, atol=1e-14,
        err_msg="f_thermal drift",
    )

    for d in result.drives_hz:
        np.testing.assert_allclose(
            result.f_by_drive[d], baseline.f_by_drive[d],
            rtol=1e-6, atol=1e-14,
            err_msg=f"Numerical f drift at drive={d:g} Hz",
        )
        assert result.x_qp_by_drive[d] == pytest.approx(
            baseline.x_qp_by_drive[d], rel=1e-6,
        )
        # Neumann overlays are placeholders, but they should still be
        # bit-stable — no random/parameter drift between runs.
        np.testing.assert_allclose(
            result.f0_by_drive[d], baseline.f0_by_drive[d],
            rtol=1e-10, atol=1e-30,
            err_msg=f"f^(0) placeholder drift at drive={d:g} Hz",
        )
        np.testing.assert_allclose(
            result.f01_by_drive[d], baseline.f01_by_drive[d],
            rtol=1e-10, atol=1e-30,
            err_msg=f"f^(0)+f^(1) placeholder drift at drive={d:g} Hz",
        )
        np.testing.assert_allclose(
            result.f012_by_drive[d], baseline.f012_by_drive[d],
            rtol=1e-10, atol=1e-30,
            err_msg=f"f^(0)+f^(1)+f^(2) placeholder drift at drive={d:g} Hz",
        )
