"""Regression test: Fischer 2024 Fig. 8 paper-target run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- this
run does the τ_ℓ = 0 thermal-phonon Newton solve at 12 bath
temperatures × 3 drive levels = 36 steady-state solves on the 810-bin
F24 grid; total wall-time is on the order of a few minutes. Opt in
with ``pytest -m slow``.

The Hz ↔ ns⁻¹ unit-audit assertion is exercised inside :func:`run`
itself; this test additionally pins the conversion factor and the
paper-target drive list against the baseline metadata so that any
silent unit-slip would fail loudly here.

First-time generation::

    python -m validation.fischer_2024.fig8_paper
"""

from __future__ import annotations

import numpy as np
import pytest

from validation.fischer_2024.fig8_paper import (
    HZ_TO_NS_INV,
    PAPER_DRIVES_HZ,
    baseline_path,
    read_baseline,
    run,
)


def test_unit_audit_pinned() -> None:
    """Hz → ns⁻¹ conversion factor must be 1e-9 exactly.

    Cheap structural assertion (no kinetic solve) — runs even when the
    full slow regression is skipped if invoked directly.
    """
    assert HZ_TO_NS_INV == 1e-9, (
        f"HZ_TO_NS_INV = {HZ_TO_NS_INV} (expected 1e-9). "
        "Paper drive products are quoted in Hz; the kernel takes ns⁻¹. "
        "A drift here would silently slip 9 decades of drive."
    )
    assert PAPER_DRIVES_HZ == (1e-2, 1e-4, 1e-6), (
        f"PAPER_DRIVES_HZ = {PAPER_DRIVES_HZ} (expected (1e-2, 1e-4, 1e-6) Hz). "
        "Stratification is set by the F24 Fig. 8 caption."
    )


@pytest.mark.slow
def test_matches_pinned_baseline() -> None:
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2024.fig8_paper"
        )

    baseline = read_baseline(path)
    result = run()

    np.testing.assert_allclose(
        result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14,
    )
    assert result.drives_hz == baseline.drives_hz
    # drives_ns_inv is derived (drives_hz * 1e-9) but round-trips through
    # the baseline header's lossy %g serialization: float("1e-11") differs
    # from 0.01 * 1e-9 in the last ulp, so exact equality can never hold.
    np.testing.assert_allclose(
        result.drives_ns_inv, baseline.drives_ns_inv, rtol=1e-12, atol=0.0,
    )

    np.testing.assert_allclose(
        result.x_qp_thermal, baseline.x_qp_thermal, rtol=1e-6, atol=1e-14,
        err_msg="Thermal x_qp drift",
    )
    for d_hz in result.drives_hz:
        np.testing.assert_allclose(
            result.x_qp_num_by_drive[d_hz],
            baseline.x_qp_num_by_drive[d_hz],
            rtol=1e-6, atol=1e-14,
            err_msg=f"Numerical x_qp drift at drive={d_hz:g} Hz",
        )
        np.testing.assert_allclose(
            result.x_qp_ana_by_drive[d_hz],
            baseline.x_qp_ana_by_drive[d_hz],
            rtol=1e-10, atol=0.0,
            err_msg=f"Analytic placeholder drift at drive={d_hz:g} Hz",
        )
