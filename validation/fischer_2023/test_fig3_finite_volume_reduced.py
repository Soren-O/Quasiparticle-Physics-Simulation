"""Fast driven regression for the matched finite-volume collision measure."""

from __future__ import annotations

import numpy as np

from validation.fischer_2023.fig3_solve import _build_grid_and_spectral, solve


def test_reduced_driven_branch_survives_exact_gap_edge_measure() -> None:
    """Guard the failure mode of the rejected exact-weight-only G1 patch.

    On this 81-cell proxy, mixing exact QP capacity with point photon partner
    rates drove the ratio-zero peak down to ``1.52e-17`` and made the finite
    escape-time result indistinguishable.  The matched cell-average partner
    density retains a resolved driven state and a monotone bottleneck response.
    """
    result = solve(
        num_bins=81,
        paper_ratios=(0.0, 0.1),
        continuation_ratios=(0.1,),
    )
    curves = np.asarray(result["f_ratios"], dtype=float)
    peaks = np.max(curves, axis=1)
    _, _, spectral = _build_grid_and_spectral(num_bins=81)
    populations = curves @ spectral.cell_weights

    assert np.all(np.isfinite(curves))
    assert np.all((curves >= 0.0) & (curves <= 1.0))
    assert 5e-11 < peaks[0] < 2e-10
    assert peaks[1] > 1.2 * peaks[0]
    assert populations[1] > 1.2 * populations[0]
    assert float(np.max(result["qp_backward_error"])) < 1e-5
    assert float(result["phonon_backward_error"][1]) < 1e-5
