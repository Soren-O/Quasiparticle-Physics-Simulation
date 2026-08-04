# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Regression tests for the 2026-08-03 review, packet P13."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qpsim.collisions.pair_breaking_photon import (
    pair_breaking_photon_collision_components,
    validate_pair_breaking_photon_grid,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext

GAP = 180.0


def _pb_setup(max_factor: float, num: int = 30):
    """Uniform grid over ``[GAP, max_factor*GAP]`` plus its 15-bin photon."""
    E, _ = build_energy_grid(
        gap=GAP,
        energy_min_factor=1.0,
        energy_max_factor=max_factor,
        num_energy_bins=num,
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=GAP)
    return ctx, 15.0 * float(dE[0])


def test_sub_percent_partner_lattice_offset_is_rejected() -> None:
    # max_factor=5.996 puts 2*E_min/dE 0.0096 bins off the reflection
    # lattice: inside the old 1%-of-a-bin tolerance, so the pair channel was
    # accepted silently while solving E_i + E_j = omega_PB - 0.29 ueV.
    ctx, omega = _pb_setup(5.996)
    steps = (omega - 2.0 * float(ctx.E[0])) / float(ctx.dE[0])
    assert 0.005 < abs(steps - round(steps)) < 0.01  # inside the old window
    with pytest.raises(ValueError, match="reflection partners are not grid-aligned"):
        validate_pair_breaking_photon_grid(ctx.E, ctx.dE, GAP, omega)


def test_aligned_pb_grid_keeps_thermal_pair_detailed_balance() -> None:
    # The aligned counterpart still validates, and its pair channel balances
    # to roundoff at a thermal occupation -- the property the tightened
    # lattice guard exists to protect.
    ctx, omega = _pb_setup(6.0)
    contract = validate_pair_breaking_photon_grid(ctx.E, ctx.dE, GAP, omega)
    assert contract.index_shift == 15
    assert contract.fractional_error == pytest.approx(0.0, abs=1e-12)

    kT = 0.2 * KB_UEV_PER_K
    f = 1.0 / (np.exp(ctx.E / kT) + 1.0)
    n_bar = 1.0 / np.expm1(omega / kT)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, gain_pair, loss_pair = pair_breaking_photon_collision_components(
            f, ctx, omega, n_bar, 1.0
        )
    residual = np.max(np.abs(gain_pair - loss_pair * f)) / np.max(np.abs(gain_pair))
    assert residual < 1e-12, residual

# NOTE: test_newline_only_digest_hint_names_the_checkout was removed together with
# validation.paper_parity._newline_only_mismatch_hint, which was reverted during the
# 2026-08-03 recertification: editing paper_parity.py invalidated the fig6
# author-output score, whose producer cannot re-issue it (it fails on the very
# digitizer digest mismatch the hint describes). See docs/REVIEW-2026-08-03-HELD-BACK.md
# "Addendum 2" -- the hint should return with whichever line-ending fix is chosen, and
# with its remedy sentence corrected (the committed blob is CRLF, so the mismatch is not
# a Windows-checkout artifact and `git add --renormalize .` is not the fix).
