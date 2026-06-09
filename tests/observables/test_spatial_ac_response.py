from __future__ import annotations

import numpy as np
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.frequency_shift import compute_frequency_shift
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.observables.spatial_ac_response import compute_current_weighted_ac_response
from qpsim.physics.spectral import SpectralContext


def _ctx() -> SpectralContext:
    gap = 180.0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=5.0,
        num_energy_bins=32,
    )
    return SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
    )


def test_uniform_spatial_response_matches_lumped_observables() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    f = 1e-3 * np.exp(-((ctx.E - 2.0 * ctx.gap) / (0.25 * ctx.gap)) ** 2)
    x_um = np.linspace(0.0, 10.0, 5)
    weights = np.ones_like(x_um)
    omega_0 = 22.0
    alpha = 0.08

    response = compute_current_weighted_ac_response(
        np.repeat(f[:, None], x_um.size, axis=1),
        f_ref,
        x_um,
        ctx,
        omega_0,
        alpha=alpha,
        current_weights=weights,
        full_current_integral_um=10.0,
    )

    assert response.frac_freq_shift == compute_frequency_shift(
        f,
        f_ref,
        ctx,
        omega_0,
        alpha,
    )
    assert response.qi_qp == compute_quality_factor(f, ctx, omega_0, alpha)


def test_full_current_integral_dilutes_short_strip_response() -> None:
    ctx = _ctx()
    f_ref = np.zeros_like(ctx.E)
    f = 1e-3 * np.exp(-((ctx.E - 2.0 * ctx.gap) / (0.25 * ctx.gap)) ** 2)
    x_um = np.linspace(0.0, 10.0, 5)
    weights = np.ones_like(x_um)

    strip_only = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        22.0,
        alpha=0.08,
        current_weights=weights,
        full_current_integral_um=10.0,
    )
    diluted = compute_current_weighted_ac_response(
        f,
        f_ref,
        x_um,
        ctx,
        22.0,
        alpha=0.08,
        current_weights=weights,
        full_current_integral_um=20.0,
    )

    assert diluted.frac_freq_shift == 0.5 * strip_only.frac_freq_shift
    assert diluted.inverse_qi_qp == 0.5 * strip_only.inverse_qi_qp
