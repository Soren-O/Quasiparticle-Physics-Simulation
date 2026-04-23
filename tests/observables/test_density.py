"""Tests for qpsim.observables.density."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.density import qp_fraction, qp_number_density
from qpsim.physics.spectral import SpectralContext


def _ctx(T_c: float = 1.2, num: int = 100) -> SpectralContext:
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.001, energy_max_factor=20.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    return SpectralContext(E_bins=E, dE_bins=dE, gap=gap)


class TestQpNumberDensity:
    def test_zero_f_gives_zero(self) -> None:
        ctx = _ctx()
        f = np.zeros(ctx.E.size)
        assert qp_number_density(f, ctx, rho_F=1.0) == 0.0

    def test_scales_with_rho_F(self) -> None:
        ctx = _ctx()
        f = 0.1 * np.ones(ctx.E.size)
        n1 = qp_number_density(f, ctx, rho_F=1.0)
        n2 = qp_number_density(f, ctx, rho_F=7.0)
        assert n2 == pytest.approx(7.0 * n1)

    def test_rejects_non_positive_rho_F(self) -> None:
        ctx = _ctx()
        f = np.zeros(ctx.E.size)
        with pytest.raises(ValueError, match="rho_F"):
            qp_number_density(f, ctx, rho_F=0.0)

    def test_thermal_density_grows_with_temperature(self) -> None:
        # Monotonicity check: x_qp must be strictly increasing in T_bath
        # for a Fermi-Dirac occupation.
        from itertools import pairwise

        ctx = _ctx(T_c=1.2, num=400)
        densities = []
        for T in [0.05, 0.1, 0.15, 0.2]:
            kT = KB_UEV_PER_K * T
            f = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
            densities.append(qp_number_density(f, ctx, rho_F=1.0))
        for a, b in pairwise(densities):
            assert b > a


class TestQpFraction:
    def test_independent_of_rho_F(self) -> None:
        # qp_fraction cancels ρ_F internally; it takes only Δ₀ in the
        # denominator.
        ctx = _ctx()
        f = 0.01 * np.ones(ctx.E.size)
        x1 = qp_fraction(f, ctx, delta_0=ctx.gap)
        # qp_number_density scales with ρ_F; the fraction should equal
        # qp_number_density(f, ctx, ρ_F) / (4 ρ_F Δ₀).
        n = qp_number_density(f, ctx, rho_F=1.0)
        assert x1 == pytest.approx(n / (4.0 * 1.0 * ctx.gap))

    def test_zero_f_gives_zero(self) -> None:
        ctx = _ctx()
        assert qp_fraction(np.zeros(ctx.E.size), ctx, delta_0=ctx.gap) == 0.0

    def test_rejects_non_positive_delta_0(self) -> None:
        ctx = _ctx()
        f = np.zeros(ctx.E.size)
        with pytest.raises(ValueError, match="delta_0"):
            qp_fraction(f, ctx, delta_0=0.0)

    def test_dimensionless_and_small_in_thermal_regime(self) -> None:
        # A cold Al-like film has x_qp ≪ 1 (literally Boltzmann-suppressed).
        ctx = _ctx(T_c=1.2, num=400)
        kT = KB_UEV_PER_K * 0.1
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        x = qp_fraction(f_FD, ctx, delta_0=ctx.gap)
        assert 0.0 < x < 1e-3
