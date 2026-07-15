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
        gap=gap, energy_min_factor=1.0, energy_max_factor=20.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    return SpectralContext(E_bins=E, dE_bins=dE, gap=gap)


class TestQpNumberDensity:
    def test_zero_f_gives_zero(self) -> None:
        ctx = _ctx()
        f = np.zeros(ctx.E.size)
        assert qp_number_density(f, ctx, rho_F=1.0e28) == 0.0

    def test_scales_with_rho_F(self) -> None:
        ctx = _ctx()
        f = 0.1 * np.ones(ctx.E.size)
        n1 = qp_number_density(f, ctx, rho_F=1.0e28)
        n2 = qp_number_density(f, ctx, rho_F=7.0e28)
        assert n2 == pytest.approx(7.0 * n1)

    @pytest.mark.parametrize("rho_F", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_rho_F(self, rho_F: float) -> None:
        ctx = _ctx()
        f = np.zeros(ctx.E.size)
        with pytest.raises(ValueError, match="rho_F"):
            qp_number_density(f, ctx, rho_F=rho_F)

    def test_converts_uev_integral_to_eV(self) -> None:
        ctx = _ctx()
        f = 0.01 * np.ones(ctx.E.size)
        rho_F = 1.74e28  # eV⁻¹ m⁻³
        integral_uev = qp_fraction(f, ctx, delta_0=ctx.gap) * ctx.gap
        expected = 4.0 * rho_F * integral_uev / 1.0e6
        assert qp_number_density(f, ctx, rho_F) == pytest.approx(expected)

    def test_rejects_silent_legacy_per_uev_input(self) -> None:
        ctx = _ctx()
        with pytest.raises(ValueError, match="implausibly small"):
            qp_number_density(np.zeros(ctx.E.size), ctx, 1.74e22)

    def test_constant_occupation_uses_exact_bcs_dos_measure(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=3.0,
            num_energy_bins=40,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        f0 = 0.02
        f = np.full(E.size, f0)
        rho_F = 1.74e28

        # ∫_Δ^(3Δ) E/sqrt(E²-Δ²) dE = sqrt((3Δ)²-Δ²).
        exact_integral_uev = f0 * np.sqrt((3.0 * gap) ** 2 - gap**2)
        expected = 4.0 * rho_F * exact_integral_uev / 1.0e6
        assert qp_number_density(f, ctx, rho_F) == pytest.approx(
            expected, rel=1e-14,
        )

    def test_rejects_grid_that_starts_above_the_gap(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 1.01, 3.0, 200)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )

        with pytest.raises(ValueError, match=r"does not cover.*BCS lower bound"):
            qp_number_density(np.full(E.size, 0.01), ctx, rho_F=1.74e28)

    def test_thermal_density_grows_with_temperature(self) -> None:
        # Monotonicity check: x_qp must be strictly increasing in T_bath
        # for a Fermi-Dirac occupation.
        from itertools import pairwise

        ctx = _ctx(T_c=1.2, num=400)
        densities = []
        for T in [0.05, 0.1, 0.15, 0.2]:
            kT = KB_UEV_PER_K * T
            f = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
            densities.append(qp_number_density(f, ctx, rho_F=1.0e28))
        for a, b in pairwise(densities):
            assert b > a


class TestQpFraction:
    def test_independent_of_rho_F(self) -> None:
        # qp_fraction cancels ρ_F internally; it takes only Δ₀ in the
        # denominator.
        ctx = _ctx()
        f = 0.01 * np.ones(ctx.E.size)
        x1 = qp_fraction(f, ctx, delta_0=ctx.gap)
        # qp_number_density uses ρ_F per eV, so its dimensional denominator
        # uses Δ₀ converted from µeV to eV.
        rho_F = 1.0e28
        n = qp_number_density(f, ctx, rho_F=rho_F)
        delta_0_eV = ctx.gap / 1.0e6
        assert x1 == pytest.approx(n / (4.0 * rho_F * delta_0_eV))

    def test_zero_f_gives_zero(self) -> None:
        ctx = _ctx()
        assert qp_fraction(np.zeros(ctx.E.size), ctx, delta_0=ctx.gap) == 0.0

    def test_constant_occupation_matches_exact_bcs_primitive(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 1.0, 3.0, 40)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        f0 = 0.02

        expected = f0 * np.sqrt((3.0 * gap) ** 2 - gap**2) / gap
        assert qp_fraction(np.full(E.size, f0), ctx, gap) == pytest.approx(
            expected, rel=1e-14,
        )

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
