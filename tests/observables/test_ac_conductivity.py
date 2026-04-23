"""Tests for qpsim.observables.ac_conductivity."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.physics.spectral import SpectralContext


def _thermal_ctx_and_f(T_bath: float = 0.3, T_c: float = 1.2, num: int = 200):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.001, energy_max_factor=10.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    kT = KB_UEV_PER_K * T_bath
    f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return ctx, f


class TestComputeAcConductivity:
    def test_rejects_zero_omega(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="omega_0"):
            compute_ac_conductivity(f, ctx, omega_0=0.0)

    def test_rejects_negative_omega(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="omega_0"):
            compute_ac_conductivity(f, ctx, omega_0=-1.0)

    def test_rejects_dynes_context(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        # Rebuild ctx with Dynes broadening.
        ctx_dynes = SpectralContext(
            E_bins=ctx.E, dE_bins=ctx.dE, gap=ctx.gap, dynes_gamma=0.01
        )
        with pytest.raises(ValueError, match="Dynes"):
            compute_ac_conductivity(f, ctx_dynes, omega_0=1.0)

    def test_returns_finite_thermal(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        s1, s2 = compute_ac_conductivity(f, ctx, omega_0=1.0)
        assert np.isfinite(s1)
        assert np.isfinite(s2)

    def test_sigma_1_vanishes_at_T_zero(self) -> None:
        # At T → 0 with no photon drive, f ≈ 0 above the gap ⇒ σ₁ → 0.
        ctx, f_hot = _thermal_ctx_and_f(T_bath=0.01, num=200)
        s1, _ = compute_ac_conductivity(f_hot, ctx, omega_0=0.5)
        assert s1 == pytest.approx(0.0, abs=1e-4)

    def test_sigma_2_positive_at_low_T(self) -> None:
        # σ₂ is the kinetic-inductance response; > 0 in the superconducting state.
        ctx, f_hot = _thermal_ctx_and_f(T_bath=0.01, num=200)
        _, s2 = compute_ac_conductivity(f_hot, ctx, omega_0=0.5)
        assert s2 > 0.0

    def test_sigma_2_monotonically_decreases_with_T(self) -> None:
        # σ₂ decreases as QP density rises (kinetic inductance weakens).
        ctx_low, f_low = _thermal_ctx_and_f(T_bath=0.1)
        _ctx_hi, f_hi = _thermal_ctx_and_f(T_bath=0.8)
        # Same gap so the two contexts share grid/gap; only f differs.
        _, s2_low = compute_ac_conductivity(f_low, ctx_low, omega_0=1.0)
        _, s2_hi = compute_ac_conductivity(f_hi, ctx_low, omega_0=1.0)
        assert s2_low > s2_hi
