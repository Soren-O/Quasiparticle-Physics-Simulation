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

    def test_rejects_zero_n_subgap(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="n_subgap"):
            compute_ac_conductivity(f, ctx, omega_0=1.0, n_subgap=0)

    def test_rejects_negative_n_subgap(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="n_subgap"):
            compute_ac_conductivity(f, ctx, omega_0=1.0, n_subgap=-10)

    def test_n_subgap_propagates_from_quality_factor(self) -> None:
        # Bad n_subgap must also be caught when the call arrives via
        # compute_quality_factor (the delegating caller).
        from qpsim.observables.quality_factor import compute_quality_factor

        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="n_subgap"):
            compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.1, n_subgap=0)

    def test_n_subgap_propagates_from_frequency_shift(self) -> None:
        from qpsim.observables.frequency_shift import compute_frequency_shift

        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="n_subgap"):
            compute_frequency_shift(f, f, ctx, omega_0=1.0, alpha=0.1, n_subgap=0)

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

    def test_sigma_1_thermal_gap_edge_matches_adaptive_reference(self) -> None:
        from scipy.integrate import quad

        gap = 180.0
        temperature = 0.2
        omega_0 = 20.0
        # dE=4 µeV, so omega_0 is exactly five cells and interpolation of the
        # thermal partner occupation adds no grid-offset error.
        E, _ = build_energy_grid(gap, 1.0, 10.0, 405)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        kT = KB_UEV_PER_K * temperature
        f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

        sigma_1, _ = compute_ac_conductivity(f, ctx, omega_0)

        # Independent ξ-space reference: dξ = ρ(E)dE removes the leading
        # gap-edge singularity without using qpsim's cell weights.
        def reference_integrand(xi: float) -> float:
            energy = np.sqrt(xi * xi + gap * gap)
            partner = energy + omega_0
            f_energy = 1.0 / (np.exp(min(energy / kT, 500.0)) + 1.0)
            f_partner = 1.0 / (np.exp(min(partner / kT, 500.0)) + 1.0)
            rho_partner = partner / np.sqrt(partner * partner - gap * gap)
            coherence = 1.0 + gap * gap / (energy * partner)
            return (f_energy - f_partner) * rho_partner * coherence

        reference = (2.0 / omega_0) * quad(
            reference_integrand,
            0.0,
            np.inf,
            epsabs=1e-18,
            epsrel=1e-11,
            limit=500,
        )[0]
        assert sigma_1 == pytest.approx(reference, rel=0.05)

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
