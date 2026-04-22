"""Tests for qpsim.physics.spectral — DOS, coherence factors, SpectralContext."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.physics.spectral import (
    SpectralContext,
    bcs_density_of_states,
    coherence_factor_minus,
    coherence_factor_plus,
    dynes_density_of_states,
    thermal_qp_weights,
)


class TestBcsDos:
    def test_zero_below_gap(self) -> None:
        E = np.array([0.5, 0.9])
        assert np.all(bcs_density_of_states(E, gap=1.0) == 0.0)

    def test_singular_behavior_at_edge(self) -> None:
        rho = bcs_density_of_states(np.array([1.001, 10.0]), gap=1.0)
        # ρ → ∞ as E → Δ⁺; far above gap, ρ → 1.
        assert rho[0] > rho[1]
        assert rho[1] == pytest.approx(1.0, rel=0.02)

    def test_normal_state_limit(self) -> None:
        # Δ = 0 ⇒ ρ(E) = 1 for all E > 0.
        rho = bcs_density_of_states(np.array([1.0, 2.0, 3.0]), gap=0.0)
        np.testing.assert_allclose(rho, 1.0)


class TestDynesDos:
    def test_falls_back_to_bcs_at_zero_gamma(self) -> None:
        E = np.array([1.5, 2.0, 3.0])
        np.testing.assert_allclose(
            dynes_density_of_states(E, gap=1.0, gamma=0.0),
            bcs_density_of_states(E, gap=1.0),
        )

    def test_smooths_gap_edge(self) -> None:
        # With γ > 0, DOS is finite right at E = Δ (unlike BCS).
        rho = dynes_density_of_states(np.array([1.0]), gap=1.0, gamma=0.01)
        assert 0 < rho[0] < np.inf

    def test_non_negative(self) -> None:
        E = np.linspace(0.1, 5.0, 50)
        assert np.all(dynes_density_of_states(E, gap=1.0, gamma=0.05) >= 0)


class TestCoherenceFactors:
    def test_shapes(self) -> None:
        E = np.array([1.5, 2.0, 3.0])
        assert coherence_factor_plus(E, gap=1.0).shape == (3, 3)
        assert coherence_factor_minus(E, gap=1.0).shape == (3, 3)

    def test_plus_exceeds_one(self) -> None:
        # K⁺ = 1 + Δ²/(E_i E_j) > 1 strictly for Δ > 0 and finite E.
        K = coherence_factor_plus(np.array([1.5, 2.0]), gap=1.0)
        assert np.all(K > 1.0)

    def test_minus_non_negative(self) -> None:
        K = coherence_factor_minus(np.array([1.5, 2.0]), gap=1.0)
        assert np.all(K >= 0)

    def test_normal_state_limit(self) -> None:
        # Δ = 0 ⇒ K⁺ = K⁻ = 1.
        E = np.array([1.0, 2.0])
        np.testing.assert_allclose(coherence_factor_plus(E, gap=0.0), 1.0)
        np.testing.assert_allclose(coherence_factor_minus(E, gap=0.0), 1.0)

    def test_symmetry(self) -> None:
        # Both K± are symmetric under i↔j (depend on E_i E_j only).
        E = np.array([1.5, 2.0, 3.5])
        K_p = coherence_factor_plus(E, gap=1.0)
        K_m = coherence_factor_minus(E, gap=1.0)
        np.testing.assert_allclose(K_p, K_p.T)
        np.testing.assert_allclose(K_m, K_m.T)


class TestThermalQpWeights:
    def test_zero_at_T_zero(self) -> None:
        w = thermal_qp_weights(np.array([1.5, 2.0, 3.0]), gap=1.0, temperature=0.0)
        np.testing.assert_allclose(w, 0.0)

    def test_non_negative(self) -> None:
        E = np.linspace(1.01, 10.0, 50)
        assert np.all(thermal_qp_weights(E, gap=1.0, temperature=0.5) >= 0)

    def test_zero_below_gap(self) -> None:
        # ρ = 0 below the gap ⇒ weight = 0 there regardless of T.
        E = np.array([0.5, 0.9])
        assert np.all(thermal_qp_weights(E, gap=1.0, temperature=1.0) == 0.0)


class TestSpectralContext:
    def test_build_and_query(self) -> None:
        E = np.linspace(1.01, 5.0, 20)
        dE = np.full_like(E, (E[-1] - E[0]) / 19)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=1.0)
        assert ctx.gap == 1.0
        assert ctx.rho.shape == E.shape
        assert ctx.K_plus.shape == (20, 20)
        assert ctx.K_minus.shape == (20, 20)

    def test_rebuild_skipped_within_tolerance(self) -> None:
        E = np.linspace(1.01, 5.0, 10)
        dE = np.full_like(E, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0, rebuild_tolerance=1e-3)
        assert not ctx.maybe_rebuild(1.0 + 1e-6)
        assert ctx.gap == 1.0

    def test_rebuild_triggered_outside_tolerance(self) -> None:
        E = np.linspace(1.01, 5.0, 10)
        dE = np.full_like(E, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0, rebuild_tolerance=1e-3)
        assert ctx.maybe_rebuild(2.0)
        assert ctx.gap == 2.0

    def test_diffusion_coefficient_legacy_form(self) -> None:
        # D(E) = D₀ √(1 − (Δ/E)²)
        E = np.array([2.0, 5.0])
        dE = np.array([1.0, 1.0])
        ctx = SpectralContext(E, dE, gap=1.0, diffusion_coefficient=1.0)
        expected = np.sqrt(1.0 - (1.0 / E) ** 2)
        np.testing.assert_allclose(ctx.D_E, expected)

    def test_diffusion_coefficient_zero_when_D0_zero(self) -> None:
        E = np.array([2.0, 5.0])
        dE = np.array([1.0, 1.0])
        ctx = SpectralContext(E, dE, gap=1.0, diffusion_coefficient=0.0)
        np.testing.assert_allclose(ctx.D_E, 0.0)

    def test_active_mask(self) -> None:
        # margin = active_margin_factor · mean(dE) = 0.1 · 0.5 = 0.05
        # so active mask = E ≥ 1.05
        E = np.array([1.0, 1.5, 2.0])
        dE = np.array([0.5, 0.5, 0.5])
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        assert ctx.active_mask.tolist() == [False, True, True]

    def test_rejects_mismatched_grid_sizes(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            SpectralContext(E_bins=np.array([1.0, 2.0]), dE_bins=np.array([1.0]), gap=1.0)
