"""Tests for qpsim.physics.spectral — DOS, coherence factors, SpectralContext."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.physics.spectral import (
    SpectralContext,
    bcs_anomalous_weight,
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


class TestFiniteInputs:
    @pytest.mark.parametrize(
        "function",
        [
            bcs_density_of_states,
            bcs_anomalous_weight,
            coherence_factor_plus,
            coherence_factor_minus,
        ],
    )
    def test_bcs_primitives_reject_non_finite_energy(self, function) -> None:
        with pytest.raises(ValueError, match="E must contain only finite"):
            function(np.array([1.0, float("nan")]), gap=1.0)

    @pytest.mark.parametrize(
        "function",
        [
            bcs_density_of_states,
            bcs_anomalous_weight,
            coherence_factor_plus,
            coherence_factor_minus,
        ],
    )
    def test_bcs_primitives_reject_non_finite_gap(self, function) -> None:
        with pytest.raises(ValueError, match="gap must be finite"):
            function(np.array([2.0]), gap=float("nan"))

    @pytest.mark.parametrize(
        "function",
        [
            bcs_density_of_states,
            bcs_anomalous_weight,
            coherence_factor_plus,
            coherence_factor_minus,
        ],
    )
    def test_bcs_primitives_reject_negative_gap(self, function) -> None:
        with pytest.raises(ValueError, match="gap must be non-negative"):
            function(np.array([2.0]), gap=-1.0)

    def test_dynes_rejects_non_finite_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must be finite"):
            dynes_density_of_states(np.array([2.0]), gap=1.0, gamma=float("nan"))

    def test_dynes_rejects_negative_gamma(self) -> None:
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            dynes_density_of_states(np.array([2.0]), gap=1.0, gamma=-0.1)

    def test_thermal_weights_reject_non_finite_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature must be finite"):
            thermal_qp_weights(np.array([2.0]), gap=1.0, temperature=float("nan"))


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
        # Uniform grid: first-above-gap dE = mean dE = 0.5, so
        # epsilon = 0.1 · 0.5 = 0.05 and active mask = E ≥ 1.05.
        E = np.array([1.0, 1.5, 2.0])
        dE = np.array([0.5, 0.5, 0.5])
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        assert ctx.active_mask.tolist() == [False, True, True]

    def test_active_mask_uses_local_dE_on_piecewise_grid(self) -> None:
        # Piecewise grid: fine bins near the gap (dE=0.01), wide bins
        # far from the gap (dE=10.0). Pre-Phase-5c the threshold used
        # mean(dE) ≈ 5, which excluded the entire fine sub-band. The
        # corrected epsilon uses the bin spacing local to the gap edge.
        E = np.concatenate([
            np.linspace(1.005, 1.04, 4),  # 4 fine bins near gap
            np.linspace(11.0, 41.0, 4),   # 4 wide bins far from gap
        ])
        dE = np.array([0.01] * 4 + [10.0] * 4)
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        # Local dE = dE of first bin > gap = 0.01.
        # epsilon = 0.1 · 0.01 = 0.001, so active = E ≥ 1.001.
        # All bins satisfy this: full active mask.
        assert ctx.active_mask.tolist() == [True] * 8

    def test_active_mask_immune_to_tiny_far_tail_bin(self) -> None:
        # A tiny bin far from the gap must NOT shrink epsilon for
        # near-gap bins (the bug a naïve global ``min(dE)`` would have
        # introduced). Here the near-gap bin spacing is 0.5 and a
        # 1e-4-wide bin lives at the far tail. Active threshold should
        # still be set by the near-gap dE = 0.5, rejecting E=1.0.
        E = np.array([1.0, 1.5, 2.0, 100.0, 100.0001])
        dE = np.array([0.5, 0.5, 0.5, 100.0, 1e-4])
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        # epsilon = 0.1 · 0.5 = 0.05 (from first-above-gap bin),
        # not 0.1 · 1e-4 = 1e-5 (which would mark E=1.0 as active).
        assert ctx.active_mask.tolist() == [False, True, True, True, True]

    def test_rejects_mismatched_grid_sizes(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            SpectralContext(E_bins=np.array([1.0, 2.0]), dE_bins=np.array([1.0]), gap=1.0)

    @pytest.mark.parametrize(
        ("E", "dE", "message"),
        [
            (np.array([float("nan")]), np.array([1.0]), "E_bins"),
            (np.array([2.0]), np.array([float("inf")]), "dE_bins"),
        ],
    )
    def test_rejects_non_finite_grids(
        self, E: np.ndarray, dE: np.ndarray, message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SpectralContext(E_bins=E, dE_bins=dE, gap=1.0)

    def test_rejects_non_finite_gap_rebuild(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([2.0]), dE_bins=np.array([1.0]), gap=1.0,
        )

        with pytest.raises(ValueError, match="new_gap must be finite"):
            ctx.maybe_rebuild(float("nan"))
        assert ctx.gap == 1.0

    @pytest.mark.parametrize(
        ("E", "dE", "message"),
        [
            (np.array([2.0, 1.0]), np.ones(2), "strictly increasing"),
            (np.array([1.0, 1.0]), np.ones(2), "strictly increasing"),
            (np.array([1.0, 2.0]), np.array([1.0, 0.0]), "must be positive"),
            (np.array([1.0, 2.0]), np.array([1.0, -1.0]), "must be positive"),
        ],
    )
    def test_rejects_non_physical_grids(
        self, E: np.ndarray, dE: np.ndarray, message: str,
    ) -> None:
        with pytest.raises(ValueError, match=message):
            SpectralContext(E_bins=E, dE_bins=dE, gap=0.5)

    def test_rejects_grid_without_above_gap_support(self) -> None:
        with pytest.raises(ValueError, match="at least one energy bin above gap"):
            SpectralContext(
                E_bins=np.array([1.0, 2.0]),
                dE_bins=np.ones(2),
                gap=2.0,
            )

    @pytest.mark.parametrize(
        ("name", "kwargs"),
        [
            ("dynes_gamma", {"dynes_gamma": -0.1}),
            ("diffusion_coefficient", {"diffusion_coefficient": -1.0}),
            ("rebuild_tolerance", {"rebuild_tolerance": -1.0}),
            ("active_margin_factor", {"active_margin_factor": -1.0}),
        ],
    )
    def test_rejects_negative_configuration(
        self, name: str, kwargs: dict[str, float],
    ) -> None:
        with pytest.raises(ValueError, match=rf"{name} must be non-negative"):
            SpectralContext(
                E_bins=np.array([1.0, 2.0]),
                dE_bins=np.ones(2),
                gap=0.5,
                **kwargs,
            )
