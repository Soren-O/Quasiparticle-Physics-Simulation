"""Tests for qpsim.physics.spectral — DOS, coherence factors, SpectralContext."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import (
    SpectralContext,
    bcs_anomalous_weight,
    bcs_density_of_states,
    coherence_factor_minus,
    coherence_factor_plus,
    dynes_density_of_states,
    fermi_dirac_occupation,
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


    def test_representable_cold_tail_is_not_floored(self) -> None:
        E = np.array([360.0])
        gap = 180.0
        temperature = 0.007
        exponent = E / (KB_UEV_PER_K * temperature)
        exp_negative = np.exp(-exponent)
        fermi = exp_negative / (1.0 + exp_negative)
        expected = bcs_density_of_states(E, gap) * fermi

        got = thermal_qp_weights(E, gap=gap, temperature=temperature)

        np.testing.assert_allclose(got, expected, rtol=2e-14)
        assert 0.0 < got[0] < np.exp(-500.0)

    def test_shared_fermi_evaluator_handles_limits(self) -> None:
        E = np.array([0.0, 360.0, 800.0 * KB_UEV_PER_K * 0.007])
        got = fermi_dirac_occupation(E, 0.007)
        exponent = E / (KB_UEV_PER_K * 0.007)
        exp_negative = np.exp(-exponent)
        expected = exp_negative / (1.0 + exp_negative)

        np.testing.assert_array_equal(got, expected)
        np.testing.assert_array_equal(
            fermi_dirac_occupation(E, 0.0),
            np.zeros_like(E),
        )

    def test_shared_fermi_evaluator_rejects_complex_energy(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            fermi_dirac_occupation(np.array([1.0 + 2.0j]), 0.1)

    def test_shared_fermi_evaluator_preserves_subnormal_input_ratio(self) -> None:
        smallest = np.nextafter(0.0, 1.0)
        energy = np.array([smallest, 100.0 * smallest])
        exponent = (energy / smallest) / KB_UEV_PER_K
        expected = np.exp(-exponent) / (1.0 + np.exp(-exponent))

        got = fermi_dirac_occupation(energy, smallest)

        np.testing.assert_allclose(got, expected, rtol=2e-15)


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
        dE = np.full_like(E, E[1] - E[0])
        ctx = SpectralContext(E, dE, gap=1.0, rebuild_tolerance=1e-3)
        assert not ctx.maybe_rebuild(1.0 + 1e-6)
        assert ctx.gap == 1.0

    def test_rebuild_triggered_outside_tolerance(self) -> None:
        E = np.linspace(1.01, 5.0, 10)
        dE = np.full_like(E, E[1] - E[0])
        ctx = SpectralContext(E, dE, gap=1.0, rebuild_tolerance=1e-3)
        assert ctx.maybe_rebuild(2.0)
        assert ctx.gap == 2.0

    def test_diffusion_coefficient_legacy_form(self) -> None:
        # D(E) = D₀ √(1 − (Δ/E)²)
        E = np.array([2.0, 5.0])
        dE = np.array([3.0, 3.0])
        ctx = SpectralContext(E, dE, gap=1.0, diffusion_coefficient=1.0)
        expected = np.sqrt(1.0 - (1.0 / E) ** 2)
        np.testing.assert_allclose(ctx.D_E, expected)

    def test_diffusion_coefficient_zero_when_D0_zero(self) -> None:
        E = np.array([2.0, 5.0])
        dE = np.array([3.0, 3.0])
        ctx = SpectralContext(E, dE, gap=1.0, diffusion_coefficient=0.0)
        np.testing.assert_allclose(ctx.D_E, 0.0)

    def test_active_mask(self) -> None:
        # The first cell straddles the BCS gap despite its center being at it.
        E = np.array([1.0, 1.5, 2.0])
        dE = np.array([0.5, 0.5, 0.5])
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        assert ctx.active_mask.tolist() == [True, True, True]

    def test_active_mask_never_excludes_supported_near_gap_bin(self) -> None:
        # The former 0.1*dE margin excluded E=1.01 despite finite capacity.
        E = np.array([0.91, 1.01, 1.11])
        dE = np.array([0.1, 0.1, 0.1])
        ctx = SpectralContext(E, dE, gap=1.0, active_margin_factor=0.1)
        np.testing.assert_array_equal(ctx.active_mask, ctx.cell_weights > 0.0)
        assert ctx.active_mask.tolist() == [False, True, True]

    def test_dynes_subgap_support_is_active(self) -> None:
        E = np.array([0.5, 0.99, 1.01, 1.5])
        dE = np.full(E.size, 0.25)
        ctx = SpectralContext(E, dE, gap=1.0, dynes_gamma=0.01)
        assert np.all(ctx.rho > 0.0)
        assert np.all(ctx.active_mask)

    def test_active_support_includes_cell_cut_by_gap_edge(self) -> None:
        E = np.array([0.9, 1.3, 1.7])
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        finite_volume_support = bcs_dos_cell_weights(E, dE, gap=1.0) > 0.0

        np.testing.assert_array_equal(ctx.active_mask, finite_volume_support)
        assert ctx.rho[0] == 0.0
        assert ctx.cell_weights[0] > 0.0
        assert ctx.cell_density[0] == pytest.approx(
            ctx.cell_weights[0] / ctx.dE[0]
        )

    def test_cut_cell_coherence_is_exact_product_measure_average(self) -> None:
        E = np.array([0.9, 1.3, 1.7])
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        ratio = np.divide(
            ctx.cell_anomalous_density,
            ctx.cell_density,
            out=np.zeros_like(E),
            where=ctx.cell_density > 0.0,
        )
        product = ratio[:, None] * ratio[None, :]

        np.testing.assert_allclose(ctx.K_plus, 1.0 + product)
        np.testing.assert_allclose(ctx.K_minus, 1.0 - product)
        assert ctx.K_minus[0, 1] > 0.0

    def test_cell_measure_cache_is_read_only(self) -> None:
        ctx = SpectralContext(
            np.array([0.9, 1.3, 1.7]), np.full(3, 0.4), gap=1.0,
        )
        for values in (
            ctx.cell_weights,
            ctx.cell_density,
            ctx.cell_anomalous_density,
            ctx.active_mask,
        ):
            assert not values.flags.writeable
            with pytest.raises(ValueError):
                values[0] = 0.0

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

    def test_last_cell_cut_by_gap_is_valid_without_above_gap_center(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([1.0, 2.0]),
            dE_bins=np.ones(2),
            gap=2.0,
        )
        np.testing.assert_array_equal(ctx.active_mask, [False, True])
        assert ctx.rho[1] == 0.0
        assert ctx.cell_weights[1] > 0.0

    def test_rejects_grid_without_represented_spectral_capacity(self) -> None:
        with pytest.raises(ValueError, match="positive represented spectral capacity"):
            SpectralContext(
                E_bins=np.array([1.0, 2.0]),
                dE_bins=np.ones(2),
                gap=2.6,
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


class TestRebuildExceptionSafety:
    """2026-07-19 audit: a failed _rebuild must not leave a torn object."""

    def test_failed_rebuild_leaves_context_intact_and_retryable(self) -> None:
        E = np.linspace(1.01, 5.0, 10)
        dE = np.full_like(E, E[1] - E[0])
        ctx = SpectralContext(E, dE, gap=0.5, diffusion_coefficient=1.0)
        before = {
            "gap": ctx.gap,
            "rho": ctx.rho.copy(),
            "cell_weights": ctx.cell_weights.copy(),
            "cell_density": ctx.cell_density.copy(),
            "K_plus": ctx.K_plus.copy(),
            "K_minus": ctx.K_minus.copy(),
            "D_E": ctx.D_E.copy(),
            "active_mask": ctx.active_mask.copy(),
        }

        # A gap above the whole grid has zero represented capacity: the
        # rebuild must raise...
        with pytest.raises(ValueError, match="positive represented spectral"):
            ctx.maybe_rebuild(10.0)

        # ...and leave EVERY field at its pre-call value (the pre-fix
        # version committed gap/weights first, mixing two gaps).
        assert ctx.gap == before["gap"]
        np.testing.assert_array_equal(ctx.rho, before["rho"])
        np.testing.assert_array_equal(ctx.cell_weights, before["cell_weights"])
        np.testing.assert_array_equal(ctx.cell_density, before["cell_density"])
        np.testing.assert_array_equal(ctx.K_plus, before["K_plus"])
        np.testing.assert_array_equal(ctx.K_minus, before["K_minus"])
        np.testing.assert_array_equal(ctx.D_E, before["D_E"])
        np.testing.assert_array_equal(ctx.active_mask, before["active_mask"])

        # A retry of the same invalid gap must raise again — not silently
        # no-op because the failed gap was committed as current.
        with pytest.raises(ValueError, match="positive represented spectral"):
            ctx.maybe_rebuild(10.0)

        # And a valid rebuild afterwards still works normally.
        assert ctx.maybe_rebuild(0.6) is True
        assert ctx.gap == 0.6
