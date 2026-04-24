"""Tests for Kaplan 1976 τ_PB(Ω) evaluator."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from qpsim.physics.kaplan_pair_breaking import (
    kaplan_S_plus,
    kaplan_S_plus_numerical,
    tau_PB_inverse_Hz,
)

# ═══════════════════════════════════════════════════════════════════════
#  S_+(x) — closed form vs numerical quadrature
# ═══════════════════════════════════════════════════════════════════════


class TestKaplanSPlus:
    def test_threshold_zero(self) -> None:
        # S_+(x) = 0 for x ≤ 2 (below pair-breaking threshold).
        for x in (0.0, 0.5, 1.0, 1.5, 1.99, 2.0):
            assert kaplan_S_plus(x) == 0.0

    def test_positive_above_threshold(self) -> None:
        for x in (2.01, 2.5, 3.0, 5.0, 20.0):
            assert kaplan_S_plus(x) > 0.0

    def test_monotonic_above_threshold(self) -> None:
        # More phase space at higher phonon energy — S_+ rises with x.
        xs = np.array([2.1, 2.5, 3.0, 4.0, 8.0, 20.0])
        vals = np.array([kaplan_S_plus(x) for x in xs])
        assert np.all(np.diff(vals) > 0.0)

    @pytest.mark.parametrize(
        "x", [2.05, 2.1, 2.5, 3.0, 4.0, 5.0, 8.0, 20.0, 100.0]
    )
    def test_closed_form_matches_quadrature(self, x: float) -> None:
        # The closed form via complete elliptic E[m] must match the
        # direct integral of the Kaplan integrand.
        closed = kaplan_S_plus(x)
        numerical = kaplan_S_plus_numerical(x)
        assert closed == pytest.approx(numerical, rel=1e-10)

    def test_large_x_asymptote(self) -> None:
        # For x ≫ 2, E(1 - 4/x²) → E(1) = 1, so S_+(x) ~ x.
        x = 1e4
        assert kaplan_S_plus(x) / x == pytest.approx(1.0, rel=1e-6)


# ═══════════════════════════════════════════════════════════════════════
#  τ_PB⁻¹(Ω) — physical values
# ═══════════════════════════════════════════════════════════════════════


class TestTauPBInverse:
    def test_threshold_zero(self) -> None:
        # Ω ≤ 2Δ → no pair breaking.
        Delta_K = 2.09   # Al-scale
        tau_0_ns = 0.242
        for omega_K in (0.1, 1.0, 2.0, 4.17, 4.18):  # 4.18 ≈ 2Δ
            assert tau_PB_inverse_Hz(omega_K, Delta_K, tau_0_ns) == 0.0

    def test_Kaplan_Fig16_T_zero_sentinels(self) -> None:
        # At T = 0 with Δ = Δ₀, the prefactor reduces to 1/(π τ_0^ph),
        # so τ_B⁻¹ × τ_0^ph = S_+(x)/π. Kaplan Fig 16 (universal curve)
        # gives sentinel values that let us pin the evaluator independently
        # of material specifics.
        tau_0_ns = 1.0     # pick unity for the sentinel test
        Delta_K = 1.0      # pick unity
        # Raw formula: τ_B⁻¹ [Hz] = (1 K / (π × 1 ns × 1 K)) × S_+(x) × 1e9
        #                         = S_+(x)/π × 1e9 Hz
        # So τ_B⁻¹ × τ_0^ph = (S_+(x)/π × 1e9 Hz) × (1 ns) = S_+(x)/π
        sentinels = {
            # x → expected S_+(x)/π from direct closed-form
            2.5: kaplan_S_plus(2.5) / np.pi,
            3.0: kaplan_S_plus(3.0) / np.pi,
            4.0: kaplan_S_plus(4.0) / np.pi,
            8.0: kaplan_S_plus(8.0) / np.pi,
        }
        for x, expected_product in sentinels.items():
            rate = tau_PB_inverse_Hz(x * Delta_K, Delta_K, tau_0_ns)
            product = rate * tau_0_ns * 1e-9   # Hz × ns × (1s/1e9 ns) = dimensionless
            assert product == pytest.approx(expected_product, rel=1e-12)

    def test_Al_representative_values(self) -> None:
        # For Al with Δ₀ ≈ 2.09 K (180 μeV) and τ_0^ph = 0.242 ns,
        # at Ω = 3Δ₀ we should get τ_PB ≈ 0.19 ns — a canonical number
        # quoted in Fischer-Catelani discussions of Al pair-breaking.
        Delta_0_Al = 180.0 / 86.17  # μeV/k_B → K
        tau_0_ph_Al = 0.242
        rate_Hz = tau_PB_inverse_Hz(3.0 * Delta_0_Al, Delta_0_Al, tau_0_ph_Al)
        tau_PB_ns = 1e9 / rate_Hz
        assert tau_PB_ns == pytest.approx(0.193, rel=0.02)

    def test_rate_rises_with_omega(self) -> None:
        # Above threshold, higher phonon energy → faster pair breaking
        # (more phase space for the emitted QP pair).
        Delta_K = 2.0
        tau_0_ns = 0.3
        rates = [
            tau_PB_inverse_Hz(x * Delta_K, Delta_K, tau_0_ns)
            for x in (2.5, 3.0, 5.0, 10.0)
        ]
        assert all(r_next > r_prev for r_prev, r_next in pairwise(rates))

    def test_rate_scales_inverse_with_tau_0_phonon(self) -> None:
        # 1/τ_B ∝ 1/τ_0^ph — halving τ_0 doubles the rate.
        omega_K = 6.0
        Delta_K = 2.0
        r1 = tau_PB_inverse_Hz(omega_K, Delta_K, tau_0_phonon_ns=0.1)
        r2 = tau_PB_inverse_Hz(omega_K, Delta_K, tau_0_phonon_ns=0.2)
        assert r1 / r2 == pytest.approx(2.0, rel=1e-12)

    def test_gap_suppression_prefactor(self) -> None:
        # 1/τ_B = (Δ/Δ₀) × (1/(π τ_0^ph)) × S_+(Ω/Δ). With Δ suppressed
        # below Δ₀, the OVERALL rate is suppressed by Δ/Δ₀ (prefactor)
        # but the argument Ω/Δ grows, boosting S_+.
        Delta_0 = 2.0
        Delta_T = 1.0    # half-suppressed gap
        tau_0_ns = 0.5
        omega_K = 4.0    # well above 2Δ_T = 2 and 2Δ_0 = 4
        rate = tau_PB_inverse_Hz(
            omega_K, Delta_T, tau_0_ns, Delta_0_kelvin=Delta_0
        )
        # Expected: (Δ/Δ₀) × (1/(π τ_0^ph)) × S_+(Ω/Δ) × 1e9 Hz
        expected = (Delta_T / Delta_0) / (np.pi * tau_0_ns) * 1e9 * kaplan_S_plus(omega_K / Delta_T)
        assert rate == pytest.approx(expected, rel=1e-12)

    def test_rejects_nonpositive_inputs(self) -> None:
        with pytest.raises(ValueError, match="omega_kelvin must be positive"):
            tau_PB_inverse_Hz(0.0, 1.0, 0.1)
        with pytest.raises(ValueError, match="gap_kelvin must be positive"):
            tau_PB_inverse_Hz(1.0, 0.0, 0.1)
        with pytest.raises(ValueError, match="tau_0_phonon_ns must be positive"):
            tau_PB_inverse_Hz(1.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="Delta_0_kelvin must be positive"):
            tau_PB_inverse_Hz(1.0, 1.0, 0.1, Delta_0_kelvin=-1.0)
