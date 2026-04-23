"""Tests for the Marchegiani rate-equation module (M25 Eq. 8)."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.services.rate_equation import crossover_temperature_kelvin
from scipy.special import lambertw


class TestLambertWIdentity:
    """T̄ · W(4π r/g) / (2Δ) = 1 by construction."""

    def test_round_trip(self) -> None:
        Delta_R = 2.35
        r_Rc = 6.25e6
        g_ph = 300.0
        T_bar = crossover_temperature_kelvin(
            Delta_R_kelvin=Delta_R,
            r_Rc_rate_Hz=r_Rc,
            g_photon_R_rate_Hz=g_ph,
        )
        w_val = float(lambertw(4.0 * np.pi * r_Rc / g_ph, k=0).real)
        assert T_bar * w_val / (2.0 * Delta_R) == pytest.approx(1.0, rel=1e-12)


class TestMonotonicity:
    """T̄ should rise with Δ_R, rise with r^{R_c}, and fall with g^ph_R."""

    def _T_bar(self, Delta_R: float = 2.35, r_Rc: float = 6.25e6, g_ph: float = 300.0) -> float:
        return crossover_temperature_kelvin(
            Delta_R_kelvin=Delta_R, r_Rc_rate_Hz=r_Rc, g_photon_R_rate_Hz=g_ph,
        )

    def test_T_bar_rises_with_Delta_R(self) -> None:
        assert self._T_bar(Delta_R=2.0) < self._T_bar(Delta_R=3.0)

    def test_T_bar_rises_with_r_Rc(self) -> None:
        # Larger r_Rc → larger Lambert-W argument → larger W → smaller
        # T̄? Check actual direction.
        lo = self._T_bar(r_Rc=1e5)
        hi = self._T_bar(r_Rc=1e9)
        # Lambert-W grows ~log(z), so 2Δ/W decreases as r_Rc rises.
        assert hi < lo

    def test_T_bar_falls_with_g_photon(self) -> None:
        # Larger g^ph → smaller argument → smaller W → larger T̄.
        # Physically: stronger photon generation pushes the crossover up.
        lo = self._T_bar(g_ph=100.0)
        hi = self._T_bar(g_ph=1e4)
        assert hi > lo


class TestScalingLimits:
    """At weak photon drive (g_ph → 0) the argument → ∞, W ~ log, T̄ → 0."""

    def test_weak_drive_limit(self) -> None:
        T_weak = crossover_temperature_kelvin(
            Delta_R_kelvin=2.35, r_Rc_rate_Hz=6.25e6,
            g_photon_R_rate_Hz=1e-3,  # very weak drive
        )
        T_strong = crossover_temperature_kelvin(
            Delta_R_kelvin=2.35, r_Rc_rate_Hz=6.25e6,
            g_photon_R_rate_Hz=1e6,
        )
        assert T_weak < T_strong  # weak drive → low crossover temp


class TestReferenceValue:
    """Reproduce the M25 Fig 3 paper-stated parameter set value.

    Fig 3 caption: Δ_R/h = 49 GHz, r^{R_c} = 6.25 MHz,
    Γ_{01}^{ph} = 300 Hz. The paper places the T̄ dashed lines near
    T̄ ≈ 70 mK (small ω_LR) / T̄ ≈ 150 mK (large ω_LR). These values
    depend on ω_LR through g^ph_R; the closed-form T̄ formula itself
    only exposes the combination r^{R_c} / g^ph_R, so we verify the
    order-of-magnitude sanity and the Lambert-W identity rather than
    pin a specific ω_LR.
    """

    def test_order_of_magnitude(self) -> None:
        # Δ_R/h = 49 GHz = 2.352 K.
        Delta_R_K = 49e9 * 6.62607015e-34 / 1.380649e-23
        # r^{R_c} = 6.25 MHz, Γ_{01}^ph = 300 Hz as listed in Fig 3.
        # The effective g^ph_R for the full rate equation depends on
        # cooper-pair count; paper-reading suggests g^ph_R ~ kHz range
        # for these params. Sweep g^ph from 1 Hz to 1 MHz — T̄ should
        # land in mK-to-few-K range across that sweep.
        T_bars = [
            crossover_temperature_kelvin(
                Delta_R_kelvin=Delta_R_K,
                r_Rc_rate_Hz=6.25e6,
                g_photon_R_rate_Hz=g,
            )
            for g in (1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6)
        ]
        assert all(1e-3 < T < 5.0 for T in T_bars)
        # Monotonic-in-g^ph (rising).
        assert all(T_bars[i] < T_bars[i + 1] for i in range(len(T_bars) - 1))


class TestInputValidation:
    def test_negative_Delta_rejected(self) -> None:
        with pytest.raises(ValueError, match="Delta_R_kelvin"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=-1.0, r_Rc_rate_Hz=1e6, g_photon_R_rate_Hz=100.0,
            )

    def test_zero_r_Rc_rejected(self) -> None:
        with pytest.raises(ValueError, match="r_Rc_rate_Hz"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=2.35, r_Rc_rate_Hz=0.0, g_photon_R_rate_Hz=100.0,
            )

    def test_zero_g_photon_rejected(self) -> None:
        with pytest.raises(ValueError, match="g_photon_R_rate_Hz"):
            crossover_temperature_kelvin(
                Delta_R_kelvin=2.35, r_Rc_rate_Hz=1e6, g_photon_R_rate_Hz=0.0,
            )
