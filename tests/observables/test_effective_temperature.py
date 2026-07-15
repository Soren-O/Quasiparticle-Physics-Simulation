"""Tests for qpsim.observables.effective_temperature."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.observables.effective_temperature import effective_phonon_temperature
from qpsim.physics.kernels import thermal_phonon_occupation


class TestPureBoseEinsteinRecovery:
    """A pure BE distribution at temperature T must round-trip to T."""

    def test_hot_BE_recovered(self) -> None:
        # 0.5 K phonon bath, ω_bins covering 2Δ → 10Δ with Δ = 180 μeV.
        gap = 180.0
        omega = np.linspace(2.0 * gap, 10.0 * gap, 200)
        T_true = 0.5
        n_ph = thermal_phonon_occupation(omega, T_true)

        T_star = effective_phonon_temperature(
            n_ph, omega, gap, T_bath=0.05,
        )
        # Round-trip precision is limited by minimize_scalar's default xatol.
        assert T_star == pytest.approx(T_true, rel=1e-3)

    def test_bath_BE_recovered(self) -> None:
        gap = 180.0
        omega = np.linspace(2.0 * gap, 10.0 * gap, 200)
        T_true = 0.1
        n_ph = thermal_phonon_occupation(omega, T_true)

        T_star = effective_phonon_temperature(
            n_ph, omega, gap, T_bath=T_true,
        )
        assert T_star == pytest.approx(T_true, rel=1e-3)


class TestDrivenNotEqualBath:
    """If n_ph is inflated above thermal, T_* must exceed T_bath."""

    def test_recombination_heated_band(self) -> None:
        gap = 180.0
        omega = np.linspace(1.9 * gap, 10.0 * gap, 300)
        T_bath = 0.1
        # Bath baseline …
        n_thermal = thermal_phonon_occupation(omega, T_bath)
        # … plus a multiplicative boost in the pair-breaking band only.
        boost_mask = omega >= 2.0 * gap
        n_ph = n_thermal.copy()
        n_ph[boost_mask] = thermal_phonon_occupation(omega[boost_mask], 0.4)

        T_star = effective_phonon_temperature(
            n_ph, omega, gap, T_bath=T_bath,
        )
        assert T_star > T_bath
        # The fit should see the pair-breaking band only, giving ~0.4 K.
        assert T_star == pytest.approx(0.4, rel=1e-2)


class TestSubGapSampling:
    """Only the pair-breaking band (ω ≥ 2Δ) should contribute to the fit."""

    def test_subgap_occupancy_is_ignored(self) -> None:
        gap = 180.0
        omega = np.linspace(0.5 * gap, 10.0 * gap, 300)
        T_bath = 0.1
        T_true = 0.3

        # Make the pair-breaking band BE at T_true; stuff garbage
        # sub-gap (where it shouldn't affect the fit).
        n_ph = np.zeros_like(omega)
        pb_mask = omega >= 2.0 * gap
        n_ph[pb_mask] = thermal_phonon_occupation(omega[pb_mask], T_true)
        n_ph[~pb_mask] = 1e3  # bogus occupancy below 2Δ

        T_star = effective_phonon_temperature(
            n_ph, omega, gap, T_bath=T_bath,
        )
        assert T_star == pytest.approx(T_true, rel=5e-3)


class TestLowerBound:
    def test_never_returns_below_T_bath(self) -> None:
        # A mildly-depleted pair-breaking band (n_ph at 0.05 K while
        # T_bath = 0.1 K). The fit should clamp to T_bath, not slip
        # below.
        gap = 180.0
        omega = np.linspace(2.0 * gap, 10.0 * gap, 200)
        T_bath = 0.1
        n_ph = thermal_phonon_occupation(omega, 0.05)  # colder than bath

        T_star = effective_phonon_temperature(
            n_ph, omega, gap, T_bath=T_bath,
        )
        assert T_star >= T_bath


class TestEdgeCases:
    def test_all_zero_returns_T_bath(self) -> None:
        gap = 180.0
        omega = np.linspace(2.0 * gap, 10.0 * gap, 50)
        T_bath = 0.15
        assert effective_phonon_temperature(
            np.zeros_like(omega), omega, gap, T_bath=T_bath,
        ) == pytest.approx(T_bath)

    def test_no_pair_breaking_modes_returns_T_bath(self) -> None:
        gap = 180.0
        # All frequencies sub-gap, so mask is empty.
        omega = np.linspace(0.5 * gap, 1.5 * gap, 50)
        T_bath = 0.2
        n_ph = thermal_phonon_occupation(omega, T_bath)
        assert effective_phonon_temperature(
            n_ph, omega, gap, T_bath=T_bath,
        ) == pytest.approx(T_bath)

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            effective_phonon_temperature(
                np.zeros(10), np.zeros(11), 180.0, T_bath=0.1,
            )

    def test_single_occupied_mode_is_reported_as_underdetermined(self) -> None:
        omega = np.array([100.0, 400.0])
        n_ph = np.array([0.0, 1e-3])
        with pytest.raises(ValueError, match="underdetermined"):
            effective_phonon_temperature(
                n_ph, omega, 180.0, T_bath=0.1,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_invalid_occupations_rejected(self, bad: float) -> None:
        omega = np.linspace(400.0, 1800.0, 50)
        n_ph = np.ones_like(omega) * 1e-3
        n_ph[3] = bad
        with pytest.raises(ValueError, match="n_ph"):
            effective_phonon_temperature(
                n_ph, omega, 180.0, T_bath=0.1,
            )

    def test_nonpositive_gap_rejected(self) -> None:
        omega = np.linspace(100.0, 1000.0, 20)
        n = np.zeros_like(omega)
        with pytest.raises(ValueError, match="gap"):
            effective_phonon_temperature(n, omega, 0.0, T_bath=0.1)

    def test_nonpositive_T_bath_rejected(self) -> None:
        omega = np.linspace(400.0, 1800.0, 50)
        n = np.ones_like(omega) * 1e-3
        with pytest.raises(ValueError, match="T_bath"):
            effective_phonon_temperature(n, omega, 180.0, T_bath=0.0)

    def test_T_max_below_T_bath_rejected(self) -> None:
        omega = np.linspace(400.0, 1800.0, 50)
        n = np.ones_like(omega) * 1e-3
        with pytest.raises(ValueError, match="T_max"):
            effective_phonon_temperature(
                n, omega, 180.0, T_bath=0.2, T_max=0.1,
            )
