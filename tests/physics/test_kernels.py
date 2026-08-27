"""Tests for qpsim.physics.kernels — phonon kernels and thermal occupation."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics.kernels import (
    recombination_kernel,
    recombination_kernel_base,
    scattering_kernel,
    scattering_kernel_base,
    thermal_phonon_occupation,
)


class TestThermalPhononOccupation:
    def test_zero_at_T_zero(self) -> None:
        assert np.all(thermal_phonon_occupation(np.array([1.0, 2.0]), 0.0) == 0.0)

    def test_high_T_limit(self) -> None:
        # n_BE(ω, T → ∞) ≈ kT/ω.
        omega = np.array([1.0])
        T = 1e6
        kT = KB_UEV_PER_K * T
        got = thermal_phonon_occupation(omega, T)
        np.testing.assert_allclose(got, kT / omega, rtol=1e-3)

    def test_tiny_positive_frequency_does_not_collapse_to_zero(self) -> None:
        omega = np.array([1e-18])
        temperature = 1.0
        expected = 1.0 / np.expm1(omega / (KB_UEV_PER_K * temperature))
        got = thermal_phonon_occupation(omega, temperature)
        np.testing.assert_allclose(got, expected, rtol=2e-15)
        assert np.isfinite(got[0])
        assert got[0] > 0.0

    def test_cold_representable_tail_is_not_floored_at_exp_minus_500(self) -> None:
        # Actual preliminary-campaign bath/gap scale: 2*Delta/kT ~= 596.8.
        # exp(-x) remains representable here and must not be replaced by the
        # much larger historical exp(-500) floor.
        omega = np.array([360.0])
        temperature = 0.007
        exponent = omega / (KB_UEV_PER_K * temperature)
        exp_negative = np.exp(-exponent)
        expected = exp_negative / (-np.expm1(-exponent))

        got = thermal_phonon_occupation(omega, temperature)

        np.testing.assert_allclose(got, expected, rtol=2e-14)
        assert 0.0 < got[0] < np.exp(-500.0)

    def test_mathematically_underflowed_cold_tail_is_zero(self) -> None:
        temperature = 0.007
        omega = np.array([800.0 * KB_UEV_PER_K * temperature])
        np.testing.assert_array_equal(
            thermal_phonon_occupation(omega, temperature),
            np.array([0.0]),
        )

    def test_exact_zero_frequency_is_decoupled_bookkeeping_mode(self) -> None:
        got = thermal_phonon_occupation(np.array([0.0]), 1.0)
        np.testing.assert_array_equal(got, np.array([0.0]))

    def test_rejects_complex_frequency_before_casting(self) -> None:
        with pytest.raises(ValueError, match="real-valued"):
            thermal_phonon_occupation(np.array([1.0 + 2.0j]), 0.1)

    def test_finite_extreme_temperature_saturates_without_overflow(self) -> None:
        got = thermal_phonon_occupation(
            np.array([1.0]),
            np.finfo(float).max,
        )
        np.testing.assert_array_equal(got, np.array([np.finfo(float).max]))

    def test_subnormal_energy_temperature_ratio_remains_representable(
        self,
    ) -> None:
        smallest = np.nextafter(0.0, 1.0)
        omega = np.array([smallest, 10.0 * smallest])
        exponent = (omega / smallest) / KB_UEV_PER_K
        expected = np.exp(-exponent) / (-np.expm1(-exponent))

        got = thermal_phonon_occupation(omega, smallest)

        np.testing.assert_allclose(got, expected, rtol=2e-15)

    def test_non_negative(self) -> None:
        occ = thermal_phonon_occupation(np.linspace(0.0, 10.0, 50), 1.0)
        assert np.all(occ >= 0)

    def test_rejects_negative_omega(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            thermal_phonon_occupation(np.array([-1.0]), 1.0)

    @pytest.mark.parametrize("temperature", [float("nan"), float("inf")])
    def test_rejects_non_finite_temperature(self, temperature: float) -> None:
        with pytest.raises(ValueError, match="temperature must be finite"):
            thermal_phonon_occupation(np.array([1.0]), temperature)

    def test_rejects_negative_temperature(self) -> None:
        with pytest.raises(ValueError, match="temperature must be non-negative"):
            thermal_phonon_occupation(np.array([1.0]), -1.0)


class TestKernelsBase:
    def test_shapes(self) -> None:
        E = np.linspace(1.01, 5.0, 8)
        K_r = recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        K_s = scattering_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        assert K_r.shape == (8, 8)
        assert K_s.shape == (8, 8)

    def test_scattering_diagonal_zero(self) -> None:
        E = np.linspace(1.01, 5.0, 8)
        K_s = scattering_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        np.testing.assert_allclose(np.diag(K_s), 0.0)

    def test_recombination_symmetric(self) -> None:
        # Depends on (E_i + E_j)² and K⁺ — both symmetric under i↔j.
        E = np.linspace(1.01, 5.0, 8)
        K_r = recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        np.testing.assert_allclose(K_r, K_r.T, atol=1e-14)

    def test_uses_precomputed_coherence(self) -> None:
        # Passing ctx.K_plus should reproduce the default K⁺ path.
        E = np.linspace(1.01, 5.0, 6)
        from qpsim.physics.spectral import coherence_factor_plus

        K_p = coherence_factor_plus(E, gap=1.0)
        K_r_default = recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        K_r_with_coh = recombination_kernel_base(
            E, gap=1.0, tau_0=1.0, T_c=1.2, coherence_factor=K_p
        )
        np.testing.assert_allclose(K_r_default, K_r_with_coh)

    @pytest.mark.parametrize(
        "builder", [recombination_kernel_base, scattering_kernel_base],
    )
    def test_rejects_non_finite_energy(self, builder) -> None:
        with pytest.raises(ValueError, match="E_bins must contain only finite"):
            builder(np.array([2.0, float("nan")]), gap=1.0, tau_0=1.0, T_c=1.2)

    @pytest.mark.parametrize("E", [np.array([]), np.array([[2.0]])])
    def test_rejects_invalid_energy_grid_shape(self, E: np.ndarray) -> None:
        with pytest.raises(ValueError, match="E_bins must be"):
            recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)

    @pytest.mark.parametrize("name", ["gap", "tau_0", "T_c"])
    def test_rejects_non_finite_scalar_parameters(self, name: str) -> None:
        kwargs = {"gap": 1.0, "tau_0": 1.0, "T_c": 1.2}
        kwargs[name] = float("nan")
        with pytest.raises(ValueError, match=rf"{name} must be finite"):
            recombination_kernel_base(np.array([2.0]), **kwargs)

    @pytest.mark.parametrize(
        ("name", "value", "message"),
        [
            ("gap", -1.0, "gap must be non-negative"),
            ("tau_0", 0.0, "tau_0 must be positive"),
            ("tau_0", -1.0, "tau_0 must be positive"),
            ("T_c", 0.0, "T_c must be positive"),
            ("T_c", -1.0, "T_c must be positive"),
        ],
    )
    def test_rejects_non_physical_scalar_parameters(
        self, name: str, value: float, message: str,
    ) -> None:
        kwargs = {"gap": 1.0, "tau_0": 1.0, "T_c": 1.2}
        kwargs[name] = value
        with pytest.raises(ValueError, match=message):
            scattering_kernel_base(np.array([2.0]), **kwargs)

    def test_rejects_non_finite_precomputed_coherence(self) -> None:
        with pytest.raises(ValueError, match="coherence_factor"):
            recombination_kernel_base(
                np.array([2.0]),
                gap=1.0,
                tau_0=1.0,
                T_c=1.2,
                coherence_factor=np.array([[float("nan")]]),
            )

    def test_rejects_wrong_precomputed_coherence_shape(self) -> None:
        with pytest.raises(ValueError, match="coherence_factor must have shape"):
            scattering_kernel_base(
                np.array([2.0, 3.0]),
                gap=1.0,
                tau_0=1.0,
                T_c=1.2,
                coherence_factor=np.ones(2),
            )


class TestKernelsWithPhonon:
    def test_zero_T_recombination_matches_base(self) -> None:
        # At T = 0, N_p = 1 everywhere ⇒ K^r = K₀ʳ.
        E = np.linspace(1.01, 5.0, 6)
        K_r0 = recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        K_r = recombination_kernel(E, gap=1.0, tau_0=1.0, T_c=1.2, bath_temperature=0.0)
        np.testing.assert_allclose(K_r, K_r0)

    def test_zero_T_scattering_emission_only(self) -> None:
        # At T = 0, absorption vanishes ⇒ K^s zero where E_i < E_j.
        E = np.linspace(1.01, 5.0, 6)
        K_s = scattering_kernel(E, gap=1.0, tau_0=1.0, T_c=1.2, bath_temperature=0.0)
        for i in range(6):
            for j in range(i + 1, 6):
                assert K_s[i, j] == 0.0

    def test_finite_T_recombination_exceeds_base(self) -> None:
        # At T > 0, N_p > 1 for all off-zero ω ⇒ K^r > K₀ʳ elementwise.
        E = np.linspace(1.01, 5.0, 6)
        K_r0 = recombination_kernel_base(E, gap=1.0, tau_0=1.0, T_c=1.2)
        K_r = recombination_kernel(E, gap=1.0, tau_0=1.0, T_c=1.2, bath_temperature=0.5)
        assert np.all(K_r >= K_r0 - 1e-14)
        assert np.any(K_r > K_r0 + 1e-12)

    def test_nearly_degenerate_scattering_energies_remain_finite(self) -> None:
        E = np.array([2.0, np.nextafter(2.0, np.inf)])
        K_s = scattering_kernel(
            E,
            gap=1.0,
            tau_0=1.0,
            T_c=1.2,
            bath_temperature=1.0,
        )
        assert np.all(np.isfinite(K_s))
        assert np.all(K_s >= 0.0)
        assert K_s[1, 0] > 0.0
        assert K_s[0, 1] > 0.0

    @pytest.mark.parametrize("builder", [recombination_kernel, scattering_kernel])
    def test_rejects_non_finite_bath_temperature(self, builder) -> None:
        with pytest.raises(ValueError, match="bath_temperature must be finite"):
            builder(
                np.array([2.0]),
                gap=1.0,
                tau_0=1.0,
                T_c=1.2,
                bath_temperature=float("nan"),
            )

    @pytest.mark.parametrize("builder", [recombination_kernel, scattering_kernel])
    def test_rejects_negative_bath_temperature(self, builder) -> None:
        with pytest.raises(ValueError, match="bath_temperature must be non-negative"):
            builder(
                np.array([2.0]),
                gap=1.0,
                tau_0=1.0,
                T_c=1.2,
                bath_temperature=-0.1,
            )

class TestTheDebyeAssumptionIsIntact:
    """The phonon spectrum is Debye, and the kernels encode it asymmetrically.

    alpha^2 F(omega) = b * omega^2, with tau_0 defined by Kaplan (1976) under
    that assumption. The quasiparticle equation integrates OVER phonon modes
    and so carries D(omega) ~ omega^2; the phonon equation is written PER MODE
    and carries none. Their ratio is therefore exactly omega^2 times a
    constant.

    That ratio is the cheap check that the structure is intact, and it is
    tested rather than only documented because the assumption was implicit for
    long enough to invite adding a per-bin phonon density of states -- which
    would have double-counted it. A change to either kernel's frequency
    dependence alone breaks this.

    See docs/Phonon_Model_Decisions.md, "The phonon spectrum is Debye".
    """

    @staticmethod
    def _context(num_bins: int = 60, gap: float = 180.0):
        from qpsim.grid.energy_grid import (
            build_energy_grid,
            integration_widths_from_centers,
        )
        from qpsim.physics.spectral import SpectralContext

        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.0, energy_max_factor=10.0,
            num_energy_bins=num_bins,
        )
        return SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap,
        )

    def test_the_kernel_ratio_is_exactly_omega_squared(self) -> None:
        from qpsim.collisions.phonon import (
            build_scattering_kernel_base,
            build_scattering_kernel_phonon_side,
        )

        ctx = self._context()
        qp_side = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        phonon_side = build_scattering_kernel_phonon_side(ctx, 0.255)

        omega = np.abs(np.subtract.outer(ctx.E, ctx.E))
        live = (phonon_side != 0.0) & (omega > 0.0)
        ratio = qp_side[live] / phonon_side[live]
        scaled = ratio / omega[live] ** 2

        # Constant to machine precision: that IS the Debye structure.
        assert np.ptp(scaled) / np.mean(scaled) < 1e-13

    def test_the_phonon_side_carries_no_frequency_dependence(self) -> None:
        """Per mode, so no density of states -- the asymmetry is deliberate.

        Checked against the coherence factor rather than against a constant:
        the kernel is 2*K_minus/(pi*Delta*tau_0_pb), so dividing it out must
        leave a pure number with no omega in it.
        """
        from qpsim.collisions.phonon import build_scattering_kernel_phonon_side

        ctx = self._context()
        phonon_side = build_scattering_kernel_phonon_side(ctx, 0.255)
        live = ctx.K_minus != 0.0
        prefactor = phonon_side[live] / ctx.K_minus[live]

        assert np.ptp(prefactor) / np.mean(prefactor) < 1e-13

    def test_the_quasiparticle_side_does_carry_it(self) -> None:
        """Non-vacuity: the test above would pass if BOTH sides were flat."""
        from qpsim.collisions.phonon import build_scattering_kernel_base

        ctx = self._context()
        qp_side = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.2)
        live = ctx.K_minus != 0.0
        prefactor = qp_side[live] / ctx.K_minus[live]

        assert np.ptp(prefactor) / np.mean(prefactor) > 1.0
