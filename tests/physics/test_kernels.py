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
