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
        gap=gap, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    kT = KB_UEV_PER_K * T_bath
    f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return ctx, f


class TestComputeAcConductivity:
    @pytest.mark.parametrize(
        "bad_value",
        (1.0j, complex(float("nan"), 0.0), float("inf"), -0.1, 1.1),
    )
    def test_rejects_invalid_occupation(self, bad_value: complex | float) -> None:
        ctx, _ = _thermal_ctx_and_f()
        dtype = complex if isinstance(bad_value, complex) else float
        f = np.zeros_like(ctx.E, dtype=dtype)
        f[0] = bad_value
        with pytest.raises(ValueError, match=r"real-valued|finite|occupations"):
            compute_ac_conductivity(f, ctx, omega_0=1.0)

    def test_rejects_zero_omega(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="omega_0"):
            compute_ac_conductivity(f, ctx, omega_0=0.0)

    def test_rejects_negative_omega(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="omega_0"):
            compute_ac_conductivity(f, ctx, omega_0=-1.0)

    @pytest.mark.parametrize(
        "bad_omega",
        [np.nan, np.inf, -np.inf, complex(1.0, 0.0), True],
    )
    def test_rejects_nonfinite_complex_or_boolean_omega(
        self, bad_omega: complex | float
    ) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="omega_0"):
            compute_ac_conductivity(f, ctx, omega_0=bad_omega)

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

    @pytest.mark.parametrize(
        "bad_count", [1.5, np.nan, np.inf, complex(10.0, 0.0), True]
    )
    def test_rejects_noninteger_n_subgap(
        self, bad_count: complex | float
    ) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="n_subgap"):
            compute_ac_conductivity(
                f,
                ctx,
                omega_0=1.0,
                n_subgap=bad_count,  # type: ignore[arg-type]
            )

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

    def test_rejects_sigma_1_grid_with_dropped_gap_support(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 1.01, 10.0, 1620)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )
        f = np.zeros_like(E)

        with pytest.raises(ValueError, match=r"does not cover.*BCS lower bound"):
            compute_ac_conductivity(f, ctx, omega_0=20.0)

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

    def test_sigma_2_endpoint_transform_matches_adaptive_reference(self) -> None:
        from scipy.integrate import quad

        gap = 180.0
        omega_0 = 20.0
        E, _ = build_energy_grid(gap, 1.0, 10.0, 405)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )
        f = np.zeros_like(E)

        _, sigma_2 = compute_ac_conductivity(
            f, ctx, omega_0, n_subgap=500,
        )

        # Independent adaptive integral after the endpoint-regularizing
        # E = Delta - omega*cos(theta)^2 substitution.
        def reference_integrand(theta: float) -> float:
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            energy = gap - omega_0 * cos_theta**2
            partner = gap + omega_0 * sin_theta**2
            rho_partner = partner / np.sqrt(partner**2 - gap**2)
            coherence = 1.0 + gap**2 / (energy * partner)
            subgap_dos = energy / np.sqrt(gap**2 - energy**2)
            jacobian = 2.0 * omega_0 * sin_theta * cos_theta
            return rho_partner * coherence * subgap_dos * jacobian / omega_0

        reference = quad(
            reference_integrand,
            0.0,
            0.5 * np.pi,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=200,
        )[0]

        # Reproduce the old raw-E midpoint bias: 500 points remained about
        # 1.72% low because it sampled two square-root singular endpoints.
        dE_sub = omega_0 / 500
        E_sub = gap - omega_0 + (np.arange(500) + 0.5) * dE_sub
        partner = E_sub + omega_0
        legacy = (1.0 / omega_0) * float(np.sum(
            partner / np.sqrt(partner**2 - gap**2)
            * (1.0 + gap**2 / (E_sub * partner))
            * E_sub / np.sqrt(gap**2 - E_sub**2)
            * dE_sub
        ))
        assert abs(legacy / reference - 1.0) > 0.015
        assert sigma_2 == pytest.approx(reference, rel=2e-11)


class TestSigma2IgnoresSubGapPlaceholders:
    """Sub-gap cells carry no spectral capacity, so their f is a placeholder.

    sigma_2 used to interpolate the partner occupation across ALL centres,
    which blended those placeholders in through the low-theta nodes just above
    the gap. The dependence was invisible on any grid starting at Delta, which
    is every shipped preset -- so a fixture at min_factor = 1 is vacuously
    green and proves nothing. This one extends below the gap on purpose.
    """

    @staticmethod
    def _sigma2(placeholder, *, min_factor=0.9, num_bins=64):
        gap = 180.0
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=min_factor,
            energy_max_factor=4.0, num_energy_bins=num_bins,
        )
        ctx = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap,
        )
        f = 1e-4 * np.exp(-(E - gap) / 50.0)
        f = np.where(ctx.active_mask, f, placeholder)
        return compute_ac_conductivity(f, ctx, 22.0)[1]

    def test_placeholder_value_cannot_move_sigma2(self):
        assert self._sigma2(0.0) == self._sigma2(1.0)

    def test_a_grid_starting_at_the_gap_is_unaffected(self):
        """The masked and unmasked conventions must agree where no cell is
        inactive -- otherwise this fix would have moved published numbers."""
        assert self._sigma2(0.0, min_factor=1.0) == pytest.approx(
            self._sigma2(1.0, min_factor=1.0), rel=0.0, abs=0.0,
        )

    def test_one_active_cell_is_refused_not_silently_zero_filled(self):
        gap = 180.0
        E = np.array([gap - 1.0, gap + 1.0])
        ctx = SpectralContext(
            E_bins=E, dE_bins=np.full(2, 2.0), gap=gap,
        )
        with pytest.raises(ValueError, match="at least two active"):
            compute_ac_conductivity(np.array([0.5, 1e-4]), ctx, 22.0)
