"""Layer-1 validation — Mattis-Bardeen σ at the T → 0 analytic limit.

At ``f = f_FD(T)`` the σ_1 and σ_2 integrals should reduce
to the textbook BCS coherence-factor expressions (Mattis-Bardeen
1958). These tests check the two simplest closed-form cases:

* **T → 0, ω < 2Δ**: σ_1/σ_N = 0 (no pair breaking by the sub-gap
  probe, no thermal QPs to absorb it).
* **T → 0, ω < 2Δ**: σ_2/σ_N ≈ π Δ / ω (the standard kinetic-
  inductance limit of Mattis-Bardeen).
"""

from __future__ import annotations

import numpy as np
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation


def _fischer_like_ctx(num: int = 1620) -> SpectralContext:
    delta_0 = 180.0
    E, _ = build_energy_grid(
        gap=delta_0, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    return SpectralContext(E_bins=E, dE_bins=dE, gap=delta_0)


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    return fermi_dirac_occupation(E, T)


class TestMBThermalLimit:
    def test_sigma_1_vanishes_at_zero_T(self) -> None:
        # Far below T_c, with a sub-gap probe, σ_1 / σ_N should be
        # exponentially suppressed. Pick T = 10 mK ≪ Δ/k_B ≈ 2 K.
        ctx = _fischer_like_ctx(num=810)
        T = 0.01
        omega_0 = ctx.gap / 9.0  # sub-gap; commensurate at NE=810 (m=10)
        f = _fermi_dirac(ctx.E, T)
        s1, _ = compute_ac_conductivity(f, ctx, omega_0=omega_0)
        # σ_1 at T→0 for ω < 2Δ is driven by exp(-Δ/kT) — at T=0.01 K,
        # Δ/kT ≈ 208, so e^{-208} is ~1e-90. Accept anything below the
        # Mattis-Bardeen sub-gap quadrature floor (~1e-30).
        assert s1 < 1e-20

    def test_sigma_2_matches_kinetic_inductance_limit(self) -> None:
        # At T → 0, ω ≪ 2Δ the Mattis-Bardeen σ_2 reduces to the
        # classic kinetic-inductance limit σ_2/σ_N → π Δ / ω. At
        # ω/Δ = 1/9 the correction to this leading form is ≈ 0.07 %
        # (verified against scipy.integrate.quad with 'alg' weight on
        # the 1/√ endpoint singularities). The code's 500-point
        # sub-gap midpoint rule picks up an additional ~1.5 % error
        # from the two weak-singularity endpoints.
        import math

        ctx = _fischer_like_ctx(num=810)
        T = 0.01
        omega_0 = ctx.gap / 9.0  # ω/Δ = 1/9
        f = _fermi_dirac(ctx.E, T)
        _, s2 = compute_ac_conductivity(f, ctx, omega_0=omega_0)

        mb_leading = math.pi * ctx.gap / omega_0
        rel_err = abs(s2 - mb_leading) / mb_leading
        # 2 % budget: 0.07 % next-order MB correction + ~1.5 % midpoint
        # error on the sqrt-singular sub-gap integrand.
        assert rel_err < 0.02

    def test_sigma_2_scales_inversely_with_omega(self) -> None:
        # σ_2 ~ π Δ / ω at low T gives a 1/ω dependence once the
        # sub-gap window is opened. Doubling ω should halve σ_2
        # (to first order in the π Δ / ω limit).
        ctx = _fischer_like_ctx(num=1620)
        T = 0.01
        f = _fermi_dirac(ctx.E, T)
        _, s2_low = compute_ac_conductivity(f, ctx, omega_0=ctx.gap / 18.0)
        _, s2_hi = compute_ac_conductivity(f, ctx, omega_0=ctx.gap / 9.0)
        ratio = s2_low / s2_hi
        # Ideal ratio is 2 (halving ω doubles σ_2); elliptic correction
        # nudges it slightly.
        assert 1.8 < ratio < 2.1
