"""Tests for qpsim.observables.frequency_shift."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.frequency_shift import compute_frequency_shift
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


class TestComputeFrequencyShift:
    def test_equal_f_gives_zero_shift(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        shift = compute_frequency_shift(f, f, ctx, omega_0=1.0, alpha=0.1)
        assert shift == pytest.approx(0.0, abs=1e-12)

    def test_hotter_f_gives_negative_shift(self) -> None:
        # Hotter ⇒ smaller σ₂ ⇒ negative shift (resonator gets softer).
        ctx, f_ref = _thermal_ctx_and_f(T_bath=0.1)
        _, f_hot = _thermal_ctx_and_f(T_bath=0.8)
        shift = compute_frequency_shift(f_hot, f_ref, ctx, omega_0=1.0, alpha=0.1)
        assert shift < 0.0

    def test_alpha_zero_gives_zero_shift(self) -> None:
        # With α = 0 the kinetic-inductance fraction is zero ⇒ no shift
        # even if σ₂ differs.
        ctx, f_ref = _thermal_ctx_and_f(T_bath=0.1)
        _, f_hot = _thermal_ctx_and_f(T_bath=0.8)
        shift = compute_frequency_shift(f_hot, f_ref, ctx, omega_0=1.0, alpha=0.0)
        assert shift == 0.0

    def test_zero_reference_sigma2_with_change_fails_loudly(self) -> None:
        # A normal reference has sigma2=0, so a nonzero current response makes
        # the relative shift undefined and must not be mislabeled as zero.
        ctx, _ = _thermal_ctx_and_f()
        f_normal = np.full(ctx.E.size, 0.5)
        f_other = np.full(ctx.E.size, 0.4)
        with pytest.raises(ValueError, match="reference sigma2 is zero"):
            compute_frequency_shift(
                f_other,
                f_normal,
                ctx,
                omega_0=1.0,
                alpha=0.1,
            )

    def test_negative_reference_sigma2_preserves_signed_shift(self) -> None:
        ctx, _ = _thermal_ctx_and_f(num=32)
        omega_0 = 5.0
        alpha = 0.2
        f_ref = 0.9 * np.exp(-(ctx.E - ctx.gap) / (10.0 * omega_0))
        f = np.zeros_like(ctx.E)

        from qpsim.observables.ac_conductivity import compute_ac_conductivity

        _, sigma2 = compute_ac_conductivity(f, ctx, omega_0)
        _, sigma2_ref = compute_ac_conductivity(f_ref, ctx, omega_0)
        assert sigma2_ref < 0.0

        shift = compute_frequency_shift(
            f,
            f_ref,
            ctx,
            omega_0=omega_0,
            alpha=alpha,
        )
        assert shift == pytest.approx(
            0.5 * alpha * (sigma2 - sigma2_ref) / sigma2_ref
        )
