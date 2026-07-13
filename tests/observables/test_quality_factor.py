"""Tests for qpsim.observables.quality_factor."""

from __future__ import annotations

import numpy as np
import pytest
import qpsim.observables.quality_factor as quality_factor_module
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.observables.quality_factor import compute_quality_factor
from qpsim.physics.spectral import SpectralContext


def _thermal_ctx_and_f(T_bath: float = 0.3, T_c: float = 1.2, num: int = 200):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.001, energy_max_factor=10.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    kT = KB_UEV_PER_K * T_bath
    f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return ctx, f


class TestComputeQualityFactor:
    def test_rejects_non_positive_alpha(self) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="alpha"):
            compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.0)

    def test_positive_Q_at_finite_T(self) -> None:
        ctx, f = _thermal_ctx_and_f(T_bath=0.3)
        Q = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.1)
        assert Q > 0.0

    def test_Q_infinite_at_vanishing_dissipation(self) -> None:
        # σ₁ → 0 at T → 0 ⇒ Q_qp → ∞.
        ctx, f_cold = _thermal_ctx_and_f(T_bath=0.01)
        Q = compute_quality_factor(f_cold, ctx, omega_0=1.0, alpha=0.1)
        assert float("inf") == Q or Q > 1e10

    def test_external_Q_combination(self) -> None:
        # With finite Q_ext, Q_tot ≤ min(Q_qp, Q_ext).
        ctx, f = _thermal_ctx_and_f(T_bath=0.3)
        Q_qp = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.1)
        Q_ext = 1e5
        Q_tot = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.1, Q_ext=Q_ext)
        assert Q_tot < Q_qp
        assert Q_tot < Q_ext
        expected = 1.0 / (1.0 / Q_qp + 1.0 / Q_ext)
        assert Q_tot == pytest.approx(expected)

    def test_alpha_scales_inversely(self) -> None:
        # Q_qp = σ₂ / (α σ₁) ⇒ doubling α halves Q_qp.
        ctx, f = _thermal_ctx_and_f(T_bath=0.3)
        Q1 = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.1)
        Q2 = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.2)
        if np.isfinite(Q1):
            assert pytest.approx(0.5 * Q1, rel=1e-6) == Q2

    def test_negative_sigma1_reports_signed_gain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx, f = _thermal_ctx_and_f()
        monkeypatch.setattr(
            quality_factor_module,
            "compute_ac_conductivity",
            lambda *_args, **_kwargs: (-2.0, 6.0),
        )

        q_gain = compute_quality_factor(f, ctx, omega_0=1.0, alpha=0.5)

        assert q_gain == pytest.approx(-6.0)
        assert compute_quality_factor(
            f, ctx, omega_0=1.0, alpha=0.5, Q_ext=6.0
        ) == float("inf")

    @pytest.mark.parametrize("bad_q_ext", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_invalid_external_q(self, bad_q_ext: float) -> None:
        ctx, f = _thermal_ctx_and_f()
        with pytest.raises(ValueError, match="Q_ext"):
            compute_quality_factor(
                f, ctx, omega_0=1.0, alpha=0.1, Q_ext=bad_q_ext
            )
