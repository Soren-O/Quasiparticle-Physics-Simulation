"""Tests for qpsim.collisions.sub_gap_photon."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext


def _setup(gap: float = 180.0, num: int = 40):
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    return ctx


class TestShapesAndNullCases:
    def test_output_shapes(self) -> None:
        ctx = _setup()
        NE = ctx.E.size
        f = np.zeros(NE)
        f[10] = 0.5
        dE = float(ctx.dE[0])
        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=3 * dE, n_bar=1.0, c_phot=1.0,
        )
        assert gain.shape == (NE,)
        assert loss.shape == (NE,)

    def test_zero_omega_returns_zero(self) -> None:
        ctx = _setup()
        f = 0.1 * np.ones(ctx.E.size)
        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=0.0, n_bar=1.0, c_phot=1.0,
        )
        np.testing.assert_allclose(gain, 0.0)
        np.testing.assert_allclose(loss, 0.0)

    def test_positive_frequency_below_half_bin_fails_loudly(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        with pytest.raises(ValueError, match="below half the grid spacing"):
            sub_gap_photon_collision_rates(
                0.1 * np.ones(ctx.E.size),
                ctx,
                omega_0=0.4 * dE,
                n_bar=1.0,
                c_phot=1.0,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_rejects_invalid_channel_parameters(self, bad: float) -> None:
        ctx = _setup()
        f = 0.1 * np.ones(ctx.E.size)
        with pytest.raises(ValueError):
            sub_gap_photon_collision_rates(f, ctx, bad, 1.0, 1.0)
        with pytest.raises(ValueError):
            sub_gap_photon_collision_rates(f, ctx, 1.0, bad, 1.0)
        with pytest.raises(ValueError):
            sub_gap_photon_collision_rates(f, ctx, 1.0, 1.0, bad)

    def test_zero_f_zero_nbar_gives_zero_gain(self) -> None:
        # Spontaneous-emission terms (∝ 1 + n_bar) still contribute to
        # the loss-rate coefficient even at n_bar = 0, but with f = 0
        # the actual loss rate is zero.
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = np.zeros(ctx.E.size)
        gain, _ = sub_gap_photon_collision_rates(
            f, ctx, omega_0=5 * dE, n_bar=0.0, c_phot=1.0,
        )
        np.testing.assert_allclose(gain, 0.0)

    def test_rejects_nonuniform_grid(self) -> None:
        E = np.array([181.0, 184.0, 195.0, 230.0])
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=180.0)
        with pytest.raises(ValueError, match="uniform energy grid"):
            sub_gap_photon_collision_rates(
                np.zeros(E.size), ctx, omega_0=3.0, n_bar=1.0, c_phot=1.0,
            )


class TestCommensurateWarning:
    def test_warns_when_off_grid(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = 0.1 * np.ones(ctx.E.size)
        # 0.4 · dE is about 40% off — well outside the 1% tolerance.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sub_gap_photon_collision_rates(
                f, ctx, omega_0=3 * dE + 0.4 * dE, n_bar=1.0, c_phot=1.0,
            )
        assert any("not grid-commensurate" in str(w.message) for w in caught)

    def test_no_warning_within_tolerance(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = 0.1 * np.ones(ctx.E.size)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sub_gap_photon_collision_rates(
                f, ctx, omega_0=3 * dE + 0.005 * dE, n_bar=1.0, c_phot=1.0,
            )
        assert not any("commensurate" in str(w.message) for w in caught)


class TestPhysicalConsistency:
    def test_thermal_balance_low_nbar(self) -> None:
        # With n_bar = n_BE(ω₀, T), a thermal f should leave df/dt ≈ 0.
        # This is the "photon bath at bath temperature" limit.
        ctx = _setup()
        T_bath = 0.3
        dE = float(ctx.dE[0])
        omega_0 = 3 * dE
        kT = KB_UEV_PER_K * T_bath
        n_bar_thermal = 1.0 / (np.exp(omega_0 / kT) - 1.0)
        f = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)

        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=omega_0, n_bar=n_bar_thermal, c_phot=1.0,
        )
        residual = gain - loss * f
        scale = np.maximum(gain + loss * f, 1e-30)
        rel = np.abs(residual / scale)
        # Boundary bins that lose partners to the grid edge can drift;
        # the bulk must balance.
        assert rel[2:-2].max() < 1e-8

    def test_photon_drive_transfers_up_in_energy(self) -> None:
        # Start with QPs concentrated at the gap edge; a n_bar > 0 bath
        # should absorb and populate bins ~m above the original support
        # with net positive rate.
        ctx = _setup(num=40)
        dE = float(ctx.dE[0])
        NE = ctx.E.size
        f = np.zeros(NE)
        f[:5] = 0.1  # low-energy-concentrated initial condition
        m = 3
        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=m * dE, n_bar=10.0, c_phot=1.0,
        )
        net = gain - loss * f
        # Bins ~m above the initial support must see net gain.
        assert np.any(net[5:5 + m + 2] > 0)
