"""Tests for qpsim.collisions.sub_gap_photon."""

from __future__ import annotations

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
    @pytest.mark.parametrize(
        "bad_f",
        [np.zeros(3), np.full(40, np.nan), np.full(40, -0.1), np.full(40, 1.1)],
    )
    def test_no_op_still_rejects_invalid_occupation(
        self,
        bad_f: np.ndarray,
    ) -> None:
        ctx = _setup()
        with pytest.raises(ValueError, match=r"finite occupations|shape"):
            sub_gap_photon_collision_rates(
                bad_f,
                ctx,
                omega_0=0.0,
                n_bar=0.0,
                c_phot=0.0,
            )

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

    def test_zero_coupling_bypasses_inapplicable_grid_guards(self) -> None:
        E = np.array([1.1, 1.3, 1.8, 2.6])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        f = np.linspace(0.0, 0.2, E.size)

        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=1.7, n_bar=1.0, c_phot=0.0,
        )

        np.testing.assert_array_equal(gain, np.zeros_like(f))
        np.testing.assert_array_equal(loss, np.zeros_like(f))

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

    def test_rejects_frequency_at_or_above_pair_threshold(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        m = int(np.ceil(2.0 * ctx.gap / dE))
        with pytest.raises(ValueError, match="omega_0 < 2\\*gap"):
            sub_gap_photon_collision_rates(
                np.zeros(ctx.E.size), ctx, omega_0=m * dE,
                n_bar=1.0, c_phot=1.0,
            )

    def test_rejects_dynes_context(self) -> None:
        ctx = _setup()
        dynes = SpectralContext(
            E_bins=ctx.E,
            dE_bins=ctx.dE,
            gap=ctx.gap,
            dynes_gamma=0.1,
        )
        with pytest.raises(ValueError, match="dynes_gamma"):
            sub_gap_photon_collision_rates(
                np.zeros(ctx.E.size), dynes, omega_0=3.0 * ctx.dE[0],
                n_bar=1.0, c_phot=1.0,
            )


class TestCommensurateGrid:
    def test_rejects_when_off_grid(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = 0.1 * np.ones(ctx.E.size)
        # 0.4 · dE is about 40% off — well outside the 1% tolerance.
        with pytest.raises(ValueError, match="not grid-commensurate"):
            sub_gap_photon_collision_rates(
                f, ctx, omega_0=3 * dE + 0.4 * dE, n_bar=1.0, c_phot=1.0,
            )

    def test_accepts_within_tolerance(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = 0.1 * np.ones(ctx.E.size)
        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=3 * dE + 0.005 * dE, n_bar=1.0, c_phot=1.0,
        )
        assert np.all(np.isfinite(gain))
        assert np.all(np.isfinite(loss))


class TestPhysicalConsistency:
    def test_exact_capacity_conservation_exposes_hybrid_point_rate_bug(self) -> None:
        # The gap cuts cell 0, whose center is sub-gap: point rho[0] is zero
        # while its exact capacity is finite.  Fixed-lattice scattering must
        # use rho_bar=w/dE on the partner leg so w_i*T_ij=w_j*T_ji.
        E = np.arange(0.9, 3.0, 0.4)
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        f = np.array([0.08, 0.12, 0.03, 0.20, 0.01, 0.15])

        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=0.8, n_bar=2.3, c_phot=0.7,
        )
        exact_drift = float(ctx.cell_weights @ (gain - loss * f))
        assert abs(exact_drift) < 2e-15

        # Recreate the rejected G1 hybrid locally: exact conserved capacity,
        # but point-sampled photon partner density.  It creates an O(1e-2)
        # artificial drain on this tiny grid, the mechanism that collapsed
        # the driven Fischer solutions by several orders of magnitude.
        hybrid = SpectralContext(E, dE, gap=1.0)
        hybrid._cell_density = np.asarray(hybrid.rho).copy()
        gain_bad, loss_bad = sub_gap_photon_collision_rates(
            f, hybrid, omega_0=0.8, n_bar=2.3, c_phot=0.7,
        )
        hybrid_drift = float(
            hybrid.cell_weights @ (gain_bad - loss_bad * f)
        )
        assert abs(hybrid_drift) > 1e-3

    def test_zero_dos_target_rows_are_zero(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 0.75, 4.0, 60)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E, dE, gap)
        f = np.zeros(E.size)
        f[ctx.active_mask] = 0.1
        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=3.0 * dE[0], n_bar=0.0, c_phot=1.0,
        )
        np.testing.assert_array_equal(gain[~ctx.active_mask], 0.0)
        np.testing.assert_array_equal(loss[~ctx.active_mask], 0.0)

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
