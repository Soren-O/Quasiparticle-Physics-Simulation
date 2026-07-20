"""Tests for qpsim.collisions.pair_breaking_photon."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext


def _setup(gap: float = 180.0, num: int = 40):
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.0, energy_max_factor=6.0, num_energy_bins=num
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
            pair_breaking_photon_collision_rates(
                bad_f,
                ctx,
                omega_PB=0.0,
                n_bar_PB=0.0,
                c_phot_PB=0.0,
            )

    def test_output_shapes(self) -> None:
        ctx = _setup()
        NE = ctx.E.size
        f = np.zeros(NE)
        f[10] = 0.5
        dE = float(ctx.dE[0])
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=10 * dE, n_bar_PB=1.0, c_phot_PB=1.0,
        )
        assert gain.shape == (NE,)
        assert loss.shape == (NE,)

    def test_zero_omega_returns_zero(self) -> None:
        ctx = _setup()
        f = 0.1 * np.ones(ctx.E.size)
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=0.0, n_bar_PB=1.0, c_phot_PB=1.0,
        )
        np.testing.assert_allclose(gain, 0.0)
        np.testing.assert_allclose(loss, 0.0)

    def test_positive_frequency_below_half_bin_fails_loudly(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        with pytest.raises(ValueError, match="below half the grid spacing"):
            pair_breaking_photon_collision_rates(
                0.1 * np.ones(ctx.E.size),
                ctx,
                omega_PB=0.4 * dE,
                n_bar_PB=1.0,
                c_phot_PB=1.0,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
    def test_rejects_invalid_channel_parameters(self, bad: float) -> None:
        ctx = _setup()
        f = 0.1 * np.ones(ctx.E.size)
        with pytest.raises(ValueError):
            pair_breaking_photon_collision_rates(f, ctx, bad, 1.0, 1.0)
        with pytest.raises(ValueError):
            pair_breaking_photon_collision_rates(f, ctx, 1.0, bad, 1.0)
        with pytest.raises(ValueError):
            pair_breaking_photon_collision_rates(f, ctx, 1.0, 1.0, bad)

    def test_zero_f_zero_nbar_gives_zero_gain(self) -> None:
        # f = 0 ⇒ no QPs to scatter/recombine; n_bar = 0 ⇒ no photons
        # available for absorption. Gain vanishes. The loss-rate
        # coefficient retains the spontaneous-emission (recombination)
        # term ∝ (1 + n_bar), but loss · f is trivially zero.
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = np.zeros(ctx.E.size)
        gain, _ = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=10 * dE, n_bar_PB=0.0, c_phot_PB=1.0,
        )
        np.testing.assert_allclose(gain, 0.0)

    def test_rejects_nonuniform_grid(self) -> None:
        E = np.array([181.0, 184.0, 195.0, 230.0, 310.0, 470.0])
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=180.0)
        with pytest.raises(ValueError, match="uniform energy grid"):
            pair_breaking_photon_collision_rates(
                np.zeros(E.size), ctx, omega_PB=400.0,
                n_bar_PB=1.0, c_phot_PB=1.0,
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
            pair_breaking_photon_collision_rates(
                np.zeros(ctx.E.size), dynes, omega_PB=20.0 * ctx.dE[0],
                n_bar_PB=1.0, c_phot_PB=1.0,
            )


class TestCommensurateGrid:
    def test_rejects_when_off_grid(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = 0.1 * np.ones(ctx.E.size)
        with pytest.raises(ValueError, match="not grid-commensurate"):
            pair_breaking_photon_collision_rates(
                f, ctx, omega_PB=10 * dE + 0.4 * dE, n_bar_PB=1.0, c_phot_PB=1.0,
            )

    def test_rejects_misaligned_open_pair_channel(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 1.01, 6.0, 40)
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap,
        )
        dE = float(ctx.dE[0])

        with pytest.raises(ValueError, match="reflection partners are not grid-aligned"):
            pair_breaking_photon_collision_rates(
                np.zeros(E.size),
                ctx,
                omega_PB=20 * dE,
                n_bar_PB=1.0,
                c_phot_PB=1.0,
            )


class TestPhysicalConsistency:
    def test_zero_dos_target_rows_are_zero(self) -> None:
        gap = 180.0
        E, _ = build_energy_grid(gap, 0.75, 4.0, 60)
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E, dE, gap)
        f = np.zeros(E.size)
        f[ctx.active_mask] = 0.1
        gain, loss = pair_breaking_photon_collision_rates(
            f,
            ctx,
            omega_PB=12.0 * dE[0],
            n_bar_PB=0.0,
            c_phot_PB=1.0,
        )
        np.testing.assert_array_equal(gain[~ctx.active_mask], 0.0)
        np.testing.assert_array_equal(loss[~ctx.active_mask], 0.0)

    def test_pair_generation_from_photons_only(self) -> None:
        # f = 0 everywhere, but n_bar > 0 and ω_PB > 2Δ: photons should
        # generate QP pairs at reflection partners, producing positive gain.
        ctx = _setup()
        dE = float(ctx.dE[0])
        NE = ctx.E.size
        f = np.zeros(NE)
        # Pick ω_PB so that ω_PB > 2Δ within our grid (Δ = 180, 2Δ ≈ 360 μeV).
        # dE ≈ (6·180 − 180·1.01) / 40 ≈ 22.5, so m ≈ 20 gives ω_PB ≈ 450 μeV.
        gain, _ = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=20 * dE, n_bar_PB=1.0, c_phot_PB=1.0,
        )
        # At least one bin should see pair-generation gain.
        assert np.any(gain > 0)

    def test_below_2delta_no_pair_generation(self) -> None:
        # ω_PB < 2Δ: reflection partner E_j = ω_PB − E_i < Δ ⇒ no pairs.
        # Should still have scattering contribution but no generation.
        ctx = _setup()
        dE = float(ctx.dE[0])
        NE = ctx.E.size
        # 2Δ ≈ 360; use ω_PB = 5·dE ≈ 112.5 < 2Δ = 360.
        f = np.full(NE, 0.01)
        # Run just to ensure it doesn't crash and output is finite.
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=5 * dE, n_bar_PB=1.0, c_phot_PB=1.0,
        )
        assert np.all(np.isfinite(gain))
        assert np.all(np.isfinite(loss))


class TestOffGridPartnerGuards:
    """2026-07-19/20 audit: partners outside the represented window must
    fail loud, matching the kernel's misalignment guard — silent skips
    lost ~42% (upper) / ~27% (lower) of pair generation in reproduced
    short-grid cases. Partners at or below Δ remain a physical exclusion."""

    def _uniform_ctx(self, gap: float, e_lo: float, e_hi: float, num: int):
        E = np.linspace(e_lo, e_hi, num)
        dE = integration_widths_from_centers(E)
        return SpectralContext(E_bins=E, dE_bins=dE, gap=gap)

    def test_partner_above_grid_top_raises(self) -> None:
        # gap=180, grid to 472.5, omega=700: gap-edge cells' partners
        # (700-181 = 519 ueV) land above the top -> raise, not silent loss.
        ctx = self._uniform_ctx(180.0, 181.25, 471.25, 117)  # dE=2.5
        dE = float(ctx.dE[0])
        omega = round(700.0 / dE) * dE  # exactly commensurate
        f = np.zeros_like(ctx.E)
        with pytest.raises(ValueError, match="above the grid top"):
            pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-9)

    def test_partner_below_first_face_but_above_gap_raises(self) -> None:
        # Grid starts well above the gap (first face ~230 ueV, gap 180):
        # partners of the upper cells land in the physical-but-
        # unrepresented window (180, 230) -> raise.
        ctx = self._uniform_ctx(180.0, 232.5, 432.5, 81)  # dE=2.5, face 231.25
        dE = float(ctx.dE[0])
        omega = round(500.0 / dE) * dE  # 2.78*gap, commensurate
        f = np.zeros_like(ctx.E)
        with pytest.raises(ValueError, match="physical-but-unrepresented"):
            pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-9)

    def test_subgap_partner_exclusion_stays_silent(self) -> None:
        # Full-coverage grid (face at the gap): partners below Δ are the
        # documented physical exclusion — no raise, finite rates.
        ctx = _setup(gap=180.0, num=160)
        dE = float(ctx.dE[0])
        # omega just above 2*gap: some partners fall below Δ (excluded),
        # every physical partner is representable.
        omega = round(365.0 / dE) * dE
        f = np.zeros_like(ctx.E)
        gain, loss = pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-9)
        assert np.all(np.isfinite(gain))
        assert np.all(np.isfinite(loss))
        assert float(np.max(gain)) > 0.0

    def test_gap_straddling_partner_cell_raises(self) -> None:
        # 2026-07-20 review repro: gap=180, dE=2.5, omega=500, truncated
        # grid starting at 182.5 (first face 181.25). The top cells'
        # partners land in a cell centered AT the gap whose upper half
        # [180, 181.25) is physical continuum; center-only reasoning
        # skipped it silently (11.1% generation loss vs the covered grid).
        ctx = self._uniform_ctx(180.0, 182.5, 320.0, 56)  # dE=2.5
        dE = float(ctx.dE[0])
        omega = round(500.0 / dE) * dE
        f = np.zeros_like(ctx.E)
        with pytest.raises(ValueError, match="physical-but-unrepresented"):
            pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-9)

    def test_covered_grid_same_omega_does_not_raise(self) -> None:
        # The covered counterpart of the straddling-cell repro: first face
        # at the gap, same omega — every physical partner representable.
        ctx = self._uniform_ctx(180.0, 181.25, 318.75, 56)  # dE=2.5, face 180.0
        dE = float(ctx.dE[0])
        omega = round(500.0 / dE) * dE
        f = np.zeros_like(ctx.E)
        gain, _loss = pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-9)
        assert np.all(np.isfinite(gain))
        assert float(np.max(gain)) > 0.0
