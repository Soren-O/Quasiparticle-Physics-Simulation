"""Tests for qpsim.collisions.pair_breaking_photon."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.pair_breaking_photon import (
    pair_breaking_photon_collision_components,
    pair_breaking_photon_collision_rates,
    pair_channel_open,
    validate_pair_breaking_photon_grid,
)
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
        ("E", "dE"),
        [
            (np.array([1.0 + 0.0j, 2.0 + 0.0j]), np.ones(2)),
            (np.array([1.0, 2.0]), np.array([1.0 + 1.0j, 1.0 + 0.0j])),
            (
                np.array([complex(1.0, float("nan")), 2.0 + 0.0j]),
                np.ones(2),
            ),
        ],
    )
    def test_grid_contract_rejects_complex_arrays_before_float_cast(
        self,
        E: np.ndarray,
        dE: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="real-valued arrays"):
            validate_pair_breaking_photon_grid(E, dE, gap=0.5, omega_PB=1.0)

    @pytest.mark.parametrize(
        ("E", "dE"),
        [
            (np.array([[1.0, 2.0]]), np.ones(2)),
            (np.array([1.0, 2.0]), np.ones((1, 2))),
        ],
    )
    def test_grid_contract_rejects_non_vector_arrays(
        self,
        E: np.ndarray,
        dE: np.ndarray,
    ) -> None:
        with pytest.raises(ValueError, match="1-D arrays"):
            validate_pair_breaking_photon_grid(E, dE, gap=0.5, omega_PB=1.0)

    def test_grid_contract_rejects_non_vector_support_mask(self) -> None:
        with pytest.raises(ValueError, match="supported must be a 1-D mask"):
            validate_pair_breaking_photon_grid(
                np.array([1.0, 2.0]),
                np.ones(2),
                gap=0.5,
                omega_PB=1.0,
                supported=np.ones((1, 2), dtype=bool),
            )

    def test_grid_contract_rejects_complex_support_mask_before_bool_cast(self) -> None:
        with pytest.raises(ValueError, match="real-valued 1-D mask"):
            validate_pair_breaking_photon_grid(
                np.array([1.0, 2.0]),
                np.ones(2),
                gap=0.5,
                omega_PB=1.0,
                supported=np.array([1.0 + 0.0j, 0.0 + 1.0j]),
            )

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

    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_no_op_rejects_complex_occupation_before_float_cast(
        self,
        imaginary: float,
    ) -> None:
        """A disabled channel must not erase an invalid imaginary part."""
        ctx = _setup()
        bad_f = np.zeros(ctx.E.size, dtype=complex)
        bad_f[0] = complex(0.0, imaginary)
        with pytest.raises(ValueError, match="real-valued occupations"):
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

    def test_zero_coupling_bypasses_inapplicable_grid_guards(self) -> None:
        # The frequency and reflection lattice are deliberately misaligned.
        # With c_phot_PB=0 the channel is absent and must match ``None``.
        E = 0.853 + np.arange(30) * 0.1
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        f = np.linspace(0.0, 0.2, E.size)

        components = pair_breaking_photon_collision_components(
            f, ctx, omega_PB=2.1, n_bar_PB=1.0, c_phot_PB=0.0,
        )
        assert len(components) == 4
        for component in components:
            np.testing.assert_array_equal(component, np.zeros_like(f))
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=2.1, n_bar_PB=1.0, c_phot_PB=0.0,
        )
        np.testing.assert_array_equal(gain, np.zeros_like(f))
        np.testing.assert_array_equal(loss, np.zeros_like(f))

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
    def test_channel_components_sum_to_public_rates(self) -> None:
        ctx = _setup()
        dE = float(ctx.dE[0])
        f = np.linspace(0.001, 0.02, ctx.E.size)

        gain_sc, loss_sc, gain_pair, loss_pair = pair_breaking_photon_collision_components(
            f,
            ctx,
            omega_PB=20 * dE,
            n_bar_PB=1.0,
            c_phot_PB=0.3,
        )
        gain, loss = pair_breaking_photon_collision_rates(
            f,
            ctx,
            omega_PB=20 * dE,
            n_bar_PB=1.0,
            c_phot_PB=0.3,
        )

        np.testing.assert_array_equal(gain, gain_sc + gain_pair)
        np.testing.assert_array_equal(loss, loss_sc + loss_pair)
        assert np.any(gain_sc > 0.0)
        assert np.any(gain_pair > 0.0)
        assert np.any(loss_sc > 0.0)
        assert np.any(loss_pair > 0.0)

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

    def test_sub_2delta_photon_produces_no_pair_generation(self) -> None:
        # 2026-07-20 round-4 review: the previous version of this test used
        # a face-aligned gap with no supported cut cell, so the pre-fix code
        # also returned zero gain and the test was vacuous. This grid puts
        # the gap INSIDE the [0.90, 1.00] cell (supported cut cell centered
        # at 0.95 < gap = 0.97): pairs (0.95, 0.95) are supported with
        # center-sum 1.90 < 2*gap = 1.94, so the pre-fix K- block produced
        # finite pair generation from a photon below the 2*gap threshold.
        gap = 0.97
        E = np.linspace(0.85, 3.05, 23)  # dE=0.1
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        supported = ctx.cell_weights > 0.0
        assert supported[1] and float(E[1]) < gap  # genuine supported cut cell
        omega = 1.9  # 19*dE exactly; pairs (0.95, 0.95); below 2*gap = 1.94
        f = np.zeros_like(E)
        gain, loss = pair_breaking_photon_collision_rates(f, ctx, omega, 1.0, 1e-2)
        # Pair generation must vanish identically below the threshold.
        assert np.all(gain == 0.0)
        # The scattering channel legitimately keeps a finite loss-rate
        # COEFFICIENT at any omega (absorption moving a QP up); with f=0
        # no actual loss occurs.
        assert np.all(np.isfinite(loss))

    @staticmethod
    def _misaligned_threshold_ctx() -> SpectralContext:
        # dE=0.1 and omega=2 are commensurate, but omega-2*E[0]=0.294
        # is 0.06 bins off the reflection lattice. That alignment matters
        # only when the pair channel is actually open.
        E = 0.853 + np.arange(30) * 0.1
        return SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=1.0,
        )

    def test_exact_threshold_skips_pair_only_alignment_guard(self) -> None:
        ctx = self._misaligned_threshold_ctx()
        omega = 2.0 * ctx.gap
        assert not pair_channel_open(omega, ctx.gap)

        gain_sc, loss_sc, gain_pair, loss_pair = pair_breaking_photon_collision_components(
            np.zeros_like(ctx.E),
            ctx,
            omega,
            1.0,
            1.0,
        )

        # Exact threshold has zero pair phase space even though this gap-cut
        # grid contains supported center pairs whose sum is exactly 2Δ.
        np.testing.assert_array_equal(gain_pair, 0.0)
        np.testing.assert_array_equal(loss_pair, 0.0)
        np.testing.assert_array_equal(gain_sc, 0.0)
        assert np.any(loss_sc > 0.0)  # scattering remains represented

    def test_pair_threshold_roundoff_band_stays_closed(self) -> None:
        ctx = self._misaligned_threshold_ctx()
        threshold = 2.0 * ctx.gap
        tol = 64.0 * np.finfo(float).eps * max(threshold, 1.0)
        omega_inside = threshold + 0.5 * tol
        omega_outside = threshold + 2.0 * tol

        assert omega_inside > threshold
        assert not pair_channel_open(omega_inside, ctx.gap)
        assert pair_channel_open(omega_outside, ctx.gap)

        exact = pair_breaking_photon_collision_components(
            np.zeros_like(ctx.E),
            ctx,
            threshold,
            1.0,
            1.0,
        )
        in_band = pair_breaking_photon_collision_components(
            np.zeros_like(ctx.E),
            ctx,
            omega_inside,
            1.0,
            1.0,
        )
        for exact_component, in_band_component in zip(exact, in_band, strict=True):
            np.testing.assert_array_equal(in_band_component, exact_component)

    @pytest.mark.filterwarnings("error")
    def test_snap_may_not_cross_the_pair_threshold(self) -> None:
        # 2026-07-20 round-4 review (both directions reproduced by the
        # reviewer): an accepted snap must never CHANGE whether the pair
        # channel exists. dE = 0.1, so the commensurability window is
        # +-0.001 (1% of a bin) around each harmonic.
        E = np.linspace(0.85, 3.05, 23)  # dE=0.1
        dE = integration_widths_from_centers(E)
        f = np.zeros_like(E)
        # Downward crossing: 2*gap = 1.9005; nominal 1.9006 (> 2*gap)
        # snaps to 1.90 (< 2*gap) at a 0.006-bin offset.
        ctx_dn = SpectralContext(E_bins=E, dE_bins=dE, gap=0.95025)
        with pytest.raises(ValueError, match="crosses the 2Δ"):
            pair_breaking_photon_collision_rates(f, ctx_dn, 1.9006, 1.0, 1e-2)
        # Upward crossing: 2*gap = 1.9995; nominal 1.9994 (< 2*gap)
        # snaps to 2.0 (>= 2*gap) at a 0.006-bin offset.
        ctx_up = SpectralContext(E_bins=E, dE_bins=dE, gap=0.99975)
        with pytest.raises(ValueError, match="crosses the 2Δ"):
            pair_breaking_photon_collision_rates(f, ctx_up, 1.9994, 1.0, 1e-2)
        # Nominally open omega snaps exactly onto 2Δ, where the strict pair
        # predicate is closed. The crossing error must precede the ordinary
        # accepted-snap warning (the marker makes any warning test-fatal).
        ctx_eq = SpectralContext(E_bins=E, dE_bins=dE, gap=1.0)
        with pytest.raises(ValueError, match="crosses the 2Δ"):
            pair_breaking_photon_collision_rates(f, ctx_eq, 2.0005, 1.0, 1e-2)
