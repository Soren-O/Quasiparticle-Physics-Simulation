"""Layer-1 validation — detailed balance at thermal equilibrium.

NFP §6.1: at ``(f = f_FD(T), n_ph = n_BE(T))`` the full collision
integral (e-ph + whatever photon channel we're exercising) must
vanish to roundoff. These tests fix ``T_bath`` and a set of
physically consistent phonon / photon occupations and assert the
residual ``gain − loss_rate · f`` is small everywhere in the active
window.

The e-ph channel is already covered in
``tests/collisions/test_phonon.py::TestDetailedBalance`` at modest
scale; this module adds the two photon-channel tests that weren't
previously formalized, and re-runs the e-ph check at Fischer-like
grid resolution.
"""

from __future__ import annotations

import numpy as np
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    phonon_collision_rates,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext


def _fischer_like_ctx(num: int = 405) -> tuple[SpectralContext, float]:
    """Build a Fischer-Table-I grid and return (ctx, T_c)."""
    delta_0 = 180.0
    T_c = delta_0 / (1.764 * KB_UEV_PER_K)
    E, _ = build_energy_grid(
        gap=delta_0, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    return SpectralContext(E_bins=E, dE_bins=dE, gap=delta_0), T_c


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


class TestEphDetailedBalance:
    """e-ph collision integral vanishes at thermal equilibrium at Fischer grid."""

    def test_vanishes_at_T_bath(self) -> None:
        ctx, T_c = _fischer_like_ctx(num=405)
        T_bath = 0.1
        f = _fermi_dirac(ctx.E, T_bath)

        K_s0 = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=T_c)
        K_r0 = build_recombination_kernel_base(ctx, tau_0=438.0, T_c=T_c)

        gain, loss = phonon_collision_rates(f, ctx, K_s0, K_r0, T_bath)
        residual = gain - loss * f
        scale = np.maximum(np.abs(gain) + np.abs(loss * f), 1e-30)
        rel = np.abs(residual / scale)
        active = ctx.active_mask
        # Detailed balance to roundoff on the active window.
        assert float(np.max(rel[active])) < 1e-10


class TestSubGapPhotonDetailedBalance:
    """Sub-gap photon scattering vanishes at thermal with n̄ = n_BE(ω₀, T)."""

    def test_vanishes_when_photon_at_bath_temperature(self) -> None:
        ctx, _ = _fischer_like_ctx(num=405)
        T_bath = 0.1
        omega_0 = ctx.gap / 9.0  # sub-gap, commensurate at NE=405 (m=5)
        n_bar_thermal = 1.0 / (np.exp(omega_0 / (KB_UEV_PER_K * T_bath)) - 1.0)
        f = _fermi_dirac(ctx.E, T_bath)

        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=omega_0, n_bar=n_bar_thermal, c_phot=1.0,
        )
        residual = gain - loss * f
        scale = np.maximum(np.abs(gain) + np.abs(loss * f), 1e-30)
        rel = np.abs(residual / scale)

        # Photon partners at bins i ± m only populate an ~m-bin interior.
        # The boundary bins can't balance because their partners lie off-grid,
        # so we only assert detailed balance on the bulk.
        m = round(omega_0 / float(ctx.dE[0]))
        bulk = np.zeros_like(ctx.E, dtype=bool)
        bulk[m + 1 : -m - 1] = True
        bulk &= ctx.active_mask
        assert float(np.max(rel[bulk])) < 1e-8

    def test_fails_when_photon_is_hotter_than_bath(self) -> None:
        # Sanity: the test really is testing something — putting the photon
        # mode out of thermal equilibrium (hotter than the QP bath) should
        # produce a non-vanishing residual.
        ctx, _ = _fischer_like_ctx(num=405)
        T_bath = 0.1
        omega_0 = ctx.gap / 9.0
        n_bar_thermal = 1.0 / (np.exp(omega_0 / (KB_UEV_PER_K * T_bath)) - 1.0)
        f = _fermi_dirac(ctx.E, T_bath)

        gain, loss = sub_gap_photon_collision_rates(
            f, ctx, omega_0=omega_0, n_bar=n_bar_thermal * 10.0, c_phot=1.0,
        )
        residual = gain - loss * f
        assert float(np.max(np.abs(residual[ctx.active_mask]))) > 0.0


class TestPairBreakingPhotonDetailedBalance:
    """PB photon channel vanishes at thermal with n̄_PB = n_BE(ω_PB, T)."""

    def test_vanishes_when_photon_at_bath_temperature(self) -> None:
        ctx, _ = _fischer_like_ctx(num=810)  # finer grid; PB needs ω > 2Δ
        T_bath = 0.2
        omega_PB = 2.8 * ctx.gap  # commensurate at NE=810 (m=252)
        n_bar_thermal = 1.0 / (np.exp(omega_PB / (KB_UEV_PER_K * T_bath)) - 1.0)
        f = _fermi_dirac(ctx.E, T_bath)

        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=omega_PB, n_bar_PB=n_bar_thermal, c_phot_PB=1.0,
        )
        residual = gain - loss * f
        active = ctx.active_mask

        # Every implemented process obeys detailed balance wherever its
        # partner lies on-grid; off-grid processes simply contribute zero to
        # both directions.  The former m-bin guard removed the entire
        # pair-generation/recombination window, so it only tested the safer
        # scattering terms.  Exercise all active bins, including the reflected
        # pair partner E_j = ω_PB - E_i.
        scale = np.abs(gain) + np.abs(loss * f)
        nonzero = active & (scale > np.finfo(float).tiny)
        assert np.any(nonzero)
        rel = np.abs(residual[nonzero]) / scale[nonzero]
        max_rel = float(np.max(rel))
        assert max_rel < 1e-12, (
            f"PB detailed balance failed: max relative residual = {max_rel:.2e}"
        )

    def test_nonzero_when_photon_is_not_thermal(self) -> None:
        ctx, _ = _fischer_like_ctx(num=810)
        T_bath = 0.2
        omega_PB = 2.8 * ctx.gap
        f = _fermi_dirac(ctx.E, T_bath)

        # Much larger n̄ than thermal → strong net pair generation.
        gain, loss = pair_breaking_photon_collision_rates(
            f, ctx, omega_PB=omega_PB, n_bar_PB=1e6, c_phot_PB=1e-9,
        )
        residual = gain - loss * f
        assert float(np.max(np.abs(residual[ctx.active_mask]))) > 0.0
