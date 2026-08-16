"""Regressions for review packet P11 (2026-08-03).

Covers the gap-cut ``q = 0`` face weight (previously unwitnessed by any
benchmark gate), the shape-only semantics of the effective phonon
temperature, and the coupled-Newton projected-vacuum collapse whose repair is
held back for physicist sign-off.
"""

# ruff: noqa: N999  # filename is the review packet id, not a module name

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.observables.effective_temperature import effective_phonon_temperature
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.solvers.coupled_newton import coupled_newton_solve
from validation.diffusion_operators.uniform_gap_packet import run


def _support_fraction(E: np.ndarray, gap: float) -> np.ndarray:
    """Above-gap fraction of each uniform cell, recomputed independently."""
    dE = float(E[1] - E[0])
    edges = (float(E[0]) - 0.5 * dE) + dE * np.arange(E.size + 1)
    return np.clip((edges[1:] - np.maximum(edges[:-1], gap)) / dE, 0.0, 1.0)


class TestGapCutLongitudinalFaceWeight:
    """The dirty-limit ``q = 0`` face weight on a partially represented cell.

    Every benchmark in ``validation/diffusion_operators`` builds its grid above
    the local gap, so ``support_fraction`` is identically one and the repaired
    branch in ``T3Spatial1DBackend._build_transport_operators`` is
    indistinguishable from the unrepaired ``flux_weight(D0, N1, 0)``. Lowering
    the packet benchmark's grid floor below Δ separates them by ``1 /
    support_fraction``.
    """

    def test_measured_deff_carries_the_support_fraction(self) -> None:
        result = run(
            NE=14, NX=21, dt=2.0, conservation_steps=2, energy_min_factor=0.9
        )
        support = _support_fraction(result.E, result.gap)
        cut = np.flatnonzero((support > 0.0) & (support < 1.0))
        assert cut.size >= 1, "grid floor did not produce a gap-cut cell"
        # A1 is q = 0, p = 1: D_eff/D_N = support_fraction / N_1 exactly.
        measured = result.deff_over_dn["A1"] * result.N1
        np.testing.assert_allclose(measured, support, rtol=1e-9, atol=0.0)
        # Discriminating power: the unrepaired kernel would put 1.0 here.
        assert support[cut[0]] < 0.7
        assert abs(measured[cut[0]] - 1.0) > 0.3

    def test_benchmark_oracle_agrees_on_the_cut_cell(self) -> None:
        result = run(
            NE=14, NX=21, dt=2.0, conservation_steps=2, energy_min_factor=0.9
        )
        support = _support_fraction(result.E, result.gap)
        assert np.any(support < 1.0)
        for name in ("A1", "A1P", "A2", "C", "B"):
            measured = result.deff_over_dn[name]
            analytic = result.analytic_over_dn[name]
            rel = np.max(np.abs(measured - analytic) / np.abs(analytic))
            assert rel < 1e-9, (name, rel)


class TestEffectivePhononTemperatureIsShapeOnly:
    """The fit measures shape, not amplitude — as the module docstring says."""

    def test_ten_times_the_bath_band_is_reported_as_unheated(self) -> None:
        gap, T_bath = 180.0, 0.2
        omega = np.linspace(2.0 * gap, 12.0 * gap, 40)
        n_bath = 1.0 / np.expm1(omega / (KB_UEV_PER_K * T_bath))
        hot = effective_phonon_temperature(
            10.0 * n_bath, omega, gap, T_bath=T_bath, T_max=5.0
        )
        assert hot == pytest.approx(T_bath, rel=1e-12)


def _undriven_ph0_setup(T_c: float = 1.2, T_bath: float = 0.2, NE: int = 40):
    """Undriven Ph0 problem whose root is the unforced thermal fixed point."""
    gap = 1.764 * KB_UEV_PER_K * T_c
    dE = gap / 8.0
    E = gap + dE * (np.arange(NE) + 0.5)
    ctx = SpectralContext(E_bins=E, dE_bins=np.full(NE, dE), gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
    n_th = thermal_phonon_occupation(omega, T_bath)
    return ctx, K_s0, K_r0, omega, idx_diff, idx_sum, diff_sign, n_th


# The strict xfail that used to stand here is DELETED, which is this item's
# own same-commit requirement (HELD-BACK-ADJUDICATION-2026-08-11.md, "P11
# coupled-Newton projected-vacuum guard"). It held the defect open until the
# guard landed, and it worked exactly as designed: the marker was made strict
# precisely so that an XPASS would report as a FAILURE rather than a silent
# pass, and that failure is what surfaced the landed guard during the first
# full-suite run. Nothing else found it -- the twelve directories exercised
# during the fix all pass either way.
#
# The guard is now in coupled_newton_solve: a trial that clips every active
# bin to the exact vacuum is rejected and the step backtracked. Measured over
# 30 perturbed cold seeds, 14/30 solves became 25/30, and no returned answer
# changed.
def test_flat_hot_seed_reaches_the_unforced_thermal_root() -> None:
    ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, n_th = _undriven_ph0_setup()
    root = float(np.max(fermi_dirac_occupation(ctx.E, 0.2)))
    f, _ = coupled_newton_solve(
        ctx,
        np.full(ctx.E.size, 1e-3),
        n_th.copy(),
        omega_bins=omega,
        omega_idx_diff=idx_d,
        omega_idx_sum=idx_s,
        diff_sign=sgn,
        K_s0=K_s0,
        K_r0=K_r0,
        T_bath=0.2,
        tau_l=2.0,
        tol=1e-14,
        step_rtol=1e-8,
        max_iter=120,
        analytic_cross=True,
    )
    assert float(np.max(f)) == pytest.approx(root, rel=1e-6)


def test_shaped_seeds_still_reach_the_unforced_thermal_root() -> None:
    """Guardrail for the held-back repair: these seeds must stay converging."""
    ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, n_th = _undriven_ph0_setup()
    root = float(np.max(fermi_dirac_occupation(ctx.E, 0.2)))
    for scale in (1e-1, 1e-4, 1e-5):
        f, _ = coupled_newton_solve(
            ctx,
            np.full(ctx.E.size, scale),
            n_th.copy(),
            omega_bins=omega,
            omega_idx_diff=idx_d,
            omega_idx_sum=idx_s,
            diff_sign=sgn,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.2,
            tau_l=2.0,
            tol=1e-14,
            step_rtol=1e-8,
            max_iter=120,
            analytic_cross=True,
        )
        assert float(np.max(f)) == pytest.approx(root, rel=1e-6), scale
