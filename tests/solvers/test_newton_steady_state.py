"""Tests for qpsim.solvers.newton_steady_state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import newton_solve_f


def _setup(T_bath: float = 0.3, T_c: float = 1.2, num: int = 30):
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    return ctx, K_s0, K_r0, T_bath


class TestNewtonSolveF:
    def test_thermal_fixed_point(self) -> None:
        # Starting from Fermi-Dirac at T_bath, Newton should barely
        # move it: f_FD is the fixed point of the e-ph collision integral
        # at thermal equilibrium.
        ctx, K_s0, K_r0, T_bath = _setup()
        kT = KB_UEV_PER_K * T_bath
        f0 = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        f_star = newton_solve_f(
            ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            tol=1e-12, max_iter=50,
        )
        np.testing.assert_allclose(f_star, f0, atol=1e-8)

    def test_converges_from_perturbed_initial(self) -> None:
        # A smooth perturbation around Fermi-Dirac should still converge
        # back to it since it's the unique steady state of the e-ph
        # collision integral at thermal phonons.
        ctx, K_s0, K_r0, T_bath = _setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        # Perturb by a smooth multiplicative factor.
        f0 = np.clip(f_FD * (1.0 + 0.5 * np.sin(ctx.E / ctx.gap)), 0.0, 1.0)
        f_star = newton_solve_f(
            ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            tol=1e-12, max_iter=100,
        )
        np.testing.assert_allclose(f_star, f_FD, atol=1e-6)

    def test_all_active_false_returns_initial(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        f0 = np.full(ctx.E.size, 0.1)
        mask = np.zeros(ctx.E.size, dtype=bool)
        got = newton_solve_f(
            ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, active=mask,
        )
        np.testing.assert_allclose(got, f0)

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf"), -1e-12, 1.0 + 1e-12],
    )
    def test_rejects_nonphysical_initial_occupation(self, bad_value: float) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        f0 = np.full(ctx.E.size, 0.1)
        f0[0] = bad_value

        with pytest.raises(ValueError, match="initial occupation"):
            newton_solve_f(
                ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            )

    def test_rejects_non_vector_initial_occupation(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        f0 = np.full((1, ctx.E.size), 0.1)

        with pytest.raises(ValueError, match="must have shape"):
            newton_solve_f(
                ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            )

    def test_overrides_match_thermal(self) -> None:
        # Explicitly passing the thermal occupation matrices as overrides
        # must agree with the default thermal path.
        from qpsim.collisions.phonon import (
            _thermal_phonon_recombination_occupations,
            _thermal_phonon_scattering_occupation,
        )

        ctx, K_s0, K_r0, T_bath = _setup()
        kT = KB_UEV_PER_K * T_bath
        f0 = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        f_default = newton_solve_f(
            ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, tol=1e-12
        )
        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        f_explicit = newton_solve_f(
            ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, tol=1e-12,
            N_p_override=N_p, N_emit_override=N_emit, N_abs_override=N_abs,
        )
        np.testing.assert_allclose(f_explicit, f_default, atol=1e-10)
