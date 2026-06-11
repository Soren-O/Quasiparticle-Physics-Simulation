"""Tests for qpsim.solvers.coupled_newton."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_jacobian_nph,
    phonon_occupation_matrices_from_state,
    phonon_source_sink_jacobian_f,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import (
    solve_steady_state,
)
from qpsim.solvers.coupled_newton import coupled_newton_solve
from qpsim.solvers.newton_steady_state import _gain_loss_sum


def _thermal_setup(T_bath: float = 0.3, T_c: float = 1.2, num: int = 18):
    """Modest-size thermal setup (keeps FD Jacobian cost manageable)."""
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
    return ctx, K_s0, K_r0, omega_bins, idx_diff, idx_sum, diff_sign, T_bath


class TestCoupledNewtonSolve:
    def test_thermal_equilibrium_is_fixed_point(self) -> None:
        # At thermal equilibrium, (f_FD, n_BE) is the joint fixed point
        # of the coupled residual; one Newton pass shouldn't move it.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        f_out, n_out = coupled_newton_solve(
            ctx, f_FD, n_BE,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.25,
            tol=1e-8,
        )
        np.testing.assert_allclose(f_out, f_FD, atol=1e-8)
        np.testing.assert_allclose(n_out, n_BE, atol=1e-8)

    def test_converges_from_perturbed_initial(self) -> None:
        # Perturb f smoothly; coupled Newton should converge back to thermal.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)
        f0 = np.clip(f_FD * (1.0 + 0.2 * np.cos(ctx.E / ctx.gap)), 0.0, 1.0)

        f_out, n_out = coupled_newton_solve(
            ctx, f0, n_BE,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.25,
            tol=1e-8,
        )
        np.testing.assert_allclose(f_out, f_FD, atol=1e-6)
        np.testing.assert_allclose(n_out, n_BE, atol=1e-6)

    def test_step_rtol_refines_warm_seed_below_abs_tol(self) -> None:
        # Regression for the Fischer Fig. 6 warm-continuation freeze. At
        # T_bath=0.1 K the fixed point has f ~ 1e-9, so a warm seed only a few
        # percent off already has a residual below any reasonable absolute tol.
        # tol-only therefore early-exits at iteration 0 with the stale seed;
        # the scale-invariant step_rtol forces a refining step and converges to
        # the true fixed point. Pins the actual failure mode, not the FD math.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(T_bath=0.1)
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)
        f_seed = np.clip(f_FD * 1.05, 0.0, 1.0)  # 5% off the fixed point

        common = {
            "omega_bins": omega, "omega_idx_diff": idx_d, "omega_idx_sum": idx_s,
            "diff_sign": sgn, "K_s0": K_s0, "K_r0": K_r0,
            "T_bath": T_bath, "tau_l": 0.25,
        }
        # Loose absolute tol → early-exit at iteration 0 with the stale seed.
        f_abs, _ = coupled_newton_solve(ctx, f_seed, n_BE, tol=1e-1, **common)
        # Same loose tol as a safety floor, but step_rtol drives refinement.
        f_rel, _ = coupled_newton_solve(
            ctx, f_seed, n_BE, tol=1e-1, step_rtol=1e-8, **common
        )

        scale = float(np.max(np.abs(f_FD)))
        rel_err_abs = float(np.max(np.abs(f_abs - f_FD))) / scale
        rel_err_rel = float(np.max(np.abs(f_rel - f_FD))) / scale
        assert rel_err_abs > 1e-2   # tol-only stayed ~5% off (frozen on seed)
        assert rel_err_rel < 1e-4   # step_rtol refined to the fixed point

    def test_matches_picard_on_shared_case(self) -> None:
        # Where Picard converges (thermal case), coupled Newton should
        # land on the same (f, n_ph) within tolerance.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=15)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)

        # Picard via the service orchestrator.
        phonon_out: dict[str, np.ndarray] = {}
        f_picard = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            initial_guess=f_init,
            phonon_escape_time=0.5,
            tol=1e-12, picard_tol=1e-10,
            phonon_out=phonon_out,
        )
        n_picard = phonon_out["n_ph"]

        # Coupled Newton from the same initial.
        f_newton, n_newton = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.5,
            tol=1e-10,
        )

        np.testing.assert_allclose(f_newton, f_picard, atol=1e-6)
        np.testing.assert_allclose(n_newton, n_picard, atol=1e-6)

    def test_output_shapes(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=12)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        f_out, n_out = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.1,
            tol=1e-8,
        )
        assert f_out.shape == (ctx.E.size,)
        assert n_out.shape == omega.shape

    def test_rejects_mismatched_n_ph_init(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=10)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        with pytest.raises(ValueError, match="n_ph_init length"):
            coupled_newton_solve(
                ctx, f_init, np.zeros(omega.size + 5),  # wrong length
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.1,
            )

    def test_zero_tau_l_branch(self) -> None:
        # τ_l = 0: no substrate coupling. Residual R_ph reduces to
        # a_ph + b_ph · n_ph. The solver should still converge.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=12)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        f_out, n_out = coupled_newton_solve(
            ctx, f_init, n_init,
            omega_bins=omega,
            omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
            K_s0=K_s0, K_r0=K_r0,
            T_bath=T_bath, tau_l=0.0,
            tol=1e-8,
        )
        # Just check everything is finite and bounded.
        assert np.all(np.isfinite(f_out))
        assert np.all(np.isfinite(n_out))
        assert np.all(f_out >= 0.0)
        assert np.all(f_out <= 1.0)
        assert np.all(n_out >= 0.0)


def _generic_cross_state(ctx, omega):
    """Smooth, well-scaled non-equilibrium (f, n_ph) for cross-Jacobian FD checks.

    Deliberately off any fixed point so every Jacobian entry is generic and
    nonzero, and O(0.01..1) so central differences are well conditioned.
    """
    E = ctx.E
    f = np.clip(0.3 * np.exp(-(E - ctx.gap) / ctx.gap) + 0.02, 1e-4, 0.9)
    n_ph = 0.5 * np.exp(-omega / ctx.gap) + 0.1
    return f, n_ph


class TestAnalyticCrossJacobian:
    """Closed-form cross blocks must equal well-scaled finite differences.

    Pins :issue:`coupled-newton-analytical-cross`: the analytic
    ``J_fn = ∂R_f/∂n_ph`` and ``J_nf = ∂R_ph/∂f`` replace the O(NE³),
    scale-fragile FD secant that branch-hops at strong drive (Fischer Fig. 6).
    Checked on both the QP-side and the phonon-side (fig6) kernel path.
    """

    TAU_L = 0.25

    def _residual(self, f, n_ph, ctx, K_s0, K_r0, maps, T_bath, ps):
        idx_d, idx_s, sgn, omega = maps
        N_p, N_e, N_a = phonon_occupation_matrices_from_state(n_ph, idx_d, idx_s, sgn)
        gain, loss = _gain_loss_sum(
            f, ctx, K_s0, K_r0, T_bath, None, None, N_p, N_e, N_a, None
        )
        R_f = gain - loss * f
        a, b = compute_phonon_source_sink(
            f, ctx, K_s0, K_r0, idx_d, idx_s, sgn, omega.size,
            K_s0_phonon_side=ps[0], K_r0_phonon_side=ps[1],
        )
        n_th = thermal_phonon_occupation(omega, T_bath)
        R_ph = a + b * n_ph + (n_th - n_ph) / self.TAU_L
        return R_f, R_ph

    @pytest.mark.parametrize("phonon_side", [False, True])
    def test_cross_blocks_match_central_fd(self, phonon_side: bool) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        maps = (idx_d, idx_s, sgn, omega)
        N_omega, NE = omega.size, ctx.E.size
        f, n_ph = _generic_cross_state(ctx, omega)
        if phonon_side:
            ps = (
                build_scattering_kernel_phonon_side(ctx, 0.255),
                build_recombination_kernel_phonon_side(ctx, 0.255),
            )
        else:
            ps = (None, None)

        # Analytic blocks.
        J_fn = phonon_collision_jacobian_nph(
            f, ctx, K_s0, K_r0, idx_d, idx_s, sgn, N_omega
        )
        da_df, db_df = phonon_source_sink_jacobian_f(
            f, ctx, K_s0, K_r0, idx_d, idx_s, sgn, N_omega,
            K_s0_phonon_side=ps[0], K_r0_phonon_side=ps[1],
        )
        J_nf = da_df + n_ph[:, None] * db_df

        # Central-difference reference (truncation ~h² ≈ 1e-14, roundoff ~1e-9).
        h = 1e-7
        J_fn_fd = np.zeros((NE, N_omega))
        for k in range(N_omega):
            up = n_ph.copy()
            up[k] += h
            dn = n_ph.copy()
            dn[k] -= h
            R_up, _ = self._residual(f, up, ctx, K_s0, K_r0, maps, T_bath, ps)
            R_dn, _ = self._residual(f, dn, ctx, K_s0, K_r0, maps, T_bath, ps)
            J_fn_fd[:, k] = (R_up - R_dn) / (2.0 * h)
        J_nf_fd = np.zeros((N_omega, NE))
        for j in range(NE):
            up = f.copy()
            up[j] += h
            dn = f.copy()
            dn[j] -= h
            _, R_up = self._residual(up, n_ph, ctx, K_s0, K_r0, maps, T_bath, ps)
            _, R_dn = self._residual(dn, n_ph, ctx, K_s0, K_r0, maps, T_bath, ps)
            J_nf_fd[:, j] = (R_up - R_dn) / (2.0 * h)

        def rel(A, F):
            return np.max(np.abs(A - F)) / (np.max(np.abs(F)) + 1e-300)

        assert rel(J_fn, J_fn_fd) < 1e-6
        assert rel(J_nf, J_nf_fd) < 1e-6

    def test_analytic_and_fd_solve_agree(self) -> None:
        # End-to-end: the analytic-cross solve lands on the same (f, n_ph) as the
        # default FD path on a case both converge cleanly (Newton finds the same
        # root regardless of how the cross-Jacobian is built).
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_th = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        f0 = np.clip(f_th * 1.1, 0.0, 1.0)
        n0 = thermal_phonon_occupation(omega, T_bath)
        common = {
            "omega_bins": omega, "omega_idx_diff": idx_d, "omega_idx_sum": idx_s,
            "diff_sign": sgn, "K_s0": K_s0, "K_r0": K_r0, "T_bath": T_bath,
            "tau_l": 0.25, "tol": 1e-12,
        }
        f_fd, n_fd = coupled_newton_solve(ctx, f0, n0, analytic_cross=False, **common)
        f_an, n_an = coupled_newton_solve(ctx, f0, n0, analytic_cross=True, **common)
        np.testing.assert_allclose(f_an, f_fd, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(n_an, n_fd, rtol=1e-6, atol=1e-9)
