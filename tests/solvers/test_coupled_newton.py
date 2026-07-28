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
from qpsim.devices.external_flux import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.services.steady_state import (
    solve_steady_state,
)
from qpsim.solvers.coupled_newton import (
    CoupledNewtonLineSearchError,
    coupled_newton_solve,
)
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
    @pytest.mark.parametrize(
        "control",
        ["T_bath", "tau_l", "tol", "step_rtol", "fd_step", "fd_floor"],
    )
    @pytest.mark.parametrize("bad_value", [True, 1.0 + 0.0j, "1e-6"])
    def test_rejects_non_real_numeric_controls(
        self,
        control: str,
        bad_value: object,
    ) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        controls: dict[str, object] = {
            "T_bath": T_bath,
            "tau_l": 0.25,
            "tol": 1e-10,
            "step_rtol": 1e-8,
            "fd_step": 1e-8,
            "fd_floor": 1e-12,
        }
        controls[control] = bad_value

        with pytest.raises(ValueError, match=control):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                **controls,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("bad_value", [True, 2.0 + 0.0j, "2"])
    def test_rejects_non_integer_max_iter_controls(
        self,
        bad_value: object,
    ) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)

        with pytest.raises(ValueError, match="max_iter"):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                max_iter=bad_value,  # type: ignore[arg-type]
            )

    def test_finite_temperature_underflowed_vacuum_fails_closed(self) -> None:
        """Zero float64 turnover is not proof of a finite-T vacuum root."""
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, _ = _thermal_setup(
            T_bath=1e-6,
            num=4,
        )
        zeros_f = np.zeros(ctx.E.size)
        zeros_ph = np.zeros_like(omega)
        loss_only = ExternalFlux(
            gain=zeros_f,
            loss_rate=np.ones_like(zeros_f),
        )

        with pytest.raises(CoupledNewtonLineSearchError):
            coupled_newton_solve(
                ctx,
                zeros_f,
                zeros_ph,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=1e-6,
                tau_l=1e308,
                external_flux=loss_only,
                tol=1e-10,
                step_rtol=1e-8,
                analytic_cross=True,
                max_iter=2,
            )

    def test_zero_temperature_loss_only_vacuum_is_absorbing(self) -> None:
        """The finite-T guard must preserve the exact T=0 vacuum root."""
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, _ = _thermal_setup(
            T_bath=0.0,
            num=4,
        )
        zeros_f = np.zeros(ctx.E.size)
        zeros_ph = np.zeros_like(omega)
        loss_only = ExternalFlux(
            gain=zeros_f,
            loss_rate=np.ones_like(zeros_f),
        )

        f_out, n_out = coupled_newton_solve(
            ctx,
            zeros_f,
            zeros_ph,
            omega_bins=omega,
            omega_idx_diff=idx_d,
            omega_idx_sum=idx_s,
            diff_sign=sgn,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.0,
            tau_l=1e308,
            external_flux=loss_only,
            tol=1e-10,
            step_rtol=1e-8,
            analytic_cross=True,
            max_iter=2,
        )
        np.testing.assert_array_equal(f_out, zeros_f)
        np.testing.assert_array_equal(n_out, zeros_ph)

    def test_cold_half_amplitude_seed_cannot_false_converge(self) -> None:
        """The coupled certificate must see the cold QP-number mode.

        At 55 mK, number-conserving scattering dominates both aggregate
        residual denominators.  Before the channel-specific certificates,
        the finite-difference-cross path returned the seed essentially
        unchanged at ``0.5*f_FD``.  Its pair-number error was only 1.18e-4
        (so a standalone 1% gate also passed), while the full phonon residual
        was 6.6e-2 relative to the pair/escape turnover.  The FD Jacobian
        cannot resolve that cold mode, so the honest result is a loud stall,
        not a silently wrong solution.
        """
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            T_bath=0.055,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        with pytest.raises(CoupledNewtonLineSearchError):
            coupled_newton_solve(
                ctx,
                0.5 * f_FD,
                n_BE,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                tol=1e-10,
                step_rtol=1e-8,
                analytic_cross=False,
            )

    def test_exact_cold_thermal_seed_remains_certifiable(self) -> None:
        """Known detailed balance is valid below the pair/escape roundoff floor."""
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            T_bath=0.05,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        f_out, n_out = coupled_newton_solve(
            ctx,
            f_FD,
            n_BE,
            omega_bins=omega,
            omega_idx_diff=idx_d,
            omega_idx_sum=idx_s,
            diff_sign=sgn,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=T_bath,
            tau_l=0.25,
            tol=1e-10,
            step_rtol=1e-8,
            analytic_cross=False,
        )
        np.testing.assert_array_equal(f_out, f_FD)
        np.testing.assert_array_equal(n_out, n_BE)

    def test_actual_7mk_thermal_shortcut_uses_unfloored_fermi_state(self) -> None:
        gap = 180.0
        T_bath = 0.007
        E, _ = build_energy_grid(
            gap=gap,
            energy_min_factor=1.0,
            energy_max_factor=5.0,
            num_energy_bins=28,
        )
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=gap,
        )
        K_s0 = build_scattering_kernel_base(ctx, tau_0=438.0, T_c=1.18)
        K_r0 = build_recombination_kernel_base(ctx, tau_0=438.0, T_c=1.18)
        omega, idx_d, idx_s, sign = build_phonon_frequency_map(E)
        f_thermal = fermi_dirac_occupation(E, T_bath)
        n_thermal = thermal_phonon_occupation(omega, T_bath)

        f_out, n_out = coupled_newton_solve(
            ctx,
            f_thermal,
            n_thermal,
            omega_bins=omega,
            omega_idx_diff=idx_d,
            omega_idx_sum=idx_s,
            diff_sign=sign,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=T_bath,
            tau_l=0.25,
        )

        np.testing.assert_array_equal(f_out, f_thermal)
        np.testing.assert_array_equal(n_out, n_thermal)
        assert float(np.max(f_out)) < np.exp(-300.0)

    def test_thermal_shortcut_rejects_non_detailed_balance_kernel(self) -> None:
        """Analytic Fermi/Bose arrays do not certify an arbitrary custom kernel."""
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            T_bath=0.05,
        )
        asymmetric = K_s0.copy()
        asymmetric[0, 1] *= 10.0
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        with pytest.raises(RuntimeError):
            coupled_newton_solve(
                ctx,
                f_FD,
                n_BE,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=asymmetric,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                step_rtol=1e-8,
                max_iter=1,
            )

    def test_thermal_shortcut_rejects_noncanonical_frequency_map(self) -> None:
        """Thermal-looking arrays do not certify a malformed phonon map."""
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            T_bath=0.05,
        )
        bad_idx_s = np.zeros_like(idx_s)
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)

        # The mapping is shape-, dtype-, and range-valid, but routing every
        # pair to omega=0 destroys detailed balance. Before the canonical-map
        # guard the cold shortcut returned these inputs unchanged even though
        # their QP-number backward error was exactly one.
        with pytest.raises(RuntimeError):
            coupled_newton_solve(
                ctx,
                f_FD,
                n_BE,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=bad_idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                step_rtol=1e-8,
                max_iter=1,
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_frequency_or_kernel(self, bad_value: float) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_BE = thermal_phonon_occupation(omega, T_bath)
        bad_omega = omega.copy()
        bad_omega[0] = bad_value
        with pytest.raises(ValueError, match="omega_bins"):
            coupled_newton_solve(
                ctx,
                f_FD,
                n_BE,
                omega_bins=bad_omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
            )

        bad_kernel = K_s0.copy()
        bad_kernel[0, 0] = bad_value
        with pytest.raises(ValueError, match="K_s0"):
            coupled_newton_solve(
                ctx,
                f_FD,
                n_BE,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=bad_kernel,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
            )

    @pytest.mark.parametrize("target", ["f_init", "n_ph_init", "omega_bins"])
    @pytest.mark.parametrize("imaginary", [1e-20, np.nan])
    def test_rejects_complex_state_and_frequency_before_float_cast(
        self,
        target: str,
        imaginary: float,
    ) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        values = {
            "f_init": f_init.astype(complex),
            "n_ph_init": n_init.astype(complex),
            "omega_bins": omega.astype(complex),
        }
        values[target][0] = complex(float(values[target][0].real), imaginary)

        with pytest.raises(ValueError, match=target):
            coupled_newton_solve(
                ctx,
                values["f_init"] if target == "f_init" else f_init,
                values["n_ph_init"] if target == "n_ph_init" else n_init,
                omega_bins=(
                    values["omega_bins"] if target == "omega_bins" else omega
                ),
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
            )

    @pytest.mark.parametrize(
        "kernel_name",
        ["K_s0", "K_r0", "K_s0_phonon_side", "K_r0_phonon_side"],
    )
    @pytest.mark.parametrize("imaginary", [1e-20, np.nan])
    def test_rejects_complex_kernel_before_rate_assembly(
        self,
        kernel_name: str,
        imaginary: float,
    ) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup()
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        bad_kernel = K_s0.astype(complex)
        bad_kernel[0, 0] = complex(float(bad_kernel[0, 0].real), imaginary)
        kernel_kwargs: dict[str, np.ndarray] = {
            "K_s0": K_s0,
            "K_r0": K_r0,
        }
        kernel_kwargs[kernel_name] = bad_kernel

        with pytest.raises(ValueError, match=kernel_name):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                T_bath=T_bath,
                tau_l=0.25,
                **kernel_kwargs,
            )

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
        f_abs, _ = coupled_newton_solve(
            ctx, f_seed, n_BE, tol=1e-1, step_rtol=0.0, **common,
        )
        # Same loose tol as a safety floor, but step_rtol drives refinement.
        f_rel, _ = coupled_newton_solve(
            ctx, f_seed, n_BE, tol=1e-1, step_rtol=1e-8, **common
        )

        scale = float(np.max(np.abs(f_FD)))
        rel_err_abs = float(np.max(np.abs(f_abs - f_FD))) / scale
        rel_err_rel = float(np.max(np.abs(f_rel - f_FD))) / scale
        assert rel_err_abs > 1e-2   # tol-only stayed ~5% off (frozen on seed)
        assert rel_err_rel < 1e-4   # step_rtol refined to the fixed point

    def test_step_rtol_requires_a_residual_certificate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.solvers import coupled_newton as coupled_newton_module

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            num=3,
        )
        f_init = np.full(ctx.E.size, 0.5)
        n_init = thermal_phonon_occupation(omega, T_bath)

        # Pin R_f=1 while presenting Newton with an artificial 1e12
        # f-Jacobian. The resulting relative step is ~1e-12, but the balance
        # residual remains exactly one: step size alone cannot certify a root.
        monkeypatch.setattr(
            coupled_newton_module,
            "_gain_loss_sum",
            lambda f, *args, **kwargs: (np.ones_like(f), np.zeros_like(f)),
        )
        monkeypatch.setattr(
            coupled_newton_module,
            "_jacobian_analytical",
            lambda f, *args, **kwargs: np.eye(f.size) * 1e12,
        )

        def zero_phonon_balance(
            f: np.ndarray,
            ctx: SpectralContext,
            K_s0: np.ndarray | None,
            K_r0: np.ndarray | None,
            omega_idx_diff: np.ndarray,
            omega_idx_sum: np.ndarray,
            diff_sign: np.ndarray,
            n_omega: int,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(n_omega), np.zeros(n_omega)

        monkeypatch.setattr(
            coupled_newton_module,
            "compute_phonon_source_sink",
            zero_phonon_balance,
        )

        with pytest.raises(RuntimeError, match="line search failed"):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                tol=1e-12,
                step_rtol=1e-6,
                max_iter=1,
            )

    def test_step_rtol_cannot_fall_back_to_tiny_absolute_residual(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A line-search stall must not bypass the balance certificate."""
        from qpsim.solvers import coupled_newton as coupled_newton_module

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            num=3,
        )
        f_init = np.full(ctx.E.size, 0.5)
        n_init = thermal_phonon_occupation(omega, T_bath)

        # The dimensional residual is far below tol but its backward balance
        # error is exactly one.  A zero Jacobian step makes every line-search
        # trial identical, exercising the post-line-search fallback that used
        # to return this false root whenever norm < tol.
        monkeypatch.setattr(
            coupled_newton_module,
            "_gain_loss_sum",
            lambda f, *args, **kwargs: (
                np.full_like(f, 1.0e-20), np.zeros_like(f),
            ),
        )
        monkeypatch.setattr(
            coupled_newton_module,
            "_jacobian_analytical",
            lambda f, *args, **kwargs: np.eye(f.size) * 1.0e20,
        )

        def zero_phonon_balance(
            f: np.ndarray,
            ctx: SpectralContext,
            K_s0: np.ndarray | None,
            K_r0: np.ndarray | None,
            omega_idx_diff: np.ndarray,
            omega_idx_sum: np.ndarray,
            diff_sign: np.ndarray,
            n_omega: int,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros(n_omega), np.zeros(n_omega)

        monkeypatch.setattr(
            coupled_newton_module,
            "compute_phonon_source_sink",
            zero_phonon_balance,
        )

        with pytest.raises(RuntimeError, match="line search failed"):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                tol=1e-10,
                step_rtol=1e-6,
                max_iter=1,
            )

    def test_non_finite_residual_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from qpsim.solvers import coupled_newton as coupled_newton_module

        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(
            num=3,
        )
        f_init = np.full(ctx.E.size, 0.5)
        n_init = thermal_phonon_occupation(omega, T_bath)
        monkeypatch.setattr(
            coupled_newton_module,
            "_gain_loss_sum",
            lambda f, *args, **kwargs: (
                np.full_like(f, np.nan),
                np.zeros_like(f),
            ),
        )

        with pytest.raises(RuntimeError, match="residual became non-finite"):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.25,
                tol=1e-12,
                step_rtol=1e-6,
                max_iter=1,
            )

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
        with pytest.raises(ValueError, match="n_ph_init must have shape"):
            coupled_newton_solve(
                ctx, f_init, np.zeros(omega.size + 5),  # wrong length
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.1,
            )

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, -0.1, 1.1])
    def test_rejects_nonphysical_f_init(self, bad_value: float) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=10)
        f_init = np.full(ctx.E.size, 0.1)
        f_init[0] = bad_value
        n_init = thermal_phonon_occupation(omega, T_bath)

        with pytest.raises(ValueError, match="f_init"):
            coupled_newton_solve(
                ctx, f_init, n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.1,
            )

    def test_fd_cross_jacobian_respects_occupation_upper_bound(self) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=3)
        f_init = np.ones(ctx.E.size)
        n_init = thermal_phonon_occupation(omega, T_bath)

        try:
            coupled_newton_solve(
                ctx, f_init, n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                tau_l=0.1,
                max_iter=1,
                analytic_cross=False,
            )
        except RuntimeError as error:
            # One iteration need not converge from the saturated state, but
            # assembling its finite-difference cross block must not probe
            # f > 1 and fail the physical-state guard.
            assert "non-physical trial state" not in str(error)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf, -0.1])
    def test_rejects_nonphysical_n_ph_init(self, bad_value: float) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=10)
        f_init = np.full(ctx.E.size, 0.1)
        n_init = thermal_phonon_occupation(omega, T_bath)
        n_init[0] = bad_value

        with pytest.raises(ValueError, match="n_ph_init"):
            coupled_newton_solve(
                ctx, f_init, n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.1,
            )

    @pytest.mark.parametrize("bad_temperature", [-0.1, np.nan, np.inf])
    def test_rejects_invalid_bath_temperature(self, bad_temperature: float) -> None:
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=10)
        f_init = np.full(ctx.E.size, 0.1)
        n_init = thermal_phonon_occupation(omega, T_bath)

        with pytest.raises(ValueError, match="T_bath"):
            coupled_newton_solve(
                ctx,
                f_init,
                n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d,
                omega_idx_sum=idx_s,
                diff_sign=sgn,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=bad_temperature,
                tau_l=0.1,
            )

    def test_zero_tau_l_branch(self) -> None:
        # τ_l = 0 leaves total energy unconstrained. Starting exactly at a
        # thermal root used to exit before assembling the singular Jacobian,
        # making this a vacuous "solver" test for an underdetermined problem.
        ctx, K_s0, K_r0, omega, idx_d, idx_s, sgn, T_bath = _thermal_setup(num=12)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        n_init = thermal_phonon_occupation(omega, T_bath)
        with pytest.raises(ValueError, match="finite positive tau_l"):
            coupled_newton_solve(
                ctx, f_init, n_init,
                omega_bins=omega,
                omega_idx_diff=idx_d, omega_idx_sum=idx_s, diff_sign=sgn,
                K_s0=K_s0, K_r0=K_r0,
                T_bath=T_bath, tau_l=0.0,
                tol=1e-8,
            )


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
