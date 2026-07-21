"""Tests for qpsim.solvers.newton_steady_state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import (
    _gain_loss_backward_error,
    _jacobian_analytical,
    _residual,
    newton_solve_f,
)


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
    @pytest.mark.parametrize("T_bath", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_bath_temperature(self, T_bath: float) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="T_bath"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                T_bath=T_bath,
            )

    @pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_invalid_tolerance(self, tol: float) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="tol must be finite and positive"):
            newton_solve_f(ctx, np.full(ctx.E.size, 0.1), tol=tol)

    @pytest.mark.parametrize(
        "tol",
        [0.0, -1.0, float("nan"), float("inf")],
    )
    def test_rejects_invalid_backward_error_tolerance(self, tol: float) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="backward_error_tol"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                backward_error_tol=tol,
            )

    @pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
    def test_rejects_invalid_max_iter(self, max_iter) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="positive integer"):
            newton_solve_f(ctx, np.full(ctx.E.size, 0.1), max_iter=max_iter)

    @pytest.mark.parametrize(
        "active",
        [
            np.ones(30, dtype=int),
            np.ones((1, 30), dtype=bool),
            np.ones(29, dtype=bool),
        ],
    )
    def test_rejects_non_bool_or_misshaped_active_mask(
        self,
        active: np.ndarray,
    ) -> None:
        ctx, _, _, _ = _setup(num=30)
        with pytest.raises(ValueError, match="one-dimensional bool mask"):
            newton_solve_f(ctx, np.full(ctx.E.size, 0.1), active=active)

    def test_cut_cell_analytical_jacobian_matches_finite_difference(self) -> None:
        E = np.arange(0.9, 3.0, 0.4)
        dE = np.full(E.size, 0.4)
        ctx = SpectralContext(E, dE, gap=1.0)
        f = np.array([0.08, 0.12, 0.03, 0.20, 0.01, 0.15])
        K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=1.2)
        K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=1.2)
        N_p = _thermal_phonon_scattering_occupation(E, 0.3)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(E, 0.3)
        photon = {"omega_0": 0.8, "n_bar": 2.3, "c_phot": 0.7}
        analytical = _jacobian_analytical(
            f,
            ctx,
            K_s0,
            K_r0,
            photon,
            None,
            N_p,
            N_emit,
            N_abs,
        )

        h = 1e-7
        finite_difference = np.empty_like(analytical)
        for j in range(f.size):
            up = f.copy()
            down = f.copy()
            up[j] += h
            down[j] -= h
            r_up = _residual(
                up, ctx, K_s0, K_r0, 0.3, photon, None,
                N_p, N_emit, N_abs,
            )
            r_down = _residual(
                down, ctx, K_s0, K_r0, 0.3, photon, None,
                N_p, N_emit, N_abs,
            )
            finite_difference[:, j] = (r_up - r_down) / (2.0 * h)

        np.testing.assert_allclose(
            analytical, finite_difference, rtol=2e-6, atol=2e-9,
        )

    def test_near_gap_supported_bin_cannot_hide_nonzero_residual(self) -> None:
        E = np.array([1.01, 1.5])
        dE = np.array([0.49, 0.49])
        ctx = SpectralContext(
            E, dE, gap=1.0, active_margin_factor=0.1,
        )
        f0 = np.array([0.1, 0.1])
        flux = ExternalFlux(
            gain=np.array([0.25, 0.1]),
            loss_rate=np.ones(2),
        )

        solved = newton_solve_f(
            ctx,
            f0,
            external_flux=flux,
            tol=1e-12,
            max_iter=10,
        )

        np.testing.assert_allclose(solved, [0.25, 0.1], atol=1e-13)

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

    def test_tiny_absolute_false_root_is_rejected(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A tiny dimensional rate cannot bypass an O(1) balance error."""
        import qpsim.solvers.newton_steady_state as newton_module

        ctx, _, _, _ = _setup(num=8)
        active = np.ones(ctx.E.size, dtype=bool)

        def constant_unbalanced_rates(f, *args, **kwargs):
            return np.full_like(f, 1e-30), np.zeros_like(f)

        monkeypatch.setattr(
            newton_module,
            "_gain_loss_sum",
            constant_unbalanced_rates,
        )
        monkeypatch.setattr(
            newton_module,
            "_jacobian_analytical",
            lambda f, *args, **kwargs: np.eye(f.size),
        )

        with pytest.raises(RuntimeError, match="backward error"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.5),
                active=active,
                tol=1e-14,
                backward_error_tol=1e-6,
                max_iter=2,
            )


class TestGainLossBackwardError:
    def test_is_scale_invariant(self) -> None:
        gain = np.array([1.0, 4.0])
        loss = np.array([0.0, 1.0])
        f = np.array([0.0, 2.0])
        active = np.ones(2, dtype=bool)
        reference = _gain_loss_backward_error(gain, loss, f, active)
        scaled = _gain_loss_backward_error(
            1e-30 * gain,
            1e-30 * loss,
            f,
            active,
        )

        assert reference == pytest.approx(3.0 / 7.0)
        assert scaled == pytest.approx(reference)


class TestPairNumberCertificate:
    """2026-07-21 round-5 review: the aggregate Newton metrics are
    amplitude-blind at cold temperatures — c*f_FD returned unchanged at
    50-80 mK even with tol=1e-30, because number-conserving scattering
    dominates the certification turnover while the number-changing pair
    channel scales as e^{-2*Delta/kT}. The standalone solver now
    certifies the conserved-QP-number mode against the pair turnover."""

    def _thermal_setup(self, T_bath: float):
        from qpsim.materials.database import load_material

        material = load_material("Al")
        gap = 1.764 * KB_UEV_PER_K * material.T_c
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.01, energy_max_factor=6.0,
            num_energy_bins=30,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        K_s0 = build_scattering_kernel_base(
            ctx, tau_0=material.tau_0, T_c=material.T_c
        )
        K_r0 = build_recombination_kernel_base(
            ctx, tau_0=material.tau_0, T_c=material.T_c
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        return ctx, K_s0, K_r0, f_FD

    @pytest.mark.parametrize("T_bath,c", [(0.05, 0.5), (0.05, 2.0), (0.08, 1.05)])
    def test_wrong_number_cold_seed_fails_loud(
        self, T_bath: float, c: float
    ) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(T_bath)
        with pytest.raises(RuntimeError, match="conserved-QP-number"):
            newton_solve_f(
                ctx, np.clip(c * f_FD, 0.0, 1.0),
                K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            )

    @pytest.mark.parametrize("T_bath", [0.05, 0.1])
    def test_thermal_seed_still_certifies(self, T_bath: float) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(T_bath)
        out = newton_solve_f(ctx, f_FD, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath)
        head = f_FD > 1e-16
        np.testing.assert_allclose(out[head], f_FD[head], rtol=1e-6)
