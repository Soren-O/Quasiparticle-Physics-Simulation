"""Tests for qpsim.solvers.newton_steady_state."""

from __future__ import annotations

import numpy as np
import pytest
import qpsim.solvers.newton_steady_state as newton_mod
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import (
    _gain_loss_backward_error,
    _jacobian_analytical,
    _residual,
    _weighted_number_backward_error,
    newton_solve_f,
    number_changing_gain_loss,
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
    @pytest.mark.parametrize(
        "control",
        [
            "T_bath",
            "tol",
            "backward_error_tol",
            "number_polish_shape_tol",
        ],
    )
    @pytest.mark.parametrize("bad_value", [True, 1.0 + 0.0j, "1e-6"])
    def test_rejects_non_real_numeric_controls(
        self,
        control: str,
        bad_value: object,
    ) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match=control):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                **{control: bad_value},
            )

    @pytest.mark.parametrize("bad_value", [True, 2.0 + 0.0j, "2"])
    def test_rejects_non_integer_max_iter_controls(
        self,
        bad_value: object,
    ) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="max_iter"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                max_iter=bad_value,  # type: ignore[arg-type]
            )

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

    @pytest.mark.parametrize(
        "tol",
        [0.0, -1.0, float("nan"), float("inf")],
    )
    def test_rejects_invalid_number_polish_shape_tolerance(
        self,
        tol: float,
    ) -> None:
        ctx, _, _, _ = _setup()
        with pytest.raises(ValueError, match="number_polish_shape_tol"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                number_polish_shape_tol=tol,
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

    def test_active_mask_cannot_enable_unsupported_storage_rows(self) -> None:
        # A caller-supplied superset used to disable the total-number
        # certificate while letting a zero-capacity row participate in the
        # Newton system. At cold T that could accept a wrong-amplitude seed.
        E = np.array([0.8, 1.0, 1.2, 1.4])
        dE = np.full(E.size, 0.2)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=1.0)
        assert not ctx.active_mask[0]

        with pytest.raises(ValueError, match="represented spectral capacity"):
            newton_solve_f(
                ctx,
                np.full(E.size, 0.1),
                active=np.ones(E.size, dtype=bool),
            )

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

    def test_actual_7mk_pair_fixed_point_is_not_exp_minus_250_floor(self) -> None:
        # With the historical Bose exponent cap, every seed converged to the
        # artificial sqrt(exp(-500)) occupation instead of this cold thermal
        # state. The exact seed must now remain the certified fixed point.
        gap = 180.0
        T_bath = 0.007
        E = np.linspace(gap, 3.0 * gap, 80)
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=gap,
        )
        K_r0 = build_recombination_kernel_base(
            ctx,
            tau_0=438.0,
            T_c=1.18,
        )
        exponent = E / (KB_UEV_PER_K * T_bath)
        exp_negative = np.exp(-exponent)
        f_thermal = exp_negative / (1.0 + exp_negative)

        solved = newton_solve_f(
            ctx,
            f_thermal,
            K_r0=K_r0,
            T_bath=T_bath,
        )

        np.testing.assert_array_equal(solved, f_thermal)
        assert float(np.max(solved)) < np.exp(-275.0)

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

    def test_partial_active_mask_fails_loudly(self) -> None:
        """A proper subset omits unmodelled transition-boundary flux."""
        E = np.array([1.1, 1.5])
        dE = np.array([0.4, 0.4])
        ctx = SpectralContext(E, dE, gap=1.0)
        K_s0 = np.array([[0.0, 1.0], [1.0, 0.0]])
        K_r0 = np.full((2, 2), 0.2)
        N_p = np.ones((2, 2))
        N_emit = np.ones((2, 2))
        N_abs = np.zeros((2, 2))
        active = np.array([True, False])

        with pytest.raises(ValueError, match="proper subsets omit"):
            newton_solve_f(
                ctx,
                np.array([0.3, 0.5]),
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=0.0,
                active=active,
                N_p_override=N_p,
                N_emit_override=N_emit,
                N_abs_override=N_abs,
                tol=1e-14,
                max_iter=50,
            )

    def test_cold_wrong_amplitude_partial_mask_cannot_self_certify(self) -> None:
        """Regression: excluding one bin used to return 0.5*f_FD unchanged."""
        ctx, K_s0, K_r0, T_bath = _setup(T_bath=0.05)
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        active = ctx.active_mask.copy()
        active[np.flatnonzero(active)[-1]] = False

        with pytest.raises(ValueError, match="cannot be number-certified"):
            newton_solve_f(
                ctx,
                0.5 * f_FD,
                K_s0=K_s0,
                K_r0=K_r0,
                T_bath=T_bath,
                active=active,
                tol=1e-30,
            )

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

    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_initial_occupation_before_float_cast(
        self,
        imaginary: float,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        f0 = np.full(ctx.E.size, 0.1, dtype=complex)
        f0[0] = complex(0.1, imaginary)

        with pytest.raises(ValueError, match="initial occupation f must be real-valued"):
            newton_solve_f(
                ctx, f0, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath,
            )

    @pytest.mark.parametrize(
        "name",
        ["K_s0", "K_r0", "N_p_override", "N_emit_override", "N_abs_override"],
    )
    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_source_arrays_before_float_cast(
        self,
        name: str,
        imaginary: float,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        N_p = _thermal_phonon_scattering_occupation(ctx.E, T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(ctx.E, T_bath)
        inputs = {
            "K_s0": K_s0,
            "K_r0": K_r0,
            "N_p_override": N_p,
            "N_emit_override": N_emit,
            "N_abs_override": N_abs,
        }
        bad = np.asarray(inputs[name], dtype=complex).copy()
        bad.flat[0] = complex(float(np.real(bad.flat[0])), imaginary)
        inputs[name] = bad

        with pytest.raises(ValueError, match=rf"{name} must be real-valued"):
            newton_solve_f(
                ctx,
                np.full(ctx.E.size, 0.1),
                T_bath=T_bath,
                **inputs,
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

    def test_tiny_absolute_residual_can_progress_on_balance_merit(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A cold nonlinear root may require more than 20 absolute halvings.

        This two-row model has a tiny generation term and quadratic loss. At
        vacuum the raw Newton step overshoots the root by about six decades;
        none of the legacy line search's 20 trial steps lowers the already-
        tiny dimensional residual. The normalized balance merit nevertheless
        improves, so Newton must be allowed to leave vacuum and converge while
        retaining both unchanged return gates.
        """
        E = np.array([1.1, 1.5])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        row_scales = np.array([1e-140, 1.0])
        generation = 1e-22
        linear_loss = 1e-17

        def nonlinear_gain_loss(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            del args, kwargs
            return (
                row_scales * generation,
                row_scales * (linear_loss + f),
            )

        def nonlinear_jacobian(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            del args, kwargs
            return np.diag(row_scales * (-linear_loss - 2.0 * f))

        monkeypatch.setattr(newton_mod, "_gain_loss_sum", nonlinear_gain_loss)
        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            nonlinear_jacobian,
        )

        out = newton_solve_f(
            ctx,
            np.zeros(2),
            tol=1e-13,
            backward_error_tol=1e-8,
            max_iter=50,
        )
        expected = 0.5 * (
            np.sqrt(linear_loss**2 + 4.0 * generation) - linear_loss
        )
        np.testing.assert_allclose(out, expected, rtol=1e-8, atol=0.0)

    def test_subtol_residual_drop_cannot_worsen_balance_to_vacuum(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Below the absolute gate, normalized balance controls line search.

        For ``R = g - sqrt(f)`` the first Newton step from ``9*g**2``
        overshoots and clips to vacuum.  Vacuum has a smaller dimensional
        residual (``g`` versus ``2*g``) but a worse gain/loss backward error
        (one versus one half).  Accepting that raw residual decrease strands
        the solve at a singular boundary; balance-aware backtracking reaches
        the positive root instead.
        """
        E = np.array([1.1, 1.5])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        generation = 1e-20
        root = generation**2

        def square_root_gain_loss(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            del args, kwargs
            loss = np.zeros_like(f)
            positive = f > 0.0
            loss[positive] = 1.0 / np.sqrt(f[positive])
            return np.full_like(f, generation), loss

        def square_root_jacobian(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            del args, kwargs
            diagonal = np.zeros_like(f)
            positive = f > 0.0
            diagonal[positive] = -0.5 / np.sqrt(f[positive])
            return np.diag(diagonal)

        monkeypatch.setattr(
            newton_mod,
            "_gain_loss_sum",
            square_root_gain_loss,
        )
        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            square_root_jacobian,
        )

        out = newton_solve_f(
            ctx,
            np.full(2, 9.0 * root),
            tol=1e-14,
            backward_error_tol=1e-10,
            max_iter=20,
        )

        np.testing.assert_allclose(out, root, rtol=1e-9, atol=0.0)

    def test_line_search_targets_the_only_failed_number_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A passing aggregate merit must not starve number-mode progress.

        The deliberately overlong Newton direction reflects the state around
        the physical number root: a full step moves 0.4 -> 0.6 and improves
        the already-passing aggregate metric, but leaves the independent
        number error unchanged.  The half step reaches 0.5 and clears the
        number gate while keeping the aggregate gate satisfied.  The former
        line search accepted the irrelevant full step and exhausted its
        iteration budget; the active-certificate merit must choose the half
        step without relaxing any return tolerance.
        """
        E = np.array([1.1, 1.5])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        rate = 1e-20

        def gain_loss(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            del args, kwargs
            return np.full_like(f, 1.5 * rate), np.full_like(f, 3.0 * rate)

        def deliberately_overlong_jacobian(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            del args, kwargs
            # The physical residual derivative is -3*rate. Halving it makes
            # the full Newton step reflect around f=0.5.
            return np.eye(f.size) * (-1.5 * rate)

        def number_channels(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray, bool]:
            del args, kwargs
            return np.ones_like(f), np.ones_like(f), True

        def number_error(
            gain: np.ndarray,
            loss_rate: np.ndarray,
            f: np.ndarray,
            weights: np.ndarray,
            active: np.ndarray,
        ) -> float:
            del gain, loss_rate, weights, active
            mean = float(np.mean(f))
            if np.isclose(mean, 0.5, rtol=0.0, atol=1e-15):
                return 0.0
            if np.isclose(mean, 0.6, rtol=0.0, atol=1e-15):
                # The full step makes strict but irrelevant progress. A
                # first-improvement line search would accept this and miss the
                # half step that actually clears the gate.
                return 0.099
            if not np.isclose(mean, 0.4, rtol=0.0, atol=1e-15):
                raise AssertionError(
                    "line search continued after the number gate cleared"
                )
            return 0.1

        def aggregate_error(
            gain: np.ndarray,
            loss_rate: np.ndarray,
            f: np.ndarray,
            active: np.ndarray,
            **kwargs: object,
        ) -> float:
            del gain, loss_rate, active, kwargs
            mean = float(np.mean(f))
            if np.isclose(mean, 0.6, rtol=0.0, atol=1e-15):
                return 1e-12
            if np.isclose(mean, 0.5, rtol=0.0, atol=1e-15):
                return 1.5e-12
            return 2e-12

        monkeypatch.setattr(newton_mod, "_gain_loss_sum", gain_loss)
        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            deliberately_overlong_jacobian,
        )
        monkeypatch.setattr(
            newton_mod,
            "number_changing_gain_loss",
            number_channels,
        )
        monkeypatch.setattr(
            newton_mod,
            "_weighted_number_backward_error",
            number_error,
        )
        monkeypatch.setattr(
            newton_mod,
            "_gain_loss_backward_error",
            aggregate_error,
        )
        monkeypatch.setattr(
            newton_mod,
            "_number_conserving_gain_loss",
            lambda *args, **kwargs: (
                np.zeros(E.size),
                np.zeros(E.size),
            ),
        )
        monkeypatch.setattr(
            newton_mod,
            "_polish_number_mode_amplitude",
            lambda *args, **kwargs: None,
        )

        out = newton_solve_f(
            ctx,
            np.full(E.size, 0.4),
            K_r0=np.eye(E.size),
            T_bath=0.0,
            tol=1e-14,
            backward_error_tol=1e-10,
            max_iter=2,
        )

        np.testing.assert_allclose(out, 0.5, rtol=0.0, atol=1e-15)


class TestRowScaledNewtonSystem:
    def test_preserves_each_nonzero_row_without_a_global_floor(self) -> None:
        scales = np.array([1e-170, 1e-80, 1.0])
        normalized_jacobian = np.array(
            [
                [-3.0, 1.0, 0.5],
                [0.25, -2.0, 0.75],
                [1.0, 0.5, -4.0],
            ]
        )
        normalized_residual = np.array([0.5, -0.25, 2.0])
        jacobian = scales[:, None] * normalized_jacobian
        residual = scales * normalized_residual

        scaled_jacobian, scaled_residual = (
            newton_mod._row_scaled_newton_system(
                jacobian,
                residual,
                scales,
                np.zeros(3),
            )
        )

        np.testing.assert_allclose(
            scaled_jacobian,
            normalized_jacobian,
            rtol=2e-16,
            atol=0.0,
        )
        np.testing.assert_allclose(
            scaled_residual,
            normalized_residual,
            rtol=0.0,
            atol=0.0,
        )

    def test_exact_zero_turnover_row_is_left_unscaled(self) -> None:
        jacobian = np.array([[-2.0, 1.0], [1e-120, -3e-120]])
        residual = np.array([0.0, 2e-120])

        scaled_jacobian, scaled_residual = (
            newton_mod._row_scaled_newton_system(
                jacobian,
                residual,
                np.array([0.0, 2e-120]),
                np.zeros(2),
            )
        )

        np.testing.assert_array_equal(scaled_jacobian[0], jacobian[0])
        assert scaled_residual[0] == residual[0]
        np.testing.assert_allclose(scaled_jacobian[1], [0.5, -1.5])
        assert scaled_residual[1] == pytest.approx(1.0)


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

    def test_direct_conserving_balance_distinguishes_shape_quality(self) -> None:
        """The scalar number polish is gated by independently assembled shape.

        A flat ultracold seed has tiny dimensional rates but a bad spectral
        shape, whereas scaling a thermal shape changes only its conserved
        number mode.  The direct conserving-channel aggregate must separate
        those cases without recovering it as ``full - pair``.
        """
        cold_ctx, cold_Ks, _cold_Kr, _cold_fd = self._thermal_setup(0.02)
        cold_np = _thermal_phonon_scattering_occupation(cold_ctx.E, 0.02)
        cold_gain, cold_loss = newton_mod._number_conserving_gain_loss(
            np.full_like(cold_ctx.E, 0.01),
            cold_ctx,
            cold_Ks,
            0.02,
            N_p=cold_np,
        )
        cold_error = _gain_loss_backward_error(
            cold_gain,
            cold_loss,
            np.full_like(cold_ctx.E, 0.01),
            cold_ctx.active_mask,
        )
        assert cold_error > 1e-2

        ctx, K_s0, _K_r0, f_FD = self._thermal_setup(0.06)
        N_p = _thermal_phonon_scattering_occupation(ctx.E, 0.06)
        thermal_gain, thermal_loss = newton_mod._number_conserving_gain_loss(
            0.5 * f_FD,
            ctx,
            K_s0,
            0.06,
            N_p=N_p,
        )
        thermal_error = _gain_loss_backward_error(
            thermal_gain,
            thermal_loss,
            0.5 * f_FD,
            ctx.active_mask,
        )
        assert thermal_error < 1e-13

    def test_projected_nonabsorbing_vacuum_is_backtracked(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A roundoff-overshooting Newton direction may not erase a source.

        On the production 1,620-bin Fig. 3 ratio-zero solve, multithreaded
        LAPACK returned a valid but slightly different direction that
        overshot the O(1e-10) occupation by a few ulps.  Projection mapped the
        full trial to ``f=0``; its tiny dimensional residual looked better,
        even though finite-temperature pair absorption makes vacuum a
        non-root.  Force that direction deterministically and assert the line
        search backtracks before assembling another Jacobian at vacuum.
        """
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.1)
        photon_params = {
            "omega_0": float(ctx.dE[0]),
            "n_bar": 1e7,
            "c_phot": 1e-9,
        }
        original_solve = newton_mod.np.linalg.solve
        solve_calls = 0

        def first_direction_overshoots(
            matrix: np.ndarray,
            rhs: np.ndarray,
        ) -> np.ndarray:
            nonlocal solve_calls
            solve_calls += 1
            if solve_calls == 1:
                return -1.0001 * f_FD
            return original_solve(matrix, rhs)

        original_jacobian = newton_mod._jacobian_analytical

        def reject_vacuum_jacobian(
            f: np.ndarray,
            *args: object,
            **kwargs: object,
        ) -> np.ndarray:
            assert np.any(f[ctx.active_mask] != 0.0), (
                "the non-absorbing vacuum was accepted as an iterate"
            )
            return original_jacobian(f, *args, **kwargs)

        monkeypatch.setattr(newton_mod.np.linalg, "solve", first_direction_overshoots)
        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            reject_vacuum_jacobian,
        )

        out = newton_solve_f(
            ctx,
            f_FD,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.1,
            photon_params=photon_params,
            tol=1e-12,
            backward_error_tol=1e-10,
            number_polish_shape_tol=1e-8,
            max_iter=500,
        )

        assert solve_calls >= 2
        assert np.all(out[ctx.active_mask] > 0.0)
        gain, loss, _ = number_changing_gain_loss(out, ctx, K_r0, 0.1)
        number_error = _weighted_number_backward_error(
            gain,
            loss,
            out,
            ctx.cell_weights,
            ctx.active_mask,
        )
        assert number_error is not None
        assert number_error <= 1e-10

    def test_roundoff_quiet_shape_routes_to_polish_before_dense_newton(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A nearly fixed shape should take the cheap scalar amplitude path.

        Fig. 3's ratio-zero shape error was about ``2.4e-10`` while only its
        amplitude/number mode remained unresolved.  The former hard ``1e-13``
        routing gate forced many 1620x1620 dense solves before reaching the
        same scalar polish as a line-search fallback.  A routing-only gate
        admits roundoff-quiet shapes, while the polished state still has to
        pass every unchanged return certificate.
        """
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.06)
        original_backward_error = newton_mod._gain_loss_backward_error

        def controlled_backward_error(
            gain: np.ndarray,
            loss_rate: np.ndarray,
            f: np.ndarray,
            active: np.ndarray,
            **kwargs: object,
        ) -> float:
            if "scale_gain" in kwargs:
                return 0.0
            # Direct number-conserving shape diagnostic.
            return 5e-9

        monkeypatch.setattr(
            newton_mod,
            "_gain_loss_backward_error",
            controlled_backward_error,
        )

        def dense_newton_must_not_run(*args: object, **kwargs: object):
            raise AssertionError("scalar-polishable state reached dense Newton")

        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            dense_newton_must_not_run,
        )
        out = newton_solve_f(
            ctx,
            0.5 * f_FD,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.06,
            tol=1e-14,
            backward_error_tol=1e-10,
            number_polish_shape_tol=1e-8,
            max_iter=2,
        )

        np.testing.assert_allclose(out, f_FD, rtol=1e-8, atol=0.0)
        # Keep a real call alive in this test so a signature drift in the
        # monkeypatched helper cannot make the routing assertion vacuous.
        assert original_backward_error(
            np.ones(1),
            np.ones(1),
            np.ones(1),
            np.ones(1, dtype=bool),
        ) == 0.0

    @pytest.mark.parametrize(
        "T_bath,c",
        [(0.05, 0.5), (0.05, 2.0), (0.05, 1.01), (0.08, 1.05)],
    )
    def test_representable_wrong_number_cold_seed_is_corrected(
        self, T_bath: float, c: float
    ) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(T_bath)
        out = newton_solve_f(
            ctx,
            np.clip(c * f_FD, 0.0, 1.0),
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=T_bath,
        )
        np.testing.assert_allclose(out, f_FD, rtol=1e-14, atol=0.0)
        gain, loss, _ = number_changing_gain_loss(
            out, ctx, K_r0, T_bath
        )
        error = _weighted_number_backward_error(
            gain, loss, out, ctx.cell_weights, ctx.active_mask
        )
        assert error is not None
        assert error <= 1e-6

    def test_number_polish_precedes_aggregate_balance_gate(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A quiet cold number mode must not wait on aggregate convergence.

        The Fig. 7 recovery at 60 mK reached ``max|R| = 1.92e-32`` but
        stalled with aggregate backward error ``2.78e-3``.  The amplitude
        polish used to sit behind that failed aggregate gate, so it was
        unreachable.  Make the ordinary Jacobian deliberately singular:
        this solve can succeed only if the quiet absolute residual triggers
        number-mode polishing before the aggregate gate.
        """
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.06)
        seed = 0.5 * f_FD
        original_backward_error = newton_mod._gain_loss_backward_error

        def aggregate_error(
            gain: np.ndarray,
            loss_rate: np.ndarray,
            f: np.ndarray,
            active: np.ndarray,
            **kwargs: object,
        ) -> float:
            if "scale_gain" in kwargs:
                if np.allclose(f, f_FD, rtol=1e-10, atol=0.0):
                    return 0.0
                return 2.78e-3
            return original_backward_error(gain, loss_rate, f, active)

        monkeypatch.setattr(
            newton_mod,
            "_gain_loss_backward_error",
            aggregate_error,
        )
        monkeypatch.setattr(
            newton_mod,
            "_jacobian_analytical",
            lambda f, *args, **kwargs: np.zeros((f.size, f.size)),
        )

        out = newton_solve_f(
            ctx,
            seed,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.06,
            tol=1e-14,
            backward_error_tol=1e-8,
            max_iter=2,
        )

        np.testing.assert_allclose(out, f_FD, rtol=1e-10, atol=0.0)

    @pytest.mark.parametrize("T_bath", [0.02, 0.03, 0.04, 0.045, 0.05, 0.1])
    def test_thermal_seed_still_certifies(self, T_bath: float) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(T_bath)
        out = newton_solve_f(ctx, f_FD, K_s0=K_s0, K_r0=K_r0, T_bath=T_bath)
        # Full-array equality matters below 45 mK, where a >1e-16 head
        # selection is empty even though the represented tail is nonzero.
        np.testing.assert_array_equal(out, f_FD)

    def test_cold_root_many_decades_below_seed_is_polished(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A tiny positive root must not exhaust scalar-root iterations.

        At 20 mK the equilibrium amplitude is roughly 46 decades below
        this finite seed.  Solving directly on the linear scale with a
        subnormal ``xtol`` exhausted Brent's 100-iteration budget; the
        log-amplitude bracket must reach and certify the thermal root.
        """
        T_bath = 0.02
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(T_bath)
        original_brentq = newton_mod.brentq

        def brentq_must_stop_on_certificate(*args: object, **kwargs: object) -> float:
            original_brentq(*args, **kwargs)
            raise AssertionError(
                "amplitude polish over-solved the log root after its number "
                "certificate had reached the binary64 compatibility floor"
            )

        monkeypatch.setattr(newton_mod, "brentq", brentq_must_stop_on_certificate)

        out = newton_solve_f(
            ctx,
            np.full_like(f_FD, 0.01),
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=T_bath,
        )

        # ``_thermal_setup`` caps its reference exponent at 500, while the
        # production thermal occupations continue below that artificial
        # floor.  Compare the representable, uncapped head and certify the
        # number mode directly.
        head = ctx.E / (KB_UEV_PER_K * T_bath) < 500.0
        np.testing.assert_allclose(out[head], f_FD[head], rtol=1e-13, atol=0.0)
        gain, loss, _ = number_changing_gain_loss(out, ctx, K_r0, T_bath)
        error = _weighted_number_backward_error(
            gain, loss, out, ctx.cell_weights, ctx.active_mask
        )
        assert error is not None
        assert error <= 1e-6

    def test_number_polish_upper_scale_stays_inside_occupation_bounds(
        self,
    ) -> None:
        """Log-scale roundoff at a saturating root must not produce f > 1."""
        E = np.array([1.1, 1.2])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        seed = np.full(E.size, 1e-7)
        flux = ExternalFlux(
            gain=np.full(E.size, 1e-20),
            loss_rate=np.full(E.size, 1e-20),
        )

        out = newton_solve_f(
            ctx,
            seed,
            external_flux=flux,
            tol=1e-14,
            backward_error_tol=1e-12,
            max_iter=5,
        )

        np.testing.assert_array_equal(out, np.ones_like(out))
        gain = flux.gain
        loss = flux.loss_rate
        assert _gain_loss_backward_error(
            gain,
            loss,
            out,
            ctx.active_mask,
        ) == 0.0

    def test_exact_zero_temperature_vacuum_certifies(self) -> None:
        """Configured pair loss at T=0 has the exact absorbing vacuum root."""
        ctx, K_s0, K_r0, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)

        out = newton_solve_f(
            ctx,
            vacuum,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.0,
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_nonzero_seed_reaches_zero_temperature_absorbing_root(self) -> None:
        """The amplitude polish must admit the exact scale-zero endpoint."""
        ctx, K_s0, K_r0, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)

        out = newton_solve_f(
            ctx,
            np.full_like(ctx.E, 0.01),
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.0,
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_all_zero_pair_kernel_is_disabled_not_underflowed(self) -> None:
        """A zero-valued K_r0 is mathematically identical to ``None``."""
        ctx, _, _, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)

        out = newton_solve_f(
            ctx,
            vacuum,
            K_r0=np.zeros((ctx.E.size, ctx.E.size)),
            T_bath=0.2,
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_zero_pair_kernel_does_not_poison_loss_only_vacuum(self) -> None:
        """An inert K_r0 must not masquerade as a finite-T absorbing bath."""
        ctx, _, _, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)
        loss_only = ExternalFlux(
            gain=np.zeros_like(vacuum),
            loss_rate=np.ones_like(vacuum),
        )

        out = newton_solve_f(
            ctx,
            vacuum,
            K_r0=np.zeros((ctx.E.size, ctx.E.size)),
            T_bath=0.2,
            external_flux=loss_only,
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_zero_occupancy_pb_bath_is_absorbing_at_finite_temperature(
        self,
    ) -> None:
        material = load_material("Al")
        gap = 1.764 * KB_UEV_PER_K * material.T_c
        E = gap * np.arange(1.05, 4.05, 0.1)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )
        vacuum = np.zeros_like(ctx.E)
        omega_pb = 2.2 * gap

        out = newton_solve_f(
            ctx,
            vacuum,
            K_r0=None,
            T_bath=0.2,
            pb_photon_params={
                "omega_PB": omega_pb,
                "n_bar_PB": 0.0,
                "c_phot_PB": 1e-8,
            },
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_nonzero_seed_reaches_zero_occupancy_pb_absorbing_root(
        self,
    ) -> None:
        material = load_material("Al")
        gap = 1.764 * KB_UEV_PER_K * material.T_c
        E = gap * np.arange(1.05, 4.05, 0.1)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )
        vacuum = np.zeros_like(ctx.E)

        out = newton_solve_f(
            ctx,
            np.full_like(ctx.E, 0.01),
            T_bath=0.2,
            pb_photon_params={
                "omega_PB": 2.2 * gap,
                "n_bar_PB": 0.0,
                "c_phot_PB": 1e-8,
            },
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_explicit_zero_pair_absorption_is_absorbing(self) -> None:
        ctx, _, K_r0, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)
        shape = (ctx.E.size, ctx.E.size)

        out = newton_solve_f(
            ctx,
            vacuum,
            K_r0=K_r0,
            T_bath=0.2,
            N_emit_override=np.ones(shape),
            N_abs_override=np.zeros(shape),
        )

        np.testing.assert_array_equal(out, vacuum)

    @pytest.mark.parametrize("occupation", [0.0, 1e-200])
    def test_underflowed_finite_temperature_pair_bath_fails_loud(
        self,
        occupation: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Implicit finite-T absorption can underflow; it is not loss-only."""
        ctx, _, K_r0, _ = self._thermal_setup(0.05)
        unresolvable = np.full_like(ctx.E, occupation)

        def underflowed_thermal_pair_bath(
            E: np.ndarray, T_bath: float
        ) -> tuple[np.ndarray, np.ndarray]:
            del T_bath
            shape = (E.size, E.size)
            return np.ones(shape), np.zeros(shape)

        monkeypatch.setattr(
            newton_mod,
            "_thermal_phonon_recombination_occupations",
            underflowed_thermal_pair_bath,
        )

        with pytest.raises(RuntimeError, match="zero representable float64 turnover"):
            newton_solve_f(
                ctx,
                unresolvable,
                K_r0=K_r0,
                T_bath=1e-6,
            )

    def test_loss_only_external_flux_is_absorbing(self) -> None:
        ctx, _, _, _ = self._thermal_setup(0.05)
        vacuum = np.zeros_like(ctx.E)
        loss_only = ExternalFlux(
            gain=np.zeros_like(vacuum),
            loss_rate=np.ones_like(vacuum),
        )

        out = newton_solve_f(
            ctx,
            vacuum,
            T_bath=0.2,
            external_flux=loss_only,
        )

        np.testing.assert_array_equal(out, vacuum)

    def test_zero_external_flux_cannot_bypass_number_certificate(self) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.05)
        without_flux = newton_solve_f(
            ctx,
            0.5 * f_FD,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.05,
        )
        with_zero_flux = newton_solve_f(
            ctx,
            0.5 * f_FD,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.05,
            external_flux=ExternalFlux.zero(f_FD.size),
        )
        np.testing.assert_array_equal(with_zero_flux, without_flux)
        np.testing.assert_allclose(with_zero_flux, f_FD, rtol=1e-14, atol=0.0)

    def test_zero_pb_coupling_is_equivalent_to_no_channel(self) -> None:
        E = np.array([1.05, 1.2, 1.5, 2.1, 2.8, 3.4])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        seed = np.linspace(0.01, 0.2, E.size)

        without_channel = newton_solve_f(ctx, seed)
        with_disabled_channel = newton_solve_f(
            ctx,
            seed,
            pb_photon_params={
                "omega_PB": 2.1,
                "n_bar_PB": 1.0,
                "c_phot_PB": 0.0,
            },
        )

        np.testing.assert_array_equal(with_disabled_channel, without_channel)

    @pytest.mark.parametrize(
        "channel",
        (
            {"omega_0": 1.7, "n_bar": 1.0, "c_phot": 0.0},
            {"omega_PB": 2.1, "n_bar_PB": 1.0, "c_phot_PB": 0.0},
        ),
        ids=("subgap", "pair-breaking"),
    )
    def test_disabled_photon_channel_bypasses_jacobian_grid_guard(
        self,
        channel: dict[str, float],
    ) -> None:
        """A zero-coupling dict is equivalent to ``None`` through Newton."""
        E = np.array([1.1, 1.3, 1.8, 2.6])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        seed = np.full(E.size, 0.2)
        flux = ExternalFlux(
            gain=np.full(E.size, 0.1),
            loss_rate=np.ones(E.size),
        )
        kwargs = (
            {"photon_params": channel}
            if "omega_0" in channel
            else {"pb_photon_params": channel}
        )

        expected = newton_solve_f(ctx, seed, external_flux=flux)
        actual = newton_solve_f(
            ctx,
            seed,
            external_flux=flux,
            **kwargs,
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(actual, 0.1, rtol=0.0, atol=1e-15)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        (
            (
                {
                    "photon_params": {
                        "omega_0": 1.0,
                        "n_bar": float("nan"),
                        "c_phot": 0.0,
                    }
                },
                "n_bar",
            ),
            (
                {
                    "pb_photon_params": {
                        "omega_PB": 2.0,
                        "n_bar_PB": -1.0,
                        "c_phot_PB": 0.0,
                    }
                },
                "n_bar_PB",
            ),
        ),
        ids=("subgap", "pair-breaking"),
    )
    def test_disabled_photon_dict_still_validates_scalars(
        self,
        kwargs: dict[str, dict[str, float]],
        match: str,
    ) -> None:
        ctx, _, _, _ = _setup(num=8)
        with pytest.raises(ValueError, match=match):
            newton_solve_f(ctx, np.full(ctx.E.size, 0.1), **kwargs)

    def test_prescribed_flux_uses_full_turnover_with_tiny_internal_channel(
        self,
    ) -> None:
        """A prescribed source is not conservative exchange normalization."""
        E = np.array([1.1, 1.5])
        ctx = SpectralContext(
            E,
            integration_widths_from_centers(E),
            gap=1.0,
        )
        seed = np.array([0.28420116, 0.64854721])
        flux = ExternalFlux(
            gain=np.array([0.17794493, 0.23885144]),
            loss_rate=np.array([1.14270267, 1.27376035]),
        )
        expected = newton_solve_f(ctx, seed, external_flux=flux, T_bath=0.2)
        actual = newton_solve_f(
            ctx,
            seed,
            K_r0=np.full((2, 2), 1e-25),
            external_flux=flux,
            T_bath=0.2,
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)

    def test_number_accuracy_uses_advertised_backward_error_tol(self) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.05)
        seed = 1.01 * f_FD
        # The ~0.995% pair-number error is corrected under the default
        # 1e-6 contract, while an explicit 2% request accepts it unchanged.
        # There is no hidden 1% gate.
        strict = newton_solve_f(
            ctx, seed, K_s0=K_s0, K_r0=K_r0, T_bath=0.05,
        )
        np.testing.assert_allclose(strict, f_FD, rtol=1e-14, atol=0.0)
        out = newton_solve_f(
            ctx,
            seed,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.05,
            backward_error_tol=0.02,
        )
        np.testing.assert_array_equal(out, seed)

    def test_subgap_scattering_cannot_dilute_pair_number_error(self) -> None:
        ctx, K_s0, K_r0, f_FD = self._thermal_setup(0.05)
        omega = float(ctx.E[1] - ctx.E[0])
        n_bar = float(1.0 / np.expm1(omega / (KB_UEV_PER_K * 0.05)))
        out = newton_solve_f(
            ctx,
            0.5 * f_FD,
            K_s0=K_s0,
            K_r0=K_r0,
            T_bath=0.05,
            photon_params={
                "omega_0": omega,
                "n_bar": n_bar,
                "c_phot": 1e-9,
            },
        )
        np.testing.assert_allclose(out, f_FD, rtol=1e-14, atol=0.0)

    def test_pb_pair_certificate_runs_without_phonon_pair_kernel(self) -> None:
        material = load_material("Al")
        gap = 1.764 * KB_UEV_PER_K * material.T_c
        E = gap * np.arange(1.05, 4.05, 0.1)
        ctx = SpectralContext(
            E_bins=E,
            dE_bins=integration_widths_from_centers(E),
            gap=gap,
        )
        K_s0 = build_scattering_kernel_base(
            ctx, tau_0=material.tau_0, T_c=material.T_c
        )
        T_bath = 0.05
        f_FD = 1.0 / (
            np.exp(np.minimum(E / (KB_UEV_PER_K * T_bath), 500.0)) + 1.0
        )
        omega_pb = 2.2 * gap
        n_bar_pb = float(
            1.0 / np.expm1(omega_pb / (KB_UEV_PER_K * T_bath))
        )

        pb_params = {
            "omega_PB": omega_pb,
            "n_bar_PB": n_bar_pb,
            "c_phot_PB": 1e-8,
        }
        out = newton_solve_f(
            ctx,
            0.5 * f_FD,
            K_s0=K_s0,
            K_r0=None,
            T_bath=T_bath,
            pb_photon_params=pb_params,
        )
        np.testing.assert_allclose(out, f_FD, rtol=1e-14, atol=0.0)
        gain, loss, configured = number_changing_gain_loss(
            out,
            ctx,
            None,
            T_bath,
            pb_photon_params=pb_params,
        )
        assert configured
        error = _weighted_number_backward_error(
            gain, loss, out, ctx.cell_weights, ctx.active_mask
        )
        assert error is not None
        assert error <= 1e-6
