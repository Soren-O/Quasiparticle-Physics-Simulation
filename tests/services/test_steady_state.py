"""Tests for qpsim.services.steady_state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import (
    _fermi_dirac,
    _phonon_balance_backward_error,
    _picard_convergence_ratio,
    _resolve_picard_balance_tol,
    solve_steady_state,
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


class TestThermalPhononPath:
    @pytest.mark.parametrize("T_bath", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_bath_temperature(self, T_bath: float) -> None:
        ctx, K_s0, K_r0, _ = _setup()
        with pytest.raises(ValueError, match="T_bath"):
            solve_steady_state(ctx, K_s0, K_r0, T_bath)

    @pytest.mark.parametrize(
        "phonon_escape_time",
        [-1.0, float("nan"), float("inf")],
    )
    def test_rejects_invalid_phonon_escape_time(
        self,
        phonon_escape_time: float,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        with pytest.raises(ValueError, match="phonon_escape_time"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                phonon_escape_time=phonon_escape_time,
            )

    @pytest.mark.parametrize(
        ("control", "value", "message"),
        [
            ("max_picard_iter", 0, "max_picard_iter"),
            ("max_picard_iter", 1.5, "max_picard_iter"),
            ("max_picard_iter", True, "max_picard_iter"),
            ("picard_tol", 0.0, "picard_tol"),
            ("picard_tol", float("nan"), "picard_tol"),
            ("picard_atol", -1.0, "picard_atol"),
            ("picard_atol", float("inf"), "picard_atol"),
            ("picard_balance_tol", 0.0, "picard_balance_tol"),
            ("picard_mixing", 0.0, "picard_mixing"),
            ("picard_mixing", 1.01, "picard_mixing"),
            ("picard_mixing", float("nan"), "picard_mixing"),
            ("anderson_depth", -1, "anderson_depth"),
            ("anderson_depth", 1.5, "anderson_depth"),
            ("anderson_depth", True, "anderson_depth"),
        ],
    )
    def test_rejects_invalid_picard_controls(
        self,
        control: str,
        value: object,
        message: str,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        with pytest.raises(ValueError, match=message):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                **{control: value},
            )

    def test_rejects_dynes_collision_context(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        dynes = SpectralContext(
            E_bins=ctx.E,
            dE_bins=ctx.dE,
            gap=ctx.gap,
            dynes_gamma=0.1,
        )

        with pytest.raises(ValueError, match="Dynes-broadened collision solves"):
            solve_steady_state(dynes, K_s0, K_r0, T_bath)

    def test_recovers_fermi_dirac(self) -> None:
        # At thermal phonons + bath temperature T_bath, the steady state
        # should be Fermi-Dirac at T_bath.
        ctx, K_s0, K_r0, T_bath = _setup()
        f = solve_steady_state(ctx, K_s0, K_r0, T_bath, tol=1e-12)
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        np.testing.assert_allclose(f, f_FD, atol=1e-8)

    def test_cold_fermi_seed_preserves_representable_tail(self) -> None:
        E = np.array([360.0])
        T_bath = 0.007
        exponent = E / (KB_UEV_PER_K * T_bath)
        exp_negative = np.exp(-exponent)
        expected = exp_negative / (1.0 + exp_negative)

        got = _fermi_dirac(E, T_bath)

        np.testing.assert_allclose(got, expected, rtol=2e-14)
        assert 0.0 < got[0] < np.exp(-500.0)

    def test_rejects_bad_initial_guess_shape(self) -> None:
        import pytest

        ctx, K_s0, K_r0, T_bath = _setup()
        with pytest.raises(ValueError, match="initial_guess shape"):
            solve_steady_state(
                ctx, K_s0, K_r0, T_bath, initial_guess=np.zeros(3),
            )

    def test_rejects_matrix_initial_guess_even_when_size_matches(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        with pytest.raises(ValueError, match="one-dimensional"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_guess=np.zeros((1, ctx.E.size)),
            )

    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_initial_guesses_before_float_cast(
        self,
        imaginary: float,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        bad_f = np.zeros(ctx.E.size, dtype=complex)
        bad_f[0] = complex(0.0, imaginary)
        with pytest.raises(ValueError, match="initial_guess must be real-valued"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_guess=bad_f,
            )

        from qpsim.collisions.phonon import build_phonon_frequency_map

        omega = build_phonon_frequency_map(ctx.E)[0]
        bad_phonon = np.zeros(omega.size, dtype=complex)
        bad_phonon[0] = complex(0.0, imaginary)
        with pytest.raises(
            ValueError,
            match="initial_phonon_guess must be real-valued",
        ):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_phonon_guess=bad_phonon,
                phonon_escape_time=0.25,
            )

    @pytest.mark.parametrize(
        ("control", "value", "message"),
        [
            ("tol", 0.0, "tol"),
            ("newton_backward_error_tol", 0.0, "newton_backward_error_tol"),
            ("max_iter", 0, "max_iter"),
        ],
    )
    def test_no_active_bins_still_validate_newton_controls(
        self,
        control: str,
        value: object,
        message: str,
    ) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        ctx._active_mask = np.zeros(ctx.E.size, dtype=bool)  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match=message):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                **{control: value},
            )

    def test_no_active_bins_still_validate_phonon_seed_shape(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup()
        ctx._active_mask = np.zeros(ctx.E.size, dtype=bool)  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="initial_phonon_guess"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_phonon_guess=np.zeros(1),
                phonon_escape_time=0.1,
            )


class TestFiniteTauLPath:
    def test_picard_consumes_explicit_initial_phonon_guess(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.services.steady_state as steady_state_module

        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        from qpsim.collisions.phonon import build_phonon_frequency_map

        omega = build_phonon_frequency_map(ctx.E)[0]
        seed = thermal_phonon_occupation(omega, T_bath) + 0.125
        observed: list[np.ndarray] = []
        original_project = steady_state_module.phonon_occupation_matrices_from_state

        def capture_project(n_ph, *args, **kwargs):
            observed.append(np.asarray(n_ph, dtype=float).copy())
            return original_project(n_ph, *args, **kwargs)

        def fixed_phonon_map(*args, coefficients_out, **kwargs):
            coefficients_out["a_ph"] = np.zeros(omega.size)
            coefficients_out["b_ph"] = np.zeros(omega.size)
            return seed.copy()

        monkeypatch.setattr(
            steady_state_module,
            "phonon_occupation_matrices_from_state",
            capture_project,
        )
        monkeypatch.setattr(
            steady_state_module,
            "newton_solve_f",
            lambda _ctx, f, **kwargs: np.asarray(f, dtype=float).copy(),
        )
        monkeypatch.setattr(
            steady_state_module,
            "phonon_steady_state",
            fixed_phonon_map,
        )
        monkeypatch.setattr(
            steady_state_module,
            "_phonon_balance_backward_error",
            lambda *args, **kwargs: 0.0,
        )

        solve_steady_state(
            ctx,
            K_s0,
            K_r0,
            T_bath,
            initial_guess=np.full(ctx.E.size, 0.01),
            initial_phonon_guess=seed,
            phonon_escape_time=0.1,
            max_picard_iter=1,
        )

        assert len(observed) == 1
        np.testing.assert_array_equal(observed[0], seed)

    @pytest.mark.parametrize("initial_level", [0.0, 0.5])
    def test_anderson_branch_collapse_falls_back_before_phonon_map_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        initial_level: float,
    ) -> None:
        """A collapsed AA inner root is rejected immediately and only once.

        The third inner solve deliberately returns the same drained state
        after Anderson has been disabled.  It must be allowed to converge:
        the guard prevents an accelerated branch hop, not a genuine plain-
        Picard drain.
        """
        import qpsim.services.steady_state as steady_state_module
        from qpsim.collisions.phonon import build_phonon_frequency_map

        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        omega = build_phonon_frequency_map(ctx.E)[0]
        n_seed = thermal_phonon_occupation(omega, T_bath) + 0.125
        # The zero case proves the collapse reference follows the physical
        # branch after it grows away from a vacuum/tiny seed.
        initial_f = np.full(ctx.E.size, initial_level)
        physical_f = np.full(ctx.E.size, 0.4)
        collapsed_f = np.full(ctx.E.size, 0.01)

        f_inputs: list[np.ndarray] = []
        n_inputs: list[np.ndarray] = []
        newton_outputs = iter((physical_f, collapsed_f, collapsed_f))
        phonon_map_calls = 0
        anderson_calls = 0
        ratio_values = iter((2.0, 0.0))

        original_project = steady_state_module.phonon_occupation_matrices_from_state

        def capture_project(n_ph, *args, **kwargs):
            n_inputs.append(np.asarray(n_ph, dtype=float).copy())
            return original_project(n_ph, *args, **kwargs)

        def fake_newton(_ctx, f, **kwargs):
            f_inputs.append(np.asarray(f, dtype=float).copy())
            return next(newton_outputs).copy()

        def fixed_phonon_map(*args, coefficients_out, **kwargs):
            nonlocal phonon_map_calls
            phonon_map_calls += 1
            coefficients_out["a_ph"] = np.zeros(omega.size)
            coefficients_out["b_ph"] = np.zeros(omega.size)
            return n_seed.copy()

        def fake_anderson(*args, **kwargs):
            nonlocal anderson_calls
            anderson_calls += 1
            return n_seed + 0.25

        monkeypatch.setattr(
            steady_state_module,
            "phonon_occupation_matrices_from_state",
            capture_project,
        )
        monkeypatch.setattr(steady_state_module, "newton_solve_f", fake_newton)
        monkeypatch.setattr(
            steady_state_module,
            "phonon_steady_state",
            fixed_phonon_map,
        )
        monkeypatch.setattr(
            steady_state_module,
            "_picard_convergence_ratio",
            lambda *args, **kwargs: next(ratio_values),
        )
        monkeypatch.setattr(
            steady_state_module,
            "_phonon_balance_backward_error",
            lambda *args, **kwargs: 0.0,
        )
        monkeypatch.setattr(
            steady_state_module,
            "anderson_extrapolate",
            fake_anderson,
        )

        phonon_out: dict[str, np.ndarray] = {}
        result = solve_steady_state(
            ctx,
            K_s0,
            K_r0,
            T_bath,
            initial_guess=initial_f,
            initial_phonon_guess=n_seed,
            phonon_escape_time=0.1,
            max_picard_iter=3,
            anderson_depth=2,
            phonon_out=phonon_out,
        )

        # Iteration 2's collapsed Newton output never reaches the phonon map.
        # Iteration 3 starts from the last matched physical pair, with AA now
        # disabled, and may return the genuine drained state without looping.
        assert phonon_map_calls == 2
        assert anderson_calls == 1
        assert len(f_inputs) == 3
        np.testing.assert_array_equal(f_inputs[2], physical_f)
        assert len(n_inputs) == 3
        np.testing.assert_array_equal(n_inputs[0], n_seed)
        np.testing.assert_array_equal(n_inputs[1], n_seed + 0.25)
        np.testing.assert_array_equal(n_inputs[2], n_seed)
        np.testing.assert_array_equal(result, collapsed_f)
        np.testing.assert_array_equal(phonon_out["n_ph"], n_seed)

    def test_first_inner_correction_does_not_masquerade_as_aa_collapse(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hot user seed may move by >10x before Anderson is ever causal."""
        import qpsim.services.steady_state as steady_state_module

        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        flux = ExternalFlux(
            gain=np.full(ctx.E.size, 1.0e-4),
            loss_rate=np.zeros(ctx.E.size),
        )
        original_anderson = steady_state_module.anderson_extrapolate
        calls = 0

        def count_anderson(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original_anderson(*args, **kwargs)

        monkeypatch.setattr(
            steady_state_module,
            "anderson_extrapolate",
            count_anderson,
        )
        result = solve_steady_state(
            ctx,
            K_s0,
            K_r0,
            T_bath,
            initial_guess=np.full(ctx.E.size, 0.5),
            phonon_escape_time=1.0e-3,
            anderson_depth=2,
            picard_tol=1.0e-8,
            max_picard_iter=100,
            tol=1.0e-10,
            external_flux=flux,
        )

        assert float(np.max(result)) < 0.05
        assert calls > 0

    @pytest.mark.parametrize(
        "bad",
        [np.zeros((1, 2)), np.array([-1.0]), np.array([np.nan])],
    )
    def test_rejects_invalid_initial_phonon_guess(self, bad: np.ndarray) -> None:
        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        with pytest.raises(ValueError, match="initial_phonon_guess"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_phonon_guess=bad,
                phonon_escape_time=0.1,
            )

    def test_rejects_phonon_seed_on_thermal_shortcut(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup(num=12)
        with pytest.raises(ValueError, match="initial_phonon_guess"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                initial_phonon_guess=np.zeros(1),
            )

    def test_tau_l_zero_matches_thermal_at_low_rates(self) -> None:
        # With phonon_escape_time=0 (no substrate coupling) and weak
        # coupling, the Picard solve should still approximately recover
        # f_FD — the only way the QP distribution stays at equilibrium
        # is if the e-ph integral vanishes, which requires thermal f.
        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        phonon_out: dict = {}
        f = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            phonon_escape_time=0.0,
            tol=1e-10, max_iter=200,
            picard_tol=1e-8, max_picard_iter=100,
            phonon_out=phonon_out,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        # Detailed balance is exact only for thermal phonons; with
        # self-consistent n_ph there can be small drift, so we allow
        # a looser bound than the pure-thermal test above.
        np.testing.assert_allclose(f, f_FD, atol=1e-4)
        assert "n_ph" in phonon_out
        assert "omega_bins" in phonon_out

    def test_finite_tau_l_converges(self) -> None:
        # Just exercise the finite-tau_l path end-to-end. With a small
        # tau_l the phonon bath dominates and f should be near f_FD.
        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        f = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            phonon_escape_time=1e-3,  # 1 ps — very fast bath coupling
            tol=1e-10, max_iter=200,
            picard_tol=1e-8, max_picard_iter=200,
        )
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        np.testing.assert_allclose(f, f_FD, atol=1e-4)

    def test_picard_atol_is_independent_of_newton_tol(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.services.steady_state as steady_state_module

        ctx, K_s0, K_r0, T_bath = _setup(num=20)
        observed: list[tuple[float, float]] = []

        def capture_tolerances(
            n_ph: np.ndarray,
            n_ph_new: np.ndarray,
            *,
            rtol: float,
            atol: float,
        ) -> float:
            observed.append((rtol, atol))
            return 0.0

        monkeypatch.setattr(
            steady_state_module,
            "_picard_convergence_ratio",
            capture_tolerances,
        )
        solve_steady_state(
            ctx,
            K_s0,
            K_r0,
            T_bath,
            phonon_escape_time=1e-3,
            tol=1e-10,
            picard_tol=2e-8,
            picard_atol=3e-13,
        )

        assert observed == [(2e-8, 3e-13)]

    def test_candidate_requires_physical_phonon_balance(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.services.steady_state as steady_state_module

        ctx, K_s0, K_r0, T_bath = _setup(num=12)

        # Isolate the new gate: manufacture an occupation-space candidate that
        # the fixed-point metric accepts, while its affine physical balance is
        # O(1) wrong in backward-error units.
        monkeypatch.setattr(
            steady_state_module,
            "_picard_convergence_ratio",
            lambda *args, **kwargs: 0.0,
        )
        monkeypatch.setattr(
            steady_state_module,
            "newton_solve_f",
            lambda _ctx, f, **kwargs: np.asarray(f, dtype=float).copy(),
        )

        def unbalanced_fixed_point(*args, coefficients_out, **kwargs):
            n_omega = args[4].size
            coefficients_out["a_ph"] = np.full(n_omega, 1e-12)
            coefficients_out["b_ph"] = np.zeros(n_omega)
            return thermal_phonon_occupation(args[4], T_bath) + 1e-13

        monkeypatch.setattr(
            steady_state_module,
            "phonon_steady_state",
            unbalanced_fixed_point,
        )

        with pytest.raises(RuntimeError, match="phonon balance backward error"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                phonon_escape_time=0.1,
                max_picard_iter=1,
                picard_tol=1e-7,
                picard_atol=1e-11,
                picard_balance_tol=1e-6,
            )


class TestPicardConvergenceMetric:
    """Unit tests for the finite-τ_l Picard convergence metric.

    Regression coverage for the near-zero-bin stall fixed in f041a85
    (Fischer Fig. 7 at P_read=-64 dBm, T_B=0.10 K): a fully-settled solve whose
    only "unconverged" bin was a sub-gap occupation oscillating at the inner
    Newton's ~1e-11 float-noise floor.
    """

    def test_near_zero_bin_noise_does_not_pin_metric(self) -> None:
        # One dominant bin settled to its true relative tolerance, plus a
        # near-zero sub-gap bin carrying only ~1e-11 numerical noise on a
        # ~1e-6 occupation (|Δ|/n ~ 1e-5). The explicit occupation-space
        # allowance must prevent that jitter from pinning Picard.
        picard_tol = 1e-7  # Fischer Fig. 7's picard_tol
        picard_atol = 1e-11
        n_ph = np.array([8.0, 1e-6])
        n_ph_new = np.array([8.0 * (1.0 + 1e-9), 1e-6 + 1e-11])

        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=picard_tol, atol=picard_atol,
        ) <= 1.0

    def test_all_small_state_cannot_converge_only_under_atol(self) -> None:
        # Every component-wise update lies below atol, so the former local-only
        # test accepted this state even though the update is O(1) relative to
        # the state as a whole.  The normwise guard is amplitude independent.
        n_ph = np.full(8, 1e-12)
        n_ph_new = np.full(8, 9e-12)
        rtol = 1e-7
        atol = 1e-11

        local_allowed = atol + rtol * np.maximum(n_ph, n_ph_new)
        assert np.all(np.abs(n_ph_new - n_ph) <= local_allowed)
        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=rtol, atol=atol,
        ) > 1.0

    def test_small_physical_bin_is_not_hidden_by_unrelated_peak(self) -> None:
        # Regression for Fischer Fig. 5: a large low-energy occupation must not
        # set a global floor that hides evolution in a tiny but dynamically
        # decisive above-gap pair-breaking bin.
        rtol = 1e-7
        atol = 1e-11
        n_ph = np.array([8.0, 1e-12])
        n_ph_new = np.array([8.0 * (1.0 + 1e-9), 5e-10])

        ratio = _picard_convergence_ratio(n_ph, n_ph_new, rtol=rtol, atol=atol)
        assert ratio > 1.0

        # The removed peak-scaled floor (1e-3 * peak) falsely accepted this.
        peak_floor_metric = float(
            np.max(np.abs(n_ph_new - n_ph) / (np.maximum(n_ph, n_ph_new) + 8e-3))
        )
        assert peak_floor_metric < rtol

    def test_meaningful_peak_change_is_not_converged(self) -> None:
        n_ph = np.array([8.0, 4.0])
        n_ph_new = np.array([8.0 * (1.0 + 1e-3), 4.0])
        assert _picard_convergence_ratio(
            n_ph, n_ph_new, rtol=1e-4, atol=1e-11,
        ) > 1.0

    def test_all_zero_is_trivially_converged(self) -> None:
        z = np.zeros(5)
        assert _picard_convergence_ratio(z, z, rtol=1e-7, atol=1e-11) == 0.0

    def test_identical_iterates_are_converged(self) -> None:
        n_ph = np.array([8.0, 1e-6, 0.0])
        assert _picard_convergence_ratio(
            n_ph, n_ph.copy(), rtol=1e-7, atol=1e-11,
        ) == 0.0

    @pytest.mark.parametrize(
        "rtol,atol",
        [
            (0.0, 1e-11),
            (-1.0, 1e-11),
            (1e-7, -1.0),
            (float("nan"), 1e-11),
            (float("inf"), 1e-11),
            (1e-7, float("nan")),
            (1e-7, float("inf")),
        ],
    )
    def test_rejects_invalid_tolerances(self, rtol: float, atol: float) -> None:
        with pytest.raises(ValueError):
            _picard_convergence_ratio(
                np.ones(2), np.ones(2), rtol=rtol, atol=atol,
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_rejects_non_finite_iterates(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            _picard_convergence_ratio(
                np.array([1.0]), np.array([bad]), rtol=1e-7, atol=1e-11,
            )


class TestPhononBalanceBackwardError:
    def test_is_scale_invariant(self) -> None:
        n_ph = np.array([0.0, 2.0])
        n_th = np.zeros(2)
        a_ph = np.array([1.0, 4.0])
        b_ph = np.array([0.0, -1.0])
        reference = _phonon_balance_backward_error(
            n_ph, a_ph, b_ph, n_th, tau_l=0.0,
        )
        scaled = _phonon_balance_backward_error(
            n_ph, 1e-30 * a_ph, 1e-30 * b_ph, n_th, tau_l=0.0,
        )
        assert reference == pytest.approx(3.0 / 7.0)
        assert scaled == pytest.approx(reference)


class TestPicardBalanceTolerance:
    def test_default_has_roundoff_headroom(self) -> None:
        assert _resolve_picard_balance_tol(1e-10, None) == pytest.approx(1e-6)
        assert _resolve_picard_balance_tol(2e-4, None) == pytest.approx(2e-3)

    def test_explicit_limit_is_preserved(self) -> None:
        assert _resolve_picard_balance_tol(1e-7, 3e-5) == pytest.approx(3e-5)
