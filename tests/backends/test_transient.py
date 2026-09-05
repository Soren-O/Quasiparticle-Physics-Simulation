"""Tests for fixed-gap and stage-constrained moving-gap transients."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from qpsim.backends.diffusion import DiffusionBackend, DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononState
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)
from qpsim.physics.gap_equation import calibrate_gap, solve_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def _state_on_physics_grid(
    T_bath: float = 0.3,
    num_energy: int = 20,
) -> DiffusionState:
    """Build a state whose PhononState sits on the physics ω grid."""
    material = load_material("Al")
    gap = material.Delta_0

    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=0.75,
        energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)

    # Physics ω grid derived from E (matches what apply_collisions
    # expects and what steady_state would produce).
    omega, _, _, _ = build_phonon_frequency_map(E)
    n_ph_thermal = thermal_phonon_occupation(omega, T_bath)

    phonon = PhononState(
        n_ph=n_ph_thermal.reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), 0.25),
        branches=[PhononBranchSpec(name="debye_average")],
    )

    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    return DiffusionState(
        f=f_init,
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _state_on_custom_grid(
    E: np.ndarray,
    *,
    gap: float = 1.0,
    f: np.ndarray | None = None,
    dynes_gamma: float = 0.0,
) -> DiffusionState:
    """Reuse the lightweight phonon fixture with a prescribed QP grid."""
    state = _state_on_physics_grid(T_bath=0.1, num_energy=E.size)
    dE = integration_widths_from_centers(E)
    state.gap = gap
    state.spectral = SpectralContext(
        E_bins=E,
        dE_bins=dE,
        gap=gap,
        dynes_gamma=dynes_gamma,
    )
    state.f = np.asarray(f, dtype=float) if f is not None else np.where(gap < E, 0.1, 0.0)
    return state


class TestPublicOccupationValidation:
    @pytest.mark.parametrize(
        "operation",
        ("apply_collisions", "apply_gap_update", "step"),
    )
    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_state_before_float_cast(
        self,
        operation: str,
        imaginary: float,
    ) -> None:
        state = _state_on_physics_grid(T_bath=0.1, num_energy=8)
        bad_f = state.f.astype(complex)
        bad_f[0] = complex(float(bad_f[0].real), imaginary)
        state.f = bad_f
        backend = DiffusionBackend()

        with pytest.raises(ValueError, match=r"state\.f must be real-valued"):
            getattr(backend, operation)(state, 1.0e-3)


class TestApplyCollisions:
    def test_diagnostics_count_accepted_etd_substeps_not_rejected_trials(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _state_on_physics_grid(T_bath=0.1, num_energy=8)
        state.f = np.zeros_like(state.f)
        supported = state.spectral.rho > 0.0
        rhs_calls = 0

        def adversarial_rates(
            f: np.ndarray,
            *_args: object,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            nonlocal rhs_calls
            rhs_calls += 1
            return (
                np.where(supported, 1.0, 0.0),
                np.where(supported, 1.0e6 * f, 0.0),
            )

        monkeypatch.setattr(
            backend_module,
            "phonon_collision_rates",
            adversarial_rates,
        )
        updated, residual, accepted = (
            DiffusionBackend().apply_collisions_with_diagnostics(
                state,
                0.01,
            )
        )

        assert residual is None
        assert accepted > 1
        assert rhs_calls > 2 * accepted
        np.testing.assert_allclose(updated.f[supported], 1.0e-3, rtol=2.0e-3)

    def test_rejects_dynes_collision_context(self) -> None:
        state = _state_on_custom_grid(
            np.linspace(0.8, 2.0, 21),
            dynes_gamma=0.01,
        )

        with pytest.raises(ValueError, match="Dynes-broadened transient collisions"):
            DiffusionBackend().apply_collisions(state, dt=0.01)

    @pytest.mark.parametrize("bad_shape", ["short", "spatial"])
    def test_rejects_invalid_f_shape_before_kernel_build(
        self,
        bad_shape: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _state_on_physics_grid()
        state.f = (  # type: ignore[misc]
            state.f[:-1] if bad_shape == "short" else state.f[:, None]
        )
        monkeypatch.setattr(
            "qpsim.backends.diffusion.build_scattering_kernel_base",
            lambda *_args, **_kwargs: pytest.fail("kernel build should not run"),
        )

        with pytest.raises(ValueError, match="one-dimensional shape"):
            DiffusionBackend().apply_collisions(state, dt=0.01)

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), -1e-12, 1.0 + 1e-12],
    )
    def test_rejects_invalid_f_value_before_kernel_build(
        self,
        bad_value: float,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = _state_on_physics_grid()
        state.f = state.f.copy()  # type: ignore[misc]
        state.f[0] = bad_value
        monkeypatch.setattr(
            "qpsim.backends.diffusion.build_scattering_kernel_base",
            lambda *_args, **_kwargs: pytest.fail("kernel build should not run"),
        )

        with pytest.raises(ValueError, match=r"finite occupations in \[0, 1\]"):
            DiffusionBackend().apply_collisions(state, dt=0.01)

    def test_thermal_is_near_fixed_point(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = DiffusionBackend()
        new_state = backend.apply_collisions(state, dt=0.01)
        np.testing.assert_allclose(new_state.f, state.f, atol=1e-8)

    def test_preserves_bounds(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = DiffusionBackend()
        # Perturb f to stress the stepper.
        state.f = np.clip(  # type: ignore[misc]
            state.f * (1.0 + 0.3 * np.cos(state.spectral.E / state.gap)),
            0.0,
            1.0,
        )
        new_state = backend.apply_collisions(state, dt=0.05)
        assert np.all(new_state.f >= 0.0)
        assert np.all(new_state.f <= 1.0)

    def test_phonon_unchanged(self) -> None:
        # apply_collisions freezes n_ph; the phonon advance is a separate
        # step and is not part of it.
        state = _state_on_physics_grid()
        backend = DiffusionBackend()
        new_state = backend.apply_collisions(state, dt=0.01)
        assert new_state.phonon is state.phonon

    def test_rejects_off_grid_phonon(self) -> None:
        # PhononState with a non-physics ω grid must be rejected.
        state = _state_on_physics_grid()
        off_grid_omega = np.linspace(0.1, 5.0, 40)  # wrong length + values
        state.phonon = PhononState(  # type: ignore[misc]
            n_ph=np.zeros((1, off_grid_omega.size, 1)),
            omega_bins=off_grid_omega.reshape(1, -1),
            tau_l=np.full((1, off_grid_omega.size), 0.25),
            branches=state.phonon.branches,
        )
        with pytest.raises(ValueError, match="physics grid"):
            DiffusionBackend().apply_collisions(state, dt=0.01)


class TestApplyTransport:
    def test_is_no_op_for_homogeneous(self) -> None:
        state = _state_on_physics_grid()
        backend = DiffusionBackend()
        new_state = backend.apply_transport(state, dt=0.1)
        assert new_state is state  # identical object — nothing moved

    def test_rejects_non_1d_f(self) -> None:
        state = _state_on_physics_grid()
        state.f = state.f[:, None]  # type: ignore[misc] — fake spatial axis
        with pytest.raises(ValueError, match="must be 1D over energy"):
            DiffusionBackend().apply_transport(state, dt=0.1)


class TestApplyGapUpdate:
    def test_positive_dt_magnitude_does_not_scale_projection(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        state.f = np.clip(
            state.f + 0.01 * np.exp(-(((state.spectral.E / state.gap - 1.4) / 0.2) ** 2)),
            0.0,
            1.0,
        )

        short_label = DiffusionBackend().apply_gap_update(state, dt=1e-6)
        long_label = DiffusionBackend().apply_gap_update(state, dt=1e6)

        assert short_label.gap == long_label.gap
        np.testing.assert_array_equal(short_label.f, long_label.f)

    def test_projection_is_idempotent(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3, num_energy=80)
        state.f = np.clip(
            state.f + 0.05 * np.exp(-(((state.spectral.E / state.gap - 1.4) / 0.2) ** 2)),
            0.0,
            1.0,
        )
        mass_before = float(
            np.sum(
                bcs_dos_cell_weights(
                    state.spectral.E,
                    state.spectral.dE,
                    state.gap,
                )
                * state.f
            )
        )

        projected = DiffusionBackend().apply_gap_update(state, dt=1.0)
        projected_again = DiffusionBackend().apply_gap_update(projected, dt=1.0)
        mass_after = float(
            np.sum(
                bcs_dos_cell_weights(
                    projected.spectral.E,
                    projected.spectral.dE,
                    projected.gap,
                )
                * projected.f
            )
        )

        assert abs(projected.gap - state.gap) > 1e-6
        assert mass_after == pytest.approx(mass_before, rel=5e-12, abs=1e-15)
        assert projected_again is projected

    def test_projection_certifies_final_gap_equation(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3, num_energy=80)
        state.f = np.clip(
            state.f + 0.05 * np.exp(-(((state.spectral.E / state.gap - 1.4) / 0.2) ** 2)),
            0.0,
            1.0,
        )

        projected = DiffusionBackend().apply_gap_update(state, dt=1.0)
        calibration = calibrate_gap(
            T_c=projected.material.T_c,
            T_bath=projected.T_bath,
            Delta_0=projected.material.Delta_0,
        )
        constrained_gap = solve_gap(
            calibration,
            projected.f,
            projected.spectral.E,
            dE_bins=projected.spectral.dE,
            xtol=1e-12,
        )

        assert abs(constrained_gap - projected.gap) <= 5e-10

    def test_projection_fails_loudly_at_iteration_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _state_on_custom_grid(np.linspace(0.75, 6.0, 241))
        constrained_gaps = iter((0.999, 0.998, 0.997))
        reference_gaps: list[float] = []

        def next_constrained_gap(*_args: object, **kwargs: object) -> float:
            reference_gaps.append(float(kwargs["reference_gap"]))
            return next(constrained_gaps)

        monkeypatch.setattr(backend_module, "_GAP_PROJECTION_MAX_ITER", 3)
        monkeypatch.setattr(
            backend_module,
            "solve_gap",
            next_constrained_gap,
        )

        with pytest.raises(RuntimeError, match="did not converge in 3 iterations"):
            DiffusionBackend().apply_gap_update(state, dt=1.0)
        np.testing.assert_allclose(reference_gaps, [1.0, 0.999, 0.998])

    def test_gap_calibration_uses_material_delta_0(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        state = _state_on_physics_grid(T_bath=0.01)
        original = backend_module.calibrate_gap
        observed: list[float | None] = []

        def capture_calibration(*args, **kwargs):
            observed.append(kwargs.get("Delta_0"))
            return original(*args, **kwargs)

        monkeypatch.setattr(backend_module, "calibrate_gap", capture_calibration)

        DiffusionBackend().apply_gap_update(state, dt=0.1)

        assert observed == [state.material.Delta_0]

    def test_near_thermal_minimal_change(self) -> None:
        # At near-thermal f with a gap set to the BCS Δ(0), the solve
        # should return a very similar gap (calibration at T_bath uses
        # Δ_eq(T_bath), which is close to Δ(0) for low T_bath / T_c).
        state = _state_on_physics_grid(T_bath=0.01)  # very low T
        gap_before = state.gap
        backend = DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.1)
        rel_change = abs(new_state.gap - gap_before) / gap_before
        assert rel_change < 0.01  # within 1%

    def test_zero_dt_is_no_op(self) -> None:
        state = _state_on_physics_grid()
        backend = DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.0)
        assert new_state is state

    @pytest.mark.parametrize("bad_dt", [float("nan"), float("inf")])
    def test_nonfinite_dt_is_rejected(self, bad_dt: float) -> None:
        state = _state_on_physics_grid()
        with pytest.raises(ValueError, match="dt must be finite"):
            DiffusionBackend().apply_gap_update(state, dt=bad_dt)

    def test_preserves_f_bounds(self) -> None:
        state = _state_on_physics_grid()
        backend = DiffusionBackend()
        new_state = backend.apply_gap_update(state, dt=0.01)
        assert np.all(new_state.f >= 0.0)
        assert np.all(new_state.f <= 1.0)

    def test_nonuniform_grid_conserves_analytic_bcs_cell_mass(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.array([0.8, 0.9, 1.01, 1.04, 1.3, 1.6, 2.0])
        f = np.where((E > 1.0) & (E < 1.6), 0.1, 0.0)
        state = _state_on_custom_grid(E, f=f)
        dE = state.spectral.dE
        weights_before = bcs_dos_cell_weights(E, dE, state.gap)
        mass_before = float(np.sum(weights_before * state.f))
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.02,
        )

        out = DiffusionBackend().apply_gap_update(state, dt=1.0)

        weights_after = bcs_dos_cell_weights(E, dE, out.gap)
        mass_after = float(np.sum(weights_after * out.f))
        assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-15)

    def test_edge_remap_spreads_mass_without_saturating_one_bin(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(0.8, 2.0, 81)
        f = np.zeros_like(E)
        f[(E > 1.0) & (E < 1.08)] = 0.9
        state = _state_on_custom_grid(E, f=f)
        weights_before = bcs_dos_cell_weights(
            state.spectral.E,
            state.spectral.dE,
            state.gap,
        )
        mass_before = float(np.sum(weights_before * state.f))
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.1,
        )

        out = DiffusionBackend().apply_gap_update(state, dt=1.0)

        weights_after = bcs_dos_cell_weights(
            out.spectral.E,
            out.spectral.dE,
            out.gap,
        )
        mass_after = float(np.sum(weights_after * out.f))
        assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-15)
        assert np.count_nonzero(out.f > 1e-10) >= 2
        assert float(np.max(out.f)) < 1.0

    def test_gap_remap_tracks_frozen_xi_characteristic(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        gap_old = 1.0
        gap_new = 1.01
        E, _ = build_energy_grid(gap_old, 0.75, 6.0, 241)
        dE = integration_widths_from_centers(E)
        edges = cell_edges_from_widths(E, dE)

        def xi_edges(gap: float) -> np.ndarray:
            xi = np.zeros_like(edges)
            above = edges > gap
            xi[above] = np.sqrt((edges[above] - gap) * (edges[above] + gap))
            return xi

        def primitive(xi: np.ndarray) -> np.ndarray:
            x = np.minimum(xi, 3.0)
            return 0.2 * (x - 2.0 * x**3 / 27.0 + x**5 / 405.0)

        old_xi = xi_edges(gap_old)
        old_widths = np.diff(old_xi)
        f_old = np.zeros_like(E)
        active_old = old_widths > 0.0
        f_old[active_old] = np.diff(primitive(old_xi))[active_old] / old_widths[active_old]
        state = _state_on_custom_grid(E, gap=gap_old, f=f_old)
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: gap_new,
        )

        out = DiffusionBackend().apply_gap_update(state, dt=1.0)

        new_xi = xi_edges(gap_new)
        new_widths = np.diff(new_xi)
        target = np.zeros_like(E)
        active_new = new_widths > 0.0
        target[active_new] = np.diff(primitive(new_xi))[active_new] / new_widths[active_new]
        weights = bcs_dos_cell_weights(E, dE, gap_new)
        relative_l1 = float(np.sum(weights * np.abs(out.f - target))) / float(
            np.sum(weights * np.abs(target))
        )
        mass_before = float(np.sum(bcs_dos_cell_weights(E, dE, gap_old) * f_old))
        mass_after = float(np.sum(weights * out.f))

        assert relative_l1 < 1e-3
        assert mass_after == pytest.approx(mass_before, rel=1e-12, abs=1e-15)

    def test_rejects_material_finite_emax_characteristic_loss(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(0.8, 1.5, 101)
        state = _state_on_custom_grid(E, f=np.full_like(E, 0.2))
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.2,
        )

        with pytest.raises(RuntimeError, match="finite E_max boundary"):
            DiffusionBackend().apply_gap_update(state, dt=1.0)

    def test_rejects_finite_emax_tail_that_cannot_fit_last_cell(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E, _ = build_energy_grid(1.0, 0.75, 6.0, 80)
        state = _state_on_custom_grid(E, f=np.full_like(E, 0.999))
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.01,
        )

        with pytest.raises(RuntimeError, match="does not fit"):
            DiffusionBackend().apply_gap_update(state, dt=1.0)

    def test_gap_flow_uses_cell_average_dos_not_midpoint_rho(self) -> None:
        # A constant occupation exposes only the spectral measure error. On a
        # 40-cell grid the old midpoint sum misses 3.51% of the exact BCS
        # primitive; the finite-volume representation is exact by construction.
        gap = 1.0
        E, _ = build_energy_grid(gap, 0.75, 6.0, 40)
        dE = integration_widths_from_centers(E)
        spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        exact_weights = bcs_dos_cell_weights(E, dE, gap)
        f = np.full(E.size, 0.2)

        exact_mass = float(np.sum(exact_weights * f))
        midpoint_mass = float(np.sum(spectral.rho * dE * f))
        primitive = 0.2 * np.sqrt((6.0 * gap) ** 2 - gap**2)

        assert midpoint_mass / primitive - 1.0 == pytest.approx(
            -0.03512580555882283,
            rel=1e-12,
        )
        assert exact_mass == pytest.approx(primitive, rel=1e-14)

    def test_rejects_moving_gap_grid_with_missing_edge_support(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E, _ = build_energy_grid(1.0, 1.01, 2.0, 80)
        state = _state_on_custom_grid(E)
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.02,
        )

        with pytest.raises(ValueError, match="does not cover"):
            DiffusionBackend().apply_gap_update(state, dt=0.1)

    @pytest.mark.parametrize("collapsed_gap", [0.0, -1.0, float("nan")])
    def test_rejects_collapsed_or_nonfinite_gap_root(
        self,
        monkeypatch: pytest.MonkeyPatch,
        collapsed_gap: float,
    ) -> None:
        state = _state_on_physics_grid()
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: collapsed_gap,
        )
        with pytest.raises(RuntimeError, match="collapsed or non-finite"):
            DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_dynes_gap_motion(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(0.8, 2.0, 81)
        state = _state_on_custom_grid(E, dynes_gamma=0.01)
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 1.01,
        )
        with pytest.raises(ValueError, match="dynes_gamma == 0"):
            DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_falling_gap_without_lower_grid_room(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        E = np.linspace(1.01, 2.0, 81)
        state = _state_on_custom_grid(E)
        monkeypatch.setattr(
            "qpsim.backends.diffusion.solve_gap",
            lambda *_args, **_kwargs: 0.9,
        )
        with pytest.raises(ValueError, match="does not cover"):
            DiffusionBackend().apply_gap_update(state, dt=0.1)

    def test_rejects_state_spectral_gap_mismatch(self) -> None:
        state = _state_on_physics_grid()
        state.gap *= 0.99
        with pytest.raises(ValueError, match="must match"):
            DiffusionBackend().apply_gap_update(state, dt=0.0)


class TestStep:
    def test_moving_gap_coupling_is_second_order_and_constrained(self) -> None:
        backend = DiffusionBackend()
        initial = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        E = initial.spectral.E
        kT = KB_UEV_PER_K * initial.T_bath
        f_thermal = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        initial.f = np.clip(
            f_thermal + 0.012 * np.exp(-(((E / initial.gap - 1.35) / 0.18) ** 2)),
            0.0,
            1.0,
        )

        # Remove the algebraic start-up layer so the order measured below is
        # from time integration rather than an inconsistent initial state.
        initial = backend.apply_gap_update(initial, dt=1.0)

        gain = 0.05 * np.exp(-(((E / initial.gap - 1.6) / 0.25) ** 2))
        gain = np.where(initial.spectral.active_mask, gain, 0.0)
        external_flux = ExternalFlux(
            gain=gain,
            loss_rate=np.zeros_like(E),
        )

        def integrate(dt: float) -> DiffusionState:
            state = initial
            for _ in range(round(0.08 / dt)):
                state = backend.step(state, dt, external_flux=external_flux)
            return state

        step_sizes = (0.02, 0.01, 0.005, 0.0025)
        reference = integrate(0.000625)
        states = [integrate(dt) for dt in step_sizes]
        f_errors = [float(np.max(np.abs(state.f - reference.f))) for state in states]
        gap_errors = [abs(state.gap - reference.gap) for state in states]
        f_orders = [float(np.log2(coarse / fine)) for coarse, fine in pairwise(f_errors)]
        gap_orders = [float(np.log2(coarse / fine)) for coarse, fine in pairwise(gap_errors)]

        assert all(1.8 < order < 2.2 for order in f_orders)
        assert all(1.8 < order < 2.2 for order in gap_orders)
        calibration = calibrate_gap(
            T_c=initial.material.T_c,
            T_bath=initial.T_bath,
            Delta_0=initial.material.Delta_0,
        )
        for state in (*states, reference):
            constrained_gap = solve_gap(
                calibration,
                state.f,
                state.spectral.E,
                dE_bins=state.spectral.dE,
                reference_gap=state.gap,
                xtol=1e-12,
            )
            assert constrained_gap == pytest.approx(state.gap, abs=5e-10)
            assert np.all((state.f >= 0.0) & (state.f <= 1.0))

    def test_zero_collision_preserves_persistent_material_shells(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        backend = DiffusionBackend()
        state = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        E = state.spectral.E
        state.f = np.clip(
            state.f + 0.012 * np.exp(-(((E / state.gap - 1.35) / 0.18) ** 2)),
            0.0,
            1.0,
        )
        state = backend.apply_gap_update(state, dt=1.0)
        coordinates = backend._persistent_coordinates_for_state(state)
        xi_widths = np.diff(coordinates.xi_edges)
        mass_before = float(xi_widths @ coordinates.occupation)

        def zero_rates(
            _self: DiffusionBackend,
            _state: DiffusionState,
            _spectral: SpectralContext,
            f: np.ndarray,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(f), np.zeros_like(f)

        monkeypatch.setattr(
            DiffusionBackend,
            "_moving_gap_energy_collision_rates",
            zero_rates,
        )
        updated = backend.step(state, dt=0.1)
        updated_coordinates = updated._moving_gap_coordinates
        assert updated_coordinates is not None

        np.testing.assert_array_equal(
            updated_coordinates.occupation,
            coordinates.occupation,
        )
        assert updated.gap == state.gap
        assert float(updated.spectral.cell_weights @ updated.f) == pytest.approx(
            mass_before,
            rel=2e-15,
            abs=2e-15,
        )

        repeated = backend.step(updated, dt=0.1)
        assert repeated.gap == updated.gap
        np.testing.assert_array_equal(repeated.f, updated.f)
        np.testing.assert_array_equal(
            repeated._moving_gap_coordinates.occupation,
            updated_coordinates.occupation,
        )

    def test_scattering_only_conserves_exact_material_mass(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        backend = DiffusionBackend()
        state = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        E = state.spectral.E
        state.f = np.clip(
            state.f + 0.03 * np.exp(-(((E / state.gap - 2.0) / 0.3) ** 2)),
            0.0,
            1.0,
        )
        state = backend.apply_gap_update(state, dt=1.0)
        coordinates = backend._persistent_coordinates_for_state(state)
        mass_before = float(np.diff(coordinates.xi_edges) @ coordinates.occupation)
        monkeypatch.setattr(
            backend_module,
            "build_recombination_kernel_base",
            lambda spectral, *_args, **_kwargs: np.zeros_like(spectral.K_plus),
        )

        updated = backend.step(state, dt=0.05)
        updated_coordinates = updated._moving_gap_coordinates
        assert updated_coordinates is not None
        hidden_mass = float(np.diff(updated_coordinates.xi_edges) @ updated_coordinates.occupation)
        public_mass = float(updated.spectral.cell_weights @ updated.f)

        assert np.max(np.abs(updated.f - state.f)) > 1e-7
        assert hidden_mass == pytest.approx(mass_before, rel=2e-15, abs=2e-15)
        assert public_mass == pytest.approx(mass_before, rel=2e-15, abs=2e-15)
        assert np.all((updated.f >= 0.0) & (updated.f <= 1.0))

    def test_fixed_gap_limit_matches_apply_collisions(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module

        backend = DiffusionBackend()
        state = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        E = state.spectral.E
        state.f = np.clip(
            state.f + 0.012 * np.exp(-(((E / state.gap - 1.35) / 0.18) ** 2)),
            0.0,
            1.0,
        )
        state = backend.apply_gap_update(state, dt=1.0)

        # Holding the algebraic variable fixed must reduce the material-grid
        # method to the established uniform-E ETD2 collision update.
        monkeypatch.setattr(
            backend_module,
            "solve_gap",
            lambda *_args, **kwargs: float(kwargs["reference_gap"]),
        )
        fixed = backend.apply_collisions(state, dt=0.001)
        moving = backend.step(state, dt=0.001)

        assert moving.gap == state.gap
        np.testing.assert_allclose(moving.f, fixed.f, rtol=2e-14, atol=2e-15)

    def test_moving_gap_supports_fixed_energy_photon_channels(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3, num_energy=40)
        spacing = state.gap / 8.0
        E = (np.arange(40, dtype=float) + 7.5) * spacing
        dE = np.full_like(E, spacing)
        state.spectral = SpectralContext(
            E_bins=E,
            dE_bins=dE,
            gap=state.gap,
        )
        kT = KB_UEV_PER_K * state.T_bath
        state.f = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        omega, _, _, _ = build_phonon_frequency_map(E)
        state.phonon = PhononState(
            n_ph=thermal_phonon_occupation(omega, state.T_bath).reshape(1, -1, 1),
            omega_bins=omega.reshape(1, -1),
            tau_l=np.full((1, omega.size), 0.25),
            branches=[PhononBranchSpec(name="debye_average")],
        )
        backend = DiffusionBackend()
        state = backend.apply_gap_update(state, dt=1.0)
        photon_params = {
            "omega_0": 2.0 * spacing,
            "n_bar": 1.0e3,
            "c_phot": 1.0e-5,
        }
        pb_photon_params = {
            "omega_PB": 20.0 * spacing,
            "n_bar_PB": 1.0e3,
            "c_phot_PB": 1.0e-5,
        }

        undriven = backend.step(state, dt=0.001)
        driven = backend.step(
            state,
            dt=0.001,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
        )

        assert np.max(np.abs(driven.f - undriven.f)) > 1e-7
        assert abs(driven.gap - undriven.gap) > 1e-5
        assert np.all((driven.f >= 0.0) & (driven.f <= 1.0))

    def test_rejected_stage_trials_do_not_mutate_persistent_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import qpsim.backends.diffusion as backend_module
        from qpsim.solvers.etd import etd2_step as real_etd2_step

        backend = DiffusionBackend()
        state = _state_on_physics_grid(T_bath=0.3, num_energy=24)
        E = state.spectral.E
        state.f = np.where(
            (E / state.gap > 1.2) & (E / state.gap < 2.0),
            0.01,
            0.0,
        )
        state = backend.apply_gap_update(state, dt=1.0)

        def zero_rates(
            _self: DiffusionBackend,
            _state: DiffusionState,
            _spectral: SpectralContext,
            f: np.ndarray,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            return np.zeros_like(f), np.zeros_like(f)

        monkeypatch.setattr(
            DiffusionBackend,
            "_moving_gap_energy_collision_rates",
            zero_rates,
        )
        seeded = backend.step(state, dt=0.001)
        accepted = seeded._moving_gap_coordinates
        assert accepted is not None
        occupation_before = accepted.occupation.copy()
        public_before = seeded.f.copy()
        gap_before = seeded.gap

        def predictor_stiff_rates(
            _self: DiffusionBackend,
            _state: DiffusionState,
            _spectral: SpectralContext,
            f: np.ndarray,
            **_kwargs: object,
        ) -> tuple[np.ndarray, np.ndarray]:
            loss = np.zeros_like(f)
            occupied = f > 0.005
            loss[occupied] = 0.01 / f[occupied] ** 2
            return np.zeros_like(f), loss

        diagnostics: dict[str, int] = {}

        def monitored_etd2(*args: object, **kwargs: object) -> np.ndarray:
            kwargs["_diagnostics"] = diagnostics
            return real_etd2_step(*args, **kwargs)

        monkeypatch.setattr(
            DiffusionBackend,
            "_moving_gap_energy_collision_rates",
            predictor_stiff_rates,
        )
        monkeypatch.setattr(backend_module, "etd2_step", monitored_etd2)
        updated = backend.step(seeded, dt=0.003)

        assert diagnostics["rejected_attempts"] > 0
        assert diagnostics["accepted_substeps"] > 1
        np.testing.assert_array_equal(accepted.occupation, occupation_before)
        np.testing.assert_array_equal(seeded.f, public_before)
        assert seeded.gap == gap_before
        assert not accepted.occupation.flags.writeable
        assert updated._moving_gap_coordinates is not accepted
        assert np.all((updated.f >= 0.0) & (updated.f <= 1.0))
        assert np.all(
            (updated._moving_gap_coordinates.occupation >= 0.0)
            & (updated._moving_gap_coordinates.occupation <= 1.0)
        )

    def test_thermal_is_near_fixed_point(self) -> None:
        # A coupled moving-gap step on the thermal distribution should barely
        # move it (f_FD is the fixed point of the collision integral, and
        # Δ_eq(T_bath) is the fixed point of the gap equation) — but the
        # fixture seeds Δ_0 = 180 μeV, not Δ_eq(0.01 K) = 179.345 μeV, so the
        # first step also has to relax that algebraic start-up layer.
        #
        # Gate on the state's own scale: at 10 mK the occupations peak at
        # ~1e-80, so the historical atol=1e-4 stood 76 orders of magnitude
        # above the data and certified nothing — f ≡ 0 (total annihilation),
        # a uniform 1e-5 population and the energy-reversed array all passed.
        state = _state_on_physics_grid(T_bath=0.01)
        backend = DiffusionBackend()
        scale = float(np.max(state.f))
        started = backend.step(state, dt=0.01)
        np.testing.assert_allclose(started.f, state.f, rtol=0.0, atol=0.2 * scale)

        # With the start-up layer removed the same way the order-2 test above
        # removes it, the state is a genuine fixed point: every bin holds to a
        # few percent and the gap does not move at all. One coupled step lands
        # on that same relaxed gap.
        equilibrated = backend.apply_gap_update(state, dt=1.0)
        assert started.gap == pytest.approx(equilibrated.gap, rel=1e-12)
        stepped = backend.step(equilibrated, dt=0.01)
        np.testing.assert_allclose(stepped.f, equilibrated.f, rtol=5e-2, atol=0.0)
        assert stepped.gap == pytest.approx(equilibrated.gap, rel=1e-12)

        # The DOS-weighted mass rides the gap move to roundoff; annihilation
        # and a uniform warm population both break it.
        mass = float(equilibrated.spectral.cell_weights @ equilibrated.f)
        assert float(
            stepped.spectral.cell_weights @ stepped.f
        ) == pytest.approx(mass, rel=1e-12)

    def test_multiple_steps_remain_bounded(self) -> None:
        state = _state_on_physics_grid(T_bath=0.3)
        backend = DiffusionBackend()
        for _ in range(5):
            state = backend.step(state, dt=0.01)
            assert np.all(state.f >= 0.0)
            assert np.all(state.f <= 1.0)
            assert state.gap > 0


class TestRisingGapRecoveryTolerance:
    """2026-07-20 adjudication of the audit's split verdict: the persistent
    moving-gap path tolerates a stranded finite-E_max tail up to the same
    1e-3 fraction as apply_gap_update, warning above 1e-9, instead of
    refusing every rising-gap (recovery) trajectory at ~5e-12. The stranded
    mass stays frozen in the persistent representation (zero overlap) and
    re-enters if the gap falls."""

    def _grid_state(self) -> DiffusionState:
        # Sub-gap guard cells (E_min factor 0.9) so a risen gap up to ~1.05
        # stays covered by the grid bottom; uniform occupation so the
        # stranded xi sliver carries a controlled mass fraction.
        E, _ = build_energy_grid(1.0, 0.9, 3.0, 160)
        state = _state_on_custom_grid(E, gap=1.0)
        state.f = np.where(state.spectral.rho > 0.0, 1e-3, 0.0)
        return state

    def test_subcap_stranded_tail_warns_and_materializes(self) -> None:
        state = self._grid_state()
        backend = DiffusionBackend()
        coordinates = backend._persistent_coordinates_for_state(state)

        # Rising gap 1.0 -> 1.005 strands ~6e-4 of the uniform mass above
        # the fixed E_max window: below the 1e-3 cap, above the 1e-9 warn
        # threshold. Pre-fix this raised RuntimeError.
        with pytest.warns(UserWarning, match="left a finite-E_max"):
            spectral, f, overlap = backend._materialize_persistent_occupation(
                state, coordinates, 1.005
            )

        xi_widths = np.diff(coordinates.xi_edges)
        material_mass = float(xi_widths @ coordinates.occupation)
        represented_mass = float(spectral.cell_weights @ f)
        tail_fraction = (material_mass - represented_mass) / material_mass
        assert 1e-9 < tail_fraction < 1e-3
        # The exact-adjoint projection invariant survives untouched.
        np.testing.assert_allclose(
            overlap.T @ coordinates.occupation,
            spectral.cell_weights * f,
            rtol=5e-13,
            atol=1e-15,
        )

    def test_material_truncation_still_rejected_at_cap(self) -> None:
        state = self._grid_state()
        backend = DiffusionBackend()
        coordinates = backend._persistent_coordinates_for_state(state)

        # Rising gap 1.0 -> 1.05 strands ~6.5e-3 > the 1e-3 cap.
        with pytest.raises(RuntimeError, match="Extend the energy grid"):
            backend._materialize_persistent_occupation(state, coordinates, 1.05)

    def test_tail_fraction_cap_is_independent_of_occupation_amplitude(self) -> None:
        state = self._grid_state()
        state.f = np.where(state.spectral.rho > 0.0, 1e-16, 0.0)
        backend = DiffusionBackend()
        coordinates = backend._persistent_coordinates_for_state(state)

        # The geometry still strands ~0.66% of the mass.  An absolute
        # floating-point floor must not make that relative policy disappear.
        with pytest.raises(RuntimeError, match="Extend the energy grid"):
            backend._materialize_persistent_occupation(state, coordinates, 1.05)
