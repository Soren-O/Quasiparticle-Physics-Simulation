"""Tests for the ExternalFlux dataclass + threading through the solver stack.

``ExternalFlux`` is the external-flux surface of a Region: the
``(gain, loss_rate)`` pair the Device layer adds to the RHS of the
region-local kinetic equation. The contract tests below pin:

* Dataclass validation (shape, signs, finite values, immutability).
* Linear ODE closed form: with all collision kernels disabled,
  ``f = gain / loss_rate`` is the unique steady state — pinned by the
  Newton solver and the backend ``steady_state`` method.
* Detailed-balance variant: gain/loss_rate matches Fermi-Dirac.
* Threading: nonzero ExternalFlux changes the solver output across
  every advertised public surface (forwarding regression tests for
  ``solve_steady_state``, ``coupled_newton_solve``, the diffusion
  backend methods, and ``run_time_dependent``).
* Grid-shape validation: length-1 broadcast and length mismatches
  are rejected at solver entry.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import ExternalFlux
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state
from qpsim.solvers.coupled_newton import coupled_newton_solve
from qpsim.solvers.newton_steady_state import newton_solve_f

# ═══════════════════════════════════════════════════════════════════════
#  Dataclass validation
# ═══════════════════════════════════════════════════════════════════════


class TestExternalFluxValidation:
    def test_zero_factory_shape(self) -> None:
        ef = ExternalFlux.zero(NE=10)
        assert ef.gain.shape == (10,)
        assert ef.loss_rate.shape == (10,)
        assert np.all(ef.gain == 0.0)
        assert np.all(ef.loss_rate == 0.0)

    def test_accepts_NE_1_squeeze(self) -> None:
        # (NE, 1) shape is accepted via transparent squeeze.
        gain_2d = np.ones((5, 1))
        loss_2d = np.full((5, 1), 0.5)
        ef = ExternalFlux(gain=gain_2d, loss_rate=loss_2d)
        assert ef.gain.shape == (5,)
        assert ef.loss_rate.shape == (5,)

    def test_rejects_2d_with_NR_gt_1(self) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            ExternalFlux(gain=np.ones((5, 3)), loss_rate=np.zeros((5, 3)))

    def test_rejects_negative_gain(self) -> None:
        with pytest.raises(ValueError, match="gain must be non-negative"):
            ExternalFlux(gain=-1 * np.ones(5), loss_rate=np.zeros(5))

    def test_rejects_negative_loss_rate(self) -> None:
        with pytest.raises(ValueError, match="loss_rate must be non-negative"):
            ExternalFlux(gain=np.zeros(5), loss_rate=-0.1 * np.ones(5))

    def test_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="shapes must match"):
            ExternalFlux(gain=np.zeros(5), loss_rate=np.zeros(6))

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="non-finite"):
            ExternalFlux(gain=np.array([1.0, np.nan, 1.0]), loss_rate=np.zeros(3))
        with pytest.raises(ValueError, match="non-finite"):
            ExternalFlux(gain=np.zeros(3), loss_rate=np.array([1.0, np.inf, 1.0]))

    @pytest.mark.parametrize("field", ["gain", "loss_rate"])
    @pytest.mark.parametrize("imaginary", [1.0, float("nan")])
    def test_rejects_complex_before_float_cast(
        self,
        field: str,
        imaginary: float,
    ) -> None:
        arrays = {
            "gain": np.zeros(3),
            "loss_rate": np.zeros(3),
        }
        arrays[field] = arrays[field].astype(complex)
        arrays[field][0] = complex(0.0, imaginary)

        with pytest.raises(ValueError, match=rf"{field} must be real-valued"):
            ExternalFlux(**arrays)

    def test_frozen_immutable(self) -> None:
        ef = ExternalFlux.zero(5)
        # frozen=True dataclass raises FrozenInstanceError on attribute set.
        from dataclasses import FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            ef.gain = np.ones(5)  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════
#  Linear-ODE closed form: f = gain / loss_rate when kernels disabled
# ═══════════════════════════════════════════════════════════════════════


def _make_ctx(NE: int = 30, gap: float = 175.0, dE: float = 5.0) -> SpectralContext:
    """Minimal SpectralContext for collision-free ExternalFlux contract tests.

    Energy grid spans [gap, gap + (NE-1)*dE] in μeV.
    """
    # Keep this general closed-form fixture entirely on represented BCS rows.
    # Zero-capacity source behavior has dedicated tests below.
    E = gap + dE * (np.arange(NE, dtype=float) + 1.0)
    dE_arr = np.full_like(E, dE)
    return SpectralContext(E_bins=E, dE_bins=dE_arr, gap=gap)


# ═══════════════════════════════════════════════════════════════════════
#  Grid-shape validation: catches the silent length-1 broadcast pathology
#  AT THE SOLVER ENTRY (the dataclass alone can't know the grid size).
# ═══════════════════════════════════════════════════════════════════════


class TestGridShapeValidation:
    """The dataclass constructor only checks 1D-ness and matching sizes
    between gain/loss_rate. A length-``M`` flux passed to a length-``NE``
    grid sneaks through if ``M ≠ NE`` AND the offending solver site
    relies on NumPy broadcasting. The pathological case is ``M = 1``,
    which broadcasts silently across every bin and turns a single-bin
    contract into an all-bin one. ``_validate_for_NE`` at solver entry
    rejects it with a clear error.
    """

    def test_length_one_flux_rejected_in_newton(self) -> None:
        ctx = _make_ctx(NE=20)
        ef = ExternalFlux(gain=np.full(1, 0.1), loss_rate=np.full(1, 1.0))
        with pytest.raises(ValueError, match="sized for 1 energy bins"):
            newton_solve_f(ctx, np.full(20, 0.5), external_flux=ef)

    def test_length_mismatch_flux_rejected_in_newton(self) -> None:
        ctx = _make_ctx(NE=20)
        ef = ExternalFlux(gain=np.zeros(15), loss_rate=np.zeros(15))
        with pytest.raises(ValueError, match="sized for 15 energy bins"):
            newton_solve_f(ctx, np.full(20, 0.5), external_flux=ef)

    def test_length_mismatch_rejected_in_solve_steady_state(self) -> None:
        # The validation lives in solve_steady_state too — pinning the
        # service-layer kwarg threading.
        from qpsim.collisions.phonon import (
            build_recombination_kernel_base as _build_r,
        )
        from qpsim.collisions.phonon import (
            build_scattering_kernel_base as _build_s,
        )
        from qpsim.grid.energy_grid import (
            build_energy_grid as _grid,
        )
        from qpsim.grid.energy_grid import (
            integration_widths_from_centers as _widths,
        )

        T_c = 1.2
        gap = 1.764 * KB_UEV_PER_K * T_c
        E, _ = _grid(gap=gap, energy_min_factor=1.01, energy_max_factor=6.0,
                     num_energy_bins=20)
        ctx = SpectralContext(E_bins=E, dE_bins=_widths(E), gap=gap)
        K_s0 = _build_s(ctx, tau_0=1.0, T_c=T_c)
        K_r0 = _build_r(ctx, tau_0=1.0, T_c=T_c)
        bad = ExternalFlux(gain=np.zeros(1), loss_rate=np.zeros(1))
        with pytest.raises(ValueError, match="sized for 1 energy bins"):
            solve_steady_state(ctx, K_s0, K_r0, T_bath=0.3, external_flux=bad)

    def test_gain_on_zero_spectral_capacity_is_rejected_in_newton(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([0.8, 1.2]),
            dE_bins=np.array([0.4, 0.4]),
            gap=1.0,
        )
        flux = ExternalFlux(
            gain=np.array([0.1, 0.0]),
            loss_rate=np.zeros(2),
        )

        with pytest.raises(ValueError, match="zero-spectral-capacity"):
            newton_solve_f(ctx, np.zeros(2), external_flux=flux)

    def test_loss_only_on_zero_capacity_row_is_inert(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([0.8, 1.2]),
            dE_bins=np.array([0.4, 0.4]),
            gap=1.0,
        )
        initial = np.array([0.7, 0.2])
        flux = ExternalFlux(
            gain=np.zeros(2),
            loss_rate=np.array([3.0, 0.0]),
        )

        solved = newton_solve_f(ctx, initial, external_flux=flux)

        np.testing.assert_array_equal(solved, initial)

    def test_gain_on_zero_capacity_is_rejected_in_coupled_newton(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([0.8, 1.2]),
            dE_bins=np.array([0.4, 0.4]),
            gap=1.0,
        )
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
        flux = ExternalFlux(
            gain=np.array([0.1, 0.0]),
            loss_rate=np.zeros(2),
        )

        with pytest.raises(ValueError, match="zero-spectral-capacity"):
            coupled_newton_solve(
                ctx,
                np.zeros(2),
                np.zeros(omega.size),
                omega_bins=omega,
                omega_idx_diff=idx_diff,
                omega_idx_sum=idx_sum,
                diff_sign=diff_sign,
                tau_l=0.5,
                external_flux=flux,
            )

    def test_coupled_newton_solves_only_supported_f_rows(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([0.8, 1.2]),
            dE_bins=np.array([0.4, 0.4]),
            gap=1.0,
        )
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
        flux = ExternalFlux(
            gain=np.array([0.0, 0.1]),
            loss_rate=np.array([0.0, 1.0]),
        )

        solved, _ = coupled_newton_solve(
            ctx,
            np.array([0.7, 0.2]),
            np.zeros(omega.size),
            omega_bins=omega,
            omega_idx_diff=idx_diff,
            omega_idx_sum=idx_sum,
            diff_sign=diff_sign,
            tau_l=0.5,
            external_flux=flux,
            tol=1e-14,
        )

        np.testing.assert_allclose(solved, [0.7, 0.1], atol=1e-14)

    def test_coupled_newton_rejects_all_zero_dos_context(self) -> None:
        ctx = SpectralContext(
            E_bins=np.array([0.8, 1.2]),
            dE_bins=np.array([0.4, 0.4]),
            gap=1.0,
        )
        # SpectralContext normally guarantees positive represented capacity. This
        # defensive synthetic context pins the solver boundary if a future
        # spectral model legitimately supplies no quasiparticle capacity.
        ctx._rho = np.zeros(2)  # type: ignore[attr-defined]
        ctx._cell_weights = np.zeros(2)  # type: ignore[attr-defined]
        ctx._cell_density = np.zeros(2)  # type: ignore[attr-defined]
        ctx._active_mask = np.zeros(2, dtype=bool)  # type: ignore[attr-defined]
        omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)

        with pytest.raises(ValueError, match="at least one quasiparticle energy row"):
            coupled_newton_solve(
                ctx,
                np.zeros(2),
                np.zeros(omega.size),
                omega_bins=omega,
                omega_idx_diff=idx_diff,
                omega_idx_sum=idx_sum,
                diff_sign=diff_sign,
                T_bath=0.2,
                tau_l=0.5,
            )


class TestLinearODEClosedForm:
    """With collision kernels disabled, f satisfies df/dt = gain - loss_rate · f.
    Steady state is f = gain / loss_rate by direct construction.
    """

    def test_constant_flux_recovers_closed_form(self) -> None:
        ctx = _make_ctx()
        NE = ctx.E.size
        gain = np.full(NE, 0.05)        # 0.05 / ns
        loss_rate = np.full(NE, 0.5)    # 0.5 / ns; expected f = 0.1 ∈ [0, 1]
        ef = ExternalFlux(gain=gain, loss_rate=loss_rate)

        # No K_s0 / K_r0 / photon kernels — purely linear ODE in f.
        # Pass active=all-True so the contract test exercises every bin.
        # The default active mask drops bins near the gap edge.
        f_solved = newton_solve_f(
            ctx, f=np.full(NE, 0.5),  # any seed
            active=np.ones(NE, dtype=bool),
            external_flux=ef,
            tol=1e-14,
        )
        expected = np.full(NE, 0.1)
        assert np.allclose(f_solved, expected, atol=1e-14, rtol=0)

    def test_varying_flux_recovers_closed_form(self) -> None:
        # Pointwise variation: f(E) = gain(E) / loss_rate(E).
        ctx = _make_ctx()
        NE = ctx.E.size
        rng = np.random.default_rng(seed=42)
        gain = 0.1 * rng.uniform(0.5, 1.5, size=NE)
        loss_rate = 1.0 * rng.uniform(0.5, 1.5, size=NE)
        # Cap so f stays in [0, 1]
        loss_rate = np.maximum(loss_rate, 2.0 * gain)
        ef = ExternalFlux(gain=gain, loss_rate=loss_rate)

        f_solved = newton_solve_f(
            ctx, f=np.full(NE, 0.5),
            active=np.ones(NE, dtype=bool),
            external_flux=ef,
            tol=1e-14,
        )
        expected = gain / loss_rate
        assert np.allclose(f_solved, expected, atol=1e-13, rtol=0)

    def test_zero_external_flux_yields_initial_guess(self) -> None:
        # No drive, no collisions, no flux → any f is a fixed point.
        # Newton's first step makes the residual zero by construction
        # (residual = 0 already at any f when nothing's enabled).
        ctx = _make_ctx()
        NE = ctx.E.size
        ef = ExternalFlux.zero(NE)
        f0 = np.full(NE, 0.3)
        f_solved = newton_solve_f(ctx, f0, external_flux=ef, tol=1e-14)
        assert np.allclose(f_solved, f0, atol=1e-14)

    def test_external_flux_None_path_unchanged(self) -> None:
        # The default path (external_flux=None) remains a no-op.
        ctx = _make_ctx()
        NE = ctx.E.size
        f0 = np.full(NE, 0.3)
        f_with_none = newton_solve_f(ctx, f0, external_flux=None, tol=1e-14)
        f_without_kwarg = newton_solve_f(ctx, f0, tol=1e-14)
        assert np.allclose(f_with_none, f_without_kwarg, atol=1e-14)


# ═══════════════════════════════════════════════════════════════════════
#  Detailed-balance ansatz via the linear ODE
# ═══════════════════════════════════════════════════════════════════════


class TestDetailedBalance:
    def test_fermi_dirac_via_constructed_gain_loss(self) -> None:
        # Pick gain/loss_rate so the steady state IS f_FD(E, T_bath).
        # Setting loss_rate = 1/ns and gain = f_FD(E, T) /ns gives
        # f_steady = f_FD(E, T) by construction. This is the "contract
        # supports detailed-balance ansätze" check, not a physics
        # derivation.
        from qpsim.constants import KB_UEV_PER_K as _KB
        ctx = _make_ctx()
        T = 0.1  # K
        f_FD = 1.0 / (np.exp(ctx.E / (_KB * T)) + 1.0)
        gain = 1.0 * f_FD
        loss_rate = np.ones_like(f_FD)
        ef = ExternalFlux(gain=gain, loss_rate=loss_rate)

        f_solved = newton_solve_f(
            ctx, f=np.full(ctx.E.size, 0.5),
            active=np.ones(ctx.E.size, dtype=bool),
            external_flux=ef,
            tol=1e-14,
        )
        assert np.allclose(f_solved, f_FD, atol=1e-13)


# ═══════════════════════════════════════════════════════════════════════
#  Conservation invariant for ExternalFlux (no collisions)
# ═══════════════════════════════════════════════════════════════════════


class TestConservationUnderInjection:
    def test_dn_qp_dt_at_initial_matches_flux_observable(self) -> None:
        # With kernels disabled, ∂_t n_qp at the initial state equals
        # 4 ρ_F ∫ ρ(E) (gain − loss_rate × f) dE — this is the
        # observable-level form the design doc §3.2.1 specifies.
        # (Tested via the analytical residual at the initial state,
        # since the observable's own internals aren't needed here.)
        ctx = _make_ctx()
        NE = ctx.E.size
        rho_F = 1.74e28
        gain = np.full(NE, 0.05)
        loss_rate = np.full(NE, 0.5)
        ef = ExternalFlux(gain=gain, loss_rate=loss_rate)
        f0 = np.full(NE, 0.2)

        # Direct construction of d(n_qp)/dt at f = f0.
        rhs_per_bin = ef.gain - ef.loss_rate * f0
        expected_dn_qp_dt = 4.0 * rho_F * float(
            np.sum(ctx.rho * rhs_per_bin * ctx.dE)
        )

        # Solve to steady state and verify d(n_qp)/dt = 0 there.
        f_steady = newton_solve_f(
            ctx, f=f0,
            active=np.ones(NE, dtype=bool),
            external_flux=ef, tol=1e-14,
        )
        rhs_steady = ef.gain - ef.loss_rate * f_steady
        dn_qp_dt_steady = 4.0 * rho_F * float(
            np.sum(ctx.rho * rhs_steady * ctx.dE)
        )
        assert abs(dn_qp_dt_steady) < 1e-10 * abs(expected_dn_qp_dt)


# ═══════════════════════════════════════════════════════════════════════
#  Forwarding regression tests across every public surface
# ═══════════════════════════════════════════════════════════════════════
#
#  These exercise the threading rather than the physics: each test
#  builds a Fischer-like setup, runs the surface twice (with and
#  without an ExternalFlux), and asserts the two outputs differ. The
#  goal is to catch a silent dropped-kwarg regression on any of the
#  five advertised public surfaces.


def _setup_fischer_like(num: int = 25):
    """A small Fischer-like SpectralContext + e-ph kernels."""
    T_c = 1.2
    T_bath = 0.3
    gap = 1.764 * KB_UEV_PER_K * T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=num
    )
    dE = integration_widths_from_centers(E)
    ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    K_s0 = build_scattering_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    K_r0 = build_recombination_kernel_base(ctx, tau_0=1.0, T_c=T_c)
    return ctx, K_s0, K_r0, T_bath


def _modest_external_flux(NE: int) -> ExternalFlux:
    """A small ExternalFlux that perturbs the steady state without
    derailing iterative solvers (Picard, coupled Newton)."""
    return ExternalFlux(gain=np.full(NE, 1e-6), loss_rate=np.full(NE, 1e-4))


class TestForwardingThroughPublicSurfaces:
    def test_solve_steady_state_thermal_phonon_path(self) -> None:
        # Thermal-phonon path (phonon_escape_time=None): just Newton.
        ctx, K_s0, K_r0, T_bath = _setup_fischer_like()
        ef = _modest_external_flux(ctx.E.size)
        f_no_flux = solve_steady_state(ctx, K_s0, K_r0, T_bath, tol=1e-12)
        f_with_flux = solve_steady_state(
            ctx, K_s0, K_r0, T_bath, external_flux=ef, tol=1e-12,
        )
        # Some bins must change non-trivially.
        assert np.max(np.abs(f_with_flux - f_no_flux)) > 1e-6

    def test_coupled_newton_solve(self) -> None:
        ctx, K_s0, K_r0, T_bath = _setup_fischer_like()
        ef = _modest_external_flux(ctx.E.size)
        omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
        n_omega = omega_bins.size
        n_ph_init = np.zeros(n_omega)
        kT = KB_UEV_PER_K * T_bath
        f_init = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)

        f_no, _ = coupled_newton_solve(
            ctx, f_init, n_ph_init,
            omega_bins=omega_bins,
            omega_idx_diff=idx_diff, omega_idx_sum=idx_sum,
            diff_sign=diff_sign,
            K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, tau_l=0.5,
        )
        f_yes, _ = coupled_newton_solve(
            ctx, f_init, n_ph_init,
            omega_bins=omega_bins,
            omega_idx_diff=idx_diff, omega_idx_sum=idx_sum,
            diff_sign=diff_sign,
            K_s0=K_s0, K_r0=K_r0, T_bath=T_bath, tau_l=0.5,
            external_flux=ef,
        )
        assert np.max(np.abs(f_yes - f_no)) > 1e-6


class TestForwardingThroughDiffusionBackend:
    def _build_state(self, num_energy: int = 25):
        from qpsim.backends.diffusion import DiffusionState
        from qpsim.materials.database import Material
        from qpsim.phonon_models.state import PhononBranchSpec, PhononState

        T_c = 1.2
        gap = 1.764 * KB_UEV_PER_K * T_c
        E, _ = build_energy_grid(
            gap=gap, energy_min_factor=1.01, energy_max_factor=6.0,
            num_energy_bins=num_energy,
        )
        dE = integration_widths_from_centers(E)
        ctx = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
        material = Material(
            name="test", Delta_0=gap, T_c=T_c, tau_0=1.0,
            tau_0_pb_ns=0.255,
        )
        omega_bins, _, _, _ = build_phonon_frequency_map(ctx.E)
        phonon = PhononState(
            n_ph=thermal_phonon_occupation(omega_bins, 0.3).reshape(1, -1, 1),
            omega_bins=omega_bins.reshape(1, -1),
            tau_l=np.full((1, omega_bins.size), 0.5),
            branches=[PhononBranchSpec(name="debye_average")],
        )
        kT = KB_UEV_PER_K * 0.3
        f0 = 1.0 / (np.exp(np.minimum(ctx.E / kT, 500.0)) + 1.0)
        return DiffusionState(
            f=f0, gap=gap, spectral=ctx, phonon=phonon,
            material=material, T_bath=0.3,
        )

    def test_steady_state_thermal_phonons(self) -> None:
        # backend.steady_state(use_thermal_phonons=True) routes through
        # solve_steady_state thermal-phonon path → newton_solve_f.
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        s_no = backend.steady_state(state, use_thermal_phonons=True)
        s_yes = backend.steady_state(
            state, use_thermal_phonons=True, external_flux=ef,
        )
        assert np.max(np.abs(s_yes.f - s_no.f)) > 1e-6

    def test_steady_state_coupled_newton(self) -> None:
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        s_no = backend.steady_state(state, method="coupled_newton")
        s_yes = backend.steady_state(
            state, method="coupled_newton", external_flux=ef,
        )
        assert np.max(np.abs(s_yes.f - s_no.f)) > 1e-6

    def test_steady_state_picard_with_anderson(self) -> None:
        # The default backend method is method="picard" (finite-τ_l Picard
        # outer loop on n_ph). With anderson_depth ≥ 1 the path stabilizes
        # under ExternalFlux. This test pins the kwarg threading through
        # the Anderson-Picard path; the unaccelerated-Picard guard test
        # below pins the explicit-error contract.
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        # Exercise the minimum advertised accelerated depth: depth=1 must be a
        # real secant update, not silently routed to plain Picard.
        s_no = backend.steady_state(state, method="picard", anderson_depth=1)
        s_yes = backend.steady_state(
            state, method="picard", anderson_depth=1, external_flux=ef,
        )
        assert np.max(np.abs(s_yes.f - s_no.f)) > 1e-6

    def test_default_picard_with_external_flux_raises_helpful_error(self) -> None:
        # The default (method='picard', anderson_depth=0) was previously
        # silently failing with a "did not converge in 200 iterations"
        # error when ExternalFlux is non-zero — confusing to users since
        # the no-flux default works. The backend now catches this at the
        # API boundary and raises a ValueError pointing at the three
        # routes: anderson_depth >= 1, coupled_newton, or use_thermal_phonons.
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        with pytest.raises(ValueError, match=r"anderson_depth >= 1.*coupled_newton"):
            backend.steady_state(state, external_flux=ef)
        # Zero external_flux still works on the default path.
        zero_ef = ExternalFlux.zero(state.spectral.E.size)
        # Should NOT raise the new ValueError (the gain+loss == 0 short-circuit).
        backend.steady_state(state, external_flux=zero_ef)

    def test_shape_validation_runs_before_picard_guard(self) -> None:
        # When BOTH a wrong-sized flux AND default-Picard are in play,
        # the user should see the clearer "sized for {M} energy bins"
        # shape error, not the Picard-routing error. Validation order:
        # shape first, then Picard guard.
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        # length-1 flux on a 25-bin grid would silently broadcast.
        bad = ExternalFlux(gain=np.full(1, 0.1), loss_rate=np.full(1, 1.0))
        with pytest.raises(ValueError, match="sized for 1 energy bins"):
            # Default method='picard' would also trip the Picard guard,
            # but shape validation must win.
            backend.steady_state(state, external_flux=bad)

    def test_apply_collisions_one_step(self) -> None:
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        s_no = backend.apply_collisions(state, dt=0.5)
        s_yes = backend.apply_collisions(state, dt=0.5, external_flux=ef)
        assert np.max(np.abs(s_yes.f - s_no.f)) > 1e-9

    def test_fixed_gap_collision_substep(self) -> None:
        # This compact fixture intentionally starts above the BCS edge, so it
        # exercises ExternalFlux forwarding through the fixed-gap collision
        # API. Moving-gap ``step`` has separate tests on a supporting grid.
        from qpsim.backends.diffusion import DiffusionBackend

        backend = DiffusionBackend()
        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        s_no = backend.apply_collisions(state, dt=0.5)
        s_yes = backend.apply_collisions(state, dt=0.5, external_flux=ef)
        assert np.max(np.abs(s_yes.f - s_no.f)) > 1e-9


class TestServiceLayerPicardGuard:
    """``solve_steady_state`` mirrors the backend-level Picard guard so
    direct service callers get the same explicit routing hint instead
    of a confusing 200-iteration timeout.
    """

    def test_solve_steady_state_default_picard_with_flux_raises(self) -> None:
        # Direct service call with finite phonon_escape_time and
        # anderson_depth=0 (the default) raises the same helpful error
        # that the backend layer raises.
        ctx, K_s0, K_r0, T_bath = _setup_fischer_like()
        ef = _modest_external_flux(ctx.E.size)
        with pytest.raises(ValueError, match=r"anderson_depth >= 1"):
            solve_steady_state(
                ctx,
                K_s0,
                K_r0,
                T_bath,
                external_flux=ef,
                phonon_escape_time=0.5,
            )

    def test_solve_steady_state_picard_with_anderson_works(self) -> None:
        # Anderson-accelerated Picard threads ExternalFlux through the
        # service path successfully.
        ctx, K_s0, K_r0, T_bath = _setup_fischer_like()
        ef = _modest_external_flux(ctx.E.size)
        f_no = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            phonon_escape_time=0.5, anderson_depth=1,
        )
        f_yes = solve_steady_state(
            ctx, K_s0, K_r0, T_bath, external_flux=ef,
            phonon_escape_time=0.5, anderson_depth=1,
        )
        assert np.max(np.abs(f_yes - f_no)) > 1e-6

    def test_solve_steady_state_zero_flux_does_not_trip_guard(self) -> None:
        # Zero ExternalFlux on the default-Picard service path is fine.
        ctx, K_s0, K_r0, T_bath = _setup_fischer_like()
        zero_ef = ExternalFlux.zero(ctx.E.size)
        f = solve_steady_state(
            ctx, K_s0, K_r0, T_bath,
            external_flux=zero_ef,
            phonon_escape_time=0.5, picard_tol=1e-8, picard_mixing=0.3,
        )
        assert f.shape == (ctx.E.size,)


class TestForwardingThroughTransient:
    def _build_state(self, num_energy: int = 25):
        return TestForwardingThroughDiffusionBackend()._build_state(num_energy)

    def test_static_external_flux(self) -> None:
        from qpsim.services.transient import run_time_dependent

        state = self._build_state()
        ef = _modest_external_flux(state.spectral.E.size)
        r_no = run_time_dependent(state, dt=0.5, total_time=2.0, snapshot_interval=1.0)
        r_yes = run_time_dependent(
            state, dt=0.5, total_time=2.0, snapshot_interval=1.0,
            external_flux=ef,
        )
        # Final-time f differs.
        assert np.max(np.abs(r_yes.snapshots[-1].f - r_no.snapshots[-1].f)) > 1e-9

    def test_callable_external_flux(self) -> None:
        from qpsim.services.transient import run_time_dependent

        state = self._build_state()
        NE = state.spectral.E.size

        # Time-varying flux: gain ramps from 1e-4 to 1e-2 across the window.
        def flux_fn(t: float) -> ExternalFlux:
            scale = 1e-4 + 1e-2 * (t / 2.0)
            return ExternalFlux(gain=np.full(NE, scale), loss_rate=np.full(NE, 0.1))

        r_callable = run_time_dependent(
            state, dt=0.5, total_time=2.0, snapshot_interval=1.0,
            external_flux=flux_fn,
        )
        r_static_endpoint = run_time_dependent(
            state, dt=0.5, total_time=2.0, snapshot_interval=1.0,
            external_flux=flux_fn(2.0),
        )
        # The two should differ — callable evaluates at each step,
        # static uses the t=2.0 value at every step.
        assert np.max(np.abs(
            r_callable.snapshots[-1].f - r_static_endpoint.snapshots[-1].f
        )) > 1e-9
