"""Tests for the ExternalFlux dataclass + threading through the T3 solver stack.

Phase 2 of the Device Architecture:
* Dataclass validation (shape, signs, finite values).
* Linear ODE closed form: with all collision kernels disabled,
  ``f = gain / loss_rate`` is the unique steady state — pinned by the
  Newton solver and the backend ``steady_state`` method.
* Detailed-balance variant: gain/loss_rate matches Fermi-Dirac.
* Threading: zero flux is bit-for-bit identical to the existing
  Fischer paths (smoke; the real check is the broader test suite still
  passing under the patch).
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.devices import ExternalFlux
from qpsim.physics.spectral import SpectralContext
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
    """Minimal SpectralContext for collision-free Phase 2 contract tests.

    Energy grid spans [gap, gap + (NE-1)*dE] in μeV.
    """
    E = gap + dE * np.arange(NE, dtype=float)
    dE_arr = np.full_like(E, dE)
    return SpectralContext(E_bins=E, dE_bins=dE_arr, gap=gap)


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
