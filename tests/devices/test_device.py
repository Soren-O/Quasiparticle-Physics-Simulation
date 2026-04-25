"""Tests for Device, Junction, and solve_device_steady_state.

Phase 3 of the Device Architecture: the framework now supports
multi-region devices with tunnel coupling via Junctions. The
contract tests below pin the architectural invariants:

* Device + Junction validation (region names match).
* SymmetricGapTunnelingJunction.evaluate gives the right per-region
  gain/loss decomposition.
* Detailed balance: at matched temperature with no drive, two
  tunnel-coupled regions reach the same Fermi-Dirac steady state.
* The junction flux at the converged state is small (no net QP
  transport at detailed balance).
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import (
    Device,
    DeviceSolution,
    JunctionResult,
    Region,
    SymmetricGapTunnelingJunction,
    solve_device_steady_state,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.spectral import SpectralContext

# ═══════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════


def _build_state(
    *, T_bath: float, num_energy: int = 30, name_suffix: str = "",
) -> T3DiffusionState:
    """Build an Al-like T3 state at thermal equilibrium."""
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega_bins, _, _, _ = build_phonon_frequency_map(spectral.E)
    phonon = PhononState(
        n_ph=np.zeros((1, omega_bins.size, 1)),
        omega_bins=omega_bins.reshape(1, -1),
        tau_l=np.full((1, omega_bins.size), 0.25),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_init, gap=gap, spectral=spectral, phonon=phonon,
        material=material, T_bath=T_bath,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Device validation
# ═══════════════════════════════════════════════════════════════════════


class TestDeviceValidation:
    def test_rejects_junction_with_unknown_region_a(self) -> None:
        regions = {"L": Region(name="L", state=_build_state(T_bath=0.1))}
        bad_junction = SymmetricGapTunnelingJunction(
            name="J", region_a="ghost", region_b="L", alpha_per_ns=0.01,
        )
        with pytest.raises(ValueError, match="unknown region_a"):
            Device(regions=regions, junctions=[bad_junction])

    def test_rejects_junction_with_unknown_region_b(self) -> None:
        regions = {"L": Region(name="L", state=_build_state(T_bath=0.1))}
        bad_junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="ghost", alpha_per_ns=0.01,
        )
        with pytest.raises(ValueError, match="unknown region_b"):
            Device(regions=regions, junctions=[bad_junction])

    def test_accepts_single_region_zero_junction_device(self) -> None:
        # The simplest Device: one Region, no Junctions. Used to wrap
        # existing single-region Fischer code in the device framework.
        regions = {"main": Region(name="main", state=_build_state(T_bath=0.1))}
        device = Device(regions=regions, junctions=[])
        assert "main" in device.regions
        assert device.junctions == []


# ═══════════════════════════════════════════════════════════════════════
#  Junction validation + evaluate correctness
# ═══════════════════════════════════════════════════════════════════════


class TestSymmetricGapTunnelingJunction:
    def test_rejects_negative_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha_per_ns must be non-negative"):
            SymmetricGapTunnelingJunction(
                name="J", region_a="L", region_b="R", alpha_per_ns=-0.1,
            )

    def test_rejects_self_loop(self) -> None:
        with pytest.raises(ValueError, match="couple two different regions"):
            SymmetricGapTunnelingJunction(
                name="J", region_a="X", region_b="X", alpha_per_ns=0.01,
            )

    def test_evaluate_returns_symmetric_gain_loss(self) -> None:
        # gain_a ∝ f_b, loss_a = α (constant). Same for B.
        state_L = _build_state(T_bath=0.1, num_energy=20)
        state_R = _build_state(T_bath=0.1, num_energy=20)
        # Perturb each so f_L ≠ f_R.
        state_L = T3DiffusionState(
            f=state_L.f * 0.5,  # half-occupied
            gap=state_L.gap, spectral=state_L.spectral,
            phonon=state_L.phonon, material=state_L.material,
            T_bath=state_L.T_bath,
        )
        state_R = T3DiffusionState(
            f=state_R.f * 0.8,
            gap=state_R.gap, spectral=state_R.spectral,
            phonon=state_R.phonon, material=state_R.material,
            T_bath=state_R.T_bath,
        )

        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )
        result = junction.evaluate(state_L, state_R)

        assert isinstance(result, JunctionResult)
        # gain_a = α f_b
        np.testing.assert_allclose(
            result.external_flux_a.gain, 0.01 * state_R.f, atol=1e-15,
        )
        # loss_a = α everywhere
        np.testing.assert_allclose(
            result.external_flux_a.loss_rate, np.full_like(state_R.f, 0.01),
            atol=1e-15,
        )
        # symmetric: gain_b = α f_a, loss_b = α
        np.testing.assert_allclose(
            result.external_flux_b.gain, 0.01 * state_L.f, atol=1e-15,
        )
        np.testing.assert_allclose(
            result.external_flux_b.loss_rate, np.full_like(state_L.f, 0.01),
            atol=1e-15,
        )

    def test_evaluate_rejects_mismatched_E_grids(self) -> None:
        state_L = _build_state(T_bath=0.1, num_energy=20)
        state_R = _build_state(T_bath=0.1, num_energy=25)  # different
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )
        with pytest.raises(ValueError, match="matching E grids"):
            junction.evaluate(state_L, state_R)


# ═══════════════════════════════════════════════════════════════════════
#  Detailed-balance contract: matched T → both regions reach f_FD
# ═══════════════════════════════════════════════════════════════════════


class TestDetailedBalanceMatchedTemperature:
    """The headline Phase 3 invariant: at matched temperature with no
    external drive, two tunnel-coupled regions converge to the same
    Fermi-Dirac steady state, and the junction-mediated net flux
    between them vanishes.
    """

    def test_two_regions_matched_T_reach_thermal_steady_state(self) -> None:
        T_bath = 0.1   # 100 mK, well below T_c = 1.18 K for Al
        state_L = _build_state(T_bath=T_bath, num_energy=30)
        state_R = _build_state(T_bath=T_bath, num_energy=30)
        # Perturb the initial guesses away from the thermal fixed
        # point so the solver actually has work to do. Region L
        # starts colder (smaller f), region R starts warmer.
        state_L = T3DiffusionState(
            f=state_L.f * 0.5,
            gap=state_L.gap, spectral=state_L.spectral,
            phonon=state_L.phonon, material=state_L.material,
            T_bath=T_bath,
        )
        state_R = T3DiffusionState(
            f=state_R.f * 1.5,
            gap=state_R.gap, spectral=state_R.spectral,
            phonon=state_R.phonon, material=state_R.material,
            T_bath=T_bath,
        )

        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R",
                    alpha_per_ns=0.01,
                ),
            ],
        )

        sol = solve_device_steady_state(device, outer_tol=1e-9)
        assert isinstance(sol, DeviceSolution)
        assert sol.final_max_delta_f < 1e-9

        # Both regions should land on the SAME f, and that f should
        # match the bath Fermi-Dirac to within thermal-Newton tol.
        f_L_final = sol.states["L"].f
        f_R_final = sol.states["R"].f
        np.testing.assert_allclose(f_L_final, f_R_final, atol=1e-9)

        # Both should match the bath Fermi-Dirac.
        E = sol.states["L"].spectral.E
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        # Generous tolerance: the T3 Newton uses 1e-12 internally,
        # so the per-region answer is at machine precision but the
        # outer Picard adds another few orders of slop.
        np.testing.assert_allclose(f_L_final, f_FD, atol=1e-8)

    def test_junction_flux_vanishes_at_detailed_balance(self) -> None:
        # The net junction current at the converged state should be
        # zero (within solver tol). Compute (gain - loss·f) per region
        # and verify it's tiny.
        T_bath = 0.1
        state_L = _build_state(T_bath=T_bath, num_energy=30)
        state_R = _build_state(T_bath=T_bath, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R",
                    alpha_per_ns=0.01,
                ),
            ],
        )
        sol = solve_device_steady_state(device, outer_tol=1e-10)

        junction = device.junctions[0]
        result = junction.evaluate(sol.states["L"], sol.states["R"])
        # At detailed balance, gain - loss·f → 0 per bin.
        net_L = result.external_flux_a.gain - result.external_flux_a.loss_rate * sol.states["L"].f
        net_R = result.external_flux_b.gain - result.external_flux_b.loss_rate * sol.states["R"].f
        # Tolerance set to outer_tol × junction strength; junction
        # rate is 0.01 / ns, expect net ~ 0.01 × 1e-10 = 1e-12 / ns.
        assert np.max(np.abs(net_L)) < 1e-11
        assert np.max(np.abs(net_R)) < 1e-11


# ═══════════════════════════════════════════════════════════════════════
#  Convergence at mismatched temperatures
# ═══════════════════════════════════════════════════════════════════════


class TestMismatchedTemperatures:
    """At T_L ≠ T_R with weak tunneling, both regions stay close to
    their own bath FD. The Picard outer loop converges.
    """

    def test_mismatched_T_converges(self) -> None:
        T_L = 0.2  # K
        T_R = 0.1  # K
        sL = _build_state(T_bath=T_L, num_energy=30)
        sR = _build_state(T_bath=T_R, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=sL),
                "R": Region(name="R", state=sR),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R",
                    alpha_per_ns=1e-4,  # weak relative to e-ph
                ),
            ],
        )
        # Outer Picard convergence at 1e-5 — the detailed-balance test
        # above pins 1e-9 in the matched-T case where the outer loop
        # is effectively trivial. Mismatched T has slow outer modes the
        # plain Picard takes many iterations to drain; outer Anderson
        # acceleration is a Phase 4+ improvement.
        sol = solve_device_steady_state(device, outer_tol=1e-5)

        # Both regions reach valid distributions in [0, 1].
        for name in ("L", "R"):
            f = sol.states[name].f
            assert np.all(f >= 0.0)
            assert np.all(f <= 1.0)

        # Each region stays close to its own bath FD (weak tunneling).
        E = sol.states["L"].spectral.E
        f_FD_L = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * T_L), 500.0)) + 1.0)
        f_FD_R = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * T_R), 500.0)) + 1.0)
        # Loose tol matching the outer-Picard residual; strict check
        # is in the matched-T detailed-balance test above.
        np.testing.assert_allclose(sol.states["L"].f, f_FD_L, atol=1e-4)
        np.testing.assert_allclose(sol.states["R"].f, f_FD_R, atol=1e-4)
