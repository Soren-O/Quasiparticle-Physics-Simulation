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

import copy
from dataclasses import dataclass, replace
from typing import ClassVar

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import (
    Device,
    DeviceSolution,
    ExternalFlux,
    Junction,
    JunctionResult,
    Qubit,
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
    @pytest.mark.parametrize(
        "control",
        ["alpha_per_ns", "capacity_ratio_a_to_b"],
    )
    @pytest.mark.parametrize("bad_value", [True, 1.0 + 0.0j, "0.1"])
    def test_rejects_coercible_non_real_scalar_controls(
        self,
        control: str,
        bad_value: object,
    ) -> None:
        kwargs: dict[str, object] = {
            "name": "J",
            "region_a": "L",
            "region_b": "R",
            "alpha_per_ns": 0.1,
            "capacity_ratio_a_to_b": 1.0,
        }
        kwargs[control] = bad_value
        with pytest.raises(ValueError, match=control):
            SymmetricGapTunnelingJunction(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("alpha", [-0.1, np.inf, np.nan])
    def test_rejects_invalid_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="finite and non-negative"):
            SymmetricGapTunnelingJunction(
                name="J", region_a="L", region_b="R", alpha_per_ns=alpha,
            )

    @pytest.mark.parametrize("ratio", [0.0, -1.0, np.inf, np.nan])
    def test_rejects_invalid_capacity_ratio(self, ratio: float) -> None:
        with pytest.raises(ValueError, match="capacity_ratio_a_to_b"):
            SymmetricGapTunnelingJunction(
                name="J",
                region_a="L",
                region_b="R",
                alpha_per_ns=0.1,
                capacity_ratio_a_to_b=ratio,
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

    def test_unequal_capacities_conserve_weighted_population(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        state_a = T3DiffusionState(
            f=np.full_like(state_a.f, 0.2),
            gap=state_a.gap,
            spectral=state_a.spectral,
            phonon=state_a.phonon,
            material=state_a.material,
            T_bath=state_a.T_bath,
        )
        state_b = T3DiffusionState(
            f=np.full_like(state_b.f, 0.8),
            gap=state_b.gap,
            spectral=state_b.spectral,
            phonon=state_b.phonon,
            material=state_b.material,
            T_bath=state_b.T_bath,
        )
        capacity_ratio = 3.0
        result = SymmetricGapTunnelingJunction(
            name="J",
            region_a="L",
            region_b="R",
            alpha_per_ns=0.04,
            capacity_ratio_a_to_b=capacity_ratio,
        ).evaluate(state_a, state_b)

        rhs_a = (
            result.external_flux_a.gain
            - result.external_flux_a.loss_rate * state_a.f
        )
        rhs_b = (
            result.external_flux_b.gain
            - result.external_flux_b.loss_rate * state_b.f
        )
        # Set C_b = 1, hence C_a = capacity_ratio.  Conservation must hold
        # independently in every matched energy bin.
        np.testing.assert_allclose(capacity_ratio * rhs_a + rhs_b, 0.0, atol=1e-15)
        assert result.external_flux_b.diagnostics["alpha_b_per_ns"] == pytest.approx(
            0.12
        )

    def test_capacity_contract_is_present_only_for_active_coupling(self) -> None:
        active = SymmetricGapTunnelingJunction(
            name="active",
            region_a="L",
            region_b="R",
            alpha_per_ns=0.1,
            capacity_ratio_a_to_b=3.0,
        )
        inactive = SymmetricGapTunnelingJunction(
            name="inactive",
            region_a="L",
            region_b="R",
            alpha_per_ns=0.0,
            capacity_ratio_a_to_b=np.nextafter(0.0, 1.0),
        )
        assert active.qp_number_capacity_ratio_a_to_b() == 3.0
        assert inactive.qp_number_capacity_ratio_a_to_b() is None

    def test_zero_alpha_is_noop_even_on_unmatched_grids(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=25)
        junction = SymmetricGapTunnelingJunction(
            name="off",
            region_a="L",
            region_b="R",
            alpha_per_ns=0.0,
            capacity_ratio_a_to_b=np.nextafter(0.0, 1.0),
        )
        result = junction.evaluate(state_a, state_b)
        assert result.external_flux_a.gain.shape == state_a.f.shape
        assert result.external_flux_b.gain.shape == state_b.f.shape
        assert not np.any(result.external_flux_a.gain)
        assert not np.any(result.external_flux_a.loss_rate)
        assert not np.any(result.external_flux_b.gain)
        assert not np.any(result.external_flux_b.loss_rate)

    def test_evaluate_rejects_mismatched_E_grids(self) -> None:
        state_L = _build_state(T_bath=0.1, num_energy=20)
        state_R = _build_state(T_bath=0.1, num_energy=25)  # different
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )
        with pytest.raises(ValueError, match="matching E grids"):
            junction.evaluate(state_L, state_R)

    def test_evaluate_rejects_state_gap_spectral_mismatch(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        state_b.gap *= 0.95
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )

        with pytest.raises(ValueError, match="public gap to match its spectral gap"):
            junction.evaluate(state_a, state_b)

    def test_evaluate_rejects_mismatched_cell_widths(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        state_b.spectral = SpectralContext(
            E_bins=state_b.spectral.E,
            dE_bins=1.01 * state_b.spectral.dE,
            gap=state_b.gap,
        )
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )

        with pytest.raises(ValueError, match="cell widths"):
            junction.evaluate(state_a, state_b)

    def test_evaluate_rejects_mismatched_dynes_measure(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        state_b.spectral = SpectralContext(
            E_bins=state_b.spectral.E,
            dE_bins=state_b.spectral.dE,
            gap=state_b.gap,
            dynes_gamma=0.1,
        )
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )

        with pytest.raises(ValueError, match="identical Dynes broadening"):
            junction.evaluate(state_a, state_b)

    def test_evaluate_rejects_mismatched_gaps(self) -> None:
        # The "symmetric gap" name is a contract — different gaps must
        # raise rather than silently produce wrong physics.
        state_L = _build_state(T_bath=0.1, num_energy=20)
        # Manually craft a coherent state with a different gap on the same E
        # centers; the cross-region symmetric-gap contract must reject it.
        gap_b = state_L.gap * 1.05
        spectral_b = SpectralContext(
            E_bins=state_L.spectral.E,
            dE_bins=state_L.spectral.dE,
            gap=gap_b,
        )
        state_R_diff_gap = T3DiffusionState(
            f=state_L.f,
            gap=gap_b,
            spectral=spectral_b,
            phonon=state_L.phonon,
            material=state_L.material,
            T_bath=state_L.T_bath,
        )
        junction = SymmetricGapTunnelingJunction(
            name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
        )
        with pytest.raises(ValueError, match="matched gaps"):
            junction.evaluate(state_L, state_R_diff_gap)


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

        sol = solve_device_steady_state(device)
        assert isinstance(sol, DeviceSolution)

        # SCALE-AWARE assertions (2026-07-20 review): at 100 mK the whole
        # occupation signal is ~8e-10, so the former absolute atol=1e-8/1e-9
        # gates were larger than the signal and passed for ANY state —
        # including a 50%-wrong region swap the undamped outer loop used to
        # certify. Compare relatively on the resolved head of the
        # distribution and absolutely only far below the signal.
        f_L_final = sol.states["L"].f
        f_R_final = sol.states["R"].f
        E = sol.states["L"].spectral.E
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        signal = float(np.max(f_FD))
        assert sol.final_max_delta_f < 1e-6 * signal

        head = f_FD > 1e-16
        assert head.any()
        np.testing.assert_allclose(f_L_final[head], f_R_final[head], rtol=1e-6)
        np.testing.assert_allclose(f_L_final[head], f_FD[head], rtol=1e-3)
        np.testing.assert_allclose(f_L_final, f_FD, atol=1e-3 * signal)

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
    """At T_L ≠ T_R, junction transfer can dominate the cold region.

    The attainable frozen-flux Picard state and its explicit number-balance
    floor are tested below; this is not an independent-bath-equilibrium case.
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
                    # Small absolutely, but dominant over exponentially slow
                    # cold-side electron-phonon relaxation.
                    alpha_per_ns=1e-4,
                ),
            ],
        )
        # Mismatched T has a genuinely slow outer mode: under the
        # scale-aware relative certification (2026-07-20 review) the
        # damped Picard needs ~1200 iterations (~5 s) to drain it to
        # 1e-5 relative. The former absolute 1e-5 gate certified after
        # far fewer iterations at up to ~30% relative slack on the warm
        # region. Outer Anderson acceleration remains a Phase 4+
        # improvement.
        # This splitting stalls at a real pair-number backward-error floor
        # around 3.8e-5. Ask for an achievable accuracy here, then verify a
        # stricter request fails rather than returning the same looser state.
        sol = solve_device_steady_state(
            device, outer_tol=5e-5, outer_max_iter=2000,
        )
        assert sol.final_max_number_backward_error <= 5e-5
        assert sol.final_max_number_backward_error > 1e-5

        strict_device = Device(
            regions={
                name: Region(name=name, state=state)
                for name, state in sol.states.items()
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R",
                    alpha_per_ns=1e-4,
                ),
            ],
        )
        with pytest.raises(RuntimeError, match="conserved-QP-number"):
            solve_device_steady_state(
                strict_device, outer_tol=1e-5, outer_max_iter=2,
            )

        # Both regions reach valid distributions in [0, 1].
        for name in ("L", "R"):
            f = sol.states[name].f
            assert np.all(f >= 0.0)
            assert np.all(f <= 1.0)

        # The former atol=1e-4 gate exceeded both signals (max f:
        # L ~8e-6, R bath ~8e-10) and asserted "each region stays close
        # to its own bath FD" — which is NOT the physics here: at
        # 100 mK the cold region's e-ph relaxation is exponentially
        # slow, so alpha=1e-4/ns tunneling DOMINATES it and the regions
        # nearly equilibrate through the junction. Measured fixed
        # point: L depressed 27-47% below its bath FD on the head; R
        # elevated ~1e5x above its own bath FD, tracking L. Assert that
        # real structure (2026-07-20 review).
        E = sol.states["L"].spectral.E
        f_FD_L = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * T_L), 500.0)) + 1.0)
        f_FD_R = 1.0 / (np.exp(np.minimum(E / (KB_UEV_PER_K * T_R), 500.0)) + 1.0)
        head_L = f_FD_L > 1e-16
        f_L = sol.states["L"].f
        f_R = sol.states["R"].f
        # L: depleted below its bath FD by the drain into R, but within
        # a factor of 2.
        assert np.all(f_L[head_L] < f_FD_L[head_L])
        assert np.all(f_L[head_L] > 0.5 * f_FD_L[head_L])
        # R: far above its own bath FD (junction-dominated; ~1e4x at the
        # gap edge). Near-equilibration with L holds only at the gap
        # edge — at higher E the e-ph rates grow (Kaplan) and R's f
        # decays away from L's (measured ratio 1.02 at the edge falling
        # to ~1e-3 at the top of L's head).
        assert np.all(f_R[head_L] > f_FD_R[head_L])
        assert float(np.max(f_R)) > 1e3 * float(np.max(f_FD_R))
        edge_ratio = float(f_R[head_L][0] / f_L[head_L][0])
        assert 1.0 / 3.0 < edge_ratio < 3.0


# ═══════════════════════════════════════════════════════════════════════
#  Forwarding regression: junction flux must actually reach the solver
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _ConstantFluxJunction(Junction):
    """Test-only Junction that injects a known constant ExternalFlux
    into ``region_a`` and zero into ``region_b``. Decouples the forward-
    ing assertion from any state-dependent physics.
    """

    name: str
    region_a: str
    region_b: str
    flux_a: ExternalFlux
    prescribed_region_flux: ClassVar[bool] = True

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
        qubit_state=None,
    ) -> JunctionResult:
        del qubit_state  # not used
        zero_b = ExternalFlux.zero(int(state_b.spectral.E.size))
        return JunctionResult(
            external_flux_a=self.flux_a,
            external_flux_b=zero_b,
        )


@dataclass
class _UnknownFluxJunction(Junction):
    """No declared number-mode contract; zero output alone is harmless."""

    name: str
    region_a: str
    region_b: str
    gain_per_ns: float = 0.0

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
        qubit_state=None,
    ) -> JunctionResult:
        del qubit_state
        gain_a = np.full_like(state_a.f, self.gain_per_ns)
        return JunctionResult(
            external_flux_a=ExternalFlux(
                gain=gain_a,
                loss_rate=np.zeros_like(state_a.f),
            ),
            external_flux_b=ExternalFlux.zero(state_b.f.size),
        )


@dataclass
class _DeclaredConservativeJunction(Junction):
    """Test-only conservative edge exercising the public base contract."""

    name: str
    region_a: str
    region_b: str
    alpha_per_ns: float = 0.01
    capacity_ratio_a_to_b: float = 1.0
    violate_contract: bool = False
    overflow_contract: bool = False
    relative_contract_error: float = 0.0

    def qp_number_capacity_ratio_a_to_b(self) -> float | None:
        if self.alpha_per_ns == 0.0:
            return None
        return self.capacity_ratio_a_to_b

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
        qubit_state=None,
    ) -> JunctionResult:
        del qubit_state
        if self.overflow_contract:
            return JunctionResult(
                external_flux_a=ExternalFlux(
                    gain=np.full_like(state_a.f, 1e308),
                    loss_rate=np.zeros_like(state_a.f),
                ),
                external_flux_b=ExternalFlux.zero(state_b.f.size),
            )
        alpha_a = self.alpha_per_ns
        alpha_b = alpha_a * self.capacity_ratio_a_to_b
        gain_b = alpha_b * state_a.f
        if self.violate_contract:
            gain_b = np.zeros_like(gain_b)
        else:
            gain_b = gain_b * (1.0 + self.relative_contract_error)
        return JunctionResult(
            external_flux_a=ExternalFlux(
                gain=alpha_a * state_b.f,
                loss_rate=np.full_like(state_a.f, alpha_a),
            ),
            external_flux_b=ExternalFlux(
                gain=gain_b,
                loss_rate=np.full_like(state_b.f, alpha_b),
            ),
        )


class _EchoBackend:
    """Return each state unchanged while recording Device forwarding."""

    def __init__(self) -> None:
        self.use_thermal_phonons: list[bool] = []
        self.conservative_transfer: list[bool] = []

    def steady_state(self, state: T3DiffusionState, **kwargs):
        self.use_thermal_phonons.append(bool(kwargs["use_thermal_phonons"]))
        self.conservative_transfer.append(
            bool(kwargs["external_flux_is_conservative_transfer"])
        )
        return state


class _NonlinearMapBackend:
    """A map whose defect grows sharply after an unverified damped step."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        mapped = np.clip(state.f + 0.004 + 100.0 * (state.f - 0.5), 0.0, 1.0)
        return replace(state, f=mapped)


class _NonFiniteBackend:
    """Backend contract violator used to guard Device acceptance."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        return replace(state, f=np.full_like(state.f, np.nan))


class _InPlaceBackend:
    """Illegal mutating map that used to erase its own Device defect."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        state.f[:] = np.clip(state.f + 0.1, 0.0, 1.0)
        return state


class _InvalidAncillaryBackend:
    """Return a valid occupation with corrupt backend-owned state fields."""

    def __init__(self, field_name: str) -> None:
        self.field_name = field_name

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        if self.field_name == "gap":
            return replace(state, gap=float("nan"))
        if self.field_name == "T_bath":
            return replace(state, T_bath=float("nan"))
        if self.field_name == "phonon":
            bad_phonon = replace(state.phonon, n_ph=state.phonon.n_ph.copy())
            bad_phonon.n_ph[0, 0, 0] = np.nan
            return replace(state, phonon=bad_phonon)
        raise AssertionError(self.field_name)


class _TinyImaginaryBackend:
    """Return a dynamic state with a tiny imaginary component."""

    def __init__(self, field_name: str, imaginary: float) -> None:
        self.field_name = field_name
        self.imaginary = imaginary

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        if self.field_name == "f":
            bad_f = state.f.astype(complex)
            bad_f.imag[:] = self.imaginary
            return replace(state, f=bad_f)
        if self.field_name == "phonon.n_ph":
            bad_phonon = copy.deepcopy(state.phonon)
            bad_phonon.n_ph = bad_phonon.n_ph.astype(complex)
            bad_phonon.n_ph.imag[:] = self.imaginary
            return replace(state, phonon=bad_phonon)
        raise AssertionError(self.field_name)


class _SpectralCacheCorruptingBackend:
    """Corrupt a derived SpectralContext cache in a returned or input state."""

    def __init__(self, cache_name: str, *, in_place: bool = False) -> None:
        self.cache_name = cache_name
        self.in_place = in_place

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        target = state if self.in_place else copy.deepcopy(state)
        spectral = target.spectral
        cached = np.asarray(getattr(spectral, self.cache_name)).copy()
        cached.flat[0] = cached.flat[0] + 1
        setattr(spectral, self.cache_name, cached)
        return target


class _TogglePhononBackend:
    """Two-cycle in n_ph with an exact midpoint fixed point."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        mapped_phonon = replace(
            state.phonon,
            n_ph=1.0 - state.phonon.n_ph,
        )
        return replace(state, phonon=mapped_phonon)


class _StaticTemperatureChangingBackend:
    """Valid-looking candidate outside the Device outer map's state space."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        return replace(state, T_bath=state.T_bath + 0.01)


class _ValueCopyBackend:
    """Pure extension that returns an equivalent, non-aliased state."""

    def steady_state(self, state: T3DiffusionState, **kwargs):
        del kwargs
        return copy.deepcopy(state)


@dataclass
class _MutatingInputJunction(Junction):
    """Illegal extension that mutates referenced region/qubit state."""

    name: str
    region_a: str
    region_b: str
    mutate_qubit: bool = False

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
        qubit_state=None,
    ) -> JunctionResult:
        if self.mutate_qubit:
            assert qubit_state is not None
            qubit_state.p[...] = 0.0
            qubit_state.p.flat[0] = 1.0
        else:
            state_a.f[:] = np.clip(state_a.f + 0.1, 0.0, 1.0)
        return JunctionResult(
            external_flux_a=ExternalFlux.zero(state_a.f.size),
            external_flux_b=ExternalFlux.zero(state_b.f.size),
        )


class TestSolverForwardsJunctionFlux:
    """Tests that would FAIL if ``solve_device_steady_state`` stopped
    passing the aggregated ``ExternalFlux`` into ``backend.steady_state``.
    The detailed-balance and matched-bath tests above pass even if the
    flux is dropped on the floor (each region's inner Newton finds its
    own thermal fixed point regardless). These tests use a custom
    Junction that injects a known constant flux and verify that the
    converged ``f`` is materially different from the no-flux baseline.
    """

    def test_constant_injection_perturbs_steady_state(self) -> None:
        T_bath = 0.1
        state_L = _build_state(T_bath=T_bath, num_energy=30)
        state_R = _build_state(T_bath=T_bath, num_energy=30)
        NE = state_L.spectral.E.size
        # gain pushes f UP, loss_rate is balanced so the equilibrium
        # shifts noticeably above bath FD.
        injection = ExternalFlux(
            gain=np.full(NE, 1e-4), loss_rate=np.full(NE, 1e-2),
        )

        # With injection junction.
        device_with = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                _ConstantFluxJunction(
                    name="inj", region_a="L", region_b="R", flux_a=injection,
                ),
            ],
        )
        sol_with = solve_device_steady_state(device_with, outer_tol=1e-9)

        # Without any junction (baseline).
        device_no = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[],
        )
        sol_no = solve_device_steady_state(device_no, outer_tol=1e-9)

        # If the solver were dropping external_flux, sol_with["L"].f
        # would equal sol_no["L"].f. They must differ materially.
        diff_L = float(np.max(np.abs(sol_with.states["L"].f - sol_no.states["L"].f)))
        assert diff_L > 1e-4, (
            f"Junction flux was not forwarded to the inner solver — "
            f"with-injection and no-junction states match to {diff_L:.2e}."
        )

        # Region R got zero injection from the junction → unchanged.
        diff_R = float(np.max(np.abs(sol_with.states["R"].f - sol_no.states["R"].f)))
        assert diff_R < 1e-10

    def test_zero_alpha_symmetric_junction_matches_no_junction(self) -> None:
        # SymmetricGapTunnelingJunction(alpha=0) should produce zero
        # ExternalFlux on both regions, so the device-with-junction
        # result should match the no-junction baseline.
        T_bath = 0.1
        state_L = _build_state(T_bath=T_bath, num_energy=30)
        state_R = _build_state(T_bath=T_bath, num_energy=30)

        device_alpha0 = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="J", region_a="L", region_b="R", alpha_per_ns=0.0,
                ),
            ],
        )
        device_no_j = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[],
        )
        sol_alpha0 = solve_device_steady_state(device_alpha0, outer_tol=1e-10)
        sol_no_j = solve_device_steady_state(device_no_j, outer_tol=1e-10)
        np.testing.assert_allclose(
            sol_alpha0.states["L"].f, sol_no_j.states["L"].f, atol=1e-12,
        )
        np.testing.assert_allclose(
            sol_alpha0.states["R"].f, sol_no_j.states["R"].f, atol=1e-12,
        )


class TestCommonModeCertification:
    """2026-07-20 round-4 review: two regions at the SAME c*f_FD (zero
    antisymmetric part) previously certified unchanged for any c — the
    inner Newton's backward error was normalized by the large but exactly
    balanced junction exchange, and the absolute residual tolerance is
    below the cold collision scale. The certification now normalizes by
    INTERNAL turnover only, so the common mode must genuinely relax."""

    @pytest.mark.parametrize("c", [0.5, 10.0])
    def test_common_mode_relaxes_to_thermal(self, c: float) -> None:
        T_bath = 0.1
        states = {}
        for name in ("L", "R"):
            s = _build_state(T_bath=T_bath, num_energy=30)
            states[name] = T3DiffusionState(
                f=np.clip(s.f * c, 0.0, 1.0),
                gap=s.gap, spectral=s.spectral,
                phonon=s.phonon, material=s.material,
                T_bath=T_bath,
            )
        device = Device(
            regions={n: Region(name=n, state=st) for n, st in states.items()},
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R", alpha_per_ns=0.01,
                ),
            ],
        )
        # Pre-fix: returned c*f_FD unchanged (50%-1000% wrong) with defect
        # exactly zero. The frozen-flux Picard splitting genuinely CANNOT
        # drain this exchange-dominated common mode (drainage ~
        # collision/exchange per iteration ~ 1e-7 here), so the honest
        # post-fix behavior is a loud refusal via the global
        # conserved-mode certificate — never a silently wrong answer.
        with pytest.raises(RuntimeError, match=r"conserved-mode|common-mode"):
            solve_device_steady_state(device)


class TestControlValidation:
    def test_device_requires_at_least_one_region(self) -> None:
        with pytest.raises(ValueError, match="at least one region"):
            Device(regions={})

    @pytest.mark.parametrize(
        "bad_mode",
        ("false", float("nan"), 1 + 1j, 0, 1),
    )
    def test_rejects_non_boolean_phonon_mode(self, bad_mode: object) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"main": Region(name="main", state=state)},
            junctions=[],
        )
        with pytest.raises(ValueError, match="use_thermal_phonons"):
            solve_device_steady_state(
                device,
                use_thermal_phonons=bad_mode,  # type: ignore[arg-type]
            )

    def test_rejects_nonfinite_tol_and_bad_damping(self) -> None:
        state_L = _build_state(T_bath=0.1, num_energy=20)
        state_R = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R", alpha_per_ns=0.01,
                ),
            ],
        )
        for bad_tol in (
            float("nan"), float("inf"), 0.0, -1e-6, True, np.bool_(True),
            "1e-6", 1e-6j,
        ):
            with pytest.raises(ValueError, match="outer_tol"):
                solve_device_steady_state(device, outer_tol=bad_tol)
        for bad_damp in (
            float("nan"), 0.0, -0.5, 1.5, True, np.bool_(True),
            "0.5", 0.5j,
        ):
            with pytest.raises(ValueError, match="outer_damping"):
                solve_device_steady_state(device, outer_damping=bad_damp)
        for bad_max_iter in (0, True, np.bool_(True)):
            with pytest.raises(ValueError, match="outer_max_iter"):
                solve_device_steady_state(device, outer_max_iter=bad_max_iter)

    def test_legacy_positional_device_solution_layout_is_preserved(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        solution = DeviceSolution({"L": state}, 2, 3.0, None, 4.0, 5.0)
        assert solution.states == {"L": state}
        assert solution.n_outer_iterations == 2
        assert solution.final_max_delta_f == 3.0
        assert solution.qubit_state is None
        assert solution.final_max_delta_p == 4.0
        assert solution.final_max_number_backward_error == 5.0
        assert solution.final_max_delta_n_ph == 0.0

    def test_returned_state_is_the_snapshot_whose_defect_was_measured(
        self,
    ) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        state = replace(state, f=np.full_like(state.f, 0.5))
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )
        backend = _NonlinearMapBackend()

        solution = solve_device_steady_state(
            device,
            backend=backend,
            outer_tol=0.01,
            outer_damping=0.5,
        )

        returned = solution.states["L"]
        mapped = backend.steady_state(returned)
        actual_defect = float(np.max(np.abs(mapped.f - returned.f)))
        assert actual_defect == pytest.approx(solution.final_max_delta_f)
        assert actual_defect <= 0.01 * float(np.max(np.abs(mapped.f)))

    def test_nonfinite_backend_state_fails_before_defect_arithmetic(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match="non-finite or out-of-domain"):
            solve_device_steady_state(device, backend=_NonFiniteBackend())

    @pytest.mark.parametrize("field_name", ["f", "phonon.n_ph"])
    @pytest.mark.parametrize("imaginary", [1e-20, np.nan])
    def test_backend_complex_dynamic_state_fails_before_float_cast(
        self,
        field_name: str,
        imaginary: float,
    ) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(
            RuntimeError, match=r"numeric and complete|phonon state"
        ):
            solve_device_steady_state(
                device,
                backend=_TinyImaginaryBackend(field_name, imaginary),
                outer_tol=1e-6,
            )

    @pytest.mark.parametrize("cache_name", ["_cell_weights", "_K_plus", "_D_E"])
    def test_backend_may_not_return_corrupt_derived_spectral_cache(
        self,
        cache_name: str,
    ) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match="changed static T3 fields"):
            solve_device_steady_state(
                device,
                backend=_SpectralCacheCorruptingBackend(cache_name),
            )

    def test_backend_may_not_mutate_derived_spectral_cache_in_place(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match=r"mutated.*input state"):
            solve_device_steady_state(
                device,
                backend=_SpectralCacheCorruptingBackend(
                    "_cell_weights", in_place=True
                ),
            )

    def test_backend_may_not_mutate_device_input_state_in_place(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match=r"mutated.*input state"):
            solve_device_steady_state(
                device,
                backend=_InPlaceBackend(),
                outer_tol=1e-12,
            )

    @pytest.mark.parametrize("field_name", ["gap", "T_bath", "phonon"])
    def test_backend_candidate_ancillary_state_is_validated(
        self,
        field_name: str,
    ) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match=field_name.replace("_", ".") + "|phonon"):
            solve_device_steady_state(
                device,
                backend=_InvalidAncillaryBackend(field_name),
            )

    def test_quiet_initial_f_does_not_return_stale_phonon_state(self) -> None:
        state = _build_state(T_bath=0.3, num_energy=20)
        backend = T3DiffusionBackend()
        expected = backend.steady_state(state, use_thermal_phonons=True)
        assert not np.any(state.phonon.n_ph)
        assert np.any(expected.phonon.n_ph > 0.0)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        solution = solve_device_steady_state(
            device,
            backend=backend,
            outer_tol=1e-6,
        )

        np.testing.assert_array_equal(
            solution.states["L"].phonon.n_ph,
            expected.phonon.n_ph,
        )

    def test_phonon_two_cycle_is_damped_and_certified(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        assert not np.any(state.phonon.n_ph)
        backend = _TogglePhononBackend()
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        solution = solve_device_steady_state(
            device,
            backend=backend,
            outer_tol=1e-12,
            outer_damping=0.5,
        )
        returned = solution.states["L"]
        mapped = backend.steady_state(returned)

        np.testing.assert_array_equal(returned.phonon.n_ph, 0.5)
        np.testing.assert_array_equal(
            mapped.phonon.n_ph,
            returned.phonon.n_ph,
        )
        assert solution.final_max_delta_n_ph == 0.0

    def test_backend_may_not_change_valid_static_t3_fields(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        with pytest.raises(RuntimeError, match="changed static T3 fields"):
            solve_device_steady_state(
                device,
                backend=_StaticTemperatureChangingBackend(),
            )

    def test_value_equivalent_deep_copy_is_not_a_static_field_change(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"L": Region(name="L", state=state)},
            junctions=[],
        )

        solution = solve_device_steady_state(
            device,
            backend=_ValueCopyBackend(),
            outer_tol=1e-12,
        )

        returned = solution.states["L"]
        assert returned is not state
        assert returned.spectral is not state.spectral
        assert returned.material is not state.material
        np.testing.assert_array_equal(returned.f, state.f)
        np.testing.assert_array_equal(returned.phonon.n_ph, state.phonon.n_ph)

    def test_junction_may_not_mutate_region_input_in_place(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={
                "L": Region(name="L", state=state_a),
                "R": Region(name="R", state=state_b),
            },
            junctions=[
                _MutatingInputJunction(
                    name="mutator",
                    region_a="L",
                    region_b="R",
                )
            ],
        )

        with pytest.raises(RuntimeError, match=r"Junction.*mutated.*input"):
            solve_device_steady_state(device, backend=_EchoBackend())

    def test_junction_may_not_mutate_qubit_input_in_place(self) -> None:
        state_a = _build_state(T_bath=0.1, num_energy=20)
        state_b = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={
                "L": Region(name="L", state=state_a),
                "R": Region(name="R", state=state_b),
            },
            junctions=[
                _MutatingInputJunction(
                    name="qubit-mutator",
                    region_a="L",
                    region_b="R",
                    mutate_qubit=True,
                )
            ],
            qubit=Qubit(
                n_levels=2,
                track_parity=False,
                omega_kelvin=np.array([0.0, 1.0]),
            ),
        )

        with pytest.raises(RuntimeError, match=r"Junction.*mutated.*input"):
            solve_device_steady_state(device, backend=_EchoBackend())


class TestConservedModeCertificateRound5:
    """2026-07-21 round-5 review counterexamples: threshold scaling,
    floating-point absorption at 50 mK, disconnected-component
    cancellation, and same-family misconfiguration refusal."""

    def _pair(self, T_bath: float, c_L: float, c_R: float, *, alpha=0.01):
        states = {}
        for name, c in (("L", c_L), ("R", c_R)):
            s = _build_state(T_bath=T_bath, num_energy=30)
            states[name] = T3DiffusionState(
                f=np.clip(s.f * c, 0.0, 1.0), gap=s.gap, spectral=s.spectral,
                phonon=s.phonon, material=s.material, T_bath=T_bath,
            )
        return Device(
            regions={n: Region(name=n, state=st) for n, st in states.items()},
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R", alpha_per_ns=alpha,
                ),
            ],
        )

    def test_cold_50mK_manifold_refused_not_absorbed(self) -> None:
        # Round-5 hole (b): at 50 mK the collision residual (~1e-38) was
        # absorbed by the balanced junction terms (~1e-12) in floating
        # point; 0.5x/2x/10x f_FD all certified with defect zero. The
        # collision-only per-component sum keeps the signal.
        for c in (0.5, 2.0):
            device = self._pair(0.05, c, c)
            with pytest.raises(RuntimeError, match=r"conserved-QP-number|common-mode"):
                solve_device_steady_state(device)

    def test_five_percent_common_mode_refused_at_tight_tol(self) -> None:
        # Round-5 hole (a): a fixed 0.05 detector threshold accepted
        # ~5%-off common modes at any outer_tol. The advertised outer_tol
        # is now the actual component-certificate threshold.
        device = self._pair(0.1, 1.05, 1.05)
        with pytest.raises(RuntimeError, match=r"conserved-QP-number|common-mode"):
            solve_device_steady_state(device, outer_tol=1e-9, outer_max_iter=50)

    def test_disconnected_components_cannot_cancel(self) -> None:
        # Round-5 hole (c): one signed device-wide sum let two
        # disconnected components cancel (0.5x against ~1.32x summed to
        # ~4e-30). Components are now certified independently.
        states = {}
        for name, c in (("A1", 0.5), ("A2", 0.5), ("B1", 2.0), ("B2", 2.0)):
            s = _build_state(T_bath=0.1, num_energy=30)
            states[name] = T3DiffusionState(
                f=np.clip(s.f * c, 0.0, 1.0), gap=s.gap, spectral=s.spectral,
                phonon=s.phonon, material=s.material, T_bath=0.1,
            )
        device = Device(
            regions={n: Region(name=n, state=st) for n, st in states.items()},
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JA", region_a="A1", region_b="A2", alpha_per_ns=0.01,
                ),
                SymmetricGapTunnelingJunction(
                    name="JB", region_a="B1", region_b="B2", alpha_per_ns=0.01,
                ),
            ],
        )
        with pytest.raises(RuntimeError, match=r"conserved-QP-number|common-mode"):
            solve_device_steady_state(device)

    def test_unequal_capacity_ratio_is_weighted_and_certified(self) -> None:
        # C_a/C_b is part of the Junction's public conservation contract.
        # The Device certificate propagates that weight instead of rejecting
        # an otherwise supported unequal-volume device.
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R", alpha_per_ns=0.01,
                    capacity_ratio_a_to_b=2.0,
                ),
            ],
        )
        sol = solve_device_steady_state(device, outer_tol=1e-8)
        np.testing.assert_allclose(sol.states["L"].f, s1.f, rtol=1e-6)
        np.testing.assert_allclose(sol.states["R"].f, s2.f, rtol=1e-6)

    def test_capacity_cycle_consistency_honors_tighter_outer_tol(self) -> None:
        states = {
            name: _build_state(T_bath=0.1, num_energy=20)
            for name in ("A", "B", "C")
        }
        device = Device(
            regions={
                name: Region(name=name, state=state)
                for name, state in states.items()
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="AB",
                    region_a="A",
                    region_b="B",
                    capacity_ratio_a_to_b=1.0,
                ),
                _DeclaredConservativeJunction(
                    name="BC",
                    region_a="B",
                    region_b="C",
                    capacity_ratio_a_to_b=1.0,
                ),
                _DeclaredConservativeJunction(
                    name="CA",
                    region_a="C",
                    region_b="A",
                    capacity_ratio_a_to_b=1.0 + 5e-13,
                ),
            ],
        )

        # The inconsistency is below the ordinary 1e-12 roundoff ceiling but
        # above the precision the caller explicitly requested.
        with pytest.raises(ValueError, match="inconsistent around a connected cycle"):
            solve_device_steady_state(
                device,
                backend=_EchoBackend(),
                outer_tol=1e-14,
            )

    def test_unequal_capacity_wrong_common_mode_refuses(self) -> None:
        device = self._pair(0.1, 0.5, 0.5)
        device.junctions[0] = SymmetricGapTunnelingJunction(
            name="JJ",
            region_a="L",
            region_b="R",
            alpha_per_ns=0.01,
            capacity_ratio_a_to_b=2.0,
        )
        with pytest.raises(RuntimeError, match=r"conserved-QP-number|common-mode"):
            solve_device_steady_state(device, outer_max_iter=3)

    def test_unrepresentable_capacity_ratio_refuses(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ",
                    region_a="L",
                    region_b="R",
                    alpha_per_ns=0.01,
                    capacity_ratio_a_to_b=np.nextafter(0.0, 1.0),
                ),
            ],
        )
        with pytest.raises(ValueError, match="cannot be represented safely"):
            solve_device_steady_state(device)

    def test_inert_edge_does_not_validate_unused_capacity_ratio(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=20)
        s2 = _build_state(T_bath=0.1, num_energy=25)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="off",
                    region_a="L",
                    region_b="R",
                    alpha_per_ns=0.0,
                    capacity_ratio_a_to_b=np.nextafter(0.0, 1.0),
                ),
            ],
        )
        backend = _EchoBackend()
        sol = solve_device_steady_state(device, backend=backend)
        assert sol.final_max_number_backward_error == 0.0
        # Two regions × (promote exact map output + certify that endpoint).
        assert backend.use_thermal_phonons == [True, True, True, True]

    def test_custom_conservative_contract_is_not_concrete_type_gated(
        self,
    ) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="custom",
                    region_a="L",
                    region_b="R",
                    capacity_ratio_a_to_b=2.0,
                ),
            ],
        )
        sol = solve_device_steady_state(device, outer_tol=1e-8)
        assert sol.final_max_number_backward_error <= 1e-8

    def test_conservative_transfer_normalization_is_explicitly_forwarded(
        self,
    ) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=20)
        s2 = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="custom",
                    region_a="L",
                    region_b="R",
                ),
            ],
        )
        backend = _EchoBackend()

        solve_device_steady_state(device, backend=backend, outer_tol=1e-8)

        assert backend.conservative_transfer
        assert all(backend.conservative_transfer)

    def test_false_conservative_declaration_is_verified(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="dishonest",
                    region_a="L",
                    region_b="R",
                    violate_contract=True,
                ),
            ],
        )
        with pytest.raises(ValueError, match="violates its declared"):
            solve_device_steady_state(device)

    def test_declared_edge_verification_honors_tighter_outer_tol(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="slightly-dishonest",
                    region_a="L",
                    region_b="R",
                    relative_contract_error=5e-13,
                ),
            ],
        )

        # This error is below the ordinary 1e-12 verification ceiling but
        # above the precision the caller explicitly requested.
        with pytest.raises(ValueError, match="violates its declared"):
            solve_device_steady_state(device, outer_tol=1e-14)

    def test_conservative_contract_overflow_cannot_nan_pass(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=30)
        s2 = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                _DeclaredConservativeJunction(
                    name="overflowing",
                    region_a="L",
                    region_b="R",
                    overflow_contract=True,
                ),
            ],
        )

        with pytest.raises(ValueError, match="violates its declared"):
            solve_device_steady_state(device, backend=_EchoBackend())

    def test_finite_phonon_device_mode_refuses_before_solving(self) -> None:
        device = self._pair(0.1, 1.0, 1.0)
        with pytest.raises(ValueError, match="cannot certify finite-phonon"):
            solve_device_steady_state(device, use_thermal_phonons=False)

    def test_finite_phonon_junction_free_wrapper_is_allowed(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        device = Device(
            regions={"main": Region(name="main", state=state)},
            junctions=[],
        )
        backend = _EchoBackend()
        sol = solve_device_steady_state(
            device,
            backend=backend,
            use_thermal_phonons=False,
        )
        np.testing.assert_array_equal(sol.states["main"].f, state.f)
        assert backend.use_thermal_phonons == [False, False]

    def test_finite_phonon_all_inert_edges_are_allowed(self) -> None:
        s1 = _build_state(T_bath=0.1, num_energy=20)
        s2 = _build_state(T_bath=0.1, num_energy=25)
        device = Device(
            regions={
                "L": Region(name="L", state=s1),
                "R": Region(name="R", state=s2),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="off", region_a="L", region_b="R", alpha_per_ns=0.0,
                ),
            ],
        )
        backend = _EchoBackend()
        solve_device_steady_state(
            device,
            backend=backend,
            use_thermal_phonons=False,
        )
        assert backend.use_thermal_phonons == [False, False, False, False]

    def test_exact_zero_temperature_device_vacuum_certifies(self) -> None:
        warm = _build_state(T_bath=0.1, num_energy=20)
        vacuum = replace(warm, f=np.zeros_like(warm.f), T_bath=0.0)
        device = Device(
            regions={
                "L": Region(name="L", state=vacuum),
                "R": Region(name="R", state=replace(vacuum)),
            },
            junctions=[
                SymmetricGapTunnelingJunction(
                    name="JJ", region_a="L", region_b="R", alpha_per_ns=0.01,
                ),
            ],
        )
        sol = solve_device_steady_state(device, outer_tol=1e-10)
        assert not np.any(sol.states["L"].f)
        assert not np.any(sol.states["R"].f)
        assert sol.final_max_number_backward_error == 0.0

    def test_disconnected_unknown_zero_flux_does_not_disable_certificate(
        self,
    ) -> None:
        # A zero-output custom fixture in another component is harmless, but
        # must not switch off the symmetric L/R certificate globally.
        pair = self._pair(0.05, 0.5, 0.5)
        u = _build_state(T_bath=0.1, num_energy=30)
        v = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                **pair.regions,
                "U": Region(name="U", state=u),
                "V": Region(name="V", state=v),
            },
            junctions=[
                *pair.junctions,
                _UnknownFluxJunction(
                    name="unknown-zero", region_a="U", region_b="V",
                ),
            ],
        )
        with pytest.raises(RuntimeError, match=r"conserved-QP-number|common-mode"):
            solve_device_steady_state(device, outer_max_iter=3)

    def test_unknown_nonzero_state_dependent_flux_refuses(self) -> None:
        u = _build_state(T_bath=0.1, num_energy=30)
        v = _build_state(T_bath=0.1, num_energy=30)
        device = Device(
            regions={
                "U": Region(name="U", state=u),
                "V": Region(name="V", state=v),
            },
            junctions=[
                _UnknownFluxJunction(
                    name="unknown", region_a="U", region_b="V",
                    gain_per_ns=1e-8,
                ),
            ],
        )
        with pytest.raises(ValueError, match="no Device number-mode safety contract"):
            solve_device_steady_state(device)
