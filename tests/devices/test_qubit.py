"""Tests for Qubit, QubitState, QubitTransitionChannel, JunctionQubitCoupling,
and the qubit master-equation evolver.

Phase 4 of the Device Architecture: a discrete two-level (or more)
system optionally coupled to junction tunneling events. The contract
tests below pin:

* Dataclass validation (Qubit, QubitState, QubitTransitionChannel,
  JunctionQubitCoupling).
* Master-equation rate-matrix structure (off-diagonals = incoming,
  diagonals = -outgoing, parity-flipping channels advance the parity
  axis).
* Steady-state solver: detailed-balance Boltzmann ratio is recovered
  when channel rates obey ``Γ_↑/Γ_↓ = exp(-ω_10/T)``.
* Trivial limits: relaxation-only channels give p_1 = 0; matched
  rates give equipartition.
* Junction integration: a Junction with qubit_coupling emits the
  right channels; the device solver pools them and converges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import (
    Device,
    ExternalFlux,
    Junction,
    JunctionQubitCoupling,
    JunctionResult,
    Qubit,
    QubitState,
    QubitTransitionChannel,
    Region,
    SymmetricGapTunnelingJunction,
    build_rate_matrix,
    solve_device_steady_state,
    solve_qubit_master_equation_steady_state,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.spectral import SpectralContext

# ═══════════════════════════════════════════════════════════════════════
#  Dataclass validation
# ═══════════════════════════════════════════════════════════════════════


class TestQubitValidation:
    def test_default_two_level(self) -> None:
        q = Qubit()
        assert q.n_levels == 2
        assert q.track_parity is True
        assert q.omega_kelvin.shape == (2,)

    def test_rejects_zero_levels(self) -> None:
        with pytest.raises(ValueError, match="n_levels must be"):
            Qubit(n_levels=0)

    def test_rejects_omega_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="omega_kelvin must have shape"):
            Qubit(n_levels=2, omega_kelvin=np.array([0.0, 1.0, 2.0]))

    def test_rejects_non_monotone_omega(self) -> None:
        with pytest.raises(ValueError, match="non-decreasing"):
            Qubit(n_levels=2, omega_kelvin=np.array([1.0, 0.0]))


class TestQubitStateValidation:
    def test_2d_with_parity(self) -> None:
        s = QubitState(p=np.array([[0.6, 0.1], [0.2, 0.1]]))
        assert s.p.shape == (2, 2)

    def test_1d_no_parity(self) -> None:
        s = QubitState(p=np.array([0.7, 0.3]))
        assert s.p.shape == (2,)

    def test_rejects_non_normalized(self) -> None:
        with pytest.raises(ValueError, match="must sum to 1"):
            QubitState(p=np.array([0.5, 0.7]))

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QubitState(p=np.array([1.5, -0.5]))

    def test_rejects_3d(self) -> None:
        with pytest.raises(ValueError, match="must be 1D"):
            QubitState(p=np.zeros((2, 2, 2)))


class TestQubitTransitionChannelValidation:
    def test_valid_construction(self) -> None:
        ch = QubitTransitionChannel(
            level_from=1, level_to=0, rate_per_ns=0.05, flips_parity=True,
        )
        assert ch.label == ""

    def test_rejects_negative_rate(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QubitTransitionChannel(
                level_from=0, level_to=1, rate_per_ns=-0.1, flips_parity=False,
            )

    def test_rejects_negative_level(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            QubitTransitionChannel(
                level_from=-1, level_to=0, rate_per_ns=0.1, flips_parity=True,
            )


class TestJunctionQubitCouplingValidation:
    def test_valid_construction(self) -> None:
        cpl = JunctionQubitCoupling(
            sin_matrix_elements=np.array([[0.0, 0.05], [0.05, 0.0]]),
            cos_matrix_elements=np.array([[1.0, 0.0], [0.0, 0.95]]),
        )
        assert cpl.sin_matrix_elements.shape == (2, 2)

    def test_rejects_non_square(self) -> None:
        with pytest.raises(ValueError, match="square 2D array"):
            JunctionQubitCoupling(
                sin_matrix_elements=np.zeros((2, 3)),
                cos_matrix_elements=np.zeros((2, 3)),
            )

    def test_rejects_shape_mismatch(self) -> None:
        with pytest.raises(ValueError, match="must match"):
            JunctionQubitCoupling(
                sin_matrix_elements=np.zeros((2, 2)),
                cos_matrix_elements=np.zeros((3, 3)),
            )

    def test_rejects_nonfinite_parity_preserving_rates(self) -> None:
        # The parity_preserving_rates field should reject inf/nan
        # the same way sin_matrix_elements and cos_matrix_elements do.
        with pytest.raises(ValueError, match="non-finite"):
            JunctionQubitCoupling(
                sin_matrix_elements=np.zeros((2, 2)),
                cos_matrix_elements=np.eye(2),
                parity_preserving_rates=np.array([[0.0, np.inf], [0.0, 0.0]]),
            )


# ═══════════════════════════════════════════════════════════════════════
#  Rate-matrix construction
# ═══════════════════════════════════════════════════════════════════════


class TestBuildRateMatrix:
    def test_parity_flip_channel_off_diagonal(self) -> None:
        # |0,e> → |0,o> at rate 0.1 (level-conserving but parity-flipping).
        # Flat indices: |0,e>=0, |0,o>=1, |1,e>=2, |1,o>=3.
        ch = QubitTransitionChannel(
            level_from=0, level_to=0, rate_per_ns=0.1, flips_parity=True,
        )
        M = build_rate_matrix([ch], n_levels=2, track_parity=True)
        assert M.shape == (4, 4)
        # Off-diagonal: M[to=1=|0,o>, from=0=|0,e>] = 0.1
        assert M[1, 0] == 0.1
        # And the symmetric flip: |0,o> → |0,e> at rate 0.1
        assert M[0, 1] == 0.1
        # Diagonal losses
        assert M[0, 0] == -0.1
        assert M[1, 1] == -0.1

    def test_parity_preserving_channel(self) -> None:
        # |1,e> → |0,e> at rate 0.05 (ee channel, no parity flip).
        ch = QubitTransitionChannel(
            level_from=1, level_to=0, rate_per_ns=0.05, flips_parity=False,
        )
        M = build_rate_matrix([ch], n_levels=2, track_parity=True)
        # |1,e>=2 → |0,e>=0
        assert M[0, 2] == 0.05
        # |1,o>=3 → |0,o>=1 (parity preserved separately)
        assert M[1, 3] == 0.05
        # Diagonals
        assert M[2, 2] == -0.05
        assert M[3, 3] == -0.05

    def test_no_parity_axis(self) -> None:
        ch = QubitTransitionChannel(
            level_from=1, level_to=0, rate_per_ns=0.1, flips_parity=False,
        )
        M = build_rate_matrix([ch], n_levels=2, track_parity=False)
        assert M.shape == (2, 2)
        assert M[0, 1] == 0.1
        assert M[1, 1] == -0.1

    def test_rejects_out_of_range_levels(self) -> None:
        ch = QubitTransitionChannel(
            level_from=5, level_to=0, rate_per_ns=0.1, flips_parity=True,
        )
        with pytest.raises(ValueError, match="exceeds n_levels"):
            build_rate_matrix([ch], n_levels=2)


# ═══════════════════════════════════════════════════════════════════════
#  Steady-state master equation: detailed balance
# ═══════════════════════════════════════════════════════════════════════


class TestQubitDetailedBalance:
    """The headline Phase 4 contract: with channel rates obeying
    detailed balance ``Γ_↑/Γ_↓ = exp(-ω_10/T)``, the qubit reaches
    the Boltzmann distribution ``p_1/p_0 = exp(-ω_10/T)``.
    """

    def test_relaxation_only_drives_p1_to_zero(self) -> None:
        # Pure relaxation |1> → |0>, no excitation. Steady state:
        # all population in |0>.
        relax = QubitTransitionChannel(
            level_from=1, level_to=0, rate_per_ns=0.1, flips_parity=False,
        )
        qubit = Qubit(n_levels=2, track_parity=False)
        # Need at least one channel to avoid trivial null space.
        sol = solve_qubit_master_equation_steady_state([relax], qubit)
        # p_0 = 1, p_1 = 0
        assert sol.p[0] == pytest.approx(1.0, abs=1e-12)
        assert sol.p[1] == pytest.approx(0.0, abs=1e-12)

    def test_balanced_rates_give_equipartition(self) -> None:
        # Γ_↑ = Γ_↓ → p_0 = p_1 = 0.5 (infinite-T limit).
        rate = 0.1
        channels = [
            QubitTransitionChannel(
                level_from=1, level_to=0, rate_per_ns=rate, flips_parity=False,
            ),
            QubitTransitionChannel(
                level_from=0, level_to=1, rate_per_ns=rate, flips_parity=False,
            ),
        ]
        qubit = Qubit(n_levels=2, track_parity=False)
        sol = solve_qubit_master_equation_steady_state(channels, qubit)
        np.testing.assert_allclose(sol.p, np.array([0.5, 0.5]), atol=1e-12)

    def test_detailed_balance_recovers_boltzmann(self) -> None:
        # Γ_↑/Γ_↓ = exp(-ω_10/T) → p_1/p_0 = exp(-ω_10/T).
        omega_10_K = 0.5  # K
        T_qubit_K = 0.2   # K
        ratio = float(np.exp(-omega_10_K / T_qubit_K))  # ≈ 0.082

        gamma_down = 0.1   # arbitrary scale
        gamma_up = gamma_down * ratio
        channels = [
            QubitTransitionChannel(
                level_from=1, level_to=0, rate_per_ns=gamma_down,
                flips_parity=False, label="relax",
            ),
            QubitTransitionChannel(
                level_from=0, level_to=1, rate_per_ns=gamma_up,
                flips_parity=False, label="excite",
            ),
        ]
        qubit = Qubit(
            n_levels=2, track_parity=False,
            omega_kelvin=np.array([0.0, omega_10_K]),
        )
        sol = solve_qubit_master_equation_steady_state(channels, qubit)
        assert sol.p[1] / sol.p[0] == pytest.approx(ratio, rel=1e-12)
        # And p_0 + p_1 = 1
        assert sol.p.sum() == pytest.approx(1.0, abs=1e-12)

    def test_detailed_balance_with_parity_axis(self) -> None:
        # Same detailed-balance setup with parity-tracking on. Without
        # any parity-flipping channels, parity becomes an unused
        # symmetry axis: solver should give equipartition over parity
        # within each level.
        omega_10_K = 0.5
        T_qubit_K = 0.2
        ratio = float(np.exp(-omega_10_K / T_qubit_K))

        gamma_down = 0.1
        gamma_up = gamma_down * ratio
        channels = [
            QubitTransitionChannel(
                level_from=1, level_to=0, rate_per_ns=gamma_down,
                flips_parity=False,
            ),
            QubitTransitionChannel(
                level_from=0, level_to=1, rate_per_ns=gamma_up,
                flips_parity=False,
            ),
            # Add a parity-flipping channel to connect the parity axis;
            # without it, the parity sectors are decoupled and the
            # solver picks up an arbitrary parity ratio from the
            # initial-condition projection.
            QubitTransitionChannel(
                level_from=0, level_to=0, rate_per_ns=0.5,
                flips_parity=True, label="parity_flip",
            ),
        ]
        qubit = Qubit(
            n_levels=2, track_parity=True,
            omega_kelvin=np.array([0.0, omega_10_K]),
        )
        sol = solve_qubit_master_equation_steady_state(channels, qubit)

        # p_1/p_0 (sum over parity): Boltzmann ratio.
        p_per_level = sol.p.sum(axis=1)
        assert p_per_level[1] / p_per_level[0] == pytest.approx(ratio, rel=1e-10)
        # Within each level: parity-flip rate >> level transitions, so
        # equipartition over parity.
        np.testing.assert_allclose(
            sol.p[0, 0], sol.p[0, 1], rtol=1e-3,
        )

    def test_no_channels_raises(self) -> None:
        qubit = Qubit(n_levels=2, track_parity=False)
        with pytest.raises(RuntimeError, match="No channels supplied"):
            solve_qubit_master_equation_steady_state([], qubit)


# ═══════════════════════════════════════════════════════════════════════
#  Junction integration: Junction emits qubit_channels, Device pools them
# ═══════════════════════════════════════════════════════════════════════


def _build_state(*, T_bath: float, num_energy: int = 30) -> T3DiffusionState:
    """Al-like T3 state at thermal equilibrium (matches Phase 3 fixture)."""
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


@dataclass
class _QubitBathJunction(Junction):
    """Test-only Junction that emits a fixed pair of qubit channels
    consistent with detailed balance at temperature ``T_qubit_K``,
    plus zero region fluxes. Lets us exercise the Device-level qubit
    plumbing without needing a full M25-style coupling.
    """

    name: str
    region_a: str
    region_b: str
    T_qubit_K: float
    omega_10_K: float
    gamma_down_per_ns: float

    def evaluate(
        self, state_a: T3DiffusionState, state_b: T3DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        del qubit_state
        zero_a = ExternalFlux.zero(int(state_a.spectral.E.size))
        zero_b = ExternalFlux.zero(int(state_b.spectral.E.size))
        ratio = float(np.exp(-self.omega_10_K / self.T_qubit_K))
        channels = [
            QubitTransitionChannel(
                level_from=1, level_to=0,
                rate_per_ns=self.gamma_down_per_ns,
                flips_parity=False, label="bath_relax",
            ),
            QubitTransitionChannel(
                level_from=0, level_to=1,
                rate_per_ns=self.gamma_down_per_ns * ratio,
                flips_parity=False, label="bath_excite",
            ),
        ]
        return JunctionResult(
            external_flux_a=zero_a,
            external_flux_b=zero_b,
            qubit_channels=channels,
        )


class TestDeviceWithQubit:
    def test_device_solves_qubit_alongside_regions(self) -> None:
        # Build a 2-region Device with no actual junction-driven QP
        # transport (the junction returns zero region fluxes), but
        # WITH a qubit_coupling that produces detailed-balance channels
        # at T_qubit. Verify solve_device_steady_state pools the
        # channels and the qubit converges to Boltzmann.
        T_bath = 0.1
        omega_10_K = 0.5
        T_qubit_K = 0.2
        ratio = float(np.exp(-omega_10_K / T_qubit_K))

        device = Device(
            regions={
                "L": Region(name="L", state=_build_state(T_bath=T_bath)),
                "R": Region(name="R", state=_build_state(T_bath=T_bath)),
            },
            junctions=[
                _QubitBathJunction(
                    name="bath",
                    region_a="L", region_b="R",
                    T_qubit_K=T_qubit_K,
                    omega_10_K=omega_10_K,
                    gamma_down_per_ns=0.1,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=False,
                omega_kelvin=np.array([0.0, omega_10_K]),
            ),
        )
        sol = solve_device_steady_state(device, outer_tol=1e-10)
        assert sol.qubit_state is not None
        # Boltzmann ratio. rel=1e-9, not 1e-10: the damped outer loop
        # certifies the fixed-point defect at outer_tol*||p||, leaving a
        # ~defect/damping residual bias (~2e-10 here) in the endpoint.
        p = sol.qubit_state.p
        assert p[1] / p[0] == pytest.approx(ratio, rel=1e-9)
        # Regions should still hit thermal equilibrium (no flux drives them).
        E = sol.states["L"].spectral.E
        kT = KB_UEV_PER_K * T_bath
        f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
        np.testing.assert_allclose(sol.states["L"].f, f_FD, atol=1e-9)

    def test_device_without_qubit_yields_no_qubit_state(self) -> None:
        # Backward compat: Device with no qubit returns
        # qubit_state=None; final_max_delta_p stays 0.
        T_bath = 0.1
        device = Device(
            regions={"L": Region(name="L", state=_build_state(T_bath=T_bath))},
            junctions=[],
            qubit=None,
        )
        sol = solve_device_steady_state(device, outer_tol=1e-10)
        assert sol.qubit_state is None
        assert sol.final_max_delta_p == 0.0

    def test_qubit_with_no_emitted_channels_raises(self) -> None:
        # A Qubit attached to a Device whose junctions emit no channels
        # would silently return the initial uniform state — masking a
        # miswired coupling. The solver raises a clear error pointing
        # at the two fixes (remove the qubit, or wire a Junction with
        # a qubit_coupling).
        T_bath = 0.1
        device = Device(
            regions={
                "L": Region(name="L", state=_build_state(T_bath=T_bath)),
                "R": Region(name="R", state=_build_state(T_bath=T_bath)),
            },
            junctions=[
                # SymmetricGapTunneling has no qubit_coupling → emits no channels
                SymmetricGapTunnelingJunction(
                    name="J", region_a="L", region_b="R", alpha_per_ns=0.01,
                ),
            ],
            qubit=Qubit(n_levels=2, track_parity=False),
        )
        with pytest.raises(RuntimeError, match="no junction emitted"):
            solve_device_steady_state(device, outer_tol=1e-10)

    def test_qubit_with_no_junctions_at_all_raises(self) -> None:
        # Even simpler miswire: Qubit on a Device with zero junctions.
        T_bath = 0.1
        device = Device(
            regions={"L": Region(name="L", state=_build_state(T_bath=T_bath))},
            junctions=[],
            qubit=Qubit(n_levels=2, track_parity=False),
        )
        with pytest.raises(RuntimeError, match="no junction emitted"):
            solve_device_steady_state(device, outer_tol=1e-10)
