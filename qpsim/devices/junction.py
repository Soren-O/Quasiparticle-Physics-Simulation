"""Junction — tunnel coupling between two named Regions in a Device.

Each Junction implementation evaluates ``(state_a, state_b)`` of the
two coupled regions and returns a :class:`JunctionResult` with the
per-region :class:`ExternalFlux` contributions to push into each
region's f-equation.

v1 ships one concrete Junction:

* :class:`SymmetricGapTunnelingJunction` — energy-conserving tunneling
  between two regions sharing the same superconducting gap and energy
  grid. No qubit, no photon drive. Reaches detailed balance at matched
  temperature with no drive.

Phase 4 will add ``JunctionQubitCoupling`` + ``QubitTransitionChannel``
to the result for qubit-coupled junctions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from qpsim.devices.external_flux import ExternalFlux

if TYPE_CHECKING:
    from qpsim.backends.t3_diffusion import T3DiffusionState


@dataclass
class JunctionResult:
    """One Junction's contribution to each connected region's f-equation.

    Parameters
    ----------
    external_flux_a, external_flux_b
        ``ExternalFlux`` contributions for region_a and region_b
        respectively. The Device solver sums contributions from all
        Junctions touching a region before passing the aggregate to
        that region's backend step.
    """

    external_flux_a: ExternalFlux
    external_flux_b: ExternalFlux


class Junction(ABC):
    """Abstract base for Junction implementations.

    Subclasses provide a specific physical tunneling model. The
    framework owns the protocol; the physics lives in subclasses.
    """

    name: str
    region_a: str
    region_b: str

    @abstractmethod
    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
    ) -> JunctionResult:
        """Compute (gain, loss_rate) contributions for both regions.

        Implementations must respect the contract from
        ``docs/Device_Architecture.md`` §3.2.1: gain and loss_rate are
        each non-negative; signed extraction is encoded as
        ``loss_rate * f``, never as a negative gain.
        """


@dataclass
class SymmetricGapTunnelingJunction(Junction):
    r"""Energy-conserving tunneling between two regions with matched gaps.

    The simplest physically-meaningful junction: at each energy ``E``,
    quasiparticles tunnel between regions A and B at rate
    ``α(E)`` proportional to ``[f_a(E) - f_b(E)]`` (the chemical-
    potential-difference drive at the bin level). For the v1
    implementation ``α`` is a scalar tunneling rate ``α_per_ns``
    (units 1/ns).

    Decomposition into the ``(gain, loss_rate)`` ``ExternalFlux`` form:

    .. math::

        \text{net } \dot{f_a}(E) = α [f_b(E) - f_a(E)]
                                = α f_b(E) - α f_a(E)

    so

    .. math::

        \text{gain}_a(E) &= α\, f_b(E),\\
        \text{loss\_rate}_a(E) &= α,

    and symmetrically for region B.

    The (1 − f) Pauli-blocking factors that appear in the full
    tunneling matrix element are absorbed into the cross-region
    detailed balance: the gain and loss_rate above are valid in the
    non-degenerate limit ``f_α ≪ 1`` that holds throughout the
    superconducting regime where this framework lives. A more
    complete junction would carry full ``α(E) (1 - f_partner)`` into
    loss_rate; deferred for v1.

    Both regions must share an energy grid. The constructor checks
    this; ``evaluate`` re-checks at runtime.

    Parameters
    ----------
    name
        Junction identifier.
    region_a, region_b
        Names of the two coupled regions in the parent Device.
    alpha_per_ns
        Per-bin tunneling rate ``α`` in 1/ns. Constant in energy for
        v1; a future extension can take an E-resolved array.
    """

    name: str
    region_a: str
    region_b: str
    alpha_per_ns: float

    def __post_init__(self) -> None:
        if self.alpha_per_ns < 0.0:
            raise ValueError(
                f"alpha_per_ns must be non-negative; got {self.alpha_per_ns}"
            )
        if self.region_a == self.region_b:
            raise ValueError(
                f"Junction must couple two different regions; got "
                f"both region_a and region_b = {self.region_a!r}."
            )

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
    ) -> JunctionResult:
        if state_a.spectral.E.size != state_b.spectral.E.size:
            raise ValueError(
                f"SymmetricGapTunnelingJunction requires matching E grids; "
                f"got {state_a.spectral.E.size} vs {state_b.spectral.E.size}."
            )
        if not np.allclose(state_a.spectral.E, state_b.spectral.E, rtol=1e-12):
            raise ValueError(
                "SymmetricGapTunnelingJunction requires identical E grids "
                "between the two regions; got different values."
            )
        # The "symmetric gap" name is contractual — reject mismatched
        # gaps explicitly so callers don't get silent wrong physics.
        if not np.isclose(state_a.gap, state_b.gap, rtol=1e-12):
            raise ValueError(
                f"SymmetricGapTunnelingJunction requires matched gaps; "
                f"got Δ_a = {state_a.gap}, Δ_b = {state_b.gap}. Use a "
                f"gap-asymmetric Junction subclass for Δ_a ≠ Δ_b."
            )
        if not np.isclose(
            state_a.spectral.gap, state_b.spectral.gap, rtol=1e-12,
        ):
            raise ValueError(
                f"SymmetricGapTunnelingJunction requires matched spectral "
                f"gaps; got {state_a.spectral.gap} vs {state_b.spectral.gap}."
            )

        alpha = float(self.alpha_per_ns)
        f_a = state_a.f
        f_b = state_b.f

        gain_a = alpha * f_b
        loss_a = np.full_like(f_b, alpha)
        gain_b = alpha * f_a
        loss_b = np.full_like(f_a, alpha)

        return JunctionResult(
            external_flux_a=ExternalFlux(
                gain=gain_a,
                loss_rate=loss_a,
                diagnostics={"junction": self.name, "alpha_per_ns": alpha},
            ),
            external_flux_b=ExternalFlux(
                gain=gain_b,
                loss_rate=loss_b,
                diagnostics={"junction": self.name, "alpha_per_ns": alpha},
            ),
        )
