"""Junction — tunnel coupling between two named Regions in a Device.

Each Junction implementation evaluates ``(state_a, state_b)`` of the
two coupled regions and returns a :class:`JunctionResult` with the
per-region :class:`ExternalFlux` contributions to push into each
region's f-equation.

This module ships one concrete Junction:

* :class:`SymmetricGapTunnelingJunction` — energy-conserving tunneling
  between two regions sharing the same superconducting gap and energy
  grid. No qubit, no photon drive. Reaches detailed balance at matched
  temperature with no drive.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.qubit import QubitTransitionChannel

if TYPE_CHECKING:
    from qpsim.backends.diffusion import DiffusionState
    from qpsim.devices.qubit import QubitState


def _real_scalar_control(name: str, value: Any) -> float:
    """Normalize one real scalar parameter without accepting coercion traps."""
    if (
        isinstance(value, (bool, np.bool_, str, bytes))
        or np.iscomplexobj(value)
        or np.asarray(value).ndim != 0
    ):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {value!r}."
        ) from exc
    return normalized


@dataclass
class JunctionResult:
    """One Junction's contribution to each connected region's f-equation
    and (optionally) the qubit master equation.

    Parameters
    ----------
    external_flux_a, external_flux_b
        ``ExternalFlux`` contributions for region_a and region_b
        respectively. The Device solver sums contributions from all
        Junctions touching a region before passing the aggregate to
        that region's backend step.
    qubit_channels
        Optional list of :class:`QubitTransitionChannel` records.
        Empty when the Junction has no qubit coupling. The Device
        solver pools channels from all junctions before invoking the
        qubit master-equation evolver. Each channel is parity-tagged
        (``flips_parity``) so the evolver can advance the parity axis
        correctly.
    """

    external_flux_a: ExternalFlux
    external_flux_b: ExternalFlux
    qubit_channels: list[QubitTransitionChannel] = field(default_factory=list)


class Junction(ABC):
    """Abstract base for Junction implementations.

    Subclasses provide a specific physical tunneling model. The
    framework owns the protocol; the physics lives in subclasses.

    Attributes
    ----------
    owns_region_dissipation
        Class-level flag. Default ``False``: the Junction provides
        only a tunneling/coupling flux on top of the region's own
        e-ph collision kernel. ``True``: the Junction's external
        flux already includes the moment-integrated dissipation
        physics (e.g. M25's ``r_α x_α²`` recombination and ``g_α``
        thermal-phonon generation), so the device solver MUST
        disable the e-ph kernel on the touched regions to avoid
        double-counting. At most one such Junction may touch any
        region; the device solver enforces this.
    requires_exclusive_regions
        Class-level flag. Default ``False``. ``True`` means this Junction's
        closure is only valid when each touched region belongs to this
        Junction alone. The :class:`Device` constructor rejects any other
        Junction sharing either region.
    prescribed_region_flux
        Class-level safety contract for custom Junctions. ``True`` means
        each emitted region flux is a prescribed local source/sink rather
        than a state-dependent exchange between the two regions. Such a
        flux is certified by each region's Newton number-mode check. The
        conservative Device component certificate cannot infer this from
        arbitrary ``evaluate`` code, so the default is ``False``.
    """

    name: str
    region_a: str
    region_b: str
    owns_region_dissipation: bool = False
    requires_exclusive_regions: bool = False
    prescribed_region_flux: ClassVar[bool] = False

    def qp_number_capacity_ratio_a_to_b(self) -> float | None:
        """Describe an active conservative QP-transfer edge, if any.

        Return ``C_a / C_b`` when this Junction is configured to transfer
        quasiparticles conservatively between its two regions, where the
        conserved discrete population is

        ``C_a * sum(w_a * f_a) + C_b * sum(w_b * f_b)``.

        This is a public safety contract, not a diagnostic hint: it must
        remain present when the instantaneous *net* current happens to
        vanish, and every :meth:`evaluate` result must conserve the stated
        weighted population. The Device solver verifies that identity before
        using the ratio in its connected-component number-mode certificate.

        Return ``None`` for Junctions without such a contract and for a
        coupling that is identically disabled by configuration. The default
        means a nonzero state-dependent flux from an undeclared custom
        Junction is refused by the Device solver.
        """
        return None

    @abstractmethod
    def evaluate(
        self,
        state_a: DiffusionState,
        state_b: DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        """Compute (gain, loss_rate) contributions for both regions
        and (optionally) parity-tagged qubit transition channels.

        Implementations must respect the contract from
        ``docs/Device_Architecture.md`` §3.2.1: gain and loss_rate are
        each non-negative; signed extraction is encoded as
        ``loss_rate * f``, never as a negative gain. Qubit channels
        carry rates in 1/ns and a ``flips_parity`` tag.

        ``qubit_state`` is provided for the small set of Junctions
        whose tunneling rates depend on qubit populations (e.g. via
        Pauli-blocking on the qubit). Most Junctions ignore it.
        """


@dataclass
class SymmetricGapTunnelingJunction(Junction):
    r"""Energy-conserving tunneling between two regions with matched gaps.

    The simplest physically-meaningful junction: at each energy ``E``,
    quasiparticles tunnel between regions A and B in proportion to
    ``[f_a(E) - f_b(E)]`` (the chemical-potential-difference drive at the
    bin level). The rates are scalar in energy.

    Decomposition into the ``(gain, loss_rate)`` ``ExternalFlux`` form:

    .. math::

        \dot{f_a}(E) &= α_a [f_b(E) - f_a(E)],\\
        \dot{f_b}(E) &= α_b [f_a(E) - f_b(E)].

    so

    .. math::

        \text{gain}_a(E) &= α_a\, f_b(E),\\
        \text{loss\_rate}_a(E) &= α_a,

    with the analogous ``α_b`` decomposition for region B.  The rates obey
    ``C_a α_a = C_b α_b``; consequently the weighted population
    ``C_a f_a + C_b f_b`` is conserved even when the two matched-gap
    electrodes have unequal volumes or normal-state densities of states.

    The (1 − f) Pauli-blocking factors that appear in the full
    tunneling matrix element are absorbed into the cross-region
    detailed balance: the gain and loss_rate above are valid in the
    non-degenerate limit ``f_α ≪ 1`` that holds throughout the
    superconducting regime where this framework lives. A more
    complete junction would carry full ``α(E) (1 - f_partner)`` into
    loss_rate.

    Both regions must share the complete finite-volume spectral
    discretization: energy centers, cell widths, gap, broadening model, and
    represented DOS capacities. ``evaluate`` checks this at runtime. A single
    scalar material/volume capacity ratio cannot repair an energy-dependent
    mismatch between the two discrete measures.

    Parameters
    ----------
    name
        Junction identifier.
    region_a, region_b
        Names of the two coupled regions in the parent Device.
    alpha_per_ns
        Per-bin tunneling rate out of region A, ``α_a``, in 1/ns.
        Constant in energy.
    capacity_ratio_a_to_b
        Ratio ``C_a / C_b`` of the two regions' quasiparticle capacities
        (for matched spectra, proportional to ``rho_F * volume``).  Region
        B's rate is ``α_b = α_a C_a/C_b``, so the junction conserves
        ``C_a f_a + C_b f_b``.  The default ``1`` is the
        equal-capacity model.
    """

    name: str
    region_a: str
    region_b: str
    alpha_per_ns: float
    capacity_ratio_a_to_b: float = 1.0

    def __post_init__(self) -> None:
        self.alpha_per_ns = _real_scalar_control(
            "alpha_per_ns", self.alpha_per_ns
        )
        if not np.isfinite(self.alpha_per_ns) or self.alpha_per_ns < 0.0:
            raise ValueError(
                "alpha_per_ns must be finite and non-negative; got "
                f"{self.alpha_per_ns}"
            )
        self.capacity_ratio_a_to_b = _real_scalar_control(
            "capacity_ratio_a_to_b", self.capacity_ratio_a_to_b
        )
        if (
            not np.isfinite(self.capacity_ratio_a_to_b)
            or self.capacity_ratio_a_to_b <= 0.0
        ):
            raise ValueError(
                "capacity_ratio_a_to_b must be finite and positive; got "
                f"{self.capacity_ratio_a_to_b}"
            )
        if self.region_a == self.region_b:
            raise ValueError(
                f"Junction must couple two different regions; got "
                f"both region_a and region_b = {self.region_a!r}."
            )

    def qp_number_capacity_ratio_a_to_b(self) -> float | None:
        """Return this active edge's documented ``C_a/C_b`` ratio.

        ``alpha_per_ns == 0`` is a genuinely disabled coupling, not a graph
        edge. An inert Junction must not merge independent conservation
        components or make its unused capacity ratio part of certification.
        """
        if self.alpha_per_ns == 0.0:
            return None
        return float(self.capacity_ratio_a_to_b)

    def evaluate(
        self,
        state_a: DiffusionState,
        state_b: DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        # qubit_state is unused — this Junction has no qubit coupling.
        del qubit_state
        if self.alpha_per_ns == 0.0:
            # A disabled coupling is a true no-op. Do this before cross-region
            # compatibility checks: no tunneling is evaluated, so unrelated
            # grids/gaps and its unused capacity ratio cannot affect either
            # region or the Device conservation graph.
            return JunctionResult(
                external_flux_a=ExternalFlux.zero(int(np.asarray(state_a.f).size)),
                external_flux_b=ExternalFlux.zero(int(np.asarray(state_b.f).size)),
            )
        for label, state in (("a", state_a), ("b", state_b)):
            gap_scale = max(
                abs(float(state.gap)),
                abs(float(state.spectral.gap)),
                1.0,
            )
            if not np.isclose(
                state.gap,
                state.spectral.gap,
                rtol=1e-12,
                atol=1e-12 * gap_scale,
            ):
                raise ValueError(
                    "SymmetricGapTunnelingJunction requires each state's "
                    "public gap to match its spectral gap; "
                    f"state_{label} has {state.gap} vs "
                    f"{state.spectral.gap}."
                )
            occupation = np.asarray(state.f, dtype=float)
            if occupation.shape != state.spectral.E.shape:
                raise ValueError(
                    f"state_{label}.f must match its spectral energy-grid "
                    f"shape {state.spectral.E.shape}; got {occupation.shape}."
                )
            if np.any(~np.isfinite(occupation)) or np.any(
                (occupation < 0.0) | (occupation > 1.0)
            ):
                raise ValueError(
                    f"state_{label}.f must contain finite occupations in [0, 1]."
                )
        if state_a.spectral.E.size != state_b.spectral.E.size:
            raise ValueError(
                f"SymmetricGapTunnelingJunction requires matching E grids; "
                f"got {state_a.spectral.E.size} vs {state_b.spectral.E.size}."
            )
        if not np.allclose(
            state_a.spectral.E,
            state_b.spectral.E,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "SymmetricGapTunnelingJunction requires identical E grids "
                "between the two regions; got different values."
            )
        if not np.allclose(
            state_a.spectral.dE,
            state_b.spectral.dE,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "SymmetricGapTunnelingJunction requires identical finite-volume "
                "cell widths between the two regions."
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

        if not np.isclose(
            state_a.spectral.dynes_gamma,
            state_b.spectral.dynes_gamma,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "SymmetricGapTunnelingJunction requires identical Dynes "
                "broadening in both regions; got "
                f"{state_a.spectral.dynes_gamma} vs "
                f"{state_b.spectral.dynes_gamma}."
            )
        if not np.allclose(
            state_a.spectral.cell_weights,
            state_b.spectral.cell_weights,
            rtol=1e-12,
            atol=0.0,
        ):
            raise ValueError(
                "SymmetricGapTunnelingJunction requires identical per-bin "
                "finite-volume spectral capacities; a scalar "
                "capacity_ratio_a_to_b cannot represent an energy-dependent "
                "measure mismatch."
            )

        alpha_a = float(self.alpha_per_ns)
        alpha_b = alpha_a * float(self.capacity_ratio_a_to_b)
        f_a = state_a.f
        f_b = state_b.f

        gain_a = alpha_a * f_b
        loss_a = np.full_like(f_b, alpha_a)
        gain_b = alpha_b * f_a
        loss_b = np.full_like(f_a, alpha_b)

        diagnostics: dict[str, str | float] = {
            "junction": self.name,
            # Retain the historical diagnostic key as an alias for the
            # region-A rate while exposing both rates explicitly.
            "alpha_per_ns": alpha_a,
            "alpha_a_per_ns": alpha_a,
            "alpha_b_per_ns": alpha_b,
            "capacity_ratio_a_to_b": float(self.capacity_ratio_a_to_b),
        }

        return JunctionResult(
            external_flux_a=ExternalFlux(
                gain=gain_a,
                loss_rate=loss_a,
                diagnostics=diagnostics,
            ),
            external_flux_b=ExternalFlux(
                gain=gain_b,
                loss_rate=loss_b,
                diagnostics=diagnostics,
            ),
        )
