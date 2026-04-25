"""Device-level abstractions: Region, Junction, Qubit, Device, ExternalFlux.

Phase 2: ``ExternalFlux`` — the boundary ``(gain, loss_rate)``
source/sink contract that lets a Region's kinetic equation accept
input from an external coupling.

Phase 3: ``Region`` (one superconducting region wrapping a backend
state), ``Junction`` (abstract base) + ``SymmetricGapTunnelingJunction``
(simplest concrete subclass), ``Device`` (composition of regions +
junctions), ``solve_device_steady_state`` (outer Picard loop on
junction fluxes ↔ per-region steady states).

Phase 4: ``Qubit`` (N-level system with optional parity tracking),
``QubitState``, ``QubitTransitionChannel``, ``JunctionQubitCoupling``,
``solve_qubit_master_equation_steady_state``. ``Junction.evaluate``
gains an optional ``qubit_state`` parameter and returns
``qubit_channels`` alongside the per-region fluxes. ``Device``
carries an optional ``qubit``; the device solver evolves it
alongside the regions, pooling channels from all junctions.

See ``docs/Device_Architecture.md``.
"""

from qpsim.devices.device import Device, DeviceSolution, solve_device_steady_state
from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import (
    Junction,
    JunctionResult,
    SymmetricGapTunnelingJunction,
)
from qpsim.devices.qubit import (
    JunctionQubitCoupling,
    Qubit,
    QubitState,
    QubitTransitionChannel,
    build_rate_matrix,
    solve_qubit_master_equation_steady_state,
)
from qpsim.devices.region import Region

__all__ = [
    "Device",
    "DeviceSolution",
    "ExternalFlux",
    "Junction",
    "JunctionQubitCoupling",
    "JunctionResult",
    "Qubit",
    "QubitState",
    "QubitTransitionChannel",
    "Region",
    "SymmetricGapTunnelingJunction",
    "build_rate_matrix",
    "solve_device_steady_state",
    "solve_qubit_master_equation_steady_state",
]
