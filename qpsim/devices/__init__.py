"""Device-level abstractions: Region, Junction, Device, ExternalFlux.

Phase 2: ``ExternalFlux`` — the boundary ``(gain, loss_rate)``
source/sink contract that lets a Region's kinetic equation accept
input from an external coupling.

Phase 3: ``Region`` (one superconducting region wrapping a backend
state), ``Junction`` (abstract base) + ``SymmetricGapTunnelingJunction``
(simplest concrete subclass), ``Device`` (composition of regions +
junctions), ``solve_device_steady_state`` (outer Picard loop on
junction fluxes ↔ per-region steady states).

Planned (Phase 4): ``Qubit`` with parity tracking +
``JunctionQubitCoupling`` for the M25-style qubit-coupled paths.
See ``docs/Device_Architecture.md``.
"""

from qpsim.devices.device import Device, DeviceSolution, solve_device_steady_state
from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import (
    Junction,
    JunctionResult,
    SymmetricGapTunnelingJunction,
)
from qpsim.devices.region import Region

__all__ = [
    "Device",
    "DeviceSolution",
    "ExternalFlux",
    "Junction",
    "JunctionResult",
    "Region",
    "SymmetricGapTunnelingJunction",
    "solve_device_steady_state",
]
