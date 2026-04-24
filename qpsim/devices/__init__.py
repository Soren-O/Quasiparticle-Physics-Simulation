"""Device-level abstractions: Region, Junction, Qubit, ExternalFlux.

Phase 2 (this module ships): ``ExternalFlux`` — the boundary
``(gain, loss_rate)`` source/sink contract that lets a Region's
kinetic equation accept input from an external coupling (a
Junction, an explicit drive, or any other source). Threaded
through ``newton_solve_f``, ``coupled_newton_solve``,
``solve_steady_state``, ``T3DiffusionBackend.step``, and
``run_time_dependent`` (callable form).

Planned (Phase 3+): ``Region``, ``Junction``, ``Qubit``, ``Device``,
``solve_device_steady_state``. See ``docs/Device_Architecture.md``.
"""

from qpsim.devices.external_flux import ExternalFlux

__all__ = ["ExternalFlux"]
