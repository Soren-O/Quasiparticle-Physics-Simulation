"""Device — composition of Regions + Junctions, with a top-level solver.

Phase 3 ships:

* :class:`Device` — dataclass holding regions (keyed by name) and a
  list of junctions.
* :func:`solve_device_steady_state` — outer Picard loop:
    1. Evaluate every Junction at the current region states; aggregate
       the per-region ExternalFlux contributions.
    2. Solve each region's steady state with that aggregated flux as
       the boundary input.
    3. Repeat until per-region |Δf| converges below ``outer_tol``.

For Phase 3 the inner per-region solve uses ``use_thermal_phonons=True``
(τ_l → 0 limit, no inner Picard on n_ph). Phase 4+ extends to coupled
(f, n_ph) inner solves and qubit master-equation evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import Junction
from qpsim.devices.region import Region

if TYPE_CHECKING:
    from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState


@dataclass
class Device:
    """Composition of named superconducting regions + tunnel junctions.

    Parameters
    ----------
    regions
        Mapping from region name to :class:`Region`. Names must match
        the region_a / region_b strings on every Junction.
    junctions
        List of :class:`Junction` instances. Each Junction's
        ``region_a`` / ``region_b`` must reference keys in ``regions``.

    Raises
    ------
    ValueError
        At construction, if any Junction references an unknown region.
    """

    regions: dict[str, Region]
    junctions: list[Junction] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = set(self.regions.keys())
        for j in self.junctions:
            if j.region_a not in names:
                raise ValueError(
                    f"Junction {j.name!r} references unknown region_a "
                    f"{j.region_a!r}; known regions are {sorted(names)}."
                )
            if j.region_b not in names:
                raise ValueError(
                    f"Junction {j.name!r} references unknown region_b "
                    f"{j.region_b!r}; known regions are {sorted(names)}."
                )


@dataclass
class DeviceSolution:
    """Result of :func:`solve_device_steady_state`.

    Attributes
    ----------
    states
        Converged ``T3DiffusionState`` per region (keyed by name).
    n_outer_iterations
        Number of outer Picard iterations consumed.
    final_max_delta_f
        Largest ``max|Δf|`` across all regions on the last iteration.
        Below ``outer_tol`` at success.
    """

    states: dict[str, T3DiffusionState]
    n_outer_iterations: int
    final_max_delta_f: float


def _aggregate_flux(
    a: ExternalFlux | None, b: ExternalFlux
) -> ExternalFlux:
    """Sum two ExternalFlux contributions on the same region."""
    if a is None:
        return b
    return ExternalFlux(
        gain=a.gain + b.gain,
        loss_rate=a.loss_rate + b.loss_rate,
        diagnostics={**a.diagnostics, **b.diagnostics},
    )


def solve_device_steady_state(
    device: Device,
    *,
    backend: T3DiffusionBackend | None = None,
    use_thermal_phonons: bool = True,
    inner_anderson_depth: int = 3,
    outer_tol: float = 1e-8,
    outer_max_iter: int = 100,
    inner_newton_tol: float = 1e-12,
    inner_newton_max_iter: int = 200,
) -> DeviceSolution:
    """Outer Picard loop on (junction fluxes ↔ per-region steady states).

    Each outer iteration:
      1. Evaluate every Junction at the current region states; sum the
         per-region :class:`ExternalFlux` contributions.
      2. For each region, solve the steady-state ``f(E)`` at the
         aggregated boundary flux via the T3 backend.
      3. Check convergence: max over regions of ``max|f_new - f_old|``.

    Parameters
    ----------
    device
        Region+Junction composition to solve.
    backend
        Optional shared T3 backend instance; one is constructed if
        omitted.
    use_thermal_phonons
        Pin n_ph at the substrate Bose-Einstein distribution and run
        Newton-only on f for each region. Default for Phase 3
        (sidesteps the inner Picard-on-n_ph complication). Phase 4+
        will allow finite-τ_l inner solves and coupled-Newton.
    inner_anderson_depth
        Forwarded to the inner backend solve when relevant. Ignored
        when ``use_thermal_phonons=True``.
    outer_tol
        Convergence threshold on the largest per-region ``|Δf|`` over
        consecutive outer iterations.
    outer_max_iter
        Hard cap on the outer Picard iteration count.
    inner_newton_tol, inner_newton_max_iter
        Inner backend solver controls.

    Returns
    -------
    DeviceSolution
        Converged per-region states plus diagnostics.

    Raises
    ------
    RuntimeError
        If the outer loop fails to converge within ``outer_max_iter``.
    """
    if backend is None:
        # Lazy import to avoid circular dep with qpsim.backends.t3_diffusion
        # which itself imports qpsim.devices.external_flux.
        from qpsim.backends.t3_diffusion import T3DiffusionBackend
        backend = T3DiffusionBackend()

    # Initial region states (copies are taken implicitly by replace())
    states: dict[str, T3DiffusionState] = {
        name: r.state for name, r in device.regions.items()
    }

    last_delta = float("inf")
    for outer_iter in range(outer_max_iter):
        # Step 1: aggregate junction fluxes per region
        fluxes: dict[str, ExternalFlux | None] = dict.fromkeys(device.regions)
        for j in device.junctions:
            result = j.evaluate(states[j.region_a], states[j.region_b])
            fluxes[j.region_a] = _aggregate_flux(
                fluxes[j.region_a], result.external_flux_a
            )
            fluxes[j.region_b] = _aggregate_flux(
                fluxes[j.region_b], result.external_flux_b
            )

        # Step 2: per-region steady-state solve at frozen flux
        new_states: dict[str, T3DiffusionState] = {}
        for name, state in states.items():
            ef = fluxes[name]
            new_states[name] = backend.steady_state(
                state,
                use_thermal_phonons=use_thermal_phonons,
                external_flux=ef,
                anderson_depth=inner_anderson_depth,
                newton_tol=inner_newton_tol,
                newton_max_iter=inner_newton_max_iter,
            )

        # Step 3: convergence check
        last_delta = max(
            float(np.max(np.abs(new_states[name].f - states[name].f)))
            for name in states
        )
        states = new_states

        if last_delta < outer_tol:
            return DeviceSolution(
                states=states,
                n_outer_iterations=outer_iter + 1,
                final_max_delta_f=last_delta,
            )

    raise RuntimeError(
        f"Device outer Picard loop did not converge in {outer_max_iter} "
        f"iterations. Final max |Δf| = {last_delta:.2e} (tol {outer_tol:.2e})."
    )
