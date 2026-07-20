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

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import Junction
from qpsim.devices.qubit import (
    Qubit,
    QubitState,
    QubitTransitionChannel,
    solve_qubit_master_equation_steady_state,
)
from qpsim.devices.region import Region

if TYPE_CHECKING:
    from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState


@dataclass
class Device:
    """Composition of named superconducting regions + tunnel junctions
    + optional coupled Qubit.

    Parameters
    ----------
    regions
        Mapping from region name to :class:`Region`. Names must match
        the region_a / region_b strings on every Junction.
    junctions
        List of :class:`Junction` instances. Each Junction's
        ``region_a`` / ``region_b`` must reference keys in ``regions``.
    qubit
        Optional coupled :class:`Qubit`. When present, junctions with
        a ``qubit_coupling`` emit ``QubitTransitionChannel`` records
        that the device solver pools and feeds into the qubit master
        equation alongside the per-region kinetic-equation solves.

    Raises
    ------
    ValueError
        At construction, if any Junction references an unknown region.
    """

    regions: dict[str, Region]
    junctions: list[Junction] = field(default_factory=list)
    qubit: Qubit | None = None

    def __post_init__(self) -> None:
        names = set(self.regions.keys())
        junctions_by_region: dict[str, list[Junction]] = {
            name: [] for name in names
        }
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
            junctions_by_region[j.region_a].append(j)
            junctions_by_region[j.region_b].append(j)

        # Reduced closures that solve an isolated subsystem internally cannot
        # respond to another Junction changing either region's f(E). Letting
        # one share a region produces a superficially converged but
        # non-self-consistent Device solution, so reject the topology early.
        for j in self.junctions:
            if not getattr(j, "requires_exclusive_regions", False):
                continue
            for region_name in (j.region_a, j.region_b):
                touching = junctions_by_region[region_name]
                if len(touching) > 1:
                    other_names = [other.name for other in touching if other is not j]
                    if not other_names:
                        other_names = [j.name]
                    raise ValueError(
                        f"Junction {j.name!r} requires exclusive regions, but "
                        f"region {region_name!r} is also touched by junction(s) "
                        f"{other_names}. Its isolated closure does not consume "
                        "the evolving region occupation from other junctions."
                    )


@dataclass
class DeviceSolution:
    """Result of :func:`solve_device_steady_state`.

    Attributes
    ----------
    states
        Converged ``T3DiffusionState`` per region (keyed by name).
    qubit_state
        Converged ``QubitState`` when ``device.qubit`` is set; ``None``
        otherwise.
    n_outer_iterations
        Number of outer Picard iterations consumed.
    final_max_delta_f
        Largest UNDAMPED fixed-point defect ``max|F(f) - f|`` across
        all regions on the last iteration. At success every region's
        defect is below ``outer_tol * ||f||_inf`` (scale-aware).
    final_max_delta_p
        Qubit fixed-point defect ``max|Δp|`` on the last iteration.
        Zero when no qubit is present; below ``outer_tol * ||p||_inf``
        at success.
    """

    states: dict[str, T3DiffusionState]
    n_outer_iterations: int
    final_max_delta_f: float
    qubit_state: QubitState | None = None
    final_max_delta_p: float = 0.0


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
    outer_tol: float = 1e-6,
    outer_max_iter: int = 100,
    outer_damping: float = 0.5,
    inner_newton_tol: float = 1e-12,
    inner_newton_max_iter: int = 200,
) -> DeviceSolution:
    """Damped outer Picard loop on (junction fluxes ↔ region states).

    Each outer iteration:
      1. Evaluate every Junction at the current region states; sum the
         per-region :class:`ExternalFlux` contributions.
      2. For each region, solve the steady-state ``f(E)`` at the
         aggregated boundary flux via the T3 backend.
      3. Measure the UNDAMPED fixed-point defect ``max|F(x) - x|`` per
         region, then take the damped step
         ``x_next = x + outer_damping*(F(x) - x)``.
      4. Certify convergence with a SCALE-AWARE tolerance: every
         region's defect must satisfy
         ``defect <= outer_tol * max(||f||_inf, tiny)`` (and the qubit
         defect likewise against ``||p||_inf = O(1)``).

    History (2026-07-20 external review): the previous undamped
    simultaneous update certified successive-iterate deltas against an
    ABSOLUTE 1e-8 tolerance. At 100 mK the entire occupation signal is
    ~8e-10, so any state pair — including a 50%-wrong region swap on a
    period-2 orbit of the Jacobi map — certified as converged. Damping
    breaks two-cycles; the relative defect certification makes cold
    regions as strict as warm ones.

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
        RELATIVE convergence threshold: each region's undamped
        fixed-point defect ``max|F(f) - f|`` must fall below
        ``outer_tol * ||f||_inf`` (with a tiny absolute floor for
        all-zero states). The qubit defect is certified the same way
        against ``||p||_inf``.
    outer_max_iter
        Hard cap on the outer Picard iteration count.
    outer_damping
        Damping factor θ in ``x_next = x + θ(F(x) - x)``; ``1.0``
        recovers the undamped update. The 0.5 default suppresses the
        period-2 Jacobi oscillations of symmetric exchange-coupled
        regions.
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
    # Initial qubit state: uniform mixture if not specified.
    qubit_state: QubitState | None = None
    if device.qubit is not None:
        n_par = 2 if device.qubit.track_parity else 1
        n_states_q = device.qubit.n_levels * n_par
        qubit_state = QubitState(
            p=np.full(n_states_q, 1.0 / n_states_q).reshape(
                (device.qubit.n_levels, 2) if device.qubit.track_parity
                else (device.qubit.n_levels,)
            ),
        )

    # Build the dissipation-ownership map: region_name → owning Junction.
    # At most one Junction may claim ownership over any given region;
    # multiple owners would each compute a complete dissipation flux and
    # the sum would over-count.
    dissipation_owner: dict[str, str] = {}
    for j in device.junctions:
        if not getattr(j, "owns_region_dissipation", False):
            continue
        for region_name in (j.region_a, j.region_b):
            if region_name in dissipation_owner:
                raise ValueError(
                    f"Region {region_name!r} has two junctions claiming "
                    f"dissipation ownership: {dissipation_owner[region_name]!r} "
                    f"and {j.name!r}. At most one Junction per region may set "
                    "owns_region_dissipation=True."
                )
            dissipation_owner[region_name] = j.name

    last_delta_f = float("inf")
    last_delta_p = 0.0
    for outer_iter in range(outer_max_iter):
        # Step 1: aggregate junction fluxes per region + pool qubit channels
        fluxes: dict[str, ExternalFlux | None] = dict.fromkeys(device.regions)
        all_qubit_channels: list[QubitTransitionChannel] = []
        for j in device.junctions:
            result = j.evaluate(
                states[j.region_a], states[j.region_b], qubit_state,
            )
            fluxes[j.region_a] = _aggregate_flux(
                fluxes[j.region_a], result.external_flux_a
            )
            fluxes[j.region_b] = _aggregate_flux(
                fluxes[j.region_b], result.external_flux_b
            )
            all_qubit_channels.extend(result.qubit_channels)

        # Step 2: per-region steady-state solve at frozen flux
        new_states: dict[str, T3DiffusionState] = {}
        for name, state in states.items():
            ef = fluxes[name]
            owns_dissipation = name in dissipation_owner
            new_states[name] = backend.steady_state(
                state,
                use_thermal_phonons=use_thermal_phonons,
                external_dissipation_only=owns_dissipation,
                external_flux=ef,
                anderson_depth=inner_anderson_depth,
                newton_tol=inner_newton_tol,
                newton_max_iter=inner_newton_max_iter,
            )

        # Step 2b: qubit master-equation steady state at frozen channels
        new_qubit_state = qubit_state
        if device.qubit is not None:
            if not all_qubit_channels:
                raise RuntimeError(
                    "Device has a Qubit but no junction emitted any "
                    "qubit_channels. The qubit's steady state is undefined "
                    "without channels — silently returning the initial "
                    "uniform mixture would hide a miswired coupling. Either "
                    "remove device.qubit, or wire at least one Junction with "
                    "a qubit_coupling that emits QubitTransitionChannel "
                    "records."
                )
            new_qubit_state = solve_qubit_master_equation_steady_state(
                all_qubit_channels, device.qubit,
            )

        # Step 3: scale-aware fixed-point defect per region, BEFORE damping.
        # A tiny absolute floor keeps an all-zero region from demanding
        # defect == 0 exactly.
        converged = True
        last_delta_f = 0.0
        for name in states:
            defect = float(np.max(np.abs(new_states[name].f - states[name].f)))
            scale = max(float(np.max(np.abs(new_states[name].f))), 1e-300)
            last_delta_f = max(last_delta_f, defect)
            if defect > outer_tol * scale:
                converged = False
        last_delta_p = 0.0
        if qubit_state is not None and new_qubit_state is not None:
            last_delta_p = float(np.max(np.abs(new_qubit_state.p - qubit_state.p)))
            p_scale = max(float(np.max(np.abs(new_qubit_state.p))), 1e-300)
            if last_delta_p > outer_tol * p_scale:
                converged = False

        # Step 4: damped update x_next = x + θ(F(x) − x). Damping breaks
        # the period-2 orbits of the undamped simultaneous (Jacobi) map.
        damped_states: dict[str, T3DiffusionState] = {}
        for name in states:
            f_damped = states[name].f + outer_damping * (
                new_states[name].f - states[name].f
            )
            damped_states[name] = replace(new_states[name], f=f_damped)
        states = damped_states
        if qubit_state is not None and new_qubit_state is not None:
            p_damped = qubit_state.p + outer_damping * (
                new_qubit_state.p - qubit_state.p
            )
            p_damped = np.maximum(p_damped, 0.0)
            p_damped = p_damped / p_damped.sum()
            qubit_state = QubitState(p=p_damped)
        else:
            qubit_state = new_qubit_state

        if converged:
            return DeviceSolution(
                states=states,
                qubit_state=qubit_state,
                n_outer_iterations=outer_iter + 1,
                final_max_delta_f=last_delta_f,
                final_max_delta_p=last_delta_p,
            )

    raise RuntimeError(
        f"Device outer Picard loop did not converge in {outer_max_iter} "
        f"iterations. Final fixed-point defect max |F(f)-f| = "
        f"{last_delta_f:.2e}, max |Δp| = {last_delta_p:.2e} "
        f"(relative tol {outer_tol:.2e}, damping {outer_damping:g})."
    )
