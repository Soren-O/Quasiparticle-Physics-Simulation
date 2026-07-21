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

import warnings
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


#: Dimensionless limit for the global conserved-number-mode certificate
#: (see solve_device_steady_state). Manifold pathology measures ~0.6;
#: accepted converged states measure <= ~1e-3.
_CONSERVED_MODE_LIMIT = 0.05


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
    # Control validation (2026-07-20 review: NaN/inf tolerances previously
    # reported success and NaN damping returned non-finite occupations).
    if not (np.isfinite(outer_tol) and outer_tol > 0.0):
        raise ValueError(f"outer_tol must be finite and positive; got {outer_tol!r}.")
    if not (np.isfinite(outer_damping) and 0.0 < outer_damping <= 1.0):
        raise ValueError(
            f"outer_damping must lie in (0, 1]; got {outer_damping!r}."
        )
    if not isinstance(outer_max_iter, (int, np.integer)) or outer_max_iter < 1:
        raise ValueError(
            f"outer_max_iter must be a positive integer; got {outer_max_iter!r}."
        )

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

    # Per-region collision kernels for the global slow-mode certificate
    # (grids/materials are fixed across outer iterations).
    from qpsim.collisions.phonon import (
        build_recombination_kernel_base,
        build_scattering_kernel_base,
    )
    from qpsim.solvers.newton_steady_state import thermal_collision_gain_loss

    region_kernels: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, st in states.items():
        region_kernels[name] = (
            build_scattering_kernel_base(
                st.spectral, tau_0=st.material.tau_0, T_c=st.material.T_c
            ),
            build_recombination_kernel_base(
                st.spectral, tau_0=st.material.tau_0, T_c=st.material.T_c
            ),
        )

    # The conserved-mode certificate needs the junction transfer to cancel
    # bin-wise in the weighted number sum. That is provable here only for
    # SymmetricGapTunnelingJunction at capacity ratio 1 between regions
    # with identical cell weights — exactly the configuration of the
    # demonstrated failure. Outside that contract the certificate is
    # skipped LOUDLY (once): the common-mode certification gap remains an
    # open limitation for heterogeneous junction sets (needs a coupled
    # multi-region solve / per-junction number-flux accounting).
    from qpsim.devices.junction import SymmetricGapTunnelingJunction

    def _conserved_mode_certifiable() -> bool:
        for j in device.junctions:
            if not isinstance(j, SymmetricGapTunnelingJunction):
                return False
            if getattr(j, "capacity_ratio_a_to_b", 1.0) != 1.0:
                return False
            w_a = device.regions[j.region_a].state.spectral.cell_weights
            w_b = device.regions[j.region_b].state.spectral.cell_weights
            if w_a.shape != w_b.shape or not np.array_equal(w_a, w_b):
                return False
        return True

    conserved_mode_in_scope = _conserved_mode_certifiable()
    if not conserved_mode_in_scope:
        warnings.warn(
            "solve_device_steady_state: the global conserved-mode "
            "certificate is only implemented for symmetric ratio-1 "
            "tunneling junctions with matched cell weights; this device "
            "is outside that contract, so exchange-dominated common-mode "
            "errors (all regions off by a shared factor) are NOT "
            "certified against. Treat cold-temperature results with a "
            "poor initial state with care.",
            RuntimeWarning,
            stacklevel=2,
        )

    def _global_slow_mode_error(
        current: dict[str, T3DiffusionState],
        current_fluxes: dict[str, ExternalFlux | None],
    ) -> float:
        """Backward error of the conserved total-QP-number mode.

        Junction exchange conserves total QP number (within the
        certified contract above), so summing the FULL per-region number
        balance (collisions + flux) cancels the internal transfer and
        exposes the global slow mode the per-region frozen-flux solves
        cannot see: exchange-dominated regions pin each other to a
        common-mode manifold whose Picard defect is
        ~(collision/exchange) x error — far below any defect tolerance —
        while the state is arbitrarily far from equilibrium (2026-07-20
        round-4 review: both regions at c*f_FD certified for any c).
        Normalized by the COLLISION-ONLY number turnover so balanced
        exchange cannot self-certify.
        """
        number_residual = 0.0
        turnover = 0.0
        for rname, rstate in current.items():
            K_s0_r, K_r0_r = region_kernels[rname]
            gain_c, loss_c = thermal_collision_gain_loss(
                rstate.f, rstate.spectral, K_s0_r, K_r0_r, rstate.T_bath,
            )
            # Number-CHANGING (pair generation/recombination) turnover only:
            # scattering dominates the raw turnover but conserves number
            # exactly, and normalizing by it buries the conserved-mode
            # signal under an e^{-Delta/kT} factor (a 0.5*f_FD manifold
            # measured 2.9e-6 against full turnover vs 0.60 against pair
            # turnover).
            gain_p, loss_p = thermal_collision_gain_loss(
                rstate.f, rstate.spectral, None, K_r0_r, rstate.T_bath,
            )
            gain = gain_c
            loss = loss_c
            ef = current_fluxes.get(rname)
            if ef is not None:
                gain = gain + ef.gain
                loss = loss + ef.loss_rate
            w = rstate.spectral.cell_weights
            number_residual += float(np.sum(w * (gain - loss * rstate.f)))
            turnover += float(
                np.sum(w * (np.abs(gain_p) + np.abs(loss_p * rstate.f)))
            )
        return abs(number_residual) / max(turnover, 1e-300)

    def _accepted_state_slow_mode_error(
        accepted: dict[str, T3DiffusionState],
        accepted_qubit: QubitState | None,
    ) -> float:
        """Re-evaluate junction fluxes AT the accepted states and certify."""
        accepted_fluxes: dict[str, ExternalFlux | None] = dict.fromkeys(
            device.regions
        )
        for j in device.junctions:
            result = j.evaluate(
                accepted[j.region_a], accepted[j.region_b], accepted_qubit,
            )
            accepted_fluxes[j.region_a] = _aggregate_flux(
                accepted_fluxes[j.region_a], result.external_flux_a
            )
            accepted_fluxes[j.region_b] = _aggregate_flux(
                accepted_fluxes[j.region_b], result.external_flux_b
            )
        return _global_slow_mode_error(accepted, accepted_fluxes)

    last_delta_f = float("inf")
    last_delta_p = 0.0
    last_slow_mode_error = float("inf")
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

        if converged and conserved_mode_in_scope:
            # Certify the conserved global mode AT the accepted states
            # with re-evaluated fluxes (exact cancellation, no snapshot
            # lag). A quiet defect with an unbalanced conserved mode
            # means the regions sit on an exchange-dominated common-mode
            # manifold this frozen-flux Picard splitting cannot drain
            # (drainage ~ collision/exchange per iteration): refuse.
            # The conserved-mode certificate is a MANIFOLD DETECTOR with
            # a fixed dimensionless threshold: measured calibration —
            # 0.60 for regions on a common-factor manifold (any
            # temperature), 1.0e-3 for the legitimately converged
            # mismatched-T fixture, ~1e-11 at thermal equilibrium. 5%
            # sits an order above accepted states and an order below the
            # pathology.
            last_slow_mode_error = _accepted_state_slow_mode_error(
                states, qubit_state
            )
            if last_slow_mode_error > _CONSERVED_MODE_LIMIT:
                raise RuntimeError(
                    "Device outer loop reached a quiet fixed-point defect "
                    f"({last_delta_f:.2e}) while the global conserved-mode "
                    f"backward error is {last_slow_mode_error:.2e} > "
                    f"{outer_tol:.2e}: the regions are pinned on an "
                    "exchange-dominated common-mode manifold away from "
                    "collision equilibrium (e.g. all regions scaled by a "
                    "common factor). The frozen-flux Picard splitting "
                    "cannot drain this mode; use a coupled multi-region "
                    "solve or start from a better initial state."
                )
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
        f"{last_delta_f:.2e}, max |Δp| = {last_delta_p:.2e}, global "
        f"conserved-mode backward error = {last_slow_mode_error:.2e} "
        f"(relative tol {outer_tol:.2e}, damping {outer_damping:g})."
    )
