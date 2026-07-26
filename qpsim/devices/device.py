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

Independent regions may use either thermal or stored finite-phonon inner
solves. Active conservative cross-region components currently require
``use_thermal_phonons=True`` because their component-level number certificate
does not yet include the stored non-equilibrium phonon state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import TYPE_CHECKING, Any

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
from qpsim.phonon_models.state import PhononModel, PhononState

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
        if not names:
            raise ValueError("Device must contain at least one region.")
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
    final_max_delta_n_ph
        Largest undamped phonon fixed-point defect
        ``max|F(n_ph) - n_ph|`` across all regions on the last iteration.
        Zero only when every stored phonon occupation is fixed (or absent).
    final_max_delta_p
        Qubit fixed-point defect ``max|Δp|`` on the last iteration.
        Zero when no qubit is present; below ``outer_tol * ||p||_inf``
        at success.
    final_max_number_backward_error
        Largest capacity-weighted number-changing backward error across
        conservative junction components. Zero when none are present;
        no larger than ``outer_tol`` at success.
    """

    states: dict[str, T3DiffusionState]
    n_outer_iterations: int
    final_max_delta_f: float
    qubit_state: QubitState | None = None
    final_max_delta_p: float = 0.0
    final_max_number_backward_error: float = 0.0
    # Appended after the pre-existing fields so legacy positional
    # construction keeps its meaning. New code should still use keywords.
    final_max_delta_n_ph: float = 0.0


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


def _array_signature(values: object) -> tuple[str, tuple[int, ...], bytes]:
    """Exact immutable signature for a numeric public-state array."""
    arr = np.asarray(values)
    contiguous = np.ascontiguousarray(arr)
    return contiguous.dtype.str, contiguous.shape, contiguous.tobytes()


def _dataclass_public_signature(value: object) -> object:
    """Recursively snapshot public dataclass content without aliasing arrays."""
    if isinstance(value, np.ndarray):
        return ("array", _array_signature(value))
    if isinstance(value, np.generic):
        return (type(value.item()), value.item())
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (item.name, _dataclass_public_signature(getattr(value, item.name)))
                for item in fields(value)
                if not item.name.startswith("_")
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value),
            tuple(_dataclass_public_signature(item) for item in value),
        )
    if isinstance(value, dict):
        return (
            type(value),
            tuple(
                sorted(
                    (
                        _dataclass_public_signature(key),
                        _dataclass_public_signature(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    return (type(value), value)


def _spectral_static_signature(state: T3DiffusionState) -> tuple[object, ...]:
    """Value signature for the static public spectral contract.

    Object identity is deliberately excluded: an injected pure backend may
    return a value-equivalent deep copy.  Identity is tracked separately in
    :func:`_region_input_signature`, where it is useful for detecting
    in-place mutation/replacement of the object that was handed to an
    extension.
    """
    spectral = state.spectral
    return (
        float(spectral.gap),
        _array_signature(spectral.E),
        _array_signature(spectral.dE),
        _array_signature(spectral.rho),
        _array_signature(spectral.cell_weights),
        _array_signature(spectral.cell_density),
        _array_signature(spectral.cell_anomalous_density),
        _array_signature(spectral.K_plus),
        _array_signature(spectral.K_minus),
        _array_signature(spectral.D_E),
        _array_signature(spectral.active_mask),
        float(spectral.dynes_gamma),
        float(spectral.diffusion_coefficient),
        float(spectral.rebuild_tolerance),
        float(spectral.active_margin_factor),
    )


def _phonon_static_signature(state: T3DiffusionState) -> tuple[object, ...]:
    phonon = state.phonon
    return (
        _array_signature(phonon.omega_bins),
        _array_signature(phonon.tau_l),
        phonon.model,
        _dataclass_public_signature(phonon.branches),
    )


def _region_static_signature(state: T3DiffusionState) -> tuple[object, ...]:
    """Fields the current Device outer map does not iterate or damp."""
    return (
        float(state.gap),
        _spectral_static_signature(state),
        _phonon_static_signature(state),
        _dataclass_public_signature(state.material),
        float(state.T_bath),
        state.tier,
    )


def _region_input_signature(state: T3DiffusionState) -> tuple[object, ...]:
    """Exact snapshot of every public region field visible to extensions."""
    return (
        _array_signature(state.f),
        _region_static_signature(state),
        id(state.spectral),
        id(state.material),
        id(state.phonon),
        _array_signature(state.phonon.n_ph),
    )


def _qubit_input_signature(state: QubitState | None) -> object:
    if state is None:
        return None
    return id(state), _array_signature(state.p), float(state.t_ns)


def _validate_backend_candidate_state(
    candidate: T3DiffusionState,
    expected: T3DiffusionState,
    *,
    region_name: str,
) -> T3DiffusionState:
    """Validate and normalize the complete T3 state from an injected backend."""
    try:
        candidate_f_raw = np.asarray(candidate.f)
        if np.iscomplexobj(candidate_f_raw):
            raise ValueError("f must be real-valued")
        candidate_f = np.asarray(candidate_f_raw, dtype=float).copy()
        gap = float(candidate.gap)
        spectral_gap = float(candidate.spectral.gap)
        T_bath = float(candidate.T_bath)
        E = np.asarray(candidate.spectral.E, dtype=float)
        dE = np.asarray(candidate.spectral.dE, dtype=float)
        active = np.asarray(candidate.spectral.active_mask)
        cell_weights = np.asarray(candidate.spectral.cell_weights, dtype=float)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Backend returned an invalid state for region {region_name!r}; "
            "the T3 state fields must be numeric and complete."
        ) from exc

    if candidate_f.shape != expected.f.shape:
        raise RuntimeError(
            f"Backend returned f with shape {candidate_f.shape} for "
            f"region {region_name!r}; expected {expected.f.shape}."
        )
    if np.any(~np.isfinite(candidate_f)) or np.any(
        (candidate_f < 0.0) | (candidate_f > 1.0)
    ):
        raise RuntimeError(
            f"Backend returned a non-finite or out-of-domain occupation "
            f"for region {region_name!r}; every f value must lie in [0, 1]."
        )
    if not np.isfinite(gap) or gap <= 0.0:
        raise RuntimeError(
            f"Backend returned a non-finite or non-positive gap for "
            f"region {region_name!r}."
        )
    gap_scale = max(abs(gap), abs(spectral_gap), 1.0)
    if not np.isfinite(spectral_gap) or not np.isclose(
        gap,
        spectral_gap,
        rtol=1e-12,
        atol=1e-12 * gap_scale,
    ):
        raise RuntimeError(
            f"Backend returned inconsistent state.gap/spectral.gap for "
            f"region {region_name!r}: {gap!r} versus {spectral_gap!r}."
        )
    if not np.isfinite(T_bath) or T_bath < 0.0:
        raise RuntimeError(
            f"Backend returned a non-finite or negative T_bath for "
            f"region {region_name!r}."
        )

    expected_E = np.asarray(expected.spectral.E, dtype=float)
    expected_dE = np.asarray(expected.spectral.dE, dtype=float)
    if (
        E.shape != candidate_f.shape
        or dE.shape != candidate_f.shape
        or active.shape != candidate_f.shape
        or active.dtype != np.bool_
        or cell_weights.shape != candidate_f.shape
        or np.any(~np.isfinite(E))
        or np.any(~np.isfinite(dE))
        or np.any(~np.isfinite(cell_weights))
        or np.any(np.diff(E) <= 0.0)
        or np.any(dE <= 0.0)
        or np.any(cell_weights < 0.0)
        or not np.array_equal(E, expected_E)
        or not np.array_equal(dE, expected_dE)
    ):
        raise RuntimeError(
            f"Backend returned an invalid or changed spectral grid for "
            f"region {region_name!r}. Device iteration requires one fixed, "
            "finite energy grid with positive widths."
        )

    try:
        phonon = candidate.phonon
        n_ph_raw = np.asarray(phonon.n_ph)
        omega_raw = np.asarray(phonon.omega_bins)
        tau_l_raw = np.asarray(phonon.tau_l)
        if any(
            np.iscomplexobj(values)
            for values in (n_ph_raw, omega_raw, tau_l_raw)
        ):
            raise ValueError("phonon arrays must be real-valued")
        validated_phonon = PhononState(
            n_ph=np.asarray(n_ph_raw, dtype=float).copy(),
            omega_bins=np.asarray(omega_raw, dtype=float).copy(),
            tau_l=np.asarray(tau_l_raw, dtype=float).copy(),
            model=phonon.model,
            branches=list(phonon.branches),
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Backend returned an invalid phonon state for region "
            f"{region_name!r}."
        ) from exc
    if (
        validated_phonon.model is not PhononModel.PH0_LOCAL
        or validated_phonon.n_branch != 1
        or validated_phonon.n_spatial != 1
    ):
        raise RuntimeError(
            f"Backend returned a phonon state outside the homogeneous "
            f"PH0_LOCAL Device contract for region {region_name!r}."
        )
    try:
        static_changed = (
            _region_static_signature(candidate)
            != _region_static_signature(expected)
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Backend returned invalid static state for region {region_name!r}."
        ) from exc
    if static_changed:
        raise RuntimeError(
            f"Backend changed static T3 fields for region {region_name!r}. "
            "The Device outer solver currently iterates only f and n_ph; "
            "gap, spectral grid/configuration, phonon grid/escape model, "
            "material, T_bath, and tier must remain invariant."
        )
    return replace(candidate, f=candidate_f, phonon=validated_phonon)


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
      3. Measure the UNDAMPED fixed-point defects in both ``f`` and stored
         ``n_ph`` per region, then take the damped step
         ``x_next = x + outer_damping*(F(x) - x)``.
      4. Certify convergence with a SCALE-AWARE tolerance: every
         region's defects must satisfy that relative bound against the
         corresponding ``f``/``n_ph`` scale (and the qubit defect likewise
         against ``||p||_inf = O(1)``). Every supported
         conservative junction component must also have a capacity-weighted
         number-changing backward error no larger than ``outer_tol``.

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
        (sidesteps the inner Picard-on-n_ph complication). Finite-phonon
        solves are supported only when the Device has no active conservative
        cross-region transfer component; such components need a future
        n_ph-aware Device-level number certificate.
    inner_anderson_depth
        Forwarded to the inner backend solve when relevant. Ignored
        when ``use_thermal_phonons=True``.
    outer_tol
        RELATIVE convergence threshold: each region's undamped fixed-point
        defects ``max|F(f) - f|`` and ``max|F(n_ph) - n_ph|`` must fall
        below ``outer_tol`` times their respective infinity-norm scales
        (with a tiny absolute floor for all-zero states). The qubit defect is
        certified the same way against ``||p||_inf``. This is also the
        explicit upper bound on
        the capacity-weighted conserved-number backward error; the solver
        never substitutes a looser hidden detector threshold.
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
        If the outer loop fails to converge within ``outer_max_iter`` or a
        configured number-changing component has no representable turnover.
    ValueError
        If controls are invalid or the requested Device mode cannot be
        certified by this solver.
    """
    # Control validation (2026-07-20/21 reviews): NaN/inf tolerances once
    # reported success, NaN damping returned non-finite occupations, and
    # complex/string controls leaked raw TypeError from NumPy comparisons.
    if not isinstance(use_thermal_phonons, (bool, np.bool_)):
        raise ValueError(
            "use_thermal_phonons must be a bool; "
            f"got {use_thermal_phonons!r}."
        )
    use_thermal_phonons = bool(use_thermal_phonons)

    def _real_control(name: str, value: Any) -> float:
        if (
            isinstance(value, (bool, np.bool_, str, bytes))
            or np.iscomplexobj(value)
        ):
            raise ValueError(f"{name} must be a finite real number; got {value!r}.")
        try:
            normalized = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{name} must be a finite real number; got {value!r}."
            ) from exc
        if not np.isfinite(normalized):
            raise ValueError(f"{name} must be finite; got {value!r}.")
        return normalized

    outer_tol = _real_control("outer_tol", outer_tol)
    if outer_tol <= 0.0:
        raise ValueError(f"outer_tol must be finite and positive; got {outer_tol!r}.")
    outer_damping = _real_control("outer_damping", outer_damping)
    if not 0.0 < outer_damping <= 1.0:
        raise ValueError(
            f"outer_damping must lie in (0, 1]; got {outer_damping!r}."
        )
    if (
        isinstance(outer_max_iter, (bool, np.bool_))
        or not isinstance(outer_max_iter, (int, np.integer))
        or outer_max_iter < 1
    ):
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

    # Per-region PAIR kernels for the global slow-mode certificate. Number-
    # conserving scattering is intentionally never assembled here: even its
    # roundoff can overwhelm pair turnover by many orders at 20--50 mK.
    from qpsim.collisions.phonon import build_recombination_kernel_base
    from qpsim.solvers.newton_steady_state import number_changing_gain_loss

    # Public conservative-edge contract. This deliberately does not key on a
    # concrete Junction type: custom kinetic junctions can participate once
    # they state their capacity ratio and satisfy the evaluated-flux check
    # below. A disabled edge returns None and is absent from the graph.
    conservative_junctions: list[tuple[Junction, float]] = []
    for j in device.junctions:
        ratio = j.qp_number_capacity_ratio_a_to_b()
        if ratio is None:
            continue
        ratio = float(ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError(
                f"Junction {j.name!r} returned invalid conservative "
                f"QP-capacity ratio C_a/C_b={ratio!r}; it must be finite "
                "and positive."
            )
        conservative_junctions.append((j, ratio))

    conservative_junction_ids = {id(j) for j, _ratio in conservative_junctions}
    conservative_regions = {
        region_name
        for j, _ratio in conservative_junctions
        for region_name in (j.region_a, j.region_b)
    }
    region_pair_kernels: dict[str, np.ndarray] = {}
    for name, st in states.items():
        if name not in conservative_regions:
            continue
        region_pair_kernels[name] = build_recombination_kernel_base(
            st.spectral, tau_0=st.material.tau_0, T_c=st.material.T_c
        )

    # Build capacity-weighted conservation components from the junction's
    # documented C_a/C_b ratio. Propagating the ratios detects inconsistent
    # cycles. Each Region's own finite-volume weights stay in the component
    # certificate; a concrete Junction is responsible for mapping between
    # unlike grids and its evaluated transfer is checked explicitly below.
    adjacency: dict[str, list[tuple[str, float]]] = {
        name: [] for name in device.regions
    }
    for j, ratio in conservative_junctions:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            inverse_ratio = 1.0 / ratio
        if not np.isfinite(inverse_ratio) or inverse_ratio <= 0.0:
            raise ValueError(
                f"Junction {j.name!r}: capacity_ratio_a_to_b={ratio!r} "
                "cannot be represented safely in the component capacity "
                "weights. Use a less extreme finite ratio."
            )
        # ratio = C_a/C_b: knowing C_a gives C_b=C_a/ratio.
        adjacency[j.region_a].append((j.region_b, inverse_ratio))
        adjacency[j.region_b].append((j.region_a, ratio))

    capacity_factors: dict[str, float] = {}
    conservation_components: list[list[str]] = []
    for seed in device.regions:
        if not adjacency[seed] or seed in capacity_factors:
            continue
        capacity_factors[seed] = 1.0
        component: list[str] = []
        stack = [seed]
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor, factor in adjacency[current]:
                proposed = capacity_factors[current] * factor
                if not np.isfinite(proposed) or proposed <= 0.0:
                    raise ValueError(
                        "Conservative junction capacity ratios overflow or "
                        "underflow while constructing a conserved component; "
                        "no numerically safe capacity weighting exists."
                    )
                if neighbor in capacity_factors:
                    cycle_rtol = min(1e-12, outer_tol)
                    if not np.isclose(
                        capacity_factors[neighbor],
                        proposed,
                        rtol=cycle_rtol,
                        atol=0.0,
                    ):
                        raise ValueError(
                            "Conservative junction capacity ratios are "
                            "inconsistent around a connected cycle; no "
                            "single conserved capacity weighting exists."
                        )
                    continue
                capacity_factors[neighbor] = proposed
                stack.append(neighbor)
        # Only relative capacities matter. Normalize each completed
        # component to avoid carrying an arbitrary large common scale into
        # the certificate, and reject any normalization underflow.
        component_scale = max(capacity_factors[name] for name in component)
        for name in component:
            normalized = capacity_factors[name] / component_scale
            if not np.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(
                    "Conservative junction capacity ratios have an extreme "
                    "dynamic range that cannot be represented after "
                    "component normalization."
                )
            capacity_factors[name] = normalized
        conservation_components.append(component)

    if not use_thermal_phonons and conservation_components:
        raise ValueError(
            "solve_device_steady_state cannot certify finite-phonon Device "
            "solutions with active conservative cross-region transfer: the "
            "component certificate does not yet include the stored "
            "non-equilibrium phonon state. Junction-free Devices and Devices "
            "whose conservative couplings are identically disabled remain "
            "valid finite-phonon wrappers around independent region solves."
        )

    # Unknown state-dependent Junctions cannot be assumed conservative.
    # An exclusive dissipation-owning closure (M25) anchors its own region
    # equations and is certified locally by Newton. A custom subclass may
    # explicitly promise prescribed, non-exchange region fluxes. Otherwise
    # only an exactly-zero emitted region flux is safe, checked every outer
    # iteration below; nonzero flux refuses before a result can be returned.
    uncertified_junctions = [
        j
        for j in device.junctions
        if id(j) not in conservative_junction_ids
        and not (
            getattr(j, "owns_region_dissipation", False)
            and getattr(j, "requires_exclusive_regions", False)
        )
        and not getattr(j, "prescribed_region_flux", False)
    ]
    uncertified_junction_ids = {id(j) for j in uncertified_junctions}

    conservative_ratio_by_id = {
        id(j): ratio for j, ratio in conservative_junctions
    }

    def _validate_conservative_transfer(
        junction: Junction,
        result_a: ExternalFlux,
        result_b: ExternalFlux,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
    ) -> None:
        """Verify one declared edge's capacity-weighted transfer balance."""
        ratio = conservative_ratio_by_id[id(junction)]
        for flux, state in ((result_a, state_a), (result_b, state_b)):
            flux._validate_for_NE(int(state.f.size))
            flux._validate_gain_support(state.spectral.active_mask)

        # Only relative capacities matter. Normalize before multiplication so
        # a large but representable ratio cannot overflow this verification.
        capacity_scale = max(ratio, 1.0)
        capacity_a = ratio / capacity_scale
        capacity_b = 1.0 / capacity_scale
        terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        rate_scale = 0.0
        for capacity, flux, state in (
            (capacity_a, result_a, state_a),
            (capacity_b, result_b, state_b),
        ):
            active = state.spectral.active_mask
            weights = capacity * state.spectral.cell_weights[active]
            gain_rate = flux.gain[active]
            with np.errstate(over="ignore", invalid="ignore"):
                loss_rate = flux.loss_rate[active] * state.f[active]
            if (
                np.any(~np.isfinite(weights))
                or np.any(~np.isfinite(gain_rate))
                or np.any(~np.isfinite(loss_rate))
            ):
                raise ValueError(
                    f"Junction {junction.name!r} produced non-finite "
                    "capacity-weighted transfer inputs while verifying its "
                    "declared conservative QP-capacity contract."
                )
            terms.append((weights, gain_rate, loss_rate))
            rate_scale = max(
                rate_scale,
                float(np.max(np.abs(gain_rate), initial=0.0)),
                float(np.max(np.abs(loss_rate), initial=0.0)),
            )
        if rate_scale == 0.0:
            return
        if not np.isfinite(rate_scale):
            raise ValueError(
                f"Junction {junction.name!r} produced a non-finite transfer "
                "scale while verifying its conservative contract."
            )
        weight_scale = max(
            (
                float(np.max(np.abs(weights), initial=0.0))
                for weights, _, _ in terms
            ),
            default=0.0,
        )
        if not np.isfinite(weight_scale) or weight_scale <= 0.0:
            raise ValueError(
                f"Junction {junction.name!r} has no finite positive "
                "capacity weight on its active transfer support."
            )
        numerator = 0.0
        denominator = 0.0
        for weights, gain_rate, loss_rate in terms:
            weights_scaled = weights / weight_scale
            gain_scaled = weights_scaled * (gain_rate / rate_scale)
            loss_scaled = weights_scaled * (loss_rate / rate_scale)
            if np.any(~np.isfinite(gain_scaled)) or np.any(
                ~np.isfinite(loss_scaled)
            ):
                raise ValueError(
                    f"Junction {junction.name!r} produced non-finite scaled "
                    "transfer terms while verifying its conservative contract."
                )
            numerator += float(np.sum(gain_scaled - loss_scaled))
            denominator += float(
                np.sum(np.abs(gain_scaled)) + np.sum(np.abs(loss_scaled))
            )
        error = abs(numerator) / max(denominator, 1e-300)
        # A declared conservative edge is part of the same fixed-point
        # contract advertised by ``outer_tol``.  Keep the historical
        # roundoff ceiling for ordinary solves, but do not silently relax a
        # caller's tighter request (2026-07-21 Round-6 final review).
        conservation_tol = min(1e-12, float(outer_tol))
        if not np.isfinite(error) or error > conservation_tol:
            raise ValueError(
                f"Junction {junction.name!r} violates its declared "
                "conservative QP-capacity contract: capacity-weighted "
                f"transfer backward error {error:.3e} exceeds "
                f"{conservation_tol:.3e}."
            )

    def _component_conserved_mode_error(
        accepted: dict[str, T3DiffusionState],
    ) -> float:
        """Max capacity-weighted pair-number error over components."""
        worst = 0.0
        for component in conservation_components:
            channel_terms: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
            rate_scale = 0.0
            configured = False
            exact_zero_temperature_vacuum = True
            for rname in component:
                rstate = accepted[rname]
                gain_pair, loss_pair, region_configured = (
                    number_changing_gain_loss(
                        rstate.f,
                        rstate.spectral,
                        region_pair_kernels[rname],
                        rstate.T_bath,
                    )
                )
                active = rstate.spectral.active_mask
                loss_term = loss_pair[active] * rstate.f[active]
                exact_zero_temperature_vacuum = (
                    exact_zero_temperature_vacuum
                    and rstate.T_bath == 0.0
                    and not np.any(rstate.f[active] != 0.0)
                    and not np.any(gain_pair[active] != 0.0)
                )
                rate_scale = max(
                    rate_scale,
                    float(np.max(np.abs(gain_pair[active]), initial=0.0)),
                    float(np.max(np.abs(loss_term), initial=0.0)),
                )
                weights = (
                    capacity_factors[rname]
                    * rstate.spectral.cell_weights[active]
                )
                channel_terms.append((gain_pair[active], loss_term, weights))
                configured = configured or region_configured
            if rate_scale == 0.0:
                if configured:
                    if exact_zero_temperature_vacuum:
                        # At exact T=0, f=0 with no represented pair gain is
                        # the physical ground-state root. Recombination and
                        # transfer loss coefficients have zero action on that
                        # state, so zero turnover is an exact certificate, not
                        # an underflowed finite-temperature number mode.
                        continue
                    raise RuntimeError(
                        "A conservative Device component has configured "
                        "number-changing channels but zero representable "
                        "float64 turnover; its QP-number mode cannot be "
                        "certified at this temperature/grid."
                    )
                continue
            numerator = 0.0
            denominator = 0.0
            for gain_pair, loss_term, weights in channel_terms:
                gain_scaled = weights * (gain_pair / rate_scale)
                loss_scaled = weights * (loss_term / rate_scale)
                numerator += float(np.sum(gain_scaled - loss_scaled))
                denominator += float(
                    np.sum(np.abs(gain_scaled))
                    + np.sum(np.abs(loss_scaled))
                )
            if denominator == 0.0:
                raise RuntimeError(
                    "A conservative Device component's capacity-weighted "
                    "number turnover is numerically unresolvable."
                )
            worst = max(worst, abs(numerator) / denominator)
        return worst

    last_delta_f = float("inf")
    last_delta_n_ph = float("inf")
    last_delta_p = 0.0
    last_slow_mode_error = 0.0
    # A converged defect certifies the *input* at which F(x) was evaluated.
    # Return it only when that full T3/qubit state is itself a prior map
    # output; otherwise ancillary backend-owned fields (notably n_ph) may be
    # stale even when f is already fixed. The first quiet evaluation promotes
    # the exact map output and a second evaluation certifies that snapshot.
    current_is_map_output = False
    for outer_iter in range(outer_max_iter):
        # Step 1: aggregate junction fluxes per region + pool qubit channels
        fluxes: dict[str, ExternalFlux | None] = dict.fromkeys(device.regions)
        all_qubit_channels: list[QubitTransitionChannel] = []
        for j in device.junctions:
            state_a = states[j.region_a]
            state_b = states[j.region_b]
            try:
                state_a_before = _region_input_signature(state_a)
                state_b_before = _region_input_signature(state_b)
                qubit_before = _qubit_input_signature(qubit_state)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Could not snapshot Junction {j.name!r}'s input state."
                ) from exc
            result = j.evaluate(
                state_a, state_b, qubit_state,
            )
            try:
                junction_mutated_input = (
                    _region_input_signature(state_a) != state_a_before
                    or _region_input_signature(state_b) != state_b_before
                    or _qubit_input_signature(qubit_state) != qubit_before
                )
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Junction {j.name!r} mutated an input into an invalid "
                    "public region/qubit state."
                ) from exc
            if junction_mutated_input:
                raise RuntimeError(
                    f"Junction {j.name!r} mutated a region or qubit input "
                    "state in place. Device fixed-point evaluation requires "
                    "a pure Junction map; return fluxes/channels without "
                    "modifying the referenced inputs."
                )
            if id(j) in conservative_junction_ids:
                _validate_conservative_transfer(
                    j,
                    result.external_flux_a,
                    result.external_flux_b,
                    states[j.region_a],
                    states[j.region_b],
                )
            emits_region_flux = bool(
                np.any(result.external_flux_a.gain != 0.0)
                or np.any(result.external_flux_a.loss_rate != 0.0)
                or np.any(result.external_flux_b.gain != 0.0)
                or np.any(result.external_flux_b.loss_rate != 0.0)
            )
            if emits_region_flux and id(j) in uncertified_junction_ids:
                raise ValueError(
                    f"Junction {j.name!r} emits nonzero state-dependent "
                    "region flux but has no Device number-mode safety "
                    "contract. Returning a steady state could hide an "
                    "exchange-dominated common mode. Declare a conservative "
                    "capacity ratio, declare a genuinely "
                    "prescribed local source with "
                    "prescribed_region_flux=True, or implement explicit "
                    "capacity/number accounting for this Junction."
                )
            if emits_region_flux and id(j) not in conservative_junction_ids and (
                j.region_a in conservative_regions
                or j.region_b in conservative_regions
            ):
                raise ValueError(
                    f"Junction {j.name!r} adds non-conservative or "
                    "uncertified flux to a conservative transfer "
                    "component. Its contribution is not part of that "
                    "component's capacity-weighted number certificate."
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
            try:
                state_before = _region_input_signature(state)
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Could not snapshot backend input for region {name!r}."
                ) from exc
            candidate = backend.steady_state(
                state,
                use_thermal_phonons=use_thermal_phonons,
                external_dissipation_only=owns_dissipation,
                external_flux=ef,
                external_flux_is_conservative_transfer=(
                    name in conservative_regions
                ),
                anderson_depth=inner_anderson_depth,
                newton_tol=inner_newton_tol,
                newton_max_iter=inner_newton_max_iter,
            )
            try:
                state_input_mutated = (
                    _region_input_signature(state) != state_before
                )
            except (AttributeError, TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(
                    f"Backend mutated region {name!r}'s input state into an "
                    "invalid public T3 state."
                ) from exc
            if state_input_mutated:
                raise RuntimeError(
                    f"Backend mutated region {name!r}'s input state in "
                    "place. Device fixed-point defects require a pure map; "
                    "return a new T3DiffusionState instead."
                )
            candidate = _validate_backend_candidate_state(
                candidate,
                state,
                region_name=name,
            )
            new_states[name] = candidate

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
            try:
                candidate_p = np.asarray(new_qubit_state.p, dtype=float)
            except (AttributeError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Qubit steady-state solver returned an invalid probability "
                    "array."
                ) from exc
            assert qubit_state is not None
            if candidate_p.shape != qubit_state.p.shape:
                raise RuntimeError(
                    "Qubit steady-state solver returned p with shape "
                    f"{candidate_p.shape}; expected {qubit_state.p.shape}."
                )
            if (
                np.any(~np.isfinite(candidate_p))
                or np.any(candidate_p < 0.0)
                or not np.isclose(
                    float(np.sum(candidate_p)),
                    1.0,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise RuntimeError(
                    "Qubit steady-state solver returned non-finite, negative, "
                    "or unnormalized probabilities."
                )

        # Step 3: scale-aware fixed-point defect per region, BEFORE damping.
        # A tiny absolute floor keeps an all-zero region from demanding
        # defect == 0 exactly.
        converged = True
        last_delta_f = 0.0
        last_delta_n_ph = 0.0
        for name in states:
            defect = float(np.max(np.abs(new_states[name].f - states[name].f)))
            scale = max(float(np.max(np.abs(new_states[name].f))), 1e-300)
            last_delta_f = max(last_delta_f, defect)
            if defect > outer_tol * scale:
                converged = False
            phonon_defect = float(
                np.max(
                    np.abs(
                        new_states[name].phonon.n_ph
                        - states[name].phonon.n_ph
                    )
                )
            )
            phonon_scale = max(
                float(np.max(np.abs(new_states[name].phonon.n_ph))),
                1e-300,
            )
            last_delta_n_ph = max(last_delta_n_ph, phonon_defect)
            if phonon_defect > outer_tol * phonon_scale:
                converged = False
        last_delta_p = 0.0
        if qubit_state is not None and new_qubit_state is not None:
            last_delta_p = float(np.max(np.abs(new_qubit_state.p - qubit_state.p)))
            p_scale = max(float(np.max(np.abs(new_qubit_state.p))), 1e-300)
            if last_delta_p > outer_tol * p_scale:
                converged = False

        if converged and not current_is_map_output:
            if outer_iter + 1 == outer_max_iter:
                break
            states = new_states
            qubit_state = new_qubit_state
            current_is_map_output = True
            continue

        if converged and conservation_components:
            # This is an acceptance tolerance, not a coarse pathology
            # detector: the advertised outer_tol is the actual maximum
            # capacity-weighted number backward error. If the fixed-point
            # defect becomes quiet first, keep iterating; never return a
            # state certified only to a hidden 1000x-looser threshold.
            last_slow_mode_error = _component_conserved_mode_error(states)
            if last_slow_mode_error > outer_tol:
                converged = False
        if converged:
            # ``last_delta_*`` was measured at this exact pre-update state.
            # Returning a freshly damped candidate would make the diagnostic
            # describe a different snapshot and, for nonlinear maps, could
            # return a state whose true defect is much larger. The current
            # state is the one actually certified by F(x)-x above.
            return DeviceSolution(
                states=states,
                qubit_state=qubit_state,
                n_outer_iterations=outer_iter + 1,
                final_max_delta_f=last_delta_f,
                final_max_delta_n_ph=last_delta_n_ph,
                final_max_delta_p=last_delta_p,
                final_max_number_backward_error=last_slow_mode_error,
            )

        # There is no next iteration in which to certify a final damped
        # candidate. Keep ``states`` at the snapshot whose defect was just
        # measured so the failure diagnostics below remain truthful too.
        if outer_iter + 1 == outer_max_iter:
            break

        # Step 4: damped update x_next = x + θ(F(x) − x). Damping breaks
        # the period-2 orbits of the undamped simultaneous (Jacobi) map.
        damped_states: dict[str, T3DiffusionState] = {}
        for name in states:
            f_damped = states[name].f + outer_damping * (
                new_states[name].f - states[name].f
            )
            n_ph_damped = states[name].phonon.n_ph + outer_damping * (
                new_states[name].phonon.n_ph - states[name].phonon.n_ph
            )
            phonon_damped = replace(
                new_states[name].phonon,
                n_ph=n_ph_damped,
            )
            damped_states[name] = replace(
                new_states[name],
                f=f_damped,
                phonon=phonon_damped,
            )
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
        current_is_map_output = outer_damping == 1.0

    if conservation_components:
        last_slow_mode_error = _component_conserved_mode_error(states)
    raise RuntimeError(
        f"Device outer Picard loop did not converge in {outer_max_iter} "
        f"iterations. Final fixed-point defect max |F(f)-f| = "
        f"{last_delta_f:.2e}, max |F(n_ph)-n_ph| = "
        f"{last_delta_n_ph:.2e}, max |Δp| = {last_delta_p:.2e}, global "
        "conserved-QP-number/common-mode backward error = "
        f"{last_slow_mode_error:.2e} "
        f"(relative tol {outer_tol:.2e}, damping {outer_damping:g})."
    )
