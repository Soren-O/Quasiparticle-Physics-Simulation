"""T3 backend: isotropic dirty-limit diffusion (scalar occupation).

Composes the pieces from earlier Gate 2 commits into a working
backend with both steady-state and transient time-evolution paths.

Scope for Gate 2: spatially-homogeneous runs (``N_spatial = 1``), a
scalar gap, the e-phonon integral, and optional sub-gap / PB photon
channels via ``photon_params`` / ``pb_photon_params``. Moving-gap ``step()``
uses a stage-constrained ETD2 reduction on persistent BCS material
coordinates. The standalone ``apply_gap_update`` remains an algebraic
projection, not a time-parameterized flow. ``apply_transport`` remains a
no-op for ``N_spatial = 1`` (real spatial diffusion lands at Gate 5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

from qpsim.backends.base import Tier
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.devices.external_flux import ExternalFlux
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononModel, PhononState
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)
from qpsim.physics.gap_equation import (
    GapBelowGridSupportError,
    GapCalibration,
    calibrate_gap,
    solve_gap,
)
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state
from qpsim.solvers.coupled_newton import coupled_newton_solve
from qpsim.solvers.etd import etd2_step


class SelfConsistentGapCollapseError(RuntimeError):
    """The driven state has no supported superconducting gap root."""

    def __init__(self, *, iteration: int, max_occupation: float) -> None:
        self.iteration = int(iteration)
        self.max_occupation = float(max_occupation)
        super().__init__(
            "Self-consistent gap collapsed: solve_gap found no supported "
            "superconducting gap (Delta=0 or below the represented grid "
            f"support) at iteration {self.iteration} with "
            f"|f|_max={self.max_occupation:.3e}. "
            "The drive has exceeded the pair-breaking threshold; this solver "
            "does not yet support the normal state."
        )


_TAU_L_UNIFORMITY_RTOL = 1e-10
_GAP_STATE_MATCH_RTOL = 1e-12
_GAP_SUPPORT_EPS = 1e-30
# ``apply_gap_update`` is a nonlinear algebraic projection: solving the gap
# changes the spectral coordinates and therefore remaps ``f``; that remapped
# occupation changes the gap equation in turn.  Certify the completed state,
# rather than accepting one solve/remap pass.  The Brent tolerance is kept
# well below the projection tolerance so root-solver noise cannot set the
# fixed-point floor.
_GAP_PROJECTION_RTOL = 1e-12
_GAP_PROJECTION_ATOL_UEV = 1e-10
_GAP_PROJECTION_SOLVE_XTOL_UEV = 1e-12
_GAP_PROJECTION_MAX_ITER = 50
_MOVING_GAP_TAIL_RTOL = 5e-12
_EDGE_REMAP_MIN_BINS = 4
_EDGE_REMAP_OCCUPATION_CEILING = 1.0 - 1e-12


def _spread_mass_change(
    mass: np.ndarray,
    capacity: np.ndarray,
    support_indices: np.ndarray,
    delta: float,
    tolerance: float,
) -> int:
    """Spread a signed mass correction over a compact edge stencil.

    At least ``_EDGE_REMAP_MIN_BINS`` above-gap bins participate when they
    exist.  The stencil expands until it has enough room (or removable mass),
    and the change is proportional to that room.  This avoids the old
    one-bin deposit, which could force the first active occupation above one.
    """
    if abs(delta) <= tolerance:
        return 0

    mass_before = float(np.sum(mass))
    need = abs(float(delta))
    add_mass = delta > 0.0
    n_support = int(support_indices.size)
    n_selected = min(_EDGE_REMAP_MIN_BINS, n_support)

    while True:
        selected = support_indices[:n_selected]
        available = capacity[selected] - mass[selected] if add_mass else mass[selected]
        available = np.maximum(available, 0.0)
        available_total = float(np.sum(available))
        if available_total + tolerance >= need or n_selected == n_support:
            break
        n_selected = min(n_support, max(n_selected + 1, 2 * n_selected))

    if available_total + tolerance < need:
        action = "store" if add_mass else "remove"
        raise RuntimeError(
            "apply_gap_update could not conservatively "
            f"{action} {need:g} of quasiparticle mass within the physical "
            "occupation bounds; refine or extend the energy grid."
        )

    if available_total > 0.0:
        fraction = min(need / available_total, 1.0)
        change = fraction * available
        mass[selected] += change if add_mass else -change

    # Close the last few ulps without violating a bound.  This also handles
    # the case where ``available_total`` and ``need`` agreed only within the
    # tolerance above.
    signed_remaining = float(delta - (np.sum(mass) - mass_before))
    remaining = abs(signed_remaining)
    if remaining > tolerance:
        correction_adds = signed_remaining > 0.0
        for idx in support_indices:
            room = capacity[idx] - mass[idx] if correction_adds else mass[idx]
            amount = min(max(float(room), 0.0), remaining)
            mass[idx] += amount if correction_adds else -amount
            remaining -= amount
            if remaining <= tolerance:
                break
    return int(np.count_nonzero(available > 0.0))


def _recover_conservative_gap_occupation(
    u_advected: np.ndarray,
    rho_cell_average_new: np.ndarray,
    dE: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    """Recover bounded ``f`` while conserving ``sum(u*dE)`` exactly.

    Numerical edge diffusion can leave mass below the new BCS support, and a
    coarse step can also overfill an above-gap cell.  Work with finite-volume
    cell masses, clip only the provisional allocation, then spread the signed
    remainder across a compact set of above-gap bins.  Returns ``(f,
    remapped_mass, n_touched)``.
    """
    u = np.asarray(u_advected, dtype=float)
    rho = np.asarray(rho_cell_average_new, dtype=float)
    widths = np.asarray(dE, dtype=float)
    cell_mass = u * widths
    if np.any(~np.isfinite(cell_mass)):
        raise RuntimeError("apply_gap_update produced non-finite spectral mass.")

    mass_scale = float(np.sum(np.abs(cell_mass)))
    tolerance = (
        256.0
        * np.finfo(float).eps
        * max(
            mass_scale,
            np.finfo(float).tiny,
        )
    )
    total_mass = float(np.sum(cell_mass))
    if total_mass < -tolerance:
        raise RuntimeError(
            "apply_gap_update produced negative total quasiparticle mass; "
            "the spectral-flow step is outside its stable regime."
        )
    total_mass = max(total_mass, 0.0)

    support = rho > _GAP_SUPPORT_EPS
    support_indices = np.flatnonzero(support)
    if support_indices.size == 0:
        if total_mass > tolerance:
            raise RuntimeError(
                "The updated gap leaves no above-gap energy bins able to "
                "hold the conserved quasiparticle mass."
            )
        return np.zeros_like(u), 0.0, 0

    capacity = np.zeros_like(cell_mass)
    capacity[support] = rho[support] * widths[support]
    capacity_total = float(np.sum(capacity[support]))
    if total_mass > capacity_total + tolerance:
        raise RuntimeError(
            "The updated energy grid has insufficient above-gap capacity to "
            "conserve quasiparticle mass with 0 <= f <= 1."
        )

    # Retain a few ulps of headroom when the global mass permits it, so a
    # local numerical overshoot is spread rather than pinning one edge bin at
    # exactly f=1.  Truly saturated states still use the physical capacity.
    allocation_capacity = capacity * _EDGE_REMAP_OCCUPATION_CEILING
    if total_mass > float(np.sum(allocation_capacity[support])) + tolerance:
        allocation_capacity = capacity

    mass_out = np.zeros_like(cell_mass)
    mass_out[support] = np.clip(
        cell_mass[support],
        0.0,
        allocation_capacity[support],
    )
    delta = total_mass - float(np.sum(mass_out))
    n_touched = _spread_mass_change(
        mass_out,
        allocation_capacity,
        support_indices,
        delta,
        tolerance,
    )

    final_mass = float(np.sum(mass_out))
    residual = total_mass - final_mass
    if abs(residual) > tolerance:
        raise RuntimeError(
            f"apply_gap_update conservative remap left a mass residual of {residual:g}."
        )

    f_new = np.zeros_like(u)
    f_new[support] = mass_out[support] / capacity[support]
    if np.any(f_new < -1e-13) or np.any(f_new > 1.0 + 1e-13):
        raise RuntimeError("apply_gap_update conservative remap violated occupation bounds.")
    f_new = np.clip(f_new, 0.0, 1.0)
    remapped_mass = 0.5 * float(np.sum(np.abs(mass_out - cell_mass)))
    return f_new, remapped_mass, n_touched


def _remap_bcs_frozen_xi_cell_mass(
    f_old: np.ndarray,
    E: np.ndarray,
    dE: np.ndarray,
    old_gap: float,
    new_gap: float,
) -> tuple[np.ndarray, float]:
    r"""Map pure-BCS finite-volume mass exactly along frozen-``xi`` shells.

    For ``E = sqrt(xi^2 + Delta^2)``, the ideal BCS measure obeys
    ``rho(E) dE = dxi`` and gap motion keeps ``xi`` fixed. Treating ``f`` as
    cell-constant therefore turns the exact characteristic update into a
    conservative overlap of the old and new ``xi`` cells. The returned array
    is the quasiparticle mass in each new energy cell; the scalar is old mass
    whose characteristic exits through the finite ``E_max`` boundary.
    """
    occupations = np.asarray(f_old, dtype=float)
    energies = np.asarray(E, dtype=float)
    widths = np.asarray(dE, dtype=float)
    if occupations.shape != energies.shape or widths.shape != energies.shape:
        raise ValueError("f_old, E, and dE must have identical one-dimensional shapes.")
    if np.any(~np.isfinite(occupations)) or np.any((occupations < 0.0) | (occupations > 1.0)):
        raise ValueError("f_old must contain finite occupations in [0, 1].")

    cell_edges = cell_edges_from_widths(energies, widths)

    def xi_edges(gap: float) -> np.ndarray:
        xi = np.zeros_like(cell_edges)
        above = cell_edges > gap
        xi[above] = np.sqrt((cell_edges[above] - gap) * (cell_edges[above] + gap))
        return xi

    old_xi = xi_edges(old_gap)
    new_xi = xi_edges(new_gap)
    old_widths = np.diff(old_xi)
    new_widths = np.diff(new_xi)
    new_mass = np.zeros_like(occupations)

    scale = max(float(old_xi[-1]), float(new_xi[-1]), 1.0)
    tolerance = 128.0 * np.finfo(float).eps * scale
    old_index = 0
    new_index = 0
    n_cells = occupations.size
    while old_index < n_cells and new_index < n_cells:
        old_lo = old_xi[old_index]
        old_hi = old_xi[old_index + 1]
        new_lo = new_xi[new_index]
        new_hi = new_xi[new_index + 1]

        if old_hi <= old_lo + tolerance:
            old_index += 1
            continue
        if new_hi <= new_lo + tolerance:
            new_index += 1
            continue

        overlap = min(old_hi, new_hi) - max(old_lo, new_lo)
        if overlap > tolerance:
            new_mass[new_index] += occupations[old_index] * overlap

        if old_hi <= new_hi + tolerance:
            old_index += 1
        if new_hi <= old_hi + tolerance:
            new_index += 1

    old_mass = float(np.sum(old_widths * occupations))
    mapped_mass = float(np.sum(new_mass))
    mass_tolerance = (
        256.0
        * np.finfo(float).eps
        * max(
            abs(old_mass),
            np.finfo(float).tiny,
        )
    )
    escaped_mass = old_mass - mapped_mass
    if escaped_mass < -mass_tolerance:
        raise RuntimeError(
            f"Frozen-xi gap remap created quasiparticle mass; signed excess {-escaped_mass:g}."
        )
    escaped_mass = max(escaped_mass, 0.0)

    # ``new_widths`` is intentionally evaluated even though the caller also
    # has analytic BCS weights: asserting the identity here protects this
    # characteristic remap from drifting away from that shared measure.
    expected_new_widths = bcs_dos_cell_weights(energies, widths, new_gap)
    if not np.allclose(new_widths, expected_new_widths, rtol=5e-14, atol=tolerance):
        raise RuntimeError("Frozen-xi cell widths disagree with BCS DOS weights.")
    return new_mass, escaped_mass


@dataclass(frozen=True)
class _PersistentGapCoordinates:
    r"""Authoritative moving-gap occupation on fixed material ``xi`` cells.

    ``public_f`` and the grid/gap snapshots are synchronization sentinels.  A
    caller is free to mutate or replace the public state; if that happens, the
    next moving-gap step deliberately rebuilds material coordinates instead of
    reusing stale hidden data.  Arrays are copied and made read-only so rejected
    ETD trials cannot mutate an accepted state.
    """

    xi_edges: np.ndarray
    occupation: np.ndarray
    energy_grid: np.ndarray
    energy_widths: np.ndarray
    synchronized_gap: float
    public_f: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "xi_edges": self.xi_edges,
            "occupation": self.occupation,
            "energy_grid": self.energy_grid,
            "energy_widths": self.energy_widths,
            "public_f": self.public_f,
        }
        for name, value in arrays.items():
            copied = np.asarray(value, dtype=float).copy()
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)
        if self.xi_edges.ndim != 1 or self.occupation.ndim != 1:
            raise ValueError("Persistent moving-gap coordinates must be one-dimensional.")
        if self.xi_edges.size != self.occupation.size + 1:
            raise ValueError("xi_edges must contain one more entry than occupation.")
        if np.any(~np.isfinite(self.xi_edges)) or np.any(np.diff(self.xi_edges) <= 0.0):
            raise ValueError("xi_edges must be finite and strictly increasing.")
        if np.any(~np.isfinite(self.occupation)) or np.any(
            (self.occupation < 0.0) | (self.occupation > 1.0)
        ):
            raise ValueError("Persistent occupations must be finite and lie in [0, 1].")
        if (
            self.energy_grid.ndim != 1
            or self.energy_widths.shape != self.energy_grid.shape
            or self.public_f.shape != self.energy_grid.shape
        ):
            raise ValueError("Persistent energy snapshots must have one matching shape.")
        if (
            self.energy_grid.size == 0
            or np.any(~np.isfinite(self.energy_grid))
            or np.any(np.diff(self.energy_grid) <= 0.0)
            or np.any(~np.isfinite(self.energy_widths))
            or np.any(self.energy_widths <= 0.0)
        ):
            raise ValueError(
                "Persistent energy snapshots require a finite increasing grid "
                "and positive finite widths."
            )
        if np.any(~np.isfinite(self.public_f)) or np.any(
            (self.public_f < 0.0) | (self.public_f > 1.0)
        ):
            raise ValueError("The synchronized public occupation must lie in [0, 1].")
        if not np.isfinite(self.synchronized_gap) or self.synchronized_gap <= 0.0:
            raise ValueError("The synchronized moving gap must be finite and positive.")


def _bcs_xi_edges(cell_edges: np.ndarray, gap: float) -> np.ndarray:
    r"""Map fixed energy faces to ``xi=sqrt(E**2-gap**2)``."""
    edges = np.asarray(cell_edges, dtype=float)
    xi = np.zeros_like(edges)
    above = edges > gap
    xi[above] = np.sqrt((edges[above] - gap) * (edges[above] + gap))
    return xi


def _material_overlap(
    xi_edges: np.ndarray,
    energy_xi_edges: np.ndarray,
) -> np.ndarray:
    """Exact ``d xi`` overlap from persistent rows to energy-cell columns."""
    left = np.maximum(xi_edges[:-1, None], energy_xi_edges[None, :-1])
    right = np.minimum(xi_edges[1:, None], energy_xi_edges[None, 1:])
    return np.maximum(right - left, 0.0)


@dataclass
class T3DiffusionState:
    """State carried by the T3 diffusion backend.

    For v1 (Gate 2) the gap is a scalar and ``f`` is 1D over energy
    only. Multi-material / spatially-varying gaps add a ``GapState``
    (NFP §4.1) with per-cell ``SpectralContext`` slots; that's a
    future-gate extension.

    Attributes
    ----------
    f
        QP occupation on the energy grid, shape ``(NE,)``.
    gap
        Scalar gap ``Δ`` (μeV).
    spectral
        Gap-dependent cache (DOS, K±, ``D(E)``). Must match ``gap``.
    phonon
        Phonon state (n_ph, τ_l, model, branches). For v1 single-branch
        Ph0 the first entry of ``phonon.tau_l`` is used as the scalar
        bath escape time passed into the Picard path.
    material
        Source of ``T_c``, ``τ_0``, and other material parameters used
        by the kernel builders.
    T_bath
        Substrate bath temperature in K.
    tier
        Always :attr:`Tier.T3_DIFFUSION`; included for downstream code
        that branches on the tier enum.

    Notes
    -----
    Accepted moving-gap steps also carry a private, read-only material-grid
    synchronization record. It is intentionally excluded from ``repr`` and
    equality and is invalidated automatically if the public state is edited.
    """

    f: np.ndarray
    gap: float
    spectral: SpectralContext
    phonon: PhononState
    material: Material
    T_bath: float
    tier: Tier = Tier.T3_DIFFUSION
    _moving_gap_coordinates: _PersistentGapCoordinates | None = field(
        default=None,
        repr=False,
        compare=False,
    )


class T3DiffusionBackend:
    """Homogeneous steady-state and transient solver for the T3 tier.

    Backend instances are stateless and reusable. Accepted moving-gap material
    coordinates belong to :class:`T3DiffusionState`, not to this object.
    """

    def steady_state(
        self,
        state: T3DiffusionState,
        *,
        method: str = "picard",
        self_consistent_gap: bool = False,
        use_thermal_phonons: bool = False,
        external_dissipation_only: bool = False,
        use_phonon_side_kernel: bool = True,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
        newton_tol: float = 1e-14,
        newton_backward_error_tol: float = 1e-6,
        newton_max_iter: int = 200,
        picard_tol: float = 1e-10,
        picard_atol: float = 1e-11,
        picard_balance_tol: float | None = None,
        picard_max_iter: int = 200,
        picard_mixing: float = 0.3,
        anderson_depth: int = 0,
        coupled_newton_tol: float = 1e-10,
        coupled_newton_step_rtol: float = 1e-8,
        coupled_newton_max_iter: int = 50,
        coupled_newton_fd_step: float = 1e-8,
        coupled_newton_analytic_cross: bool = False,
        gap_tol: float = 1e-6,
        gap_max_iter: int = 20,
        gap_under_relaxation: float = 0.5,
        gap_solve_xtol: float | None = None,
    ) -> T3DiffusionState:
        """Solve for the steady-state ``f(E)`` and return an updated state.

        Rebuilds the e-ph kernels from the current
        :class:`SpectralContext` and Material, and dispatches to the
        requested solver. The returned state has both ``f`` and
        ``phonon`` updated; ``phonon`` is rebuilt on the physics ``ω``
        grid.

        Gate 2 scope requires the input ``state.phonon`` to have
        ``n_branch = 1``, ``n_spatial = 1``, and constant
        ``τ_l`` (all entries equal). Violations raise ``ValueError``
        rather than silently mis-solving.

        Parameters
        ----------
        state
            Initial T3 state; ``state.f`` is the Newton/Picard initial
            guess.
        method
            ``"picard"`` (default) — the Newton + Picard orchestrator
            from :func:`qpsim.services.steady_state.solve_steady_state`.
            ``"coupled_newton"`` — the monolithic coupled Newton from
            :func:`qpsim.solvers.coupled_newton.coupled_newton_solve`.
            Use coupled Newton for the strong-bottleneck regime
            (``τ_l/τ_PB ≳ 10``) where Picard + Anderson stalls.
            Ignored when ``use_thermal_phonons=True``.
        self_consistent_gap
            When ``True``, wraps the fixed-gap steady-state solve in an
            outer fixed-point iteration on ``Δ``. Each outer step solves
            ``f`` (and, if applicable, ``n_ph``) at the current gap,
            updates ``Δ`` via :func:`qpsim.physics.gap_equation.solve_gap`,
            and repeats until the relative gap change is below
            ``gap_tol``. Convergence is accepted only on a state whose
            fixed-gap kinetic solve has already completed and whose own
            branch-anchored gap-map residual passes the tolerance; no
            unchecked post-convergence kinetic polish is returned.
        use_thermal_phonons
            ``True`` pins ``n_ph`` at the substrate Bose-Einstein
            distribution and runs Newton-only on ``f`` (no Picard, no
            coupled system). This is Fischer's ``τ_l → 0`` limit —
            physically, the thermalization timescale is instantaneous
            so the phonon field is always at the bath. Mutually
            exclusive with ``method="coupled_newton"``.

            Caution: this flag is the ONLY spelling of the bath-pinned
            limit. Setting ``state.phonon.tau_l`` to ``0.0`` does NOT
            mean the same thing — the Picard/coupled paths forward that
            scalar to :func:`qpsim.phonon_models.ph0_local.phonon_steady_state`,
            where ``0.0`` is the no-substrate-coupling sentinel (the
            opposite, ``τ_l → ∞``, limit of the escape term).
        external_dissipation_only
            ``True`` disables the e-ph scattering and recombination
            kernels for this solve, so the only source/sink of f(E)
            is the supplied ``external_flux``. Used by Layer-2
            Junctions (e.g. :class:`M25GapAsymmetricJJ`) that own the
            dissipation at the moment level — running the e-ph kernel
            alongside would double-count and dwarf the moment-level
            external flux. Requires ``external_flux`` to be non-None
            (otherwise nothing constrains f). Mutually exclusive with
            ``self_consistent_gap=True`` (the gap equation depends on
            e-ph occupations).
        use_phonon_side_kernel
            F&C 2023 Eq. 12 phonon-side kernel for the phonon-equation
            rate (the default). When ``True``, builds a sibling
            ``K_s0_phonon_side = 2K⁻/(π Δ τ_0^PB)`` and
            ``K_r0_phonon_side = K⁺/(π Δ τ_0^PB)`` from
            ``state.material.tau_0_pb_ns`` (via
            :func:`qpsim.collisions.phonon.build_scattering_kernel_phonon_side`
            and
            :func:`qpsim.collisions.phonon.build_recombination_kernel_phonon_side`)
            and forwards it through to
            :func:`qpsim.phonon_models.ph0_local.phonon_steady_state`
            (Picard path) and :func:`coupled_newton_solve`. The
            QP-equation residual continues to use the QP-side ``K_r0``
            with its ``(E_sum/k_BT_c)²/(τ₀ k_BT_c)`` prefactor — the
            two kernels are physically distinct (Eqs. 10/11 vs Eq. 12).
            Requires ``state.material.tau_0_pb_ns`` to be set;
            otherwise raises ``ValueError``. Ignored when
            ``use_thermal_phonons=True`` or
            ``external_dissipation_only=True``. Set ``False`` only to
            reproduce the legacy behavior that reused the QP-side kernel in
            the phonon equation.
        photon_params, pb_photon_params
            Optional photon channel dicts.
        newton_tol, newton_backward_error_tol, newton_max_iter
            Inner Newton dimensional residual tolerance, scale-independent
            gain/loss backward-error limit, and iteration cap (used by Picard
            and thermal-phonon paths).
        picard_tol, picard_atol, picard_balance_tol, picard_max_iter,
        picard_mixing, anderson_depth
            Picard path controls. ``picard_atol`` is an absolute tolerance on
            dimensionless phonon occupation, independent of ``newton_tol``.
            ``picard_balance_tol`` is the normwise physical phonon-balance
            limit; ``None`` derives ``max(10*picard_tol, 1e-6)``.
        coupled_newton_tol, coupled_newton_step_rtol,
        coupled_newton_max_iter, coupled_newton_fd_step
            Coupled-Newton path controls. ``coupled_newton_step_rtol`` defaults
            to a scale-aware step plus gain/loss-balance certificate; pass
            ``0.0`` only to reproduce the legacy absolute-residual exit.
        gap_tol, gap_max_iter, gap_under_relaxation
            Outer self-consistent-gap loop controls. ``gap_tol`` applies to
            the unrelaxed fixed-point residual; under-relaxation changes the
            step size but cannot weaken the convergence criterion. Ignored
            when ``self_consistent_gap=False``.
        """
        gap_scale = max(abs(float(state.gap)), abs(float(state.spectral.gap)), 1.0)
        if not np.isclose(
            state.gap,
            state.spectral.gap,
            rtol=_GAP_STATE_MATCH_RTOL,
            atol=_GAP_STATE_MATCH_RTOL * gap_scale,
        ):
            raise ValueError(
                "state.gap and state.spectral.gap must match before a "
                "steady-state solve; got "
                f"{state.gap:g} and {state.spectral.gap:g}."
            )

        dynes_collision_enabled = (
            not external_dissipation_only
            or photon_params is not None
            or pb_photon_params is not None
        )
        if state.spectral.dynes_gamma > 0.0 and dynes_collision_enabled:
            raise ValueError(
                "Dynes-broadened collision solves are not supported: the "
                "current electron-phonon and photon coherence kernels are "
                "ideal-BCS while the density of states is broadened. Set "
                "dynes_gamma=0 until consistent broadened kernels are implemented."
            )

        if use_thermal_phonons and method == "coupled_newton":
            raise ValueError(
                "use_thermal_phonons=True pins n_ph at Bose-Einstein, so the "
                "(f, n_ph) system reduces to Newton-on-f alone; "
                "method='coupled_newton' has nothing to solve for. "
                "Use method='picard' (default) with use_thermal_phonons=True."
            )

        if external_dissipation_only:
            if external_flux is None:
                raise ValueError(
                    "external_dissipation_only=True disables the e-ph kernels, "
                    "so the supplied external_flux is the sole source/sink of "
                    "f(E). external_flux=None leaves f unconstrained — pass an "
                    "ExternalFlux."
                )
            if self_consistent_gap:
                raise ValueError(
                    "external_dissipation_only=True is incompatible with "
                    "self_consistent_gap=True: the gap equation depends on "
                    "the e-ph-driven occupation. Solve at fixed gap, or let "
                    "the e-ph kernel run."
                )
            if method == "coupled_newton":
                raise ValueError(
                    "external_dissipation_only=True turns off e-ph kernels, "
                    "so there are no phonon dynamics to couple. Use "
                    "method='picard' with use_thermal_phonons=True."
                )
            if not use_thermal_phonons:
                raise ValueError(
                    "external_dissipation_only=True turns off the e-ph "
                    "kernels, so the phonon Picard loop has nothing to "
                    "drive n_ph away from thermal. Pass "
                    "use_thermal_phonons=True to make the contract explicit."
                )

        # Validate flux shape FIRST so basic contract errors raise the
        # clearer "sized for {M} energy bins" message rather than getting
        # masked by the Picard-stability guard below.
        if external_flux is not None:
            external_flux._validate_for_NE(int(state.spectral.E.size))
            external_flux._validate_gain_support(state.spectral.active_mask)

        # Default unaccelerated Picard (anderson_depth=0) is brittle when
        # ANY perturbation feeds through the phonon-emission cycle —
        # ExternalFlux on the f-equation reliably pushes it into oscillating
        # non-convergence. The kwarg threading IS correct; the issue is the
        # underlying Picard scheme (which is also why coupled_newton was
        # added). Catch this at the API boundary so users get a clear
        # routing hint instead of a "did not converge in 200 iterations"
        # surprise after the fact.
        if (
            external_flux is not None
            and method == "picard"
            and not use_thermal_phonons
            and anderson_depth == 0
        ):
            ef_max = float(np.max(external_flux.gain) + np.max(external_flux.loss_rate))
            if ef_max > 0.0:
                raise ValueError(
                    "Default unaccelerated Picard (method='picard', "
                    "anderson_depth=0) is unstable when external_flux is "
                    "non-zero — the f-perturbation feeds through the "
                    "phonon-emission cycle and Picard oscillates. "
                    "Pass anderson_depth >= 1 to stabilize, OR switch to "
                    "method='coupled_newton', OR use_thermal_phonons=True "
                    "if the τ_l → 0 limit applies. Pre-existing Picard "
                    "stability concern; not a kwarg-threading bug."
                )

        if not self_consistent_gap:
            return self._steady_state_fixed_gap(
                state,
                method=method,
                use_thermal_phonons=use_thermal_phonons,
                external_dissipation_only=external_dissipation_only,
                use_phonon_side_kernel=use_phonon_side_kernel,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                newton_tol=newton_tol,
                newton_backward_error_tol=newton_backward_error_tol,
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_balance_tol=picard_balance_tol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_step_rtol=coupled_newton_step_rtol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
                coupled_newton_analytic_cross=coupled_newton_analytic_cross,
            )

        if not np.isfinite(gap_tol) or gap_tol <= 0.0:
            raise ValueError("gap_tol must be finite and positive.")
        if (
            isinstance(gap_max_iter, (bool, np.bool_))
            or not isinstance(gap_max_iter, (int, np.integer))
            or gap_max_iter <= 0
        ):
            raise ValueError("gap_max_iter must be a positive integer.")
        if (
            not np.isfinite(gap_under_relaxation)
            or not 0.0 < gap_under_relaxation <= 1.0
        ):
            raise ValueError(
                "gap_under_relaxation must be finite and lie in the interval (0, 1]."
            )

        if state.material.T_c <= 0:
            raise ValueError("state.material.T_c must be positive for self-consistent-gap solves.")
        if state.T_bath >= state.material.T_c:
            raise ValueError(
                "self-consistent-gap steady state requires T_bath < T_c; "
                f"got T_bath={state.T_bath} K and T_c={state.material.T_c} K."
            )

        calibration = calibrate_gap(
            T_c=state.material.T_c,
            T_bath=state.T_bath,
            Delta_0=state.material.Delta_0,
        )
        current = state
        last_rel_change = float("inf")

        for iteration in range(1, gap_max_iter + 1):
            solved = self._steady_state_fixed_gap(
                current,
                method=method,
                use_thermal_phonons=use_thermal_phonons,
                use_phonon_side_kernel=use_phonon_side_kernel,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                newton_tol=newton_tol,
                newton_backward_error_tol=newton_backward_error_tol,
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_balance_tol=picard_balance_tol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_step_rtol=coupled_newton_step_rtol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
                coupled_newton_analytic_cross=coupled_newton_analytic_cross,
            )

            try:
                delta_raw = solve_gap(
                    calibration,
                    solved.f,
                    solved.spectral.E,
                    dE_bins=solved.spectral.dE,
                    reference_gap=solved.gap,
                    xtol=gap_solve_xtol,
                )
            except GapBelowGridSupportError as exc:
                # The solved gap (root or normal-state decision) fell below
                # the represented grid support. On every shipped BCS grid the
                # first cell face is far above zero, so this — not a literal
                # <= 0 return — is the reachable form of superconducting
                # collapse (2026-07-19 audit: the <= 0 branch below was dead
                # code on physical grids and a genuine collapse aborted the
                # sweep unclassified).
                raise SelfConsistentGapCollapseError(
                    iteration=iteration,
                    max_occupation=float(solved.f.max()),
                ) from exc
            if delta_raw <= 0.0:
                # The current occupation no longer supports a superconducting
                # solution; collapse to the normal state directly. Under-relaxing
                # zero against the previous gap would have drifted us to a
                # spurious tiny-Δ "almost-superconducting" fixed point.
                raise SelfConsistentGapCollapseError(
                    iteration=iteration,
                    max_occupation=float(solved.f.max()),
                )

            cell_edges = cell_edges_from_widths(
                solved.spectral.E,
                solved.spectral.dE,
            )
            lower_edge = float(cell_edges[0])
            support_tolerance = 64.0 * np.finfo(float).eps * max(
                abs(delta_raw),
                abs(lower_edge),
                1.0,
            )
            if delta_raw < lower_edge - support_tolerance:
                raise ValueError(
                    "Self-consistent gap solve returned a candidate below the "
                    "first represented energy-cell face: "
                    f"Delta={delta_raw:.12g} micro-eV, "
                    f"E_face_min={lower_edge:.12g} micro-eV. The gap equation "
                    "would depend on extrapolating f through unsampled BCS "
                    "gap-edge support; extend the energy grid below every "
                    "candidate gap."
                )

            # Test the gap equation itself, before under-relaxation. Testing
            # the relaxed update would scale the residual by the relaxation
            # factor and could accept an arbitrarily poor fixed point.  Return
            # ``solved`` itself when this passes: its kinetic equation was
            # solved at ``solved.gap`` and this exact occupation produced the
            # checked, branch-anchored ``delta_raw`` above.
            last_rel_change = abs(delta_raw - solved.gap) / max(
                abs(solved.gap),
                1e-30,
            )

            if last_rel_change < gap_tol:
                return solved

            relaxed_delta = (
                1.0 - gap_under_relaxation
            ) * solved.gap + gap_under_relaxation * delta_raw

            current = replace(
                solved,
                gap=relaxed_delta,
                spectral=self._rebuild_spectral_context(
                    solved.spectral,
                    new_gap=relaxed_delta,
                ),
            )

        raise RuntimeError(
            "Self-consistent gap iteration did not converge in "
            f"{gap_max_iter} iterations. Final |Δ_raw - Δ| / Δ = "
            f"{last_rel_change:.2e}."
        )

    def _steady_state_fixed_gap(
        self,
        state: T3DiffusionState,
        *,
        method: str,
        use_thermal_phonons: bool,
        external_dissipation_only: bool = False,
        use_phonon_side_kernel: bool = True,
        photon_params: dict[str, float] | None,
        pb_photon_params: dict[str, float] | None,
        external_flux: ExternalFlux | None,
        newton_tol: float,
        newton_backward_error_tol: float,
        newton_max_iter: int,
        picard_tol: float,
        picard_atol: float,
        picard_balance_tol: float | None,
        picard_max_iter: int,
        picard_mixing: float,
        anderson_depth: int,
        coupled_newton_tol: float,
        coupled_newton_step_rtol: float,
        coupled_newton_max_iter: int,
        coupled_newton_fd_step: float,
        coupled_newton_analytic_cross: bool,
    ) -> T3DiffusionState:
        """Inner steady-state solve at fixed ``Δ``."""
        self._validate_gate2_scope(state.phonon)

        K_s0: np.ndarray | None
        K_r0: np.ndarray | None
        K_s0_phonon_side: np.ndarray | None = None
        K_r0_phonon_side: np.ndarray | None = None
        if external_dissipation_only:
            # external_flux owns dissipation — kill the e-ph kernels so
            # they don't double-count. Both nones short-circuit the
            # phonon-occupation defaults inside newton_solve_f.
            K_s0 = None
            K_r0 = None
        else:
            K_s0 = build_scattering_kernel_base(
                state.spectral,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            K_r0 = build_recombination_kernel_base(
                state.spectral,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            if use_phonon_side_kernel and not use_thermal_phonons:
                # F&C 2023 Eq. 12 phonon-side kernel for the
                # phonon-equation rate. Built here so the same matrix
                # is shared by Picard (via solve_steady_state →
                # phonon_steady_state) and coupled-Newton paths.
                tau_0_pb_ns = state.material.tau_0_pb_ns
                if tau_0_pb_ns is None or not np.isfinite(tau_0_pb_ns) or tau_0_pb_ns <= 0.0:
                    raise ValueError(
                        "Dynamic Ph0 with use_phonon_side_kernel=True requires "
                        "state.material.tau_0_pb_ns to be finite and positive; "
                        f"got {tau_0_pb_ns!r}. "
                        "Set τ_0^PB on the Material (e.g. via the YAML "
                        "database key 'tau_0_pb_ns'). Pass "
                        "use_phonon_side_kernel=False only to reproduce the "
                        "legacy QP-side-kernel behavior."
                    )
                K_r0_phonon_side = build_recombination_kernel_phonon_side(
                    state.spectral,
                    tau_0_pb_ns=tau_0_pb_ns,
                )
                K_s0_phonon_side = build_scattering_kernel_phonon_side(
                    state.spectral,
                    tau_0_pb_ns=tau_0_pb_ns,
                )

        tau_l_scalar = float(state.phonon.tau_l[0, 0])

        if use_thermal_phonons:
            # Newton-only shortcut: solve_steady_state routes to newton_solve_f
            # when phonon_escape_time=None, with n_ph held at Bose-Einstein.
            f_new = solve_steady_state(
                state.spectral,
                K_s0,
                K_r0,
                state.T_bath,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                initial_guess=state.f,
                tol=newton_tol,
                newton_backward_error_tol=newton_backward_error_tol,
                max_iter=newton_max_iter,
                phonon_escape_time=None,
            )
            omega_conv, _, _, _ = build_phonon_frequency_map(state.spectral.E)
            n_ph_conv = thermal_phonon_occupation(omega_conv, state.T_bath)
        elif method == "picard":
            omega_seed, _, _, _ = build_phonon_frequency_map(state.spectral.E)
            initial_phonon_guess: np.ndarray | None = None
            if (
                state.phonon.omega_bins.shape == (1, omega_seed.size)
                and np.array_equal(state.phonon.omega_bins[0], omega_seed)
            ):
                initial_phonon_guess = state.phonon.n_ph[0, :, 0].copy()
            phonon_out: dict[str, np.ndarray] = {}
            f_new = solve_steady_state(
                state.spectral,
                K_s0,
                K_r0,
                state.T_bath,
                K_r0_phonon_side=K_r0_phonon_side,
                K_s0_phonon_side=K_s0_phonon_side,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                initial_guess=state.f,
                initial_phonon_guess=initial_phonon_guess,
                tol=newton_tol,
                newton_backward_error_tol=newton_backward_error_tol,
                max_iter=newton_max_iter,
                phonon_escape_time=tau_l_scalar,
                max_picard_iter=picard_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_balance_tol=picard_balance_tol,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                phonon_out=phonon_out,
            )
            n_ph_conv = phonon_out["n_ph"]
            omega_conv = phonon_out["omega_bins"]
        elif method == "coupled_newton":
            omega_conv, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(state.spectral.E)
            # Seed n_ph from state if it's already on the physics grid,
            # else start from thermal.
            if state.phonon.omega_bins.shape == (1, omega_conv.size) and np.array_equal(
                state.phonon.omega_bins[0], omega_conv
            ):
                n_ph_init = state.phonon.n_ph[0, :, 0].copy()
            else:
                n_ph_init = thermal_phonon_occupation(omega_conv, state.T_bath)
            f_new, n_ph_conv = coupled_newton_solve(
                state.spectral,
                state.f,
                n_ph_init,
                omega_bins=omega_conv,
                omega_idx_diff=idx_diff,
                omega_idx_sum=idx_sum,
                diff_sign=diff_sign,
                K_s0=K_s0,
                K_r0=K_r0,
                K_s0_phonon_side=K_s0_phonon_side,
                K_r0_phonon_side=K_r0_phonon_side,
                T_bath=state.T_bath,
                tau_l=tau_l_scalar,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                tol=coupled_newton_tol,
                step_rtol=coupled_newton_step_rtol,
                max_iter=coupled_newton_max_iter,
                fd_step=coupled_newton_fd_step,
                analytic_cross=coupled_newton_analytic_cross,
            )
        else:
            raise ValueError(f"Unknown method {method!r}. Use 'picard' or 'coupled_newton'.")

        new_phonon = replace(
            state.phonon,
            n_ph=n_ph_conv.reshape(1, -1, 1),
            omega_bins=omega_conv.reshape(1, -1),
            tau_l=np.full((1, omega_conv.size), tau_l_scalar),
        )
        return replace(state, f=f_new, phonon=new_phonon)

    @staticmethod
    def _rebuild_spectral_context(
        spectral: SpectralContext,
        *,
        new_gap: float,
    ) -> SpectralContext:
        """Clone ``spectral`` at a new gap, preserving all non-gap config."""
        return SpectralContext(
            E_bins=spectral.E,
            dE_bins=spectral.dE,
            gap=new_gap,
            dynes_gamma=spectral.dynes_gamma,
            diffusion_coefficient=spectral.diffusion_coefficient,
            rebuild_tolerance=spectral.rebuild_tolerance,
            active_margin_factor=spectral.active_margin_factor,
        )

    def apply_collisions(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
        _diagnostics: dict[str, object] | None = None,
    ) -> T3DiffusionState:
        """One ETD2 collision substep on ``f`` with ``n_ph`` frozen.

        Builds the e-ph kernels + the phonon occupation matrices from
        ``state.phonon.n_ph``, wraps optional photon channels into the
        ``rhs`` closure, and runs
        :func:`qpsim.solvers.etd.etd2_step`. Returns a new state with
        updated ``f``; the phonon field is unchanged (transient phonon
        dynamics are out of Gate 2 scope — for coupled ``(f, n_ph)``
        steady state use :func:`steady_state` or
        :func:`qpsim.solvers.coupled_newton.coupled_newton_solve`).
        """
        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Dynes-broadened transient collisions are not supported: the "
                "current electron-phonon and photon coherence kernels are "
                "ideal-BCS while the density of states is broadened. Set "
                "dynes_gamma=0 until consistent broadened kernels are implemented."
            )

        self._validate_gate2_scope(state.phonon)
        self._validate_phonon_on_physics_grid(state)

        # Fail-loud input guards, matching apply_gap_update and the spatial
        # backend's apply_collisions (which already validate dt): a NaN /
        # non-positive dt or a NaN state would otherwise propagate silently, and
        # a state whose gap disagrees with its spectral gap would run kernels
        # built from spectral.gap while ignoring state.gap.
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"apply_collisions requires a finite positive dt; got {dt}.")
        f_state = np.asarray(state.f, dtype=float)
        if f_state.shape != state.spectral.E.shape:
            raise ValueError(
                "state.f must have the same one-dimensional shape as the "
                "spectral energy grid."
            )
        if np.any(~np.isfinite(f_state)) or np.any(
            (f_state < 0.0) | (f_state > 1.0)
        ):
            raise ValueError("state.f must contain finite occupations in [0, 1].")
        gap_scale = max(abs(float(state.gap)), abs(float(state.spectral.gap)), 1.0)
        if not np.isclose(
            state.gap,
            state.spectral.gap,
            rtol=_GAP_STATE_MATCH_RTOL,
            atol=_GAP_STATE_MATCH_RTOL * gap_scale,
        ):
            raise ValueError(
                "state.gap and state.spectral.gap must match; got "
                f"{state.gap:g} and {state.spectral.gap:g}."
            )

        if external_flux is not None:
            external_flux._validate_for_NE(int(state.spectral.E.size))
            external_flux._validate_gain_support(state.spectral.active_mask)

        K_s0 = build_scattering_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        K_r0 = build_recombination_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )

        _, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(state.spectral.E)
        n_ph_1d = state.phonon.n_ph[0, :, 0]
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph_1d,
            idx_diff,
            idx_sum,
            diff_sign,
        )

        def rhs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain, loss = phonon_collision_rates(
                f,
                state.spectral,
                K_s0,
                K_r0,
                state.T_bath,
                N_p_override=N_p,
                N_emit_override=N_emit,
                N_abs_override=N_abs,
            )
            if photon_params is not None:
                gp, lp = sub_gap_photon_collision_rates(
                    f,
                    state.spectral,
                    photon_params["omega_0"],
                    photon_params["n_bar"],
                    photon_params["c_phot"],
                )
                gain = gain + gp
                loss = loss + lp
            if pb_photon_params is not None:
                gp, lp = pair_breaking_photon_collision_rates(
                    f,
                    state.spectral,
                    pb_photon_params["omega_PB"],
                    pb_photon_params["n_bar_PB"],
                    pb_photon_params["c_phot_PB"],
                )
                gain = gain + gp
                loss = loss + lp
            if external_flux is not None:
                gain = gain + external_flux.gain
                loss = loss + external_flux.loss_rate
            unsupported = ~state.spectral.active_mask
            gain[unsupported] = 0.0
            loss[unsupported] = 0.0
            return gain, loss

        etd_diagnostics: dict[str, int] = {}
        f_new = etd2_step(
            f_state,
            rhs,
            dt,
            balance_weights=state.spectral.cell_weights,
            max_loss_step=0.25,
            _diagnostics=etd_diagnostics,
        )
        if _diagnostics is not None:
            # Rejected nonlinear predictor trials add RHS evaluations but are
            # not completed integration work. Forward the stepper's exact
            # accepted-substep diagnostic instead of inferring it from calls.
            accepted_substeps = etd_diagnostics.get("accepted_substeps")
            if not isinstance(accepted_substeps, int) or accepted_substeps < 1:
                raise RuntimeError("ETD2 returned an invalid accepted-step count.")
            _diagnostics["etd_substeps"] = accepted_substeps
            if bool(_diagnostics.get("evaluate_residual", False)):
                # Evaluate the endpoint once more using the same built kernels,
                # frozen phonon matrices, and caller-sampled flux. This gives
                # early stopping an unclipped residual without rebuilding
                # O(NE^2) operators or resampling a time callback.
                gain_end, loss_end = rhs(f_new)
                _diagnostics["raw_collision_rate"] = gain_end - loss_end * f_new
        return replace(state, f=f_new)

    def apply_collisions_with_diagnostics(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
        evaluate_residual: bool = False,
    ) -> tuple[T3DiffusionState, np.ndarray | None, int]:
        """Advance once and report raw residual plus internal ETD substeps.

        ``evaluate_residual=False`` records rate-limited ETD2 work without an
        extra RHS evaluation. When true, the returned residual uses the same
        frozen operator and external-flux sample as the step. Custom backends
        may implement this protocol; otherwise the transient driver counts one
        opaque backend step and uses a conservative stopping fallback.
        """
        diagnostics: dict[str, object] = {
            "evaluate_residual": bool(evaluate_residual),
        }
        updated = self.apply_collisions(
            state,
            dt,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            external_flux=external_flux,
            _diagnostics=diagnostics,
        )
        raw_rate = diagnostics.get("raw_collision_rate")
        if raw_rate is not None and not isinstance(raw_rate, np.ndarray):
            raise RuntimeError("Collision diagnostics returned an invalid residual.")
        substeps = diagnostics.get("etd_substeps")
        if not isinstance(substeps, int) or substeps < 1:
            raise RuntimeError("Collision diagnostics returned an invalid step count.")
        return updated, raw_rate, substeps

    def apply_collisions_with_residual(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
    ) -> tuple[T3DiffusionState, np.ndarray]:
        """Advance once and return the raw endpoint collision residual.

        This is the diagnostics protocol used by the transient service when
        ``stop_tol`` is enabled.  The residual is evaluated with the exact
        same frozen operator and external-flux sample as the ETD2 step.
        Custom backends may implement the same method to provide a trustworthy
        stopping certificate; otherwise the service uses a conservative
        finite-difference fallback.
        """
        updated, raw_rate, _substeps = self.apply_collisions_with_diagnostics(
            state,
            dt,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            external_flux=external_flux,
            evaluate_residual=True,
        )
        if raw_rate is None:
            raise RuntimeError("Collision diagnostics did not return a raw residual.")
        return updated, raw_rate

    def apply_transport(
        self,
        state: T3DiffusionState,
        dt: float,
    ) -> T3DiffusionState:
        """QP spatial transport substep — a no-op for v1 homogeneous.

        Gate 2 treats the film as spatially homogeneous (``N_spatial = 1``,
        ``state.f`` is 1D over energy only). The real T3 diffusion operator
        — Crank-Nicolson on the Laplacian with an energy-dependent ``D(E)``
        — lands at Gate 5 when the spatial grid is wired up.
        """
        if state.f.ndim != 1:
            raise NotImplementedError(
                "T3 spatial transport is not implemented yet; "
                "state.f must be 1D (homogeneous) in Gate 2. "
                "Full Usadel / LEGACY diffusion lands at Gate 5."
            )
        return state

    @staticmethod
    def _persistent_coordinates_for_state(
        state: T3DiffusionState,
    ) -> _PersistentGapCoordinates:
        """Reuse synchronized material data or initialize it from public ``f``."""
        cached = state._moving_gap_coordinates
        if cached is not None and (
            cached.synchronized_gap == float(state.gap)
            and np.array_equal(cached.energy_grid, state.spectral.E)
            and np.array_equal(cached.energy_widths, state.spectral.dE)
            and np.array_equal(cached.public_f, state.f)
        ):
            return cached

        E = state.spectral.E
        dE = state.spectral.dE
        f = np.asarray(state.f, dtype=float)
        cell_edges = cell_edges_from_widths(E, dE)
        scale = max(abs(float(state.gap)), float(np.max(np.abs(cell_edges))), 1.0)
        tolerance = 128.0 * np.finfo(float).eps * scale
        if float(cell_edges[0]) > state.gap + tolerance:
            raise ValueError(
                "The energy grid does not cover the moving BCS gap edge. "
                "Extend E_min below every gap reached by the trajectory."
            )

        mapped_edges = _bcs_xi_edges(cell_edges, state.gap)
        mapped_widths = np.diff(mapped_edges)
        represented = np.flatnonzero(mapped_widths > 0.0)
        if represented.size == 0:
            raise ValueError("The energy grid has no represented material cells.")
        if not np.array_equal(
            represented,
            np.arange(represented[0], represented[-1] + 1),
        ):
            raise RuntimeError("BCS material support must be one contiguous interval.")

        xi_edges = np.concatenate(([mapped_edges[represented[0]]], mapped_edges[represented + 1]))
        occupation = f[represented].copy()

        # Reserve vacuum material cells up to the largest xi that the fixed
        # energy grid can expose as Delta -> 0.  This prevents a falling gap
        # from creating work-grid cells with no persistent degree of freedom.
        # The extension is never populated while it lies beyond E_max; if a
        # later rising gap would strand material occupied mass there, the
        # materialization audit below fails loudly.
        xi_limit = max(float(cell_edges[-1]), float(xi_edges[-1]))
        remaining = xi_limit - float(xi_edges[-1])
        if remaining > tolerance:
            typical_width = float(np.median(np.diff(xi_edges)))
            n_extra = max(1, int(np.ceil(remaining / typical_width)))
            extra_edges = np.linspace(
                float(xi_edges[-1]),
                xi_limit,
                n_extra + 1,
            )[1:]
            xi_edges = np.concatenate((xi_edges, extra_edges))
            occupation = np.concatenate((occupation, np.zeros(n_extra)))

        return _PersistentGapCoordinates(
            xi_edges=xi_edges,
            occupation=occupation,
            energy_grid=E,
            energy_widths=dE,
            synchronized_gap=float(state.gap),
            public_f=f,
        )

    def _materialize_persistent_occupation(
        self,
        state: T3DiffusionState,
        coordinates: _PersistentGapCoordinates,
        gap: float,
        *,
        occupation: np.ndarray | None = None,
    ) -> tuple[SpectralContext, np.ndarray, np.ndarray]:
        """Conservatively expose persistent ``xi`` averages on the work grid."""
        material_occupation = (
            coordinates.occupation if occupation is None else np.asarray(occupation, dtype=float)
        )
        if material_occupation.shape != coordinates.occupation.shape:
            raise ValueError("Material occupation has the wrong persistent-xi shape.")
        if np.any(~np.isfinite(material_occupation)) or np.any(
            (material_occupation < 0.0) | (material_occupation > 1.0)
        ):
            raise ValueError("Material occupation must be finite and lie in [0, 1].")
        spectral = self._rebuild_spectral_context(state.spectral, new_gap=gap)
        cell_edges = cell_edges_from_widths(spectral.E, spectral.dE)
        scale = max(abs(float(gap)), float(np.max(np.abs(cell_edges))), 1.0)
        tolerance = 128.0 * np.finfo(float).eps * scale
        if float(cell_edges[0]) > gap + tolerance:
            raise ValueError(
                "The energy grid does not cover the moving BCS gap edge. "
                "Extend E_min below every gap reached by the trajectory."
            )

        energy_xi_edges = _bcs_xi_edges(cell_edges, gap)
        exact_weights = np.diff(energy_xi_edges)
        if not np.allclose(
            exact_weights,
            spectral.cell_weights,
            rtol=5e-14,
            atol=tolerance,
        ):
            raise RuntimeError(
                "Persistent material cells disagree with SpectralContext.cell_weights."
            )

        overlap = _material_overlap(coordinates.xi_edges, energy_xi_edges)
        covered_weights = np.sum(overlap, axis=0)
        if not np.allclose(
            covered_weights,
            exact_weights,
            rtol=5e-14,
            atol=tolerance,
        ):
            raise RuntimeError("The persistent xi grid does not cover the energy work grid.")

        mass = overlap.T @ material_occupation
        f = np.divide(
            mass,
            exact_weights,
            out=np.zeros_like(mass),
            where=exact_weights > _GAP_SUPPORT_EPS,
        )
        bound_tolerance = 256.0 * np.finfo(float).eps
        if np.any(f < -bound_tolerance) or np.any(f > 1.0 + bound_tolerance):
            raise RuntimeError("Persistent materialization violated occupation bounds.")
        f = np.clip(f, 0.0, 1.0)

        xi_widths = np.diff(coordinates.xi_edges)
        material_mass = float(xi_widths @ material_occupation)
        represented_mass = float(exact_weights @ f)
        tail_mass = material_mass - represented_mass
        tail_tolerance = _MOVING_GAP_TAIL_RTOL * max(
            material_mass, np.finfo(float).tiny
        ) + 256.0 * np.finfo(float).eps * max(material_mass, 1.0)
        if tail_mass < -tail_tolerance:
            raise RuntimeError("Persistent materialization created finite-volume mass.")
        if tail_mass > tail_tolerance:
            raise RuntimeError(
                "The moving gap would strand occupied persistent-xi mass "
                "above the fixed E_max boundary. Extend the energy grid."
            )
        return spectral, f, overlap

    def _constrain_persistent_occupation(
        self,
        state: T3DiffusionState,
        coordinates: _PersistentGapCoordinates,
        occupation: np.ndarray,
        calibration: GapCalibration,
        *,
        reference_gap: float,
    ) -> tuple[float, SpectralContext, np.ndarray, np.ndarray]:
        """Solve the coupled discrete constraint ``G(P_Delta g, Delta)=0``."""
        current_gap = float(reference_gap)
        last_error = float("inf")
        last_tolerance = float("nan")
        for _iteration in range(1, _GAP_PROJECTION_MAX_ITER + 1):
            spectral, f, overlap = self._materialize_persistent_occupation(
                state,
                coordinates,
                current_gap,
                occupation=occupation,
            )
            constrained_gap = float(
                solve_gap(
                    calibration,
                    f,
                    spectral.E,
                    dE_bins=spectral.dE,
                    reference_gap=current_gap,
                    xtol=_GAP_PROJECTION_SOLVE_XTOL_UEV,
                )
            )
            if not np.isfinite(constrained_gap) or constrained_gap <= 0.0:
                raise RuntimeError(
                    "The stage gap collapsed or became non-finite; "
                    "normal-state moving-gap dynamics are not implemented."
                )
            last_error = abs(constrained_gap - current_gap)
            last_tolerance = _GAP_PROJECTION_ATOL_UEV + _GAP_PROJECTION_RTOL * max(
                abs(constrained_gap),
                abs(current_gap),
            )
            if last_error <= last_tolerance:
                return current_gap, spectral, f, overlap
            current_gap = constrained_gap

        raise RuntimeError(
            "The stage-coupled moving-gap constraint did not converge in "
            f"{_GAP_PROJECTION_MAX_ITER} iterations: final error "
            f"{last_error:.3e} micro-eV exceeds {last_tolerance:.3e} micro-eV."
        )

    def _moving_gap_energy_collision_rates(
        self,
        state: T3DiffusionState,
        spectral: SpectralContext,
        f: np.ndarray,
        *,
        N_p: np.ndarray,
        N_emit: np.ndarray,
        N_abs: np.ndarray,
        photon_params: dict[str, float] | None,
        pb_photon_params: dict[str, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate all Pauli-limited collision channels on uniform ``E``."""
        K_s0 = build_scattering_kernel_base(
            spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        K_r0 = build_recombination_kernel_base(
            spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        gain, loss = phonon_collision_rates(
            f,
            spectral,
            K_s0,
            K_r0,
            state.T_bath,
            N_p_override=N_p,
            N_emit_override=N_emit,
            N_abs_override=N_abs,
        )
        if photon_params is not None:
            photon_gain, photon_loss = sub_gap_photon_collision_rates(
                f,
                spectral,
                photon_params["omega_0"],
                photon_params["n_bar"],
                photon_params["c_phot"],
            )
            gain = gain + photon_gain
            loss = loss + photon_loss
        if pb_photon_params is not None:
            photon_gain, photon_loss = pair_breaking_photon_collision_rates(
                f,
                spectral,
                pb_photon_params["omega_PB"],
                pb_photon_params["n_bar_PB"],
                pb_photon_params["c_phot_PB"],
            )
            gain = gain + photon_gain
            loss = loss + photon_loss
        if np.any(~np.isfinite(gain)) or np.any(~np.isfinite(loss)):
            raise RuntimeError("Moving-gap collision rates became non-finite.")
        return gain, loss

    @staticmethod
    def _lift_energy_rates_to_material(
        coordinates: _PersistentGapCoordinates,
        occupation: np.ndarray,
        f: np.ndarray,
        energy_weights: np.ndarray,
        overlap: np.ndarray,
        collision_gain: np.ndarray,
        collision_loss: np.ndarray,
        external_flux: ExternalFlux | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Lift work-grid events to ``xi`` without feeding ``P_Delta g`` back.

        Internal collision gain already contains the target Pauli factor
        ``1-f_i``.  Recover its non-negative bare rate and deposit it uniformly
        over the holes in each overlapping material cell.  Loss is local per
        particle and is averaged as a coefficient.  This event-level lift
        exactly aggregates back to ``w_i (gain_i-loss_i f_i)`` while retaining
        a diagonal gain/loss form and the bounds.  ``ExternalFlux.gain`` keeps
        its documented additive (not Pauli-limited) fixed-E-cell semantics.
        """
        xi_widths = np.diff(coordinates.xi_edges)
        g = np.asarray(occupation, dtype=float)
        holes = np.maximum(1.0 - g, 0.0)
        energy_hole_mass = overlap.T @ holes
        expected_hole_mass = energy_weights * np.maximum(1.0 - f, 0.0)
        scale = max(float(np.max(energy_weights)), 1.0)
        if not np.allclose(
            energy_hole_mass,
            expected_hole_mass,
            rtol=2e-13,
            atol=256.0 * np.finfo(float).eps * scale,
        ):
            raise RuntimeError("The xi/E transfer does not preserve hole capacity.")

        collision_gain_mass = energy_weights * collision_gain
        bare_gain = np.divide(
            collision_gain_mass,
            energy_hole_mass,
            out=np.zeros_like(collision_gain_mass),
            where=energy_hole_mass > _GAP_SUPPORT_EPS,
        )
        unresolved_gain = (energy_hole_mass <= _GAP_SUPPORT_EPS) & (
            collision_gain_mass > 256.0 * np.finfo(float).eps
        )
        if np.any(unresolved_gain):
            raise RuntimeError("A collision requested gain in an energy cell with no holes.")

        gain = holes * (overlap @ bare_gain) / xi_widths
        loss = (overlap @ collision_loss) / xi_widths
        if external_flux is not None:
            gain = gain + (overlap @ external_flux.gain) / xi_widths
            loss = loss + (overlap @ external_flux.loss_rate) / xi_widths

        expected_gain = float(energy_weights @ collision_gain)
        expected_sink = float(energy_weights @ (collision_loss * f))
        if external_flux is not None:
            expected_gain += float(energy_weights @ external_flux.gain)
            expected_sink += float(energy_weights @ (external_flux.loss_rate * f))
        lifted_gain = float(xi_widths @ gain)
        lifted_sink = float(xi_widths @ (loss * g))
        balance_scale = max(
            abs(expected_gain),
            abs(expected_sink),
            abs(lifted_gain),
            abs(lifted_sink),
            np.finfo(float).tiny,
        )
        balance_tolerance = 2e-12 * balance_scale
        if (
            abs(lifted_gain - expected_gain) > balance_tolerance
            or abs(lifted_sink - expected_sink) > balance_tolerance
        ):
            raise RuntimeError("The xi/E collision lift violated event balance.")
        return gain, loss

    def _remap_gap_state_once(
        self,
        state: T3DiffusionState,
        new_gap: float,
    ) -> T3DiffusionState:
        """Conservatively remap one state from its current gap to ``new_gap``."""
        E = state.spectral.E
        dE = state.spectral.dE
        # The conserved finite-volume variable is the cell integral of
        # ``rho*f``, not the point sample ``rho(E_i)*f_i``. Near the BCS edge
        # the latter has an O(1), grid-alignment-dependent midpoint error.
        old_dos_weights = bcs_dos_cell_weights(E, dE, state.gap)
        new_dos_weights = bcs_dos_cell_weights(E, dE, new_gap)
        rho_cell_average_new = new_dos_weights / dE
        mass_before = float(np.sum(old_dos_weights * state.f))
        remapped_cell_mass, escaped_mass = _remap_bcs_frozen_xi_cell_mass(
            state.f,
            E,
            dE,
            state.gap,
            new_gap,
        )

        # A rising gap lowers the represented xi_max at fixed E_max.  The
        # exact characteristic map therefore exposes any finite-domain tail
        # explicitly.  Reject a material truncation and retain only a tiny
        # numerical tail in the highest represented active cell so the
        # closed-domain update remains conservative.
        mass_scale = max(abs(mass_before), np.finfo(float).tiny)
        escaped_fraction = escaped_mass / mass_scale
        if escaped_fraction > 1e-3:
            raise RuntimeError(
                "apply_gap_update would lose "
                f"{escaped_fraction:.2%} of quasiparticle mass through the "
                "finite E_max boundary. Extend the energy grid before "
                "evolving this gap change."
            )
        if escaped_mass > 0.0:
            support_indices = np.flatnonzero(new_dos_weights > _GAP_SUPPORT_EPS)
            tail_tolerance = 256.0 * np.finfo(float).eps * mass_scale
            if support_indices.size == 0:
                if escaped_mass > tail_tolerance:
                    raise RuntimeError(
                        "The updated gap leaves no represented BCS support "
                        "for the finite-E_max characteristic tail."
                    )
            else:
                tail_index = int(support_indices[-1])
                tail_capacity = new_dos_weights[tail_index] - remapped_cell_mass[tail_index]
                if escaped_mass > tail_capacity + tail_tolerance:
                    raise RuntimeError(
                        "The finite-E_max characteristic tail does not fit "
                        "in the highest represented active cell without "
                        "violating f <= 1. Extend the energy grid."
                    )
                remapped_cell_mass[tail_index] += escaped_mass
            if escaped_fraction > 1e-9:
                warnings.warn(
                    "apply_gap_update retained a finite-E_max characteristic "
                    f"tail containing {escaped_fraction:.3e} of the "
                    "quasiparticle mass in the highest represented active "
                    "cell. Extend E_max to remove this boundary dependence.",
                    stacklevel=2,
                )

        u_new = remapped_cell_mass / dE

        # Preserve every non-gap configuration from the incoming
        # SpectralContext: the caller may have set a custom
        # diffusion_coefficient (not just material.D_0), dynes_gamma,
        # rebuild_tolerance, or active_margin_factor. Only Delta changes.
        new_spectral = self._rebuild_spectral_context(
            state.spectral,
            new_gap=new_gap,
        )

        f_new, remapped_mass, n_touched = _recover_conservative_gap_occupation(
            u_new,
            rho_cell_average_new,
            dE,
        )

        # Audit the physical finite-volume invariant, including non-uniform
        # widths.  The recovery helper already enforces 0 <= f <= 1 without a
        # lossy clip; a large remap fraction is reported as a resolution
        # diagnostic, with the signed invariant drift stated separately.
        mass_after = float(np.sum(new_dos_weights * f_new))
        signed_drift = (mass_after - mass_before) / mass_scale
        if abs(signed_drift) > 5e-12:
            raise RuntimeError(
                "apply_gap_update failed to conserve the finite-volume "
                "invariant sum(f*integral_cell(rho dE)): "
                f"signed drift {signed_drift:+.3e}."
            )
        if mass_before > 0.0 and remapped_mass > 0.05 * mass_before:
            warnings.warn(
                "apply_gap_update conservatively remapped "
                f"{remapped_mass / mass_before:.2%} of the analytic BCS "
                "cell measure across "
                f"{n_touched} above-gap bins for Delta "
                f"{state.gap:.4g}->{new_gap:.4g} micro-eV; the signed change "
                f"in the invariant is {signed_drift:+.3e}. Increase energy "
                "resolution if this edge-remap fraction is too large.",
                stacklevel=2,
            )

        return replace(state, gap=new_gap, spectral=new_spectral, f=f_new)

    def apply_gap_update(
        self,
        state: T3DiffusionState,
        dt: float,
    ) -> T3DiffusionState:
        r"""Advance ``Delta`` and remap BCS cell mass at frozen ``xi``.

        Solve the reference-subtracted BCS gap equation and, when the gap
        moves, map the ideal-BCS finite-volume mass exactly along frozen-
        ``xi`` characteristics. This is the characteristic solution of the
        conservative spectral-flow equation because ``rho(E) dE = dxi``.
        Rebuild the :class:`SpectralContext`, recover a bounded ``f``, and
        repeat the solve/remap cycle because the remapped occupation changes
        the gap equation. A state is returned only after its own occupation
        satisfies the gap constraint to ``atol + rtol * gap_scale``. Failure
        to reach that fixed point within the iteration cap is explicit.

        Moving-gap flow is currently supported only for ideal BCS spectra.
        The grid cells must cover the new gap edge.  A finite upper energy
        boundary may truncate rising-gap characteristics; material loss at
        that boundary is diagnosed rather than silently discarded.

        ``dt`` is retained for backend-interface compatibility and as a
        positive/no-op gate only. For every positive value this routine
        performs the same complete algebraic gap solve and frozen-``xi``
        remap. It is therefore a projection, not a ``dt``-scaled evolution
        operator; in particular, passing ``dt / 2`` does not produce a
        half-gap flow suitable for Strang splitting.
        """
        gap_scale = max(abs(float(state.gap)), abs(float(state.spectral.gap)), 1.0)
        if not np.isclose(
            state.gap,
            state.spectral.gap,
            rtol=_GAP_STATE_MATCH_RTOL,
            atol=_GAP_STATE_MATCH_RTOL * gap_scale,
        ):
            raise ValueError(
                "state.gap and state.spectral.gap must match before a gap "
                f"update; got {state.gap:g} and {state.spectral.gap:g}."
            )
        if not np.isfinite(state.gap) or state.gap <= 0.0:
            raise ValueError(f"apply_gap_update requires a finite positive gap; got {state.gap}.")
        if not np.isfinite(dt):
            raise ValueError(f"dt must be finite; got {dt}.")
        if dt <= 0:
            return state

        f_state = np.asarray(state.f, dtype=float)
        if f_state.shape != state.spectral.E.shape:
            raise ValueError(
                "state.f must have the same one-dimensional shape as the spectral energy grid."
            )
        if np.any(~np.isfinite(f_state)) or np.any((f_state < 0.0) | (f_state > 1.0)):
            raise ValueError("state.f must contain finite occupations in [0, 1].")

        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Moving-gap spectral flow is implemented only for an ideal "
                "BCS spectrum (dynes_gamma == 0). A Dynes DOS requires the "
                "complex anomalous spectral flux, not (Delta/E)*rho."
            )

        calibration = calibrate_gap(
            T_c=state.material.T_c,
            T_bath=state.T_bath,
            Delta_0=state.material.Delta_0,
        )
        current = state
        last_gap_error = float("inf")
        last_gap_tolerance = float("nan")
        for iteration in range(1, _GAP_PROJECTION_MAX_ITER + 1):
            constrained_gap = float(
                solve_gap(
                    calibration,
                    current.f,
                    current.spectral.E,
                    dE_bins=current.spectral.dE,
                    reference_gap=current.gap,
                    xtol=_GAP_PROJECTION_SOLVE_XTOL_UEV,
                )
            )
            if not np.isfinite(constrained_gap) or constrained_gap <= 0.0:
                raise RuntimeError(
                    "solve_gap returned a collapsed or non-finite gap "
                    f"({constrained_gap}); normal-state moving-gap dynamics "
                    "are not implemented."
                )

            last_gap_error = abs(constrained_gap - current.gap)
            last_gap_tolerance = _GAP_PROJECTION_ATOL_UEV + _GAP_PROJECTION_RTOL * max(
                abs(constrained_gap), abs(current.gap)
            )
            if last_gap_error <= last_gap_tolerance:
                # ``current`` is the state whose occupation produced
                # ``constrained_gap``. Returning it certifies the final
                # algebraic constraint. Remapping once more here would
                # recreate the former one-pass inconsistency.
                return current

            if iteration == _GAP_PROJECTION_MAX_ITER:
                break
            current = self._remap_gap_state_once(current, constrained_gap)

        raise RuntimeError(
            "apply_gap_update algebraic projection did not converge in "
            f"{_GAP_PROJECTION_MAX_ITER} iterations: final "
            f"|Delta_gap(f) - Delta|={last_gap_error:.3e} micro-eV exceeds "
            f"the tolerance {last_gap_tolerance:.3e} micro-eV."
        )

    def step(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
    ) -> T3DiffusionState:
        r"""Advance collisions and a self-consistent gap to second order.

        The authoritative unknown is a cell-average occupation ``g(xi)`` on a
        persistent material grid, ``xi=sqrt(E**2-Delta**2)``.  It is retained
        privately in the returned state and is never reconstructed from the
        public fixed-E materialization on the next accepted step.  At each ETD2
        stage, exact ``d xi`` overlaps expose ``g`` on the existing uniform
        energy work grid, the coupled discrete constraint
        ``G(P_Delta g, Delta)=0`` is solved, and all collision channels are
        evaluated on that work grid.  Event gain/loss is then conservatively
        lifted back to material cells before the predictor/corrector advances.

        This is ETD2 applied to the reduced index-one DAE
        ``g_dot=C(g, Delta), G(g, Delta)=0``.  It is not a composition of full
        algebraic projections.  Frozen phonons remain on their original
        uniform pair-frequency lattice; sub-gap/PB photon partners and an
        array-valued ``ExternalFlux`` retain their fixed-E-cell semantics.
        ``apply_collisions`` remains the unchanged fixed-gap ETD2 path.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"step requires a finite positive dt; got {dt}.")
        gap_scale = max(abs(float(state.gap)), abs(float(state.spectral.gap)), 1.0)
        if not np.isclose(
            state.gap,
            state.spectral.gap,
            rtol=_GAP_STATE_MATCH_RTOL,
            atol=_GAP_STATE_MATCH_RTOL * gap_scale,
        ):
            raise ValueError(
                "state.gap and state.spectral.gap must match before a moving-gap "
                f"step; got {state.gap:g} and {state.spectral.gap:g}."
            )
        if not np.isfinite(state.gap) or state.gap <= 0.0:
            raise ValueError(f"step requires a finite positive gap; got {state.gap}.")
        f_state = np.asarray(state.f, dtype=float)
        if f_state.shape != state.spectral.E.shape:
            raise ValueError(
                "state.f must have the same one-dimensional shape as the spectral energy grid."
            )
        if np.any(~np.isfinite(f_state)) or np.any((f_state < 0.0) | (f_state > 1.0)):
            raise ValueError("state.f must contain finite occupations in [0, 1].")
        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Second-order moving-gap dynamics require an ideal BCS spectrum (dynes_gamma == 0)."
            )

        self._validate_gate2_scope(state.phonon)
        self._validate_phonon_on_physics_grid(state)
        if external_flux is not None:
            external_flux._validate_for_NE(int(state.spectral.E.size))

        coordinates = self._persistent_coordinates_for_state(state)
        calibration = calibrate_gap(
            T_c=state.material.T_c,
            T_bath=state.T_bath,
            Delta_0=state.material.Delta_0,
        )
        _, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(state.spectral.E)
        n_ph = state.phonon.n_ph[0, :, 0]
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph,
            idx_diff,
            idx_sum,
            diff_sign,
        )
        reference_gap = float(state.gap)

        def material_rhs(g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            _, spectral, f, overlap = self._constrain_persistent_occupation(
                state,
                coordinates,
                g,
                calibration,
                reference_gap=reference_gap,
            )
            if external_flux is not None:
                external_flux._validate_gain_support(spectral.active_mask)
            collision_gain, collision_loss = self._moving_gap_energy_collision_rates(
                state,
                spectral,
                f,
                N_p=N_p,
                N_emit=N_emit,
                N_abs=N_abs,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
            )
            return self._lift_energy_rates_to_material(
                coordinates,
                g,
                f,
                spectral.cell_weights,
                overlap,
                collision_gain,
                collision_loss,
                external_flux,
            )

        xi_widths = np.diff(coordinates.xi_edges)
        occupation_new = etd2_step(
            coordinates.occupation,
            material_rhs,
            dt,
            balance_weights=xi_widths,
            max_loss_step=0.25,
        )
        final_gap, final_spectral, final_f, _ = self._constrain_persistent_occupation(
            state,
            coordinates,
            occupation_new,
            calibration,
            reference_gap=reference_gap,
        )
        persistent = _PersistentGapCoordinates(
            xi_edges=coordinates.xi_edges,
            occupation=occupation_new,
            energy_grid=final_spectral.E,
            energy_widths=final_spectral.dE,
            synchronized_gap=final_gap,
            public_f=final_f,
        )
        return replace(
            state,
            gap=final_gap,
            spectral=final_spectral,
            f=final_f,
            _moving_gap_coordinates=persistent,
        )

    @staticmethod
    def _validate_gate2_scope(phonon: PhononState) -> None:
        """Reject PhononState shapes the Gate 2 T3 backend can't handle.

        Gate 2 supports the single-branch, spatially-homogeneous,
        constant-``τ_l`` case. Multi-branch (v3), spatially-resolved
        (Ph1/Ph2), and ω-dependent ``τ_l`` land in later gates.
        """
        if phonon.model is not PhononModel.PH0_LOCAL:
            raise ValueError(
                "T3DiffusionBackend implements only the PH0_LOCAL phonon "
                f"model; got {phonon.model.value!r}. PH1/PH2 transport must "
                "use a backend that evolves those models explicitly."
            )
        if phonon.n_branch != 1:
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports single-branch phonons only; "
                f"got n_branch = {phonon.n_branch}. Multi-branch support arrives "
                "with v3 per D5."
            )
        if phonon.n_spatial != 1:
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports spatially-homogeneous "
                f"phonons only; got n_spatial = {phonon.n_spatial}. Ph1 lateral "
                "transport lands at Gate 5."
            )
        tau0 = float(phonon.tau_l[0, 0])
        if not np.allclose(phonon.tau_l, tau0, rtol=_TAU_L_UNIFORMITY_RTOL):
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports constant-τ_l only; "
                "every entry of state.phonon.tau_l must be equal. "
                "Frequency-dependent τ_l(ω) needs solve_steady_state to "
                "accept an array, which is a post-Gate-4 upgrade."
            )

    @staticmethod
    def _validate_phonon_on_physics_grid(state: T3DiffusionState) -> None:
        """Ensure ``state.phonon.omega_bins`` matches the QP-derived ω grid.

        ``apply_collisions`` consumes ``state.phonon.n_ph`` directly, so
        the ω grid it lives on must match the pair-sum / pair-difference
        grid that :func:`build_phonon_frequency_map` derives from
        ``state.spectral.E``. (``steady_state`` rebuilds the grid
        internally and returns a phonon on the physics grid; the usual
        workflow is ``steady_state`` first, then transient ``step``s.)
        """
        expected, _, _, _ = build_phonon_frequency_map(state.spectral.E)
        if state.phonon.omega_bins.shape != (1, expected.size):
            raise ValueError(
                f"state.phonon.omega_bins shape {state.phonon.omega_bins.shape} "
                f"does not match physics grid shape (1, {expected.size}) derived "
                "from state.spectral.E. Run backend.steady_state(state) first, or "
                "build PhononState on the physics ω grid."
            )
        if not np.allclose(state.phonon.omega_bins[0], expected):
            raise ValueError(
                "state.phonon.omega_bins does not match the physics ω grid "
                "(pair-sum / pair-diff of state.spectral.E)."
            )
