"""T3 backend: isotropic dirty-limit diffusion (scalar occupation).

Composes the pieces from earlier Gate 2 commits into a working
backend with both steady-state and transient time-evolution paths.

Scope for Gate 2: spatially-homogeneous runs (``N_spatial = 1``), a
scalar gap, the e-phonon integral, and optional sub-gap / PB photon
channels via ``photon_params`` / ``pb_photon_params``. The transient
``step()`` uses a symmetric 3-operator Strang split with
``apply_transport`` as a no-op for ``N_spatial = 1`` (real spatial
diffusion lands at Gate 5).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

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
from qpsim.phonon_models.state import PhononState
from qpsim.physics.gap_equation import calibrate_gap, solve_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state
from qpsim.solvers.coupled_newton import coupled_newton_solve
from qpsim.solvers.etd import etd2_step
from qpsim.solvers.spectral_flow_tvd import advect_spectral_flow

_TAU_L_UNIFORMITY_RTOL = 1e-10
_GAP_STATE_MATCH_RTOL = 1e-12
_GAP_SUPPORT_EPS = 1e-30
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
        available = (
            capacity[selected] - mass[selected]
            if add_mass
            else mass[selected]
        )
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
            room = (
                capacity[idx] - mass[idx]
                if correction_adds
                else mass[idx]
            )
            amount = min(max(float(room), 0.0), remaining)
            mass[idx] += amount if correction_adds else -amount
            remaining -= amount
            if remaining <= tolerance:
                break
    return int(np.count_nonzero(available > 0.0))


def _recover_conservative_gap_occupation(
    u_advected: np.ndarray,
    rho_new: np.ndarray,
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
    rho = np.asarray(rho_new, dtype=float)
    widths = np.asarray(dE, dtype=float)
    cell_mass = u * widths
    if np.any(~np.isfinite(cell_mass)):
        raise RuntimeError("apply_gap_update produced non-finite spectral mass.")

    mass_scale = float(np.sum(np.abs(cell_mass)))
    tolerance = 256.0 * np.finfo(float).eps * max(
        mass_scale, np.finfo(float).tiny,
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
        cell_mass[support], 0.0, allocation_capacity[support],
    )
    delta = total_mass - float(np.sum(mass_out))
    n_touched = _spread_mass_change(
        mass_out, allocation_capacity, support_indices, delta, tolerance,
    )

    final_mass = float(np.sum(mass_out))
    residual = total_mass - final_mass
    if abs(residual) > tolerance:
        raise RuntimeError(
            "apply_gap_update conservative remap left a mass residual of "
            f"{residual:g}."
        )

    f_new = np.zeros_like(u)
    f_new[support] = mass_out[support] / capacity[support]
    if np.any(f_new < -1e-13) or np.any(f_new > 1.0 + 1e-13):
        raise RuntimeError(
            "apply_gap_update conservative remap violated occupation bounds."
        )
    f_new = np.clip(f_new, 0.0, 1.0)
    remapped_mass = 0.5 * float(np.sum(np.abs(mass_out - cell_mass)))
    return f_new, remapped_mass, n_touched


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
    """

    f: np.ndarray
    gap: float
    spectral: SpectralContext
    phonon: PhononState
    material: Material
    T_bath: float
    tier: Tier = Tier.T3_DIFFUSION


class T3DiffusionBackend:
    """Steady-state solver for the T3 tier.

    Stateless; a single instance can be reused across many runs.
    Transient methods land in task 12.
    """

    def steady_state(
        self,
        state: T3DiffusionState,
        *,
        method: str = "picard",
        self_consistent_gap: bool = False,
        use_thermal_phonons: bool = False,
        external_dissipation_only: bool = False,
        use_phonon_side_kernel: bool = False,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
        newton_tol: float = 1e-14,
        newton_max_iter: int = 200,
        picard_tol: float = 1e-10,
        picard_atol: float = 1e-11,
        picard_max_iter: int = 200,
        picard_mixing: float = 0.3,
        anderson_depth: int = 0,
        coupled_newton_tol: float = 1e-10,
        coupled_newton_step_rtol: float = 0.0,
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
            ``gap_tol``. The final state is re-solved once more at the
            converged gap so ``f``/``n_ph`` and ``Δ`` are exactly
            consistent.
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
            **Opt-in** F&C 2023 Eq. 12 phonon-side kernel for the
            phonon-equation rate. When ``True``, builds a sibling
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
            ``external_dissipation_only=True``. ``False`` (default)
            preserves legacy behavior bit-for-bit.
        photon_params, pb_photon_params
            Optional photon channel dicts.
        newton_tol, newton_max_iter
            Inner Newton controls (used by Picard and thermal-phonon paths).
        picard_tol, picard_atol, picard_max_iter, picard_mixing, anderson_depth
            Picard path controls. ``picard_atol`` is an absolute tolerance on
            dimensionless phonon occupation, independent of ``newton_tol``.
        coupled_newton_tol, coupled_newton_max_iter, coupled_newton_fd_step
            Coupled-Newton path controls.
        gap_tol, gap_max_iter, gap_under_relaxation
            Outer self-consistent-gap loop controls. Ignored when
            ``self_consistent_gap=False``.
        """
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

        if gap_tol <= 0:
            raise ValueError("gap_tol must be positive.")
        if gap_max_iter <= 0:
            raise ValueError("gap_max_iter must be positive.")
        if not (0.0 < gap_under_relaxation <= 1.0):
            raise ValueError(
                "gap_under_relaxation must lie in the interval (0, 1]."
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
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_step_rtol=coupled_newton_step_rtol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
                coupled_newton_analytic_cross=coupled_newton_analytic_cross,
            )

        if state.material.T_c <= 0:
            raise ValueError(
                "state.material.T_c must be positive for self-consistent-gap solves."
            )
        if state.T_bath >= state.material.T_c:
            raise ValueError(
                "self-consistent-gap steady state requires T_bath < T_c; "
                f"got T_bath={state.T_bath} K and T_c={state.material.T_c} K."
            )

        calibration = calibrate_gap(T_c=state.material.T_c, T_bath=state.T_bath)
        current = state
        final_delta = float(state.gap)
        last_solved: T3DiffusionState | None = None
        last_rel_change = float("inf")

        for _ in range(gap_max_iter):
            solved = self._steady_state_fixed_gap(
                current,
                method=method,
                use_thermal_phonons=use_thermal_phonons,
                use_phonon_side_kernel=use_phonon_side_kernel,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                newton_tol=newton_tol,
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_step_rtol=coupled_newton_step_rtol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
                coupled_newton_analytic_cross=coupled_newton_analytic_cross,
            )
            last_solved = solved

            delta_raw = solve_gap(
                calibration, solved.f, solved.spectral.E,
                xtol=gap_solve_xtol,
            )
            if delta_raw <= 0.0:
                # The current occupation no longer supports a superconducting
                # solution; collapse to the normal state directly. Under-relaxing
                # zero against the previous gap would have drifted us to a
                # spurious tiny-Δ "almost-superconducting" fixed point.
                raise RuntimeError(
                    "Self-consistent gap collapsed: solve_gap returned Δ=0 at "
                    f"iteration with |f|_max={float(solved.f.max()):.3e}. "
                    "The drive has exceeded the pair-breaking threshold; "
                    "this solver does not yet support the normal state."
                )
            final_delta = (
                (1.0 - gap_under_relaxation) * solved.gap
                + gap_under_relaxation * delta_raw
            )
            last_rel_change = abs(final_delta - solved.gap) / max(abs(solved.gap), 1e-30)

            if last_rel_change < gap_tol:
                break

            current = replace(
                solved,
                gap=final_delta,
                spectral=self._rebuild_spectral_context(
                    solved.spectral, new_gap=final_delta,
                ),
            )
        else:
            raise RuntimeError(
                "Self-consistent gap iteration did not converge in "
                f"{gap_max_iter} iterations. Final |Δ_new - Δ| / Δ = "
                f"{last_rel_change:.2e}."
            )

        if last_solved is None:
            raise RuntimeError("Internal error: no steady-state solve was performed.")

        if abs(final_delta - last_solved.gap) < 1e-14:
            return last_solved

        final_state = replace(
            last_solved,
            gap=final_delta,
            spectral=self._rebuild_spectral_context(
                last_solved.spectral, new_gap=final_delta,
            ),
        )
        return self._steady_state_fixed_gap(
            final_state,
            method=method,
            use_thermal_phonons=use_thermal_phonons,
            use_phonon_side_kernel=use_phonon_side_kernel,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            external_flux=external_flux,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
            picard_tol=picard_tol,
            picard_atol=picard_atol,
            picard_max_iter=picard_max_iter,
            picard_mixing=picard_mixing,
            anderson_depth=anderson_depth,
            coupled_newton_tol=coupled_newton_tol,
            coupled_newton_step_rtol=coupled_newton_step_rtol,
            coupled_newton_max_iter=coupled_newton_max_iter,
            coupled_newton_fd_step=coupled_newton_fd_step,
            coupled_newton_analytic_cross=coupled_newton_analytic_cross,
        )

    def _steady_state_fixed_gap(
        self,
        state: T3DiffusionState,
        *,
        method: str,
        use_thermal_phonons: bool,
        external_dissipation_only: bool = False,
        use_phonon_side_kernel: bool = False,
        photon_params: dict[str, float] | None,
        pb_photon_params: dict[str, float] | None,
        external_flux: ExternalFlux | None,
        newton_tol: float,
        newton_max_iter: int,
        picard_tol: float,
        picard_atol: float,
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
                if state.material.tau_0_pb_ns is None:
                    raise ValueError(
                        "use_phonon_side_kernel=True requires "
                        "state.material.tau_0_pb_ns to be set; got None. "
                        "Set τ_0^PB on the Material (e.g. via the YAML "
                        "database key 'tau_0_pb_ns') or leave the flag "
                        "False to retain the legacy QP-side kernel."
                    )
                K_r0_phonon_side = build_recombination_kernel_phonon_side(
                    state.spectral,
                    tau_0_pb_ns=state.material.tau_0_pb_ns,
                )
                K_s0_phonon_side = build_scattering_kernel_phonon_side(
                    state.spectral,
                    tau_0_pb_ns=state.material.tau_0_pb_ns,
                )

        tau_l_scalar = float(state.phonon.tau_l[0, 0])

        if use_thermal_phonons:
            # Newton-only shortcut: solve_steady_state routes to newton_solve_f
            # when phonon_escape_time=None, with n_ph held at Bose-Einstein.
            f_new = solve_steady_state(
                state.spectral, K_s0, K_r0, state.T_bath,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=external_flux,
                initial_guess=state.f,
                tol=newton_tol, max_iter=newton_max_iter,
                phonon_escape_time=None,
            )
            omega_conv, _, _, _ = build_phonon_frequency_map(state.spectral.E)
            n_ph_conv = thermal_phonon_occupation(omega_conv, state.T_bath)
        elif method == "picard":
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
                tol=newton_tol,
                max_iter=newton_max_iter,
                phonon_escape_time=tau_l_scalar,
                max_picard_iter=picard_max_iter,
                picard_tol=picard_tol,
                picard_atol=picard_atol,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                phonon_out=phonon_out,
            )
            n_ph_conv = phonon_out["n_ph"]
            omega_conv = phonon_out["omega_bins"]
        elif method == "coupled_newton":
            omega_conv, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
                state.spectral.E
            )
            # Seed n_ph from state if it's already on the physics grid,
            # else start from thermal.
            if (
                state.phonon.omega_bins.shape == (1, omega_conv.size)
                and np.allclose(state.phonon.omega_bins[0], omega_conv)
            ):
                n_ph_init = state.phonon.n_ph[0, :, 0].copy()
            else:
                n_ph_init = thermal_phonon_occupation(omega_conv, state.T_bath)
            f_new, n_ph_conv = coupled_newton_solve(
                state.spectral, state.f, n_ph_init,
                omega_bins=omega_conv,
                omega_idx_diff=idx_diff,
                omega_idx_sum=idx_sum,
                diff_sign=diff_sign,
                K_s0=K_s0, K_r0=K_r0,
                K_s0_phonon_side=K_s0_phonon_side,
                K_r0_phonon_side=K_r0_phonon_side,
                T_bath=state.T_bath, tau_l=tau_l_scalar,
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
            raise ValueError(
                f"Unknown method {method!r}. Use 'picard' or 'coupled_newton'."
            )

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
        self._validate_gate2_scope(state.phonon)
        self._validate_phonon_on_physics_grid(state)

        if external_flux is not None:
            external_flux._validate_for_NE(int(state.spectral.E.size))

        K_s0 = build_scattering_kernel_base(
            state.spectral, tau_0=state.material.tau_0, T_c=state.material.T_c,
        )
        K_r0 = build_recombination_kernel_base(
            state.spectral, tau_0=state.material.tau_0, T_c=state.material.T_c,
        )

        _, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(state.spectral.E)
        n_ph_1d = state.phonon.n_ph[0, :, 0]
        N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
            n_ph_1d, idx_diff, idx_sum, diff_sign,
        )

        def rhs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain, loss = phonon_collision_rates(
                f, state.spectral, K_s0, K_r0, state.T_bath,
                N_p_override=N_p, N_emit_override=N_emit, N_abs_override=N_abs,
            )
            if photon_params is not None:
                gp, lp = sub_gap_photon_collision_rates(
                    f, state.spectral,
                    photon_params["omega_0"],
                    photon_params["n_bar"],
                    photon_params["c_phot"],
                )
                gain = gain + gp
                loss = loss + lp
            if pb_photon_params is not None:
                gp, lp = pair_breaking_photon_collision_rates(
                    f, state.spectral,
                    pb_photon_params["omega_PB"],
                    pb_photon_params["n_bar_PB"],
                    pb_photon_params["c_phot_PB"],
                )
                gain = gain + gp
                loss = loss + lp
            if external_flux is not None:
                gain = gain + external_flux.gain
                loss = loss + external_flux.loss_rate
            return gain, loss

        f_new = etd2_step(state.f, rhs, dt)
        return replace(state, f=f_new)

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

    def apply_gap_update(
        self,
        state: T3DiffusionState,
        dt: float,
    ) -> T3DiffusionState:
        """Advance Δ via ``solve_gap`` and advect ``ρ·f`` via spectral flow.

        1. Solve for the new gap from the current ``f`` (reference-
           subtracted BCS).
        2. If Δ moved, compute ``Δ̇ = (Δ_new − Δ_old) / dt`` and apply a
           displacement-subcycled TVD+SSPRK step on the finite-volume
           conserved variable ``u = ρ · f``.
        3. Rebuild the :class:`SpectralContext` at the new Δ and recover a
           bounded ``f`` with a ``dE``-weighted conservative edge remap.

        Moving-gap flow is currently supported only for ideal BCS spectra.
        A falling gap also requires at least one energy-bin centre below the
        new edge; otherwise the fixed lower boundary would trap outflow.
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
            raise ValueError(
                f"apply_gap_update requires a finite positive gap; got {state.gap}."
            )
        if dt <= 0:
            return state

        calibration = calibrate_gap(T_c=state.material.T_c, T_bath=state.T_bath)
        new_gap = float(solve_gap(calibration, state.f, state.spectral.E))

        if not np.isfinite(new_gap) or new_gap <= 0.0:
            raise RuntimeError(
                "solve_gap returned a collapsed or non-finite gap "
                f"({new_gap}); normal-state moving-gap dynamics are not "
                "implemented."
            )
        if abs(new_gap - state.gap) < 1e-14:
            return state
        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Moving-gap spectral flow is implemented only for an ideal "
                "BCS spectrum (dynes_gamma == 0). A Dynes DOS requires the "
                "complex anomalous spectral flux, not (Delta/E)*rho."
            )

        E = state.spectral.E
        dE = state.spectral.dE
        if new_gap < state.gap and not (float(E[0]) < new_gap):
            raise ValueError(
                "A falling-gap update requires sub-gap grid room: at least "
                "one energy-bin centre must lie below the new gap. "
                f"Got E[0]={float(E[0]):g} and Delta_new={new_gap:g}. "
                "Extend the energy grid below the minimum expected gap."
            )

        gap_dot = (new_gap - state.gap) / dt
        u_old = state.spectral.rho * state.f
        # No active_mask here: for a falling gap the conservative flux
        # ∂_E[(Δ/E)Δ̇ N₁f] (eq:full_kinetic_conservative) legitimately
        # carries density below the OLD gap edge into the newly opened
        # band (Δ_new, Δ_old) — the pre-step mask would zero that
        # spectral inflow. Numerical residue outside the NEW support is
        # conservatively remapped after the advection.
        u_new = advect_spectral_flow(
            u_old, E, dE,
            gap=state.gap, gap_dot=gap_dot, dt=dt,
        )

        # Preserve every non-gap configuration from the incoming
        # SpectralContext: the caller may have set a custom
        # diffusion_coefficient (not just material.D_0), dynes_gamma,
        # rebuild_tolerance, or active_margin_factor. Only Δ changes.
        new_spectral = self._rebuild_spectral_context(
            state.spectral, new_gap=new_gap,
        )

        rho_new = new_spectral.rho
        f_new, remapped_mass, n_touched = _recover_conservative_gap_occupation(
            u_new, rho_new, dE,
        )

        # Audit the physical finite-volume invariant, including non-uniform
        # widths.  The recovery helper already enforces 0 <= f <= 1 without a
        # lossy clip; a large remap fraction is reported as a resolution
        # diagnostic, with the signed invariant drift stated separately.
        mass_before = float(np.sum(u_old * dE))
        mass_after = float(np.sum(rho_new * f_new * dE))
        mass_scale = max(abs(mass_before), np.finfo(float).tiny)
        signed_drift = (mass_after - mass_before) / mass_scale
        if abs(signed_drift) > 5e-12:
            raise RuntimeError(
                "apply_gap_update failed to conserve the finite-volume "
                f"invariant sum(rho*f*dE): signed drift {signed_drift:+.3e}."
            )
        if mass_before > 0.0 and remapped_mass > 0.05 * mass_before:
            warnings.warn(
                "apply_gap_update conservatively remapped "
                f"{remapped_mass / mass_before:.2%} of sum(rho*f*dE) across "
                f"{n_touched} above-gap bins for Delta "
                f"{state.gap:.4g}->{new_gap:.4g} micro-eV; the signed change "
                f"in the invariant is {signed_drift:+.3e}. Increase energy "
                "resolution if this edge-remap fraction is too large.",
                stacklevel=2,
            )

        return replace(state, gap=new_gap, spectral=new_spectral, f=f_new)

    def step(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        external_flux: ExternalFlux | None = None,
    ) -> T3DiffusionState:
        """One symmetric-Strang time step.

        Three-operator split with gap/transport as the "outer"
        half-step operators and collisions as the "inner" full-step
        operator:

            gap/2, transport/2, collisions(dt), transport/2, gap/2

        For ``N_spatial = 1`` (Gate 2 scope), ``apply_transport`` is a
        no-op so the effective step reduces to
        ``gap/2, collisions(dt), gap/2``.
        """
        s = self.apply_gap_update(state, dt / 2)
        s = self.apply_transport(s, dt / 2)
        s = self.apply_collisions(
            s, dt,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            external_flux=external_flux,
        )
        s = self.apply_transport(s, dt / 2)
        s = self.apply_gap_update(s, dt / 2)
        return s

    @staticmethod
    def _validate_gate2_scope(phonon: PhononState) -> None:
        """Reject PhononState shapes the Gate 2 T3 backend can't handle.

        Gate 2 supports the single-branch, spatially-homogeneous,
        constant-``τ_l`` case. Multi-branch (v3), spatially-resolved
        (Ph1/Ph2), and ω-dependent ``τ_l`` land in later gates.
        """
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
