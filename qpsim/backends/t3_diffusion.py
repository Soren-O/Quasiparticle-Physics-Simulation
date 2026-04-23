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

from dataclasses import dataclass, replace

import numpy as np

from qpsim.backends.base import Tier
from qpsim.collisions.pair_breaking_photon import pair_breaking_photon_collision_rates
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
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
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        newton_tol: float = 1e-14,
        newton_max_iter: int = 200,
        picard_tol: float = 1e-10,
        picard_max_iter: int = 200,
        picard_mixing: float = 0.3,
        anderson_depth: int = 0,
        coupled_newton_tol: float = 1e-10,
        coupled_newton_max_iter: int = 50,
        coupled_newton_fd_step: float = 1e-8,
        gap_tol: float = 1e-6,
        gap_max_iter: int = 20,
        gap_under_relaxation: float = 0.5,
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
            coupled system). This is Fischer's ``τ_l=0`` limit —
            physically, the thermalization timescale is instantaneous
            so the phonon field is always at the bath. Mutually
            exclusive with ``method="coupled_newton"``.
        photon_params, pb_photon_params
            Optional photon channel dicts.
        newton_tol, newton_max_iter
            Inner Newton controls (used by Picard and thermal-phonon paths).
        picard_tol, picard_max_iter, picard_mixing, anderson_depth
            Picard path controls.
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
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                newton_tol=newton_tol,
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
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
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                newton_tol=newton_tol,
                newton_max_iter=newton_max_iter,
                picard_tol=picard_tol,
                picard_max_iter=picard_max_iter,
                picard_mixing=picard_mixing,
                anderson_depth=anderson_depth,
                coupled_newton_tol=coupled_newton_tol,
                coupled_newton_max_iter=coupled_newton_max_iter,
                coupled_newton_fd_step=coupled_newton_fd_step,
            )
            last_solved = solved

            delta_raw = solve_gap(calibration, solved.f, solved.spectral.E)
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
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            newton_tol=newton_tol,
            newton_max_iter=newton_max_iter,
            picard_tol=picard_tol,
            picard_max_iter=picard_max_iter,
            picard_mixing=picard_mixing,
            anderson_depth=anderson_depth,
            coupled_newton_tol=coupled_newton_tol,
            coupled_newton_max_iter=coupled_newton_max_iter,
            coupled_newton_fd_step=coupled_newton_fd_step,
        )

    def _steady_state_fixed_gap(
        self,
        state: T3DiffusionState,
        *,
        method: str,
        use_thermal_phonons: bool,
        photon_params: dict[str, float] | None,
        pb_photon_params: dict[str, float] | None,
        newton_tol: float,
        newton_max_iter: int,
        picard_tol: float,
        picard_max_iter: int,
        picard_mixing: float,
        anderson_depth: int,
        coupled_newton_tol: float,
        coupled_newton_max_iter: int,
        coupled_newton_fd_step: float,
    ) -> T3DiffusionState:
        """Inner steady-state solve at fixed ``Δ``."""
        self._validate_gate2_scope(state.phonon)

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

        tau_l_scalar = float(state.phonon.tau_l[0, 0])

        if use_thermal_phonons:
            # Newton-only shortcut: solve_steady_state routes to newton_solve_f
            # when phonon_escape_time=None, with n_ph held at Bose-Einstein.
            f_new = solve_steady_state(
                state.spectral, K_s0, K_r0, state.T_bath,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
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
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                initial_guess=state.f,
                tol=newton_tol,
                max_iter=newton_max_iter,
                phonon_escape_time=tau_l_scalar,
                max_picard_iter=picard_max_iter,
                picard_tol=picard_tol,
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
                T_bath=state.T_bath, tau_l=tau_l_scalar,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                tol=coupled_newton_tol,
                max_iter=coupled_newton_max_iter,
                fd_step=coupled_newton_fd_step,
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
        2. If Δ moved, compute ``Δ̇ = (Δ_new − Δ_old) / dt`` and apply
           one TVD+SSPRK step on the conserved variable ``u = ρ · f``.
        3. Rebuild the :class:`SpectralContext` at the new Δ and recover
           ``f = u / ρ_new`` above the new gap edge (clipped to [0, 1]).
        """
        if dt <= 0:
            return state

        calibration = calibrate_gap(T_c=state.material.T_c, T_bath=state.T_bath)
        new_gap = solve_gap(calibration, state.f, state.spectral.E)

        if abs(new_gap - state.gap) < 1e-14:
            return state

        gap_dot = (new_gap - state.gap) / dt
        u_old = state.spectral.rho * state.f
        u_new = advect_spectral_flow(
            u_old, state.spectral.E, state.spectral.dE,
            gap=state.gap, gap_dot=gap_dot, dt=dt,
            active_mask=state.spectral.active_mask,
        )

        # Preserve every non-gap configuration from the incoming
        # SpectralContext: the caller may have set a custom
        # diffusion_coefficient (not just material.D_0), dynes_gamma,
        # rebuild_tolerance, or active_margin_factor. Only Δ changes.
        new_spectral = self._rebuild_spectral_context(
            state.spectral, new_gap=new_gap,
        )

        rho_new = new_spectral.rho
        f_new = np.zeros_like(u_new)
        mask = rho_new > 1e-30
        f_new[mask] = u_new[mask] / rho_new[mask]
        f_new = np.clip(f_new, 0.0, 1.0)

        return replace(state, gap=new_gap, spectral=new_spectral, f=f_new)

    def step(
        self,
        state: T3DiffusionState,
        dt: float,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
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
