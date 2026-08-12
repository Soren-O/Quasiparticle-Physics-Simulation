r"""Transient driver: evolve ``f(E, t)`` from an initial state.

V1 scope: collisional relaxation at frozen ``n_ph`` and frozen ``Δ``.
Repeated ETD2 collision substeps produce a time series of ``f(E)``
snapshots the caller can post-process.

What's *not* in v1
------------------
* No phonon dynamics — ``n_ph`` stays at whatever the initial state
  carries. For coupled ``(f, n_ph)`` steady state, use
  :func:`qpsim.services.steady_state.solve_steady_state` or the
  backend's ``steady_state(method="coupled_newton")``.
* No transport — ``apply_transport`` is a no-op in the v1
  homogeneous backend (real Crank-Nicolson diffusion lands at Gate 5).
* No gap update — ``Δ`` is held fixed. A self-consistent-gap transient
  would need the spectral-flow advection wired into the time loop
  (``apply_gap_update``); v1 transient leaves that off for simplicity.

Use cases
---------
* Relaxation to steady state from a non-equilibrium initial ``f``.
* Response to a sudden drive change (photon kick: evolve from
  ``f_FD`` with the drive on).
* Sanity-checking that the steady-state solver's fixed point is
  dynamically stable under the same collision operator.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.devices.external_flux import ExternalFlux

# Hard backstop on emitted snapshots. The webui schema rejects dense cadences
# up front (``_MAX_SNAPSHOTS`` there); this catches direct callers so a
# ``snapshot_interval`` far below ``dt`` fails loud instead of draining an
# unbounded inner loop (a ``1e-300`` interval would otherwise hang forever).
_SNAPSHOT_HARD_CAP = 200_000


@dataclass(frozen=True)
class TransientSnapshot:
    """One entry in a time series of ``f(E)``."""

    t: float                    # simulation time (ns)
    f: np.ndarray               # shape (NE,) — f(E) at this time
    observables: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TransientResult:
    """Output of :func:`run_time_dependent`."""

    snapshots: list[TransientSnapshot]
    total_time: float           # last snapshot time actually reached (ns)
    n_steps: int                # driver-level steps (ETD2 may rate-subcycle internally)
    n_etd_substeps: int         # actual ETD2 substeps when backend exposes diagnostics
    converged: bool             # True iff stop_tol was met mid-run


def run_time_dependent(
    state: T3DiffusionState,
    *,
    dt: float,
    total_time: float,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    external_flux: ExternalFlux | Callable[[float], ExternalFlux] | None = None,
    snapshot_interval: float | None = None,
    observables: dict[str, Callable[[T3DiffusionState], float]] | None = None,
    stop_tol: float | None = None,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
    backend: T3DiffusionBackend | None = None,
    progress_hook: Callable[[float, float], bool] | None = None,
) -> TransientResult:
    r"""Evolve ``f(E)`` under repeated collision substeps.

    Parameters
    ----------
    state
        Initial T3 state — ``state.phonon.n_ph`` must already be on
        the physics ω-grid (``backend.steady_state`` enforces this by
        construction; if you hand-built a state, call
        :func:`qpsim.collisions.phonon.build_phonon_frequency_map`
        first).
    dt
        ETD2 substep (ns). Too small → slow; too large → ETD2 linear
        stability limits kick in. Empirically ``dt ≲ τ_0 / 10`` is
        safe for Fischer parameters.
    total_time
        Total simulation time (ns).
    photon_params, pb_photon_params
        Photon-channel dicts, same structure the backend's
        ``apply_collisions`` expects. Drive is constant across the
        transient (no built-in pulse shaping — for a drive step, run
        two transients and splice).
    external_flux
        Optional :class:`qpsim.devices.ExternalFlux` boundary
        source/sink contract. Either a static instance applied at
        every substep, or a callable ``f(t) -> ExternalFlux`` that
        returns the flux at the midpoint of the current substep (for
        time-varying junction couplings). Midpoint sampling preserves the
        second-order time centering of ETD2. A callable cannot be combined
        with ``stop_tol`` because an instantaneous residual cannot certify
        future convergence of a non-autonomous problem. ``None`` disables.
    snapshot_interval
        Time between saved snapshots (ns). Defaults to
        ``total_time / 50``. When a substep crosses one or more interval
        boundaries, snapshots are linearly interpolated between its ETD2
        endpoints, which preserves the requested cadence without changing the
        integration step. Only the step endpoints ``t = k·dt`` carry the
        integrator's accuracy: the underlying relaxation is exponential, so an
        interpolated ``f`` (and every observable derived from it) is faithful
        only while a step relaxes ``f`` by much less than itself, and its error
        grows to O(1) once a step spans a relaxation time. A cadence finer than
        ``dt`` therefore warns — shorten ``dt`` to the resolution you want,
        which the internal rate subcycling makes nearly free.
    observables
        Optional dict ``{name: fn(state) → float}``. Each snapshot's
        ``observables`` dict is populated with the current values —
        useful for plotting ``x_qp(t)``, ``Q_i(t)``, etc. without
        retaining the full ``f`` array at every snapshot.
    stop_tol
        Optional early-termination threshold on the raw instantaneous
        collision residual ``max|df/dt|``.  Inspecting the raw RHS prevents
        clipping at ``f=0`` or ``f=1`` from masquerading as convergence.
        This convergence certificate is defined only for an autonomous
        problem, so ``external_flux`` must be ``None`` or a constant
        :class:`ExternalFlux`, not a callable. Custom backends can provide an
        exact certificate via
        ``apply_collisions_with_diagnostics``; otherwise a finite-difference
        fallback is used, which refuses to certify a step that saturated at a
        bound (``f`` at 1, or ``f`` driven down onto 0) but does accept a
        legitimately tiny occupation.
        ``None`` disables early stopping.
    backend
        T3 backend instance. Defaults to a fresh
        :class:`T3DiffusionBackend`.
    progress_hook
        Optional physics-neutral driver hook, called after every
        driver-level step with ``(t, total_time)``. Return ``True`` to continue;
        returning ``False`` stops the run cleanly at the current time,
        exactly as if ``total_time`` had been reached there (the final
        state is still snapshotted; ``converged`` stays ``False``
        unless ``stop_tol`` was already met). Intended for progress
        reporting and cooperative cancellation from interactive
        callers. ``None`` (the default) leaves the time loop
        bit-for-bit unchanged.

    Returns
    -------
    TransientResult
        Time series of snapshots plus convergence metadata.

    Raises
    ------
    ValueError
        For non-physical inputs (``dt ≤ 0``, ``total_time ≤ 0``,
        ``snapshot_interval ≤ 0``, or ``stop_tol < 0``), or when
        ``stop_tol`` is combined with a callable ``external_flux``.
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive.")
    if not np.isfinite(total_time) or total_time <= 0:
        raise ValueError("total_time must be positive.")
    if snapshot_interval is not None and (
        not np.isfinite(snapshot_interval) or snapshot_interval <= 0
    ):
        raise ValueError("snapshot_interval must be positive when provided.")
    if stop_tol is not None and (not np.isfinite(stop_tol) or stop_tol < 0):
        raise ValueError("stop_tol must be non-negative when provided.")
    if stop_tol is not None and callable(external_flux):
        raise ValueError(
            "stop_tol cannot be used with callable external_flux: an "
            "instantaneous residual cannot certify convergence of a "
            "non-autonomous problem. Supply a constant ExternalFlux for "
            "autonomous convergence, or omit stop_tol."
        )

    if backend is None:
        backend = T3DiffusionBackend()
    if snapshot_interval is None:
        snapshot_interval = total_time / 50.0
    if snapshot_interval < dt:
        # Interior snapshots are linear dense output between ETD2 endpoints,
        # which is only first-order consistent with exponential relaxation.
        # Endpoints stay exact, so a coarse dt looks free on the final state
        # while every interpolated point drifts — say so instead of shipping
        # a piecewise-linear curve that reports the step size as a lifetime.
        warnings.warn(
            f"snapshot_interval={snapshot_interval:g} ns is finer than "
            f"dt={dt:g} ns: interior snapshots are linearly interpolated "
            "between integration endpoints and their f(E) — plus every "
            "observable computed from it — is accurate only while a step "
            "relaxes f by much less than itself. Set dt ≤ snapshot_interval "
            "to resolve the requested cadence.",
            UserWarning,
            stacklevel=2,
        )

    def _snapshot(t: float, s: T3DiffusionState) -> TransientSnapshot:
        obs = (
            {name: float(fn(s)) for name, fn in observables.items()}
            if observables
            else {}
        )
        return TransientSnapshot(t=float(t), f=s.f.copy(), observables=obs)

    snapshots: list[TransientSnapshot] = [_snapshot(0.0, state)]
    t = 0.0
    next_snap = snapshot_interval
    n_steps = 0
    n_etd_substeps = 0
    converged = False
    current = state

    # Integer cap avoids unbounded loops, while the final substep is
    # shortened so the transient lands exactly on total_time.
    max_steps = int(np.ceil(total_time / dt))

    NE = int(state.f.size)

    def _flux_at(t_now: float) -> ExternalFlux | None:
        if external_flux is None:
            return None
        ef = external_flux(t_now) if callable(external_flux) else external_flux
        ef._validate_for_NE(NE)
        ef._validate_gain_support(state.spectral.active_mask)
        return ef

    for _ in range(max_steps):
        remaining = total_time - t
        if remaining <= 1e-12:
            break
        step_dt = min(dt, remaining)
        t_previous = t
        prev_f = current.f.copy()
        # Pass a term flag only when it is off: a third-party backend that
        # predates these kwargs must keep working at the defaults, and must
        # fail loudly rather than silently ignore a term you switched off.
        term_kwargs: dict[str, bool] = {}
        if not enable_scattering:
            term_kwargs["enable_scattering"] = False
        if not enable_recombination:
            term_kwargs["enable_recombination"] = False
        step_flux = _flux_at(t + 0.5 * step_dt)
        raw_rate: np.ndarray | None = None
        diagnostics_method = getattr(
            backend, "apply_collisions_with_diagnostics", None,
        )
        method_owner = getattr(
            type(backend), "apply_collisions_with_diagnostics", None,
        )
        has_trustworthy_diagnostics = callable(diagnostics_method) and (
            type(backend) is T3DiffusionBackend
            or method_owner is not T3DiffusionBackend.apply_collisions_with_diagnostics
        )
        if has_trustworthy_diagnostics:
            assert callable(diagnostics_method)
            current, raw_rate, step_etd_substeps = diagnostics_method(
                current,
                step_dt,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=step_flux,
                **term_kwargs,
                evaluate_residual=stop_tol is not None,
            )
        else:
            current = backend.apply_collisions(
                current, step_dt,
                photon_params=photon_params,
                pb_photon_params=pb_photon_params,
                external_flux=step_flux,
                **term_kwargs,
            )
            step_etd_substeps = 1
        t += step_dt
        n_steps += 1
        n_etd_substeps += int(step_etd_substeps)

        # Emit every crossed cadence boundary. Linear dense output between
        # second-order step endpoints avoids dropping intervals when dt is
        # larger than snapshot_interval without changing the integration grid.
        # It is only first-order consistent with the exponential relaxation
        # the ETD steps solve, so an interior point is trustworthy only for
        # dt well inside the local relaxation time (the constructor warns
        # otherwise). Only ``f`` is interpolated; the remaining fields carry
        # their end-of-step values, which is exact while v1 holds n_ph and Δ
        # frozen and must be revisited when either becomes dynamic.
        time_tol = 16.0 * np.finfo(float).eps * max(
            1.0, abs(t), abs(next_snap),
        )
        while next_snap <= t + time_tol:
            # Bound the cadence drain: a snapshot_interval far below step_dt
            # would otherwise loop unboundedly (a 1e-300 interval never advances
            # next_snap past t). The webui schema rejects such setups up front;
            # this fails loud for direct callers instead of hanging.
            if len(snapshots) >= _SNAPSHOT_HARD_CAP:
                raise ValueError(
                    f"snapshot_interval={snapshot_interval:g} ns is too small for "
                    f"total_time={total_time:g} ns: the cadence would emit more than "
                    f"{_SNAPSHOT_HARD_CAP} snapshots. Increase snapshot_interval."
                )
            # A cadence boundary can round a few ulps beyond an exactly
            # coincident step/terminal boundary. Snap that timestamp to the
            # endpoint; never extrapolate f beyond the state we integrated.
            snapshot_time = min(next_snap, t)
            fraction = (snapshot_time - t_previous) / step_dt
            f_snapshot = prev_f + fraction * (current.f - prev_f)
            snapshots.append(
                _snapshot(snapshot_time, replace(current, f=f_snapshot))
            )
            next_snap += snapshot_interval

        if stop_tol is not None:
            if raw_rate is not None:
                rate = float(np.max(np.abs(raw_rate)))
            else:
                # A custom backend has no public raw-RHS contract.  Retain the
                # finite-difference fallback only for interior states; at a
                # clipped bound it cannot certify complementarity and must not
                # claim convergence.  "At a bound" means the step actually
                # saturated there.  f → 1 always disqualifies: the blocked gain
                # G·(1−f) it hides is unbounded.  A merely tiny f does not — a
                # thermal tail is legitimately far below 32·eps in most bins of
                # a mK grid, and a low clip can only hide loss·f, i.e. ≲1e-14/ns
                # at physical rates, orders below any usable stop_tol.  Testing
                # smallness instead of saturation pinned the fallback at ``inf``
                # for every cold state and disabled early stopping outright.
                rate = float(np.max(np.abs(current.f - prev_f)) / step_dt)
                bound_tol = 32.0 * np.finfo(float).eps
                clipped_low = (prev_f > 0.0) & (current.f <= 0.0)
                if np.any(current.f >= 1.0 - bound_tol) or np.any(clipped_low):
                    rate = np.inf
            if rate < stop_tol:
                converged = True
                if snapshots[-1].t < t - 1e-12:
                    snapshots.append(_snapshot(t, current))
                break

        if progress_hook is not None and not progress_hook(t, total_time):
            break

    if snapshots[-1].t < t:
        # Final state wasn't captured by the snapshot cadence; append it.
        snapshots.append(_snapshot(t, current))

    return TransientResult(
        snapshots=snapshots,
        total_time=t,
        n_steps=n_steps,
        n_etd_substeps=n_etd_substeps,
        converged=converged,
    )
