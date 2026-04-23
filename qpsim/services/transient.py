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

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState


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
    n_steps: int                # total ETD2 substeps taken
    converged: bool             # True iff stop_tol was met mid-run


def run_time_dependent(
    state: T3DiffusionState,
    *,
    dt: float,
    total_time: float,
    photon_params: dict[str, float] | None = None,
    pb_photon_params: dict[str, float] | None = None,
    snapshot_interval: float | None = None,
    observables: dict[str, Callable[[T3DiffusionState], float]] | None = None,
    stop_tol: float | None = None,
    backend: T3DiffusionBackend | None = None,
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
    snapshot_interval
        Time between saved snapshots (ns). Defaults to
        ``total_time / 50``. Snapshots are written when the running
        time first crosses each interval boundary, so the actual
        snapshot times may land slightly after the requested
        boundaries (by at most ``dt``).
    observables
        Optional dict ``{name: fn(state) → float}``. Each snapshot's
        ``observables`` dict is populated with the current values —
        useful for plotting ``x_qp(t)``, ``Q_i(t)``, etc. without
        retaining the full ``f`` array at every snapshot.
    stop_tol
        Optional early-termination threshold on
        ``max|f_new - f_old| / dt``. When the instantaneous rate-of-
        change falls below this value, the driver returns early with
        ``converged=True``. ``None`` disables early stopping.
    backend
        T3 backend instance. Defaults to a fresh
        :class:`T3DiffusionBackend`.

    Returns
    -------
    TransientResult
        Time series of snapshots plus convergence metadata.

    Raises
    ------
    ValueError
        For non-physical inputs (``dt ≤ 0``, ``total_time ≤ 0``,
        ``snapshot_interval ≤ 0``, or ``stop_tol < 0``).
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if total_time <= 0:
        raise ValueError("total_time must be positive.")
    if snapshot_interval is not None and snapshot_interval <= 0:
        raise ValueError("snapshot_interval must be positive when provided.")
    if stop_tol is not None and stop_tol < 0:
        raise ValueError("stop_tol must be non-negative when provided.")

    if backend is None:
        backend = T3DiffusionBackend()
    if snapshot_interval is None:
        snapshot_interval = total_time / 50.0

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
    converged = False
    current = state

    # Integer step count keeps the final time exactly at N·dt (or
    # closest multiple) rather than drifting with float accumulation.
    max_steps = int(np.ceil(total_time / dt))

    for _ in range(max_steps):
        prev_f = current.f
        current = backend.apply_collisions(
            current, dt,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
        )
        t += dt
        n_steps += 1

        if stop_tol is not None:
            rate = float(np.max(np.abs(current.f - prev_f)) / dt)
            if rate < stop_tol:
                converged = True
                snapshots.append(_snapshot(t, current))
                break

        if t >= next_snap - 1e-12:
            snapshots.append(_snapshot(t, current))
            next_snap += snapshot_interval

    if snapshots[-1].t < t:
        # Final state wasn't captured by the snapshot cadence; append it.
        snapshots.append(_snapshot(t, current))

    return TransientResult(
        snapshots=snapshots,
        total_time=t,
        n_steps=n_steps,
        converged=converged,
    )
