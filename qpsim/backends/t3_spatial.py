"""T3 kinetics on an arbitrary geometry: transport composed with collisions.

One backend for every dimensionality. The geometry is a mask, so a single
cell is 0-D, a one-cell-wide mask is a 1-D strip, and anything else is 2-D --
configurations of the same object rather than separate code paths.

Composes :class:`qpsim.transport.spatial_transport.SpatialTransport` with
:class:`qpsim.collisions.spatial.SpatialCollisions` in a symmetric split,
matching the 1-D backend it generalises:

    transport(dt/2) -> collisions(dt) -> transport(dt/2)

``qpsim/backends/t3_spatial_1d.py`` stays in the tree until this is
benchmark-certified. A one-cell-wide geometry here must reproduce it; that is
the acceptance gate, and it is sharper than any single analytic check because
that backend already carries validated face weights, subcycling and interface
handling.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from qpsim.collisions.spatial import SpatialCollisions
from qpsim.fields.drive import ExternalDrive, StaticDrive
from qpsim.geometries import Geometry
from qpsim.grid.spatial_grid import BoundaryCondition
from qpsim.materials.database import Material
from qpsim.physics.bcs_quadrature import (
    bcs_support_fraction,
    represented_bcs_weights,
)
from qpsim.physics.gap_equation import calibrate_gap, solve_gap
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import (
    DiffusionModel,
    density_weight,
    flux_weight,
)
from qpsim.transport.interface import interface_face_conductances
from qpsim.transport.spatial_operator import face_condition_lookup
from qpsim.transport.spatial_transport import (
    EnergyTransportOp,
    SpatialTransport,
)

__all__ = ["T3SpatialBackend", "T3SpatialState"]


@dataclass
class T3SpatialState:
    """Occupation ``f(E, cell)`` on a geometry.

    ``f`` is ``(NE, Ncells)`` with cells in the geometry's mask order, i.e.
    row-major over ``mask.nonzero()``. ``gap_per_cell`` gives the local gap so
    the density of states, the transport dressing and the collision kernels
    all agree cell by cell; ``None`` means the spectral context's scalar gap
    everywhere.
    """

    f: np.ndarray
    geometry: Geometry
    spectral: SpectralContext
    material: Material
    T_bath: float
    gap_per_cell: np.ndarray | None = None
    conditions: dict[str, BoundaryCondition] | None = None
    diffusion_model: DiffusionModel = DiffusionModel.A1
    interface_conductance: float | None = None
    _: dict[str, float] = field(default_factory=dict, repr=False)

    @property
    def gap(self) -> float:
        """The scalar reference gap. Local gaps live in ``gap_per_cell``."""
        return float(self.spectral.gap)

    def gaps(self) -> np.ndarray:
        """Local gap per cell, as an array either way."""
        if self.gap_per_cell is None:
            return np.full(self.f.shape[1], float(self.spectral.gap))
        return np.asarray(self.gap_per_cell, dtype=float)

    def boundary(self) -> dict[str, BoundaryCondition]:
        """Declared conditions, defaulting to a reflective device."""
        return (
            self.geometry.conditions() if self.conditions is None
            else self.conditions
        )


@dataclass(frozen=True)
class SpatialSnapshot:
    """One recorded frame of a spatial run.

    Carries the whole field, not a reduction of it: a per-energy map and a
    phonon map are questions a reader asks after the run, and a scalar
    recorded now cannot answer them later.
    """

    t: float                          # simulation time (ns)
    f: np.ndarray                     # (NE, Ncells)
    gap_per_cell: np.ndarray          # (Ncells,)
    n_ph: np.ndarray | None           # (Nomega, Ncells), None while pinned
    max_rate: float                   # residual at this frame (1/ns)


@dataclass(frozen=True)
class SpatialRunResult:
    """Outcome of :meth:`T3SpatialBackend.run`."""

    state: T3SpatialState
    n_steps: int
    converged: bool
    last_max_rate: float
    elapsed: float                    # simulation time actually reached (ns)
    snapshots: list[SpatialSnapshot] = field(default_factory=list)


# Backstop on emitted frames. Each is a full (NE, Ncells) field plus a phonon
# map, so an accidentally fine cadence exhausts memory rather than merely
# running slowly; fail loud instead.
_SNAPSHOT_HARD_CAP = 4_000


class T3SpatialBackend:
    """Split-step T3 kinetics on a geometry of any dimensionality."""

    def __init__(
        self,
        *,
        enable_scattering: bool = True,
        enable_recombination: bool = True,
        enable_phonon_scattering_source: bool = True,
        enable_phonon_recombination_source: bool = True,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        phonon_escape_time: float | None = None,
        phonon_seed: np.ndarray | None = None,
    ) -> None:
        self.enable_scattering = bool(enable_scattering)
        self.enable_recombination = bool(enable_recombination)
        self.enable_phonon_scattering_source = bool(enable_phonon_scattering_source)
        self.enable_phonon_recombination_source = bool(
            enable_phonon_recombination_source
        )
        self.photon_params = photon_params
        self.pb_photon_params = pb_photon_params
        self.phonon_escape_time = phonon_escape_time
        # Applied only when the collision layer is first built; a later
        # rebuild carries the EVOLVED population forward instead (see
        # _collisions_for), because re-seeding mid-run would silently discard
        # the history the run is measuring.
        self.phonon_seed = phonon_seed
        self._transport: SpatialTransport | None = None
        self._collisions: SpatialCollisions | None = None
        self._transport_signature: object = None
        self._collision_signature: object = None

    # -- per-cell spectral quantities -------------------------------------

    @staticmethod
    def _per_cell(
        spectral: SpectralContext,
        gaps: np.ndarray,
        kind: str,
    ) -> np.ndarray:
        """``(NE, Ncells)`` spectral weight, one integration per distinct gap.

        Cells are grouped by exact gap first, so a piecewise-constant profile
        costs one quadrature per material rather than one per cell -- which is
        what keeps a fine 2-D mesh affordable.
        """
        energies, widths = spectral.E, spectral.dE
        distinct, group_index = np.unique(gaps, return_inverse=True)
        if kind == "density":
            columns = [
                represented_bcs_weights(energies, widths, float(g)) / widths
                for g in distinct
            ]
        elif kind == "support":
            columns = [
                bcs_support_fraction(energies, widths, float(g))
                for g in distinct
            ]
        else:  # pragma: no cover - internal
            raise ValueError(f"unknown per-cell weight {kind!r}")
        return np.column_stack(columns)[:, group_index]

    def _transport_weights(
        self, state: T3SpatialState,
    ) -> tuple[np.ndarray, np.ndarray]:
        gaps = state.gaps()
        n1 = self._per_cell(state.spectral, gaps, "density")
        model = state.diffusion_model
        if model.q == 0:
            # Dirty-limit D_L is an above-gap indicator. Its exact
            # finite-volume average is the supported fraction of a cut cell,
            # not one merely because the cell has some capacity.
            weights = state.spectral.diffusion_coefficient * self._per_cell(
                state.spectral, gaps, "support",
            )
        else:
            weights = flux_weight(
                state.spectral.diffusion_coefficient, n1, model.q,
            )
        return weights, density_weight(n1, model.p)

    # -- the two halves ---------------------------------------------------

    def _transport_for(self, state: T3SpatialState) -> SpatialTransport:
        signature = (
            id(state.geometry), state.geometry.mask.tobytes(),
            float(state.geometry.mesh_size), id(state.conditions),
        )
        if self._transport is None or signature != self._transport_signature:
            self._transport = SpatialTransport(
                state.geometry.mask,
                face_condition_lookup(state.geometry.edges, state.boundary()),
                state.geometry.mesh_size,
            )
            self._transport_signature = signature
        return self._transport

    def _collisions_for(self, state: T3SpatialState) -> SpatialCollisions:
        gaps = state.gaps()
        signature = (
            gaps.tobytes(), float(state.T_bath), float(state.material.tau_0),
            float(state.material.T_c), state.spectral.E.tobytes(),
            repr(self.photon_params), repr(self.pb_photon_params),
            repr(self.phonon_escape_time),
            # In the signature so switching a phonon source rebuilds the
            # layer rather than reusing kernels built under the old setting.
            self.enable_phonon_scattering_source,
            self.enable_phonon_recombination_source,
        )
        if self._collisions is None or signature != self._collision_signature:
            previous = self._collisions
            self._collisions = SpatialCollisions(
                state.spectral, gaps,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
                T_bath=state.T_bath,
                enable_scattering=self.enable_scattering,
                enable_recombination=self.enable_recombination,
                enable_phonon_scattering_source=(
                    self.enable_phonon_scattering_source
                ),
                enable_phonon_recombination_source=(
                    self.enable_phonon_recombination_source
                ),
                photon_params=self.photon_params,
                pb_photon_params=self.pb_photon_params,
                phonon_escape_time=self.phonon_escape_time,
                phonon_seed=self.phonon_seed,
            )
            # The collision layer is a KERNEL CACHE keyed on the gap, but it
            # also owns n_ph, which is STATE. Rebuilding the cache must not
            # destroy the state, and it silently did: a self-consistent solve
            # moves the gap on every step, so every step re-seeded the evolved
            # phonon population back at the bath and phonon trapping -- the
            # entire reason the dynamic sector exists -- quietly vanished. A
            # gap change of one part in 10^6 was enough.
            #
            # The frequency grid comes from spectral.E, which is part of the
            # signature, so an unchanged signature component means the two
            # grids are identical and the population transfers as-is. When E
            # itself changed there is no correspondence to carry and the
            # thermal re-seed is correct.
            if (
                previous is not None
                and previous.n_ph is not None
                and self._collisions.n_ph is not None
                and previous.n_ph.shape == self._collisions.n_ph.shape
                and np.array_equal(previous.omega_bins, self._collisions.omega_bins)
            ):
                self._collisions.n_ph = previous.n_ph
            self._collision_signature = signature
        return self._collisions

    def _n1_per_cell(self, state: T3SpatialState) -> np.ndarray:
        """Finite-volume BCS density per energy and cell, ``(NE, Ncells)``.

        Same quantity and name as the 1-D backend's helper, so validation
        producers that read it keep working across the migration.
        """
        return self._per_cell(state.spectral, state.gaps(), "density")

    def _build_transport_operators(
        self, state: T3SpatialState, dt: float,
    ) -> list[EnergyTransportOp | None]:
        """Per-energy transport operators, in the 1-D backend's tuple order.

        Kept as a named entry point because validation producers inspect the
        operators directly -- the substep count in particular, which is a
        reported diagnostic of the Crank-Nicolson stiffness bound.
        """
        weights, density = self._transport_weights(state)
        return self._transport_for(state).build(weights, density, dt)

    def apply_transport(
        self, state: T3SpatialState, dt: float,
    ) -> T3SpatialState:
        """One CN transport step. A single cell has nothing to transport."""
        if state.f.shape[1] == 1:
            return state
        weights, density = self._transport_weights(state)
        transport = self._transport_for(state)

        conductance = state.interface_conductance
        overrides: dict[int, dict[tuple[int, int], float]] | None = None
        # The barrier conductance belongs in the key: two runs differing only
        # in it must not share an operator set.
        interface_key: tuple[float, int] | None = None
        if conductance is not None and state.gap_per_cell is not None:
            overrides = self._interface_overrides(state, float(conductance))
            interface_key = (float(conductance), len(overrides))

        ops = transport.build(
            weights, density, dt,
            cache_key=(weights.tobytes(), density.tobytes(), interface_key),
            face_overrides=overrides,
        )
        f_new, _diagnostics = transport.apply(state.f, ops)
        return replace(state, f=f_new)

    def _interface_overrides(
        self, state: T3SpatialState, conductance: float,
    ) -> dict[int, dict[tuple[int, int], float]]:
        """Kupriyanov-Lukichev face conductances, per energy bin."""
        gap_grid = np.zeros(state.geometry.mask.shape, dtype=float)
        gap_grid[state.geometry.mask] = state.gaps()
        weights, _density = self._transport_weights(state)
        overrides: dict[int, dict[tuple[int, int], float]] = {}
        rows, cols = np.nonzero(state.geometry.mask)
        for i in range(weights.shape[0]):
            active = weights[i] > 0.0
            if int(np.count_nonzero(active)) < 2:
                continue
            submask = np.zeros_like(state.geometry.mask)
            submask[rows[active], cols[active]] = True
            found = interface_face_conductances(
                submask, gap_grid, state.spectral.E, state.spectral.dE,
                float(conductance), state.geometry.mesh_size, i,
            )
            if found:
                overrides[i] = found
        return overrides

    def apply_collisions(
        self,
        state: T3SpatialState,
        dt: float,
        *,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> T3SpatialState:
        """One local ETD2 collision/source step at every cell."""
        collisions = self._collisions_for(state)
        f_new = collisions.apply(
            state.f, dt,
            external_gain=external_gain, external_loss=external_loss,
        )
        return replace(state, f=f_new)

    # -- composed step ----------------------------------------------------

    def step(
        self,
        state: T3SpatialState,
        dt: float,
        *,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> T3SpatialState:
        """Symmetric split step: transport/2, collisions+source, transport/2.

        A live phonon population advances after the collision half, at the new
        ``f`` -- the same Lie arrangement the 0-D transient uses, one level out.
        """
        s = self.apply_transport(state, 0.5 * dt)
        s = self.apply_collisions(
            s, dt, external_gain=external_gain, external_loss=external_loss,
        )
        if self.phonon_escape_time is not None:
            self._collisions_for(s).advance_phonons(s.f, dt)
        return self.apply_transport(s, 0.5 * dt)

    # -- residual and stepping loop ---------------------------------------

    def rates(
        self,
        state: T3SpatialState,
        *,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> np.ndarray:
        """Endpoint ``df/dt`` from transport and collisions together.

        Used as the convergence certificate. A finite difference of the last
        step would be cheaper and wrong in the case that matters: an occupation
        pinned at a clip bound stops changing while its governing operator is
        still nonzero, and would then be reported as converged.
        """
        collisions = self._collisions_for(state)
        total = np.zeros_like(state.f)
        for gap, columns in collisions._groups():
            operator = collisions.local_operator(gap)
            gain, loss = collisions.combined_group_rates(
                state.f[:, columns], columns, operator, gap,
                None if external_gain is None else external_gain[:, columns],
                None if external_loss is None else external_loss[:, columns],
            )
            total[:, columns] = gain - loss * state.f[:, columns]

        if state.f.shape[1] > 1:
            weights, density = self._transport_weights(state)
            transport = self._transport_for(state)
            # dt only sets the substep count, which the rate does not use.
            for i, op in enumerate(transport.build(weights, density, 1.0)):
                if op is None:
                    continue
                _b, _lu, idx, rho_p, _n, _forcing, operator = op
                # The state carried is f while the operator acts on the
                # conserved density u = rho_p f, so divide back through.
                u = rho_p * state.f[i, idx]
                total[i, idx] += (operator @ u) / rho_p
        return total

    def run(
        self,
        state: T3SpatialState,
        *,
        dt: float,
        max_time: float,
        stop_tol: float = 1e-10,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
        drive: ExternalDrive | None = None,
        self_consistent_gap: bool = False,
        gap_quantum: float | None = None,
        snapshot_interval: float | None = None,
        progress_hook: Callable[[float, float], bool] | None = None,
    ) -> SpatialRunResult:
        """Fixed-step dynamics until the residual falls below ``stop_tol``.

        ``snapshot_interval`` records the evolving field on the way, which is
        the difference between a solver and an instrument. Without it the only
        observable is the endpoint, so "has it reached steady state" and "how
        fast does the front move" are both unanswerable, and a run still
        slowly drifting looks exactly like one that has settled.

        Snapshots are taken at step boundaries at or after each requested
        time, and each records the time actually reached rather than the time
        asked for: interpolating a whole ``(NE, Ncells)`` field would be both
        expensive and a claim about a state that was never computed.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")
        if snapshot_interval is not None and (
            not np.isfinite(snapshot_interval) or snapshot_interval <= 0.0
        ):
            raise ValueError("snapshot_interval must be positive when provided.")
        if drive is not None and (
            external_gain is not None or external_loss is not None
        ):
            raise ValueError(
                "Pass either a drive or external_gain/external_loss, not both: "
                "silently summing them would make the applied source depend on "
                "which argument the caller happened to reach for."
            )
        if drive is None:
            drive = StaticDrive(external_gain, external_loss)
        static_drive = drive.is_static
        # A time-dependent drive has no steady state to find, and its residual
        # passes through zero whenever the drive does -- so an early stop on
        # stop_tol would end the run in the gap between two pulses and report
        # it as converged. Run the requested time instead.
        if not static_drive:
            stop_tol = 0.0
        gain, loss = drive.sample(0.0)

        snapshots: list[SpatialSnapshot] = []
        capture = snapshot_interval is not None
        if capture:
            expected = int(max_time / float(snapshot_interval)) + 2
            if expected > _SNAPSHOT_HARD_CAP:
                raise ValueError(
                    f"snapshot_interval={snapshot_interval:g} ns over "
                    f"max_time={max_time:g} ns would emit about {expected} "
                    f"frames of the full field, past the {_SNAPSHOT_HARD_CAP} "
                    "cap. Coarsen the interval."
                )
            # The initial residual is a real measurement -- it says how far
            # from steady the run started -- and recording it as inf would put
            # a non-finite value in the first row of every exported series.
            snapshots.append(self._snapshot(
                state, 0.0,
                float(np.max(np.abs(self.rates(
                    state, external_gain=gain, external_loss=loss,
                )))),
            ))
        next_capture = float(snapshot_interval) if capture else float("inf")

        current = state
        elapsed = 0.0
        n_steps = 0
        converged = False
        last_rate = float("inf")
        for _ in range(int(np.ceil(max_time / dt))):
            remaining = max_time - elapsed
            if remaining <= 1e-12:
                break
            step_dt = min(dt, remaining)
            if not static_drive:
                # Midpoint, matching the Strang composition around it: a drive
                # sampled at the left endpoint drops the whole scheme to first
                # order and systematically lags a rising pulse.
                gain, loss = drive.sample(elapsed + 0.5 * step_dt)
            current = self.step(
                current, step_dt, external_gain=gain, external_loss=loss,
            )
            if self_consistent_gap:
                # Re-solve the gap against the occupation the step produced,
                # so the well follows the population rather than lagging a
                # whole run behind it.
                current, _iters, _change = self.relax_gap(
                    current, quantum=gap_quantum,
                )
            elapsed += step_dt
            n_steps += 1
            if not static_drive:
                gain, loss = drive.sample(elapsed)
            last_rate = float(np.max(np.abs(
                self.rates(current, external_gain=gain, external_loss=loss)
            )))
            if elapsed >= next_capture:
                snapshots.append(self._snapshot(current, elapsed, last_rate))
                # Advance past every boundary already overtaken, so a coarse
                # dt against a fine interval emits one frame per step instead
                # of spinning out a backlog of duplicates at the same time.
                while next_capture <= elapsed:
                    next_capture += float(snapshot_interval)
            if last_rate < stop_tol:
                converged = True
                break
            if progress_hook is not None and not progress_hook(elapsed, max_time):
                break

        # The endpoint is the one frame a reader always wants, and a run that
        # converges or is cancelled between cadence boundaries would otherwise
        # end on a stale frame.
        if capture and (not snapshots or snapshots[-1].t < elapsed):
            snapshots.append(self._snapshot(current, elapsed, last_rate))
        return SpatialRunResult(
            state=current,
            n_steps=n_steps,
            converged=converged,
            last_max_rate=last_rate,
            elapsed=elapsed,
            snapshots=snapshots,
        )

    def _snapshot(
        self, state: T3SpatialState, t: float, max_rate: float
    ) -> SpatialSnapshot:
        """One frame, copied so a later step cannot rewrite recorded history.

        Goes through ``_collisions_for`` rather than reading the cached
        attribute: at t = 0 nothing has built the layer yet, so reading the
        attribute gave the first frame a phonon map of ``None`` and every
        consumer then treated the whole run as having no phonon history.
        """
        phonons = self._collisions_for(state).n_ph
        return SpatialSnapshot(
            t=float(t),
            f=state.f.copy(),
            gap_per_cell=state.gaps().copy(),
            n_ph=None if phonons is None else phonons.copy(),
            max_rate=float(max_rate),
        )

    # -- self-consistent gap ----------------------------------------------

    def solve_gaps(
        self,
        state: T3SpatialState,
        *,
        quantum: float | None = None,
    ) -> np.ndarray:
        """Local gap in every cell, from that cell's own occupation.

        Each cell solves the BCS closure against its own ``f``, so a hot
        region digs its own gap well. The result is quantised before it is
        used, because collisions group by exact gap and a continuous profile
        would give one group -- and one set of dense kernels -- per cell.
        See SpatialCollisions for the measured cost of the quantum.
        """
        calibration = calibrate_gap(
            state.material.T_c, state.T_bath, Delta_0=state.material.Delta_0,
        )
        gaps = state.gaps()
        solved = np.empty_like(gaps)
        for cell in range(state.f.shape[1]):
            solved[cell] = solve_gap(
                calibration,
                state.f[:, cell],
                state.spectral.E,
                state.spectral.dE,
                reference_gap=float(gaps[cell]),
                allow_gap_edge_extrapolation=True,
            )
        if quantum is not None and quantum > 0.0:
            solved = np.round(solved / quantum) * quantum
        return solved

    def relax_gap(
        self,
        state: T3SpatialState,
        *,
        quantum: float | None = None,
        tol: float = 1e-9,
        max_iter: int = 20,
        mixing: float = 0.5,
    ) -> tuple[T3SpatialState, int, float]:
        """Iterate the gap to a fixed point at frozen ``f``.

        Under-relaxed because the closure is strongly nonlinear near the gap
        edge: a full step can oscillate between two roots rather than settle.
        Returns ``(state, iterations, last change)``.
        """
        current = state
        change = float("inf")
        for iteration in range(1, max_iter + 1):
            solved = self.solve_gaps(current, quantum=quantum)
            previous = current.gaps()
            blended = previous + mixing * (solved - previous)
            if quantum is not None and quantum > 0.0:
                blended = np.round(blended / quantum) * quantum
            change = float(np.max(np.abs(blended - previous)))
            current = replace(current, gap_per_cell=blended)
            # The collision layer is keyed on the gap profile, so a changed
            # profile must invalidate it or the next step runs the old
            # kernels. Clear the SIGNATURE, not the object: `_collisions_for`
            # needs the outgoing layer to carry the evolved phonon population
            # onto the new kernels. Dropping the object here is what used to
            # reset n_ph to the bath on every iteration of this loop.
            self._collision_signature = None
            if change <= tol:
                return current, iteration, change
        return current, max_iter, change
