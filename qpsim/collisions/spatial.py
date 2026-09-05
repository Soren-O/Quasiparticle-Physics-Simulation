"""Electron-phonon collisions over an arbitrary set of spatial cells.

Cells are grouped by their exact local gap, and each group is evaluated with a
spectral context and e-ph kernel built for that gap. This keeps the collision
support consistent with transport and stops a high-gap region inheriting
low-gap states and rates.

Nothing here knows the geometry's dimensionality: a group is a set of column
indices into an ``(NE, Ncells)`` occupation array, so the same code serves a
0-D cell, a 1-D strip and a 2-D mask. Copied and generalised from the 1-D
backend's collision half, which was already written this way.

A cost note that matters more in 2-D than it did in 1-D. Grouping is by EXACT
gap, so the work scales with the number of DISTINCT gaps, not with the number
of cells. A device made of two materials has two groups however fine the mesh.
A continuously varying gap -- a self-consistent solve, say -- has as many
groups as cells, and each group carries its own dense kernels. That was
tolerable when a 1-D strip had tens of cells; a 2-D mask has thousands.
:meth:`SpatialCollisions.distinct_gap_count` is exposed so a caller can check
before committing, and the constructor warns past a threshold rather than
quietly grinding.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from typing import Any

import numpy as np

from qpsim.collisions.pair_breaking_photon import (
    pair_breaking_photon_collision_rates,
)
from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
    validate_phonon_lattice_coupling,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.etd import etd2_step

__all__ = ["CollisionOperator", "SpatialCollisions"]

# Groups sharing one batched ETD2 call. Above this the driver streams one
# group at a time, which bounds resident dense-kernel memory independently of
# the cell count.
_MAX_BATCHED_GAPS = 2
# Local operators held at once. Each owns three dense (NE, NE) matrices.
_MAX_CACHED_OPERATORS = 2
# Past this many distinct gaps the exact-gap grouping is the wrong tool.
_GAP_COUNT_WARN = 64
# How much of an external source may land on unrepresented states before the
# run is refused. Not zero: an injection line is a Gaussian and has a tail
# everywhere, so zero would reject ordinary setups over e^-50. Small enough
# that a real truncation -- the gap-step case discards tens of percent -- can
# never slip through.
_MAX_DISCARDED_GAIN_FRACTION = 1e-6

# (K_s_eff, K_r_emit, K_r_abs, cell weights, represented mask)
CollisionOperator = tuple[
    np.ndarray | None, np.ndarray | None, np.ndarray | None,
    np.ndarray, np.ndarray,
]


class SpatialCollisions:
    """Local e-ph collisions for one material and bath, cached by local gap."""

    def __init__(
        self,
        spectral: SpectralContext,
        gap_per_cell: np.ndarray,
        *,
        tau_0: float,
        T_c: float,
        T_bath: float,
        enable_scattering: bool = True,
        enable_recombination: bool = True,
        enable_phonon_scattering_source: bool = True,
        enable_phonon_recombination_source: bool = True,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        phonon_escape_time: float | None = None,
        tau_0_pb_ns: float | None = None,
        use_phonon_side_kernel: bool = True,
        gap_quantum: float | None = None,
        phonon_seed: np.ndarray | None = None,
    ) -> None:
        if spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Spatial collisions require a pure-BCS SpectralContext "
                "(dynes_gamma == 0). A Dynes DOS needs consistently broadened "
                "normal/anomalous coherence functions, which this collision "
                "kernel does not implement."
            )
        self.spectral = spectral
        raw_gaps = np.asarray(gap_per_cell, dtype=float)
        # Grouping is by EXACT gap, so a continuously varying gap -- what a
        # self-consistent solve produces -- would give one group per cell, each
        # with its own dense kernels. Snapping to a quantum bounds that.
        #
        # The error is measured, not guessed. The kernel depends on the gap
        # through the coherence factor 1 - D^2/(E_i E_j) and the density of
        # states, both smooth: max|dK|/max|K| runs about 2e-2 per grid spacing
        # of gap shift, linearly. So a quantum of dE/10 costs ~2e-3, well under
        # the harmonic-mean face bias already carried. What is NOT smooth is
        # which bins sit above the gap: crossing one cell edge moves the kernel
        # by 3.3e-1. That discontinuity is the density-of-states support edge
        # and is physical, so the quantum should stay well inside a cell.
        self.gap_quantum = gap_quantum
        self.gap_per_cell = (
            raw_gaps if gap_quantum is None
            else np.round(raw_gaps / gap_quantum) * gap_quantum
        )
        self.tau_0 = float(tau_0)
        self.T_c = float(T_c)
        self.T_bath = float(T_bath)
        self.enable_scattering = bool(enable_scattering)
        self.enable_recombination = bool(enable_recombination)
        # The phonon side is booked separately. Each pair is ONE process
        # recorded on both sides of the ledger, and whether both sides are
        # recorded is exactly what these control.
        self.enable_phonon_scattering_source = bool(enable_phonon_scattering_source)
        self.enable_phonon_recombination_source = bool(
            enable_phonon_recombination_source
        )
        self.photon_params = photon_params
        self.pb_photon_params = pb_photon_params

        # None keeps the phonons pinned at the bath, which is the shipped
        # behaviour and what every published spatial result assumes. A value
        # solves the phonon population per cell; 0.0 is the engine's
        # no-substrate tau -> infinity sentinel, not instantaneous escape.
        self.phonon_escape_time = phonon_escape_time
        # The phonon equation is written on its OWN kernels (F&C 2023 Eq. 12).
        # Passing the quasiparticle-side matrices instead is a defect: they
        # carry an extra omega^2/(tau_0 (kB T_c)^3) and a tau_0 about 1700x
        # larger than tau_0^PB: measured n_ph 28x low at omega = 2 Delta and 6x
        # at 3.8 Delta, so the same setup lands on a different fixed point
        # here than in the steady-state solver. Same defect, same reason, as
        # crossing the phonon-source flags on this call.
        self.use_phonon_side_kernel = bool(use_phonon_side_kernel)
        self.tau_0_pb_ns = tau_0_pb_ns
        if phonon_escape_time is not None and self.use_phonon_side_kernel:
            if (
                tau_0_pb_ns is None
                or not np.isfinite(tau_0_pb_ns)
                or tau_0_pb_ns <= 0.0
            ):
                raise ValueError(
                    "A dynamic spatial phonon sector with "
                    "use_phonon_side_kernel=True requires a finite, positive "
                    f"tau_0_pb_ns; got {tau_0_pb_ns!r}. Set tau_0_pb_ns on the "
                    "Material, or pass use_phonon_side_kernel=False to reuse "
                    "the quasiparticle-side kernel as the legacy path did."
                )
        self.omega_bins, self._idx_diff, self._idx_sum, self._diff_sign = (
            build_phonon_frequency_map(spectral.E)
        )
        if phonon_escape_time is not None:
            # Enforced HERE, not only in the web-UI validator, because the
            # validator guards one entry point and this guards the model. A
            # script, a validation campaign or a notebook builds this layer
            # directly, and on a decoupled grid it would run to completion:
            # scattering phonons above the pair threshold with no recombination
            # partner never break a pair, recombination phonons are never
            # reabsorbed, and the two halves of Eq. 12 converge to DIFFERENT
            # limits under refinement. That is an inconsistent discretisation
            # returning a confident number, not a coarse one.
            #
            # Gated on a live phonon sector for the same reason the validator
            # is: a thermal bath never couples the channels through n_ph, so
            # the constraint does not apply and enforcing it there would reject
            # grids that are perfectly sound for the run being asked for.
            validate_phonon_lattice_coupling(
                self.omega_bins,
                self._idx_diff,
                self._idx_sum,
                self._diff_sign,
                E_bins=spectral.E,
            )
        self.n_ph: np.ndarray | None = None
        if phonon_escape_time is not None:
            # The bath value is both the default seed AND the target the
            # escape term relaxes toward, so a run left at the default starts
            # exactly at that term's own fixed point and nothing moves. That
            # makes the escape rate unmeasurable: a benchmark on it would pass
            # with the term switched off. `phonon_seed` is how a caller
            # prepares the departure the measurement needs.
            if phonon_seed is None:
                seed = thermal_phonon_occupation(self.omega_bins, self.T_bath)
                self.n_ph = np.repeat(
                    seed[:, None], self.gap_per_cell.size, axis=1,
                )
            else:
                seed = np.asarray(phonon_seed, dtype=float)
                shape = (self.omega_bins.size, self.gap_per_cell.size)
                if seed.ndim == 1:
                    seed = np.repeat(seed[:, None], shape[1], axis=1)
                if seed.shape != shape:
                    raise ValueError(
                        f"phonon_seed has shape {seed.shape}, expected "
                        f"{shape} = (frequency bins, cells)."
                    )
                if not np.all(np.isfinite(seed)) or np.any(seed < 0.0):
                    raise ValueError(
                        "phonon_seed must be finite and non-negative: a "
                        "negative occupation is not a colder phonon field."
                    )
                self.n_ph = seed.copy()

        self._distinct_gaps, self._group_index = np.unique(
            self.gap_per_cell, return_inverse=True,
        )
        self._cache: OrderedDict[Any, CollisionOperator] = OrderedDict()

        if self._distinct_gaps.size > _GAP_COUNT_WARN:
            spacing = float(np.min(spectral.dE))
            suggestion = (
                f"pass gap_quantum={spacing / 10.0:.4g} (dE/10, costing about "
                "2e-3 in the kernel)"
                if gap_quantum is None
                else f"the quantum is {gap_quantum:.4g}; coarsen it"
            )
            warnings.warn(
                f"{self._distinct_gaps.size} distinct local gaps across "
                f"{self.gap_per_cell.size} cells: collisions are grouped by "
                "EXACT gap, so each one builds and holds its own dense "
                f"kernels. To bound that, {suggestion}, or coarsen the mesh.",
                RuntimeWarning,
                stacklevel=2,
            )

    @property
    def distinct_gap_count(self) -> int:
        """Groups the collision step will evaluate."""
        return int(self._distinct_gaps.size)

    # -- operators --------------------------------------------------------

    def local_operator(self, local_gap: float) -> CollisionOperator:
        """Thermal e-ph matrices for one local gap, LRU-cached."""
        key = (
            float(local_gap), self.tau_0, self.T_c, self.T_bath,
            self.enable_scattering, self.enable_recombination,
            self.spectral.E.tobytes(), self.spectral.dE.tobytes(),
        )
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        # Evict before allocating: an entry already owns three dense matrices,
        # so building the new one first would hold both sets at once.
        while len(self._cache) >= _MAX_CACHED_OPERATORS:
            self._cache.popitem(last=False)

        spectral = self._local_spectral(local_gap)
        k_s_eff: np.ndarray | None = None
        if self.enable_scattering:
            k_s0 = build_scattering_kernel_base(
                spectral, tau_0=self.tau_0, T_c=self.T_c,
            )
            k_s_eff = k_s0 * _thermal_phonon_scattering_occupation(
                spectral.E, self.T_bath,
            )

        k_r_emit: np.ndarray | None = None
        k_r_abs: np.ndarray | None = None
        if self.enable_recombination:
            k_r0 = build_recombination_kernel_base(
                spectral, tau_0=self.tau_0, T_c=self.T_c,
            )
            n_emit, n_abs = _thermal_phonon_recombination_occupations(
                spectral.E, self.T_bath,
            )
            k_r_emit = k_r0 * n_emit
            k_r_abs = k_r0 * n_abs

        operator: CollisionOperator = (
            k_s_eff, k_r_emit, k_r_abs,
            np.asarray(spectral.cell_weights),
            np.asarray(spectral.active_mask),
        )
        self._cache[key] = operator
        return operator

    def raw_kernels(
        self, local_gap: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Undressed ``(K_s0, K_r0)`` for one local gap.

        The cached operator bakes the THERMAL occupations into its matrices,
        which is exactly right while the phonons are pinned and useless once
        they are not: a live n_ph differs cell by cell.
        """
        spectral = self._local_spectral(local_gap)
        k_s0 = (
            build_scattering_kernel_base(spectral, tau_0=self.tau_0, T_c=self.T_c)
            if self.enable_scattering else None
        )
        k_r0 = (
            build_recombination_kernel_base(spectral, tau_0=self.tau_0, T_c=self.T_c)
            if self.enable_recombination else None
        )
        return k_s0, k_r0

    def phonon_side_kernels(
        self, local_gap: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(K_s0, K_r0)`` for the PHONON equation at one local gap.

        Gated on the phonon-side switches, not the quasiparticle-side ones.
        That matters twice over: these kernels carry F&C Eq. 12's own
        prefactor ``1/(pi Delta tau_0^PB)`` rather than the QP-side
        ``omega^2/(tau_0 (kB T_c)^3)``, and supplying them is also what frees
        the phonon source from the QP-side flags -- with ``None`` the callee
        falls back to the QP matrices, so switching off quasiparticle
        scattering would switch off the phonon source with it.
        """
        if not self.use_phonon_side_kernel:
            return None, None
        spectral = self._local_spectral(local_gap)
        tau_0_pb = float(self.tau_0_pb_ns)  # __init__ has already validated it
        k_s0_ph = (
            build_scattering_kernel_phonon_side(spectral, tau_0_pb_ns=tau_0_pb)
            if self.enable_phonon_scattering_source else None
        )
        k_r0_ph = (
            build_recombination_kernel_phonon_side(spectral, tau_0_pb_ns=tau_0_pb)
            if self.enable_phonon_recombination_source else None
        )
        return k_s0_ph, k_r0_ph

    def dynamic_group_rates(
        self,
        f_group: np.ndarray,
        columns: np.ndarray,
        local_gap: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Collision rates against the live per-cell phonon population.

        Evaluated cell by cell: the occupation matrices are built from that
        cell's own n_ph, so nothing is shared across the group. This is the
        real cost of a solved phonon sector on a mesh, and it is why the
        thermal bath remains the default.
        """
        spectral = self._local_spectral(local_gap)
        k_s0, k_r0 = self.raw_kernels(local_gap)
        assert self.n_ph is not None
        gain = np.zeros_like(f_group)
        loss = np.zeros_like(f_group)
        for local, cell in enumerate(columns):
            n_p, n_emit, n_abs = phonon_occupation_matrices_from_state(
                self.n_ph[:, cell], self._idx_diff, self._idx_sum, self._diff_sign,
            )
            g, ell = phonon_collision_rates(
                f_group[:, local], spectral, k_s0, k_r0, self.T_bath,
                enable_scattering=self.enable_scattering,
                enable_recombination=self.enable_recombination,
                N_p_override=n_p, N_emit_override=n_emit, N_abs_override=n_abs,
            )
            gain[:, local] = g
            loss[:, local] = ell
        return gain, loss

    def advance_phonons(self, f: np.ndarray, dt: float) -> None:
        """Advance the per-cell phonon population over ``dt`` at fixed ``f``.

        Per cell the equation is diagonal in omega and affine,
            dn/dt = a_ph + b_ph*n + (n_th - n)/tau_l = A + B*n,
        solved exactly over the step. phi_1 goes through expm1 for the same
        reason the 0-D transient does: sub-2-Delta bins reach |B h| ~ 1e-7,
        where forming (exp(Bh) - 1) directly cancels to zero and silently
        deletes the whole source term.
        """
        if self.n_ph is None or self.phonon_escape_time is None:
            return
        n_th = thermal_phonon_occupation(self.omega_bins, self.T_bath)
        for gap, columns in self._groups():
            spectral = self._local_spectral(gap)
            k_s0, k_r0 = self.raw_kernels(gap)
            # Built once per gap group, not per cell: they depend on the local
            # gap and nothing else, while n_ph is what varies cell to cell.
            k_s0_ph, k_r0_ph = self.phonon_side_kernels(gap)
            for cell in columns:
                a_ph, b_ph = compute_phonon_source_sink(
                    f[:, cell], spectral, k_s0, k_r0,
                    self._idx_diff, self._idx_sum, self._diff_sign,
                    int(self.omega_bins.size),
                    # The PHONON-side switches, not the quasiparticle-side
                    # ones. Passing the latter left both phonon switches with
                    # nothing to act on -- bit-for-bit inert -- while tying the
                    # phonon source to a flag naming the other side of the
                    # ledger.
                    enable_scattering=self.enable_phonon_scattering_source,
                    enable_recombination=self.enable_phonon_recombination_source,
                    # F&C Eq. 12 kernels. Passing these also enables the Kaplan
                    # S+ threshold quadrature correction, which the callee gates
                    # on K_r0_phonon_side being present -- worth up to ~56% on
                    # the bins just above 2 Delta.
                    K_s0_phonon_side=k_s0_ph,
                    K_r0_phonon_side=k_r0_ph,
                )
                if self.phonon_escape_time == 0.0:
                    a_eff, b_eff = a_ph, b_ph
                else:
                    inv_tau = 1.0 / self.phonon_escape_time
                    a_eff = a_ph + inv_tau * n_th
                    b_eff = b_ph - inv_tau
                bh = b_eff * dt
                nonzero = b_eff != 0.0
                phi1 = np.where(
                    nonzero, np.expm1(bh) / np.where(nonzero, b_eff, 1.0), dt,
                )
                self.n_ph[:, cell] = np.clip(
                    self.n_ph[:, cell] * np.exp(bh) + a_eff * phi1, 0.0, None,
                )

    def _local_spectral(self, local_gap: float) -> SpectralContext:
        if float(local_gap) == float(self.spectral.gap):
            return self.spectral
        return SpectralContext(
            E_bins=self.spectral.E,
            dE_bins=self.spectral.dE,
            gap=float(local_gap),
            dynes_gamma=self.spectral.dynes_gamma,
            diffusion_coefficient=self.spectral.diffusion_coefficient,
            rebuild_tolerance=self.spectral.rebuild_tolerance,
            active_margin_factor=self.spectral.active_margin_factor,
        )

    # -- rates ------------------------------------------------------------

    def photon_rates(
        self, f_group: np.ndarray, local_gap: float,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Photon-channel gain and loss for one gap group, or ``None``.

        The kernels validate ``f`` against ``(NE,)`` -- they are written for a
        single spatial pixel -- so this evaluates them column by column. That
        is the honest cost of reusing the validated channel rather than
        writing a second, vectorised copy that would have to be validated
        again.
        """
        if self.photon_params is None and self.pb_photon_params is None:
            return None
        spectral = self._local_spectral(local_gap)
        gain = np.zeros_like(f_group)
        loss = np.zeros_like(f_group)
        for column in range(f_group.shape[1]):
            f_cell = f_group[:, column]
            if self.photon_params is not None:
                g, ell = sub_gap_photon_collision_rates(
                    f_cell, spectral,
                    self.photon_params["omega_0"],
                    self.photon_params["n_bar"],
                    self.photon_params["c_phot"],
                )
                gain[:, column] += g
                loss[:, column] += ell
            if self.pb_photon_params is not None:
                g, ell = pair_breaking_photon_collision_rates(
                    f_cell, spectral,
                    self.pb_photon_params["omega_PB"],
                    self.pb_photon_params["n_bar_PB"],
                    self.pb_photon_params["c_phot_PB"],
                )
                gain[:, column] += g
                loss[:, column] += ell
        return gain, loss

    @staticmethod
    def group_rates(
        f_group: np.ndarray,
        operator: CollisionOperator,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gain and loss for one exact-gap group."""
        k_s_eff, k_r_emit, k_r_abs, weights, physical = operator
        one_minus = np.maximum(1.0 - f_group, 0.0)

        if k_s_eff is not None:
            gain = one_minus * (k_s_eff.T @ (weights[:, None] * f_group))
            loss = k_s_eff @ (weights[:, None] * one_minus)
        else:
            gain = np.zeros_like(f_group)
            loss = np.zeros_like(f_group)

        # Kaplan Eq. (8) per-QP normalization -- see
        # qpsim.collisions.phonon.phonon_collision_rates.
        if k_r_emit is not None and k_r_abs is not None:
            loss += k_r_emit @ (weights[:, None] * f_group)
            gain += one_minus * (k_r_abs @ (weights[:, None] * one_minus))

        if external_gain is not None:
            gain = gain + external_gain
        if external_loss is not None:
            loss = loss + external_loss

        # A zero-capacity bin carries no represented state.
        unsupported = ~physical
        gain[unsupported, :] = 0.0
        loss[unsupported, :] = 0.0
        return gain, loss

    def combined_group_rates(
        self,
        f_group: np.ndarray,
        columns: np.ndarray,
        operator: CollisionOperator,
        local_gap: float,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """THE right-hand side for one gap group. One author, deliberately.

        Every term belongs here: the electron-phonon channels against either a
        pinned or a live phonon population, the photon drives, and any external
        source. The stepper and the convergence residual both call this, so
        they cannot disagree about what the model contains.

        That is not a stylistic preference. Assembling the residual separately
        omitted the external source once and the photon drives once, and in
        both cases a driven device was certified steady on its first step --
        the same bug twice, in two different terms.
        """
        if self.n_ph is not None:
            gain, loss = self.dynamic_group_rates(f_group, columns, local_gap)
        else:
            gain, loss = self.group_rates(f_group, operator)

        photon = self.photon_rates(f_group, local_gap)
        if photon is not None:
            gain = gain + photon[0]
            loss = loss + photon[1]
        unsupported = ~operator[4]
        # Refuse an external source aimed at states that do not exist here,
        # rather than deleting it below. Injection into an unrepresented bin is
        # undefined, and the diffusion backend raises on it too (via
        # external_flux._validate_gain_support). Silently zeroing it is how a
        # gap-step device could be asked for the same injection on both sides
        # and get it on only one, with nothing in the output saying so.
        if external_gain is not None:
            self._reject_unsupported_gain(external_gain, columns, unsupported)
            gain = gain + external_gain
        if external_loss is not None:
            loss = loss + external_loss

        gain[unsupported, :] = 0.0
        loss[unsupported, :] = 0.0
        return gain, loss

    @staticmethod
    def _reject_unsupported_gain(
        external_gain: np.ndarray,
        columns: np.ndarray,
        unsupported: np.ndarray,
    ) -> None:
        """Fail before advancing a group that would DISCARD a real source.

        The test is what fraction of the requested gain lands on states that
        do not exist here, not whether any lands there at all. A Gaussian
        injection line has an exponential tail on every bin -- at the shipped
        2.0 Delta centre and 0.1 Delta width it is ~e^-50 below the gap -- and
        refusing that would reject an ordinary setup over floating-point dust.
        What must not pass silently is a materially truncated source, which is
        what a gap-step device produces: injection asked for on both sides of
        the step and delivered on only one.
        """
        gain = np.asarray(external_gain, dtype=float)
        total = float(np.sum(np.abs(gain)))
        if total <= 0.0:
            return
        discarded = float(np.sum(np.abs(gain[unsupported, :])))
        fraction = discarded / total
        if fraction <= _MAX_DISCARDED_GAIN_FRACTION:
            return
        rows = np.flatnonzero(np.any(gain[unsupported, :] > 0.0, axis=1))
        bins = np.flatnonzero(unsupported)[rows][:5]
        preview = ", ".join(str(int(b)) for b in bins)
        raise ValueError(
            f"{fraction:.1%} of the external gain lands on zero-spectral-"
            f"capacity states (energy bins {preview}, ... at this cell's gap) "
            "and would be discarded. Injection into an unrepresented state is "
            "undefined: move the source onto supported energies, or widen the "
            "energy grid so those states exist at every gap in the device."
        )

    # -- stepping ---------------------------------------------------------

    def apply(
        self,
        f: np.ndarray,
        dt: float,
        *,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> np.ndarray:
        """One local ETD2 collision step over ``(NE, Ncells)`` occupations.

        One or two gap groups share a batched call; more are streamed one at a
        time so resident dense-kernel memory stays bounded however many cells
        the geometry has.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self._distinct_gaps.size <= _MAX_BATCHED_GAPS:
            return self._apply_batched(f, dt, external_gain, external_loss)
        return self._apply_streamed(f, dt, external_gain, external_loss)

    def _groups(self) -> list[tuple[float, np.ndarray]]:
        return [
            (float(gap), np.flatnonzero(self._group_index == g))
            for g, gap in enumerate(self._distinct_gaps)
        ]

    def _apply_batched(
        self,
        f: np.ndarray,
        dt: float,
        external_gain: np.ndarray | None,
        external_loss: np.ndarray | None,
    ) -> np.ndarray:
        groups = [
            (columns, self.local_operator(gap))
            for gap, columns in self._groups()
        ]
        balance_weights = np.zeros_like(f)
        for columns, operator in groups:
            balance_weights[:, columns] = operator[3][:, None]

        def rhs(current: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain = np.zeros_like(current)
            loss = np.zeros_like(current)
            for (columns, operator), (gap, _cols) in zip(
                groups, self._groups(), strict=True,
            ):
                g, ell = self.combined_group_rates(
                    current[:, columns], columns, operator, gap,
                    None if external_gain is None else external_gain[:, columns],
                    None if external_loss is None else external_loss[:, columns],
                )
                gain[:, columns] = g
                loss[:, columns] = ell
            return gain, loss

        return etd2_step(
            f, rhs, dt, balance_weights=balance_weights,
            balance_axis=0, max_loss_step=0.25,
        )

    def _apply_streamed(
        self,
        f: np.ndarray,
        dt: float,
        external_gain: np.ndarray | None,
        external_loss: np.ndarray | None,
    ) -> np.ndarray:
        updated = f.copy()
        for gap, columns in self._groups():
            operator = self.local_operator(gap)
            group_gain = None if external_gain is None else external_gain[:, columns]
            group_loss = None if external_loss is None else external_loss[:, columns]

            def rhs(
                f_group: np.ndarray,
                bound_operator: CollisionOperator = operator,
                bound_gain: np.ndarray | None = group_gain,
                bound_loss: np.ndarray | None = group_loss,
                bound_gap: float = gap,
                bound_columns: np.ndarray = columns,
            ) -> tuple[np.ndarray, np.ndarray]:
                return self.combined_group_rates(
                    f_group, bound_columns, bound_operator, bound_gap,
                    bound_gain, bound_loss,
                )

            f_group = f[:, columns]
            updated[:, columns] = etd2_step(
                f_group, rhs, dt,
                balance_weights=np.broadcast_to(
                    operator[3][:, None], f_group.shape,
                ),
                balance_axis=0, max_loss_step=0.25,
            )
            # Do not let the loop locals retain an evicted dense operator while
            # the next gap is built; the LRU is the sole cross-group owner.
            del rhs, operator
        return updated
