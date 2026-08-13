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

from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
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
    ) -> None:
        if spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Spatial collisions require a pure-BCS SpectralContext "
                "(dynes_gamma == 0). A Dynes DOS needs consistently broadened "
                "normal/anomalous coherence functions, which this collision "
                "kernel does not implement."
            )
        self.spectral = spectral
        self.gap_per_cell = np.asarray(gap_per_cell, dtype=float)
        self.tau_0 = float(tau_0)
        self.T_c = float(T_c)
        self.T_bath = float(T_bath)
        self.enable_scattering = bool(enable_scattering)
        self.enable_recombination = bool(enable_recombination)

        self._distinct_gaps, self._group_index = np.unique(
            self.gap_per_cell, return_inverse=True,
        )
        self._cache: OrderedDict[Any, CollisionOperator] = OrderedDict()

        if self._distinct_gaps.size > _GAP_COUNT_WARN:
            warnings.warn(
                f"{self._distinct_gaps.size} distinct local gaps across "
                f"{self.gap_per_cell.size} cells: collisions are grouped by "
                "EXACT gap, so each one builds and holds its own dense "
                "kernels. This is efficient for a device of a few materials "
                "and quadratic-ish for a continuously varying gap. Quantise "
                "the gap profile, or coarsen the mesh, before running.",
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
            for columns, operator in groups:
                g, ell = self.group_rates(
                    current[:, columns],
                    operator,
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
            ) -> tuple[np.ndarray, np.ndarray]:
                return self.group_rates(
                    f_group, bound_operator, bound_gain, bound_loss,
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
