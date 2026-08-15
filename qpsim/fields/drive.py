"""External drives: a prescribed source over energy, space and time.

The general object is a rate field ``g(E, x, y, t)``. Everything users
normally want — a constant injection, a pulse, a focused laser spot, a
monoenergetic line at the gap edge — is a special case of it, so this module
represents the general case and treats the presets as ways of writing one
down. That ordering matters: the previous shape held a *fixed* ``(NE,
Ncells)`` array for the whole run, which makes the canonical experiment —
pulse the sample, switch the drive off, fit the decay — inexpressible, and
no amount of preset-adding fixes that.

Two rates, because the kinetic equation has two places to put one. ``gain``
is a source, added to ``df/dt`` directly. ``loss`` is a rate *coefficient*
multiplying ``f``, so it drains in proportion to what is there — a trap, an
out-tunnelling channel, an absorbing contact. A drive may set either or both.

Separable by construction where it can be. ``A · g_E(E) · g_S(x,y) · g_T(t)``
covers nearly every real drive and costs one precomputed ``(NE, Ncells)``
array times a scalar per step, so the common case stays free. The
non-separable form is there for the cases it cannot express, and pays for
itself only when used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from qpsim.geometries import Geometry

Rates = tuple[np.ndarray | None, np.ndarray | None]


class DriveEvaluationError(ValueError):
    """A prescribed drive that could not be evaluated at some time."""


@runtime_checkable
class ExternalDrive(Protocol):
    """A source the solver samples as it steps."""

    def sample(self, t: float) -> Rates:
        """``(gain, loss)`` at simulation time ``t``, each ``(NE, Ncells)``.

        Either may be ``None``, meaning that channel is absent — which is not
        the same as an array of zeros. ``None`` lets the backend skip the term
        entirely, and a term that is never built is the one form of "off" that
        cannot be accidentally switched back on downstream.
        """

    @property
    def is_static(self) -> bool:
        """True when ``sample`` does not depend on ``t``.

        Lets a caller hoist the sample out of the step loop, which is the
        difference between the common case costing nothing and costing a
        field evaluation per step.
        """


@dataclass(frozen=True)
class StaticDrive:
    """Fixed rates for the whole run.

    The shape every drive had before time dependence existed, kept as a
    first-class case: a steady-state search wants exactly this.
    """

    gain: np.ndarray | None = None
    loss: np.ndarray | None = None

    def sample(self, t: float) -> Rates:
        return self.gain, self.loss

    @property
    def is_static(self) -> bool:
        return True


@dataclass(frozen=True)
class SeparableDrive:
    """``A · g_E(E) · g_S(x,y) · g_T(t)``, with the spatial-energy part cached.

    ``pattern`` is the precomputed ``(NE, Ncells)`` outer product; the time
    factor is a scalar applied per step. ``channel`` selects whether the
    result is a source or a loss coefficient.
    """

    pattern: np.ndarray
    time_factor: Callable[[float], float]
    channel: str = "gain"
    static: bool = False

    def __post_init__(self) -> None:
        if self.channel not in ("gain", "loss"):
            raise ValueError(
                f"channel must be 'gain' or 'loss', not {self.channel!r}."
            )

    def sample(self, t: float) -> Rates:
        scale = float(self.time_factor(t))
        # Exactly zero is worth its own branch: an off pulse should cost
        # nothing and should hand the backend None rather than a zero field.
        field = None if scale == 0.0 else self.pattern * scale
        return (field, None) if self.channel == "gain" else (None, field)

    @property
    def is_static(self) -> bool:
        return self.static


@dataclass(frozen=True)
class ExpressionDrive:
    """A non-separable ``g(E, x, y, t)`` evaluated onto the solver's grid.

    The escape hatch for a drive whose shape in one variable depends on
    another — a spot that moves, a spectrum that hardens as it decays.
    Evaluated on the full ``(NE, Ncells)`` mesh each step, so it costs
    materially more than the separable form; that is the price of generality
    and only the runs that need it pay.
    """

    fn: Callable[..., np.ndarray]
    energies: np.ndarray          # (NE,)
    gap: float                    # reference gap (ueV)
    x_um: np.ndarray              # (Ncells,)
    y_um: np.ndarray              # (Ncells,)
    x_norm: np.ndarray            # (Ncells,) in [0, 1]
    y_norm: np.ndarray            # (Ncells,) in [0, 1]
    params: dict[str, float] | None = None
    channel: str = "gain"

    def sample(self, t: float) -> Rates:
        e = self.energies[:, None]
        ones = np.ones((self.energies.size, 1))
        shape = (self.energies.size, self.x_um.size)
        try:
            value = self.fn(
                E=np.broadcast_to(e, shape),
                gap=float(self.gap),
                x=ones * self.x_norm[None, :],
                y=ones * self.y_norm[None, :],
                x_um=ones * self.x_um[None, :],
                y_um=ones * self.y_um[None, :],
                t=float(t),
                params=self.params,
            )
            field = np.broadcast_to(
                np.asarray(value, dtype=float), shape
            ).astype(float, copy=True)
        except Exception as exc:
            # The failure is a hundred steps into a run and the traceback
            # points at `eval`, which says nothing. Name the time, so the
            # author knows which part of their own expression to look at.
            raise DriveEvaluationError(
                f"The prescribed drive failed at t={t:g} ns: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not np.all(np.isfinite(field)):
            raise DriveEvaluationError(
                f"The prescribed drive returned a non-finite value at "
                f"t={t:g} ns. Check the expression at the edges of its range."
            )
        return (field, None) if self.channel == "gain" else (None, field)

    @property
    def is_static(self) -> bool:
        return False


@dataclass(frozen=True)
class SumDrive:
    """Several drives acting at once.

    A device can be under a steady bias and a pulse at the same time, and
    forcing those into one expression would be a worse way of saying it.
    """

    parts: tuple[ExternalDrive, ...]

    def sample(self, t: float) -> Rates:
        gain: np.ndarray | None = None
        loss: np.ndarray | None = None
        for part in self.parts:
            part_gain, part_loss = part.sample(t)
            if part_gain is not None:
                gain = part_gain if gain is None else gain + part_gain
            if part_loss is not None:
                loss = part_loss if loss is None else loss + part_loss
        return gain, loss

    @property
    def is_static(self) -> bool:
        return all(p.is_static for p in self.parts)


# -- coordinates -------------------------------------------------------


def cell_coordinates(
    geometry: Geometry,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x_um, y_um, x_norm, y_norm)`` for every cell, in mask order.

    Both physical and normalised, because both get asked for and converting
    between them by hand is where sign and offset mistakes live. Normalised
    coordinates use cell *centres* over the mask's bounding box, so a
    single-cell device sits at 0.5 rather than at an end — the archived
    implementation used two different conventions for this in the same file
    and a preset centred at "0.5" landed half a cell off.
    """
    rows, cols = np.nonzero(np.asarray(geometry.mask, dtype=bool))
    mesh = float(geometry.mesh_size)
    x_um = (cols + 0.5) * mesh
    y_um = (rows + 0.5) * mesh
    if rows.size == 0:
        return (x_um.astype(float), y_um.astype(float),
                x_um.astype(float), y_um.astype(float))
    # The DEVICE's bounding box, not the raster's extent. Normalising by
    # mask.shape is the same thing only when the mask fills its array, which a
    # rectangle does and an imported layout does not: a GDS raster is padded by
    # a mesh cell and floored at 8x8, so a device occupying the middle of its
    # array had "x = 0.5" land off-centre, and the default centred Gaussian hot
    # spot peaked on an edge column -- or off the device entirely, leaving only
    # a tail, which spatial_profile then peak-normalises to 1.0 so the
    # amplitude stops meaning the requested excess. Nothing raises: the
    # peak == 0 check cannot catch a Gaussian, which is never exactly zero.
    row_lo, row_hi = int(rows.min()), int(rows.max())
    col_lo, col_hi = int(cols.min()), int(cols.max())
    n_rows = row_hi - row_lo + 1
    n_cols = col_hi - col_lo + 1
    return (
        x_um.astype(float),
        y_um.astype(float),
        ((cols - col_lo + 0.5) / n_cols).astype(float),
        ((rows - row_lo + 0.5) / n_rows).astype(float),
    )
