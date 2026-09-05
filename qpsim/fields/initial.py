"""Initial conditions: where a run starts, when that is not equilibrium.

A run that starts at ``f_FD(E, T_bath)`` can reach only the dynamics a
drive produces. That rules out the whole family of *relaxation*
measurements — prepare quasiparticles above equilibrium and watch
them recombine, seed a hot spot and watch it spread, put an excess in the top
energy bin and watch it scatter down — which is where recombination times,
diffusion constants and phonon trapping are actually measured from.

Units, stated once because a port from the archived implementation gets this
wrong by default. That code carried a **density** ``n(E, x)``: the spatial
field held the energy-integrated amplitude and the energy profile was
normalised to unit integral. This state is an **occupation** ``f in [0, 1]``.
So ``amplitude`` here is a *peak excess occupation*, not a particle count, and
the separable form is

    f(E, x, y) = f_FD(E, T_bath) + A · g_E(E) · g_S(x, y)

with ``g_E`` and ``g_S`` each peak-normalised to 1. Reading a density-shaped
number into it silently produces a different experiment.

Occupations that leave ``[0, 1]`` are clipped, and clipping is reported rather
than absorbed: an initial condition quietly flattened against the ceiling is
not the one that was asked for, and every observable downstream would be
measured from a state nobody chose.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qpsim.observables.gap_suppression import fermi_dirac_distribution
from qpsim.physics.spectral import SpectralContext


class InitialConditionError(ValueError):
    """An initial condition that cannot be built as stated."""


@dataclass(frozen=True)
class SeededState:
    """An initial occupation and what had to be done to make it legal."""

    f: np.ndarray                       # (NE, Ncells)
    notes: tuple[str, ...] = ()

    @property
    def clipped(self) -> bool:
        return any("clipped" in n for n in self.notes)


# -- energy profiles ---------------------------------------------------


def energy_profile(
    kind: str,
    spectral: SpectralContext,
    *,
    T_eff: float | None = None,
    E_0: float | None = None,
    width: float | None = None,
    expression: Callable[..., np.ndarray] | None = None,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """A peak-normalised shape in energy, ``(NE,)``.

    Peak-normalised rather than unit-integral so that ``amplitude`` means the
    same thing — a peak excess occupation — whichever shape is chosen. With
    unit-integral normalisation, narrowing a line would silently raise its
    height, and "the same amplitude" would mean a different experiment on
    every grid.
    """
    energies = np.asarray(spectral.E, dtype=float)
    supported = np.asarray(spectral.cell_density, dtype=float) > 0.0

    if kind == "flat":
        shape = supported.astype(float)
    elif kind == "thermal":
        if T_eff is None or T_eff <= 0.0:
            raise InitialConditionError(
                "A thermal energy profile needs a positive T_eff (K)."
            )
        shape = fermi_dirac_distribution(energies, float(T_eff))
    elif kind == "monoenergetic":
        if E_0 is None:
            raise InitialConditionError(
                "A monoenergetic profile needs E_0 (ueV)."
            )
        # A finite width, not a single bin: a delta on one bin makes the
        # result depend on where the grid happens to put its edges, so
        # refining the grid would change the experiment rather than resolve it.
        sigma = float(width) if width else 2.0 * float(np.min(spectral.dE))
        if sigma <= 0.0:
            raise InitialConditionError("width must be positive.")
        shape = np.exp(-0.5 * ((energies - float(E_0)) / sigma) ** 2)
    elif kind == "gap_edge":
        sigma = float(width) if width else 4.0 * float(np.min(spectral.dE))
        shape = np.exp(-0.5 * ((energies - float(spectral.gap)) / sigma) ** 2)
    elif kind == "expression":
        if expression is None:
            raise InitialConditionError(
                "An expression energy profile needs a compiled expression."
            )
        shape = np.broadcast_to(
            np.asarray(
                expression(E=energies, gap=float(spectral.gap), params=params),
                dtype=float,
            ),
            energies.shape,
        ).astype(float, copy=True)
    else:
        raise InitialConditionError(f"Unknown energy profile {kind!r}.")

    # Below the gap there is no spectral weight, so occupation there is not a
    # state the solver can carry -- it would be silently discarded by every
    # observable while still looking like a prepared population.
    shape = np.where(supported, shape, 0.0)
    if not np.all(np.isfinite(shape)):
        raise InitialConditionError(
            f"The {kind!r} energy profile produced a non-finite value."
        )
    peak = float(np.max(np.abs(shape)))
    if peak == 0.0:
        raise InitialConditionError(
            f"The {kind!r} energy profile is zero in every supported bin, so "
            "it would prepare nothing. Check E_0 and width against the grid."
        )
    return shape / peak


# -- spatial profiles --------------------------------------------------


def spatial_profile(
    kind: str,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    *,
    x_0: float = 0.5,
    y_0: float = 0.5,
    sigma: float = 0.12,
    expression: Callable[..., np.ndarray] | None = None,
    x_um: np.ndarray | None = None,
    y_um: np.ndarray | None = None,
    params: dict[str, float] | None = None,
) -> np.ndarray:
    """A peak-normalised shape over the cells, ``(Ncells,)``."""
    x_norm = np.asarray(x_norm, dtype=float)
    y_norm = np.asarray(y_norm, dtype=float)

    if kind == "uniform":
        shape = np.ones_like(x_norm)
    elif kind == "gaussian":
        if sigma <= 0.0:
            raise InitialConditionError("sigma must be positive.")
        shape = np.exp(
            -((x_norm - x_0) ** 2 + (y_norm - y_0) ** 2) / (2.0 * sigma**2)
        )
    elif kind == "point":
        # Nearest cell centre by normalised distance. A point source on a mask
        # may name a coordinate that is outside the material; snapping to the
        # nearest cell that exists is the only defined reading of it.
        index = int(np.argmin((x_norm - x_0) ** 2 + (y_norm - y_0) ** 2))
        shape = np.zeros_like(x_norm)
        shape[index] = 1.0
    elif kind == "expression":
        if expression is None:
            raise InitialConditionError(
                "An expression spatial profile needs a compiled expression."
            )
        shape = np.broadcast_to(
            np.asarray(
                expression(
                    x=x_norm, y=y_norm,
                    x_um=x_norm if x_um is None else np.asarray(x_um, float),
                    y_um=y_norm if y_um is None else np.asarray(y_um, float),
                    params=params,
                ),
                dtype=float,
            ),
            x_norm.shape,
        ).astype(float, copy=True)
    else:
        raise InitialConditionError(f"Unknown spatial profile {kind!r}.")

    if not np.all(np.isfinite(shape)):
        raise InitialConditionError(
            f"The {kind!r} spatial profile produced a non-finite value."
        )
    peak = float(np.max(np.abs(shape)))
    if peak == 0.0:
        raise InitialConditionError(
            f"The {kind!r} spatial profile is zero in every cell, so it would "
            "prepare nothing."
        )
    return shape / peak


# -- assembly ----------------------------------------------------------


def seed_occupation(
    spectral: SpectralContext,
    n_cells: int,
    T_bath: float,
    *,
    excess: np.ndarray | None = None,
    absolute: np.ndarray | None = None,
) -> SeededState:
    """Build ``f(E, cell)``, clipped to ``[0, 1]`` and reporting if it clipped.

    ``excess`` is added on top of the thermal distribution; ``absolute``
    replaces it outright. Exactly one may be given, because "add to thermal"
    and "this IS the occupation" differ by the thermal population itself, and
    a caller who means one and gets the other has run a different experiment.
    """
    if excess is not None and absolute is not None:
        raise InitialConditionError(
            "Give either an excess over thermal or an absolute occupation, "
            "not both."
        )
    thermal = fermi_dirac_distribution(np.asarray(spectral.E, float), float(T_bath))
    base = np.repeat(thermal[:, None], n_cells, axis=1)
    given = excess if excess is not None else absolute
    if given is None:
        return SeededState(f=base)

    # Checked before the arithmetic: numpy's own broadcast error names shapes
    # but not which argument was wrong or what it should have been, and a
    # mismatch that happens to broadcast would be worse than one that raises.
    given = np.asarray(given, dtype=float)
    if given.shape != base.shape:
        raise InitialConditionError(
            f"Initial occupation has shape {given.shape}, expected "
            f"{base.shape} = (energy bins, cells)."
        )
    field = given if absolute is not None else base + given

    if not np.all(np.isfinite(field)):
        raise InitialConditionError(
            "The initial occupation contains a non-finite value."
        )

    supported = np.asarray(spectral.cell_density, dtype=float) > 0.0
    field = np.where(supported[:, None], field, 0.0)

    notes: list[str] = []
    over = float(np.max(field)) if field.size else 0.0
    under = float(np.min(field)) if field.size else 0.0
    if over > 1.0 or under < 0.0:
        n_over = int(np.count_nonzero(field > 1.0))
        n_under = int(np.count_nonzero(field < 0.0))
        notes.append(
            f"The initial occupation was clipped to [0, 1]: "
            f"{n_over} value(s) above 1 (max {over:.4g}) and {n_under} below 0 "
            f"(min {under:.4g}). The run starts from the clipped state, which "
            "is not the one requested -- lower the amplitude to prepare the "
            "distribution as stated."
        )
        field = np.clip(field, 0.0, 1.0)

    return SeededState(f=field, notes=tuple(notes))


def separable_excess(
    energy_shape: np.ndarray, spatial_shape: np.ndarray, amplitude: float
) -> np.ndarray:
    """``A · g_E(E) · g_S(x, y)`` as an ``(NE, Ncells)`` excess occupation."""
    if amplitude < 0.0:
        raise InitialConditionError(
            "amplitude must be non-negative: a negative excess would prepare "
            "an occupation below the bath, which is a different experiment "
            "and is better stated as an absolute profile."
        )
    return float(amplitude) * np.outer(
        np.asarray(energy_shape, float), np.asarray(spatial_shape, float)
    )
