"""Region — one superconducting region inside a Device.

A Region is a thin wrapper around a :class:`DiffusionState`, naming it
so that Junctions can reference it within a Device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qpsim.backends.diffusion import DiffusionState


@dataclass
class Region:
    """One named superconducting region with its own backend state.

    Parameters
    ----------
    name
        Unique identifier within the parent :class:`Device`. Junctions
        reference regions by this name.
    state
        DiffusionState carrying material, energy grid, phonon state,
        and the current ``f(E)``.
    """

    name: str
    state: DiffusionState
