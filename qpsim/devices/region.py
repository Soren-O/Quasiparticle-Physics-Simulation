"""Region — one superconducting region inside a Device.

For v1 (Gate 2 T3 only), a Region is a thin wrapper around a
:class:`T3DiffusionState`. Future tiers (T2 with momentum direction,
T1 two-component) will plug in by replacing the ``state`` field type
with a tier union.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qpsim.backends.t3_diffusion import T3DiffusionState


@dataclass
class Region:
    """One named superconducting region with its own backend state.

    Parameters
    ----------
    name
        Unique identifier within the parent :class:`Device`. Junctions
        reference regions by this name.
    state
        T3DiffusionState carrying material, energy grid, phonon state,
        and the current ``f(E)``. Tier-specific; v1 supports T3 only.
    """

    name: str
    state: T3DiffusionState
