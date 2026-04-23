"""Tier-agnostic types for backends.

Provides the :class:`Tier` enum shared across the electronic-tier
hierarchy. The ``Backend`` protocol and ``BackendState`` union will
appear once T2 / T1 backends arrive — for Gate 2 only ``T3`` exists,
so the protocol would be premature.
"""

from __future__ import annotations

from enum import Enum


class Tier(Enum):
    """Electronic-tier tag per NFP §2.2.

    * ``T3_DIFFUSION`` — isotropic dirty-limit diffusion (v1 target).
    * ``T2_KINETIC`` — scalar occupation with angular grid (v2).
    * ``T1_TWO_COMPONENT`` — full ``(f_L, f_T)`` two-component (v3).
    """

    T3_DIFFUSION = "t3_diffusion"
    T2_KINETIC = "t2_kinetic"
    T1_TWO_COMPONENT = "t1_two_component"
