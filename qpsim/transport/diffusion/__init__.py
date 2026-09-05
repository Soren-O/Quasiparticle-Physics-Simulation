"""Diffusion-model dispatch.

Re-exports the :class:`~qpsim.transport.diffusion.base.DiffusionModel`
operator family (A1 / A2 / B / C) and its dressing helpers. See
:mod:`qpsim.transport.diffusion.base` for the ``(p, q)`` parametrization
and the A1 = dirty-Usadel / A2 = rejected-diagnostic distinction.

Naming trap: the form ``(D / N_1^2) d_x[ N_1^2 d_x f ]``, which the name
"Usadel" invites, is the *rejected* diagnostic A2 (it conserves
``N_1^2 f``), not the dirty-limit Usadel operator -- which is A1
(conserves ``N_1 f``).
"""

from __future__ import annotations

from qpsim.transport.diffusion.base import (
    DEFAULT_DIFFUSION_MODEL,
    DiffusionModel,
    density_weight,
    dressing_exponents,
    effective_rate,
    flux_weight,
    from_name,
)

__all__ = [
    "DEFAULT_DIFFUSION_MODEL",
    "DiffusionModel",
    "density_weight",
    "dressing_exponents",
    "effective_rate",
    "flux_weight",
    "from_name",
]
