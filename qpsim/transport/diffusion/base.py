"""Energy-dependent spatial-diffusion operator family.

The dirty-limit Keldysh--Usadel reduction, the rejected diagnostic
variant, and the two scalar-Boltzmann closures form a one-parameter
family in the dressing exponents ``(p, q)``::

    L_{p,q}[f] = N_1^{-p} d/dx ( D_0 N_1^q  d f/dx )

equivalently, multiplying through by ``N_1^p``,

    d/dt ( N_1^p f ) = d/dx ( D_0 N_1^q  d f/dx ).

So each member fixes three linked quantities:

* conserved (energy-resolved) density  ``rho_cons = N_1^p f``,
* spatial flux coefficient            ``W = D_0 N_1^q``,
* uniform-gap effective diffusivity    ``D_eff = D_0 N_1^{q-p}``.

Here ``N_1(E) = Re[ E / sqrt(E^2 - Delta^2) ]`` is the BCS density of
states (the ``rho`` carried by
:class:`qpsim.physics.spectral.SpectralContext`), which rises toward the
gap edge and is zero below it.

Members
-------
======  =======  ==============  ======================  =========================
model   (p, q)   conserved       uniform-gap D_eff       role
======  =======  ==============  ======================  =========================
A1      (1, 2)   N_1 f           D_0 N_1   (rises)       dirty Usadel -- default
A2      (2, 2)   N_1^2 f         D_0       (flat)        diagnostic (rejected)
C       (0, -1)  f               D_0 / N_1 (falls)       clean / BRT / legacy D_E
B       (0, -2)  f               D_0 / N_1^2             constant-tau Boltzmann
======  =======  ==============  ======================  =========================

``A1`` is the physically correct operator: the dirty-limit Usadel kinetic
equation conserves the energy-resolved quasiparticle density ``N_1 f`` and
dresses the flux by ``N_1^2``, giving a rate that *rises* toward the gap
edge. ``A2`` (conserves ``N_1^2 f``, flat rate) is a diagnostic foil -- it
is the operator the legacy ``transport.diffusion`` docstrings mislabeled
"usadel". ``C`` reproduces the legacy
``SpectralContext.D_E = D_0 sqrt(1 - (Delta/E)^2) = D_0 / N_1`` closure and
is kept as a clean-limit / backward-compatible switch; ``B`` is the
constant-collision-time scalar-Boltzmann closure.

Legacy aliases (old enum names) map as ``LEGACY -> C``, ``BOLTZMANN -> B``,
``USADEL -> A2`` -- see :func:`from_name`. Note that the old "USADEL" is
the rejected diagnostic A2, *not* the dirty-limit Usadel operator A1.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = [
    "DEFAULT_DIFFUSION_MODEL",
    "DiffusionModel",
    "density_weight",
    "dressing_exponents",
    "effective_rate",
    "flux_weight",
    "from_name",
]


class DiffusionModel(Enum):
    """Spatial-diffusion operator, identified by its ``(p, q)`` dressing.

    The enum value is the ``(p, q)`` tuple; read it via :attr:`p` /
    :attr:`q` or :func:`dressing_exponents`.
    """

    A1 = (1, 2)
    A2 = (2, 2)
    C = (0, -1)
    B = (0, -2)

    @property
    def p(self) -> int:
        """Density-dressing exponent: conserved density is ``N_1**p * f``."""
        return int(self.value[0])

    @property
    def q(self) -> int:
        """Flux-dressing exponent: flux coefficient is ``D_0 * N_1**q``."""
        return int(self.value[1])


#: Physically correct default (dirty-limit Keldysh--Usadel reduction).
DEFAULT_DIFFUSION_MODEL = DiffusionModel.A1

#: Old enum names, kept so callers/configs written against the previous
#: ``LEGACY / BOLTZMANN / USADEL`` vocabulary keep resolving. The old
#: "USADEL" is the *rejected* diagnostic A2, not the dirty-limit Usadel
#: operator (which is A1).
_LEGACY_ALIASES: dict[str, DiffusionModel] = {
    "LEGACY": DiffusionModel.C,
    "BOLTZMANN": DiffusionModel.B,
    "USADEL": DiffusionModel.A2,
}


def from_name(name: str) -> DiffusionModel:
    """Resolve a :class:`DiffusionModel` from a member name or legacy alias.

    Accepts the member names ``"A1" / "A2" / "C" / "B"`` (case-insensitive)
    and the legacy aliases ``"LEGACY" -> C``, ``"BOLTZMANN" -> B``,
    ``"USADEL" -> A2``.
    """
    key = name.strip().upper()
    if key in DiffusionModel.__members__:
        return DiffusionModel[key]
    if key in _LEGACY_ALIASES:
        return _LEGACY_ALIASES[key]
    valid = sorted(DiffusionModel.__members__) + sorted(_LEGACY_ALIASES)
    raise ValueError(f"Unknown diffusion model {name!r}; expected one of {valid}.")


def dressing_exponents(model: DiffusionModel) -> tuple[int, int]:
    """Return the ``(p, q)`` dressing exponents for ``model``."""
    return model.p, model.q


def density_weight(N1: np.ndarray, p: int) -> np.ndarray:
    """Conserved-density weight ``N_1**p`` (the ``rho_cons = N_1^p f`` factor).

    ``p = 0`` returns ones (bare ``f`` is conserved). All shipped models
    use ``p >= 0``, so this is finite everywhere ``N_1`` is (and zero below
    the gap for ``p > 0``, where there are no states).
    """
    N1_arr = np.asarray(N1, dtype=float)
    if p == 0:
        return np.ones_like(N1_arr)
    return np.power(N1_arr, p)


def flux_weight(D0: float, N1: np.ndarray, q: int) -> np.ndarray:
    """Spatial flux coefficient ``W = D_0 N_1**q``.

    For ``q < 0`` the coefficient is set to zero wherever ``N_1 == 0``
    (sub-gap: no states, hence no flux) instead of diverging.
    """
    return _powered_coefficient(D0, N1, q)


def effective_rate(D0: float, N1: np.ndarray, model: DiffusionModel) -> np.ndarray:
    """Uniform-gap effective diffusivity ``D_eff = D_0 N_1**(q - p)``.

    This is the per-energy scalar rate the conserved density ``N_1^p f``
    diffuses with when the gap (hence ``N_1``) is spatially uniform, so the
    operator collapses to ``d_t u = D_eff * d_xx u``. It rises toward the
    gap edge for ``A1`` (exponent ``+1``), is flat for ``A2`` (exponent
    ``0``), and falls for ``C`` / ``B`` (negative exponents).
    """
    p, q = dressing_exponents(model)
    return _powered_coefficient(D0, N1, q - p)


def _powered_coefficient(D0: float, N1: np.ndarray, exponent: int) -> np.ndarray:
    """``D_0 * N_1**exponent`` with sub-gap (``N_1 == 0``) guarded to 0.

    Non-negative exponents need no guard (``0**n`` is ``1`` or ``0``);
    negative exponents would diverge where ``N_1 == 0``, so those entries
    are zeroed -- no states means no transport.
    """
    N1_arr = np.asarray(N1, dtype=float)
    if exponent == 0:
        return np.full_like(N1_arr, float(D0))
    if exponent > 0:
        return float(D0) * np.power(N1_arr, exponent)
    coeff = np.zeros_like(N1_arr)
    positive = N1_arr > 0.0
    coeff[positive] = float(D0) * np.power(N1_arr[positive], exponent)
    return coeff
