"""Quasiparticle number density and Fischer-convention ``x_qp``.

Two observables:

* :func:`qp_number_density` — returns ``n_qp = 4 ρ_F ∫_Δ^∞ ρ(E) f(E) dE``,
  the QP number density per volume (the factor-of-4 absorbs spin × 2 and
  particle/hole × 2, matching Fischer 2023 Eq. 4 normalization).
* :func:`qp_fraction` — returns the dimensionless ``x_qp = n_qp / (4 ρ_F Δ_0)``.
  The ``ρ_F`` factor cancels, so this function doesn't need it.

Both take a :class:`SpectralContext` for ``ρ(E)`` (BCS or Dynes) and
the cell-centered integration weights ``dE``.
"""

from __future__ import annotations

import numpy as np

from qpsim.physics.spectral import SpectralContext


def qp_number_density(
    f: np.ndarray,
    ctx: SpectralContext,
    rho_F: float,
) -> float:
    """QP number density ``n_qp = 4 ρ_F ∫ ρ(E) f(E) dE``.

    Parameters
    ----------
    f
        Occupation on ``ctx.E``.
    ctx
        SpectralContext with ``ρ(E)`` and ``dE``.
    rho_F
        Single-spin DOS at the Fermi level (J⁻¹ m⁻³ or user units).
    """
    if rho_F <= 0:
        raise ValueError("rho_F must be positive.")
    return 4.0 * rho_F * float(np.sum(ctx.rho * f * ctx.dE))


def qp_fraction(f: np.ndarray, ctx: SpectralContext, delta_0: float) -> float:
    """Fischer-convention dimensionless QP fraction ``x_qp``.

    ``x_qp = n_qp / (4 ρ_F Δ_0) = (1 / Δ_0) ∫_Δ^∞ ρ(E) f(E) dE``.

    ``ρ_F`` cancels in the ratio, so it isn't an argument.
    """
    if delta_0 <= 0:
        raise ValueError("delta_0 must be positive.")
    return float(np.sum(ctx.rho * f * ctx.dE)) / delta_0
