"""Quasiparticle number density and qpsim-convention ``x_qp``.

Two observables:

* :func:`qp_number_density` — returns ``n_qp = 4 ρ_F ∫_Δ^∞ ρ(E) f(E) dE``,
  the QP number density per volume (the factor-of-4 absorbs spin × 2 and
  particle/hole × 2, matching Fischer 2023 Eq. 4 normalization).
* :func:`qp_fraction` — returns the dimensionless ``x_qp = n_qp / (4 ρ_F Δ_0)``,
  which is **half** the Fischer/Catelani paper convention ``n_qp / (2 ρ_F Δ_0)``
  (see that function's docstring). The ``ρ_F`` factor cancels, so it isn't an
  argument.

Both take a :class:`SpectralContext` for ``ρ(E)`` (BCS or Dynes). Pure-BCS
contexts use analytic cell weights for the integrable DOS singularity; Dynes
contexts retain the ordinary cell-centered ``dE`` quadrature. The grid
energies are in µeV, while the material database stores the conventional
``ρ_F`` in eV⁻¹ m⁻³; :func:`qp_number_density` performs that conversion
explicitly.
"""

from __future__ import annotations

import numpy as np

from qpsim.materials.database import validate_rho_F_eV
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext

_UEV_PER_EV = 1.0e6


def _qp_integral_uev(f: np.ndarray, ctx: SpectralContext) -> float:
    """Return ``∫ ρ(E) f(E) dE`` with the appropriate spectral measure."""
    f_arr = np.asarray(f, dtype=float)
    if f_arr.shape != ctx.E.shape:
        raise ValueError(
            f"f must have shape {ctx.E.shape}; got {f_arr.shape}."
        )
    if ctx.dynes_gamma > 0.0:
        # The analytic weights below are specific to the pure-BCS DOS. Dynes
        # broadening removes the ideal square-root singularity, so retain the
        # context's ordinary cell-centered quadrature there.
        return float(np.sum(ctx.rho * f_arr * ctx.dE))
    weights = bcs_dos_cell_weights(ctx.E, ctx.dE, ctx.gap)
    return float(np.sum(f_arr * weights))


def qp_number_density(
    f: np.ndarray,
    ctx: SpectralContext,
    rho_F: float,
) -> float:
    """QP number density ``n_qp = 4 ρ_F ∫ ρ(E) f(E) dE``.

    The integral is evaluated on the µeV grid and divided by ``1e6`` so
    its energy measure is in eV before multiplication by ``ρ_F``.

    Parameters
    ----------
    f
        Occupation on ``ctx.E``.
    ctx
        SpectralContext with ``ρ(E)`` and ``dE``.
    rho_F
        Conventional single-spin DOS at the Fermi level in **eV⁻¹ m⁻³**
        (e.g. Al ``1.74e28``). This is not J⁻¹ m⁻³, which would be
        approximately ``1e47`` for Al.
    """
    rho_F = validate_rho_F_eV(rho_F, allow_zero=False)
    integral_uev = _qp_integral_uev(f, ctx)
    return 4.0 * rho_F * integral_uev / _UEV_PER_EV


def qp_fraction(f: np.ndarray, ctx: SpectralContext, delta_0: float) -> float:
    """qpsim-convention dimensionless QP fraction ``x_qp``.

    ``x_qp = n_qp / (4 ρ_F Δ_0[eV])``. Equivalently, on qpsim's
    µeV grid, ``x_qp = (1 / Δ_0[µeV]) ∫_Δ^∞ ρ(E) f(E) dE``.

    NOTE: this is **half** the Fischer/Catelani paper convention
    ``x_qp^paper = n_qp / (2 ρ_F Δ_0)`` — multiply by 2 to compare against the
    paper (the validation figures do this at the plotting / analytic-overlay
    layer). The denominator is consistent with :func:`qp_number_density`'s
    factor-of-4 ``n_qp``.

    ``ρ_F`` cancels in the ratio, so it isn't an argument.
    """
    if delta_0 <= 0:
        raise ValueError("delta_0 must be positive.")
    return _qp_integral_uev(f, ctx) / delta_0
