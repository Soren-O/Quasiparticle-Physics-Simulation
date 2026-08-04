"""Quasiparticle number density and qpsim-convention ``x_qp``.

Two observables:

* :func:`qp_number_density` — returns ``n_qp = 4 ρ_F ∫ ρ(E) f(E) dE``,
  the QP number density per volume (the factor-of-4 absorbs spin × 2 and
  particle/hole × 2, matching Fischer 2023 Eq. 4 normalization). The
  integration domain is the full support of ``ρ``: ``[Δ, ∞)`` for pure BCS,
  ``[0, ∞)`` for a Dynes context, whose sub-gap weight is physical.
* :func:`qp_fraction` — returns the historical qpsim dimensionless fraction
  ``x_qp = n_qp / (4 ρ_F Δ_0)``,
  which is **half** the Fischer/Catelani paper convention ``n_qp / (2 ρ_F Δ_0)``
  (see that function's docstring).
* :func:`qp_fraction_paper` — returns the Fischer/Catelani convention
  explicitly, avoiding a silent factor-of-two conversion at call sites.

The ``ρ_F`` factor cancels from both fractions, so it is not an argument.

Both take a :class:`SpectralContext` for ``ρ(E)`` (BCS or Dynes). Pure-BCS
contexts use analytic cell weights for the integrable DOS singularity; Dynes
contexts retain the ordinary cell-centered ``dE`` quadrature. The grid
energies are in µeV, while the material database stores the conventional
``ρ_F`` in eV⁻¹ m⁻³; :func:`qp_number_density` performs that conversion
explicitly. The grid must cover the full support of the spectrum: a pure-BCS
grid must reach the gap edge with its first cell, and a Dynes grid must reach
``E = 0``. Starting above the relevant band edge is rejected because the
missing occupation there cannot be reconstructed honestly.
"""

from __future__ import annotations

import numpy as np

from qpsim.materials.database import validate_rho_F_eV
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.physics.spectral import SpectralContext

_UEV_PER_EV = 1.0e6


def _finite_real_scalar(name: str, raw: float) -> float:
    """Return one finite real scalar without silent complex coercion."""
    if isinstance(raw, (bool, np.bool_)) or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real scalar; got {raw!r}.")
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {raw!r}."
        ) from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return value


def _qp_integral_uev(f: np.ndarray, ctx: SpectralContext) -> float:
    """Return ``∫ ρ(E) f(E) dE`` with the appropriate spectral measure."""
    f_raw = np.asarray(f)
    if np.iscomplexobj(f_raw):
        raise ValueError("f must be real-valued.")
    try:
        f_arr = np.asarray(f_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("f must be a real numeric array.") from exc
    if f_arr.shape != ctx.E.shape:
        raise ValueError(
            f"f must have shape {ctx.E.shape}; got {f_arr.shape}."
        )
    if np.any(~np.isfinite(f_arr)) or np.any((f_arr < 0.0) | (f_arr > 1.0)):
        raise ValueError("f must contain finite occupations in [0, 1].")
    if ctx.dynes_gamma <= 0.0:
        # Observable density retains the stricter full-gap coverage contract:
        # a context may evolve only its represented domain, but it cannot call
        # that truncated quantity the physical total from Delta to infinity.
        # The helper raises when the first grid edge lies above the gap.
        bcs_dos_cell_weights(ctx.E, ctx.dE, ctx.gap)
    else:
        # Same contract with the band edge at zero: a Dynes DOS has finite
        # weight everywhere on [0, Delta), and at low T that sub-gap region
        # carries most of the density (measured 87% of n_qp missing at
        # gamma/Delta = 1e-3 and 99.8% at 5e-2, both at T = 0.2 K, on a grid
        # starting at Delta). The missing mass is set by k_B T, not by
        # gamma -- a grid starting at Delta/2 still loses 84% -- so no
        # gamma-scaled window is admissible and the requirement is the full
        # physical domain from zero.
        lower_edge = float(ctx.E[0]) - 0.5 * float(ctx.dE[0])
        coverage_tol = 128.0 * np.finfo(float).eps * max(1.0, abs(lower_edge))
        if lower_edge > coverage_tol:
            raise ValueError(
                "The energy grid does not cover the Dynes sub-gap band: "
                f"first cell edge {lower_edge:g} > 0. A Dynes spectrum has "
                "finite occupied weight on [0, Delta), which dominates n_qp "
                "at low T; start the grid at zero for a Dynes context."
            )
    return float(np.sum(f_arr * ctx.cell_weights))


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
    rho_F = _finite_real_scalar("rho_F", rho_F)
    rho_F = validate_rho_F_eV(rho_F, allow_zero=False)
    integral_uev = _qp_integral_uev(f, ctx)
    return 4.0 * rho_F * integral_uev / _UEV_PER_EV


def qp_fraction(f: np.ndarray, ctx: SpectralContext, delta_0: float) -> float:
    """qpsim-convention dimensionless QP fraction ``x_qp``.

    ``x_qp = n_qp / (4 ρ_F Δ_0[eV])``. Equivalently, on qpsim's
    µeV grid, ``x_qp = (1 / Δ_0[µeV]) ∫ ρ(E) f(E) dE`` over the full
    spectral support (``[Δ, ∞)`` for BCS, ``[0, ∞)`` for Dynes).

    NOTE: this is **half** the Fischer/Catelani paper convention
    ``x_qp^paper = n_qp / (2 ρ_F Δ_0)`` — multiply by 2 to compare against the
    paper (the validation figures do this at the plotting / analytic-overlay
    layer). The denominator is consistent with :func:`qp_number_density`'s
    factor-of-4 ``n_qp``.

    ``ρ_F`` cancels in the ratio, so it isn't an argument.
    """
    delta_0 = _finite_real_scalar("delta_0", delta_0)
    if delta_0 <= 0.0:
        raise ValueError("delta_0 must be finite and positive.")
    return _qp_integral_uev(f, ctx) / delta_0


def qp_fraction_paper(
    f: np.ndarray,
    ctx: SpectralContext,
    delta_0: float,
) -> float:
    """Fischer/Catelani quasiparticle fraction ``n_qp/(2 ρ_F Δ_0)``.

    With :func:`qp_number_density`'s single-spin-DOS convention this is
    ``2 / Δ_0 * ∫ ρ(E) f(E) dE``, exactly twice the historical
    :func:`qp_fraction` result. The explicit name keeps existing callers
    source-compatible while making paper comparisons unambiguous.
    """
    return 2.0 * qp_fraction(f, ctx, delta_0)
