"""Mattis–Bardeen ac conductivity from a quasiparticle distribution.

Returns normalized conductivities ``σ₁/σ_N`` and ``σ₂/σ_N`` — the
normal-state ``σ_N`` cancels out of the ``Q_i`` and ``δω/ω``
observables that build on these integrals.

Ported from ``qpsim/numerics/observables.py`` at Gate 2 (no logic
changes). Pure-BCS spectral functions only; Dynes-broadened contexts
are rejected (the Mattis–Bardeen integrands assume ``ρ(E) = 0`` below
the gap, which Dynes violates).
"""

from __future__ import annotations

import numpy as np

from qpsim.physics.spectral import SpectralContext, bcs_density_of_states


def compute_ac_conductivity(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    *,
    n_subgap: int = 500,
) -> tuple[float, float]:
    r"""Normalized ac conductivity ``(σ₁/σ_N, σ₂/σ_N)`` at probe frequency ``ω₀``.

    .. math::

        \sigma_1/\sigma_N = \frac{2}{\omega_0}\int_\Delta^\infty dE\;
            [f(E) - f(E+\omega_0)]\,\rho(E)\,U^+(E, E+\omega_0),

        \sigma_2/\sigma_N = \frac{1}{\omega_0}
            \int_{\Delta-\omega_0}^{\Delta} dE\;
            [1 - 2f(E+\omega_0)]\,U^+(E, E+\omega_0)\,
            \frac{E}{\sqrt{\Delta^2 - E^2}},

    with ``U⁺(E, E') = ρ(E') · K⁺(E, E')``. The sub-gap ``σ₂``
    integral uses an ``n_subgap``-point midpoint rule that avoids the
    integrable singularity at ``E = Δ``.

    Raises
    ------
    ValueError
        If ``omega_0 ≤ 0``, ``omega_0 ≥ ctx.gap``, or
        ``ctx.dynes_gamma > 0``.
    """
    if omega_0 <= 0:
        raise ValueError("omega_0 must be positive.")
    if n_subgap <= 0:
        raise ValueError("n_subgap must be a positive integer.")
    if ctx.dynes_gamma > 0:
        raise ValueError(
            "Mattis-Bardeen observables assume pure BCS spectral functions. "
            "Dynes-broadened contexts (dynes_gamma > 0) are not supported."
        )
    if omega_0 >= ctx.gap:
        raise ValueError(
            f"omega_0={omega_0:g} must be below the gap ({ctx.gap:g} μeV): "
            "this implementation keeps only the sub-gap Mattis-Bardeen "
            "terms (σ₂'s lower limit is clamped at Δ−ω₀ ≥ 0 and σ₁ omits "
            "the pair-breaking contribution), which are wrong above it."
        )

    gap = ctx.gap
    E = ctx.E
    dE = ctx.dE
    rho = ctx.rho

    # σ₁: supergap integral over [Δ, ∞).
    E_partner = E + omega_0
    f_partner = np.interp(E_partner, E, f, right=0.0)
    rho_partner = bcs_density_of_states(E_partner, gap)
    K_plus_partner = 1.0 + gap ** 2 / np.maximum(E * E_partner, 1e-30)

    U_plus = rho_partner * K_plus_partner
    integrand_1 = (f - f_partner) * rho * U_plus
    sigma_1_norm = (2.0 / omega_0) * float(np.sum(integrand_1 * dE))

    # σ₂: sub-gap integral over [Δ − ω₀, Δ]. Below the QP grid; use a
    # dedicated midpoint-rule quadrature that excludes the singular endpoint.
    E_lo = max(gap - omega_0, 0.0)
    if E_lo >= gap:
        sigma_2_norm = 0.0
    else:
        dE_sub = (gap - E_lo) / n_subgap
        E_sub = E_lo + (np.arange(n_subgap) + 0.5) * dE_sub

        E_sub_partner = E_sub + omega_0
        f_sub_partner = np.interp(E_sub_partner, E, f, right=0.0)
        rho_sub_partner = bcs_density_of_states(E_sub_partner, gap)
        K_plus_sub = 1.0 + gap ** 2 / np.maximum(E_sub * E_sub_partner, 1e-30)
        U_plus_sub = rho_sub_partner * K_plus_sub

        # E / √(Δ² − E²): analytic continuation of the DOS below the gap.
        subgap_dos = E_sub / np.sqrt(np.maximum(gap ** 2 - E_sub ** 2, 1e-30))

        integrand_2 = (1.0 - 2.0 * f_sub_partner) * U_plus_sub * subgap_dos
        sigma_2_norm = (1.0 / omega_0) * float(np.sum(integrand_2 * dE_sub))

    return sigma_1_norm, sigma_2_norm
