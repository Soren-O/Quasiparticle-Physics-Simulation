"""Mattis–Bardeen ac conductivity from a quasiparticle distribution.

Returns normalized conductivities ``σ₁/σ_N`` and ``σ₂/σ_N`` — the
normal-state ``σ_N`` cancels out of the ``Q_i`` and ``δω/ω``
observables that build on these integrals.

Ported from ``qpsim/numerics/observables.py`` at Gate 2. The super-gap
``σ₁`` integral uses the analytic pure-BCS DOS measure in each cell so the
integrable gap-edge singularity is not under-sampled. The sub-gap ``σ₂``
integral uses a sine-squared coordinate that removes both square-root endpoint
singularities before applying a midpoint rule. Pure-BCS spectral functions
only; Dynes-broadened contexts are rejected (the Mattis–Bardeen integrands
assume ``ρ(E) = 0`` below the gap, which Dynes violates).
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
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
    integral uses ``E = Δ - ω₀ cos²(θ)`` and an ``n_subgap``-point
    midpoint rule in ``θ``. The Jacobian cancels the integrable square-root
    singularities at both ``E = Δ-ω₀`` and ``E = Δ``.

    Raises
    ------
    ValueError
        If ``omega_0 ≤ 0``, ``omega_0 ≥ ctx.gap``, or
        ``ctx.dynes_gamma > 0``.
    """
    f_raw = np.asarray(f)
    if np.iscomplexobj(f_raw):
        raise ValueError("f must be real-valued.")
    f_arr = np.asarray(f_raw, dtype=float)
    if f_arr.shape != ctx.E.shape:
        raise ValueError(
            f"f must have the same shape as ctx.E; got {f_arr.shape} "
            f"and {ctx.E.shape}."
        )
    if np.any(~np.isfinite(f_arr)):
        raise ValueError("f must contain only finite values.")
    if np.any((f_arr < 0.0) | (f_arr > 1.0)):
        raise ValueError("f must contain physical occupations in [0, 1].")
    f = f_arr

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

    # σ₁'s partner term zero-fills f above E_max (np.interp right=0.0). For a
    # physical, decaying occupation this drops a negligible top-slice tail, but
    # a non-decayed grid would silently under-integrate — warn, don't hide it.
    if float(f[-1]) > 1e-5:
        warnings.warn(
            "compute_ac_conductivity: occupation at the top of the energy grid "
            f"f(E_max)={float(f[-1]):.3g} is non-negligible; the σ₁ partner term "
            "zero-fills f above E_max, dropping the high-energy tail. Extend the grid.",
            RuntimeWarning,
            stacklevel=2,
        )

    # σ₁: supergap integral over [Δ, ∞).
    E_partner = E + omega_0
    f_partner = np.interp(E_partner, E, f, right=0.0)
    rho_partner = bcs_density_of_states(E_partner, gap)
    K_plus_partner = 1.0 + gap ** 2 / np.maximum(E * E_partner, 1e-30)

    U_plus = rho_partner * K_plus_partner
    # Integrate the singular leading DOS analytically over every cell. The
    # remaining factor is regular at E=Δ for a sub-gap probe (E+ω₀>Δ).
    dos_weights = bcs_dos_cell_weights(E, dE, gap)
    integrand_1_regular = (f - f_partner) * U_plus
    sigma_1_norm = (2.0 / omega_0) * float(
        np.sum(integrand_1_regular * dos_weights)
    )

    # σ₂: sub-gap integral over [Δ − ω₀, Δ]. The raw-E
    # midpoint rule converges only as O(n^{-1/2}) because BOTH endpoints are
    # square-root singular: the analytically continued DOS diverges at Δ,
    # while rho(E+ω₀) diverges at Δ-ω₀. Use
    # E = Δ - ω₀ cos²(theta); its Jacobian cancels both singularities,
    # leaving a smooth integrand for the midpoint rule.
    E_lo = max(gap - omega_0, 0.0)
    if E_lo >= gap:
        sigma_2_norm = 0.0
    else:
        dtheta = 0.5 * np.pi / n_subgap
        theta = (np.arange(n_subgap) + 0.5) * dtheta
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        span = gap - E_lo
        E_sub = gap - span * cos_theta**2

        E_sub_partner = E_sub + omega_0
        f_sub_partner = np.interp(E_sub_partner, E, f, right=0.0)
        rho_sub_partner = bcs_density_of_states(E_sub_partner, gap)
        K_plus_sub = 1.0 + gap ** 2 / np.maximum(E_sub * E_sub_partner, 1e-30)
        U_plus_sub = rho_sub_partner * K_plus_sub

        # E / √(Δ² − E²): analytic continuation of the DOS below the gap.
        subgap_dos = E_sub / np.sqrt(np.maximum(gap ** 2 - E_sub ** 2, 1e-30))
        jacobian = 2.0 * span * sin_theta * cos_theta

        integrand_2 = (
            (1.0 - 2.0 * f_sub_partner)
            * U_plus_sub
            * subgap_dos
            * jacobian
        )
        sigma_2_norm = (1.0 / omega_0) * float(np.sum(integrand_2) * dtheta)

    return sigma_1_norm, sigma_2_norm
