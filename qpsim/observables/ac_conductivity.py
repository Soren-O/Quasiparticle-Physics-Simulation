"""Mattis–Bardeen ac conductivity from a quasiparticle distribution.

Returns normalized conductivities ``σ₁/σ_N`` and ``σ₂/σ_N`` — the
normal-state ``σ_N`` cancels out of the ``Q_i`` and ``δω/ω``
observables that build on these integrals.

The super-gap ``σ₁`` integral pairs the analytic pure-BCS DOS measure of each
cell with the remaining regular factor sampled at that cell's center — the
finite-volume convention of ``qpsim.physics.bcs_quadrature``. The measure is
exact, but the
pairing is not: a gap-edge cell's measure is concentrated near its lower edge
while ``f``, ``ρ(E+ω₀)`` and ``K⁺`` all fall steeply away from that edge, so
``σ₁`` is one-signed LOW and converges only as ``O(dE^{3/2})``. Against
adaptive quadrature of the same integrand (Δ = 182.4 μeV, ω₀ = 20.7 μeV,
thermal ``f``) the deficit is −12.4 % at 40 bins on ``[Δ, 5Δ]`` and T = 0.5 K,
−26.7 % there at T = 0.2 K, and −3.7 % at 405 bins on ``[Δ, 10Δ]`` at
T = 0.2 K; a nonequilibrium ``f`` peaked inside the first cell is worse
(−63 % at 28 bins on ``[Δ, 5Δ]`` for a 10 μeV edge decay). ``Q_i ∝ 1/σ₁``
is correspondingly overstated on coarse grids, so resolve ``σ₁`` in ``dE``
before quoting one. The sub-gap ``σ₂``
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
    try:
        f_arr = np.asarray(f_raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("f must be a real numeric array.") from exc
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

    omega_0 = _finite_real_scalar("omega_0", omega_0)
    if omega_0 <= 0.0:
        raise ValueError("omega_0 must be positive.")
    if (
        not isinstance(n_subgap, (int, np.integer))
        or isinstance(n_subgap, (bool, np.bool_))
        or n_subgap <= 0
    ):
        raise ValueError("n_subgap must be a positive integer.")
    n_subgap = int(n_subgap)
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
    # Integrate the singular leading DOS analytically over every cell and
    # sample the remaining factor at the cell center. That factor is finite at
    # E=Δ for a sub-gap probe (E+ω₀>Δ) but steep there, while the cell measure
    # is concentrated at the cell's lower edge, so this cell-constant pairing
    # is one-signed low and only O(dE^{3/2}) accurate — see the module
    # docstring for the measured deficit at production grid sizes.
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

        if int(np.count_nonzero(ctx.active_mask)) < 2:
            # With one active centre there is nothing to interpolate between:
            # every node lands beyond it and `right=0.0` would silently
            # zero-fill the whole partner reconstruction, returning a
            # confident sigma_2 built from no occupation at all.
            raise ValueError(
                "sigma_2 needs at least two active energy cells to "
                "reconstruct the partner occupation; this context has "
                f"{int(np.count_nonzero(ctx.active_mask))}. Widen the energy "
                "grid above the gap."
            )
        E_sub_partner = E_sub + omega_0
        # Reconstruct f from the ACTIVE centers only. On a grid extended below
        # the gap, the sub-gap centers carry zero spectral capacity and stay
        # frozen at whatever seed the caller supplied -- they are placeholders,
        # not occupations. Interpolating over all centers blended those
        # placeholders into sigma_2 through the low-theta nodes, which sit just
        # above Delta. So sigma_2 depended on a value that means nothing, and
        # the dependence was invisible on any grid starting at Delta.
        # gap_suppression.edge_samples_from_centers already masks exactly these
        # nodes for exactly this reason; this brings sigma_2 onto the same
        # convention. np.interp's left clamp holds the first active cell value
        # below that centre, which is the bound-preserving finite-volume choice.
        #
        # sigma_1 above is structurally immune and is deliberately untouched:
        # sub-gap cells carry zero DOS weight there, and every weighted cell's
        # partner E_i + omega_0 lies above the first active centre, so no frozen
        # centre can enter its bracket.
        active = ctx.active_mask
        f_sub_partner = np.interp(
            E_sub_partner, E[active], f[active], right=0.0,
        )
        rho_sub_partner = bcs_density_of_states(E_sub_partner, gap)
        K_plus_sub = 1.0 + gap ** 2 / np.maximum(E_sub * E_sub_partner, 1e-30)
        U_plus_sub = rho_sub_partner * K_plus_sub

        # E / √(Δ² − E²): analytic continuation of the DOS below the gap.
        #
        # FACTORED, not Δ² − E². Squaring first destroys the significance of
        # Δ − E exactly where this substitution puts its nodes: E = Δ − ω₀cos²θ
        # clusters them AT the gap edge on purpose, reaching within 5.4e-05 µeV
        # of Δ at the shipped 500-point rule. The two squares then agree to
        # ~13 digits and their difference is nearly all cancellation, while
        # Δ − E is computed directly and stays exact — by Sterbenz's lemma the
        # subtraction is exact outright for Δ/2 ≤ E ≤ 2Δ, which is every node
        # here.
        #
        # Measured against 60-digit arithmetic at the shipped grid: worst
        # relative error on the integrand falls 1.53e-11 → 2.45e-16, a factor
        # of 6.2e4, and σ₂'s sub-gap sum moves 3.9e-12. This is the mirror of
        # the same reformulation in physics/spectral.py, and unlike that one it
        # is reachable BY CONSTRUCTION rather than only on a grid nobody uses.
        subgap_dos = E_sub / np.sqrt(
            np.maximum((gap - E_sub) * (gap + E_sub), 1e-30)
        )
        jacobian = 2.0 * span * sin_theta * cos_theta

        integrand_2 = (
            (1.0 - 2.0 * f_sub_partner)
            * U_plus_sub
            * subgap_dos
            * jacobian
        )
        sigma_2_norm = (1.0 / omega_0) * float(np.sum(integrand_2) * dtheta)

    return sigma_1_norm, sigma_2_norm
