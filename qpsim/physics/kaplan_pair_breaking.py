r"""Kaplan 1976 frequency-resolved pair-breaking phonon lifetime τ_PB(Ω).

Implements the zero-temperature phonon pair-breaking rate from
Kaplan, Chi, Langenberg, Chang, Jafarey, Scalapino,
*Phys. Rev. B* **14**, 4854 (1976), Eq. (27):

.. math::

    \frac{1}{\tau_B(\Omega, T)}
    = \frac{4\pi N(0)\,\alpha^2(\Omega)}{\hbar N}
      \int_\Delta^{\Omega - \Delta}
        \frac{\omega(\Omega - \omega) + \Delta^2}
             {\sqrt{\omega^2 - \Delta^2}\,\sqrt{(\Omega - \omega)^2 - \Delta^2}}
        [1 - f(\omega) - f(\Omega - \omega)]\,d\omega.

Using Kaplan Eq. (30) to define the characteristic time
``τ_0^ph = ℏ N / (4 π² N(0) ⟨α²⟩_av Δ(0))``, the zero-temperature
(``f = 0``) form of the integral factors into a dimensionless
Kaplan integral ``S_+(Ω/Δ)`` and a simple prefactor:

.. math::

    \frac{1}{\tau_B(\Omega, 0)}
    = \frac{\Delta}{\pi\,\Delta_0\,\tau_0^{ph}}\,S_+(\Omega/\Delta),
    \quad
    S_+(x)
    = \int_1^{x-1} \frac{u(x-u) + 1}
                          {\sqrt{u^2 - 1}\,\sqrt{(x-u)^2 - 1}}\,du.

The ``S_+`` integrand is the symmetric-gap (``z = 1``) limit of the
M25 photon spectral density ``S^+_{ph}(x; z)`` already implemented
in :mod:`qpsim.services.rate_equation_coefficients`; both reduce
(via a single change of variable) to the same complete-elliptic
closed form:

.. math::

    S_+(x) = x \cdot E\!\left(1 - 4/x^2\right),

where ``E`` is the complete elliptic integral of the second kind with
the ``m = k²`` parameter convention used by scipy.

Distinct from the thin-film acoustic escape time ``τ_l ≈ 4 d / (η s)``
of Kaplan **1979** (see :mod:`qpsim.physics.phonon_escape`). The 1976
τ_PB is a bulk-BCS quantity independent of film geometry.

Material-specific ``τ_0^{ph}`` values from Kaplan 1976 Table II:

=====  ================
Metal  τ_0^ph (ns)
=====  ================
Pb     0.0340
In     0.169
Sn     0.110
Hg     0.135
Tl     0.205
Ta     0.0227
Nb     0.00417
Al     0.242
Zn     2.31
=====  ================
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.special import ellipe


def _finite_real_scalar(name: str, raw: float) -> float:
    """Return one finite real scalar, rejecting bool/complex coercions."""
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


def kaplan_S_plus(x: float) -> float:
    r"""Dimensionless Kaplan integral :math:`S_+(x)`, closed form.

    .. math::

        S_+(x) = \int_1^{x-1}
                   \frac{u(x-u) + 1}
                        {\sqrt{u^2 - 1}\,\sqrt{(x-u)^2 - 1}}\,du
               = x\,E\!\left(1 - 4/x^2\right)
                 \quad (x \ge 2),

    where ``E(m)`` is the complete elliptic integral of the second
    kind with parameter ``m = k²``.

    Parameters
    ----------
    x
        Dimensionless phonon energy ``Ω / Δ``. Must be positive.

    Returns
    -------
    float
        ``S_+(x)`` for ``x >= 2``; zero below the pair-breaking threshold.
        At ``x = 2`` the integration interval collapses onto the two BCS
        singular endpoints, but their finite right-limit is
        :math:`S_+(2) = \pi`.
    """
    x = _finite_real_scalar("x", x)
    if x < 2.0:
        return 0.0
    # m = k² with k² = 1 - 4/x². For x → 2⁺, m → 0; for x → ∞, m → 1.
    m = 1.0 - 4.0 / (x * x)
    return float(x * ellipe(m))


def kaplan_S_plus_numerical(x: float) -> float:
    r"""Fallback numerical quadrature of the Kaplan integral.

    Used by the test suite to cross-check :func:`kaplan_S_plus`
    against a direct evaluation of the integrand.
    """
    x = _finite_real_scalar("x", x)
    if x < 2.0:
        return 0.0
    if x == 2.0:
        # Direct quadrature over the zero-width interval cannot represent the
        # integrable endpoint singularities. Match the analytic right-limit.
        return float(np.pi)

    def integrand(u: float) -> float:
        denom = np.sqrt(u * u - 1.0) * np.sqrt((x - u) * (x - u) - 1.0)
        if denom <= 0.0:
            return 0.0
        return (u * (x - u) + 1.0) / denom

    val, _ = quad(integrand, 1.0, x - 1.0, limit=200)
    return float(val)


def tau_PB_inverse_Hz(
    omega_kelvin: float,
    gap_kelvin: float,
    tau_0_phonon_ns: float,
    *,
    Delta_0_kelvin: float | None = None,
) -> float:
    r"""Zero-temperature Kaplan pair-breaking rate :math:`\tau_B^{-1}(\Omega)` in Hz.

    Evaluates

    .. math::

        \frac{1}{\tau_B(\Omega, 0)}
        = \frac{\Delta}{\pi\,\Delta_0\,\tau_0^{ph}}\,S_+(\Omega/\Delta)

    Below the pair-breaking threshold ``Ω < 2Δ``, returns 0. At exact
    threshold the ideal-BCS ``K+`` endpoint singularity has the finite
    right-limit :math:`1/\tau_B = \Delta/(\Delta_0 \tau_0^{ph})`. This is
    distinct from
    electromagnetic ``K-`` pair generation, whose channel remains strictly
    closed at ``Ω = 2Δ``.

    Parameters
    ----------
    omega_kelvin
        Phonon energy ``Ω`` (K). Equivalent to ``ℏΩ / k_B`` for
        angular frequency Ω.
    gap_kelvin
        Current gap ``Δ`` (K). For the T=0 limit this equals Δ(0);
        pass Δ(T) explicitly if you want the T>0 formal form.
    tau_0_phonon_ns
        Kaplan characteristic phonon time ``τ_0^{ph}`` in nanoseconds.
        See Kaplan 1976 Table II.
    Delta_0_kelvin
        T=0 reference gap Δ(0) used to normalize the prefactor. If
        ``None`` (default), set equal to ``gap_kelvin`` — the
        T=0/zero-suppression case.

    Returns
    -------
    float
        Pair-breaking rate ``1/τ_B`` in Hz.

    Raises
    ------
    ValueError
        If any input is nonpositive.
    """
    omega_kelvin = _finite_real_scalar("omega_kelvin", omega_kelvin)
    gap_kelvin = _finite_real_scalar("gap_kelvin", gap_kelvin)
    tau_0_phonon_ns = _finite_real_scalar(
        "tau_0_phonon_ns", tau_0_phonon_ns
    )
    if omega_kelvin <= 0.0:
        raise ValueError(f"omega_kelvin must be positive; got {omega_kelvin}")
    if gap_kelvin <= 0.0:
        raise ValueError(f"gap_kelvin must be positive; got {gap_kelvin}")
    if tau_0_phonon_ns <= 0.0:
        raise ValueError(
            f"tau_0_phonon_ns must be positive; got {tau_0_phonon_ns}"
        )
    Delta_0 = (
        gap_kelvin
        if Delta_0_kelvin is None
        else _finite_real_scalar("Delta_0_kelvin", Delta_0_kelvin)
    )
    if Delta_0 <= 0.0:
        raise ValueError(f"Delta_0_kelvin must be positive; got {Delta_0}")

    x = omega_kelvin / gap_kelvin
    S = kaplan_S_plus(x)
    # (Δ / (π Δ₀ τ_0^ph)) × S; with τ_0^ph in ns → multiply by 1e9 for Hz.
    return float((gap_kelvin / Delta_0) / (np.pi * tau_0_phonon_ns) * 1e9 * S)
