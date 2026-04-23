r"""Marchegiani 2025 rate-equation closure — core observables.

Minimal v1: the Lambert-W crossover temperature ``T̄`` from
Marchegiani & Catelani, *Commun. Phys.* **8**, 120 (2025), Eq. 8.
This is the temperature separating the low-``T`` nonequilibrium
regime (photon-assisted pair breaking dominates on the right
electrode) from the local-quasiequilibrium regime (thermal-phonon
pair breaking takes over) in a gap-asymmetric Josephson junction.

From the paper:

.. math::

    \bar T = \frac{2\,\Delta_R}{
        W\!\left( 4\pi \, r^{R_c} / g^\mathrm{ph}_R \right)
    }

where ``W(z)`` is the Lambert-W function (a.k.a. product-log),
``Δ_R`` is the smaller of the two island gaps (Δ_R < Δ_L), ``r^{R_c}``
is the R< recombination-rate coefficient, and ``g^\mathrm{ph}_R`` is
the photon-assisted pair-breaking generation rate on the right
electrode. The paper validates Eq. 8 "within a few percent of the
numerical results" from the full rate-equation integration.

Full rate-equation closure (Eqs. 3-6 with all ``Γ_{ij}^α`` tunneling
integrals, phonon relaxation ``τ_E^{-1}, τ_R^{-1}``, and the
three-chemical-potential ansatz) is deferred per NFP §7.10 Gate 8
strategy A.
"""

from __future__ import annotations

import numpy as np
from scipy.special import lambertw


def crossover_temperature_kelvin(
    *,
    Delta_R_kelvin: float,
    r_Rc_rate_Hz: float,
    g_photon_R_rate_Hz: float,
) -> float:
    r"""Return ``T̄`` (Kelvin) from M25 Eq. 8.

    Parameters
    ----------
    Delta_R_kelvin
        Right-electrode gap in Kelvin (``Δ_R / k_B``). For Al at the
        M25 Fig. 3 parameter set, ``Δ_R/h = 49 GHz`` ⇒ 2.35 K.
    r_Rc_rate_Hz
        Right-electrode recombination-rate coefficient in Hz. The
        M25 Fig. 3 caption gives ``r^{R_c} = 6.25 MHz``.
    g_photon_R_rate_Hz
        Photon-assisted pair-breaking generation rate on R in Hz.
        This is the rate that a photon absorbed at the junction
        creates a QP pair on the R island (one QP in R<, one in R>).
        At the M25 Fig. 3 parameter set this is set by
        ``Γ_{01}^{ph} = 300 Hz`` plus Cooper-pair-count normalization.

    Returns
    -------
    float
        Crossover temperature ``T̄`` in Kelvin. The Lambert-W real
        principal branch ``W_0`` is used; for the physical parameter
        range ``r^{R_c} / g^\mathrm{ph}_R ≫ 1`` this is guaranteed
        real.

    Raises
    ------
    ValueError
        For non-positive inputs.
    RuntimeError
        If the Lambert-W argument lands outside its real-valued
        domain ``[−1/e, ∞)``.
    """
    if Delta_R_kelvin <= 0:
        raise ValueError("Delta_R_kelvin must be positive.")
    if r_Rc_rate_Hz <= 0:
        raise ValueError("r_Rc_rate_Hz must be positive.")
    if g_photon_R_rate_Hz <= 0:
        raise ValueError("g_photon_R_rate_Hz must be positive.")

    arg = 4.0 * np.pi * r_Rc_rate_Hz / g_photon_R_rate_Hz
    if arg < -1.0 / np.e:
        raise RuntimeError(
            f"Lambert-W argument {arg} outside the real-valued domain "
            "[-1/e, ∞). Check input signs."
        )
    w = complex(lambertw(arg, k=0))
    if abs(w.imag) > 1e-12 * max(abs(w.real), 1.0):
        raise RuntimeError(
            f"Lambert-W returned non-real value {w}; check inputs."
        )
    return 2.0 * Delta_R_kelvin / float(w.real)
