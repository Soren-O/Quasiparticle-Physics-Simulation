"""Fractional resonator frequency shift ``δω/ω`` from Mattis–Bardeen σ₂.

δω/ω = (α/2) · (σ₂(f) − σ₂(f_ref)) / σ₂(f_ref).
"""

from __future__ import annotations

import numpy as np

from qpsim.observables.ac_conductivity import (
    _finite_real_scalar,
    compute_ac_conductivity,
)
from qpsim.physics.spectral import SpectralContext


def compute_frequency_shift(
    f: np.ndarray,
    f_ref: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    alpha: float,
    *,
    n_subgap: int = 500,
) -> float:
    r"""Fractional frequency shift ``δω/ω`` between ``f`` and ``f_ref``.

    .. math::

        \frac{\delta\omega}{\omega}
            = \frac{\alpha}{2}\,
              \frac{\sigma_2(f) - \sigma_2(f_{\mathrm{ref}})}
                   {\sigma_2(f_{\mathrm{ref}})}.

    A negative nonzero ``σ₂(f_ref)`` is an active reference and retains its
    sign in the ratio.  If the reference response is exactly zero, the ratio
    is undefined unless the current response is also zero; that case fails
    loudly rather than being mislabeled as no shift.
    """
    alpha = _finite_real_scalar("alpha", alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1].")
    _, s2 = compute_ac_conductivity(f, ctx, omega_0, n_subgap=n_subgap)
    _, s2_ref = compute_ac_conductivity(f_ref, ctx, omega_0, n_subgap=n_subgap)

    if s2_ref == 0.0:
        if s2 != 0.0:
            raise ValueError(
                "Cannot form the fractional frequency shift: reference "
                "sigma2 is zero while the current sigma2 differs."
            )
        return 0.0

    return (alpha / 2.0) * (s2 - s2_ref) / s2_ref
