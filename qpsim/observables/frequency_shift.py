"""Fractional resonator frequency shift ``δω/ω`` from Mattis–Bardeen σ₂.

δω/ω = (α/2) · (σ₂(f) − σ₂(f_ref)) / σ₂(f_ref).
"""

from __future__ import annotations

import numpy as np

from qpsim.observables.ac_conductivity import compute_ac_conductivity
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

    Returns ``0.0`` if ``σ₂(f_ref) ≤ 0`` (normal state reference).
    """
    _, s2 = compute_ac_conductivity(f, ctx, omega_0, n_subgap=n_subgap)
    _, s2_ref = compute_ac_conductivity(f_ref, ctx, omega_0, n_subgap=n_subgap)

    if s2_ref <= 0:
        return 0.0

    return (alpha / 2.0) * (s2 - s2_ref) / s2_ref
