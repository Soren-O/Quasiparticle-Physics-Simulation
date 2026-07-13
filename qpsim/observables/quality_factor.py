"""Internal quality factor Q_i from quasiparticle losses.

Q_{i,qp} = σ₂ / (α σ₁) in the Mattis–Bardeen picture, where α is the
kinetic-inductance fraction of the resonator.
"""

from __future__ import annotations

import numpy as np

from qpsim.observables.ac_conductivity import compute_ac_conductivity
from qpsim.physics.spectral import SpectralContext


def compute_quality_factor(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    alpha: float,
    *,
    Q_ext: float | None = None,
    n_subgap: int = 500,
) -> float:
    r"""Internal quality factor ``Q_i`` from the occupation ``f(E)``.

    .. math::

        Q_{i,\mathrm{qp}} = \frac{\sigma_2}{\alpha \, \sigma_1}.

    If ``Q_ext`` is given, combines with ``Q_qp`` via
    ``1/Q_tot = 1/Q_qp + 1/Q_ext`` and returns ``Q_tot``.

    Parameters
    ----------
    f
        Occupation probability, shape ``(NE,)``.
    ctx
        SpectralContext with current Δ (pure BCS only).
    omega_0
        Probe frequency (μeV).
    alpha
        Kinetic-inductance fraction ``α ∈ (0, 1]``.
    Q_ext
        Optional extrinsic quality factor.
    n_subgap
        Quadrature points for the ``σ₂`` sub-gap integral.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    s1, s2 = compute_ac_conductivity(f, ctx, omega_0, n_subgap=n_subgap)
    # sigma_1 < 0 is microwave gain (negative damping), not the zero-loss
    # limit. Preserve its sign in Q so callers can distinguish an active,
    # unstable response from a passive resonator with Q -> infinity.
    Q_qp = s2 / (alpha * s1) if s1 != 0.0 else float(np.inf)

    if Q_ext is not None:
        if not np.isfinite(Q_ext) or Q_ext <= 0.0:
            raise ValueError("Q_ext must be finite and positive when provided.")
        inverse_total = 1.0 / Q_qp + 1.0 / Q_ext
        # Exact gain/loss cancellation is the net zero-damping threshold.
        return float(np.inf) if inverse_total == 0.0 else 1.0 / inverse_total
    return Q_qp
