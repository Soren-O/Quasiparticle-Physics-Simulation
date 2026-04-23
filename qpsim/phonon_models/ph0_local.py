"""Ph0 local-phonon tier: steady-state ``n_ph(ω)`` from QP distribution.

At phonon steady state with a local bath of escape time ``τ_l``, the
Ph0 equation is

    0 = a_ph[f] + b_ph[f] · n_ph + (n_th − n_ph) / τ_l,

where ``(a_ph, b_ph)`` are the affine coefficients of the e-ph
source-sink on the QP distribution (see
:func:`qpsim.collisions.phonon.compute_phonon_source_sink`). Solving:

* ``τ_l = 0`` (no substrate coupling): ``n_ph = −a_ph / b_ph``.
* ``τ_l > 0``: ``n_ph = (a_ph + n_th / τ_l) / (1/τ_l − b_ph)``.

Moved here from the private ``_phonon_steady_state`` helper that
previously lived in :mod:`qpsim.services.steady_state` (Gate 2
task 11).
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions.phonon import compute_phonon_source_sink
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


def phonon_steady_state(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_bins: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    T_bath: float,
    tau_l: float,
) -> np.ndarray:
    """Solve the Ph0 phonon-balance equation for ``n_ph(ω)`` given ``f``.

    Parameters
    ----------
    f
        QP occupation, shape ``(NE,)``.
    ctx
        SpectralContext with current Δ.
    K_s0, K_r0
        Base e-ph kernels (shape ``(NE, NE)``) or ``None`` to disable
        the corresponding channel.
    omega_bins, omega_idx_diff, omega_idx_sum, diff_sign
        Outputs of :func:`qpsim.collisions.phonon.build_phonon_frequency_map`
        for ``ctx.E``.
    T_bath
        Substrate bath temperature (K), used to compute the thermal
        ``n_th(ω)``.
    tau_l
        Phonon bath-escape time (ns). ``0.0`` means no substrate
        coupling; ``> 0`` means finite escape time.

    Returns
    -------
    n_ph
        1D array of shape ``(len(omega_bins),)``, clipped to ``≥ 0``.
    """
    n_omega = len(omega_bins)
    a_ph, b_ph = compute_phonon_source_sink(
        f, ctx, K_s0, K_r0,
        omega_idx_diff, omega_idx_sum, diff_sign, n_omega,
    )

    if tau_l == 0.0:
        denom = b_ph.copy()
        safe = np.abs(denom) > 1e-30
        n_ph = np.zeros(n_omega)
        n_ph[safe] = -a_ph[safe] / denom[safe]
    else:
        inv_tau_l = 1.0 / tau_l
        n_th = thermal_phonon_occupation(omega_bins, T_bath)
        denom = inv_tau_l - b_ph
        safe = np.abs(denom) > 1e-30
        n_ph = np.zeros(n_omega)
        n_ph[safe] = (a_ph[safe] + inv_tau_l * n_th[safe]) / denom[safe]

    return np.maximum(n_ph, 0.0)
