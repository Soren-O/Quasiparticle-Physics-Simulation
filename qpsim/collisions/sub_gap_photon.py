"""Sub-gap (ω₀ < 2Δ) photon scattering collision integral.

Single-mode photon scattering under the photon coherence convention
(``K⁺`` instead of ``K⁻``). The photon frequency must be grid-
commensurate with the QP energy grid to avoid silent interpolation
errors; a ``UserWarning`` fires when the mismatch exceeds 1%.

Ported from ``photon_collision_rates`` in the old
``qpsim/numerics/collision_phonon.py``.
"""

from __future__ import annotations

import warnings

import numpy as np

from qpsim.physics.spectral import SpectralContext

_COMMENSURATE_TOL = 0.01


def sub_gap_photon_collision_rates(
    f: np.ndarray,
    ctx: SpectralContext,
    omega_0: float,
    n_bar: float,
    c_phot: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sub-gap photon ``(gain, loss_rate)`` at one spatial pixel.

    Single-mode photon scattering with coherence factor ``K⁺`` (photon
    convention, reversed from the QP–phonon case). Partners at bins
    ``i ± m`` where ``m = round(ω₀ / dE)``; warns if
    ``|ω₀ − m·dE| / dE > 1%``.

    Returns
    -------
    gain, loss_rate
        Each of shape ``(NE,)``, with ``df/dt = gain − loss_rate · f``.
        The ``gain`` includes the ``(1 − f_i)`` Pauli factor.
    """
    E = ctx.E
    NE = E.size
    dE_scalar = float(ctx.dE[0])

    m = round(omega_0 / dE_scalar)
    if m <= 0:
        return np.zeros(NE), np.zeros(NE)

    frac_err = abs(omega_0 - m * dE_scalar) / dE_scalar
    if frac_err > _COMMENSURATE_TOL:
        warnings.warn(
            f"Photon frequency omega_0={omega_0:.6g} μeV is not grid-"
            f"commensurate (dE={dE_scalar:.6g} μeV, nearest m={m}, "
            f"fractional error={frac_err:.4f} > tol={_COMMENSURATE_TOL}). "
            f"Snapped to m·dE={m * dE_scalar:.6g} μeV.",
            stacklevel=2,
        )

    gap = ctx.gap
    rho = ctx.rho
    K_plus = ctx.K_plus

    gain = np.zeros(NE)
    loss_rate = np.zeros(NE)
    one_minus_f = np.maximum(1.0 - f, 0.0)

    for i in range(NE):
        # Absorb-up partner: i + m (QP absorbs photon).
        j_up = i + m
        if j_up < NE:
            U_plus = rho[j_up] * K_plus[i, j_up]
            gain[i] += c_phot * U_plus * f[j_up] * (n_bar + 1)
            loss_rate[i] += c_phot * U_plus * one_minus_f[j_up] * n_bar

        # Emit-down partner: i − m (QP emits photon).
        j_down = i - m
        if j_down >= 0 and E[j_down] >= gap:
            U_plus = rho[j_down] * K_plus[i, j_down]
            gain[i] += c_phot * U_plus * f[j_down] * n_bar
            loss_rate[i] += c_phot * U_plus * one_minus_f[j_down] * (n_bar + 1)

    gain_with_pauli = gain * one_minus_f
    return gain_with_pauli, loss_rate
