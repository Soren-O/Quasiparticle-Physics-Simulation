"""Sub-gap (ω₀ < 2Δ) photon scattering collision integral.

Single-mode photon scattering under the photon coherence convention
(``K⁺`` instead of ``K⁻``). The photon frequency must be grid-
commensurate with the QP energy grid to avoid silent interpolation
errors; mismatches above 1% are rejected.

Ported from ``photon_collision_rates`` in the old
``qpsim/numerics/collision_phonon.py``.
"""

from __future__ import annotations

import numpy as np

from qpsim.collisions._uniform_grid import uniform_grid_spacing
from qpsim.collisions._validation import validated_occupation
from qpsim.physics.spectral import SpectralContext

# Public so frontends can pre-check commensurability against the same
# threshold the kernel accepts before snapping to the nearest lattice point.
COMMENSURATE_TOL = 0.01
_COMMENSURATE_TOL = COMMENSURATE_TOL


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
    ``i ± m`` where ``m = round(ω₀ / dE)``. Frequencies with
    ``|ω₀ − m·dE| / dE > 1%`` are rejected.

    Returns
    -------
    gain, loss_rate
        Each of shape ``(NE,)``, with ``df/dt = gain − loss_rate · f``.
        The ``gain`` includes the ``(1 − f_i)`` Pauli factor.
    """
    E = ctx.E
    NE = E.size
    f = validated_occupation(f, E.shape, "Sub-gap photon collision")
    dE_scalar = uniform_grid_spacing(E, ctx.dE, "Sub-gap photon collision")

    if not np.isfinite(omega_0) or omega_0 < 0.0:
        raise ValueError(f"omega_0 must be finite and non-negative; got {omega_0}.")
    if not np.isfinite(n_bar) or n_bar < 0.0:
        raise ValueError(f"n_bar must be finite and non-negative; got {n_bar}.")
    if not np.isfinite(c_phot) or c_phot < 0.0:
        raise ValueError(f"c_phot must be finite and non-negative; got {c_phot}.")
    if ctx.dynes_gamma > 0.0:
        raise ValueError(
            "Sub-gap photon collisions do not support dynes_gamma > 0: "
            "the collision coherence factors are ideal BCS."
        )
    if omega_0 >= 2.0 * ctx.gap:
        raise ValueError(
            f"Sub-gap photon collisions require omega_0 < 2*gap; got "
            f"omega_0={omega_0:.6g} and 2*gap={2.0 * ctx.gap:.6g} micro-eV. "
            "Use pair_breaking_photon_collision_rates for an above-gap drive."
        )

    m = round(omega_0 / dE_scalar)
    if omega_0 == 0.0:
        return np.zeros(NE), np.zeros(NE)
    if m <= 0:
        raise ValueError(
            f"omega_0={omega_0:.6g} μeV is below half the grid spacing "
            f"dE={dE_scalar:.6g} μeV and cannot be represented; refine the "
            "energy grid or set omega_0=0 explicitly to disable the channel."
        )

    frac_err = abs(omega_0 - m * dE_scalar) / dE_scalar
    if frac_err > _COMMENSURATE_TOL:
        raise ValueError(
            f"Photon frequency omega_0={omega_0:.6g} μeV is not grid-"
            f"commensurate (dE={dE_scalar:.6g} μeV, nearest m={m}, "
            f"fractional error={frac_err:.4f} > tol={_COMMENSURATE_TOL}). "
            f"Use m·dE={m * dE_scalar:.6g} μeV or refine the energy grid."
        )

    # A fixed photon step maps one finite-volume cell to another.  The
    # transition coefficient is the partner's cell-average DOS, not its
    # point sample: with capacity w_i this gives pairwise event flux
    # w_i*rho_bar_j = w_i*w_j/dE = w_j*rho_bar_i on the uniform grid.
    # Mixing exact w_i with point rho_j violates number conservation and can
    # turn a driven scattering channel into a large artificial sink.
    rho = ctx.cell_density
    supported = ctx.active_mask
    K_plus = ctx.K_plus

    gain = np.zeros(NE)
    loss_rate = np.zeros(NE)
    one_minus_f = np.maximum(1.0 - f, 0.0)

    for i in range(NE):
        if not supported[i]:
            continue

        # Partner above at i + m: the gain at i is photon EMISSION by
        # the QP at i+m (factor n̄+1); the loss line below is
        # absorption at i up to i+m (factor n̄).
        j_up = i + m
        if j_up < NE:
            U_plus = rho[j_up] * K_plus[i, j_up]
            gain[i] += c_phot * U_plus * f[j_up] * (n_bar + 1)
            loss_rate[i] += c_phot * U_plus * one_minus_f[j_up] * n_bar

        # Emit-down partner: i − m (QP emits photon).
        j_down = i - m
        if j_down >= 0 and supported[j_down]:
            U_plus = rho[j_down] * K_plus[i, j_down]
            gain[i] += c_phot * U_plus * f[j_down] * n_bar
            loss_rate[i] += c_phot * U_plus * one_minus_f[j_down] * (n_bar + 1)

    gain_with_pauli = gain * one_minus_f
    return gain_with_pauli, loss_rate
