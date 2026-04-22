"""Electron–phonon collision integral (J_1^eph).

Ported from the old ``qpsim/numerics/collision_phonon.py`` at Gate 2
with the numba acceleration path dropped. Provides:

- ``phonon_collision_rates``: the main ``(gain, loss_rate)`` computation
  for a quasiparticle distribution ``f(E)`` at one spatial pixel.
- ``CoherenceAssignment``, ``build_scattering_kernel_base``,
  ``build_recombination_kernel_base``: coherence-factor wiring (swap
  K⁺ ↔ K⁻ under the photon convention).
- ``build_phonon_frequency_map``,
  ``phonon_occupation_matrices_from_state``,
  ``compute_phonon_source_sink``: the dynamic-phonon coupling pieces
  used when ``n_ph(ω, t)`` is a live dynamical variable.
- ``apply_phonon_collision``: ETD1 step that combines the above for the
  thermal-bath case.

The ``_etd1_step`` helper will migrate to ``qpsim.solvers.etd`` when
the ETD2 upgrade lands (Build Handoff commitment).
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.physics.kernels import recombination_kernel_base as _recombination_kernel_base
from qpsim.physics.kernels import scattering_kernel_base as _scattering_kernel_base
from qpsim.physics.spectral import SpectralContext


class CoherenceAssignment(Enum):
    """Selects which coherence factor pairs with scattering vs recombination.

    * ``PHONON`` — standard QP-phonon: scattering uses K⁻, recombination K⁺.
    * ``PHOTON`` — QP-photon (Fischer 2023 sub-gap): swap K⁺ ↔ K⁻.
    """

    PHONON = "phonon"
    PHOTON = "photon"


def build_scattering_kernel_base(
    ctx: SpectralContext,
    tau_0: float,
    T_c: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    """Base scattering kernel K₀ˢ(E_i, E_j) with coherence wiring, shape (NE, NE).

    Thin wrapper around ``qpsim.physics.kernels.scattering_kernel_base`` that
    pulls the precomputed coherence matrix from ``ctx`` and honors the
    photon-convention swap.
    """
    coh = ctx.K_minus if coherence is CoherenceAssignment.PHONON else ctx.K_plus
    return _scattering_kernel_base(ctx.E, ctx.gap, tau_0, T_c, coherence_factor=coh)


def build_recombination_kernel_base(
    ctx: SpectralContext,
    tau_0: float,
    T_c: float,
    *,
    coherence: CoherenceAssignment = CoherenceAssignment.PHONON,
) -> np.ndarray:
    """Base recombination kernel K₀ʳ(E_i, E_j), shape (NE, NE)."""
    coh = ctx.K_plus if coherence is CoherenceAssignment.PHONON else ctx.K_minus
    return _recombination_kernel_base(ctx.E, ctx.gap, tau_0, T_c, coherence_factor=coh)


def _thermal_phonon_scattering_occupation(
    E: np.ndarray, T_bath: float,
) -> np.ndarray:
    """(NE, NE) scattering phonon occupation at bath temperature T_bath.

    * Emission (E_i > E_j): ``1 + n_BE(E_i − E_j)``
    * Absorption (E_i < E_j): ``n_BE(E_j − E_i)``
    * Diagonal: 0.
    """
    E_diff = E[:, None] - E[None, :]
    kBT = _KB_UEV_PER_K * T_bath if T_bath > 0 else 0.0
    N_p = np.zeros_like(E_diff)

    if kBT > 0:
        emission = E_diff > 0
        absorption = E_diff < 0
        exp_em = np.minimum(E_diff[emission] / kBT, 500.0)
        exp_abs = np.minimum(-E_diff[absorption] / kBT, 500.0)
        N_p[emission] = 1.0 + 1.0 / (np.exp(exp_em) - 1.0)
        N_p[absorption] = 1.0 / (np.exp(exp_abs) - 1.0)
    else:
        N_p[E_diff > 0] = 1.0
        N_p[E_diff < 0] = 0.0

    return N_p


def _thermal_phonon_recombination_occupations(
    E: np.ndarray, T_bath: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(N_emit, N_abs)`` at bath temperature ``T_bath``.

    * ``N_emit = 1 + n_BE(E_i + E_j)``: phonon emission from recombination.
    * ``N_abs  = n_BE(E_i + E_j)``: pair-breaking absorption.
    """
    E_sum = E[:, None] + E[None, :]
    kBT = _KB_UEV_PER_K * T_bath if T_bath > 0 else 0.0

    if kBT > 0:
        exp_sum = np.minimum(E_sum / kBT, 500.0)
        N_BE = 1.0 / (np.exp(exp_sum) - 1.0)
    else:
        N_BE = np.zeros_like(E_sum)

    return 1.0 + N_BE, N_BE


def build_phonon_frequency_map(
    E_bins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build an ω-grid and (i, j) → ω-index maps for QP–phonon coupling.

    For every ``(E_i, E_j)`` pair, identifies the corresponding scattering
    phonon frequency ``|E_i − E_j|`` and the recombination phonon
    frequency ``E_i + E_j``. Returns:

    * ``omega_bins``: unique-sorted 1D ω grid.
    * ``omega_idx_diff``, ``omega_idx_sum``: ``(NE, NE)`` int indices
      into ``omega_bins``.
    * ``diff_sign``: ``(NE, NE)`` int8, ``sign(E_i − E_j)``. Used to
      distinguish emission (+) from absorption (−) in the dynamic-n_ph
      projection.
    """
    E = np.asarray(E_bins, dtype=float)
    if E.ndim != 1:
        raise ValueError("E_bins must be a 1D array.")
    E_diff_abs = np.abs(E[:, None] - E[None, :])
    E_sum = E[:, None] + E[None, :]
    all_vals = np.concatenate([E_diff_abs.ravel(), E_sum.ravel()])
    omega_bins, inverse = np.unique(np.round(all_vals, 12), return_inverse=True)
    n_pairs = E.size * E.size
    omega_idx_diff = inverse[:n_pairs].reshape((E.size, E.size))
    omega_idx_sum = inverse[n_pairs:].reshape((E.size, E.size))
    diff_sign = np.sign(E[:, None] - E[None, :]).astype(np.int8)
    return omega_bins, omega_idx_diff, omega_idx_sum, diff_sign


def phonon_occupation_matrices_from_state(
    n_ph: np.ndarray,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project a non-equilibrium ``n_ph(ω)`` onto the three (NE, NE) matrices.

    * ``N_p``: scattering occupation (emission/absorption by sign of i − j).
    * ``N_emit``: recombination emission, ``1 + n_ph(E_i + E_j)``.
    * ``N_abs``: pair-breaking absorption, ``n_ph(E_i + E_j)``.
    """
    n_diff = n_ph[omega_idx_diff]
    n_sum = n_ph[omega_idx_sum]

    N_p = np.where(diff_sign > 0, 1.0 + n_diff, n_diff)
    np.fill_diagonal(N_p, 0.0)

    N_emit = 1.0 + n_sum
    N_abs = n_sum.copy()

    return N_p, N_emit, N_abs


def compute_phonon_source_sink(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    omega_idx_diff: np.ndarray,
    omega_idx_sum: np.ndarray,
    diff_sign: np.ndarray,
    n_omega: int,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Affine-ODE coefficients ``(a_ph, b_ph)`` for the phonon distribution.

    From the current QP occupation ``f(E)``, produces ``(a_ph, b_ph)`` of
    shape ``(n_omega,)`` such that ``dn_ph/dt = a_ph + b_ph · n_ph``.
    Used by the dynamic-phonon backend when ``n_ph`` evolves
    self-consistently (finite-τ_l regime).
    """
    rho = ctx.rho
    dE = ctx.dE
    n_qp = rho * f
    one_minus_f = np.maximum(1.0 - f, 0.0)
    partner = rho * one_minus_f

    a_ph = np.zeros(n_omega)
    b_ph = np.zeros(n_omega)

    if enable_scattering and K_s0 is not None:
        base_sc = dE * (n_qp[:, None] * K_s0 * (rho[None, :] * one_minus_f[None, :]))
        emit_mask = diff_sign > 0
        abs_mask = diff_sign < 0
        if np.any(emit_mask):
            emit = np.bincount(
                omega_idx_diff[emit_mask].ravel(),
                weights=base_sc[emit_mask].ravel(),
                minlength=n_omega,
            )
            a_ph += emit
            b_ph += emit
        if np.any(abs_mask):
            absor = np.bincount(
                omega_idx_diff[abs_mask].ravel(),
                weights=base_sc[abs_mask].ravel(),
                minlength=n_omega,
            )
            b_ph -= absor

    if enable_recombination and K_r0 is not None:
        base_rec = dE * (n_qp[:, None] * K_r0 * n_qp[None, :])
        rec = np.bincount(
            omega_idx_sum.ravel(),
            weights=base_rec.ravel(),
            minlength=n_omega,
        )
        a_ph += rec
        b_ph += rec
        base_pb = dE * (partner[:, None] * K_r0 * partner[None, :])
        pb = np.bincount(
            omega_idx_sum.ravel(),
            weights=base_pb.ravel(),
            minlength=n_omega,
        )
        b_ph -= pb

    return a_ph, b_ph


def phonon_collision_rates(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
    N_p_override: np.ndarray | None = None,
    N_emit_override: np.ndarray | None = None,
    N_abs_override: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``(gain, loss_rate)`` for the e–phonon collision integral.

    Returns arrays of shape ``(NE,)`` such that ``df/dt = gain − loss_rate · f``.
    Works on the occupation ``f`` (not the spectral density ``n = ρ·f``).
    The ``gain`` includes the ``(1 − f_i)`` Pauli factor already.

    When ``N_*_override`` arguments are supplied, they replace the thermal
    Bose-Einstein factors that would otherwise be computed from ``T_bath``.
    Used by the dynamic-phonon backend when ``n_ph`` is evolved
    self-consistently.
    """
    E = ctx.E
    rho = ctx.rho
    dE = ctx.dE
    one_minus_f = np.maximum(1.0 - f, 0.0)

    gain = np.zeros_like(f)
    loss_rate = np.zeros_like(f)

    if enable_scattering and K_s0 is not None:
        N_p = (
            N_p_override
            if N_p_override is not None
            else _thermal_phonon_scattering_occupation(E, T_bath)
        )
        K_s_eff = K_s0 * N_p
        n_qp = rho * f
        gain += one_minus_f * (K_s_eff.T @ (n_qp * dE))
        loss_rate += K_s_eff @ (rho * one_minus_f * dE)

    if enable_recombination and K_r0 is not None:
        if N_emit_override is not None and N_abs_override is not None:
            N_emit, N_abs = N_emit_override, N_abs_override
        else:
            N_emit, N_abs = _thermal_phonon_recombination_occupations(E, T_bath)
        partner = rho * one_minus_f
        loss_rate += 2.0 * ((K_r0 * N_emit) @ (rho * f * dE))
        gain += 2.0 * one_minus_f * ((K_r0 * N_abs) @ (partner * dE))

    return gain, loss_rate


def _etd1_step(
    f: np.ndarray,
    gain: np.ndarray,
    loss_rate: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Exponential-Euler (ETD1) step for ``df/dt = gain − loss_rate · f``.

    Preserves ``0 ≤ f ≤ 1`` when ``gain ≥ 0``. Will migrate to
    ``qpsim.solvers.etd`` with the ETD2 upgrade (Build Handoff).
    """
    mu = np.maximum(loss_rate, 0.0)
    p_term = np.maximum(gain + (mu - loss_rate) * f, 0.0)

    decay = np.exp(-mu * dt)
    coeff = np.empty_like(mu)
    small = mu < 1e-14
    coeff[~small] = (1.0 - decay[~small]) / mu[~small]
    coeff[small] = dt

    updated = decay * f + coeff * p_term
    return np.clip(updated, 0.0, 1.0)


def apply_phonon_collision(
    f: np.ndarray,
    ctx: SpectralContext,
    K_s0: np.ndarray | None,
    K_r0: np.ndarray | None,
    T_bath: float,
    dt: float,
    *,
    enable_scattering: bool = True,
    enable_recombination: bool = True,
) -> np.ndarray:
    """One ETD1 collision step at a single spatial pixel (thermal bath).

    Convenience wrapper: calls ``phonon_collision_rates`` with the
    thermal-bath phonon factors, then applies one ``_etd1_step``. For
    the dynamic-phonon case, callers assemble the step explicitly using
    ``phonon_collision_rates`` with ``N_*_override`` arguments.
    """
    gain, loss_rate = phonon_collision_rates(
        f, ctx, K_s0, K_r0, T_bath,
        enable_scattering=enable_scattering,
        enable_recombination=enable_recombination,
    )
    return _etd1_step(f, gain, loss_rate, dt)
