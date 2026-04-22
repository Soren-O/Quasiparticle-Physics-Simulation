"""K₀ˢ / K₀ʳ electron–phonon scattering and recombination kernels.

Kernel definitions (phonon-energy convention, thesis Ch. 2):

    K₀ʳ(E_i, E_j) = (1/τ₀) · (E_i + E_j)² / (kB T_c)³ · K⁺(E_i, E_j)
    K₀ˢ(E_i, E_j) = (1/τ₀) · (E_i − E_j)² / (kB T_c)³ · K⁻(E_i, E_j)

The "with-phonon" variants (``recombination_kernel``,
``scattering_kernel``) multiply by the Bose–Einstein phonon-occupation
factor N_p for a thermal phonon bath.

Ported from the old ``qpsim/numerics/kernels.py`` at Gate 2 with the
numba acceleration path stripped (v1 is pure numpy per Build Handoff).
``thermal_qp_weights`` moved to ``qpsim.physics.spectral``;
``diffusion_coefficient_table`` and ``build_fixed_phonon_history``
are not ported here (they live in ``qpsim.transport.diffusion`` and
``qpsim.phonon_models`` respectively when those land).
"""

from __future__ import annotations

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.physics.spectral import coherence_factor_minus, coherence_factor_plus


def thermal_phonon_occupation(
    omega_bins: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Bose–Einstein thermal phonon occupation n_BE(ω, T) = 1/(exp(ω/kT) − 1).

    Returns zeros at ``temperature <= 0``. Finite positive values elsewhere;
    the exponent is clipped at 500 for numerical safety.
    """
    omega = np.asarray(omega_bins, dtype=float)
    if omega.ndim != 1:
        raise ValueError("omega_bins must be a 1D array.")
    if np.any(~np.isfinite(omega)):
        raise ValueError("omega_bins must contain only finite values.")
    if np.any(omega < 0):
        raise ValueError("omega_bins must be non-negative.")
    if temperature <= 0:
        return np.zeros_like(omega)

    kT = _KB_UEV_PER_K * float(temperature)
    exponent = np.minimum(omega / max(kT, 1e-30), 500.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        occ = 1.0 / (np.exp(exponent) - 1.0)
    occ[~np.isfinite(occ)] = 0.0
    return np.maximum(occ, 0.0)


def recombination_kernel_base(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    *,
    coherence_factor: np.ndarray | None = None,
) -> np.ndarray:
    """K₀ʳ(E_i, E_j) without phonon occupancy, shape ``(NE, NE)``.

    K₀ʳ = (1/τ₀) · ((E_i + E_j)/(kB T_c))² · K⁺ / (kB T_c)

    Pass ``coherence_factor`` to reuse a precomputed K matrix — e.g.
    ``ctx.K_plus`` for the phonon convention, or ``ctx.K_minus`` for
    the photon convention.
    """
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    kBTc = _KB_UEV_PER_K * T_c
    E_sum = E[:, None] + E[None, :]
    coh = (
        coherence_factor_plus(E, gap)
        if coherence_factor is None
        else np.asarray(coherence_factor, dtype=float)
    )
    return (1.0 / tau_0) * (E_sum / kBTc) ** 2 / kBTc * coh


def scattering_kernel_base(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    *,
    coherence_factor: np.ndarray | None = None,
) -> np.ndarray:
    """K₀ˢ(E_i, E_j) without phonon occupancy, shape ``(NE, NE)``.

    K₀ˢ = (1/τ₀) · (E_i − E_j)² · K⁻ / (kB T_c)³

    The diagonal is zeroed (no transfer ⇒ no scattering). Pass
    ``coherence_factor`` to reuse a precomputed K matrix — e.g.
    ``ctx.K_minus`` for phonon convention, ``ctx.K_plus`` for photon.
    """
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    kBTc = _KB_UEV_PER_K * T_c
    E_diff = E[:, None] - E[None, :]
    coh = (
        coherence_factor_minus(E, gap)
        if coherence_factor is None
        else np.asarray(coherence_factor, dtype=float)
    )
    K_s0 = (1.0 / tau_0) * (E_diff ** 2) / kBTc ** 3 * coh
    np.fill_diagonal(K_s0, 0.0)
    return K_s0


def recombination_kernel(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    bath_temperature: float,
) -> np.ndarray:
    """K^r_{ij} = K₀ʳ · N_p(E_i + E_j; T_bath), shape ``(NE, NE)``.

    N_p(ω, T) = 1 + n_BE(ω, T) = 1 + (exp(ω/kT) − 1)⁻¹ is the with-phonon
    factor for phonon emission into a thermal bath. At T = 0, N_p = 1.
    """
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    kBTp = _KB_UEV_PER_K * bath_temperature
    E_sum = E[:, None] + E[None, :]
    if kBTp > 0:
        exponent = np.minimum(E_sum / kBTp, 500.0)
        N_p = 1.0 / (np.exp(exponent) - 1.0) + 1.0
    else:
        N_p = np.ones_like(E_sum, dtype=float)
    return recombination_kernel_base(E, gap, tau_0, T_c) * N_p


def scattering_kernel(
    E_bins: np.ndarray,
    gap: float,
    tau_0: float,
    T_c: float,
    bath_temperature: float,
) -> np.ndarray:
    """K^s_{ij} = K₀ˢ · N_p(E_i − E_j; T_bath), shape ``(NE, NE)``.

    Phonon factor depends on sign of the energy transfer:

    * ``E_i > E_j`` (emission): ``1 + n_BE(E_i − E_j, T)``
    * ``E_i < E_j`` (absorption): ``n_BE(E_j − E_i, T)``

    Diagonal is zero. At T = 0, absorption vanishes and emission is 1.
    """
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    E_diff = E[:, None] - E[None, :]
    kBTp = _KB_UEV_PER_K * bath_temperature
    N_p = np.zeros_like(E_diff)
    if kBTp > 0:
        emission = E_diff > 0
        absorption = E_diff < 0
        exponent_em = np.minimum(E_diff[emission] / kBTp, 500.0)
        exponent_abs = np.minimum((-E_diff[absorption]) / kBTp, 500.0)
        N_p[emission] = 1.0 + 1.0 / (np.exp(exponent_em) - 1.0)
        N_p[absorption] = 1.0 / (np.exp(exponent_abs) - 1.0)
    else:
        N_p[E_diff > 0] = 1.0
        N_p[E_diff < 0] = 0.0
    return scattering_kernel_base(E, gap, tau_0, T_c) * N_p
