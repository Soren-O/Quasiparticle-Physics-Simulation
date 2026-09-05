"""K₀ˢ / K₀ʳ electron–phonon scattering and recombination kernels.

Kernel definitions (phonon-energy convention, thesis Ch. 2):

    K₀ʳ(E_i, E_j) = (1/τ₀) · (E_i + E_j)² / (kB T_c)³ · K⁺(E_i, E_j)
    K₀ˢ(E_i, E_j) = (1/τ₀) · (E_i − E_j)² / (kB T_c)³ · K⁻(E_i, E_j)

The "with-phonon" variants (``recombination_kernel``,
``scattering_kernel``) multiply by the Bose–Einstein phonon-occupation
factor N_p for a thermal phonon bath.

The implementation is pure numpy. ``thermal_qp_weights`` lives in
``qpsim.physics.spectral``.
"""

from __future__ import annotations

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.physics.spectral import (
    _energy_over_kbt,
    coherence_factor_minus,
    coherence_factor_plus,
)


def _finite_energy_grid(E_bins: np.ndarray) -> np.ndarray:
    """Return a one-dimensional, non-empty, finite floating energy grid."""
    raw = np.asarray(E_bins)
    if np.iscomplexobj(raw):
        raise ValueError("E_bins must be real-valued.")
    try:
        E = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("E_bins must be a real numeric array.") from exc
    if E.ndim != 1:
        raise ValueError("E_bins must be a 1D array.")
    if E.size == 0:
        raise ValueError("E_bins must be non-empty.")
    if np.any(~np.isfinite(E)):
        raise ValueError("E_bins must contain only finite values.")
    return E


def _finite_scalar(name: str, value: float) -> float:
    """Return ``value`` as a float, rejecting NaN and infinity."""
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result}.")
    return result


def _non_negative_scalar(name: str, value: float) -> float:
    """Return a finite scalar satisfying a non-negative contract."""
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative; got {result}.")
    return result


def _positive_scalar(name: str, value: float) -> float:
    """Return a finite scalar satisfying a strictly-positive contract."""
    result = _finite_scalar(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive; got {result}.")
    return result


def _bose_occupation_from_exponent(exponent: np.ndarray) -> np.ndarray:
    """Evaluate ``1 / (exp(x) - 1)`` stably for non-negative ``x``.

    The exact zero-frequency mode is assigned zero occupation because it is a
    decoupled, zero-transfer bookkeeping bin in qpsim's discrete phonon grid.
    Positive exponents are evaluated as ``exp(-x) / -expm1(-x)``: ``expm1``
    resolves arbitrarily small transfers, while the negative-exponent form
    preserves representable occupations through the subnormal range instead
    of overflowing at ``exp(x)``.  If the mathematical occupation exceeds
    binary64 range, the result saturates at the largest finite float rather
    than introducing an infinity into downstream products.  ``x = +inf`` is
    the exact limiting zero occupation.
    """
    raw = np.asarray(exponent)
    if np.iscomplexobj(raw):
        raise ValueError("Bose exponents must be real-valued.")
    x = np.asarray(raw, dtype=float)
    if np.any(np.isnan(x)) or np.any(x < 0.0):
        raise ValueError("Bose exponents must be non-negative and not NaN.")

    occupation = np.zeros_like(x)
    positive = x > 0.0
    if not np.any(positive):
        return occupation

    positive_x = x[positive]
    with np.errstate(divide="ignore", over="ignore", under="ignore"):
        exp_negative = np.exp(-positive_x)
        values = exp_negative / (-np.expm1(-positive_x))
    occupation[positive] = np.minimum(values, np.finfo(float).max)
    return occupation


def _thermal_bose_occupation(
    energy_transfer: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Return a stable Bose occupation for non-negative energy transfers."""
    raw_energy = np.asarray(energy_transfer)
    if np.iscomplexobj(raw_energy):
        raise ValueError("Bose energy transfers must be real-valued.")
    energy = np.asarray(raw_energy, dtype=float)
    if np.any(~np.isfinite(energy)) or np.any(energy < 0.0):
        raise ValueError("Bose energy transfers must be finite and non-negative.")
    temperature_value = _positive_scalar("temperature", temperature)

    # A positive ratio that mathematically underflows to zero represents an
    # occupation beyond binary64 range. Feed the smallest positive exponent
    # to the saturating evaluator rather than confusing it with the exact
    # zero-transfer bookkeeping mode.
    exponent = _energy_over_kbt(energy, temperature_value)
    underflowed = (energy > 0.0) & (exponent == 0.0)
    if np.any(underflowed):
        exponent = exponent.copy()
        exponent[underflowed] = np.nextafter(0.0, 1.0)
    return _bose_occupation_from_exponent(exponent)


def thermal_phonon_occupation(
    omega_bins: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Bose–Einstein thermal phonon occupation n_BE(ω, T) = 1/(exp(ω/kT) − 1).

    Returns zeros at ``temperature == 0``. The exact ``omega == 0`` bookkeeping
    mode is also zero because it carries no energy transfer. Finite positive
    values are returned for positive frequencies and temperatures throughout
    the representable binary64 range; physically underflowed occupations are
    returned as zero.
    """
    raw_omega = np.asarray(omega_bins)
    if np.iscomplexobj(raw_omega):
        raise ValueError("omega_bins must be real-valued.")
    try:
        omega = np.asarray(raw_omega, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("omega_bins must be a real numeric array.") from exc
    if omega.ndim != 1:
        raise ValueError("omega_bins must be a 1D array.")
    if np.any(~np.isfinite(omega)):
        raise ValueError("omega_bins must contain only finite values.")
    if np.any(omega < 0):
        raise ValueError("omega_bins must be non-negative.")
    temperature = _non_negative_scalar("temperature", temperature)
    if temperature == 0:
        return np.zeros_like(omega)

    return _thermal_bose_occupation(omega, temperature)


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
    E = _finite_energy_grid(E_bins)
    gap = _non_negative_scalar("gap", gap)
    tau_0 = _positive_scalar("tau_0", tau_0)
    T_c = _positive_scalar("T_c", T_c)
    kBTc = _KB_UEV_PER_K * T_c
    E_sum = E[:, None] + E[None, :]
    coh = (
        coherence_factor_plus(E, gap)
        if coherence_factor is None
        else np.asarray(coherence_factor, dtype=float)
    )
    expected_shape = (E.size, E.size)
    if coh.shape != expected_shape:
        raise ValueError(
            "coherence_factor must have shape "
            f"{expected_shape}; got {coh.shape}."
        )
    if np.any(~np.isfinite(coh)):
        raise ValueError("coherence_factor must contain only finite values.")
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

    THE (E_i − E_j)² IS NOT KINEMATIC. It is the Debye phonon spectral
    function α²F(ω) = b·ω², and τ₀ is *defined* by Kaplan (1976) under that
    assumption — it absorbs b together with the (kB T_c)³ scale. The two
    cannot be varied independently: a different phonon spectrum changes this
    frequency dependence AND the meaning of τ₀.

    This is also why the phonon-side kernel carries no ω² at all. The QP
    equation integrates OVER phonon modes and so carries D(ω) ∝ ω²; the phonon
    equation is written PER MODE and a single mode has no density of states.
    Their ratio is exactly ω² times a constant, which is a cheap check that the
    structure is intact.

    See ``docs/Phonon_Model_Decisions.md``, "The phonon spectrum is Debye" —
    the assumption went undocumented long enough to invite adding a per-bin
    phonon density of states, which would have double-counted it.

    The diagonal is zeroed (no transfer ⇒ no scattering). Pass
    ``coherence_factor`` to reuse a precomputed K matrix — e.g.
    ``ctx.K_minus`` for phonon convention, ``ctx.K_plus`` for photon.
    """
    E = _finite_energy_grid(E_bins)
    gap = _non_negative_scalar("gap", gap)
    tau_0 = _positive_scalar("tau_0", tau_0)
    T_c = _positive_scalar("T_c", T_c)
    kBTc = _KB_UEV_PER_K * T_c
    E_diff = E[:, None] - E[None, :]
    coh = (
        coherence_factor_minus(E, gap)
        if coherence_factor is None
        else np.asarray(coherence_factor, dtype=float)
    )
    expected_shape = (E.size, E.size)
    if coh.shape != expected_shape:
        raise ValueError(
            "coherence_factor must have shape "
            f"{expected_shape}; got {coh.shape}."
        )
    if np.any(~np.isfinite(coh)):
        raise ValueError("coherence_factor must contain only finite values.")
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
    E = _finite_energy_grid(E_bins)
    bath_temperature = _non_negative_scalar("bath_temperature", bath_temperature)
    K_r0 = recombination_kernel_base(E, gap, tau_0, T_c)
    E_sum = E[:, None] + E[None, :]
    if bath_temperature > 0:
        N_p = _thermal_bose_occupation(E_sum, bath_temperature) + 1.0
    else:
        N_p = np.ones_like(E_sum, dtype=float)
    return K_r0 * N_p


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
    E = _finite_energy_grid(E_bins)
    bath_temperature = _non_negative_scalar("bath_temperature", bath_temperature)
    K_s0 = scattering_kernel_base(E, gap, tau_0, T_c)
    E_diff = E[:, None] - E[None, :]
    N_p = np.zeros_like(E_diff)
    if bath_temperature > 0:
        emission = E_diff > 0
        absorption = E_diff < 0
        N_p[emission] = 1.0 + _thermal_bose_occupation(
            E_diff[emission], bath_temperature
        )
        N_p[absorption] = _thermal_bose_occupation(
            -E_diff[absorption], bath_temperature
        )
    else:
        N_p[E_diff > 0] = 1.0
        N_p[E_diff < 0] = 0.0
    return K_s0 * N_p
