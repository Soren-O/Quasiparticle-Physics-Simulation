r"""Marchegiani 2025 rate-equation coefficient integrals (SI Notes III + IV).

Builds an :class:`M25Coefficients` bundle (consumed by
:func:`qpsim.services.rate_equation.solve_rate_equation_steady_state`)
from a primitive set of physical parameters: electrode gaps, qubit
frequency, transmon E_J/E_C, temperature, the junction tunneling scale,
the caption-level recombination rates, and the photon-driven inputs.

Implements:

* 12 tunneling rates :math:`\widetilde\Gamma^\alpha_{ij}` —
  SI Eqs. (S30)–(S36), with the single-junction-transmon matrix-element
  approximations (S25)–(S28) applied: ``s_{ii} = c_{10} = 0``,
  ``c_{ii} = 1`` (leading order in ``E_J/E_C``),
  ``s_{10} = (E_C/(8 E_J))^{1/4}``.
* Recombination ``r^L, r^{R<}, r^{R>}, r^{<>}`` — SI Note IV C leading-
  order, with ``r^{R>} = r^{<>} = r^{R<}`` (symmetric-gap limit at
  leading order in ``δ = Δ_R/Δ_L``).
* Thermal-phonon generation ``g^{pn}_L, g^{pn}_{R<}, g^{pn}_{R>}`` —
  thesis Appendix A via detailed balance at
  ``µ_α = 0`` (closed form).
* Intraband relaxation ``τ_R^{-1}, τ_E^{-1}`` — SI (S50) leading
  order in ``T/ω_LR`` and ``ω_LR/Δ_R``; detailed balance for ``τ_E``.
* Branching fraction ``ξ`` — SI (S37) single-junction reduction.

Photon-driven pieces (``g^{ph}_α`` and ``Γ̃^{ph}_{ij}``) are
**primitive inputs** to :class:`M25PhysicalParameters`: the SI Note V
photon-spectral-density integrals (S55–S59) are out of scope for this
pass. Callers can either hard-code caption values (``Γ^{ph}_{00} =
300`` Hz for the Fig 3 parameter set) or supply externally-computed
numbers.

Energy convention: all energies (``Δ``, ``ω_10``, ``T``, ``E_J``,
``E_C``) are in **Kelvin**. All rates are in **Hz**. The ``y``
variables that appear in Bessel-function arguments are dimensionless
(ratios of Kelvin quantities).

Source: Marchegiani & Catelani, *Commun. Phys.* **8**, 120 (2025),
Supplementary Information — see ``docs/M25_coefficient_integrals.md``
for the transcribed equation-by-equation reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, erfc, k0, k1

from qpsim.services.rate_equation import M25Coefficients

# ─────────────────────────────────────────────────────────────────────
#  Input bundle
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M25PhysicalParameters:
    r"""Primitive physical parameters for the M25 rate-equation closure.

    All energy-like quantities are in **Kelvin**; all rate-like
    quantities are in **Hz**. See ``docs/M25_coefficient_integrals.md``
    for the physical meaning of each field and the conversion from
    common ``/h`` or ``/(2π)`` paper conventions.

    Parameters
    ----------
    Delta_L_kelvin, Delta_R_kelvin
        Electrode superconducting gaps. Must satisfy
        ``Delta_L_kelvin > Delta_R_kelvin > 0`` (gap-asymmetric
        convention: L is the high-gap electrode; equal gaps are
        rejected because the Strategy B formulas are singular at
        ``ω_LR = 0``).
    omega_10_kelvin
        Qubit transition frequency ``ω_10``. Equivalent to
        ``h f_10 / k_B`` if ``f_10`` is given in Hz.
    T_kelvin
        Bath temperature.
    E_J_kelvin, E_C_kelvin
        Transmon Josephson and charging energies. Must satisfy
        ``E_J_kelvin > E_C_kelvin > 0`` (transmon regime).
    R_T_Hz
        Characteristic junction tunneling rate
        ``R_T = g_T Δ̄ / e²`` (SI Eq. S11 prefactor), where ``g_T``
        is the normal-state conductance and ``Δ̄ = (Δ_L+Δ_R)/2``.
        For a transmon with given ``E_J`` and ``Δ_L``, the
        Ambegaokar-Baratoff relation ``E_J = (h/(8 e² R_N)) Δ_L``
        fixes ``g_T = 1/R_N``, hence
        ``R_T = (8 E_J / h) (Δ̄/Δ_L)`` (all in consistent units).
    r_L_Hz
        Left-electrode recombination coefficient ``r^L`` (Hz).
        Caption value for M25 Fig 3: 6.25 MHz.
    r_Rlt_Hz
        Right-electrode trapped-band recombination coefficient
        ``r^{R<}`` (Hz). Caption value for M25 Fig 3: 6.25 MHz.
        ``r^{R>}`` and ``r^{<>}`` are set equal to ``r^{R<}`` at
        leading order in ``δ → 1`` per SI Note IV C.
    g_ph_L_Hz, g_ph_Rlt_Hz, g_ph_Rgt_Hz
        Photon-assisted quasiparticle generation rates (Hz) in the
        L, R<, R> sub-bands. Zero for thermal-only drive.
    Gamma_ph_00_Hz, Gamma_ph_01_Hz, Gamma_ph_10_Hz, Gamma_ph_11_Hz
        Photon-assisted parity-changing qubit transition rates (Hz).
        At M25 Fig 3 parameters: ``Γ^{ph}_{00} = Γ^{ph}_{11} ≈
        300`` Hz with the relaxation/excitation entries exp-
        suppressed by the transmon matrix-element ratio.
    Gamma_ee_10_Hz
        Parity-preserving relaxation rate. Detailed balance at
        temperature ``T`` fixes
        ``Γ^{ee}_{01} = Γ^{ee}_{10} e^{-ω_10/T}`` — this conversion
        is done inside :func:`coefficients_from_physical_parameters`.
        Caption value for M25 Fig 3: 100 kHz.

    Raises
    ------
    ValueError
        If the gap or transmon inequalities are violated, or any
        nonneg rate is negative.
    """

    Delta_L_kelvin: float
    Delta_R_kelvin: float
    omega_10_kelvin: float
    T_kelvin: float
    E_J_kelvin: float
    E_C_kelvin: float
    R_T_Hz: float
    r_L_Hz: float
    r_Rlt_Hz: float
    g_ph_L_Hz: float = 0.0
    g_ph_Rlt_Hz: float = 0.0
    g_ph_Rgt_Hz: float = 0.0
    Gamma_ph_00_Hz: float = 0.0
    Gamma_ph_01_Hz: float = 0.0
    Gamma_ph_10_Hz: float = 0.0
    Gamma_ph_11_Hz: float = 0.0
    Gamma_ee_10_Hz: float = 0.0

    def __post_init__(self) -> None:
        if not (self.Delta_L_kelvin > self.Delta_R_kelvin > 0.0):
            raise ValueError(
                f"Require Delta_L_kelvin > Delta_R_kelvin > 0 (gap-asymmetric "
                f"junction); the Strategy B formulas diverge at ω_LR = 0 "
                f"(Bessel K_1(y)→∞, erf(0)=0). Got Delta_L={self.Delta_L_kelvin}, "
                f"Delta_R={self.Delta_R_kelvin}."
            )
        if self.omega_10_kelvin <= 0.0:
            raise ValueError(f"omega_10_kelvin must be positive; got {self.omega_10_kelvin}")
        if self.T_kelvin <= 0.0:
            raise ValueError(f"T_kelvin must be positive; got {self.T_kelvin}")
        if not (self.E_J_kelvin > self.E_C_kelvin > 0.0):
            raise ValueError(
                f"Require E_J_kelvin > E_C_kelvin > 0 (transmon regime); "
                f"got E_J={self.E_J_kelvin}, E_C={self.E_C_kelvin}"
            )
        for name in (
            "R_T_Hz", "r_L_Hz", "r_Rlt_Hz",
            "g_ph_L_Hz", "g_ph_Rlt_Hz", "g_ph_Rgt_Hz",
            "Gamma_ph_00_Hz", "Gamma_ph_01_Hz",
            "Gamma_ph_10_Hz", "Gamma_ph_11_Hz",
            "Gamma_ee_10_Hz",
        ):
            val = float(getattr(self, name))
            if val < 0.0:
                raise ValueError(f"{name} must be nonnegative; got {val}")

    # ── Convenience accessors (all in Kelvin unless tagged) ──────────

    @property
    def omega_LR_kelvin(self) -> float:
        """Gap asymmetry ``ω_LR = Δ_L − Δ_R``."""
        return self.Delta_L_kelvin - self.Delta_R_kelvin

    @property
    def Delta_bar_kelvin(self) -> float:
        """Mean gap ``Δ̄ = (Δ_L + Δ_R)/2``."""
        return 0.5 * (self.Delta_L_kelvin + self.Delta_R_kelvin)

    @property
    def delta(self) -> float:
        """Gap ratio ``δ = Δ_R / Δ_L ∈ (0, 1]``."""
        return self.Delta_R_kelvin / self.Delta_L_kelvin


# ─────────────────────────────────────────────────────────────────────
#  Transmon matrix elements (SI Eqs. S25–S28)
# ─────────────────────────────────────────────────────────────────────


def _s_10_squared(E_J_kelvin: float, E_C_kelvin: float) -> float:
    r""":math:`s_{10}^2 = \sqrt{E_C/(8 E_J)}` (SI Eq. S25)."""
    return float(np.sqrt(E_C_kelvin / (8.0 * E_J_kelvin)))


def _c_ii_squared(
    i: int, E_J_kelvin: float, E_C_kelvin: float
) -> float:
    r"""Return :math:`c_{ii}^2` per SI Eq. S27 with ``i ∈ {0, 1}``.

    At leading order in ``E_J/E_C ≫ 1`` this is independent of ``i`` and
    equals 1; we carry the next-to-leading correction.
    """
    x = np.sqrt(E_C_kelvin / (8.0 * E_J_kelvin))
    c = 1.0 - (i + 0.5) * x - 1.5 * (i + 0.25) * x**2
    return float(c * c)


# ─────────────────────────────────────────────────────────────────────
#  Lower incomplete modified Bessel function K_n(z, w)  (SI Eq. S19)
# ─────────────────────────────────────────────────────────────────────


def _K_incomplete(n: int, z: float, w: float) -> float:
    r"""Return :math:`K_n(z, w) = \int_w^{\infty} e^{-z \cosh t} \cosh(n t)\, dt`.

    The integrand decays super-exponentially in ``t``; we integrate up
    to ``t_max`` chosen so that ``z cosh(t_max) > 700``, which puts
    ``e^{-z cosh(t)}`` below float64 underflow. Below the cutoff we
    guard against intermediate overflow in ``cosh(n t)`` by computing
    the integrand in log-space.
    """
    if z <= 0.0:
        raise ValueError(f"z must be positive; got {z}")
    if w < 0.0:
        raise ValueError(f"w must be nonneg; got {w}")

    t_cutoff = float(np.arccosh(max(2.0, 700.0 / z)))

    def integrand(t: float) -> float:
        # log(cosh(n t)) = |n t| + log((1 + e^{-2|n t|})/2)  (stable for large t)
        log_cosh = abs(n * t) + np.log1p(np.exp(-2.0 * abs(n * t))) - np.log(2.0)
        log_val = -z * np.cosh(t) + log_cosh
        return float(np.exp(log_val)) if log_val > -700.0 else 0.0

    if w >= t_cutoff:
        return 0.0
    val, _ = quad(integrand, w, t_cutoff, limit=200)
    return float(val)


# ─────────────────────────────────────────────────────────────────────
#  Tunneling rates (SI Eqs. S30–S36)
# ─────────────────────────────────────────────────────────────────────
#
#  All 12 rates share the prefactor R_T = g_T Δ̄ / e² (in Hz). The
#  closed forms below are written as
#
#      Γ̃^α_{ij} = R_T × (dimensionless algebraic factor)
#
#  per SI (S30)–(S36) with the single-junction-transmon reductions
#  ``s_{ii} = c_{10} = 0`` imposed upstream.
#
#  ``y   = ω_LR / (2T)``
#  ``y_± = (ω_10 ± ω_LR) / (2T)``.


def _gamma_L_ii(params: M25PhysicalParameters, i: int) -> float:
    r""":math:`\widetilde\Gamma^L_{ii}` per SI Eq. S30."""
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    Delta_L = params.Delta_L_kelvin
    Delta_R = params.Delta_R_kelvin
    y = omega_LR / (2.0 * T)
    c_ii2 = _c_ii_squared(i, params.E_J_kelvin, params.E_C_kelvin)
    # R_T × c²_{ii} × sqrt((Δ_L-Δ_R)/(2Δ_L)) × sqrt(2y/π) × eʸ × K_1(y)
    # Note: SI S30 has prefactor (g_T Δ_L / e²) not (g_T Δ̄ / e²); we
    # absorb the Δ_L/Δ̄ ratio into the algebraic factor.
    prefactor_ratio = Delta_L / params.Delta_bar_kelvin
    return (
        params.R_T_Hz
        * c_ii2
        * prefactor_ratio
        * np.sqrt((Delta_L - Delta_R) / (2.0 * Delta_L))
        * np.sqrt(2.0 * y / np.pi)
        * np.exp(y)
        * k1(y)
    )


def _gamma_Rgt_ii(params: M25PhysicalParameters, i: int) -> float:
    r""":math:`\widetilde\Gamma^{R>}_{ii}` per SI Eq. S31."""
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    Delta_L = params.Delta_L_kelvin
    Delta_R = params.Delta_R_kelvin
    y = omega_LR / (2.0 * T)
    c_ii2 = _c_ii_squared(i, params.E_J_kelvin, params.E_C_kelvin)
    prefactor_ratio = Delta_R / params.Delta_bar_kelvin
    # R_T × c²_{ii} × (Δ_R/Δ̄) × (Δ_L-Δ_R)/sqrt(2TΔ_R/π) × e⁻ʸ K_1(y)/(π erfc(√(2y)))
    denom_erfc = erfc(np.sqrt(2.0 * y))
    if denom_erfc <= 0.0:
        raise RuntimeError(
            f"erfc(sqrt(2y)) underflowed at y={y}; "
            "caller should clamp T_kelvin from below."
        )
    return (
        params.R_T_Hz
        * c_ii2
        * prefactor_ratio
        * (Delta_L - Delta_R)
        / np.sqrt(2.0 * T * Delta_R / np.pi)
        * np.exp(-y)
        * k1(y)
        / (np.pi * denom_erfc)
    )


def _gamma_L_10(params: M25PhysicalParameters) -> float:
    r""":math:`\widetilde\Gamma^L_{10}` per SI Eq. S32 (``s²_{10}`` channel)."""
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    Delta_L = params.Delta_L_kelvin
    y_plus = (omega_10 + omega_LR) / (2.0 * T)
    s_10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
    return (
        params.R_T_Hz
        * s_10_sq
        * np.sqrt(2.0 * Delta_L / (omega_LR + omega_10))
        * np.sqrt(2.0 * y_plus / np.pi)
        * np.exp(y_plus)
        * k0(y_plus)
    )


def _gamma_Rgt_10(params: M25PhysicalParameters) -> float:
    r""":math:`\widetilde\Gamma^{R>}_{10}` per SI Eq. S33."""
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    Delta_R = params.Delta_R_kelvin
    y_minus = (omega_10 - omega_LR) / (2.0 * T)
    abs_y_minus = abs(y_minus)
    diff = abs(omega_10 - omega_LR)
    if diff == 0.0:
        raise RuntimeError("ω_10 = ω_LR resonance not handled; shift by ε.")
    s_10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
    w = np.arccosh((omega_10 + omega_LR) / diff)
    k0_incomplete = _K_incomplete(0, abs_y_minus, w)
    denom_erfc = erfc(np.sqrt(omega_LR / T))
    if denom_erfc <= 0.0:
        raise RuntimeError(f"erfc(sqrt(ω_LR/T)) underflowed at T={T}.")
    return (
        params.R_T_Hz
        * s_10_sq
        * np.sqrt(2.0 * Delta_R / diff)
        * np.sqrt(2.0 * abs_y_minus / np.pi)
        * np.exp(y_minus)
        * k0_incomplete
        / denom_erfc
    )


def _gamma_Rlt_10(params: M25PhysicalParameters) -> float:
    r""":math:`\widetilde\Gamma^{R<}_{10}` per SI Eq. S34.

    Finite for ``ω_10 > ω_LR`` (case I); exp-suppressed
    ``∝ e^{-(ω_LR - ω_10)/T}`` for ``ω_10 < ω_LR`` (case II).
    """
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    Delta_R = params.Delta_R_kelvin
    y_minus = (omega_10 - omega_LR) / (2.0 * T)
    abs_y_minus = abs(y_minus)
    diff = abs(omega_LR - omega_10)
    if diff == 0.0:
        raise RuntimeError("ω_10 = ω_LR resonance not handled; shift by ε.")
    s_10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
    w = np.arccosh((omega_10 + omega_LR) / diff)
    k0_full = float(k0(abs_y_minus))
    k0_incomplete = _K_incomplete(0, abs_y_minus, w)
    denom_erf = erf(np.sqrt(omega_LR / T))
    if denom_erf <= 0.0:
        raise RuntimeError(f"erf(sqrt(ω_LR/T)) vanished at T={T}.")
    return (
        params.R_T_Hz
        * s_10_sq
        * np.sqrt(2.0 * Delta_R / diff)
        * np.sqrt(2.0 * abs_y_minus / np.pi)
        * np.exp(y_minus)
        * (k0_full - k0_incomplete)
        / denom_erf
    )


def _gamma_L_01(params: M25PhysicalParameters) -> float:
    r""":math:`\widetilde\Gamma^L_{01}` per SI Eq. S35.

    Finite for ``ω_10 < ω_LR`` (case II); exp-suppressed
    ``∝ e^{-(ω_10 - ω_LR)/T}`` for ``ω_10 > ω_LR`` (case I) via the
    ``exp(-y_-)`` factor.
    """
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    Delta_L = params.Delta_L_kelvin
    y_minus = (omega_10 - omega_LR) / (2.0 * T)
    abs_y_minus = abs(y_minus)
    diff = abs(omega_LR - omega_10)
    if diff == 0.0:
        raise RuntimeError("ω_10 = ω_LR resonance not handled; shift by ε.")
    s_10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
    return (
        params.R_T_Hz
        * s_10_sq
        * np.sqrt(2.0 * Delta_L / diff)
        * np.sqrt(2.0 * abs_y_minus / np.pi)
        * np.exp(-y_minus)
        * float(k0(abs_y_minus))
    )


def _gamma_Rgt_01(params: M25PhysicalParameters) -> float:
    r""":math:`\widetilde\Gamma^{R>}_{01}` per SI Eq. S36.

    Always ``∝ e^{-ω_10/T}`` — typically negligible at low T.
    """
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    Delta_R = params.Delta_R_kelvin
    y_plus = (omega_10 + omega_LR) / (2.0 * T)
    s_10_sq = _s_10_squared(params.E_J_kelvin, params.E_C_kelvin)
    denom_erfc = erfc(np.sqrt(omega_LR / T))
    if denom_erfc <= 0.0:
        raise RuntimeError(f"erfc(sqrt(ω_LR/T)) underflowed at T={T}.")
    return (
        params.R_T_Hz
        * s_10_sq
        * np.sqrt(2.0 * Delta_R / (omega_10 + omega_LR))
        * np.sqrt(2.0 * y_plus / np.pi)
        * np.exp(-y_plus)
        * float(k0(y_plus))
        / denom_erfc
    )


# ─────────────────────────────────────────────────────────────────────
#  Branching fraction ξ (SI Eq. S37 single-junction reduction)
# ─────────────────────────────────────────────────────────────────────


def _branching_fraction(params: M25PhysicalParameters) -> float:
    r""":math:`\xi = K_0(z, w) / K_0(z)` (single-junction transmon limit)."""
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    omega_10 = params.omega_10_kelvin
    diff = abs(omega_10 - omega_LR)
    if diff == 0.0:
        raise RuntimeError("ω_10 = ω_LR resonance not handled; shift by ε.")
    z = diff / (2.0 * T)
    w = np.arccosh((omega_10 + omega_LR) / diff)
    return _K_incomplete(0, z, w) / float(k0(z))


# ─────────────────────────────────────────────────────────────────────
#  Intraband relaxation τ_R⁻¹, τ_E⁻¹ (SI S50 leading order)
# ─────────────────────────────────────────────────────────────────────


def _tau_R_inverse(params: M25PhysicalParameters) -> float:
    r""":math:`\tau_R^{-1}` per SI Eq. S50 leading order in ``T/ω_LR, ω_LR/Δ_R``.

    Uses the relation ``2π b_R Δ_R³ = r^{R<}/4`` to eliminate ``b_R``
    in favor of the caption input ``r^{R<}``.
    """
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    Delta_R = params.Delta_R_kelvin
    two_pi_bR_DeltaR_cubed = params.r_Rlt_Hz / 4.0
    ratio = omega_LR / Delta_R
    correction = 1.0 + 3.5 * (T / omega_LR) + 7.0 * (T / omega_LR) ** 2
    return float(
        two_pi_bR_DeltaR_cubed
        * (64.0 * np.sqrt(2.0) / 105.0)
        * ratio ** 3.5
        * correction
    )


def _tau_E_inverse(params: M25PhysicalParameters) -> float:
    r""":math:`\tau_E^{-1}` via detailed balance against :math:`\tau_R^{-1}`.

    The physical detailed balance ``τ_E⁻¹ x_{R<}^{eq} = τ_R⁻¹ x_{R>}^{eq}``
    at ``µ_α = 0`` gives

    .. math::

        \tau_E^{-1} / \tau_R^{-1} = x_{R>}^{eq} / x_{R<}^{eq}
                                  = \mathrm{erfc}(\sqrt{ω_LR/T}) / \mathrm{erf}(\sqrt{ω_LR/T}).

    The ``exp(-ω_LR/T)`` suppression the thesis appendix quotes in the
    low-T limit is **already inside** ``erfc(√(ω_LR/T))`` asymptotically
    (``erfc(z) ≃ e^{-z²}/(z√π)`` for ``z ≫ 1``), so we do not multiply
    by it again.
    """
    T = params.T_kelvin
    omega_LR = params.omega_LR_kelvin
    sqrt_ratio = np.sqrt(omega_LR / T)
    balance_factor = erfc(sqrt_ratio) / erf(sqrt_ratio)
    return float(_tau_R_inverse(params) * balance_factor)


# ─────────────────────────────────────────────────────────────────────
#  Thermal-phonon generation g^{pn}_α
# ─────────────────────────────────────────────────────────────────────
#
#  At full thermal equilibrium (µ_α = 0), generation balances
#  recombination. For the left electrode (single sub-band),
#  ``g^{pn}_L = r^L × (x_L^{eq})²``. For the right electrode, each
#  pair-breaking event creates two quasiparticles that partition
#  independently between R< and R>; summing the contributions of
#  both-in-<, one-in-each, and both-in-> events gives
#
#      g^{pn}_{R<} = r × x_{R<}^{eq} × x_R^{eq}     (first power of erf)
#      g^{pn}_{R>} = r × x_{R>}^{eq} × x_R^{eq}     (first power of erfc)
#
#  with x_R^{eq} = x_{R<}^{eq} + x_{R>}^{eq} the full R-electrode
#  equilibrium density. The sum
#  ``g^{pn}_{R<} + g^{pn}_{R>} = r × (x_R^{eq})² = r × (2πT/Δ_R) e^{-2Δ_R/T}``
#  collapses back to the un-partitioned Eq. 8 prefactor, which is the
#  invariant the Lambert-W crossover observable depends on.
#
#  Eq. S2/S4/S5 closed forms for the equilibrium densities:
#    x_L^{eq}    = sqrt(2πT/Δ_L) e^{-Δ_L/T}
#    x_{R<}^{eq} = sqrt(2πT/Δ_R) e^{-Δ_R/T} erf(sqrt(ω_LR/T))
#    x_{R>}^{eq} = sqrt(2πT/Δ_R) e^{-Δ_R/T} erfc(sqrt(ω_LR/T))


def _g_pn_L(params: M25PhysicalParameters) -> float:
    T = params.T_kelvin
    Delta_L = params.Delta_L_kelvin
    x_eq_sq = (2.0 * np.pi * T / Delta_L) * np.exp(-2.0 * Delta_L / T)
    return float(params.r_L_Hz * x_eq_sq)


def _g_pn_Rlt(params: M25PhysicalParameters) -> float:
    T = params.T_kelvin
    Delta_R = params.Delta_R_kelvin
    omega_LR = params.omega_LR_kelvin
    # r × x_R<^eq × x_R^eq = r × (2πT/Δ_R) e^{-2Δ_R/T} × erf(√(ω_LR/T))
    # (first power; erf(z) + erfc(z) = 1 makes g_R< + g_R> = r × x_R²).
    return float(
        params.r_Rlt_Hz
        * (2.0 * np.pi * T / Delta_R)
        * np.exp(-2.0 * Delta_R / T)
        * erf(np.sqrt(omega_LR / T))
    )


def _g_pn_Rgt(params: M25PhysicalParameters) -> float:
    T = params.T_kelvin
    Delta_R = params.Delta_R_kelvin
    omega_LR = params.omega_LR_kelvin
    # r × x_R>^eq × x_R^eq = r × (2πT/Δ_R) e^{-2Δ_R/T} × erfc(√(ω_LR/T))
    # r^{R>} ≃ r^{R<} at leading order δ → 1 (SI Note IV C).
    return float(
        params.r_Rlt_Hz
        * (2.0 * np.pi * T / Delta_R)
        * np.exp(-2.0 * Delta_R / T)
        * erfc(np.sqrt(omega_LR / T))
    )


# ─────────────────────────────────────────────────────────────────────
#  Public assembler
# ─────────────────────────────────────────────────────────────────────


def coefficients_from_physical_parameters(
    params: M25PhysicalParameters,
) -> M25Coefficients:
    r"""Evaluate every rate coefficient and return the packed bundle.

    Parameters
    ----------
    params
        Primitive physical parameters; see :class:`M25PhysicalParameters`.

    Returns
    -------
    M25Coefficients
        The frozen bundle consumed by
        :func:`qpsim.services.rate_equation.solve_rate_equation_steady_state`.

    Notes
    -----
    ``Γ̃^{R<}_{01} = Γ̃^{R<}_{11} = Γ̃^{R<}_{00} = 0`` by kinematic
    constraint (M25 low-T ansatz). Only ``Γ̃^{R<}_{10}`` survives.

    The parity-preserving rates satisfy detailed balance at the bath
    temperature:
    ``Γ̃^{ee}_{01} = Γ̃^{ee}_{10} exp(-ω_10/T)``.
    """
    # ── Tunneling rates ──────────────────────────────────────────────
    gammas_L = np.array(
        [
            [_gamma_L_ii(params, 0), _gamma_L_01(params)],
            [_gamma_L_10(params),    _gamma_L_ii(params, 1)],
        ],
        dtype=float,
    )
    gammas_Rgt = np.array(
        [
            [_gamma_Rgt_ii(params, 0), _gamma_Rgt_01(params)],
            [_gamma_Rgt_10(params),    _gamma_Rgt_ii(params, 1)],
        ],
        dtype=float,
    )
    gammas_Rlt = np.zeros((2, 2), dtype=float)
    gammas_Rlt[1, 0] = _gamma_Rlt_10(params)

    # ── Parity-preserving (detailed balance) ─────────────────────────
    T = params.T_kelvin
    omega_10 = params.omega_10_kelvin
    Gamma_ee_01 = params.Gamma_ee_10_Hz * float(np.exp(-omega_10 / T))
    gamma_ee = np.array(
        [[0.0, Gamma_ee_01], [params.Gamma_ee_10_Hz, 0.0]],
        dtype=float,
    )

    # ── Photon-assisted tunneling (primitive inputs; Note V deferred) ─
    gamma_ph = np.array(
        [
            [params.Gamma_ph_00_Hz, params.Gamma_ph_01_Hz],
            [params.Gamma_ph_10_Hz, params.Gamma_ph_11_Hz],
        ],
        dtype=float,
    )

    # ── Recombination ────────────────────────────────────────────────
    r_L = params.r_L_Hz
    r_Rgt = params.r_Rlt_Hz        # SI IV C leading order δ → 1
    r_Rlt = params.r_Rlt_Hz
    r_cross = params.r_Rlt_Hz      # same

    # ── Generation (thermal + photon) ────────────────────────────────
    g_L = _g_pn_L(params) + params.g_ph_L_Hz
    g_Rgt = _g_pn_Rgt(params) + params.g_ph_Rgt_Hz
    g_Rlt = _g_pn_Rlt(params) + params.g_ph_Rlt_Hz

    # ── Intraband + branching ────────────────────────────────────────
    tau_R_inv = _tau_R_inverse(params)
    tau_E_inv = _tau_E_inverse(params)
    xi = _branching_fraction(params)

    return M25Coefficients(
        gammas_L=gammas_L,
        gammas_Rgt=gammas_Rgt,
        gammas_Rlt=gammas_Rlt,
        gamma_ee=gamma_ee,
        gamma_ph=gamma_ph,
        r_L=r_L,
        r_Rgt=r_Rgt,
        r_Rlt=r_Rlt,
        r_cross=r_cross,
        g_L=g_L,
        g_Rgt=g_Rgt,
        g_Rlt=g_Rlt,
        tau_R_inv=tau_R_inv,
        tau_E_inv=tau_E_inv,
        xi=xi,
        delta=params.delta,
    )
