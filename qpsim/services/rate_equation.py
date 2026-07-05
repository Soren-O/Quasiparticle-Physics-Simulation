r"""Marchegiani 2025 rate-equation closure — core observables and steady-state solver.

Three entry points, all named for the M25 equations they implement:

* :func:`crossover_temperature_kelvin` — closed-form Lambert-W
  crossover temperature ``T̄`` (M25 Eq. 8). Inputs: three scalars
  (``Δ_R``, ``r^{R<}``, ``g^ph_R``).
* :func:`solve_rate_equation_steady_state` — Newton fixed-point for
  the three-chemical-potential system (M25 Eqs. 3-6, derived in
  thesis Part III Appendix A). Inputs: all rate coefficients packed
  in an :class:`M25Coefficients` dataclass. Outputs: the four
  unknowns ``(p_1, x_L, x_{R>}, x_{R<})`` of the boxed system.
* :func:`solve_rate_equation_branch` — deterministic branch-continuation
  driver for temperature sweeps: natural-parameter continuation with
  warm starts and adaptive step refinement, a reduced 1-D bracketing
  fallback that survives folds, bidirectional (photon-branch up /
  thermal-branch down) tracking, and an explicit branch-exchange rule.
  This is the paper-figure (M25 Figs. 3-4) sweep entry point.

Both entry points treat rate coefficients as \"opaque inputs\" —
the energy integrals that produce them (tunneling-rate integrals in
M25 Supplementary Note III; recombination/generation/intraband-
relaxation integrals in Note IV) are not implemented here. A caller
that wants to reproduce M25 Figs. 3–5 assembles the coefficients
externally and calls this service to close the steady state.

Marchegiani & Catelani, *Commun. Phys.* **8**, 120 (2025).
"""

from __future__ import annotations

import itertools
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import brentq, least_squares, root
from scipy.special import erf, erfc, lambertw


def crossover_temperature_kelvin(
    *,
    Delta_R_kelvin: float,
    r_Rlt_rate_Hz: float,
    g_photon_R_rate_Hz: float,
) -> float:
    r"""Return ``T̄`` (Kelvin) from M25 Eq. 8.

    .. math::

        \bar T = \frac{2\,\Delta_R}{
            W\!\left( 4\pi \, r^{R<} / g^\mathrm{ph}_R \right)
        }

    Parameters
    ----------
    Delta_R_kelvin
        Right-electrode gap in Kelvin (``Δ_R / k_B``). For Al at the
        M25 Fig. 3 parameter set, ``Δ_R/h = 49 GHz`` ⇒ 2.35 K.
    r_Rlt_rate_Hz
        Trapped-band (R<) recombination-rate coefficient in Hz. The
        M25 Fig. 3 caption gives ``r^L = r^{R<} = 6.25 MHz``.
    g_photon_R_rate_Hz
        Photon-assisted pair-breaking generation rate on R in Hz.
        At the M25 Fig. 3 parameter set this is set by
        ``Γ_{01}^{ph} = 300 Hz`` plus Cooper-pair-count normalization.

    Returns
    -------
    float
        Crossover temperature ``T̄`` in Kelvin. The Lambert-W real
        principal branch ``W_0`` is used; for the physical parameter
        range ``r^{R<} / g^\mathrm{ph}_R ≫ 1`` this is guaranteed
        real.

    Raises
    ------
    ValueError
        For non-positive inputs.
    RuntimeError
        If the Lambert-W argument lands outside its real-valued
        domain ``[−1/e, ∞)``.
    """
    if Delta_R_kelvin <= 0:
        raise ValueError("Delta_R_kelvin must be positive.")
    if r_Rlt_rate_Hz <= 0:
        raise ValueError("r_Rlt_rate_Hz must be positive.")
    if g_photon_R_rate_Hz <= 0:
        raise ValueError("g_photon_R_rate_Hz must be positive.")

    arg = 4.0 * np.pi * r_Rlt_rate_Hz / g_photon_R_rate_Hz
    if arg < -1.0 / np.e:
        raise RuntimeError(
            f"Lambert-W argument {arg} outside the real-valued domain "
            "[-1/e, ∞). Check input signs."
        )
    w = complex(lambertw(arg, k=0))
    if abs(w.imag) > 1e-12 * max(abs(w.real), 1.0):
        raise RuntimeError(
            f"Lambert-W returned non-real value {w}; check inputs."
        )
    return 2.0 * Delta_R_kelvin / float(w.real)


# ─────────────────────────────────────────────────────────────────────
#  Full three-chemical-potential steady-state solver (M25 Eqs. 3-6).
#
#  State vector: y = (p_1, x_L, x_{R>}, x_{R<}) with p_0 = 1 - p_1.
#
#  Equations follow thesis Part III Appendix A (boxed result
#  \cref{res:M25_rate_eqs}). Each rate coefficient is an opaque
#  input; the solver itself is pure algebra on the boxed system.
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M25Coefficients:
    r"""Rate coefficients for the M25 three-chemical-potential system.

    All rates are in Hz (inverse time). Densities ``x_L``, ``x_{R>}``,
    ``x_{R<}`` are dimensionless (normalized to local-gap Cooper-pair
    number per M25 Eq. normalization). The steady-state residual is
    dimensionally ``(Hz, Hz, Hz, Hz)`` — all inputs must share the
    same time unit.

    Parameters
    ----------
    gammas_L, gammas_Rgt, gammas_Rlt
        Tunneling rates ``Γ̃^α_{ij}`` as ``(2, 2)`` arrays indexed
        ``[i, j]`` with ``i`` the initial and ``j`` the final qubit
        logical state. ``α ∈ {L, R>, R<}`` labels the initial
        quasiparticle sub-band. By the low-temperature ansatz in
        M25 text below Eq. 2, ``Γ̃^{R<}_{01} = Γ̃^{R<}_{11} = 0``;
        combined with the kinematic constraint ``Γ̃^{R<}_{00} = 0``,
        only ``gammas_Rlt[1, 0]`` (the R< → L relaxation channel
        with qubit 1→0) is physically nonzero. ``__post_init__``
        enforces this: nonzero entries in any other R< position are
        rejected, because they would drive the qubit master equation
        without a matching density-transport term. For R>, the paper
        states ``Γ̃^{R>}_{01} ∝ e^{-ω_10/T}`` is exponentially
        suppressed but not structurally zero; all four ``gammas_Rgt``
        entries are accepted.
    gamma_ee
        Parity-preserving tunneling rate ``Γ̃^{ee}_{ij}`` as a
        ``(2, 2)`` array. Detailed balance (M25 text above Eq. 4)
        fixes ``Γ̃^{ee}_{01} = e^{-ω_10/T}\, Γ̃^{ee}_{10}``; the
        solver does not enforce this — the user supplies both.
        Only ``[0, 1]`` and ``[1, 0]`` entries enter the qubit
        master equation.
    gamma_ph
        Bulk photon-assisted contribution ``Γ̃^{ph}_{ij}`` to
        ``Γ̃^{eo}_{ij}`` (M25, decomposition of Γ̃^{eo}). Not
        density-dependent. Shape ``(2, 2)``.
    r_L, r_Rgt, r_Rlt, r_cross
        Recombination coefficients ``r^L``, ``r^{R>}``, ``r^{R<}``,
        ``r^{<>}`` (Hz). The first three multiply ``x_α^2``; the
        last multiplies ``x_{R<}\, x_{R>}``.
    g_L, g_Rgt, g_Rlt
        Population-independent generation rates (Hz) in each sector.
        Typically ``g^{pn}_α`` (thermal-phonon pair-breaking) plus
        any photon-driven pieces that a caller has already averaged
        over qubit state. For population-dependent photon rates
        (M25 main text: ``g^{ph}_R = Σ_i p_i (Γ^{ph}_{ij} sums) /
        N_CP(R)``), supply the per-state arrays below instead (or
        in addition) and leave the ``g_α`` scalars at the thermal
        value only.
    g_ph_L_per_state, g_ph_Rgt_per_state, g_ph_Rlt_per_state
        Photon-assisted generation rates (Hz) as length-2 arrays
        ``[coefficient of p_0, coefficient of p_1]``. The residual
        adds ``p_0 · array[0] + p_1 · array[1]`` to the ``g_α``
        scalar. Defaults to all zeros (backward compatible —
        scalars alone reproduce the pre-Note-V behavior). See
        :func:`qpsim.services.rate_equation_coefficients.coefficients_from_physical_parameters_with_photon_drive`
        for the Note-V builder that populates these.
    tau_R_inv, tau_E_inv
        Intraband relaxation ``τ_R^{-1}`` (R> → R< spontaneous
        emission) and excitation ``τ_E^{-1}`` (R< → R> thermal
        phonon absorption) rates (Hz).
    xi
        Branching fraction ``ξ ∈ [0, 1]``: the probability that a
        quasiparticle tunneling from L via the ``Γ̃^L_{01}`` channel
        (qubit absorbs ``ω_10``) lands in the ``R>`` band rather
        than ``R<``. Exponentially suppressed at low temperature
        (M25, text below Eq. 6).
    delta
        Gap ratio ``δ = Δ_R / Δ_L ∈ (0, 1]``. Appears in ``ẋ_L``
        wherever a tunneling event connects the two electrodes
        (normalization conversion).
    cooper_pair_number_R
        Cooper-pair number of the low-gap electrode,
        ``N_CP(R) = 2 ν_0 Δ_R V`` (dimensionless count). The qubit
        master equation (M25 Eq. 3) uses the ensemble rates
        ``Γ̃^α_{ij}`` directly, but the density equations (M25
        Eqs. 4–6) use the **single-quasiparticle** rates
        ``Γ̄^α_{ij} = Γ̃^α_{ij} / N_CP(R)`` (M25 main text below
        Eq. 6: "These rates correspond to the tilde rates …
        divided by the Cooper pair number in the low-gap
        electrode"). The residual divides ``gammas_*`` by this
        number in the density equations only. Default ``1.0``
        keeps the legacy behavior ``Γ̄ = Γ̃`` for opaque-coefficient
        callers (toy/unit-test bundles); physical M25 runs must set
        it — the Note-V builder
        :func:`qpsim.services.rate_equation_coefficients.coefficients_from_physical_parameters_with_photon_drive`
        does so automatically. At the M25 Fig 3 parameter set
        ``N_CP(R) ≈ 1.6 × 10¹⁰``; omitting it inflates the density-
        equation tunneling terms by that factor, which was the root
        cause of the historical "flat-valley multi-stability"
        conditioning noise (residual mixing ~10¹⁰ Hz tunneling
        currents against ~10⁻⁸ Hz generation).

    Notes
    -----
    Every density-equation consumer must take the tunneling arrays
    through :meth:`density_gammas` (the Γ̄ triple), never the raw
    ``gammas_*`` attributes — those are the ensemble ``Γ̃`` rates and
    feed only the qubit channels (master equation Eq. 3 and the
    ``Γ̃^{eo}`` assembly). Open-coding the ``/ cooper_pair_number_R``
    division at call sites is how normalization bugs sneak in.

    Raises
    ------
    ValueError
        If any shape is wrong, ``xi`` is outside ``[0, 1]``,
        ``delta`` is outside ``(0, 1]``, ``cooper_pair_number_R``
        is not positive/finite, or a nonnegative rate is negative.
    """

    gammas_L: np.ndarray
    gammas_Rgt: np.ndarray
    gammas_Rlt: np.ndarray
    gamma_ee: np.ndarray
    gamma_ph: np.ndarray
    r_L: float
    r_Rgt: float
    r_Rlt: float
    r_cross: float
    g_L: float
    g_Rgt: float
    g_Rlt: float
    tau_R_inv: float
    tau_E_inv: float
    xi: float
    delta: float
    g_ph_L_per_state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    g_ph_Rgt_per_state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    g_ph_Rlt_per_state: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cooper_pair_number_R: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gammas_L", "gammas_Rgt", "gammas_Rlt", "gamma_ee", "gamma_ph"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.shape != (2, 2):
                raise ValueError(
                    f"{name} must be a numpy ndarray of shape (2, 2); "
                    f"got shape {getattr(arr, 'shape', None)}"
                )
            if np.any(arr < 0):
                raise ValueError(
                    f"{name} entries must be nonneg (rates); "
                    f"got min {float(arr.min())}"
                )
        for name in ("g_ph_L_per_state", "g_ph_Rgt_per_state", "g_ph_Rlt_per_state"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray) or arr.shape != (2,):
                raise ValueError(
                    f"{name} must be a numpy ndarray of shape (2,); "
                    f"got shape {getattr(arr, 'shape', None)}"
                )
            if np.any(arr < 0):
                raise ValueError(
                    f"{name} entries must be nonneg (rates); "
                    f"got min {float(arr.min())}"
                )
        # The boxed residual (M25 Eqs. 4-6) assumes the ansatz in M25
        # text below Eq. 2: Γ̃^{R<}_{01} = Γ̃^{R<}_{11} = 0 (trapped-
        # band QP cannot tunnel out without gaining ω_LR) and
        # Γ̃^{R<}_{00} = 0 kinematically. Only the [1, 0] entry (the
        # cross-electrode relaxation channel) enters the density
        # transport. If the caller passes nonzero values in the other
        # R< entries, they would feed the qubit master equation via
        # Γ̃^{eo}_{ij} (through _rate_equation_residual's gamma_eo
        # assembly) without a corresponding density-transport term —
        # an inconsistent state. Reject early.
        rlt = self.gammas_Rlt
        for i, j in ((0, 0), (0, 1), (1, 1)):
            if rlt[i, j] != 0.0:
                raise ValueError(
                    f"gammas_Rlt[{i}, {j}] must be 0 (M25 low-T ansatz: "
                    f"only Γ̃^{{R<}}_{{10}} survives for the trapped band); "
                    f"got {float(rlt[i, j])}."
                )
        if not (0.0 <= self.xi <= 1.0):
            raise ValueError(f"xi must lie in [0, 1]; got {self.xi}")
        if not (0.0 < self.delta <= 1.0):
            raise ValueError(f"delta must lie in (0, 1]; got {self.delta}")
        if not (np.isfinite(self.cooper_pair_number_R) and self.cooper_pair_number_R > 0.0):
            raise ValueError(
                f"cooper_pair_number_R must be positive and finite; "
                f"got {self.cooper_pair_number_R}"
            )
        for name in (
            "r_L", "r_Rgt", "r_Rlt", "r_cross",
            "g_L", "g_Rgt", "g_Rlt",
            "tau_R_inv", "tau_E_inv",
        ):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"{name} must be nonnegative; got {val}")

    def density_gammas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""Return the single-quasiparticle rates ``(Γ̄^L, Γ̄^{R>}, Γ̄^{R<})``.

        Each entry is the corresponding ensemble array divided by
        ``cooper_pair_number_R``:
        ``Γ̄^α_{ij} = Γ̃^α_{ij} / N_CP(R)`` (M25 text below Eq. 6).
        This is the single normalization chokepoint for the density
        equations (M25 Eqs. 4–6) — the residual, the SI analytic
        seeds, and the device-layer flux spreading all consume this
        triple, while the qubit channels keep the raw ``Γ̃`` arrays.
        """
        return (
            self.gammas_L / self.cooper_pair_number_R,
            self.gammas_Rgt / self.cooper_pair_number_R,
            self.gammas_Rlt / self.cooper_pair_number_R,
        )


@dataclass(frozen=True)
class M25SteadyState:
    """Steady-state solution of the M25 rate-equation system.

    Attributes
    ----------
    p_0, p_1
        Qubit logical-state probabilities, ``p_0 + p_1 = 1``.
    x_L, x_Rgt, x_Rlt
        Dimensionless quasiparticle densities in the L, R>, R<
        sub-bands (normalized to local-gap Cooper-pair number).
    residual_inf_norm
        ``||R(y)||_∞`` at convergence, in Hz. For a well-converged
        solve this should be below the ``residual_tol`` passed to
        :func:`solve_rate_equation_steady_state`.
    n_function_evaluations
        Total residual evaluations consumed by the Newton solver
        (function + finite-difference Jacobian columns).
    """

    p_0: float
    p_1: float
    x_L: float
    x_Rgt: float
    x_Rlt: float
    residual_inf_norm: float
    n_function_evaluations: int


def _rate_equation_residual(y: np.ndarray, coefs: M25Coefficients) -> np.ndarray:
    r"""Compute ``R(y) = (ṗ_1, ẋ_L, ẋ_{R>}, ẋ_{R<})`` at steady state.

    At the steady state ``R(y) = 0``. The equations reproduce M25
    Eqs. 3-6 (boxed in thesis Part III Appendix A,
    \cref{res:M25_rate_eqs}).
    """
    p_1, x_L, x_Rgt, x_Rlt = y
    p_0 = 1.0 - p_1

    # Population-dependent photon-assisted generation (M25 main text):
    # g^{ph}_α = Σ_i p_i × (coefs.g_ph_α_per_state[i]). When the arrays
    # are zero (default), this reduces to the old scalar behavior.
    g_L_eff = (
        coefs.g_L
        + p_0 * coefs.g_ph_L_per_state[0]
        + p_1 * coefs.g_ph_L_per_state[1]
    )
    g_Rgt_eff = (
        coefs.g_Rgt
        + p_0 * coefs.g_ph_Rgt_per_state[0]
        + p_1 * coefs.g_ph_Rgt_per_state[1]
    )
    g_Rlt_eff = (
        coefs.g_Rlt
        + p_0 * coefs.g_ph_Rlt_per_state[0]
        + p_1 * coefs.g_ph_Rlt_per_state[1]
    )

    # Parity-changing (eo) tunneling rates assembled from bulk and
    # density-dependent contributions: Γ̃^{eo}_{ij} = Γ̃^{ph}_{ij}
    # + Σ_α Γ̃^α_{ij} x_α  (Γ̃^{eo} decomposition in M25).
    gamma_eo = (
        coefs.gamma_ph
        + coefs.gammas_L * x_L
        + coefs.gammas_Rgt * x_Rgt
        + coefs.gammas_Rlt * x_Rlt
    )

    # Qubit master equation on p_1:
    #   ṗ_1 = −(Γ̃^{eo}_{10} + Γ̃^{ee}_{10}) p_1
    #         + (Γ̃^{eo}_{01} + Γ̃^{ee}_{01}) p_0
    p_1_dot = (
        -(gamma_eo[1, 0] + coefs.gamma_ee[1, 0]) * p_1
        + (gamma_eo[0, 1] + coefs.gamma_ee[0, 1]) * p_0
    )

    # Density equations (M25 Eqs. 4–6) use the single-quasiparticle
    # rates Γ̄^α_{ij} = Γ̃^α_{ij} / N_CP(R) (M25 text below Eq. 6),
    # NOT the ensemble Γ̃ rates that enter the qubit master equation
    # above. cooper_pair_number_R defaults to 1.0 (legacy Γ̄ = Γ̃)
    # for opaque-coefficient callers; physical builders set it.
    bars_L, bars_Rgt, bars_Rlt = coefs.density_gammas()

    # Bookkeeping objects (thesis Appendix Eqs. T^α, S^{L→R>}):
    #   𝒯^α(p) = (Γ̄^α_{00} + Γ̄^α_{01}) p_0 + (Γ̄^α_{11} + Γ̄^α_{10}) p_1
    def _T(gammas: np.ndarray) -> float:
        return (gammas[0, 0] + gammas[0, 1]) * p_0 + (gammas[1, 1] + gammas[1, 0]) * p_1

    T_L = _T(bars_L)
    T_Rgt = _T(bars_Rgt)
    # 𝒮^{L→R>}(p) = Γ̄^L_{00} p_0 + (Γ̄^L_{11} + Γ̄^L_{10}) p_1
    # (drops the 01 channel because its final R-state can lie below Δ_L)
    S_L_to_Rgt = bars_L[0, 0] * p_0 + (bars_L[1, 1] + bars_L[1, 0]) * p_1

    delta = coefs.delta
    gamma_Rlt_10 = bars_Rlt[1, 0]
    gamma_L_01 = bars_L[0, 1]

    x_L_dot = (
        g_L_eff
        - coefs.r_L * x_L**2
        - delta * T_L * x_L
        + delta * T_Rgt * x_Rgt
        + delta * gamma_Rlt_10 * p_1 * x_Rlt
    )

    x_Rgt_dot = (
        g_Rgt_eff
        - coefs.r_Rgt * x_Rgt**2
        - coefs.r_cross * x_Rlt * x_Rgt
        - T_Rgt * x_Rgt
        + S_L_to_Rgt * x_L
        + coefs.xi * gamma_L_01 * p_0 * x_L
        - coefs.tau_R_inv * x_Rgt
        + coefs.tau_E_inv * x_Rlt
    )

    x_Rlt_dot = (
        g_Rlt_eff
        - coefs.r_Rlt * x_Rlt**2
        - coefs.r_cross * x_Rlt * x_Rgt
        - gamma_Rlt_10 * p_1 * x_Rlt
        + (1.0 - coefs.xi) * gamma_L_01 * p_0 * x_L
        + coefs.tau_R_inv * x_Rgt
        - coefs.tau_E_inv * x_Rlt
    )

    return np.array([p_1_dot, x_L_dot, x_Rgt_dot, x_Rlt_dot], dtype=float)


def solve_rate_equation_steady_state(
    coefs: M25Coefficients,
    *,
    initial_guess: np.ndarray | None = None,
    residual_tol: float | None = None,
    residual_tol_relative: float = 1e-3,
    accept_lm_convergence: bool = False,
    max_function_evaluations: int = 500,
) -> M25SteadyState:
    r"""Solve the M25 three-chemical-potential system for its steady state.

    Newton (``scipy.optimize.root(method='hybr')`` — MINPACK ``hybrd``,
    finite-difference Jacobian) on the 4-unknown residual
    ``(ṗ_1, ẋ_L, ẋ_{R>}, ẋ_{R<}) = 0``. The system is polynomial in
    the unknowns (quadratic in densities, bilinear in ``p × x``), so
    when the guess is in the basin of the physical fixed point the
    solver converges in ``≲ 10`` steps.

    The ``method='hybr'`` choice is deliberate: the legacy
    ``method='lm'`` wrapper has FORTRAN COMMON-block state and is
    non-deterministic across repeated calls with identical inputs.
    The ``accept_lm_convergence`` parameter name is kept for
    backward compatibility but governs acceptance of hybr's
    "iteration not making good progress" status, not LM's.

    Parameters
    ----------
    coefs
        Rate coefficients, see :class:`M25Coefficients`.
    initial_guess
        Optional length-4 array ``(p_1, x_L, x_{R>}, x_{R<})``.
        Default seeds the qubit at ee detailed balance,
        ``p_1 = Γ̃^{ee}_{01}/(Γ̃^{ee}_{01}+Γ̃^{ee}_{10})``, and each QP
        density at its generation/recombination balance scale
        ``√(g_eff/r)`` (0 when there is no generation) — which
        converges to the physical (nonequilibrium) branch for
        all validated parameter sets. Alternative branches (e.g. the
        thermal-equilibrium fixed point at high T) can be reached by
        passing a guess close to them.
    residual_tol
        Absolute acceptance threshold on ``||R||_∞`` in Hz. If
        ``None`` (default), an automatic **source-based** tolerance
        ``min_nonzero_source_rate × residual_tol_relative`` is used
        (floored at 1e-14 Hz). At steady state, every residual
        component balances against the drive/source terms ``g_α``,
        ``Γ^{ee}``, ``Γ^{ph}``, so ``||R||_∞ << min(sources)`` is
        the right physical accuracy criterion. Coefficient-magnitude-
        based auto-scaling (the previous default) accepted residuals
        far above the physical source scale whenever tunneling
        coefficients dwarfed the drive — this is specifically the
        regime of SI-Note-V-built Fig 3 coefficients, so that path
        now fails loudly (as intended) until variable rescaling or
        T-continuation lands (Stage B).
    residual_tol_relative
        Multiplier for the auto-scaled default; ignored when
        ``residual_tol`` is given explicitly. The default ``1e-3``
        demands ~3-significant-figure balance relative to the smallest
        source rate — enough for physics precision while staying
        achievable at float64 for SI-derived M25 coefficients (where
        cancellation between tunneling terms at ``~10¹¹ Hz`` floors
        the achievable ``||R||_∞`` at ``~10⁻⁴`` Hz). Tighten this
        when coefficients are smaller or variable rescaling is in
        place (Stage B).
    accept_lm_convergence
        Backward-compatible name (kept across the lm→hybr solver
        switch). When ``True``, accept the result if hybr stalls
        with the specific "iteration is not making good progress"
        status AND the residual is at or below 1.0 Hz. This is the
        cancellation-floor escape hatch for M25 Fig 3 inputs, where
        ``Γ̃ × x ~ 1e4 Hz`` tunneling currents cancel to ~1e-5 Hz —
        below ``residual_tol`` is unreachable but the answer is
        still physically meaningful. The bypass does NOT cover
        other failure modes (maxfev hit, "no further improvement",
        etc.) which always raise. Default ``False`` keeps the strict
        residual check.
    max_function_evaluations
        Hard cap passed to scipy. Each Newton step typically costs
        5 evaluations (1 residual + 4 FD-Jacobian columns).

    Returns
    -------
    M25SteadyState
        Converged state plus solver diagnostics.

    Raises
    ------
    RuntimeError
        If the solver does not converge or converges to an
        unphysical branch (negative density or ``p_1 ∉ [0, 1]``).
    """
    if initial_guess is None:
        ee_01 = float(coefs.gamma_ee[0, 1])
        ee_10 = float(coefs.gamma_ee[1, 0])
        # Qubit seed per SI Eq. S73 zeroth iteration: detailed-balance
        # ee ratio plus the photon-drive excitation term
        # Γ̃^{ph}_{01} / (Γ̃^{ee}_{10} (1 + e^{-ω_10/T})). When
        # gamma_ph[0, 1] = 0 this reduces exactly to the legacy
        # ee-balance seed ee_01 / (ee_01 + ee_10). Including the
        # photon term matters at M25 Fig 3 inputs, where the true
        # p_1 is photon-dominated (~1e-3) and the ee-only seed
        # (~1e-6) starts hybr far enough off to stall.
        if ee_10 > 0.0:
            boltzmann = ee_01 / ee_10
            p_1_guess = (
                float(coefs.gamma_ph[0, 1]) / (ee_10 * (1.0 + boltzmann))
                + boltzmann / (1.0 + boltzmann)
            )
            p_1_guess = float(np.clip(p_1_guess, 0.0, 1.0))
        else:
            p_1_guess = ee_01 / (ee_01 + ee_10) if ee_01 + ee_10 > 0.0 else 0.0
        p_0_guess = 1.0 - p_1_guess

        # Seed each density with √(g_eff/r) using the population-
        # weighted effective generation (thermal scalar + per-state
        # photon contribution at the guessed qubit populations).
        # When g_eff = 0 the x_α = 0 fixed point is already exact.
        def _density_seed(g_thermal: float, g_ph: np.ndarray, r: float) -> float:
            g_eff = g_thermal + p_0_guess * g_ph[0] + p_1_guess * g_ph[1]
            if g_eff > 0.0 and r > 0.0:
                return float(np.sqrt(g_eff / r))
            return 0.0
        y0 = np.array([
            p_1_guess,
            _density_seed(coefs.g_L, coefs.g_ph_L_per_state, coefs.r_L),
            _density_seed(coefs.g_Rgt, coefs.g_ph_Rgt_per_state, coefs.r_Rgt),
            _density_seed(coefs.g_Rlt, coefs.g_ph_Rlt_per_state, coefs.r_Rlt),
        ], dtype=float)
    else:
        y0 = np.asarray(initial_guess, dtype=float)
        if y0.shape != (4,):
            raise ValueError(
                f"initial_guess must have shape (4,); got {y0.shape}"
            )

    if residual_tol is None:
        source_rates = [
            coefs.g_L, coefs.g_Rgt, coefs.g_Rlt,
            float(np.max(coefs.g_ph_L_per_state)),
            float(np.max(coefs.g_ph_Rgt_per_state)),
            float(np.max(coefs.g_ph_Rlt_per_state)),
            float(np.max(coefs.gamma_ee)),
            float(np.max(coefs.gamma_ph)),
        ]
        # Filter out thermal-phonon generation noise: at low T/Δ the
        # ``g_α`` terms scale as ``exp(-Δ/T)`` and can be ~1e-50 or
        # smaller without representing any meaningful physical drive.
        # Including them in min(sources) drives the auto-tol below the
        # 1e-14 floor for any setup with a sub-Kelvin bath, which then
        # demands machine-precision balancing of ~1e10 Hz tunneling
        # currents — unachievable. Use 1e-30 Hz as the "physically
        # meaningful source" cutoff; well below the smallest realistic
        # photon-driven generation rate (~1e-15 Hz at the M25 Fig 3
        # tail) but far above the exp(-Δ/T) thermal noise floor.
        meaningful_sources = [s for s in source_rates if s > 1e-30]
        if meaningful_sources:
            residual_tol = max(
                min(meaningful_sources) * residual_tol_relative, 1e-14,
            )
        else:
            # No driving — exact steady state is (p from ee-balance,
            # all x = 0). Machine-precision floor only.
            residual_tol = 1e-14

    # Use ``scipy.optimize.root(method='hybr')`` for deterministic
    # behavior across repeated calls — the legacy ``method='lm'``
    # wrapper has FORTRAN COMMON-block state and gives different
    # answers from identical inputs. ``hybr`` (MINPACK ``hybrd``)
    # also finds different fixed points from different seeds, which
    # is what we need for M25 multi-stable input regimes (see
    # :func:`solve_rate_equation_steady_state_multi_seed`). The
    # ``success=False`` "no further improvement possible" status is
    # treated as accepted when ``accept_lm_convergence=True``,
    # because at M25 Fig 3 inputs hybr's stop criterion fires at the
    # float64 cancellation floor of the polynomial residual.
    sol = root(
        _rate_equation_residual,
        y0,
        args=(coefs,),
        method="hybr",
        options={"xtol": 1e-13, "maxfev": int(max_function_evaluations)},
    )

    residual_inf_norm = float(np.max(np.abs(sol.fun)))
    residual_check_failed = residual_inf_norm > residual_tol
    # Acceptance rules:
    # 1. ``sol.success=True`` → accept only if the physical residual
    #    check passes.
    # 2. ``sol.success=False`` with the "no progress" message →
    #    hybr stalled at the float64 cancellation floor of the
    #    polynomial residual; accept iff ``accept_lm_convergence``
    #    AND the residual is bounded (we cap at 1.0 Hz so a
    #    runaway maxfev hit doesn't sneak through with a huge
    #    residual just because the caller set the bypass flag).
    # 3. Any other failure (maxfev, etc.) → raise unconditionally.
    NO_PROGRESS_MARKER = "iteration is not making good progress"
    is_no_progress_stall = (
        not sol.success and NO_PROGRESS_MARKER in str(sol.message)
    )
    if not sol.success and not is_no_progress_stall:
        raise RuntimeError(
            f"M25 Newton solve failed: {sol.message}; "
            f"||R||_∞ = {residual_inf_norm:g} (tol {residual_tol:g}); "
            f"nfev = {sol.nfev}."
        )
    if is_no_progress_stall and accept_lm_convergence and residual_inf_norm > 1.0:
        # Even the bypass should not accept a residual this large —
        # the cancellation-floor regime sits at ~1e-5 Hz, not order 1.
        raise RuntimeError(
            f"M25 Newton stalled with residual far above the expected "
            f"cancellation floor: {sol.message}; "
            f"||R||_∞ = {residual_inf_norm:g} > 1.0 Hz; nfev = {sol.nfev}. "
            "accept_lm_convergence does not bypass this safety check."
        )
    allow_residual_bypass = accept_lm_convergence and is_no_progress_stall
    if residual_check_failed and not allow_residual_bypass:
        raise RuntimeError(
            f"M25 Newton converged with high residual: {sol.message}; "
            f"||R||_∞ = {residual_inf_norm:g} > tol = {residual_tol:g}; "
            f"nfev = {sol.nfev}. Pass accept_lm_convergence=True if "
            "your problem sits at the float64 cancellation floor "
            "(typical for M25 Fig 3 inputs)."
        )

    p_1, x_L, x_Rgt, x_Rlt = (float(v) for v in sol.x)

    if not (0.0 <= p_1 <= 1.0):
        raise RuntimeError(
            f"M25 Newton converged to unphysical qubit probability "
            f"p_1 = {p_1} (must be in [0, 1]). Try a different "
            f"initial_guess."
        )
    if min(x_L, x_Rgt, x_Rlt) < 0.0:
        raise RuntimeError(
            f"M25 Newton converged to negative quasiparticle density: "
            f"(x_L, x_Rgt, x_Rlt) = ({x_L}, {x_Rgt}, {x_Rlt}). "
            f"Try a different initial_guess."
        )

    return M25SteadyState(
        p_0=1.0 - p_1,
        p_1=p_1,
        x_L=x_L,
        x_Rgt=x_Rgt,
        x_Rlt=x_Rlt,
        residual_inf_norm=residual_inf_norm,
        n_function_evaluations=int(sol.nfev),
    )


# ─────────────────────────────────────────────────────────────────────
#  Analytic T → 0 seed (SI Eqs. S64–S71).
# ─────────────────────────────────────────────────────────────────────


def analytic_low_T_seed(
    coefs: M25Coefficients,
    T_kelvin: float,
    *,
    case: str = "large_asymmetry",
) -> np.ndarray:
    r"""Return a length-4 ``(p_1, x_L, x_{R>}, x_{R<})`` analytic seed.

    Implements the M25 SI low-temperature analytical expressions:

    * ``case="small_asymmetry"`` — SI Eqs. S64–S66 (T ≪ ω_LR, gap
      asymmetry ≪ ω_10). The physical ordering is roughly
      ``x_L ≃ x_R> ≃ x_R<`` all at the same scale,
      ``x_L ≃ √(g^L / r^L)`` (Eq. S65),
      ``x_R> = (Γ̄^L − (1−ξ) Γ̃^L_01) x_L / Γ̄^R>`` (Eq. S64),
      ``x_R< = √(g^R / r^R<) − x_R>`` (Eq. S66).

    * ``case="large_asymmetry"`` — SI Eqs. S69–S71 (T < T̄, gap
      asymmetry comparable to ω_10). The physical ordering is
      ``x_R< ≫ x_L ≫ x_R>``, with
      ``x_R< = √((g^L/δ + g^R< + g^R>) / r^R<)`` (Eq. S69),
      ``x_L`` from Eq. S70 and ``x_R>`` from Eq. S71 (linear system in
      the assumption ``δ Γ̄^L ≫ r^L x_L`` and
      ``Γ̄^R> + τ_R^{-1} ≫ r^R> x_R> + r^R< x_R<``).

    The qubit excited-state population ``p_1`` is taken from the
    detailed-balance expression
    ``p_1 ≃ Γ̃^{ph}_{01}/(Γ̃^{ee}_{10}(1+e^{-ω_10/T}))
            + e^{-ω_10/T}/(1+e^{-ω_10/T})`` (SI Eq. S73 with
    ``Γ̃^L_{01} x_L = Γ̃^{R>}_{01} x_{R>} = 0`` at zeroth iteration —
    valid for T ≪ ω_10 − ω_LR; the caption to Fig. S3 confirms this is
    the form used in the SI's analytic plots). The temperature
    ``T_kelvin`` enters only through this Boltzmann factor.

    Parameters
    ----------
    coefs
        Rate coefficients at the target temperature. The
        :func:`rate_equation_coefficients.coefficients_from_physical_parameters_with_photon_drive`
        builder is the typical caller.
    T_kelvin
        Bath temperature ``T`` in Kelvin (used only in the
        ``e^{-ω_10/T}`` thermal factor for ``p_1``).
    case
        ``"large_asymmetry"`` for the M25 Fig 3b regime
        (``ω_LR/2π ≳ 1 GHz``) or ``"small_asymmetry"`` for Fig 3a.
        See SI Note VI A vs VI B.

    Returns
    -------
    np.ndarray
        Length-4 array ``(p_1, x_L, x_{R>}, x_{R<})``. Values are
        clamped to the strict positive half-line (``≥ 1e-30``) to
        keep the seed inside Newton's basin for the photon-driven
        branch.

    Notes
    -----
    The seed is structurally consistent with the SI's iterative
    procedure but evaluated at the **zeroth iteration**: we use the
    p_1 from Eq. S73 to evaluate Γ̄^L (Eq. S61) and Γ̄^R> (Eq. S62)
    once, then plug into S64-S66 / S69-S71. Newton then polishes
    against the full residual.
    """
    if case not in ("large_asymmetry", "small_asymmetry"):
        raise ValueError(
            f"case must be 'large_asymmetry' or 'small_asymmetry'; "
            f"got {case!r}"
        )
    if T_kelvin <= 0:
        raise ValueError(f"T_kelvin must be positive; got {T_kelvin}")

    # ── p_1 from SI Eq. S73 at zeroth iteration ─────────────────────
    # Need ω_10 in Kelvin; derive from gamma_ee detailed balance:
    # Γ̃^{ee}_{01} = e^{-ω_10/T} · Γ̃^{ee}_{10}  (M25 text above Eq. 4).
    # When both ee entries are zero (no qubit drive), use 0 for the
    # Boltzmann factor (T1-limit by photon drive only).
    ee_01 = float(coefs.gamma_ee[0, 1])
    ee_10 = float(coefs.gamma_ee[1, 0])
    if ee_10 > 0.0 and ee_01 > 0.0:
        # Recover e^{-ω_10/T} from the supplied detailed-balance ratio.
        boltzmann = ee_01 / ee_10
        # Numerical safety: coefficients_from_physical_parameters
        # enforces detailed balance with the input T, so this matches
        # exp(-ω_10/T_kelvin) up to float64 rounding.
        boltzmann = float(np.clip(boltzmann, 0.0, 1.0))
    else:
        boltzmann = 0.0

    Gamma_ph_01 = float(coefs.gamma_ph[0, 1])
    p_1 = (
        Gamma_ph_01 / (ee_10 * (1.0 + boltzmann)) if ee_10 > 0.0 else 0.0
    ) + (boltzmann / (1.0 + boltzmann))
    p_1 = float(np.clip(p_1, 1e-12, 1.0 - 1e-12))
    p_0 = 1.0 - p_1

    # ── Γ̄^L and Γ̄^R> at this p_1 (SI Eqs. S61, S62) ────────────────
    # The SI density formulas are written in the single-quasiparticle
    # (bar) normalization Γ̄ = Γ̃ / N_CP(R) — same convention as the
    # density equations in _rate_equation_residual.
    gL, gRgt, gRlt = coefs.density_gammas()
    Gamma_bar_L = (
        p_0 * (gL[0, 0] + gL[0, 1]) + p_1 * (gL[1, 1] + gL[1, 0])
    )
    Gamma_bar_Rgt = (
        p_0 * (gRgt[0, 0] + gRgt[0, 1]) + p_1 * (gRgt[1, 1] + gRgt[1, 0])
    )

    # Effective generation rates at this p_1 (M25 main text:
    # g^{ph}_α(p) = p_0 · array[0] + p_1 · array[1] plus thermal scalar).
    g_L_eff = (
        coefs.g_L
        + p_0 * float(coefs.g_ph_L_per_state[0])
        + p_1 * float(coefs.g_ph_L_per_state[1])
    )
    g_Rgt_eff = (
        coefs.g_Rgt
        + p_0 * float(coefs.g_ph_Rgt_per_state[0])
        + p_1 * float(coefs.g_ph_Rgt_per_state[1])
    )
    g_Rlt_eff = (
        coefs.g_Rlt
        + p_0 * float(coefs.g_ph_Rlt_per_state[0])
        + p_1 * float(coefs.g_ph_Rlt_per_state[1])
    )
    g_R_total = g_Rlt_eff + g_Rgt_eff

    delta = float(coefs.delta)
    xi = float(coefs.xi)
    Gamma_L_01 = float(gL[0, 1])
    Gamma_Rlt_10 = float(gRlt[1, 0])

    floor = 1e-30
    if case == "small_asymmetry":
        # SI Eqs. S64–S66 (T ≪ ω_LR, small gap asymmetry).
        # x_L ≃ √(g^L / r^L)   (Eq. S65 simplified form, valid when
        #     Γ̃^L_01 ≪ √(g^L r^L) — the dominant branch is recombination
        #     against the photon drive).
        x_L = (
            float(np.sqrt(g_L_eff / coefs.r_L))
            if coefs.r_L > 0.0 and g_L_eff > 0.0
            else floor
        )
        # x_R> = (Γ̄^L − (1 − ξ) Γ̃^L_01) x_L / Γ̄^R>     (Eq. S64).
        if Gamma_bar_Rgt > 0.0:
            x_Rgt = max(
                (Gamma_bar_L - (1.0 - xi) * Gamma_L_01) * x_L / Gamma_bar_Rgt,
                floor,
            )
        else:
            x_Rgt = floor
        # x_R< = √(g^R / r^R<) − x_R>     (Eq. S66).
        if coefs.r_Rlt > 0.0 and g_R_total > 0.0:
            x_Rlt = max(
                float(np.sqrt(g_R_total / coefs.r_Rlt)) - x_Rgt, floor,
            )
        else:
            x_Rlt = floor
    else:  # large_asymmetry
        # SI Eq. S69: x_R< = √((g^L/δ + g^R< + g^R>) / r^R<).
        if coefs.r_Rlt > 0.0:
            inside = max(g_L_eff / max(delta, floor) + g_R_total, 0.0)
            x_Rlt = max(float(np.sqrt(inside / coefs.r_Rlt)), floor)
        else:
            x_Rlt = floor

        tau_R_inv = float(coefs.tau_R_inv)
        tau_E_inv = float(coefs.tau_E_inv)
        # Common denominator for Eqs. S70 and S71:
        # D = Γ̄^L · τ_R^{-1} + Γ̄^R> · p_0 · Γ̃^L_01 · (1 − ξ).
        D = (
            Gamma_bar_L * tau_R_inv
            + Gamma_bar_Rgt * p_0 * Gamma_L_01 * (1.0 - xi)
        )
        # Common term in numerators: (g^L/δ + p_1 Γ̃^{R<}_10 x_R<).
        N1 = g_L_eff / max(delta, floor) + p_1 * Gamma_Rlt_10 * x_Rlt
        # Second numerator term: (g^R> + τ_E^{-1} x_R<) · Γ̄^R>  for x_L
        #                     and (g^R> + τ_E^{-1} x_R<) · Γ̄^L  for x_R>.
        N2 = g_Rgt_eff + tau_E_inv * x_Rlt

        if D > 0.0:
            # SI Eq. S70: x_L = [N1 (Γ̄^R> + τ_R^{-1}) + N2 · Γ̄^R>] / D.
            x_L = (N1 * (Gamma_bar_Rgt + tau_R_inv) + N2 * Gamma_bar_Rgt) / D
            # SI Eq. S71: x_R> = [N1 · (Γ̄^L − p_0 Γ̃^L_01 (1−ξ)) + N2 · Γ̄^L] / D.
            x_Rgt = (
                N1 * (Gamma_bar_L - p_0 * Gamma_L_01 * (1.0 - xi))
                + N2 * Gamma_bar_L
            ) / D
        else:
            # Degenerate denominator: fall back to S64-S66 ordering with
            # x_L sized by photon-driven balance and x_R> small.
            x_L = (
                float(np.sqrt(g_L_eff / coefs.r_L))
                if coefs.r_L > 0.0 and g_L_eff > 0.0 else floor
            )
            x_Rgt = floor
        x_L = max(x_L, floor)
        x_Rgt = max(x_Rgt, floor)

    return np.array([p_1, x_L, x_Rgt, x_Rlt], dtype=float)


# ─────────────────────────────────────────────────────────────────────
#  Thermal-equilibrium seed (SI Eqs. S2, S4, S5 at μ_α = 0).
# ─────────────────────────────────────────────────────────────────────


def thermal_equilibrium_seed(
    *,
    Delta_L_kelvin: float,
    Delta_R_kelvin: float,
    omega_10_kelvin: float,
    T_kelvin: float,
) -> np.ndarray:
    r"""Return a length-4 ``(p_1, x_L, x_{R>}, x_{R<})`` thermal seed.

    Evaluates the full-equilibrium densities at ``μ_α = 0`` (M25 SI
    Eqs. S2, S4, S5 — arXiv Eqs. 10, 12, 13):

    .. math::

        x_L^{eq} &= \sqrt{2\pi T/\Delta_L}\, e^{-\Delta_L/T},\\
        x_{R<}^{eq} &= \sqrt{2\pi T/\Delta_R}\, e^{-\Delta_R/T}\,
            \mathrm{erf}\sqrt{\omega_{LR}/T},\\
        x_{R>}^{eq} &= \sqrt{2\pi T/\Delta_R}\, e^{-\Delta_R/T}\,
            \mathrm{erfc}\sqrt{\omega_{LR}/T},

    and the detailed-balance qubit population
    ``p_1 = e^{-ω_10/T} / (1 + e^{-ω_10/T})`` (SI Eq. S73 thermal
    limit). Used by :func:`solve_rate_equation_branch` to seed the
    thermal branch at the high-temperature end of a sweep.

    Raises
    ------
    ValueError
        Unless ``Δ_L > Δ_R > 0``, ``ω_10 > 0`` and ``T > 0``.
    """
    if not (Delta_L_kelvin > Delta_R_kelvin > 0.0):
        raise ValueError(
            f"Require Delta_L_kelvin > Delta_R_kelvin > 0; got "
            f"{Delta_L_kelvin}, {Delta_R_kelvin}."
        )
    if omega_10_kelvin <= 0.0:
        raise ValueError(f"omega_10_kelvin must be positive; got {omega_10_kelvin}")
    if T_kelvin <= 0.0:
        raise ValueError(f"T_kelvin must be positive; got {T_kelvin}")
    omega_LR = Delta_L_kelvin - Delta_R_kelvin
    floor = 1e-300
    x_L = max(
        float(np.sqrt(2.0 * np.pi * T_kelvin / Delta_L_kelvin)
              * np.exp(-Delta_L_kelvin / T_kelvin)),
        floor,
    )
    x_R = float(np.sqrt(2.0 * np.pi * T_kelvin / Delta_R_kelvin)
                * np.exp(-Delta_R_kelvin / T_kelvin))
    sqrt_ratio = float(np.sqrt(omega_LR / T_kelvin))
    x_Rlt = max(x_R * float(erf(sqrt_ratio)), floor)
    x_Rgt = max(x_R * float(erfc(sqrt_ratio)), floor)
    boltz = float(np.exp(-omega_10_kelvin / T_kelvin))
    p_1 = boltz / (1.0 + boltz)
    return np.array([p_1, x_L, x_Rgt, x_Rlt], dtype=float)


# ─────────────────────────────────────────────────────────────────────
#  Chemical-potential inversion (SI Eqs. S2, S4, S5).
# ─────────────────────────────────────────────────────────────────────


def chemical_potentials_kelvin(
    *,
    Delta_L_kelvin: float,
    Delta_R_kelvin: float,
    T_kelvin: np.ndarray | float,
    x_L: np.ndarray | float,
    x_Rgt: np.ndarray | float,
    x_Rlt: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Return ``(μ_L, μ_{R>}, μ_{R<})`` in Kelvin from the QP densities.

    Paper-exact inversion of the M25 density formulas (arXiv
    2408.17218 Eqs. 10, 12, 13 — published SI Eqs. S2, S4, S5), the
    exact inverse of :func:`thermal_equilibrium_seed`'s forward
    evaluation:

    .. math::

        \mu_L    &= \Delta_L + T \ln\!\big( x_L
                    \sqrt{\Delta_L / 2\pi T} \big),\\
        \mu_{R>} &= \Delta_R + T \ln\!\big( x_{R>}
                    \sqrt{\Delta_R / 2\pi T} \,/\,
                    \mathrm{erfc}\sqrt{\omega_{LR}/T} \big),\\
        \mu_{R<} &= \Delta_R + T \ln\!\big( x_{R<}
                    \sqrt{\Delta_R / 2\pi T} \,/\,
                    \mathrm{erf}\sqrt{\omega_{LR}/T} \big),

    with ``ω_LR = Δ_L − Δ_R``. The ``√(Δ/2πT)`` prefactor and the
    erf/erfc partition factors matter: the naive inversion
    ``μ = Δ + T ln x`` understates ``μ_{R>}`` by
    ``−T ln erfc√(ω_LR/T)`` (≈ 5 GHz at the M25 Fig 3b panel's
    10 mK), which flips the ``μ_{R>}`` vs ``μ_{R<}`` ordering
    relative to the published figure. At full thermal equilibrium
    all three chemical potentials → 0.

    Parameters
    ----------
    Delta_L_kelvin, Delta_R_kelvin
        Electrode gaps in Kelvin (``Δ_α / k_B``), ``Δ_L > Δ_R > 0``.
    T_kelvin
        Bath temperature(s) in Kelvin, scalar or array, strictly
        positive.
    x_L, x_Rgt, x_Rlt
        Dimensionless quasiparticle densities (normalized to the
        local-gap Cooper-pair number), broadcastable against
        ``T_kelvin``. Non-positive or NaN densities yield NaN/−inf
        entries (no warning is emitted) — callers with failed sweep
        points can pass them through unfiltered.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(μ_L, μ_{R>}, μ_{R<})`` in Kelvin, matching the broadcast
        shape of the inputs.

    Raises
    ------
    ValueError
        Unless ``Δ_L > Δ_R > 0`` and all temperatures are positive.
    """
    if not (Delta_L_kelvin > Delta_R_kelvin > 0.0):
        raise ValueError(
            f"Require Delta_L_kelvin > Delta_R_kelvin > 0; got "
            f"{Delta_L_kelvin}, {Delta_R_kelvin}."
        )
    T = np.asarray(T_kelvin, dtype=float)
    if not np.all(T > 0.0):
        raise ValueError(f"T_kelvin must be positive; got {T_kelvin}")
    xL = np.asarray(x_L, dtype=float)
    xRgt = np.asarray(x_Rgt, dtype=float)
    xRlt = np.asarray(x_Rlt, dtype=float)

    omega_LR_kelvin = Delta_L_kelvin - Delta_R_kelvin
    with np.errstate(divide="ignore", invalid="ignore"):
        sqrt_ratio = np.sqrt(omega_LR_kelvin / T)
        mu_L = Delta_L_kelvin + T * np.log(
            xL * np.sqrt(Delta_L_kelvin / (2.0 * np.pi * T))
        )
        mu_Rgt = Delta_R_kelvin + T * np.log(
            xRgt * np.sqrt(Delta_R_kelvin / (2.0 * np.pi * T)) / erfc(sqrt_ratio)
        )
        mu_Rlt = Delta_R_kelvin + T * np.log(
            xRlt * np.sqrt(Delta_R_kelvin / (2.0 * np.pi * T)) / erf(sqrt_ratio)
        )
    return np.asarray(mu_L), np.asarray(mu_Rgt), np.asarray(mu_Rlt)


# ─────────────────────────────────────────────────────────────────────
#  Branch-continuation driver (M25 Figs. 3-4 temperature sweeps).
# ─────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M25BranchSweep:
    """Composite physical curve from bidirectional branch continuation.

    Attributes
    ----------
    T_kelvin
        The requested temperature grid (strictly increasing).
    states
        Composite physical curve, one :class:`M25SteadyState` per
        grid point, assembled from the two directional passes by the
        selection rule documented on
        :func:`solve_rate_equation_branch`.
    branch_labels
        Per-point provenance: ``"merged"`` (both passes converged to
        the same fixed point — the generic M25 outcome, since the
        correctly normalized system has a unique physical root),
        ``"photon"`` (photon-driven low-T branch), or ``"thermal"``
        (thermal high-T branch).
    T_exchange_kelvin
        Temperature at which the composite switches from the photon
        to the thermal branch. ``None`` when the branches merged at
        every grid point (no exchange needed).
    photon_states, thermal_states
        Raw per-pass results (``None`` where a pass terminated at a
        fold or never reached the point).
    coefficients
        The exact per-grid-point :class:`M25Coefficients` bundles the
        driver solved against, in grid order (from its internal
        per-T cache). Callers computing post-hoc observables (e.g.
        ``Γ̃^{eo}`` assembly for the M25 Fig 4 parity rates) must use
        these rather than re-calling their coefficient builder per T
        — structurally the same objects the states were converged on.
    """

    T_kelvin: np.ndarray
    states: tuple[M25SteadyState, ...]
    branch_labels: tuple[str, ...]
    T_exchange_kelvin: float | None
    photon_states: tuple[M25SteadyState | None, ...]
    thermal_states: tuple[M25SteadyState | None, ...]
    coefficients: tuple[M25Coefficients, ...]


def _branch_corrector(
    coefs: M25Coefficients,
    seed: np.ndarray | None,
    residual_tol_relative: float,
    max_function_evaluations: int = 500,
) -> M25SteadyState | None:
    """Newton corrector step: strict solve from ``seed``, or ``None``.

    Wraps :func:`solve_rate_equation_steady_state` with its strict
    source-scaled residual acceptance (no ``accept_lm_convergence``
    bypass): the continuation driver only ever tracks true fixed
    points, never cancellation-floor pseudo-roots.
    """
    try:
        return solve_rate_equation_steady_state(
            coefs,
            initial_guess=seed,
            residual_tol_relative=residual_tol_relative,
            max_function_evaluations=max_function_evaluations,
        )
    except RuntimeError:
        return None


def _log10_jump(a: M25SteadyState, b: M25SteadyState) -> float:
    """Max |log10| density ratio between two states (branch metric)."""
    floor = 1e-300
    return max(
        abs(math.log10((a.x_L + floor) / (b.x_L + floor))),
        abs(math.log10((a.x_Rgt + floor) / (b.x_Rgt + floor))),
        abs(math.log10((a.x_Rlt + floor) / (b.x_Rlt + floor))),
    )


def _states_agree(a: M25SteadyState, b: M25SteadyState, rtol: float) -> bool:
    """Componentwise relative agreement of two fixed points."""
    for va, vb in (
        (a.p_1, b.p_1), (a.x_L, b.x_L), (a.x_Rgt, b.x_Rgt), (a.x_Rlt, b.x_Rlt),
    ):
        if abs(va - vb) > rtol * max(abs(va), abs(vb), 1e-300):
            return False
    return True


def _transverse_solve(
    coefs: M25Coefficients,
    x_L: float,
    z_seed: np.ndarray,
) -> np.ndarray | None:
    """Solve the transverse subsystem ``(ṗ_1, ẋ_{R>}, ẋ_{R<}) = 0``.

    Holds ``x_L`` fixed and solves for ``(p_1, x_{R>}, x_{R<})`` by
    Newton (``hybr``). Unlike the full 4-unknown system near a flat
    valley, this subsystem is well conditioned, so plain ``hybr``
    convergence is reliable. Returns ``None`` when Newton fails or
    lands outside the physical box.
    """
    def _res(z: np.ndarray) -> np.ndarray:
        y = np.array([z[0], x_L, z[1], z[2]], dtype=float)
        return _rate_equation_residual(y, coefs)[[0, 2, 3]]

    # Seed ladder. A caller-provided warm start is preferred, but a
    # degenerate one (densities many decades below the residual's
    # resolvable scale) yields a numerically-zero finite-difference
    # Jacobian column and hybr cannot move — so retry from generic
    # scale-aware seeds tied to the x_L being scanned.
    p_1_generic = float(np.clip(z_seed[0], 1e-6, 1.0 - 1e-6))
    seeds = (
        np.asarray(z_seed, dtype=float),
        np.array([p_1_generic, 0.4 * x_L, 0.4 * x_L]),
        np.array([p_1_generic, 1e-3 * x_L, 1e-3 * x_L]),
    )
    for seed in seeds:
        sol = root(_res, seed, method="hybr", options={"xtol": 1e-13})
        if not sol.success:
            continue
        p_1, x_Rgt, x_Rlt = (float(v) for v in sol.x)
        if not (0.0 <= p_1 <= 1.0) or x_Rgt < 0.0 or x_Rlt < 0.0:
            continue
        return np.asarray(sol.x, dtype=float)
    return None


def _relocate_root_1d(
    coefs: M25Coefficients,
    x_L_center: float,
    z_seed: np.ndarray,
    *,
    window_decades: float,
    n_samples: int,
    residual_tol_relative: float,
    max_function_evaluations: int = 500,
) -> M25SteadyState | None:
    r"""Reduced 1-D bracketing fallback (fold-capable root relocation).

    Fold-handling scheme equivalent to pseudo-arclength stepping,
    specialized to the M25 system's geometry: the ill-conditioned
    direction of the 4-unknown Jacobian is (empirically) the ``x_L``
    density scale, while the transverse subsystem
    ``(p_1, x_{R>}, x_{R<})`` is well conditioned at fixed ``x_L``.
    Parametrizing the solution manifold by ``x_L`` therefore turns
    branch location into a scalar root-find of the reduced residual

    .. math:: R(x_L) \equiv \dot x_L\big(p_1^*(x_L), x_L,
              x_{R>}^*(x_L), x_{R<}^*(x_L)\big)

    which is bracketed on a log grid and polished with Brent's method
    — deterministic, derivative-free, and immune to the valley
    conditioning that defeats damped-Newton continuation. A fold
    (saddle-node in T) appears as the loss of the sign change inside
    the window: the function then returns ``None`` and the caller
    records the fold instead of silently jumping branches. Multiple
    coexisting fixed points appear as multiple sign changes; the
    bracket nearest ``x_L_center`` (in log space) is selected, which
    is what keeps the continuation on *its own* branch.
    """
    log_c = math.log10(max(x_L_center, 1e-300))
    xs = np.logspace(log_c - window_decades / 2.0, log_c + window_decades / 2.0, n_samples)

    samples: list[tuple[float, float, np.ndarray]] = []
    z = np.asarray(z_seed, dtype=float)
    for x_L in xs:
        z_new = _transverse_solve(coefs, float(x_L), z)
        if z_new is None:
            continue
        z = z_new
        y = np.array([z[0], float(x_L), z[1], z[2]], dtype=float)
        R_xL = float(_rate_equation_residual(y, coefs)[1])
        samples.append((float(x_L), R_xL, z.copy()))

    # Sign-change brackets; pick the one nearest the center in log-x.
    best: tuple[float, float, np.ndarray] | None = None
    best_dist = math.inf
    for (x_a, R_a, z_a), (x_b, R_b, _z_b) in itertools.pairwise(samples):
        if R_a == 0.0:
            mid = math.log10(x_a)
            if abs(mid - log_c) < best_dist:
                best, best_dist = (x_a, x_a, z_a), abs(mid - log_c)
            continue
        if np.sign(R_a) != np.sign(R_b):
            mid = 0.5 * (math.log10(x_a) + math.log10(x_b))
            if abs(mid - log_c) < best_dist:
                best, best_dist = (x_a, x_b, z_a), abs(mid - log_c)
    if best is None:
        return None
    x_lo, x_hi, z_bracket = best

    if x_lo == x_hi:
        x_root = x_lo
        z_root: np.ndarray | None = z_bracket
    else:
        z_carry = {"z": z_bracket.copy()}

        def _reduced(x_L: float) -> float:
            z_new = _transverse_solve(coefs, x_L, z_carry["z"])
            if z_new is None:
                # Brent cannot handle a missing sample; treat as the
                # previous transverse solution's residual sign carrier
                # by re-raising — caller falls back to no-root.
                raise RuntimeError("transverse solve failed inside bracket")
            z_carry["z"] = z_new
            y = np.array([z_new[0], x_L, z_new[1], z_new[2]], dtype=float)
            return float(_rate_equation_residual(y, coefs)[1])

        try:
            x_root = float(brentq(_reduced, x_lo, x_hi, xtol=1e-300, rtol=8.9e-16))
        except (RuntimeError, ValueError):
            return None
        z_root = _transverse_solve(coefs, x_root, z_carry["z"])
        if z_root is None:
            return None

    assert z_root is not None  # both branches above guarantee it
    y_root = np.array([z_root[0], x_root, z_root[1], z_root[2]], dtype=float)
    # Polish on the full 4-unknown system (also applies the strict
    # residual acceptance and physical-branch checks).
    polished = _branch_corrector(
        coefs, y_root, residual_tol_relative, max_function_evaluations,
    )
    if polished is not None:
        return polished
    # Accept the reduced root as-is when the full Newton polish is
    # unavailable (e.g. exactly at a fold, where the full Jacobian is
    # singular); record the measured residual honestly.
    residual = float(np.max(np.abs(_rate_equation_residual(y_root, coefs))))
    return M25SteadyState(
        p_0=1.0 - float(z_root[0]),
        p_1=float(z_root[0]),
        x_L=float(x_root),
        x_Rgt=float(z_root[1]),
        x_Rlt=float(z_root[2]),
        residual_inf_norm=residual,
        n_function_evaluations=0,
    )


def _state_to_array(state: M25SteadyState) -> np.ndarray:
    return np.array([state.p_1, state.x_L, state.x_Rgt, state.x_Rlt], dtype=float)


def _continuation_pass(
    coefficients_at: Callable[[float], M25Coefficients],
    T_grid: np.ndarray,
    indices: Sequence[int],
    seed0: np.ndarray | None,
    *,
    jump_tol_decades: float,
    max_step_bisections: int,
    residual_tol_relative: float,
    scan_window_decades: float,
    full_scan_bounds: tuple[float, float],
    max_function_evaluations: int = 500,
) -> list[M25SteadyState | None]:
    """One directional continuation pass over ``T_grid[indices]``.

    Natural-parameter continuation with warm starts; on corrector
    failure or a continuity violation, the T-step is bisected
    adaptively (up to ``max_step_bisections`` levels). When bisection
    bottoms out, the reduced 1-D bracketing fallback
    (:func:`_relocate_root_1d`) relocates the branch — first in a
    local window, then over ``full_scan_bounds``. If the relocated
    root is farther than the continuity tolerance, the branch has
    terminated at a fold: the pass stops and the remaining points are
    left ``None`` (the opposite-direction pass covers them).
    """
    out: list[M25SteadyState | None] = [None] * len(T_grid)
    base_step = float(np.max(np.abs(np.diff(T_grid)))) if len(T_grid) > 1 else 1.0

    def _solve_at(T: float, seed: np.ndarray | None) -> M25SteadyState | None:
        return _branch_corrector(
            coefficients_at(float(T)), seed, residual_tol_relative,
            max_function_evaluations,
        )

    def _advance(
        prev: M25SteadyState, T_from: float, T_to: float, depth: int,
    ) -> M25SteadyState | None:
        seed = _state_to_array(prev)
        sol = _solve_at(T_to, seed)
        # Continuity budget scales with the fraction of the base grid
        # step this sub-step spans (a bisected half-step may only move
        # the densities half as far, in log space, as a full step).
        frac = max(abs(T_to - T_from) / base_step, 1.0 / 2 ** max_step_bisections)
        allowed = jump_tol_decades * max(frac, 0.25)
        if sol is not None and _log10_jump(sol, prev) <= allowed:
            return sol
        if depth < max_step_bisections:
            T_mid = 0.5 * (T_from + T_to)
            mid = _advance(prev, T_from, T_mid, depth + 1)
            if mid is not None:
                stepped = _advance(mid, T_mid, T_to, depth + 1)
                if stepped is not None:
                    return stepped
        # Reduced 1-D fallback: local window around the previous root,
        # then a full-range rescan.
        coefs_to = coefficients_at(float(T_to))
        z_seed = np.array([prev.p_1, prev.x_Rgt, prev.x_Rlt], dtype=float)
        full_window = math.log10(full_scan_bounds[1] / full_scan_bounds[0])
        full_center = math.sqrt(full_scan_bounds[0] * full_scan_bounds[1])
        for center, window in (
            (prev.x_L, scan_window_decades),
            (full_center, full_window),
        ):
            relocated = _relocate_root_1d(
                coefs_to, center, z_seed,
                window_decades=window,
                n_samples=max(9, int(window * 6) | 1),
                residual_tol_relative=residual_tol_relative,
                max_function_evaluations=max_function_evaluations,
            )
            if relocated is not None and _log10_jump(relocated, prev) <= jump_tol_decades:
                return relocated
        return None  # fold: branch does not continue to T_to

    first = indices[0]
    current = _solve_at(float(T_grid[first]), seed0)
    if current is None and seed0 is not None:
        # Strict corrector could not converge from the caller's seed;
        # try the reduced 1-D locator — first in a local window around
        # the seed, then across the full scan range but still centered
        # on the (clipped) seed so the bracket *nearest the seed* wins
        # and branch identity is preserved. The full-range attempt
        # covers seeds far outside any root's basin (e.g. a thermal-
        # equilibrium seed at 1e-50 on a grid entirely below T̄).
        coefs_first = coefficients_at(float(T_grid[first]))
        z_first = np.array([seed0[0], seed0[2], seed0[3]], dtype=float)
        lo, hi = full_scan_bounds
        seed_center = min(max(float(seed0[1]), lo), hi)
        full_window = 2.0 * max(
            math.log10(seed_center / lo), math.log10(hi / seed_center), 1.0,
        )
        for center, window in (
            (float(seed0[1]), scan_window_decades),
            (seed_center, full_window),
        ):
            current = _relocate_root_1d(
                coefs_first, center, z_first,
                window_decades=window,
                n_samples=max(9, int(window * 6) | 1),
                residual_tol_relative=residual_tol_relative,
                max_function_evaluations=max_function_evaluations,
            )
            if current is not None:
                break
    if current is None:
        return out
    out[first] = current

    for prev_idx, idx in itertools.pairwise(indices):
        nxt = _advance(current, float(T_grid[prev_idx]), float(T_grid[idx]), 0)
        if nxt is None:
            break  # fold — remaining points stay None
        out[idx] = nxt
        current = nxt
    return out


def solve_rate_equation_branch(
    coefficients_at: Callable[[float], M25Coefficients],
    T_grid_kelvin: Sequence[float] | np.ndarray,
    *,
    photon_seed: np.ndarray | None = None,
    photon_seed_case: str = "large_asymmetry",
    thermal_seed: np.ndarray | None = None,
    T_exchange_hint_kelvin: float | None = None,
    merge_rtol: float = 0.05,
    jump_tol_decades: float = 2.0,
    max_step_bisections: int = 8,
    residual_tol_relative: float = 1e-3,
    scan_window_decades: float = 4.0,
    full_scan_bounds: tuple[float, float] = (1e-16, 1e-1),
    max_function_evaluations: int = 500,
) -> M25BranchSweep:
    r"""Track the M25 steady state across a temperature sweep.

    Deterministic branch-continuation driver for the 4-unknown M25
    system (Eqs. 3–6). Two directional passes are run:

    * **photon pass** — from the lowest temperature upward, seeded by
      the SI low-temperature analytics (:func:`analytic_low_T_seed`,
      SI Eqs. S64–S71) or the caller's ``photon_seed``;
    * **thermal pass** — from the highest temperature downward, seeded
      by the full-equilibrium densities
      (:func:`thermal_equilibrium_seed`, SI Eqs. S2/S4/S5) or the
      caller's ``thermal_seed``. When no thermal seed can be built
      (the coefficient bundle does not carry the gaps), the corrector's
      default ``√(g_eff/r)`` seed is used — above ``T̄`` thermal
      generation dominates ``g_eff``, so this lands on the thermal
      branch.

    For unique-root systems (the M25 Fig 3/4 parameter family with
    the Γ̄-normalized density equations) the second (thermal) pass is
    not redundant even though the photon pass already covers the
    grid: it is the driver's built-in determinism/agreement
    cross-check — an independently seeded, opposite-direction
    continuation that must land on the same fixed point at every grid
    point ("merged" labels everywhere). Any disagreement flags either
    a genuine fold/branch pair (other parameter sets) or a solver
    regression, so the pass is deliberately kept.

    Each pass uses natural-parameter continuation (previous solution
    as Newton warm start) with adaptive step bisection on failure,
    and a reduced 1-D bracketing fallback (see
    :func:`_relocate_root_1d`) in place of pseudo-arclength stepping:
    the M25 Jacobian's near-singular direction is the ``x_L`` density
    scale, so parametrizing by ``x_L`` and Brent-bracketing the
    reduced residual handles folds robustly where arclength stepping
    on the ill-conditioned full system would not. A fold terminates
    only the affected *pass*; the composite sweep continues on the
    other branch.

    **Branch-exchange rule** (explicit, not "max x_L wins"):

    1. Grid points where both passes converged to the same fixed point
       (componentwise agreement within ``merge_rtol``) are labeled
       ``"merged"``. With the correct single-quasiparticle
       normalization of the density equations (M25 text below Eq. 6;
       :attr:`M25Coefficients.cooper_pair_number_R`), the M25 Fig 3
       parameter set has a unique physical root at every temperature,
       so the generic outcome is "merged everywhere" and no exchange
       is needed — the historical "photon vs thermal branch conflict
       above ~80 mK" was an artifact of the missing normalization.
    2. Otherwise the composite uses the photon-pass state below the
       exchange temperature and the thermal-pass state at or above it.
       The exchange temperature is the grid point nearest
       ``T_exchange_hint_kelvin`` when given (pass the M25 Eq. 8
       Lambert-W ``T̄`` from :func:`crossover_temperature_kelvin`);
       without a hint it is the lowest merged grid temperature —
       valid only when the merged points form a contiguous suffix of
       the grid (agreement, once reached, persists upward). If the
       branches never merge, or the merged pattern is non-contiguous,
       and no hint is given, a ``RuntimeError`` (with the merged
       pattern) asks for the hint rather than guessing.
    3. Where the preferred pass has no state (fold), the other pass's
       state is used; if neither pass reached a grid point, the sweep
       fails loudly.

    Determinism: fixed grids, deterministic solvers (``hybr``,
    Brent), no randomness — identical output run-to-run.

    Parameters
    ----------
    coefficients_at
        Callable mapping a temperature (Kelvin) to the
        :class:`M25Coefficients` bundle at that temperature (e.g.
        rebuilt with the photon drive recalibrated per T, as in the
        M25 Fig 3 caption).
    T_grid_kelvin
        Strictly increasing temperature grid.
    photon_seed
        Optional length-4 ``(p_1, x_L, x_{R>}, x_{R<})`` seed for the
        photon pass at ``T_grid[0]``; default builds
        :func:`analytic_low_T_seed` with ``photon_seed_case``.
    photon_seed_case
        Forwarded to :func:`analytic_low_T_seed` (``"large_asymmetry"``
        or ``"small_asymmetry"``).
    thermal_seed
        Optional seed for the thermal pass at ``T_grid[-1]``;
        typically :func:`thermal_equilibrium_seed`. ``None`` uses the
        corrector's default ``√(g_eff/r)`` seed.
    T_exchange_hint_kelvin
        A-priori branch-exchange estimate (M25 Eq. 8 ``T̄``); only
        consulted when the branches genuinely differ somewhere.
    merge_rtol
        Componentwise relative tolerance for declaring the two passes
        merged at a grid point.
    jump_tol_decades
        Continuity budget: max ``|log10|`` density ratio between
        adjacent accepted points on one branch (scaled by the T-step
        fraction during bisection). Also the branch-identity radius
        for the 1-D relocation fallback.
    max_step_bisections
        Adaptive refinement depth for natural continuation.
    residual_tol_relative
        Forwarded to :func:`solve_rate_equation_steady_state`.
    scan_window_decades, full_scan_bounds
        Geometry of the reduced 1-D fallback's local window and
        full-range rescan.
    max_function_evaluations
        Newton budget per corrector solve, forwarded to
        :func:`solve_rate_equation_steady_state` (default 500, the
        solver's own default).

    Returns
    -------
    M25BranchSweep
        Composite curve, per-point branch labels, exchange
        temperature, and both raw passes.

    Raises
    ------
    ValueError
        For an empty or non-increasing temperature grid.
    RuntimeError
        If some grid point is covered by neither pass, or the
        branches never merge and no exchange hint was provided.
    """
    T_grid = np.asarray(T_grid_kelvin, dtype=float)
    if T_grid.ndim != 1 or T_grid.size == 0:
        raise ValueError(f"T_grid_kelvin must be a nonempty 1-D grid; got shape {T_grid.shape}")
    if T_grid.size > 1 and not np.all(np.diff(T_grid) > 0.0):
        raise ValueError("T_grid_kelvin must be strictly increasing.")

    n = int(T_grid.size)

    # Per-T coefficient cache shared by both passes (and bisection
    # midpoints), keyed by the exact float temperature.
    cache: dict[float, M25Coefficients] = {}

    def _coefs(T: float) -> M25Coefficients:
        key = float(T)
        if key not in cache:
            cache[key] = coefficients_at(key)
        return cache[key]

    if photon_seed is None:
        photon_seed = analytic_low_T_seed(
            _coefs(float(T_grid[0])), float(T_grid[0]), case=photon_seed_case,
        )

    photon_states = _continuation_pass(
        _coefs, T_grid, list(range(n)), photon_seed,
        jump_tol_decades=jump_tol_decades,
        max_step_bisections=max_step_bisections,
        residual_tol_relative=residual_tol_relative,
        scan_window_decades=scan_window_decades,
        full_scan_bounds=full_scan_bounds,
        max_function_evaluations=max_function_evaluations,
    )
    thermal_states = _continuation_pass(
        _coefs, T_grid, list(range(n - 1, -1, -1)), thermal_seed,
        jump_tol_decades=jump_tol_decades,
        max_step_bisections=max_step_bisections,
        residual_tol_relative=residual_tol_relative,
        scan_window_decades=scan_window_decades,
        full_scan_bounds=full_scan_bounds,
        max_function_evaluations=max_function_evaluations,
    )

    merged = [
        p is not None and t is not None and _states_agree(p, t, merge_rtol)
        for p, t in zip(photon_states, thermal_states, strict=True)
    ]

    T_exchange: float | None
    if all(merged):
        T_exchange = None
    elif T_exchange_hint_kelvin is not None:
        T_exchange = float(T_grid[int(np.argmin(np.abs(T_grid - T_exchange_hint_kelvin)))])
    else:
        merged_indices = [i for i in range(n) if merged[i]]
        if not merged_indices:
            raise RuntimeError(
                "M25 branch sweep: the photon and thermal branches never "
                "merge on this grid and no T_exchange_hint_kelvin was "
                "given. Pass the M25 Eq. 8 Lambert-W T̄ estimate "
                "(crossover_temperature_kelvin) as the hint."
            )
        first_merged = merged_indices[0]
        if merged_indices != list(range(first_merged, n)):
            # "Lowest merged temperature" is only a meaningful
            # exchange point when agreement, once reached, persists
            # to the top of the grid (photon branch below, merged
            # above). Interleaved merged/unmerged points mean the
            # passes disagree in a pattern this inference cannot
            # arbitrate — fail with the pattern instead of guessing.
            pattern = "".join("M" if m else "." for m in merged)
            raise RuntimeError(
                "M25 branch sweep: without a T_exchange_hint_kelvin "
                "the exchange temperature is inferred as the lowest "
                "merged grid point, which requires the merged points "
                "to form a contiguous suffix of the T grid. Merged "
                f"pattern (M = merged, . = split): '{pattern}' over "
                f"T = {float(T_grid[0]):.6g}..{float(T_grid[-1]):.6g} K "
                f"({n} points). Pass the M25 Eq. 8 Lambert-W T̄ "
                "estimate (crossover_temperature_kelvin) as the hint."
            )
        T_exchange = float(T_grid[first_merged])

    states: list[M25SteadyState] = []
    labels: list[str] = []
    for i in range(n):
        p, t = photon_states[i], thermal_states[i]
        if merged[i]:
            assert p is not None
            states.append(p)
            labels.append("merged")
            continue
        prefer_photon = T_exchange is None or float(T_grid[i]) < T_exchange
        primary, secondary = (p, t) if prefer_photon else (t, p)
        chosen = primary if primary is not None else secondary
        if chosen is None:
            raise RuntimeError(
                f"M25 branch sweep: neither pass reached "
                f"T = {float(T_grid[i]):.6g} K (photon and thermal "
                f"branches both terminated before this point)."
            )
        states.append(chosen)
        labels.append(
            "photon" if chosen is p else "thermal"
        )

    return M25BranchSweep(
        T_kelvin=T_grid,
        states=tuple(states),
        branch_labels=tuple(labels),
        T_exchange_kelvin=T_exchange,
        photon_states=tuple(photon_states),
        thermal_states=tuple(thermal_states),
        # Grid-ordered coefficient bundles from the shared per-T
        # cache — the exact objects the passes solved against (every
        # grid temperature was visited by at least one pass, so these
        # are cache hits, not rebuilds).
        coefficients=tuple(_coefs(float(T)) for T in T_grid),
    )


# ─────────────────────────────────────────────────────────────────────
#  Multi-seed branch picker — selects the photon-driven nonequilibrium
#  branch from the M25 system's multiple fixed points.
# ─────────────────────────────────────────────────────────────────────


def _default_seed_grid() -> list[np.ndarray]:
    """Hand-tuned seeds covering the M25 nonequilibrium branch family.

    Each entry is a length-4 ``(p_1, x_L, x_{R>}, x_{R<})`` initial
    guess. Spans:

    * ``p_1 ∈ {1e-4, 3e-4, 1e-3}`` — covers the M25 Fig 3 caption
      range; the legacy single ``p_1 = 1e-3`` over-biased toward
      high-p_1 fixed points and missed the paper's branch (which sits
      at ``p_1 ≈ 3e-4`` for Fig 3a low-T points).
    * ``x_L ∈ ~8 decades [1e-11, 1e-4]`` — wide enough to bracket the
      photon-driven branch under the full Fig 3 parameter sweep
      without overlapping the unphysical near-zero-x noise tier.
    * ``x_{R>}/x_L = 0.4`` — matches the M25 tunneling-balance ratio
      ``T_L/T_{R>}`` at typical Fig 3a coefficients (replaces the
      legacy 0.5 ratio that biased seeds away from the paper branch).
    * ``x_{R<}/x_L = 0.02`` — matches paper's measured ratio at low T.
    """
    seeds: list[np.ndarray] = []
    for p_1 in (1e-4, 3e-4, 1e-3):
        for x in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
            seeds.append(np.array([p_1, x, 0.4 * x, 0.02 * x]))
    return seeds


def _solve_with_lm(
    coefs: M25Coefficients,
    seed: np.ndarray,
    *,
    residual_ceiling_Hz: float,
) -> M25SteadyState | None:
    """Try ``scipy.optimize.root(method='lm')`` from ``seed``.

    lm (MINPACK ``lmder``) finds true fixed points where ``hybr``
    stalls at the float64 cancellation floor of M25's ~1e10 Hz
    tunneling-current cancellations. Used only as an additional
    candidate source in the multi-seed picker — the standalone
    :func:`solve_rate_equation_steady_state` keeps hybr for its
    well-understood stall semantics.

    Returns ``None`` if the solve fails, lands on an unphysical
    branch, or has residual above ``residual_ceiling_Hz``.

    Note on determinism: lm has been verified bit-identical across
    100 repeated calls on the test platform (scipy ≥ 1.13). The
    accompanying regression test in
    ``tests/services/test_rate_equation.py::TestLmDeterminism``
    will catch any platform/version regression.
    """
    try:
        sol = root(
            _rate_equation_residual,
            seed,
            args=(coefs,),
            method="lm",
            options={"xtol": 1e-13, "maxiter": 5000},
        )
    except Exception:
        return None
    residual_inf_norm = float(np.max(np.abs(sol.fun)))
    if residual_inf_norm > residual_ceiling_Hz:
        return None
    p_1, x_L, x_Rgt, x_Rlt = (float(v) for v in sol.x)
    if not (0.0 <= p_1 <= 1.0):
        return None
    if min(x_L, x_Rgt, x_Rlt) <= 0.0:
        return None
    return M25SteadyState(
        p_0=1.0 - p_1,
        p_1=p_1,
        x_L=x_L,
        x_Rgt=x_Rgt,
        x_Rlt=x_Rlt,
        residual_inf_norm=residual_inf_norm,
        n_function_evaluations=int(sol.nfev),
    )


def _solve_with_lsq(
    coefs: M25Coefficients,
    seed: np.ndarray,
    *,
    residual_ceiling_Hz: float,
) -> M25SteadyState | None:
    r"""Try ``scipy.optimize.least_squares`` with bounds from ``seed``.

    Bounded Levenberg-Marquardt (``method='trf'``) on the same residual,
    constrained to ``p_1 ∈ [0, 1]`` and ``x_α ≥ 0``. Unlike the
    unbounded ``hybr`` and ``lm`` solvers, the trust-region reflective
    method physically cannot leave the feasible box, so it does not
    drift to negative densities and trigger the post-hoc rejection
    that loses the small-asymmetry branch in M25 Fig 3a.

    Used only as an additional candidate source in the multi-seed
    picker — the standalone :func:`solve_rate_equation_steady_state`
    keeps the original hybr semantics.

    Returns ``None`` if the solve fails, the post-hoc physical-branch
    check fails, or the residual exceeds ``residual_ceiling_Hz``.
    """
    # Lower bound: every component nonnegative. Upper bound: p_1 ≤ 1,
    # densities unbounded above (the physical branch can sit at any
    # positive scale; the residual itself prevents runaway).
    lower = np.zeros(4)
    upper = np.array([1.0, np.inf, np.inf, np.inf])
    # trf requires the seed strictly inside the box. Clip with a small
    # epsilon away from the bounds so the trust region can move freely
    # at the start. (The choice 1e-30 matches analytic_low_T_seed's
    # floor and is well below any physical density of interest.)
    seed_clipped = np.clip(np.asarray(seed, dtype=float), 1e-30, None)
    seed_clipped[0] = float(np.clip(seed_clipped[0], 1e-30, 1.0 - 1e-30))
    try:
        sol = least_squares(
            _rate_equation_residual,
            seed_clipped,
            args=(coefs,),
            method="trf",
            bounds=(lower, upper),
            # Match the hybr/lm tolerance budget on the problem:
            # M25 Fig 3 inputs cancel ~1e10 Hz tunneling currents to a
            # ~1e-5 Hz residual floor, so xtol/ftol below ~1e-12 just
            # spin on round-off without reaching a meaningfully better
            # fixed point. ``max_nfev=500`` matches the default budget
            # of :func:`solve_rate_equation_steady_state`; the lsq
            # solver only needs to be competitive with hybr for one
            # candidate, not exhaustively explore the seed grid.
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
            max_nfev=500,
        )
    except Exception:
        return None
    if not sol.success:
        # least_squares reports non-convergence by returning with
        # status 0 (max_nfev exhausted) rather than raising; the
        # unconverged iterate is not a fixed point and must not enter
        # the candidate pool, where its inflated x_L could win the
        # max-x_L picker over the genuine physical branch.
        return None
    residual_inf_norm = float(np.max(np.abs(sol.fun)))
    if residual_inf_norm > residual_ceiling_Hz:
        return None
    p_1, x_L, x_Rgt, x_Rlt = (float(v) for v in sol.x)
    # Box constraints guarantee p_1 ∈ [0, 1] and x_α ≥ 0, but the
    # multi-seed picker treats x_α = 0 as unphysical (would yield
    # log(0) for the chemical-potential plot). Reject as in
    # _solve_with_lm.
    if not (0.0 <= p_1 <= 1.0):
        return None
    if min(x_L, x_Rgt, x_Rlt) <= 0.0:
        return None
    return M25SteadyState(
        p_0=1.0 - p_1,
        p_1=p_1,
        x_L=x_L,
        x_Rgt=x_Rgt,
        x_Rlt=x_Rlt,
        residual_inf_norm=residual_inf_norm,
        n_function_evaluations=int(sol.nfev),
    )


def _candidate_satisfies_ordering(
    sol: M25SteadyState, expected_ordering: tuple[str, ...]
) -> bool:
    """Return True iff ``sol`` densities are sorted descending by name.

    ``expected_ordering`` is a tuple of density labels drawn from
    ``{"x_L", "x_Rgt", "x_Rlt"}`` listed in **descending** order, e.g.
    ``("x_Rlt", "x_L", "x_Rgt")`` enforces ``x_R< ≥ x_L ≥ x_R>``
    (the M25 large-asymmetry physical branch).
    """
    name_to_value = {
        "x_L": sol.x_L,
        "x_Rgt": sol.x_Rgt,
        "x_Rlt": sol.x_Rlt,
    }
    values = [name_to_value[name] for name in expected_ordering]
    return all(values[i] >= values[i + 1] for i in range(len(values) - 1))


def solve_rate_equation_steady_state_multi_seed(
    coefs: M25Coefficients,
    *,
    preferred_seed: np.ndarray | None = None,
    extra_seeds: list[np.ndarray] | None = None,
    branch_continuation_ratio: float = 5.0,
    accept_lm_convergence: bool = True,
    residual_tol_relative: float = 1e-3,
    max_function_evaluations: int = 500,
    branch_picker_mode: str = "min_residual",
    expected_ordering: tuple[str, ...] | None = None,
) -> M25SteadyState:
    """Pick the physical fixed point by multi-seed solve.

    Historical context: with the density equations run on the
    ensemble ``Γ̃`` rates (the pre-``cooper_pair_number_R`` behavior,
    still reproducible by leaving that field at 1.0), the M25 system
    appeared multi-stable — a float64-flat valley of pseudo-roots
    with residuals below the 1 Hz candidate ceiling. With the correct
    single-quasiparticle normalization (M25 text below Eq. 6) the
    M25 Fig 3/4 parameter set has a **unique physical root** at every
    temperature and all picker modes that filter on residual agree on
    it. This helper is retained for single-point solves and legacy
    bundles; temperature sweeps should use
    :func:`solve_rate_equation_branch`.

    Three branch-picker modes are supported (``branch_picker_mode``):

    * ``"min_residual"`` (default since the normalization fix):
      return the candidate with the smallest residual ∞-norm — the
      true fixed point beats any valley pseudo-root by many orders of
      magnitude. The previous default ``"max_x_L"`` selected sub-1-Hz
      pseudo-roots on the ``r x_L²`` slope (residual ~1e-3 Hz vs the
      ~1e-8 Hz physical source scale) and is no longer a defensible
      default.

    * ``"max_x_L"`` (DEPRECATED — emits ``DeprecationWarning``): return
      the candidate with the largest ``x_L``. Validated against early
      (pre-normalization-fix) M25 baselines but selects slope
      pseudo-roots admitted by the 1 Hz lm/lsq candidate ceiling; also
      documented to pick the wrong branch for large gap asymmetry
      (M25 Fig 3b/4b, ~600× parity-rate inflation). Kept only so
      historical baselines remain reproducible; scheduled for removal
      (2026-07-04 deep review, finding 10).

    * ``"lock_to_preferred"``: if ``preferred_seed`` converges to a
      positive-density branch, **always** return it (no max-x_L /
      branch_continuation_ratio escape hatch). For first-call no-seed
      cases, falls back to ``"min_residual"``. Pair with a low-T
      starting point and prev-T continuation to track a single branch
      across the sweep.

    ``preferred_seed`` exception (legacy): when in ``"max_x_L"`` mode,
    if the preferred-seed solution lies within ``branch_continuation_ratio``
    of the max-x_L candidate it is returned in preference to the max.
    The new modes ignore this ratio.

    ``expected_ordering`` (optional): a length-2 or length-3 tuple of
    density labels drawn from ``{"x_L", "x_Rgt", "x_Rlt"}`` listed in
    descending order (e.g. ``("x_Rlt", "x_L", "x_Rgt")`` for the
    M25 Fig 3b large-asymmetry physical branch ``x_R< ≥ x_L ≥ x_R>``).
    When set, candidates that violate the ordering are filtered out
    **before** the branch-picker dispatch. If no candidate satisfies
    the ordering, a stderr warning is emitted and the picker falls
    back to the unfiltered candidate set.

    Raises ``RuntimeError`` if no seed yields a physical solution.
    """
    seeds: list[np.ndarray | None] = [None]
    if preferred_seed is not None:
        seeds.append(preferred_seed)
    if extra_seeds is not None:
        seeds.extend(extra_seeds)
    seeds.extend(_default_seed_grid())

    candidates: list[tuple[np.ndarray | None, M25SteadyState]] = []
    for seed in seeds:
        # Primary: hybr via the validated solve_rate_equation_steady_state
        # path (with all its residual / bypass / unphysical-branch checks).
        try:
            sol = solve_rate_equation_steady_state(
                coefs,
                initial_guess=seed,
                accept_lm_convergence=accept_lm_convergence,
                residual_tol_relative=residual_tol_relative,
                max_function_evaluations=max_function_evaluations,
            )
        except RuntimeError:
            sol = None
        if sol is not None and sol.x_L > 0.0 and sol.x_Rgt > 0.0 and sol.x_Rlt > 0.0:
            candidates.append((seed, sol))

        # Secondary: lm as an additional candidate source. lm finds true
        # fixed points at the high-x_L end of M25's multi-stable manifold
        # where hybr stalls at the cancellation floor. We accept lm
        # candidates with residual below the same 1.0 Hz safety ceiling
        # used for hybr's no-progress bypass.
        if seed is None:
            continue  # lm and lsq need an explicit seed
        lm_sol = _solve_with_lm(coefs, seed, residual_ceiling_Hz=1.0)
        if lm_sol is not None:
            candidates.append((seed, lm_sol))

        # Tertiary: bounded least-squares (trust-region reflective).
        # Both hybr and lm are unbounded and can drift to negative
        # densities — when the post-hoc rejection fires the seed is
        # lost as a candidate source even though the physical fixed
        # point sits right at the seed (M25 Fig 3a small-asymmetry
        # case at low T). The bounded solver enforces ``p_1 ∈ [0, 1]``
        # and ``x_α ≥ 0`` during iteration, so the analytic-seed root
        # is recovered as a candidate. Same residual ceiling as lm.
        lsq_sol = _solve_with_lsq(coefs, seed, residual_ceiling_Hz=1.0)
        if lsq_sol is not None:
            candidates.append((seed, lsq_sol))
    if not candidates:
        raise RuntimeError(
            "M25 multi-seed solve: no seed produced a positive-density "
            "physical solution. Coefficients may be degenerate or the "
            "seed grid may not bracket the relevant branch."
        )

    # Density-ordering filter (optional). Applied before branch picker
    # dispatch so all picker modes (max_x_L, min_residual, lock_to_preferred)
    # see only candidates consistent with the paper's expected ordering.
    candidates_for_picker = candidates
    if expected_ordering is not None:
        valid_names = {"x_L", "x_Rgt", "x_Rlt"}
        if not all(name in valid_names for name in expected_ordering):
            raise ValueError(
                f"expected_ordering must contain only labels from "
                f"{sorted(valid_names)}; got {expected_ordering}"
            )
        if len(set(expected_ordering)) != len(expected_ordering):
            raise ValueError(
                f"expected_ordering must not repeat labels; got "
                f"{expected_ordering}"
            )
        filtered = [
            (seed, sol) for seed, sol in candidates
            if _candidate_satisfies_ordering(sol, expected_ordering)
        ]
        if filtered:
            candidates_for_picker = filtered
        else:
            import sys
            print(
                f"M25 multi-seed: no candidate satisfies "
                f"expected_ordering={expected_ordering}; falling back to "
                f"unfiltered candidate set ({len(candidates)} candidates).",
                file=sys.stderr,
            )

    # Branch picker dispatch.
    if branch_picker_mode == "lock_to_preferred":
        if preferred_seed is not None:
            for seed, sol in candidates_for_picker:
                if seed is preferred_seed:
                    return sol
        # Fall back to min_residual when no preferred seed converged.
        return min(
            candidates_for_picker, key=lambda c: c[1].residual_inf_norm,
        )[1]

    if branch_picker_mode == "min_residual":
        return min(
            candidates_for_picker, key=lambda c: c[1].residual_inf_norm,
        )[1]

    if branch_picker_mode != "max_x_L":
        raise ValueError(
            f"Unknown branch_picker_mode={branch_picker_mode!r}. "
            "Use 'min_residual' or 'lock_to_preferred' "
            "('max_x_L' is deprecated)."
        )

    # DEPRECATED (2026-07-04 deep review, finding 10): even on the
    # Γ̄-normalized single-root system, lm/lsq candidate pools admit
    # sub-1-Hz slope pseudo-roots and max-x_L selects them over the true
    # root (observed: x_L = 3.16e-6 at residual 6.5e-5 Hz picked over
    # x_L = 5.28e-8 at 2.5e-22 Hz). Kept only so historical baselines
    # remain reproducible; scheduled for removal.
    warnings.warn(
        "branch_picker_mode='max_x_L' is deprecated: it can select "
        "large-x_L pseudo-roots over the true fixed point (see "
        "docs/REVIEW-2026-07-04-deep-review.md, finding 10). Use "
        "'min_residual' (default) or 'lock_to_preferred'.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Legacy behavior: max-x_L with optional preferred-seed override.
    max_sol = max(candidates_for_picker, key=lambda c: c[1].x_L)[1]
    if preferred_seed is not None:
        for seed, sol in candidates_for_picker:
            if seed is preferred_seed and (
                sol.x_L * branch_continuation_ratio >= max_sol.x_L
            ):
                return sol
    return max_sol
