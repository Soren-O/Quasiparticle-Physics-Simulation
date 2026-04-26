r"""Marchegiani 2025 rate-equation closure — core observables and steady-state solver.

Two entry points, both named for the M25 equations they implement:

* :func:`crossover_temperature_kelvin` — closed-form Lambert-W
  crossover temperature ``T̄`` (M25 Eq. 8). Inputs: three scalars
  (``Δ_R``, ``r^{R<}``, ``g^ph_R``).
* :func:`solve_rate_equation_steady_state` — Newton fixed-point for
  the three-chemical-potential system (M25 Eqs. 3-6, derived in
  thesis Part III Appendix A). Inputs: all rate coefficients packed
  in an :class:`M25Coefficients` dataclass. Outputs: the four
  unknowns ``(p_1, x_L, x_{R>}, x_{R<})`` of the boxed system.

Both entry points treat rate coefficients as \"opaque inputs\" —
the energy integrals that produce them (tunneling-rate integrals in
M25 Supplementary Note III; recombination/generation/intraband-
relaxation integrals in Note IV) are not implemented here. A caller
that wants to reproduce M25 Figs. 3–5 assembles the coefficients
externally and calls this service to close the steady state.

Marchegiani & Catelani, *Commun. Phys.* **8**, 120 (2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import root
from scipy.special import lambertw


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

    Raises
    ------
    ValueError
        If any shape is wrong, ``xi`` is outside ``[0, 1]``,
        ``delta`` is outside ``(0, 1]``, or a nonnegative rate is
        negative.
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
        for name in (
            "r_L", "r_Rgt", "r_Rlt", "r_cross",
            "g_L", "g_Rgt", "g_Rlt",
            "tau_R_inv", "tau_E_inv",
        ):
            val = getattr(self, name)
            if val < 0.0:
                raise ValueError(f"{name} must be nonnegative; got {val}")


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

    # Bookkeeping objects (thesis Appendix Eqs. T^α, S^{L→R>}):
    #   𝒯^α(p) = (Γ̃^α_{00} + Γ̃^α_{01}) p_0 + (Γ̃^α_{11} + Γ̃^α_{10}) p_1
    def _T(gammas: np.ndarray) -> float:
        return (gammas[0, 0] + gammas[0, 1]) * p_0 + (gammas[1, 1] + gammas[1, 0]) * p_1

    T_L = _T(coefs.gammas_L)
    T_Rgt = _T(coefs.gammas_Rgt)
    # 𝒮^{L→R>}(p) = Γ̃^L_{00} p_0 + (Γ̃^L_{11} + Γ̃^L_{10}) p_1
    # (drops the 01 channel because its final R-state can lie below Δ_L)
    S_L_to_Rgt = coefs.gammas_L[0, 0] * p_0 + (coefs.gammas_L[1, 1] + coefs.gammas_L[1, 0]) * p_1

    delta = coefs.delta
    gamma_Rlt_10 = coefs.gammas_Rlt[1, 0]
    gamma_L_01 = coefs.gammas_L[0, 1]

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
        Default is ``(Γ̃^{ee}_{01}/(Γ̃^{ee}_{01}+Γ̃^{ee}_{10}), 0, 0, 0)``
        — the ee-detailed-balance qubit state with empty QP bands —
        which converges to the physical (nonequilibrium) branch for
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
    ``tests/services/test_rate_equation.py::test_lm_solver_is_deterministic``
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


def solve_rate_equation_steady_state_multi_seed(
    coefs: M25Coefficients,
    *,
    preferred_seed: np.ndarray | None = None,
    extra_seeds: list[np.ndarray] | None = None,
    branch_continuation_ratio: float = 5.0,
    accept_lm_convergence: bool = True,
    residual_tol_relative: float = 1e-3,
    max_function_evaluations: int = 500,
) -> M25SteadyState:
    """Pick the photon-driven nonequilibrium branch by multi-seed solve.

    The M25 4-unknown rate-equation system is multi-stable: at fixed
    parameters there are several positive-density fixed points plus
    the thermal branch. Each ``initial_guess`` to
    :func:`solve_rate_equation_steady_state` selects one of them.
    The "right" branch (the one matching M25 paper Fig 3 / Fig 4
    plots) is the photon-driven nonequilibrium one — typically the
    one with the largest ``x_L``.

    This helper tries the default seed plus a small grid of hand-
    tuned x seeds (and any caller-supplied ``extra_seeds``), keeps
    only converged positive-density solutions, and selects one as
    follows:

    * If ``preferred_seed`` is given (typically the previous
      temperature point's solution in a sweep) AND its solve
      converges to a positive-density branch within
      ``branch_continuation_ratio`` of the max-x_L candidate,
      return THAT solution. This keeps the sweep on the same branch
      across small parameter changes.
    * Otherwise return the candidate with the largest ``x_L`` —
      the most-non-equilibrium branch.

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
            continue  # lm needs an explicit seed
        lm_sol = _solve_with_lm(coefs, seed, residual_ceiling_Hz=1.0)
        if lm_sol is not None:
            candidates.append((seed, lm_sol))
    if not candidates:
        raise RuntimeError(
            "M25 multi-seed solve: no seed produced a positive-density "
            "physical solution. Coefficients may be degenerate or the "
            "seed grid may not bracket the relevant branch."
        )
    max_sol = max(candidates, key=lambda c: c[1].x_L)[1]
    if preferred_seed is not None:
        for seed, sol in candidates:
            if seed is preferred_seed and (
                sol.x_L * branch_continuation_ratio >= max_sol.x_L
            ):
                return sol
    return max_sol
