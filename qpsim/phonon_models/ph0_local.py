"""Ph0 local-phonon tier: steady-state ``n_ph(ω)`` from QP distribution.

At phonon steady state with a local bath of escape time ``τ_l``, the
Ph0 equation is

    0 = a_ph[f] + b_ph[f] · n_ph + (n_th − n_ph) / τ_l,

where ``(a_ph, b_ph)`` are the affine coefficients of the e-ph
source-sink on the QP distribution (see
:func:`qpsim.collisions.phonon.compute_phonon_source_sink`). Solving:

* ``τ_l = 0`` — **sentinel for no substrate coupling**: the escape term
  is dropped entirely and ``n_ph = −a_ph / b_ph``. Physically this is
  the ``τ_l → ∞`` limit of the escape term, *not* the instantaneous-
  escape limit the literal value suggests.
* ``τ_l > 0``: ``n_ph = (a_ph + n_th / τ_l) / (1/τ_l − b_ph)``.

Naming trap (audited 2026-06-10): the *opposite* limit — phonons pinned
at the bath, Fischer's instantaneous-thermalization ``τ_l → 0`` — is
spelled ``phonon_escape_time=None`` in
:func:`qpsim.services.steady_state.solve_steady_state` and
``use_thermal_phonons=True`` on the backend; it never reaches this
module. Passing the float ``0.0`` here therefore means the opposite of
passing ``None`` upstream.

Grid trap (review 2026-08-03, unfixed): the ω grid built by
:func:`qpsim.collisions.phonon.build_phonon_frequency_map` is the union
of the pair-difference lattice ``{k·dE}`` and the pair-sum lattice
``{2·E[0] + m·dE}``. On a uniform energy grid the two coincide only when
``2·E_min/dE`` is an integer; otherwise they are *disjoint* — every ω bin
then receives exactly one of the two channels, and this solver integrates
two decoupled phonon fields: difference bins driven by scattering and
escape only, sum bins by recombination/pair-breaking and escape only.
Difference-only bins above 2Δ therefore carry no Kaplan pair-breaking
sink at all, so substrate escape rather than pair breaking sets their
lifetime, and the coupled ``(f, n_ph)`` solution has two accumulation
points under grid refinement. Nothing here or upstream tests the
condition, and the affine balance below is satisfied to roundoff either
way, so it is silent. The promoted paper grids happen to satisfy it, as
does any ``E_min = 0`` grid (ratio 0); ``E_min = Δ`` with NE = 400 out to
10Δ (ratio 88.9), NE = 101 out to 5Δ (50.5) and NE = 64 out to 4Δ (42.7)
does not. Check ``2·E_min/dE`` before trusting a dynamic-phonon run,
and prefer a refinement ladder that holds it integral. The bath-pinned path
(``phonon_escape_time=None`` upstream) never reaches this module and is
unaffected.

Moved here from the private ``_phonon_steady_state`` helper that
previously lived in :mod:`qpsim.services.steady_state` (Gate 2
task 11).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from qpsim.collisions.phonon import compute_phonon_source_sink
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext


@dataclass(frozen=True)
class PhononBalanceDiagnostics:
    """Raw and representability-aware diagnostics for the affine Ph0 balance.

    ``raw_backward_error`` evaluates the physical three-term equation directly.
    ``certified_backward_error`` removes only the residual that can be caused by
    rounding the exact affine root to the nearest representable occupation (plus
    a conservative floating-arithmetic allowance).  Keeping both values makes a
    genuine imbalance distinguishable from an unrepresentable bath correction.
    """

    residual: np.ndarray
    physical_scale: np.ndarray
    representability_allowance: np.ndarray
    raw_backward_error: float
    certified_backward_error: float


def _normwise_error(residual_magnitude: np.ndarray, scale: np.ndarray) -> float:
    residual_magnitude = np.asarray(residual_magnitude, dtype=float)
    scale = np.asarray(scale, dtype=float)
    if np.any(~np.isfinite(residual_magnitude)) or np.any(~np.isfinite(scale)):
        raise ValueError("Residual magnitudes and scales must be finite.")
    common = float(
        max(
            np.max(residual_magnitude, initial=0.0),
            np.max(scale, initial=0.0),
        )
    )
    if common == 0.0:
        return 0.0
    denominator = float(np.sum(scale / common))
    numerator = float(np.sum(residual_magnitude / common))
    return numerator / denominator if denominator > 0.0 else float("inf")


def _half_ulp(values: np.ndarray) -> np.ndarray:
    """Half the larger adjacent binary64 spacing, including zero."""
    values = np.asarray(values, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        up_spacing = np.nextafter(values, np.inf) - values
        down_spacing = values - np.nextafter(values, -np.inf)
    up_spacing = np.where(np.isfinite(up_spacing), up_spacing, 0.0)
    down_spacing = np.where(np.isfinite(down_spacing), down_spacing, 0.0)
    spacing = np.maximum(up_spacing, down_spacing)
    half = 0.5 * spacing
    # Half the smallest subnormal underflows to zero; one whole subnormal is a
    # conservative representability bound at that single boundary case.
    return np.where((spacing > 0.0) & (half == 0.0), spacing, half)


def _two_sum_error(
    left: np.ndarray,
    right: np.ndarray,
    rounded_sum: np.ndarray,
) -> np.ndarray:
    """Exact residual of a non-overflowing binary64 addition (TwoSum)."""
    virtual_right = rounded_sum - left
    return (left - (rounded_sum - virtual_right)) + (right - virtual_right)


def phonon_balance_diagnostics(
    n_ph: np.ndarray,
    a_ph: np.ndarray,
    b_ph: np.ndarray,
    n_th: np.ndarray,
    *,
    tau_l: float,
) -> PhononBalanceDiagnostics:
    """Certify the nearest representable solution of the affine Ph0 balance.

    The physical equation is

    ``a_ph + b_ph*n_ph + (n_th - n_ph)/tau_l = 0``.

    At weak drive the required displacement from ``n_th`` can be much smaller
    than one binary64 ULP of ``n_th``.  The nearest floating-point occupation is
    then the best representable solution even though direct evaluation of the
    bath subtraction has an O(1) *relative* residual against the tiny e-ph
    terms.  Per bin, this routine bounds that unavoidable residual by half the
    larger adjacent ULP times the magnitude of the affine slope.  A small,
    explicit arithmetic bound covers the two fused multiply-add evaluations.

    The allowance is disabled when the float-input affine equation has no
    non-negative finite root.  Thus a clipped negative occupation cannot be
    certified as a physical fixed point.  ``tau_l=0`` retains qpsim's established
    no-substrate sentinel and certifies ``a_ph + b_ph*n_ph = 0``.
    """
    n = np.asarray(n_ph, dtype=float)
    a = np.asarray(a_ph, dtype=float)
    b = np.asarray(b_ph, dtype=float)
    nth = np.asarray(n_th, dtype=float)
    if not (n.shape == a.shape == b.shape == nth.shape):
        raise ValueError(
            "n_ph, a_ph, b_ph, and n_th must have the same shape; got "
            f"{n.shape}, {a.shape}, {b.shape}, and {nth.shape}."
        )
    if np.any(~np.isfinite(np.stack((n, a, b, nth), axis=0))):
        raise ValueError("Phonon balance terms must contain only finite values.")
    if np.any(n < 0.0) or np.any(nth < 0.0):
        raise ValueError("Phonon occupations must be non-negative.")
    if not np.isfinite(tau_l) or tau_l < 0.0:
        raise ValueError(f"tau_l must be finite and non-negative; got {tau_l}.")

    residual = np.empty_like(n)
    arithmetic_allowance = np.zeros_like(n)
    assembly_allowance = np.zeros_like(n)
    if tau_l == 0.0:
        with np.errstate(over="ignore", invalid="ignore"):
            driven = b * n
        bath = np.zeros_like(n)
        slope = b
        numerator = -a
        denominator = b
        if np.any(
            ~np.isfinite(np.stack((driven, slope, numerator, denominator), axis=0))
        ):
            raise ValueError("Derived phonon balance terms must be finite.")
        for index in np.ndindex(n.shape):
            residual[index] = math.fma(
                float(b[index]),
                float(n[index]),
                float(a[index]),
            )
        numerator_exact = numerator
        denominator_exact = denominator
        arithmetic_allowance = _half_ulp(residual)
    else:
        inv_tau = 1.0 / tau_l
        if not np.isfinite(inv_tau):
            raise ValueError(
                "1/tau_l must be representable as a finite float; "
                f"got tau_l={tau_l}."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            driven = b * n
            bath_difference = nth - n
            bath = inv_tau * bath_difference
            slope = b - inv_tau
            thermal_product = inv_tau * nth
            numerator = a + thermal_product
            denominator = inv_tau - b
        if np.any(
            ~np.isfinite(
                np.stack(
                    (driven, bath, slope, numerator, denominator),
                    axis=0,
                )
            )
        ):
            raise ValueError("Derived phonon balance terms must be finite.")

        product_error = np.empty_like(n)
        qp_terms = np.empty_like(n)
        for index in np.ndindex(n.shape):
            product_error[index] = math.fma(
                inv_tau,
                float(nth[index]),
                -float(thermal_product[index]),
            )
            qp_terms[index] = math.fma(
                float(b[index]),
                float(n[index]),
                float(a[index]),
            )
            residual[index] = math.fma(
                inv_tau,
                float(bath_difference[index]),
                float(qp_terms[index]),
            )

        numerator_error = product_error + _two_sum_error(
            a,
            thermal_product,
            numerator,
        )
        denominator_error = _two_sum_error(
            np.full_like(b, inv_tau),
            -b,
            denominator,
        )
        bath_difference_error = _two_sum_error(
            nth,
            -n,
            bath_difference,
        )
        numerator_exact = np.empty_like(numerator)
        denominator_exact = np.empty_like(denominator)
        try:
            for index in np.ndindex(n.shape):
                numerator_exact[index] = math.fsum(
                    (float(numerator[index]), float(numerator_error[index]))
                )
                denominator_exact[index] = math.fsum(
                    (float(denominator[index]), float(denominator_error[index]))
                )
        except OverflowError as error:
            raise ValueError(
                "Exact reconstructed phonon balance coefficients overflow."
            ) from error
        assembly_allowance = (
            np.abs(numerator_error) + np.abs(n) * np.abs(denominator_error)
        )
        arithmetic_allowance = (
            np.abs(inv_tau * bath_difference_error)
            + _half_ulp(qp_terms)
            + _half_ulp(residual)
        )

    with np.errstate(over="ignore", invalid="ignore"):
        physical_scale = np.abs(a) + np.abs(driven) + np.abs(bath)
    if np.any(~np.isfinite(physical_scale)):
        raise ValueError("Assembled phonon physical scale must be finite.")
    raw_backward_error = _normwise_error(np.abs(residual), physical_scale)

    # The exact affine root is numerator / denominator in both branches.  Test
    # its sign without forming the quotient, which could under/overflow.  A
    # zero numerator is the physical root n=0; a zero denominator is singular.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        affine_root = numerator_exact / denominator_exact
    nonnegative_sign = (numerator_exact == 0.0) | (
        np.signbit(numerator_exact) == np.signbit(denominator_exact)
    )
    finite_nonnegative_root = (
        np.isfinite(affine_root)
        & (affine_root >= 0.0)
        & nonnegative_sign
    )

    representability = _half_ulp(n) * np.abs(denominator_exact)
    representability = np.where(finite_nonnegative_root, representability, 0.0)

    with np.errstate(over="ignore", invalid="ignore"):
        total_allowance = (
            representability + assembly_allowance + arithmetic_allowance
        )
    if np.any(finite_nonnegative_root & ~np.isfinite(total_allowance)):
        raise ValueError("Phonon representability allowance must be finite.")
    total_allowance = np.where(
        finite_nonnegative_root,
        total_allowance,
        0.0,
    )
    excess = np.maximum(
        np.abs(residual) - total_allowance,
        0.0,
    )
    certified_backward_error = _normwise_error(excess, physical_scale)
    return PhononBalanceDiagnostics(
        residual=residual,
        physical_scale=physical_scale,
        representability_allowance=total_allowance,
        raw_backward_error=raw_backward_error,
        certified_backward_error=certified_backward_error,
    )


def _solve_affine_balance(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    label: str,
) -> np.ndarray:
    """Solve ``denominator * n = numerator`` and reject unphysical states."""
    singular = denominator == 0.0
    inconsistent = singular & (numerator != 0.0)
    if np.any(inconsistent):
        bad = np.flatnonzero(inconsistent)[:5].tolist()
        raise RuntimeError(
            f"Ph0 phonon steady state has no finite solution in {label}; "
            f"singular balance at omega indices {bad}."
        )

    n_ph = np.zeros_like(numerator, dtype=float)
    regular = ~singular
    n_ph[regular] = numerator[regular] / denominator[regular]

    negative_exact_root = regular & (numerator != 0.0) & (
        np.signbit(numerator) != np.signbit(denominator)
    )
    invalid = (~np.isfinite(n_ph)) | (n_ph < 0.0) | negative_exact_root
    if np.any(invalid):
        bad = np.flatnonzero(invalid)[:5].tolist()
        raise RuntimeError(
            f"Ph0 phonon steady state is unphysical in {label}; "
            f"computed negative or non-finite occupation at omega indices {bad}. "
            "This indicates phonon runaway or no non-negative fixed point."
        )

    return n_ph


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
    *,
    K_s0_phonon_side: np.ndarray | None = None,
    K_r0_phonon_side: np.ndarray | None = None,
    coefficients_out: dict[str, np.ndarray] | None = None,
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
        for ``ctx.E``. Unvalidated precondition: the two index sets are
        disjoint unless ``2·E_min/dE`` is an integer, which silently removes
        the pair-breaking channel above 2Δ — see the module docstring's grid
        trap.
    T_bath
        Substrate bath temperature (K), used to compute the thermal
        ``n_th(ω)``.
    tau_l
        Phonon bath-escape time (ns). ``0.0`` is the **no-substrate
        sentinel** (escape term dropped — the ``τ_l → ∞`` physics
        limit); ``> 0`` means finite escape time. For bath-pinned
        phonons (the physical ``τ_l → 0`` limit) use the upstream
        thermal-phonon path instead of this solver (see module
        docstring).
    K_s0_phonon_side, K_r0_phonon_side
        **Opt-in** phonon-side scattering and recombination/pair-breaking
        kernels ``2K⁻/(π Δ τ_0^PB)`` and ``K⁺/(π Δ τ_0^PB)`` (build via
        :func:`qpsim.collisions.phonon.build_scattering_kernel_phonon_side` and
        :func:`qpsim.collisions.phonon.build_recombination_kernel_phonon_side`).
        Forwarded to :func:`compute_phonon_source_sink`; when supplied,
        the phonon-equation rates use the F&C 2023 Eq. 12 prefactors
        instead of the QP-side ``K_s0`` / ``K_r0``. ``None`` (default)
        preserves legacy behavior bit-for-bit.
    coefficients_out
        Optional diagnostics mapping populated with the already assembled
        affine coefficients ``"a_ph"`` and ``"b_ph"``.  This lets the Picard
        orchestrator certify the physical phonon balance at a candidate fixed
        point without repeating the O(NE^2) source/sink contraction.

    Returns
    -------
    n_ph
        1D array of shape ``(len(omega_bins),)``. A negative or non-finite
        affine root is rejected rather than clipped.
    """
    n_omega = len(omega_bins)
    a_ph, b_ph = compute_phonon_source_sink(
        f, ctx, K_s0, K_r0,
        omega_idx_diff, omega_idx_sum, diff_sign, n_omega,
        K_s0_phonon_side=K_s0_phonon_side,
        K_r0_phonon_side=K_r0_phonon_side,
    )
    if coefficients_out is not None:
        coefficients_out["a_ph"] = a_ph
        coefficients_out["b_ph"] = b_ph

    if tau_l < 0.0:
        raise ValueError("tau_l must be non-negative.")

    if tau_l == 0.0:
        n_ph = _solve_affine_balance(
            -a_ph, b_ph, label="tau_l=0 no-bath branch"
        )
    else:
        inv_tau_l = 1.0 / tau_l
        n_th = thermal_phonon_occupation(omega_bins, T_bath)
        denom = inv_tau_l - b_ph
        n_ph = _solve_affine_balance(
            a_ph + inv_tau_l * n_th,
            denom,
            label="finite-tau_l branch",
        )

    return n_ph
