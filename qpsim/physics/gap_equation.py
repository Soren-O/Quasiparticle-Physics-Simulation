"""BCS gap self-consistency: equilibrium calibration + reference-subtracted runtime solve.

Calibration (once per T_c and T_bath) computes Δ_eq and caches 1/λ and ω_D.
The pairing constant is fixed by the linearized finite-cutoff equation at
``T_c`` so that the superconducting solution vanishes continuously there.
An optional measured material ``Delta_0`` is retained as diagnostic metadata;
it cannot independently anchor the same weak-coupling model.
The runtime solver then takes an occupation ``f`` and returns the
current Δ via Brent's method on the reference-subtracted residual.

Note on naming: the old repo called the runtime occupation parameter
``f_L`` even though the function body computes ``1 − 2 f`` from it
(i.e. it actually expects the Fermi-Dirac occupation ``f``, not the
longitudinal combination ``f_L = 1 − 2f``). Renamed to ``f`` here.

Ported from the old ``qpsim/numerics/gap_equation.py`` at Gate 2.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.physics.bcs_quadrature import cell_edges_from_widths


class GapBelowGridSupportError(ValueError):
    """The SOLVED gap fell below the represented energy-grid support edge.

    Raised when :func:`solve_gap`'s accepted root — or its normal-state
    ``Delta = 0`` decision — lies below the reconstructed first cell face,
    i.e. the physical solution left the domain the kinetic grid can
    represent. On every shipped BCS grid (first face > 0) this is the
    reachable form of superconducting collapse, and callers running
    self-consistent sweeps classify it as such (the literal ``<= 0``
    return is unreachable on those grids — 2026-07-19 audit). Input
    validation of a caller-supplied ``reference_gap`` deliberately raises
    plain :class:`ValueError` instead: a bad anchor is misuse, not
    collapse. Subclasses ``ValueError`` so existing handlers keep
    working.
    """


@dataclass
class GapCalibration:
    """Frozen equilibrium constants for the reference-subtracted gap solver.

    Produced once by :func:`calibrate_gap`; consumed by :func:`solve_gap`.
    Leading-underscore fields are private implementation details;
    ``delta_eq`` / ``delta_0_bcs`` / ``delta_0_reference`` / ``T_c`` /
    ``T_bath`` are the user-facing values.
    """

    delta_eq: float  # Equilibrium gap at T_bath (μeV)
    delta_0_bcs: float  # T=0 gap implied by T_c and the finite cutoff (μeV)
    delta_0_reference: float | None  # Optional measured Δ0; diagnostic only
    T_c: float  # Critical temperature (K)
    T_bath: float  # Bath temperature (K)
    _ref_integral: float  # Linearized T_c integral = 1/λ
    _omega_D: float  # Debye cutoff (μeV)
    _inv_lambda: float  # 1/λ


def _fermi_dirac(E: np.ndarray | float, T: float) -> np.ndarray | float:
    """f_FD(E, T) = 1 / (exp(E/kT) + 1). Zero-T limit is a step at E=0."""
    if T <= 0:
        return np.where(np.asarray(E) > 0, 0.0, 1.0)
    kT = _KB_UEV_PER_K * T
    exponent = np.minimum(np.asarray(E) / kT, 500.0)
    return 1.0 / (np.exp(exponent) + 1.0)


def _gap_integral_cosh(delta: float, T: float, omega_D: float, n_quad: int) -> float:
    """∫₀^{u_max} [1 − 2 f_FD(Δ cosh u, T)] du.

    Uses E = Δ cosh u to remove the 1/√(E² − Δ²) singularity at the gap edge.
    """
    if delta <= 0:
        return 0.0
    u_max = np.arccosh(omega_D / delta) if omega_D > delta else 0.0
    if u_max <= 0:
        return 0.0
    u = np.linspace(0, u_max, n_quad + 1)
    E_u = delta * np.cosh(u)
    f_u = _fermi_dirac(E_u, T)
    integrand = 1.0 - 2.0 * f_u
    return float(np.trapezoid(integrand, u))


def _linearized_gap_integral(T: float, omega_D: float) -> float:
    r"""Return the finite-cutoff ``Δ -> 0`` BCS gap integral.

    The linearized equation is

    ``1/λ = ∫_0^ωD tanh(E / (2 k_B T)) / E dE``.

    Scaling by ``k_B T`` removes units and the apparent singularity has the
    finite limit ``1/2`` at the origin.  Adaptive quadrature is used here so
    the coupling does not inherit the ``O(n_quad^-2)`` endpoint error of a
    uniform trapezoid.  That accuracy is essential near ``T_c``, where even a
    small offset in ``1/λ`` creates a spurious finite limiting gap.
    """
    if not np.isfinite(T) or T <= 0.0:
        raise ValueError("T must be finite and positive for the linearized gap integral.")
    if not np.isfinite(omega_D) or omega_D <= 0.0:
        raise ValueError("omega_D must be finite and positive.")

    from scipy.integrate import quad

    scaled_cutoff = omega_D / (_KB_UEV_PER_K * T)

    def integrand(x: float) -> float:
        if x == 0.0:
            return 0.5
        return float(np.tanh(0.5 * x) / x)

    value, _error = quad(
        integrand,
        0.0,
        scaled_cutoff,
        epsabs=5e-13,
        epsrel=5e-13,
        limit=200,
    )
    return float(value)


def _gap_integral_f(
    delta: float,
    f: np.ndarray,
    E_bins: np.ndarray,
    omega_D: float,
    dE_bins: np.ndarray | None = None,
) -> float:
    """Cell-exact finite-volume gap integral for an occupation grid.

    The vacuum contribution is retained analytically as
    ``acosh(omega_D / delta)``.  The finite-occupation correction treats each
    stored ``f_i`` as constant over its finite-volume cell and integrates
    ``1/sqrt(E**2 - delta**2)`` analytically over the cell intersection with
    ``[delta, omega_D]``.  No off-grid interpolation of ``f`` is performed.

    ``dE_bins=None`` derives the standard midpoint widths from ``E_bins`` for
    backward compatibility.  State-backed callers should pass their actual
    spectral widths.
    """
    if delta <= 0:
        return 0.0
    if omega_D <= delta:
        return 0.0
    E = np.asarray(E_bins, dtype=float).reshape(-1)
    widths = (
        integration_widths_from_centers(E)
        if dE_bins is None
        else np.asarray(dE_bins, dtype=float).reshape(-1)
    )
    edges = cell_edges_from_widths(E, widths)
    return _gap_integral_f_from_edges(
        delta,
        np.asarray(f, dtype=float).reshape(-1),
        edges,
        omega_D,
    )


def _gap_integral_f_from_edges(
    delta: float,
    f: np.ndarray,
    cell_edges: np.ndarray,
    omega_D: float,
) -> float:
    """Evaluate :func:`_gap_integral_f` from prevalidated cell edges."""

    def primitive(energy: np.ndarray | float) -> np.ndarray:
        energy_array = np.asarray(energy, dtype=float)
        # asinh(sqrt(E²-Δ²)/Δ) == acosh(E/Δ), while the factored
        # radicand remains accurate for a cell face arbitrarily close to Δ.
        xi = np.sqrt(
            np.maximum(
                (energy_array - delta) * (energy_array + delta),
                0.0,
            )
        )
        return np.arcsinh(xi / delta)

    vacuum = float(np.arccosh(omega_D / delta))
    lo = np.maximum(cell_edges[:-1], delta)
    hi = np.minimum(cell_edges[1:], omega_D)
    hi = np.maximum(hi, lo)

    # Preserve the historical constant-left extrapolation when a candidate
    # gap falls below the represented support.  solve_gap warns about this
    # unsupported geometry; assigning the missing interval to the first cell
    # keeps the result backward-compatible and makes that assumption explicit.
    if delta < float(cell_edges[0]):
        lo[0] = delta
        hi[0] = max(delta, min(float(cell_edges[1]), omega_D))

    occupation_weights = primitive(hi) - primitive(lo)
    return vacuum - 2.0 * float(np.dot(f, occupation_weights))


def calibrate_gap(
    T_c: float,
    T_bath: float,
    *,
    Delta_0: float | None = None,
    omega_D_over_Tc: float = 100.0,
    n_quadrature: int = 512,
    xtol: float | None = None,
) -> GapCalibration:
    """Compute Δ_eq(T_bath) and cache 1/λ and ω_D for the runtime solver.

    ``T_c`` is the authoritative pairing-scale input.  The coupling is derived
    from the finite-cutoff linearized equation at ``T_c`` and the equilibrium
    gap is then solved with the ``E = Δ cosh(u)`` substitution.  This guarantees
    ``Δ_eq -> 0`` continuously as ``T_bath -> T_c`` from below.

    ``Delta_0`` optionally records a measured material zero-temperature gap as
    ``GapCalibration.delta_0_reference`` for provenance and model diagnostics.
    It deliberately does *not* change ``1/λ`` or ``Δ_eq``.  A one-coupling
    weak-coupling BCS kernel cannot in general reproduce both an arbitrary
    measured ``Delta_0`` and an independently declared ``T_c``; doing so would
    move the model's critical temperature and create a discontinuity when the
    declared ``T_c`` is imposed.  The model prediction is exposed separately as
    ``GapCalibration.delta_0_bcs``.

    ``xtol`` (μeV) overrides the default brentq tolerance of ``1e-6 * kBTc``.
    Tighten when the caller needs ``δΔ_T = Δ_0 − Δ_eq`` resolved well below
    the default ~1e-4 μeV — e.g. the Fischer 2023 Fig 6 observable, which
    divides by an exponentially small δΔ_T at T_B ≪ T_c.
    """
    if not np.isfinite(T_c) or T_c <= 0:
        raise ValueError("T_c must be positive and finite.")
    if not np.isfinite(T_bath) or T_bath < 0:
        raise ValueError("T_bath must be non-negative and finite.")
    if Delta_0 is not None and (not np.isfinite(Delta_0) or Delta_0 <= 0.0):
        raise ValueError("Delta_0 must be finite and positive when supplied.")
    if not np.isfinite(omega_D_over_Tc) or omega_D_over_Tc <= 0.0:
        raise ValueError("omega_D_over_Tc must be finite and positive.")
    if (
        isinstance(n_quadrature, (bool, np.bool_))
        or not isinstance(n_quadrature, (int, np.integer))
        or n_quadrature < 2
    ):
        raise ValueError("n_quadrature must be an integer >= 2.")
    if xtol is not None and (not np.isfinite(xtol) or xtol <= 0.0):
        raise ValueError("xtol must be finite and positive when supplied.")

    kBTc = _KB_UEV_PER_K * T_c
    omega_D = omega_D_over_Tc * kBTc
    # The linearized equation makes the declared T_c the actual critical
    # temperature of this finite-cutoff BCS model.  Anchoring 1/λ at an
    # independent measured Δ0 instead would generally imply a different T_c.
    inv_lambda = _linearized_gap_integral(T_c, omega_D)
    if inv_lambda <= 0:
        raise RuntimeError("Failed to compute 1/λ from the linearized T_c equation.")

    # At T=0 the gap equation is acosh(ω_D/Δ0) = 1/λ, so the model's
    # zero-temperature gap follows analytically once the coupling is fixed.
    delta_0_bcs = omega_D / np.cosh(inv_lambda)

    if T_bath >= T_c:
        delta_eq = 0.0
    elif T_bath <= 0:
        delta_eq = delta_0_bcs
    else:
        from scipy.optimize import brentq

        def residual(delta: float) -> float:
            if delta == 0.0:
                return _linearized_gap_integral(T_bath, omega_D) - inv_lambda
            return _gap_integral_cosh(delta, T_bath, omega_D, n_quadrature) - inv_lambda

        xtol_brentq = 1e-6 * kBTc if xtol is None else xtol
        delta_eq = brentq(residual, 0.0, delta_0_bcs, xtol=xtol_brentq)

    return GapCalibration(
        delta_eq=float(delta_eq),
        delta_0_bcs=float(delta_0_bcs),
        delta_0_reference=None if Delta_0 is None else float(Delta_0),
        T_c=T_c,
        T_bath=T_bath,
        _ref_integral=inv_lambda,
        _omega_D=omega_D,
        _inv_lambda=inv_lambda,
    )


def solve_gap(
    calibration: GapCalibration,
    f: np.ndarray,
    E_bins: np.ndarray,
    dE_bins: np.ndarray | None = None,
    *,
    bracket_factor: float = 0.5,
    reference_gap: float | None = None,
    xtol: float | None = None,
    allow_gap_edge_extrapolation: bool = False,
) -> float:
    """Runtime gap solve from the Fermi-Dirac occupation ``f(E)``.

    Solves ``∫_Δ^{ω_D} (1 − 2f(E))/√(E² − Δ²) dE − 1/λ = 0`` using
    Brent's method.  When ``reference_gap`` is supplied, a bounded adaptive
    scan isolates sampled sign-change intervals near that reference and the
    nearest detected root is selected, providing deterministic continuation
    for a non-equilibrium residual with multiple sign-changing roots. Tangent
    (even-multiplicity) roots do not define a sign-change branch and are not
    detected by this continuation scan.
    Returns 0 if no superconducting solution is found (normal state).

    Parameters
    ----------
    calibration
        From :func:`calibrate_gap`.
    f
        Fermi-Dirac occupation on ``E_bins``, shape ``(NE,)``.
    E_bins
        Energy bin centers, shape ``(NE,)``.
    dE_bins
        Optional finite-volume cell widths, shape ``(NE,)``.  When omitted,
        standard midpoint widths are derived from ``E_bins``.  Pass explicit
        widths for one-bin or nonstandard state grids.
    bracket_factor
        Initial half-width of the legacy search bracket as a fraction of Δ_eq.
        This is used only when ``reference_gap`` is omitted.  It cannot select
        the branch of a reference-anchored solve.
    reference_gap
        Optional previous/current gap in μeV.  For a multi-root non-equilibrium
        residual, select the nearest detected sign-changing root. Time-dependent
        and outer self-consistency loops should pass their current gap so the
        solution remains on a continuous branch. ``None`` retains the legacy
        bracketed solve, with a loud ambiguity diagnostic if nearby
        sign-changing multiplicity is found.
    xtol
        brentq absolute tolerance in μeV. Default ``1e-6 * delta_eq`` matches
        legacy behavior; tighten when the caller's observable depends on
        sub-default-precision shifts in Δ (e.g. fig6_paper at low T_B).
    allow_gap_edge_extrapolation
        By default, fail if the selected gap (including a numerical normal
        state) lies below the reconstructed first energy-cell face. ``True``
        explicitly restores the historical constant-left occupation
        extrapolation and emits a runtime warning. Quantitative state-backed
        solves should extend the grid instead.
    """
    delta_eq = calibration.delta_eq
    if reference_gap is None and (not np.isfinite(bracket_factor) or bracket_factor <= 0.0):
        raise ValueError("bracket_factor must be finite and positive.")
    if reference_gap is not None and (not np.isfinite(reference_gap) or reference_gap <= 0.0):
        raise ValueError("reference_gap must be finite and positive when supplied.")
    if xtol is not None and (not np.isfinite(xtol) or xtol <= 0.0):
        raise ValueError("xtol must be finite and positive when supplied.")
    if not isinstance(allow_gap_edge_extrapolation, bool):
        raise TypeError("allow_gap_edge_extrapolation must be a bool.")

    ref_integral = calibration._ref_integral
    E = np.asarray(E_bins, dtype=float).ravel()
    f_arr = np.asarray(f, dtype=float).ravel()
    if E.size == 0:
        raise ValueError("E_bins must be non-empty.")
    if E.shape != f_arr.shape:
        raise ValueError(f"f and E_bins must have the same shape; got {f_arr.shape} and {E.shape}.")
    if np.any(~np.isfinite(E)) or np.any(~np.isfinite(f_arr)):
        raise ValueError("f and E_bins must contain only finite values.")
    if np.any((f_arr < 0.0) | (f_arr > 1.0)):
        raise ValueError("f must contain physical occupations in [0, 1].")
    if np.any(np.diff(E) <= 0.0):
        raise ValueError("E_bins must be strictly increasing.")
    widths = (
        integration_widths_from_centers(E)
        if dE_bins is None
        else np.asarray(dE_bins, dtype=float).ravel()
    )
    cell_edges = cell_edges_from_widths(E, widths)
    if delta_eq <= 0:
        return 0.0
    omega_D = calibration._omega_D

    # The gap integral runs to ω_D but takes f=0 above the final cell edge.
    # If the occupation has not decayed at the top of the grid, that tail is
    # silently dropped and the solved gap depends on E_max. Warn — mirroring
    # the low-edge support check below — so a mis-sized grid does not silently
    # under-integrate. (Physical, decaying f: no warning.)
    if float(cell_edges[-1]) < omega_D and float(f_arr[-1]) > 1e-5:
        warnings.warn(
            "solve_gap: occupation at the top of the energy grid "
            f"f(E_max={float(E[-1]):.6g} μeV)={float(f_arr[-1]):.3g} is non-negligible "
            f"while the final cell edge {float(cell_edges[-1]):.6g} μeV is below "
            f"ω_D={omega_D:.6g} μeV; the gap integral takes f=0 above the grid, "
            "so the result may depend on grid extent. Extend the energy grid until "
            "the high-energy tail has decayed.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Resolution guard (2026-07-19 audit): the solved gap inherits a
    # cell-constant discretization error ~O(dE/Δ_eq) that the flattening
    # residual slope near T_c amplifies dramatically (measured ~175x
    # amplification between T/T_c = 0.25 and 0.95: a dE ≈ 0.31·Δ_eq grid
    # gave a silent +4.7% gap error at T/T_c = 0.95). This is distinct
    # from the below-gap-support warning: the grid can cover the full
    # support and still be too coarse for the collapsing gap scale. Warn
    # on coarse-relative-to-Δ_eq cells; shipped grids (dE ≲ 0.03·Δ_eq)
    # never trigger.
    edge_idx = min(int(np.searchsorted(E, delta_eq)), widths.size - 1)
    edge_width = float(widths[edge_idx])
    if edge_width > 0.25 * delta_eq:
        warnings.warn(
            "solve_gap: the energy-cell width near the gap edge "
            f"(dE={edge_width:.6g} μeV) is {edge_width / delta_eq:.0%} of the "
            f"equilibrium gap Δ_eq={delta_eq:.6g} μeV "
            f"(T_bath/T_c={calibration.T_bath / calibration.T_c:.2f}). The "
            "solved gap carries a cell-constant discretization error of this "
            "relative order, and the flattening residual slope near T_c "
            "amplifies it to percent level. Refine the energy grid near the "
            "gap edge before trusting the solved gap.",
            RuntimeWarning,
            stacklevel=2,
        )

    lower_support_edge = float(cell_edges[0])

    def enforce_grid_support(candidate: float, *, solved_state: bool = True) -> None:
        # Coverage is a geometric input contract, not a root-accuracy test.
        # Accept only a sub-part-per-billion mismatch between two boundaries
        # that are numerically intended to coincide.  The fixed geometric
        # tolerance is deliberately independent of a caller-supplied (and
        # potentially very large) ``xtol`` so root accuracy cannot silently
        # opt into materially unsampled gap-edge support.
        # ``solved_state=True`` marks a SOLVED gap (accepted root or the
        # normal-state decision): failing support then raises the
        # classified GapBelowGridSupportError so self-consistent callers
        # can fold it into their collapse handling. ``solved_state=False``
        # is input validation of a caller anchor and stays a plain
        # ValueError.
        scale = max(abs(candidate), abs(lower_support_edge))
        support_tol = max(
            64.0 * np.finfo(float).eps * max(scale, 1.0),
            1e-9 * scale,
        )
        if candidate < lower_support_edge - support_tol:
            message = (
                "solve_gap: candidate gap "
                f"Δ={candidate:.12g} μeV lies below the reconstructed "
                f"energy-grid support edge {lower_support_edge:.12g} μeV "
                f"by {lower_support_edge - candidate:.12g} μeV. The gap "
                "integral is therefore extrapolating f across unsampled "
                "gap-edge support; use an energy grid extending below the "
                "minimum candidate gap."
            )
            if not allow_gap_edge_extrapolation:
                error_type = (
                    GapBelowGridSupportError if solved_state else ValueError
                )
                raise error_type(
                    message
                    + " Set allow_gap_edge_extrapolation=True only to opt "
                    "into the historical constant-left extrapolation."
                )
            warnings.warn(
                message
                + " Constant-left extrapolation was explicitly enabled by "
                "the caller.",
                RuntimeWarning,
                stacklevel=3,
            )

    def residual(delta: float) -> float:
        return (
            _gap_integral_f_from_edges(
                delta,
                f_arr,
                cell_edges,
                omega_D,
            )
            - ref_integral
        )

    from scipy.optimize import brentq

    xtol_brentq = 1e-6 * delta_eq if xtol is None else xtol
    # Keep the residual strictly inside Δ > 0 without imposing an absolute
    # μeV scale.  The first term stays well clear of overflow in ω_D/Δ; the
    # second remains far below both the model gap and Brent resolution.  This
    # preserves low-gap models and the continuous Δ -> 0 branch near T_c.
    floating_floor = 4.0 * (omega_D / np.finfo(float).max)
    resolution_floor = min(1e-12 * delta_eq, 0.01 * xtol_brentq)
    lo_floor = max(
        np.nextafter(0.0, 1.0),
        floating_floor,
        resolution_floor,
    )
    # Δ must lie below the Debye cutoff: the residual's integral runs over
    # [Δ, ω_D], so as Δ → ω_D the domain vanishes and residual → −1/λ < 0.
    # A colder-than-thermal population drives the self-consistent gap far
    # above Δ_eq (toward the T=0 gap), which can be tens of times Δ_eq near
    # T_c — so widen the high side all the way to ω_D rather than capping at
    # a fixed step count that silently underestimated. NOTE: residual(Δ) is
    # monotone in Δ only for an equilibrium (monotone-decreasing) f, where
    # 1 − 2f increases in E. For a strongly non-equilibrium f with a gap-edge
    # occupation bump the residual can have several roots. A supplied
    # reference_gap activates deterministic continuation below; without it the
    # legacy bracketed root may depend on bracket_factor and the diagnostic at
    # the end of the function surfaces that ambiguity.
    hi_ceiling = omega_D * (1.0 - 1e-9)

    if reference_gap is not None and reference_gap >= hi_ceiling:
        raise ValueError(
            "reference_gap must lie below the Debye cutoff; got "
            f"{reference_gap:g} μeV for ω_D={omega_D:g} μeV."
        )
    if reference_gap is not None and not allow_gap_edge_extrapolation:
        # A continuation anchor outside the represented occupation domain can
        # steer the branch scan through invented gap-edge states even if a
        # later root happens to lie on-grid. Reject it before scanning.
        enforce_grid_support(float(reference_gap), solved_state=False)

    if reference_gap is not None:
        continuation_roots = _continuation_roots(
            residual,
            reference_gap=float(reference_gap),
            equilibrium_gap=delta_eq,
            lo_floor=lo_floor,
            hi_ceiling=hi_ceiling,
            cell_edges=cell_edges,
            xtol=xtol_brentq,
        )
        if continuation_roots:
            candidate = min(
                continuation_roots,
                key=lambda root: (abs(root - float(reference_gap)), root),
            )
            if len(continuation_roots) > 1:
                roots_text = ", ".join(f"{root:.8g}" for root in continuation_roots)
                warnings.warn(
                    "solve_gap: the non-equilibrium gap residual has multiple "
                    f"roots ({len(continuation_roots)} detected near the continuation "
                    f"branch: [{roots_text}] μeV). Selected Δ={candidate:.8g} μeV, "
                    f"the root nearest reference_gap={float(reference_gap):.8g} μeV. "
                    "The algebraic gap is physically branch-ambiguous; continuation "
                    "keeps the numerical branch deterministic.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            enforce_grid_support(candidate)
            return candidate

    if reference_gap is None:
        lo = max(delta_eq * (1.0 - bracket_factor), lo_floor)
        hi = min(delta_eq * (1.0 + bracket_factor), hi_ceiling)
        r_lo, r_hi = residual(lo), residual(hi)
        while r_lo * r_hi > 0 and (lo > lo_floor or hi < hi_ceiling):
            lo = max(lo * 0.5, lo_floor)
            hi = min(hi * 1.5, hi_ceiling)
            r_lo, r_hi = residual(lo), residual(hi)
    else:
        # The bounded continuation scan covered the physical interval but did
        # not isolate a root. Keep the no-root decision independent of the
        # legacy bracket_factor as well.
        lo, hi = lo_floor, hi_ceiling
        r_lo, r_hi = residual(lo), residual(hi)

    if r_lo * r_hi >= 0:
        # No sign change even after widening to the physical limits.
        # Negative residual at the cold floor ⇒ no superconducting solution
        # (the gap has collapsed to the normal state).
        if r_lo < 0:
            enforce_grid_support(0.0)
            return 0.0
        # Positive residual all the way to Delta approximately omega_D is
        # incompatible with the implemented finite-cutoff equation: the
        # integration domain vanishes at the upper limit, so the residual
        # must be negative there. Returning delta_eq would knowingly return a
        # non-root and could let an outer self-consistency loop certify a
        # mislabeled state. Fail closed with the endpoint evidence instead.
        raise RuntimeError(
            "solve_gap could not bracket a physical root: the residual stayed "
            "non-negative from the numerical zero-gap floor to the Debye "
            f"cutoff (R_lo={r_lo:.6e}, R_hi={r_hi:.6e}). This indicates an "
            "inconsistent calibration or a non-finite/under-resolved gap "
            "integral; no fallback gap is returned."
        )

    candidate = float(brentq(residual, lo, hi, xtol=xtol_brentq))
    if reference_gap is None:
        _warn_if_multirooted(residual, candidate, bracket_factor, lo_floor, hi_ceiling)
    enforce_grid_support(candidate)
    return candidate


_CONTINUATION_SCAN_POINTS = 65
_CONTINUATION_EDGE_SAMPLES = 64
_CONTINUATION_MAX_WINDOWS = 12


def _continuation_roots(
    residual: Callable[[float], float],
    *,
    reference_gap: float,
    equilibrium_gap: float,
    lo_floor: float,
    hi_ceiling: float,
    cell_edges: np.ndarray,
    xtol: float,
) -> list[float]:
    """Find sign-changing roots in expanding windows around a previous gap.

    A full high-resolution scan from ``lo_floor`` to the Debye cutoff would be
    prohibitively expensive inside every moving-gap projection.  Instead, the
    first window covers 15% of the local gap scale and at least eight nearby
    energy cells.  It is sampled by a fixed-size uniform lattice augmented by
    finite-volume cell edges, then doubled only when it contains no root.  The
    usual continuation case therefore costs one bounded local scan; collapse or
    a very distant branch triggers adaptive expansion.  The last iteration
    explicitly covers the entire physical interval, so the search remains
    bounded even for an adversarial reference.
    """
    edge_array = np.asarray(cell_edges, dtype=float)
    local_index = int(
        np.clip(
            np.searchsorted(edge_array, reference_gap, side="right") - 1,
            0,
            edge_array.size - 2,
        )
    )
    local_width = float(edge_array[local_index + 1] - edge_array[local_index])
    gap_scale = max(
        abs(reference_gap),
        abs(equilibrium_gap),
        np.finfo(float).tiny,
    )
    half_width = max(0.15 * gap_scale, 8.0 * local_width, 64.0 * xtol)

    for window_index in range(_CONTINUATION_MAX_WINDOWS):
        if window_index == _CONTINUATION_MAX_WINDOWS - 1:
            # A deterministic bounded fallback: cover the physical interval
            # once rather than continuing geometric expansion indefinitely.
            lo = lo_floor
            hi = hi_ceiling
        else:
            lo = max(lo_floor, reference_gap - half_width)
            hi = min(hi_ceiling, reference_gap + half_width)

        roots = _roots_in_scan_window(
            residual,
            lo=lo,
            hi=hi,
            reference_gap=reference_gap,
            cell_edges=edge_array,
            xtol=xtol,
        )
        if roots:
            return roots
        if lo <= lo_floor and hi >= hi_ceiling:
            return []
        half_width *= 2.0

    return []


def _roots_in_scan_window(
    residual: Callable[[float], float],
    *,
    lo: float,
    hi: float,
    reference_gap: float,
    cell_edges: np.ndarray,
    xtol: float,
) -> list[float]:
    """Isolate and refine every sampled sign change in one bounded window."""
    if hi <= lo:
        return []

    nodes = np.linspace(lo, hi, _CONTINUATION_SCAN_POINTS)
    relevant_edges = cell_edges[(cell_edges > lo) & (cell_edges < hi)]
    if relevant_edges.size > _CONTINUATION_EDGE_SAMPLES:
        sample_indices = np.linspace(
            0,
            relevant_edges.size - 1,
            _CONTINUATION_EDGE_SAMPLES,
            dtype=int,
        )
        relevant_edges = relevant_edges[np.unique(sample_indices)]
    if lo < reference_gap < hi:
        nodes = np.concatenate((nodes, relevant_edges, np.array([reference_gap])))
    else:
        nodes = np.concatenate((nodes, relevant_edges))
    nodes = np.unique(nodes)

    values = np.array([residual(float(node)) for node in nodes])
    if np.any(~np.isfinite(values)):
        raise RuntimeError("Gap residual returned a non-finite value during branch scan.")

    from scipy.optimize import brentq

    roots: list[float] = [float(nodes[index]) for index in np.flatnonzero(values == 0.0)]
    sign_changes = np.flatnonzero(values[:-1] * values[1:] < 0.0)
    roots.extend(
        float(brentq(residual, float(nodes[index]), float(nodes[index + 1]), xtol=xtol))
        for index in sign_changes
    )
    if not roots:
        return []

    merge_tolerance = max(
        8.0 * xtol,
        128.0 * np.finfo(float).eps * max(abs(lo), abs(hi), 1.0),
    )
    distinct: list[float] = []
    for root in sorted(roots):
        if not distinct or abs(root - distinct[-1]) > merge_tolerance:
            distinct.append(root)
    return distinct


def _warn_if_multirooted(
    residual: Callable[[float], float],
    candidate: float,
    bracket_factor: float,
    lo_floor: float,
    hi_ceiling: float,
) -> None:
    """Warn if the gap residual has more than one root near ``candidate``.

    For a non-equilibrium occupation the residual is not monotone, so several
    physical gaps can satisfy it. Without ``reference_gap``, ``solve_gap``
    retains the legacy bracketed root, which can depend on ``bracket_factor``.
    This diagnostic scan makes the ambiguity loud and directs continuation
    callers to the deterministic reference-anchored contract.
    """
    half = max(2.0 * bracket_factor, 0.1) * abs(candidate)
    lo = max(candidate - half, lo_floor)
    hi = min(candidate + half, hi_ceiling)
    if hi <= lo:
        return
    grid = np.linspace(lo, hi, 257)
    signs = np.sign(np.array([residual(float(x)) for x in grid]))
    nonzero = signs[signs != 0]
    n_changes = int(np.count_nonzero(np.diff(nonzero) != 0)) if nonzero.size else 0
    if n_changes > 1:
        warnings.warn(
            "solve_gap: the gap residual has multiple roots near "
            f"Δ={candidate:.6g} μeV for this non-equilibrium occupation "
            f"({n_changes} sign changes in the search neighborhood). The returned "
            "gap is the legacy bracketed root and can depend on bracket_factor. "
            "Pass reference_gap=<previous gap> for deterministic branch "
            "continuation; the physical self-consistent gap remains ambiguous.",
            RuntimeWarning,
            stacklevel=3,
        )
