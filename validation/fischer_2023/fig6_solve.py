"""Fischer 2023 Fig. 6 — paper-topology gap-suppression reproduction.

This is the **structural** Fig. 6 reproduction: gap suppression $\\delta\\Delta$
caused by the nonequilibrium distribution, plotted in the paper's
ordinate

    $\\frac{\\delta\\Delta_T - \\delta\\Delta}{\\delta\\Delta_T}
       = \\frac{\\Delta_\\mathrm{driven} - \\Delta_\\mathrm{eq}(T_B)}
              {\\Delta_0 - \\Delta_\\mathrm{eq}(T_B)},$

against the Eq. 35 drive-equivalent temperature ratio $T_*/\\Delta$, swept
over $\\bar n$ at three bath temperatures $T_B \\in \\{0.10, 0.15, 0.20\\}$ K
on a paper-resolution grid (1640 bins: the original 1620 above-gap cells plus
20 sub-gap guard cells, $dE = 1\\,\\mu$eV). Solid lines: numerical
joint kinetic-equation + self-consistent gap solve. Dashed lines:
analytical Eq. 53.

The ordinate is the paper's normalized form $(\\delta\\Delta_T - \\delta\\Delta)/\\delta\\Delta_T$,
which goes negative on the strong-drive side; the 1640-bin grid resolves
the sign change cleanly.

$\\tau_\\ell$ model
------------------

The paper sets $\\tau_\\ell = \\tau_0^{PB} \\approx 255$ ps for Fig. 6, and
that is the default here (``TAU_L_MODEL = "tau_0_pb"``, overridable via
the ``FISCHER2023_FIG6_TAU_L_MODEL`` environment variable). The
extracted $\\tau_0^{PB}$ diagnostic is pinned to the phonon-side
F&C/Kaplan pair-breaking rate and reproduces the paper-quoted ~255 ps
for the Table I parameters.

For comparison, :func:`qpsim.physics.acoustic_escape_tau_l` with
Fischer's 63 nm film and $\\eta = 0.2$ gives $\\tau_\\ell \\approx 368$ ps
(Debye-averaged sound velocity) — ~44 % longer than the paper's nominal
$\\tau_0^{PB}$. The dimensionless $T_*/\\Delta$ axis from Eq. 35 is
independent of $\\tau_\\ell$, so the x-axis is invariant under the model
choice; the y-axis position of the curves is sensitive to it.

Eq. 53 reads

       $\\delta\\Delta/\\Delta_0 = x_\\mathrm{qp}^\\mathrm{paper} \\cdot
          \\bigl[1 - 0.42\\,(T_*/\\Delta_0) + 0.22\\,(T_*/\\Delta_0)^2\\bigr]$,

with the *paper's* prefactor convention $x_\\mathrm{qp}^\\mathrm{paper}
= n_\\mathrm{qp}/(2\\rho_F\\Delta_0)$ (text following Eq. 53,
p. 054087-10). qpsim's Fischer-convention $x_\\mathrm{qp}$
(:mod:`qpsim.observables.density`) is $n_\\mathrm{qp}/(4\\rho_F\\Delta_0)$,
half the paper's, so the Eq. 53 helpers apply the factor-2 conversion
internally (``_XQP_QPSIM_TO_PAPER``) — Eq. 47 *density*-level
comparisons stay in qpsim convention and take no factor. The bracketed
factor is verified closed-form from the paper text, and the dashed
overlay feeds it the analytical Eq. 47 + Appendix-E
$x_\\mathrm{qp}$ from :func:`fig5_paper._xqp_analytic_eq47`, using the
same scalar $\\tau_\\ell$ as the numerical solve. The thermal counterpart
$\\delta\\Delta_T$ uses the BCS gap calibration in the denominator,
matching the numerical observable definition.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:
parameters identical to :mod:`fig3_paper` and :mod:`fig5_paper`.

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig6_paper
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Any

import numpy as np
from qpsim.backends.t3_diffusion import (
    SelfConsistentGapCollapseError,
    T3DiffusionBackend,
    T3DiffusionState,
)
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material, load_material
from qpsim.observables.density import qp_fraction
from qpsim.observables.gap_suppression import (
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
    gap_suppression_ratio_from_integrals,
    thermal_gap_integral_direct,
)
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics import build_tau_l, calibrate_gap, solve_gap
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.coupled_newton import CoupledNewtonLineSearchError

from validation.fischer_2023.fig5_paper import _xqp_analytic_eq47
from validation.fischer_2023.steady_state_certificate import (
    CERTIFICATE_FIELDS,
    CERTIFICATE_METRIC_VERSION,
    steady_state_certificate,
)

# ── Fischer 2023 Table I parameters (shared with fig3_paper, fig5_paper) ──

DELTA_0 = 180.0            # μeV (zero-T gap; also Eq. 53 normalization)
TAU_0 = 438.0              # ns
# The finite-cutoff calibrator, rather than the infinite-cutoff rounded 1.764
# ratio, is the single pairing-scale anchor.  This makes the model's T=0 gap
# equal DELTA_0 while preserving continuous closure at the declared T_C.
GAP_SOLVE_XTOL_UEV = 1e-12
FINITE_CUTOFF_DELTA0_OVER_KBTC = (
    calibrate_gap(T_c=1.0, T_bath=0.0, xtol=GAP_SOLVE_XTOL_UEV).delta_0_bcs
    / KB_UEV_PER_K
)
T_C = DELTA_0 / (FINITE_CUTOFF_DELTA0_OVER_KBTC * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV
C_PHOT = 1e-9              # ns^-1 (1 Hz)

# Acoustic-escape film geometry (Fischer 2023 §V text).
FILM_THICKNESS_NM = 63.0
SUBSTRATE_ETA = 0.2

# Phonon-escape-time model (qpsim.physics.build_tau_l). The paper sets
# τ_l = τ_0^PB ≈ 255 ps exactly, so "tau_0_pb" is the paper-faithful default.
# The Kaplan thin-film acoustic estimate ("acoustic_escape") gives ~368 ps for
# this 63 nm film (1.44× the paper) and inflates the Fig. 6 ordinate amplitude;
# override with FISCHER2023_FIG6_TAU_L_MODEL to compare.
TAU_L_MODEL = os.environ.get("FISCHER2023_FIG6_TAU_L_MODEL", "tau_0_pb").lower()

# Paper-resolution grid: retain dE = 1 micro-eV while adding one drive-photon
# quantum of sub-gap headroom. A self-consistent solution necessarily has
# Delta < Delta_0; starting the first cell edge at Delta_0 used to extrapolate
# across missing square-root-singular support and now correctly fails the
# finite-volume density contract. The 20 prepended cells are inactive until
# the solved gap moves below Delta_0; every original above-gap center remains.
E_MIN_FACTOR = (DELTA_0 - OMEGA_0) / DELTA_0
E_MAX_FACTOR = 10.0
NUM_BINS = 1640

# Paper sweep — three bath temperatures, swept over n̄.
T_BATH_VALUES: tuple[float, ...] = (0.10, 0.15, 0.20)

# n̄ sweep: log-spaced across seven decades. The Eq. 35 map turns this
# into T_*/Δ ∈ [~0.10, ~0.65], which contains the paper's plotted range
# of [0.25, 0.65] with margin on both sides for the analytic overlay.
N_BAR_VALUES: np.ndarray = np.logspace(4.0, 8.2, 22)
# Paper Fig. 6 plots T_*/Δ ∈ ~[0.16, 0.66]; under Eq. 35 this maps to
# n̄ ∈ ~[10^4, 10^8.2]. Above ~10^8 the gap collapses non-physically and
# the paper observable diverges (off-paper-scale). 22 points span the
# paper region with ~5 points/decade — adequate for the smooth dashed
# Eq. 53 overlay and the sign-change peak.

# Picard fixed-point tolerance for the joint (f, n_ph) inner iteration.
# Paper-faithful default 1e-12 is needed for the low-T_B observable to
# resolve to ~3 sig figs; the --fast CLI flag lowers this to 1e-9 for
# dev iteration (~30× speedup at the cost of ~1 sig fig).
PICARD_TOL: float = 1e-12
PICARD_ATOL: float = 1e-14
TARGET_BACKWARD_ERROR_LIMIT = 1e-5
"""Maximum independently reassembled balance error at a converged target."""

NEWTON_BACKWARD_ERROR_TOL = 1e-6
"""Inner Newton balance gate used by the Picard kinetic solve."""

DIRECT_GAP_NEWTON_BACKWARD_ERROR_TOL = 1e-10
"""Inner-Newton balance gate for the fixed-gap direct-observable fallback."""

DIRECT_GAP_BACKWARD_ERROR_LIMIT = 1e-9
"""Independent kinetic-balance gate for accepted direct-gap points.

The direct observable divides two integrals that are both exponentially small
at low bath temperature, so the ordinary paper-sweep gate is not sufficiently
selective. This limit applies to the QP and representability-aware certified
phonon metrics. The raw phonon metric remains diagnostic: on cold grids it can
sit near ``1e-6`` solely because the affine phonon root is not representable in
float64 even when the certified excess is zero.
"""

GAP_FIXED_POINT_ABS_TOL_UEV = 1e-10
"""Absolute independently reassembled gap-map target for accepted points."""

GAP_FIXED_POINT_REL_TOL = GAP_FIXED_POINT_ABS_TOL_UEV / DELTA_0
"""Backend relative tolerance implying the absolute target for gap <= DELTA_0."""

GAP_FIXED_POINT_CERTIFICATE_FIELD = "gap_fixed_point_abs_error_uev"
FIG6_CERTIFICATE_FIELDS = (*CERTIFICATE_FIELDS, GAP_FIXED_POINT_CERTIFICATE_FIELD)
"""Certificate fields stored in the Fig. 6 raw payload and current CSV."""

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the extracted τ_0^PB diverges from the paper-quoted 255 ps."""

# Eq. 53 verified bracketed-factor coefficients (Fischer 2023 §IV.B).
_EQ53_DRIVE_C1 = 0.42
_EQ53_DRIVE_C2 = 0.22
# Thermal Sommerfeld counterpart, also paragraph after Eq. 53.
_EQ53_THERMAL_C1 = 0.5
_EQ53_THERMAL_C2 = 3.0 / 8.0

# Paper Eq. 53/54 prefactor is N_qp/(2 rho_F Delta_0); qpsim's Fischer-
# convention x_qp (qpsim.observables.density) is N_qp/(4 rho_F Delta_0).
# The Eq. 53 helpers below take qpsim-convention x_qp and convert at this
# boundary so no caller can forget the factor.
_XQP_QPSIM_TO_PAPER = 2.0


def _fischer_material() -> Material:
    """Al film with Fischer 2023 SC parameters and 63 nm thickness."""
    return replace(
        load_material("Al"),
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,  # F&C 2023 Table I: τ_0^PB = 255 ps
        film_thickness=FILM_THICKNESS_NM,
        substrate_transmission_eta=SUBSTRATE_ETA,
    )


def _build_grid_and_spectral() -> tuple[np.ndarray, np.ndarray, SpectralContext]:
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=NUM_BINS,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    if abs(OMEGA_0 - m * dE_scalar) / OMEGA_0 > 1e-10:
        raise RuntimeError(
            f"Fischer Fig. 6 paper grid not commensurate: ω_0={OMEGA_0}, m·dE={m*dE_scalar}"
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _compute_tau_0_pb(spectral: SpectralContext) -> float:
    """Same definition as :func:`fig3_paper._compute_tau_0_pb`.

    Reported for the run-banner sanity check and for the $x_{\\rm qp}$
    Eq. 47 overlay; the numerical $\\tau_\\ell$ this script uses depends on
    ``TAU_L_MODEL`` (default ``"tau_0_pb"`` ties it to this extraction).
    """
    K_r0_phonon_side = build_recombination_kernel_phonon_side(
        spectral, tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,
    )
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
        spectral.E,
    )
    f_zero = np.zeros(spectral.E.size)
    _, b_ph = compute_phonon_source_sink(
        f_zero, spectral, None, None,
        idx_diff, idx_sum, diff_sign,
        omega_bins.size,
        enable_scattering=False, enable_recombination=True,
        K_r0_phonon_side=K_r0_phonon_side,
    )
    threshold = 2.0 * spectral.gap
    above = (omega_bins >= threshold) & (b_ph < -1e-30)
    if not np.any(above):
        raise RuntimeError(
            "Could not find a phonon bin above 2Δ with a pair-breaking rate."
        )
    first_idx = int(np.argmax(above))
    return float(1.0 / -b_ph[first_idx])


def _kBTstar_eq35(n_bar: float) -> float:
    """Fischer 2023 Eq. 35: $k_B T_* = (A\\bar n)^{1/6}$, in μeV.

    Same expression as :func:`fig5_paper._kBTstar_eq35`.
    """
    if n_bar <= 0:
        return 0.0
    A = (
        (105.0 / 64.0)
        * (KB_UEV_PER_K * T_C) ** 3
        * C_PHOT
        * TAU_0
        * OMEGA_0 ** 2
        * DELTA_0
    )
    return float((A * n_bar) ** (1.0 / 6.0))


def _paper_eq53_analytic_drive(x_qp_qpsim: float, T_star_over_delta: float) -> float:
    r"""Fischer 2023 Eq. 53 — nonequilibrium (drive) gap suppression.

    .. math::
        \delta\Delta/\Delta_0
        = x_\mathrm{qp}^\mathrm{paper}
          \bigl[1 - 0.42\,(T_*/\Delta_0) + 0.22\,(T_*/\Delta_0)^2\bigr],

    with the paper's :math:`x_\mathrm{qp}^\mathrm{paper}
    = N_\mathrm{QP}/(2\rho_F\Delta_0)`. The input ``x_qp_qpsim`` is in
    qpsim's Fischer convention :math:`N_\mathrm{QP}/(4\rho_F\Delta_0)`
    (half the paper's); this helper applies the factor-2 conversion
    internally so the caller passes qp_fraction/Eq.-47 values unchanged.

    The bracketed factor is the verified closed form from the paper text
    (paragraph following Eq. 53, p. 054087-10). The input
    :math:`x_\mathrm{qp}` should --- per the paper --- come from the
    *analytical* density Eq. 47 with appendix corrections.

    Returns
    -------
    delta_Delta_over_Delta0
        :math:`\delta\Delta_\mathrm{drive}/\Delta_0` --- so multiply by
        ``DELTA_0`` to get the suppression in μeV.
    """
    r = T_star_over_delta
    bracket = 1.0 - _EQ53_DRIVE_C1 * r + _EQ53_DRIVE_C2 * r * r
    return float(_XQP_QPSIM_TO_PAPER * x_qp_qpsim * bracket)


def _paper_eq53_analytic_thermal(x_qp_th_qpsim: float, T_bath_over_delta: float) -> float:
    r"""Thermal-equilibrium Sommerfeld counterpart of Eq. 53.

    From the paragraph following Eq. 53 (p. 054087-11), in equilibrium
    the bracketed factor reads :math:`1 - \tfrac{1}{2}(T_B/\Delta_0)
    + \tfrac{3}{8}(T_B/\Delta_0)^2`. As in
    :func:`_paper_eq53_analytic_drive`, the input is qpsim-convention
    :math:`N_\mathrm{QP}/(4\rho_F\Delta_0)` and the paper's
    :math:`N_\mathrm{QP}/(2\rho_F\Delta_0)` prefactor is restored
    internally.
    """
    r = T_bath_over_delta
    bracket = 1.0 - _EQ53_THERMAL_C1 * r + _EQ53_THERMAL_C2 * r * r
    return float(_XQP_QPSIM_TO_PAPER * x_qp_th_qpsim * bracket)


def _x_qp_thermal(spectral: SpectralContext, T_bath: float) -> float:
    """Thermal-equilibrium $x_\\mathrm{qp}$ at $T_B$ from Fermi-Dirac."""
    kT = KB_UEV_PER_K * T_bath
    if kT <= 0.0:
        return 0.0
    f_FD = 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)
    return float(qp_fraction(f_FD, spectral, delta_0=DELTA_0))


def _build_state(
    material: Material,
    spectral: SpectralContext,
    T_bath: float,
    *,
    f_seed: np.ndarray | None = None,
    n_ph_seed: np.ndarray | None = None,
    continuation_seed: T3DiffusionState | None = None,
) -> T3DiffusionState:
    """Build a state, optionally cloning a complete same-temperature seed.

    A self-consistent continuation seed carries its gap and matching
    :class:`SpectralContext` as well as ``f`` and ``n_ph``. Rebuilding it on
    the original ``DELTA_0`` context would discard outer-gap continuation
    between adjacent ``n_bar`` targets.
    """
    if continuation_seed is not None:
        if f_seed is not None or n_ph_seed is not None:
            raise ValueError(
                "continuation_seed is mutually exclusive with f_seed/n_ph_seed."
            )
        if continuation_seed.T_bath != T_bath:
            raise ValueError(
                "Fig. 6 continuation seeds cannot cross temperature rows: "
                f"seed T_bath={continuation_seed.T_bath}, requested {T_bath}."
            )
        seeded_spectral = continuation_seed.spectral
        if not (
            np.array_equal(seeded_spectral.E, spectral.E)
            and np.array_equal(seeded_spectral.dE, spectral.dE)
            and seeded_spectral.dynes_gamma == spectral.dynes_gamma
        ):
            raise ValueError(
                "Fig. 6 continuation seed must use the configured energy grid."
            )
        if continuation_seed.gap != seeded_spectral.gap:
            raise ValueError(
                "Fig. 6 continuation seed gap must match its SpectralContext."
            )
        return replace(
            continuation_seed,
            f=continuation_seed.f.copy(),
            phonon=replace(
                continuation_seed.phonon,
                n_ph=continuation_seed.phonon.n_ph.copy(),
                omega_bins=continuation_seed.phonon.omega_bins.copy(),
                tau_l=continuation_seed.phonon.tau_l.copy(),
            ),
            material=material,
            T_bath=T_bath,
            _moving_gap_coordinates=None,
        )

    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    omega_2d = omega.reshape(1, -1)
    if n_ph_seed is None:
        n_ph_seed = thermal_phonon_occupation(omega, T_bath)
    phonon = PhononState(
        n_ph=n_ph_seed.reshape(1, -1, 1).copy(),
        omega_bins=omega_2d,
        tau_l=build_tau_l(TAU_L_MODEL, omega_2d, material),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    if f_seed is None:
        kT = KB_UEV_PER_K * T_bath
        f_seed = 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_seed.copy(),
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _solve_picard_sc_gap(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float] | None,
) -> T3DiffusionState:
    """Joint Picard + self-consistent BCS gap solve.

    Inner Picard iterates (f, n_ph); outer iteration re-solves the BCS
    gap equation against the converged $f$.
    """
    # The backend accepts a relative outer-gap tolerance. Since every
    # superconducting target satisfies gap <= DELTA_0, dividing the absolute
    # certificate target by DELTA_0 makes backend acceptance at least as tight
    # as GAP_FIXED_POINT_ABS_TOL_UEV. The independent branch-anchored check in
    # _solve_and_measure remains authoritative.
    #
    # gap_tol/picard_tol were tightened because the
    # paper observable (Δ_driven - Δ_eq)/(Δ_0 - Δ_eq) divides by the
    # exponentially small thermal suppression δΔ_T ≈ √(2π Δ T_B) e^{-Δ/T_B}:
    # δΔ_T ≈ 8e-8 μeV at T_B=0.10 K, ~1e-4 at T_B=0.15 K. With the loose
    # gap_tol the noise-to-signal ratio of the observable was unity at low T_B;
    # tighter gap is needed for the low-T_B curves to plot in the paper's
    # [0, 0.25] band. T_B=0.20 K curve already works with the loose tol.
    return backend.steady_state(
        state,
        method="picard",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        self_consistent_gap=True,
        gap_tol=GAP_FIXED_POINT_REL_TOL,
        gap_max_iter=50,
        gap_under_relaxation=0.5,
        gap_solve_xtol=GAP_SOLVE_XTOL_UEV,
        picard_tol=PICARD_TOL,
        picard_atol=PICARD_ATOL,
        # Fig. 6 independently reassembles and hard-gates the same balance
        # below. Align Picard termination eligibility with that validated
        # target; a stricter hidden inner gate can stall on roundoff without
        # improving the accepted paper observable.
        picard_balance_tol=TARGET_BACKWARD_ERROR_LIMIT,
        picard_max_iter=2000,
        picard_mixing=0.3,
        anderson_depth=0,
        newton_tol=1e-14,
        newton_backward_error_tol=NEWTON_BACKWARD_ERROR_TOL,
        newton_max_iter=500,
    )


def _solve_coupled_newton_fixed_gap(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float] | None,
) -> T3DiffusionState:
    """Joint coupled-Newton solve at fixed Delta0."""
    return backend.steady_state(
        state,
        method="coupled_newton",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        self_consistent_gap=False,
        coupled_newton_tol=PICARD_TOL,
        # Fig. 6's ordinate divides by an exponentially small δΔ_T, so the
        # driven f must be converged far tighter than an absolute residual tol
        # can guarantee at f ~ 1e-10. Opt into the scale-invariant relative-step
        # criterion so warm continuation seeds keep refining instead of exiting
        # at iteration 0 with a stale state.
        coupled_newton_step_rtol=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        coupled_newton_max_iter=80,
        coupled_newton_fd_step=1e-8,
        # Analytical cross-Jacobians (J_fn, J_nf): exact and O(NE²), where the
        # finite-difference secant is both ~30 min/point at 1640 bins and
        # unreliable at strong drive (f, n_ph ≪ 1), branch-hopping / NaN-ing the
        # post-peak tail. Closed form fills the tail and cuts runtime to seconds.
        coupled_newton_analytic_cross=True,
    )


def _solve_picard_fixed_gap(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float] | None,
) -> T3DiffusionState:
    """Strict fixed-gap Picard solve used when coupled Newton stalls.

    Coupled Newton is the fast path for the paper grid, but at a fully
    converged cold-grid root its line search can be unable to reduce an
    already-roundoff-level dimensional residual. Retrying with the independent
    Picard formulation avoids classifying that numerical stall as a physical
    continuation fold. The caller still independently reassembles and gates
    the returned balance.
    """
    return backend.steady_state(
        state,
        method="picard",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        self_consistent_gap=False,
        picard_tol=PICARD_TOL,
        picard_atol=PICARD_ATOL,
        picard_balance_tol=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        picard_max_iter=2000,
        picard_mixing=0.3,
        anderson_depth=0,
        newton_tol=1e-14,
        newton_backward_error_tol=DIRECT_GAP_NEWTON_BACKWARD_ERROR_TOL,
        newton_max_iter=500,
    )


def _solve_fixed_gap_kinetics(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float] | None,
) -> T3DiffusionState:
    """Use coupled Newton first, with strict fixed-gap Picard as fallback."""
    try:
        return _solve_coupled_newton_fixed_gap(backend, state, photon_params)
    except CoupledNewtonLineSearchError as coupled_error:
        # Picard is an independent polish only for the observed cold-grid
        # roundoff stall. Ordinary line-search failure above the configured
        # dimensional tolerance, singular Jacobians, non-finite states, and
        # iteration exhaustion remain fail-loud.
        if (
            not np.isfinite(coupled_error.residual_norm)
            or coupled_error.residual_norm > PICARD_TOL
        ):
            raise
        try:
            return _solve_picard_fixed_gap(backend, state, photon_params)
        except RuntimeError as picard_error:
            raise RuntimeError(
                "Fischer Fig. 6 fixed-gap kinetics failed with both coupled "
                f"Newton ({coupled_error}) and strict Picard ({picard_error})."
            ) from picard_error


def _check_tau_0_pb(tau_0_pb: float, tau_l: float) -> None:
    tau_pb_ps = tau_0_pb * 1000.0
    tau_l_ps = tau_l * 1000.0
    print(f"  tau_0^PB (phonon-side extracted)     = {tau_0_pb:.4f} ns "
          f"({tau_pb_ps:.1f} ps)")
    print(f"  Paper-quoted tau_0^PB                 ~= {PAPER_TAU_0_PB_PS:.0f} ps")
    print(f"  tau_l (model={TAU_L_MODEL!r})         = {tau_l:.4f} ns "
          f"({tau_l_ps:.1f} ps)")
    print(f"  tau_l / paper-tau_0^PB                ~= "
          f"{tau_l_ps / PAPER_TAU_0_PB_PS:.2f}x "
          f"(paper sets tau_l = tau_0^PB exactly)")
    pb_ratio = tau_pb_ps / PAPER_TAU_0_PB_PS
    if pb_ratio > TAU_0_PB_WARN_FACTOR or pb_ratio < 1.0 / TAU_0_PB_WARN_FACTOR:
        print(
            f"  (Note: extracted tau_0^PB / paper tau_0^PB = {pb_ratio:.2f}x; "
            f"check the phonon-side pair-breaking diagnostic before\n"
            f"   regenerating baselines.)",
            flush=True,
        )
    if TAU_L_MODEL not in ("tau_0_pb", "tau0_pb", "pair_breaking", "paper"):
        print(
            f"  (Note: tau_l model is {TAU_L_MODEL!r}, not the paper's "
            f"tau_l = tau_0^PB convention; the Fig. 6 ordinate amplitude is\n"
            f"   sensitive to tau_l, so expect a y-axis offset vs the paper.)",
            flush=True,
        )


def _acceptable_ratio(obs: float) -> bool:
    """Accept every finite signed gap-suppression ratio.

    Both formulations evaluate ``(Δ_driven − Δ_eq) / (Δ_0 − Δ_eq)``.
    A negative value legitimately means that drive suppresses the gap below
    its thermal value. Solver failures and independent root certificates,
    rather than the sign of a derived observable, remain authoritative.
    """
    return bool(np.isfinite(obs))


def _require_finite_measurements(
    obs: float,
    delta_driven: float,
    x_qp: float,
    *,
    T_bath: float,
    n_bar: float,
) -> None:
    """Fail loudly when a solved state yields a non-finite derived quantity."""
    measurements = {
        "gap-suppression observable": obs,
        "driven gap": delta_driven,
        "x_qp": x_qp,
    }
    nonfinite = []
    if not _acceptable_ratio(obs):
        nonfinite.append("gap-suppression observable")
    nonfinite.extend(
        name
        for name, value in measurements.items()
        if name != "gap-suppression observable" and not np.isfinite(value)
    )
    if nonfinite:
        raise RuntimeError(
            "Fischer Fig. 6 produced non-finite derived measurement(s) "
            f"{', '.join(nonfinite)} at T_B={T_bath:g} K, n_bar={n_bar:.6e}."
        )


def _require_supported_mode(
    *,
    direct_gap_observable: bool,
    fixed_gap_kinetics: bool,
) -> None:
    """Require one of the two internally consistent Fig. 6 formulations."""
    if bool(direct_gap_observable) != bool(fixed_gap_kinetics):
        raise ValueError(
            "direct_gap_observable and fixed_gap_kinetics must be enabled "
            "together: use either the self-consistent-gap formulation or "
            "the author-style fixed-gap/direct-observable formulation."
        )


def _require_target_certificate(
    certificate: dict[str, float],
    *,
    T_bath: float,
    n_bar: float,
    require_gap_fixed_point: bool,
    backward_error_limit: float = TARGET_BACKWARD_ERROR_LIMIT,
) -> None:
    """Hard-fail a returned kinetic state that is not a certified root.

    ``backward_error_limit`` gates the QP and representability-aware certified
    phonon metrics. The raw phonon metric remains diagnostic because it includes
    irreducible float64 rounding of the affine phonon root.
    """
    if not np.isfinite(backward_error_limit) or backward_error_limit <= 0.0:
        raise ValueError(
            "backward_error_limit must be finite and positive; "
            f"got {backward_error_limit}."
        )
    qp_backward = certificate["qp_backward_error"]
    phonon_backward = certificate["phonon_backward_error"]
    if (
        not np.isfinite(qp_backward)
        or not np.isfinite(phonon_backward)
        or qp_backward > backward_error_limit
        or phonon_backward > backward_error_limit
    ):
        raise RuntimeError(
            "Fischer Fig. 6 converged target failed the independent "
            f"steady-state certificate at T_B={T_bath:g} K, "
            f"n_bar={n_bar:.6e} (limit={backward_error_limit:g}): "
            f"qp={qp_backward:.3e}, phonon={phonon_backward:.3e}."
        )

    gap_error = certificate[GAP_FIXED_POINT_CERTIFICATE_FIELD]
    if require_gap_fixed_point and (
        not np.isfinite(gap_error)
        or gap_error > GAP_FIXED_POINT_ABS_TOL_UEV
    ):
        raise RuntimeError(
            "Fischer Fig. 6 converged target failed the independent "
            f"gap-map certificate at T_B={T_bath:g} K, "
            f"n_bar={n_bar:.6e}: abs_error={gap_error:.3e} micro-eV, "
            f"limit={GAP_FIXED_POINT_ABS_TOL_UEV:.3e} micro-eV."
        )


class _PointSolveError(RuntimeError):
    """Explicit superconducting collapse recorded as a missing curve point."""


def _solve_and_measure(
    backend: T3DiffusionBackend,
    material: Material,
    spectral: SpectralContext,
    T_bath: float,
    n_bar_val: float,
    continuation_seed: T3DiffusionState | None,
    *,
    fixed_gap_kinetics: bool,
    direct_gap_observable: bool,
    thermal_integral: float | None,
    delta_eq: float,
    delta_T: float,
) -> tuple[T3DiffusionState, float, float, float, dict[str, float]]:
    """Build a seeded state, solve, and measure the gap-suppression observable.

    Returns ``(converged, obs, delta_driven, x_qp, certificate)``; raises
    ``RuntimeError`` on solver non-convergence. The certificate independently
    reassembles the fixed-gap balance represented by the returned state, even
    when the state came from the outer self-consistent-gap loop.
    """
    state = _build_state(
        material,
        spectral,
        T_bath,
        continuation_seed=continuation_seed,
    )
    photon_params = {
        "omega_0": OMEGA_0, "n_bar": float(n_bar_val), "c_phot": C_PHOT,
    }
    if fixed_gap_kinetics:
        # Fixed-gap/direct mode has no outer gap-continuation fold. Its solver
        # failures therefore remain fail-loud all the way through _solve_sweep.
        converged = _solve_fixed_gap_kinetics(backend, state, photon_params)
    else:
        try:
            converged = _solve_picard_sc_gap(backend, state, photon_params)
        except SelfConsistentGapCollapseError as exc:
            # Only an explicitly classified superconducting collapse is
            # fold-to-NaN eligible. Singular/nonfinite failures, inner-solver
            # exhaustion, and ordinary outer-gap nonconvergence propagate.
            raise _PointSolveError(
                "Fischer Fig. 6 self-consistent gap collapsed."
            ) from exc
    if direct_gap_observable:
        # _solve_sweep always supplies thermal_integral in direct-gap mode.
        assert thermal_integral is not None
        driven_integral = gap_integral_from_distribution_direct(
            converged.f, spectral.E, gap=DELTA_0, samples="centers",
        )
        delta_driven = gap_from_distribution_direct(
            converged.f, spectral.E, gap=DELTA_0, delta0=DELTA_0, samples="centers",
        )
        obs = gap_suppression_ratio_from_integrals(driven_integral, thermal_integral)
    else:
        delta_driven = float(converged.gap)
        obs = (converged.gap - delta_eq) / delta_T
    x_qp = qp_fraction(converged.f, converged.spectral, delta_0=DELTA_0)
    _require_finite_measurements(
        obs,
        delta_driven,
        x_qp,
        T_bath=T_bath,
        n_bar=n_bar_val,
    )
    tau_l = float(converged.phonon.tau_l[0, 0])
    certificate = steady_state_certificate(
        converged,
        photon_params=photon_params,
        tau_l=tau_l,
    )
    certificate[GAP_FIXED_POINT_CERTIFICATE_FIELD] = float("nan")
    if not direct_gap_observable and not fixed_gap_kinetics:
        calibration = calibrate_gap(
            T_c=T_C,
            T_bath=T_bath,
            Delta_0=DELTA_0,
            xtol=GAP_SOLVE_XTOL_UEV,
        )
        mapped_gap = solve_gap(
            calibration,
            converged.f,
            converged.spectral.E,
            dE_bins=converged.spectral.dE,
            reference_gap=converged.gap,
            xtol=GAP_SOLVE_XTOL_UEV,
        )
        certificate[GAP_FIXED_POINT_CERTIFICATE_FIELD] = abs(
            mapped_gap - converged.gap
        )
    return converged, obs, delta_driven, x_qp, certificate


def _solve_sweep(
    backend: T3DiffusionBackend,
    material: Material,
    spectral: SpectralContext,
    tau_0_pb: float,
    *,
    direct_gap_observable: bool = False,
    fixed_gap_kinetics: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    dict[str, np.ndarray],
]:
    """For each $T_B$, sweep n̄ low → high with (f, n_ph) continuation.

    Returns (T_star_over_delta, delta_eq, delta_driven, x_qp_num,
             x_qp_eq47, paper_observable_num, paper_observable_eq53,
             tau_l_used, certificates).
    All shaped ``(len(T_BATH_VALUES), N_BAR_VALUES.size)`` except
    ``delta_eq`` (per-T_B) and ``tau_l_used`` (scalar).
    """
    _require_supported_mode(
        direct_gap_observable=direct_gap_observable,
        fixed_gap_kinetics=fixed_gap_kinetics,
    )
    n_T = len(T_BATH_VALUES)
    n_n = N_BAR_VALUES.size
    T_star = np.zeros((n_T, n_n))
    delta_driven = np.zeros((n_T, n_n))
    x_qp_num = np.zeros((n_T, n_n))
    x_qp_eq47 = np.zeros((n_T, n_n))
    obs_num = np.zeros((n_T, n_n))
    obs_eq53 = np.zeros((n_T, n_n))
    delta_eq_per_T = np.zeros(n_T)
    tau_l_used: float | None = None
    certificates = {
        field: np.full((n_T, n_n), np.nan, dtype=float)
        for field in FIG6_CERTIFICATE_FIELDS
    }

    # δΔ_T = Δ_0 − Δ_eq(T_B) is exponentially small at low T_B (≈ 8e-8 μeV
    # at T_B=0.10 K, ≈ 1e-4 at T_B=0.15 K). Default brentq xtol of
    # 1e-6 * kBTc ≈ 1e-4 μeV would round δΔ_T to zero at T_B ≤ 0.15 K and
    # collapse the paper observable. Tighten to ~1e-12 μeV.
    for i, T_bath in enumerate(T_BATH_VALUES):
        thermal_integral: float | None = None  # set below only in direct-gap mode
        if direct_gap_observable:
            thermal_integral = thermal_gap_integral_direct(
                spectral.E,
                gap=DELTA_0,
                T_bath=T_bath,
                samples="centers",
            )
            delta_eq_per_T[i] = DELTA_0 * float(np.exp(-thermal_integral))
            # δΔ_T / Δ_0 from -expm1(-I_T), avoiding root-find cancellation.
            delta_T = DELTA_0 * float(-np.expm1(-thermal_integral))
        else:
            calibration = calibrate_gap(
                T_c=T_C,
                T_bath=T_bath,
                Delta_0=DELTA_0,
                xtol=GAP_SOLVE_XTOL_UEV,
            )
            delta_eq_per_T[i] = calibration.delta_eq
            # δΔ_T  = Δ_0 - Δ_eq(T_B), the thermal-equilibrium suppression
            # at this T_B (independent of drive). Used as the denominator of
            # the paper observable.
            delta_T = DELTA_0 - calibration.delta_eq
        if delta_T <= 0:
            # Bath at or above T_c — observable undefined; skip whole row.
            obs_num[i, :] = np.nan
            obs_eq53[i, :] = np.nan
            continue

        continuation_seed: T3DiffusionState | None = None
        tau_l_val: float | None = None
        for j, n_bar in enumerate(N_BAR_VALUES):
            if tau_l_val is None:
                tau_l_val = float(
                    _build_state(material, spectral, T_bath).phonon.tau_l[0, 0]
                )
                if tau_l_used is None:
                    tau_l_used = tau_l_val
            kBT_star = _kBTstar_eq35(float(n_bar))
            T_star[i, j] = kBT_star / DELTA_0
            # Analytic overlay fields are independent of the numerical solve; set
            # them first so a failed strong-drive point keeps its Eq. 53 overlay.
            x_qp_eq47[i, j] = _xqp_analytic_eq47(
                T_bath, float(n_bar), tau_l=tau_l_val, tau_0_pb=tau_0_pb,
            )
            delta_drive_analytic = (
                DELTA_0 * _paper_eq53_analytic_drive(x_qp_eq47[i, j], T_star[i, j])
            )
            obs_eq53[i, j] = (delta_T - delta_drive_analytic) / delta_T

            # Warm-continuation solve. Signed finite observables are retained:
            # Delta_driven < Delta_eq legitimately gives a negative ordinate.
            # An explicitly classified self-consistent superconducting collapse
            # is recorded as NaN. Every other solver or certificate failure
            # propagates; fixed-gap/direct mode has no outer gap collapse.
            converged: T3DiffusionState | None = None
            certificate: dict[str, float] | None = None
            obs = delta_driven_pt = x_qp_pt = float("nan")
            ok = False
            try:
                (
                    converged,
                    obs,
                    delta_driven_pt,
                    x_qp_pt,
                    certificate,
                ) = _solve_and_measure(
                    backend, material, spectral, T_bath, n_bar,
                    continuation_seed,
                    fixed_gap_kinetics=fixed_gap_kinetics,
                    direct_gap_observable=direct_gap_observable,
                    thermal_integral=thermal_integral,
                    delta_eq=delta_eq_per_T[i], delta_T=delta_T,
                )
                # Keep the sweep fail-loud even if a future test double or
                # alternate point solver bypasses _solve_and_measure's check.
                _require_finite_measurements(
                    obs,
                    delta_driven_pt,
                    x_qp_pt,
                    T_bath=T_bath,
                    n_bar=float(n_bar),
                )
                ok = True
            except _PointSolveError:
                ok = False

            if certificate is not None:
                for field in FIG6_CERTIFICATE_FIELDS:
                    certificates[field][i, j] = certificate[field]
                # This is not the physical gap-collapse fold: the solver
                # returned a state, but that state must actually be a root of
                # the kinetic equations before any observable is accepted or
                # folded to NaN.
                _require_target_certificate(
                    certificate,
                    T_bath=T_bath,
                    n_bar=float(n_bar),
                    require_gap_fixed_point=not direct_gap_observable,
                    backward_error_limit=(
                        DIRECT_GAP_BACKWARD_ERROR_LIMIT
                        if direct_gap_observable
                        else TARGET_BACKWARD_ERROR_LIMIT
                    ),
                )

            if not ok or converged is None:
                obs_num[i, j] = np.nan
                delta_driven[i, j] = np.nan
                x_qp_num[i, j] = np.nan
                print(
                    f"  T_B={T_bath:.2f} K  nbar={n_bar:.2e}  "
                    f"T_*/Delta={T_star[i, j]:.3f}  "
                    f"SOLVE FAILED (no acceptable solution); recorded NaN",
                    flush=True,
                )
                continue

            obs_num[i, j] = obs
            delta_driven[i, j] = delta_driven_pt
            x_qp_num[i, j] = x_qp_pt
            # Carry the complete converged state within this temperature row.
            # In particular, preserve the self-consistent gap and its matching
            # SpectralContext instead of silently resetting both to DELTA_0.
            continuation_seed = converged
            print(
                f"  T_B={T_bath:.2f} K  nbar={n_bar:.2e}  "
                f"T_*/Delta={T_star[i, j]:.3f}  "
                f"Delta_driven={delta_driven[i, j]:.4f} micro-eV  "
                f"(Delta_T-Delta)/(Delta_0-Delta_eq)="
                f"{obs_num[i, j]:+.4f}",
                flush=True,
            )

    if tau_l_used is None:
        raise RuntimeError(
            "Sweep produced no points — every T_bath at or above T_c?"
        )
    return (
        T_star, delta_eq_per_T, delta_driven, x_qp_num, x_qp_eq47,
        obs_num, obs_eq53, np.array([tau_l_used]), certificates,
    )


def solve(
    *,
    direct_gap_observable: bool = False,
    fixed_gap_kinetics: bool = False,
) -> dict[str, np.ndarray]:
    """Solve Fischer Fig. 6 on the paper grid; return the raw sweep arrays.

    The expensive half of Fig. 6 (historically ~6.04 h serial): for each T_B the
    self-consistent-gap (or fixed-gap + direct-Δ[f]) joint solve is swept over n̄
    with (f, n_ph) continuation, gated by the gap-suppression acceptance test (a
    fold sits past each curve's peak). The numerical observable is computed here
    because it gates the seed chain, so it cannot be deferred; the cache stores
    these raw arrays and :func:`fig6_paper.observables` repackages them.

    Grid / n̄ / picard-tol read the module globals (mutated by the ``--fast`` CLI
    path), so :func:`solver_fingerprint` — which reads the same globals — and the
    cache key both reflect ``--fast`` automatically.
    """
    _require_supported_mode(
        direct_gap_observable=direct_gap_observable,
        fixed_gap_kinetics=fixed_gap_kinetics,
    )
    material = _fischer_material()
    _, _, spectral = _build_grid_and_spectral()

    tau_0_pb = _compute_tau_0_pb(spectral)

    backend = T3DiffusionBackend()

    if direct_gap_observable:
        if fixed_gap_kinetics:
            print("Sweep nbar at three T_B with fixed gap + direct Delta[f] observable:")
        else:
            print("Sweep nbar at three T_B with direct Delta[f] observable:")
    else:
        print(
            "Sweep nbar at three T_B with self-consistent gap + "
            f"tau_l (model={TAU_L_MODEL!r}):"
        )
    (
        T_star, delta_eq_per_T, delta_driven, x_qp_num, x_qp_eq47,
        obs_num, obs_eq53, tau_l_arr, certificates,
    ) = _solve_sweep(
        backend,
        material,
        spectral,
        tau_0_pb,
        direct_gap_observable=direct_gap_observable,
        fixed_gap_kinetics=fixed_gap_kinetics,
    )
    tau_l_ns = float(tau_l_arr[0])

    _check_tau_0_pb(tau_0_pb, tau_l_ns)

    return {
        "tau_0_pb_ns": np.array([tau_0_pb]),
        "tau_l_ns": np.array([tau_l_ns]),
        "T_bath": np.array(T_BATH_VALUES, dtype=float),
        "n_bar": N_BAR_VALUES.copy(),
        "T_star_over_delta": T_star,
        "delta_eq": delta_eq_per_T,
        "delta_driven": delta_driven,
        "paper_observable_num": obs_num,
        "paper_observable_eq53": obs_eq53,
        "x_qp_num": x_qp_num,
        "x_qp_eq47": x_qp_eq47,
        **certificates,
    }


def solver_fingerprint() -> dict[str, Any]:
    """Resolved physics + grid + sweep knobs for the cache key's provenance.

    Reads the module globals (incl. the ``--fast``-mutated NUM_BINS /
    N_BAR_VALUES / PICARD_TOL and the occupation-space PICARD_ATOL), so the cache
    key distinguishes a ``--fast`` run from a paper run. Safety also rests on
    the full-module ``extra_source`` and the qpsim solver-subtree digest; this
    keeps the provenance legible.
    """
    return {
        "delta_0": DELTA_0,
        "finite_cutoff_delta0_over_kbtc": FINITE_CUTOFF_DELTA0_OVER_KBTC,
        "tau_0": TAU_0,
        "t_c": T_C,
        "omega_0": OMEGA_0,
        "c_phot": C_PHOT,
        "film_thickness_nm": FILM_THICKNESS_NM,
        "eta": SUBSTRATE_ETA,
        "num_bins": int(NUM_BINS),
        "e_min_factor": E_MIN_FACTOR,
        "e_max_factor": E_MAX_FACTOR,
        "picard_tol": PICARD_TOL,
        "picard_atol": PICARD_ATOL,
        "picard_balance_tol": TARGET_BACKWARD_ERROR_LIMIT,
        "newton_backward_error_tol": NEWTON_BACKWARD_ERROR_TOL,
        "target_backward_error_limit": TARGET_BACKWARD_ERROR_LIMIT,
        "direct_gap_newton_backward_error_tol": (
            DIRECT_GAP_NEWTON_BACKWARD_ERROR_TOL
        ),
        "direct_gap_backward_error_limit": DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        "gap_solve_xtol_uev": GAP_SOLVE_XTOL_UEV,
        "gap_fixed_point_abs_tol_uev": GAP_FIXED_POINT_ABS_TOL_UEV,
        "gap_fixed_point_rel_tol": GAP_FIXED_POINT_REL_TOL,
        "certificate_fields": list(FIG6_CERTIFICATE_FIELDS),
        "certificate_metric_version": CERTIFICATE_METRIC_VERSION,
        "tau_l_model": TAU_L_MODEL,
        "t_bath_values": [float(x) for x in T_BATH_VALUES],
        "n_bar_values": [float(x) for x in N_BAR_VALUES],
    }
