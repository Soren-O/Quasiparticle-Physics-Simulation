r"""M25GapAsymmetricJJ — Layer-2 wrap of the Stage A M25 physics.

A Junction subclass in the Device Architecture that
implements the Marchegiani-Catelani 2025 gap-asymmetric Josephson-
junction physics (with parity-tracked transmon qubit and pair-
breaking-photon drive). Wraps the Stage A coefficient evaluators
plus the 4-unknown moment solver to deliver quantitative M25 Fig 3
reproduction inside the Device contract.

Architecture:

* On the first ``evaluate`` call the Junction caches both
  ``M25Coefficients`` (state-independent) and the moment-solver
  fixed point ``(p_1, x_L, x_{R>}, x_{R<})`` from
  :func:`solve_rate_equation_steady_state`. Subsequent
  ``evaluate`` calls reuse both caches while their value-based input
  fingerprints match. Replacing any physics or branch-selection input
  automatically rebuilds the affected cache.
* Per-region ``ExternalFlux(gain, loss_rate)`` is built from the
  *cached* moment-solver values, NOT from integrating ``state.f``
  or reading ``qubit_state.p``. This sidesteps the cross-electrode
  tunneling bootstrap problem at Δ_L ≈ Δ_R, where the inner
  Newton's residual floor masks the x_L ↔ x_R> exchange and a
  state-driven Picard locks orders of magnitude below the M25
  fixed point. The Device solver's outer Picard converges in 2
  iterations because the emitted flux is constant across iterates.
* Per-region rates spread **uniformly in spectral measure over each
  electrode's active sub-band(s)**, normalized so the moment-integral identity
  ``(2/Δ_α) ∫ ρ × gain dE = gain_moment`` holds exactly.
  Pure-BCS cells use analytic DOS weights, and the R</R> boundary must be a
  cell face so one cell never carries two incompatible moment rates.
* Qubit channels: parity-flipping ``eo`` from QP tunneling
  (``Γ̃^α_{ij} × x_α`` for α ∈ {L, R>, R<}), parity-flipping ``eo``
  from photon-assisted tunneling (``Γ̃^{ph}_{ij}``), and parity-
  preserving ``ee`` (``Γ̃^{ee}_{ij}``). The Device's qubit master
  equation reaches the same fixed-point ``p_1`` as the moment
  solver because the channels carry the same effective rates.

E-ph double-counting: the moment-solver ``g_α`` and ``r_α x_α²``
already include the moment-integrated e-ph generation /
recombination. The class sets ``owns_region_dissipation = True``,
which the Device solver routes to the backend's
``external_dissipation_only=True`` path so the inner Newton sees
only the M25-supplied (gain, loss_rate) — not also the e-ph
collision kernel that would otherwise crush f(E) to thermal.

Caveats:

* The Stage A coefficient evaluators internally assume the Fermi-
  Dirac per-sub-band ansatz, so this remains a moment-closure
  Junction. A ``KineticJunction`` operating directly on f(E)
  would drop that assumption.
* ``qubit_state`` is accepted for API parity with other Junctions
  but ignored. The M25 master equation owns ``p_1`` self-
  consistently with the QP densities, so external qubit-state
  perturbations would over-determine the coupled system.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

import numpy as np

from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import Junction, JunctionResult
from qpsim.devices.qubit import QubitTransitionChannel
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)
from qpsim.physics.spectral import SpectralContext
from qpsim.services.rate_equation import (
    M25Coefficients,
    M25SteadyState,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    coefficients_from_physical_parameters_with_photon_drive,
)

if TYPE_CHECKING:
    from qpsim.backends.diffusion import DiffusionState
    from qpsim.devices.qubit import QubitState


def _float_dataclass_fingerprint(
    value: M25PhysicalParameters | M25PhotonDrive,
) -> tuple[float, ...]:
    """Snapshot every primitive input field for cache invalidation.

    The input bundles are frozen, but :class:`M25GapAsymmetricJJ` is
    intentionally mutable for backward compatibility: callers can replace
    either bundle after construction.  A value snapshot, rather than the
    bundle object itself, also remains correct if a caller bypasses the
    frozen guard with ``object.__setattr__``.
    """
    return tuple(float(getattr(value, item.name)) for item in fields(value))


# ─────────────────────────────────────────────────────────────────────
#  Moment extraction (Region f(E) → x_α)
# ─────────────────────────────────────────────────────────────────────


def _spectral_band_weights(
    spectral: SpectralContext,
    *,
    lower_bound: float,
    upper_bound: float | None = None,
) -> np.ndarray:
    """Return per-cell spectral measure for one physical energy band.

    Pure BCS cells use the analytic DOS primitive -- the DOS INTEGRATED across
    the cell, which stays finite at the gap edge where the DOS itself does not.
    Both paths require the grid to cover the requested lower boundary rather
    than silently dropping support.

    The Dynes path is the other convention: the DOS at the cell CENTRE times
    the geometric overlap, which is a point sample standing in for a cell
    average (see "Quasiparticles are stored as cell averages, phonons as point
    samples" in ``docs/Phonon_Model_Decisions.md``).

    Why that is tolerable HERE and not in the pure-BCS branch is worth being
    precise about, because the two failures look alike and are not. Without
    broadening the DOS is singular at the gap, and a point sample undercounts
    the gap-edge cell by exactly 1/sqrt(2) -- 29.3% -- at EVERY resolution;
    no mesh resolves a singularity, so that error never goes away, which is
    why this branch uses the exact integral instead. Broadening replaces the
    singularity with a Lorentzian of finite width Gamma, and a point sample of
    a smooth function is merely under-resolved: it converges normally, once
    the cells are small enough to see the peak. Measured gap-edge error
    against the exact cell integral, versus Gamma / dE:

        0.05 -> 82%    0.27 -> 14%    1.09 -> 2.9%    5.4 -> 0.10%

    So the requirement is a resolution condition -- cells no wider than the
    broadening -- and refining the grid at fixed Gamma moves RIGHT along that
    table and fixes it. The hazard is only that nothing stated the condition,
    and at the default resolution a physically small Gamma sits at the left
    end. Nothing downstream can detect it: the weights stay positive and sum
    to something plausible, and only the gap-edge cell is wrong.

    This is presently unreachable -- the transport layer rejects a Dynes
    context outright, so only the pure-BCS branch ships -- and the guard below
    is here so that implementing broadened kernels has to confront the
    resolution condition rather than inherit it silently.
    """
    if not np.isfinite(lower_bound):
        raise ValueError("lower_bound must be finite.")
    if upper_bound is not None and (
        np.isnan(upper_bound) or upper_bound <= lower_bound
    ):
        raise ValueError("upper_bound must be greater than lower_bound.")

    if spectral.dynes_gamma == 0.0:
        return bcs_dos_cell_weights(
            spectral.E,
            spectral.dE,
            spectral.gap,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    edges = cell_edges_from_widths(spectral.E, spectral.dE)
    # The point-sample path is only accurate while the broadening is at least
    # as wide as the cells it is being sampled on. Measure that against the
    # cells NEAR THE GAP, where the DOS structure actually is -- an average
    # over the whole grid is dominated by the wide high-energy cells and would
    # report the ratio as comfortable exactly when it is not.
    near_gap = np.abs(spectral.E - spectral.gap) <= 2.0 * spectral.dE
    widest = float(np.max(spectral.dE[near_gap])) if near_gap.any() else float(
        np.min(spectral.dE)
    )
    if spectral.dynes_gamma < widest:
        warnings.warn(
            "Dynes band weights are sampled at cell centres, which resolves "
            f"the gap edge only when the broadening (Gamma = "
            f"{spectral.dynes_gamma:g}) is at least the local cell width "
            f"({widest:g}). At this ratio ({spectral.dynes_gamma / widest:.3g}) "
            "the gap-edge weight is materially wrong; use finer energy bins "
            "near the gap. See 'Quasiparticles are stored as cell averages, "
            "phonons as point samples' in docs/Phonon_Model_Decisions.md.",
            RuntimeWarning,
            stacklevel=2,
        )

    coverage_tol = 128.0 * np.finfo(float).eps * max(
        1.0, abs(float(edges[0])), abs(lower_bound),
    )
    if float(edges[0]) > lower_bound + coverage_tol:
        raise ValueError(
            "The energy grid does not cover the requested M25 band edge: "
            f"first cell edge {float(edges[0]):g} > {lower_bound:g}."
        )
    hi_bound = float(edges[-1]) if upper_bound is None else upper_bound
    lo = np.maximum(edges[:-1], lower_bound)
    hi = np.minimum(edges[1:], hi_bound)
    overlap = np.maximum(hi - lo, 0.0)
    return spectral.rho * overlap


def _moment_x_M25(
    f: np.ndarray,
    spectral: SpectralContext,
    gap_alpha_uev: float,
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> float:
    r"""Compute M25-convention dimensionless density ``x_α``.

    .. math::

        x_α = \frac{2}{\Delta_α} \int_{\Delta_α}^∞ ρ_α(E) f_α(E) \, dE

    Parameters
    ----------
    f, spectral
        Region occupation and spectral context on the same energy grid.
    gap_alpha_uev
        Sub-band gap reference in μeV (Δ_L, Δ_R, etc.). Sets the
        normalization.
    lower_bound, upper_bound
        Optional physical band bounds. Cells crossed by a bound are split by
        their spectral measure rather than assigned by center location.
    """
    f_arr = np.asarray(f, dtype=float)
    if f_arr.shape != spectral.E.shape:
        raise ValueError(
            f"f must have shape {spectral.E.shape}; got {f_arr.shape}."
        )
    if not np.isfinite(gap_alpha_uev) or gap_alpha_uev <= 0.0:
        raise ValueError("gap_alpha_uev must be finite and positive.")
    band_lo = spectral.gap if lower_bound is None else lower_bound
    weights = _spectral_band_weights(
        spectral,
        lower_bound=band_lo,
        upper_bound=upper_bound,
    )
    return 2.0 * float(np.sum(weights * f_arr)) / gap_alpha_uev


# ─────────────────────────────────────────────────────────────────────
#  M25GapAsymmetricJJ
# ─────────────────────────────────────────────────────────────────────


@dataclass
class M25GapAsymmetricJJ(Junction):
    r"""Marchegiani-Catelani 2025 gap-asymmetric JJ as a Layer-2 Junction.

    Wraps Stage A's coefficient evaluators
    (:func:`coefficients_from_physical_parameters_with_photon_drive`)
    inside the Layer-2 Junction protocol. Caches ``M25Coefficients``
    on first ``evaluate`` (state-independent for fixed parameters) and
    refreshes the cache if its mutable junction inputs change.

    Parameters
    ----------
    name
        Junction identifier.
    region_a
        Name of the left (high-gap) electrode region in the parent
        Device.
    region_b
        Name of the right (low-gap) electrode region. Must have an
        energy grid spanning ``[Δ_R, ∞]`` so both R< (E ∈ [Δ_R, Δ_L])
        and R> (E ≥ Δ_L) sub-band moments can be extracted.
    m25_params
        Stage A :class:`M25PhysicalParameters` carrying gaps,
        ω_10, transmon E_J/E_C, T, etc.
    m25_drive
        Stage A :class:`M25PhotonDrive` for the pair-breaking
        photon channel.
    branch_picker_mode
        Fixed-point selection mode forwarded to
        :func:`solve_rate_equation_steady_state_multi_seed`. The
        default ``"min_residual"`` selects the true fixed point —
        with the single-quasiparticle normalization of the density
        equations (``M25Coefficients.cooper_pair_number_R``, set by
        the Note-V builder used here) the M25 system has a unique
        physical root and this mode finds it. The legacy
        ``"max_x_L"`` mode is DEPRECATED (emits ``DeprecationWarning``):
        it reproduces pre-normalization-fix baselines only and can pick
        sub-1-Hz pseudo-roots on the recombination slope; see the
        solver docstring and REVIEW-2026-07-04, finding 10.
    expected_ordering
        Optional moment-ordering hint forwarded to the solver
        (used by ``"min_residual"``/``"lock_to_preferred"`` modes).

    Notes
    -----
    See module docstring for the moment-closure approximations and
    the caveats.
    """

    name: str
    region_a: str
    region_b: str
    m25_params: M25PhysicalParameters
    m25_drive: M25PhotonDrive
    branch_picker_mode: str = "min_residual"
    expected_ordering: tuple[str, ...] | None = None
    # M25's external_flux already aggregates the moment-integrated
    # e-ph dissipation (g_α generation by thermal phonons + r_α x_α²
    # recombination), so the Device solver must run the inner
    # backend with external_dissipation_only=True to avoid
    # double-counting against the e-ph collision kernel.
    owns_region_dissipation: bool = field(default=True, init=False, repr=False)
    # The cached fixed point is an isolated two-electrode closure: evaluate()
    # intentionally does not consume state_a.f/state_b.f. Another Junction
    # touching either region would change f without feeding back into it.
    requires_exclusive_regions: bool = field(default=True, init=False, repr=False)
    _coefficients: M25Coefficients | None = field(default=None, init=False, repr=False)
    _coefficients_fingerprint: tuple[tuple[float, ...], tuple[float, ...]] | None = field(
        default=None, init=False, repr=False,
    )
    _moment_solution: M25SteadyState | None = field(
        default=None, init=False, repr=False,
    )
    _moment_solution_fingerprint: tuple[
        tuple[tuple[float, ...], tuple[float, ...]],
        str,
        tuple[str, ...] | None,
    ] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.region_a == self.region_b:
            raise ValueError(
                f"Junction must couple two different regions; got both "
                f"region_a and region_b = {self.region_a!r}."
            )

    def _coefficient_inputs_fingerprint(
        self,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return a value snapshot of all coefficient-builder inputs."""
        return (
            _float_dataclass_fingerprint(self.m25_params),
            _float_dataclass_fingerprint(self.m25_drive),
        )

    def _ensure_coefficients_cached(self) -> M25Coefficients:
        fingerprint = self._coefficient_inputs_fingerprint()
        if self._coefficients is None or self._coefficients_fingerprint != fingerprint:
            coefficients = coefficients_from_physical_parameters_with_photon_drive(
                self.m25_params, self.m25_drive,
            )
            object.__setattr__(
                self, "_coefficients", coefficients,
            )
            object.__setattr__(self, "_coefficients_fingerprint", fingerprint)
        assert self._coefficients is not None  # for mypy
        return self._coefficients

    def _ensure_moment_solution_cached(self) -> M25SteadyState:
        """Solve and cache the M25 4-unknown moment system.

        The M25 fixed point ``(p_1, x_L, x_{R>}, x_{R<})`` is purely a
        function of ``m25_params`` and ``m25_drive``; it does not
        depend on the input region states' ``f(E)``. Solving it
        bypasses the Device-level Picard outer loop's bootstrap
        problem (the inner Newton's residual floor masks the
        cross-electrode tunneling cycle x_L ↔ x_R> at Δ_L ≈ Δ_R
        inputs). The Device Picard then reduces to a one-shot
        per-region inner solve of ``f = gain/loss`` with the moment-
        consistent (gain, loss_rate).

        Uses :func:`solve_rate_equation_steady_state_multi_seed` so
        the photon-driven nonequilibrium branch is selected
        deterministically — single-seed calls land on the lower
        thermal-ish branch at typical M25 Fig 3 inputs.
        """
        coefs = self._ensure_coefficients_cached()
        expected_ordering = (
            None if self.expected_ordering is None else tuple(self.expected_ordering)
        )
        coefficient_fingerprint = self._coefficient_inputs_fingerprint()
        fingerprint = (
            coefficient_fingerprint,
            self.branch_picker_mode,
            expected_ordering,
        )
        if self._moment_solution is None or self._moment_solution_fingerprint != fingerprint:
            moment_solution = solve_rate_equation_steady_state_multi_seed(
                coefs,
                branch_picker_mode=self.branch_picker_mode,
                expected_ordering=self.expected_ordering,
            )
            object.__setattr__(
                self, "_moment_solution", moment_solution,
            )
            object.__setattr__(self, "_moment_solution_fingerprint", fingerprint)
        assert self._moment_solution is not None  # for mypy
        return self._moment_solution

    def evaluate(
        self,
        state_a: DiffusionState,
        state_b: DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        r"""Compute per-region (gain, loss_rate) + qubit channels.

        The M25 fixed point ``(p_1, x_L, x_{R>}, x_{R<})`` is solved
        analytically (cached) by the moment solver and
        used to build the moment-level rates here. The input
        ``state_a.f``, ``state_b.f``, and ``qubit_state.p`` are
        only consulted for grid metadata (gap, energy bins) — their
        actual occupation values do NOT feed back into the M25
        equations. This bypasses the Device-level Picard's
        bootstrap problem at Δ_L ≈ Δ_R, where the inner Newton's
        residual floor masks the cross-electrode tunneling cycle.

        ``qubit_state`` is accepted for API compatibility but
        unused; the M25 master equation owns ``p_1``.
        """
        del qubit_state  # M25 owns p_1; see _ensure_moment_solution_cached
        coefs = self._ensure_coefficients_cached()

        # ── Validate gap consistency before extracting masks ──
        Delta_L_uev = float(state_a.spectral.gap)
        Delta_R_uev = float(state_b.spectral.gap)

        # Region-state gaps must agree with the m25_params bundle the
        # cached coefficients were built from; otherwise we'd be
        # mixing rates for one junction with moments from another.
        # DiffusionState carries both `gap` and `spectral.gap` and
        # documents them to be in sync — check both, since downstream
        # physics (DOS, kinetic kernels) reads `state.gap` while the
        # M25 moment normalization reads `spectral.gap`.
        # Tolerances: the quantity at risk is not Δ but the asymmetry
        # ω_LR = Δ_L − Δ_R, which the whole sub-band structure is built on and
        # which is ~1/99 of Δ at the Fig. 3a point. A 0.1% per-gap slack there
        # admits ~20% of ω_LR — ≈9% in the emitted f_{R<} (the R< band measure
        # is √(Δ_L²−Δ_R²)) and ≈17% in the moments the cached coefficients
        # carry, i.e. more drift than the class's own acceptance pin. So hold
        # both gaps to round-trip noise instead: state.gap vs spectral.gap at
        # the Device-level invariant, spectral.gap vs m25_params at a few ULP
        # of the Kelvin→μeV conversion, and ω_LR itself explicitly. Every
        # supported caller builds both sides from the same
        # ``Δ_kelvin · KB_UEV_PER_K`` expression, so this is exact today.
        Delta_L_param_uev = self.m25_params.Delta_L_kelvin * KB_UEV_PER_K
        Delta_R_param_uev = self.m25_params.Delta_R_kelvin * KB_UEV_PER_K
        for region_name, region_state in (
            (self.region_a, state_a),
            (self.region_b, state_b),
        ):
            if not np.isclose(
                region_state.T_bath,
                self.m25_params.T_kelvin,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise ValueError(
                    f"Region {region_name!r} T_bath={region_state.T_bath:g} K "
                    "disagrees with the temperature used to build the M25 "
                    f"closure ({self.m25_params.T_kelvin:g} K). Rebuild the "
                    "junction parameters and region states at one temperature."
                )
        for region_label, region_name, state, spectral_gap_uev, param_gap_uev, param_field in (
            ("L", self.region_a, state_a, Delta_L_uev, Delta_L_param_uev, "Delta_L_kelvin"),
            ("R", self.region_b, state_b, Delta_R_uev, Delta_R_param_uev, "Delta_R_kelvin"),
        ):
            state_gap_uev = float(state.gap)
            gap_scale = max(abs(state_gap_uev), abs(spectral_gap_uev), 1.0)
            if not np.isclose(
                state_gap_uev,
                spectral_gap_uev,
                rtol=1e-12,
                atol=1e-12 * gap_scale,
            ):
                raise ValueError(
                    f"Region {region_name!r} state.gap "
                    f"{state_gap_uev / KB_UEV_PER_K:.6g} K disagrees with "
                    f"state.spectral.gap "
                    f"{spectral_gap_uev / KB_UEV_PER_K:.6g} K — "
                    "DiffusionState invariants violated."
                )
            if not np.isclose(
                spectral_gap_uev, param_gap_uev, rtol=1e-9, atol=0.0,
            ):
                raise ValueError(
                    f"Region {region_name!r} ({region_label}) gap "
                    f"{spectral_gap_uev / KB_UEV_PER_K:.6g} K does not match "
                    f"m25_params.{param_field} = "
                    f"{getattr(self.m25_params, param_field):.6g} K. "
                    "Coefficients and moments must be built from the same gaps."
                )

        # Pin the asymmetry directly: the R< band the coefficients' x_{R<}
        # is spread over is (Δ_R, Δ_L) taken from the states, while x_{R<}
        # itself comes from m25_params, so ω_LR is what has to agree.
        omega_LR_uev = Delta_L_uev - Delta_R_uev
        omega_LR_param_uev = Delta_L_param_uev - Delta_R_param_uev
        if not np.isclose(omega_LR_uev, omega_LR_param_uev, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"Region gap asymmetry ω_LR = Δ_L − Δ_R = "
                f"{omega_LR_uev / KB_UEV_PER_K:.6g} K disagrees with the "
                f"m25_params asymmetry {omega_LR_param_uev / KB_UEV_PER_K:.6g} K. "
                "The M25 sub-band structure lives entirely in ω_LR; rebuild "
                "the coefficients and the region states from one gap pair."
            )

        # ── L-electrode active band: positive DOS support above Δ_L.
        #    A boundary-only grid (max(E) == Δ_L) would pass a bin-
        #    counting check but BCS DOS is zero at E == Δ, so the
        #    rho × dE × mask integral inside _build_per_region_flux
        #    would still vanish and silently emit ExternalFlux.zero —
        #    deleting the M25 L moment terms.
        E_L = state_a.spectral.E
        mask_L = Delta_L_uev < E_L
        weights_L = _spectral_band_weights(
            state_a.spectral, lower_bound=Delta_L_uev,
        )
        if not np.any(mask_L & (weights_L > 0.0)):
            raise ValueError(
                f"Region {self.region_a!r} energy grid has no positive "
                f"DOS support in the L band E ≥ Δ_L = {Delta_L_uev:.6g} μeV "
                f"(grid max = {float(E_L.max()):.6g} μeV). Extend the grid "
                "upper bound or add bins with E strictly above Δ_L."
            )

        # ── R-electrode sub-band masks (still required for spreading) ─
        E_R = state_b.spectral.E
        mask_Rlt = (Delta_R_uev < E_R) & (Delta_L_uev > E_R)
        mask_Rgt = Delta_L_uev <= E_R
        weights_Rlt = _spectral_band_weights(
            state_b.spectral,
            lower_bound=Delta_R_uev,
            upper_bound=Delta_L_uev,
        )
        weights_Rgt = _spectral_band_weights(
            state_b.spectral,
            lower_bound=Delta_L_uev,
        )
        # Require an actual representative center in each band, not only a
        # cell sliver crossed by the boundary.
        if not np.any(mask_Rlt & (weights_Rlt > 0.0)):
            raise ValueError(
                f"Region {self.region_b!r} energy grid has no positive "
                f"DOS support in the R< sub-band (Δ_R, Δ_L) = "
                f"({Delta_R_uev:.6g}, {Delta_L_uev:.6g}) μeV. Extend the "
                "grid lower bound or add bins with E strictly above Δ_R."
            )
        if not np.any(mask_Rgt & (weights_Rgt > 0.0)):
            raise ValueError(
                f"Region {self.region_b!r} energy grid has no positive "
                f"DOS support in the R> sub-band E ≥ Δ_L = "
                f"{Delta_L_uev:.6g} μeV (grid max = {float(E_R.max()):.6g} "
                "μeV). Extend the grid upper bound."
            )

        # ── Use cached moment-solver fixed point ──
        moment = self._ensure_moment_solution_cached()
        p_1 = moment.p_1
        p_0 = moment.p_0
        x_L = moment.x_L
        x_Rgt = moment.x_Rgt
        x_Rlt = moment.x_Rlt

        # ── Per-region moment rates (M25 Eqs. 4-6 in Stage A form) ─
        # The density equations run on the single-quasiparticle rates
        # Γ̄ = Γ̃ / N_CP(R) (M25 text below Eq. 6) via the
        # M25Coefficients.density_gammas() chokepoint — same
        # convention as qpsim.services.rate_equation.
        # _rate_equation_residual. The qubit channels below keep the
        # ensemble Γ̃ rates.
        delta = coefs.delta
        gammas_L, gammas_Rgt, gammas_Rlt = coefs.density_gammas()

        # Bookkeeping objects (Stage A residual)
        T_L = (gammas_L[0, 0] + gammas_L[0, 1]) * p_0 + (gammas_L[1, 1] + gammas_L[1, 0]) * p_1
        T_Rgt = (
            (gammas_Rgt[0, 0] + gammas_Rgt[0, 1]) * p_0
            + (gammas_Rgt[1, 1] + gammas_Rgt[1, 0]) * p_1
        )
        S_L_to_Rgt = gammas_L[0, 0] * p_0 + (gammas_L[1, 1] + gammas_L[1, 0]) * p_1
        gamma_Rlt_10 = gammas_Rlt[1, 0]
        gamma_L_01 = gammas_L[0, 1]

        # g^{ph} per-state contributions weighted at the M25-solved p
        g_ph_L_eff = p_0 * coefs.g_ph_L_per_state[0] + p_1 * coefs.g_ph_L_per_state[1]
        g_ph_Rgt_eff = p_0 * coefs.g_ph_Rgt_per_state[0] + p_1 * coefs.g_ph_Rgt_per_state[1]
        g_ph_Rlt_eff = p_0 * coefs.g_ph_Rlt_per_state[0] + p_1 * coefs.g_ph_Rlt_per_state[1]

        # M25 Eq. 4 (L electrode):
        #   ẋ_L = g_L - r_L x_L² - δ T_L x_L + δ T_R> x_R> + δ Γ̃^R<_10 p_1 x_R<
        gain_L_moment = (
            coefs.g_L + g_ph_L_eff
            + delta * T_Rgt * x_Rgt
            + delta * gamma_Rlt_10 * p_1 * x_Rlt
        )
        loss_rate_L_moment = coefs.r_L * x_L + delta * T_L

        # M25 Eq. 5 (R> sub-band):
        gain_Rgt_moment = (
            coefs.g_Rgt + g_ph_Rgt_eff
            + S_L_to_Rgt * x_L
            + coefs.xi * gamma_L_01 * p_0 * x_L
            + coefs.tau_E_inv * x_Rlt
        )
        loss_rate_Rgt_moment = (
            coefs.r_Rgt * x_Rgt + coefs.r_cross * x_Rlt
            + T_Rgt + coefs.tau_R_inv
        )

        # M25 Eq. 6 (R< sub-band):
        gain_Rlt_moment = (
            coefs.g_Rlt + g_ph_Rlt_eff
            + (1.0 - coefs.xi) * gamma_L_01 * p_0 * x_L
            + coefs.tau_R_inv * x_Rgt
        )
        loss_rate_Rlt_moment = (
            coefs.r_Rlt * x_Rlt + coefs.r_cross * x_Rgt
            + gamma_Rlt_10 * p_1 + coefs.tau_E_inv
        )

        # ── Spread moment rates → per-bin (gain, loss_rate) ──
        # Pass the explicit M25 L lower bound. This excludes Dynes subgap
        # weight and uses the exact finite-volume spectral measure rather than
        # a center mask / rho(E_i)*dE_i approximation.
        ef_L = self._build_per_region_flux(
            state_a, gain_L_moment, loss_rate_L_moment,
            gap_uev=Delta_L_uev,
            lower_bound=Delta_L_uev,
            label="L",
        )
        ef_R = self._build_per_region_flux_two_band(
            state_b,
            gain_Rlt=gain_Rlt_moment, loss_rate_Rlt=loss_rate_Rlt_moment,
            gain_Rgt=gain_Rgt_moment, loss_rate_Rgt=loss_rate_Rgt_moment,
            Delta_R_uev=Delta_R_uev, Delta_L_uev=Delta_L_uev,
        )

        # ── Qubit channels: parity-flip (eo) + parity-preserve (ee) ──
        channels = self._build_qubit_channels(
            coefs, x_L=x_L, x_Rlt=x_Rlt, x_Rgt=x_Rgt,
        )

        return JunctionResult(
            external_flux_a=ef_L,
            external_flux_b=ef_R,
            qubit_channels=channels,
        )

    @staticmethod
    def _build_per_region_flux(
        state: DiffusionState,
        gain_moment: float,
        loss_rate_moment: float,
        *,
        gap_uev: float,
        lower_bound: float,
        upper_bound: float | None = None,
        label: str,
    ) -> ExternalFlux:
        """Spread a single (gain_moment, loss_rate_moment) over the
        active band of a region.

        The moment-integral identities (see module docstring) give:
        * loss_rate_per_bin = loss_rate_moment / 1e9  (Hz → 1/ns)
        * gain_per_bin = gain_moment × (Δ/2) / ∫_band ρ dE × (1/1e9)

        The band integral is a finite-volume spectral measure: analytic BCS
        cell weights (or cell-overlap Dynes weights), not ``rho(E_i)*dE_i``.
        """
        E = state.spectral.E
        weights = _spectral_band_weights(
            state.spectral,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        support = weights > 0.0
        rho_band_integral = float(np.sum(weights))
        if rho_band_integral <= 0.0:
            # Empty band; emit zero flux.
            NE = E.size
            return ExternalFlux.zero(NE)

        # Stage A rates are in Hz; ExternalFlux is in 1/ns.
        gain_per_bin_value = (
            gain_moment * gap_uev / (2.0 * rho_band_integral) * 1e-9
        )
        loss_rate_per_bin_value = loss_rate_moment * 1e-9

        gain = np.where(support, gain_per_bin_value, 0.0)
        loss_rate = np.where(support, loss_rate_per_bin_value, 0.0)
        return ExternalFlux(
            gain=gain, loss_rate=loss_rate,
            diagnostics={
                "junction_label": label,
                "gain_moment_Hz": gain_moment,
                "loss_rate_moment_Hz": loss_rate_moment,
            },
        )

    @staticmethod
    def _build_per_region_flux_two_band(
        state: DiffusionState,
        *,
        gain_Rlt: float, loss_rate_Rlt: float,
        gain_Rgt: float, loss_rate_Rgt: float,
        Delta_R_uev: float, Delta_L_uev: float,
    ) -> ExternalFlux:
        """Combine R< and R> moment rates into a single ExternalFlux
        on the full R-electrode energy grid."""
        E = state.spectral.E
        edges = cell_edges_from_widths(E, state.spectral.dE)
        alignment_scale = max(abs(Delta_L_uev), 1.0)
        if not np.any(
            np.isclose(edges, Delta_L_uev, rtol=0.0, atol=1e-12 * alignment_scale)
        ):
            raise ValueError(
                "The M25 R</R> boundary Delta_L must coincide with an "
                "energy-cell face. A cell that straddles Delta_L cannot "
                "carry two independent moment gain/loss rates; build a "
                "piecewise grid with an explicit face at Delta_L."
            )

        weights_Rlt = _spectral_band_weights(
            state.spectral,
            lower_bound=Delta_R_uev,
            upper_bound=Delta_L_uev,
        )
        weights_Rgt = _spectral_band_weights(
            state.spectral,
            lower_bound=Delta_L_uev,
        )
        support_Rlt = weights_Rlt > 0.0
        support_Rgt = weights_Rgt > 0.0
        if np.any(support_Rlt & support_Rgt):
            raise RuntimeError(
                "M25 band quadrature produced overlapping R</R> cell support."
            )

        rho_Rlt_int = float(np.sum(weights_Rlt))
        rho_Rgt_int = float(np.sum(weights_Rgt))

        gain = np.zeros_like(E)
        loss_rate = np.zeros_like(E)

        if rho_Rlt_int > 0.0:
            gain[support_Rlt] = (
                gain_Rlt * Delta_R_uev / (2.0 * rho_Rlt_int) * 1e-9
            )
            loss_rate[support_Rlt] = loss_rate_Rlt * 1e-9
        if rho_Rgt_int > 0.0:
            gain[support_Rgt] = (
                gain_Rgt * Delta_R_uev / (2.0 * rho_Rgt_int) * 1e-9
            )
            loss_rate[support_Rgt] = loss_rate_Rgt * 1e-9

        return ExternalFlux(
            gain=gain, loss_rate=loss_rate,
            diagnostics={
                "junction_label": "R",
                "gain_Rlt_Hz": gain_Rlt,
                "gain_Rgt_Hz": gain_Rgt,
                "loss_rate_Rlt_Hz": loss_rate_Rlt,
                "loss_rate_Rgt_Hz": loss_rate_Rgt,
            },
        )

    @staticmethod
    def _build_qubit_channels(
        coefs: M25Coefficients,
        *, x_L: float, x_Rlt: float, x_Rgt: float,
    ) -> list[QubitTransitionChannel]:
        """Emit M25 qubit transition channels for this iteration's x_α.

        Three families:
        * parity-flipping eo from QP tunneling: ``Γ̃^α_{ij} × x_α``
          for α ∈ {L, R>, R<} and (i, j) ∈ {00, 01, 10, 11}. Many
          entries are zero by kinematic constraints (R< only has
          [1,0]).
        * parity-flipping eo from photon-assisted tunneling:
          ``Γ̃^{ph}_{ij}``.
        * parity-preserving ee from the parity-conserving channel:
          ``Γ̃^{ee}_{ij}``.

        All rates are in 1/ns (converted from Stage A's Hz).
        """
        channels: list[QubitTransitionChannel] = []

        # eo channels from QP tunneling (parity flip per event)
        for alpha_label, gammas, x_alpha in (
            ("L", coefs.gammas_L, x_L),
            ("Rgt", coefs.gammas_Rgt, x_Rgt),
            ("Rlt", coefs.gammas_Rlt, x_Rlt),
        ):
            for i in range(2):
                for j in range(2):
                    rate_Hz = gammas[i, j] * x_alpha
                    if rate_Hz > 0.0:
                        channels.append(QubitTransitionChannel(
                            level_from=i, level_to=j,
                            rate_per_ns=rate_Hz * 1e-9,
                            flips_parity=True,
                            label=f"Γ̃^{alpha_label}_{i}{j}·x_{alpha_label}",
                        ))

        # eo channels from photon-assisted tunneling
        for i in range(2):
            for j in range(2):
                rate_Hz = coefs.gamma_ph[i, j]
                if rate_Hz > 0.0:
                    channels.append(QubitTransitionChannel(
                        level_from=i, level_to=j,
                        rate_per_ns=rate_Hz * 1e-9,
                        flips_parity=True,
                        label=f"Γ̃^ph_{i}{j}",
                    ))

        # ee channels (parity preserved)
        for i in range(2):
            for j in range(2):
                rate_Hz = coefs.gamma_ee[i, j]
                if rate_Hz > 0.0:
                    channels.append(QubitTransitionChannel(
                        level_from=i, level_to=j,
                        rate_per_ns=rate_Hz * 1e-9,
                        flips_parity=False,
                        label=f"Γ̃^ee_{i}{j}",
                    ))

        return channels
