r"""M25GapAsymmetricJJ — Layer-2 wrap of the Stage A M25 physics.

Phase 5 of the Device Architecture: a Junction subclass that
implements the Marchegiani-Catelani 2025 gap-asymmetric Josephson-
junction physics (with parity-tracked transmon qubit and pair-
breaking-photon drive) using the Stage A coefficient evaluators
under the hood.

The Junction reads the current QP densities ``(x_L, x_{R<}, x_{R>})``
out of the two Region states (via Fischer-convention moment
integrals), feeds them into the Stage A rate-equation RHS, and
emits:

* per-region ``ExternalFlux(gain, loss_rate)`` for the L and R
  electrode kinetic equations, with the moment-level rates spread
  uniformly over each electrode's active energy band(s);
* a list of ``QubitTransitionChannel`` records covering all the
  M25 transition pathways: parity-flipping ``eo`` channels driven
  by QP tunneling (``Γ̃^α_{ij} × x_α`` for α ∈ {L, R>, R<}),
  parity-preserving ``ee`` channels (``Γ̃^{ee}_{ij}``), and
  parity-flipping photon-assisted channels (``Γ̃^{ph}_{ij}``).

Caveats (per ``docs/Device_Architecture.md`` §6.1):

* This is a **moment-closure** Junction: the Stage A evaluators
  internally assume the Fermi-Dirac per-sub-band ansatz. A
  ``KineticJunction`` operating directly on f(E) is the
  architecturally cleaner way to drop that assumption — deferred
  to Phase 5b/6 if quantitative M25 Fig 3 reproduction needs it.
* The moment-rate to per-bin (gain, loss_rate) spread is **uniform
  over the active band**, normalized so that the moment-integral
  identity ``(2/Δ_α) ∫ ρ × gain dE = gain_moment`` holds exactly.
  This is the simplest spread that preserves the M25 Eq. 4-6 RHS;
  more sophisticated kinematic shapes (placing the gain at the
  partner-energy bin selected by each tunneling channel) can be
  added later.
* The M25-side recombination ``r_α x_α²`` is non-linear in x_α and
  doesn't fit the (gain, loss_rate × f) per-bin form cleanly; we
  approximate as ``loss_rate = r_α × x_α^prev + ...`` using the
  previous outer-iteration's x_α. Inner Newton sees this as
  effectively constant; outer Picard updates it.
* M25 owns the moment-integrated e-ph dissipation (``r_α x_α²``
  + ``g_α``). The class sets ``owns_region_dissipation = True``;
  the Device solver routes that to the T3 backend's
  ``external_dissipation_only=True`` path which disables the e-ph
  scattering and recombination kernels for both touched regions
  during the inner solve. Without this routing the e-ph kernel
  would forcibly thermalize f(E) at every iteration and crush
  x_α to the bath Fermi-Dirac value (~1e-52 at typical inputs),
  drowning the M25 ExternalFlux. Phase 5b plumbing.
* The Picard scheme converges geometrically when the cross-
  electrode tunneling cycle (x_L ↔ x_R> via δ T_α) is below 1.
  At Fig 3a inputs this cycle is *near* 1 because Δ_L ≈ Δ_R, and
  the inner Newton's residual floor swamps the cross-tunneling
  signal — the outer Picard "converges" at a fixed point where
  x_L is many orders below the published M25 Fig 3 values.
  Quantitative reproduction needs a moment-coupled Picard
  (Anderson on (x_L, x_R<, x_R>) directly) — Phase 5c.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from qpsim.constants import KB_UEV_PER_K
from qpsim.devices.external_flux import ExternalFlux
from qpsim.devices.junction import Junction, JunctionResult
from qpsim.devices.qubit import QubitTransitionChannel
from qpsim.services.rate_equation import M25Coefficients
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    coefficients_from_physical_parameters_with_photon_drive,
)

if TYPE_CHECKING:
    from qpsim.backends.t3_diffusion import T3DiffusionState
    from qpsim.devices.qubit import QubitState


# ─────────────────────────────────────────────────────────────────────
#  Moment extraction (Region f(E) → x_α)
# ─────────────────────────────────────────────────────────────────────


def _moment_x_M25(
    f: np.ndarray, rho: np.ndarray, dE: np.ndarray,
    gap_alpha_uev: float, mask: np.ndarray | None = None,
) -> float:
    r"""Compute M25-convention dimensionless density ``x_α``.

    .. math::

        x_α = \frac{2}{\Delta_α} \int_{\Delta_α}^∞ ρ_α(E) f_α(E) \, dE

    Parameters
    ----------
    f, rho, dE
        Region's f(E), DOS ρ(E), and integration weights dE on the
        same energy grid (μeV).
    gap_alpha_uev
        Sub-band gap reference in μeV (Δ_L, Δ_R, etc.). Sets the
        normalization.
    mask
        Optional boolean mask selecting a sub-band (e.g. R< vs R>).
        Restricts the integral to ``E`` bins where the mask is True.
    """
    integrand = rho * f * dE
    if mask is not None:
        integrand = integrand * mask
    return 2.0 * float(np.sum(integrand)) / gap_alpha_uev


# ─────────────────────────────────────────────────────────────────────
#  M25GapAsymmetricJJ
# ─────────────────────────────────────────────────────────────────────


@dataclass
class M25GapAsymmetricJJ(Junction):
    r"""Marchegiani-Catelani 2025 gap-asymmetric JJ as a Layer-2 Junction.

    Wraps Stage A's coefficient evaluators
    (:func:`coefficients_from_physical_parameters_with_photon_drive`)
    inside the Layer-2 Junction protocol. Caches ``M25Coefficients``
    on first ``evaluate`` (state-independent for fixed parameters).

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

    Notes
    -----
    See module docstring for the moment-closure approximations and
    the deferred items.
    """

    name: str
    region_a: str
    region_b: str
    m25_params: M25PhysicalParameters
    m25_drive: M25PhotonDrive
    # M25's external_flux already aggregates the moment-integrated
    # e-ph dissipation (g_α generation by thermal phonons + r_α x_α²
    # recombination), so the Device solver must run the inner
    # T3 backend with external_dissipation_only=True to avoid
    # double-counting against the e-ph collision kernel.
    owns_region_dissipation: bool = field(default=True, init=False, repr=False)
    _coefficients: M25Coefficients | None = field(default=None, init=False, repr=False)
    _last_x_L: float = field(default=0.0, init=False, repr=False)
    _last_x_Rlt: float = field(default=0.0, init=False, repr=False)
    _last_x_Rgt: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.region_a == self.region_b:
            raise ValueError(
                f"Junction must couple two different regions; got both "
                f"region_a and region_b = {self.region_a!r}."
            )

    def _ensure_coefficients_cached(self) -> M25Coefficients:
        if self._coefficients is None:
            object.__setattr__(
                self, "_coefficients",
                coefficients_from_physical_parameters_with_photon_drive(
                    self.m25_params, self.m25_drive,
                ),
            )
        assert self._coefficients is not None  # for mypy
        return self._coefficients

    def evaluate(
        self,
        state_a: T3DiffusionState,
        state_b: T3DiffusionState,
        qubit_state: QubitState | None = None,
    ) -> JunctionResult:
        r"""Compute per-region (gain, loss_rate) + qubit channels.

        Picard outer loop semantics: this evaluator uses the current
        region states to produce both the kinetic-equation flux for
        each region and the qubit channels for that outer iteration.
        The next iteration re-evaluates with the updated states.
        """
        coefs = self._ensure_coefficients_cached()

        # ── Extract M25-convention moments from the region f(E)'s ──
        Delta_L_uev = float(state_a.spectral.gap)
        Delta_R_uev = float(state_b.spectral.gap)

        # Region-state gaps must agree with the m25_params bundle the
        # cached coefficients were built from; otherwise we'd be
        # mixing rates for one junction with moments from another.
        # T3DiffusionState carries both `gap` and `spectral.gap` and
        # documents them to be in sync — check both, since downstream
        # physics (DOS, kinetic kernels) reads `state.gap` while the
        # M25 moment normalization reads `spectral.gap`.
        # Tolerance: 0.1% — well below any physically meaningful gap
        # asymmetry but above float round-trip noise.
        Delta_L_param_uev = self.m25_params.Delta_L_kelvin * KB_UEV_PER_K
        Delta_R_param_uev = self.m25_params.Delta_R_kelvin * KB_UEV_PER_K
        for region_label, region_name, state, spectral_gap_uev, param_gap_uev, param_field in (
            ("L", self.region_a, state_a, Delta_L_uev, Delta_L_param_uev, "Delta_L_kelvin"),
            ("R", self.region_b, state_b, Delta_R_uev, Delta_R_param_uev, "Delta_R_kelvin"),
        ):
            state_gap_uev = float(state.gap)
            if not np.isclose(state_gap_uev, spectral_gap_uev, rtol=1e-3):
                raise ValueError(
                    f"Region {region_name!r} state.gap "
                    f"{state_gap_uev / KB_UEV_PER_K:.6g} K disagrees with "
                    f"state.spectral.gap "
                    f"{spectral_gap_uev / KB_UEV_PER_K:.6g} K — "
                    "T3DiffusionState invariants violated."
                )
            if not np.isclose(spectral_gap_uev, param_gap_uev, rtol=1e-3):
                raise ValueError(
                    f"Region {region_name!r} ({region_label}) gap "
                    f"{spectral_gap_uev / KB_UEV_PER_K:.6g} K does not match "
                    f"m25_params.{param_field} = "
                    f"{getattr(self.m25_params, param_field):.6g} K. "
                    "Coefficients and moments must be built from the same gaps."
                )

        # x_L: integrate f_L over [Δ_L, ∞]
        x_L = _moment_x_M25(
            state_a.f, state_a.spectral.rho, state_a.spectral.dE,
            gap_alpha_uev=Delta_L_uev,
        )

        # x_R< vs x_R>: split f_R at E = Δ_L
        E_R = state_b.spectral.E
        mask_Rlt = (Delta_R_uev <= E_R) & (Delta_L_uev > E_R)
        mask_Rgt = Delta_L_uev <= E_R
        # Both sub-bands must be populated by the R-electrode grid,
        # otherwise an entire M25 channel silently drops out.
        if not np.any(mask_Rlt):
            raise ValueError(
                f"Region {self.region_b!r} energy grid does not span the "
                f"R< sub-band [Δ_R, Δ_L) = [{Delta_R_uev:.6g}, "
                f"{Delta_L_uev:.6g}] μeV. Extend the grid lower bound."
            )
        if not np.any(mask_Rgt):
            raise ValueError(
                f"Region {self.region_b!r} energy grid does not reach the "
                f"R> sub-band E ≥ Δ_L = {Delta_L_uev:.6g} μeV "
                f"(grid max = {float(E_R.max()):.6g} μeV). Extend the grid "
                "upper bound."
            )
        x_Rlt = _moment_x_M25(
            state_b.f, state_b.spectral.rho, state_b.spectral.dE,
            gap_alpha_uev=Delta_R_uev, mask=mask_Rlt,
        )
        x_Rgt = _moment_x_M25(
            state_b.f, state_b.spectral.rho, state_b.spectral.dE,
            gap_alpha_uev=Delta_R_uev, mask=mask_Rgt,
        )

        # Qubit populations: needed for parity-flip eo channels driven
        # by Γ̃^α_{ij} × x_α and S_L→R> / T objects.
        if qubit_state is not None:
            p = qubit_state.p
            p_per_level = p.sum(axis=1) if p.ndim == 2 else p
            p_0 = float(p_per_level[0])
            p_1 = float(p_per_level[1]) if len(p_per_level) > 1 else 0.0
        else:
            p_0, p_1 = 1.0, 0.0

        # ── Per-region moment rates (M25 Eqs. 4-6 in Stage A form) ─
        delta = coefs.delta
        gammas_L = coefs.gammas_L
        gammas_Rgt = coefs.gammas_Rgt
        gammas_Rlt = coefs.gammas_Rlt

        # Bookkeeping objects (Stage A residual)
        T_L = (gammas_L[0, 0] + gammas_L[0, 1]) * p_0 + (gammas_L[1, 1] + gammas_L[1, 0]) * p_1
        T_Rgt = (
            (gammas_Rgt[0, 0] + gammas_Rgt[0, 1]) * p_0
            + (gammas_Rgt[1, 1] + gammas_Rgt[1, 0]) * p_1
        )
        S_L_to_Rgt = gammas_L[0, 0] * p_0 + (gammas_L[1, 1] + gammas_L[1, 0]) * p_1
        gamma_Rlt_10 = gammas_Rlt[1, 0]
        gamma_L_01 = gammas_L[0, 1]

        # g^{ph} per-state contributions averaged at this iteration's p
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
        loss_rate_L_moment = coefs.r_L * x_L + delta * T_L  # uses x_L from current iter

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
        ef_L = self._build_per_region_flux(
            state_a, gain_L_moment, loss_rate_L_moment,
            gap_uev=Delta_L_uev, mask=None,  # full L band [Δ_L, ∞]
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

        # Cache moments for the next outer-iteration's use
        object.__setattr__(self, "_last_x_L", x_L)
        object.__setattr__(self, "_last_x_Rlt", x_Rlt)
        object.__setattr__(self, "_last_x_Rgt", x_Rgt)

        return JunctionResult(
            external_flux_a=ef_L,
            external_flux_b=ef_R,
            qubit_channels=channels,
        )

    @staticmethod
    def _build_per_region_flux(
        state: T3DiffusionState,
        gain_moment: float,
        loss_rate_moment: float,
        *,
        gap_uev: float,
        mask: np.ndarray | None,
        label: str,
    ) -> ExternalFlux:
        """Spread a single (gain_moment, loss_rate_moment) over the
        active band of a region.

        The moment-integral identities (see module docstring) give:
        * loss_rate_per_bin = loss_rate_moment / 1e9  (Hz → 1/ns)
        * gain_per_bin = gain_moment × (Δ/2) / ∫_band ρ dE × (1/1e9)
        """
        rho = state.spectral.rho
        dE = state.spectral.dE
        E = state.spectral.E
        if mask is None:
            mask = np.ones_like(E, dtype=bool)
        rho_band_integral = float(np.sum(rho * dE * mask))
        if rho_band_integral <= 0.0:
            # Empty band; emit zero flux.
            NE = E.size
            return ExternalFlux.zero(NE)

        # Stage A rates are in Hz; ExternalFlux is in 1/ns.
        gain_per_bin_value = (
            gain_moment * gap_uev / (2.0 * rho_band_integral) * 1e-9
        )
        loss_rate_per_bin_value = loss_rate_moment * 1e-9

        gain = np.where(mask, gain_per_bin_value, 0.0)
        loss_rate = np.where(mask, loss_rate_per_bin_value, 0.0)
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
        state: T3DiffusionState,
        *,
        gain_Rlt: float, loss_rate_Rlt: float,
        gain_Rgt: float, loss_rate_Rgt: float,
        Delta_R_uev: float, Delta_L_uev: float,
    ) -> ExternalFlux:
        """Combine R< and R> moment rates into a single ExternalFlux
        on the full R-electrode energy grid."""
        E = state.spectral.E
        rho = state.spectral.rho
        dE = state.spectral.dE

        mask_Rlt = (Delta_R_uev <= E) & (Delta_L_uev > E)
        mask_Rgt = Delta_L_uev <= E

        rho_Rlt_int = float(np.sum(rho * dE * mask_Rlt))
        rho_Rgt_int = float(np.sum(rho * dE * mask_Rgt))

        gain = np.zeros_like(E)
        loss_rate = np.zeros_like(E)

        if rho_Rlt_int > 0.0:
            gain[mask_Rlt] = gain_Rlt * Delta_R_uev / (2.0 * rho_Rlt_int) * 1e-9
            loss_rate[mask_Rlt] = loss_rate_Rlt * 1e-9
        if rho_Rgt_int > 0.0:
            gain[mask_Rgt] = gain_Rgt * Delta_R_uev / (2.0 * rho_Rgt_int) * 1e-9
            loss_rate[mask_Rgt] = loss_rate_Rgt * 1e-9

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
