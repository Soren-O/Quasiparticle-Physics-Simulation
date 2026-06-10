"""BCS / Dynes spectral functions and the Δ-dependent SpectralContext cache.

Pure-physics module — no state mutation beyond ``SpectralContext``'s
own Δ cache. Inputs are energy grids and a gap value; outputs are
DOS arrays, coherence-factor matrices, and thermal weights.

Ported from the old ``qpsim/numerics/spectral.py`` at Gate 2, with
``thermal_qp_weights`` moved here from the old ``kernels.py`` (it is
a spectral × Fermi–Dirac combination, not a phonon kernel).
"""

from __future__ import annotations

import numpy as np

from qpsim.constants import KB_UEV_PER_K as _KB_UEV_PER_K


def bcs_density_of_states(E: np.ndarray, gap: float) -> np.ndarray:
    """BCS density of states ρ(E) = E / √(E² − Δ²) for E > Δ, else 0."""
    rho = np.zeros_like(E, dtype=float)
    valid = gap < E
    rho[valid] = E[valid] / np.sqrt(E[valid] ** 2 - gap ** 2)
    return rho


def bcs_anomalous_weight(E: np.ndarray, gap: float) -> np.ndarray:
    """BCS anomalous weight N₂(E) = Δ / √(E² − Δ²) for E > Δ, else 0.

    The companion to :func:`bcs_density_of_states` (N₂ = (Δ/E) N₁); it
    enters the coherence-factor combination N₁N₁′ − N₂N₂′ carried by the
    Kupriyanov–Lukichev interface projection of the energy channel.
    """
    n2 = np.zeros_like(E, dtype=float)
    valid = gap < E
    n2[valid] = gap / np.sqrt(E[valid] ** 2 - gap ** 2)
    return n2


def dynes_density_of_states(E: np.ndarray, gap: float, gamma: float) -> np.ndarray:
    """Dynes DOS: Re{(E − iΓ) / √((E − iΓ)² − Δ²)}. Γ=0 falls back to BCS."""
    if gamma <= 0:
        return bcs_density_of_states(E, gap)
    z = E - 1j * gamma
    with np.errstate(invalid="ignore"):
        result = np.real(z / np.sqrt(z ** 2 - gap ** 2))
    return np.maximum(result, 0.0)


def coherence_factor_plus(E: np.ndarray, gap: float) -> np.ndarray:
    """K⁺(E_i, E_j) = 1 + Δ² / (E_i E_j). Shape ``(NE, NE)``."""
    E_prod = np.maximum(E[:, None] * E[None, :], 1e-30)
    return 1.0 + gap ** 2 / E_prod


def coherence_factor_minus(E: np.ndarray, gap: float) -> np.ndarray:
    """K⁻(E_i, E_j) = max(0, 1 − Δ² / (E_i E_j)). Shape ``(NE, NE)``."""
    E_prod = np.maximum(E[:, None] * E[None, :], 1e-30)
    return np.maximum(1.0 - gap ** 2 / E_prod, 0.0)


def thermal_qp_weights(
    E_bins: np.ndarray,
    gap: float,
    temperature: float,
    dynes_gamma: float = 0.0,
) -> np.ndarray:
    """Thermal quasiparticle weight ρ(E) · f_FD(E, T).

    Returns zeros at ``temperature <= 0`` (no thermal quasiparticles).
    ``dynes_gamma > 0`` uses the Dynes DOS; 0 falls back to pure BCS.
    """
    rho = dynes_density_of_states(E_bins, gap, dynes_gamma)
    if temperature <= 0:
        return np.zeros_like(rho)
    kT = _KB_UEV_PER_K * temperature
    exponent = np.minimum(E_bins / kT, 500.0)
    fermi = 1.0 / (np.exp(exponent) + 1.0)
    return rho * fermi


class SpectralContext:
    """Cached container for Δ-dependent spectral quantities.

    Holds DOS ρ(E), coherence-factor matrices K±, the energy-dependent
    diffusion coefficient under the LEGACY closure
    (``D(E) = D₀ √(1 − (Δ/E)²)``), and an active-energy mask.
    ``maybe_rebuild(new_gap)`` recomputes only when the gap has moved
    beyond ``rebuild_tolerance``.

    Scalar-Δ. Multi-material / spatially-varying-Δ runs hold one
    ``SpectralContext`` per distinct-Δ slot (see ``GapState``).

    Note on diffusion closures: this class uses the LEGACY form for
    backward compatibility with the reference implementation. The
    BOLTZMANN / USADEL alternatives live in ``qpsim.transport.diffusion``
    and operate on the occupation field directly rather than through
    an energy-indexed coefficient array.
    """

    def __init__(
        self,
        E_bins: np.ndarray,
        dE_bins: np.ndarray,
        gap: float,
        *,
        dynes_gamma: float = 0.0,
        diffusion_coefficient: float = 0.0,
        rebuild_tolerance: float = 1e-4,
        active_margin_factor: float = 0.1,
    ) -> None:
        self._E = np.asarray(E_bins, dtype=float).ravel()
        self._dE = np.asarray(dE_bins, dtype=float).ravel()
        if self._E.size != self._dE.size:
            raise ValueError("E_bins and dE_bins must have the same length.")
        self._dynes_gamma = float(dynes_gamma)
        self._D0 = float(diffusion_coefficient)
        self._rebuild_tolerance = float(rebuild_tolerance)
        self._active_margin_factor = float(active_margin_factor)

        self._gap: float = 0.0
        self._rho: np.ndarray = np.empty(0)
        self._K_plus: np.ndarray = np.empty(0)
        self._K_minus: np.ndarray = np.empty(0)
        self._D_E: np.ndarray = np.empty(0)
        self._active_mask: np.ndarray = np.empty(0, dtype=bool)

        self._rebuild(float(gap))

    @property
    def gap(self) -> float:
        return self._gap

    @property
    def E(self) -> np.ndarray:
        return self._E

    @property
    def dE(self) -> np.ndarray:
        return self._dE

    @property
    def rho(self) -> np.ndarray:
        """DOS ρ(E), shape ``(NE,)``."""
        return self._rho

    @property
    def K_plus(self) -> np.ndarray:
        """K⁺(E_i, E_j), shape ``(NE, NE)``."""
        return self._K_plus

    @property
    def K_minus(self) -> np.ndarray:
        """K⁻(E_i, E_j), shape ``(NE, NE)``."""
        return self._K_minus

    @property
    def D_E(self) -> np.ndarray:
        """D(E) under the LEGACY closure, shape ``(NE,)``."""
        return self._D_E

    @property
    def active_mask(self) -> np.ndarray:
        """``E ≥ Δ + margin·dE_local``, shape ``(NE,)`` bool.

        ``dE_local`` is the bin spacing at the first bin above the gap
        (equals ``mean(dE)`` on uniform grids; preserves correct
        margin behavior on piecewise / nonuniform grids).
        """
        return self._active_mask

    @property
    def dynes_gamma(self) -> float:
        return self._dynes_gamma

    @property
    def diffusion_coefficient(self) -> float:
        """Normal-state diffusion coefficient D₀ (0 disables D(E))."""
        return self._D0

    @property
    def rebuild_tolerance(self) -> float:
        """Relative-Δ threshold for :meth:`maybe_rebuild`."""
        return self._rebuild_tolerance

    @property
    def active_margin_factor(self) -> float:
        """Fractional margin above Δ used to set :attr:`active_mask`."""
        return self._active_margin_factor

    def maybe_rebuild(self, new_gap: float) -> bool:
        """Rebuild iff |Δ − Δ_new|/|Δ| > ``rebuild_tolerance``. Returns True on rebuild."""
        rel_change = abs(new_gap - self._gap) / max(abs(self._gap), 1e-30)
        if rel_change <= self._rebuild_tolerance:
            return False
        self._rebuild(new_gap)
        return True

    def _rebuild(self, gap: float) -> None:
        self._gap = float(gap)
        E = self._E

        if self._dynes_gamma > 0:
            self._rho = dynes_density_of_states(E, gap, self._dynes_gamma)
        else:
            self._rho = bcs_density_of_states(E, gap)

        self._K_plus = coherence_factor_plus(E, gap)
        self._K_minus = coherence_factor_minus(E, gap)

        if self._D0 > 0 and gap > 0:
            ratio = np.minimum(gap / E, 1.0)
            self._D_E = self._D0 * np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))
        elif self._D0 > 0:
            self._D_E = np.full_like(E, self._D0)
        else:
            self._D_E = np.zeros_like(E)

        # Active-margin epsilon is set by the bin spacing local to the
        # gap edge, not by global statistics. On uniform grids this
        # equals mean(dE) (the legacy heuristic); on piecewise grids
        # (e.g. M25's two-band R electrode with dense R< near the gap
        # and sparse R> far above it) it picks up the fine R< spacing.
        # A previous attempt used global ``min(dE)``, but that lets a
        # tiny far-tail bin shrink epsilon globally and re-enables
        # near-gap bins the margin was meant to exclude.
        above_gap = gap < E
        if np.any(above_gap):
            first_above_gap = int(np.argmax(above_gap))
            local_dE = float(self._dE[first_above_gap])
        else:
            # No bin above the gap — exotic grid, fall back to the
            # legacy mean(dE) so this branch behaves identically to
            # the pre-Phase-5c default.
            local_dE = float(np.mean(self._dE))
        epsilon = self._active_margin_factor * local_dE
        self._active_mask = (gap + epsilon) <= E
