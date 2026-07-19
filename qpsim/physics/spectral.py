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
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)


def _finite_array(name: str, values: np.ndarray) -> np.ndarray:
    """Convert an array-like input to float and reject NaN/infinity."""
    result = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _finite_scalar(name: str, value: float) -> float:
    """Convert a scalar input to float and reject NaN/infinity."""
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result}.")
    return result


def _non_negative_scalar(name: str, value: float) -> float:
    """Return a finite scalar that satisfies a non-negative contract."""
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative; got {result}.")
    return result


def bcs_density_of_states(E: np.ndarray, gap: float) -> np.ndarray:
    """BCS density of states ρ(E) = E / √(E² − Δ²) for E > Δ, else 0."""
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
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
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
    n2 = np.zeros_like(E, dtype=float)
    valid = gap < E
    n2[valid] = gap / np.sqrt(E[valid] ** 2 - gap ** 2)
    return n2


def dynes_density_of_states(E: np.ndarray, gap: float, gamma: float) -> np.ndarray:
    """Dynes DOS: Re{(E − iΓ) / √((E − iΓ)² − Δ²)}. Γ=0 falls back to BCS."""
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
    gamma = _non_negative_scalar("gamma", gamma)
    if gamma <= 0:
        return bcs_density_of_states(E, gap)
    z = E - 1j * gamma
    with np.errstate(invalid="ignore"):
        result = np.real(z / np.sqrt(z ** 2 - gap ** 2))
    return np.maximum(result, 0.0)


def coherence_factor_plus(E: np.ndarray, gap: float) -> np.ndarray:
    """K⁺(E_i, E_j) = 1 + Δ² / (E_i E_j). Shape ``(NE, NE)``."""
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
    E_prod = np.maximum(E[:, None] * E[None, :], 1e-30)
    return 1.0 + gap ** 2 / E_prod


def coherence_factor_minus(E: np.ndarray, gap: float) -> np.ndarray:
    """K⁻(E_i, E_j) = max(0, 1 − Δ² / (E_i E_j)). Shape ``(NE, NE)``."""
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
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
    E_bins = _finite_array("E_bins", E_bins)
    temperature = _finite_scalar("temperature", temperature)
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
        # Defensive copy: asarray aliases a contiguous float64 caller buffer, so
        # a later in-place mutation of the passed grid would silently desync the
        # cached ρ/K±/mask (all built from this E at construction time).
        self._E = _finite_array("E_bins", E_bins).ravel().copy()
        self._dE = _finite_array("dE_bins", dE_bins).ravel().copy()
        if self._E.size == 0:
            raise ValueError("E_bins must be non-empty.")
        if self._E.size != self._dE.size:
            raise ValueError("E_bins and dE_bins must have the same length.")
        if np.any(np.diff(self._E) <= 0.0):
            raise ValueError("E_bins must be strictly increasing.")
        if np.any(self._dE <= 0.0):
            raise ValueError("dE_bins must be positive.")
        self._dynes_gamma = _non_negative_scalar("dynes_gamma", dynes_gamma)
        self._D0 = _non_negative_scalar(
            "diffusion_coefficient", diffusion_coefficient,
        )
        self._rebuild_tolerance = _non_negative_scalar(
            "rebuild_tolerance", rebuild_tolerance,
        )
        self._active_margin_factor = _non_negative_scalar(
            "active_margin_factor", active_margin_factor,
        )

        self._gap: float = 0.0
        self._rho: np.ndarray = np.empty(0)
        self._cell_weights: np.ndarray = np.empty(0)
        self._cell_density: np.ndarray = np.empty(0)
        self._cell_anomalous_density: np.ndarray = np.empty(0)
        self._K_plus: np.ndarray = np.empty(0)
        self._K_minus: np.ndarray = np.empty(0)
        self._D_E: np.ndarray = np.empty(0)
        self._active_mask: np.ndarray = np.empty(0, dtype=bool)

        self._rebuild(float(gap))

    @staticmethod
    def _ro(arr: np.ndarray) -> np.ndarray:
        """Return a read-only view so callers can't mutate the cache in place."""
        view = arr.view()
        view.flags.writeable = False
        return view

    @property
    def gap(self) -> float:
        return self._gap

    @property
    def E(self) -> np.ndarray:
        return self._ro(self._E)

    @property
    def dE(self) -> np.ndarray:
        return self._ro(self._dE)

    @property
    def rho(self) -> np.ndarray:
        """Point-sampled DOS ``ρ(E_i)``, shape ``(NE,)``.

        Collision quadratures must use :attr:`cell_weights` for an
        integrated partner measure and :attr:`cell_density` for a density
        sampled on a fixed energy-transition lattice.  This point value is
        retained for analytic formulas and backwards-compatible inspection.
        """
        return self._ro(self._rho)

    @property
    def cell_weights(self) -> np.ndarray:
        r"""Represented DOS measure ``w_i = ∫_cell_i ρ(E) dE``.

        Pure-BCS cells are integrated analytically, including a cell cut by
        the square-root gap edge.  Dynes spectra are nonsingular and retain
        the midpoint rule ``rho * dE``.  These weights define quasiparticle
        capacity, conservation, remapping, and finite-volume observables.
        """
        return self._ro(self._cell_weights)

    @property
    def cell_density(self) -> np.ndarray:
        r"""Cell-average DOS ``rho_bar_i = cell_weights_i / dE_i``.

        Fixed-lattice photon rates and phonon line integrals use this density
        so their event flux is paired with the same finite-volume capacity as
        the quasiparticle equation.
        """
        return self._ro(self._cell_density)

    @property
    def cell_anomalous_density(self) -> np.ndarray:
        r"""Cell-average ideal-BCS anomalous weight ``N2(E)``.

        For pure BCS this is the analytic average of
        ``gap / sqrt(E**2-gap**2)`` over each represented cell.  It is zero
        outside the BCS band.  Dynes contexts currently expose the ideal-BCS
        point value times the support convention because broadened collision
        coherence factors are deliberately unsupported elsewhere.
        """
        return self._ro(self._cell_anomalous_density)

    @property
    def K_plus(self) -> np.ndarray:
        """Finite-volume ``K⁺`` coherence matrix, shape ``(NE, NE)``."""
        return self._ro(self._K_plus)

    @property
    def K_minus(self) -> np.ndarray:
        """Finite-volume ``K⁻`` coherence matrix, shape ``(NE, NE)``."""
        return self._ro(self._K_minus)

    @property
    def D_E(self) -> np.ndarray:
        """D(E) under the LEGACY closure, shape ``(NE,)``."""
        return self._ro(self._D_E)

    @property
    def active_mask(self) -> np.ndarray:
        """Bins with non-zero represented spectral capacity.

        Every represented state must participate in a steady-state solve.
        In particular, no numerical margin is removed above the BCS gap:
        excluding such a bin can hide a non-zero residual and produce a
        false steady state.  A pure-BCS cell cut by the gap is active even if
        its center lies below the gap; a Dynes DOS naturally includes its
        broadened sub-gap support.
        """
        return self._ro(self._active_mask)

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
        """Legacy configuration value, retained for API compatibility.

        The value no longer removes positive-capacity states from
        :attr:`active_mask`.
        """
        return self._active_margin_factor

    def maybe_rebuild(self, new_gap: float) -> bool:
        """Rebuild iff |Δ − Δ_new|/|Δ| > ``rebuild_tolerance``. Returns True on rebuild."""
        new_gap = _non_negative_scalar("new_gap", new_gap)
        rel_change = abs(new_gap - self._gap) / max(abs(self._gap), 1e-30)
        if rel_change <= self._rebuild_tolerance:
            return False
        self._rebuild(new_gap)
        return True

    def _rebuild(self, gap: float) -> None:
        gap = _non_negative_scalar("gap", gap)
        self._gap = gap
        E = self._E

        if self._dynes_gamma > 0:
            self._rho = dynes_density_of_states(E, gap, self._dynes_gamma)
            self._cell_weights = self._rho * self._dE
            self._cell_anomalous_density = bcs_anomalous_weight(E, gap)
        else:
            self._rho = bcs_density_of_states(E, gap)
            # A kinetic grid represents only its own finite-volume domain.
            # If its lower edge starts above the physical gap, do not invent
            # the missing interval; integrate exactly from that represented
            # edge.  Observable APIs may impose a stricter full-gap coverage
            # contract when the missing singular interval matters.
            first_edge = float(cell_edges_from_widths(E, self._dE)[0])
            self._cell_weights = bcs_dos_cell_weights(
                E,
                self._dE,
                gap,
                lower_bound=max(gap, first_edge),
            )
        self._cell_density = self._cell_weights / self._dE
        if not np.any(self._cell_weights > 0.0):
            raise ValueError(
                "SpectralContext requires positive represented spectral "
                "capacity above/broadened through the gap."
            )

        if self._dynes_gamma > 0.0:
            self._K_plus = coherence_factor_plus(E, gap)
            self._K_minus = coherence_factor_minus(E, gap)
        elif gap == 0.0:
            self._cell_anomalous_density = np.zeros_like(E)
            self._K_plus = np.ones((E.size, E.size))
            self._K_minus = np.ones((E.size, E.size))
        else:
            # K± = 1 ± (Delta/E_i)(Delta/E_j).  Under the product
            # finite-volume DOS measure the double-cell average therefore
            # factorizes exactly.  Integrate N2 = rho*Delta/E analytically;
            # this keeps a gap-cut cell physical even when its center lies
            # below Delta, where a point coherence factor is undefined.
            edges = cell_edges_from_widths(E, self._dE)
            lo = np.maximum(edges[:-1], gap)
            hi = np.maximum(edges[1:], lo)
            anomalous_weight = gap * (
                np.arccosh(np.maximum(hi / gap, 1.0))
                - np.arccosh(np.maximum(lo / gap, 1.0))
            )
            self._cell_anomalous_density = anomalous_weight / self._dE
            ratio = np.zeros_like(E)
            supported = self._cell_weights > 0.0
            ratio[supported] = (
                anomalous_weight[supported] / self._cell_weights[supported]
            )
            ratio = np.clip(ratio, 0.0, 1.0)
            product = ratio[:, None] * ratio[None, :]
            self._K_plus = 1.0 + product
            self._K_minus = np.maximum(1.0 - product, 0.0)

        if self._D0 > 0 and gap > 0:
            ratio = np.minimum(gap / E, 1.0)
            self._D_E = self._D0 * np.sqrt(np.maximum(0.0, 1.0 - ratio ** 2))
        elif self._D0 > 0:
            self._D_E = np.full_like(E, self._D0)
        else:
            self._D_E = np.zeros_like(E)

        # The support of the represented DOS measure is the only physically
        # meaningful active-set criterion.  A former ``0.1*dE`` gap-edge margin could
        # freeze a bin with finite BCS DOS and let Newton certify a false
        # steady state because that bin's residual was never inspected.
        # Exact BCS cell capacity additionally preserves cells cut by a moving
        # gap even when their center is sub-gap.  Dynes midpoint capacity
        # preserves its intended broadened sub-gap semantics.
        self._active_mask = self._cell_weights > 0.0
