"""BCS / Dynes spectral functions and the Δ-dependent SpectralContext cache.

Pure-physics module — no state mutation beyond ``SpectralContext``'s
own Δ cache. Inputs are energy grids and a gap value; outputs are
DOS arrays, coherence-factor matrices, and thermal weights.

``thermal_qp_weights`` lives here rather than with the phonon kernels:
it is a spectral × Fermi–Dirac combination, not a phonon kernel.
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
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued.")
    try:
        result = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric array.") from exc
    if np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _finite_scalar(name: str, value: float) -> float:
    """Convert a scalar input to float and reject NaN/infinity."""
    if isinstance(value, (bool, np.bool_)) or np.iscomplexobj(value):
        raise ValueError(f"{name} must be a finite real scalar; got {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{name} must be a finite real scalar; got {value!r}."
        ) from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite; got {result}.")
    return result


def _non_negative_scalar(name: str, value: float) -> float:
    """Return a finite scalar that satisfies a non-negative contract."""
    result = _finite_scalar(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative; got {result}.")
    return result


def _fermi_dirac_from_exponent(exponent: np.ndarray) -> np.ndarray:
    """Evaluate ``1 / (exp(x) + 1)`` over the full binary64 range."""
    x = np.asarray(exponent, dtype=float)
    if np.any(np.isnan(x)):
        raise ValueError("Fermi-Dirac exponents must not contain NaN.")
    occupation = np.empty_like(x)
    non_negative = x >= 0.0
    with np.errstate(over="ignore", under="ignore"):
        exp_negative = np.exp(-x[non_negative])
        occupation[non_negative] = exp_negative / (1.0 + exp_negative)
        exp_positive = np.exp(x[~non_negative])
        occupation[~non_negative] = 1.0 / (1.0 + exp_positive)
    return occupation


def _energy_over_kbt(
    energy: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """Evaluate ``energy / (k_B * temperature)`` without scale loss.

    Neither multiplying ``k_B * temperature`` nor either sequential division
    is safe over the whole binary64 range: an intermediate can overflow or
    underflow even when the final ratio is representable. Decomposing both
    variable inputs into mantissa and power of two keeps that intermediate in
    range and leaves only mathematically correct final overflow/underflow.
    """
    energy_mantissa, energy_exponent = np.frexp(energy)
    temperature_mantissa, temperature_exponent = np.frexp(temperature)
    mantissa = (
        energy_mantissa
        / temperature_mantissa
        / _KB_UEV_PER_K
    )
    with np.errstate(over="ignore", under="ignore"):
        return np.ldexp(
            mantissa,
            energy_exponent - temperature_exponent,
        )


def fermi_dirac_occupation(
    energy: np.ndarray | float,
    temperature: float,
) -> np.ndarray:
    """Return the Fermi-Dirac occupation on a non-negative energy grid.

    The sign-split exponential form preserves representable cold tails without
    overflowing at large ``E / (k_B T)``. At exactly zero temperature every
    positive-energy quasiparticle occupation is zero.
    """
    energy_array = _finite_array("energy", np.asarray(energy))
    if np.any(energy_array < 0.0):
        raise ValueError("energy must contain only non-negative values.")
    temperature_value = _non_negative_scalar("temperature", temperature)
    if temperature_value == 0.0:
        return np.zeros_like(energy_array)
    exponent = _energy_over_kbt(energy_array, temperature_value)
    return _fermi_dirac_from_exponent(exponent)


def bcs_density_of_states(E: np.ndarray, gap: float) -> np.ndarray:
    """BCS density of states ρ(E) = E / √(E² − Δ²) for E > Δ, else 0."""
    E = _finite_array("E", E)
    gap = _non_negative_scalar("gap", gap)
    rho = np.zeros_like(E, dtype=float)
    valid = gap < E
    # Factored, not E**2 - gap**2. Squaring first destroys the significance of
    # E - gap exactly where this function is most singular: the two squares
    # agree to many digits near the gap edge and their difference is nearly all
    # cancellation. By Sterbenz's lemma E - gap is EXACT in floating point for
    # gap/2 <= E <= 2*gap, which covers the whole gap-edge region, so the
    # factored form carries the small factor at full precision and only the
    # benign sum is rounded. Same convention already used by
    # bcs_quadrature.py:142-145 and gap_equation.py:192-200 -- this brings the
    # two remaining sites onto it.
    rho[valid] = E[valid] / np.sqrt((E[valid] - gap) * (E[valid] + gap))
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
    # Factored for the same reason as bcs_density_of_states above.
    n2[valid] = gap / np.sqrt((E[valid] - gap) * (E[valid] + gap))
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
    fermi = fermi_dirac_occupation(E_bins, temperature)
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
        """Configuration value retained for API compatibility.

        It does not remove positive-capacity states from
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
        E = self._E

        # Compute the complete replacement state into locals and commit
        # only after every validation has passed. A failed rebuild must
        # leave the object exactly as it was: committing gap/rho/weights
        # before the positive-capacity check raises leaves a torn object
        # mixing two gaps, and the pre-committed gap then makes a retry
        # maybe_rebuild() near the failed gap silently no-op.
        if self._dynes_gamma > 0:
            rho = dynes_density_of_states(E, gap, self._dynes_gamma)
            cell_weights = rho * self._dE
            cell_anomalous_density = bcs_anomalous_weight(E, gap)
        else:
            rho = bcs_density_of_states(E, gap)
            # A kinetic grid represents only its own finite-volume domain.
            # If its lower edge starts above the physical gap, do not invent
            # the missing interval; integrate exactly from that represented
            # edge.  Observable APIs may impose a stricter full-gap coverage
            # contract when the missing singular interval matters.
            first_edge = float(cell_edges_from_widths(E, self._dE)[0])
            cell_weights = bcs_dos_cell_weights(
                E,
                self._dE,
                gap,
                lower_bound=max(gap, first_edge),
            )
        cell_density = cell_weights / self._dE
        if not np.any(cell_weights > 0.0):
            raise ValueError(
                "SpectralContext requires positive represented spectral "
                "capacity above/broadened through the gap."
            )

        if self._dynes_gamma > 0.0:
            K_plus = coherence_factor_plus(E, gap)
            K_minus = coherence_factor_minus(E, gap)
        elif gap == 0.0:
            cell_anomalous_density = np.zeros_like(E)
            K_plus = np.ones((E.size, E.size))
            K_minus = np.ones((E.size, E.size))
        else:
            # K± = 1 ± (Delta/E_i)(Delta/E_j).  Under the product
            # finite-volume DOS measure the double-cell average therefore
            # factorizes exactly.  Integrate N2 = rho*Delta/E analytically;
            # this keeps a gap-cut cell physical even when its center lies
            # below Delta, where a point coherence factor is undefined.
            edges = cell_edges_from_widths(E, self._dE)
            lo = np.maximum(edges[:-1], gap)
            hi = np.maximum(edges[1:], lo)
            # asinh(sqrt(E^2 - D^2)/D) == acosh(E/D) identically, but the two
            # differ sharply in floating point at the gap edge. acosh(x) loses
            # half its significant digits as x -> 1, and the argument E/gap is
            # ALREADY rounded before acosh sees it, so the information is gone
            # before the function is called. The arcsinh form carries the small
            # quantity xi = sqrt((E-D)(E+D)) explicitly, and E - gap is exact
            # in floating point over gap/2 <= E <= 2*gap by Sterbenz's lemma --
            # precisely the gap-edge region where this cell measure matters.
            # The same operation sequence is mirrored in the C3 author-score
            # script and compared bit-exact, so the sequence itself is
            # load-bearing: reordering it here breaks that comparison.
            xi_lo = np.sqrt(np.maximum((lo - gap) * (lo + gap), 0.0))
            xi_hi = np.sqrt(np.maximum((hi - gap) * (hi + gap), 0.0))
            anomalous_weight = gap * (
                np.arcsinh(xi_hi / gap) - np.arcsinh(xi_lo / gap)
            )
            cell_anomalous_density = anomalous_weight / self._dE
            ratio = np.zeros_like(E)
            supported = cell_weights > 0.0
            ratio[supported] = (
                anomalous_weight[supported] / cell_weights[supported]
            )
            ratio = np.clip(ratio, 0.0, 1.0)
            product = ratio[:, None] * ratio[None, :]
            K_plus = 1.0 + product
            K_minus = np.maximum(1.0 - product, 0.0)

        if self._D0 > 0 and gap > 0:
            # Left as 1 - (Δ/E)² DELIBERATELY, unlike the factored radicand
            # above and the one in observables/ac_conductivity.py. The
            # cancellation is real but this expression is evaluated on the
            # ENERGY GRID, whose first node sits half a bin above Δ, and there
            # it never bites. Measured against 60-digit arithmetic at the
            # closest node of three shipped grids:
            #
            #   405 bins  [Δ,10Δ]   2 µeV above Δ    1.1e-15 → 1.6e-16
            #   1620 bins [Δ,10Δ]   0.5 µeV          8.2e-15 → 1.9e-16
            #   66 bins   [Δ,4Δ]    4.1 µeV          4.7e-16 → 1.5e-16
            #
            # Every one is already at the noise floor and orders below the
            # discretisation error of anything that consumes D(E), so
            # rewriting it would move stored numbers for no reachable gain.
            # The distinguishing question at each of these three sites is
            # whether the nodes actually approach the gap edge: the σ₂
            # substitution puts them there on purpose and reaches 5e-05 µeV,
            # which is why that one was reformulated and this one was not.
            d_ratio = np.minimum(gap / E, 1.0)
            D_E = self._D0 * np.sqrt(np.maximum(0.0, 1.0 - d_ratio ** 2))
        elif self._D0 > 0:
            D_E = np.full_like(E, self._D0)
        else:
            D_E = np.zeros_like(E)

        # The support of the represented DOS measure is the only physically
        # meaningful active-set criterion.  A former ``0.1*dE`` gap-edge margin could
        # freeze a bin with finite BCS DOS and let Newton certify a false
        # steady state because that bin's residual was never inspected.
        # Exact BCS cell capacity additionally preserves cells cut by a moving
        # gap even when their center is sub-gap.  Dynes midpoint capacity
        # preserves its intended broadened sub-gap semantics.
        active_mask = cell_weights > 0.0

        # Commit — nothing below this line may raise.
        self._gap = gap
        self._rho = rho
        self._cell_weights = cell_weights
        self._cell_anomalous_density = cell_anomalous_density
        self._cell_density = cell_density
        self._K_plus = K_plus
        self._K_minus = K_minus
        self._D_E = D_E
        self._active_mask = active_mask
