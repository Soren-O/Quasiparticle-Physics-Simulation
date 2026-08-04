"""One-dimensional spatial T3 diffusion backend.

This is a narrow Gate-5 preview for strip-like devices: ``f(E, x)`` on a
uniform 1D mesh, reflective end boundaries, Crank-Nicolson diffusion in
space, and local T3 electron-phonon collisions at fixed gap.  Phonons are
held at the thermal bath in this first spatial path; finite-``tau_l``
spatial phonons can be layered on once the Ph1/Ph2 state is available.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import sparse
from scipy.integrate import quad

from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.materials.database import Material
from qpsim.physics.bcs_quadrature import (
    bcs_dos_cell_weights,
    cell_edges_from_widths,
)
from qpsim.physics.spectral import (
    SpectralContext,
)
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.solvers.etd import etd2_step
from qpsim.transport.diffusion.base import (
    DEFAULT_DIFFUSION_MODEL,
    DiffusionModel,
    density_weight,
    flux_weight,
)

# Hard backstop on emitted snapshots (see the 0-D transient driver): a
# ``snapshot_interval`` far below ``dt`` would otherwise drain an unbounded
# inner cadence loop before cancellation is polled. The webui schema rejects
# dense cadences up front; this protects direct callers.
_SNAPSHOT_HARD_CAP = 200_000

# Each exact local-gap collision operator retains three dense ``NE x NE``
# matrices. Up to two distinct gaps can be advanced in one batched ETD call.
# Larger profiles are advanced one gap-group at a time, so exact smooth-profile
# collisions remain O(NE**2) in resident kernel memory rather than
# O(NX * NE**2).
_MAX_BATCHED_COLLISION_GAPS = 2
_MAX_COLLISION_CACHE_ENTRIES = 2

# Crank--Nicolson is A-stable but not L-stable: for a semidiscrete diffusion
# eigenvalue ``lambda < 0`` its amplification factor
# ``(1 + h*lambda/2) / (1 - h*lambda/2)`` changes sign when
# ``h*abs(lambda) > 2``. The finite-volume generator below obeys
# ``abs(lambda_max) <= 2*max_i(exit_rate_i)``. Keeping
# ``h*max(exit_rate) <= 1`` therefore prevents stiff modal phase flips while
# retaining conservative, second-order CN substeps.
_MAX_CN_EXIT_STEP = 1.0
_MAX_CN_SUBSTEPS = 1_000_000

#: One energy's cached Crank-Nicolson transport operator:
#: ``(B, LU[A], active_indices, N_1**p, n_substeps)`` for repeated
#: ``A u^{n+1} = B u^n`` over one caller step.
_EnergyOp = tuple[Any, Any, np.ndarray, np.ndarray, int]

#: Cached local collision data:
#: ``(K_s*N_p, K_r*N_emit, K_r*N_abs, cell_weights, active_mask)``.
_CollisionOp = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _represented_bcs_weights(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
) -> np.ndarray:
    """Exact BCS capacity on only the represented energy domain."""
    edges = cell_edges_from_widths(E, dE)
    return bcs_dos_cell_weights(
        E,
        dE,
        gap,
        lower_bound=max(float(gap), float(edges[0])),
    )


def _bcs_support_fraction(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
) -> np.ndarray:
    """Fraction of each energy cell lying in the ideal-BCS continuum."""
    edges = cell_edges_from_widths(E, dE)
    overlap = np.maximum(edges[1:] - np.maximum(edges[:-1], float(gap)), 0.0)
    return overlap / dE


def _bcs_anomalous_cell_density(
    E: np.ndarray,
    dE: np.ndarray,
    gap: float,
) -> np.ndarray:
    """Analytic cell average of ``Delta/sqrt(E**2-Delta**2)``."""
    if gap == 0.0:
        return np.zeros_like(E)
    edges = cell_edges_from_widths(E, dE)
    lo = np.maximum(edges[:-1], float(gap))
    hi = np.maximum(edges[1:], lo)
    integral = float(gap) * (
        np.arccosh(np.maximum(hi / float(gap), 1.0))
        - np.arccosh(np.maximum(lo / float(gap), 1.0))
    )
    return integral / dE


def _kl_interface_cell_average(
    E: np.ndarray,
    dE: np.ndarray,
    gap_left: float,
    gap_right: float,
) -> np.ndarray:
    r"""Cell-average KL energy weight at a two-gap interface.

    Integrates ``N1_L*N1_R - N2_L*N2_R`` over the represented part of each
    energy cell.  A xi substitution for the larger gap removes the integrable
    endpoint singularity.  Matched gaps are handled analytically, preserving
    the exact cancellation to the above-gap support fraction.
    """
    edges = cell_edges_from_widths(E, dE)
    high = max(float(gap_left), float(gap_right))
    low = min(float(gap_left), float(gap_right))
    lo = np.maximum(edges[:-1], high)
    hi = np.maximum(edges[1:], lo)
    result = np.zeros_like(E)

    scale = max(1.0, high, low)
    if abs(high - low) <= 64.0 * np.finfo(float).eps * scale:
        result = (hi - lo) / dE
        return result
    if high == 0.0:
        result = (hi - lo) / dE
        return result

    delta_sq = high * high - low * low

    def transformed(xi: float) -> float:
        energy = np.sqrt(high * high + xi * xi)
        return (
            energy * energy - high * low
        ) / (energy * np.sqrt(xi * xi + delta_sq))

    active = np.flatnonzero(hi > lo)
    for i in active:
        xi_lo = np.sqrt(max((lo[i] - high) * (lo[i] + high), 0.0))
        xi_hi = np.sqrt(max((hi[i] - high) * (hi[i] + high), 0.0))
        integral, _error = quad(
            transformed,
            float(xi_lo),
            float(xi_hi),
            epsabs=1e-13 * max(1.0, float(dE[i])),
            epsrel=1e-12,
            limit=100,
        )
        result[i] = integral / dE[i]
    return result


@dataclass(frozen=True)
class T3SpatialFlux1D:
    """External source/sink for ``f(E, x)``.

    ``gain`` and ``loss_rate`` have units ``1/ns`` and shape ``(NE, NX)``.
    Extraction should be encoded as ``loss_rate * f`` rather than negative
    gain, matching :class:`qpsim.devices.external_flux.ExternalFlux`.
    """

    gain: np.ndarray
    loss_rate: np.ndarray
    diagnostics: dict[str, str | float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if np.iscomplexobj(self.gain):
            raise ValueError("gain must be real-valued.")
        if np.iscomplexobj(self.loss_rate):
            raise ValueError("loss_rate must be real-valued.")
        gain = np.asarray(self.gain, dtype=float)
        loss_rate = np.asarray(self.loss_rate, dtype=float)
        if gain.ndim != 2:
            raise ValueError(f"gain must have shape (NE, NX); got {gain.shape}.")
        if loss_rate.shape != gain.shape:
            raise ValueError(
                f"gain/loss_rate shapes must match; got {gain.shape} and "
                f"{loss_rate.shape}."
            )
        if not np.all(np.isfinite(gain)):
            raise ValueError("gain contains non-finite entries.")
        if not np.all(np.isfinite(loss_rate)):
            raise ValueError("loss_rate contains non-finite entries.")
        if np.any(gain < 0.0):
            raise ValueError("gain must be non-negative everywhere.")
        if np.any(loss_rate < 0.0):
            raise ValueError("loss_rate must be non-negative everywhere.")

        gain = gain.copy()
        loss_rate = loss_rate.copy()
        gain.flags.writeable = False
        loss_rate.flags.writeable = False
        object.__setattr__(self, "gain", gain)
        object.__setattr__(self, "loss_rate", loss_rate)

    @classmethod
    def zero(cls, NE: int, NX: int) -> T3SpatialFlux1D:
        """Return a zero flux for an ``(NE, NX)`` spatial state."""
        return cls(gain=np.zeros((NE, NX)), loss_rate=np.zeros((NE, NX)))

    def validate_for_shape(self, shape: tuple[int, int]) -> None:
        """Reject flux arrays not matching ``f.shape``."""
        if self.gain.shape != shape:
            raise ValueError(
                f"Spatial flux is shaped {self.gain.shape}, but f has shape "
                f"{shape}."
            )

    def validate_gain_support(self, supported: np.ndarray) -> None:
        """Reject gain on zero-capacity storage rows."""
        support = np.asarray(supported, dtype=bool)
        if support.shape != self.gain.shape:
            raise ValueError(
                "Spatial flux support mask must have shape "
                f"{self.gain.shape}; got {support.shape}."
            )
        unsupported_gain = (~support) & (self.gain > 0.0)
        if np.any(unsupported_gain):
            locations = np.argwhere(unsupported_gain)
            preview = ", ".join(
                f"({int(i)}, {int(j)})" for i, j in locations[:5]
            )
            suffix = "" if locations.shape[0] <= 5 else ", ..."
            raise ValueError(
                "Spatial flux gain is positive on "
                f"{locations.shape[0]} zero-spectral-capacity (energy, cell) "
                f"location(s) ({preview}{suffix}). Injection into "
                "an unrepresented state is undefined; move the source onto "
                "positive-capacity bins."
            )


@dataclass
class T3Spatial1DState:
    """T3 occupation on an ``(energy, position)`` mesh.

    ``x`` contains the centers of equal-width finite-volume cells.  Reflective
    boundaries lie half a mesh spacing beyond the first and last centers, so
    the physical spatial measure is ``dx * sum_x`` rather than a trapezoidal
    rule that assigns half volume to the end cells.  This convention is what
    makes the conservative transport operator's unweighted cell sum a
    physical volume integral.

    ``diffusion_model`` selects the spatial-diffusion operator (see
    :class:`qpsim.transport.diffusion.base.DiffusionModel`); it defaults
    to the physically correct dirty-limit Usadel reduction ``A1``.

    ``gap_profile`` (optional, shape ``(NX,)``) gives a spatially-varying
    gap so the DOS ``N_1(E, x)`` -- and hence the transport dressing -- is
    evaluated per cell; ``None`` means a uniform scalar gap. With a
    ``gap_profile`` and a finite ``interface_conductance``, the
    profile must contain exactly one piecewise-constant step. That face
    becomes a Kupriyanov-Lukichev interface
    carrying the energy-channel current
    ``F = g_N (N_1^L N_1^R - N_2^L N_2^R) (f_L - f_R)`` -- the
    coherence-factor (Maki-Griffin) weight, regular at matched gaps --
    instead of a bulk diffusive flux. ``interface_conductance`` is that
    ``g_N``: the interface conductance in diffusion units, i.e. an interface
    velocity in microns per nanosecond, dimensionally ``D_0/length``. It is
    not the paper's conductance density ``G_N`` (S/m^2); the two are bridged
    by ``g_N = G_N / (2 e^2 N_0)`` (paper eq:KL_normalization_bridge), and
    the benchmark value is ``g_N = 0.1 um/ns``.

    Local collision kernels are built from
    the same per-cell gap profile, so transport and collisions share one
    spectral support. Each exact gap requires three dense energy matrices. A
    two-entry LRU serves repeated gaps; profiles with more than two distinct
    values are streamed one gap-group at a time so resident kernel memory stays
    bounded. Smooth profiles are therefore exact but can be compute-intensive.
    """

    f: np.ndarray
    x: np.ndarray
    gap: float
    spectral: SpectralContext
    material: Material
    T_bath: float
    diffusion_model: DiffusionModel = DEFAULT_DIFFUSION_MODEL
    gap_profile: np.ndarray | None = None
    interface_conductance: float | None = None  # g_N (um/ns), see docstring

    @property
    def dx(self) -> float:
        """Uniform finite-volume cell width in microns."""
        if self.x.size < 2:
            return 1.0
        return float(np.mean(np.diff(self.x)))

    @property
    def cell_widths(self) -> np.ndarray:
        """Equal control-volume widths associated with :attr:`x`.

        A one-cell state has no coordinate spacing from which to infer its
        physical width, so it retains the backend's historical unit-width
        convention.  Physical campaign drivers with a known domain should
        use at least two cells and verify that these widths sum to that
        domain length.
        """
        return np.full(self.x.size, self.dx, dtype=float)


@dataclass(frozen=True)
class SpatialSnapshot:
    """Compact transient checkpoint for a spatial run."""

    t: float
    max_rate: float
    observables: dict[str, float]


@dataclass(frozen=True)
class SpatialTransientResult:
    """Output of :meth:`T3Spatial1DBackend.run_until_steady_state`."""

    state: T3Spatial1DState
    snapshots: list[SpatialSnapshot]
    total_time: float
    n_steps: int
    converged: bool


class T3Spatial1DBackend:
    """Spatial diffusion + local collision time stepper for 1D Al strips."""

    def __init__(self) -> None:
        self._transport_cn_cache: dict[
            tuple[object, ...],
            list[_EnergyOp | None],
        ] = {}
        self._collision_cache: OrderedDict[
            tuple[object, ...],
            _CollisionOp,
        ] = OrderedDict()

    @staticmethod
    def _spectral_cache_key(
        spectral: SpectralContext,
    ) -> tuple[bytes, bytes, float, float]:
        """Value fingerprint of the spectral content entering cached
        operators/kernels. Identity (``id``) is not safe — a backend can
        outlive a state's SpectralContext, and same-shaped grids with
        different energies, gaps, or Dynes broadening must not share
        cache entries."""
        return (
            spectral.E.tobytes(),
            spectral.dE.tobytes(),
            float(spectral.gap),
            float(spectral.dynes_gamma),
        )

    @staticmethod
    def _spectral_cache_key_for_gap(
        spectral: SpectralContext,
        gap: float,
    ) -> tuple[bytes, bytes, float, float]:
        """Collision fingerprint for ``gap`` without constructing a context."""
        return (
            spectral.E.tobytes(),
            spectral.dE.tobytes(),
            float(gap),
            float(spectral.dynes_gamma),
        )

    def apply_transport(self, state: T3Spatial1DState, dt: float) -> T3Spatial1DState:
        """Conservative finite-volume diffusion step (Crank-Nicolson).

        Advances each energy under the selected
        :class:`~qpsim.transport.diffusion.base.DiffusionModel`,
        ``d_t (N_1**p f) = d_x( D_0 N_1**q  d_x f )``, on the conserved
        density ``u = N_1**p f`` with harmonic-mean face weights
        ``W = D_0 N_1**q`` and reflective (zero-flux) ends -- so
        ``dx * sum_x N_1**p f`` is conserved per energy to round-off. Recovers
        ``f = u / N_1**p`` and clips to ``[0, 1]``. Each energy channel is
        conservatively subcycled until every CN modal amplification is
        non-negative. This removes stiff-CN ringing / pulse swapping at a
        large caller ``dt`` while retaining second-order CN accuracy. The
        final clip is therefore normally a round-off no-op; any material
        clip remains a fail-loud numerical diagnostic.

        With ``(p, q) = (0, -1)`` (model ``C``) at a uniform gap this is the
        same PDE as the legacy ``D_E = D_0 sqrt(1 - (Delta/E)**2)`` step.
        """
        self._validate_state(state)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")

        _NE, NX = state.f.shape
        if NX == 1:
            return state

        ops = self._build_transport_operators(state, dt)
        f_new = state.f.copy()
        signed_clip_change = 0.0
        absolute_clip_change = 0.0
        mass_scale = 0.0
        for i, op in enumerate(ops):
            if op is None:
                continue
            b_mat, lu, idx, rho_p, n_substeps = op
            u = rho_p * f_new[i, idx]
            mass_scale += float(np.sum(np.abs(u)))
            for _ in range(n_substeps):
                u_next = lu.solve(b_mat @ u)
                f_clipped = np.clip(u_next / rho_p, 0.0, 1.0)
                # Track density the [0, 1] clip removed/added: exact
                # conservation holds only while CN stays in the representable
                # box. The stiffness bound makes a material clip unexpected,
                # so retain the warning/failure backstop.
                clip_change = u_next - rho_p * f_clipped
                signed_clip_change += float(np.sum(clip_change))
                absolute_clip_change += float(np.sum(np.abs(clip_change)))
                u = rho_p * f_clipped
            f_new[i, idx] = u / rho_p

        if mass_scale > 0.0:
            absolute_clip_fraction = absolute_clip_change / mass_scale
            signed_clip_fraction = signed_clip_change / mass_scale
            clip_message = (
                "apply_transport: the [0, 1] occupation clip changed an "
                "absolute conserved density Σ N₁^p f of "
                f"{absolute_clip_fraction:.2%} this step "
                f"(signed net {signed_clip_fraction:+.2%}). This indicates a "
                "Crank–Nicolson over/undershoot despite monotonicity "
                "subcycling. Inspect the transport coefficients, reduce dt, "
                "or resolve the front to keep the step conservative."
            )
            if absolute_clip_fraction > 1e-3:
                # An O(1) clipped step can otherwise continue through the
                # outer driver and even be reported as converged. Once more
                # than 0.1% of the represented mass is altered, the step is
                # not a defensible approximation: fail and force dt/grid
                # refinement rather than returning corrupted physics.
                raise RuntimeError(clip_message)
            if absolute_clip_fraction > 1e-9:
                warnings.warn(clip_message, stacklevel=2)

        return replace(state, f=f_new)

    def _transport_rate(
        self,
        state: T3Spatial1DState,
        dt: float,
    ) -> np.ndarray:
        """Return the endpoint semidiscrete diffusion ``df/dt``.

        Recover the generator from the same cached Crank--Nicolson
        ``B = I + h L / 2`` matrices used by :meth:`apply_transport`. This
        avoids maintaining a second transport discretization solely for the
        steady-state certificate.
        """
        self._validate_state(state)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")

        rate = np.zeros_like(state.f)
        if state.f.shape[1] == 1:
            return rate

        for i, op in enumerate(self._build_transport_operators(state, dt)):
            if op is None:
                continue
            b_mat, _lu, idx, rho_p, n_substeps = op
            sub_dt = dt / n_substeps
            u = rho_p * state.f[i, idx]
            du_dt = (2.0 / sub_dt) * (b_mat @ u - u)
            rate[i, idx] = du_dt / rho_p
        return rate

    def _build_transport_operators(
        self, state: T3Spatial1DState, dt: float
    ) -> list[_EnergyOp | None]:
        """Per-energy Crank-Nicolson transport operators (cached).

        One entry per energy bin:
        ``(B, LU[A], active_indices, N_1**p, n_substeps)`` for the
        conserved-density update ``A u^{n+1} = B u^n``, or ``None`` where
        the bin has fewer than two states above the gap (nothing to diffuse).
        ``n_substeps`` is the smallest integer that keeps the CN step below
        the non-negative-amplification stiffness bound.
        """
        NE, _NX = state.f.shape
        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Spatial transport requires a pure-BCS SpectralContext "
                "(dynes_gamma == 0). The transport dressings implement the "
                "clean-BCS traces — D_L as the indicator of N_1 > 0 and the "
                "Kupriyanov-Lukichev weight N_1 N_1' - N_2 N_2' from the "
                "real spectral functions — which rely on the above-gap "
                "identity N_1**2 - N_2**2 = 1. With a finite Dynes Gamma "
                "that identity fails and the coefficients must be "
                "re-evaluated from the complex spectral functions (paper, "
                "Dynes footnote below eq:bcs_dos); silently combining the "
                "broadened DOS with the clean-BCS dressings is wrong "
                "(e.g. full-strength sub-gap transport wherever the "
                "broadened N_1 > 0)."
            )
        model = state.diffusion_model
        p, q = model.p, model.q
        D0 = float(state.spectral.diffusion_coefficient)
        dx = state.dx
        inv_dx2 = 1.0 / (dx * dx)
        # The lookup precedes every per-cell spectral evaluation: the key
        # already fingerprints everything the arrays below derive from, so
        # building them first would discard the whole cost on each hit.
        key = (
            _NX,
            float(dx),
            float(dt),
            model,
            D0,
            self._gap_cache_key(state),
            self._spectral_cache_key(state.spectral),
        )
        cached = self._transport_cn_cache.get(key)
        if cached is not None:
            return cached

        N1 = self._n1_per_cell(state)
        support_fraction = self._support_fraction_per_cell(state)
        interface_faces = self._interface_faces(state)
        g_N = state.interface_conductance
        g_interface = (
            float(g_N) if (interface_faces and g_N is not None) else 0.0
        )
        interface_weights: dict[int, np.ndarray] = {}
        if interface_faces:
            if state.gap_profile is None:  # pragma: no cover - validated state
                raise RuntimeError("Interface faces require a gap profile.")
            for face in interface_faces:
                interface_weights[face] = _kl_interface_cell_average(
                    state.spectral.E,
                    state.spectral.dE,
                    float(state.gap_profile[face]),
                    float(state.gap_profile[face + 1]),
                )

        ops: list[_EnergyOp | None] = []
        for i in range(NE):
            N1_i = N1[i]
            active = N1_i > 0.0
            na = int(np.count_nonzero(active))
            if na < 2:
                ops.append(None)
                continue
            idx = np.flatnonzero(active)
            if int(idx[-1] - idx[0]) + 1 != na:
                raise NotImplementedError(
                    "Non-contiguous active spatial region is not supported "
                    "by the 1D transport operator."
                )
            N1_a = N1_i[idx]
            rho_p = density_weight(N1_a, p)
            w_cell = flux_weight(D0, N1_a, q)
            if q == 0:
                # Dirty-limit D_L is an above-gap indicator.  Its exact
                # finite-volume average is the supported fraction of a cut
                # cell, not one merely because the cell has some capacity.
                w_cell = D0 * support_fraction[i, idx]
            # Face weights are the harmonic mean of those cell averages.
            # That is exact wherever the two neighbours share a gap.  On a
            # face whose neighbours have different gaps the exact energy-cell
            # average of the face coefficient is min(s_L, s_R) instead, so the
            # one bin cut by the larger gap is over-weighted (up to 2x in that
            # bin); the bias is first order in dx and vanishes under energy
            # refinement.  Documented limitation, not yet corrected.
            g_face = _harmonic_face_weights(w_cell) * inv_dx2
            if interface_faces:
                for m in range(na - 1):
                    if int(idx[m]) in interface_faces:
                        # Kupriyanov-Lukichev finite interface conductance,
                        # energy channel:
                        # F = g_N (N_1^L N_1^R - N_2^L N_2^R)(f_L - f_R),
                        # dx-independent (the 1/dx is the flux-divergence
                        # factor). The coherence-factor weight equals
                        # (E^2 - D_L D_R)/(Omega_L Omega_R) > 0 above both
                        # gaps and is regular at matched gaps; with
                        # N_1 = N_2 = 0 on one side (sub-gap there) the
                        # interface is automatically closed.
                        weight = interface_weights[int(idx[m])][i]
                        g_face[m] = g_interface * weight / dx
            laplacian = _flux_laplacian_from_conductances(g_face, na)
            operator = (laplacian @ sparse.diags(1.0 / rho_p)).tocsr()
            exit_rate = float(np.max(-operator.diagonal()))
            stiffness = dt * exit_rate
            if not np.isfinite(stiffness) or stiffness < 0.0:
                raise RuntimeError(
                    "Spatial diffusion produced a non-finite or negative "
                    "stiffness estimate; inspect the transport coefficients."
                )
            n_substeps = max(1, int(np.ceil(stiffness / _MAX_CN_EXIT_STEP)))
            if n_substeps > _MAX_CN_SUBSTEPS:
                raise RuntimeError(
                    "Crank--Nicolson monotonicity would require "
                    f"{n_substeps:,} internal substeps for one energy "
                    f"channel (dt*max_exit_rate={stiffness:g}). Reduce dt "
                    "or coarsen the spatial mesh."
                )
            sub_dt = dt / n_substeps
            b_mat, lu = build_cn_operators(operator, sub_dt, 1.0)
            ops.append((b_mat, lu, idx, rho_p, n_substeps))

        self._transport_cn_cache[key] = ops
        return ops

    def _n1_per_cell(self, state: T3Spatial1DState) -> np.ndarray:
        """Finite-volume BCS density ``w_i/dE_i``, shape ``(NE, NX)``.

        Uniform-gap path (no ``gap_profile``): ``N_1`` is x-independent,
        the spectral context's cell-average DOS broadcast across the mesh.
        With a ``gap_profile`` the exact singular capacity is integrated for
        each local gap.  This makes the default A1 conserved density identical
        to the collision/remap/observable finite-volume capacity.  Cells are
        grouped by exact gap first (as :meth:`_collision_layout` does), so a
        piecewise-constant profile costs one integration per distinct gap
        rather than one per cell.
        """
        _NE, NX = state.f.shape
        if state.gap_profile is None:
            return np.repeat(state.spectral.cell_density[:, None], NX, axis=1)
        E = state.spectral.E
        dE = state.spectral.dE
        distinct_gaps, group_index = np.unique(
            state.gap_profile, return_inverse=True
        )
        columns = [
            _represented_bcs_weights(E, dE, float(g)) / dE
            for g in distinct_gaps
        ]
        return np.column_stack(columns)[:, group_index]

    def _support_fraction_per_cell(self, state: T3Spatial1DState) -> np.ndarray:
        """Above-gap geometric fraction for every energy/spatial cell.

        Grouped by exact gap like :meth:`_n1_per_cell`.
        """
        _NE, NX = state.f.shape
        E = state.spectral.E
        dE = state.spectral.dE
        if state.gap_profile is None:
            fraction = _bcs_support_fraction(E, dE, state.spectral.gap)
            return np.repeat(fraction[:, None], NX, axis=1)
        distinct_gaps, group_index = np.unique(
            state.gap_profile, return_inverse=True
        )
        columns = [
            _bcs_support_fraction(E, dE, float(g))
            for g in distinct_gaps
        ]
        return np.column_stack(columns)[:, group_index]

    def _n2_per_cell(self, state: T3Spatial1DState) -> np.ndarray:
        """BCS anomalous weight ``N_2(E_i, x_j)``, shape ``(NE, NX)``.

        Companion to :meth:`_n1_per_cell`; used by the coherence-factor
        weight of the Kupriyanov-Lukichev interface faces.
        """
        _NE, NX = state.f.shape
        E = state.spectral.E
        dE = state.spectral.dE
        if state.gap_profile is None:
            n2 = state.spectral.cell_anomalous_density
            return np.repeat(n2[:, None], NX, axis=1)
        columns = [
            _bcs_anomalous_cell_density(E, dE, float(g))
            for g in state.gap_profile
        ]
        return np.column_stack(columns)

    @staticmethod
    def _interface_faces(state: T3Spatial1DState) -> set[int]:
        """Full-grid face indices carrying a Kupriyanov-Lukichev interface.

        A face ``k`` (between cells ``k`` and ``k+1``) is an interface when a
        ``gap_profile`` is present, a finite ``interface_conductance`` is
        set, and the gap steps across that face.
        """
        if state.gap_profile is None or state.interface_conductance is None:
            return set()
        gap = state.gap_profile
        return {int(k) for k in np.flatnonzero(gap[:-1] != gap[1:])}

    @staticmethod
    def _gap_cache_key(state: T3Spatial1DState) -> object:
        """Transport-operator cache discriminator for the gap profile."""
        if state.gap_profile is None:
            return float(state.spectral.gap)
        return (state.gap_profile.tobytes(), state.interface_conductance)

    def apply_collisions(
        self,
        state: T3Spatial1DState,
        dt: float,
        *,
        external_flux: T3SpatialFlux1D | None = None,
    ) -> T3Spatial1DState:
        """One local ETD2 collision/source step at every spatial cell.

        With ``gap_profile`` present, columns are grouped by local gap and
        each group uses a matching spectral context and e-ph kernel. This
        keeps the collision support consistent with transport and prevents a
        high-gap region from inheriting low-gap states and rates. One or two
        gap groups share a batched ETD2 call. Larger profiles are streamed one
        group at a time, retaining exact local operators while bounding dense
        kernel memory independently of ``NX``.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")
        distinct_gaps, group_index = self._collision_layout(state, external_flux)
        if distinct_gaps.size > _MAX_BATCHED_COLLISION_GAPS:
            return self._apply_collisions_streamed(
                state,
                dt,
                external_flux,
                distinct_gaps,
                group_index,
            )
        rhs, balance_weights = self._collision_system(state, external_flux)

        return replace(
            state,
            f=etd2_step(
                state.f,
                rhs,
                dt,
                balance_weights=balance_weights,
                balance_axis=0,
                max_loss_step=0.25,
            ),
        )

    def _collision_layout(
        self,
        state: T3Spatial1DState,
        external_flux: T3SpatialFlux1D | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate collision inputs and group spatial cells by exact gap."""
        self._validate_state(state)
        if external_flux is not None:
            external_flux.validate_for_shape(state.f.shape)

        if state.spectral.dynes_gamma > 0.0:
            raise ValueError(
                "Spatial collisions require a pure-BCS SpectralContext "
                "(dynes_gamma == 0). A Dynes DOS needs consistently broadened "
                "normal/anomalous coherence functions, which this collision "
                "kernel does not implement."
            )

        _NE, NX = state.f.shape
        gap_by_cell = (
            np.full(NX, float(state.spectral.gap))
            if state.gap_profile is None
            else np.asarray(state.gap_profile, dtype=float)
        )
        return np.unique(gap_by_cell, return_inverse=True)

    @staticmethod
    def _collision_group_rates(
        f_group: np.ndarray,
        operator: _CollisionOp,
        external_gain: np.ndarray | None = None,
        external_loss: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate one exact local-gap group without retaining other kernels."""
        K_s_eff, K_r_emit, K_r_abs, weights, physical = operator
        one_minus = np.maximum(1.0 - f_group, 0.0)

        gain = one_minus * (K_s_eff.T @ (weights[:, None] * f_group))
        loss = K_s_eff @ (weights[:, None] * one_minus)

        # Kaplan Eq. (8) per-QP normalization -- see
        # qpsim.collisions.phonon.phonon_collision_rates.
        loss += K_r_emit @ (weights[:, None] * f_group)
        gain += one_minus * (K_r_abs @ (weights[:, None] * one_minus))

        # A zero-capacity bin carries no represented state. A cell cut by the
        # gap remains physical through its exact finite-volume weight.
        gain[~physical, :] = 0.0
        loss[~physical, :] = 0.0
        if external_gain is not None:
            if external_loss is None:  # pragma: no cover - private invariant
                raise RuntimeError("external gain/loss must be supplied together.")
            gain[physical, :] += external_gain[physical, :]
            loss[physical, :] += external_loss[physical, :]
        return gain, loss

    @staticmethod
    def _validate_group_gain_support(
        physical: np.ndarray,
        columns: np.ndarray,
        external_flux: T3SpatialFlux1D | None,
    ) -> None:
        """Fail before advancing a group with gain on zero-capacity rows."""
        if external_flux is None:
            return
        unsupported = (~physical[:, None]) & (external_flux.gain[:, columns] > 0.0)
        if not np.any(unsupported):
            return
        local_locations = np.argwhere(unsupported)
        locations = np.column_stack(
            (local_locations[:, 0], columns[local_locations[:, 1]])
        )
        preview = ", ".join(
            f"({int(i)}, {int(j)})" for i, j in locations[:5]
        )
        suffix = "" if locations.shape[0] <= 5 else ", ..."
        raise ValueError(
            "Spatial flux gain is positive on "
            f"{locations.shape[0]} zero-spectral-capacity (energy, cell) "
            f"location(s) ({preview}{suffix}). Injection into an "
            "unrepresented state is undefined; move the source onto "
            "positive-capacity bins."
        )

    def _apply_collisions_streamed(
        self,
        state: T3Spatial1DState,
        dt: float,
        external_flux: T3SpatialFlux1D | None,
        distinct_gaps: np.ndarray,
        group_index: np.ndarray,
    ) -> T3Spatial1DState:
        """Advance 3+ exact gap groups with bounded resident kernel memory."""
        updated = state.f.copy()
        for group in self._streaming_group_order(state, distinct_gaps):
            local_gap = float(distinct_gaps[group])
            columns = np.flatnonzero(group_index == group)
            operator = self._local_collision_operator(state, local_gap)
            physical = operator[4]
            self._validate_group_gain_support(physical, columns, external_flux)
            external_gain = (
                None if external_flux is None else external_flux.gain[:, columns]
            )
            external_loss = (
                None if external_flux is None else external_flux.loss_rate[:, columns]
            )

            def rhs(
                f_group: np.ndarray,
                operator_bound: _CollisionOp = operator,
                external_gain_bound: np.ndarray | None = external_gain,
                external_loss_bound: np.ndarray | None = external_loss,
            ) -> tuple[np.ndarray, np.ndarray]:
                return self._collision_group_rates(
                    f_group,
                    operator_bound,
                    external_gain_bound,
                    external_loss_bound,
                )

            f_group = state.f[:, columns]
            balance_weights = np.broadcast_to(
                operator[3][:, None],
                f_group.shape,
            )
            updated[:, columns] = etd2_step(
                f_group,
                rhs,
                dt,
                balance_weights=balance_weights,
                balance_axis=0,
                max_loss_step=0.25,
            )
            # Do not let loop locals retain an evicted dense operator while the
            # next gap is built. The LRU remains the sole cross-group owner.
            del rhs, operator
        return replace(state, f=updated)

    def _streaming_group_order(
        self,
        state: T3Spatial1DState,
        distinct_gaps: np.ndarray,
    ) -> list[int]:
        """Visit resident gaps first so a repeated smooth profile gets LRU hits.

        A fixed cyclic order gives a cache smaller than the working set zero
        hits: the first miss evicts a gap needed later in that same traversal.
        Visiting the two resident operators first saves those builds; the
        remaining exact groups are still streamed with the same bounded memory.
        """
        resident: list[int] = []
        missing: list[int] = []
        for group, local_gap in enumerate(distinct_gaps):
            key = self._local_collision_cache_key(state, float(local_gap))
            (resident if key in self._collision_cache else missing).append(group)
        return resident + missing

    def _collision_system(
        self,
        state: T3Spatial1DState,
        external_flux: T3SpatialFlux1D | None,
    ) -> tuple[
        Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
        np.ndarray,
    ]:
        """Build the gain/loss operator shared by stepping and diagnostics."""
        distinct_gaps, group_index = self._collision_layout(state, external_flux)
        if distinct_gaps.size > _MAX_BATCHED_COLLISION_GAPS:
            raise RuntimeError(
                "Internal batched collision assembly received more than two "
                "gap groups; use the streamed collision path."
            )

        groups: list[tuple[np.ndarray, _CollisionOp]] = []
        balance_weights = np.zeros_like(state.f)
        collision_support = np.zeros_like(state.f, dtype=bool)
        for group in self._streaming_group_order(state, distinct_gaps):
            local_gap = float(distinct_gaps[group])
            columns = np.flatnonzero(group_index == group)
            operator = self._local_collision_operator(state, local_gap)
            groups.append((columns, operator))
            balance_weights[:, columns] = operator[3][:, None]
            collision_support[:, columns] = operator[4][:, None]

        if external_flux is not None:
            external_flux.validate_gain_support(collision_support)

        def rhs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            gain = np.zeros_like(f)
            loss = np.zeros_like(f)
            for columns, operator in groups:
                f_group = f[:, columns]
                gain_group, loss_group = self._collision_group_rates(
                    f_group,
                    operator,
                    None if external_flux is None else external_flux.gain[:, columns],
                    (
                        None
                        if external_flux is None
                        else external_flux.loss_rate[:, columns]
                    ),
                )

                gain[:, columns] = gain_group
                loss[:, columns] = loss_group
            return gain, loss

        return rhs, balance_weights

    def _collision_rate(
        self,
        state: T3Spatial1DState,
        external_flux: T3SpatialFlux1D | None,
    ) -> np.ndarray:
        """Return the unclipped endpoint collision/source ``df/dt``."""
        distinct_gaps, group_index = self._collision_layout(state, external_flux)
        if distinct_gaps.size <= _MAX_BATCHED_COLLISION_GAPS:
            rhs, _balance_weights = self._collision_system(state, external_flux)
            gain, loss = rhs(state.f)
            return gain - loss * state.f

        rate = np.zeros_like(state.f)
        for group in self._streaming_group_order(state, distinct_gaps):
            columns = np.flatnonzero(group_index == group)
            operator = self._local_collision_operator(
                state,
                float(distinct_gaps[group]),
            )
            physical = operator[4]
            self._validate_group_gain_support(physical, columns, external_flux)
            f_group = state.f[:, columns]
            gain, loss = self._collision_group_rates(
                f_group,
                operator,
                None if external_flux is None else external_flux.gain[:, columns],
                (
                    None
                    if external_flux is None
                    else external_flux.loss_rate[:, columns]
                ),
            )
            rate[:, columns] = gain - loss * f_group
            del operator
        return rate

    def _local_collision_operator(
        self,
        state: T3Spatial1DState,
        local_gap: float,
    ) -> _CollisionOp:
        """Return bounded-LRU thermal e-ph matrices for one local gap.

        The value-only key is formed before a local :class:`SpectralContext`
        is constructed. Repeated calls therefore reuse both the dense kernels
        and the context-derived coherence matrices instead of eagerly paying
        that allocation even on a cache hit.
        """
        cache_key = self._local_collision_cache_key(state, local_gap)
        cached = self._collision_cache.get(cache_key)
        if cached is not None:
            self._collision_cache.move_to_end(cache_key)
            return cached

        # Evict before allocating a miss. Each entry already owns three dense
        # matrices; post-build eviction would temporarily retain the full
        # cache plus the local context, two base kernels, and three effective
        # matrices for the new gap.
        while len(self._collision_cache) >= _MAX_COLLISION_CACHE_ENTRIES:
            self._collision_cache.popitem(last=False)

        spectral = self._local_spectral_context(state.spectral, local_gap)

        K_s0 = build_scattering_kernel_base(
            spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        K_r0 = build_recombination_kernel_base(
            spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        N_p = _thermal_phonon_scattering_occupation(spectral.E, state.T_bath)
        N_emit, N_abs = _thermal_phonon_recombination_occupations(
            spectral.E,
            state.T_bath,
        )
        cached = (
            K_s0 * N_p,
            K_r0 * N_emit,
            K_r0 * N_abs,
            np.asarray(spectral.cell_weights),
            np.asarray(spectral.active_mask),
        )
        self._collision_cache[cache_key] = cached
        return cached

    def _local_collision_cache_key(
        self,
        state: T3Spatial1DState,
        local_gap: float,
    ) -> tuple[object, ...]:
        """Return the exact value key for one local collision operator."""
        return (
            self._spectral_cache_key_for_gap(state.spectral, local_gap),
            float(state.material.tau_0),
            float(state.material.T_c),
            float(state.T_bath),
        )

    @staticmethod
    def _local_spectral_context(
        base: SpectralContext,
        local_gap: float,
    ) -> SpectralContext:
        """Build a local-gap context only after a collision-cache miss."""
        if float(local_gap) == float(base.gap):
            return base
        return SpectralContext(
            E_bins=base.E,
            dE_bins=base.dE,
            gap=float(local_gap),
            diffusion_coefficient=base.diffusion_coefficient,
            rebuild_tolerance=base.rebuild_tolerance,
            active_margin_factor=base.active_margin_factor,
        )

    def step(
        self,
        state: T3Spatial1DState,
        dt: float,
        *,
        external_flux: T3SpatialFlux1D | None = None,
    ) -> T3Spatial1DState:
        """Symmetric split step: diffusion/2, collisions+source, diffusion/2."""
        s = self.apply_transport(state, 0.5 * dt)
        s = self.apply_collisions(s, dt, external_flux=external_flux)
        return self.apply_transport(s, 0.5 * dt)

    def run_until_steady_state(
        self,
        state: T3Spatial1DState,
        *,
        dt: float,
        max_time: float,
        external_flux: T3SpatialFlux1D | None = None,
        stop_tol: float = 1e-10,
        snapshot_interval: float | None = None,
        observables: dict[str, Callable[[T3Spatial1DState], float]] | None = None,
        progress_hook: Callable[[float, float], bool] | None = None,
    ) -> SpatialTransientResult:
        """Run fixed-step dynamics until ``max|df/dt| < stop_tol`` or timeout.

        The stopping residual is the raw endpoint sum of the semidiscrete
        diffusion and collision/source operators. It is evaluated before any
        occupation clipping, so a saturated accepted step cannot masquerade
        as a steady state while its governing operator remains nonzero.

        Snapshot boundaries crossed by a step are all emitted. Observable
        values at an interior boundary use linear dense output between the
        split-step endpoints, so ``dt > snapshot_interval`` does not drop
        cadence points or mislabel endpoint states.

        ``progress_hook``, when given, is called after every step with
        ``(t, max_time)``; returning ``False`` stops the run cleanly at
        the current time exactly as if ``max_time`` had been reached
        there (final state still recorded; ``converged`` unaffected).
        Physics-neutral — ``None`` leaves the loop bit-for-bit
        unchanged. Intended for progress reporting and cooperative
        cancellation from interactive callers.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive.")
        if not np.isfinite(max_time) or max_time <= 0.0:
            raise ValueError("max_time must be positive.")
        if not np.isfinite(stop_tol) or stop_tol < 0.0:
            raise ValueError("stop_tol must be non-negative.")
        if snapshot_interval is None:
            snapshot_interval = max_time / 50.0
        if not np.isfinite(snapshot_interval) or snapshot_interval <= 0.0:
            raise ValueError("snapshot_interval must be positive.")

        current = state
        t = 0.0
        n_steps = 0
        next_snapshot = snapshot_interval
        snapshots: list[SpatialSnapshot] = []

        def record(
            snapshot_time: float,
            snapshot_state: T3Spatial1DState,
            max_rate: float,
        ) -> None:
            obs = (
                {
                    name: float(fn(snapshot_state))
                    for name, fn in observables.items()
                }
                if observables
                else {}
            )
            snapshots.append(
                SpatialSnapshot(
                    t=float(snapshot_time),
                    max_rate=float(max_rate),
                    observables=obs,
                )
            )

        record(0.0, current, float("inf"))
        converged = False
        last_max_rate = float("inf")
        max_steps = int(np.ceil(max_time / dt))

        for _ in range(max_steps):
            remaining = max_time - t
            if remaining <= 1e-12:
                break
            step_dt = min(dt, remaining)
            t_previous = t
            old_f = current.f.copy()
            current = self.step(current, step_dt, external_flux=external_flux)
            t += step_dt
            n_steps += 1
            transport_rate = self._transport_rate(current, 0.5 * step_dt)
            collision_rate = self._collision_rate(current, external_flux)
            max_rate = float(np.max(np.abs(transport_rate + collision_rate)))
            last_max_rate = max_rate

            time_tol = 16.0 * np.finfo(float).eps * max(
                1.0, abs(t), abs(next_snapshot),
            )
            while next_snapshot <= t + time_tol:
                # Bound the cadence drain (see the 0-D transient driver): a
                # snapshot_interval far below step_dt would otherwise loop
                # unboundedly. Fail loud for direct callers; the webui schema
                # rejects such setups up front.
                if len(snapshots) >= _SNAPSHOT_HARD_CAP:
                    raise ValueError(
                        f"snapshot_interval={snapshot_interval:g} ns is too small for "
                        f"max_time={max_time:g} ns: the cadence would emit more than "
                        f"{_SNAPSHOT_HARD_CAP} snapshots. Increase snapshot_interval."
                    )
                snapshot_time = min(next_snapshot, t)
                fraction = (snapshot_time - t_previous) / step_dt
                snapshot_f = old_f + fraction * (current.f - old_f)
                record(
                    snapshot_time,
                    replace(current, f=snapshot_f),
                    max_rate,
                )
                next_snapshot += snapshot_interval

            if max_rate < stop_tol:
                converged = True
                if snapshots[-1].t < t - time_tol:
                    record(t, current, max_rate)
                break

            if progress_hook is not None and not progress_hook(t, max_time):
                break

        if snapshots[-1].t < t:
            record(t, current, 0.0 if n_steps == 0 else last_max_rate)

        return SpatialTransientResult(
            state=current,
            snapshots=snapshots,
            total_time=float(t),
            n_steps=n_steps,
            converged=converged,
        )

    @staticmethod
    def _validate_state(state: T3Spatial1DState) -> None:
        f = np.asarray(state.f)
        x = np.asarray(state.x)
        if f.ndim != 2:
            raise ValueError(f"state.f must have shape (NE, NX); got {f.shape}.")
        if x.ndim != 1:
            raise ValueError("state.x must be 1D.")
        if x.size == 0 or np.any(~np.isfinite(x)):
            raise ValueError("state.x must be non-empty and finite.")
        if f.shape[0] != state.spectral.E.size:
            raise ValueError(
                f"state.f has {f.shape[0]} energy bins, but spectral has "
                f"{state.spectral.E.size}."
            )
        if f.shape[1] != x.size:
            raise ValueError(
                f"state.f has {f.shape[1]} spatial cells, but x has {x.size}."
            )
        if x.size > 1:
            diffs = np.diff(x)
            # Uniform *and* strictly increasing: an equal-diffs test alone
            # admits a zero mesh ([0,0,0] -> dx=0 -> divide-by-zero) and a
            # descending mesh ([3,2,1] -> dx<0), both of which corrupt every
            # flux/Laplacian downstream that divides by dx.
            if diffs[0] <= 0.0 or not np.allclose(diffs, diffs[0]):
                raise ValueError(
                    "state.x must be uniformly spaced and strictly increasing."
                )
        if np.any(~np.isfinite(f)):
            raise ValueError("state.f contains non-finite values.")
        if np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("state.f must lie in [0, 1].")
        if not np.isfinite(state.T_bath) or state.T_bath < 0.0:
            raise ValueError("state.T_bath must be finite and non-negative.")
        if not np.isfinite(state.gap) or state.gap <= 0.0:
            raise ValueError("state.gap must be finite and positive.")
        gap_scale = max(abs(float(state.gap)), abs(state.spectral.gap), 1.0)
        if not np.isclose(
            state.gap,
            state.spectral.gap,
            rtol=1e-10,
            atol=1e-10 * gap_scale,
        ):
            raise ValueError(
                "state.gap and state.spectral.gap must match; "
                f"got {state.gap:g} and {state.spectral.gap:g}."
            )
        if state.gap_profile is not None:
            gap_profile = np.asarray(state.gap_profile)
            if gap_profile.shape != (f.shape[1],):
                raise ValueError(
                    f"gap_profile must have shape (NX,)=({f.shape[1]},); "
                    f"got {gap_profile.shape}."
                )
            if np.any(~np.isfinite(gap_profile)) or np.any(gap_profile < 0.0):
                raise ValueError("gap_profile must be finite and non-negative.")
        if state.interface_conductance is not None and not (
            np.isfinite(state.interface_conductance)
            and state.interface_conductance >= 0.0
        ):
            raise ValueError(
                "interface_conductance must be finite and non-negative; "
                f"got {state.interface_conductance}."
            )
        if state.interface_conductance is not None:
            if state.gap_profile is None:
                raise ValueError(
                    "interface_conductance requires a gap_profile with exactly "
                    "one piecewise-constant step; without it the conductance "
                    "would be silently unused."
                )
            step_faces = np.flatnonzero(
                state.gap_profile[:-1] != state.gap_profile[1:]
            )
            if step_faces.size != 1:
                raise ValueError(
                    "interface_conductance requires a piecewise-constant "
                    "gap_profile with exactly one step. A smooth or multi-step "
                    "profile needs an explicit interface-face mask, which this "
                    "backend does not yet expose; otherwise every unequal face "
                    "would be misclassified as a KL interface."
                )


def _harmonic_face_weights(W_cell: np.ndarray) -> np.ndarray:
    """Harmonic-mean interior-face weights ``2 W_j W_{j+1} / (W_j + W_{j+1})``.

    Matches :func:`qpsim.grid.spatial_grid.build_variable_diffusion_laplacian`,
    so the discrete flux stays conservative across jumps in ``W``. A face with
    a zero-``W`` neighbour (no states there) gets zero weight.
    """
    w_left = W_cell[:-1]
    w_right = W_cell[1:]
    lo = np.minimum(w_left, w_right)
    hi = np.maximum(w_left, w_right)
    w_face = np.zeros(W_cell.size - 1, dtype=float)
    nonzero = lo > 0.0
    # Algebraically equal to ``2*a*b/(a+b)``, but neither multiplies two
    # large weights nor adds them before scaling.  Both operations can
    # overflow for perfectly finite diffusivities (e.g. a=b=1e308).
    w_face[nonzero] = lo[nonzero] * (
        2.0 / (1.0 + lo[nonzero] / hi[nonzero])
    )
    return w_face


def _flux_laplacian_from_conductances(
    g_face: np.ndarray, n: int
) -> sparse.csr_matrix:
    """Tridiagonal ``d_x(.)`` operator from per-face conductances.

    Each interior face ``j`` couples cells ``j`` and ``j+1`` as
    ``du_j/dt -= g_face[j] (u_j - u_{j+1})`` (and the antisymmetric term on
    cell ``j+1``). Row and column sums vanish -- zero-flux ends -- so the
    Crank-Nicolson update conserves ``sum_x u`` exactly. Bulk diffusion uses
    ``g_face = W_face / dx**2``; a Kupriyanov-Lukichev interface overrides
    its face with ``g_N (N_1^L N_1^R - N_2^L N_2^R) / dx`` (``g_N`` in
    um/ns, see :class:`T3Spatial1DState`). The override is the bare
    interface conductance, not its series combination with the two adjoining
    half-cells of bulk, so the face is first-order accurate in ``dx`` and
    does not cap at the bulk face as ``g_N`` grows. With constant
    ``g_face`` this is the standard reflective Laplacian.
    """
    off = np.asarray(g_face, dtype=float)
    main = np.zeros(n, dtype=float)
    main[:-1] -= off
    main[1:] -= off
    return sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
