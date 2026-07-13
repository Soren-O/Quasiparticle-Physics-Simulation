"""One-dimensional spatial T3 diffusion backend.

This is a narrow Gate-5 preview for strip-like devices: ``f(E, x)`` on a
uniform 1D mesh, reflective end boundaries, Crank-Nicolson diffusion in
space, and local T3 electron-phonon collisions at fixed gap.  Phonons are
held at the thermal bath in this first spatial path; finite-``tau_l``
spatial phonons can be layered on once the Ph1/Ph2 state is available.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import sparse

from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.materials.database import Material
from qpsim.physics.spectral import (
    SpectralContext,
    bcs_anomalous_weight,
    bcs_density_of_states,
)
from qpsim.solvers.crank_nicolson import build_cn_operators
from qpsim.solvers.etd import etd2_step
from qpsim.transport.diffusion.base import (
    DEFAULT_DIFFUSION_MODEL,
    DiffusionModel,
    density_weight,
    flux_weight,
)

#: One energy's cached Crank-Nicolson transport operator:
#: ``(B, LU[A], active_indices, N_1**p)`` for ``A u^{n+1} = B u^n``.
_EnergyOp = tuple[Any, Any, np.ndarray, np.ndarray]


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


@dataclass
class T3Spatial1DState:
    """T3 occupation on an ``(energy, position)`` mesh.

    ``diffusion_model`` selects the spatial-diffusion operator (see
    :class:`qpsim.transport.diffusion.base.DiffusionModel`); it defaults
    to the physically correct dirty-limit Usadel reduction ``A1``.

    ``gap_profile`` (optional, shape ``(NX,)``) gives a spatially-varying
    gap so the DOS ``N_1(E, x)`` -- and hence the transport dressing -- is
    evaluated per cell; ``None`` means a uniform scalar gap. With a
    ``gap_profile`` and a finite ``interface_conductance`` ``G_N``, every
    face where the gap steps becomes a Kupriyanov-Lukichev interface
    carrying the energy-channel current
    ``F = G_N (N_1^L N_1^R - N_2^L N_2^R) (f_L - f_R)`` -- the
    coherence-factor (Maki-Griffin) weight, regular at matched gaps --
    instead of a bulk diffusive
    flux. Both only affect transport; the collision term still uses the
    scalar-gap ``spectral`` context.
    """

    f: np.ndarray
    x: np.ndarray
    gap: float
    spectral: SpectralContext
    material: Material
    T_bath: float
    diffusion_model: DiffusionModel = DEFAULT_DIFFUSION_MODEL
    gap_profile: np.ndarray | None = None
    interface_conductance: float | None = None

    @property
    def dx(self) -> float:
        """Uniform spatial mesh spacing in microns."""
        if self.x.size < 2:
            return 1.0
        return float(np.mean(np.diff(self.x)))


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
        self._collision_cache: dict[
            tuple[object, ...],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

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

    def apply_transport(self, state: T3Spatial1DState, dt: float) -> T3Spatial1DState:
        """Conservative finite-volume diffusion step (Crank-Nicolson).

        Advances each energy under the selected
        :class:`~qpsim.transport.diffusion.base.DiffusionModel`,
        ``d_t (N_1**p f) = d_x( D_0 N_1**q  d_x f )``, on the conserved
        density ``u = N_1**p f`` with harmonic-mean face weights
        ``W = D_0 N_1**q`` and reflective (zero-flux) ends -- so
        ``sum_x N_1**p f`` is conserved per energy to round-off. Recovers
        ``f = u / N_1**p`` and clips to ``[0, 1]``. Caveat: the clip is a
        no-op for resolved profiles, but if CN over/undershoots (large
        dt against a sharp front) it trims the excursion and the trimmed
        mass is NOT restored — exact conservation holds only while the
        solution stays inside ``[0, 1]``.

        With ``(p, q) = (0, -1)`` (model ``C``) at a uniform gap this is the
        same PDE as the legacy ``D_E = D_0 sqrt(1 - (Delta/E)**2)`` step.
        """
        self._validate_state(state)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        _NE, NX = state.f.shape
        if NX == 1:
            return state

        ops = self._build_transport_operators(state, dt)
        f_new = state.f.copy()
        clip_loss = 0.0
        mass_scale = 0.0
        for i, op in enumerate(ops):
            if op is None:
                continue
            b_mat, lu, idx, rho_p = op
            u = rho_p * f_new[i, idx]
            u_next = lu.solve(b_mat @ u)
            f_clipped = np.clip(u_next / rho_p, 0.0, 1.0)
            # Track density the [0, 1] clip removed/added: exact conservation
            # (Σ_x N₁^p f) holds only while the CN update stays in [0, 1];
            # clipping an over/undershoot on an unresolved front trims mass
            # that is NOT restored.
            clip_loss += float(np.sum(u_next - rho_p * f_clipped))
            mass_scale += float(np.sum(np.abs(u)))
            f_new[i, idx] = f_clipped

        if mass_scale > 0.0 and abs(clip_loss) > 1e-9 * mass_scale:
            warnings.warn(
                f"apply_transport: the [0, 1] occupation clip changed the "
                f"conserved density Σ N₁^p f by {clip_loss / mass_scale:+.2%} "
                "this step — a Crank–Nicolson over/undershoot on an unresolved "
                "front (large dt against a sharp gradient). Reduce dt or "
                "resolve the front to keep the step exactly conservative.",
                stacklevel=2,
            )

        return replace(state, f=f_new)

    def _build_transport_operators(
        self, state: T3Spatial1DState, dt: float
    ) -> list[_EnergyOp | None]:
        """Per-energy Crank-Nicolson transport operators (cached).

        One entry per energy bin: ``(B, LU[A], active_indices, N_1**p)``
        for the conserved-density update ``A u^{n+1} = B u^n``, or ``None``
        where the bin has fewer than two states above the gap (nothing to
        diffuse).
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
        N1 = self._n1_per_cell(state)
        N2 = self._n2_per_cell(state)
        interface_faces = self._interface_faces(state)
        G_N = state.interface_conductance
        g_interface = (
            float(G_N) if (interface_faces and G_N is not None) else 0.0
        )

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

        ops: list[_EnergyOp | None] = []
        for i in range(NE):
            N1_i = N1[i]
            N2_i = N2[i]
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
            N2_a = N2_i[idx]
            rho_p = density_weight(N1_a, p)
            w_cell = flux_weight(D0, N1_a, q)
            g_face = _harmonic_face_weights(w_cell) * inv_dx2
            if interface_faces:
                for m in range(na - 1):
                    if int(idx[m]) in interface_faces:
                        # Kupriyanov-Lukichev finite interface conductance,
                        # energy channel:
                        # F = G_N (N_1^L N_1^R - N_2^L N_2^R)(f_L - f_R),
                        # dx-independent (the 1/dx is the flux-divergence
                        # factor). The coherence-factor weight equals
                        # (E^2 - D_L D_R)/(Omega_L Omega_R) > 0 above both
                        # gaps and is regular at matched gaps; with
                        # N_1 = N_2 = 0 on one side (sub-gap there) the
                        # interface is automatically closed.
                        weight = (
                            N1_a[m] * N1_a[m + 1] - N2_a[m] * N2_a[m + 1]
                        )
                        g_face[m] = g_interface * weight / dx
            laplacian = _flux_laplacian_from_conductances(g_face, na)
            operator = (laplacian @ sparse.diags(1.0 / rho_p)).tocsr()
            b_mat, lu = build_cn_operators(operator, dt, 1.0)
            ops.append((b_mat, lu, idx, rho_p))

        self._transport_cn_cache[key] = ops
        return ops

    def _n1_per_cell(self, state: T3Spatial1DState) -> np.ndarray:
        """BCS density of states ``N_1(E_i, x_j)``, shape ``(NE, NX)``.

        Uniform-gap path (no ``gap_profile``): ``N_1`` is x-independent,
        the spectral context's DOS at the scalar gap broadcast across the
        mesh. With a ``gap_profile`` it is evaluated per cell from the local
        gap, giving the spatially-varying DOS the dressings act on.
        """
        _NE, NX = state.f.shape
        if state.gap_profile is None:
            return np.repeat(state.spectral.rho[:, None], NX, axis=1)
        E = state.spectral.E
        columns = [bcs_density_of_states(E, float(g)) for g in state.gap_profile]
        return np.column_stack(columns)

    def _n2_per_cell(self, state: T3Spatial1DState) -> np.ndarray:
        """BCS anomalous weight ``N_2(E_i, x_j)``, shape ``(NE, NX)``.

        Companion to :meth:`_n1_per_cell`; used by the coherence-factor
        weight of the Kupriyanov-Lukichev interface faces.
        """
        _NE, NX = state.f.shape
        E = state.spectral.E
        if state.gap_profile is None:
            n2 = bcs_anomalous_weight(E, float(state.spectral.gap))
            return np.repeat(n2[:, None], NX, axis=1)
        columns = [bcs_anomalous_weight(E, float(g)) for g in state.gap_profile]
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
        """One local ETD2 collision/source step at every spatial cell."""
        self._validate_state(state)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if external_flux is not None:
            external_flux.validate_for_shape(state.f.shape)

        cache_key = (
            self._spectral_cache_key(state.spectral),
            float(state.material.tau_0),
            float(state.material.T_c),
            float(state.T_bath),
        )
        cached = self._collision_cache.get(cache_key)
        if cached is None:
            K_s0 = build_scattering_kernel_base(
                state.spectral,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            K_r0 = build_recombination_kernel_base(
                state.spectral,
                tau_0=state.material.tau_0,
                T_c=state.material.T_c,
            )
            N_p = _thermal_phonon_scattering_occupation(state.spectral.E, state.T_bath)
            N_emit, N_abs = _thermal_phonon_recombination_occupations(
                state.spectral.E,
                state.T_bath,
            )
            cached = (
                K_s0 * N_p,
                K_r0 * N_emit,
                K_r0 * N_abs,
                state.spectral.rho * state.spectral.dE,
            )
            self._collision_cache[cache_key] = cached
        K_s_eff, K_r_emit, K_r_abs, rho_dE = cached

        def rhs(f: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            one_minus = np.maximum(1.0 - f, 0.0)
            n_qp = state.spectral.rho[:, None] * f

            gain = one_minus * (K_s_eff.T @ (n_qp * state.spectral.dE[:, None]))
            loss = K_s_eff @ (rho_dE[:, None] * one_minus)

            # Kaplan Eq. (8) per-QP normalization — see
            # qpsim.collisions.phonon.phonon_collision_rates.
            loss = loss + (K_r_emit @ (rho_dE[:, None] * f))
            gain = gain + one_minus * (K_r_abs @ (rho_dE[:, None] * one_minus))

            if external_flux is not None:
                gain = gain + external_flux.gain
                loss = loss + external_flux.loss_rate
            return gain, loss

        return replace(state, f=etd2_step(state.f, rhs, dt))

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

        ``progress_hook``, when given, is called after every step with
        ``(t, max_time)``; returning ``False`` stops the run cleanly at
        the current time exactly as if ``max_time`` had been reached
        there (final state still recorded; ``converged`` unaffected).
        Physics-neutral — ``None`` leaves the loop bit-for-bit
        unchanged. Intended for progress reporting and cooperative
        cancellation from interactive callers.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        if max_time <= 0.0:
            raise ValueError("max_time must be positive.")
        if stop_tol < 0.0:
            raise ValueError("stop_tol must be non-negative.")
        if snapshot_interval is None:
            snapshot_interval = max_time / 50.0
        if snapshot_interval <= 0.0:
            raise ValueError("snapshot_interval must be positive.")

        current = state
        t = 0.0
        n_steps = 0
        next_snapshot = 0.0
        snapshots: list[SpatialSnapshot] = []

        def record(max_rate: float) -> None:
            obs = (
                {name: float(fn(current)) for name, fn in observables.items()}
                if observables
                else {}
            )
            snapshots.append(
                SpatialSnapshot(t=float(t), max_rate=float(max_rate), observables=obs)
            )

        record(float("inf"))
        next_snapshot += snapshot_interval
        converged = False
        last_max_rate = float("inf")
        max_steps = int(np.ceil(max_time / dt))

        for _ in range(max_steps):
            remaining = max_time - t
            if remaining <= 1e-12:
                break
            step_dt = min(dt, remaining)
            old_f = current.f
            current = self.step(current, step_dt, external_flux=external_flux)
            t += step_dt
            n_steps += 1
            max_rate = float(np.max(np.abs(current.f - old_f)) / step_dt)
            last_max_rate = max_rate

            if max_rate < stop_tol:
                converged = True
                record(max_rate)
                break
            if t >= next_snapshot - 1e-12:
                record(max_rate)
                next_snapshot += snapshot_interval

            if progress_hook is not None and not progress_hook(t, max_time):
                break

        if snapshots[-1].t < t:
            record(0.0 if n_steps == 0 else last_max_rate)

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


def _harmonic_face_weights(W_cell: np.ndarray) -> np.ndarray:
    """Harmonic-mean interior-face weights ``2 W_j W_{j+1} / (W_j + W_{j+1})``.

    Matches :func:`qpsim.grid.spatial_grid.build_variable_diffusion_laplacian`,
    so the discrete flux stays conservative across jumps in ``W``. A face with
    a zero-``W`` neighbour (no states there) gets zero weight.
    """
    w_left = W_cell[:-1]
    w_right = W_cell[1:]
    denom = w_left + w_right
    w_face = np.zeros(W_cell.size - 1, dtype=float)
    nonzero = denom > 0.0
    w_face[nonzero] = 2.0 * w_left[nonzero] * w_right[nonzero] / denom[nonzero]
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
    its face with ``G_N (N_1^L N_1^R - N_2^L N_2^R) / dx``. With constant
    ``g_face`` this is the standard reflective Laplacian.
    """
    off = np.asarray(g_face, dtype=float)
    main = np.zeros(n, dtype=float)
    main[:-1] -= off
    main[1:] -= off
    return sparse.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
