"""One-dimensional spatial T3 diffusion backend.

This is a narrow Gate-5 preview for strip-like devices: ``f(E, x)`` on a
uniform 1D mesh, reflective end boundaries, Crank-Nicolson diffusion in
space, and local T3 electron-phonon collisions at fixed gap.  Phonons are
held at the thermal bath in this first spatial path; finite-``tau_l``
spatial phonons can be layered on once the Ph1/Ph2 state is available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
from scipy import sparse

from qpsim.collisions.phonon import (
    _thermal_phonon_recombination_occupations,
    _thermal_phonon_scattering_occupation,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.materials.database import Material
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.etd import etd2_step


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
    """T3 occupation on an ``(energy, position)`` mesh."""

    f: np.ndarray
    x: np.ndarray
    gap: float
    spectral: SpectralContext
    material: Material
    T_bath: float

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
        self._transport_eigen_cache: dict[
            tuple[int, float],
            tuple[np.ndarray, np.ndarray],
        ] = {}
        self._collision_cache: dict[
            tuple[int, float, float, float],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}

    def apply_transport(self, state: T3Spatial1DState, dt: float) -> T3Spatial1DState:
        """Crank-Nicolson diffusion step with reflective end boundaries."""
        self._validate_state(state)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")

        _NE, NX = state.f.shape
        if NX == 1:
            return state

        key = (NX, state.dx)
        cached = self._transport_eigen_cache.get(key)
        if cached is None:
            laplacian = _reflective_1d_laplacian(NX, state.dx).toarray()
            cached = np.linalg.eigh(laplacian)
            self._transport_eigen_cache[key] = cached
        eigenvalues, eigenvectors = cached

        modal = state.f @ eigenvectors
        alpha = 0.5 * dt * state.spectral.D_E[:, None] * eigenvalues[None, :]
        amplification = (1.0 + alpha) / (1.0 - alpha)
        f_new = (modal * amplification) @ eigenvectors.T

        return replace(state, f=np.clip(f_new, 0.0, 1.0))

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
            id(state.spectral),
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

            loss = loss + 2.0 * (K_r_emit @ (rho_dE[:, None] * f))
            gain = gain + 2.0 * one_minus * (K_r_abs @ (rho_dE[:, None] * one_minus))

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
    ) -> SpatialTransientResult:
        """Run fixed-step dynamics until ``max|df/dt| < stop_tol`` or timeout."""
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
        if x.size > 1 and not np.allclose(np.diff(x), np.diff(x)[0]):
            raise ValueError("state.x must be uniformly spaced.")
        if np.any(~np.isfinite(f)):
            raise ValueError("state.f contains non-finite values.")
        if np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("state.f must lie in [0, 1].")


def _reflective_1d_laplacian(NX: int, dx: float) -> sparse.csr_matrix:
    """Finite-volume 1D Laplacian with zero-flux boundary faces."""
    if NX <= 0:
        raise ValueError("NX must be positive.")
    if dx <= 0.0:
        raise ValueError("dx must be positive.")
    if NX == 1:
        return sparse.csr_matrix((1, 1))

    main = -2.0 * np.ones(NX)
    upper = np.ones(NX - 1)
    lower = np.ones(NX - 1)
    main[0] = -1.0
    main[-1] = -1.0
    return sparse.diags([lower, main, upper], offsets=[-1, 0, 1], format="csr") / dx**2
