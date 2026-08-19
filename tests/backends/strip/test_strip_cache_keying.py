"""The 1-D reduction of the unified spatial backend: operator-cache keying.

Translated from ``TestTransportCacheKeying`` in
``tests/backends/test_t3_spatial_1d.py``, which covered the retired
``T3Spatial1DBackend`` / ``T3Spatial1DState``. The unified
:class:`~qpsim.backends.t3_spatial.T3SpatialBackend` solves the same strip as a
``(1, N)`` mask, so the property is unchanged -- only the place the cache lives
has moved (from the backend itself onto the per-geometry
:class:`~qpsim.transport.spatial_transport.SpatialTransport`).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.constants import KB_UEV_PER_K
from qpsim.geometries import Geometry, strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _strip_geometry(NX: int = 11) -> Geometry:
    """The retired tests' grid: ``x = linspace(0, 100, NX)``, so ``dx`` is
    ``100/(NX-1)`` -- taken off the array rather than assumed."""
    x = np.linspace(0.0, 100.0, NX)
    return strip(NX, mesh_size=float(x[1] - x[0]))


def _build_state(
    geometry: Geometry,
    *,
    D0: float = 6.0,
    T_bath: float = 0.1,
    NE: int = 28,
) -> T3SpatialState:
    material = load_material("Al")
    gap = material.Delta_0
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=5.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap,
        diffusion_coefficient=D0,
    )
    ncells = geometry.cell_count
    f0 = np.repeat(_fermi_dirac(E, T_bath)[:, None], ncells, axis=1)
    return T3SpatialState(
        f=f0,
        geometry=geometry,
        spectral=spectral,
        material=material,
        T_bath=T_bath,
    )


class TestStripTransportCacheKeying:
    """Regression: the operator/kernel caches must key on the spectral
    CONTENT, not on grid shape or object identity -- a backend reused
    across states with different energy grids previously crashed
    (smaller NE) or silently froze the extra rows (larger NE).

    This is the 1-D reduction of the unified backend, translated from
    ``TestTransportCacheKeying`` of the retired ``T3Spatial1DBackend``.
    """

    def test_backend_reused_across_different_energy_grids(self) -> None:
        backend = T3SpatialBackend()
        # ONE geometry object for the whole loop, deliberately. The unified
        # backend caches its SpatialTransport -- and it is that object that
        # owns the operator cache -- on a signature containing the geometry's
        # identity, so a fresh geometry per iteration would hand out a fresh
        # empty cache and the test would assert nothing. Holding the geometry
        # fixed is what leaves the previous grid's operators sitting in the
        # cache when the next energy grid arrives.
        geometry = _strip_geometry()
        transports = []
        for NE in (28, 20, 36):
            state = _build_state(geometry, T_bath=0.1, NE=NE)
            centre = state.f.shape[1] // 2
            f = state.f.copy()
            f[:, centre] = np.clip(f[:, centre] + 0.1, 0.0, 1.0)
            state = replace(state, f=f)
            out = backend.apply_transport(state, dt=0.05)
            transports.append(backend._transport_for(state))
            moved = np.abs(out.f - state.f).max(axis=1)
            # Every energy row is above the gap in _build_state, so every
            # row must diffuse — a stale cached operator for a previous
            # grid leaves the tail rows exactly untouched.
            assert moved.min() > 0.0, (
                f"NE={NE}: some energy rows did not diffuse "
                f"(stale transport-operator cache reused across grids)"
            )
        # Non-vacuity guard: the operator cache is only under test while the
        # SAME SpatialTransport is reused across the three grids. If a future
        # change makes the backend rebuild it per state, this test would go
        # green for the wrong reason -- so fail here instead.
        assert all(t is transports[0] for t in transports), (
            "the backend rebuilt its SpatialTransport between energy grids, "
            "so the operator cache was never actually reused"
        )
