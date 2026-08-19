"""Gap-edge packet fixture on the unified backend, reduced to 1-D.

The 1-D reduction of :class:`qpsim.backends.t3_spatial.T3SpatialBackend`: the
strip is a ``(1, N)`` mask, so the 5-point Laplacian degenerates to the
3-point chain exactly. Translated from ``TestGapEdgePacketFixture`` in the
retired ``tests/backends/test_t3_spatial_1d.py``.
"""

from __future__ import annotations

import numpy as np
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext
from qpsim.transport.diffusion.base import DiffusionModel


class TestStripGapEdgePacket:
    """Paper fixture: a packet pushed against a spatial gap ramp must
    conserve ∫N₁f with zero leakage past the local gap edge (the
    weak-form zero-flux face of paper §V — diffusive Andreev
    retroreflection for the energy mode).

    The 1-D reduction of the unified backend; came from
    ``TestGapEdgePacketFixture`` on the retired ``T3Spatial1DBackend``.
    """

    def test_packet_conserves_with_zero_subedge_leakage(self) -> None:
        # Custom grid: the energies must span the gap band
        # (base_gap, gap_max) so mid-band energies have their local edge
        # inside the strip (the shared helper's grid starts above
        # gap_max and would never see an edge).
        material = load_material("Al")
        base_gap = material.Delta_0
        gap_max = 1.6 * base_gap
        NE, NX = 24, 41
        E, _ = build_energy_grid(
            gap=base_gap, energy_min_factor=1.02,
            energy_max_factor=4.8, num_energy_bins=NE,
        )
        spectral = SpectralContext(
            E_bins=E, dE_bins=integration_widths_from_centers(E),
            gap=gap_max, diffusion_coefficient=6.0,
        )
        # The retired state carried the node array itself; the unified state
        # carries a uniform mesh, so the pitch is what crosses over. The
        # packet shape is still laid out on the same coordinates, which keeps
        # the initial condition identical to the 1-D fixture's.
        x = np.linspace(0.0, 100.0, NX)
        dx = float(x[1] - x[0])
        profile = np.linspace(base_gap, gap_max, NX)
        # Packet near the low-gap end; diffusion pushes it up the ramp
        # into each energy's local edge.
        f0 = np.tile(0.3 * np.exp(-(((x - 15.0) / 8.0) ** 2)), (NE, 1))
        state = T3SpatialState(
            f=f0.copy(), geometry=strip(NX, mesh_size=dx), spectral=spectral,
            material=material, T_bath=0.1,
            diffusion_model=DiffusionModel.A1, gap_per_cell=profile,
        )
        backend = T3SpatialBackend()
        N1 = backend._n1_per_cell(state)
        state.f[N1 == 0.0] = 0.0  # no occupation below the local edge
        before = (N1 * state.f).sum(axis=1)  # per-energy conserved density

        evolving = state
        for _ in range(200):
            evolving = backend.apply_transport(evolving, 1.0)

        after = (N1 * evolving.f).sum(axis=1)
        sub_edge = N1 == 0.0

        # mid-band energies (edge inside the grid) must have hit the edge
        mid_band = (profile.min() < spectral.E) & (profile.max() > spectral.E)
        assert mid_band.any()
        hit = 0
        for i in np.flatnonzero(mid_band):
            active = np.flatnonzero(N1[i] > 0.0)
            if active.size and evolving.f[i, active[-1]] > 1e-6:
                hit += 1
        assert hit > 0

        # exact conservation of the per-energy ∫N₁f and zero leakage
        nz = before > 0
        np.testing.assert_allclose(after[nz], before[nz], rtol=1e-11)
        assert float(np.abs(evolving.f[sub_edge]).max(initial=0.0)) == 0.0
