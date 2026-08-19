"""Kupriyanov-Lukichev interface weight on the unified backend's 1-D strip.

The 1-D reduction of :class:`qpsim.backends.t3_spatial.T3SpatialBackend`:
translated from ``TestKupriyanovLukichevWeightFixtures`` in the retired
``tests/backends/test_t3_spatial_1d.py`` (which drove
``T3Spatial1DBackend`` / ``T3Spatial1DState``). The strip is a
``(1, N)`` mask, so the gap step lives on the single face between cells
``face`` and ``face + 1``.
"""

from __future__ import annotations

import numpy as np
from qpsim.backends.t3_spatial import T3SpatialBackend, T3SpatialState
from qpsim.geometries import strip
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.bcs_quadrature import cell_edges_from_widths
from qpsim.physics.spectral import SpectralContext, bcs_density_of_states
from qpsim.transport.diffusion.base import DiffusionModel
from scipy.integrate import quad


def _varying_gap_setup(*, NE: int = 16, NX: int = 21, interface: bool = False):
    material = load_material("Al")
    base_gap = material.Delta_0
    gap_max = 1.6 * base_gap
    E, _ = build_energy_grid(
        gap=gap_max, energy_min_factor=1.05, energy_max_factor=4.0, num_energy_bins=NE
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=gap_max,
        diffusion_coefficient=6.0,
    )
    x = np.linspace(0.0, 100.0, NX)
    if interface:
        profile = np.where(np.arange(NX) < NX // 2, gap_max, base_gap).astype(float)
    else:
        profile = np.linspace(base_gap, gap_max, NX)
    return material, spectral, x, gap_max, profile


def _kl_weight(E: np.ndarray, gap_L: float, gap_R: float) -> np.ndarray:
    """Analytic KL energy weight 𝒲_L = N₁N₁′ − N₂N₂′ (eq:scalar_BC_energy)."""
    from qpsim.physics.spectral import bcs_anomalous_weight

    return (
        bcs_density_of_states(E, gap_L) * bcs_density_of_states(E, gap_R)
        - bcs_anomalous_weight(E, gap_L) * bcs_anomalous_weight(E, gap_R)
    )


def _kl_cell_average_reference(
    E: np.ndarray,
    dE: np.ndarray,
    index: int,
    gap_L: float,
    gap_R: float,
) -> float:
    """Independent direct-energy quadrature of one KL finite volume."""
    edges = cell_edges_from_widths(E, dE)
    lo = max(float(edges[index]), gap_L, gap_R)
    hi = float(edges[index + 1])
    if hi <= lo:
        return 0.0
    if gap_L == gap_R:
        return (hi - lo) / dE[index]

    def integrand(energy: float) -> float:
        return (energy * energy - gap_L * gap_R) / np.sqrt(
            (energy * energy - gap_L * gap_L)
            * (energy * energy - gap_R * gap_R)
        )

    value, _error = quad(
        integrand,
        lo,
        hi,
        points=[lo],
        epsabs=1e-11,
        epsrel=1e-11,
    )
    return value / dE[index]


class TestStripKupriyanovLukichevWeight:
    """Paper fixtures for the KL energy-channel weight (eq:scalar_BC_energy).

    𝒲_L = N₁N₁′ − N₂N₂′ must be exactly 1 at matched gaps (the
    coherence-factor cancellation that removes the SIS matched-gap
    singularity carried by the charge-channel product N₁N₁′) and reduce
    to N₁ against a normal contact.

    1-D reduction of the unified backend, translated from
    ``TestKupriyanovLukichevWeightFixtures`` of the retired
    ``T3Spatial1DBackend`` suite.
    """

    def test_matched_gap_weight_is_one(self) -> None:
        gap = 180.0
        E = np.linspace(gap * 1.0001, gap * 8.0, 4000)
        W = _kl_weight(E, gap, gap)
        np.testing.assert_allclose(W, 1.0, rtol=1e-9)

    def test_normal_contact_weight_is_N1(self) -> None:
        gap = 180.0
        E = np.linspace(gap * 1.0001, gap * 8.0, 4000)
        W = _kl_weight(E, gap, 0.0)
        np.testing.assert_allclose(W, bcs_density_of_states(E, gap), rtol=1e-12)

    def test_sub_gap_side_closes_interface(self) -> None:
        gap_L, gap_R = 288.0, 180.0
        E = np.linspace(gap_R * 1.001, gap_L * 0.999, 200)  # above R, below L
        np.testing.assert_array_equal(_kl_weight(E, gap_L, gap_R), 0.0)

    def test_backend_face_carries_exact_weight(self) -> None:
        # Extract the interface face conductance from the assembled CN
        # operator: B = I + (dt/2)·L·diag(1/ρ_p), so the (m, m+1) entry
        # is (dt/2)·g_face[m]/ρ_p[m+1] with g_face = G_N·𝒲_L/dx at the
        # gap step.
        #
        # The operators are taken from ``_transport_ops`` rather than
        # ``_build_transport_operators``: on the unified backend only the
        # former installs the Kupriyanov-Lukichev face overrides, and it is
        # the set the stepper actually solves with.
        material, spectral, x, gap_max, profile = _varying_gap_setup(interface=True)
        NE, NX = spectral.E.size, x.size
        G_N, dt = 2.0, 0.5
        dx = float(x[1] - x[0])
        state = T3SpatialState(
            f=np.zeros((NE, NX)), geometry=strip(NX, mesh_size=dx),
            spectral=spectral, material=material, T_bath=0.1,
            diffusion_model=DiffusionModel.A1,
            gap_per_cell=profile, interface_conductance=G_N,
        )
        _transport, ops = T3SpatialBackend()._transport_ops(state, dt)
        face = NX // 2 - 1  # the gap steps between cells face, face+1
        gap_L, gap_R = float(profile[face]), float(profile[face + 1])
        checked = 0
        for i, op in enumerate(ops):
            if op is None:
                continue
            b_mat, _, idx, rho_p, n_substeps, _forcing, _operator = op
            if idx[0] != 0 or idx[-1] != NX - 1:
                continue
            m = face
            W_expected = _kl_cell_average_reference(
                spectral.E, spectral.dE, i, gap_L, gap_R,
            )
            B = b_mat.toarray()
            sub_dt = dt / n_substeps
            W_measured = (
                B[m, m + 1]
                * rho_p[m + 1]
                * dx
                / (0.5 * sub_dt * G_N)
            )
            np.testing.assert_allclose(W_measured, W_expected, rtol=1e-10)
            checked += 1
        assert checked > 0
