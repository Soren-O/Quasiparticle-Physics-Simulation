"""Tests for qpsim.solvers.spectral_flow_tvd."""

from __future__ import annotations

import warnings

import numpy as np
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.solvers.spectral_flow_tvd import advect_spectral_flow


class TestAdvectSpectralFlow:
    def test_zero_gap_dot_is_no_op(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.random.default_rng(0).random(E.size)
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.0, dt=0.01)
        np.testing.assert_array_equal(u_new, u)
        assert u_new is not u  # returns a copy

    def test_handles_two_component_shape(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.random.default_rng(1).random((2, E.size))
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.05, dt=0.01)
        assert u_new.shape == u.shape

    def test_rejects_wrong_shape(self) -> None:
        import pytest
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.zeros((3, E.size))
        with pytest.raises(ValueError, match=r"\(NE,\) or \(2, NE\)"):
            advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.05, dt=0.01)

    def test_cfl_warning(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.ones(E.size)
        # Enormous dt → CFL >> 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=10.0, dt=1.0)
        assert any("CFL" in str(w.message) for w in caught)

    def test_active_mask_zeros_outside(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.ones(E.size)
        mask = np.zeros(E.size, dtype=bool)
        mask[5:15] = True
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.01, dt=0.001, active_mask=mask)
        np.testing.assert_array_equal(u_new[~mask], 0.0)

    def test_total_mass_conserved(self) -> None:
        # Zero-flux BCs at both ends of the energy grid ⇒ ∫u·dE is
        # the discrete invariant of ∂_t u + ∂_E(v u) = 0. (A uniform u
        # is *not* preserved when v varies across the grid — that's
        # real advection physics, not a numerical artifact.)
        import pytest

        E, _ = build_energy_grid(
            gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=50
        )
        dE = integration_widths_from_centers(E)
        u = np.exp(-(((E - 2.5) / 0.5) ** 2))  # Gaussian well inside the grid
        total_before = float(np.sum(u * dE))
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.01, dt=0.001)
        total_after = float(np.sum(u_new * dE))
        assert total_after == pytest.approx(total_before, rel=1e-10)
