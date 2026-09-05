"""Tests for qpsim.solvers.spectral_flow_tvd."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.solvers.spectral_flow_tvd import _interface_fluxes, advect_spectral_flow


class TestAdvectSpectralFlow:
    def test_zero_gap_dot_is_no_op(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.random.default_rng(0).random(E.size)
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.0, dt=0.01)
        np.testing.assert_array_equal(u_new, u)
        assert u_new is not u  # returns a copy

    @pytest.mark.parametrize("bad_gap", [0.0, -1.0, float("nan"), float("inf")])
    def test_zero_motion_still_rejects_invalid_old_gap(self, bad_gap: float) -> None:
        E = np.linspace(1.1, 2.0, 8)
        dE = integration_widths_from_centers(E)

        with pytest.raises(ValueError, match="gap must be finite and positive"):
            advect_spectral_flow(
                np.ones(E.size), E, dE, gap=bad_gap, gap_dot=0.0, dt=0.0
            )

    def test_rejects_gap_motion_that_crosses_zero(self) -> None:
        E = np.linspace(1.1, 2.0, 8)
        dE = integration_widths_from_centers(E)

        with pytest.raises(ValueError, match="end of the step"):
            advect_spectral_flow(
                np.ones(E.size), E, dE, gap=1.0, gap_dot=-2.0, dt=1.0
            )

    def test_zero_motion_still_validates_state_shape(self) -> None:
        E = np.linspace(1.1, 2.0, 8)
        dE = integration_widths_from_centers(E)

        with pytest.raises(ValueError, match="one entry per energy bin"):
            advect_spectral_flow(
                np.ones(E.size - 1), E, dE, gap=1.0, gap_dot=0.0, dt=0.0
            )

    def test_zero_dt_still_validates_grid(self) -> None:
        E = np.array([1.1, 1.3, 1.2])
        dE = np.ones_like(E)

        with pytest.raises(ValueError, match="strictly increasing"):
            advect_spectral_flow(
                np.ones(E.size), E, dE, gap=1.0, gap_dot=0.1, dt=0.0
            )

    @pytest.mark.parametrize("bad_dt", [-1.0, float("nan"), float("inf")])
    def test_rejects_invalid_dt(self, bad_dt: float) -> None:
        E = np.linspace(1.1, 2.0, 8)
        dE = integration_widths_from_centers(E)

        with pytest.raises(ValueError, match="dt must be finite and non-negative"):
            advect_spectral_flow(
                np.ones(E.size), E, dE, gap=1.0, gap_dot=0.0, dt=bad_dt
            )

    def test_rejects_wrong_shape(self) -> None:
        import pytest
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.zeros((3, E.size))
        with pytest.raises(ValueError, match=r"u must be shape \(NE,\)"):
            advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.05, dt=0.01)

    def test_cfl_warning(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.ones(E.size)
        # Enormous dt → CFL >> 1.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=1.0, dt=1.0)
        messages = [str(w.message) for w in caught]
        assert any(
            "CFL" in message and "internally subcycling" in message
            for message in messages
        )
        assert float(np.min(out)) >= -1e-14
        np.testing.assert_allclose(
            np.sum(out * dE), np.sum(u * dE), rtol=1e-12,
        )

    def test_nonuniform_grid_subcycling_conserves_finite_volume_mass(self) -> None:
        import pytest

        E = np.geomspace(0.7, 5.0, 120)
        dE = integration_widths_from_centers(E)
        u = np.exp(-((E - 1.4) / 0.25) ** 2)
        before = float(np.sum(u * dE))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = advect_spectral_flow(
                u, E, dE, gap=1.0, gap_dot=0.4, dt=1.0,
            )
        after = float(np.sum(out * dE))
        assert after == pytest.approx(before, rel=1e-12, abs=1e-15)
        assert float(np.min(out)) >= -1e-14

    def test_nonuniform_reconstruction_uses_actual_face_distance(self) -> None:
        E = np.array([1.0, 11.0, 12.0, 13.0])
        dE = integration_widths_from_centers(E)
        u = np.array([0.0, 30.0, 31.0, 32.0]) / 32.0
        v = np.ones_like(E)

        flux = _interface_fluxes(u, E, v, dE)

        # Interface 2 is halfway between E=11 and E=12, only 0.5 from
        # either center. The old dE[1]/2 extrapolation used 2.75 and
        # reconstructed 1.109375, outside both adjacent values.
        assert flux[2] == pytest.approx(31.0 / 32.0)
        assert u[1] <= flux[2] <= u[2]

    def test_active_mask_zeros_outside(self) -> None:
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.ones(E.size)
        mask = np.zeros(E.size, dtype=bool)
        mask[5:15] = True
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=0.01, dt=0.001, active_mask=mask)
        np.testing.assert_array_equal(u_new[~mask], 0.0)

    @pytest.mark.parametrize("gap_dot, dt", [(0.0, 0.001), (1e-31, 0.001), (0.01, 0.0)])
    def test_active_mask_zeros_outside_when_nothing_moves(
        self, gap_dot: float, dt: float
    ) -> None:
        """The mask defines spectral support on the no-advection return too.

        Below the motion threshold, or at dt = 0, the step advects nothing;
        the bins outside the mask are still zeroed, so a caller reading
        support from the result gets the same answer as on a moving step.
        """
        E, _ = build_energy_grid(gap=1.0, energy_min_factor=1.01, energy_max_factor=5.0, num_energy_bins=20)
        dE = integration_widths_from_centers(E)
        u = np.ones(E.size)
        mask = np.zeros(E.size, dtype=bool)
        mask[5:15] = True
        u_new = advect_spectral_flow(u, E, dE, gap=1.0, gap_dot=gap_dot, dt=dt, active_mask=mask)
        np.testing.assert_array_equal(u_new[~mask], 0.0)
        np.testing.assert_array_equal(u_new[mask], 1.0)

    @pytest.mark.parametrize(
        "mask",
        [np.ones(20, dtype=int), np.ones(20, dtype=float)],
    )
    def test_rejects_non_boolean_active_mask(self, mask: np.ndarray) -> None:
        E, _ = build_energy_grid(
            gap=1.0,
            energy_min_factor=1.01,
            energy_max_factor=5.0,
            num_energy_bins=20,
        )
        dE = integration_widths_from_centers(E)

        with pytest.raises(ValueError, match="boolean dtype"):
            advect_spectral_flow(
                np.ones(E.size),
                E,
                dE,
                gap=1.0,
                gap_dot=0.0,
                dt=0.0,
                active_mask=mask,
            )

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
