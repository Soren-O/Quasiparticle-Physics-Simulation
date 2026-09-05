# ruff: noqa: N999  (file name is the review packet id, fixed by the workflow)
"""Packet P14 regression tests (2026-08-03 code review).

Two guards:

* :class:`TestDeviceRegionKeyReconciliation` — ``Device.__post_init__`` now
  rejects a ``regions`` mapping whose key contradicts its own
  ``Region.name``. The dict key is the sole identity the Device layer
  resolves by, so a transposed mapping used to bind silently to the key.
* :class:`TestFig6ThermalReferenceConvention` — pins the KNOWN, unfixed
  continuum-vs-finite-volume thermal-reference offset of the promoted
  Fig. 6 ordinate (see the ``fig6_solve`` module docstring). These values
  are NOT the physically correct answer — the zero-drive ordinate should be
  exactly 0 — they are the current convention, pinned so that adopting the
  grid-consistent reference becomes a deliberate, visible regeneration
  rather than silent drift.
"""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.diffusion import DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import Device, Region, SymmetricGapTunnelingJunction
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononState
from qpsim.physics.spectral import SpectralContext


def _thermal_state(T_bath: float) -> DiffusionState:
    """Minimal Al-like diffusion state at thermal equilibrium (self-contained)."""
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap, energy_min_factor=1.01, energy_max_factor=6.0, num_energy_bins=16,
    )
    spectral = SpectralContext(
        E_bins=E, dE_bins=integration_widths_from_centers(E), gap=gap,
    )
    omega_bins, _, _, _ = build_phonon_frequency_map(spectral.E)
    phonon = PhononState(
        n_ph=np.zeros((1, omega_bins.size, 1)),
        omega_bins=omega_bins.reshape(1, -1),
        tau_l=np.full((1, omega_bins.size), 0.25),
        branches=[PhononBranchSpec(name="debye_average")],
    )
    kT = KB_UEV_PER_K * T_bath
    return DiffusionState(
        f=1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0),
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


class TestDeviceRegionKeyReconciliation:
    def test_transposed_region_mapping_is_rejected(self) -> None:
        state_L = _thermal_state(0.1)
        state_R = _thermal_state(0.2)
        with pytest.raises(ValueError, match=r"must equal their Region\.name"):
            Device(
                regions={
                    "L": Region(name="R", state=state_R),
                    "R": Region(name="L", state=state_L),
                },
                junctions=[
                    SymmetricGapTunnelingJunction(
                        name="J",
                        region_a="L",
                        region_b="R",
                        alpha_per_ns=1e-3,
                        capacity_ratio_a_to_b=10.0,
                    )
                ],
            )

    def test_error_names_every_offending_pair(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Device(regions={"main": Region(name="trap", state=_thermal_state(0.1))})
        assert "('main', 'trap')" in str(excinfo.value)

    def test_consistent_mapping_still_constructs(self) -> None:
        device = Device(
            regions={"main": Region(name="main", state=_thermal_state(0.1))}
        )
        assert set(device.regions) == {"main"}


class TestFig6ThermalReferenceConvention:
    """Pin the mixed-quadrature thermal reference of the promoted Fig. 6 path.

    ``obs = (Delta_driven - Delta_eq)/(Delta_0 - Delta_eq)`` takes
    ``Delta_driven`` from the finite-volume ``solve_gap`` and ``Delta_eq``
    from the continuum ``calibrate_gap``. At zero drive the kinetic root is
    the thermal distribution, so the ordinate must be 0; it is not.
    """

    # (T_bath, zero-drive ordinate offset) on the canonical 1640-cell
    # dE = 1 μeV Fig. 6 grid.
    EXPECTED = (
        (0.10, 0.00861920570515185),
        (0.15, 0.004542843782215993),
        (0.20, 0.002188321799775852),
    )

    def test_zero_drive_ordinate_is_not_zero(self) -> None:
        from qpsim.physics.gap_equation import calibrate_gap, solve_gap
        from qpsim.physics.spectral import fermi_dirac_occupation
        from validation.fischer_2023 import fig6_solve

        _, _, spectral = fig6_solve._build_grid_and_spectral()
        for T_bath, expected in self.EXPECTED:
            calibration = calibrate_gap(
                T_c=fig6_solve.T_C,
                T_bath=T_bath,
                Delta_0=fig6_solve.DELTA_0,
                xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
            )
            gap_grid = solve_gap(
                calibration,
                fermi_dirac_occupation(spectral.E, T_bath),
                spectral.E,
                dE_bins=spectral.dE,
                reference_gap=calibration.delta_eq,
                xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
            )
            delta_T = fig6_solve.DELTA_0 - calibration.delta_eq
            offset = (float(gap_grid) - calibration.delta_eq) / delta_T
            # Physics requires exactly 0 here; the convention gives this.
            assert offset > 0.0
            np.testing.assert_allclose(offset, expected, rtol=1e-9)
