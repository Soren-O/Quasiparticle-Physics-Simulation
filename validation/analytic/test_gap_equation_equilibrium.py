"""Layer-1 validation — gap equation recovers Δ_eq at Fischer scale.

At ``f = f_FD(T_B)``, the gap self-consistency solver must
return ``Δ_eq(T_B)``. The unit-test suite has a small-scale version
of this in ``tests/physics/test_gap_equation.py``; this is the
Fischer-scale version (1620 bins / Fischer Table I).

Tolerance note
--------------
``solve_gap`` treats the kinetic occupation as a finite-volume,
cell-constant field and integrates the singular gap kernel exactly in each
cell. At finite temperature the remaining round-trip error is therefore the
cell-constant representation error relative to the continuous calibration,
not an off-grid interpolation error. At very low temperature, where
``f ≈ 0`` on the active window, even that contribution vanishes.
"""

from __future__ import annotations

import numpy as np
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.physics.gap_equation import calibrate_gap, solve_gap
from qpsim.physics.spectral import fermi_dirac_occupation


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    return fermi_dirac_occupation(E, T)


class TestGapEquilibriumFischerScale:
    def test_round_trip_at_Fischer_params(self) -> None:
        # Fischer Table I: Δ₀ = 180 μeV, T_c from BCS ratio, T_bath = 0.1 K.
        delta_0 = 180.0
        T_c = delta_0 / (1.764 * KB_UEV_PER_K)
        T_bath = 0.1

        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        # At low T_bath / T_c the calibration gives Δ_eq ≈ Δ(0).
        assert cal.delta_eq / delta_0 > 0.999

        # Build the Fischer-scale E grid from Δ_eq and solve the runtime
        # gap equation on f_FD(E, T_bath) — should recover Δ_eq.
        E, _ = build_energy_grid(
            gap=cal.delta_eq, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=1620
        )
        integration_widths_from_centers(E)  # grid validation
        f_FD = _fermi_dirac(E, T_bath)
        delta_recovered = solve_gap(cal, f_FD, E)
        rel_err = abs(delta_recovered - cal.delta_eq) / cal.delta_eq
        # At T=0.1 K, f ≈ 0 on the active window so the finite-volume
        # representation error vanishes; Brent xtol sets the floor.
        assert rel_err < 1e-8

    def test_round_trip_at_half_Tc(self) -> None:
        # Closer to T_c, where Δ_eq is suppressed below Δ(0), the
        # cell-constant representation error in solve_gap dominates.
        delta_0 = 180.0
        T_c = delta_0 / (1.764 * KB_UEV_PER_K)
        T_bath = 0.5 * T_c  # ≈ 0.59 K

        cal = calibrate_gap(T_c=T_c, T_bath=T_bath)
        assert 0.0 < cal.delta_eq < delta_0  # gap is partially suppressed

        E, _ = build_energy_grid(
            gap=cal.delta_eq, energy_min_factor=1.0, energy_max_factor=10.0, num_energy_bins=1620
        )
        f_FD = _fermi_dirac(E, T_bath)
        delta_recovered = solve_gap(cal, f_FD, E)
        rel_err = abs(delta_recovered - cal.delta_eq) / cal.delta_eq
        # At T=T_c/2 and 1620 bins this is approximately 3e-5
        # (grid-refinement verified).
        assert rel_err < 9e-5
