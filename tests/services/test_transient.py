"""Tests for the v1 transient driver (collisions-only, frozen n_ph)."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.observables.density import qp_fraction
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext
from qpsim.services.transient import run_time_dependent


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(T_bath: float = 0.1, num_energy: int = 40) -> T3DiffusionState:
    material = load_material("Al")
    gap = 1.764 * KB_UEV_PER_K * material.T_c
    E, _ = build_energy_grid(
        gap=gap,
        energy_min_factor=1.01,
        energy_max_factor=6.0,
        num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap)
    omega, _, _, _ = build_phonon_frequency_map(E)
    phonon = PhononState(
        n_ph=thermal_phonon_occupation(omega, T_bath).reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.zeros((1, omega.size)),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=_fermi_dirac(E, T_bath),
        gap=gap,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


class TestSteadyStateStability:
    """A state that starts at steady state should stay there."""

    def test_thermal_equilibrium_is_stationary(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=30)
        result = run_time_dependent(
            state, dt=1.0, total_time=50.0, snapshot_interval=10.0,
        )
        # ΔF < 1e-9 over 50 ns at f_FD = thermal_phonon = n_th.
        final_f = result.snapshots[-1].f
        assert np.max(np.abs(final_f - state.f)) < 1e-9


class TestRelaxationToSteadyState:
    """A perturbed initial state should relax back to f_FD under the
    collision operator with thermal n_ph and no drive."""

    def test_heated_f_decays_toward_thermal(self) -> None:
        # Pick T_bath warm enough that thermal f_FD is measurable (not
        # already at 1e-9-and-below floor). At T=0.4 K, Δ/kT ≈ 5 so
        # f_FD(E=Δ) ≈ 7e-3, and 10× inflation gives ~7e-2 — well above
        # any floor/roundoff.
        state = _build_state(T_bath=0.4, num_energy=30)
        f_initial = np.clip(state.f * 10.0, 0.0, 1.0)
        state.f = f_initial.copy()
        x_qp_start = qp_fraction(state.f, state.spectral, delta_0=state.gap)

        result = run_time_dependent(
            state,
            dt=0.5,
            total_time=2000.0,
            snapshot_interval=100.0,
            observables={
                "x_qp": lambda s: qp_fraction(s.f, s.spectral, delta_0=s.gap),
            },
        )
        x_qp_series = [snap.observables["x_qp"] for snap in result.snapshots]
        # x_qp strictly decreases — no drive, only thermal n_ph.
        assert x_qp_series[-1] < x_qp_start
        assert all(
            x_qp_series[i + 1] <= x_qp_series[i] * (1.0 + 1e-6)
            for i in range(len(x_qp_series) - 1)
        )


class TestEarlyStopping:
    def test_stop_tol_triggers_early_exit(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=30)
        # Already at steady state: first substep's rate of change
        # should be below any reasonable tol.
        result = run_time_dependent(
            state, dt=1.0, total_time=10000.0,
            snapshot_interval=1.0, stop_tol=1e-6,
        )
        assert result.converged
        # Should exit after essentially 1 step (at most 2 for safety).
        assert result.n_steps <= 2


class TestSnapshotCadence:
    def test_snapshot_count_matches_interval(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        result = run_time_dependent(
            state, dt=0.5, total_time=10.0, snapshot_interval=2.0,
        )
        # Snapshots at t = 0 (always), then every 2 ns through t=10.
        # That's t = 0, 2, 4, 6, 8, 10 → 6 snapshots.
        expected_times = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
        snap_times = [snap.t for snap in result.snapshots]
        np.testing.assert_allclose(snap_times, expected_times, atol=0.5)
        assert len(snap_times) == 6

    def test_default_snapshot_count_is_roughly_50(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        result = run_time_dependent(state, dt=1.0, total_time=50.0)
        # Default snapshot_interval = total_time/50 = 1.0, so we get
        # 51 snapshots (t=0 plus t=1…50).
        assert 45 <= len(result.snapshots) <= 55


class TestObservables:
    def test_observable_callables_invoked_per_snapshot(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=20)
        call_count = [0]

        def _counting_obs(_state: T3DiffusionState) -> float:
            call_count[0] += 1
            return float(call_count[0])

        result = run_time_dependent(
            state, dt=1.0, total_time=5.0, snapshot_interval=1.0,
            observables={"counter": _counting_obs},
        )
        # 6 snapshots (t=0 through t=5), 6 observable calls.
        assert call_count[0] == 6
        counter_values = [snap.observables["counter"] for snap in result.snapshots]
        assert counter_values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


class TestInputValidation:
    def test_zero_dt_rejected(self) -> None:
        state = _build_state(num_energy=10)
        with pytest.raises(ValueError, match="dt"):
            run_time_dependent(state, dt=0.0, total_time=1.0)

    def test_zero_total_time_rejected(self) -> None:
        state = _build_state(num_energy=10)
        with pytest.raises(ValueError, match="total_time"):
            run_time_dependent(state, dt=0.1, total_time=0.0)

    def test_zero_snapshot_interval_rejected(self) -> None:
        state = _build_state(num_energy=10)
        with pytest.raises(ValueError, match="snapshot_interval"):
            run_time_dependent(
                state, dt=0.1, total_time=1.0, snapshot_interval=0.0,
            )

    def test_negative_stop_tol_rejected(self) -> None:
        state = _build_state(num_energy=10)
        with pytest.raises(ValueError, match="stop_tol"):
            run_time_dependent(
                state, dt=0.1, total_time=1.0, stop_tol=-1e-3,
            )


class TestDriveKick:
    """Start at thermal equilibrium, turn on a sub-gap drive, watch f
    climb toward the driven steady state."""

    def test_x_qp_rises_under_drive(self) -> None:
        state = _build_state(T_bath=0.1, num_energy=40)
        drive = {"omega_0": 20.0, "n_bar": 1e8, "c_phot": 1e-9}

        result = run_time_dependent(
            state,
            dt=0.5,
            total_time=200.0,
            snapshot_interval=20.0,
            photon_params=drive,
            observables={
                "x_qp": lambda s: qp_fraction(s.f, s.spectral, delta_0=s.gap),
            },
        )
        x_qp_series = [snap.observables["x_qp"] for snap in result.snapshots]
        # Non-decreasing (drive adds QPs), ends strictly above initial.
        assert x_qp_series[-1] > x_qp_series[0]
        assert all(
            x_qp_series[i + 1] >= x_qp_series[i] * 0.99
            for i in range(len(x_qp_series) - 1)
        )
