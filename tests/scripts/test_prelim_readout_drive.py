"""Tests for the prelim finite-phonon readout-heating hook."""

from __future__ import annotations

import numpy as np
import pytest
import scripts.run_prelim_spatial_finite_phonon_one as finite_phonon
from qpsim.backends.t3_spatial_1d import T3Spatial1DState, T3SpatialFlux1D
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.physics.spectral import SpectralContext


def _fermi_dirac(E: np.ndarray, T: float) -> np.ndarray:
    if T <= 0.0:
        return np.zeros_like(E, dtype=float)
    kT = KB_UEV_PER_K * T
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(*, NE: int = 18, NX: int = 5) -> T3Spatial1DState:
    material = load_material("Al")
    E, _ = build_energy_grid(
        gap=material.Delta_0,
        energy_min_factor=1.01,
        energy_max_factor=5.0,
        num_energy_bins=NE,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=material.Delta_0,
        diffusion_coefficient=20.0,
    )
    x = np.linspace(0.0, 100.0, NX)
    return T3Spatial1DState(
        f=np.repeat(_fermi_dirac(E, 0.0)[:, None], NX, axis=1),
        x=x,
        gap=material.Delta_0,
        spectral=spectral,
        material=material,
        T_bath=0.0,
    )


def test_readout_drive_rejects_wrong_spatial_shape() -> None:
    state = _build_state()
    drive = finite_phonon.ReadoutPhotonDrive(
        omega_0=float(3.0 * state.spectral.dE[0]),
        n_bar=1.0,
        c_phot=1e-9,
        spatial_profile=np.ones(state.x.size - 1),
    )

    with pytest.raises(ValueError, match="spatial_profile length"):
        drive.validate_for_state(state)


def test_readout_photon_drive_uses_local_current_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _build_state()
    source = T3SpatialFlux1D.zero(*state.f.shape)
    drive = finite_phonon.ReadoutPhotonDrive(
        omega_0=float(3.0 * state.spectral.dE[0]),
        n_bar=2.0,
        c_phot=0.05,
        spatial_profile=np.linspace(1.0, 0.0, state.x.size),
    )

    def fake_photon_rates(
        f_col: np.ndarray,
        _ctx: SpectralContext,
        _omega_0: float,
        local_n_bar: float,
        c_phot: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.full_like(f_col, local_n_bar * c_phot), np.zeros_like(f_col)

    monkeypatch.setattr(
        finite_phonon,
        "sub_gap_photon_collision_rates",
        fake_photon_rates,
    )

    runner = finite_phonon.FinitePhononSpatialRunner(state, tau_l_ns=1.0)
    out = runner._qp_collision_step(
        state,
        dt_ns=0.1,
        source=source,
        readout_drive=drive,
    )

    assert float(np.mean(out[:, 0])) > float(np.mean(out[:, -1]))


def _build_probe_grid_state() -> T3Spatial1DState:
    """The exact NE=101 prelim probe grid (Al, E_min_factor=1.0, E_max=5Δ)."""
    material = load_material("Al")
    E, _ = build_energy_grid(
        gap=material.Delta_0,
        energy_min_factor=1.0,
        energy_max_factor=5.0,
        num_energy_bins=101,
    )
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=material.Delta_0,
        diffusion_coefficient=20.0,
    )
    x = np.linspace(0.0, 100.0, 11)
    return T3Spatial1DState(
        f=np.repeat(_fermi_dirac(E, 0.007)[:, None], 11, axis=1),
        x=x,
        gap=material.Delta_0,
        spectral=spectral,
        material=material,
        T_bath=0.007,
    )


def test_prelim_mode_needs_snapping_on_probe_grid() -> None:
    """Documents WHY the drive builder snaps: the physical 5.142857 GHz mode
    is ~1.64% of a cell off-grid at NE=101, above the kernel's 1% tolerance,
    so the raw mode energy would be rejected by the real kernel."""
    from qpsim.experiments.prelim_resonators import PRELIM_RESONATORS

    state = _build_probe_grid_state()
    dE = float(state.spectral.dE[0])
    omega_nominal = float(PRELIM_RESONATORS[0].probe_energy_uev)
    m = round(omega_nominal / dE)
    frac_err_vs_cell = abs(omega_nominal - m * dE) / dE
    assert 0.01 < frac_err_vs_cell < 0.05  # off-grid beyond kernel tolerance

    snapped, harmonic, rel_shift = finite_phonon.snap_omega_to_grid(omega_nominal, dE)
    assert harmonic == m
    assert snapped == m * dE
    assert 0.0 < rel_shift < 0.01  # ~0.55% of the mode energy


def test_readout_drive_from_resonator_passes_real_kernel() -> None:
    """Regression guard for the 2026-07-19 audit H2 finding: the drive built
    from the physical resonator must be accepted by the REAL sub-gap photon
    kernel (no monkeypatching) on the shipped NE=101 probe grid."""
    from qpsim.experiments.prelim_resonators import PRELIM_RESONATORS

    state = _build_probe_grid_state()
    with pytest.warns(RuntimeWarning, match="snapped to grid harmonic"):
        drive = finite_phonon.readout_drive_from_resonator(
            state,
            PRELIM_RESONATORS[0],
            n_bar=1e5,
            c_phot=1e-9,
        )
    assert drive.omega_nominal_uev == pytest.approx(
        PRELIM_RESONATORS[0].probe_energy_uev
    )
    assert drive.omega_0 != drive.omega_nominal_uev

    gain, loss = finite_phonon.sub_gap_photon_collision_rates(
        state.f[:, 0],
        state.spectral,
        drive.omega_0,
        drive.n_bar,
        drive.c_phot,
    )
    assert np.all(np.isfinite(gain))
    assert np.all(np.isfinite(loss))
    assert float(np.max(np.abs(gain))) > 0.0
