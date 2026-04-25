"""Tests for M25GapAsymmetricJJ — Layer-2 wrap of Stage A M25 physics.

Phase 5 of the Device Architecture: the M25 gap-asymmetric Josephson
junction is now a first-class Device-level Junction subclass that
wraps the Stage A coefficient evaluators inside the Layer-2
``Junction.evaluate(state_a, state_b, qubit_state) -> JunctionResult``
contract.

Per the design doc §6.1 caveat: this is a moment-closure wrapping,
inheriting Stage A's Fermi-Dirac per-sub-band assumption and the
moment-solver numerics. v1 ships with composability + qualitative
behavior tests; full M25 Fig 3 quantitative reproduction is Phase 5b
if the per-region Newton + outer Picard cycle resolves the
coefficient-to-density scale pathology that stranded the standalone
solve_rate_equation_steady_state at Fig 3 inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.devices import (
    Device,
    M25GapAsymmetricJJ,
    Qubit,
    Region,
)
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import load_material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.spectral import SpectralContext
from qpsim.services.rate_equation_coefficients import (
    M25PhotonDrive,
    M25PhysicalParameters,
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
)

_H_OVER_KB = 4.799243e-11   # K / Hz


# ═══════════════════════════════════════════════════════════════════════
#  Test fixtures
# ═══════════════════════════════════════════════════════════════════════


def _build_region_state(
    *, T_bath: float, gap_kelvin: float,
    num_energy: int = 30, energy_max_factor: float = 6.0,
) -> T3DiffusionState:
    """Build a Fermi-Dirac thermal state on an energy grid above ``gap_kelvin``."""
    gap_uev = gap_kelvin * KB_UEV_PER_K
    E, _ = build_energy_grid(
        gap=gap_uev, energy_min_factor=1.01,
        energy_max_factor=energy_max_factor, num_energy_bins=num_energy,
    )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=gap_uev)
    omega_bins, _, _, _ = build_phonon_frequency_map(spectral.E)
    phonon = PhononState(
        n_ph=np.zeros((1, omega_bins.size, 1)),
        omega_bins=omega_bins.reshape(1, -1),
        tau_l=np.full((1, omega_bins.size), 0.25),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    # Cheap thermal Fermi-Dirac initial guess.
    kT = KB_UEV_PER_K * T_bath
    f_init = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)
    # Material: use Al as the carrier; we just need T_c, tau_0.
    mat = load_material("Al")
    # Override the gap on a copy of the material to match this region.
    custom_mat = replace(mat, Delta_0=gap_uev)
    return T3DiffusionState(
        f=f_init, gap=gap_uev, spectral=spectral, phonon=phonon,
        material=custom_mat, T_bath=T_bath,
    )


def _fig3a_setup() -> tuple[M25PhysicalParameters, M25PhotonDrive]:
    """M25 Fig 3a (small gap asymmetry) parameters + back-solved drive."""
    params = M25PhysicalParameters(
        Delta_L_kelvin=49.5e9 * _H_OVER_KB,
        Delta_R_kelvin=49.0e9 * _H_OVER_KB,
        omega_10_kelvin=5.5e9 * _H_OVER_KB,
        T_kelvin=0.020,
        E_J_kelvin=14.5e9 * _H_OVER_KB,
        E_C_kelvin=290e6 * _H_OVER_KB,
        R_T_Hz=8.0 * 14.5e9 * (49.25 / 49.5),
        r_L_Hz=6.25e6,
        r_Rlt_Hz=6.25e6,
        Gamma_ee_10_Hz=100e3,
    )
    drive_template = M25PhotonDrive(
        omega_nu_kelvin=119e9 * _H_OVER_KB,
        Gamma_nu_scale_Hz=1.0,
        nu_0_per_J_per_m3=0.73e47,
        volume_m3=506e-6 * 240e-6 * 0.028e-6,  # 3.4e-15 m³
    )
    scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
        params, drive_template, 300.0,
    )
    drive = replace(drive_template, Gamma_nu_scale_Hz=scale)
    return params, drive


# ═══════════════════════════════════════════════════════════════════════
#  Construction + composition
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionConstruction:
    def test_rejects_self_loop(self) -> None:
        params, drive = _fig3a_setup()
        with pytest.raises(ValueError, match="couple two different regions"):
            M25GapAsymmetricJJ(
                name="J", region_a="X", region_b="X",
                m25_params=params, m25_drive=drive,
            )

    def test_constructs_with_valid_inputs(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        assert j.name == "JJ"
        # Coefficients are not computed until first evaluate.
        assert j._coefficients is None


# ═══════════════════════════════════════════════════════════════════════
#  Junction.evaluate produces sensible outputs
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionEvaluate:
    def test_evaluate_returns_correct_structure(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        # Build region states matching the M25 gaps.
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K)
        # Need a qubit_state for the channels' p_0/p_1 weighting.
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[0.5, 0.5], [0.0, 0.0]]))

        result = j.evaluate(state_L, state_R, qstate)

        # Per-region fluxes have correct shape.
        assert result.external_flux_a.gain.shape == state_L.f.shape
        assert result.external_flux_b.gain.shape == state_R.f.shape
        # Non-empty qubit channels.
        assert len(result.qubit_channels) > 0
        # All channel rates are non-negative and finite.
        for ch in result.qubit_channels:
            assert ch.rate_per_ns >= 0.0
            assert np.isfinite(ch.rate_per_ns)

    def test_evaluate_caches_coefficients(self) -> None:
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K)
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))

        # First call computes and caches.
        assert j._coefficients is None
        j.evaluate(state_L, state_R, qstate)
        assert j._coefficients is not None
        cached = j._coefficients
        # Second call reuses the same object.
        j.evaluate(state_L, state_R, qstate)
        assert j._coefficients is cached

    def test_emits_eo_and_ee_channels(self) -> None:
        # M25 has both parity-flipping (eo, from QP tunneling and
        # photon-assisted) and parity-preserving (ee) channels.
        # Verify at least one of each appears in the output.
        params, drive = _fig3a_setup()
        j = M25GapAsymmetricJJ(
            name="JJ", region_a="L", region_b="R",
            m25_params=params, m25_drive=drive,
        )
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K)
        from qpsim.devices import QubitState
        qstate = QubitState(p=np.array([[1.0, 0.0], [0.0, 0.0]]))

        result = j.evaluate(state_L, state_R, qstate)
        eo_channels = [c for c in result.qubit_channels if c.flips_parity]
        ee_channels = [c for c in result.qubit_channels if not c.flips_parity]
        assert len(eo_channels) > 0
        assert len(ee_channels) > 0  # Γ̃^ee_10 = 100 kHz at Fig 3 caption


# ═══════════════════════════════════════════════════════════════════════
#  Composition with Device + Qubit (architectural contract)
# ═══════════════════════════════════════════════════════════════════════


class TestM25JunctionInDevice:
    def test_composes_in_device_with_qubit(self) -> None:
        # Build full Device(L, R, M25Junction, Qubit) and verify it
        # composes structurally (no validation errors).
        params, drive = _fig3a_setup()
        Delta_L_K = params.Delta_L_kelvin
        Delta_R_K = params.Delta_R_kelvin
        omega_10_K = params.omega_10_kelvin

        state_L = _build_region_state(T_bath=0.020, gap_kelvin=Delta_L_K)
        state_R = _build_region_state(T_bath=0.020, gap_kelvin=Delta_R_K)
        device = Device(
            regions={
                "L": Region(name="L", state=state_L),
                "R": Region(name="R", state=state_R),
            },
            junctions=[
                M25GapAsymmetricJJ(
                    name="JJ", region_a="L", region_b="R",
                    m25_params=params, m25_drive=drive,
                ),
            ],
            qubit=Qubit(
                n_levels=2, track_parity=True,
                omega_kelvin=np.array([0.0, omega_10_K]),
                E_J_kelvin=params.E_J_kelvin,
                E_C_kelvin=params.E_C_kelvin,
            ),
        )
        # Just constructing should not raise.
        assert "L" in device.regions
        assert "R" in device.regions
        assert device.qubit is not None
        assert len(device.junctions) == 1
