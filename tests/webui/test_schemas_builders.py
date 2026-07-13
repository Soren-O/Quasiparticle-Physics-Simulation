"""Schema round-trips and cross-field validation in the webui builders."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.transport.diffusion.base import DiffusionModel
from qpsim.webui.builders import (
    build_injection_flux,
    build_m25_inputs,
    build_state_0d,
    build_state_1d,
    drive_dicts,
    steady_state_solver_kwargs,
    validate_setup,
)
from qpsim.webui.schemas import (
    MODE_CLASSES,
    M25JunctionSetup,
    SetupEnvelope,
    Spatial1DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)


class TestSchemas:
    def test_every_mode_default_constructs_and_round_trips(self) -> None:
        for mode, cls in MODE_CLASSES.items():
            setup = cls()
            assert setup.mode == mode
            envelope = SetupEnvelope.model_validate(
                {"name": "t", "setup": setup.model_dump()}
            )
            assert envelope.setup == setup

    def test_partial_dict_fills_defaults(self) -> None:
        envelope = SetupEnvelope.model_validate(
            {"name": "t", "setup": {"mode": "steady_state_0d", "grid": {"num_bins": 48}}}
        )
        assert isinstance(envelope.setup, SteadyState0DSetup)
        assert envelope.setup.grid.num_bins == 48
        assert envelope.setup.material.name == "Al"

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            SetupEnvelope.model_validate(
                {"name": "t", "setup": {"mode": "steady_state_0d", "bogus": 1}}
            )


class TestValidateSetup:
    def test_defaults_validate_clean(self) -> None:
        for cls in MODE_CLASSES.values():
            report = validate_setup(cls())
            assert report.ok, report.errors

    def test_subgap_drive_above_2delta_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.subgap_drive.enabled = True
        setup.subgap_drive.omega_0 = 2.5 * setup.material.Delta_0
        report = validate_setup(setup)
        assert any("2Δ" in e for e in report.errors)

    def test_pb_drive_below_2delta_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.pb_drive.enabled = True
        setup.pb_drive.omega_PB = 1.5 * setup.material.Delta_0
        report = validate_setup(setup)
        assert any("ω_PB" in e for e in report.errors)

    def test_incommensurate_photon_warns_with_snap_value(self) -> None:
        setup = SteadyState0DSetup()
        setup.subgap_drive.enabled = True
        # dE = 9Δ/400 = 4.05 μeV; ω₀ = 6.0 μeV → frac err 0.48.
        setup.subgap_drive.omega_0 = 6.0
        report = validate_setup(setup)
        assert report.ok
        assert any("commensurate" in w for w in report.warnings)

    def test_probe_at_or_above_gap_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.probe.omega_0 = setup.material.Delta_0
        report = validate_setup(setup)
        assert any("Mattis" in e for e in report.errors)

    def test_dynes_probe_warns_not_errors(self) -> None:
        setup = SteadyState0DSetup()
        setup.material.dynes_gamma = 1.0
        report = validate_setup(setup)
        assert report.ok
        assert any("Dynes" in w for w in report.warnings)

    def test_t_bath_at_tc_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.T_bath = setup.material.T_c
        report = validate_setup(setup)
        assert any("T_c" in e for e in report.errors)

    def test_coupled_newton_with_thermal_bath_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.solver.method = "coupled_newton"
        report = validate_setup(setup)
        assert any("coupled-Newton" in e for e in report.errors)

    def test_coupled_newton_with_closed_phonons_rejected(self) -> None:
        setup = SteadyState0DSetup()
        setup.solver.method = "coupled_newton"
        setup.phonons.mode = "dynamic_closed"
        report = validate_setup(setup)
        assert any("conserved-energy mode" in e for e in report.errors)

    def test_spatial_dynes_rejected(self) -> None:
        setup = Spatial1DSetup()
        setup.material.dynes_gamma = 0.5
        report = validate_setup(setup)
        assert any("pure-BCS" in e for e in report.errors)

    def test_spatial_injection_outside_grid_rejected(self) -> None:
        setup = Spatial1DSetup()
        setup.injection.center_over_delta = setup.grid.max_factor + 1.0
        report = validate_setup(setup)
        assert any("outside the energy grid" in e for e in report.errors)

    def test_m25_ej_below_ec_rejected(self) -> None:
        setup = M25JunctionSetup()
        setup.E_C_over_h_GHz = setup.E_J_over_h_GHz + 1.0
        report = validate_setup(setup)
        assert any("E_J > E_C" in e for e in report.errors)


class TestBuilders:
    def test_state_0d_shapes_and_thermal_seed(self) -> None:
        setup = SteadyState0DSetup()
        setup.grid.num_bins = 32
        state = build_state_0d(setup)
        assert state.f.shape == (32,)
        assert np.all((state.f >= 0.0) & (state.f <= 0.5))
        assert state.phonon.n_ph.shape[1] == state.phonon.omega_bins.shape[1]

    def test_tau_l_selection_per_phonon_mode(self) -> None:
        setup = SteadyState0DSetup()
        setup.grid.num_bins = 16
        setup.phonons.mode = "dynamic_escape"
        setup.phonons.tau_l_ns = 0.7
        assert float(build_state_0d(setup).phonon.tau_l[0, 0]) == 0.7
        setup.phonons.mode = "dynamic_closed"
        assert float(build_state_0d(setup).phonon.tau_l[0, 0]) == 0.0

    def test_solver_kwargs_thermal_vs_dynamic(self) -> None:
        setup = SteadyState0DSetup()
        kwargs = steady_state_solver_kwargs(setup)
        assert kwargs["use_thermal_phonons"] is True
        assert "method" not in kwargs
        setup.phonons.mode = "dynamic_escape"
        kwargs = steady_state_solver_kwargs(setup)
        assert kwargs["method"] == "picard"
        assert kwargs["anderson_depth"] == setup.solver.anderson_depth

    def test_drive_dicts_match_backend_keys(self) -> None:
        setup = Transient0DSetup()
        setup.subgap_drive.enabled = True
        setup.pb_drive.enabled = True
        photon, pb = drive_dicts(setup)
        assert photon is not None and set(photon) == {"omega_0", "n_bar", "c_phot"}
        assert pb is not None and set(pb) == {"omega_PB", "n_bar_PB", "c_phot_PB"}
        setup.subgap_drive.enabled = False
        setup.pb_drive.enabled = False
        assert drive_dicts(setup) == (None, None)

    def test_state_1d_gap_step_and_interface(self) -> None:
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 10
        setup.gap_profile.kind = "step"
        setup.gap_profile.gap_left = 170.0
        setup.gap_profile.gap_right = 200.0
        setup.gap_profile.interface_G_N = 2.0
        state = build_state_1d(setup)
        assert state.f.shape == (12, 10)
        assert state.diffusion_model is DiffusionModel.A1
        assert state.gap_profile is not None
        assert float(state.gap_profile[0]) == 170.0
        assert float(state.gap_profile[-1]) == 200.0
        assert state.interface_conductance == 2.0

    def test_injection_flux_placement(self) -> None:
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 6
        state = build_state_1d(setup)
        flux = build_injection_flux(setup, state)
        assert flux is not None
        assert flux.gain.shape == (12, 6)
        # Gaussian line: positive at the source column, capped by the peak
        # rate (the exact max depends on where bins land on the line).
        assert 0.0 < np.max(flux.gain[:, 0]) <= setup.injection.rate_per_ns
        assert np.all(flux.gain[:, 1:] == 0.0)
        setup.injection.where = "uniform"
        flux = build_injection_flux(setup, state)
        assert flux is not None
        assert np.max(flux.gain[:, -1]) > 0.0

    def test_m25_inputs_unit_conversion(self) -> None:
        setup = M25JunctionSetup()
        params, drive = build_m25_inputs(setup, 0.020)
        # Δ_R = 49 GHz ≈ 2.35 K; Δ_L = Δ_R + 5 GHz.
        assert params.Delta_R_kelvin == pytest.approx(2.3516, rel=1e-3)
        assert params.Delta_L_kelvin > params.Delta_R_kelvin
        assert params.T_kelvin == 0.020
        assert drive.Gamma_nu_scale_Hz == 1.0
