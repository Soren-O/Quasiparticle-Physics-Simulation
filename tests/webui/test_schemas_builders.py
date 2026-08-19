"""Schema round-trips and cross-field validation in the webui builders."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.transport.diffusion.base import DiffusionModel
from qpsim.webui.builders import (
    build_injection_2d,
    build_state_2d,
    build_m25_inputs,
    build_state_0d,
    drive_dicts,
    steady_state_solver_kwargs,
    validate_setup,
)
from qpsim.webui.schemas import (
    MODE_CLASSES,
    EnergyGrid,
    M25JunctionSetup,
    SetupEnvelope,
    KineticsSetup,
)



def _steady_state_0d() -> KineticsSetup:
    """What the retired 0-D steady-state mode is now: a one-cell root find."""
    setup = KineticsSetup(strategy="steady_state")
    setup.geometry.rows = setup.geometry.cols = 1
    setup.grid.max_factor, setup.grid.num_bins = 10.0, 405
    setup.probe.enabled = True
    return setup


def _transient_0d() -> KineticsSetup:
    """What the retired 0-D transient is now: a one-cell time march."""
    setup = KineticsSetup(strategy="time_march")
    setup.geometry.rows = setup.geometry.cols = 1
    setup.grid.max_factor, setup.grid.num_bins = 10.0, 405
    setup.dt, setup.stop_tol = 0.1, 0.0
    setup.probe.enabled = True
    return setup


def _strip_1d(cells: int = 31, length_um: float = 100.0) -> KineticsSetup:
    """What the retired 1-D strip is now: a one-row mask."""
    setup = KineticsSetup(strategy="time_march")
    setup.geometry.rows, setup.geometry.cols = 1, cells
    setup.geometry.mesh_size_um = length_um / cells
    setup.grid.num_bins = 66
    setup.injection.enabled = True
    return setup


class _StopForCapture(Exception):
    """Abort the solve once the forwarded keyword has been captured."""


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
        # A retired mode name upgrades to the mode that replaced it; the
        # partial dict still fills its defaults on the way through.
        assert isinstance(envelope.setup, KineticsSetup)
        assert envelope.setup.strategy == "steady_state"
        assert envelope.setup.grid.num_bins == 48
        assert envelope.setup.material.name == "Al"

    def test_unknown_keys_rejected(self) -> None:
        with pytest.raises(ValueError, match="extra"):
            SetupEnvelope.model_validate(
                {"name": "t", "setup": {"mode": "steady_state_0d", "bogus": 1}}
            )

    def test_energy_grid_accepts_subgap_support(self) -> None:
        grid = EnergyGrid(min_factor=0.8, max_factor=4.0, num_bins=64)
        assert grid.min_factor == pytest.approx(0.8)

    def test_dynamic_phonons_default_to_phonon_side_kernel(self) -> None:
        setup = _steady_state_0d()
        assert setup.phonons.use_phonon_side_kernel is True


class TestValidateSetup:
    def test_defaults_validate_clean(self) -> None:
        for cls in MODE_CLASSES.values():
            report = validate_setup(cls())
            assert report.ok, report.errors

    def test_subgap_drive_above_2delta_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.subgap_drive.enabled = True
        setup.subgap_drive.omega_0 = 2.5 * setup.material.Delta_0
        report = validate_setup(setup)
        assert any("2Δ" in e for e in report.errors)

    def test_pb_drive_below_2delta_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.pb_drive.enabled = True
        setup.pb_drive.omega_PB = 1.5 * setup.material.Delta_0
        report = validate_setup(setup)
        assert any("ω_PB" in e for e in report.errors)

    def test_pb_reflection_partner_misalignment_is_rejected(self) -> None:
        """Pins its own grid: the misalignment must be the test's construction.

        This used to inherit ``EnergyGrid``'s default bin count and pick a
        multiplier that happened to be misaligned on it. When that default
        moved for an unrelated reason (the phonon-lattice commensurability
        guard, 400 -> 405) the very same multiplier became ALIGNED and the test
        failed -- not because reflection-partner checking broke, but because
        the misalignment it meant to build no longer existed. A test about
        alignment must not depend on a default it does not set.
        """
        setup = _steady_state_0d()
        setup.grid.num_bins = 400
        setup.pb_drive.enabled = True
        dE = (
            (setup.grid.max_factor - setup.grid.min_factor)
            * setup.material.Delta_0
            / setup.grid.num_bins
        )
        setup.pb_drive.omega_PB = 131.0 * dE

        report = validate_setup(setup)

        assert any("reflection partners are not grid-aligned" in e for e in report.errors)

    def test_pb_aligned_frequency_and_origin_are_accepted(self) -> None:
        setup = _steady_state_0d()
        setup.pb_drive.enabled = True
        setup.grid.num_bins = 405
        dE = (
            (setup.grid.max_factor - setup.grid.min_factor)
            * setup.material.Delta_0
            / setup.grid.num_bins
        )
        setup.pb_drive.omega_PB = 132.0 * dE

        report = validate_setup(setup)

        assert not any("Pair-breaking drive" in error for error in report.errors)

    def test_incommensurate_photon_is_rejected_with_nearest_value(self) -> None:
        setup = _steady_state_0d()
        setup.subgap_drive.enabled = True
        # dE = 9Δ/400 = 4.05 μeV; ω₀ = 6.0 μeV → frac err 0.48.
        setup.subgap_drive.omega_0 = 6.0
        report = validate_setup(setup)
        assert not report.ok
        assert any("commensurate" in error for error in report.errors)
        assert any("nearest commensurate" in error for error in report.errors)

    def test_probe_at_or_above_gap_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.probe.omega_0 = setup.material.Delta_0
        report = validate_setup(setup)
        assert any("Mattis" in e for e in report.errors)

    @pytest.mark.parametrize("setup", [_steady_state_0d(), _transient_0d()])
    def test_dynes_collision_modes_are_rejected(
        self, setup: KineticsSetup
    ) -> None:
        setup.material.dynes_gamma = 1.0
        report = validate_setup(setup)
        assert not report.ok
        assert any("collision kernels" in error for error in report.errors)

    def test_t_bath_at_tc_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.T_bath = setup.material.T_c
        report = validate_setup(setup)
        assert any("T_c" in e for e in report.errors)

    def test_coupled_newton_with_thermal_bath_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.solver.method = "coupled_newton"
        report = validate_setup(setup)
        assert any("coupled-Newton" in e for e in report.errors)

    def test_coupled_newton_with_closed_phonons_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.solver.method = "coupled_newton"
        setup.phonons.mode = "dynamic_closed"
        report = validate_setup(setup)
        assert any("conserved-energy mode" in e for e in report.errors)

    def test_dynamic_default_requires_pair_breaking_time(self) -> None:
        setup = _steady_state_0d()
        setup.phonons.mode = "dynamic_escape"
        setup.material.tau_0_pb_ns = None

        report = validate_setup(setup)

        assert any("tau_0_pb_ns" in error for error in report.errors)

    def test_thermal_bath_does_not_require_pair_breaking_time(self) -> None:
        setup = _steady_state_0d()
        setup.phonons.mode = "thermal_bath"
        setup.phonons.use_phonon_side_kernel = True
        setup.material.tau_0_pb_ns = None

        report = validate_setup(setup)

        assert report.ok, report.errors

    @pytest.mark.parametrize(
        ("omega_pb", "message"),
        [
            (360.09, "crosses the 2Δ"),
            (1998.0, "partners are off-grid"),
        ],
    )
    def test_pb_preflight_matches_runtime_grid_guards(
        self, omega_pb: float, message: str
    ) -> None:
        setup = _steady_state_0d()
        setup.grid.num_bins = 90
        setup.pb_drive.enabled = True
        setup.pb_drive.omega_PB = omega_pb

        report = validate_setup(setup)

        assert not report.ok
        assert any(message in error for error in report.errors)

    @pytest.mark.parametrize("drive", ["subgap", "pair_breaking"])
    def test_zero_coupling_drive_skips_frequency_contract(self, drive: str) -> None:
        setup = _steady_state_0d()
        if drive == "subgap":
            setup.subgap_drive.enabled = True
            setup.subgap_drive.c_phot = 0.0
            setup.subgap_drive.omega_0 = 2.0 * setup.material.Delta_0
        else:
            setup.grid.num_bins = 90
            setup.pb_drive.enabled = True
            setup.pb_drive.c_phot_PB = 0.0
            setup.pb_drive.omega_PB = 360.09

        report = validate_setup(setup)

        assert report.ok, report.errors

    def test_self_consistent_gap_warns_without_subgap_support(self) -> None:
        setup = _steady_state_0d()
        setup.self_consistent_gap = True
        report = validate_setup(setup)
        assert report.ok
        assert any("does not extend below" in warning for warning in report.warnings)

        setup.grid.min_factor = 0.8
        report = validate_setup(setup)
        assert not any("does not extend below" in warning for warning in report.warnings)

    def test_pure_bcs_grid_starting_above_gap_is_rejected(self) -> None:
        setup = _steady_state_0d()
        setup.grid.min_factor = 1.01

        report = validate_setup(setup)

        assert any("grid.min_factor <= 1" in error for error in report.errors)

    def test_spatial_dynes_rejected(self) -> None:
        setup = _strip_1d()
        setup.material.dynes_gamma = 0.5
        report = validate_setup(setup)
        assert any("pure-BCS" in e for e in report.errors)

    def test_spatial_injection_outside_grid_rejected(self) -> None:
        setup = _strip_1d()
        setup.injection.center_over_delta = setup.grid.max_factor + 1.0
        report = validate_setup(setup)
        assert any("outside the energy grid" in e for e in report.errors)

    def test_spatial_gap_below_grid_rejected(self) -> None:
        setup = _strip_1d()
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_left = 0.9 * setup.material.Delta_0
        report = validate_setup(setup)
        # The merged mode words this as the gap map vs the grid floor; what
        # matters is that the guard survived the retirement, not its phrasing.
        assert any(
            "smallest local gap" in error and "energy grid starts at" in error
            for error in report.errors
        )

    def test_spatial_interface_requires_distinct_step_gaps(self) -> None:
        setup = _strip_1d()
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_left = 180.0
        setup.gap_regions.gap_right = 180.0
        setup.gap_regions.interface_G_N = 1.0

        report = validate_setup(setup)

        assert not report.ok
        assert any("interface_G_N requires distinct" in error for error in report.errors)

    def test_m25_ej_below_ec_rejected(self) -> None:
        setup = M25JunctionSetup()
        setup.E_C_over_h_GHz = setup.E_J_over_h_GHz + 1.0
        report = validate_setup(setup)
        assert any("E_J > E_C" in e for e in report.errors)


class TestBuilders:
    def test_state_0d_shapes_and_thermal_seed(self) -> None:
        setup = _steady_state_0d()
        setup.grid.num_bins = 32
        state = build_state_0d(setup)
        assert state.f.shape == (32,)
        assert np.all((state.f >= 0.0) & (state.f <= 0.5))
        assert state.phonon.n_ph.shape[1] == state.phonon.omega_bins.shape[1]

    def test_tau_l_selection_per_phonon_mode(self) -> None:
        setup = _steady_state_0d()
        setup.grid.num_bins = 16
        setup.phonons.mode = "dynamic_escape"
        setup.phonons.tau_l_ns = 0.7
        assert float(build_state_0d(setup).phonon.tau_l[0, 0]) == 0.7
        setup.phonons.mode = "dynamic_closed"
        assert float(build_state_0d(setup).phonon.tau_l[0, 0]) == 0.0

    def test_solver_kwargs_thermal_vs_dynamic(self) -> None:
        setup = _steady_state_0d()
        kwargs = steady_state_solver_kwargs(setup)
        assert kwargs["use_thermal_phonons"] is True
        assert "method" not in kwargs
        setup.phonons.mode = "dynamic_escape"
        kwargs = steady_state_solver_kwargs(setup)
        assert kwargs["method"] == "picard"
        assert kwargs["anderson_depth"] == setup.solver.anderson_depth

    def test_solver_kwargs_map_controls_to_coupled_newton(self) -> None:
        setup = _steady_state_0d()
        setup.phonons.mode = "dynamic_escape"
        setup.solver.method = "coupled_newton"
        setup.solver.newton_tol = 2.5e-7
        setup.solver.newton_max_iter = 73

        kwargs = steady_state_solver_kwargs(setup)

        assert kwargs["coupled_newton_tol"] == 2.5e-7
        assert kwargs["coupled_newton_max_iter"] == 73
        # Defaults to the analytic cross blocks. This used to be unmapped, so
        # the route silently took the backend's legacy finite-difference
        # default and paid NE + N_omega residual assemblies per iteration for
        # the same root.
        assert kwargs["coupled_newton_analytic_cross"] is True

        setup.solver.coupled_newton_analytic_cross = False
        assert steady_state_solver_kwargs(setup)["coupled_newton_analytic_cross"] is False

    def test_analytic_cross_reaches_the_solver_not_just_the_kwargs(
        self, monkeypatch
    ) -> None:
        """The kwargs dict is not the contract; what the solver receives is.

        Asserting on steady_state_solver_kwargs alone would pass even if the
        backend dropped the keyword on the floor -- which is exactly the class
        of defect this repo keeps finding. Capture it at the solver boundary.
        """
        import qpsim.backends.t3_diffusion as t3

        captured: dict[str, object] = {}

        def _capture(*args, **kwargs):
            captured.update(kwargs)
            raise _StopForCapture

        monkeypatch.setattr(t3, "coupled_newton_solve", _capture)

        setup = _steady_state_0d()
        setup.phonons.mode = "dynamic_escape"
        setup.solver.method = "coupled_newton"
        for requested in (True, False):
            captured.clear()
            setup.solver.coupled_newton_analytic_cross = requested
            state = build_state_0d(setup)
            with pytest.raises(_StopForCapture):
                t3.T3DiffusionBackend().steady_state(
                    state, **steady_state_solver_kwargs(setup),
                )
            assert captured.get("analytic_cross") is requested, (
                f"backend did not forward analytic_cross={requested}"
            )

    def test_drive_dicts_match_backend_keys(self) -> None:
        setup = _transient_0d()
        setup.subgap_drive.enabled = True
        setup.pb_drive.enabled = True
        photon, pb = drive_dicts(setup)
        assert photon is not None and set(photon) == {"omega_0", "n_bar", "c_phot"}
        assert pb is not None and set(pb) == {"omega_PB", "n_bar_PB", "c_phot_PB"}
        setup.subgap_drive.enabled = False
        setup.pb_drive.enabled = False
        assert drive_dicts(setup) == (None, None)

    def test_strip_gap_step_and_interface(self) -> None:
        """The 1-D strip's step is a one-row mask's column step."""
        setup = _strip_1d(cells=10)
        setup.grid.num_bins = 12
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_left = 170.0
        setup.gap_regions.gap_right = 200.0
        setup.gap_regions.interface_G_N = 2.0
        state = build_state_2d(setup)
        assert state.f.shape == (12, 10)
        assert state.diffusion_model is DiffusionModel.A1
        assert state.gap_per_cell is not None
        assert float(state.gap_per_cell[0]) == 170.0
        assert float(state.gap_per_cell[-1]) == 200.0
        assert state.interface_conductance == 2.0

    @pytest.mark.parametrize("cells", [2, 31])
    def test_strip_centres_span_exactly_the_requested_length(
        self, cells: int
    ) -> None:
        """Centres at (i + 1/2)h, and h*cells is the length exactly.

        Kept from the retired 1-D mode because this convention is load-bearing
        and easy to get wrong by half a cell: the reported x_um was i*h for a
        while, which is invisible in a plot and wrong in a fit.
        """
        from qpsim.webui.execute import execute_setup

        length_um = 100.0
        setup = _strip_1d(cells=cells, length_um=length_um)
        setup.grid.num_bins = 12
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 1.0, 0.0
        h = length_um / cells
        assert setup.geometry.mesh_size_um == pytest.approx(h)

        payload = execute_setup(setup, lambda *a: None, lambda: False)
        x = payload.arrays["x_um"]
        assert x[0] == pytest.approx(0.5 * h)
        assert x[-1] == pytest.approx(length_um - 0.5 * h)

    def test_xqp_profile_uses_each_local_gap_measure(self) -> None:
        from qpsim.webui.execute import _xqp_profile_2d

        setup = _strip_1d(cells=4)
        setup.grid.min_factor = 0.8
        setup.grid.num_bins = 24
        setup.gap_regions.kind = "column_step"
        setup.gap_regions.gap_left = 170.0
        setup.gap_regions.gap_right = 200.0
        state = build_state_2d(setup)
        state.f[:] = 0.01

        profile = _xqp_profile_2d(state, setup.material.Delta_0)

        assert state.gap_per_cell is not None
        for column, local_gap in enumerate(state.gap_per_cell):
            weights = bcs_dos_cell_weights(
                state.spectral.E, state.spectral.dE, float(local_gap),
            )
            expected = float(
                np.sum(weights * state.f[:, column])
            ) / setup.material.Delta_0
            assert profile[column] == pytest.approx(expected, rel=1e-14)
        assert profile[0] != pytest.approx(profile[-1])

    def test_injection_placement(self) -> None:
        setup = _strip_1d(cells=6)
        setup.grid.num_bins = 12
        state = build_state_2d(setup)
        gain, _loss = build_injection_2d(setup, state)
        assert gain.shape == (12, 6)
        # Gaussian line: positive at the source column, capped by the peak
        # rate (the exact max depends on where bins land on the line).
        assert 0.0 < np.max(gain[:, 0]) <= setup.injection.rate_per_ns
        assert np.all(gain[:, 1:] == 0.0)
        setup.injection.where = "uniform"
        gain, _loss = build_injection_2d(setup, state)
        assert np.max(gain[:, -1]) > 0.0

    def test_m25_inputs_unit_conversion(self) -> None:
        setup = M25JunctionSetup()
        params, drive = build_m25_inputs(setup, 0.020)
        # Δ_R = 49 GHz ≈ 2.35 K; Δ_L = Δ_R + 5 GHz.
        assert params.Delta_R_kelvin == pytest.approx(2.3516, rel=1e-3)
        assert params.Delta_L_kelvin > params.Delta_R_kelvin
        assert params.T_kelvin == 0.020
        assert drive.Gamma_nu_scale_Hz == 1.0


class TestPrescribedGapMap:
    """A gap map from an expression, and the grid it needs underneath it."""

    @staticmethod
    def _setup(expression: str, min_factor: float = 0.7):
        from qpsim.webui.schemas import KineticsSetup
        setup = KineticsSetup()
        setup.T_bath = 0.2
        setup.grid.num_bins = 24
        setup.grid.min_factor = min_factor
        setup.geometry.rows, setup.geometry.cols = 9, 9
        setup.gap_regions.kind = "expression"
        setup.gap_regions.expression = expression
        return setup

    WELL = (
        "gap * (1.0 - params.get('depth', 0.25) * "
        "np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / (2 * 0.15**2)))"
    )

    def test_a_radial_well_produces_a_graded_gap(self):
        """Not reachable from any step: the well is a continuum of gaps."""
        from qpsim.webui.builders import build_gap_per_cell_2d, build_geometry_2d
        setup = self._setup(self.WELL)
        setup.gap_regions.params = {"depth": 0.25}
        gaps = build_gap_per_cell_2d(setup, build_geometry_2d(setup))
        assert gaps is not None
        assert len(np.unique(gaps)) > 2, "a step would give exactly two gaps"
        assert np.min(gaps) == pytest.approx(0.75 * setup.material.Delta_0, rel=1e-9)
        assert np.max(gaps) < setup.material.Delta_0

    def test_the_grid_floor_is_checked_before_the_run(self):
        """Raised inside the quadrature this is an opaque bound violation."""
        from qpsim.webui.builders import validate_setup
        setup = self._setup(self.WELL, min_factor=1.0)
        report = validate_setup(setup)
        assert not report.ok
        assert any("smallest local gap" in e for e in report.errors)
        assert any("min_factor <= 0.75" in e for e in report.errors)

    def test_a_map_that_reaches_zero_is_refused(self):
        """A non-positive gap is a normal metal, not a smaller gap."""
        from qpsim.webui.builders import build_gap_per_cell_2d, build_geometry_2d
        setup = self._setup("gap * (1.0 - 2.0 * x)")  # negative past mid-device
        with pytest.raises(ValueError, match="strictly positive"):
            build_gap_per_cell_2d(setup, build_geometry_2d(setup))
