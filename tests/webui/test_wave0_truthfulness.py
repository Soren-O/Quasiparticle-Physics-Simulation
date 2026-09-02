"""Wave 0: the places where the app stated something that was not true.

Each test here pins a spot where a setting was accepted, displayed as active,
and then ignored -- this repo's signature defect. The audit measured the
silence; these assert it is gone.

Note the shape of the plot tests. Asserting on PNG bytes cannot show that a
figure used a particular array, so instead they render twice, with and without
it, and require the output to CHANGE. A test that only checked the call
succeeds would pass just as well with the array ignored, which is the failure
mode being guarded against.
"""

from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest
from pydantic import ValidationError

from qpsim.webui.builders import validate_setup
from qpsim.webui.plots import render_plot
from qpsim.webui.schemas import (
    KineticsSetup,
    PhononInitialCondition,
    PhononSector,
    SetupEnvelope,
)
from qpsim.webui.store import Workspace
from qpsim.webui.terms import term_status


def _steady_state() -> KineticsSetup:
    setup = KineticsSetup()
    setup.geometry.rows = 1
    setup.geometry.cols = 1
    setup.strategy = "steady_state"
    return setup


class TestSteadyStateDiscardsNothingSilently:
    """`steady_state` is a root find: no time axis, no initial condition.

    Measured before the fix: an injected setup ran clean and returned x_qp
    identical to the thermal value, i.e. the drive had no effect, while the
    terms panel reported the source as on. Wave 7 then folded a STATIC
    injection into the solver's external flux, so injection is accepted and
    acts (tests/webui/test_wave7_deferred.py holds the number); what a root
    find still cannot take -- prescribed drives, a prepared start -- is
    still refused here.
    """

    def test_a_static_injection_is_accepted_and_acts(self) -> None:
        setup = _steady_state()
        setup.injection.enabled = True
        setup.injection.rate_per_ns = 2e-1
        assert validate_setup(setup).ok

    def test_a_prescribed_drive_is_refused(self) -> None:
        from qpsim.webui.schemas import DriveSpec

        setup = _steady_state()
        setup.drives = [DriveSpec(enabled=True, amplitude=1e-3)]
        report = validate_setup(setup)
        assert not report.ok
        assert any("silently discard" in e and "drives" in e for e in report.errors)

    def test_a_non_thermal_initial_condition_is_refused(self) -> None:
        setup = _steady_state()
        setup.initial.kind = "excess"
        setup.initial.amplitude = 1e-3
        setup.initial.energy.kind = "thermal"
        setup.initial.energy.T_eff = 0.5
        report = validate_setup(setup)
        assert not report.ok
        assert any("initial.kind" in e for e in report.errors)

    def test_an_inert_drive_is_not_refused(self) -> None:
        """Zero coupling applies nothing, so nothing is being discarded.

        The switch is the COUPLING, not `enabled` -- every kernel term is
        multiplied by it. Refusing on `enabled` alone would reject setups
        where the outcome is identical either way.
        """
        setup = _steady_state()
        setup.subgap_drive.enabled = True
        setup.subgap_drive.c_phot = 0.0
        setup.subgap_drive.omega_0 = 2.0 * setup.material.Delta_0
        assert validate_setup(setup).ok

    def test_the_same_setup_is_fine_when_time_marched(self) -> None:
        setup = _steady_state()
        setup.strategy = "time_march"
        setup.injection.enabled = True
        setup.injection.rate_per_ns = 2e-1
        assert validate_setup(setup).ok

    def test_the_terms_panel_calls_the_source_on_under_both_strategies(self) -> None:
        setup = _steady_state()
        setup.injection.enabled = True
        setup.injection.rate_per_ns = 2e-1
        assert term_status(setup)["src"].state == "on"
        setup.strategy = "time_march"
        assert term_status(setup)["src"].state == "on"


class TestAPhononSeedNeedsAPhononEquation:
    def test_a_seed_under_a_pinned_bath_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="needs a phonon equation"):
            PhononSector(
                mode="thermal_bath",
                initial=PhononInitialCondition(kind="scaled", factor=2.0),
            )

    def test_the_same_seed_is_accepted_in_a_dynamic_sector(self) -> None:
        sector = PhononSector(
            mode="dynamic_escape",
            initial=PhononInitialCondition(kind="scaled", factor=2.0),
        )
        assert sector.initial.factor == 2.0

    def test_the_default_bath_seed_is_still_accepted(self) -> None:
        assert PhononSector(mode="thermal_bath").initial.kind == "bath"


class TestSavedSetupsKeepTheirBenchmark:
    def test_benchmark_survives_a_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            workspace = Workspace(pathlib.Path(d))
            slug = workspace.save_setup(
                SetupEnvelope(
                    name="Bench Setup",
                    setup=KineticsSetup(),
                    benchmark="diffusion",
                )
            )
            assert workspace.load_setup(slug).benchmark == "diffusion"

    def test_a_setup_without_one_still_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            workspace = Workspace(pathlib.Path(d))
            slug = workspace.save_setup(
                SetupEnvelope(name="Plain", setup=KineticsSetup())
            )
            assert workspace.load_setup(slug).benchmark is None


def _spatial_arrays(*, with_omega: bool, with_x: bool) -> dict[str, np.ndarray]:
    """A 1x3 strip with two frames, shaped as a real run records it."""
    arrays: dict[str, np.ndarray] = {
        "snap_t_ns": np.array([0.0, 1.0]),
        "snap_n_ph": np.linspace(1e-4, 2e-4, 2 * 4 * 3).reshape(2, 4, 3),
        "snap_f": np.linspace(1e-5, 2e-5, 2 * 5 * 3).reshape(2, 5, 3),
        "E_bins": np.linspace(180.0, 220.0, 5),
        "mask": np.ones((1, 3), dtype=np.int8),
        "xqp_profile": np.array([3e-6, 2e-6, 1e-6]),
    }
    if with_omega:
        arrays["snap_omega_bins"] = np.array([0.0, 12.0, 24.0, 36.0])
    if with_x:
        arrays["x_um"] = np.array([2.0, 6.0, 10.0])
    return arrays


_SUMMARY = {"gap_ueV": 180.0, "rows": 1, "cols": 3, "mesh_size_um": 4.0}


class TestFiguresNameThePhysicalQuantity:
    def test_the_phonon_frame_title_changes_when_the_axis_is_present(self) -> None:
        """Naming a frequency rather than a bin index.

        "bin 2" is not a physical quantity and is a different frequency on a
        different grid. Rendered with and without the recorded axis: if the
        bytes match, the axis was ignored and the title still says "bin".
        """
        with_axis = render_plot(
            "kinetics", "phonon_field_over_time",
            _spatial_arrays(with_omega=True, with_x=False), _SUMMARY,
        )
        without = render_plot(
            "kinetics", "phonon_field_over_time",
            _spatial_arrays(with_omega=False, with_x=False), _SUMMARY,
        )
        assert with_axis.startswith(b"\x89PNG\r\n\x1a\n")
        assert with_axis != without

    def test_the_profile_changes_when_distances_are_present(self) -> None:
        """Cell centres, not indices -- half a cell out is wrong in a fit."""
        with_x = render_plot(
            "kinetics", "xqp_profile",
            _spatial_arrays(with_omega=True, with_x=True), _SUMMARY,
        )
        without = render_plot(
            "kinetics", "xqp_profile",
            _spatial_arrays(with_omega=True, with_x=False), _SUMMARY,
        )
        assert with_x.startswith(b"\x89PNG\r\n\x1a\n")
        assert with_x != without


class TestPhotonDrivesActInTheSteadyState:
    """The guard above once listed the photon drives among what steady_state
    discards, by analogy with injection and without a measurement. That
    refused eight catalogue cases -- the Fischer Fig. 6 point among them --
    which are driven steady states. Measured on the catalogue's own case:
    the pair-breaking drive moves x_qp by a factor of ~150. So the guard
    must accept it, the terms panel must call it on, and this test holds
    the number that decides between the two claims.
    """

    @staticmethod
    def _driven_case() -> KineticsSetup:
        from qpsim.webui.verdicts import build_case_setup, catalogue_cases

        case = next(c for c in catalogue_cases() if c.id == "pair-breaking-driven")
        setup = build_case_setup(case)
        assert isinstance(setup, KineticsSetup)
        assert setup.strategy == "steady_state"
        assert setup.pb_drive.enabled and setup.pb_drive.c_phot_PB > 0.0
        return setup

    def test_an_acting_photon_drive_is_accepted(self) -> None:
        assert validate_setup(self._driven_case()).ok

    def test_the_terms_panel_calls_it_on(self) -> None:
        assert term_status(self._driven_case())["photpb"].state == "on"

    def test_and_it_moves_the_answer(self) -> None:
        from qpsim.webui.execute import run_kinetics

        driven = self._driven_case()
        quiet = self._driven_case()
        quiet.pb_drive.enabled = False
        x_on = run_kinetics(driven, lambda *a, **k: None, lambda: False).summary["x_qp"]
        x_off = run_kinetics(quiet, lambda *a, **k: None, lambda: False).summary["x_qp"]
        assert x_on > 10.0 * x_off, (x_on, x_off)
