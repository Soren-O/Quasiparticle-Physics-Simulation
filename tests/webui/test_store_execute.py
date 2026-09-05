"""Workspace persistence and the mode executors on tiny grids."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from qpsim.physics.gap_equation import calibrate_gap
from qpsim.webui.execute import RunCancelledError, execute_setup
from pydantic import ValidationError
from qpsim.webui.schemas import (
    KineticsSetup,
    M25JunctionSetup,
    SetupEnvelope,
    SolverOptions,
)
from qpsim.webui.store import Workspace


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


class TestWorkspace:
    def test_setup_save_load_list_delete(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        envelope = SetupEnvelope(name="My probe sweep", setup=KineticsSetup())
        slug = ws.save_setup(envelope)
        assert slug == "my-probe-sweep"
        assert [s["slug"] for s in ws.list_setups()] == [slug]
        loaded = ws.load_setup(slug)
        assert loaded.name == "My probe sweep"
        assert loaded.setup == envelope.setup
        ws.delete_setup(slug)
        assert ws.list_setups() == []

    def test_a_setup_saved_under_an_older_mode_name_still_loads(
        self, tmp_path: Path,
    ) -> None:
        """Renaming a mode must not make saved work unloadable.

        `store.py` round-trips `setup["mode"]` through JSON, so every setup
        already on disk names the mode it was saved under. The rename is only
        safe because `SetupEnvelope` upgrades the string before the
        discriminated union resolves on it.
        """
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        legacy = KineticsSetup().model_dump()
        legacy["mode"] = "spatial_2d"          # what a saved file says today
        (setup_dir / "legacy-mode.json").write_text(
            json.dumps({"name": "legacy mode", "setup": legacy}),
            encoding="utf-8",
        )

        loaded = ws.load_setup("legacy-mode")

        assert loaded.setup.mode == "kinetics"
        assert loaded.setup == KineticsSetup()
        # The listing reads raw JSON rather than parsing an envelope, so it
        # needs the alias applied separately or it reports a mode the picker
        # no longer offers for a setup that loads and runs perfectly.
        assert [s["mode"] for s in ws.list_setups()] == ["kinetics"]

    def test_an_unknown_mode_is_still_refused(self, tmp_path: Path) -> None:
        """The alias map is a rename table, not a way to accept anything."""
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        bogus = KineticsSetup().model_dump()
        bogus["mode"] = "spatial_3d"
        (setup_dir / "bogus.json").write_text(
            json.dumps({"name": "bogus", "setup": bogus}),
            encoding="utf-8",
        )

        with pytest.raises(ValidationError):
            ws.load_setup("bogus")

    def test_load_migrates_legacy_rho_f_units(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        legacy = KineticsSetup().model_dump()
        legacy["material"]["rho_F"] = 1.74e22  # v1: µeV^-1 m^-3
        (setup_dir / "legacy.json").write_text(
            json.dumps({"name": "legacy", "setup": legacy}),
            encoding="utf-8",
        )

        loaded = ws.load_setup("legacy")

        assert loaded.setup.material.rho_F == pytest.approx(1.74e28)

    def test_load_keeps_versionless_ev_rho_f_unmigrated(self, tmp_path: Path) -> None:
        # The shipped webui wrote versionless setups already on the
        # eV^-1 m^-3 contract (Al 1.74e28); loading one must NOT apply
        # the x1e6 µeV migration.
        ws = Workspace(tmp_path)
        setup_dir = tmp_path / "setups"
        setup_dir.mkdir(parents=True)
        shipped = KineticsSetup().model_dump()
        shipped["material"]["rho_F"] = 1.74e28
        (setup_dir / "shipped.json").write_text(
            json.dumps({"name": "shipped", "setup": shipped}),
            encoding="utf-8",
        )

        loaded = ws.load_setup("shipped")

        assert loaded.setup.material.rho_F == pytest.approx(1.74e28)

    def test_saved_setup_stamps_current_schema_version(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        slug = ws.save_setup(
            SetupEnvelope(name="versioned", setup=KineticsSetup())
        )
        stored = json.loads((ws.setups_dir / f"{slug}.json").read_text("utf-8"))
        assert stored["schema_version"] == 2

    @pytest.mark.parametrize("bad_version", [0, 2.5, 999])
    def test_load_rejects_unsupported_schema_version(
        self, tmp_path: Path, bad_version: object
    ) -> None:
        ws = Workspace(tmp_path)
        ws.setups_dir.mkdir(parents=True)
        (ws.setups_dir / "bad.json").write_text(
            json.dumps(
                {
                    "name": "bad",
                    "schema_version": bad_version,
                    "setup": KineticsSetup().model_dump(),
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unsupported setup schema_version"):
            ws.load_setup("bad")

    def test_run_manifest_and_arrays_round_trip(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        run_id = ws.new_run_id()
        ws.write_manifest(
            run_id,
            {
                "id": run_id,
                "name": "round trip",
                "mode": "steady_state_0d",
                "status": "done",
                "created": "2026-07-13T00:00:00",
                "setup": {},
                "summary": {},
                "notes": [],
            },
        )
        ws.write_arrays(run_id, {"a": np.arange(4.0)})
        assert ws.read_manifest(run_id)["status"] == "done"
        np.testing.assert_array_equal(ws.read_arrays(run_id)["a"], np.arange(4.0))
        assert [m["id"] for m in ws.list_runs()] == [run_id]
        ws.delete_run(run_id)
        assert ws.list_runs() == []


def _tiny_steady_state() -> KineticsSetup:
    """A one-cell mask solved by the steady-state root find, probe on."""
    setup = KineticsSetup(strategy="steady_state")
    setup.geometry.rows = setup.geometry.cols = 1
    setup.grid.max_factor, setup.grid.num_bins = 10.0, 48
    setup.probe.enabled = True
    return setup


def _tiny_transient() -> KineticsSetup:
    """A one-cell mask time-marched, with no early stop."""
    setup = KineticsSetup(strategy="time_march")
    setup.geometry.rows = setup.geometry.cols = 1
    setup.grid.max_factor, setup.grid.num_bins = 10.0, 24
    setup.stop_tol = 0.0
    return setup


def _tiny_strip(cells: int = 8) -> KineticsSetup:
    """A one-row mask of ``cells`` columns spanning 100 um, injection on."""
    setup = KineticsSetup(strategy="time_march")
    setup.geometry.rows, setup.geometry.cols = 1, cells
    setup.geometry.mesh_size_um = 100.0 / cells
    setup.grid.num_bins = 24
    setup.stop_tol = 0.0
    # Injection defaults OFF, so a helper that stayed silent here would build a
    # DIFFERENT run -- the profile comes out flat and "the source end carries
    # more" asserts equality against equality. Same class of trap as
    # grid.max_factor.
    setup.injection.enabled = True
    return setup


class TestSteadyState0DExecutor:
    def test_undriven_thermal_solve_recovers_fermi_dirac(self) -> None:
        setup = _tiny_steady_state()
        payload = execute_setup(setup, _noop_progress, _never)
        f = payload.arrays["f"]
        f_thermal = payload.arrays["f_thermal"]
        np.testing.assert_allclose(f, f_thermal, atol=1e-10)
        summary = payload.summary
        assert summary["x_qp"] == pytest.approx(summary["x_qp_thermal"], rel=1e-6)
        assert summary["x_qp_paper"] == pytest.approx(2.0 * summary["x_qp"])
        assert summary["x_qp_thermal_paper"] == pytest.approx(
            2.0 * summary["x_qp_thermal"]
        )
        # Probe enabled by default with ω₀ = 22 μeV < Δ.
        assert summary["Q_i"] > 0.0
        assert summary["sigma2_over_sigmaN"] > 0.0
        calibration = calibrate_gap(
            T_c=setup.material.T_c,
            T_bath=setup.T_bath,
        )
        assert summary["delta_eq_ueV"] == pytest.approx(calibration.delta_eq)
        # The default fixed-gap grid begins at Delta_0.  Its independently
        # calibrated equilibrium gap lies just below that edge, so N31's
        # fail-closed support contract correctly withholds only the
        # occupation-derived suppression fields.
        assert "delta_suppression_ueV" not in summary
        assert "rel_gap_suppression" not in summary
        assert any("below the reconstructed" in note for note in payload.notes)

    def test_dynes_collision_setup_is_rejected(self) -> None:
        setup = _tiny_steady_state()
        setup.material.dynes_gamma = 0.5

        with pytest.raises(ValueError, match="Dynes-broadened collision solves"):
            execute_setup(setup, _noop_progress, _never)

    def test_negative_sigma1_is_reported_as_active_gain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import qpsim.webui.execute as execute_mod

        monkeypatch.setattr(
            execute_mod,
            "compute_ac_conductivity",
            lambda *_args, **_kwargs: (-1.0, 2.0),
        )
        monkeypatch.setattr(
            execute_mod,
            "compute_quality_factor",
            lambda *_args, **_kwargs: -20.0,
        )

        payload = execute_setup(_tiny_steady_state(), _noop_progress, _never)

        assert payload.summary["Q_i"] == -20.0
        assert any("active microwave gain" in note for note in payload.notes)

    def test_cancel_before_solve(self) -> None:
        with pytest.raises(RunCancelledError):
            execute_setup(_tiny_steady_state(), _noop_progress, lambda: True)




class TestSetupsSavedUnderAnOlderModeNameStillLoad:
    """A setup saved under an older mode name loads, and means the same run.

    The payloads are LITERALS rather than any model's `model_dump()`: what is
    frozen here is the on-disk FORMAT, which is what a user's file actually
    contains, and which no model in the tree emits.

    A rename could be a string swap. These are translations: they carry fields
    `KineticsSetup` does not have (`total_time`, `num_cells`, `gap_profile`)
    and lack ones it requires (`geometry`, `strategy`).
    """

    @staticmethod
    def _load(setup: dict):
        return SetupEnvelope.model_validate({"name": "legacy", "setup": setup}).setup

    def test_steady_state_0d_becomes_a_one_cell_steady_state_solve(self) -> None:
        up = self._load({
            "mode": "steady_state_0d",
            "solver": {"method": "picard", "self_consistent_gap": True},
        })
        assert up.mode == "kinetics"
        assert up.strategy == "steady_state"
        assert (up.geometry.rows, up.geometry.cols) == (1, 1)
        # The gap switch moves to the single authority, and must NOT be left
        # on solver as well -- KineticsSetup refuses that duplicate outright.
        assert up.self_consistent_gap is True
        assert up.solver.self_consistent_gap is False
        assert up.solver.method == "picard", "unrelated solver settings survive"

    def test_transient_0d_becomes_a_one_cell_time_march(self) -> None:
        up = self._load({"mode": "transient_0d", "total_time": 250.0, "dt": 0.2})
        assert up.strategy == "time_march"
        assert (up.geometry.rows, up.geometry.cols) == (1, 1)
        assert up.max_time == 250.0
        assert up.dt == 0.2
        # A saved setup may carry stop_tol as None, meaning "never stop early";
        # `KineticsSetup` types it as a float, and 0.0 says the same.
        assert up.stop_tol == 0.0
        # A saved setup carrying no interval means max_time/50, which the
        # upgrade has to supply: the time march records nothing on silence, so
        # the entire time series would be dropped rather than defaulted.
        assert up.snapshot_interval == 5.0

    def test_spatial_1d_becomes_a_one_row_mask(self) -> None:
        up = self._load({
            "mode": "spatial_1d", "num_cells": 8, "length_um": 100.0,
            "injection": {"enabled": True, "where": "left_end"},
        })
        assert (up.geometry.rows, up.geometry.cols) == (1, 8)
        assert up.geometry.mesh_size_um == 12.5      # length / cells, exactly
        assert up.injection.where == "left_edge"     # renamed, same meaning

    def test_the_gap_step_lands_on_the_same_cells(self) -> None:
        """The two conventions place the interface differently.

        The strip compares a CENTRE, x_i = (i+1/2)h, against fraction*length,
        i.e. i < f*n - 1/2. The mask compares a column index, i < f*ncols. At
        f = 0.5 on 31 cells that is 15 cells against 16, so the fraction has to
        be restated as the exact cell count or the interface moves one cell and
        nothing says so.
        """
        up = self._load({
            "mode": "spatial_1d", "num_cells": 31, "length_um": 100.0,
            "gap_profile": {"kind": "step", "gap_left": 180.0,
                            "gap_right": 200.0, "step_position_fraction": 0.5},
        })
        assert up.gap_regions.kind == "column_step"
        assert up.gap_regions.step_fraction == 15 / 31
        # gap_right defaulted to 200.0 on the strip and 180.0 on the mask, so a
        # silent carry would turn a step into no step at all.
        assert up.gap_regions.gap_right == 200.0

        from qpsim.webui.builders import build_gap_per_cell_2d, build_geometry_2d
        gaps = build_gap_per_cell_2d(up, build_geometry_2d(up))
        assert int(np.count_nonzero(gaps == 180.0)) == 15
        assert int(np.count_nonzero(gaps == 200.0)) == 16

    def test_an_older_saved_setup_still_runs(self) -> None:
        up = self._load({
            "mode": "transient_0d", "total_time": 20.0, "dt": 1.0,
            "grid": {"min_factor": 1.0, "max_factor": 10.0, "num_bins": 24},
        })
        payload = execute_setup(up, _noop_progress, _never)
        assert payload.summary["x_qp_mean"] > 0.0

    def test_an_unknown_mode_is_still_refused(self) -> None:
        with pytest.raises(ValidationError):
            self._load({"mode": "spatial_3d"})


class TestTheAnswerSheetStaysComplete:
    """Every observable a reader is promised must actually reach the payload.

    The expected keys below are a FROZEN LITERAL, written out by hand rather
    than computed from the code under test. That is the whole point: a list
    derived from the emit site would agree with it by construction and could
    never catch a key silently disappearing. Adding a key here is a deliberate
    act, and so is removing one.
    """

    REQUIRED_ARRAYS = frozenset({
        "E_bins", "f_final", "f_thermal",
        "xqp_profile", "xqp_profile_paper", "x_um",
        "snap_f", "snap_t_ns", "snap_max_rate",
        "obs_x_qp_mean", "obs_x_qp_max",
        "obs_x_qp_mean_paper", "obs_x_qp_max_paper",
        "obs_Q_i",
    })
    REQUIRED_SUMMARY = frozenset({
        "converged", "gap_ueV", "n_steps", "total_time_ns",
        "x_qp_convention", "x_qp_initial", "x_qp_thermal",
        "x_qp_mean", "x_qp_max", "x_qp_mean_paper", "x_qp_max_paper",
    })

    @staticmethod
    def _payload():
        setup = KineticsSetup()
        setup.grid.num_bins = 24
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 20.0, 0.0
        setup.geometry.rows = setup.geometry.cols = 1
        setup.snapshot_interval = 5.0
        setup.probe.enabled = True
        return execute_setup(setup, _noop_progress, _never)

    def test_every_documented_observable_is_emitted(self) -> None:
        payload = self._payload()
        assert not (self.REQUIRED_ARRAYS - set(payload.arrays))
        assert not (self.REQUIRED_SUMMARY - set(payload.summary))

    def test_the_paper_convention_is_exactly_twice_the_qpsim_one(self) -> None:
        """The two conventions differ only in the denominator's spin counting.

        Pinned because it is the relation the published Fischer comparisons
        depend on, and because a 'paper' array computed by a second route
        could drift from the one it is supposed to be twice.
        """
        payload = self._payload()
        np.testing.assert_array_equal(
            payload.arrays["xqp_profile_paper"],
            2.0 * payload.arrays["xqp_profile"],
        )
        assert payload.summary["x_qp_mean_paper"] == 2.0 * payload.summary["x_qp_mean"]
        assert payload.summary["x_qp_convention"] == "qpsim: n_qp/(4 rho_F Delta_0)"

    def test_x_qp_initial_tracks_the_seed_not_the_bath(self) -> None:
        """Otherwise it silently answers a question about the seed with a
        fact about the bath, and the two coincide only for an unseeded run."""
        setup = KineticsSetup()
        setup.grid.num_bins = 24
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 5.0, 0.0
        setup.geometry.rows = setup.geometry.cols = 1
        setup.initial.kind = "excess"
        setup.initial.amplitude = 1e-3
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.summary["x_qp_initial"] > payload.summary["x_qp_thermal"], (
            "a seeded run must not report the bath's x_qp as its initial one"
        )


class TestTheProbeActsOrSaysWhyNot:
    """`probe` reached this mode with the 0-D merge, so it must not sit inert.

    A field the interface shows and the engine ignores is the defect this repo
    keeps finding. It is live under strategy='steady_state' because that route
    is the 0-D executor; this pins the time-march route as well.
    """

    @staticmethod
    def _run(rows: int, cols: int):
        setup = KineticsSetup()
        setup.grid.num_bins = 24
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 5.0, 0.0
        setup.geometry.rows, setup.geometry.cols = rows, cols
        setup.probe.enabled = True
        return execute_setup(setup, _noop_progress, _never)

    def test_a_single_cell_gets_the_observables(self) -> None:
        summary = self._run(1, 1).summary
        assert "sigma1_over_sigmaN" in summary
        assert "sigma2_over_sigmaN" in summary
        assert summary["Q_i"] != 0.0

    def test_a_multi_cell_device_is_told_why_it_cannot(self) -> None:
        """Silence would be the defect; a wrong average would be worse.

        sigma(f) is nonlinear and cells can carry different local gaps, so
        mean-of-sigma, sigma-of-mean-f and a per-cell field are three different
        physical claims. Picking one here would publish a convention nobody
        chose.
        """
        payload = self._run(2, 3)
        assert not [k for k in payload.summary if "sigma" in k or k.startswith("Q_")]
        assert any("Mattis-Bardeen probe skipped" in n for n in payload.notes)

    def test_the_probe_is_off_by_default_in_this_mode(self) -> None:
        """A merge must not change the payload of setups that predate it.

        The probe defaults ON in the 0-D modes, which are about the probe. It
        arrived here with the merge, so defaulting it on would silently add
        summary keys to every single-cell setup and a 'skipped' note to every
        multi-cell one, neither of which asked for a probe.
        """
        setup = KineticsSetup()
        assert setup.probe.enabled is False
        setup.grid.num_bins = 24
        setup.dt, setup.max_time, setup.stop_tol = 1.0, 5.0, 0.0
        setup.geometry.rows = setup.geometry.cols = 1
        payload = execute_setup(setup, _noop_progress, _never)
        assert not [k for k in payload.summary if "sigma" in k]
        assert not any("Mattis-Bardeen" in n for n in payload.notes)


class TestTransient0DExecutor:
    def test_short_relaxation_run(self) -> None:
        setup = _tiny_transient()
        setup.dt = 1.0
        setup.max_time = 5.0
        setup.snapshot_interval = 1.0
        setup.probe.enabled = False
        fractions: list[float] = []
        payload = execute_setup(
            setup, lambda fr, _m: fractions.append(fr), _never
        )
        assert payload.arrays["snap_f"].shape[0] == payload.arrays["snap_t_ns"].size
        np.testing.assert_allclose(
            payload.arrays["obs_x_qp_mean_paper"],
            2.0 * payload.arrays["obs_x_qp_mean"],
        )
        assert payload.summary["n_steps"] == 5
        # n_etd_substeps has no equivalent: it counted adaptive substeps inside
        # the retired 0-D ETD2 driver and the spatial stepper exposes none.
        assert fractions and fractions[-1] == 1.0

    def test_cancel_mid_run(self) -> None:
        setup = _tiny_transient()
        setup.dt = 1.0
        setup.max_time = 50.0
        setup.probe.enabled = False
        count = [0]

        def cancelled() -> bool:
            count[0] += 1
            return count[0] > 3

        with pytest.raises(RunCancelledError):
            execute_setup(setup, _noop_progress, cancelled)


class TestSpatial1DExecutor:
    def test_short_injection_run(self) -> None:
        setup = _tiny_strip(cells=7)
        setup.grid.num_bins = 12
        setup.dt = 1.0
        setup.max_time = 5.0
        setup.stop_tol = 0.0
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.arrays["f_final"].shape == (12, 7)
        assert payload.arrays["xqp_profile"].shape == (7,)
        np.testing.assert_allclose(
            payload.arrays["xqp_profile_paper"],
            2.0 * payload.arrays["xqp_profile"],
        )
        # Left-end injection: the source end carries more QPs.
        profile = payload.arrays["xqp_profile"]
        assert profile[0] > profile[-1]
        assert payload.summary["n_steps"] == 5


class TestM25Executor:
    def test_two_point_sweep(self) -> None:
        setup = M25JunctionSetup()
        setup.T_start_mK = 20.0
        setup.T_stop_mK = 30.0
        setup.T_points = 2
        payload = execute_setup(setup, _noop_progress, _never)
        assert payload.arrays["T_mK"].size == 2
        assert payload.summary["points_total"] == 2
        assert payload.summary["points_converged"] >= 1
        finite = np.isfinite(payload.arrays["x_L"])
        assert np.all(payload.arrays["x_L"][finite] > 0.0)
        mu = payload.arrays["mu_L_over_Delta_L"][finite]
        assert np.all((mu > 0.0) & (mu <= 1.05))

    def test_chemical_potentials_paper_exact_inversion(self) -> None:
        # Default M25 config (ω_LR = 5 GHz — the paper's Fig 3b-like
        # large-asymmetry case) at 20/25/30 mK. Guards the services-
        # layer μ inversion (M25 SI Eqs. S2/S4/S5): the naive
        # μ = Δ + T·ln(x) inversion that used to live here dropped
        # the √(Δ/2πT) prefactor and the erf/erfc sub-band partition
        # and got the μ_R> vs μ_R< ordering backwards.
        setup = M25JunctionSetup()
        setup.T_start_mK = 20.0
        setup.T_stop_mK = 30.0
        setup.T_points = 3
        payload = execute_setup(setup, _noop_progress, _never)
        T_mK = payload.arrays["T_mK"]
        mu_L = payload.arrays["mu_L_over_Delta_L"]
        mu_Rgt = payload.arrays["mu_Rgt_over_Delta_L"]
        mu_Rlt = payload.arrays["mu_Rlt_over_Delta_L"]
        i20 = int(np.argmin(np.abs(T_mK - 20.0)))
        assert np.isfinite(mu_L[i20]), "20 mK point did not converge"
        # (a) Sub-band ordering: the erf/erfc partition puts μ_R>
        # above μ_R< (the naive inversion flipped this).
        finite = np.isfinite(mu_Rgt) & np.isfinite(mu_Rlt)
        assert finite[i20]
        assert np.all(mu_Rgt[finite] > mu_Rlt[finite])
        # (b) Published Fig 3 anchor: μ_L/Δ_L ≈ 0.87 at 20 mK (both
        # panels of the paper's Fig 3 sit at this value; the pinned
        # validation baselines give 0.872 (a) / 0.869 (b)).
        assert mu_L[i20] == pytest.approx(0.872, abs=0.02)
