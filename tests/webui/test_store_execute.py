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
    Spatial1DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)
from qpsim.webui.store import Workspace


def _noop_progress(_fraction: float, _message: str) -> None:
    pass


def _never() -> bool:
    return False


class TestWorkspace:
    def test_setup_save_load_list_delete(self, tmp_path: Path) -> None:
        ws = Workspace(tmp_path)
        envelope = SetupEnvelope(name="My probe sweep", setup=SteadyState0DSetup())
        slug = ws.save_setup(envelope)
        assert slug == "my-probe-sweep"
        assert [s["slug"] for s in ws.list_setups()] == [slug]
        loaded = ws.load_setup(slug)
        assert loaded.name == "My probe sweep"
        assert loaded.setup == envelope.setup
        ws.delete_setup(slug)
        assert ws.list_setups() == []

    def test_a_setup_saved_under_the_retired_mode_name_still_loads(
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
        legacy = SteadyState0DSetup().model_dump()
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
        shipped = SteadyState0DSetup().model_dump()
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
            SetupEnvelope(name="versioned", setup=SteadyState0DSetup())
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
                    "setup": SteadyState0DSetup().model_dump(),
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


def _tiny_steady_state() -> SteadyState0DSetup:
    setup = SteadyState0DSetup()
    setup.grid.num_bins = 48
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


class TestSteadyStateStrategyReducesToTheZeroDMode:
    """A 1x1 mask asking for a steady state must BE the 0-D mode.

    This is the gate the whole mode collapse rests on. It is bit-identity
    rather than agreement to a tolerance, and it holds structurally because
    both routes call the same ``run_steady_state_0d`` on a state built by the
    same ``build_state_0d`` -- a parallel implementation could agree today and
    drift on the next edit, which is exactly how this repo's recurring defect
    starts.
    """

    @staticmethod
    def _pair(**on_old):
        old = SteadyState0DSetup()
        old.grid.num_bins = 24
        old.phonons.mode = "dynamic_escape"
        for path, value in on_old.items():
            target, _, attr = path.rpartition(".")
            obj = old
            for part in target.split(".") if target else []:
                obj = getattr(obj, part)
            setattr(obj, attr, value)
        new = KineticsSetup(strategy="steady_state")
        for field in (
            "material", "T_bath", "grid", "phonons", "collisions",
            "subgap_drive", "pb_drive", "probe",
        ):
            setattr(new, field, getattr(old, field))
        # The gap switch has ONE authority on the merged mode, and it is the
        # top-level field -- solver.self_consistent_gap is refused there.
        new.self_consistent_gap = old.solver.self_consistent_gap
        new.solver = old.solver.model_copy(update={"self_consistent_gap": False})
        new.geometry.rows = new.geometry.cols = 1
        return old, new

    def _assert_identical(self, old, new) -> None:
        a = execute_setup(old, _noop_progress, _never)
        b = execute_setup(new, _noop_progress, _never)
        assert sorted(a.arrays) == sorted(b.arrays)
        for name in a.arrays:
            assert np.array_equal(a.arrays[name], b.arrays[name]), name
        assert a.summary == b.summary
        # Guard against a vacuous pass: an empty payload compares equal to an
        # empty payload.
        assert a.summary["x_qp"] > 0.0

    def test_thermal_bath(self) -> None:
        self._assert_identical(*self._pair(**{"phonons.mode": "thermal_bath"}))

    def test_dynamic_escape(self) -> None:
        self._assert_identical(*self._pair())

    def test_self_consistent_gap_reads_the_top_level_field(self) -> None:
        # min_factor < 1 because a self-consistent gap needs sub-gap cells.
        self._assert_identical(*self._pair(**{
            "solver.self_consistent_gap": True, "grid.min_factor": 0.9,
        }))

    def test_both_drives(self) -> None:
        """Both photon drives on, at frequencies the grid actually admits.

        The pair-breaking drive has to satisfy TWO conditions at once --
        omega_PB on the dE lattice, and (omega_PB - 2*E[0])/dE integral so
        every generation/recombination partner lands on a represented energy.
        Since 2*E[0] = 2*Delta + dE, the two are simultaneously satisfiable
        only when 2*Delta/dE is itself an integer, which is the same
        commensurability condition the phonon lattices obey. 24 bins does not
        satisfy it and 27 does, so the bin count is pinned here and asserted
        rather than inherited.
        """
        from qpsim.webui.builders import build_spectral

        old, new = self._pair(**{
            "subgap_drive.enabled": True, "pb_drive.enabled": True,
            "grid.num_bins": 27,
        })
        new.grid = old.grid
        E = build_spectral(old).E
        dE = float(E[1] - E[0])
        ratio = 2.0 * old.material.Delta_0 / dE
        assert abs(ratio - round(ratio)) < 1e-9, (
            f"2*Delta/dE = {ratio} is not integral, so no omega_PB satisfies "
            "both grid conditions and this test cannot run"
        )
        for s in (old, new):
            s.subgap_drive.omega_0 = dE                     # < Delta, on-lattice
            s.pb_drive.omega_PB = round(540.0 / dE) * dE    # > 2*Delta
        self._assert_identical(old, new)

    def test_a_multi_cell_mask_is_refused_by_name(self) -> None:
        """And the message must blame the solver, not the device."""
        _old, new = self._pair()
        new.geometry.rows, new.geometry.cols = 2, 3

        with pytest.raises(ValueError, match=r"steady-state solver") as excinfo:
            execute_setup(new, _noop_progress, _never)

        message = str(excinfo.value)
        assert "6 cells" in message
        assert "time_march" in message, "the message must offer the way forward"

    def test_the_duplicate_gap_switch_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="top level"):
            KineticsSetup(solver=SolverOptions(self_consistent_gap=True))


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
        setup = Transient0DSetup()
        setup.grid.num_bins = 24
        setup.dt = 1.0
        setup.total_time = 5.0
        setup.snapshot_interval = 1.0
        setup.probe.enabled = False
        fractions: list[float] = []
        payload = execute_setup(
            setup, lambda fr, _m: fractions.append(fr), _never
        )
        assert payload.arrays["f_snapshots"].shape[0] == payload.arrays["t_ns"].size
        np.testing.assert_allclose(
            payload.arrays["obs_x_qp_paper"],
            2.0 * payload.arrays["obs_x_qp"],
        )
        assert payload.summary["n_steps"] == 5
        assert payload.summary["n_etd_substeps"] >= payload.summary["n_steps"]
        assert fractions and fractions[-1] == 1.0

    def test_cancel_mid_run(self) -> None:
        setup = Transient0DSetup()
        setup.grid.num_bins = 24
        setup.dt = 1.0
        setup.total_time = 50.0
        setup.probe.enabled = False
        count = [0]

        def cancelled() -> bool:
            count[0] += 1
            return count[0] > 3

        with pytest.raises(RunCancelledError):
            execute_setup(setup, _noop_progress, cancelled)


class TestSpatial1DExecutor:
    def test_short_injection_run(self) -> None:
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 7
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

    def test_backend_warnings_are_exposed_in_run_notes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend

        original = T3Spatial1DBackend.run_until_steady_state

        def warning_wrapper(self: T3Spatial1DBackend, *args: object, **kwargs: object):
            warnings.warn(
                "spatial conservation diagnostic", RuntimeWarning, stacklevel=2
            )
            return original(self, *args, **kwargs)

        monkeypatch.setattr(
            T3Spatial1DBackend, "run_until_steady_state", warning_wrapper
        )
        setup = Spatial1DSetup()
        setup.grid.num_bins = 12
        setup.num_cells = 7
        setup.dt = 1.0
        setup.max_time = 1.0
        setup.stop_tol = 0.0

        payload = execute_setup(setup, _noop_progress, _never)

        assert "spatial conservation diagnostic" in payload.notes


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
