"""Regression test: Fischer 2023 Fig. 6 paper-topology run matches the pinned CSV.

Iterative-mode tolerance per NFP §6.4.1 (1e-6). Slow-marked --- this
sweep does $|T_B|\\times|\\bar n|$ joint Picard + self-consistent BCS gap
solves on the 1640-bin paper-resolution grid (including sub-gap guard cells);
total wall-time is on the order of an
hour. Opt in with ``pytest -m slow``.

The expensive sweep range (``N_BAR_VALUES``) is tunable in
:mod:`fig6_paper`; tighten it if this test starts to dominate the slow
suite. The pinned baseline is self-consistent against whichever range is
configured at generation time.

First-time generation::

    python -m validation.fischer_2023.fig6_paper
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    phonon_occupation_matrices_from_state,
)
from qpsim.constants import KB_UEV_PER_K
from qpsim.physics import calibrate_gap
from qpsim.physics.spectral import SpectralContext
from qpsim.solvers.newton_steady_state import (
    _weighted_number_backward_error,
    number_changing_gain_loss,
)

import validation.fischer_2023.fig5_paper as fig5_paper
import validation.fischer_2023.fig5_solve as fig5_solve
import validation.fischer_2023.fig6_paper as fig6_paper
import validation.fischer_2023.fig6_solve as fig6_solve
from validation.fischer_2023.fig6_paper import (
    FIG6_BASELINE_COLUMNS,
    T_BATH_VALUES,
    baseline_path,
    config_metadata,
    read_baseline,
    read_baseline_metadata,
    run,
)
from validation.fischer_2023.fig6_solve import (
    DELTA_0,
    DIRECT_GAP_BACKWARD_ERROR_LIMIT,
    FIG6_CERTIFICATE_FIELDS,
    FINITE_CUTOFF_DELTA0_OVER_KBTC,
    GAP_FIXED_POINT_ABS_TOL_UEV,
    N_BAR_VALUES,
    OMEGA_0,
    T_C,
    TARGET_BACKWARD_ERROR_LIMIT,
    _build_grid_and_spectral,
    _require_target_certificate,
)
from validation.fischer_2023.steady_state_certificate import (
    QP_NUMBER_CERTIFICATE_FIELD,
)


def test_console_diagnostic_literals_are_ascii() -> None:
    """An expensive Windows CLI run must not fail while printing results."""
    offenders: list[tuple[str, int, str]] = []
    for module in (fig5_solve, fig5_paper, fig6_solve, fig6_paper):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            console_roots: list[ast.AST] = []
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                console_roots.append(node)
            console_roots.extend(
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"description", "help"}
            )
            for root in console_roots:
                for child in ast.walk(root):
                    if not (
                        isinstance(child, ast.Constant)
                        and isinstance(child.value, str)
                    ):
                        continue
                    try:
                        child.value.encode("ascii")
                    except UnicodeEncodeError:
                        offenders.append(
                            (module.__name__, node.lineno, child.value)
                        )

    assert offenders == []


def test_direct_plot_limits_expose_signed_low_drive_point() -> None:
    x_limits, y_limits = fig6_paper._direct_plot_limits(
        np.array([0.159, 0.30, 0.70]),
        np.array([-0.0168, 0.10, 0.20]),
    )

    assert x_limits[0] < 0.159
    assert x_limits[1] > 0.70
    assert y_limits[0] < -0.0168
    assert y_limits[1] == 0.25


def test_programmatic_direct_generation_cannot_clobber_canonical(
    monkeypatch,
) -> None:
    reference = read_baseline(baseline_path())
    written: dict[str, object] = {}

    monkeypatch.setattr(fig6_paper, "run_cached", lambda **_kwargs: reference)

    def record_csv(_result, path=None):
        written["csv"] = path
        return path

    def record_pdf(_result, path=None, **kwargs):
        written["pdf"] = path
        written["direct"] = kwargs["direct_gap_observable"]
        return path

    monkeypatch.setattr(fig6_paper, "write_baseline", record_csv)
    monkeypatch.setattr(fig6_paper, "write_plot", record_pdf)

    csv_path, pdf_path = fig6_paper.generate_baseline(
        direct_gap_observable=True,
        fixed_gap_kinetics=True,
    )

    canonical = baseline_path()
    assert csv_path == fig6_paper.baseline_path(direct_gap_observable=True)
    assert pdf_path == fig6_paper.plot_path(direct_gap_observable=True)
    assert csv_path != canonical
    assert pdf_path != canonical.with_suffix(".pdf")
    assert written == {"csv": csv_path, "pdf": pdf_path, "direct": True}


def test_grid_covers_self_consistent_gap_support() -> None:
    """The kinetic grid must not begin at the gap it is allowed to suppress."""
    E, dE, spectral = _build_grid_and_spectral()
    first_edge = float(E[0] - 0.5 * dE[0])
    spacing = float(dE[0])

    assert first_edge <= DELTA_0 - OMEGA_0
    assert OMEGA_0 / spacing == pytest.approx(round(OMEGA_0 / spacing))
    assert not np.any(spectral.active_mask[DELTA_0 > E])


def test_finite_cutoff_calibration_anchors_delta0_and_tc() -> None:
    assert pytest.approx(
        1.7637398024450115,
        rel=0.0,
        abs=1e-15,
    ) == FINITE_CUTOFF_DELTA0_OVER_KBTC
    assert pytest.approx(1.184309192877208, rel=0.0, abs=1e-15) == T_C

    calibration = calibrate_gap(T_c=T_C, T_bath=0.0, xtol=1e-12)
    assert calibration.delta_0_bcs == pytest.approx(
        DELTA_0,
        rel=0.0,
        abs=5e-14,
    )
    assert calibration.delta_0_bcs / (KB_UEV_PER_K * T_C) == pytest.approx(
        FINITE_CUTOFF_DELTA0_OVER_KBTC,
        rel=0.0,
        abs=1e-15,
    )

    fingerprint = fig6_solve.solver_fingerprint()
    assert fingerprint["finite_cutoff_delta0_over_kbtc"] == pytest.approx(
        FINITE_CUTOFF_DELTA0_OVER_KBTC,
        rel=0.0,
        abs=0.0,
    )
    assert fingerprint["t_c"] == pytest.approx(T_C, rel=0.0, abs=0.0)
    assert fingerprint["gap_fixed_point_abs_tol_uev"] == pytest.approx(
        GAP_FIXED_POINT_ABS_TOL_UEV,
        rel=0.0,
        abs=0.0,
    )
    assert fingerprint["certificate_fields"] == list(FIG6_CERTIFICATE_FIELDS)


@pytest.mark.parametrize(
    ("T_bath", "expected_suppression_uev"),
    [
        (0.10, 8.321535460709129e-8),
        (0.15, 1.0737232068436242e-4),
        (0.20, 4.019584551002708e-3),
    ],
)
def test_finite_cutoff_thermal_suppressions_are_resolved(
    T_bath: float,
    expected_suppression_uev: float,
) -> None:
    calibration = calibrate_gap(
        T_c=T_C,
        T_bath=T_bath,
        Delta_0=DELTA_0,
        xtol=1e-12,
    )
    assert DELTA_0 - calibration.delta_eq == pytest.approx(
        expected_suppression_uev,
        rel=1e-6,
        abs=2e-12,
    )


def _assert_certified_baseline_balances(path, axes) -> None:
    """Require the current schema and gate every accepted certificate field."""
    text = path.read_text(encoding="utf-8")
    header = next(
        (line for line in text.splitlines() if line.startswith("T_bath_K,")),
        "",
    )
    assert tuple(header.split(",")) == FIG6_BASELINE_COLUMNS, (
        "certified Fig. 6 preflight requires the current 15-column schema"
    )
    accepted = np.isfinite(axes.x_qp_num)
    for field in FIG6_CERTIFICATE_FIELDS:
        values = getattr(axes, field)[accepted]
        assert np.all(np.isfinite(values)), (
            f"certified baseline has non-finite {field} at an accepted point"
        )
    for field in ("qp_backward_error", "phonon_backward_error"):
        values = getattr(axes, field)[accepted]
        assert np.all(values <= TARGET_BACKWARD_ERROR_LIMIT), (
            f"certified baseline {field} exceeds "
            f"{TARGET_BACKWARD_ERROR_LIMIT:g}: {values.tolist()}"
        )
    gap_errors = axes.gap_fixed_point_abs_error_uev[accepted]
    assert np.all(gap_errors <= GAP_FIXED_POINT_ABS_TOL_UEV), (
        "certified baseline gap-map error exceeds "
        f"{GAP_FIXED_POINT_ABS_TOL_UEV:g}: {gap_errors.tolist()}"
    )


def _assert_config_matches_baseline(path) -> None:
    """Cheap preflight (~1 s, no solve): the live module config must match the
    pinned baseline's stamped header + sweep axes.

    Gating the ~6.04 h serial :func:`run` behind this turns a stale config/baseline
    pairing — a ``TAU_L_MODEL`` swap, a grid change, a sweep-range edit — into
    a seconds-long failure instead of one discovered only after the full
    sweep. Compares the config fingerprint against the baseline header, and
    the configured sweep axes against the baseline's data rows.
    """
    cfg = config_metadata()
    meta = read_baseline_metadata(path)
    axes = read_baseline(path)

    assert cfg.tau_l_model == meta.tau_l_model, (
        f"TAU_L_MODEL config={cfg.tau_l_model!r} != baseline {meta.tau_l_model!r}; "
        "regenerate the baseline or restore the model before the slow run."
    )
    assert cfg.num_bins == meta.num_bins, (
        f"grid NE config={cfg.num_bins} != baseline {meta.num_bins}"
    )
    assert cfg.e_min_factor == pytest.approx(meta.e_min_factor)
    assert cfg.e_max_factor == pytest.approx(meta.e_max_factor)
    assert cfg.delta_0 == pytest.approx(meta.delta_0)
    assert cfg.finite_cutoff_delta0_over_kbtc == pytest.approx(
        meta.finite_cutoff_delta0_over_kbtc,
        rel=0.0,
        abs=1e-15,
    )
    assert cfg.tau_0 == pytest.approx(meta.tau_0)
    assert cfg.t_c == pytest.approx(meta.t_c, rel=0.0, abs=1e-15)
    assert cfg.omega_0 == pytest.approx(meta.omega_0)
    assert cfg.c_phot == pytest.approx(meta.c_phot)
    assert cfg.film_thickness_nm == pytest.approx(meta.film_thickness_nm)
    assert cfg.eta == pytest.approx(meta.eta)
    assert cfg.tau_0_pb_ns == pytest.approx(meta.tau_0_pb_ns, rel=1e-8)
    assert cfg.tau_l_ns == pytest.approx(meta.tau_l_ns, rel=1e-8)
    assert cfg.gap_fixed_point_abs_tol_uev == pytest.approx(
        meta.gap_fixed_point_abs_tol_uev,
        rel=0.0,
        abs=0.0,
    )
    assert cfg.certificate_metric_version == meta.certificate_metric_version
    np.testing.assert_allclose(
        np.asarray(T_BATH_VALUES, dtype=float), axes.T_bath,
        rtol=0.0, atol=1e-14,
        err_msg="T_bath sweep axis differs from baseline",
    )
    np.testing.assert_allclose(
        N_BAR_VALUES, axes.n_bar, rtol=1e-12, atol=0.0,
        err_msg="n_bar sweep axis (range/count) differs from baseline",
    )
    _assert_certified_baseline_balances(path, axes)


def test_config_matches_baseline_metadata() -> None:
    """Fast tripwire (not slow-marked): config fingerprint matches the pinned
    baseline header.

    This is the standing fast-suite guard that would have caught the τ_ℓ-model
    / baseline mismatch that once wasted 9.5 h. The slow
    ``test_matches_pinned_baseline`` re-runs the same check inline so the 6.04 h
    sweep is gated even when this fast test is not selected.
    """
    path = baseline_path()
    if not path.exists():
        pytest.skip(f"Baseline not found at {path}.")
    _assert_config_matches_baseline(path)


def test_converged_target_with_bad_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 2e-5,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": 0.0,
    }
    with pytest.raises(RuntimeError, match="converged target failed"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_converged_target_with_bad_gap_map_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 1e-8,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": 1.01 * GAP_FIXED_POINT_ABS_TOL_UEV,
    }
    with pytest.raises(RuntimeError, match="gap-map certificate"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_fixed_gap_certificate_allows_nan_gap_map_metric() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 1e-8,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 1e-8,
        "phonon_backward_error": 1e-8,
        "gap_fixed_point_abs_error_uev": float("nan"),
    }
    _require_target_certificate(
        certificate,
        T_bath=0.1,
        n_bar=1e6,
        require_gap_fixed_point=False,
    )


def test_converged_target_with_bad_number_certificate_hard_fails() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 0.0,
        "qp_number_backward_error": 0.6,
        "phonon_residual_inf": 1e-20,
        "phonon_raw_backward_error": 0.0,
        "phonon_backward_error": 0.0,
        "gap_fixed_point_abs_error_uev": 0.0,
    }
    with pytest.raises(RuntimeError, match="qp_number"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e6,
            require_gap_fixed_point=True,
        )


def test_direct_gap_certificate_uses_strict_certified_metrics() -> None:
    certificate = {
        "qp_residual_inf": 1e-20,
        "qp_backward_error": 0.5 * DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        "qp_number_backward_error": 0.0,
        "phonon_residual_inf": 1e-20,
        # Raw phonon balance includes irreducible affine-root rounding; direct
        # mode gates the representability-aware certified excess below.
        "phonon_raw_backward_error": 1e-6,
        "phonon_backward_error": 0.0,
        "gap_fixed_point_abs_error_uev": float("nan"),
    }
    _require_target_certificate(
        certificate,
        T_bath=0.1,
        n_bar=1e4,
        require_gap_fixed_point=False,
        backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
    )

    certificate["qp_backward_error"] = 2.0 * DIRECT_GAP_BACKWARD_ERROR_LIMIT
    with pytest.raises(RuntimeError, match="limit=1e-09"):
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e4,
            require_gap_fixed_point=False,
            backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        )


@pytest.mark.parametrize("obs", (-1.04, -0.0168056837102, 0.0, 1.0))
def test_signed_finite_gap_suppression_ratio_is_acceptable(obs: float) -> None:
    assert fig6_solve._acceptable_ratio(obs)


@pytest.mark.parametrize("obs", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_gap_suppression_ratio_is_rejected(obs: float) -> None:
    assert not fig6_solve._acceptable_ratio(obs)


def test_fixed_gap_solver_falls_back_to_picard(monkeypatch) -> None:
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    state = fig6_solve._build_state(material, spectral, 0.1)
    calls: list[str] = []

    def fail_coupled(*_args, **_kwargs):
        calls.append("coupled")
        raise fig6_solve.CoupledNewtonLineSearchError(
            iteration=1,
            residual_norm=0.5 * fig6_solve.PICARD_TOL,
        )

    def pass_picard(_backend, state_arg, _photon_params):
        calls.append("picard")
        return state_arg

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    monkeypatch.setattr(fig6_solve, "_solve_picard_fixed_gap", pass_picard)

    result = fig6_solve._solve_fixed_gap_kinetics(
        T3DiffusionBackend(),
        state,
        {"omega_0": OMEGA_0, "n_bar": 1e4, "c_phot": fig6_solve.C_PHOT},
    )

    assert result is state
    assert calls == ["coupled", "picard"]


@pytest.mark.parametrize(
    "error",
    (
        fig6_solve.CoupledNewtonLineSearchError(
            iteration=2,
            residual_norm=2.0 * fig6_solve.PICARD_TOL,
        ),
        RuntimeError("Coupled Newton Jacobian singular at iteration 2."),
    ),
)
def test_fixed_gap_solver_does_not_mask_non_roundoff_failure(
    monkeypatch,
    error: RuntimeError,
) -> None:
    def fail_coupled(*_args, **_kwargs):
        raise error

    def fail_if_picard_runs(*_args, **_kwargs):
        pytest.fail("Picard fallback must be reserved for a roundoff-level stall")

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    monkeypatch.setattr(
        fig6_solve,
        "_solve_picard_fixed_gap",
        fail_if_picard_runs,
    )

    with pytest.raises(type(error), match=str(error).split(" at iteration")[0]):
        fig6_solve._solve_fixed_gap_kinetics(
            T3DiffusionBackend(),
            object(),  # type: ignore[arg-type]
            None,
        )


def test_direct_point_solve_propagates_fixed_gap_failure(monkeypatch) -> None:
    """The public point path must not relabel fixed-gap errors as folds."""
    expected = RuntimeError("synthetic fixed-gap singular Jacobian")

    def fail_fixed_gap(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(fig6_solve, "_solve_fixed_gap_kinetics", fail_fixed_gap)
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    with pytest.raises(RuntimeError, match="singular Jacobian") as caught:
        fig6_solve._solve_and_measure(
            T3DiffusionBackend(),
            material,
            spectral,
            0.1,
            1e4,
            None,
            fixed_gap_kinetics=True,
            direct_gap_observable=True,
            thermal_integral=0.0,
            delta_eq=DELTA_0,
            delta_T=1.0,
        )

    assert caught.value is expected


def test_reduced_direct_gap_picard_fallback_is_repeatable_and_certified(
    monkeypatch,
) -> None:
    """Exercise the real strict fallback on a small commensurate grid."""
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "PICARD_TOL", 1e-9)

    def fail_coupled(*_args, **_kwargs):
        raise fig6_solve.CoupledNewtonLineSearchError(
            iteration=1,
            residual_norm=0.5 * fig6_solve.PICARD_TOL,
        )

    monkeypatch.setattr(
        fig6_solve,
        "_solve_coupled_newton_fixed_gap",
        fail_coupled,
    )
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    thermal_integral = fig6_solve.thermal_gap_integral_direct(
        spectral.E,
        gap=DELTA_0,
        T_bath=0.1,
        samples="centers",
    )
    delta_eq = DELTA_0 * float(np.exp(-thermal_integral))
    kwargs = {
        "fixed_gap_kinetics": True,
        "direct_gap_observable": True,
        "thermal_integral": thermal_integral,
        "delta_eq": delta_eq,
        "delta_T": DELTA_0 - delta_eq,
    }

    first = fig6_solve._solve_and_measure(
        T3DiffusionBackend(), material, spectral, 0.1, 1e4, None, **kwargs,
    )
    repeated = fig6_solve._solve_and_measure(
        T3DiffusionBackend(), material, spectral, 0.1, 1e4, None, **kwargs,
    )

    assert first[0].gap == DELTA_0
    assert first[1] == pytest.approx(repeated[1], rel=0.0, abs=1e-14)
    np.testing.assert_array_equal(first[0].f, repeated[0].f)
    np.testing.assert_array_equal(
        first[0].phonon.n_ph,
        repeated[0].phonon.n_ph,
    )
    for result in (first, repeated):
        certificate = result[4]
        _require_target_certificate(
            certificate,
            T_bath=0.1,
            n_bar=1e4,
            require_gap_fixed_point=False,
            backward_error_limit=DIRECT_GAP_BACKWARD_ERROR_LIMIT,
        )
        assert certificate["qp_backward_error"] < DIRECT_GAP_BACKWARD_ERROR_LIMIT
        # The number-mode amplitude polish can move the returned phonon
        # fixed point to the adjacent representable float. Its balance is
        # then roundoff-small rather than bitwise zero.
        assert certificate["phonon_backward_error"] < 1e-14


def test_legacy_nine_column_baseline_reads_with_nan_certificates(tmp_path) -> None:
    path = tmp_path / "legacy_fig6.csv"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n",
        encoding="utf-8",
    )

    result = read_baseline(path)
    for field in (
        "qp_residual_inf",
        "qp_backward_error",
        "phonon_residual_inf",
        "phonon_raw_backward_error",
        "phonon_backward_error",
        "gap_fixed_point_abs_error_uev",
    ):
        assert np.all(np.isnan(getattr(result, field)))


def test_baseline_reader_rejects_duplicate_coordinates(tmp_path) -> None:
    path = tmp_path / "duplicate_fig6.csv"
    row = "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS[:9])
        + "\n"
        + row
        + row,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"duplicate \(T_bath, n_bar\)"):
        read_baseline(path)


def test_baseline_reader_rejects_missing_cartesian_coordinate(tmp_path) -> None:
    path = tmp_path / "missing_coordinate_fig6.csv"
    path.write_text(
        "# legacy Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS[:9])
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
        "0.1,20000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n"
        "0.2,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="missing Cartesian"):
        read_baseline(path)


def test_old_thirteen_column_certificate_maps_without_reinterpretation(
    tmp_path,
) -> None:
    path = tmp_path / "old_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53,qp_residual_inf,qp_backward_error,"
        "phonon_residual_inf,phonon_backward_error\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "2e-5,1e-20,1e-8\n",
        encoding="utf-8",
    )
    result = read_baseline(path)

    assert result.phonon_backward_error[0, 0] == pytest.approx(1e-8)
    assert np.isnan(result.phonon_raw_backward_error[0, 0])
    assert np.isnan(result.gap_fixed_point_abs_error_uev[0, 0])

    with pytest.raises(AssertionError, match="current 15-column schema"):
        _assert_certified_baseline_balances(path, result)


def test_current_baseline_preflight_rejects_bad_balance(tmp_path) -> None:
    path = tmp_path / "bad_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS)
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "2e-5,1e-20,1e-8,1e-8,1e-12\n",
        encoding="utf-8",
    )
    result = read_baseline(path)

    with pytest.raises(AssertionError, match="qp_backward_error exceeds"):
        _assert_certified_baseline_balances(path, result)


def test_current_baseline_preflight_rejects_bad_gap_map_error(tmp_path) -> None:
    path = tmp_path / "bad_gap_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        "T_bath_K,n_bar,T_star_over_delta,delta_eq_T_bath_ueV,"
        "delta_driven_ueV,x_qp_num,x_qp_eq47,paper_observable_num,"
        "paper_observable_eq53,qp_residual_inf,qp_backward_error,"
        "phonon_residual_inf,phonon_raw_backward_error,"
        "phonon_backward_error,gap_fixed_point_abs_error_uev\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        f"1e-8,1e-20,1e-8,1e-8,{1.01 * GAP_FIXED_POINT_ABS_TOL_UEV}\n",
        encoding="utf-8",
    )
    result = read_baseline(path)

    with pytest.raises(AssertionError, match="gap-map error exceeds"):
        _assert_certified_baseline_balances(path, result)


def test_current_baseline_preflight_requires_finite_certificate_fields(
    tmp_path,
) -> None:
    path = tmp_path / "nonfinite_certified_fig6.csv"
    path.write_text(
        "# certified Fig. 6 baseline\n"
        "# tau_0_pb_ns=0.255 tau_l_ns=0.255\n"
        + ",".join(FIG6_BASELINE_COLUMNS)
        + "\n"
        "0.1,10000,0.1,179.9,179.8,1e-9,2e-9,0.1,0.2,1e-20,"
        "1e-8,1e-20,nan,1e-8,1e-12\n",
        encoding="utf-8",
    )
    result = read_baseline(path)

    with pytest.raises(AssertionError, match="non-finite phonon_raw_backward_error"):
        _assert_certified_baseline_balances(path, result)


def test_sweep_carries_full_state_within_row_and_resets_between_rows(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.10, 0.20))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4, 2.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    calls: list[tuple[float, float | None, float | None]] = []
    row_counts = {0.10: 0, 0.20: 0}

    def fake_solve_and_measure(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        continuation_seed,
        **_kwargs,
    ):
        seed_gap = None if continuation_seed is None else continuation_seed.gap
        seed_spectral_gap = (
            None if continuation_seed is None else continuation_seed.spectral.gap
        )
        calls.append((T_bath, seed_gap, seed_spectral_gap))
        state = fig6_solve._build_state(
            material_arg,
            spectral_arg,
            T_bath,
            continuation_seed=continuation_seed,
        )

        row_counts[T_bath] += 1
        target_gap = DELTA_0 - 0.25 * (1 + int(T_bath > 0.15)) - 0.01 * row_counts[
            T_bath
        ]
        converged = replace(
            state,
            gap=target_gap,
            spectral=SpectralContext(
                E_bins=state.spectral.E,
                dE_bins=state.spectral.dE,
                gap=target_gap,
            ),
        )
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        return converged, 0.1, target_gap, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        fake_solve_and_measure,
    )
    result = fig6_solve._solve_sweep(
        T3DiffusionBackend(),
        material,
        spectral,
        tau_0_pb=0.255,
    )

    assert calls[0] == (0.10, None, None)
    assert calls[1][0] == 0.10
    assert calls[1][1] == calls[1][2] == DELTA_0 - 0.26
    assert calls[2] == (0.20, None, None)
    assert calls[3][0] == 0.20
    assert calls[3][1] == calls[3][2] == DELTA_0 - 0.51

    certificates = result[-1]
    for field in FIG6_CERTIFICATE_FIELDS:
        np.testing.assert_array_equal(certificates[field], np.zeros((2, 2)))


def test_direct_gap_sweep_applies_strict_certificate_limit(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.10,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def fake_solve_and_measure(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        _continuation_seed,
        **_kwargs,
    ):
        state = fig6_solve._build_state(material_arg, spectral_arg, T_bath)
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        certificate["qp_backward_error"] = (
            2.0 * DIRECT_GAP_BACKWARD_ERROR_LIMIT
        )
        certificate["gap_fixed_point_abs_error_uev"] = float("nan")
        return state, 0.1, DELTA_0, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        fake_solve_and_measure,
    )

    with pytest.raises(RuntimeError, match="limit=1e-09"):
        fig6_solve._solve_sweep(
            T3DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
            direct_gap_observable=True,
            fixed_gap_kinetics=True,
        )


@pytest.mark.parametrize(
    ("direct_gap_observable", "fixed_gap_kinetics"),
    ((True, False), (False, True)),
)
def test_solve_rejects_inconsistent_numerical_modes_before_setup(
    monkeypatch,
    direct_gap_observable: bool,
    fixed_gap_kinetics: bool,
) -> None:
    def fail_if_setup_runs():
        pytest.fail("mode validation must precede the expensive grid setup")

    monkeypatch.setattr(fig6_solve, "_fischer_material", fail_if_setup_runs)

    with pytest.raises(ValueError, match="must be enabled together"):
        fig6_solve.solve(
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=fixed_gap_kinetics,
        )


def test_independent_certificate_runtime_error_propagates(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    monkeypatch.setattr(
        fig6_solve,
        "_solve_picard_sc_gap",
        lambda _backend, state, _photon_params: state,
    )

    def fail_certificate(*_args, **_kwargs):
        raise RuntimeError("independent certificate assembly failed")

    monkeypatch.setattr(
        fig6_solve,
        "steady_state_certificate",
        fail_certificate,
    )

    with pytest.raises(RuntimeError, match="independent certificate assembly failed"):
        fig6_solve._solve_sweep(
            T3DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
        )


@pytest.mark.parametrize("direct_gap_observable", (False, True))
def test_nonfinite_derived_observable_propagates_through_sweep(
    monkeypatch,
    direct_gap_observable: bool,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def nonfinite_measurement(
        _backend,
        material_arg,
        spectral_arg,
        T_bath,
        _n_bar,
        _continuation_seed,
        **_kwargs,
    ):
        state = fig6_solve._build_state(material_arg, spectral_arg, T_bath)
        certificate = dict.fromkeys(FIG6_CERTIFICATE_FIELDS, 0.0)
        certificate[QP_NUMBER_CERTIFICATE_FIELD] = 0.0
        if direct_gap_observable:
            certificate["gap_fixed_point_abs_error_uev"] = float("nan")
        return state, float("nan"), DELTA_0, 1e-8, certificate

    monkeypatch.setattr(
        fig6_solve,
        "_solve_and_measure",
        nonfinite_measurement,
    )

    with pytest.raises(RuntimeError, match="non-finite derived measurement"):
        fig6_solve._solve_sweep(
            T3DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
            direct_gap_observable=direct_gap_observable,
            fixed_gap_kinetics=direct_gap_observable,
        )


def test_self_consistent_inner_solver_failure_propagates_through_sweep(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    expected = RuntimeError("synthetic inner Picard exhaustion")

    def fail_inner_solver(*_args, **_kwargs):
        raise expected

    monkeypatch.setattr(fig6_solve, "_solve_picard_sc_gap", fail_inner_solver)

    with pytest.raises(RuntimeError, match="inner Picard exhaustion") as caught:
        fig6_solve._solve_sweep(
            T3DiffusionBackend(),
            material,
            spectral,
            tau_0_pb=0.255,
        )

    assert caught.value is expected


def test_explicit_self_consistent_collapse_is_recorded_as_nan(monkeypatch) -> None:
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "T_BATH_VALUES", (0.20,))
    monkeypatch.setattr(fig6_solve, "N_BAR_VALUES", np.array([1.0e4]))
    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()

    def collapse(*_args, **_kwargs):
        raise fig6_solve.SelfConsistentGapCollapseError(
            iteration=4,
            max_occupation=0.25,
        )

    monkeypatch.setattr(fig6_solve, "_solve_picard_sc_gap", collapse)
    result = fig6_solve._solve_sweep(
        T3DiffusionBackend(),
        material,
        spectral,
        tau_0_pb=0.255,
    )

    assert np.isnan(result[2][0, 0])
    assert np.isnan(result[3][0, 0])
    assert np.isnan(result[5][0, 0])


@pytest.mark.slow
def test_reduced_full_state_continuation_is_certified_and_repeatable(
    monkeypatch,
) -> None:
    """A real reduced full-state continuation is deterministic and certified."""
    monkeypatch.setattr(fig6_solve, "NUM_BINS", 82)
    monkeypatch.setattr(fig6_solve, "PICARD_TOL", 1e-9)

    material = fig6_solve._fischer_material()
    _, _, spectral = fig6_solve._build_grid_and_spectral()
    calibration = calibrate_gap(
        T_c=T_C,
        T_bath=0.20,
        Delta_0=DELTA_0,
        xtol=fig6_solve.GAP_SOLVE_XTOL_UEV,
    )
    delta_eq = calibration.delta_eq
    delta_T = DELTA_0 - delta_eq
    solve_kwargs = {
        "fixed_gap_kinetics": False,
        "direct_gap_observable": False,
        "thermal_integral": None,
        "delta_eq": delta_eq,
        "delta_T": delta_T,
    }
    backend = T3DiffusionBackend()
    first = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e4,
        None,
        **solve_kwargs,
    )
    continued = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e5,
        first[0],
        **solve_kwargs,
    )
    repeated_first = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e4,
        None,
        **solve_kwargs,
    )
    repeated = fig6_solve._solve_and_measure(
        backend,
        material,
        spectral,
        0.20,
        1.0e5,
        repeated_first[0],
        **solve_kwargs,
    )

    assert first[0].gap == first[0].spectral.gap
    assert first[0].gap < DELTA_0
    # Round-6 Newton number-mode certification moved this reduced-grid
    # endpoint off the former aggregate-only root.  The old pin had an
    # independently reassembled pair-number backward error of 7.45e-6,
    # above NEWTON_BACKWARD_ERROR_TOL=1e-6; this root is deterministic and
    # improves that error to 4.50e-7.  Keep the original tight pin tolerances.
    assert continued[0].gap == pytest.approx(
        179.9969260485,
        rel=0.0,
        abs=2e-9,
    )
    assert continued[1] == pytest.approx(0.23525642, rel=0.0, abs=1e-6)
    assert continued[0].gap == pytest.approx(
        repeated[0].gap,
        rel=0.0,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        continued[0].f,
        repeated[0].f,
        rtol=0.0,
        atol=1e-14,
    )
    assert continued[1] == pytest.approx(repeated[1], rel=0.0, abs=1e-10)
    for result in (continued, repeated):
        state = result[0]
        certificate = result[4]
        assert certificate["qp_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT
        assert certificate["phonon_backward_error"] <= TARGET_BACKWARD_ERROR_LIMIT
        assert (
            certificate["gap_fixed_point_abs_error_uev"]
            <= GAP_FIXED_POINT_ABS_TOL_UEV
        )

        # Independently reassemble the number-changing pair channel from the
        # returned joint (f, n_ph) snapshot.  This is the decisive Round-6
        # gate that invalidated the old absolute pin; aggregate QP turnover
        # alone is dominated by number-conserving scattering.
        K_r0 = build_recombination_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        _, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
            state.spectral.E
        )
        _, N_emit, N_abs = phonon_occupation_matrices_from_state(
            state.phonon.n_ph[0, :, 0], idx_diff, idx_sum, diff_sign
        )
        gain_number, loss_number, configured = number_changing_gain_loss(
            state.f,
            state.spectral,
            K_r0,
            state.T_bath,
            N_emit=N_emit,
            N_abs=N_abs,
        )
        number_error = _weighted_number_backward_error(
            gain_number,
            loss_number,
            state.f,
            state.spectral.cell_weights,
            state.spectral.active_mask,
        )
        assert configured
        assert number_error is not None
        assert number_error <= fig6_solve.NEWTON_BACKWARD_ERROR_TOL


@pytest.mark.slow
@pytest.mark.manual_slow
def test_matches_pinned_baseline() -> None:
    """Run the full 1640-bin paper-resolution sweep (manual validation only).

    The source itself documents a roughly 6.04-hour serial runtime, which is not a
    bounded pull-request check. Keep this test executable with
    ``pytest -m 'slow and manual_slow'`` while the author-style fixed-gap,
    direct-Delta[f] path is evaluated as the replacement CI target.
    """
    path = baseline_path()
    if not path.exists():
        pytest.skip(
            f"Baseline not found at {path}. "
            "Generate it with: python -m validation.fischer_2023.fig6_paper"
        )

    # Cheap preflight first (~1 s): reject a stale config/baseline pairing
    # before the ~6.04 h serial run() below, instead of after it.
    _assert_config_matches_baseline(path)

    baseline = read_baseline(path)
    result = run()

    # τ values --- 1e-8 relative per the pattern in test_fig5_paper.py.
    assert result.tau_0_pb_ns == pytest.approx(baseline.tau_0_pb_ns, rel=1e-8)
    assert result.tau_l_ns == pytest.approx(baseline.tau_l_ns, rel=1e-8)

    # Sweep axes match exactly (T_B values are literal floats; n̄ values
    # are np.logspace, so allow a tiny relative slack).
    np.testing.assert_allclose(
        result.T_bath, baseline.T_bath, rtol=0.0, atol=1e-14,
    )
    np.testing.assert_allclose(
        result.n_bar, baseline.n_bar, rtol=1e-12, atol=0.0,
    )

    # T_*/Δ is closed-form in n̄ — should be reproducible to ~1e-10.
    np.testing.assert_allclose(
        result.T_star_over_delta, baseline.T_star_over_delta,
        rtol=1e-10, atol=0.0,
        err_msg="T_*/Δ axis drift (Eq. 35)",
    )

    # Numerical observables — 1e-6 abs (NFP §6.4.1 iterative-mode tol).
    np.testing.assert_allclose(
        result.delta_eq, baseline.delta_eq, rtol=0.0, atol=1e-6,
        err_msg="Δ_eq(T_B) drift",
    )
    np.testing.assert_allclose(
        result.delta_driven, baseline.delta_driven, rtol=0.0, atol=1e-6,
        err_msg="Δ_driven (self-consistent BCS) drift",
    )
    np.testing.assert_allclose(
        result.x_qp_num, baseline.x_qp_num, rtol=1e-3, atol=1e-14,
        err_msg="numerical x_qp drift",
    )
    np.testing.assert_allclose(
        result.paper_observable_num, baseline.paper_observable_num,
        rtol=0.0, atol=1e-6,
        err_msg="(δΔ_T - δΔ)/δΔ_T numerical drift",
    )
    # x_qp_eq47 is pure closed-form (Eq. 47 + Appendix-E in T_bath, n_bar,
    # τ_ℓ, τ_0^PB). Pin tightly — drift means a coefficient changed.
    np.testing.assert_allclose(
        result.x_qp_eq47, baseline.x_qp_eq47, rtol=1e-10, atol=0.0,
        err_msg="Eq. 47 analytic x_qp drift",
    )
    # Dashed overlay is Eq. 53 evaluated at (x_qp_eq47, T_*/Δ) and combined
    # with the numerical Δ_eq(T_B). It inherits the Δ_eq tolerance (1e-6 abs
    # in μeV) via composition; the closed-form Eq. 47 + Eq. 53 part itself
    # is float64-exact.
    np.testing.assert_allclose(
        result.paper_observable_eq53, baseline.paper_observable_eq53,
        rtol=0.0, atol=1e-6,
        err_msg="Eq. 53 dashed-overlay drift",
    )


class TestFig6CacheIntegration:
    """The cached regen path (:func:`run_cached`) wraps the same solve/observables
    split and serves the (otherwise ~6.04 h serial) solve from disk. The expensive solve is
    stubbed so the test is fast; it exercises the real cache + observables (pure
    unpack) wiring. Engine-level key/store properties are covered in
    ``tests/validation/test_sweep_cache.py``.
    """

    def _stub_payload(self) -> dict:
        # Synthetic raw payload (fig6's observables is a pure unpack — no grid
        # rebuild — so tiny placeholder arrays suffice). One T_B, two n̄.
        return {
            "tau_0_pb_ns": np.array([0.255]),
            "tau_l_ns": np.array([0.255]),
            "T_bath": np.array([0.20]),
            "n_bar": np.array([1.0e4, 1.0e5]),
            "T_star_over_delta": np.array([[0.10, 0.20]]),
            "delta_eq": np.array([179.9]),
            "delta_driven": np.array([[179.8, 179.5]]),
            "paper_observable_num": np.array([[0.10, 0.20]]),
            "paper_observable_eq53": np.array([[0.11, 0.21]]),
            "x_qp_num": np.array([[1.0e-5, 2.0e-5]]),
            "x_qp_eq47": np.array([[1.1e-5, 2.1e-5]]),
            "qp_residual_inf": np.array([[1.0e-15, 2.0e-15]]),
            "qp_backward_error": np.array([[1.0e-8, 2.0e-8]]),
            "phonon_residual_inf": np.array([[3.0e-15, 4.0e-15]]),
            "phonon_raw_backward_error": np.array([[3.1e-8, 4.1e-8]]),
            "phonon_backward_error": np.array([[3.0e-8, 4.0e-8]]),
            "gap_fixed_point_abs_error_uev": np.array([[1.0e-12, 2.0e-12]]),
        }

    def test_run_cached_hits_disk_on_second_call(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig6_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "1")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        r1 = fp.run_cached()
        assert calls["n"] == 1  # cache miss -> solve ran once

        r2 = fp.run_cached()
        assert calls["n"] == 1  # cache hit -> solve NOT re-run

        ref = fp.observables(payload)
        for res in (r1, r2):
            for fld in ("paper_observable_num", "paper_observable_eq53",
                        "delta_driven", "x_qp_num", "x_qp_eq47",
                         "qp_residual_inf", "qp_backward_error",
                         "phonon_residual_inf", "phonon_raw_backward_error",
                         "phonon_backward_error",
                         "gap_fixed_point_abs_error_uev"):
                np.testing.assert_array_equal(getattr(res, fld), getattr(ref, fld))

    def test_certified_payload_csv_round_trip(self, tmp_path) -> None:
        import validation.fischer_2023.fig6_paper as fp

        reference = fp.observables(self._stub_payload())
        path = fp.write_baseline(reference, tmp_path / "certified_fig6.csv")
        restored = fp.read_baseline(path)

        payload = path.read_bytes()
        assert b"\r\n" not in payload
        assert payload.endswith(b"\n")

        for field in (
            "qp_residual_inf",
            "qp_backward_error",
            "phonon_residual_inf",
            "phonon_raw_backward_error",
            "phonon_backward_error",
            "gap_fixed_point_abs_error_uev",
        ):
            np.testing.assert_array_equal(
                getattr(restored, field),
                getattr(reference, field),
            )

        metadata = fp.read_baseline_metadata(path)
        assert metadata.finite_cutoff_delta0_over_kbtc == pytest.approx(
            FINITE_CUTOFF_DELTA0_OVER_KBTC,
            rel=0.0,
            abs=1e-15,
        )
        assert metadata.t_c == pytest.approx(T_C, rel=0.0, abs=1e-15)
        assert metadata.gap_fixed_point_abs_tol_uev == pytest.approx(
            GAP_FIXED_POINT_ABS_TOL_UEV,
            rel=0.0,
            abs=0.0,
        )
        assert (
            metadata.certificate_metric_version
            == fp.certificate_module.CERTIFICATE_METRIC_VERSION
        )

    def test_baseline_write_is_atomic_on_failure(self, tmp_path) -> None:
        import validation.fischer_2023.fig6_paper as fp

        path = tmp_path / "certified_fig6.csv"
        path.write_text("previous-good-baseline\n", encoding="utf-8")
        reference = fp.observables(self._stub_payload())
        malformed = replace(
            reference,
            T_star_over_delta=np.empty((0, 0)),
        )

        with pytest.raises(IndexError):
            fp.write_baseline(malformed, path)

        assert path.read_text(encoding="utf-8") == "previous-good-baseline\n"
        assert not path.with_name(f".{path.name}.{fp.os.getpid()}.tmp").exists()

    def test_run_cached_disabled_always_recomputes(self, tmp_path, monkeypatch) -> None:
        import validation.fischer_2023.fig6_paper as fp

        monkeypatch.setenv("QPSIM_SWEEP_CACHE", "0")
        monkeypatch.setenv("QPSIM_SWEEP_CACHE_DIR", str(tmp_path))

        calls = {"n": 0}
        payload = self._stub_payload()

        def stub_solve(**kwargs):
            calls["n"] += 1
            return {k: v.copy() for k, v in payload.items()}

        monkeypatch.setattr(fp, "solve", stub_solve)

        fp.run_cached()
        fp.run_cached()
        assert calls["n"] == 2  # disabled -> recompute each call
