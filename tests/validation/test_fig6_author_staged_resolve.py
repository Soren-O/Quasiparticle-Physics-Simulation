from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from validation.fischer_2023.fig6_author_staged_resolve import (
    AUTHOR_CONTROL,
    C3A,
    C3B,
    C3C,
    PAIR_LABELS_ONLY,
    PRIMARY_STAGES,
    AuthorSolveParameters,
    build_staged_operator,
    evaluate_staged_system,
    solve_staged_system,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    evaluate_author_system,
    solve_author_system,
)
from validation.reference_models.fischer_2023.fig6_author_operators import (
    AuthorNumericalConstants,
    evaluate_frozen_author_operators,
)

_AUTHENTICATED_JACOBIAN_ORACLE = {
    "source_sha256s": {
        "PhysApplPaper_Figure_6/quasiparticle_solver.py": (
            "bb6cd6fa8ff7d2ce11d96cbe79bbc64dea7226c0a814e6d0c4aa8c02cb3d2c61"
        ),
        "PhysApplPaper_Figure_6/quasiparticle_and_phonon_solver.py": (
            "cf7664866b8bd9abfaf7f008353e5d7d4d36dc371407df4024341d8f56f412dc"
        ),
    },
    # Extracted by executing the two source files above at N=9.  The state
    # and scalar inputs are constructed in _authenticated_oracle_parameters.
    "photon_diagonal": [
        -2311.074320324702,
        -7186.214555695313,
        -3873.0967049675505,
        -3327.031447220755,
        -3080.9629289550307,
        -2944.959361093889,
        -2860.9103036506367,
        -2805.024440394422,
        -1577.5613701649768,
    ],
    "photon_upper": [
        3021.849828383107,
        2201.814932333735,
        1903.3409473653658,
        1759.2419627678375,
        1678.1455860092956,
        1627.8070439396781,
        1594.3439379326894,
        1570.939023506403,
    ],
    "photon_lower": [
        4208.154156783821,
        1848.859456259762,
        1515.5109533249288,
        1374.7492399132286,
        1300.1745456625656,
        1255.4639465206158,
        1226.4093362900037,
        1206.4119734933715,
    ],
    "phonon_diagonal": [
        -3921571026.219335,
        -3921573069.9530597,
        -3921574603.95275,
        -3990819188.298673,
        -3984949915.6189337,
        -3988605392.9480476,
        -3993503523.0467205,
        -3998945403.7296834,
    ],
}


def _parameters(*, max_steps: int = 3) -> AuthorSolveParameters:
    return AuthorSolveParameters(
        gap_eV=2.0e-6,
        h_eV=1.0e-6,
        temperature_K=0.2,
        T_c_K=1.184,
        tau_0_s=438e-9,
        tau_0_pb_s=255e-12,
        tau_l_s=255e-12,
        photon_bin=1,
        n_bar=2.3,
        c_photon_s_inv=1.0e5,
        delta0_eV=2.0e-6,
        thermal_gap_eV=1.9e-6,
        max_newton_steps=max_steps,
        relative_step_threshold=1.0e-7,
        constants=AuthorNumericalConstants(),
    )


def _authenticated_oracle_parameters() -> AuthorSolveParameters:
    return AuthorSolveParameters(
        gap_eV=2.0e-6,
        h_eV=1.0e-6,
        temperature_K=0.2,
        T_c_K=1.184,
        tau_0_s=438e-9,
        tau_0_pb_s=2.2980241805523498e-08,
        tau_l_s=255e-12,
        photon_bin=1,
        n_bar=3.25,
        c_photon_s_inv=347.41225225493395,
        delta0_eV=2.0e-6,
        thermal_gap_eV=2.0e-6,
        max_newton_steps=3,
        relative_step_threshold=1.0e-7,
        constants=AuthorNumericalConstants(),
    )


def _state(size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260729)
    return (
        rng.uniform(0.01, 0.15, size),
        rng.uniform(0.01, 0.30, size - 1),
    )


def test_author_control_residual_matches_cleanroom_operator() -> None:
    params = _parameters()
    E = params.gap_eV + params.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    operator = build_staged_operator(E, params, AUTHOR_CONTROL)

    staged = evaluate_staged_system(
        operator,
        f,
        n_phonon,
        build_update_matrix=False,
    )
    frozen = evaluate_frozen_author_operators(
        E,
        f,
        n_phonon,
        gap_eV=params.gap_eV,
        h_eV=params.h_eV,
        temperature_K=params.temperature_K,
        T_c_K=params.T_c_K,
        tau_0_s=params.tau_0_s,
        tau_0_pb_s=params.tau_0_pb_s,
        tau_l_s=params.tau_l_s,
        photon_bin=params.photon_bin,
        n_bar=params.n_bar,
        c_photon_s_inv=params.c_photon_s_inv,
        constants=params.constants,
    )

    np.testing.assert_allclose(staged.qp_photon_s_inv, frozen.qp_photon_s_inv)
    np.testing.assert_allclose(
        staged.qp_scattering_s_inv,
        frozen.qp_scattering_s_inv,
    )
    np.testing.assert_allclose(staged.qp_pair_s_inv, frozen.qp_pair_s_inv)
    np.testing.assert_allclose(
        staged.phonon_scattering_s_inv,
        frozen.phonon_scattering_s_inv,
    )
    np.testing.assert_allclose(staged.phonon_pair_s_inv, frozen.phonon_pair_s_inv)
    np.testing.assert_allclose(
        staged.phonon_escape_s_inv,
        frozen.phonon_escape_s_inv,
    )


def test_stages_change_only_named_conventions_at_fixed_state() -> None:
    params = _parameters()
    E = params.gap_eV + params.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    evaluations = {
        spec.stage_id: evaluate_staged_system(
            build_staged_operator(E, params, spec),
            f,
            n_phonon,
            build_update_matrix=False,
        )
        for spec in PRIMARY_STAGES
    }
    control = evaluations[AUTHOR_CONTROL.stage_id]
    pair_only = evaluations[PAIR_LABELS_ONLY.stage_id]
    c3a = evaluations[C3A.stage_id]
    c3b = evaluations[C3B.stage_id]
    c3c = evaluations[C3C.stage_id]

    for candidate in (pair_only, c3a, c3b, c3c):
        np.testing.assert_array_equal(
            candidate.phonon_escape_s_inv,
            control.phonon_escape_s_inv,
        )
    for candidate in (pair_only,):
        np.testing.assert_array_equal(
            candidate.qp_photon_s_inv,
            control.qp_photon_s_inv,
        )
        np.testing.assert_array_equal(
            candidate.qp_scattering_s_inv,
            control.qp_scattering_s_inv,
        )
        np.testing.assert_array_equal(
            candidate.phonon_scattering_s_inv,
            control.phonon_scattering_s_inv,
        )
    for left, right in (
        (c3a.qp_photon_s_inv, c3b.qp_photon_s_inv),
        (c3a.qp_scattering_s_inv, c3b.qp_scattering_s_inv),
        (c3a.phonon_scattering_s_inv, c3b.phonon_scattering_s_inv),
    ):
        np.testing.assert_array_equal(left, right)

    assert not np.array_equal(pair_only.qp_pair_s_inv, control.qp_pair_s_inv)
    assert not np.array_equal(
        pair_only.phonon_pair_s_inv,
        control.phonon_pair_s_inv,
    )
    assert not np.array_equal(c3a.qp_scattering_s_inv, control.qp_scattering_s_inv)
    assert not np.array_equal(c3a.qp_pair_s_inv, control.qp_pair_s_inv)
    assert not np.array_equal(c3b.qp_pair_s_inv, c3a.qp_pair_s_inv)
    assert not np.array_equal(c3b.phonon_pair_s_inv, c3a.phonon_pair_s_inv)
    assert not np.array_equal(c3c.qp_photon_s_inv, c3b.qp_photon_s_inv)
    assert not np.array_equal(c3c.qp_scattering_s_inv, c3b.qp_scattering_s_inv)
    assert not np.array_equal(c3c.qp_pair_s_inv, c3b.qp_pair_s_inv)
    assert not np.array_equal(
        c3c.phonon_scattering_s_inv,
        c3b.phonon_scattering_s_inv,
    )
    assert not np.array_equal(c3c.phonon_pair_s_inv, c3b.phonon_pair_s_inv)


def test_c3c_changes_only_native_density_arithmetic() -> None:
    params = _parameters()
    E = params.gap_eV + params.h_eV * np.arange(9)
    c3b = build_staged_operator(E, params, C3B)
    c3c = build_staged_operator(E, params, C3C)

    np.testing.assert_array_equal(c3c.K_plus, c3b.K_plus)
    np.testing.assert_array_equal(c3c.K_minus, c3b.K_minus)
    assert (
        c3c.author_operator.pair_frequency_offset_bins
        == c3b.author_operator.pair_frequency_offset_bins
        == 1
    )
    assert not np.array_equal(c3c.rho, c3b.rho)
    signed_relative_first_cell = (c3c.rho[0] - c3b.rho[0]) / c3b.rho[0]
    assert 0.0 < abs(signed_relative_first_cell) < 2e-7


def test_update_matrix_matches_authenticated_executable_oracle() -> None:
    source_manifest = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "validation"
            / "paper_data"
            / "fischer_2023"
            / "fig6"
            / "author-source.json"
        ).read_text(encoding="utf-8")
    )
    member_hashes = {
        member["path"]: member["sha256"] for member in source_manifest["members"]
    }
    assert {
        path: member_hashes[path]
        for path in _AUTHENTICATED_JACOBIAN_ORACLE["source_sha256s"]
    } == _AUTHENTICATED_JACOBIAN_ORACLE["source_sha256s"]

    params = _authenticated_oracle_parameters()
    E = params.gap_eV + params.h_eV * np.arange(9)
    f = np.linspace(2.2e-4, 4.1e-5, E.size)
    n_phonon = np.linspace(3.3e-3, 2.2e-5, E.size - 1)
    evaluated = evaluate_staged_system(
        build_staged_operator(E, params, AUTHOR_CONTROL),
        f,
        n_phonon,
        build_update_matrix=True,
    )
    without_photons = evaluate_staged_system(
        build_staged_operator(
            E,
            replace(params, c_photon_s_inv=0.0),
            AUTHOR_CONTROL,
        ),
        f,
        n_phonon,
        build_update_matrix=True,
    )
    assert evaluated.update_matrix_s_inv is not None
    assert without_photons.update_matrix_s_inv is not None

    photon = (
        evaluated.update_matrix_s_inv[: E.size, : E.size]
        - without_photons.update_matrix_s_inv[: E.size, : E.size]
    )
    np.testing.assert_allclose(
        np.diag(photon),
        _AUTHENTICATED_JACOBIAN_ORACLE["photon_diagonal"],
        rtol=2.0e-15,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.diag(photon, 1),
        _AUTHENTICATED_JACOBIAN_ORACLE["photon_upper"],
        rtol=2.0e-15,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.diag(photon, -1),
        _AUTHENTICATED_JACOBIAN_ORACLE["photon_lower"],
        rtol=2.0e-15,
        atol=1.0e-10,
    )
    photon_tridiagonal = np.zeros_like(photon)
    np.fill_diagonal(photon_tridiagonal, np.diag(photon))
    rows = np.arange(E.size - 1)
    photon_tridiagonal[rows, rows + 1] = np.diag(photon, 1)
    photon_tridiagonal[rows + 1, rows] = np.diag(photon, -1)
    np.testing.assert_array_equal(photon, photon_tridiagonal)

    np.testing.assert_allclose(
        np.diag(evaluated.update_matrix_s_inv[E.size :, E.size :]),
        _AUTHENTICATED_JACOBIAN_ORACLE["phonon_diagonal"],
        rtol=2.0e-15,
        atol=1.0e-5,
    )
    np.testing.assert_array_equal(
        evaluated.update_matrix_s_inv[E.size :, E.size :]
        - np.diag(np.diag(evaluated.update_matrix_s_inv[E.size :, E.size :])),
        0.0,
    )


def test_update_matrix_preserves_authenticated_derivative_mismatches() -> None:
    params = _parameters()
    E = params.gap_eV + params.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    state = np.concatenate((f, n_phonon))
    for specification in PRIMARY_STAGES:
        operator = build_staged_operator(E, params, specification)
        evaluated = evaluate_staged_system(
            operator,
            f,
            n_phonon,
            build_update_matrix=True,
        )
        assert evaluated.update_matrix_s_inv is not None
        numerical = np.empty_like(evaluated.update_matrix_s_inv)
        for column in range(state.size):
            epsilon = 1.0e-7 * max(1.0, abs(float(state[column])))
            plus = state.copy()
            minus = state.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            residual_plus = evaluate_staged_system(
                operator,
                plus[: E.size],
                plus[E.size :],
                build_update_matrix=False,
            ).residual_s_inv
            residual_minus = evaluate_staged_system(
                operator,
                minus[: E.size],
                minus[E.size :],
                build_update_matrix=False,
            ).residual_s_inv
            numerical[:, column] = (
                residual_plus - residual_minus
            ) / (2.0 * epsilon)

        exact_entries = np.ones_like(numerical, dtype=bool)
        endpoint_rows = np.array([E.size - 1 - params.photon_bin, E.size - 1])
        exact_entries[endpoint_rows, :] = False
        photon_rows = np.arange(E.size - params.photon_bin)
        exact_entries[photon_rows, photon_rows + params.photon_bin] = False
        photon_rows = np.arange(params.photon_bin, E.size)
        exact_entries[photon_rows, photon_rows - params.photon_bin] = False
        for level in range(1, E.size):
            pair_sum = (
                level
                - 2 * operator.a_delta
                - specification.pair_frequency_offset_bins
            )
            if pair_sum >= 0:
                exact_entries[E.size + level - 1, E.size + level - 1] = False

        scale = float(np.max(np.abs(numerical[exact_entries])))
        error = float(
            np.max(
                np.abs(
                    evaluated.update_matrix_s_inv[exact_entries]
                    - numerical[exact_entries]
                )
            )
        )
        assert error / scale < 2.0e-8
        assert float(
            np.max(
                np.abs(
                    evaluated.update_matrix_s_inv[~exact_entries]
                    - numerical[~exact_entries]
                )
            )
        ) > 1.0e3


def test_each_primary_solve_retains_its_supplied_initial_state() -> None:
    params = _parameters(max_steps=1)
    E = params.gap_eV + params.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    initial = np.concatenate((f, n_phonon))

    for specification in PRIMARY_STAGES:
        result = solve_staged_system(
            build_staged_operator(E, params, specification),
            initial,
            solve_path="test_same_seed",
        )
        np.testing.assert_array_equal(result.state_history[0], initial)
        assert result.state_history.shape == (2, initial.size)
        assert result.newton_deltas.shape == (1, initial.size)
        np.testing.assert_array_equal(
            result.state_history[1],
            result.state_history[0] + result.newton_deltas[0],
        )


def test_staged_wrapper_is_numerically_transparent_to_pure_core() -> None:
    params = _parameters(max_steps=1)
    E = params.gap_eV + params.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    initial = np.concatenate((f, n_phonon))
    staged_operator = build_staged_operator(E, params, AUTHOR_CONTROL)

    wrapped_evaluation = evaluate_staged_system(
        staged_operator,
        f,
        n_phonon,
        build_update_matrix=True,
    )
    core_evaluation = evaluate_author_system(
        staged_operator.author_operator,
        f,
        n_phonon,
        build_update_matrix=True,
    )
    np.testing.assert_array_equal(
        wrapped_evaluation.residual_s_inv,
        core_evaluation.residual_s_inv,
    )
    np.testing.assert_array_equal(
        wrapped_evaluation.update_matrix_s_inv,
        core_evaluation.update_matrix_s_inv,
    )

    wrapped_result = solve_staged_system(
        staged_operator,
        initial,
        solve_path="wrapper_parity",
    )
    core_result = solve_author_system(staged_operator.author_operator, initial)
    np.testing.assert_array_equal(
        wrapped_result.state_history,
        core_result.state_history,
    )
    np.testing.assert_array_equal(
        wrapped_result.newton_deltas,
        core_result.newton_deltas,
    )
