from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    author_direct_gap,
    author_direct_gap_integral,
    author_fermi_occupation,
    build_author_operator,
    evaluate_author_system,
    solve_author_system,
)


def _parameters(*, max_steps: int = 1) -> AuthorSolveParameters:
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
        relative_step_threshold=1.0e-20,
        constants=AuthorNumericalConstants(),
    )


def _state(size: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260729)
    return (
        rng.uniform(0.01, 0.15, size),
        rng.uniform(0.01, 0.30, size - 1),
    )


def test_numerical_core_has_no_qpsim_import() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "validation"
        / "reference_models"
        / "fischer_2023"
        / "fig6_author_c0.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(name == "qpsim" or name.startswith("qpsim.") for name in imported)


def test_gain_loss_evidence_reassembles_source_order_channels() -> None:
    parameters = _parameters()
    E = parameters.gap_eV + parameters.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    evaluated = evaluate_author_system(
        build_author_operator(E, parameters),
        f,
        n_phonon,
        build_update_matrix=False,
    )
    channels = (
        evaluated.qp_photon,
        evaluated.qp_scattering,
        evaluated.qp_pair,
        evaluated.phonon_scattering,
        evaluated.phonon_pair,
        evaluated.phonon_escape,
    )
    for channel in channels:
        assert np.all(channel.gain_s_inv >= 0.0)
        assert np.all(channel.loss_s_inv >= 0.0)
        np.testing.assert_allclose(
            channel.gain_s_inv - channel.loss_s_inv,
            channel.net_s_inv,
            rtol=3.0e-15,
            atol=1.0e-8,
        )


def test_newton_result_retains_exact_transition_delta() -> None:
    parameters = _parameters(max_steps=1)
    E = parameters.gap_eV + parameters.h_eV * np.arange(9)
    f, n_phonon = _state(E.size)
    initial = np.concatenate((f, n_phonon))
    operator = build_author_operator(E, parameters)
    result = solve_author_system(operator, initial)

    assert result.newton_deltas.shape == (1, initial.size)
    np.testing.assert_array_equal(
        result.state_history[1],
        result.state_history[0] + result.newton_deltas[0],
    )
    evaluated = evaluate_author_system(
        operator,
        initial[: E.size],
        initial[E.size :],
        build_update_matrix=True,
    )
    assert evaluated.update_matrix_s_inv is not None
    independently_recomputed = -np.matmul(
        np.linalg.inv(evaluated.update_matrix_s_inv),
        evaluated.residual_s_inv,
    )
    np.testing.assert_array_equal(result.newton_deltas[0], independently_recomputed)


def test_direct_gap_constant_occupation_has_closed_form_integral() -> None:
    gap = 2.0e-6
    h = 1.0e-6
    delta0 = 2.1e-6
    f = np.full(9, 0.025)
    expected_integral = (
        4.0
        * f[0]
        * np.arcsinh(np.sqrt(f.size * h / (2.0 * gap)))
    )
    integral = author_direct_gap_integral(f, gap_eV=gap, h_eV=h)
    assert integral == expected_integral
    assert author_direct_gap(
        f,
        gap_eV=gap,
        h_eV=h,
        delta0_eV=delta0,
    ) == delta0 * np.exp(-expected_integral)


def test_thermal_occupation_can_feed_author_direct_observable() -> None:
    parameters = _parameters()
    E = parameters.gap_eV + parameters.h_eV * np.arange(9)
    thermal = author_fermi_occupation(
        E,
        parameters.temperature_K,
        parameters.constants,
    )
    assert np.all(np.diff(thermal) < 0.0)
    gap = author_direct_gap(
        thermal,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
        delta0_eV=parameters.delta0_eV,
    )
    assert 0.0 < gap < parameters.delta0_eV
