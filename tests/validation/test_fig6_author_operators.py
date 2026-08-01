from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from validation.reference_models.fischer_2023.fig6_author_operators import (
    AuthorNumericalConstants,
    evaluate_frozen_author_operators,
)


def test_reference_model_has_no_qpsim_import() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (
        root
        / "validation"
        / "reference_models"
        / "fischer_2023"
        / "fig6_author_operators.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert not any(name == "qpsim" or name.startswith("qpsim.") for name in imported)


def test_frozen_operator_decomposition_and_represented_balance() -> None:
    constants = AuthorNumericalConstants()
    gap = 5.0e-6
    h = 1.0e-6
    size = 14
    temperature = 0.2
    E = gap + h * np.arange(size)
    k_B_eV_per_K = (
        constants.boltzmann_constant_J_per_K / constants.electron_charge_C
    )
    f = 1.0 / (np.exp(E / (k_B_eV_per_K * temperature)) + 1.0)
    omega = h * np.arange(1, size)
    n_ph = 1.0 / np.expm1(omega / (k_B_eV_per_K * temperature))
    photon_bin = 2
    n_bar = float(
        1.0
        / np.expm1(
            photon_bin * h / (k_B_eV_per_K * temperature),
        )
    )

    result = evaluate_frozen_author_operators(
        E,
        f,
        n_ph,
        gap_eV=gap,
        h_eV=h,
        temperature_K=temperature,
        T_c_K=1.184,
        tau_0_s=438e-9,
        tau_0_pb_s=255e-12,
        tau_l_s=255e-12,
        photon_bin=photon_bin,
        n_bar=n_bar,
        c_photon_s_inv=1.0,
        constants=constants,
    )

    np.testing.assert_allclose(
        result.qp_phonon_s_inv,
        result.qp_scattering_s_inv + result.qp_pair_s_inv,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.qp_total_s_inv,
        result.qp_photon_s_inv + result.qp_phonon_s_inv,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.phonon_total_s_inv,
        (
            result.phonon_scattering_s_inv
            + result.phonon_pair_s_inv
            + result.phonon_escape_s_inv
        ),
        rtol=0.0,
        atol=0.0,
    )
    # Photon transitions and represented phonon channels obey thermal
    # balance. The truncated author QP equation deliberately emits
    # recombination phonons beyond its N-1 phonon grid, so its complete QP
    # residual is not a thermal-root assertion on this deliberately tiny grid.
    assert float(np.max(np.abs(result.qp_photon_s_inv))) < 1e-12
    assert float(np.max(np.abs(result.phonon_total_s_inv))) < 1e-5


def test_author_photon_residual_omits_terminal_bin_transition() -> None:
    gap = 2.0e-6
    h = 1.0e-6
    size = 10
    photon_bin = 2
    E = gap + h * np.arange(size)
    f = np.linspace(0.16, 0.01, size)
    n_ph = np.linspace(0.20, 0.01, size - 1)
    n_bar = 3.0
    c_photon = 7.0

    result = evaluate_frozen_author_operators(
        E,
        f,
        n_ph,
        gap_eV=gap,
        h_eV=h,
        temperature_K=0.2,
        T_c_K=1.184,
        tau_0_s=438e-9,
        tau_0_pb_s=255e-12,
        tau_l_s=255e-12,
        photon_bin=photon_bin,
        n_bar=n_bar,
        c_photon_s_inv=c_photon,
    )

    # At i=N-1-p the authenticated source retains only the transition from
    # i-p; its upward transition into terminal bin N-1 is absent.
    i = size - 1 - photon_bin
    j = i - photon_bin
    rho_j = (
        np.sqrt((E[j] + h) ** 2 - gap**2)
        - np.sqrt(E[j] ** 2 - gap**2)
    ) / h
    U_lower = (1.0 + gap**2 / (E[i] * E[j])) * rho_j
    expected_lower_only = c_photon * U_lower * (
        f[j] * (1.0 - f[i]) * n_bar
        - f[i] * (1.0 - f[j]) * (1.0 + n_bar)
    )
    assert result.qp_photon_s_inv[i] == pytest.approx(expected_lower_only)
    assert result.qp_photon_s_inv[-1] == 0.0
