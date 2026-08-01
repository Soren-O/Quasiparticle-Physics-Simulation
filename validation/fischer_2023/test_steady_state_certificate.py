"""Focused tests for the independent Fischer balance certificate."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation

from validation.fischer_2023.steady_state_certificate import (
    steady_state_certificate,
    weighted_number_backward_error,
)

_GAP = 189.0
_T_BATH = 0.10
_TAU_L = 0.17
_PHOTON_PARAMS = {"omega_0": 22.0, "n_bar": 0.0, "c_phot": 0.0}


def test_weighted_number_backward_error_fails_closed_without_pair_turnover() -> None:
    zeros = np.zeros(3)
    assert np.isinf(
        weighted_number_backward_error(
            zeros,
            zeros,
            zeros,
            np.ones(3),
            np.ones(3, dtype=bool),
        )
    )


@pytest.fixture(scope="module")
def certified_equilibrium() -> T3DiffusionState:
    # Commensurate grid: omega_0 / dE = 4.  Zero photon coupling leaves the
    # exact thermal state, which gives a clean reference for perturbation tests.
    E = _GAP + 5.5 * np.arange(32, dtype=float)
    spectral = SpectralContext(
        E_bins=E,
        dE_bins=integration_widths_from_centers(E),
        gap=_GAP,
    )
    omega, _, _, _ = build_phonon_frequency_map(E)
    f = fermi_dirac_occupation(E, _T_BATH)
    n_ph = thermal_phonon_occupation(omega, _T_BATH)
    material = Material(
        name="certificate_test",
        Delta_0=_GAP,
        T_c=_GAP / (1.764 * KB_UEV_PER_K),
        tau_0=63.0,
        tau_0_pb_ns=0.040,
    )
    state = T3DiffusionState(
        f=f,
        gap=_GAP,
        spectral=spectral,
        phonon=PhononState(
            n_ph=n_ph.reshape(1, -1, 1),
            omega_bins=omega.reshape(1, -1),
            tau_l=np.full((1, omega.size), _TAU_L),
            model=PhononModel.PH0_LOCAL,
            branches=[PhononBranchSpec(name="debye_average")],
        ),
        material=material,
        T_bath=_T_BATH,
    )
    return T3DiffusionBackend().steady_state(
        state,
        photon_params=_PHOTON_PARAMS,
        use_phonon_side_kernel=True,
        newton_tol=1e-12,
        newton_max_iter=100,
        picard_tol=1e-8,
        picard_atol=0.0,
        picard_max_iter=100,
    )


def test_certificate_detects_phonon_residual_perturbation(
    certified_equilibrium: T3DiffusionState,
) -> None:
    reference = steady_state_certificate(
        certified_equilibrium,
        photon_params=_PHOTON_PARAMS,
        tau_l=_TAU_L,
    )
    assert reference["qp_backward_error"] < 1e-12
    assert reference["qp_number_backward_error"] < 1e-12
    assert reference["phonon_backward_error"] < 1e-12
    assert np.isfinite(reference["phonon_raw_backward_error"])
    assert (
        reference["phonon_raw_backward_error"]
        >= reference["phonon_backward_error"]
    )

    n_ph = certified_equilibrium.phonon.n_ph.copy()
    n_ph[0, 4, 0] += 1e-7
    perturbed = replace(
        certified_equilibrium,
        phonon=replace(certified_equilibrium.phonon, n_ph=n_ph),
    )
    certificate = steady_state_certificate(
        perturbed,
        photon_params=_PHOTON_PARAMS,
        tau_l=_TAU_L,
    )

    assert certificate["phonon_residual_inf"] > 1e-8
    assert certificate["phonon_backward_error"] > 1e-2
    assert certificate["qp_backward_error"] > 1e-10


def test_certificate_detects_qp_residual_perturbation(
    certified_equilibrium: T3DiffusionState,
) -> None:
    f = certified_equilibrium.f.copy()
    f[4] += 1e-6
    perturbed = replace(certified_equilibrium, f=f)
    certificate = steady_state_certificate(
        perturbed,
        photon_params=_PHOTON_PARAMS,
        tau_l=_TAU_L,
    )

    assert certificate["qp_residual_inf"] > 1e-12
    assert certificate["qp_backward_error"] > 1e-2
    assert certificate["phonon_backward_error"] > 1e-2


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_certificate_detects_wrong_number_on_correct_thermal_shape(
    certified_equilibrium: T3DiffusionState,
    scale: float,
) -> None:
    """Number-conserving turnover must not certify a scaled cold solution."""
    scaled = replace(
        certified_equilibrium,
        f=np.clip(scale * certified_equilibrium.f, 0.0, 1.0),
    )
    certificate = steady_state_certificate(
        scaled,
        photon_params=_PHOTON_PARAMS,
        tau_l=_TAU_L,
    )

    assert certificate["qp_backward_error"] < 1e-5
    assert certificate["phonon_backward_error"] < 1e-5
    assert certificate["qp_number_backward_error"] == pytest.approx(0.6, rel=1e-8)


def test_certificate_rejects_unassembled_channel_scope(
    certified_equilibrium: T3DiffusionState,
) -> None:
    unsupported = dict(_PHOTON_PARAMS, omega_PB=400.0)
    with pytest.raises(ValueError, match="supports exactly one sub-gap photon"):
        steady_state_certificate(
            certified_equilibrium,
            photon_params=unsupported,
            tau_l=_TAU_L,
        )
