"""Tests for qpsim.phonon_models.state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState


def _single_branch_state(n_omega: int = 5, n_spatial: int = 1) -> PhononState:
    omega_bins = np.linspace(0.1, 1.0, n_omega).reshape(1, n_omega)
    tau_l = np.full((1, n_omega), 0.25)
    n_ph = np.zeros((1, n_omega, n_spatial))
    return PhononState(
        n_ph=n_ph,
        omega_bins=omega_bins,
        tau_l=tau_l,
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )


class TestPhononState:
    def test_single_branch_homogeneous_construction(self) -> None:
        state = _single_branch_state(n_omega=5, n_spatial=1)
        assert state.n_branch == 1
        assert state.n_omega == 5
        assert state.n_spatial == 1

    def test_multi_branch_construction(self) -> None:
        omega_bins = np.tile(np.linspace(0.1, 1.0, 4), (2, 1))
        tau_l = np.full((2, 4), 0.5)
        n_ph = np.zeros((2, 4, 1))
        state = PhononState(
            n_ph=n_ph,
            omega_bins=omega_bins,
            tau_l=tau_l,
            model=PhononModel.PH0_LOCAL,
            branches=[
                PhononBranchSpec(name="longitudinal", sound_velocity=6000.0),
                PhononBranchSpec(name="transverse", sound_velocity=3000.0),
            ],
        )
        assert state.n_branch == 2

    def test_rejects_non_3d_n_ph(self) -> None:
        with pytest.raises(ValueError, match="n_ph must be 3D"):
            PhononState(
                n_ph=np.zeros(5),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((1, 5), 0.25),
                model=PhononModel.PH0_LOCAL,
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_omega_bins_shape(self) -> None:
        with pytest.raises(ValueError, match="omega_bins shape"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 4).reshape(1, 4),  # wrong NO
                tau_l=np.full((1, 5), 0.25),
                model=PhononModel.PH0_LOCAL,
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_tau_l_shape(self) -> None:
        with pytest.raises(ValueError, match="tau_l shape"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((2, 5), 0.25),  # wrong N_branch
                model=PhononModel.PH0_LOCAL,
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_branches_length(self) -> None:
        with pytest.raises(ValueError, match="branches list length"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((1, 5), 0.25),
                model=PhononModel.PH0_LOCAL,
                branches=[],  # empty for N_branch = 1
            )


class TestPhononModel:
    def test_enum_values(self) -> None:
        assert PhononModel.PH0_LOCAL.value == "ph0_local"
        assert PhononModel.PH1_BALLISTIC.value == "ph1_ballistic"
        assert PhononModel.PH2_DIFFUSIVE.value == "ph2_diffusive"
