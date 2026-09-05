"""Tests for qpsim.phonon_models.state."""

from __future__ import annotations

import numpy as np
import pytest
from qpsim.phonon_models.state import PhononBranchSpec, PhononState


def _single_branch_state(n_omega: int = 5, n_spatial: int = 1) -> PhononState:
    omega_bins = np.linspace(0.1, 1.0, n_omega).reshape(1, n_omega)
    tau_l = np.full((1, n_omega), 0.25)
    n_ph = np.zeros((1, n_omega, n_spatial))
    return PhononState(
        n_ph=n_ph,
        omega_bins=omega_bins,
        tau_l=tau_l,
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
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_omega_bins_shape(self) -> None:
        with pytest.raises(ValueError, match="omega_bins shape"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 4).reshape(1, 4),  # wrong NO
                tau_l=np.full((1, 5), 0.25),
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_tau_l_shape(self) -> None:
        with pytest.raises(ValueError, match="tau_l shape"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((2, 5), 0.25),  # wrong N_branch
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_mismatched_branches_length(self) -> None:
        with pytest.raises(ValueError, match="branches list length"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((1, 5), 0.25),
                branches=[],  # empty for N_branch = 1
            )

    def test_rejects_negative_n_ph(self) -> None:
        with pytest.raises(ValueError, match="n_ph must be non-negative"):
            PhononState(
                n_ph=-np.ones((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=np.full((1, 5), 0.25),
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_nonmonotone_omega_bins(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.array([[0.1, 0.2, 0.2, 0.4, 0.5]]),
                tau_l=np.full((1, 5), 0.25),
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_negative_tau_l(self) -> None:
        with pytest.raises(ValueError, match="tau_l must be non-negative"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=np.linspace(0.1, 1.0, 5).reshape(1, 5),
                tau_l=-np.ones((1, 5)),
                branches=[PhononBranchSpec(name="x")],
            )

    def test_rejects_nonfinite_arrays(self) -> None:
        omega = np.linspace(0.1, 1.0, 5).reshape(1, 5)
        omega[0, 2] = np.inf
        with pytest.raises(ValueError, match="omega_bins"):
            PhononState(
                n_ph=np.zeros((1, 5, 1)),
                omega_bins=omega,
                tau_l=np.full((1, 5), 0.25),
                branches=[PhononBranchSpec(name="x")],
            )
