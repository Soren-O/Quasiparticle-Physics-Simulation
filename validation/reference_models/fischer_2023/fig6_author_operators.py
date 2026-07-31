"""Compatibility facade for the qpsim-free Fischer Figure 6 C0 core.

The original frozen-state diagnostic API remains available here.  Its
numerical implementation now delegates to :mod:`fig6_author_c0`, so the
clean-room residual algebra has one maintained source.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    build_author_operator,
    evaluate_author_system,
)

__all__ = [
    "AuthorNumericalConstants",
    "FrozenAuthorOperatorResult",
    "evaluate_frozen_author_operators",
]


@dataclass(frozen=True)
class FrozenAuthorOperatorResult:
    """Author-equivalent per-channel residuals, all in inverse seconds."""

    phonon_escape_s_inv: np.ndarray
    phonon_pair_s_inv: np.ndarray
    phonon_scattering_s_inv: np.ndarray
    phonon_total_s_inv: np.ndarray
    qp_pair_s_inv: np.ndarray
    qp_phonon_s_inv: np.ndarray
    qp_photon_s_inv: np.ndarray
    qp_scattering_s_inv: np.ndarray
    qp_total_s_inv: np.ndarray


def evaluate_frozen_author_operators(
    E_left_eV: np.ndarray,
    f: np.ndarray,
    n_phonon: np.ndarray,
    *,
    gap_eV: float,
    h_eV: float,
    temperature_K: float,
    T_c_K: float,
    tau_0_s: float,
    tau_0_pb_s: float,
    tau_l_s: float,
    photon_bin: int,
    n_bar: float,
    c_photon_s_inv: float,
    constants: AuthorNumericalConstants | None = None,
) -> FrozenAuthorOperatorResult:
    """Evaluate the author discretization on one frozen ``(f, n_ph)`` state."""

    selected_constants = (
        AuthorNumericalConstants() if constants is None else constants
    )
    parameters = AuthorSolveParameters(
        gap_eV=float(gap_eV),
        h_eV=float(h_eV),
        temperature_K=float(temperature_K),
        T_c_K=float(T_c_K),
        tau_0_s=float(tau_0_s),
        tau_0_pb_s=float(tau_0_pb_s),
        tau_l_s=float(tau_l_s),
        photon_bin=int(photon_bin),
        n_bar=float(n_bar),
        c_photon_s_inv=float(c_photon_s_inv),
        # These fields do not enter frozen collision evaluation.
        delta0_eV=float(gap_eV),
        thermal_gap_eV=float(gap_eV),
        max_newton_steps=1,
        relative_step_threshold=1.0,
        constants=selected_constants,
    )
    evaluated = evaluate_author_system(
        build_author_operator(E_left_eV, parameters),
        f,
        n_phonon,
        build_update_matrix=False,
    )
    qp_phonon = evaluated.qp_scattering_s_inv + evaluated.qp_pair_s_inv
    qp_total = evaluated.qp_photon_s_inv + qp_phonon
    phonon_total = (
        evaluated.phonon_scattering_s_inv
        + evaluated.phonon_pair_s_inv
        + evaluated.phonon_escape_s_inv
    )
    return FrozenAuthorOperatorResult(
        phonon_escape_s_inv=evaluated.phonon_escape_s_inv,
        phonon_pair_s_inv=evaluated.phonon_pair_s_inv,
        phonon_scattering_s_inv=evaluated.phonon_scattering_s_inv,
        phonon_total_s_inv=phonon_total,
        qp_pair_s_inv=evaluated.qp_pair_s_inv,
        qp_phonon_s_inv=qp_phonon,
        qp_photon_s_inv=evaluated.qp_photon_s_inv,
        qp_scattering_s_inv=evaluated.qp_scattering_s_inv,
        qp_total_s_inv=qp_total,
    )
