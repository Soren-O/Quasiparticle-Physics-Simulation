"""Phonon state representation with the (N_branch, N_omega, N_spatial) shape.

Carries the non-equilibrium phonon distribution ``n_ph(ω, r)`` over
possibly multiple phonon branches (single-branch Debye for v1 per
D5; two-branch L+T for v3), the per-branch frequency grid
``omega_bins``, and the per-branch bath-relaxation time
``tau_l(ω)`` (acoustic escape).

``PhononState`` is *data* only — it holds arrays and a model tag.
The steady-state solve for ``n_ph`` given ``f`` lives in
:mod:`qpsim.phonon_models.ph0_local`. ``tau_l`` builders (constant,
acoustic-escape) live in :mod:`qpsim.physics.phonon_escape`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PhononModel(Enum):
    """Phonon-sector tier (see Phonon_Model_Decisions.md)."""

    PH0_LOCAL = "ph0_local"          # local bath with escape (v1 target)
    PH1_BALLISTIC = "ph1_ballistic"  # lateral transport (deferred v2)
    PH2_DIFFUSIVE = "ph2_diffusive"  # explicit substrate (deferred v3)


@dataclass
class PhononBranchSpec:
    """Per-branch acoustic metadata.

    ``name`` is a label like ``"longitudinal"``, ``"transverse"``, or
    ``"debye_average"``. ``sound_velocity`` (m/s) and ``omega_debye``
    (μeV) are optional; they are needed when the backend wants to
    evaluate branch-specific kernels rather than a scalar Debye-averaged
    form.
    """

    name: str
    sound_velocity: float | None = None
    omega_debye: float | None = None


@dataclass
class PhononState:
    """Phonon distribution, frequency grid, and bath escape-time.

    Shape conventions (from NFP §4.1):

    * ``n_ph`` is ``(N_branch, N_omega, N_spatial)``.
    * ``omega_bins`` is ``(N_branch, N_omega)``.
    * ``tau_l`` is ``(N_branch, N_omega)``.
    * ``branches`` is a length-``N_branch`` list of
      :class:`PhononBranchSpec`.

    For v1 (single-branch Debye, spatially homogeneous), ``N_branch = 1``
    and ``N_spatial = 1``. The singleton axes are retained so that
    extending to multi-branch (v3) or spatially-resolved (Ph1/Ph2) is
    an additive change.
    """

    n_ph: np.ndarray
    omega_bins: np.ndarray
    tau_l: np.ndarray
    model: PhononModel
    branches: list[PhononBranchSpec]

    def __post_init__(self) -> None:
        if self.n_ph.ndim != 3:
            raise ValueError(
                f"n_ph must be 3D (N_branch, N_omega, N_spatial); got shape {self.n_ph.shape}."
            )
        if self.omega_bins.ndim != 2:
            raise ValueError(
                f"omega_bins must be 2D (N_branch, N_omega); got shape {self.omega_bins.shape}."
            )
        if self.tau_l.ndim != 2:
            raise ValueError(
                f"tau_l must be 2D (N_branch, N_omega); got shape {self.tau_l.shape}."
            )

        nb, no, _ = self.n_ph.shape
        if self.omega_bins.shape != (nb, no):
            raise ValueError(
                f"omega_bins shape {self.omega_bins.shape} does not match "
                f"(N_branch, N_omega) = ({nb}, {no})."
            )
        if self.tau_l.shape != (nb, no):
            raise ValueError(
                f"tau_l shape {self.tau_l.shape} does not match "
                f"(N_branch, N_omega) = ({nb}, {no})."
            )
        if len(self.branches) != nb:
            raise ValueError(
                f"branches list length {len(self.branches)} does not match "
                f"N_branch = {nb}."
            )

    @property
    def n_branch(self) -> int:
        return int(self.n_ph.shape[0])

    @property
    def n_omega(self) -> int:
        return int(self.n_ph.shape[1])

    @property
    def n_spatial(self) -> int:
        return int(self.n_ph.shape[2])
