"""T3 backend: isotropic dirty-limit diffusion (scalar occupation).

Composes the pieces from earlier Gate 2 commits into a working
steady-state solver:

* :mod:`qpsim.physics.spectral` — :class:`SpectralContext` for the
  gap-dependent DOS / coherence matrices / diffusion coefficient.
* :mod:`qpsim.collisions.phonon` — kernel builders with coherence-
  assignment wiring.
* :mod:`qpsim.services.steady_state` — the Newton + Picard
  orchestrator that handles both the thermal-phonon limit and the
  finite-``τ_l`` Picard iteration.

Scope for Gate 2: spatially-homogeneous runs (``N_spatial = 1``), a
scalar gap, the e-phonon integral, and optional sub-gap / PB photon
channels via ``photon_params`` / ``pb_photon_params``. Transient
evolution (``step``, ``apply_collisions``, ``apply_transport``,
``apply_gap_update``) lands with the Strang + ETD2 upgrades in task 12.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from qpsim.backends.base import Tier
from qpsim.collisions.phonon import (
    build_recombination_kernel_base,
    build_scattering_kernel_base,
)
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononState
from qpsim.physics.spectral import SpectralContext
from qpsim.services.steady_state import solve_steady_state

_TAU_L_UNIFORMITY_RTOL = 1e-10


@dataclass
class T3DiffusionState:
    """State carried by the T3 diffusion backend.

    For v1 (Gate 2) the gap is a scalar and ``f`` is 1D over energy
    only. Multi-material / spatially-varying gaps add a ``GapState``
    (NFP §4.1) with per-cell ``SpectralContext`` slots; that's a
    future-gate extension.

    Attributes
    ----------
    f
        QP occupation on the energy grid, shape ``(NE,)``.
    gap
        Scalar gap ``Δ`` (μeV).
    spectral
        Gap-dependent cache (DOS, K±, ``D(E)``). Must match ``gap``.
    phonon
        Phonon state (n_ph, τ_l, model, branches). For v1 single-branch
        Ph0 the first entry of ``phonon.tau_l`` is used as the scalar
        bath escape time passed into the Picard path.
    material
        Source of ``T_c``, ``τ_0``, and other material parameters used
        by the kernel builders.
    T_bath
        Substrate bath temperature in K.
    tier
        Always :attr:`Tier.T3_DIFFUSION`; included for downstream code
        that branches on the tier enum.
    """

    f: np.ndarray
    gap: float
    spectral: SpectralContext
    phonon: PhononState
    material: Material
    T_bath: float
    tier: Tier = Tier.T3_DIFFUSION


class T3DiffusionBackend:
    """Steady-state solver for the T3 tier.

    Stateless; a single instance can be reused across many runs.
    Transient methods land in task 12.
    """

    def steady_state(
        self,
        state: T3DiffusionState,
        *,
        photon_params: dict[str, float] | None = None,
        pb_photon_params: dict[str, float] | None = None,
        newton_tol: float = 1e-14,
        newton_max_iter: int = 200,
        picard_tol: float = 1e-10,
        picard_max_iter: int = 200,
        picard_mixing: float = 0.3,
        anderson_depth: int = 0,
    ) -> T3DiffusionState:
        """Solve for the steady-state ``f(E)`` and return an updated state.

        Rebuilds the e-ph kernels from the current
        :class:`SpectralContext` and Material, extracts the (scalar)
        ``τ_l`` from the PhononState, and delegates to
        :func:`qpsim.services.steady_state.solve_steady_state`. The
        returned state has both ``f`` and ``phonon`` updated; ``phonon``
        is rebuilt on the physics ``ω`` grid (the pair-sum / pair-
        difference grid derived from ``state.spectral.E`` via
        :func:`qpsim.collisions.phonon.build_phonon_frequency_map`)
        with the converged ``n_ph`` and a tiled scalar ``τ_l``.

        Gate 2 scope requires the input ``state.phonon`` to have
        ``n_branch = 1``, ``n_spatial = 1``, and constant
        ``τ_l`` (all entries equal). Violations raise ``ValueError``
        rather than silently mis-solving.

        Parameters
        ----------
        state
            Initial T3 state; ``state.f`` is the Newton/Picard initial
            guess.
        photon_params
            Optional sub-gap photon dict
            ``{"omega_0", "n_bar", "c_phot"}``.
        pb_photon_params
            Optional pair-breaking photon dict
            ``{"omega_PB", "n_bar_PB", "c_phot_PB"}``.
        newton_tol, newton_max_iter
            Inner Newton controls.
        picard_tol, picard_max_iter, picard_mixing, anderson_depth
            Outer Picard controls for the finite-``τ_l`` path.
        """
        self._validate_gate2_scope(state.phonon)

        K_s0 = build_scattering_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )
        K_r0 = build_recombination_kernel_base(
            state.spectral,
            tau_0=state.material.tau_0,
            T_c=state.material.T_c,
        )

        tau_l_scalar = float(state.phonon.tau_l[0, 0])

        phonon_out: dict[str, np.ndarray] = {}
        f_new = solve_steady_state(
            state.spectral,
            K_s0,
            K_r0,
            state.T_bath,
            photon_params=photon_params,
            pb_photon_params=pb_photon_params,
            initial_guess=state.f,
            tol=newton_tol,
            max_iter=newton_max_iter,
            phonon_escape_time=tau_l_scalar,
            max_picard_iter=picard_max_iter,
            picard_tol=picard_tol,
            picard_mixing=picard_mixing,
            anderson_depth=anderson_depth,
            phonon_out=phonon_out,
        )

        # Rebuild the phonon state on the physics ω grid with the
        # converged n_ph. The input state's omega_bins / n_ph are not
        # used by the solver — the Picard loop builds its own ω grid
        # from the QP energy grid — so the returned PhononState must
        # reflect what the physics actually computed.
        n_ph_conv = phonon_out["n_ph"]
        omega_conv = phonon_out["omega_bins"]
        new_phonon = replace(
            state.phonon,
            n_ph=n_ph_conv.reshape(1, -1, 1),
            omega_bins=omega_conv.reshape(1, -1),
            tau_l=np.full((1, omega_conv.size), tau_l_scalar),
        )
        return replace(state, f=f_new, phonon=new_phonon)

    @staticmethod
    def _validate_gate2_scope(phonon: PhononState) -> None:
        """Reject PhononState shapes the Gate 2 T3 backend can't handle.

        Gate 2 supports the single-branch, spatially-homogeneous,
        constant-``τ_l`` case. Multi-branch (v3), spatially-resolved
        (Ph1/Ph2), and ω-dependent ``τ_l`` land in later gates.
        """
        if phonon.n_branch != 1:
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports single-branch phonons only; "
                f"got n_branch = {phonon.n_branch}. Multi-branch support arrives "
                "with v3 per D5."
            )
        if phonon.n_spatial != 1:
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports spatially-homogeneous "
                f"phonons only; got n_spatial = {phonon.n_spatial}. Ph1 lateral "
                "transport lands at Gate 5."
            )
        tau0 = float(phonon.tau_l[0, 0])
        if not np.allclose(phonon.tau_l, tau0, rtol=_TAU_L_UNIFORMITY_RTOL):
            raise ValueError(
                "T3DiffusionBackend (Gate 2) supports constant-τ_l only; "
                "every entry of state.phonon.tau_l must be equal. "
                "Frequency-dependent τ_l(ω) needs solve_steady_state to "
                "accept an array, which is a post-Gate-4 upgrade."
            )
