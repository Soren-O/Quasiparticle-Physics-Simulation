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
        returned state shares every field with the input *except* ``f``.

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

        # v1 single-branch, spatially homogeneous ⇒ scalar τ_l from the
        # first branch/first ω entry. An ω-resolved τ_l(ω) path requires
        # changing solve_steady_state to accept an array, which we defer
        # until the Kaplan-escape model needs it (post-Gate 4).
        tau_l_scalar = float(state.phonon.tau_l[0, 0])

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
        )
        return replace(state, f=f_new)
