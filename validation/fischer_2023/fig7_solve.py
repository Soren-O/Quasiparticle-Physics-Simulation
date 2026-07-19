"""Fischer 2023 Fig. 7 — the expensive Q_i(T_B) kinetic solve (cache payload).

Split out from :mod:`fig7_paper` so the config+code-hashed sweep cache
(:mod:`validation.sweep_cache`) can hash *this whole module* as the figure's
solve source: every solve-affecting constant, helper, and solver knob lives
here. Editing the downstream observables / plotting in :mod:`fig7_paper` keeps a
cached solve warm, while any change in this module correctly invalidates it.

The boundary is the point of the split: ``fig7_paper`` passes
``inspect.getsource(fig7_solve)`` as the cache's ``extra_source``. Hashing the
module text (not ``inspect.getsource(solve)``) is deliberate — it captures the
module-level constants ``solve`` reads as globals and any helper it delegates
to, which a function-only source hash would silently miss.

:func:`solve` returns a flat dict of float64 arrays (the converged occupations
plus the sweep axes) — the raw payload the cache stores. :func:`fig7_paper.observables`
derives σ₁/Q from it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import build_phonon_frequency_map
from qpsim.constants import KB_UEV_PER_K
from qpsim.grid.energy_grid import build_energy_grid, integration_widths_from_centers
from qpsim.materials.database import Material
from qpsim.phonon_models.state import PhononBranchSpec, PhononModel, PhononState
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext

from validation.fischer_2023.steady_state_certificate import (
    CERTIFICATE_FIELDS,
    CERTIFICATE_METRIC_VERSION,
    steady_state_certificate,
)

# Fischer 2023 Tables II/III — solve-affecting constants.
DELTA_0 = 189.0
TAU_0 = 63.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = 22.0
C_PHOT = 0.06e-9
TAU_0_PB = 0.040
TAU_L = 0.170

E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1701  # h = Delta / 189, E in [Delta, 10 Delta].

P_READ_DBM: tuple[float, ...] = (-100.0, -90.0, -80.0, -72.0, -68.0, -64.0)
TSTAR_OVER_DELTA: dict[float, float] = {
    -100.0: 0.12,
    -90.0: 0.18,
    -80.0: 0.26,
    -72.0: 0.36,
    -68.0: 0.42,
    -64.0: 0.49,
}

# A compact curve grid.  The standalone reproduction uses a denser visual grid;
# this is enough to pin the plateau and high-temperature rolloff without making
# the slow validation suite painfully large.
T_BATH_VALUES: tuple[float, ...] = (0.06, 0.10, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34)

# Finite-tau_l Picard solver knobs (Table II/III paper-faithful). Kept as a
# module constant so they are part of this module's source hash (the cache's
# extra_source) and are recorded verbatim in the provenance fingerprint.
SOLVER_KWARGS: dict[str, Any] = {
    "method": "picard",
    "use_phonon_side_kernel": True,
    # The former 1e-7/1e-11 outer and 1e-6 inner-backward-error gates admitted
    # visibly different approximate roots across NumPy/SciPy/BLAS builds even
    # though every state passed the looser independent certificate.  The tight
    # contract was checked at all 48 production targets on Windows and at the
    # two worst cross-platform targets on Linux before promotion.
    "picard_tol": 1e-9,
    "picard_atol": 1e-13,
    "picard_balance_tol": 1e-8,
    "picard_max_iter": 2000,
    "picard_mixing": 0.2,
    "anderson_depth": 3,
    "newton_tol": 1e-13,
    "newton_backward_error_tol": 1e-8,
    "newton_max_iter": 300,
}

TARGET_BACKWARD_ERROR_LIMIT = 2e-8
"""Maximum independently reassembled balance error at every Fig. 7 target."""


def _paper_material() -> Material:
    return Material(
        name="Al_Fischer2023_TableII",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=TAU_0_PB,
    )


def _nbar_from_table_iii(power_dbm: float) -> float:
    """Return nbar whose Tstar,0/Delta matches Fischer Table III."""
    target = TSTAR_OVER_DELTA[float(power_dbm)] * DELTA_0
    T_c_uev = T_C * KB_UEV_PER_K
    denom = (
        (105.0 / 64.0)
        * T_c_uev**3
        * DELTA_0
        * (C_PHOT * TAU_0)
        * OMEGA_0**2
    )
    return float(target**6 / denom)


def _validated_sweep_request(
    temperatures: tuple[float, ...],
    powers_dbm: tuple[float, ...],
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return canonical float axes after rejecting ambiguous sweep requests.

    Fig. 7 points are independent, but their raw payload is later keyed by
    floating-point power.  Duplicate powers would therefore collapse through
    the observable dictionaries, while duplicate temperatures would create an
    ambiguous Cartesian artifact.  Reject both at the solve boundary instead
    of relying on the downstream writer to notice after an expensive run.
    """
    temperatures_array = np.asarray(temperatures, dtype=float)
    if temperatures_array.ndim != 1 or temperatures_array.size == 0:
        raise ValueError("Fig. 7 temperatures must be a non-empty one-dimensional axis.")
    if (
        np.any(~np.isfinite(temperatures_array))
        or np.any(temperatures_array <= 0.0)
        or np.unique(temperatures_array).size != temperatures_array.size
    ):
        raise ValueError("Fig. 7 temperatures must be finite, positive, and unique.")

    powers_array = np.asarray(powers_dbm, dtype=float)
    if powers_array.ndim != 1 or powers_array.size == 0:
        raise ValueError("Fig. 7 powers must be a non-empty one-dimensional axis.")
    if (
        np.any(~np.isfinite(powers_array))
        or np.unique(powers_array).size != powers_array.size
    ):
        raise ValueError("Fig. 7 powers must be finite and unique.")
    unsupported = sorted({float(power) for power in powers_array} - set(TSTAR_OVER_DELTA))
    if unsupported:
        raise ValueError(
            "Fig. 7 powers must have Table-III T*/Delta entries; "
            f"unsupported powers={unsupported}."
        )

    if (
        isinstance(num_bins, (bool, np.bool_))
        or not isinstance(num_bins, (int, np.integer))
        or int(num_bins) < 2
    ):
        raise ValueError(f"Fig. 7 num_bins must be an integer >= 2; got {num_bins!r}.")
    return temperatures_array, powers_array, int(num_bins)


def _build_grid(num_bins: int = NUM_BINS) -> tuple[SpectralContext, np.ndarray]:
    E, dE_scalar = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=num_bins,
    )
    shift = OMEGA_0 / dE_scalar
    if abs(shift - round(shift)) > 1e-10:
        raise ValueError(
            f"Fig. 7 paper grid must be commensurate with omega_0. "
            f"Got dE={dE_scalar:.6g}, omega_0/dE={shift:.6g}."
        )
    dE = integration_widths_from_centers(E)
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    omega, _, _, _ = build_phonon_frequency_map(E)
    return spectral, omega


def _fermi_dirac(E: np.ndarray, T_bath: float) -> np.ndarray:
    kT = KB_UEV_PER_K * T_bath
    return 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)


def _build_state(
    material: Material,
    spectral: SpectralContext,
    omega: np.ndarray,
    T_bath: float,
    *,
    f_seed: np.ndarray | None = None,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    f0 = _fermi_dirac(spectral.E, T_bath) if f_seed is None else f_seed.copy()
    n_ph = (
        thermal_phonon_occupation(omega, T_bath)
        if n_ph_seed is None
        else n_ph_seed.copy()
    )
    phonon = PhononState(
        n_ph=n_ph.reshape(1, -1, 1),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), TAU_L),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=f0,
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def solve(
    *,
    temperatures: tuple[float, ...] = T_BATH_VALUES,
    powers_dbm: tuple[float, ...] = P_READ_DBM,
    num_bins: int = NUM_BINS,
) -> dict[str, np.ndarray]:
    """Run the finite-tau_l kinetic sweep; return the raw converged occupations.

    The expensive half of Fig. 7. For each (readout power, T_bath) point the T3
    backend solves the steady-state quasiparticle distribution from a fresh
    thermal seed (no warm-seeding between points — every point is independent,
    which is what makes the solve/observable split bit-for-bit faithful to the
    original ``run``). The returned payload is what the sweep cache stores.

    Returns
    -------
    dict[str, np.ndarray]
        ``f_solved`` (n_powers, n_T, NE) converged occupations, plus the
        ``temperatures``, ``powers_dbm``, per-power ``n_bar``, and ``num_bins``
        axes needed to reconstruct the observables downstream.
    """
    T_values, powers, resolved_num_bins = _validated_sweep_request(
        temperatures,
        powers_dbm,
        num_bins,
    )
    material = _paper_material()
    spectral, omega = _build_grid(resolved_num_bins)
    backend = T3DiffusionBackend()

    n_bar = np.asarray([_nbar_from_table_iii(p) for p in powers], dtype=float)

    NE = spectral.E.size
    f_solved = np.zeros((powers.size, T_values.size, NE), dtype=float)
    certificates = {
        field: np.zeros((powers.size, T_values.size), dtype=float)
        for field in CERTIFICATE_FIELDS
    }

    for pi, _p_dbm in enumerate(powers):
        photon_params = {"omega_0": OMEGA_0, "n_bar": float(n_bar[pi]), "c_phot": C_PHOT}
        for ti, T_bath in enumerate(T_values):
            state = _build_state(material, spectral, omega, float(T_bath))
            solved = backend.steady_state(
                state,
                photon_params=photon_params,
                **SOLVER_KWARGS,
            )
            f_solved[pi, ti] = solved.f
            certificate = steady_state_certificate(
                solved,
                photon_params=photon_params,
                tau_l=TAU_L,
            )
            for field in CERTIFICATE_FIELDS:
                certificates[field][pi, ti] = certificate[field]
            qp_backward = certificate["qp_backward_error"]
            phonon_backward = certificate["phonon_backward_error"]
            if (
                not np.isfinite(qp_backward)
                or not np.isfinite(phonon_backward)
                or qp_backward > TARGET_BACKWARD_ERROR_LIMIT
                or phonon_backward > TARGET_BACKWARD_ERROR_LIMIT
            ):
                raise RuntimeError(
                    "Fischer Fig. 7 target failed the independent steady-state "
                    f"certificate at P_read={_p_dbm:g} dBm, "
                    f"T_bath={T_bath:g} K (limit="
                    f"{TARGET_BACKWARD_ERROR_LIMIT:g}): "
                    f"qp={qp_backward:.3e}, phonon={phonon_backward:.3e}."
                )

    return {
        "f_solved": f_solved,
        "temperatures": T_values,
        "powers_dbm": powers,
        "n_bar": n_bar,
        "num_bins": np.asarray([resolved_num_bins]),
        **certificates,
    }


def solver_fingerprint(*, num_bins: int = NUM_BINS) -> dict[str, Any]:
    """Resolved physics + solver knobs, for the cache key's provenance sidecar.

    Safety does *not* hinge on this being exhaustive: the cache key also folds in
    this module's full source (``extra_source``), the ``qpsim`` solver-subtree
    digest, and the ``run`` kwargs. This dict keeps the stored provenance
    human-legible and gives the fingerprint a stable, inspectable core.
    """
    return {
        "delta_0": DELTA_0,
        "tau_0": TAU_0,
        "t_c": T_C,
        "omega_0": OMEGA_0,
        "c_phot": C_PHOT,
        "tau_0_pb": TAU_0_PB,
        "tau_l": TAU_L,
        "e_min_factor": E_MIN_FACTOR,
        "e_max_factor": E_MAX_FACTOR,
        "num_bins": int(num_bins),
        "solver": dict(SOLVER_KWARGS),
        "certificate_metric_version": CERTIFICATE_METRIC_VERSION,
        "target_backward_error_limit": TARGET_BACKWARD_ERROR_LIMIT,
    }
