"""Fischer 2023 Fig. 5 — the expensive two-panel kinetic solve (cache payload).

Split out from :mod:`fig5_paper` so the config+code-hashed sweep cache
(:mod:`validation.sweep_cache`) can hash *this whole module* as the figure's
solve source: every solve-affecting constant (including the ``_A_EQ35`` Eq.-35
prefactor that sets the n̄ sweep axes), helper, and solver knob lives here, so
editing the observables / analytic-overlay / plotting code in :mod:`fig5_paper`
keeps a cached solve warm while any change here correctly invalidates it. The
whole module is passed as ``extra_source`` (not ``inspect.getsource(solve)``,
which would miss the module globals ``solve`` reads and the helpers it calls).

:func:`solve` returns the converged ``f(E)`` for every (panel, point) as the
raw payload; :func:`fig5_paper.observables` derives ``x_qp`` (via
``qp_fraction``), the Eq.-47 analytic overlay, and the Eq.-35 ``T_*`` axis.

The two panels (upper: sweep n̄ at fixed T_B; lower: sweep T_B at fixed n̄) are
parameterized by ``num_bins`` and the four sweep axes (defaulting to the paper
config) so reduced-scale dev runs and equivalence checks are cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionBackend, T3DiffusionState
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    compute_phonon_source_sink,
)
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

# ── Fischer 2023 Table I parameters (shared with fig3) ───────────────

DELTA_0 = 180.0
TAU_0 = 438.0
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
OMEGA_0 = DELTA_0 / 9.0
C_PHOT = 1e-9

# Paper grid: 1620 bins, dE = 1 μeV (same as fig3).
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1620

# Eq. 35 prefactor (μeV^6) — A·n̄ has units μeV^6, (A·n̄)^(1/6) is in μeV.
# Sets the n̄ sweep axes below, so it is solve-affecting and lives here;
# fig5_paper._kBTstar_eq35 imports it for the analytic T_* overlay.
_A_EQ35 = (
    (105.0 / 64.0)
    * (KB_UEV_PER_K * T_C) ** 3
    * C_PHOT
    * TAU_0
    * OMEGA_0 ** 2
    * DELTA_0
)

# Upper panel — three bath temperatures, swept over n̄ chosen so the
# T_*/Δ axis (Eq. 35) covers the paper's plotted range [0.30, 0.95].
UPPER_T_BATH_K: tuple[float, ...] = (0.10, 0.15, 0.20)
UPPER_T_STAR_OVER_DELTA: np.ndarray = np.linspace(0.30, 0.95, 14)
UPPER_NBAR_VALUES: np.ndarray = (UPPER_T_STAR_OVER_DELTA * DELTA_0) ** 6 / _A_EQ35

# Lower panel — sweep T_B at three FIXED T_*/Δ values. The paper plots
# fixed T_*/Δ; under Eq. 35 with A independent of T_B, fixed T_*/Δ
# corresponds to a fixed n̄, so the per-T_*/Δ continuation is just a
# T_B sweep at the corresponding n̄.
LOWER_T_STAR_OVER_DELTA: tuple[float, ...] = (0.50, 0.66, 0.74)
LOWER_NBAR: tuple[float, ...] = tuple(
    float((t * DELTA_0) ** 6 / _A_EQ35) for t in LOWER_T_STAR_OVER_DELTA
)
LOWER_T_BATH_K: np.ndarray = np.linspace(0.10, 0.40, 13)

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the numerical tau_0^PB diverges from the paper-quoted 255 ps."""

TARGET_BACKWARD_ERROR_LIMIT = 1.0e-5
"""Maximum independently reassembled QP and phonon balance error."""

SOLVER_KWARGS: dict[str, Any] = {
    "method": "picard",
    "use_phonon_side_kernel": True,
    "picard_tol": 1.0e-8,
    "picard_atol": 1.0e-12,
    "picard_balance_tol": 1.0e-6,
    "picard_max_iter": 10000,
    "anderson_depth": 0,
    "newton_tol": 1.0e-12,
    "newton_backward_error_tol": 1.0e-6,
    "newton_max_iter": 500,
}


def _fischer_material() -> Material:
    return Material(
        name="Al_Fischer2023",
        Delta_0=DELTA_0,
        T_c=T_C,
        tau_0=TAU_0,
        tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,  # F&C 2023 Table I: τ_0^PB = 255 ps
    )


def _build_grid_and_spectral(
    num_bins: int = NUM_BINS,
) -> tuple[np.ndarray, np.ndarray, SpectralContext]:
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=num_bins,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    if abs(OMEGA_0 - m * dE_scalar) / OMEGA_0 > 1e-10:
        raise RuntimeError(
            f"Fischer Fig. 5 paper grid not commensurate: ω_0={OMEGA_0}, m·dE={m*dE_scalar}"
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _compute_tau_0_pb(spectral: SpectralContext) -> float:
    """Same definition as :func:`fig3_solve._compute_tau_0_pb`."""
    K_r0_phonon_side = build_recombination_kernel_phonon_side(
        spectral, tau_0_pb_ns=PAPER_TAU_0_PB_PS / 1000.0,
    )
    omega_bins, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(
        spectral.E,
    )
    f_zero = np.zeros(spectral.E.size)
    _, b_ph = compute_phonon_source_sink(
        f_zero, spectral, None, None,
        idx_diff, idx_sum, diff_sign,
        omega_bins.size,
        enable_scattering=False, enable_recombination=True,
        K_r0_phonon_side=K_r0_phonon_side,
    )
    threshold = 2.0 * spectral.gap
    above = (omega_bins >= threshold) & (b_ph < -1e-30)
    if not np.any(above):
        raise RuntimeError(
            "Could not find a phonon bin above 2Δ with a pair-breaking rate."
        )
    first_idx = int(np.argmax(above))
    return float(1.0 / -b_ph[first_idx])


def _build_state(
    material: Material,
    spectral: SpectralContext,
    T_bath: float,
    tau_l_scalar: float,
    *,
    f_seed: np.ndarray | None = None,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    if n_ph_seed is None:
        n_ph_seed = thermal_phonon_occupation(omega, T_bath)
    phonon = PhononState(
        n_ph=n_ph_seed.reshape(1, -1, 1).copy(),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_scalar),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    if f_seed is None:
        kT = KB_UEV_PER_K * T_bath
        f_seed = 1.0 / (np.exp(np.minimum(spectral.E / kT, 500.0)) + 1.0)
    return T3DiffusionState(
        f=f_seed.copy(),
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_bath,
    )


def _solve_picard(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    mixing: float = 0.30,
) -> T3DiffusionState:
    kwargs = dict(SOLVER_KWARGS)
    kwargs["picard_mixing"] = mixing
    return backend.steady_state(
        state,
        photon_params=photon_params,
        **kwargs,
    )


def _check_tau_0_pb(tau_0_pb: float) -> None:
    tau_ps = tau_0_pb * 1000.0
    print(f"  τ_0^PB (phonon-side extracted)       = {tau_0_pb:.4f} ns "
          f"({tau_ps:.1f} ps)")
    print(f"  Paper-quoted τ_0^PB                   ≈ {PAPER_TAU_0_PB_PS:.0f} ps")
    ratio = tau_ps / PAPER_TAU_0_PB_PS
    if ratio > TAU_0_PB_WARN_FACTOR or ratio < 1.0 / TAU_0_PB_WARN_FACTOR:
        print(
            f"  ⚠ τ_0^PB normalization mismatch: extracted/paper = {ratio:.2f}×.",
            flush=True,
        )


def solve(
    *,
    num_bins: int = NUM_BINS,
    upper_T_bath: tuple[float, ...] = UPPER_T_BATH_K,
    upper_nbar: np.ndarray | tuple[float, ...] = UPPER_NBAR_VALUES,
    lower_nbar: tuple[float, ...] = LOWER_NBAR,
    lower_T_bath: np.ndarray | tuple[float, ...] = LOWER_T_BATH_K,
) -> dict[str, np.ndarray]:
    """Solve both Fig. 5 panels via warm-seeded continuation; return raw f(E).

    The expensive half of Fig. 5: dozens of finite-τ_l Picard solves
    (τ_l = τ_0^PB). Upper panel sweeps n̄ at each T_B, warm-seeding each step
    from the previous n̄; lower panel sweeps T_B at each n̄, warm-seeding from
    the adjacent T_B. Returns the converged ``f`` per (panel, point);
    :func:`fig5_paper.observables` derives x_qp / overlays from it.
    """
    material = _fischer_material()
    _, _, spectral = _build_grid_and_spectral(num_bins)

    tau_0_pb = _compute_tau_0_pb(spectral)
    _check_tau_0_pb(tau_0_pb)
    tau_l = 1.0 * tau_0_pb  # paper: τ_ℓ = τ_0^PB throughout Fig. 5

    backend = T3DiffusionBackend()
    NE = spectral.E.size

    up_T = np.asarray(upper_T_bath, dtype=float)
    up_N = np.asarray(upper_nbar, dtype=float)
    lo_N = np.asarray(lower_nbar, dtype=float)
    lo_T = np.asarray(lower_T_bath, dtype=float)

    upper_f = np.zeros((up_T.size, up_N.size, NE))
    lower_f = np.zeros((lo_N.size, lo_T.size, NE))
    omega_bins, _, _, _ = build_phonon_frequency_map(spectral.E)
    upper_n_ph = np.zeros((up_T.size, up_N.size, omega_bins.size))
    lower_n_ph = np.zeros((lo_N.size, lo_T.size, omega_bins.size))
    upper_certificates = {
        field: np.zeros((up_T.size, up_N.size), dtype=float)
        for field in CERTIFICATE_FIELDS
    }
    lower_certificates = {
        field: np.zeros((lo_N.size, lo_T.size), dtype=float)
        for field in CERTIFICATE_FIELDS
    }

    print("Upper panel — sweep n̄ at three T_B:")
    for i, T_bath in enumerate(up_T):
        f_seed: np.ndarray | None = None
        n_ph_seed: np.ndarray | None = None
        for j, n_bar in enumerate(up_N):
            state = _build_state(
                material, spectral, float(T_bath), tau_l,
                f_seed=f_seed, n_ph_seed=n_ph_seed,
            )
            photon_params = {
                "omega_0": OMEGA_0, "n_bar": float(n_bar), "c_phot": C_PHOT,
            }
            converged = _solve_picard(backend, state, photon_params)
            upper_f[i, j] = converged.f
            upper_n_ph[i, j] = converged.phonon.n_ph[0, :, 0]
            certificate = steady_state_certificate(
                converged,
                photon_params=photon_params,
                tau_l=tau_l,
            )
            for field in CERTIFICATE_FIELDS:
                upper_certificates[field][i, j] = certificate[field]
            _require_certified_point(
                certificate,
                context=f"upper T_B={float(T_bath):g} K, n_bar={float(n_bar):.6e}",
            )
            f_seed = converged.f.copy()
            n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()
            print(f"  upper  T_B={float(T_bath):.2f} K  n̄={float(n_bar):.2e}", flush=True)

    print("Lower panel — sweep T_B at three n̄:")
    for i, n_bar in enumerate(lo_N):
        f_seed = None
        n_ph_seed = None
        for j, T_bath in enumerate(lo_T):
            state = _build_state(
                material, spectral, float(T_bath), tau_l,
                f_seed=f_seed, n_ph_seed=n_ph_seed,
            )
            photon_params = {
                "omega_0": OMEGA_0, "n_bar": float(n_bar), "c_phot": C_PHOT,
            }
            converged = _solve_picard(backend, state, photon_params)
            lower_f[i, j] = converged.f
            lower_n_ph[i, j] = converged.phonon.n_ph[0, :, 0]
            certificate = steady_state_certificate(
                converged,
                photon_params=photon_params,
                tau_l=tau_l,
            )
            for field in CERTIFICATE_FIELDS:
                lower_certificates[field][i, j] = certificate[field]
            _require_certified_point(
                certificate,
                context=f"lower n_bar={float(n_bar):.6e}, T_B={float(T_bath):g} K",
            )
            f_seed = converged.f.copy()
            n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()
            print(f"  lower  n̄={float(n_bar):.2e}  T_B={float(T_bath):.3f} K", flush=True)

    return {
        "upper_f": upper_f,
        "lower_f": lower_f,
        "upper_n_ph": upper_n_ph,
        "lower_n_ph": lower_n_ph,
        "upper_T_bath": up_T,
        "upper_nbar": up_N,
        "lower_nbar": lo_N,
        "lower_T_bath": lo_T,
        "tau_0_pb_ns": np.asarray([tau_0_pb], dtype=float),
        "tau_l_ns": np.asarray([tau_l], dtype=float),
        "num_bins": np.asarray([int(num_bins)]),
        **{f"upper_{field}": values for field, values in upper_certificates.items()},
        **{f"lower_{field}": values for field, values in lower_certificates.items()},
    }


def _require_certified_point(
    certificate: dict[str, float],
    *,
    context: str,
) -> None:
    """Fail a solve point whose independently rebuilt balances are not roots."""
    qp_backward = float(certificate["qp_backward_error"])
    phonon_backward = float(certificate["phonon_backward_error"])
    if (
        not np.isfinite(qp_backward)
        or not np.isfinite(phonon_backward)
        or qp_backward > TARGET_BACKWARD_ERROR_LIMIT
        or phonon_backward > TARGET_BACKWARD_ERROR_LIMIT
    ):
        raise RuntimeError(
            "Fischer Fig. 5 independent steady-state certificate failed at "
            f"{context} (limit={TARGET_BACKWARD_ERROR_LIMIT:g}): "
            f"qp={qp_backward:.3e}, phonon={phonon_backward:.3e}."
        )


def solver_fingerprint(*, num_bins: int = NUM_BINS) -> dict[str, Any]:
    """Resolved physics + grid knobs, for the cache key's provenance sidecar.

    Safety does not hinge on this being exhaustive: the cache key also folds in
    this module's full source (``extra_source``), the ``qpsim`` solver-subtree
    digest, and the ``run`` kwargs (num_bins + the four sweep axes).
    """
    return {
        "delta_0": DELTA_0,
        "tau_0": TAU_0,
        "t_c": T_C,
        "omega_0": OMEGA_0,
        "c_phot": C_PHOT,
        "e_min_factor": E_MIN_FACTOR,
        "e_max_factor": E_MAX_FACTOR,
        "num_bins": int(num_bins),
        "a_eq35": _A_EQ35,
        "paper_tau_0_pb_ps": PAPER_TAU_0_PB_PS,
        "gap_mode": "fixed",
        "solver": dict(SOLVER_KWARGS),
        "picard_mixing": 0.30,
        "certificate_metric_version": CERTIFICATE_METRIC_VERSION,
        "certificate_fields": list(CERTIFICATE_FIELDS),
        "target_backward_error_limit": TARGET_BACKWARD_ERROR_LIMIT,
    }
