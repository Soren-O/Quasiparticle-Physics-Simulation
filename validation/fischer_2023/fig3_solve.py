"""Fischer 2023 Fig. 3 — the expensive continuation-ladder solve (cache payload).

Split out from :mod:`fig3_paper` so the config+code-hashed sweep cache
(:mod:`validation.sweep_cache`) can hash *this whole module* as the figure's
solve source: every solve-affecting constant, helper, and solver knob lives
here, so editing the plotting / envelope-overlay code in :mod:`fig3_paper` keeps
a cached solve warm while any change here correctly invalidates it. The whole
module is passed as ``extra_source`` (not ``inspect.getsource(solve)``, which
would miss the module globals ``solve`` reads and the helpers it delegates to).

Fig. 3's plotted data *is* the converged ``f(E)`` per ratio, so :func:`solve`
returns it directly as the raw payload and :func:`fig3_paper.observables` just
repackages it; the value of caching here is skipping the τ_l = 0 Newton solve
plus the multi-minute continuation ladder when re-plotting.

The sweep is parameterized by ``num_bins`` / ratio sets (defaulting to the paper
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

# ── Fischer 2023 Table I parameters ──────────────────────────────────

DELTA_0 = 180.0            # μeV
TAU_0 = 438.0              # ns
T_C = DELTA_0 / (1.764 * KB_UEV_PER_K)
T_BATH = 0.1               # K
OMEGA_0 = DELTA_0 / 9.0    # 20 μeV (sub-gap, integer-commensurate with dE)
N_BAR = 1e7
C_PHOT = 1e-9              # ns^-1 (1 Hz)

# Paper grid: 1620 bins, dE = 1 μeV. ω_0 / dE = 20 (integer).
E_MIN_FACTOR = 1.0
E_MAX_FACTOR = 10.0
NUM_BINS = 1620

# Paper-target legend ratios (Fischer 2023 Fig. 3).
PAPER_RATIOS: tuple[float, ...] = (0.0, 0.1, 1.0, 10.0)

# Continuation ladder: smooth ramp through intermediate ratios so the
# Picard fixed point stays in the basin of attraction. Targets pulled
# out and stored; non-target ratios are discarded after the continuation
# step. Picard struggles above ratio ~5 (the map is non-contractile near
# the strong-bottleneck branch), so the final 5 → 10 step switches to
# coupled Newton on the joint (f, n_ph) state.
CONTINUATION_RATIOS: tuple[float, ...] = (
    0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0,
)

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the numerical tau_0^PB diverges from the paper-quoted 255 ps."""


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
    """Build the paper-grid energy axis + dE widths + spectral context."""
    E, _ = build_energy_grid(
        gap=DELTA_0,
        energy_min_factor=E_MIN_FACTOR,
        energy_max_factor=E_MAX_FACTOR,
        num_energy_bins=num_bins,
    )
    dE = integration_widths_from_centers(E)
    dE_scalar = float(dE[0])
    m = round(OMEGA_0 / dE_scalar)
    frac_err = abs(OMEGA_0 - m * dE_scalar) / OMEGA_0
    if frac_err > 1e-10:
        raise RuntimeError(
            f"Fischer Fig. 3 paper grid not commensurate: "
            f"ω_0 = {OMEGA_0}, m·dE = {m * dE_scalar}, frac_err = {frac_err}."
        )
    spectral = SpectralContext(E_bins=E, dE_bins=dE, gap=DELTA_0)
    return E, dE, spectral


def _compute_tau_0_pb(spectral: SpectralContext) -> float:
    """Numerical τ_0^PB from the simulator's phonon-side kernel at f=0, ω≈2Δ.

    Uses the F&C 2023 Eq. 12 phonon-side kernel ``K⁺/(π Δ τ_0^PB)``
    (built via :func:`build_recombination_kernel_phonon_side`).
    """
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
    f_seed: np.ndarray,
    tau_l_scalar: float,
    *,
    n_ph_seed: np.ndarray | None = None,
) -> T3DiffusionState:
    """Build a T3 state with the given τ_l scalar + (f, n_ph) seeds."""
    omega, _, _, _ = build_phonon_frequency_map(spectral.E)
    if n_ph_seed is None:
        n_ph_seed = thermal_phonon_occupation(omega, T_BATH)
    phonon = PhononState(
        n_ph=n_ph_seed.reshape(1, -1, 1).copy(),
        omega_bins=omega.reshape(1, -1),
        tau_l=np.full((1, omega.size), tau_l_scalar),
        model=PhononModel.PH0_LOCAL,
        branches=[PhononBranchSpec(name="debye_average")],
    )
    return T3DiffusionState(
        f=f_seed.copy(),
        gap=DELTA_0,
        spectral=spectral,
        phonon=phonon,
        material=material,
        T_bath=T_BATH,
    )


def _solve_tau_l_zero(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
) -> T3DiffusionState:
    """τ_l = 0: thermal-phonon shortcut. Newton-on-f only."""
    return backend.steady_state(
        state,
        use_thermal_phonons=True,
        photon_params=photon_params,
        newton_tol=1e-12,
        newton_max_iter=500,
    )


def _solve_picard(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    mixing: float,
) -> T3DiffusionState:
    """Picard + Anderson on (f, n_ph). Mixing under-relaxed at high ratios."""
    return backend.steady_state(
        state,
        method="picard",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        picard_tol=1e-8,
        picard_max_iter=10000,
        picard_mixing=mixing,
        anderson_depth=0,
        newton_tol=1e-12,
        newton_max_iter=500,
    )


def _solve_coupled_newton(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
) -> T3DiffusionState:
    """Coupled Newton on the joint (f, n_ph) vector (strong-bottleneck branch)."""
    return backend.steady_state(
        state,
        method="coupled_newton",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        coupled_newton_tol=1e-10,
        coupled_newton_max_iter=50,
        coupled_newton_fd_step=1e-8,
    )


def _picard_mixing_for_ratio(ratio: float) -> float:
    return 0.15 if ratio > 2.0 else 0.30


def solve(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
) -> dict[str, np.ndarray]:
    """Solve Fischer Fig. 3 at all paper ratios via continuation.

    The expensive half of Fig. 3: the τ_l = 0 thermal-phonon Newton solve plus
    the warm-seeded continuation ladder (Picard up to ratio ~5, coupled Newton
    above). Returns the converged ``f(E)`` per paper ratio as the raw cache
    payload; :func:`fig3_paper.observables` repackages it into ``Fig3PaperResult``.
    """
    material = _fischer_material()
    E, _, spectral = _build_grid_and_spectral(num_bins)

    tau_0_pb = _compute_tau_0_pb(spectral)
    tau_0_pb_ps = tau_0_pb * 1000.0  # ns → ps
    ratio_paper = tau_0_pb_ps / PAPER_TAU_0_PB_PS
    print(f"  τ_0^PB (phonon-side extracted)       = {tau_0_pb:.4f} ns "
          f"({tau_0_pb_ps:.1f} ps)")
    print(f"  Paper-quoted τ_0^PB                   ≈ {PAPER_TAU_0_PB_PS:.0f} ps")
    if ratio_paper > TAU_0_PB_WARN_FACTOR or ratio_paper < 1.0 / TAU_0_PB_WARN_FACTOR:
        print(
            f"  ⚠ τ_0^PB normalization mismatch: extracted/paper = {ratio_paper:.2f}×.",
            flush=True,
        )

    kT = KB_UEV_PER_K * T_BATH
    f_FD = 1.0 / (np.exp(np.minimum(E / kT, 500.0)) + 1.0)

    backend = T3DiffusionBackend()
    photon_params = {"omega_0": OMEGA_0, "n_bar": N_BAR, "c_phot": C_PHOT}

    f_by_ratio: dict[float, np.ndarray] = {}

    # ── ratio 0: thermal-phonon shortcut (Newton-only, paper τ_l=0 curve) ──
    state0 = _build_state(material, spectral, f_FD, tau_l_scalar=0.0)
    print("  τ_l/τ_0^PB = 0     → target  thermal-phonon shortcut", flush=True)
    converged0 = _solve_tau_l_zero(backend, state0, photon_params)
    f_by_ratio[0.0] = converged0.f.copy()

    # ── continuation ladder for finite ratios ──
    f_seed = converged0.f.copy()
    n_ph_seed: np.ndarray | None = None  # bath thermal at first finite step
    for ratio in continuation_ratios:
        tau_l = ratio * tau_0_pb
        state = _build_state(
            material, spectral, f_seed, tau_l, n_ph_seed=n_ph_seed,
        )
        is_target = ratio in paper_ratios
        tag = "→ target  " if is_target else "(continuation)"

        if ratio > 5.0:
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} coupled_newton "
                f"(seeded from prior-ratio (f, n_ph))",
                flush=True,
            )
            converged = _solve_coupled_newton(backend, state, photon_params)
        else:
            mixing = _picard_mixing_for_ratio(ratio)
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} picard "
                f"(mixing={mixing}, AA=0)",
                flush=True,
            )
            converged = _solve_picard(backend, state, photon_params, mixing=mixing)

        if is_target:
            f_by_ratio[ratio] = converged.f.copy()
        f_seed = converged.f.copy()
        n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()

    # Sanity-check all paper ratios captured.
    missing = [r for r in paper_ratios if r not in f_by_ratio]
    if missing:
        raise RuntimeError(f"Continuation did not capture paper ratios: {missing}")

    f_ratios = np.stack([f_by_ratio[r] for r in paper_ratios], axis=0)
    return {
        "E": E,
        "f_FD": f_FD,
        "f_ratios": f_ratios,
        "ratios": np.asarray(paper_ratios, dtype=float),
        "tau_0_pb_ns": np.asarray([tau_0_pb], dtype=float),
    }


def solver_fingerprint(*, num_bins: int = NUM_BINS) -> dict[str, Any]:
    """Resolved physics + grid knobs, for the cache key's provenance sidecar.

    Safety does not hinge on this being exhaustive: the cache key also folds in
    this module's full source (``extra_source``), the ``qpsim`` solver-subtree
    digest, and the ``run`` kwargs (num_bins + ratio sets). This keeps the stored
    provenance human-legible.
    """
    return {
        "delta_0": DELTA_0,
        "tau_0": TAU_0,
        "t_bath": T_BATH,
        "omega_0": OMEGA_0,
        "n_bar": N_BAR,
        "c_phot": C_PHOT,
        "e_min_factor": E_MIN_FACTOR,
        "e_max_factor": E_MAX_FACTOR,
        "num_bins": int(num_bins),
        "paper_tau_0_pb_ps": PAPER_TAU_0_PB_PS,
    }
