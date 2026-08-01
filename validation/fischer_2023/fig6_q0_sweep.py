"""Q0 diagnostic sweep: the author-semantics qpsim endpoint over Figure 6.

This script drives qpsim's public ``coupled_newton_solve`` over the complete
authenticated Figure 6 sweep -- ``n_bar = geomspace(1e4, 1e8, 100)`` for bath
temperatures ``{0.10, 0.15, 0.20} K`` -- under the accepted C4/C5/C6 operator
configuration (public photon channel, QP-side kernels, phonon-side balance
with the Kaplan ``S_+`` correction) and the author model semantics (fixed
kinetic gap, direct post-processed gap observable).  Each temperature starts
from the native thermal state and continues point-to-point, mirroring the
author sweep flow.

Every point records the Figure 6 ordinate under BOTH sampling conventions:

* ``authors`` -- the solved occupation read at the author left-edge nodes,
  the convention the published figure used;
* ``centers`` -- the same occupation read at qpsim's cell centers.

An optional ``--de-scale`` refines the grid (``dE = 1/scale`` micro-eV) so
the convention gap can be shown to close under refinement.

This is a provenance-recorded DIAGNOSTIC artifact, not a formal ladder
stage: it makes no certified per-point verification claim, and its curves
must not be promoted to paper-parity evidence without the ladder's
independent-verifier treatment.  Progress streams to ``<output>.jsonl`` one
completed point per line; the final JSON binds the source closure, runtime,
solver configuration, and every per-point record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_base,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.observables.gap_suppression import gap_from_distribution_direct
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from qpsim.solvers.coupled_newton import coupled_newton_solve

from validation.fischer_2023.fig6_author_c2_parameters import (
    NativeFig6Parameters,
)
from validation.source_provenance import canonical_source_bytes, source_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-q0-diagnostic-sweep.v1"

GAP_UEV = 180.0
GAP_EV = 180 * 10**-6
DELTA0_EV = GAP_EV
T_C_K = 1.184309192877208
TAU_0_NS = 438.0
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
OMEGA_0_UEV = 20.0
C_PHOT_NS_INV = 1.0e-9
STEP_RTOL = 1.0e-7
E_MIN_FACE_UEV = 160.0
E_MAX_FACE_UEV = 1800.0
GUARD_FACE_UEV = 180.0
T_LIST = (0.1, 0.15, 0.2)
N_SWEEP = 100

_SOURCE_HASHES = source_manifest(
    Path(__file__),
    extra_validation_modules=(
        REPOSITORY_ROOT / "validation" / "__init__.py",
        REPOSITORY_ROOT / "validation" / "source_provenance.py",
        REPOSITORY_ROOT / "validation" / "fischer_2023" / "__init__.py",
        REPOSITORY_ROOT
        / "validation"
        / "fischer_2023"
        / "fig6_author_c2_parameters.py",
    ),
)


def _runtime_record() -> dict[str, object]:
    return {
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
    }


def _build_grid(de_scale: int) -> tuple[SpectralContext, dict[str, np.ndarray]]:
    h_ueV = 1.0 / de_scale
    n_cells = round((E_MAX_FACE_UEV - E_MIN_FACE_UEV) / h_ueV)
    faces = E_MIN_FACE_UEV + h_ueV * np.arange(n_cells + 1, dtype=np.float64)
    centers = faces[:-1] + 0.5 * h_ueV
    dE = np.full(n_cells, h_ueV, dtype=np.float64)
    ctx = SpectralContext(centers, dE, GAP_UEV)
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(centers)
    return ctx, {
        "centers": centers,
        "dE": dE,
        "omega": omega,
        "idx_diff": idx_diff,
        "idx_sum": idx_sum,
        "diff_sign": diff_sign,
    }


def _active_author_slice(de_scale: int) -> slice:
    guard_cells = round((GUARD_FACE_UEV - E_MIN_FACE_UEV) * de_scale)
    return slice(guard_cells, None)


def _ordinates(
    f_solved: np.ndarray,
    thermal_left: np.ndarray,
    thermal_center: np.ndarray,
    *,
    de_scale: int,
) -> dict[str, float]:
    h_eV = 1.0e-6 / de_scale
    active = _active_author_slice(de_scale)
    n_active = f_solved[active].size
    left_eV = GAP_EV + h_eV * np.arange(n_active, dtype=np.float64)
    carrier_eV = left_eV + 0.5 * h_eV

    def gap(f_active: np.ndarray, samples: str) -> float:
        return gap_from_distribution_direct(
            f_active,
            carrier_eV,
            gap=GAP_EV,
            delta0=DELTA0_EV,
            samples=samples,
        )

    driven_authors = gap(f_solved[active], "authors")
    thermal_authors = gap(thermal_left[active], "authors")
    driven_centers = gap(f_solved[active], "centers")
    thermal_centers = gap(thermal_center[active], "centers")
    return {
        "driven_gap_eV_authors": driven_authors,
        "driven_gap_eV_centers": driven_centers,
        "ordinate_authors": (driven_authors - thermal_authors)
        / (DELTA0_EV - thermal_authors),
        "ordinate_centers": (driven_centers - thermal_centers)
        / (DELTA0_EV - thermal_centers),
        "thermal_gap_eV_authors": thermal_authors,
        "thermal_gap_eV_centers": thermal_centers,
    }


def _balance_ratios(
    f_state: np.ndarray,
    n_state: np.ndarray,
    *,
    ctx: SpectralContext,
    grid: dict[str, np.ndarray],
    K_s0: np.ndarray,
    K_r0: np.ndarray,
    K_s_ph: np.ndarray,
    K_r_ph: np.ndarray,
    n_th: np.ndarray,
    n_bar: float,
    T_bath: float,
) -> dict[str, float]:
    """Measured L1 residual-to-turnover ratios at a candidate root."""

    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_state, grid["idx_diff"], grid["idx_sum"], grid["diff_sign"],
    )
    gain, loss_rate = phonon_collision_rates(
        f_state, ctx, K_s0, K_r0, T_bath,
        N_p_override=N_p, N_emit_override=N_emit, N_abs_override=N_abs,
    )
    gain_ph, loss_ph = sub_gap_photon_collision_rates(
        f_state, ctx, OMEGA_0_UEV, n_bar, C_PHOT_NS_INV,
    )
    gain = gain + gain_ph
    loss_rate = loss_rate + loss_ph
    unsupported = ~ctx.active_mask
    gain[unsupported] = 0.0
    loss_rate[unsupported] = 0.0
    R_f = gain - loss_rate * f_state
    a_ph, b_ph = compute_phonon_source_sink(
        f_state, ctx, K_s0, K_r0,
        grid["idx_diff"], grid["idx_sum"], grid["diff_sign"],
        int(grid["omega"].size),
        K_s0_phonon_side=K_s_ph, K_r0_phonon_side=K_r_ph,
    )
    driven = b_ph * n_state
    bath = (n_th - n_state) * (1.0 / TAU_L_NS)
    R_ph = a_ph + driven + bath
    tiny = float(np.finfo(np.float64).tiny)
    qp_ratio = float(np.sum(np.abs(R_f))) / max(
        float(np.sum(np.abs(gain) + np.abs(loss_rate * f_state))), tiny
    )
    ph_ratio = float(np.sum(np.abs(R_ph))) / max(
        float(np.sum(np.abs(a_ph)) + np.sum(np.abs(driven)) + np.sum(np.abs(bath))),
        tiny,
    )
    return {
        "measured_ph_balance_l1_ratio": ph_ratio,
        "measured_qp_balance_l1_ratio": qp_ratio,
    }


def run_sweep(
    *,
    de_scale: int,
    temperatures: tuple[float, ...],
    indices: tuple[int, ...],
    max_iter: int,
    output: Path,
    step_rtol: float = STEP_RTOL,
    tol: float = 1.0e-10,
) -> Path:
    ctx, grid = _build_grid(de_scale)
    centers = grid["centers"]
    omega = grid["omega"]
    h_ueV = 1.0 / de_scale
    active = _active_author_slice(de_scale)
    n_active = centers[active].size
    left_ueV = GAP_UEV + h_ueV * np.arange(n_active, dtype=np.float64)
    photon_bin = round(OMEGA_0_UEV / h_ueV)

    K_s0 = build_scattering_kernel_base(ctx, TAU_0_NS, T_C_K)
    K_r0 = build_recombination_kernel_base(ctx, TAU_0_NS, T_C_K)
    K_s_ph = build_scattering_kernel_phonon_side(ctx, TAU_PB_NS)
    K_r_ph = build_recombination_kernel_phonon_side(ctx, TAU_PB_NS)
    n_bar_ladder = np.geomspace(10.0**4, 10.0**8, N_SWEEP)

    progress_path = output.with_suffix(output.suffix + ".jsonl")
    records: list[dict[str, Any]] = []
    started = time.time()
    with progress_path.open("w", encoding="utf-8") as progress:
        for T_bath in temperatures:
            f_thermal_center = fermi_dirac_occupation(centers, T_bath)
            f_thermal_center = np.where(
                ctx.active_mask, f_thermal_center, 0.0
            )
            thermal_left_active = fermi_dirac_occupation(left_ueV, T_bath)
            thermal_left = np.zeros_like(f_thermal_center)
            thermal_left[active] = thermal_left_active
            n_thermal = thermal_phonon_occupation(omega, T_bath)
            f_seed = f_thermal_center.copy()
            n_seed = n_thermal.copy()
            for index in indices:
                n_bar = float(n_bar_ladder[index])
                parameters = NativeFig6Parameters(
                    gap_ueV=GAP_UEV,
                    delta0_ueV=GAP_UEV,
                    h_ueV=h_ueV,
                    temperature_K=T_bath,
                    T_c_K=T_C_K,
                    tau_0_ns=TAU_0_NS,
                    tau_0_pb_ns=TAU_PB_NS,
                    tau_l_ns=TAU_L_NS,
                    photon_bin=photon_bin,
                    n_bar=n_bar,
                    c_photon_ns_inv=C_PHOT_NS_INV,
                    kB_ueV_per_K=86.17333262145,
                )
                point: dict[str, Any] = {
                    "T_bath_K": T_bath,
                    "de_scale": de_scale,
                    "eq35_t_star_over_delta": parameters.eq35_t_star_over_delta,
                    "n_bar": n_bar,
                    "sweep_index": index,
                }
                wall = time.time()
                try:
                    f_root, n_root = coupled_newton_solve(
                        ctx,
                        f_seed,
                        n_seed,
                        omega_bins=omega,
                        omega_idx_diff=grid["idx_diff"],
                        omega_idx_sum=grid["idx_sum"],
                        diff_sign=grid["diff_sign"],
                        K_s0=K_s0,
                        K_r0=K_r0,
                        K_s0_phonon_side=K_s_ph,
                        K_r0_phonon_side=K_r_ph,
                        T_bath=T_bath,
                        tau_l=TAU_L_NS,
                        photon_params={
                            "omega_0": OMEGA_0_UEV,
                            "n_bar": n_bar,
                            "c_phot": C_PHOT_NS_INV,
                        },
                        step_rtol=step_rtol,
                        tol=tol,
                        max_iter=max_iter,
                        analytic_cross=True,
                    )
                except (RuntimeError, ValueError) as exc:
                    point["converged"] = False
                    point["exception"] = f"{type(exc).__name__}: {exc}"
                    point["wall_seconds"] = time.time() - wall
                else:
                    point["converged"] = True
                    point["wall_seconds"] = time.time() - wall
                    point.update(
                        _balance_ratios(
                            f_root,
                            n_root,
                            ctx=ctx,
                            grid=grid,
                            K_s0=K_s0,
                            K_r0=K_r0,
                            K_s_ph=K_s_ph,
                            K_r_ph=K_r_ph,
                            n_th=n_thermal,
                            n_bar=n_bar,
                            T_bath=T_bath,
                        )
                    )
                    point.update(
                        _ordinates(
                            f_root,
                            thermal_left,
                            f_thermal_center,
                            de_scale=de_scale,
                        )
                    )
                    f_seed, n_seed = f_root, n_root
                records.append(point)
                progress.write(json.dumps(point, sort_keys=True) + "\n")
                progress.flush()

    artifact = {
        "grid": {
            "de_scale": de_scale,
            "faces_ueV": [E_MIN_FACE_UEV, E_MAX_FACE_UEV],
            "h_ueV": h_ueV,
            "n_cells": int(centers.size),
            "n_omega": int(omega.size),
            "photon_bin": photon_bin,
        },
        "points": records,
        "qualification": (
            "Provenance-recorded diagnostic sweep of the author-semantics "
            "qpsim endpoint under both sampling conventions; not a formal "
            "reproduction-ladder stage and not paper-parity evidence."
        ),
        "runtime": _runtime_record(),
        "schema": SCHEMA,
        "seeding": (
            "per temperature: native thermal state, then point-to-point "
            "continuation in ascending n_bar"
        ),
        "solver": {
            "analytic_cross": True,
            "max_iter": max_iter,
            "step_rtol": step_rtol,
            "tol": tol,
            "tau_l_ns": TAU_L_NS,
        },
        "sources": dict(_SOURCE_HASHES),
        "sweep": {
            "n_bar_geomspace": [1.0e4, 1.0e8, N_SWEEP],
            "temperatures_K": list(temperatures),
        },
        "total_wall_seconds": time.time() - started,
    }
    output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"{output} sha256={digest}")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--de-scale", type=int, default=1)
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=list(T_LIST),
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=list(range(N_SWEEP)),
    )
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--step-rtol", type=float, default=STEP_RTOL)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for path in _SOURCE_HASHES:
        canonical_source_bytes(REPOSITORY_ROOT / path)
    run_sweep(
        de_scale=args.de_scale,
        temperatures=tuple(args.temperatures),
        indices=tuple(args.indices),
        max_iter=args.max_iter,
        output=args.output,
        step_rtol=args.step_rtol,
        tol=args.tol,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
