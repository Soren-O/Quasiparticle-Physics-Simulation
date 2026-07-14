"""Mode executors — run a validated setup and return a plain payload.

One function per run mode. Each takes the setup plus two driver
callbacks — ``progress(fraction, message)`` and ``is_cancelled()`` —
and returns a :class:`RunPayload` of NumPy arrays (persisted as NPZ),
a JSON-serializable summary, and human-readable notes (captured
engine warnings, skipped observables, non-converged sweep points).

Cancellation raises :class:`RunCancelledError`; the job runner records
the run as cancelled. Long loops (transient substeps, spatial stepping,
M25 temperature sweeps) check the flag via the services'
``progress_hook`` or between sweep points. Secondary diagnostics never
sink a run: a solve that converged is persisted even when a
post-solve observable fails (the failure becomes a note).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from qpsim.backends.t3_diffusion import T3DiffusionBackend
from qpsim.backends.t3_spatial_1d import T3Spatial1DBackend, T3Spatial1DState
from qpsim.constants import H_OVER_KB_K_PER_HZ
from qpsim.observables import (
    compute_ac_conductivity,
    compute_frequency_shift,
    compute_gap_suppression,
    compute_quality_factor,
    effective_phonon_temperature,
    fermi_dirac_distribution,
    qp_fraction,
    qp_number_density,
)
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights
from qpsim.services.rate_equation import (
    chemical_potentials_kelvin,
    solve_rate_equation_steady_state_multi_seed,
)
from qpsim.services.rate_equation_coefficients import (
    calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00,
    coefficients_from_physical_parameters_with_photon_drive,
)
from qpsim.services.transient import run_time_dependent
from qpsim.webui.builders import (
    build_injection_flux,
    build_m25_inputs,
    build_state_0d,
    build_state_1d,
    drive_dicts,
    mb_probe_invalid_reason,
    steady_state_solver_kwargs,
)
from qpsim.webui.schemas import (
    AnySetup,
    M25JunctionSetup,
    ProbeConfig,
    Spatial1DSetup,
    SteadyState0DSetup,
    Transient0DSetup,
)

ProgressFn = Callable[[float, str], None]
CancelledFn = Callable[[], bool]


class RunCancelledError(Exception):
    """The driver requested cooperative cancellation."""


@dataclass
class RunPayload:
    """Executor output: arrays for NPZ, a JSON summary, and notes."""

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _check_cancel(is_cancelled: CancelledFn) -> None:
    if is_cancelled():
        raise RunCancelledError


def _time_progress_hook(
    progress: ProgressFn, is_cancelled: CancelledFn
) -> Callable[[float, float], bool]:
    """Engine ``progress_hook`` bridging to the job's progress/cancel.

    The message is formatted only when the fraction has advanced
    visibly (≥0.5%) — the hook runs every substep, and a 100k-step run
    should not spend its time building strings nobody reads.
    """
    last_reported = -1.0

    def hook(t: float, total: float) -> bool:
        nonlocal last_reported
        fraction = min(t / total, 1.0)
        if fraction - last_reported >= 0.005 or fraction >= 1.0:
            last_reported = fraction
            progress(fraction, f"t = {t:.4g} / {total:.4g} ns")
        return not is_cancelled()

    return hook


def _mb_observables(
    summary: dict[str, Any],
    notes: list[str],
    f: np.ndarray,
    f_ref: np.ndarray,
    ctx: Any,
    probe: ProbeConfig,
) -> None:
    """Mattis–Bardeen probe block (skipped with a note when invalid)."""
    if not probe.enabled:
        return
    reason = mb_probe_invalid_reason(probe, ctx.dynes_gamma, ctx.gap)
    if reason is not None:
        notes.append(reason)
        return
    sigma1, sigma2 = compute_ac_conductivity(f, ctx, probe.omega_0)
    summary["sigma1_over_sigmaN"] = sigma1
    summary["sigma2_over_sigmaN"] = sigma2
    if sigma1 != 0.0:
        summary["Q_i"] = compute_quality_factor(f, ctx, probe.omega_0, probe.alpha)
        if probe.Q_ext is not None:
            summary["Q_tot"] = compute_quality_factor(
                f, ctx, probe.omega_0, probe.alpha, Q_ext=probe.Q_ext
            )
        if sigma1 < 0.0:
            notes.append(
                "The probe has σ₁ < 0: this is active microwave gain "
                "(negative damping), reported as a signed quality factor."
            )
    else:
        notes.append(
            "Q_i skipped: σ₁ = 0, so the quasiparticle quality factor is "
            "unbounded."
        )
    summary["frac_freq_shift"] = compute_frequency_shift(
        f, f_ref, ctx, probe.omega_0, probe.alpha
    )


def run_steady_state_0d(
    setup: SteadyState0DSetup, progress: ProgressFn, is_cancelled: CancelledFn
) -> RunPayload:
    payload = RunPayload()
    _check_cancel(is_cancelled)
    progress(0.05, "building state")

    state = build_state_0d(setup)
    photon_params, pb_params = drive_dicts(setup)
    kwargs = steady_state_solver_kwargs(setup)
    f_ref = state.f.copy()

    _check_cancel(is_cancelled)
    progress(0.15, "solving steady state")
    backend = T3DiffusionBackend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        solved = backend.steady_state(
            state,
            photon_params=photon_params,
            pb_photon_params=pb_params,
            **kwargs,  # type: ignore[arg-type]
        )
    payload.notes.extend(str(w.message) for w in caught)

    progress(0.85, "computing observables")
    ctx = solved.spectral
    gap0 = setup.material.Delta_0
    payload.arrays["E_bins"] = ctx.E
    payload.arrays["f"] = solved.f
    payload.arrays["f_thermal"] = f_ref
    payload.arrays["n_ph"] = solved.phonon.n_ph[0, :, 0]
    payload.arrays["omega_bins"] = solved.phonon.omega_bins[0]

    summary = payload.summary
    summary["gap_ueV"] = solved.gap
    summary["x_qp"] = qp_fraction(solved.f, ctx, delta_0=gap0)
    summary["x_qp_thermal"] = qp_fraction(f_ref, ctx, delta_0=gap0)
    if setup.material.rho_F > 0.0:
        summary["n_qp_per_m3"] = qp_number_density(solved.f, ctx, setup.material.rho_F)

    _mb_observables(summary, payload.notes, solved.f, f_ref, ctx, setup.probe)

    # Secondary diagnostics: a converged kinetic solve must never be
    # discarded because one of these post-solve fits fails.
    if setup.material.dynes_gamma == 0.0:
        try:
            gs = compute_gap_suppression(
                solved.f, ctx.E, T_c=setup.material.T_c, T_bath=setup.T_bath
            )
        except (ValueError, RuntimeError) as exc:
            payload.notes.append(f"Gap-suppression diagnostic failed: {exc}")
        else:
            summary["delta_eq_ueV"] = gs.delta_eq
            summary["delta_suppression_ueV"] = gs.delta_suppression
            summary["rel_gap_suppression"] = gs.rel_suppression

    if setup.phonons.mode != "thermal_bath":
        try:
            summary["T_phonon_eff_K"] = effective_phonon_temperature(
                payload.arrays["n_ph"],
                payload.arrays["omega_bins"],
                solved.gap,
                T_bath=setup.T_bath,
            )
        except (ValueError, RuntimeError) as exc:
            payload.notes.append(f"Effective-phonon-temperature fit failed: {exc}")

    # Honor a cancel requested during the (uninterruptible) blocking solve,
    # matching run_transient_0d / run_spatial_1d — otherwise the run is
    # persisted as "done" despite the user having cancelled it.
    _check_cancel(is_cancelled)
    progress(1.0, "done")
    return payload


def _transient_observables(
    setup: Transient0DSetup,
) -> tuple[dict[str, Callable[[Any], float]], list[str]]:
    gap = setup.material.Delta_0
    obs: dict[str, Callable[[Any], float]] = {
        "x_qp": lambda s: qp_fraction(s.f, s.spectral, delta_0=gap),
    }
    notes: list[str] = []
    if setup.probe.enabled:
        reason = mb_probe_invalid_reason(setup.probe, setup.material.dynes_gamma, gap)
        if reason is not None:
            notes.append(f"Q_i(t) time series skipped: {reason}")
        else:
            omega, alpha = setup.probe.omega_0, setup.probe.alpha
            obs["Q_i"] = lambda s: compute_quality_factor(s.f, s.spectral, omega, alpha)
    return obs, notes


def run_transient_0d(
    setup: Transient0DSetup, progress: ProgressFn, is_cancelled: CancelledFn
) -> RunPayload:
    payload = RunPayload()
    _check_cancel(is_cancelled)
    progress(0.02, "building state")

    state = build_state_0d(setup)
    photon_params, pb_params = drive_dicts(setup)
    obs, notes = _transient_observables(setup)
    payload.notes.extend(notes)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        result = run_time_dependent(
            state,
            dt=setup.dt,
            total_time=setup.total_time,
            photon_params=photon_params,
            pb_photon_params=pb_params,
            snapshot_interval=setup.snapshot_interval,
            observables=obs,
            stop_tol=setup.stop_tol,
            progress_hook=_time_progress_hook(progress, is_cancelled),
        )
    payload.notes.extend(dict.fromkeys(str(w.message) for w in caught))
    if is_cancelled():
        raise RunCancelledError

    payload.arrays["E_bins"] = state.spectral.E
    payload.arrays["t_ns"] = np.array([s.t for s in result.snapshots])
    payload.arrays["f_snapshots"] = np.stack([s.f for s in result.snapshots])
    payload.arrays["f_thermal"] = fermi_dirac_distribution(state.spectral.E, setup.T_bath)
    for name in obs:
        payload.arrays[f"obs_{name}"] = np.array(
            [s.observables[name] for s in result.snapshots]
        )

    payload.summary["n_steps"] = result.n_steps
    payload.summary["total_time_ns"] = result.total_time
    payload.summary["converged"] = result.converged
    payload.summary["x_qp_final"] = float(payload.arrays["obs_x_qp"][-1])
    payload.summary["gap_ueV"] = setup.material.Delta_0
    progress(1.0, "done")
    return payload


def _xqp_profile(state: T3Spatial1DState, delta_0: float) -> np.ndarray:
    """Per-cell ``x_qp`` using each cell's local BCS spectral measure.

    The numerator uses the local gap from ``gap_profile`` because transport
    does too. The denominator intentionally remains the material reference
    ``delta_0`` so values on different sides of a gap step share one
    dimensionless normalization and can be compared directly.
    """
    if not np.isfinite(delta_0) or delta_0 <= 0.0:
        raise ValueError("delta_0 must be finite and positive.")
    local_gaps = (
        np.full(state.f.shape[1], state.spectral.gap, dtype=float)
        if state.gap_profile is None
        else np.asarray(state.gap_profile, dtype=float)
    )
    if local_gaps.shape != (state.f.shape[1],):
        raise ValueError(
            f"gap_profile must have shape ({state.f.shape[1]},); "
            f"got {local_gaps.shape}."
        )

    weights_by_gap: dict[float, np.ndarray] = {}
    profile = np.empty(state.f.shape[1], dtype=float)
    for column, local_gap in enumerate(local_gaps):
        gap_key = float(local_gap)
        weights = weights_by_gap.get(gap_key)
        if weights is None:
            weights = bcs_dos_cell_weights(
                state.spectral.E,
                state.spectral.dE,
                gap_key,
            )
            weights_by_gap[gap_key] = weights
        profile[column] = float(np.sum(weights * state.f[:, column])) / delta_0
    return profile


def _profile_observables(
    delta_0: float,
) -> dict[str, Callable[[T3Spatial1DState], float]]:
    """Mean/max strip observables sharing one profile pass per state.

    Both reductions are evaluated back-to-back on the same state at
    each snapshot; caching on the state's ``f`` identity halves the
    O(NE·NX) profile cost without assuming anything about call order.
    """
    cached_f: np.ndarray | None = None
    cached_profile: np.ndarray | None = None

    def profile(s: T3Spatial1DState) -> np.ndarray:
        nonlocal cached_f, cached_profile
        if cached_profile is None or cached_f is not s.f:
            cached_f = s.f
            cached_profile = _xqp_profile(s, delta_0)
        return cached_profile

    return {
        "x_qp_mean": lambda s: float(np.mean(profile(s))),
        "x_qp_max": lambda s: float(np.max(profile(s))),
    }


def run_spatial_1d(
    setup: Spatial1DSetup, progress: ProgressFn, is_cancelled: CancelledFn
) -> RunPayload:
    payload = RunPayload()
    _check_cancel(is_cancelled)
    progress(0.02, "building strip state")

    state = build_state_1d(setup)
    flux = build_injection_flux(setup, state)
    gap = setup.material.Delta_0
    obs = _profile_observables(gap)

    backend = T3Spatial1DBackend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        result = backend.run_until_steady_state(
            state,
            dt=setup.dt,
            max_time=setup.max_time,
            external_flux=flux,
            stop_tol=setup.stop_tol,
            snapshot_interval=setup.snapshot_interval,
            observables=obs,
            progress_hook=_time_progress_hook(progress, is_cancelled),
        )
    payload.notes.extend(dict.fromkeys(str(w.message) for w in caught))
    if is_cancelled():
        raise RunCancelledError

    final = result.state
    payload.arrays["E_bins"] = final.spectral.E
    payload.arrays["x_um"] = final.x
    payload.arrays["f_final"] = final.f
    payload.arrays["f_thermal"] = fermi_dirac_distribution(final.spectral.E, setup.T_bath)
    payload.arrays["xqp_profile"] = _xqp_profile(final, gap)
    if final.gap_profile is not None:
        payload.arrays["gap_profile"] = np.asarray(final.gap_profile)
    payload.arrays["snap_t_ns"] = np.array([s.t for s in result.snapshots])
    payload.arrays["snap_max_rate"] = np.array([s.max_rate for s in result.snapshots])
    for name in obs:
        payload.arrays[f"obs_{name}"] = np.array(
            [s.observables.get(name, np.nan) for s in result.snapshots]
        )

    payload.summary["converged"] = result.converged
    payload.summary["n_steps"] = result.n_steps
    payload.summary["total_time_ns"] = result.total_time
    payload.summary["x_qp_mean"] = float(np.mean(payload.arrays["xqp_profile"]))
    payload.summary["x_qp_max"] = float(np.max(payload.arrays["xqp_profile"]))
    payload.summary["gap_ueV"] = gap
    if not result.converged:
        payload.notes.append(
            f"Strip did not reach stop_tol = {setup.stop_tol:g} within "
            f"{setup.max_time:g} ns (max|df/dt| is in the convergence plot)."
        )
    progress(1.0, "done")
    return payload


def run_m25_junction(
    setup: M25JunctionSetup, progress: ProgressFn, is_cancelled: CancelledFn
) -> RunPayload:
    payload = RunPayload()
    T_kelvin = np.linspace(
        setup.T_start_mK / 1000.0, setup.T_stop_mK / 1000.0, setup.T_points
    )
    n = T_kelvin.size
    x_L = np.full(n, np.nan)
    x_Rgt = np.full(n, np.nan)
    x_Rlt = np.full(n, np.nan)
    p_1 = np.full(n, np.nan)
    residual = np.full(n, np.nan)

    ghz_to_K = 1e9 * H_OVER_KB_K_PER_HZ
    Delta_L_K = (setup.Delta_R_over_h_GHz + setup.omega_LR_over_h_GHz) * ghz_to_K
    Delta_R_K = setup.Delta_R_over_h_GHz * ghz_to_K

    last_y: np.ndarray | None = None
    prev_prev_y: np.ndarray | None = None
    failed: list[float] = []
    for i, T in enumerate(T_kelvin):
        _check_cancel(is_cancelled)
        progress(i / n, f"T = {T * 1000:.1f} mK")
        try:
            params, drive = build_m25_inputs(setup, float(T))
            scale = calibrate_Gamma_nu_scale_Hz_from_Gamma_ph_00(
                params, drive, setup.drive.Gamma_ph_00_Hz
            )
            coefs = coefficients_from_physical_parameters_with_photon_drive(
                params, replace(drive, Gamma_nu_scale_Hz=scale)
            )
            # Continuation seeding as in the Fig. 3 reproduction: warm
            # start from the previous point, plus a linear (second-order)
            # predictor once two points exist — it carries the solver
            # across sharp T dependence a bare prev-T seed misses
            # (historically, the multi-stable kinks of the
            # unnormalized system).
            if last_y is None:
                sol = solve_rate_equation_steady_state_multi_seed(coefs)
            else:
                extra: list[np.ndarray] = []
                if prev_prev_y is not None and i >= 2:
                    y_pred = last_y + (last_y - prev_prev_y)
                    if np.all(np.isfinite(y_pred)) and np.all(y_pred > 0.0):
                        extra.append(y_pred)
                sol = solve_rate_equation_steady_state_multi_seed(
                    coefs,
                    preferred_seed=last_y,
                    extra_seeds=extra or None,
                    branch_picker_mode=setup.branch_picker_mode,
                )
        except (RuntimeError, ValueError) as exc:
            failed.append(float(T))
            payload.notes.append(f"T = {T * 1000:.1f} mK failed: {exc}")
            continue
        prev_prev_y = last_y
        last_y = np.array([sol.p_1, sol.x_L, sol.x_Rgt, sol.x_Rlt])
        x_L[i], x_Rgt[i], x_Rlt[i] = sol.x_L, sol.x_Rgt, sol.x_Rlt
        p_1[i] = sol.p_1
        residual[i] = sol.residual_inf_norm

    # Paper-exact μ inversion (M25 SI Eqs. S2/S4/S5) — the naive
    # μ = Δ + T·ln(x) drops the √(Δ/2πT) prefactor and the erf/erfc
    # sub-band partition, which inverts the μ_R> vs μ_R< ordering.
    # Inputs are already in Kelvin here; NaN (failed sweep points)
    # passes through as NaN.
    mu_L, mu_Rgt, mu_Rlt = chemical_potentials_kelvin(
        Delta_L_kelvin=Delta_L_K,
        Delta_R_kelvin=Delta_R_K,
        T_kelvin=T_kelvin,
        x_L=x_L,
        x_Rgt=x_Rgt,
        x_Rlt=x_Rlt,
    )

    payload.arrays["T_mK"] = T_kelvin * 1000.0
    payload.arrays["x_L"] = x_L
    payload.arrays["x_Rgt"] = x_Rgt
    payload.arrays["x_Rlt"] = x_Rlt
    payload.arrays["p_1"] = p_1
    payload.arrays["residual_Hz"] = residual
    payload.arrays["mu_L_over_Delta_L"] = mu_L / Delta_L_K
    payload.arrays["mu_Rgt_over_Delta_L"] = mu_Rgt / Delta_L_K
    payload.arrays["mu_Rlt_over_Delta_L"] = mu_Rlt / Delta_L_K

    ok = int(np.sum(np.isfinite(x_L)))
    payload.summary["points_converged"] = ok
    payload.summary["points_total"] = int(n)
    payload.summary["Delta_L_over_h_GHz"] = Delta_L_K / ghz_to_K
    payload.summary["Delta_R_over_h_GHz"] = setup.Delta_R_over_h_GHz
    if ok:
        j = int(np.flatnonzero(np.isfinite(x_L))[0])
        payload.summary["x_L_lowT"] = float(x_L[j])
        payload.summary["p_1_lowT"] = float(p_1[j])
    if failed:
        payload.notes.append(
            f"{len(failed)}/{n} sweep points did not converge. For M25-like parameters "
            f"the steady state has a single physical root, so a failure indicates "
            f"numerical non-convergence rather than branch ambiguity — try a denser "
            f"temperature grid (better warm starts) or a different seed; far from the "
            f"paper's parameter regime multiple roots remain possible."
        )
    # Honor a cancel requested during the final sweep point's solve, which the
    # per-iteration top-of-loop check cannot catch (there is no next iteration).
    _check_cancel(is_cancelled)
    progress(1.0, "done")
    return payload


def execute_setup(
    setup: AnySetup, progress: ProgressFn, is_cancelled: CancelledFn
) -> RunPayload:
    """Dispatch a validated setup to its mode executor."""
    if isinstance(setup, SteadyState0DSetup):
        return run_steady_state_0d(setup, progress, is_cancelled)
    if isinstance(setup, Transient0DSetup):
        return run_transient_0d(setup, progress, is_cancelled)
    if isinstance(setup, Spatial1DSetup):
        return run_spatial_1d(setup, progress, is_cancelled)
    return run_m25_junction(setup, progress, is_cancelled)
