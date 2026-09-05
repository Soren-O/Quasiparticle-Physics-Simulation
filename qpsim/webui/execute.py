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

from qpsim.backends.diffusion import DiffusionBackend
from qpsim.backends.spatial import SpatialBackend, SpatialState
from qpsim.constants import H_OVER_KB_K_PER_HZ
from qpsim.devices.external_flux import ExternalFlux
from qpsim.fields.drive import StaticDrive, SumDrive
from qpsim.grid.spatial_grid import reconstruct_field
from qpsim.observables import (
    compute_ac_conductivity,
    compute_frequency_shift,
    compute_gap_suppression,
    compute_quality_factor,
    effective_phonon_temperature,
    fermi_dirac_distribution,
    qp_fraction,
    qp_fraction_paper,
    qp_number_density,
)
from qpsim.physics.bcs_quadrature import bcs_dos_cell_weights, bcs_energy_cell_weights
from qpsim.physics.gap_equation import calibrate_gap
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
    build_drives_2d,
    build_geometry_2d,
    build_initial_state_2d,
    build_injection_2d,
    build_m25_inputs,
    build_phonon_seed_2d,
    build_state_0d,
    build_state_2d,
    drive_dicts,
    injection_line,
    mb_probe_invalid_reason,
    steady_state_solver_kwargs,
)
from qpsim.webui.schemas import (
    AnySetup,
    KineticsSetup,
    M25JunctionSetup,
    ProbeConfig,
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
    setup: KineticsSetup,
    progress: ProgressFn,
    is_cancelled: CancelledFn,
) -> RunPayload:
    """The steady-state strategy: a root find on a single cell.

    Named for what it solves: ``DiffusionBackend.steady_state`` carries
    f:(NE,) and a scalar gap, so its state has no cell axis at all. It is
    reached only through :func:`run_kinetics`, which is why a one-cell mask is
    a precondition rather than a coincidence.
    """
    payload = RunPayload()
    _check_cancel(is_cancelled)
    progress(0.05, "building state")

    state = build_state_0d(setup)
    photon_params, pb_params = drive_dicts(setup)
    kwargs = steady_state_solver_kwargs(setup)
    f_ref = state.f.copy()
    # A STATIC injection is a source with no time axis, which is exactly what
    # a root find can carry: the same Gaussian line the time march spreads
    # over its cells, here on the single cell, as the solver's ExternalFlux.
    # Before this the field validated, was displayed as on, and was dropped.
    flux = None
    if setup.injection.enabled and setup.injection.rate_per_ns > 0.0:
        gain = setup.injection.rate_per_ns * injection_line(setup, state.spectral.E)
        flux = ExternalFlux(gain=gain, loss_rate=np.zeros_like(gain))

    _check_cancel(is_cancelled)
    progress(0.15, "solving steady state")
    backend = DiffusionBackend()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        solved = backend.steady_state(
            state,
            photon_params=photon_params,
            pb_photon_params=pb_params,
            external_flux=flux,
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
    summary["x_qp_paper"] = qp_fraction_paper(solved.f, ctx, delta_0=gap0)
    summary["x_qp_thermal_paper"] = qp_fraction_paper(
        f_ref, ctx, delta_0=gap0
    )
    summary["x_qp_convention"] = X_QP_CONVENTION
    summary["injection_rate_per_ns"] = (
        float(setup.injection.rate_per_ns) if flux is not None else 0.0
    )
    # The mean energy per quasiparticle from the two moments of one
    # quadrature (number and energy weights on the same cells).
    e_weights = bcs_energy_cell_weights(ctx.E, ctx.dE, ctx.gap)
    number = float(np.sum(ctx.cell_weights * solved.f))
    if number > 0.0:
        summary["E_qp_mean_ueV"] = float(np.sum(e_weights * solved.f)) / number
        summary["E_qp_mean_over_gap"] = summary["E_qp_mean_ueV"] / float(ctx.gap)
    if setup.material.rho_F > 0.0:
        summary["n_qp_per_m3"] = qp_number_density(solved.f, ctx, setup.material.rho_F)

    _mb_observables(summary, payload.notes, solved.f, f_ref, ctx, setup.probe)

    # Secondary diagnostics: a converged kinetic solve must never be
    # discarded because one of these post-solve fits fails.  The equilibrium
    # reference is calibration metadata and does not depend on the sampled
    # occupation.  Persist it independently: on a fixed-gap grid the
    # occupation-derived gap can legitimately lie below the represented
    # support and fail closed, while Delta_eq remains well defined.
    if setup.material.dynes_gamma == 0.0:
        try:
            calibration = calibrate_gap(
                T_c=setup.material.T_c,
                T_bath=setup.T_bath,
            )
        except (ValueError, RuntimeError) as exc:
            payload.notes.append(f"Equilibrium-gap calibration failed: {exc}")
        else:
            summary["delta_eq_ueV"] = calibration.delta_eq
            try:
                gs = compute_gap_suppression(
                    solved.f,
                    ctx.E,
                    T_c=setup.material.T_c,
                    T_bath=setup.T_bath,
                )
            except (ValueError, RuntimeError) as exc:
                payload.notes.append(f"Gap-suppression diagnostic failed: {exc}")
            else:
                summary["delta_suppression_ueV"] = gs.delta_suppression
                summary["rel_gap_suppression"] = gs.rel_suppression

    if setup.phonons.mode != "thermal_bath":
        with warnings.catch_warnings(record=True) as fit_warnings:
            warnings.simplefilter("once")
            try:
                summary["T_phonon_eff_K"] = effective_phonon_temperature(
                    payload.arrays["n_ph"],
                    payload.arrays["omega_bins"],
                    solved.gap,
                    T_bath=setup.T_bath,
                )
            except (ValueError, RuntimeError) as exc:
                payload.notes.append(
                    f"Effective-phonon-temperature fit failed: {exc}"
                )
        payload.notes.extend(str(w.message) for w in fit_warnings)

    # Honor a cancel requested during the (uninterruptible) blocking solve,
    # matching the time-march route — otherwise the run is
    # persisted as "done" despite the user having cancelled it.
    _check_cancel(is_cancelled)
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


@dataclass(frozen=True)
class LiveFrame:
    """One recorded frame, handed out WHILE the run is still going.

    The same ``x_qp`` profile the finished run stores in
    ``snap_xqp_profile`` -- computed by the same call at the moment the
    backend records the snapshot -- so what a person watches during a run
    is what the run will have recorded, not a preview of something else.
    """

    t_ns: float
    xqp_profile: np.ndarray       # (Ncells,), mask order
    mask: np.ndarray              # (rows, cols) bool
    mesh_size_um: float
    x_qp_convention: str


FrameFn = Callable[[LiveFrame], None]


def execute_setup(
    setup: AnySetup,
    progress: ProgressFn,
    is_cancelled: CancelledFn,
    *,
    on_frame: FrameFn | None = None,
) -> RunPayload:
    """Dispatch a validated setup to its mode executor.

    ``on_frame`` receives each recorded frame as it is captured (spatial time
    march only); the other routes record no frames and never call it.
    """
    if isinstance(setup, KineticsSetup):
        return run_kinetics(setup, progress, is_cancelled, on_frame=on_frame)
    return run_m25_junction(setup, progress, is_cancelled)


def _require_single_cell_for_steady_state(setup: KineticsSetup) -> None:
    """The steady-state solver has no cell axis, so it needs a one-cell mask.

    The message names the SOLVER rather than the geometry, because a 40-cell
    strip is a perfectly good device and there is nothing to fix about it --
    what cannot be done is asking this particular solver for its fixed point.
    ``DiffusionBackend.steady_state`` carries ``f:(NE,)`` and a scalar gap;
    there is nowhere to put a second cell.
    """
    cells = build_geometry_2d(setup).cell_count
    if cells != 1:
        raise ValueError(
            f"strategy='steady_state' uses the 0-D steady-state solver, whose "
            f"state has no cell axis, and this geometry has {cells} cells. "
            "Either reduce the mask to a single cell, or keep the geometry and "
            "use strategy='time_march', which reaches a steady state by "
            "advancing to stop_tol and imposes no such restriction."
        )


# x_qp is a RATIO whose value depends on a convention (the paper's is twice
# this), so every place that reports one names it -- with this one string.
X_QP_CONVENTION = "qpsim: n_qp/(4 rho_F Delta_0)"


def _qp_energy_profile_2d(state: SpatialState, delta_0: float) -> np.ndarray:
    """Per-cell quasiparticle ENERGY in the x_qp normalisation.

    ``Σ_i W_i f_i / Δ_0`` with ``W_i`` the exact energy-weighted BCS cell
    integral, so that divided by the number profile it is the mean energy per
    quasiparticle in μeV. Same grouping by distinct gap as the number profile.
    """
    if not np.isfinite(delta_0) or delta_0 <= 0.0:
        raise ValueError("delta_0 must be finite and positive.")
    gaps = state.gaps()
    distinct, group_index = np.unique(gaps, return_inverse=True)
    weights = np.column_stack([
        bcs_energy_cell_weights(state.spectral.E, state.spectral.dE, float(g))
        for g in distinct
    ])[:, group_index]
    return np.einsum("ec,ec->c", weights, state.f) / delta_0


# The phonon-temperature fit is trusted only where the spectrum's shape is
# within this of a single Bose-Einstein (weighted std of the log ratio over
# the pair-breaking band). Beyond it the cell is reported as having no single
# temperature, which is the physics, not a failure.
PHONON_T_EFF_RESIDUAL_MAX = 0.05


def _phonon_temperature_field(
    n_ph: np.ndarray, omega: np.ndarray, gap: float, t_bath: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell effective phonon temperature and the fit's residual.

    ``T_eff`` is NaN where the residual exceeds ``PHONON_T_EFF_RESIDUAL_MAX``
    or the fit is underdetermined; the residual is always reported so the
    gate can be read off rather than trusted.
    """
    n_cells = n_ph.shape[1]
    t_eff = np.full(n_cells, np.nan)
    residual = np.full(n_cells, np.nan)
    for j in range(n_cells):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t_j, r_j = effective_phonon_temperature(
                    n_ph[:, j], omega, gap, T_bath=t_bath, return_residual=True,
                )
        except ValueError:
            continue
        residual[j] = r_j
        if r_j <= PHONON_T_EFF_RESIDUAL_MAX:
            t_eff[j] = t_j
    return t_eff, residual


def _xqp_profile_2d(state: SpatialState, delta_0: float) -> np.ndarray:
    """Per-cell ``x_qp`` on a geometry, one quadrature per distinct gap.

    The numerator uses each cell's local gap because transport does, while
    the denominator stays the material reference so
    values across a gap step share one normalization and stay comparable.
    """
    if not np.isfinite(delta_0) or delta_0 <= 0.0:
        raise ValueError("delta_0 must be finite and positive.")
    gaps = state.gaps()
    distinct, group_index = np.unique(gaps, return_inverse=True)
    weights = np.column_stack([
        bcs_dos_cell_weights(state.spectral.E, state.spectral.dE, float(g))
        for g in distinct
    ])[:, group_index]
    return np.einsum("ec,ec->c", weights, state.f) / delta_0


def run_kinetics(
    setup: KineticsSetup,
    progress: ProgressFn,
    is_cancelled: CancelledFn,
    on_frame: FrameFn | None = None,
) -> RunPayload:
    if setup.strategy == "steady_state":
        _require_single_cell_for_steady_state(setup)
        return run_steady_state_0d(setup, progress, is_cancelled)

    payload = RunPayload()
    _check_cancel(is_cancelled)
    progress(0.02, "building geometry")

    state = build_state_2d(setup)
    state, seed_notes = build_initial_state_2d(setup, state)
    payload.notes.extend(seed_notes)
    # Captured BEFORE the run, because `initial` can seed a non-thermal state
    # and then x_qp_initial and x_qp_thermal are different numbers. Reporting
    # only the thermal one would silently answer a question about the seed with
    # a fact about the bath.
    seeded_f = state.f.copy()
    geometry = state.geometry
    injection = build_injection_2d(setup, state)
    external_gain, external_loss = (None, None) if injection is None else injection
    prescribed = build_drives_2d(setup, state)
    if prescribed is not None and injection is not None:
        # Both would be legitimate sources, but `run` refuses a drive
        # alongside raw arrays on purpose, so fold the older narrow knob into
        # the general one rather than silently dropping either.
        prescribed = SumDrive((
            StaticDrive(external_gain, external_loss), prescribed,
        ))
        external_gain = external_loss = None
    delta_0 = setup.material.Delta_0
    photon_params, pb_photon_params = drive_dicts(setup)
    backend = SpatialBackend(
        enable_scattering=setup.collisions.scattering,
        enable_recombination=setup.collisions.recombination,
        enable_phonon_scattering_source=(
            setup.collisions.phonon_scattering_source
        ),
        enable_phonon_recombination_source=(
            setup.collisions.phonon_recombination_source
        ),
        photon_params=photon_params,
        pb_photon_params=pb_photon_params,
        # thermal_bath pins n_ph, which is the shipped behaviour; the dynamic
        # modes solve it per cell. 0.0 is the no-substrate sentinel.
        phonon_escape_time=(
            None if setup.phonons.mode == "thermal_bath"
            else (setup.phonons.tau_l_ns
                  if setup.phonons.mode == "dynamic_escape" else 0.0)
        ),
        # Was not forwarded, so the setup carried this switch and the spatial
        # engine could not read it: every 2-D dynamic-phonon run took the
        # quasiparticle-side kernel regardless of what the setup said.
        use_phonon_side_kernel=setup.phonons.use_phonon_side_kernel,
        phonon_seed=build_phonon_seed_2d(setup, geometry),
    )

    def emit_frame(snapshot: Any) -> None:
        # The SAME reconstruction the finished run stores: a live frame that
        # differed from the recorded one would be a picture of nothing.
        if on_frame is None:
            return
        on_frame(LiveFrame(
            t_ns=float(snapshot.t),
            xqp_profile=_xqp_profile_2d(
                replace(state, f=snapshot.f, gap_per_cell=snapshot.gap_per_cell),
                delta_0,
            ),
            mask=np.asarray(geometry.mask, dtype=bool),
            mesh_size_um=float(geometry.mesh_size),
            x_qp_convention=X_QP_CONVENTION,
        ))

    def hook(elapsed: float, total: float) -> bool:
        # "stepping" says only that it has not finished. The 0-D path already
        # reports simulated time, and on a long spatial march that is the
        # difference between a progress bar and a number the user can act on.
        progress(
            0.05 + 0.9 * min(1.0, elapsed / total),
            f"t = {elapsed:.4g} / {total:.4g} ns",
        )
        return not is_cancelled()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("once")
        result = backend.run(
            state,
            dt=setup.dt,
            max_time=setup.max_time,
            stop_tol=setup.stop_tol,
            external_gain=external_gain,
            external_loss=external_loss,
            drive=prescribed,
            self_consistent_gap=setup.self_consistent_gap,
            gap_quantum=(
                setup.gap_quantum_over_dE * float(np.min(state.spectral.dE))
            ),
            snapshot_interval=setup.snapshot_interval,
            progress_hook=hook,
            on_snapshot=emit_frame if on_frame is not None else None,
        )
    final = result.state
    n_steps, converged, last_rate = (
        result.n_steps, result.converged, result.last_max_rate
    )
    payload.notes.extend(dict.fromkeys(str(w.message) for w in caught))
    if is_cancelled():
        raise RunCancelledError

    profile = _xqp_profile_2d(final, delta_0)
    # Reconstructed onto the mask so a viewer can show the device rather than
    # a flat vector; NaN outside, which plots as blank rather than as zero.
    field = reconstruct_field(geometry.mask, profile)

    payload.arrays["E_bins"] = final.spectral.E
    payload.arrays["f_final"] = final.f
    payload.arrays["mask"] = geometry.mask.astype(np.int8)
    payload.arrays["xqp_field"] = field
    payload.arrays["xqp_profile"] = profile
    payload.arrays["gap_per_cell"] = final.gaps()
    # The rest of the answer sheet a reader needs to read the profile at all:
    # the reference state everything is measured against, the CONVENTION x_qp
    # is quoted in, and the paper-convention variants the published Fischer
    # comparisons are expressed in.
    payload.arrays["f_thermal"] = fermi_dirac_distribution(
        final.spectral.E, setup.T_bath,
    )
    # Factor 2 exactly: the two conventions differ only in whether the
    # denominator counts both spin species. Derived from the profile rather
    # than recomputed, so the two can never disagree.
    payload.arrays["xqp_profile_paper"] = 2.0 * profile
    if geometry.dimensionality <= 1:
        # A strip has a distance coordinate and a reader plots against it (a
        # single cell is the degenerate strip and keeps its one position). The
        # mask plus mesh_size encodes the same information, but making every
        # consumer reconstruct it puts the convention in every consumer.
        # CELL CENTRES, (i + 1/2) h.
        # Emitting i*h instead offsets every profile by half a cell, which is
        # invisible in a plot and wrong in a fit. The index is the cell's own
        # row or column, not its position in mask order: for a rectangle the
        # two coincide, for a rasterised outline (padded, offset) they do not.
        rows_idx, cols_idx = np.nonzero(np.asarray(geometry.mask, dtype=bool))
        along = cols_idx if geometry.occupied_shape[0] == 1 else rows_idx
        payload.arrays["x_um"] = (along.astype(float) + 0.5) * geometry.mesh_size

    if result.snapshots:
        payload.arrays["snap_t_ns"] = np.array([s.t for s in result.snapshots])
        payload.arrays["snap_max_rate"] = np.array(
            [s.max_rate for s in result.snapshots]
        )
        # (frames, NE, Ncells). Stored whole rather than reduced: which
        # reduction a reader wants -- a per-energy map, the total, the profile
        # along one axis -- is a question asked after the run, and a scalar
        # recorded now cannot answer it later.
        payload.arrays["snap_f"] = np.stack([s.f for s in result.snapshots])
        payload.arrays["snap_gap"] = np.stack(
            [s.gap_per_cell for s in result.snapshots]
        )
        if all(s.n_ph is not None for s in result.snapshots):
            payload.arrays["snap_n_ph"] = np.stack(
                [s.n_ph for s in result.snapshots]
            )
            # The axis those populations live on, recorded WITH them. Without
            # it a reader can only re-derive the lattice and check its length,
            # which passes even when every frequency in it is wrong -- see
            # SpatialBackend.phonon_frequency_axis.
            payload.arrays["snap_omega_bins"] = backend.phonon_frequency_axis(
                final
            )
            # The endpoint's phonon temperature FIELD, gated by the fit's own
            # residual: a non-thermal spectrum has no single temperature, and
            # a cell where the Bose-Einstein shape does not fit says so as NaN
            # rather than as a number.
            t_eff, t_res = _phonon_temperature_field(
                np.asarray(result.snapshots[-1].n_ph, dtype=float),
                np.asarray(payload.arrays["snap_omega_bins"], dtype=float),
                float(final.spectral.gap), float(setup.T_bath),
            )
            payload.arrays["phonon_T_eff"] = t_eff
            payload.arrays["phonon_T_eff_residual"] = t_res
            fitted = np.isfinite(t_eff)
            payload.summary["phonon_T_eff_cells_fitted"] = int(fitted.sum())
            payload.summary["phonon_T_eff_mean_K"] = (
                float(np.mean(t_eff[fitted])) if fitted.any() else float("nan")
            )
            if not fitted.all():
                payload.notes.append(
                    f"Phonon temperature: {int((~fitted).sum())} of {fitted.size} "
                    "cells have no single temperature -- their n_ph(ω) departs "
                    "from a Bose-Einstein shape by more than "
                    f"{PHONON_T_EFF_RESIDUAL_MAX:g} in log ratio over the "
                    "pair-breaking band -- and are reported as NaN."
                )
        payload.arrays["snap_xqp_profile"] = np.stack([
            _xqp_profile_2d(replace(final, f=s.f, gap_per_cell=s.gap_per_cell),
                            delta_0)
            for s in result.snapshots
        ])
        payload.arrays["obs_x_qp_mean"] = np.array([
            float(np.mean(p)) for p in payload.arrays["snap_xqp_profile"]
        ])
        payload.arrays["obs_x_qp_max"] = np.array([
            float(np.max(p)) for p in payload.arrays["snap_xqp_profile"]
        ])
        payload.arrays["obs_x_qp_mean_paper"] = (
            2.0 * payload.arrays["obs_x_qp_mean"]
        )
        payload.arrays["obs_x_qp_max_paper"] = (
            2.0 * payload.arrays["obs_x_qp_max"]
        )
        # The QUASIPARTICLE budget, per frame, from the two moments of one
        # quadrature: number Σ w_i f_i and energy Σ W_i f_i, both summed over
        # cells, in the x_qp normalisation (units of 4 rho_F Delta_0 per
        # cell). Their ratio is the mean energy per quasiparticle, which is
        # what "hot" means. Transport alone conserves both bin by bin; a
        # source adds to both; recombination removes number AND energy; the
        # phonon side is deliberately NOT reported -- see the note below.
        number_frames = np.array([
            float(np.sum(_xqp_profile_2d(
                replace(final, f=s.f, gap_per_cell=s.gap_per_cell), delta_0,
            )))
            for s in result.snapshots
        ])
        energy_frames = np.array([
            float(np.sum(_qp_energy_profile_2d(
                replace(final, f=s.f, gap_per_cell=s.gap_per_cell), delta_0,
            )))
            for s in result.snapshots
        ])
        payload.arrays["obs_x_qp_total"] = number_frames
        payload.arrays["obs_E_qp_total"] = energy_frames
        with np.errstate(divide="ignore", invalid="ignore"):
            payload.arrays["obs_E_qp_mean"] = np.where(
                number_frames > 0.0, energy_frames / number_frames, np.nan,
            )
        # The probe AS A TIME SERIES, which is what makes a readout transient
        # legible -- the 0-D transient reports it and the endpoint value alone
        # cannot answer "when does Q_i recover". Single cell only, for the same
        # reason the endpoint probe is: sigma(f) is nonlinear, so there is no
        # single sigma for a spatially varying f.
        if setup.probe.enabled and geometry.cell_count == 1:
            reason = mb_probe_invalid_reason(
                setup.probe, final.spectral.dynes_gamma, final.spectral.gap,
            )
            if reason is None:
                payload.arrays["obs_Q_i"] = np.array([
                    compute_quality_factor(
                        s.f[:, 0], final.spectral,
                        setup.probe.omega_0, setup.probe.alpha,
                    )
                    for s in result.snapshots
                ])

    # Mean energy per quasiparticle at the end, and the phonon side's honest
    # absence: n_ph on the recorded lattice is an occupation PER MODE, and a
    # phonon energy needs the mode density (Debye omega^2 per unit volume, in
    # sound velocities the material table mostly lacks). Adding one on top of
    # a lattice whose kernels may already carry it is the double count this
    # repo once came within inches of; until that is established, no
    # omega-weighted phonon quantity is reported -- see the plan's 5.2.
    number_end = float(np.sum(profile))
    energy_end = float(np.sum(_qp_energy_profile_2d(final, delta_0)))
    if setup.phonons.mode != "thermal_bath":
        payload.notes.append(
            "Phonon energy is not reported: the recorded n_ph is an occupation "
            "per mode, and turning it into an energy needs a phonon mode "
            "density the engine does not carry (and must not add on top of "
            "kernels that may already carry it). The quasiparticle-side budget "
            "(obs_x_qp_total, obs_E_qp_total, obs_E_qp_mean) is exact."
        )
    payload.summary.update({
        # The reference gap the figures normalise energy by. Without it
        # `plots._gap` falls back to 1.0 and the occupation spectrum plots
        # micro-eV against an axis labelled "E / Δ" — a factor of ~180 for
        # aluminium, in the only spectral figure this mode draws.
        "gap_ueV": float(delta_0),
        "cells": int(geometry.cell_count),
        "dimensionality": int(geometry.dimensionality),
        "rows": int(geometry.shape[0]),
        "cols": int(geometry.shape[1]),
        "mesh_size_um": float(geometry.mesh_size),
        # `n_steps`, not `steps`: one vocabulary across the manifests, and
        # readers' existing scripts and plots key on this name.
        "n_steps": int(n_steps),
        "converged": bool(converged),
        "E_qp_mean_ueV": (energy_end / number_end) if number_end > 0.0 else float("nan"),
        "E_qp_mean_over_gap": (
            (energy_end / number_end) / delta_0 if number_end > 0.0 else float("nan")
        ),
        "final_max_rate": float(last_rate),
        "x_qp_mean": float(np.mean(profile)),
        "x_qp_max": float(np.max(profile)),
        "x_qp_min": float(np.min(profile)),
        # The paper convention, as in the array block above. x_qp is a RATIO
        # whose value depends on a convention, so quoting it without naming the
        # convention is quoting a number without its units.
        "x_qp_mean_paper": 2.0 * float(np.mean(profile)),
        "x_qp_max_paper": 2.0 * float(np.max(profile)),
        "x_qp_min_paper": 2.0 * float(np.min(profile)),
        "x_qp_convention": X_QP_CONVENTION,
        "total_time_ns": float(result.elapsed),
        # The reference the run is measured AGAINST. Without it "x_qp = 1.1e-5"
        # is unreadable: the question is always how far above thermal it sits.
        "x_qp_thermal": float(
            qp_fraction(payload.arrays["f_thermal"], final.spectral,
                        delta_0=delta_0)
        ),
        "x_qp_initial": float(np.mean(
            _xqp_profile_2d(replace(final, f=seeded_f), delta_0)
        )),
    })
    # Deliberately NOT added: `x_qp_final` and `x_qp_paper_final`. On a single
    # cell they are exactly `x_qp_mean` and `x_qp_mean_paper`, already above,
    # and a second name for a number that is already there is a duplication a
    # reader then has to check for agreement.
    # The probe is part of THIS mode's model, so it has to act here or say why
    # not. Leaving it silently unread would be a switch the interface shows and
    # the engine ignores, which is the defect this repo keeps finding.
    if setup.probe.enabled:
        if geometry.cell_count == 1:
            _mb_observables(
                payload.summary, payload.notes, final.f[:, 0],
                fermi_dirac_distribution(final.spectral.E, setup.T_bath),
                final.spectral, setup.probe,
            )
        else:
            # NOT averaged silently. sigma(f) is nonlinear and the cells can
            # carry different local gaps, so mean-of-sigma, sigma-of-mean-f and
            # a per-cell field are three different physical claims about one
            # device, and picking one here would publish a convention nobody
            # chose. Kaplan/Mattis-Bardeen as used in this repo is a 0-D
            # statement about a uniform film.
            payload.notes.append(
                f"Mattis-Bardeen probe skipped: this geometry has "
                f"{geometry.cell_count} cells and sigma(f) is nonlinear, so "
                "there is no single sigma for a device with a spatially "
                "varying f -- mean-of-sigma, sigma-of-mean-f and a per-cell "
                "field are different claims. Run the probe on a single-cell "
                "mask, where the quantity is well defined."
            )
    if not converged:
        payload.notes.append(
            f"Did not reach stop_tol={setup.stop_tol:g} within "
            f"max_time={setup.max_time:g} ns; the final residual was "
            f"{last_rate:.3e}. The result is the state at that time, not a "
            "steady state."
        )
    progress(1.0, "done")
    return payload
