"""Fischer 2023 Fig. 3 — the expensive continuation-ladder solve (cache payload).

Split out from :mod:`fig3_paper` so the config+code-hashed sweep cache
(:mod:`validation.sweep_cache`) can hash this whole module plus the shared
independent-certificate module as the figure's solve source. Every other
solve-affecting constant, helper, and solver knob lives here, so editing the
plotting / envelope-overlay code in :mod:`fig3_paper` keeps a cached solve warm
while a numerical change correctly invalidates it. The modules are passed as
``extra_source`` (not ``inspect.getsource(solve)``, which would miss globals
``solve`` reads and helpers it delegates to).

Fig. 3's plotted data *is* the converged ``f(E)`` per ratio, so :func:`solve`
returns it directly as the raw payload and :func:`fig3_paper.observables` just
repackages it; the value of caching here is skipping the τ_l = 0 Newton solve
plus the multi-minute continuation ladder when re-plotting.

The sweep is parameterized by ``num_bins`` / ratio sets (defaulting to the paper
config) so reduced-scale dev runs and equivalence checks are cheap.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
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
from qpsim.solvers.anderson import AndersonAccelerationError

from validation.fischer_2023.steady_state_certificate import (
    NUMBER_CERTIFICATE_FIELDS,
    steady_state_certificate,
)


@dataclass(frozen=True)
class Fig3StepEvent:
    """One completed Fig. 3 continuation step.

    With restart enabled, callbacks receive target and non-target steps
    immediately after an atomic checkpoint is durable; without restart they
    run immediately after step completion. Completed paper targets are replayed
    with ``resumed=True`` before any new work after a restart. Delivery is
    intentionally at-least-once: a process crash after callback return but
    before its acknowledgement checkpoint can replay the event, so side-effect
    callbacks should be idempotent.
    """

    ratio: float
    step_index: int
    step_count: int
    is_target: bool
    solver: str
    resumed: bool
    elapsed_seconds: float
    cumulative_seconds: float
    f: np.ndarray
    certificate: Mapping[str, float] | None


_CHECKPOINT_SCHEMA_VERSION = 2

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
# step. Above ratio ~5 the map is only weakly contractile: a direct 5 → 10
# coupled-Newton jump can leave the physical branch and approach the exact
# vacuum branch. Under-relaxed unit-ratio Picard predictors preserve the
# nonzero branch.  Predictors use safeguarded Anderson acceleration and fall
# back to their historical plain-Picard mixing if acceleration fails or leaves
# the monotone branch (5% mixing above ratio 5); coupled Newton is used only to
# polish requested high-ratio targets after the predictor has reached their
# parameter value.
CONTINUATION_RATIOS: tuple[float, ...] = (
    0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
    6.0, 7.0, 8.0, 9.0, 10.0,
)

# τ_0^PB normalization sanity check (paper Eq. 1 in §IV).
PAPER_TAU_0_PB_PS = 255.0
TAU_0_PB_WARN_FACTOR = 1.05
"""Warn if the numerical tau_0^PB diverges from the paper-quoted 255 ps."""

TARGET_BACKWARD_ERROR_LIMIT = 1e-5
"""Maximum Fig. 3 Picard/target normwise balance error."""

POLISH_FALLBACK_BACKWARD_ERROR_LIMIT = 1e-8
"""Certificate required to retain a predictor if Newton hits roundoff."""

INNER_QP_BACKWARD_ERROR_LIMIT = 1e-10
"""Resolve the weak finite-escape-time response inside each Picard map."""

INNER_QP_NUMBER_POLISH_SHAPE_LIMIT = 1e-8
"""Routing-only shape gate for the scalar QP number-mode polish.

The Fig. 3 grid's independently assembled number-conserving shape error
floors near ``2.4e-10`` after its spectral shape is resolved.  The generic
``1e-13`` routing default therefore spent many 1620x1620 dense Newton solves
before reaching the same amplitude polish as a line-search fallback.  This
wider explicit gate changes no return criterion: every polished state still
re-enters the full residual, aggregate, and number-mode certificates at
``INNER_QP_BACKWARD_ERROR_LIMIT``.
"""

HIGH_RATIO_PICARD_MIXING = 0.15
PICARD_ANDERSON_DEPTH = 3
HIGH_RATIO_FALLBACK_MIXING = 0.05
"""Safeguarded high-bottleneck Picard policy.

The former unconditional ``mixing=0.05, anderson_depth=0`` needed thousands
of outer iterations (4,502 at ratio 6 on the 81-bin diagnostic grid).
Depth-3 Anderson with 15% mixing resolves the same independently certified
branch quickly.  Every accelerated step is checked for forward branch
progress, and any failure retries from the untouched input state with the
historical conservative policy.
"""


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
        newton_backward_error_tol=INNER_QP_BACKWARD_ERROR_LIMIT,
        newton_number_polish_shape_tol=INNER_QP_NUMBER_POLISH_SHAPE_LIMIT,
        newton_max_iter=500,
    )


def _solve_picard(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    mixing: float,
    anderson_depth: int = 0,
) -> T3DiffusionState:
    """Under-relaxed Picard on ``(f, n_ph)`` for branch continuation.

    No absolute phonon-occupation allowance is used here: Fig. 3's driven
    above-gap phonon populations can be far below ``1e-12``, so the former
    ``picard_atol=1e-12`` accepted visibly different curves with O(1) phonon
    backward error at low escape ratios. Relative convergence can reach the
    floating-point fixed-point floor on the reduced and paper grids. The
    explicit Picard balance limit matches the independently reassembled target
    certificate below instead of inheriting a generic scale-dependent default.
    The inner QP solve likewise uses the Fig. 3-specific backward-error limit:
    the generic ``1e-6`` accepts the ratio-zero occupation unchanged at ratio
    ``0.1`` (QP error about ``1.7e-10``) and erases the resolved bottleneck
    response before the outer map can develop it.
    """
    return backend.steady_state(
        state,
        method="picard",
        photon_params=photon_params,
        use_phonon_side_kernel=True,
        picard_tol=1e-8,
        picard_atol=0.0,
        picard_balance_tol=TARGET_BACKWARD_ERROR_LIMIT,
        picard_max_iter=10000,
        picard_mixing=mixing,
        anderson_depth=anderson_depth,
        newton_tol=1e-12,
        newton_backward_error_tol=INNER_QP_BACKWARD_ERROR_LIMIT,
        newton_number_polish_shape_tol=INNER_QP_NUMBER_POLISH_SHAPE_LIMIT,
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
        # The strong-bottleneck state is already balance-certified when the
        # refining Newton step reaches this relative scale.  A tighter 1e-8
        # step gate can demand a strictly smaller raw residual after roundoff,
        # causing Linux builds to discard the polished state and fall back to
        # the less accurate Picard predictor.
        coupled_newton_step_rtol=1e-6,
        coupled_newton_max_iter=50,
        coupled_newton_fd_step=1e-8,
        coupled_newton_analytic_cross=True,
    )


def _picard_mixing_for_ratio(ratio: float) -> float:
    if ratio > 5.0:
        return HIGH_RATIO_PICARD_MIXING
    return 0.15 if ratio > 2.0 else 0.30


def _advances_bottleneck_branch(
    before: T3DiffusionState,
    after: T3DiffusionState,
) -> bool:
    """Whether a continuation step advances the nonzero bottleneck branch."""
    weights = before.spectral.cell_weights
    before_peak = float(np.max(before.f))
    after_peak = float(np.max(after.f))
    before_population = float(np.sum(weights * before.f))
    after_population = float(np.sum(weights * after.f))
    return (
        np.isfinite(after_peak)
        and np.isfinite(after_population)
        and after_peak > before_peak
        and after_population > before_population
    )


def _solve_picard_predictor(
    backend: T3DiffusionBackend,
    state: T3DiffusionState,
    photon_params: dict[str, float],
    *,
    ratio: float,
    mixing: float,
    fallback_mixing: float,
) -> T3DiffusionState:
    """Accelerate one continuation step, with a branch-checked fallback."""
    try:
        accelerated = _solve_picard(
            backend,
            state,
            photon_params,
            mixing=mixing,
            anderson_depth=PICARD_ANDERSON_DEPTH,
        )
        if not _advances_bottleneck_branch(state, accelerated):
            raise RuntimeError(
                "accelerated predictor did not advance both peak and weighted "
                "population on the nonzero bottleneck branch"
            )
        return accelerated
    except (AndersonAccelerationError, RuntimeError) as error:
        print(
            f"    safeguarded Anderson predictor failed at ratio {ratio:g} "
            f"({error}); retrying historical plain Picard "
            f"(mixing={fallback_mixing:g}, AA=0)",
            flush=True,
        )
        fallback = _solve_picard(
            backend,
            state,
            photon_params,
            mixing=fallback_mixing,
            anderson_depth=0,
        )
        if not _advances_bottleneck_branch(state, fallback):
            raise RuntimeError(
                "Fischer Fig. 3 high-ratio continuation failed to advance "
                f"the nonzero bottleneck branch at ratio {ratio:g}."
            ) from error
        return fallback


def _validate_ratio_ladder(
    paper_ratios: tuple[float, ...],
    continuation_ratios: tuple[float, ...],
) -> None:
    """Fail before an expensive solve when continuation cannot hit targets."""
    paper = np.asarray(paper_ratios, dtype=float)
    continuation = np.asarray(continuation_ratios, dtype=float)
    if paper.ndim != 1 or paper.size == 0:
        raise ValueError("paper_ratios must be a non-empty one-dimensional tuple.")
    if np.any(~np.isfinite(paper)) or np.any(paper < 0.0):
        raise ValueError("paper_ratios must be finite and non-negative.")
    if np.unique(paper).size != paper.size:
        raise ValueError("paper_ratios must not contain duplicates.")
    if continuation.ndim != 1:
        raise ValueError("continuation_ratios must be one-dimensional.")
    if np.any(~np.isfinite(continuation)) or np.any(continuation <= 0.0):
        raise ValueError("continuation_ratios must be finite and positive.")
    if continuation.size > 1 and np.any(np.diff(continuation) <= 0.0):
        raise ValueError("continuation_ratios must be strictly increasing.")
    missing = [
        float(r)
        for r in paper
        if r > 0.0 and not np.any(continuation == r)
    ]
    if missing:
        raise ValueError(
            "Every positive paper ratio must occur exactly on the continuation "
            f"ladder; missing {missing}."
        )


def _checkpoint_json(
    values: Mapping[float, Mapping[str, float]] | Mapping[float, float],
) -> str:
    return json.dumps(
        {format(float(key), ".17g"): value for key, value in values.items()},
        sort_keys=True,
        separators=(",", ":"),
    )


def _decode_nested_checkpoint_json(text: str) -> dict[float, dict[str, float]]:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint certificate payload must be an object")
    decoded: dict[float, dict[str, float]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise ValueError("checkpoint certificate entry must be an object")
        decoded[float(key)] = {
            str(name): float(number) for name, number in value.items()
        }
    return decoded


def _decode_scalar_checkpoint_json(text: str) -> dict[float, float]:
    raw = json.loads(text)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint scalar payload must be an object")
    return {float(key): float(value) for key, value in raw.items()}


def _valid_checkpoint_certificate(
    certificate: Mapping[str, float],
    *,
    thermal_phonons: bool,
) -> bool:
    """Return whether one saved certificate has the exact current contract."""
    if set(certificate) != set(NUMBER_CERTIFICATE_FIELDS):
        return False
    qp_fields = (
        "qp_residual_inf",
        "qp_backward_error",
        "qp_number_backward_error",
    )
    if any(
        not np.isfinite(certificate[name]) or certificate[name] < 0.0
        for name in qp_fields
    ):
        return False
    phonon_fields = (
        "phonon_residual_inf",
        "phonon_raw_backward_error",
        "phonon_backward_error",
    )
    if thermal_phonons:
        return all(np.isnan(certificate[name]) for name in phonon_fields)
    return all(
        np.isfinite(certificate[name]) and certificate[name] >= 0.0
        for name in phonon_fields
    )


def _require_target_certificate(
    ratio: float,
    certificate: Mapping[str, float],
) -> None:
    """Reject an uncertified paper target before continuing the ladder.

    Fig. 3's high-ratio continuation is expensive enough that deferring this
    check until every target has completed can waste the whole run after an
    early number-mode failure.  Validate the exact persisted contract and the
    three acceptance metrics at the point each paper target is produced,
    before it is checkpointed, emitted to a callback, or used as the next
    continuation seed.
    """
    thermal_phonons = ratio == 0.0
    if not _valid_checkpoint_certificate(
        certificate,
        thermal_phonons=thermal_phonons,
    ):
        raise RuntimeError(
            "Fischer Fig. 3 paper-ratio state produced an invalid residual "
            f"certificate at ratio {ratio:g}; expected fields "
            f"{list(NUMBER_CERTIFICATE_FIELDS)} with finite non-negative QP "
            "metrics and "
            + (
                "NaN phonon metrics for the thermal shortcut."
                if thermal_phonons
                else "finite non-negative phonon metrics."
            )
        )

    certified_fields = [
        "qp_backward_error",
        "qp_number_backward_error",
    ]
    if not thermal_phonons:
        certified_fields.append("phonon_backward_error")
    failures = {
        name: float(certificate[name])
        for name in certified_fields
        if certificate[name] > TARGET_BACKWARD_ERROR_LIMIT
    }
    if failures:
        detail = ", ".join(
            f"{name}={value:.3e}" for name, value in failures.items()
        )
        raise RuntimeError(
            "Fischer Fig. 3 paper-ratio state failed its immediate "
            f"certificate at ratio {ratio:g}: {detail} exceeds "
            f"{TARGET_BACKWARD_ERROR_LIMIT:g}."
        )


def _write_restart_checkpoint(
    path: Path,
    *,
    identity: str,
    completed_step_index: int,
    pending_callback_step_index: int,
    f_seed: np.ndarray,
    n_ph_seed: np.ndarray | None,
    target_complete: np.ndarray,
    target_f: np.ndarray,
    certificate_by_ratio: Mapping[float, Mapping[str, float]],
    predictor_certificate_by_ratio: Mapping[float, Mapping[str, float]],
    polish_relative_f_by_ratio: Mapping[float, float],
    elapsed_by_step: np.ndarray,
) -> None:
    """Atomically persist a continuation state, including a complete one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(
                fh,
                schema_version=np.asarray([_CHECKPOINT_SCHEMA_VERSION]),
                identity=np.asarray(identity),
                completed_step_index=np.asarray([completed_step_index]),
                pending_callback_step_index=np.asarray(
                    [pending_callback_step_index]
                ),
                f_seed=np.asarray(f_seed, dtype=float),
                n_ph_seed=(
                    np.asarray([], dtype=float)
                    if n_ph_seed is None
                    else np.asarray(n_ph_seed, dtype=float)
                ),
                target_complete=np.asarray(target_complete, dtype=bool),
                target_f=np.asarray(target_f, dtype=float),
                certificate_json=np.asarray(
                    _checkpoint_json(certificate_by_ratio)
                ),
                predictor_certificate_json=np.asarray(
                    _checkpoint_json(predictor_certificate_by_ratio)
                ),
                polish_relative_f_json=np.asarray(
                    _checkpoint_json(polish_relative_f_by_ratio)
                ),
                elapsed_by_step=np.asarray(elapsed_by_step, dtype=float),
            )
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _load_restart_checkpoint(
    path: Path,
    *,
    identity: str,
    num_bins: int,
    num_phonon_bins: int,
    paper_ratios: tuple[float, ...],
    step_ratios: tuple[float, ...],
) -> dict[str, Any] | None:
    """Load a matching restart checkpoint; stale/corrupt files fail cold."""
    if not path.exists():
        return None
    step_count = len(step_ratios)
    try:
        with np.load(path, allow_pickle=False) as saved:
            schema = int(np.asarray(saved["schema_version"]).reshape(-1)[0])
            saved_identity = str(np.asarray(saved["identity"]).item())
            if schema != _CHECKPOINT_SCHEMA_VERSION or saved_identity != identity:
                warnings.warn(
                    "Ignoring stale Fischer Fig. 3 restart checkpoint "
                    f"{path}: schema/content identity mismatch.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            completed = int(
                np.asarray(saved["completed_step_index"]).reshape(-1)[0]
            )
            pending_callback = int(
                np.asarray(saved["pending_callback_step_index"]).reshape(-1)[0]
            )
            f_seed = np.asarray(saved["f_seed"], dtype=float)
            n_ph_seed = np.asarray(saved["n_ph_seed"], dtype=float)
            target_complete = np.asarray(saved["target_complete"], dtype=bool)
            target_f = np.asarray(saved["target_f"], dtype=float)
            elapsed_by_step = np.asarray(saved["elapsed_by_step"], dtype=float)
            certificate_by_ratio = _decode_nested_checkpoint_json(
                str(np.asarray(saved["certificate_json"]).item())
            )
            predictor_certificate_by_ratio = _decode_nested_checkpoint_json(
                str(np.asarray(saved["predictor_certificate_json"]).item())
            )
            polish_relative_f_by_ratio = _decode_scalar_checkpoint_json(
                str(np.asarray(saved["polish_relative_f_json"]).item())
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        warnings.warn(
            f"Ignoring unreadable Fischer Fig. 3 restart checkpoint {path}: "
            f"{error}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    if (
        not 0 <= completed < step_count
        or pending_callback not in (-1, completed)
        or f_seed.shape != (num_bins,)
        or np.any(~np.isfinite(f_seed))
        or np.any((f_seed < 0.0) | (f_seed > 1.0))
        or n_ph_seed.shape not in ((0,), (num_phonon_bins,))
        or np.any(~np.isfinite(n_ph_seed))
        or np.any(n_ph_seed < 0.0)
        or target_complete.shape != (len(paper_ratios),)
        or target_f.shape != (len(paper_ratios), num_bins)
        or np.any(~np.isfinite(target_f))
        or np.any(
            (target_f[target_complete] < 0.0)
            | (target_f[target_complete] > 1.0)
        )
        or elapsed_by_step.shape != (step_count,)
        or np.any(~np.isfinite(elapsed_by_step[: completed + 1]))
        or np.any(elapsed_by_step[: completed + 1] < 0.0)
        or (completed == 0 and n_ph_seed.size != 0)
        or (completed > 0 and n_ph_seed.shape != (num_phonon_bins,))
    ):
        warnings.warn(
            f"Ignoring malformed Fischer Fig. 3 restart checkpoint {path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    for target_index, ratio in enumerate(paper_ratios):
        step_index = step_ratios.index(ratio)
        should_be_complete = step_index <= completed
        if (
            bool(target_complete[target_index]) != should_be_complete
            or (should_be_complete and ratio not in certificate_by_ratio)
            or (
                should_be_complete
                and not _valid_checkpoint_certificate(
                    certificate_by_ratio[ratio],
                    thermal_phonons=(ratio == 0.0),
                )
            )
        ):
            warnings.warn(
                f"Ignoring inconsistent Fischer Fig. 3 restart checkpoint "
                f"{path}.",
                RuntimeWarning,
                stacklevel=2,
            )
            return None
    completed_target_ratios = {
        ratio
        for ratio in paper_ratios
        if step_ratios.index(ratio) <= completed
    }
    if set(certificate_by_ratio) != completed_target_ratios:
        warnings.warn(
            f"Ignoring inconsistent Fischer Fig. 3 restart checkpoint "
            f"{path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    completed_polished_ratios = {
        ratio for ratio in completed_target_ratios if ratio > 5.0
    }
    if (
        set(predictor_certificate_by_ratio) != completed_polished_ratios
        or set(polish_relative_f_by_ratio) != completed_polished_ratios
    ):
        warnings.warn(
            f"Ignoring inconsistent Fischer Fig. 3 restart checkpoint "
            f"{path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if any(
        not _valid_checkpoint_certificate(
            certificate,
            thermal_phonons=False,
        )
        for certificate in predictor_certificate_by_ratio.values()
    ):
        warnings.warn(
            f"Ignoring inconsistent Fischer Fig. 3 restart checkpoint "
            f"{path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if any(
        not np.isfinite(relative_f)
        or relative_f < 0.0
        for relative_f in polish_relative_f_by_ratio.values()
    ):
        warnings.warn(
            f"Ignoring inconsistent Fischer Fig. 3 restart checkpoint "
            f"{path}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    return {
        "completed_step_index": completed,
        "pending_callback_step_index": pending_callback,
        "f_seed": f_seed.copy(),
        "n_ph_seed": None if n_ph_seed.size == 0 else n_ph_seed.copy(),
        "target_complete": target_complete.copy(),
        "target_f": target_f.copy(),
        "certificate_by_ratio": certificate_by_ratio,
        "predictor_certificate_by_ratio": predictor_certificate_by_ratio,
        "polish_relative_f_by_ratio": polish_relative_f_by_ratio,
        "elapsed_by_step": elapsed_by_step.copy(),
    }


def _step_solver_label(ratio: float, *, is_target: bool) -> str:
    if ratio == 0.0:
        return "thermal_newton"
    if ratio > 5.0 and is_target:
        return "picard+coupled_newton"
    return "picard"


def solve(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    checkpoint_path: Path | None = None,
    checkpoint_identity: str | None = None,
    on_step: Callable[[Fig3StepEvent], None] | None = None,
) -> dict[str, np.ndarray]:
    """Solve Fischer Fig. 3 at all paper ratios via continuation.

    The expensive half of Fig. 3: the τ_l = 0 thermal-phonon Newton solve plus
    the warm-seeded continuation ladder (Picard up to ratio ~5, coupled Newton
    above). Returns the converged ``f(E)`` per paper ratio as the raw cache
    payload; :func:`fig3_paper.observables` repackages it into ``Fig3PaperResult``.

    ``checkpoint_path`` and ``checkpoint_identity`` opt into durable restart
    state. Every continuation step is atomically persisted before
    ``on_step`` runs, and callback acknowledgement is persisted separately so
    an exception at a non-target step cannot be bypassed on resume. Completed
    targets are always replayed for exact baseline validation. A completed
    checkpoint is retained until the outer persistence owner has atomically
    stored its final artifact; reopening it performs no numerical work. It is
    content-bound restart state, not a cross-version final-result cache.
    """
    _validate_ratio_ladder(paper_ratios, continuation_ratios)
    if (checkpoint_path is None) != (checkpoint_identity is None):
        raise ValueError(
            "checkpoint_path and checkpoint_identity must be provided together."
        )
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
    if checkpoint_identity is not None and not checkpoint_identity:
        raise ValueError("checkpoint_identity must be a non-empty string.")
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
    certificate_by_ratio: dict[float, dict[str, float]] = {}
    predictor_certificate_by_ratio: dict[float, dict[str, float]] = {}
    polish_relative_f_by_ratio: dict[float, float] = {}
    step_ratios = (0.0, *continuation_ratios)
    step_count = len(step_ratios)
    target_complete = np.zeros(len(paper_ratios), dtype=bool)
    target_f = np.zeros((len(paper_ratios), num_bins), dtype=float)
    elapsed_by_step = np.full(step_count, np.nan)
    completed_step_index = -1
    pending_callback_step_index = -1
    f_seed: np.ndarray
    n_ph_seed: np.ndarray | None

    def record_target(
        ratio: float,
        state: T3DiffusionState,
        certificate: Mapping[str, float],
    ) -> None:
        if ratio not in paper_ratios:
            return
        _require_target_certificate(ratio, certificate)
        target_index = paper_ratios.index(ratio)
        target_complete[target_index] = True
        target_f[target_index] = state.f
        f_by_ratio[ratio] = state.f.copy()
        certificate_by_ratio[ratio] = {
            str(name): float(value) for name, value in certificate.items()
        }

    def persist_restart() -> None:
        if checkpoint_path is None or checkpoint_identity is None:
            return
        _write_restart_checkpoint(
            checkpoint_path,
            identity=checkpoint_identity,
            completed_step_index=completed_step_index,
            pending_callback_step_index=pending_callback_step_index,
            f_seed=f_seed,
            n_ph_seed=n_ph_seed,
            target_complete=target_complete,
            target_f=target_f,
            certificate_by_ratio=certificate_by_ratio,
            predictor_certificate_by_ratio=predictor_certificate_by_ratio,
            polish_relative_f_by_ratio=polish_relative_f_by_ratio,
            elapsed_by_step=elapsed_by_step,
        )

    def emit_step(
        ratio: float,
        step_index: int,
        state_f: np.ndarray,
        *,
        resumed: bool,
    ) -> None:
        if on_step is None:
            return
        is_target = ratio in paper_ratios
        certificate = certificate_by_ratio.get(ratio)
        on_step(
            Fig3StepEvent(
                ratio=float(ratio),
                step_index=step_index,
                step_count=step_count,
                is_target=is_target,
                solver=_step_solver_label(ratio, is_target=is_target),
                resumed=resumed,
                elapsed_seconds=float(elapsed_by_step[step_index]),
                cumulative_seconds=float(
                    np.nansum(elapsed_by_step[: step_index + 1])
                ),
                f=np.asarray(state_f, dtype=float).copy(),
                certificate=(
                    None if certificate is None else dict(certificate)
                ),
            )
        )

    restart = None
    if checkpoint_path is not None and checkpoint_identity is not None:
        omega_size = build_phonon_frequency_map(spectral.E)[0].size
        restart = _load_restart_checkpoint(
            checkpoint_path,
            identity=checkpoint_identity,
            num_bins=num_bins,
            num_phonon_bins=omega_size,
            paper_ratios=paper_ratios,
            step_ratios=step_ratios,
        )
    if restart is not None:
        completed_step_index = int(restart["completed_step_index"])
        pending_callback_step_index = int(
            restart["pending_callback_step_index"]
        )
        f_seed = np.asarray(restart["f_seed"], dtype=float)
        n_ph_seed = restart["n_ph_seed"]
        target_complete = np.asarray(restart["target_complete"], dtype=bool)
        target_f = np.asarray(restart["target_f"], dtype=float)
        certificate_by_ratio = dict(restart["certificate_by_ratio"])
        predictor_certificate_by_ratio = dict(
            restart["predictor_certificate_by_ratio"]
        )
        polish_relative_f_by_ratio = dict(
            restart["polish_relative_f_by_ratio"]
        )
        elapsed_by_step = np.asarray(restart["elapsed_by_step"], dtype=float)
        for target_index, ratio in enumerate(paper_ratios):
            if target_complete[target_index]:
                f_by_ratio[ratio] = target_f[target_index].copy()
        print(
            "  resumed Fischer Fig. 3 after continuation step "
            f"{completed_step_index + 1}/{step_count}",
            flush=True,
        )
        # Re-run completed target validation before spending any more compute.
        # A callback that failed on the latest non-target step is replayed from
        # the durable continuation seed too; no later step can exist because
        # callback exceptions abort the solve immediately.
        for step_index, ratio in enumerate(
            step_ratios[: completed_step_index + 1]
        ):
            if ratio in paper_ratios:
                emit_step(
                    ratio,
                    step_index,
                    f_by_ratio[ratio],
                    resumed=True,
                )
            elif step_index == pending_callback_step_index:
                emit_step(
                    ratio,
                    step_index,
                    f_seed,
                    resumed=True,
                )
        if pending_callback_step_index != -1:
            pending_callback_step_index = -1
            persist_restart()
    else:
        # ── ratio 0: thermal-phonon shortcut (Newton-only, paper τ_l=0 curve) ──
        state0 = _build_state(material, spectral, f_FD, tau_l_scalar=0.0)
        print("  τ_l/τ_0^PB = 0     → target  thermal-phonon shortcut", flush=True)
        started = time.perf_counter()
        converged0 = _solve_tau_l_zero(backend, state0, photon_params)
        elapsed_by_step[0] = time.perf_counter() - started
        if 0.0 in paper_ratios:
            certificate0 = steady_state_certificate(
                converged0,
                photon_params=photon_params,
                tau_l=None,
            )
            record_target(0.0, converged0, certificate0)
        f_seed = converged0.f.copy()
        n_ph_seed = None  # bath thermal at first finite step
        completed_step_index = 0
        pending_callback_step_index = 0 if on_step is not None else -1
        persist_restart()
        emit_step(0.0, 0, converged0.f, resumed=False)
        if pending_callback_step_index != -1:
            pending_callback_step_index = -1
            persist_restart()

    # ── continuation ladder for finite ratios ──
    for step_index, ratio in enumerate(continuation_ratios, start=1):
        if step_index <= completed_step_index:
            continue
        tau_l = ratio * tau_0_pb
        state = _build_state(
            material, spectral, f_seed, tau_l, n_ph_seed=n_ph_seed,
        )
        is_target = ratio in paper_ratios
        tag = "→ target  " if is_target else "(continuation)"
        started = time.perf_counter()

        if ratio > 5.0:
            mixing = _picard_mixing_for_ratio(ratio)
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} branch-preserving "
                f"Picard predictor (mixing={mixing}, "
                f"AA={PICARD_ANDERSON_DEPTH}, safeguarded)",
                flush=True,
            )
            converged = _solve_picard_predictor(
                backend,
                state,
                photon_params,
                ratio=ratio,
                mixing=mixing,
                fallback_mixing=HIGH_RATIO_FALLBACK_MIXING,
            )
            if is_target:
                predictor = converged
                predictor_certificate_by_ratio[ratio] = (
                    steady_state_certificate(
                        predictor,
                        photon_params=photon_params,
                        tau_l=tau_l,
                    )
                )
                print(
                    f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} coupled_newton "
                    "polish from same-ratio physical-branch predictor",
                    flush=True,
                )
                try:
                    converged = _solve_coupled_newton(
                        backend, predictor, photon_params
                    )
                except RuntimeError as error:
                    predictor_certificate = predictor_certificate_by_ratio[ratio]
                    predictor_balance = max(
                        predictor_certificate["qp_backward_error"],
                        predictor_certificate["qp_number_backward_error"],
                        predictor_certificate["phonon_backward_error"],
                    )
                    if (
                        not np.isfinite(predictor_balance)
                        or predictor_balance
                        > POLISH_FALLBACK_BACKWARD_ERROR_LIMIT
                    ):
                        raise
                    # A fully converged Picard predictor can already sit at the
                    # floating-point residual floor. Coupled Newton still tries
                    # a refining step (by design) and may then be unable to
                    # produce a strictly smaller raw residual. Retain the
                    # independently certified same-ratio state rather than
                    # making a finer continuation ladder fail spuriously.
                    print(
                        f"    coupled_newton polish reached roundoff "
                        f"({error}); retaining predictor with max backward "
                        f"error {predictor_balance:.3e}",
                        flush=True,
                    )
                    converged = predictor
                predictor_peak = float(np.max(predictor.f))
                predictor_population = float(
                    np.sum(spectral.cell_weights * predictor.f)
                )
                polished_peak = float(np.max(converged.f))
                polished_population = float(
                    np.sum(spectral.cell_weights * converged.f)
                )
                if (
                    polished_peak <= 0.5 * predictor_peak
                    or polished_population <= 0.5 * predictor_population
                ):
                    raise RuntimeError(
                        "Fischer Fig. 3 same-ratio Newton polish collapsed the "
                        f"Picard branch at ratio {ratio:g}: peaks "
                        f"{predictor_peak:.6e} -> {polished_peak:.6e}, "
                        "weighted populations "
                        f"{predictor_population:.6e} -> "
                        f"{polished_population:.6e}."
                    )
                polish_relative_f_by_ratio[ratio] = float(
                    np.max(np.abs(converged.f - predictor.f))
                ) / max(predictor_peak, np.finfo(float).tiny)
        else:
            mixing = _picard_mixing_for_ratio(ratio)
            print(
                f"  τ_l/τ_0^PB = {ratio:<4g}  {tag} picard "
                f"(mixing={mixing}, AA={PICARD_ANDERSON_DEPTH}, safeguarded)",
                flush=True,
            )
            converged = _solve_picard_predictor(
                backend,
                state,
                photon_params,
                ratio=ratio,
                mixing=mixing,
                fallback_mixing=mixing,
            )

        if is_target:
            certificate = steady_state_certificate(
                converged,
                photon_params=photon_params,
                tau_l=tau_l,
            )
            record_target(ratio, converged, certificate)
            print(
                f"    certificate: |R_qp|∞={certificate['qp_residual_inf']:.3e}, "
                f"η_qp={certificate['qp_backward_error']:.3e}, "
                f"η_qp,N={certificate['qp_number_backward_error']:.3e}, "
                f"|R_ph|∞={certificate['phonon_residual_inf']:.3e}, "
                f"η_ph,raw={certificate['phonon_raw_backward_error']:.3e}, "
                f"η_ph={certificate['phonon_backward_error']:.3e}",
                flush=True,
            )
        f_seed = converged.f.copy()
        n_ph_seed = converged.phonon.n_ph[0, :, 0].copy()
        elapsed_by_step[step_index] = time.perf_counter() - started
        completed_step_index = step_index
        pending_callback_step_index = (
            step_index if on_step is not None else -1
        )
        persist_restart()
        emit_step(ratio, step_index, converged.f, resumed=False)
        if pending_callback_step_index != -1:
            pending_callback_step_index = -1
            persist_restart()

    # Sanity-check all paper ratios captured.
    missing = [r for r in paper_ratios if r not in f_by_ratio]
    if missing:
        raise RuntimeError(f"Continuation did not capture paper ratios: {missing}")

    # Fail before a collapsed/vacuum branch can ever be cached or pinned.  The
    # bottleneck family at fixed positive drive has a resolved nonzero signal,
    # and increasing escape time increases both its peak and integrated
    # occupation.  These are branch-identity checks, not loose plot checks.
    ordered_ratios = sorted(paper_ratios)
    ordered = [f_by_ratio[r] for r in ordered_ratios]
    if any(
        np.any(~np.isfinite(f))
        or np.any((f < 0.0) | (f > 1.0))
        or float(np.max(f)) <= 1e-14
        for f in ordered
    ):
        raise RuntimeError(
            "Fischer Fig. 3 continuation produced a non-finite, unphysical, "
            "or numerically empty paper-ratio branch."
        )
    peaks = np.asarray([np.max(f) for f in ordered], dtype=float)
    populations = np.asarray(
        [np.sum(spectral.cell_weights * f) for f in ordered],
        dtype=float,
    )
    if np.any(np.diff(peaks) <= 0.0) or np.any(np.diff(populations) <= 0.0):
        raise RuntimeError(
            "Fischer Fig. 3 continuation left the monotone bottleneck branch: "
            f"peaks={peaks.tolist()}, weighted populations={populations.tolist()}."
        )

    qp_backward = np.asarray(
        [certificate_by_ratio[r]["qp_backward_error"] for r in paper_ratios],
        dtype=float,
    )
    qp_number_backward = np.asarray(
        [
            certificate_by_ratio[r]["qp_number_backward_error"]
            for r in paper_ratios
        ],
        dtype=float,
    )
    phonon_backward = np.asarray(
        [certificate_by_ratio[r]["phonon_backward_error"] for r in paper_ratios],
        dtype=float,
    )
    if (
        np.any(~np.isfinite(qp_backward))
        or np.any(~np.isfinite(qp_number_backward))
        or np.any(
        ~np.isfinite(phonon_backward[np.asarray(paper_ratios) > 0.0])
        )
    ):
        raise RuntimeError(
            "Fischer Fig. 3 continuation produced a non-finite residual "
            "backward-error certificate."
        )
    positive_targets = np.asarray(paper_ratios, dtype=float) > 0.0
    if (
        np.any(qp_backward > TARGET_BACKWARD_ERROR_LIMIT)
        or np.any(qp_number_backward > TARGET_BACKWARD_ERROR_LIMIT)
        or np.any(
            phonon_backward[positive_targets] > TARGET_BACKWARD_ERROR_LIMIT
        )
    ):
        raise RuntimeError(
            "Fischer Fig. 3 paper-ratio state failed the normwise balance "
            f"certificate (limit={TARGET_BACKWARD_ERROR_LIMIT:g}): "
            f"qp={qp_backward.tolist()}, "
            f"qp_number={qp_number_backward.tolist()}, "
            f"phonon={phonon_backward.tolist()}."
        )

    f_ratios = np.stack([f_by_ratio[r] for r in paper_ratios], axis=0)
    payload = {
        "E": E,
        "f_FD": f_FD,
        "f_ratios": f_ratios,
        "ratios": np.asarray(paper_ratios, dtype=float),
        "tau_0_pb_ns": np.asarray([tau_0_pb], dtype=float),
        "qp_residual_inf": np.asarray(
            [certificate_by_ratio[r]["qp_residual_inf"] for r in paper_ratios],
            dtype=float,
        ),
        "phonon_residual_inf": np.asarray(
            [certificate_by_ratio[r]["phonon_residual_inf"] for r in paper_ratios],
            dtype=float,
        ),
        "phonon_raw_backward_error": np.asarray(
            [
                certificate_by_ratio[r]["phonon_raw_backward_error"]
                for r in paper_ratios
            ],
            dtype=float,
        ),
        "qp_backward_error": qp_backward,
        "qp_number_backward_error": qp_number_backward,
        "phonon_backward_error": phonon_backward,
        "predictor_qp_backward_error": np.asarray(
            [
                predictor_certificate_by_ratio.get(r, {}).get(
                    "qp_backward_error", float("nan")
                )
                for r in paper_ratios
            ],
            dtype=float,
        ),
        "predictor_qp_number_backward_error": np.asarray(
            [
                predictor_certificate_by_ratio.get(r, {}).get(
                    "qp_number_backward_error", float("nan")
                )
                for r in paper_ratios
            ],
            dtype=float,
        ),
        "predictor_phonon_backward_error": np.asarray(
            [
                predictor_certificate_by_ratio.get(r, {}).get(
                    "phonon_backward_error", float("nan")
                )
                for r in paper_ratios
            ],
            dtype=float,
        ),
        "polish_relative_f": np.asarray(
            [polish_relative_f_by_ratio.get(r, float("nan")) for r in paper_ratios],
            dtype=float,
        ),
    }
    # Keep a content-bound COMPLETE checkpoint until the caller has made its
    # own output durable.  Deleting it here would create a crash window between
    # this return and the outer cache/baseline writer's atomic promotion.  A
    # later invocation replays the completed targets without recomputation;
    # the persistence owner may remove the checkpoint only after its final
    # artifact is safely stored.
    return payload


def solver_fingerprint(*, num_bins: int = NUM_BINS) -> dict[str, Any]:
    """Resolved physics + grid knobs, for the cache key's provenance sidecar.

    Safety does not hinge on this being exhaustive: the cache key also folds in
    this module's full source plus the shared certificate module
    (``extra_source``), the ``qpsim`` solver-subtree digest, and the ``run``
    kwargs (num_bins + ratio sets). This keeps the stored provenance
    human-legible.
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
        "inner_qp_backward_error_limit": INNER_QP_BACKWARD_ERROR_LIMIT,
        "inner_qp_number_polish_shape_limit": (
            INNER_QP_NUMBER_POLISH_SHAPE_LIMIT
        ),
        "picard_anderson_depth": PICARD_ANDERSON_DEPTH,
        "high_ratio_picard_mixing": HIGH_RATIO_PICARD_MIXING,
        "high_ratio_fallback_mixing": HIGH_RATIO_FALLBACK_MIXING,
    }
