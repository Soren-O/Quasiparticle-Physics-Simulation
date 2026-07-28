"""Fischer 2023 Fig. 3 — paper-topology qpsim numerical regression.

Four curves at $\\tau_\\ell / \\tau_0^{PB} \\in \\{0, 0.1, 1, 10\\}$ on the
paper's energy grid, with the paper-input $\\tau_0^{PB}$ checked by a
phonon-side discrete-kernel normalization round-trip.

* Paper grid: ``NE = 1620``, ``dE = 1 μeV``, integer-commensurate with
  $\\omega_0 = \\Delta_0/9 = 20\\,\\mu$eV.
* Paper legend ratios: $\\{0, 0.1, 1, 10\\}$.
* Continuation through intermediate ratios for stability of the strong-
  bottleneck branch.
* Paper-style axis: $f(E)$ vs $E/\\Delta_0 - 1$ on $[0, 4]$, with photon-
  step markers at $n\\,\\omega_0/\\Delta_0$.

The module uses the F&C/Kaplan phonon-side pair-breaking kernel and the
analytic near-threshold S_+ quadrature correction. Inverting the kernel sink
recovers the 255 ps Table-I input up to the documented discrete
threshold/quadrature factor; this is not an independent lifetime extraction.

Fischer, Catelani --- Phys. Rev. Applied 19, 054087 (2023), Table I:

    Δ_0     = 180 μeV
    τ_0     = 438 ns
    T_bath  = 0.1 K
    ω_0     = Δ_0 / 9 = 20 μeV
    n̄       = 1 × 10^7
    c_phot  = 1 Hz = 1 × 10^-9 ns^-1

The generated CSV/PDF/validation record authenticate a qpsim solve at the
paper's parameter topology. They do not compare with digitized paper data.

Usage --- generate baseline + PDF::

    python -m validation.fischer_2023.fig3_paper
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import platform
import re
import sys
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO

import numpy as np
from qpsim.constants import KB_UEV_PER_K

from validation import source_provenance, sweep_cache
from validation.fischer_2023 import fig3_solve
from validation.fischer_2023 import (
    steady_state_certificate as certificate_module,
)
from validation.fischer_2023.fig3_solve import (
    C_PHOT,
    CONTINUATION_RATIOS,
    DELTA_0,
    E_MAX_FACTOR,
    E_MIN_FACTOR,
    N_BAR,
    NUM_BINS,
    OMEGA_0,
    PAPER_RATIOS,
    T_BATH,
    T_C,
    TARGET_BACKWARD_ERROR_LIMIT,
    TAU_0,
    Fig3StepEvent,
    _build_grid_and_spectral,
    _compute_tau_0_pb,
    solve,
    solver_fingerprint,
)
from validation.fischer_2024._pdf import (
    validate_single_nonempty_matplotlib_pdf,
)

CURVE_REGRESSION_RTOL = 1e-4
CURVE_REGRESSION_ATOL_OVER_PEAK = 1e-6
STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL = 1.5e-2
VALIDATION_RECORD_SCHEMA = "qpsim-fischer-2023-fig3-validation-v4"


def curve_regression_rtol(
    ratio: float,
    *,
    pinned_on: str,
    running_on: str | None = None,
) -> float:
    """Return the measured fixed-grid curve envelope for this platform.

    Ratios through one remain stable at the strict regression tolerance on
    Windows and Linux.  The ratio-10 state is a residual-polished,
    near-degenerate strong-bottleneck solution: exact single-thread Linux
    runs differ from the Windows pin by 1.27723% while agreeing with each
    other to about 1e-8 relative.  Keep the wider envelope restricted to that
    curve and only for the Windows/Linux OS-family case calibrated by those
    hosted runs.
    """
    current = sys.platform if running_on is None else running_on
    measured_pair = {pinned_on, current} == {"win32", "linux"}
    if ratio == 10.0 and measured_pair:
        return STRONG_BOTTLENECK_CROSS_PLATFORM_RTOL
    return CURVE_REGRESSION_RTOL


@dataclass(frozen=True)
class Fig3PaperResult:
    """Arrays returned by :func:`run`."""

    E: np.ndarray
    tau_0_pb_ns: float
    ratios: tuple[float, ...]   # paper ratios {0, 0.1, 1, 10}
    f_by_ratio: Mapping[float, np.ndarray]
    f_FD: np.ndarray            # thermal reference at T_bath
    certificate_maxima: Mapping[str, float]
    raw_payload_sha256: str
    producer_solve_contract_digest: str
    validated_solve_contract_digest: str


def _readonly_float_array(values: object) -> np.ndarray:
    """Return an owned real float64 snapshot that callers cannot mutate."""

    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError("Fig. 3 numerical arrays must be real-valued.")
    try:
        result = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Fig. 3 numerical arrays must be real numeric values.") from exc
    result.setflags(write=False)
    return result


def _certificate_maxima(raw: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Reduce per-ratio certificate arrays to compact artifact provenance."""
    ratios = np.asarray(raw["ratios"], dtype=float)
    expected_shape = (ratios.size,)
    zero_ratio = ratios == 0.0
    maxima: dict[str, float] = {}
    for name in certificate_module.NUMBER_CERTIFICATE_FIELDS:
        if name not in raw:
            raise ValueError(
                f"Fig. 3 raw solve payload is missing certificate field {name!r}."
            )
        values = np.asarray(raw[name], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"Fig. 3 certificate field {name!r} has shape {values.shape}; "
                f"expected {expected_shape}."
            )
        if name.startswith("phonon_"):
            if np.any(~np.isnan(values[zero_ratio])):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be NaN exactly "
                    "for the ratio-zero thermal-phonon shortcut."
                )
            dynamic_values = values[~zero_ratio]
            if np.any(~np.isfinite(dynamic_values)) or np.any(dynamic_values < 0.0):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be finite and "
                    "non-negative at every positive ratio."
                )
            maxima[name] = (
                float(np.max(dynamic_values))
                if dynamic_values.size
                else float("nan")
            )
        else:
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError(
                    f"Fig. 3 certificate field {name!r} must be finite and "
                    "non-negative at every ratio."
                )
            maxima[name] = float(np.max(values))
    return maxima


def observables(
    raw: Mapping[str, np.ndarray],
    *,
    producer_solve_contract_digest: str,
    validated_solve_contract_digest: str,
) -> Fig3PaperResult:
    """Repackage a raw :func:`fig3_solve.solve` payload into a Fig3PaperResult.

    Fig. 3's plotted quantity is the converged f(E) per ratio, so there is no
    expensive downstream derivation here — this is a pure unpacking of the cached
    arrays. It exists to keep the ``run() = observables(solve())`` split uniform
    with the other figures and to be the boundary the cache leaves uncached.
    """
    # Take one owned snapshot first. A frozen dataclass alone is shallow:
    # without defensive copies its arrays and dictionaries remain mutable,
    # allowing the derived fields to drift after ``raw_payload_sha256`` was
    # computed while retaining an apparently authentic origin hash.
    raw_snapshot = {
        name: np.array(values, copy=True)
        for name, values in raw.items()
    }
    E = _readonly_float_array(raw_snapshot["E"])
    f_FD = _readonly_float_array(raw_snapshot["f_FD"])
    ratios = tuple(float(r) for r in raw_snapshot["ratios"])
    tau_0_pb_ns = float(
        np.asarray(raw_snapshot["tau_0_pb_ns"]).reshape(-1)[0]
    )
    f_ratios = np.asarray(raw_snapshot["f_ratios"])
    f_by_ratio = MappingProxyType({
        r: _readonly_float_array(f_ratios[i])
        for i, r in enumerate(ratios)
    })
    certificate_maxima = MappingProxyType(
        dict(_certificate_maxima(raw_snapshot))
    )
    return Fig3PaperResult(
        E=E,
        tau_0_pb_ns=tau_0_pb_ns,
        ratios=ratios,
        f_by_ratio=f_by_ratio,
        f_FD=f_FD,
        certificate_maxima=certificate_maxima,
        raw_payload_sha256=sweep_cache._array_payload_sha256(
            raw_snapshot
        ),
        producer_solve_contract_digest=producer_solve_contract_digest,
        validated_solve_contract_digest=validated_solve_contract_digest,
    )


def solve_contract_digest() -> str:
    """Return a runtime-neutral digest of the Fig. 3 numerical contract.

    The restart/cache identity deliberately also includes Python, NumPy,
    SciPy, BLAS, platform, and thread controls. Those are useful producer
    provenance but cannot be a cross-platform baseline-currentness gate.
    This digest instead binds only the conservative qpsim solve tree and the
    Fig. 3 solver/certificate sources; the separately stamped configuration
    fields bind the resolved grid and physics parameters.
    """
    digest = hashlib.sha256()
    ingredients = (
        ("qpsim", sweep_cache.solve_source_digest()),
        ("solve_source_digest", inspect.getsource(sweep_cache.solve_source_digest)),
        (
            "canonical_source_bytes",
            inspect.getsource(source_provenance.canonical_source_bytes),
        ),
        ("fig3_solve", inspect.getsource(fig3_solve)),
        ("certificate", inspect.getsource(certificate_module)),
    )
    for label, source in ingredients:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def run(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    on_step: Callable[[Fig3StepEvent], None] | None = None,
    restart_checkpoint_path: Path | None = None,
) -> Fig3PaperResult:
    """Solve Fischer Fig. 3 and repackage — the pure, uncached path.

    Exactly ``observables(solve(...))``. The ``@pytest.mark.slow`` regression
    test calls this (no args) so it always truly recomputes against the pinned
    baseline; the cached dev / regen path is :func:`run_cached`. Restart state
    is likewise opt-in: pass ``restart_checkpoint_path`` explicitly when an
    interruptible long run should resume. A completed checkpoint is retained;
    no automatic cleanup is performed, so the persistence owner should remove
    it explicitly only after the final artifact is durable.
    """
    _kwargs, _fingerprint, _extra_source, identity = _solve_cache_inputs(
        num_bins=num_bins,
        paper_ratios=paper_ratios,
        continuation_ratios=continuation_ratios,
    )
    solve_kwargs: dict[str, Any] = {
        "num_bins": num_bins,
        "paper_ratios": paper_ratios,
        "continuation_ratios": continuation_ratios,
        "on_step": on_step,
    }
    if restart_checkpoint_path is not None:
        solve_kwargs.update(
            checkpoint_path=Path(restart_checkpoint_path),
            checkpoint_identity=identity,
        )
    contract_digest = solve_contract_digest()
    return observables(
        solve(**solve_kwargs),
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def _solve_cache_inputs(
    *,
    num_bins: int,
    paper_ratios: tuple[float, ...],
    continuation_ratios: tuple[float, ...],
) -> tuple[dict[str, object], dict[str, Any], str, str]:
    """Return the one canonical content identity for cache and restart state."""
    kwargs: dict[str, object] = {
        "num_bins": int(num_bins),
        "paper_ratios": [float(r) for r in paper_ratios],
        "continuation_ratios": [float(r) for r in continuation_ratios],
    }
    fingerprint = solver_fingerprint(num_bins=num_bins)
    extra_source = (
        inspect.getsource(fig3_solve)
        + inspect.getsource(certificate_module)
    )
    identity = sweep_cache.cache_key(
        "fischer_2023/fig3",
        fingerprint,
        kwargs,
        extra_source=extra_source,
    )
    return kwargs, fingerprint, extra_source, identity


def _default_restart_checkpoint_path(identity: str) -> Path:
    return (
        sweep_cache.default_cache_dir()
        / "fischer_2023__fig3_restart"
        / f"{identity}.npz"
    )


def run_cached(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    restart_checkpoint_path: Path | None = None,
) -> Fig3PaperResult:
    """Like :func:`run`, but the expensive continuation solve is served from the
    disk cache when nothing solve-relevant has changed (see
    :mod:`validation.sweep_cache`). Used by the regen / ``__main__`` path; editing
    the plotting / observable code here does not invalidate the cached solve.
    Disable with ``QPSIM_SWEEP_CACHE=0``. In that mode the solve is genuinely
    fresh: no default restart state is read or written. Pass
    ``restart_checkpoint_path`` explicitly to opt back into interruption
    recovery while leaving the final-result cache disabled. Completed restart
    files are retained until explicitly removed by the persistence owner.
    """
    kwargs, fingerprint, extra_source, identity = _solve_cache_inputs(
        num_bins=num_bins,
        paper_ratios=paper_ratios,
        continuation_ratios=continuation_ratios,
    )
    checkpoint_path = (
        Path(restart_checkpoint_path)
        if restart_checkpoint_path is not None
        else (
            _default_restart_checkpoint_path(identity)
            if sweep_cache.is_enabled()
            else None
        )
    )

    def solve_current() -> Mapping[str, np.ndarray]:
        solve_kwargs: dict[str, Any] = {
            "num_bins": num_bins,
            "paper_ratios": paper_ratios,
            "continuation_ratios": continuation_ratios,
        }
        if checkpoint_path is not None:
            solve_kwargs.update(
                checkpoint_path=checkpoint_path,
                checkpoint_identity=identity,
            )
        return solve(**solve_kwargs)

    raw = sweep_cache.cached_solve(
        "fischer_2023/fig3",
        solve_current,
        fingerprint=fingerprint,
        kwargs=kwargs,
        extra_source=extra_source,
    )
    contract_digest = solve_contract_digest()
    return observables(
        raw,
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def run_cached_with_evidence(
    *,
    num_bins: int = NUM_BINS,
    paper_ratios: tuple[float, ...] = PAPER_RATIOS,
    continuation_ratios: tuple[float, ...] = CONTINUATION_RATIOS,
    restart_checkpoint_path: Path | None = None,
) -> tuple[Fig3PaperResult, dict[str, Any]]:
    """Run the regeneration path and return authenticated cache/restart evidence."""
    kwargs, fingerprint, extra_source, identity = _solve_cache_inputs(
        num_bins=num_bins,
        paper_ratios=paper_ratios,
        continuation_ratios=continuation_ratios,
    )
    checkpoint_path = (
        Path(restart_checkpoint_path)
        if restart_checkpoint_path is not None
        else (
            _default_restart_checkpoint_path(identity)
            if sweep_cache.is_enabled()
            else None
        )
    )
    checkpoint_existed_before = bool(
        checkpoint_path is not None and checkpoint_path.is_file()
    )

    def solve_current() -> Mapping[str, np.ndarray]:
        solve_kwargs: dict[str, Any] = {
            "num_bins": num_bins,
            "paper_ratios": paper_ratios,
            "continuation_ratios": continuation_ratios,
        }
        if checkpoint_path is not None:
            solve_kwargs.update(
                checkpoint_path=checkpoint_path,
                checkpoint_identity=identity,
            )
        return solve(**solve_kwargs)

    raw, cache_evidence = sweep_cache.cached_solve_with_evidence(
        "fischer_2023/fig3",
        solve_current,
        fingerprint=fingerprint,
        kwargs=kwargs,
        extra_source=extra_source,
    )
    contract_digest = solve_contract_digest()
    result = observables(
        raw,
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )
    return result, {
        "cache": cache_evidence,
        "restart": {
            "canonical_request": kwargs,
            "checkpoint_existed_before": checkpoint_existed_before,
            "checkpoint_identity": identity,
            "checkpoint_path": (
                str(checkpoint_path.resolve(strict=False))
                if checkpoint_path is not None
                else None
            ),
            "qualification": (
                "A solver invocation may replay a content-bound restart; "
                "an authenticated final-result cache hit invokes no solver."
            ),
        },
    }


def baseline_path() -> Path:
    """Output CSV path.

    Named ``fischer_fig3_paper.csv`` because the paper-input tau_0^PB
    normalization is checked by the Kaplan phonon-side kernel round-trip.
    """
    root = Path(__file__).resolve().parents[2]
    return (
        root / "validation" / "baselines" / "ph0_constant"
        / "fischer_fig3_paper.csv"
    )


def plot_path() -> Path:
    return baseline_path().with_suffix(".pdf")


def validation_record_path() -> Path:
    return baseline_path().with_suffix(".validation.json")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validation_source_identity() -> dict[str, Any]:
    """Portable source/configuration identity for solve and verifier."""
    validation_root = Path(__file__).resolve().parent
    return {
        "solve_contract_digest": solve_contract_digest(),
        "solver_fingerprint": solver_fingerprint(num_bins=NUM_BINS),
        "source_sha256": source_provenance.source_manifest(
            Path(__file__),
            extra_validation_modules=(
                Path(fig3_solve.__file__),
                Path(certificate_module.__file__),
                validation_root / "_paper_envelope.py",
                Path(sweep_cache.__file__),
                Path(__file__).resolve().parents[1]
                / "fischer_2024"
                / "_pdf.py",
            ),
        ),
    }


def _validation_runtime_provenance() -> dict[str, Any]:
    """Diagnostic runtime identity, frozen but not used as a reader gate."""
    import matplotlib

    return {
        "library_versions": sweep_cache._lib_versions(),
        "matplotlib": {
            "backend": str(matplotlib.get_backend()),
            "version": matplotlib.__version__,
        },
        "numeric_runtime_identity": sweep_cache._numeric_runtime_identity(),
        "platform": {
            "detail": platform.platform(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
    }


def _json_safe_number(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def _real_occupation_array(values: object, *, context: str) -> np.ndarray:
    """Return a float occupation array without discarding complex parts."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise RuntimeError(f"{context} occupations must be real-valued.")
    try:
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{context} occupations must be real numeric values.") from exc


def _validate_persisted_grid_and_thermal_reference(
    result: Fig3PaperResult,
    *,
    expected_E: np.ndarray,
) -> None:
    """Bind both persisted axes to the current deterministic references."""
    from qpsim.physics.spectral import fermi_dirac_occupation

    if not np.array_equal(result.E, expected_E):
        raise RuntimeError("Fig. 3 persisted energy grid is not the current paper grid.")
    expected_f_FD = fermi_dirac_occupation(expected_E, T_BATH)
    if not np.array_equal(result.f_FD, expected_f_FD):
        raise RuntimeError(
            "Fig. 3 persisted f_FD is not the current-grid thermal reference."
        )


def _reassemble_artifact_certificates(
    result: Fig3PaperResult,
) -> dict[str, Any]:
    """Reconstruct the implied Ph0 root and certify every persisted f(E)."""
    from qpsim.collisions.phonon import (
        build_phonon_frequency_map,
        build_recombination_kernel_base,
        build_recombination_kernel_phonon_side,
        build_scattering_kernel_base,
        build_scattering_kernel_phonon_side,
    )
    from qpsim.phonon_models.ph0_local import phonon_steady_state
    from qpsim.physics.kernels import thermal_phonon_occupation

    _E, _dE, spectral = fig3_solve._build_grid_and_spectral(NUM_BINS)
    _validate_persisted_grid_and_thermal_reference(
        result,
        expected_E=spectral.E,
    )
    material = fig3_solve._fischer_material()
    if material.tau_0_pb_ns is None:
        raise RuntimeError("Fig. 3 material is missing tau_0_pb_ns.")
    tau_0_pb_ns = float(material.tau_0_pb_ns)
    K_s0 = build_scattering_kernel_base(
        spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )
    K_r0 = build_recombination_kernel_base(
        spectral,
        tau_0=material.tau_0,
        T_c=material.T_c,
    )
    K_s0_phonon = build_scattering_kernel_phonon_side(
        spectral,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    K_r0_phonon = build_recombination_kernel_phonon_side(
        spectral,
        tau_0_pb_ns=tau_0_pb_ns,
    )
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(spectral.E)
    photon_params = {
        "omega_0": OMEGA_0,
        "n_bar": N_BAR,
        "c_phot": C_PHOT,
    }
    rows: list[dict[str, Any]] = []
    values_by_field: dict[str, list[float]] = {
        field: []
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for ratio in result.ratios:
        f = _real_occupation_array(
            result.f_by_ratio[ratio],
            context=f"Fig. 3 ratio {ratio:g}",
        )
        if ratio == 0.0:
            tau_l_for_state = 0.0
            tau_l_for_certificate: float | None = None
            n_ph = thermal_phonon_occupation(omega, T_BATH)
        else:
            tau_l_for_state = ratio * result.tau_0_pb_ns
            tau_l_for_certificate = tau_l_for_state
            n_ph = phonon_steady_state(
                f,
                spectral,
                K_s0,
                K_r0,
                omega,
                idx_diff,
                idx_sum,
                diff_sign,
                T_BATH,
                tau_l_for_state,
                K_s0_phonon_side=K_s0_phonon,
                K_r0_phonon_side=K_r0_phonon,
            )
        state = fig3_solve._build_state(
            material,
            spectral,
            f,
            tau_l_for_state,
            n_ph_seed=n_ph,
        )
        certificate = certificate_module.steady_state_certificate(
            state,
            photon_params=photon_params,
            tau_l=tau_l_for_certificate,
        )
        if set(certificate) != set(certificate_module.NUMBER_CERTIFICATE_FIELDS):
            raise RuntimeError(
                f"Fig. 3 ratio {ratio:g} reassembly returned an incomplete certificate."
            )
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
            value = float(certificate[field])
            if field.startswith("phonon_") and ratio == 0.0:
                if not np.isnan(value):
                    raise RuntimeError(
                        f"Fig. 3 ratio-zero certificate {field} must be NaN."
                    )
            elif not np.isfinite(value) or value < 0.0:
                raise RuntimeError(
                    f"Fig. 3 ratio {ratio:g} certificate {field} is invalid: {value}."
                )
            values_by_field[field].append(value)
        for field in _CERTIFIED_BACKWARD_ERROR_FIELDS:
            value = float(certificate[field])
            if np.isfinite(value) and value > TARGET_BACKWARD_ERROR_LIMIT:
                raise RuntimeError(
                    f"Fig. 3 ratio {ratio:g} certificate {field}={value:.3e} "
                    f"exceeds {TARGET_BACKWARD_ERROR_LIMIT:.3e}."
                )
        rows.append(
            {
                "ratio": ratio,
                "persisted_f_sha256": _sha256_array(f),
                "certificate": {
                    field: _json_safe_number(certificate[field])
                    for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
                },
                "reconstructed_n_ph": {
                    "max": float(np.max(n_ph)),
                    "min": float(np.min(n_ph)),
                    "sha256": _sha256_array(n_ph),
                },
            }
        )
    maxima = {
        field: max(
            value
            for value in values
            if np.isfinite(value)
        )
        for field, values in values_by_field.items()
    }
    _validate_certificate_maxima(
        maxima,
        target=TARGET_BACKWARD_ERROR_LIMIT,
        context="Fig. 3 independent artifact reassembly",
    )
    return {
        "maxima": maxima,
        "rows": rows,
        "scope": (
            "Each persisted f(E) is paired with the unique Ph0 affine phonon "
            "fixed point it implies under the bound current equations. The "
            "complete QP, phonon, and pair-number certificate is then "
            "independently reassembled from that pair."
        ),
        "qualification": (
            "The CSV does not persist the producer's original finite-ratio "
            "n_ph arrays. This proves current-equation membership of the "
            "reconstructed pair, not byte identity with the omitted producer n_ph."
        ),
    }


_TAU_0_PB_RE = re.compile(r"tau_0_pb_ns=([\deE.+-]+)")
_RATIOS_RE = re.compile(r"ratios=\[([^\]]+)\]")
_GRID_NE_RE = re.compile(r"NE=(\d+)")
_E_MIN_RE = re.compile(r"E_min=([\deE.+-]+)\*Delta")
_E_MAX_RE = re.compile(r"E_max=([\deE.+-]+)\*Delta")
_CERTIFICATE_VERSION_RE = re.compile(r"certificate_metric_version='([^']*)'")
_CERTIFICATE_LIMIT_RE = re.compile(
    r"target_backward_error_limit=([\deE.+-]+)"
)
_CERTIFICATE_MAXIMA_RE = re.compile(r"^# certificate_maxima\s+(.+)$", re.MULTILINE)
_PIN_PLATFORM_RE = re.compile(r"^# pinned_on: ([^\r\n]+)$", re.MULTILINE)
_PRODUCER_SOLVE_CONTRACT_DIGEST_RE = re.compile(
    r"^# producer_solve_contract_digest=([0-9a-f]{64})$",
    re.MULTILINE,
)
_VALIDATED_SOLVE_CONTRACT_DIGEST_RE = re.compile(
    r"^# validated_solve_contract_digest=([0-9a-f]{64})$",
    re.MULTILINE,
)
_RAW_PAYLOAD_SHA256_RE = re.compile(
    r"^# raw_payload_sha256=([0-9a-f]{64})$",
    re.MULTILINE,
)
_CERTIFIED_BACKWARD_ERROR_FIELDS = (
    "qp_backward_error",
    "qp_number_backward_error",
    "phonon_backward_error",
)
_HEADER_PARAM_RE = {
    "delta_0": re.compile(r"Delta_0=([\deE.+-]+)"),
    "tau_0": re.compile(r"tau_0=([\deE.+-]+)"),
    "t_bath": re.compile(r"T_bath=([\deE.+-]+)"),
    "omega_0": re.compile(r"omega_0=([\deE.+-]+)"),
    "n_bar": re.compile(r"n_bar=([\deE.+-]+)"),
    "c_phot": re.compile(r"c_phot=([\deE.+-]+)"),
}


def _read_baseline_text(path: Path) -> str:
    """Read current UTF-8 artifacts and legacy Windows-local CSV output."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Historical ``path.open()`` used the platform default. A baseline
        # generated on Windows therefore encoded the em dash as cp1252 even
        # though Linux-generated pinned artifacts were UTF-8.
        return path.read_text(encoding="cp1252")


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    """Yield a same-directory temporary file and replace ``path`` on success."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_certificate_maxima(
    maxima: Mapping[str, float],
    *,
    target: float,
    context: str,
) -> dict[str, float]:
    """Return a normalized maxima map or reject an uncertified artifact."""
    expected = set(certificate_module.NUMBER_CERTIFICATE_FIELDS)
    if set(maxima) != expected:
        missing = sorted(expected.difference(maxima))
        extra = sorted(set(maxima).difference(expected))
        raise RuntimeError(
            f"{context} certificate maxima fields are incomplete; "
            f"missing={missing}, extra={extra}."
        )
    normalized = {
        name: float(maxima[name])
        for name in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for name, value in normalized.items():
        if not np.isfinite(value) or value < 0.0:
            raise RuntimeError(
                f"{context} certificate maximum {name!r} must be finite "
                "and non-negative."
            )
    for name in _CERTIFIED_BACKWARD_ERROR_FIELDS:
        if normalized[name] > target:
            raise RuntimeError(
                f"{context} certificate maximum {name!r} is "
                f"{normalized[name]:.17g}, above target {target:.17g}."
            )
    return normalized


def _parse_certificate_maxima(text: str, path: Path) -> dict[str, float]:
    matches = _CERTIFICATE_MAXIMA_RE.findall(text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Baseline header at {path} must contain exactly one "
            "'# certificate_maxima' record."
        )
    maxima: dict[str, float] = {}
    for item in matches[0].split():
        name, separator, value = item.partition("=")
        if not separator or name in maxima:
            raise RuntimeError(
                f"Baseline header at {path} has malformed certificate maxima."
            )
        try:
            maxima[name] = float(value)
        except ValueError as exc:
            raise RuntimeError(
                f"Baseline header at {path} has a non-numeric certificate maximum."
            ) from exc
    return maxima


def _validate_result_for_artifact(result: Fig3PaperResult) -> None:
    energies = np.asarray(result.E, dtype=float)
    if energies.ndim != 1 or energies.size == 0:
        raise RuntimeError("Fig. 3 artifact energies must be a non-empty 1-D array.")
    if not np.all(np.isfinite(energies)) or np.any(np.diff(energies) <= 0.0):
        raise RuntimeError("Fig. 3 artifact energies must be finite and increasing.")
    ratios = np.asarray(result.ratios, dtype=float)
    if (
        ratios.ndim != 1
        or ratios.size == 0
        or not np.all(np.isfinite(ratios))
        or np.unique(ratios).size != ratios.size
    ):
        raise RuntimeError("Fig. 3 artifact ratios must be finite and unique.")
    if not np.isfinite(result.tau_0_pb_ns) or result.tau_0_pb_ns <= 0.0:
        raise RuntimeError("Fig. 3 artifact tau_0_pb_ns must be finite and positive.")
    thermal = _real_occupation_array(
        result.f_FD,
        context="Fig. 3 artifact f_FD",
    )
    if thermal.shape != energies.shape or not np.all(np.isfinite(thermal)):
        raise RuntimeError("Fig. 3 artifact f_FD must be finite on the energy grid.")
    if np.any(thermal < 0.0) or np.any(thermal > 1.0):
        raise RuntimeError(
            "Fig. 3 artifact f_FD occupations must lie in inclusive [0, 1]."
        )
    if set(result.f_by_ratio) != set(result.ratios):
        raise RuntimeError("Fig. 3 artifact curve keys must exactly match its ratios.")
    for ratio in result.ratios:
        values = _real_occupation_array(
            result.f_by_ratio[ratio],
            context=f"Fig. 3 artifact curve at ratio {ratio:g}",
        )
        if values.shape != energies.shape or not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Fig. 3 artifact curve at ratio {ratio:g} must be finite "
                "on the energy grid."
            )
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise RuntimeError(
                f"Fig. 3 artifact curve at ratio {ratio:g} occupations must "
                "lie in inclusive [0, 1]."
            )
    _validate_certificate_maxima(
        result.certificate_maxima,
        target=TARGET_BACKWARD_ERROR_LIMIT,
        context="Fig. 3 artifact",
    )
    for name, digest in (
        ("raw_payload_sha256", result.raw_payload_sha256),
        ("producer_solve_contract_digest", result.producer_solve_contract_digest),
        ("validated_solve_contract_digest", result.validated_solve_contract_digest),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(
                f"Fig. 3 artifact {name} must be a lowercase 64-character "
                "SHA-256 content identity."
            )


def write_baseline(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Write the four paper-ratio f(E) arrays + thermal reference to CSV."""
    if path is None or path.resolve() == baseline_path().resolve():
        raise RuntimeError(
            "Direct canonical Fig. 3 CSV writes are forbidden; use "
            "publish_baseline_triple() or generate_baseline()."
        )
    _validate_result_for_artifact(result)
    path.parent.mkdir(parents=True, exist_ok=True)
    header_cols = ["E_uev", "f_FD"] + [f"f_ratio_{r:g}" for r in result.ratios]
    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([
            "# Fischer 2023 Fig. 3 — paper-topology qpsim numerical regression"
        ])
        writer.writerow([f"# pinned_on: {sys.platform}"])
        writer.writerow([
            f"# Delta_0={DELTA_0} tau_0={TAU_0} T_bath={T_BATH} "
            f"omega_0={OMEGA_0} n_bar={N_BAR} c_phot={C_PHOT}"
        ])
        writer.writerow([
            f"# Grid: NE={result.E.size} E_min={E_MIN_FACTOR}*Delta "
            f"E_max={E_MAX_FACTOR}*Delta"
        ])
        writer.writerow([f"# tau_0_pb_ns={result.tau_0_pb_ns} ratios={list(result.ratios)}"])
        writer.writerow([
            "# producer_solve_contract_digest="
            f"{result.producer_solve_contract_digest}"
        ])
        writer.writerow([
            "# validated_solve_contract_digest="
            f"{result.validated_solve_contract_digest}"
        ])
        writer.writerow([f"# raw_payload_sha256={result.raw_payload_sha256}"])
        writer.writerow([
            "# certificate_metric_version="
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r} "
            "target_backward_error_limit="
            f"{TARGET_BACKWARD_ERROR_LIMIT}"
        ])
        writer.writerow([
            "# certificate_maxima "
            + " ".join(
                f"{name}={result.certificate_maxima.get(name, float('nan')):.17e}"
                for name in certificate_module.NUMBER_CERTIFICATE_FIELDS
            )
        ])
        writer.writerow(header_cols)
        n = result.E.size
        for i in range(n):
            row = [f"{result.E[i]:.17e}", f"{result.f_FD[i]:.17e}"]
            row.extend(f"{result.f_by_ratio[r][i]:.17e}" for r in result.ratios)
            writer.writerow(row)
    return path


def read_baseline(
    path: Path | None = None,
    *,
    _publication_lock_held: bool = False,
) -> Fig3PaperResult:
    """Read a Fig. 3 CSV, authenticating the canonical evidence triple.

    A noncanonical ``path`` is an explicitly uncommitted fixture/readback: its
    self-contained schema and certificates are validated, but it is not bound
    to the canonical publication record or current source tree.
    """
    resolved = baseline_path() if path is None else path
    canonical_read = resolved.resolve() == baseline_path().resolve()
    if canonical_read and not _publication_lock_held:
        with _publication_lock(validation_record_path()):
            return read_baseline(
                resolved,
                _publication_lock_held=True,
            )
    if canonical_read:
        read_validation_record(_publication_lock_held=True)
    return _read_baseline_unlocked(resolved)


def _read_baseline_unlocked(path: Path) -> Fig3PaperResult:
    """Parse one CSV after its caller has established any required snapshot."""
    metadata = _read_baseline_metadata_unlocked(path)
    rows: list[list[float]] = []
    tau_0_pb: float | None = None
    ratios: tuple[float, ...] = ()
    ratio_records = 0
    columns: tuple[str, ...] | None = None
    reader = csv.reader(_read_baseline_text(path).splitlines())
    for line in reader:
        if not line:
            continue
        first = line[0]
        if first.startswith("# tau_0_pb_ns"):
            ratio_records += 1
            if ratio_records != 1:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has duplicate ratio metadata."
                )
            m_tau = _TAU_0_PB_RE.search(first)
            m_ratios = _RATIOS_RE.search(first)
            if m_tau:
                tau_0_pb = float(m_tau.group(1))
            if m_ratios:
                ratios = tuple(
                    float(x.strip()) for x in m_ratios.group(1).split(",") if x.strip()
                )
            continue
        if first.startswith("#"):
            continue
        if first == "E_uev":
            current_columns = tuple(line)
            expected_columns = (
                "E_uev",
                "f_FD",
                *(f"f_ratio_{ratio:g}" for ratio in metadata.ratios),
            )
            if columns is not None:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has duplicate column headers."
                )
            if current_columns != expected_columns:
                raise RuntimeError(
                    f"Fig. 3 baseline at {path} has columns {current_columns}; "
                    f"expected {expected_columns}."
                )
            columns = current_columns
            continue
        if columns is None:
            raise RuntimeError(
                f"Fig. 3 baseline at {path} has data before its column header."
            )
        expected_width = 2 + len(metadata.ratios)
        if len(line) != expected_width:
            raise RuntimeError(
                f"Fig. 3 baseline row has {len(line)} columns; expected "
                f"{expected_width}."
            )
        try:
            rows.append([float(x) for x in line])
        except ValueError as exc:
            raise RuntimeError(
                f"Fig. 3 baseline at {path} contains a non-numeric data row."
            ) from exc
    if tau_0_pb is None or not ratios:
        raise RuntimeError(f"Baseline header at {path} missing tau_0_pb_ns / ratios metadata.")
    if columns is None:
        raise RuntimeError(f"Fig. 3 baseline at {path} missing its column header.")
    if tau_0_pb != metadata.tau_0_pb_ns or ratios != metadata.ratios:
        raise RuntimeError(f"Fig. 3 baseline at {path} has inconsistent metadata.")
    if len(rows) != metadata.num_bins:
        raise RuntimeError(
            f"Fig. 3 baseline at {path} has {len(rows)} data rows; "
            f"expected NE={metadata.num_bins}."
        )
    data = np.array(rows, dtype=float)
    if data.shape != (metadata.num_bins, 2 + len(ratios)):
        raise RuntimeError(
            f"Fig. 3 baseline at {path} has data shape {data.shape}; expected "
            f"{(metadata.num_bins, 2 + len(ratios))}."
        )
    if not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0.0):
        raise RuntimeError(
            f"Fig. 3 baseline at {path} must contain finite rows on a strictly "
            "increasing energy grid."
        )
    if np.any(data[:, 1:] < 0.0) or np.any(data[:, 1:] > 1.0):
        raise RuntimeError(
            f"Fig. 3 baseline at {path} occupations must lie in inclusive "
            "[0, 1]."
        )
    # Column layout: E_uev, f_FD, f_ratio_<r0>, f_ratio_<r1>, ...
    return Fig3PaperResult(
        E=_readonly_float_array(data[:, 0]),
        tau_0_pb_ns=tau_0_pb,
        ratios=ratios,
        f_by_ratio=MappingProxyType({
            r: _readonly_float_array(data[:, 2 + i])
            for i, r in enumerate(ratios)
        }),
        f_FD=_readonly_float_array(data[:, 1]),
        certificate_maxima=MappingProxyType(
            dict(metadata.certificate_maxima)
        ),
        raw_payload_sha256=metadata.raw_payload_sha256,
        producer_solve_contract_digest=metadata.producer_solve_contract_digest,
        validated_solve_contract_digest=metadata.validated_solve_contract_digest,
    )


@dataclass(frozen=True)
class BaselineMetadata:
    """The config fingerprint :func:`write_baseline` stamps into the CSV
    comment header — parsed back (or recomputed from the live config)
    without touching the data rows or running the continuation ladder.

    Comparing the live config's fingerprint against the pinned baseline's is
    the cheap preflight that lets the slow regression test reject a stale
    config/baseline pairing in seconds instead of after the several-minute
    run (see :mod:`fig6_paper` for the same pattern, where the payoff is ~14 h).
    """

    delta_0: float
    tau_0: float
    t_bath: float
    omega_0: float
    n_bar: float
    c_phot: float
    num_bins: int
    e_min_factor: float
    e_max_factor: float
    tau_0_pb_ns: float
    ratios: tuple[float, ...]
    certificate_metric_version: str
    target_backward_error_limit: float
    certificate_maxima: dict[str, float]
    raw_payload_sha256: str
    pinned_on: str
    producer_solve_contract_digest: str
    validated_solve_contract_digest: str


def read_baseline_metadata(
    path: Path | None = None,
    *,
    _publication_lock_held: bool = False,
) -> BaselineMetadata:
    """Read metadata, authenticating the complete canonical evidence triple.

    A noncanonical ``path`` remains available for staged artifacts and unit
    fixtures. Such reads validate only the supplied CSV, not the canonical
    commit record or the current source identity.
    """
    resolved = baseline_path() if path is None else path
    canonical_read = resolved.resolve() == baseline_path().resolve()
    if canonical_read and not _publication_lock_held:
        with _publication_lock(validation_record_path()):
            return read_baseline_metadata(
                resolved,
                _publication_lock_held=True,
            )
    if canonical_read:
        read_validation_record(_publication_lock_held=True)
    return _read_baseline_metadata_unlocked(resolved)


def _read_baseline_metadata_unlocked(path: Path) -> BaselineMetadata:
    """Parse a baseline CSV's comment header into a :class:`BaselineMetadata`.

    Reads only the comment block (no data rows, no solve). Raises
    ``RuntimeError`` if any stamped field is missing — an old/malformed header
    should fail loudly rather than silently skip the check.
    """
    text = _read_baseline_text(path)

    def _num(rx: re.Pattern[str], field: str) -> float:
        m = rx.search(text)
        if m is None:
            raise RuntimeError(
                f"Baseline header at {path} missing {field} metadata."
            )
        value = float(m.group(1))
        if not np.isfinite(value):
            raise RuntimeError(
                f"Baseline header at {path} has non-finite {field} metadata."
            )
        return value

    ne_m = _GRID_NE_RE.search(text)
    ratios_m = _RATIOS_RE.search(text)
    versions = _CERTIFICATE_VERSION_RE.findall(text)
    limits = _CERTIFICATE_LIMIT_RE.findall(text)
    pinned_platforms = _PIN_PLATFORM_RE.findall(text)
    producer_solve_contract_digests = (
        _PRODUCER_SOLVE_CONTRACT_DIGEST_RE.findall(text)
    )
    validated_solve_contract_digests = (
        _VALIDATED_SOLVE_CONTRACT_DIGEST_RE.findall(text)
    )
    raw_payload_sha256s = _RAW_PAYLOAD_SHA256_RE.findall(text)
    if (
        ne_m is None
        or ratios_m is None
        or len(versions) != 1
        or len(limits) != 1
        or len(pinned_platforms) != 1
        or not pinned_platforms[0].strip()
        or len(producer_solve_contract_digests) != 1
        or len(validated_solve_contract_digests) != 1
        or len(raw_payload_sha256s) != 1
    ):
        raise RuntimeError(
            f"Baseline header at {path} must contain NE / ratios and exactly "
            "one pin-platform / producer contract / validated contract / "
            "raw-payload hash / certificate metric / target record."
        )
    ratios = tuple(
        float(x.strip()) for x in ratios_m.group(1).split(",") if x.strip()
    )
    if (
        not ratios
        or not np.all(np.isfinite(ratios))
        or len(set(ratios)) != len(ratios)
    ):
        raise RuntimeError(
            f"Baseline header at {path} has empty, non-finite, or duplicate ratios."
        )
    version = versions[0]
    if version != certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION:
        raise RuntimeError(
            f"Baseline header at {path} uses unsupported certificate metric "
            f"{version!r}; expected "
            f"{certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION!r}."
        )
    target = _num(_CERTIFICATE_LIMIT_RE, "target_backward_error_limit")
    if target != TARGET_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"Baseline header at {path} has certificate target {target:.17g}; "
            f"expected {TARGET_BACKWARD_ERROR_LIMIT:.17g}."
        )
    maxima = _validate_certificate_maxima(
        _parse_certificate_maxima(text, path),
        target=target,
        context=f"Fig. 3 baseline at {path}",
    )
    num_bins = int(ne_m.group(1))
    if num_bins <= 0:
        raise RuntimeError(f"Baseline header at {path} has non-positive NE.")
    return BaselineMetadata(
        delta_0=_num(_HEADER_PARAM_RE["delta_0"], "Delta_0"),
        tau_0=_num(_HEADER_PARAM_RE["tau_0"], "tau_0"),
        t_bath=_num(_HEADER_PARAM_RE["t_bath"], "T_bath"),
        omega_0=_num(_HEADER_PARAM_RE["omega_0"], "omega_0"),
        n_bar=_num(_HEADER_PARAM_RE["n_bar"], "n_bar"),
        c_phot=_num(_HEADER_PARAM_RE["c_phot"], "c_phot"),
        num_bins=num_bins,
        e_min_factor=_num(_E_MIN_RE, "E_min"),
        e_max_factor=_num(_E_MAX_RE, "E_max"),
        tau_0_pb_ns=_num(_TAU_0_PB_RE, "tau_0_pb_ns"),
        ratios=ratios,
        certificate_metric_version=version,
        target_backward_error_limit=target,
        certificate_maxima=maxima,
        raw_payload_sha256=raw_payload_sha256s[0],
        pinned_on=pinned_platforms[0].strip(),
        producer_solve_contract_digest=producer_solve_contract_digests[0],
        validated_solve_contract_digest=validated_solve_contract_digests[0],
    )


def config_metadata() -> BaselineMetadata:
    """Fingerprint the *current module config* would stamp into a fresh
    baseline header — computed without the (several-minute) continuation run.

    ``tau_0_pb_ns`` is produced by the exact :func:`_compute_tau_0_pb` call
    :func:`run` makes, so it can never drift from a real run; everything else
    is read straight off the module constants.
    """
    _, _, spectral = _build_grid_and_spectral()
    tau_0_pb = _compute_tau_0_pb(spectral)
    contract_digest = solve_contract_digest()
    return BaselineMetadata(
        delta_0=DELTA_0,
        tau_0=TAU_0,
        t_bath=T_BATH,
        omega_0=OMEGA_0,
        n_bar=N_BAR,
        c_phot=C_PHOT,
        num_bins=NUM_BINS,
        e_min_factor=E_MIN_FACTOR,
        e_max_factor=E_MAX_FACTOR,
        tau_0_pb_ns=tau_0_pb,
        ratios=PAPER_RATIOS,
        certificate_metric_version=(
            certificate_module.NUMBER_CERTIFICATE_METRIC_VERSION
        ),
        target_backward_error_limit=TARGET_BACKWARD_ERROR_LIMIT,
        certificate_maxima={},
        raw_payload_sha256="",
        pinned_on=sys.platform,
        producer_solve_contract_digest=contract_digest,
        validated_solve_contract_digest=contract_digest,
    )


def _metadata_attestation(metadata: BaselineMetadata) -> dict[str, Any]:
    """Return the complete JSON-safe identity of one parsed CSV header."""
    return {
        "c_phot": metadata.c_phot,
        "certificate_maxima": dict(metadata.certificate_maxima),
        "certificate_metric_version": metadata.certificate_metric_version,
        "delta_0": metadata.delta_0,
        "e_max_factor": metadata.e_max_factor,
        "e_min_factor": metadata.e_min_factor,
        "n_bar": metadata.n_bar,
        "num_bins": metadata.num_bins,
        "omega_0": metadata.omega_0,
        "pinned_on": metadata.pinned_on,
        "producer_solve_contract_digest": (
            metadata.producer_solve_contract_digest
        ),
        "raw_payload_sha256": metadata.raw_payload_sha256,
        "ratios": list(metadata.ratios),
        "t_bath": metadata.t_bath,
        "target_backward_error_limit": metadata.target_backward_error_limit,
        "tau_0": metadata.tau_0,
        "tau_0_pb_ns": metadata.tau_0_pb_ns,
        "validated_solve_contract_digest": (
            metadata.validated_solve_contract_digest
        ),
    }


def _validate_metadata_matches_current_contract(
    metadata: BaselineMetadata,
    *,
    context: str,
) -> None:
    """Require every solve/config field in a CSV header to be current."""
    expected = config_metadata()
    fields = (
        "c_phot",
        "certificate_metric_version",
        "delta_0",
        "e_max_factor",
        "e_min_factor",
        "n_bar",
        "num_bins",
        "omega_0",
        "producer_solve_contract_digest",
        "ratios",
        "t_bath",
        "target_backward_error_limit",
        "tau_0",
        "tau_0_pb_ns",
        "validated_solve_contract_digest",
    )
    mismatched = [
        field
        for field in fields
        if getattr(metadata, field) != getattr(expected, field)
    ]
    if mismatched:
        raise RuntimeError(
            f"{context} does not match the current Fig. 3 contract; "
            f"mismatched fields={mismatched}."
        )


def write_plot(result: Fig3PaperResult, path: Path | None = None) -> Path:
    """Paper-style plot: log-scale f(E) with all four ratios + thermal."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if path is None or path.resolve() == plot_path().resolve():
        raise RuntimeError(
            "Direct canonical Fig. 3 PDF writes are forbidden; use "
            "publish_baseline_triple() or generate_baseline()."
        )
    _validate_result_for_artifact(result)
    path.parent.mkdir(parents=True, exist_ok=True)

    from validation.fischer_2023._paper_envelope import (
        EnvelopeParams,
        envelope_no_thermal,
        envelope_with_thermal,
        solve_b0,
    )

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    # Paper Fig. 3 axis: f(E) vs E/Δ_0 - 1 on [0, 4].
    x = result.E / DELTA_0 - 1.0

    # Paper-style palette retained from the historical standalone plotting
    # helper: solid grayscale numerical, dashed analytical.
    solid_colors = ["k", "dimgray", "gray", "lightgray"]
    dash_colors = ["red", "green", "blue", "blue"]
    Eplot = np.linspace(DELTA_0 + 1e-3, 5.0 * DELTA_0, 4000)
    TB_uev = T_BATH * KB_UEV_PER_K

    for r, sc, dc in zip(result.ratios, solid_colors, dash_colors, strict=True):
        ax.semilogy(
            x, np.maximum(result.f_by_ratio[r], 1e-40),
            color=sc, lw=1.0,
            label=rf"$\tau_\ell/\tau_0^{{\rm PB}}={r:g}$ (num)",
        )
        ep = EnvelopeParams(
            Delta0=DELTA_0,
            Tc_uev=T_C * KB_UEV_PER_K,
            tau0=TAU_0,
            tau0_PB=result.tau_0_pb_ns,
            tau_l=r * result.tau_0_pb_ns,
            TB_uev=TB_uev,
            nbar=N_BAR,
            omega0=OMEGA_0,
            cphot_QP=C_PHOT,
        )
        b0 = solve_b0(ep)
        f_env = (envelope_with_thermal(Eplot, ep, b0) if r == 0.0
                 else envelope_no_thermal(Eplot, ep, b0))
        ax.semilogy(Eplot / DELTA_0 - 1.0, np.maximum(f_env, 1e-40),
                    color=dc, ls="--", lw=0.8)

    ax.set_xlabel(r"$E/\Delta_0 - 1$")
    ax.set_ylabel(r"$f(E)$")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(1e-35, 3e-7)
    ax.legend(fontsize=8, loc="lower left")

    fig.tight_layout()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        fig.savefig(
            temporary,
            format=path.suffix.removeprefix("."),
            bbox_inches="tight",
        )
        temporary.replace(path)
    finally:
        plt.close(fig)
        temporary.unlink(missing_ok=True)
    return path


def _strict_staged_readback(
    result: Fig3PaperResult,
    *,
    csv_path: Path,
) -> dict[str, Any]:
    restored = read_baseline(csv_path)
    metadata = read_baseline_metadata(csv_path)
    if (
        not np.array_equal(restored.E, result.E)
        or not np.array_equal(restored.f_FD, result.f_FD)
        or restored.ratios != result.ratios
        or restored.tau_0_pb_ns != result.tau_0_pb_ns
        or restored.certificate_maxima != result.certificate_maxima
        or restored.raw_payload_sha256 != result.raw_payload_sha256
        or restored.producer_solve_contract_digest
        != result.producer_solve_contract_digest
        or restored.validated_solve_contract_digest
        != result.validated_solve_contract_digest
    ):
        raise RuntimeError("Staged Fig. 3 CSV failed exact strict readback.")
    for ratio in result.ratios:
        if not np.array_equal(restored.f_by_ratio[ratio], result.f_by_ratio[ratio]):
            raise RuntimeError(
                f"Staged Fig. 3 CSV changed the ratio {ratio:g} curve."
            )
    return {
        "arrays": {
            "E": _sha256_array(restored.E),
            "f_FD": _sha256_array(restored.f_FD),
            "f_by_ratio": {
                f"{ratio:.17g}": _sha256_array(restored.f_by_ratio[ratio])
                for ratio in restored.ratios
            },
        },
        "strict_read_passed": True,
        "metadata": _metadata_attestation(metadata),
    }


def _strict_json_load(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard numeric constant {value!r}")

    try:
        decoded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"Fig. 3 validation record at {path} is invalid: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError(f"Fig. 3 validation record at {path} is not an object.")
    return decoded


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_reassembly_evidence(
    reassembly: Mapping[str, Any],
    *,
    context: str,
    result: Fig3PaperResult,
) -> None:
    """Reject incomplete or internally inconsistent per-ratio evidence."""
    rows = reassembly.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != len(PAPER_RATIOS)
        or not isinstance(reassembly.get("scope"), str)
        or not reassembly["scope"].strip()
        or not isinstance(reassembly.get("qualification"), str)
        or not reassembly["qualification"].strip()
    ):
        raise RuntimeError(f"{context} has incomplete reassembly evidence.")

    values_by_field: dict[str, list[float]] = {
        field: [] for field in certificate_module.NUMBER_CERTIFICATE_FIELDS
    }
    for expected_ratio, row in zip(PAPER_RATIOS, rows, strict=True):
        if (
            not isinstance(row, dict)
            or set(row)
            != {
                "ratio",
                "persisted_f_sha256",
                "certificate",
                "reconstructed_n_ph",
            }
            or isinstance(row.get("ratio"), bool)
            or not isinstance(row.get("ratio"), (int, float))
            or float(row["ratio"]) != expected_ratio
            or not isinstance(row.get("persisted_f_sha256"), str)
            or row["persisted_f_sha256"]
            != _sha256_array(result.f_by_ratio[expected_ratio])
        ):
            raise RuntimeError(
                f"{context} has malformed evidence for ratio {expected_ratio:g}."
            )
        certificate = row["certificate"]
        if (
            not isinstance(certificate, dict)
            or set(certificate)
            != set(certificate_module.NUMBER_CERTIFICATE_FIELDS)
        ):
            raise RuntimeError(
                f"{context} has an incomplete certificate for ratio "
                f"{expected_ratio:g}."
            )
        for field in certificate_module.NUMBER_CERTIFICATE_FIELDS:
            value = certificate[field]
            expects_null = expected_ratio == 0.0 and field.startswith("phonon_")
            if expects_null:
                if value is not None:
                    raise RuntimeError(
                        f"{context} ratio-zero field {field!r} must be null."
                    )
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise RuntimeError(
                    f"{context} ratio {expected_ratio:g} field {field!r} "
                    "must be finite and non-negative."
                )
            values_by_field[field].append(float(value))

        reconstructed = row["reconstructed_n_ph"]
        if (
            not isinstance(reconstructed, dict)
            or set(reconstructed) != {"max", "min", "sha256"}
            or isinstance(reconstructed.get("min"), bool)
            or isinstance(reconstructed.get("max"), bool)
            or not isinstance(reconstructed.get("min"), (int, float))
            or not isinstance(reconstructed.get("max"), (int, float))
            or not np.isfinite(float(reconstructed["min"]))
            or not np.isfinite(float(reconstructed["max"]))
            or float(reconstructed["min"]) < 0.0
            or float(reconstructed["max"]) < float(reconstructed["min"])
            or not isinstance(reconstructed.get("sha256"), str)
            or _SHA256_RE.fullmatch(reconstructed["sha256"]) is None
        ):
            raise RuntimeError(
                f"{context} has invalid reconstructed phonons for ratio "
                f"{expected_ratio:g}."
            )

    maxima = _validate_certificate_maxima(
        reassembly.get("maxima", {}),
        target=TARGET_BACKWARD_ERROR_LIMIT,
        context=context,
    )
    recomputed_maxima = {
        field: max(values)
        for field, values in values_by_field.items()
    }
    if maxima != recomputed_maxima:
        raise RuntimeError(
            f"{context} certificate maxima do not match the per-ratio rows."
        )


def _validate_producer_payload_evidence(
    evidence: Any,
    *,
    context: str,
    expected_raw_payload_sha256: str | None,
    require_current_runtime_identity: bool = False,
) -> None:
    """Validate the cache/restart origin record without claiming a fresh solve."""
    if (
        not isinstance(evidence, dict)
        or set(evidence) != {"cache", "restart"}
        or not isinstance(evidence.get("cache"), dict)
        or not isinstance(evidence.get("restart"), dict)
    ):
        raise RuntimeError(f"{context} has incomplete payload-origin evidence.")
    cache = evidence["cache"]
    canonical_kwargs, canonical_fingerprint, canonical_extra_source, canonical_key = (
        _solve_cache_inputs(
            num_bins=NUM_BINS,
            paper_ratios=PAPER_RATIOS,
            continuation_ratios=CONTINUATION_RATIOS,
        )
    )
    if (
        cache.get("execution_mode")
        not in {"solver_invoked", "authenticated_cache_hit"}
        or not isinstance(cache.get("cache_enabled"), bool)
        or not isinstance(cache.get("array_payload_sha256"), str)
        or _SHA256_RE.fullmatch(cache["array_payload_sha256"]) is None
    ):
        raise RuntimeError(f"{context} has invalid cache execution evidence.")
    if (
        expected_raw_payload_sha256 is not None
        and cache["array_payload_sha256"] != expected_raw_payload_sha256
    ):
        raise RuntimeError(f"{context} is not bound to the raw solve payload.")
    if cache["cache_enabled"]:
        identity = cache.get("producer_identity")
        if (
            not isinstance(cache.get("cache_key"), str)
            or _SHA256_RE.fullmatch(cache["cache_key"]) is None
            or not isinstance(identity, dict)
            or identity.get("format_version") != sweep_cache._FORMAT_VERSION
            or identity.get("figure") != "fischer_2023/fig3"
            or identity.get("fingerprint") != canonical_fingerprint
            or identity.get("kwargs") != canonical_kwargs
            or identity.get("solve_source") != sweep_cache.solve_source_digest()
            or identity.get("extra_source")
            != hashlib.sha256(
                source_provenance.canonical_source_text(
                    canonical_extra_source
                ).encode()
            ).hexdigest()
            or not isinstance(identity.get("versions"), dict)
            or not isinstance(identity.get("numeric_runtime"), dict)
            or cache["cache_key"] != sweep_cache._cache_key_from_identity(identity)
        ):
            raise RuntimeError(f"{context} has invalid cache identity evidence.")
        if require_current_runtime_identity and cache["cache_key"] != canonical_key:
            raise RuntimeError(
                f"{context} cache identity is not the current canonical request."
            )
        if cache["execution_mode"] == "authenticated_cache_hit":
            provenance = cache.get("cache_provenance")
            if (
                not isinstance(provenance, dict)
                or set(provenance)
                != {
                    "array_payload_sha256",
                    "created_utc",
                    "extra_source",
                    "figure",
                    "fingerprint",
                    "key",
                    "kwargs",
                    "numeric_runtime",
                    "solve_source",
                    "versions",
                }
                or provenance.get("key") != cache["cache_key"]
                or provenance.get("array_payload_sha256")
                != cache["array_payload_sha256"]
                or any(
                    provenance.get(field) != identity.get(field)
                    for field in (
                        "extra_source",
                        "figure",
                        "fingerprint",
                        "kwargs",
                        "numeric_runtime",
                        "solve_source",
                        "versions",
                    )
                )
            ):
                raise RuntimeError(
                    f"{context} has inconsistent authenticated-cache evidence."
                )
    elif cache["execution_mode"] != "solver_invoked":
        raise RuntimeError(
            f"{context} claims a cache hit while the cache was disabled."
        )

    restart = evidence["restart"]
    if (
        set(restart)
        != {
            "canonical_request",
            "checkpoint_existed_before",
            "checkpoint_identity",
            "checkpoint_path",
            "qualification",
        }
        or restart["canonical_request"] != canonical_kwargs
        or not isinstance(restart["checkpoint_existed_before"], bool)
        or not isinstance(restart["checkpoint_identity"], str)
        or _SHA256_RE.fullmatch(restart["checkpoint_identity"]) is None
        or (
            restart["checkpoint_path"] is not None
            and not isinstance(restart["checkpoint_path"], str)
        )
        or not isinstance(restart["qualification"], str)
        or not restart["qualification"].strip()
    ):
        raise RuntimeError(f"{context} has invalid restart evidence.")
    if (
        cache["cache_enabled"]
        and restart["checkpoint_identity"] != cache["cache_key"]
    ):
        raise RuntimeError(
            f"{context} cache and restart identities do not describe one request."
        )
    if (
        require_current_runtime_identity
        and restart["checkpoint_identity"] != canonical_key
    ):
        raise RuntimeError(
            f"{context} restart identity is not the current canonical request."
        )


def read_validation_record(
    path: Path | None = None,
    *,
    csv_path: Path | None = None,
    pdf_path: Path | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
    _publication_lock_held: bool = False,
) -> dict[str, Any]:
    """Authenticate a complete current Fig. 3 CSV/PDF/record triple."""
    if path is None:
        path = validation_record_path()
    if csv_path is None:
        csv_path = baseline_path()
    if pdf_path is None:
        pdf_path = plot_path()
    canonical_snapshot = (
        path.resolve() == validation_record_path().resolve()
        and csv_path.resolve() == baseline_path().resolve()
        and pdf_path.resolve() == plot_path().resolve()
    )
    if canonical_snapshot and not _publication_lock_held:
        with _publication_lock(validation_record_path()):
            return read_validation_record(
                path,
                csv_path=csv_path,
                pdf_path=pdf_path,
                expected_source_identity=expected_source_identity,
                _publication_lock_held=True,
            )
    expected = (
        dict(expected_source_identity)
        if expected_source_identity is not None
        else _validation_source_identity()
    )
    record_sha256_before = _sha256_file(path)
    record = _strict_json_load(path)
    if record.get("schema") != VALIDATION_RECORD_SCHEMA or record.get("status") != "pass":
        raise RuntimeError(
            f"Fig. 3 validation record at {path} has stale schema/status."
        )
    producer = record.get("producer")
    verifier = record.get("verifier")
    producer_runtime = (
        producer.get("runtime") if isinstance(producer, dict) else None
    )
    if (
        not isinstance(producer, dict)
        or producer.get("current_solve_contract_payload") is not True
        or producer.get("source_identity") != expected
        or not isinstance(producer_runtime, dict)
        or not isinstance(verifier, dict)
        or verifier.get("source_unchanged") is not True
        or verifier.get("runtime_unchanged") is not True
        or verifier.get("source_identity_before") != expected
        or verifier.get("source_identity_after") != expected
        or verifier.get("runtime_before") != producer_runtime
        or verifier.get("runtime_after") != producer_runtime
    ):
        raise RuntimeError(
            f"Fig. 3 validation record at {path} is not bound to current source."
        )
    _validate_producer_payload_evidence(
        producer.get("payload_origin"),
        context=f"Fig. 3 validation record at {path}",
        expected_raw_payload_sha256=None,
    )
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"Fig. 3 validation record at {path} has no artifact map.")
    for name, artifact_path in (("csv", csv_path), ("pdf", pdf_path)):
        attestation = artifacts.get(name)
        if (
            not isinstance(attestation, dict)
            or attestation.get("size_bytes") != artifact_path.stat().st_size
            or attestation.get("sha256") != _sha256_file(artifact_path)
        ):
            raise RuntimeError(
                f"Fig. 3 validation record at {path} does not authenticate "
                f"{artifact_path}."
            )
    pdf_bytes = pdf_path.read_bytes()
    try:
        validate_single_nonempty_matplotlib_pdf(pdf_bytes, path=pdf_path)
    except ValueError as exc:
        raise RuntimeError(
            f"Fig. 3 companion at {pdf_path} is not a complete "
            "one-page Matplotlib PDF."
        ) from exc
    metadata = _read_baseline_metadata_unlocked(csv_path)
    _validate_metadata_matches_current_contract(
        metadata,
        context=f"Fig. 3 CSV at {csv_path}",
    )
    solve_digest = expected["solve_contract_digest"]
    if (
        metadata.producer_solve_contract_digest != solve_digest
        or metadata.validated_solve_contract_digest != solve_digest
    ):
        raise RuntimeError(
            f"Fig. 3 CSV at {csv_path} has a stale producer/validated "
            "solve contract."
        )
    restored = _read_baseline_unlocked(csv_path)
    _E, _dE, spectral = fig3_solve._build_grid_and_spectral(NUM_BINS)
    _validate_persisted_grid_and_thermal_reference(
        restored,
        expected_E=spectral.E,
    )
    if (
        producer["payload_origin"]["cache"]["array_payload_sha256"]
        != metadata.raw_payload_sha256
    ):
        raise RuntimeError(
            f"Fig. 3 CSV at {csv_path} is not bound to the recorded raw payload."
        )
    readback = artifacts.get("readback")
    expected_array_hashes = {
        "E": _sha256_array(restored.E),
        "f_FD": _sha256_array(restored.f_FD),
        "f_by_ratio": {
            f"{ratio:.17g}": _sha256_array(restored.f_by_ratio[ratio])
            for ratio in restored.ratios
        },
    }
    if (
        not isinstance(readback, dict)
        or readback.get("strict_read_passed") is not True
        or readback.get("metadata") != _metadata_attestation(metadata)
        or readback.get("arrays") != expected_array_hashes
    ):
        raise RuntimeError(
            f"Fig. 3 validation record at {path} has stale readback evidence."
        )
    reassembly = record.get("fresh_certificate_reassembly")
    if not isinstance(reassembly, dict):
        raise RuntimeError(
            f"Fig. 3 validation record at {path} has incomplete reassembly evidence."
        )
    _validate_reassembly_evidence(
        reassembly,
        context=f"Fig. 3 validation record at {path}",
        result=restored,
    )
    for name, artifact_path in (("csv", csv_path), ("pdf", pdf_path)):
        attestation = artifacts[name]
        if (
            attestation["size_bytes"] != artifact_path.stat().st_size
            or attestation["sha256"] != _sha256_file(artifact_path)
        ):
            raise RuntimeError(
                "Fig. 3 evidence changed while it was being authenticated."
            )
    if record_sha256_before != _sha256_file(path):
        raise RuntimeError(
            "Fig. 3 validation record changed while it was being authenticated."
        )
    return record


def _restore_artifact(path: Path, payload: bytes | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.rollback")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _publication_lock(record_path: Path) -> Iterator[None]:
    """Serialize canonical publishers with an OS lock released on process exit."""
    lock_path = record_path.with_suffix(record_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(  # type: ignore[attr-defined]
                    stream.fileno(),
                    fcntl.LOCK_EX | fcntl.LOCK_NB,  # type: ignore[attr-defined]
                )
        except OSError as exc:
            raise RuntimeError(
                f"Another Fig. 3 publisher holds {lock_path}; refusing "
                "concurrent canonical publication."
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(  # type: ignore[attr-defined]
                    stream.fileno(),
                    fcntl.LOCK_UN,  # type: ignore[attr-defined]
                )


def publish_baseline_triple(
    result: Fig3PaperResult,
    *,
    source_identity: dict[str, Any],
    producer_runtime: dict[str, Any],
    producer_evidence: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Serialize and publish one authenticated Fig. 3 evidence triple."""
    with _publication_lock(validation_record_path()):
        return _publish_baseline_triple_locked(
            result,
            source_identity=source_identity,
            producer_runtime=producer_runtime,
            producer_evidence=producer_evidence,
        )


def _publish_baseline_triple_locked(
    result: Fig3PaperResult,
    *,
    source_identity: dict[str, Any],
    producer_runtime: dict[str, Any],
    producer_evidence: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Stage, verify, and promote Fig. 3 PDF -> CSV -> validation record."""
    csv_path = baseline_path()
    pdf_path = plot_path()
    record_path = validation_record_path()
    if (
        source_identity != _validation_source_identity()
        or producer_runtime != _validation_runtime_provenance()
    ):
        raise RuntimeError(
            "Fig. 3 source/configuration/runtime changed during the solve; "
            "refusing mixed-provenance publication."
        )
    _validate_producer_payload_evidence(
        producer_evidence,
        context="Fig. 3 publisher",
        expected_raw_payload_sha256=result.raw_payload_sha256,
        require_current_runtime_identity=True,
    )
    solve_digest = str(source_identity["solve_contract_digest"])
    if (
        result.producer_solve_contract_digest != solve_digest
        or result.validated_solve_contract_digest != solve_digest
    ):
        raise RuntimeError(
            "Fig. 3 result is not an exact current-source solve-contract payload."
        )
    stages = {
        "csv": csv_path.with_name(f".{csv_path.stem}.{os.getpid()}.stage.csv"),
        "pdf": pdf_path.with_name(f".{pdf_path.stem}.{os.getpid()}.stage.pdf"),
        "record": record_path.with_name(
            f".{record_path.stem}.{os.getpid()}.stage.json"
        ),
    }
    for stage in stages.values():
        stage.unlink(missing_ok=True)
    try:
        write_plot(result, stages["pdf"])
        pdf_bytes = stages["pdf"].read_bytes()
        try:
            validate_single_nonempty_matplotlib_pdf(
                pdf_bytes,
                path=stages["pdf"],
            )
        except ValueError as exc:
            raise RuntimeError(
                "Staged Fig. 3 plot is not a complete one-page "
                "Matplotlib PDF."
            ) from exc
        pdf_identity = {
            "sha256": _sha256_file(stages["pdf"]),
            "size_bytes": stages["pdf"].stat().st_size,
        }
        write_baseline(result, stages["csv"])
        if pdf_identity != {
            "sha256": _sha256_file(stages["pdf"]),
            "size_bytes": stages["pdf"].stat().st_size,
        }:
            raise RuntimeError(
                "Staged Fig. 3 PDF changed while its CSV was written."
            )
        readback = _strict_staged_readback(result, csv_path=stages["csv"])
        staged_identity = {
            "csv": {
                "sha256": _sha256_file(stages["csv"]),
                "size_bytes": stages["csv"].stat().st_size,
            },
            "pdf": pdf_identity,
        }
        reassembly = _reassemble_artifact_certificates(result)
        reassembly = {
            **reassembly,
            "maxima": dict(reassembly["maxima"]),
        }
        if any(
            staged_identity[name]
            != {
                "sha256": _sha256_file(stages[name]),
                "size_bytes": stages[name].stat().st_size,
            }
            for name in ("csv", "pdf")
        ):
            raise RuntimeError(
                "Staged Fig. 3 CSV/PDF changed during semantic validation."
            )
        source_after = _validation_source_identity()
        runtime_after = _validation_runtime_provenance()
        if source_after != source_identity or runtime_after != producer_runtime:
            raise RuntimeError(
                "Fig. 3 source/configuration/runtime changed during validation; "
                "discarding staged artifacts."
            )
        record = {
            "artifacts": {
                "csv": {
                    **staged_identity["csv"],
                },
                "pdf": {
                    **staged_identity["pdf"],
                },
                "readback": readback,
            },
            "assumptions_and_gaps": [
                reassembly["qualification"],
                (
                    "This is an authenticated qpsim code-regression artifact at "
                    "the paper parameter topology, not digitized-paper-data parity."
                ),
            ],
            "claim": (
                "The source/configuration-bound current Fig. 3 solve was "
                "strictly read back and freshly recertified at the Ph0 fixed "
                "point implied by every persisted occupation curve. The fresh "
                "pass uses the shared certificate implementation; it is not "
                "an independent implementation."
            ),
            "fresh_certificate_reassembly": reassembly,
            "producer": {
                "current_solve_contract_payload": True,
                "payload_origin": producer_evidence,
                "runtime": producer_runtime,
                "source_identity": source_identity,
            },
            "schema": VALIDATION_RECORD_SCHEMA,
            "status": "pass",
            "verifier": {
                "runtime_after": runtime_after,
                "runtime_before": producer_runtime,
                "runtime_unchanged": True,
                "source_identity_after": source_after,
                "source_identity_before": source_identity,
                "source_unchanged": True,
            },
        }
        with _atomic_text_file(stages["record"]) as stream:
            stream.write(
                json.dumps(
                    record,
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        read_validation_record(
            stages["record"],
            csv_path=stages["csv"],
            pdf_path=stages["pdf"],
            expected_source_identity=source_identity,
        )
        if (
            _validation_source_identity() != source_identity
            or _validation_runtime_provenance() != producer_runtime
        ):
            raise RuntimeError(
                "Fig. 3 source/configuration/runtime changed immediately "
                "before promotion; discarding staged artifacts."
            )
        old_payloads = {
            path: path.read_bytes() if path.is_file() else None
            for path in (pdf_path, csv_path, record_path)
        }
        try:
            os.replace(stages["pdf"], pdf_path)
            os.replace(stages["csv"], csv_path)
            os.replace(stages["record"], record_path)
            if (
                _validation_source_identity() != source_identity
                or _validation_runtime_provenance() != producer_runtime
            ):
                raise RuntimeError(
                    "Fig. 3 source/configuration/runtime changed during "
                    "promotion; restoring the prior evidence triple."
                )
            read_validation_record(
                record_path,
                csv_path=csv_path,
                pdf_path=pdf_path,
                _publication_lock_held=True,
            )
        except BaseException:
            for artifact_path, payload in old_payloads.items():
                _restore_artifact(artifact_path, payload)
            raise
        return csv_path, pdf_path, record_path
    finally:
        for stage in stages.values():
            stage.unlink(missing_ok=True)


def generate_baseline() -> tuple[Path, Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    print("Fischer 2023 Fig. 3 — paper-topology qpsim numerical regression ...")
    print(
        f"  Δ_0={DELTA_0} μeV, τ_0={TAU_0} ns, T_bath={T_BATH} K, "
        f"ω_0={OMEGA_0:.2f} μeV"
    )
    print(f"  Grid: NE={NUM_BINS}, dE={(E_MAX_FACTOR-E_MIN_FACTOR)*DELTA_0/NUM_BINS:.3f} μeV")
    print(f"  Paper ratios: {list(PAPER_RATIOS)}")
    source_identity = _validation_source_identity()
    producer_runtime = _validation_runtime_provenance()
    result, producer_evidence = run_cached_with_evidence()
    csv_path, pdf_path, record_path = publish_baseline_triple(
        result,
        source_identity=source_identity,
        producer_runtime=producer_runtime,
        producer_evidence=producer_evidence,
    )
    print(
        "  Payload mode: "
        f"{producer_evidence['cache']['execution_mode']}"
    )
    print(f"  Baseline CSV: {csv_path}")
    print(f"  PDF plot:     {pdf_path}")
    print(f"  Validation:   {record_path}")
    return csv_path, pdf_path, record_path


if __name__ == "__main__":
    generate_baseline()
