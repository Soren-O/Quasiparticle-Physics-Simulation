"""Strict artifact and quasiparticle-balance helpers for Fischer 2024.

The Fischer 2024 scripts predate qpsim's certified validation artifacts.  This
module supplies the common, deliberately small contract used by all four
scripts:

* an independently reassembled thermal-phonon + pair-breaking-photon balance;
* a versioned L1 gain/loss backward error and maximum absolute residual; and
* an atomic, fail-closed CSV container with an exact configuration fingerprint.

Legacy CSVs intentionally do not satisfy this contract.  A reader must reject
them instead of treating an uncertified numerical snapshot as a current pin.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO, cast

import matplotlib
import numpy as np
import scipy
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.pair_breaking_photon import (
    pair_breaking_photon_collision_components,
    pair_breaking_photon_collision_rates,
)
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.physics.kernels import thermal_phonon_occupation

from validation.fischer_2024._pdf import validate_single_nonempty_matplotlib_pdf
from validation.source_provenance import source_manifest

QP_CERTIFICATE_METRIC_VERSION = (
    "qp-gain-loss-l1-maxabs-pair-number-thermal-pb-v2"
)
# Repo-wide acceptance limits. They are deliberately module constants and
# not caller-parameterizable (unlike ``residual_inf_limit``): they are
# written into the artifact metadata and re-checked on read, so a promoted
# artifact advertises one acceptance gate that a later reader cannot
# loosen. A driver must therefore buy its headroom on the *producer* side,
# by setting its own NEWTON_BACKWARD_ERROR_TOL at or below 0.1x these
# limits — otherwise the solve stops on the first iterate under the gate,
# one Newton step short of the tight root, and the acceptance check adds
# no independent evidence. fig5_paper.py does this (1e-7, after the
# 2026-07-28 hosted-Linux branch incident); fig8_paper.py still sits at
# 1e-6 with a promoted maximum at 73% of the limit and is pending
# re-promotion. fig8_xqp_pb.py and figs_5_7_fe_pb.py are unaffected —
# their promoted maxima are 1.2e-11 and 4.5e-13, so their gate never binds.
TARGET_QP_BACKWARD_ERROR_LIMIT = 1.0e-6
TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT = 1.0e-6
TARGET_QP_RESIDUAL_INF_LIMIT = 1.0e-10
THERMAL_OCCUPATION_RTOL = 8.0 * np.finfo(np.float64).eps


def thermal_occupations_match(values: np.ndarray, expected: np.ndarray) -> bool:
    """Return whether two thermal seeds agree up to cross-libm roundoff.

    Fermi-Dirac seeds contain ``exp`` evaluations whose final bits may differ
    between otherwise equivalent math libraries.  Artifact checksums remain
    exact; this semantic live-state check admits only a small multiple of
    machine epsilon and no absolute floor.
    """
    actual_raw = np.asarray(values)
    reference_raw = np.asarray(expected)
    if np.iscomplexobj(actual_raw) or np.iscomplexobj(reference_raw):
        return False
    try:
        actual = np.asarray(actual_raw, dtype=float)
        reference = np.asarray(reference_raw, dtype=float)
    except (TypeError, ValueError):
        return False
    return (
        actual.shape == reference.shape
        and bool(np.all(np.isfinite(actual)))
        and bool(np.all(np.isfinite(reference)))
        and bool(
            np.allclose(
                actual,
                reference,
                rtol=THERMAL_OCCUPATION_RTOL,
                atol=0.0,
            )
        )
    )


def source_hashes(
    validation_module: Path,
    *,
    extra_validation_modules: Sequence[Path] = (),
) -> dict[str, str]:
    """Hash the validation modules and numerical sources defining a solve.

    Every qpsim Python module and material YAML input is listed explicitly.
    This conservative tree manifest closes omissions in a hand-maintained
    import list (including dynamic or future dependencies) and keeps each
    contributing file independently auditable. It deliberately prefers
    harmless over-invalidation to stale numerical evidence. Source newlines
    are normalized so LF and CRLF checkouts have the same identity.
    """
    return source_manifest(
        validation_module,
        extra_validation_modules=(
            Path(__file__),
            Path(__file__).with_name("_pdf.py"),
            *extra_validation_modules,
        ),
    )


class ArtifactValidationError(RuntimeError):
    """Raised when a persisted validation artifact is not current and valid."""


class LegacyArtifactError(ArtifactValidationError):
    """Raised only for artifacts that predate the qpsim schema markers."""


@dataclass(frozen=True)
class QPCertificate:
    """Independent certificate for one returned quasiparticle state."""

    backward_error: float
    residual_inf: float
    qp_number_backward_error: float


@dataclass(frozen=True)
class ArtifactTable:
    """Strictly decoded artifact payload."""

    data: np.ndarray
    certificates: dict[str, QPCertificate]


@dataclass(frozen=True)
class ProducerIdentity:
    """Frozen numerical inputs and runtime captured before an expensive solve.

    ``fingerprint`` is the portable, correctness-bearing source/configuration
    identity used by readers to reject stale artifacts. ``runtime`` records
    the producer host and numerical stack for diagnosis and reproducibility,
    but readers deliberately do not require their current host to match it.
    """

    fingerprint: dict[str, Any]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class CompanionArtifactRecord:
    """Content identity for the staged PDF paired with a CSV commit marker."""

    sha256: str
    size_bytes: int


_THREAD_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_CBWR",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "OPENBLAS_NUM_THREADS",
)


def _numeric_build_provenance(module: Any) -> dict[str, Any]:
    """Return portable BLAS/LAPACK/compiler facts from NumPy-style config."""
    config = getattr(getattr(module, "__config__", None), "CONFIG", {})
    if not isinstance(config, Mapping):
        return {}
    dependencies = config.get("Build Dependencies", {})
    dependency_result: dict[str, Any] = {}
    if isinstance(dependencies, Mapping):
        for name in ("blas", "lapack"):
            raw = dependencies.get(name, {})
            if not isinstance(raw, Mapping):
                continue
            dependency_result[name] = {
                key: raw[key]
                for key in (
                    "detection method",
                    "found",
                    "name",
                    "openblas configuration",
                    "version",
                )
                if key in raw
            }
    compilers = config.get("Compilers", {})
    compiler_result: dict[str, Any] = {}
    if isinstance(compilers, Mapping):
        for name, raw in compilers.items():
            if isinstance(raw, Mapping):
                compiler_result[str(name)] = {
                    key: raw[key]
                    for key in ("name", "version")
                    if key in raw
                }
    simd = config.get("SIMD Extensions", {})
    simd_result = dict(simd) if isinstance(simd, Mapping) else {}
    return {
        "build_dependencies": dependency_result,
        "compilers": compiler_result,
        "simd_extensions": simd_result,
    }


def producer_runtime_provenance() -> dict[str, Any]:
    """Capture the numerical runtime without installation-specific paths."""
    multiarray = getattr(getattr(np, "_core", None), "_multiarray_umath", None)
    raw_cpu_features = getattr(multiarray, "__cpu_features__", {})
    cpu_features = (
        {
            str(name): bool(enabled)
            for name, enabled in sorted(raw_cpu_features.items())
        }
        if isinstance(raw_cpu_features, Mapping)
        else {}
    )
    return {
        "matplotlib": {
            "backend": str(matplotlib.get_backend()),
            "freetype": getattr(
                getattr(matplotlib, "ft2font", None),
                "__freetype_version__",
                None,
            ),
            "version": matplotlib.__version__,
        },
        "numpy": {
            "build": _numeric_build_provenance(np),
            "version": np.__version__,
        },
        "platform": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "release": platform.release(),
            "system": platform.system(),
        },
        "python": {
            "byteorder": sys.byteorder,
            "compiler": platform.python_compiler(),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "runtime_cpu_features": cpu_features,
        "scipy": {
            "build": _numeric_build_provenance(scipy),
            "version": scipy.__version__,
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in _THREAD_ENVIRONMENT_KEYS
        },
    }


def _json_copy(value: Any) -> Any:
    """Return an immutable-by-convention JSON round-trip copy."""
    return json.loads(_canonical_json(value))


def capture_producer_identity(fingerprint: Mapping[str, Any]) -> ProducerIdentity:
    """Freeze source/configuration and runtime identities before a solve."""
    # PDF bytes are part of the promoted artifact pair. Select the declared
    # non-interactive renderer before freezing the producer runtime.
    matplotlib.use("Agg")
    return ProducerIdentity(
        fingerprint=cast(dict[str, Any], _json_copy(dict(fingerprint))),
        runtime=cast(dict[str, Any], _json_copy(producer_runtime_provenance())),
    )


def assert_producer_identity_current(
    producer: ProducerIdentity,
    fingerprint: Mapping[str, Any],
) -> None:
    """Refuse promotion when source, configuration, or runtime changed."""
    current = capture_producer_identity(fingerprint)
    if producer.fingerprint != current.fingerprint:
        raise ArtifactValidationError(
            "Fischer 2024 producer fingerprint changed during the solve; "
            "discard the result and rerun from a stable source/configuration tree."
        )
    if producer.runtime != current.runtime:
        raise ArtifactValidationError(
            "Fischer 2024 numerical runtime changed during the solve; "
            "discard the result and rerun in one stable runtime."
        )


def require_staging_path(
    path: Path,
    canonical_path: Path,
    *,
    artifact_kind: str,
) -> Path:
    """Reject direct writes to a canonical artifact commit path.

    Canonical CSV/PDF pairs are published only by :func:`publish_artifact_pair`,
    which renders both payloads off-path, binds the CSV to the staged PDF, and
    replaces the CSV last. A public single-file writer must therefore never
    be able to invalidate one half of the canonical pair.
    """
    if path.resolve() == canonical_path.resolve():
        raise ArtifactValidationError(
            f"Direct writes to the canonical Fischer 2024 {artifact_kind} "
            "are forbidden; use generate_baseline() so the authenticated "
            "CSV/PDF pair is staged and promoted together."
        )
    return path


def companion_artifact_record(path: Path) -> CompanionArtifactRecord:
    """Parse and hash exactly one complete, nonempty Matplotlib PDF page."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactValidationError(
            f"Cannot read staged companion PDF at {path}: {exc}"
        ) from exc
    try:
        validate_single_nonempty_matplotlib_pdf(payload, path=path)
    except ValueError as exc:
        raise ArtifactValidationError(
            f"Companion artifact at {path} is not a valid one-page "
            "Matplotlib PDF."
        ) from exc
    return CompanionArtifactRecord(
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _require_staged_csv_pdf_binding(
    path: Path,
    expected: CompanionArtifactRecord,
) -> None:
    """Verify the staged CSV actually records the staged PDF identity."""
    try:
        with path.open(encoding="utf-8", newline="") as fp:
            rows = list(csv.reader(fp))
        if (
            len(rows) < 2
            or len(rows[1]) != 1
            or not rows[1][0].startswith("# qpsim_metadata=")
        ):
            raise ValueError("missing metadata row")
        metadata = json.loads(rows[1][0].split("=", 1)[1])
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Staged CSV at {path} is incomplete or malformed."
        ) from exc
    if not isinstance(metadata, dict) or metadata.get("companion_pdf") != {
        "sha256": expected.sha256,
        "size_bytes": expected.size_bytes,
    }:
        raise ArtifactValidationError(
            f"Staged CSV at {path} does not bind the staged companion PDF."
        )


@contextmanager
def _artifact_pair_lock(csv_path: Path, *, operation: str) -> Iterator[None]:
    """Hold the OS lock shared by canonical pair publishers and readers.

    The CSV is the pair's commit marker, so its adjacent ``.lock`` file is
    the stable rendezvous point even while the CSV/PDF payloads are replaced.
    The lock is deliberately non-blocking: validation should fail loudly
    instead of waiting on, or sampling state from, another publication.
    OS locks are released automatically if a process exits unexpectedly.
    """
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
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
            raise ArtifactValidationError(
                f"Fischer 2024 artifact pair at {csv_path} is locked by "
                f"another reader or publisher; refusing concurrent {operation}."
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


def publish_artifact_pair(
    *,
    csv_path: Path,
    pdf_path: Path,
    producer: ProducerIdentity,
    current_fingerprint: Callable[[], Mapping[str, Any]],
    render_pdf: Callable[[Path], Path],
    write_csv: Callable[
        [Path, ProducerIdentity, CompanionArtifactRecord],
        Path,
    ],
) -> tuple[Path, Path]:
    """Stage and promote a PDF/CSV pair, with the CSV as commit marker.

    Both payloads are completed off-path. The staged PDF's exact hash and
    byte count are passed to ``write_csv`` for inclusion in CSV metadata.
    After a final source/runtime drift check, the PDF is promoted first and
    the binding CSV last. A crash in between is fail-closed: the old CSV
    cannot authenticate the new PDF. An OS lock shared with
    :func:`read_artifact` covers the complete transaction.
    """
    with _artifact_pair_lock(csv_path, operation="publication"):
        return _publish_artifact_pair_locked(
            csv_path=csv_path,
            pdf_path=pdf_path,
            producer=producer,
            current_fingerprint=current_fingerprint,
            render_pdf=render_pdf,
            write_csv=write_csv,
        )


def _publish_artifact_pair_locked(
    *,
    csv_path: Path,
    pdf_path: Path,
    producer: ProducerIdentity,
    current_fingerprint: Callable[[], Mapping[str, Any]],
    render_pdf: Callable[[Path], Path],
    write_csv: Callable[
        [Path, ProducerIdentity, CompanionArtifactRecord],
        Path,
    ],
) -> tuple[Path, Path]:
    """Implement pair publication while :func:`_artifact_pair_lock` is held."""
    assert_producer_identity_current(producer, current_fingerprint())
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    csv_stage: Path | None = None
    pdf_stage: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=pdf_path.parent,
            prefix=f".{pdf_path.name}.",
            suffix=".pdf",
            delete=False,
        ) as temporary:
            pdf_stage = Path(temporary.name)
        with NamedTemporaryFile(
            mode="wb",
            dir=csv_path.parent,
            prefix=f".{csv_path.name}.",
            suffix=".csv",
            delete=False,
        ) as temporary:
            csv_stage = Path(temporary.name)

        rendered_path = render_pdf(pdf_stage)
        if rendered_path.resolve() != pdf_stage.resolve():
            raise ArtifactValidationError(
                "Fischer 2024 PDF renderer did not write the requested staged path."
            )
        pdf_record = companion_artifact_record(pdf_stage)
        written_path = write_csv(csv_stage, producer, pdf_record)
        if written_path.resolve() != csv_stage.resolve():
            raise ArtifactValidationError(
                "Fischer 2024 CSV writer did not write the requested staged path."
            )
        _require_staged_csv_pdf_binding(csv_stage, pdf_record)
        assert_producer_identity_current(producer, current_fingerprint())

        os.replace(pdf_stage, pdf_path)
        pdf_stage = None
        os.replace(csv_stage, csv_path)
        csv_stage = None
        return csv_path, pdf_path
    finally:
        if csv_stage is not None:
            csv_stage.unlink(missing_ok=True)
        if pdf_stage is not None:
            pdf_stage.unlink(missing_ok=True)


def validated_numeric_array(
    values: Any,
    *,
    context: str,
    expected_shape: tuple[int, ...],
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """Return a finite, shape-checked array inside an inclusive domain."""
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ArtifactValidationError(f"{context} must be real-valued.")
    try:
        array = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ArtifactValidationError(f"{context} must be numeric.") from exc
    if array.shape != expected_shape:
        raise ArtifactValidationError(
            f"{context} has shape {array.shape}; expected {expected_shape}."
        )
    if np.any(~np.isfinite(array)):
        raise ArtifactValidationError(f"{context} must contain only finite values.")
    if lower is not None and np.any(array < lower):
        raise ArtifactValidationError(f"{context} contains a value below {lower}.")
    if upper is not None and np.any(array > upper):
        raise ArtifactValidationError(f"{context} contains a value above {upper}.")
    return array


def bind_certificate(
    stamped: QPCertificate,
    reassembled: QPCertificate,
    *,
    context: str,
    residual_inf_limit: float,
) -> QPCertificate:
    """Validate both certificate claims and return the fresh reassembly.

    The persisted stamp records what the producer measured, while
    ``reassembled`` is the reader's current-runtime measurement of the same
    stored state.  Normwise diagnostics at the cancellation floor depend on
    reduction order (including BLAS thread count), so their last bits are not
    portable observables and must not be pinned to each other.  Each claim is
    instead required independently to satisfy the advertised acceptance
    bounds; the reader then consumes the freshly reassembled certificate.
    """
    validate_certificate(
        stamped,
        context=f"{context} stamped certificate",
        residual_inf_limit=residual_inf_limit,
    )
    validate_certificate(
        reassembled,
        context=f"{context} reassembled certificate",
        residual_inf_limit=residual_inf_limit,
    )
    return reassembled


def _normwise_backward_error(
    residual: np.ndarray,
    gain: np.ndarray,
    loss_term: np.ndarray,
) -> float:
    """Stable L1 error of ``gain - loss_term = 0``."""
    common = float(
        max(
            np.max(np.abs(residual), initial=0.0),
            np.max(np.abs(gain), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
        )
    )
    if common == 0.0:
        return 0.0
    numerator = float(np.sum(np.abs(residual) / common))
    denominator = float(np.sum(np.abs(gain) / common) + np.sum(np.abs(loss_term) / common))
    return numerator / denominator if denominator > 0.0 else float("inf")


def _weighted_pair_number_error(
    pair_gain: np.ndarray,
    pair_loss_rate: np.ndarray,
    f: np.ndarray,
    weights: np.ndarray,
    active: np.ndarray,
) -> float:
    """Certify the number-changing mode without scattering turnover."""
    gain = pair_gain[active]
    loss_term = pair_loss_rate[active] * f[active]
    weight = weights[active]
    common = float(
        max(
            np.max(np.abs(gain), initial=0.0),
            np.max(np.abs(loss_term), initial=0.0),
        )
    )
    if common == 0.0:
        return float("inf")
    gain_scaled = weight * (gain / common)
    loss_scaled = weight * (loss_term / common)
    denominator = float(
        np.sum(np.abs(gain_scaled)) + np.sum(np.abs(loss_scaled))
    )
    if denominator == 0.0:
        return float("inf")
    return abs(float(np.sum(gain_scaled - loss_scaled))) / denominator


def qp_certificate(
    state: T3DiffusionState,
    *,
    pb_photon_params: Mapping[str, float],
    residual_inf_limit: float = TARGET_QP_RESIDUAL_INF_LIMIT,
) -> QPCertificate:
    """Reassemble and certify the QP equation solved by the F24 scripts.

    The supported equation is intentionally exact and narrow: ideal BCS,
    homogeneous fixed gap, equilibrium phonons at ``state.T_bath``, and one
    pair-breaking photon channel.  It does not trust the solver's convergence
    flag or reuse solver-internal residual values.
    """
    expected_keys = {"omega_PB", "n_bar_PB", "c_phot_PB"}
    if set(pb_photon_params) != expected_keys:
        raise ValueError(
            "Fischer 2024 QP certificate requires exactly the PB photon keys "
            f"{sorted(expected_keys)}; got {sorted(pb_photon_params)}."
        )
    raw_values = np.asarray([
        pb_photon_params["omega_PB"],
        pb_photon_params["n_bar_PB"],
        pb_photon_params["c_phot_PB"],
    ])
    if np.iscomplexobj(raw_values):
        raise ValueError("PB photon certificate parameters must be real-valued.")
    try:
        values = np.asarray(raw_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("PB photon certificate parameters must be numeric.") from exc
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PB photon certificate parameters must be finite and positive.")

    spectral = state.spectral
    material = state.material
    if spectral.dynes_gamma != 0.0:
        raise ValueError("Fischer 2024 QP certificate supports ideal BCS only.")
    if np.iscomplexobj(state.f):
        raise ValueError("state.f must be real-valued.")
    if state.f.shape != spectral.E.shape:
        raise ValueError("state.f and the spectral energy grid must have equal shape.")
    if np.any(~np.isfinite(state.f)) or np.any((state.f < 0.0) | (state.f > 1.0)):
        raise ValueError("state.f must be finite and lie in [0, 1].")
    if state.gap != spectral.gap:
        raise ValueError("state.gap must exactly match state.spectral.gap.")

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
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(spectral.E)
    n_ph = thermal_phonon_occupation(omega, state.T_bath)
    N_p, N_emit, N_abs = phonon_occupation_matrices_from_state(
        n_ph,
        idx_diff,
        idx_sum,
        diff_sign,
    )
    gain, loss = phonon_collision_rates(
        state.f,
        spectral,
        K_s0,
        K_r0,
        state.T_bath,
        N_p_override=N_p,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    gain_pair_phonon, loss_pair_phonon = phonon_collision_rates(
        state.f,
        spectral,
        None,
        K_r0,
        state.T_bath,
        N_emit_override=N_emit,
        N_abs_override=N_abs,
    )
    gain_pb, loss_pb = pair_breaking_photon_collision_rates(
        state.f,
        spectral,
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )
    (
        _gain_pb_scattering,
        _loss_pb_scattering,
        gain_pair_pb,
        loss_pair_pb,
    ) = pair_breaking_photon_collision_components(
        state.f,
        spectral,
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )
    gain = gain + gain_pb
    loss = loss + loss_pb
    active = spectral.active_mask
    gain_active = gain[active]
    loss_term = loss[active] * state.f[active]
    residual = gain_active - loss_term
    if np.any(~np.isfinite(gain_active)) or np.any(~np.isfinite(loss_term)):
        raise ValueError("Reassembled QP gain/loss terms must be finite.")
    certificate = QPCertificate(
        backward_error=_normwise_backward_error(
            residual,
            gain_active,
            loss_term,
        ),
        residual_inf=float(np.max(np.abs(residual), initial=0.0)),
        qp_number_backward_error=_weighted_pair_number_error(
            gain_pair_phonon + gain_pair_pb,
            loss_pair_phonon + loss_pair_pb,
            state.f,
            spectral.cell_weights,
            active,
        ),
    )
    validate_certificate(
        certificate,
        context="returned state",
        residual_inf_limit=residual_inf_limit,
    )
    return certificate


def validate_certificate(
    certificate: QPCertificate,
    *,
    context: str,
    residual_inf_limit: float = TARGET_QP_RESIDUAL_INF_LIMIT,
) -> None:
    """Reject non-finite, negative, or over-target certificate values."""
    backward = float(certificate.backward_error)
    residual = float(certificate.residual_inf)
    number_backward = float(certificate.qp_number_backward_error)
    residual_limit = float(residual_inf_limit)
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("residual_inf_limit must be finite and positive.")
    if not np.isfinite(backward) or backward < 0.0:
        raise RuntimeError(f"{context}: invalid QP backward error {backward}.")
    if not np.isfinite(residual) or residual < 0.0:
        raise RuntimeError(f"{context}: invalid QP residual_inf {residual}.")
    if not np.isfinite(number_backward) or number_backward < 0.0:
        raise RuntimeError(
            f"{context}: invalid QP number backward error {number_backward}."
        )
    if backward > TARGET_QP_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"{context}: QP backward error {backward:.3e} exceeds target "
            f"{TARGET_QP_BACKWARD_ERROR_LIMIT:.3e}."
        )
    if residual >= residual_limit:
        raise RuntimeError(
            f"{context}: QP residual_inf {residual:.3e} exceeds target {residual_limit:.3e}."
        )
    if number_backward > TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"{context}: QP number backward error {number_backward:.3e} "
            f"exceeds target {TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT:.3e}."
        )


@contextmanager
def _atomic_text_file(path: Path) -> Iterator[TextIO]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            yield cast(TextIO, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _certified_payload_sha256(
    data: np.ndarray,
    certificates: Mapping[str, QPCertificate],
) -> str:
    """Hash the exact persisted table together with its ordered certificates."""
    digest = hashlib.sha256()
    digest.update(f"shape={data.shape}\n".encode())
    for row in data:
        digest.update(",".join(f"{float(value):.17e}" for value in row).encode())
        digest.update(b"\n")
    for point_id, certificate in certificates.items():
        digest.update(point_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(f"{float(certificate.backward_error):.17e}".encode())
        digest.update(b"\0")
        digest.update(f"{float(certificate.residual_inf):.17e}".encode())
        digest.update(b"\0")
        digest.update(
            f"{float(certificate.qp_number_backward_error):.17e}".encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def write_artifact(
    path: Path,
    *,
    schema: str,
    fingerprint: Mapping[str, Any],
    columns: Sequence[str],
    rows: Sequence[Sequence[float]],
    certificates: Mapping[str, QPCertificate],
    target_qp_residual_inf: float = TARGET_QP_RESIDUAL_INF_LIMIT,
    producer: ProducerIdentity | None = None,
    companion_pdf: CompanionArtifactRecord | None = None,
) -> Path:
    """Validate and atomically write one strict Fischer 2024 artifact."""
    if producer is None:
        producer = capture_producer_identity(fingerprint)
    if producer.fingerprint != _json_copy(dict(fingerprint)):
        raise ArtifactValidationError(
            "Artifact fingerprint does not match the pre-solve producer identity."
        )
    if (
        companion_pdf is not None
        and (
            len(companion_pdf.sha256) != 64
            or any(c not in "0123456789abcdef" for c in companion_pdf.sha256)
            or companion_pdf.size_bytes <= 0
        )
    ):
        raise ValueError("Companion PDF record must contain a SHA-256 and positive size.")
    column_list = list(columns)
    if not schema or not column_list or len(set(column_list)) != len(column_list):
        raise ValueError("Artifact schema and unique columns are required.")
    raw_data = np.asarray(rows)
    if np.iscomplexobj(raw_data):
        raise ValueError("Artifact table values must be real-valued.")
    try:
        data = np.asarray(raw_data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Artifact table values must be numeric.") from exc
    expected_shape = (len(rows), len(column_list))
    if data.shape != expected_shape:
        raise ValueError(f"Artifact table shape {data.shape} does not match {expected_shape}.")
    if np.any(~np.isfinite(data)):
        raise ValueError("Artifact table values must all be finite.")
    if not certificates:
        raise ValueError("Artifact must contain at least one QP certificate.")
    certificate_rows: list[dict[str, float | str]] = []
    for point_id, certificate in certificates.items():
        if not point_id:
            raise ValueError("Certificate point identifiers must be non-empty.")
        validate_certificate(
            certificate,
            context=f"certificate {point_id}",
            residual_inf_limit=target_qp_residual_inf,
        )
        certificate_rows.append(
            {
                "point_id": point_id,
                "qp_backward_error": float(certificate.backward_error),
                "qp_number_backward_error": float(
                    certificate.qp_number_backward_error
                ),
                "qp_residual_inf": float(certificate.residual_inf),
            }
        )
    backward_max = max(float(c.backward_error) for c in certificates.values())
    residual_max = max(float(c.residual_inf) for c in certificates.values())
    number_backward_max = max(
        float(c.qp_number_backward_error) for c in certificates.values()
    )
    metadata = {
        "certificate_max_qp_backward_error": backward_max,
        "certificate_max_qp_number_backward_error": number_backward_max,
        "certificate_max_qp_residual_inf": residual_max,
        "certificate_metric_version": QP_CERTIFICATE_METRIC_VERSION,
        "certificate_points": certificate_rows,
        "certificate_target_qp_backward_error": TARGET_QP_BACKWARD_ERROR_LIMIT,
        "certificate_target_qp_number_backward_error": (
            TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT
        ),
        "certificate_target_qp_residual_inf": target_qp_residual_inf,
        "certified_payload_sha256": _certified_payload_sha256(data, certificates),
        "columns": column_list,
        "companion_pdf": (
            {
                "sha256": companion_pdf.sha256,
                "size_bytes": companion_pdf.size_bytes,
            }
            if companion_pdf is not None
            else None
        ),
        "fingerprint": dict(fingerprint),
        "producer_runtime": producer.runtime,
        "row_count": len(rows),
        "schema": schema,
    }

    with _atomic_text_file(path) as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow([f"# qpsim_artifact_schema={schema}"])
        writer.writerow([f"# qpsim_metadata={_canonical_json(metadata)}"])
        writer.writerow(column_list)
        for row in data:
            writer.writerow([f"{value:.17e}" for value in row])
    return path


_METADATA_KEYS = {
    "certificate_max_qp_backward_error",
    "certificate_max_qp_number_backward_error",
    "certificate_max_qp_residual_inf",
    "certificate_metric_version",
    "certificate_points",
    "certificate_target_qp_backward_error",
    "certificate_target_qp_number_backward_error",
    "certificate_target_qp_residual_inf",
    "certified_payload_sha256",
    "columns",
    "companion_pdf",
    "fingerprint",
    "producer_runtime",
    "row_count",
    "schema",
}


def read_artifact(
    path: Path,
    *,
    schema: str,
    fingerprint: Mapping[str, Any],
    columns: Sequence[str],
    expected_row_count: int,
    expected_certificate_ids: Sequence[str],
    target_qp_residual_inf: float = TARGET_QP_RESIDUAL_INF_LIMIT,
    companion_pdf_path: Path | None = None,
    require_companion_pdf: bool = False,
) -> ArtifactTable:
    """Read one pair snapshot while holding the publisher's OS lock."""
    with _artifact_pair_lock(path, operation="read"):
        return _read_artifact_locked(
            path,
            schema=schema,
            fingerprint=fingerprint,
            columns=columns,
            expected_row_count=expected_row_count,
            expected_certificate_ids=expected_certificate_ids,
            target_qp_residual_inf=target_qp_residual_inf,
            companion_pdf_path=companion_pdf_path,
            require_companion_pdf=require_companion_pdf,
        )


def _read_artifact_locked(
    path: Path,
    *,
    schema: str,
    fingerprint: Mapping[str, Any],
    columns: Sequence[str],
    expected_row_count: int,
    expected_certificate_ids: Sequence[str],
    target_qp_residual_inf: float = TARGET_QP_RESIDUAL_INF_LIMIT,
    companion_pdf_path: Path | None = None,
    require_companion_pdf: bool = False,
) -> ArtifactTable:
    """Decode a current artifact while :func:`_artifact_pair_lock` is held."""
    try:
        with path.open(encoding="utf-8", newline="") as fp:
            csv_rows = list(csv.reader(fp))
    except (OSError, UnicodeError) as exc:
        raise ArtifactValidationError(f"Cannot read artifact at {path}: {exc}") from exc

    schema_row = [f"# qpsim_artifact_schema={schema}"]
    if not csv_rows or csv_rows[0] != schema_row:
        has_current_marker = any(
            row
            and (
                row[0].startswith("# qpsim_artifact_schema=")
                or row[0].startswith("# qpsim_metadata=")
            )
            for row in csv_rows
        )
        error_type = ArtifactValidationError if has_current_marker else LegacyArtifactError
        raise error_type(
            f"Artifact at {path} is legacy or has the wrong schema; expected {schema_row[0]!r}."
        )
    if len(csv_rows) != expected_row_count + 3:
        raise ArtifactValidationError(
            f"Artifact at {path} has {len(csv_rows)} CSV rows; expected exactly "
            f"{expected_row_count + 3}."
        )
    if len(csv_rows[1]) != 1 or not csv_rows[1][0].startswith("# qpsim_metadata="):
        raise ArtifactValidationError(
            f"Artifact at {path} is missing the unique qpsim metadata record."
        )
    try:
        metadata = json.loads(csv_rows[1][0].split("=", 1)[1])
    except (json.JSONDecodeError, TypeError) as exc:
        raise ArtifactValidationError(f"Artifact at {path} has malformed JSON metadata.") from exc
    if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
        actual = set(metadata) if isinstance(metadata, dict) else set()
        raise ArtifactValidationError(
            f"Artifact at {path} metadata fields differ from the current schema; "
            f"missing={sorted(_METADATA_KEYS - actual)}, "
            f"extra={sorted(actual - _METADATA_KEYS)}."
        )

    expected_columns = list(columns)
    checks = {
        "schema": schema,
        "fingerprint": dict(fingerprint),
        "columns": expected_columns,
        "row_count": expected_row_count,
        "certificate_metric_version": QP_CERTIFICATE_METRIC_VERSION,
        "certificate_target_qp_backward_error": TARGET_QP_BACKWARD_ERROR_LIMIT,
        "certificate_target_qp_number_backward_error": (
            TARGET_QP_NUMBER_BACKWARD_ERROR_LIMIT
        ),
        "certificate_target_qp_residual_inf": target_qp_residual_inf,
    }
    for field, expected in checks.items():
        if metadata[field] != expected:
            raise ArtifactValidationError(
                f"Artifact at {path} has stale {field}: {metadata[field]!r}; expected {expected!r}."
            )
    producer_runtime = metadata["producer_runtime"]
    if (
        not isinstance(producer_runtime, dict)
        or set(producer_runtime)
        != {
            "matplotlib",
            "numpy",
            "platform",
            "python",
            "runtime_cpu_features",
            "scipy",
            "thread_environment",
        }
    ):
        raise ArtifactValidationError(
            f"Artifact at {path} has malformed producer runtime provenance."
        )
    companion = metadata["companion_pdf"]
    if companion is None:
        if require_companion_pdf:
            raise ArtifactValidationError(
                f"Artifact at {path} is missing its required companion PDF binding."
            )
    elif (
        not isinstance(companion, dict)
        or set(companion) != {"sha256", "size_bytes"}
        or not isinstance(companion["sha256"], str)
        or len(companion["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in companion["sha256"])
        or not isinstance(companion["size_bytes"], int)
        or companion["size_bytes"] <= 0
    ):
        raise ArtifactValidationError(
            f"Artifact at {path} has malformed companion PDF provenance."
        )
    elif companion_pdf_path is not None:
        actual_companion = companion_artifact_record(companion_pdf_path)
        if (
            actual_companion.sha256 != companion["sha256"]
            or actual_companion.size_bytes != companion["size_bytes"]
        ):
            raise ArtifactValidationError(
                f"Artifact at {path} does not authenticate companion PDF "
                f"{companion_pdf_path}."
            )
    if csv_rows[2] != expected_columns:
        raise ArtifactValidationError(
            f"Artifact at {path} has wrong or duplicate columns: {csv_rows[2]!r}."
        )

    point_rows = metadata["certificate_points"]
    if not isinstance(point_rows, list):
        raise ArtifactValidationError(f"Artifact at {path} certificate_points must be a list.")
    point_ids: list[str] = []
    certificates: dict[str, QPCertificate] = {}
    for point in point_rows:
        if not isinstance(point, dict) or set(point) != {
            "point_id",
            "qp_backward_error",
            "qp_number_backward_error",
            "qp_residual_inf",
        }:
            raise ArtifactValidationError(f"Artifact at {path} has a malformed certificate point.")
        point_id = point["point_id"]
        if not isinstance(point_id, str):
            raise ArtifactValidationError(
                f"Artifact at {path} has a non-string certificate point id."
            )
        try:
            certificate = QPCertificate(
                backward_error=float(point["qp_backward_error"]),
                residual_inf=float(point["qp_residual_inf"]),
                qp_number_backward_error=float(
                    point["qp_number_backward_error"]
                ),
            )
        except (TypeError, ValueError) as exc:
            raise ArtifactValidationError(
                f"Artifact at {path} has non-numeric certificate values."
            ) from exc
        try:
            validate_certificate(
                certificate,
                context=f"artifact point {point_id}",
                residual_inf_limit=target_qp_residual_inf,
            )
        except RuntimeError as exc:
            raise ArtifactValidationError(f"Artifact at {path}: {exc}") from exc
        point_ids.append(point_id)
        if point_id in certificates:
            raise ArtifactValidationError(
                f"Artifact at {path} has duplicate certificate point {point_id!r}."
            )
        certificates[point_id] = certificate
    if point_ids != list(expected_certificate_ids):
        raise ArtifactValidationError(
            f"Artifact at {path} has missing, duplicate, or reordered certificate "
            f"points: {point_ids!r}; expected {list(expected_certificate_ids)!r}."
        )

    backward_max = max(c.backward_error for c in certificates.values())
    residual_max = max(c.residual_inf for c in certificates.values())
    number_backward_max = max(
        c.qp_number_backward_error for c in certificates.values()
    )
    maxima = {
        "certificate_max_qp_backward_error": backward_max,
        "certificate_max_qp_number_backward_error": number_backward_max,
        "certificate_max_qp_residual_inf": residual_max,
    }
    for field, expected in maxima.items():
        actual = metadata[field]
        if not isinstance(actual, (int, float)) or not np.isfinite(actual):
            raise ArtifactValidationError(f"Artifact at {path} has non-finite {field}.")
        if float(actual) != expected:
            raise ArtifactValidationError(
                f"Artifact at {path} has inconsistent {field}: {actual!r}; "
                f"certificate points imply {expected!r}."
            )

    numeric_rows = csv_rows[3:]
    if any(len(row) != len(expected_columns) for row in numeric_rows):
        raise ArtifactValidationError(
            f"Artifact at {path} has a data row with the wrong column count."
        )
    try:
        data = np.asarray(
            [[float(value) for value in row] for row in numeric_rows],
            dtype=float,
        )
    except ValueError as exc:
        raise ArtifactValidationError(
            f"Artifact at {path} contains a non-numeric table value."
        ) from exc
    if data.shape != (expected_row_count, len(expected_columns)):
        raise ArtifactValidationError(
            f"Artifact at {path} has table shape {data.shape}; expected "
            f"{(expected_row_count, len(expected_columns))}."
        )
    if np.any(~np.isfinite(data)):
        raise ArtifactValidationError(f"Artifact at {path} contains a non-finite table value.")
    expected_payload_hash = _certified_payload_sha256(data, certificates)
    if metadata["certified_payload_sha256"] != expected_payload_hash:
        raise ArtifactValidationError(
            f"Artifact at {path} certified payload hash does not match its table/certificates."
        )
    return ArtifactTable(data=data, certificates=certificates)
