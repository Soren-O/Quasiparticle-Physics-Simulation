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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, TextIO, cast

import numpy as np
from qpsim.backends.t3_diffusion import T3DiffusionState
from qpsim.collisions.pair_breaking_photon import (
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

from validation.source_provenance import source_sha256

QP_CERTIFICATE_METRIC_VERSION = "qp-gain-loss-l1-maxabs-thermal-pb-v1"
TARGET_QP_BACKWARD_ERROR_LIMIT = 1.0e-6
TARGET_QP_RESIDUAL_INF_LIMIT = 1.0e-10


def source_hashes(
    validation_module: Path,
    *,
    extra_validation_modules: Sequence[Path] = (),
) -> dict[str, str]:
    """Hash the validation modules and numerical sources defining a solve.

    Keep this list explicit: it is provenance for the persisted numerical
    contract, not a package-version surrogate.  In particular it includes the
    low-level quadrature, validation, constants, material, and state modules
    that the public collision/backend modules import. Source newlines are
    normalized so LF and CRLF checkouts have the same identity.
    """
    root = Path(__file__).resolve().parents[2]
    sources = {
        "qpsim/backends/base.py": root / "qpsim/backends/base.py",
        "qpsim/backends/t3_diffusion.py": root / "qpsim/backends/t3_diffusion.py",
        "qpsim/collisions/_uniform_grid.py": root / "qpsim/collisions/_uniform_grid.py",
        "qpsim/collisions/_validation.py": root / "qpsim/collisions/_validation.py",
        "qpsim/collisions/pair_breaking_photon.py": (
            root / "qpsim/collisions/pair_breaking_photon.py"
        ),
        "qpsim/collisions/phonon.py": root / "qpsim/collisions/phonon.py",
        "qpsim/constants.py": root / "qpsim/constants.py",
        "qpsim/grid/energy_grid.py": root / "qpsim/grid/energy_grid.py",
        "qpsim/materials/database.py": root / "qpsim/materials/database.py",
        "qpsim/observables/density.py": root / "qpsim/observables/density.py",
        "qpsim/phonon_models/state.py": root / "qpsim/phonon_models/state.py",
        "qpsim/physics/bcs_quadrature.py": root / "qpsim/physics/bcs_quadrature.py",
        "qpsim/physics/kernels.py": root / "qpsim/physics/kernels.py",
        "qpsim/physics/spectral.py": root / "qpsim/physics/spectral.py",
        "qpsim/services/steady_state.py": root / "qpsim/services/steady_state.py",
        "qpsim/solvers/newton_steady_state.py": (root / "qpsim/solvers/newton_steady_state.py"),
        "validation/source_provenance.py": root / "validation/source_provenance.py",
        "validation/fischer_2024/_artifact.py": Path(__file__).resolve(),
    }
    for module in (validation_module, *extra_validation_modules):
        resolved = module.resolve()
        try:
            name = resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Validation fingerprint source {resolved} is outside {root}."
            ) from exc
        sources[name] = resolved
    return {name: source_sha256(path) for name, path in sources.items()}


class ArtifactValidationError(RuntimeError):
    """Raised when a persisted validation artifact is not current and valid."""


class LegacyArtifactError(ArtifactValidationError):
    """Raised only for artifacts that predate the qpsim schema markers."""


@dataclass(frozen=True)
class QPCertificate:
    """Independent certificate for one returned quasiparticle state."""

    backward_error: float
    residual_inf: float


@dataclass(frozen=True)
class ArtifactTable:
    """Strictly decoded artifact payload."""

    data: np.ndarray
    certificates: dict[str, QPCertificate]


def validated_numeric_array(
    values: Any,
    *,
    context: str,
    expected_shape: tuple[int, ...],
    lower: float | None = None,
    upper: float | None = None,
) -> np.ndarray:
    """Return a finite, shape-checked array inside an inclusive domain."""
    array = np.asarray(values, dtype=float)
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
    """Require a result's certificate stamp to match a fresh reassembly."""
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
    pairs = (
        ("QP backward error", stamped.backward_error, reassembled.backward_error),
        ("QP residual_inf", stamped.residual_inf, reassembled.residual_inf),
    )
    for name, actual, expected in pairs:
        if not np.isclose(float(actual), float(expected), rtol=1.0e-12, atol=0.0):
            raise ArtifactValidationError(
                f"{context} {name} stamp {actual:.17e} does not match "
                f"the persisted state's reassembled value {expected:.17e}."
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
    values = np.asarray(
        [
            pb_photon_params["omega_PB"],
            pb_photon_params["n_bar_PB"],
            pb_photon_params["c_phot_PB"],
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("PB photon certificate parameters must be finite and positive.")

    spectral = state.spectral
    material = state.material
    if spectral.dynes_gamma != 0.0:
        raise ValueError("Fischer 2024 QP certificate supports ideal BCS only.")
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
    gain_pb, loss_pb = pair_breaking_photon_collision_rates(
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
    residual_limit = float(residual_inf_limit)
    if not np.isfinite(residual_limit) or residual_limit <= 0.0:
        raise ValueError("residual_inf_limit must be finite and positive.")
    if not np.isfinite(backward) or backward < 0.0:
        raise RuntimeError(f"{context}: invalid QP backward error {backward}.")
    if not np.isfinite(residual) or residual < 0.0:
        raise RuntimeError(f"{context}: invalid QP residual_inf {residual}.")
    if backward > TARGET_QP_BACKWARD_ERROR_LIMIT:
        raise RuntimeError(
            f"{context}: QP backward error {backward:.3e} exceeds target "
            f"{TARGET_QP_BACKWARD_ERROR_LIMIT:.3e}."
        )
    if residual >= residual_limit:
        raise RuntimeError(
            f"{context}: QP residual_inf {residual:.3e} exceeds target {residual_limit:.3e}."
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
) -> Path:
    """Validate and atomically write one strict Fischer 2024 artifact."""
    column_list = list(columns)
    if not schema or not column_list or len(set(column_list)) != len(column_list):
        raise ValueError("Artifact schema and unique columns are required.")
    data = np.asarray(rows, dtype=float)
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
                "qp_residual_inf": float(certificate.residual_inf),
            }
        )
    backward_max = max(float(c.backward_error) for c in certificates.values())
    residual_max = max(float(c.residual_inf) for c in certificates.values())
    metadata = {
        "certificate_max_qp_backward_error": backward_max,
        "certificate_max_qp_residual_inf": residual_max,
        "certificate_metric_version": QP_CERTIFICATE_METRIC_VERSION,
        "certificate_points": certificate_rows,
        "certificate_target_qp_backward_error": TARGET_QP_BACKWARD_ERROR_LIMIT,
        "certificate_target_qp_residual_inf": target_qp_residual_inf,
        "certified_payload_sha256": _certified_payload_sha256(data, certificates),
        "columns": column_list,
        "fingerprint": dict(fingerprint),
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
    "certificate_max_qp_residual_inf",
    "certificate_metric_version",
    "certificate_points",
    "certificate_target_qp_backward_error",
    "certificate_target_qp_residual_inf",
    "certified_payload_sha256",
    "columns",
    "fingerprint",
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
) -> ArtifactTable:
    """Read a current artifact, rejecting every incomplete/stale variant."""
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
        "certificate_target_qp_residual_inf": target_qp_residual_inf,
    }
    for field, expected in checks.items():
        if metadata[field] != expected:
            raise ArtifactValidationError(
                f"Artifact at {path} has stale {field}: {metadata[field]!r}; expected {expected!r}."
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
    maxima = {
        "certificate_max_qp_backward_error": backward_max,
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
