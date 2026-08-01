"""Single-point staged nonlinear re-solves for Fischer 2023 Figure 6.

This producer starts every primary stage from the exact initial Newton state
captured by an independently anchored author run.  It retains the author's
grid, constants, truncated phonon support, drive indexing, explicit-inverse
Newton update, stopping rule, and historical endpoint behavior.  The stages
change only:

* the coherence matrix (author left-edge values or qpsim finite-volume values);
* the pair-phonon frequency label (author sum or qpsim center sum, one bin up).
* the density-of-states arithmetic (author eV formula or the same
  ``SpectralContext``'s native-micro-eV ``cell_density``).

The authenticated author Newton matrix is not the exact derivative of its
residual in three places: the residual omits photon transitions involving the
last QP bin while the matrix includes them, the photon off-diagonal entries use
the partner occupation rather than the row occupation, and the phonon-pair
``D_n N`` diagonal uses two historically shifted occupation indices.  All
three source-level inconsistencies are intentionally preserved here.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from qpsim.observables.gap_suppression import (
    gap_from_distribution_direct,
    gap_integral_from_distribution_direct,
)
from qpsim.physics.spectral import SpectralContext

from validation.fischer_2023.fig6_author_frozen_state import (
    FrozenStateDiagnosticError,
    load_author_point_bundle,
    validate_bundle_anchor,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorC0Error,
    AuthorNumericalConstants,
    AuthorOperator,
    AuthorSolveParameters,
    SpectralCoefficients,
    SystemEvaluation,
    author_cell_average_bcs_dos,
    build_author_coefficients,
    build_author_operator,
    evaluate_author_system,
    solve_author_system,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    NewtonSolveResult as CoreNewtonSolveResult,
)
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHOR_OPERATOR_SOURCE = (
    REPOSITORY_ROOT
    / "validation"
    / "reference_models"
    / "fischer_2023"
    / "fig6_author_c0.py"
)
ANCHOR_HELPER_SOURCE = (
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_frozen_state.py"
)
VALIDATION_SOURCE_DEPENDENCIES = (
    REPOSITORY_ROOT / "validation" / "author_source.py",
    REPOSITORY_ROOT / "validation" / "reproduction_ladder.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_adapter.py",
    ANCHOR_HELPER_SOURCE,
    AUTHOR_OPERATOR_SOURCE,
)


def _numerical_source_paths() -> dict[str, Path]:
    """Return the deliberately conservative staged-producer source closure."""

    qpsim_root = REPOSITORY_ROOT / "qpsim"
    paths = {
        Path(__file__).resolve(),
        *VALIDATION_SOURCE_DEPENDENCIES,
        *qpsim_root.rglob("*.py"),
        *qpsim_root.rglob("*.yaml"),
        *qpsim_root.rglob("*.yml"),
    }
    return {
        path.relative_to(REPOSITORY_ROOT).as_posix(): path
        for path in sorted(paths, key=lambda candidate: candidate.as_posix())
    }


_SOURCE_PATHS_AT_IMPORT = _numerical_source_paths()
_SOURCE_BYTES_AT_IMPORT = {
    relative: canonical_source_bytes(path)
    for relative, path in _SOURCE_PATHS_AT_IMPORT.items()
}
_SOURCE_SHA256S_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}
SOURCE_BINDING = {
    "hash_kind": "canonical_sha256_import_time_disk_snapshot",
    "qualification": (
        "The staged producer, explicit validation dependencies, and complete "
        "qpsim source/material closure are retained as byte snapshots at module "
        "import. Live files must match before and after evaluation and writing. "
        "This guards source drift but is not an independent interpreter-bytecode "
        "attestation."
    ),
    "scope": (
        "staged producer + explicit validation dependencies + all qpsim "
        "Python and material YAML sources"
    ),
}
SCHEMA = "qpsim.fischer2023.fig6-author-staged-resolve.v3"

CoherenceConvention = Literal["author_left_edges", "qpsim_finite_volume"]
DensityConvention = Literal["author_cell_average_eV", "qpsim_cell_density_ueV"]


@dataclass(frozen=True)
class StageSpecification:
    """One controlled discretization convention."""

    stage_id: str
    parent_stage_id: str
    coherence_convention: CoherenceConvention
    density_convention: DensityConvention
    pair_frequency_offset_bins: int
    changed_convention: str


AUTHOR_CONTROL = StageSpecification(
    stage_id="author_control",
    parent_stage_id="A1",
    coherence_convention="author_left_edges",
    density_convention="author_cell_average_eV",
    pair_frequency_offset_bins=0,
    changed_convention="none; exact author numerical control",
)
PAIR_LABELS_ONLY = StageSpecification(
    stage_id="pair_labels_only",
    parent_stage_id="author_control",
    coherence_convention="author_left_edges",
    density_convention="author_cell_average_eV",
    pair_frequency_offset_bins=1,
    changed_convention="pair frequency 2*Delta+(i+j)*h -> 2*Delta+(i+j+1)*h",
)
C3A = StageSpecification(
    stage_id="c3a_finite_volume_coherence",
    parent_stage_id="author_control",
    coherence_convention="qpsim_finite_volume",
    density_convention="author_cell_average_eV",
    pair_frequency_offset_bins=0,
    changed_convention="left-edge coherence -> qpsim finite-volume coherence",
)
C3B = StageSpecification(
    stage_id="c3b_center_pair_labels",
    parent_stage_id="c3a_finite_volume_coherence",
    coherence_convention="qpsim_finite_volume",
    density_convention="author_cell_average_eV",
    pair_frequency_offset_bins=1,
    changed_convention="pair frequency 2*Delta+(i+j)*h -> 2*Delta+(i+j+1)*h",
)
C3C = StageSpecification(
    stage_id="c3c_native_cell_density",
    parent_stage_id="c3b_center_pair_labels",
    coherence_convention="qpsim_finite_volume",
    density_convention="qpsim_cell_density_ueV",
    pair_frequency_offset_bins=1,
    changed_convention=(
        "author eV cell-average DOS arithmetic -> the same qpsim "
        "SpectralContext's native-micro-eV cell_density"
    ),
)
PRIMARY_STAGES = (AUTHOR_CONTROL, PAIR_LABELS_ONLY, C3A, C3B, C3C)


@dataclass(frozen=True)
class StagedOperator:
    """Precomputed fixed-grid coefficients for one stage."""

    specification: StageSpecification
    author_operator: AuthorOperator

    @property
    def parameters(self) -> AuthorSolveParameters:
        return self.author_operator.parameters

    @property
    def E_left_eV(self) -> np.ndarray:
        return self.author_operator.E_left_eV

    @property
    def rho(self) -> np.ndarray:
        return self.author_operator.rho

    @property
    def K_plus(self) -> np.ndarray:
        return self.author_operator.K_plus

    @property
    def K_minus(self) -> np.ndarray:
        return self.author_operator.K_minus

    @property
    def n_thermal(self) -> np.ndarray:
        return self.author_operator.n_thermal

    @property
    def a_delta(self) -> int:
        return self.author_operator.a_delta

    @property
    def qp_prefactor_s_inv(self) -> float:
        return self.author_operator.qp_prefactor_s_inv

    @property
    def phonon_prefactor_per_eV_s(self) -> float:
        return self.author_operator.phonon_prefactor_per_eV_s


@dataclass(frozen=True)
class NewtonSolveResult:
    """One complete Newton path."""

    specification: StageSpecification
    solve_path: str
    state_history: np.ndarray
    newton_deltas: np.ndarray
    relative_qp_steps: np.ndarray
    linear_solve_backward_errors: np.ndarray
    converged: bool
    termination: str
    final_evaluation: SystemEvaluation


@dataclass(frozen=True)
class StagedResolveResult:
    """Authenticated staged-solve metadata and native arrays."""

    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]


def _assert_numerical_source_snapshots() -> None:
    """Refuse evidence collection after any retained source has changed."""

    if _numerical_source_paths() != _SOURCE_PATHS_AT_IMPORT:
        raise FrozenStateDiagnosticError(
            "Staged-solve numerical source closure changed after module import; "
            "launch a fresh Python process."
        )
    current = {
        relative: canonical_source_bytes(path)
        for relative, path in _SOURCE_PATHS_AT_IMPORT.items()
    }
    if current != _SOURCE_BYTES_AT_IMPORT:
        raise FrozenStateDiagnosticError(
            "Staged-solve numerical source bytes changed after module import; "
            "launch a fresh Python process."
        )


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise FrozenStateDiagnosticError(f"{label} must be a finite positive scalar.")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise FrozenStateDiagnosticError(f"{label} must be a finite positive scalar.")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise FrozenStateDiagnosticError(f"{label} must be a positive integer.")
    result = int(value)
    if result <= 0:
        raise FrozenStateDiagnosticError(f"{label} must be a positive integer.")
    return result


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrozenStateDiagnosticError(f"{label} must be a mapping.")
    return value


def _native_array_record(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise FrozenStateDiagnosticError("Object arrays are forbidden.")
    stream = io.BytesIO()
    np.lib.format.write_array(stream, array, version=(3, 0), allow_pickle=False)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(stream.getvalue()).hexdigest(),
        "shape": [int(size) for size in array.shape],
    }


def build_staged_operator(
    E_left_eV: np.ndarray,
    parameters: AuthorSolveParameters,
    specification: StageSpecification,
) -> StagedOperator:
    """Inject one staged coefficient convention into the qpsim-free core."""

    E = np.asarray(E_left_eV, dtype=float)
    try:
        author_coefficients = build_author_coefficients(E, parameters)
        author_rho = author_cell_average_bcs_dos(
            E,
            gap_eV=parameters.gap_eV,
            h_eV=parameters.h_eV,
        )
        ctx: SpectralContext | None = None
        if (
            specification.coherence_convention == "qpsim_finite_volume"
            or specification.density_convention == "qpsim_cell_density_ueV"
        ):
            # Construct one context in qpsim's native micro-eV unit. C3c
            # deliberately takes density and coherence from this same object.
            centers = (E + 0.5 * parameters.h_eV) * 1.0e6
            ctx = SpectralContext(
                E_bins=centers,
                dE_bins=np.full_like(centers, parameters.h_eV * 1.0e6),
                gap=parameters.gap_eV * 1.0e6,
            )
            np.testing.assert_allclose(
                ctx.cell_density,
                author_rho,
                rtol=2e-7,
                atol=0.0,
                err_msg="qpsim cell density no longer matches the author DOS cells",
            )
        if specification.density_convention == "author_cell_average_eV":
            rho = author_rho
        elif specification.density_convention == "qpsim_cell_density_ueV":
            if ctx is None:  # pragma: no cover - constructed above.
                raise AssertionError("C3c SpectralContext was not constructed.")
            rho = np.asarray(ctx.cell_density)
        else:  # pragma: no cover - protected by Literal constants.
            raise FrozenStateDiagnosticError("Unknown density convention.")
        if specification.coherence_convention == "author_left_edges":
            K_plus = author_coefficients.K_plus
            K_minus = author_coefficients.K_minus
        elif specification.coherence_convention == "qpsim_finite_volume":
            if ctx is None:  # pragma: no cover - constructed above.
                raise AssertionError("Finite-volume SpectralContext was not constructed.")
            K_plus = np.asarray(ctx.K_plus)
            K_minus = np.asarray(ctx.K_minus)
        else:  # pragma: no cover - protected by Literal constants.
            raise FrozenStateDiagnosticError("Unknown coherence convention.")
        coefficients = SpectralCoefficients(
            rho=rho,
            K_plus=K_plus,
            K_minus=K_minus,
            pair_frequency_offset_bins=specification.pair_frequency_offset_bins,
        )
        author_operator = build_author_operator(
            E,
            parameters,
            coefficients=coefficients,
        )
    except AuthorC0Error as exc:
        raise FrozenStateDiagnosticError(str(exc)) from exc
    return StagedOperator(
        specification=specification,
        author_operator=author_operator,
    )


def evaluate_staged_system(
    operator: StagedOperator,
    f: np.ndarray,
    n_phonon: np.ndarray,
    *,
    build_update_matrix: bool,
) -> SystemEvaluation:
    """Delegate staged evaluation to the qpsim-free numerical core."""

    try:
        return evaluate_author_system(
            operator.author_operator,
            f,
            n_phonon,
            build_update_matrix=build_update_matrix,
        )
    except AuthorC0Error as exc:
        raise FrozenStateDiagnosticError(str(exc)) from exc


def solve_staged_system(
    operator: StagedOperator,
    initial_state: np.ndarray,
    *,
    solve_path: str,
) -> NewtonSolveResult:
    """Run the core author policy and attach the staged identity/path."""

    try:
        core_result: CoreNewtonSolveResult = solve_author_system(
            operator.author_operator,
            initial_state,
        )
    except AuthorC0Error as exc:
        raise FrozenStateDiagnosticError(
            f"{operator.specification.stage_id}: {exc}"
        ) from exc
    return NewtonSolveResult(
        specification=operator.specification,
        solve_path=solve_path,
        state_history=core_result.state_history,
        newton_deltas=core_result.newton_deltas,
        relative_qp_steps=core_result.relative_qp_steps,
        linear_solve_backward_errors=core_result.linear_solve_backward_errors,
        converged=core_result.converged,
        termination=core_result.termination,
        final_evaluation=core_result.final_evaluation,
    )


def _cancellation_certificate(
    evaluation: SystemEvaluation,
    size: int,
) -> dict[str, dict[str, float]]:
    qp_residual = evaluation.residual_s_inv[:size]
    phonon_residual = evaluation.residual_s_inv[size:]
    qp_turnover = float(
        np.sum(np.abs(evaluation.qp_photon_s_inv))
        + np.sum(np.abs(evaluation.qp_scattering_s_inv))
        + np.sum(np.abs(evaluation.qp_pair_s_inv))
    )
    phonon_turnover = float(
        np.sum(np.abs(evaluation.phonon_scattering_s_inv))
        + np.sum(np.abs(evaluation.phonon_pair_s_inv))
        + np.sum(np.abs(evaluation.phonon_escape_s_inv))
    )

    def record(residual: np.ndarray, turnover: float) -> dict[str, float]:
        residual_l1 = float(np.sum(np.abs(residual)))
        return {
            "channel_turnover_l1_s_inv": turnover,
            "residual_l1_over_channel_turnover": (
                residual_l1 / turnover if turnover > 0.0 else 0.0
            ),
            "residual_l1_s_inv": residual_l1,
            "residual_l2_s_inv": float(np.linalg.norm(residual)),
            "residual_linf_s_inv": float(np.max(np.abs(residual), initial=0.0)),
        }

    return {
        "phonon": record(phonon_residual, phonon_turnover),
        "qp": record(qp_residual, qp_turnover),
    }


def _observable_record(
    result: NewtonSolveResult,
    operator: StagedOperator,
    *,
    author_observable: float,
    author_gap_eV: float,
) -> dict[str, float]:
    size = operator.E_left_eV.size
    f = result.state_history[-1, :size]
    params = operator.parameters
    centers = operator.E_left_eV + 0.5 * params.h_eV
    integral = gap_integral_from_distribution_direct(
        f,
        centers,
        gap=params.gap_eV,
        samples="authors",
    )
    gap = gap_from_distribution_direct(
        f,
        centers,
        gap=params.gap_eV,
        delta0=params.delta0_eV,
        samples="authors",
    )
    denominator = params.gap_eV - params.thermal_gap_eV
    if denominator == 0.0:
        raise FrozenStateDiagnosticError("The author thermal observable denominator is zero.")
    ordinate = (gap - params.thermal_gap_eV) / denominator
    return {
        "author_control_gap_reference_eV": author_gap_eV,
        "author_control_ordinate_reference": author_observable,
        "direct_gap_eV": gap,
        "direct_integral": integral,
        "figure6_ordinate": ordinate,
        "gap_minus_author_control_eV": gap - author_gap_eV,
        "ordinate_minus_author_control": ordinate - author_observable,
        "thermal_gap_eV_held_fixed": params.thermal_gap_eV,
    }


def _stage_metadata(
    result: NewtonSolveResult,
    operator: StagedOperator,
    *,
    author_observable: float,
    author_gap_eV: float,
    initial_state_record: dict[str, object],
) -> dict[str, Any]:
    size = operator.E_left_eV.size
    final_state = result.state_history[-1]
    return {
        "cancellation_certificate": _cancellation_certificate(
            result.final_evaluation,
            size,
        ),
        "converged": result.converged,
        "final_bounds": {
            "f_max": float(np.max(final_state[:size])),
            "f_min": float(np.min(final_state[:size])),
            "n_phonon_max": float(np.max(final_state[size:])),
            "n_phonon_min": float(np.min(final_state[size:])),
        },
        "initial_state": initial_state_record,
        "last_relative_qp_step": float(result.relative_qp_steps[-1]),
        "max_linear_solve_backward_error": float(
            np.max(result.linear_solve_backward_errors, initial=0.0)
        ),
        "newton_iterations": int(result.relative_qp_steps.size),
        "observable": _observable_record(
            result,
            operator,
            author_observable=author_observable,
            author_gap_eV=author_gap_eV,
        ),
        "solve_path": result.solve_path,
        "specification": {
            "changed_convention": result.specification.changed_convention,
            "coherence_convention": result.specification.coherence_convention,
            "density_convention": result.specification.density_convention,
            "pair_frequency_offset_bins": (
                result.specification.pair_frequency_offset_bins
            ),
            "parent_stage_id": result.specification.parent_stage_id,
            "stage_id": result.specification.stage_id,
        },
        "termination": result.termination,
    }


def _result_arrays(
    results: dict[str, NewtonSolveResult],
    *,
    initial_state: np.ndarray,
) -> dict[str, np.ndarray]:
    arrays = {"captured_author_initial_state": np.asarray(initial_state)}
    for result_id, result in results.items():
        prefix = result_id
        evaluation = result.final_evaluation
        arrays[f"{prefix}__final_state"] = result.state_history[-1]
        arrays[f"{prefix}__state_history"] = result.state_history
        arrays[f"{prefix}__newton_deltas"] = result.newton_deltas
        arrays[f"{prefix}__relative_qp_steps"] = result.relative_qp_steps
        arrays[f"{prefix}__linear_solve_backward_errors"] = (
            result.linear_solve_backward_errors
        )
        arrays[f"{prefix}__residual_s_inv"] = evaluation.residual_s_inv
        arrays[f"{prefix}__qp_photon_s_inv"] = evaluation.qp_photon_s_inv
        arrays[f"{prefix}__qp_scattering_s_inv"] = evaluation.qp_scattering_s_inv
        arrays[f"{prefix}__qp_pair_s_inv"] = evaluation.qp_pair_s_inv
        arrays[f"{prefix}__phonon_scattering_s_inv"] = (
            evaluation.phonon_scattering_s_inv
        )
        arrays[f"{prefix}__phonon_pair_s_inv"] = evaluation.phonon_pair_s_inv
        arrays[f"{prefix}__phonon_escape_s_inv"] = evaluation.phonon_escape_s_inv
    return arrays


def _extract_parameters(metadata: dict[str, Any]) -> AuthorSolveParameters:
    material = _mapping(metadata.get("material_parameters"), "material_parameters")
    constants = _mapping(
        metadata.get("author_numerical_constants"),
        "author_numerical_constants",
    )
    inputs = _mapping(metadata.get("input"), "input")
    photon_bin = _positive_integer(metadata.get("photon_bin"), "photon_bin")
    max_steps = _positive_integer(
        inputs.get("max_newton_steps"),
        "input.max_newton_steps",
    )
    return AuthorSolveParameters(
        gap_eV=_positive(metadata.get("rounded_gap_eV"), "rounded_gap_eV"),
        h_eV=_positive(metadata.get("h_eV"), "h_eV"),
        temperature_K=_positive(inputs.get("temperature_K"), "input.temperature_K"),
        T_c_K=_positive(material.get("T_c_K"), "material_parameters.T_c_K"),
        tau_0_s=_positive(material.get("tau_0_s"), "material_parameters.tau_0_s"),
        tau_0_pb_s=_positive(
            material.get("tau_0_pb_s"),
            "material_parameters.tau_0_pb_s",
        ),
        tau_l_s=_positive(material.get("tau_l_s"), "material_parameters.tau_l_s"),
        photon_bin=photon_bin,
        n_bar=_positive(metadata.get("computed_n_bar"), "computed_n_bar"),
        c_photon_s_inv=_positive(
            metadata.get("c_photon_per_second"),
            "c_photon_per_second",
        ),
        delta0_eV=_positive(
            material.get("zero_temperature_gap_eV"),
            "material_parameters.zero_temperature_gap_eV",
        ),
        thermal_gap_eV=_positive(metadata.get("gap_thermal_eV"), "gap_thermal_eV"),
        max_newton_steps=max_steps,
        relative_step_threshold=_positive(
            inputs.get("relative_step_threshold"),
            "input.relative_step_threshold",
        ),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=_positive(
                constants.get("boltzmann_constant_J_per_K"),
                "author_numerical_constants.boltzmann_constant_J_per_K",
            ),
            electron_charge_C=_positive(
                constants.get("electron_charge_C"),
                "author_numerical_constants.electron_charge_C",
            ),
        ),
    )


def build_staged_resolve(
    bundle_dir: Path,
    *,
    anchor_path: Path,
    include_c3b_continuation: bool = True,
    include_c3c_continuation: bool = True,
) -> StagedResolveResult:
    """Run all primary stages from the same captured author initial state."""

    _assert_numerical_source_snapshots()
    producer_sources = dict(_SOURCE_SHA256S_AT_IMPORT)
    bundle = load_author_point_bundle(bundle_dir)
    anchor_binding = validate_bundle_anchor(bundle, anchor_path)
    arrays = bundle.arrays
    required = {"E_qp", "f_final", "initial_state", "n_phonon_final"}
    missing = required - set(arrays)
    if missing:
        raise FrozenStateDiagnosticError(
            f"Author bundle lacks staged-solve inputs {sorted(missing)!r}."
        )
    E_left = np.asarray(arrays["E_qp"])
    author_final = np.concatenate(
        (
            np.asarray(arrays["f_final"]),
            np.asarray(arrays["n_phonon_final"]),
        )
    )
    initial_state = np.asarray(arrays["initial_state"])
    if (
        E_left.ndim != 1
        or author_final.shape != (2 * E_left.size - 1,)
        or initial_state.shape != author_final.shape
        or any(
            np.any(~np.isfinite(value))
            for value in (E_left, author_final, initial_state)
        )
    ):
        raise FrozenStateDiagnosticError("Captured author solve arrays are malformed.")

    params = _extract_parameters(bundle.metadata)
    author_observable = _positive(
        bundle.metadata.get("normalized_numerical_observable"),
        "normalized_numerical_observable",
    )
    author_gap = _positive(bundle.metadata.get("gap_driven_eV"), "gap_driven_eV")
    initial_record = _native_array_record(initial_state)

    operators = {
        spec.stage_id: build_staged_operator(E_left, params, spec)
        for spec in PRIMARY_STAGES
    }
    results: dict[str, NewtonSolveResult] = {}
    for spec in PRIMARY_STAGES:
        results[spec.stage_id] = solve_staged_system(
            operators[spec.stage_id],
            initial_state,
            solve_path="independent_from_captured_author_initial_state",
        )
    if include_c3b_continuation:
        results["c3b_continuation_from_c3a"] = solve_staged_system(
            operators[C3B.stage_id],
            results[C3A.stage_id].state_history[-1],
            solve_path="continuation_from_c3a_final_state",
        )
    if include_c3c_continuation:
        results["c3c_continuation_from_c3b"] = solve_staged_system(
            operators[C3C.stage_id],
            results[C3B.stage_id].state_history[-1],
            solve_path="continuation_from_c3b_final_state",
        )

    result_arrays = _result_arrays(results, initial_state=initial_state)
    stage_metadata = {
        result_id: _stage_metadata(
            result,
            operators[result.specification.stage_id],
            author_observable=author_observable,
            author_gap_eV=author_gap,
            initial_state_record=(
                initial_record
                if result.solve_path
                == "independent_from_captured_author_initial_state"
                else _native_array_record(result.state_history[0])
            ),
        )
        for result_id, result in results.items()
    }

    control_final = results[AUTHOR_CONTROL.stage_id].state_history[-1]
    control_scale = float(np.linalg.norm(author_final))
    path_records: dict[str, dict[str, float]] = {}
    if "c3b_continuation_from_c3a" in results:
        independent = results[C3B.stage_id].state_history[-1]
        continued = results["c3b_continuation_from_c3a"].state_history[-1]
        state_scale = float(np.linalg.norm(independent))
        path_records["c3b_continuation_from_c3a"] = {
            "final_state_relative_l2": (
                float(np.linalg.norm(continued - independent)) / state_scale
                if state_scale > 0.0
                else 0.0
            ),
            "figure6_ordinate_signed_difference": (
                stage_metadata["c3b_continuation_from_c3a"]["observable"][
                    "figure6_ordinate"
                ]
                - stage_metadata[C3B.stage_id]["observable"]["figure6_ordinate"]
            ),
        }
    if "c3c_continuation_from_c3b" in results:
        independent = results[C3C.stage_id].state_history[-1]
        continued = results["c3c_continuation_from_c3b"].state_history[-1]
        state_scale = float(np.linalg.norm(independent))
        path_records["c3c_continuation_from_c3b"] = {
            "final_state_relative_l2": (
                float(np.linalg.norm(continued - independent)) / state_scale
                if state_scale > 0.0
                else 0.0
            ),
            "figure6_ordinate_signed_difference": (
                stage_metadata["c3c_continuation_from_c3b"]["observable"][
                    "figure6_ordinate"
                ]
                - stage_metadata[C3C.stage_id]["observable"]["figure6_ordinate"]
            ),
        }

    metadata_result: dict[str, Any] = {
        "array_descriptors": {
            name: _native_array_record(value)
            for name, value in sorted(result_arrays.items())
        },
        "author_control_reproduction": {
            "captured_author_final_state": _native_array_record(author_final),
            "final_state_relative_l2": (
                float(np.linalg.norm(control_final - author_final)) / control_scale
                if control_scale > 0.0
                else 0.0
            ),
            "qp_final_relative_l2": float(
                np.linalg.norm(
                    control_final[: E_left.size] - author_final[: E_left.size]
                )
                / np.linalg.norm(author_final[: E_left.size])
            ),
        },
        "coordinate_contract": {
            "carrier_occupation_samples": "author left edges E_i=Delta+i*h",
            "density_of_states": "exact author cell average over [E_i,E_i+h]",
            "native_density_of_states": (
                "C3c uses the same qpsim SpectralContext's native-micro-eV "
                "cell_density without algebraic re-evaluation in eV"
            ),
            "finite_volume_coherence": (
                "qpsim SpectralContext exact double-cell-average K_plus/K_minus"
            ),
            "pair_frequency_offset_meaning": (
                "offset changes the n lookup, represented phonon aggregation, "
                "and the pair-frequency-squared QP rate factor together"
            ),
            "phonon_support": "h through (N-1)h; higher occupations fixed to zero",
        },
        "diagnostic_id": "fischer-catelani-2023-fig6-c3-staged-resolve-v3",
        "input_bundle": {
            "anchor": anchor_binding,
            "manifest_sha256": bundle.manifest_sha256,
        },
        "limitations": {
            "scope": "one anchored Figure 6 point; no 300-point sweep",
            "seed_sensitivity": (
                "all primary stages use one identical captured author initial "
                "state; this does not survey independent random seeds"
            ),
            "solver": (
                "author explicit matrix inverse and no damping/clipping are "
                "retained, including the authenticated photon endpoint, photon "
                "off-diagonal, and phonon-pair DnN residual/Jacobian mismatches"
            ),
        },
        "parameters": {
            "T_c_K": params.T_c_K,
            "c_photon_s_inv": params.c_photon_s_inv,
            "gap_eV": params.gap_eV,
            "h_eV": params.h_eV,
            "max_newton_steps": params.max_newton_steps,
            "n_bar": params.n_bar,
            "num_energy_cells": int(E_left.size),
            "photon_bin": params.photon_bin,
            "relative_step_threshold": params.relative_step_threshold,
            "tau_0_pb_s": params.tau_0_pb_s,
            "tau_0_s": params.tau_0_s,
            "tau_l_s": params.tau_l_s,
            "temperature_K": params.temperature_K,
        },
        "path_dependence": path_records,
        "schema": SCHEMA,
        "source_binding": dict(SOURCE_BINDING),
        "solver_contract": {
            "linear_algebra": "np.linalg.inv(Dx) followed by matrix-vector product",
            "primary_stage_initialization": "same captured author initial state",
            "state_constraints": "none; no clipping or line search",
            "stopping_rule": "relative L2 norm of quasiparticle Newton step",
        },
        "sources": producer_sources,
        "stages": stage_metadata,
    }
    _assert_numerical_source_snapshots()
    return StagedResolveResult(metadata=metadata_result, arrays=result_arrays)


def write_staged_resolve_bundle(
    bundle_dir: Path,
    output_dir: Path,
    *,
    anchor_path: Path,
    include_c3b_continuation: bool = True,
    include_c3c_continuation: bool = True,
) -> Path:
    """Write staged arrays and a self-describing manifest to a new directory."""

    _assert_numerical_source_snapshots()
    result = build_staged_resolve(
        bundle_dir,
        anchor_path=anchor_path,
        include_c3b_continuation=include_c3b_continuation,
        include_c3c_continuation=include_c3c_continuation,
    )
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, object]] = {}
    for name, value in sorted(result.arrays.items()):
        path = root / f"{name}.npy"
        stream = io.BytesIO()
        np.lib.format.write_array(
            stream,
            np.asarray(value),
            version=(3, 0),
            allow_pickle=False,
        )
        content = stream.getvalue()
        with path.open("xb") as handle:
            handle.write(content)
        files[path.name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {
        "files": files,
        "metadata": result.metadata,
        "schema": SCHEMA,
    }
    manifest_path = root / "manifest.json"
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _assert_numerical_source_snapshots()
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--omit-c3b-continuation",
        action="store_true",
        help="Omit the C3b continuation seeded from the C3a final state.",
    )
    parser.add_argument(
        "--omit-c3c-continuation",
        action="store_true",
        help="Omit the C3c continuation seeded from the C3b final state.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_staged_resolve_bundle(
            args.bundle,
            args.output_dir,
            anchor_path=args.anchor,
            include_c3b_continuation=not args.omit_c3b_continuation,
            include_c3c_continuation=not args.omit_c3c_continuation,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
