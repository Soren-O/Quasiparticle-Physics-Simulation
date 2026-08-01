"""Produce a qpsim-free, author-equivalent C0 replay bundle for Figure 6.

The producer starts from the exact initial state in the reviewed A1 bundle,
runs the minimal author-equivalent numerical core, and retains enough native
arrays to audit every Newton transition and every final gain/loss channel.
The raw bundle is an external development artifact; a separate strict summary
generator decides whether it is acceptable evidence for ladder stage C0.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

from validation.fischer_2023.fig6_author_frozen_state import (
    load_author_point_bundle,
    validate_bundle_anchor,
)
from validation.reference_models.fischer_2023.fig6_author_c0 import (
    AuthorNumericalConstants,
    AuthorSolveParameters,
    ChannelBalance,
    author_direct_gap,
    author_direct_gap_integral,
    author_fermi_occupation,
    build_author_operator,
    qp_number_diagnostics,
    solve_author_system,
)
from validation.source_provenance import canonical_source_bytes

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "qpsim.fischer2023.fig6-author-c0-bundle.v1"
STAGE_ID = "C0"
COMPARISON_STAGE_ID = "A1"
PAPER_ORACLE_TARGET_T_STAR_OVER_DELTA = 0.340068493151

_SOURCE_PATHS = (
    Path(__file__).resolve(),
    REPOSITORY_ROOT / "validation" / "author_source.py",
    REPOSITORY_ROOT / "validation" / "reproduction_ladder.py",
    REPOSITORY_ROOT / "validation" / "source_provenance.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_adapter.py",
    REPOSITORY_ROOT / "validation" / "fischer_2023" / "fig6_author_frozen_state.py",
    REPOSITORY_ROOT / "validation" / "reference_models" / "fischer_2023" / "fig6_author_c0.py",
)
_SOURCE_BYTES_AT_IMPORT = {
    path.relative_to(REPOSITORY_ROOT).as_posix(): canonical_source_bytes(path)
    for path in _SOURCE_PATHS
}
_SOURCE_SHA256S_AT_IMPORT = {
    relative: hashlib.sha256(content).hexdigest()
    for relative, content in _SOURCE_BYTES_AT_IMPORT.items()
}
SOURCE_BINDING = {
    "hash_kind": "canonical_sha256_import_time_disk_snapshot",
    "qualification": (
        "The C0 producer, qpsim-free numerical core, and A1 provenance-loader "
        "closure are retained as byte snapshots at module import. Live files "
        "must match before and after computation and writing. The provenance "
        "loader contains a dormant C1 helper with a lazy qpsim import, but the "
        "C0 execution path does not import or call qpsim."
    ),
    "scope": (
        "C0 producer + qpsim-free numerical core + A1 authentication and "
        "source-provenance helpers; no qpsim numerical source"
    ),
}

_THREAD_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class C0BundleError(ValueError):
    """The C0 replay input or output is malformed."""


def _assert_source_snapshots() -> None:
    for relative, expected in _SOURCE_BYTES_AT_IMPORT.items():
        path = REPOSITORY_ROOT / relative
        if canonical_source_bytes(path) != expected:
            raise C0BundleError(f"C0 numerical source changed during execution: {relative}.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise C0BundleError(f"{label} must be an object.")
    return value


def _finite(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise C0BundleError(f"{label} must be a finite real scalar.")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise C0BundleError(f"{label} must be a finite real scalar.") from exc
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise C0BundleError(f"{label} must be a finite {qualifier}real scalar.")
    return result


def _positive_integer(value: object, label: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
    ):
        raise C0BundleError(f"{label} must be a positive integer.")
    return int(value)


def parameters_from_a1_metadata(metadata: dict[str, Any]) -> AuthorSolveParameters:
    """Derive every C0 scalar directly from authenticated A1 metadata."""

    material = _mapping(metadata.get("material_parameters"), "material_parameters")
    constants = _mapping(
        metadata.get("author_numerical_constants"),
        "author_numerical_constants",
    )
    inputs = _mapping(metadata.get("input"), "input")
    return AuthorSolveParameters(
        gap_eV=_finite(
            metadata.get("rounded_gap_eV"),
            "rounded_gap_eV",
            positive=True,
        ),
        h_eV=_finite(metadata.get("h_eV"), "h_eV", positive=True),
        temperature_K=_finite(
            inputs.get("temperature_K"),
            "input.temperature_K",
            positive=True,
        ),
        T_c_K=_finite(
            material.get("T_c_K"),
            "material_parameters.T_c_K",
            positive=True,
        ),
        tau_0_s=_finite(
            material.get("tau_0_s"),
            "material_parameters.tau_0_s",
            positive=True,
        ),
        tau_0_pb_s=_finite(
            material.get("tau_0_pb_s"),
            "material_parameters.tau_0_pb_s",
            positive=True,
        ),
        tau_l_s=_finite(
            material.get("tau_l_s"),
            "material_parameters.tau_l_s",
            positive=True,
        ),
        photon_bin=_positive_integer(metadata.get("photon_bin"), "photon_bin"),
        n_bar=_finite(metadata.get("computed_n_bar"), "computed_n_bar", positive=True),
        c_photon_s_inv=_finite(
            metadata.get("c_photon_per_second"),
            "c_photon_per_second",
            positive=True,
        ),
        delta0_eV=_finite(
            material.get("zero_temperature_gap_eV"),
            "material_parameters.zero_temperature_gap_eV",
            positive=True,
        ),
        thermal_gap_eV=_finite(
            metadata.get("gap_thermal_eV"),
            "gap_thermal_eV",
            positive=True,
        ),
        max_newton_steps=_positive_integer(
            inputs.get("max_newton_steps"),
            "input.max_newton_steps",
        ),
        relative_step_threshold=_finite(
            inputs.get("relative_step_threshold"),
            "input.relative_step_threshold",
            positive=True,
        ),
        constants=AuthorNumericalConstants(
            boltzmann_constant_J_per_K=_finite(
                constants.get("boltzmann_constant_J_per_K"),
                "author_numerical_constants.boltzmann_constant_J_per_K",
                positive=True,
            ),
            electron_charge_C=_finite(
                constants.get("electron_charge_C"),
                "author_numerical_constants.electron_charge_C",
                positive=True,
            ),
        ),
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=(3, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _array_descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _balance_arrays(
    arrays: dict[str, np.ndarray],
    *,
    prefix: str,
    balance: ChannelBalance,
) -> None:
    arrays[f"{prefix}__gain_s_inv"] = np.asarray(balance.gain_s_inv)
    arrays[f"{prefix}__loss_s_inv"] = np.asarray(balance.loss_s_inv)


def _residual_certificate(
    *,
    gains: tuple[np.ndarray, ...],
    losses: tuple[np.ndarray, ...],
) -> dict[str, float]:
    gain_total = np.sum(np.stack(gains), axis=0)
    loss_total = np.sum(np.stack(losses), axis=0)
    residual = gain_total - loss_total
    turnover = float(np.sum(np.abs(gain_total)) + np.sum(np.abs(loss_total)))
    residual_l1 = float(np.sum(np.abs(residual)))
    return {
        "residual_l1_s_inv": residual_l1,
        "residual_over_turnover": residual_l1 / max(turnover, np.finfo(float).tiny),
        "turnover_l1_s_inv": turnover,
    }


def _runtime_record() -> dict[str, object]:
    try:
        configuration: dict[str, Any] = dict(np.show_config(mode="dicts"))
    except TypeError:  # pragma: no cover - NumPy <2 compatibility.
        configuration = {}
    build = configuration.get("Build Dependencies", {})
    linear_algebra: dict[str, object] = {}
    if isinstance(build, dict):
        for name in ("blas", "lapack"):
            raw = build.get(name)
            if isinstance(raw, dict):
                linear_algebra[name] = {
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
    return {
        "architecture": platform.architecture()[0],
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "linear_algebra": linear_algebra,
        "thread_environment": {key: os.environ.get(key) for key in _THREAD_ENVIRONMENT_KEYS},
    }


def build_c0_bundle(
    a1_bundle_dir: Path,
    *,
    anchor_path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the qpsim-free C0 core from the authenticated A1 initial state."""

    _assert_source_snapshots()
    bundle = load_author_point_bundle(a1_bundle_dir)
    anchor_binding = validate_bundle_anchor(bundle, anchor_path)
    required = {
        "E_qp",
        "f_final",
        "initial_state",
        "n_phonon_final",
        "omega_phonon",
    }
    missing = required - set(bundle.arrays)
    if missing:
        raise C0BundleError(f"A1 bundle lacks C0 inputs {sorted(missing)!r}.")

    E_left = np.asarray(bundle.arrays["E_qp"])
    initial_state = np.asarray(bundle.arrays["initial_state"])
    a1_final_state = np.concatenate(
        (
            np.asarray(bundle.arrays["f_final"]),
            np.asarray(bundle.arrays["n_phonon_final"]),
        )
    )
    if (
        E_left.ndim != 1
        or E_left.size < 4
        or initial_state.shape != (2 * E_left.size - 1,)
        or a1_final_state.shape != initial_state.shape
        or any(np.any(~np.isfinite(value)) for value in (E_left, initial_state, a1_final_state))
    ):
        raise C0BundleError("A1 C0 input arrays are malformed.")

    parameters = parameters_from_a1_metadata(bundle.metadata)
    operator = build_author_operator(E_left, parameters)
    solve = solve_author_system(operator, initial_state)
    if not solve.converged:
        raise C0BundleError("C0 author-equivalent solve did not meet the author stopping rule.")
    final_state = solve.state_history[-1]
    f_final = final_state[: E_left.size]
    n_final = final_state[E_left.size :]
    thermal_f = author_fermi_occupation(
        E_left,
        parameters.temperature_K,
        parameters.constants,
    )
    driven_integral = author_direct_gap_integral(
        f_final,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    thermal_integral = author_direct_gap_integral(
        thermal_f,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
    )
    driven_gap = author_direct_gap(
        f_final,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
        delta0_eV=parameters.delta0_eV,
    )
    thermal_gap = author_direct_gap(
        thermal_f,
        gap_eV=parameters.gap_eV,
        h_eV=parameters.h_eV,
        delta0_eV=parameters.delta0_eV,
    )
    denominator = parameters.gap_eV - thermal_gap
    if denominator == 0.0:
        raise C0BundleError("C0 normalized observable has a zero denominator.")
    ordinate = (driven_gap - thermal_gap) / denominator

    final_evaluation = solve.final_evaluation
    arrays: dict[str, np.ndarray] = {
        "E_left_eV": E_left,
        "omega_phonon_eV": np.asarray(bundle.arrays["omega_phonon"]),
        "initial_state": initial_state,
        "a1_final_state": a1_final_state,
        "state_history": solve.state_history,
        "newton_deltas": solve.newton_deltas,
        "relative_qp_steps": solve.relative_qp_steps,
        "linear_solve_backward_errors": solve.linear_solve_backward_errors,
        "f_final": f_final,
        "n_phonon_final": n_final,
        "thermal_f": thermal_f,
        "final_residual_s_inv": final_evaluation.residual_s_inv,
    }
    _balance_arrays(arrays, prefix="qp_photon", balance=final_evaluation.qp_photon)
    _balance_arrays(
        arrays,
        prefix="qp_scattering",
        balance=final_evaluation.qp_scattering,
    )
    _balance_arrays(arrays, prefix="qp_pair", balance=final_evaluation.qp_pair)
    _balance_arrays(
        arrays,
        prefix="phonon_scattering",
        balance=final_evaluation.phonon_scattering,
    )
    _balance_arrays(
        arrays,
        prefix="phonon_pair",
        balance=final_evaluation.phonon_pair,
    )
    _balance_arrays(
        arrays,
        prefix="phonon_escape",
        balance=final_evaluation.phonon_escape,
    )

    qp_certificate = _residual_certificate(
        gains=(
            final_evaluation.qp_photon.gain_s_inv,
            final_evaluation.qp_scattering.gain_s_inv,
            final_evaluation.qp_pair.gain_s_inv,
        ),
        losses=(
            final_evaluation.qp_photon.loss_s_inv,
            final_evaluation.qp_scattering.loss_s_inv,
            final_evaluation.qp_pair.loss_s_inv,
        ),
    )
    phonon_certificate = _residual_certificate(
        gains=(
            final_evaluation.phonon_scattering.gain_s_inv,
            final_evaluation.phonon_pair.gain_s_inv,
            final_evaluation.phonon_escape.gain_s_inv,
        ),
        losses=(
            final_evaluation.phonon_scattering.loss_s_inv,
            final_evaluation.phonon_pair.loss_s_inv,
            final_evaluation.phonon_escape.loss_s_inv,
        ),
    )
    state_scale = float(np.linalg.norm(a1_final_state))
    qp_scale = float(np.linalg.norm(a1_final_state[: E_left.size]))
    phonon_scale = float(np.linalg.norm(a1_final_state[E_left.size :]))

    metadata: dict[str, Any] = {
        "array_descriptors": {
            name: _array_descriptor(value) for name, value in sorted(arrays.items())
        },
        "comparison": {
            "a1_final_state_relative_l2": (
                float(np.linalg.norm(final_state - a1_final_state)) / state_scale
                if state_scale > 0.0
                else 0.0
            ),
            "a1_figure6_ordinate": _finite(
                bundle.metadata.get("normalized_numerical_observable"),
                "normalized_numerical_observable",
            ),
            "a1_phonon_final_relative_l2": (
                float(np.linalg.norm(n_final - a1_final_state[E_left.size :])) / phonon_scale
                if phonon_scale > 0.0
                else 0.0
            ),
            "a1_qp_final_relative_l2": (
                float(np.linalg.norm(f_final - a1_final_state[: E_left.size])) / qp_scale
                if qp_scale > 0.0
                else 0.0
            ),
            "figure6_ordinate_signed_difference": (
                ordinate
                - _finite(
                    bundle.metadata.get("normalized_numerical_observable"),
                    "normalized_numerical_observable",
                )
            ),
        },
        "coordinate_contract": {
            "E_left_eV": "author left-edge nodes E_i=Delta+i*h",
            "omega_phonon_eV": "author phonon nodes h through (N-1)h",
            "state_layout": "[f(E_0..E_N-1), n(h..(N-1)h)]",
        },
        "diagnostics": {
            "final_bounds": {
                "f_max": float(np.max(f_final)),
                "f_min": float(np.min(f_final)),
                "n_phonon_max": float(np.max(n_final)),
                "n_phonon_min": float(np.min(n_final)),
            },
            "phonon_cancellation": phonon_certificate,
            "qp_cancellation": qp_certificate,
            "qp_number": qp_number_diagnostics(final_evaluation, operator),
        },
        "input_bundle": {
            "anchor": anchor_binding,
            "a1_manifest_sha256": bundle.manifest_sha256,
        },
        "observable": {
            "driven_gap_eV": driven_gap,
            "driven_integral": driven_integral,
            "figure6_ordinate": ordinate,
            "thermal_gap_eV": thermal_gap,
            "thermal_integral": thermal_integral,
        },
        "parameters": {
            "T_c_K": parameters.T_c_K,
            "boltzmann_constant_J_per_K": (parameters.constants.boltzmann_constant_J_per_K),
            "c_photon_s_inv": parameters.c_photon_s_inv,
            "delta0_eV": parameters.delta0_eV,
            "electron_charge_C": parameters.constants.electron_charge_C,
            "gap_eV": parameters.gap_eV,
            "h_eV": parameters.h_eV,
            "max_newton_steps": parameters.max_newton_steps,
            "n_bar": parameters.n_bar,
            "num_energy_cells": int(E_left.size),
            "photon_bin": parameters.photon_bin,
            "relative_step_threshold": parameters.relative_step_threshold,
            "tau_0_pb_s": parameters.tau_0_pb_s,
            "tau_0_s": parameters.tau_0_s,
            "tau_l_s": parameters.tau_l_s,
            "temperature_K": parameters.temperature_K,
        },
        "point_selection": {
            "actual_author_t_star_over_delta": _finite(
                bundle.metadata.get("actual_t_star_over_delta"),
                "actual_t_star_over_delta",
            ),
            "nearest_author_sweep_index": _positive_integer(
                _mapping(bundle.metadata.get("input"), "input").get("sweep_index"),
                "input.sweep_index",
            ),
            "paper_oracle_target_t_star_over_delta": (PAPER_ORACLE_TARGET_T_STAR_OVER_DELTA),
        },
        "qualification": (
            "Formal C0 evidence is limited to one authenticated Figure 6 point "
            "(T_B=0.20 K, author sweep index 49). It validates the minimal "
            "author-equivalent port from the captured A1 initial state, not "
            "the author's initializer, random-seed distribution, continuation "
            "path, or the full 300-point numerical curve."
        ),
        "runtime": _runtime_record(),
        "schema": SCHEMA,
        "solver": {
            "converged": solve.converged,
            "iterations": int(solve.newton_deltas.shape[0]),
            "last_relative_qp_step": float(solve.relative_qp_steps[-1]),
            "linear_algebra": "np.linalg.inv(Dx), then -inv(Dx) @ residual",
            "state_constraints": "none; no damping, clipping, or line search",
            "stopping_rule": "relative L2 norm of the QP Newton update",
            "termination": solve.termination,
        },
        "source_binding": dict(SOURCE_BINDING),
        "sources": dict(_SOURCE_SHA256S_AT_IMPORT),
        "stage": {
            "comparison_stage_id": COMPARISON_STAGE_ID,
            "evidence_class": "clean_room_author_equivalent_numerics",
            "parent_stage_id": None,
            "stage_id": STAGE_ID,
        },
    }
    _assert_source_snapshots()
    return metadata, arrays


def write_c0_bundle(
    a1_bundle_dir: Path,
    output_dir: Path,
    *,
    anchor_path: Path,
) -> Path:
    """Write one immutable C0 raw bundle into a new directory."""

    _assert_source_snapshots()
    metadata, arrays = build_c0_bundle(a1_bundle_dir, anchor_path=anchor_path)
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, object]] = {}
    for name, value in sorted(arrays.items()):
        content = _npy_bytes(value)
        path = root / f"{name}.npy"
        with path.open("xb") as handle:
            handle.write(content)
        files[path.name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
    manifest = {"files": files, "metadata": metadata, "schema": SCHEMA}
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    manifest_path = root / "manifest.json"
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    _assert_source_snapshots()
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a1-bundle", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    print(
        write_c0_bundle(
            args.a1_bundle,
            args.output_dir,
            anchor_path=args.anchor,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
