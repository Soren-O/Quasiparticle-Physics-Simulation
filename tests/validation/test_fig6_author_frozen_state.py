from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
from qpsim.observables.gap_suppression import gap_from_distribution_direct
from validation.fischer_2023.fig6_author_adapter import (
    ADAPTER_DEPENDENCIES,
    ADAPTER_SCHEMA,
    ADAPTER_SOURCE_EXECUTION_BINDING,
    ADAPTER_SOURCE_HASH_KIND,
    SUBPROCESS_WRAPPER_SHA256,
    AuthorPointResult,
    write_author_point_bundle,
)
from validation.fischer_2023.fig6_author_frozen_state import (
    ANCHOR_SCHEMA,
    AUTHOR_DELTA0_EV,
    AUTHOR_SOURCE_MANIFEST,
    DIAGNOSTIC_SCHEMA,
    FrozenStateDiagnosticError,
    build_frozen_gap_comparison,
    load_author_point_bundle,
    write_frozen_gap_comparison,
)
from validation.source_provenance import source_sha256


def _author_direct_gap(f: np.ndarray, *, h: float, gap: float) -> float:
    """Independent transcription of the attachment's delta_not_consistent."""

    indices = np.arange(f.size, dtype=float)
    const_term = np.sum(
        4.0
        * f
        * (
            np.arcsinh(np.sqrt((indices + 1.0) * h / (2.0 * gap)))
            - np.arcsinh(np.sqrt(indices * h / (2.0 * gap)))
        )
    )
    left = h * np.arange(f.size - 1, dtype=float)
    linear_term = np.sum(
        2.0
        * np.diff(f)
        / h
        * (
            np.sqrt((left + h) * (left + h + 2.0 * gap))
            - np.sqrt(left * (left + 2.0 * gap))
            - 2.0
            * (left + gap)
            * (
                np.arcsinh(np.sqrt((left + h) / (2.0 * gap)))
                - np.arcsinh(np.sqrt(left / (2.0 * gap)))
            )
        )
    )
    return AUTHOR_DELTA0_EV * float(np.exp(-(const_term + linear_term)))


def _make_bundle(
    tmp_path: Path,
    *,
    recorded_h: float | None = None,
) -> tuple[Path, Path, np.ndarray, np.ndarray, float]:
    gap = AUTHOR_DELTA0_EV
    h = 20.0e-6
    E_qp = gap + np.arange(18, dtype=float) * h
    f = 2.5e-4 * np.exp(-np.arange(E_qp.size, dtype=float) / 5.0)
    gap_driven = _author_direct_gap(f, h=h, gap=gap)
    boltzmann_constant = 1.38064852e-23
    electron_charge = 1.60217662e-19
    temperature = 0.2
    thermal_f = 1.0 / (
        np.exp(
            E_qp
            * electron_charge
            / (boltzmann_constant * temperature)
        )
        + 1.0
    )
    gap_thermal = _author_direct_gap(thermal_f, h=h, gap=gap)
    normalized = (gap_driven - gap_thermal) / (gap - gap_thermal)
    author_manifest = json.loads(AUTHOR_SOURCE_MANIFEST.read_text(encoding="utf-8"))
    source_members = {
        member["path"]: member["sha256"]
        for member in author_manifest["members"]
        if member["role"]
        in {
            "array_helpers",
            "coupled_solver",
            "entry_point",
            "material_parameters",
            "quasiparticle_solver",
        }
    }
    executed_source_members = {
        path: digest
        for path, digest in source_members.items()
        if Path(path).name != "Figure_6.py"
    }
    repository_root = Path(__file__).resolve().parents[2]
    adapter_path = (
        repository_root / "validation" / "fischer_2023" / "fig6_author_adapter.py"
    )
    metadata = {
        "adapter": {
            "dependency_sha256s": {
                path.relative_to(Path(__file__).resolve().parents[2]).as_posix(): (
                    source_sha256(path)
                )
                for path in ADAPTER_DEPENDENCIES
            },
            "execution_binding": ADAPTER_SOURCE_EXECUTION_BINDING,
            "hash_kind": ADAPTER_SOURCE_HASH_KIND,
            "path": adapter_path.relative_to(repository_root).as_posix(),
            "sha256": source_sha256(adapter_path),
            "subprocess_wrapper_sha256": SUBPROCESS_WRAPPER_SHA256,
        },
        "author_source": {
            "archive_sha256": author_manifest["archive"]["sha256"],
            "member_sha256s": source_members,
            "source_id": author_manifest["source_id"],
        },
        "all_finite": True,
        "actual_t_star_over_delta": 0.34,
        "array_origins": {
            "E_qp": "adapter-derived",
            "f_final": "author final state",
            "initial_state": "author initial state",
            "n_phonon_final": "author final state",
            "omega_phonon": "adapter-derived",
            "phonon_residual_author_s_inv": "author phonon residual",
            "qp_phonon_residual_author_s_inv": "author QP phonon residual",
            "qp_photon_residual_author_s_inv": "author QP photon residual",
            "residual": "author residual",
        },
        "author_driver_contract": {"fragment_count": 17},
        "author_numerical_constants": {
            "boltzmann_constant_J_per_K": boltzmann_constant,
            "electron_charge_C": electron_charge,
            "reduced_planck_constant_J_s": 6.626070e-34 / (2.0 * np.pi),
        },
        "c_photon_per_second": 1.0,
        "computed_n_bar": 9.0e5,
        "constructor_seconds": 0.1,
        "executed_source": {
            "loading": "immutable_child_byte_snapshots_v1",
            "member_sha256s": executed_source_members,
            "module_sha256s": {
                Path(path).stem: digest
                for path, digest in executed_source_members.items()
            },
            "transformations": [],
        },
        "execution_security": "synthetic isolated-Python test fixture",
        "f_max": float(np.max(f)),
        "f_min": float(np.min(f)),
        "gap_driven_eV": gap_driven,
        "gap_thermal_eV": gap_thermal,
        "h_eV": h if recorded_h is None else recorded_h,
        "input": {
            "max_newton_steps": 10,
            "mode": "compatibility_smoke",
            "num_energy_cells": int(E_qp.size),
            "relative_step_threshold": 1e-7,
            "rng_seed": 0,
            "sweep_index": None,
            "target_t_star_over_delta": 0.34,
            "temperature_K": temperature,
        },
        "last_relative_qp_step": 1e-9,
        "material_parameters": {
            "T_c_K": 1.184,
            "debye_energy_eV": 0.03688,
            "fermi_density_of_states_per_eV_m3": 1.74e28,
            "ion_density_per_m3": 6.02e28,
            "tau_0_s": 438e-9,
            "tau_0_pb_s": 255e-12,
            "tau_l_s": 255e-12,
            "zero_temperature_gap_eV": gap,
        },
        "max_power_mismatch": 1e-12,
        "n_ph_max": 1.0,
        "n_ph_min": 0.0,
        "newton_iterations": 4,
        "newton_seconds": 0.2,
        "normalized_numerical_observable": normalized,
        "photon_bin": 1,
        "qualification": "synthetic completed author-adapter bundle",
        "rounded_gap_eV": gap,
        "runtime": {
            "matplotlib": "test",
            "numba": "test",
            "numpy": np.__version__,
            "platform": "test",
            "python": "test",
            "scipy": "test",
        },
        "schema": ADAPTER_SCHEMA,
        "scientific_status": "finite_state",
        "solved_photon_energy_eV": h,
        "subprocess_isolation": {
            "flags": ["-I", "-B"],
            "isolated_python": True,
            "os_sandbox": False,
            "user_site_enabled": False,
        },
    }
    n_ph = np.zeros(E_qp.size - 1, dtype=float)
    omega_ph = h * np.arange(1, E_qp.size, dtype=float)
    initial = np.concatenate((f, n_ph))
    residual = np.zeros_like(initial)
    bundle = tmp_path / ("bundle" if recorded_h is None else "bad-bundle")
    write_author_point_bundle(
        AuthorPointResult(
            metadata=metadata,
            arrays={
                "E_qp": E_qp,
                "f_final": f,
                "initial_state": initial,
                "n_phonon_final": n_ph,
                "omega_phonon": omega_ph,
                "phonon_residual_author_s_inv": np.zeros_like(n_ph),
                "qp_phonon_residual_author_s_inv": np.zeros_like(f),
                "qp_photon_residual_author_s_inv": np.zeros_like(f),
                "residual": residual,
            },
            stdout="author stdout\n",
            stderr="author stderr\n",
        ),
        bundle,
    )
    bundle_manifest = json.loads(
        (bundle / "manifest.json").read_text(encoding="utf-8")
    )
    anchor_path = tmp_path / (
        "anchor.json" if recorded_h is None else "bad-anchor.json"
    )
    anchor_path.write_text(
        json.dumps(
            {
                "adapter": metadata["adapter"],
                "anchor_id": "synthetic-reviewed-test-anchor",
                "author_source_manifest": {
                    "path": AUTHOR_SOURCE_MANIFEST.relative_to(
                        repository_root
                    ).as_posix(),
                    "sha256": source_sha256(AUTHOR_SOURCE_MANIFEST),
                },
                "bundle_manifest_sha256": hashlib.sha256(
                    (bundle / "manifest.json").read_bytes()
                ).hexdigest(),
                "expected_input": {
                    "max_newton_steps": 10,
                    "mode": "compatibility_smoke",
                    "num_energy_cells": int(E_qp.size),
                    "relative_step_threshold": 1e-7,
                    "rng_seed": 0,
                    "sweep_index": None,
                    "target_t_star_over_delta": 0.34,
                    "temperature_K": 0.2,
                },
                "expected_parameters": {
                    "actual_t_star_over_delta": 0.34,
                    "author_numerical_constants": metadata[
                        "author_numerical_constants"
                    ],
                    "computed_n_bar": 9.0e5,
                    "h_eV": h if recorded_h is None else recorded_h,
                    "rounded_gap_eV": gap,
                    "solved_photon_energy_eV": h,
                },
                "expected_result": {
                    "arrays": bundle_manifest["arrays"],
                    "gap_driven_eV": metadata["gap_driven_eV"],
                    "gap_thermal_eV": metadata["gap_thermal_eV"],
                    "last_relative_qp_step": metadata[
                        "last_relative_qp_step"
                    ],
                    "max_power_mismatch": metadata["max_power_mismatch"],
                    "newton_iterations": metadata["newton_iterations"],
                    "normalized_numerical_observable": metadata[
                        "normalized_numerical_observable"
                    ],
                },
                "expected_scientific_status": "finite_state",
                "schema": ANCHOR_SCHEMA,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return bundle, anchor_path, E_qp, f, gap_driven


def test_frozen_state_diagnostic_reproduces_author_linear_gap(
    tmp_path: Path,
) -> None:
    bundle, anchor, E_qp, f, author_gap = _make_bundle(tmp_path)

    payload = build_frozen_gap_comparison(bundle, anchor_path=anchor)

    assert payload["schema"] == DIAGNOSTIC_SCHEMA
    calculation = payload["calculation"]
    assert calculation["qpsim_recomputed"]["direct_gap_eV"] == pytest.approx(
        author_gap,
        rel=2e-15,
        abs=0.0,
    )
    assert calculation["deltas"]["direct_gap_absolute_eV"] < 1e-18
    assert calculation["deltas"]["direct_integral_absolute"] < 2e-15
    assert calculation["deltas"]["normalized_observable_absolute"] < 1e-12
    assert calculation["deltas"]["thermal_gap_absolute_eV"] < 1e-18
    assert calculation["deltas"]["thermal_integral_absolute"] < 2e-15
    assert (
        calculation["parameters"]["delta0_eV"].hex()
        == AUTHOR_DELTA0_EV.hex()
    )
    assert (
        calculation["qpsim_recomputed"]["thermal_gap_eV"]
        == calculation["author_reported"]["thermal_gap_eV"]
    )
    assert (
        calculation["qpsim_recomputed"]["normalized_numerical_observable"]
        == calculation["author_reported"]["normalized_numerical_observable"]
    )
    assert payload["coordinate_contract"]["occupation_transform"] == "none"
    assert payload["input_bundle"]["manifest_sha256"] == hashlib.sha256(
        (bundle / "manifest.json").read_bytes()
    ).hexdigest()
    assert "stage_id" not in json.dumps(payload)
    assert "nonlinear component-replacement ladder" in payload["qualification"]
    assert payload["runtime"] == {
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }

    # Passing the raw author nodes as centers would silently shift every
    # integration interval by h/2. The explicit coordinate carrier is material.
    raw_coordinate_gap = gap_from_distribution_direct(
        f,
        E_qp,
        gap=AUTHOR_DELTA0_EV,
        delta0=AUTHOR_DELTA0_EV,
        samples="authors",
    )
    assert abs(raw_coordinate_gap - author_gap) > 1e-10


def test_loader_rejects_tampered_native_array(tmp_path: Path) -> None:
    bundle, _anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    path = bundle / "f_final.npy"
    tampered = bytearray(path.read_bytes())
    tampered[-1] ^= 1
    path.write_bytes(tampered)

    with pytest.raises(FrozenStateDiagnosticError, match="SHA-256"):
        load_author_point_bundle(bundle)


def test_loader_hashes_and_consumes_each_array_from_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, _anchor, _E_qp, f, _gap = _make_bundle(tmp_path)
    target = bundle / "f_final.npy"
    original_read_bytes = Path.read_bytes
    reads = 0

    def guarded_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == target:
            reads += 1
            if reads > 1:
                raise AssertionError("array path was re-read after authentication")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    loaded = load_author_point_bundle(bundle)
    np.testing.assert_array_equal(loaded.arrays["f_final"], f)
    assert reads == 1


def test_anchor_is_parsed_and_used_from_one_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle, anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    original_read_bytes = Path.read_bytes
    reads = 0

    def guarded_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == anchor:
            reads += 1
            if reads > 1:
                raise AssertionError("anchor path was re-read after validation")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    payload = build_frozen_gap_comparison(bundle, anchor_path=anchor)
    assert payload["input_bundle"]["anchor"]["anchor_id"] == (
        "synthetic-reviewed-test-anchor"
    )
    assert reads == 1


def test_author_equivalent_rejects_any_source_transformation(
    tmp_path: Path,
) -> None:
    bundle, _anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["input"]["mode"] = "author_equivalent"
    manifest["metadata"]["executed_source"]["transformations"] = [
        {"transform_id": "unreviewed-change"}
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        FrozenStateDiagnosticError,
        match="must execute untransformed",
    ):
        load_author_point_bundle(bundle)


def test_author_equivalent_executed_digests_must_equal_authenticated_closure(
    tmp_path: Path,
) -> None:
    bundle, _anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["input"]["mode"] = "author_equivalent"
    executed = manifest["metadata"]["executed_source"]
    member_path = next(iter(executed["member_sha256s"]))
    module_name = Path(member_path).stem
    executed["member_sha256s"][member_path] = "0" * 64
    executed["module_sha256s"][module_name] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(
        FrozenStateDiagnosticError,
        match="do not equal the authenticated",
    ):
        load_author_point_bundle(bundle)


def test_anchor_result_rejects_scalar_or_array_substitution(
    tmp_path: Path,
) -> None:
    bundle, anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    original = json.loads(anchor.read_text(encoding="utf-8"))

    scalar_tamper = json.loads(json.dumps(original))
    scalar_tamper["expected_result"]["normalized_numerical_observable"] += 1e-6
    anchor.write_text(json.dumps(scalar_tamper), encoding="utf-8")
    with pytest.raises(FrozenStateDiagnosticError, match="scalar result"):
        build_frozen_gap_comparison(bundle, anchor_path=anchor)

    array_tamper = json.loads(json.dumps(original))
    array_tamper["expected_result"]["arrays"]["f_final"]["npy_sha256"] = "0" * 64
    anchor.write_text(json.dumps(array_tamper), encoding="utf-8")
    with pytest.raises(FrozenStateDiagnosticError, match="array result"):
        build_frozen_gap_comparison(bundle, anchor_path=anchor)


def test_reviewed_anchor_rejects_a_self_consistent_manifest_rewrite(
    tmp_path: Path,
) -> None:
    bundle, anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"]["qualification"] = "rewritten together with its files"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FrozenStateDiagnosticError, match="reviewed anchor"):
        build_frozen_gap_comparison(bundle, anchor_path=anchor)


def test_diagnostic_rejects_grid_metadata_that_does_not_match_state(
    tmp_path: Path,
) -> None:
    bundle, anchor, _E_qp, _f, _gap = _make_bundle(
        tmp_path,
        recorded_h=19.0e-6,
    )

    with pytest.raises(FrozenStateDiagnosticError, match="recorded h_eV"):
        build_frozen_gap_comparison(bundle, anchor_path=anchor)


def test_comparison_manifest_is_deterministic(tmp_path: Path) -> None:
    bundle, anchor, _E_qp, _f, _gap = _make_bundle(tmp_path)
    first = write_frozen_gap_comparison(
        bundle,
        tmp_path / "first.json",
        anchor_path=anchor,
    )
    second = write_frozen_gap_comparison(
        bundle,
        tmp_path / "second.json",
        anchor_path=anchor,
    )

    assert first.read_bytes() == second.read_bytes()
