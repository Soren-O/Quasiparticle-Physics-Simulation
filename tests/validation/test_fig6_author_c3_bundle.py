"""Focused contract tests for the formal Figure 6 C3 frozen-grid bundle."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from validation.fischer_2023 import fig6_author_c3_bundle as c3_bundle
from validation.fischer_2023.fig6_author_c0_summary import (
    DEFAULT_SUMMARY as C0_SUMMARY,
)
from validation.fischer_2023.fig6_author_c0_summary import load_c0_summary
from validation.fischer_2023.fig6_author_c2_bundle import (
    array_descriptor,
    load_c2_raw_bundle,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_SCORE as C2_SCORE,
)
from validation.fischer_2023.fig6_author_c2_score import load_c2_score
from validation.fischer_2023.fig6_author_c3_bundle import (
    C3BundleError,
    build_c3_bundle,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
C2_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C2-parameters-v1"
C2B5_SLUG = "c2b5_finite_cutoff_critical_temperature"
ACTIVE = slice(20, 1640)
CHANNELS = (
    "qp_photon",
    "qp_scattering",
    "qp_pair",
    "phonon_scattering",
    "phonon_pair",
    "phonon_escape",
)
FIELDS = ("gain_s_inv", "loss_s_inv", "net_s_inv")

Evidence = tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
    str,
]


def _require_external_c2() -> None:
    if not (C2_BUNDLE / "manifest.json").is_file():
        pytest.skip("Canonical external C2 raw bundle is unavailable.")


@pytest.fixture(scope="module")
def formal_c3() -> Evidence:
    _require_external_c2()
    c2_metadata, c2_arrays, c2_manifest_sha = load_c2_raw_bundle(C2_BUNDLE)
    metadata, arrays = build_c3_bundle(C2_BUNDLE)
    return metadata, arrays, c2_metadata, c2_arrays, c2_manifest_sha


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    assert left.dtype.str == right.dtype.str
    assert left.shape == right.shape
    assert left.tobytes(order="C") == right.tobytes(order="C")


def _bit_exact(actual: np.ndarray, expected: np.ndarray) -> bool:
    left = np.asarray(actual)
    right = np.asarray(expected)
    return (
        left.dtype.str == right.dtype.str
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _assert_canonical_positive_zero(value: np.ndarray) -> None:
    array = np.asarray(value)
    assert np.issubdtype(array.dtype, np.floating)
    assert np.all(array == 0.0)
    assert not np.any(np.signbit(array))


def _stage_records(metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = metadata["stages"]
    assert isinstance(records, list)
    return {record["stage_id"]: record for record in records}


def _stage_array(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    stage_id: str,
    channel: str,
    field: str,
) -> np.ndarray:
    record = _stage_records(metadata)[stage_id]
    name = record["array_names"]["channels"][channel][field]
    return arrays[name]


def _assert_json_scalar_bit_exact(actual: object, expected: object) -> None:
    assert type(actual) is type(expected)
    if isinstance(expected, float):
        assert isinstance(actual, float)
        assert actual.hex() == expected.hex()
    else:
        assert actual == expected


def test_formal_c3_binds_the_accepted_c2_endpoint_not_author_parameters(
    formal_c3: Evidence,
) -> None:
    metadata, _arrays, c2_metadata, c2_arrays, c2_manifest_sha = formal_c3
    score = load_c2_score(C2_SCORE, receipt_path=C2_RECEIPT)
    assert score["acceptance"]["accepted"] is True
    assert score["stage"]["stage_id"] == "C2"
    assert score["stage"]["status"] == "completed"

    bindings = metadata["parent_bindings"]
    assert bindings["c2_raw_manifest_sha256"] == c2_manifest_sha
    assert bindings["c2_raw_manifest_sha256"] == (score["raw_bundle"]["manifest_sha256"])
    assert bindings["c2_score_sha256"] == hashlib.sha256(C2_SCORE.read_bytes()).hexdigest()
    assert bindings["c2_receipt_sha256"] == hashlib.sha256(C2_RECEIPT.read_bytes()).hexdigest()
    assert bindings["c2b5_step_id"] == ("C2b5-finite-cutoff-critical-temperature")
    assert bindings["c2b5_parent_residual"] == array_descriptor(
        c2_arrays[f"{C2B5_SLUG}__residual_s_inv"]
    )

    c2_endpoint = c2_metadata["steps"][-1]["effective_author_units"]
    c3_parameters = metadata["parameters"]
    for key, expected in c2_endpoint.items():
        _assert_json_scalar_bit_exact(c3_parameters["values"][key], expected)
        if isinstance(expected, float):
            assert c3_parameters["hex"][key] == expected.hex()

    author_parameters = load_c0_summary(C0_SUMMARY)["parameters"]
    for key in (
        "T_c_K",
        "boltzmann_constant_J_per_K",
        "c_photon_s_inv",
        "tau_0_pb_s",
    ):
        assert c3_parameters["values"][key].hex() != author_parameters[key].hex()


def test_live_grid_has_exactly_twenty_inactive_guard_cells(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, _c2_metadata, _c2_arrays, _c2_manifest_sha = formal_c3
    active = arrays["native_active_mask"]
    expected_active = np.ones(1640, dtype=bool)
    expected_active[:20] = False
    _assert_bit_exact(active, expected_active)

    projection = metadata["projection"]
    assert projection == {
        "active_cell_count": 1620,
        "guard_cell_count": 20,
        "mapped_left_edge_delta_ueV_max": projection["mapped_left_edge_delta_ueV_max"],
        "mapped_left_edge_delta_ueV_min": projection["mapped_left_edge_delta_ueV_min"],
        "mapped_left_edge_nonzero_count": projection["mapped_left_edge_nonzero_count"],
        "native_cell_count": 1640,
        "native_omega_count": 3600,
        "parent_cell_count": 1620,
        "parent_phonon_count": 1619,
        "projection_kind": "ordinal_identity_embedding_no_interpolation",
        "sample_carrier_delta_ueV_max": projection["sample_carrier_delta_ueV_max"],
        "sample_carrier_delta_ueV_min": projection["sample_carrier_delta_ueV_min"],
        "sample_carrier_nonzero_count": projection["sample_carrier_nonzero_count"],
    }
    _assert_canonical_positive_zero(arrays["native_cell_weights_full"][:20])
    _assert_canonical_positive_zero(arrays["native_cell_density_full"][:20])


def test_parent_embedding_is_i_plus_twenty_with_bit_exact_suffixes_and_padding(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, _c2_metadata, c2_arrays, _c2_manifest_sha = formal_c3
    expected_mapping = np.arange(20, 1640, dtype=np.int64)
    _assert_bit_exact(arrays["parent_to_native_index"], expected_mapping)

    for parent_name, c2_name, projected_name in (
        ("parent_f", "f_final", "projected_f"),
        ("parent_thermal_f", "thermal_f", "projected_thermal_f"),
    ):
        _assert_bit_exact(arrays[parent_name], c2_arrays[c2_name])
        _assert_bit_exact(arrays[projected_name][ACTIVE], arrays[parent_name])
        _assert_canonical_positive_zero(arrays[projected_name][:20])

    for record in metadata["stages"]:
        names = record["array_names"]
        for channel in ("qp_photon", "qp_scattering", "qp_pair"):
            for field in FIELDS:
                _assert_canonical_positive_zero(arrays[names["channels"][channel][field]][:20])
        _assert_canonical_positive_zero(arrays[names["qp_residual_s_inv"]][:20])


def test_parent_phonons_map_exactly_onto_full_omega_with_positive_zero_placeholders(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, _c2_metadata, c2_arrays, _c2_manifest_sha = formal_c3
    _assert_bit_exact(arrays["parent_n_phonon"], c2_arrays["n_phonon_final"])

    omega = arrays["native_omega_ueV"]
    mapping = arrays["parent_phonon_to_native_omega_index"]
    support = arrays["legacy_phonon_support_mask"]
    expected_mapping = np.arange(1, 1620, dtype=np.int64)
    expected_support = np.zeros(3600, dtype=bool)
    expected_support[expected_mapping] = True

    _assert_bit_exact(omega, np.arange(3600, dtype=float))
    _assert_bit_exact(mapping, expected_mapping)
    _assert_bit_exact(support, expected_support)
    _assert_bit_exact(omega[mapping], np.arange(1, 1620, dtype=float))
    _assert_bit_exact(
        arrays["projected_n_phonon"][mapping],
        arrays["parent_n_phonon"],
    )
    _assert_canonical_positive_zero(arrays["projected_n_phonon"][~support])
    assert (
        "non-solved serialization placeholders"
        in (metadata["coordinate_contract"]["native_omega_policy"])
    )


def test_face_roundoff_and_sample_carrier_shift_are_recorded_separately(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, _c2_metadata, _c2_arrays, _c2_manifest_sha = formal_c3
    mapping = arrays["parent_to_native_index"]
    parent_left = arrays["parent_E_left_eV"] * 1.0e6
    expected_face = arrays["native_cell_edges_ueV"][mapping] - parent_left
    face_delta = arrays["mapped_left_edge_delta_ueV"]
    _assert_bit_exact(face_delta, expected_face)

    face_nonzero = int(np.count_nonzero(face_delta))
    assert face_nonzero > 0
    assert float(np.min(face_delta)) < 0.0 < float(np.max(face_delta))
    projection = metadata["projection"]
    assert projection["mapped_left_edge_nonzero_count"] == face_nonzero
    assert projection["mapped_left_edge_delta_ueV_min"].hex() == float(np.min(face_delta)).hex()
    assert projection["mapped_left_edge_delta_ueV_max"].hex() == float(np.max(face_delta)).hex()

    expected_carrier = arrays["native_E_centers_ueV"][mapping] - parent_left
    carrier_delta = arrays["sample_carrier_delta_ueV"]
    _assert_bit_exact(carrier_delta, expected_carrier)
    assert np.all(carrier_delta > 0.0)
    assert np.allclose(carrier_delta, 0.5, rtol=0.0, atol=3e-13)
    assert projection["sample_carrier_nonzero_count"] == 1620
    assert projection["sample_carrier_delta_ueV_min"].hex() == float(np.min(carrier_delta)).hex()
    assert projection["sample_carrier_delta_ueV_max"].hex() == float(np.max(carrier_delta)).hex()
    relabeling = metadata["coordinate_contract"]["sample_relabeling"]
    assert "half-bin sample-carrier shift" in relabeling
    assert "roundoff in the mapped left cell faces" in relabeling


def test_c3p_active_channel_balances_are_bit_exact_c2b5(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, c2_metadata, c2_arrays, _c2_manifest_sha = formal_c3
    stage = _stage_records(metadata)["c3p_projected_author_control"]
    assert stage["parent_stage_id"] == "C2"
    assert stage["coherence_convention"] == "author_left_edge"
    assert stage["density_convention"] == "author_cell_average_eV"
    assert stage["pair_frequency_offset_bins"] == 0

    for channel in CHANNELS:
        for field in FIELDS:
            expected = c2_arrays[f"{C2B5_SLUG}__{channel}__{field}"]
            actual = _stage_array(
                metadata,
                arrays,
                stage["stage_id"],
                channel,
                field,
            )
            _assert_bit_exact(actual[ACTIVE] if channel.startswith("qp_") else actual, expected)

    c2_residual = c2_arrays[f"{C2B5_SLUG}__residual_s_inv"]
    _assert_bit_exact(
        arrays[stage["array_names"]["qp_residual_s_inv"]][ACTIVE],
        c2_residual[:1620],
    )
    _assert_bit_exact(
        arrays[stage["array_names"]["phonon_residual_s_inv"]],
        c2_residual[1620:],
    )
    c2_scalars = c2_metadata["steps"][-1]["operator_scalars"]
    for key in (
        "a_delta",
        "phonon_prefactor_per_eV_s",
        "qp_prefactor_s_inv",
    ):
        expected = c2_scalars[key]
        _assert_json_scalar_bit_exact(stage["operator_scalars"][key], expected)
    assert stage["pair_frequency_offset_bins"] == (c2_scalars["pair_frequency_offset_bins"])


@pytest.mark.parametrize(
    ("left", "right", "changed_channels"),
    (
        (
            "c3p_projected_author_control",
            "c3a_finite_volume_coherence",
            {
                "qp_photon",
                "qp_scattering",
                "qp_pair",
                "phonon_scattering",
                "phonon_pair",
            },
        ),
        (
            "c3a_finite_volume_coherence",
            "c3b_center_pair_labels",
            {"qp_pair", "phonon_pair"},
        ),
        (
            "c3b_center_pair_labels",
            "c3c_native_cell_density",
            {
                "qp_photon",
                "qp_scattering",
                "qp_pair",
                "phonon_scattering",
                "phonon_pair",
            },
        ),
    ),
)
def test_cumulative_stage_substitutions_are_channel_local(
    formal_c3: Evidence,
    left: str,
    right: str,
    changed_channels: set[str],
) -> None:
    metadata, arrays, _c2_metadata, _c2_arrays, _c2_manifest_sha = formal_c3
    stages = _stage_records(metadata)
    assert stages[right]["parent_stage_id"] == left

    for channel in CHANNELS:
        for field in FIELDS:
            left_value = _stage_array(metadata, arrays, left, channel, field)
            right_value = _stage_array(metadata, arrays, right, channel, field)
            if channel in changed_channels:
                assert not _bit_exact(left_value, right_value)
            else:
                _assert_bit_exact(left_value, right_value)


def test_c3_metadata_makes_no_nonlinear_root_or_ordinate_claim(
    formal_c3: Evidence,
) -> None:
    metadata, arrays, _c2_metadata, _c2_arrays, _c2_manifest_sha = formal_c3
    assert metadata["stage"] == {
        "changed_component": "grid_sampling",
        "comparison_stage_id": "C2",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C2",
        "stage_id": "C3",
    }
    assert metadata["limitations"]["scope"] == ("one authenticated C2 frozen point only")
    statement = metadata["limitations"]["statement"]
    for excluded_claim in (
        "No C3 nonlinear root",
        "Newton history",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "paper-parity claim",
    ):
        assert excluded_claim in statement
    assert metadata["observable_control"]["claim"].endswith(
        "Neither is a C3 root or plotted ordinate."
    )

    forbidden_result_keys = {
        "converged",
        "curve",
        "iteration_count",
        "newton_history",
        "ordinate",
        "root",
        "solution",
        "stopping_result",
    }

    def collect_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            result = set(value)
            for nested in value.values():
                result.update(collect_keys(nested))
            return result
        if isinstance(value, list):
            result: set[str] = set()
            for nested in value:
                result.update(collect_keys(nested))
            return result
        return set()

    assert collect_keys(metadata).isdisjoint(forbidden_result_keys)
    assert all(not forbidden_result_keys.intersection(name.lower().split("__")) for name in arrays)


def test_observable_control_does_not_hide_center_carrier_reinterpretation(
    formal_c3: Evidence,
) -> None:
    metadata, _arrays, _c2_metadata, _c2_arrays, _c2_manifest_sha = formal_c3
    control = metadata["observable_control"]
    parent = control["parent_author_left_edge"]
    reembedded = control["author_semantics_reembedding"]
    native = control["native_center_carrier"]

    assert "left-edge samples" in reembedded["interpretation"]
    assert "qpsim cell centers" in native["interpretation"]
    assert abs(reembedded["differences_from_parent"]["driven_integral_signed"]) < 4e-18
    assert abs(reembedded["differences_from_parent"]["thermal_integral_signed"]) < 4e-18
    assert abs(native["differences_from_parent"]["driven_integral_relative_signed"]) > 1e-2
    assert abs(native["differences_from_parent"]["thermal_integral_relative_signed"]) > 1e-2
    assert (
        native["child_full_grid"]["frozen_suppression_ratio"] != parent["frozen_suppression_ratio"]
    )
    author_gap_uev = metadata["parameters"]["values"]["gap_eV"] * 1.0e6
    native_parameters = metadata["native_qpsim_grid_parameters"]
    assert author_gap_uev == np.nextafter(180.0, 0.0)
    assert native_parameters == {
        "delta0_ueV": 180.0,
        "delta0_ueV_hex": (180.0).hex(),
        "gap_ueV": 180.0,
        "gap_ueV_hex": (180.0).hex(),
        "uniform_dE_ueV": 1.0,
        "uniform_dE_ueV_hex": (1.0).hex(),
    }
    assert native["differences_from_parent"]["driven_integral_relative_signed"] > 0.11
    assert native["differences_from_parent"]["thermal_integral_relative_signed"] > 0.029


def test_source_drift_fails_before_c3_evidence_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c2()
    target_relative = "qpsim/physics/spectral.py"
    target = REPOSITORY_ROOT / target_relative
    original = c3_bundle.canonical_source_bytes

    def drifted(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == target.resolve():
            return content + b"\n# simulated source drift\n"
        return content

    assert target_relative in c3_bundle._SOURCE_BYTES_AT_IMPORT
    monkeypatch.setattr(c3_bundle, "canonical_source_bytes", drifted)
    with pytest.raises(
        C3BundleError,
        match=r"C3 numerical source changed during execution: qpsim/physics/spectral\.py",
    ):
        build_c3_bundle(C2_BUNDLE)


def test_c2_trust_anchors_may_not_change_during_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c2()
    original_loader = c3_bundle.load_c2_score
    with tempfile.TemporaryDirectory(
        dir=REPOSITORY_ROOT / "tmp",
        prefix="c3-anchor-race-",
    ) as directory:
        root = Path(directory)
        score = root / "score.json"
        receipt = root / "receipt.json"
        score.write_bytes(C2_SCORE.read_bytes())
        receipt.write_bytes(C2_RECEIPT.read_bytes())

        def mutating_loader(
            path: Path,
            *,
            receipt_path: Path,
        ) -> dict[str, Any]:
            result = original_loader(path, receipt_path=receipt_path)
            path.write_bytes(path.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(c3_bundle, "load_c2_score", mutating_loader)
        with pytest.raises(
            C3BundleError,
            match="C2 score or receipt changed during C3 validation",
        ):
            build_c3_bundle(
                C2_BUNDLE,
                c2_score_path=score,
                c2_receipt_path=receipt,
            )


def test_bundle_write_cleans_partial_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal-c3"
    monkeypatch.setattr(
        c3_bundle,
        "build_c3_bundle",
        lambda *_args, **_kwargs: (
            {"minimal": True},
            {
                "a": np.array([1.0]),
                "b": np.array([2.0]),
            },
        ),
    )
    original_encoder = c3_bundle._npy_bytes
    calls = 0

    def interrupted_encoder(value: np.ndarray) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interrupted write")
        return original_encoder(value)

    monkeypatch.setattr(c3_bundle, "_npy_bytes", interrupted_encoder)
    with pytest.raises(OSError, match="simulated interrupted write"):
        c3_bundle.write_c3_bundle(C2_BUNDLE, output)
    assert not output.exists()
    assert list(tmp_path.glob(".formal-c3.*.tmp")) == []
