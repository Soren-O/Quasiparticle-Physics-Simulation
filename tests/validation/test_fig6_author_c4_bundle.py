"""Focused producer tests for the formal Figure 6 C4 photon bundle."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qpsim.collisions.sub_gap_photon import sub_gap_photon_collision_rates
from qpsim.physics.spectral import SpectralContext
from validation.fischer_2023 import fig6_author_c4_bundle as c4_bundle
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_SCORE as C3_SCORE,
)
from validation.fischer_2023.fig6_author_c3_score import (
    RAW_SCHEMA as C3_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    RECEIPT_SCHEMA as C3_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    SCHEMA as C3_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c3_score import (
    load_c3_raw_bundle,
    load_c3_score,
)
from validation.fischer_2023.fig6_author_c4_bundle import (
    PARENT_OPERATOR_STAGE_ID,
    SCHEMA,
    SECONDS_PER_NS,
    build_c4_bundle,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _first_listable_bundle(*candidates: Path) -> Path:
    """Prefer the original bundle directory; fall back to a byte-identical
    regeneration when the original is present but unreadable on this host."""

    for candidate in candidates:
        try:
            if (candidate / "manifest.json").is_file():
                next(iter(candidate.iterdir()), None)
                return candidate
        except OSError:
            continue
    return candidates[0]

C2_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C2-parameters-v1"
C3_BUNDLE = _first_listable_bundle(
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C3-grid-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C3-grid-regen-v1",
)
GUARD_COUNT = 20
ACTIVE = slice(GUARD_COUNT, 1640)
TERMINAL_INDICES = np.array([1619, 1639], dtype=np.int64)

EXPECTED_ARRAY_NAMES = {
    "arithmetic_delta_gain_s_inv",
    "arithmetic_delta_loss_s_inv",
    "arithmetic_delta_net_s_inv",
    "hybrid_phonon_residual_s_inv",
    "hybrid_qp_residual_s_inv",
    "operator_delta_gain_s_inv",
    "operator_delta_loss_s_inv",
    "operator_delta_net_s_inv",
    "parent_active_mask",
    "parent_cell_weights_ueV",
    "parent_f",
    "parent_phonon_residual_s_inv",
    "parent_qp_photon_gain_s_inv",
    "parent_qp_photon_loss_s_inv",
    "parent_qp_photon_net_s_inv",
    "parent_qp_residual_s_inv",
    "qpsim_author_endpoint_gain_s_inv",
    "qpsim_author_endpoint_loss_s_inv",
    "qpsim_author_endpoint_net_s_inv",
    "qpsim_gain_ns_inv",
    "qpsim_gain_s_inv",
    "qpsim_loss_ns_inv",
    "qpsim_loss_rate_ns_inv",
    "qpsim_loss_s_inv",
    "qpsim_net_ns_inv",
    "qpsim_net_s_inv",
    "terminal_extension_gain_s_inv",
    "terminal_extension_loss_s_inv",
    "terminal_extension_net_s_inv",
    "terminal_extension_support_mask",
}

PARENT_CHANNELS = (
    "qp_photon",
    "qp_scattering",
    "qp_pair",
    "phonon_scattering",
    "phonon_pair",
    "phonon_escape",
)
PARENT_FIELDS = ("gain", "loss", "net")
EXPECTED_FROZEN_NAMES = {
    "projected_f",
    "native_E_centers_ueV",
    "native_dE_ueV",
    "native_active_mask",
    "native_cell_density_full",
    "native_cell_weights_full",
    "native_K_plus_full",
    *(
        f"{PARENT_OPERATOR_STAGE_ID}__{channel}__{field}_s_inv"
        for channel in PARENT_CHANNELS
        for field in PARENT_FIELDS
    ),
    f"{PARENT_OPERATOR_STAGE_ID}__qp_residual_s_inv",
    f"{PARENT_OPERATOR_STAGE_ID}__phonon_residual_s_inv",
}

ReplayCall = tuple[Path, Path]
Evidence = tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, Any],
    dict[str, np.ndarray],
    str,
    list[ReplayCall],
]


def _require_formal_parents() -> None:
    required = (
        C2_BUNDLE / "manifest.json",
        C3_BUNDLE / "manifest.json",
        C3_SCORE,
        C3_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Canonical C2/C3 evidence needed by formal C4 is unavailable.")


@pytest.fixture(scope="module")
def formal_c4() -> Evidence:
    _require_formal_parents()
    replay_calls: list[ReplayCall] = []
    original_rebuild = c4_bundle.build_c3_score

    def recording_rebuild(
        c3_bundle_dir: Path,
        *,
        c2_bundle_dir: Path,
    ) -> dict[str, Any]:
        replay_calls.append((c3_bundle_dir, c2_bundle_dir))
        return original_rebuild(
            c3_bundle_dir,
            c2_bundle_dir=c2_bundle_dir,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(c4_bundle, "build_c3_score", recording_rebuild)
        metadata, arrays = build_c4_bundle(
            C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )
    c3_metadata, c3_arrays, c3_manifest_sha = load_c3_raw_bundle(C3_BUNDLE)
    return (
        metadata,
        arrays,
        c3_metadata,
        c3_arrays,
        c3_manifest_sha,
        replay_calls,
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


def _descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(_npy_bytes(array)).hexdigest(),
        "shape": list(array.shape),
    }


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    assert left.dtype.str == right.dtype.str
    assert left.shape == right.shape
    assert left.tobytes(order="C") == right.tobytes(order="C")


def _assert_positive_zero(value: np.ndarray) -> None:
    array = np.asarray(value)
    assert np.issubdtype(array.dtype, np.floating)
    assert np.all(array == 0.0)
    assert not np.any(np.signbit(array))


def _manual_photon_operator(
    f: np.ndarray,
    active: np.ndarray,
    cell_density: np.ndarray,
    K_plus: np.ndarray,
    *,
    photon_step: int,
    n_bar: float,
    c_photon_ns_inv: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent transcription of the public gain/loss-rate contract."""

    occupation = np.asarray(f)
    one_minus_f = np.maximum(1.0 - occupation, 0.0)
    gain = np.zeros_like(occupation)
    loss_rate = np.zeros_like(occupation)
    size = occupation.size
    for i in range(size):
        if not active[i]:
            continue
        j_up = i + photon_step
        if j_up < size:
            coefficient = cell_density[j_up] * K_plus[i, j_up]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_up] * (n_bar + 1.0)
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_up] * n_bar
        j_down = i - photon_step
        if j_down >= 0 and active[j_down]:
            coefficient = cell_density[j_down] * K_plus[i, j_down]
            gain[i] += c_photon_ns_inv * coefficient * occupation[j_down] * n_bar
            loss_rate[i] += c_photon_ns_inv * coefficient * one_minus_f[j_down] * (n_bar + 1.0)
    return gain * one_minus_f, loss_rate


def test_c4_binds_and_independently_replays_the_exact_accepted_c3_parent(
    formal_c4: Evidence,
) -> None:
    metadata, _arrays, _c3_metadata, _c3_arrays, c3_manifest_sha, calls = formal_c4
    accepted = load_c3_score(C3_SCORE, receipt_path=C3_RECEIPT)

    assert calls == [(C3_BUNDLE, C2_BUNDLE)]
    assert accepted["acceptance"]["accepted"] is True
    assert accepted["stage"]["stage_id"] == "C3"
    assert accepted["stage"]["status"] == "completed"
    bindings = metadata["parent_bindings"]
    assert bindings == {
        "c2_raw_manifest_sha256": accepted["parent_bindings"]["c2_raw_manifest_sha256"],
        "c3_operator_stage_id": "c3c_native_cell_density",
        "c3_raw_manifest_sha256": c3_manifest_sha,
        "c3_raw_schema": C3_RAW_SCHEMA,
        "c3_receipt_path": C3_RECEIPT.relative_to(REPOSITORY_ROOT).as_posix(),
        "c3_receipt_schema": C3_RECEIPT_SCHEMA,
        "c3_receipt_sha256": hashlib.sha256(C3_RECEIPT.read_bytes()).hexdigest(),
        "c3_score_path": C3_SCORE.relative_to(REPOSITORY_ROOT).as_posix(),
        "c3_score_schema": C3_SCORE_SCHEMA,
        "c3_score_sha256": hashlib.sha256(C3_SCORE.read_bytes()).hexdigest(),
        "c3_stage_id": "C3",
    }
    assert c3_manifest_sha == accepted["raw_bundle"]["manifest_sha256"]


def test_c4_has_exact_thirty_array_closure_and_descriptor_coverage(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    assert len(EXPECTED_ARRAY_NAMES) == 30
    assert set(arrays) == EXPECTED_ARRAY_NAMES
    assert set(metadata["array_descriptors"]) == EXPECTED_ARRAY_NAMES
    for name, value in arrays.items():
        assert metadata["array_descriptors"][name] == _descriptor(value)


def test_public_loss_rate_is_converted_to_actual_loss_before_comparison(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    f = arrays["parent_f"]

    _assert_bit_exact(
        arrays["qpsim_loss_ns_inv"],
        arrays["qpsim_loss_rate_ns_inv"] * f,
    )
    _assert_bit_exact(
        arrays["qpsim_net_ns_inv"],
        arrays["qpsim_gain_ns_inv"] - arrays["qpsim_loss_ns_inv"],
    )
    assert not np.array_equal(
        arrays["qpsim_loss_rate_ns_inv"],
        arrays["qpsim_loss_ns_inv"],
    )
    assert float(np.sum(arrays["qpsim_loss_rate_ns_inv"][ACTIVE])) > 1.0e5 * float(
        np.sum(arrays["qpsim_loss_ns_inv"][ACTIVE])
    )
    assert metadata["units"]["public_return_contract"] == (
        "gain includes target Pauli factor; loss_rate multiplies f to form actual loss"
    )


def test_native_per_ns_and_comparison_per_second_arrays_are_exactly_bound(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    for field in ("gain", "loss", "net"):
        _assert_bit_exact(
            arrays[f"qpsim_{field}_s_inv"],
            arrays[f"qpsim_{field}_ns_inv"] / SECONDS_PER_NS,
        )
    inputs = metadata["operator_inputs"]
    assert inputs["seconds_per_ns"] == {
        "hex": SECONDS_PER_NS.hex(),
        "value": SECONDS_PER_NS,
    }
    assert inputs["c_photon_ns_inv"]["value"] == (
        inputs["c_photon_s_inv"]["value"] * SECONDS_PER_NS
    )
    assert metadata["units"]["comparison_arrays"] == "per second"
    assert metadata["units"]["public_native_arrays"] == "per nanosecond"


def test_formal_operator_uses_exact_m20_snap_and_inherited_drive(
    formal_c4: Evidence,
) -> None:
    metadata, _arrays, c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    inputs = metadata["operator_inputs"]
    parent = c3_metadata["parameters"]["values"]

    assert inputs["photon_step_bins"] == parent["photon_bin"] == 20
    assert inputs["dE_ueV"] == {"hex": (1.0).hex(), "value": 1.0}
    assert inputs["omega_0_ueV"] == {"hex": (20.0).hex(), "value": 20.0}
    assert inputs["snap_fraction_of_bin"] == {
        "hex": (0.0).hex(),
        "value": 0.0,
    }
    assert inputs["omega_0_ueV"]["value"] == (parent["photon_bin"] * parent["h_eV"] * 1.0e6)
    assert inputs["n_bar"]["value"].hex() == parent["n_bar"].hex()
    assert inputs["c_photon_s_inv"]["value"].hex() == (parent["c_photon_s_inv"].hex())


def test_comparison_and_coordinate_contracts_state_exact_c4_semantics(
    formal_c4: Evidence,
) -> None:
    metadata, _arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    assert metadata["comparison_contract"] == {
        "arithmetic_control": (
            "qpsim source-order arithmetic and per-nanosecond units, "
            "with the author terminal omission restored"
        ),
        "candidate": "public qpsim sub-gap photon gain and loss_rate",
        "loss_comparison": (
            "physical loss = returned loss_rate * frozen f; the raw loss_rate "
            "coefficient is never compared directly to C3c loss"
        ),
        "parent": "accepted C3c author-form QP photon gain/loss/net",
        "semantic_delta": (
            "candidate minus arithmetic control, isolated from candidate "
            "minus C3c floating-point reordering"
        ),
    }
    assert metadata["coordinate_contract"] == {
        "active_child_indices": "[20, 1640)",
        "coherence": "accepted C3c native SpectralContext K_plus",
        "density": "accepted C3c native partner cell_density",
        "guard_child_indices": "[0, 20), canonical positive zero",
        "native_cell_count": 1640,
        "photon_mapping": "child i <-> child i+20; no interpolation",
    }


def test_formal_public_arrays_reproduce_kplus_and_native_cell_density(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, c3_arrays, _sha, _calls = formal_c4
    inputs = metadata["operator_inputs"]
    expected_gain, expected_loss_rate = _manual_photon_operator(
        arrays["parent_f"],
        c3_arrays["native_active_mask"],
        c3_arrays["native_cell_density_full"],
        c3_arrays["native_K_plus_full"],
        photon_step=inputs["photon_step_bins"],
        n_bar=inputs["n_bar"]["value"],
        c_photon_ns_inv=inputs["c_photon_ns_inv"]["value"],
    )
    _assert_bit_exact(arrays["qpsim_gain_ns_inv"], expected_gain)
    _assert_bit_exact(arrays["qpsim_loss_rate_ns_inv"], expected_loss_rate)


def test_all_qp_guard_outputs_are_canonical_positive_zero(
    formal_c4: Evidence,
) -> None:
    _metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    checked = 0
    for value in arrays.values():
        if value.dtype.kind == "f" and value.shape == (1640,):
            _assert_positive_zero(value[:GUARD_COUNT])
            checked += 1
    assert checked == 26
    assert not np.any(arrays["parent_active_mask"][:GUARD_COUNT])
    assert not np.any(arrays["terminal_extension_support_mask"][:GUARD_COUNT])


def test_formal_point_is_scientifically_nonvacuous(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    inputs = metadata["operator_inputs"]
    active = arrays["parent_active_mask"]
    f_active = arrays["parent_f"][active]

    assert inputs["omega_0_ueV"]["value"] > 0.0
    assert inputs["n_bar"]["value"] > 0.0
    assert inputs["c_photon_s_inv"]["value"] > 0.0
    assert np.all(f_active > 0.0)
    assert np.ptp(f_active) > 0.0
    for name in (
        "qpsim_gain_s_inv",
        "qpsim_loss_s_inv",
        "qpsim_net_s_inv",
    ):
        assert np.all(arrays[name][active] != 0.0)
    assert (
        float(
            arrays["parent_cell_weights_ueV"]
            @ (arrays["qpsim_gain_s_inv"] + arrays["qpsim_loss_s_inv"])
        )
        > 0.0
    )
    for field in ("gain", "loss", "net"):
        extension = arrays[f"terminal_extension_{field}_s_inv"]
        assert np.all(extension[TERMINAL_INDICES] != 0.0)


def test_terminal_extension_is_exactly_supported_at_1619_and_1639(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    assert metadata["endpoint_contract"]["terminal_child_indices"] == [1619, 1639]
    _assert_bit_exact(
        np.flatnonzero(arrays["terminal_extension_support_mask"]),
        TERMINAL_INDICES,
    )
    for field in ("gain", "loss", "net"):
        extension = arrays[f"terminal_extension_{field}_s_inv"]
        _assert_bit_exact(np.flatnonzero(extension), TERMINAL_INDICES)
        _assert_bit_exact(
            arrays[f"qpsim_author_endpoint_{field}_s_inv"] + extension,
            arrays[f"qpsim_{field}_s_inv"],
        )


def test_qpsim_photon_operator_conserves_weighted_qp_number(
    formal_c4: Evidence,
) -> None:
    _metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    weights = arrays["parent_cell_weights_ueV"]
    gain = arrays["qpsim_gain_s_inv"]
    loss = arrays["qpsim_loss_s_inv"]
    net = arrays["qpsim_net_s_inv"]
    turnover = float(weights @ (gain + loss))
    relative_error = abs(float(weights @ net)) / turnover

    assert turnover > 2.5e4
    assert relative_error < 1.0e-12

    # Treating the returned coefficient as physical loss is not a benign
    # relabeling: even after unit conversion it destroys number conservation.
    wrong_net = gain - arrays["qpsim_loss_rate_ns_inv"] / SECONDS_PER_NS
    assert abs(float(weights @ wrong_net)) / turnover > 1.0e4


def test_only_photon_and_qp_residual_change_at_c4(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, c3_arrays, _sha, _calls = formal_c4
    _assert_bit_exact(
        arrays["hybrid_phonon_residual_s_inv"],
        arrays["parent_phonon_residual_s_inv"],
    )
    _assert_bit_exact(
        arrays["hybrid_qp_residual_s_inv"],
        arrays["parent_qp_residual_s_inv"] + arrays["operator_delta_net_s_inv"],
    )
    locality = metadata["component_locality"]
    assert locality["phonon_residual_bit_exact"] is True
    assert locality["qp_residual_update"] == (
        "hybrid_qp_residual = parent_qp_residual + (qpsim_photon_net - parent_photon_net)"
    )

    descriptors = metadata["frozen_inputs"]["descriptors"]
    assert set(descriptors) == EXPECTED_FROZEN_NAMES
    for name in EXPECTED_FROZEN_NAMES:
        assert descriptors[name] == _descriptor(c3_arrays[name])
    assert metadata["frozen_inputs"]["mutation_check_after_operator"] is True


def test_c4_metadata_makes_no_nonlinear_or_observable_claim(
    formal_c4: Evidence,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    assert metadata["stage"] == {
        "changed_component": "photon_operator",
        "comparison_stage_id": "c3c_native_cell_density",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C3",
        "stage_id": "C4",
    }
    assert metadata["limitations"]["scope"] == ("one authenticated C3c frozen point only")
    statement = metadata["limitations"]["statement"]
    for excluded_claim in (
        "No C4 nonlinear root",
        "Newton history",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "observable change",
        "paper-parity claim",
    ):
        assert excluded_claim in statement

    forbidden = {
        "converged",
        "curve",
        "iteration_count",
        "newton_history",
        "ordinate",
        "root",
        "solution",
        "stopping_result",
    }

    def nested_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value).union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert nested_keys(metadata).isdisjoint(forbidden)
    assert all(not set(name.lower().split("__")).intersection(forbidden) for name in arrays)


def test_write_is_exclusive_and_manifest_closes_over_exact_arrays(
    formal_c4: Evidence,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    output = tmp_path / "formal-c4"
    monkeypatch.setattr(
        c4_bundle,
        "build_c4_bundle",
        lambda *_args, **_kwargs: (metadata, arrays),
    )

    manifest_path = c4_bundle.write_c4_bundle(
        C3_BUNDLE,
        output,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == SCHEMA
    assert manifest["metadata"]["schema"] == SCHEMA
    expected_files = {f"{name}.npy" for name in EXPECTED_ARRAY_NAMES}
    assert set(manifest["files"]) == expected_files
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        *expected_files,
    }
    for name, expected in arrays.items():
        path = output / f"{name}.npy"
        content = path.read_bytes()
        assert manifest["files"][path.name] == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        with path.open("rb") as handle:
            loaded = np.load(handle, allow_pickle=False)
        _assert_bit_exact(loaded, expected)

    with pytest.raises(FileExistsError, match="C4 output already exists"):
        c4_bundle.write_c4_bundle(
            C3_BUNDLE,
            output,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_interrupted_write_removes_partial_temporary_bundle(
    formal_c4: Evidence,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    metadata, arrays, _c3_metadata, _c3_arrays, _sha, _calls = formal_c4
    output = tmp_path / "interrupted-c4"
    monkeypatch.setattr(
        c4_bundle,
        "build_c4_bundle",
        lambda *_args, **_kwargs: (metadata, arrays),
    )
    original_encoder = c4_bundle._npy_bytes
    calls = 0

    def interrupted_encoder(value: np.ndarray) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated C4 write interruption")
        return original_encoder(value)

    monkeypatch.setattr(c4_bundle, "_npy_bytes", interrupted_encoder)
    with pytest.raises(OSError, match="simulated C4 write interruption"):
        c4_bundle.write_c4_bundle(
            C3_BUNDLE,
            output,
            c2_bundle_dir=C2_BUNDLE,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".interrupted-c4.*.tmp")) == []


def test_nonunit_width_microgrid_uses_density_not_capacity_as_coefficient() -> None:
    dE = 0.4
    E = np.array([1.2, 1.6, 2.0, 2.4, 2.8])
    ctx = SpectralContext(E, np.full(E.size, dE), gap=1.0)
    f = np.array([0.08, 0.12, 0.03, 0.20, 0.01])
    n_bar = 2.3
    c_photon = 0.7

    gain, loss_rate = sub_gap_photon_collision_rates(
        f,
        ctx,
        omega_0=2 * dE,
        n_bar=n_bar,
        c_phot=c_photon,
    )
    expected_gain, expected_loss_rate = _manual_photon_operator(
        f,
        ctx.active_mask,
        ctx.cell_density,
        ctx.K_plus,
        photon_step=2,
        n_bar=n_bar,
        c_photon_ns_inv=c_photon,
    )
    wrong_gain, wrong_loss_rate = _manual_photon_operator(
        f,
        ctx.active_mask,
        ctx.cell_weights,
        ctx.K_plus,
        photon_step=2,
        n_bar=n_bar,
        c_photon_ns_inv=c_photon,
    )

    _assert_bit_exact(gain, expected_gain)
    _assert_bit_exact(loss_rate, expected_loss_rate)
    assert not np.array_equal(gain, wrong_gain)
    assert not np.array_equal(loss_rate, wrong_loss_rate)
    np.testing.assert_allclose(wrong_gain, dE * gain, rtol=2e-15, atol=0.0)
    np.testing.assert_allclose(
        wrong_loss_rate,
        dE * loss_rate,
        rtol=2e-15,
        atol=0.0,
    )


def test_terminal_bin_microgrid_includes_both_directions_of_final_pair() -> None:
    dE = 0.4
    E = np.array([1.2, 1.6, 2.0, 2.4, 2.8])
    ctx = SpectralContext(E, np.full(E.size, dE), gap=1.0)
    f = np.zeros(E.size)
    f[-1] = 0.25
    c_photon = 0.7

    gain, loss_rate = sub_gap_photon_collision_rates(
        f,
        ctx,
        omega_0=dE,
        n_bar=0.0,
        c_phot=c_photon,
    )
    physical_loss = loss_rate * f
    expected_gain = c_photon * ctx.cell_density[-1] * ctx.K_plus[-2, -1] * f[-1] * (1.0 - f[-2])
    expected_loss = c_photon * ctx.cell_density[-2] * ctx.K_plus[-1, -2] * (1.0 - f[-2]) * f[-1]

    _assert_bit_exact(np.flatnonzero(gain), np.array([E.size - 2]))
    _assert_bit_exact(np.flatnonzero(physical_loss), np.array([E.size - 1]))
    assert gain[-2].hex() == expected_gain.hex()
    assert physical_loss[-1].hex() == expected_loss.hex()
    assert ctx.cell_weights[-2] * gain[-2] == pytest.approx(
        ctx.cell_weights[-1] * physical_loss[-1],
        rel=2e-15,
        abs=0.0,
    )
