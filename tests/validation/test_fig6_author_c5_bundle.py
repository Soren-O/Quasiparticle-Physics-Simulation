"""Focused producer tests for the formal Figure 6 C5 QP-phonon bundle."""

from __future__ import annotations

import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    build_phonon_frequency_map,
    build_recombination_kernel_base,
    build_scattering_kernel_base,
    phonon_collision_rates,
    phonon_occupation_matrices_from_state,
)
from qpsim.physics.spectral import SpectralContext
from validation.fischer_2023 import fig6_author_c5_bundle as c5_bundle
from validation.fischer_2023.fig6_author_c2_score import (
    DEFAULT_RECEIPT as C2_RECEIPT,
)
from validation.fischer_2023.fig6_author_c2_score import DEFAULT_SCORE as C2_SCORE
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import DEFAULT_SCORE as C3_SCORE
from validation.fischer_2023.fig6_author_c3_score import load_c3_raw_bundle
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import DEFAULT_SCORE as C4_SCORE
from validation.fischer_2023.fig6_author_c4_score import (
    RAW_SCHEMA as C4_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    RECEIPT_SCHEMA as C4_RECEIPT_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    SCHEMA as C4_SCORE_SCHEMA,
)
from validation.fischer_2023.fig6_author_c4_score import (
    load_c4_raw_bundle,
    load_c4_score,
)
from validation.fischer_2023.fig6_author_c5_score import load_c5_raw_bundle

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
C4_BUNDLE = _first_listable_bundle(
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C4-photon-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C4-photon-regen-v1",
)
C5_BUNDLE = _first_listable_bundle(
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-regen-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-producer-dev-v5",
)

N_QP = 1640
N_OMEGA = 3600
GUARD_COUNT = 20
AUTHOR_OMEGA_STOP = 1619
SECONDS_PER_NS = 1.0e-9
ACTIVE = slice(GUARD_COUNT, N_QP)
FLOAT_EPS = float(np.finfo(np.float64).eps)
REDUCTION_OPERATION_BUDGET = 8 * N_QP + 64
REDUCTION_GAMMA = (
    REDUCTION_OPERATION_BUDGET * FLOAT_EPS / (1.0 - REDUCTION_OPERATION_BUDGET * FLOAT_EPS)
)

EXPECTED_ARRAY_NAMES = frozenset(
    {
        "c5p_qp_residual_s_inv",
        "c5s_qp_residual_s_inv",
        "c5sp_phonon_residual_s_inv",
        "c5sp_qp_residual_s_inv",
        "parent_E_centers_ueV",
        "parent_active_mask",
        "parent_cell_weights_ueV",
        "parent_dE_ueV",
        "parent_f",
        "parent_legacy_phonon_support_mask",
        "parent_phonon_residual_s_inv",
        "parent_projected_n_phonon",
        "parent_public_qp_photon_gain_s_inv",
        "parent_public_qp_photon_loss_s_inv",
        "parent_public_qp_photon_net_s_inv",
        "parent_qp_pair_gain_s_inv",
        "parent_qp_pair_loss_s_inv",
        "parent_qp_pair_net_s_inv",
        "parent_qp_residual_s_inv",
        "parent_qp_scattering_gain_s_inv",
        "parent_qp_scattering_loss_s_inv",
        "parent_qp_scattering_net_s_inv",
        "parent_qp_scattering_rebucketed_gain_s_inv",
        "parent_qp_scattering_rebucketed_loss_s_inv",
        "qp_pair_delta_gain_s_inv",
        "qp_pair_delta_loss_s_inv",
        "qp_pair_delta_net_s_inv",
        "qp_scattering_delta_gain_s_inv",
        "qp_scattering_delta_loss_s_inv",
        "qp_scattering_delta_net_s_inv",
        "qp_scattering_rebucketed_delta_gain_s_inv",
        "qp_scattering_rebucketed_delta_loss_s_inv",
        "qpsim_N_abs",
        "qpsim_N_emit",
        "qpsim_N_p",
        "qpsim_diff_sign",
        "qpsim_omega_idx_diff",
        "qpsim_omega_idx_sum",
        "qpsim_omega_ueV",
        "qpsim_qp_pair_gain_ns_inv",
        "qpsim_qp_pair_gain_s_inv",
        "qpsim_qp_pair_kernel_ns_inv_ueV_inv",
        "qpsim_qp_pair_loss_ns_inv",
        "qpsim_qp_pair_loss_rate_ns_inv",
        "qpsim_qp_pair_loss_rate_s_inv",
        "qpsim_qp_pair_loss_s_inv",
        "qpsim_qp_pair_net_ns_inv",
        "qpsim_qp_pair_net_s_inv",
        "qpsim_qp_scattering_gain_ns_inv",
        "qpsim_qp_scattering_gain_s_inv",
        "qpsim_qp_scattering_kernel_ns_inv_ueV_inv",
        "qpsim_qp_scattering_loss_ns_inv",
        "qpsim_qp_scattering_loss_rate_ns_inv",
        "qpsim_qp_scattering_loss_rate_s_inv",
        "qpsim_qp_scattering_loss_s_inv",
        "qpsim_qp_scattering_net_ns_inv",
        "qpsim_qp_scattering_net_s_inv",
        "scattering_pauli_cross_term_s_inv",
    }
)


@dataclass
class FormalC5:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    c4_metadata: dict[str, Any]
    c4_arrays: dict[str, np.ndarray]
    c4_manifest_sha256: str
    c3_metadata: dict[str, Any]
    c3_arrays: dict[str, np.ndarray]
    c3_manifest_sha256: str


def _require_formal_parents() -> None:
    required = (
        C2_BUNDLE / "manifest.json",
        C3_BUNDLE / "manifest.json",
        C4_BUNDLE / "manifest.json",
        C5_BUNDLE / "manifest.json",
        C2_SCORE,
        C2_RECEIPT,
        C3_SCORE,
        C3_RECEIPT,
        C4_SCORE,
        C4_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Canonical C2/C3/C4 evidence needed by formal C5 is unavailable.")


@pytest.fixture(scope="module")
def formal_c5() -> FormalC5:
    _require_formal_parents()
    metadata, arrays, _c5_sha = load_c5_raw_bundle(C5_BUNDLE)
    c4_metadata, c4_arrays, c4_sha = load_c4_raw_bundle(C4_BUNDLE)
    c3_metadata, c3_arrays, c3_sha = load_c3_raw_bundle(C3_BUNDLE)
    return FormalC5(
        metadata,
        arrays,
        c4_metadata,
        c4_arrays,
        c4_sha,
        c3_metadata,
        c3_arrays,
        c3_sha,
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
    assert np.all(array == 0.0)
    assert not np.any(np.signbit(array))


def _symmetric_relative_l1(left: np.ndarray, right: np.ndarray) -> float:
    numerator = float(np.sum(np.abs(np.asarray(left) - np.asarray(right))))
    denominator = float(np.sum(np.abs(np.asarray(left))) + np.sum(np.abs(np.asarray(right))))
    return numerator / max(denominator, np.finfo(float).tiny)


def _assert_within_rounding_bound(
    actual: np.ndarray,
    expected: np.ndarray,
    *,
    gamma: float = REDUCTION_GAMMA,
) -> None:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    assert left.shape == right.shape
    scale = np.maximum(
        np.abs(left) + np.abs(right),
        np.finfo(np.float64).tiny,
    )
    assert float(np.max(np.abs(left - right) / (gamma * scale))) <= 1.0


def _context(formal: FormalC5) -> SpectralContext:
    inputs = formal.metadata["operator_inputs"]
    return SpectralContext(
        formal.arrays["parent_E_centers_ueV"],
        formal.arrays["parent_dE_ueV"],
        inputs["gap_ueV"]["value"],
    )


def test_c5_exactly_replays_the_accepted_c4_to_c3_to_c2_chain(
    formal_c5: FormalC5,
) -> None:
    accepted = load_c4_score(C4_SCORE, receipt_path=C4_RECEIPT)
    assert accepted["acceptance"]["status"] == "pass"
    assert all(value is True for key, value in accepted["acceptance"].items() if key != "status")
    assert accepted["stage"]["stage_id"] == "C4"
    assert accepted["stage"]["status"] == "completed"

    bindings = formal_c5.metadata["parent_bindings"]
    assert bindings["c4_raw_schema"] == C4_RAW_SCHEMA
    assert bindings["c4_raw_manifest_sha256"] == formal_c5.c4_manifest_sha256
    assert bindings["c4_raw_manifest_sha256"] == accepted["raw_bundle"]["manifest_sha256"]
    assert bindings["c4_score_schema"] == C4_SCORE_SCHEMA
    assert bindings["c4_receipt_schema"] == C4_RECEIPT_SCHEMA
    assert bindings["c4_score_sha256"] == hashlib.sha256(C4_SCORE.read_bytes()).hexdigest()
    assert bindings["c4_receipt_sha256"] == hashlib.sha256(C4_RECEIPT.read_bytes()).hexdigest()
    assert bindings["c3_raw_manifest_sha256"] == formal_c5.c3_manifest_sha256
    assert (
        bindings["c2_raw_manifest_sha256"] == accepted["parent_bindings"]["c2_raw_manifest_sha256"]
    )


def test_c5_has_exact_fifty_eight_array_closure_and_descriptors(
    formal_c5: FormalC5,
) -> None:
    assert len(EXPECTED_ARRAY_NAMES) == 58
    assert set(c5_bundle.ARRAY_NAMES) == EXPECTED_ARRAY_NAMES
    assert set(formal_c5.arrays) == EXPECTED_ARRAY_NAMES
    assert set(formal_c5.metadata["array_descriptors"]) == EXPECTED_ARRAY_NAMES
    for name, value in formal_c5.arrays.items():
        assert formal_c5.metadata["array_descriptors"][name] == _descriptor(value)


def test_c5_array_shapes_and_dtypes_match_the_frozen_contract(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    matrix_float_names = {
        "qpsim_N_abs",
        "qpsim_N_emit",
        "qpsim_N_p",
        "qpsim_qp_pair_kernel_ns_inv_ueV_inv",
        "qpsim_qp_scattering_kernel_ns_inv_ueV_inv",
    }
    matrix_int64_names = {"qpsim_omega_idx_diff", "qpsim_omega_idx_sum"}
    for name in matrix_float_names:
        assert arrays[name].dtype.str == "<f8"
        assert arrays[name].shape == (N_QP, N_QP)
    for name in matrix_int64_names:
        assert arrays[name].dtype.str == "<i8"
        assert arrays[name].shape == (N_QP, N_QP)
    assert arrays["qpsim_diff_sign"].dtype.str == "|i1"
    assert arrays["qpsim_diff_sign"].shape == (N_QP, N_QP)
    assert arrays["parent_active_mask"].dtype.str == "|b1"
    assert arrays["parent_legacy_phonon_support_mask"].dtype.str == "|b1"
    assert arrays["parent_legacy_phonon_support_mask"].shape == (N_OMEGA,)
    assert arrays["qpsim_omega_ueV"].shape == (N_OMEGA,)
    assert arrays["parent_projected_n_phonon"].shape == (N_OMEGA,)
    assert arrays["parent_phonon_residual_s_inv"].shape == (AUTHOR_OMEGA_STOP,)
    assert arrays["c5sp_phonon_residual_s_inv"].shape == (AUTHOR_OMEGA_STOP,)
    for name, value in arrays.items():
        if name in matrix_float_names | matrix_int64_names | {
            "qpsim_diff_sign",
            "parent_active_mask",
            "parent_legacy_phonon_support_mask",
            "qpsim_omega_ueV",
            "parent_projected_n_phonon",
            "parent_phonon_residual_s_inv",
            "c5sp_phonon_residual_s_inv",
        }:
            continue
        assert value.dtype.str == "<f8", name
        assert value.shape == (N_QP,), name


def test_c5_freezes_exact_parent_state_channels_and_residuals(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    c3 = formal_c5.c3_arrays
    c4 = formal_c5.c4_arrays
    parent = "c3c_native_cell_density"
    comparisons = {
        "parent_E_centers_ueV": c3["native_E_centers_ueV"],
        "parent_dE_ueV": c3["native_dE_ueV"],
        "parent_f": c3["projected_f"],
        "parent_active_mask": c3["native_active_mask"],
        "parent_cell_weights_ueV": c3["native_cell_weights_full"],
        "parent_projected_n_phonon": c3["projected_n_phonon"],
        "parent_legacy_phonon_support_mask": c3["legacy_phonon_support_mask"],
        "parent_qp_scattering_gain_s_inv": c3[f"{parent}__qp_scattering__gain_s_inv"],
        "parent_qp_scattering_loss_s_inv": c3[f"{parent}__qp_scattering__loss_s_inv"],
        "parent_qp_scattering_net_s_inv": c3[f"{parent}__qp_scattering__net_s_inv"],
        "parent_qp_pair_gain_s_inv": c3[f"{parent}__qp_pair__gain_s_inv"],
        "parent_qp_pair_loss_s_inv": c3[f"{parent}__qp_pair__loss_s_inv"],
        "parent_qp_pair_net_s_inv": c3[f"{parent}__qp_pair__net_s_inv"],
        "parent_public_qp_photon_gain_s_inv": c4["qpsim_gain_s_inv"],
        "parent_public_qp_photon_loss_s_inv": c4["qpsim_loss_s_inv"],
        "parent_public_qp_photon_net_s_inv": c4["qpsim_net_s_inv"],
        "parent_qp_residual_s_inv": c4["hybrid_qp_residual_s_inv"],
        "parent_phonon_residual_s_inv": c4["hybrid_phonon_residual_s_inv"],
    }
    for name, expected in comparisons.items():
        _assert_bit_exact(arrays[name], expected)

    frozen = formal_c5.metadata["frozen_inputs"]
    assert frozen["c4_mutation_check_after_operator"] is True
    assert frozen["c3_mutation_check_after_operator"] is True
    assert set(frozen["c4_descriptors"]) == set(formal_c5.c4_arrays)
    for name, value in formal_c5.c4_arrays.items():
        assert frozen["c4_descriptors"][name] == _descriptor(value)
    assert set(frozen["c3_descriptors"]) == set(c5_bundle._c3_frozen_names())
    for name in c5_bundle._c3_frozen_names():
        assert frozen["c3_descriptors"][name] == _descriptor(formal_c5.c3_arrays[name])


def test_c5_operator_inputs_are_exact_inherited_physical_and_unit_values(
    formal_c5: FormalC5,
) -> None:
    inputs = formal_c5.metadata["operator_inputs"]
    parent = formal_c5.c3_metadata["parameters"]["values"]

    assert inputs["T_c_K"] == {
        "hex": c5_bundle.T_C_K.hex(),
        "value": c5_bundle.T_C_K,
    }
    assert inputs["tau_0_ns"] == {
        "hex": c5_bundle.TAU_0_NS.hex(),
        "value": c5_bundle.TAU_0_NS,
    }
    assert inputs["T_bath_K"] == {
        "hex": c5_bundle.T_BATH_K.hex(),
        "value": c5_bundle.T_BATH_K,
    }
    assert inputs["seconds_per_ns"] == {
        "hex": SECONDS_PER_NS.hex(),
        "value": SECONDS_PER_NS,
    }
    assert inputs["T_c_K"]["value"].hex() == parent["T_c_K"].hex()
    assert inputs["T_bath_K"]["value"].hex() == parent["temperature_K"].hex()
    assert inputs["tau_0_ns"]["value"] == parent["tau_0_s"] / SECONDS_PER_NS
    assert inputs["tau_0_parent_s"]["value"].hex() == parent["tau_0_s"].hex()
    assert inputs["gap_ueV"]["value"] == 180.0
    assert (
        inputs["gap_ueV"]["value"]
        == formal_c5.c3_metadata["native_qpsim_grid_parameters"]["gap_ueV"]
    )
    assert inputs["gap_ueV"]["value"] != parent["gap_eV"] * 1.0e6
    assert inputs["gap_parent_eV"]["value"].hex() == parent["gap_eV"].hex()
    assert (
        inputs["boltzmann_constant_J_per_K"]["value"].hex()
        == parent["boltzmann_constant_J_per_K"].hex()
    )
    assert inputs["electron_charge_C"]["value"].hex() == parent["electron_charge_C"].hex()
    assert (
        inputs["kB_ueV_per_K"]["value"]
        == parent["boltzmann_constant_J_per_K"] / parent["electron_charge_C"] * 1.0e6
    )
    assert (
        inputs["kB_T_c_ueV"]["value"] == inputs["kB_ueV_per_K"]["value"] * inputs["T_c_K"]["value"]
    )
    for record in inputs.values():
        assert record["hex"] == record["value"].hex()


def test_public_channels_reproduce_separately_with_only_frozen_overrides(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    ctx = _context(formal_c5)
    # Complete overrides make the result independent of T_bath. Evaluating
    # at zero must reproduce the retained T=0.2 K arrays; any thermal fallback
    # would make the comparison fail.
    scattering_gain, scattering_loss_rate = phonon_collision_rates(
        arrays["parent_f"],
        ctx,
        arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"],
        None,
        0.0,
        enable_scattering=True,
        enable_recombination=False,
        N_p_override=arrays["qpsim_N_p"],
    )
    pair_gain, pair_loss_rate = phonon_collision_rates(
        arrays["parent_f"],
        ctx,
        None,
        arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"],
        0.0,
        enable_scattering=False,
        enable_recombination=True,
        N_emit_override=arrays["qpsim_N_emit"],
        N_abs_override=arrays["qpsim_N_abs"],
    )
    _assert_within_rounding_bound(
        scattering_gain,
        arrays["qpsim_qp_scattering_gain_ns_inv"],
    )
    _assert_within_rounding_bound(
        scattering_loss_rate,
        arrays["qpsim_qp_scattering_loss_rate_ns_inv"],
    )
    _assert_within_rounding_bound(
        pair_gain,
        arrays["qpsim_qp_pair_gain_ns_inv"],
    )
    _assert_within_rounding_bound(
        pair_loss_rate,
        arrays["qpsim_qp_pair_loss_rate_ns_inv"],
    )


def test_frequency_map_and_occupations_are_exact_public_outputs(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    omega, idx_diff, idx_sum, sign = build_phonon_frequency_map(arrays["parent_E_centers_ueV"])
    n_p, n_emit, n_abs = phonon_occupation_matrices_from_state(
        arrays["parent_projected_n_phonon"],
        idx_diff,
        idx_sum,
        sign,
    )
    for name, expected in {
        "qpsim_omega_ueV": omega,
        "qpsim_omega_idx_diff": idx_diff,
        "qpsim_omega_idx_sum": idx_sum,
        "qpsim_diff_sign": sign,
        "qpsim_N_p": n_p,
        "qpsim_N_emit": n_emit,
        "qpsim_N_abs": n_abs,
    }.items():
        _assert_bit_exact(arrays[name], expected)
    assert np.array_equal(omega, np.arange(N_OMEGA, dtype=np.float64))
    assert np.all(idx_diff >= 0)
    assert np.all(idx_diff < N_OMEGA)
    assert np.all(idx_sum >= 0)
    assert np.all(idx_sum < N_OMEGA)
    assert set(np.unique(sign)) == {-1, 0, 1}


def test_qp_kernels_use_kminus_for_scattering_kplus_for_pairs(
    formal_c5: FormalC5,
) -> None:
    ctx = _context(formal_c5)
    expected_scattering = build_scattering_kernel_base(
        ctx,
        c5_bundle.TAU_0_NS,
        c5_bundle.T_C_K,
    )
    expected_pair = build_recombination_kernel_base(
        ctx,
        c5_bundle.TAU_0_NS,
        c5_bundle.T_C_K,
    )
    _assert_within_rounding_bound(
        formal_c5.arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"],
        expected_scattering,
        gamma=32.0 * FLOAT_EPS,
    )
    _assert_within_rounding_bound(
        formal_c5.arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"],
        expected_pair,
        gamma=32.0 * FLOAT_EPS,
    )
    assert not np.array_equal(expected_scattering, expected_pair)
    energy = formal_c5.arrays["parent_E_centers_ueV"]
    kbt_c = formal_c5.metadata["operator_inputs"]["kB_T_c_ueV"]["value"]
    scattering_formula = (
        (energy[:, None] - energy[None, :]) ** 2
        / (c5_bundle.TAU_0_NS * kbt_c**3)
        * formal_c5.c3_arrays["native_K_minus_full"]
    )
    pair_formula = (
        ((energy[:, None] + energy[None, :]) / kbt_c) ** 2
        / (c5_bundle.TAU_0_NS * kbt_c)
        * formal_c5.c3_arrays["native_K_plus_full"]
    )
    _assert_within_rounding_bound(
        expected_scattering,
        scattering_formula,
        gamma=32.0 * FLOAT_EPS,
    )
    _assert_within_rounding_bound(
        expected_pair,
        pair_formula,
        gamma=32.0 * FLOAT_EPS,
    )


@pytest.mark.parametrize("channel", ["scattering", "pair"])
def test_public_return_loss_and_unit_contracts_are_exact(
    formal_c5: FormalC5,
    channel: str,
) -> None:
    arrays = formal_c5.arrays
    f = arrays["parent_f"]
    _assert_bit_exact(
        arrays[f"qpsim_qp_{channel}_loss_ns_inv"],
        arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"] * f,
    )
    _assert_bit_exact(
        arrays[f"qpsim_qp_{channel}_net_ns_inv"],
        arrays[f"qpsim_qp_{channel}_gain_ns_inv"] - arrays[f"qpsim_qp_{channel}_loss_ns_inv"],
    )
    for field in ("gain", "loss_rate", "loss", "net"):
        _assert_bit_exact(
            arrays[f"qpsim_qp_{channel}_{field}_s_inv"],
            arrays[f"qpsim_qp_{channel}_{field}_ns_inv"] / SECONDS_PER_NS,
        )
    assert not np.array_equal(
        arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"],
        arrays[f"qpsim_qp_{channel}_loss_ns_inv"],
    )


def test_c5_residual_reconstructions_and_unchanged_phonon_residual(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    _assert_bit_exact(
        arrays["c5s_qp_residual_s_inv"],
        arrays["parent_qp_residual_s_inv"] + arrays["qp_scattering_delta_net_s_inv"],
    )
    _assert_bit_exact(
        arrays["c5p_qp_residual_s_inv"],
        arrays["parent_qp_residual_s_inv"] + arrays["qp_pair_delta_net_s_inv"],
    )
    _assert_bit_exact(
        arrays["c5sp_qp_residual_s_inv"],
        arrays["parent_qp_residual_s_inv"]
        + arrays["qp_scattering_delta_net_s_inv"]
        + arrays["qp_pair_delta_net_s_inv"],
    )
    _assert_bit_exact(
        arrays["c5sp_phonon_residual_s_inv"],
        arrays["parent_phonon_residual_s_inv"],
    )


def test_scattering_pauli_cross_term_exactly_rebuckets_author_channels(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    f = arrays["parent_f"]
    weights = arrays["parent_cell_weights_ueV"]
    kernel = arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"]
    n_diff = arrays["parent_projected_n_phonon"][arrays["qpsim_omega_idx_diff"]]
    expected_cross = f * ((kernel * n_diff).T @ (weights * f)) / SECONDS_PER_NS
    _assert_within_rounding_bound(
        arrays["scattering_pauli_cross_term_s_inv"],
        expected_cross,
    )
    retained_cross = arrays["scattering_pauli_cross_term_s_inv"]
    _assert_bit_exact(
        arrays["parent_qp_scattering_rebucketed_gain_s_inv"],
        arrays["parent_qp_scattering_gain_s_inv"] - retained_cross,
    )
    _assert_bit_exact(
        arrays["parent_qp_scattering_rebucketed_loss_s_inv"],
        arrays["parent_qp_scattering_loss_s_inv"] - retained_cross,
    )
    _assert_bit_exact(
        arrays["qp_scattering_rebucketed_delta_gain_s_inv"],
        arrays["qpsim_qp_scattering_gain_s_inv"]
        - arrays["parent_qp_scattering_rebucketed_gain_s_inv"],
    )
    _assert_bit_exact(
        arrays["qp_scattering_rebucketed_delta_loss_s_inv"],
        arrays["qpsim_qp_scattering_loss_s_inv"]
        - arrays["parent_qp_scattering_rebucketed_loss_s_inv"],
    )
    for field in ("gain", "loss", "net"):
        _assert_bit_exact(
            arrays[f"qp_scattering_delta_{field}_s_inv"],
            arrays[f"qpsim_qp_scattering_{field}_s_inv"]
            - arrays[f"parent_qp_scattering_{field}_s_inv"],
        )
    assert float(np.sum(retained_cross[ACTIVE])) > 0.0


def test_scattering_physical_net_matches_author_despite_bucket_difference(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    raw_gain_error = _symmetric_relative_l1(
        arrays["qpsim_qp_scattering_gain_s_inv"],
        arrays["parent_qp_scattering_gain_s_inv"],
    )
    rebucketed_gain_error = _symmetric_relative_l1(
        arrays["qpsim_qp_scattering_gain_s_inv"],
        arrays["parent_qp_scattering_rebucketed_gain_s_inv"],
    )
    net_error = _symmetric_relative_l1(
        arrays["qpsim_qp_scattering_net_s_inv"],
        arrays["parent_qp_scattering_net_s_inv"],
    )
    assert raw_gain_error > 1.0e-8
    assert rebucketed_gain_error < 1.0e-12
    assert net_error < 1.0e-12
    assert np.max(np.abs(arrays["qp_scattering_delta_net_s_inv"])) > 0.0


def test_pair_channel_uses_factor_one_per_qp_normalization(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    f = arrays["parent_f"]
    one_minus = 1.0 - f
    weights = arrays["parent_cell_weights_ueV"]
    kernel = arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"]
    expected_loss_rate = (kernel * arrays["qpsim_N_emit"]) @ (weights * f)
    expected_gain = one_minus * ((kernel * arrays["qpsim_N_abs"]) @ (weights * one_minus))
    unsupported = ~arrays["parent_active_mask"]
    expected_loss_rate[unsupported] = 0.0
    expected_gain[unsupported] = 0.0
    _assert_within_rounding_bound(
        arrays["qpsim_qp_pair_loss_rate_ns_inv"],
        expected_loss_rate,
    )
    _assert_within_rounding_bound(
        arrays["qpsim_qp_pair_gain_ns_inv"],
        expected_gain,
    )
    assert not np.array_equal(
        arrays["qpsim_qp_pair_loss_rate_ns_inv"],
        2.0 * expected_loss_rate,
    )


def test_only_scattering_conserves_weighted_qp_number(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    weights = arrays["parent_cell_weights_ueV"]
    scattering_net = arrays["qpsim_qp_scattering_net_s_inv"]
    scattering_turnover = float(
        weights
        @ (arrays["qpsim_qp_scattering_gain_s_inv"] + arrays["qpsim_qp_scattering_loss_s_inv"])
    )
    scattering_error = abs(float(weights @ scattering_net)) / scattering_turnover
    assert scattering_turnover > 0.0
    assert scattering_error < 1.0e-12

    pair_net = arrays["qpsim_qp_pair_net_s_inv"]
    pair_turnover = float(
        weights @ (arrays["qpsim_qp_pair_gain_s_inv"] + arrays["qpsim_qp_pair_loss_s_inv"])
    )
    pair_number_change = float(weights @ pair_net)
    assert pair_turnover > 0.0
    assert abs(pair_number_change) / pair_turnover > 1.0e-3


def test_pair_weighted_qp_change_obeys_two_qps_per_event_identity(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    f = arrays["parent_f"]
    one_minus = 1.0 - f
    weights = arrays["parent_cell_weights_ueV"]
    kernel_s = arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"] / SECONDS_PER_NS
    generation_integrand = (
        weights[:, None]
        * weights[None, :]
        * kernel_s
        * arrays["qpsim_N_abs"]
        * one_minus[:, None]
        * one_minus[None, :]
    )
    recombination_integrand = (
        weights[:, None]
        * weights[None, :]
        * kernel_s
        * arrays["qpsim_N_emit"]
        * f[:, None]
        * f[None, :]
    )
    pair_generation_events = 0.5 * math.fsum(generation_integrand.ravel())
    pair_recombination_events = 0.5 * math.fsum(recombination_integrand.ravel())
    qp_generation = math.fsum((weights * arrays["qpsim_qp_pair_gain_s_inv"]).tolist())
    qp_loss = math.fsum((weights * arrays["qpsim_qp_pair_loss_s_inv"]).tolist())
    assert qp_generation == pytest.approx(
        2.0 * pair_generation_events,
        rel=2.0e-12,
    )
    assert qp_loss == pytest.approx(
        2.0 * pair_recombination_events,
        rel=2.0e-12,
    )
    assert qp_generation - qp_loss != pytest.approx(0.0, abs=0.0)


def test_guards_support_and_nonvacuous_operator_contracts(
    formal_c5: FormalC5,
) -> None:
    arrays = formal_c5.arrays
    assert not np.any(arrays["parent_active_mask"][:GUARD_COUNT])
    assert np.all(arrays["parent_active_mask"][ACTIVE])
    for channel in ("scattering", "pair"):
        for field in ("gain", "loss_rate", "loss", "net"):
            _assert_positive_zero(arrays[f"qpsim_qp_{channel}_{field}_ns_inv"][:GUARD_COUNT])
            _assert_positive_zero(arrays[f"qpsim_qp_{channel}_{field}_s_inv"][:GUARD_COUNT])
        assert float(np.sum(np.abs(arrays[f"qpsim_qp_{channel}_net_s_inv"][ACTIVE]))) > 0.0
    assert np.all(arrays["parent_f"][ACTIVE] > 0.0)
    assert np.ptp(arrays["parent_f"][ACTIVE]) > 0.0
    assert np.any(arrays["parent_projected_n_phonon"] > 0.0)
    expected_phonon_support = np.zeros(N_OMEGA, dtype=bool)
    expected_phonon_support[1 : AUTHOR_OMEGA_STOP + 1] = True
    _assert_bit_exact(
        arrays["parent_legacy_phonon_support_mask"],
        expected_phonon_support,
    )
    _assert_positive_zero(arrays["parent_projected_n_phonon"][~expected_phonon_support])


def test_nonunit_width_microgrid_uses_integrated_partner_weights() -> None:
    energy = np.array([1.3, 1.9, 2.8], dtype=float)
    widths = np.array([0.35, 0.65, 1.15], dtype=float)
    ctx = SpectralContext(energy, widths, gap=1.0)
    f = np.array([0.08, 0.17, 0.31], dtype=float)
    kernel = np.array(
        [
            [0.0, 1.1, 0.7],
            [0.9, 0.0, 1.3],
            [0.4, 1.6, 0.0],
        ],
        dtype=float,
    )
    n_p = np.array(
        [
            [0.0, 0.2, 0.4],
            [1.2, 0.0, 0.6],
            [1.4, 1.6, 0.0],
        ],
        dtype=float,
    )
    gain, loss_rate = phonon_collision_rates(
        f,
        ctx,
        kernel,
        None,
        0.0,
        enable_scattering=True,
        enable_recombination=False,
        N_p_override=n_p,
    )
    expected_gain = (1.0 - f) * ((kernel * n_p).T @ (ctx.cell_weights * f))
    expected_loss_rate = (kernel * n_p) @ (ctx.cell_weights * (1.0 - f))
    density_gain = (1.0 - f) * ((kernel * n_p).T @ (ctx.cell_density * f))
    _assert_bit_exact(gain, expected_gain)
    _assert_bit_exact(loss_rate, expected_loss_rate)
    assert not np.allclose(gain, density_gain, rtol=1.0e-12, atol=0.0)

    n_emit = np.array(
        [
            [1.1, 1.2, 1.3],
            [1.2, 1.4, 1.5],
            [1.3, 1.5, 1.7],
        ],
        dtype=float,
    )
    n_abs = n_emit - 1.0
    pair_gain, pair_loss_rate = phonon_collision_rates(
        f,
        ctx,
        None,
        kernel,
        0.0,
        enable_scattering=False,
        enable_recombination=True,
        N_emit_override=n_emit,
        N_abs_override=n_abs,
    )
    expected_pair_gain = (1.0 - f) * ((kernel * n_abs) @ (ctx.cell_weights * (1.0 - f)))
    expected_pair_loss_rate = (kernel * n_emit) @ (ctx.cell_weights * f)
    density_pair_gain = (1.0 - f) * ((kernel * n_abs) @ (ctx.cell_density * (1.0 - f)))
    _assert_bit_exact(pair_gain, expected_pair_gain)
    _assert_bit_exact(pair_loss_rate, expected_pair_loss_rate)
    assert not np.allclose(
        pair_gain,
        density_pair_gain,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_c5_metadata_makes_only_a_frozen_operator_differential_claim(
    formal_c5: FormalC5,
) -> None:
    metadata = formal_c5.metadata
    assert metadata["stage"] == {
        "changed_component": "qp_phonon_operator",
        "comparison_stage_id": "C4",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C4",
        "stage_id": "C5",
    }
    statement = metadata["limitations"]["statement"]
    for excluded in (
        "No C5 nonlinear root",
        "Newton",
        "stopping",
        "observable",
        "curve",
        "paper",
    ):
        assert excluded.lower() in statement.lower()
    assert metadata["component_locality"]["phonon_residual_bit_exact"] is True
    forbidden = {
        "c6",
        "converged",
        "iteration_count",
        "newton_history",
        "observable",
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
    assert "compute_phonon_source_sink" not in c5_bundle.__dict__
    assert metadata["limitations"]["scope"] == ("one authenticated C4 frozen point only")
    assert metadata["units"] == {
        "comparison_arrays": "per second",
        "kernel_arrays": "per nanosecond per microelectronvolt",
        "public_native_arrays": "per nanosecond",
        "public_return_contract": (
            "gain includes target Pauli factor; loss_rate multiplies f to form physical loss"
        ),
    }
    assert set(metadata["bookkeeping_contract"]) == {
        "pair_buckets",
        "pair_weighted_number",
        "scattering_cross_term",
        "scattering_pauli_cross_term_formula",
        "scattering_rebucketed_controls",
        "scattering_weighted_number_conservation",
        "warning",
    }
    assert metadata["bookkeeping_contract"]["scattering_pauli_cross_term_formula"] == (
        "f * ((K_s0 * n_ph[omega_idx_diff]).T @ (cell_weights * f))"
    )
    assert set(metadata["comparison_contract"]) == {
        "candidate",
        "loss_comparison",
        "parent",
        "parent_photon",
        "public_arithmetic",
    }
    assert "evaluated separately" in metadata["comparison_contract"]["candidate"]
    assert "copied bit-exact" in metadata["comparison_contract"]["parent_photon"]
    assert metadata["coordinate_contract"] == {
        "active_child_indices": "[20, 1640)",
        "frequency_map": (
            "public build_phonon_frequency_map on the accepted C3 1640-cell center grid"
        ),
        "guard_child_indices": "[0, 20), canonical positive zero",
        "legacy_phonon_support": (
            "omega indices [1, 1620); every other projected n_ph entry is canonical zero"
        ),
        "native_cell_count": N_QP,
        "native_omega_count": N_OMEGA,
        "phonon_projection": (
            "public phonon_occupation_matrices_from_state on the frozen C3 projected_n_phonon"
        ),
    }
    assert metadata["stage"]["stage_id"] != "C6"
    assert all("c6" not in name.lower() for name in formal_c5.arrays)


def test_write_is_atomic_exclusive_and_closes_over_exact_arrays(
    formal_c5: FormalC5,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "formal-c5"
    monkeypatch.setattr(
        c5_bundle,
        "build_c5_bundle",
        lambda *_args, **_kwargs: (formal_c5.metadata, formal_c5.arrays),
    )
    manifest_path = c5_bundle.write_c5_bundle(
        C4_BUNDLE,
        output,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert manifest_path == output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == c5_bundle.SCHEMA
    assert manifest["metadata"]["schema"] == c5_bundle.SCHEMA
    expected_files = {f"{name}.npy" for name in EXPECTED_ARRAY_NAMES}
    assert set(manifest["files"]) == expected_files
    assert {path.name for path in output.iterdir()} == {
        "manifest.json",
        *expected_files,
    }
    for name, expected in formal_c5.arrays.items():
        path = output / f"{name}.npy"
        content = path.read_bytes()
        assert manifest["files"][path.name] == {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
        }
        with path.open("rb") as handle:
            _assert_bit_exact(np.load(handle, allow_pickle=False), expected)
    with pytest.raises(FileExistsError, match="C5 output already exists"):
        c5_bundle.write_c5_bundle(
            C4_BUNDLE,
            output,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_interrupted_write_removes_partial_temporary_bundle(
    formal_c5: FormalC5,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "interrupted-c5"
    monkeypatch.setattr(
        c5_bundle,
        "build_c5_bundle",
        lambda *_args, **_kwargs: (formal_c5.metadata, formal_c5.arrays),
    )
    original_encoder = c5_bundle._npy_bytes
    calls = 0

    def interrupted_encoder(value: np.ndarray) -> bytes:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated C5 write interruption")
        return original_encoder(value)

    monkeypatch.setattr(c5_bundle, "_npy_bytes", interrupted_encoder)
    with pytest.raises(OSError, match="simulated C5 write interruption"):
        c5_bundle.write_c5_bundle(
            C4_BUNDLE,
            output,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )
    assert not output.exists()
    assert list(tmp_path.glob(".interrupted-c5.*.tmp")) == []


def test_source_drift_fails_before_c5_evidence_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_formal_parents()
    relative = "qpsim/collisions/phonon.py"
    target = REPOSITORY_ROOT / relative
    original = c5_bundle.canonical_source_bytes

    def drifted(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == target.resolve():
            return content + b"\n# simulated source drift\n"
        return content

    assert relative in c5_bundle._SOURCE_BYTES_AT_IMPORT
    monkeypatch.setattr(c5_bundle, "canonical_source_bytes", drifted)
    with pytest.raises(
        c5_bundle.C5BundleError,
        match=r"C5 numerical source changed during execution: qpsim/collisions/phonon\.py",
    ):
        c5_bundle.build_c5_bundle(
            C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )
