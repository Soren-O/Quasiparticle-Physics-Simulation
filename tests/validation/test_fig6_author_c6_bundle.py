"""Focused producer tests for the formal Figure 6 C6 phonon-balance bundle."""

from __future__ import annotations

import hashlib
import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qpsim.collisions.phonon import (
    _pair_breaking_quadrature_correction,
    build_phonon_frequency_map,
    build_recombination_kernel_phonon_side,
    build_scattering_kernel_phonon_side,
    compute_phonon_source_sink,
)
from qpsim.phonon_models.ph0_local import phonon_balance_diagnostics
from qpsim.physics.kernels import thermal_phonon_occupation
from qpsim.physics.spectral import SpectralContext, fermi_dirac_occupation
from validation.fischer_2023 import fig6_author_c6_bundle as c6_bundle
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
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT as C5_RECEIPT,
)
from validation.fischer_2023.fig6_author_c5_score import DEFAULT_SCORE as C5_SCORE
from validation.fischer_2023.fig6_author_c5_score import (
    RAW_SCHEMA as C5_RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_c5_score import (
    load_c5_raw_bundle,
    load_c5_score,
)
from validation.fischer_2023.fig6_author_c6_score import load_c6_raw_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNS = REPOSITORY_ROOT / "tmp" / "author-runs"
C2_BUNDLE = _RUNS / "fig6-T020-sweep049-C2-parameters-v1"
C3_BUNDLE = _RUNS / "fig6-T020-sweep049-C3-grid-regen-v1"
C4_BUNDLE = _RUNS / "fig6-T020-sweep049-C4-photon-regen-v1"
C5_BUNDLE = _RUNS / "fig6-T020-sweep049-C5-qp-phonon-regen-v1"
C6_BUNDLE = _RUNS / "fig6-T020-sweep049-C6-phonon-balance-v1"

N_QP = 1640
N_OMEGA = 3600
GUARD_COUNT = 20
AUTHOR_OMEGA_STOP = 1619
SECONDS_PER_NS = 1.0e-9
TAU_PB_NS = 0.255
TAU_L_NS = 0.255
T_BATH_K = 0.2
CHANNELS = ("scattering", "pair", "pair_control", "escape")
FIELDS = ("gain", "loss", "net")


@dataclass
class FormalC6:
    metadata: dict[str, Any]
    arrays: dict[str, np.ndarray]
    c5_metadata: dict[str, Any]
    c5_arrays: dict[str, np.ndarray]
    c5_manifest_sha256: str
    c3_metadata: dict[str, Any]
    c3_arrays: dict[str, np.ndarray]
    c3_manifest_sha256: str


def _require_formal_parents() -> None:
    required = (
        C2_BUNDLE / "manifest.json",
        C3_BUNDLE / "manifest.json",
        C4_BUNDLE / "manifest.json",
        C5_BUNDLE / "manifest.json",
        C6_BUNDLE / "manifest.json",
        C2_SCORE,
        C2_RECEIPT,
        C3_SCORE,
        C3_RECEIPT,
        C4_SCORE,
        C4_RECEIPT,
        C5_SCORE,
        C5_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Canonical C2/C3/C4/C5/C6 evidence needed by formal C6 is unavailable.")


@pytest.fixture(scope="module")
def formal_c6() -> FormalC6:
    _require_formal_parents()
    metadata, arrays, _c6_sha = load_c6_raw_bundle(C6_BUNDLE)
    c5_metadata, c5_arrays, c5_sha = load_c5_raw_bundle(C5_BUNDLE)
    c3_metadata, c3_arrays, c3_sha = load_c3_raw_bundle(C3_BUNDLE)
    return FormalC6(
        metadata,
        arrays,
        c5_metadata,
        c5_arrays,
        c5_sha,
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


def _assert_bit_exact(actual: np.ndarray, expected: np.ndarray) -> None:
    left = np.asarray(actual)
    right = np.asarray(expected)
    assert left.dtype.str == right.dtype.str
    assert left.shape == right.shape
    assert left.tobytes(order="C") == right.tobytes(order="C")


def _positive_zero(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    if result.dtype.kind == "f":
        result[result == 0.0] = 0.0
    return result


def _context(formal: FormalC6) -> SpectralContext:
    inputs = formal.metadata["operator_inputs"]
    return SpectralContext(
        formal.arrays["parent_E_centers_ueV"],
        formal.arrays["parent_dE_ueV"],
        inputs["gap_ueV"]["value"],
    )


def test_c6_exactly_replays_the_accepted_c5_chain(formal_c6: FormalC6) -> None:
    accepted = load_c5_score(C5_SCORE, receipt_path=C5_RECEIPT)
    raw_binding = accepted["raw_bundle"]
    assert raw_binding == {
        "manifest_sha256": formal_c6.c5_manifest_sha256,
        "schema": C5_RAW_SCHEMA,
    }
    bindings = formal_c6.metadata["parent_bindings"]
    assert bindings["c5_raw_manifest_sha256"] == formal_c6.c5_manifest_sha256
    assert bindings["c3_raw_manifest_sha256"] == formal_c6.c3_manifest_sha256
    assert bindings["c5_stage_id"] == "C5"
    assert bindings["c3_operator_stage_id"] == "c3c_native_cell_density"
    assert (
        bindings["c5_score_sha256"]
        == hashlib.sha256(C5_SCORE.read_bytes()).hexdigest()
    )
    assert (
        bindings["c5_receipt_sha256"]
        == hashlib.sha256(C5_RECEIPT.read_bytes()).hexdigest()
    )


def test_c6_has_exact_eighty_six_array_closure_and_descriptors(
    formal_c6: FormalC6,
) -> None:
    assert set(formal_c6.arrays) == set(c6_bundle.ARRAY_NAMES)
    assert len(formal_c6.arrays) == 86
    descriptors = formal_c6.metadata["array_descriptors"]
    assert set(descriptors) == set(c6_bundle.ARRAY_NAMES)
    for name, value in formal_c6.arrays.items():
        assert descriptors[name] == {
            "dtype": value.dtype.str,
            "npy_sha256": hashlib.sha256(_npy_bytes(value)).hexdigest(),
            "shape": list(value.shape),
        }


def test_c6_freezes_exact_parent_state_channels_and_residuals(
    formal_c6: FormalC6,
) -> None:
    c3 = formal_c6.c3_arrays
    c5 = formal_c6.c5_arrays
    arrays = formal_c6.arrays
    _assert_bit_exact(arrays["parent_f"], _positive_zero(c3["projected_f"]))
    _assert_bit_exact(
        arrays["parent_projected_n_phonon"],
        _positive_zero(c3["projected_n_phonon"]),
    )
    _assert_bit_exact(
        arrays["parent_cell_density"],
        _positive_zero(c3["native_cell_density_full"]),
    )
    _assert_bit_exact(
        arrays["parent_qp_residual_s_inv"],
        _positive_zero(c5["c5sp_qp_residual_s_inv"]),
    )
    _assert_bit_exact(
        arrays["parent_phonon_residual_s_inv"],
        _positive_zero(c5["c5sp_phonon_residual_s_inv"]),
    )
    _assert_bit_exact(
        arrays["parent_phonon_residual_s_inv"],
        _positive_zero(c3["c3c_native_cell_density__phonon_residual_s_inv"]),
    )
    for channel in ("scattering", "pair", "escape"):
        for field in FIELDS:
            _assert_bit_exact(
                arrays[f"parent_phonon_{channel}_{field}_s_inv"],
                _positive_zero(
                    c3[
                        "c3c_native_cell_density__phonon_"
                        f"{channel}__{field}_s_inv"
                    ]
                ),
            )
    assert np.array_equal(
        arrays["parent_phonon_to_native_omega_index"],
        np.arange(1, AUTHOR_OMEGA_STOP + 1, dtype=np.int64),
    )
    _assert_bit_exact(
        arrays["c6_qp_residual_s_inv"],
        arrays["parent_qp_residual_s_inv"],
    )
    assert formal_c6.metadata["component_locality"]["qp_residual_bit_exact"] is True


def test_c6_operator_inputs_are_exact_inherited_values(
    formal_c6: FormalC6,
) -> None:
    inputs = formal_c6.metadata["operator_inputs"]
    values = formal_c6.c3_metadata["parameters"]["values"]
    assert inputs["tau_0_pb_ns"]["value"] == TAU_PB_NS
    assert inputs["tau_l_ns"]["value"] == TAU_L_NS
    assert inputs["tau_0_pb_parent_s"]["value"] == values["tau_0_pb_s"]
    assert inputs["tau_l_parent_s"]["value"] == values["tau_l_s"]
    assert inputs["T_bath_K"]["value"] == T_BATH_K
    assert inputs["gap_ueV"]["value"] == 180.0
    assert inputs["tau_0_pb_ns"]["hex"] == (0.255).hex()
    kb = (
        values["boltzmann_constant_J_per_K"]
        / values["electron_charge_C"]
        * 1.0e6
    )
    assert inputs["kB_ueV_per_K"]["value"] == kb


def test_public_channels_reproduce_with_only_frozen_overrides(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays
    ctx = _context(formal_c6)
    omega, idx_diff, idx_sum, diff_sign = build_phonon_frequency_map(ctx.E)
    _assert_bit_exact(arrays["qpsim_omega_ueV"], _positive_zero(omega))
    _assert_bit_exact(arrays["qpsim_omega_idx_diff"], idx_diff)
    _assert_bit_exact(arrays["qpsim_omega_idx_sum"], idx_sum)
    _assert_bit_exact(arrays["qpsim_diff_sign"], diff_sign)
    k_s = build_scattering_kernel_phonon_side(ctx, TAU_PB_NS)
    k_r = build_recombination_kernel_phonon_side(ctx, TAU_PB_NS)
    _assert_bit_exact(
        arrays["qpsim_phonon_scattering_kernel_ns_inv_ueV_inv"],
        _positive_zero(k_s),
    )
    _assert_bit_exact(
        arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"],
        _positive_zero(k_r),
    )
    f = arrays["parent_f"]
    a_s, b_s = compute_phonon_source_sink(
        f,
        ctx,
        None,
        None,
        idx_diff,
        idx_sum,
        diff_sign,
        omega.size,
        enable_scattering=True,
        enable_recombination=False,
        K_s0_phonon_side=k_s,
    )
    _assert_bit_exact(arrays["qpsim_scattering_a_ns_inv"], _positive_zero(a_s))
    _assert_bit_exact(arrays["qpsim_scattering_b_ns_inv"], _positive_zero(b_s))
    a_p, b_p = compute_phonon_source_sink(
        f,
        ctx,
        None,
        None,
        idx_diff,
        idx_sum,
        diff_sign,
        omega.size,
        enable_scattering=False,
        enable_recombination=True,
        K_r0_phonon_side=k_r,
    )
    _assert_bit_exact(arrays["qpsim_pair_a_ns_inv"], _positive_zero(a_p))
    _assert_bit_exact(arrays["qpsim_pair_b_ns_inv"], _positive_zero(b_p))
    a_p0, b_p0 = compute_phonon_source_sink(
        f,
        ctx,
        None,
        k_r,
        idx_diff,
        idx_sum,
        diff_sign,
        omega.size,
        enable_scattering=False,
        enable_recombination=True,
    )
    _assert_bit_exact(
        arrays["qpsim_pair_control_a_ns_inv"],
        _positive_zero(a_p0),
    )
    _assert_bit_exact(
        arrays["qpsim_pair_control_b_ns_inv"],
        _positive_zero(b_p0),
    )
    _assert_bit_exact(
        arrays["qpsim_thermal_n_ph"],
        _positive_zero(thermal_phonon_occupation(omega, T_BATH_K)),
    )
    _assert_bit_exact(
        arrays["qpsim_db_f"],
        _positive_zero(fermi_dirac_occupation(ctx.E, T_BATH_K)),
    )


def test_kaplan_correction_is_the_only_pair_difference(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays
    ctx = _context(formal_c6)
    correction = _pair_breaking_quadrature_correction(
        ctx,
        arrays["qpsim_phonon_pair_kernel_ns_inv_ueV_inv"],
        arrays["qpsim_omega_idx_sum"],
        N_OMEGA,
    )
    _assert_bit_exact(
        arrays["qpsim_pair_a_ns_inv"],
        _positive_zero(arrays["qpsim_pair_control_a_ns_inv"] * correction),
    )
    support = arrays["parent_legacy_phonon_support_mask"]
    on_support = correction[support]
    assert float(np.min(on_support)) > 0.75
    assert float(np.max(on_support)) <= 1.0 + 1e-12
    assert np.any(np.abs(on_support - 1.0) > 1.0e-3)
    control_delta = arrays["phonon_pair_control_delta_net_s_inv"]
    public_delta = arrays["phonon_pair_delta_net_s_inv"]
    parent_net = arrays["parent_phonon_pair_net_s_inv"]
    control_relative = float(np.sum(np.abs(control_delta))) / float(
        np.sum(np.abs(parent_net))
    )
    public_relative = float(np.sum(np.abs(public_delta))) / float(
        np.sum(np.abs(parent_net))
    )
    assert control_relative < 1.0e-12
    assert public_relative > 1.0e-3


def test_escape_channel_follows_ph0_local_balance_form(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays
    n_ph = arrays["parent_projected_n_phonon"]
    n_th = arrays["qpsim_thermal_n_ph"]
    _assert_bit_exact(
        arrays["qpsim_phonon_escape_gain_ns_inv"],
        _positive_zero(n_th / TAU_L_NS),
    )
    _assert_bit_exact(
        arrays["qpsim_phonon_escape_loss_ns_inv"],
        _positive_zero(n_ph / TAU_L_NS),
    )
    _assert_bit_exact(
        arrays["qpsim_phonon_escape_net_ns_inv"],
        _positive_zero((n_th - n_ph) / TAU_L_NS),
    )
    balance = phonon_balance_diagnostics(
        n_ph,
        arrays["qpsim_combined_a_ns_inv"],
        arrays["qpsim_combined_b_ns_inv"],
        n_th,
        tau_l=TAU_L_NS,
    )
    _assert_bit_exact(
        arrays["qpsim_balance_residual_ns_inv"],
        _positive_zero(balance.residual),
    )
    certification = formal_c6.metadata["balance_certification"]
    assert certification["raw_backward_error"]["value"] == balance.raw_backward_error
    assert (
        certification["certified_backward_error"]["value"]
        == balance.certified_backward_error
    )


def test_channel_arrays_are_declared_affine_identities(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays
    n_ph = arrays["parent_projected_n_phonon"]
    affine = {
        "scattering": ("qpsim_scattering_a_ns_inv", "qpsim_scattering_b_ns_inv"),
        "pair": ("qpsim_pair_a_ns_inv", "qpsim_pair_b_ns_inv"),
        "pair_control": (
            "qpsim_pair_control_a_ns_inv",
            "qpsim_pair_control_b_ns_inv",
        ),
    }
    for channel, (a_name, b_name) in affine.items():
        a = arrays[a_name]
        b = arrays[b_name]
        _assert_bit_exact(
            arrays[f"qpsim_phonon_{channel}_gain_ns_inv"],
            _positive_zero(a * (1.0 + n_ph)),
        )
        _assert_bit_exact(
            arrays[f"qpsim_phonon_{channel}_loss_ns_inv"],
            _positive_zero((a - b) * n_ph),
        )
        _assert_bit_exact(
            arrays[f"qpsim_phonon_{channel}_net_ns_inv"],
            _positive_zero(a + b * n_ph),
        )
    for channel in CHANNELS:
        for field in FIELDS:
            _assert_bit_exact(
                arrays[f"qpsim_phonon_{channel}_{field}_s_inv"],
                _positive_zero(
                    arrays[f"qpsim_phonon_{channel}_{field}_ns_inv"]
                    / SECONDS_PER_NS
                ),
            )


def test_c6_residual_reconstructions_are_exact(formal_c6: FormalC6) -> None:
    arrays = formal_c6.arrays
    parent = arrays["parent_phonon_residual_s_inv"]
    d_s = arrays["phonon_scattering_delta_net_s_inv"]
    d_p = arrays["phonon_pair_delta_net_s_inv"]
    d_p0 = arrays["phonon_pair_control_delta_net_s_inv"]
    d_e = arrays["phonon_escape_delta_net_s_inv"]
    _assert_bit_exact(
        arrays["c6s_phonon_residual_s_inv"],
        _positive_zero(parent + d_s),
    )
    _assert_bit_exact(
        arrays["c6p_phonon_residual_s_inv"],
        _positive_zero(parent + d_p),
    )
    _assert_bit_exact(
        arrays["c6p0_phonon_residual_s_inv"],
        _positive_zero(parent + d_p0),
    )
    _assert_bit_exact(
        arrays["c6e_phonon_residual_s_inv"],
        _positive_zero(parent + d_e),
    )
    _assert_bit_exact(
        arrays["c6spe_phonon_residual_s_inv"],
        _positive_zero(parent + d_s + d_p + d_e),
    )
    _assert_bit_exact(
        arrays["c6spe0_phonon_residual_s_inv"],
        _positive_zero(parent + d_s + d_p0 + d_e),
    )
    idx = arrays["parent_phonon_to_native_omega_index"]
    for channel in CHANNELS:
        parent_channel = "pair" if channel == "pair_control" else channel
        for field in FIELDS:
            _assert_bit_exact(
                arrays[f"phonon_{channel}_delta_{field}_s_inv"],
                _positive_zero(
                    arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][idx]
                    - arrays[f"parent_phonon_{parent_channel}_{field}_s_inv"]
                ),
            )


def test_detailed_balance_holds_at_thermal_control(formal_c6: FormalC6) -> None:
    arrays = formal_c6.arrays
    n_th = arrays["qpsim_thermal_n_ph"]
    for channel in ("scattering", "pair"):
        a = arrays[f"qpsim_db_{channel}_a_ns_inv"]
        b = arrays[f"qpsim_db_{channel}_b_ns_inv"]
        net = arrays[f"qpsim_db_{channel}_net_ns_inv"]
        _assert_bit_exact(net, _positive_zero(a + b * n_th))
        gain = a * (1.0 + n_th)
        loss = (a - b) * n_th
        relative = float(np.sum(np.abs(net))) / float(
            np.sum(np.abs(gain) + np.abs(loss))
        )
        assert relative <= 1.0e-12
        recorded = formal_c6.metadata["detailed_balance"][channel]
        assert recorded["relative_imbalance"]["value"] <= 1.0e-12


def test_scattering_confined_to_support_and_omega_zero_zero(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays
    outside = ~arrays["parent_legacy_phonon_support_mask"]
    for field in FIELDS:
        assert not np.any(
            arrays[f"qpsim_phonon_scattering_{field}_s_inv"][outside] != 0.0
        )
    for channel in CHANNELS:
        for field in FIELDS:
            assert float(arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][0]) == 0.0
    extension = formal_c6.metadata["extension_policy"]
    for channel in CHANNELS:
        for field in FIELDS:
            values = arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][outside]
            magnitude = np.abs(values)
            record = extension[channel][field]
            assert record["l1_s_inv"]["value"] == float(np.sum(magnitude))
            assert record["linf_s_inv"]["value"] == float(
                np.max(magnitude, initial=0.0)
            )
            assert record["nonzero_bins"] == int(np.count_nonzero(magnitude))


def test_scattering_and_escape_match_author_channels_to_roundoff(
    formal_c6: FormalC6,
) -> None:
    arrays = formal_c6.arrays

    def symmetric(delta: np.ndarray, parent: np.ndarray, mine: np.ndarray) -> float:
        return float(np.sum(np.abs(delta))) / float(
            np.sum(np.abs(parent)) + np.sum(np.abs(mine))
        )

    idx = arrays["parent_phonon_to_native_omega_index"]
    for channel in ("scattering", "escape"):
        for field in ("gain", "loss"):
            mine = arrays[f"qpsim_phonon_{channel}_{field}_s_inv"][idx]
            parent = arrays[f"parent_phonon_{channel}_{field}_s_inv"]
            assert (
                symmetric(
                    arrays[f"phonon_{channel}_delta_{field}_s_inv"],
                    parent,
                    mine,
                )
                <= 1.0e-12
            )
    scattering_net = symmetric(
        arrays["phonon_scattering_delta_net_s_inv"],
        arrays["parent_phonon_scattering_net_s_inv"],
        arrays["qpsim_phonon_scattering_net_s_inv"][idx],
    )
    assert scattering_net <= 1.0e-12


def test_c6_metadata_makes_only_a_frozen_operator_differential_claim(
    formal_c6: FormalC6,
) -> None:
    stage = formal_c6.metadata["stage"]
    assert stage == {
        "changed_component": "phonon_balance",
        "comparison_stage_id": "C5",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C5",
        "stage_id": "C6",
    }
    statement = formal_c6.metadata["limitations"]["statement"]
    for phrase in (
        "No C6 nonlinear root",
        "plotted ordinate",
        "300-point curve",
        "paper-parity",
    ):
        assert phrase in statement
    assert formal_c6.metadata["limitations"]["scope"] == (
        "one authenticated C5 frozen point only"
    )


def test_write_refuses_existing_output(tmp_path: Path) -> None:
    _require_formal_parents()
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        c6_bundle.write_c6_bundle(
            C5_BUNDLE,
            target,
            c4_bundle_dir=C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_source_drift_fails_before_c6_evidence_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = dict(c6_bundle._SOURCE_HASHES_AT_IMPORT)
    key = next(iter(drifted))
    drifted[key] = "0" * 64
    monkeypatch.setattr(c6_bundle, "_SOURCE_HASHES_AT_IMPORT", drifted)
    with pytest.raises(c6_bundle.C6BundleError):
        c6_bundle._assert_source_snapshots()


def test_reduction_gamma_matches_fixed_budget() -> None:
    budget = 8 * N_QP + 64
    eps = float(np.finfo(np.float64).eps)
    gamma = budget * eps / (1.0 - budget * eps)
    assert math.isfinite(gamma)
    assert 0.0 < gamma < 1.0e-11
