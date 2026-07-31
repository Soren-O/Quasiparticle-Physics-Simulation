"""Focused producer tests for the formal Figure 6 C7 re-solve bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from qpsim.observables.gap_suppression import gap_from_distribution_direct
from validation.fischer_2023 import fig6_author_c7_bundle as c7_bundle
from validation.fischer_2023.fig6_author_c6_score import (
    DEFAULT_RECEIPT as C6_RECEIPT,
)
from validation.fischer_2023.fig6_author_c6_score import DEFAULT_SCORE as C6_SCORE
from validation.fischer_2023.fig6_author_c6_score import load_c6_raw_bundle
from validation.fischer_2023.fig6_author_c7_score import load_c7_raw_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_RUNS = REPOSITORY_ROOT / "tmp" / "author-runs"
C0_BUNDLE = _RUNS / "fig6-T020-sweep049-C0-author-equivalent-v1"
C6_BUNDLE = _RUNS / "fig6-T020-sweep049-C6-phonon-balance-v1"
C7_BUNDLE = _RUNS / "fig6-T020-sweep049-C7-solver-v1"

N_QP = 1640
N_OMEGA = 3600
N_AUTHOR = 1620
GUARD = 20
SECONDS_PER_NS = 1.0e-9


def _require_c7() -> None:
    required = (
        C0_BUNDLE / "manifest.json",
        C6_BUNDLE / "manifest.json",
        C7_BUNDLE / "manifest.json",
        C6_SCORE,
        C6_RECEIPT,
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Formal C7 evidence or its parents are unavailable.")


@pytest.fixture(scope="module")
def formal_c7() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    _require_c7()
    metadata, arrays, _sha = load_c7_raw_bundle(C7_BUNDLE)
    return metadata, arrays


def test_c7_has_exact_forty_four_array_closure(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, arrays = formal_c7
    assert set(arrays) == set(c7_bundle.ARRAY_NAMES)
    assert len(arrays) == 44
    assert set(metadata["array_descriptors"]) == set(c7_bundle.ARRAY_NAMES)


def test_c7_seed_is_the_authenticated_a1_continuation_state(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, arrays = formal_c7
    seed = arrays["author_seed_state"]
    c0_score = json.loads(
        (
            REPOSITORY_ROOT
            / "validation/paper_data/fischer_2023/fig6/c0-author-equivalent-score.json"
        ).read_text(encoding="utf-8")
    )
    descriptor = c0_score["array_descriptors"]["initial_state"]
    stream = (C0_BUNDLE / "initial_state.npy").read_bytes()
    assert hashlib.sha256(stream).hexdigest() == descriptor["npy_sha256"]
    assert metadata["parent_bindings"]["c0_raw_manifest_sha256"] == (
        c0_score["raw_bundle"]["manifest_sha256"]
    )
    assert np.array_equal(arrays["seed_f"][GUARD:], seed[:N_AUTHOR])
    assert not np.any(arrays["seed_f"][:GUARD] != 0.0)
    assert np.array_equal(arrays["seed_n_ph"][1:N_AUTHOR], seed[N_AUTHOR:])
    assert not np.any(arrays["seed_n_ph"][N_AUTHOR:] != 0.0)
    assert arrays["seed_n_ph"][0] == 0.0


def test_c7_root_satisfies_declared_certificates(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, arrays = formal_c7
    root_f = arrays["qpsim_root_f"]
    root_n = arrays["qpsim_root_n_ph"]
    assert np.all((root_f >= 0.0) & (root_f <= 1.0))
    assert np.all(root_n >= 0.0)
    R_f = arrays["qpsim_root_R_f_ns_inv"]
    expected_R_f = (
        arrays["qpsim_root_qp_gain_ns_inv"]
        - arrays["qpsim_root_qp_loss_rate_ns_inv"] * root_f
    )
    assert np.array_equal(np.where(expected_R_f == 0.0, 0.0, expected_R_f), R_f)
    certificates = metadata["solver_contract"]["balance_certificates"]
    qp_scale = np.abs(arrays["qpsim_root_qp_gain_ns_inv"]) + np.abs(
        arrays["qpsim_root_qp_loss_rate_ns_inv"] * root_f
    )
    qp_ratio = float(np.sum(np.abs(R_f))) / float(np.sum(qp_scale))
    assert qp_ratio < 1.0e-7
    assert certificates["qp_l1_ratio"]["value"] == qp_ratio
    assert certificates["ph_l1_ratio"]["value"] < 1.0e-7


def test_c7_iteration_probe_is_publicly_measured(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, _arrays = formal_c7
    contract = metadata["solver_contract"]
    probe = contract["iteration_probe"]
    accepted = contract["accepted_max_iter"]
    assert len(probe) == accepted
    assert [entry["max_iter"] for entry in probe] == list(range(1, accepted + 1))
    assert all(entry["converged"] is False for entry in probe[:-1])
    assert probe[-1]["converged"] is True
    assert probe[-1]["exception"] is None
    assert contract["declared_max_iter"] == 10
    assert contract["analytic_cross"] is True
    assert contract["step_rtol"]["value"] == 1.0e-7


def test_c7_frozen_assembly_binds_the_accepted_c6_residuals(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    _metadata, arrays = formal_c7
    _c6_metadata, c6_arrays, _sha = load_c6_raw_bundle(C6_BUNDLE)
    assert np.array_equal(
        arrays["parent_qp_residual_s_inv"],
        c6_arrays["c6_qp_residual_s_inv"],
    )
    assert np.array_equal(
        arrays["parent_phonon_residual_s_inv"],
        c6_arrays["c6spe_phonon_residual_s_inv"],
    )
    idx = arrays["parent_phonon_to_native_omega_index"]
    for mine, parent, limit in (
        (
            arrays["qpsim_frozen_R_f_ns_inv"] / SECONDS_PER_NS,
            arrays["parent_qp_residual_s_inv"],
            1.0e-12,
        ),
        (
            arrays["qpsim_frozen_R_ph_ns_inv"][idx] / SECONDS_PER_NS,
            arrays["parent_phonon_residual_s_inv"],
            1.0e-12,
        ),
    ):
        delta = float(np.sum(np.abs(mine - parent)))
        denominator = float(np.sum(np.abs(mine)) + np.sum(np.abs(parent)))
        assert delta / denominator <= limit


def test_c7_observable_records_are_reproducible(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, arrays = formal_c7
    observable = metadata["observable"]
    gap_eV = 0.00017999999999999998
    h_eV = 1.0e-6
    carrier = gap_eV + h_eV * np.arange(N_AUTHOR) + 0.5 * h_eV
    driven = gap_from_distribution_direct(
        arrays["qpsim_root_f"][GUARD:],
        carrier,
        gap=gap_eV,
        delta0=gap_eV,
        samples="authors",
    )
    thermal = gap_from_distribution_direct(
        arrays["parent_thermal_f"][GUARD:],
        carrier,
        gap=gap_eV,
        delta0=gap_eV,
        samples="authors",
    )
    assert observable["root_driven_gap_eV_authors"]["value"] == driven
    assert observable["thermal_gap_eV_authors"]["value"] == thermal
    ordinate = (driven - thermal) / (gap_eV - thermal)
    assert observable["root_ordinate_authors"]["value"] == ordinate
    assert observable["frozen_root_ordinate_authors"]["value"] == (
        0.12090908988993258
    )
    assert observable["author_control_ordinate"]["value"] == 0.12090908988993258
    assert 0.0 < observable["root_ordinate_centers"]["value"] < ordinate


def test_c7_metadata_makes_only_a_single_point_resolve_claim(
    formal_c7: tuple[dict[str, Any], dict[str, np.ndarray]],
) -> None:
    metadata, _arrays = formal_c7
    assert metadata["stage"] == {
        "changed_component": "nonlinear_solver",
        "comparison_stage_id": "C6",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C6",
        "stage_id": "C7",
    }
    statement = metadata["limitations"]["statement"]
    for phrase in ("300-point curve", "paper-parity", "author-iteration-equivalence"):
        assert phrase in statement


def test_write_refuses_existing_output(tmp_path: Path) -> None:
    _require_c7()
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        c7_bundle.write_c7_bundle(
            C6_BUNDLE,
            target,
            c5_bundle_dir=_RUNS / "fig6-T020-sweep049-C5-qp-phonon-regen-v1",
            c4_bundle_dir=_RUNS / "fig6-T020-sweep049-C4-photon-regen-v1",
            c3_bundle_dir=_RUNS / "fig6-T020-sweep049-C3-grid-regen-v1",
            c2_bundle_dir=_RUNS / "fig6-T020-sweep049-C2-parameters-v1",
            c0_bundle_dir=C0_BUNDLE,
        )


def test_source_drift_fails_before_c7_evidence_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    drifted = dict(c7_bundle._SOURCE_HASHES_AT_IMPORT)
    key = next(iter(drifted))
    drifted[key] = "0" * 64
    monkeypatch.setattr(c7_bundle, "_SOURCE_HASHES_AT_IMPORT", drifted)
    with pytest.raises(c7_bundle.C7BundleError):
        c7_bundle._assert_source_snapshots()
