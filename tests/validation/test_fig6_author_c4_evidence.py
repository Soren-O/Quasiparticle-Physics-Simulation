"""Adversarial evidence tests for the formal Figure 6 C4 score and receipt."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from validation.fischer_2023 import fig6_author_c4_score as c4_score
from validation.fischer_2023.fig6_author_c2_bundle import C2BundleError
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import DEFAULT_SCORE as C3_SCORE
from validation.fischer_2023.fig6_author_c3_score import (
    C3ScoreError,
    load_c3_raw_bundle,
)
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT,
    DEFAULT_SCORE,
    RAW_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA,
    C4ScoreError,
    build_c4_receipt,
    build_c4_score,
    canonical_score_bytes,
    load_c4_raw_bundle,
    load_c4_receipt,
    load_c4_score,
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
C4_BUNDLE = _first_listable_bundle(
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C4-photon-v1",
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C4-photon-regen-v1",
)
SECONDS_PER_NS = 1.0e-9

RawEvidence = tuple[dict[str, Any], dict[str, np.ndarray], str]


def _require_external_c4() -> None:
    if not (C4_BUNDLE / "manifest.json").is_file():
        pytest.skip("Canonical external C4 raw bundle is unavailable.")


@pytest.fixture(scope="module")
def checked_score() -> dict[str, Any]:
    # Checked artifacts are repository evidence. Once committed, absence or
    # rejection is a hard failure rather than an optional external-data skip.
    return load_c4_score(DEFAULT_SCORE, receipt_path=DEFAULT_RECEIPT)


@pytest.fixture(scope="module")
def raw_evidence() -> RawEvidence:
    _require_external_c4()
    return load_c4_raw_bundle(C4_BUNDLE)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _npy_bytes(
    value: np.ndarray,
    *,
    version: tuple[int, int] = (3, 0),
) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.asarray(value),
        version=version,
        allow_pickle=False,
    )
    return stream.getvalue()


def _descriptor(value: np.ndarray) -> dict[str, object]:
    array = np.asarray(value)
    raw = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(raw).hexdigest(),
        "shape": list(array.shape),
    }


def _clone_raw_bundle(tmp_path: Path) -> Path:
    _require_external_c4()
    target = tmp_path / "c4"

    def link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    shutil.copytree(C4_BUNDLE, target, copy_function=link_or_copy)
    return target


def _replace_bytes(path: Path, content: bytes) -> None:
    # Raw clones normally hard-link their arrays. Detach before every
    # mutation so an aborted adversarial test cannot alter canonical evidence.
    path.unlink()
    path.write_bytes(content)


def _load_manifest(target: Path) -> dict[str, Any]:
    value = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_manifest(target: Path, manifest: dict[str, Any]) -> None:
    _replace_bytes(target / "manifest.json", _canonical_json(manifest))


def _replace_arrays(
    target: Path,
    replacements: dict[str, np.ndarray],
    *,
    rebind_descriptors: bool = True,
) -> None:
    manifest = _load_manifest(target)
    for name, value in replacements.items():
        array = np.asarray(value)
        raw = _npy_bytes(array)
        _replace_bytes(target / f"{name}.npy", raw)
        manifest["files"][f"{name}.npy"] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        if rebind_descriptors:
            manifest["metadata"]["array_descriptors"][name] = _descriptor(array)
    _write_manifest(target, manifest)


def _replace_array_encoding(
    target: Path,
    name: str,
    value: np.ndarray,
    *,
    version: tuple[int, int] = (3, 0),
    trailing: bytes = b"",
) -> None:
    raw = _npy_bytes(np.asarray(value), version=version) + trailing
    _replace_bytes(target / f"{name}.npy", raw)
    manifest = _load_manifest(target)
    manifest["files"][f"{name}.npy"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    _write_manifest(target, manifest)


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    return {"hex": result.hex(), "value": result}


def _forged_matching_receipt(
    score: dict[str, Any],
    tmp_path: Path,
) -> tuple[Path, Path]:
    score_path = tmp_path / "score.json"
    score_raw = _canonical_json(score)
    score_path.write_bytes(score_raw)
    receipt = json.loads(DEFAULT_RECEIPT.read_text(encoding="utf-8"))
    receipt["checked_score"]["file_sha256"] = hashlib.sha256(score_raw).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(_canonical_json(receipt))
    return score_path, receipt_path


def _photon_operator(
    f: np.ndarray,
    active: np.ndarray,
    density: np.ndarray,
    coherence: np.ndarray,
    *,
    step: int,
    n_bar: float,
    c_ns_inv: float,
    omit_terminal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independent source-order photon loop used to construct hostile bundles."""

    occupation = np.asarray(f, dtype=float)
    supported = np.asarray(active, dtype=bool)
    rho = np.asarray(density, dtype=float)
    K = np.asarray(coherence, dtype=float)
    one_minus_f = np.maximum(1.0 - occupation, 0.0)
    gain = np.zeros_like(occupation)
    loss_rate = np.zeros_like(occupation)
    size = occupation.size
    for i in range(size):
        if not supported[i]:
            continue
        j_up = i + step
        upper_limit = size - 1 if omit_terminal else size
        if j_up < upper_limit:
            coefficient = rho[j_up] * K[i, j_up]
            gain[i] += c_ns_inv * coefficient * occupation[j_up] * (n_bar + 1.0)
            loss_rate[i] += c_ns_inv * coefficient * one_minus_f[j_up] * n_bar
        j_down = i - step
        if j_down >= 0 and supported[j_down] and (not omit_terminal or i < size - 1):
            coefficient = rho[j_down] * K[i, j_down]
            gain[i] += c_ns_inv * coefficient * occupation[j_down] * n_bar
            loss_rate[i] += c_ns_inv * coefficient * one_minus_f[j_down] * (n_bar + 1.0)
    gain *= one_minus_f
    loss = loss_rate * occupation
    return gain, loss_rate, loss, gain - loss


def _candidate_family(
    raw_metadata: dict[str, Any],
    raw_arrays: dict[str, np.ndarray],
    c3_arrays: dict[str, np.ndarray],
    *,
    density: np.ndarray | None = None,
    coherence: np.ndarray | None = None,
    step: int = 20,
    wrong_loss_semantics: bool = False,
    wrong_unit_scale: bool = False,
    omit_terminal_candidate: bool = False,
) -> dict[str, np.ndarray]:
    """Return every C4-derived array rebound around one wrong hypothesis."""

    operator = raw_metadata["operator_inputs"]
    n_bar = float(operator["n_bar"]["value"])
    c_ns_inv = float(operator["c_photon_ns_inv"]["value"])
    f = np.asarray(raw_arrays["parent_f"])
    active = np.asarray(raw_arrays["parent_active_mask"])
    rho = (
        np.asarray(c3_arrays["native_cell_density_full"])
        if density is None
        else np.asarray(density)
    )
    K = np.asarray(c3_arrays["native_K_plus_full"]) if coherence is None else np.asarray(coherence)
    gain_ns, loss_rate_ns, loss_ns, net_ns = _photon_operator(
        f,
        active,
        rho,
        K,
        step=step,
        n_bar=n_bar,
        c_ns_inv=c_ns_inv,
        omit_terminal=omit_terminal_candidate,
    )
    endpoint_gain_ns, endpoint_rate_ns, endpoint_loss_ns, endpoint_net_ns = _photon_operator(
        f,
        active,
        rho,
        K,
        step=step,
        n_bar=n_bar,
        c_ns_inv=c_ns_inv,
        omit_terminal=True,
    )
    if wrong_loss_semantics:
        loss_ns = loss_rate_ns.copy()
        net_ns = gain_ns - loss_ns
        endpoint_loss_ns = endpoint_rate_ns.copy()
        endpoint_net_ns = endpoint_gain_ns - endpoint_loss_ns

    divisor = 1.0 if wrong_unit_scale else SECONDS_PER_NS
    gain_s = gain_ns / divisor
    loss_s = loss_ns / divisor
    net_s = net_ns / divisor
    endpoint_gain_s = endpoint_gain_ns / divisor
    endpoint_loss_s = endpoint_loss_ns / divisor
    endpoint_net_s = endpoint_net_ns / divisor

    parent_gain = np.asarray(raw_arrays["parent_qp_photon_gain_s_inv"])
    parent_loss = np.asarray(raw_arrays["parent_qp_photon_loss_s_inv"])
    parent_net = np.asarray(raw_arrays["parent_qp_photon_net_s_inv"])
    parent_qp_residual = np.asarray(raw_arrays["parent_qp_residual_s_inv"])
    terminal_gain = gain_s - endpoint_gain_s
    terminal_loss = loss_s - endpoint_loss_s
    terminal_net = net_s - endpoint_net_s
    terminal_support = (terminal_gain != 0.0) | (terminal_loss != 0.0) | (terminal_net != 0.0)
    return {
        "arithmetic_delta_gain_s_inv": endpoint_gain_s - parent_gain,
        "arithmetic_delta_loss_s_inv": endpoint_loss_s - parent_loss,
        "arithmetic_delta_net_s_inv": endpoint_net_s - parent_net,
        "operator_delta_gain_s_inv": gain_s - parent_gain,
        "operator_delta_loss_s_inv": loss_s - parent_loss,
        "operator_delta_net_s_inv": net_s - parent_net,
        "hybrid_phonon_residual_s_inv": np.asarray(
            raw_arrays["parent_phonon_residual_s_inv"]
        ).copy(),
        "hybrid_qp_residual_s_inv": parent_qp_residual + (net_s - parent_net),
        "qpsim_author_endpoint_gain_s_inv": endpoint_gain_s,
        "qpsim_author_endpoint_loss_s_inv": endpoint_loss_s,
        "qpsim_author_endpoint_net_s_inv": endpoint_net_s,
        "qpsim_gain_ns_inv": gain_ns,
        "qpsim_gain_s_inv": gain_s,
        "qpsim_loss_ns_inv": loss_ns,
        "qpsim_loss_rate_ns_inv": loss_rate_ns,
        "qpsim_loss_s_inv": loss_s,
        "qpsim_net_ns_inv": net_ns,
        "qpsim_net_s_inv": net_s,
        "terminal_extension_gain_s_inv": terminal_gain,
        "terminal_extension_loss_s_inv": terminal_loss,
        "terminal_extension_net_s_inv": terminal_net,
        "terminal_extension_support_mask": terminal_support,
    }


def _zero_candidate_family(
    raw_arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    zeros = np.zeros(1640)
    false = np.zeros(1640, dtype=bool)
    parent_gain = np.asarray(raw_arrays["parent_qp_photon_gain_s_inv"])
    parent_loss = np.asarray(raw_arrays["parent_qp_photon_loss_s_inv"])
    parent_net = np.asarray(raw_arrays["parent_qp_photon_net_s_inv"])
    parent_qp_residual = np.asarray(raw_arrays["parent_qp_residual_s_inv"])
    return {
        "arithmetic_delta_gain_s_inv": -parent_gain,
        "arithmetic_delta_loss_s_inv": -parent_loss,
        "arithmetic_delta_net_s_inv": -parent_net,
        "operator_delta_gain_s_inv": -parent_gain,
        "operator_delta_loss_s_inv": -parent_loss,
        "operator_delta_net_s_inv": -parent_net,
        "hybrid_phonon_residual_s_inv": np.asarray(
            raw_arrays["parent_phonon_residual_s_inv"]
        ).copy(),
        "hybrid_qp_residual_s_inv": parent_qp_residual - parent_net,
        "qpsim_author_endpoint_gain_s_inv": zeros.copy(),
        "qpsim_author_endpoint_loss_s_inv": zeros.copy(),
        "qpsim_author_endpoint_net_s_inv": zeros.copy(),
        "qpsim_gain_ns_inv": zeros.copy(),
        "qpsim_gain_s_inv": zeros.copy(),
        "qpsim_loss_ns_inv": zeros.copy(),
        "qpsim_loss_rate_ns_inv": zeros.copy(),
        "qpsim_loss_s_inv": zeros.copy(),
        "qpsim_net_ns_inv": zeros.copy(),
        "qpsim_net_s_inv": zeros.copy(),
        "terminal_extension_gain_s_inv": zeros.copy(),
        "terminal_extension_loss_s_inv": zeros.copy(),
        "terminal_extension_net_s_inv": zeros.copy(),
        "terminal_extension_support_mask": false,
    }


def test_checked_score_and_receipt_load_strictly(
    checked_score: dict[str, Any],
) -> None:
    receipt = load_c4_receipt(DEFAULT_RECEIPT)
    assert checked_score["schema"] == SCHEMA
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert checked_score["raw_bundle"]["schema"] == RAW_SCHEMA
    assert checked_score["stage"] == {
        "changed_component": "photon_operator",
        "comparison_stage_id": "c3c_native_cell_density",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C3",
        "stage_id": "C4",
        "status": "completed",
    }


def test_receipt_binds_complete_score_raw_and_replayed_parent(
    checked_score: dict[str, Any],
) -> None:
    _require_external_c4()
    receipt = load_c4_receipt(DEFAULT_RECEIPT)
    assert receipt == build_c4_receipt(
        DEFAULT_SCORE,
        c4_bundle_dir=C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert receipt["checked_score"] == {
        "file_sha256": hashlib.sha256(DEFAULT_SCORE.read_bytes()).hexdigest(),
        "schema": SCHEMA,
    }
    assert receipt["raw_bundle"] == checked_score["raw_bundle"]


def test_external_raw_rebuilds_checked_score_canonically(
    raw_evidence: RawEvidence,
) -> None:
    _metadata, arrays, manifest_sha = raw_evidence
    assert len(arrays) == 30
    rebuilt = build_c4_score(
        C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert canonical_score_bytes(rebuilt) == DEFAULT_SCORE.read_bytes()
    assert rebuilt["raw_bundle"] == {
        "manifest_sha256": manifest_sha,
        "schema": RAW_SCHEMA,
    }


def test_receipt_rejects_a_structurally_valid_numeric_score_forgery(
    tmp_path: Path,
) -> None:
    _require_external_c4()
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    record = score["conservation"]["public_photon"]["absolute_number_residual_ueV_s_inv"]
    record["value"] = float(np.nextafter(record["value"], np.inf))
    record["hex"] = float(record["value"]).hex()
    forged = tmp_path / "forged-score.json"
    forged.write_bytes(_canonical_json(score))
    with pytest.raises(C4ScoreError, match="do not independently reproduce"):
        build_c4_receipt(
            forged,
            c4_bundle_dir=C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        pytest.param(
            lambda score: score["stage"].__setitem__(
                "changed_component",
                "pair_operator",
            ),
            "stage identity",
            id="nested-stage",
        ),
        pytest.param(
            lambda score: score["units"].__setitem__(
                "public_return_contract",
                "loss is already physical",
            ),
            "units",
            id="nested-loss-contract",
        ),
        pytest.param(
            lambda score: score["endpoint_comparison"].__setitem__(
                "semantic_terminal_child_indices",
                [1619],
            ),
            "endpoint support",
            id="nested-terminal-support",
        ),
        pytest.param(
            lambda score: score["operator_inputs"].__setitem__(
                "photon_step_bins",
                19,
            ),
            "operator-input relationship",
            id="nested-bin",
        ),
    ),
)
def test_matching_receipt_cannot_hide_a_nested_score_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    mutate(score)
    score_path, receipt_path = _forged_matching_receipt(score, tmp_path)
    with pytest.raises(C4ScoreError, match=match):
        load_c4_score(score_path, receipt_path=receipt_path)


def test_checked_score_rejects_an_expanded_no_root_claim(tmp_path: Path) -> None:
    score = json.loads(DEFAULT_SCORE.read_text(encoding="utf-8"))
    score["limitations"] = {
        "root_verified": True,
        "scope": "the complete 300-point C4 curve",
        "statement": (
            "C4 nonlinear root, Newton stopping, plotted ordinate, "
            "observable, and paper parity are all verified."
        ),
    }
    score_path, receipt_path = _forged_matching_receipt(score, tmp_path)
    with pytest.raises(C4ScoreError, match="limitation statement"):
        load_c4_score(score_path, receipt_path=receipt_path)


def test_scope_explicitly_excludes_roots_curves_and_observables(
    checked_score: dict[str, Any],
) -> None:
    limitations = checked_score["limitations"]
    assert limitations["scope"] == "one authenticated C3c frozen point only"
    for excluded in (
        "No C4 nonlinear root",
        "Newton history",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "observable change",
        "paper-parity claim",
    ):
        assert excluded in limitations["statement"]


def test_raw_loader_rejects_an_extra_file(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    (target / "undeclared.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(C4ScoreError, match="directory closure"):
        load_c4_raw_bundle(target)


def test_raw_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    (target / "qpsim_gain_s_inv.npy").unlink()
    with pytest.raises(C4ScoreError, match="directory closure"):
        load_c4_raw_bundle(target)


def test_raw_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    raw = (target / "manifest.json").read_text(encoding="utf-8")
    duplicate = raw.replace(
        "{\n",
        f'{{\n  "schema": "{RAW_SCHEMA}",\n',
        1,
    )
    _replace_bytes(target / "manifest.json", duplicate.encode("utf-8"))
    with pytest.raises(C4ScoreError, match="Duplicate JSON key 'schema'"):
        load_c4_raw_bundle(target)


def test_raw_loader_rejects_noncanonical_json(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    manifest = _load_manifest(target)
    _replace_bytes(
        target / "manifest.json",
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8"),
    )
    with pytest.raises(C4ScoreError, match="not canonical JSON"):
        load_c4_raw_bundle(target)


def test_raw_loader_rejects_a_symlinked_bundle_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c4()
    original = Path.lstat

    def selected_root_is_symlink(path: Path) -> os.stat_result | SimpleNamespace:
        if path == C4_BUNDLE:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original(path)

    monkeypatch.setattr(Path, "lstat", selected_root_is_symlink)
    with pytest.raises(C4ScoreError, match="non-symlink directory"):
        load_c4_raw_bundle(C4_BUNDLE)


def test_raw_loader_rejects_noncanonical_npy_version_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "parent_f"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    _replace_array_encoding(target, name, array, version=(2, 0))
    with pytest.raises(C4ScoreError, match="not canonical NPY v3"):
        load_c4_raw_bundle(target)


def test_raw_loader_rejects_trailing_npy_bytes_after_file_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "parent_f"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    _replace_array_encoding(target, name, array, trailing=b"forged")
    with pytest.raises(C4ScoreError, match="trailing bytes"):
        load_c4_raw_bundle(target)


def test_independent_replay_rejects_wrong_dtype_after_full_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "qpsim_gain_s_inv"
    array = np.load(target / f"{name}.npy", allow_pickle=False).astype(np.float32)
    _replace_arrays(target, {name: array})
    load_c4_raw_bundle(target)
    with pytest.raises(C4ScoreError, match="does not match independent recomputation"):
        build_c4_score(
            target,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_independent_replay_rejects_signed_zero_after_full_rebind(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "qpsim_gain_s_inv"
    array = np.load(target / f"{name}.npy", allow_pickle=False)
    array = np.asarray(array).copy()
    assert array[0] == 0.0 and not np.signbit(array[0])
    array[0] = -0.0
    _replace_arrays(target, {name: array})
    load_c4_raw_bundle(target)
    with pytest.raises(C4ScoreError, match="does not match independent recomputation"):
        build_c4_score(
            target,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_independent_replay_rejects_a_fully_rebound_one_ulp_gain_forgery(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    metadata, arrays, _manifest_sha = load_c4_raw_bundle(target)
    del metadata
    candidate = {
        name: np.asarray(arrays[name]).copy()
        for name in (
            "qpsim_gain_ns_inv",
            "qpsim_gain_s_inv",
            "qpsim_net_ns_inv",
            "qpsim_net_s_inv",
            "operator_delta_gain_s_inv",
            "operator_delta_net_s_inv",
            "hybrid_qp_residual_s_inv",
            "terminal_extension_gain_s_inv",
            "terminal_extension_net_s_inv",
            "terminal_extension_support_mask",
        )
    }
    index = int(np.argmax(candidate["qpsim_gain_ns_inv"]))
    candidate["qpsim_gain_ns_inv"][index] = np.nextafter(
        candidate["qpsim_gain_ns_inv"][index],
        np.inf,
    )
    candidate["qpsim_gain_s_inv"] = candidate["qpsim_gain_ns_inv"] / SECONDS_PER_NS
    candidate["qpsim_net_ns_inv"] = candidate["qpsim_gain_ns_inv"] - arrays["qpsim_loss_ns_inv"]
    candidate["qpsim_net_s_inv"] = candidate["qpsim_net_ns_inv"] / SECONDS_PER_NS
    candidate["operator_delta_gain_s_inv"] = (
        candidate["qpsim_gain_s_inv"] - arrays["parent_qp_photon_gain_s_inv"]
    )
    candidate["operator_delta_net_s_inv"] = (
        candidate["qpsim_net_s_inv"] - arrays["parent_qp_photon_net_s_inv"]
    )
    candidate["hybrid_qp_residual_s_inv"] = (
        arrays["parent_qp_residual_s_inv"] + candidate["operator_delta_net_s_inv"]
    )
    candidate["terminal_extension_gain_s_inv"] = (
        candidate["qpsim_gain_s_inv"] - arrays["qpsim_author_endpoint_gain_s_inv"]
    )
    candidate["terminal_extension_net_s_inv"] = (
        candidate["qpsim_net_s_inv"] - arrays["qpsim_author_endpoint_net_s_inv"]
    )
    candidate["terminal_extension_support_mask"] = (
        (candidate["terminal_extension_gain_s_inv"] != 0.0)
        | (arrays["terminal_extension_loss_s_inv"] != 0.0)
        | (candidate["terminal_extension_net_s_inv"] != 0.0)
    )
    _replace_arrays(target, candidate)
    load_c4_raw_bundle(target)
    with pytest.raises(C4ScoreError, match="does not match independent recomputation"):
        build_c4_score(
            target,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


@pytest.mark.parametrize(
    "case",
    (
        "zero",
        "wrong-loss-semantics",
        "wrong-units",
        "point-density",
        "k-minus",
        "terminal-omission",
    ),
)
def test_independent_replay_rejects_fully_rebound_wrong_science(
    tmp_path: Path,
    case: str,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    metadata, arrays, _manifest_sha = load_c4_raw_bundle(target)
    _c3_metadata, c3_arrays, _c3_manifest_sha = load_c3_raw_bundle(C3_BUNDLE)

    if case == "zero":
        replacements = _zero_candidate_family(arrays)
    elif case == "wrong-loss-semantics":
        replacements = _candidate_family(
            metadata,
            arrays,
            c3_arrays,
            wrong_loss_semantics=True,
        )
    elif case == "wrong-units":
        replacements = _candidate_family(
            metadata,
            arrays,
            c3_arrays,
            wrong_unit_scale=True,
        )
    elif case == "point-density":
        energy = np.asarray(c3_arrays["native_E_centers_ueV"])
        active = np.asarray(c3_arrays["native_active_mask"])
        gap = float(metadata["operator_inputs"]["gap_ueV"]["value"])
        density = np.zeros_like(energy)
        density[active] = energy[active] / np.sqrt(energy[active] ** 2 - gap**2)
        assert not np.array_equal(
            density,
            c3_arrays["native_cell_density_full"],
        )
        replacements = _candidate_family(
            metadata,
            arrays,
            c3_arrays,
            density=density,
        )
    elif case == "k-minus":
        replacements = _candidate_family(
            metadata,
            arrays,
            c3_arrays,
            coherence=c3_arrays["native_K_minus_full"],
        )
    else:
        replacements = _candidate_family(
            metadata,
            arrays,
            c3_arrays,
            omit_terminal_candidate=True,
        )

    _replace_arrays(target, replacements)
    # Every file hash and every scientific descriptor now agrees with the
    # hostile hypothesis. Only source-independent replay can reject it.
    load_c4_raw_bundle(target)
    with pytest.raises(C4ScoreError, match="does not match independent recomputation"):
        build_c4_score(
            target,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_wrong_photon_bin_fails_after_arrays_and_metadata_are_fully_rebound(
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    metadata, arrays, _manifest_sha = load_c4_raw_bundle(target)
    _c3_metadata, c3_arrays, _c3_manifest_sha = load_c3_raw_bundle(C3_BUNDLE)
    replacements = _candidate_family(
        metadata,
        arrays,
        c3_arrays,
        step=19,
    )
    _replace_arrays(target, replacements)
    manifest = _load_manifest(target)
    operator = manifest["metadata"]["operator_inputs"]
    operator["photon_step_bins"] = 19
    operator["omega_0_ueV"] = _float_record(19.0)
    _write_manifest(target, manifest)
    load_c4_raw_bundle(target)
    with pytest.raises(C4ScoreError, match="operator-input closure"):
        build_c4_score(
            target,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )


def test_c4_refuses_forged_c3_score_and_matching_receipt_even_when_c3_loads() -> None:
    _require_external_c4()
    with tempfile.TemporaryDirectory(
        dir=REPOSITORY_ROOT / "tmp",
        prefix="c4-forged-c3-",
    ) as directory:
        root = Path(directory)
        score = json.loads(C3_SCORE.read_text(encoding="utf-8"))
        comparison = score["comparison"]
        original = float(comparison["net_subtraction_worst_fraction_of_limit"])
        comparison["net_subtraction_worst_fraction_of_limit"] = float(np.nextafter(original, 1.0))
        score_raw = _canonical_json(score)
        score_path = root / "c3-score.json"
        score_path.write_bytes(score_raw)
        receipt = json.loads(C3_RECEIPT.read_text(encoding="utf-8"))
        receipt["checked_score"]["file_sha256"] = hashlib.sha256(score_raw).hexdigest()
        receipt_path = root / "c3-receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))

        with pytest.raises(C4ScoreError, match="does not independently reproduce"):
            build_c4_score(
                C4_BUNDLE,
                c3_bundle_dir=C3_BUNDLE,
                c2_bundle_dir=C2_BUNDLE,
                c3_score_path=score_path,
                c3_receipt_path=receipt_path,
            )


@pytest.mark.parametrize(
    ("c3_bundle", "c2_bundle"),
    (
        pytest.param(C2_BUNDLE, C2_BUNDLE, id="c2-raw-passed-as-c3"),
        pytest.param(C3_BUNDLE, C3_BUNDLE, id="c3-raw-passed-as-c2"),
    ),
)
def test_c4_refuses_wrong_c3_or_c2_raw_parent(
    c3_bundle: Path,
    c2_bundle: Path,
) -> None:
    _require_external_c4()
    with pytest.raises((C4ScoreError, C3ScoreError, C2BundleError)):
        build_c4_score(
            C4_BUNDLE,
            c3_bundle_dir=c3_bundle,
            c2_bundle_dir=c2_bundle,
        )


def test_score_builder_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c4()
    target_relative = "qpsim/collisions/sub_gap_photon.py"
    target = REPOSITORY_ROOT / target_relative
    original = c4_score.canonical_source_bytes

    def drifted(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == target.resolve():
            return content + b"\n# simulated verifier-source drift\n"
        return content

    assert target_relative in c4_score._SOURCE_BYTES_AT_IMPORT
    monkeypatch.setattr(c4_score, "canonical_source_bytes", drifted)
    with pytest.raises(
        C4ScoreError,
        match=r"C4 score source changed during execution: "
        r"qpsim/collisions/sub_gap_photon\.py",
    ):
        build_c4_score(
            C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
        )
