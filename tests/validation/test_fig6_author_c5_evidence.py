"""Adversarial evidence tests for the formal Figure 6 C5 score and receipt."""

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
from qpsim.collisions.phonon import phonon_collision_rates
from qpsim.physics.spectral import SpectralContext
from validation.fischer_2023 import fig6_author_c5_score as c5_score
from validation.fischer_2023.fig6_author_c2_bundle import C2BundleError
from validation.fischer_2023.fig6_author_c3_score import (
    DEFAULT_RECEIPT as C3_RECEIPT,
)
from validation.fischer_2023.fig6_author_c3_score import C3ScoreError
from validation.fischer_2023.fig6_author_c4_score import (
    DEFAULT_RECEIPT as C4_RECEIPT,
)
from validation.fischer_2023.fig6_author_c4_score import DEFAULT_SCORE as C4_SCORE
from validation.fischer_2023.fig6_author_c4_score import C4ScoreError
from validation.fischer_2023.fig6_author_c5_score import (
    DEFAULT_RECEIPT,
    DEFAULT_SCORE,
    RAW_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA,
    C5ScoreError,
    build_c5_receipt,
    build_c5_score,
    canonical_score_bytes,
    load_c5_raw_bundle,
    load_c5_receipt,
    load_c5_score,
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
CANONICAL_C5_BUNDLE = REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-v1"
DEV_C5_BUNDLE_V6 = (
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-producer-dev-v6"
)
DEV_C5_BUNDLE = (
    REPOSITORY_ROOT / "tmp" / "author-runs" / "fig6-T020-sweep049-C5-qp-phonon-producer-dev-v5"
)
SECONDS_PER_NS = 1.0e-9
N_QP = 1640

RawEvidence = tuple[dict[str, Any], dict[str, np.ndarray], str]


def _c5_bundle() -> Path:
    return _first_listable_bundle(
        CANONICAL_C5_BUNDLE,
        REPOSITORY_ROOT
        / "tmp"
        / "author-runs"
        / "fig6-T020-sweep049-C5-qp-phonon-regen-v1",
        DEV_C5_BUNDLE_V6,
        DEV_C5_BUNDLE,
    )


def _require_external_c5() -> None:
    required = (
        _c5_bundle() / "manifest.json",
        C2_BUNDLE / "manifest.json",
        C3_BUNDLE / "manifest.json",
        C4_BUNDLE / "manifest.json",
    )
    if not all(path.is_file() for path in required):
        pytest.skip("Formal C5/C4/C3/C2 raw evidence is unavailable.")


def _require_checked_c5() -> tuple[Path, Path]:
    # Deliberately no skip: the checked C5 score and receipt are committed
    # under validation/paper_data, so their absence is a hard repository
    # failure -- the same rule test_fig6_author_c3_evidence.py applies to C3.
    for path in (DEFAULT_SCORE, DEFAULT_RECEIPT):
        if not path.is_file():
            pytest.fail(f"Committed C5 artifact is missing: {path}")
    return DEFAULT_SCORE, DEFAULT_RECEIPT


@pytest.fixture(scope="module")
def raw_evidence() -> RawEvidence:
    _require_external_c5()
    return load_c5_raw_bundle(_c5_bundle())


@pytest.fixture(scope="module")
def checked_score() -> dict[str, Any]:
    score_path, receipt_path = _require_checked_c5()
    return load_c5_score(score_path, receipt_path=receipt_path)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _expected_raw_source_closure() -> set[str]:
    qpsim_root = REPOSITORY_ROOT / "qpsim"
    qpsim_sources = {
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for pattern in ("*.py", "*.yaml", "*.yml")
        for path in qpsim_root.rglob(pattern)
    }
    validation_sources = {
        "validation/__init__.py",
        "validation/author_source.py",
        "validation/reproduction_ladder.py",
        "validation/source_provenance.py",
        "validation/fischer_2023/__init__.py",
        "validation/fischer_2023/fig6_author_adapter.py",
        "validation/fischer_2023/fig6_author_frozen_state.py",
        "validation/fischer_2023/fig6_author_c0_bundle.py",
        "validation/fischer_2023/fig6_author_c0_summary.py",
        "validation/fischer_2023/fig6_author_c1_score.py",
        "validation/fischer_2023/fig6_author_c2_bundle.py",
        "validation/fischer_2023/fig6_author_c2_parameters.py",
        "validation/fischer_2023/fig6_author_c2_score.py",
        "validation/fischer_2023/fig6_author_c3_bundle.py",
        "validation/fischer_2023/fig6_author_c3_score.py",
        "validation/fischer_2023/fig6_author_c4_bundle.py",
        "validation/fischer_2023/fig6_author_c4_score.py",
        "validation/fischer_2023/fig6_author_c5_bundle.py",
        "validation/fischer_2023/fig6_solve.py",
        "validation/reference_models/__init__.py",
        "validation/reference_models/fischer_2023/__init__.py",
        "validation/reference_models/fischer_2023/fig6_author_c0.py",
    }
    return qpsim_sources | validation_sources


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
    content = _npy_bytes(array)
    return {
        "dtype": array.dtype.str,
        "npy_sha256": hashlib.sha256(content).hexdigest(),
        "shape": list(array.shape),
    }


def _float_record(value: float) -> dict[str, object]:
    result = float(value)
    return {"hex": result.hex(), "value": result}


def _clone_raw_bundle(tmp_path: Path) -> Path:
    _require_external_c5()
    target = tmp_path / "c5"

    def link_or_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        return destination

    shutil.copytree(_c5_bundle(), target, copy_function=link_or_copy)
    return target


def _replace_bytes(path: Path, content: bytes) -> None:
    # Raw clones are normally hard-linked. Always detach before mutation so a
    # failed hostile test cannot modify the retained external evidence.
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
        content = _npy_bytes(array)
        _replace_bytes(target / f"{name}.npy", content)
        manifest["files"][f"{name}.npy"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "size_bytes": len(content),
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
    content = _npy_bytes(np.asarray(value), version=version) + trailing
    _replace_bytes(target / f"{name}.npy", content)
    manifest = _load_manifest(target)
    manifest["files"][f"{name}.npy"] = {
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }
    _write_manifest(target, manifest)


def _mutate_metadata(target: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    manifest = _load_manifest(target)
    mutate(manifest["metadata"])
    _write_manifest(target, manifest)


def _forged_matching_receipt(
    score: dict[str, Any],
    tmp_path: Path,
) -> tuple[Path, Path]:
    _base_score, base_receipt = _require_checked_c5()
    score_path = tmp_path / "score.json"
    score_raw = _canonical_json(score)
    score_path.write_bytes(score_raw)
    receipt = json.loads(base_receipt.read_text(encoding="utf-8"))
    receipt["checked_score"]["file_sha256"] = hashlib.sha256(score_raw).hexdigest()
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(_canonical_json(receipt))
    return score_path, receipt_path


def _build_score(target: Path) -> dict[str, Any]:
    return build_c5_score(
        target,
        c4_bundle_dir=C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )


def _positive_zero(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value).copy()
    if result.dtype.kind == "f":
        result[result == 0.0] = 0.0
    return result


def _context(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
) -> SpectralContext:
    return SpectralContext(
        arrays["parent_E_centers_ueV"],
        arrays["parent_dE_ueV"],
        metadata["operator_inputs"]["gap_ueV"]["value"],
    )


def _refresh_residuals(
    arrays: dict[str, np.ndarray],
    replacements: dict[str, np.ndarray],
) -> None:
    scattering_delta = replacements.get(
        "qp_scattering_delta_net_s_inv",
        arrays["qp_scattering_delta_net_s_inv"],
    )
    pair_delta = replacements.get(
        "qp_pair_delta_net_s_inv",
        arrays["qp_pair_delta_net_s_inv"],
    )
    parent = arrays["parent_qp_residual_s_inv"]
    replacements.update(
        {
            "c5s_qp_residual_s_inv": _positive_zero(parent + scattering_delta),
            "c5p_qp_residual_s_inv": _positive_zero(parent + pair_delta),
            "c5sp_qp_residual_s_inv": _positive_zero(parent + scattering_delta + pair_delta),
        }
    )


def _channel_family(
    arrays: dict[str, np.ndarray],
    channel: str,
    gain_ns: np.ndarray,
    loss_rate_ns: np.ndarray,
    *,
    kernel: np.ndarray | None = None,
    physical_loss_ns: np.ndarray | None = None,
    seconds_divisor: float = SECONDS_PER_NS,
) -> dict[str, np.ndarray]:
    """Rebind every derived vector around one hostile channel hypothesis."""

    f = arrays["parent_f"]
    gain_native = _positive_zero(gain_ns)
    rate_native = _positive_zero(loss_rate_ns)
    loss_native = _positive_zero(rate_native * f if physical_loss_ns is None else physical_loss_ns)
    net_native = _positive_zero(gain_native - loss_native)
    gain_s = _positive_zero(gain_native / seconds_divisor)
    rate_s = _positive_zero(rate_native / seconds_divisor)
    loss_s = _positive_zero(loss_native / seconds_divisor)
    net_s = _positive_zero(net_native / seconds_divisor)
    replacements = {
        f"qpsim_qp_{channel}_gain_ns_inv": gain_native,
        f"qpsim_qp_{channel}_loss_rate_ns_inv": rate_native,
        f"qpsim_qp_{channel}_loss_ns_inv": loss_native,
        f"qpsim_qp_{channel}_net_ns_inv": net_native,
        f"qpsim_qp_{channel}_gain_s_inv": gain_s,
        f"qpsim_qp_{channel}_loss_rate_s_inv": rate_s,
        f"qpsim_qp_{channel}_loss_s_inv": loss_s,
        f"qpsim_qp_{channel}_net_s_inv": net_s,
        f"qp_{channel}_delta_gain_s_inv": _positive_zero(
            gain_s - arrays[f"parent_qp_{channel}_gain_s_inv"]
        ),
        f"qp_{channel}_delta_loss_s_inv": _positive_zero(
            loss_s - arrays[f"parent_qp_{channel}_loss_s_inv"]
        ),
        f"qp_{channel}_delta_net_s_inv": _positive_zero(
            net_s - arrays[f"parent_qp_{channel}_net_s_inv"]
        ),
    }
    if kernel is not None:
        replacements[f"qpsim_qp_{channel}_kernel_ns_inv_ueV_inv"] = _positive_zero(kernel)
    if channel == "scattering":
        selected_kernel = (
            arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"] if kernel is None else kernel
        )
        n_diff = arrays["parent_projected_n_phonon"][arrays["qpsim_omega_idx_diff"]]
        cross_s = _positive_zero(
            arrays["parent_f"]
            * (
                (selected_kernel * n_diff).T
                @ (arrays["parent_cell_weights_ueV"] * arrays["parent_f"])
            )
            / SECONDS_PER_NS
        )
        parent_gain = _positive_zero(arrays["parent_qp_scattering_gain_s_inv"] - cross_s)
        parent_loss = _positive_zero(arrays["parent_qp_scattering_loss_s_inv"] - cross_s)
        replacements.update(
            {
                "scattering_pauli_cross_term_s_inv": cross_s,
                "parent_qp_scattering_rebucketed_gain_s_inv": parent_gain,
                "parent_qp_scattering_rebucketed_loss_s_inv": parent_loss,
                "qp_scattering_rebucketed_delta_gain_s_inv": (_positive_zero(gain_s - parent_gain)),
                "qp_scattering_rebucketed_delta_loss_s_inv": (_positive_zero(loss_s - parent_loss)),
            }
        )
    _refresh_residuals(arrays, replacements)
    return replacements


def _evaluate_channel(
    metadata: dict[str, Any],
    arrays: dict[str, np.ndarray],
    channel: str,
    kernel: np.ndarray,
    *,
    n_p: np.ndarray | None = None,
    n_emit: np.ndarray | None = None,
    n_abs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = _context(metadata, arrays)
    if channel == "scattering":
        return phonon_collision_rates(
            arrays["parent_f"],
            ctx,
            kernel,
            None,
            0.0,
            enable_scattering=True,
            enable_recombination=False,
            N_p_override=arrays["qpsim_N_p"] if n_p is None else n_p,
        )
    return phonon_collision_rates(
        arrays["parent_f"],
        ctx,
        None,
        kernel,
        0.0,
        enable_scattering=False,
        enable_recombination=True,
        N_emit_override=arrays["qpsim_N_emit"] if n_emit is None else n_emit,
        N_abs_override=arrays["qpsim_N_abs"] if n_abs is None else n_abs,
    )


def _wrong_unit_family(
    arrays: dict[str, np.ndarray],
    channel: str,
) -> dict[str, np.ndarray]:
    replacements = _channel_family(
        arrays,
        channel,
        arrays[f"qpsim_qp_{channel}_gain_ns_inv"],
        arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"],
        seconds_divisor=1.0,
    )
    return replacements


def _equal_gain_loss_shift_family(
    arrays: dict[str, np.ndarray],
    channel: str,
) -> dict[str, np.ndarray]:
    f = arrays["parent_f"]
    rate = arrays[f"qpsim_qp_{channel}_loss_rate_ns_inv"]
    rate_increment = 0.01 * rate
    physical_shift = rate_increment * f
    return _channel_family(
        arrays,
        channel,
        arrays[f"qpsim_qp_{channel}_gain_ns_inv"] + physical_shift,
        rate + rate_increment,
    )


def test_checked_score_and_receipt_load_strictly(
    checked_score: dict[str, Any],
) -> None:
    score_path, receipt_path = _require_checked_c5()
    receipt = load_c5_receipt(receipt_path)
    assert load_c5_score(score_path, receipt_path=receipt_path) == checked_score
    assert checked_score["schema"] == SCHEMA
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert checked_score["raw_bundle"]["schema"] == RAW_SCHEMA
    assert checked_score["stage"] == {
        "changed_component": "qp_phonon_operator",
        "comparison_stage_id": "C4",
        "evidence_class": "hybrid_component_substitution",
        "parent_stage_id": "C4",
        "stage_id": "C5",
        "status": "completed",
    }


def test_external_raw_rebuilds_complete_checked_score(
    raw_evidence: RawEvidence,
    checked_score: dict[str, Any],
) -> None:
    _metadata, arrays, manifest_sha = raw_evidence
    assert len(arrays) == 58
    rebuilt = _build_score(_c5_bundle())
    assert canonical_score_bytes(rebuilt) == canonical_score_bytes(checked_score)
    assert rebuilt["raw_bundle"] == {
        "manifest_sha256": manifest_sha,
        "schema": RAW_SCHEMA,
    }


def test_raw_source_closure_is_complete_and_acyclic(
    raw_evidence: RawEvidence,
) -> None:
    metadata, _arrays, _manifest_sha = raw_evidence
    raw_sources = set(metadata["sources"])
    expected = _expected_raw_source_closure()
    assert raw_sources == expected
    assert raw_sources == set(c5_score._RAW_SOURCE_HASHES_AT_IMPORT)
    assert "validation/fischer_2023/fig6_author_c5_score.py" not in raw_sources

    score_sources = set(c5_score._SOURCE_HASHES_AT_IMPORT)
    assert score_sources == expected | {"validation/fischer_2023/fig6_author_c5_score.py"}


def test_receipt_rebuild_binds_complete_score_raw_and_parent(
    checked_score: dict[str, Any],
) -> None:
    _require_external_c5()
    score_path, receipt_path = _require_checked_c5()
    receipt = load_c5_receipt(receipt_path)
    assert receipt == build_c5_receipt(
        score_path,
        c5_bundle_dir=_c5_bundle(),
        c4_bundle_dir=C4_BUNDLE,
        c3_bundle_dir=C3_BUNDLE,
        c2_bundle_dir=C2_BUNDLE,
    )
    assert receipt["checked_score"] == {
        "file_sha256": hashlib.sha256(score_path.read_bytes()).hexdigest(),
        "schema": SCHEMA,
    }
    assert receipt["raw_bundle"] == checked_score["raw_bundle"]
    assert (
        receipt["parent_c4"]["score_file_sha256"]
        == hashlib.sha256(C4_SCORE.read_bytes()).hexdigest()
    )
    assert (
        receipt["parent_c4"]["receipt_file_sha256"]
        == hashlib.sha256(C4_RECEIPT.read_bytes()).hexdigest()
    )


def test_fully_rebound_numeric_score_and_receipt_forgery_is_rejected(
    tmp_path: Path,
) -> None:
    score_path, _receipt_path = _require_checked_c5()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    record = score["conservation"]["pair_number_change_diagnostic_not_a_conservation_gate"][
        "weighted_net_s_inv_ueV"
    ]
    record["value"] = float(np.nextafter(record["value"], np.inf))
    record["hex"] = float(record["value"]).hex()
    forged_score, forged_receipt = _forged_matching_receipt(score, tmp_path)

    # The receipt digest is rebound to the forged complete score bytes. Only
    # the independently fixed numerical/evidence pins can reject it.
    with pytest.raises(
        C5ScoreError,
        match=r"canonical numerical metric pins|evidence digest",
    ):
        load_c5_score(forged_score, receipt_path=forged_receipt)


def test_receipt_builder_refuses_structurally_valid_numeric_forgery() -> None:
    score_path, _receipt_path = _require_checked_c5()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    record = score["channel_comparison"]["combined_net"]["l1_absolute_s_inv"]
    record["value"] = float(np.nextafter(record["value"], np.inf))
    record["hex"] = float(record["value"]).hex()
    with tempfile.TemporaryDirectory(
        dir=REPOSITORY_ROOT / "tmp",
        prefix="c5-score-forgery-",
    ) as directory:
        forged = Path(directory) / "numeric-forgery.json"
        forged.write_bytes(_canonical_json(score))
        with pytest.raises(
            C5ScoreError,
            match=r"canonical numerical metric pins|evidence digest",
        ):
            build_c5_receipt(
                forged,
                c5_bundle_dir=_c5_bundle(),
                c4_bundle_dir=C4_BUNDLE,
                c3_bundle_dir=C3_BUNDLE,
                c2_bundle_dir=C2_BUNDLE,
            )


@pytest.mark.parametrize(
    ("mutate", "match"),
    (
        pytest.param(
            lambda score: score["sources"].__setitem__(
                sorted(score["sources"])[0],
                "0" * 64,
            ),
            "source closure",
            id="source",
        ),
        pytest.param(
            lambda score: score["runtime"]["producer_public_array_generation"].__setitem__(
                "platform",
                score["runtime"]["producer_public_array_generation"]["platform"] + "-forged",
            ),
            "evidence digest",
            id="runtime",
        ),
        pytest.param(
            lambda score: score["limitations"].update(
                {
                    "scope": "complete C6 nonlinear curve",
                    "statement": (
                        "C5 proves a nonlinear root, plotted observable, and paper parity."
                    ),
                }
            ),
            "evidence digest",
            id="claims",
        ),
        pytest.param(
            lambda score: score["units"].__setitem__(
                "public_return_contract",
                "loss_rate is already physical loss",
            ),
            "evidence digest",
            id="loss-contract",
        ),
        pytest.param(
            lambda score: score["parent_bindings"].__setitem__(
                "c4_raw_manifest_sha256",
                "0" * 64,
            ),
            r"evidence digest|parent",
            id="ancestry",
        ),
        pytest.param(
            lambda score: score["acceptance"]["limits"].__setitem__(
                "combined_and_per_channel_net_symmetric_relative_l1",
                _float_record(2.0e-12),
            ),
            r"acceptance limits|evidence digest",
            id="acceptance",
        ),
        pytest.param(
            lambda score: score["source_binding"].__setitem__(
                "scope",
                score["source_binding"]["scope"] + " (forged)",
            ),
            r"source_binding|evidence digest",
            id="source-binding",
        ),
    ),
)
def test_matching_receipt_cannot_hide_semantic_score_mutation(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    score_path, _receipt_path = _require_checked_c5()
    score = json.loads(score_path.read_text(encoding="utf-8"))
    mutate(score)
    forged_score, forged_receipt = _forged_matching_receipt(score, tmp_path)
    with pytest.raises(C5ScoreError, match=match):
        load_c5_score(forged_score, receipt_path=forged_receipt)


def test_scope_explicitly_excludes_solver_curve_and_observable_claims(
    checked_score: dict[str, Any],
) -> None:
    limitations = checked_score["limitations"]
    assert limitations["scope"] == "one authenticated C4 frozen point only"
    for excluded in (
        "No C5 nonlinear root",
        "Newton history",
        "stopping result",
        "plotted ordinate",
        "300-point curve",
        "observable change",
        "paper-parity claim",
    ):
        assert excluded in limitations["statement"]


def test_receipt_mutations_cannot_rebind_raw_or_parent(
    tmp_path: Path,
) -> None:
    score_path, receipt_path = _require_checked_c5()
    for field_path in ("raw", "parent"):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if field_path == "raw":
            receipt["raw_bundle"]["manifest_sha256"] = "0" * 64
        else:
            receipt["parent_c4"]["score_file_sha256"] = "0" * 64
        forged = tmp_path / f"{field_path}-receipt.json"
        forged.write_bytes(_canonical_json(receipt))
        with pytest.raises(C5ScoreError, match=r"does not match|binding"):
            load_c5_score(score_path, receipt_path=forged)


def test_raw_loader_rejects_an_extra_file(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    (target / "undeclared.txt").write_text("not evidence", encoding="utf-8")
    with pytest.raises(C5ScoreError, match="directory closure"):
        load_c5_raw_bundle(target)


def test_raw_loader_rejects_a_missing_file(tmp_path: Path) -> None:
    target = _clone_raw_bundle(tmp_path)
    (target / "qpsim_qp_pair_gain_s_inv.npy").unlink()
    with pytest.raises(C5ScoreError, match="directory closure"):
        load_c5_raw_bundle(target)


def test_raw_loader_rejects_duplicate_or_noncanonical_json(
    tmp_path: Path,
) -> None:
    duplicate_target = _clone_raw_bundle(tmp_path / "duplicate")
    raw = (duplicate_target / "manifest.json").read_text(encoding="utf-8")
    duplicate = raw.replace(
        "{\n",
        f'{{\n  "schema": "{RAW_SCHEMA}",\n',
        1,
    )
    _replace_bytes(
        duplicate_target / "manifest.json",
        duplicate.encode("utf-8"),
    )
    with pytest.raises(C5ScoreError, match="Duplicate JSON key 'schema'"):
        load_c5_raw_bundle(duplicate_target)

    noncanonical_target = _clone_raw_bundle(tmp_path / "noncanonical")
    manifest = _load_manifest(noncanonical_target)
    _replace_bytes(
        noncanonical_target / "manifest.json",
        json.dumps(manifest, sort_keys=True, allow_nan=False).encode("utf-8"),
    )
    with pytest.raises(C5ScoreError, match="not canonical JSON"):
        load_c5_raw_bundle(noncanonical_target)


def test_raw_loader_rejects_a_symlinked_bundle_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c5()
    selected = _c5_bundle()
    original = Path.lstat

    def selected_root_is_symlink(
        path: Path,
    ) -> os.stat_result | SimpleNamespace:
        if path == selected:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original(path)

    monkeypatch.setattr(Path, "lstat", selected_root_is_symlink)
    with pytest.raises(C5ScoreError, match="non-symlink directory"):
        load_c5_raw_bundle(selected)


def test_raw_loader_rejects_npy_v2_and_trailing_bytes(
    tmp_path: Path,
) -> None:
    v2_target = _clone_raw_bundle(tmp_path / "v2")
    name = "parent_f"
    array = np.load(v2_target / f"{name}.npy", allow_pickle=False)
    _replace_array_encoding(v2_target, name, array, version=(2, 0))
    with pytest.raises(C5ScoreError, match="not canonical NPY v3"):
        load_c5_raw_bundle(v2_target)

    trailing_target = _clone_raw_bundle(tmp_path / "trailing")
    array = np.load(trailing_target / f"{name}.npy", allow_pickle=False)
    _replace_array_encoding(
        trailing_target,
        name,
        array,
        trailing=b"forged",
    )
    with pytest.raises(C5ScoreError, match="trailing bytes"):
        load_c5_raw_bundle(trailing_target)


@pytest.mark.parametrize(
    ("case", "match"),
    (
        ("dtype", "expected dtype/shape"),
        ("shape", "expected dtype/shape"),
        ("nonfinite", "non-finite"),
        ("signed-zero", "signed zero"),
    ),
)
def test_raw_loader_rejects_rebound_invalid_array_representations(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    name = "qpsim_qp_pair_gain_s_inv"
    value = np.load(target / f"{name}.npy", allow_pickle=False)
    if case == "dtype":
        replacement = value.astype(np.float32)
    elif case == "shape":
        replacement = value[:-1].copy()
    else:
        replacement = value.copy()
        if case == "nonfinite":
            replacement[100] = np.inf
        else:
            assert replacement[0] == 0.0
            replacement[0] = -0.0
    _replace_arrays(target, {name: replacement})
    with pytest.raises(C5ScoreError, match=match):
        load_c5_raw_bundle(target)


def test_raw_loader_detects_directory_toctou(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = _clone_raw_bundle(tmp_path)
    original = c5_score._read_regular_file_once
    mutated = False

    def mutating_read(path: Path, label: str) -> bytes:
        nonlocal mutated
        result = original(path, label)
        if not mutated and label == "C5 raw parent_f.npy":
            (target / "race.txt").write_text("changed", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(c5_score, "_read_regular_file_once", mutating_read)
    with pytest.raises(C5ScoreError, match="changed during C5 verification"):
        load_c5_raw_bundle(target)


def test_score_builder_detects_raw_directory_change_after_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The whole build must retain the raw-directory snapshot until return."""

    target = _clone_raw_bundle(tmp_path)
    target_resolved = target.resolve()
    original = c5_score.load_c5_raw_bundle
    mutated = False

    def mutate_after_load(
        bundle_dir: Path,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray], str]:
        nonlocal mutated
        result = original(bundle_dir)
        if not mutated and Path(bundle_dir).resolve() == target_resolved:
            (target / "post-load-race.txt").write_text("changed", encoding="utf-8")
            mutated = True
        return result

    monkeypatch.setattr(c5_score, "load_c5_raw_bundle", mutate_after_load)
    with pytest.raises(
        C5ScoreError,
        match=r"selected C5 raw bundle changed during C5 verification",
    ):
        _build_score(target)
    assert mutated


@pytest.mark.parametrize(
    "case",
    (
        "wrong-k",
        "wrong-map",
        "wrong-n",
        "wrong-temperature",
        "wrong-tau",
        "wrong-units",
        "wrong-loss-semantics",
        "wrong-pauli-rebucketing",
        "pair-factor-two",
        "channel-swap",
        "equal-gain-loss-shift",
        "phonon-mutation",
        "residual-closure",
    ),
)
def test_independent_replay_rejects_fully_rebound_semantic_forgeries(
    raw_evidence: RawEvidence,
    tmp_path: Path,
    case: str,
) -> None:
    metadata, base_arrays, _manifest_sha = raw_evidence
    target = _clone_raw_bundle(tmp_path)
    replacements: dict[str, np.ndarray] = {}
    metadata_mutation: Callable[[dict[str, Any]], None] | None = None

    if case == "wrong-k":
        kernel = base_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"] * 1.0001
        gain, rate = _evaluate_channel(
            metadata,
            base_arrays,
            "scattering",
            kernel,
        )
        replacements = _channel_family(
            base_arrays,
            "scattering",
            gain,
            rate,
            kernel=kernel,
        )
    elif case == "wrong-map":
        mapping = base_arrays["qpsim_omega_idx_diff"].copy()
        mapping[20, 21] += 1
        replacements = {"qpsim_omega_idx_diff": mapping}
    elif case == "wrong-n":
        n_p = base_arrays["qpsim_N_p"].copy()
        n_p[20, 21] = np.nextafter(n_p[20, 21], np.inf)
        kernel = base_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"]
        gain, rate = _evaluate_channel(
            metadata,
            base_arrays,
            "scattering",
            kernel,
            n_p=n_p,
        )
        replacements = {"qpsim_N_p": n_p}
        replacements.update(
            _channel_family(
                base_arrays,
                "scattering",
                gain,
                rate,
            )
        )
    elif case == "wrong-temperature":

        def wrong_temperature(raw: dict[str, Any]) -> None:
            raw["operator_inputs"]["T_bath_K"] = _float_record(
                np.nextafter(
                    raw["operator_inputs"]["T_bath_K"]["value"],
                    np.inf,
                )
            )

        metadata_mutation = wrong_temperature
    elif case == "wrong-tau":
        tau = 439.0
        scale = 438.0 / tau
        scattering_kernel = base_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"] * scale
        pair_kernel = base_arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"] * scale
        gain_s, rate_s = _evaluate_channel(
            metadata,
            base_arrays,
            "scattering",
            scattering_kernel,
        )
        scattering = _channel_family(
            base_arrays,
            "scattering",
            gain_s,
            rate_s,
            kernel=scattering_kernel,
        )
        working = {**base_arrays, **scattering}
        gain_p, rate_p = _evaluate_channel(
            metadata,
            working,
            "pair",
            pair_kernel,
        )
        pair = _channel_family(
            working,
            "pair",
            gain_p,
            rate_p,
            kernel=pair_kernel,
        )
        replacements = {**scattering, **pair}

        def wrong_tau(raw: dict[str, Any]) -> None:
            raw["operator_inputs"]["tau_0_ns"] = _float_record(tau)
            raw["operator_inputs"]["tau_0_parent_s"] = _float_record(tau * SECONDS_PER_NS)

        metadata_mutation = wrong_tau
    elif case == "wrong-units":
        replacements = _wrong_unit_family(base_arrays, "pair")
    elif case == "wrong-loss-semantics":
        replacements = _channel_family(
            base_arrays,
            "pair",
            base_arrays["qpsim_qp_pair_gain_ns_inv"],
            base_arrays["qpsim_qp_pair_loss_rate_ns_inv"],
            physical_loss_ns=base_arrays["qpsim_qp_pair_loss_rate_ns_inv"],
        )
    elif case == "wrong-pauli-rebucketing":
        cross = base_arrays["scattering_pauli_cross_term_s_inv"] * 1.01
        parent_gain = base_arrays["parent_qp_scattering_gain_s_inv"] - cross
        parent_loss = base_arrays["parent_qp_scattering_loss_s_inv"] - cross
        replacements = {
            "scattering_pauli_cross_term_s_inv": _positive_zero(cross),
            "parent_qp_scattering_rebucketed_gain_s_inv": (_positive_zero(parent_gain)),
            "parent_qp_scattering_rebucketed_loss_s_inv": (_positive_zero(parent_loss)),
            "qp_scattering_rebucketed_delta_gain_s_inv": _positive_zero(
                base_arrays["qpsim_qp_scattering_gain_s_inv"] - parent_gain
            ),
            "qp_scattering_rebucketed_delta_loss_s_inv": _positive_zero(
                base_arrays["qpsim_qp_scattering_loss_s_inv"] - parent_loss
            ),
        }
    elif case == "pair-factor-two":
        kernel = 2.0 * base_arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"]
        gain, rate = _evaluate_channel(
            metadata,
            base_arrays,
            "pair",
            kernel,
        )
        replacements = _channel_family(
            base_arrays,
            "pair",
            gain,
            rate,
            kernel=kernel,
        )
    elif case == "channel-swap":
        scattering_kernel = base_arrays["qpsim_qp_pair_kernel_ns_inv_ueV_inv"]
        pair_kernel = base_arrays["qpsim_qp_scattering_kernel_ns_inv_ueV_inv"]
        gain_s, rate_s = _evaluate_channel(
            metadata,
            base_arrays,
            "scattering",
            scattering_kernel,
        )
        scattering = _channel_family(
            base_arrays,
            "scattering",
            gain_s,
            rate_s,
            kernel=scattering_kernel,
        )
        working = {**base_arrays, **scattering}
        gain_p, rate_p = _evaluate_channel(
            metadata,
            working,
            "pair",
            pair_kernel,
        )
        pair = _channel_family(
            working,
            "pair",
            gain_p,
            rate_p,
            kernel=pair_kernel,
        )
        replacements = {**scattering, **pair}
    elif case == "equal-gain-loss-shift":
        replacements = _equal_gain_loss_shift_family(
            base_arrays,
            "scattering",
        )
    elif case == "phonon-mutation":
        parent = base_arrays["parent_phonon_residual_s_inv"].copy()
        parent[100] = np.nextafter(parent[100], np.inf)
        replacements = {
            "parent_phonon_residual_s_inv": parent,
            "c5sp_phonon_residual_s_inv": parent.copy(),
        }
    else:
        residual = base_arrays["c5sp_qp_residual_s_inv"].copy()
        residual[100] = np.nextafter(residual[100], np.inf)
        replacements = {"c5sp_qp_residual_s_inv": residual}

    if replacements:
        _replace_arrays(target, replacements)
    if metadata_mutation is not None:
        _mutate_metadata(target, metadata_mutation)

    # Every file hash and descriptor now agrees with the hostile hypothesis.
    # The transport layer accepts it; independent replay must reject it.
    load_c5_raw_bundle(target)
    with pytest.raises(C5ScoreError):
        _build_score(target)


@pytest.mark.parametrize(
    ("c4_bundle", "c3_bundle", "c2_bundle"),
    (
        pytest.param(
            C3_BUNDLE,
            C3_BUNDLE,
            C2_BUNDLE,
            id="c3-raw-passed-as-c4",
        ),
        pytest.param(
            C4_BUNDLE,
            C2_BUNDLE,
            C2_BUNDLE,
            id="c2-raw-passed-as-c3",
        ),
        pytest.param(
            C4_BUNDLE,
            C3_BUNDLE,
            C3_BUNDLE,
            id="c3-raw-passed-as-c2",
        ),
    ),
)
def test_c5_refuses_wrong_raw_parent_chain(
    c4_bundle: Path,
    c3_bundle: Path,
    c2_bundle: Path,
) -> None:
    _require_external_c5()
    with pytest.raises(
        (C5ScoreError, C4ScoreError, C3ScoreError, C2BundleError),
    ):
        build_c5_score(
            _c5_bundle(),
            c4_bundle_dir=c4_bundle,
            c3_bundle_dir=c3_bundle,
            c2_bundle_dir=c2_bundle,
        )


def test_c5_refuses_forged_c4_score_and_matching_receipt() -> None:
    _require_external_c5()
    with tempfile.TemporaryDirectory(
        dir=REPOSITORY_ROOT / "tmp",
        prefix="c5-forged-c4-",
    ) as directory:
        root = Path(directory)
        score = json.loads(C4_SCORE.read_text(encoding="utf-8"))
        # Preserve JSON structure and all raw bindings while changing one
        # checked numerical result.
        record = score["conservation"]["public_photon"]["absolute_number_residual_ueV_s_inv"]
        record["value"] = float(np.nextafter(record["value"], np.inf))
        record["hex"] = float(record["value"]).hex()
        score_raw = _canonical_json(score)
        score_path = root / "c4-score.json"
        score_path.write_bytes(score_raw)
        receipt = json.loads(C4_RECEIPT.read_text(encoding="utf-8"))
        receipt["checked_score"]["file_sha256"] = hashlib.sha256(score_raw).hexdigest()
        receipt_path = root / "c4-receipt.json"
        receipt_path.write_bytes(_canonical_json(receipt))

        with pytest.raises((C5ScoreError, C4ScoreError)):
            build_c5_score(
                _c5_bundle(),
                c4_bundle_dir=C4_BUNDLE,
                c3_bundle_dir=C3_BUNDLE,
                c2_bundle_dir=C2_BUNDLE,
                c4_score_path=score_path,
                c4_receipt_path=receipt_path,
            )


def test_c5_refuses_wrong_c4_receipt() -> None:
    _require_external_c5()
    with pytest.raises((C5ScoreError, C4ScoreError)):
        build_c5_score(
            _c5_bundle(),
            c4_bundle_dir=C4_BUNDLE,
            c3_bundle_dir=C3_BUNDLE,
            c2_bundle_dir=C2_BUNDLE,
            c4_receipt_path=C3_RECEIPT,
        )


def test_score_builder_rejects_source_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_external_c5()
    # This package initializer was absent from the old hand-maintained list;
    # the closed v6 manifest must bind it like every other qpsim source.
    relative = "qpsim/__init__.py"
    target = REPOSITORY_ROOT / relative
    original = c5_score.canonical_source_bytes

    def drifted(path: Path) -> bytes:
        content = original(path)
        if path.resolve() == target.resolve():
            return content + b"\n# simulated C5 verifier source drift\n"
        return content

    assert relative in c5_score._SOURCE_BYTES_AT_IMPORT
    monkeypatch.setattr(c5_score, "canonical_source_bytes", drifted)
    with pytest.raises(
        C5ScoreError,
        match=r"C5 numerical source changed during execution: "
        r"qpsim/__init__\.py",
    ):
        _build_score(_c5_bundle())


@pytest.mark.parametrize("race_target", ("score", "receipt"))
def test_parent_score_or_receipt_toctou_is_detected(
    monkeypatch: pytest.MonkeyPatch,
    race_target: str,
) -> None:
    _require_external_c5()
    with tempfile.TemporaryDirectory(
        dir=REPOSITORY_ROOT / "tmp",
        prefix="c5-parent-race-",
    ) as directory:
        root = Path(directory)
        score_path = root / "c4-score.json"
        receipt_path = root / "c4-receipt.json"
        score_path.write_bytes(C4_SCORE.read_bytes())
        receipt_path.write_bytes(C4_RECEIPT.read_bytes())
        original = c5_score.build_c4_score
        calls = 0

        def mutate_after_replay(
            c4_bundle_dir: Path,
            **kwargs: Any,
        ) -> dict[str, Any]:
            nonlocal calls
            result = original(c4_bundle_dir, **kwargs)
            calls += 1
            if calls == 1:
                selected = score_path if race_target == "score" else receipt_path
                selected.write_bytes(selected.read_bytes() + b"\n")
            return result

        monkeypatch.setattr(c5_score, "build_c4_score", mutate_after_replay)
        with pytest.raises(
            C5ScoreError,
            match=r"changed during C5 verification|changed during C5",
        ):
            build_c5_score(
                _c5_bundle(),
                c4_bundle_dir=C4_BUNDLE,
                c3_bundle_dir=C3_BUNDLE,
                c2_bundle_dir=C2_BUNDLE,
                c4_score_path=score_path,
                c4_receipt_path=receipt_path,
            )
