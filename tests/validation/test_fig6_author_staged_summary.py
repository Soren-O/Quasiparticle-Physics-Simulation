from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023.fig6_author_staged_resolve import (
    SCHEMA as RAW_SCHEMA,
)
from validation.fischer_2023.fig6_author_staged_resolve import SOURCE_BINDING
from validation.fischer_2023.fig6_author_staged_summary import (
    ALL_RESULT_IDS,
    DEFAULT_PILOT,
    PILOT_SCHEMA,
    PILOT_STATUS,
    StagedResolvePilotError,
    build_staged_resolve_pilot,
    canonical_pilot_bytes,
    load_staged_resolve_pilot,
)
from validation.source_provenance import source_sha256


def test_checked_staged_resolve_pilot_is_strict_and_supplemental() -> None:
    if not DEFAULT_PILOT.is_file():
        pytest.skip("Checked staged pilot is withheld until the audited producer is rerun.")
    pilot = load_staged_resolve_pilot(DEFAULT_PILOT)

    assert pilot["schema"] == PILOT_SCHEMA
    assert pilot["status"] == PILOT_STATUS
    assert pilot["staged_bundle"]["schema"] == RAW_SCHEMA
    assert set(pilot["stages"]) == ALL_RESULT_IDS
    assert set(pilot["final_states"]) == ALL_RESULT_IDS
    assert all(stage["converged"] for stage in pilot["stages"].values())
    assert (
        pilot["stages"]["c3c_native_cell_density"]["specification"][
            "density_convention"
        ]
        == "qpsim_cell_density_ueV"
    )
    assert set(pilot["path_dependence"]) == {
        "c3b_continuation_from_c3a",
        "c3c_continuation_from_c3b",
    }
    assert "does not mark C3" in pilot["qualification"]
    assert (
        pilot["raw_array_policy"]["availability"]
        == "external_non_redistributed_development_artifact"
    )


def test_checked_staged_resolve_pilot_rejects_extra_fields(tmp_path: Path) -> None:
    if not DEFAULT_PILOT.is_file():
        pytest.skip("Checked staged pilot is withheld until the audited producer is rerun.")
    pilot = load_staged_resolve_pilot(DEFAULT_PILOT)
    pilot["unreviewed_claim"] = True
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StagedResolvePilotError, match="fields are invalid"):
        load_staged_resolve_pilot(path)


def test_checked_staged_resolve_pilot_rejects_stale_pilot_id(
    tmp_path: Path,
) -> None:
    if not DEFAULT_PILOT.is_file():
        pytest.skip("Checked staged pilot is withheld until the audited producer is rerun.")
    pilot = load_staged_resolve_pilot(DEFAULT_PILOT)
    pilot["pilot_id"] = "fischer-catelani-2023-fig6-c3-single-point-pilot-v2"
    path = tmp_path / "pilot.json"
    path.write_text(
        json.dumps(pilot, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(StagedResolvePilotError, match="pilot ID is stale"):
        load_staged_resolve_pilot(path)


def test_staged_resolve_pilot_regenerates_byte_exactly_when_raw_bundle_available() -> None:
    raw = os.environ.get("QPSIM_FISCHER2023_FIG6_STAGED_RESOLVE_BUNDLE")
    if raw is None:
        pytest.skip(
            "Set QPSIM_FISCHER2023_FIG6_STAGED_RESOLVE_BUNDLE to the "
            "external staged bundle bound by the checked pilot for byte-exact "
            "regeneration."
        )

    regenerated = build_staged_resolve_pilot(Path(raw))
    assert canonical_pilot_bytes(regenerated) == DEFAULT_PILOT.read_bytes()


def _raw_bundle() -> Path:
    raw = os.environ.get("QPSIM_FISCHER2023_FIG6_STAGED_RESOLVE_BUNDLE")
    if raw is None:
        pytest.skip(
            "Set QPSIM_FISCHER2023_FIG6_STAGED_RESOLVE_BUNDLE to exercise "
            "raw staged-bundle adversarial tests."
        )
    return Path(raw)


def _copied_bundle(tmp_path: Path) -> Path:
    target = tmp_path / "staged"
    shutil.copytree(_raw_bundle(), target)
    # Adversarial semantic tests operate on a private copy. Rebind its declared
    # source closure to the current live tree so the deliberately forged
    # semantic field, not unrelated source staleness, is the rejection under
    # test.
    def rebind(payload: dict[str, object]) -> None:
        metadata = payload["metadata"]
        assert isinstance(metadata, dict)
        metadata["source_binding"] = dict(SOURCE_BINDING)
        sources = metadata["sources"]
        assert isinstance(sources, dict)
        repository_root = Path(__file__).resolve().parents[2]
        for relative in list(sources):
            sources[relative] = source_sha256(repository_root / relative)

    _mutate_manifest(target, rebind)
    return target


def _mutate_manifest(bundle: Path, mutation: object) -> None:
    path = bundle / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert callable(mutation)
    mutation(payload)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _replace_npy(
    bundle: Path,
    array_name: str,
    mutation: object,
) -> None:
    path = bundle / f"{array_name}.npy"
    with path.open("rb") as handle:
        array = np.lib.format.read_array(handle, allow_pickle=False)
    assert callable(mutation)
    mutation(array)
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        array,
        version=(3, 0),
        allow_pickle=False,
    )
    raw = stream.getvalue()
    path.write_bytes(raw)

    def update(payload: dict[str, object]) -> None:
        files = payload["files"]
        metadata = payload["metadata"]
        assert isinstance(files, dict)
        assert isinstance(metadata, dict)
        descriptors = metadata["array_descriptors"]
        assert isinstance(descriptors, dict)
        files[path.name] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
        descriptors[array_name] = {
            "dtype": array.dtype.str,
            "npy_sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(array.shape),
        }

    _mutate_manifest(bundle, update)


def test_raw_staged_summary_rejects_forged_stage_specification(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["metadata"]["stages"]["author_control"]["specification"][
            "pair_frequency_offset_bins"
        ] = 1

    _mutate_manifest(bundle, forge)
    with pytest.raises(StagedResolvePilotError, match="specification"):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_forged_c3c_density_convention(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["metadata"]["stages"]["c3c_native_cell_density"]["specification"][
            "density_convention"
        ] = "author_cell_average_eV"

    _mutate_manifest(bundle, forge)
    with pytest.raises(StagedResolvePilotError, match="specification"):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_incomplete_source_closure(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)

    def omit(payload: dict[str, object]) -> None:
        sources = payload["metadata"]["sources"]
        assert isinstance(sources, dict)
        sources.pop("qpsim/constants.py")

    _mutate_manifest(bundle, omit)
    with pytest.raises(StagedResolvePilotError, match="exact producer source closure"):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_large_unreported_final_step(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)

    def enlarge(array: np.ndarray) -> None:
        array[-2, 0] = array[-1, 0] + 0.1

    _replace_npy(bundle, "author_control__state_history", enlarge)
    with pytest.raises(
        StagedResolvePilotError,
        match=r"relative steps|Newton history transition",
    ):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_rehashed_intermediate_history(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)
    history_name = "c3c_native_cell_density__state_history"

    def corrupt_history(array: np.ndarray) -> None:
        array[1] = 2.0 * array[0]

    _replace_npy(bundle, history_name, corrupt_history)
    with (bundle / f"{history_name}.npy").open("rb") as handle:
        history = np.lib.format.read_array(handle, allow_pickle=False)
    size = (history.shape[1] + 1) // 2
    forged_steps = np.linalg.norm(
        history[1:, :size] - history[:-1, :size],
        axis=1,
    ) / np.linalg.norm(history[1:, :size], axis=1)

    def replace_steps(array: np.ndarray) -> None:
        array[:] = forged_steps

    _replace_npy(
        bundle,
        "c3c_native_cell_density__relative_qp_steps",
        replace_steps,
    )
    with pytest.raises(
        StagedResolvePilotError,
        match=r"Newton history transition|authenticated update",
    ):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_forged_certificate(tmp_path: Path) -> None:
    bundle = _copied_bundle(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["metadata"]["stages"]["author_control"][
            "cancellation_certificate"
        ]["qp"]["residual_l1_s_inv"] = 0.0

    _mutate_manifest(bundle, forge)
    with pytest.raises(StagedResolvePilotError, match="cancellation_certificate"):
        build_staged_resolve_pilot(bundle)


def test_raw_staged_summary_rejects_forged_nonnegative_bounds(
    tmp_path: Path,
) -> None:
    bundle = _copied_bundle(tmp_path)

    def forge(payload: dict[str, object]) -> None:
        payload["metadata"]["stages"]["author_control"]["final_bounds"]["f_min"] = -1.0

    _mutate_manifest(bundle, forge)
    with pytest.raises(StagedResolvePilotError, match="final_bounds"):
        build_staged_resolve_pilot(bundle)
