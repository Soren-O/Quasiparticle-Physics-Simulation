"""Focused adversarial checks for Figure 6 C2 raw-bundle closure."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest
from validation.fischer_2023.fig6_author_c2_bundle import (
    SCHEMA,
    C2BundleError,
    array_descriptor,
    load_c2_raw_bundle,
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


def _synthetic_c2_bundle(
    tmp_path: Path,
    *,
    arrays: dict[str, np.ndarray] | None = None,
) -> Path:
    target = tmp_path / "c2"
    target.mkdir()
    descriptors: dict[str, dict[str, object]] = {}
    files: dict[str, dict[str, object]] = {}
    for name, array in (arrays or {}).items():
        raw = _npy_bytes(array)
        filename = f"{name}.npy"
        (target / filename).write_bytes(raw)
        descriptors[name] = array_descriptor(array)
        files[filename] = {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        }
    manifest = {
        "files": files,
        "metadata": {
            "array_descriptors": descriptors,
            "coordinate_contract": {},
            "frozen_inputs": {},
            "parameter_plan": {},
            "parent_bindings": {},
            "schema": SCHEMA,
            "source_binding": {},
            "sources": {},
            "stage": {},
            "steps": [],
        },
        "schema": SCHEMA,
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return target


def test_loader_accepts_minimal_canonical_synthetic_bundle(tmp_path: Path) -> None:
    target = _synthetic_c2_bundle(
        tmp_path,
        arrays={"sample": np.array([0.0, 1.0], dtype=np.float64)},
    )

    _metadata, arrays, _manifest_sha = load_c2_raw_bundle(target)

    np.testing.assert_array_equal(arrays["sample"], [0.0, 1.0], strict=True)


def test_loader_rejects_unmanifested_regular_file(tmp_path: Path) -> None:
    target = _synthetic_c2_bundle(tmp_path)
    (target / "unmanifested.txt").write_text("undeclared", encoding="utf-8")

    with pytest.raises(C2BundleError, match="missing or extra entries"):
        load_c2_raw_bundle(target)


def test_loader_rejects_unmanifested_subdirectory(tmp_path: Path) -> None:
    target = _synthetic_c2_bundle(tmp_path)
    (target / "unmanifested").mkdir()

    with pytest.raises(C2BundleError, match="missing or extra entries"):
        load_c2_raw_bundle(target)


def test_loader_rejects_unmanifested_symlink(tmp_path: Path) -> None:
    target = _synthetic_c2_bundle(tmp_path)
    link = target / "unmanifested-link"
    try:
        link.symlink_to(target / "manifest.json")
    except OSError as exc:
        pytest.skip(f"Symlink creation is unavailable: {exc}")

    with pytest.raises(C2BundleError, match="missing or extra entries"):
        load_c2_raw_bundle(target)


def test_loader_rejects_trailing_bytes_after_rebound_file_record(
    tmp_path: Path,
) -> None:
    target = _synthetic_c2_bundle(
        tmp_path,
        arrays={"sample": np.array([0.0, 1.0], dtype=np.float64)},
    )
    array_path = target / "sample.npy"
    raw = array_path.read_bytes() + b"undeclared trailing payload"
    array_path.write_bytes(raw)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["sample.npy"] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(C2BundleError, match="canonical NPY encoding"):
        load_c2_raw_bundle(target)


def test_loader_rejects_python_equal_but_type_distinct_array_descriptor(
    tmp_path: Path,
) -> None:
    target = _synthetic_c2_bundle(
        tmp_path,
        arrays={"sample": np.empty((0,), dtype=np.float64)},
    )
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Python says ``False == 0``; provenance metadata must retain JSON types.
    manifest["metadata"]["array_descriptors"]["sample"]["shape"] = [False]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(C2BundleError, match="forged or stale"):
        load_c2_raw_bundle(target)
