"""Workspace persistence: named setups and completed runs.

Layout (created on demand under the chosen workspace directory):

.. code-block:: text

    <workspace>/
        setups/<slug>.json            # SetupEnvelope + created timestamp
        runs/<run_id>/manifest.json   # setup snapshot, status, summary, notes
        runs/<run_id>/result.npz      # the executor's array payload

Setups are human-readable JSON (the old app's convention). Run
manifests carry everything the UI lists and the result page shows
except the arrays themselves, which live in the compressed NPZ
sidecar and are only loaded to render plots or CSV exports.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qpsim.webui.schemas import SetupEnvelope

_SETUP_SCHEMA_VERSION = 2


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "setup"


# A stored slug / run_id becomes a single path component. Anything with a
# separator, drive letter, or ``..`` could escape its directory once joined —
# on Windows a raw ``..\name`` segment survives Starlette's routing (only the
# ``/`` form is rejected) and resolves outside the workspace. Constrain every
# externally-supplied segment to this alphabet before it touches the filesystem.
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


def _safe_segment(value: str, kind: str) -> str:
    if value in {".", ".."} or not _SAFE_SEGMENT.fullmatch(value):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def json_sanitize(value: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Strict JSON has no Infinity/NaN tokens: an ``inf`` observable (e.g.
    Q_i when σ₁ ≤ 0) written verbatim would make every later strict
    serialization of the manifest — FastAPI's JSONResponse included —
    raise, taking the whole runs listing down with it.
    """
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    return value


def _replace_with_retry(tmp: Path, path: Path) -> None:
    # On Windows the replace fails with a sharing violation while a
    # reader briefly holds the target open (CPython opens without
    # FILE_SHARE_DELETE), so retry before letting it raise.
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.025)
    tmp.replace(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Atomic replace: run manifests are re-written by the worker thread
    # while request handlers read them; a plain write_text would let a
    # reader see a half-written file.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_sanitize(data), indent=2), encoding="utf-8")
    _replace_with_retry(tmp, path)


def _read_text_with_retry(path: Path) -> str:
    # Read-side mirror of _replace_with_retry: on Windows a reader can
    # get ACCESS_DENIED for the instant a writer's replace holds the
    # target. FileNotFoundError passes straight through — only the
    # transient sharing violation is retried.
    for _ in range(40):
        try:
            return path.read_text(encoding="utf-8")
        except PermissionError:
            time.sleep(0.025)
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(_read_text_with_retry(path))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return loaded


@dataclass
class Workspace:
    """Filesystem-backed store for setups and runs."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def setups_dir(self) -> Path:
        return self.root / "setups"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    # -- setups ---------------------------------------------------------

    def save_setup(self, envelope: SetupEnvelope, *, slug: str | None = None) -> str:
        """Persist a named setup; returns the slug it was stored under."""
        slug = _safe_segment(slug or slugify(envelope.name), "slug")
        _write_json(
            self.setups_dir / f"{slug}.json",
            {
                "name": envelope.name,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "schema_version": _SETUP_SCHEMA_VERSION,
                "setup": envelope.setup.model_dump(),
            },
        )
        return slug

    def list_setups(self) -> list[dict[str, Any]]:
        entries = []
        if self.setups_dir.is_dir():
            for path in sorted(self.setups_dir.glob("*.json")):
                try:
                    data = _read_json(path)
                    entries.append(
                        {
                            "slug": path.stem,
                            "name": data.get("name", path.stem),
                            "mode": data.get("setup", {}).get("mode", "?"),
                            "saved_at": data.get("saved_at", ""),
                        }
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    entries.append({"slug": path.stem, "name": path.stem, "mode": "unreadable"})
        return entries

    def load_setup(self, slug: str) -> SetupEnvelope:
        slug = _safe_segment(slug, "slug")
        data = _read_json(self.setups_dir / f"{slug}.json")
        version_raw = data.get("schema_version", 1)
        if type(version_raw) is not int or not (1 <= version_raw <= _SETUP_SCHEMA_VERSION):
            raise ValueError(
                "unsupported setup schema_version "
                f"{version_raw!r}; supported versions are 1..{_SETUP_SCHEMA_VERSION}"
            )
        version = version_raw
        setup_data = data["setup"]
        if version < 2:
            # Schema v1 persisted rho_F in µeV^-1 m^-3 (Al 1.74e22).
            # Schema v2 restores the conventional eV^-1 m^-3 contract, so
            # migrate stored values by 1e6.  This is keyed by the persisted
            # schema version, never by magnitude, and therefore does not
            # reinterpret a deliberately small v2 DOS.
            material = setup_data.get("material")
            if isinstance(material, dict) and "rho_F" in material:
                material["rho_F"] = float(material["rho_F"]) * 1.0e6
        return SetupEnvelope.model_validate(
            {"name": data.get("name", slug), "setup": setup_data}
        )

    def delete_setup(self, slug: str) -> None:
        slug = _safe_segment(slug, "slug")
        (self.setups_dir / f"{slug}.json").unlink(missing_ok=True)

    # -- runs -----------------------------------------------------------

    def new_run_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def run_dir(self, run_id: str) -> Path:
        # Single chokepoint for run_id -> path: every read/write/delete of a
        # run's manifest and arrays routes through here, so validating once
        # closes the traversal for all of them.
        return self.runs_dir / _safe_segment(run_id, "run_id")

    def write_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        _write_json(self.run_dir(run_id) / "manifest.json", manifest)

    def read_manifest(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "manifest.json")

    def write_arrays(self, run_id: str, arrays: dict[str, np.ndarray]) -> None:
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        # Same atomic discipline as the manifests: a killed writer must
        # not leave a truncated result.npz at the final path. The tmp
        # name must end in .npz or savez appends another extension.
        tmp = directory / "result.tmp.npz"
        # numpy's stub types the **kwds of savez_compressed as the
        # allow_pickle flag; the call itself is the documented form.
        np.savez_compressed(tmp, **arrays)  # type: ignore[arg-type]
        _replace_with_retry(tmp, directory / "result.npz")

    def read_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        with np.load(self.run_dir(run_id) / "result.npz", allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}

    def array_names(self, run_id: str) -> set[str]:
        """Array names from the NPZ zip directory — no decompression."""
        with np.load(self.run_dir(run_id) / "result.npz", allow_pickle=False) as data:
            return set(data.files)

    def list_runs(self) -> list[dict[str, Any]]:
        """Run manifests, newest first (run ids sort chronologically).

        A run whose manifest fails to parse stays visible as an
        ``unreadable`` placeholder — dropping it would strand an
        undeletable directory (its NPZ included) that the UI can
        never show or clean up.
        """
        manifests = []
        if self.runs_dir.is_dir():
            for directory in sorted(self.runs_dir.iterdir(), reverse=True):
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifests.append(_read_json(manifest_path))
                except (OSError, ValueError):
                    manifests.append(
                        {
                            "id": directory.name,
                            "name": directory.name,
                            "mode": "?",
                            "status": "unreadable",
                            "created": "",
                            "summary": {},
                            "notes": ["manifest.json is unreadable — delete and re-run."],
                            "error": None,
                            "elapsed_s": None,
                        }
                    )
        return manifests

    def delete_run(self, run_id: str) -> None:
        directory = self.run_dir(run_id)
        if directory.is_dir():
            for child in directory.iterdir():
                child.unlink(missing_ok=True)
            directory.rmdir()
