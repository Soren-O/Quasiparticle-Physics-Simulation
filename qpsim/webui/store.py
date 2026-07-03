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


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "setup"


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Atomic replace: run manifests are re-written by the worker thread
    # while request handlers read them; a plain write_text would let a
    # reader see a half-written file. On Windows the replace itself
    # fails with a sharing violation while a reader briefly holds the
    # target open (CPython opens without FILE_SHARE_DELETE), so retry.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for _ in range(40):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.025)
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
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
        slug = slug or slugify(envelope.name)
        _write_json(
            self.setups_dir / f"{slug}.json",
            {
                "name": envelope.name,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
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
        data = _read_json(self.setups_dir / f"{slug}.json")
        return SetupEnvelope.model_validate(
            {"name": data.get("name", slug), "setup": data["setup"]}
        )

    def delete_setup(self, slug: str) -> None:
        (self.setups_dir / f"{slug}.json").unlink(missing_ok=True)

    # -- runs -----------------------------------------------------------

    def new_run_id(self) -> str:
        return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def write_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        _write_json(self.run_dir(run_id) / "manifest.json", manifest)

    def read_manifest(self, run_id: str) -> dict[str, Any]:
        return _read_json(self.run_dir(run_id) / "manifest.json")

    def write_arrays(self, run_id: str, arrays: dict[str, np.ndarray]) -> None:
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        # numpy's stub types the **kwds of savez_compressed as the
        # allow_pickle flag; the call itself is the documented form.
        np.savez_compressed(directory / "result.npz", **arrays)  # type: ignore[arg-type]

    def read_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        with np.load(self.run_dir(run_id) / "result.npz", allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}

    def list_runs(self) -> list[dict[str, Any]]:
        """Run manifests, newest first (run ids sort chronologically)."""
        manifests = []
        if self.runs_dir.is_dir():
            for directory in sorted(self.runs_dir.iterdir(), reverse=True):
                manifest_path = directory / "manifest.json"
                if not manifest_path.is_file():
                    continue
                try:
                    manifests.append(_read_json(manifest_path))
                except (OSError, ValueError, json.JSONDecodeError):
                    continue
        return manifests

    def delete_run(self, run_id: str) -> None:
        directory = self.run_dir(run_id)
        if directory.is_dir():
            for child in directory.iterdir():
                child.unlink(missing_ok=True)
            directory.rmdir()
