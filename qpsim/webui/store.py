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

import hashlib
import json
import re
import threading
import time
import uuid
import zipfile
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format

from qpsim.materials.database import _LEGACY_RHO_F_MAX
from qpsim.webui.schemas import SetupEnvelope, canonical_mode

_SETUP_SCHEMA_VERSION = 2

# Versionless (v1) setup files predate the schema_version key and are
# ambiguous: the shipped webui always wrote rho_F in eV^-1 m^-3
# (Al 1.74e28), while a short-lived intermediate build wrote µeV^-1 m^-3
# (Al 1.74e22). Values below this cutoff get the x1e6 migration; values at
# or above it are already on the eV contract and must pass through
# untouched.
_RHO_F_MIGRATION_CUTOFF_EV = _LEGACY_RHO_F_MAX
_RUN_STATUSES = {"queued", "running", "done", "failed", "cancelled"}
_MAX_PATH_SEGMENT_LENGTH = 120
_MAX_AUTO_SLUG_BASE_LENGTH = 96


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    slug = slug or "setup"
    if len(slug) > _MAX_AUTO_SLUG_BASE_LENGTH:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
        slug = f"{slug[:_MAX_AUTO_SLUG_BASE_LENGTH].rstrip('-')}-{digest}"
    return slug


# A stored slug / run_id becomes a single path component. Anything with a
# separator, drive letter, or ``..`` could escape its directory once joined —
# on Windows a raw ``..\name`` segment survives Starlette's routing (only the
# ``/`` form is rejected) and resolves outside the workspace. Constrain every
# externally-supplied segment to this alphabet before it touches the filesystem.
_SAFE_SEGMENT = re.compile(r"[A-Za-z0-9._-]+")


def _safe_segment(value: str, kind: str) -> str:
    if (
        value in {".", ".."}
        or len(value) > _MAX_PATH_SEGMENT_LENGTH
        or not _SAFE_SEGMENT.fullmatch(value)
    ):
        raise ValueError(f"unsafe {kind}: {value!r}")
    return value


def _collision_safe_slug(directory: Path, name: str) -> str:
    """Return a stable automatic slug without overwriting another name."""
    base = slugify(name)
    candidate = base
    path = directory / f"{candidate}.json"
    if not path.exists():
        return candidate

    try:
        existing_name = _read_json(path).get("name")
    except (OSError, ValueError, json.JSONDecodeError):
        existing_name = None
    if existing_name == name:
        return candidate

    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:12]
    max_base = _MAX_PATH_SEGMENT_LENGTH - len(digest) - 1
    candidate = f"{base[:max_base].rstrip('-')}-{digest}"
    collision_path = directory / f"{candidate}.json"
    if collision_path.exists():
        try:
            collision_name = _read_json(collision_path).get("name")
        except (OSError, ValueError, json.JSONDecodeError):
            collision_name = None
        if collision_name not in {None, name}:
            raise ValueError(
                "setup-name hash collision; provide an explicit unique slug."
            )
    return candidate


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


def _unlink_with_retry(path: Path) -> None:
    """Unlink a file despite brief Windows reader sharing violations."""
    for _ in range(40):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.025)
    path.unlink(missing_ok=True)


def _rmdir_with_retry(path: Path) -> None:
    """Remove an empty directory after transient Windows handles close."""
    for _ in range(40):
        try:
            path.rmdir()
            return
        except PermissionError:
            time.sleep(0.025)
    path.rmdir()


class LazyArrays(Mapping[str, np.ndarray]):
    """A run's NPZ as a read-only mapping that inflates on first access.

    Behaves like the dict the plot and CSV layers already expect -- they do
    ``arrays["snap_f"]``, ``"mask" in arrays`` and iterate the keys -- so it
    substitutes without touching them. The zip is opened per member rather
    than held: a lingering handle is what makes delete_run fail on Windows.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._cache: dict[str, np.ndarray] = {}
        with zipfile.ZipFile(path) as archive:
            self._names = tuple(
                sorted(
                    member[:-4]
                    for member in archive.namelist()
                    if member.endswith(".npy")
                )
            )

    def __getitem__(self, name: str) -> np.ndarray:
        if name in self._cache:
            return self._cache[name]
        if name not in self._names:
            raise KeyError(name)
        with np.load(self._path, allow_pickle=False) as data:
            value = np.asarray(data[name])
        self._cache[name] = value
        return value

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return name in self._names


def _write_json(path: Path, data: dict[str, Any]) -> None:
    # Atomic replace: run manifests are re-written by the worker thread
    # while request handlers read them; a plain write_text would let a
    # reader see a half-written file.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-writer unique tmp: two concurrent saves of the same slug would
    # otherwise race on one fixed ``<name>.tmp`` and clobber each other
    # (spurious PermissionError / FileNotFoundError). The final replace stays
    # atomic, so this is last-writer-wins without spurious failures.
    tmp = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(json_sanitize(data), indent=2), encoding="utf-8")
    try:
        _replace_with_retry(tmp, path)
    except BaseException:
        # The tmp name carries a fresh uuid, so a failure that repeats leaves a
        # NEW orphan every time rather than overwriting one. A manifest that
        # cannot be replaced (read-only attribute, an indexer or backup agent
        # holding a handle) is retried from overlay() on every runs poll, i.e.
        # every two seconds for as long as a browser tab is open, and nothing
        # but delete_run ever cleans these up. Same guard the NPZ writer
        # already uses.
        _unlink_with_retry(tmp)
        raise


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


def _read_manifest(path: Path) -> dict[str, Any]:
    """Read and structurally validate one run manifest.

    Syntactically valid JSON is not necessarily a usable manifest. The API
    indexes these fields directly and the UI relies on their types, so treating
    ``{}`` as readable only moves the failure to a request-time ``KeyError``.
    """
    manifest = _read_json(path)
    required_types: dict[str, type | tuple[type, ...]] = {
        "id": str,
        "name": str,
        "mode": str,
        "status": str,
        "created": str,
        "setup": dict,
        "summary": dict,
        "notes": list,
    }
    for key, expected_type in required_types.items():
        if key not in manifest or not isinstance(manifest[key], expected_type):
            raise ValueError(
                f"{path} has an invalid or missing manifest field {key!r}."
            )
    if not manifest["id"] or manifest["id"] != path.parent.name:
        raise ValueError(
            f"{path} manifest id does not match its run directory."
        )
    if manifest["status"] not in _RUN_STATUSES:
        raise ValueError(
            f"{path} has unknown run status {manifest['status']!r}."
        )
    if not all(isinstance(note, str) for note in manifest["notes"]):
        raise ValueError(f"{path} manifest notes must be strings.")
    return manifest


@dataclass
class Workspace:
    """Filesystem-backed store for setups and runs."""

    root: Path
    _setup_save_lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

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
        if slug is None:
            # Automatic slug selection is a read-then-write transaction.
            # Serialize it across request threads so colliding names cannot
            # both observe the base slug as free and overwrite one another.
            # Explicit slugs retain caller-controlled last-writer-wins
            # behavior.
            with self._setup_save_lock:
                slug = _collision_safe_slug(self.setups_dir, envelope.name)
                slug = _safe_segment(slug, "slug")
                _write_json(
                    self.setups_dir / f"{slug}.json",
                    {
                        "name": envelope.name,
                        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "schema_version": _SETUP_SCHEMA_VERSION,
                        "setup": envelope.setup.model_dump(),
                        # The benchmark is part of what the setup IS -- a case
                        # whose point is "score this against the diffusion
                        # closed form" comes back as an unscored setup without
                        # it, and silently, which is worse than failing to save.
                        "benchmark": envelope.benchmark,
                    },
                )
                return slug
        slug = _safe_segment(slug, "slug")
        _write_json(
            self.setups_dir / f"{slug}.json",
            {
                "name": envelope.name,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "schema_version": _SETUP_SCHEMA_VERSION,
                "setup": envelope.setup.model_dump(),
                "benchmark": envelope.benchmark,   # see the note above
            },
        )
        return slug

    def list_setups(self) -> list[dict[str, Any]]:
        entries = []
        if self.setups_dir.is_dir():
            for path in sorted(self.setups_dir.glob("*.json")):
                try:
                    data = _read_json(path)
                    setup_data = data.get("setup")
                    if not isinstance(setup_data, dict):
                        raise ValueError(
                            f"{path} has an invalid or missing 'setup' object."
                        )
                    name = data.get("name", path.stem)
                    if not isinstance(name, str):
                        raise ValueError(f"{path} has a non-string setup name.")
                    mode = setup_data.get("mode", "?")
                    if not isinstance(mode, str):
                        raise ValueError(f"{path} has a non-string setup mode.")
                    # Report the CURRENT name. This listing reads the raw JSON
                    # rather than parsing an envelope, so without this a saved
                    # setup would be listed under a mode the picker no longer
                    # offers while still loading and running perfectly.
                    mode = canonical_mode(mode)
                    entries.append(
                        {
                            "slug": path.stem,
                            "name": name,
                            "mode": mode,
                            "saved_at": data.get("saved_at", ""),
                        }
                    )
                except (OSError, ValueError, json.JSONDecodeError):
                    entries.append({"slug": path.stem, "name": path.stem, "mode": "unreadable"})
        return entries

    def load_setup(self, slug: str) -> SetupEnvelope:
        slug = _safe_segment(slug, "slug")
        data = _read_json(self.setups_dir / f"{slug}.json")
        setup_data = data.get("setup")
        if not isinstance(setup_data, dict):
            raise ValueError(
                f"setup {slug!r} has an invalid or missing 'setup' object."
            )
        version_raw = data.get("schema_version", 1)
        if type(version_raw) is not int or not (1 <= version_raw <= _SETUP_SCHEMA_VERSION):
            raise ValueError(
                "unsupported setup schema_version "
                f"{version_raw!r}; supported versions are 1..{_SETUP_SCHEMA_VERSION}"
            )
        version = version_raw
        if version < 2:
            # v1 files are ambiguous between the shipped eV^-1 m^-3 contract
            # and the short-lived µeV^-1 m^-3 build — see
            # _RHO_F_MIGRATION_CUTOFF_EV. Migrate by magnitude, not blindly.
            material = setup_data.get("material")
            if isinstance(material, dict) and "rho_F" in material:
                rho_f = float(material["rho_F"])
                if rho_f < _RHO_F_MIGRATION_CUTOFF_EV:
                    material["rho_F"] = rho_f * 1.0e6
        return SetupEnvelope.model_validate(
            # `benchmark` is absent from every file written before it was
            # persisted, so read it with a default rather than requiring it.
            {
                "name": data.get("name", slug),
                "setup": setup_data,
                "benchmark": data.get("benchmark"),
            }
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
        return _read_manifest(self.run_dir(run_id) / "manifest.json")

    def write_arrays(self, run_id: str, arrays: dict[str, np.ndarray]) -> None:
        safe_arrays: dict[str, np.ndarray] = {}
        for name, value in arrays.items():
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(
                    f"result array {name!r} has object dtype, which cannot be "
                    "loaded safely with allow_pickle=False."
                )
            safe_arrays[name] = array
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        # Same atomic discipline as the manifests: a killed writer must
        # not leave a truncated result.npz at the final path. The tmp
        # name must end in .npz or savez appends another extension; the
        # uuid makes it unique so concurrent same-run writers don't collide.
        tmp = directory / f"result.{uuid.uuid4().hex}.tmp.npz"
        # numpy's stub types the **kwds of savez_compressed as the
        # allow_pickle flag; the call itself is the documented form.
        try:
            np.savez_compressed(tmp, **safe_arrays)  # type: ignore[arg-type]
            _replace_with_retry(tmp, directory / "result.npz")
        except BaseException:
            _unlink_with_retry(tmp)
            raise

    def read_arrays(self, run_id: str) -> dict[str, np.ndarray]:
        with np.load(self.run_dir(run_id) / "result.npz", allow_pickle=False) as data:
            return {name: np.asarray(data[name]) for name in data.files}

    def open_arrays(self, run_id: str) -> LazyArrays:
        """The payload as a mapping that inflates members ON ACCESS.

        A single-frame PNG indexes one array and reads one slice of it, but
        went through read_arrays, which materialises EVERY array in the file.
        The frontend then preloads one request per frame per figure family the
        moment a run detail opens, so a 2-D run with three families issued 3N
        full-payload decompressions. Lazy access does not make a member cheaper
        -- a deflate stream has no random access, so a frame still costs its
        whole array -- but it stops a phonon-field figure paying for the
        quasiparticle stack and vice versa.
        """
        return LazyArrays(self.run_dir(run_id) / "result.npz")

    def delete_arrays(self, run_id: str) -> None:
        """Remove a result payload, if present, without deleting its manifest."""
        _unlink_with_retry(self.run_dir(run_id) / "result.npz")

    def array_names(self, run_id: str) -> set[str]:
        """Array names from the NPZ zip directory — no decompression."""
        with np.load(self.run_dir(run_id) / "result.npz", allow_pickle=False) as data:
            return set(data.files)

    def array_shapes(self, run_id: str, names: Collection[str]) -> dict[str, tuple[int, ...]]:
        """Shapes of the NAMED arrays only, from their .npy HEADERS.

        The docstring already said "decompresses just those members", but
        ``data[name].shape`` inflates each member in full to build an array and
        then reads one attribute off it -- and this sits on the two-second
        run-detail poll, whose only use for the answer is the scrubber's frame
        range, a number that cannot change for a finished run. On a 72 MB
        payload that measured ~1 s of CPU and ~100 MB of transient allocation
        every two seconds, forever.

        A .npy header is about a hundred bytes at the front of the member, so
        reading it inflates that much and stops.
        """
        path = self.run_dir(run_id) / "result.npz"
        shapes: dict[str, tuple[int, ...]] = {}
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            for name in names:
                member = f"{name}.npy"
                if member not in members:
                    continue
                with archive.open(member) as handle:
                    try:
                        version = npy_format.read_magic(handle)
                        if version == (1, 0):
                            shape, _fortran, _dtype = (
                                npy_format.read_array_header_1_0(handle)
                            )
                        elif version == (2, 0):
                            shape, _fortran, _dtype = (
                                npy_format.read_array_header_2_0(handle)
                            )
                        else:
                            # An unknown .npy revision is not worth guessing
                            # at; fall back rather than report a wrong shape.
                            raise ValueError(f"unsupported .npy version {version}")
                    except (ValueError, OSError):
                        with np.load(path, allow_pickle=False) as data:
                            shape = tuple(data[name].shape)
                shapes[name] = tuple(shape)
        return shapes

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
                    manifests.append(_read_manifest(manifest_path))
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
            # Delete the manifest last. If a concurrent Windows download
            # keeps result.npz open past the retry window, the run remains
            # visible and retryable instead of becoming a half-deleted,
            # orphaned directory that the UI cannot list.
            children = list(directory.iterdir())
            manifest = directory / "manifest.json"
            for child in children:
                if child != manifest:
                    _unlink_with_retry(child)
            _unlink_with_retry(manifest)
            _rmdir_with_retry(directory)
