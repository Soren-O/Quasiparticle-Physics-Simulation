"""Config+code-hashed disk cache for expensive validation sweeps.

The paper-track sweeps are slow (Fischer Fig. 6 ~14 h, Figs 3/5 tens of
minutes). When nothing that affects the result has changed, recomputing is
pure waste. This module caches a sweep's *raw solve output* on disk, keyed by
a hash of everything that determines that output, so an unchanged rerun loads
in milliseconds while any meaningful change recomputes.

Safety is the whole game: a cache is only as correct as its key, and a
**config-only** key is unsafe. A solver change can shift results with zero
change to the physics config — e.g. commit ``f041a85`` moved Fischer Fig. 7
σ₁/Q by ~0.5 % while ``config_metadata()`` was byte-identical; a config-keyed
cache would have silently served the pre-fix numbers. So the key folds in:

* the figure id and the ``run()`` kwargs (which points / which grid),
* a **solver fingerprint** (the resolved physics + solver knobs),
* a **solve-source digest** — a content hash of every ``qpsim/**/*.py`` file
  *except* ``qpsim/observables/**`` (the cheap downstream derivations:
  ac_conductivity, quality_factor, gap_suppression, …). Editing an observable
  or plotting routine therefore keeps the cache warm, while *any* solver /
  collision / physics edit correctly invalidates it,
* the figure's own solve-path source (passed as ``extra_source``),
* numpy / scipy / Python versions, and a cache-format version.

Over-invalidation (recompute) is the deliberately-safe failure direction; a
stale serve is never acceptable, so the source digest is conservative (it
hashes the whole solver subtree, not a per-figure import closure).

The cache serves the **regen / dev** path only. The ``@pytest.mark.slow``
regression tests call the pure ``run()`` so they always truly recompute and
re-verify against the pinned baseline.

Controls
--------
``QPSIM_SWEEP_CACHE=0`` (or false/no/off) disables caching entirely.
``QPSIM_SWEEP_CACHE_DIR=/path`` relocates the store (default
``validation/.sweep_cache/``).

CLI
---
``python -m validation.sweep_cache clear [--figure FIG]`` — drop entries.
``python -m validation.sweep_cache list`` — show stored entries.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np

# Bump to invalidate every existing entry on a key-schema / codec change.
_FORMAT_VERSION = 1

_ENV_ENABLE = "QPSIM_SWEEP_CACHE"
_ENV_DIR = "QPSIM_SWEEP_CACHE_DIR"

# qpsim subpackage holding the cheap downstream observable derivations; excluded
# from the solve-source digest so editing it does not invalidate solve caches.
_OBSERVABLES_PKG = "observables"

_DISABLED_VALUES = {"0", "false", "no", "off", ""}

# Figure ids are logical namespaces (for example ``fischer_2023/fig7``), not
# filesystem paths. Each namespace component and every content-addressed key
# must therefore use a deliberately small, portable filename alphabet.
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def _is_safe_component(value: str) -> bool:
    """Whether ``value`` is one portable, non-aliasing path component."""
    stem = value.split(".", 1)[0].upper()
    return bool(
        _SAFE_COMPONENT.fullmatch(value)
        and not value.endswith(".")
        and stem not in _WINDOWS_RESERVED_NAMES
    )


def _repo_root() -> Path:
    # <root>/validation/sweep_cache.py -> parents[1] == <root>
    return Path(__file__).resolve().parents[1]


def is_enabled() -> bool:
    """Whether caching is active. Default on; disabled via ``QPSIM_SWEEP_CACHE``."""
    val = os.environ.get(_ENV_ENABLE)
    if val is None:
        return True
    return val.strip().lower() not in _DISABLED_VALUES


def default_cache_dir() -> Path:
    """Cache root: ``QPSIM_SWEEP_CACHE_DIR`` if set, else ``validation/.sweep_cache``."""
    override = os.environ.get(_ENV_DIR)
    if override:
        return Path(override).expanduser()
    return _repo_root() / "validation" / ".sweep_cache"


def solve_source_digest(qpsim_root: Path | None = None) -> str:
    """SHA-256 over all ``qpsim/**/*.py`` except the ``observables`` subpackage.

    The relative path is folded in alongside each file's bytes so that moving or
    renaming a module also changes the digest. Conservative by design: it hashes
    the entire solver subtree rather than a per-figure import closure, trading a
    little over-invalidation for the guarantee that no solve-relevant edit is
    ever missed.
    """
    if qpsim_root is None:
        qpsim_root = _repo_root() / "qpsim"
    h = hashlib.sha256()
    files = sorted(
        p
        for p in qpsim_root.rglob("*.py")
        if p.relative_to(qpsim_root).parts[0] != _OBSERVABLES_PKG
    )
    for p in files:
        rel = p.relative_to(qpsim_root).as_posix()
        h.update(rel.encode())
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _lib_versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import scipy

        versions["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover - scipy is a hard dep in practice
        versions["scipy"] = "unknown"
    return versions


def _canonical(obj: Any) -> Any:
    """Deterministic, JSON-safe canonicalization for hashing."""
    if isinstance(obj, Mapping):
        return {str(k): _canonical(obj[k]) for k in sorted(obj, key=str)}
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [_canonical(x) for x in sorted(obj, key=str)]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def cache_key(
    figure: str,
    fingerprint: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    *,
    extra_source: str = "",
    qpsim_root: Path | None = None,
) -> str:
    """The content-addressed key for a sweep result.

    Parameters
    ----------
    figure
        Stable identifier, e.g. ``"fischer_2023/fig7"``.
    fingerprint
        Resolved physics + solver knobs (a figure's ``solver_fingerprint()``).
    kwargs
        The ``run()`` kwargs that select what is solved.
    extra_source
        The figure's own solve-path source (e.g. ``inspect.getsource(solve)``
        joined with its helpers), so figure-side solver edits invalidate too —
        while its plotting / observable code, omitted here, does not.
    qpsim_root
        Override the library root (tests point this at a temp tree).
    """
    payload = {
        "format_version": _FORMAT_VERSION,
        "figure": figure,
        "fingerprint": _canonical(fingerprint),
        "kwargs": _canonical(kwargs),
        "solve_source": solve_source_digest(qpsim_root),
        "extra_source": hashlib.sha256(extra_source.encode()).hexdigest(),
        "versions": _lib_versions(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _validate_figure_id(figure: str) -> str:
    """Return a safe directory name for a logical figure identifier.

    Forward slashes are supported only as the documented logical namespace
    delimiter; they are encoded as ``__`` before any path is constructed.
    That token is consequently reserved inside components so the encoding is
    injective (``a/b`` must not alias a literal ``a__b`` namespace). Native
    or foreign path syntax, empty components, and traversal are rejected.
    """
    if not figure or "\\" in figure:
        raise ValueError(f"Unsafe sweep-cache figure id: {figure!r}")
    if PurePosixPath(figure).is_absolute() or PureWindowsPath(figure).is_absolute():
        raise ValueError(f"Unsafe sweep-cache figure id: {figure!r}")
    parts = figure.split("/")
    if any(not _is_safe_component(part) or "__" in part for part in parts):
        raise ValueError(f"Unsafe sweep-cache figure id: {figure!r}")
    return "__".join(parts)


def _validate_key(key: str) -> str:
    """Validate a cache key as one portable filename component."""
    if not _is_safe_component(key):
        raise ValueError(f"Unsafe sweep-cache key: {key!r}")
    return key


def _cache_root(cache_dir: Path) -> Path:
    """Resolve the cache root without requiring it to exist yet."""
    return cache_dir.expanduser().resolve(strict=False)


def _contained_path(root: Path, candidate: Path) -> Path:
    """Resolve ``candidate`` and prove that it is below ``root``."""
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Sweep-cache path escapes cache root: {candidate}") from exc
    if resolved == root:
        raise ValueError("Sweep-cache entry may not be the cache root itself")
    return resolved


def _figure_dir(cache_dir: Path, figure: str) -> Path:
    root = _cache_root(cache_dir)
    return _contained_path(root, root / _validate_figure_id(figure))


def _entry_paths(cache_dir: Path, figure: str, key: str) -> tuple[Path, Path]:
    root = _cache_root(cache_dir)
    d = _figure_dir(root, figure)
    safe_key = _validate_key(key)
    return (
        _contained_path(root, d / f"{safe_key}.npz"),
        _contained_path(root, d / f"{safe_key}.meta.json"),
    )


def load(
    figure: str, key: str, *, cache_dir: Path | None = None
) -> dict[str, np.ndarray] | None:
    """Return the stored arrays for ``key`` or ``None`` on a miss.

    A corrupt or partially-written entry is treated as a miss (recompute), never
    raised — the cache must never be able to break a solve.
    """
    cache_dir = cache_dir or default_cache_dir()
    npz_path, _ = _entry_paths(cache_dir, figure, key)
    if not npz_path.exists():
        return None
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            return {name: z[name] for name in z.files}
    except Exception:
        return None


def store(
    figure: str,
    key: str,
    arrays: Mapping[str, np.ndarray],
    *,
    provenance: Mapping[str, Any] | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Write ``arrays`` (the solve payload) atomically; record provenance sidecar.

    The ``.npz`` is written to a temp file in the same directory and atomically
    renamed, so an interrupted write (e.g. a killed 14 h solve) can never leave a
    half-written entry that loads as a false hit.
    """
    cache_dir = cache_dir or default_cache_dir()
    npz_path, meta_path = _entry_paths(cache_dir, figure, key)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir so a pre-existing figure-directory symlink cannot
    # redirect either file outside the configured cache root.
    npz_path, meta_path = _entry_paths(cache_dir, figure, key)

    fd, tmp_name = tempfile.mkstemp(dir=npz_path.parent, suffix=".npz.tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            # Pass a file object so np.savez does not append a second ".npz".
            # numpy's savez stub mistypes **kwds as bool; the values are arrays.
            np.savez(fh, **dict(arrays))  # type: ignore[arg-type]
        os.replace(tmp_name, npz_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise

    if provenance is not None:
        meta_path.write_text(json.dumps(_canonical(provenance), indent=2, sort_keys=True))
    return npz_path


def cached_solve(
    figure: str,
    solve_fn: Callable[[], Mapping[str, np.ndarray]],
    *,
    fingerprint: Mapping[str, Any],
    kwargs: Mapping[str, Any] | None = None,
    extra_source: str = "",
    cache_dir: Path | None = None,
    qpsim_root: Path | None = None,
) -> dict[str, np.ndarray]:
    """Return the cached solve payload, computing and storing it on a miss.

    When disabled (``QPSIM_SWEEP_CACHE=0``) this is a transparent pass-through:
    ``solve_fn()`` runs and nothing is read or written.
    """
    kwargs = dict(kwargs or {})
    if not is_enabled():
        return dict(solve_fn())

    cache_dir = cache_dir or default_cache_dir()
    key = cache_key(
        figure, fingerprint, kwargs, extra_source=extra_source, qpsim_root=qpsim_root
    )
    hit = load(figure, key, cache_dir=cache_dir)
    if hit is not None:
        return hit

    arrays = dict(solve_fn())
    provenance = {
        "figure": figure,
        "key": key,
        "created_utc": datetime.now(UTC).isoformat(),
        "fingerprint": dict(fingerprint),
        "kwargs": kwargs,
        "versions": _lib_versions(),
        "solve_source": solve_source_digest(qpsim_root),
    }
    store(figure, key, arrays, provenance=provenance, cache_dir=cache_dir)
    return arrays


def clear(*, figure: str | None = None, cache_dir: Path | None = None) -> None:
    """Remove cached entries — one figure's subdir, or the whole cache."""
    cache_dir = cache_dir or default_cache_dir()
    target = _figure_dir(cache_dir, figure) if figure is not None else _cache_root(cache_dir)
    if target.exists():
        shutil.rmtree(target)


def _iter_entries(cache_dir: Path) -> Iterator[tuple[str, Path]]:
    if not cache_dir.exists():
        return
    for fig_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        for npz in sorted(fig_dir.glob("*.npz")):
            yield fig_dir.name, npz


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="validation.sweep_cache")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_clear = sub.add_parser("clear", help="remove cached entries")
    p_clear.add_argument("--figure", default=None, help="only this figure's entries")
    sub.add_parser("list", help="list stored entries")
    args = parser.parse_args(argv)

    cache_dir = default_cache_dir()
    if args.cmd == "clear":
        clear(figure=args.figure)
        scope = args.figure or "all figures"
        print(f"cleared sweep cache ({scope}) under {cache_dir}")
        return 0
    if args.cmd == "list":
        entries = list(_iter_entries(cache_dir))
        if not entries:
            print(f"(empty) {cache_dir}")
            return 0
        for figure, npz in entries:
            size_kb = npz.stat().st_size / 1024.0
            print(f"{figure:28s} {npz.stem[:16]}…  {size_kb:8.1f} KB")
        return 0
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
