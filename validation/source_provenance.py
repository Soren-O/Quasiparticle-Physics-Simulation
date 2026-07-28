"""Portable hashing helpers for validation source provenance."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path


def canonical_source_bytes(path: Path) -> bytes:
    """Return source bytes with every text newline represented as LF.

    Git stores the validation sources with LF endings, but a checkout may use
    CRLF.  Provenance identifies logical source content, so checkout newline
    policy must not change a source fingerprint.
    """
    data = path.read_bytes()
    return data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def canonical_source_text(source: str) -> str:
    """Return in-memory source text with every newline represented as LF."""
    return source.replace("\r\n", "\n").replace("\r", "\n")


def source_sha256(path: Path) -> str:
    """Return the newline-independent SHA-256 of a source file."""
    return hashlib.sha256(canonical_source_bytes(path)).hexdigest()


def source_manifest(
    validation_module: Path,
    *,
    extra_validation_modules: Sequence[Path] = (),
) -> dict[str, str]:
    """Hash one validation producer and the complete numerical source tree.

    The manifest deliberately over-invalidates: every qpsim Python module and
    material YAML file is included, together with this hashing helper and the
    validation producer modules named by the caller.  This closes omissions
    from hand-maintained import lists while leaving each contributing file
    independently auditable.  Logical source content, rather than checkout
    newline policy, defines the identity.
    """
    root = Path(__file__).resolve().parents[1]
    qpsim_root = root / "qpsim"
    source_paths = {
        *qpsim_root.rglob("*.py"),
        *qpsim_root.rglob("*.yaml"),
        *qpsim_root.rglob("*.yml"),
        Path(__file__).resolve(),
        validation_module.resolve(),
        *(module.resolve() for module in extra_validation_modules),
    }
    sources: dict[str, Path] = {}
    for path in source_paths:
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Validation fingerprint source {path} is outside {root}."
            ) from exc
        if not path.is_file():
            raise ValueError(f"Validation fingerprint source {path} is not a file.")
        sources[relative] = path
    return {
        relative: source_sha256(sources[relative])
        for relative in sorted(sources)
    }
