"""Portable hashing helpers for validation source provenance."""

from __future__ import annotations

import hashlib
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
