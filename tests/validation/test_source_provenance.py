"""Tests for portable validation source fingerprints."""

from __future__ import annotations

import hashlib

from validation.source_provenance import (
    canonical_source_bytes,
    canonical_source_text,
    source_sha256,
)


def test_canonical_source_bytes_normalizes_text_newlines(tmp_path) -> None:
    path = tmp_path / "source.py"
    expected = b"value = 1\nother = 2\n"

    for contents in (
        expected,
        b"value = 1\r\nother = 2\r\n",
        b"value = 1\rother = 2\r",
    ):
        path.write_bytes(contents)
        assert canonical_source_bytes(path) == expected
        assert source_sha256(path) == hashlib.sha256(expected).hexdigest()


def test_source_sha256_changes_for_logical_source_edit(tmp_path) -> None:
    path = tmp_path / "source.py"
    path.write_bytes(b"value = 1\r\n")
    before = source_sha256(path)

    path.write_bytes(b"value = 2\n")

    assert source_sha256(path) != before


def test_canonical_source_text_normalizes_text_newlines() -> None:
    assert canonical_source_text("a\nb\n") == "a\nb\n"
    assert canonical_source_text("a\r\nb\r\n") == "a\nb\n"
    assert canonical_source_text("a\rb\r") == "a\nb\n"
