"""Strict structural parser for one-page Matplotlib PDF artifacts."""

from __future__ import annotations

import re
import zlib
from collections.abc import Mapping
from pathlib import Path

_STARTXREF_RE = re.compile(
    rb"startxref[ \t\r\n]+(\d+)[ \t\r\n]+%%EOF[ \t\r\n]*\Z"
)
_REF_RE = re.compile(rb"(\d+)\s+(\d+)\s+R")
_PDF_WHITESPACE = frozenset(b"\x00\t\n\f\r ")
_PDF_DELIMITERS = frozenset(b"()<>[]{}/%")
_PURE_FILL_OPERATORS = frozenset({b"f", b"F", b"f*"})
_VISIBLE_MARKING_OPERATORS = frozenset(
    {
        b"S",
        b"s",
        b"B",
        b"B*",
        b"b",
        b"b*",
        b"Do",
        b"sh",
        b"BI",
        b"Tj",
        b"TJ",
        b"'",
        b'"',
    }
)


def _fail(path: Path, message: str) -> ValueError:
    return ValueError(f"M25 companion at {path} {message}.")


def _classic_xref(payload: bytes, *, path: Path) -> dict[int, tuple[int, int]]:
    match = _STARTXREF_RE.search(payload)
    if match is None:
        raise _fail(path, "has no terminal PDF xref/EOF record")
    xref_offset = int(match.group(1))
    if not payload[xref_offset:].startswith(b"xref"):
        raise _fail(path, "has an invalid PDF startxref offset")
    trailer_offset = payload.find(b"trailer", xref_offset + len(b"xref"))
    if trailer_offset < 0:
        raise _fail(path, "has no PDF trailer")
    lines = payload[xref_offset + len(b"xref"):trailer_offset].splitlines()
    entries: dict[int, tuple[int, int]] = {}
    cursor = 0
    while cursor < len(lines):
        line = lines[cursor].strip()
        cursor += 1
        if not line:
            continue
        header = line.split()
        if len(header) != 2 or not all(token.isdigit() for token in header):
            raise _fail(path, "has a malformed PDF xref subsection")
        first, count = (int(token) for token in header)
        if count <= 0 or cursor + count > len(lines):
            raise _fail(path, "has a truncated PDF xref subsection")
        for index in range(count):
            fields = lines[cursor + index].strip().split()
            if (
                len(fields) != 3
                or len(fields[0]) != 10
                or len(fields[1]) != 5
                or not fields[0].isdigit()
                or not fields[1].isdigit()
                or fields[2] not in {b"n", b"f"}
            ):
                raise _fail(path, "has a malformed PDF xref entry")
            if fields[2] == b"n":
                entries[first + index] = (int(fields[0]), int(fields[1]))
        cursor += count
    if not entries:
        raise _fail(path, "has no live PDF objects")
    return entries


def _object(
    payload: bytes,
    entries: Mapping[int, tuple[int, int]],
    object_number: int,
    generation: int,
    *,
    path: Path,
) -> bytes:
    entry = entries.get(object_number)
    if entry is None or entry[1] != generation:
        raise _fail(path, "references a missing PDF object")
    offset = entry[0]
    header = re.compile(
        rb"\A"
        + str(object_number).encode()
        + rb"\s+"
        + str(generation).encode()
        + rb"\s+obj\b"
    )
    match = header.match(payload[offset:])
    if match is None:
        raise _fail(path, "has a false PDF object offset")
    body_start = offset + match.end()
    later_offsets = [
        candidate_offset
        for candidate_offset, _candidate_generation in entries.values()
        if candidate_offset > offset
    ]
    object_limit = min(later_offsets, default=len(payload))
    body_end = payload.rfind(b"endobj", body_start, object_limit)
    if body_end < 0:
        raise _fail(path, "has an unterminated PDF object")
    return payload[body_start:body_end]


def _stream(
    payload: bytes,
    entries: Mapping[int, tuple[int, int]],
    object_number: int,
    generation: int,
    *,
    path: Path,
) -> bytes:
    body = _object(
        payload,
        entries,
        object_number,
        generation,
        path=path,
    )
    stream_marker = re.search(rb"\bstream\r?\n", body)
    if stream_marker is None:
        raise _fail(path, "has a page content object without a stream")
    dictionary = body[:stream_marker.start()]
    direct_length = re.search(rb"/Length\s+(\d+)(?!\s+\d+\s+R)", dictionary)
    indirect_length = re.search(rb"/Length\s+(\d+)\s+(\d+)\s+R", dictionary)
    if indirect_length is not None:
        length_object = _object(
            payload,
            entries,
            int(indirect_length.group(1)),
            int(indirect_length.group(2)),
            path=path,
        )
        length_match = re.fullmatch(rb"\s*(\d+)\s*", length_object)
        if length_match is None:
            raise _fail(path, "has an invalid PDF stream length object")
        length = int(length_match.group(1))
    elif direct_length is not None:
        length = int(direct_length.group(1))
    else:
        raise _fail(path, "has a page stream without a length")
    stream_start = stream_marker.end()
    stream_end = stream_start + length
    if length <= 0 or stream_end > len(body):
        raise _fail(path, "has an empty or truncated PDF page stream")
    encoded = body[stream_start:stream_end]
    if body[stream_end:].lstrip(b"\r\n \t")[:9] != b"endstream":
        raise _fail(path, "has a malformed PDF page stream")
    filters = re.findall(rb"/Filter\s*/([A-Za-z0-9]+)", dictionary)
    if filters == [b"FlateDecode"]:
        try:
            decoded = zlib.decompress(encoded)
        except zlib.error as exc:
            raise _fail(path, "has an invalid compressed PDF page stream") from exc
    elif not filters:
        decoded = encoded
    else:
        raise _fail(path, "uses an unsupported PDF page-stream filter")
    if not decoded.strip():
        raise _fail(path, "has an empty PDF page")
    return decoded


def _content_tokens(payload: bytes) -> list[bytes]:
    """Lex page operators while excluding strings, names, and comments."""
    tokens: list[bytes] = []
    cursor = 0
    size = len(payload)
    while cursor < size:
        byte = payload[cursor]
        if byte in _PDF_WHITESPACE:
            cursor += 1
            continue
        if byte == ord("%"):
            cursor += 1
            while cursor < size and payload[cursor] not in b"\r\n":
                cursor += 1
            continue
        if byte == ord("("):
            cursor += 1
            depth = 1
            while cursor < size and depth:
                current = payload[cursor]
                if current == ord("\\"):
                    cursor += 2
                    continue
                if current == ord("("):
                    depth += 1
                elif current == ord(")"):
                    depth -= 1
                cursor += 1
            continue
        if byte == ord("<") and (
            cursor + 1 >= size or payload[cursor + 1] != ord("<")
        ):
            cursor += 1
            while cursor < size and payload[cursor] != ord(">"):
                cursor += 1
            cursor += cursor < size
            continue
        if byte == ord("/"):
            start = cursor
            cursor += 1
            while (
                cursor < size
                and payload[cursor] not in _PDF_WHITESPACE
                and payload[cursor] not in _PDF_DELIMITERS
            ):
                cursor += 1
            tokens.append(payload[start:cursor])
            continue
        if byte in _PDF_DELIMITERS:
            if (
                byte in b"<>"
                and cursor + 1 < size
                and payload[cursor + 1] == byte
            ):
                tokens.append(payload[cursor:cursor + 2])
                cursor += 2
            else:
                tokens.append(payload[cursor:cursor + 1])
                cursor += 1
            continue
        start = cursor
        while (
            cursor < size
            and payload[cursor] not in _PDF_WHITESPACE
            and payload[cursor] not in _PDF_DELIMITERS
        ):
            cursor += 1
        tokens.append(payload[start:cursor])
    return tokens


def _has_visible_semantic_mark(decoded_streams: list[bytes]) -> bool:
    """Require a visible mark beyond Matplotlib's first canvas fill."""
    ignored_canvas_fill = False
    for stream in decoded_streams:
        for token in _content_tokens(stream):
            if token in _VISIBLE_MARKING_OPERATORS:
                return True
            if token in _PURE_FILL_OPERATORS:
                if ignored_canvas_fill:
                    return True
                ignored_canvas_fill = True
    return False


def validate_single_nonempty_matplotlib_pdf(
    payload: bytes,
    *,
    path: Path,
) -> None:
    """Require one complete, nonempty page in Matplotlib's classic format."""
    if re.match(rb"\A%PDF-\d\.\d(?:\r?\n)", payload) is None:
        raise _fail(path, "has no valid PDF header")
    entries = _classic_xref(payload, path=path)
    startxref = _STARTXREF_RE.search(payload)
    if startxref is None:
        raise _fail(path, "has no terminal PDF xref/EOF record")
    trailer_offset = payload.find(b"trailer", int(startxref.group(1)))
    trailer = payload[trailer_offset:payload.find(b"startxref", trailer_offset)]
    if re.search(rb"/Prev\s+\d+", trailer) is not None:
        raise _fail(path, "uses an incremental PDF update")
    root_match = re.search(rb"/Root\s+(\d+)\s+(\d+)\s+R", trailer)
    if root_match is None:
        raise _fail(path, "has no PDF catalog reference")
    catalog = _object(
        payload,
        entries,
        int(root_match.group(1)),
        int(root_match.group(2)),
        path=path,
    )
    if re.search(rb"/Type\s*/Catalog\b", catalog) is None:
        raise _fail(path, "has an invalid PDF catalog")
    pages_match = re.search(rb"/Pages\s+(\d+)\s+(\d+)\s+R", catalog)
    if pages_match is None:
        raise _fail(path, "has no PDF page tree")
    pages = _object(
        payload,
        entries,
        int(pages_match.group(1)),
        int(pages_match.group(2)),
        path=path,
    )
    count_match = re.search(rb"/Count\s+(\d+)", pages)
    kids_match = re.search(rb"/Kids\s*\[(.*?)\]", pages, flags=re.DOTALL)
    kids = _REF_RE.findall(kids_match.group(1)) if kids_match is not None else []
    if count_match is None or int(count_match.group(1)) != 1 or len(kids) != 1:
        raise _fail(path, "must contain exactly one PDF page")
    page = _object(
        payload,
        entries,
        int(kids[0][0]),
        int(kids[0][1]),
        path=path,
    )
    if re.search(rb"/Type\s*/Page(?!s)\b", page) is None:
        raise _fail(path, "page tree does not reference a PDF page")
    contents_array = re.search(rb"/Contents\s*\[(.*?)\]", page, flags=re.DOTALL)
    if contents_array is not None:
        content_refs = _REF_RE.findall(contents_array.group(1))
    else:
        contents = re.search(rb"/Contents\s+(\d+)\s+(\d+)\s+R", page)
        content_refs = (
            [(contents.group(1), contents.group(2))]
            if contents is not None
            else []
        )
    if not content_refs:
        raise _fail(path, "has a PDF page without content")
    decoded_streams = [
        _stream(
            payload,
            entries,
            int(number),
            int(generation),
            path=path,
        )
        for number, generation in content_refs
    ]
    if not _has_visible_semantic_mark(decoded_streams):
        raise _fail(path, "has no visible PDF page marks beyond its canvas")
