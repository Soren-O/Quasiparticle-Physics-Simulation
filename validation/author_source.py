"""Authenticate externally held, author-supplied validation source archives.

Author attachments are not silently vendored into qpsim.  A checked manifest
binds the exact archive and the scientifically relevant members, while the
archive itself is supplied explicitly at replay time.  This keeps provenance
claims auditable without assuming redistribution permission.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

AUTHOR_SOURCE_SCHEMA = "qpsim.author-source.v1"
_NON_SOURCE_ROLES = {"bundled_output_oracle"}


class AuthorSourceError(ValueError):
    """The author-source manifest or supplied archive is invalid."""


class AuthorSourceUnavailableError(AuthorSourceError):
    """The manifest is valid, but its external archive was not supplied."""


@dataclass(frozen=True)
class AuthenticatedMember:
    """One exact member authenticated inside an author-source archive."""

    path: str
    role: str
    size_bytes: int
    sha256: str
    content: bytes = field(repr=False)


@dataclass(frozen=True)
class AuthenticatedAuthorSource:
    """Result of authenticating an external author-source archive."""

    source_id: str
    archive_path: Path
    archive_sha256: str
    manifest_sha256: str
    members: tuple[AuthenticatedMember, ...]


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuthorSourceError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise AuthorSourceError(f"Non-finite JSON number {value!r} is forbidden.")


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AuthorSourceError(f"{label} must be a JSON object.")
    return value


def _exact_keys(mapping: dict[str, Any], expected: set[str], label: str) -> None:
    if set(mapping) != expected:
        raise AuthorSourceError(
            f"{label} fields are invalid: expected {sorted(expected)!r}, "
            f"got {sorted(mapping)!r}."
        )


def _nonempty_string(mapping: dict[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise AuthorSourceError(f"{label}.{key} must be a nonempty string.")
    return value


def _sha256(mapping: dict[str, Any], key: str, label: str) -> str:
    value = _nonempty_string(mapping, key, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise AuthorSourceError(f"{label}.{key} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(mapping: dict[str, Any], key: str, label: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AuthorSourceError(f"{label}.{key} must be a positive integer.")
    return value


def _safe_posix_parts(
    raw: str,
    label: str,
    *,
    allow_trailing_slash: bool,
) -> tuple[str, ...]:
    """Return canonical relative POSIX path components.

    ``str.startswith`` is not a containment check: ``rooted/file`` starts
    with ``root``.  ZIP member names are POSIX paths regardless of the host,
    so validate their literal components before comparing a member to its
    declared root.
    """

    if "\x00" in raw or "\\" in raw:
        raise AuthorSourceError(f"{label} must be a safe relative POSIX path.")
    if raw.endswith("/"):
        if not allow_trailing_slash:
            raise AuthorSourceError(f"{label} must name a file, not a directory.")
        raw = raw[:-1]
    literal_parts = raw.split("/")
    pure = PurePosixPath(raw)
    if (
        not raw
        or pure.is_absolute()
        or not literal_parts
        or any(part in {"", ".", ".."} for part in literal_parts)
        or pure.parts != tuple(literal_parts)
    ):
        raise AuthorSourceError(f"{label} must be a safe relative POSIX path.")
    return tuple(literal_parts)


def _load_author_source_manifest_snapshot(
    path: Path,
) -> tuple[dict[str, Any], str]:
    """Read, hash, parse, and validate one immutable manifest snapshot."""

    try:
        manifest_bytes = path.read_bytes()
        manifest_text = manifest_bytes.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuthorSourceError(f"Cannot read author-source manifest {path}: {exc}") from exc

    try:
        payload = json.loads(
            manifest_text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise AuthorSourceError(f"Cannot read author-source manifest {path}: {exc}") from exc
    manifest = _mapping(payload, "author-source manifest")
    _exact_keys(
        manifest,
        {
            "acquisition",
            "archive",
            "classification",
            "execution",
            "members",
            "paper",
            "schema",
            "source_id",
        },
        "author-source manifest",
    )
    if manifest.get("schema") != AUTHOR_SOURCE_SCHEMA:
        raise AuthorSourceError("The author-source manifest schema is stale.")
    _nonempty_string(manifest, "source_id", "author-source manifest")

    classification = _mapping(manifest["classification"], "classification")
    _exact_keys(
        classification,
        {
            "evidence_class",
            "publication_source_identity",
            "redistribution_status",
            "scope",
        },
        "classification",
    )
    if classification.get("evidence_class") != "author_supplied_reproduction_source":
        raise AuthorSourceError("Unsupported author-source evidence class.")
    if classification.get("publication_source_identity") != "not_established":
        raise AuthorSourceError(
            "This schema deliberately distinguishes author-supplied reproduction "
            "source from a proven historical publication checkout."
        )
    if classification.get("redistribution_status") != "permission_not_established":
        raise AuthorSourceError("Unexpected redistribution status.")
    _nonempty_string(classification, "scope", "classification")

    paper = _mapping(manifest["paper"], "paper")
    _exact_keys(paper, {"arxiv_id", "citation", "figure", "version"}, "paper")
    for key in paper:
        _nonempty_string(paper, key, "paper")

    acquisition = _mapping(manifest["acquisition"], "acquisition")
    _exact_keys(
        acquisition,
        {
            "acquired_by",
            "acquisition_date",
            "method",
            "provenance_note",
            "repository_email_metadata",
        },
        "acquisition",
    )
    if acquisition.get("method") != "email_attachment":
        raise AuthorSourceError("Author source must identify its acquisition method.")
    for key in acquisition:
        _nonempty_string(acquisition, key, "acquisition")

    archive = _mapping(manifest["archive"], "archive")
    _exact_keys(
        archive,
        {
            "filename",
            "locator_environment_variable",
            "member_root",
            "sha256",
            "size_bytes",
        },
        "archive",
    )
    _nonempty_string(archive, "filename", "archive")
    _nonempty_string(archive, "locator_environment_variable", "archive")
    member_root = _nonempty_string(archive, "member_root", "archive")
    member_root_parts = _safe_posix_parts(
        member_root,
        "archive.member_root",
        allow_trailing_slash=True,
    )
    _sha256(archive, "sha256", "archive")
    _positive_int(archive, "size_bytes", "archive")

    execution = _mapping(manifest["execution"], "execution")
    _exact_keys(
        execution,
        {
            "default_newton_steps",
            "default_sweep_points",
            "dependencies",
            "entry_point",
            "original_runtime_evidence",
            "output_member",
            "replay_status",
        },
        "execution",
    )
    _positive_int(execution, "default_newton_steps", "execution")
    _positive_int(execution, "default_sweep_points", "execution")
    _nonempty_string(execution, "entry_point", "execution")
    _nonempty_string(execution, "original_runtime_evidence", "execution")
    _nonempty_string(execution, "output_member", "execution")
    _nonempty_string(execution, "replay_status", "execution")
    dependencies = execution.get("dependencies")
    if (
        not isinstance(dependencies, list)
        or not dependencies
        or any(not isinstance(item, str) or not item for item in dependencies)
    ):
        raise AuthorSourceError("execution.dependencies must be nonempty strings.")

    members = manifest.get("members")
    if not isinstance(members, list) or not members:
        raise AuthorSourceError("members must be a nonempty list.")
    seen_paths: set[str] = set()
    role_paths: dict[str, str] = {}
    source_basenames: dict[str, str] = {}
    for index, raw_member in enumerate(members):
        label = f"members[{index}]"
        member = _mapping(raw_member, label)
        _exact_keys(member, {"path", "role", "sha256", "size_bytes"}, label)
        member_path = _nonempty_string(member, "path", label)
        member_parts = _safe_posix_parts(
            member_path,
            f"{label}.path",
            allow_trailing_slash=False,
        )
        if (
            member_path in seen_paths
            or len(member_parts) <= len(member_root_parts)
            or member_parts[: len(member_root_parts)] != member_root_parts
        ):
            raise AuthorSourceError(f"{label}.path is duplicate, unsafe, or outside member_root.")
        seen_paths.add(member_path)
        role = _nonempty_string(member, "role", label)
        if role in role_paths:
            raise AuthorSourceError(
                f"{label}.role duplicates the role already assigned to "
                f"{role_paths[role]!r}."
            )
        role_paths[role] = member_path
        if role not in _NON_SOURCE_ROLES and member_path.endswith(".py"):
            basename = member_parts[-1]
            if basename in source_basenames:
                raise AuthorSourceError(
                    f"{label}.path collides at source basename {basename!r} "
                    f"with {source_basenames[basename]!r}."
                )
            source_basenames[basename] = member_path
        _sha256(member, "sha256", label)
        _positive_int(member, "size_bytes", label)
    if "entry_point" not in role_paths or "bundled_output_oracle" not in role_paths:
        raise AuthorSourceError("members must bind the entry point and bundled output oracle.")
    if execution["entry_point"] not in seen_paths or execution["output_member"] not in seen_paths:
        raise AuthorSourceError("execution paths must name authenticated members.")
    if role_paths["entry_point"] != execution["entry_point"]:
        raise AuthorSourceError("execution.entry_point must carry the entry_point role.")
    if role_paths["bundled_output_oracle"] != execution["output_member"]:
        raise AuthorSourceError(
            "execution.output_member must carry the bundled_output_oracle role."
        )
    return manifest, hashlib.sha256(manifest_bytes).hexdigest()


def load_author_source_manifest(path: Path) -> dict[str, Any]:
    """Load and strictly validate one checked author-source manifest."""

    manifest, _manifest_sha256 = _load_author_source_manifest_snapshot(path)
    return manifest


def authenticate_author_source(
    manifest_path: Path,
    archive_path: Path | None = None,
) -> AuthenticatedAuthorSource:
    """Authenticate the exact external archive and every declared member.

    If ``archive_path`` is omitted, the manifest's locator environment
    variable is consulted.  No fallback search is performed: an audit must
    never authenticate whichever similarly named archive happens to be found.
    """

    manifest, manifest_sha256 = _load_author_source_manifest_snapshot(manifest_path)
    archive = _mapping(manifest["archive"], "archive")
    if archive_path is None:
        locator = _nonempty_string(archive, "locator_environment_variable", "archive")
        configured = os.environ.get(locator)
        if not configured:
            raise AuthorSourceUnavailableError(
                f"Set {locator} to the exact {archive['filename']} attachment "
                "before replaying author source."
            )
        archive_path = Path(configured)
    resolved = archive_path.expanduser().resolve()
    if not resolved.is_file():
        raise AuthorSourceUnavailableError(
            f"Configured author-source archive is missing: {resolved}"
        )
    try:
        archive_bytes = resolved.read_bytes()
    except OSError as exc:
        raise AuthorSourceUnavailableError(
            f"Cannot read configured author-source archive {resolved}: {exc}"
        ) from exc
    if len(archive_bytes) != archive["size_bytes"]:
        raise AuthorSourceError("Author-source archive byte size does not match its manifest.")
    actual_archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_archive_sha != archive["sha256"]:
        raise AuthorSourceError("Author-source archive SHA-256 does not match its manifest.")

    declared_members = {
        member["path"]: member for member in manifest["members"] if isinstance(member, dict)
    }
    member_root_parts = _safe_posix_parts(
        archive["member_root"],
        "archive.member_root",
        allow_trailing_slash=True,
    )
    authenticated: list[AuthenticatedMember] = []
    try:
        # Authenticate and retain one immutable in-memory snapshot. Reopening
        # the path after hashing would permit a concurrent replacement to
        # supply different bytes under the authenticated metadata.
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as source_zip:
            names = [info.filename for info in source_zip.infolist()]
            if len(names) != len(set(names)):
                raise AuthorSourceError("Author-source ZIP contains duplicate member names.")
            for index, info in enumerate(source_zip.infolist()):
                member_parts = _safe_posix_parts(
                    info.filename,
                    f"ZIP member[{index}]",
                    allow_trailing_slash=info.is_dir(),
                )
                if (
                    len(member_parts) < len(member_root_parts)
                    or member_parts[: len(member_root_parts)] != member_root_parts
                ):
                    raise AuthorSourceError(
                        f"ZIP member {info.filename!r} is outside the exact "
                        "archive.member_root component path."
                    )
            available = set(names)
            for member_path, declaration in declared_members.items():
                if member_path not in available:
                    raise AuthorSourceError(
                        f"Author-source ZIP is missing declared member {member_path!r}."
                    )
                member_bytes = source_zip.read(member_path)
                if len(member_bytes) != declaration["size_bytes"]:
                    raise AuthorSourceError(
                        f"Author-source member {member_path!r} has the wrong byte size."
                    )
                member_sha = hashlib.sha256(member_bytes).hexdigest()
                if member_sha != declaration["sha256"]:
                    raise AuthorSourceError(
                        f"Author-source member {member_path!r} failed SHA-256 authentication."
                    )
                authenticated.append(
                    AuthenticatedMember(
                        path=member_path,
                        role=declaration["role"],
                        size_bytes=len(member_bytes),
                        sha256=member_sha,
                        content=member_bytes,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise AuthorSourceError(f"Author-source archive is not a valid ZIP: {exc}") from exc

    return AuthenticatedAuthorSource(
        source_id=manifest["source_id"],
        archive_path=resolved,
        archive_sha256=actual_archive_sha,
        manifest_sha256=manifest_sha256,
        members=tuple(authenticated),
    )


def read_authenticated_member(
    manifest_path: Path,
    *,
    role: str,
    archive_path: Path | None = None,
) -> bytes:
    """Return one uniquely role-identified member after full authentication."""

    if not role:
        raise AuthorSourceError("role must be a nonempty string.")
    authenticated = authenticate_author_source(manifest_path, archive_path)
    matches = [member for member in authenticated.members if member.role == role]
    if len(matches) != 1:
        raise AuthorSourceError(
            f"Expected exactly one authenticated member with role {role!r}; "
            f"found {len(matches)}."
        )
    return matches[0].content
