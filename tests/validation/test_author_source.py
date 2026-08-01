from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest
from validation.author_source import (
    AuthorSourceError,
    AuthorSourceUnavailableError,
    authenticate_author_source,
    load_author_source_manifest,
    read_authenticated_member,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    REPOSITORY_ROOT
    / "validation"
    / "paper_data"
    / "fischer_2023"
    / "fig6"
    / "author-source.json"
)


def _synthetic_source(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    archive = tmp_path / "author.zip"
    members = {
        "root/run.py": b"print('author')\n",
        "root/output.png": b"\x89PNG\r\n\x1a\nsynthetic",
    }
    with zipfile.ZipFile(archive, "w") as handle:
        for name, content in members.items():
            handle.writestr(name, content)
    payload: dict[str, object] = {
        "acquisition": {
            "acquired_by": "test",
            "acquisition_date": "2026-07-29",
            "method": "email_attachment",
            "provenance_note": "synthetic positive-path fixture",
            "repository_email_metadata": "not applicable",
        },
        "archive": {
            "filename": archive.name,
            "locator_environment_variable": "TEST_AUTHOR_ARCHIVE",
            "member_root": "root/",
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "size_bytes": archive.stat().st_size,
        },
        "classification": {
            "evidence_class": "author_supplied_reproduction_source",
            "publication_source_identity": "not_established",
            "redistribution_status": "permission_not_established",
            "scope": "synthetic test only",
        },
        "execution": {
            "default_newton_steps": 1,
            "default_sweep_points": 1,
            "dependencies": ["Python"],
            "entry_point": "root/run.py",
            "original_runtime_evidence": "synthetic",
            "output_member": "root/output.png",
            "replay_status": "synthetic",
        },
        "members": [
            {
                "path": path,
                "role": role,
                "sha256": hashlib.sha256(members[path]).hexdigest(),
                "size_bytes": len(members[path]),
            }
            for path, role in (
                ("root/run.py", "entry_point"),
                ("root/output.png", "bundled_output_oracle"),
            )
        ],
        "paper": {
            "arxiv_id": "test",
            "citation": "synthetic",
            "figure": "test",
            "version": "test",
        },
        "schema": "qpsim.author-source.v1",
        "source_id": "synthetic-author-source",
    }
    manifest = tmp_path / "author-source.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest, archive, payload


def test_fig6_author_source_manifest_is_narrow_and_honest() -> None:
    manifest = load_author_source_manifest(MANIFEST)
    assert manifest["classification"] == {
        "evidence_class": "author_supplied_reproduction_source",
        "publication_source_identity": "not_established",
        "redistribution_status": "permission_not_established",
        "scope": "Fischer-Catelani 2023 Figure 6 only",
    }
    assert manifest["execution"]["default_sweep_points"] == 300
    assert manifest["execution"]["default_newton_steps"] == 10
    assert manifest["archive"]["sha256"] == (
        "31d76c92ef12c8056b583ef0d770f9f5c1ef48c561db971cc0dab1b733481bc1"
    )


def test_missing_external_author_source_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE", raising=False)
    with pytest.raises(AuthorSourceUnavailableError, match="Set QPSIM_FISCHER2023"):
        authenticate_author_source(MANIFEST)


def test_tampered_author_archive_is_rejected(tmp_path: Path) -> None:
    damaged = tmp_path / "PhysApplPaper_Figure_6.zip"
    damaged.write_bytes(b"not the authenticated attachment")
    with pytest.raises(AuthorSourceError, match="byte size"):
        authenticate_author_source(MANIFEST, damaged)


def test_synthetic_archive_exercises_positive_authentication_path(
    tmp_path: Path,
) -> None:
    manifest, archive, _payload = _synthetic_source(tmp_path)
    expected_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    authenticated = authenticate_author_source(manifest, archive)
    assert authenticated.source_id == "synthetic-author-source"
    assert authenticated.manifest_sha256 == expected_manifest_sha
    assert {member.role for member in authenticated.members} == {
        "entry_point",
        "bundled_output_oracle",
    }
    assert read_authenticated_member(
        manifest,
        role="entry_point",
        archive_path=archive,
    ) == b"print('author')\n"


def test_authenticated_members_retain_the_verified_byte_snapshot(tmp_path: Path) -> None:
    manifest, archive, _payload = _synthetic_source(tmp_path)
    authenticated = authenticate_author_source(manifest, archive)
    entry = next(member for member in authenticated.members if member.role == "entry_point")
    archive.write_bytes(b"replaced after authentication")
    assert entry.content == b"print('author')\n"


def test_authenticated_source_retains_the_validated_manifest_snapshot_identity(
    tmp_path: Path,
) -> None:
    manifest, archive, payload = _synthetic_source(tmp_path)
    authenticated = authenticate_author_source(manifest, archive)
    authenticated_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()

    payload["source_id"] = "replacement-after-authentication"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert authenticated.source_id == "synthetic-author-source"
    assert authenticated.manifest_sha256 == authenticated_sha
    assert authenticated.manifest_sha256 != hashlib.sha256(manifest.read_bytes()).hexdigest()


def test_authentication_parses_and_hashes_one_manifest_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, archive, _payload = _synthetic_source(tmp_path)
    original_read_bytes = Path.read_bytes
    reads = 0

    def guarded_read_bytes(path: Path) -> bytes:
        nonlocal reads
        if path == manifest:
            reads += 1
            if reads > 1:
                raise AssertionError("manifest path was re-read")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    authenticated = authenticate_author_source(manifest, archive)
    assert authenticated.source_id == "synthetic-author-source"
    assert reads == 1


def test_manifest_rejects_duplicate_or_misassigned_roles(tmp_path: Path) -> None:
    manifest, _archive, payload = _synthetic_source(tmp_path)
    members = payload["members"]
    assert isinstance(members, list)
    duplicate = json.loads(json.dumps(payload))
    duplicate["members"][1]["role"] = "entry_point"
    manifest.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(AuthorSourceError, match="duplicates the role"):
        load_author_source_manifest(manifest)

    misassigned = json.loads(json.dumps(payload))
    misassigned["execution"]["entry_point"] = "root/output.png"
    manifest.write_text(json.dumps(misassigned), encoding="utf-8")
    with pytest.raises(AuthorSourceError, match="entry_point role"):
        load_author_source_manifest(manifest)


@pytest.mark.parametrize(
    ("member_root", "member_path"),
    [
        ("root", "rooted/run.py"),
        ("root/", "rooted/run.py"),
        ("root/./", "root/run.py"),
        ("root\\", "root/run.py"),
    ],
)
def test_member_root_containment_uses_literal_posix_components(
    tmp_path: Path,
    member_root: str,
    member_path: str,
) -> None:
    manifest, _archive, payload = _synthetic_source(tmp_path)
    archive_record = payload["archive"]
    members = payload["members"]
    assert isinstance(archive_record, dict)
    assert isinstance(members, list)
    archive_record["member_root"] = member_root
    members[0]["path"] = member_path
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuthorSourceError, match=r"member_root|outside"):
        load_author_source_manifest(manifest)


def test_duplicate_zip_member_names_are_rejected(tmp_path: Path) -> None:
    manifest, archive, payload = _synthetic_source(tmp_path)
    with (
        pytest.warns(UserWarning, match="Duplicate name"),
        zipfile.ZipFile(archive, "a") as handle,
    ):
        handle.writestr("root/run.py", b"print('second')\n")
    archive_record = payload["archive"]
    assert isinstance(archive_record, dict)
    archive_record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_record["size_bytes"] = archive.stat().st_size
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AuthorSourceError, match="duplicate member names"):
        authenticate_author_source(manifest, archive)


def test_zip_members_must_share_the_exact_declared_root(tmp_path: Path) -> None:
    manifest, archive, payload = _synthetic_source(tmp_path)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("rooted/undeclared.py", b"outside the exact root\n")
    archive_record = payload["archive"]
    assert isinstance(archive_record, dict)
    archive_record["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive_record["size_bytes"] = archive.stat().st_size
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuthorSourceError, match="outside the exact"):
        authenticate_author_source(manifest, archive)


def test_source_roles_may_not_collide_at_import_basename(tmp_path: Path) -> None:
    manifest, _archive, payload = _synthetic_source(tmp_path)
    members = payload["members"]
    assert isinstance(members, list)
    members.append(
        {
            "path": "root/nested/run.py",
            "role": "helper_source",
            "sha256": "0" * 64,
            "size_bytes": 1,
        }
    )
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AuthorSourceError, match="source basename"):
        load_author_source_manifest(manifest)


def test_exact_external_author_source_authenticates_when_supplied() -> None:
    configured = os.environ.get("QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE")
    if not configured:
        pytest.skip("Set QPSIM_FISCHER2023_FIG6_AUTHOR_ARCHIVE for exact-source replay.")
    authenticated = authenticate_author_source(MANIFEST, Path(configured))
    assert authenticated.source_id == "fischer-catelani-2023-fig6-author-email-attachment-v1"
    assert len(authenticated.members) == 6
    assert {member.role for member in authenticated.members} >= {
        "entry_point",
        "coupled_solver",
        "bundled_output_oracle",
    }
    png = read_authenticated_member(
        MANIFEST,
        role="bundled_output_oracle",
        archive_path=Path(configured),
    )
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
