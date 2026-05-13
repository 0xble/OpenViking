# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Security regression tests for ovpack import target-policy enforcement."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.ovpack.format import manifest_content_sha256
from openviking.storage.ovpack.operations import import_ovpack
from openviking_cli.exceptions import InvalidArgumentError, NotFoundError
from openviking_cli.session.user_id import UserIdentifier


class FakeVikingFS:
    def __init__(self) -> None:
        self.written_files: list[str] = []
        self.created_dirs: list[str] = []

    async def stat(self, uri: str, ctx=None):
        return {"uri": uri, "isDir": True}

    async def mkdir(self, uri: str, exist_ok: bool = False, ctx=None):
        self.created_dirs.append(uri)

    async def ls(self, uri: str, ctx=None):
        raise NotFoundError(uri, "file")

    async def write_file_bytes(self, uri: str, data: bytes, ctx=None):
        self.written_files.append(uri)

    async def tree(self, uri: str, node_limit: int = 100000, level_limit: int = 1000, ctx=None):
        return []

    async def exists(self, uri: str, ctx=None):
        return False

    async def read_file(self, uri: str, ctx=None):
        raise FileNotFoundError(uri)


@pytest.fixture
def request_ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier("acct", "alice", "agent1"), role=Role.USER)


@pytest.fixture
def temp_ovpack_path() -> Path:
    fd, path = tempfile.mkstemp(suffix=".ovpack")
    os.close(fd)
    ovpack_path = Path(path)
    try:
        yield ovpack_path
    finally:
        ovpack_path.unlink(missing_ok=True)


def _write_ovpack(
    path: Path,
    entries: dict[str, str],
    *,
    root_name: str = "demo",
    root_uri: str = "viking://resources/demo",
) -> None:
    manifest_entries: list[dict[str, object]] = [{"path": "", "kind": "directory"}]
    directories = {
        parent for rel_path in entries for parent in ["/".join(rel_path.split("/")[:-1])] if parent
    }
    for directory in sorted(directories):
        manifest_entries.append({"path": directory, "kind": "directory"})
    for rel_path, content in sorted(entries.items()):
        data = content.encode("utf-8")
        manifest_entries.append(
            {
                "path": rel_path,
                "kind": "file",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest_files = {
        str(entry["path"]): entry for entry in manifest_entries if entry.get("kind") == "file"
    }
    index_records = b""
    manifest = {
        "kind": "openviking.ovpack",
        "format_version": 2,
        "root": {
            "name": root_name,
            "uri": root_uri,
            "scope": root_uri.split("://", 1)[1].split("/", 1)[0],
        },
        "entries": manifest_entries,
        "content_sha256": manifest_content_sha256(manifest_files),
        "index": {
            "records": {
                "path": "_ovpack/index_records.jsonl",
                "count": 0,
                "sha256": hashlib.sha256(index_records).hexdigest(),
            }
        },
    }
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{root_name}/", "")
        zf.writestr(f"{root_name}/files/", "")
        zf.writestr(f"{root_name}/_ovpack/", "")
        zf.writestr(f"{root_name}/_ovpack/index_records.jsonl", index_records)
        zf.writestr(f"{root_name}/_ovpack/manifest.json", json.dumps(manifest))
        for directory in sorted(directories):
            zf.writestr(f"{root_name}/files/{directory}/", "")
        for rel_path, content in entries.items():
            zf.writestr(f"{root_name}/files/{rel_path}", content)


@pytest.mark.asyncio
async def test_import_ovpack_allows_exported_derived_semantic_files(
    temp_ovpack_path: Path, request_ctx: RequestContext
):
    _write_ovpack(
        temp_ovpack_path,
        {
            ".overview.md": "ATTACKER_OVERVIEW",
            "notes.txt": "hello",
        },
    )
    fake_fs = FakeVikingFS()

    await import_ovpack(fake_fs, str(temp_ovpack_path), "viking://resources", request_ctx)

    assert fake_fs.written_files == [
        "viking://resources/demo/.overview.md",
        "viking://resources/demo/notes.txt",
    ]


@pytest.mark.asyncio
async def test_import_ovpack_rejects_watch_task_control_root(
    temp_ovpack_path: Path, request_ctx: RequestContext
):
    _write_ovpack(
        temp_ovpack_path,
        {
            ".watch_tasks.json": "{}",
        },
        root_name="resources",
        root_uri="viking://resources",
    )
    fake_fs = FakeVikingFS()

    with pytest.raises(
        InvalidArgumentError,
        match=r"cannot import watch task control file: viking://resources/\.watch_tasks\.json",
    ):
        await import_ovpack(fake_fs, str(temp_ovpack_path), "viking://", request_ctx)

    assert fake_fs.written_files == []


@pytest.mark.asyncio
async def test_import_ovpack_rejects_cross_scope_session_target(
    temp_ovpack_path: Path, request_ctx: RequestContext
):
    _write_ovpack(
        temp_ovpack_path,
        {
            ".meta.json": json.dumps({"session_id": "victim"}),
            "messages.jsonl": '{"id":"msg_attacker","role":"user","parts":[{"type":"text","text":"forged"}],"created_at":"2026-01-01T00:00:00Z"}\n',
        },
        root_name="default",
        root_uri="viking://session/default",
    )
    fake_fs = FakeVikingFS()

    with pytest.raises(
        InvalidArgumentError,
        match=r"ovpack source scope does not match target scope",
    ):
        await import_ovpack(fake_fs, str(temp_ovpack_path), "viking://resources", request_ctx)

    assert fake_fs.written_files == []
