# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.viking_fs import VikingFS
from openviking_cli.session.user_id import UserIdentifier


class _FakeAGFS:
    def __init__(self, content: bytes = b"payload") -> None:
        self.content = content
        self.writes = []
        self.removals = []

    def read(self, path: str) -> bytes:
        return self.content

    def write(self, path: str, data: bytes) -> str:
        self.writes.append((path, data))
        return path

    def rm(self, path: str) -> None:
        self.removals.append(path)


@pytest.mark.asyncio
async def test_move_file_updates_vector_store(monkeypatch):
    agfs = _FakeAGFS()
    viking_fs = VikingFS(agfs=agfs)
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    sync_calls = []

    monkeypatch.setattr(viking_fs, "_ensure_access", lambda uri, ctx: None)
    monkeypatch.setattr(
        viking_fs,
        "_uri_to_path",
        lambda uri, ctx=None: f"/virtual/{uri.removeprefix('viking://')}",
    )

    async def fake_ensure_parent_dirs(path: str) -> None:
        return None

    async def fake_update_vector_store_uris(uris, old_uri, new_uri, ctx=None) -> None:
        sync_calls.append((uris, old_uri, new_uri, ctx))

    monkeypatch.setattr(viking_fs, "_ensure_parent_dirs", fake_ensure_parent_dirs)
    monkeypatch.setattr(viking_fs, "_update_vector_store_uris", fake_update_vector_store_uris)

    await viking_fs.move_file(
        "viking://resources/source.txt",
        "viking://resources/dest.txt",
        ctx=ctx,
    )

    assert agfs.writes == [("/virtual/resources/dest.txt", b"payload")]
    assert agfs.removals == ["/virtual/resources/source.txt"]
    assert sync_calls == [
        (
            ["viking://resources/source.txt"],
            "viking://resources/source.txt",
            "viking://resources/dest.txt",
            ctx,
        )
    ]
