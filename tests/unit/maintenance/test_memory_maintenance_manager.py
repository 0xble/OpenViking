# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for dirty-scope memory maintenance state."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.maintenance.memory_maintenance_manager import (
    MEMORY_MAINTENANCE_STORAGE_URI,
    MemoryMaintenanceManager,
    dirty_scopes_from_memory_diff,
    extract_changed_memory_uris,
    memory_scope_for_uri,
)


def _diff():
    return {
        "archive_uri": "viking://session/s/history/archive_001",
        "operations": {
            "adds": [
                {"uri": "viking://user/u/memories/preferences/editor.md"},
                {"uri": "viking://agent/a/memories/patterns/review.md"},
            ],
            "updates": [
                {"uri": "viking://user/u/memories/preferences/editor.md"},
            ],
            "deletes": [
                {"uri": "viking://user/u/memories/profile.md"},
                {"uri": "viking://user/u/memories/preferences/_archive/old.md"},
            ],
        },
    }


def _ctx():
    ctx = MagicMock()
    ctx.account_id = "acc"
    ctx.user.user_id = "user"
    ctx.user.agent_id = "agent"
    return ctx


class _MemoryFS:
    def __init__(self):
        self.files = {}

    async def read_file(self, uri, **_kwargs):
        from openviking_cli.exceptions import NotFoundError

        if uri not in self.files:
            raise NotFoundError(uri)
        return self.files[uri]

    async def write_file(self, uri, content, **_kwargs):
        self.files[uri] = content

    async def exists(self, uri, **_kwargs):
        return uri in self.files

    async def rm(self, uri, **_kwargs):
        self.files.pop(uri, None)

    async def mv(self, src, dst, **_kwargs):
        self.files[dst] = self.files.pop(src)


def test_memory_scope_for_uri_groups_category_scopes():
    assert (
        memory_scope_for_uri("viking://user/u/memories/preferences/editor.md")
        == "viking://user/u/memories/preferences/"
    )
    assert (
        memory_scope_for_uri("viking://user/u/memories/profile.md")
        == "viking://user/u/memories/"
    )
    assert memory_scope_for_uri("viking://resources/x.md") == ""


def test_extract_changed_memory_uris_skips_archived_and_non_memory():
    assert extract_changed_memory_uris(_diff()) == [
        "viking://agent/a/memories/patterns/review.md",
        "viking://user/u/memories/preferences/editor.md",
        "viking://user/u/memories/profile.md",
    ]


def test_dirty_scopes_from_memory_diff_groups_uris():
    grouped = dirty_scopes_from_memory_diff(_diff())

    assert grouped == {
        "viking://agent/a/memories/patterns/": [
            "viking://agent/a/memories/patterns/review.md"
        ],
        "viking://user/u/memories/preferences/": [
            "viking://user/u/memories/preferences/editor.md"
        ],
        "viking://user/u/memories/": ["viking://user/u/memories/profile.md"],
    }


@pytest.mark.asyncio
async def test_manager_records_and_persists_dirty_scopes():
    fs = _MemoryFS()
    manager = MemoryMaintenanceManager(viking_fs=fs)

    changed = await manager.record_memory_diff(_diff(), _ctx())

    assert len(changed) == 3
    assert MEMORY_MAINTENANCE_STORAGE_URI in fs.files

    reloaded = MemoryMaintenanceManager(viking_fs=fs)
    scopes = await reloaded.list_scopes(active_only=True, account_id="acc", user_id="user")

    assert {s.scope_uri for s in scopes} == {
        "viking://agent/a/memories/patterns/",
        "viking://user/u/memories/preferences/",
        "viking://user/u/memories/",
    }
    assert all(s.dirty_count > 0 for s in scopes)


@pytest.mark.asyncio
async def test_manager_dirty_count_matches_retained_uris():
    fs = _MemoryFS()
    manager = MemoryMaintenanceManager(viking_fs=fs)
    memory_diff = {
        "operations": {
            "adds": [
                {"uri": f"viking://user/u/memories/preferences/mem_{i}.md"}
                for i in range(205)
            ],
        },
    }

    changed = await manager.record_memory_diff(memory_diff, _ctx())

    assert len(changed) == 1
    assert changed[0].dirty_count == 200
    assert len(changed[0].dirty_uris) == 200


@pytest.mark.asyncio
async def test_mark_run_complete_clears_dirty_state_only_for_apply():
    fs = _MemoryFS()
    manager = MemoryMaintenanceManager(viking_fs=fs)
    await manager.record_memory_diff(_diff(), _ctx())

    scope = "viking://user/u/memories/preferences/"
    dry = await manager.mark_run_complete(scope, audit_uri="viking://audit", dry_run=True)
    assert dry is not None
    assert dry.dirty_count == 1
    assert dry.is_active is True

    applied = await manager.mark_run_complete(scope, audit_uri="viking://audit", dry_run=False)
    assert applied is not None
    assert applied.dirty_count == 0
    assert applied.dirty_uris == []
    assert applied.is_active is False


@pytest.mark.asyncio
async def test_manager_uses_non_atomic_write_when_backend_lacks_mv():
    class WriteOnlyFS:
        def __init__(self):
            self.read_file = AsyncMock(side_effect=Exception("missing"))
            self.write_file = AsyncMock()

    fs = WriteOnlyFS()
    manager = MemoryMaintenanceManager(viking_fs=fs)
    await manager.record_memory_diff(_diff(), _ctx())

    fs.write_file.assert_awaited()


@pytest.mark.asyncio
async def test_manager_serializes_concurrent_saves():
    class SlowWriteFS(_MemoryFS):
        def __init__(self):
            super().__init__()
            self.active_writes = 0
            self.max_active_writes = 0

        async def write_file(self, uri, content, **kwargs):
            self.active_writes += 1
            self.max_active_writes = max(self.max_active_writes, self.active_writes)
            await asyncio.sleep(0.01)
            await super().write_file(uri, content, **kwargs)
            self.active_writes -= 1

    fs = SlowWriteFS()
    manager = MemoryMaintenanceManager(viking_fs=fs)
    await manager.initialize()

    await asyncio.gather(
        manager.mark_run_failed("viking://user/u/memories/preferences/", "one"),
        manager.mark_run_failed("viking://agent/a/memories/patterns/", "two"),
    )

    assert fs.max_active_writes == 1


@pytest.mark.asyncio
async def test_manager_restores_backup_when_final_promote_fails():
    class FailingPromoteFS(_MemoryFS):
        async def mv(self, src, dst, **kwargs):
            if (
                src == MemoryMaintenanceManager.STORAGE_TMP_URI
                and dst == MemoryMaintenanceManager.STORAGE_URI
            ):
                raise RuntimeError("promote failed")
            await super().mv(src, dst, **kwargs)

    fs = FailingPromoteFS()
    fs.files[MemoryMaintenanceManager.STORAGE_URI] = '{"scopes": []}'
    manager = MemoryMaintenanceManager(viking_fs=fs)

    await manager.record_memory_diff(_diff(), _ctx())

    assert fs.files[MemoryMaintenanceManager.STORAGE_URI] == '{"scopes": []}'
    assert await manager.list_scopes(active_only=False) == []
