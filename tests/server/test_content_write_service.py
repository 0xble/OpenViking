# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Service-level tests for direct content write coordination."""

import pytest

import openviking.storage.content_write as content_write_module
from openviking.server.identity import RequestContext, Role
from openviking.session.memory.utils.content import deserialize_full, serialize_with_metadata
from openviking.storage.content_write import ContentWriteCoordinator
from openviking_cli.exceptions import (
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
)
from openviking_cli.session.user_id import UserIdentifier


@pytest.mark.asyncio
async def test_write_updates_memory_file_without_refresh(service):
    ctx = RequestContext(user=service.user, role=Role.USER)
    memory_uri = f"viking://user/{ctx.user.user_space_name()}/memories/preferences/theme.md"

    await service.viking_fs.write_file(memory_uri, "Original preference", ctx=ctx)

    result = await service.fs.write(
        memory_uri,
        content="Updated preference",
        ctx=ctx,
        mode="replace",
        wait=True,
        timeout=0,
    )

    assert result["context_type"] == "memory"
    assert result["content_updated"] is True
    assert result["semantic_status"] == "not_refreshed"
    assert result["vector_status"] == "not_refreshed"
    assert result["queue_status"] is None
    assert await service.viking_fs.read_file(memory_uri, ctx=ctx) == "Updated preference"


@pytest.mark.asyncio
async def test_write_denies_foreign_user_memory_space(service):
    owner_ctx = RequestContext(user=service.user, role=Role.USER)
    memory_uri = (
        f"viking://user/{owner_ctx.user.user_space_name()}/memories/preferences/private-note.md"
    )
    await service.viking_fs.write_file(memory_uri, "Owner note", ctx=owner_ctx)

    foreign_ctx = RequestContext(
        user=UserIdentifier(owner_ctx.account_id, "other_user", owner_ctx.user.agent_id),
        role=Role.USER,
    )

    with pytest.raises(NotFoundError):
        await service.fs.write(
            memory_uri,
            content="Intruder update",
            ctx=foreign_ctx,
        )


@pytest.mark.asyncio
async def test_memory_replace_preserves_metadata(service):
    ctx = RequestContext(user=service.user, role=Role.USER)
    memory_uri = f"viking://user/{ctx.user.user_space_name()}/memories/preferences/theme.md"
    metadata = {
        "tags": ["ui", "preference"],
        "created_at": "2026-04-01T10:00:00",
        "updated_at": "2026-04-01T10:05:00",
        "fields": {"topic": "theme"},
    }
    full_content = serialize_with_metadata({**metadata, "content": "Original preference"})
    _, expected_metadata = deserialize_full(full_content)
    await service.viking_fs.write_file(memory_uri, full_content, ctx=ctx)

    await service.fs.write(
        memory_uri,
        content="Updated preference",
        ctx=ctx,
        mode="replace",
    )

    stored = await service.viking_fs.read_file(memory_uri, ctx=ctx)
    stored_content, stored_metadata = deserialize_full(stored)

    assert stored_content == "Updated preference"
    assert stored_metadata == expected_metadata


@pytest.mark.asyncio
async def test_memory_append_preserves_metadata(service):
    ctx = RequestContext(user=service.user, role=Role.USER)
    memory_uri = f"viking://user/{ctx.user.user_space_name()}/memories/preferences/theme.md"
    metadata = {
        "tags": ["ui", "preference"],
        "created_at": "2026-04-01T10:00:00",
        "updated_at": "2026-04-01T10:05:00",
        "fields": {"topic": "theme"},
    }
    full_content = serialize_with_metadata({**metadata, "content": "Original preference"})
    _, expected_metadata = deserialize_full(full_content)
    await service.viking_fs.write_file(memory_uri, full_content, ctx=ctx)

    await service.fs.write(
        memory_uri,
        content="\nUpdated preference",
        ctx=ctx,
        mode="append",
    )

    stored = await service.viking_fs.read_file(memory_uri, ctx=ctx)
    stored_content, stored_metadata = deserialize_full(stored)

    assert stored_content == "Original preference\nUpdated preference"
    assert stored_metadata == expected_metadata


class _FakeVikingFS:
    def __init__(
        self,
        *,
        files: dict[str, str] | None = None,
        dirs: set[str] | None = None,
        not_found_exception: Exception | None = None,
    ):
        self.files = files or {}
        self.dirs = dirs or set()
        self.not_found_exception = not_found_exception
        self.write_file_calls: list[tuple[str, str]] = []

    async def stat(self, uri: str, ctx=None):
        del ctx
        if uri in self.files:
            return {"isDir": False}
        if uri in self.dirs:
            return {"isDir": True}
        if self.not_found_exception:
            raise self.not_found_exception
        raise NotFoundError(uri, "path")

    async def read_file(self, uri: str, ctx=None):
        del ctx
        if uri not in self.files:
            raise NotFoundError(uri, "file")
        return self.files[uri]

    async def write_file(self, uri: str, content: str, ctx=None):
        del ctx
        self.write_file_calls.append((uri, content))
        self.files[uri] = content


def _forbid_side_effect_helpers(monkeypatch):
    def _explode(*args, **kwargs):
        del args, kwargs
        raise AssertionError("direct content write must not use refresh infrastructure")

    monkeypatch.setattr(content_write_module, "get_lock_manager", _explode, raising=False)
    monkeypatch.setattr(content_write_module, "get_queue_manager", _explode, raising=False)
    monkeypatch.setattr(content_write_module, "get_request_wait_tracker", _explode, raising=False)
    monkeypatch.setattr(content_write_module, "vectorize_file", _explode, raising=False)


@pytest.mark.asyncio
async def test_resource_replace_is_direct_and_ignores_wait(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://resources/demo/doc.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(files={file_uri: "original"}, dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="updated",
        ctx=ctx,
        wait=True,
        timeout=0,
    )

    assert viking_fs.files[file_uri] == "updated"
    assert result == {
        "uri": file_uri,
        "root_uri": root_uri,
        "context_type": "resource",
        "mode": "replace",
        "created": False,
        "written_bytes": len("updated"),
        "content_updated": True,
        "semantic_status": "not_refreshed",
        "vector_status": "not_refreshed",
        "semantic_updated": False,
        "vector_updated": False,
        "queue_status": None,
    }


@pytest.mark.asyncio
async def test_resource_append_is_direct(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://resources/demo/doc.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(files={file_uri: "first"}, dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="\nsecond",
        ctx=ctx,
        mode="append",
    )

    assert viking_fs.files[file_uri] == "first\nsecond"
    assert result["mode"] == "append"
    assert result["created"] is False
    assert result["semantic_status"] == "not_refreshed"


@pytest.mark.asyncio
async def test_resource_create_is_direct(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://resources/demo/new.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="new content",
        mode="create",
        ctx=ctx,
        wait=True,
        timeout=0,
    )

    assert viking_fs.write_file_calls == [(file_uri, "new content")]
    assert result["context_type"] == "resource"
    assert result["created"] is True
    assert result["semantic_status"] == "not_refreshed"
    assert result["queue_status"] is None


@pytest.mark.asyncio
async def test_memory_replace_on_missing_creates_directly(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://user/default/memories/preferences/theme.md"
    root_uri = "viking://user/default/memories/preferences"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="new memory",
        ctx=ctx,
        mode="replace",
        wait=True,
    )

    assert viking_fs.files[file_uri] == "new memory"
    assert result["context_type"] == "memory"
    assert result["created"] is True
    assert result["mode"] == "replace"


@pytest.mark.asyncio
async def test_memory_append_on_missing_downgrades_to_replace(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://user/default/memories/events/fresh.md"
    root_uri = "viking://user/default/memories/events"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="Only entry.\n",
        ctx=ctx,
        mode="append",
    )

    assert viking_fs.files[file_uri] == "Only entry.\n"
    assert result["created"] is True
    assert result["mode"] == "replace"


@pytest.mark.asyncio
async def test_replace_missing_resource_still_raises_not_found(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://resources/demo/missing.md"
    root_uri = "viking://resources/demo"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    with pytest.raises(NotFoundError):
        await coordinator.write(uri=file_uri, content="missing", ctx=ctx)


@pytest.mark.asyncio
async def test_create_mode_existing_file_raises_409(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://user/default/memories/existing.md"
    root_uri = "viking://user/default/memories"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(files={file_uri: "exists"}, dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    with pytest.raises(AlreadyExistsError):
        await coordinator.write(uri=file_uri, content="content", mode="create", ctx=ctx)


@pytest.mark.asyncio
async def test_create_mode_invalid_extension_raises_400(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://user/default/memories/test.exe"
    root_uri = "viking://user/default/memories"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(dirs={root_uri})
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    with pytest.raises(InvalidArgumentError):
        await coordinator.write(uri=file_uri, content="content", mode="create", ctx=ctx)


@pytest.mark.asyncio
async def test_create_mode_treats_runtime_not_found_as_missing(monkeypatch):
    _forbid_side_effect_helpers(monkeypatch)
    file_uri = "viking://user/default/memories/new_file.md"
    root_uri = "viking://user/default/memories"
    ctx = RequestContext(user=UserIdentifier.the_default_user(), role=Role.USER)
    viking_fs = _FakeVikingFS(
        dirs={root_uri},
        not_found_exception=RuntimeError("not found: /default/user/default/memories/new_file.md"),
    )
    coordinator = ContentWriteCoordinator(viking_fs=viking_fs)

    result = await coordinator.write(
        uri=file_uri,
        content="new content",
        mode="create",
        ctx=ctx,
    )

    assert result["mode"] == "create"
    assert result["created"] is True
    assert viking_fs.write_file_calls == [(file_uri, "new content")]
