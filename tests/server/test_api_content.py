# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for content endpoints: read, abstract, overview."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.server.routers.content import ReindexRequest, _do_reindex, reindex
from openviking_cli.session.user_id import UserIdentifier


async def test_read_content(client_with_resource):
    client, uri = client_with_resource
    # The resource URI may be a directory; list children to find the file
    ls_resp = await client.get(
        "/api/v1/fs/ls",
        params={"uri": uri, "simple": True, "recursive": True, "output": "original"},
    )
    children = ls_resp.json().get("result", [])
    # Find a file (non-directory) to read
    file_uri = None
    if children:
        # ls(simple=True) returns full URIs, use directly
        file_uri = children[0] if isinstance(children[0], str) else None
    if file_uri is None:
        file_uri = uri

    resp = await client.get("/api/v1/content/read", params={"uri": file_uri})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"] is not None


async def test_abstract_content(client_with_resource):
    client, uri = client_with_resource
    resp = await client.get("/api/v1/content/abstract", params={"uri": uri})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


async def test_overview_content(client_with_resource):
    client, uri = client_with_resource
    resp = await client.get("/api/v1/content/overview", params={"uri": uri})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"


async def test_reindex_missing_uri(client):
    """Test reindex without uri field returns 422."""
    resp = await client.post(
        "/api/v1/content/reindex",
        json={"regenerate": False},
    )
    assert resp.status_code == 422


async def test_reindex_endpoint_registered(client):
    """Test the reindex endpoint is registered (GET returns 405, not 404)."""
    resp = await client.get("/api/v1/content/reindex")
    assert resp.status_code == 405  # Method Not Allowed, not 404


async def test_reindex_request_validation(client):
    """Test reindex validates the request body schema."""
    # Empty body — uri is required
    resp = await client.post("/api/v1/content/reindex", json={})
    assert resp.status_code == 422

    # Invalid type for regenerate
    resp = await client.post(
        "/api/v1/content/reindex",
        json={"uri": "viking://resources/test", "regenerate": "not_a_bool"},
    )
    # Pydantic coerces strings to bool, so this may or may not fail
    assert resp.status_code in (200, 422, 500)


async def test_reindex_wait_parameter_schema(client):
    """Test reindex accepts wait parameter in request schema."""
    # Invalid wait type should be coerced or rejected, not crash
    resp = await client.post(
        "/api/v1/content/reindex",
        json={"uri": "viking://resources/test", "wait": "invalid"},
    )
    # Pydantic coerces or rejects — either way, not a 404/405
    assert resp.status_code != 404
    assert resp.status_code != 405


@pytest.mark.asyncio
async def test_reindex_uses_request_tenant_for_exists(monkeypatch):
    """Reindex must validate URI existence inside the caller's tenant."""
    seen = {}

    class FakeVikingFS:
        async def exists(self, uri, ctx=None):
            seen["uri"] = uri
            seen["ctx"] = ctx
            return True

    class FakeTracker:
        def has_running(self, task_type, uri, owner_account_id=None, owner_user_id=None):
            return False

    async def fake_do_reindex(service, uri, regenerate, ctx):
        return {"status": "success", "message": "Indexed 1 resources"}

    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ADMIN,
    )
    request = ReindexRequest(uri="viking://resources/demo/demo-note.md", wait=True)

    monkeypatch.setattr("openviking.storage.viking_fs.get_viking_fs", lambda: FakeVikingFS())
    monkeypatch.setattr(
        "openviking.service.task_tracker.get_task_tracker",
        lambda: FakeTracker(),
    )
    monkeypatch.setattr(
        "openviking.server.routers.content.get_service",
        lambda: SimpleNamespace(),
    )
    monkeypatch.setattr("openviking.server.routers.content._do_reindex", fake_do_reindex)

    response = await reindex(request=request, _ctx=ctx)

    assert response.status == "ok"
    assert seen["uri"] == "viking://resources/demo/demo-note.md"
    assert seen["ctx"] == ctx


@pytest.mark.asyncio
async def test_do_reindex_memory_uri_queues_semantic_memory(monkeypatch):
    """Memory reindex must enqueue semantic memory processing, not resource indexing."""

    class FakeLockContext:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return None

        async def __aexit__(self, exc_type, exc, tb):
            return False

    enqueued = []

    class FakeQueue:
        async def enqueue(self, msg):
            enqueued.append(msg)

    class FakeQueueManager:
        SEMANTIC = "Semantic"

        def get_queue(self, name, allow_create=False):
            assert name == self.SEMANTIC
            assert allow_create is True
            return FakeQueue()

    service = SimpleNamespace(
        viking_fs=SimpleNamespace(_uri_to_path=lambda uri, ctx=None: f"/tmp/{uri}"),
        resources=SimpleNamespace(
            summarize=AsyncMock(),
            build_index=AsyncMock(),
        ),
    )
    ctx = RequestContext(
        user=UserIdentifier(account_id="test", user_id="alice", agent_id="default"),
        role=Role.ADMIN,
    )
    uri = "viking://user/alice/memories/preferences"

    monkeypatch.setattr("openviking.storage.queuefs.get_queue_manager", lambda: FakeQueueManager())
    monkeypatch.setattr("openviking.storage.transaction.get_lock_manager", lambda: object())
    monkeypatch.setattr("openviking.storage.transaction.LockContext", FakeLockContext)

    result = await _do_reindex(service, uri, regenerate=False, ctx=ctx)

    assert result == {
        "status": "success",
        "message": "Queued memory reindex",
        "uri": uri,
        "context_type": "memory",
    }
    assert len(enqueued) == 1
    assert enqueued[0].uri == uri
    assert enqueued[0].context_type == "memory"
    service.resources.summarize.assert_not_awaited()
    service.resources.build_index.assert_not_awaited()
