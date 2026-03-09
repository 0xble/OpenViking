# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for session endpoints."""

import httpx

from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


async def test_create_session(client: httpx.AsyncClient):
    resp = await client.post("/api/v1/sessions", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "session_id" in body["result"]


async def test_list_sessions(client: httpx.AsyncClient):
    # Create a session first
    await client.post("/api/v1/sessions", json={})
    resp = await client.get("/api/v1/sessions")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert isinstance(body["result"], list)


async def test_get_session(client: httpx.AsyncClient):
    create_resp = await client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["result"]["session_id"]

    resp = await client.get(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["session_id"] == session_id


async def test_add_message(client: httpx.AsyncClient):
    create_resp = await client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["result"]["session_id"]

    resp = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Hello, world!"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["message_count"] == 1


async def test_add_multiple_messages(client: httpx.AsyncClient):
    create_resp = await client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["result"]["session_id"]

    # Add messages one by one; each add_message call should see
    # the accumulated count (messages are loaded from storage each time)
    resp1 = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Message 0"},
    )
    assert resp1.json()["result"]["message_count"] >= 1

    resp2 = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Message 1"},
    )
    count2 = resp2.json()["result"]["message_count"]

    resp3 = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Message 2"},
    )
    count3 = resp3.json()["result"]["message_count"]

    # Each add should increase the count
    assert count3 >= count2


async def test_add_message_persistence_regression(client: httpx.AsyncClient, service):
    """Regression: message payload must persist as valid parts across loads."""
    create_resp = await client.post("/api/v1/sessions", json={"user": "test"})
    session_id = create_resp.json()["result"]["session_id"]

    resp1 = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Message A"},
    )
    assert resp1.status_code == 200
    assert resp1.json()["result"]["message_count"] == 1

    resp2 = await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Message B"},
    )
    assert resp2.status_code == 200
    assert resp2.json()["result"]["message_count"] == 2

    # Re-load through API path to ensure session file can be parsed back.
    get_resp = await client.get(f"/api/v1/sessions/{session_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["result"]["message_count"] == 2

    # Verify stored message content survives load/decode.
    ctx = RequestContext(user=service.user, role=Role.ROOT)
    session = service.sessions.session(ctx, session_id)
    await session.load()
    assert len(session.messages) == 2
    assert session.messages[0].content == "Message A"
    assert session.messages[1].content == "Message B"


async def test_delete_session(client: httpx.AsyncClient):
    create_resp = await client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["result"]["session_id"]

    # Add a message so the session file exists in storage
    await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "ensure persisted"},
    )
    # Compress to persist
    await client.post(f"/api/v1/sessions/{session_id}/commit")

    resp = await client.delete(f"/api/v1/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_compress_session(client: httpx.AsyncClient):
    create_resp = await client.post("/api/v1/sessions", json={})
    session_id = create_resp.json()["result"]["session_id"]

    # Add some messages before committing
    await client.post(
        f"/api/v1/sessions/{session_id}/messages",
        json={"role": "user", "content": "Hello"},
    )

    resp = await client.post(f"/api/v1/sessions/{session_id}/commit")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_extract_session_jsonable_regression(client: httpx.AsyncClient, service, monkeypatch):
    """Regression: extract endpoint should serialize internal objects."""

    class FakeMemory:
        __slots__ = ("uri",)

        def __init__(self, uri: str):
            self.uri = uri

        def to_dict(self):
            return {"uri": self.uri}

    async def fake_extract(_session_id: str, _ctx):
        return [FakeMemory("viking://user/memories/mock.md")]

    monkeypatch.setattr(service.sessions, "extract", fake_extract)

    create_resp = await client.post("/api/v1/sessions", json={"user": "test"})
    session_id = create_resp.json()["result"]["session_id"]

    resp = await client.post(f"/api/v1/sessions/{session_id}/extract")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"] == [{"uri": "viking://user/memories/mock.md"}]


async def test_import_session_log(client: httpx.AsyncClient, service, tmp_path):
    raw = tmp_path / "codex.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"import-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resp = await client.post(
        "/api/v1/sessions/import",
        json={
            "adapter": "codex",
            "path": str(raw),
            "build_index": False,
            "preserve_original": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] == "imported"
    assert body["result"]["session_id"] == "codex-import-123"
    assert body["result"]["message_count"] == 2

    ctx = RequestContext(user=service.user, role=Role.ROOT)
    session = await service.sessions.get("codex-import-123", ctx)
    assert len(session.messages) == 2
    assert session.messages[0].content == "hello"


async def test_import_session_indexes_as_resource(client: httpx.AsyncClient, monkeypatch, tmp_path):
    raw = tmp_path / "openclaw-indexed.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session","id":"sess-index","cwd":"/tmp/project","version":"1"}',
                '{"type":"message","timestamp":"2026-03-09T12:10:00Z","message":{"role":"user","content":[{"type":"text","text":"ping"}]}}',
                '{"type":"message","timestamp":"2026-03-09T12:10:01Z","message":{"role":"assistant","content":[{"type":"text","text":"pong"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    recorded = {}

    async def fake_vectorize(uri, abstract, overview, context_type, ctx):
        recorded["uri"] = uri
        recorded["abstract"] = abstract
        recorded["overview"] = overview
        recorded["context_type"] = context_type
        recorded["ctx_user"] = ctx.user.user_space_name()

    monkeypatch.setattr("openviking.session.session.vectorize_directory_meta", fake_vectorize)

    resp = await client.post(
        "/api/v1/sessions/import",
        json={
            "adapter": "openclaw",
            "path": str(raw),
            "build_index": True,
            "preserve_original": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] == "imported"
    assert recorded["uri"].endswith("/openclaw-sess-index")
    assert recorded["uri"].startswith("viking://session/")
    assert recorded["context_type"] == "resource"
    assert recorded["ctx_user"] == "test_user"


async def test_import_session_log_skips_hook_only_claude(client: httpx.AsyncClient, tmp_path):
    raw = tmp_path / "claude-hook-only.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"progress","sessionId":"claude-empty","cwd":"/Users/brianle","timestamp":"2026-03-09T00:45:57.290Z","version":"2.1.71","data":{"type":"hook_progress"}}',
                '{"type":"progress","sessionId":"claude-empty","cwd":"/Users/brianle","timestamp":"2026-03-09T00:45:58.290Z","version":"2.1.71","data":{"type":"hook_progress"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    resp = await client.post(
        "/api/v1/sessions/import",
        json={
            "adapter": "claude",
            "path": str(raw),
            "build_index": False,
            "preserve_original": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] == "skipped"
    assert body["result"]["session_id"] == "claude-claude-empty"
    assert body["result"]["reason"] == "no normalizable messages"
