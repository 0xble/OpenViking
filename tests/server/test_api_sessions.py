# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for session endpoints."""

import json

import httpx

from openviking.server.identity import RequestContext, Role


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
    assert resp.json()["result"]["stats"]["total_turns"] == 1


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


async def test_get_imported_session_by_raw_source_id(client: httpx.AsyncClient, tmp_path):
    raw = tmp_path / "codex-raw-get.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"import-raw-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import_resp = await client.post(
        "/api/v1/sessions/import",
        json={"adapter": "codex", "path": str(raw), "build_index": False},
    )
    assert import_resp.status_code == 200

    resp = await client.get("/api/v1/sessions/import-raw-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["session_id"] == "codex-import-raw-123"
    assert body["result"]["message_count"] == 2


async def test_imported_session_creates_source_alias_index(
    client: httpx.AsyncClient, service, tmp_path
):
    raw = tmp_path / "codex-alias-index.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"alias-index-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import_resp = await client.post(
        "/api/v1/sessions/import",
        json={"adapter": "codex", "path": str(raw), "build_index": False},
    )
    assert import_resp.status_code == 200

    ctx = RequestContext(user=service.user, role=Role.ROOT)
    alias_uri = service.sessions._source_alias_uri("alias-index-123", ctx)
    alias_payload = json.loads(await service.sessions._viking_fs.read_file(alias_uri, ctx=ctx))
    assert alias_payload["source_session_id"] == "alias-index-123"
    assert alias_payload["session_id"] == "codex-alias-index-123"
    assert alias_payload["adapter"] == "codex"

    list_resp = await client.get("/api/v1/sessions")
    assert list_resp.status_code == 200
    listed_ids = {entry["session_id"] for entry in list_resp.json()["result"]}
    assert ".aliases" not in listed_ids


async def test_raw_source_lookup_backfills_missing_alias_index(
    client: httpx.AsyncClient, service, monkeypatch, tmp_path
):
    raw = tmp_path / "codex-alias-backfill.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"alias-backfill-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import_resp = await client.post(
        "/api/v1/sessions/import",
        json={"adapter": "codex", "path": str(raw), "build_index": False},
    )
    assert import_resp.status_code == 200

    ctx = RequestContext(user=service.user, role=Role.ROOT)
    alias_uri = service.sessions._source_alias_uri("alias-backfill-123", ctx)
    await service.sessions._viking_fs.rm(alias_uri, ctx=ctx)

    original_ls = service.sessions._viking_fs.ls
    observed_node_limits: list[int] = []

    async def recording_ls(uri: str, *args, **kwargs):
        observed_node_limits.append(kwargs.get("node_limit"))
        return await original_ls(uri, *args, **kwargs)

    monkeypatch.setattr(service.sessions._viking_fs, "ls", recording_ls)

    resp = await client.get("/api/v1/sessions/alias-backfill-123")
    assert resp.status_code == 200
    assert resp.json()["result"]["session_id"] == "codex-alias-backfill-123"
    assert 1_000_000 in observed_node_limits

    alias_payload = json.loads(await service.sessions._viking_fs.read_file(alias_uri, ctx=ctx))
    assert alias_payload["session_id"] == "codex-alias-backfill-123"


async def test_add_message_by_raw_source_id_reuses_imported_session(
    client: httpx.AsyncClient, service, tmp_path
):
    raw = tmp_path / "codex-raw-add.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"import-raw-add-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    import_resp = await client.post(
        "/api/v1/sessions/import",
        json={"adapter": "codex", "path": str(raw), "build_index": False},
    )
    assert import_resp.status_code == 200

    add_resp = await client.post(
        "/api/v1/sessions/import-raw-add-123/messages",
        json={"role": "user", "content": "follow-up"},
    )
    assert add_resp.status_code == 200
    add_body = add_resp.json()
    assert add_body["status"] == "ok"
    assert add_body["result"]["session_id"] == "codex-import-raw-add-123"
    assert add_body["result"]["message_count"] == 3

    ctx = RequestContext(user=service.user, role=Role.ROOT)
    canonical_session = await service.sessions.get("import-raw-add-123", ctx)
    assert canonical_session.session_id == "codex-import-raw-add-123"
    assert len(canonical_session.messages) == 3

    literal_session = service.sessions.session(ctx, "import-raw-add-123")
    assert not await literal_session.exists()


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


async def test_import_session_log_skips_index_for_low_signal_first_prompt(
    client: httpx.AsyncClient, service, monkeypatch, tmp_path
):
    raw = tmp_path / "codex-low-signal.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"low-signal-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Tell me what model you are"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"I am a model."}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    vectorized = {"called": False}

    async def fake_vectorize(*args, **kwargs):
        vectorized["called"] = True

    monkeypatch.setattr("openviking.session.session.vectorize_directory_meta", fake_vectorize)

    resp = await client.post(
        "/api/v1/sessions/import",
        json={
            "adapter": "codex",
            "path": str(raw),
            "build_index": True,
            "preserve_original": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] == "imported"
    assert body["result"]["indexed"] is False
    assert body["result"]["index_skip_category"] == "model-check"
    assert body["result"]["index_skip_reason"] == "low-signal first user prompt (model-check)"
    assert vectorized["called"] is False

    ctx = RequestContext(user=service.user, role=Role.ROOT)
    source_uri = "viking://session/test_user/codex-low-signal-123/.source.json"
    source_payload = json.loads(await service.sessions._viking_fs.read_file(source_uri, ctx=ctx))
    assert source_payload["index_eligible"] is False
    assert source_payload["index_skip_category"] == "model-check"


async def test_import_session_log_indexes_when_real_followup_exists(
    client: httpx.AsyncClient, monkeypatch, tmp_path
):
    raw = tmp_path / "codex-followup-index.jsonl"
    raw.write_text(
        "\n".join(
            [
                '{"type":"session_meta","payload":{"id":"followup-123","cwd":"/tmp/project"}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}}',
                '{"type":"response_item","timestamp":"2026-03-09T12:00:02Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"Can you walk me through why my Redis cache is serving stale data after deploys?"}]}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    vectorized = {"called": False}

    async def fake_vectorize(*args, **kwargs):
        vectorized["called"] = True

    monkeypatch.setattr("openviking.session.session.vectorize_directory_meta", fake_vectorize)

    resp = await client.post(
        "/api/v1/sessions/import",
        json={
            "adapter": "codex",
            "path": str(raw),
            "build_index": True,
            "preserve_original": True,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["status"] == "imported"
    assert body["result"]["indexed"] is True
    assert vectorized["called"] is True
