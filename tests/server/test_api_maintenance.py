# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for maintenance endpoints."""

from openviking.maintenance import MemoryMaintenanceManager
from openviking.maintenance.memory_consolidator import ConsolidationResult
from openviking.server.identity import RequestContext, Role
from openviking_cli.session.user_id import UserIdentifier


def _memory_diff(uri: str) -> dict:
    return {
        "archive_uri": "viking://session/test/_archive/memory/1",
        "operations": {
            "adds": [{"uri": uri}],
            "updates": [],
            "deletes": [],
        },
    }


async def test_memory_maintenance_scopes_endpoint_registered(client):
    resp = await client.get("/api/v1/maintenance/memory/scopes")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"]["active_only"] is True
    assert body["result"]["scopes"] == []


async def test_memory_maintenance_run_empty_dirty_scopes(client):
    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={"dry_run": True, "wait": True, "limit": 10},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"] == {
        "status": "completed",
        "dry_run": True,
        "scopes": [],
        "runs": [],
    }


async def test_memory_maintenance_scopes_filters_current_agent(client, service):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    current_agent_uri = "viking://agent/default/memories/preferences/editor.md"
    other_agent_uri = "viking://agent/research/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(current_agent_uri),
        RequestContext(
            user=UserIdentifier("test_account", "test_user", "default"),
            role=Role.ROOT,
        ),
    )
    await manager.record_memory_diff(
        _memory_diff(other_agent_uri),
        RequestContext(
            user=UserIdentifier("test_account", "test_user", "research"),
            role=Role.ROOT,
        ),
    )

    resp = await client.get("/api/v1/maintenance/memory/scopes")

    assert resp.status_code == 200
    scopes = resp.json()["result"]["scopes"]
    assert [scope["scope_uri"] for scope in scopes] == [
        "viking://agent/default/memories/preferences/"
    ]


async def test_memory_maintenance_shared_user_scope_visible_across_agents(
    client,
    service,
    monkeypatch,
):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    memory_uri = "viking://user/test_user/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(memory_uri),
        RequestContext(
            user=UserIdentifier("test_account", "test_user", "research"),
            role=Role.ROOT,
        ),
    )
    seen = {}

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run=False, target_uris=None):
            seen["scope_uri"] = scope_uri
            seen["target_uris"] = target_uris
            return ConsolidationResult(scope_uri=scope_uri, dry_run=dry_run)

    monkeypatch.setattr(
        "openviking.server.routers.maintenance._consolidator",
        lambda: FakeConsolidator(),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={"dry_run": True, "wait": True, "limit": 10},
    )

    assert resp.status_code == 200
    assert seen["scope_uri"] == "viking://user/test_user/memories/preferences/"
    assert seen["target_uris"] == [memory_uri]


async def test_memory_maintenance_shared_agent_scope_visible_across_users(
    client,
    service,
    monkeypatch,
):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    memory_uri = "viking://agent/default/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(memory_uri),
        RequestContext(
            user=UserIdentifier("test_account", "other_user", "default"),
            role=Role.ROOT,
        ),
    )
    seen = {}

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run=False, target_uris=None):
            seen["scope_uri"] = scope_uri
            seen["target_uris"] = target_uris
            return ConsolidationResult(scope_uri=scope_uri, dry_run=dry_run)

    monkeypatch.setattr(
        "openviking.server.routers.maintenance._consolidator",
        lambda: FakeConsolidator(),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={"dry_run": True, "wait": True, "limit": 10},
    )

    assert resp.status_code == 200
    assert seen["scope_uri"] == "viking://agent/default/memories/preferences/"
    assert seen["target_uris"] == [memory_uri]


async def test_memory_maintenance_run_rejects_nonblocking_request(client):
    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={"dry_run": True, "wait": False, "limit": 10},
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert "wait=false is not supported" in body["error"]["message"]


async def test_memory_maintenance_explicit_scope_rejects_foreign_tenant(client, service):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    foreign_uri = "viking://user/foreign_user/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(foreign_uri),
        RequestContext(
            user=UserIdentifier("foreign_account", "foreign_user", "default"),
            role=Role.ROOT,
        ),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={
            "scope": "viking://user/foreign_user/memories/preferences/",
            "dry_run": True,
            "wait": True,
            "limit": 10,
        },
    )

    assert resp.status_code == 403
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "PERMISSION_DENIED"
    assert "does not belong" in body["error"]["message"]


async def test_memory_maintenance_explicit_missing_scope_rejects_foreign_tenant(client):
    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={
            "scope": "viking://user/foreign_user/memories/preferences/",
            "dry_run": True,
            "wait": True,
            "limit": 10,
        },
    )

    assert resp.status_code == 403
    body = resp.json()
    assert body["status"] == "error"
    assert body["error"]["code"] == "PERMISSION_DENIED"
    assert "does not belong" in body["error"]["message"]


async def test_memory_maintenance_explicit_scope_canonicalizes_lookup(
    client,
    service,
    monkeypatch,
):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    memory_uri = "viking://user/test_user/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(memory_uri),
        RequestContext(
            user=UserIdentifier("test_account", "test_user", "default"),
            role=Role.ROOT,
        ),
    )
    seen = {}

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run=False, target_uris=None):
            seen["scope_uri"] = scope_uri
            seen["target_uris"] = target_uris
            return ConsolidationResult(scope_uri=scope_uri, dry_run=dry_run)

    monkeypatch.setattr(
        "openviking.server.routers.maintenance._consolidator",
        lambda: FakeConsolidator(),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={
            "scope": "viking://user/test_user/memories/preferences",
            "dry_run": True,
            "wait": True,
            "limit": 10,
        },
    )

    assert resp.status_code == 200
    assert seen["scope_uri"] == "viking://user/test_user/memories/preferences/"
    assert seen["target_uris"] == [memory_uri]


async def test_memory_maintenance_explicit_clean_scope_preserves_tenant(
    client,
    service,
    monkeypatch,
):
    seen = {}

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run=False, target_uris=None):
            seen["target_uris"] = target_uris
            return ConsolidationResult(scope_uri=scope_uri, dry_run=dry_run)

    monkeypatch.setattr(
        "openviking.server.routers.maintenance._consolidator",
        lambda: FakeConsolidator(),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={
            "scope": "viking://user/test_user/memories/preferences/",
            "dry_run": False,
            "wait": True,
            "limit": 10,
        },
    )

    assert resp.status_code == 200
    assert seen["target_uris"] is None

    reloaded = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    scope = await reloaded.get_scope("viking://user/test_user/memories/preferences/")
    assert scope is not None
    assert scope.account_id == "test_account"
    assert scope.user_id == "test_user"
    assert scope.agent_id == "default"
    assert scope.is_active is False


async def test_memory_maintenance_partial_result_keeps_scope_dirty(
    client,
    service,
    monkeypatch,
):
    manager = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    memory_uri = "viking://user/test_user/memories/preferences/editor.md"
    await manager.record_memory_diff(
        _memory_diff(memory_uri),
        RequestContext(
            user=UserIdentifier("test_account", "test_user", "default"),
            role=Role.ROOT,
        ),
    )

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run=False, target_uris=None):
            result = ConsolidationResult(scope_uri=scope_uri, dry_run=dry_run)
            result.partial = True
            result.errors = ["cluster failed"]
            return result

    monkeypatch.setattr(
        "openviking.server.routers.maintenance._consolidator",
        lambda: FakeConsolidator(),
    )

    resp = await client.post(
        "/api/v1/maintenance/memory/run",
        json={"dry_run": False, "wait": True, "limit": 10},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["result"]["status"] == "error"

    reloaded = MemoryMaintenanceManager(viking_fs=service.viking_fs)
    scopes = await reloaded.list_scopes(
        active_only=True, account_id="test_account", user_id="test_user"
    )
    assert len(scopes) == 1
    assert scopes[0].dirty_uris == [memory_uri]
    assert scopes[0].last_error == "cluster failed"
