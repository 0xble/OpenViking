# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for maintenance endpoints."""

from openviking.maintenance import MemoryMaintenanceManager
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
