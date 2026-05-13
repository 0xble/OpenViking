# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for maintenance endpoints."""


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
