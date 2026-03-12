# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Tests for search endpoints: find, search, grep, glob."""

from datetime import datetime, timezone

import httpx

from openviking.utils.time_utils import parse_iso_datetime


async def test_find_basic(client_with_resource):
    client, uri = client_with_resource
    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "sample document", "limit": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"] is not None


async def test_find_with_target_uri(client_with_resource):
    client, uri = client_with_resource
    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "sample", "target_uri": uri, "limit": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_find_with_score_threshold(client_with_resource):
    client, uri = client_with_resource
    resp = await client.post(
        "/api/v1/search/find",
        json={
            "query": "sample document",
            "score_threshold": 0.01,
            "limit": 10,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_find_no_results(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "completely_random_nonexistent_xyz123"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_find_with_since_compiles_time_range(client: httpx.AsyncClient, service, monkeypatch):
    captured = {}

    async def fake_find(*, filter=None, **kwargs):
        captured["filter"] = filter
        captured["kwargs"] = kwargs
        return {"items": []}

    monkeypatch.setattr(service.search, "find", fake_find)

    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "sample", "since": "2h"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert captured["filter"]["op"] == "time_range"
    assert captured["filter"]["field"] == "updated_at"
    gte = parse_iso_datetime(captured["filter"]["gte"])
    delta = datetime.now(timezone.utc) - gte
    assert 7_100 <= delta.total_seconds() <= 7_300


async def test_find_combines_existing_filter_with_time_range(
    client: httpx.AsyncClient, service, monkeypatch
):
    captured = {}

    async def fake_find(*, filter=None, **kwargs):
        captured["filter"] = filter
        return {"items": []}

    monkeypatch.setattr(service.search, "find", fake_find)

    resp = await client.post(
        "/api/v1/search/find",
        json={
            "query": "sample",
            "filter": {"op": "must", "field": "kind", "conds": ["email"]},
            "since": "2026-03-10",
            "time_field": "created_at",
        },
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert captured["filter"] == {
        "op": "and",
        "conds": [
            {"op": "must", "field": "kind", "conds": ["email"]},
            {
                "op": "time_range",
                "field": "created_at",
                "gte": "2026-03-10T00:00:00.000",
            },
        ],
    }


async def test_find_with_invalid_time_returns_422(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "sample", "since": "not-a-time"},
    )

    assert resp.status_code == 422
    assert resp.json()["detail"]


async def test_find_with_inverted_mixed_time_range_returns_422(client: httpx.AsyncClient):
    resp = await client.post(
        "/api/v1/search/find",
        json={"query": "sample", "since": "2099-01-01", "until": "2h"},
    )

    assert resp.status_code == 422
    assert "earlier than or equal to" in resp.json()["detail"]


async def test_search_basic(client_with_resource):
    client, uri = client_with_resource
    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "sample document", "limit": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["result"] is not None


async def test_search_with_session(client_with_resource):
    client, uri = client_with_resource
    # Create a session first
    sess_resp = await client.post("/api/v1/sessions", json={"user": "test"})
    session_id = sess_resp.json()["result"]["session_id"]

    resp = await client.post(
        "/api/v1/search/search",
        json={
            "query": "sample",
            "session_id": session_id,
            "limit": 5,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_search_with_until_compiles_time_range(
    client: httpx.AsyncClient, service, monkeypatch
):
    captured = {}

    async def fake_search(*, filter=None, **kwargs):
        captured["filter"] = filter
        return {"items": []}

    monkeypatch.setattr(service.search, "search", fake_search)

    resp = await client.post(
        "/api/v1/search/search",
        json={"query": "sample", "until": "2026-03-11", "time_field": "created_at"},
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert captured["filter"] == {
        "op": "time_range",
        "field": "created_at",
        "lte": "2026-03-11T23:59:59.999",
    }


async def test_grep(client_with_resource):
    client, uri = client_with_resource
    parent_uri = "/".join(uri.split("/")[:-1]) + "/"
    resp = await client.post(
        "/api/v1/search/grep",
        json={"uri": parent_uri, "pattern": "Sample"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_grep_case_insensitive(client_with_resource):
    client, uri = client_with_resource
    parent_uri = "/".join(uri.split("/")[:-1]) + "/"
    resp = await client.post(
        "/api/v1/search/grep",
        json={
            "uri": parent_uri,
            "pattern": "sample",
            "case_insensitive": True,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


async def test_glob(client_with_resource):
    client, _ = client_with_resource
    resp = await client.post(
        "/api/v1/search/glob",
        json={"pattern": "*.md"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
