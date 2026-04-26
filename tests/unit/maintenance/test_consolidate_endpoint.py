# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for /maintenance/consolidate endpoint helpers (Phase C)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.maintenance.memory_consolidator import ConsolidationResult
from openviking.server.routers.maintenance import (
    ConsolidateRequest,
    MemoryMaintenanceRequest,
    _build_consolidator,
    _consolidation_payload,
    _maintenance_resource_id,
    _resolve_maintenance_scopes,
    _run_memory_maintenance_scopes,
)


def test_consolidate_request_defaults():
    body = ConsolidateRequest(uri="viking://agent/x/memories/patterns/")
    assert body.uri == "viking://agent/x/memories/patterns/"
    assert body.dry_run is False
    assert body.wait is True
    assert body.canaries is None


def test_consolidate_request_overrides():
    body = ConsolidateRequest(
        uri="viking://agent/x/memories/patterns/",
        dry_run=True,
        wait=False,
    )
    assert body.dry_run is True
    assert body.wait is False


def test_consolidate_request_accepts_canaries_with_top_n():
    from openviking.server.routers.maintenance import CanarySpec

    body = ConsolidateRequest(
        uri="viking://agent/x/memories/patterns/",
        canaries=[
            CanarySpec(query="strict", expected_top_uri="viking://x/a.md", top_n=1),
            CanarySpec(query="loose", expected_top_uri="viking://x/b.md"),
        ],
    )
    assert body.canaries is not None
    assert body.canaries[0].top_n == 1
    assert body.canaries[1].top_n == 5


def test_memory_maintenance_request_defaults_to_dry_run():
    body = MemoryMaintenanceRequest()
    assert body.scope is None
    assert body.dry_run is True
    assert body.wait is True
    assert body.limit == 10


def test_maintenance_resource_id_is_stable_and_collision_safe_for_commas():
    left = _maintenance_resource_id(["viking://x/a,b", "viking://x/c"])
    right = _maintenance_resource_id(["viking://x/c", "viking://x/a,b"])
    ambiguous = _maintenance_resource_id(["viking://x/a", "b", "viking://x/c"])

    assert left == right
    assert left.startswith("memory-scopes:")
    assert left != ambiguous


def test_consolidation_payload_serializes_dataclass():
    result = ConsolidationResult(
        scope_uri="viking://agent/x/memories/patterns/",
        dry_run=True,
        started_at="2026-04-19T23:00:00",
        completed_at="2026-04-19T23:00:01",
    )
    result.candidates["merge_clusters"] = 2
    result.ops_applied["merged"] = 5
    payload = _consolidation_payload(result)

    assert payload["scope_uri"] == "viking://agent/x/memories/patterns/"
    assert payload["dry_run"] is True
    assert payload["candidates"]["merge_clusters"] == 2
    assert payload["ops_applied"]["merged"] == 5
    assert "applied_uris" in payload
    assert "phase_durations" in payload


def test_build_consolidator_wires_dependencies():
    service = MagicMock()
    service.viking_fs = MagicMock()
    service.vikingdb_manager = MagicMock()
    ctx = MagicMock()
    ctx.account_id = "test-account"

    consolidator = _build_consolidator(service, ctx)

    assert consolidator is not None
    assert consolidator.viking_fs is service.viking_fs
    assert consolidator.dedup is not None
    assert consolidator.archiver is not None
    assert consolidator.service is service


@pytest.mark.asyncio
async def test_resolve_maintenance_scopes_prefers_explicit_scope():
    manager = MagicMock()
    manager.list_scopes = AsyncMock()
    ctx = MagicMock()

    scopes = await _resolve_maintenance_scopes(
        manager,
        MemoryMaintenanceRequest(scope="viking://user/u/memories/preferences/"),
        ctx,
    )

    assert scopes == ["viking://user/u/memories/preferences/"]
    manager.list_scopes.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_maintenance_scopes_lists_dirty_scopes():
    scope = MagicMock()
    scope.scope_uri = "viking://agent/a/memories/patterns/"
    manager = MagicMock()
    manager.list_scopes = AsyncMock(return_value=[scope])
    ctx = MagicMock()
    ctx.account_id = "acc"
    ctx.user.user_id = "user"

    scopes = await _resolve_maintenance_scopes(
        manager,
        MemoryMaintenanceRequest(limit=1),
        ctx,
    )

    assert scopes == ["viking://agent/a/memories/patterns/"]
    manager.list_scopes.assert_awaited_once_with(
        active_only=True,
        account_id="acc",
        user_id="user",
        limit=1,
    )


@pytest.mark.asyncio
async def test_run_memory_maintenance_continues_after_scope_failure(monkeypatch):
    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run, canaries, target_uris=None):
            if scope_uri.endswith("/bad/"):
                raise RuntimeError("bad scope")
            return ConsolidationResult(
                scope_uri=scope_uri,
                dry_run=dry_run,
                started_at="2026-04-19T23:00:00",
                completed_at="2026-04-19T23:00:01",
            )

    manager = MagicMock()
    manager.get_scope = AsyncMock(return_value=None)
    manager.mark_run_complete = AsyncMock()
    manager.mark_run_failed = AsyncMock()
    monkeypatch.setattr(
        "openviking.server.routers.maintenance._build_consolidator",
        lambda service, ctx: FakeConsolidator(),
    )

    with pytest.raises(RuntimeError, match="bad scope"):
        await _run_memory_maintenance_scopes(
            MagicMock(),
            manager,
            [
                "viking://user/u/memories/bad/",
                "viking://user/u/memories/good/",
            ],
            True,
            MagicMock(),
        )

    manager.mark_run_failed.assert_awaited_once()
    manager.mark_run_complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_memory_maintenance_passes_dirty_uris_to_consolidator(monkeypatch):
    captured = {}

    class FakeConsolidator:
        async def run(self, scope_uri, ctx, *, dry_run, canaries, target_uris=None):
            captured["scope_uri"] = scope_uri
            captured["target_uris"] = target_uris
            return ConsolidationResult(
                scope_uri=scope_uri,
                dry_run=dry_run,
                started_at="2026-04-19T23:00:00",
                completed_at="2026-04-19T23:00:01",
            )

    scope = MagicMock()
    scope.dirty_uris = ["viking://agent/a/memories/skills/changed.md"]
    manager = MagicMock()
    manager.get_scope = AsyncMock(return_value=scope)
    manager.mark_run_complete = AsyncMock()
    manager.mark_run_failed = AsyncMock()
    monkeypatch.setattr(
        "openviking.server.routers.maintenance._build_consolidator",
        lambda service, ctx: FakeConsolidator(),
    )

    await _run_memory_maintenance_scopes(
        MagicMock(),
        manager,
        ["viking://agent/a/memories/skills/"],
        True,
        MagicMock(),
    )

    assert captured == {
        "scope_uri": "viking://agent/a/memories/skills/",
        "target_uris": ["viking://agent/a/memories/skills/changed.md"],
    }


class TestListRunsParsesViking_FSEntries:
    """Regression: viking_fs.ls returns List[Dict] with 'uri' key, not bare strings.

    Earlier impl did `[e for e in entries if isinstance(e, str)]` which silently
    filtered everything out. This test pins the dict-shaped contract.
    """

    def test_filter_extracts_uri_from_dict_entries(self):
        entries = [
            {"uri": "viking://x/run1.json", "isDir": False, "size": 100},
            {"uri": "viking://x/run2.json", "isDir": False, "size": 200},
            {"uri": "viking://x/.overview.md", "isDir": False, "size": 50},
            {"uri": "viking://x/subdir", "isDir": True, "size": 0},
        ]
        # Mirror the filter in list_consolidate_runs.
        file_uris = []
        for entry in entries:
            if isinstance(entry, dict):
                uri = entry.get("uri", "")
                is_dir = entry.get("isDir", False)
            else:
                uri = str(entry)
                is_dir = False
            if not uri or is_dir or not uri.endswith(".json"):
                continue
            file_uris.append(uri)
        assert file_uris == ["viking://x/run1.json", "viking://x/run2.json"]

    def test_filter_handles_string_fallback(self):
        # Defensive: if some other backend returns bare strings, still works.
        entries = ["viking://x/run.json", "viking://x/other.md"]
        file_uris = []
        for entry in entries:
            if isinstance(entry, dict):
                uri = entry.get("uri", "")
                is_dir = entry.get("isDir", False)
            else:
                uri = str(entry)
                is_dir = False
            if not uri or is_dir or not uri.endswith(".json"):
                continue
            file_uris.append(uri)
        assert file_uris == ["viking://x/run.json"]
