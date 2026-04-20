# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for /maintenance/consolidate endpoint helpers (Phase C)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.maintenance.memory_consolidator import ConsolidationResult
from openviking.server.routers.maintenance import (
    ConsolidateRequest,
    _build_consolidator,
    _consolidation_payload,
)


def test_consolidate_request_defaults():
    body = ConsolidateRequest(uri="viking://agent/x/memories/patterns/")
    assert body.uri == "viking://agent/x/memories/patterns/"
    assert body.dry_run is False
    assert body.wait is True


def test_consolidate_request_overrides():
    body = ConsolidateRequest(
        uri="viking://agent/x/memories/patterns/",
        dry_run=True,
        wait=False,
    )
    assert body.dry_run is True
    assert body.wait is False


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
