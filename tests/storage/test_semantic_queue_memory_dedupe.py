# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for parent-semantic enqueue deduplication (#769, #505)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.storage.queuefs.named_queue import NamedQueue
from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_queue import SemanticQueue


@pytest.mark.asyncio
async def test_memory_semantic_enqueue_deduped_within_window():
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        msg = SemanticMsg(
            uri="viking://user/default/memories/entities",
            context_type="memory",
            account_id="acc",
            user_id="u1",
            agent_id="a1",
        )
        r1 = await q.enqueue(msg)
        r2 = await q.enqueue(
            SemanticMsg(
                uri="viking://user/default/memories/entities",
                context_type="memory",
                account_id="acc",
                user_id="u1",
                agent_id="a1",
            )
        )
        assert r1 == "queued-id"
        assert r2 == "deduplicated"
        assert named_enqueue.call_count == 1


@pytest.mark.asyncio
async def test_memory_semantic_enqueue_different_uri_not_deduped():
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        await q.enqueue(
            SemanticMsg(
                uri="viking://user/default/memories/entities",
                context_type="memory",
            )
        )
        await q.enqueue(
            SemanticMsg(
                uri="viking://user/default/memories/patterns",
                context_type="memory",
            )
        )
        assert named_enqueue.call_count == 2


@pytest.mark.asyncio
async def test_resource_semantic_enqueue_deduped_within_window():
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        uri = "viking://resources/slack/canvases/abc"
        r1 = await q.enqueue(
            SemanticMsg(uri=uri, context_type="resource", account_id="acc", user_id="u1")
        )
        r2 = await q.enqueue(
            SemanticMsg(uri=uri, context_type="resource", account_id="acc", user_id="u1")
        )
        assert r1 == "queued-id"
        assert r2 == "deduplicated"
        assert named_enqueue.call_count == 1


@pytest.mark.asyncio
async def test_session_semantic_enqueue_deduped_within_window():
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        uri = "viking://session/claude-019dd0ab-192f-77eb-b3fd-68436e5bcc72"
        r1 = await q.enqueue(
            SemanticMsg(uri=uri, context_type="session", account_id="acc", user_id="u1")
        )
        r2 = await q.enqueue(
            SemanticMsg(uri=uri, context_type="session", account_id="acc", user_id="u1")
        )
        assert r1 == "queued-id"
        assert r2 == "deduplicated"
        assert named_enqueue.call_count == 1


@pytest.mark.asyncio
async def test_resource_and_memory_same_uri_have_separate_windows():
    """Different context types for the same URI must not collapse into a single window."""
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        uri = "viking://resources/shared"
        await q.enqueue(SemanticMsg(uri=uri, context_type="resource"))
        await q.enqueue(SemanticMsg(uri=uri, context_type="memory"))
        assert named_enqueue.call_count == 2


@pytest.mark.asyncio
async def test_skill_context_not_deduped():
    """Skill context should pass through; only memory/resource/session are deduped."""
    mock_agfs = MagicMock()
    with patch.object(NamedQueue, "enqueue", new_callable=AsyncMock) as named_enqueue:
        named_enqueue.return_value = "queued-id"
        q = SemanticQueue(mock_agfs, "/queue", "semantic")
        uri = "viking://agent/cli/skills/example"
        await q.enqueue(SemanticMsg(uri=uri, context_type="skill"))
        await q.enqueue(SemanticMsg(uri=uri, context_type="skill"))
        assert named_enqueue.call_count == 2
