# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from unittest.mock import AsyncMock, Mock

import pytest

import openviking.service.session_service as session_service_module
from openviking.message import Message, TextPart
from openviking.metrics.datasources.session import SessionLifecycleDataSource
from openviking.server.identity import RequestContext, Role
from openviking.service.session_service import SessionService
from openviking_cli.session.user_id import UserIdentifier


def _make_ctx() -> RequestContext:
    return RequestContext(
        user=UserIdentifier("acme", "alice", "agent1"),
        role=Role.ADMIN,
    )


@pytest.mark.asyncio
async def test_create_keeps_working_when_lifecycle_metrics_fail(monkeypatch: pytest.MonkeyPatch):
    service = SessionService()
    ctx = _make_ctx()
    session = Mock()
    session.exists = AsyncMock(return_value=False)
    session.ensure_exists = AsyncMock()
    debug = Mock()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(service, "session", Mock(return_value=session))
    monkeypatch.setattr(SessionLifecycleDataSource, "record_lifecycle", staticmethod(_boom))
    monkeypatch.setattr(session_service_module.logger, "debug", debug)

    result = await service.create(ctx, "sess-1")

    assert result is session
    session.exists.assert_awaited_once()
    session.ensure_exists.assert_awaited_once()
    assert debug.call_count == 2


@pytest.mark.asyncio
async def test_commit_async_keeps_working_when_session_metrics_fail(
    monkeypatch: pytest.MonkeyPatch,
):
    service = SessionService(viking_fs=Mock())
    ctx = _make_ctx()
    session = Mock()
    session.commit_async = AsyncMock(return_value={"status": "queued", "archived": False})
    debug = Mock()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("metrics failed")

    monkeypatch.setattr(service, "get", AsyncMock(return_value=session))
    monkeypatch.setattr(SessionLifecycleDataSource, "record_lifecycle", staticmethod(_boom))
    monkeypatch.setattr(SessionLifecycleDataSource, "record_archive", staticmethod(_boom))
    monkeypatch.setattr(session_service_module.logger, "debug", debug)

    result = await service.commit_async("sess-1", ctx)

    assert result == {"status": "queued", "archived": False}
    session.commit_async.assert_awaited_once()
    assert debug.call_count == 2


@pytest.mark.asyncio
async def test_sessions_returns_empty_and_logs_when_storage_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
):
    service = SessionService(viking_fs=Mock())
    ctx = _make_ctx()
    debug = Mock()

    service._viking_fs.ls = AsyncMock(side_effect=RuntimeError("ls failed"))
    monkeypatch.setattr(session_service_module.logger, "debug", debug)

    result = await service.sessions(ctx)

    assert result == []
    debug.assert_called_once()


@pytest.mark.asyncio
async def test_extract_skips_heartbeat_only_session(monkeypatch: pytest.MonkeyPatch):
    compressor = Mock()
    compressor.extract_long_term_memories = AsyncMock()
    service = SessionService(viking_fs=Mock(), session_compressor=compressor)
    ctx = _make_ctx()
    session = Mock()
    session.load_latest_archive_messages = AsyncMock(
        return_value=[
            Message(
                id="m1",
                role="user",
                parts=[TextPart(text="[OPENCLAW_HEARTBEAT] HEARTBEAT_OK")],
            )
        ]
    )

    monkeypatch.setattr(service, "get", AsyncMock(return_value=session))

    result = await service.extract("sess-heartbeat", ctx)

    assert result == []
    compressor.extract_long_term_memories.assert_not_awaited()


def test_synthetic_session_detection_extracts_structured_text():
    message = Mock()
    message.role = "user"
    message.content = [{"type": "text", "text": "Remember that I prefer concise reviews."}]

    assert session_service_module._is_synthetic_extract_session([message]) is False


def test_synthetic_session_detection_keeps_short_real_prompts():
    message = Mock()
    message.role = "user"
    message.content = [{"type": "text", "text": "Help?"}]

    assert session_service_module._is_synthetic_extract_session([message]) is False
