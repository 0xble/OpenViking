# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

"""Tests for request-scoped wait behavior on write APIs."""

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.telemetry.context import bind_telemetry
from openviking.telemetry.operation import OperationTelemetry


class _FakeRequestWaitTracker:
    def __init__(self, queue_status):
        self.queue_status = queue_status
        self.registered_requests = []
        self.wait_calls = []
        self.build_calls = []
        self.cleaned = []

    def register_request(self, telemetry_id: str) -> None:
        self.registered_requests.append(telemetry_id)

    async def wait_for_request(self, telemetry_id: str, timeout):
        self.wait_calls.append((telemetry_id, timeout))

    def build_queue_status(self, telemetry_id: str):
        self.build_calls.append(telemetry_id)
        return self.queue_status

    def cleanup(self, telemetry_id: str) -> None:
        self.cleaned.append(telemetry_id)


class _ExplodingQueueManager:
    async def wait_complete(self, *args, **kwargs):
        raise AssertionError("global queue wait should not be used")


@pytest.mark.asyncio
async def test_add_resource_wait_uses_request_tracker(service, monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 1, "error_count": 0, "errors": []},
            "Embedding": {"processed": 2, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=service.user, role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_resource", enabled=True)

    async def _fake_process_resource(**kwargs):
        del kwargs
        return {"status": "success", "root_uri": "viking://resources/demo"}

    monkeypatch.setattr(
        service.resources._resource_processor, "process_resource", _fake_process_resource
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await service.resources.add_resource(
            path="/tmp/demo.md",
            ctx=ctx,
            reason="request wait test",
            wait=True,
            timeout=12.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 12.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]


@pytest.mark.asyncio
async def test_add_resource_wait_uses_request_tracker_when_telemetry_disabled(service, monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 1, "error_count": 0, "errors": []},
            "Embedding": {"processed": 2, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=service.user, role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_resource", enabled=False)

    async def _fake_process_resource(**kwargs):
        del kwargs
        return {"status": "success", "root_uri": "viking://resources/demo"}

    monkeypatch.setattr(
        service.resources._resource_processor, "process_resource", _fake_process_resource
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await service.resources.add_resource(
            path="/tmp/demo.md",
            ctx=ctx,
            reason="request wait test",
            wait=True,
            timeout=12.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 12.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]


@pytest.mark.asyncio
async def test_add_skill_wait_uses_request_tracker(service, monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 0, "error_count": 0, "errors": []},
            "Embedding": {"processed": 1, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=service.user, role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_skill", enabled=True)

    async def _fake_process_skill(**kwargs):
        del kwargs
        return {"status": "success", "uri": "viking://agent/skills/demo", "name": "demo"}

    monkeypatch.setattr(service.resources._skill_processor, "process_skill", _fake_process_skill)
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await service.resources.add_skill(
            data={"name": "demo", "content": "# Demo"},
            ctx=ctx,
            wait=True,
            timeout=9.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 9.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]


@pytest.mark.asyncio
async def test_add_skill_wait_uses_request_tracker_when_telemetry_disabled(service, monkeypatch):
    tracker = _FakeRequestWaitTracker(
        {
            "Semantic": {"processed": 0, "error_count": 0, "errors": []},
            "Embedding": {"processed": 1, "error_count": 0, "errors": []},
        }
    )
    ctx = RequestContext(user=service.user, role=Role.ROOT)
    telemetry = OperationTelemetry(operation="resources.add_skill", enabled=False)

    async def _fake_process_skill(**kwargs):
        del kwargs
        return {"status": "success", "uri": "viking://agent/skills/demo", "name": "demo"}

    monkeypatch.setattr(service.resources._skill_processor, "process_skill", _fake_process_skill)
    monkeypatch.setattr(
        "openviking.service.resource_service.get_queue_manager",
        lambda: _ExplodingQueueManager(),
    )
    monkeypatch.setattr(
        "openviking.service.resource_service.get_request_wait_tracker",
        lambda: tracker,
        raising=False,
    )

    with bind_telemetry(telemetry):
        result = await service.resources.add_skill(
            data={"name": "demo", "content": "# Demo"},
            ctx=ctx,
            wait=True,
            timeout=9.0,
        )

    assert result["queue_status"] == tracker.queue_status
    assert tracker.registered_requests == [telemetry.telemetry_id]
    assert tracker.wait_calls == [(telemetry.telemetry_id, 9.0)]
    assert tracker.build_calls == [telemetry.telemetry_id]
    assert tracker.cleaned == [telemetry.telemetry_id]
