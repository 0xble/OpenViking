# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json

import pytest

from openviking.storage.queuefs.named_queue import DequeueHandlerBase, NamedQueue


class _FakeQueueAgfs:
    def __init__(self, messages):
        self.messages = list(messages)
        self.acked = []

    def mkdir(self, path):
        del path

    def read(self, path):
        if path.endswith("/size"):
            return str(len(self.messages)).encode("utf-8")
        if path.endswith("/dequeue"):
            if not self.messages:
                return b"{}"
            return json.dumps(self.messages.pop(0)).encode("utf-8")
        return b"{}"

    def write(self, path, data):
        if path.endswith("/ack"):
            self.acked.append(data.decode("utf-8"))
        return "msg-id"


class _ReturningHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        return data


class _ReportingHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        self.report_success()
        return data


class _FailingHandler(DequeueHandlerBase):
    async def on_dequeue(self, data):
        raise RuntimeError("handler failed")


@pytest.mark.asyncio
async def test_dequeue_return_without_callback_closes_in_progress_status():
    queue = NamedQueue(
        _FakeQueueAgfs([{"id": "msg-1", "value": "ok"}]),
        "/queue",
        "test",
        dequeue_handler=_ReturningHandler(),
    )

    assert await queue.dequeue() == {"id": "msg-1", "value": "ok"}

    status = await queue.get_status()
    assert status.pending == 0
    assert status.in_progress == 0
    assert status.processed == 1
    assert status.error_count == 0


@pytest.mark.asyncio
async def test_dequeue_report_success_does_not_double_count_processed():
    queue = NamedQueue(
        _FakeQueueAgfs([{"id": "msg-1", "value": "ok"}]),
        "/queue",
        "test",
        dequeue_handler=_ReportingHandler(),
    )

    assert await queue.dequeue() == {"id": "msg-1", "value": "ok"}

    status = await queue.get_status()
    assert status.pending == 0
    assert status.in_progress == 0
    assert status.processed == 1
    assert status.error_count == 0


@pytest.mark.asyncio
async def test_dequeue_exception_closes_in_progress_without_ack():
    agfs = _FakeQueueAgfs([{"id": "msg-1", "value": "fail"}])
    queue = NamedQueue(
        agfs,
        "/queue",
        "test",
        dequeue_handler=_FailingHandler(),
    )

    assert await queue.dequeue() is None

    status = await queue.get_status()
    assert status.pending == 0
    assert status.in_progress == 0
    assert status.processed == 0
    assert status.error_count == 1
    assert agfs.acked == []


@pytest.mark.asyncio
async def test_process_dequeued_return_without_callback_closes_in_progress_status():
    queue = NamedQueue(
        _FakeQueueAgfs([]),
        "/queue",
        "test",
        dequeue_handler=_ReturningHandler(),
    )
    token = queue._on_dequeue_start()

    assert await queue.process_dequeued({"id": "msg-1"}, token) == {"id": "msg-1"}
    queue._finish_process_if_unreported(token)

    status = await queue.get_status()
    assert status.in_progress == 0
    assert status.processed == 1
