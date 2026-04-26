# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.storage.queuefs.semantic_msg import SemanticMsg
from openviking.storage.queuefs.semantic_processor import SemanticProcessor


def _make_msg(uri="viking://user/memories", context_type="memory", **kwargs):
    defaults = {
        "id": "test-msg-1",
        "uri": uri,
        "context_type": context_type,
        "recursive": False,
        "role": "root",
        "account_id": "acc1",
        "user_id": "usr1",
        "agent_id": "test-agent",
        "telemetry_id": "",
        "target_uri": "",
        "lifecycle_lock_handle_id": "",
        "changes": None,
        "is_code_repo": False,
    }
    defaults.update(kwargs)
    return SemanticMsg.from_dict(defaults)


@pytest.mark.asyncio
async def test_memory_empty_dir_still_reports_success():
    processor = SemanticProcessor()
    fake_fs = MagicMock()
    fake_fs.ls = AsyncMock(return_value=[])

    success_called = False
    error_called = False

    def on_success():
        nonlocal success_called
        success_called = True

    def on_error(error_msg, error_data=None):
        del error_msg, error_data
        nonlocal error_called
        error_called = True

    processor.set_callbacks(on_success, lambda: None, on_error)

    with (
        patch("openviking.storage.queuefs.semantic_processor.get_viking_fs", return_value=fake_fs),
        patch("openviking.storage.queuefs.semantic_processor.resolve_telemetry", return_value=None),
    ):
        await processor.on_dequeue(_make_msg().to_dict())

    assert success_called
    assert not error_called


@pytest.mark.asyncio
async def test_memory_ls_filesystem_error_reports_error():
    processor = SemanticProcessor()
    fake_fs = MagicMock()
    fake_fs.ls = AsyncMock(side_effect=FileNotFoundError("/memories not found"))

    success_called = False
    error_info = {}

    def on_success():
        nonlocal success_called
        success_called = True

    def on_error(error_msg, error_data=None):
        del error_data
        error_info["msg"] = error_msg

    processor.set_callbacks(on_success, lambda: None, on_error)

    with (
        patch("openviking.storage.queuefs.semantic_processor.get_viking_fs", return_value=fake_fs),
        patch("openviking.storage.queuefs.semantic_processor.resolve_telemetry", return_value=None),
    ):
        await processor.on_dequeue(_make_msg().to_dict())

    assert not success_called
    assert "/memories not found" in error_info["msg"]


@pytest.mark.asyncio
async def test_memory_ls_transient_error_requeues():
    processor = SemanticProcessor()
    fake_fs = MagicMock()
    fake_fs.ls = AsyncMock(side_effect=RuntimeError("500 Internal Server Error"))

    success_called = False
    requeue_called = False
    error_called = False

    def on_success():
        nonlocal success_called
        success_called = True

    def on_requeue():
        nonlocal requeue_called
        requeue_called = True

    def on_error(error_msg, error_data=None):
        del error_msg, error_data
        nonlocal error_called
        error_called = True

    processor.set_callbacks(on_success, on_requeue, on_error)
    reenqueue_mock = AsyncMock()

    with (
        patch("openviking.storage.queuefs.semantic_processor.get_viking_fs", return_value=fake_fs),
        patch("openviking.storage.queuefs.semantic_processor.resolve_telemetry", return_value=None),
        patch.object(processor, "_reenqueue_semantic_msg", new=reenqueue_mock),
    ):
        await processor.on_dequeue(_make_msg(telemetry_id="tel-1").to_dict())

    assert requeue_called
    assert success_called
    assert not error_called
    reenqueue_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_write_filesystem_error_reports_error():
    processor = SemanticProcessor()
    fake_fs = MagicMock()
    fake_fs.ls = AsyncMock(return_value=[{"name": "file1.md", "isDir": False}])
    fake_fs.read_file = AsyncMock(return_value="some content")
    fake_fs.write_file = AsyncMock(side_effect=PermissionError("Permission denied"))

    success_called = False
    error_info = {}

    def on_success():
        nonlocal success_called
        success_called = True

    def on_error(error_msg, error_data=None):
        del error_data
        error_info["msg"] = error_msg

    processor.set_callbacks(on_success, lambda: None, on_error)

    with (
        patch("openviking.storage.queuefs.semantic_processor.get_viking_fs", return_value=fake_fs),
        patch("openviking.storage.queuefs.semantic_processor.resolve_telemetry", return_value=None),
        patch.object(
            processor,
            "_generate_single_file_summary",
            new=AsyncMock(return_value={"name": "file1.md", "summary": "test summary"}),
        ),
        patch.object(
            processor, "_generate_overview", new=AsyncMock(return_value="# Overview\nbody")
        ),
    ):
        await processor.on_dequeue(_make_msg().to_dict())

    assert not success_called
    assert "Permission denied" in error_info["msg"]
