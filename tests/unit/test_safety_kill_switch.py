# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OPENVIKING_DISABLE_VLM kill-switch."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.utils.safety import (
    VLMDisabledError,
    check_vlm_enabled,
    vlm_disabled,
)


def test_vlm_disabled_unset(monkeypatch):
    monkeypatch.delenv("OPENVIKING_DISABLE_VLM", raising=False)
    assert vlm_disabled() is False


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
def test_vlm_disabled_truthy_values(monkeypatch, value):
    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", value)
    assert vlm_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything-else"])
def test_vlm_disabled_falsy_values(monkeypatch, value):
    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", value)
    assert vlm_disabled() is False


def test_check_vlm_enabled_raises_when_disabled(monkeypatch):
    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", "1")
    with pytest.raises(VLMDisabledError, match="Summarizer.summarize"):
        check_vlm_enabled("Summarizer.summarize")


def test_check_vlm_enabled_passthrough_when_enabled(monkeypatch):
    monkeypatch.delenv("OPENVIKING_DISABLE_VLM", raising=False)
    check_vlm_enabled("Summarizer.summarize")


@pytest.mark.asyncio
async def test_summarizer_blocked_by_kill_switch(monkeypatch):
    """Summarizer.summarize must raise VLMDisabledError when kill-switch is set,
    before any queue enqueue."""
    from openviking.utils.summarizer import Summarizer

    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", "1")
    summarizer = Summarizer(vlm_processor=MagicMock())
    ctx = MagicMock()
    ctx.account_id = "test"

    with patch("openviking.utils.summarizer.get_queue_manager") as qm:
        with pytest.raises(VLMDisabledError):
            await summarizer.summarize(["viking://resources/x"], ctx)
        qm.assert_not_called()


@pytest.mark.asyncio
async def test_embedder_async_blocked_by_kill_switch(monkeypatch):
    """EmbedderBase._run_with_async_retry must raise VLMDisabledError before any
    semaphore acquire or actual API call."""
    from openviking.models.embedder.base import EmbedderBase

    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", "1")

    class _StubEmbedder(EmbedderBase):
        def embed(self, text, is_query=False):
            raise NotImplementedError

    embedder = _StubEmbedder("test-model", {})
    inner = AsyncMock(return_value="result")

    with pytest.raises(VLMDisabledError):
        await embedder._run_with_async_retry(inner, operation_name="embed_async")

    inner.assert_not_called()


def test_embedder_sync_blocked_by_kill_switch(monkeypatch):
    """EmbedderBase._run_with_retry must also gate on the kill-switch."""
    from openviking.models.embedder.base import EmbedderBase

    monkeypatch.setenv("OPENVIKING_DISABLE_VLM", "1")

    class _StubEmbedder(EmbedderBase):
        def embed(self, text, is_query=False):
            raise NotImplementedError

    embedder = _StubEmbedder("test-model", {})
    inner = MagicMock(return_value="result")

    with pytest.raises(VLMDisabledError):
        embedder._run_with_retry(inner, operation_name="embed")

    inner.assert_not_called()
