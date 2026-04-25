# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OPENVIKING_DISABLE_MODEL_CALLS kill-switch."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.utils.safety import (
    ModelCallDisabledError,
    check_model_calls_enabled,
    check_vlm_enabled,
    model_calls_disabled,
    vlm_disabled,
)


def _clear_kill_switch_env(monkeypatch):
    monkeypatch.delenv("OPENVIKING_DISABLE_MODEL_CALLS", raising=False)
    monkeypatch.delenv("OPENVIKING_DISABLE_VLM", raising=False)


def test_model_calls_disabled_unset(monkeypatch):
    _clear_kill_switch_env(monkeypatch)
    assert model_calls_disabled() is False


@pytest.mark.parametrize(
    "env_var",
    ["OPENVIKING_DISABLE_MODEL_CALLS", "OPENVIKING_DISABLE_VLM"],
)
@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on"])
def test_model_calls_disabled_truthy_values(monkeypatch, env_var, value):
    _clear_kill_switch_env(monkeypatch)
    monkeypatch.setenv(env_var, value)
    assert model_calls_disabled() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "anything-else"])
def test_model_calls_disabled_falsy_values(monkeypatch, value):
    _clear_kill_switch_env(monkeypatch)
    monkeypatch.setenv("OPENVIKING_DISABLE_MODEL_CALLS", value)
    assert model_calls_disabled() is False


def test_legacy_aliases_resolve_to_canonical():
    """Pre-rename names must continue to resolve so external callers don't break."""
    assert vlm_disabled is model_calls_disabled
    assert check_vlm_enabled is check_model_calls_enabled


def test_check_model_calls_enabled_raises_when_disabled(monkeypatch):
    monkeypatch.setenv("OPENVIKING_DISABLE_MODEL_CALLS", "1")
    with pytest.raises(ModelCallDisabledError, match="Summarizer.summarize"):
        check_model_calls_enabled("Summarizer.summarize")


def test_check_model_calls_enabled_passthrough_when_enabled(monkeypatch):
    _clear_kill_switch_env(monkeypatch)
    check_model_calls_enabled("Summarizer.summarize")


@pytest.mark.asyncio
async def test_summarizer_blocked_by_kill_switch(monkeypatch):
    """Summarizer.summarize must raise ModelCallDisabledError when kill-switch
    is set, before any queue enqueue."""
    from openviking.utils.summarizer import Summarizer

    monkeypatch.setenv("OPENVIKING_DISABLE_MODEL_CALLS", "1")
    summarizer = Summarizer(vlm_processor=MagicMock())
    ctx = MagicMock()
    ctx.account_id = "test"

    with patch("openviking.utils.summarizer.get_queue_manager") as qm:
        with pytest.raises(ModelCallDisabledError):
            await summarizer.summarize(["viking://resources/x"], ctx)
        qm.assert_not_called()


@pytest.mark.asyncio
async def test_embedder_async_blocked_by_kill_switch(monkeypatch):
    """EmbedderBase._run_with_async_retry must raise ModelCallDisabledError
    before any semaphore acquire or actual API call."""
    from openviking.models.embedder.base import EmbedderBase

    monkeypatch.setenv("OPENVIKING_DISABLE_MODEL_CALLS", "1")

    class _StubEmbedder(EmbedderBase):
        def embed(self, text, is_query=False):
            raise NotImplementedError

    embedder = _StubEmbedder("test-model", {})
    inner = AsyncMock(return_value="result")

    with pytest.raises(ModelCallDisabledError):
        await embedder._run_with_async_retry(inner, operation_name="embed_async")

    inner.assert_not_called()


def test_embedder_sync_blocked_by_kill_switch(monkeypatch):
    """EmbedderBase._run_with_retry must also gate on the kill-switch."""
    from openviking.models.embedder.base import EmbedderBase

    monkeypatch.setenv("OPENVIKING_DISABLE_MODEL_CALLS", "1")

    class _StubEmbedder(EmbedderBase):
        def embed(self, text, is_query=False):
            raise NotImplementedError

    embedder = _StubEmbedder("test-model", {})
    inner = MagicMock(return_value="result")

    with pytest.raises(ModelCallDisabledError):
        embedder._run_with_retry(inner, operation_name="embed")

    inner.assert_not_called()
