# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for VolcEngineVLM retry behavior."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openviking.models.vlm.backends.volcengine_vlm import VolcEngineVLM


@pytest.fixture
def vlm():
    return VolcEngineVLM({"api_key": "test", "max_retries": 3})


@pytest.mark.asyncio
async def test_get_completion_async_does_not_retry_permanent_error(vlm):
    """401 Unauthorized must not be retried (was previously retried 4 times)."""
    failing = AsyncMock(side_effect=RuntimeError("401 Unauthorized"))
    fake_client = MagicMock()
    fake_client.chat.completions.create = failing

    with patch.object(vlm, "get_async_client", return_value=fake_client):
        with pytest.raises(RuntimeError, match="401"):
            await vlm.get_completion_async(prompt="hello")

    assert failing.await_count == 1


@pytest.mark.asyncio
async def test_get_completion_async_does_not_retry_400(vlm):
    failing = AsyncMock(side_effect=RuntimeError("400 BadRequest"))
    fake_client = MagicMock()
    fake_client.chat.completions.create = failing

    with patch.object(vlm, "get_async_client", return_value=fake_client):
        with pytest.raises(RuntimeError):
            await vlm.get_completion_async(prompt="hello")

    assert failing.await_count == 1


@pytest.mark.asyncio
async def test_get_completion_async_does_not_retry_rate_limit(vlm):
    """429 must not be retried inline (cf. D3 classification split)."""
    failing = AsyncMock(side_effect=RuntimeError("429 too many requests"))
    fake_client = MagicMock()
    fake_client.chat.completions.create = failing

    with patch.object(vlm, "get_async_client", return_value=fake_client):
        with pytest.raises(RuntimeError):
            await vlm.get_completion_async(prompt="hello")

    assert failing.await_count == 1


@pytest.mark.asyncio
async def test_get_completion_async_retries_transient_error(vlm):
    """5xx errors should retry up to max_retries before giving up."""
    failing = AsyncMock(side_effect=RuntimeError("503 service unavailable"))
    fake_client = MagicMock()
    fake_client.chat.completions.create = failing

    with patch.object(vlm, "get_async_client", return_value=fake_client):
        with patch("openviking.utils.model_retry.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(RuntimeError):
                await vlm.get_completion_async(prompt="hello")

    # max_retries=3 means initial attempt + 3 retries = 4 calls.
    assert failing.await_count == 4
