# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared model retry helpers."""

import pytest

from openviking.utils.model_retry import (
    classify_api_error,
    is_rate_limit_error,
    is_retryable_api_error,
    retry_async,
    retry_sync,
)


def test_classify_api_error_recognizes_request_burst_too_fast():
    # RequestBurstTooFast is a rate-limit signal, not a generic transient signal.
    assert classify_api_error(RuntimeError("RequestBurstTooFast")) == "rate_limit"


def test_classify_api_error_recognizes_rate_limit_429():
    assert classify_api_error(RuntimeError("HTTP 429 too many requests")) == "rate_limit"


def test_classify_api_error_keeps_5xx_as_transient():
    assert classify_api_error(RuntimeError("502 bad gateway")) == "transient"


def test_is_retryable_api_error_excludes_rate_limit():
    assert is_retryable_api_error(RuntimeError("429")) is False
    assert is_retryable_api_error(RuntimeError("503")) is True


def test_is_rate_limit_error_helper():
    assert is_rate_limit_error(RuntimeError("RateLimit exceeded")) is True
    assert is_rate_limit_error(RuntimeError("503 service unavailable")) is False


def test_retry_sync_retries_transient_error_until_success():
    attempts = {"count": 0}

    def _call():
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("503 service unavailable")
        return "ok"

    assert retry_sync(_call, max_retries=3, base_delay=0.0) == "ok"
    assert attempts["count"] == 3


def test_retry_sync_does_not_retry_rate_limit_error():
    attempts = {"count": 0}

    def _call():
        attempts["count"] += 1
        raise RuntimeError("429 too many requests")

    with pytest.raises(RuntimeError):
        retry_sync(_call, max_retries=3)

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_rate_limit_error():
    attempts = {"count": 0}

    async def _call():
        attempts["count"] += 1
        raise RuntimeError("RateLimit exceeded")

    with pytest.raises(RuntimeError):
        await retry_async(_call, max_retries=3)

    assert attempts["count"] == 1


@pytest.mark.asyncio
async def test_retry_async_does_not_retry_unknown_error():
    attempts = {"count": 0}

    async def _call():
        attempts["count"] += 1
        raise RuntimeError("some unexpected validation failure")

    with pytest.raises(RuntimeError):
        await retry_async(_call, max_retries=3)

    assert attempts["count"] == 1
