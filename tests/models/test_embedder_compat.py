# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for embedder compatibility helpers."""

import pytest

from openviking.models.embedder.base import _callable_accepts_is_query, embed_compat


def test_signature_failure_does_not_assume_is_query_support():
    target = {}

    assert _callable_accepts_is_query(target.update) is False


@pytest.mark.asyncio
async def test_embed_compat_omits_is_query_when_signature_cannot_be_inspected():
    target = {}

    class _BuiltinMethodEmbedder:
        embed = target.update

    result = await embed_compat(_BuiltinMethodEmbedder(), {"key": "value"}, is_query=True)

    assert result is None
    assert target == {"key": "value"}
