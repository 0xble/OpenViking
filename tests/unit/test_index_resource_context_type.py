# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Regression tests for URI-derived context_type during manual reindex/index."""

from unittest.mock import AsyncMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.utils import embedding_utils
from openviking_cli.session.user_id import UserIdentifier


def _ctx() -> RequestContext:
    return RequestContext(
        user=UserIdentifier(account_id="acct_test", user_id="alice", agent_id="default"),
        role=Role.ADMIN,
    )


@pytest.mark.asyncio
async def test_index_resource_uses_memory_context_type(monkeypatch):
    """Memory URIs must stay in the memory bucket during manual reindex."""
    uri = "viking://user/alice/memories/preferences/coding-style"
    file_uri = f"{uri}/preference.md"

    fake_viking_fs = AsyncMock()
    fake_viking_fs.exists.side_effect = [True, True]
    fake_viking_fs.read_file.side_effect = [b"abstract text", b"overview text"]
    fake_viking_fs.ls.return_value = [
        {"name": "preference.md", "isDir": False, "uri": file_uri},
    ]

    captured: dict[str, object] = {}

    async def _fake_vectorize_directory_meta(
        target_uri,
        abstract,
        overview,
        *,
        context_type="resource",
        ctx=None,
        semantic_msg_id=None,
    ):
        del abstract, overview, ctx, semantic_msg_id
        captured["dir_uri"] = target_uri
        captured["dir_context_type"] = context_type

    async def _fake_vectorize_file(
        *,
        file_path,
        summary_dict,
        parent_uri,
        context_type="resource",
        ctx=None,
        semantic_msg_id=None,
        use_summary=False,
    ):
        del summary_dict, ctx, semantic_msg_id, use_summary
        captured["file_uri"] = file_path
        captured["file_parent_uri"] = parent_uri
        captured["file_context_type"] = context_type

    monkeypatch.setattr(embedding_utils, "get_viking_fs", lambda: fake_viking_fs)
    monkeypatch.setattr(embedding_utils, "vectorize_directory_meta", _fake_vectorize_directory_meta)
    monkeypatch.setattr(embedding_utils, "vectorize_file", _fake_vectorize_file)

    await embedding_utils.index_resource(uri, _ctx())

    assert captured["dir_uri"] == uri
    assert captured["dir_context_type"] == "memory"
    assert captured["file_uri"] == file_uri
    assert captured["file_parent_uri"] == uri
    assert captured["file_context_type"] == "memory"
