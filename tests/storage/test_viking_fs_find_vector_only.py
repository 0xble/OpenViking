# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from openviking.storage.viking_fs import VikingFS


class _FakeAGFS:
    def read(self, path: str) -> bytes:
        return b""

    def write(self, path: str, data: bytes) -> str:
        return path

    def rm(self, path: str) -> None:
        return None


@pytest.mark.asyncio
async def test_find_allows_vector_only_retrieval(monkeypatch):
    viking_fs = VikingFS(
        agfs=_FakeAGFS(),
        query_embedder=object(),
        rerank_config=None,
        vector_store=object(),
    )

    captured = {}

    class FakeRetriever:
        def __init__(self, storage, embedder, rerank_config):
            captured["storage"] = storage
            captured["embedder"] = embedder
            captured["rerank_config"] = rerank_config

        async def retrieve(self, typed_query, ctx, limit, **kwargs):
            captured["typed_query"] = typed_query
            captured["ctx"] = ctx
            captured["limit"] = limit
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                matched_contexts=[],
                searched_directories=typed_query.target_directories or [],
            )

    monkeypatch.setattr(
        "openviking.retrieve.hierarchical_retriever.HierarchicalRetriever",
        FakeRetriever,
    )

    result = await viking_fs.find("alcove license key")

    assert result.resources == []
    assert captured["rerank_config"] is None
