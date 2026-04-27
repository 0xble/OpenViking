# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_dag import SemanticDagExecutor
from openviking_cli.session.user_id import UserIdentifier


def _mock_transaction_layer(monkeypatch):
    """Patch lock layer to no-op for DAG tests."""
    mock_handle = MagicMock()
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aenter__",
        AsyncMock(return_value=mock_handle),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.lock_context.LockContext.__aexit__",
        AsyncMock(return_value=False),
    )
    monkeypatch.setattr(
        "openviking.storage.transaction.get_lock_manager",
        lambda: MagicMock(),
    )


class _FakeVikingFS:
    def __init__(self, tree):
        self._tree = tree
        self.writes = []

    async def ls(self, uri, ctx=None):
        return self._tree.get(uri, [])

    async def write_file(self, path, content, ctx=None):
        self.writes.append((path, content))

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("viking://", "/local/acc1/")


class _FakeProcessor:
    def __init__(self):
        self.summarized_files = []
        self.vectorized_files = []
        self.overview_inputs = []

    async def _generate_single_file_summary(self, file_path, llm_sem=None, ctx=None):
        self.summarized_files.append(file_path)
        return {"name": file_path.split("/")[-1], "summary": "summary"}

    async def _generate_overview(self, dir_uri, file_summaries, children_abstracts):
        self.overview_inputs.append((dir_uri, file_summaries, children_abstracts))
        return "overview"

    def _extract_abstract_from_overview(self, overview):
        return "abstract"

    def _enforce_size_limits(self, overview, abstract):
        return overview, abstract

    async def _vectorize_directory(
        self, uri, context_type, abstract, overview, ctx=None, semantic_msg_id=None
    ):
        pass

    async def _vectorize_directory_simple(self, uri, context_type, abstract, overview, ctx=None):
        await self._vectorize_directory(uri, context_type, abstract, overview, ctx=ctx)

    async def _vectorize_single_file(
        self,
        parent_uri,
        context_type,
        file_path,
        summary_dict,
        ctx=None,
        semantic_msg_id=None,
        use_summary=False,
    ):
        self.vectorized_files.append(file_path)


class _DummyTracker:
    async def register(self, **_kwargs):
        return None


def _queued_file_names(executor):
    return [
        task.file_path.split("/")[-1]
        for task in executor._pending_vectorize_tasks
        if task.task_type == "file" and task.file_path
    ]


def _overview_file_names(processor):
    return [
        item["name"]
        for _dir_uri, file_summaries, _children_abstracts in processor.overview_inputs
        for item in file_summaries
    ]


def _summary_cache_names(fake_fs):
    names = set()
    for path, content in fake_fs.writes:
        if path.endswith("/.summary_cache.json"):
            names.update(json.loads(content).keys())
    return names


@pytest.mark.asyncio
async def test_messages_jsonl_excluded_from_summary(monkeypatch):
    """messages.jsonl should skip VLM summary while still being vectorized."""
    _mock_transaction_layer(monkeypatch)
    root_uri = "viking://session/test-session"
    tree = {
        root_uri: [
            {"name": "messages.jsonl", "isDir": False},
            {"name": "notes.txt", "isDir": False},
            {"name": "document.pdf", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: _DummyTracker(),
    )

    processor = _FakeProcessor()
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="session",
        max_concurrent_llm=2,
        ctx=ctx,
    )
    await executor.run(root_uri)

    summarized_names = [p.split("/")[-1] for p in processor.summarized_files]
    vectorized_names = _queued_file_names(executor)
    assert "messages.jsonl" not in summarized_names
    assert "messages.jsonl" in vectorized_names
    assert "messages.jsonl" not in _overview_file_names(processor)
    assert "messages.jsonl" not in _summary_cache_names(fake_fs)
    assert "notes.txt" in summarized_names
    assert "document.pdf" in summarized_names


@pytest.mark.asyncio
async def test_messages_jsonl_excluded_in_subdirectory(monkeypatch):
    """messages.jsonl in a subdirectory should skip VLM summary only."""
    _mock_transaction_layer(monkeypatch)
    root_uri = "viking://session/test-session"
    tree = {
        root_uri: [
            {"name": "subdir", "isDir": True},
        ],
        f"{root_uri}/subdir": [
            {"name": "messages.jsonl", "isDir": False},
            {"name": "data.csv", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: _DummyTracker(),
    )

    processor = _FakeProcessor()
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="session",
        max_concurrent_llm=2,
        ctx=ctx,
    )
    await executor.run(root_uri)

    summarized_names = [p.split("/")[-1] for p in processor.summarized_files]
    vectorized_names = _queued_file_names(executor)
    assert "messages.jsonl" not in summarized_names
    assert "messages.jsonl" in vectorized_names
    assert "messages.jsonl" not in _overview_file_names(processor)
    assert "messages.jsonl" not in _summary_cache_names(fake_fs)
    assert "data.csv" in summarized_names


@pytest.mark.asyncio
async def test_imported_session_jsonl_chunks_skip_summary_but_vectorize(monkeypatch):
    """Generated JSONL-derived imported session chunks should not call VLM."""
    _mock_transaction_layer(monkeypatch)
    root_uri = "viking://resources/imported/sessions/019dc695"
    tree = {
        root_uri: [
            {"name": "rollout-2026-04-26.jsonl_001.md", "isDir": False},
            {"name": "normal-notes.md", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: _DummyTracker(),
    )

    processor = _FakeProcessor()
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    executor = SemanticDagExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=2,
        ctx=ctx,
    )
    await executor.run(root_uri)

    summarized_names = [p.split("/")[-1] for p in processor.summarized_files]
    vectorized_names = _queued_file_names(executor)
    assert "rollout-2026-04-26.jsonl_001.md" not in summarized_names
    assert "rollout-2026-04-26.jsonl_001.md" in vectorized_names
    assert "rollout-2026-04-26.jsonl_001.md" not in _overview_file_names(processor)
    assert "rollout-2026-04-26.jsonl_001.md" not in _summary_cache_names(fake_fs)
    assert "normal-notes.md" in summarized_names


if __name__ == "__main__":
    pytest.main([__file__])
