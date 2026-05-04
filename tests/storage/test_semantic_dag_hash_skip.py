# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for overview hash-skip cache (#505)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import openviking.storage.queuefs.semantic_dag as semantic_dag
from openviking.server.identity import RequestContext, Role
from openviking.storage.queuefs.semantic_dag import (
    OVERVIEW_HASH_FILENAME,
    SemanticDagExecutor,
)
from openviking_cli.session.user_id import UserIdentifier


def _mock_transaction_layer(monkeypatch):
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
    """Fake viking_fs that records writes and serves prepopulated reads."""

    def __init__(self, tree, files=None):
        self._tree = tree
        self._files: dict[str, str] = files or {}
        self.writes: list[tuple[str, str]] = []

    async def ls(self, uri, ctx=None):
        return self._tree.get(uri, [])

    async def write_file(self, path, content, ctx=None):
        self.writes.append((path, content))
        self._files[path] = content

    async def read_file(self, path, ctx=None):
        if path in self._files:
            return self._files[path]
        raise FileNotFoundError(path)

    def _uri_to_path(self, uri, ctx=None):
        return uri.replace("viking://", "/local/acc1/")


class _FakeProcessor:
    def __init__(self):
        self.overview_calls = 0
        self.file_instructions = []
        self.overview_instructions = []

    async def _generate_single_file_summary(
        self, file_path, llm_sem=None, ctx=None, instruction=""
    ):
        self.file_instructions.append(instruction)
        return {"name": file_path.split("/")[-1], "summary": "summary"}

    async def _generate_overview(self, dir_uri, file_summaries, children_abstracts, instruction=""):
        self.overview_calls += 1
        self.overview_instructions.append(instruction)
        return "fresh-overview"

    def _extract_abstract_from_overview(self, overview):
        return "fresh-abstract"

    def _enforce_size_limits(self, overview, abstract):
        return overview, abstract

    async def _vectorize_directory(self, *args, **kwargs):
        pass

    async def _vectorize_single_file(self, *args, **kwargs):
        pass


class _DummyTracker:
    async def register(self, **_kwargs):
        return None


def _make_executor(monkeypatch, fake_fs, processor, instruction=""):
    monkeypatch.setattr("openviking.storage.queuefs.semantic_dag.get_viking_fs", lambda: fake_fs)
    monkeypatch.setattr(
        "openviking.storage.queuefs.embedding_tracker.EmbeddingTaskTracker.get_instance",
        lambda: _DummyTracker(),
    )
    ctx = RequestContext(user=UserIdentifier("acc1", "user1", "agent1"), role=Role.USER)
    return SemanticDagExecutor(
        processor=processor,
        context_type="resource",
        max_concurrent_llm=2,
        ctx=ctx,
        instruction=instruction,
    )


def _seed_prompt_sha(monkeypatch, value: str):
    """Force the prompt SHA to a known value to keep tests deterministic."""
    monkeypatch.setattr(semantic_dag, "_get_overview_prompt_sha", lambda: value)


def test_compute_overview_input_hash_is_deterministic(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    fake_fs = _FakeVikingFS({})
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())

    files = [{"name": "a.md", "summary": "alpha"}]
    children = [{"name": "child", "abstract": "child-abs"}]

    h1 = executor._compute_overview_input_hash(files, children)
    h2 = executor._compute_overview_input_hash(files, children)
    assert h1 == h2
    assert h1 != ""


def test_compute_overview_input_hash_changes_on_input_change(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    fake_fs = _FakeVikingFS({})
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())

    base = executor._compute_overview_input_hash(
        [{"name": "a.md", "summary": "alpha"}],
        [{"name": "child", "abstract": "child-abs"}],
    )
    mutated_files = executor._compute_overview_input_hash(
        [{"name": "a.md", "summary": "beta"}],
        [{"name": "child", "abstract": "child-abs"}],
    )
    mutated_children = executor._compute_overview_input_hash(
        [{"name": "a.md", "summary": "alpha"}],
        [{"name": "child", "abstract": "child-abs-changed"}],
    )
    assert base != mutated_files
    assert base != mutated_children


def test_compute_overview_input_hash_changes_on_instruction_change(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    fake_fs = _FakeVikingFS({})
    processor = _FakeProcessor()
    base = _make_executor(monkeypatch, fake_fs, processor)._compute_overview_input_hash(
        [{"name": "a.md", "summary": "alpha"}],
        [],
    )
    instructed = _make_executor(
        monkeypatch,
        fake_fs,
        processor,
        instruction="Prefer personal conversation context.",
    )._compute_overview_input_hash(
        [{"name": "a.md", "summary": "alpha"}],
        [],
    )
    assert base != instructed


def test_compute_overview_input_hash_invalidates_on_prompt_change(monkeypatch):
    fake_fs = _FakeVikingFS({})
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())

    files = [{"name": "a.md", "summary": "alpha"}]
    children: list[dict[str, str]] = []

    _seed_prompt_sha(monkeypatch, "prompt-sha-v1")
    h1 = executor._compute_overview_input_hash(files, children)

    _seed_prompt_sha(monkeypatch, "prompt-sha-v2")
    h2 = executor._compute_overview_input_hash(files, children)
    assert h1 != h2


@pytest.mark.asyncio
async def test_read_cached_overview_returns_none_on_missing_hash(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub")
    fake_fs = _FakeVikingFS({})
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())
    result = await executor._read_cached_overview("viking://resources/foo", "any-hash")
    assert result is None


@pytest.mark.asyncio
async def test_read_cached_overview_returns_none_on_hash_mismatch(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub")
    dir_uri = "viking://resources/foo"
    fake_fs = _FakeVikingFS(
        {},
        files={
            f"{dir_uri}/{OVERVIEW_HASH_FILENAME}": "old-hash",
            f"{dir_uri}/.overview.md": "stale-overview",
            f"{dir_uri}/.abstract.md": "stale-abstract",
        },
    )
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())
    result = await executor._read_cached_overview(dir_uri, "new-hash")
    assert result is None


@pytest.mark.asyncio
async def test_read_cached_overview_returns_outputs_on_hash_match(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub")
    dir_uri = "viking://resources/foo"
    fake_fs = _FakeVikingFS(
        {},
        files={
            f"{dir_uri}/{OVERVIEW_HASH_FILENAME}": "good-hash",
            f"{dir_uri}/.overview.md": "cached-overview",
            f"{dir_uri}/.abstract.md": "cached-abstract",
        },
    )
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())
    result = await executor._read_cached_overview(dir_uri, "good-hash")
    assert result == ("cached-overview", "cached-abstract")


@pytest.mark.asyncio
async def test_read_cached_overview_returns_none_when_outputs_empty(monkeypatch):
    _seed_prompt_sha(monkeypatch, "stub")
    dir_uri = "viking://resources/foo"
    fake_fs = _FakeVikingFS(
        {},
        files={
            f"{dir_uri}/{OVERVIEW_HASH_FILENAME}": "good-hash",
            f"{dir_uri}/.overview.md": "",
            f"{dir_uri}/.abstract.md": "cached-abstract",
        },
    )
    executor = _make_executor(monkeypatch, fake_fs, _FakeProcessor())
    result = await executor._read_cached_overview(dir_uri, "good-hash")
    assert result is None


@pytest.mark.asyncio
async def test_overview_task_writes_hash_after_fresh_render(monkeypatch):
    """End-to-end: fresh render writes .overview.md, .abstract.md, and .overview.hash."""
    _mock_transaction_layer(monkeypatch)
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    root_uri = "viking://resources/cache-test"
    tree = {
        root_uri: [
            {"name": "doc.md", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    processor = _FakeProcessor()
    executor = _make_executor(monkeypatch, fake_fs, processor)

    await executor.run(root_uri)

    written_paths = [path for path, _ in fake_fs.writes]
    assert f"{root_uri}/.overview.md" in written_paths
    assert f"{root_uri}/.abstract.md" in written_paths
    assert f"{root_uri}/{OVERVIEW_HASH_FILENAME}" in written_paths
    assert processor.overview_calls == 1


@pytest.mark.asyncio
async def test_instruction_passed_to_file_and_overview_generation(monkeypatch):
    _mock_transaction_layer(monkeypatch)
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    root_uri = "viking://resources/instruction-test"
    tree = {
        root_uri: [
            {"name": "doc.md", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    processor = _FakeProcessor()
    instruction = "Prefer personal conversation context."
    executor = _make_executor(monkeypatch, fake_fs, processor, instruction=instruction)

    await executor.run(root_uri)

    assert processor.file_instructions == [instruction]
    assert processor.overview_instructions == [instruction]


@pytest.mark.asyncio
async def test_overview_task_skips_vlm_on_cache_hit(monkeypatch):
    """End-to-end: when .overview.hash matches the input bundle, skip VLM."""
    _mock_transaction_layer(monkeypatch)
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    root_uri = "viking://resources/cache-hit"
    tree = {
        root_uri: [
            {"name": "doc.md", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    processor = _FakeProcessor()
    executor = _make_executor(monkeypatch, fake_fs, processor)

    # Compute the hash that the executor would produce for these inputs.
    expected_files = [{"name": "doc.md", "summary": "summary"}]
    expected_hash = executor._compute_overview_input_hash(expected_files, [])

    fake_fs._files[f"{root_uri}/{OVERVIEW_HASH_FILENAME}"] = expected_hash
    fake_fs._files[f"{root_uri}/.overview.md"] = "cached-overview"
    fake_fs._files[f"{root_uri}/.abstract.md"] = "cached-abstract"

    await executor.run(root_uri)
    assert processor.overview_calls == 0


@pytest.mark.asyncio
async def test_overview_task_skips_disk_writes_on_cache_hit(monkeypatch):
    """Cache hit must avoid rewriting .overview.md/.abstract.md/.overview.hash."""
    _mock_transaction_layer(monkeypatch)
    _seed_prompt_sha(monkeypatch, "stub-prompt-sha")
    root_uri = "viking://resources/cache-no-rewrite"
    tree = {
        root_uri: [
            {"name": "doc.md", "isDir": False},
        ],
    }
    fake_fs = _FakeVikingFS(tree)
    processor = _FakeProcessor()
    executor = _make_executor(monkeypatch, fake_fs, processor)

    expected_files = [{"name": "doc.md", "summary": "summary"}]
    expected_hash = executor._compute_overview_input_hash(expected_files, [])

    fake_fs._files[f"{root_uri}/{OVERVIEW_HASH_FILENAME}"] = expected_hash
    fake_fs._files[f"{root_uri}/.overview.md"] = "cached-overview"
    fake_fs._files[f"{root_uri}/.abstract.md"] = "cached-abstract"

    await executor.run(root_uri)

    written_paths = [path for path, _ in fake_fs.writes]
    assert f"{root_uri}/.overview.md" not in written_paths
    assert f"{root_uri}/.abstract.md" not in written_paths
    assert f"{root_uri}/{OVERVIEW_HASH_FILENAME}" not in written_paths
