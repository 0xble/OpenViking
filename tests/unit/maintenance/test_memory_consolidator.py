# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for MemoryConsolidator orchestrator."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from openviking.maintenance.memory_consolidator import (
    AUDIT_PATH_FRAGMENT,
    ConsolidationResult,
)
from openviking.session.memory_deduplicator import (
    ClusterDecision,
    ClusterDecisionType,
)
from tests.unit.conftest import make_test_context as _ctx
from tests.unit.maintenance.conftest import make_consolidator, make_request_ctx


# Local aliases keep the existing test bodies untouched.
def _make_consolidator(**kwargs):
    return make_consolidator(with_service=False, **kwargs)


_make_request_ctx = make_request_ctx


class TestRunHappyPath:
    @pytest.mark.asyncio
    async def test_dry_run_writes_no_files_and_records_plan(self):
        archive = [MagicMock()]
        consolidator = _make_consolidator(archive_candidates=archive)
        consolidator._cluster_scope = AsyncMock(
            return_value=[
                [
                    _ctx("viking://agent/a/memories/patterns/x"),
                    _ctx("viking://agent/a/memories/patterns/y"),
                ]
            ]
        )

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx(),
            dry_run=True,
        )

        assert result.dry_run is True
        assert result.candidates["merge_clusters"] == 1
        assert result.candidates["archive"] == 1
        consolidator.dedup.consolidate_cluster.assert_not_called()
        consolidator.archiver.archive.assert_not_called()
        consolidator.viking_fs.write.assert_called_once()  # audit only

    @pytest.mark.asyncio
    async def test_targeted_dry_run_skips_full_archive_scan(self):
        consolidator = _make_consolidator(archive_candidates=[MagicMock()])
        consolidator._cluster_scope = AsyncMock(return_value=[])
        target_uris = ["viking://agent/a/memories/skills/changed.md"]
        ctx = _make_request_ctx()

        await consolidator.run(
            "viking://agent/a/memories/skills/",
            ctx,
            dry_run=True,
            target_uris=target_uris,
        )

        consolidator.archiver.scan.assert_not_called()
        consolidator._cluster_scope.assert_awaited_once_with(
            "viking://agent/a/memories/skills/",
            ctx,
            target_uris=target_uris,
        )

    @pytest.mark.asyncio
    async def test_keep_and_merge_writes_keeper_and_archives_sources(self):
        cluster = [
            _ctx("viking://agent/a/memories/patterns/keeper"),
            _ctx("viking://agent/a/memories/patterns/dup"),
        ]
        decision = ClusterDecision(
            decision=ClusterDecisionType.KEEP_AND_MERGE,
            cluster=cluster,
            keeper_uri="viking://agent/a/memories/patterns/keeper",
            merge_into=["viking://agent/a/memories/patterns/dup"],
            merged_content="merged body",
            merged_abstract="merged abstract",
            reason="same fact",
        )
        consolidator = _make_consolidator(cluster_decision=decision)
        consolidator._cluster_scope = AsyncMock(return_value=[cluster])

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx(),
        )

        # keeper write + audit = 2
        assert consolidator.viking_fs.write.call_count == 2
        consolidator.viking_fs.rm.assert_not_called()
        consolidator.archiver.archive.assert_awaited_once()
        archive_candidates = consolidator.archiver.archive.await_args.args[0]
        assert [c.uri for c in archive_candidates] == ["viking://agent/a/memories/patterns/dup"]
        assert result.ops_applied["merged"] == 1
        assert "viking://agent/a/memories/patterns/keeper" in result.applied_uris
        assert "viking://agent/a/memories/patterns/dup" in result.applied_uris

    @pytest.mark.asyncio
    async def test_keep_and_merge_with_empty_content_skips_deletes(self):
        # Regression: empty merged_content used to delete sources without
        # writing keeper -> data loss. Now skipped, marked partial.
        cluster = [
            _ctx("viking://agent/a/memories/patterns/keeper"),
            _ctx("viking://agent/a/memories/patterns/dup"),
        ]
        decision = ClusterDecision(
            decision=ClusterDecisionType.KEEP_AND_MERGE,
            cluster=cluster,
            keeper_uri="viking://agent/a/memories/patterns/keeper",
            merge_into=["viking://agent/a/memories/patterns/dup"],
            merged_content="",  # bug trigger
            merged_abstract="",
        )
        consolidator = _make_consolidator(cluster_decision=decision)
        consolidator._cluster_scope = AsyncMock(return_value=[cluster])

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx(),
        )

        consolidator.viking_fs.rm.assert_not_called()
        assert result.ops_applied["merged"] == 0
        assert result.partial is True
        assert any("merge_skipped_empty_content" in e for e in result.errors)

    @pytest.mark.asyncio
    async def test_overlapping_clusters_are_consolidated_once(self):
        first_cluster = [
            _ctx("viking://agent/a/memories/skills/keeper"),
            _ctx("viking://agent/a/memories/skills/dup"),
        ]
        second_cluster = [
            _ctx("viking://agent/a/memories/skills/dup"),
            _ctx("viking://agent/a/memories/skills/other"),
        ]
        decision = ClusterDecision(
            decision=ClusterDecisionType.KEEP_AND_MERGE,
            cluster=first_cluster,
            keeper_uri="viking://agent/a/memories/skills/keeper",
            merge_into=["viking://agent/a/memories/skills/dup"],
            merged_content="merged body",
            merged_abstract="merged abstract",
            reason="same fact",
        )
        consolidator = _make_consolidator(cluster_decision=decision)
        consolidator._cluster_scope = AsyncMock(return_value=[first_cluster, second_cluster])

        result = await consolidator.run(
            "viking://agent/a/memories/skills/",
            _make_request_ctx(),
        )

        consolidator.dedup.consolidate_cluster.assert_awaited_once()
        assert len(result.cluster_decisions) == 1
        assert result.ops_applied["merged"] == 1

    @pytest.mark.asyncio
    async def test_keep_and_delete_archives_invalidated_members(self):
        cluster = [
            _ctx("viking://agent/a/memories/preferences/k"),
            _ctx("viking://agent/a/memories/preferences/old"),
        ]
        decision = ClusterDecision(
            decision=ClusterDecisionType.KEEP_AND_DELETE,
            cluster=cluster,
            keeper_uri="viking://agent/a/memories/preferences/k",
            delete=["viking://agent/a/memories/preferences/old"],
            reason="user changed editors",
        )
        consolidator = _make_consolidator(cluster_decision=decision)
        consolidator._cluster_scope = AsyncMock(return_value=[cluster])

        result = await consolidator.run(
            "viking://agent/a/memories/preferences/",
            _make_request_ctx(),
        )

        consolidator.viking_fs.rm.assert_not_called()
        assert result.ops_applied["archived"] == 1
        assert result.ops_applied["deleted"] == 0
        assert result.ops_applied["merged"] == 0


class TestEmptyScope:
    @pytest.mark.asyncio
    async def test_empty_scope_is_clean_noop(self):
        consolidator = _make_consolidator()  # no clusters, no archive

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx(),
        )

        assert result.candidates["merge_clusters"] == 0
        assert result.candidates["archive"] == 0
        assert result.ops_applied["merged"] == 0
        assert result.ops_applied["deleted"] == 0
        assert result.ops_applied["archived"] == 0
        assert not result.partial
        # Audit still written.
        consolidator.viking_fs.write.assert_called_once()


class TestTargetedClustering:
    @pytest.mark.asyncio
    async def test_targeted_clustering_searches_only_dirty_uri_neighbors(self, monkeypatch):
        class EmbedResult:
            dense_vector = [0.1, 0.2, 0.3]

        async def fake_embed_compat(*_args, **_kwargs):
            return EmbedResult()

        monkeypatch.setattr(
            "openviking.models.embedder.base.embed_compat",
            fake_embed_compat,
        )
        consolidator = _make_consolidator()
        consolidator.dedup.embedder = object()
        consolidator.vikingdb.search_similar_memories = AsyncMock(
            return_value=[
                {
                    "uri": "viking://agent/a/memories/skills/similar.md",
                    "abstract": "same skill",
                    "level": 2,
                    "_score": 0.99,
                }
            ]
        )
        consolidator.vikingdb.scroll = AsyncMock(
            side_effect=AssertionError("targeted clustering must not scroll")
        )

        clusters = await consolidator._cluster_target_uris(
            "viking://agent/a/memories/skills/",
            ["viking://agent/a/memories/skills/changed.md"],
            _make_request_ctx(),
        )

        assert [[context.uri for context in cluster] for cluster in clusters] == [
            [
                "viking://agent/a/memories/skills/changed.md",
                "viking://agent/a/memories/skills/similar.md",
            ]
        ]
        consolidator.vikingdb.search_similar_memories.assert_awaited_once()
        consolidator.vikingdb.scroll.assert_not_called()

    @pytest.mark.asyncio
    async def test_targeted_clustering_ignores_chunk_fragment_hits(self, monkeypatch):
        class EmbedResult:
            dense_vector = [0.1, 0.2, 0.3]

        async def fake_embed_compat(*_args, **_kwargs):
            return EmbedResult()

        monkeypatch.setattr(
            "openviking.models.embedder.base.embed_compat",
            fake_embed_compat,
        )
        consolidator = _make_consolidator()
        consolidator.dedup.embedder = object()
        consolidator.vikingdb.search_similar_memories = AsyncMock(
            return_value=[
                {
                    "uri": "viking://agent/a/memories/skills/torrent.md#chunk_0000",
                    "abstract": "same skill chunk",
                    "level": 2,
                    "_score": 0.99,
                },
                {
                    "uri": "viking://agent/a/memories/skills/similar.md",
                    "abstract": "same skill",
                    "level": 2,
                    "_score": 0.99,
                },
            ]
        )

        clusters = await consolidator._cluster_target_uris(
            "viking://agent/a/memories/skills/",
            ["viking://agent/a/memories/skills/changed.md"],
            _make_request_ctx(),
        )

        assert [[context.uri for context in cluster] for cluster in clusters] == [
            [
                "viking://agent/a/memories/skills/changed.md",
                "viking://agent/a/memories/skills/similar.md",
            ]
        ]


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_one_cluster_fails_others_commit(self):
        good_cluster = [
            _ctx("viking://agent/a/memories/patterns/g1"),
            _ctx("viking://agent/a/memories/patterns/g2"),
        ]
        bad_cluster = [
            _ctx("viking://agent/a/memories/patterns/b1"),
            _ctx("viking://agent/a/memories/patterns/b2"),
        ]
        consolidator = _make_consolidator()
        consolidator._cluster_scope = AsyncMock(return_value=[good_cluster, bad_cluster])

        good_decision = ClusterDecision(
            decision=ClusterDecisionType.KEEP_AND_DELETE,
            cluster=good_cluster,
            keeper_uri="viking://agent/a/memories/patterns/g1",
            delete=["viking://agent/a/memories/patterns/g2"],
        )

        async def consolidate_side_effect(cluster, **kwargs):
            if cluster is bad_cluster:
                raise RuntimeError("bad cluster boom")
            return good_decision

        consolidator.dedup.consolidate_cluster = AsyncMock(side_effect=consolidate_side_effect)

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx(),
        )

        assert result.partial is True
        assert any("cluster_failed" in e for e in result.errors)
        # Good cluster's delete still applied.
        assert result.ops_applied["archived"] == 1


class TestAuditRecord:
    @pytest.mark.asyncio
    async def test_audit_uri_is_account_scoped_and_payload_is_valid_json(self):
        consolidator = _make_consolidator()

        result = await consolidator.run(
            "viking://agent/test-account/memories/patterns/",
            _make_request_ctx("test-account"),
        )

        assert result.audit_uri.startswith(f"viking://agent/test-account/{AUDIT_PATH_FRAGMENT}/")
        assert result.audit_uri.endswith(".json")
        # Last write call is the audit; payload must be valid JSON.
        write_call = consolidator.viking_fs.write.call_args_list[-1]
        payload = write_call.args[1]
        parsed = json.loads(payload)
        assert parsed["scope_uri"] == "viking://agent/test-account/memories/patterns/"
        assert "phase_durations" in parsed
        assert "ops_applied" in parsed

    @pytest.mark.asyncio
    async def test_default_account_when_ctx_missing_account_id(self):
        consolidator = _make_consolidator()
        ctx = MagicMock(spec=[])  # no account_id attribute

        result = await consolidator.run(
            "viking://agent/x/memories/patterns/",
            ctx,
        )

        assert "/agent/default/" in result.audit_uri


class TestReindexRegenerateGate:
    @pytest.mark.parametrize(
        "ops_applied",
        [
            {"archived": 3},
            {"merged": 1},
            {"deleted": 2},
        ],
    )
    @pytest.mark.asyncio
    async def test_reindex_runs_for_mutations(self, ops_applied):
        consolidator = make_consolidator(with_service=True)
        consolidator.service.reindex = AsyncMock()
        result = ConsolidationResult(scope_uri="viking://agent/a/memories/patterns/")
        result.ops_applied.update(ops_applied)

        await consolidator._reindex(result.scope_uri, _make_request_ctx("a"), result)

        assert consolidator.service.reindex.await_count == 1
        kwargs = consolidator.service.reindex.await_args.kwargs
        assert kwargs["mode"] == "semantic_and_vectors"
        assert kwargs["lock_already_held"] is False
        assert kwargs["uri"] == result.scope_uri

    @pytest.mark.asyncio
    async def test_targeted_reindex_only_existing_applied_uris(self):
        consolidator = make_consolidator(with_service=True)
        consolidator.service.reindex = AsyncMock()
        consolidator.viking_fs.exists = AsyncMock(
            side_effect=lambda uri, **_: uri.endswith("/keeper")
        )
        result = ConsolidationResult(scope_uri="viking://user/a/memories/events/")
        result.ops_applied["merged"] = 1
        result.applied_uris = [
            "viking://user/a/memories/events/archived",
            "viking://user/a/memories/events/keeper",
            "viking://user/a/memories/events/keeper#chunk_0000",
        ]

        await consolidator._reindex(
            result.scope_uri,
            _make_request_ctx("a"),
            result,
            target_uris=["viking://user/a/memories/events/keeper"],
        )

        consolidator.service.reindex.assert_awaited_once()
        kwargs = consolidator.service.reindex.await_args.kwargs
        assert kwargs["uri"] == "viking://user/a/memories/events/keeper"
        assert kwargs["mode"] == "semantic_and_vectors"

    @pytest.mark.asyncio
    async def test_targeted_reindex_skips_when_only_archived_sources_changed(self):
        consolidator = make_consolidator(with_service=True)
        consolidator.service.reindex = AsyncMock()
        consolidator.viking_fs.exists = AsyncMock(return_value=False)
        result = ConsolidationResult(scope_uri="viking://user/a/memories/events/")
        result.ops_applied["archived"] = 1
        result.applied_uris = ["viking://user/a/memories/events/archived"]

        await consolidator._reindex(
            result.scope_uri,
            _make_request_ctx("a"),
            result,
            target_uris=["viking://user/a/memories/events/archived"],
        )

        consolidator.service.reindex.assert_not_awaited()
        assert result.phase_durations["reindex"] == 0.0

    @pytest.mark.asyncio
    async def test_reindex_skips_idle_pass(self):
        consolidator = make_consolidator(with_service=True)
        consolidator.service.reindex = AsyncMock()
        result = ConsolidationResult(scope_uri="viking://agent/a/memories/patterns/")

        await consolidator._reindex(result.scope_uri, _make_request_ctx("a"), result)

        consolidator.service.reindex.assert_not_awaited()
        assert result.phase_durations["reindex"] == 0.0

    @pytest.mark.asyncio
    async def test_run_records_zero_reindex_duration_on_idle_pass(self):
        consolidator = make_consolidator(with_service=True)
        consolidator.service.reindex = AsyncMock()

        result = await consolidator.run(
            "viking://agent/a/memories/patterns/",
            _make_request_ctx("a"),
        )

        consolidator.service.reindex.assert_not_awaited()
        assert result.phase_durations["reindex"] == 0.0
