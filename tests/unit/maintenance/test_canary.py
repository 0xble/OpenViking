# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Tests for the canary phase of MemoryConsolidator (Phase D)."""

from unittest.mock import MagicMock, patch

import pytest

from openviking.maintenance import Canary, CanaryResult
from openviking.maintenance.memory_consolidator import MemoryConsolidator
from tests.unit.maintenance.conftest import (
    make_consolidator as _make_consolidator,
    make_request_ctx as _make_request_ctx,
    noop_lock as _noop_lock,
)


class TestCanaryStructure:
    def test_canary_from_dict(self):
        c = Canary.from_dict({"query": "how do I X", "expected_top_uri": "viking://x"})
        assert c.query == "how do I X"
        assert c.expected_top_uri == "viking://x"

    def test_canary_from_dict_handles_missing_keys(self):
        c = Canary.from_dict({})
        assert c.query == ""
        assert c.expected_top_uri == ""


class TestRunCanaries:
    @pytest.mark.asyncio
    async def test_canary_satisfied_when_expected_uri_in_top(self):
        consolidator = _make_consolidator(
            search_results={
                "memories": [
                    {"uri": "viking://x/memories/patterns/keeper.md"},
                    {"uri": "viking://x/memories/patterns/other.md"},
                ]
            }
        )
        canaries = [
            Canary(
                query="how do I build",
                expected_top_uri="viking://x/memories/patterns/keeper.md",
            )
        ]
        results = await consolidator._run_canaries(
            "viking://x/memories/patterns/", canaries, _make_request_ctx()
        )
        assert len(results) == 1
        r = results[0]
        assert r["found_in_top_n"] is True
        assert r["found_position"] == 0
        assert r["found_top_uri"] == "viking://x/memories/patterns/keeper.md"

    @pytest.mark.asyncio
    async def test_canary_unsatisfied_when_expected_missing(self):
        consolidator = _make_consolidator(
            search_results={"memories": [{"uri": "viking://x/memories/patterns/other.md"}]}
        )
        canaries = [
            Canary(
                query="how do I build",
                expected_top_uri="viking://x/memories/patterns/keeper.md",
            )
        ]
        results = await consolidator._run_canaries(
            "viking://x/memories/patterns/", canaries, _make_request_ctx()
        )
        assert results[0]["found_in_top_n"] is False
        assert results[0]["found_position"] == -1

    @pytest.mark.asyncio
    async def test_canary_swallows_search_failure(self):
        consolidator = _make_consolidator(
            search_results=lambda **_: (_ for _ in ()).throw(RuntimeError("search down"))
        )
        canaries = [Canary(query="x", expected_top_uri="viking://y")]
        results = await consolidator._run_canaries(
            "viking://x/", canaries, _make_request_ctx()
        )
        assert results[0]["found_in_top_n"] is False

    @pytest.mark.asyncio
    async def test_no_service_returns_empty_uris(self):
        consolidator = _make_consolidator(with_service=False)
        canaries = [Canary(query="x", expected_top_uri="viking://y")]
        results = await consolidator._run_canaries(
            "viking://x/", canaries, _make_request_ctx()
        )
        assert results[0]["found_in_top_n"] is False


class TestCanaryRegression:
    def test_no_regression_when_both_satisfied(self):
        pre = [{"query": "q", "found_in_top_n": True, "found_position": 0}]
        post = [{"query": "q", "found_in_top_n": True, "found_position": 1}]
        assert MemoryConsolidator._canary_regressed(pre, post) is False

    def test_regression_when_pre_passed_post_failed(self):
        pre = [{"query": "q", "found_in_top_n": True, "found_position": 0}]
        post = [{"query": "q", "found_in_top_n": False, "found_position": -1}]
        assert MemoryConsolidator._canary_regressed(pre, post) is True

    def test_no_regression_when_both_failed(self):
        # Pre-existing miss is not a regression.
        pre = [{"query": "q", "found_in_top_n": False, "found_position": -1}]
        post = [{"query": "q", "found_in_top_n": False, "found_position": -1}]
        assert MemoryConsolidator._canary_regressed(pre, post) is False

    def test_no_regression_for_post_only_canary(self):
        pre = []
        post = [{"query": "q", "found_in_top_n": False, "found_position": -1}]
        assert MemoryConsolidator._canary_regressed(pre, post) is False


class TestRunWithCanaries:
    @pytest.mark.asyncio
    async def test_canary_phases_recorded_on_run(self):
        consolidator = _make_consolidator(
            search_results={"memories": [{"uri": "viking://x/m/keeper.md"}]}
        )
        canaries = [Canary(query="x", expected_top_uri="viking://x/m/keeper.md")]
        with (
            patch("openviking.maintenance.memory_consolidator.LockContext", _noop_lock),
            patch(
                "openviking.maintenance.memory_consolidator.get_lock_manager",
                return_value=MagicMock(),
            ),
        ):
            result = await consolidator.run(
                "viking://x/m/",
                _make_request_ctx(),
                canaries=canaries,
            )
        assert len(result.canaries_pre) == 1
        assert len(result.canaries_post) == 1
        assert result.canary_failed is False

    @pytest.mark.asyncio
    async def test_dry_run_skips_canaries(self):
        consolidator = _make_consolidator()
        canaries = [Canary(query="x", expected_top_uri="viking://x")]
        with (
            patch("openviking.maintenance.memory_consolidator.LockContext", _noop_lock),
            patch(
                "openviking.maintenance.memory_consolidator.get_lock_manager",
                return_value=MagicMock(),
            ),
        ):
            result = await consolidator.run(
                "viking://x/m/",
                _make_request_ctx(),
                dry_run=True,
                canaries=canaries,
            )
        assert result.canaries_pre == []
        assert result.canaries_post == []
