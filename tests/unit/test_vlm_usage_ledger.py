# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0

from datetime import datetime, timezone

from openviking.models.vlm.usage_ledger import read_vlm_usage_summary, record_vlm_call


def test_usage_ledger_persists_and_summarizes(monkeypatch, tmp_path):
    ledger = tmp_path / "vlm_calls.jsonl"
    monkeypatch.setenv("OPENVIKING_VLM_USAGE_LEDGER", str(ledger))

    record_vlm_call(
        provider="litellm",
        model_name="gemini-3.1-flash-lite-preview",
        prompt_tokens=10,
        completion_tokens=3,
        duration_seconds=0.5,
        account_id="acct",
        operation="session_memory_extract",
        stage="memory_extract",
    )

    summary = read_vlm_usage_summary(now=datetime.now(timezone.utc))

    assert summary["available"] is True
    assert summary["totals"]["all_time"]["total_tokens"] == 13
    assert summary["totals"]["last_24h"]["calls"] == 1
    assert summary["totals"]["by_operation"]["session_memory_extract"]["prompt_tokens"] == 10
