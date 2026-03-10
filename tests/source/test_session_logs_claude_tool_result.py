# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

from openviking.source import normalize_session_log


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normalize_claude_counts_nested_tool_results(tmp_path):
    raw = tmp_path / "claude-tool.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "user",
                "sessionId": "claude-tools",
                "timestamp": "2026-03-09T12:00:00Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "ok",
                        }
                    ]
                },
            }
        ],
    )

    result = normalize_session_log("claude", raw)

    assert result.session_id == "claude-claude-tools"
    assert result.messages == []
    assert result.metadata["tool_event_count"] == 1
