# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from openviking.source import normalize_session_log


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_normalize_codex_session(tmp_path):
    raw = tmp_path / "codex.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "session_meta",
                "payload": {"id": "abc123", "cwd": "/tmp/project", "cli_version": "1.0.0"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "world"}],
                },
            },
        ],
    )

    result = normalize_session_log("codex", raw)

    assert result.session_id == "codex-abc123"
    assert [message.role for message in result.messages] == ["user", "assistant"]
    assert result.messages[0].content == "hello"
    assert result.messages[1].content == "world"
    assert result.metadata["event_count"] == 3


def test_normalize_codex_session_preserves_explicit_id_and_raw_source_id(tmp_path):
    raw = tmp_path / "codex.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "session_meta",
                "payload": {"id": "abc123", "cwd": "/tmp/project", "cli_version": "1.0.0"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ],
    )

    overridden = normalize_session_log("codex", raw, session_id="incident-42")
    assert overridden.session_id == "incident-42"
    assert overridden.metadata["source_session_id"] == "abc123"

    prefixed = normalize_session_log("codex", raw, session_id="codex-incident-42")
    assert prefixed.session_id == "codex-incident-42"
    assert prefixed.metadata["source_session_id"] == "abc123"


def test_normalize_openclaw_session(tmp_path):
    raw = tmp_path / "openclaw.jsonl"
    _write_jsonl(
        raw,
        [
            {"type": "session", "id": "sess-openclaw", "cwd": "/tmp/project", "version": "1"},
            {
                "type": "message",
                "timestamp": "2026-03-09T12:10:00Z",
                "message": {"role": "user", "content": [{"type": "text", "text": "ping"}]},
            },
            {
                "type": "message",
                "timestamp": "2026-03-09T12:10:01Z",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "pong"}]},
            },
        ],
    )

    result = normalize_session_log("openclaw", raw)

    assert result.session_id == "openclaw-sess-openclaw"
    assert [message.content for message in result.messages] == ["ping", "pong"]


def test_normalize_claude_hook_only_session(tmp_path):
    raw = tmp_path / "claude-hook-only.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "progress",
                "sessionId": "claude-empty",
                "cwd": "/Users/brianle",
                "timestamp": "2026-03-09T00:45:57.290Z",
                "version": "2.1.71",
                "data": {"type": "hook_progress"},
            },
            {
                "type": "progress",
                "sessionId": "claude-empty",
                "cwd": "/Users/brianle",
                "timestamp": "2026-03-09T00:45:58.290Z",
                "version": "2.1.71",
                "data": {"type": "hook_progress"},
            },
        ],
    )

    result = normalize_session_log("claude", raw)

    assert result.session_id == "claude-claude-empty"
    assert result.messages == []
    assert result.metadata["progress_event_count"] == 2
    assert result.metadata["message_count"] == 0


def test_normalize_claude_requires_exact_timestamps(tmp_path):
    raw = tmp_path / "claude.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "user",
                "sessionId": "claude-session",
                "message": {"content": [{"type": "text", "text": "missing timestamp"}]},
            }
        ],
    )

    with pytest.raises(ValueError, match="Missing exact timestamp"):
        normalize_session_log("claude", raw)
