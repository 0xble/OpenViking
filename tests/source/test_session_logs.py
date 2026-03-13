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


def test_normalize_session_marks_low_signal_greeting_as_not_index_eligible(tmp_path):
    raw = tmp_path / "codex-hi.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "session_meta",
                "payload": {"id": "hi-only", "cwd": "/tmp/project", "cli_version": "1.0.0"},
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi!!!"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello"}],
                },
            },
        ],
    )

    result = normalize_session_log("codex", raw)

    assert result.metadata["first_user_prompt"] == "Hi!!!"
    assert result.metadata["index_eligible"] is False
    assert result.metadata["index_skip_category"] == "greeting"
    assert result.metadata["index_skip_reason"] == "low-signal first user prompt (greeting)"


def test_normalize_session_keeps_indexing_when_trivial_opener_has_real_followup(tmp_path):
    raw = tmp_path / "codex-hi-followup.jsonl"
    _write_jsonl(
        raw,
        [
            {
                "type": "session_meta",
                "payload": {
                    "id": "hi-followup",
                    "cwd": "/tmp/project",
                    "cli_version": "1.0.0",
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hey"}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:02Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Can you explain why my postgres migration is deadlocking in CI?",
                        }
                    ],
                },
            },
        ],
    )

    result = normalize_session_log("codex", raw)

    assert result.metadata["index_eligible"] is True
    assert "index_skip_reason" not in result.metadata


@pytest.mark.parametrize(
    ("prompt", "expected_category"),
    [
        ("hello!!!", "greeting"),
        ("pls hello", "greeting"),
        ("this is a test", "test"),
        ("just testing", "test"),
        ("smoke test", "test"),
        ("tell me a short bedtime story", "story"),
        ("make up a story", "story"),
        ("please tell me what model you are", "model-check"),
        ("what model are you running right now", "model-check"),
        ("who are you", "identity-check"),
    ],
)
def test_normalize_session_skips_expected_low_signal_variants(tmp_path, prompt, expected_category):
    raw = tmp_path / "codex-low-signal-variants.jsonl"
    _write_jsonl(
        raw,
        [
            {"type": "session_meta", "payload": {"id": "variant-check", "cwd": "/tmp/project"}},
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            },
        ],
    )

    result = normalize_session_log("codex", raw)

    assert result.metadata["index_eligible"] is False
    assert result.metadata["index_skip_category"] == expected_category


@pytest.mark.parametrize(
    "prompt",
    [
        "hello can you debug this stack trace for me",
        "this is a test of my webhook parser and I need help understanding why it fails",
        "what model should I use for embedding restaurant menus",
        "tell me a story about raft leader election tradeoffs",
        "who are you working for on this incident response timeline",
    ],
)
def test_normalize_session_does_not_skip_substantive_prompts_with_overlapping_words(
    tmp_path, prompt
):
    raw = tmp_path / "codex-substantive-overlap.jsonl"
    _write_jsonl(
        raw,
        [
            {"type": "session_meta", "payload": {"id": "overlap-check", "cwd": "/tmp/project"}},
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                },
            },
            {
                "type": "response_item",
                "timestamp": "2026-03-09T12:00:01Z",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}],
                },
            },
        ],
    )

    result = normalize_session_log("codex", raw)

    assert result.metadata["index_eligible"] is True
    assert "index_skip_reason" not in result.metadata
