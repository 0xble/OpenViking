# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Normalize external session logs into native OpenViking messages."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openviking.message import Message
from openviking.message.part import TextPart
from openviking.utils.time_utils import parse_iso_datetime

SUPPORTED_SESSION_ADAPTERS = ("claude", "codex", "openclaw")

_GREETING_PROMPTS = {
    ("hi",),
    ("hi", "there"),
    ("hello",),
    ("hello", "there"),
    ("hey",),
    ("hey", "there"),
    ("hiya",),
    ("yo",),
    ("sup",),
    ("good", "morning"),
    ("good", "afternoon"),
    ("good", "evening"),
}

_TEST_PREFIXES = (
    ("this", "is", "a"),
    ("this", "is"),
    ("it", "is", "a"),
    ("it", "is"),
    ("its", "a"),
    ("just", "a"),
    ("just",),
)

_TEST_PROMPTS = {
    ("test",),
    ("testing",),
    ("smoke", "test"),
    ("test", "run"),
    ("dry", "run"),
    ("trial",),
}

_STORY_PROMPTS = {
    ("tell", "me", "a", "story"),
    ("tell", "me", "a", "short", "story"),
    ("tell", "me", "a", "bedtime", "story"),
    ("tell", "me", "a", "short", "bedtime", "story"),
    ("tell", "me", "another", "story"),
    ("write", "a", "story"),
    ("write", "me", "a", "story"),
    ("write", "a", "short", "story"),
    ("write", "a", "bedtime", "story"),
    ("make", "up", "a", "story"),
    ("make", "up", "a", "short", "story"),
    ("make", "up", "a", "bedtime", "story"),
}

_MODEL_CHECK_PROMPTS = {
    ("what", "model", "are", "you"),
    ("what", "model", "are", "you", "running"),
    ("what", "model", "is", "this"),
    ("what", "model", "am", "i", "talking", "to"),
    ("which", "model", "are", "you"),
    ("which", "model", "are", "you", "running"),
    ("which", "model", "is", "this"),
    ("tell", "me", "what", "model", "you", "are"),
    ("can", "you", "tell", "me", "what", "model", "you", "are"),
    ("do", "you", "know", "what", "model", "you", "are"),
}

_IDENTITY_CHECK_PROMPTS = {
    ("who", "are", "you"),
    ("what", "are", "you"),
}

_EDGE_POLITENESS_TOKENS = {"please", "pls"}


@dataclass
class SessionImportResult:
    adapter: str
    session_id: str
    messages: List[Message]
    metadata: Dict[str, Any]


def normalize_session_log(
    adapter: str,
    path: str | Path,
    session_id: Optional[str] = None,
) -> SessionImportResult:
    normalized_adapter = adapter.strip().lower()
    if normalized_adapter not in SUPPORTED_SESSION_ADAPTERS:
        raise ValueError(
            f"Unsupported session adapter: {adapter}. "
            f"Expected one of {', '.join(SUPPORTED_SESSION_ADAPTERS)}"
        )

    raw_path = Path(path).expanduser().resolve()
    if not raw_path.exists():
        raise FileNotFoundError(f"Session log does not exist: {raw_path}")

    parser = {
        "claude": _parse_claude,
        "codex": _parse_codex,
        "openclaw": _parse_openclaw,
    }[normalized_adapter]
    result = parser(raw_path, session_id=session_id)
    root_session_eligible, root_skip_reason = root_session_eligibility(normalized_adapter, raw_path)
    result.metadata["root_session_eligible"] = root_session_eligible
    if root_skip_reason:
        result.metadata["root_skip_reason"] = root_skip_reason
    first_user_prompt = _first_user_prompt(result.messages)
    index_eligible, index_skip_category, index_skip_reason = _index_eligibility(result.messages)
    result.metadata["first_user_prompt"] = first_user_prompt
    result.metadata["index_eligible"] = index_eligible
    if index_skip_category:
        result.metadata["index_skip_category"] = index_skip_category
    if index_skip_reason:
        result.metadata["index_skip_reason"] = index_skip_reason
    return result


def root_session_eligibility(adapter: str, path: str | Path) -> tuple[bool, Optional[str]]:
    normalized_adapter = adapter.strip().lower()
    raw_path = Path(path).expanduser().resolve()
    first_event = _first_session_event(raw_path)

    if normalized_adapter == "claude":
        if any(part.lower() == "subagents" for part in raw_path.parts):
            return False, "subagent session"
        if not isinstance(first_event, dict):
            return False, "non-root session"
        return (
            first_event.get("isSidechain") is False,
            None if first_event.get("isSidechain") is False else "non-root session",
        )

    if normalized_adapter == "codex":
        if not isinstance(first_event, dict) or first_event.get("type") != "session_meta":
            return False, "ephemeral session"
        payload = first_event.get("payload")
        if not isinstance(payload, dict):
            return False, "ephemeral session"
        if payload.get("originator") != "codex_cli_rs":
            return False, "ephemeral session"
        if payload.get("source") != "cli":
            return False, "subagent session"
        return True, None

    if normalized_adapter == "openclaw":
        if "-topic-" not in raw_path.name:
            return False, "ephemeral session"
        return True, None

    return True, None


def _parse_codex(path: Path, session_id: Optional[str]) -> SessionImportResult:
    explicit_session_id = session_id.strip() if session_id and session_id.strip() else None
    source_session_id: Optional[str] = None
    messages: List[Message] = []
    instruction_messages: List[Dict[str, Any]] = []
    tool_event_count = 0
    session_meta: Dict[str, Any] = {}
    message_index = 0

    for event in _iter_jsonl(path):
        event_type = event.get("type")
        if event_type == "session_meta":
            payload = event.get("payload") or {}
            session_meta = {
                "cwd": payload.get("cwd"),
                "cli_version": payload.get("cli_version"),
                "originator": payload.get("originator"),
                "source": payload.get("source"),
                "model_provider": payload.get("model_provider"),
            }
            if source_session_id is None:
                source_session_id = str(payload.get("id") or "").strip() or None
            continue

        if event_type != "response_item":
            continue

        payload = event.get("payload") or {}
        payload_type = payload.get("type")
        if payload_type == "message":
            timestamp = _parse_timestamp(
                event.get("timestamp"),
                required=True,
                context=f"codex message event in {path}",
            )
            role = str(payload.get("role") or "").strip().lower()
            text = _extract_codex_message_text(payload)
            if not text:
                continue
            if role == "user":
                messages.append(_create_message("user", text, timestamp, message_index))
                message_index += 1
            elif role == "assistant":
                messages.append(_create_message("assistant", text, timestamp, message_index))
                message_index += 1
            elif role in {"developer", "system"}:
                instruction_messages.append(
                    {"role": role, "timestamp": _format_timestamp(timestamp), "text": text}
                )
        elif payload_type in {"function_call", "function_call_output"}:
            tool_event_count += 1

    resolved_session_id = _resolve_session_id("codex", path, source_session_id, explicit_session_id)
    return SessionImportResult(
        adapter="codex",
        session_id=resolved_session_id,
        messages=messages,
        metadata={
            "adapter": "codex",
            "source_session_id": source_session_id,
            "source_path": str(path),
            "instruction_messages": instruction_messages,
            "tool_event_count": tool_event_count,
            "event_count": _count_jsonl_lines(path),
            "message_count": len(messages),
            "session_meta": session_meta,
        },
    )


def _parse_openclaw(path: Path, session_id: Optional[str]) -> SessionImportResult:
    explicit_session_id = session_id.strip() if session_id and session_id.strip() else None
    source_session_id: Optional[str] = None
    messages: List[Message] = []
    tool_event_count = 0
    session_meta: Dict[str, Any] = {}
    message_index = 0

    for event in _iter_jsonl(path):
        event_type = event.get("type")

        if event_type == "session":
            session_meta = {
                "version": event.get("version"),
                "cwd": event.get("cwd"),
            }
            if source_session_id is None:
                source_session_id = str(event.get("id") or "").strip() or None
            continue

        if event_type != "message":
            continue

        payload = event.get("message") or {}
        role = str(payload.get("role") or "").strip()
        if role in {"user", "assistant"}:
            timestamp = _parse_timestamp(
                event.get("timestamp"),
                required=True,
                context=f"openclaw message event in {path}",
            )
            text = _extract_openclaw_text(payload.get("content"))
            if text:
                messages.append(_create_message(role, text, timestamp, message_index))
                message_index += 1
        elif role == "toolResult":
            tool_event_count += 1

    resolved_session_id = _resolve_session_id(
        "openclaw", path, source_session_id, explicit_session_id
    )
    return SessionImportResult(
        adapter="openclaw",
        session_id=resolved_session_id,
        messages=messages,
        metadata={
            "adapter": "openclaw",
            "source_session_id": source_session_id,
            "source_path": str(path),
            "tool_event_count": tool_event_count,
            "event_count": _count_jsonl_lines(path),
            "message_count": len(messages),
            "session_meta": session_meta,
        },
    )


def _parse_claude(path: Path, session_id: Optional[str]) -> SessionImportResult:
    explicit_session_id = session_id.strip() if session_id and session_id.strip() else None
    source_session_id: Optional[str] = None
    messages: List[Message] = []
    tool_event_count = 0
    progress_event_count = 0
    session_meta: Dict[str, Any] = {}
    message_index = 0

    for event in _iter_jsonl(path):
        event_type = event.get("type")

        if source_session_id is None:
            raw_session_id = str(event.get("sessionId") or "").strip()
            if raw_session_id:
                source_session_id = raw_session_id

        if not session_meta:
            session_meta = {
                "cwd": event.get("cwd"),
                "git_branch": event.get("gitBranch"),
                "version": event.get("version"),
                "user_type": event.get("userType"),
            }

        if event_type == "progress":
            progress_event_count += 1
            continue

        if event_type == "assistant":
            timestamp = _parse_timestamp(
                event.get("timestamp"),
                required=True,
                context=f"claude assistant event in {path}",
            )
            payload = event.get("message") or {}
            text = _extract_claude_content(payload.get("content"))
            if text:
                messages.append(_create_message("assistant", text, timestamp, message_index))
                message_index += 1
            continue

        if event_type == "user":
            timestamp = _parse_timestamp(
                event.get("timestamp"),
                required=True,
                context=f"claude user event in {path}",
            )
            payload = event.get("message")
            if _is_claude_tool_result(payload):
                tool_event_count += 1
                continue
            text = _extract_claude_content(payload if payload is not None else event.get("data"))
            if not text:
                continue
            messages.append(_create_message("user", text, timestamp, message_index))
            message_index += 1

    resolved_session_id = _resolve_session_id(
        "claude", path, source_session_id, explicit_session_id
    )
    return SessionImportResult(
        adapter="claude",
        session_id=resolved_session_id,
        messages=messages,
        metadata={
            "adapter": "claude",
            "source_session_id": source_session_id,
            "source_path": str(path),
            "tool_event_count": tool_event_count,
            "progress_event_count": progress_event_count,
            "event_count": _count_jsonl_lines(path),
            "message_count": len(messages),
            "session_meta": session_meta,
        },
    )


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _count_jsonl_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _first_session_event(path: Path) -> Optional[Dict[str, Any]]:
    for event in _iter_jsonl(path):
        return event
    return None


def _parse_timestamp(
    value: Any,
    *,
    required: bool = False,
    context: str = "event",
) -> datetime:
    if isinstance(value, str) and value.strip():
        return parse_iso_datetime(value)
    if required:
        raise ValueError(f"Missing exact timestamp for {context}")
    return datetime.now(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.isoformat()


def _create_message(role: str, text: str, timestamp: datetime, index: int) -> Message:
    return Message(
        id=f"imported_{role}_{index:06d}_{int(timestamp.timestamp() * 1_000_000)}",
        role=role,  # type: ignore[arg-type]
        parts=[TextPart(text=text)],
        created_at=timestamp,
    )


def _extract_codex_message_text(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list):
        return ""
    texts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type in {"input_text", "output_text"}:
            text = str(item.get("text") or "").strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts).strip()


def _extract_openclaw_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    texts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = str(item.get("text") or "").strip()
            if text:
                texts.append(text)
    return "\n\n".join(texts).strip()


def _extract_claude_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        return _extract_claude_content(content.get("content"))
    if not isinstance(content, list):
        return ""
    texts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text":
            text = str(item.get("text") or "").strip()
            if text:
                texts.append(text)
        elif item_type == "thinking":
            continue
    return "\n\n".join(texts).strip()


def _is_claude_tool_result(payload: Any) -> bool:
    if isinstance(payload, dict):
        if payload.get("type") == "tool_result":
            return True
        for key in ("content", "message", "data"):
            if _is_claude_tool_result(payload.get(key)):
                return True
        return False

    if isinstance(payload, list):
        return any(_is_claude_tool_result(item) for item in payload)

    return False


def _resolve_session_id(
    adapter: str,
    path: Path,
    source_session_id: Optional[str],
    explicit_session_id: Optional[str] = None,
) -> str:
    if explicit_session_id:
        return explicit_session_id

    if source_session_id and source_session_id.strip():
        return f"{adapter}-{source_session_id.strip()}"

    base_name = path.name
    stem = base_name.removesuffix(".jsonl")
    return f"{adapter}-{stem}"


def _first_user_prompt(messages: List[Message]) -> Optional[str]:
    for message in messages:
        if message.role != "user":
            continue
        text = str(message.content or "").strip()
        if text:
            return text
    return None


def _index_eligibility(messages: List[Message]) -> tuple[bool, Optional[str], Optional[str]]:
    user_messages = [
        str(message.content or "").strip() for message in messages if message.role == "user"
    ]
    user_messages = [message for message in user_messages if message]
    if not user_messages:
        return True, None, None

    category = _low_signal_prompt_category(user_messages[0])
    if category is None:
        return True, None, None

    # Keep indexing when the conversation quickly moves past a trivial opener.
    for followup in user_messages[1:]:
        normalized_followup = _normalize_prompt(followup)
        if len(normalized_followup) >= 24 and _low_signal_prompt_category(followup) is None:
            return True, None, None

    return False, category, f"low-signal first user prompt ({category})"


def _low_signal_prompt_category(prompt: str) -> Optional[str]:
    tokens = _normalized_prompt_tokens(prompt)
    if not tokens:
        return "empty"

    polite_tokens = _strip_edge_tokens(tokens, _EDGE_POLITENESS_TOKENS)
    if polite_tokens in _GREETING_PROMPTS:
        return "greeting"

    test_tokens = _strip_test_prefix(polite_tokens)
    if test_tokens in _TEST_PROMPTS:
        return "test"

    if polite_tokens in _STORY_PROMPTS:
        return "story"

    model_tokens = _strip_trailing_tokens(polite_tokens, ("today", "right", "now", "currently"))
    if model_tokens in _MODEL_CHECK_PROMPTS:
        return "model-check"

    if polite_tokens in _IDENTITY_CHECK_PROMPTS:
        return "identity-check"

    return None


def _normalize_prompt(prompt: str) -> str:
    normalized = re.sub(r"[^a-z0-9\s]", " ", prompt.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _normalized_prompt_tokens(prompt: str) -> tuple[str, ...]:
    normalized = _normalize_prompt(prompt)
    if not normalized:
        return ()
    return tuple(normalized.split())


def _strip_edge_tokens(tokens: tuple[str, ...], removable: set[str]) -> tuple[str, ...]:
    start = 0
    end = len(tokens)
    while start < end and tokens[start] in removable:
        start += 1
    while end > start and tokens[end - 1] in removable:
        end -= 1
    return tokens[start:end]


def _strip_test_prefix(tokens: tuple[str, ...]) -> tuple[str, ...]:
    for prefix in _TEST_PREFIXES:
        if len(tokens) > len(prefix) and tokens[: len(prefix)] == prefix:
            return tokens[len(prefix) :]
    return tokens


def _strip_trailing_tokens(
    tokens: tuple[str, ...], trailing_words: tuple[str, ...]
) -> tuple[str, ...]:
    trimmed = tokens
    while trimmed and trimmed[-1] in trailing_words:
        trimmed = trimmed[:-1]
    return trimmed
