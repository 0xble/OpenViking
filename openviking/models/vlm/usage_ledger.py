# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persistent best-effort VLM usage ledger.

The in-memory token tracker is useful for live process diagnostics, but it is
lost on restart. This module records a small JSONL event per VLM call so
background systems can audit recent usage without depending on provider billing
exports.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()
_DEFAULT_PATH = Path.home() / ".openviking" / "usage" / "vlm_calls.jsonl"


def _ledger_path() -> Path:
    configured = os.environ.get("OPENVIKING_VLM_USAGE_LEDGER")
    if configured:
        return Path(configured).expanduser()
    return _DEFAULT_PATH


def record_vlm_call(
    *,
    provider: str,
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    duration_seconds: float,
    account_id: str | None,
    operation: str | None,
    stage: str | None,
) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": str(provider),
        "model_name": str(model_name),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens) + int(completion_tokens),
        "duration_seconds": float(duration_seconds),
        "account_id": account_id,
        "operation": operation,
        "stage": stage,
    }
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    with _LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def read_vlm_usage_summary(*, now: datetime | None = None) -> dict[str, Any]:
    path = _ledger_path()
    if not path.exists():
        return {"available": False, "ledger_path": str(path), "totals": {}}

    current = now or datetime.now(timezone.utc)
    one_day_ago = current - timedelta(hours=24)
    totals: dict[str, Any] = {
        "all_time": _empty_bucket(),
        "last_24h": _empty_bucket(),
        "by_operation": {},
    }

    for event in _iter_events(path):
        occurred_at = _parse_timestamp(event.get("timestamp"))
        _add_event(totals["all_time"], event)
        if occurred_at is not None and occurred_at >= one_day_ago:
            _add_event(totals["last_24h"], event)
            operation = _normalized_key(event.get("operation"), "unknown")
            bucket = totals["by_operation"].setdefault(operation, _empty_bucket())
            _add_event(bucket, event)

    return {"available": True, "ledger_path": str(path), "totals": _plain(totals)}


def _iter_events(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def _empty_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "by_model": defaultdict(_empty_model_bucket),
    }


def _empty_model_bucket() -> dict[str, int]:
    return {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def _add_event(bucket: dict[str, Any], event: dict[str, Any]) -> None:
    prompt_tokens = _int(event.get("prompt_tokens"))
    completion_tokens = _int(event.get("completion_tokens"))
    total_tokens = _int(event.get("total_tokens")) or prompt_tokens + completion_tokens
    bucket["calls"] += 1
    bucket["prompt_tokens"] += prompt_tokens
    bucket["completion_tokens"] += completion_tokens
    bucket["total_tokens"] += total_tokens

    model = _normalized_key(event.get("model_name"), "unknown")
    provider = _normalized_key(event.get("provider"), "unknown")
    key = f"{provider}/{model}"
    model_bucket = bucket["by_model"][key]
    model_bucket["calls"] += 1
    model_bucket["prompt_tokens"] += prompt_tokens
    model_bucket["completion_tokens"] += completion_tokens
    model_bucket["total_tokens"] += total_tokens


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalized_key(value: Any, default: str) -> str:
    normalized = str(value or "").strip()
    return normalized or default


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _plain(value: Any) -> Any:
    if isinstance(value, defaultdict):
        value = dict(value)
    if isinstance(value, dict):
        return {str(key): _plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain(child) for child in value]
    return value
