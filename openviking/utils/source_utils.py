# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Utilities for canonical source classification."""

from __future__ import annotations

from typing import Optional

_SOURCE_ALIASES = {
    "session": "sessions",
    "sessions": "sessions",
    "skill": "skill",
    "skills": "skill",
    "memory": "memory",
    "memories": "memory",
    "resource": "resource",
    "resources": "resource",
}


def normalize_source_name(source: Optional[str]) -> str:
    """Normalize source labels to a stable canonical value."""
    if not source:
        return ""

    normalized = source.strip().lower().replace("-", "_").replace(" ", "_")
    return _SOURCE_ALIASES.get(normalized, normalized)


def infer_source(uri: str, context_type: Optional[str] = None) -> str:
    """Infer a canonical source classification from URI and context type."""
    normalized_context_type = (context_type or "").strip().lower()
    raw_uri = (uri or "").strip()

    if not raw_uri:
        if normalized_context_type == "skill":
            return "skill"
        if normalized_context_type == "memory":
            return "memory"
        return "resource"

    suffix = raw_uri[len("viking://") :] if raw_uri.startswith("viking://") else raw_uri
    parts = [part for part in suffix.strip("/").split("/") if part]

    if not parts:
        return "resource"

    if parts[0] == "session":
        return "sessions"

    if parts[0] == "agent":
        if len(parts) > 1 and parts[1] == "skills":
            return "skill"
        if "memories" in parts:
            return "memory"
        return "agent"

    if parts[0] == "user":
        if "memories" in parts:
            return "memory"
        return "user"

    if parts[0] == "resources" and len(parts) > 2 and parts[1] == "sources":
        return normalize_source_name(parts[2]) or "resource"

    if "memories" in parts or normalized_context_type == "memory":
        return "memory"

    if normalized_context_type == "skill":
        return "skill"

    return "resource"
