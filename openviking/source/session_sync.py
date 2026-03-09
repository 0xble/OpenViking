# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Session source discovery helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_SESSION_GLOBS = {
    "claude": "**/*.jsonl",
    "codex": "**/*.jsonl",
    "openclaw": "**/*.jsonl",
}


def discover_session_logs(
    adapter: str,
    path: str,
    glob_pattern: Optional[str] = None,
) -> List[Path]:
    """Discover raw session log files for a configured source."""
    normalized_adapter = adapter.strip().lower()
    pattern = glob_pattern or DEFAULT_SESSION_GLOBS.get(normalized_adapter, "**/*.jsonl")

    root = Path(path).expanduser().resolve()
    if not root.exists():
        return []

    return sorted(
        _iter_files(root, pattern),
        key=lambda file_path: (file_path.stat().st_mtime, str(file_path)),
    )


def _iter_files(root: Path, pattern: str) -> Iterable[Path]:
    for candidate in root.glob(pattern):
        if candidate.is_file():
            yield candidate.resolve()
