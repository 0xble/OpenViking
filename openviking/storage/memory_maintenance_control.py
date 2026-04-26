# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Internal memory-maintenance control URIs."""

MEMORY_MAINTENANCE_STORAGE_URI = "viking://resources/.memory_maintenance_tasks.json"
MEMORY_MAINTENANCE_STORAGE_BAK_URI = "viking://resources/.memory_maintenance_tasks.json.bak"
MEMORY_MAINTENANCE_STORAGE_TMP_URI = "viking://resources/.memory_maintenance_tasks.json.tmp"

MEMORY_MAINTENANCE_CONTROL_URIS = frozenset(
    {
        MEMORY_MAINTENANCE_STORAGE_URI,
        MEMORY_MAINTENANCE_STORAGE_BAK_URI,
        MEMORY_MAINTENANCE_STORAGE_TMP_URI,
    }
)


def is_memory_maintenance_control_uri(uri: str) -> bool:
    """Return True when a URI points at internal memory-maintenance state."""
    if not isinstance(uri, str):
        return False
    return uri.rstrip("/") in MEMORY_MAINTENANCE_CONTROL_URIS
