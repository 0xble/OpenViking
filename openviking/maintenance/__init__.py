# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Maintenance utilities for OpenViking.

Houses bounded lifecycle helpers that operate on persisted state
(memories, resources, vector index) outside the hot request path.
"""

from openviking.maintenance.consolidation_scheduler import (
    DEFAULT_CHECK_INTERVAL_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_SCAN_INTERVAL_SECONDS,
    MemoryConsolidationScheduler,
    SchedulerGates,
    ScopeStatus,
)
from openviking.maintenance.memory_consolidator import (
    Canary,
    CanaryResult,
    ConsolidationResult,
    MemoryConsolidator,
)
from openviking.maintenance.memory_maintenance_manager import (
    MEMORY_MAINTENANCE_STORAGE_URI,
    MemoryMaintenanceManager,
    MemoryMaintenanceScope,
    dirty_scopes_from_memory_diff,
    extract_changed_memory_uris,
    memory_scope_for_uri,
)

__all__ = [
    "Canary",
    "CanaryResult",
    "ConsolidationResult",
    "DEFAULT_CHECK_INTERVAL_SECONDS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_SCAN_INTERVAL_SECONDS",
    "MemoryConsolidationScheduler",
    "MemoryConsolidator",
    "MEMORY_MAINTENANCE_STORAGE_URI",
    "MemoryMaintenanceManager",
    "MemoryMaintenanceScope",
    "SchedulerGates",
    "ScopeStatus",
    "dirty_scopes_from_memory_diff",
    "extract_changed_memory_uris",
    "memory_scope_for_uri",
]
