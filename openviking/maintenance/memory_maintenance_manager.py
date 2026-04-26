# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Persistent dirty-scope state for memory lifecycle maintenance."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext, Role
from openviking.storage.memory_maintenance_control import (
    MEMORY_MAINTENANCE_STORAGE_BAK_URI,
    MEMORY_MAINTENANCE_STORAGE_TMP_URI,
    MEMORY_MAINTENANCE_STORAGE_URI,
)
from openviking_cli.exceptions import NotFoundError
from openviking_cli.session.user_id import UserIdentifier
from openviking_cli.utils.logger import get_logger

logger = get_logger(__name__)

MAX_DIRTY_URIS_PER_SCOPE = 200


class MemoryMaintenanceScope(BaseModel):
    """Persistent maintenance state for one dirty memory scope."""

    model_config = ConfigDict(extra="ignore")

    scope_uri: str
    account_id: str = "default"
    user_id: str = "default"
    agent_id: str = "default"
    dirty_count: int = 0
    dirty_uris: List[str] = Field(default_factory=list)
    first_dirty_at: Optional[str] = None
    updated_at: Optional[str] = None
    last_archive_uri: str = ""
    last_run_at: Optional[str] = None
    last_audit_uri: str = ""
    retry_count: int = 0
    last_error: str = ""
    is_active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryMaintenanceScope":
        return cls(**data)


def extract_changed_memory_uris(memory_diff: Dict[str, Any]) -> List[str]:
    """Extract changed memory URIs from an archive memory_diff.json payload."""
    operations = memory_diff.get("operations", {})
    if not isinstance(operations, dict):
        return []

    uris: List[str] = []
    for key in ("adds", "updates", "deletes"):
        entries = operations.get(key, [])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            uri = entry.get("uri")
            if isinstance(uri, str) and "/memories/" in uri and "/_archive/" not in uri:
                uris.append(uri)
    return sorted(set(uris))


def memory_scope_for_uri(uri: str) -> str:
    """Return the smallest stable maintenance scope for a memory URI.

    Examples:
      viking://user/u/memories/preferences/editor.md
      -> viking://user/u/memories/preferences/

      viking://user/u/memories/profile.md
      -> viking://user/u/memories/
    """
    if "/memories/" not in uri:
        return ""

    prefix, suffix = uri.split("/memories/", 1)
    parts = [p for p in suffix.split("/") if p]
    if not parts:
        return f"{prefix}/memories/"

    first = parts[0]
    if first.endswith(".md"):
        return f"{prefix}/memories/"
    return f"{prefix}/memories/{first}/"


def dirty_scopes_from_memory_diff(memory_diff: Dict[str, Any]) -> Dict[str, List[str]]:
    """Group changed memory URIs by maintenance scope."""
    grouped: Dict[str, List[str]] = {}
    for uri in extract_changed_memory_uris(memory_diff):
        scope_uri = memory_scope_for_uri(uri)
        if not scope_uri:
            continue
        grouped.setdefault(scope_uri, []).append(uri)
    return grouped


class MemoryMaintenanceManager:
    """Persistent manager for dirty memory maintenance scopes."""

    STORAGE_URI = MEMORY_MAINTENANCE_STORAGE_URI
    STORAGE_BAK_URI = MEMORY_MAINTENANCE_STORAGE_BAK_URI
    STORAGE_TMP_URI = MEMORY_MAINTENANCE_STORAGE_TMP_URI

    def __init__(self, viking_fs: Optional[Any] = None):
        self._viking_fs = viking_fs
        self._scopes: Dict[str, MemoryMaintenanceScope] = {}
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._load()
            self._initialized = True
            logger.info(
                "[MemoryMaintenanceManager] Initialized with %d scopes",
                len(self._scopes),
            )

    async def record_memory_diff(
        self,
        memory_diff: Dict[str, Any],
        ctx: RequestContext,
    ) -> List[MemoryMaintenanceScope]:
        """Mark scopes dirty from a memory_diff.json payload."""
        await self.initialize()
        archive_uri = str(memory_diff.get("archive_uri", "") or "")
        grouped = dirty_scopes_from_memory_diff(memory_diff)
        if not grouped:
            return []

        now = _now_iso()
        changed: List[MemoryMaintenanceScope] = []
        async with self._write_lock:
            for scope_uri, uris in grouped.items():
                scope = self._scopes.get(scope_uri)
                if scope is None:
                    scope = MemoryMaintenanceScope(
                        scope_uri=scope_uri,
                        account_id=getattr(ctx, "account_id", "default") or "default",
                        user_id=getattr(getattr(ctx, "user", None), "user_id", "default")
                        or "default",
                        agent_id=getattr(
                            getattr(ctx, "user", None),
                            "agent_id",
                            "default",
                        )
                        or "default",
                        first_dirty_at=now,
                    )
                    self._scopes[scope_uri] = scope

                scope.updated_at = now
                scope.last_archive_uri = archive_uri
                scope.is_active = True
                scope.last_error = ""
                merged = list(dict.fromkeys([*scope.dirty_uris, *uris]))
                scope.dirty_uris = merged[-MAX_DIRTY_URIS_PER_SCOPE:]
                scope.dirty_count = len(scope.dirty_uris)
                changed.append(scope.model_copy(deep=True))

            await self._save_unlocked()

        return changed

    async def list_scopes(
        self,
        *,
        active_only: bool = True,
        account_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[MemoryMaintenanceScope]:
        await self.initialize()
        scopes = list(self._scopes.values())
        if active_only:
            scopes = [s for s in scopes if s.is_active and s.dirty_count > 0]
        if account_id is not None:
            scopes = [s for s in scopes if s.account_id == account_id]
        if user_id is not None:
            scopes = [s for s in scopes if s.user_id == user_id]
        scopes.sort(key=lambda s: s.updated_at or "", reverse=True)
        return [s.model_copy(deep=True) for s in scopes[: max(0, min(limit, 500))]]

    async def get_scope(self, scope_uri: str) -> Optional[MemoryMaintenanceScope]:
        await self.initialize()
        scope = self._scopes.get(scope_uri)
        return scope.model_copy(deep=True) if scope else None

    async def mark_run_complete(
        self,
        scope_uri: str,
        *,
        audit_uri: str = "",
        dry_run: bool = False,
    ) -> Optional[MemoryMaintenanceScope]:
        await self.initialize()
        async with self._write_lock:
            scope = self._scopes.get(scope_uri)
            if scope is None:
                scope = MemoryMaintenanceScope(scope_uri=scope_uri)
                self._scopes[scope_uri] = scope

            scope.last_run_at = _now_iso()
            scope.last_audit_uri = audit_uri
            scope.retry_count = 0
            scope.last_error = ""
            if not dry_run:
                scope.dirty_count = 0
                scope.dirty_uris = []
                scope.is_active = False
            await self._save_unlocked()
            result = scope.model_copy(deep=True)

        return result

    async def mark_run_failed(self, scope_uri: str, error: str) -> MemoryMaintenanceScope:
        await self.initialize()
        async with self._write_lock:
            scope = self._scopes.get(scope_uri)
            if scope is None:
                scope = MemoryMaintenanceScope(scope_uri=scope_uri)
                self._scopes[scope_uri] = scope
            scope.retry_count += 1
            scope.last_error = error[:500]
            scope.updated_at = _now_iso()
            await self._save_unlocked()
            result = scope.model_copy(deep=True)

        return result

    async def _load(self) -> None:
        if not self._viking_fs:
            return

        data = None
        for uri in (self.STORAGE_URI, self.STORAGE_BAK_URI):
            try:
                content = await self._viking_fs.read_file(uri, ctx=_root_ctx())
                if content and content.strip():
                    data = json.loads(content)
                    break
            except NotFoundError:
                continue
            except Exception as exc:
                logger.warning(
                    "[MemoryMaintenanceManager] Failed to read %s: %s",
                    uri,
                    exc,
                )

        if not isinstance(data, dict):
            return

        self._replace_scopes_from_payload(data)

    def _replace_scopes_from_payload(self, data: Dict[str, Any]) -> None:
        scopes: Dict[str, MemoryMaintenanceScope] = {}
        for raw in data.get("scopes", []):
            if not isinstance(raw, dict):
                continue
            try:
                scope = MemoryMaintenanceScope.from_dict(raw)
                scopes[scope.scope_uri] = scope
            except Exception as exc:
                logger.warning(
                    "[MemoryMaintenanceManager] Failed to load scope state: %s",
                    exc,
                )
        self._scopes = scopes

    async def _save(self) -> None:
        async with self._write_lock:
            await self._save_unlocked()

    async def _save_unlocked(self) -> None:
        if not self._viking_fs:
            return

        payload = {
            "updated_at": _now_iso(),
            "scopes": [scope.to_dict() for scope in self._scopes.values()],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        ctx = _root_ctx()

        supports_atomic = all(
            hasattr(self._viking_fs, name)
            for name in ("write_file", "exists", "mv", "rm")
        )
        if not supports_atomic:
            await self._viking_fs.write_file(self.STORAGE_URI, content, ctx=ctx)
            return

        await self._viking_fs.write_file(self.STORAGE_TMP_URI, content, ctx=ctx)
        try:
            await self._viking_fs.rm(self.STORAGE_BAK_URI, ctx=ctx)
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning(
                    "[MemoryMaintenanceManager] Failed to remove old backup "
                    "backup=%s: %s",
                    self.STORAGE_BAK_URI,
                    exc,
                )
        try:
            await self._viking_fs.mv(self.STORAGE_URI, self.STORAGE_BAK_URI, ctx=ctx)
        except Exception as exc:
            if not is_not_found_error(exc):
                logger.warning(
                    "[MemoryMaintenanceManager] Failed to rotate maintenance backup: %s",
                    exc,
                )
        try:
            await self._viking_fs.mv(self.STORAGE_TMP_URI, self.STORAGE_URI, ctx=ctx)
        except Exception as exc:
            logger.error(
                "[MemoryMaintenanceManager] Failed to promote maintenance state "
                "tmp=%s storage=%s backup=%s: %s",
                self.STORAGE_TMP_URI,
                self.STORAGE_URI,
                self.STORAGE_BAK_URI,
                exc,
            )
            try:
                if await self._viking_fs.exists(self.STORAGE_BAK_URI, ctx=ctx):
                    await self._viking_fs.mv(
                        self.STORAGE_BAK_URI,
                        self.STORAGE_URI,
                        ctx=ctx,
                    )
                    await self._reload_storage_state(ctx)
            except Exception as rollback_exc:
                logger.error(
                    "[MemoryMaintenanceManager] Failed to restore backup "
                    "backup=%s storage=%s: %s",
                    self.STORAGE_BAK_URI,
                    self.STORAGE_URI,
                    rollback_exc,
                )

    async def _reload_storage_state(self, ctx: RequestContext) -> None:
        try:
            content = await self._viking_fs.read_file(self.STORAGE_URI, ctx=ctx)
            data = json.loads(content) if content and content.strip() else {}
            if isinstance(data, dict):
                self._replace_scopes_from_payload(data)
        except Exception as exc:
            logger.error(
                "[MemoryMaintenanceManager] Failed to reload restored state "
                "storage=%s: %s",
                self.STORAGE_URI,
                exc,
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _root_ctx() -> RequestContext:
    return RequestContext(user=UserIdentifier.the_default_user(), role=Role.ROOT)
