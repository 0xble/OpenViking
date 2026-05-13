# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Maintenance endpoints for OpenViking HTTP Server."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from openviking.core.namespace import canonical_agent_root, canonical_user_root
from openviking.maintenance import (
    MemoryConsolidator,
    MemoryMaintenanceManager,
    MemoryMaintenanceScope,
)
from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext
from openviking.server.models import Response
from openviking.session.memory_archiver import MemoryArchiver
from openviking.session.memory_deduplicator import MemoryDeduplicator
from openviking_cli.exceptions import (
    InvalidArgumentError,
    NotInitializedError,
    PermissionDeniedError,
)

router = APIRouter(prefix="/api/v1/maintenance", tags=["maintenance"])


class MemoryMaintenanceRunRequest(BaseModel):
    """Request model for dirty-scope memory maintenance."""

    model_config = ConfigDict(extra="forbid")

    scope: str = ""
    dry_run: bool = False
    wait: bool = True
    limit: int = 10


def _require_service_components() -> tuple[Any, Any]:
    service = get_service()
    viking_fs = service.viking_fs
    vikingdb = service.vikingdb_manager
    if viking_fs is None or vikingdb is None:
        raise NotInitializedError("OpenVikingService is not initialized")
    return viking_fs, vikingdb


def _manager() -> MemoryMaintenanceManager:
    viking_fs, _vikingdb = _require_service_components()
    return MemoryMaintenanceManager(viking_fs=viking_fs)


def _consolidator() -> MemoryConsolidator:
    viking_fs, vikingdb = _require_service_components()
    service = get_service()
    return MemoryConsolidator(
        vikingdb=vikingdb,
        viking_fs=viking_fs,
        dedup=MemoryDeduplicator(vikingdb=vikingdb),
        archiver=MemoryArchiver(viking_fs=viking_fs, storage=vikingdb),
        service=service,
    )


@router.get("/memory/scopes")
async def list_memory_maintenance_scopes(
    active_only: bool = True,
    limit: int = 100,
    ctx: RequestContext = Depends(get_request_context),
):
    """List persisted dirty memory scopes for the request tenant."""
    manager = _manager()
    scopes = await manager.list_scopes(
        active_only=active_only,
        account_id=ctx.account_id,
        limit=None,
    )
    scopes = [scope for scope in scopes if _scope_belongs_to_request(scope, ctx)][
        : max(0, min(limit, 500))
    ]
    return Response(
        status="ok",
        result={
            "active_only": active_only,
            "scopes": [scope.to_dict() for scope in scopes],
        },
    )


@router.post("/memory/run")
async def run_memory_maintenance(
    request: MemoryMaintenanceRunRequest,
    ctx: RequestContext = Depends(get_request_context),
):
    """Run memory maintenance for one scope or the current dirty scopes."""
    manager = _manager()
    if not request.wait:
        raise InvalidArgumentError("wait=false is not supported for memory maintenance runs")
    limit = max(1, min(request.limit, 100))
    requested_scope = _normalize_scope_uri(request.scope)

    if requested_scope:
        if not _scope_uri_allowed_for_request(requested_scope, ctx):
            raise PermissionDeniedError(
                "memory maintenance scope does not belong to the request tenant"
            )
        scope = await manager.get_scope(requested_scope)
        if scope is not None and not _scope_belongs_to_request(scope, ctx):
            raise PermissionDeniedError(
                "memory maintenance scope does not belong to the request tenant"
            )
        if scope is None:
            scope = await manager.ensure_scope(_request_scope(requested_scope, ctx))
        scope_entries = [scope]
    else:
        scope_entries = await manager.list_scopes(
            active_only=True,
            account_id=ctx.account_id,
            limit=None,
        )
        scope_entries = [scope for scope in scope_entries if _scope_belongs_to_request(scope, ctx)][
            :limit
        ]

    if not scope_entries:
        return Response(
            status="ok",
            result={
                "status": "completed",
                "dry_run": request.dry_run,
                "scopes": [],
                "runs": [],
            },
        )

    consolidator: Optional[MemoryConsolidator] = None
    runs: list[dict[str, Any]] = []
    processed_scopes: list[str] = []
    status = "completed"

    for scope in scope_entries[:limit]:
        scope_uri = scope.scope_uri
        processed_scopes.append(scope_uri)
        target_uris = scope.dirty_uris or None
        try:
            if consolidator is None:
                consolidator = _consolidator()
            result = await consolidator.run(
                scope_uri,
                ctx,
                dry_run=request.dry_run,
                target_uris=target_uris,
            )
            runs.append(asdict(result))
            if result.partial or result.errors:
                status = "error"
                if not request.dry_run:
                    await manager.mark_run_failed(
                        scope_uri,
                        "; ".join(result.errors) or "maintenance completed partially",
                    )
            else:
                await manager.mark_run_complete(
                    scope_uri,
                    audit_uri=result.audit_uri,
                    dry_run=request.dry_run,
                    processed_uris=target_uris or [],
                )
        except Exception as exc:
            status = "error"
            error = str(exc)
            if not request.dry_run:
                failed = await manager.mark_run_failed(scope_uri, error)
                error = failed.last_error
            runs.append(
                {
                    "scope_uri": scope_uri,
                    "status": "error",
                    "error": error,
                }
            )

    return Response(
        status="ok",
        result={
            "status": status,
            "dry_run": request.dry_run,
            "scopes": processed_scopes,
            "runs": runs,
        },
    )


def _request_scope(scope_uri: str, ctx: RequestContext) -> MemoryMaintenanceScope:
    return MemoryMaintenanceScope(
        scope_uri=scope_uri,
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        agent_id=ctx.user.agent_id,
    )


def _normalize_scope_uri(scope_uri: str) -> str:
    normalized = scope_uri.strip()
    if not normalized:
        return ""
    return normalized.rstrip("/") + "/"


def _scope_belongs_to_request(scope: MemoryMaintenanceScope, ctx: RequestContext) -> bool:
    if scope.account_id != ctx.account_id:
        return False

    scope_uri = _normalize_scope_uri(scope.scope_uri)
    if scope_uri.startswith(canonical_agent_root(ctx).rstrip("/") + "/memories/"):
        if scope.agent_id != ctx.user.agent_id:
            return False
        if ctx.namespace_policy.isolate_agent_scope_by_user:
            return scope.user_id == ctx.user.user_id
        return True

    if scope_uri.startswith(canonical_user_root(ctx).rstrip("/") + "/memories/"):
        if scope.user_id != ctx.user.user_id:
            return False
        if ctx.namespace_policy.isolate_user_scope_by_agent:
            return scope.agent_id == ctx.user.agent_id
        return True

    return False


def _scope_uri_allowed_for_request(scope_uri: str, ctx: RequestContext) -> bool:
    normalized = _normalize_scope_uri(scope_uri)
    allowed_prefixes = (
        canonical_user_root(ctx).rstrip("/") + "/memories/",
        canonical_agent_root(ctx).rstrip("/") + "/memories/",
    )
    return any(normalized.startswith(prefix) for prefix in allowed_prefixes)
