# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Maintenance endpoints for OpenViking HTTP Server."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

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
from openviking_cli.exceptions import InvalidArgumentError, NotInitializedError

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
        user_id=ctx.user.user_id,
        limit=limit,
    )
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
    requested_scope = request.scope.strip()

    if requested_scope:
        scope = await manager.get_scope(requested_scope)
        scope_entries = [scope] if scope is not None else [_request_scope(requested_scope, ctx)]
    else:
        scope_entries = await manager.list_scopes(
            active_only=True,
            account_id=ctx.account_id,
            user_id=ctx.user.user_id,
            limit=limit,
        )

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
            await manager.mark_run_complete(
                scope_uri,
                audit_uri=result.audit_uri,
                dry_run=request.dry_run,
            )
        except Exception as exc:
            status = "error"
            failed = await manager.mark_run_failed(scope_uri, str(exc))
            runs.append(
                {
                    "scope_uri": scope_uri,
                    "status": "error",
                    "error": failed.last_error,
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
