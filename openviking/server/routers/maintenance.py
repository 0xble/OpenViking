# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Maintenance endpoints for OpenViking HTTP Server."""

import asyncio
import hashlib
import json
from contextlib import contextmanager
from typing import Any, List, Optional

from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from openviking.maintenance.memory_consolidator import DEFAULT_CANARY_LIMIT
from openviking.server.auth import require_role
from openviking.server.dependencies import get_service
from openviking.server.identity import RequestContext, Role
from openviking.server.models import ErrorInfo, Response
from openviking.server.responses import response_from_result
from openviking.server.telemetry import run_operation
from openviking.telemetry import OperationTelemetry, bind_telemetry
from openviking_cli.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/maintenance", tags=["maintenance"])
# ---------- Memory consolidation (Phase C + D) ----------

CONSOLIDATE_TASK_TYPE = "memory_consolidation"
MEMORY_MAINTENANCE_TASK_TYPE = "memory_maintenance"


@contextmanager
def _maintenance_operation(operation: str):
    """Bind operation context for maintenance work that can outlive HTTP telemetry."""
    with bind_telemetry(OperationTelemetry(operation=operation, enabled=False)):
        yield


class CanarySpec(BaseModel):
    """One canary entry on the consolidate request.

    top_n is the per-canary sensitivity knob. Set to 1 for strict
    canaries that must remain at position 0 post-consolidation; larger
    values allow the expected URI to live anywhere in top-N.
    """

    query: str
    expected_top_uri: str
    top_n: int = Field(default=DEFAULT_CANARY_LIMIT, ge=1)


class ConsolidateRequest(BaseModel):
    """Request to consolidate memories under a scope URI."""

    uri: str
    dry_run: bool = False
    wait: bool = True
    canaries: Optional[List[CanarySpec]] = None


class MemoryMaintenanceRequest(BaseModel):
    """Request a manual memory maintenance pass."""

    scope: Optional[str] = None
    dry_run: bool = True
    wait: bool = True
    limit: int = Field(default=10, ge=1, le=50)
    canaries: Optional[List[CanarySpec]] = None


class ReindexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str
    mode: str = "vectors_only"
    wait: bool = True
    regenerate: bool | None = None


async def _do_reindex(service, uri: str, regenerate: bool, ctx: RequestContext):
    mode = "semantic_and_vectors" if regenerate else "vectors_only"
    return await service.reindex(uri=uri, mode=mode, wait=True, ctx=ctx)


def _build_consolidator(service, ctx: RequestContext):
    """Construct a MemoryConsolidator wired to the live service."""
    from openviking.maintenance import MemoryConsolidator
    from openviking.session.memory_archiver import MemoryArchiver
    from openviking.session.memory_deduplicator import MemoryDeduplicator
    from openviking.storage import VikingDBManagerProxy

    viking_fs = service.viking_fs
    vikingdb = VikingDBManagerProxy(service.vikingdb_manager, ctx)
    dedup = MemoryDeduplicator(vikingdb)
    archiver = MemoryArchiver(viking_fs=viking_fs, storage=vikingdb)
    return MemoryConsolidator(
        vikingdb=vikingdb,
        viking_fs=viking_fs,
        dedup=dedup,
        archiver=archiver,
        service=service,
    )


def _build_maintenance_manager(service):
    from openviking.maintenance import MemoryMaintenanceManager

    return MemoryMaintenanceManager(viking_fs=service.viking_fs)


@router.post("/reindex")
async def reindex(
    body: ReindexRequest = Body(...),
    ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    service = get_service()
    if callable(getattr(service, "reindex", None)):
        return JSONResponse(status_code=404, content={"detail": "Not Found"})
    regenerate = body.regenerate if body.regenerate is not None else body.mode != "vectors_only"
    mode = "semantic_and_vectors" if regenerate else "vectors_only"
    execution = await run_operation(
        operation="maintenance.reindex",
        telemetry=False,
        fn=lambda: (
            _do_reindex(service, body.uri, regenerate, ctx)
            if body.wait
            else service.reindex(uri=body.uri, mode=mode, wait=False, ctx=ctx)
        ),
    )
    response = response_from_result(execution.result)
    if isinstance(response, dict):
        return Response(status="ok", result=execution.result)
    return response


@router.post("/consolidate")
async def consolidate(
    request: ConsolidateRequest = Body(...),
    _ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    """Consolidate memories under a scope URI.

    Runs a manual memory lifecycle pass: cluster duplicates, LLM-merge,
    archive cold entries, refresh overview. dry_run=true returns the
    plan without writes. wait=false enqueues and returns a task_id for
    polling via the task API. Optional canaries run pre/post and set
    canary_failed=true on hard regression.
    """
    from openviking.service.task_tracker import get_task_tracker
    from openviking.storage.viking_fs import get_viking_fs

    uri = request.uri
    viking_fs = get_viking_fs()

    if not await viking_fs.exists(uri, ctx=_ctx):
        return Response(
            status="error",
            error=ErrorInfo(code="NOT_FOUND", message=f"URI not found: {uri}"),
        )

    service = get_service()
    tracker = get_task_tracker()

    if request.wait:
        if tracker.has_running(
            CONSOLIDATE_TASK_TYPE,
            uri,
            owner_account_id=_ctx.account_id,
            owner_user_id=_ctx.user.user_id,
        ):
            return Response(
                status="error",
                error=ErrorInfo(
                    code="CONFLICT",
                    message=f"URI {uri} already has a consolidation in progress",
                ),
            )
        consolidator = _build_consolidator(service, _ctx)
        with _maintenance_operation("maintenance.consolidate"):
            result = await consolidator.run(
                uri,
                _ctx,
                dry_run=request.dry_run,
                canaries=_canaries_from_request(request.canaries),
            )
        return Response(status="ok", result=_consolidation_payload(result))

    task = tracker.create_if_no_running(
        CONSOLIDATE_TASK_TYPE,
        uri,
        owner_account_id=_ctx.account_id,
        owner_user_id=_ctx.user.user_id,
    )
    if task is None:
        return Response(
            status="error",
            error=ErrorInfo(
                code="CONFLICT",
                message=f"URI {uri} already has a consolidation in progress",
            ),
        )
    asyncio.create_task(
        _background_consolidate_tracked(
            service,
            uri,
            request.dry_run,
            _ctx,
            task.task_id,
            _canaries_from_request(request.canaries),
        )
    )
    return Response(
        status="ok",
        result={
            "uri": uri,
            "status": "accepted",
            "task_id": task.task_id,
            "message": "Consolidation is processing in the background",
            "dry_run": request.dry_run,
        },
    )


@router.get("/consolidate/runs")
async def list_consolidate_runs(
    scope: str,
    limit: int = 20,
    _ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    """List recent consolidation audit records for a scope.

    Audit records live at
    viking://agent/<account>/maintenance/consolidation_runs/<scope_hash>/<iso>.json
    written by MemoryConsolidator._record. Returned in reverse
    chronological order, capped at 100.
    """
    from openviking.maintenance import MemoryConsolidator
    from openviking.storage.viking_fs import get_viking_fs

    viking_fs = get_viking_fs()
    audit_dir = MemoryConsolidator.audit_dir_for(_ctx, scope)

    try:
        entries = await viking_fs.ls(audit_dir, ctx=_ctx)
    except Exception:
        return Response(status="ok", result={"scope": scope, "runs": []})

    # viking_fs.ls returns List[Dict] with a 'uri' key per entry, not bare
    # strings. Extract the URI and filter to .json audit files.
    file_uris = []
    for entry in entries:
        if isinstance(entry, dict):
            uri = entry.get("uri", "")
            is_dir = entry.get("isDir", False)
        else:
            uri = str(entry)
            is_dir = False
        if not uri or is_dir or not uri.endswith(".json"):
            continue
        file_uris.append(uri)

    file_uris.sort(reverse=True)
    capped_limit = min(max(0, limit), 100)
    file_uris = file_uris[:capped_limit]

    runs = []
    for run_uri in file_uris:
        try:
            body_text = await viking_fs.read(run_uri, ctx=_ctx)
            if isinstance(body_text, bytes):
                body_text = body_text.decode("utf-8", errors="replace")
            runs.append({"uri": run_uri, "body": body_text})
        except Exception as e:
            runs.append({"uri": run_uri, "error": str(e)})

    return Response(status="ok", result={"scope": scope, "runs": runs})


async def _background_consolidate_tracked(
    service,
    uri: str,
    dry_run: bool,
    ctx: RequestContext,
    task_id: str,
    canaries=None,
) -> None:
    """Run consolidation in background with task tracking."""
    from openviking.service.task_tracker import get_task_tracker

    tracker = get_task_tracker()
    tracker.start(task_id)
    try:
        consolidator = _build_consolidator(service, ctx)
        with _maintenance_operation("maintenance.consolidate"):
            result = await consolidator.run(uri, ctx, dry_run=dry_run, canaries=canaries)
        tracker.complete(task_id, _consolidation_payload(result))
        logger.info("Background consolidation completed: uri=%s task=%s", uri, task_id)
    except Exception as exc:
        tracker.fail(task_id, str(exc))
        logger.exception("Background consolidation failed: uri=%s task=%s", uri, task_id)


@router.get("/memory/scopes")
async def list_memory_maintenance_scopes(
    active_only: bool = True,
    limit: int = 100,
    _ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    """List persisted dirty memory scopes from memory_diff.json processing."""
    service = get_service()
    manager = _build_maintenance_manager(service)
    scopes = await manager.list_scopes(
        active_only=active_only,
        account_id=_ctx.account_id,
        user_id=_ctx.user.user_id,
        limit=limit,
    )
    return Response(
        status="ok",
        result={
            "scopes": [scope.to_dict() for scope in scopes],
            "active_only": active_only,
        },
    )


@router.post("/memory/run")
async def run_memory_maintenance(
    request: MemoryMaintenanceRequest = Body(...),
    _ctx: RequestContext = require_role(Role.ROOT, Role.ADMIN),
):
    """Run manual memory maintenance for one scope or the current dirty scopes."""
    from openviking.service.task_tracker import get_task_tracker

    service = get_service()
    manager = _build_maintenance_manager(service)
    scope_uris = await _resolve_maintenance_scopes(manager, request, _ctx)
    if not scope_uris:
        return Response(
            status="ok",
            result={"status": "noop", "scopes": [], "dry_run": request.dry_run},
        )

    resource_id = _maintenance_resource_id(scope_uris)
    tracker = get_task_tracker()
    if request.wait:
        if tracker.has_running(
            MEMORY_MAINTENANCE_TASK_TYPE,
            resource_id,
            owner_account_id=_ctx.account_id,
            owner_user_id=_ctx.user.user_id,
        ):
            return Response(
                status="error",
                error=ErrorInfo(
                    code="CONFLICT",
                    message="Selected memory scopes already have maintenance in progress",
                ),
            )
        with _maintenance_operation("maintenance.memory.run"):
            runs = await _run_memory_maintenance_scopes(
                service,
                manager,
                scope_uris,
                request.dry_run,
                _ctx,
                _canaries_from_request(request.canaries),
            )
        return Response(
            status="ok",
            result={"status": "completed", "dry_run": request.dry_run, "runs": runs},
        )

    task = tracker.create_if_no_running(
        MEMORY_MAINTENANCE_TASK_TYPE,
        resource_id,
        owner_account_id=_ctx.account_id,
        owner_user_id=_ctx.user.user_id,
    )
    if task is None:
        return Response(
            status="error",
            error=ErrorInfo(
                code="CONFLICT",
                message="Selected memory scopes already have maintenance in progress",
            ),
        )

    asyncio.create_task(
        _background_memory_maintenance_tracked(
            service,
            scope_uris,
            request.dry_run,
            _ctx,
            task.task_id,
            _canaries_from_request(request.canaries),
        )
    )
    return Response(
        status="ok",
        result={
            "status": "accepted",
            "task_id": task.task_id,
            "scopes": scope_uris,
            "dry_run": request.dry_run,
        },
    )


async def _resolve_maintenance_scopes(
    manager,
    request: MemoryMaintenanceRequest,
    ctx: RequestContext,
) -> List[str]:
    if request.scope:
        return [request.scope]
    scopes = await manager.list_scopes(
        active_only=True,
        account_id=ctx.account_id,
        user_id=ctx.user.user_id,
        limit=request.limit,
    )
    return [scope.scope_uri for scope in scopes]


def _maintenance_resource_id(scope_uris: List[str]) -> str:
    encoded = json.dumps(sorted(scope_uris), ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"memory-scopes:{digest}"


async def _run_memory_maintenance_scopes(
    service,
    manager,
    scope_uris: List[str],
    dry_run: bool,
    ctx: RequestContext,
    canaries=None,
) -> List[dict[str, Any]]:
    consolidator = _build_consolidator(service, ctx)
    runs: List[dict[str, Any]] = []
    errors: List[tuple[str, str]] = []
    for scope_uri in scope_uris:
        try:
            scope_state = await manager.get_scope(scope_uri)
            result = await consolidator.run(
                scope_uri,
                ctx,
                dry_run=dry_run,
                canaries=canaries,
                target_uris=scope_state.dirty_uris if scope_state else None,
            )
            if result.partial or result.errors:
                error = "; ".join(result.errors) or "partial maintenance result"
                await manager.mark_run_failed(scope_uri, error)
                runs.append(_consolidation_payload(result))
                errors.append((scope_uri, error))
                continue
            await manager.mark_run_complete(
                scope_uri,
                audit_uri=result.audit_uri,
                dry_run=dry_run,
            )
            runs.append(_consolidation_payload(result))
        except Exception as exc:
            await manager.mark_run_failed(scope_uri, str(exc))
            errors.append((scope_uri, str(exc)))
            continue
    if errors:
        detail = "; ".join(f"{scope_uri}: {message}" for scope_uri, message in errors)
        raise RuntimeError(f"Memory maintenance failed for {len(errors)} scope(s): {detail}")
    return runs


async def _background_memory_maintenance_tracked(
    service,
    scope_uris: List[str],
    dry_run: bool,
    ctx: RequestContext,
    task_id: str,
    canaries=None,
) -> None:
    from openviking.service.task_tracker import get_task_tracker

    tracker = get_task_tracker()
    tracker.start(task_id)
    manager = _build_maintenance_manager(service)
    try:
        with _maintenance_operation("maintenance.memory.run"):
            runs = await _run_memory_maintenance_scopes(
                service,
                manager,
                scope_uris,
                dry_run,
                ctx,
                canaries,
            )
        tracker.complete(
            task_id,
            {"status": "completed", "dry_run": dry_run, "runs": runs},
        )
    except Exception as exc:
        tracker.fail(task_id, str(exc))
        logger.exception("Background memory maintenance failed: task=%s", task_id)


def _consolidation_payload(result) -> dict:
    """Project ConsolidationResult into a JSON-safe dict for HTTP."""
    from dataclasses import asdict

    return asdict(result)


def _canaries_from_request(specs):
    """Translate request CanarySpec entries into Canary domain objects.

    CanarySpec.top_n is already validated (ge=1) by Pydantic at the
    HTTP boundary, so no defensive clamping needed here.
    """
    if not specs:
        return None
    from openviking.maintenance import Canary

    return [
        Canary(
            query=s.query,
            expected_top_uri=s.expected_top_uri,
            top_n=s.top_n,
        )
        for s in specs
    ]
