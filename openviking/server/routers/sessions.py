# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Sessions endpoints for OpenViking HTTP Server."""

import logging
import re
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Body, Depends, Path, Query, Request
from pydantic import BaseModel, Field, model_validator

from openviking.message.part import TextPart, part_from_dict
from openviking.server.auth import get_request_context
from openviking.server.dependencies import get_service
from openviking.server.identity import AuthMode, RequestContext
from openviking.server.models import ErrorInfo, Response
from openviking.server.responses import error_response
from openviking_cli.exceptions import NotFoundError

_BARE_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_SESSION_ID_PROVIDER_PREFIXES = ("claude-", "codex-", "openclaw-")

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])
logger = logging.getLogger(__name__)


class TextPartRequest(BaseModel):
    """Text part request model."""

    type: Literal["text"] = "text"
    text: str


class ContextPartRequest(BaseModel):
    """Context part request model."""

    type: Literal["context"] = "context"
    uri: str = ""
    context_type: Literal["memory", "resource", "skill"] = "memory"
    abstract: str = ""


class ToolPartRequest(BaseModel):
    """Tool part request model."""

    type: Literal["tool"] = "tool"
    tool_id: str = ""
    tool_name: str = ""
    tool_uri: str = ""
    skill_uri: str = ""
    tool_input: Optional[Dict[str, Any]] = None
    tool_output: str = ""
    tool_status: str = "pending"


PartRequest = TextPartRequest | ContextPartRequest | ToolPartRequest


class AddMessageRequest(BaseModel):
    """Request model for adding a message.

    Supports two modes:
    1. Simple mode: provide `content` string (backward compatible)
    2. Parts mode: provide `parts` array for full Part support

    If both are provided, `parts` takes precedence.
    """

    role: str
    role_id: Optional[str] = None
    content: Optional[str] = None
    parts: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None

    @model_validator(mode="after")
    def validate_content_or_parts(self) -> "AddMessageRequest":
        if self.content is None and self.parts is None:
            raise ValueError("Either 'content' or 'parts' must be provided")
        return self


class UsedRequest(BaseModel):
    """Request model for recording usage."""

    contexts: Optional[List[str]] = None
    skill: Optional[Dict[str, Any]] = None


class CreateSessionRequest(BaseModel):
    """Request model for creating a session."""

    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class PatchSessionMetadataRequest(BaseModel):
    """Request model for patching opaque session metadata."""

    metadata: Dict[str, Any]


def _to_jsonable(value: Any) -> Any:
    """Convert internal objects (e.g. Context) into JSON-serializable values."""
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _request_auth_mode(request: Request) -> AuthMode:
    config = getattr(request.app.state, "config", None)
    if config is not None and hasattr(config, "get_effective_auth_mode"):
        return config.get_effective_auth_mode()
    return AuthMode.API_KEY


def _tolerate_bare_session_id(request: Request) -> bool:
    config = getattr(request.app.state, "config", None)
    return bool(config and getattr(config, "tolerate_bare_session_id", False))


def _resolve_message_role_id(
    http_request: Request,
    request: AddMessageRequest,
    ctx: RequestContext,
) -> Optional[str]:
    if request.role not in {"user", "assistant"}:
        return request.role_id

    role_id = request.role_id
    if not role_id:
        role_id = ctx.user.user_id if request.role == "user" else ctx.user.agent_id

    return role_id


@router.post("")
async def create_session(
    request: Optional[CreateSessionRequest] = None,
    _ctx: RequestContext = Depends(get_request_context),
):
    """Create a new session.

    If session_id is provided, creates a session with the given ID.
    If session_id is None, creates a new session with auto-generated ID.
    """
    service = get_service()
    await service.initialize_user_directories(_ctx)
    await service.initialize_agent_directories(_ctx)
    session_id = request.session_id if request else None
    metadata = request.metadata if request else None
    session = await service.sessions.create(_ctx, session_id, metadata=metadata)
    stored_metadata = await session.get_metadata()
    result = {
        "session_id": session.session_id,
        "user": session.user.to_dict(),
    }
    if stored_metadata is not None:
        result["metadata"] = stored_metadata
    return Response(
        status="ok",
        result=result,
    )


@router.get("")
async def list_sessions(
    since: Optional[str] = Query(
        None, description="Only include sessions updated on or after this time"
    ),
    until: Optional[str] = Query(
        None, description="Only include sessions updated on or before this time"
    ),
    _ctx: RequestContext = Depends(get_request_context),
):
    """List all sessions."""
    service = get_service()
    result = await service.sessions.sessions(_ctx, since=since, until=until)
    return Response(status="ok", result=result)


def _bare_session_id_candidates(session_id: str) -> List[str]:
    """Return prefixed candidates for a bare UUID, in lookup order. Empty if not bare.

    The provider-prefix tuple gates which inputs skip fallback (already-prefixed),
    while the returned list omits openclaw because openclaw IDs need a scope segment
    (`openclaw-<scope>-<uuid>`) that cannot be inferred from a bare UUID alone.
    """
    if session_id.startswith(_SESSION_ID_PROVIDER_PREFIXES):
        return []
    if not _BARE_UUID_RE.match(session_id):
        return []
    return [f"claude-{session_id}", f"codex-{session_id}"]


@router.get("/{session_id}")
async def get_session(
    http_request: Request,
    session_id: str = Path(..., description="Session ID"),
    auto_create: bool = Query(False, description="Create the session if it does not exist"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get session details."""
    service = get_service()
    try:
        session = await service.sessions.get(session_id, _ctx, auto_create=auto_create)
    except NotFoundError:
        # Skip fallback when auto_create=True: a bare UUID would otherwise create a
        # phantom session under the wrong-provider prefix.
        if auto_create or not _tolerate_bare_session_id(http_request):
            raise
        hits = []
        for candidate in _bare_session_id_candidates(session_id):
            try:
                hits.append(await service.sessions.get(candidate, _ctx, auto_create=False))
            except NotFoundError:
                continue
        if not hits:
            raise
        if len(hits) > 1:
            logger.warning(
                "Bare session id %r matched multiple providers %s; returning first.",
                session_id,
                [s.session_id for s in hits],
            )
    session = hits[0]
    result = session.meta.to_dict()
    result["user"] = session.user.to_dict()
    metadata = await session.get_metadata()
    if metadata is not None:
        result["metadata"] = metadata
    result["pending_tokens"] = int(session.meta.pending_tokens or 0)
    return Response(status="ok", result=result)


@router.patch("/{session_id}/metadata")
async def patch_session_metadata(
    request: PatchSessionMetadataRequest,
    session_id: str = Path(..., description="Session ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Patch opaque session metadata."""
    service = get_service()
    metadata = await service.sessions.patch_metadata(session_id, request.metadata, _ctx)
    return Response(status="ok", result={"session_id": session_id, "metadata": metadata})


@router.get("/{session_id}/context")
async def get_session_context(
    session_id: str = Path(..., description="Session ID"),
    token_budget: int = Query(128_000, description="Token budget for session context"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get assembled session context."""
    if token_budget < 0:
        return error_response(
            "INVALID_ARGUMENT",
            "token_budget must be greater than or equal to 0",
            details={"field": "token_budget", "value": token_budget},
        )

    service = get_service()
    session = service.sessions.session(_ctx, session_id)
    await session.load()
    result = await session.get_session_context(token_budget=token_budget)
    return Response(status="ok", result=_to_jsonable(result))


@router.get("/{session_id}/archives/{archive_id}")
async def get_session_archive(
    session_id: str = Path(..., description="Session ID"),
    archive_id: str = Path(..., description="Archive ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Get one completed archive for a session."""
    from openviking_cli.exceptions import NotFoundError

    service = get_service()
    session = service.sessions.session(_ctx, session_id)
    await session.load()
    try:
        result = await session.get_session_archive(archive_id)
    except NotFoundError:
        return Response(
            status="error",
            error=ErrorInfo(code="NOT_FOUND", message=f"Archive {archive_id} not found"),
        )
    return Response(status="ok", result=_to_jsonable(result))


@router.delete("/{session_id}")
async def delete_session(
    session_id: str = Path(..., description="Session ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Delete a session."""
    service = get_service()
    await service.sessions.delete(session_id, _ctx)
    return Response(status="ok", result={"session_id": session_id})


class CommitRequest(BaseModel):
    """Commit request body.

    WM v2: ``keep_recent_count`` allows the plugin to retain a tail of recent
    messages in the live session after commit so the next turn still has
    immediate context. Default 0 preserves the pre-v2 "archive everything"
    behavior.
    """

    keep_recent_count: int = Field(
        default=0,
        ge=0,
        le=10_000,
        description=(
            "Number of most-recent messages to keep live after commit. "
            "Plugin's afterTurn path typically passes its configured value "
            "(default 10); compact path passes 0 to archive everything."
        ),
    )


@router.post("/{session_id}/commit")
async def commit_session(
    session_id: str = Path(..., description="Session ID"),
    body: CommitRequest = Body(default_factory=CommitRequest),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Commit a session (archive and extract memories).

    Archive (Phase 1) completes before returning.  Memory extraction
    (Phase 2) runs in the background.  A ``task_id`` is returned for
    polling progress via ``GET /tasks/{task_id}``.
    """
    service = get_service()
    result = await service.sessions.commit_async(
        session_id, _ctx, keep_recent_count=body.keep_recent_count
    )
    return Response(status="ok", result=result).model_dump(exclude_none=True)


@router.post("/{session_id}/extract")
async def extract_session(
    session_id: str = Path(..., description="Session ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Extract memories from a session."""
    service = get_service()
    result = await service.sessions.extract(session_id, _ctx)
    return Response(status="ok", result=_to_jsonable(result))


@router.post("/{session_id}/messages")
async def add_message(
    request: AddMessageRequest,
    http_request: Request,
    session_id: str = Path(..., description="Session ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Add a message to a session.

    Supports two modes:
    1. Simple mode: provide `content` string (backward compatible)
       Example: {"role": "user", "content": "Hello"}

    2. Parts mode: provide `parts` array for full Part support
       Example: {"role": "assistant", "parts": [
           {"type": "text", "text": "Here's the answer"},
           {"type": "context", "uri": "viking://resources/doc.md", "abstract": "..."}
       ]}

    If both `content` and `parts` are provided, `parts` takes precedence.
    Missing sessions are auto-created on first add.
    """
    service = get_service()
    session = await service.sessions.get(session_id, _ctx, auto_create=True)
    role_id = _resolve_message_role_id(http_request, request, _ctx)

    if request.parts is not None:
        parts = [part_from_dict(p) for p in request.parts]
    else:
        parts = [TextPart(text=request.content or "")]

    # created_at 直接传递给 session (ISO string)
    session.add_message(
        request.role,
        parts,
        role_id=role_id,
        created_at=request.created_at,
    )
    return Response(
        status="ok",
        result={
            "session_id": session_id,
            "message_count": len(session.messages),
        },
    )


@router.post("/{session_id}/used")
async def record_used(
    request: UsedRequest,
    session_id: str = Path(..., description="Session ID"),
    _ctx: RequestContext = Depends(get_request_context),
):
    """Record actually used contexts and skills in a session."""
    service = get_service()
    session = service.sessions.session(_ctx, session_id)
    await session.load()
    session.used(contexts=request.contexts, skill=request.skill)
    return Response(
        status="ok",
        result={
            "session_id": session_id,
            "contexts_used": session.stats.contexts_used,
            "skills_used": session.stats.skills_used,
        },
    )
