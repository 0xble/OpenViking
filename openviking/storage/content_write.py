# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Coordinator for content write operations."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from openviking.resource.watch_storage import is_watch_task_control_uri
from openviking.server.error_mapping import is_not_found_error
from openviking.server.identity import RequestContext
from openviking.session.memory.utils.content import deserialize_full, serialize_with_metadata
from openviking.storage.memory_maintenance_control import is_memory_maintenance_control_uri
from openviking.storage.transaction import get_lock_manager
from openviking.storage.viking_fs import VikingFS
from openviking.telemetry.request_wait_tracker import get_request_wait_tracker
from openviking_cli.exceptions import (
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
    PermissionDeniedError,
)
from openviking_cli.utils import VikingURI, get_logger

logger = get_logger(__name__)

_DERIVED_FILENAMES = frozenset(
    {".abstract.md", ".overview.md", ".relations.json", ".resource.metadata.json"}
)
_CREATE_ALLOWED_EXTENSIONS = frozenset(
    {".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".py", ".js", ".ts"}
)


class ContentWriteCoordinator:
    """Write file content directly.

    Direct content writes are storage-only: they update the target file and
    return without taking semantic lifecycle locks or enqueueing refresh work.
    Semantic/vector freshness is handled by explicit reindex/refresh flows.
    """

    def __init__(self, viking_fs: VikingFS):
        self._viking_fs = viking_fs

    async def write(
        self,
        *,
        uri: str,
        content: str,
        ctx: RequestContext,
        mode: str = "replace",
        wait: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        del wait, timeout
        normalized_uri = VikingURI.normalize(uri)
        self._validate_mode(mode)
        self._validate_target_uri(normalized_uri)

        if mode == "create":
            return await self._create_and_write(
                uri=normalized_uri,
                content=content,
                ctx=ctx,
            )

        context_type = self._context_type_for_uri(normalized_uri)

        existing_stat = await self._safe_stat(
            normalized_uri,
            ctx=ctx,
            allow_not_found=True,
        )
        if existing_stat.get("isDir"):
            raise InvalidArgumentError(f"write only supports files, got directory: {uri}")

        existed_before = not existing_stat.get("not_found")
        if not existed_before and context_type != "memory":
            raise NotFoundError(uri, "file")
        if not existed_before and mode == "append":
            mode = "replace"

        written_bytes = len(content.encode("utf-8"))
        root_uri = await self._resolve_direct_root_uri(normalized_uri, context_type, ctx=ctx)
        if existed_before:
            await self._write_in_place(normalized_uri, content, mode=mode, ctx=ctx)
        else:
            await self._viking_fs.write_file(normalized_uri, content, ctx=ctx)

        return self._build_direct_write_result(
            uri=normalized_uri,
            root_uri=root_uri,
            context_type=context_type,
            mode=mode,
            created=not existed_before,
            written_bytes=written_bytes,
        )

    def _validate_mode(self, mode: str) -> None:
        if mode not in {"replace", "append", "create"}:
            raise InvalidArgumentError(f"unsupported write mode: {mode}")

    def _validate_target_uri(self, uri: str) -> None:
        name = uri.rstrip("/").split("/")[-1]
        if name in _DERIVED_FILENAMES:
            raise InvalidArgumentError(f"cannot write derived semantic file directly: {uri}")
        if is_watch_task_control_uri(uri):
            raise InvalidArgumentError(f"cannot write watch task control file directly: {uri}")
        if is_memory_maintenance_control_uri(uri):
            raise InvalidArgumentError(
                f"cannot write memory maintenance control file directly: {uri}"
            )

        parsed = VikingURI(uri)
        if parsed.scope not in {"resources", "user", "agent"}:
            raise InvalidArgumentError(f"write is not supported for scope: {parsed.scope}")

    def _is_not_found(self, exc: Exception) -> bool:
        """Check if an exception indicates a not-found error."""
        if isinstance(exc, NotFoundError):
            return True
        return is_not_found_error(exc)

    async def _safe_stat(
        self,
        uri: str,
        *,
        ctx: RequestContext,
        allow_not_found: bool = False,
    ) -> Dict[str, Any]:
        try:
            return await self._viking_fs.stat(uri, ctx=ctx)
        except PermissionDeniedError as exc:
            if allow_not_found:
                raise NotFoundError(uri, "file") from exc
            raise
        except Exception as exc:
            if self._is_not_found(exc):
                if allow_not_found:
                    return {"not_found": True}
                if isinstance(exc, NotFoundError):
                    raise
                raise NotFoundError(uri, "file") from exc
            raise

    def _validate_create_extension(self, uri: str) -> None:
        _, ext = os.path.splitext(uri)
        if ext.lower() not in _CREATE_ALLOWED_EXTENSIONS:
            raise InvalidArgumentError(f"create mode does not allow extension '{ext}': {uri}")

    async def _write_in_place(
        self,
        uri: str,
        content: str,
        *,
        mode: str,
        ctx: RequestContext,
    ) -> None:
        if mode == "replace" and self._context_type_for_uri(uri) == "memory":
            existing_raw = await self._viking_fs.read_file(uri, ctx=ctx)
            _, metadata = deserialize_full(existing_raw)
            if metadata:
                metadata_with_content = metadata.copy()
                metadata_with_content["content"] = content
                content = serialize_with_metadata(metadata_with_content)
            await self._viking_fs.write_file(uri, content, ctx=ctx)
            return

        if mode == "append":
            existing_raw = await self._viking_fs.read_file(uri, ctx=ctx)
            existing_content, metadata = deserialize_full(existing_raw)
            updated_content = existing_content + content
            if metadata:
                metadata_with_content = metadata.copy()
                metadata_with_content["content"] = updated_content
                updated_raw = serialize_with_metadata(metadata_with_content)
            else:
                updated_raw = updated_content
            await self._viking_fs.write_file(uri, updated_raw, ctx=ctx)
            return
        await self._viking_fs.write_file(uri, content, ctx=ctx)

    async def _resolve_root_uri(
        self,
        uri: str,
        *,
        ctx: RequestContext,
        _allow_not_found: bool = False,
    ) -> str:
        parsed = VikingURI(uri)
        parts = [part for part in parsed.full_path.split("/") if part]
        if not parts:
            raise InvalidArgumentError(f"invalid write uri: {uri}")

        root_uri = uri
        if parts[0] == "resources":
            if len(parts) >= 2:
                root_uri = VikingURI.build("resources", parts[1])
        elif parts[0] == "user":
            try:
                memories_idx = parts.index("memories")
            except ValueError as exc:
                raise InvalidArgumentError(
                    f"write only supports memory files under user scope: {uri}"
                ) from exc
            if len(parts) <= memories_idx + 1:
                raise InvalidArgumentError(
                    f"memory write target must be inside a memory type directory: {uri}"
                )
            root_uri = VikingURI.build(*parts[: memories_idx + 2])
        elif parts[0] == "agent":
            if len(parts) >= 3 and parts[1] == "skills":
                root_uri = VikingURI.build(*parts[:3])
            else:
                try:
                    memories_idx = parts.index("memories")
                except ValueError as exc:
                    raise InvalidArgumentError(
                        f"write only supports memory or skill files under agent scope: {uri}"
                    ) from exc
                if len(parts) <= memories_idx + 1:
                    raise InvalidArgumentError(
                        f"memory write target must be inside a memory type directory: {uri}"
                    )
                root_uri = VikingURI.build(*parts[: memories_idx + 2])

        stat = await self._safe_stat(
            root_uri,
            ctx=ctx,
            allow_not_found=_allow_not_found,
        )
        if stat.get("not_found") or not stat.get("isDir"):
            parent = VikingURI(uri).parent
            if parent is None:
                raise InvalidArgumentError(f"could not resolve write root for {uri}")
            root_uri = parent.uri
        return root_uri

    def _context_type_for_uri(self, uri: str) -> str:
        if "/memories/" in uri:
            return "memory"
        if "/skills/" in uri or uri.startswith("viking://agent/skills/"):
            return "skill"
        return "resource"

    async def _resolve_memory_root_uri(self, uri: str) -> str:
        parsed = VikingURI(uri)
        parts = [part for part in parsed.full_path.split("/") if part]
        try:
            memories_idx = parts.index("memories")
        except ValueError as exc:
            raise InvalidArgumentError(
                f"memory uri must contain a 'memories' segment: {uri}"
            ) from exc
        tail = parts[memories_idx + 1 :]
        if not tail:
            raise InvalidArgumentError(f"memory uri must include a bucket or singleton file: {uri}")
        # Singleton memory file (e.g. profile.md) lives directly under memories/;
        # its root is the memories directory itself. Bucket subdirectories
        # (preferences/, entities/, etc.) use the bucket dir as the root.
        if len(tail) == 1:
            return VikingURI.build(*parts[: memories_idx + 1])
        return VikingURI.build(*parts[: memories_idx + 2])

    async def _resolve_direct_root_uri(
        self,
        uri: str,
        context_type: str,
        *,
        ctx: RequestContext,
    ) -> str:
        if context_type == "memory":
            return await self._resolve_memory_root_uri(uri)
        return await self._resolve_root_uri(uri, ctx=ctx, _allow_not_found=True)

    def _build_direct_write_result(
        self,
        *,
        uri: str,
        root_uri: str,
        context_type: str,
        mode: str,
        created: bool,
        written_bytes: int,
    ) -> Dict[str, Any]:
        return {
            "uri": uri,
            "root_uri": root_uri,
            "context_type": context_type,
            "mode": mode,
            "created": created,
            "written_bytes": written_bytes,
            "content_updated": True,
            "semantic_status": "not_refreshed",
            "vector_status": "not_refreshed",
            "semantic_updated": False,
            "vector_updated": False,
            "queue_status": None,
        }

    async def _create_and_write(
        self,
        *,
        uri: str,
        content: str,
        ctx: RequestContext,
    ) -> Dict[str, Any]:
        self._validate_create_extension(uri)

        stat = await self._safe_stat(uri, ctx=ctx, allow_not_found=True)
        if not stat.get("not_found"):
            raise AlreadyExistsError(uri, "file")

        context_type = self._context_type_for_uri(uri)
        root_uri = (
            await self._resolve_memory_root_uri(uri)
            if context_type == "memory"
            else await self._resolve_root_uri(uri, ctx=ctx, _allow_not_found=True)
        )
        written_bytes = len(content.encode("utf-8"))
        await self._viking_fs.write_file(uri, content, ctx=ctx)
        return self._build_direct_write_result(
            uri=uri,
            root_uri=root_uri,
            context_type=context_type,
            mode="create",
            created=True,
            written_bytes=written_bytes,
        )

    def _build_write_result(
        self,
        *,
        uri: str,
        root_uri: str,
        context_type: str,
        mode: str,
        written_bytes: int,
        wait: bool,
        queue_status: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        semantic_status, vector_status = self._refresh_statuses(
            wait=wait,
            queue_status=queue_status,
        )
        return {
            "uri": uri,
            "root_uri": root_uri,
            "context_type": context_type,
            "mode": mode,
            "written_bytes": written_bytes,
            "content_updated": True,
            "semantic_status": semantic_status,
            "vector_status": vector_status,
            "queue_status": queue_status,
        }

    def _refresh_statuses(
        self,
        *,
        wait: bool,
        queue_status: Optional[Dict[str, Any]],
    ) -> tuple[str, str]:
        if not wait:
            return "queued", "queued"
        if not queue_status:
            return "complete", "complete"

        def _has_errors(name: str) -> bool:
            status = queue_status.get(name, {})
            if not isinstance(status, dict):
                return False
            try:
                return int(status.get("error_count", 0) or 0) > 0
            except (TypeError, ValueError):
                return bool(status.get("errors"))

        semantic_status = "failed" if _has_errors("Semantic") else "complete"
        vector_status = "failed" if _has_errors("Embedding") else "complete"
        return semantic_status, vector_status

    async def _write_direct_with_refresh(
        self,
        *,
        uri: str,
        root_uri: str,
        content: str,
        mode: str,
        context_type: str,
        wait: bool,
        timeout: Optional[float],
        ctx: RequestContext,
        written_bytes: int,
        telemetry_id: str,
    ) -> Dict[str, Any]:
        lock_manager = get_lock_manager()
        handle = lock_manager.create_handle()
        lock_path = self._viking_fs._uri_to_path(root_uri, ctx=ctx)
        acquired = await lock_manager.acquire_subtree(handle, lock_path)
        if not acquired:
            await lock_manager.release(handle)
            raise InvalidArgumentError(f"resource is busy and cannot be written now: {uri}")

        previous_content: Optional[str] = None
        content_written = False
        lock_transferred = False
        try:
            if mode != "create":
                previous_content = await self._viking_fs.read_file(uri, ctx=ctx)
            if wait and telemetry_id:
                get_request_wait_tracker().register_request(telemetry_id)
            await self._write_in_place(uri, content, mode=mode, ctx=ctx)
            content_written = True
            await self._enqueue_semantic_refresh(
                root_uri=root_uri,
                changed_uri=uri,
                context_type=context_type,
                ctx=ctx,
                lifecycle_lock_handle_id=handle.id,
                change_type="added" if mode == "create" else "modified",
            )
            lock_transferred = True
            queue_status = (
                await self._wait_for_request(telemetry_id=telemetry_id, timeout=timeout)
                if wait
                else None
            )
            return self._build_write_result(
                uri=uri,
                root_uri=root_uri,
                context_type=context_type,
                mode=mode,
                written_bytes=written_bytes,
                wait=wait,
                queue_status=queue_status,
            )
        except Exception:
            if not lock_transferred and content_written:
                await self._rollback_direct_write(
                    uri=uri,
                    previous_content=previous_content,
                    mode=mode,
                    ctx=ctx,
                    lock_handle=handle,
                )
            if not lock_transferred:
                await lock_manager.release(handle)
            raise
        finally:
            if wait and telemetry_id:
                get_request_wait_tracker().cleanup(telemetry_id)

    async def _rollback_direct_write(
        self,
        *,
        uri: str,
        previous_content: Optional[str],
        mode: str,
        ctx: RequestContext,
        lock_handle: Any,
    ) -> None:
        try:
            if mode == "create":
                await self._viking_fs.rm(uri, ctx=ctx, lock_handle=lock_handle)
                return
            if previous_content is not None:
                await self._viking_fs.write_file(uri, previous_content, ctx=ctx)
        except Exception:
            logger.error("Failed to rollback direct content write for %s", uri, exc_info=True)

    async def _write_memory_with_refresh(
        self,
        *,
        uri: str,
        root_uri: str,
        content: str,
        mode: str,
        wait: bool,
        timeout: Optional[float],
        ctx: RequestContext,
        written_bytes: int,
        telemetry_id: str,
    ) -> Dict[str, Any]:
        lock_manager = get_lock_manager()
        handle = lock_manager.create_handle()
        lock_path = self._viking_fs._uri_to_path(root_uri, ctx=ctx)
        acquired = await lock_manager.acquire_subtree(handle, lock_path)
        if not acquired:
            await lock_manager.release(handle)
            raise InvalidArgumentError(f"resource is busy and cannot be written now: {uri}")

        previous_content: Optional[str] = None
        content_written = False
        lock_transferred = False
        try:
            if mode != "create":
                previous_content = await self._viking_fs.read_file(uri, ctx=ctx)
            if wait and telemetry_id:
                get_request_wait_tracker().register_request(telemetry_id)
            await self._write_in_place(uri, content, mode=mode, ctx=ctx)
            content_written = True
            await self._vectorize_single_file(uri, context_type="memory", ctx=ctx)
            await self._enqueue_memory_refresh(
                root_uri=root_uri,
                modified_uri=uri,
                ctx=ctx,
                lifecycle_lock_handle_id=handle.id,
            )
            lock_transferred = True
            queue_status = (
                await self._wait_for_request(telemetry_id=telemetry_id, timeout=timeout)
                if wait
                else None
            )
            return self._build_write_result(
                uri=uri,
                root_uri=root_uri,
                context_type="memory",
                mode=mode,
                written_bytes=written_bytes,
                wait=wait,
                queue_status=queue_status,
            )
        except Exception:
            if not lock_transferred and content_written:
                await self._rollback_direct_write(
                    uri=uri,
                    previous_content=previous_content,
                    mode=mode,
                    ctx=ctx,
                    lock_handle=handle,
                )
            if not lock_transferred:
                await lock_manager.release(handle)
            raise
        finally:
            if wait and telemetry_id:
                get_request_wait_tracker().cleanup(telemetry_id)
