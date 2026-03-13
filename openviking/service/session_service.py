# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Session Service for OpenViking.

Provides native session lifecycle operations plus external session import and sync.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from openviking.server.identity import RequestContext
from openviking.session import Session
from openviking.session.compressor import SessionCompressor
from openviking.source import SessionImportResult, discover_session_logs, normalize_session_log
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import VikingFS
from openviking.utils.time_utils import get_current_timestamp
from openviking_cli.exceptions import NotFoundError, NotInitializedError
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import SessionSourceConfig, get_openviking_config

logger = get_logger(__name__)
_SOURCE_ALIAS_DIR = ".aliases/source-session-id"


class SessionService:
    """Session management service."""

    def __init__(
        self,
        vikingdb: Optional[VikingDBManager] = None,
        viking_fs: Optional[VikingFS] = None,
        session_compressor: Optional[SessionCompressor] = None,
    ):
        self._vikingdb = vikingdb
        self._viking_fs = viking_fs
        self._session_compressor = session_compressor

    def set_dependencies(
        self,
        vikingdb: VikingDBManager,
        viking_fs: VikingFS,
        session_compressor: SessionCompressor,
    ) -> None:
        """Set dependencies (for deferred initialization)."""
        self._vikingdb = vikingdb
        self._viking_fs = viking_fs
        self._session_compressor = session_compressor

    def _ensure_initialized(self) -> None:
        """Ensure all dependencies are initialized."""
        if not self._viking_fs:
            raise NotInitializedError("VikingFS")

    def session(self, ctx: RequestContext, session_id: Optional[str] = None) -> Session:
        """Create a new session or load an existing one.

        Args:
            session_id: Session ID, creates a new session (auto-generated ID) if None

        Returns:
            Session instance
        """
        self._ensure_initialized()
        return Session(
            viking_fs=self._viking_fs,
            vikingdb_manager=self._vikingdb,
            session_compressor=self._session_compressor,
            user=ctx.user,
            ctx=ctx,
            session_id=session_id,
        )

    async def create(self, ctx: RequestContext) -> Session:
        """Create a session and persist its root path."""
        session = self.session(ctx)
        await session.ensure_exists()
        return session

    def _session_base_uri(self, ctx: RequestContext) -> str:
        return f"viking://session/{ctx.user.user_space_name()}"

    def _source_alias_root_uri(self, ctx: RequestContext) -> str:
        return f"{self._session_base_uri(ctx)}/{_SOURCE_ALIAS_DIR}"

    def _source_alias_uri(self, source_session_id: str, ctx: RequestContext) -> str:
        encoded_source_session_id = quote(source_session_id.strip(), safe="")
        return f"{self._source_alias_root_uri(ctx)}/{encoded_source_session_id}.json"

    async def _read_json_file(self, uri: str, ctx: RequestContext) -> Dict[str, Any]:
        content = await self._viking_fs.read_file(uri, ctx=ctx)
        if isinstance(content, bytes):
            content = content.decode("utf-8")
        return json.loads(content)

    async def _write_source_session_alias(
        self,
        source_session_id: str,
        canonical_session_id: str,
        adapter: str,
        ctx: RequestContext,
    ) -> None:
        normalized_source_session_id = str(source_session_id or "").strip()
        if not normalized_source_session_id:
            return
        await self._viking_fs.mkdir(self._source_alias_root_uri(ctx), exist_ok=True, ctx=ctx)
        await self._viking_fs.write_file(
            self._source_alias_uri(normalized_source_session_id, ctx),
            json.dumps(
                {
                    "source_session_id": normalized_source_session_id,
                    "session_id": canonical_session_id,
                    "adapter": adapter,
                    "updated_at": get_current_timestamp(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            ctx=ctx,
        )

    async def _remove_source_session_alias(
        self, source_session_id: str, ctx: RequestContext
    ) -> None:
        normalized_source_session_id = str(source_session_id or "").strip()
        if not normalized_source_session_id:
            return
        try:
            await self._viking_fs.rm(
                self._source_alias_uri(normalized_source_session_id, ctx),
                ctx=ctx,
            )
        except Exception:
            logger.debug(
                "Alias removal skipped for missing source session id %s", source_session_id
            )

    async def _resolve_source_session_alias(
        self, source_session_id: str, ctx: RequestContext
    ) -> Optional[str]:
        normalized_source_session_id = source_session_id.strip()
        if not normalized_source_session_id:
            return None
        try:
            payload = await self._read_json_file(
                self._source_alias_uri(normalized_source_session_id, ctx),
                ctx,
            )
        except Exception:
            return None

        canonical_session_id = str(payload.get("session_id") or "").strip()
        if not canonical_session_id:
            return None

        session = self.session(ctx, canonical_session_id)
        if await session.exists():
            return canonical_session_id

        logger.warning(
            "Session alias for source %s points to missing session %s; removing stale alias",
            normalized_source_session_id,
            canonical_session_id,
        )
        await self._remove_source_session_alias(normalized_source_session_id, ctx)
        return None

    async def _scan_session_aliases(
        self, source_session_id: str, ctx: RequestContext
    ) -> Optional[str]:
        normalized_source_session_id = source_session_id.strip()
        if not normalized_source_session_id:
            return None

        try:
            entries = await self._viking_fs.ls(
                self._session_base_uri(ctx),
                node_limit=1_000_000,
                ctx=ctx,
            )
        except Exception:
            return None

        matched_session_ids: List[str] = []
        for entry in entries:
            session_name = str(entry.get("name") or "").strip()
            if not session_name or session_name.startswith(".") or not entry.get("isDir", False):
                continue
            uri = f"{self._session_base_uri(ctx)}/{session_name}/.source.json"
            try:
                payload = await self._read_json_file(uri, ctx)
            except Exception:
                continue
            if str(payload.get("source_session_id") or "").strip() != normalized_source_session_id:
                continue
            matched_session_ids.append(session_name)

        matched_session_ids = sorted(set(matched_session_ids))
        if not matched_session_ids:
            return None
        if len(matched_session_ids) > 1:
            logger.warning(
                "Ambiguous source session id %s matched multiple canonical sessions: %s",
                normalized_source_session_id,
                matched_session_ids,
            )
            return None

        canonical_session_id = matched_session_ids[0]
        adapter = (
            canonical_session_id.split("-", 1)[0] if "-" in canonical_session_id else "unknown"
        )
        await self._write_source_session_alias(
            normalized_source_session_id,
            canonical_session_id,
            adapter,
            ctx,
        )
        return canonical_session_id

    async def _source_session_id_for_session(
        self, session_id: str, ctx: RequestContext
    ) -> Optional[str]:
        source_uri = f"{self._session_base_uri(ctx)}/{session_id}/.source.json"
        try:
            payload = await self._read_json_file(source_uri, ctx)
        except Exception:
            return None
        source_session_id = str(payload.get("source_session_id") or "").strip()
        return source_session_id or None

    async def resolve_existing_session_id(
        self, session_id: str, ctx: RequestContext
    ) -> Optional[str]:
        """Resolve a caller-provided session ID to an existing canonical ID."""
        normalized_session_id = session_id.strip()
        if not normalized_session_id:
            return None

        literal_session = self.session(ctx, normalized_session_id)
        if await literal_session.exists():
            return normalized_session_id

        aliased_session_id = await self._resolve_source_session_alias(normalized_session_id, ctx)
        if aliased_session_id:
            return aliased_session_id

        # Backfill alias entries for older imported sessions that predate the index.
        return await self._scan_session_aliases(normalized_session_id, ctx)

    async def get(self, session_id: str, ctx: RequestContext) -> Session:
        """Get an existing session.

        Raises NotFoundError when the session does not exist under current user scope.
        """
        resolved_session_id = await self.resolve_existing_session_id(session_id, ctx)
        if not resolved_session_id:
            raise NotFoundError(session_id, "session")
        session = self.session(ctx, resolved_session_id)
        await session.load()
        return session

    async def sessions(self, ctx: RequestContext) -> List[Dict[str, Any]]:
        """Get all sessions for the current user.

        Returns:
            List of session info dicts
        """
        self._ensure_initialized()
        session_base_uri = self._session_base_uri(ctx)

        try:
            entries = await self._viking_fs.ls(session_base_uri, ctx=ctx)
            sessions = []
            for entry in entries:
                name = entry.get("name", "")
                if name in [".", ".."] or name.startswith("."):
                    continue
                sessions.append(
                    {
                        "session_id": name,
                        "uri": f"{session_base_uri}/{name}",
                        "is_dir": entry.get("isDir", False),
                    }
                )
            return sessions
        except Exception:
            return []

    async def delete(self, session_id: str, ctx: RequestContext) -> bool:
        """Delete a session.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted successfully
        """
        self._ensure_initialized()
        resolved_session_id = await self.resolve_existing_session_id(session_id, ctx)
        if not resolved_session_id:
            raise NotFoundError(session_id, "session")
        session_uri = f"viking://session/{ctx.user.user_space_name()}/{resolved_session_id}"
        source_session_id = await self._source_session_id_for_session(resolved_session_id, ctx)

        try:
            await self._viking_fs.rm(session_uri, recursive=True, ctx=ctx)
            await self._remove_source_session_alias(source_session_id or "", ctx)
            logger.info(f"Deleted session: {resolved_session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")
            raise NotFoundError(session_id, "session")

    async def commit(self, session_id: str, ctx: RequestContext) -> Dict[str, Any]:
        """Commit a session (archive messages and extract memories).

        Delegates to commit_async() for true non-blocking behavior.

        Args:
            session_id: Session ID to commit

        Returns:
            Commit result
        """
        return await self.commit_async(session_id, ctx)

    async def commit_async(self, session_id: str, ctx: RequestContext) -> Dict[str, Any]:
        """Async commit a session without blocking the event loop.

        Unlike the previous implementation which used run_async() (blocking
        the calling thread during LLM calls), this method uses native async/await
        throughout, keeping the event loop free to serve other requests.

        Args:
            session_id: Session ID to commit

        Returns:
            Commit result with keys: session_id, status, memories_extracted,
            active_count_updated, archived, stats
        """
        self._ensure_initialized()
        session = await self.get(session_id, ctx)
        return await session.commit_async()

    async def extract(self, session_id: str, ctx: RequestContext) -> List[Any]:
        """Extract memories from a session.

        Args:
            session_id: Session ID to extract from

        Returns:
            List of extracted memories
        """
        self._ensure_initialized()
        if not self._session_compressor:
            raise NotInitializedError("SessionCompressor")

        session = await self.get(session_id, ctx)

        return await self._session_compressor.extract_long_term_memories(
            messages=session.messages,
            user=ctx.user,
            session_id=session_id,
            ctx=ctx,
        )

    async def import_session_log(
        self,
        adapter: str,
        path: str,
        ctx: RequestContext,
        session_id: Optional[str] = None,
        build_index: bool = True,
        preserve_original: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Import one raw external session log into a native OpenViking session."""
        normalized = normalize_session_log(adapter=adapter, path=path, session_id=session_id)
        return await self.import_normalized_session(
            normalized=normalized,
            original_path=path,
            ctx=ctx,
            build_index=build_index,
            preserve_original=preserve_original,
            overwrite=overwrite,
        )

    async def import_normalized_session(
        self,
        normalized: SessionImportResult,
        original_path: str,
        ctx: RequestContext,
        build_index: bool = True,
        preserve_original: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Import an already-normalized session."""
        self._ensure_initialized()
        resolved_path = str(Path(original_path).expanduser().resolve())
        index_eligible = bool(normalized.metadata.get("index_eligible", True))
        resolved_build_index = build_index and index_eligible
        if not normalized.messages:
            return {
                "status": "skipped",
                "session_id": normalized.session_id,
                "adapter": normalized.adapter,
                "path": resolved_path,
                "reason": "no normalizable messages",
                "metadata": normalized.metadata,
            }

        session = self.session(ctx, normalized.session_id)
        if await session.exists():
            await self._write_source_session_alias(
                str(normalized.metadata.get("source_session_id") or ""),
                normalized.session_id,
                normalized.adapter,
                ctx,
            )
            if not overwrite:
                return {
                    "status": "skipped",
                    "session_id": normalized.session_id,
                    "session_uri": session.uri,
                    "adapter": normalized.adapter,
                    "path": resolved_path,
                    "reason": "session already exists",
                }
            await self.delete(normalized.session_id, ctx)
            session = self.session(ctx, normalized.session_id)

        stored = await session.import_messages(
            normalized.messages,
            metadata=normalized.metadata,
            original_path=resolved_path,
            preserve_original=preserve_original,
            build_index=resolved_build_index,
        )
        await self._write_source_session_alias(
            str(normalized.metadata.get("source_session_id") or ""),
            normalized.session_id,
            normalized.adapter,
            ctx,
        )
        result = {
            "status": "imported",
            "adapter": normalized.adapter,
            "path": resolved_path,
            **stored,
        }
        if build_index and not resolved_build_index:
            result["index_skip_reason"] = normalized.metadata.get("index_skip_reason")
            result["index_skip_category"] = normalized.metadata.get("index_skip_category")
        return result

    async def sync_session_sources(
        self,
        ctx: RequestContext,
        sources: Optional[List[SessionSourceConfig]] = None,
        build_index: bool = True,
        preserve_original: bool = True,
        overwrite: bool = False,
    ) -> Dict[str, Any]:
        """Sync all configured session sources into native OpenViking sessions."""
        self._ensure_initialized()
        configured_sources = (
            sources if sources is not None else get_openviking_config().sources.sessions
        )
        active_sources = [source for source in configured_sources if source.enabled]

        results: List[Dict[str, Any]] = []
        imported = 0
        skipped = 0
        failed = 0

        for source in active_sources:
            paths = discover_session_logs(source.adapter, source.path, source.glob)
            for session_log_path in paths:
                try:
                    normalized = normalize_session_log(source.adapter, session_log_path)
                    result = await self.import_normalized_session(
                        normalized=normalized,
                        original_path=str(session_log_path),
                        ctx=ctx,
                        build_index=build_index,
                        preserve_original=preserve_original,
                        overwrite=overwrite,
                    )
                    result["source_name"] = source.name
                    results.append(result)
                    if result["status"] == "imported":
                        imported += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    results.append(
                        {
                            "status": "error",
                            "adapter": source.adapter,
                            "path": str(session_log_path),
                            "source_name": source.name,
                            "error": str(exc),
                        }
                    )

        return {
            "status": "ok",
            "source_count": len(active_sources),
            "file_count": len(results),
            "imported": imported,
            "skipped": skipped,
            "failed": failed,
            "results": results,
        }
