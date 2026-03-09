# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""
Session Service for OpenViking.

Provides native session lifecycle operations plus external session import and sync.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from openviking.server.identity import RequestContext
from openviking.source import SessionImportResult, discover_session_logs, normalize_session_log
from openviking.session import Session
from openviking.session.compressor import SessionCompressor
from openviking.storage import VikingDBManager
from openviking.storage.viking_fs import VikingFS
from openviking_cli.exceptions import NotFoundError, NotInitializedError
from openviking_cli.utils import get_logger
from openviking_cli.utils.config import SessionSourceConfig, get_openviking_config

logger = get_logger(__name__)


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

    async def get(self, session_id: str, ctx: RequestContext) -> Session:
        """Get an existing session.

        Raises NotFoundError when the session does not exist under current user scope.
        """
        session = self.session(ctx, session_id)
        if not await session.exists():
            raise NotFoundError(session_id, "session")
        await session.load()
        return session

    async def sessions(self, ctx: RequestContext) -> List[Dict[str, Any]]:
        """Get all sessions for the current user.

        Returns:
            List of session info dicts
        """
        self._ensure_initialized()
        session_base_uri = f"viking://session/{ctx.user.user_space_name()}"

        try:
            entries = await self._viking_fs.ls(session_base_uri, ctx=ctx)
            sessions = []
            for entry in entries:
                name = entry.get("name", "")
                if name in [".", ".."]:
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
        session_uri = f"viking://session/{ctx.user.user_space_name()}/{session_id}"

        try:
            await self._viking_fs.rm(session_uri, recursive=True, ctx=ctx)
            logger.info(f"Deleted session: {session_id}")
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
            build_index=build_index,
        )
        return {
            "status": "imported",
            "adapter": normalized.adapter,
            "path": resolved_path,
            **stored,
        }

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
        configured_sources = sources if sources is not None else get_openviking_config().sources.sessions
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
