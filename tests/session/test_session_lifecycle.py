# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Session lifecycle tests"""

from openviking import AsyncOpenViking
from openviking.session import Session


class TestSessionCreate:
    """Test Session creation"""

    async def test_create_new_session(self, client: AsyncOpenViking):
        """Test creating new session"""
        session = client.session()

        assert session is not None
        assert session.session_id is not None
        assert len(session.session_id) > 0

    async def test_create_with_id(self, client: AsyncOpenViking):
        """Test creating session with specified ID"""
        session_id = "custom_session_id_123"
        session = client.session(session_id=session_id)

        assert session.session_id == session_id

    async def test_create_multiple_sessions(self, client: AsyncOpenViking):
        """Test creating multiple sessions"""
        session1 = client.session(session_id="session_1")
        session2 = client.session(session_id="session_2")

        assert session1.session_id != session2.session_id

    async def test_session_uri(self, session: Session):
        """Test session URI"""
        uri = session.uri

        assert uri.startswith("viking://")
        assert "session" in uri
        assert session.session_id in uri


class TestSessionLoad:
    """Test Session loading"""

    async def test_load_existing_session(
        self, session_with_messages: Session, client: AsyncOpenViking
    ):
        """Test loading existing session"""
        session_id = session_with_messages.session_id

        # Create new session instance and load
        new_session = client.session(session_id=session_id)
        await new_session.load()

        # Verify messages loaded
        assert len(new_session.messages) > 0

    async def test_load_nonexistent_session(self, client: AsyncOpenViking):
        """Test loading nonexistent session"""
        session = client.session(session_id="nonexistent_session_xyz")
        await session.load()

        # Nonexistent session should be empty after loading
        assert len(session.messages) == 0

    async def test_session_properties(self, session: Session):
        """Test session properties"""
        assert hasattr(session, "uri")
        assert hasattr(session, "messages")
        assert hasattr(session, "session_id")


class TestSessionMustExist:
    """Test session(must_exist=True) raises when session does not exist."""

    async def test_must_exist_raises_for_nonexistent(self, client: AsyncOpenViking):
        """must_exist=True should raise NotFoundError for an unknown session_id."""
        import pytest

        from openviking_cli.exceptions import NotFoundError

        with pytest.raises(NotFoundError):
            client.session(session_id="definitely_not_a_real_session", must_exist=True)

    async def test_must_exist_succeeds_after_create(self, client: AsyncOpenViking):
        """must_exist=True should succeed for a session created via create_session()."""
        result = await client.create_session()
        existing_id = result["session_id"]

        session = client.session(session_id=existing_id, must_exist=True)
        assert session.session_id == existing_id

    async def test_must_exist_resolves_imported_raw_source_id(
        self, client: AsyncOpenViking, temp_dir
    ):
        """must_exist=True should resolve imported raw source IDs to canonical IDs."""
        raw = temp_dir / "codex-imported.jsonl"
        raw.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"imported-raw-123","cwd":"/tmp/project"}}',
                    '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                    '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        result = await client._client.import_session(
            adapter="codex", path=str(raw), build_index=False
        )
        assert result["status"] == "imported"
        assert result["session_id"] == "codex-imported-raw-123"

        session = client.session(session_id="imported-raw-123", must_exist=True)
        assert session.session_id == "codex-imported-raw-123"
        await session.load()
        assert len(session.messages) == 2

    async def test_must_exist_false_default_accepts_unknown_id(self, client: AsyncOpenViking):
        """Default must_exist=False should silently accept any session_id (backward compat)."""
        session = client.session(session_id="fabricated_id_abc")
        await session.load()
        assert session.session_id == "fabricated_id_abc"


class TestSessionExists:
    """Test session_exists() convenience method."""

    async def test_session_exists_true_after_create(self, client: AsyncOpenViking):
        """session_exists() should return True for a created session."""
        result = await client.create_session()
        session_id = result["session_id"]

        assert await client.session_exists(session_id) is True

    async def test_session_exists_false_for_unknown(self, client: AsyncOpenViking):
        """session_exists() should return False for an unknown session_id."""
        assert await client.session_exists("definitely_not_a_real_session") is False

    async def test_session_exists_true_after_add_message(
        self, session_with_messages: Session, client: AsyncOpenViking
    ):
        """session_exists() should return True for a session that has messages."""
        assert await client.session_exists(session_with_messages.session_id) is True

    async def test_session_exists_true_for_imported_raw_source_id(
        self, client: AsyncOpenViking, temp_dir
    ):
        """session_exists() should accept imported raw source IDs."""
        raw = temp_dir / "codex-session-exists.jsonl"
        raw.write_text(
            "\n".join(
                [
                    '{"type":"session_meta","payload":{"id":"exists-raw-123","cwd":"/tmp/project"}}',
                    '{"type":"response_item","timestamp":"2026-03-09T12:00:00Z","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]}}',
                    '{"type":"response_item","timestamp":"2026-03-09T12:00:01Z","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"world"}]}}',
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        await client._client.import_session(adapter="codex", path=str(raw), build_index=False)

        assert await client.session_exists("exists-raw-123") is True
