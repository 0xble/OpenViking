# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Fixtures for transaction tests that exercise the real local AGFS backend."""

from pathlib import Path
from typing import Any

import pytest

from openviking.utils.agfs_utils import create_agfs_client
from openviking_cli.utils.config.agfs_config import AGFSConfig


@pytest.fixture
def agfs_client(tmp_path: Path) -> Any:
    agfs_config = AGFSConfig(path=str(tmp_path / "agfs"), backend="local")
    return create_agfs_client(agfs_config)


@pytest.fixture
def test_dir(agfs_client: Any, tmp_path: Path) -> str:
    path = f"/local/test-{tmp_path.name}"
    agfs_client.mkdir(path)
    return path
