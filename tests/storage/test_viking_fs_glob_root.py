# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import pytest

from openviking.storage.viking_fs import VikingFS


class _FakeAGFS:
    def read(self, path: str) -> bytes:
        return b""

    def write(self, path: str, data: bytes) -> str:
        return path

    def rm(self, path: str) -> None:
        return None


@pytest.mark.asyncio
async def test_glob_root_preserves_viking_scheme(monkeypatch):
    viking_fs = VikingFS(agfs=_FakeAGFS())

    async def fake_tree(uri: str, node_limit: int | None = None, ctx=None):
        return [{"rel_path": "resources/demo/readme.md"}]

    monkeypatch.setattr(viking_fs, "tree", fake_tree)

    result = await viking_fs.glob("**/*.md", uri="viking://")

    assert result["matches"] == ["viking://resources/demo/readme.md"]
