# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Source sync configuration for external session logs."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

SUPPORTED_SESSION_SOURCE_ADAPTERS = ("claude", "codex", "openclaw")


class SessionSourceConfig(BaseModel):
    """Configuration for a syncable external session source."""

    name: str = Field(default="", description="Human-readable source name")
    adapter: str = Field(description="Source adapter, e.g. codex/openclaw/claude")
    path: str = Field(description="Filesystem path that contains raw session logs")
    glob: str | None = Field(default=None, description="Optional file glob override")
    enabled: bool = Field(default=True, description="Whether the source is active")

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def normalize(self):
        self.adapter = self.adapter.strip().lower()
        if self.adapter not in SUPPORTED_SESSION_SOURCE_ADAPTERS:
            supported = ", ".join(SUPPORTED_SESSION_SOURCE_ADAPTERS)
            raise ValueError(
                f"Unsupported session source adapter '{self.adapter}', expected {supported}"
            )

        source_path = Path(self.path).expanduser()
        self.path = str(source_path)
        if not self.name:
            self.name = f"{self.adapter}:{source_path.name or 'root'}"
        return self


class SourcesConfig(BaseModel):
    """Top-level source sync configuration."""

    sessions: list[SessionSourceConfig] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
