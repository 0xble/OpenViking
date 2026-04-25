# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Operator-facing kill-switch for VLM and embedding model calls.

Set ``OPENVIKING_DISABLE_MODEL_CALLS`` (or the legacy
``OPENVIKING_DISABLE_VLM`` alias) to a truthy value (``1``, ``true``, ``yes``,
``on``, case-insensitive) to halt all model dispatches in-band. Each gated
call site raises ``ModelCallDisabledError`` instead of contacting the
provider.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_PRIMARY_ENV_VAR = "OPENVIKING_DISABLE_MODEL_CALLS"
_LEGACY_ENV_VAR = "OPENVIKING_DISABLE_VLM"


def model_calls_disabled() -> bool:
    """Return True when the kill-switch is set on either env var."""
    for name in (_PRIMARY_ENV_VAR, _LEGACY_ENV_VAR):
        if os.environ.get(name, "").strip().lower() in _TRUTHY:
            return True
    return False


# Backward-compat alias kept for the safety.py public surface so that
# transitive consumers (and our pre-rename test fixtures) keep working.
vlm_disabled = model_calls_disabled


class ModelCallDisabledError(RuntimeError):
    """Raised when a VLM or embedding call is blocked by the kill-switch."""


# Pre-rename alias; safe to delete once no tests/imports reference it.
VLMDisabledError = ModelCallDisabledError


def check_model_calls_enabled(operation: str) -> None:
    """Raise ModelCallDisabledError if the kill-switch is set."""
    if model_calls_disabled():
        raise ModelCallDisabledError(
            f"{operation} blocked: {_PRIMARY_ENV_VAR} is set. Unset the env var to resume."
        )


# Backward-compat alias for callers using the older name.
check_vlm_enabled = check_model_calls_enabled
