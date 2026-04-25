# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Operator-facing safety controls for OpenViking.

Provides an in-band kill-switch for VLM and embedding calls. Activated by
setting the ``OPENVIKING_DISABLE_VLM`` environment variable to a truthy
value (``1``, ``true``, ``yes``, ``on``, case-insensitive).

When the next runaway-cost alarm fires, the operator runs::

    export OPENVIKING_DISABLE_VLM=1

and every VLM/embedding call raises ``VLMDisabledError`` immediately
instead of dispatching. The queue backs up; the operator drains and
investigates without a full-process kill.
"""

from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def vlm_disabled() -> bool:
    """Return True when the OPENVIKING_DISABLE_VLM kill-switch is set."""
    raw = os.environ.get("OPENVIKING_DISABLE_VLM", "")
    return raw.strip().lower() in _TRUTHY


class VLMDisabledError(RuntimeError):
    """Raised when a VLM/embedding call is attempted while the kill-switch is set."""


def check_vlm_enabled(operation: str) -> None:
    """Raise VLMDisabledError if the kill-switch is set.

    Args:
        operation: short name of the call site, used in the error message
            so operators know which path was halted.
    """
    if vlm_disabled():
        raise VLMDisabledError(
            f"{operation} blocked: OPENVIKING_DISABLE_VLM is set. "
            f"Unset the env var to resume."
        )
