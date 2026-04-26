# Copyright (c) 2026 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: AGPL-3.0
"""Helpers for running blocking async operations away from the HTTP loop."""

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def _run_with_worker_runner(coro: Awaitable[T]) -> T:
    with asyncio.Runner() as runner:
        return runner.run(coro)


async def run_async_in_worker(factory: Callable[[], Awaitable[T]]) -> T:
    """Run an async operation on a worker thread with its own event loop."""

    return await asyncio.to_thread(lambda: _run_with_worker_runner(factory()))
