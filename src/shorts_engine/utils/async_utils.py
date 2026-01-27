"""Async utilities for running coroutines in sync contexts."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a sync context.

    Creates a new event loop, runs the coroutine to completion,
    and properly cleans up the loop.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
