"""Async utilities for running coroutines in sync contexts."""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine in a sync context.

    Reuses the current event loop if available, otherwise creates a new one.
    The loop is NOT closed after use because async libraries (e.g. fal_client,
    httpx) cache internal clients bound to a specific loop. Closing the loop
    between sequential calls within the same Celery task causes
    "Event loop is closed" errors.

    Args:
        coro: The coroutine to execute.

    Returns:
        The result of the coroutine.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
