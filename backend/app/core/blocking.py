"""The single chokepoint for blocking work.

pandas, NumPy, DuckDB, PyArrow and yfinance are all synchronous and CPU- or
IO-bound. Called directly from a FastAPI coroutine they stall the event loop and
serialise the whole server. Every call into those libraries from the service layer
must be routed through `run_blocking`.

`tests/test_async_rule.py` enforces this mechanically.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_blocking(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run a synchronous callable on a worker thread.

    Note this buys concurrency, not parallelism across the GIL: pandas releases the
    GIL inside most numeric kernels, which is where the win is. It is not a licence
    to fan out unbounded work — callers issuing many of these should bound them with
    a semaphore, since each occupies a thread from the default executor.
    """
    if kwargs:
        # asyncio.to_thread forwards **kwargs directly.
        return await asyncio.to_thread(fn, *args, **kwargs)
    # No kwargs: the call is made without the trailing `**kwargs` because that is the
    # only form mypy can verify against `to_thread`'s ParamSpec signature. Passing an
    # empty mapping is the same call at runtime, so this is a typing accommodation and
    # not a behavioural branch.
    return await asyncio.to_thread(fn, *args)  # type: ignore[call-arg]
