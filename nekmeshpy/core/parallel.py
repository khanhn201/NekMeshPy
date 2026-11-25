"""Chunked element loops spread over forked worker processes.

The mesh checks (sampled Jacobians, solver metrics, the overlap search) are per-element
loops in chunks with no state carried between chunks, so they split cleanly: each worker
gets a contiguous range and the inputs are handed over by fork (copy-on-write), never
pickled.  Below :data:`SERIAL_BELOW` elements, or where ``fork`` is unavailable, the
chunks run in this process; the answer never depends on the worker count."""

from __future__ import annotations

import multiprocessing as mp
import os
from collections.abc import Callable
from typing import Any, TypeVar

#: element count below which the work runs serially (a pool costs more than it saves)
SERIAL_BELOW = 200_000
#: most workers used by default
MAX_WORKERS = 16

T = TypeVar("T")
_STATE: dict[str, Any] = {}


def default_workers(n: int) -> int:
    """Workers for ``n`` elements: 1 below :data:`SERIAL_BELOW`, else up to
    :data:`MAX_WORKERS` (bounded by the CPU count)."""
    if n < SERIAL_BELOW:
        return 1
    return max(1, min(MAX_WORKERS, os.cpu_count() or 1))


def _call(args: tuple[Callable[[dict[str, Any], int, int], Any], int, int]) -> Any:
    fn, lo, hi = args
    return fn(_STATE, lo, hi)


def chunk_map(fn: Callable[[dict[str, Any], int, int], T], n: int, per: int,
              state: dict[str, Any], workers: int | None = None) -> list[T]:
    """``[fn(state, lo, hi) for each chunk [lo, hi) of range(n)]``, in order, chunks of
    ``per``; spread over ``workers`` forked processes (default :func:`default_workers`).
    ``fn`` must be a module-level function (it is passed to the workers by reference)."""
    ranges = [(lo, min(lo + per, n)) for lo in range(0, n, max(1, per))]
    w = default_workers(n) if workers is None else int(workers)
    _STATE.clear()
    _STATE.update(state)
    try:
        if w > 1 and len(ranges) > 1 and "fork" in mp.get_all_start_methods():
            with mp.get_context("fork").Pool(min(w, len(ranges))) as pool:
                return pool.map(_call, [(fn, lo, hi) for lo, hi in ranges])
        return [fn(_STATE, lo, hi) for lo, hi in ranges]
    finally:
        _STATE.clear()
