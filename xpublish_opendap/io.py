"""Async infrastructure for data loading and computation."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import xarray as xr

try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

# Byte budget per rechunked block for the dask-graph fast path.
# 20 MB matches the old opendap-protocol approach — large enough to amortise
# per-block overhead, small enough to keep peak memory reasonable.
RECHUNK_BYTES = 20_000_000

# Skip slab streaming for arrays smaller than this (estimated in-memory bytes).
# 32 MB is conservative — well within server memory — and avoids sequential
# round-trip overhead for small responses where dask's eager path is faster.
SLAB_THRESHOLD_BYTES = 32 * 1024 * 1024

EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="xpublish-opendap-pool",
)

_compute_semaphores: dict[int, asyncio.Semaphore] = {}
_data_load_semaphores: dict[int, asyncio.Semaphore] = {}


def get_compute_semaphore(max_workers: int = 8) -> asyncio.Semaphore:
    """Get or create a semaphore limiting concurrent blocking compute operations.

    Args:
        max_workers: Maximum concurrent compute operations.

    Returns:
        An asyncio.Semaphore bound to the current event loop.
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    if key not in _compute_semaphores:
        _compute_semaphores[key] = asyncio.Semaphore(max_workers)
    return _compute_semaphores[key]


def get_data_load_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    """Get or create a semaphore limiting concurrent xarray data loads.

    Args:
        max_concurrent: Maximum concurrent data load operations.

    Returns:
        An asyncio.Semaphore bound to the current event loop.
    """
    loop = asyncio.get_running_loop()
    key = id(loop)
    if key not in _data_load_semaphores:
        _data_load_semaphores[key] = asyncio.Semaphore(max_concurrent)
    return _data_load_semaphores[key]


def get_slab_boundaries(
    da: xr.DataArray,
    *,
    threshold_bytes: int = SLAB_THRESHOLD_BYTES,
) -> tuple[str, list[tuple[int, int]]] | None:
    """Return the best dimension name and (start, stop) index pairs for slab streaming.

    Finds the dimension with the most dask chunks and returns its name alongside
    the slab boundaries. Slab streaming activates only for dask-backed, N-dimensional
    arrays with multiple chunks along at least one dimension, non-string dtype, and
    estimated in-memory size above *threshold_bytes*.

    Args:
        da: The DataArray to inspect.
        threshold_bytes: Minimum estimated byte size to activate slab streaming.
            Arrays smaller than this use the eager (parallel) load path.

    Returns:
        Tuple of (dim_name, boundaries) where boundaries is a list of (start, stop)
        pairs, or None if slab streaming is not applicable.
    """
    if not HAS_DASK:
        return None
    if da.ndim == 0 or not hasattr(da.data, "chunks") or da.data.chunks is None:
        return None
    if da.dtype.kind in ("U", "S", "O", "M", "m"):
        return None

    estimated_bytes = math.prod(da.shape) * da.dtype.itemsize
    if estimated_bytes <= threshold_bytes:
        return None

    # Find the dimension with the most chunks
    best_idx = max(range(da.ndim), key=lambda i: len(da.data.chunks[i]))
    best_chunks = da.data.chunks[best_idx]
    if len(best_chunks) <= 1:
        return None

    boundaries = []
    offset = 0
    for size in best_chunks:
        boundaries.append((offset, offset + size))
        offset += size
    return str(da.dims[best_idx]), boundaries


async def run_in_executor(func: Callable[..., Any], *args: Any) -> Any:
    """Run a blocking function in the thread pool with semaphore limiting.

    Args:
        func: The blocking function to run.
        *args: Arguments to pass to the function.

    Returns:
        The function's return value.
    """
    sem = get_compute_semaphore()
    loop = asyncio.get_running_loop()
    async with sem:
        return await loop.run_in_executor(EXECUTOR, func, *args)


def load_variable(
    var: xr.DataArray | xr.Variable,
    dask_num_workers: int = 4,
) -> None:
    """Load a single variable into memory with parallel dask scheduling.

    This is a synchronous function intended to be called from a thread pool
    executor. It loads the variable in-place.

    Args:
        var: The variable to load (modified in-place).
        dask_num_workers: Number of dask threads for parallel chunk loading.
    """
    if HAS_DASK:
        with dask.config.set(scheduler="threads", num_workers=dask_num_workers):
            var.load()
    else:
        var.load()


def _is_dask_graph_eligible(da: xr.DataArray) -> bool:
    """Check whether a DataArray can use the dask-graph encoding fast path.

    Eligible when the array is dask-backed, N-dimensional, and has a simple
    numeric dtype (not string, not datetime/timedelta).

    Args:
        da: The DataArray to inspect.

    Returns:
        True if the dask-graph fast path can be used.
    """
    if not HAS_DASK:
        return False
    if da.ndim == 0:
        return False
    if not hasattr(da.data, "dask"):
        return False
    if da.dtype.kind in ("U", "S", "O", "M", "m"):
        return False
    return True


def dask_graph_encode_data_bytes(
    da: xr.DataArray,
    wire_dtype: np.dtype,
    dask_num_workers: int = 4,
    rechunk_bytes: int = RECHUNK_BYTES,
) -> bytes:
    """Encode a dask-backed DataArray by fusing dtype conversion into the graph.

    Flattens the array, rechunks into ~*rechunk_bytes* blocks, and computes
    each block sequentially — letting dask parallelise the chunk fetches within
    each block while keeping peak memory bounded.

    Returns **raw data bytes only** — no length prefixes, padding, or chunk
    headers. Callers are responsible for framing.

    Args:
        da: A dask-backed DataArray (caller must verify eligibility).
        wire_dtype: Target numpy dtype for the wire format.
        dask_num_workers: Number of dask threads for parallel chunk loading.
        rechunk_bytes: Target byte budget per rechunked block.

    Returns:
        Concatenated raw bytes of the encoded data.
    """
    flat = da.data.ravel()
    chunk_elems = max(1, rechunk_bytes // wire_dtype.itemsize)
    flat = flat.rechunk(chunk_elems)

    parts: list[bytes] = []
    with dask.config.set(scheduler="threads", num_workers=dask_num_workers):
        for block in flat.blocks:
            parts.append(block.astype(wire_dtype).compute().tobytes())
    return b"".join(parts)


async def load_dataset_async(
    ds: xr.Dataset,
    *,
    timeout: float = 30.0,
    dask_num_workers: int = 4,
) -> xr.Dataset:
    """Load a lazy Dataset into memory off the event loop.

    Runs ds.load() in a thread pool executor so the event loop is not blocked.
    When dask is available, uses the threaded scheduler for parallel chunk loading.
    The dataset structure, encoding metadata, and coordinate info are preserved.

    Args:
        ds: The lazy xarray Dataset (already subsetted).
        timeout: Maximum time in seconds for loading to complete.
        dask_num_workers: Number of dask threads for parallel chunk loading.

    Returns:
        The same Dataset with all data loaded into memory.
    """
    sem = get_data_load_semaphore()
    loop = asyncio.get_running_loop()

    def _load() -> xr.Dataset:
        if HAS_DASK:
            with dask.config.set(scheduler="threads", num_workers=dask_num_workers):
                return ds.load()
        return ds.load()

    async with asyncio.timeout(timeout):
        async with sem:
            return await loop.run_in_executor(EXECUTOR, _load)
