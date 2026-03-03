"""Async infrastructure for data loading and computation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import xarray as xr

try:
    import dask

    HAS_DASK = True
except ImportError:
    HAS_DASK = False

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
        with dask.config.set(scheduler='threads', num_workers=dask_num_workers):
            var.load()
    else:
        var.load()


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
            with dask.config.set(scheduler='threads', num_workers=dask_num_workers):
                return ds.load()
        return ds.load()

    async with asyncio.timeout(timeout):
        async with sem:
            return await loop.run_in_executor(EXECUTOR, _load)
