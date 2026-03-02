"""Async infrastructure for data loading and computation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import xarray as xr

EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix='xpublish-opendap-pool',
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


async def load_dataset_async(
    ds: xr.Dataset,
    *,
    timeout: float = 30.0,
) -> xr.Dataset:
    """Load a lazy Dataset into memory off the event loop.

    Runs ds.load() in a thread pool executor so the event loop is not blocked.
    The dataset structure, encoding metadata, and coordinate info are preserved.

    Args:
        ds: The lazy xarray Dataset (already subsetted).
        timeout: Maximum time in seconds for loading to complete.

    Returns:
        The same Dataset with all data loaded into memory.
    """
    sem = get_data_load_semaphore()
    loop = asyncio.get_running_loop()

    async with asyncio.timeout(timeout):
        async with sem:
            return await loop.run_in_executor(EXECUTOR, ds.load)
