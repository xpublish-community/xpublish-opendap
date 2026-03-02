# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for io.py — async data loading, semaphores, executor, concurrent requests."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import xarray as xr
import xpublish
from httpx import ASGITransport, AsyncClient

from xpublish_opendap import OpenDapPlugin
from xpublish_opendap.io import (
    EXECUTOR,
    get_compute_semaphore,
    get_data_load_semaphore,
    load_dataset_async,
    run_in_executor,
)


@pytest.fixture
def simple_ds():
    """Small in-memory dataset for async tests."""
    return xr.Dataset(
        {
            'temp': xr.DataArray(
                np.arange(60, dtype='float64').reshape(3, 4, 5),
                dims=['time', 'y', 'x'],
            ),
        },
        coords={
            'time': np.arange(3, dtype='float64'),
            'y': np.arange(4, dtype='float32'),
            'x': np.arange(5, dtype='float32'),
        },
    )


class TestLoadDatasetAsync:
    async def test_returns_xr_dataset(self, simple_ds):
        result = await load_dataset_async(simple_ds)
        assert isinstance(result, xr.Dataset)

    async def test_loaded_data_matches(self, simple_ds):
        result = await load_dataset_async(simple_ds)
        np.testing.assert_array_equal(result['temp'].values, simple_ds['temp'].values)

    async def test_preserves_encoding(self, tmp_path):
        ds = xr.Dataset({'temp': xr.DataArray(np.arange(10, dtype='float32'), dims=['x'])})
        ds['temp'].encoding = {'dtype': 'int16', 'scale_factor': 0.1}
        path = tmp_path / 'test.nc'
        ds.to_netcdf(path)

        opened = xr.open_dataset(path)
        loaded = await load_dataset_async(opened)
        # Encoding metadata should be preserved after load
        assert 'dtype' in loaded['temp'].encoding
        opened.close()

    async def test_loads_lazily_backed(self, tmp_path):
        ds = xr.Dataset(
            {'val': xr.DataArray(np.array([1.0, 2.0, 3.0], dtype='float64'), dims=['x'])},
            coords={'x': np.arange(3, dtype='float64')},
        )
        path = tmp_path / 'lazy.nc'
        ds.to_netcdf(path)

        opened = xr.open_dataset(path)
        loaded = await load_dataset_async(opened)
        np.testing.assert_array_equal(loaded['val'].values, [1.0, 2.0, 3.0])
        opened.close()

    async def test_timeout_parameter(self, simple_ds):
        # With a reasonable timeout the load should complete normally
        result = await load_dataset_async(simple_ds, timeout=10.0)
        assert isinstance(result, xr.Dataset)

    async def test_concurrent_loads(self, simple_ds):
        results = await asyncio.gather(*(load_dataset_async(simple_ds) for _ in range(5)))
        assert len(results) == 5
        for r in results:
            assert isinstance(r, xr.Dataset)
            np.testing.assert_array_equal(r['temp'].values, simple_ds['temp'].values)


class TestSemaphores:
    async def test_data_load_semaphore_returns_semaphore(self):
        sem = get_data_load_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    async def test_data_load_semaphore_same_per_loop(self):
        sem1 = get_data_load_semaphore()
        sem2 = get_data_load_semaphore()
        assert sem1 is sem2

    async def test_compute_semaphore_returns_semaphore(self):
        sem = get_compute_semaphore()
        assert isinstance(sem, asyncio.Semaphore)

    async def test_compute_semaphore_same_per_loop(self):
        sem1 = get_compute_semaphore()
        sem2 = get_compute_semaphore()
        assert sem1 is sem2

    async def test_semaphore_limits_concurrency(self):
        sem = get_data_load_semaphore(max_concurrent=4)
        # Acquire 4 times — should all succeed
        for _ in range(4):
            await sem.acquire()
        # 5th acquire should block (timeout quickly)
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(sem.acquire(), timeout=0.05)
        # Release all to clean up
        for _ in range(4):
            sem.release()


class TestRunInExecutor:
    async def test_runs_sync_function(self):
        result = await run_in_executor(lambda: 42)
        assert result == 42

    async def test_executor_is_thread_pool(self):
        assert isinstance(EXECUTOR, ThreadPoolExecutor)


class TestConcurrentRequests:
    @pytest.fixture
    def app(self, simple_ds):
        rest = xpublish.Rest({'test': simple_ds}, plugins={'opendap': OpenDapPlugin()})
        return rest.app

    async def test_concurrent_dods_requests(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            coros = [client.get('/datasets/test/opendap.dods') for _ in range(10)]
            responses = await asyncio.gather(*coros)
        assert all(r.status_code == 200 for r in responses)

    async def test_concurrent_mixed_protocol(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            dds_coros = [client.get('/datasets/test/opendap.dds') for _ in range(5)]
            dmr_coros = [client.get('/datasets/test/opendap.dmr') for _ in range(5)]
            responses = await asyncio.gather(*dds_coros, *dmr_coros)
        assert all(r.status_code == 200 for r in responses)
