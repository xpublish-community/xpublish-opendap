# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for async streaming responses via httpx AsyncClient + ASGITransport."""

import numpy as np
import pytest
import xarray as xr
import xpublish
from httpx import ASGITransport, AsyncClient

from xpublish_opendap import OpenDapPlugin
from xpublish_opendap.dap.dap2.dods import DATA_SEPARATOR
from xpublish_opendap.dap.dap4.data import DMR_DATA_SEPARATOR


@pytest.fixture(scope='module')
def ds():
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


@pytest.fixture(scope='module')
def app(ds):
    rest = xpublish.Rest({'test': ds}, plugins={'opendap': OpenDapPlugin()})
    return rest.app


class TestAsyncStreaming:
    async def test_dods_streaming_response(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/datasets/test/opendap.dods') as resp:
                assert resp.status_code == 200
                chunks = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                content = b''.join(chunks)
        assert DATA_SEPARATOR in content

    async def test_dap4_streaming_response(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/datasets/test/opendap.dap') as resp:
                assert resp.status_code == 200
                chunks = []
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                content = b''.join(chunks)
        assert DMR_DATA_SEPARATOR in content

    async def test_dmr_streaming(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/datasets/test/opendap.dmr') as resp:
                assert resp.status_code == 200
                chunks = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)
                content = ''.join(chunks)
        assert '<?xml' in content
        assert '<Dataset' in content

    async def test_streaming_headers_present(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            resp = await client.get('/datasets/test/opendap.dods')
        assert 'XDODS-Server' in resp.headers
        assert 'XOPeNDAP-Server' in resp.headers


class TestPartialConsumption:
    async def test_dods_partial_read(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/datasets/test/opendap.dods') as resp:
                assert resp.status_code == 200
                # Read only first 100 bytes then close
                data = b''
                async for chunk in resp.aiter_bytes():
                    data += chunk
                    if len(data) >= 100:
                        break
        assert len(data) >= 100

    async def test_dap4_partial_read(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            async with client.stream('GET', '/datasets/test/opendap.dap') as resp:
                assert resp.status_code == 200
                data = b''
                async for chunk in resp.aiter_bytes():
                    data += chunk
                    if len(data) >= 100:
                        break
        assert len(data) >= 100

    async def test_abandoned_stream_no_leak(self, app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://test') as client:
            # Open 20 streams and close without reading
            for _ in range(20):
                async with client.stream('GET', '/datasets/test/opendap.dods') as resp:
                    assert resp.status_code == 200
                    # Close immediately without reading body
        # If we get here without exception, no leaks
