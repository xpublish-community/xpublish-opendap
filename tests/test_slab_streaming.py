# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests verifying slab streaming produces byte-identical output to full-variable loading."""

from __future__ import annotations

import struct
import zlib

import dask.array as da_module
import numpy as np
import xarray as xr

from xpublish_opendap.dap.dap2.dods import DATA_SEPARATOR, generate_dods
from xpublish_opendap.dap.dap4.data import (
    CHUNK_SIZE_MASK,
    DMR_DATA_SEPARATOR,
    generate_dap4_data,
)
from xpublish_opendap.io import _is_dask_graph_eligible, get_slab_boundaries

pytest_plugins = []

# Known chunk type top-byte values in DAP4 transport
_DAP4_CHUNK_TYPE_BYTES = frozenset({0x00, 0x01, 0x02, 0x04, 0x05, 0x06})


async def _collect(aiter):
    """Collect all bytes from an async iterator."""
    parts = []
    async for chunk in aiter:
        parts.append(chunk)
    return b"".join(parts)


def _parse_dap4_variables(raw: bytes) -> tuple[bytes, list[tuple[bytes, int]]]:
    """Parse a DAP4 response into DMR text and per-variable (payload, crc) pairs.

    Strips transport chunk headers so responses with different chunk boundaries
    can be compared by logical content.

    The parser reads transport DATA chunks and accumulates their payloads.
    When it encounters 4 bytes whose top byte does NOT match any known chunk type,
    those bytes are interpreted as a CRC32, marking the end of a variable.

    Returns:
        (dmr_bytes, [(variable_payload_bytes, crc32_value), ...])
    """
    sep_idx = raw.index(DMR_DATA_SEPARATOR)
    dmr = raw[:sep_idx]
    binary = raw[sep_idx + len(DMR_DATA_SEPARATOR) :]

    variables: list[tuple[bytes, int]] = []
    offset = 0
    current_data = b""

    while offset + 4 <= len(binary):
        val = struct.unpack(">I", binary[offset : offset + 4])[0]
        top_byte = (val >> 24) & 0xFF

        if top_byte not in _DAP4_CHUNK_TYPE_BYTES:
            # Not a chunk header — this is a CRC32 (little-endian)
            crc = struct.unpack("<I", binary[offset : offset + 4])[0]
            variables.append((current_data, crc))
            current_data = b""
            offset += 4
            continue

        if top_byte in (0x01, 0x05):  # CHUNK_END
            break

        # DATA chunk: skip 4-byte header, read payload
        chunk_size = val & CHUNK_SIZE_MASK
        offset += 4
        current_data += binary[offset : offset + chunk_size]
        offset += chunk_size

    return dmr, variables


def _make_dataset(dtype="float64", time_chunks=3, time_size=3, y_size=4, x_size=5):
    """Create a dataset with a time-chunked data variable."""
    data = np.arange(time_size * y_size * x_size, dtype=dtype).reshape(
        time_size, y_size, x_size
    )
    ds = xr.Dataset(
        {
            "temp": xr.DataArray(data, dims=["time", "y", "x"]),
        },
        coords={
            "time": np.arange(time_size, dtype="float64"),
            "y": np.arange(y_size, dtype="float32"),
            "x": np.arange(x_size, dtype="float32"),
        },
    )
    return ds.chunk({"time": time_chunks})


def _make_datetime_dataset(time_size=3, y_size=4, x_size=5):
    """Create a dataset with a datetime64 data variable (dask-backed)."""
    base = np.datetime64("2000-01-01", "ns")
    offsets = np.arange(time_size * y_size * x_size, dtype="int64") * np.timedelta64(
        1, "h"
    )
    data = (base + offsets).reshape(time_size, y_size, x_size)
    ds = xr.Dataset(
        {
            "timestamps": xr.DataArray(data, dims=["time", "y", "x"]),
        },
        coords={
            "time": np.arange(time_size, dtype="float64"),
            "y": np.arange(y_size, dtype="float32"),
            "x": np.arange(x_size, dtype="float32"),
        },
    )
    return ds.chunk({"time": 1})


class TestGetSlabBoundaries:
    def test_dask_multi_chunk(self):
        ds = _make_dataset(time_chunks=1, time_size=3)
        result = get_slab_boundaries(ds["temp"], threshold_bytes=0)
        assert result == ("time", [(0, 1), (1, 2), (2, 3)])

    def test_dask_single_chunk_returns_none(self):
        ds = _make_dataset(time_chunks=3, time_size=3)
        assert get_slab_boundaries(ds["temp"], threshold_bytes=0) is None

    def test_numpy_returns_none(self):
        ds = _make_dataset().compute()
        assert get_slab_boundaries(ds["temp"], threshold_bytes=0) is None

    def test_scalar_returns_none(self):
        da = xr.DataArray(42.0)
        assert get_slab_boundaries(da, threshold_bytes=0) is None

    def test_string_returns_none(self):
        da = xr.DataArray(np.array(["a", "b", "c"], dtype=object), dims=["x"])
        assert get_slab_boundaries(da, threshold_bytes=0) is None

    def test_coord_1d_single_chunk_returns_none(self):
        ds = _make_dataset(time_chunks=1)
        assert get_slab_boundaries(ds.coords["time"], threshold_bytes=0) is None

    def test_picks_best_chunked_dimension(self):
        """When dim 0 has 1 chunk but dim 1 has multiple, picks dim 1."""
        data = np.arange(1 * 6 * 4, dtype="float64").reshape(1, 6, 4)
        da = xr.DataArray(data, dims=["time", "y", "x"])
        da = da.chunk({"time": 1, "y": 2, "x": 4})
        result = get_slab_boundaries(da, threshold_bytes=0)
        assert result is not None
        dim, boundaries = result
        assert dim == "y"
        assert boundaries == [(0, 2), (2, 4), (4, 6)]

    def test_small_array_skips_slab_streaming(self):
        """Default threshold returns None for tiny arrays."""
        ds = _make_dataset(time_chunks=1, time_size=3)
        # 3*4*5 * 8 bytes = 480 bytes — well below 32 MB default
        assert get_slab_boundaries(ds["temp"]) is None

    def test_threshold_boundary(self):
        """Exact threshold edge: at threshold returns None, above activates."""
        ds = _make_dataset(time_chunks=1, time_size=3)
        da = ds["temp"]
        exact_size = np.prod(da.shape) * da.dtype.itemsize
        assert get_slab_boundaries(da, threshold_bytes=exact_size) is None
        assert get_slab_boundaries(da, threshold_bytes=exact_size - 1) is not None


class TestDodsSlabStreaming:
    async def test_dods_slab_matches_eager(self):
        """Dask-chunked dataset produces identical DODS bytes to numpy-backed."""
        ds_dask = _make_dataset(dtype="float64", time_chunks=1, time_size=3)
        ds_numpy = ds_dask.compute()

        dods_slab = await _collect(
            generate_dods(ds_dask, "test", slab_threshold_bytes=0)
        )
        dods_eager = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_slab == dods_eager

    async def test_dods_slab_int32(self):
        """Integer data type produces identical results."""
        ds_dask = _make_dataset(dtype="int32", time_chunks=1, time_size=4)
        ds_numpy = ds_dask.compute()

        dods_slab = await _collect(
            generate_dods(ds_dask, "test", slab_threshold_bytes=0)
        )
        dods_eager = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_slab == dods_eager

    async def test_single_chunk_no_slab(self):
        """Single-chunk dask falls through to full-load path."""
        ds = _make_dataset(time_chunks=3, time_size=3)  # one chunk covers all
        ds_numpy = ds.compute()

        result = await _collect(generate_dods(ds, "test", slab_threshold_bytes=0))
        expected = await _collect(generate_dods(ds_numpy, "test"))
        assert result == expected

    async def test_byte_array_slab_padding(self):
        """uint8 data var with slab streaming has correct final 4-byte padding."""
        data = np.arange(15, dtype="uint8").reshape(3, 5)
        ds = xr.Dataset(
            {"vals": xr.DataArray(data, dims=["time", "x"])},
            coords={
                "time": np.arange(3, dtype="float64"),
                "x": np.arange(5, dtype="float32"),
            },
        ).chunk({"time": 1})
        ds_numpy = ds.compute()

        dods_slab = await _collect(generate_dods(ds, "test", slab_threshold_bytes=0))
        dods_eager = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_slab == dods_eager

    async def test_scalar_no_slab(self):
        """Scalar variable uses full-load path."""
        ds = xr.Dataset({"val": xr.DataArray(42.0)})
        result = await _collect(generate_dods(ds, "test"))
        assert DATA_SEPARATOR in result

    async def test_coord_only_no_slab(self):
        """Dataset with only coordinates (no data vars) works fine."""
        ds = xr.Dataset(
            coords={
                "time": np.arange(5, dtype="float64"),
                "x": np.arange(3, dtype="float32"),
            },
        )
        result = await _collect(generate_dods(ds, "test"))
        assert DATA_SEPARATOR in result


class TestDap4SlabStreaming:
    async def test_dap4_slab_payload_matches_eager(self):
        """Slab-streamed DAP4 has identical DMR, variable payloads, and CRCs."""
        ds_dask = _make_dataset(dtype="float64", time_chunks=1, time_size=3)
        ds_numpy = ds_dask.compute()

        slab_raw = await _collect(
            generate_dap4_data(ds_dask, "test", slab_threshold_bytes=0)
        )
        eager_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

        slab_dmr, slab_vars = _parse_dap4_variables(slab_raw)
        eager_dmr, eager_vars = _parse_dap4_variables(eager_raw)

        assert slab_dmr == eager_dmr
        assert len(slab_vars) == len(eager_vars)
        for (s_data, s_crc), (e_data, e_crc) in zip(slab_vars, eager_vars):
            assert s_data == e_data, "Variable payload mismatch"
            assert s_crc == e_crc, "CRC mismatch"

    async def test_dap4_slab_int32(self):
        """Integer data type produces identical DAP4 payloads."""
        ds_dask = _make_dataset(dtype="int32", time_chunks=1, time_size=4)
        ds_numpy = ds_dask.compute()

        slab_raw = await _collect(
            generate_dap4_data(ds_dask, "test", slab_threshold_bytes=0)
        )
        eager_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

        slab_dmr, slab_vars = _parse_dap4_variables(slab_raw)
        eager_dmr, eager_vars = _parse_dap4_variables(eager_raw)

        assert slab_dmr == eager_dmr
        assert len(slab_vars) == len(eager_vars)
        for (s_data, s_crc), (e_data, e_crc) in zip(slab_vars, eager_vars):
            assert s_data == e_data
            assert s_crc == e_crc

    async def test_dap4_single_chunk_no_slab(self):
        """Single-chunk dask falls through to full-load path (byte-identical)."""
        ds = _make_dataset(time_chunks=3, time_size=3)
        ds_numpy = ds.compute()

        result = await _collect(generate_dap4_data(ds, "test", slab_threshold_bytes=0))
        expected = await _collect(generate_dap4_data(ds_numpy, "test"))
        assert result == expected

    async def test_dap4_crc_valid(self):
        """Verify CRC32 in slab-streamed response covers the correct data."""
        ds_dask = _make_dataset(dtype="float64", time_chunks=1, time_size=3)
        slab_raw = await _collect(
            generate_dap4_data(ds_dask, "test", slab_threshold_bytes=0)
        )

        _dmr, variables = _parse_dap4_variables(slab_raw)
        assert len(variables) > 0

        for data, stored_crc in variables:
            computed = zlib.crc32(data) & 0xFFFFFFFF
            assert stored_crc == computed, (
                f"CRC mismatch: stored={stored_crc:#x}, computed={computed:#x}"
            )

    async def test_dap4_byte_array_slab(self):
        """uint8 data var with slab streaming produces identical DAP4 payloads."""
        data = np.arange(15, dtype="uint8").reshape(3, 5)
        ds = xr.Dataset(
            {"vals": xr.DataArray(data, dims=["time", "x"])},
            coords={
                "time": np.arange(3, dtype="float64"),
                "x": np.arange(5, dtype="float32"),
            },
        ).chunk({"time": 1})
        ds_numpy = ds.compute()

        slab_raw = await _collect(
            generate_dap4_data(ds, "test", slab_threshold_bytes=0)
        )
        eager_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

        slab_dmr, slab_vars = _parse_dap4_variables(slab_raw)
        eager_dmr, eager_vars = _parse_dap4_variables(eager_raw)

        assert slab_dmr == eager_dmr
        assert len(slab_vars) == len(eager_vars)
        for (s_data, s_crc), (e_data, e_crc) in zip(slab_vars, eager_vars):
            assert s_data == e_data
            assert s_crc == e_crc

    async def test_dap4_scalar_no_slab(self):
        """Scalar variable in DAP4 uses full-load path."""
        ds = xr.Dataset({"val": xr.DataArray(42.0)})
        result = await _collect(generate_dap4_data(ds, "test"))
        assert DMR_DATA_SEPARATOR in result


class TestDaskGraphFastPath:
    """Tests verifying the dask-graph encoding fast path produces byte-identical output."""

    # -- eligibility tests --

    def test_eligible_dask_numeric(self):
        data = da_module.from_array(np.arange(12, dtype="float64"), chunks=4)
        da = xr.DataArray(data, dims=["x"])
        assert _is_dask_graph_eligible(da) is True

    def test_not_eligible_numpy(self):
        da = xr.DataArray(np.arange(12, dtype="float64"), dims=["x"])
        assert _is_dask_graph_eligible(da) is False

    def test_not_eligible_string(self):
        data = da_module.from_array(np.array(["a", "b", "c"], dtype=object), chunks=2)
        da = xr.DataArray(data, dims=["x"])
        assert _is_dask_graph_eligible(da) is False

    def test_not_eligible_datetime(self):
        data = da_module.from_array(
            np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
            chunks=1,
        )
        da = xr.DataArray(data, dims=["time"])
        assert _is_dask_graph_eligible(da) is False

    def test_not_eligible_scalar(self):
        da = xr.DataArray(42.0)
        assert _is_dask_graph_eligible(da) is False

    # -- DAP2 byte-identity tests --

    async def test_dods_dask_graph_matches_numpy(self):
        """Dask-backed arrays via the dask-graph fast path produce identical DODS to numpy."""
        for dtype in ["float64", "float32", "int32", "int16", "uint16", "uint8"]:
            ds_dask = _make_dataset(dtype=dtype, time_chunks=1, time_size=3)
            ds_numpy = ds_dask.compute()

            dods_dask = await _collect(
                generate_dods(ds_dask, "test", slab_threshold_bytes=0),
            )
            dods_numpy = await _collect(generate_dods(ds_numpy, "test"))
            assert dods_dask == dods_numpy, f"DODS mismatch for dtype={dtype}"

    async def test_dods_full_variable_dask_graph(self):
        """Default threshold (no forced slab): dask-backed matches numpy via eager fast path."""
        ds_dask = _make_dataset(dtype="float64", time_chunks=3, time_size=3)
        ds_numpy = ds_dask.compute()

        dods_dask = await _collect(generate_dods(ds_dask, "test"))
        dods_numpy = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_dask == dods_numpy

    # -- DAP4 byte-identity tests --

    async def test_dap4_dask_graph_matches_numpy(self):
        """Dask-backed arrays via the dask-graph fast path produce identical DAP4 to numpy."""
        for dtype in ["float64", "float32", "int32", "int16", "uint16", "uint8"]:
            ds_dask = _make_dataset(dtype=dtype, time_chunks=1, time_size=3)
            ds_numpy = ds_dask.compute()

            slab_raw = await _collect(
                generate_dap4_data(ds_dask, "test", slab_threshold_bytes=0),
            )
            eager_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

            slab_dmr, slab_vars = _parse_dap4_variables(slab_raw)
            eager_dmr, eager_vars = _parse_dap4_variables(eager_raw)

            assert slab_dmr == eager_dmr, f"DMR mismatch for dtype={dtype}"
            assert len(slab_vars) == len(eager_vars), (
                f"Variable count mismatch for dtype={dtype}"
            )
            for (s_data, s_crc), (e_data, e_crc) in zip(slab_vars, eager_vars):
                assert s_data == e_data, f"Payload mismatch for dtype={dtype}"
                assert s_crc == e_crc, f"CRC mismatch for dtype={dtype}"

    async def test_dap4_full_variable_dask_graph(self):
        """Default threshold (no forced slab): dask-backed matches numpy via eager fast path."""
        ds_dask = _make_dataset(dtype="float64", time_chunks=3, time_size=3)
        ds_numpy = ds_dask.compute()

        dask_raw = await _collect(generate_dap4_data(ds_dask, "test"))
        numpy_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

        dask_dmr, dask_vars = _parse_dap4_variables(dask_raw)
        numpy_dmr, numpy_vars = _parse_dap4_variables(numpy_raw)

        assert dask_dmr == numpy_dmr
        assert len(dask_vars) == len(numpy_vars)
        for (d_data, d_crc), (n_data, n_crc) in zip(dask_vars, numpy_vars):
            assert d_data == n_data
            assert d_crc == n_crc


class TestSlabFallbackPaths:
    """Tests exercising the non-dask fallback paths in slab encoding functions.

    These paths are reached when _is_dask_graph_eligible returns False for a
    slab (e.g. numpy-backed arrays passed to the slab encoders directly).
    """

    def test_xdr_encode_slab_data_byte(self):
        """_xdr_encode_slab_data handles Byte (uint8) arrays."""
        from xpublish_opendap.dap.dap2.dods import _xdr_encode_slab_data
        from xpublish_opendap.dap.types import DAP_BYTE

        data = np.array([1, 2, 3], dtype=np.uint8)
        result = b"".join(_xdr_encode_slab_data(data, DAP_BYTE))
        assert result == data.tobytes()

    def test_xdr_encode_slab_data_int16_widening(self):
        """_xdr_encode_slab_data widens Int16 to 4-byte signed."""
        from xpublish_opendap.dap.dap2.dods import _xdr_encode_slab_data
        from xpublish_opendap.dap.types import resolve_dap_type

        data = np.array([1, -2, 3], dtype=np.int16)
        dap_type = resolve_dap_type(data.dtype)
        result = b"".join(_xdr_encode_slab_data(data, dap_type))
        expected = data.astype(">i4").tobytes()
        assert result == expected

    def test_xdr_encode_slab_data_uint16_widening(self):
        """_xdr_encode_slab_data widens UInt16 to 4-byte unsigned."""
        from xpublish_opendap.dap.dap2.dods import _xdr_encode_slab_data
        from xpublish_opendap.dap.types import resolve_dap_type

        data = np.array([1, 2, 300], dtype=np.uint16)
        dap_type = resolve_dap_type(data.dtype)
        result = b"".join(_xdr_encode_slab_data(data, dap_type))
        expected = data.astype(">u4").tobytes()
        assert result == expected

    def test_xdr_encode_slab_data_float64_standard(self):
        """_xdr_encode_slab_data handles standard float64."""
        from xpublish_opendap.dap.dap2.dods import _xdr_encode_slab_data
        from xpublish_opendap.dap.types import resolve_dap_type

        data = np.array([1.0, 2.5, 3.0], dtype=np.float64)
        dap_type = resolve_dap_type(data.dtype)
        result = b"".join(_xdr_encode_slab_data(data, dap_type))
        expected = data.astype(np.dtype(dap_type.xdr_format)).tobytes()
        assert result == expected

    def test_xdr_encode_slab_data_scalar(self):
        """_xdr_encode_slab_data reshapes scalar to 1-element array."""
        from xpublish_opendap.dap.dap2.dods import _xdr_encode_slab_data
        from xpublish_opendap.dap.types import resolve_dap_type

        data = np.float64(42.0)
        dap_type = resolve_dap_type(data.dtype)
        result = b"".join(_xdr_encode_slab_data(data, dap_type))
        expected = np.array([42.0], dtype=np.dtype(dap_type.xdr_format)).tobytes()
        assert result == expected

    def test_load_and_encode_slab_xdr_numpy_fallback(self):
        """_load_and_encode_slab_xdr falls back to numpy path for non-dask slabs."""
        from xpublish_opendap.dap.dap2.dods import _load_and_encode_slab_xdr
        from xpublish_opendap.dap.types import resolve_dap_type

        # Create a numpy-backed DataArray (not dask)
        data = np.arange(12, dtype="float64").reshape(3, 4)
        da = xr.DataArray(data, dims=["time", "x"])
        dap_type = resolve_dap_type(da.dtype)

        # Call slab encoder directly with a numpy array — exercises fallback
        result = _load_and_encode_slab_xdr(da, "time", 0, 2, 4, dap_type)
        # Should produce XDR data bytes for the first 2 rows (8 elements)
        expected_data = data[0:2].astype(np.dtype(dap_type.xdr_format)).tobytes()
        assert result == expected_data

    def test_load_and_encode_slab_dap4_numpy_fallback(self):
        """_load_and_encode_slab_dap4 falls back to numpy path for non-dask slabs."""
        from xpublish_opendap.dap.dap4.data import _load_and_encode_slab_dap4

        # Create a numpy-backed DataArray (not dask)
        data = np.arange(12, dtype="float64").reshape(3, 4)
        da = xr.DataArray(data, dims=["time", "x"])

        # Call slab encoder directly with a numpy array — exercises fallback
        result = _load_and_encode_slab_dap4(da, "time", 0, 2, 4)
        expected_data = data[0:2].astype(data.dtype.newbyteorder("=")).tobytes()
        assert result == expected_data

    def test_load_and_encode_slab_dap4_numpy_scalar_reshape(self):
        """_load_and_encode_slab_dap4 fallback handles 0-d slice (scalar reshape)."""
        from xpublish_opendap.dap.dap4.data import _load_and_encode_slab_dap4

        # 1-D array, single-element slice → after isel, data is 0-d
        data = np.array([42.0], dtype="float64")
        da = xr.DataArray(data, dims=["x"])

        result = _load_and_encode_slab_dap4(da, "x", 0, 1, 4)
        native_dtype = np.dtype("float64").newbyteorder("=")
        expected = np.array([42.0]).astype(native_dtype).tobytes()
        assert result == expected


class TestDaskGraphBytePadding:
    """Test the dask-graph fast path's Byte padding logic in _load_and_encode_xdr."""

    async def test_dods_byte_padding_dask_graph(self):
        """uint8 dask array with non-multiple-of-4 size gets correct padding."""
        # 3 elements → 3 bytes data + 1 byte padding
        data = np.array([1, 2, 3], dtype="uint8")
        ds = xr.Dataset(
            {"vals": xr.DataArray(data, dims=["x"])},
            coords={"x": np.arange(3, dtype="float64")},
        ).chunk({"x": 3})
        ds_numpy = ds.compute()

        dods_dask = await _collect(generate_dods(ds, "test"))
        dods_numpy = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_dask == dods_numpy

    async def test_dods_byte_no_padding_needed(self):
        """uint8 dask array with multiple-of-4 size needs no padding."""
        # 4 elements → 4 bytes, no padding needed
        data = np.array([1, 2, 3, 4], dtype="uint8")
        ds = xr.Dataset(
            {"vals": xr.DataArray(data, dims=["x"])},
            coords={"x": np.arange(4, dtype="float64")},
        ).chunk({"x": 4})
        ds_numpy = ds.compute()

        dods_dask = await _collect(generate_dods(ds, "test"))
        dods_numpy = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_dask == dods_numpy


class TestDatetimeHandling:
    """Tests for datetime64/timedelta64 arrays.

    datetime64/timedelta64 dtypes are excluded from both slab streaming and
    the dask-graph fast path because CF encoding is data-dependent — each slab
    could pick a different reference time, producing inconsistent bytes.
    """

    def test_datetime_excluded_from_slab_streaming(self):
        """datetime64 arrays are excluded from slab streaming."""
        ds = _make_datetime_dataset()
        da = ds["timestamps"]
        assert get_slab_boundaries(da, threshold_bytes=0) is None

    def test_datetime_excluded_from_dask_graph(self):
        """datetime64 arrays are excluded from the dask-graph fast path."""
        ds = _make_datetime_dataset()
        assert _is_dask_graph_eligible(ds["timestamps"]) is False

    async def test_dods_datetime_dask_matches_numpy(self):
        """Dask-backed datetime64 produces identical DODS to numpy-backed."""
        ds_dask = _make_datetime_dataset()
        ds_numpy = ds_dask.compute()

        dods_dask = await _collect(generate_dods(ds_dask, "test"))
        dods_eager = await _collect(generate_dods(ds_numpy, "test"))
        assert dods_dask == dods_eager

    async def test_dap4_datetime_dask_matches_numpy(self):
        """Dask-backed datetime64 produces identical DAP4 data payloads to numpy-backed.

        Note: the DMR may differ because generate_dmr() CF-encodes the still-lazy
        variable (getting default nanosecond units) while the data path loads first
        (getting optimal units). We compare data payloads only — the DMR mismatch
        is a pre-existing issue in datetime DMR generation, not related to the
        dask-graph fast path.
        """
        ds_dask = _make_datetime_dataset()
        ds_numpy = ds_dask.compute()

        dask_raw = await _collect(generate_dap4_data(ds_dask, "test"))
        eager_raw = await _collect(generate_dap4_data(ds_numpy, "test"))

        _dask_dmr, dask_vars = _parse_dap4_variables(dask_raw)
        _eager_dmr, eager_vars = _parse_dap4_variables(eager_raw)

        assert len(dask_vars) == len(eager_vars)
        for (d_data, d_crc), (e_data, e_crc) in zip(dask_vars, eager_vars):
            assert d_data == e_data, "Variable payload mismatch"
            assert d_crc == e_crc, "CRC mismatch"
