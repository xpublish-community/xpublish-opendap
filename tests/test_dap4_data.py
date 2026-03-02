# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/dap4/data.py — DAP4 chunked binary encoding."""

import struct
import sys
import zlib

import numpy as np
import pytest
import xarray as xr

from xpublish_opendap.dap.dap4.data import (
    CHUNK_DATA,
    CHUNK_END,
    CHUNK_LITTLE_ENDIAN,
    CHUNK_SIZE_MASK,
    DMR_DATA_SEPARATOR,
    generate_dap4_data,
)


async def _collect_dap4(ds, name='test'):
    """Collect all bytes from generate_dap4_data."""
    chunks = []
    async for chunk in generate_dap4_data(ds, name):
        chunks.append(chunk)
    return b''.join(chunks)


def _split_dmr_and_binary(data):
    """Split DAP4 response into DMR text and binary portion."""
    idx = data.index(DMR_DATA_SEPARATOR)
    dmr = data[:idx].decode('utf-8')
    binary = data[idx + len(DMR_DATA_SEPARATOR) :]
    return dmr, binary


def _read_chunk_header(data, offset):
    """Read a chunk header at the given offset. Returns (type_flags, size, next_offset)."""
    raw = struct.unpack('>I', data[offset : offset + 4])[0]
    chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
    size = raw & CHUNK_SIZE_MASK
    return chunk_type, size, offset + 4


class TestDAP4ResponseStructure:
    @pytest.mark.asyncio
    async def test_contains_dmr_and_separator(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0], dtype='float32')})
        data = await _collect_dap4(ds)
        assert DMR_DATA_SEPARATOR in data
        dmr, _ = _split_dmr_and_binary(data)
        assert '<Dataset' in dmr

    @pytest.mark.asyncio
    async def test_ends_with_end_chunk(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)
        # Last 4 bytes should be the end chunk header
        last_4 = binary[-4:]
        raw = struct.unpack('>I', last_4)[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
        size = raw & CHUNK_SIZE_MASK
        assert chunk_type == CHUNK_END
        assert size == 0


class TestChunkHeaders:
    @pytest.mark.asyncio
    async def test_data_chunk_has_correct_flags(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # First chunk should be CHUNK_DATA with endian flag
        raw = struct.unpack('>I', binary[0:4])[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
        assert chunk_type == CHUNK_DATA

        # Check endianness flag matches system
        if sys.byteorder == 'little':
            assert raw & CHUNK_LITTLE_ENDIAN == CHUNK_LITTLE_ENDIAN

    @pytest.mark.asyncio
    async def test_chunk_size_field(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # x coord: uint64 length prefix (8) + 2 * 4 bytes float32 = 16
        assert size == 16


class TestFloat32Encoding:
    @pytest.mark.asyncio
    async def test_float32_values(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0, 3.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Parse first chunk
        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # uint64 length prefix
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 3

        # 3 float32 in native byte order
        values = np.frombuffer(chunk_data[8:20], dtype=np.float32)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


class TestFloat64Encoding:
    @pytest.mark.asyncio
    async def test_float64_values(self):
        ds = xr.Dataset(coords={'x': np.array([10.0, 20.0], dtype='float64')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:24], dtype=np.float64)
        np.testing.assert_array_equal(values, [10.0, 20.0])


class TestInt16Encoding:
    @pytest.mark.asyncio
    async def test_int16_natural_size(self):
        """DAP4 Int16 should use natural 2-byte size (not widened to 4 like XDR)."""
        ds = xr.Dataset(coords={'x': np.array([100, 200], dtype='int16')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # uint64 (8) + 2 * 2 bytes = 12 (not 8+8=16 as in XDR)
        assert size == 12

        chunk_data = binary[4 : 4 + size]
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:12], dtype=np.int16)
        np.testing.assert_array_equal(values, [100, 200])


class TestByteEncoding:
    @pytest.mark.asyncio
    async def test_byte_no_padding(self):
        """DAP4 byte arrays should NOT be padded (unlike XDR)."""
        ds = xr.Dataset(coords={'x': np.array([1, 2, 3], dtype='uint8')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # uint64 (8) + 3 bytes = 11 (no padding to 4-byte boundary)
        assert size == 11


class TestInt64Encoding:
    @pytest.mark.asyncio
    async def test_int64_values(self):
        ds = xr.Dataset(coords={'x': np.array([1000000000000, -1], dtype='int64')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:24], dtype=np.int64)
        np.testing.assert_array_equal(values, [1000000000000, -1])


class TestStringEncoding:
    @pytest.mark.asyncio
    async def test_string_encoding(self):
        ds = xr.Dataset(
            {'s': xr.DataArray(np.array(['hello', 'world'], dtype=object), dims=['x'])},
            coords={'x': np.array([0, 1], dtype='int32')},
        )
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Skip coordinate chunk (x: int32, 2 elements) + CRC
        # x chunk: header(4) + uint64(8) + 2*4 bytes(8) = 20 + CRC(4) = 24
        offset = 0
        raw = struct.unpack('>I', binary[offset : offset + 4])[0]
        x_size = raw & CHUNK_SIZE_MASK
        offset += 4 + x_size + 4  # header + data + CRC

        # String var chunk
        raw = struct.unpack('>I', binary[offset : offset + 4])[0]
        s_size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[offset + 4 : offset + 4 + s_size]

        # Array length
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        # First string: "hello"
        pos = 8
        str_len = struct.unpack('<Q', chunk_data[pos : pos + 8])[0]
        assert str_len == 5
        assert chunk_data[pos + 8 : pos + 8 + 5] == b'hello'

        # Second string: "world"
        pos += 8 + 5
        str_len = struct.unpack('<Q', chunk_data[pos : pos + 8])[0]
        assert str_len == 5
        assert chunk_data[pos + 8 : pos + 8 + 5] == b'world'


class TestCRC32:
    @pytest.mark.asyncio
    async def test_crc32_after_data_chunk(self):
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Read first data chunk
        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # CRC32 follows immediately after the chunk data
        crc_bytes = binary[4 + size : 4 + size + 4]
        crc_received = struct.unpack('<I', crc_bytes)[0]

        # Compute expected CRC32
        crc_expected = zlib.crc32(chunk_data) & 0xFFFFFFFF
        assert crc_received == crc_expected

    @pytest.mark.asyncio
    async def test_crc32_per_variable(self):
        """Each variable should have its own CRC32."""
        ds = xr.Dataset(
            {'var': xr.DataArray(np.array([1.0, 2.0], dtype='float64'), dims=['x'])},
            coords={'x': np.array([0.0, 1.0], dtype='float64')},
        )
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Read coord chunk + CRC
        raw = struct.unpack('>I', binary[0:4])[0]
        size1 = raw & CHUNK_SIZE_MASK
        coord_data = binary[4 : 4 + size1]
        crc1 = struct.unpack('<I', binary[4 + size1 : 4 + size1 + 4])[0]
        assert crc1 == zlib.crc32(coord_data) & 0xFFFFFFFF

        # Read var chunk + CRC
        offset = 4 + size1 + 4
        raw = struct.unpack('>I', binary[offset : offset + 4])[0]
        size2 = raw & CHUNK_SIZE_MASK
        var_data = binary[offset + 4 : offset + 4 + size2]
        crc2 = struct.unpack('<I', binary[offset + 4 + size2 : offset + 4 + size2 + 4])[0]
        assert crc2 == zlib.crc32(var_data) & 0xFFFFFFFF


class TestLengthPrefix:
    @pytest.mark.asyncio
    async def test_uint64_length_prefix(self):
        """DAP4 uses uint64 length prefix sent once (not doubled like DAP2)."""
        ds = xr.Dataset(coords={'x': np.array([1.0, 2.0, 3.0], dtype='float32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # Length prefix is 8-byte uint64
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 3

        # The next 8 bytes should be float32 data, NOT another length prefix
        values = np.frombuffer(chunk_data[8:20], dtype=np.float32)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


class TestInt32Encoding:
    @pytest.mark.asyncio
    async def test_int32_values(self):
        ds = xr.Dataset(coords={'x': np.array([100000, -100000], dtype='int32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # uint64 length prefix (8) + 2 * 4 bytes = 16
        assert size == 16

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:16], dtype=np.int32)
        np.testing.assert_array_equal(values, [100000, -100000])


class TestUInt32Encoding:
    @pytest.mark.asyncio
    async def test_uint32_values(self):
        ds = xr.Dataset(coords={'x': np.array([3_000_000_000, 1], dtype='uint32')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        assert size == 16

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:16], dtype=np.uint32)
        np.testing.assert_array_equal(values, [3_000_000_000, 1])


class TestInt8Encoding:
    @pytest.mark.asyncio
    async def test_int8_natural_size(self):
        """DAP4 Int8 should use natural 1-byte size."""
        ds = xr.Dataset(coords={'x': np.array([-1, 0, 1], dtype='int8')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # uint64 (8) + 3 * 1 byte = 11
        assert size == 11

        chunk_data = binary[4 : 4 + size]
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 3

        values = np.frombuffer(chunk_data[8:11], dtype=np.int8)
        np.testing.assert_array_equal(values, [-1, 0, 1])


class TestUInt16Encoding:
    @pytest.mark.asyncio
    async def test_uint16_natural_size(self):
        """DAP4 UInt16 should use natural 2-byte size."""
        ds = xr.Dataset(coords={'x': np.array([1000, 2000], dtype='uint16')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # uint64 (8) + 2 * 2 bytes = 12
        assert size == 12

        chunk_data = binary[4 : 4 + size]
        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:12], dtype=np.uint16)
        np.testing.assert_array_equal(values, [1000, 2000])


class TestBoolEncoding:
    @pytest.mark.asyncio
    async def test_bool_as_uint8(self):
        """DAP4 bool → UInt8, 1 byte per element, no padding."""
        ds = xr.Dataset(coords={'x': np.array([True, False, True])})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # uint64 (8) + 3 * 1 byte = 11 (no padding)
        assert size == 11

        chunk_data = binary[4 : 4 + size]
        values = np.frombuffer(chunk_data[8:11], dtype=np.uint8)
        np.testing.assert_array_equal(values, [1, 0, 1])


class TestUInt64MaxEncoding:
    @pytest.mark.asyncio
    async def test_uint64_max_roundtrip(self):
        """uint64 max value (18446744073709551615) should round-trip."""
        max_val = np.uint64(np.iinfo(np.uint64).max)
        ds = xr.Dataset(coords={'x': np.array([max_val, np.uint64(0)], dtype='uint64')})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 2

        values = np.frombuffer(chunk_data[8:24], dtype=np.uint64)
        assert values[0] == np.iinfo(np.uint64).max
        assert values[1] == 0


class TestScalarEncoding:
    @pytest.mark.asyncio
    async def test_scalar_encoded_as_single_element(self):
        ds = xr.Dataset({'value': xr.DataArray(np.float64(42.0))})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack('>I', binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        n = struct.unpack('<Q', chunk_data[0:8])[0]
        assert n == 1

        value = struct.unpack('<d', chunk_data[8:16])[0]
        assert value == 42.0
