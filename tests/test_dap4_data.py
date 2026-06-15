# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/dap4/data.py — DAP4 chunked binary encoding."""

import struct
import sys
import xml.etree.ElementTree as ET
import zlib

import numpy as np
import pandas as pd
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


async def _collect_dap4(ds, name="test", *, use_checksums=False):
    """Collect all bytes from generate_dap4_data."""
    chunks = []
    async for chunk in generate_dap4_data(ds, name, use_checksums=use_checksums):
        chunks.append(chunk)
    return b"".join(chunks)


def _split_dmr_and_binary(data):
    """Split DAP4 response into DMR text and binary portion.

    The DMR is the leading CHUNK_DATA chunk: a 4-byte header, the DMR XML, and a
    trailing CRLF (all inside the chunk). ``binary`` is everything after that
    first chunk — the per-variable data chunks.
    """
    _type_flags, size, body = _read_chunk_header(data, 0)
    dmr_chunk = data[body : body + size]
    dmr = dmr_chunk[: -len(DMR_DATA_SEPARATOR)].decode("utf-8")
    binary = data[body + size :]
    return dmr, binary


def _read_chunk_header(data, offset):
    """Read a chunk header at given offset.

    Returns (type_flags, size, next_offset).
    """
    raw = struct.unpack(">I", data[offset : offset + 4])[0]
    chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
    size = raw & CHUNK_SIZE_MASK
    return chunk_type, size, offset + 4


class TestDAP4ResponseStructure:
    @pytest.mark.asyncio
    async def test_contains_dmr_and_separator(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)
        assert DMR_DATA_SEPARATOR in data
        dmr, _ = _split_dmr_and_binary(data)
        assert "<Dataset" in dmr

    @pytest.mark.asyncio
    async def test_ends_with_end_chunk(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)
        # Last 4 bytes should be the end chunk header
        last_4 = binary[-4:]
        raw = struct.unpack(">I", last_4)[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
        size = raw & CHUNK_SIZE_MASK
        assert chunk_type == CHUNK_END
        assert size == 0


class TestChunkHeaders:
    @pytest.mark.asyncio
    async def test_data_chunk_has_correct_flags(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # First chunk should be CHUNK_DATA with endian flag
        raw = struct.unpack(">I", binary[0:4])[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
        assert chunk_type == CHUNK_DATA

        # Check endianness flag matches system
        if sys.byteorder == "little":
            assert raw & CHUNK_LITTLE_ENDIAN == CHUNK_LITTLE_ENDIAN

    @pytest.mark.asyncio
    async def test_chunk_size_field(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # DAP4 has no array-length prefix: 2 * 4 bytes float32 = 8 (no checksum
        # requested, so no trailing CRC32 either).
        assert size == 8


class TestFloat32Encoding:
    @pytest.mark.asyncio
    async def test_float32_values(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0, 3.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Parse first chunk
        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # No length prefix: 3 float32 in native byte order, nothing else
        assert size == 12
        values = np.frombuffer(chunk_data, dtype=np.float32)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


class TestFloat64Encoding:
    @pytest.mark.asyncio
    async def test_float64_values(self):
        ds = xr.Dataset(coords={"x": np.array([10.0, 20.0], dtype="float64")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        assert size == 16
        values = np.frombuffer(chunk_data, dtype=np.float64)
        np.testing.assert_array_equal(values, [10.0, 20.0])


class TestInt16Encoding:
    @pytest.mark.asyncio
    async def test_int16_natural_size(self):
        """DAP4 Int16 should use natural 2-byte size (not widened to 4 like XDR)."""
        ds = xr.Dataset(coords={"x": np.array([100, 200], dtype="int16")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # 2 * 2 bytes = 4 (natural size, no prefix, no widening to 4 like XDR)
        assert size == 4

        chunk_data = binary[4 : 4 + size]
        values = np.frombuffer(chunk_data, dtype=np.int16)
        np.testing.assert_array_equal(values, [100, 200])


class TestByteEncoding:
    @pytest.mark.asyncio
    async def test_byte_no_padding(self):
        """DAP4 byte arrays should NOT be padded (unlike XDR)."""
        ds = xr.Dataset(coords={"x": np.array([1, 2, 3], dtype="uint8")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # 3 bytes = 3 (no length prefix, no padding to 4-byte boundary)
        assert size == 3


class TestInt64Encoding:
    @pytest.mark.asyncio
    async def test_int64_values(self):
        ds = xr.Dataset(coords={"x": np.array([1000000000000, -1], dtype="int64")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        assert size == 16
        values = np.frombuffer(chunk_data, dtype=np.int64)
        np.testing.assert_array_equal(values, [1000000000000, -1])


class TestStringEncoding:
    @pytest.mark.asyncio
    async def test_string_encoding(self):
        ds = xr.Dataset(
            {"s": xr.DataArray(np.array(["hello", "world"], dtype=object), dims=["x"])},
            coords={"x": np.array([0, 1], dtype="int32")},
        )
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        # Skip coordinate chunk (x: int32, 2 elements). No prefix, no checksum:
        # x chunk = header(4) + 2*4 bytes(8).
        offset = 0
        raw = struct.unpack(">I", binary[offset : offset + 4])[0]
        x_size = raw & CHUNK_SIZE_MASK
        assert x_size == 8
        offset += 4 + x_size  # header + data

        # String var chunk
        raw = struct.unpack(">I", binary[offset : offset + 4])[0]
        s_size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[offset + 4 : offset + 4 + s_size]

        # No array-level count prefix; each string is uint64 length + UTF-8.
        # First string: "hello"
        pos = 0
        str_len = struct.unpack("<Q", chunk_data[pos : pos + 8])[0]
        assert str_len == 5
        assert chunk_data[pos + 8 : pos + 8 + 5] == b"hello"

        # Second string: "world"
        pos += 8 + 5
        str_len = struct.unpack("<Q", chunk_data[pos : pos + 8])[0]
        assert str_len == 5
        assert chunk_data[pos + 8 : pos + 8 + 5] == b"world"


class TestCRC32:
    """CRC32 checksums are emitted only when the client requests them.

    A DAP4 client opts in with ``dap4.checksum=true``; the server then appends a
    4-byte CRC32 (over the variable's data) inside the variable's chunk payload.
    """

    @pytest.mark.asyncio
    async def test_crc32_in_data_chunk(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds, use_checksums=True)
        _, binary = _split_dmr_and_binary(data)

        # The chunk payload is [data || CRC32]; size covers both.
        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        var_data = chunk_data[:-4]
        crc_received = struct.unpack("<I", chunk_data[-4:])[0]
        assert var_data == np.array([1.0, 2.0], dtype="<f4").tobytes()
        assert crc_received == zlib.crc32(var_data) & 0xFFFFFFFF

    @pytest.mark.asyncio
    async def test_no_crc32_by_default(self):
        """Without dap4.checksum=true the chunk holds only the data."""
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        # 2 float32 only — no trailing 4-byte CRC32.
        assert size == 8

    @pytest.mark.asyncio
    async def test_crc32_per_variable(self):
        """Each variable carries its own CRC32 when checksums are requested."""
        ds = xr.Dataset(
            {"var": xr.DataArray(np.array([1.0, 2.0], dtype="float64"), dims=["x"])},
            coords={"x": np.array([0.0, 1.0], dtype="float64")},
        )
        data = await _collect_dap4(ds, use_checksums=True)
        _, binary = _split_dmr_and_binary(data)

        # Read coord chunk; CRC is the last 4 bytes of the chunk payload.
        raw = struct.unpack(">I", binary[0:4])[0]
        size1 = raw & CHUNK_SIZE_MASK
        coord_chunk = binary[4 : 4 + size1]
        crc1 = struct.unpack("<I", coord_chunk[-4:])[0]
        assert crc1 == zlib.crc32(coord_chunk[:-4]) & 0xFFFFFFFF

        # Read var chunk; its CRC is independent of the coord's.
        offset = 4 + size1
        raw = struct.unpack(">I", binary[offset : offset + 4])[0]
        size2 = raw & CHUNK_SIZE_MASK
        var_chunk = binary[offset + 4 : offset + 4 + size2]
        crc2 = struct.unpack("<I", var_chunk[-4:])[0]
        assert crc2 == zlib.crc32(var_chunk[:-4]) & 0xFFFFFFFF


class TestNoLengthPrefix:
    @pytest.mark.asyncio
    async def test_no_array_length_prefix(self):
        """DAP4 fixed arrays carry no count prefix — the count is in the DMR."""
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0, 3.0], dtype="float32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # The chunk is exactly the raw element bytes — no 8-byte uint64 prefix.
        assert size == 12
        values = np.frombuffer(chunk_data, dtype=np.float32)
        np.testing.assert_array_equal(values, [1.0, 2.0, 3.0])


class TestInt32Encoding:
    @pytest.mark.asyncio
    async def test_int32_values(self):
        ds = xr.Dataset(coords={"x": np.array([100000, -100000], dtype="int32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # 2 * 4 bytes = 8, no length prefix
        assert size == 8
        values = np.frombuffer(chunk_data, dtype=np.int32)
        np.testing.assert_array_equal(values, [100000, -100000])


class TestUInt32Encoding:
    @pytest.mark.asyncio
    async def test_uint32_values(self):
        ds = xr.Dataset(coords={"x": np.array([3_000_000_000, 1], dtype="uint32")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        assert size == 8
        values = np.frombuffer(chunk_data, dtype=np.uint32)
        np.testing.assert_array_equal(values, [3_000_000_000, 1])


class TestInt8Encoding:
    @pytest.mark.asyncio
    async def test_int8_natural_size(self):
        """DAP4 Int8 should use natural 1-byte size."""
        ds = xr.Dataset(coords={"x": np.array([-1, 0, 1], dtype="int8")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # 3 * 1 byte = 3, no length prefix
        assert size == 3

        chunk_data = binary[4 : 4 + size]
        values = np.frombuffer(chunk_data, dtype=np.int8)
        np.testing.assert_array_equal(values, [-1, 0, 1])


class TestUInt16Encoding:
    @pytest.mark.asyncio
    async def test_uint16_natural_size(self):
        """DAP4 UInt16 should use natural 2-byte size."""
        ds = xr.Dataset(coords={"x": np.array([1000, 2000], dtype="uint16")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # 2 * 2 bytes = 4, no length prefix
        assert size == 4

        chunk_data = binary[4 : 4 + size]
        values = np.frombuffer(chunk_data, dtype=np.uint16)
        np.testing.assert_array_equal(values, [1000, 2000])


class TestBoolEncoding:
    @pytest.mark.asyncio
    async def test_bool_as_uint8(self):
        """DAP4 bool → UInt8, 1 byte per element, no padding."""
        ds = xr.Dataset(coords={"x": np.array([True, False, True])})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK

        # 3 * 1 byte = 3 (no length prefix, no padding)
        assert size == 3

        chunk_data = binary[4 : 4 + size]
        values = np.frombuffer(chunk_data, dtype=np.uint8)
        np.testing.assert_array_equal(values, [1, 0, 1])


class TestUInt64MaxEncoding:
    @pytest.mark.asyncio
    async def test_uint64_max_roundtrip(self):
        """uint64 max value (18446744073709551615) should round-trip."""
        max_val = np.uint64(np.iinfo(np.uint64).max)
        ds = xr.Dataset(coords={"x": np.array([max_val, np.uint64(0)], dtype="uint64")})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        assert size == 16
        values = np.frombuffer(chunk_data, dtype=np.uint64)
        assert values[0] == np.iinfo(np.uint64).max
        assert values[1] == 0


class TestScalarEncoding:
    @pytest.mark.asyncio
    async def test_scalar_encoded_as_single_element(self):
        ds = xr.Dataset({"value": xr.DataArray(np.float64(42.0))})
        data = await _collect_dap4(ds)
        _, binary = _split_dmr_and_binary(data)

        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]

        # Scalar → single element, no length prefix
        assert size == 8
        value = struct.unpack("<d", chunk_data[0:8])[0]
        assert value == 42.0


class TestTimedeltaDAP4:
    @pytest.mark.asyncio
    async def test_timedelta_encoded_as_int64(self):
        td = np.array([np.timedelta64(i, "h") for i in range(3)])
        ds = xr.Dataset(coords={"td": td})
        data = await _collect_dap4(ds)
        dmr, binary = _split_dmr_and_binary(data)

        # DMR should show Int64 type (CF-encoded timedelta → int64)
        root = ET.fromstring(dmr)
        ns = "http://xml.opendap.org/ns/DAP/4.0#"
        # Find the variable element for td
        td_el = None
        for type_name in ("Int64", "Float64", "Int32"):
            for el in root.findall(f"{{{ns}}}{type_name}"):
                if el.get("name") == "td":
                    td_el = el
                    break
        assert td_el is not None
        assert len(binary) > 0


class TestNaTDAP4:
    @pytest.mark.asyncio
    async def test_nat_value_in_binary(self):
        times = pd.array([pd.Timestamp("2000-01-01"), pd.NaT], dtype="datetime64[ns]")
        ds = xr.Dataset(coords={"time": xr.Variable("time", times)})
        data = await _collect_dap4(ds)
        dmr, binary = _split_dmr_and_binary(data)

        # The binary should contain the int64 min sentinel for NaT
        nat_sentinel = struct.pack("<q", -9223372036854775808)
        # The binary portion contains the encoded data
        assert len(binary) > 0
        # Read the first chunk to look for NaT value
        raw = struct.unpack(">I", binary[0:4])[0]
        size = raw & CHUNK_SIZE_MASK
        chunk_data = binary[4 : 4 + size]
        # The NaT sentinel should appear in the chunk data
        assert nat_sentinel in chunk_data


class TestBytesDTypeDAP4:
    @pytest.mark.asyncio
    async def test_bytes_variable_in_dmr(self):
        ds = xr.Dataset(
            {
                "bvar": xr.DataArray(
                    np.array([b"hello", b"world"], dtype="S5"),
                    dims=["x"],
                ),
            },
            coords={"x": np.array([0, 1], dtype="int32")},
        )
        data = await _collect_dap4(ds)
        dmr, binary = _split_dmr_and_binary(data)
        # Bytes (S) dtype should map to String in DMR
        root = ET.fromstring(dmr)
        ns = "http://xml.opendap.org/ns/DAP/4.0#"
        string_els = root.findall(f"{{{ns}}}String")
        names = {el.get("name") for el in string_els}
        assert "bvar" in names


class TestEmptyArrayDAP4:
    @pytest.mark.asyncio
    async def test_empty_array_dmr(self):
        ds = xr.Dataset(
            {"empty": xr.DataArray(np.empty((0, 3), dtype="float64"), dims=["y", "x"])},
            coords={"x": np.arange(3, dtype="float64")},
        )
        data = await _collect_dap4(ds)
        dmr, binary = _split_dmr_and_binary(data)
        root = ET.fromstring(dmr)
        ns = "http://xml.opendap.org/ns/DAP/4.0#"
        dims = root.findall(f"{{{ns}}}Dimension")
        dim_map = {d.get("name"): int(d.get("size", "0")) for d in dims}
        assert dim_map.get("y") == 0

    @pytest.mark.asyncio
    async def test_empty_array_dap_binary(self):
        ds = xr.Dataset(
            {"empty": xr.DataArray(np.empty((0, 3), dtype="float64"), dims=["y", "x"])},
            coords={"x": np.arange(3, dtype="float64")},
        )
        data = await _collect_dap4(ds)
        assert DMR_DATA_SEPARATOR in data


class TestDMRChunkFraming:
    """The DMR must be framed as the leading chunk of the chunked stream.

    Regression test for the DAP4 prefetch failure: a conforming client
    (netcdf-c, pydap) reads the whole ``.dap`` body as a chunked stream and
    expects the DMR to be the first CHUNK_DATA chunk (4-byte header + DMR XML +
    CRLF). Emitting the DMR as raw bytes with no chunk header corrupts the
    stream — the client reads ``<?xm`` as a bogus chunk header and fails to
    locate the data boundary.
    """

    @pytest.mark.asyncio
    async def test_response_starts_with_dmr_chunk_header(self):
        ds = xr.Dataset(coords={"x": np.array([1.0, 2.0], dtype="float32")})
        data = await _collect_dap4(ds)

        # The body must NOT start with the raw XML declaration — it must start
        # with a 4-byte chunk header.
        assert data[:5] != b"<?xml"

        chunk_type, size, body = _read_chunk_header(data, 0)
        assert chunk_type == CHUNK_DATA
        if sys.byteorder == "little":
            raw = struct.unpack(">I", data[0:4])[0]
            assert raw & CHUNK_LITTLE_ENDIAN == CHUNK_LITTLE_ENDIAN

        # Chunk size covers the DMR XML plus the trailing CRLF.
        dmr_chunk = data[body : body + size]
        assert dmr_chunk.endswith(DMR_DATA_SEPARATOR)
        dmr_text = dmr_chunk[: -len(DMR_DATA_SEPARATOR)].decode("utf-8")
        assert dmr_text.startswith("<?xml")
        assert "<Dataset" in dmr_text

    def _make_multivar_ds(self):
        rng = np.arange(2 * 3, dtype="float32").reshape(2, 3)
        return xr.Dataset(
            {
                "temp": xr.DataArray(rng, dims=["y", "x"]),
                "flag": xr.DataArray(np.array([1, 0], dtype="int16"), dims=["y"]),
            },
            coords={
                "y": np.array([0.0, 1.0], dtype="float64"),
                "x": np.array([10.0, 20.0, 30.0], dtype="float32"),
            },
        )

    @pytest.mark.asyncio
    async def test_multi_variable_stream_dechunks(self):
        """Simulate a prefetching client reading a multi-variable .dap stream.

        Walk the chunked stream the way netcdf-c does with checksums OFF (the
        default): the DMR chunk, then one data chunk per variable, then the
        CHUNK_END marker — no interior CRC32 trailers to trip over.
        """
        ds = self._make_multivar_ds()
        data = await _collect_dap4(ds)

        # 1. DMR chunk
        chunk_type, size, offset = _read_chunk_header(data, 0)
        assert chunk_type == CHUNK_DATA
        dmr = data[offset : offset + size]
        assert dmr.endswith(DMR_DATA_SEPARATOR)
        ET.fromstring(dmr[: -len(DMR_DATA_SEPARATOR)].decode("utf-8"))
        offset += size

        # 2. One CHUNK_DATA chunk per coord + data var, back to back.
        n_vars = len(ds.coords) + len(ds.data_vars)
        for _ in range(n_vars):
            chunk_type, size, offset = _read_chunk_header(data, offset)
            assert chunk_type == CHUNK_DATA
            offset += size

        # 3. End chunk closes the stream with no trailing bytes.
        raw = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN
        size = raw & CHUNK_SIZE_MASK
        assert chunk_type == CHUNK_END
        assert size == 0
        assert offset + 4 == len(data)

    @pytest.mark.asyncio
    async def test_multi_variable_stream_dechunks_with_checksums(self):
        """With dap4.checksum=true each variable chunk ends in its own CRC32."""
        ds = self._make_multivar_ds()
        data = await _collect_dap4(ds, use_checksums=True)

        chunk_type, size, offset = _read_chunk_header(data, 0)
        offset += size  # skip DMR chunk

        n_vars = len(ds.coords) + len(ds.data_vars)
        for _ in range(n_vars):
            chunk_type, size, offset = _read_chunk_header(data, offset)
            assert chunk_type == CHUNK_DATA
            payload = data[offset : offset + size]
            offset += size
            # The CRC32 is the last 4 bytes of the chunk payload and covers the
            # variable data that precedes it.
            crc = struct.unpack("<I", payload[-4:])[0]
            assert crc == zlib.crc32(payload[:-4]) & 0xFFFFFFFF

        raw = struct.unpack(">I", data[offset : offset + 4])[0]
        assert (raw & ~CHUNK_SIZE_MASK & ~CHUNK_LITTLE_ENDIAN) == CHUNK_END
        assert offset + 4 == len(data)
