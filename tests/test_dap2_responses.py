# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/dap2/ — DDS, DAS, DODS, and utility response generators."""

import struct

import numpy as np
import pytest
import xarray as xr

from xpublish_opendap.dap.dap2.das import generate_das
from xpublish_opendap.dap.dap2.dds import generate_dds
from xpublish_opendap.dap.dap2.dods import DATA_SEPARATOR, generate_dods
from xpublish_opendap.dap.dap2.headers import (
    CONTENT_DESCRIPTIONS,
    CONTENT_TYPES,
    DAP2_HEADERS,
)
from xpublish_opendap.dap.dap2.responses import (
    generate_error,
    generate_help,
    generate_version,
)


@pytest.fixture
def basic_ds():
    """Simple dataset with coords and data vars."""
    return xr.Dataset(
        {
            "temp": xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype="float64"),
                dims=["y", "x"],
                attrs={"units": "kelvin", "long_name": "Temperature"},
            ),
        },
        coords={
            "x": np.array([10.0, 20.0], dtype="float32"),
            "y": np.array([30.0, 40.0], dtype="float32"),
        },
        attrs={"title": "Test dataset", "history": "Created for testing"},
    )


@pytest.fixture
def scalar_ds():
    """Dataset with scalar variables."""
    return xr.Dataset(
        {
            "value": xr.DataArray(np.float64(42.0)),
        },
    )


class TestDDS:
    def test_dataset_wrapper(self, basic_ds):
        dds = "".join(generate_dds(basic_ds, "test"))
        assert dds.startswith("Dataset {\n")
        assert dds.endswith("} test;\n")

    def test_coordinate_declarations(self, basic_ds):
        dds = "".join(generate_dds(basic_ds, "test"))
        assert "Float32 x[x = 2]" in dds
        assert "Float32 y[y = 2]" in dds

    def test_grid_declaration(self, basic_ds):
        dds = "".join(generate_dds(basic_ds, "test"))
        assert "Grid {" in dds
        assert "Float64 temp[y = 2][x = 2]" in dds
        assert "Maps:" in dds

    def test_scalar_variable(self, scalar_ds):
        dds = "".join(generate_dds(scalar_ds, "test"))
        assert "Float64 value;" in dds
        assert "Grid" not in dds

    def test_grid_maps(self, basic_ds):
        dds = "".join(generate_dds(basic_ds, "test"))
        # Maps should appear inside the Grid
        lines = dds.split("\n")
        in_maps = False
        map_vars = []
        for line in lines:
            if "Maps:" in line:
                in_maps = True
                continue
            if in_maps and "}" in line:
                break
            if in_maps:
                map_vars.append(line.strip())
        assert any("Float32 y[y = 2]" in m for m in map_vars)
        assert any("Float32 x[x = 2]" in m for m in map_vars)


class TestDAS:
    def test_attributes_wrapper(self, basic_ds):
        das = "".join(generate_das(basic_ds))
        assert das.startswith("Attributes {\n")
        assert das.endswith("}\n")

    def test_variable_attributes(self, basic_ds):
        das = "".join(generate_das(basic_ds))
        assert "temp {" in das
        assert 'String units "kelvin"' in das
        assert 'String long_name "Temperature"' in das

    def test_global_attributes(self, basic_ds):
        das = "".join(generate_das(basic_ds))
        assert "NC_GLOBAL {" in das
        assert 'String title "Test dataset"' in das

    def test_quote_escaping(self):
        ds = xr.Dataset(attrs={"note": 'uses "quotes" inside'})
        das = "".join(generate_das(ds))
        assert r"uses \"quotes\" inside" in das

    def test_numeric_attributes(self):
        ds = xr.Dataset(
            {
                "var": xr.DataArray(
                    [1.0],
                    attrs={
                        "int_attr": 42,
                        "float_attr": 3.14,
                        "np_int16": np.int16(16),
                        "np_int32": np.int32(32),
                    },
                ),
            },
        )
        das = "".join(generate_das(ds))
        assert "Int32 int_attr 42" in das
        assert "Float64 float_attr 3.14" in das
        assert "Int16 np_int16 16" in das
        assert "Int32 np_int32 32" in das


class TestDODS:
    @pytest.mark.asyncio
    async def test_contains_dds_and_separator(self, basic_ds):
        chunks = []
        async for chunk in generate_dods(basic_ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        assert DATA_SEPARATOR in data
        # DDS text appears before separator
        parts = data.split(DATA_SEPARATOR)
        dds_text = parts[0].decode("utf-8")
        assert "Dataset {" in dds_text

    @pytest.mark.asyncio
    async def test_binary_data_after_separator(self, basic_ds):
        chunks = []
        async for chunk in generate_dods(basic_ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        parts = data.split(DATA_SEPARATOR)
        binary = parts[1]
        # Should have some binary data
        assert len(binary) > 0

    @pytest.mark.asyncio
    async def test_float32_array_encoding(self):
        ds = xr.Dataset(
            coords={"x": np.array([1.0, 2.0, 3.0], dtype="float32")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]

        # First 8 bytes: length prefix (3) sent twice
        length1 = struct.unpack(">I", binary[0:4])[0]
        length2 = struct.unpack(">I", binary[4:8])[0]
        assert length1 == 3
        assert length2 == 3

        # Next 12 bytes: 3 float32 values in big-endian
        values = struct.unpack(">3f", binary[8:20])
        np.testing.assert_array_almost_equal(values, [1.0, 2.0, 3.0])

    @pytest.mark.asyncio
    async def test_float64_array_encoding(self):
        ds = xr.Dataset(
            {"var": xr.DataArray(np.array([1.0, 2.0], dtype="float64"), dims=["x"])},
            coords={"x": np.array([10.0, 20.0], dtype="float64")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]

        # x is a Grid Map of var, so no top-level coordinate binary.
        # var (Grid): main array length(2) twice + data
        offset = 0
        n1, n2 = struct.unpack(">II", binary[offset : offset + 8])
        assert n1 == 2
        offset += 8
        var_vals = struct.unpack(">2d", binary[offset : offset + 16])
        np.testing.assert_array_almost_equal(var_vals, [1.0, 2.0])
        offset += 16

        # Grid map (x): length(2) twice + data
        n1, n2 = struct.unpack(">II", binary[offset : offset + 8])
        assert n1 == 2
        offset += 8
        map_vals = struct.unpack(">2d", binary[offset : offset + 16])
        np.testing.assert_array_almost_equal(map_vals, [10.0, 20.0])

    @pytest.mark.asyncio
    async def test_byte_array_padding(self):
        """Byte arrays should be padded to 4-byte boundary."""
        ds = xr.Dataset(
            coords={"x": np.array([1, 2, 3], dtype="uint8")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]

        # Length prefix (3) twice
        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 3
        assert n2 == 3

        # 3 bytes of data + 1 byte padding = 4 bytes
        assert binary[8:11] == bytes([1, 2, 3])
        assert binary[11:12] == b"\x00"  # padding


class TestDODSTypeEncoding:
    """Tests for XDR encoding of non-float types in _xdr_encode_array."""

    @pytest.mark.asyncio
    async def test_int16_widened_to_4_bytes(self):
        ds = xr.Dataset(coords={"x": np.array([100, -200], dtype="int16")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        # Length prefix twice
        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 2
        assert n2 == 2

        # Each int16 widened to 4 bytes (big-endian int32)
        vals = struct.unpack(">2i", binary[8:16])
        assert vals == (100, -200)

    @pytest.mark.asyncio
    async def test_uint16_widened_to_4_bytes(self):
        ds = xr.Dataset(coords={"x": np.array([1000, 2000], dtype="uint16")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 2
        assert n2 == 2

        vals = struct.unpack(">2I", binary[8:16])
        assert vals == (1000, 2000)

    @pytest.mark.asyncio
    async def test_int32_encoding(self):
        ds = xr.Dataset(coords={"x": np.array([100000, -100000], dtype="int32")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 2
        vals = struct.unpack(">2i", binary[8:16])
        assert vals == (100000, -100000)

    @pytest.mark.asyncio
    async def test_uint32_encoding(self):
        ds = xr.Dataset(coords={"x": np.array([3_000_000_000, 1], dtype="uint32")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 2
        vals = struct.unpack(">2I", binary[8:16])
        assert vals == (3_000_000_000, 1)

    @pytest.mark.asyncio
    async def test_int8_encoded_as_byte(self):
        """int8 maps to DAP_INT16 which has xdr_wire_size=4, so it gets widened."""
        ds = xr.Dataset(coords={"x": np.array([-1, 0, 1], dtype="int8")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 3
        assert n2 == 3

        # int8 → DAP_INT16 → widened to >i4
        vals = struct.unpack(">3i", binary[8:20])
        assert vals == (-1, 0, 1)

    @pytest.mark.asyncio
    async def test_string_array_encoding(self):
        ds = xr.Dataset(
            {"s": xr.DataArray(np.array(["hi", "bye"], dtype=object), dims=["x"])},
            coords={"x": np.array([0, 1], dtype="int32")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        # x is a Grid Map of s, so no top-level coordinate binary.
        # Grid for s: main array (string) then map (x)
        offset = 0

        # String array: length prefix once (not doubled)
        n = struct.unpack(">I", binary[offset : offset + 4])[0]
        assert n == 2
        offset += 4

        # First string: "hi" (length=2, padded to 4)
        slen = struct.unpack(">I", binary[offset : offset + 4])[0]
        assert slen == 2
        offset += 4
        assert binary[offset : offset + 2] == b"hi"
        offset += 2
        # Padding to 4-byte boundary
        assert binary[offset : offset + 2] == b"\x00\x00"
        offset += 2

        # Second string: "bye" (length=3, padded to 4)
        slen = struct.unpack(">I", binary[offset : offset + 4])[0]
        assert slen == 3
        offset += 4
        assert binary[offset : offset + 3] == b"bye"
        offset += 3
        assert binary[offset : offset + 1] == b"\x00"

    @pytest.mark.asyncio
    async def test_scalar_dods_encoding(self):
        """Scalar values get reshaped to 1-element arrays."""
        ds = xr.Dataset({"value": xr.DataArray(np.int32(42))})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        # Scalar: length prefix (1) twice, then 4-byte big-endian int32
        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 1
        assert n2 == 1
        val = struct.unpack(">i", binary[8:12])[0]
        assert val == 42

    @pytest.mark.asyncio
    async def test_bool_encoded_as_byte(self):
        """Bool → DAP_BYTE, packed contiguously and padded."""
        ds = xr.Dataset(coords={"x": np.array([True, False, True])})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 3
        assert n2 == 3

        # 3 bytes of data + 1 byte padding
        assert binary[8:11] == bytes([1, 0, 1])
        assert binary[11:12] == b"\x00"  # padding


class TestResponses:
    def test_error_response(self):
        text = "".join(generate_error(1000, "Bad constraint"))
        assert "Error {" in text
        assert "code = 1000" in text
        assert 'message = "Bad constraint"' in text
        assert "};" in text

    def test_error_escaping(self):
        text = "".join(generate_error(1000, 'has "quotes"'))
        assert r"has \"quotes\"" in text

    def test_version_response(self):
        text = "".join(generate_version())
        assert "DAP/2.0" in text
        assert "xpublish-opendap/2.0" in text

    def test_help_response(self):
        html = generate_help()
        assert "<html>" in html
        assert ".dds" in html
        assert ".das" in html
        assert ".dods" in html

    def test_error_structure_complete(self):
        text = "".join(generate_error(1000, "Bad constraint"))
        lines = text.split("\n")
        assert lines[0] == "Error {"
        assert lines[1].strip() == "code = 1000;"
        assert lines[2].strip() == 'message = "Bad constraint";'
        assert lines[3] == "};"

    def test_error_code_types(self):
        for code, msg in [
            (1000, "syntax error"),
            (1001, "not supported"),
            (1002, "not found"),
            (1003, "out of range"),
            (1004, "too large"),
        ]:
            text = "".join(generate_error(code, msg))
            assert f"code = {code}" in text
            assert f'message = "{msg}"' in text


class TestMultiTypeDDS:
    """Verify generate_dds produces correct DAP2 type names for all dtypes."""

    def test_int8_maps_to_int16(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        # int8 → DAP_INT16 → "Int16" in DDS
        assert "Int16 i8_var" in dds

    def test_int16_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "Int16 i16_var" in dds

    def test_uint16_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "UInt16 u16_var" in dds

    def test_int32_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "Int32 i32_var" in dds

    def test_uint32_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "UInt32 u32_var" in dds

    def test_int64_maps_to_float64(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        # int64 → DAP_FLOAT64 → "Float64" in DAP2
        assert "Float64 i64_var" in dds

    def test_bool_maps_to_byte(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "Byte bool_var" in dds

    def test_string_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "String str_var" in dds

    def test_coord_int32_type(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        assert "Int32 x[x = 2]" in dds
        assert "Int32 y[y = 2]" in dds

    def test_datetime_coord_is_numeric(self, multi_type_ds):
        dds = "".join(generate_dds(multi_type_ds, "test"))
        # After CF encoding, datetime64 becomes float64 or int64 → numeric type
        # Should NOT contain "datetime" in the DDS type
        assert "time_coord" in dds
        # The CF-encoded type should be numeric (Float64 or Int64)
        lines = [ln for ln in dds.split("\n") if "time_coord" in ln]
        for line in lines:
            assert "Float64" in line or "Int64" in line or "Int32" in line


class TestMultiTypeDAS:
    """Verify generate_das handles edge-case global attributes."""

    def test_array_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert "Float64 array_attr 1.0, 2.0, 3.0" in das

    def test_bool_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert "Byte bool_attr 1" in das

    def test_nan_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert "Float64 nan_attr nan" in das

    def test_inf_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert "Float64 inf_attr inf" in das

    def test_empty_str_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert 'String empty_str_attr ""' in das

    def test_list_int_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert "Float64 list_int_attr 10, 20, 30" in das

    def test_list_str_attr(self, multi_type_ds):
        das = "".join(generate_das(multi_type_ds))
        assert 'String list_str_attr "alpha", "beta"' in das


class TestMultiTypeDODS:
    """End-to-end DODS with multi_type_ds."""

    @pytest.mark.asyncio
    async def test_all_vars_in_dds_text(self, multi_type_ds):
        chunks = []
        async for chunk in generate_dods(multi_type_ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        dds_text = data.split(DATA_SEPARATOR)[0].decode("utf-8")
        for var_name in multi_type_ds.data_vars:
            assert var_name in dds_text

    @pytest.mark.asyncio
    async def test_binary_data_non_empty(self, multi_type_ds):
        chunks = []
        async for chunk in generate_dods(multi_type_ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]
        assert len(binary) > 0


class TestLossyTypeEncoding:
    """Test lossy DAP2 encoding of int64/uint64 as float64."""

    @pytest.mark.asyncio
    async def test_int64_encoded_as_float64_xdr(self):
        val = 2**53 + 1
        ds = xr.Dataset(coords={"x": np.array([val], dtype="int64")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        # int64 → DAP2 Float64 → big-endian float64 on wire
        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 1
        wire_val = struct.unpack(">d", binary[8:16])[0]
        # Lossy: float64 cannot represent 2**53 + 1 exactly
        assert isinstance(wire_val, float)

    @pytest.mark.asyncio
    async def test_uint64_encoded_as_float64_xdr(self):
        val = np.uint64(2**53 + 1)
        ds = xr.Dataset(coords={"x": np.array([val], dtype="uint64")})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        binary = b"".join(chunks).split(DATA_SEPARATOR)[1]

        n1, n2 = struct.unpack(">II", binary[0:8])
        assert n1 == 1
        wire_val = struct.unpack(">d", binary[8:16])[0]
        assert isinstance(wire_val, float)


class TestTimedeltaDODS:
    """Test timedelta64 CF-encoding in DAP2."""

    @pytest.mark.asyncio
    async def test_timedelta_cf_encodes_to_numeric(self):
        td = np.array([np.timedelta64(i, "h") for i in range(3)])
        ds = xr.Dataset(coords={"td": td})
        dds = "".join(generate_dds(ds, "test"))
        # After CF encoding, timedelta becomes numeric (Int64 or Float64)
        lines = [ln for ln in dds.split("\n") if "td" in ln]
        for line in lines:
            assert "Float64" in line or "Int64" in line or "Int32" in line

    @pytest.mark.asyncio
    async def test_timedelta_dods_binary(self):
        td = np.array([np.timedelta64(i, "h") for i in range(3)])
        ds = xr.Dataset(coords={"td": td})
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]
        assert len(binary) > 0


class TestBytesAndUnicode:
    """Test bytes (S) and unicode (U) dtype encoding in DAP2."""

    def test_bytes_dtype_in_dds(self):
        ds = xr.Dataset(
            {
                "bvar": xr.DataArray(
                    np.array([b"hello", b"world"], dtype="S5"), dims=["x"]
                )
            },
            coords={"x": np.arange(2, dtype="int32")},
        )
        dds = "".join(generate_dds(ds, "test"))
        assert "String bvar" in dds

    def test_unicode_dds(self):
        ds = xr.Dataset(
            {
                "uvar": xr.DataArray(
                    np.array(["hello", "world"], dtype="U10"), dims=["x"]
                )
            },
            coords={"x": np.arange(2, dtype="int32")},
        )
        dds = "".join(generate_dds(ds, "test"))
        assert "String uvar" in dds

    @pytest.mark.asyncio
    async def test_mixed_string_dods_binary(self):
        ds = xr.Dataset(
            {"svar": xr.DataArray(np.array([b"abc", b"de"], dtype="S3"), dims=["x"])},
            coords={"x": np.arange(2, dtype="int32")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        binary = data.split(DATA_SEPARATOR)[1]
        assert len(binary) > 0


class TestEmptyArrayDAP2:
    """Test empty arrays (shape with 0 dimension) in DAP2."""

    def test_empty_array_dds(self):
        ds = xr.Dataset(
            {"empty": xr.DataArray(np.empty((0, 3), dtype="float64"), dims=["y", "x"])},
            coords={"x": np.arange(3, dtype="float64")},
        )
        dds = "".join(generate_dds(ds, "test"))
        assert "Dataset {" in dds
        assert "empty" in dds

    @pytest.mark.asyncio
    async def test_empty_array_dods(self):
        ds = xr.Dataset(
            {"empty": xr.DataArray(np.empty((0, 3), dtype="float64"), dims=["y", "x"])},
            coords={"x": np.arange(3, dtype="float64")},
        )
        chunks = []
        async for chunk in generate_dods(ds, "test"):
            chunks.append(chunk)
        data = b"".join(chunks)
        assert DATA_SEPARATOR in data


class TestDASFallbackAttributes:
    """Test fallback branches in _attribute_type_name and _format_string_value."""

    def test_none_attr_type_is_string(self):
        ds = xr.Dataset(attrs={"val": None})
        das = "".join(generate_das(ds))
        assert 'String val "None"' in das

    def test_complex_array_attr_fallback(self):
        ds = xr.Dataset(attrs={"arr": np.array([1 + 2j], dtype="complex128")})
        das = "".join(generate_das(ds))
        # complex dtype cannot resolve to DapType → falls back to String
        assert "String arr" in das

    def test_object_attr_type_is_string(self):
        ds = xr.Dataset(attrs={"obj": 42.0 + 0j})  # complex value, not ndarray
        das = "".join(generate_das(ds))
        # complex is not str/bool/np.integer/int/np.floating/float/ndarray/list → String
        assert "String obj" in das


class TestScalarCoordDDS:
    def test_scalar_coordinate_in_dds(self):
        ds = xr.Dataset(
            {"v": xr.DataArray([1.0, 2.0], dims=["x"])},
            coords={
                "x": np.array([0.0, 1.0], dtype="float64"),
                "ref": np.float64(0.0),
            },
        )
        dds = "".join(generate_dds(ds, "test"))
        # Scalar coord should appear without brackets
        assert "Float64 ref;" in dds


class TestHeaders:
    def test_dap2_headers_exist(self):
        assert "XDODS-Server" in DAP2_HEADERS

    def test_content_descriptions(self):
        assert CONTENT_DESCRIPTIONS["dds"] == "dods-dds"
        assert CONTENT_DESCRIPTIONS["das"] == "dods-das"
        assert CONTENT_DESCRIPTIONS["dods"] == "dods-data"

    def test_content_types(self):
        assert CONTENT_TYPES["dds"] == "text/plain"
        assert CONTENT_TYPES["dods"] == "application/octet-stream"
