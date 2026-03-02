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

        # x coordinate: length(2) twice + 2*8 bytes
        offset = 0
        n1, n2 = struct.unpack(">II", binary[offset : offset + 8])
        assert n1 == 2
        assert n2 == 2
        offset += 8
        x_vals = struct.unpack(">2d", binary[offset : offset + 16])
        np.testing.assert_array_almost_equal(x_vals, [10.0, 20.0])
        offset += 16

        # var (Grid): main array length(2) twice + data
        n1, n2 = struct.unpack(">II", binary[offset : offset + 8])
        assert n1 == 2
        offset += 8
        var_vals = struct.unpack(">2d", binary[offset : offset + 16])
        np.testing.assert_array_almost_equal(var_vals, [1.0, 2.0])
        offset += 16

        # Grid map (x again): length(2) twice + data
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
