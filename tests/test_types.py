# ruff: noqa: D100,D101,D102,D103,PLR2004
"""Tests for dap/types.py — type mapping and CF encoding."""

import numpy as np
import pytest
import xarray as xr

from xpublish_opendap.dap.types import (
    DAP_BYTE,
    DAP_FLOAT32,
    DAP_FLOAT64,
    DAP_INT8,
    DAP_INT16,
    DAP_INT32,
    DAP_INT64,
    DAP_STRING,
    DAP_UINT16,
    DAP_UINT32,
    DAP_UINT64,
    cf_encode_variable,
    resolve_dap_type,
)


class TestResolveDapType:
    def test_float64(self):
        assert resolve_dap_type(np.dtype("float64")) is DAP_FLOAT64

    def test_float32(self):
        assert resolve_dap_type(np.dtype("float32")) is DAP_FLOAT32

    def test_int32(self):
        assert resolve_dap_type(np.dtype("int32")) is DAP_INT32

    def test_int16(self):
        assert resolve_dap_type(np.dtype("int16")) is DAP_INT16

    def test_uint16(self):
        assert resolve_dap_type(np.dtype("uint16")) is DAP_UINT16

    def test_uint32(self):
        assert resolve_dap_type(np.dtype("uint32")) is DAP_UINT32

    def test_int64_maps_to_float64(self):
        assert resolve_dap_type(np.dtype("int64")) is DAP_FLOAT64

    def test_uint64_maps_to_float64(self):
        assert resolve_dap_type(np.dtype("uint64")) is DAP_FLOAT64

    def test_int8_maps_to_int16(self):
        assert resolve_dap_type(np.dtype("int8")) is DAP_INT16

    def test_uint8_maps_to_byte(self):
        assert resolve_dap_type(np.dtype("uint8")) is DAP_BYTE

    def test_bool_maps_to_byte(self):
        assert resolve_dap_type(np.dtype("bool")) is DAP_BYTE

    def test_str_object_maps_to_string(self):
        assert resolve_dap_type(np.dtype("O")) is DAP_STRING

    def test_unicode_maps_to_string(self):
        assert resolve_dap_type(np.dtype("U10")) is DAP_STRING

    def test_datetime64_maps_to_float64(self):
        assert resolve_dap_type(np.dtype("datetime64[ns]")) is DAP_FLOAT64

    def test_timedelta64_maps_to_float64(self):
        assert resolve_dap_type(np.dtype("timedelta64[ns]")) is DAP_FLOAT64

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="No DAP2 mapping"):
            resolve_dap_type(np.dtype("complex128"))

    def test_bytes_dtype_maps_to_string(self):
        assert resolve_dap_type(np.dtype("S10")) is DAP_STRING


class TestResolveDapTypeDAP4:
    def test_float64(self):
        assert resolve_dap_type(np.dtype("float64"), protocol="dap4") is DAP_FLOAT64

    def test_float32(self):
        assert resolve_dap_type(np.dtype("float32"), protocol="dap4") is DAP_FLOAT32

    def test_int32(self):
        assert resolve_dap_type(np.dtype("int32"), protocol="dap4") is DAP_INT32

    def test_int16(self):
        assert resolve_dap_type(np.dtype("int16"), protocol="dap4") is DAP_INT16

    def test_uint16(self):
        assert resolve_dap_type(np.dtype("uint16"), protocol="dap4") is DAP_UINT16

    def test_uint32(self):
        assert resolve_dap_type(np.dtype("uint32"), protocol="dap4") is DAP_UINT32

    def test_int64_native(self):
        """DAP4 maps int64 natively (not lossy Float64 like DAP2)."""
        assert resolve_dap_type(np.dtype("int64"), protocol="dap4") is DAP_INT64

    def test_uint64_native(self):
        """DAP4 maps uint64 natively (not lossy Float64 like DAP2)."""
        assert resolve_dap_type(np.dtype("uint64"), protocol="dap4") is DAP_UINT64

    def test_int8_native(self):
        """DAP4 has native Int8 (not widened to Int16 like DAP2)."""
        assert resolve_dap_type(np.dtype("int8"), protocol="dap4") is DAP_INT8

    def test_uint8_maps_to_byte(self):
        assert resolve_dap_type(np.dtype("uint8"), protocol="dap4") is DAP_BYTE

    def test_bool_maps_to_byte(self):
        assert resolve_dap_type(np.dtype("bool"), protocol="dap4") is DAP_BYTE

    def test_str_object_maps_to_string(self):
        assert resolve_dap_type(np.dtype("O"), protocol="dap4") is DAP_STRING

    def test_datetime64_maps_to_float64(self):
        assert (
            resolve_dap_type(np.dtype("datetime64[ns]"), protocol="dap4") is DAP_FLOAT64
        )

    def test_unknown_dtype_raises(self):
        with pytest.raises(ValueError, match="No DAP4 mapping"):
            resolve_dap_type(np.dtype("complex128"), protocol="dap4")

    def test_dap4_name_fields(self):
        assert DAP_INT64.dap4_name == "Int64"
        assert DAP_UINT64.dap4_name == "UInt64"
        assert DAP_INT8.dap4_name == "Int8"
        assert DAP_BYTE.dap4_name == "UInt8"
        assert DAP_FLOAT64.dap4_name == "Float64"
        assert DAP_STRING.dap4_name == "String"

    def test_dap4_wire_size(self):
        assert DAP_INT16.dap4_wire_size == 2  # natural size, not XDR 4
        assert DAP_UINT16.dap4_wire_size == 2
        assert DAP_INT64.dap4_wire_size == 8
        assert DAP_FLOAT64.dap4_wire_size == 8
        assert DAP_BYTE.dap4_wire_size == 1


class TestDapTypeProperties:
    def test_string_is_string(self):
        assert DAP_STRING.is_string is True

    def test_float64_is_not_string(self):
        assert DAP_FLOAT64.is_string is False

    def test_string_not_numeric(self):
        assert DAP_STRING.is_numeric is False

    def test_float64_is_numeric(self):
        assert DAP_FLOAT64.is_numeric is True

    def test_string_no_doubled_length(self):
        assert DAP_STRING.needs_xdr_length_doubled is False

    def test_float64_doubled_length(self):
        assert DAP_FLOAT64.needs_xdr_length_doubled is True


class TestCfEncodeVariable:
    def test_float32_passthrough(self):
        var = xr.Variable("x", np.array([1.0, 2.0], dtype="float32"))
        encoded = cf_encode_variable(var)
        assert encoded.dtype == np.dtype("float32")
        np.testing.assert_array_equal(encoded.data, var.data)

    def test_datetime64_encoded(self):
        times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
        var = xr.Variable("time", times)
        encoded = cf_encode_variable(var)
        # Should be numeric after CF encoding
        assert encoded.dtype.kind in ("f", "i")

    def test_big_endian_normalized(self):
        data = np.array([1.0, 2.0], dtype=">f4")
        var = xr.Variable("x", data)
        encoded = cf_encode_variable(var)
        assert encoded.dtype.byteorder in ("=", "|", "<")
