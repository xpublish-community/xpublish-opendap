"""DAP type system: numpy dtype to DAP type mapping and CF encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class DapType:
    """Mapping between a numpy dtype and its DAP wire representation."""

    dap2_name: str
    xdr_format: str | None
    xdr_wire_size: int
    numpy_dtype: np.dtype
    is_numeric: bool = True
    needs_xdr_length_doubled: bool = True
    dap4_name: str | None = None
    dap4_wire_size: int = 0

    @property
    def is_string(self) -> bool:
        """Whether this is a variable-length string type."""
        return self.dap2_name == "String"


# Canonical DapType instances
DAP_BYTE = DapType(
    dap2_name="Byte",
    xdr_format="B",
    xdr_wire_size=1,
    numpy_dtype=np.dtype("uint8"),
    dap4_name="UInt8",
    dap4_wire_size=1,
)
DAP_INT8 = DapType(
    dap2_name="Byte",
    xdr_format="B",
    xdr_wire_size=1,
    numpy_dtype=np.dtype("int8"),
    dap4_name="Int8",
    dap4_wire_size=1,
)
DAP_INT16 = DapType(
    dap2_name="Int16",
    xdr_format=">i4",
    xdr_wire_size=4,
    numpy_dtype=np.dtype("int16"),
    dap4_name="Int16",
    dap4_wire_size=2,
)
DAP_UINT16 = DapType(
    dap2_name="UInt16",
    xdr_format=">u4",
    xdr_wire_size=4,
    numpy_dtype=np.dtype("uint16"),
    dap4_name="UInt16",
    dap4_wire_size=2,
)
DAP_INT32 = DapType(
    dap2_name="Int32",
    xdr_format=">i4",
    xdr_wire_size=4,
    numpy_dtype=np.dtype("int32"),
    dap4_name="Int32",
    dap4_wire_size=4,
)
DAP_UINT32 = DapType(
    dap2_name="UInt32",
    xdr_format=">u4",
    xdr_wire_size=4,
    numpy_dtype=np.dtype("uint32"),
    dap4_name="UInt32",
    dap4_wire_size=4,
)
DAP_INT64 = DapType(
    dap2_name="Float64",
    xdr_format=">f8",
    xdr_wire_size=8,
    numpy_dtype=np.dtype("int64"),
    dap4_name="Int64",
    dap4_wire_size=8,
)
DAP_UINT64 = DapType(
    dap2_name="Float64",
    xdr_format=">f8",
    xdr_wire_size=8,
    numpy_dtype=np.dtype("uint64"),
    dap4_name="UInt64",
    dap4_wire_size=8,
)
DAP_FLOAT32 = DapType(
    dap2_name="Float32",
    xdr_format=">f4",
    xdr_wire_size=4,
    numpy_dtype=np.dtype("float32"),
    dap4_name="Float32",
    dap4_wire_size=4,
)
DAP_FLOAT64 = DapType(
    dap2_name="Float64",
    xdr_format=">f8",
    xdr_wire_size=8,
    numpy_dtype=np.dtype("float64"),
    dap4_name="Float64",
    dap4_wire_size=8,
)
DAP_STRING = DapType(
    dap2_name="String",
    xdr_format=None,
    xdr_wire_size=0,
    numpy_dtype=np.dtype("O"),
    is_numeric=False,
    needs_xdr_length_doubled=False,
    dap4_name="String",
    dap4_wire_size=0,
)


# numpy dtype → DapType lookup (DAP2)
_NUMPY_TO_DAP2: dict[np.dtype, DapType] = {
    np.dtype("bool"): DAP_BYTE,
    np.dtype("uint8"): DAP_BYTE,
    np.dtype("int8"): DAP_INT16,
    np.dtype("int16"): DAP_INT16,
    np.dtype("uint16"): DAP_UINT16,
    np.dtype("int32"): DAP_INT32,
    np.dtype("uint32"): DAP_UINT32,
    np.dtype("int64"): DAP_FLOAT64,
    np.dtype("uint64"): DAP_FLOAT64,
    np.dtype("float32"): DAP_FLOAT32,
    np.dtype("float64"): DAP_FLOAT64,
}

# numpy dtype → DapType lookup (DAP4) — native types, no lossy conversions
_NUMPY_TO_DAP4: dict[np.dtype, DapType] = {
    np.dtype("bool"): DAP_BYTE,
    np.dtype("uint8"): DAP_BYTE,
    np.dtype("int8"): DAP_INT8,
    np.dtype("int16"): DAP_INT16,
    np.dtype("uint16"): DAP_UINT16,
    np.dtype("int32"): DAP_INT32,
    np.dtype("uint32"): DAP_UINT32,
    np.dtype("int64"): DAP_INT64,
    np.dtype("uint64"): DAP_UINT64,
    np.dtype("float32"): DAP_FLOAT32,
    np.dtype("float64"): DAP_FLOAT64,
}


def resolve_dap_type(  # noqa: PLR0911, PLR0912
    dtype: np.dtype,
    *,
    protocol: Literal["dap2", "dap4"] = "dap2",
) -> DapType:
    """Map a numpy dtype to the appropriate DapType for the given protocol.

    Args:
        dtype: The numpy dtype to map.
        protocol: Which DAP protocol version to target.

    Returns:
        The corresponding DapType.
    """
    # Handle string/object dtypes (same for both protocols)
    if dtype.kind in ("U", "S", "O"):
        return DAP_STRING

    # Handle datetime/timedelta (should already be CF-encoded, but fallback)
    if dtype.kind in ("m", "M"):
        return DAP_FLOAT64

    if protocol == "dap4":
        result = _NUMPY_TO_DAP4.get(dtype)
        if result is not None:
            return result

        # Fallback by kind and itemsize
        if dtype.kind == "f":
            return DAP_FLOAT64 if dtype.itemsize >= 8 else DAP_FLOAT32  # noqa: PLR2004
        if dtype.kind == "i":
            if dtype.itemsize >= 8:  # noqa: PLR2004
                return DAP_INT64
            if dtype.itemsize >= 4:  # noqa: PLR2004
                return DAP_INT32
            if dtype.itemsize >= 2:  # noqa: PLR2004
                return DAP_INT16
            return DAP_INT8
        if dtype.kind == "u":
            if dtype.itemsize >= 8:  # noqa: PLR2004
                return DAP_UINT64
            if dtype.itemsize >= 4:  # noqa: PLR2004
                return DAP_UINT32
            if dtype.itemsize >= 2:  # noqa: PLR2004
                return DAP_UINT16
            return DAP_BYTE

        raise ValueError(f"No DAP4 mapping for numpy dtype {dtype!r}")

    # DAP2
    result = _NUMPY_TO_DAP2.get(dtype)
    if result is not None:
        return result

    # Try to match by kind and itemsize for less common dtypes
    if dtype.kind == "f":
        return DAP_FLOAT64 if dtype.itemsize >= 8 else DAP_FLOAT32
    if dtype.kind in ("i", "u"):
        return DAP_FLOAT64 if dtype.itemsize >= 8 else DAP_INT32

    raise ValueError(f"No DAP2 mapping for numpy dtype {dtype!r}")


def cf_encode_variable(var: xr.Variable) -> xr.Variable:
    """Encode datetime/timedelta variables to numeric and normalize byte order.

    Only applies CF encoding for datetime64/timedelta64 types which have no
    direct DAP2 representation. Other types are passed through with their
    decoded xarray dtype preserved.

    Args:
        var: The xarray Variable to encode.

    Returns:
        A Variable with time types CF-encoded and native byte order.
    """
    if var.dtype.kind in ("M", "m"):
        # datetime64/timedelta64 → numeric via CF conventions
        encoded = xr.conventions.encode_cf_variable(var)
    else:
        encoded = var

    # Normalize byte order to native (little-endian on most systems)
    if encoded.dtype.byteorder not in ("=", "|", "<"):
        encoded = encoded.astype(encoded.dtype.newbyteorder("="), copy=False)

    return encoded
