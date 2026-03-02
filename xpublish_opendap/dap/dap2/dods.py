"""DODS (DataDDS) binary response generation with XDR encoding.

The DODS response consists of:
1. DDS text as UTF-8 bytes
2. The separator b'\\nData:\\n'
3. XDR-encoded binary data for each variable
"""

from __future__ import annotations

import struct
from collections.abc import AsyncIterator, Iterator

import numpy as np
import xarray as xr

from xpublish_opendap.dap.dap2.dds import generate_dds
from xpublish_opendap.dap.types import (
    DAP_BYTE,
    DapType,
    cf_encode_variable,
    resolve_dap_type,
)

# Separator between DDS text and binary data per DAP2 spec v1.2
DATA_SEPARATOR = b"\nData:\n"


async def generate_dods(
    ds: xr.Dataset,
    dataset_name: str,
) -> AsyncIterator[bytes]:
    """Yield the complete DODS response as byte chunks.

    Structure:
        [DDS text as UTF-8]
        b'\\nData:\\n'
        [XDR-encoded binary data for each variable]

    Args:
        ds: The (subsetted, loaded) xarray Dataset.
        dataset_name: The name for the Dataset declaration.

    Yields:
        Chunks of the DODS response as bytes.
    """
    # Yield DDS text
    dds_text = "".join(generate_dds(ds, dataset_name))
    yield dds_text.encode("utf-8")

    # Yield separator
    yield DATA_SEPARATOR

    # Yield XDR-encoded data for each coordinate, then each data variable
    # Order: coordinates first (as declared in DDS), then data variables as Grids
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        encoded_var = cf_encode_variable(coord.variable)
        data = np.asarray(encoded_var.data)
        dap_type = resolve_dap_type(encoded_var.dtype)

        for chunk in _xdr_encode_array(data, dap_type):
            yield chunk

    for var_name in ds.data_vars:
        var = ds[var_name]
        encoded_var = cf_encode_variable(var.variable)
        data = np.asarray(encoded_var.data)
        dap_type = resolve_dap_type(encoded_var.dtype)

        # Grid encoding: main array first, then each map (coordinate) vector
        for chunk in _xdr_encode_array(data, dap_type):
            yield chunk

        # Map vectors for this Grid
        for dim in var.dims:
            if dim in ds.coords:
                dim_coord = ds.coords[dim]
                dim_encoded = cf_encode_variable(dim_coord.variable)
                dim_data = np.asarray(dim_encoded.data)
                dim_dap_type = resolve_dap_type(dim_encoded.dtype)
                for chunk in _xdr_encode_array(dim_data, dim_dap_type):
                    yield chunk


def _xdr_encode_array(data: np.ndarray, dap_type: DapType) -> Iterator[bytes]:
    """Encode a numpy array as XDR bytes.

    Rules:
    1. For atomic types: emit length as big-endian int32 TWICE, then values
    2. For Byte arrays: values packed contiguously, padded to 4-byte boundary
    3. For Int16/UInt16 arrays: each element occupies 4 bytes on wire
    4. For String arrays: each string individually length-prefixed + padded

    Args:
        data: The numpy array to encode.
        dap_type: The DAP type for encoding.

    Yields:
        XDR-encoded byte chunks.
    """
    if data.ndim == 0:
        # Scalar: encode as 1-element array
        data = data.reshape(1)

    n = data.size

    if dap_type.is_string:
        yield from _xdr_encode_string_array(data)
        return

    # Length prefix: sent twice for atomic arrays in DAP2
    length_bytes = _xdr_length_prefix(n)
    if dap_type.needs_xdr_length_doubled:
        yield length_bytes + length_bytes
    else:
        yield length_bytes

    if dap_type is DAP_BYTE:
        # Byte arrays: pack contiguously and pad to 4-byte boundary
        raw = data.astype(np.uint8).tobytes()
        yield raw
        padding = _pad_size(len(raw))
        if padding > 0:
            yield b"\x00" * padding
    elif dap_type.xdr_wire_size == 4 and data.dtype.itemsize < 4:
        # Int16/UInt16: each element widened to 4 bytes on wire
        if dap_type.dap2_name in ("Int16", "Int32"):
            wire_data = data.astype(">i4")
        else:
            wire_data = data.astype(">u4")
        yield wire_data.tobytes()
    else:
        # Standard encoding: convert to big-endian wire format
        wire_dtype = np.dtype(dap_type.xdr_format)
        wire_data = data.astype(wire_dtype)
        yield wire_data.tobytes()


def _xdr_encode_string_array(data: np.ndarray) -> Iterator[bytes]:
    """Encode a string array as XDR bytes.

    Each string is individually length-prefixed and padded to 4-byte boundary.

    Args:
        data: The numpy string array.

    Yields:
        XDR-encoded byte chunks.
    """
    n = data.size
    # String arrays get length prefix once
    yield _xdr_length_prefix(n)

    for item in data.flat:
        s = str(item).encode("utf-8")
        length = len(s)
        yield struct.pack(">I", length)
        yield s
        padding = _pad_size(length)
        if padding > 0:
            yield b"\x00" * padding


def _xdr_length_prefix(n: int) -> bytes:
    """Encode array length as big-endian int32 (4 bytes)."""
    return struct.pack(">I", n)


def _pad_size(length: int) -> int:
    """Calculate padding needed to reach 4-byte boundary."""
    remainder = length % 4
    return (4 - remainder) % 4
