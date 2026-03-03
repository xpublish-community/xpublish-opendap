r"""DODS (DataDDS) binary response generation with XDR encoding.

The DODS response consists of:
1. DDS text as UTF-8 bytes
2. The separator b'\nData:\n'
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
from xpublish_opendap.io import load_variable, run_in_executor

# Separator between DDS text and binary data per DAP2 spec v1.2
DATA_SEPARATOR = b"\nData:\n"


def _load_and_encode_xdr(da: xr.DataArray, dask_num_workers: int) -> bytes:
    """Load a DataArray and XDR-encode it.

    This is a synchronous function intended to be called in a thread pool
    executor so that both data loading and encoding happen off the event loop.

    Args:
        da: The (possibly lazy) DataArray to load and encode.
        dask_num_workers: Number of dask threads for parallel chunk loading.

    Returns:
        XDR-encoded bytes for this variable.
    """
    load_variable(da, dask_num_workers)
    encoded_var = cf_encode_variable(da.variable)
    data = np.asarray(encoded_var.data)
    dap_type = resolve_dap_type(encoded_var.dtype)
    return b"".join(_xdr_encode_array(data, dap_type))


def _get_grid_map_coords(ds: xr.Dataset) -> set:
    """Identify coordinates used as Grid Maps of non-scalar data variables."""
    grid_map_coords: set = set()
    for var_name in ds.data_vars:
        var = ds[var_name]
        if var.ndim > 0:
            for dim in var.dims:
                if dim in ds.coords:
                    grid_map_coords.add(dim)
    return grid_map_coords


async def generate_dods(
    ds: xr.Dataset,
    dataset_name: str,
    *,
    dask_num_workers: int = 4,
) -> AsyncIterator[bytes]:
    r"""Yield the complete DODS response as byte chunks.

    Loads data per-variable in a thread pool executor so that both data
    loading and XDR encoding happen off the event loop. Coordinates are
    loaded first (typically small), then data variables stream one at a time.

    Structure:
        [DDS text as UTF-8]
        b'\nData:\n'
        [XDR-encoded binary data for each variable]

    Args:
        ds: The (subsetted, possibly lazy) xarray Dataset.
        dataset_name: The name for the Dataset declaration.
        dask_num_workers: Number of dask threads for parallel chunk loading.

    Yields:
        Chunks of the DODS response as bytes.
    """
    # DDS from metadata — no data load needed
    dds_text = "".join(generate_dds(ds, dataset_name))
    yield dds_text.encode("utf-8")
    yield DATA_SEPARATOR

    grid_map_coords = _get_grid_map_coords(ds)

    # Load + encode all coordinates in executor (typically small 1-D arrays).
    # Cache the encoded bytes since Grid Maps are re-emitted per data variable.
    coord_encoded: dict = {}
    for coord_name in ds.coords:
        coord_bytes: bytes = await run_in_executor(
            _load_and_encode_xdr, ds.coords[coord_name], dask_num_workers
        )
        coord_encoded[coord_name] = coord_bytes

    # Yield orphan coordinates (not referenced as Grid Maps)
    for coord_name in ds.coords:
        if coord_name not in grid_map_coords:
            yield coord_encoded[coord_name]

    # Yield data variables as Grids — load + encode each in executor
    for var_name in ds.data_vars:
        var = ds[var_name]
        var_bytes: bytes = await run_in_executor(
            _load_and_encode_xdr, var, dask_num_workers
        )
        yield var_bytes

        # Map vectors from cache (already loaded + encoded above)
        for dim in var.dims:
            if dim in ds.coords:
                yield coord_encoded[dim]


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
    # All non-string types double the length prefix in DAP2 XDR encoding
    yield length_bytes + length_bytes

    if dap_type is DAP_BYTE:
        # Byte arrays: pack contiguously and pad to 4-byte boundary
        raw = data.astype(np.uint8).tobytes()
        yield raw
        padding = _pad_size(len(raw))
        if padding > 0:
            yield b"\x00" * padding
    elif dap_type.xdr_wire_size == 4 and data.dtype.itemsize < 4:  # noqa: PLR2004
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
