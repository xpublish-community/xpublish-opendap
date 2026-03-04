"""DAP4 data response generation with chunked binary encoding.

The DAP4 data response consists of:
1. DMR XML as UTF-8 bytes
2. CRLF separator
3. Chunked binary data with native byte order and CRC32 checksums

Chunk header format (from libdap4/chunked_stream.h):
    Single uint32 in network (big-endian) byte order:
    type_flags | endian_flag | size

    CHUNK_DATA         = 0x00000000
    CHUNK_END          = 0x01000000
    CHUNK_ERR          = 0x02000000
    CHUNK_LITTLE_ENDIAN = 0x04000000
    CHUNK_SIZE_MASK    = 0x00FFFFFF
"""

from __future__ import annotations

import asyncio
import struct
import sys
import zlib
from collections.abc import AsyncIterator, Iterator

import numpy as np
import xarray as xr

from xpublish_opendap.dap.dap4.dmr import generate_dmr
from xpublish_opendap.dap.types import DapType, cf_encode_variable, resolve_dap_type
from xpublish_opendap.io import (
    SLAB_THRESHOLD_BYTES,
    _is_dask_graph_eligible,
    dask_graph_encode_data_bytes,
    get_slab_boundaries,
    load_variable,
    run_in_executor,
)

# Chunk type flags
CHUNK_DATA = 0x00000000
CHUNK_END = 0x01000000
CHUNK_ERR = 0x02000000
CHUNK_LITTLE_ENDIAN = 0x04000000
CHUNK_SIZE_MASK = 0x00FFFFFF

# Maximum data size per chunk (just under 16 MB to fit in 24-bit size field)
MAX_CHUNK_SIZE = CHUNK_SIZE_MASK

# CRLF separator between DMR and binary data
DMR_DATA_SEPARATOR = b"\r\n"


def _load_and_encode_dap4(
    da: xr.DataArray,
    dask_num_workers: int,
    endian_flag: int,
) -> bytes:
    """Load a DataArray and encode as DAP4 chunked binary.

    This is a synchronous function intended to be called in a thread pool
    executor so that both data loading and encoding happen off the event loop.

    Args:
        da: The (possibly lazy) DataArray to load and encode.
        dask_num_workers: Number of dask threads for parallel chunk loading.
        endian_flag: Endianness flag for chunk headers.

    Returns:
        Chunked DAP4 binary bytes (headers + data + CRC32) for this variable.
    """
    if _is_dask_graph_eligible(da):
        dap_type = resolve_dap_type(da.dtype, protocol="dap4")
        if not dap_type.is_string:
            wire_dtype = da.dtype.newbyteorder("=")
            raw_data = dask_graph_encode_data_bytes(da, wire_dtype, dask_num_workers)
            var_bytes = struct.pack("<Q", da.size) + raw_data
            return b"".join(_emit_data_chunks(var_bytes, endian_flag))

    load_variable(da, dask_num_workers)
    encoded_var = cf_encode_variable(da.variable)
    data = np.asarray(encoded_var.data)
    dap_type = resolve_dap_type(encoded_var.dtype, protocol="dap4")
    var_bytes = b"".join(_dap4_encode_variable(data, dap_type))
    return b"".join(_emit_data_chunks(var_bytes, endian_flag))


def _load_and_encode_slab_dap4(
    da: xr.DataArray,
    dim: str,
    start: int,
    stop: int,
    dask_num_workers: int,
) -> bytes:
    """Load a slab along the outermost dimension and encode as DAP4 native binary.

    Returns only the raw data bytes (no length prefix). The caller handles
    length prefix, chunk headers, and CRC32 accumulation.

    Args:
        da: The (lazy) DataArray to slice.
        dim: The outermost dimension name.
        start: Start index along dim.
        stop: Stop index along dim.
        dask_num_workers: Number of dask threads for parallel chunk loading.

    Returns:
        Native-byte-order encoded data bytes for this slab.
    """
    slab = da.isel({dim: slice(start, stop)})

    if _is_dask_graph_eligible(slab):
        wire_dtype = da.dtype.newbyteorder("=")
        return dask_graph_encode_data_bytes(slab, wire_dtype, dask_num_workers)

    load_variable(slab, dask_num_workers)
    encoded_var = cf_encode_variable(slab.variable)
    data = np.asarray(encoded_var.data)

    if data.ndim == 0:
        data = data.reshape(1)

    target_dtype = data.dtype.newbyteorder("=")
    native_data = data.astype(target_dtype, copy=False)
    return native_data.tobytes()


async def generate_dap4_data(
    ds: xr.Dataset,
    dataset_name: str,
    *,
    dask_num_workers: int = 4,
    slab_threshold_bytes: int = SLAB_THRESHOLD_BYTES,
) -> AsyncIterator[bytes]:
    """Yield the complete DAP4 data response as byte chunks.

    Loads data per-variable in a thread pool executor so that both data
    loading and encoding happen off the event loop.

    Structure:
        [DMR XML as UTF-8]
        CRLF
        [Chunk header + data for each variable with CRC32]
        [End chunk]

    Args:
        ds: The (subsetted, possibly lazy) xarray Dataset.
        dataset_name: The name for the Dataset declaration.
        dask_num_workers: Number of dask threads for parallel chunk loading.
        slab_threshold_bytes: Minimum estimated byte size to activate slab streaming.

    Yields:
        Chunks of the DAP4 data response as bytes.
    """
    # DMR from metadata — no data load needed
    dmr_text = generate_dmr(ds, dataset_name)
    yield dmr_text.encode("utf-8")

    # Yield separator
    yield DMR_DATA_SEPARATOR

    # Determine endianness flag
    endian_flag = CHUNK_LITTLE_ENDIAN if sys.byteorder == "little" else 0

    # Emit each variable (coords then data vars)
    all_vars = [(name, ds.coords[name]) for name in ds.coords]
    all_vars += [(name, ds[name]) for name in ds.data_vars]

    for _var_name, da in all_vars:
        slab_info = get_slab_boundaries(da, threshold_bytes=slab_threshold_bytes)

        if slab_info is not None:
            dim, boundaries = slab_info
            # Slab streaming path

            # Length prefix (uint64 LE, once)
            length_prefix = struct.pack("<Q", da.size)
            crc = zlib.crc32(length_prefix)

            # Load first slab
            s0, s1 = boundaries[0]
            pending = asyncio.ensure_future(
                run_in_executor(
                    _load_and_encode_slab_dap4,
                    da,
                    dim,
                    s0,
                    s1,
                    dask_num_workers,
                ),
            )

            for i, (_start, _stop) in enumerate(boundaries):
                slab_bytes = await pending
                # Prefetch next slab
                if i + 1 < len(boundaries):
                    s0, s1 = boundaries[i + 1]
                    pending = asyncio.ensure_future(
                        run_in_executor(
                            _load_and_encode_slab_dap4,
                            da,
                            dim,
                            s0,
                            s1,
                            dask_num_workers,
                        ),
                    )

                crc = zlib.crc32(slab_bytes, crc)

                if i == 0:
                    # First chunk includes the length prefix
                    chunk_data = length_prefix + slab_bytes
                else:
                    chunk_data = slab_bytes

                flags = CHUNK_DATA | endian_flag
                yield _chunk_header(flags, len(chunk_data))
                yield chunk_data

            # CRC32 after all slabs
            yield struct.pack("<I", crc & 0xFFFFFFFF)
        else:
            # Full-variable load path
            chunk_bytes: bytes = await run_in_executor(
                _load_and_encode_dap4,
                da,
                dask_num_workers,
                endian_flag,
            )
            yield chunk_bytes

    # End chunk
    yield _chunk_header(CHUNK_END, 0)


def _chunk_header(chunk_type: int, size: int) -> bytes:
    """Encode a chunk header as a big-endian uint32.

    The header format is: chunk_type | endian_flag | size
    where size is masked to 24 bits.

    Args:
        chunk_type: Chunk type flags (CHUNK_DATA, CHUNK_END, etc.)
        size: Data payload size in bytes (max 0x00FFFFFF).

    Returns:
        4-byte chunk header in network byte order.
    """
    header = chunk_type | (size & CHUNK_SIZE_MASK)
    return struct.pack(">I", header)


def _emit_data_chunks(data: bytes, endian_flag: int) -> Iterator[bytes]:
    """Split variable data into chunks and yield header+data+CRC32.

    Each variable's data is emitted as one or more CHUNK_DATA chunks.
    The CRC32 checksum covers the entire variable's data and is appended
    after the last chunk.

    Args:
        data: The complete encoded bytes for one variable.
        endian_flag: Endianness flag (CHUNK_LITTLE_ENDIAN or 0).

    Yields:
        Chunk header + data bytes, followed by 4-byte CRC32.
    """
    crc = zlib.crc32(data) & 0xFFFFFFFF
    offset = 0
    remaining = len(data)

    while remaining > 0:
        chunk_size = min(remaining, MAX_CHUNK_SIZE)
        chunk_data = data[offset : offset + chunk_size]

        flags = CHUNK_DATA | endian_flag
        yield _chunk_header(flags, chunk_size)
        yield chunk_data

        offset += chunk_size
        remaining -= chunk_size

    # CRC32 appended after the variable's data (little-endian per DAP4 spec)
    yield struct.pack("<I", crc)


def _dap4_encode_variable(data: np.ndarray, dap_type: DapType) -> Iterator[bytes]:
    """Encode a single variable's data in DAP4 native binary format.

    DAP4 encoding differences from DAP2/XDR:
    - Native byte order (little-endian on x86/ARM), no XDR big-endian
    - Array length prefix: 8-byte uint64, sent once (not doubled)
    - Int16/UInt16: natural 2-byte size (no widening to 4 bytes)
    - Byte arrays: no padding to 4-byte boundary
    - Strings: UTF-8, 8-byte length prefix, no padding

    Args:
        data: The numpy array to encode.
        dap_type: The DAP type for encoding.

    Yields:
        Encoded byte chunks.
    """
    if data.ndim == 0:
        data = data.reshape(1)

    n = data.size

    if dap_type.is_string:
        yield from _dap4_encode_string_array(data)
        return

    # Length prefix: uint64 sent once
    yield struct.pack("<Q", n)

    # Encode in native byte order at the type's natural size
    target_dtype = data.dtype.newbyteorder("=")
    native_data = data.astype(target_dtype, copy=False)
    yield native_data.tobytes()


def _dap4_encode_string_array(data: np.ndarray) -> Iterator[bytes]:
    """Encode a string array in DAP4 binary format.

    Each string is: 8-byte uint64 length + UTF-8 bytes (no padding).

    Args:
        data: The numpy string array.

    Yields:
        Encoded byte chunks.
    """
    n = data.size
    # Array length prefix: uint64
    yield struct.pack("<Q", n)

    for item in data.flat:
        s = str(item).encode("utf-8")
        yield struct.pack("<Q", len(s))
        yield s
