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

import struct
import sys
import zlib
from collections.abc import AsyncIterator, Iterator

import numpy as np
import xarray as xr

from xpublish_opendap.dap.dap4.dmr import generate_dmr
from xpublish_opendap.dap.types import DapType, cf_encode_variable, resolve_dap_type

# Chunk type flags
CHUNK_DATA = 0x00000000
CHUNK_END = 0x01000000
CHUNK_ERR = 0x02000000
CHUNK_LITTLE_ENDIAN = 0x04000000
CHUNK_SIZE_MASK = 0x00FFFFFF

# Maximum data size per chunk (just under 16 MB to fit in 24-bit size field)
MAX_CHUNK_SIZE = CHUNK_SIZE_MASK

# CRLF separator between DMR and binary data
DMR_DATA_SEPARATOR = b'\r\n'


async def generate_dap4_data(
    ds: xr.Dataset,
    dataset_name: str,
) -> AsyncIterator[bytes]:
    """Yield the complete DAP4 data response as byte chunks.

    Structure:
        [DMR XML as UTF-8]
        CRLF
        [Chunk header + data for each variable with CRC32]
        [End chunk]

    Args:
        ds: The (subsetted, loaded) xarray Dataset.
        dataset_name: The name for the Dataset declaration.

    Yields:
        Chunks of the DAP4 data response as bytes.
    """
    # Yield DMR text
    dmr_text = generate_dmr(ds, dataset_name)
    yield dmr_text.encode('utf-8')

    # Yield separator
    yield DMR_DATA_SEPARATOR

    # Determine endianness flag
    endian_flag = CHUNK_LITTLE_ENDIAN if sys.byteorder == 'little' else 0

    # Encode and yield each coordinate variable
    for coord_name in ds.coords:
        coord = ds.coords[coord_name]
        encoded_var = cf_encode_variable(coord.variable)
        data = np.asarray(encoded_var.data)
        dap_type = resolve_dap_type(encoded_var.dtype, protocol='dap4')

        var_bytes = b''.join(_dap4_encode_variable(data, dap_type))
        for chunk in _emit_data_chunks(var_bytes, endian_flag):
            yield chunk

    # Encode and yield each data variable
    for var_name in ds.data_vars:
        var = ds[var_name]
        encoded_var = cf_encode_variable(var.variable)
        data = np.asarray(encoded_var.data)
        dap_type = resolve_dap_type(encoded_var.dtype, protocol='dap4')

        var_bytes = b''.join(_dap4_encode_variable(data, dap_type))
        for chunk in _emit_data_chunks(var_bytes, endian_flag):
            yield chunk

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
    return struct.pack('>I', header)


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
    yield struct.pack('<I', crc)


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
    yield struct.pack('<Q', n)

    # Encode in native byte order at the type's natural size
    target_dtype = data.dtype.newbyteorder('=')
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
    yield struct.pack('<Q', n)

    for item in data.flat:
        s = str(item).encode('utf-8')
        yield struct.pack('<Q', len(s))
        yield s
