# Design: Lazy Loading and Streaming for Cloud-Backed Datasets

> **Status:** Strategies 1 (variable pipelining) and 2 (slab streaming) are
> implemented. A dask-graph encoding fast path (not originally discussed here)
> was added post-design and provides the largest performance win. Strategy 3
> (dask-native async) was evaluated and rejected.

## Context

xpublish-opendap serves DAP2/DAP4 responses from xarray Datasets that may be
backed by cloud object storage (Zarr on S3, GCS, Arraylake, etc.). These stores
have high per-request latency (~30–100ms per chunk fetch) but high throughput
once streaming. A single variable slice may span dozens or hundreds of chunks.

This document explores the ideal data path from lazy xarray Dataset through to
streamed HTTP response bytes, given these storage characteristics.

## Current Architecture (as of 2026-03)

```
HTTP request
  │
  ▼
constraint parsing + plan_subsetting()     ← metadata only, instant
  │
  ▼
apply_plan() → lazy xr.Dataset            ← builds dask graph, no I/O
  │
  ▼
StreamingResponse(generate_dods(lazy_ds))
  │
  ▼
async generator iterates per-variable:
  ┌──────────────────────────────────────┐
  │  for each coord:                     │
  │    await run_in_executor(            │
  │      load_variable + xdr_encode      │   ← thread pool, dask // chunks
  │    )                                 │
  │    yield encoded_bytes               │
  │                                      │
  │  for each data_var:                  │
  │    await run_in_executor(            │
  │      load_variable + xdr_encode      │
  │    )                                 │
  │    yield encoded_bytes               │
  │    yield cached_coord_map_bytes      │
  └──────────────────────────────────────┘
```

**What works well:**
- DDS/DMR header streams immediately (metadata-only, no I/O)
- Loading + encoding happen in the thread pool, off the event loop
- Dask threaded scheduler parallelizes chunk fetches within a variable
- Coordinates are loaded once, cached for Grid Map re-emission
- Per-variable granularity keeps peak memory bounded

**Limitations addressed by the implementations below:**
- ~~Each variable is fully materialized before any of its bytes are yielded~~
  → Solved by slab streaming (Strategy 2) and dask-graph fast path
- ~~No pipelining~~ → Slab streaming uses prefetch=1 for inter-slab pipelining
- Backpressure from slow clients is implicit (generator blocks at yield)
  but doesn't propagate to dask — all chunks are fetched eagerly
- The compute semaphore (8 slots shared across all requests) can become a
  bottleneck under concurrent load

## Protocol Constraints

Understanding what the wire formats allow is critical before designing a
streaming strategy.

### DAP2 DODS

The binary section of a DODS response encodes each variable as:

```
[length: uint32 BE] [length: uint32 BE]    ← element count, sent twice
[data: N elements in row-major order, big-endian XDR]
```

The length prefix must be emitted before any data bytes. However, for a
subsetted lazy array, the element count is known from the dask graph metadata
(shape is determined at `.isel()` time). So we CAN emit the length prefix
before loading any data, then stream the encoded data bytes as chunks arrive.

**Constraint:** Data must be in row-major (C-contiguous) order. For a 2-D
array with dask chunks `(cy, cx)`, each chunk covers a rectangular tile, not a
contiguous row-major span. Streaming individual chunks directly would require
reordering. Only chunks aligned along the outermost dimension can be streamed
sequentially without buffering.

### DAP4

DAP4 uses a transport chunking layer:

```
[chunk_header: uint32 BE]  ← type flags | endian | size (24-bit)
[data: size bytes, native byte order]
[crc32: uint32 LE]
```

Each variable's data is split into one or more transport chunks (max ~16 MB).
The CRC32 covers the entire variable's data and is appended after the last
chunk.

**Constraint:** The CRC32 requires a pass over the full variable data. This
forces at least a two-pass approach (compute CRC while encoding, then emit) or
an in-memory buffer of the variable's encoded bytes. Streaming individual dask
chunks through DAP4 transport chunks would require per-chunk CRC, which
deviates from the spec.

## Possible Streaming Strategies

### Strategy 1: Variable-Level Pipelining (low complexity, moderate benefit)

> **Status:** Not implemented as a separate feature. The slab streaming
> implementation (Strategy 2) provides intra-variable pipelining via
> prefetch=1, which delivers a similar benefit for the common single-variable
> request pattern.

Pipeline the loading of variable N+1 while yielding variable N's bytes.

```python
async def generate_dods(ds, name, *, dask_num_workers=4):
    yield dds_text
    yield DATA_SEPARATOR

    # ... coordinates as before ...

    data_vars = list(ds.data_vars)
    if not data_vars:
        return

    # Kick off first variable
    pending = asyncio.ensure_future(
        run_in_executor(_load_and_encode_xdr, ds[data_vars[0]], dask_num_workers)
    )

    for i, var_name in enumerate(data_vars):
        current_bytes = await pending

        # Start next variable loading while we yield current
        if i + 1 < len(data_vars):
            pending = asyncio.ensure_future(
                run_in_executor(
                    _load_and_encode_xdr, ds[data_vars[i + 1]], dask_num_workers
                )
            )

        yield current_bytes
        for dim in ds[var_name].dims:
            if dim in coord_encoded:
                yield coord_encoded[dim]
```

**Benefit:** Loading of the next variable overlaps with the network send of the
current variable's bytes. For requests selecting multiple variables, this hides
one full load latency per variable (except the first and last).

**Cost:** Minimal code change. Peak memory increases by ~1 variable (two
variables in flight simultaneously).

**When it helps:** Multi-variable requests, which are less common since most
DAP clients request one variable at a time with a constraint expression.

### Strategy 2: Slab Streaming (moderate complexity, high benefit) — IMPLEMENTED

> **Status:** Implemented in `io.py` (`get_slab_boundaries`), `dods.py`
> (`_load_and_encode_slab_xdr`), and `data.py` (`_load_and_encode_slab_dap4`).
> Activates for dask-backed, non-string, non-datetime arrays above 32 MB with
> multiple chunks along at least one dimension.

Stream dask chunks individually within a variable, encoding each chunk's bytes
as they arrive from storage.

```
generate_dods yields:
  [DDS text]
  [separator]
  [length prefix]                    ← from metadata, no I/O
  [encoded chunk 0 bytes]            ← load chunk 0, encode, yield
  [encoded chunk 1 bytes]            ← load chunk 1, encode, yield
  ...
  [encoded chunk N bytes]
```

This requires careful handling of the row-major ordering constraint.

#### Case A: 1-D arrays (coordinates)

Straightforward. Chunks are contiguous segments. Each chunk's data can be
independently byte-swapped and yielded.

```python
async def _stream_1d_xdr(da, dask_num_workers):
    """Stream a 1-D dask array chunk by chunk."""
    encoded = cf_encode_variable(da.variable)
    dap_type = resolve_dap_type(encoded.dtype)
    n = da.size

    # Length prefix from metadata
    yield _xdr_length_prefix(n) + _xdr_length_prefix(n)

    # Yield each chunk as it loads
    dask_array = encoded.data
    for i in range(dask_array.numblocks[0]):
        chunk = await run_in_executor(_load_chunk, dask_array, i, dask_num_workers)
        yield _encode_chunk_bytes(chunk, dap_type)
```

#### Case B: N-D arrays

For an array with shape `(T, Y, X)` and chunks `(ct, cy, cx)`:

- Row-major order requires all of `[t=0, y=0, :]` before `[t=0, y=1, :]`
- If `cx == X` (chunks span the full innermost dimension), then each chunk
  along `(t, y)` is a contiguous row-major span → streamable
- If chunks split the innermost dimension, chunks must be concatenated across
  the inner axis before yielding

**Practical observation:** Zarr datasets stored for analysis typically chunk
along the outermost (time) dimension and leave spatial dimensions contiguous,
or use regular rectangular chunks. The IFS dataset chunks as
`(1, 721, 1440)` — one timestep per chunk, full spatial extent. This is the
ideal case for chunk-level streaming: each chunk is a complete time slice
that can be independently encoded.

**General approach:**

```python
async def _stream_nd_xdr(da, dask_num_workers):
    """Stream an N-D array by loading chunk-slabs along the outermost dim."""
    encoded = cf_encode_variable(da.variable)
    dap_type = resolve_dap_type(encoded.dtype)
    n = da.size

    yield _xdr_length_prefix(n) + _xdr_length_prefix(n)

    dask_array = encoded.data
    outer_chunks = dask_array.chunks[0]  # chunk sizes along dim 0

    offset = 0
    for chunk_size in outer_chunks:
        # Load one "slab" — all chunks for one outermost chunk index
        slab = da.isel({da.dims[0]: slice(offset, offset + chunk_size)})
        slab_bytes = await run_in_executor(
            _load_and_encode_slab, slab, dask_num_workers, dap_type
        )
        yield slab_bytes
        offset += chunk_size
```

This streams "slabs" (one outermost-dimension chunk group at a time). For the
IFS dataset with chunks `(1, 721, 1440)`, each slab is one timestep. For a
10-timestep request, this would yield 10 slabs instead of waiting for all 10
to load.

**Benefit:** TTFB scales with single-slab load time, not total variable load
time. Memory is bounded to one slab at a time. For 10 timesteps of IFS data
(~210 chunks), the first bytes arrive after ~1 chunk load (~30ms) instead of
after all 210 chunks (~810ms with 4 threads).

**Cost:** Significant complexity. Must handle:
- Determining slab boundaries from dask chunk structure
- Ensuring row-major ordering within slabs
- Falling back to full-variable load when chunk layout doesn't permit slab streaming
- DAP4 CRC32 must still cover the full variable (requires streaming CRC accumulation)

#### Chunk-level streaming with prefetch

To maximize throughput, combine chunk-level streaming with prefetch:

```python
async def _stream_nd_prefetch(da, dask_num_workers, prefetch=2):
    """Stream slabs with N-slab lookahead."""
    ...
    slabs = [da.isel(...) for ...]  # all slab specs (lazy)

    # Fill prefetch buffer
    pending = deque()
    for slab in slabs[:prefetch]:
        pending.append(asyncio.ensure_future(
            run_in_executor(_load_and_encode_slab, slab, ...)
        ))

    slab_idx = prefetch
    while pending:
        current = await pending.popleft()
        yield current

        if slab_idx < len(slabs):
            pending.append(asyncio.ensure_future(
                run_in_executor(_load_and_encode_slab, slabs[slab_idx], ...)
            ))
            slab_idx += 1
```

This keeps `prefetch` slabs loading in parallel while the current slab is
being sent to the client. With prefetch=2, at most 3 slabs are in memory
(1 being sent, 2 loading).

### Strategy 3: Dask-Native Async (evaluated, rejected)

Use dask's async/distributed capabilities directly instead of wrapping
synchronous loads in a thread pool.

```python
# Hypothetical — requires dask.distributed or dask async support
async def _stream_chunks_async(da):
    """Use dask's async API to stream chunks."""
    from dask.distributed import Client

    async with Client(asynchronous=True) as client:
        futures = client.compute(da.data.to_delayed().ravel())
        for future in as_completed(futures):
            chunk = await future
            yield encode(chunk)
```

**Benefit:** No thread pool overhead. Native async chunk fetching. Dask
handles scheduling and parallelism.

**Cost:** Requires `dask.distributed`, which is a heavy dependency. The
`as_completed` ordering may not match row-major order. Not suitable for the
threaded scheduler case (which is what most single-machine deployments use).

**Verdict:** Not practical for xpublish-opendap's target deployment. The
threaded scheduler + `run_in_executor` pattern is the right fit.

## Backpressure

With any streaming strategy, backpressure matters: if the client reads slowly,
we shouldn't eagerly load the entire dataset into memory.

**Current behavior:** The async generator `yield` naturally provides
backpressure. When Starlette calls `async for chunk in generator`, each
`yield` returns control to the ASGI send loop. If the TCP send buffer is full,
`await send(...)` blocks, which means the next `yield` in the generator is
delayed, which delays the next `await run_in_executor(...)` call. So loading
pauses when the client can't consume fast enough.

**With prefetch:** The prefetch buffer (Strategy 2) intentionally loads ahead.
Limiting the buffer size (e.g., prefetch=2) bounds memory growth from slow
clients to `prefetch × slab_size`.

**Recommendation:** Keep prefetch small (1–2 slabs). For the IFS dataset with
~4 MB per slab, this means at most ~12 MB of prefetch buffer per request.

## Memory Model

```
Peak memory per request (current):
  = max(variable_size for variable in request)
  ≈ size of largest data variable after constraint subsetting

Peak memory per request (slab streaming, prefetch=2):
  = (prefetch + 1) × slab_size + coord_cache_size
  ≈ 3 × (single_chunk_group_size) + small

Example (IFS t2m, 10 timesteps):
  Current:   10 × 721 × 1440 × 4 bytes = ~40 MB (all timesteps at once)
  Streaming: 3 × 721 × 1440 × 4 bytes  = ~12 MB (3 slabs in flight)
```

The `max_request_memory_bytes` check (currently 512 MB) validates the total
response size before streaming begins. With slab streaming, the actual peak
memory is much lower than this limit.

## Dask Scheduler Considerations

The choice of dask scheduler affects chunk-level streaming:

| Scheduler | Parallelism | Async-friendly | Notes |
|---|---|---|---|
| `synchronous` | None | N/A (blocks) | Default without dask.distributed. Chunks fetched one at a time. |
| `threads` | Thread pool | Via `run_in_executor` | Good for I/O-bound chunk fetches. Current choice. |
| `distributed` | Full cluster | Native async | Overkill for single-machine. Adds dependency. |

**Current setup:** `dask.config.set(scheduler='threads', num_workers=N)` inside
`run_in_executor`. This means:

1. An executor thread calls `var.load()`
2. Dask's threaded scheduler spawns N worker threads
3. Worker threads fetch chunks in parallel from cloud storage
4. When all chunks arrive, dask concatenates them into one numpy array
5. The executor thread returns the loaded array

**Key insight:** Dask's threaded scheduler parallelizes chunk fetches WITHIN a
single `.load()` call. But we can only observe the result after ALL chunks are
loaded. There's no callback or async interface to get individual chunks as they
arrive.

**For slab streaming:** Each slab is a separate `.load()` call on a smaller
slice. Dask parallelizes chunks within each slab. Slabs themselves are
pipelined via asyncio futures. This gives us two levels of parallelism:
- Intra-slab: dask threads fetch chunks in parallel
- Inter-slab: asyncio futures overlap loading of next slab with sending current

## What Was Implemented

### Slab streaming (Strategy 2)

Implemented in `io.py`, `dods.py`, and `data.py`. The implementation:

- **Detects slab eligibility** via `get_slab_boundaries()`: dask-backed,
  non-string, non-datetime/timedelta (`M`/`m` excluded — CF encoding picks
  data-dependent reference times per slab), multiple chunks along at least one
  dimension, estimated size above `SLAB_THRESHOLD_BYTES` (32 MB).
- **Chooses the dimension with the most dask chunks** (not necessarily the
  outermost dimension) for maximum streaming granularity.
- **Uses prefetch=1**: while the current slab is being sent, the next slab
  is loading in the thread pool.
- **For DAP4**, accumulates CRC32 incrementally across slabs using
  `zlib.crc32`'s running checksum parameter.

### Dask-graph encoding fast path (not in original design)

A significant additional optimization discovered during benchmarking. Instead
of `load_variable()` → encode, the fast path fuses dtype conversion into the
dask graph:

```python
flat = da.data.ravel()
flat = flat.rechunk(~20MB)
for block in flat.blocks:
    parts.append(block.astype(wire_dtype).compute().tobytes())
```

This is ~37% faster than the eager load path for large arrays (benchmarked on
IFS cloud data with 1740 chunks). The win comes from letting dask parallelize
chunk fetches more effectively within each ~20MB block.

Eligible when: dask-backed, ndim > 0, non-string (`U`/`S`/`O`), non-datetime
(`M`/`m`). Implemented in `io.py` (`_is_dask_graph_eligible`,
`dask_graph_encode_data_bytes`) and used by both slab and full-variable paths.

### Future considerations

- **Chunk-level streaming**: Slab streaming with the dask-graph fast path gets
  most of the benefit. True per-chunk streaming adds complexity for diminishing
  returns.
- **Caching**: Chunk-level LRU cache for repeated coordinate loads.
- **Connection pooling**: For cloud storage clients.

## Summary

| Strategy | TTFB | Peak memory | Complexity | Status |
|---|---|---|---|---|
| Per-variable (eager) | All chunks of var | 1 variable | Low | Implemented (fallback path) |
| Slab streaming + prefetch | 1 slab of chunks | ~2 slabs | Moderate | **Implemented** |
| Dask-graph fast path | 1 block (~20MB) | ~20 MB | Low | **Implemented** |
| Variable pipelining | All chunks of var | 2 variables | Low | Not implemented (slab streaming subsumes) |
| Chunk-level streaming | 1 chunk | ~3 chunks | High | Not implemented (diminishing returns) |
| Dask-native async | 1 chunk | ~3 chunks | High | Rejected (requires dask.distributed) |
