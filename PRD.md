# xpublish-opendap v2: PRD

## Context

xpublish-opendap is an xpublish plugin that serves xarray Datasets via the
OPeNDAP protocol. It is deployed as part of **Flux**, Earthmover's cloud-native
API gateway built on xpublish. Today, the plugin provides partial DAP2 support
(DDS, DAS, DODS responses only) atop the `opendap-protocol` library (~570 lines
of Python). DAP endpoints are available in Flux's autoscaling xpublish
deployment but are not widely used because the implementation has significant
gaps that cause failures with modern DAP clients (netCDF4-python, pydap, xarray,
NCO).

This PRD defines a major upgrade to make xpublish-opendap a production-quality
OPeNDAP server that correctly interoperates with the DAP client ecosystem.

---

## Objectives

### O1 — Complete DAP2 Conformance

Bring the DAP2 implementation into full conformance with ESE-RFC-004.1.2 ("The
Data Access Protocol — DAP 2.0"). The current implementation is missing required
response types, required HTTP headers, complete constraint expression support,
and several data types.

### O2 — DAP4 Support

Add support for the DAP4 protocol as specified at
https://opendap.github.io/dap4-specification/DAP4.html. DAP4 is the current
version of the protocol and is required by modern clients (pydap with
`protocol='dap4'`, netCDF4-python with `dap4://` URLs). DAP4 introduces a
unified metadata response (DMR), chunked binary encoding with checksums, groups,
64-bit integers, and a richer constraint expression language.

### O3 — Async I/O

All route handlers must be `async def`. Data loading from xarray must use
`Dataset.load_async()` / `DataArray.load_async()` with structured concurrency
(TaskGroup + semaphore limiting), following the patterns established in
xpublish-tiles. Blocking operations (encoding, checksums) must be offloaded to a
thread pool via `loop.run_in_executor()` with semaphore-bounded concurrency.

### O4 — Memory Management

Requests that would require loading more than a configurable memory threshold
must be rejected or streamed in bounded-memory chunks. The server must never
buffer an entire large variable in memory when serving a DODS/DAP4 data
response. Constraint-based subsetting must be applied *before* data loading, not
after.

### O5 — Comprehensive Test Suite

A test suite covering correctness, async behavior, performance, and integration
with downstream clients (pydap, xarray/netCDF4, xarray/pydap, NCO/ncdump).

---

## Current State Analysis

### What exists today (xpublish-opendap + opendap-protocol)

| Aspect | Status | Detail |
|--------|--------|--------|
| DAP2 DDS response | Partial | Works for Arrays and Grids. No Structure or Sequence. |
| DAP2 DAS response | Partial | Works. Attribute quote escaping implemented. |
| DAP2 DODS response | Partial | Binary encoding works for numeric arrays. String arrays untested in integration. |
| DAP2 Error response | Missing | No structured error responses. |
| DAP2 Version response | Missing | Required by spec. |
| DAP2 Help response | Missing | Required by spec. |
| DAP2 ASCII response | Missing | Not required by spec but expected by many clients. |
| HTTP headers | Missing | `Content-Description`, `XDODS-Server`, `Last-Modified` headers not set. |
| Constraint expressions | Minimal | Field projection and index slicing work. No selection (filtering), no server-side functions. |
| Type coverage | Partial | 9 atomic types. int64→Float64 is lossy. No bool, complex, or object mapping. |
| Structure type | Missing | No support in opendap-protocol or plugin. |
| Sequence type | Stub | Classes exist in opendap-protocol but validation is a no-op TODO. |
| DAP4 | None | No DMR, no chunked encoding, no groups, no 64-bit integers. |
| Async | None | All handlers are synchronous `def`, blocking the event loop. |
| Memory management | None | Entire xarray Dataset converted to DAP objects eagerly and cached permanently (TTL=99999). Constraints applied post-load. |
| Caching | Naive | Full DAP Dataset object cached per dataset_id. No invalidation. No constraint-aware caching. |
| Test coverage | Minimal | 281 lines. Basic DDS/DAS/DODS content checks. One integration test with netCDF4. No async tests, no performance tests, no NCO/pydap integration tests. |

### Key architectural problems

1. **opendap-protocol materializes everything eagerly.** `dods_encode()` calls
   `.tobytes()` on full arrays. For a 10GB variable this is fatal.

2. **Constraint subsetting happens inside opendap-protocol after full
   conversion.** The xarray→DAP conversion in `dap_xarray.py` loads all
   variables regardless of the constraint.

3. **The DAP object model is the wrong caching unit.** Caching the converted DAP
   Dataset means every unique dataset_id holds a full copy of all variable data
   in memory as Python objects — in addition to whatever xarray/dask already
   holds.

4. **No streaming.** `StreamingResponse` is used but the generators it wraps
   still materialize full arrays in `dods_encode()`.

---

## Architecture Decision: Build New Protocol Layer vs. Bind libdap4

### Option A: New Python protocol implementation

Replace `opendap-protocol` with a new library purpose-built for streaming,
async-compatible DAP2+DAP4 encoding. The library operates directly on xarray
objects (or numpy/dask arrays) without an intermediate DAP object model.

**Pros:**
- Native Python, no C++ build complexity
- Can stream chunks directly from dask graph without materializing
- Natural async integration
- Full control over encoding correctness
- Zero deployment friction (pure Python wheel)

**Cons:**
- Must implement and maintain XDR encoding (DAP2) and chunked binary encoding
  (DAP4) ourselves
- Must implement constraint expression parsing ourselves
- Risk of subtle encoding bugs vs. reference implementation

### Option B: Python bindings for libdap4

Use pybind11/cffi to wrap the C++ libdap4 library for encoding and constraint
parsing, with a Python layer for async I/O and xarray integration.

**Pros:**
- Reference implementation, proven correct
- Mature constraint expression parser (Bison/Flex)
- Both DAP2 and DAP4 already implemented

**Cons:**
- Complex build (CMake + autotools, Bison/Flex, libxml2, libcurl dependencies)
- Difficult to distribute (no pure-Python wheel; needs conda or system packages)
- C++ library is synchronous and not designed for streaming — would need to
  buffer data anyway, negating the memory management objective
- Tight coupling to C++ object model makes xarray integration awkward
- libdap4 is oriented toward client-side use; server-side encoding is in BES,
  not libdap4

### Recommendation: Option A

The primary constraints of this project — streaming, async I/O, bounded memory,
and easy deployment in a cloud-native Python stack — all favor a native Python
implementation. The XDR encoding for DAP2 is straightforward (the existing
`opendap-protocol` `dods_encode` is ~30 lines). The DAP4 binary format is
simpler than XDR (native byte order, no padding). The hard part is constraint
expression parsing, but for DAP2 the grammar is small enough for a hand-written
recursive-descent parser, and for DAP4 we can use a parser generator (lark or
similar).

The libdap4 reference implementation remains valuable as a *test oracle* — we
can compare our encoding output against libdap4's for correctness validation.

---

## Scope and Phasing

### Phase 1: Foundation (DAP2 conformance + async + memory)

**Goal:** A correct, async, memory-safe DAP2 server that works with netCDF4-python,
xarray, pydap, and NCO.

#### 1.1 New protocol layer (in-repo)

Replace the `opendap-protocol` dependency with a clean-room protocol
implementation that lives inside `xpublish-opendap` (under
`xpublish_opendap/dap/`). This is not a separate package — it ships as part of
`xpublish-opendap`. The new protocol layer:

- **Encodes DAP2 responses as async generators** — `async def dds(dataset, constraint)`,
  `async def das(dataset, constraint)`, `async def dods(dataset, constraint)` each
  yield `bytes` chunks.
- **Operates directly on xarray objects** — no intermediate DAP object model. The
  encoder walks the xarray Dataset structure and emits protocol bytes.
- **Parses DAP2 constraint expressions** — recursive-descent parser supporting:
  - Field projection (variable selection)
  - Hyperslab projection (`[start:stride:stop]`)
  - Selection clauses on Sequences (`&field>value`)
  - Server-side functions (extensibility hook, not required for MVP)
- **Applies constraints before loading** — the constraint parser produces an
  xarray-native subsetting plan (variable selection + `.isel()` slices) that is
  applied to the lazy Dataset *before* any `.load()` / `.compute()`.
- **Streams XDR encoding in bounded chunks** — for large arrays, encode and yield
  fixed-size blocks (e.g. 2MB) rather than materializing the full encoded array.
- **Complete type mapping:**

  | numpy dtype | DAP2 type | Notes |
  |-------------|-----------|-------|
  | bool | Byte | 0/1 |
  | int8 | Int16 | DAP2 has no Int8 |
  | uint8 | Byte | |
  | int16 | Int16 | |
  | uint16 | UInt16 | |
  | int32 | Int32 | |
  | uint32 | UInt32 | |
  | int64 | Float64 | Lossy but only option in DAP2 |
  | uint64 | Float64 | Lossy but only option in DAP2 |
  | float32 | Float32 | |
  | float64 | Float64 | |
  | datetime64 | Float64 | CF-encoded via `cftime` |
  | timedelta64 | Float64 | CF-encoded |
  | bytes/object(str) | String | |

- **XDR encoding correctness:** Array length prefix sent twice for atomic types,
  once for constructor types. Byte arrays padded to 4-byte boundary. Int16/UInt16
  encoded as 32-bit on the wire. DataDDS separator is `\nData:\n` (LF, not CRLF).
- **All six DAP2 response types:**
  - DDS (required)
  - DAS (required)
  - DataDDS / DODS (required)
  - Error (required) — structured error with code + message
  - Version (required) — DAP version, server version
  - Help (required)
- **Generates proper HTTP headers:** `Content-Description`, `XDODS-Server`,
  `Content-Type`, `Last-Modified`.
- **Structure and Grid types** fully supported. Sequence support deferred (see
  TODO.txt).

#### 1.2 Plugin rewrite

Rewrite `xpublish_opendap/plugin.py` with:

- **Async route handlers** — all endpoints use `async def`.
- **Async data loading** — use `DataArray.load_async()` within
  `asyncio.TaskGroup`, bounded by a configurable semaphore
  (`num_concurrent_data_loads`).
- **Memory threshold enforcement** — before loading, estimate the memory
  required (from dask graph or `.nbytes`). If it exceeds a configurable limit
  (default: 512MB per request), return a DAP Error response
  (`413 Payload Too Large` or DAP Error with descriptive message).
- **Constraint-first pipeline:**
  1. Parse constraint expression
  2. Select variables and compute isel slices from constraint
  3. Apply subsetting to lazy xarray Dataset (no compute yet)
  4. Estimate memory of subsetted result
  5. Load data async (respecting memory threshold)
  6. Stream-encode response
- **Streaming responses** — use `StreamingResponse` with an async generator that
  yields encoded chunks, ensuring the full response is never buffered in memory.
- **Remove DAP-object caching** — the dataset_id→DAP-Dataset cache is removed.
  Metadata (DDS/DAS) is cheap to regenerate. Data responses must not be cached
  in full.
- **Configurable via plugin attributes** (Pydantic fields on Plugin subclass):
  - `max_request_memory_bytes: int = 512 * 1024 * 1024`
  - `num_concurrent_data_loads: int = 4`
  - `encoding_chunk_size: int = 2 * 1024 * 1024`
  - `async_load_timeout: float = 30.0`

#### 1.3 DAP2 HTTP compliance

- Set `Content-Description` header on DDS (`dods-dds`), DAS (`dods-das`), DODS
  (`dods-data`), Error (`dods-error`) responses.
- Set `XDODS-Server: dods/2.0` header on all responses.
- Set `Last-Modified` header on all responses.
- Return proper `Content-Type`: `text/plain` for DDS/DAS/Error/Version,
  `application/octet-stream` for DODS, `text/html` for Help.
- Return DAP Error responses (not HTTP 500) for constraint syntax errors,
  unknown variables, out-of-range indices.

### Phase 2: DAP4

**Goal:** Full DAP4 protocol support, enabling pydap (`protocol='dap4'`),
netCDF4-python (`dap4://` URLs), and xarray DAP4 access.

#### 2.1 DAP4 encoding (in-repo protocol layer)

- **DMR (Dataset Metadata Response)** — generate XML conforming to the DAP4 RELAX
  NG schema. Walk xarray Dataset or `DataTree` and emit:
  - `<Dataset>` root with `name`, `dap4.0` namespace
  - `<Dimension>` declarations (shared named dimensions)
  - `<Group>` hierarchy (from DataTree)
  - Variable declarations with DAP4 types, dimensions, attributes
  - `<Map>` elements for coordinate variables
  - `<Enum>` type definitions where applicable
- **DAP4 type system:**

  | numpy dtype | DAP4 type | Notes |
  |-------------|-----------|-------|
  | int8 | Int8 | Native support in DAP4 |
  | uint8 | UInt8 | |
  | int16 | Int16 | |
  | uint16 | UInt16 | |
  | int32 | Int32 | |
  | uint32 | UInt32 | |
  | int64 | Int64 | Native 64-bit — no lossy conversion |
  | uint64 | UInt64 | |
  | float32 | Float32 | |
  | float64 | Float64 | |
  | bytes | Opaque | |
  | str/object | String | UTF-8 |

- **DAP4 binary serialization:**
  - Chunked format: 4-byte chunk size + 1-byte type flag + payload
  - "Reader Make Right" byte order (server native, little-endian on all our
    deployment targets)
  - Byte-order indicated by `_DAP4_Little_Endian` attribute in DMR
  - No XDR padding — values serialized at natural sizes
  - Strings: 4-byte length prefix + UTF-8 bytes (no null terminator)
  - Arrays: 4-byte element count + row-major values
  - Structures: fields in declaration order, no length prefix
  - Sequences: 4-byte record count + records
  - CRC32 checksum per top-level variable
  - End chunk: type=0xFF, size=0
- **DAP4 data response structure:**
  ```
  [DMR XML, UTF-8]
  [separator]
  [chunk: variable 1 data + CRC32]
  [chunk: variable 2 data + CRC32]
  ...
  [end chunk]
  ```
- **DAP4 constraint expressions:**
  - Field subsetting: `{var1;var2;structure.field}`
  - Index subsetting: `variable[start:stride:stop]`
  - Filtering: `sequence|field>value`
  - Group-qualified names: `/GroupA/variable`
  - Shared dimension subsetting
- **DAP4 error response** — structured XML error document.

#### 2.2 Plugin DAP4 routes

- `GET .dmr` — returns DMR XML (`Content-Type: application/vnd.opendap.dap4.dataset-metadata+xml`)
- `GET .dap` — returns DAP4 data response (`Content-Type: application/vnd.opendap.dap4.data`)
- `GET .dsr` — returns Dataset Services Response (capabilities document)
- `GET .dap4.errs` — DAP4 error (returned in place of any response on error)
- Shared constraint parsing with DAP2 where syntax overlaps; separate parser
  entry point for DAP4-specific syntax (brace-delimited field lists, pipe
  filters).
- **DataTree/group support is required.** If the dataset is an `xr.DataTree`,
  emit groups in DMR and serialize group variables. This is not optional — DAP4
  groups map directly to DataTree and modern clients expect group support.

#### 2.3 Content negotiation and URL routing

- Support both URL-extension routing (`.dds`, `.das`, `.dods`, `.dmr`, `.dap`)
  and Accept-header content negotiation.
- Support `dap4://` and `dap2://` URL scheme prefixes (these are conventions used
  by clients to signal protocol version — the server sees them as regular HTTP
  after the client rewrites the scheme).
- Auto-detect protocol version from request context when possible.

### Phase 3: Test Suite

#### 3.1 Unit tests — protocol encoding

- XDR encoding of every atomic type (compare against `struct.pack` / known byte
  sequences).
- XDR array encoding: length-prefix doubling for atomic types, padding for byte
  arrays, Int16/UInt16 as 32-bit.
- DAP4 binary encoding of every type.
- Chunked encoding: chunk boundaries, CRC32 checksums, end markers.
- DMR XML generation: validate against RELAX NG schema.
- DDS/DAS string generation: exact format matching against reference outputs.
- Constraint expression parsing: valid expressions, edge cases, malformed input.
- Constraint-to-isel conversion: verify correct slicing semantics (inclusive
  end → Python exclusive end, stride handling).

#### 3.2 Unit tests — plugin behavior

- Async route handlers: verify all handlers are coroutines.
- Memory threshold: mock a dataset exceeding the limit, verify rejection.
- Constraint-first loading: mock dask compute, verify `.isel()` applied before
  `.compute()`.
- Streaming: verify response is generated incrementally (not buffered).
- Error responses: constraint syntax errors, unknown variables, out-of-range
  slices.
- HTTP headers: verify `Content-Description`, `XDODS-Server`, `Last-Modified`
  on every response type.

#### 3.3 Integration tests — client compatibility

For each client, open a known test dataset via OPeNDAP and verify round-trip
correctness (values, dtypes, dimensions, attributes, coordinate variables):

- **netCDF4-python** (`netCDF4.Dataset("http://...")`) — DAP2
- **netCDF4-python** (`netCDF4.Dataset("dap4://...")`) — DAP4
- **xarray + netCDF4 engine** (`xr.open_dataset(url, engine='netcdf4')`)
- **xarray + pydap engine** (`xr.open_dataset(url, engine='pydap')`)
- **pydap client** (`pydap.client.open_url(url)` and `open_url(url, protocol='dap4')`)
- **NCO** (`ncdump -h url`, `ncks -v var url output.nc`) — validates C-library
  compatibility
- Test with constrained requests: single variable, hyperslab, strided access
- Test with datasets containing: empty dimensions, scalar variables, large
  attributes, special characters in names, NaN/Inf values, unsigned types,
  string arrays

#### 3.4 Async correctness tests

- Verify concurrent requests are handled without blocking (launch N requests,
  assert wall time < N × single-request time).
- Verify semaphore limiting: launch more requests than `num_concurrent_data_loads`,
  verify at most N are loading simultaneously.
- Verify timeout: mock a slow `.load_async()`, verify timeout triggers DAP Error.

#### 3.5 Performance / memory tests

- Serve a large synthetic dataset (e.g. 1GB variable). Request a small slice via
  constraint. Assert peak memory stays below threshold (use `tracemalloc` or
  `/proc/self/status`).
- Serve a dataset with many variables. Request a single variable. Assert only
  that variable's data is loaded (mock and assert `.compute()` call count).
- Benchmark: measure throughput (MB/s) for a constrained DODS request on a
  ~100MB variable. Establish baseline for regression tracking.

---

## Non-Goals

- **Client-side DAP** — this project is server-side only. Client functionality is
  handled by netCDF4-python, pydap, and xarray.
- **Aggregation / virtual datasets** — dataset assembly is xpublish's concern,
  not this plugin's.
- **Server-side functions beyond extensibility hooks** — DAP2 server-side
  functions and DAP4 function extensions are not in scope for initial delivery.
  The architecture must not preclude adding them later.
- **DAP4 async response extension** (HTTP 202 / polling) — not needed for
  synchronous xpublish request model.
- **HTML / human-browsable dataset interface** — nice to have but not a core
  objective. The Help response satisfies the spec requirement.
- **DAP2 Sequence type** — deferred to a future phase. The architecture must
  support adding Sequences later, but encoding, selection filtering, and the
  0x5A/0xA5 wire markers are out of scope for the initial delivery (see
  TODO.txt).

---

## Success Criteria

1. **netCDF4-python round-trip**: `netCDF4.Dataset(url)` and
   `netCDF4.Dataset("dap4://" + url)` both return data identical to the source
   xarray Dataset for all supported dtypes.
2. **xarray round-trip**: `xr.open_dataset(url, engine='netcdf4')` and
   `xr.open_dataset(url, engine='pydap')` both return data identical to source.
3. **NCO compatibility**: `ncdump -h url` produces valid CDL; `ncks` can extract
   variables.
4. **Memory bounded**: serving a 1GB variable with a 100MB slice constraint uses
   <200MB peak memory.
5. **Async**: 10 concurrent constrained requests to a dask-backed dataset
   complete in <2× the time of a single request (on a multi-core machine).
6. **All DAP2 required responses** pass a conformance check (DDS, DAS, DODS,
   Error, Version, Help with correct headers).
7. **DAP4 DMR validates** against the RELAX NG schema.
8. **DAP4 data response** is correctly decoded by pydap and netCDF4-python.

---

## Resolved Decisions

1. **Sequence support:** Deferred to a future phase. Tracked in TODO.txt. The
   architecture must not preclude adding Sequences later.

2. **opendap-protocol:** Clean-room replacement. The existing `opendap-protocol`
   library is not forked or contributed to — we build fresh to support streaming,
   async, and DAP4 from the ground up.

3. **Package boundaries:** The new protocol layer lives inside
   `xpublish-opendap` (as `xpublish_opendap/dap/`), not as a separate package.
   This avoids release coordination overhead and keeps the protocol tightly
   coupled to the plugin's needs.

4. **DataTree / group support:** Required in Phase 2. DAP4 groups map to
   `xr.DataTree` and modern clients expect group support.

5. **CF encoding:** The plugin handles CF encoding internally
   (`xr.conventions.encode_cf_variable()`). This keeps the API clean for callers
   — they pass a normal xarray Dataset and the plugin ensures wire-compatible
   encoding.
