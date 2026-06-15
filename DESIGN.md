# xpublish-opendap v2: Technical Design

This document describes the architecture and implementation of the
xpublish-opendap rewrite (Phase 1: DAP2 conformance + async + memory
management, Phase 2: DAP4). See PRD.md for requirements.

---

## 1. Directory Layout

```
xpublish_opendap/
├── __init__.py                      # Public API: OpenDapPlugin
├── _version.py                      # Package version
├── plugin.py                        # Plugin class, async route handlers
├── errors.py                        # DAP error types, error response helpers
├── io.py                            # Async infrastructure (executor, semaphores,
│                                    #   slab streaming, dask-graph fast path)
├── dap/                             # Protocol layer (clean-room, no opendap-protocol)
│   ├── __init__.py
│   ├── types.py                     # numpy↔DAP type mapping, CF encoding
│   ├── constraint.py                # Constraint expression parser (DAP2 + DAP4)
│   ├── xarray_adapter.py            # Constraint → xarray subsetting plan
│   ├── dap2/
│   │   ├── __init__.py
│   │   ├── dds.py                   # DDS text generation
│   │   ├── das.py                   # DAS text generation
│   │   ├── dods.py                  # DODS binary (XDR) encoding + slab streaming
│   │   ├── responses.py             # Error, Version, Help response generators
│   │   └── headers.py               # DAP2 HTTP header constants
│   └── dap4/
│       ├── __init__.py
│       ├── dmr.py                   # DMR XML generation
│       ├── data.py                  # DAP4 chunked binary encoding + slab streaming
│       ├── responses.py             # DSR, DAP4 error responses
│       └── headers.py               # DAP4 HTTP header / content-type constants
tests/
├── conftest.py                      # Shared fixtures (datasets, ASGI clients)
├── server.py                        # Test server helper
├── test_types.py                    # Type mapping unit tests
├── test_constraint.py               # Constraint parser unit tests
├── test_xarray_adapter.py           # Constraint → isel tests
├── test_dap2_responses.py           # DDS/DAS/DODS + Error/Version/Help tests
├── test_dap4_dmr.py                 # DMR XML tests
├── test_dap4_data.py                # DAP4 binary encoding tests
├── test_dap4_headers.py             # DAP4 header tests
├── test_dap4_routes.py              # DAP4 route handler tests
├── test_plugin_routes.py            # Plugin route tests (headers, errors)
├── test_opendap_router.py           # Router registration tests
├── test_io.py                       # Async infrastructure tests
├── test_async_streaming.py          # Streaming response tests
├── test_slab_streaming.py           # Slab streaming + dask-graph fast path tests
├── test_server.py                   # Server lifecycle tests
├── test_init.py                     # Package init tests
└── integration/
    ├── conftest.py                  # Live server fixture
    ├── test_httpx_raw.py            # Raw HTTP round-trip
    ├── test_netcdf4_client.py       # netCDF4-python round-trip
    ├── test_pydap_client.py         # pydap client round-trip
    ├── test_xarray_engines.py       # xarray engine round-trip
    ├── test_ncdump.py               # ncdump compatibility
    ├── test_ncks.py                 # ncks compatibility
    ├── test_julia_client.py         # Julia NCDatasets.jl round-trip
    └── test_r_client.py             # R ncdf4 round-trip
```

---

## 2. Module Specifications

### 2.1 `dap/types.py` — Type System

This module is the single source of truth for mapping between numpy dtypes and
DAP types, for both DAP2 and DAP4.

```python
@dataclass(frozen=True)
class DapType:
    """A DAP type descriptor."""
    dap2_name: str              # "Float64", "Int32", "Byte", "String", etc.
    dap4_name: str | None       # "Int64", "UInt64", "Opaque", None if no DAP4 analog
    xdr_format: str             # struct format for big-endian XDR: '>f8', '>i4', 'B', etc.
    xdr_wire_size: int          # bytes per element on the wire (after XDR padding)
    numpy_dtype: np.dtype       # canonical numpy dtype for this DAP type
    is_string: bool = False     # True for String type (different wire encoding)
```

**Key type table** (populated as module-level constants):

| numpy dtype | `DapType` | dap2_name | xdr_format | xdr_wire_size | Notes |
|---|---|---|---|---|---|
| `bool_` | `DAP_BYTE` | Byte | `B` | 1 | encode as 0/1, pad array to 4-byte boundary |
| `uint8` | `DAP_BYTE` | Byte | `B` | 1 | pad array to 4-byte boundary |
| `int8` | `DAP_INT16` | Int16 | `>i4` | 4 | no Int8 in DAP2; stored as 32-bit XDR |
| `int16` | `DAP_INT16` | Int16 | `>i4` | 4 | 16-bit value, but 32-bit on wire |
| `uint16` | `DAP_UINT16` | UInt16 | `>u4` | 4 | 16-bit value, but 32-bit on wire |
| `int32` | `DAP_INT32` | Int32 | `>i4` | 4 | |
| `uint32` | `DAP_UINT32` | UInt32 | `>u4` | 4 | |
| `int64` | `DAP_FLOAT64` | Float64 | `>f8` | 8 | lossy in DAP2; Int64 in DAP4 |
| `uint64` | `DAP_FLOAT64` | Float64 | `>f8` | 8 | lossy in DAP2; UInt64 in DAP4 |
| `float32` | `DAP_FLOAT32` | Float32 | `>f4` | 4 | |
| `float64` | `DAP_FLOAT64` | Float64 | `>f8` | 8 | |
| `datetime64` | `DAP_FLOAT64` | Float64 | `>f8` | 8 | CF-encoded first |
| `timedelta64` | `DAP_FLOAT64` | Float64 | `>f8` | 8 | CF-encoded first |
| `str_`/`object` | `DAP_STRING` | String | — | variable | length-prefixed on wire |

**Functions:**

```python
def resolve_dap_type(dtype: np.dtype, *, protocol: str = "dap2") -> DapType:
    """Map a numpy dtype to the appropriate DapType.

    For protocol="dap2", int64/uint64 map to Float64 (lossy).
    For protocol="dap4", int64/uint64 map to Int64/UInt64 (lossless).
    """

def cf_encode_variable(var: xr.Variable) -> xr.Variable:
    """Apply CF encoding (datetime64→numeric, etc.) and normalize byte order."""
```

CF encoding is centralized here. The function calls
`xr.conventions.encode_cf_variable()` and normalizes big-endian dtypes to
little-endian (matching what numpy operates on natively). CF encoding is
applied only for `datetime64` and `timedelta64` types — `scale_factor` /
`add_offset` are intentionally NOT applied because the plugin serves decoded
xarray values.

### 2.2 `dap/constraint.py` — Constraint Expression Parser

A hand-written recursive-descent parser for DAP2 constraint expressions, with a
separate entry point for DAP4 syntax added in Phase 2.

**Parsed representation:**

```python
@dataclass(frozen=True)
class HyperSlab:
    """A single dimension slice: [start:stride:stop] (DAP-style, inclusive)."""
    start: int
    stop: int           # inclusive (DAP convention)
    stride: int = 1

@dataclass(frozen=True)
class ProjectionItem:
    """A single item in a projection clause."""
    name: str                                # variable name (may be dotted: "grid.temp")
    slices: tuple[HyperSlab, ...] | None = None  # None means "no subsetting"

@dataclass(frozen=True)
class SelectionClause:
    """A single selection sub-expression: field op value."""
    field: str
    operator: str       # '<', '<=', '>', '>=', '=', '!=', '=~'
    value: str | float | int

@dataclass
class Constraint:
    """The fully parsed constraint expression."""
    projections: list[ProjectionItem] = field(default_factory=list)   # empty = select all
    selections: list[SelectionClause] = field(default_factory=list)   # empty = no filtering
```

**Parser interface:**

```python
def parse_dap2_constraint(raw: str) -> Constraint:
    """Parse a DAP2 constraint expression string.

    Grammar (simplified):
        CE          = [projection] *("&" selection)
        projection  = proj_item ("," proj_item)*
        proj_item   = name [hyperslab]
        hyperslab   = ("[" start ":" [stride ":"] stop "]")+
        selection   = field relop value
    """

def parse_dap4_constraint(raw: str) -> Constraint:
    """Parse a DAP4 constraint expression string.

    Normalizes DAP4 syntax to a common Constraint representation:
    - Semicolons → commas (variable separators)
    - Leading slashes stripped from variable names
    - Selection/filter clauses raise ConstraintNotSupportedError
    """
```

**Design notes:**

- The parser is ~150-200 lines of hand-written Python. The DAP2 CE grammar is
  small enough that a parser generator (lark, etc.) adds unnecessary
  complexity. A recursive-descent parser is easier to debug and produces
  better error messages.
- Selection clauses are parsed but raise `ConstraintNotSupportedError` until
  Sequence support is implemented (per TODO.txt). This ensures the parser
  handles the full grammar even before selection is functional.
- Function calls in projections/selections are recognized syntactically
  (`id(args)`) but raise `ConstraintNotSupportedError`. This reserves the
  syntax for future server-side function support.

### 2.3 `dap/xarray_adapter.py` — Constraint → xarray Subsetting

Translates a parsed `Constraint` into xarray operations. This is the bridge
between the protocol layer and xarray — it ensures constraints are applied to
the lazy Dataset *before* data loading.

```python
@dataclass
class SubsettingPlan:
    """An executable plan for subsetting an xr.Dataset."""
    variables: list[str] | None     # None = all variables
    isel_args: dict[str, slice]     # dimension → slice (Python-style, exclusive stop)
    estimated_bytes: int            # estimated memory after subsetting

def plan_subsetting(
    ds: xr.Dataset,
    constraint: Constraint,
) -> SubsettingPlan:
    """Build a subsetting plan from a parsed constraint.

    1. Map projection variable names to ds variables + coordinates.
    2. Convert HyperSlab (inclusive DAP stop) → Python slice (exclusive stop):
       slice(start, stop + 1, stride)
    3. For Grids: when a Grid variable is sliced, apply the same slices to its
       coordinate variables (map vectors).
    4. Estimate memory: sum of (subsetted_size * dtype.itemsize) for each variable.
    """

def apply_plan(ds: xr.Dataset, plan: SubsettingPlan) -> xr.Dataset:
    """Apply a SubsettingPlan to a lazy xr.Dataset.

    Returns a new Dataset with only the selected variables and dimensions
    sliced. No data is loaded — this operates on the dask/numpy graph.
    """
```

**Inclusive→exclusive stop conversion:**

DAP uses `[0:10]` to mean elements 0 through 10 inclusive (11 elements). Python
uses `slice(0, 11)`. The adapter converts `HyperSlab(start=0, stop=10,
stride=1)` → `slice(0, 11, 1)`. With stride > 1: `[0:2:10]` → `slice(0, 11,
2)` → elements at indices 0, 2, 4, 6, 8, 10.

**Grid handling:**

When a constraint projects a Grid variable with hyperslabs (e.g.
`temp[0:10][20:30]`), the adapter must also apply the corresponding slices to
the coordinate dimensions used by that Grid. This maintains the Grid invariant
that map vectors have the same size as the corresponding array dimension.

### 2.4 `dap/dap2/dds.py` — DDS Generation

Generates the Dataset Descriptor Structure text from an xr.Dataset. Operates
directly on xarray metadata (no intermediate objects).

```python
def generate_dds(
    ds: xr.Dataset,
    dataset_name: str,
) -> Iterator[str]:
    """Yield DDS text lines for the dataset.

    Structure:
        Dataset {
            <type> <coord>[<coord> = <len>];       # for each coordinate
            Grid {
              Array:
                <type> <var>[<dim1> = <len>]...;
              Maps:
                <type> <dim1>[<dim1> = <len>];
                ...
            } <var>;                                # for each data variable
        } <dataset_name>;
    """
```

**Key behaviors:**
- Constraints are applied *before* calling `generate_dds` — the function
  receives an already-subsetted Dataset. No constraint parameter is needed.
- Coordinates are emitted as top-level Array declarations.
- Data variables with named dimensions are emitted as Grid declarations. A Grid
  includes an "Array:" section (the data) and a "Maps:" section (its coordinate
  variables). This matches what netCDF4-python and pydap expect.
- Scalar variables (0-d) are emitted as bare atomic type declarations.
- Variable and dimension names containing special characters (spaces, dots) are
  escaped per DAP2 rules.

### 2.5 `dap/dap2/das.py` — DAS Generation

Generates the Dataset Attribute Structure text.

```python
def generate_das(ds: xr.Dataset) -> Iterator[str]:
    """Yield DAS text lines.

    Structure:
        Attributes {
            <coord> {
                <type> <attr_name> <value>;
                ...
            }
            <var> {
                <type> <attr_name> <value>;
                ...
            }
            NC_GLOBAL {
                <type> <attr_name> <value>;
                ...
            }
        }
    """
```

**Key behaviors:**
- Constraints are applied *before* calling `generate_das` — the function
  receives an already-subsetted Dataset.
- Each variable (coordinates and data vars) gets its own attribute container.
- Global attributes are emitted in an `NC_GLOBAL` container (Hyrax convention).
- String attribute values are double-quoted. Internal double quotes are
  escaped as `\"`. Backslashes are escaped as `\\`.
- Numeric attributes use the appropriate DAP type name (`Int32`, `Float64`, etc.)
  based on the numpy dtype of the attribute value.

### 2.6 `dap/dap2/dods.py` — DODS/DataDDS Binary Encoding

This is the most critical module. It encodes data as XDR binary and streams it
in bounded chunks.

```python
async def generate_dods(
    ds: xr.Dataset,
    dataset_name: str,
    *,
    dask_num_workers: int = 4,
    slab_threshold_bytes: int = SLAB_THRESHOLD_BYTES,
) -> AsyncIterator[bytes]:
    """Yield the complete DODS response as bytes chunks.

    Structure:
        [DDS text as UTF-8 bytes]
        b'\\nData:\\n'
        [XDR-encoded binary data for each variable]
    """
```

The async generator operates on an already-subsetted Dataset (constraints are
applied by the plugin before calling this function). It iterates per-variable,
loading and encoding each in a thread pool executor via `run_in_executor`.

**XDR encoding internals** (private functions within the module):

```python
def _xdr_encode_array(data: np.ndarray, dap_type: DapType) -> Iterator[bytes]:
    """Encode a numpy array as XDR bytes.

    Rules:
    1. For atomic types: emit length as big-endian uint32 TWICE, then values.
    2. For Byte arrays: values packed contiguously, padded to 4-byte boundary.
    3. For Int16/UInt16: each element widened to 4 bytes on wire.
    4. For String arrays: each string individually length-prefixed + padded.
    """

def _xdr_length_prefix(n: int) -> bytes:
    """Encode an array length as big-endian uint32 (4 bytes)."""
    return struct.pack('>I', n)

def _pad_size(length: int) -> int:
    """Calculate padding needed to reach 4-byte boundary."""
```

**Three encoding paths for data variables:**

1. **Dask-graph fast path** (`_is_dask_graph_eligible`): For dask-backed,
   non-string, non-datetime arrays, fuses dtype conversion into the dask graph:
   `ravel → rechunk(~20MB) → block.astype(wire_dtype).compute().tobytes()`.
   This avoids materializing the full array, keeping peak memory bounded to one
   ~20MB block. ~37% faster than the eager path for large arrays.

2. **Slab streaming** (`get_slab_boundaries`): For dask-backed arrays above
   `slab_threshold_bytes` (default 32MB) with multiple chunks along at least one
   dimension, emits the length prefix from metadata then streams encoded slabs
   along the most-chunked dimension. Each slab is loaded and encoded in the
   thread pool with prefetch=1 for pipelining.

3. **Eager path**: For numpy-backed, string, datetime, or small arrays — loads
   the full variable, applies CF encoding, and encodes via `_xdr_encode_array`.

**Grid encoding:**

A Grid is encoded as a Structure on the wire: the data array first, then each
map vector in order. Coordinate arrays are loaded once and cached — Grid Map
vectors are re-emitted from cache for each data variable that references them.

**DataDDS separator:**

Per the corrected spec (v1.2), the separator between DDS text and binary data
is `\nData:\n` (LF only, not CRLF). The old `opendap-protocol` library uses
`\r\n` which is incorrect and may cause issues with strict clients.

### 2.7 `dap/dap2/responses.py` — Error, Version, Help

```python
def generate_error(code: int, message: str) -> Iterator[str]:
    """Yield DAP2 Error response text.

    Format:
        Error {
            code = <code>;
            message = "<message>";
        };
    """

def generate_version(
    dap_version: str = "2.0",
    server_version: str = "xpublish-opendap/2.0",
) -> Iterator[str]:
    """Yield DAP2 Version response text."""

def generate_help() -> str:
    """Return HTML Help response listing recognized extensions."""
```

### 2.8 `dap/dap2/headers.py` — HTTP Header Constants

```python
DAP2_HEADERS = {
    "XDODS-Server": "xpublish-opendap/2.0",
    "XOPeNDAP-Server": "xpublish-opendap/2.0",
}

CONTENT_DESCRIPTIONS = {
    "dds": "dods-dds",
    "das": "dods-das",
    "dods": "dods-data",
    "error": "dods-error",
}

CONTENT_TYPES = {
    "dds": "text/plain",
    "das": "text/plain",
    "dods": "application/octet-stream",
    "error": "text/plain",
    "version": "text/plain",
    "help": "text/html",
}
```

### 2.9 `errors.py` — Error Types

```python
class DapError(Exception):
    """Base class for DAP protocol errors."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message

class ConstraintSyntaxError(DapError):
    """Raised when a constraint expression is malformed."""
    def __init__(self, message: str):
        super().__init__(code=1000, message=message)

class ConstraintNotSupportedError(DapError):
    """Raised for valid but unsupported constraint features (Sequences, functions)."""
    def __init__(self, message: str):
        super().__init__(code=1001, message=message)

class VariableNotFoundError(DapError):
    """Raised when a constraint references a non-existent variable."""
    def __init__(self, name: str):
        super().__init__(code=1002, message=f"Variable '{name}' not found")

class IndexOutOfRangeError(DapError):
    """Raised when a hyperslab index exceeds array bounds."""
    def __init__(self, message: str):
        super().__init__(code=1003, message=message)

class RequestTooLargeError(DapError):
    """Raised when estimated memory exceeds the configured threshold."""
    def __init__(self, estimated_bytes: int, max_bytes: int):
        super().__init__(
            code=1004,
            message=f"Estimated response size ({estimated_bytes} bytes) exceeds "
                    f"limit ({max_bytes} bytes). Use a constraint to request a subset.",
        )
```

### 2.10 `io.py` — Async Infrastructure

Follows the xpublish-tiles pattern: module-level ThreadPoolExecutor with
per-event-loop semaphores, plus slab streaming and dask-graph encoding support.

```python
EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="xpublish-opendap-pool",
)

# Byte budget per rechunked block for the dask-graph fast path.
# 20 MB matches the old opendap-protocol approach.
RECHUNK_BYTES = 20_000_000

# Skip slab streaming for arrays smaller than this (32 MB).
SLAB_THRESHOLD_BYTES = 32 * 1024 * 1024

_compute_semaphores: dict[int, asyncio.Semaphore] = {}   # id(loop) → semaphore
_data_load_semaphores: dict[int, asyncio.Semaphore] = {}

def get_compute_semaphore(max_workers: int = 8) -> asyncio.Semaphore:
    """Semaphore limiting concurrent blocking compute operations."""

def get_data_load_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    """Semaphore limiting concurrent xarray data loads."""

async def run_in_executor(func: Callable, *args: Any) -> Any:
    """Run a blocking function in the thread pool with semaphore limiting."""

def load_variable(var: xr.DataArray | xr.Variable, dask_num_workers: int = 4) -> None:
    """Load a single variable into memory with parallel dask scheduling.

    Synchronous — intended to be called from a thread pool executor.
    """

def get_slab_boundaries(
    da: xr.DataArray,
    *,
    threshold_bytes: int = SLAB_THRESHOLD_BYTES,
) -> tuple[str, list[tuple[int, int]]] | None:
    """Return best dimension and (start, stop) pairs for slab streaming.

    Returns None if slab streaming is not applicable (numpy-backed,
    too small, string/datetime dtype, single chunk along all dims).
    """

def _is_dask_graph_eligible(da: xr.DataArray) -> bool:
    """Check whether a DataArray can use the dask-graph encoding fast path.

    Eligible when dask-backed, ndim > 0, non-string, non-datetime.
    """

def dask_graph_encode_data_bytes(
    da: xr.DataArray,
    wire_dtype: np.dtype,
    dask_num_workers: int = 4,
    rechunk_bytes: int = RECHUNK_BYTES,
) -> bytes:
    """Encode a dask-backed DataArray by fusing dtype conversion into the graph.

    Flattens, rechunks to ~20MB blocks, computes block-by-block.
    Returns raw data bytes only — no length prefixes or padding.
    """

async def load_dataset_async(
    ds: xr.Dataset,
    *,
    timeout: float = 30.0,
    dask_num_workers: int = 4,
) -> xr.Dataset:
    """Load a lazy Dataset into memory off the event loop.

    Runs ds.load() in a thread pool executor. Used for full-dataset
    loading when per-variable streaming is not needed.
    """
```

### 2.11 `plugin.py` — Plugin Rewrite

```python
class OpenDapPlugin(Plugin):
    """OPeNDAP plugin for xpublish."""

    name: str = "opendap"
    dataset_router_prefix: str = "/opendap"
    dataset_router_tags: list[str] = ["opendap"]

    # Configuration
    max_request_memory_bytes: int = 512 * 1024 * 1024   # 512 MB
    num_concurrent_data_loads: int = 4
    async_load_timeout: float = 30.0
    dask_num_workers: int = 4

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:
        router = APIRouter(...)
        config = self

        # DAP2 endpoints
        @router.get(".dds")    → DDS text response
        @router.get(".das")    → DAS text response
        @router.get(".dods")   → DODS StreamingResponse
        @router.get(".ver")    → Version text response
        @router.get(".help")   → Help HTML response

        # DAP4 endpoints
        @router.get(".dmr")    → DMR XML response
        @router.get(".dap")    → DAP4 Data StreamingResponse
        @router.get(".dsr")    → DSR XML response

        # Bare path → Help
        @router.get("")        → Help HTML response

        return router
```

**Request pipeline** (common to DDS, DAS, DODS):

```
1. Extract constraint string from request.url query component
2. parse_dap2_constraint(raw_constraint) → Constraint
   - On ConstraintSyntaxError → return DAP Error response
3. plan_subsetting(ds, constraint) → SubsettingPlan
   - On VariableNotFoundError → return DAP Error response
   - On IndexOutOfRangeError → return DAP Error response
4. [DODS only] Check plan.estimated_bytes vs config.max_request_memory_bytes
   - On RequestTooLargeError → return DAP Error response
5. apply_plan(ds, plan) → subsetted lazy Dataset
6. Generate response (DDS text / DAS text / DODS binary stream)
7. Attach DAP2 headers (Content-Description, XDODS-Server, XOPeNDAP-Server)
```

DAP4 endpoints follow the same pipeline but use `parse_dap4_constraint` (from
the `dap4.ce` query parameter) and attach DAP4 headers (`XDAP: 4.0`,
`XOPeNDAP-Server`).

For DDS/DAS/DMR, no data loading occurs — only metadata is needed.

For DODS/DAP4 Data, the response is a `StreamingResponse` wrapping an async
generator (`generate_dods` or `generate_dap4_data`). The async generator loads
and encodes each variable in a thread pool executor, yielding encoded bytes
per-variable (or per-slab for large arrays).

**Error responses:**

All `DapError` exceptions are caught in a try/except wrapper and converted to
proper DAP Error responses with `Content-Description: dods-error` and
`Content-Type: text/plain`. The HTTP status code is 400 for constraint errors,
413 for request-too-large, and 500 for unexpected errors.

**Header injection:**

Headers (`XDODS-Server`, `XOPeNDAP-Server`, `Content-Description`) are set
manually per-response via helper functions. `Last-Modified` is not yet
implemented (tracked in TODO.txt — requires dataset timestamp metadata).

---

## 3. Phase 2 Additions (DAP4)

### 3.1 `dap/dap4/dmr.py` — DMR XML Generation

```python
def generate_dmr(ds: xr.Dataset, dataset_name: str) -> str:
    """Generate a DAP4 DMR XML document.

    Uses xml.etree.ElementTree for XML generation.

    Structure:
        <Dataset name="..." xmlns="http://xml.opendap.org/ns/DAP/4.0#"
                 dapVersion="4.0" dmrVersion="1.0">
          <Attribute name="_DAP4_Little_Endian" type="UInt8"><Value>1</Value></Attribute>
          <Dimension name="x" size="100"/>
          <Float64 name="temperature">
            <Dim name="/x"/>
            <Map name="/lon"/>
            <Map name="/lat"/>
            <Attribute name="units" type="String">
              <Value>kelvin</Value>
            </Attribute>
          </Float64>
          <Attribute name="NC_GLOBAL" type="Container">
            ...
          </Attribute>
        </Dataset>
    """
```

Constraints are applied *before* calling `generate_dmr` — the function receives
an already-subsetted Dataset. The DMR includes a `_DAP4_Little_Endian`
attribute indicating the byte order of the binary data payload.

**Coverage/Map handling:**

Coordinate variables referenced by data variables are emitted as `<Map>`
elements inside the variable declaration. This replaces DAP2's Grid type.

**DataTree/Group support:** Not yet implemented. The function accepts
`xr.Dataset` only. Group support would require `xr.DataTree` input.

### 3.2 `dap/dap4/data.py` — DAP4 Chunked Binary Encoding

```python
async def generate_dap4_data(
    ds: xr.Dataset,
    dataset_name: str,
    *,
    dask_num_workers: int = 4,
    slab_threshold_bytes: int = SLAB_THRESHOLD_BYTES,
) -> AsyncIterator[bytes]:
    """Yield a complete DAP4 data response.

    Structure:
        [DMR XML as UTF-8]
        [CRLF separator]
        [chunk header: uint32 BE with type+endian+size]
        [variable 1 serialized data]
        [CRC32 of variable 1 data: uint32 LE]
        [chunk header]
        [variable 2 serialized data]
        [CRC32 of variable 2 data: uint32 LE]
        ...
        [end chunk: CHUNK_END = 0x01000000]
    """
```

The chunk header is a single big-endian uint32 encoding type flags, endianness
flag, and payload size (24-bit) in one word:

```python
CHUNK_DATA          = 0x00000000
CHUNK_END           = 0x01000000
CHUNK_ERR           = 0x02000000
CHUNK_LITTLE_ENDIAN = 0x04000000
CHUNK_SIZE_MASK     = 0x00FFFFFF  # max ~16 MB per chunk
```

The async generator supports the same three encoding paths as DAP2 (dask-graph
fast path, slab streaming, and eager), with native byte order and CRC32 per
variable. For slab streaming, the CRC32 is accumulated incrementally across
slabs using `zlib.crc32`'s running checksum parameter.

**DAP4 serialization rules (contrast with DAP2/XDR):**

| Aspect | DAP2 (XDR) | DAP4 |
|--------|-----------|------|
| Byte order | Big-endian (network) | Server native (little-endian on x86) |
| Array length prefix | 4-byte uint32, sent twice for atomics | 8-byte uint64, sent once |
| Int16/UInt16 | Padded to 32-bit on wire | Natural 2-byte size |
| Byte arrays | Padded to 4-byte boundary | No padding |
| Strings | ASCII, length-prefixed, padded to 4 bytes | UTF-8, 8-byte uint64 length prefix, no padding |
| Structures | Fields in order, no framing | Fields in order, no framing |
| Checksums | None | CRC32 (LE) per top-level variable |
| Error mid-stream | Not possible | CHUNK_ERR type flag (not implemented) |

**Byte-order indicator:** The DMR includes `_DAP4_Little_Endian` attribute
set to `1` (since all our deployment targets are x86/ARM little-endian).

### 3.3 Plugin DAP4 Routes

Added to `plugin.py` alongside DAP2 routes:

```python
@router.get(".dmr")
async def dmr_response(...) -> Response:
    """DAP4 Dataset Metadata Response."""
    # Content-Type: application/vnd.opendap.dap4.dataset-metadata+xml

@router.get(".dap")
async def dap4_data_response(...) -> StreamingResponse | Response:
    """DAP4 Data Response."""
    # Content-Type: application/vnd.opendap.dap4.data

@router.get(".dsr")
async def dsr_response(...) -> Response:
    """DAP4 Dataset Services Response (capabilities)."""
    # Content-Type: application/vnd.opendap.dap4.dataset-services+xml
```

DAP4 headers are defined in `dap4/headers.py`:

```python
DAP4_HEADERS = {
    "XDAP": "4.0",
    "XOPeNDAP-Server": "xpublish-opendap/2.0",
}
```

DAP4 constraint expressions are extracted from the `dap4.ce` query parameter
(e.g. `?dap4.ce=/temp[0:10]`) and parsed via `parse_dap4_constraint()`. DAP4
error responses use XML format with `ErrorCode` and `Message` elements.

---

## 4. Data Flow Diagrams

### 4.1 DODS Request

```
Client: GET /datasets/air/opendap.dods?air[0:10][0:5][0:5]

  ┌─────────┐
  │ FastAPI  │ extracts query string
  └────┬─────┘
       │ "air[0:10][0:5][0:5]"
       ▼
  ┌─────────────┐
  │  constraint  │ parse_dap2_constraint()
  │   parser     │ → Constraint(projections=[ProjectionItem("air", slices=...)])
  └────┬─────────┘
       ▼
  ┌─────────────┐
  │   xarray     │ plan_subsetting(ds, constraint)
  │  adapter     │ → SubsettingPlan(variables=["air"], isel_args={time: 0:11, lat: 0:6, lon: 0:6})
  └────┬─────────┘
       │
       ├─ estimated_bytes = 11 * 6 * 6 * 8 = 3168  (well under 512MB)
       ▼
  ┌─────────────┐
  │ apply_plan   │ ds[["air"]].isel(time=slice(0,11), lat=slice(0,6), lon=slice(0,6))
  │              │ → lazy subsetted Dataset (no compute yet)
  └────┬─────────┘
       ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │ StreamingResponse(generate_dods(subsetted_ds, "air"))           │
  │                                                                  │
  │ Async generator yields per-variable:                             │
  │                                                                  │
  │   yield DDS text (from metadata, no I/O)                        │
  │   yield b'\nData:\n'                                            │
  │                                                                  │
  │   for each coord:                                                │
  │     await run_in_executor(_load_and_encode_xdr, coord, workers) │
  │     yield XDR-encoded coord bytes                                │
  │     (cache for Grid Map re-emission)                             │
  │                                                                  │
  │   for each data_var:                                             │
  │     if dask-eligible + large → slab streaming:                   │
  │       yield length_prefix × 2 (from metadata)                   │
  │       for each slab (with prefetch=1):                           │
  │         await run_in_executor(_load_and_encode_slab_xdr, ...)   │
  │         yield slab bytes                                         │
  │     elif dask-eligible → dask-graph fast path:                   │
  │       await run_in_executor(_load_and_encode_xdr, ...)          │
  │       yield encoded bytes (ravel→rechunk→compute block-by-block)│
  │     else → eager load:                                           │
  │       await run_in_executor(_load_and_encode_xdr, ...)          │
  │       yield full variable XDR bytes                              │
  │     yield cached Grid Map coord bytes                            │
  └──────────────────────────────────────────────────────────────────┘
       │
       ▼
  Client receives streamed bytes
```

### 4.2 DAP4 Data Request

```
Client: GET /datasets/air/opendap.dap?dap4.ce=/air[0:10][0:5][0:5]

  ┌─────────┐
  │ FastAPI  │ extracts dap4.ce query param
  └────┬─────┘
       ▼
  ┌─────────────┐
  │  constraint  │ parse_dap4_constraint()
  │   parser     │ (normalizes DAP4 syntax → common Constraint)
  └────┬─────────┘
       ▼
  [same subsetting pipeline as DAP2: plan_subsetting → apply_plan]
       ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ StreamingResponse(generate_dap4_data(subsetted_ds, "air"))  │
  │                                                              │
  │   yield DMR XML (subsetted metadata)                        │
  │   yield CRLF separator                                      │
  │                                                              │
  │   for each variable (coords then data_vars):                 │
  │     if slab streaming:                                       │
  │       yield chunk header + length prefix + slab bytes ...    │
  │       (CRC32 accumulated incrementally)                      │
  │       yield CRC32                                            │
  │     else:                                                    │
  │       yield chunk headers + data + CRC32                     │
  │                                                              │
  │   yield end chunk (CHUNK_END = 0x01000000)                  │
  └──────────────────────────────────────────────────────────────┘
       ▼
  Client receives streamed bytes
```

---

## 5. Memory Management Design

### 5.1 Pre-load Estimation

Before any data is loaded, `plan_subsetting()` computes:

```python
estimated_bytes = sum(
    math.prod(subsetted_shape) * dtype.itemsize
    for var_name in selected_variables
    for subsetted_shape, dtype in [(compute_shape(ds[var_name], isel_args), ds[var_name].dtype)]
)
```

If `estimated_bytes > config.max_request_memory_bytes`, the request is rejected
immediately with a `RequestTooLargeError` → DAP Error response. No data is
loaded.

### 5.2 Slab Streaming

For dask-backed arrays above `SLAB_THRESHOLD_BYTES` (32 MB) with multiple
chunks along at least one dimension, the encoder never materializes the full
variable. Instead, it streams "slabs" along the most-chunked dimension:

1. The length prefix is emitted from metadata (no I/O needed).
2. Each slab (one chunk-group along the chosen dimension) is loaded and encoded
   in the thread pool via `run_in_executor`.
3. Slabs are pipelined with prefetch=1: while the current slab is being sent,
   the next slab is loading.
4. Each slab's bytes are yielded to the `StreamingResponse`.

Slab streaming is disabled for string and datetime/timedelta arrays (which
require CF encoding that produces data-dependent reference times per slab).

### 5.3 Dask-Graph Fast Path

For dask-backed numeric arrays, both slab and full-variable paths use the
dask-graph encoding approach: `ravel → rechunk(~20MB blocks) →
block.astype(wire_dtype).compute().tobytes()`. This fuses dtype conversion
into the dask graph and computes block-by-block, keeping peak memory to one
~20MB block regardless of total variable size.

### 5.4 Peak Memory Model

```
Peak memory per request (slab streaming + dask-graph):
  = (prefetch + 1) × slab_encoding_buffer + coord_cache_size
  ≈ 2 × 20 MB + small
  ≈ ~40 MB per concurrent request

Peak memory per request (full-variable, dask-graph):
  = ~20 MB block buffer + encoded bytes accumulator
  ≈ ~40 MB per concurrent request
```

### 5.5 Concurrent Request Limiting

The `compute_semaphore` (default: 8 slots) limits concurrent blocking
operations in the thread pool. The `data_load_semaphore` (default: 4 slots)
limits concurrent `load_dataset_async` calls. Together these prevent memory
explosion from many concurrent requests.

---

## 6. Testing Strategy

The test suite contains ~440 tests across 27 files, organized as:

### 6.1 Unit Tests (flat layout under `tests/`)

- **`test_types.py`** — Type mapping: every dtype to DapType, CF encoding
- **`test_constraint.py`** — Parser: valid expressions, edge cases, malformed input
- **`test_xarray_adapter.py`** — Constraint→isel: slicing, Grid paths, bounds
- **`test_dap2_responses.py`** — DDS/DAS/DODS encoding, XDR gold-file bytes
- **`test_dap4_dmr.py`** — DMR XML structure validation
- **`test_dap4_data.py`** — DAP4 binary encoding, CRC32 checksums
- **`test_dap4_headers.py`** — DAP4 header constants
- **`test_dap4_routes.py`** — DAP4 route handler behavior
- **`test_plugin_routes.py`** — Plugin route registration, headers, errors
- **`test_io.py`** — Async infrastructure: executor, semaphores, load helpers
- **`test_async_streaming.py`** — Streaming response correctness
- **`test_slab_streaming.py`** — Slab boundaries, dask-graph fast path,
  byte-identity tests (dask vs numpy produce identical output), datetime safety
- **`test_server.py`** — Server lifecycle, dataset serving

### 6.2 Integration Tests (`tests/integration/`)

Integration tests launch a real xpublish server (using `pytest-xprocess` for
TCP servers or httpx `ASGITransport` for in-process testing) and verify
round-trip correctness with real DAP clients:

- **`test_netcdf4_client.py`** — netCDF4-python DAP2 round-trip
- **`test_pydap_client.py`** — pydap client round-trip
- **`test_xarray_engines.py`** — xarray with netcdf4 and pydap engines
- **`test_ncdump.py`** — ncdump CDL output validation
- **`test_ncks.py`** — NCO ncks extraction
- **`test_julia_client.py`** — Julia NCDatasets.jl round-trip
- **`test_r_client.py`** — R ncdf4 package round-trip
- **`test_httpx_raw.py`** — Raw HTTP byte-level validation

### 6.3 Test Fixtures (`conftest.py`)

Fixtures provide test datasets covering the type space: float64/float32 data,
all integer types, string arrays, datetime64/timedelta64, boolean, scalar
variables, special characters in names, dask-backed arrays, and multi-variable
datasets.

---

## 7. Migration Path

### 7.1 Dependency Changes

**Remove:**
- `opendap-protocol` from `requirements.txt`

**Add:**
- No new runtime dependencies. The protocol layer is pure Python using only
  `struct`, `xml.etree.ElementTree`, `zlib` (for CRC32 in Phase 2), and
  `numpy` (already a transitive dependency via xarray).

**Update `pyproject.toml`:**
```toml
[tool.setuptools]
packages = ["xpublish_opendap", "xpublish_opendap.dap",
            "xpublish_opendap.dap.dap2", "xpublish_opendap.dap.dap4"]
```

### 7.2 Entry Point

The entry point remains unchanged:
```toml
[project.entry-points."xpublish.plugin"]
opendap = "xpublish_opendap.plugin:OpenDapPlugin"
```

### 7.3 Breaking Changes

- The `get_dap_dataset` dependency (which returns an `opendap_protocol.Dataset`)
  is removed. There is no public API for accessing the intermediate DAP object
  model because there is no intermediate object model.
- The cache key `opendap_dataset_{dataset_id}` is no longer written.
- The `dap_xarray` module is removed.

---

## 8. Implementation Order

Implementation proceeded bottom-up through the module dependency graph:

**Phase 1 (DAP2 + async + memory management):**
1. `dap/types.py` → 2. `dap/constraint.py` → 3. `dap/xarray_adapter.py` →
4. `dap/dap2/dds.py` → 5. `dap/dap2/das.py` → 6. `dap/dap2/dods.py` →
7. `dap/dap2/responses.py` + `errors.py` → 8. `io.py` → 9. `plugin.py` →
10. Integration tests

**Phase 2 (DAP4):**
11. `dap/dap4/dmr.py` → 12. `dap/dap4/data.py` → 13. `dap/dap4/responses.py`
+ `dap/dap4/headers.py` → 14. DAP4 routes in `plugin.py` → 15. DAP4 tests

**Performance optimizations (post Phase 2):**
16. Slab streaming in `io.py` + `dods.py` + `data.py` →
17. Dask-graph encoding fast path → 18. Benchmark suite

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| XDR encoding bugs break client compatibility | High | Gold-file tests against reference byte sequences; integration tests with 8 different clients (netCDF4, pydap, xarray, NCO, ncdump, Julia, R, httpx) |
| DAP4 chunked format not correctly parsed by clients | High | Tested against real clients; CRC32 checksums detect corruption; Hyrax responses as reference |
| Memory estimation inaccurate for complex dask graphs | Medium | Estimation is conservative (dtype.itemsize × product of shape); slab streaming + dask-graph fast path bound actual peak memory independently of estimation |
| Constraint parser edge cases | Medium | Extensive parser test suite; `ConstraintNotSupportedError` for unrecognized features |
| CF encoding per-slab inconsistency | Medium | Datetime/timedelta arrays excluded from slab streaming (CF encoding picks data-dependent reference times) |
| StreamingResponse backpressure | Low | FastAPI/Starlette handles TCP backpressure; async generator naturally pauses when consumer is slow |
