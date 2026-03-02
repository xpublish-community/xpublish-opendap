# xpublish-opendap v2: Technical Design

This document details the implementation plan for Phase 1 (DAP2 conformance,
async, memory management) and Phase 2 (DAP4) of the xpublish-opendap rewrite
described in PRD.md.

---

## 1. Target Directory Layout

```
xpublish_opendap/
├── __init__.py                      # Public API: OpenDapPlugin
├── plugin.py                        # Plugin class, async route handlers
├── errors.py                        # DAP error types, error response helpers
├── io.py                            # Async infrastructure (executor, semaphores)
├── dap/                             # Protocol layer (clean-room, no opendap-protocol)
│   ├── __init__.py
│   ├── types.py                     # numpy↔DAP type mapping, CF encoding
│   ├── constraint.py                # Constraint expression parser (DAP2 + DAP4)
│   ├── xarray_adapter.py            # Constraint → xarray subsetting plan
│   ├── dap2/
│   │   ├── __init__.py
│   │   ├── dds.py                   # DDS text generation
│   │   ├── das.py                   # DAS text generation
│   │   ├── dods.py                  # DODS binary (XDR) encoding
│   │   ├── responses.py             # Error, Version, Help response generators
│   │   └── headers.py               # DAP2 HTTP header constants
│   └── dap4/                        # Phase 2
│       ├── __init__.py
│       ├── dmr.py                   # DMR XML generation
│       ├── data.py                  # DAP4 chunked binary encoding
│       ├── responses.py             # DSR, DAP4 error responses
│       └── headers.py               # DAP4 HTTP header / content-type constants
tests/
├── conftest.py                      # Shared fixtures
├── dap/
│   ├── test_types.py                # Type mapping unit tests
│   ├── test_constraint.py           # Constraint parser unit tests
│   ├── test_xarray_adapter.py       # Constraint → isel tests
│   ├── dap2/
│   │   ├── test_dds.py              # DDS output tests
│   │   ├── test_das.py              # DAS output tests
│   │   ├── test_dods.py             # XDR encoding tests
│   │   └── test_responses.py        # Error/Version/Help tests
│   └── dap4/
│       ├── test_dmr.py              # DMR XML tests
│       └── test_data.py             # DAP4 binary encoding tests
├── test_plugin.py                   # Plugin route tests (headers, async, errors)
├── test_memory.py                   # Memory threshold tests
└── integration/
    ├── conftest.py                  # Live server fixture
    ├── test_netcdf4.py              # netCDF4-python round-trip
    ├── test_pydap.py                # pydap client round-trip
    ├── test_xarray.py               # xarray engine round-trip
    └── test_nco.py                  # ncdump/ncks compatibility
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
    is_numeric: bool            # for constraint relational operators
    needs_xdr_length_doubled: bool  # True for atomic arrays in DAP2
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
def resolve_dap_type(dtype: np.dtype, *, protocol: Literal['dap2', 'dap4'] = 'dap2') -> DapType:
    """Map a numpy dtype to the appropriate DapType for the given protocol."""

def cf_encode_variable(var: xr.Variable) -> xr.Variable:
    """Apply CF encoding (datetime64→numeric, etc.) and normalize byte order."""
```

CF encoding is centralized here. The function calls
`xr.conventions.encode_cf_variable()` and normalizes big-endian dtypes to
little-endian (matching what numpy operates on natively). This is called once
per variable before any DDS/DAS/DODS generation.

### 2.2 `dap/constraint.py` — Constraint Expression Parser

A hand-written recursive-descent parser for DAP2 constraint expressions, with a
separate entry point for DAP4 syntax added in Phase 2.

**Parsed representation:**

```python
@dataclass
class HyperSlab:
    """A single dimension slice: [start:stride:stop] (DAP-style, inclusive)."""
    start: int
    stop: int           # inclusive (DAP convention)
    stride: int = 1

@dataclass
class ProjectionItem:
    """A single item in a projection clause."""
    name: str                                # variable name (may be dotted: "grid.temp")
    slices: tuple[HyperSlab, ...] | None     # None means "no subsetting"

@dataclass
class SelectionClause:
    """A single selection sub-expression: field op value."""
    field: str
    operator: str       # '<', '<=', '>', '>=', '=', '!=', '=~'
    value: str | float | int | list  # list for {val1, val2, ...} multi-value

@dataclass
class Constraint:
    """The fully parsed constraint expression."""
    projections: list[ProjectionItem]   # empty = select all
    selections: list[SelectionClause]   # empty = no filtering (only valid on Sequences)
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

# Phase 2:
def parse_dap4_constraint(raw: str) -> Constraint:
    """Parse a DAP4 constraint expression string.

    Key differences from DAP2:
    - Field subsetting uses braces: {var1;var2}
    - Filters use pipe: sequence|field>value
    - Group-qualified names: /GroupA/variable
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
    constraint: Constraint | None = None,
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
- Coordinates are emitted as top-level Array declarations.
- Data variables with named dimensions are emitted as Grid declarations. A Grid
  includes an "Array:" section (the data) and a "Maps:" section (its coordinate
  variables). This matches what netCDF4-python and pydap expect.
- Scalar variables (0-d) are emitted as bare atomic type declarations.
- If a constraint is provided, only projected variables appear, and dimension
  sizes reflect the sliced shape.
- Variable and dimension names containing special characters (spaces, dots) are
  escaped per DAP2 rules.

### 2.5 `dap/dap2/das.py` — DAS Generation

Generates the Dataset Attribute Structure text.

```python
def generate_das(
    ds: xr.Dataset,
    constraint: Constraint | None = None,
) -> Iterator[str]:
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
            <global attrs as bare attributes at top level>
        }
    """
```

**Key behaviors:**
- Each variable (coordinates and data vars) gets its own attribute container.
- Global attributes are emitted in a top-level container named after the
  dataset, or as bare `Attributes {}` depending on what clients expect (we
  follow the Hyrax convention: global attrs in a top-level `NC_GLOBAL` or
  bare `Attributes` block).
- String attribute values are double-quoted. Internal double quotes are
  escaped as `\"`. Backslashes are escaped as `\\`.
- Numeric attributes use the appropriate DAP type name (`Int32`, `Float64`, etc.)
  based on the numpy dtype of the attribute value.
- Constraints filter which variable attribute containers appear.

### 2.6 `dap/dap2/dods.py` — DODS/DataDDS Binary Encoding

This is the most critical module. It encodes data as XDR binary and streams it
in bounded chunks.

```python
async def generate_dods(
    ds: xr.Dataset,
    dataset_name: str,
    constraint: Constraint | None,
    *,
    chunk_size: int = 2 * 1024 * 1024,  # 2 MB streaming chunks
) -> AsyncIterator[bytes]:
    """Yield the complete DODS response as bytes chunks.

    Structure:
        [DDS text as UTF-8 bytes]
        b'\\nData:\\n'
        [XDR-encoded binary data for each variable]
    """
```

**XDR encoding internals** (private functions within the module):

```python
def _xdr_encode_array(
    data: np.ndarray,
    dap_type: DapType,
) -> Iterator[bytes]:
    """Encode a numpy array as XDR bytes, yielding bounded chunks.

    Rules:
    1. For atomic types: emit length as big-endian int32 TWICE, then values.
    2. For Byte arrays: values are packed contiguously, then padded to 4-byte
       boundary with zero bytes.
    3. For Int16/UInt16 arrays: each element occupies 4 bytes on the wire
       (XDR encodes as int32/uint32).
    4. For String arrays: each string is individually length-prefixed
       (4-byte big-endian length + UTF-8 bytes + padding to 4-byte boundary).
    """

def _xdr_length_prefix(n: int) -> bytes:
    """Encode an array length as big-endian int32 (4 bytes)."""
    return struct.pack('>i', n)

def _pad_to_4(data: bytes) -> bytes:
    """Pad bytes to 4-byte boundary with zero bytes."""
    remainder = len(data) % 4
    if remainder:
        return data + b'\\x00' * (4 - remainder)
    return data
```

**Streaming strategy for large arrays:**

For arrays backed by dask, the encoder iterates over dask chunks (or rechunks
to `chunk_size // dtype.itemsize` elements per chunk). For each chunk:
1. Compute the chunk (blocking, offloaded via `run_in_executor`).
2. XDR-encode the chunk.
3. Yield the encoded bytes.

The first chunk also emits the length prefix (total array length, sent twice).
Subsequent chunks emit only data bytes.

```
[length, length]  # emitted with first chunk
[chunk 0 data]    # yielded
[chunk 1 data]    # yielded
...
[chunk N data]    # yielded
```

This means the full array is never materialized in memory — only one chunk at a
time.

**Grid encoding:**

A Grid is encoded as a Structure on the wire: the data array first, then each
map vector in order. The encoder calls `_xdr_encode_array` for the main array,
then for each coordinate.

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

def generate_version(dap_version: str = "2.0", server_version: str = ...) -> Iterator[str]:
    """Yield DAP2 Version response text."""

def generate_help(extensions: list[str]) -> str:
    """Return HTML Help response listing recognized extensions."""
```

### 2.8 `dap/dap2/headers.py` — HTTP Header Constants

```python
DAP2_HEADERS = {
    'XDODS-Server': 'xpublish-opendap/2.0',
}

CONTENT_DESCRIPTIONS = {
    'dds': 'dods-dds',
    'das': 'dods-das',
    'dods': 'dods-data',
    'error': 'dods-error',
}

CONTENT_TYPES = {
    'dds': 'text/plain',
    'das': 'text/plain',
    'dods': 'application/octet-stream',
    'error': 'text/plain',
    'version': 'text/plain',
    'help': 'text/html',
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
per-event-loop semaphores.

```python
EXECUTOR = ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix='xpublish-opendap-pool',
)

_compute_semaphores: dict[int, asyncio.Semaphore] = {}   # id(loop) → semaphore
_data_load_semaphores: dict[int, asyncio.Semaphore] = {}

def get_compute_semaphore(max_workers: int = 8) -> asyncio.Semaphore:
    """Semaphore limiting concurrent blocking compute operations."""

def get_data_load_semaphore(max_concurrent: int = 4) -> asyncio.Semaphore:
    """Semaphore limiting concurrent xarray data loads."""

async def run_in_executor(func: Callable, *args: Any) -> Any:
    """Run a blocking function in the thread pool with semaphore limiting."""

async def load_variables_async(
    ds: xr.Dataset,
    variables: list[str],
    *,
    timeout: float = 30.0,
    max_concurrent: int = 4,
) -> xr.Dataset:
    """Load specific variables from a lazy Dataset using structured concurrency.

    Uses asyncio.TaskGroup + asyncio.timeout + data load semaphore.
    Each variable is loaded via DataArray.load_async() in a separate task.
    """
```

### 2.11 `plugin.py` — Plugin Rewrite

```python
class OpenDapPlugin(Plugin):
    """OPeNDAP plugin for xpublish."""

    name: str = 'opendap'
    dataset_router_prefix: str = '/opendap'
    dataset_router_tags: list[str] = ['opendap']

    # Configuration
    max_request_memory_bytes: int = 512 * 1024 * 1024   # 512 MB
    num_concurrent_data_loads: int = 4
    encoding_chunk_size: int = 2 * 1024 * 1024          # 2 MB
    async_load_timeout: float = 30.0

    @hookimpl
    def dataset_router(self, deps: Dependencies) -> APIRouter:
        router = APIRouter(
            prefix=self.dataset_router_prefix,
            tags=self.dataset_router_tags,
        )
        # Capture plugin config in closure
        config = self

        @router.get('.dds')
        async def dds_response(
            request: Request,
            ds: xr.Dataset = Depends(deps.dataset),
        ) -> Response:
            ...

        @router.get('.das')
        async def das_response(...) -> Response: ...

        @router.get('.dods')
        async def dods_response(...) -> StreamingResponse: ...

        @router.get('.ver')
        async def version_response(...) -> Response: ...

        # Bare path (no extension) → Help
        @router.get('')
        async def help_response(...) -> HTMLResponse: ...

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
6. [DODS only] load_variables_async(subsetted, ...) → loaded Dataset
7. Generate response (DDS text / DAS text / DODS binary stream)
8. Attach DAP2 headers (Content-Description, XDODS-Server, Last-Modified)
```

For DDS/DAS, data loading (step 6) is skipped — only metadata is needed.

For DODS, the response is a `StreamingResponse` wrapping the async generator
from `generate_dods()`. The async generator yields encoded chunks, each
produced by computing one dask chunk at a time via `run_in_executor`.

**Error responses:**

All `DapError` exceptions are caught in a try/except wrapper and converted to
proper DAP Error responses with `Content-Description: dods-error` and
`Content-Type: text/plain`. The HTTP status code is 400 for constraint errors,
413 for request-too-large, and 500 for unexpected errors.

**Header injection:**

A response middleware or manual header setting adds `XDODS-Server`,
`Content-Description`, and `Last-Modified` to every response. `Last-Modified`
uses `datetime.now(UTC)` if the dataset doesn't carry a timestamp (matching
the spec requirement: "use current date/time if true value unknown").

---

## 3. Phase 2 Additions (DAP4)

### 3.1 `dap/dap4/dmr.py` — DMR XML Generation

```python
def generate_dmr(
    ds: xr.Dataset | xr.DataTree,
    dataset_name: str,
    constraint: Constraint | None = None,
) -> str:
    """Generate a DAP4 DMR XML document.

    Uses xml.etree.ElementTree for XML generation.

    Structure:
        <Dataset name="..." xmlns="http://xml.opendap.org/ns/DAP/4.0#"
                 dapVersion="4.0">
          <Dimension name="x" size="100"/>
          <Float64 name="temperature">
            <Dim name="/x"/>
            <Map name="/lon"/>
            <Map name="/lat"/>
            <Attribute name="units" type="String">
              <Value>kelvin</Value>
            </Attribute>
          </Float64>
          <Group name="subgroup">
            ...
          </Group>
        </Dataset>
    """
```

**DataTree handling:**

When the input is an `xr.DataTree`, each non-root node becomes a `<Group>`.
Variables, dimensions, and attributes within each group are emitted nested
inside their `<Group>` element. Shared dimensions at the root level are
available to all groups.

**Coverage/Map handling:**

Coordinate variables referenced by data variables are emitted as `<Map>`
elements inside the variable declaration. This replaces DAP2's Grid type.

### 3.2 `dap/dap4/data.py` — DAP4 Chunked Binary Encoding

```python
async def generate_dap4_data(
    ds: xr.Dataset | xr.DataTree,
    dataset_name: str,
    constraint: Constraint | None,
    *,
    chunk_size: int = 2 * 1024 * 1024,
) -> AsyncIterator[bytes]:
    """Yield a complete DAP4 data response.

    Structure:
        [DMR XML as UTF-8]
        [CRLF separator]
        [chunk header: 4-byte size (big-endian) + 1-byte type flag 0x00]
        [variable 1 serialized data]
        [4-byte CRC32 of variable 1 data]
        [chunk header]
        [variable 2 serialized data]
        [4-byte CRC32 of variable 2 data]
        ...
        [end chunk: 4-byte size=0 + 1-byte type=0xFF]
    """
```

**DAP4 serialization rules (contrast with DAP2/XDR):**

| Aspect | DAP2 (XDR) | DAP4 |
|--------|-----------|------|
| Byte order | Big-endian (network) | Server native (little-endian on x86) |
| Array length prefix | 4 bytes, sent twice for atomics | 8 bytes (uint64), sent once |
| Int16/UInt16 | Padded to 32-bit on wire | Natural 2-byte size |
| Byte arrays | Padded to 4-byte boundary | No padding |
| Strings | ASCII, length-prefixed, padded to 4 bytes | UTF-8, 4-byte length prefix, no padding |
| Structures | Fields in order, no framing | Fields in order, no framing |
| Checksums | None | CRC32 per top-level variable |
| Error mid-stream | Not possible | Escape chunk (type flag) |

**Byte-order indicator:** The DMR includes `_DAP4_Little_Endian` attribute
set to `1` (since all our deployment targets are x86/ARM little-endian).

### 3.3 Plugin DAP4 Routes

Added to `plugin.py` alongside DAP2 routes:

```python
@router.get('.dmr')
async def dmr_response(
    request: Request,
    ds: xr.Dataset = Depends(deps.dataset),
) -> Response:
    """DAP4 Dataset Metadata Response."""
    # Content-Type: application/vnd.opendap.dap4.dataset-metadata+xml

@router.get('.dap')
async def dap4_data_response(
    request: Request,
    ds: xr.Dataset = Depends(deps.dataset),
) -> StreamingResponse:
    """DAP4 Data Response."""
    # Content-Type: application/vnd.opendap.dap4.data

@router.get('.dsr')
async def dsr_response(
    request: Request,
    ds: xr.Dataset = Depends(deps.dataset),
) -> Response:
    """DAP4 Dataset Services Response (capabilities)."""
    # Content-Type: application/vnd.opendap.dap4.dataset-services+xml
```

The constraint parser detects protocol version from the request context:
- `.dds`, `.das`, `.dods` → DAP2 constraint syntax
- `.dmr`, `.dap` → DAP4 constraint syntax

---

## 4. Data Flow Diagrams

### 4.1 DODS Request (Phase 1)

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
  ┌─────────────┐
  │ load_async   │ asyncio.TaskGroup: ds["air"].load_async()
  │              │ → loaded Dataset with numpy arrays
  └────┬─────────┘
       ▼
  ┌─────────────┐
  │ generate_    │ yield DDS text (for subsetted shape)
  │   dods()     │ yield b'\nData:\n'
  │              │ yield XDR-encoded air array (length prefix × 2, then data)
  │              │ yield XDR-encoded time coord
  │              │ yield XDR-encoded lat coord
  │              │ yield XDR-encoded lon coord
  └────┬─────────┘
       │ StreamingResponse
       ▼
  Client receives streamed bytes
```

### 4.2 DAP4 Data Request (Phase 2)

```
Client: GET /datasets/air/opendap.dap?dap4.ce={air[0:10][0:5][0:5]}

  ┌─────────┐
  │ FastAPI  │
  └────┬─────┘
       ▼
  ┌─────────────┐
  │  constraint  │ parse_dap4_constraint()
  │   parser     │
  └────┬─────────┘
       ▼
  [same subsetting + loading pipeline as DAP2]
       ▼
  ┌─────────────┐
  │ generate_    │ yield DMR XML (subsetted metadata)
  │ dap4_data()  │ yield separator
  │              │ yield chunk header (size + type=0x00)
  │              │ yield air data (native byte order, no XDR padding)
  │              │ yield CRC32 of air data
  │              │ yield chunk header for time coord...
  │              │ ...
  │              │ yield end chunk (size=0, type=0xFF)
  └────┬─────────┘
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

### 5.2 Chunk-at-a-Time Streaming

For arrays that are within the memory limit but still large, the DODS encoder
never materializes the full encoded byte array. Instead:

1. The dask array is rechunked to `encoding_chunk_size // itemsize` elements
   per chunk (if needed).
2. Each chunk is computed in a thread pool (`run_in_executor`).
3. The chunk is XDR-encoded into bytes.
4. The bytes are yielded to the `StreamingResponse`.
5. The chunk's numpy array is released (GC'd) before the next chunk is computed.

Peak memory for a single DODS request is approximately:
```
max(chunk_data_bytes, encoding_chunk_size) + overhead
```

For the default 2MB `encoding_chunk_size`, this means ~4-6MB per concurrent
request regardless of total variable size.

### 5.3 Concurrent Request Limiting

The `data_load_semaphore` (default: 4 concurrent loads) prevents memory
explosion from many concurrent DODS requests each loading data simultaneously.
With 4 concurrent loads at 512MB max each, worst-case memory is ~2GB, which is
reasonable for a cloud container.

---

## 6. Testing Strategy

### 6.1 Reference Test Datasets

Define fixtures in `conftest.py` that cover the type space:

```python
@pytest.fixture
def basic_dataset() -> xr.Dataset:
    """Small dataset with float64 data, float32 coords, string attrs."""

@pytest.fixture
def all_types_dataset() -> xr.Dataset:
    """Dataset with every supported dtype: bool, int8..int64, uint8..uint64,
    float32, float64, string, datetime64, timedelta64."""

@pytest.fixture
def large_dask_dataset() -> xr.Dataset:
    """Dask-backed dataset with ~100MB variable for memory/streaming tests."""

@pytest.fixture
def scalar_and_empty_dataset() -> xr.Dataset:
    """Dataset with scalar variables, 0-length dimensions, NaN/Inf values."""

@pytest.fixture
def special_names_dataset() -> xr.Dataset:
    """Dataset with spaces, dots, unicode in variable/attribute names."""
```

### 6.2 XDR Encoding Gold Files

For each atomic type, store known-correct byte sequences as test fixtures:

```python
# test_dods.py
def test_xdr_int32_array():
    data = np.array([1, 2, 3], dtype=np.int32)
    result = b''.join(_xdr_encode_array(data, DAP_INT32))
    expected = (
        b'\x00\x00\x00\x03'  # length = 3 (first copy)
        b'\x00\x00\x00\x03'  # length = 3 (second copy)
        b'\x00\x00\x00\x01'  # value 1
        b'\x00\x00\x00\x02'  # value 2
        b'\x00\x00\x00\x03'  # value 3
    )
    assert result == expected

def test_xdr_byte_array_padding():
    data = np.array([1, 2, 3], dtype=np.uint8)
    result = b''.join(_xdr_encode_array(data, DAP_BYTE))
    expected = (
        b'\x00\x00\x00\x03'  # length (first copy)
        b'\x00\x00\x00\x03'  # length (second copy)
        b'\x01\x02\x03\x00'  # 3 bytes + 1 byte padding to 4-byte boundary
    )
    assert result == expected
```

### 6.3 Integration Test Architecture

Integration tests launch a real xpublish server (using `pytest-xprocess` or
an in-process ASGI server) and connect with real clients.

```python
# integration/conftest.py
@pytest.fixture(scope='session')
def opendap_server(basic_dataset):
    """Start an xpublish server with the OpenDapPlugin."""
    rest = xpublish.Rest(
        {'test': basic_dataset},
        plugins={'opendap': OpenDapPlugin()},
    )
    # Use httpx ASGITransport for in-process testing (no real TCP)
    transport = httpx.ASGITransport(app=rest.app)
    client = httpx.AsyncClient(transport=transport, base_url='http://test')
    yield client

# integration/test_netcdf4.py
def test_netcdf4_dap2_roundtrip(opendap_server_url, basic_dataset):
    """Verify netCDF4-python can open and read the dataset via DAP2."""
    with netCDF4.Dataset(f'{opendap_server_url}/datasets/test/opendap.dods') as nc:
        for var_name in basic_dataset.data_vars:
            np.testing.assert_array_equal(nc[var_name][:], basic_dataset[var_name].values)
```

### 6.4 Async Correctness Tests

```python
async def test_concurrent_requests_not_serialized(opendap_server):
    """Verify N concurrent requests complete faster than N × serial time."""
    url = '/datasets/test/opendap.dods?air[0:10][0:10][0:10]'

    # Measure single request time
    start = time.monotonic()
    await opendap_server.get(url)
    single_time = time.monotonic() - start

    # Measure 10 concurrent requests
    start = time.monotonic()
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(opendap_server.get(url)) for _ in range(10)]
    concurrent_time = time.monotonic() - start

    # Should be well under 10× (allow 3× for overhead)
    assert concurrent_time < single_time * 3
```

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

Phase 1 implementation proceeds bottom-up through the module dependency graph:

1. **`dap/types.py`** — type mapping, CF encoding. No dependencies on other new
   modules. Can be tested in isolation.

2. **`dap/constraint.py`** — parser. Depends only on its own dataclasses. Can
   be tested with pure string input/output.

3. **`dap/xarray_adapter.py`** — depends on `constraint.py` and `types.py`.
   Tests use xarray fixtures.

4. **`dap/dap2/dds.py`** — depends on `types.py`. Tests compare text output
   against gold strings.

5. **`dap/dap2/das.py`** — depends on `types.py`. Same testing pattern.

6. **`dap/dap2/dods.py`** — depends on `types.py`. Tests compare binary
   output against known byte sequences.

7. **`dap/dap2/responses.py`** — Error, Version, Help. Standalone.

8. **`errors.py`** — exception classes. Standalone.

9. **`io.py`** — async infrastructure. Tested with mock blocking functions.

10. **`plugin.py`** — integrates everything. Tested with httpx/TestClient against
    real xarray Datasets.

11. **Integration tests** — after plugin.py works, add tests with netCDF4,
    pydap, xarray engines.

Phase 2 adds `dap/dap4/` modules after Phase 1 is complete and passing
integration tests.

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| XDR encoding bugs break client compatibility | High | Gold-file tests against reference byte sequences; integration tests with 4 different clients |
| `DataArray.load_async()` not available in older xarray | Medium | Require xarray >= 2025.1.0 (already in requirements); fall back to sync `.load()` in executor if needed |
| DAP4 chunked format not correctly parsed by pydap/netCDF4 | High | Test against real clients early in Phase 2; use Hyrax responses as reference |
| Memory estimation inaccurate for complex dask graphs | Medium | Estimation is conservative (uses dtype.itemsize × product of shape); add escape hatch config to disable check |
| Constraint parser doesn't handle edge cases | Medium | Extensive parser test suite with edge cases from DAP2 spec; fuzz testing |
| StreamingResponse backpressure missing | Low | FastAPI/Starlette handles TCP backpressure; async generator naturally pauses when consumer is slow |
