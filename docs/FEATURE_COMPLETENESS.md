# xpublish-opendap Feature Completeness

This document compares the xpublish-opendap implementation against the OPeNDAP
DAP2 and DAP4 specifications, using the Hyrax/libdap4 reference implementation
as baseline.

Legend: implemented, partial, not implemented, n/a

---

## 1. HTTP Routes / Response Types

### DAP2 Responses

| Response | Suffix | Spec | xpublish-opendap | Notes |
|----------|--------|------|-------------------|-------|
| DDS (Dataset Descriptor Structure) | `.dds` | Required | Implemented | Structure, types, dimensions |
| DAS (Dataset Attribute Structure) | `.das` | Required | Implemented | Variable + NC_GLOBAL attrs |
| DODS (DataDDS binary) | `.dods` | Required | Implemented | DDS + `\nData:\n` + XDR binary |
| DDX (XML combined DDS+DAS) | `.ddx` | Optional | Not implemented | Pre-DAP4 XML extension |
| ASCII data | `.asc`/`.ascii` | Optional | Not implemented | Human-readable tabular output |
| HTML request form | `.html` | Optional | Not implemented | Interactive subsetting form |
| Info (combined DAS+DDS) | `.info` | Optional | Not implemented | Human-readable HTML summary |
| Version | `.ver` | Optional | Implemented | Returns server version string |
| Help | `.help` | Optional | Implemented | HTML listing all endpoints |

### DAP4 Responses

| Response | Suffix | Spec | xpublish-opendap | Notes |
|----------|--------|------|-------------------|-------|
| DMR (Dataset Metadata Response) | `.dmr` | Required | Implemented | Full XML with dims, vars, attrs, maps |
| Data Response (chunked binary) | `.dap` | Required | Implemented | DMR + CRLF + chunked binary + CRC32 |
| DSR (Dataset Services Response) | `.dsr` | Required | Implemented | Lists DMR, DAP, and DAP2 services |
| Error Response | in-band | Required | Implemented | XML Error with ErrorCode + Message |

### HTTP Headers

| Header | DAP2 | DAP4 | xpublish-opendap |
|--------|------|------|-------------------|
| `XDODS-Server` | Required | -- | Implemented |
| `XOPeNDAP-Server` | Required | Required | Implemented |
| `XDAP` | `2.0` | `4.0` | Partial (DAP4 only: `XDAP: 4.0`; not set on DAP2 responses) |
| `Content-Description` | Required | -- | Implemented (`dods-dds`, `dods-das`, `dods-data`, `dods-error`) |
| `Content-Type` | Required | Required | Implemented (see table below) |
| `Content-Encoding: deflate` | Optional | Optional | Not implemented |
| `Last-Modified` | Recommended | Recommended | Not implemented |

### Content Types

| Response | Spec Content-Type | xpublish-opendap |
|----------|-------------------|-------------------|
| DDS | `text/plain` | `text/plain` |
| DAS | `text/plain` | `text/plain` |
| DODS | `application/octet-stream` | `application/octet-stream` |
| DMR | `application/vnd.opendap.dap4.dataset-metadata+xml` | Implemented |
| DAP4 Data | `application/vnd.opendap.dap4.data` | Implemented |
| DSR | `application/vnd.opendap.dap4.dataset-services+xml` | Implemented |
| DAP4 Error | `application/vnd.opendap.dap4.error+xml` | Implemented |

---

## 2. Data Types

### Atomic Types

| Type | numpy dtype | DAP2 Name | DAP4 Name | XDR Wire | DAP4 Wire | xpublish-opendap |
|------|-------------|-----------|-----------|----------|-----------|-------------------|
| Unsigned 8-bit | `uint8` | `Byte` | `UInt8` | 4 bytes (padded) | 1 byte | Implemented |
| Signed 8-bit | `int8` | `Int16` (promoted) | `Int8` | 4 bytes | 1 byte | Implemented (DAP2 lossy) |
| Signed 16-bit | `int16` | `Int16` | `Int16` | 4 bytes (widened) | 2 bytes | Implemented |
| Unsigned 16-bit | `uint16` | `UInt16` | `UInt16` | 4 bytes (widened) | 2 bytes | Implemented |
| Signed 32-bit | `int32` | `Int32` | `Int32` | 4 bytes | 4 bytes | Implemented |
| Unsigned 32-bit | `uint32` | `UInt32` | `UInt32` | 4 bytes | 4 bytes | Implemented |
| Signed 64-bit | `int64` | `Float64` (lossy) | `Int64` | 8 bytes | 8 bytes | Implemented (DAP2 lossy) |
| Unsigned 64-bit | `uint64` | `Float64` (lossy) | `UInt64` | 8 bytes | 8 bytes | Implemented (DAP2 lossy) |
| 32-bit float | `float32` | `Float32` | `Float32` | 4 bytes | 4 bytes | Implemented |
| 64-bit float | `float64` | `Float64` | `Float64` | 8 bytes | 8 bytes | Implemented |
| Boolean | `bool` | `Byte` | `UInt8` | 4 bytes | 1 byte | Implemented |
| String | `object`/`U`/`S` | `String` | `String` | len-prefixed + pad | len-prefixed | Implemented |
| URL | -- | `Url` | `URI` | same as String | same as String | Not implemented |
| Opaque | -- | -- | `Opaque` | -- | len-prefixed | Not implemented |
| Char | -- | -- | `Char` | -- | 1 byte | Not implemented |
| Enum | -- | -- | `Enum` | -- | basetype-sized | Not implemented |

### Temporal Types (via CF Encoding)

| Type | Encoding | xpublish-opendap |
|------|----------|-------------------|
| `datetime64` | CF-encoded to `int64` (numeric) | Implemented |
| `timedelta64` | CF-encoded to `int64` (numeric) | Implemented |
| NaT sentinel | Maps to int64 fill value | Implemented |

### Type Limitations

- **DAP2 int64/uint64**: Mapped to `Float64`, which loses precision for integers > 2^53. This matches the DAP2 spec (no 64-bit integer types exist).
- **DAP2 int8**: Promoted to `Int16` since DAP2 has no signed byte type.
- **Complex types** (`complex64`, `complex128`): Not supported by either DAP protocol. Memory estimation falls back to `dtype.itemsize`.
- **`scale_factor`/`add_offset`**: NOT applied. The plugin serves decoded xarray values, so CF scale/offset encoding is intentionally skipped.

---

## 3. Container / Compound Types

### DAP2 Containers

| Type | Spec | xpublish-opendap | Notes |
|------|------|-------------------|-------|
| **Array** | Multi-dimensional indexed array | Implemented | All xarray data/coord variables with dims > 0 |
| **Grid** | Array + coordinate map vectors | Implemented | Multi-dim data vars emitted as Grid in DDS; map members accessible via dotted paths (`air.time`) |
| **Structure** | Named collection of typed fields | Not implemented | xarray has no direct Structure equivalent |
| **Sequence** | Variable-length table rows | Not implemented | xarray has no Sequence equivalent |
| Scalar | 0-dimensional variable | Implemented | Emitted as bare type declaration in DDS |

### DAP4 Containers

| Type | Spec | xpublish-opendap | Notes |
|------|------|-------------------|-------|
| Typed variable + `<Dim>` refs | Array with shared dimensions | Implemented | All variables use `<Dim name="/dim"/>` refs |
| `<Map>` elements | Coordinate relationships (replaces Grid) | Implemented | Data vars reference coord maps |
| **Structure** | Composite fields | Not implemented | |
| **Sequence** | Variable-length records | Not implemented | |
| **Group** | Hierarchical namespace | Not implemented | xarray has no group hierarchy |
| **Enumeration** | Named integer constants | Not implemented | |

---

## 4. Constraint Expressions

### Projection (Variable Selection)

| Feature | DAP2 Syntax | DAP4 Syntax | xpublish-opendap |
|---------|-------------|-------------|-------------------|
| Select all variables | empty CE | empty CE | Implemented |
| Select single variable | `temp` | `/temp` or `dap4.ce=/temp` | Implemented |
| Select multiple variables | `temp,lat,lon` | `/temp;/lat;/lon` | Implemented |
| URL percent-encoding | `temp%5B0%3A10%5D` | `%2Ftemp%5B0%3A10%5D` | Implemented |
| Wildcard `*` | Select all | -- | Not implemented |
| Structure field projection | `station.temp` | `station{temp}` | Not implemented |
| Group path projection | -- | `/group/var` | Not implemented |

### Hyperslab (Array Subsetting)

| Feature | DAP2 Syntax | DAP4 Syntax | xpublish-opendap |
|---------|-------------|-------------|-------------------|
| Single index | `var[5]` | `var[5]` | Implemented |
| Range (inclusive) | `var[0:10]` | `var[0:10]` | Implemented |
| Range with stride | `var[0:2:10]` | `var[0:2:10]` | Implemented |
| Multi-dimensional | `var[0:10][0:5]` | `var[0:10][0:5]` | Implemented |
| Open-ended range | -- | `var[5:]` | Not implemented |
| Open-ended with stride | -- | `var[0:2:]` | Not implemented |
| Empty brackets (all) | -- | `var[]` | Not implemented |
| Shared dimension slice | -- | `nlat=[0:9];temp` | Not implemented |

### Selection (Row Filtering)

| Feature | DAP2 Syntax | xpublish-opendap |
|---------|-------------|-------------------|
| Relational filter | `&temp>20.0` | Not implemented (raises ConstraintNotSupportedError) |
| Equality | `&name="foo"` | Not implemented |
| Regex match | `&name~"pat.*"` | Not implemented |
| Set membership | `&month={4,5,6}` | Not implemented |
| Multiple conditions (AND) | `&temp>20&lat>-30` | Not implemented |

### DAP4 Filter Expressions

| Feature | DAP4 Syntax | xpublish-opendap |
|---------|-------------|-------------------|
| Sequence filter | `seq\|field>5` | Not implemented (raises ConstraintNotSupportedError) |
| Range filter | `seq\|5<x<100` | Not implemented |
| Regex filter | `seq\|name~="pat"` | Not implemented |
| AND predicates | `seq\|x>5,y<10` | Not implemented |

### Server-Side Functions

| Feature | Syntax | xpublish-opendap |
|---------|--------|-------------------|
| Projection functions | `?func(arg1,arg2)` | Not implemented (raises ConstraintNotSupportedError) |
| `grid()` | `grid(SST,10,20,-10,40)` | Not implemented |
| `geogrid()` | `geogrid(SST,N,W,S,E)` | Not implemented |
| `linear_scale()` | `linear_scale(var,m,b)` | Not implemented |
| `version()` | `version()` | Not implemented |

### Grid Path Resolution

| Path | Meaning | xpublish-opendap |
|------|---------|-------------------|
| `air` | Direct variable | Implemented |
| `air.air` | Grid array member | Implemented |
| `air.time` | Grid map (coordinate) member | Implemented |
| `air.nonexistent` | Invalid member | Implemented (raises VariableNotFoundError) |
| `scalar.scalar` | Non-Grid dotted path | Implemented (raises VariableNotFoundError) |

---

## 5. Binary Encoding

### DAP2 XDR Encoding

| Feature | Spec | xpublish-opendap |
|---------|------|-------------------|
| Big-endian (network) byte order | Required | Implemented |
| Doubled length prefix for atomic arrays | Required | Implemented (`length + length + data`) |
| Single length prefix for string arrays | Required | Implemented |
| Byte arrays: packed + 4-byte pad | Required | Implemented |
| Int16/UInt16 widened to 4 bytes | Required | Implemented |
| String: 4-byte len + UTF-8 + pad | Required | Implemented |
| Structure serialization (sequential fields) | Required | N/A (no Structure support) |
| Grid serialization (array then maps) | Required | Implemented (via DDS variable ordering) |
| Sequence markers (SOI=0x5A, EOS=0xA5) | Required | N/A (no Sequence support) |
| `\nData:\n` separator | Required | Implemented |

### DAP4 Chunked Encoding

| Feature | Spec | xpublish-opendap |
|---------|------|-------------------|
| DMR + CRLF separator | Required | Implemented (`\r\n`) |
| 4-byte chunk headers (big-endian) | Required | Implemented |
| CHUNK_DATA (0x00) | Required | Implemented |
| CHUNK_END (0x01) | Required | Implemented |
| CHUNK_ERR (0x02) | Required | Not implemented |
| Little-endian flag (0x04) | Required | Implemented (set from `sys.byteorder`) |
| Max chunk size 2^24 (~16 MB) | Required | Implemented |
| Native byte order data | Required | Implemented |
| uint64 length prefix for arrays | Required | Implemented |
| No padding between elements | Required | Implemented |
| Int16/UInt16 at natural 2-byte size | Required | Implemented |
| Byte/Int8 at natural 1-byte size | Required | Implemented |
| String: uint64 len + UTF-8, no pad | Required | Implemented |
| CRC32 per top-level variable | Required | Implemented |
| Opaque: int64 len + raw bytes | Required | N/A (no Opaque type) |
| Structure: sequential fields, no pad | Required | N/A (no Structure support) |
| Sequence: int64 count + rows | Required | N/A (no Sequence support) |

---

## 6. Error Handling

### Error Types

| Error | HTTP Code | DAP Error Code | xpublish-opendap |
|-------|-----------|---------------|-------------------|
| Constraint syntax error | 400 | 1000 | Implemented (`ConstraintSyntaxError`) |
| Unsupported feature | 400 | 1001 | Implemented (`ConstraintNotSupportedError`) |
| Variable not found | 400 | 1002 | Implemented (`VariableNotFoundError`) |
| Index out of range | 400 | 1003 | Implemented (`IndexOutOfRangeError`) |
| Request too large | 413 | 1004 | Implemented (`RequestTooLargeError`) |
| Internal server error | 500 | -- | Implemented (catch-all) |
| No authorization | 401 | 1006 | Not implemented |
| Cannot read file | 400 | 1007 | Not implemented |
| Not implemented | 501 | 1008 | Not implemented |

### Error Response Formats

| Protocol | Format | xpublish-opendap |
|----------|--------|-------------------|
| DAP2 | `Error { code = N; message = "..."; };` | Implemented |
| DAP4 | XML `<Error httpcode="N"><ErrorCode>N</ErrorCode><Message>...</Message></Error>` | Implemented |
| DAP4 `<Context>` | Additional error context | Not implemented |
| DAP4 `<OtherInformation>` | Stack trace / debug info | Not implemented |
| DAP4 in-band error chunk | CHUNK_ERR during streaming | Not implemented |

---

## 7. Plugin Configuration

| Option | Default | Description |
|--------|---------|-------------|
| `max_request_memory_bytes` | 512 MB | Rejects requests estimated to exceed this threshold |
| `async_load_timeout` | 30.0 s | Timeout for loading dataset into memory |
| `num_concurrent_data_loads` | 4 | Concurrent dataset load limit (semaphore) |
| `dataset_router_prefix` | `/opendap` | URL prefix for all endpoints |
| `dataset_router_tags` | `["opendap"]` | OpenAPI tags |

---

## 8. Feature Completeness Summary

### Fully Implemented

- DAP2: DDS, DAS, DODS responses with correct XDR encoding
- DAP4: DMR, Data, DSR responses with chunked binary + CRC32
- All numeric types (8/16/32/64-bit signed/unsigned, float32/64)
- Boolean and string types
- Datetime/timedelta via CF encoding
- Grid container type (DAP2) and Map elements (DAP4)
- Constraint expression projection with multi-dimensional hyperslabs
- Grid-qualified path resolution (`grid.member` syntax)
- Memory estimation and request size limiting
- Async streaming data responses
- Proper DAP2/DAP4 error response formatting
- HTTP headers and content types per specification

### Partially Implemented

- **DAP2 int8**: Promoted to Int16 (correct per spec, but lossy)
- **DAP2 int64/uint64**: Mapped to Float64 (correct per spec, but lossy for large values)
- **DAP4 error elements**: `ErrorCode` and `Message` present; `Context` and `OtherInformation` omitted
- **`XDAP` header**: Set on DAP4 responses (`4.0`) but not on DAP2 responses
- **Response compression**: Not implemented (no `Content-Encoding: deflate`)
- **`Last-Modified` header**: Not implemented (requires dataset timestamp metadata)

### Not Implemented

- **Container types**: Structure, Sequence (DAP2 + DAP4), Group (DAP4), Enum (DAP4)
- **Constraint selections**: Row filtering (`&temp>20`), regex matching, set membership
- **DAP4 filters**: Sequence filter clauses (`seq|field>5`)
- **Server-side functions**: `grid()`, `geogrid()`, `linear_scale()`, custom functions
- **DAP4 advanced CE**: Open-ended ranges (`[5:]`), shared dimension slices, field selection (`{x;y}`)
- **Optional responses**: DDX, ASCII, HTML form, Info
- **Data types**: URL/URI, Opaque, Char, Enum
- **In-band error chunks**: DAP4 CHUNK_ERR for mid-stream errors
- **Async responses**: HTTP 202 accept + polling pattern

### Not Applicable

These spec features have no xarray equivalent and are unlikely to be needed:

- **Sequence type**: xarray datasets are array-oriented, not record-oriented
- **Nested structures**: xarray variables are flat (no nested fields)
- **Groups**: xarray datasets are single-level (xarray-datatree would be needed)
- **Enum types**: xarray has no native enum support
- **Opaque blobs**: xarray has no opaque binary type

---

## 9. Client Compatibility

The implementation has been tested against:

| Client | Protocol | Status |
|--------|----------|--------|
| Python `netCDF4` (via DODS) | DAP2 | Integration tests pass |
| NCO `ncks` | DAP2 | Integration tests pass |
| `ncdump` | DAP2 | Integration tests pass |
| Julia `NCDatasets.jl` | DAP2 | Integration tests exist |
| R `ncdf4` package | DAP2 | Integration tests exist |
| xarray `open_dataset(engine="pydap")` | DAP2 | Integration tests pass |
| xarray `open_dataset(engine="netcdf4")` | DAP2 | Integration tests pass |
| Raw HTTP (httpx) | DAP2 + DAP4 | Integration tests pass |
| Generic DAP4 XML parsing | DAP4 | Unit tests pass |
| DAP4 binary decoding | DAP4 | Unit tests pass |

---

## 10. Specification References

- **DAP2**: [ESE-RFC-004](https://www.earthdata.nasa.gov/s3fs-public/imported/ESE-RFC-004v1.1.pdf), [OPeNDAP User Guide](https://opendap.github.io/documentation/UserGuideComprehensive.html)
- **DAP4**: [DAP4 Specification](https://opendap.github.io/dap4-specification/DAP4.html), [Data Encoding](https://docs.opendap.org/index.php/DAP4:_Encoding_for_the_Data_Response), [Constraint Expressions v2](https://docs.opendap.org/index.php?title=DAP4:_Constraint_Expressions,_v2)
- **Reference implementation**: [libdap4](https://github.com/OPENDAP/libdap4), [BES](https://github.com/OPENDAP/bes), [Hyrax](https://github.com/OPENDAP/hyrax)
