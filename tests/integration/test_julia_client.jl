#!/usr/bin/env julia
# Integration tests for xpublish-opendap using Julia's NCDatasets.jl.
#
# Usage:
#   julia test_julia_client.jl <opendap_url> --test <test_name>
#
# Test names: open, dimensions, variables, variable_shape, read_lat,
#             read_data_slice, attributes

using NCDatasets

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

if length(ARGS) < 3 || ARGS[2] != "--test"
    println("Usage: julia test_julia_client.jl <url> --test <name>")
    exit(1)
end

url = ARGS[1]
test_name = ARGS[3]

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

function test_open()
    ds = NCDataset(url)
    close(ds)
    println("PASS: opened and closed dataset")
end

function test_dimensions()
    ds = NCDataset(url)
    @assert ds.dim["time"] == 2920 "time: expected 2920 got $(ds.dim["time"])"
    @assert ds.dim["lat"] == 25 "lat: expected 25 got $(ds.dim["lat"])"
    @assert ds.dim["lon"] == 53 "lon: expected 53 got $(ds.dim["lon"])"
    close(ds)
    println("PASS: dimensions match")
end

function test_variables()
    ds = NCDataset(url)
    @assert haskey(ds, "air") "'air' not in dataset"
    @assert haskey(ds, "lat") "'lat' not in dataset"
    @assert haskey(ds, "lon") "'lon' not in dataset"
    @assert haskey(ds, "time") "'time' not in dataset"
    close(ds)
    println("PASS: variables present")
end

function test_variable_shape()
    ds = NCDataset(url)
    # Column-major order: (lon, lat, time)
    s = size(ds["air"])
    @assert s == (53, 25, 2920) "air shape: expected (53, 25, 2920) got $s"
    close(ds)
    println("PASS: variable shape matches")
end

function test_read_lat()
    ds = NCDataset(url)
    lat = ds["lat"][:]
    @assert length(lat) == 25 "lat length: expected 25 got $(length(lat))"
    @assert isapprox(lat[1], 75.0; atol=1e-4) "lat[1]: expected 75.0 got $(lat[1])"
    @assert isapprox(lat[25], 15.0; atol=1e-4) "lat[25]: expected 15.0 got $(lat[25])"
    close(ds)
    println("PASS: latitude values match")
end

function test_read_data_slice()
    ds = NCDataset(url)
    # 1-based indexing: [1,1,1] corresponds to Python's [0,0,0]
    val = Float64(ds["air"][1, 1, 1])
    @assert isapprox(val, 241.2; atol=0.1) "air[1,1,1]: expected ~241.2 got $val"
    close(ds)
    println("PASS: data slice matches")
end

function test_attributes()
    ds = NCDataset(url)
    long_name = ds["air"].attrib["long_name"]
    expected = "4xDaily Air temperature at sigma level 995"
    @assert long_name == expected "air long_name: expected '$expected' got '$long_name'"
    close(ds)
    println("PASS: attributes match")
end

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

tests = Dict(
    "open" => test_open,
    "dimensions" => test_dimensions,
    "variables" => test_variables,
    "variable_shape" => test_variable_shape,
    "read_lat" => test_read_lat,
    "read_data_slice" => test_read_data_slice,
    "attributes" => test_attributes,
)

if !haskey(tests, test_name)
    println("FAIL: unknown test: $test_name. Available: $(join(keys(tests), ", "))")
    exit(1)
end

try
    tests[test_name]()
catch e
    println("FAIL: $e")
    exit(1)
end
