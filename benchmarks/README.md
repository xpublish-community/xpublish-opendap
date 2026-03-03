# Xpublish OpenDAP Benchmarks

Performance benchmark suite for xpublish's OpenDAP (DAP2/DAP4) endpoints.

## Prerequisites

```bash
pip install -r dev-requirements.txt && pip install --no-deps -e .
cd ../xpublish-opendap && pip install --no-deps -e .
```

For the IFS dataset benchmark, also install `arraylake`:

```bash
pip install arraylake
```

## Quick Start

```bash
# Quick local test with tutorial dataset
python benchmarks/run.py --dataset air -n 5

# Full IFS benchmark (requires Arraylake access)
python benchmarks/run.py

# Filter to specific scenario types
python benchmarks/run.py --dataset air --scenarios dap4
python benchmarks/run.py --dataset air --scenarios metadata
```

Every run automatically saves JSON results to `benchmarks/results/` with a
timestamped filename (e.g. `20260302-130518_air.json`). Use `--no-save` to
skip this, or `-o path.json` to write to a specific path instead.

```bash
# Write to a specific path
python benchmarks/run.py --dataset air -o my_results.json

# Also export summary CSV
python benchmarks/run.py --dataset air --csv summary.csv

# Skip auto-saving JSON
python benchmarks/run.py --dataset air --no-save
```

## Comparing Runs

### Inline (single baseline)

```bash
python benchmarks/run.py --dataset air -o current.json -c baseline.json
```

### Multi-run comparison script

Use `benchmarks/compare.py` to compare two or more saved result files
side by side. The first file is treated as the baseline.

```bash
# Two-run comparison
python benchmarks/compare.py baseline.json feature.json

# Three-way comparison
python benchmarks/compare.py v1.json v2.json v3.json

# Compare all saved results (glob)
python benchmarks/compare.py benchmarks/results/*.json

# Filter scenarios and choose metric
python benchmarks/compare.py run1.json run2.json --scenarios metadata --metric p95_latency_s

# Sort by largest regression
python benchmarks/compare.py run1.json run2.json --sort delta

# Output as CSV or JSON for further analysis
python benchmarks/compare.py run1.json run2.json --format csv > comparison.csv
python benchmarks/compare.py run1.json run2.json --format json > comparison.json
```

## Version Comparison Workflow

```bash
# 1. On the baseline branch (e.g., main or latest release)
git checkout main
pip install --no-deps -e .
cd ../xpublish-opendap && pip install --no-deps -e . && cd ../xpublish
python benchmarks/run.py --dataset air -n 20 -o baseline.json

# 2. On the feature branch
git checkout feature/dap4
pip install --no-deps -e .
cd ../xpublish-opendap && pip install --no-deps -e . && cd ../xpublish
python benchmarks/run.py --dataset air -n 20 -o feature.json

# 3. Compare
python benchmarks/compare.py baseline.json feature.json
```

## CLI Reference

### `run.py`

| Option | Default | Description |
|---|---|---|
| `-n`, `--iterations` | 10 | Timed iterations per scenario |
| `--warmup` | 2 | Warmup iterations (not measured) |
| `--port` | 9876 | Server port |
| `--host` | 127.0.0.1 | Server host |
| `-o`, `--output` | auto | JSON output path (default: `benchmarks/results/<timestamp>.json`) |
| `--csv` | — | Also write summary CSV to this path |
| `--no-save` | — | Skip auto-saving JSON to `benchmarks/results/` |
| `-c`, `--compare` | — | Compare against a baseline JSON file |
| `--dataset` | ifs | Dataset: `ifs` or `air` |
| `--scenarios` | — | Filter by scenario name prefix |
| `--timeout` | 60 | Request timeout (seconds) |
| `-v`, `--verbose` | — | Print per-request errors |

### `compare.py`

| Option | Default | Description |
|---|---|---|
| `files` (positional) | — | 2+ JSON result files (first = baseline) |
| `--metric` | mean_latency_s | Stat to compare (see list below) |
| `--format` | table | Output format: `table`, `csv`, `json` |
| `--scenarios` | — | Filter by scenario name prefix |
| `--sort` | name | Sort order: `name`, `delta`, `value` |

Available metrics: `mean_latency_s`, `median_latency_s`, `p95_latency_s`,
`min_latency_s`, `max_latency_s`, `throughput_mbps`, `success_rate`,
`total_requests`, `total_bytes`.

## Scenarios

The suite generates ~21 scenarios organized into four categories:

### Metadata
DDS, DAS, DMR, DSR, and version endpoints — measures metadata generation overhead without data I/O.

### Data (DAP2)
DODS requests with varying data volumes: scalar points, single/multi-timestep slices, zonal transects, multi-variable projections, and strided subsamples.

### Data (DAP4)
DAP4 binary data responses (`.dap`) with equivalent access patterns to the DAP2 scenarios.

### Constraint Parsing
DDS requests with increasingly complex constraint expressions — isolates the parsing/subsetting overhead from data serialization.

## Output Formats

### Console Table

Results are printed grouped by category with per-scenario statistics:
- **Mean/Median/P95/Min/Max** — latency in milliseconds
- **MB/s** — throughput (response bytes / total latency)
- **OK%** — success rate
- **N** — number of timed requests

### JSON

Full run metadata (git SHA, branch, xpublish version, dataset dimensions,
timestamp) plus per-request latency/status/bytes data. Used as input for
`compare.py` and for custom analysis scripts.

### CSV

Flat summary table with one row per scenario and columns for each stat.
Useful for import into spreadsheets or pandas.
